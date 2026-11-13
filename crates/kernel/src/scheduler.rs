//! Multi-core scheduler: one run queue per emulated core, no global lock.
//!
//! A runnable thread sits in the queue of its active core. Threads change state under their own
//! lock and then take the queue lock (thread -> queue). A core that pulls a thread out of a queue
//! holds the queue lock and only `try_lock`s the thread, backing off on contention, so the two
//! orders never deadlock. An idle core steals the best thread whose affinity allows it from a
//! busy core's queue, which is how threads migrate. Wakeups, address arbitration and IPC touch
//! only the woken thread, one queue and at most one other core's interrupt line, so cores do not
//! serialise on each other (NX2_NOTES row 109).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use lichii_cpu::InterruptHandle;
use parking_lot::{Condvar, Mutex};

use crate::thread::{Resume, Sched, Thread, ThreadId, ThreadState, WakeResult};

const LEVELS: usize = 64;
const IDLE_PRIORITY: u32 = LEVELS as u32;
const PARK_TIMEOUT: Duration = Duration::from_millis(10);
/// How often a pause looks again for cores still in guest code, should a wakeup be missed.
const GUEST_EXIT_POLL: Duration = Duration::from_millis(1);
const NO_SVC: u32 = u32::MAX;

struct RunQueue {
    levels: [VecDeque<Arc<Thread>>; LEVELS],
    mask: u64,
}

impl RunQueue {
    fn new() -> Self {
        Self {
            levels: std::array::from_fn(|_| VecDeque::new()),
            mask: 0,
        }
    }

    fn push(&mut self, thread: Arc<Thread>, priority: u8, front: bool) {
        let level = &mut self.levels[priority as usize];
        if front {
            level.push_front(thread);
        } else {
            level.push_back(thread);
        }
        self.mask |= 1 << priority;
    }

    fn remove(&mut self, thread: &Arc<Thread>, priority: u8) -> bool {
        let level = &mut self.levels[priority as usize];
        let Some(pos) = level.iter().position(|t| Arc::ptr_eq(t, thread)) else {
            return false;
        };
        level.remove(pos);
        if level.is_empty() {
            self.mask &= !(1 << priority);
        }
        true
    }

    fn take(&mut self, priority: usize, pos: usize) -> Arc<Thread> {
        let level = &mut self.levels[priority];
        let thread = level.remove(pos).expect("queued thread");
        if level.is_empty() {
            self.mask &= !(1 << priority);
        }
        thread
    }

    /// Highest (numerically lowest) queued priority.
    fn top(&self) -> u32 {
        if self.mask == 0 {
            IDLE_PRIORITY
        } else {
            self.mask.trailing_zeros()
        }
    }
}

struct Core {
    queue: Mutex<RunQueue>,
    notified: Mutex<bool>,
    wakeup: Condvar,
    interrupt: Mutex<Option<InterruptHandle>>,
    running_priority: AtomicU32,
    resched: AtomicBool,
    rotate: AtomicBool,
    /// The core is running guest code (inside `CpuCore::run`).
    in_guest: AtomicBool,
    /// The SVC the core is handling, or [`NO_SVC`].
    svc: AtomicU32,
}

/// A thread taken from a run queue to run on a core.
pub(crate) struct Picked {
    pub thread: Arc<Thread>,
    pub resume: Resume,
    pub wake: WakeResult,
    pub terminate: bool,
}

/// Why a running thread should leave its core.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preempt {
    Stay,
    Switch,
    Pause,
    Terminate,
}

pub struct Scheduler {
    cores: Box<[Core]>,
    idle: AtomicU64,
    shutdown: AtomicBool,
    paused: AtomicBool,
    /// Signalled when a core leaves guest code while paused.
    left_guest: Condvar,
    left_guest_lock: Mutex<()>,
}

impl Scheduler {
    pub(crate) fn new(core_count: usize) -> Self {
        Self {
            cores: (0..core_count)
                .map(|_| Core {
                    queue: Mutex::new(RunQueue::new()),
                    notified: Mutex::new(false),
                    wakeup: Condvar::new(),
                    interrupt: Mutex::new(None),
                    running_priority: AtomicU32::new(IDLE_PRIORITY),
                    resched: AtomicBool::new(false),
                    rotate: AtomicBool::new(false),
                    in_guest: AtomicBool::new(false),
                    svc: AtomicU32::new(NO_SVC),
                })
                .collect(),
            idle: AtomicU64::new(0),
            shutdown: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            left_guest: Condvar::new(),
            left_guest_lock: Mutex::new(()),
        }
    }

    pub fn core_count(&self) -> usize {
        self.cores.len()
    }

    pub(crate) fn all_cores_mask(&self) -> u64 {
        (1u64 << self.cores.len()) - 1
    }

    /// Horizon rotates threads of this priority on a core every 10 ms.
    pub(crate) fn preemption_priority(&self, core: usize) -> u32 {
        if core + 1 == self.cores.len() { 63 } else { 59 }
    }

    /// Installs the interrupt line of the CPU about to run on `core`. A reschedule requested
    /// before the line existed (or while it pointed at another process's CPU) is forwarded, so
    /// it cannot be lost: requesters set `resched` before taking this lock.
    pub(crate) fn set_interrupt(&self, core: usize, handle: InterruptHandle) {
        let c = &self.cores[core];
        let mut slot = c.interrupt.lock();
        if c.resched.load(Ordering::Acquire)
            || self.shutdown.load(Ordering::Acquire)
            || self.paused.load(Ordering::Acquire)
        {
            handle.signal();
        }
        *slot = Some(handle);
    }

    fn interrupt(&self, core: usize) {
        if let Some(handle) = self.cores[core].interrupt.lock().as_ref() {
            handle.signal();
        }
    }

    fn unpark(&self, core: usize) {
        let c = &self.cores[core];
        *c.notified.lock() = true;
        c.wakeup.notify_one();
    }

    /// Asks `core` to re-evaluate its running thread at the next halt.
    pub(crate) fn request_reschedule(&self, core: usize) {
        self.cores[core].resched.store(true, Ordering::Release);
        self.interrupt(core);
        if self.idle.load(Ordering::SeqCst) & (1 << core) != 0 {
            self.unpark(core);
        }
    }

    pub(crate) fn take_reschedule(&self, core: usize) -> bool {
        self.cores[core].resched.load(Ordering::Relaxed)
            && self.cores[core].resched.swap(false, Ordering::AcqRel)
    }

    pub(crate) fn set_running_priority(&self, core: usize, priority: u8) {
        self.cores[core]
            .running_priority
            .store(priority as u32, Ordering::Release);
    }

    pub(crate) fn clear_running(&self, core: usize) {
        self.cores[core]
            .running_priority
            .store(IDLE_PRIORITY, Ordering::Release);
    }

    /// Makes a thread runnable on its active core. Called with the thread's lock held.
    pub(crate) fn enqueue(&self, thread: &Arc<Thread>, s: &mut Sched, front: bool) {
        if s.affinity & (1 << s.active_core) == 0 {
            s.active_core = s.affinity.trailing_zeros() as usize;
        }
        s.state = ThreadState::Runnable;
        let core = s.active_core;
        self.cores[core]
            .queue
            .lock()
            .push(thread.clone(), s.priority, front);
        let idle = self.idle.load(Ordering::SeqCst);
        if idle & (1 << core) != 0 {
            self.unpark(core);
            return;
        }
        if (s.priority as u32) < self.cores[core].running_priority.load(Ordering::Acquire) {
            self.cores[core].resched.store(true, Ordering::Release);
            self.interrupt(core);
            return;
        }
        let helpers = idle & s.affinity & !(1 << core);
        if helpers != 0 {
            self.unpark(helpers.trailing_zeros() as usize);
        }
    }

    /// Takes a runnable thread out of its queue. Called with the thread's lock held.
    pub(crate) fn dequeue(&self, thread: &Arc<Thread>, s: &Sched) {
        if s.state == ThreadState::Runnable {
            self.cores[s.active_core]
                .queue
                .lock()
                .remove(thread, s.priority);
        }
    }

    /// Picks the next thread for `core`, parking until one is available. `None` on shutdown.
    pub(crate) fn next(&self, core: usize) -> Option<Picked> {
        loop {
            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }
            let paused = self.paused.load(Ordering::Acquire);
            if !paused && let Some(picked) = self.pick(core) {
                return Some(picked);
            }
            self.idle.fetch_or(1 << core, Ordering::SeqCst);
            if !self.paused.load(Ordering::Acquire)
                && let Some(picked) = self.pick(core)
            {
                self.idle.fetch_and(!(1 << core), Ordering::SeqCst);
                return Some(picked);
            }
            let c = &self.cores[core];
            let mut notified = c.notified.lock();
            if !*notified && !self.shutdown.load(Ordering::Acquire) {
                c.wakeup.wait_for(&mut notified, PARK_TIMEOUT);
            }
            *notified = false;
            drop(notified);
            self.idle.fetch_and(!(1 << core), Ordering::SeqCst);
        }
    }

    fn pick(&self, core: usize) -> Option<Picked> {
        loop {
            match self.take_from(core, core, true) {
                Take::Found(p) => return Some(p),
                Take::Contended => {
                    std::hint::spin_loop();
                    continue;
                }
                Take::Empty => {}
            }
            let n = self.cores.len();
            let mut contended = false;
            for i in 1..n {
                match self.take_from((core + i) % n, core, false) {
                    Take::Found(p) => return Some(p),
                    Take::Contended => contended = true,
                    Take::Empty => {}
                }
            }
            if !contended {
                return None;
            }
            std::thread::yield_now();
        }
    }

    fn take_from(&self, source: usize, core: usize, own: bool) -> Take {
        let mut queue = self.cores[source].queue.lock();
        let mut mask = queue.mask;
        while mask != 0 {
            let priority = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            for pos in 0..queue.levels[priority].len() {
                let thread = &queue.levels[priority][pos];
                let Some(mut s) = thread.sched.try_lock() else {
                    return Take::Contended;
                };
                if !own && s.affinity & (1 << core) == 0 {
                    continue;
                }
                s.state = ThreadState::Running(core);
                s.active_core = core;
                s.runs += 1;
                let picked = Picked {
                    thread: thread.clone(),
                    resume: std::mem::take(&mut s.resume),
                    wake: s.wake,
                    terminate: s.terminate_requested,
                };
                self.cores[core]
                    .running_priority
                    .store(s.priority as u32, Ordering::Release);
                drop(s);
                queue.take(priority, pos);
                return Take::Found(picked);
            }
            if own {
                break;
            }
        }
        Take::Empty
    }

    /// Decides whether the thread running on `core` keeps it. On `Switch` the thread is back in a
    /// run queue (or suspended) and the core must save its context first: the caller passes
    /// `save`, which runs before the thread becomes visible to other cores.
    pub(crate) fn preempt(
        &self,
        core: usize,
        thread: &Arc<Thread>,
        yielding: bool,
        save: impl FnOnce(),
    ) -> Preempt {
        let mut s = thread.sched.lock();
        if s.terminate_requested {
            return Preempt::Terminate;
        }
        if self.shutdown.load(Ordering::Acquire) {
            save();
            self.clear_running(core);
            return Preempt::Switch;
        }
        if s.pause_requested {
            save();
            s.state = ThreadState::Suspended;
            self.clear_running(core);
            return Preempt::Pause;
        }
        if self.paused.load(Ordering::Acquire) {
            save();
            self.enqueue(thread, &mut s, true);
            self.clear_running(core);
            return Preempt::Switch;
        }
        let c = &self.cores[core];
        let rotate = c.rotate.swap(false, Ordering::AcqRel);
        let top = c.queue.lock().top();
        let priority = s.priority as u32;
        let wrong_core = s.affinity & (1 << core) == 0;
        let to_back = (yielding || rotate) && top <= priority;
        if !wrong_core && top >= priority && !to_back {
            return Preempt::Stay;
        }
        save();
        if wrong_core {
            s.active_core = s.affinity.trailing_zeros() as usize;
        }
        self.enqueue(thread, &mut s, !to_back && !wrong_core);
        self.clear_running(core);
        Preempt::Switch
    }

    /// Rotates same-priority threads at each core's preemption priority (the 10 ms tick).
    pub(crate) fn preemption_tick(&self) {
        for (core, c) in self.cores.iter().enumerate() {
            let priority = self.preemption_priority(core);
            if c.running_priority.load(Ordering::Acquire) != priority {
                continue;
            }
            if c.queue.lock().mask & (1 << priority) != 0 {
                c.rotate.store(true, Ordering::Release);
                c.resched.store(true, Ordering::Release);
                self.interrupt(core);
            }
        }
    }

    /// Takes every running thread off its core and keeps the cores idle until [`Scheduler::resume`].
    /// Threads stay runnable; waits and timers go on. Returns once no core runs guest code.
    pub(crate) fn pause(&self) {
        self.paused.store(true, Ordering::SeqCst);
        for core in 0..self.cores.len() {
            self.request_reschedule(core);
        }
        let mut guard = self.left_guest_lock.lock();
        while self.cores.iter().any(|c| c.in_guest.load(Ordering::SeqCst)) {
            self.left_guest.wait_for(&mut guard, GUEST_EXIT_POLL);
        }
    }

    /// Marks `core` as running guest code. False while paused: the core must not enter then.
    pub(crate) fn enter_guest(&self, core: usize) -> bool {
        self.cores[core].in_guest.store(true, Ordering::SeqCst);
        if self.paused.load(Ordering::SeqCst) {
            self.leave_guest(core);
            return false;
        }
        true
    }

    /// Records the SVC `core` is handling (`None` when done), for the thread dump.
    pub(crate) fn set_svc(&self, core: usize, svc: Option<u32>) {
        self.cores[core]
            .svc
            .store(svc.unwrap_or(NO_SVC), Ordering::Relaxed);
    }

    pub(crate) fn leave_guest(&self, core: usize) {
        self.cores[core].in_guest.store(false, Ordering::SeqCst);
        if self.paused.load(Ordering::SeqCst) {
            let _guard = self.left_guest_lock.lock();
            self.left_guest.notify_all();
        }
    }

    pub(crate) fn resume(&self) {
        self.paused.store(false, Ordering::Release);
        for core in 0..self.cores.len() {
            self.unpark(core);
        }
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    pub(crate) fn is_shut_down(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    pub(crate) fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        for core in 0..self.cores.len() {
            self.request_reschedule(core);
            self.unpark(core);
        }
    }

    /// Every thread in a run queue.
    pub(crate) fn queued_threads(&self) -> Vec<Arc<Thread>> {
        self.cores
            .iter()
            .flat_map(|core| {
                let queue = core.queue.lock();
                queue.levels.iter().flatten().cloned().collect::<Vec<_>>()
            })
            .collect()
    }

    /// Each core's running priority and queued threads, for the `debug.svc_trace` dump at
    /// shutdown.
    pub(crate) fn describe(&self, running: &[Vec<ThreadId>]) -> Vec<String> {
        self.cores
            .iter()
            .enumerate()
            .map(|(index, core)| {
                let queue = core.queue.lock();
                let queued: Vec<(usize, Vec<u64>)> = (0..LEVELS)
                    .filter(|&priority| !queue.levels[priority].is_empty())
                    .map(|priority| {
                        let ids = queue.levels[priority].iter().map(|t| t.id()).collect();
                        (priority, ids)
                    })
                    .collect();
                format!(
                    "core={index} running={:?} running_priority={} in_guest={} svc={} resched={} idle={} queued={queued:?}",
                    running[index],
                    core.running_priority.load(Ordering::Acquire),
                    core.in_guest.load(Ordering::Acquire),
                    match core.svc.load(Ordering::Relaxed) {
                        NO_SVC => "-".to_string(),
                        svc => format!("{svc:#04x}"),
                    },
                    core.resched.load(Ordering::Acquire),
                    self.idle.load(Ordering::Acquire) & (1 << index) != 0,
                )
            })
            .collect()
    }

    /// Drops every queued thread (on shutdown, to break thread/scheduler reference cycles).
    pub(crate) fn clear(&self) {
        for c in self.cores.iter() {
            *c.queue.lock() = RunQueue::new();
        }
    }
}

enum Take {
    Found(Picked),
    Contended,
    Empty,
}
