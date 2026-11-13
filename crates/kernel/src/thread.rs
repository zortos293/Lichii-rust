//! Guest threads.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use lichii_common::{ResultCode, VAddr};
use lichii_cpu::ThreadContext;
use parking_lot::Mutex;

use crate::process::Process;
use crate::scheduler::Scheduler;
use crate::sync::SyncState;

pub type ThreadId = u64;

/// Lowest priority a guest thread can have (numerically highest).
pub const LOWEST_PRIORITY: u8 = 63;
/// Size of a thread's TLS slot (`TPIDRRO_EL0`).
pub const TLS_SIZE: u64 = 0x200;
/// Longest chain of mutex owners a priority change is passed along.
const MAX_INHERITANCE_DEPTH: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadState {
    /// Created and not started yet.
    Created,
    /// In its active core's run queue.
    Runnable,
    /// Running on the given core.
    Running(usize),
    /// Blocked until woken through its wait token.
    Waiting,
    /// Paused (`SetThreadActivity`) while it would otherwise be runnable.
    Suspended,
    Terminated,
}

/// What a waker delivers to a blocked thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WakeResult {
    pub result: ResultCode,
    /// Index of the signalled object for `WaitSynchronization`, -1 otherwise.
    pub index: i32,
}

impl WakeResult {
    pub const SUCCESS: Self = Self {
        result: ResultCode::SUCCESS,
        index: -1,
    };

    pub const fn error(result: ResultCode) -> Self {
        Self { result, index: -1 }
    }
}

/// How a wake result is written into the context when the thread resumes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Resume {
    /// Nothing is written (sleeps, yields, preemption).
    #[default]
    Nothing,
    /// `x0` = result.
    Result,
    /// `x0` = result, `x1` = index (`WaitSynchronization`).
    ResultAndIndex,
}

pub(crate) struct Sched {
    pub state: ThreadState,
    /// The effective priority: the base priority, raised by the threads waiting for mutexes this
    /// thread owns.
    pub priority: u8,
    /// The priority the guest set.
    pub base_priority: u8,
    pub affinity: u64,
    pub ideal_core: i32,
    pub active_core: usize,
    pub next_token: u64,
    /// Token of the wait in progress, 0 when none.
    pub wait: u64,
    pub cancellable: bool,
    pub wake: WakeResult,
    pub resume: Resume,
    pub pause_requested: bool,
    pub cancel_pending: bool,
    pub terminate_requested: bool,
}

/// A guest thread: its saved context, scheduling state and TLS slot.
pub struct Thread {
    id: ThreadId,
    process: Arc<Process>,
    tls: VAddr,
    stack_top: VAddr,
    pub(crate) scheduler: Arc<Scheduler>,
    pub(crate) context: Mutex<ThreadContext>,
    pub(crate) sched: Mutex<Sched>,
    wait_token: AtomicU64,
    pub(crate) sync: SyncState,
    /// Threads waiting for mutexes this thread owns; they lend it their priority.
    lock_waiters: Mutex<Vec<LockWaiter>>,
    /// The owner of the mutex this thread waits for.
    lock_owner: Mutex<Weak<Thread>>,
}

struct LockWaiter {
    thread: Arc<Thread>,
    token: u64,
    mutex: VAddr,
}

pub(crate) struct NewThread {
    pub id: ThreadId,
    pub process: Arc<Process>,
    pub scheduler: Arc<Scheduler>,
    pub entry: VAddr,
    pub arg: u64,
    pub stack_top: VAddr,
    pub tls: VAddr,
    pub priority: u8,
    pub ideal_core: i32,
    pub affinity: u64,
}

impl Thread {
    pub(crate) fn new(new: NewThread) -> Arc<Self> {
        let mut context = ThreadContext {
            pc: new.entry,
            sp: new.stack_top & !0xf,
            tpidrro_el0: new.tls,
            ..ThreadContext::default()
        };
        context.x[0] = new.arg;
        Arc::new(Self {
            id: new.id,
            process: new.process,
            tls: new.tls,
            stack_top: new.stack_top,
            scheduler: new.scheduler,
            context: Mutex::new(context),
            sched: Mutex::new(Sched {
                state: ThreadState::Created,
                priority: new.priority,
                base_priority: new.priority,
                affinity: new.affinity,
                ideal_core: new.ideal_core,
                active_core: new.ideal_core.max(0) as usize,
                next_token: 1,
                wait: 0,
                cancellable: false,
                wake: WakeResult::SUCCESS,
                resume: Resume::Nothing,
                pause_requested: false,
                cancel_pending: false,
                terminate_requested: false,
            }),
            wait_token: AtomicU64::new(0),
            sync: SyncState::default(),
            lock_waiters: Mutex::default(),
            lock_owner: Mutex::default(),
        })
    }

    pub fn id(&self) -> ThreadId {
        self.id
    }

    pub fn process(&self) -> &Arc<Process> {
        &self.process
    }

    /// Address of the thread's TLS slot.
    pub fn tls_address(&self) -> VAddr {
        self.tls
    }

    pub fn stack_top(&self) -> VAddr {
        self.stack_top
    }

    pub fn state(&self) -> ThreadState {
        self.sched.lock().state
    }

    pub fn priority(&self) -> u8 {
        self.sched.lock().priority
    }

    /// Sets the priority the guest asked for; the effective priority stays raised while
    /// higher-priority threads wait for mutexes this thread owns.
    pub(crate) fn set_base_priority(self: &Arc<Self>, priority: u8) {
        self.sched.lock().base_priority = priority;
        self.restore_priority();
    }

    /// `waiter` (in wait `token`) waits for the mutex at `mutex`, which this thread owns.
    pub(crate) fn add_lock_waiter(
        self: &Arc<Self>,
        waiter: &Arc<Thread>,
        token: u64,
        mutex: VAddr,
    ) {
        *waiter.lock_owner.lock() = Arc::downgrade(self);
        self.lock_waiters.lock().push(LockWaiter {
            thread: waiter.clone(),
            token,
            mutex,
        });
        self.restore_priority();
    }

    /// This thread released the mutex at `mutex` to `next`, or to nobody: its waiters now wait
    /// for `next`, and lend their priority to it instead.
    pub(crate) fn pass_lock(self: &Arc<Self>, mutex: VAddr, next: Option<&Arc<Thread>>) {
        let moved: Vec<LockWaiter> = {
            let mut waiters = self.lock_waiters.lock();
            let (moved, kept) = std::mem::take(&mut *waiters)
                .into_iter()
                .partition(|waiter| waiter.mutex == mutex);
            *waiters = kept;
            moved
        };
        if let Some(next) = next {
            *next.lock_owner.lock() = Weak::new();
            let moved: Vec<LockWaiter> = moved
                .into_iter()
                .filter(|w| !Arc::ptr_eq(&w.thread, next) && w.thread.is_waiting_on(w.token))
                .collect();
            for waiter in &moved {
                *waiter.thread.lock_owner.lock() = Arc::downgrade(next);
            }
            next.lock_waiters.lock().extend(moved);
            next.restore_priority();
        }
        self.restore_priority();
    }

    /// Recomputes the effective priority from the base priority and the live lock waiters, and
    /// passes a change on along the chain of mutex owners.
    fn restore_priority(self: &Arc<Self>) {
        let mut thread = self.clone();
        for _ in 0..MAX_INHERITANCE_DEPTH {
            let inherited = {
                let mut waiters = thread.lock_waiters.lock();
                waiters.retain(|waiter| waiter.thread.is_waiting_on(waiter.token));
                waiters.iter().map(|waiter| waiter.thread.priority()).min()
            };
            {
                let mut s = thread.sched.lock();
                let priority = inherited.map_or(s.base_priority, |p| p.min(s.base_priority));
                if s.priority == priority {
                    return;
                }
                thread.apply_priority(&mut s, priority);
            }
            let Some(owner) = thread.lock_owner.lock().upgrade() else {
                return;
            };
            thread = owner;
        }
    }

    fn apply_priority(self: &Arc<Self>, s: &mut Sched, priority: u8) {
        match s.state {
            ThreadState::Runnable => {
                self.scheduler.dequeue(self, s);
                s.priority = priority;
                self.scheduler.enqueue(self, s, false);
            }
            ThreadState::Running(core) => {
                s.priority = priority;
                self.scheduler.set_running_priority(core, priority);
                self.scheduler.request_reschedule(core);
            }
            _ => s.priority = priority,
        }
    }

    /// `(ideal core, affinity mask)`.
    pub fn core_mask(&self) -> (i32, u64) {
        let s = self.sched.lock();
        (s.ideal_core, s.affinity)
    }

    /// Copy of the saved context. Only meaningful while the thread is not running.
    pub fn context(&self) -> ThreadContext {
        self.context.lock().clone()
    }

    pub fn set_context(&self, context: ThreadContext) {
        *self.context.lock() = context;
    }

    /// Whether the thread terminated. Threads are signalled when they exit.
    pub fn is_terminated(&self) -> bool {
        self.sync.is_signaled()
    }

    pub(crate) fn is_waiting_on(&self, token: u64) -> bool {
        token != 0 && self.wait_token.load(Ordering::Acquire) == token
    }

    /// Starts a wait from the thread itself. Fails with the pending result when the wait is
    /// cancellable and a cancel or termination is pending.
    pub(crate) fn begin_wait(&self, resume: Resume, cancellable: bool) -> Result<u64, ResultCode> {
        let mut s = self.sched.lock();
        if s.terminate_requested {
            return Err(crate::result::TERMINATION_REQUESTED);
        }
        if cancellable && std::mem::take(&mut s.cancel_pending) {
            return Err(crate::result::CANCELLED);
        }
        let token = s.next_token;
        s.next_token += 1;
        s.wait = token;
        s.cancellable = cancellable;
        s.resume = resume;
        s.wake = WakeResult::SUCCESS;
        self.wait_token.store(token, Ordering::Release);
        Ok(token)
    }

    /// Abandons a wait that turned out not to be needed, together with its resume action (also
    /// when a waker got there first), so a later context switch cannot replay it.
    pub(crate) fn cancel_wait(&self, token: u64) {
        let mut s = self.sched.lock();
        if s.wait == token {
            s.wait = 0;
            self.wait_token.store(0, Ordering::Release);
        }
        if s.wait == 0 {
            s.resume = Resume::Nothing;
        }
    }

    /// Wakes the thread if `token` is its current wait. Returns whether it did.
    pub(crate) fn wake(self: &Arc<Self>, token: u64, wake: WakeResult) -> bool {
        let mut s = self.sched.lock();
        if token == 0 || s.wait != token {
            return false;
        }
        self.wake_locked(&mut s, wake);
        true
    }

    pub(crate) fn wake_locked(self: &Arc<Self>, s: &mut Sched, wake: WakeResult) {
        s.wait = 0;
        s.wake = wake;
        self.wait_token.store(0, Ordering::Release);
        if s.state == ThreadState::Waiting {
            if s.pause_requested {
                s.state = ThreadState::Suspended;
            } else {
                self.scheduler.enqueue(self, s, false);
            }
        }
    }

    /// Wakes the thread from the cancellable wait it is in, or makes its next one fail.
    pub(crate) fn cancel_synchronization(self: &Arc<Self>) {
        let mut s = self.sched.lock();
        if s.wait == 0 || !s.cancellable {
            s.cancel_pending = true;
            return;
        }
        self.wake_locked(&mut s, WakeResult::error(crate::result::CANCELLED));
    }

    /// Asks the thread to terminate: any wait ends with `TerminationRequested` and the thread
    /// exits the next time a core looks at it.
    pub(crate) fn request_termination(self: &Arc<Self>) {
        let mut s = self.sched.lock();
        s.terminate_requested = true;
        s.pause_requested = false;
        match s.state {
            ThreadState::Waiting => self.wake_locked(
                &mut s,
                WakeResult::error(crate::result::TERMINATION_REQUESTED),
            ),
            ThreadState::Suspended | ThreadState::Created => {
                self.scheduler.enqueue(self, &mut s, true)
            }
            ThreadState::Running(core) => self.scheduler.request_reschedule(core),
            ThreadState::Runnable | ThreadState::Terminated => {}
        }
    }
}
