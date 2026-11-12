//! Guest threads.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
    pub priority: u8,
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

    /// Abandons a wait that turned out not to be needed.
    pub(crate) fn cancel_wait(&self, token: u64) {
        let mut s = self.sched.lock();
        if s.wait == token {
            s.wait = 0;
            self.wait_token.store(0, Ordering::Release);
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
