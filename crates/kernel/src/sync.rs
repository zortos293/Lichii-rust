//! Waitable kernel objects and the wait/wake protocol.
//!
//! A thread that blocks takes a fresh wait token, registers `(thread, token)` with every queue it
//! waits on, then commits. Whoever wakes it first (a signal, a timeout, a cancel) clears the token
//! under the thread's lock; every other registration becomes stale and is skipped when its queue
//! is next walked. No global lock is involved: each queue has its own lock, ordered
//! queue -> thread -> run queue.

use std::fmt;
use std::sync::Arc;

use lichii_common::ResultCode;
use parking_lot::Mutex;

use crate::thread::{Thread, ThreadId, WakeResult};

thread_local! {
    static HOST_THREAD_NAME: Arc<str> = std::thread::current().name().unwrap_or("unnamed").into();
}

pub(crate) struct Waiter {
    pub thread: Arc<Thread>,
    pub token: u64,
    pub index: i32,
}

impl Waiter {
    pub fn is_live(&self) -> bool {
        self.thread.is_waiting_on(self.token)
    }

    pub fn wake(self, result: ResultCode) -> bool {
        self.thread.wake(
            self.token,
            WakeResult {
                result,
                index: self.index,
            },
        )
    }
}

/// Who last signalled a waitable object, for the watchdog's thread log.
#[derive(Clone, Default, PartialEq, Debug)]
pub(crate) enum Signaller {
    #[default]
    Never,
    Guest(ThreadId),
    Host(Arc<str>),
}

impl fmt::Display for Signaller {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Signaller::Never => f.write_str("never"),
            Signaller::Guest(thread) => write!(f, "thread {thread}"),
            Signaller::Host(name) => write!(f, "host {name}"),
        }
    }
}

/// A waitable object's state as the watchdog logs it.
pub(crate) struct SyncSnapshot {
    pub signaled: bool,
    pub signals: u32,
    pub last_signaller: Signaller,
    pub waiters: Vec<ThreadId>,
}

#[derive(Default)]
struct Inner {
    signaled: bool,
    signals: u32,
    last_signaller: Signaller,
    waiters: Vec<Waiter>,
}

impl Inner {
    fn signal(&mut self) {
        self.signaled = true;
        self.signals = self.signals.wrapping_add(1);
        for waiter in self.waiters.drain(..) {
            waiter.wake(ResultCode::SUCCESS);
        }
    }
}

/// The signal state and waiter list of a waitable object (events, threads, processes, sessions).
#[derive(Default)]
pub struct SyncState {
    inner: Mutex<Inner>,
}

impl SyncState {
    pub fn is_signaled(&self) -> bool {
        self.inner.lock().signaled
    }

    /// Sets the object signaled and wakes every waiter, noting the host thread as the signaller.
    pub fn signal(&self) {
        let mut inner = self.inner.lock();
        HOST_THREAD_NAME.with(|name| {
            if !matches!(&inner.last_signaller, Signaller::Host(last) if Arc::ptr_eq(last, name)) {
                inner.last_signaller = Signaller::Host(name.clone());
            }
        });
        inner.signal();
    }

    /// Sets the object signaled on behalf of a guest thread and wakes every waiter.
    pub(crate) fn signal_from_guest(&self, thread: ThreadId) {
        let mut inner = self.inner.lock();
        inner.last_signaller = Signaller::Guest(thread);
        inner.signal();
    }

    /// Clears the signal. Returns whether it was set.
    pub fn reset(&self) -> bool {
        std::mem::take(&mut self.inner.lock().signaled)
    }

    /// Registers `waiter` unless the object is already signaled.
    pub(crate) fn add_waiter_or_signaled(&self, waiter: Waiter) -> bool {
        let mut inner = self.inner.lock();
        if inner.signaled {
            return true;
        }
        inner.waiters.retain(Waiter::is_live);
        inner.waiters.push(waiter);
        false
    }

    /// The signal state, signal history and the threads still waiting.
    pub(crate) fn snapshot(&self) -> SyncSnapshot {
        let inner = self.inner.lock();
        SyncSnapshot {
            signaled: inner.signaled,
            signals: inner.signals,
            last_signaller: inner.last_signaller.clone(),
            waiters: inner
                .waiters
                .iter()
                .filter(|waiter| waiter.is_live())
                .map(|waiter| waiter.thread.id())
                .collect(),
        }
    }
}
