//! Waitable kernel objects and the wait/wake protocol.
//!
//! A thread that blocks takes a fresh wait token, registers `(thread, token)` with every queue it
//! waits on, then commits. Whoever wakes it first (a signal, a timeout, a cancel) clears the token
//! under the thread's lock; every other registration becomes stale and is skipped when its queue
//! is next walked. No global lock is involved: each queue has its own lock, ordered
//! queue -> thread -> run queue.

use std::sync::Arc;

use lichii_common::ResultCode;
use parking_lot::Mutex;

use crate::thread::{Thread, WakeResult};

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

#[derive(Default)]
struct Inner {
    signaled: bool,
    waiters: Vec<Waiter>,
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

    /// Sets the object signaled and wakes every waiter.
    pub fn signal(&self) {
        let mut inner = self.inner.lock();
        inner.signaled = true;
        for waiter in inner.waiters.drain(..) {
            waiter.wake(ResultCode::SUCCESS);
        }
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
}
