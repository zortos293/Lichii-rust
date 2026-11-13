//! Live guest pcs of the threads the cores are running, for the kernel watchdog's thread log.
//!
//! A running thread's saved context is stale, so a thread that spins in guest code without a
//! supervisor call shows where it last stopped, not where it is. [`LivePcs::sample`] interrupts
//! the core instead: translated code checks the interrupt flag at every block entry, so the core
//! stops at the next block boundary with its context current, records it and carries on. Set
//! `LICHII_LIVE_PC` to make the watchdog add these samples to every thread log.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use lichii_common::VAddr;
use lichii_cpu::ThreadContext;
use parking_lot::{Condvar, Mutex};

use crate::scheduler::Scheduler;
use crate::thread::Thread;

/// Where a core's running thread was when the core next stopped at a block boundary.
pub(crate) struct LivePc {
    pub thread: Arc<Thread>,
    pub pc: VAddr,
    pub lr: VAddr,
    pub sp: VAddr,
}

pub(crate) struct LivePcs {
    requested: Box<[AtomicBool]>,
    samples: Mutex<Vec<Option<LivePc>>>,
    recorded: Condvar,
}

impl LivePcs {
    pub fn new(cores: usize) -> Self {
        Self {
            requested: (0..cores).map(|_| AtomicBool::new(false)).collect(),
            samples: Mutex::new((0..cores).map(|_| None).collect()),
            recorded: Condvar::new(),
        }
    }

    /// Called by `core` whenever its CPU halts for an interrupt: records `ctx` when a sample was
    /// requested, and costs one relaxed load otherwise.
    pub fn record(&self, core: usize, thread: &Arc<Thread>, ctx: &ThreadContext) {
        let requested = &self.requested[core];
        if !requested.load(Ordering::Relaxed) || !requested.swap(false, Ordering::AcqRel) {
            return;
        }
        self.samples.lock()[core] = Some(LivePc {
            thread: thread.clone(),
            pc: ctx.pc,
            lr: ctx.x[30],
            sp: ctx.sp,
        });
        self.recorded.notify_all();
    }

    /// Interrupts `core` and waits up to `timeout` for it to record where its running thread is.
    /// `None` when the core reached no block boundary in time: it runs host code, or no thread.
    pub fn sample(&self, scheduler: &Scheduler, core: usize, timeout: Duration) -> Option<LivePc> {
        let deadline = Instant::now() + timeout;
        let mut samples = self.samples.lock();
        samples[core] = None;
        self.requested[core].store(true, Ordering::Release);
        scheduler.interrupt(core);
        while samples[core].is_none() {
            if self.recorded.wait_until(&mut samples, deadline).timed_out() {
                break;
            }
        }
        self.requested[core].store(false, Ordering::Release);
        samples[core].take()
    }
}
