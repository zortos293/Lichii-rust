//! Guest CPU accesses to memory the GPU modified: the flush trap handler the GPU installs on its
//! device address space, and the queue the back serves it from. The back writes a waiting range
//! back at its next record boundary, or as soon as it idles, from the work it already executed;
//! a read never waits for queued submissions to be parsed or executed.

use std::sync::Weak;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::ThreadId;

use lichii_common::DAddr;
use lichii_memory::FlushTrapHandler;
use parking_lot::{Condvar, Mutex};

use crate::throttle::Throttle;

/// Device ranges guest CPU threads wait for the back to write back.
pub(crate) struct CpuReads {
    pending: AtomicBool,
    state: Mutex<State>,
    served: Condvar,
    /// Gets the thread that runs the back to look at the queue.
    wake: Box<dyn Fn() + Send + Sync>,
}

#[derive(Default)]
struct State {
    ranges: Vec<(DAddr, u64)>,
    requested: u64,
    served: u64,
    closed: bool,
}

impl CpuReads {
    pub(crate) fn new(wake: Box<dyn Fn() + Send + Sync>) -> Self {
        Self {
            pending: AtomicBool::new(false),
            state: Mutex::default(),
            served: Condvar::new(),
            wake,
        }
    }

    /// Queues `[daddr, daddr + size)` and returns once the back wrote it back, or the GPU stopped.
    fn request(&self, daddr: DAddr, size: u64) {
        let ticket = {
            let mut state = self.state.lock();
            if state.closed {
                return;
            }
            state.ranges.push((daddr, size));
            state.requested += 1;
            self.pending.store(true, Ordering::Relaxed);
            state.requested
        };
        (self.wake)();
        let mut state = self.state.lock();
        while state.served < ticket && !state.closed {
            self.served.wait(&mut state);
        }
    }

    /// Whether a range waits: one relaxed load, for the back to test between records.
    #[inline]
    pub(crate) fn pending(&self) -> bool {
        self.pending.load(Ordering::Relaxed)
    }

    /// Passes every waiting range to `flush`, then releases its readers. `scratch` is an empty buffer
    /// the caller keeps, so serving allocates nothing once warm.
    pub(crate) fn serve(&self, scratch: &mut Vec<(DAddr, u64)>, mut flush: impl FnMut(DAddr, u64)) {
        let requested = {
            let mut state = self.state.lock();
            self.pending.store(false, Ordering::Relaxed);
            std::mem::swap(&mut state.ranges, scratch);
            state.requested
        };
        for &(daddr, size) in scratch.iter() {
            flush(daddr, size);
        }
        scratch.clear();
        self.state.lock().served = requested;
        self.served.notify_all();
    }

    /// Releases every reader, now and later: the back is gone.
    pub(crate) fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        state.ranges.clear();
        self.served.notify_all();
    }
}

/// The [`FlushTrapHandler`] the GPU installs: the guest CPU thread that met a trap waits for the
/// back to write the page back.
pub(crate) struct CpuReadFlusher {
    pub reads: Weak<CpuReads>,
    /// The GPU's own threads, which must never wait for the back.
    pub gpu_threads: Vec<ThreadId>,
}

impl FlushTrapHandler for CpuReadFlusher {
    fn flush_for_cpu(&self, daddr: DAddr, size: u64) {
        if self.gpu_threads.contains(&std::thread::current().id()) {
            static GPU_THREAD: Throttle = Throttle::new();
            if let Some(dropped) = GPU_THREAD.allow() {
                tracing::warn!(
                    daddr,
                    dropped,
                    "a GPU thread met a flush trap; reading stale memory"
                );
            }
            return;
        }
        if let Some(reads) = self.reads.upgrade() {
            reads.request(daddr, size);
        }
    }
}
