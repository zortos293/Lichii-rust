//! Guest CPU accesses to memory the GPU modified: the flush trap handler the GPU installs on its
//! device address space, and the queue the back serves it from. The back writes a waiting range
//! back at its next record boundary, or as soon as it idles, from the work it already executed;
//! a read never waits for queued submissions to be parsed or executed. A renderer may instead
//! start the copy and hand the read back pending ([`CpuReadDone::Pending`]): the waiting thread
//! then waits for the host GPU itself and asks again, and the back writes the copied bytes back
//! in order with everything else it does to guest memory.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::thread::ThreadId;

use lichii_common::DAddr;
use lichii_memory::FlushTrapHandler;
use parking_lot::{Condvar, Mutex};

use crate::backend::{CpuReadCompleter, CpuReadDone, CpuReadTicket, Rasterizer};
use crate::throttle::Throttle;

/// A range a guest CPU thread waits for, with the ticket of the read the back left pending for
/// it once the thread saw that read's copy finish.
pub(crate) type QueuedRead = (DAddr, u64, Option<CpuReadTicket>);

/// Device ranges guest CPU threads wait for the back to write back. Every queue holds at most
/// one entry per waiting thread.
pub(crate) struct CpuReads {
    pending: AtomicBool,
    state: Mutex<State>,
    served: Condvar,
    /// Gets the thread that runs the back to look at the queue.
    wake: Box<dyn Fn() + Send + Sync>,
    /// What waits for the copies of pending reads, taken from the renderer at its first one.
    completer: OnceLock<Option<Arc<dyn CpuReadCompleter>>>,
}

#[derive(Default)]
struct State {
    ranges: Vec<QueuedRead>,
    /// Served reads left pending, by request ticket, until their threads take them.
    results: Vec<(u64, CpuReadDone)>,
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
            completer: OnceLock::new(),
        }
    }

    /// Queues `[daddr, daddr + size)` and returns once its bytes are in guest memory, or the GPU
    /// stopped. For a read the back left pending, this thread waits for the host GPU copy and
    /// queues the range again with its ticket, which the back serves without leaving it pending.
    fn request(&self, daddr: DAddr, size: u64) {
        let mut after = None;
        loop {
            let ticket = {
                let mut state = self.state.lock();
                if state.closed {
                    return;
                }
                state.ranges.push((daddr, size, after));
                state.requested += 1;
                self.pending.store(true, Ordering::Relaxed);
                state.requested
            };
            (self.wake)();
            let done = {
                let mut state = self.state.lock();
                while state.served < ticket && !state.closed {
                    self.served.wait(&mut state);
                }
                let index = state.results.iter().position(|&(t, _)| t == ticket);
                index.map(|index| state.results.swap_remove(index).1)
            };
            let Some(CpuReadDone::Pending(read)) = done else {
                return;
            };
            if let Some(Some(completer)) = self.completer.get() {
                completer.complete(read);
            }
            after = Some(read);
        }
    }

    /// Whether a range waits: one relaxed load, for the back to test between records.
    #[inline]
    pub(crate) fn pending(&self) -> bool {
        self.pending.load(Ordering::Relaxed)
    }

    /// Passes every waiting range to `rasterizer`, then releases its readers. `scratch` is an
    /// empty buffer the caller keeps, so serving allocates nothing once warm.
    pub(crate) fn serve(&self, scratch: &mut Vec<QueuedRead>, rasterizer: &mut dyn Rasterizer) {
        let requested = {
            let mut state = self.state.lock();
            self.pending.store(false, Ordering::Relaxed);
            std::mem::swap(&mut state.ranges, scratch);
            state.requested
        };
        let first = requested + 1 - scratch.len() as u64;
        for (ticket, &(daddr, size, after)) in (first..).zip(scratch.iter()) {
            let done = rasterizer.begin_cpu_read(daddr, size, after);
            if done != CpuReadDone::Home {
                self.completer
                    .get_or_init(|| rasterizer.cpu_read_completer());
                self.state.lock().results.push((ticket, done));
            }
        }
        rasterizer.end_cpu_reads();
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use parking_lot::Mutex;

    use super::CpuReads;
    use crate::backend::{
        ChannelId, ComputeContext, CpuReadCompleter, CpuReadDone, CpuReadTicket, DrawParams,
        FenceAction, FramebufferConfig, GraphicsContext, IndirectDraw, Query, Rasterizer,
    };
    use crate::gmmu::Gmmu;

    #[derive(Debug, PartialEq, Eq)]
    enum Event {
        Begin(u64),
        Again(u64),
        End,
        Complete(u32),
    }

    struct Log {
        events: Mutex<Vec<Event>>,
        ended: AtomicBool,
    }

    impl CpuReadCompleter for Log {
        fn complete(&self, ticket: CpuReadTicket) {
            assert!(
                self.ended.load(Ordering::Acquire),
                "a read completed before its batch was submitted"
            );
            self.events.lock().push(Event::Complete(ticket.slot));
        }
    }

    /// Leaves first reads of odd addresses pending, in a slot named by the address.
    struct Renderer(Arc<Log>);

    impl Rasterizer for Renderer {
        fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}
        fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}
        fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}
        fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}
        fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, _query: &Query) {}
        fn signal_fence(&mut self, _action: FenceAction) {}
        fn release_fences(&mut self, _wait: bool) {}
        fn invalidate_region(&mut self, _addr: u64, _size: u64) {}
        fn flush_region(&mut self, _addr: u64, _size: u64) {}
        fn wait_for_idle(&mut self) {}
        fn present(&mut self, _framebuffer: &FramebufferConfig) {}

        fn begin_cpu_read(
            &mut self,
            addr: u64,
            _size: u64,
            after: Option<CpuReadTicket>,
        ) -> CpuReadDone {
            if let Some(ticket) = after {
                assert_eq!(
                    u64::from(ticket.slot),
                    addr,
                    "asked again with its own ticket"
                );
                self.0.events.lock().push(Event::Again(addr));
                return CpuReadDone::Home;
            }
            self.0.events.lock().push(Event::Begin(addr));
            if addr.is_multiple_of(2) {
                return CpuReadDone::Home;
            }
            CpuReadDone::Pending(CpuReadTicket {
                slot: addr as u32,
                generation: 0,
            })
        }

        fn end_cpu_reads(&mut self) {
            self.0.events.lock().push(Event::End);
            self.0.ended.store(true, Ordering::Release);
        }

        fn cpu_read_completer(&self) -> Option<Arc<dyn CpuReadCompleter>> {
            Some(self.0.clone())
        }
    }

    /// Reads served in one batch: the back begins each and submits them once, then releases
    /// the waiting threads. Each thread whose read was left pending waits for its copy itself and
    /// asks again with its ticket, and the back's second batch ends those reads.
    #[test]
    fn pending_reads_finish_on_their_waiting_threads() {
        let log = Arc::new(Log {
            events: Mutex::new(Vec::new()),
            ended: AtomicBool::new(false),
        });
        let reads = Arc::new(CpuReads::new(Box::new(|| {})));
        let threads: Vec<_> = [1u64, 2, 3]
            .into_iter()
            .map(|addr| {
                let reads = reads.clone();
                std::thread::spawn(move || reads.request(addr, 0x1000))
            })
            .collect();
        while reads.state.lock().ranges.len() < 3 {
            std::thread::yield_now();
        }
        let mut renderer = Renderer(log.clone());
        let mut scratch = Vec::new();
        reads.serve(&mut scratch, &mut renderer);
        while reads.state.lock().ranges.len() < 2 {
            std::thread::yield_now();
        }
        reads.serve(&mut scratch, &mut renderer);
        for thread in threads {
            thread.join().unwrap();
        }
        let events = log.events.lock();
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, Event::Begin(_)))
                .count(),
            3
        );
        assert_eq!(events[3], Event::End, "one submission after every begin");
        let mut completed: Vec<_> = events[4..6].iter().collect();
        completed.sort_by_key(|e| match e {
            Event::Complete(slot) => *slot,
            _ => u32::MAX,
        });
        assert_eq!(completed, [&Event::Complete(1), &Event::Complete(3)]);
        let mut again: Vec<_> = events[6..8].iter().collect();
        again.sort_by_key(|e| match e {
            Event::Again(addr) => *addr,
            _ => u64::MAX,
        });
        assert_eq!(again, [&Event::Again(1), &Event::Again(3)]);
        assert_eq!(
            events[8..],
            [Event::End],
            "the second batch is submitted once too"
        );
        assert!(
            reads.state.lock().results.is_empty(),
            "every result was taken"
        );
    }
}
