//! Bounded events for attributing submit-to-front latency. Only the activity reporter formats
//! logs; the GPU threads never wait for it. The decoder joins events by capture, wait and fence.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crossbeam_channel::{Receiver, Sender, bounded};
use lichii_common::DAddr;

use crate::backend::{ChannelId, Rasterizer};

const CAPACITY: usize = 32_768;
const BATCH: usize = 64;
const SEALED: usize = 1 << (usize::BITS - 1);
static NEXT_CAPTURE: AtomicU64 = AtomicU64::new(1);

#[derive(Debug)]
pub(crate) struct QueueTrace {
    base: Instant,
    capture: u64,
    unix_start_ns: u64,
    next_id: AtomicU64,
    dropped: AtomicU64,
    writers: AtomicUsize,
    workers_joined: AtomicBool,
    send: Sender<Stamped>,
    recv: Receiver<Stamped>,
}

#[derive(Clone, Copy)]
pub(crate) enum Event {
    Submit(u32, u32, u32, u64),
    Front(u64),
    Run(u32),
    Stop(u32),
    Close(u32),
    Wait(u32, char, u64, u64),
    Resume(u32, u64),
    FetchReady(u64),
    SyncReady(u32, u32),
    Acquire(u64, Option<u64>, u64, u8, bool),
    Latch(u64, u64, u64, bool),
    Offer(u64, u64, u64, Option<u64>),
    Producer(u64, u32, u64),
    Publish(u64, u64, u64, Option<u64>, bool),
    Retire(u64),
}

#[derive(Clone, Copy)]
struct Stamped {
    at: u64,
    event: Event,
}

impl fmt::Display for Stamped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Event::*;
        let optional = |value: Option<u64>| value.map_or(-1, i128::from);
        write!(f, "{},", self.at)?;
        match self.event {
            Submit(c, id, value, sample) => write!(f, "submit,{c},{id},{value},{sample}"),
            Front(sample) => write!(f, "front,{sample}"),
            Run(c) => write!(f, "run,{c}"),
            Stop(c) => write!(f, "stop,{c}"),
            Close(c) => write!(f, "close,{c}"),
            Wait(c, kind, id, key) => write!(f, "wait,{c},{kind},{id},{key}"),
            Resume(c, last_unready) => write!(f, "resume,{c},{last_unready}"),
            FetchReady(id) => write!(f, "fetch_ready,{id}"),
            SyncReady(id, value) => write!(f, "sync_ready,{id},{value}"),
            Acquire(id, addr, payload, op, wide) => write!(
                f,
                "acquire,{id},{},{payload},{op},{}",
                optional(addr),
                u8::from(wide)
            ),
            Latch(id, fence, value, queued) => {
                write!(f, "latch,{id},{fence},{value},{}", u8::from(queued))
            }
            Offer(fence, addr, bytes, value) => {
                write!(f, "offer,{fence},{addr},{bytes},{}", optional(value))
            }
            Producer(fence, c, before) => write!(f, "producer,{fence},{c},{before}"),
            Publish(fence, addr, bytes, value, ok) => write!(
                f,
                "publish,{fence},{addr},{bytes},{},{}",
                optional(value),
                u8::from(ok)
            ),
            Retire(fence) => write!(f, "retire,{fence}"),
        }
    }
}

impl QueueTrace {
    pub(crate) fn if_enabled() -> Option<Self> {
        tracing::enabled!(target: "lichii_gpu::queue_trace", tracing::Level::DEBUG)
            .then(|| Self::new(CAPACITY))
    }

    fn new(capacity: usize) -> Self {
        let (send, recv) = bounded(capacity);
        Self {
            base: Instant::now(),
            capture: NEXT_CAPTURE.fetch_add(1, Ordering::Relaxed),
            unix_start_ns: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64),
            next_id: AtomicU64::new(1),
            dropped: AtomicU64::new(0),
            writers: AtomicUsize::new(0),
            workers_joined: AtomicBool::new(false),
            send,
            recv,
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    pub(crate) fn record(&self, event: Event) {
        if self.writers.fetch_add(1, Ordering::Acquire) & SEALED == 0 {
            let at = self.now();
            if self.send.try_send(Stamped { at, event }).is_err() {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.writers.fetch_sub(1, Ordering::Release);
    }

    pub(crate) fn now(&self) -> u64 {
        self.base.elapsed().as_nanos() as u64
    }

    pub(crate) fn workers_joined(&self) {
        self.workers_joined.store(true, Ordering::Release);
    }

    /// Closes the capture to other producers, including CPU-side syncpoint increments. If
    /// an append is still in flight, the final dump is only a prefix and the reporter won't
    /// wait for a producer that might not return.
    fn seal(&self) -> bool {
        let writers = self.writers.fetch_or(SEALED, Ordering::AcqRel);
        writers == 0 && self.workers_joined.load(Ordering::Acquire)
    }

    /// The renderer owns FIFO identities. Bracket a single enqueue on its thread and
    /// associate a producer only if that call added a newer overlapping entry.
    pub(crate) fn track_producer(
        &self,
        rasterizer: &mut dyn Rasterizer,
        channel: ChannelId,
        address: DAddr,
        signal: impl FnOnce(&mut dyn Rasterizer),
    ) {
        let previous = rasterizer
            .pending_semaphore(address, false)
            .map_or(0, |w| w.fence);
        let before = self.now();
        signal(rasterizer);
        if let Some(write) = rasterizer.pending_semaphore(address, false)
            && write.fence > previous
        {
            self.record(Event::Producer(write.fence, channel.0, before));
        }
    }

    /// Only the reporter receives, and it polls. No sleeping queue receiver can make a sender
    /// acquire the channel's notification lock.
    pub(crate) fn drain(&self, final_drain: bool) {
        use fmt::Write;

        let end = final_drain && self.seal();
        if !tracing::enabled!(target: "lichii_gpu::queue_trace", tracing::Level::DEBUG) {
            return;
        }
        let capture = self.capture;
        let unix_start_ns = self.unix_start_ns;
        let mut data = String::with_capacity(BATCH * 64);
        let mut remaining = self.recv.len();
        while remaining != 0 {
            data.clear();
            for _ in 0..remaining.min(BATCH) {
                let Ok(event) = self.recv.try_recv() else {
                    break;
                };
                if !data.is_empty() {
                    data.push(';');
                }
                let _ = write!(data, "{event}");
            }
            if data.is_empty() {
                break;
            }
            tracing::debug!(target: "lichii_gpu::queue_trace", capture, unix_start_ns, data = %data, "GPU queue events");
            remaining = remaining.saturating_sub(BATCH);
        }
        let dropped = self.dropped.load(Ordering::Relaxed);
        if dropped != 0 || final_drain {
            tracing::debug!(target: "lichii_gpu::queue_trace", capture, unix_start_ns, dropped, end, "GPU queue trace status");
        }
    }

    #[cfg(test)]
    pub(crate) fn events(&self) -> Vec<Event> {
        self.recv.try_iter().map(|stamped| stamped.event).collect()
    }

    #[cfg(test)]
    pub(crate) fn joined_for_test(&self) -> bool {
        self.workers_joined.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_trace_drops_events_without_overwriting_other_samples() {
        let trace = QueueTrace::new(1);
        trace.record(Event::Front(1));
        trace.record(Event::Front(2));
        assert_eq!(trace.dropped.load(Ordering::Relaxed), 1);
        assert!(matches!(
            trace.recv.try_recv().unwrap().event,
            Event::Front(1)
        ));
        assert!(trace.recv.is_empty());
    }

    #[test]
    fn sealing_requires_joined_workers_and_no_live_append() {
        let timed_out = QueueTrace::new(2);
        assert!(!timed_out.seal());

        let appending = QueueTrace::new(2);
        appending.workers_joined();
        appending.writers.fetch_add(1, Ordering::Acquire);
        assert!(!appending.seal());

        let joined = QueueTrace::new(2);
        joined.record(Event::Front(1));
        joined.workers_joined();
        assert!(joined.seal());
        joined.record(Event::Front(2));
        assert!(matches!(joined.events().as_slice(), [Event::Front(1)]));
    }
}
