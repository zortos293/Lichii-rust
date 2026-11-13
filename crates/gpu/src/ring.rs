//! The single-producer single-consumer ring between the GPU's front and back threads
//! (NX2_GPU_THREAD_SPLIT.md).
//!
//! Slots are `u64`s. The producer writes records and makes them visible by publishing its
//! position; the consumer executes everything below the published position and reports how far
//! it got. A record never straddles the end of the ring: one that does not fit there follows a
//! [`WRAP`] slot and starts over at the beginning. Neither side takes a lock unless it has nothing
//! to do and goes to sleep.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::gpu::LONG_WAIT;

/// Slot value that sends the consumer back to the start of the ring.
pub(crate) const WRAP: u64 = 0;

/// Spins before a waiting side yields, then sleeps.
const SPINS: u32 = 256;
const YIELDS: u32 = 16;

/// Keeps the producer's and the consumer's positions on separate cache line pairs.
#[repr(align(128))]
struct Padded<T>(T);

/// A wait for a condition another thread makes true, spinning briefly before it sleeps. One
/// thread waits on a signal at a time.
pub(crate) struct Signal {
    sleeping: AtomicBool,
    lock: Mutex<()>,
    cond: Condvar,
}

impl Signal {
    fn new() -> Self {
        Self {
            sleeping: AtomicBool::new(false),
            lock: Mutex::new(()),
            cond: Condvar::new(),
        }
    }

    /// Returns once `ready` holds. `ready` must read with acquire ordering what the notifier
    /// stores before [`Signal::notify`].
    pub(crate) fn wait_until(&self, ready: impl FnMut() -> bool) {
        self.wait_until_deadline(ready, None);
    }

    /// Like [`Signal::wait_until`], but gives up at `deadline`. Whether `ready` held.
    fn wait_until_deadline(
        &self,
        mut ready: impl FnMut() -> bool,
        deadline: Option<Instant>,
    ) -> bool {
        for _ in 0..SPINS {
            if ready() {
                return true;
            }
            std::hint::spin_loop();
        }
        for _ in 0..YIELDS {
            if ready() {
                return true;
            }
            std::thread::yield_now();
        }
        let mut guard = self.lock.lock();
        self.sleeping.store(true, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        let met = loop {
            if ready() {
                break true;
            }
            match deadline {
                None => self.cond.wait(&mut guard),
                Some(deadline) => {
                    if self.cond.wait_until(&mut guard, deadline).timed_out() {
                        break ready();
                    }
                }
            }
        };
        self.sleeping.store(false, Ordering::Relaxed);
        met
    }

    /// Wakes the waiter if it sleeps. Call it after storing what its condition reads.
    pub(crate) fn notify(&self) {
        fence(Ordering::SeqCst);
        if self.sleeping.load(Ordering::Relaxed) {
            let _guard = self.lock.lock();
            self.cond.notify_all();
        }
    }
}

struct Shared {
    slots: Box<[AtomicU64]>,
    mask: u64,
    published: Padded<AtomicU64>,
    consumed: Padded<AtomicU64>,
    /// The consumer waits here for the producer to publish.
    more: Signal,
    /// The producer waits here for the consumer to make progress.
    progress: Signal,
    closed: AtomicBool,
    consumer_alive: AtomicBool,
    /// Header of the record the consumer executes, for the producer's report of a stalled
    /// consumer.
    executing: Padded<AtomicU64>,
}

/// Creates a ring of `slots` slots (a power of two of at least `min_slots`).
pub(crate) fn ring(slots: usize, min_slots: usize) -> (Producer, Consumer) {
    let slots = slots.max(min_slots).next_power_of_two();
    let shared = Arc::new(Shared {
        slots: (0..slots).map(|_| AtomicU64::new(WRAP)).collect(),
        mask: slots as u64 - 1,
        published: Padded(AtomicU64::new(0)),
        consumed: Padded(AtomicU64::new(0)),
        more: Signal::new(),
        progress: Signal::new(),
        closed: AtomicBool::new(false),
        consumer_alive: AtomicBool::new(true),
        executing: Padded(AtomicU64::new(WRAP)),
    });
    (
        Producer {
            shared: shared.clone(),
            write: 0,
            published: 0,
            consumed: 0,
        },
        Consumer {
            shared,
            read: 0,
            end: 0,
            released: 0,
        },
    )
}

/// The writing end, owned by the front.
pub(crate) struct Producer {
    shared: Arc<Shared>,
    write: u64,
    published: u64,
    consumed: u64,
}

impl Producer {
    /// The position after everything written so far.
    #[inline]
    pub(crate) fn position(&self) -> u64 {
        self.write
    }

    /// Slots written since the last publish.
    #[inline]
    pub(crate) fn unpublished(&self) -> u64 {
        self.write - self.published
    }

    /// Makes room for a record of `len` slots, wrapping to the ring's start if it would not fit
    /// before the end. False when the consumer has to free slots first.
    #[inline]
    pub(crate) fn try_reserve(&mut self, len: usize) -> bool {
        let capacity = self.shared.mask + 1;
        let offset = self.write & self.shared.mask;
        let len = len as u64;
        let pad = if offset + len > capacity {
            capacity - offset
        } else {
            0
        };
        if self.write + pad + len - self.consumed > capacity {
            self.consumed = self.shared.consumed.0.load(Ordering::Acquire);
            if self.write + pad + len - self.consumed > capacity {
                return false;
            }
        }
        if pad != 0 {
            self.shared.slots[offset as usize].store(WRAP, Ordering::Relaxed);
            self.write += pad;
        }
        true
    }

    /// Writes the next slot of a reserved record.
    #[inline]
    pub(crate) fn push(&mut self, value: u64) {
        self.shared.slots[(self.write & self.shared.mask) as usize].store(value, Ordering::Relaxed);
        self.write += 1;
    }

    /// Makes the records written so far visible to the consumer.
    #[inline]
    pub(crate) fn publish(&mut self) {
        if self.write != self.published {
            self.published = self.write;
            self.shared.published.0.store(self.write, Ordering::Release);
            self.shared.more.notify();
        }
    }

    /// Whether the consumer executed everything before `position`.
    #[inline]
    pub(crate) fn consumed(&self, position: u64) -> bool {
        self.shared.consumed.0.load(Ordering::Acquire) >= position
    }

    /// Blocks until the consumer executed everything before `position` (which must be
    /// published). Panics when the consumer thread is gone.
    pub(crate) fn wait_consumed(&self, position: u64) {
        let shared = &*self.shared;
        self.wait_progress(|| {
            shared.consumed.0.load(Ordering::Acquire) >= position
                || !shared.consumer_alive.load(Ordering::Acquire)
        });
        assert!(
            self.consumed(position),
            "the GPU back thread stopped before it executed the stream"
        );
    }

    /// Blocks until the consumer freed at least one slot.
    pub(crate) fn wait_for_room(&mut self) {
        let seen = self.consumed;
        let shared = &*self.shared;
        self.wait_progress(|| {
            shared.consumed.0.load(Ordering::Acquire) != seen
                || !shared.consumer_alive.load(Ordering::Acquire)
        });
        assert!(
            shared.consumer_alive.load(Ordering::Acquire),
            "the GPU back thread stopped before it executed the stream"
        );
    }

    /// Waits on the consumer's progress until `ready` holds, and reports a consumer that has
    /// executed one record for [`LONG_WAIT`]: the front then waits with it, and every channel
    /// stops.
    fn wait_progress(&self, mut ready: impl FnMut() -> bool) {
        let shared = &*self.shared;
        if shared
            .progress
            .wait_until_deadline(&mut ready, Some(Instant::now() + LONG_WAIT))
        {
            return;
        }
        let header = shared.executing.0.load(Ordering::Relaxed);
        tracing::warn!(
            record = header as u8,
            engine = (header >> 8) as u8,
            method = format_args!("{:#x}", (header >> 16) & 0xffff),
            argument = format_args!("{:#x}", header >> 32),
            published = shared.published.0.load(Ordering::Relaxed),
            consumed = shared.consumed.0.load(Ordering::Relaxed),
            "the GPU front has waited 2 s for the back; this is the record the back executes"
        );
        shared.progress.wait_until(ready);
    }

    /// Publishes and tells the consumer that nothing more comes.
    pub(crate) fn close(&mut self) {
        self.publish();
        self.shared.closed.store(true, Ordering::Release);
        self.shared.more.notify();
    }
}

/// The reading end, owned by the back.
pub(crate) struct Consumer {
    shared: Arc<Shared>,
    read: u64,
    end: u64,
    released: u64,
}

impl Consumer {
    /// Picks up what the producer published; false when there is nothing new.
    #[inline]
    pub(crate) fn refill(&mut self) -> bool {
        self.end = self.shared.published.0.load(Ordering::Acquire);
        self.read != self.end
    }

    /// Whether published slots are left to read.
    #[inline]
    pub(crate) fn has_slots(&self) -> bool {
        self.read != self.end
    }

    /// Slot `index` of the record at the read position.
    #[inline]
    pub(crate) fn slot(&self, index: usize) -> u64 {
        self.shared.slots[((self.read + index as u64) & self.shared.mask) as usize]
            .load(Ordering::Relaxed)
    }

    /// Notes the header of the record about to execute, which a producer waiting on a stalled
    /// consumer reports.
    #[inline]
    pub(crate) fn begin(&self, header: u64) {
        self.shared.executing.0.store(header, Ordering::Relaxed);
    }

    /// Moves past a record of `len` slots.
    #[inline]
    pub(crate) fn advance(&mut self, len: usize) {
        self.read += len as u64;
    }

    /// Moves past a [`WRAP`] slot to the ring's start.
    #[inline]
    pub(crate) fn wrap(&mut self) {
        self.read = (self.read | self.shared.mask) + 1;
    }

    /// Slots read since the last release.
    #[inline]
    pub(crate) fn unreleased(&self) -> u64 {
        self.read - self.released
    }

    /// Tells the producer everything read so far was executed.
    #[inline]
    pub(crate) fn release(&mut self) {
        if self.read != self.released {
            self.released = self.read;
            self.shared.consumed.0.store(self.read, Ordering::Release);
            self.shared.progress.notify();
        }
    }

    /// Sleeps until more is published or the producer closed the ring; false once it is closed
    /// and everything was read.
    pub(crate) fn wait_more(&mut self) -> bool {
        self.wait_more_until(None)
            .expect("a wait without a deadline returns once there is more")
    }

    /// Like [`Consumer::wait_more`], but gives up after `timeout` and returns `None`.
    pub(crate) fn wait_more_for(&mut self, timeout: Duration) -> Option<bool> {
        self.wait_more_until(Some(Instant::now() + timeout))
    }

    fn wait_more_until(&mut self, deadline: Option<Instant>) -> Option<bool> {
        let shared = &*self.shared;
        let read = self.read;
        let more = shared.more.wait_until_deadline(
            || {
                shared.published.0.load(Ordering::Acquire) != read
                    || shared.closed.load(Ordering::Acquire)
            },
            deadline,
        );
        if !more {
            return None;
        }
        let closed = shared.closed.load(Ordering::Acquire);
        Some(self.refill() || !closed)
    }

    /// A guard that marks the consumer gone when dropped (its thread ended or panicked), so a
    /// waiting producer stops waiting.
    pub(crate) fn liveness(&self) -> ConsumerLiveness {
        ConsumerLiveness(self.shared.clone())
    }
}

pub(crate) struct ConsumerLiveness(Arc<Shared>);

impl Drop for ConsumerLiveness {
    fn drop(&mut self) {
        self.0.consumer_alive.store(false, Ordering::Release);
        self.0.progress.notify();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(producer: &mut Producer, record: &[u64]) -> bool {
        if !producer.try_reserve(record.len()) {
            return false;
        }
        for &slot in record {
            producer.push(slot);
        }
        true
    }

    fn read_all(consumer: &mut Consumer, out: &mut Vec<u64>) {
        consumer.refill();
        while consumer.has_slots() {
            let slot = consumer.slot(0);
            if slot == WRAP {
                consumer.wrap();
                continue;
            }
            let len = (slot >> 32) as usize;
            out.extend((0..len).map(|i| consumer.slot(i)));
            consumer.advance(len);
        }
        consumer.release();
    }

    #[test]
    fn records_wrap_whole_and_full_ring_refuses() {
        let (mut producer, mut consumer) = ring(16, 16);
        let record = |len: u64, tag: u64| -> Vec<u64> {
            (0..len)
                .map(|i| {
                    if i == 0 {
                        len << 32 | tag
                    } else {
                        tag << 8 | i
                    }
                })
                .collect()
        };
        assert!(push(&mut producer, &record(6, 1)));
        assert!(push(&mut producer, &record(6, 2)));
        assert!(!push(&mut producer, &record(6, 3)));
        producer.publish();
        let mut out = Vec::new();
        read_all(&mut consumer, &mut out);
        assert!(push(&mut producer, &record(6, 3)));
        producer.publish();
        read_all(&mut consumer, &mut out);
        let expected: Vec<u64> = [1, 2, 3]
            .into_iter()
            .flat_map(|tag| record(6, tag))
            .collect();
        assert_eq!(out, expected);
        assert_eq!(producer.position(), 16 + 6);
        assert!(producer.consumed(22));
    }

    #[test]
    fn a_timed_wait_for_more_gives_up_or_sees_the_next_record() {
        let (mut producer, mut consumer) = ring(16, 16);
        assert_eq!(consumer.wait_more_for(Duration::from_millis(5)), None);
        assert!(push(&mut producer, &[1 << 32 | 7]));
        producer.publish();
        assert_eq!(consumer.wait_more_for(Duration::from_secs(5)), Some(true));
        let mut out = Vec::new();
        read_all(&mut consumer, &mut out);
        assert_eq!(out, [1 << 32 | 7]);
        producer.close();
        assert_eq!(consumer.wait_more_for(Duration::from_secs(5)), Some(false));
    }

    #[test]
    fn waits_cross_threads() {
        let (mut producer, mut consumer) = ring(64, 64);
        let reader = std::thread::spawn(move || {
            let mut out = Vec::new();
            while consumer.wait_more() {
                read_all(&mut consumer, &mut out);
            }
            out.len()
        });
        let mut written = 0;
        for i in 0..10_000u64 {
            while !push(&mut producer, &[1 << 32 | (i & 0xff) | 0x100]) {
                producer.publish();
                producer.wait_for_room();
            }
            written += 1;
        }
        producer.publish();
        producer.wait_consumed(producer.position());
        producer.close();
        assert_eq!(reader.join().unwrap(), written);
    }
}
