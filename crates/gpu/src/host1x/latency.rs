//! How long guest fences take, split by where the time goes. A submission's increment fence is
//! stamped when the guest submits it, when the front reaches its increments, when the back
//! executes the last of them (the host value, which the Switch 2 syncpoint page shows the guest),
//! and when the host GPU finished the work before it (the guest value). The activity reporter
//! logs the totals at debug level on `lichii_gpu::activity`; none of it exists unless that is
//! enabled when the syncpoints are created.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::channel::SyncpointFence;

const SLOTS: usize = 256;
const PROBE: usize = 8;
const EMPTY: u64 = 0;

/// The bank whose increment stamps a fence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    /// The host bank: the back executed the increment.
    Back,
    /// The guest bank: the host GPU finished the work before it.
    Done,
}

/// A fence in flight; a stamp of 0 has not happened.
struct Pending {
    key: AtomicU64,
    submitted: AtomicU64,
    front: AtomicU64,
    back: AtomicU64,
}

/// Fences completed since the last [`FenceLatency::take`], and where their time went. The three
/// parts of every fence counted in `fences` add up to its total.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FenceTotals {
    pub fences: u64,
    /// Completed without a front or back stamp, or with stamps out of order; not timed.
    pub partial: u64,
    /// Submitted while every slot near its key was taken; not tracked.
    pub untracked: u64,
    pub submit_to_front_ns: u64,
    pub front_to_back_ns: u64,
    pub back_to_done_ns: u64,
    pub total_ns: u64,
    pub longest_ns: u64,
}

/// Open-addressed fences in flight keyed by syncpoint id and value, so each stamp is a few
/// atomic operations and allocates nothing. A fence that completes adds to `totals` in one
/// step, so a report never splits it across periods.
pub(crate) struct FenceLatency {
    base: Instant,
    pending: Box<[Pending]>,
    totals: Mutex<FenceTotals>,
}

fn key(id: u32, value: u32) -> u64 {
    (u64::from(id) + 1) << 32 | u64::from(value)
}

fn slot(key: u64) -> usize {
    (key.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 56) as usize % SLOTS
}

impl FenceLatency {
    /// A tracker if `lichii_gpu::activity` logs at debug level.
    pub(crate) fn if_enabled() -> Option<Self> {
        tracing::enabled!(target: "lichii_gpu::activity", tracing::Level::DEBUG).then(Self::new)
    }

    pub(crate) fn new() -> Self {
        Self {
            base: Instant::now(),
            pending: (0..SLOTS)
                .map(|_| Pending {
                    key: AtomicU64::new(EMPTY),
                    submitted: AtomicU64::new(0),
                    front: AtomicU64::new(0),
                    back: AtomicU64::new(0),
                })
                .collect(),
            totals: Mutex::default(),
        }
    }

    fn now(&self) -> u64 {
        u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(u64::MAX - 1) + 1
    }

    fn find(&self, key: u64) -> Option<&Pending> {
        let start = slot(key);
        (0..PROBE)
            .map(|i| &self.pending[(start + i) % SLOTS])
            .find(|pending| pending.key.load(Ordering::Acquire) == key)
    }

    /// The guest submitted work whose increments reach `fence`.
    pub(crate) fn submitted(&self, fence: SyncpointFence) {
        let key = key(fence.id, fence.value);
        let start = slot(key);
        for i in 0..PROBE {
            let pending = &self.pending[(start + i) % SLOTS];
            if pending
                .key
                .compare_exchange(EMPTY, key, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                pending.front.store(0, Ordering::Relaxed);
                pending.back.store(0, Ordering::Relaxed);
                pending.submitted.store(self.now(), Ordering::Release);
                return;
            }
        }
        self.totals.lock().untracked += 1;
    }

    /// The front reached the increments of `fence`.
    pub(crate) fn front(&self, fence: SyncpointFence) {
        if let Some(pending) = self.find(key(fence.id, fence.value)) {
            pending.front.store(self.now(), Ordering::Release);
        }
    }

    /// Syncpoint `id` of the `stage` bank reached `value`.
    pub(crate) fn reached(&self, stage: Stage, id: u32, value: u32) {
        let key = key(id, value);
        let Some(pending) = self.find(key) else {
            return;
        };
        let now = self.now();
        if stage == Stage::Back {
            pending.back.store(now, Ordering::Release);
            return;
        }
        let submitted = pending.submitted.load(Ordering::Acquire);
        let front = pending.front.load(Ordering::Acquire);
        let back = pending.back.load(Ordering::Acquire);
        if pending
            .key
            .compare_exchange(key, EMPTY, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        if front == 0 || back == 0 || !(submitted <= front && front <= back && back <= now) {
            self.totals.lock().partial += 1;
            return;
        }
        let total = now - submitted;
        let mut totals = self.totals.lock();
        totals.fences += 1;
        totals.submit_to_front_ns += front - submitted;
        totals.front_to_back_ns += back - front;
        totals.back_to_done_ns += now - back;
        totals.total_ns += total;
        totals.longest_ns = totals.longest_ns.max(total);
    }

    /// Takes the totals since the last call.
    pub(crate) fn take(&self) -> FenceTotals {
        std::mem::take(&mut *self.totals.lock())
    }

    /// Logs the averages per fence since the last report.
    pub(crate) fn report(&self, period: Duration) {
        if !tracing::enabled!(target: "lichii_gpu::activity", tracing::Level::DEBUG) {
            return;
        }
        let totals = self.take();
        let average = |ns: u64| {
            let ms = ns as f64 / totals.fences.max(1) as f64 / 1e6;
            (ms * 100.0).round() / 100.0
        };
        tracing::debug!(
            target: "lichii_gpu::activity",
            ?period,
            fences = totals.fences,
            partial = totals.partial,
            untracked = totals.untracked,
            submit_to_front_ms = average(totals.submit_to_front_ns),
            front_to_back_ms = average(totals.front_to_back_ns),
            back_to_done_ms = average(totals.back_to_done_ns),
            total_ms = average(totals.total_ns),
            longest_ms = (totals.longest_ns as f64 / 1e4).round() / 100.0,
            "GPU fence latency"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FENCE: SyncpointFence = SyncpointFence { id: 7, value: 2 };

    #[test]
    fn a_fence_splits_its_total_into_three_parts() {
        let latency = FenceLatency::new();
        latency.submitted(FENCE);
        std::thread::sleep(Duration::from_millis(2));
        latency.front(FENCE);
        latency.reached(Stage::Back, FENCE.id, 1);
        std::thread::sleep(Duration::from_millis(2));
        latency.reached(Stage::Back, FENCE.id, FENCE.value);
        latency.reached(Stage::Done, FENCE.id, 1);
        std::thread::sleep(Duration::from_millis(2));
        latency.reached(Stage::Done, FENCE.id, FENCE.value);
        let totals = latency.take();
        assert_eq!((totals.fences, totals.partial, totals.untracked), (1, 0, 0));
        assert_eq!(
            totals.submit_to_front_ns + totals.front_to_back_ns + totals.back_to_done_ns,
            totals.total_ns
        );
        for part in [
            totals.submit_to_front_ns,
            totals.front_to_back_ns,
            totals.back_to_done_ns,
        ] {
            assert!(part >= 2_000_000, "{totals:?}");
        }
        assert_eq!(totals.longest_ns, totals.total_ns);
        assert_eq!(latency.take(), FenceTotals::default());
    }

    #[test]
    fn a_report_during_completions_never_splits_a_fence() {
        let latency = std::sync::Arc::new(FenceLatency::new());
        let fences = 20_000u32;
        let completer = {
            let latency = latency.clone();
            std::thread::spawn(move || {
                for value in 1..=fences {
                    let fence = SyncpointFence { id: 3, value };
                    latency.submitted(fence);
                    latency.front(fence);
                    latency.reached(Stage::Back, fence.id, value);
                    latency.reached(Stage::Done, fence.id, value);
                }
            })
        };
        let mut seen = FenceTotals::default();
        let mut check = |totals: FenceTotals| {
            assert_eq!(
                totals.submit_to_front_ns + totals.front_to_back_ns + totals.back_to_done_ns,
                totals.total_ns
            );
            assert_eq!(totals.fences == 0, totals.total_ns == 0, "{totals:?}");
            seen.fences += totals.fences;
            seen.partial += totals.partial;
            seen.untracked += totals.untracked;
        };
        while !completer.is_finished() {
            check(latency.take());
        }
        completer.join().unwrap();
        check(latency.take());
        assert_eq!(
            (seen.fences, seen.partial, seen.untracked),
            (u64::from(fences), 0, 0)
        );
    }

    #[test]
    fn a_fence_the_front_never_reached_is_partial() {
        let latency = FenceLatency::new();
        latency.submitted(FENCE);
        latency.reached(Stage::Back, FENCE.id, FENCE.value);
        latency.reached(Stage::Done, FENCE.id, FENCE.value);
        latency.reached(Stage::Done, FENCE.id, FENCE.value);
        let totals = latency.take();
        assert_eq!((totals.fences, totals.partial, totals.total_ns), (0, 1, 0));
    }

    #[test]
    fn fences_past_a_full_probe_run_are_untracked() {
        let latency = FenceLatency::new();
        let tracked = (0..SLOTS as u32 * 2)
            .map(|value| SyncpointFence { id: 1, value })
            .inspect(|&fence| latency.submitted(fence))
            .filter(|fence| latency.find(key(fence.id, fence.value)).is_some())
            .count() as u64;
        assert!(tracked <= SLOTS as u64);
        assert_eq!(tracked + latency.take().untracked, SLOTS as u64 * 2);
    }
}
