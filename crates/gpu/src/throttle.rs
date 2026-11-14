//! Rate limits for warnings that a title can trigger thousands of times a second.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// Lets a repeating warning through at most once per [`Throttle::PERIOD_MS`] and counts the
/// rest. Lock-free and allocation-free, so hot paths can use it.
#[derive(Default)]
pub struct Throttle {
    next: AtomicU64,
    dropped: AtomicU64,
}

impl Throttle {
    pub const PERIOD_MS: u64 = 5_000;

    pub const fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// `Some(n)` when this occurrence should be logged, `n` being how many were dropped since
    /// the last logged one; `None` when it is dropped.
    pub fn allow(&self) -> Option<u64> {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        self.allow_at(EPOCH.get_or_init(Instant::now).elapsed().as_millis() as u64)
    }

    fn allow_at(&self, now_ms: u64) -> Option<u64> {
        let next = self.next.load(Relaxed);
        if now_ms < next
            || self
                .next
                .compare_exchange(next, now_ms + Self::PERIOD_MS, Relaxed, Relaxed)
                .is_err()
        {
            self.dropped.fetch_add(1, Relaxed);
            return None;
        }
        Some(self.dropped.swap(0, Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_once_per_period_and_counts_the_dropped() {
        let throttle = Throttle::new();
        assert_eq!(throttle.allow_at(0), Some(0));
        for t in [1, 100, Throttle::PERIOD_MS - 1] {
            assert_eq!(throttle.allow_at(t), None, "at {t} ms");
        }
        assert_eq!(throttle.allow_at(Throttle::PERIOD_MS), Some(3));
        assert_eq!(throttle.allow_at(Throttle::PERIOD_MS + 1), None);
        assert_eq!(throttle.allow_at(3 * Throttle::PERIOD_MS), Some(1));
    }
}
