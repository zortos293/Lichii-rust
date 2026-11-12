//! Guest time for one session: the CPU's system counter, frozen while the session is paused.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use lichii_cpu::GuestClock;
use lichii_gpu::clock::GuestTime;

const RUNNING: u64 = u64::MAX;
const COUNTER_HZ: u128 = 19_200_000;

/// Nanoseconds of guest time since boot, counted from the host clock minus the time spent paused.
pub struct CoreTiming {
    start: Instant,
    paused_ns: AtomicU64,
    frozen_at: AtomicU64,
}

impl CoreTiming {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            paused_ns: AtomicU64::new(0),
            frozen_at: AtomicU64::new(RUNNING),
        }
    }

    fn host_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Guest nanoseconds since boot.
    pub fn global_time_ns(&self) -> u64 {
        match self.frozen_at.load(Ordering::Acquire) {
            RUNNING => self.host_ns() - self.paused_ns.load(Ordering::Acquire),
            frozen => frozen,
        }
    }

    /// Stops guest time until [`CoreTiming::resume`].
    pub fn pause(&self) {
        let now = self.global_time_ns();
        let _ = self
            .frozen_at
            .compare_exchange(RUNNING, now, Ordering::AcqRel, Ordering::Acquire);
    }

    pub fn resume(&self) {
        let frozen = self.frozen_at.load(Ordering::Acquire);
        if frozen == RUNNING {
            return;
        }
        self.paused_ns
            .store(self.host_ns() - frozen, Ordering::Release);
        self.frozen_at.store(RUNNING, Ordering::Release);
    }
}

impl Default for CoreTiming {
    fn default() -> Self {
        Self::new()
    }
}

impl GuestTime for CoreTiming {
    fn global_time_ns(&self) -> u64 {
        CoreTiming::global_time_ns(self)
    }
}

impl GuestClock for CoreTiming {
    fn ticks(&self) -> u64 {
        (u128::from(self.global_time_ns()) * COUNTER_HZ / 1_000_000_000) as u64
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn time_stops_while_paused() {
        let timing = CoreTiming::new();
        std::thread::sleep(Duration::from_millis(5));
        let before = timing.global_time_ns();
        assert!(before >= 5_000_000);
        timing.pause();
        let frozen = timing.global_time_ns();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(timing.global_time_ns(), frozen);
        assert_eq!(timing.ticks(), frozen * 12 / 625);
        timing.resume();
        let resumed = timing.global_time_ns();
        assert!(resumed >= frozen && resumed < frozen + 10_000_000);
        std::thread::sleep(Duration::from_millis(5));
        assert!(timing.global_time_ns() > resumed);
    }
}
