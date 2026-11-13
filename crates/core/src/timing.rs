//! Guest time for one session: the CPU's system counter, frozen while the session is paused.

use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::time::Instant;

use lichii_common::ConsoleGeneration;
use lichii_cpu::GuestClock;
use lichii_gpu::clock::GuestTime;
use parking_lot::Mutex;

const RUNNING: u64 = u64::MAX;

/// Nanoseconds of guest time since boot, counted from the host clock minus the time spent paused.
///
/// Reads take no lock: `pause` and `resume` publish their changes under a sequence counter
/// that is odd while they write, and a read that overlaps a change retries.
pub struct CoreTiming {
    start: Instant,
    counter_hz: u64,
    sequence: AtomicU64,
    paused_ns: AtomicU64,
    frozen_at: AtomicU64,
    writer: Mutex<()>,
}

impl CoreTiming {
    /// Starts guest time at zero, running, with the system counter (`CNTPCT_EL0`) at the rate the
    /// title's SDK converts ticks with: 19.2 MHz for Switch 1 titles, 31.25 MHz for Switch 2
    /// titles (at 19.2 MHz every tick-based time of a Switch 2 title is off by 1.63x).
    pub fn new(title: ConsoleGeneration) -> Self {
        Self {
            start: Instant::now(),
            counter_hz: match title {
                ConsoleGeneration::Switch1 => 19_200_000,
                ConsoleGeneration::Switch2 => 31_250_000,
            },
            sequence: AtomicU64::new(0),
            paused_ns: AtomicU64::new(0),
            frozen_at: AtomicU64::new(RUNNING),
            writer: Mutex::new(()),
        }
    }

    fn host_ns(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// Guest nanoseconds since boot.
    pub fn global_time_ns(&self) -> u64 {
        loop {
            let before = self.sequence.load(Ordering::Acquire);
            if before & 1 == 0 {
                let frozen = self.frozen_at.load(Ordering::Acquire);
                let paused = self.paused_ns.load(Ordering::Acquire);
                let now = if frozen == RUNNING {
                    self.host_ns().saturating_sub(paused)
                } else {
                    frozen
                };
                fence(Ordering::Acquire);
                if self.sequence.load(Ordering::Relaxed) == before {
                    return now;
                }
            }
            std::hint::spin_loop();
        }
    }

    fn update(&self, change: impl FnOnce()) {
        let _writer = self.writer.lock();
        self.sequence.fetch_add(1, Ordering::AcqRel);
        fence(Ordering::Release);
        change();
        self.sequence.fetch_add(1, Ordering::Release);
    }

    /// Stops guest time until [`CoreTiming::resume`].
    pub fn pause(&self) {
        self.update(|| {
            if self.frozen_at.load(Ordering::Relaxed) == RUNNING {
                let now = self.host_ns() - self.paused_ns.load(Ordering::Relaxed);
                self.frozen_at.store(now, Ordering::Relaxed);
            }
        });
    }

    /// Restarts guest time from where [`CoreTiming::pause`] froze it.
    pub fn resume(&self) {
        self.update(|| {
            let frozen = self.frozen_at.load(Ordering::Relaxed);
            if frozen != RUNNING {
                self.paused_ns
                    .store(self.host_ns() - frozen, Ordering::Relaxed);
                self.frozen_at.store(RUNNING, Ordering::Relaxed);
            }
        });
    }
}

impl GuestTime for CoreTiming {
    fn global_time_ns(&self) -> u64 {
        CoreTiming::global_time_ns(self)
    }
}

impl GuestClock for CoreTiming {
    fn ticks(&self) -> u64 {
        (u128::from(self.global_time_ns()) * u128::from(self.counter_hz) / 1_000_000_000) as u64
    }

    fn frequency(&self) -> u64 {
        self.counter_hz
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn time_stops_while_paused() {
        let timing = CoreTiming::new(ConsoleGeneration::Switch1);
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

    #[test]
    fn the_counter_rate_follows_the_title() {
        let switch1 = CoreTiming::new(ConsoleGeneration::Switch1);
        let switch2 = CoreTiming::new(ConsoleGeneration::Switch2);
        assert_eq!(switch1.frequency(), 19_200_000);
        assert_eq!(switch2.frequency(), 31_250_000);
        switch2.pause();
        assert_eq!(switch2.ticks(), switch2.global_time_ns() / 32);
    }

    #[test]
    fn reads_stay_monotonic_across_pause_and_resume() {
        let timing = std::sync::Arc::new(CoreTiming::new(ConsoleGeneration::Switch1));
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let readers: Vec<_> = (0..3)
            .map(|_| {
                let (timing, stop) = (timing.clone(), stop.clone());
                std::thread::spawn(move || {
                    let mut last = 0;
                    while !stop.load(Ordering::Relaxed) {
                        let now = timing.global_time_ns();
                        assert!(now >= last, "guest time went back from {last} to {now}");
                        last = now;
                    }
                })
            })
            .collect();
        for _ in 0..2000 {
            timing.pause();
            std::hint::spin_loop();
            timing.resume();
        }
        stop.store(true, Ordering::Relaxed);
        for reader in readers {
            reader.join().unwrap();
        }
    }
}
