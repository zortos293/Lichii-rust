//! Guest time for one session: the CPU's system counter, frozen while the session is paused.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::time::{Duration, Instant};

use lichii_common::ConsoleGeneration;
use lichii_cpu::{GuestClock, TscScale};
use lichii_gpu::clock::GuestTime;
use parking_lot::Mutex;

const RUNNING: u64 = u64::MAX;
const SHIFT: u32 = 63;

/// Guest time counted from a host TSC the OS keeps synchronized across cores, so the JIT can
/// read the counter inline.
struct TscClock {
    start: u64,
    ns_mul: u64,
    tick_mul: u64,
    scale: TscScale,
}

impl TscClock {
    fn new(counter_hz: u64, tsc_hz: u64) -> Option<Self> {
        let tsc_hz = u128::from(tsc_hz);
        let ns_mul = u64::try_from((1_000_000_000u128 << SHIFT) / tsc_hz).ok()?;
        let tick_mul = u64::try_from((u128::from(counter_hz) << SHIFT) / tsc_hz).ok()?;
        let start = read_tsc()?;
        let offset = 0u64.wrapping_sub(scaled(start, tick_mul));
        Some(Self {
            start,
            ns_mul,
            tick_mul,
            scale: TscScale::new(tick_mul, SHIFT, offset),
        })
    }

    fn host_ns(&self) -> u64 {
        let tsc = read_tsc().unwrap_or(self.start);
        scaled(tsc.saturating_sub(self.start), self.ns_mul)
    }
}

fn scaled(tsc: u64, mul: u64) -> u64 {
    ((u128::from(tsc) * u128::from(mul)) >> SHIFT) as u64
}

/// The host TSC frequency, measured once per process; `None` unless the TSC is invariant and the
/// OS itself keeps time with it, which is its guarantee that the TSC agrees across cores.
fn trusted_tsc_hz() -> Option<u64> {
    static HZ: OnceLock<Option<u64>> = OnceLock::new();
    *HZ.get_or_init(|| {
        if tsc_is_invariant() && os_clock_is_tsc() {
            measure_tsc_hz()
        } else {
            None
        }
    })
}

#[cfg(target_os = "linux")]
fn os_clock_is_tsc() -> bool {
    std::fs::read_to_string("/sys/devices/system/clocksource/clocksource0/current_clocksource")
        .is_ok_and(|source| source.trim() == "tsc")
}

#[cfg(not(target_os = "linux"))]
fn os_clock_is_tsc() -> bool {
    false
}

/// The TSC frequency against [`Instant`] over 20 ms. Each endpoint is an `Instant` read between
/// two TSC reads at most 100 000 cycles apart, so a preempted sample is retried instead of skewing
/// the rate.
fn measure_tsc_hz() -> Option<u64> {
    let sample = || {
        (0..100).find_map(|_| {
            let before = read_tsc()?;
            let now = Instant::now();
            let after = read_tsc()?;
            (after - before < 100_000).then(|| (now, before + (after - before) / 2))
        })
    };
    let (start, start_tsc) = sample()?;
    std::thread::sleep(Duration::from_millis(20));
    let (end, end_tsc) = sample()?;
    let ns = end.duration_since(start).as_nanos();
    u64::try_from(u128::from(end_tsc - start_tsc) * 1_000_000_000 / ns).ok()
}

#[cfg(target_arch = "x86_64")]
fn tsc_is_invariant() -> bool {
    use std::arch::x86_64::__cpuid;
    __cpuid(0x8000_0000).eax >= 0x8000_0007 && __cpuid(0x8000_0007).edx & (1 << 8) != 0
}

#[cfg(not(target_arch = "x86_64"))]
fn tsc_is_invariant() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn read_tsc() -> Option<u64> {
    use std::arch::x86_64::{_mm_lfence, _rdtsc};
    // SAFETY: `lfence` and `rdtsc` exist on every x86_64 CPU and touch no memory.
    Some(unsafe {
        _mm_lfence();
        _rdtsc()
    })
}

#[cfg(not(target_arch = "x86_64"))]
fn read_tsc() -> Option<u64> {
    None
}

/// Nanoseconds of guest time since boot, counted from the host clock minus the time spent paused.
///
/// Reads take no lock: `pause` and `resume` publish their changes under a sequence counter
/// that is odd while they write, and a read that overlaps a change retries. When the OS keeps time
/// with an invariant TSC, the host clock is the TSC, so the JIT's inline counter reads through
/// [`GuestClock::host_tsc_scale`] agree with [`GuestClock::ticks`].
pub struct CoreTiming {
    start: Instant,
    tsc: Option<TscClock>,
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
        Self::with_tsc_hz(title, trusted_tsc_hz())
    }

    fn with_tsc_hz(title: ConsoleGeneration, tsc_hz: Option<u64>) -> Self {
        let counter_hz = match title {
            ConsoleGeneration::Switch1 => 19_200_000,
            ConsoleGeneration::Switch2 => 31_250_000,
        };
        Self {
            start: Instant::now(),
            tsc: tsc_hz.and_then(|hz| TscClock::new(counter_hz, hz)),
            counter_hz,
            sequence: AtomicU64::new(0),
            paused_ns: AtomicU64::new(0),
            frozen_at: AtomicU64::new(RUNNING),
            writer: Mutex::new(()),
        }
    }

    fn host_ns(&self) -> u64 {
        match &self.tsc {
            Some(tsc) => tsc.host_ns(),
            None => self.start.elapsed().as_nanos() as u64,
        }
    }

    fn ticks_at(&self, ns: u64) -> u64 {
        (u128::from(ns) * u128::from(self.counter_hz) / 1_000_000_000) as u64
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

    /// Stops guest time until [`CoreTiming::resume`]. It leaves the TSC scale alone, so a core
    /// that has not stopped yet keeps reading a running counter until it does.
    pub fn pause(&self) {
        self.update(|| {
            if self.frozen_at.load(Ordering::Relaxed) == RUNNING {
                let now = self.host_ns() - self.paused_ns.load(Ordering::Relaxed);
                self.frozen_at.store(now, Ordering::Relaxed);
            }
        });
    }

    /// Restarts guest time from where [`CoreTiming::pause`] froze it.
    ///
    /// The TSC scale stays running while paused, because no core runs guest code then. Resuming
    /// moves only its offset, with one store, so a core reads either the old or the new mapping,
    /// never a mix of fields. A core that read the counter after `pause` but before it stopped saw
    /// it run past the frozen value, so its counter steps back by at most that stop window.
    pub fn resume(&self) {
        self.update(|| {
            let frozen = self.frozen_at.load(Ordering::Relaxed);
            if frozen != RUNNING {
                match &self.tsc {
                    Some(tsc) => {
                        let now = read_tsc().unwrap_or(tsc.start);
                        self.paused_ns.store(
                            scaled(now.saturating_sub(tsc.start), tsc.ns_mul) - frozen,
                            Ordering::Relaxed,
                        );
                        let offset = self
                            .ticks_at(frozen)
                            .wrapping_sub(scaled(now, tsc.tick_mul));
                        tsc.scale.offset.store(offset, Ordering::Relaxed);
                    }
                    None => self
                        .paused_ns
                        .store(self.host_ns() - frozen, Ordering::Relaxed),
                }
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
        self.ticks_at(self.global_time_ns())
    }

    fn frequency(&self) -> u64 {
        self.counter_hz
    }

    fn host_tsc_scale(&self) -> Option<&TscScale> {
        self.tsc.as_ref().map(|tsc| &tsc.scale)
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
    fn the_tsc_scale_matches_the_counter_across_pause_and_resume() {
        let Some(tsc_hz) = tsc_is_invariant().then(measure_tsc_hz).flatten() else {
            return;
        };
        let timing = CoreTiming::with_tsc_hz(ConsoleGeneration::Switch1, Some(tsc_hz));
        let scale = timing.host_tsc_scale().unwrap();
        let mul = scale.mul.load(Ordering::Relaxed);
        let agrees = |timing: &CoreTiming| {
            let before = timing.ticks();
            let inline = scale.ticks(read_tsc().unwrap());
            let after = timing.ticks();
            assert!(
                before <= inline + 1 && inline <= after + 1,
                "inline {inline} outside {before}..={after}"
            );
            inline
        };
        std::thread::sleep(Duration::from_millis(5));
        let running = agrees(&timing);
        assert!(running >= 5 * 19_200);
        timing.pause();
        let frozen = timing.ticks();
        assert!(frozen + 1 >= running);
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(timing.ticks(), frozen);
        timing.resume();
        assert_eq!(scale.mul.load(Ordering::Relaxed), mul);
        let resumed = agrees(&timing);
        assert!(resumed + 1 >= frozen && resumed < frozen + 19_200);
        std::thread::sleep(Duration::from_millis(30));
        assert!(agrees(&timing) > resumed + 25 * 19_200);
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

    #[test]
    fn probe_monotonic_paths() {
        fn run(timing: CoreTiming) -> (u64, u64) {
            let timing = std::sync::Arc::new(timing);
            let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let readers: Vec<_> = (0..3)
                .map(|_| {
                    let (timing, stop) = (timing.clone(), stop.clone());
                    std::thread::spawn(move || {
                        let (mut last, mut backs, mut worst) = (0, 0, 0);
                        while !stop.load(Ordering::Relaxed) {
                            let now = timing.global_time_ns();
                            if now < last {
                                backs += 1;
                                worst = worst.max(last - now);
                            }
                            last = last.max(now);
                        }
                        (backs, worst)
                    })
                })
                .collect();
            for _ in 0..2000 {
                timing.pause();
                std::hint::spin_loop();
                timing.resume();
            }
            stop.store(true, Ordering::Relaxed);
            readers
                .into_iter()
                .map(|r| r.join().unwrap())
                .fold((0, 0), |a, b| (a.0 + b.0, a.1.max(b.1)))
        }
        let hz = measure_tsc_hz();
        let mut tsc = (0, 0, 0);
        let mut instant = (0, 0, 0);
        for _ in 0..100 {
            let (b, w) = run(CoreTiming::with_tsc_hz(ConsoleGeneration::Switch1, hz));
            tsc = (tsc.0 + (b > 0) as u64, tsc.1 + b, tsc.2.max(w));
            let (b, w) = run(CoreTiming::with_tsc_hz(ConsoleGeneration::Switch1, None));
            instant = (instant.0 + (b > 0) as u64, instant.1 + b, instant.2.max(w));
        }
        panic!(
            "PROBE2 hz={hz:?} trusted={:?} tsc(runs,backs,worst_ns)={tsc:?} instant={instant:?}",
            trusted_tsc_hz()
        );
    }
}
