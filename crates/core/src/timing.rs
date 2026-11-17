//! Guest time for one session: the CPU's system counter, frozen while the session is paused.

use std::cell::Cell;
use std::sync::Barrier;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};
use std::time::{Duration, Instant};

use lichii_common::ConsoleGeneration;
use lichii_cpu::{GuestClock, TscScale};
use lichii_gpu::clock::GuestTime;
use parking_lot::Mutex;

const RUNNING: u64 = u64::MAX;
const SHIFT: u32 = 63;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// The id of the [`CoreTiming`] this thread last read and the guest time it returned.
    static LATEST: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

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

/// The host TSC frequency, measured once per process; `None` unless the TSC is invariant,
/// `rdtscp` can read it, the OS does not keep time with another clock, and the startup calibration
/// passes. Without it, every counter read calls [`GuestClock::ticks`].
fn trusted_tsc_hz() -> Option<u64> {
    static HZ: OnceLock<Option<u64>> = OnceLock::new();
    *HZ.get_or_init(|| accepted_tsc_hz(tsc_is_usable() && os_clock_may_be_tsc(), calibrate_tsc))
}

/// What the startup calibration measured: the TSC rate against [`Instant`] over two back-to-back
/// 10 ms windows, and whether [`tsc_is_synchronized`] confirmed every processor agrees.
struct Calibration {
    rates: [u64; 2],
    synchronized: bool,
}

/// The TSC frequency to count guest time with, or `None` to fall back to [`Instant`]. The TSC
/// has to be invariant, keep one rate across both windows (within 0.1%) and be confirmed
/// synchronized across processors; `calibrate` runs only for an invariant TSC.
fn accepted_tsc_hz(
    invariant: bool,
    calibrate: impl FnOnce() -> Option<Calibration>,
) -> Option<u64> {
    if !invariant {
        return None;
    }
    let Calibration {
        rates: [first, second],
        synchronized,
    } = calibrate()?;
    (synchronized && first.abs_diff(second) <= first / 1000).then_some(first / 2 + second / 2)
}

/// Linux names the clocksource it keeps time with, and switches away from the TSC when its own
/// checks find it unsynchronized across cores.
#[cfg(target_os = "linux")]
fn os_clock_may_be_tsc() -> bool {
    std::fs::read_to_string("/sys/devices/system/clocksource/clocksource0/current_clocksource")
        .is_ok_and(|source| source.trim() == "tsc")
}

/// Windows doesn't say which clock it keeps time with, so the calibration alone decides.
#[cfg(windows)]
fn os_clock_may_be_tsc() -> bool {
    true
}

/// Elsewhere there is no way to probe every processor, so guest time stays on [`Instant`].
#[cfg(not(any(target_os = "linux", windows)))]
fn os_clock_may_be_tsc() -> bool {
    false
}

/// Each sample is an `Instant` read between two TSC reads at most 10 000 cycles apart, so a
/// preempted sample is retried instead of skewing a rate.
fn calibrate_tsc() -> Option<Calibration> {
    let sample = || {
        (0..100).find_map(|_| {
            let before = read_tsc()?;
            let now = Instant::now();
            let after = read_tsc()?;
            (after - before < 10_000).then(|| (now, before + (after - before) / 2))
        })
    };
    let rate = |(start, start_tsc): (Instant, u64), (end, end_tsc): (Instant, u64)| {
        let ns = end.duration_since(start).as_nanos();
        u64::try_from(u128::from(end_tsc - start_tsc) * 1_000_000_000 / ns).ok()
    };
    let first = sample()?;
    std::thread::sleep(Duration::from_millis(10));
    let middle = sample()?;
    std::thread::sleep(Duration::from_millis(10));
    let last = sample()?;
    Some(Calibration {
        rates: [rate(first, middle)?, rate(middle, last)?],
        synchronized: tsc_is_synchronized(),
    })
}

/// Whether the TSCs of every processor guest threads may run on agree, tested the way Linux's
/// `check_tsc_warp` does. One thread per processor, pinned to it where [`probe_processors`]
/// names one, starts behind a barrier; the threads take turns in ticket order to read the TSC,
/// and a read below the one before it means its processor's TSC lags another's. Each thread
/// takes turns for at least 10 ms and until it has had 10, for at most 50 ms. `false` too when a
/// thread could not be pinned or had fewer than 10 turns, since then its processor went
/// unchecked.
fn tsc_is_synchronized() -> bool {
    let Some(processors) = probe_processors() else {
        return false;
    };
    let (next, serving) = (AtomicU64::new(0), AtomicU64::new(0));
    let last = AtomicU64::new(0);
    let warped = AtomicBool::new(false);
    let start = Barrier::new(processors.len());
    std::thread::scope(|scope| {
        let probes: Vec<_> = processors
            .iter()
            .map(|&processor| {
                let (next, serving, last, warped, start) =
                    (&next, &serving, &last, &warped, &start);
                scope.spawn(move || {
                    let pinned = processor.is_none_or(pin_current_thread);
                    start.wait();
                    if !pinned {
                        return 0;
                    }
                    let begun = Instant::now();
                    let mut turns = 0;
                    while !warped.load(Ordering::Relaxed) {
                        let elapsed = begun.elapsed();
                        if elapsed >= Duration::from_millis(50)
                            || (elapsed >= Duration::from_millis(10) && turns >= 10)
                        {
                            break;
                        }
                        let ticket = next.fetch_add(1, Ordering::Relaxed);
                        while serving.load(Ordering::Acquire) != ticket {
                            std::hint::spin_loop();
                        }
                        let now = read_tsc().unwrap_or(0);
                        if now < last.load(Ordering::Relaxed) {
                            warped.store(true, Ordering::Relaxed);
                        }
                        last.store(now, Ordering::Relaxed);
                        serving.store(ticket + 1, Ordering::Release);
                        turns += 1;
                    }
                    turns
                })
            })
            .collect();
        let turns: Vec<u32> = probes
            .into_iter()
            .map(|probe| probe.join().unwrap_or(0))
            .collect();
        !warped.load(Ordering::Relaxed) && turns.iter().all(|&turns| turns >= 10)
    })
}

/// The processors to probe: on Linux, as many unpinned threads as it schedules this process on,
/// since its own checks already cover every processor; on Windows, each processor in the
/// process's affinity mask. `None` on a Windows host with several processor groups, where
/// threads can leave the mask's group.
#[cfg(target_os = "linux")]
fn probe_processors() -> Option<Vec<Option<usize>>> {
    let count = std::thread::available_parallelism().map_or(2, |n| n.get().max(2));
    Some(vec![None; count])
}

#[cfg(windows)]
fn probe_processors() -> Option<Vec<Option<usize>>> {
    use windows_sys::Win32::System::Threading::{
        GetActiveProcessorGroupCount, GetCurrentProcess, GetProcessAffinityMask,
    };
    let (mut process, mut system) = (0usize, 0usize);
    // SAFETY: `GetCurrentProcess` returns a pseudo-handle valid for this process, and the
    // masks are written only into the two locals.
    let read = unsafe {
        GetActiveProcessorGroupCount() == 1
            && GetProcessAffinityMask(GetCurrentProcess(), &mut process, &mut system) != 0
    };
    let processors: Vec<_> = (0..usize::BITS as usize)
        .filter(|&bit| process & (1 << bit) != 0)
        .map(Some)
        .collect();
    (read && !processors.is_empty()).then_some(processors)
}

#[cfg(not(any(target_os = "linux", windows)))]
fn probe_processors() -> Option<Vec<Option<usize>>> {
    None
}

#[cfg(windows)]
fn pin_current_thread(processor: usize) -> bool {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    // SAFETY: `GetCurrentThread` returns a pseudo-handle valid for the calling thread, a probe
    // that exits when the check ends, so its affinity changes nothing else.
    unsafe { SetThreadAffinityMask(GetCurrentThread(), 1 << processor) != 0 }
}

#[cfg(not(windows))]
fn pin_current_thread(_processor: usize) -> bool {
    false
}

#[cfg(test)]
fn measure_tsc_hz() -> Option<u64> {
    calibrate_tsc().map(|calibration| calibration.rates[0])
}

#[cfg(target_arch = "x86_64")]
fn tsc_is_usable() -> bool {
    use std::arch::x86_64::__cpuid;
    __cpuid(0x8000_0000).eax >= 0x8000_0007
        && __cpuid(0x8000_0007).edx & (1 << 8) != 0
        && __cpuid(0x8000_0001).edx & (1 << 27) != 0
}

#[cfg(not(target_arch = "x86_64"))]
fn tsc_is_usable() -> bool {
    false
}

/// The TSC, read after every earlier instruction has finished. `lfence; rdtsc` is not enough:
/// where LFENCE does not serialize dispatch (AMD without `LfenceAlwaysSerializing`, such as on
/// Azure VMs), `rdtsc` can run before the seqlock loads ahead of it, and guest time steps back.
/// `rdtscp` waits for earlier loads, and the `lfence` after it keeps later loads behind it.
#[cfg(target_arch = "x86_64")]
fn read_tsc() -> Option<u64> {
    use std::arch::x86_64::{__rdtscp, _mm_lfence};
    let mut core = 0;
    // SAFETY: callers read the TSC only where `tsc_is_usable` holds, so `rdtscp` exists; it and
    // `lfence` touch no memory but `core`.
    let tsc = unsafe {
        let tsc = __rdtscp(&mut core);
        _mm_lfence();
        tsc
    };
    #[cfg(test)]
    let tsc = tsc + tests::SKEW.get();
    Some(tsc)
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
    id: u64,
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
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
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

    /// Guest nanoseconds since boot, never less than what this thread read before. Host cores'
    /// TSCs can disagree by a few cycles, so after the thread moves to a core whose TSC lags, or
    /// after `pause` freezes time at another core's reading, the clock can sit just behind a time
    /// this thread already returned.
    pub fn global_time_ns(&self) -> u64 {
        let now = self.read_ns();
        LATEST.with(|latest| {
            let now = match latest.get() {
                (id, time) if id == self.id => now.max(time),
                _ => now,
            };
            latest.set((self.id, now));
            now
        })
    }

    fn read_ns(&self) -> u64 {
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

    thread_local! {
        /// Cycles this thread's TSC reads run ahead, standing in for a core whose TSC is skewed.
        pub(super) static SKEW: Cell<u64> = const { Cell::new(0) };
    }

    #[test]
    fn the_tsc_is_trusted_only_when_invariant_and_calibrated() {
        let calibration = |rates, synchronized| {
            move || {
                Some(Calibration {
                    rates,
                    synchronized,
                })
            }
        };
        let hz = 3_000_000_000;
        assert_eq!(
            accepted_tsc_hz(false, || panic!("calibrated a variant TSC")),
            None
        );
        assert_eq!(accepted_tsc_hz(true, || None), None);
        assert_eq!(accepted_tsc_hz(true, calibration([hz, hz], false)), None);
        assert_eq!(
            accepted_tsc_hz(true, calibration([hz, hz + hz / 999], true)),
            None
        );
        assert_eq!(
            accepted_tsc_hz(true, calibration([hz, hz + 2_000], true)),
            Some(hz + 1_000)
        );
        let timing = CoreTiming::with_tsc_hz(ConsoleGeneration::Switch2, None);
        assert!(timing.host_tsc_scale().is_none());
    }

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
        let Some(tsc_hz) = tsc_is_usable().then(measure_tsc_hz).flatten() else {
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
        assert_monotonic_across_pause_and_resume(CoreTiming::new(ConsoleGeneration::Switch1));
    }

    #[test]
    fn tsc_reads_stay_monotonic_across_pause_and_resume() {
        if let Some(hz) = tsc_is_usable().then(measure_tsc_hz).flatten() {
            let timing = CoreTiming::with_tsc_hz(ConsoleGeneration::Switch1, Some(hz));
            assert_monotonic_across_pause_and_resume(timing);
        }
    }

    #[test]
    fn tsc_reads_stay_monotonic_when_cores_disagree() {
        let Some(hz) = tsc_is_usable().then(measure_tsc_hz).flatten() else {
            return;
        };
        let timing = CoreTiming::with_tsc_hz(ConsoleGeneration::Switch1, Some(hz));
        let read_on_core = |skew| {
            SKEW.set(skew);
            timing.global_time_ns()
        };
        let ahead = read_on_core(hz / 1000);
        assert!(read_on_core(0) >= ahead);
        SKEW.set(0);
        timing.pause();
        assert!(read_on_core(hz / 1000) >= ahead);
        SKEW.set(0);
        timing.resume();
        let resumed = read_on_core(0);
        assert!(resumed >= ahead);
        assert!(read_on_core(hz / 1000) >= resumed);
    }

    fn assert_monotonic_across_pause_and_resume(timing: CoreTiming) {
        let timing = std::sync::Arc::new(timing);
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
