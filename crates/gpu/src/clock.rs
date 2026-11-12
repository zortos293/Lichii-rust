//! The GPU timestamp clock that 16-byte releases and query reports write.
//!
//! Switch 1 (Maxwell) timestamps are GPU ticks at 614.4 MHz derived from guest time. Switch 2
//! (Ampere) titles read report timestamps as nanoseconds and calibrate them against CPU time, so
//! the clock speed setting only shrinks the time since the last present: GPU durations within a
//! frame look shorter, but a timestamp never runs ahead of the CPU clock. Before the first present
//! the Ampere clock is exact. The speed latched at a present applies until the next one, so a
//! setting change never takes effect mid-frame.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

/// Guest time, as core timing keeps it.
pub trait GuestTime: Send + Sync {
    /// Nanoseconds of guest time since boot.
    fn global_time_ns(&self) -> u64;
}

/// How fast GPU work looks to the guest.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum GpuClockSpeed {
    #[default]
    Normal = 0,
    Boost = 1,
    Overclock = 2,
}

impl GpuClockSpeed {
    fn from_bits(bits: u8) -> Self {
        match bits {
            1 => Self::Boost,
            2 => Self::Overclock,
            _ => Self::Normal,
        }
    }

    const fn ampere_divisor(self) -> u64 {
        match self {
            Self::Normal => 1,
            Self::Boost => 4,
            Self::Overclock => 16,
        }
    }

    const fn maxwell_divisor(self) -> u64 {
        match self {
            Self::Normal => 1,
            Self::Boost => 256,
            Self::Overclock => 512,
        }
    }
}

/// The timestamp clock shared by the GPU and the renderer.
pub struct GpuTimestamp {
    time: Arc<dyn GuestTime>,
    ampere: bool,
    speed: AtomicU8,
    /// Guest time of the last present shifted left by 2, with the speed latched then in the low
    /// bits: one word so both change together.
    anchor: AtomicU64,
}

impl GpuTimestamp {
    /// A clock for Switch 2 titles when `ampere`, Switch 1 titles otherwise.
    pub fn new(time: Arc<dyn GuestTime>, ampere: bool) -> Self {
        Self {
            time,
            ampere,
            speed: AtomicU8::new(GpuClockSpeed::Normal as u8),
            anchor: AtomicU64::new(0),
        }
    }

    /// Changes the clock speed setting. Ampere timestamps pick it up at the next present.
    pub fn set_speed(&self, speed: GpuClockSpeed) {
        self.speed.store(speed as u8, Ordering::Relaxed);
    }

    /// A frame was presented: Ampere timestamps scale only the time after this point.
    pub fn frame_presented(&self) {
        let now = self.time.global_time_ns();
        let speed = self.speed.load(Ordering::Relaxed);
        self.anchor
            .store(now << 2 | u64::from(speed), Ordering::Relaxed);
    }

    /// The timestamp a release written now carries.
    pub fn now(&self) -> u64 {
        let now = self.time.global_time_ns();
        if self.ampere {
            let anchor = self.anchor.load(Ordering::Relaxed);
            let present = anchor >> 2;
            if present == 0 || now <= present {
                return now;
            }
            let speed = GpuClockSpeed::from_bits((anchor & 3) as u8);
            return present + (now - present) / speed.ampere_divisor();
        }
        let ticks = (u128::from(now) * 384 / 625) as u64;
        ticks / GpuClockSpeed::from_bits(self.speed.load(Ordering::Relaxed)).maxwell_divisor()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedTime(AtomicU64);

    impl GuestTime for FixedTime {
        fn global_time_ns(&self) -> u64 {
            self.0.load(Ordering::Relaxed)
        }
    }

    fn clock(ampere: bool) -> (Arc<FixedTime>, GpuTimestamp) {
        let time = Arc::new(FixedTime(AtomicU64::new(0)));
        (time.clone(), GpuTimestamp::new(time, ampere))
    }

    #[test]
    fn maxwell_ticks_at_614_4_mhz() {
        let (time, clock) = clock(false);
        time.0.store(1_000_000_000, Ordering::Relaxed);
        assert_eq!(clock.now(), 614_400_000);
        clock.set_speed(GpuClockSpeed::Boost);
        assert_eq!(clock.now(), 614_400_000 / 256);
    }

    #[test]
    fn ampere_is_exact_until_the_first_present() {
        let (time, clock) = clock(true);
        clock.set_speed(GpuClockSpeed::Boost);
        time.0.store(5_000, Ordering::Relaxed);
        assert_eq!(clock.now(), 5_000);
    }

    #[test]
    fn ampere_scales_only_time_since_the_present_with_the_latched_speed() {
        let (time, clock) = clock(true);
        clock.set_speed(GpuClockSpeed::Boost);
        time.0.store(1_000, Ordering::Relaxed);
        clock.frame_presented();
        clock.set_speed(GpuClockSpeed::Overclock);
        time.0.store(1_400, Ordering::Relaxed);
        assert_eq!(clock.now(), 1_100);
        clock.frame_presented();
        time.0.store(3_000, Ordering::Relaxed);
        assert_eq!(clock.now(), 1_500);
    }
}
