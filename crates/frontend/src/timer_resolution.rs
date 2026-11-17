//! Raises the Windows system timer resolution to 1 ms for the life of the process, as the C++
//! Lichii raises it to the maximum. Timed waits that no high-resolution timer backs (condvar
//! timeouts, `Sleep`) round up to this tick, which is 15.6 ms by default.

use std::sync::atomic::{AtomicBool, Ordering};

const PERIOD_MS: u32 = 1;

static RAISED: AtomicBool = AtomicBool::new(false);

/// Holds the raised resolution; dropping it restores the default.
pub struct TimerResolution {
    end: fn(u32) -> u32,
}

impl TimerResolution {
    /// Raises the resolution unless this process already holds it raised. `None` when Windows
    /// refuses.
    #[cfg(windows)]
    pub fn raise() -> Option<Self> {
        Self::raise_with(windows::begin, windows::end)
    }

    /// Other hosts have no such setting.
    #[cfg(not(windows))]
    pub fn raise() -> Option<Self> {
        None
    }

    #[cfg(any(windows, test))]
    fn raise_with(begin: fn(u32) -> u32, end: fn(u32) -> u32) -> Option<Self> {
        if RAISED.swap(true, Ordering::AcqRel) {
            return None;
        }
        let result = begin(PERIOD_MS);
        if result != 0 {
            RAISED.store(false, Ordering::Release);
            tracing::warn!(
                "timeBeginPeriod({PERIOD_MS}) failed ({result}); timed waits keep the default tick"
            );
            return None;
        }
        Some(Self { end })
    }
}

impl Drop for TimerResolution {
    fn drop(&mut self) {
        (self.end)(PERIOD_MS);
        RAISED.store(false, Ordering::Release);
    }
}

#[cfg(windows)]
mod windows {
    pub(super) fn begin(period: u32) -> u32 {
        // SAFETY: timeBeginPeriod takes a plain integer and touches no memory of ours.
        unsafe { windows_sys::Win32::Media::timeBeginPeriod(period) }
    }

    pub(super) fn end(period: u32) -> u32 {
        // SAFETY: as for `begin`; it pairs with the `begin` call the guard was made from.
        unsafe { windows_sys::Win32::Media::timeEndPeriod(period) }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicU32;

    use super::*;

    static BEGINS: AtomicU32 = AtomicU32::new(0);
    static ENDS: AtomicU32 = AtomicU32::new(0);

    fn begin(period: u32) -> u32 {
        assert_eq!(period, 1);
        BEGINS.fetch_add(1, Ordering::SeqCst);
        0
    }

    fn end(period: u32) -> u32 {
        assert_eq!(period, 1);
        ENDS.fetch_add(1, Ordering::SeqCst);
        0
    }

    fn refuse(_: u32) -> u32 {
        97
    }

    #[test]
    fn the_resolution_is_raised_once_and_restored_on_drop() {
        assert!(TimerResolution::raise_with(refuse, end).is_none());
        let held = TimerResolution::raise_with(begin, end).expect("first raise");
        assert!(TimerResolution::raise_with(begin, end).is_none());
        assert_eq!(
            (BEGINS.load(Ordering::SeqCst), ENDS.load(Ordering::SeqCst)),
            (1, 0)
        );
        drop(held);
        assert_eq!(
            (BEGINS.load(Ordering::SeqCst), ENDS.load(Ordering::SeqCst)),
            (1, 1)
        );
        let again = TimerResolution::raise_with(begin, end).expect("raise after restore");
        drop(again);
        assert_eq!(
            (BEGINS.load(Ordering::SeqCst), ENDS.load(Ordering::SeqCst)),
            (2, 2)
        );
    }
}
