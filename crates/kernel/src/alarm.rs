//! How the timer thread waits for its next deadline. A condvar timeout follows the host's timer
//! resolution, which on Windows is 15.6 ms unless a process raises it, so a 1 ms guest sleep took
//! 16. There the wait runs on a high-resolution waitable timer beside an event that
//! [`Alarm::notify`] signals, as the reference does.

#[cfg(windows)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use parking_lot::{Condvar, MutexGuard};

pub(crate) struct Alarm {
    cv: Condvar,
    #[cfg(windows)]
    precise: Option<windows::PreciseWait>,
    #[cfg(windows)]
    precise_failed: AtomicBool,
}

impl Default for Alarm {
    fn default() -> Self {
        Self {
            cv: Condvar::new(),
            #[cfg(windows)]
            precise: windows::PreciseWait::new(),
            #[cfg(windows)]
            precise_failed: AtomicBool::new(false),
        }
    }
}

impl Alarm {
    /// Ends the current or next wait early. Call it with the waiter's mutex held.
    pub fn notify(&self) {
        #[cfg(windows)]
        if let Some(precise) = &self.precise {
            precise.notify();
        }
        self.cv.notify_one();
    }

    /// Releases `guard` until `deadline` or a [`Alarm::notify`]; it may return early.
    pub fn wait_until<T>(&self, guard: &mut MutexGuard<'_, T>, deadline: Instant) {
        #[cfg(windows)]
        if let Some(precise) = &self.precise
            && !self.precise_failed.load(Ordering::Relaxed)
        {
            if !MutexGuard::unlocked(guard, || precise.wait_until(deadline)) {
                tracing::warn!("high-resolution timer wait failed; using the condvar from now on");
                self.precise_failed.store(true, Ordering::Relaxed);
            }
            return;
        }
        self.cv.wait_until(guard, deadline);
    }
}

#[cfg(windows)]
mod windows {
    use std::ptr::null;
    use std::time::Instant;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, WAIT_FAILED};
    use windows_sys::Win32::System::Threading::{
        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, CreateEventW, CreateWaitableTimerExW, SetEvent,
        SetWaitableTimer, TIMER_ALL_ACCESS, WaitForMultipleObjects,
    };

    pub(super) struct PreciseWait {
        event: HANDLE,
        timer: HANDLE,
    }

    // SAFETY: both handles name kernel objects that any thread may signal or wait on.
    unsafe impl Send for PreciseWait {}
    // SAFETY: as above; the handles are never changed after creation.
    unsafe impl Sync for PreciseWait {}

    impl PreciseWait {
        /// `None` where Windows has no high-resolution timers (before 10 1803).
        pub fn new() -> Option<Self> {
            // SAFETY: null attributes and names create unnamed objects with default security.
            let wait = unsafe {
                Self {
                    event: CreateEventW(null(), 0, 0, null()),
                    timer: CreateWaitableTimerExW(
                        null(),
                        null(),
                        CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                        TIMER_ALL_ACCESS,
                    ),
                }
            };
            (!wait.event.is_null() && !wait.timer.is_null()).then_some(wait)
        }

        pub fn notify(&self) {
            // SAFETY: the event handle lives as long as self.
            unsafe { SetEvent(self.event) };
        }

        /// Waits for the timer or the event; false when Windows refused the wait.
        pub fn wait_until(&self, deadline: Instant) -> bool {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let due = -i64::try_from(remaining.as_nanos().div_ceil(100)).unwrap_or(i64::MAX);
            let backstop = u32::try_from(remaining.as_millis() + 1)
                .unwrap_or(u32::MAX)
                .min(u32::MAX - 1);
            let handles = [self.event, self.timer];
            // SAFETY: both handles live as long as self, and the timer has no completion routine.
            unsafe {
                SetWaitableTimer(self.timer, &due, 0, None, null(), 0) != 0
                    && WaitForMultipleObjects(2, handles.as_ptr(), 0, backstop) != WAIT_FAILED
            }
        }
    }

    impl Drop for PreciseWait {
        fn drop(&mut self) {
            for handle in [self.event, self.timer] {
                if !handle.is_null() {
                    // SAFETY: this object created the handle and nothing else closes it.
                    unsafe { CloseHandle(handle) };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use parking_lot::Mutex;

    use super::*;

    #[test]
    fn short_waits_end_close_to_their_deadline() {
        let alarm = Alarm::default();
        let mutex = Mutex::new(());
        let mut waits: Vec<Duration> = (0..9)
            .map(|_| {
                let start = Instant::now();
                let deadline = start + Duration::from_millis(1);
                let mut guard = mutex.lock();
                while Instant::now() < deadline {
                    alarm.wait_until(&mut guard, deadline);
                }
                start.elapsed()
            })
            .collect();
        waits.sort();
        assert!(waits[4] < Duration::from_millis(8), "{waits:?}");
    }

    #[test]
    fn notify_ends_a_long_wait() {
        let shared = Arc::new((Alarm::default(), Mutex::new(())));
        let mut guard = shared.1.lock();
        let notifier = {
            let shared = shared.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                let _guard = shared.1.lock();
                shared.0.notify();
            })
        };
        let start = Instant::now();
        shared
            .0
            .wait_until(&mut guard, start + Duration::from_secs(10));
        assert!(start.elapsed() < Duration::from_secs(5));
        drop(guard);
        notifier.join().unwrap();
    }
}
