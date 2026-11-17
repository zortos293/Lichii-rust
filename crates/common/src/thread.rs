//! Host thread priorities. Guest cores keep the default priority; threads on the GPU's critical
//! path run above it and background compiles below it, so a shader burst can't preempt a frame.
//! There is deliberately no priority above `AboveNormal`: a busy-polling thread at
//! `HIGHEST` or `TIME_CRITICAL` can starve the rest of the desktop.

/// A host thread priority relative to the default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadPriority {
    /// Background work whose result is not waited on, such as shader and pipeline compiles.
    BelowNormal,
    /// Threads every frame waits on, such as the GPU front and back.
    AboveNormal,
}

/// Sets the calling thread's priority. Returns whether the OS applied it: only Windows has
/// per-thread priorities an unprivileged process can raise, so elsewhere this does nothing.
pub fn set_current_thread_priority(priority: ThreadPriority) -> bool {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
            THREAD_PRIORITY_BELOW_NORMAL,
        };
        let level = match priority {
            ThreadPriority::BelowNormal => THREAD_PRIORITY_BELOW_NORMAL,
            ThreadPriority::AboveNormal => THREAD_PRIORITY_ABOVE_NORMAL,
        };
        // SAFETY: `GetCurrentThread` returns a pseudo-handle that is always valid for the
        // calling thread, and `SetThreadPriority` reads nothing else.
        let applied = unsafe { SetThreadPriority(GetCurrentThread(), level) } != 0;
        if !applied {
            tracing::warn!("setting the thread priority to {priority:?} failed");
        }
        applied
    }
    #[cfg(not(windows))]
    {
        let _ = priority;
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_thread_takes_the_priority_it_sets() {
        for priority in [ThreadPriority::BelowNormal, ThreadPriority::AboveNormal] {
            let (applied, now) = std::thread::spawn(move || {
                let applied = set_current_thread_priority(priority);
                (applied, current_priority())
            })
            .join()
            .unwrap();
            assert_eq!(applied, cfg!(windows));
            assert_eq!(now, applied.then_some(priority));
        }
    }

    #[cfg(windows)]
    fn current_priority() -> Option<ThreadPriority> {
        use windows_sys::Win32::System::Threading::{
            GetCurrentThread, GetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
            THREAD_PRIORITY_BELOW_NORMAL,
        };
        // SAFETY: `GetCurrentThread` returns a pseudo-handle valid for the calling thread.
        match unsafe { GetThreadPriority(GetCurrentThread()) } {
            THREAD_PRIORITY_BELOW_NORMAL => Some(ThreadPriority::BelowNormal),
            THREAD_PRIORITY_ABOVE_NORMAL => Some(ThreadPriority::AboveNormal),
            _ => None,
        }
    }

    #[cfg(not(windows))]
    fn current_priority() -> Option<ThreadPriority> {
        None
    }
}
