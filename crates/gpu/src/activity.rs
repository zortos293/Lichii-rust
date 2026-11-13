//! What the GPU threads did, logged every [`REPORT_PERIOD`] by a thread of its own: the draws,
//! clears, dispatches and presents that reached the renderer and how long its calls took, and how
//! much of the period the front was idle or waited for the back, and the back was idle or waited
//! for the host GPU. It tells a slow title's GPU threads apart from its CPU side, and a renderer
//! that never draws from one that draws nothing visible.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

const REPORT_PERIOD: Duration = Duration::from_secs(5);

/// A renderer call [`Activity`] counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererCall {
    Draw,
    Clear,
    Dispatch,
    Present,
}

/// A wait [`Activity`] times. Only one thread at a time waits on each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// The front waits for commands.
    FrontIdle,
    /// The front waits for the back: a full ring or a sync point.
    FrontWaitingForBack,
    /// The back thread waits for work.
    BackIdle,
    /// The back waits for the host GPU to finish its work, and for the read back that follows.
    HostGpu,
}

const NOT_WAITING: u64 = u64::MAX;

/// The time spent in one kind of [`Wait`] since the last report, including the part of a wait
/// still going on.
#[derive(Debug)]
struct WaitClock {
    done: AtomicU64,
    /// When the wait going on began, or the last report took its time so far; nanoseconds since
    /// [`Activity::base`].
    since: AtomicU64,
}

impl WaitClock {
    const fn new() -> Self {
        Self {
            done: AtomicU64::new(0),
            since: AtomicU64::new(NOT_WAITING),
        }
    }

    fn take(&self, now: u64) -> u64 {
        let mut ns = self.done.swap(0, Ordering::Relaxed);
        let since = self.since.load(Ordering::Acquire);
        if since != NOT_WAITING
            && self
                .since
                .compare_exchange(since, now, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
        {
            ns += now.saturating_sub(since);
        }
        ns
    }
}

/// Counters the GPU threads add to, and the reporter thread logs and resets.
#[derive(Debug)]
pub struct Activity {
    base: Instant,
    calls: [AtomicU64; 4],
    renderer_ns: AtomicU64,
    longest_ns: AtomicU64,
    waits: [WaitClock; 4],
    stopped: Mutex<bool>,
    stop: Condvar,
}

impl Default for Activity {
    fn default() -> Self {
        Self::new()
    }
}

/// Times a [`Wait`] until it is dropped.
pub struct Waiting<'a> {
    activity: &'a Activity,
    wait: Wait,
}

impl Drop for Waiting<'_> {
    fn drop(&mut self) {
        let clock = &self.activity.waits[self.wait as usize];
        let since = clock.since.swap(NOT_WAITING, Ordering::AcqRel);
        if since != NOT_WAITING {
            clock
                .done
                .fetch_add(self.activity.now().saturating_sub(since), Ordering::Relaxed);
        }
    }
}

impl Activity {
    pub fn new() -> Self {
        Self {
            base: Instant::now(),
            calls: [const { AtomicU64::new(0) }; 4],
            renderer_ns: AtomicU64::new(0),
            longest_ns: AtomicU64::new(0),
            waits: [const { WaitClock::new() }; 4],
            stopped: Mutex::new(false),
            stop: Condvar::new(),
        }
    }

    fn now(&self) -> u64 {
        u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(NOT_WAITING - 1)
    }

    /// Counts a renderer call of `kind` that started at `start` and has returned.
    #[inline]
    pub fn renderer_call(&self, kind: RendererCall, start: Instant) {
        let ns = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.calls[kind as usize].fetch_add(1, Ordering::Relaxed);
        self.renderer_ns.fetch_add(ns, Ordering::Relaxed);
        self.longest_ns.fetch_max(ns, Ordering::Relaxed);
    }

    /// Starts timing a `wait`, which lasts until the returned guard is dropped.
    #[must_use]
    pub fn waiting(&self, wait: Wait) -> Waiting<'_> {
        self.waits[wait as usize]
            .since
            .store(self.now(), Ordering::Release);
        Waiting {
            activity: self,
            wait,
        }
    }

    /// Logs a report every [`REPORT_PERIOD`] until [`Activity::stop`]. `threaded`: whether the
    /// back has a thread of its own; otherwise its time is the front's.
    pub(crate) fn report_until_stopped(&self, threaded: bool) {
        let mut since = Instant::now();
        let mut stopped = self.stopped.lock();
        while !*stopped {
            self.stop.wait_until(&mut stopped, since + REPORT_PERIOD);
            if *stopped {
                return;
            }
            let elapsed = since.elapsed();
            since = Instant::now();
            self.report(elapsed, threaded);
        }
    }

    /// Ends [`Activity::report_until_stopped`].
    pub(crate) fn stop(&self) {
        *self.stopped.lock() = true;
        self.stop.notify_all();
    }

    fn report(&self, elapsed: Duration, threaded: bool) {
        let now = self.now();
        let [front_idle, front_waiting, back_idle, host_wait] =
            self.waits.each_ref().map(|clock| clock.take(now));
        let take = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed);
        let share = |ns: u64| format!("{:.0}%", ns as f64 * 100.0 / elapsed.as_nanos() as f64);
        let [draws, clears, dispatches, presents] = self.calls.each_ref().map(take);
        let renderer = Duration::from_nanos(take(&self.renderer_ns));
        let longest_call = Duration::from_nanos(take(&self.longest_ns));
        let back_idle = if threaded {
            share(back_idle)
        } else {
            "on the front thread".into()
        };
        tracing::info!(
            period = ?elapsed,
            draws,
            clears,
            dispatches,
            presents,
            ?renderer,
            ?longest_call,
            front_idle = share(front_idle),
            front_waiting_for_back = share(front_waiting),
            back_idle,
            back_waiting_for_host_gpu = share(host_wait),
            "GPU activity"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_resets_the_counters() {
        let activity = Activity::new();
        let start = Instant::now();
        activity.renderer_call(RendererCall::Draw, start);
        activity.renderer_call(RendererCall::Draw, start);
        activity.renderer_call(RendererCall::Present, start);
        assert_eq!(
            activity.calls[RendererCall::Draw as usize].load(Ordering::Relaxed),
            2
        );
        assert_eq!(
            activity.calls[RendererCall::Present as usize].load(Ordering::Relaxed),
            1
        );
        activity.report(REPORT_PERIOD, true);
        assert!(
            activity
                .calls
                .iter()
                .all(|c| c.load(Ordering::Relaxed) == 0)
        );
        assert_eq!(activity.renderer_ns.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn a_wait_spanning_reports_counts_only_its_time_in_each_period() {
        let activity = Activity::new();
        let clock = &activity.waits[Wait::HostGpu as usize];
        let waiting = activity.waiting(Wait::HostGpu);
        clock.since.store(1_000, Ordering::Relaxed);
        assert_eq!(clock.take(5_000), 4_000);
        assert_eq!(clock.take(9_000), 4_000);
        clock.since.store(0, Ordering::Relaxed);
        drop(waiting);
        assert!(clock.take(activity.now()) > 0);
        assert_eq!(
            clock.take(activity.now()),
            0,
            "an ended wait is counted once"
        );
    }

    #[test]
    fn the_reporter_thread_stops() {
        let activity = std::sync::Arc::new(Activity::new());
        let reporter = std::thread::spawn({
            let activity = activity.clone();
            move || activity.report_until_stopped(true)
        });
        activity.stop();
        reporter.join().unwrap();
    }
}
