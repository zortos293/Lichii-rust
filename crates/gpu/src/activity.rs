//! What the GPU threads did, logged once every [`REPORT_PERIOD`]: the draws, clears, dispatches and
//! presents that reached the renderer and how long its calls took, and how much of the time the
//! front was idle or waited for the back, and the back was idle or waited for the host GPU. It tells a slow title's GPU
//! threads apart from its CPU side, and a renderer that never draws from one that draws nothing
//! visible.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

const REPORT_PERIOD: Duration = Duration::from_secs(5);

/// A renderer call [`Activity`] counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererCall {
    Draw,
    Clear,
    Dispatch,
    Present,
}

/// Counters the GPU threads add to and the front's [`Reporter`] logs and resets.
#[derive(Debug)]
pub struct Activity {
    calls: [AtomicU64; 4],
    renderer_ns: AtomicU64,
    longest_ns: AtomicU64,
    front_idle_ns: AtomicU64,
    front_waiting_ns: AtomicU64,
    back_idle_ns: AtomicU64,
    host_wait_ns: AtomicU64,
}

fn nanos_since(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

impl Activity {
    pub const fn new() -> Self {
        Self {
            calls: [const { AtomicU64::new(0) }; 4],
            renderer_ns: AtomicU64::new(0),
            longest_ns: AtomicU64::new(0),
            front_idle_ns: AtomicU64::new(0),
            front_waiting_ns: AtomicU64::new(0),
            back_idle_ns: AtomicU64::new(0),
            host_wait_ns: AtomicU64::new(0),
        }
    }

    /// Counts a renderer call of `kind` that started at `start` and has returned.
    #[inline]
    pub fn renderer_call(&self, kind: RendererCall, start: Instant) {
        let ns = nanos_since(start);
        self.calls[kind as usize].fetch_add(1, Ordering::Relaxed);
        self.renderer_ns.fetch_add(ns, Ordering::Relaxed);
        self.longest_ns.fetch_max(ns, Ordering::Relaxed);
    }

    /// The front waited for commands from `start` until now.
    pub(crate) fn front_idle(&self, start: Instant) {
        self.front_idle_ns
            .fetch_add(nanos_since(start), Ordering::Relaxed);
    }

    /// The front waited for the back from `start` until now.
    pub(crate) fn front_waited(&self, start: Instant) {
        self.front_waiting_ns
            .fetch_add(nanos_since(start), Ordering::Relaxed);
    }

    /// The back waited for the host GPU to finish its work from `start` until now.
    pub fn host_waited(&self, start: Instant) {
        self.host_wait_ns
            .fetch_add(nanos_since(start), Ordering::Relaxed);
    }

    /// The back thread waited for work from `start` until now.
    pub(crate) fn back_idle(&self, start: Instant) {
        self.back_idle_ns
            .fetch_add(nanos_since(start), Ordering::Relaxed);
    }
}

impl Default for Activity {
    fn default() -> Self {
        Self::new()
    }
}

/// Logs [`Activity`] from the front thread.
pub(crate) struct Reporter {
    activity: Arc<Activity>,
    since: Instant,
    /// Whether the back has a thread of its own; otherwise its time is the front's.
    threaded: bool,
}

impl Reporter {
    pub(crate) fn new(activity: Arc<Activity>, threaded: bool) -> Self {
        Self {
            activity,
            since: Instant::now(),
            threaded,
        }
    }

    /// Logs and resets the counters once [`REPORT_PERIOD`] passed since the last report.
    pub(crate) fn tick(&mut self) {
        let elapsed = self.since.elapsed();
        if elapsed < REPORT_PERIOD {
            return;
        }
        self.since = Instant::now();
        let activity = &*self.activity;
        let take = |counter: &AtomicU64| counter.swap(0, Ordering::Relaxed);
        let share = |ns: u64| format!("{:.0}%", ns as f64 * 100.0 / elapsed.as_nanos() as f64);
        let [draws, clears, dispatches, presents] = activity.calls.each_ref().map(take);
        let renderer = Duration::from_nanos(take(&activity.renderer_ns));
        let longest = Duration::from_nanos(take(&activity.longest_ns));
        let front_idle = share(take(&activity.front_idle_ns));
        let front_waiting = share(take(&activity.front_waiting_ns));
        let back_waiting_for_host_gpu = share(take(&activity.host_wait_ns));
        let back_idle = take(&activity.back_idle_ns);
        let back_idle = if self.threaded {
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
            renderer = ?renderer,
            longest_call = ?longest,
            front_idle,
            front_waiting_for_back = front_waiting,
            back_idle,
            back_waiting_for_host_gpu,
            "GPU activity"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_resets_the_counters() {
        let activity = Arc::new(Activity::new());
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
        let mut reporter = Reporter::new(activity.clone(), true);
        reporter.since -= REPORT_PERIOD;
        reporter.tick();
        assert!(
            activity
                .calls
                .iter()
                .all(|c| c.load(Ordering::Relaxed) == 0)
        );
        assert_eq!(activity.renderer_ns.load(Ordering::Relaxed), 0);
    }
}
