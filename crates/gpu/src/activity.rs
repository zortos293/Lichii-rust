//! What the GPU threads did, logged every [`REPORT_PERIOD`] by a thread of its own: the draws,
//! clears, dispatches and presents that reached the renderer and how long its calls took, and how
//! much of the period the front was idle or waited for the back, and the back was idle or waited
//! for the host GPU. It tells a slow title's GPU threads apart from its CPU side, and a renderer
//! that never draws from one that draws nothing visible.

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::host1x::{SYNCPOINT_COUNT, Syncpoints, is_reached};

const REPORT_PERIOD: Duration = Duration::from_secs(5);
/// How often the reporter looks for a back that stopped making progress.
const STALL_CHECK: Duration = Duration::from_secs(1);
/// A back that made no progress this long with work queued is reported.
const STALL: Duration = Duration::from_secs(2);
/// A stall still going on is reported again this often.
const STALL_REPEAT: Duration = Duration::from_secs(30);

/// A renderer call [`Activity`] counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RendererCall {
    Draw,
    Clear,
    Dispatch,
    Present,
}

impl RendererCall {
    const ALL: [Self; 4] = [Self::Draw, Self::Clear, Self::Dispatch, Self::Present];
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
    /// The renderer call going on, and since when ([`Activity::base`] nanoseconds), or
    /// [`NOT_WAITING`].
    call_kind: AtomicU8,
    call_since: AtomicU64,
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
            call_kind: AtomicU8::new(0),
            call_since: AtomicU64::new(NOT_WAITING),
            waits: [const { WaitClock::new() }; 4],
            stopped: Mutex::new(false),
            stop: Condvar::new(),
        }
    }

    fn now(&self) -> u64 {
        u64::try_from(self.base.elapsed().as_nanos()).unwrap_or(NOT_WAITING - 1)
    }

    /// A renderer call of `kind` starts; returns the start [`Activity::renderer_call`] takes when
    /// it returned.
    #[inline]
    pub fn begin_call(&self, kind: RendererCall) -> Instant {
        self.call_kind.store(kind as u8, Ordering::Relaxed);
        self.call_since.store(self.now(), Ordering::Release);
        Instant::now()
    }

    /// Counts a renderer call of `kind` that [`Activity::begin_call`] started at `start` and that
    /// has returned.
    #[inline]
    pub fn renderer_call(&self, kind: RendererCall, start: Instant) {
        self.call_since.store(NOT_WAITING, Ordering::Release);
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

    /// Logs a report every [`REPORT_PERIOD`], and a back that stopped making progress, until
    /// [`Activity::stop`]. `threaded`: whether the back has a thread of its own; otherwise its
    /// time is the front's.
    pub(crate) fn report_until_stopped(
        &self,
        threaded: bool,
        back: &dyn BackView,
        syncpoints: &Syncpoints,
    ) {
        let mut since = Instant::now();
        let mut watch = StallWatch::default();
        let mut syncpoint_watch = SyncpointWatch::default();
        let mut stopped = self.stopped.lock();
        while !*stopped {
            self.stop.wait_for(&mut stopped, STALL_CHECK);
            if *stopped {
                return;
            }
            let snapshot = back.snapshot();
            watch.check(self, snapshot);
            let idle = snapshot.published == snapshot.consumed && self.current_call().is_none();
            syncpoint_watch.check(syncpoints, idle);
            if since.elapsed() >= REPORT_PERIOD {
                let elapsed = since.elapsed();
                since = Instant::now();
                self.report(elapsed, threaded);
            }
        }
    }

    /// The renderer call going on, if any.
    fn current_call(&self) -> Option<RendererCall> {
        let since = self.call_since.load(Ordering::Acquire);
        (since != NOT_WAITING)
            .then(|| RendererCall::ALL[usize::from(self.call_kind.load(Ordering::Relaxed)) & 3])
    }

    fn waiting_for_host_gpu(&self) -> bool {
        self.waits[Wait::HostGpu as usize]
            .since
            .load(Ordering::Acquire)
            != NOT_WAITING
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

/// What the back is doing, as the ring sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BackSnapshot {
    pub published: u64,
    pub consumed: u64,
    /// The header of the record the back executes.
    pub executing: u64,
}

/// Reads [`BackSnapshot`]s from another thread.
pub(crate) trait BackView: Send + Sync {
    fn snapshot(&self) -> BackSnapshot;
}

/// Finds a back that has work queued but made no progress for [`STALL`].
#[derive(Default)]
struct StallWatch {
    /// The back's own position (consumed, executing) when last seen; the front publishing more
    /// is not progress.
    last: Option<(u64, u64)>,
    /// When the back last made progress, or had nothing to do.
    progressed: Option<Instant>,
    reported: Option<Instant>,
}

impl StallWatch {
    fn check(&mut self, activity: &Activity, now: BackSnapshot) {
        let idle = now.published == now.consumed && activity.current_call().is_none();
        let position = (now.consumed, now.executing);
        if idle || self.last != Some(position) {
            self.last = Some(position);
            self.progressed = Some(Instant::now());
            if self.reported.take().is_some() {
                tracing::warn!("the GPU back is making progress again");
            }
            return;
        }
        let stalled = self.progressed.map_or(Duration::ZERO, |at| at.elapsed());
        if stalled < STALL || self.reported.is_some_and(|at| at.elapsed() < STALL_REPEAT) {
            return;
        }
        self.reported = Some(Instant::now());
        let header = now.executing;
        tracing::warn!(
            for_ = ?stalled,
            record = header as u8,
            engine = (header >> 8) as u8,
            method = format_args!("{:#x}", (header >> 16) & 0xffff),
            argument = format_args!("{:#x}", header >> 32),
            renderer_call = ?activity.current_call(),
            waiting_for_host_gpu = activity.waiting_for_host_gpu(),
            published = now.published,
            consumed = now.consumed,
            "the GPU back has made no progress with work queued; this is the record it executes"
        );
    }
}

/// Finds syncpoints a guest fence can wait on forever while the back has nothing to do: one
/// with a registered action (an nvhost-ctrl event wait) it does not reach, or one below the
/// maximum its reserved increments reach (host value behind: an increment never executed; guest
/// value behind: host GPU work that never completed).
#[derive(Default)]
struct SyncpointWatch {
    stuck: Vec<StuckSyncpoint>,
    since: Option<Instant>,
    reported: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct StuckSyncpoint {
    id: u32,
    guest: u32,
    host: u32,
    max: u32,
    waited_for: Option<u32>,
}

impl SyncpointWatch {
    fn check(&mut self, syncpoints: &Syncpoints, back_idle: bool) {
        let stuck: Vec<_> = if back_idle {
            (0..SYNCPOINT_COUNT as u32)
                .filter_map(|id| {
                    let guest = syncpoints.guest().value(id).ok()?;
                    let max = syncpoints.max(id).ok()?;
                    let waited_for = syncpoints.guest().next_action_threshold(id).ok()?;
                    let waits = waited_for.is_some_and(|threshold| !is_reached(guest, threshold));
                    (waits || !is_reached(guest, max)).then(|| StuckSyncpoint {
                        id,
                        guest,
                        host: syncpoints.host().value(id).unwrap_or(0),
                        max,
                        waited_for,
                    })
                })
                .collect()
        } else {
            Vec::new()
        };
        if stuck.is_empty() || stuck != self.stuck {
            self.stuck = stuck;
            self.since = Some(Instant::now());
            self.reported = None;
            return;
        }
        let for_ = self.since.map_or(Duration::ZERO, |at| at.elapsed());
        if for_ < STALL || self.reported.is_some_and(|at| at.elapsed() < STALL_REPEAT) {
            return;
        }
        self.reported = Some(Instant::now());
        let list = self
            .stuck
            .iter()
            .take(8)
            .map(|s| {
                let waited = s
                    .waited_for
                    .map(|t| format!(", waited for {t:#x}"))
                    .unwrap_or_default();
                format!(
                    "{}: guest {:#x}, host {:#x}, max {:#x}{waited}",
                    s.id, s.guest, s.host, s.max
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        tracing::warn!(
            ?for_,
            count = self.stuck.len(),
            syncpoints = list,
            "syncpoints a guest fence may wait on stay unreached while the GPU back has nothing to do"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_resets_the_counters() {
        let activity = Activity::new();
        for kind in [
            RendererCall::Draw,
            RendererCall::Draw,
            RendererCall::Present,
        ] {
            let start = activity.begin_call(kind);
            activity.renderer_call(kind, start);
        }
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
        struct Idle;
        impl BackView for Idle {
            fn snapshot(&self) -> BackSnapshot {
                BackSnapshot {
                    published: 0,
                    consumed: 0,
                    executing: 0,
                }
            }
        }
        let reporter = std::thread::spawn({
            let activity = activity.clone();
            move || activity.report_until_stopped(true, &Idle, &Syncpoints::new())
        });
        activity.stop();
        reporter.join().unwrap();
    }

    #[test]
    fn a_stalled_back_is_found_only_with_work_queued() {
        let activity = Activity::new();
        let mut watch = StallWatch::default();
        let busy = BackSnapshot {
            published: 10,
            consumed: 4,
            executing: 1,
        };
        watch.check(&activity, busy);
        watch.progressed = Some(Instant::now() - STALL);
        watch.check(&activity, busy);
        assert!(
            watch.reported.is_some(),
            "no progress with work queued is a stall"
        );
        let more = BackSnapshot {
            published: 20,
            ..busy
        };
        watch.check(&activity, more);
        assert!(
            watch.reported.is_some(),
            "the front publishing more is not the back making progress"
        );
        let idle = BackSnapshot {
            published: 10,
            consumed: 10,
            executing: 1,
        };
        watch.check(&activity, idle);
        assert!(watch.reported.is_none());
        watch.progressed = Some(Instant::now() - STALL);
        watch.check(&activity, idle);
        assert!(watch.reported.is_none(), "an idle back is not stalled");
        let start = activity.begin_call(RendererCall::Present);
        assert_eq!(activity.current_call(), Some(RendererCall::Present));
        activity.renderer_call(RendererCall::Present, start);
        assert_eq!(activity.current_call(), None);
    }

    #[test]
    fn a_syncpoint_left_below_its_maximum_with_an_idle_back_is_found() {
        let syncpoints = Syncpoints::new();
        let mut watch = SyncpointWatch::default();
        watch.check(&syncpoints, true);
        assert!(watch.stuck.is_empty());
        syncpoints.increase_max(7, 2).unwrap();
        syncpoints.host().increment(7).unwrap();
        watch.check(&syncpoints, true);
        let stuck = StuckSyncpoint {
            id: 7,
            guest: 0,
            host: 1,
            max: 2,
            waited_for: None,
        };
        assert_eq!(watch.stuck, [stuck]);
        watch.since = Some(Instant::now() - STALL);
        watch.check(&syncpoints, true);
        assert!(watch.reported.is_some());
        watch.check(&syncpoints, false);
        assert!(watch.stuck.is_empty(), "a busy back is still working on it");
        syncpoints.guest().increment(7).unwrap();
        syncpoints.guest().increment(7).unwrap();
        watch.check(&syncpoints, true);
        assert!(watch.stuck.is_empty());
    }

    #[test]
    fn an_unreached_event_wait_is_found_and_a_reached_one_is_not() {
        let syncpoints = Syncpoints::new();
        let mut watch = SyncpointWatch::default();
        let _wait = syncpoints.guest().register_action(3, 5, || {}).unwrap();
        syncpoints.guest().register_action(3, 9, || {}).unwrap();
        watch.check(&syncpoints, true);
        assert_eq!(watch.stuck.len(), 1);
        assert_eq!(watch.stuck[0].waited_for, Some(5));
        for _ in 0..5 {
            syncpoints.guest().increment(3).unwrap();
        }
        watch.check(&syncpoints, true);
        assert_eq!(watch.stuck[0].waited_for, Some(9));
        for _ in 0..4 {
            syncpoints.guest().increment(3).unwrap();
        }
        watch.check(&syncpoints, true);
        assert!(watch.stuck.is_empty());
    }
}
