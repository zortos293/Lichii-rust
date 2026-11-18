//! Where guest threads spend their blocked time. Every blocking supervisor call names its wait
//! ([`WaitSite`]); the thread's scheduling state notes when it blocked and when it was woken, and
//! the core that resumes it adds both spans to that core's fixed table. The kernel watchdog logs
//! and resets the tables every [`REPORT_PERIOD`] on the `waits` target at debug level
//! (`RUST_LOG=info,waits=debug`), naming the objects waited on longest from the handle tables
//! only then. None of it exists unless that target is enabled when the kernel starts.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::ipc::ClientSession;
use crate::object::KObject;
use crate::process::Process;

/// How often the watchdog logs the wait tables.
pub(crate) const REPORT_PERIOD: Duration = Duration::from_secs(1);
/// Objects of a `WaitSynchronization` the signalled one is looked up in; a wait signalled by a
/// later object, or ended otherwise, counts against the first.
const TRACKED_OBJECTS: usize = 4;
const TABLE_SIZE: usize = 256;
const MAX_PROBE: usize = 16;
/// Objects, and event origins and services, the log line names.
const TOP_OBJECTS: usize = 5;
const TOP_NAMES: usize = 3;

/// A new id for an event or session, so a wait key never matches a later object that reuses
/// the address of one that was closed.
pub(crate) fn serial() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Why a guest thread blocked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WaitReason {
    /// `WaitSynchronization` on events only.
    Event,
    /// `WaitSynchronization` on threads only.
    Thread,
    /// `WaitSynchronization` on processes only.
    Process,
    /// `WaitSynchronization` on objects of several kinds, or none.
    Mixed,
    /// `SendSyncRequest` waiting for the service's reply.
    Ipc,
    /// `ArbitrateLock`.
    Lock,
    /// `WaitProcessWideKeyAtomic`.
    Condvar,
    /// `WaitForAddress`.
    Address,
    /// `SleepThread`.
    Sleep,
}

impl WaitReason {
    pub(crate) const ALL: [Self; 9] = [
        Self::Event,
        Self::Thread,
        Self::Process,
        Self::Mixed,
        Self::Ipc,
        Self::Lock,
        Self::Condvar,
        Self::Address,
        Self::Sleep,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Event => "event",
            Self::Thread => "thread",
            Self::Process => "process",
            Self::Mixed => "mixed",
            Self::Ipc => "ipc",
            Self::Lock => "lock",
            Self::Condvar => "condvar",
            Self::Address => "address",
            Self::Sleep => "sleep",
        }
    }
}

/// What a thread waits on: an event or session by serial, a thread or process by id, or a guest
/// address the arbiter keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum WaitObject {
    Event(u64),
    Thread(u64),
    Process(u64),
    Session(u64),
    Address(u64),
}

impl WaitObject {
    fn of(object: &KObject) -> Option<Self> {
        Some(match object {
            KObject::ReadableEvent(event) => Self::Event(event.serial()),
            KObject::Thread(thread) => Self::Thread(thread.id()),
            KObject::Process(process) => Self::Process(process.id()),
            KObject::ClientSession(session) => Self::Session(session.serial()),
            _ => return None,
        })
    }

    fn kind(self) -> &'static str {
        match self {
            Self::Event(_) => "event",
            Self::Thread(_) => "thread",
            Self::Process(_) => "process",
            Self::Session(_) => "session",
            Self::Address(_) => "address",
        }
    }

    fn slot(self) -> usize {
        let (kind, value) = match self {
            Self::Event(serial) => (1, serial),
            Self::Thread(id) => (2, id),
            Self::Process(id) => (3, id),
            Self::Session(serial) => (4, serial),
            Self::Address(address) => (5, address),
        };
        ((value ^ kind).wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 56) as usize % TABLE_SIZE
    }
}

/// A blocking call's reason and the objects it waits on, for [`WaitStats`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct WaitSite {
    reason: WaitReason,
    objects: [Option<WaitObject>; TRACKED_OBJECTS],
}

impl WaitSite {
    /// `WaitSynchronization` on `objects`, classed by their kinds.
    pub(crate) fn sync(objects: &[Option<KObject>]) -> Self {
        let mut reason = None;
        for object in objects.iter().flatten() {
            let kind = match object {
                KObject::ReadableEvent(_) => WaitReason::Event,
                KObject::Thread(_) => WaitReason::Thread,
                KObject::Process(_) => WaitReason::Process,
                _ => WaitReason::Mixed,
            };
            reason = match reason {
                Some(seen) if seen != kind => Some(WaitReason::Mixed),
                _ => Some(kind),
            };
        }
        let mut tracked = [None; TRACKED_OBJECTS];
        for (slot, object) in tracked.iter_mut().zip(objects) {
            *slot = object.as_ref().and_then(WaitObject::of);
        }
        Self {
            reason: reason.unwrap_or(WaitReason::Mixed),
            objects: tracked,
        }
    }

    /// `SendSyncRequest` on `session`.
    pub(crate) fn ipc(session: &Arc<ClientSession>) -> Self {
        Self::one(WaitReason::Ipc, WaitObject::Session(session.serial()))
    }

    /// An arbiter wait (`Lock`, `Condvar` or `Address`) keyed by the guest address `address`.
    pub(crate) fn address(reason: WaitReason, address: u64) -> Self {
        Self::one(reason, WaitObject::Address(address))
    }

    pub(crate) fn sleep() -> Self {
        Self {
            reason: WaitReason::Sleep,
            objects: [None; TRACKED_OBJECTS],
        }
    }

    fn one(reason: WaitReason, object: WaitObject) -> Self {
        let mut objects = [None; TRACKED_OBJECTS];
        objects[0] = Some(object);
        Self { reason, objects }
    }
}

/// A blocked wait in progress, kept in the thread's scheduling state.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WaitProbe {
    site: WaitSite,
    since: Instant,
    woken: Option<Instant>,
}

impl WaitProbe {
    pub(crate) fn new(site: WaitSite) -> Self {
        Self {
            site,
            since: Instant::now(),
            woken: None,
        }
    }

    /// Notes when the wait ended, the first time it is called.
    pub(crate) fn wake(&mut self) {
        self.woken.get_or_insert_with(Instant::now);
    }
}

/// Time and count of blocked waits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Bucket {
    pub ns: u64,
    pub count: u64,
}

impl Bucket {
    const ZERO: Self = Self { ns: 0, count: 0 };

    fn add(&mut self, ns: u64) {
        self.ns += ns;
        self.count += 1;
    }

    fn merge(&mut self, other: Bucket) {
        self.ns += other.ns;
        self.count += other.count;
    }
}

impl fmt::Display for Bucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}ms x{}", self.ns as f64 / 1e6, self.count)
    }
}

#[derive(Clone, Copy)]
struct Slot {
    object: WaitObject,
    reason: WaitReason,
    total: Bucket,
}

/// One core's waits since the last report: an open-addressed table, so recording allocates
/// nothing. Objects past a full probe run count in `untracked`.
struct CoreTable {
    reasons: [Bucket; WaitReason::ALL.len()],
    woken_to_running_ns: u64,
    untracked: Bucket,
    slots: [Option<Slot>; TABLE_SIZE],
}

impl CoreTable {
    const fn new() -> Self {
        Self {
            reasons: [Bucket::ZERO; WaitReason::ALL.len()],
            woken_to_running_ns: 0,
            untracked: Bucket::ZERO,
            slots: [None; TABLE_SIZE],
        }
    }

    fn add(&mut self, reason: WaitReason, object: Option<WaitObject>, ns: u64) {
        self.reasons[reason as usize].add(ns);
        let Some(object) = object else {
            return;
        };
        let start = object.slot();
        for probe in 0..MAX_PROBE {
            let slot = &mut self.slots[(start + probe) % TABLE_SIZE];
            match slot {
                Some(slot) if slot.object == object && slot.reason == reason => {
                    slot.total.add(ns);
                    return;
                }
                Some(_) => {}
                None => {
                    let mut total = Bucket::ZERO;
                    total.add(ns);
                    *slot = Some(Slot {
                        object,
                        reason,
                        total,
                    });
                    return;
                }
            }
        }
        self.untracked.add(ns);
    }
}

/// Per-core wait tables, present only when the `waits` log target is enabled. Each table is
/// locked by the core that resumes a thread and by the report, never by another core.
pub(crate) struct WaitStats {
    cores: Box<[Mutex<CoreTable>]>,
}

/// One object's waits in a [`WaitReport`], named when the report was made.
#[derive(Clone, Debug)]
pub(crate) struct NamedWait {
    pub reason: WaitReason,
    pub object: WaitObject,
    pub name: String,
    pub total: Bucket,
}

/// The waits of every core since the last report.
#[derive(Debug)]
pub(crate) struct WaitReport {
    pub reasons: [Bucket; WaitReason::ALL.len()],
    pub woken_to_running_ns: u64,
    pub untracked: Bucket,
    /// Every object waited on, longest total first.
    pub objects: Vec<NamedWait>,
}

impl WaitStats {
    /// Tables for `cores` cores if the `waits` target logs at debug level.
    pub(crate) fn if_enabled(cores: usize) -> Option<Self> {
        tracing::enabled!(target: "waits", tracing::Level::DEBUG).then(|| Self::new(cores))
    }

    fn new(cores: usize) -> Self {
        Self {
            cores: (0..cores).map(|_| Mutex::new(CoreTable::new())).collect(),
        }
    }

    /// Adds a wait that ended to `core`'s table. `signalled` is the index of the object that
    /// ended a `WaitSynchronization`, -1 otherwise.
    pub(crate) fn record(&self, core: usize, probe: WaitProbe, signalled: i32) {
        let now = Instant::now();
        let woken = probe.woken.unwrap_or(now);
        let blocked = woken.saturating_duration_since(probe.since).as_nanos() as u64;
        let ready = now.saturating_duration_since(woken).as_nanos() as u64;
        let objects = &probe.site.objects;
        let object = usize::try_from(signalled)
            .ok()
            .and_then(|i| objects.get(i).copied().flatten())
            .or(objects[0]);
        let mut table = self.cores[core].lock();
        table.add(probe.site.reason, object, blocked);
        table.woken_to_running_ns += ready;
    }

    /// Takes and resets every core's table, naming objects from `processes`' handle tables.
    pub(crate) fn report(&self, processes: &[Arc<Process>]) -> WaitReport {
        let mut reasons = [Bucket::ZERO; WaitReason::ALL.len()];
        let mut woken_to_running_ns = 0;
        let mut untracked = Bucket::ZERO;
        let mut totals: HashMap<(WaitObject, WaitReason), Bucket> = HashMap::new();
        for core in self.cores.iter() {
            let table = std::mem::replace(&mut *core.lock(), CoreTable::new());
            for (sum, bucket) in reasons.iter_mut().zip(table.reasons) {
                sum.merge(bucket);
            }
            woken_to_running_ns += table.woken_to_running_ns;
            untracked.merge(table.untracked);
            for slot in table.slots.iter().flatten() {
                totals
                    .entry((slot.object, slot.reason))
                    .or_default()
                    .merge(slot.total);
            }
        }
        let mut names: HashMap<WaitObject, String> = HashMap::new();
        for process in processes {
            for (_, object) in process.handles().read().iter() {
                if let Some(key) = WaitObject::of(object)
                    && !names.contains_key(&key)
                    && totals.keys().any(|(waited, _)| *waited == key)
                {
                    names.insert(key, object_name(object));
                }
            }
        }
        let mut objects: Vec<NamedWait> = totals
            .into_iter()
            .map(|((object, reason), total)| NamedWait {
                reason,
                object,
                name: match object {
                    WaitObject::Address(address) => format!("{address:#x}"),
                    _ => names
                        .get(&object)
                        .cloned()
                        .unwrap_or_else(|| "closed".into()),
                },
                total,
            })
            .collect();
        objects.sort_by_key(|wait| std::cmp::Reverse((wait.total.ns, wait.total.count)));
        WaitReport {
            reasons,
            woken_to_running_ns,
            untracked,
            objects,
        }
    }
}

fn object_name(object: &KObject) -> String {
    match object {
        KObject::ReadableEvent(event) => event.origin().unwrap_or("origin unknown").into(),
        KObject::Thread(thread) => format!("thread {}", thread.id()),
        KObject::Process(process) => process.name().into(),
        KObject::ClientSession(session) => session.service_name(),
        other => other.type_name().into(),
    }
}

impl WaitReport {
    pub(crate) fn reason(&self, reason: WaitReason) -> Bucket {
        self.reasons[reason as usize]
    }

    /// Waits on objects of `kind` ([`WaitObject::kind`]), summed by name, longest first.
    fn by_name(&self, kind: &str) -> Vec<(&str, Bucket)> {
        let mut named: Vec<(&str, Bucket)> = Vec::new();
        for wait in self
            .objects
            .iter()
            .filter(|wait| wait.object.kind() == kind)
        {
            match named.iter_mut().find(|(name, _)| *name == wait.name) {
                Some((_, total)) => total.merge(wait.total),
                None => named.push((&wait.name, wait.total)),
            }
        }
        named.sort_by_key(|(_, total)| std::cmp::Reverse(total.ns));
        named
    }
}

impl fmt::Display for WaitReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("blocked:")?;
        let mut any = false;
        for reason in WaitReason::ALL {
            let bucket = self.reason(reason);
            if bucket.count > 0 {
                write!(f, " {} {bucket}", reason.name())?;
                any = true;
            }
        }
        if !any {
            f.write_str(" nothing")?;
        }
        write!(
            f,
            "; woken to running {:.1}ms",
            self.woken_to_running_ns as f64 / 1e6
        )?;
        for (label, kind) in [("events by origin", "event"), ("ipc by service", "session")] {
            let named = self.by_name(kind);
            if named.is_empty() {
                continue;
            }
            write!(f, "; {label}:")?;
            for (i, (name, total)) in named.iter().take(TOP_NAMES).enumerate() {
                write!(f, "{} {name} {total}", if i == 0 { "" } else { "," })?;
            }
        }
        if !self.objects.is_empty() {
            f.write_str("; top:")?;
            for (i, wait) in self.objects.iter().take(TOP_OBJECTS).enumerate() {
                write!(
                    f,
                    "{} {} {} [{}] {}",
                    if i == 0 { "" } else { "," },
                    wait.reason.name(),
                    wait.object.kind(),
                    wait.name,
                    wait.total
                )?;
            }
        }
        if self.untracked.count > 0 {
            write!(f, "; untracked {}", self.untracked)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ended(site: WaitSite, blocked_ms: u64) -> WaitProbe {
        let woken = Instant::now();
        WaitProbe {
            site,
            since: woken - Duration::from_millis(blocked_ms),
            woken: Some(woken),
        }
    }

    fn event(id: u64) -> WaitSite {
        WaitSite::one(WaitReason::Event, WaitObject::Event(id))
    }

    #[test]
    fn the_top_objects_are_the_longest_waited_on_across_cores() {
        let stats = WaitStats::new(2);
        for (id, ms) in [(1, 10), (2, 40), (3, 5), (4, 30), (5, 20), (6, 1)] {
            stats.record(0, ended(event(id), ms), -1);
        }
        stats.record(1, ended(event(3), 50), -1);
        stats.record(
            1,
            ended(WaitSite::address(WaitReason::Lock, 0x8000), 25),
            -1,
        );
        stats.record(1, ended(WaitSite::sleep(), 100), -1);
        let report = stats.report(&[]);
        let top: Vec<(WaitObject, u64, u64)> = report
            .objects
            .iter()
            .take(TOP_OBJECTS)
            .map(|wait| (wait.object, wait.total.ns / 1_000_000, wait.total.count))
            .collect();
        assert_eq!(
            top,
            vec![
                (WaitObject::Event(3), 55, 2),
                (WaitObject::Event(2), 40, 1),
                (WaitObject::Event(4), 30, 1),
                (WaitObject::Address(0x8000), 25, 1),
                (WaitObject::Event(5), 20, 1),
            ]
        );
        assert_eq!(report.reason(WaitReason::Event).count, 7);
        assert_eq!(report.reason(WaitReason::Event).ns / 1_000_000, 156);
        assert_eq!(report.reason(WaitReason::Lock).count, 1);
        assert_eq!(report.reason(WaitReason::Sleep).count, 1);
        let line = report.to_string();
        assert!(
            line.contains("top: event event [closed] 55.0ms x2"),
            "{line}"
        );
        assert!(line.contains("lock address [0x8000] 25.0ms x1"), "{line}");
        assert!(!line.contains("10.0ms"), "{line}");
        assert!(
            stats.report(&[]).objects.is_empty(),
            "a report resets the tables"
        );
    }

    #[test]
    fn a_multi_object_wait_counts_against_the_signalled_object() {
        let stats = WaitStats::new(1);
        let mut site = event(1);
        site.objects[1] = Some(WaitObject::Event(2));
        stats.record(0, ended(site, 3), 1);
        stats.record(0, ended(site, 4), -1);
        let mut objects: Vec<(WaitObject, u64)> = stats
            .report(&[])
            .objects
            .iter()
            .map(|wait| (wait.object, wait.total.count))
            .collect();
        objects.sort_by_key(|(object, _)| matches!(object, WaitObject::Event(2)));
        assert_eq!(
            objects,
            vec![(WaitObject::Event(1), 1), (WaitObject::Event(2), 1)]
        );
    }

    #[test]
    fn a_full_probe_run_counts_as_untracked() {
        let mut table = CoreTable::new();
        let waits = TABLE_SIZE as u64 * 4;
        for address in 0..waits {
            table.add(WaitReason::Address, Some(WaitObject::Address(address)), 1);
        }
        let tracked = table.slots.iter().flatten().count() as u64;
        assert!(tracked <= TABLE_SIZE as u64 && table.untracked.count > 0);
        assert_eq!(tracked + table.untracked.count, waits);
        assert_eq!(table.reasons[WaitReason::Address as usize].count, waits);
    }
}
