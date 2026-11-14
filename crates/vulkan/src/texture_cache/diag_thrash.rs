//! Diagnostic only (diag/thrash): why images re-upload.

use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Cause {
    pub kind: &'static str,
    pub daddr: u64,
    pub size: u64,
}

thread_local! {
    static CURRENT: Cell<Cause> = const { Cell::new(Cause { kind: "unset", daddr: 0, size: 0 }) };
}

pub(crate) fn set(kind: &'static str, daddr: u64, size: u64) {
    CURRENT.with(|c| c.set(Cause { kind, daddr, size }));
}

pub(crate) fn current() -> Cause {
    CURRENT.with(|c| c.get())
}

struct Stat {
    uploads: u64,
    desc: String,
    causes: HashMap<&'static str, u64>,
    last: Cause,
}

struct Stats {
    start: Instant,
    last_print: f64,
    map: HashMap<u64, Stat>,
    total: u64,
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

pub(crate) fn uploaded(key: u64, desc: impl FnOnce() -> String, cause: Cause) {
    let mut guard = STATS.lock().unwrap();
    let stats = guard.get_or_insert_with(|| Stats {
        start: Instant::now(),
        last_print: 0.0,
        map: HashMap::new(),
        total: 0,
    });
    stats.total += 1;
    let stat = stats.map.entry(key).or_insert_with(|| Stat {
        uploads: 0,
        desc: desc(),
        causes: HashMap::new(),
        last: cause,
    });
    stat.uploads += 1;
    *stat.causes.entry(cause.kind).or_default() += 1;
    stat.last = cause;
    let now = stats.start.elapsed().as_secs_f64();
    if now - stats.last_print < 2.0 {
        return;
    }
    let window = now - stats.last_print;
    stats.last_print = now;
    let mut top: Vec<_> = stats.map.iter().collect();
    top.sort_by(|a, b| b.1.uploads.cmp(&a.1.uploads));
    tracing::info!("DIAG thrash t={now:.1} uploads={} per_s={:.0} images={}", stats.total, stats.total as f64 / window, stats.map.len());
    for (_, s) in top.iter().take(12) {
        let mut causes: Vec<_> = s.causes.iter().collect();
        causes.sort_by(|a, b| b.1.cmp(a.1));
        tracing::info!(
            "DIAG thrash img {} uploads={} per_s={:.0} causes={:?} last={}@{:#x}+{:#x}",
            s.desc,
            s.uploads,
            s.uploads as f64 / window,
            causes,
            s.last.kind,
            s.last.daddr,
            s.last.size
        );
    }
    stats.map.clear();
    stats.total = 0;
}
