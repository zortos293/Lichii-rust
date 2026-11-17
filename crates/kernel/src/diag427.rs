//! LIRU-6 diag (ZOR-427, never merged): per guest thread, time running and time blocked by
//! reason and waker, logged every 5 s.

use std::cell::Cell;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

thread_local! {
    static CURRENT: Cell<u64> = const { Cell::new(0) };
}

#[derive(Default)]
struct Guest {
    run: Duration,
    slices: u64,
    pending: Option<(String, Instant)>,
    waker: Option<String>,
    blocked: HashMap<String, (Duration, u64)>,
}

struct State {
    since: Instant,
    threads: HashMap<u64, Guest>,
}

static STATE: LazyLock<Mutex<State>> = LazyLock::new(|| {
    Mutex::new(State {
        since: Instant::now(),
        threads: HashMap::new(),
    })
});

pub(crate) fn set_current(id: u64) {
    CURRENT.with(|c| c.set(id));
}

fn waker_label() -> String {
    let current = CURRENT.with(Cell::get);
    if current != 0 {
        return format!("T{current}");
    }
    std::thread::current().name().unwrap_or("host").to_string()
}

pub(crate) fn blocked(id: u64, label: String) {
    let mut s = STATE.lock();
    let g = s.threads.entry(id).or_default();
    g.pending = Some((label, Instant::now()));
    g.waker = None;
}

pub(crate) fn unblocked(id: u64) {
    let mut s = STATE.lock();
    if let Some(g) = s.threads.get_mut(&id) {
        g.pending = None;
        g.waker = None;
    }
}

pub(crate) fn relabel(id: u64, label: String) {
    let mut s = STATE.lock();
    if let Some((l, _)) = s.threads.entry(id).or_default().pending.as_mut() {
        *l = label;
    }
}

pub(crate) fn woken(id: u64) {
    let waker = waker_label();
    let mut s = STATE.lock();
    let g = s.threads.entry(id).or_default();
    if g.waker.is_none() {
        g.waker = Some(waker);
    }
}

pub(crate) fn resumed(id: u64) {
    let now = Instant::now();
    let mut s = STATE.lock();
    let g = s.threads.entry(id).or_default();
    if let Some((label, at)) = g.pending.take() {
        let waker = g.waker.take().unwrap_or_else(|| "-".into());
        let e = g.blocked.entry(format!("{label} <- {waker}")).or_default();
        e.0 += now - at;
        e.1 += 1;
    }
    maybe_log(&mut s, now);
}

pub(crate) fn ran(id: u64, run: Duration, switched_without_block: bool) {
    let now = Instant::now();
    let mut s = STATE.lock();
    let g = s.threads.entry(id).or_default();
    g.run += run;
    g.slices += 1;
    if switched_without_block && g.pending.is_none() {
        g.pending = Some(("preempted".into(), now));
        g.waker = Some("-".into());
    }
}

fn maybe_log(s: &mut State, now: Instant) {
    let period = now - s.since;
    if period < Duration::from_secs(5) {
        return;
    }
    s.since = now;
    let mut rows: Vec<(&u64, &Guest)> = s.threads.iter().filter(|(_, g)| g.slices > 0).collect();
    rows.sort_by(|a, b| b.1.run.cmp(&a.1.run));
    let mut out = String::new();
    let _ = write!(
        out,
        "LIRU6CRIT period_ms={} threads={}",
        period.as_millis(),
        rows.len()
    );
    for (id, g) in rows.iter().take(14) {
        let _ = write!(
            out,
            "\n  T{id} run_ms={} slices={}",
            g.run.as_millis(),
            g.slices
        );
        let mut b: Vec<_> = g.blocked.iter().collect();
        b.sort_by(|x, y| y.1.0.cmp(&x.1.0));
        for (label, (d, n)) in b.iter().take(7) {
            let _ = write!(out, "\n      {:>6}ms {:>6}x {label}", d.as_millis(), n);
        }
    }
    tracing::info!("{out}");
    for g in s.threads.values_mut() {
        g.run = Duration::ZERO;
        g.slices = 0;
        g.blocked.clear();
    }
}
