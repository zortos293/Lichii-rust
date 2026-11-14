//! Diagnostic only (diag/dmaread): what macro DMA reads read and what they wait for.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::sync::Mutex;
use std::time::Instant;

use crate::backend::Query;

#[derive(Default)]
struct Call {
    method: u32,
    emitted: Vec<u32>,
    reads: Vec<(bool, u64, Vec<u32>, u64)>,
}

thread_local! {
    static CALL: RefCell<Option<Call>> = const { RefCell::new(None) };
    static FIFOED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[derive(Default)]
struct FrontStat {
    calls: u64,
    reads: u64,
    fifoed: u64,
    front_ns: u64,
    words: Vec<u32>,
    emitted: BTreeSet<u32>,
}

#[derive(Default)]
struct BackStat {
    reads: u64,
    hits: HashMap<String, u64>,
    pending_ticks: u64,
    wait_ns: u64,
    query: Option<Query>,
}

struct State {
    start: Instant,
    front_print: f64,
    back_print: f64,
    front: HashMap<(u32, u64), FrontStat>,
    back: HashMap<u64, BackStat>,
    queries: HashMap<u64, Query>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut guard = STATE.lock().unwrap();
    let state = guard.get_or_insert_with(|| State {
        start: Instant::now(),
        front_print: 0.0,
        back_print: 0.0,
        front: HashMap::new(),
        back: HashMap::new(),
        queries: HashMap::new(),
    });
    f(state)
}

pub fn call_begin(method: u32) {
    CALL.with(|c| {
        *c.borrow_mut() = Some(Call {
            method,
            ..Call::default()
        })
    });
}

pub fn emit(method: u32) {
    CALL.with(|c| {
        if let Some(call) = c.borrow_mut().as_mut()
            && call.emitted.len() < 4096
        {
            call.emitted.push(method);
        }
    });
}

pub fn set_fifoed(fifoed: bool) {
    FIFOED.with(|f| f.set(fifoed));
}

pub fn front_read(gpu_va: u64, words: &[u32], front_ns: u64) {
    let fifoed = FIFOED.with(|f| f.get());
    CALL.with(|c| {
        if let Some(call) = c.borrow_mut().as_mut() {
            call.reads.push((fifoed, gpu_va, words.to_vec(), front_ns));
        }
    });
}

pub fn call_end() {
    let Some(call) = CALL.with(|c| c.borrow_mut().take()) else {
        return;
    };
    if call.reads.is_empty() {
        return;
    }
    with(|s| {
        for (fifoed, gpu_va, words, ns) in &call.reads {
            let stat = s.front.entry((call.method, *gpu_va)).or_default();
            stat.calls += 1;
            stat.reads += 1;
            stat.fifoed += u64::from(*fifoed);
            stat.front_ns += ns;
            stat.words.clone_from(words);
            stat.emitted.extend(call.emitted.iter().copied());
        }
        let now = s.start.elapsed().as_secs_f64();
        if now - s.front_print < 5.0 {
            return;
        }
        s.front_print = now;
        let mut rows: Vec<_> = s.front.iter().collect();
        rows.sort_by(|a, b| b.1.front_ns.cmp(&a.1.front_ns));
        tracing::info!("DIAG dmaread front t={now:.1} keys={}", rows.len());
        for ((method, va), st) in rows.iter().take(12) {
            let emitted: Vec<String> = st.emitted.iter().take(40).map(|m| format!("{m:#x}")).collect();
            tracing::info!(
                "DIAG dmaread front macro={method:#x} va={va:#x} reads={} fifoed={} front_ms={:.1} words={:08x?} emits=[{}]",
                st.reads,
                st.fifoed,
                st.front_ns as f64 / 1e6,
                st.words,
                emitted.join(",")
            );
        }
        s.front.clear();
    });
}

pub fn query(daddr: u64, query: &Query) {
    with(|s| {
        if s.queries.len() > 200_000 {
            s.queries.clear();
        }
        s.queries.insert(daddr, *query);
    });
}

pub fn back_read(daddr: u64, size: u64, hit: Option<(u64, String)>, pending_ticks: u64, wait_ns: u64) {
    with(|s| {
        let query = hit.as_ref().and_then(|(a, _)| s.queries.get(a).copied());
        let stat = s.back.entry(daddr).or_default();
        stat.reads += 1;
        let key = hit.map_or_else(|| "none".to_string(), |(a, k)| format!("{k}@+{:#x}", a.wrapping_sub(daddr)));
        *stat.hits.entry(key).or_default() += 1;
        stat.pending_ticks += pending_ticks;
        stat.wait_ns += wait_ns;
        if query.is_some() {
            stat.query = query;
        }
        let now = s.start.elapsed().as_secs_f64();
        if now - s.back_print < 5.0 {
            return;
        }
        s.back_print = now;
        let mut rows: Vec<_> = s.back.iter().collect();
        rows.sort_by(|a, b| b.1.wait_ns.cmp(&a.1.wait_ns));
        tracing::info!("DIAG dmaread back t={now:.1} keys={}", rows.len());
        for (addr, st) in rows.iter().take(12) {
            tracing::info!(
                "DIAG dmaread back daddr={addr:#x} size={size} reads={} wait_ms={:.1} avg_pending_ticks={:.1} hits={:?} query={:?}",
                st.reads,
                st.wait_ns as f64 / 1e6,
                st.pending_ticks as f64 / st.reads as f64,
                st.hits,
                st.query
            );
        }
        s.back.clear();
    });
}
