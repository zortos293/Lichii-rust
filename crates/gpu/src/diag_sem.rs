//! Diagnostic only (diag/acquire): a ring of recent semaphore writes, reductions, GPU engine
//! writes, dropped releases and compute dispatches, dumped when a channel's acquire stalls.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Copy)]
struct Entry {
    t: f32,
    kind: &'static str,
    addr: u64,
    a: u64,
    b: u64,
    c: u64,
}

static RING: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());
static DROPS: Mutex<VecDeque<Entry>> = Mutex::new(VecDeque::new());
pub static DROP_COUNT: AtomicU64 = AtomicU64::new(0);

fn secs() -> f32 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f32()
}

fn push(ring: &Mutex<VecDeque<Entry>>, cap: usize, entry: Entry) {
    let mut ring = ring.lock().unwrap();
    if ring.len() == cap {
        ring.pop_front();
    }
    ring.push_back(entry);
}

pub fn rec(kind: &'static str, addr: u64, a: u64, b: u64, c: u64) {
    push(&RING, 1 << 15, Entry { t: secs(), kind, addr, a, b, c });
}

pub fn drop(kind: &'static str, addr: u64, a: u64, b: u64, c: u64) {
    DROP_COUNT.fetch_add(1, Ordering::Relaxed);
    push(&DROPS, 512, Entry { t: secs(), kind, addr, a, b, c });
}

fn line(e: &Entry) -> String {
    format!("t={:.3} {} addr={:#x} a={:#x} b={:#x} c={:#x}", e.t, e.kind, e.addr, e.a, e.b, e.c)
}

/// Dumps every recorded entry touching the page of `device` or of `gpu_va`, then the newest
/// entries and dropped releases.
pub fn dump(channel: u32, device: Option<u64>, gpu_va: u64) {
    let ring = RING.lock().unwrap().clone();
    let page = |a: u64| a & !0xfff;
    let hits: Vec<&Entry> = ring
        .iter()
        .filter(|e| {
            let lo = page(e.addr);
            let hi = e.addr.saturating_add(e.b.max(4).min(1 << 20));
            let near = |x: u64| page(x) <= hi && page(x) + 0x1000 > lo;
            device.is_some_and(near) || near(gpu_va)
        })
        .collect();
    tracing::warn!(
        "DIAG acquire dump channel={channel} device={device:#x?} gpu_va={gpu_va:#x} ring={} hits={} drops_total={}",
        ring.len(),
        hits.len(),
        DROP_COUNT.load(Ordering::Relaxed)
    );
    for e in hits.iter().rev().take(200).rev() {
        tracing::warn!("DIAG acquire hit {}", line(e));
    }
    let dispatches: Vec<&Entry> = ring.iter().filter(|e| e.kind == "dispatch").collect();
    for e in dispatches.iter().rev().take(60).rev() {
        tracing::warn!("DIAG acquire recent-dispatch {}", line(e));
    }
    for e in ring.iter().rev().take(150).rev() {
        tracing::warn!("DIAG acquire recent {}", line(e));
    }
    let drops = DROPS.lock().unwrap().clone();
    for e in drops.iter().rev().take(60).rev() {
        tracing::warn!("DIAG acquire recent-drop {}", line(e));
    }
}
