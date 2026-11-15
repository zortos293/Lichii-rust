use std::sync::atomic::{AtomicU64, AtomicU8, Ordering::Relaxed};
use std::time::Instant;

pub static CTX: AtomicU8 = AtomicU8::new(0);
pub static C: [[AtomicU64; 11]; 4] = [const { [const { AtomicU64::new(0) }; 11] }; 4];
pub static ENTRY: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
pub static BIND: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
static LAST: AtomicU64 = AtomicU64::new(0);

pub const CALLS: usize = 0;
pub const NS: usize = 1;
pub const FLUSHES: usize = 2;
pub const ASIDE: usize = 3;
pub const FINISH: usize = 4;
pub const BYTES: usize = 5;
pub const CHANGED: usize = 6;
pub const WAIT_NS: usize = 7;
pub const CHUNKS_SAME: usize = 8;

pub fn add(ctx: u8, field: usize, v: u64) {
    C[ctx as usize][field].fetch_add(v, Relaxed);
}

fn epoch() -> &'static Instant {
    static E: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    E.get_or_init(Instant::now)
}

pub fn maybe_report() {
    let now = epoch().elapsed().as_millis() as u64;
    let last = LAST.load(Relaxed);
    if now < last + 5000 || LAST.compare_exchange(last, now, Relaxed, Relaxed).is_err() {
        return;
    }
    let names = ["other", "read", "fetch_host", "fetch_ifw"];
    let mut s = String::new();
    for (i, row) in C.iter().enumerate() {
        let v: Vec<u64> = row.iter().map(|a| a.swap(0, Relaxed)).collect();
        s += &format!(
            " {}: calls={} ms={:.1} flushes={} aside={} finish={} kb={} changed_kb={} wait_ms={:.1} same_chunks={} exact_hits={} entry_changed={} |",
            names[i], v[0], v[1] as f64 / 1e6, v[2], v[3], v[4], v[5] / 1024, v[6] / 1024, v[7] as f64 / 1e6, v[8], v[9], v[10]
        );
    }
    let b: Vec<u64> = BIND.iter().map(|a| a.swap(0, Relaxed)).collect();
    s += &format!(" bind: writes={} mb={} big_writes={} big_mb={}", b[0], b[1] >> 20, b[2], b[3] >> 20);
    tracing::info!("diag sync{s}");
}
