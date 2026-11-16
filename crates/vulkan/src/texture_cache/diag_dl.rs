use std::cell::Cell;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use super::CachedImage;

type Counts = Option<(Instant, HashMap<String, u64>)>;
static COUNTS: Mutex<Counts> = Mutex::new(None);

thread_local! {
    static RAST: Cell<&'static str> = const { Cell::new("trait-flush") };
    static CACHE: Cell<&'static str> = const { Cell::new("?") };
    static REASON: Cell<&'static str> = const { Cell::new("-") };
}

pub fn set_rast(s: &'static str) {
    RAST.with(|c| c.set(s));
}
pub fn set_cache(s: &'static str) {
    CACHE.with(|c| c.set(s));
}
pub fn set_reason(s: &'static str) {
    REASON.with(|c| c.set(s));
}
pub fn reason() -> &'static str {
    REASON.with(Cell::get)
}

pub fn desc(i: &CachedImage) -> String {
    let n = &i.info;
    let (daddr, size) = i.segments.first().map_or((0, 0), |s| (s.daddr, s.size));
    format!(
        "{:?} {:?} {}x{}x{} l{} L{} s{} {:?} @{:#x}+{:#x} seg{} gm{} cd{} ko{}",
        n.format,
        n.kind,
        n.width,
        n.height,
        n.depth,
        n.layers,
        n.levels,
        n.samples,
        n.layout,
        daddr,
        size,
        i.segments.len(),
        i.gpu_modified as u8,
        i.cpu_dirty as u8,
        i.keep_out.len()
    )
}

pub fn download(i: &CachedImage) {
    let key = format!(
        "DL {} [{}] {}",
        CACHE.with(Cell::get),
        RAST.with(Cell::get),
        desc(i)
    );
    hit(key);
}

pub fn hit(key: String) {
    let mut g = COUNTS.lock().unwrap();
    let (start, map) = g.get_or_insert_with(|| (Instant::now(), HashMap::new()));
    *map.entry(key).or_default() += 1;
    if start.elapsed().as_secs() >= 5 {
        let mut v: Vec<_> = map.drain().collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        let total: u64 = v.iter().filter(|e| e.0.starts_with("DL")).map(|e| e.1).sum();
        tracing::warn!(target: "diag_dl", "---- {total} downloads in {:?}, {} keys", start.elapsed(), v.len());
        for (k, c) in v.iter().take(25) {
            tracing::warn!(target: "diag_dl", "{c:>6} {k}");
        }
        *start = Instant::now();
    }
}
