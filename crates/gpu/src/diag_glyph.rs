//! Diagnostic only (diag/glyph): counts inline-to-memory and copy-engine uploads by layout and
//! logs the first upload to each destination.

use std::collections::HashSet;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

pub static I2M_LINEAR: AtomicU64 = AtomicU64::new(0);
pub static I2M_BLOCK: AtomicU64 = AtomicU64::new(0);
pub static I2M_BLOCK_BYTES: AtomicU64 = AtomicU64::new(0);
pub static DMA_FLAT: AtomicU64 = AtomicU64::new(0);
pub static DMA_PITCH_PITCH: AtomicU64 = AtomicU64::new(0);
pub static DMA_PITCH_BLOCK: AtomicU64 = AtomicU64::new(0);
pub static DMA_BLOCK_PITCH: AtomicU64 = AtomicU64::new(0);
pub static DMA_BLOCK_BLOCK: AtomicU64 = AtomicU64::new(0);
pub static DMA_TO_BLOCK_BYTES: AtomicU64 = AtomicU64::new(0);
static LAST_PRINT: AtomicU64 = AtomicU64::new(0);
static SEEN: Mutex<Option<HashSet<(u8, u64)>>> = Mutex::new(None);

fn start() -> &'static Instant {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now)
}

pub fn secs() -> f64 {
    start().elapsed().as_secs_f64()
}

pub fn first(kind: u8, key: u64) -> bool {
    let mut seen = SEEN.lock().unwrap();
    let seen = seen.get_or_insert_with(HashSet::new);
    seen.len() < 20000 && seen.insert((kind, key))
}

pub fn tick() {
    let now = secs() as u64;
    let last = LAST_PRINT.load(Ordering::Relaxed);
    if now < last + 5 || LAST_PRINT.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let g = |c: &AtomicU64| c.load(Ordering::Relaxed);
    tracing::info!(
        "DIAG glyph census t={now} i2m_linear={} i2m_block={} i2m_block_bytes={} dma_flat={} dma_pp={} dma_pb={} dma_bp={} dma_bb={} dma_to_block_bytes={}",
        g(&I2M_LINEAR),
        g(&I2M_BLOCK),
        g(&I2M_BLOCK_BYTES),
        g(&DMA_FLAT),
        g(&DMA_PITCH_PITCH),
        g(&DMA_PITCH_BLOCK),
        g(&DMA_BLOCK_PITCH),
        g(&DMA_BLOCK_BLOCK),
        g(&DMA_TO_BLOCK_BYTES),
    );
}
