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
pub static DRAWS_DROPPED: AtomicU64 = AtomicU64::new(0);
pub static PRESENT_CACHED: AtomicU64 = AtomicU64::new(0);
pub static PRESENT_UPLOAD: AtomicU64 = AtomicU64::new(0);
type TargetKey = (u64, u32, u32, u64, u32);
static TARGETS: Mutex<Option<std::collections::HashMap<TargetKey, u64>>> = Mutex::new(None);

pub fn draw(regs: &crate::engines::maxwell_3d::Regs, renders: bool) {
    if !renders {
        DRAWS_DROPPED.fetch_add(1, Ordering::Relaxed);
    }
    let count = regs.rt_control.count();
    let key: TargetKey = (
        regs.rt[0].address(),
        count,
        regs.zeta_enable,
        if regs.zeta_enable != 0 { regs.zeta.address() } else { 0 },
        regs.depth_test_enable | (regs.stencil_enable << 1) | ((renders as u32) << 2),
    );
    *TARGETS.lock().unwrap().get_or_insert_with(Default::default).entry(key).or_default() += 1;
    if first(6, key.0 ^ (u64::from(count) << 56) ^ key.3.rotate_left(17) ^ (u64::from(key.4) << 60)) {
        let rts: Vec<String> = (0..count.min(8) as usize)
            .map(|i| {
                let t = &regs.rt[i];
                format!("{:#x}:{}x{}:fmt{:#x}:d{}", t.address(), t.width, t.height, t.format, t.depth())
            })
            .collect();
        tracing::info!(
            "DIAG glyph target t={:.1} rts=[{}] zeta_enable={} zeta={:#x} zfmt={:#x} zsize={}x{} depth_test={} stencil={} renders={} scissor0={} color_mask_common={} clip={:?}/{:?}",
            secs(),
            rts.join(" "),
            regs.zeta_enable,
            regs.zeta.address(),
            regs.zeta.format,
            regs.zeta_size.width,
            regs.zeta_size.height,
            regs.depth_test_enable,
            regs.stencil_enable,
            renders,
            regs.scissor_test[0].enable,
            regs.color_mask_common,
            regs.surface_clip.bits,
            regs.surface_clip.bits1,
        );
    }
    tick();
}

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
    let targets = TARGETS.lock().unwrap().take().unwrap_or_default();
    let mut targets: Vec<_> = targets.into_iter().collect();
    targets.sort_by(|a, b| b.1.cmp(&a.1));
    let top: Vec<String> = targets
        .iter()
        .take(10)
        .map(|((rt, n, ze, z, f), c)| format!("{c}@rt0={rt:#x}/n{n}/ze{ze}/z={z:#x}/f{f}"))
        .collect();
    tracing::info!(
        "DIAG glyph draws t={now} dropped={} present_cached={} present_upload={} targets: {}",
        g(&DRAWS_DROPPED),
        g(&PRESENT_CACHED),
        g(&PRESENT_UPLOAD),
        top.join(" ")
    );
}
