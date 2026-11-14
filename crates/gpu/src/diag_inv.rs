//! Diagnostic only (diag/thrash): which GPU-crate writer is invalidating renderer caches.

use std::sync::atomic::{AtomicU8, Ordering};

pub const OTHER: u8 = 0;
pub const I2M: u8 = 1;
pub const CB: u8 = 2;
pub const COMPUTE_INLINE: u8 = 3;
pub const BLIT_2D: u8 = 4;
pub const DMA: u8 = 5;

static SOURCE: AtomicU8 = AtomicU8::new(OTHER);

pub fn set(source: u8) {
    SOURCE.store(source, Ordering::Relaxed);
}

pub fn take() -> &'static str {
    match SOURCE.swap(OTHER, Ordering::Relaxed) {
        I2M => "inv:i2m",
        CB => "inv:cb",
        COMPUTE_INLINE => "inv:compute-inline",
        BLIT_2D => "inv:2d",
        DMA => "inv:dma",
        _ => "inv:other",
    }
}
