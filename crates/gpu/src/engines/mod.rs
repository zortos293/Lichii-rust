//! GPU engines bound to channel subchannels.
//!
//! Each engine executes its methods on the GPU's back thread through an [`EngineContext`]: the
//! channel's GPU memory, the renderer and host1x. Engines never block on the front thread.

pub mod fermi_2d;
pub mod kepler_compute;
pub mod kepler_memory;
pub mod maxwell_3d;
pub mod maxwell_dma;
pub mod upload;

use lichii_texture::block_linear::GobLayout;

use crate::backend::{ChannelId, Rasterizer};
use crate::gmmu::Gmmu;
use crate::host1x::Syncpoints;

/// Word index (method number) of a register field: `reg_index!(Regs, draw.begin)`.
#[macro_export]
macro_rules! reg_index {
    ($regs:ty, $($field:tt)+) => {
        (core::mem::offset_of!($regs, $($field)+) / 4) as u32
    };
}

/// The engine a class id selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EngineKind {
    Maxwell3d,
    Compute,
    Dma,
    Fermi2d,
    InlineToMemory,
}

impl EngineKind {
    /// Engine for a class bound with SET_OBJECT. Ampere classes run on the Maxwell engines
    /// (NX2_NOTES row 31); unknown classes (incl. NVN's class 0) bind nothing.
    pub fn from_class(class: u32) -> Option<Self> {
        Some(match class {
            0xb197 | 0xc097 | 0xc397 | 0xc597 | 0xc697 | 0xc797 => Self::Maxwell3d,
            0xb1c0 | 0xc0c0 | 0xc3c0 | 0xc5c0 | 0xc6c0 | 0xc7c0 => Self::Compute,
            0xb0b5 | 0xc0b5 | 0xc3b5 | 0xc5b5 | 0xc6b5 | 0xc7b5 => Self::Dma,
            0x902d => Self::Fermi2d,
            0xa140 => Self::InlineToMemory,
            _ => return None,
        })
    }
}

/// What a back-thread engine method may touch.
pub struct EngineContext<'a> {
    pub channel: ChannelId,
    pub memory: &'a Gmmu,
    pub rasterizer: &'a mut dyn Rasterizer,
    pub syncpoints: &'a Syncpoints,
    /// GOB byte order of block-linear surfaces (Switch 2 swaps bits 5 and 6).
    pub gob_layout: GobLayout,
}

#[cfg(test)]
pub(crate) mod testing;
