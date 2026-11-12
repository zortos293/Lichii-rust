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
    /// Every kind, in [`EngineKind::index`] order.
    pub const ALL: [Self; 5] = [
        Self::Maxwell3d,
        Self::Compute,
        Self::Dma,
        Self::Fermi2d,
        Self::InlineToMemory,
    ];
    /// Number of kinds.
    pub const COUNT: usize = Self::ALL.len();

    /// Position in [`EngineKind::ALL`], for per-kind tables.
    #[inline]
    pub const fn index(self) -> usize {
        self as usize
    }

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

/// An engine instance of one channel, executing its methods on the back thread in stream order.
pub trait Engine: Send {
    /// One method write (`method` is at or above the host class's 0x40).
    fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32);

    /// Several words for the same method in a row (a non-incrementing run, such as inline or
    /// constant buffer data), copied out of the command stream.
    fn call_multi(&mut self, ctx: &mut EngineContext<'_>, method: u32, arguments: &[u32]) {
        for &argument in arguments {
            self.call_method(ctx, method, argument);
        }
    }

    /// Finishes work the engine holds back to merge with its next methods (the 3D engine's
    /// pending draw). The back calls it before a method of another engine, a host action or a
    /// CPU-side command runs, and at the end of every submission.
    fn flush(&mut self, _ctx: &mut EngineContext<'_>) {}
}

impl Engine for kepler_memory::KeplerMemory {
    fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        kepler_memory::KeplerMemory::call_method(self, ctx, method, argument);
    }
}

impl Engine for kepler_compute::KeplerCompute {
    fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        kepler_compute::KeplerCompute::call_method(self, ctx, method, argument);
    }
}

impl Engine for maxwell_dma::MaxwellDma {
    fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        maxwell_dma::MaxwellDma::call_method(self, ctx, method, argument);
    }
}

/// Creates the engine a channel binds: `class` is the class id SET_OBJECT bound. The one place
/// engines are wired in; kinds without an engine yet return `None` and their methods are dropped.
pub fn create_engine(kind: EngineKind, class: u32) -> Option<Box<dyn Engine>> {
    match kind {
        EngineKind::InlineToMemory => Some(Box::new(kepler_memory::KeplerMemory::new())),
        EngineKind::Compute => Some(Box::new(kepler_compute::KeplerCompute::new(class))),
        EngineKind::Dma => Some(Box::new(maxwell_dma::MaxwellDma::new())),
        EngineKind::Maxwell3d | EngineKind::Fermi2d => None,
    }
}

/// Makes the engines of a channel: [`create_engine`] by default, a recording engine in tests.
pub type EngineFactory =
    std::sync::Arc<dyn Fn(EngineKind, u32) -> Option<Box<dyn Engine>> + Send + Sync>;

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
