//! The contract between the GPU and the renderer.
//!
//! The GPU crew grows this module: engines call a [\`Rasterizer\`] for draws, dispatches,
//! clears, copies, queries and cache maintenance, and hand finished frames to it as
//! [\`FramebufferConfig\`]s. The Vulkan crew implements it.

use lichii_common::DAddr;
use lichii_texture::PixelFormat;

/// A guest frame ready to present.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FramebufferConfig {
    /// Device address of the first byte.
    pub address: DAddr,
    pub width: u32,
    pub height: u32,
    /// Row pitch in bytes for pitch-linear surfaces.
    pub stride: u32,
    pub format: PixelFormat,
    /// \`Some(log2 block height in GOBs)\` for block-linear surfaces.
    pub block_height_log2: Option<u32>,
    pub flip_vertical: bool,
    pub crop: Option<CropRect>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

/// Implemented by the renderer; called from the GPU thread.
pub trait Rasterizer: Send {
    /// Presents a guest frame (scaled, filtered and anti-aliased per settings).
    fn present(&mut self, framebuffer: &FramebufferConfig);
    /// The guest CPU wrote \`[addr, addr + size)\`; drop or re-upload cached copies.
    fn invalidate_region(&mut self, addr: DAddr, size: u64);
    /// Write GPU-modified data in \`[addr, addr + size)\` back to guest memory.
    fn flush_region(&mut self, addr: DAddr, size: u64);
    /// Block until the host GPU finished all submitted work.
    fn wait_for_idle(&mut self);
}
