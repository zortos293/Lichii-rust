//! Guest surface formats and CPU-side texture work shared by the GPU and the renderer:
//! pixel formats and their block sizes, guest format decoding, block-linear (GOB) tiling, and
//! ASTC/BCn decoders.
//!
//! Owned by the Vulkan crew; the GPU crew reads [`PixelFormat`] when it decodes texture and
//! render-target descriptors, through [`PixelFormat::from_texture`],
//! [`PixelFormat::from_render_target`] and [`PixelFormat::from_depth`].

pub mod astc;
pub mod bcn;
mod block;
pub mod block_linear;
pub mod etc;
mod format;
pub mod guest;
pub mod tic;

pub use block::ImageSizeError;
pub use format::{Aspect, Compression, FormatInfo, Numeric, PixelFormat};
