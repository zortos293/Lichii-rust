//! Guest surface formats and CPU-side texture work shared by the GPU and the renderer:
//! pixel formats and their block sizes, block-linear (GOB) tiling, and ASTC/BCn decoders.
//!
//! Owned by the Vulkan crew; the GPU crew reads [\`PixelFormat\`] when it decodes texture and
//! render-target descriptors.

/// Host-independent surface format, after the GPU decoded the guest's format fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    A8B8G8R8Unorm,
    A8B8G8R8Srgb,
    B8G8R8A8Unorm,
    B8G8R8A8Srgb,
    R5G6B5Unorm,
    A2B10G10R10Unorm,
    R16G16B16A16Float,
    R32G32B32A32Float,
    R11G11B10Float,
    R8Unorm,
    R32Float,
    D32Float,
    D24UnormS8Uint,
    S8UintD24Unorm,
    Bc1RgbaUnorm,
    Bc3Unorm,
    Bc7Unorm,
    Astc4x4Unorm,
    Astc8x8Unorm,
}
