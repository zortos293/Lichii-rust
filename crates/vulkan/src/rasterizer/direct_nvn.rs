//! Synthetic replay at the NVN2 submission boundary. No guest loader or title selects it yet.
#![cfg(test)]

mod tests;

use ash::vk;
use lichii_common::DAddr;
use lichii_common::settings::ResolutionScale;
use lichii_gpu::backend::{FenceAction, Rasterizer};
use lichii_gpu::gmmu::Gmmu;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;

use super::VulkanRasterizer;
use crate::Result;
use crate::texture_cache::{ClearValue, Found, GuestLayout, ImageKind, Rect, Transfer};

const FORMAT: vk::Format = vk::Format::A8B8G8R8_UNORM_PACK32;
const FEATURES: vk::FormatFeatureFlags = vk::FormatFeatureFlags::from_raw(
    vk::FormatFeatureFlags::TRANSFER_DST.as_raw()
        | vk::FormatFeatureFlags::COLOR_ATTACHMENT.as_raw(),
);
const MAX_SIDE: u32 = 256;
const MAX_BACKING: u64 = 256 << 10;

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
enum Route {
    #[default]
    Standard,
    Direct,
}

impl Route {
    fn negotiate(requested: bool, layout: GobLayout, features: vk::FormatFeatureFlags) -> Self {
        if requested && layout == GobLayout::Switch2 && features.contains(FEATURES) {
            Self::Direct
        } else {
            Self::Standard
        }
    }
}

/// Resolved by the synthetic producer in this renderer's cache. Its owner keeps the mapping
/// and backing stable while work is in flight. Real guest mappings cannot promise that yet.
struct ResidentTarget<'a> {
    found: Found,
    memory: &'a Gmmu,
    mapping_generation: u64,
    cache_generation: u64,
    backing: DAddr,
}

enum Command<'a> {
    ClearColor {
        target: Option<&'a ResidentTarget<'a>>,
        channels: vk::ColorComponentFlags,
        color: [f32; 4],
    },
    Unsupported,
}

fn submit(
    renderer: &mut VulkanRasterizer,
    route: Route,
    commands: &[Command<'_>],
    fence: FenceAction,
    standard: impl FnOnce(&mut VulkanRasterizer),
) -> Result<Route> {
    let supported = match commands {
        [
            Command::ClearColor {
                target: Some(target),
                channels: vk::ColorComponentFlags::RGBA,
                color,
            },
        ] if route == Route::Direct && renderer.gob_layout == GobLayout::Switch2 => {
            ready(renderer, target).then_some((target.found, *color))
        }
        _ => None,
    };
    let used = if let Some((target, color)) = supported {
        let mut tx = Transfer {
            scheduler: &mut renderer.scheduler,
            staging: &mut renderer.staging,
            memory: &renderer.handles.device,
            buffers: &mut renderer.buffer_cache,
        };
        renderer.texture_cache.clear(
            &mut tx,
            target,
            ClearValue::Color {
                value: vk::ClearColorValue { float32: color },
                mask: vk::ColorComponentFlags::RGBA,
            },
            Rect::ALL,
            0,
            1,
        )?;
        renderer.unflushed = true;
        Route::Direct
    } else {
        standard(renderer);
        Route::Standard
    };
    renderer.signal_fence(fence);
    Ok(used)
}

fn ready(renderer: &mut VulkanRasterizer, target: &ResidentTarget<'_>) -> bool {
    if !std::sync::Arc::ptr_eq(target.memory.device(), &renderer.handles.device)
        || target.mapping_generation != target.memory.generation()
        || target.cache_generation != renderer.texture_cache.generation()
        || target.found.level != 0
        || target.found.layer != 0
        || renderer.handles.device.has_cpu_writes()
    {
        return false;
    }
    let Some(state) = renderer.texture_cache.state(target.found.id) else {
        return false;
    };
    let info = state.info;
    if state.format != FORMAT
        || state.scale != ResolutionScale::X1
        || info.format != PixelFormat::A8B8G8R8Unorm
        || info.kind != ImageKind::D2
        || !matches!(
            info.layout,
            GuestLayout::BlockLinear {
                gob: GobLayout::Switch2,
                ..
            }
        )
        || (info.depth, info.layers, info.levels, info.samples) != (1, 1, 1, 1)
        || !(1..=MAX_SIDE).contains(&info.width)
        || !(1..=MAX_SIDE).contains(&info.height)
    {
        return false;
    }
    let size = info.size_bytes();
    (1..=MAX_BACKING).contains(&size)
        && renderer
            .handles
            .device
            .contiguous_pointer(target.backing, size)
            .is_some()
        && !renderer.buffer_cache.is_gpu_written(target.backing, size)
        && renderer.texture_cache.keep_target(target.found.id)
}
