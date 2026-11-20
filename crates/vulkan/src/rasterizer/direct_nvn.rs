//! NVN2 replay on Standard's texture cache and queue. Only the synthetic producer installs a target.

#[cfg(test)]
mod tests;

use std::sync::Arc;

use ash::vk;
use lichii_common::settings::ResolutionScale;
use lichii_common::{DAddr, GpuVAddr};
use lichii_gpu::backend::{Nvn2Submission, Rasterizer, ReplayAdmission, ReplayError};
use lichii_gpu::gmmu::Gmmu;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::GobLayout;

use super::VulkanRasterizer;
use crate::texture_cache::{ClearValue, Found, GuestLayout, ImageKind, Rect, Transfer};

const FORMAT: vk::Format = vk::Format::A8B8G8R8_UNORM_PACK32;
const MAX_SIDE: u32 = 256;
const MAX_BACKING: u64 = 256 << 10;

/// A target that the renderer registered for replay. Registration is currently test-only.
/// The owner keeps its mapping and backing stable while commands or fences can use it.
pub(super) struct ResidentTarget {
    found: Found,
    memory: Arc<Gmmu>,
    address: GpuVAddr,
    mapping_generation: u64,
    cache_generation: u64,
    backing: DAddr,
    #[cfg(test)]
    fail_after_recording: Option<std::sync::mpsc::Sender<()>>,
}

fn supported(features: vk::FormatFeatureFlags) -> bool {
    features
        .contains(vk::FormatFeatureFlags::TRANSFER_DST | vk::FormatFeatureFlags::COLOR_ATTACHMENT)
}

impl VulkanRasterizer {
    pub(super) fn replay_nvn2(
        &mut self,
        submission: &Nvn2Submission,
    ) -> Result<ReplayAdmission, ReplayError> {
        let Some(target) = &self.nvn2_target else {
            return Ok(ReplayAdmission::UseStandard);
        };
        if self.gob_layout != GobLayout::Switch2
            || submission.color_mask != 0xf
            || !supported(self.device.format_features(FORMAT))
            || !target.ready(self, submission)
        {
            return Ok(ReplayAdmission::UseStandard);
        }
        let found = target.found;
        if !self.texture_cache.keep_target(found.id) {
            return Ok(ReplayAdmission::UseStandard);
        }
        let mut tx = Transfer {
            scheduler: &mut self.scheduler,
            staging: &mut self.staging,
            memory: &self.handles.device,
            buffers: &mut self.buffer_cache,
        };
        self.texture_cache
            .clear(
                &mut tx,
                found,
                ClearValue::Color {
                    value: vk::ClearColorValue {
                        float32: submission.color,
                    },
                    mask: vk::ColorComponentFlags::RGBA,
                },
                Rect::ALL,
                0,
                1,
            )
            .map_err(|error| {
                tracing::error!(%error, "recording NVN2 clear failed");
                ReplayError
            })?;
        self.unflushed = true;
        self.activity.clears += 1;
        #[cfg(test)]
        if let Some(notify) = &target.fail_after_recording {
            notify.send(()).unwrap();
            return Err(ReplayError);
        }
        self.signal_fence(submission.fence);
        Ok(ReplayAdmission::Recorded)
    }
}

impl ResidentTarget {
    fn ready(&self, renderer: &VulkanRasterizer, submission: &Nvn2Submission) -> bool {
        if !Arc::ptr_eq(&self.memory, &submission.memory)
            || !Arc::ptr_eq(self.memory.device(), &renderer.handles.device)
            || self.address != submission.target
            || self.mapping_generation != submission.mapping_generation
            || self.mapping_generation != self.memory.generation()
            || self.cache_generation != renderer.texture_cache.generation()
            || self.found.level != 0
            || self.found.layer != 0
            || renderer.handles.device.has_cpu_writes()
        {
            return false;
        }
        let Some(state) = renderer.texture_cache.state(self.found.id) else {
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
                .contiguous_pointer(self.backing, size)
                .is_some()
            && !renderer.buffer_cache.is_gpu_written(self.backing, size)
    }
}
