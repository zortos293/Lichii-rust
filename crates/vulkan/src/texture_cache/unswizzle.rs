//! GPU deswizzle of block-linear texture uploads. Staging holds each subresource's tiled bytes
//! as the guest stores them; `texture_unswizzle.comp` writes its blocks tightly packed into the
//! staging bytes the CPU path would have filled, so the copy into the image and the GPU ASTC
//! decode read them as before. The back only copies guest memory into staging.

use std::sync::Arc;

use ash::vk;
use lichii_texture::block_linear::{GobLayout, Tiling};

use crate::buffer_cache::BufferBinding;
use crate::descriptors::DescriptorAllocator;
use crate::device::Device;
use crate::error::Result;
use crate::host_compute::{HostCompute, storage};
use crate::scheduler::Scheduler;
use crate::shaders;

/// The workgroup size of `texture_unswizzle.comp`.
const GROUP: [u32; 2] = [32, 8];

/// The push constants of `texture_unswizzle.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    row_words: u32,
    height: u32,
    depth: u32,
    row_gobs: u32,
    block_height_log2: u32,
    block_depth_log2: u32,
    switch2_layout: u32,
    tiled_offset_words: u32,
    linear_offset_words: u32,
}

/// One subresource to untile: `tiling` from staging byte `tiled` into byte `linear`.
#[derive(Clone, Copy, Debug)]
pub(super) struct UnswizzleLevel {
    pub tiling: Tiling,
    pub tiled: u64,
    pub linear: u64,
}

pub(super) struct Unswizzler {
    device: Arc<Device>,
    compute: HostCompute,
    descriptors: DescriptorAllocator,
    alignment: u64,
    max_range: u64,
}

impl Unswizzler {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let limits = &device.properties.limits;
        Ok(Self {
            device: device.clone(),
            compute: HostCompute::new(
                device,
                shaders::TEXTURE_UNSWIZZLE_COMP,
                2,
                size_of::<Params>() as u32,
            )?,
            descriptors: DescriptorAllocator::new(device)?,
            alignment: limits.min_storage_buffer_offset_alignment,
            max_range: u64::from(limits.max_storage_buffer_range),
        })
    }

    /// Whether the shader can untile `tiling`: its rows are whole words and one binding holds
    /// its tiled bytes.
    pub fn takes(&self, tiling: &Tiling) -> bool {
        (tiling.width * tiling.bytes_per_block).is_multiple_of(4)
            && tiling.size_bytes() as u64 + self.alignment <= self.max_range
    }

    /// Records the untiling of `levels` in `staging`, then a barrier that makes the linear
    /// bytes visible to transfers and compute shaders.
    pub fn record(
        &mut self,
        scheduler: &mut Scheduler,
        cmd: vk::CommandBuffer,
        staging: vk::Buffer,
        levels: &[UnswizzleLevel],
    ) -> Result<()> {
        for level in levels {
            let tiling = &level.tiling;
            let (tiled, tiled_offset) = storage(
                BufferBinding {
                    buffer: staging,
                    offset: level.tiled,
                    size: tiling.size_bytes() as u64,
                },
                self.alignment,
            );
            let (linear, linear_offset) = storage(
                BufferBinding {
                    buffer: staging,
                    offset: level.linear,
                    size: tiling.linear_size_bytes() as u64,
                },
                self.alignment,
            );
            let row_words = tiling.width * tiling.bytes_per_block / 4;
            let params = Params {
                row_words,
                height: tiling.height,
                depth: tiling.depth,
                row_gobs: tiling.row_gobs,
                block_height_log2: tiling.block_height_log2,
                block_depth_log2: tiling.block_depth_log2,
                switch2_layout: u32::from(tiling.layout == GobLayout::Switch2),
                tiled_offset_words: tiled_offset / 4,
                linear_offset_words: linear_offset / 4,
            };
            let set = self
                .descriptors
                .allocate(scheduler, self.compute.set_layout)?;
            self.compute.write_set(set, &[tiled, linear]);
            self.compute.bind(cmd, set);
            self.compute.dispatch_bound(
                cmd,
                bytemuck::bytes_of(&params),
                [
                    row_words.div_ceil(GROUP[0]),
                    tiling.height.div_ceil(GROUP[1]),
                    tiling.depth,
                ],
            );
        }
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_READ | vk::AccessFlags2::SHADER_STORAGE_READ,
            );
        // SAFETY: `cmd` is recording outside any rendering.
        unsafe {
            self.device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[barrier]),
            );
        }
        Ok(())
    }
}
