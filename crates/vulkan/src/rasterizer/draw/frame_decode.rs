//! Block-linear framebuffers decoded on the GPU: the guest's tiled bytes go to the staging ring as
//! they are, and `block_linear_unswizzle.comp` writes them pitch-linear into a buffer the
//! present copies into its frame image.

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_texture::block_linear::{GobLayout, Tiling};

use crate::buffer_cache::BufferBinding;
use crate::device::Device;
use crate::error::Result;
use crate::host_compute::{HostCompute, storage};
use crate::memory::Buffer;
use crate::rasterizer::VulkanRasterizer;
use crate::shaders;

/// The workgroup size of `block_linear_unswizzle.comp`.
const GROUP: [u32; 2] = [32, 8];

/// The push constants of `block_linear_unswizzle.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    row_words: u32,
    height: u32,
    row_gobs: u32,
    block_height_log2: u32,
    pitch_words: u32,
    switch2_layout: u32,
    tiled_offset_words: u32,
    pad: u32,
}

/// The unswizzle pipeline and the pitch-linear buffer it writes, grown to the largest frame.
pub(super) struct FrameDecodePass {
    compute: HostCompute,
    linear: Option<Buffer>,
}

impl FrameDecodePass {
    fn new(device: &Arc<Device>) -> Result<Self> {
        Ok(Self {
            compute: HostCompute::new(
                device,
                shaders::BLOCK_LINEAR_UNSWIZZLE_COMP,
                2,
                size_of::<Params>() as u32,
            )?,
            linear: None,
        })
    }
}

impl VulkanRasterizer {
    /// Records the pass decoding the 2D block-linear level `tiling` from its tiled bytes at
    /// `tiled`, and returns the buffer holding its rows, from offset 0, and the bytes between
    /// rows (a whole number of words).
    pub(in crate::rasterizer) fn decode_block_linear(
        &mut self,
        cmd: vk::CommandBuffer,
        tiled: BufferBinding,
        tiling: &Tiling,
    ) -> Result<(vk::Buffer, u32)> {
        let pitch_words = (tiling.width * tiling.bytes_per_block).div_ceil(4);
        let size = u64::from(pitch_words) * 4 * u64::from(tiling.height);
        if self.draw.frame_decode.is_none() {
            self.draw.frame_decode = Some(FrameDecodePass::new(&self.device)?);
        }
        let pass = self.draw.frame_decode.as_mut().expect("just created");
        if pass
            .linear
            .as_ref()
            .is_none_or(|buffer| buffer.size() < size)
        {
            let buffer = Buffer::new(
                &self.device,
                size,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::GpuOnly,
                "decoded framebuffer",
            )?;
            if let Some(old) = pass.linear.replace(buffer) {
                self.scheduler.retire(old);
            }
        }
        let linear = pass.linear.as_ref().expect("just ensured").raw();
        let set = self
            .draw
            .descriptors
            .allocate(&mut self.scheduler, pass.compute.set_layout)?;
        let (tiled, offset) = storage(tiled, self.buffer_cache.storage_alignment());
        let params = Params {
            row_words: pitch_words,
            height: tiling.height,
            row_gobs: tiling.row_gobs,
            block_height_log2: tiling.block_height_log2,
            pitch_words,
            switch2_layout: u32::from(tiling.layout == GobLayout::Switch2),
            tiled_offset_words: offset / 4,
            pad: 0,
        };
        let buffers = [
            tiled,
            vk::DescriptorBufferInfo {
                buffer: linear,
                offset: 0,
                range: size,
            },
        ];
        pass.compute.dispatch(
            cmd,
            set,
            &buffers,
            bytemuck::bytes_of(&params),
            [
                pitch_words.div_ceil(GROUP[0]),
                tiling.height.div_ceil(GROUP[1]),
                1,
            ],
            (
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_READ,
            ),
        );
        Ok((linear, pitch_words * 4))
    }
}
