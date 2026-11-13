//! Dispatches whose grid GPU work may have written ([`ComputeContext::indirect_grid`]): a
//! one-invocation pass (`dispatch_grid.comp`) unpacks the descriptor's packed grid words into a
//! `VkDispatchIndirectCommand`, which `vkCmdDispatchIndirect` then reads.
//!
//! [`ComputeContext::indirect_grid`]: lichii_gpu::backend::ComputeContext::indirect_grid

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;

use super::host_compute::{HostCompute, storage};
use crate::buffer_cache::BufferBinding;
use crate::device::Device;
use crate::error::Result;
use crate::memory::Buffer;
use crate::rasterizer::VulkanRasterizer;
use crate::shaders;

const COMMAND_SIZE: u64 = size_of::<vk::DispatchIndirectCommand>() as u64;

/// The push constants of `dispatch_grid.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    limit: [u32; 3],
    grid_word: u32,
}

/// The unpack pipeline and the command it writes. Every dispatch reuses the command, ordered
/// after the dispatches that read it before.
pub(super) struct GridPass {
    compute: HostCompute,
    command: Buffer,
}

impl GridPass {
    fn new(device: &Arc<Device>) -> Result<Self> {
        Ok(Self {
            compute: HostCompute::new(
                device,
                shaders::DISPATCH_GRID_COMP,
                2,
                size_of::<Params>() as u32,
            )?,
            command: Buffer::new(
                device,
                COMMAND_SIZE,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                MemoryLocation::GpuOnly,
                "dispatch grid",
            )?,
        })
    }
}

impl VulkanRasterizer {
    /// Records the pass unpacking the grid words at byte `byte` of `grid` and returns the buffer
    /// holding the `VkDispatchIndirectCommand`, at offset 0.
    pub(super) fn unpack_grid(
        &mut self,
        cmd: vk::CommandBuffer,
        grid: BufferBinding,
        byte: u32,
    ) -> Result<vk::Buffer> {
        if self.draw.grid.is_none() {
            self.draw.grid = Some(GridPass::new(&self.device)?);
        }
        let pass = self.draw.grid.as_ref().expect("just created");
        let set = self
            .draw
            .descriptors
            .allocate(&mut self.scheduler, pass.compute.set_layout)?;
        let (grid, offset) = storage(grid, self.buffer_cache.storage_alignment());
        let command = pass.command.raw();
        let params = Params {
            limit: self.device.properties.limits.max_compute_work_group_count,
            grid_word: (offset + byte) / 4,
        };
        let buffers = [
            grid,
            vk::DescriptorBufferInfo {
                buffer: command,
                offset: 0,
                range: COMMAND_SIZE,
            },
        ];
        pass.compute.dispatch(
            cmd,
            set,
            &buffers,
            bytemuck::bytes_of(&params),
            1,
            (
                vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
        );
        Ok(command)
    }
}
