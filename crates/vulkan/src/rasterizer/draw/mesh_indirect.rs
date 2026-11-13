//! Indirect task/mesh draws. A guest command has the layout of an ordinary indirect draw, whose
//! first word is the group count, so a compute pass (`mesh_indirect.comp`) rewrites the commands
//! as `VkDrawMeshTasksIndirectCommandEXT`s in a scratch buffer the draw then reads.

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_gpu::backend::IndirectDraw;

use super::host_compute::{HostCompute, storage};
use super::{IndirectBuffers, Warning};
use crate::buffer_cache::BufferBinding;
use crate::device::Device;
use crate::error::Result;
use crate::memory::Buffer;
use crate::rasterizer::VulkanRasterizer;
use crate::shaders;

/// Guest commands one pass rewrites; later commands are dropped.
const MAX_DRAWS: u32 = 65535;
const COMMAND_SIZE: u32 = size_of::<vk::DrawMeshTasksIndirectCommandEXT>() as u32;
const COMMAND_BYTES: u64 = MAX_DRAWS as u64 * COMMAND_SIZE as u64;
const GROUP: u32 = 64;

/// The push constants of `mesh_indirect.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    command_word: u32,
    stride_words: u32,
    max_draws: u32,
    count_word: u32,
    limit: u32,
}

/// The rewrite pipeline and the scratch buffer of host commands. Every draw reuses the scratch
/// buffer, ordered after the draws that read it before.
pub(super) struct MeshIndirectPass {
    compute: HostCompute,
    commands: Buffer,
}

impl MeshIndirectPass {
    fn new(device: &Arc<Device>) -> Result<Self> {
        Ok(Self {
            compute: HostCompute::new(
                device,
                shaders::MESH_INDIRECT_COMP,
                3,
                size_of::<Params>() as u32,
            )?,
            commands: Buffer::new(
                device,
                COMMAND_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                MemoryLocation::GpuOnly,
                "mesh indirect commands",
            )?,
        })
    }
}

impl VulkanRasterizer {
    /// Records the pass rewriting the commands of `draw` from `inputs`, launching at most
    /// `limit` groups a command, and returns the host commands. They keep the guest draw count
    /// when the device takes one; otherwise the commands past it launch nothing.
    pub(super) fn rewrite_mesh_indirect(
        &mut self,
        cmd: vk::CommandBuffer,
        draw: &IndirectDraw,
        inputs: &IndirectBuffers,
        limit: u32,
    ) -> Result<IndirectBuffers> {
        if draw.max_draws > MAX_DRAWS {
            self.draw.warn_once(
                Warning::IndirectMesh,
                "indirect task/mesh draws launch at most 65535 commands; dropping the rest",
            );
        }
        if self.draw.mesh_indirect.is_none() {
            self.draw.mesh_indirect = Some(MeshIndirectPass::new(&self.device)?);
        }
        let pass = self.draw.mesh_indirect.as_ref().expect("just created");
        let scratch = pass.commands.raw();
        let set = self
            .draw
            .descriptors
            .allocate(&mut self.scheduler, pass.compute.set_layout)?;
        let align = self.buffer_cache.storage_alignment();
        let (commands, command_byte) = storage(inputs.commands, align);
        let (count, count_word) = inputs.count.map_or((commands, u32::MAX), |count| {
            let (info, byte) = storage(count, align);
            (info, byte / 4)
        });
        let max_draws = draw.max_draws.min(MAX_DRAWS);
        let params = Params {
            command_word: command_byte / 4,
            stride_words: draw.stride / 4,
            max_draws,
            count_word,
            limit,
        };
        let infos = [
            commands,
            count,
            vk::DescriptorBufferInfo {
                buffer: scratch,
                offset: 0,
                range: COMMAND_BYTES,
            },
        ];
        pass.compute.dispatch(
            cmd,
            set,
            &infos,
            bytemuck::bytes_of(&params),
            max_draws.div_ceil(GROUP),
            (
                vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
        );
        Ok(IndirectBuffers {
            commands: BufferBinding {
                buffer: scratch,
                offset: 0,
                size: u64::from(max_draws * COMMAND_SIZE),
            },
            count: inputs
                .count
                .filter(|_| self.device.features.draw_indirect_count),
            indices: None,
            stride: COMMAND_SIZE,
            max_draws,
        })
    }
}
