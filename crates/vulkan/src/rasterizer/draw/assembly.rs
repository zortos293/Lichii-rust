//! Quads, quad strips and line loops of indirect draws, assembled on the GPU. Their counts live
//! in GPU memory, so a compute pass (`indirect_assemble.comp`) reads the guest commands, the draw
//! count and the guest indices, and writes triangle list or line strip indices plus one
//! `VkDrawIndexedIndirectCommand` per guest command into a scratch buffer the draw then reads.

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_gpu::backend::IndirectDraw;
use lichii_gpu::engines::maxwell_3d::regs::{IndexFormat, PrimitiveTopology, Regs};
use lichii_gpu::gmmu::Gmmu;

use super::host_compute::{HostCompute, storage};
use super::{IndirectBuffers, Warning};
use crate::buffer_cache::{BufferBinding, IndexBinding, index_capacity};
use crate::device::Device;
use crate::error::Result;
use crate::memory::Buffer;
use crate::rasterizer::VulkanRasterizer;
use crate::shaders;

/// Guest commands one pass assembles, the workgroup count limit every device supports; later
/// commands are dropped.
const MAX_DRAWS: u32 = 65535;
/// Host indices the scratch buffer holds for all commands of one pass. A command whose indices
/// would pass it draws nothing.
const INDEX_CAPACITY: u32 = 1 << 22;
const COMMAND_SIZE: u32 = size_of::<vk::DrawIndexedIndirectCommand>() as u32;
const COMMAND_BYTES: u64 = (MAX_DRAWS as u64 * COMMAND_SIZE as u64).next_multiple_of(1 << 16);
const INDEX_BYTES: u64 = INDEX_CAPACITY as u64 * 4;

/// The push constants of `indirect_assemble.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    command_word: u32,
    stride_words: u32,
    max_draws: u32,
    count_word: u32,
    index_byte: u32,
    index_width: u32,
    index_limit: u32,
    restart: u32,
    assembly: u32,
    capacity: u32,
}

/// The guest index buffer of an assembled indexed draw.
#[derive(Clone, Copy)]
pub(super) struct GuestIndices {
    /// Starts at the index buffer's address rounded down to the storage alignment.
    binding: BufferBinding,
    /// Byte offset of the first index in `binding`.
    byte: u32,
    width: u32,
    /// Indices the guest index buffer holds.
    limit: u32,
}

/// The assembly pipeline and the scratch buffer it writes: `COMMAND_BYTES` of commands, then
/// `INDEX_CAPACITY` u32 indices. Every pass reuses the buffer, ordered after the draws that read
/// it before.
pub(super) struct AssemblyPass {
    compute: HostCompute,
    scratch: Buffer,
}

impl AssemblyPass {
    fn new(device: &Arc<Device>) -> Result<Self> {
        Ok(Self {
            compute: HostCompute::new(
                device,
                shaders::INDIRECT_ASSEMBLE_COMP,
                5,
                size_of::<Params>() as u32,
            )?,
            scratch: Buffer::new(
                device,
                COMMAND_BYTES + INDEX_BYTES,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                MemoryLocation::GpuOnly,
                "indirect assembly",
            )?,
        })
    }
}

impl VulkanRasterizer {
    /// The guest index buffer of an assembled indexed draw; `None` skips the draw.
    pub(super) fn guest_indices(
        &mut self,
        gmmu: &Gmmu,
        regs: &Regs,
    ) -> Result<Option<GuestIndices>> {
        let buffer = &regs.index_buffer;
        let width = match buffer.format {
            IndexFormat::UNSIGNED_BYTE => 1,
            IndexFormat::UNSIGNED_SHORT => 2,
            IndexFormat::UNSIGNED_INT => 4,
            _ => return Ok(None),
        };
        let start = buffer.start_addr();
        let limit = index_capacity(buffer, width).min(u64::from(u32::MAX));
        if start == 0 || limit == 0 {
            return Ok(None);
        }
        let align = self.buffer_cache.storage_alignment();
        let binding = self.buffer_cache.storage_buffer(
            &mut self.scheduler,
            &mut self.staging,
            gmmu,
            start,
            (start + limit * width).next_multiple_of(4) - start,
            false,
        )?;
        Ok(binding.map(|binding| GuestIndices {
            binding,
            byte: (start & (align - 1)) as u32,
            width: width as u32,
            limit: limit as u32,
        }))
    }

    /// Records the pass assembling `draw` from `inputs` and returns what the host draw reads
    /// instead: the u32 index buffer and the commands for `vkCmdDrawIndexedIndirect`.
    pub(super) fn assemble_indirect(
        &mut self,
        cmd: vk::CommandBuffer,
        regs: &Regs,
        draw: &IndirectDraw,
        inputs: &IndirectBuffers,
    ) -> Result<(IndexBinding, IndirectBuffers)> {
        if draw.max_draws > MAX_DRAWS {
            self.draw.warn_once(
                Warning::IndirectAssembly,
                "indirect draws of quads and line loops assemble at most 65535 commands; dropping the rest",
            );
        }
        if self.draw.assembly.is_none() {
            self.draw.assembly = Some(AssemblyPass::new(&self.device)?);
        }
        let pass = self.draw.assembly.as_ref().expect("just created");
        let scratch = pass.scratch.raw();
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
        let (indices, index_byte, index_width, index_limit) = match inputs.indices {
            Some(guest) => {
                let (info, byte) = storage(guest.binding, align);
                (info, byte + guest.byte, guest.width, guest.limit)
            }
            None => (commands, 0, 0, 0),
        };
        let max_draws = draw.max_draws.min(MAX_DRAWS);
        let params = Params {
            command_word: command_byte / 4,
            stride_words: draw.stride / 4,
            max_draws,
            count_word,
            index_byte,
            index_width,
            index_limit,
            restart: u32::from(draw.indexed && regs.primitive_restart.enabled != 0),
            assembly: match draw.topology {
                PrimitiveTopology::QUADS => 0,
                PrimitiveTopology::QUAD_STRIP => 1,
                _ => 2,
            },
            capacity: INDEX_CAPACITY,
        };
        let infos = [
            commands,
            count,
            indices,
            vk::DescriptorBufferInfo {
                buffer: scratch,
                offset: 0,
                range: COMMAND_BYTES,
            },
            vk::DescriptorBufferInfo {
                buffer: scratch,
                offset: COMMAND_BYTES,
                range: INDEX_BYTES,
            },
        ];
        pass.compute.dispatch(
            cmd,
            set,
            &infos,
            bytemuck::bytes_of(&params),
            [max_draws, 1, 1],
            (
                vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::INDEX_INPUT,
                vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::INDEX_READ,
            ),
        );
        let index = IndexBinding {
            buffer: scratch,
            offset: COMMAND_BYTES,
            index_type: vk::IndexType::UINT32,
            first_index: 0,
            index_count: INDEX_CAPACITY,
            vertex_offset: 0,
        };
        let commands = IndirectBuffers {
            commands: BufferBinding {
                buffer: scratch,
                offset: 0,
                size: u64::from(max_draws * COMMAND_SIZE),
            },
            count: None,
            indices: None,
            stride: COMMAND_SIZE,
            max_draws,
        };
        Ok((index, commands))
    }
}
