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

use super::{IndirectBuffers, Warning};
use crate::buffer_cache::{BufferBinding, IndexBinding};
use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::memory::Buffer;
use crate::present::passes::shader_module;
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
    device: Arc<Device>,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    scratch: Buffer,
}

impl AssemblyPass {
    fn new(device: &Arc<Device>) -> Result<Self> {
        let scratch = Buffer::new(
            device,
            COMMAND_BYTES + INDEX_BYTES,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::INDIRECT_BUFFER,
            MemoryLocation::GpuOnly,
            "indirect assembly",
        )?;
        let mut this = Self {
            device: device.clone(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            scratch,
        };
        let bindings: [_; 5] = std::array::from_fn(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: valid create info on a live device; Drop destroys what was created so far.
        this.set_layout = unsafe {
            device
                .raw
                .create_descriptor_set_layout(&info, None)
                .call("vkCreateDescriptorSetLayout")?
        };
        let set_layouts = [this.set_layout];
        let ranges = [vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: size_of::<Params>() as u32,
        }];
        let info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&ranges);
        // SAFETY: as above, referencing the live set layout.
        this.layout = unsafe {
            device
                .raw
                .create_pipeline_layout(&info, None)
                .call("vkCreatePipelineLayout")?
        };
        this.module = shader_module(device, shaders::INDIRECT_ASSEMBLE_COMP)?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(this.module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(this.layout);
        // SAFETY: the module is valid SPIR-V matching the live layout.
        let pipelines = unsafe {
            device
                .raw
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
        };
        this.pipeline = match pipelines {
            Ok(pipelines) => pipelines[0],
            Err((_, result)) => {
                return Err(crate::Error::Vulkan {
                    call: "vkCreateComputePipelines",
                    result,
                });
            }
        };
        Ok(this)
    }
}

impl Drop for AssemblyPass {
    fn drop(&mut self) {
        self.device.wait_idle();
        // SAFETY: the device is idle; destroying null handles is a no-op.
        unsafe {
            self.device.raw.destroy_pipeline(self.pipeline, None);
            self.device.raw.destroy_shader_module(self.module, None);
            self.device.raw.destroy_pipeline_layout(self.layout, None);
            self.device
                .raw
                .destroy_descriptor_set_layout(self.set_layout, None);
        }
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
        let limit = (buffer.limit_addr().wrapping_add(1).saturating_sub(start) / width)
            .min(u64::from(u32::MAX));
        if start == 0 || limit == 0 {
            return Ok(None);
        }
        let align = self.buffer_cache.storage_alignment();
        let binding = self.buffer_cache.storage_buffer(
            &mut self.scheduler,
            &mut self.staging,
            gmmu,
            start,
            (limit * width).next_multiple_of(4),
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
        let (layout, set_layout, pipeline) = (pass.layout, pass.set_layout, pass.pipeline);
        let scratch = pass.scratch.raw();
        let set = self
            .draw
            .descriptors
            .allocate(&mut self.scheduler, set_layout)?;
        let align = self.buffer_cache.storage_alignment();
        let storage = |binding: BufferBinding| {
            let offset = binding.offset & !(align - 1);
            let info = vk::DescriptorBufferInfo {
                buffer: binding.buffer,
                offset,
                range: binding.offset - offset + binding.size,
            };
            (info, (binding.offset - offset) as u32)
        };
        let (commands, command_byte) = storage(inputs.commands);
        let (count, count_word) = inputs.count.map_or((commands, u32::MAX), |count| {
            let (info, byte) = storage(count);
            (info, byte / 4)
        });
        let (indices, index_byte, index_width, index_limit) = match inputs.indices {
            Some(guest) => {
                let (info, byte) = storage(guest.binding);
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
        let writes: [_; 5] = std::array::from_fn(|binding| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&infos[binding]))
        });
        let host_reads = (
            vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::INDEX_INPUT,
            vk::AccessFlags2::INDIRECT_COMMAND_READ | vk::AccessFlags2::INDEX_READ,
        );
        let before = vk::MemoryBarrier2::default()
            .src_stage_mask(host_reads.0)
            .src_access_mask(host_reads.1)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE);
        let after = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(host_reads.0)
            .dst_access_mask(host_reads.1);
        let raw = &self.device.raw;
        // SAFETY: the set is live and unused by recorded work; `cmd` is recording outside any
        // rendering, and every bound buffer lives until the scheduler's current tick completes.
        unsafe {
            raw.update_descriptor_sets(&writes, &[]);
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[before]),
            );
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
            raw.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            raw.cmd_push_constants(
                cmd,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&params),
            );
            raw.cmd_dispatch(cmd, max_draws, 1, 1);
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[after]),
            );
        }
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
