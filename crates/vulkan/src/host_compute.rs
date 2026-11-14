//! Compute pipelines of host shaders whose bindings are all storage buffers, for passes that
//! rewrite guest data on the GPU before a draw or dispatch reads it.

use std::sync::Arc;

use ash::vk;

use crate::buffer_cache::BufferBinding;
use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::present::passes::shader_module;

/// Storage buffers a host pass binds at most.
const MAX_BINDINGS: usize = 5;

pub(crate) struct HostCompute {
    device: Arc<Device>,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

impl HostCompute {
    /// The pipeline of `spirv`, reading `bindings` storage buffers at set 0 and `push_size`
    /// bytes of push constants.
    pub(crate) fn new(
        device: &Arc<Device>,
        spirv: &[u32],
        bindings: usize,
        push_size: u32,
    ) -> Result<Self> {
        assert!(bindings <= MAX_BINDINGS);
        let mut this = Self {
            device: device.clone(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
        };
        let layout_bindings: [_; MAX_BINDINGS] = std::array::from_fn(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings[..bindings]);
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
            size: push_size,
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
        this.module = shader_module(device, spirv)?;
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

    /// Writes `buffers` into `set` (allocated with [`HostCompute::set_layout`]) and records
    /// `groups` workgroups with `params` as push constants. A barrier orders the earlier
    /// `reads` of the outputs before the pass writes them, and another the pass's writes before
    /// later `reads`.
    pub(crate) fn dispatch(
        &self,
        cmd: vk::CommandBuffer,
        set: vk::DescriptorSet,
        buffers: &[vk::DescriptorBufferInfo],
        params: &[u8],
        groups: [u32; 3],
        reads: (vk::PipelineStageFlags2, vk::AccessFlags2),
    ) {
        let writes: [_; MAX_BINDINGS] = std::array::from_fn(|binding| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding as u32)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(buffers.get(binding).map_or(&[], std::slice::from_ref))
        });
        let before = vk::MemoryBarrier2::default()
            .src_stage_mask(reads.0)
            .src_access_mask(reads.1)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE);
        let after = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(reads.0)
            .dst_access_mask(reads.1);
        let raw = &self.device.raw;
        // SAFETY: the set is live and unused by recorded work; `cmd` is recording outside any
        // rendering, and every bound buffer lives until the scheduler's current tick completes.
        unsafe {
            raw.update_descriptor_sets(&writes[..buffers.len()], &[]);
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[before]),
            );
            raw.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            raw.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            raw.cmd_push_constants(cmd, self.layout, vk::ShaderStageFlags::COMPUTE, 0, params);
            raw.cmd_dispatch(cmd, groups[0], groups[1], groups[2]);
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[after]),
            );
        }
    }
}

impl Drop for HostCompute {
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

/// A storage descriptor over `binding`, starting at its offset rounded down to `align`, and the
/// byte offset of `binding` in it.
pub(crate) fn storage(binding: BufferBinding, align: u64) -> (vk::DescriptorBufferInfo, u32) {
    let offset = binding.offset & !(align - 1);
    let info = vk::DescriptorBufferInfo {
        buffer: binding.buffer,
        offset,
        range: binding.offset - offset + binding.size,
    };
    (info, (binding.offset - offset) as u32)
}
