//! Render conditions the host GPU tests (`VK_EXT_conditional_rendering`): a predicate is a word
//! of a ring buffer that a pass (`conditional_reduce.comp`) fills from the occlusion queries a
//! SamplesPassed report sums, and draws recorded under a condition test it. Nothing waits on the
//! host for the queries; clears and texture draws read the pass's host-visible copy of the word.

use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_gpu::backend::{ChannelId, GpuPredicate, RenderCondition};

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::host_compute::HostCompute;
use crate::memory::Buffer;
use crate::shaders;

use super::samples::{self, SampleCounter, SampleRange};

/// Slots in the predicate ring. Each channel's latest predicate keeps its slot.
const PREDICATES: u32 = 1024;

/// The push constants of `conditional_reduce.comp`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    count: u32,
    slot: u32,
}

pub(super) struct Conditions {
    device: Arc<Device>,
    loader: ash::ext::conditional_rendering::Device,
    reduce: HostCompute,
    pool: vk::DescriptorPool,
    /// The pass's only set: every pass binds the same three buffers.
    set: vk::DescriptorSet,
    /// The 32-bit results of the queries a predicate is made from.
    results: Buffer,
    /// The predicates draws test, in device memory.
    predicates: Buffer,
    /// The same words in host-visible memory, for [`Conditions::value`].
    readback: Buffer,
    /// The scheduler tick that writes each slot.
    ticks: Box<[u64]>,
    /// Each channel's latest predicate, whose slot no new predicate takes.
    latest: Vec<(ChannelId, u32)>,
    /// The slot the next predicate tries first.
    cursor: u32,
    active: Option<RenderCondition>,
}

impl Conditions {
    /// None when the device cannot test conditions.
    pub(super) fn new(device: &Arc<Device>) -> Result<Option<Self>> {
        if !device.features.conditional_rendering {
            return Ok(None);
        }
        let reduce = HostCompute::new(
            device,
            shaders::CONDITIONAL_REDUCE_COMP,
            3,
            size_of::<Params>() as u32,
        )?;
        let sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            descriptor_count: 3,
        }];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&sizes);
        // SAFETY: valid create info on a live device.
        let pool = unsafe {
            device
                .raw
                .create_descriptor_pool(&info, None)
                .call("vkCreateDescriptorPool")?
        };
        let words = u64::from(PREDICATES) * 4;
        let mut this = Self {
            device: device.clone(),
            loader: ash::ext::conditional_rendering::Device::new(&device.instance.raw, &device.raw),
            reduce,
            pool,
            set: vk::DescriptorSet::null(),
            results: Buffer::new(
                device,
                samples::SLOTS * 4,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
                "occlusion query results",
            )?,
            predicates: Buffer::new(
                device,
                words,
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::CONDITIONAL_RENDERING_EXT,
                MemoryLocation::GpuOnly,
                "render predicates",
            )?,
            readback: Buffer::new(
                device,
                words,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                MemoryLocation::GpuToCpu,
                "render predicate readback",
            )?,
            ticks: vec![0; PREDICATES as usize].into_boxed_slice(),
            latest: Vec::new(),
            cursor: 0,
            active: None,
        };
        let layouts = [this.reduce.set_layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(this.pool)
            .set_layouts(&layouts);
        // SAFETY: the pool has room for this one set; Drop destroys the pool.
        this.set = unsafe {
            device
                .raw
                .allocate_descriptor_sets(&info)
                .call("vkAllocateDescriptorSets")?[0]
        };
        let whole = |buffer: &Buffer| vk::DescriptorBufferInfo {
            buffer: buffer.raw(),
            offset: 0,
            range: vk::WHOLE_SIZE,
        };
        let buffers = [
            whole(&this.results),
            whole(&this.predicates),
            whole(&this.readback),
        ];
        this.reduce.write_set(this.set, &buffers);
        Ok(Some(this))
    }

    /// Records the pass filling `channel`'s new latest predicate from the queries of `range`, on
    /// `cmd` outside any rendering, as part of scheduler tick `tick`. None when some of the
    /// queries' slots may already hold others.
    pub(super) fn samples(
        &mut self,
        channel: ChannelId,
        cmd: vk::CommandBuffer,
        tick: u64,
        samples: &SampleCounter,
        range: SampleRange,
    ) -> Option<GpuPredicate> {
        let device = Arc::clone(&self.device);
        let before = vk::MemoryBarrier2::default()
            .src_stage_mask(
                vk::PipelineStageFlags2::COMPUTE_SHADER
                    | vk::PipelineStageFlags2::CONDITIONAL_RENDERING_EXT,
            )
            .src_access_mask(
                vk::AccessFlags2::SHADER_STORAGE_READ
                    | vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT,
            )
            .dst_stage_mask(vk::PipelineStageFlags2::COPY | vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(
                vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE,
            );
        // SAFETY: `cmd` is recording outside any rendering.
        unsafe {
            device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[before]),
            );
        }
        let count = samples.copy_results(&device, cmd, range, self.results.raw())?;
        let copied = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ);
        let written = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
            .dst_stage_mask(
                vk::PipelineStageFlags2::CONDITIONAL_RENDERING_EXT | vk::PipelineStageFlags2::HOST,
            )
            .dst_access_mask(
                vk::AccessFlags2::CONDITIONAL_RENDERING_READ_EXT | vk::AccessFlags2::HOST_READ,
            );
        let slot = self.take_slot(channel);
        self.ticks[slot as usize] = tick;
        // SAFETY: as above.
        unsafe {
            device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[copied]),
            );
        }
        self.reduce.bind(cmd, self.set);
        let params = Params { count, slot };
        self.reduce
            .dispatch_bound(cmd, bytemuck::bytes_of(&params), [1, 1, 1]);
        // SAFETY: as above.
        unsafe {
            device.raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&[written]),
            );
        }
        Some(GpuPredicate(slot))
    }

    /// A slot no channel's latest predicate holds, which becomes `channel`'s latest.
    fn take_slot(&mut self, channel: ChannelId) -> u32 {
        let held = |slot: u32| self.latest.iter().any(|&(_, held)| held == slot);
        let mut slot = self.cursor;
        while held(slot) {
            slot = (slot + 1) % PREDICATES;
        }
        self.cursor = (slot + 1) % PREDICATES;
        match self.latest.iter_mut().find(|(owner, _)| *owner == channel) {
            Some((_, latest)) => *latest = slot,
            None => self.latest.push((channel, slot)),
        }
        slot
    }

    /// Frees the slot of `channel`'s latest predicate.
    pub(super) fn release(&mut self, channel: ChannelId) {
        self.latest.retain(|&(owner, _)| owner != channel);
    }

    /// The scheduler tick that computes `predicate`.
    pub(super) fn tick(&self, GpuPredicate(slot): GpuPredicate) -> u64 {
        self.ticks[slot as usize]
    }

    /// Whether `predicate` is nonzero, once its [`Conditions::tick`] completed.
    pub(super) fn value(&self, GpuPredicate(slot): GpuPredicate) -> bool {
        let bytes = self
            .readback
            .mapped()
            .expect("the readback buffer is host-visible");
        let at = slot as usize * 4;
        u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes")) != 0
    }

    /// The condition draws recorded from now on test.
    pub(super) fn set(&mut self, condition: Option<RenderCondition>) {
        self.active = condition;
    }

    /// Begins testing the active condition on `cmd`, inside the rendering of the draw about to
    /// be recorded; returns these conditions when it did, for [`Conditions::end`] after the draw.
    pub(super) fn begin(&self, cmd: vk::CommandBuffer) -> Option<&Self> {
        let condition = self.active?;
        let flags = match condition.inverted {
            true => vk::ConditionalRenderingFlagsEXT::INVERTED,
            false => vk::ConditionalRenderingFlagsEXT::empty(),
        };
        let info = vk::ConditionalRenderingBeginInfoEXT::default()
            .buffer(self.predicates.raw())
            .offset(u64::from(condition.predicate.0) * 4)
            .flags(flags);
        // SAFETY: `cmd` is recording inside a rendering and tests no condition yet; the
        // predicate buffer lives as long as the renderer.
        unsafe { (self.loader.fp().cmd_begin_conditional_rendering_ext)(cmd, &info) };
        Some(self)
    }

    /// Ends the test [`Conditions::begin`] began, in the same rendering.
    pub(super) fn end(&self, cmd: vk::CommandBuffer) {
        // SAFETY: `begin` began a test in this rendering of `cmd`.
        unsafe { (self.loader.fp().cmd_end_conditional_rendering_ext)(cmd) };
    }
}

impl Drop for Conditions {
    fn drop(&mut self) {
        self.device.wait_idle();
        // SAFETY: the device is idle, so no pending work uses the set; destroying the pool frees it.
        unsafe { self.device.raw.destroy_descriptor_pool(self.pool, None) };
    }
}
