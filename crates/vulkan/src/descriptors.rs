use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::scheduler::Scheduler;

const SETS_PER_POOL: u32 = 1024;
const POOL_SIZES: [vk::DescriptorPoolSize; 8] = [
    pool_size(vk::DescriptorType::UNIFORM_BUFFER, 4096),
    pool_size(vk::DescriptorType::STORAGE_BUFFER, 4096),
    pool_size(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 4096),
    pool_size(vk::DescriptorType::SAMPLED_IMAGE, 1024),
    pool_size(vk::DescriptorType::SAMPLER, 1024),
    pool_size(vk::DescriptorType::STORAGE_IMAGE, 1024),
    pool_size(vk::DescriptorType::UNIFORM_TEXEL_BUFFER, 1024),
    pool_size(vk::DescriptorType::STORAGE_TEXEL_BUFFER, 1024),
];

const fn pool_size(ty: vk::DescriptorType, descriptor_count: u32) -> vk::DescriptorPoolSize {
    vk::DescriptorPoolSize {
        ty,
        descriptor_count,
    }
}

/// Hands out descriptor sets for one scheduler's command stream. Sets live until the tick they
/// were allocated in completes; then their whole pool is reset in one call, so no set is ever
/// freed individually.
pub struct DescriptorAllocator {
    device: Arc<Device>,
    current: vk::DescriptorPool,
    retired: VecDeque<(u64, vk::DescriptorPool)>,
}

impl DescriptorAllocator {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            current: create_pool(device)?,
            retired: VecDeque::new(),
        })
    }

    /// A set with `layout`, valid for the work recorded in `scheduler` until its next flush
    /// completes.
    pub fn allocate(
        &mut self,
        scheduler: &mut Scheduler,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet> {
        let layouts = [layout];
        for _ in 0..2 {
            let info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.current)
                .set_layouts(&layouts);
            // SAFETY: the pool belongs to this allocator; the layout is live.
            match unsafe { self.device.raw.allocate_descriptor_sets(&info) } {
                Ok(sets) => return Ok(sets[0]),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY | vk::Result::ERROR_FRAGMENTED_POOL) => {
                    self.rotate(scheduler)?;
                }
                Err(result) => {
                    return Err(crate::Error::Vulkan {
                        call: "vkAllocateDescriptorSets",
                        result,
                    });
                }
            }
        }
        Err(crate::Error::Vulkan {
            call: "vkAllocateDescriptorSets",
            result: vk::Result::ERROR_OUT_OF_POOL_MEMORY,
        })
    }

    fn rotate(&mut self, scheduler: &mut Scheduler) -> Result<()> {
        let reusable = self
            .retired
            .front()
            .is_some_and(|&(tick, _)| scheduler.is_complete(tick));
        let next = if reusable {
            let (_, pool) = self.retired.pop_front().expect("checked");
            // SAFETY: every set from this pool belongs to completed work.
            unsafe {
                self.device
                    .raw
                    .reset_descriptor_pool(pool, vk::DescriptorPoolResetFlags::empty())
                    .call("vkResetDescriptorPool")?
            };
            pool
        } else {
            create_pool(&self.device)?
        };
        let full = std::mem::replace(&mut self.current, next);
        self.retired.push_back((scheduler.tick(), full));
        Ok(())
    }
}

fn create_pool(device: &Device) -> Result<vk::DescriptorPool> {
    let info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(SETS_PER_POOL)
        .pool_sizes(&POOL_SIZES);
    // SAFETY: valid create info.
    unsafe {
        device
            .raw
            .create_descriptor_pool(&info, None)
            .call("vkCreateDescriptorPool")
    }
}

impl Drop for DescriptorAllocator {
    fn drop(&mut self) {
        self.device.wait_idle();
        // SAFETY: the device is idle, so no pending work uses these pools' sets.
        unsafe {
            self.device.raw.destroy_descriptor_pool(self.current, None);
            for (_, pool) in self.retired.drain(..) {
                self.device.raw.destroy_descriptor_pool(pool, None);
            }
        }
    }
}
