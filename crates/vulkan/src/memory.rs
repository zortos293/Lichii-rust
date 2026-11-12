use std::sync::Arc;
use std::sync::atomic::Ordering;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::device::Device;
use crate::error::{Result, VkResultExt};

/// Device memory as the texture cache's collector and the frame view see it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    /// Driver-reported usage of the device-local heaps minus the allocator's unused block
    /// space, which frees only when a whole block empties (NX2_NOTES row 114).
    pub used: u64,
    /// The driver's budget for the device-local heaps, or their size without
    /// `VK_EXT_memory_budget`. Collector thresholds derive from this (NX2_NOTES row 119).
    pub budget: u64,
    /// Total size of the device-local heaps.
    pub device_local: u64,
    /// Allocated block space no allocation uses.
    pub slack: u64,
}

fn allocate(
    device: &Device,
    name: &str,
    requirements: vk::MemoryRequirements,
    location: MemoryLocation,
    linear: bool,
) -> Result<Allocation> {
    let allocation = device.allocator.lock().allocate(&AllocationCreateDesc {
        name,
        requirements,
        location,
        linear,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    device
        .allocated
        .fetch_add(allocation.size(), Ordering::Relaxed);
    Ok(allocation)
}

fn free(device: &Device, allocation: Allocation) {
    device
        .allocated
        .fetch_sub(allocation.size(), Ordering::Relaxed);
    if let Err(error) = device.allocator.lock().free(allocation) {
        tracing::error!("freeing device memory: {error}");
    }
}

impl Device {
    /// Current device memory use and budget. Cheap enough to call a few times a second.
    pub fn memory_usage(&self) -> MemoryUsage {
        let heaps = &self.memory_properties.memory_heaps
            [..self.memory_properties.memory_heap_count as usize];
        let local = |i: usize| heaps[i].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL);
        let device_local = (0..heaps.len())
            .filter(|&i| local(i))
            .map(|i| heaps[i].size)
            .sum();
        let capacity = self.allocator.lock().capacity();
        let slack = capacity.saturating_sub(self.allocated.load(Ordering::Relaxed));
        if !self.features.extensions.memory_budget {
            let used = self.allocated.load(Ordering::Relaxed);
            return MemoryUsage {
                used,
                budget: device_local,
                device_local,
                slack,
            };
        }
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut budget);
        // SAFETY: the device supports VK_EXT_memory_budget.
        unsafe {
            self.instance
                .raw
                .get_physical_device_memory_properties2(self.physical, &mut properties)
        };
        let driver_used: u64 = (0..heaps.len())
            .filter(|&i| local(i))
            .map(|i| budget.heap_usage[i])
            .sum();
        let budget_bytes = (0..heaps.len())
            .filter(|&i| local(i))
            .map(|i| budget.heap_budget[i])
            .sum();
        MemoryUsage {
            used: driver_used.saturating_sub(slack),
            budget: budget_bytes,
            device_local,
            slack,
        }
    }
}

/// A buffer with its memory. Dropping it frees both, so drop it only once the GPU is done with
/// it ([`crate::scheduler::Scheduler::retire`]).
pub struct Buffer {
    device: Arc<Device>,
    raw: vk::Buffer,
    allocation: Option<Allocation>,
    size: u64,
}

impl Buffer {
    pub fn new(
        device: &Arc<Device>,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid create info.
        let raw = unsafe {
            device
                .raw
                .create_buffer(&info, None)
                .call("vkCreateBuffer")?
        };
        // SAFETY: `raw` was just created on this device.
        let requirements = unsafe { device.raw.get_buffer_memory_requirements(raw) };
        let allocation = match allocate(device, name, requirements, location, true) {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { device.raw.destroy_buffer(raw, None) };
                return Err(error);
            }
        };
        // SAFETY: the allocation satisfies the buffer's requirements.
        let bound = unsafe {
            device
                .raw
                .bind_buffer_memory(raw, allocation.memory(), allocation.offset())
                .call("vkBindBufferMemory")
        };
        let buffer = Self {
            device: device.clone(),
            raw,
            allocation: Some(allocation),
            size,
        };
        bound.map(|()| buffer)
    }

    pub fn raw(&self) -> vk::Buffer {
        self.raw
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// The buffer's bytes when it lives in host-visible memory.
    pub fn mapped(&self) -> Option<&[u8]> {
        let slice = self.allocation.as_ref()?.mapped_slice()?;
        Some(&slice[..self.size as usize])
    }

    pub fn mapped_mut(&mut self) -> Option<&mut [u8]> {
        let size = self.size as usize;
        let slice = self.allocation.as_mut()?.mapped_slice_mut()?;
        Some(&mut slice[..size])
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses the buffer.
        unsafe { self.device.raw.destroy_buffer(self.raw, None) };
        if let Some(allocation) = self.allocation.take() {
            free(&self.device, allocation);
        }
    }
}

/// How to create an [`Image`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDesc {
    pub image_type: vk::ImageType,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: vk::SampleCountFlags,
    pub usage: vk::ImageUsageFlags,
    pub flags: vk::ImageCreateFlags,
}

impl ImageDesc {
    /// A single-level, single-sample 2D image.
    pub fn d2(format: vk::Format, width: u32, height: u32, usage: vk::ImageUsageFlags) -> Self {
        Self {
            image_type: vk::ImageType::TYPE_2D,
            format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1,
            usage,
            flags: vk::ImageCreateFlags::empty(),
        }
    }
}

/// Aspect flags for a Vulkan format.
pub fn aspect_of(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::D32_SFLOAT | vk::Format::X8_D24_UNORM_PACK32 => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// An optimally tiled image in device-local memory.
pub struct Image {
    device: Arc<Device>,
    raw: vk::Image,
    allocation: Option<Allocation>,
    desc: ImageDesc,
}

impl Image {
    pub fn new(device: &Arc<Device>, desc: &ImageDesc, name: &str) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .flags(desc.flags)
            .image_type(desc.image_type)
            .format(desc.format)
            .extent(desc.extent)
            .mip_levels(desc.mip_levels)
            .array_layers(desc.array_layers)
            .samples(desc.samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(desc.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid create info.
        let raw = unsafe { device.raw.create_image(&info, None).call("vkCreateImage")? };
        // SAFETY: `raw` was just created on this device.
        let requirements = unsafe { device.raw.get_image_memory_requirements(raw) };
        let allocation = match allocate(device, name, requirements, MemoryLocation::GpuOnly, false)
        {
            Ok(allocation) => allocation,
            Err(error) => {
                // SAFETY: the image is unused.
                unsafe { device.raw.destroy_image(raw, None) };
                return Err(error);
            }
        };
        // SAFETY: the allocation satisfies the image's requirements.
        let bound = unsafe {
            device
                .raw
                .bind_image_memory(raw, allocation.memory(), allocation.offset())
                .call("vkBindImageMemory")
        };
        let image = Self {
            device: device.clone(),
            raw,
            allocation: Some(allocation),
            desc: *desc,
        };
        bound.map(|()| image)
    }

    pub fn raw(&self) -> vk::Image {
        self.raw
    }

    pub fn desc(&self) -> &ImageDesc {
        &self.desc
    }

    /// Bytes of device memory backing the image.
    pub fn memory_size(&self) -> u64 {
        self.allocation.as_ref().map_or(0, Allocation::size)
    }

    /// Every level and layer of the image.
    pub fn full_range(&self) -> vk::ImageSubresourceRange {
        vk::ImageSubresourceRange {
            aspect_mask: aspect_of(self.desc.format),
            base_mip_level: 0,
            level_count: self.desc.mip_levels,
            base_array_layer: 0,
            layer_count: self.desc.array_layers,
        }
    }

    /// A view of the whole image with the image's own format.
    pub fn view(&self, view_type: vk::ImageViewType) -> Result<ImageView> {
        ImageView::new(
            &self.device,
            self.raw,
            view_type,
            self.desc.format,
            self.full_range(),
        )
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses the image.
        unsafe { self.device.raw.destroy_image(self.raw, None) };
        if let Some(allocation) = self.allocation.take() {
            free(&self.device, allocation);
        }
    }
}

/// An image view. It owns only the view: the image it was created from must outlive it, and
/// the view's format and range must be compatible with that image.
pub struct ImageView {
    device: Arc<Device>,
    raw: vk::ImageView,
}

impl ImageView {
    /// Creates a view of `range` of `image` read as `format`.
    pub fn new(
        device: &Arc<Device>,
        image: vk::Image,
        view_type: vk::ImageViewType,
        format: vk::Format,
        range: vk::ImageSubresourceRange,
    ) -> Result<Self> {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .subresource_range(range);
        // SAFETY: `image` is a live image of this device compatible with the view.
        let raw = unsafe {
            device
                .raw
                .create_image_view(&info, None)
                .call("vkCreateImageView")?
        };
        Ok(Self {
            device: device.clone(),
            raw,
        })
    }

    pub fn raw(&self) -> vk::ImageView {
        self.raw
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses the view.
        unsafe { self.device.raw.destroy_image_view(self.raw, None) };
    }
}
