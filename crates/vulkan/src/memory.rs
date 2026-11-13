use std::sync::Arc;
use std::sync::atomic::Ordering;

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme};

use crate::device::Device;
use crate::error::{Result, VkResultExt};
use crate::scheduler::Scheduler;

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
    sparse: Option<Box<SparseMemory>>,
}

/// Device memory of a sparse image, bound subresource by subresource as they are written.
struct SparseMemory {
    /// Bytes per sparse block, and the memory types blocks may come from.
    requirements: vk::MemoryRequirements,
    /// Per aspect: block shape and mip tail.
    aspects: Vec<vk::SparseImageMemoryRequirements>,
    /// Bound subresources, `level * layers + layer`.
    bound: Vec<bool>,
    /// Layers whose mip tail is bound (only the first entry with a single mip tail).
    tails: Vec<bool>,
    allocations: Vec<Allocation>,
    bytes: u64,
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
            sparse: None,
        };
        bound.map(|()| image)
    }

    /// An image with sparse residency and no memory: reads of unbound blocks return zero (or
    /// undefined values without `residencyNonResidentStrict`) and writes to them are dropped.
    /// [`Image::bind`] gives subresources memory. The caller checks the device supports it.
    pub fn new_sparse(device: &Arc<Device>, desc: &ImageDesc) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .flags(
                desc.flags
                    | vk::ImageCreateFlags::SPARSE_BINDING
                    | vk::ImageCreateFlags::SPARSE_RESIDENCY,
            )
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
        // SAFETY: valid create info on a device with sparse residency.
        let raw = unsafe { device.raw.create_image(&info, None).call("vkCreateImage")? };
        // SAFETY: `raw` was just created on this device.
        let (requirements, aspects) = unsafe {
            (
                device.raw.get_image_memory_requirements(raw),
                device.raw.get_image_sparse_memory_requirements(raw),
            )
        };
        let single_tail = aspects.iter().any(|a| {
            a.format_properties
                .flags
                .contains(vk::SparseImageFormatFlags::SINGLE_MIPTAIL)
        });
        Ok(Self {
            device: device.clone(),
            raw,
            allocation: None,
            desc: *desc,
            sparse: Some(Box::new(SparseMemory {
                requirements,
                aspects,
                bound: vec![false; (desc.mip_levels * desc.array_layers) as usize],
                tails: vec![
                    false;
                    if single_tail {
                        1
                    } else {
                        desc.array_layers as usize
                    }
                ],
                allocations: Vec::new(),
                bytes: 0,
            })),
        })
    }

    pub fn is_sparse(&self) -> bool {
        self.sparse.is_some()
    }

    /// Gives memory to the unbound subresources of `level` in `layers` of a sparse image, as
    /// long as the image stays within `cap` bytes; returns the layers newly bound, whose
    /// contents are undefined until written. A non-sparse image has nothing to bind.
    pub fn bind(
        &mut self,
        scheduler: &mut Scheduler,
        level: u32,
        layers: std::ops::Range<u32>,
        cap: u64,
    ) -> Result<std::ops::Range<u32>> {
        let Some(sparse) = self.sparse.as_deref_mut() else {
            return Ok(0..0);
        };
        let block = sparse.requirements.alignment.max(1);
        let mut image_binds = Vec::new();
        let mut opaque_binds = Vec::new();
        let mut newly = layers.end..layers.start;
        for layer in layers {
            let index = (level * self.desc.array_layers + layer) as usize;
            if sparse.bound[index] {
                continue;
            }
            let mut binds = Vec::new();
            let mut tails = Vec::new();
            let mut bytes = 0;
            for aspect in &sparse.aspects {
                let shape = aspect.format_properties.image_granularity;
                if level >= aspect.image_mip_tail_first_lod {
                    let tail = if sparse.tails.len() == 1 {
                        0
                    } else {
                        layer as usize
                    };
                    if !sparse.tails[tail] {
                        let offset = aspect.image_mip_tail_offset
                            + tail as u64 * aspect.image_mip_tail_stride;
                        tails.push((tail, offset, aspect.image_mip_tail_size));
                        bytes += aspect.image_mip_tail_size;
                    }
                    continue;
                }
                let extent = vk::Extent3D {
                    width: (self.desc.extent.width >> level).max(1),
                    height: (self.desc.extent.height >> level).max(1),
                    depth: (self.desc.extent.depth >> level).max(1),
                };
                let size = u64::from(extent.width.div_ceil(shape.width))
                    * u64::from(extent.height.div_ceil(shape.height))
                    * u64::from(extent.depth.div_ceil(shape.depth))
                    * block;
                binds.push((aspect.format_properties.aspect_mask, extent, size));
                bytes += size;
            }
            if sparse.bytes + bytes > cap {
                break;
            }
            for (aspect_mask, extent, size) in binds {
                let allocation = allocate(
                    &self.device,
                    "sparse image block",
                    vk::MemoryRequirements {
                        size,
                        ..sparse.requirements
                    },
                    MemoryLocation::GpuOnly,
                    false,
                )?;
                image_binds.push(
                    vk::SparseImageMemoryBind::default()
                        .subresource(vk::ImageSubresource {
                            aspect_mask,
                            mip_level: level,
                            array_layer: layer,
                        })
                        .extent(extent)
                        // SAFETY: the allocation lives as long as the image (freed on drop).
                        .memory(unsafe { allocation.memory() })
                        .memory_offset(allocation.offset()),
                );
                sparse.allocations.push(allocation);
            }
            for (tail, offset, size) in tails {
                let allocation = allocate(
                    &self.device,
                    "sparse image mip tail",
                    vk::MemoryRequirements {
                        size,
                        ..sparse.requirements
                    },
                    MemoryLocation::GpuOnly,
                    false,
                )?;
                opaque_binds.push(
                    vk::SparseMemoryBind::default()
                        .resource_offset(offset)
                        .size(size)
                        // SAFETY: as above.
                        .memory(unsafe { allocation.memory() })
                        .memory_offset(allocation.offset()),
                );
                sparse.allocations.push(allocation);
                sparse.tails[tail] = true;
            }
            sparse.bytes += bytes;
            sparse.bound[index] = true;
            newly = newly.start.min(layer)..newly.end.max(layer + 1);
        }
        if !image_binds.is_empty() || !opaque_binds.is_empty() {
            scheduler.bind_sparse(self.raw, &image_binds, &opaque_binds)?;
        }
        Ok(if newly.is_empty() { 0..0 } else { newly })
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
            + self.sparse.as_ref().map_or(0, |s| s.bytes)
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

    /// A view of the whole image with the image's own format, reading channels through
    /// `components` (`HostFormat::swizzle`, which is identity whenever the format is attachable).
    pub fn view(
        &self,
        view_type: vk::ImageViewType,
        components: vk::ComponentMapping,
    ) -> Result<ImageView> {
        ImageView::new(
            &self.device,
            self.raw,
            view_type,
            self.desc.format,
            components,
            self.full_range(),
            self.view_usage(self.desc.format),
        )
    }

    /// The usage a view read as `format` must declare: `None` (the image's usage) unless the
    /// image was created with `EXTENDED_USAGE`, whose usage can include features `format`
    /// lacks, such as storage on an sRGB image stored through its linear twin.
    pub fn view_usage(&self, format: vk::Format) -> Option<vk::ImageUsageFlags> {
        if !self
            .desc
            .flags
            .contains(vk::ImageCreateFlags::EXTENDED_USAGE)
        {
            return None;
        }
        let features = self.device.format_features(format);
        let mut usage = self.desc.usage;
        for (flag, feature) in [
            (
                vk::ImageUsageFlags::STORAGE,
                vk::FormatFeatureFlags::STORAGE_IMAGE,
            ),
            (
                vk::ImageUsageFlags::COLOR_ATTACHMENT,
                vk::FormatFeatureFlags::COLOR_ATTACHMENT,
            ),
        ] {
            if !features.contains(feature) {
                usage &= !flag;
            }
        }
        Some(usage)
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses the image.
        unsafe { self.device.raw.destroy_image(self.raw, None) };
        if let Some(allocation) = self.allocation.take() {
            free(&self.device, allocation);
        }
        if let Some(sparse) = self.sparse.take() {
            for allocation in sparse.allocations {
                free(&self.device, allocation);
            }
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
    /// Creates a view of `range` of `image` read as `format`, with channels remapped by
    /// `components`, declaring `usage` when set (see [`Image::view_usage`]).
    pub fn new(
        device: &Arc<Device>,
        image: vk::Image,
        view_type: vk::ImageViewType,
        format: vk::Format,
        components: vk::ComponentMapping,
        range: vk::ImageSubresourceRange,
        usage: Option<vk::ImageUsageFlags>,
    ) -> Result<Self> {
        let mut usage_info =
            vk::ImageViewUsageCreateInfo::default().usage(usage.unwrap_or_default());
        let mut info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(format)
            .components(components)
            .subresource_range(range);
        if usage.is_some() {
            info = info.push_next(&mut usage_info);
        }
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

/// A texel buffer view, destroyed on drop.
pub struct BufferView {
    device: Arc<Device>,
    raw: vk::BufferView,
}

impl BufferView {
    /// Creates a view of `range` bytes of `buffer` from `offset`, read as `format`.
    pub fn new(
        device: &Arc<Device>,
        buffer: vk::Buffer,
        format: vk::Format,
        offset: u64,
        range: u64,
    ) -> Result<Self> {
        let info = vk::BufferViewCreateInfo::default()
            .buffer(buffer)
            .format(format)
            .offset(offset)
            .range(range);
        // SAFETY: `buffer` is a live texel buffer of this device holding the range, and the
        // caller checked the offset alignment, the element count and the format's support.
        let raw = unsafe {
            device
                .raw
                .create_buffer_view(&info, None)
                .call("vkCreateBufferView")?
        };
        Ok(Self {
            device: device.clone(),
            raw,
        })
    }

    pub fn raw(&self) -> vk::BufferView {
        self.raw
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses the view.
        unsafe { self.device.raw.destroy_buffer_view(self.raw, None) };
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        // SAFETY: the owner guarantees the GPU no longer uses the view.
        unsafe { self.device.raw.destroy_image_view(self.raw, None) };
    }
}
