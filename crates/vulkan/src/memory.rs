use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

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

/// Live allocations of one memory location.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LiveMemory {
    pub bytes: u64,
    pub count: u64,
}

/// Live allocations by memory location ([`Device::memory_counts`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryCounts {
    /// Device-local memory: images and cached buffers.
    pub gpu_only: LiveMemory,
    /// Host-visible upload memory: staging and stream rings, their oversized buffers.
    pub cpu_to_gpu: LiveMemory,
    /// Host-visible readback memory: readbacks and capture rings.
    pub gpu_to_cpu: LiveMemory,
}

/// The `VkDeviceMemory` objects the allocator holds ([`Device::memory_objects`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryObjects {
    /// Blocks shared by suballocations, of every memory type, empty ones the allocator keeps
    /// included. Resources larger than a block get a block of their own, counted here too.
    pub blocks: u64,
    /// Allocations made for one GPU-only image or buffer of 4 MiB or more.
    pub dedicated: u64,
}

/// Live allocation bytes and counts by memory location, kept where [`allocate`] and [`free`]
/// run: relaxed atomics, so counting takes no lock.
#[derive(Default)]
pub(crate) struct MemoryCounters {
    bytes: [AtomicU64; 3],
    count: [AtomicU64; 3],
    blocks: AtomicU64,
    dedicated: AtomicU64,
}

impl MemoryCounters {
    fn slot(location: MemoryLocation) -> usize {
        match location {
            MemoryLocation::Unknown | MemoryLocation::GpuOnly => 0,
            MemoryLocation::CpuToGpu => 1,
            MemoryLocation::GpuToCpu => 2,
        }
    }

    fn add(&self, location: MemoryLocation, bytes: u64) {
        let slot = Self::slot(location);
        self.bytes[slot].fetch_add(bytes, Ordering::Relaxed);
        self.count[slot].fetch_add(1, Ordering::Relaxed);
    }

    fn sub(&self, location: MemoryLocation, bytes: u64) {
        let slot = Self::slot(location);
        self.bytes[slot].fetch_sub(bytes, Ordering::Relaxed);
        self.count[slot].fetch_sub(1, Ordering::Relaxed);
    }

    /// Bytes handed out by the allocator, which reports only block capacity cheaply.
    fn total_bytes(&self) -> u64 {
        self.bytes.iter().map(|b| b.load(Ordering::Relaxed)).sum()
    }

    fn live(&self, location: MemoryLocation) -> LiveMemory {
        let slot = Self::slot(location);
        LiveMemory {
            bytes: self.bytes[slot].load(Ordering::Relaxed),
            count: self.count[slot].load(Ordering::Relaxed),
        }
    }
}

/// GPU-only images and buffers at least this large get their own `VkDeviceMemory`, so freeing
/// one returns its memory to the driver at once. In a shared block the space becomes a hole that
/// smaller long-lived resources pin until the whole block empties, which under churn it rarely
/// does: with images from 8 MiB (ZOR-384) and no buffers dedicated, 865-953 MiB sat in such holes
/// in MKW (ZOR-432).
const DEDICATED_BYTES: u64 = 4 << 20;

/// Dedicated allocations live at once. Past it, resources up to the block size share blocks
/// again, so the allocator holds at most this many `VkDeviceMemory` objects plus one per 64 MiB
/// of device block capacity and one per 32 MiB of host block capacity: about 1200 with a 10 GB
/// card's heap full, against the 4096 `maxMemoryAllocationCount` of NVIDIA and the spec.
const DEDICATED_LIMIT: u64 = 1024;

/// Shared blocks counted before the block map first grows (about 5 KiB): 16 GiB of 64 MiB blocks.
const BLOCKS_RESERVED: usize = 256;

/// The allocator and the live suballocations of each block it shares, so the blocks it holds
/// are counted without walking them.
pub(crate) struct MemoryPool {
    allocator: Allocator,
    blocks: HashMap<vk::DeviceMemory, u32>,
}

impl MemoryPool {
    pub(crate) fn new(allocator: Allocator) -> Self {
        Self {
            allocator,
            blocks: HashMap::with_capacity(BLOCKS_RESERVED),
        }
    }
}

/// What an allocation backs, which decides its memory layout and whether it may be dedicated.
#[derive(Clone, Copy)]
enum Resource {
    Buffer(vk::Buffer),
    Image(vk::Image),
    SparseBlock,
}

/// Whether a resource of `bytes` in `location` gets its own memory while `live` dedicated
/// allocations exist.
fn dedicates(location: MemoryLocation, bytes: u64, live: u64) -> bool {
    location == MemoryLocation::GpuOnly && bytes >= DEDICATED_BYTES && live < DEDICATED_LIMIT
}

fn allocate(
    device: &Device,
    name: &str,
    requirements: vk::MemoryRequirements,
    location: MemoryLocation,
    resource: Resource,
) -> Result<Allocation> {
    let counters = &device.memory_counters;
    let mut pool = device.allocator.lock();
    let dedicated = dedicates(
        location,
        requirements.size,
        counters.dedicated.load(Ordering::Relaxed),
    );
    let allocation_scheme = match resource {
        Resource::Buffer(buffer) if dedicated => AllocationScheme::DedicatedBuffer(buffer),
        Resource::Image(image) if dedicated => AllocationScheme::DedicatedImage(image),
        _ => AllocationScheme::GpuAllocatorManaged,
    };
    let allocation = pool.allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location,
        linear: matches!(resource, Resource::Buffer(_)),
        allocation_scheme,
    })?;
    if allocation.is_dedicated() {
        counters.dedicated.fetch_add(1, Ordering::Relaxed);
    } else {
        // SAFETY: the handle only keys the block map.
        let memory = unsafe { allocation.memory() };
        *pool.blocks.entry(memory).or_default() += 1;
        counters
            .blocks
            .store(pool.blocks.len() as u64, Ordering::Relaxed);
    }
    drop(pool);
    counters.add(location, allocation.size());
    Ok(allocation)
}

fn free(device: &Device, allocation: Allocation, location: MemoryLocation) {
    let counters = &device.memory_counters;
    counters.sub(location, allocation.size());
    let mut pool = device.allocator.lock();
    let pool = &mut *pool;
    if allocation.is_dedicated() {
        counters.dedicated.fetch_sub(1, Ordering::Relaxed);
        if let Err(error) = pool.allocator.free(allocation) {
            tracing::error!("freeing device memory: {error}");
        }
        return;
    }
    // SAFETY: the handle only keys the block map.
    let memory = unsafe { allocation.memory() };
    let emptied = pool.blocks.get_mut(&memory).is_some_and(|live| {
        *live -= 1;
        *live == 0
    });
    let capacity = if emptied {
        pool.allocator.capacity()
    } else {
        0
    };
    if let Err(error) = pool.allocator.free(allocation) {
        tracing::error!("freeing device memory: {error}");
    }
    if emptied && pool.allocator.capacity() < capacity {
        pool.blocks.remove(&memory);
        counters
            .blocks
            .store(pool.blocks.len() as u64, Ordering::Relaxed);
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
        let capacity = self.allocator.lock().allocator.capacity();
        let allocated = self.memory_counters.total_bytes();
        let slack = capacity.saturating_sub(allocated);
        if !self.features.extensions.memory_budget {
            let used = allocated;
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

    /// Live pipelines the pipeline cache built or fast-linked, retired ones included until they
    /// are destroyed.
    pub fn pipelines(&self) -> u64 {
        self.pipelines.load(Ordering::Relaxed)
    }

    /// Live descriptor pools of the [`crate::descriptors::DescriptorAllocator`]s.
    pub fn descriptor_pools(&self) -> u64 {
        self.descriptor_pools.load(Ordering::Relaxed)
    }

    /// The `VkDeviceMemory` objects the allocator holds. Lock-free, for the activity log.
    pub fn memory_objects(&self) -> MemoryObjects {
        let counters = &self.memory_counters;
        MemoryObjects {
            blocks: counters.blocks.load(Ordering::Relaxed),
            dedicated: counters.dedicated.load(Ordering::Relaxed),
        }
    }

    /// Live allocations by memory location. Lock-free, for the activity log.
    pub fn memory_counts(&self) -> MemoryCounts {
        let counters = &self.memory_counters;
        MemoryCounts {
            gpu_only: counters.live(MemoryLocation::GpuOnly),
            cpu_to_gpu: counters.live(MemoryLocation::CpuToGpu),
            gpu_to_cpu: counters.live(MemoryLocation::GpuToCpu),
        }
    }
}

/// A buffer with its memory. Dropping it frees both, so drop it only once the GPU is done with
/// it ([`crate::scheduler::Scheduler::retire`]).
pub struct Buffer {
    device: Arc<Device>,
    raw: vk::Buffer,
    allocation: Option<Allocation>,
    location: MemoryLocation,
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
        let allocation = match allocate(device, name, requirements, location, Resource::Buffer(raw))
        {
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
        device.name(raw, format_args!("{name}"));
        let buffer = Self {
            device: device.clone(),
            raw,
            allocation: Some(allocation),
            location,
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
            free(&self.device, allocation, self.location);
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

/// What [`Image::bind`] did.
#[derive(Debug, Default)]
pub struct SparseBind {
    /// Layers that got memory in this call, ascending.
    pub newly: Vec<u32>,
    /// Every requested layer has memory now; false when the cap stopped the binding.
    pub complete: bool,
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
        let allocation = match allocate(
            device,
            name,
            requirements,
            MemoryLocation::GpuOnly,
            Resource::Image(raw),
        ) {
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
        device.name(raw, format_args!("{name}"));
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
    /// long as the image stays within `cap` bytes. The layers newly bound hold undefined
    /// contents until written. A non-sparse image has nothing to bind.
    pub fn bind(
        &mut self,
        scheduler: &mut Scheduler,
        level: u32,
        layers: std::ops::Range<u32>,
        cap: u64,
    ) -> Result<SparseBind> {
        let Some(sparse) = self.sparse.as_deref_mut() else {
            return Ok(SparseBind::default());
        };
        let block = sparse.requirements.alignment.max(1);
        let mut image_binds = Vec::new();
        let mut opaque_binds = Vec::new();
        let mut bound = SparseBind {
            newly: Vec::new(),
            complete: true,
        };
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
                bound.complete = false;
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
                    Resource::SparseBlock,
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
                    Resource::SparseBlock,
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
            bound.newly.push(layer);
        }
        if !image_binds.is_empty() || !opaque_binds.is_empty() {
            scheduler.bind_sparse(self.raw, &image_binds, &opaque_binds)?;
        }
        Ok(bound)
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
            free(&self.device, allocation, MemoryLocation::GpuOnly);
        }
        if let Some(sparse) = self.sparse.take() {
            for allocation in sparse.allocations {
                free(&self.device, allocation, MemoryLocation::GpuOnly);
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
#[cfg(test)]
mod tests {
    use lichii_common::settings::GraphicsSettings;

    use super::*;
    use crate::descriptors::DescriptorAllocator;
    use crate::{Instance, SurfaceSupport};

    fn device() -> Option<Arc<Device>> {
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return None;
            }
        };
        match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => Some(device),
            Err(error) => {
                eprintln!("skipping: {error}");
                None
            }
        }
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn unit(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }

        fn log_uniform(&mut self, low: u64, high: u64) -> u64 {
            let (low, high) = ((low as f64).ln(), (high as f64).ln());
            (low + self.unit() * (high - low)).exp() as u64
        }
    }

    struct Live {
        resource: (Option<Image>, Option<Buffer>),
        bytes: u64,
        dies: u64,
    }

    #[derive(Debug, Default)]
    struct Churn {
        peak_slack: u64,
        live_at_peak_slack: u64,
        peak_objects: MemoryObjects,
    }

    const KIB: u64 = 1 << 10;
    const MIB: u64 = 1 << 20;

    /// MKW-like churn at a tenth of its scale: images of 16 KiB to 32 MiB under a 420 MiB
    /// budget, a share of them kept for the whole run, the rest evicted by age and on scene
    /// changes, and guest buffers of 64 KiB to 16 MiB that grow by absorbing others. Returns the
    /// peak slack with the live bytes at that moment, and the most blocks and dedicated
    /// allocations held at once.
    fn churn(device: &Arc<Device>) -> Churn {
        const TICKS: u64 = 600;
        const IMAGE_BUDGET: u64 = 420 * MIB;
        const BUFFER_BUDGET: u64 = 130 * MIB;
        const BUFFERS: usize = 300;
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut live: Vec<Live> = Vec::new();
        let mut result = Churn::default();
        let image_usage = vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST;
        let buffer_usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let new_buffer = |bytes: u64| {
            Buffer::new(
                device,
                bytes,
                buffer_usage,
                MemoryLocation::GpuOnly,
                "churn",
            )
            .unwrap()
        };
        let is_image = |l: &Live| l.resource.0.is_some();
        for tick in 0..TICKS {
            let scene = tick % 200 == 0;
            let mut i = 0;
            while i < live.len() {
                let transient = live[i].dies != u64::MAX;
                if live[i].dies <= tick
                    || scene && is_image(&live[i]) && transient && rng.unit() < 0.8
                {
                    live.swap_remove(i);
                } else {
                    i += 1;
                }
            }
            let images = if tick < 60 {
                8
            } else if tick % 200 < 20 {
                12
            } else {
                3
            };
            for _ in 0..images {
                let x = rng.unit();
                let bytes = if x < 0.55 {
                    rng.log_uniform(16 * KIB, 256 * KIB)
                } else if x < 0.80 {
                    rng.log_uniform(256 * KIB, 2 * MIB)
                } else if x < 0.93 {
                    rng.log_uniform(2 * MIB, 8 * MIB)
                } else {
                    rng.log_uniform(8 * MIB, 32 * MIB)
                };
                let side = ((bytes / 4) as f64).sqrt() as u32;
                let desc = ImageDesc::d2(vk::Format::R8G8B8A8_UNORM, side, side, image_usage);
                let image = Image::new(device, &desc, "churn").unwrap();
                let persistent = rng.unit() < if tick < 60 { 0.4 } else { 0.08 };
                let dies = if persistent {
                    u64::MAX
                } else {
                    tick + 1 + (-(1.0 - rng.unit()).ln() * 90.0) as u64
                };
                live.push(Live {
                    bytes: image.memory_size(),
                    resource: (Some(image), None),
                    dies,
                });
            }
            let mut image_bytes: u64 = live.iter().filter(|l| is_image(l)).map(|l| l.bytes).sum();
            while image_bytes > IMAGE_BUDGET {
                let oldest = (0..live.len())
                    .filter(|&i| is_image(&live[i]))
                    .min_by_key(|&i| live[i].dies)
                    .unwrap();
                image_bytes -= live.swap_remove(oldest).bytes;
            }
            let buffers = live.iter().filter(|l| !is_image(l)).count();
            let created = if buffers >= BUFFERS {
                0
            } else if tick < 60 {
                12
            } else {
                4
            };
            for _ in 0..created {
                let x = rng.unit();
                let bytes = if x < 0.80 {
                    rng.log_uniform(64 * KIB, 256 * KIB)
                } else if x < 0.97 {
                    rng.log_uniform(256 * KIB, 2 * MIB)
                } else {
                    rng.log_uniform(2 * MIB, 16 * MIB)
                }
                .next_multiple_of(64 * KIB);
                live.push(Live {
                    resource: (None, Some(new_buffer(bytes))),
                    bytes,
                    dies: u64::MAX,
                });
            }
            let buffers: Vec<usize> = (0..live.len()).filter(|&i| !is_image(&live[i])).collect();
            if buffers.len() >= 4 {
                let mut absorbed = vec![buffers[(rng.next() % buffers.len() as u64) as usize]];
                let mut total = live[absorbed[0]].bytes;
                if rng.next() % 2 == 1 {
                    let other = buffers[(rng.next() % buffers.len() as u64) as usize];
                    if !absorbed.contains(&other) && total + live[other].bytes <= 16 * MIB {
                        total += live[other].bytes;
                        absorbed.push(other);
                    }
                }
                let bytes = (total + rng.log_uniform(64 * KIB, (total / 2).max(128 * KIB)))
                    .next_multiple_of(64 * KIB)
                    .min(16 * MIB);
                let grown = new_buffer(bytes);
                absorbed.sort_unstable_by(|a, b| b.cmp(a));
                for i in absorbed {
                    live.swap_remove(i);
                }
                live.push(Live {
                    resource: (None, Some(grown)),
                    bytes,
                    dies: u64::MAX,
                });
            }
            let mut buffer_bytes: u64 = live.iter().filter(|l| !is_image(l)).map(|l| l.bytes).sum();
            while buffer_bytes > BUFFER_BUDGET {
                let buffers: Vec<usize> =
                    (0..live.len()).filter(|&i| !is_image(&live[i])).collect();
                let victim = buffers[(rng.next() % buffers.len() as u64) as usize];
                buffer_bytes -= live.swap_remove(victim).bytes;
            }
            let slack = device.memory_usage().slack;
            if slack > result.peak_slack {
                result.peak_slack = slack;
                result.live_at_peak_slack = device.memory_counts().gpu_only.bytes;
            }
            let objects = device.memory_objects();
            result.peak_objects.blocks = result.peak_objects.blocks.max(objects.blocks);
            result.peak_objects.dedicated = result.peak_objects.dedicated.max(objects.dedicated);
        }
        result
    }

    /// Lavapipe's memory is host-visible, so its blocks are 32 MiB host blocks. With images
    /// from 8 MiB dedicated and buffers never, this churn peaks at 278-340 MiB of slack on
    /// 573 MiB live; with [`DEDICATED_BYTES`] at 4 MiB for both, at 125-189 MiB (the spread
    /// comes from gpu-allocator's hash-ordered best fit).
    #[test]
    fn churn_leaves_little_slack_in_few_objects() {
        let Some(device) = device() else { return };
        let churn = churn(&device);
        eprintln!("{churn:?}");
        assert!(
            churn.peak_slack * 20 <= churn.live_at_peak_slack * 9,
            "slack peaked at {} MiB with {} MiB live",
            churn.peak_slack >> 20,
            churn.live_at_peak_slack >> 20,
        );
        assert!(churn.peak_objects.blocks <= 16, "{:?}", churn.peak_objects);
        assert!(
            churn.peak_objects.dedicated <= 64,
            "{:?}",
            churn.peak_objects
        );
        let after = device.memory_objects();
        assert_eq!(after.dedicated, 0, "every dedicated allocation is freed");
        assert!(
            after.blocks <= u64::from(device.memory_properties.memory_type_count),
            "at most one empty block per memory type stays: {after:?}",
        );
    }

    #[test]
    fn shared_and_dedicated_allocations_are_counted_and_freed() {
        let Some(device) = device() else { return };
        let objects = device.memory_objects();
        let counts = device.memory_counts();
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let buffer = |location, size| Buffer::new(&device, size, usage, location, "mix").unwrap();
        let image = |side| {
            let desc = ImageDesc::d2(
                vk::Format::R8G8B8A8_UNORM,
                side,
                side,
                vk::ImageUsageFlags::SAMPLED,
            );
            Image::new(&device, &desc, "mix").unwrap()
        };
        let shared = [
            buffer(MemoryLocation::GpuOnly, DEDICATED_BYTES - (64 << 10)),
            buffer(MemoryLocation::GpuOnly, 64 << 10),
            buffer(MemoryLocation::CpuToGpu, DEDICATED_BYTES * 2),
        ];
        let small_image = image(256);
        assert_eq!(device.memory_objects().dedicated, objects.dedicated);
        assert!(device.memory_objects().blocks > objects.blocks);
        let dedicated = [
            buffer(MemoryLocation::GpuOnly, DEDICATED_BYTES),
            buffer(MemoryLocation::GpuOnly, 3 * DEDICATED_BYTES),
        ];
        let large_image = image(2048);
        let during = device.memory_objects();
        assert_eq!(during.dedicated, objects.dedicated + 3);
        let mut mapped = buffer(MemoryLocation::CpuToGpu, 4 << 10);
        mapped.mapped_mut().unwrap().fill(0xa5);
        assert!(shared[2].mapped().is_some());
        assert!(mapped.mapped().unwrap().iter().all(|&b| b == 0xa5));
        assert_eq!(device.memory_objects().dedicated, during.dedicated);
        let gpu_only = device.memory_counts().gpu_only;
        assert_eq!(gpu_only.count - counts.gpu_only.count, 6);
        assert!(gpu_only.bytes - counts.gpu_only.bytes >= 5 * DEDICATED_BYTES + 2048 * 2048 * 4);
        drop(dedicated);
        drop(large_image);
        assert_eq!(device.memory_objects().dedicated, objects.dedicated);
        drop((shared, small_image, mapped));
        assert_eq!(device.memory_counts(), counts, "every free is counted");
        let after = device.memory_objects();
        assert_eq!(after.dedicated, objects.dedicated);
        assert!(
            after.blocks <= objects.blocks + u64::from(device.memory_properties.memory_type_count)
        );
        let capacity = device.allocator.lock().allocator.capacity();
        assert_eq!(
            device.memory_usage().slack,
            capacity,
            "only empty blocks are left"
        );
    }

    #[test]
    fn dedicated_allocations_stop_at_the_limit() {
        let gpu = MemoryLocation::GpuOnly;
        assert!(!dedicates(gpu, DEDICATED_BYTES - 1, 0));
        assert!(dedicates(gpu, DEDICATED_BYTES, 0));
        assert!(dedicates(gpu, 1 << 30, DEDICATED_LIMIT - 1));
        assert!(!dedicates(gpu, 1 << 30, DEDICATED_LIMIT));
        assert!(!dedicates(MemoryLocation::CpuToGpu, 1 << 30, 0));
        assert!(!dedicates(MemoryLocation::GpuToCpu, 1 << 30, 0));
    }

    #[test]
    fn counters_follow_allocations_and_frees_of_each_location() {
        let Some(device) = device() else { return };
        let before = device.memory_counts();
        let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        let new = |location, size| Buffer::new(&device, size, usage, location, "test").unwrap();
        let buffers = [
            new(MemoryLocation::GpuOnly, 1 << 20),
            new(MemoryLocation::CpuToGpu, 2 << 20),
            new(MemoryLocation::CpuToGpu, 3 << 20),
            new(MemoryLocation::GpuToCpu, 4 << 20),
        ];
        let image = Image::new(
            &device,
            &ImageDesc::d2(
                vk::Format::R8G8B8A8_UNORM,
                256,
                256,
                vk::ImageUsageFlags::SAMPLED,
            ),
            "test",
        )
        .unwrap();
        let during = device.memory_counts();
        let grew = |after: LiveMemory, before: LiveMemory| {
            (after.bytes - before.bytes, after.count - before.count)
        };
        let (gpu_only_bytes, gpu_only_count) = grew(during.gpu_only, before.gpu_only);
        assert_eq!(gpu_only_count, 2, "a buffer and an image");
        assert!(gpu_only_bytes >= (1 << 20) + 256 * 256 * 4);
        let (upload_bytes, upload_count) = grew(during.cpu_to_gpu, before.cpu_to_gpu);
        assert_eq!(upload_count, 2);
        assert!(upload_bytes >= 5 << 20);
        let (readback_bytes, readback_count) = grew(during.gpu_to_cpu, before.gpu_to_cpu);
        assert_eq!(readback_count, 1);
        assert!(readback_bytes >= 4 << 20);
        drop(buffers);
        drop(image);
        assert_eq!(device.memory_counts(), before, "every free is counted");
    }

    #[test]
    fn descriptor_pools_are_counted_until_their_allocator_drops() {
        let Some(device) = device() else { return };
        let before = device.descriptor_pools();
        let descriptors = DescriptorAllocator::new(&device).unwrap();
        assert_eq!(device.descriptor_pools(), before + 1);
        drop(descriptors);
        assert_eq!(device.descriptor_pools(), before);
    }
}
