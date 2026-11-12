//! Foundation tests on whatever Vulkan device the machine has (lavapipe in CI). Each test
//! skips when no Vulkan 1.3 device is available, as on the Windows runners.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ash::vk;
use gpu_allocator::MemoryLocation;
use lichii_common::settings::{AstcRecompression, GraphicsSettings, PresentMode};
use lichii_texture::PixelFormat;
use lichii_vulkan::commands::{COLOR_RANGE, ImageAccess, transfer_to_all_barrier, transition};
use lichii_vulkan::format::{Conversion, host_format};
use lichii_vulkan::memory::{Buffer, Image, ImageDesc};
use lichii_vulkan::scheduler::{Scheduler, SemaphoreWait};
use lichii_vulkan::staging::StagingRing;
use lichii_vulkan::swapchain::{Surface, Swapchain};
use lichii_vulkan::{Device, Instance, SurfaceSupport, available_devices};

fn instance(surfaces: SurfaceSupport) -> Option<Arc<Instance>> {
    match Instance::new(true, surfaces) {
        Ok(instance) => {
            eprintln!("validation layers: {}", instance.validation_enabled());
            Some(Arc::new(instance))
        }
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

fn device() -> Option<Arc<Device>> {
    let instance = instance(SurfaceSupport::None)?;
    match Device::new(instance, &GraphicsSettings::default(), None) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

fn assert_no_validation_errors(device: &Device) {
    assert_eq!(
        device.instance().validation_errors(),
        0,
        "validation reported errors"
    );
}

#[test]
fn selects_devices_by_name_and_falls_back() {
    let Some(instance) = instance(SurfaceSupport::None) else {
        return;
    };
    let Ok(names) = available_devices(&instance) else {
        return;
    };
    let Some(first) = names.first() else { return };
    let named = GraphicsSettings {
        device: Some(first.to_uppercase()),
        ..GraphicsSettings::default()
    };
    let device = Device::new(instance.clone(), &named, None).unwrap();
    assert_eq!(&device.properties.name, first);
    drop(device);
    let missing = GraphicsSettings {
        device: Some("no such GPU".into()),
        ..GraphicsSettings::default()
    };
    let device = Device::new(instance, &missing, None).unwrap();
    assert!(names.contains(&device.properties.name));
    assert!(device.properties.api_version >= vk::API_VERSION_1_3);
}

#[test]
fn copies_through_the_scheduler_and_retires_after_completion() {
    let Some(device) = device() else { return };
    let mut scheduler = Scheduler::new(&device).unwrap();
    let data: Vec<u8> = (0..4096u32).map(|i| (i * 7) as u8).collect();
    let mut upload = Buffer::new(
        &device,
        4096,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        "up",
    )
    .unwrap();
    upload.mapped_mut().unwrap().copy_from_slice(&data);
    let middle = Buffer::new(
        &device,
        4096,
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
        "mid",
    )
    .unwrap();
    let readback = Buffer::new(
        &device,
        4096,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "down",
    )
    .unwrap();

    let first_tick = scheduler.tick();
    let cmd = scheduler.record().unwrap();
    let region = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: 4096,
    }];
    // SAFETY: all buffers are live and large enough.
    unsafe {
        device
            .raw()
            .cmd_copy_buffer(cmd, upload.raw(), middle.raw(), &region)
    };
    transfer_to_all_barrier(&device, cmd);
    // SAFETY: as above.
    unsafe {
        device
            .raw()
            .cmd_copy_buffer(cmd, middle.raw(), readback.raw(), &region)
    };
    transfer_to_all_barrier(&device, cmd);

    struct Tracked(Arc<AtomicUsize>);
    impl Drop for Tracked {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    scheduler.retire(Tracked(drops.clone()));
    scheduler.retire(upload);
    let tick = scheduler.flush().unwrap();
    assert_eq!(tick, first_tick);
    assert_eq!(scheduler.tick(), tick + 1);
    scheduler.wait(tick).unwrap();
    assert!(scheduler.is_complete(tick));
    scheduler.record().unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(readback.mapped().unwrap(), &data[..]);
    scheduler.finish().unwrap();
    drop(scheduler);
    assert_no_validation_errors(&device);
}

#[test]
fn staging_ring_wraps_and_waits_for_the_gpu() {
    let Some(device) = device() else { return };
    let mut scheduler = Scheduler::new(&device).unwrap();
    let mut ring = StagingRing::new(&device, 64 << 10).unwrap();
    const CHUNK: u64 = 7000;
    const CHUNKS: u64 = 60;
    let target = Buffer::new(
        &device,
        CHUNK * CHUNKS + (20 << 10),
        vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
        "target",
    )
    .unwrap();
    let pattern = |i: u64, j: usize| (i as usize * 31 + j * 7) as u8;
    for i in 0..CHUNKS {
        let slice = ring.alloc(&mut scheduler, CHUNK, 256).unwrap();
        assert_eq!(slice.offset % 256, 0);
        for (j, byte) in slice.data.iter_mut().enumerate() {
            *byte = pattern(i, j);
        }
        let (src, offset) = (slice.buffer, slice.offset);
        let cmd = scheduler.record().unwrap();
        let region = [vk::BufferCopy {
            src_offset: offset,
            dst_offset: i * CHUNK,
            size: CHUNK,
        }];
        // SAFETY: live buffers, in-bounds regions.
        unsafe {
            device
                .raw()
                .cmd_copy_buffer(cmd, src, target.raw(), &region)
        };
        if i % 5 == 4 {
            scheduler.flush().unwrap();
        }
    }
    let big = ring.alloc(&mut scheduler, 20 << 10, 4).unwrap();
    big.data.fill(0xab);
    let (src, len) = (big.buffer, big.data.len() as u64);
    let cmd = scheduler.record().unwrap();
    // SAFETY: as above.
    unsafe {
        device.raw().cmd_copy_buffer(
            cmd,
            src,
            target.raw(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: CHUNK * CHUNKS,
                size: len,
            }],
        )
    };
    let readback = Buffer::new(
        &device,
        target.size(),
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "rb",
    )
    .unwrap();
    transfer_to_all_barrier(&device, cmd);
    // SAFETY: as above.
    unsafe {
        device.raw().cmd_copy_buffer(
            cmd,
            target.raw(),
            readback.raw(),
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: target.size(),
            }],
        )
    };
    transfer_to_all_barrier(&device, cmd);
    scheduler.finish().unwrap();
    let out = readback.mapped().unwrap();
    for i in 0..CHUNKS {
        for j in 0..CHUNK as usize {
            assert_eq!(
                out[(i * CHUNK) as usize + j],
                pattern(i, j),
                "chunk {i} byte {j}"
            );
        }
    }
    assert!(out[(CHUNK * CHUNKS) as usize..].iter().all(|&b| b == 0xab));
    drop(ring);
    drop(scheduler);
    assert_no_validation_errors(&device);
}

#[test]
fn images_round_trip_through_staging() {
    let Some(device) = device() else { return };
    let mut scheduler = Scheduler::new(&device).unwrap();
    let mut ring = StagingRing::new(&device, 1 << 20).unwrap();
    let (w, h) = (37u32, 21u32);
    let image = Image::new(
        &device,
        &ImageDesc::d2(
            vk::Format::A8B8G8R8_UNORM_PACK32,
            w,
            h,
            vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
        ),
        "img",
    )
    .unwrap();
    let texels: Vec<u8> = (0..w * h * 4).map(|i| (i * 13 % 251) as u8).collect();
    let slice = ring.alloc(&mut scheduler, texels.len() as u64, 16).unwrap();
    slice.data.copy_from_slice(&texels);
    let (src, offset) = (slice.buffer, slice.offset);
    let cmd = scheduler.record().unwrap();
    transition(
        &device,
        cmd,
        image.raw(),
        COLOR_RANGE,
        ImageAccess::UNDEFINED,
        ImageAccess::TRANSFER_DST,
    );
    let copy = [vk::BufferImageCopy {
        buffer_offset: offset,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width: w,
            height: h,
            depth: 1,
        },
    }];
    let readback = Buffer::new(
        &device,
        texels.len() as u64,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "rb",
    )
    .unwrap();
    let mut back = copy;
    back[0].buffer_offset = 0;
    // SAFETY: live objects, regions in bounds, layouts set by the transitions.
    unsafe {
        device.raw().cmd_copy_buffer_to_image(
            cmd,
            src,
            image.raw(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &copy,
        );
    }
    transition(
        &device,
        cmd,
        image.raw(),
        COLOR_RANGE,
        ImageAccess::TRANSFER_DST,
        ImageAccess::TRANSFER_SRC,
    );
    // SAFETY: as above.
    unsafe {
        device.raw().cmd_copy_image_to_buffer(
            cmd,
            image.raw(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.raw(),
            &back,
        );
    }
    transfer_to_all_barrier(&device, cmd);
    scheduler.finish().unwrap();
    assert_eq!(readback.mapped().unwrap(), &texels[..]);
    let usage = device.memory_usage();
    assert!(usage.budget > 0 && usage.device_local > 0);
    drop((ring, scheduler, image));
    assert_no_validation_errors(&device);
}

#[test]
fn every_guest_format_gets_a_sampleable_host_format() {
    let Some(device) = device() else { return };
    for &format in PixelFormat::ALL {
        for recompression in [AstcRecompression::Uncompressed, AstcRecompression::Bc3] {
            let host = host_format(&device, format, recompression);
            let features = device.format_features(host.format);
            assert!(
                features.contains(
                    vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::TRANSFER_DST
                ),
                "{format:?} -> {:?} ({features:?})",
                host.format
            );
            if host.attachable && format.is_color() {
                assert!(
                    features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT),
                    "{format:?}"
                );
            }
            if format.is_astc() && !device.features.texture_compression_astc_ldr {
                let expected = match recompression {
                    AstcRecompression::Bc3 if device.features.texture_compression_bc => {
                        Conversion::DecodeToBc(lichii_texture::Compression::Bc3)
                    }
                    _ => Conversion::Decode,
                };
                assert_eq!(host.conversion, expected, "{format:?}");
            }
        }
    }
    let rt = host_format(
        &device,
        PixelFormat::A8B8G8R8Unorm,
        AstcRecompression::Uncompressed,
    );
    assert!(rt.attachable && rt.conversion == Conversion::None);
}

#[test]
fn headless_swapchain_presents_and_recreates() {
    let Some(instance) = instance(SurfaceSupport::Headless) else {
        return;
    };
    if !instance.supports_headless_surface() {
        eprintln!("skipping: no VK_EXT_headless_surface");
        return;
    }
    let surface = Surface::headless(&instance).unwrap();
    let Ok(device) = Device::new(
        instance.clone(),
        &GraphicsSettings::default(),
        Some(surface.raw()),
    ) else {
        return;
    };
    let mut swapchain = Swapchain::new(&device, surface, 320, 180, PresentMode::Mailbox).unwrap();
    let mut scheduler = Scheduler::new(&device).unwrap();
    let mut ticks = Vec::new();
    for (frame_no, (w, h)) in [(320, 180), (320, 180), (640, 360), (640, 360), (200, 100)]
        .into_iter()
        .enumerate()
    {
        if ticks.len() >= swapchain.image_count() {
            scheduler
                .wait(ticks[ticks.len() - swapchain.image_count()])
                .unwrap();
        }
        let frame = swapchain.acquire(w, h).unwrap().expect("a frame");
        assert_eq!((frame.extent.width, frame.extent.height), (w, h));
        let cmd = scheduler.record().unwrap();
        transition(
            &device,
            cmd,
            frame.image,
            COLOR_RANGE,
            ImageAccess::UNDEFINED,
            ImageAccess::TRANSFER_DST,
        );
        let color = vk::ClearColorValue {
            float32: [frame_no as f32 / 5.0, 0.2, 0.4, 1.0],
        };
        // SAFETY: the image is in TRANSFER_DST layout.
        unsafe {
            device.raw().cmd_clear_color_image(
                cmd,
                frame.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &color,
                &[COLOR_RANGE],
            )
        };
        transition(
            &device,
            cmd,
            frame.image,
            COLOR_RANGE,
            ImageAccess::TRANSFER_DST,
            ImageAccess::PRESENT,
        );
        let wait = SemaphoreWait {
            semaphore: frame.acquired,
            value: 0,
            stage: vk::PipelineStageFlags2::ALL_TRANSFER,
        };
        ticks.push(scheduler.flush_with(&[wait], &[frame.rendered]).unwrap());
        swapchain.present(&frame).unwrap();
    }
    assert!(swapchain.acquire(0, 0).unwrap().is_none());
    scheduler.finish().unwrap();
    drop(scheduler);
    drop(swapchain);
    assert_no_validation_errors(&device);
}

#[test]
fn descriptor_pools_rotate_and_reset_after_completion() {
    use lichii_vulkan::descriptors::DescriptorAllocator;
    let Some(device) = device() else { return };
    let mut scheduler = Scheduler::new(&device).unwrap();
    let mut descriptors = DescriptorAllocator::new(&device).unwrap();
    let binding = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding);
    // SAFETY: valid create info.
    let layout = unsafe {
        device
            .raw()
            .create_descriptor_set_layout(&info, None)
            .unwrap()
    };
    let mut seen = std::collections::HashSet::new();
    for frame in 0..6 {
        for _ in 0..1500 {
            let set = descriptors.allocate(&mut scheduler, layout).unwrap();
            assert_ne!(set, vk::DescriptorSet::null());
            seen.insert(set);
        }
        scheduler.flush().unwrap();
        if frame % 2 == 1 {
            scheduler.finish().unwrap();
        }
    }
    scheduler.finish().unwrap();
    drop(descriptors);
    // SAFETY: no pool uses the layout any more.
    unsafe { device.raw().destroy_descriptor_set_layout(layout, None) };
    drop(scheduler);
    assert!(seen.len() > 1500);
    assert_no_validation_errors(&device);
}
