use std::sync::{Arc, mpsc};
use std::time::Duration;

use gpu_allocator::MemoryLocation;
use lichii_common::settings::GraphicsSettings;
use lichii_gpu::backend::{
    ChannelId, ConstBufferBinding, FenceAction, GraphicsContext, NullRasterizer, ReleaseSize,
};
use lichii_gpu::clock::GuestTime;
use lichii_gpu::engines::maxwell_3d::regs::{MAX_CONST_BUFFERS, MAX_SHADER_STAGE};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};
use lichii_gpu::gmmu::BigPageSize;
use lichii_gpu::gpu::{Console, Gpu, GpuConfig, Threading};
use lichii_gpu::host1x::Host1x;
use lichii_memory::{DeviceAddressSpace, PhysicalMemory};
use lichii_texture::block_linear::Tiling;

use super::*;
use crate::commands::{ImageAccess, transfer_to_all_barrier};
use crate::memory::Buffer;
use crate::present::FrameMailbox;
use crate::rasterizer::tests::{DADDR, Fixture, GPU_VA, clear_regs, fixture, gmmu, guest_linear};
use crate::scheduler::SemaphoreWait;
use crate::texture_cache::ImageInfo;
use crate::{Device, Instance, SurfaceSupport};

const GREEN: [f32; 4] = [0., 1., 0., 1.];
const BLUE: [f32; 4] = [0., 0., 1., 1.];
const SEMAPHORE: DAddr = DADDR + 0x80_0000;
const TIMEOUT: Duration = Duration::from_secs(5);

struct Standard<'a> {
    memory: &'a Arc<Gmmu>,
    regs: Box<Regs>,
    dirty: DirtyState,
    const_buffers: [[ConstBufferBinding; MAX_CONST_BUFFERS]; MAX_SHADER_STAGE],
    root_tables: [[u32; 64]; 8],
}

impl<'a> Standard<'a> {
    fn new(memory: &'a Arc<Gmmu>, color: [f32; 4]) -> Self {
        Self {
            memory,
            regs: clear_regs(64, 32, 2, color),
            dirty: DirtyState::new(),
            const_buffers: [[ConstBufferBinding::default(); MAX_CONST_BUFFERS]; MAX_SHADER_STAGE],
            root_tables: [[0; 64]; 8],
        }
    }

    fn record(&mut self, renderer: &mut dyn Rasterizer) {
        let mut ctx = GraphicsContext {
            channel: ChannelId(0),
            memory: self.memory,
            regs: &self.regs,
            dirty: &mut self.dirty,
            const_buffers: &self.const_buffers,
            root_tables: &self.root_tables,
        };
        renderer.clear(&mut ctx, 1);
    }
}

fn register(renderer: &mut VulkanRasterizer, memory: &Gmmu, regs: &Regs) -> Found {
    let info = ImageInfo::from_render_target(&regs.rt[0], 0, renderer.gob_layout).unwrap();
    assert!(memory.is_continuous(GPU_VA, info.size_bytes()));
    assert_eq!(memory.translate(GPU_VA), Some(DADDR));
    let mut tx = Transfer {
        scheduler: &mut renderer.scheduler,
        staging: &mut renderer.staging,
        memory: &renderer.handles.device,
        buffers: &mut renderer.buffer_cache,
    };
    renderer
        .texture_cache
        .render_target(&mut tx, memory, GPU_VA, &info, true)
        .unwrap()
        .unwrap()
}

fn scene(layout: GobLayout) -> Option<(Fixture, Arc<Gmmu>, Found)> {
    let fixture = fixture(layout);
    assert!(
        fixture.is_some() || std::env::var_os("LICHII_REQUIRE_VALIDATION").is_none(),
        "the Direct NVN tests require a Vulkan device with validation"
    );
    let mut f = fixture?;
    let memory = gmmu(&f);
    let regs = clear_regs(64, 32, 2, [0.; 4]);
    let found = register(&mut f.rasterizer, &memory, &regs);
    f.rasterizer.wait_for_idle();
    Some((f, memory, found))
}

fn authorize(r: &mut VulkanRasterizer, memory: &Arc<Gmmu>, found: Found) {
    r.nvn2_target = Some(ResidentTarget {
        found,
        memory: memory.clone(),
        address: GPU_VA,
        mapping_generation: memory.generation(),
        cache_generation: r.texture_cache.generation(),
        backing: DADDR,
        fail_after_recording: None,
    });
}

fn packet(memory: &Arc<Gmmu>, color: [f32; 4], fence: FenceAction) -> Nvn2Submission {
    Nvn2Submission {
        memory: memory.clone(),
        target: GPU_VA,
        mapping_generation: memory.generation(),
        color,
        color_mask: 0xf,
        fence,
    }
}

fn replay_or_standard(
    r: &mut dyn Rasterizer,
    packet: &Nvn2Submission,
    standard: impl FnOnce(&mut dyn Rasterizer),
) -> Result<ReplayAdmission, ReplayError> {
    let admission = r.try_replay_nvn2(packet)?;
    if admission == ReplayAdmission::UseStandard {
        standard(r);
        r.signal_fence(packet.fence);
    }
    Ok(admission)
}

fn release(value: u32) -> FenceAction {
    FenceAction::WriteSemaphore {
        address: SEMAPHORE + u64::from(value) * 4,
        value: u64::from(value),
        size: ReleaseSize::Four,
    }
}

struct Gates {
    device: Arc<Device>,
    semaphores: [Option<vk::Semaphore>; 2],
    open: [bool; 2],
}

impl Gates {
    fn new(device: Arc<Device>) -> Self {
        let mut gates = Self {
            device,
            semaphores: [None; 2],
            open: [false; 2],
        };
        let mut timeline =
            vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
        for slot in &mut gates.semaphores {
            // SAFETY: valid semaphore create info on a live device.
            *slot = Some(unsafe { gates.device.raw.create_semaphore(&info, None).unwrap() });
        }
        gates
    }

    fn wait(&self, index: usize) -> SemaphoreWait {
        SemaphoreWait {
            semaphore: self.semaphores[index].unwrap(),
            value: 1,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
        }
    }

    fn open(&mut self, index: usize) {
        if let Some(semaphore) = self.semaphores[index]
            && !std::mem::replace(&mut self.open[index], true)
        {
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(semaphore)
                .value(1);
            // SAFETY: this owner signals the timeline once; the device only waits on it.
            unsafe { self.device.raw.signal_semaphore(&info).unwrap() };
        }
    }
}

impl Drop for Gates {
    fn drop(&mut self) {
        for index in 0..self.semaphores.len() {
            self.open(index);
        }
        self.device.wait_idle();
        for semaphore in self.semaphores.iter().flatten() {
            // SAFETY: no pending work refers to these semaphores once the device is idle.
            unsafe { self.device.raw.destroy_semaphore(*semaphore, None) };
        }
    }
}

fn sample(r: &mut VulkanRasterizer, target: Found, readback: &Buffer) {
    let image = r
        .texture_cache
        .prepare(&mut r.scheduler, target.id, ImageAccess::TRANSFER_SRC)
        .unwrap()
        .unwrap();
    let cmd = r.scheduler.record().unwrap();
    let copy = vk::BufferImageCopy::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        )
        .image_extent(vk::Extent3D {
            width: 1,
            height: 1,
            depth: 1,
        });
    // SAFETY: the cache transitioned the live image and the readback has room for the texel.
    unsafe {
        r.device.raw.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback.raw(),
            &[copy],
        );
    }
    transfer_to_all_barrier(&r.device, cmd);
}

fn run_ordered(requested: bool, layout: GobLayout, expected: ReplayAdmission) -> Option<Vec<u8>> {
    let (mut f, memory, found) = scene(layout)?;
    let r = &mut f.rasterizer;
    if requested {
        authorize(r, &memory, found);
    }
    let readback = [(); 2].map(|()| {
        Buffer::new(
            &r.device,
            4,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "direct-nvn-order-test",
        )
        .unwrap()
    });
    let (wake, wakes) = mpsc::channel();
    r.set_fence_wake(Box::new(move || {
        let _ = wake.send(());
    }));
    let mut gates = Gates::new(r.device.clone());
    let before = r.scheduler.submits();
    let mut standard = Standard::new(&memory, GREEN);
    for (i, color) in [GREEN, BLUE].into_iter().enumerate() {
        standard.regs.clear_color = color;
        let used = replay_or_standard(r, &packet(&memory, color, release(i as u32 + 1)), |r| {
            assert_eq!(expected, ReplayAdmission::UseStandard);
            standard.record(r);
        })
        .unwrap();
        assert_eq!(used, expected);
        sample(r, found, &readback[i]);
        r.scheduler.flush_with(&[gates.wait(i)], &[]).unwrap();
        r.flush_commands();
    }
    assert_eq!(r.scheduler.submits(), before + 2);
    r.release_fences(false);
    assert_eq!(r.fences.len(), 2);
    assert_eq!(wakes.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert_eq!(f.memory.read::<[u32; 2]>(SEMAPHORE + 4).unwrap(), [0, 0]);
    assert_eq!(f.memory.read::<[u8; 4]>(DADDR).unwrap(), [0; 4]);

    gates.open(0);
    wakes
        .recv_timeout(TIMEOUT)
        .expect("first GPU completion wakes the existing waiter");
    r.release_fences(false);
    assert_eq!(r.fences.len(), 1);
    assert_eq!(f.memory.read::<[u32; 2]>(SEMAPHORE + 4).unwrap(), [1, 0]);
    assert_eq!(f.memory.read::<[u8; 4]>(DADDR).unwrap(), [0; 4]);
    assert_eq!(readback[0].mapped().unwrap(), [0, 255, 0, 255]);
    assert_eq!(wakes.try_recv(), Err(mpsc::TryRecvError::Empty));

    gates.open(1);
    wakes
        .recv_timeout(TIMEOUT)
        .expect("GPU completion wakes the existing waiter");
    r.release_fences(false);
    assert!(!r.has_pending_fences());
    assert_eq!(wakes.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert_eq!(f.memory.read::<[u32; 2]>(SEMAPHORE + 4).unwrap(), [1, 2]);
    assert_eq!(readback[1].mapped().unwrap(), [0, 0, 255, 255]);

    let tiling = Tiling::new(64, 32, 1, 4, 2, 0, layout);
    r.flush_region(DADDR, tiling.size_bytes() as u64);
    let bytes = guest_linear(&f, DADDR, &tiling);
    assert!(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [0, 0, 255, 255])
    );
    f.assert_no_validation_errors();
    Some(bytes)
}

#[test]
fn dropping_closed_gates_unblocks_both_submissions() {
    let Some((mut f, _, _)) = scene(GobLayout::Switch2) else {
        return;
    };
    let r = &mut f.rasterizer;
    let gates = Gates::new(r.device.clone());
    for index in 0..2 {
        r.scheduler.record().unwrap();
        r.signal_fence(release(index as u32 + 1));
        r.scheduler.flush_with(&[gates.wait(index)], &[]).unwrap();
    }
    r.release_fences(false);
    assert_eq!(r.fences.len(), 2);
    drop(gates);
    r.release_fences(false);
    assert!(!r.has_pending_fences());
    assert_eq!(f.memory.read::<[u32; 2]>(SEMAPHORE + 4).unwrap(), [1, 2]);
    f.assert_no_validation_errors();
}

#[test]
fn direct_and_standard_clear_the_same_pixels_in_queue_order() {
    let Some(standard) = run_ordered(false, GobLayout::Switch2, ReplayAdmission::UseStandard)
    else {
        return;
    };
    assert_eq!(
        run_ordered(true, GobLayout::Switch2, ReplayAdmission::Recorded),
        Some(standard.clone())
    );
    assert_eq!(
        run_ordered(true, GobLayout::Tegra, ReplayAdmission::UseStandard),
        Some(standard)
    );
}

#[test]
fn rejected_submissions_enter_standard_before_recording_or_enqueuing_a_fence() {
    const REMAPPED: DAddr = DADDR + 0x40_0000;
    let layout = GobLayout::Switch2;
    assert!(!supported(vk::FormatFeatureFlags::COLOR_ATTACHMENT));
    let Some((mut f, memory, found)) = scene(layout) else {
        return;
    };
    let r = &mut f.rasterizer;
    let mut standard = Standard::new(&memory, GREEN);
    let mut calls = 0;
    for i in 0..4 {
        let mut packet = packet(&memory, GREEN, release(1));
        match i {
            0 => assert!(r.nvn2_target.is_none()),
            1 => {
                authorize(r, &memory, found);
                packet.target += 0x1000;
            }
            2 => packet.color_mask = 2,
            _ => {
                memory.map(GPU_VA, REMAPPED, 0x10000, 0, false).unwrap();
                assert_ne!(memory.generation(), packet.mapping_generation);
                standard.regs.clear_color = BLUE;
                packet.color = BLUE;
            }
        }
        let tick = r.scheduler.tick();
        let generations = r.texture_cache.generation();
        let was_modified = r.texture_cache.state(found.id).unwrap().gpu_modified;
        assert_eq!(
            {
                let used = r.try_replay_nvn2(&packet).unwrap();
                if used == ReplayAdmission::UseStandard {
                    calls += 1;
                    assert_eq!(r.scheduler.tick(), tick);
                    assert!(!r.unflushed);
                    assert_eq!(r.texture_cache.generation(), generations);
                    assert_eq!(
                        r.texture_cache.state(found.id).unwrap().gpu_modified,
                        was_modified
                    );
                    assert!(!r.has_pending_fences());
                    standard.record(r);
                    r.signal_fence(packet.fence);
                }
                used
            },
            ReplayAdmission::UseStandard
        );
        assert_eq!(r.fences.len(), 1);
        r.release_fences(true);
    }
    assert_eq!(calls, 4);
    assert_eq!(f.memory.read::<u32>(SEMAPHORE + 4).unwrap(), 1);
    let tiling = Tiling::new(64, 32, 1, 4, 2, 0, layout);
    r.flush_region(DADDR, tiling.size_bytes() as u64);
    r.flush_region(REMAPPED, tiling.size_bytes() as u64);
    assert_eq!(f.memory.read::<[u8; 4]>(DADDR).unwrap(), [0, 255, 0, 255]);
    assert_eq!(
        f.memory.read::<[u8; 4]>(REMAPPED).unwrap(),
        [0, 0, 255, 255]
    );
    f.assert_no_validation_errors();
}

fn stream_device() -> Option<Arc<Device>> {
    let device = Instance::new(true, SurfaceSupport::None)
        .and_then(|instance| Device::new(Arc::new(instance), &GraphicsSettings::default(), None));
    match device {
        Ok(device) => Some(device),
        Err(error) => {
            assert!(
                std::env::var_os("LICHII_REQUIRE_VALIDATION").is_none(),
                "{error}"
            );
            eprintln!("skipping: {error}");
            None
        }
    }
}

struct GuestClock;

impl GuestTime for GuestClock {
    fn global_time_ns(&self) -> u64 {
        1
    }
}

fn snapshot(r: &mut dyn Rasterizer, backing: &DeviceAddressSpace) -> ([u8; 4], [u32; 3]) {
    let tiling = Tiling::new(64, 32, 1, 4, 2, 0, GobLayout::Switch2);
    r.wait_for_idle();
    r.flush_region(DADDR, tiling.size_bytes() as u64);
    (
        backing.read(DADDR).unwrap(),
        backing.read(SEMAPHORE + 4).unwrap(),
    )
}

fn ordered_stream(
    device: &Arc<Device>,
    threading: Threading,
    expected: Result<ReplayAdmission, ReplayError>,
) {
    const DRAM: u64 = 16 << 20;
    const RED: [f32; 4] = [1., 0., 0., 1.];
    let physical = Arc::new(PhysicalMemory::new(DRAM).unwrap());
    physical.commit(0, DRAM).unwrap();
    let host1x = Host1x::new(physical, 36);
    host1x.memory.map(DADDR, 0, DRAM);
    let backing = host1x.memory.clone();
    let memory = Arc::new(Gmmu::new(backing.clone(), BigPageSize::Size64K));
    memory.map(GPU_VA, DADDR, DRAM, 0, false).unwrap();
    let (recorded, recordings) = mpsc::channel();
    let gpu = Gpu::new(
        host1x,
        Arc::new(GuestClock),
        GpuConfig {
            threading,
            ..GpuConfig::new(Console::Switch2)
        },
        |handles| {
            let mut r = VulkanRasterizer::new(
                device,
                handles,
                Arc::new(FrameMailbox::default()),
                GobLayout::Switch2,
            )
            .unwrap();
            if expected != Ok(ReplayAdmission::UseStandard) {
                let found = register(&mut r, &memory, &clear_regs(64, 32, 2, RED));
                authorize(&mut r, &memory, found);
                if expected.is_err() {
                    r.nvn2_target.as_mut().unwrap().fail_after_recording = Some(recorded);
                }
            }
            Box::new(r)
        },
    );
    let prior = memory.clone();
    gpu.run_in_order(move |r| {
        Standard::new(&prior, RED).record(r);
        r.signal_fence(release(1));
    });
    let packet = packet(&memory, GREEN, release(2));
    assert_eq!(
        NullRasterizer::new(gpu.handles().clone()).try_replay_nvn2(&packet),
        Ok(ReplayAdmission::UseStandard)
    );
    let middle = backing.clone();
    let (observed, middle_result) = mpsc::channel();
    gpu.run_in_order(move |r| {
        let prior = snapshot(r, &middle);
        let mut fallbacks = 0;
        let used = replay_or_standard(r, &packet, |r| {
            fallbacks += 1;
            Standard::new(&packet.memory, packet.color).record(r);
        });
        observed.send((prior, used, fallbacks)).unwrap();
    });
    let (observed, following_result) = mpsc::channel();
    gpu.run_in_order(move |r| {
        let middle = snapshot(r, &backing);
        Standard::new(&memory, BLUE).record(r);
        r.signal_fence(release(3));
        observed.send((middle, snapshot(r, &backing))).unwrap();
    });
    let (prior, used, fallbacks) = middle_result.recv_timeout(TIMEOUT).unwrap();
    let (middle, following) = following_result.recv_timeout(TIMEOUT).unwrap();
    gpu.shutdown();
    assert_eq!(prior, ([255, 0, 0, 255], [1, 0, 0]));
    assert_eq!(used, expected);
    assert_eq!(
        fallbacks,
        u8::from(expected == Ok(ReplayAdmission::UseStandard))
    );
    let released = if expected.is_err() { 0 } else { 2 };
    assert_eq!(middle, ([0, 255, 0, 255], [1, released, 0]));
    assert_eq!(following, ([0, 0, 255, 255], [1, released, 3]));
    assert_eq!(
        recordings.try_iter().count(),
        usize::from(expected.is_err())
    );
}

#[test]
fn owned_packets_replay_or_fall_back_in_place_on_the_gpu_stream() {
    let Some(device) = stream_device() else {
        return;
    };
    for threading in [Threading::Split, Threading::Single] {
        for expected in [
            Ok(ReplayAdmission::Recorded),
            Ok(ReplayAdmission::UseStandard),
            Err(ReplayError),
        ] {
            ordered_stream(&device, threading, expected);
        }
    }
    assert_eq!(device.instance().validation_errors(), 0);
}

#[cfg(target_os = "linux")]
#[test]
#[ignore = "matched host cost on an explicitly selected Vulkan device"]
fn matched_clear_submit_cost() {
    use std::time::Instant;

    const CLEARS: usize = 128;
    const BLOCKS: usize = 32;
    const SAMPLES: usize = 7;
    const TOTAL: usize = CLEARS * BLOCKS;

    fn scheduled_cpu_ns() -> u64 {
        std::thread::sleep(Duration::from_micros(1));
        std::fs::read_to_string("/proc/thread-self/schedstat")
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }

    let (mut f, memory, found) = scene(GobLayout::Switch2).expect("benchmark needs Vulkan");
    let r = &mut f.rasterizer;
    let routes = [ReplayAdmission::UseStandard, ReplayAdmission::Recorded];
    let mut standard = Standard::new(&memory, GREEN);
    let workload = packet(&memory, GREEN, release(1));
    let mut block = |r: &mut VulkanRasterizer, route| {
        for _ in 0..CLEARS {
            let used = replay_or_standard(r, &workload, |r| standard.record(r)).unwrap();
            assert_eq!(used, route);
        }
        r.flush_commands();
    };
    for route in routes {
        if route == ReplayAdmission::Recorded {
            authorize(r, &memory, found);
        }
        block(r, route);
        r.release_fences(true);
    }
    let allocations = r.device.memory_counts();
    let objects = r.device.memory_objects();
    let cache_generation = r.texture_cache.generation();
    let mut cpu_samples = [[0.; SAMPLES]; 2];
    let mut wall_samples = [[0.; SAMPLES]; 2];
    for sample in 0..SAMPLES {
        for i in [sample % 2, 1 - sample % 2] {
            let route = routes[i];
            if route == ReplayAdmission::Recorded {
                authorize(r, &memory, found);
            } else {
                r.nvn2_target = None;
            }
            let traffic = r.texture_cache.traffic();
            let submits = r.scheduler.submits();
            let mut cpu = 0;
            let mut wall = Duration::ZERO;
            for _ in 0..BLOCKS {
                let start_cpu = scheduled_cpu_ns();
                let start_wall = Instant::now();
                block(r, route);
                wall += start_wall.elapsed();
                cpu += scheduled_cpu_ns() - start_cpu;
                r.release_fences(true);
            }
            assert_eq!(r.scheduler.submits() - submits, BLOCKS as u64);
            assert_eq!(r.texture_cache.generation(), cache_generation);
            assert_eq!(r.device.memory_counts(), allocations);
            assert_eq!(r.device.memory_objects(), objects);
            assert_eq!(r.texture_cache.traffic(), traffic);
            cpu_samples[i][sample] = cpu as f64 / TOTAL as f64;
            wall_samples[i][sample] = wall.as_nanos() as f64 / TOTAL as f64;
        }
    }
    for (route, (cpu, wall)) in ["Standard", "Direct"]
        .into_iter()
        .zip(cpu_samples.iter_mut().zip(wall_samples.iter_mut()))
    {
        cpu.sort_by(f64::total_cmp);
        wall.sort_by(f64::total_cmp);
        eprintln!(
            "{route}: {:.1} recorder CPU ns / logical submit, {:.1} wall ns / logical submit; \
             {TOTAL} clears in {BLOCKS} Vulkan submits per sample; no new cache images, no change \
             in live Vulkan allocations, no guest texture upload/writeback bytes in the timed loop",
            cpu[SAMPLES / 2],
            wall[SAMPLES / 2],
        );
    }
    let tiling = Tiling::new(64, 32, 1, 4, 2, 0, GobLayout::Switch2);
    r.flush_region(DADDR, tiling.size_bytes() as u64);
    assert!(
        guest_linear(&f, DADDR, &tiling)
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| *p == [0, 255, 0, 255])
    );
    f.assert_no_validation_errors();
}
