use std::sync::{Arc, mpsc};
use std::time::Duration;

use gpu_allocator::MemoryLocation;
use lichii_gpu::backend::{ChannelId, ConstBufferBinding, GraphicsContext, ReleaseSize};
use lichii_gpu::engines::maxwell_3d::regs::{MAX_CONST_BUFFERS, MAX_SHADER_STAGE};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};
use lichii_texture::block_linear::Tiling;

use super::*;
use crate::Device;
use crate::commands::{ImageAccess, transfer_to_all_barrier};
use crate::memory::Buffer;
use crate::rasterizer::tests::{DADDR, Fixture, GPU_VA, clear_regs, fixture, gmmu, guest_linear};
use crate::scheduler::SemaphoreWait;
use crate::texture_cache::ImageInfo;

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

    fn record(&mut self, renderer: &mut VulkanRasterizer) {
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

fn resident<'a>(f: &Fixture, memory: &'a Gmmu, found: Found) -> ResidentTarget<'a> {
    ResidentTarget {
        found,
        memory,
        mapping_generation: memory.generation(),
        cache_generation: f.rasterizer.texture_cache.generation(),
        backing: DADDR,
    }
}

fn clear<'a>(target: Option<&'a ResidentTarget<'a>>, color: [f32; 4]) -> Command<'a> {
    Command::ClearColor {
        target,
        channels: vk::ColorComponentFlags::RGBA,
        color,
    }
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

fn run_ordered(requested: bool, layout: GobLayout, expected: Route) -> Option<Vec<u8>> {
    let (mut f, memory, found) = scene(layout)?;
    let target = resident(&f, &memory, found);
    let r = &mut f.rasterizer;
    let route = Route::negotiate(requested, layout, r.device.format_features(FORMAT));
    assert_eq!(route, expected);
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
        let used = submit(
            r,
            route,
            &[clear(Some(&target), color)],
            release(i as u32 + 1),
            |r| {
                assert_eq!(expected, Route::Standard);
                standard.record(r);
            },
        )
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
    let Some(standard) = run_ordered(false, GobLayout::Switch2, Route::Standard) else {
        return;
    };
    assert_eq!(
        run_ordered(true, GobLayout::Switch2, Route::Direct),
        Some(standard.clone())
    );
    assert_eq!(
        run_ordered(true, GobLayout::Tegra, Route::Standard),
        Some(standard)
    );
}

#[test]
fn rejected_submissions_enter_standard_before_recording_or_enqueuing_a_fence() {
    const REMAPPED: DAddr = DADDR + 0x40_0000;
    let layout = GobLayout::Switch2;
    assert_eq!(Route::default(), Route::Standard);
    let Some((mut f, memory, found)) = scene(layout) else {
        return;
    };
    let target = resident(&f, &memory, found);
    let r = &mut f.rasterizer;
    let missing = Route::negotiate(true, layout, vk::FormatFeatureFlags::COLOR_ATTACHMENT);
    let direct = Route::negotiate(true, layout, r.device.format_features(FORMAT));
    assert_eq!(direct, Route::Direct);
    let mut standard = Standard::new(&memory, GREEN);
    let unsupported_resource = ResidentTarget {
        found: Found { layer: 1, ..found },
        ..target
    };
    let mut calls = 0;
    for (i, (route, commands)) in [
        (missing, [clear(Some(&target), GREEN)].as_slice()),
        (
            direct,
            [clear(Some(&target), GREEN), Command::Unsupported].as_slice(),
        ),
        (direct, [clear(None, GREEN)].as_slice()),
        (
            direct,
            [clear(Some(&unsupported_resource), GREEN)].as_slice(),
        ),
        (
            direct,
            [Command::ClearColor {
                target: Some(&target),
                color: GREEN,
                channels: vk::ColorComponentFlags::G,
            }]
            .as_slice(),
        ),
        (direct, [clear(Some(&target), BLUE)].as_slice()),
    ]
    .into_iter()
    .enumerate()
    {
        if i == 5 {
            memory.map(GPU_VA, REMAPPED, 0x10000, 0, false).unwrap();
            assert_ne!(memory.generation(), target.mapping_generation);
            standard.regs.clear_color = BLUE;
        }
        let tick = r.scheduler.tick();
        let generations = r.texture_cache.generation();
        let was_modified = r.texture_cache.state(found.id).unwrap().gpu_modified;
        assert_eq!(
            submit(r, route, commands, release(1), |r| {
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
                if commands.len() > 1 {
                    r.barrier();
                }
            })
            .unwrap(),
            Route::Standard
        );
        assert_eq!(r.fences.len(), 1);
        r.release_fences(true);
    }
    assert_eq!(calls, 6);
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
    let target = resident(&f, &memory, found);
    let r = &mut f.rasterizer;
    let direct = Route::negotiate(true, r.gob_layout, r.device.format_features(FORMAT));
    assert_eq!(direct, Route::Direct);
    let mut standard = Standard::new(&memory, GREEN);
    let workload = [clear(Some(&target), GREEN)];
    let mut block = |r: &mut VulkanRasterizer, route| {
        for _ in 0..CLEARS {
            let used = submit(r, route, &workload, release(1), |r| standard.record(r)).unwrap();
            assert_eq!(used, route);
        }
        r.flush_commands();
    };
    for route in [Route::Standard, direct] {
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
            let route = [Route::Standard, direct][i];
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
    for (route, (cpu, wall)) in [Route::Standard, direct]
        .into_iter()
        .zip(cpu_samples.iter_mut().zip(wall_samples.iter_mut()))
    {
        cpu.sort_by(f64::total_cmp);
        wall.sort_by(f64::total_cmp);
        eprintln!(
            "{route:?}: {:.1} recorder CPU ns / logical submit, {:.1} wall ns / logical submit; \
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
