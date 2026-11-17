//! Submission counts: how many queue submissions the GPU thread makes, and that every wait
//! still sees the work it waits for.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ash::vk;
use lichii_gpu::backend::{FenceAction, FramebufferConfig, Rasterizer, ReleaseSize};
use lichii_gpu::engines::maxwell_3d::regs::{ClearControl, ScissorTestBits, ScissorTestBits1};
use lichii_texture::block_linear::{GobLayout, Tiling};

use super::VulkanRasterizer;
use super::tests::{
    DADDR, Fixture, clear, clear_regs, fixture, framebuffer, gmmu, guest_linear, unorm8_eq,
};
use crate::device::Device;
use crate::scheduler::SemaphoreWait;

const SEMAPHORE: u64 = DADDR + 0x80_0000;

/// What the back thread does when its stream runs dry (`Back::poll_fences`).
fn idle(r: &mut VulkanRasterizer) {
    if !r.has_pending_fences() {
        return;
    }
    if r.has_unsubmitted_fences() {
        r.flush_commands();
    }
    r.release_fences(false);
}

fn submits(f: &Fixture) -> u64 {
    f.rasterizer.scheduler.submits()
}

fn release(value: u64) -> FenceAction {
    FenceAction::WriteSemaphore {
        address: SEMAPHORE,
        value,
        size: ReleaseSize::Four,
    }
}

/// A fixture whose start-up work is submitted.
fn submitted(gob_layout: GobLayout) -> Option<Fixture> {
    let mut f = fixture(gob_layout)?;
    f.rasterizer.flush_commands();
    Some(f)
}

fn fill(f: &mut Fixture, color: [f32; 4]) {
    let gmmu = gmmu(f);
    clear(f, &gmmu, &clear_regs(96, 48, 2, color));
}

fn texel(f: &mut Fixture) -> [u8; 4] {
    let tiling = Tiling::new(96, 48, 1, 4, 2, 0, GobLayout::Switch2);
    f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
    guest_linear(f, DADDR, &tiling)[..4].try_into().unwrap()
}

/// A timeline semaphore the host signals, which holds back the host GPU work waiting for it.
/// Dropping it opens it and waits for the device, so a failing test does not hang.
struct Gate {
    device: Arc<Device>,
    raw: vk::Semaphore,
    open: bool,
}

impl Gate {
    fn new(device: &Arc<Device>) -> Self {
        let mut timeline = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline);
        // SAFETY: valid create info on a live device.
        let raw = unsafe { device.raw.create_semaphore(&info, None).unwrap() };
        Self {
            device: device.clone(),
            raw,
            open: false,
        }
    }

    fn wait(&self) -> SemaphoreWait {
        SemaphoreWait {
            semaphore: self.raw,
            value: 1,
            stage: vk::PipelineStageFlags2::ALL_COMMANDS,
        }
    }

    fn open(&mut self) {
        if !std::mem::replace(&mut self.open, true) {
            let signal = vk::SemaphoreSignalInfo::default()
                .semaphore(self.raw)
                .value(1);
            // SAFETY: only the host signals this semaphore, once.
            unsafe { self.device.raw.signal_semaphore(&signal).unwrap() };
        }
    }
}

impl Drop for Gate {
    fn drop(&mut self) {
        self.open();
        self.device.wait_idle();
        // SAFETY: the device is idle, so no submission waits on the semaphore any more.
        unsafe { self.device.raw.destroy_semaphore(self.raw, None) };
    }
}

#[test]
fn fences_after_submitted_work_need_no_submission() {
    let Some(mut f) = submitted(GobLayout::Tegra) else {
        return;
    };
    let before = submits(&f);
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(4));
    assert!(!f.rasterizer.has_unsubmitted_fences());
    f.rasterizer.flush_commands();
    idle(&mut f.rasterizer);
    f.rasterizer.barrier();
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(4));
    f.rasterizer.flush_commands();
    f.rasterizer.release_fences(true);
    assert_eq!(submits(&f), before, "a fence alone submits nothing");
    assert_eq!(f.syncpoints.guest().value(4).unwrap(), 2);

    let gmmu = gmmu(&f);
    clear(&mut f, &gmmu, &clear_regs(64, 64, 2, [1.0, 0.0, 0.0, 1.0]));
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(4));
    assert!(f.rasterizer.has_unsubmitted_fences());
    f.rasterizer.flush_commands();
    assert_eq!(
        submits(&f),
        before + 1,
        "the barrier rides on the next work"
    );
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(4).unwrap(), 3);
    f.assert_no_validation_errors();
}

/// A fence signalled after submitted work waits for that work's tick, held back on the host GPU
/// here, and one signalled after later recorded work waits for that work.
#[test]
fn fences_on_a_submitted_tick_signal_once_it_finishes() {
    let Some(mut f) = submitted(GobLayout::Switch2) else {
        return;
    };
    let mut gate = Gate::new(&f.rasterizer.device);
    fill(&mut f, [1.0, 0.0, 0.0, 1.0]);
    let tick = f
        .rasterizer
        .scheduler
        .flush_with(&[gate.wait()], &[])
        .unwrap();
    let before = submits(&f);
    f.rasterizer.flush_commands();
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(4));
    assert!(
        !f.rasterizer.has_unsubmitted_fences(),
        "the fence takes the submitted tick"
    );
    fill(&mut f, [0.0, 1.0, 0.0, 1.0]);
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(5));
    assert!(
        f.rasterizer.has_unsubmitted_fences(),
        "the later fence waits for recorded work"
    );
    f.rasterizer.release_fences(false);
    assert_eq!(
        f.syncpoints.guest().value(4).unwrap(),
        0,
        "the tick is held back"
    );

    gate.open();
    f.rasterizer.scheduler.wait(tick).unwrap();
    f.rasterizer.release_fences(false);
    assert_eq!(
        f.syncpoints.guest().value(4).unwrap(),
        1,
        "the tick finished"
    );
    assert_eq!(
        f.syncpoints.guest().value(5).unwrap(),
        0,
        "the later fence still waits for its unsubmitted work"
    );
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(5).unwrap(), 1);
    assert_eq!(submits(&f), before + 1);
    assert!(unorm8_eq(texel(&mut f), [0, 255, 0, 255]));
    f.assert_no_validation_errors();
}

/// `release_fences(true)` and `synchronize_region` land a release whether its fence waits for
/// recorded work, which they submit, or for submitted work, when unrelated recorded work stays
/// unsubmitted.
#[test]
fn fence_waits_submit_only_unsubmitted_work() {
    let Some(mut f) = submitted(GobLayout::Switch2) else {
        return;
    };
    type Wait = fn(&mut VulkanRasterizer);
    let waits: [(&str, Wait); 2] = [
        ("release_fences(true)", |r| r.release_fences(true)),
        ("synchronize_region", |r| r.synchronize_region(SEMAPHORE, 4)),
    ];
    let mut payload = 0x50;
    for (name, wait) in waits {
        payload += 1;
        fill(&mut f, [1.0, 0.0, 0.0, 1.0]);
        f.rasterizer.signal_fence(release(payload));
        let before = submits(&f);
        wait(&mut f.rasterizer);
        assert_eq!(
            submits(&f),
            before + 1,
            "{name} submits the work its fence waits for"
        );
        assert_eq!(
            f.memory.read::<u32>(SEMAPHORE).unwrap(),
            payload as u32,
            "{name}"
        );

        payload += 1;
        f.rasterizer.signal_fence(release(payload));
        fill(&mut f, [0.0, 0.0, 1.0, 1.0]);
        let before = submits(&f);
        wait(&mut f.rasterizer);
        assert_eq!(
            submits(&f),
            before,
            "{name} leaves unrelated work unsubmitted"
        );
        assert_eq!(
            f.memory.read::<u32>(SEMAPHORE).unwrap(),
            payload as u32,
            "{name}"
        );
        assert!(f.rasterizer.scheduler.submitted_tick().is_none());
        f.rasterizer.flush_commands();
    }
    f.assert_no_validation_errors();
}

/// A guest barrier with nothing recorded opens no command buffer and submits nothing; the
/// clears around it land in order.
#[test]
fn barriers_with_nothing_recorded_wait_for_the_next_work() {
    let Some(mut f) = submitted(GobLayout::Switch2) else {
        return;
    };
    fill(&mut f, [1.0, 0.0, 0.0, 1.0]);
    f.rasterizer.flush_commands();
    let before = submits(&f);
    f.rasterizer.barrier();
    assert!(
        f.rasterizer.scheduler.submitted_tick().is_some(),
        "the barrier records nothing yet"
    );
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(6));
    f.rasterizer.flush_commands();
    assert_eq!(submits(&f), before);
    let gmmu = gmmu(&f);
    let mut regs = clear_regs(96, 48, 2, [0.0, 0.0, 1.0, 1.0]);
    regs.clear_control = ClearControl(1 << 8);
    regs.scissor_test[0].enable = 1;
    regs.scissor_test[0].bits = ScissorTestBits(10 | (40 << 16));
    regs.scissor_test[0].bits1 = ScissorTestBits1(5 | (20 << 16));
    clear(&mut f, &gmmu, &regs);
    f.rasterizer.flush_commands();
    assert_eq!(submits(&f), before + 1);
    let tiling = Tiling::new(96, 48, 1, 4, 2, 0, GobLayout::Switch2);
    f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
    for (i, &texel) in guest_linear(&f, DADDR, &tiling)
        .as_chunks::<4>()
        .0
        .iter()
        .enumerate()
    {
        let (x, y) = (i as u32 % 96, i as u32 / 96);
        let want = if (10..40).contains(&x) && (5..20).contains(&y) {
            [0, 0, 255, 255]
        } else {
            [255, 0, 0, 255]
        };
        assert!(unorm8_eq(texel, want), "texel ({x}, {y}): {texel:?}");
    }
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(6).unwrap(), 1);
    f.assert_no_validation_errors();
}

/// A CPU read's flush submits the recorded work, so the next flush has nothing to submit; work
/// recorded after the read still goes to the host GPU at the next flush.
#[test]
fn flushes_after_a_cpu_read_submit_only_later_work() {
    let Some(mut f) = submitted(GobLayout::Switch2) else {
        return;
    };
    fill(&mut f, [1.0, 0.0, 0.0, 1.0]);
    assert!(unorm8_eq(texel(&mut f), [255, 0, 0, 255]));
    let before = submits(&f);
    f.rasterizer.flush_commands();
    assert_eq!(submits(&f), before, "the read submitted everything");

    fill(&mut f, [0.0, 1.0, 0.0, 1.0]);
    f.rasterizer.flush_commands();
    assert_eq!(submits(&f), before + 1, "later work is submitted");
    assert!(f.rasterizer.scheduler.submitted_tick().is_some());
    assert!(unorm8_eq(texel(&mut f), [0, 255, 0, 255]));
    f.assert_no_validation_errors();
}

/// Why a submission went to the queue in [`frame_model`].
#[derive(Clone, Copy)]
enum Cause {
    EndOfSubmission,
    IdlePoll,
    Present,
    Operation,
}

const CAUSES: [&str; 4] = ["end of submission", "idle poll", "present", "operation"];

/// Queue submissions by [`Cause`] and the time spent in renderer calls.
#[derive(Default)]
struct FrameCounts {
    submits: [u64; 4],
    calls: Duration,
    frames: u32,
}

impl FrameCounts {
    fn per_frame(&self) -> [f64; 4] {
        self.submits.map(|n| n as f64 / f64::from(self.frames))
    }

    fn report(&self, label: &str) {
        let per_frame = self.per_frame();
        let causes: Vec<String> = CAUSES
            .iter()
            .zip(per_frame)
            .map(|(cause, n)| format!("{cause} {n:.2}"))
            .collect();
        eprintln!(
            "{label}: {:.2} submits/frame ({}), renderer calls {:.0} µs/frame",
            per_frame.iter().sum::<f64>(),
            causes.join(", "),
            self.calls.as_secs_f64() * 1e6 / f64::from(self.frames)
        );
    }
}

/// A Switch 2 frame as the back thread sees it (LIRU-6's MKW and DKB traces): four GPFIFO
/// submissions of render work, each of three command buffers the guest gates with a semaphore
/// it releases once it wrote them, so the front stops at each gate and the idle back polls its
/// fences; each buffer ends in a semaphore release, each submission in a wait for idle, a
/// syncpoint increment and its end. Then two fence-only submissions from the present thread
/// and a present. A clear stands for each draw, and `spin` for a draw's own back time.
fn frame_model(f: &mut Fixture, frames: u32, draws_per_buffer: u32, spin: Duration) -> FrameCounts {
    let gmmu = gmmu(f);
    let (width, height) = (64, 320);
    let regs = clear_regs(width, height, 3, [0.5, 0.25, 1.0, 1.0]);
    let fb = FramebufferConfig {
        block_height_log2: Some(3),
        ..framebuffer(width, height)
    };
    let mut counts = FrameCounts {
        frames,
        ..FrameCounts::default()
    };
    let mut timed = |f: &mut Fixture, cause: Cause, call: &mut dyn FnMut(&mut Fixture)| {
        let (before, start) = (submits(f), Instant::now());
        call(f);
        counts.calls += start.elapsed();
        counts.submits[cause as usize] += submits(f) - before;
    };
    let spin_for = |time: Duration| {
        let start = Instant::now();
        while start.elapsed() < time {
            std::hint::spin_loop();
        }
    };
    let fence_only = |f: &mut Fixture, id| {
        f.rasterizer.release_fences(false);
        f.rasterizer.barrier();
        f.rasterizer
            .signal_fence(FenceAction::IncrementSyncpoint(id));
    };
    let increments = f.syncpoints.guest().value(1).unwrap();
    let mut payload = 0;
    for _ in 0..frames {
        for _ in 0..4 {
            for _ in 0..3 {
                timed(f, Cause::IdlePoll, &mut |f| idle(&mut f.rasterizer));
                for _ in 0..draws_per_buffer {
                    timed(f, Cause::Operation, &mut |f| clear(f, &gmmu, &regs));
                    spin_for(spin);
                }
                payload += 1;
                timed(f, Cause::Operation, &mut |f| {
                    f.rasterizer.signal_fence(release(payload))
                });
            }
            timed(f, Cause::Operation, &mut |f| fence_only(f, 1));
            timed(f, Cause::EndOfSubmission, &mut |f| {
                f.rasterizer.flush_commands()
            });
        }
        for _ in 0..2 {
            timed(f, Cause::Operation, &mut |f| fence_only(f, 2));
            timed(f, Cause::EndOfSubmission, &mut |f| {
                f.rasterizer.flush_commands()
            });
        }
        timed(f, Cause::Present, &mut |f| f.rasterizer.present(&fb));
        timed(f, Cause::IdlePoll, &mut |f| idle(&mut f.rasterizer));
    }
    f.rasterizer.release_fences(true);
    assert_eq!(f.memory.read::<u32>(SEMAPHORE).unwrap(), payload as u32);
    assert_eq!(
        f.syncpoints.guest().value(1).unwrap() - increments,
        4 * frames,
        "every syncpoint increment lands"
    );
    f.assert_no_validation_errors();
    counts
}

/// Main before ZOR-430 (7c61ca5) submitted 15 times a frame here: 6 ends of submissions (the
/// two fence-only ones each a barrier and a fence), 8 idle polls and the present.
#[test]
fn the_frame_model_submits_13_times_a_frame_against_15_before() {
    let Some(mut f) = submitted(GobLayout::Tegra) else {
        return;
    };
    let counts = frame_model(&mut f, 2, 2, Duration::ZERO);
    counts.report("frame model");
    assert_eq!(
        counts.per_frame(),
        [4.0, 8.0, 1.0, 0.0],
        "each render submission's end, each gate after a buffer and the present submit; \
         fence-only ends submit nothing"
    );
}

/// `cargo test -p lichii-vulkan --release frame_model_bench -- --ignored --nocapture`: the
/// model at MKW's ~2,000 draws a frame, without and with each draw's back time (~12.7 µs on
/// MKW and DKB) spun.
#[test]
#[ignore = "bench"]
fn frame_model_bench() {
    for (label, spin) in [
        ("no draw time", Duration::ZERO),
        ("12.7 µs a draw", Duration::from_nanos(12_700)),
    ] {
        let Some(mut f) = submitted(GobLayout::Tegra) else {
            return;
        };
        frame_model(&mut f, 2, 166, spin);
        frame_model(&mut f, 20, 166, spin).report(label);
    }
}
