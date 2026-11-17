//! Submission batching: how many queue submissions the GPU thread makes, and that every wait
//! still sees the work it waits for.

use std::time::{Duration, Instant};

use lichii_gpu::backend::{FenceAction, FramebufferConfig, Rasterizer, ReleaseSize};
use lichii_gpu::engines::maxwell_3d::regs::{ClearControl, ScissorTestBits, ScissorTestBits1};
use lichii_texture::block_linear::{GobLayout, Tiling};

use super::VulkanRasterizer;
use super::tests::{
    DADDR, Fixture, clear, clear_regs, fixture, framebuffer, gmmu, guest_linear, unorm8_eq,
};

const SEMAPHORE: u64 = DADDR + 0x80_0000;
const HOUR: Duration = Duration::from_secs(3600);

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

/// A fixture whose start-up work is submitted, holding ends of submissions back for `window`.
fn batching(gob_layout: GobLayout, window: Duration) -> Option<Fixture> {
    let mut f = fixture(gob_layout)?;
    f.rasterizer.submit_window = window;
    f.rasterizer.flush_commands();
    Some(f)
}

#[test]
fn fences_after_submitted_work_need_no_submission() {
    let Some(mut f) = batching(GobLayout::Tegra, HOUR) else {
        return;
    };
    let before = submits(&f);
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(4));
    assert!(!f.rasterizer.has_unsubmitted_fences());
    f.rasterizer.end_submission();
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
    f.rasterizer.flush_commands();
    assert_eq!(
        submits(&f),
        before + 1,
        "the barrier rides on the next work"
    );
    f.assert_no_validation_errors();
}

#[test]
fn ends_of_submissions_close_together_share_one_submission() {
    let Some(mut f) = batching(GobLayout::Tegra, HOUR) else {
        return;
    };
    let gmmu = gmmu(&f);
    let before = submits(&f);
    for i in 0..4 {
        clear(
            &mut f,
            &gmmu,
            &clear_regs(64, 64, 2, [i as f32 / 4.0, 0.0, 0.0, 1.0]),
        );
        f.rasterizer
            .signal_fence(FenceAction::IncrementSyncpoint(6));
        f.rasterizer.end_submission();
    }
    assert_eq!(submits(&f), before, "the ends are held back");
    assert!(f.rasterizer.has_pending_fences() && f.rasterizer.has_unsubmitted_fences());
    f.rasterizer.release_fences(false);
    assert_eq!(f.syncpoints.guest().value(6).unwrap(), 0);

    idle(&mut f.rasterizer);
    assert_eq!(submits(&f), before + 1, "an idle GPU submits them at once");
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(6).unwrap(), 4);
    assert!(!f.rasterizer.has_pending_fences());
    f.assert_no_validation_errors();
}

#[test]
fn deferred_work_without_fences_still_reaches_an_idle_gpu() {
    let Some(mut f) = batching(GobLayout::Tegra, HOUR) else {
        return;
    };
    let gmmu = gmmu(&f);
    clear(&mut f, &gmmu, &clear_regs(64, 64, 2, [0.0, 1.0, 0.0, 1.0]));
    let before = submits(&f);
    f.rasterizer.end_submission();
    assert!(
        f.rasterizer.has_pending_fences(),
        "the idle GPU keeps polling"
    );
    idle(&mut f.rasterizer);
    assert_eq!(submits(&f), before + 1);
    assert!(!f.rasterizer.has_pending_fences());
}

#[test]
fn ends_of_submissions_submit_at_once_with_no_window() {
    let Some(mut f) = batching(GobLayout::Tegra, Duration::ZERO) else {
        return;
    };
    let gmmu = gmmu(&f);
    let before = submits(&f);
    for _ in 0..3 {
        clear(&mut f, &gmmu, &clear_regs(64, 64, 2, [0.0, 0.0, 1.0, 1.0]));
        f.rasterizer
            .signal_fence(FenceAction::IncrementSyncpoint(7));
        f.rasterizer.end_submission();
    }
    assert_eq!(submits(&f), before + 3);
    assert!(!f.rasterizer.has_unsubmitted_fences());
}

#[test]
fn deferred_work_is_submitted_once_the_window_passes() {
    let window = Duration::from_millis(20);
    let Some(mut f) = batching(GobLayout::Tegra, window) else {
        return;
    };
    let gmmu = gmmu(&f);
    let regs = clear_regs(64, 64, 2, [0.0, 0.0, 1.0, 1.0]);
    clear(&mut f, &gmmu, &regs);
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(8));
    let before = submits(&f);
    f.rasterizer.end_submission();
    if f.rasterizer.scheduler.submitted_at().elapsed() >= window {
        eprintln!("skipping: the host ended the submission after the window");
        return;
    }
    assert_eq!(submits(&f), before);
    std::thread::sleep(window);
    clear(&mut f, &gmmu, &regs);
    assert_eq!(
        submits(&f),
        before + 1,
        "the next operation submits the work held back"
    );
    assert!(!f.rasterizer.has_unsubmitted_fences());
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(8).unwrap(), 1);
}

/// Each wait the guest or the front can make on deferred work submits it first and sees its
/// results.
#[test]
fn waits_on_deferred_work_submit_it_first() {
    let (width, height) = (96, 48);
    let tiling = Tiling::new(width, height, 1, 4, 2, 0, GobLayout::Switch2);
    let deferred = |f: &mut Fixture, color| {
        let gmmu = gmmu(f);
        clear(f, &gmmu, &clear_regs(width, height, 2, color));
        f.rasterizer.signal_fence(release(0x55));
        f.rasterizer
            .signal_fence(FenceAction::IncrementSyncpoint(9));
        f.rasterizer.end_submission();
        assert!(
            f.rasterizer.has_unsubmitted_fences(),
            "the work is held back"
        );
    };

    let Some(mut f) = batching(GobLayout::Switch2, HOUR) else {
        return;
    };
    deferred(&mut f, [1.0, 0.0, 0.0, 1.0]);
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(9).unwrap(), 1);
    assert_eq!(f.memory.read::<u32>(SEMAPHORE).unwrap(), 0x55);

    f.memory.write::<u32>(SEMAPHORE, 0).unwrap();
    deferred(&mut f, [0.0, 1.0, 0.0, 1.0]);
    f.rasterizer.synchronize_region(SEMAPHORE, 4);
    assert!(!f.rasterizer.has_unsubmitted_fences());
    assert_eq!(f.memory.read::<u32>(SEMAPHORE).unwrap(), 0x55);
    assert_eq!(f.syncpoints.guest().value(9).unwrap(), 2);

    deferred(&mut f, [0.0, 0.0, 1.0, 1.0]);
    f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
    let texel = guest_linear(&f, DADDR, &tiling)[..4].try_into().unwrap();
    assert!(
        unorm8_eq(texel, [0, 0, 255, 255]),
        "a CPU read sees the deferred clear: {texel:?}"
    );

    deferred(&mut f, [1.0, 1.0, 0.0, 1.0]);
    f.rasterizer
        .flush_for_cpu_read(DADDR, tiling.size_bytes() as u64);
    let texel = guest_linear(&f, DADDR, &tiling)[..4].try_into().unwrap();
    assert!(
        unorm8_eq(texel, [255, 255, 0, 255]),
        "a trapped read sees the deferred clear: {texel:?}"
    );
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(9).unwrap(), 4);
    f.assert_no_validation_errors();
}

#[test]
fn presents_submit_deferred_work_and_show_it() {
    let Some(mut f) = batching(GobLayout::Tegra, HOUR) else {
        return;
    };
    let gmmu = gmmu(&f);
    let (width, height) = (64, 320);
    clear(
        &mut f,
        &gmmu,
        &clear_regs(width, height, 3, [0.0, 1.0, 1.0, 1.0]),
    );
    f.rasterizer
        .signal_fence(FenceAction::IncrementSyncpoint(10));
    f.rasterizer.end_submission();
    let fb = FramebufferConfig {
        block_height_log2: Some(3),
        ..framebuffer(width, height)
    };
    f.rasterizer.present(&fb);
    assert!(!f.rasterizer.has_unsubmitted_fences());
    let (_, texels) = f.read_frame();
    assert!(
        texels
            .as_chunks::<4>()
            .0
            .iter()
            .all(|&t| unorm8_eq(t, [0, 255, 255, 255])),
        "the presented frame holds the deferred clear"
    );
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(10).unwrap(), 1);
}

/// Guest memory after a stream of overlapping scissored clears, flushed, with ends of
/// submissions held back for `window`.
fn scissored_stream(window: Duration) -> Option<Vec<u8>> {
    let (width, height) = (96, 48);
    let mut f = batching(GobLayout::Switch2, window)?;
    let gmmu = gmmu(&f);
    let mut regs = clear_regs(width, height, 2, [0.25, 0.5, 0.75, 1.0]);
    clear(&mut f, &gmmu, &regs);
    f.rasterizer.end_submission();
    regs.clear_control = ClearControl(1 << 8);
    regs.scissor_test[0].enable = 1;
    for i in 0..12u32 {
        regs.clear_color = [(i % 3) as f32 / 2.0, (i % 5) as f32 / 4.0, 1.0, 0.5];
        regs.scissor_test[0].bits = ScissorTestBits((i * 5) | ((i * 5 + 30) << 16));
        regs.scissor_test[0].bits1 = ScissorTestBits1((i * 2) | ((i * 2 + 20) << 16));
        clear(&mut f, &gmmu, &regs);
        f.rasterizer.barrier();
        f.rasterizer
            .signal_fence(FenceAction::IncrementSyncpoint(11));
        f.rasterizer.end_submission();
        if i % 4 == 3 {
            idle(&mut f.rasterizer);
        }
    }
    let tiling = Tiling::new(width, height, 1, 4, 2, 0, GobLayout::Switch2);
    f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
    f.rasterizer.release_fences(true);
    assert_eq!(f.syncpoints.guest().value(11).unwrap(), 12);
    f.assert_no_validation_errors();
    Some(guest_linear(&f, DADDR, &tiling))
}

#[test]
fn batching_keeps_the_bytes_the_guest_reads() {
    let Some(batched) = scissored_stream(HOUR) else {
        return;
    };
    let Some(unbatched) = scissored_stream(Duration::ZERO) else {
        return;
    };
    assert!(batched.iter().any(|&b| b != 0));
    assert_eq!(batched, unbatched);
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
    let increments = f.syncpoints.guest().value(1).unwrap();
    let mut payload = 0;
    let fence_only = |f: &mut Fixture, id| {
        f.rasterizer.release_fences(false);
        f.rasterizer.barrier();
        f.rasterizer
            .signal_fence(FenceAction::IncrementSyncpoint(id));
    };
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
                f.rasterizer.end_submission()
            });
        }
        for _ in 0..2 {
            timed(f, Cause::Operation, &mut |f| fence_only(f, 2));
            timed(f, Cause::EndOfSubmission, &mut |f| {
                f.rasterizer.end_submission()
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

#[test]
fn the_frame_model_submits_less_with_batching() {
    let model = |window| {
        let mut f = batching(GobLayout::Tegra, window)?;
        Some(frame_model(&mut f, 2, 2, Duration::ZERO))
    };
    let Some(unbatched) = model(Duration::ZERO) else {
        return;
    };
    let Some(batched) = model(HOUR) else {
        return;
    };
    unbatched.report("no window");
    batched.report("window");
    assert_eq!(
        unbatched.per_frame(),
        [4.0, 8.0, 1.0, 0.0],
        "each render submission's end, each later gate and the present submit; fence-only ends do not"
    );
    assert_eq!(
        batched.per_frame(),
        [0.0, 11.0, 1.0, 0.0],
        "each render end joins the next gate's submission, the last one the present's"
    );
}

/// `cargo test -p lichii-vulkan frame_model_bench -- --ignored --nocapture`: the model at
/// MKW's ~2,000 draws a frame, with and without each draw's back time (~12.7 µs on MKW and
/// DKB) spun, at no window and at the default one.
#[test]
#[ignore = "bench"]
fn frame_model_bench() {
    let draw = Duration::from_nanos(12_700);
    for (label, window, spin) in [
        ("no window, no draw time", Duration::ZERO, Duration::ZERO),
        (
            "default window, no draw time",
            super::submit_window(),
            Duration::ZERO,
        ),
        ("no window, 12.7 µs a draw", Duration::ZERO, draw),
        (
            "default window, 12.7 µs a draw",
            super::submit_window(),
            draw,
        ),
    ] {
        let Some(mut f) = batching(GobLayout::Tegra, window) else {
            return;
        };
        frame_model(&mut f, 2, 166, spin);
        frame_model(&mut f, 20, 166, spin).report(label);
    }
}
