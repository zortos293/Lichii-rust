//! Submission counts: how many queue submissions the GPU thread makes, and that every wait
//! still sees the work it waits for.

use lichii_gpu::backend::{FenceAction, Rasterizer};
use lichii_texture::block_linear::GobLayout;

use super::VulkanRasterizer;
use super::tests::{Fixture, clear, clear_regs, fixture, gmmu};

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

#[test]
fn fences_after_submitted_work_need_no_submission() {
    let Some(mut f) = fixture(GobLayout::Tegra) else {
        return;
    };
    f.rasterizer.flush_commands();
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
