use std::sync::Arc;

use lichii_common::GpuVAddr;

use super::FenceAction;
use crate::gmmu::Gmmu;

/// An owned, fixed-size NVN2 submission: one clear and its completion action.
/// A generation detects stale work but does not pin mappings. The producer must own
/// the target, its mapping and its physical backing through fence retirement and
/// order any required writeback or invalidation before reusing the backing. For now
/// only the synthetic producer can provide that guarantee.
pub struct Nvn2Submission {
    /// Address space owning the render target.
    pub memory: Arc<Gmmu>,
    /// Target address in that address space.
    pub target: GpuVAddr,
    /// Mapping generation when the target was recorded.
    pub mapping_generation: u64,
    /// Clear value in RGBA order.
    pub color: [f32; 4],
    /// RGBA write bits (red is bit zero).
    pub color_mask: u8,
    /// Completion to enqueue only after the operation has been recorded.
    pub fence: FenceAction,
}

/// The renderer's decision for a complete packet.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplayAdmission {
    /// The renderer recorded the operation and queued its completion action.
    Recorded,
    /// No work or completion was recorded. The caller may run Standard in place.
    UseStandard,
}

/// Terminal recording failure. Some work may already have been recorded, so neither
/// replay nor fallback is safe. This does not signal the packet's completion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReplayError;
