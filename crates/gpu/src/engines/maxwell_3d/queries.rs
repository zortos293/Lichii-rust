//! Report semaphores, counter resets and conditional rendering.

use super::regs::{
    ClearReport, Regs, RenderEnableMode, RenderEnableOverride, ReportSemaphoreCompare,
    ReportSemaphoreOperation, ReportSemaphoreReport,
};
use crate::activity::Wait;
use crate::backend::{Query, QueryKind, ReleaseSize, RenderCondition};
use crate::engines::EngineContext;
use crate::gmmu::{Gmmu, RangeTarget};

/// The write REPORT_SEMAPHORE.QUERY asks for. Acquires are waited on by the front thread and
/// traps are not emulated, so both give `None`.
///
/// Releases to address 0 also give `None` while it is unmapped: NVN2 titles release fences to a
/// null address every frame (Mario Tennis Fever: three per present), which the reference drops
/// silently. A guest that maps address 0 still gets its reports.
pub(super) fn report(regs: &Regs, memory: &Gmmu) -> Option<Query> {
    let semaphore = &regs.report_semaphore;
    let query = semaphore.query;
    let operation = query.operation();
    if operation != ReportSemaphoreOperation::RELEASE
        && operation != ReportSemaphoreOperation::REPORT_ONLY
    {
        return None;
    }
    if semaphore.address() == 0 && memory.translate(0).is_none() {
        return None;
    }
    let kind = match query.report() {
        ReportSemaphoreReport::PAYLOAD => QueryKind::Payload,
        ReportSemaphoreReport::Z_PASS_PIXEL_COUNT | ReportSemaphoreReport::Z_PASS_PIXEL_COUNT64 => {
            QueryKind::SamplesPassed
        }
        ReportSemaphoreReport::PRIMITIVES_GENERATED => QueryKind::PrimitivesGenerated,
        ReportSemaphoreReport::STREAMING_PRIMITIVES_SUCCEEDED => {
            QueryKind::TransformFeedbackPrimitivesWritten
        }
        other => QueryKind::Counter(other.0),
    };
    Some(Query {
        address: semaphore.address(),
        kind,
        payload: u64::from(semaphore.payload),
        size: if query.short_query() != 0 {
            ReleaseSize::Four
        } else {
            ReleaseSize::Sixteen
        },
        after_work: operation == ReportSemaphoreOperation::RELEASE,
    })
}

/// The counter CLEAR_REPORT_VALUE resets, when the renderer keeps one for it.
pub(super) fn counter(clear: ClearReport) -> Option<QueryKind> {
    Some(match clear {
        ClearReport::Z_PASS_PIXEL_COUNT => QueryKind::SamplesPassed,
        ClearReport::PRIMITIVES_GENERATED => QueryKind::PrimitivesGenerated,
        ClearReport::STREAMING_PRIMITIVES_SUCCEEDED => {
            QueryKind::TransformFeedbackPrimitivesWritten
        }
        ClearReport::VTG_PRIMITIVES_OUT => {
            QueryKind::Counter(ReportSemaphoreReport::VTG_PRIMITIVES_OUT.0)
        }
        _ => return None,
    })
}

/// RENDER_ENABLE as evaluated when its mode was last written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RenderEnable {
    Host(bool),
    /// The renderer tests `condition` on the host GPU.
    Gpu(RenderCondition),
}

/// Evaluates RENDER_ENABLE when its mode is written; conditional modes compare the two
/// sequence/mode pairs at its address as they are now.
pub(super) fn render_enable(regs: &Regs, ctx: &mut EngineContext<'_>) -> bool {
    let mode = regs.render_enable.mode;
    if mode == RenderEnableMode::FALSE {
        return false;
    }
    if mode != RenderEnableMode::CONDITIONAL
        && mode != RenderEnableMode::IF_EQUAL
        && mode != RenderEnableMode::IF_NOT_EQUAL
    {
        return true;
    }
    let c = read_synchronized::<ReportSemaphoreCompare>(ctx, regs.render_enable.address());
    match mode {
        RenderEnableMode::CONDITIONAL => c.initial_sequence != 0 && c.initial_mode != 0,
        RenderEnableMode::IF_EQUAL => {
            c.initial_sequence == c.current_sequence && c.initial_mode == c.current_mode
        }
        _ => c.initial_sequence != c.current_sequence || c.initial_mode != c.current_mode,
    }
}

/// Evaluates a GPU render condition on the host, waiting for the host GPU to compute its
/// predicate.
pub(super) fn host_condition(ctx: &mut EngineContext<'_>, condition: RenderCondition) -> bool {
    let _waiting = ctx.activity.waiting(Wait::HostGpu);
    ctx.rasterizer.predicate_value(condition.predicate) != condition.inverted
}

/// Reads a `T` at `address` once the renderer wrote everything it holds for it.
///
/// The renderer may still hold the results the value depends on (query releases waiting for a
/// fence, GPU writes in its caches), so this submits the recorded work, waits for the fences and
/// writes the range back before reading guest memory.
fn read_synchronized<T: bytemuck::Pod>(ctx: &mut EngineContext<'_>, address: u64) -> T {
    ctx.rasterizer.flush_commands();
    let waiting = ctx.activity.waiting(Wait::HostGpu);
    ctx.rasterizer.release_fences(true);
    ctx.memory
        .for_each_range(address, size_of::<T>() as u64, |range| {
            if let RangeTarget::Device(daddr) = range.target {
                ctx.rasterizer.flush_region(daddr, range.size);
            }
        });
    let value = ctx.memory.read::<T>(address).unwrap_or_else(|fault| {
        tracing::warn!(?fault, "render condition at unmapped GPU memory");
        bytemuck::Zeroable::zeroed()
    });
    drop(waiting);
    value
}

/// Whether draws and clears run, and under which condition the renderer tests: the override
/// register wins over the evaluated condition.
#[inline]
pub(super) fn renders(regs: &Regs, render_enable: RenderEnable) -> Option<Option<RenderCondition>> {
    match (regs.render_enable_override, render_enable) {
        (RenderEnableOverride::ALWAYS_RENDER, _) => Some(None),
        (RenderEnableOverride::NEVER_RENDER, _) => None,
        (_, RenderEnable::Host(true)) => Some(None),
        (_, RenderEnable::Host(false)) => None,
        (_, RenderEnable::Gpu(condition)) => Some(Some(condition)),
    }
}
