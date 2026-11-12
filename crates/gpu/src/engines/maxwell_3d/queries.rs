//! Report semaphores, counter resets and conditional rendering.

use super::regs::{
    ClearReport, Regs, RenderEnableMode, RenderEnableOverride, ReportSemaphoreCompare,
    ReportSemaphoreOperation, ReportSemaphoreReport,
};
use crate::backend::{Query, QueryKind, ReleaseSize};
use crate::gmmu::Gmmu;

/// The write REPORT_SEMAPHORE.QUERY asks for. Acquires are waited on by the front thread and
/// traps are not emulated, so both give `None`.
pub(super) fn report(regs: &Regs) -> Option<Query> {
    let semaphore = &regs.report_semaphore;
    let query = semaphore.query;
    let operation = query.operation();
    if operation != ReportSemaphoreOperation::RELEASE
        && operation != ReportSemaphoreOperation::REPORT_ONLY
    {
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

/// Evaluates RENDER_ENABLE when its mode is written; conditional modes compare the two
/// sequence/mode pairs at its address as they are now.
pub(super) fn render_enable(regs: &Regs, memory: &Gmmu) -> bool {
    let compare = || {
        memory
            .read::<ReportSemaphoreCompare>(regs.render_enable.address())
            .unwrap_or_else(|fault| {
                tracing::warn!(?fault, "render condition at unmapped GPU memory");
                bytemuck::Zeroable::zeroed()
            })
    };
    match regs.render_enable.mode {
        RenderEnableMode::FALSE => false,
        RenderEnableMode::CONDITIONAL => {
            let c = compare();
            c.initial_sequence != 0 && c.initial_mode != 0
        }
        RenderEnableMode::IF_EQUAL => {
            let c = compare();
            c.initial_sequence == c.current_sequence && c.initial_mode == c.current_mode
        }
        RenderEnableMode::IF_NOT_EQUAL => {
            let c = compare();
            c.initial_sequence != c.current_sequence || c.initial_mode != c.current_mode
        }
        _ => true,
    }
}

/// Whether draws and clears run: the override register wins over the evaluated condition.
#[inline]
pub(super) fn renders(regs: &Regs, render_enable: bool) -> bool {
    match regs.render_enable_override {
        RenderEnableOverride::ALWAYS_RENDER => true,
        RenderEnableOverride::NEVER_RENDER => false,
        _ => render_enable,
    }
}
