//! Control flow instructions inside blocks, from the C++ `emit_spirv_control_flow.cpp`.

use rspirv::dr::{InsertPoint, Instruction};
use rspirv::spirv::{Op, SelectionControl};

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::InstId;

pub(super) fn join(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::InvalidIr("Join survived structurization".into()))
}

/// Demotes the invocation to a helper, or kills it behind an always-true branch on hosts
/// without demote so the block stays open for the instructions after it.
pub(super) fn demote_to_helper_invocation(
    ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<()> {
    if ctx.profile.support_demote_to_helper_invocation {
        let demote = Instruction::new(Op::DemoteToHelperInvocation, None, None, Vec::new());
        ctx.b.insert_into_block(InsertPoint::End, demote)?;
        return Ok(());
    }
    crate::fallback::note("discard", "demote_as_kill");
    let kill_label = ctx.b.id();
    let impossible_label = ctx.b.id();
    ctx.b
        .selection_merge(impossible_label, SelectionControl::NONE)?;
    ctx.b
        .branch_conditional(ctx.true_value, kill_label, impossible_label, [])?;
    ctx.b.begin_block(Some(kill_label))?;
    ctx.b.kill()?;
    ctx.b.begin_block(Some(impossible_label))?;
    Ok(())
}
