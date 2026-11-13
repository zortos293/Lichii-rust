//! Boolean logic, from the C++ `emit_spirv_logical.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::InstId;

fn operands(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<(Word, Word)> {
    Ok((ctx.def_arg(inst, 0)?, ctx.def_arg(inst, 1)?))
}

pub(super) fn logical_or(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.logical_or(ctx.u1, None, a, b)?)
}

pub(super) fn logical_and(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.logical_and(ctx.u1, None, a, b)?)
}

pub(super) fn logical_xor(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.logical_not_equal(ctx.u1, None, a, b)?)
}

pub(super) fn logical_not(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.logical_not(ctx.u1, None, value)?)
}
