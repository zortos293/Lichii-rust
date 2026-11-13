//! Selects, from the C++ `emit_spirv_select.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::InstId;

fn select(ctx: &mut EmitContext<'_>, inst: InstId, ty: Word) -> EmitResult<Word> {
    let cond = ctx.def_arg(inst, 0)?;
    let true_value = ctx.def_arg(inst, 1)?;
    let false_value = ctx.def_arg(inst, 2)?;
    Ok(ctx.b.select(ty, None, cond, true_value, false_value)?)
}

pub(super) fn select_u1(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u1;
    select(ctx, inst, ty)
}

pub(super) fn select_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u8;
    select(ctx, inst, ty)
}

pub(super) fn select_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u16;
    select(ctx, inst, ty)
}

pub(super) fn select_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u32[1];
    select(ctx, inst, ty)
}

pub(super) fn select_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u64;
    select(ctx, inst, ty)
}

pub(super) fn select_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f16[1];
    select(ctx, inst, ty)
}

pub(super) fn select_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f32[1];
    select(ctx, inst, ty)
}

pub(super) fn select_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f64[1];
    select(ctx, inst, ty)
}
