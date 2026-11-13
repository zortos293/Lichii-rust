//! Vector construction, extraction and insertion, from the C++ `emit_spirv_composite.cpp`.

use rspirv::spirv::Word;
use smallvec::SmallVec;

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::InstId;

fn construct(ctx: &mut EmitContext<'_>, inst: InstId, ty: Word) -> EmitResult<Word> {
    let mut constituents = SmallVec::<[Word; 4]>::new();
    for index in 0..ctx.program.inst(inst).num_args() {
        constituents.push(ctx.def_arg(inst, index)?);
    }
    Ok(ctx.b.composite_construct(ty, None, constituents)?)
}

fn extract(ctx: &mut EmitContext<'_>, inst: InstId, ty: Word) -> EmitResult<Word> {
    let composite = ctx.def_arg(inst, 0)?;
    let index = ctx.imm_u32(inst, 1)?;
    Ok(ctx.b.composite_extract(ty, None, composite, [index])?)
}

fn insert(ctx: &mut EmitContext<'_>, inst: InstId, ty: Word) -> EmitResult<Word> {
    let composite = ctx.def_arg(inst, 0)?;
    let object = ctx.def_arg(inst, 1)?;
    let index = ctx.imm_u32(inst, 2)?;
    Ok(ctx
        .b
        .composite_insert(ty, None, object, composite, [index])?)
}

pub(super) fn composite_construct_u32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.u32[2];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_u32x3(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.u32[3];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_u32x4(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.u32[4];
    construct(ctx, inst, ty)
}

pub(super) fn composite_extract_u32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u32[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_u32x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u32[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_u32x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u32[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_insert_u32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u32[2];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_u32x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u32[3];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_u32x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.u32[4];
    insert(ctx, inst, ty)
}

pub(super) fn composite_construct_f16x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f16[2];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_f16x3(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f16[3];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_f16x4(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f16[4];
    construct(ctx, inst, ty)
}

pub(super) fn composite_extract_f16x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f16[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_f16x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f16[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_f16x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f16[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_insert_f16x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f16[2];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_f16x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f16[3];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_f16x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f16[4];
    insert(ctx, inst, ty)
}

pub(super) fn composite_construct_f32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f32[2];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_f32x3(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f32[3];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_f32x4(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f32[4];
    construct(ctx, inst, ty)
}

pub(super) fn composite_extract_f32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f32[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_f32x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f32[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_f32x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f32[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_insert_f32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f32[2];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_f32x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f32[3];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_f32x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f32[4];
    insert(ctx, inst, ty)
}

pub(super) fn composite_construct_f64x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f64[2];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_f64x3(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f64[3];
    construct(ctx, inst, ty)
}

pub(super) fn composite_construct_f64x4(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.f64[4];
    construct(ctx, inst, ty)
}

pub(super) fn composite_extract_f64x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f64[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_f64x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f64[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_extract_f64x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f64[1];
    extract(ctx, inst, ty)
}

pub(super) fn composite_insert_f64x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f64[2];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_f64x3(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f64[3];
    insert(ctx, inst, ty)
}

pub(super) fn composite_insert_f64x4(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.f64[4];
    insert(ctx, inst, ty)
}
