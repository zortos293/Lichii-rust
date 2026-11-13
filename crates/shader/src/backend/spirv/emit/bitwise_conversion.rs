//! Bit casts and packing, from the C++ `emit_spirv_bitwise_conversion.cpp`.

use rspirv::spirv::{GlslStd450Op, Word};

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::{InstId, Type};

fn bitcast(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.bitcast(ty, None, value)?)
}

/// Half bits to u16, or zero-extended in a u32 (the low half of an f16x2) without int16.
pub(super) fn bit_cast_u16_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if ctx.u16 != 0 {
        return bitcast(ctx, inst, Type::U16);
    }
    let value = ctx.def_arg(inst, 0)?;
    let f16x2 = ctx.declared_type(Type::F16X2)?;
    let zero = ctx.const_f16(0);
    let pair = ctx.b.composite_construct(f16x2, None, [value, zero])?;
    Ok(ctx.b.bitcast(ctx.u32[1], None, pair)?)
}

pub(super) fn bit_cast_u32_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::U32)
}

pub(super) fn bit_cast_u64_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::U64)
}

/// A u16 as half bits, or the low half of a u32 without int16.
pub(super) fn bit_cast_f16_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if ctx.u16 != 0 {
        return bitcast(ctx, inst, Type::F16);
    }
    let pair = bitcast(ctx, inst, Type::F16X2)?;
    let f16_t = ctx.declared_type(Type::F16)?;
    Ok(ctx.b.composite_extract(f16_t, None, pair, [0])?)
}

pub(super) fn bit_cast_f32_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::F32)
}

pub(super) fn bit_cast_f64_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::F64)
}

pub(super) fn pack_uint2x32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::U64)
}

pub(super) fn unpack_uint2x32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::U32X2)
}

pub(super) fn pack_float2x16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::U32)
}

pub(super) fn unpack_float2x16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::F16X2)
}

pub(super) fn pack_half2x16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    ctx.glsl_inst(ctx.u32[1], GlslStd450Op::PackHalf2x16, &[value])
}

pub(super) fn unpack_half2x16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    ctx.glsl_inst(ctx.f32[2], GlslStd450Op::UnpackHalf2x16, &[value])
}

pub(super) fn pack_double2x32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::F64)
}

pub(super) fn unpack_double2x32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    bitcast(ctx, inst, Type::U32X2)
}
