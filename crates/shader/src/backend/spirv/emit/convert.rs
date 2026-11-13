//! Numeric conversions, from the C++ `emit_spirv_convert.cpp`.
//!
//! 8- and 16-bit integers are native when the program declared the type (the host supports
//! the width and the shader info records its use) and otherwise live sign- or zero-extended
//! in a u32, extracted with BitFieldSExtract or BitFieldUExtract. Deciding on the declared
//! type rather than the profile keeps the two representations from mixing in one program.

use rspirv::spirv::Word;

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::{InstId, Type};

/// The low `bits` of a u32, sign- or zero-extended.
fn extract_low(
    ctx: &mut EmitContext<'_>,
    value: Word,
    bits: u32,
    signed: bool,
) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    let count = ctx.const_u32(bits);
    let offset = ctx.u32_zero_value;
    Ok(if signed {
        ctx.b
            .bit_field_s_extract(u32_t, None, value, offset, count)?
    } else {
        ctx.b
            .bit_field_u_extract(u32_t, None, value, offset, count)?
    })
}

/// A u32 narrowed to the 8- or 16-bit operand of an integer-to-float conversion.
fn narrow(ctx: &mut EmitContext<'_>, value: Word, bits: u32, signed: bool) -> EmitResult<Word> {
    let ty = match (bits, signed) {
        (8, false) => ctx.u8,
        (8, true) => ctx.s8,
        (_, false) => ctx.u16,
        (_, true) => ctx.s16,
    };
    if ty == 0 {
        return extract_low(ctx, value, bits, signed);
    }
    Ok(if signed {
        ctx.b.s_convert(ty, None, value)?
    } else {
        ctx.b.u_convert(ty, None, value)?
    })
}

/// A float converted to a 16-bit integer held sign- or zero-extended in a u32.
fn float_to_16(ctx: &mut EmitContext<'_>, inst: InstId, signed: bool) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    let (u16_t, u32_t) = (ctx.u16, ctx.u32[1]);
    if u16_t == 0 {
        let wide = if signed {
            ctx.b.convert_f_to_s(u32_t, None, value)?
        } else {
            ctx.b.convert_f_to_u(u32_t, None, value)?
        };
        return extract_low(ctx, wide, 16, signed);
    }
    Ok(if signed {
        let narrow = ctx.b.convert_f_to_s(u16_t, None, value)?;
        ctx.b.s_convert(u32_t, None, narrow)?
    } else {
        let narrow = ctx.b.convert_f_to_u(u16_t, None, value)?;
        ctx.b.u_convert(u32_t, None, narrow)?
    })
}

fn float_to_int(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    ty: Type,
    signed: bool,
) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let value = ctx.def_arg(inst, 0)?;
    Ok(if signed {
        ctx.b.convert_f_to_s(ty, None, value)?
    } else {
        ctx.b.convert_f_to_u(ty, None, value)?
    })
}

fn float_to_float(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.f_convert(ty, None, value)?)
}

/// An integer converted to a float. `narrow_bits` narrows an 8- or 16-bit source held in a u32
/// first; `broken_signed` bitcasts a signed 32-bit source to s32 on drivers with broken
/// signed operations.
fn int_to_float(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    ty: Type,
    signed: bool,
    narrow_bits: Option<u32>,
) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let mut value = ctx.def_arg(inst, 0)?;
    if let Some(bits) = narrow_bits {
        value = narrow(ctx, value, bits, signed)?;
    }
    Ok(if signed {
        ctx.b.convert_s_to_f(ty, None, value)?
    } else {
        ctx.b.convert_u_to_f(ty, None, value)?
    })
}

fn signed_32_to_float(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let mut value = ctx.def_arg(inst, 0)?;
    if ctx.profile.has_broken_signed_operations {
        value = ctx.b.bitcast(ctx.s32[1], None, value)?;
    }
    Ok(ctx.b.convert_s_to_f(ty, None, value)?)
}

/// A native 8- or 16-bit integer widened to u32, or one held in a u32 re-extended.
fn widen(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    native: Word,
    bits: u32,
    signed: bool,
) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    if native == 0 {
        return extract_low(ctx, value, bits, signed);
    }
    let u32_t = ctx.u32[1];
    Ok(if signed {
        ctx.b.s_convert(u32_t, None, value)?
    } else {
        ctx.b.u_convert(u32_t, None, value)?
    })
}

/// A u32 truncated to a native 8- or 16-bit integer, or zero-extended in a u32.
fn truncate(ctx: &mut EmitContext<'_>, inst: InstId, native: Word, bits: u32) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    if native == 0 {
        return extract_low(ctx, value, bits, false);
    }
    Ok(ctx.b.u_convert(native, None, value)?)
}

pub(super) fn convert_s16_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_16(ctx, inst, true)
}

pub(super) fn convert_s16_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_16(ctx, inst, true)
}

pub(super) fn convert_s16_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_16(ctx, inst, true)
}

pub(super) fn convert_s32_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U32, true)
}

pub(super) fn convert_s32_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    if !ctx.profile.has_broken_signed_operations {
        return float_to_int(ctx, inst, Type::U32, true);
    }
    let value = ctx.def_arg(inst, 0)?;
    let signed = ctx.b.convert_f_to_s(ctx.s32[1], None, value)?;
    Ok(ctx.b.bitcast(ctx.u32[1], None, signed)?)
}

pub(super) fn convert_s32_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U32, true)
}

pub(super) fn convert_s64_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U64, true)
}

pub(super) fn convert_s64_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U64, true)
}

pub(super) fn convert_s64_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U64, true)
}

pub(super) fn convert_u16_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_16(ctx, inst, false)
}

pub(super) fn convert_u16_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_16(ctx, inst, false)
}

pub(super) fn convert_u16_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_16(ctx, inst, false)
}

pub(super) fn convert_u32_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U32, false)
}

pub(super) fn convert_u32_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U32, false)
}

pub(super) fn convert_u32_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U32, false)
}

pub(super) fn convert_u64_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U64, false)
}

pub(super) fn convert_u64_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U64, false)
}

pub(super) fn convert_u64_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_int(ctx, inst, Type::U64, false)
}

pub(super) fn convert_u64_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::U64)?;
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.u_convert(ty, None, value)?)
}

pub(super) fn convert_u32_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.u_convert(ctx.u32[1], None, value)?)
}

/// F2F to half. A NaN result (from a NaN source) becomes zero, as the C++ backend does on
/// desktop drivers.
pub(super) fn convert_f16_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::F16)?;
    let value = ctx.def_arg(inst, 0)?;
    let result = ctx.b.f_convert(ty, None, value)?;
    let is_nan = ctx.b.is_nan(ctx.u1, None, result)?;
    let zero = ctx.const_f16(0);
    Ok(ctx.b.select(ty, None, is_nan, zero, result)?)
}

pub(super) fn convert_f32_f16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_float(ctx, inst, Type::F32)
}

pub(super) fn convert_f32_f64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_float(ctx, inst, Type::F32)
}

pub(super) fn convert_f64_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    float_to_float(ctx, inst, Type::F64)
}

pub(super) fn convert_f16_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, true, Some(8))
}

pub(super) fn convert_f16_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, true, Some(16))
}

pub(super) fn convert_f16_s32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, true, None)
}

pub(super) fn convert_f16_s64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, true, None)
}

pub(super) fn convert_f16_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, false, Some(8))
}

pub(super) fn convert_f16_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, false, Some(16))
}

pub(super) fn convert_f16_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, false, None)
}

pub(super) fn convert_f16_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F16, false, None)
}

pub(super) fn convert_f32_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F32, true, Some(8))
}

pub(super) fn convert_f32_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F32, true, Some(16))
}

pub(super) fn convert_f32_s32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    signed_32_to_float(ctx, inst, Type::F32)
}

pub(super) fn convert_f32_s64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F32, true, None)
}

pub(super) fn convert_f32_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F32, false, Some(8))
}

pub(super) fn convert_f32_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F32, false, Some(16))
}

pub(super) fn convert_f32_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F32, false, None)
}

pub(super) fn convert_f32_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F32, false, None)
}

pub(super) fn convert_f64_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F64, true, Some(8))
}

pub(super) fn convert_f64_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F64, true, Some(16))
}

pub(super) fn convert_f64_s32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    signed_32_to_float(ctx, inst, Type::F64)
}

pub(super) fn convert_f64_s64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F64, true, None)
}

pub(super) fn convert_f64_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F64, false, Some(8))
}

pub(super) fn convert_f64_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F64, false, Some(16))
}

pub(super) fn convert_f64_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F64, false, None)
}

pub(super) fn convert_f64_u64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    int_to_float(ctx, inst, Type::F64, false, None)
}

pub(super) fn convert_u16_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let native = ctx.u16;
    truncate(ctx, inst, native, 16)
}

pub(super) fn convert_u32_u16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let native = ctx.u16;
    widen(ctx, inst, native, 16, false)
}

pub(super) fn convert_u8_u32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let native = ctx.u8;
    truncate(ctx, inst, native, 8)
}

pub(super) fn convert_u32_u8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let native = ctx.u8;
    widen(ctx, inst, native, 8, false)
}

pub(super) fn convert_s32_s8(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let native = ctx.u8;
    widen(ctx, inst, native, 8, true)
}

pub(super) fn convert_s32_s16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let native = ctx.u16;
    widen(ctx, inst, native, 16, true)
}
