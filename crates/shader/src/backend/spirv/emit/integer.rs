//! Integer arithmetic, bit operations and comparisons, from the C++ `emit_spirv_integer.cpp`.
//!
//! The 64-bit operations need a host with `shaderInt64`; without it `lower_int64_to_int32`
//! rewrites them on 32-bit pairs before emission.

use rspirv::dr::{self, Builder};
use rspirv::spirv::{GlslStd450Op, Word};

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::{InstId, Opcode, Type};

type CompareOp = fn(&mut Builder, Word, Option<Word>, Word, Word) -> Result<Word, dr::Error>;

fn operands(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<(Word, Word)> {
    Ok((ctx.def_arg(inst, 0)?, ctx.def_arg(inst, 1)?))
}

fn three_operands(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<[Word; 3]> {
    Ok([
        ctx.def_arg(inst, 0)?,
        ctx.def_arg(inst, 1)?,
        ctx.def_arg(inst, 2)?,
    ])
}

/// Defines the zero and sign pseudo-instructions a 32-bit result carries.
fn set_zero_and_sign(ctx: &mut EmitContext<'_>, inst: InstId, result: Word) -> EmitResult<Word> {
    if let Some(zero) = ctx.program.pseudo_op(inst, Opcode::GetZeroFromOp) {
        let id = ctx.b.i_equal(ctx.u1, None, result, ctx.u32_zero_value)?;
        ctx.set_def(zero, id);
    }
    if let Some(sign) = ctx.program.pseudo_op(inst, Opcode::GetSignFromOp) {
        let id = ctx
            .b
            .s_less_than(ctx.u1, None, result, ctx.u32_zero_value)?;
        ctx.set_def(sign, id);
    }
    Ok(result)
}

fn unary_glsl(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    ty: Type,
    op: GlslStd450Op,
) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let value = ctx.def_arg(inst, 0)?;
    ctx.glsl_inst(ty, op, &[value])
}

fn binary_glsl(ctx: &mut EmitContext<'_>, inst: InstId, op: GlslStd450Op) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    ctx.glsl_inst(ctx.u32[1], op, &[a, b])
}

fn compare(ctx: &mut EmitContext<'_>, inst: InstId, op: CompareOp) -> EmitResult<Word> {
    let (lhs, rhs) = operands(ctx, inst)?;
    Ok(op(&mut ctx.b, ctx.u1, None, lhs, rhs)?)
}

/// IAdd32 with its carry, overflow, zero and sign pseudo-instructions. Overflow is
/// `((a ^ r) & (b ^ r)) < 0`: the result's sign differs from the sign of both operands.
pub(super) fn i_add32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    let u32_t = ctx.u32[1];
    let result = match ctx.program.pseudo_op(inst, Opcode::GetCarryFromOp) {
        Some(carry) => {
            let carry_type = ctx.iadd_carry_type();
            let pair = ctx.b.i_add_carry(carry_type, None, a, b)?;
            let result = ctx.b.composite_extract(u32_t, None, pair, [0])?;
            let carry_value = ctx.b.composite_extract(u32_t, None, pair, [1])?;
            let flag = ctx
                .b
                .i_not_equal(ctx.u1, None, carry_value, ctx.u32_zero_value)?;
            ctx.set_def(carry, flag);
            result
        }
        None => ctx.b.i_add(u32_t, None, a, b)?,
    };
    set_zero_and_sign(ctx, inst, result)?;
    if let Some(overflow) = ctx.program.pseudo_op(inst, Opcode::GetOverflowFromOp) {
        let a_changed = ctx.b.bitwise_xor(u32_t, None, a, result)?;
        let b_changed = ctx.b.bitwise_xor(u32_t, None, b, result)?;
        let both = ctx.b.bitwise_and(u32_t, None, a_changed, b_changed)?;
        let flag = ctx.b.s_less_than(ctx.u1, None, both, ctx.u32_zero_value)?;
        ctx.set_def(overflow, flag);
    }
    Ok(result)
}

pub(super) fn i_add64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::U64)?;
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.i_add(ty, None, a, b)?)
}

pub(super) fn i_sub32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.i_sub(ctx.u32[1], None, a, b)?)
}

pub(super) fn i_sub64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::U64)?;
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.i_sub(ty, None, a, b)?)
}

pub(super) fn i_mul32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.i_mul(ctx.u32[1], None, a, b)?)
}

pub(super) fn s_div32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.s_div(ctx.u32[1], None, a, b)?)
}

pub(super) fn u_div32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    Ok(ctx.b.u_div(ctx.u32[1], None, a, b)?)
}

pub(super) fn i_neg32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.s_negate(ctx.u32[1], None, value)?)
}

pub(super) fn i_neg64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::U64)?;
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.s_negate(ty, None, value)?)
}

pub(super) fn i_abs32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unary_glsl(ctx, inst, Type::U32, GlslStd450Op::SAbs)
}

pub(super) fn i_abs64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unary_glsl(ctx, inst, Type::U64, GlslStd450Op::SAbs)
}

pub(super) fn shift_left_logical32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (base, shift) = operands(ctx, inst)?;
    Ok(ctx.b.shift_left_logical(ctx.u32[1], None, base, shift)?)
}

pub(super) fn shift_left_logical64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::U64)?;
    let (base, shift) = operands(ctx, inst)?;
    Ok(ctx.b.shift_left_logical(ty, None, base, shift)?)
}

pub(super) fn shift_right_logical32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (base, shift) = operands(ctx, inst)?;
    Ok(ctx.b.shift_right_logical(ctx.u32[1], None, base, shift)?)
}

pub(super) fn shift_right_logical64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::U64)?;
    let (base, shift) = operands(ctx, inst)?;
    Ok(ctx.b.shift_right_logical(ty, None, base, shift)?)
}

pub(super) fn shift_right_arithmetic32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let (base, shift) = operands(ctx, inst)?;
    Ok(ctx
        .b
        .shift_right_arithmetic(ctx.u32[1], None, base, shift)?)
}

pub(super) fn shift_right_arithmetic64(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::U64)?;
    let (base, shift) = operands(ctx, inst)?;
    Ok(ctx.b.shift_right_arithmetic(ty, None, base, shift)?)
}

pub(super) fn bitwise_and32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    let result = ctx.b.bitwise_and(ctx.u32[1], None, a, b)?;
    set_zero_and_sign(ctx, inst, result)
}

pub(super) fn bitwise_or32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    let result = ctx.b.bitwise_or(ctx.u32[1], None, a, b)?;
    set_zero_and_sign(ctx, inst, result)
}

pub(super) fn bitwise_xor32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (a, b) = operands(ctx, inst)?;
    let result = ctx.b.bitwise_xor(ctx.u32[1], None, a, b)?;
    set_zero_and_sign(ctx, inst, result)
}

pub(super) fn bit_field_insert(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let [base, insert, offset] = three_operands(ctx, inst)?;
    let count = ctx.def_arg(inst, 3)?;
    Ok(ctx
        .b
        .bit_field_insert(ctx.u32[1], None, base, insert, offset, count)?)
}

pub(super) fn bit_field_s_extract(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let [base, offset, count] = three_operands(ctx, inst)?;
    let result = ctx
        .b
        .bit_field_s_extract(ctx.u32[1], None, base, offset, count)?;
    set_zero_and_sign(ctx, inst, result)
}

pub(super) fn bit_field_u_extract(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let [base, offset, count] = three_operands(ctx, inst)?;
    let result = ctx
        .b
        .bit_field_u_extract(ctx.u32[1], None, base, offset, count)?;
    set_zero_and_sign(ctx, inst, result)
}

pub(super) fn bit_reverse32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.bit_reverse(ctx.u32[1], None, value)?)
}

pub(super) fn bit_count32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.bit_count(ctx.u32[1], None, value)?)
}

pub(super) fn bitwise_not32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.not(ctx.u32[1], None, value)?)
}

pub(super) fn find_s_msb32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unary_glsl(ctx, inst, Type::U32, GlslStd450Op::FindSMsb)
}

pub(super) fn find_u_msb32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unary_glsl(ctx, inst, Type::U32, GlslStd450Op::FindUMsb)
}

/// A signed min or max; drivers with broken signed operations get s32 operands and result.
fn signed_min_max(ctx: &mut EmitContext<'_>, inst: InstId, op: GlslStd450Op) -> EmitResult<Word> {
    if !ctx.profile.has_broken_signed_operations {
        return binary_glsl(ctx, inst, op);
    }
    let (a, b) = operands(ctx, inst)?;
    let s32_t = ctx.s32[1];
    let a = ctx.b.bitcast(s32_t, None, a)?;
    let b = ctx.b.bitcast(s32_t, None, b)?;
    let result = ctx.glsl_inst(s32_t, op, &[a, b])?;
    Ok(ctx.b.bitcast(ctx.u32[1], None, result)?)
}

pub(super) fn s_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    signed_min_max(ctx, inst, GlslStd450Op::SMin)
}

pub(super) fn u_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    binary_glsl(ctx, inst, GlslStd450Op::UMin)
}

pub(super) fn s_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    signed_min_max(ctx, inst, GlslStd450Op::SMax)
}

pub(super) fn u_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    binary_glsl(ctx, inst, GlslStd450Op::UMax)
}

/// SClamp32 in s32 for drivers with broken signed operations, and as SMin then SMax for
/// drivers with a broken clamp.
pub(super) fn s_clamp32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let [value, min, max] = three_operands(ctx, inst)?;
    let u32_t = ctx.u32[1];
    let profile = ctx.profile;
    let result = if profile.has_broken_signed_operations || profile.has_broken_spirv_clamp {
        let s32_t = ctx.s32[1];
        let value = ctx.b.bitcast(s32_t, None, value)?;
        let min = ctx.b.bitcast(s32_t, None, min)?;
        let max = ctx.b.bitcast(s32_t, None, max)?;
        let clamped = if profile.has_broken_spirv_clamp {
            let upper = ctx.glsl_inst(s32_t, GlslStd450Op::SMin, &[value, max])?;
            ctx.glsl_inst(s32_t, GlslStd450Op::SMax, &[upper, min])?
        } else {
            ctx.glsl_inst(s32_t, GlslStd450Op::SClamp, &[value, min, max])?
        };
        ctx.b.bitcast(u32_t, None, clamped)?
    } else {
        ctx.glsl_inst(u32_t, GlslStd450Op::SClamp, &[value, min, max])?
    };
    set_zero_and_sign(ctx, inst, result)
}

pub(super) fn u_clamp32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let [value, min, max] = three_operands(ctx, inst)?;
    let u32_t = ctx.u32[1];
    let result = if ctx.profile.has_broken_spirv_clamp {
        let upper = ctx.glsl_inst(u32_t, GlslStd450Op::UMin, &[value, max])?;
        ctx.glsl_inst(u32_t, GlslStd450Op::UMax, &[upper, min])?
    } else {
        ctx.glsl_inst(u32_t, GlslStd450Op::UClamp, &[value, min, max])?
    };
    set_zero_and_sign(ctx, inst, result)
}

pub(super) fn s_less_than(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::s_less_than)
}

pub(super) fn u_less_than(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::u_less_than)
}

pub(super) fn i_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::i_equal)
}

pub(super) fn s_less_than_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::s_less_than_equal)
}

pub(super) fn u_less_than_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::u_less_than_equal)
}

pub(super) fn s_greater_than(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::s_greater_than)
}

pub(super) fn u_greater_than(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::u_greater_than)
}

pub(super) fn i_not_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::i_not_equal)
}

pub(super) fn s_greater_than_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::s_greater_than_equal)
}

pub(super) fn u_greater_than_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::u_greater_than_equal)
}
