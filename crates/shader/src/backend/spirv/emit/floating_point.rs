//! Floating-point arithmetic and comparisons, from the C++ `emit_spirv_floating_point.cpp`.
//!
//! Of an instruction's `FpControl`, only `no_contraction` reaches the instruction (as
//! NoContraction). The denormal mode is a module-wide float-controls execution mode the driver
//! sets from the shader info, and Vulkan SPIR-V has no per-instruction rounding mode for
//! arithmetic, so the rounding field is dropped as in the C++ backend.

use rspirv::dr::{self, Builder};
use rspirv::spirv::{Decoration, GlslStd450Op, Word};

use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::{FpControl, FpRounding, InstId, Type};

type CompareOp = fn(&mut Builder, Word, Option<Word>, Word, Word) -> Result<Word, dr::Error>;

/// Bits of 1.0 in half precision.
const F16_ONE: u16 = 0x3c00;

fn decorate(ctx: &mut EmitContext<'_>, inst: InstId, id: Word) -> Word {
    let control = FpControl::from_flags(ctx.flags(inst));
    if matches!(
        control.rounding,
        FpRounding::Rm | FpRounding::Rp | FpRounding::Rz
    ) {
        crate::fallback::note("FP", "rounding_mode");
    }
    if control.no_contraction {
        ctx.decorate(id, Decoration::NoContraction, &[]);
    }
    id
}

fn operands(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<(Word, Word)> {
    Ok((ctx.def_arg(inst, 0)?, ctx.def_arg(inst, 1)?))
}

fn glsl_unary(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    ty: Type,
    op: GlslStd450Op,
) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let value = ctx.def_arg(inst, 0)?;
    ctx.glsl_inst(ty, op, &[value])
}

fn glsl_binary(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    ty: Type,
    op: GlslStd450Op,
) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let (a, b) = operands(ctx, inst)?;
    ctx.glsl_inst(ty, op, &[a, b])
}

fn add(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let (a, b) = operands(ctx, inst)?;
    let id = ctx.b.f_add(ty, None, a, b)?;
    Ok(decorate(ctx, inst, id))
}

fn mul(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let (a, b) = operands(ctx, inst)?;
    let id = ctx.b.f_mul(ty, None, a, b)?;
    Ok(decorate(ctx, inst, id))
}

fn fma(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let (a, b) = operands(ctx, inst)?;
    let c = ctx.def_arg(inst, 2)?;
    let id = ctx.glsl_inst(ty, GlslStd450Op::Fma, &[a, b, c])?;
    Ok(decorate(ctx, inst, id))
}

fn neg(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.f_negate(ty, None, value)?)
}

/// Maxwell's FMNMX and .SAT return the non-NaN operand (NaN saturates to the lower bound).
/// FMin, FMax and FClamp leave NaN undefined and some drivers pass it through, so the clamp is
/// built from the NaN-defined NMax and NMin.
fn clamp(
    ctx: &mut EmitContext<'_>,
    ty: Word,
    value: Word,
    min: Word,
    max: Word,
) -> EmitResult<Word> {
    let lower = ctx.glsl_inst(ty, GlslStd450Op::NMax, &[value, min])?;
    ctx.glsl_inst(ty, GlslStd450Op::NMin, &[lower, max])
}

fn clamp_op(ctx: &mut EmitContext<'_>, inst: InstId, ty: Type) -> EmitResult<Word> {
    let ty = ctx.declared_type(ty)?;
    let value = ctx.def_arg(inst, 0)?;
    let (min, max) = (ctx.def_arg(inst, 1)?, ctx.def_arg(inst, 2)?);
    clamp(ctx, ty, value, min, max)
}

fn compare(ctx: &mut EmitContext<'_>, inst: InstId, op: CompareOp) -> EmitResult<Word> {
    let (lhs, rhs) = operands(ctx, inst)?;
    Ok(op(&mut ctx.b, ctx.u1, None, lhs, rhs)?)
}

/// An ordered not-equal. Drivers that ignore NaN in comparisons get the ordering checked
/// explicitly.
fn ord_not_equal(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (lhs, rhs) = operands(ctx, inst)?;
    let u1 = ctx.u1;
    let comp = ctx.b.f_ord_not_equal(u1, None, lhs, rhs)?;
    if !ctx.profile.ignore_nan_fp_comparisons {
        return Ok(comp);
    }
    let lhs_nan = ctx.b.is_nan(u1, None, lhs)?;
    let rhs_nan = ctx.b.is_nan(u1, None, rhs)?;
    let lhs_ordered = ctx.b.logical_not(u1, None, lhs_nan)?;
    let rhs_ordered = ctx.b.logical_not(u1, None, rhs_nan)?;
    let lhs_check = ctx.b.logical_and(u1, None, comp, lhs_ordered)?;
    Ok(ctx.b.logical_and(u1, None, lhs_check, rhs_ordered)?)
}

/// An unordered comparison. Drivers that ignore NaN in comparisons get the NaN operands
/// checked explicitly.
fn unord_compare(ctx: &mut EmitContext<'_>, inst: InstId, op: CompareOp) -> EmitResult<Word> {
    let (lhs, rhs) = operands(ctx, inst)?;
    let u1 = ctx.u1;
    if !ctx.profile.ignore_nan_fp_comparisons {
        return Ok(op(&mut ctx.b, u1, None, lhs, rhs)?);
    }
    let lhs_nan = ctx.b.is_nan(u1, None, lhs)?;
    let rhs_nan = ctx.b.is_nan(u1, None, rhs)?;
    let comp = op(&mut ctx.b, u1, None, lhs, rhs)?;
    let lhs_check = ctx.b.logical_or(u1, None, comp, lhs_nan)?;
    Ok(ctx.b.logical_or(u1, None, lhs_check, rhs_nan)?)
}

fn is_nan(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    Ok(ctx.b.is_nan(ctx.u1, None, value)?)
}

pub(super) fn fp_abs16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F16, GlslStd450Op::FAbs)
}

pub(super) fn fp_abs32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::FAbs)
}

pub(super) fn fp_abs64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F64, GlslStd450Op::FAbs)
}

pub(super) fn fp_add16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    add(ctx, inst, Type::F16)
}

pub(super) fn fp_add32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    add(ctx, inst, Type::F32)
}

pub(super) fn fp_add64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    add(ctx, inst, Type::F64)
}

pub(super) fn fp_fma16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    fma(ctx, inst, Type::F16)
}

pub(super) fn fp_fma32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    fma(ctx, inst, Type::F32)
}

pub(super) fn fp_fma64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    fma(ctx, inst, Type::F64)
}

pub(super) fn fp_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_binary(ctx, inst, Type::F32, GlslStd450Op::NMax)
}

pub(super) fn fp_max64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_binary(ctx, inst, Type::F64, GlslStd450Op::NMax)
}

pub(super) fn fp_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_binary(ctx, inst, Type::F32, GlslStd450Op::NMin)
}

pub(super) fn fp_min64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_binary(ctx, inst, Type::F64, GlslStd450Op::NMin)
}

pub(super) fn fp_mul16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    mul(ctx, inst, Type::F16)
}

pub(super) fn fp_mul32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    mul(ctx, inst, Type::F32)
}

pub(super) fn fp_mul64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    mul(ctx, inst, Type::F64)
}

pub(super) fn fp_neg16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    neg(ctx, inst, Type::F16)
}

pub(super) fn fp_neg32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    neg(ctx, inst, Type::F32)
}

pub(super) fn fp_neg64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    neg(ctx, inst, Type::F64)
}

pub(super) fn fp_recip32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    let one = ctx.const_f32(1.0);
    Ok(ctx.b.f_div(ctx.f32[1], None, one, value)?)
}

pub(super) fn fp_recip64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::F64)?;
    let value = ctx.def_arg(inst, 0)?;
    let one = ctx.const_f64(1.0);
    Ok(ctx.b.f_div(ty, None, one, value)?)
}

pub(super) fn fp_recip_sqrt32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::InverseSqrt)
}

pub(super) fn fp_recip_sqrt64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F64, GlslStd450Op::InverseSqrt)
}

pub(super) fn fp_sqrt(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Sqrt)
}

pub(super) fn fp_sin(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Sin)
}

pub(super) fn fp_exp2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Exp2)
}

/// MUFU.TANH. A series near zero avoids the cancellation of (1 - e^-2x) / (1 + e^-2x), and
/// tiny inputs (subnormals included) return themselves with their sign.
pub(super) fn fp_tanh(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    let (ty, u32_t, u1) = (ctx.f32[1], ctx.u32[1], ctx.u1);
    let abs_mask = ctx.const_u32(0x7fff_ffff);
    let sign_mask = ctx.const_u32(0x8000_0000);
    let small_limit = ctx.const_u32(0x3d80_0000);
    let tiny_limit = ctx.const_u32(0x3980_0000);
    let c5 = ctx.const_f32(2.0 / 15.0);
    let c7 = ctx.const_f32(-17.0 / 315.0);
    let c3 = ctx.const_f32(-1.0 / 3.0);
    let minus_two_log2e = ctx.const_f32(-2.0 * std::f32::consts::LOG2_E);
    let one = ctx.const_f32(1.0);
    let bits = ctx.b.bitcast(u32_t, None, value)?;
    let absolute_bits = ctx.b.bitwise_and(u32_t, None, bits, abs_mask)?;
    let absolute = ctx.b.bitcast(ty, None, absolute_bits)?;
    let square = ctx.b.f_mul(ty, None, absolute, absolute)?;
    let fifth_term = ctx.b.f_mul(ty, None, square, c7)?;
    let fifth = ctx.b.f_add(ty, None, c5, fifth_term)?;
    let third_term = ctx.b.f_mul(ty, None, square, fifth)?;
    let third = ctx.b.f_add(ty, None, c3, third_term)?;
    let cube = ctx.b.f_mul(ty, None, absolute, square)?;
    let series = ctx.b.f_mul(ty, None, cube, third)?;
    let polynomial = ctx.b.f_add(ty, None, absolute, series)?;
    let exponent = ctx.b.f_mul(ty, None, absolute, minus_two_log2e)?;
    let exponential = ctx.glsl_inst(ty, GlslStd450Op::Exp2, &[exponent])?;
    let numerator = ctx.b.f_sub(ty, None, one, exponential)?;
    let denominator = ctx.b.f_add(ty, None, one, exponential)?;
    let magnitude = ctx.b.f_div(ty, None, numerator, denominator)?;
    let small = ctx.b.u_less_than(u1, None, absolute_bits, small_limit)?;
    let selected = ctx.b.select(ty, None, small, polynomial, magnitude)?;
    let selected_bits = ctx.b.bitcast(u32_t, None, selected)?;
    let tiny = ctx.b.u_less_than(u1, None, absolute_bits, tiny_limit)?;
    let preserved = ctx
        .b
        .select(u32_t, None, tiny, absolute_bits, selected_bits)?;
    let sign = ctx.b.bitwise_and(u32_t, None, bits, sign_mask)?;
    let signed = ctx.b.bitwise_or(u32_t, None, preserved, sign)?;
    Ok(ctx.b.bitcast(ty, None, signed)?)
}

pub(super) fn fp_cos(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Cos)
}

pub(super) fn fp_log2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Log2)
}

pub(super) fn fp_saturate16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::F16)?;
    let value = ctx.def_arg(inst, 0)?;
    let zero = ctx.const_f16(0);
    let one = ctx.const_f16(F16_ONE);
    clamp(ctx, ty, value, zero, one)
}

pub(super) fn fp_saturate32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 0)?;
    let zero = ctx.const_f32(0.0);
    let one = ctx.const_f32(1.0);
    clamp(ctx, ctx.f32[1], value, zero, one)
}

pub(super) fn fp_saturate64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let ty = ctx.declared_type(Type::F64)?;
    let value = ctx.def_arg(inst, 0)?;
    let zero = ctx.const_f64(0.0);
    let one = ctx.const_f64(1.0);
    clamp(ctx, ty, value, zero, one)
}

pub(super) fn fp_clamp16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    clamp_op(ctx, inst, Type::F16)
}

pub(super) fn fp_clamp32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    clamp_op(ctx, inst, Type::F32)
}

pub(super) fn fp_clamp64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    clamp_op(ctx, inst, Type::F64)
}

pub(super) fn fp_round_even16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F16, GlslStd450Op::RoundEven)
}

pub(super) fn fp_round_even32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::RoundEven)
}

pub(super) fn fp_round_even64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F64, GlslStd450Op::RoundEven)
}

pub(super) fn fp_floor16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F16, GlslStd450Op::Floor)
}

pub(super) fn fp_floor32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Floor)
}

pub(super) fn fp_floor64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F64, GlslStd450Op::Floor)
}

pub(super) fn fp_ceil16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F16, GlslStd450Op::Ceil)
}

pub(super) fn fp_ceil32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Ceil)
}

pub(super) fn fp_ceil64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F64, GlslStd450Op::Ceil)
}

pub(super) fn fp_trunc16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F16, GlslStd450Op::Trunc)
}

pub(super) fn fp_trunc32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F32, GlslStd450Op::Trunc)
}

pub(super) fn fp_trunc64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    glsl_unary(ctx, inst, Type::F64, GlslStd450Op::Trunc)
}

pub(super) fn fp_ord_equal16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_equal)
}

pub(super) fn fp_ord_equal32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_equal)
}

pub(super) fn fp_ord_equal64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_equal)
}

pub(super) fn fp_unord_equal16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_equal)
}

pub(super) fn fp_unord_equal32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_equal)
}

pub(super) fn fp_unord_equal64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_equal)
}

pub(super) fn fp_ord_not_equal16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    ord_not_equal(ctx, inst)
}

pub(super) fn fp_ord_not_equal32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    ord_not_equal(ctx, inst)
}

pub(super) fn fp_ord_not_equal64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    ord_not_equal(ctx, inst)
}

pub(super) fn fp_unord_not_equal16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_unord_not_equal)
}

pub(super) fn fp_unord_not_equal32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_unord_not_equal)
}

pub(super) fn fp_unord_not_equal64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_unord_not_equal)
}

pub(super) fn fp_ord_less_than16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_less_than)
}

pub(super) fn fp_ord_less_than32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_less_than)
}

pub(super) fn fp_ord_less_than64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_less_than)
}

pub(super) fn fp_unord_less_than16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_less_than)
}

pub(super) fn fp_unord_less_than32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_less_than)
}

pub(super) fn fp_unord_less_than64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_less_than)
}

pub(super) fn fp_ord_greater_than16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_greater_than)
}

pub(super) fn fp_ord_greater_than32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_greater_than)
}

pub(super) fn fp_ord_greater_than64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_greater_than)
}

pub(super) fn fp_unord_greater_than16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_greater_than)
}

pub(super) fn fp_unord_greater_than32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_greater_than)
}

pub(super) fn fp_unord_greater_than64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_greater_than)
}

pub(super) fn fp_ord_less_than_equal16(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_less_than_equal)
}

pub(super) fn fp_ord_less_than_equal32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_less_than_equal)
}

pub(super) fn fp_ord_less_than_equal64(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_less_than_equal)
}

pub(super) fn fp_unord_less_than_equal16(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_less_than_equal)
}

pub(super) fn fp_unord_less_than_equal32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_less_than_equal)
}

pub(super) fn fp_unord_less_than_equal64(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_less_than_equal)
}

pub(super) fn fp_ord_greater_than_equal16(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_greater_than_equal)
}

pub(super) fn fp_ord_greater_than_equal32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_greater_than_equal)
}

pub(super) fn fp_ord_greater_than_equal64(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    compare(ctx, inst, Builder::f_ord_greater_than_equal)
}

pub(super) fn fp_unord_greater_than_equal16(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_greater_than_equal)
}

pub(super) fn fp_unord_greater_than_equal32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_greater_than_equal)
}

pub(super) fn fp_unord_greater_than_equal64(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    unord_compare(ctx, inst, Builder::f_unord_greater_than_equal)
}

pub(super) fn fp_is_nan16(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    is_nan(ctx, inst)
}

pub(super) fn fp_is_nan32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    is_nan(ctx, inst)
}

pub(super) fn fp_is_nan64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    is_nan(ctx, inst)
}
