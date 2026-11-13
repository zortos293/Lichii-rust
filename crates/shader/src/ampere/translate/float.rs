//! Single, half and double precision arithmetic and the conversions.
//!
//! Modifiers map onto IR `FpControl` the way the Maxwell translators map theirs, and the
//! packed-half, F2I and I2F semantics reuse the Maxwell helpers, so both frontends produce the
//! same IR for the same operation.

use std::f32::consts::TAU;

use super::common::{AluSrc, Operand};
use super::{TResult, Translator};
use crate::TranslateError;
use crate::ampere::decode::Form;
use crate::ampere::instruction::Instruction;
use crate::ampere::opcode::Opcode;
use crate::ir::{FmzMode, FpControl, FpRounding, ICompare, Opcode as IrOpcode, Reg, Value};
use crate::maxwell::translate::floating_point_add::{fp_rounding, ftz_mode};
use crate::maxwell::translate::floating_point_conversion_integer::{
    DestFormat, FloatToInt, SrcFormat, float_to_int,
};
use crate::maxwell::translate::floating_point_fused_multiply_add::{DenormMode, any_zero};
use crate::maxwell::translate::half_floating_point_add::{
    Swizzle, abs_neg_pair, demote, extract, half_precision, promote, zero_like,
};
use crate::maxwell::translate::integer_floating_point_conversion::{
    IntFormat, IntToFloat, int_to_float,
};
use crate::maxwell::translate::{
    BooleanOp, FpCompareOp, floating_point_compare, predicate_combine,
};

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    use Opcode::*;
    match op {
        Fadd => fadd(t),
        Ffma => ffma(t),
        Fmul => fmul(t),
        Fmnmx => fmnmx(t),
        Fset => fset(t),
        Fsetp => fsetp(t),
        Fsel => fsel(t),
        Fswzadd => fswzadd(t),
        Mufu => mufu(t),
        Hadd2 => hadd2(t),
        Hfma2 => hfma2(t),
        Hmul2 => hmul2(t),
        Hset2 => hset2(t),
        Hsetp2 => hsetp2(t),
        Hmnmx2 => hmnmx2(t),
        Dadd => dadd(t),
        Dfma => dfma(t),
        Dmul => dmul(t),
        Dsetp => dsetp(t),
        F2f | F2f64 => f2f(t, op == F2f64),
        F2i | F2i64 => f2i(t, op == F2i64),
        I2f | I2f64 => i2f(t, op == I2f64),
        Frnd | Frnd64 => frnd(t, op == Frnd64),
        F2fp | Uf2fp => f2fp(t),
        F2ip => f2ip(t),
        I2fp => i2fp(t),
        _ => Err(t.unsupported()),
    }
}

/// The second source of a two-source operation that may sit in either slot: the forms with a
/// register in b (64..72) put the operation's operand in c instead.
fn second(t: &Translator<'_>, half: bool) -> Result<AluSrc, TranslateError> {
    match t.form {
        Form::RegImm | Form::RegCbuf | Form::RegUreg => t.alu_c(half),
        _ => t.alu_b(half),
    }
}

/// The rounding field at 78..80.
fn rounding(i: Instruction) -> FpRounding {
    fp_rounding(i.bits(78, 2))
}

/// FFMA and FMUL denormal handling: .DNZ (bit 76) is D3D9's `x * 0 == 0`, .FTZ is bit 80.
fn denorm_mode(i: Instruction) -> DenormMode {
    if i.bit(76) {
        DenormMode::Fmz
    } else if i.bit(80) {
        DenormMode::Ftz
    } else {
        DenormMode::Keep
    }
}

fn compare_control(i: Instruction) -> FpControl {
    FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: ftz_mode(i.bit(80)),
    }
}

/// The predicate combiner at 74..76 (69..71 on the packed-half comparisons).
fn boolean_op(t: &Translator<'_>, offset: u32) -> Result<BooleanOp, TranslateError> {
    BooleanOp::from_bits(t.insn.bits(offset, 2)).ok_or_else(|| t.unsupported())
}

/// Writes `comparison` and its negation, each combined with the predicate at 87..90, to
/// the predicates at 81..84 and 84..87.
fn write_predicates(t: &mut Translator<'_>, comparison: Value, bop: BooleanOp) {
    let i = t.insn;
    let pred = t.pred(i.pred_c());
    let first = predicate_combine(&mut t.ir, comparison, pred, bop);
    let not_comparison = t.ir.logical_not(comparison);
    let second = predicate_combine(&mut t.ir, not_comparison, pred, bop);
    t.ir.set_pred(i.pred_dst0(), first);
    t.ir.set_pred(i.pred_dst1(), second);
}

fn saturate_if(t: &mut Translator<'_>, value: Value, sat: bool) -> Value {
    if sat { t.ir.fp_saturate(value) } else { value }
}

fn fadd(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let (a, b) = (t.alu_a(), second(t, false)?);
    let a = t.read_f32(a)?;
    let b = t.read_f32(b)?;
    let control = FpControl {
        no_contraction: true,
        rounding: rounding(i),
        fmz_mode: ftz_mode(i.bit(80)),
    };
    let value = t.ir.fp_add(a, b, control);
    let value = saturate_if(t, value, i.bit(77));
    t.set_f(t.reg_d(), value);
    Ok(())
}

fn ffma(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let (a, b, c) = (t.alu_a(), t.alu_b(false)?, t.alu_c(false)?);
    let a = t.read_f32(a)?;
    let b = t.read_f32(b)?;
    let c = t.read_f32(c)?;
    let denorm = denorm_mode(i);
    let control = FpControl {
        no_contraction: true,
        rounding: rounding(i),
        fmz_mode: denorm.ir_mode(),
    };
    let mut value = t.ir.fp_fma(a, b, c, control);
    let sat = i.bit(77);
    if denorm == DenormMode::Fmz && !sat {
        let zero = any_zero(&mut t.ir, a, b, Value::F32(0.0));
        value = t.ir.select(zero, c, value);
    }
    let value = saturate_if(t, value, sat);
    t.set_f(t.reg_d(), value);
    Ok(())
}

/// FMUL, with the power-of-two scale at 84..87 (4 is none, 4 - n divides by 2^n, 4 + n
/// multiplies by 2^n).
fn fmul(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let (denorm, rounding, scale) = (denorm_mode(i), rounding(i), i.bits(84, 3));
    if scale == 0 || (scale != 4 && (denorm != DenormMode::Ftz || rounding != FpRounding::Rn)) {
        return Err(t.unsupported());
    }
    let (a, b) = (t.alu_a(), t.alu_b(false)?);
    let mut a = t.read_f32(a)?;
    let b = t.read_f32(b)?;
    if scale != 4 {
        let factor = 2f32.powi(scale as i32 - 4);
        a = t.ir.fp_mul(a, Value::F32(factor), FpControl::default());
    }
    let control = FpControl {
        no_contraction: true,
        rounding,
        fmz_mode: denorm.ir_mode(),
    };
    let mut value = t.ir.fp_mul(a, b, control);
    let sat = i.bit(77);
    if denorm == DenormMode::Fmz && !sat {
        let zero = Value::F32(0.0);
        let is_zero = any_zero(&mut t.ir, a, b, zero);
        value = t.ir.select(is_zero, zero, value);
    }
    let value = saturate_if(t, value, sat);
    t.set_f(t.reg_d(), value);
    Ok(())
}

/// FMNMX: the minimum where the predicate at 87..90 holds, else the maximum. .NAN (bit 81)
/// gives the canonical NaN for an unordered pair.
fn fmnmx(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    if i.bit(82) {
        return Err(t.unsupported());
    }
    let (a, b) = (t.alu_a(), t.alu_b(false)?);
    let a = t.read_f32(a)?;
    let b = t.read_f32(b)?;
    let control = compare_control(i);
    let is_min = t.pred(i.pred_c());
    let max = t.ir.fp_max(a, b, control);
    let min = t.ir.fp_min(a, b, control);
    let value = t.ir.select(is_min, min, max);
    let dest = t.reg_d();
    if !i.bit(81) {
        t.set_f(dest, value);
        return Ok(());
    }
    let unordered = t.ir.fp_unordered(a, b);
    let bits = t.ir.bitcast_u32(value);
    let value = t.ir.select(unordered, Value::U32(0x7fff_ffff), bits);
    t.set_x(dest, value);
    Ok(())
}

/// FSET: 1.0 where the comparison, combined with the predicate at 87..90, holds, else 0.
fn fset(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let bop = boolean_op(t, 74)?;
    let (a, b) = (t.alu_a(), t.alu_b(false)?);
    let a = t.read_f32(a)?;
    let b = t.read_f32(b)?;
    let op = FpCompareOp::from_bits(i.bits(76, 4));
    let comparison = floating_point_compare(&mut t.ir, a, b, op, compare_control(i));
    let pred = t.pred(i.pred_c());
    let pass = predicate_combine(&mut t.ir, comparison, pred, bop);
    let value = t.ir.select(pass, Value::U32(0x3f80_0000), Value::U32(0));
    t.set_x(t.reg_d(), value);
    Ok(())
}

fn fsetp(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let bop = boolean_op(t, 74)?;
    let (a, b) = (t.alu_a(), t.alu_b(false)?);
    let a = t.read_f32(a)?;
    let b = t.read_f32(b)?;
    let op = FpCompareOp::from_bits(i.bits(76, 4));
    let comparison = floating_point_compare(&mut t.ir, a, b, op, compare_control(i));
    write_predicates(t, comparison, bop);
    Ok(())
}

/// FSEL: the first source where the predicate at 87..90 holds, each source with its modifiers.
fn fsel(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let (a, b) = (t.alu_a(), t.alu_b(false)?);
    let a = t.read_f32(a)?;
    let b = t.read_f32(b)?;
    let pred = t.pred(i.pred_c());
    let value = t.ir.select(pred, a, b);
    t.set_f(t.reg_d(), value);
    Ok(())
}

/// FSWZADD: one add, subtract or move per quad lane, picked by the 2-bit codes at 32..40.
/// Ampere's codes 1 (a - b) and 2 (b - a) are the other way round from the Maxwell codes the IR
/// takes. The .NDV mode (bit 77) only matters on Blackwell.
fn fswzadd(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let a = t.f(i.src0());
    let b = t.f(i.src2());
    if i.bit(77) {
        crate::fallback::note("FSWZADD", "ndv");
    }
    let codes = i.bits(32, 8) as u32;
    let swizzle = ((codes & 0x55) << 1) | ((codes & 0xaa) >> 1);
    let control = FpControl {
        no_contraction: false,
        rounding: rounding(i),
        fmz_mode: ftz_mode(i.bit(80)),
    };
    let value = t.ir.inst_flags(
        IrOpcode::FSwizzleAdd,
        control.to_flags(),
        &[a, b, Value::U32(swizzle)],
    );
    t.set_f(i.dst(), value);
    Ok(())
}

/// MUFU: the function at 74..80 of the second source. COS and SIN take turns, the IR radians.
/// RCP64H and RSQ64H work on the high word of a double. With .F16 (bit 73) the source is the
/// half the swizzle picks and the result a half in the low 16 bits.
fn mufu(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let half = i.bit(73);
    let src = t.alu_b(false)?;
    let input = if half {
        if matches!(src.operand, Operand::Imm(_)) || src.swizzle > 1 {
            return Err(t.unsupported());
        }
        let packed = t.read_u32(src)?;
        let lanes = t.ir.unpack_float_2x16(packed);
        let lane = t.ir.composite_extract(lanes, src.swizzle as u32);
        let lane = t.ir.fp_abs_neg(lane, src.abs, src.neg);
        t.ir.fp_convert(32, lane, FpControl::default())
    } else {
        t.read_f32(src)?
    };
    let op = match i.bits(74, 6) {
        0 => IrOpcode::FPCos,
        1 => IrOpcode::FPSin,
        2 => IrOpcode::FPExp2,
        3 => IrOpcode::FPLog2,
        4 => IrOpcode::FPRecip32,
        5 => IrOpcode::FPRecipSqrt32,
        code @ (6 | 7) if !half => return mufu_high_word(t, input, code == 7),
        8 => IrOpcode::FPSqrt,
        9 => IrOpcode::FPTanh,
        _ => return Err(t.unsupported()),
    };
    let input = if matches!(op, IrOpcode::FPCos | IrOpcode::FPSin) {
        t.ir.fp_mul(input, Value::F32(TAU), FpControl::default())
    } else {
        input
    };
    let value = t.ir.fp_unary32(op, input);
    if half {
        let value = t.ir.fp_convert(16, value, FpControl::default());
        write_float(t, 16, value)
    } else {
        write_float(t, 32, value)
    }
}

/// RCP64H and RSQ64H: the high word of the double whose high word is `high`.
fn mufu_high_word(t: &mut Translator<'_>, high: Value, sqrt: bool) -> TResult {
    let high = t.ir.bitcast_u32(high);
    let words = t.ir.composite_construct(&[Value::U32(0), high]);
    let value = t.ir.pack_double_2x32(words);
    let value = if sqrt {
        t.ir.fp_recip_sqrt(value)
    } else {
        t.ir.fp_recip(value)
    };
    let words = t.ir.unpack_double_2x32(value);
    let result = t.ir.composite_extract(words, 1);
    t.set_x(t.reg_d(), result);
    Ok(())
}

/// The lanes of packed-half sources, widened to F32 when their types differ (an .F32 swizzle
/// beside halves), with each source's modifiers. Also returns whether they were widened. The
/// operation's .BF16_V2 bit selects bfloat16 lanes, which are not supported.
fn half_lanes<const N: usize>(
    t: &mut Translator<'_>,
    srcs: [AluSrc; N],
    bf16_bit: u32,
) -> Result<([(Value, Value); N], bool), TranslateError> {
    if t.insn.bit(bf16_bit) {
        return Err(t.unsupported());
    }
    let mut lanes = [(Value::Void, Value::Void); N];
    for (lane, src) in lanes.iter_mut().zip(srcs) {
        let packed = t.read_u32(src)?;
        *lane = extract(&mut t.ir, packed, Swizzle::from_bits(src.swizzle));
    }
    let ty = t.ir.ty(lanes[0].0);
    let promotion = lanes.iter().any(|lane| t.ir.ty(lane.0) != ty);
    for (lane, src) in lanes.iter_mut().zip(srcs) {
        if promotion {
            *lane = promote(&mut t.ir, *lane);
        }
        *lane = abs_neg_pair(&mut t.ir, *lane, src.abs, src.neg);
    }
    Ok((lanes, promotion))
}

/// The destination value of a packed-half operation: both lanes as halves, or with .F32 (bit 78)
/// the first lane as an F32. Widened results are narrowed back to halves first, as Maxwell's
/// merge does.
fn pack_halves(t: &mut Translator<'_>, mut result: (Value, Value), promotion: bool) -> Value {
    if promotion {
        result = demote(&mut t.ir, result);
    }
    if t.insn.bit(78) {
        let value = t.ir.fp_convert(32, result.0, FpControl::default());
        return t.ir.bitcast_u32(value);
    }
    let (lhs, rhs) = demote(&mut t.ir, result);
    let vector = t.ir.composite_construct(&[lhs, rhs]);
    t.ir.pack_float_2x16(vector)
}

/// The .DNZ (bit 76) and .FTZ (bit 80) modes of HMUL2 and HFMA2 as Maxwell's precision field.
fn half_mode(i: Instruction) -> (FmzMode, bool) {
    half_precision(if i.bit(76) { 2 } else { u64::from(i.bit(80)) })
}

fn hadd2(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let srcs = [t.alu_a(), second(t, true)?];
    let ([a, b], promotion) = half_lanes(t, srcs, 85)?;
    let control = FpControl {
        no_contraction: true,
        rounding: FpRounding::DontCare,
        fmz_mode: ftz_mode(i.bit(80)),
    };
    let lhs = t.ir.fp_add(a.0, b.0, control);
    let rhs = t.ir.fp_add(a.1, b.1, control);
    let sat = i.bit(77);
    let result = (saturate_if(t, lhs, sat), saturate_if(t, rhs, sat));
    let value = pack_halves(t, result, promotion);
    t.set_x(t.reg_d(), value);
    Ok(())
}

fn hmul2(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    if i.bit(78) || i.bit(79) {
        return Err(t.unsupported());
    }
    let srcs = [t.alu_a(), t.alu_b(true)?];
    let ([a, b], promotion) = half_lanes(t, srcs, 85)?;
    let (fmz_mode, fmz) = half_mode(i);
    let control = FpControl {
        no_contraction: true,
        rounding: FpRounding::DontCare,
        fmz_mode,
    };
    let mut lhs = t.ir.fp_mul(a.0, b.0, control);
    let mut rhs = t.ir.fp_mul(a.1, b.1, control);
    let sat = i.bit(77);
    if fmz && !sat {
        let zero = zero_like(&mut t.ir, lhs);
        let lhs_zero = any_zero(&mut t.ir, a.0, b.0, zero);
        lhs = t.ir.select(lhs_zero, zero, lhs);
        let rhs_zero = any_zero(&mut t.ir, a.1, b.1, zero);
        rhs = t.ir.select(rhs_zero, zero, rhs);
    }
    let result = (saturate_if(t, lhs, sat), saturate_if(t, rhs, sat));
    let value = pack_halves(t, result, promotion);
    t.set_x(t.reg_d(), value);
    Ok(())
}

/// HFMA2. .H0_NH1 (bit 86) swaps the halves of the second source and negates the new low half;
/// .RELU (bit 79) clamps the packed result where the predicate at 87..90 holds.
fn hfma2(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let (sat, relu) = (i.bit(77), i.bit(79));
    if relu && (sat || i.bit(78)) {
        return Err(t.unsupported());
    }
    let srcs = [t.alu_a(), t.alu_b(true)?, t.alu_c(true)?];
    if i.bit(86) && (srcs[1].swizzle != 0 || srcs[1].abs) {
        return Err(t.unsupported());
    }
    let ([a, mut b, c], promotion) = half_lanes(t, srcs, 85)?;
    if i.bit(86) {
        b = (t.ir.fp_neg(b.1), b.0);
    }
    let (fmz_mode, fmz) = half_mode(i);
    let control = FpControl {
        no_contraction: true,
        rounding: FpRounding::DontCare,
        fmz_mode,
    };
    let mut lhs = t.ir.fp_fma(a.0, b.0, c.0, control);
    let mut rhs = t.ir.fp_fma(a.1, b.1, c.1, control);
    if fmz && !sat {
        let zero = zero_like(&mut t.ir, a.0);
        let lhs_zero = any_zero(&mut t.ir, a.0, b.0, zero);
        lhs = t.ir.select(lhs_zero, c.0, lhs);
        let rhs_zero = any_zero(&mut t.ir, a.1, b.1, zero);
        rhs = t.ir.select(rhs_zero, c.1, rhs);
    }
    let result = (saturate_if(t, lhs, sat), saturate_if(t, rhs, sat));
    let mut value = pack_halves(t, result, promotion);
    if relu {
        value = relu_halves(t, value);
    }
    t.set_x(t.reg_d(), value);
    Ok(())
}

/// The .RELU clamp of packed halves: negative lanes (negative zero included) become +0 and NaNs
/// the canonical 0x7fff, where the predicate at 87..90 holds.
fn relu_halves(t: &mut Translator<'_>, packed: Value) -> Value {
    let [low, high] = [0, 16].map(|offset| {
        let half =
            t.ir.bit_field_extract(packed, Value::U32(offset), Value::U32(16), false);
        let magnitude = t.ir.bitwise_and(half, Value::U32(0x7fff));
        let nan =
            t.ir.icompare(ICompare::UGreaterThan, magnitude, Value::U32(0x7c00));
        let sign = t.ir.bitwise_and(half, Value::U32(0x8000));
        let negative = t.ir.icompare(ICompare::NotEqual, sign, Value::U32(0));
        let positive = t.ir.select(negative, Value::U32(0), half);
        t.ir.select(nan, Value::U32(0x7fff), positive)
    });
    let high = t.ir.shift_left_logical(high, Value::U32(16));
    let clamped = t.ir.bitwise_or(low, high);
    let pred = t.pred(t.insn.pred_c());
    t.ir.select(pred, clamped, packed)
}

/// The packed-half comparisons of both lanes, each combined with the predicate at 87..90 by the
/// combiner at 69..71.
fn half_compare(t: &mut Translator<'_>) -> Result<(Value, Value), TranslateError> {
    let i = t.insn;
    let bop = boolean_op(t, 69)?;
    let srcs = [t.alu_a(), second(t, true)?];
    let ([a, b], _) = half_lanes(t, srcs, 65)?;
    let control = compare_control(i);
    let pred = t.pred(i.pred_c());
    let op = FpCompareOp::from_bits(i.bits(76, 4));
    let lhs = floating_point_compare(&mut t.ir, a.0, b.0, op, control);
    let rhs = floating_point_compare(&mut t.ir, a.1, b.1, op, control);
    let lhs = predicate_combine(&mut t.ir, lhs, pred, bop);
    let rhs = predicate_combine(&mut t.ir, rhs, pred, bop);
    Ok((lhs, rhs))
}

/// HSET2: per lane 1.0 (.BF, bit 71) or all ones where the comparison holds, else 0.
fn hset2(t: &mut Translator<'_>) -> TResult {
    let (lhs, rhs) = half_compare(t)?;
    let pass: u32 = if t.insn.bit(71) { 0x3c00 } else { 0xffff };
    let low = t.ir.select(lhs, Value::U32(pass), Value::U32(0));
    let high = t.ir.select(rhs, Value::U32(pass << 16), Value::U32(0));
    let value = t.ir.bitwise_or(low, high);
    t.set_x(t.reg_d(), value);
    Ok(())
}

/// HSETP2: the two lane results, or with .H_AND (bit 71) their conjunction and its negation.
fn hsetp2(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let (lhs, rhs) = half_compare(t)?;
    let (first, second) = if i.bit(71) {
        let both = t.ir.logical_and(lhs, rhs);
        (both, t.ir.logical_not(both))
    } else {
        (lhs, rhs)
    };
    t.ir.set_pred(i.pred_dst0(), first);
    t.ir.set_pred(i.pred_dst1(), second);
    Ok(())
}

/// HMNMX2: per lane the minimum where the predicate at 87..90 holds, else the maximum, computed
/// on exactly widened halves.
fn hmnmx2(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    if i.bit(78) || i.bit(81) || i.bit(82) {
        return Err(t.unsupported());
    }
    let srcs = [t.alu_a(), t.alu_b(true)?];
    let ([a, b], _) = half_lanes(t, srcs, 85)?;
    let (a, b) = (promote(&mut t.ir, a), promote(&mut t.ir, b));
    let control = compare_control(i);
    let is_min = t.pred(i.pred_c());
    let [lhs, rhs] = [(a.0, b.0), (a.1, b.1)].map(|(x, y)| {
        let max = t.ir.fp_max(x, y, control);
        let min = t.ir.fp_min(x, y, control);
        t.ir.select(is_min, min, max)
    });
    let (lhs, rhs) = demote(&mut t.ir, (lhs, rhs));
    let vector = t.ir.composite_construct(&[lhs, rhs]);
    let value = t.ir.pack_float_2x16(vector);
    t.set_x(t.reg_d(), value);
    Ok(())
}

fn double_control(i: Instruction) -> FpControl {
    FpControl {
        no_contraction: true,
        rounding: rounding(i),
        fmz_mode: FmzMode::None,
    }
}

/// DADD: the first source plus the third; the second slot is unused.
fn dadd(t: &mut Translator<'_>) -> TResult {
    let (a, c) = (t.alu_a(), t.alu_c(false)?);
    let a = t.read_f64(a)?;
    let c = t.read_f64(c)?;
    let value = t.ir.fp_add(a, c, double_control(t.insn));
    t.set_d(t.reg_d(), value)
}

fn dfma(t: &mut Translator<'_>) -> TResult {
    let (a, b, c) = (t.alu_a(), t.alu_b(false)?, t.alu_c(false)?);
    let a = t.read_f64(a)?;
    let b = t.read_f64(b)?;
    let c = t.read_f64(c)?;
    let value = t.ir.fp_fma(a, b, c, double_control(t.insn));
    t.set_d(t.reg_d(), value)
}

fn dmul(t: &mut Translator<'_>) -> TResult {
    let (a, b) = (t.alu_a(), t.alu_b(false)?);
    let a = t.read_f64(a)?;
    let b = t.read_f64(b)?;
    let value = t.ir.fp_mul(a, b, double_control(t.insn));
    t.set_d(t.reg_d(), value)
}

fn dsetp(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let bop = boolean_op(t, 74)?;
    let (a, b) = (t.alu_a(), second(t, false)?);
    let a = t.read_f64(a)?;
    let b = t.read_f64(b)?;
    let op = FpCompareOp::from_bits(i.bits(76, 4));
    let comparison = floating_point_compare(&mut t.ir, a, b, op, FpControl::default());
    write_predicates(t, comparison, bop);
    Ok(())
}

/// The width of a float format field (log2 of the size in bytes).
const fn float_width(field: u64) -> Option<u32> {
    match field {
        1 => Some(16),
        2 => Some(32),
        3 => Some(64),
        _ => None,
    }
}

/// A float source of `width` bits with its modifiers; an F16 is the half its swizzle picks.
fn read_float(t: &mut Translator<'_>, width: u32, src: AluSrc) -> Result<Value, TranslateError> {
    match width {
        16 => {
            if src.swizzle > 1 {
                return Err(t.unsupported());
            }
            let packed = t.read_u32(src)?;
            let lanes = t.ir.unpack_float_2x16(packed);
            let lane = t.ir.composite_extract(lanes, src.swizzle as u32);
            Ok(t.ir.fp_abs_neg(lane, src.abs, src.neg))
        }
        32 => t.read_f32(src),
        _ => t.read_f64(src),
    }
}

/// Writes a float of `width` bits to the destination; a half goes to the low 16 bits.
fn write_float(t: &mut Translator<'_>, width: u32, value: Value) -> TResult {
    let dest = t.reg_d();
    match width {
        16 => {
            let zero = t.ir.fp_convert(16, Value::F32(0.0), FpControl::default());
            let vector = t.ir.composite_construct(&[value, zero]);
            let packed = t.ir.pack_float_2x16(vector);
            t.set_x(dest, packed);
            Ok(())
        }
        32 => {
            t.set_f(dest, value);
            Ok(())
        }
        _ => t.set_d(dest, value),
    }
}

/// F2F: source format at 84..86, destination at 75..77, rounding at 78..80 for narrowing, .FTZ
/// at bit 80 unless a double is involved. `wide` is the 64-bit form, which must involve one.
fn f2f(t: &mut Translator<'_>, wide: bool) -> TResult {
    let i = t.insn;
    let (Some(src_width), Some(dst_width)) =
        (float_width(i.bits(84, 2)), float_width(i.bits(75, 2)))
    else {
        return Err(t.unsupported());
    };
    let any_f64 = src_width == 64 || dst_width == 64;
    if wide != any_f64 || (any_f64 && (src_width == 16 || dst_width == 16)) {
        return Err(t.unsupported());
    }
    let src = t.alu_b(false)?;
    let input = read_float(t, src_width, src)?;
    let mut control = FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: if i.bit(80) && !any_f64 {
            FmzMode::Ftz
        } else {
            FmzMode::None
        },
    };
    let value = if src_width == dst_width {
        if i.bits(78, 2) != 0 {
            crate::fallback::note("F2F", "same_width_rounding_op");
        }
        let zero = match src_width {
            16 => t.ir.fp_convert(16, Value::F32(0.0), FpControl::default()),
            32 => Value::F32(0.0),
            _ => Value::F64(0.0),
        };
        t.ir.fp_add(input, zero, control)
    } else {
        control.rounding = rounding(i);
        t.ir.fp_convert(dst_width, input, control)
    };
    write_float(t, dst_width, value)
}

/// F2I with Maxwell's clamping and NaN results; .NTZ (bit 77) turns NaN into 0. Signedness at
/// 72, destination size at 75..77, rounding at 78..80, .FTZ at 80, source format at 84..86.
fn f2i(t: &mut Translator<'_>, wide: bool) -> TResult {
    let i = t.insn;
    let Some(src) = SrcFormat::from_bits(i.bits(84, 2)) else {
        return Err(t.unsupported());
    };
    let dest = DestFormat::from_bits(i.bits(75, 2));
    if wide != (src == SrcFormat::F64 || dest == DestFormat::I64) {
        return Err(t.unsupported());
    }
    let width = match src {
        SrcFormat::F16 => 16,
        SrcFormat::F32 => 32,
        SrcFormat::F64 => 64,
    };
    let operand = t.alu_b(false)?;
    let op_a = read_float(t, width, operand)?;
    let conversion = FloatToInt {
        src,
        dest,
        signed: i.bit(72),
        rounding: i.bits(78, 2),
        ftz: i.bit(80),
    };
    let mut result = float_to_int(&mut t.ir, conversion, op_a);
    let zero = if dest == DestFormat::I64 {
        Value::U64(0)
    } else {
        Value::U32(0)
    };
    if i.bit(77) {
        let nan = t.ir.fp_is_nan(op_a);
        result = t.ir.select(nan, zero, result);
    }
    let dest_reg = t.reg_d();
    if dest == DestFormat::I64 {
        t.set_l(dest_reg, result)
    } else {
        t.set_x(dest_reg, result);
        Ok(())
    }
}

/// An integer source of I2F or I2FP: a U64 for 64-bit formats, else a U32. Integer sources take
/// no modifiers.
fn read_int(t: &mut Translator<'_>, format: IntFormat) -> Result<Value, TranslateError> {
    let src = t.alu_b(false)?;
    if src.abs || src.neg {
        return Err(t.unsupported());
    }
    if format == IntFormat::U64 {
        t.read_u64(src)
    } else {
        t.read_u32(src)
    }
}

/// I2F: source size at 84..86 with the byte or halfword at 60..62 for narrow sources,
/// signedness at 74, destination format at 75..77, rounding at 78..80.
fn i2f(t: &mut Translator<'_>, wide: bool) -> TResult {
    let i = t.insn;
    let format = IntFormat::from_bits(i.bits(84, 2));
    let Some(dst_bits) = float_width(i.bits(75, 2)) else {
        return Err(t.unsupported());
    };
    if wide != (format == IntFormat::U64 || dst_bits == 64) {
        return Err(t.unsupported());
    }
    let index = match (t.form, format) {
        (Form::ImmReg, _) => 0,
        (_, IntFormat::U16) => i.bits(60, 2) * 2,
        _ => i.bits(60, 2),
    };
    let conversion = IntToFloat {
        format,
        selector: index,
        signed: i.bit(74),
        abs: false,
        neg: false,
        dst_bits,
        rounding: rounding(i),
        native: true,
    };
    if !conversion.valid_selector() || index > 3 {
        return Err(t.unsupported());
    }
    let src = read_int(t, format)?;
    let value = int_to_float(&mut t.ir, conversion, src);
    write_float(t, dst_bits, value)
}

/// FRND: rounds to an integral value of the same format (at 84..86 and 75..77) by the mode at
/// 78..80: nearest even, floor, ceil or trunc.
fn frnd(t: &mut Translator<'_>, wide: bool) -> TResult {
    let i = t.insn;
    let Some(width) = float_width(i.bits(84, 2)) else {
        return Err(t.unsupported());
    };
    if float_width(i.bits(75, 2)) != Some(width) || wide != (width == 64) {
        return Err(t.unsupported());
    }
    let src = t.alu_b(false)?;
    let input = read_float(t, width, src)?;
    let control = FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: if i.bit(80) && width != 64 {
            FmzMode::Ftz
        } else {
            FmzMode::None
        },
    };
    let value = match i.bits(78, 2) {
        0 => t.ir.fp_round_even(input, control),
        1 => t.ir.fp_floor(input, control),
        2 => t.ir.fp_ceil(input, control),
        _ => t.ir.fp_trunc(input, control),
    };
    write_float(t, width, value)
}

/// F2FP and UF2FP, F32 to packed F16 rounded by the mode at 79..81. .PACK_AB puts the second
/// source in the low half and the first in the high half; .MERGE_C (bit 78) keeps the third
/// source's low half, or its high half with .H1 (bit 72), above the second source.
fn f2fp(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let merge = i.bit(78);
    let unknown = i.bit(73)
        || i.bit(74)
        || i.bit(77)
        || i.bits(81, 3) != 0
        || i.bits(86, 5) != 0
        || (i.bit(72) && !merge)
        || !matches!(i.bits(75, 2), 0 | 1)
        || !matches!(i.bits(84, 2), 0 | 2);
    if unknown {
        return Err(t.unsupported());
    }
    let control = FpControl {
        no_contraction: false,
        rounding: fp_rounding(i.bits(79, 2)),
        fmz_mode: FmzMode::DontCare,
    };
    let b = t.alu_b(false)?;
    let b = t.read_f32(b)?;
    let low = t.ir.fp_convert(16, b, control);
    let high = if merge {
        let c = t.alu_c(false)?;
        let c = t.read_u32(c)?;
        let halves = t.ir.unpack_float_2x16(c);
        t.ir.composite_extract(halves, u32::from(i.bit(72)))
    } else {
        let a = t.alu_a();
        let a = t.read_f32(a)?;
        t.ir.fp_convert(16, a, control)
    };
    let vector = t.ir.composite_construct(&[low, high]);
    let value = t.ir.pack_float_2x16(vector);
    t.set_x(t.reg_d(), value);
    Ok(())
}

/// F2IP.U8.F32: the second source's byte in bits 0..8 and the first source's in 8..16, each
/// rounded by the mode at 78..80 and clamped to 0..255 with NaN as 0. Only the encoding titles
/// use (bit 74 set, RZ third source) is known.
fn f2ip(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let known = i.bit(74)
        && !i.bit(72)
        && !i.bit(73)
        && i.bits(75, 3) == 0
        && i.bits(80, 11) == 0
        && t.alu_c(false)?.operand == Operand::Reg(Reg::RZ);
    if !known {
        return Err(t.unsupported());
    }
    let conversion = FloatToInt {
        src: SrcFormat::F32,
        dest: DestFormat::I8,
        signed: false,
        rounding: i.bits(78, 2),
        ftz: false,
    };
    let (a, b) = (t.alu_a(), t.alu_b(false)?);
    let b = t.read_f32(b)?;
    let low = float_to_int(&mut t.ir, conversion, b);
    let a = t.read_f32(a)?;
    let high = float_to_int(&mut t.ir, conversion, a);
    let high = t.ir.shift_left_logical(high, Value::U32(8));
    let value = t.ir.bitwise_or(low, high);
    t.set_x(t.reg_d(), value);
    Ok(())
}

/// I2FP: a 32-bit integer (signedness at 74) to an F32 or F16 (75..77), rounded by 78..80.
fn i2fp(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let dst_bits = match i.bits(75, 2) {
        1 => 16,
        2 => 32,
        _ => return Err(t.unsupported()),
    };
    if i.bits(84, 2) != 2 {
        return Err(t.unsupported());
    }
    let conversion = IntToFloat {
        format: IntFormat::U32,
        selector: 0,
        signed: i.bit(74),
        abs: false,
        neg: false,
        dst_bits,
        rounding: rounding(i),
        native: true,
    };
    let src = read_int(t, IntFormat::U32)?;
    let value = int_to_float(&mut t.ir, conversion, src);
    write_float(t, dst_bits, value)
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::ResolutionScale;

    use super::super::tests::{TestEnv, encode, translate};
    use crate::ampere::decode::Form::{self, *};
    use crate::ampere::opcode::Opcode::{self, *};
    use crate::profile::HostTranslateInfo;
    use crate::translate::TranslateOptions;
    use crate::{Profile, RuntimeInfo, ShaderStage};

    const D: u128 = 2 << 16;
    const A: u128 = 4 << 24;
    const B: u128 = 6 << 32;
    const C: u128 = 8 << 64;
    const RZ_C: u128 = 255 << 64;

    const fn bit(n: u32) -> u128 {
        1 << n
    }

    const fn field(value: u128, offset: u32) -> u128 {
        value << offset
    }

    /// Asserts the dump lines of one instruction, without the block header.
    fn check(op: Opcode, form: Form, fields: u128, expected: &[&str]) {
        let dump = translate(&[encode(op, form, fields)]).unwrap();
        let got: Vec<&str> = dump.lines().skip(1).map(str::trim).collect();
        assert_eq!(got, expected);
    }

    fn rejects(op: Opcode, form: Form, fields: u128) {
        assert!(translate(&[encode(op, form, fields)]).is_err());
    }

    #[test]
    fn fadd_register_modifiers() {
        check(
            Fadd,
            RegReg,
            D | A | B | bit(72) | bit(62) | bit(77) | field(1, 78) | bit(80),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPNeg32 %1",
                "%3 = GetRegister R6",
                "%4 = BitCastF32U32 %3",
                "%5 = FPAbs32 %4",
                "%6 = FPAdd32 %2, %5 (flags 0x15)",
                "%7 = FPSaturate32 %6",
                "%8 = BitCastU32F32 %7",
                "SetRegister R2, %8",
            ],
        );
    }

    #[test]
    fn fadd_immediate_in_c() {
        check(
            Fadd,
            RegImm,
            D | A | RZ_C | field(0x3f80_0000, 32),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPAdd32 %1, #1.0f (flags 0x33)",
                "%3 = BitCastU32F32 %2",
                "SetRegister R2, %3",
            ],
        );
    }

    #[test]
    fn fadd_cbuf_negated() {
        check(
            Fadd,
            CbufReg,
            D | A | RZ_C | field(1, 54) | field(0x10, 38) | bit(63),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetCbufF32 #0x1, #0x10",
                "%3 = FPNeg32 %2",
                "%4 = FPAdd32 %1, %3 (flags 0x33)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn fadd_uniform_in_c() {
        check(
            Fadd,
            RegUreg,
            D | A | RZ_C | field(5, 32) | bit(91) | bit(62),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister UR5",
                "%3 = BitCastF32U32 %2",
                "%4 = FPAbs32 %3",
                "%5 = FPAdd32 %1, %4 (flags 0x33)",
                "%6 = BitCastU32F32 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn ffma_registers() {
        check(
            Ffma,
            RegReg,
            D | A | B | C | bit(75) | field(3, 78),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R6",
                "%3 = BitCastF32U32 %2",
                "%4 = GetRegister R8",
                "%5 = BitCastF32U32 %4",
                "%6 = FPNeg32 %5",
                "%7 = FPFma32 %1, %3, %6 (flags 0x39)",
                "%8 = BitCastU32F32 %7",
                "SetRegister R2, %8",
            ],
        );
    }

    #[test]
    fn ffma_dnz_selects_addend() {
        check(
            Ffma,
            RegReg,
            D | A | B | C | bit(76),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R6",
                "%3 = BitCastF32U32 %2",
                "%4 = GetRegister R8",
                "%5 = BitCastF32U32 %4",
                "%6 = FPFma32 %1, %3, %5 (flags 0x13)",
                "%7 = FPOrdEqual32 %1, #0.0f",
                "%8 = FPOrdEqual32 %3, #0.0f",
                "%9 = LogicalOr %7, %8",
                "%10 = SelectF32 %9, %5, %6",
                "%11 = BitCastU32F32 %10",
                "SetRegister R2, %11",
            ],
        );
    }

    #[test]
    fn ffma_register_cbuf() {
        check(
            Ffma,
            RegCbuf,
            D | A | C | field(2, 54) | field(0x20, 38) | bit(74),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R8",
                "%3 = BitCastF32U32 %2",
                "%4 = FPAbs32 %3",
                "%5 = GetCbufF32 #0x2, #0x20",
                "%6 = FPFma32 %1, %4, %5 (flags 0x33)",
                "%7 = BitCastU32F32 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn fmul_scale_doubles() {
        check(
            Fmul,
            RegReg,
            D | A | B | RZ_C | field(5, 84) | bit(80),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R6",
                "%3 = BitCastF32U32 %2",
                "%4 = FPMul32 %1, #2.0f",
                "%5 = FPMul32 %4, %3 (flags 0x13)",
                "%6 = BitCastU32F32 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn fmul_dnz() {
        check(
            Fmul,
            ImmReg,
            D | A | RZ_C | field(0x4000_0000, 32) | field(4, 84) | bit(76) | bit(77),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPMul32 %1, #2.0f (flags 0x13)",
                "%3 = FPSaturate32 %2",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn fmul_scale_zero() {
        rejects(Fmul, RegReg, D | A | B | RZ_C);
    }

    #[test]
    fn fmul_scale_needs_ftz() {
        rejects(Fmul, RegReg, D | A | B | RZ_C | field(3, 84));
    }

    #[test]
    fn fmnmx_min_nan() {
        check(
            Fmnmx,
            RegReg,
            D | A | B | field(1, 87) | bit(90) | bit(81) | bit(80),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R6",
                "%3 = BitCastF32U32 %2",
                "%4 = GetPred P1",
                "%5 = LogicalNot %4",
                "%6 = FPMax32 %1, %3 (flags 0x10)",
                "%7 = FPMin32 %1, %3 (flags 0x10)",
                "%8 = SelectF32 %5, %7, %6",
                "%9 = FPIsNan32 %1",
                "%10 = FPIsNan32 %3",
                "%11 = LogicalOr %9, %10",
                "%12 = BitCastU32F32 %8",
                "%13 = SelectU32 %11, #0x7fffffff, %12",
                "SetRegister R2, %13",
            ],
        );
    }

    #[test]
    fn fmnmx_xorsign() {
        rejects(Fmnmx, RegReg, D | A | B | bit(82));
    }

    #[test]
    fn fset_or_gt() {
        check(
            Fset,
            RegReg,
            D | A | B | field(1, 74) | field(4, 76) | field(2, 87),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R6",
                "%3 = BitCastF32U32 %2",
                "%4 = FPOrdGreaterThan32 %1, %3 (flags 0x30)",
                "%5 = GetPred P2",
                "%6 = LogicalOr %4, %5",
                "%7 = SelectU32 %6, #0x3f800000, #0x0",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn fsetp_lt() {
        check(
            Fsetp,
            ImmReg,
            A | field(0x3f00_0000, 32)
                | field(1, 81)
                | field(7, 84)
                | field(1, 76)
                | field(7, 87)
                | bit(73),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPAbs32 %1",
                "%3 = FPOrdLessThan32 %2, #0.5f (flags 0x30)",
                "%4 = LogicalAnd %3, #true",
                "%5 = LogicalNot %3",
                "%6 = LogicalAnd %5, #true",
                "SetPred P1, %4",
            ],
        );
    }

    #[test]
    fn fsetp_reserved_combiner() {
        rejects(Fsetp, RegReg, A | B | field(3, 74));
    }

    #[test]
    fn fsel_negated_predicate() {
        check(
            Fsel,
            RegReg,
            D | A | B | field(3, 87) | bit(90) | bit(72) | bit(62),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPNeg32 %1",
                "%3 = GetRegister R6",
                "%4 = BitCastF32U32 %3",
                "%5 = FPAbs32 %4",
                "%6 = GetPred P3",
                "%7 = LogicalNot %6",
                "%8 = SelectF32 %7, %2, %5",
                "%9 = BitCastU32F32 %8",
                "SetRegister R2, %9",
            ],
        );
    }

    #[test]
    fn fswzadd_ddx() {
        check(
            Fswzadd,
            Fixed,
            D | A | C | field(0x66, 32) | field(2, 78),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R8",
                "%3 = BitCastF32U32 %2",
                "%4 = FSwizzleAdd %1, %3, #0x99 (flags 0x36)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn mufu_sin_in_turns() {
        check(
            Mufu,
            RegReg,
            D | B | field(1, 74),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPMul32 %1, #6.2831855f",
                "%3 = FPSin %2",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn mufu_rsq_abs() {
        check(
            Mufu,
            RegReg,
            D | B | bit(62) | field(5, 74),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPAbs32 %1",
                "%3 = FPRecipSqrt32 %2",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn mufu_ex2_half() {
        check(
            Mufu,
            RegReg,
            D | B | bit(73) | field(1, 60) | bit(63) | field(2, 74),
            &[
                "%0 = GetRegister R6",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x1",
                "%3 = FPNeg16 %2",
                "%4 = ConvertF32F16 %3",
                "%5 = FPExp2 %4",
                "%6 = ConvertF16F32 %5",
                "%7 = ConvertF16F32 #0.0f",
                "%8 = CompositeConstructF16x2 %6, %7",
                "%9 = PackFloat2x16 %8",
                "SetRegister R2, %9",
            ],
        );
    }

    #[test]
    fn mufu_rcp64h() {
        check(
            Mufu,
            RegReg,
            D | B | field(6, 74),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = BitCastU32F32 %1",
                "%3 = CompositeConstructU32x2 #0x0, %2",
                "%4 = PackDouble2x32 %3",
                "%5 = FPRecip64 %4",
                "%6 = UnpackDouble2x32 %5",
                "%7 = CompositeExtractU32x2 %6, #0x1",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn mufu_reserved() {
        rejects(Mufu, RegReg, D | B | field(10, 74));
    }

    #[test]
    fn hadd2_register_swizzles() {
        check(
            Hadd2,
            RegReg,
            D | A | B | field(2, 74) | bit(63) | field(3, 60) | bit(77),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = GetRegister R6",
                "%4 = UnpackFloat2x16 %3",
                "%5 = CompositeExtractF16x2 %4, #0x1",
                "%6 = FPNeg16 %5",
                "%7 = FPNeg16 %5",
                "%8 = FPAdd16 %2, %6 (flags 0x31)",
                "%9 = FPAdd16 %2, %7 (flags 0x31)",
                "%10 = FPSaturate16 %8",
                "%11 = FPSaturate16 %9",
                "%12 = CompositeConstructF16x2 %10, %11",
                "%13 = PackFloat2x16 %12",
                "SetRegister R2, %13",
            ],
        );
    }

    #[test]
    fn hadd2_immediate_f32_output() {
        check(
            Hadd2,
            RegImm,
            D | A | RZ_C | field(0x3c00_bc00, 32) | bit(78) | bit(80),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = UnpackFloat2x16 #0x3c00bc00",
                "%5 = CompositeExtractF16x2 %4, #0x0",
                "%6 = CompositeExtractF16x2 %4, #0x1",
                "%7 = FPAdd16 %2, %5 (flags 0x11)",
                "%8 = FPAdd16 %3, %6 (flags 0x11)",
                "%9 = ConvertF32F16 %7",
                "%10 = BitCastU32F32 %9",
                "SetRegister R2, %10",
            ],
        );
    }

    #[test]
    fn hadd2_f32_swizzle_promotes() {
        check(
            Hadd2,
            RegReg,
            D | A | B | field(1, 60) | bit(72),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetRegister R6",
                "%5 = BitCastF32U32 %4",
                "%6 = ConvertF32F16 %2",
                "%7 = ConvertF32F16 %3",
                "%8 = FPNeg32 %6",
                "%9 = FPNeg32 %7",
                "%10 = FPAdd32 %8, %5 (flags 0x31)",
                "%11 = FPAdd32 %9, %5 (flags 0x31)",
                "%12 = ConvertF16F32 %10",
                "%13 = ConvertF16F32 %11",
                "%14 = CompositeConstructF16x2 %12, %13",
                "%15 = PackFloat2x16 %14",
                "SetRegister R2, %15",
            ],
        );
    }

    #[test]
    fn hadd2_bf16() {
        rejects(Hadd2, RegReg, D | A | B | bit(85));
    }

    #[test]
    fn hfma2_relu() {
        check(
            Hfma2,
            RegReg,
            D | A | B | C | bit(79) | field(1, 87) | bit(84) | field(2, 81),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetRegister R6",
                "%5 = UnpackFloat2x16 %4",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = GetRegister R8",
                "%9 = UnpackFloat2x16 %8",
                "%10 = CompositeExtractF16x2 %9, #0x0",
                "%11 = FPNeg16 %10",
                "%12 = FPNeg16 %10",
                "%13 = FPFma16 %2, %6, %11 (flags 0x31)",
                "%14 = FPFma16 %3, %7, %12 (flags 0x31)",
                "%15 = CompositeConstructF16x2 %13, %14",
                "%16 = PackFloat2x16 %15",
                "%17 = BitFieldUExtract %16, #0x0, #0x10",
                "%18 = BitwiseAnd32 %17, #0x7fff",
                "%19 = UGreaterThan %18, #0x7c00",
                "%20 = BitwiseAnd32 %17, #0x8000",
                "%21 = INotEqual %20, #0x0",
                "%22 = SelectU32 %21, #0x0, %17",
                "%23 = SelectU32 %19, #0x7fff, %22",
                "%24 = BitFieldUExtract %16, #0x10, #0x10",
                "%25 = BitwiseAnd32 %24, #0x7fff",
                "%26 = UGreaterThan %25, #0x7c00",
                "%27 = BitwiseAnd32 %24, #0x8000",
                "%28 = INotEqual %27, #0x0",
                "%29 = SelectU32 %28, #0x0, %24",
                "%30 = SelectU32 %26, #0x7fff, %29",
                "%31 = ShiftLeftLogical32 %30, #0x10",
                "%32 = BitwiseOr32 %23, %31",
                "%33 = GetPred P1",
                "%34 = SelectU32 %33, %32, %16",
                "SetRegister R2, %34",
            ],
        );
    }

    #[test]
    fn hfma2_dnz_h0_nh1() {
        check(
            Hfma2,
            RegReg,
            D | A | B | C | bit(76) | bit(86),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetRegister R6",
                "%5 = UnpackFloat2x16 %4",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = GetRegister R8",
                "%9 = UnpackFloat2x16 %8",
                "%10 = CompositeExtractF16x2 %9, #0x0",
                "%11 = CompositeExtractF16x2 %9, #0x1",
                "%12 = FPNeg16 %7",
                "%13 = FPFma16 %2, %12, %10 (flags 0x21)",
                "%14 = FPFma16 %3, %6, %11 (flags 0x21)",
                "%15 = ConvertF16F32 #0.0f",
                "%16 = FPOrdEqual16 %2, %15",
                "%17 = FPOrdEqual16 %12, %15",
                "%18 = LogicalOr %16, %17",
                "%19 = SelectF16 %18, %10, %13",
                "%20 = FPOrdEqual16 %3, %15",
                "%21 = FPOrdEqual16 %6, %15",
                "%22 = LogicalOr %20, %21",
                "%23 = SelectF16 %22, %11, %14",
                "%24 = CompositeConstructF16x2 %19, %23",
                "%25 = PackFloat2x16 %24",
                "SetRegister R2, %25",
            ],
        );
    }

    #[test]
    fn hfma2_cbuf_c() {
        check(
            Hfma2,
            RegCbuf,
            D | A | C | field(1, 54) | field(8, 38) | field(1, 60) | bit(83),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetRegister R8",
                "%5 = UnpackFloat2x16 %4",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = GetCbufU32 #0x1, #0x8",
                "%9 = BitCastF32U32 %8",
                "%10 = ConvertF32F16 %2",
                "%11 = ConvertF32F16 %3",
                "%12 = ConvertF32F16 %6",
                "%13 = ConvertF32F16 %7",
                "%14 = FPAbs32 %12",
                "%15 = FPAbs32 %13",
                "%16 = FPFma32 %10, %14, %9 (flags 0x31)",
                "%17 = FPFma32 %11, %15, %9 (flags 0x31)",
                "%18 = ConvertF16F32 %16",
                "%19 = ConvertF16F32 %17",
                "%20 = CompositeConstructF16x2 %18, %19",
                "%21 = PackFloat2x16 %20",
                "SetRegister R2, %21",
            ],
        );
    }

    #[test]
    fn hmul2_ftz_abs() {
        check(
            Hmul2,
            RegReg,
            D | A | B | bit(73) | bit(80),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetRegister R6",
                "%5 = UnpackFloat2x16 %4",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = FPAbs16 %2",
                "%9 = FPAbs16 %3",
                "%10 = FPMul16 %8, %6 (flags 0x11)",
                "%11 = FPMul16 %9, %7 (flags 0x11)",
                "%12 = CompositeConstructF16x2 %10, %11",
                "%13 = PackFloat2x16 %12",
                "SetRegister R2, %13",
            ],
        );
    }

    #[test]
    fn hmul2_relu() {
        rejects(Hmul2, RegReg, D | A | B | bit(79));
    }

    #[test]
    fn hset2_bf_and() {
        check(
            Hset2,
            RegReg,
            D | A | B | bit(71) | field(2, 76) | field(7, 87),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetRegister R6",
                "%5 = UnpackFloat2x16 %4",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = FPOrdEqual16 %2, %6 (flags 0x30)",
                "%9 = FPOrdEqual16 %3, %7 (flags 0x30)",
                "%10 = LogicalAnd %8, #true",
                "%11 = LogicalAnd %9, #true",
                "%12 = SelectU32 %10, #0x3c00, #0x0",
                "%13 = SelectU32 %11, #0x3c000000, #0x0",
                "%14 = BitwiseOr32 %12, %13",
                "SetRegister R2, %14",
            ],
        );
    }

    #[test]
    fn hsetp2_h_and() {
        check(
            Hsetp2,
            RegCbuf,
            A | field(7, 54)
                | field(0xfc, 38)
                | field(3, 60)
                | bit(71)
                | field(4, 76)
                | field(1, 81)
                | field(7, 84)
                | field(1, 87)
                | bit(90)
                | field(1, 69),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetCbufU32 #0x7, #0xfc",
                "%5 = UnpackFloat2x16 %4",
                "%6 = CompositeExtractF16x2 %5, #0x1",
                "%7 = GetPred P1",
                "%8 = LogicalNot %7",
                "%9 = FPOrdGreaterThan16 %2, %6 (flags 0x30)",
                "%10 = FPOrdGreaterThan16 %3, %6 (flags 0x30)",
                "%11 = LogicalOr %9, %8",
                "%12 = LogicalOr %10, %8",
                "%13 = LogicalAnd %11, %12",
                "%14 = LogicalNot %13",
                "SetPred P1, %13",
            ],
        );
    }

    #[test]
    fn hmnmx2_min() {
        check(
            Hmnmx2,
            RegReg,
            D | A | B | field(2, 87) | field(3, 74),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x1",
                "%3 = GetRegister R6",
                "%4 = UnpackFloat2x16 %3",
                "%5 = CompositeExtractF16x2 %4, #0x0",
                "%6 = CompositeExtractF16x2 %4, #0x1",
                "%7 = ConvertF32F16 %2",
                "%8 = ConvertF32F16 %2",
                "%9 = ConvertF32F16 %5",
                "%10 = ConvertF32F16 %6",
                "%11 = GetPred P2",
                "%12 = FPMax32 %7, %9 (flags 0x30)",
                "%13 = FPMin32 %7, %9 (flags 0x30)",
                "%14 = SelectF32 %11, %13, %12",
                "%15 = FPMax32 %8, %10 (flags 0x30)",
                "%16 = FPMin32 %8, %10 (flags 0x30)",
                "%17 = SelectF32 %11, %16, %15",
                "%18 = ConvertF16F32 %14",
                "%19 = ConvertF16F32 %17",
                "%20 = CompositeConstructF16x2 %18, %19",
                "%21 = PackFloat2x16 %20",
                "SetRegister R2, %21",
            ],
        );
    }

    #[test]
    fn dadd_registers() {
        check(
            Dadd,
            RegReg,
            D | A | C | bit(75) | field(2, 78),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R8",
                "%5 = GetRegister R9",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPNeg64 %7",
                "%9 = FPAdd64 %3, %8 (flags 0x37)",
                "%10 = UnpackDouble2x32 %9",
                "%11 = CompositeExtractU32x2 %10, #0x0",
                "SetRegister R2, %11",
                "%12 = CompositeExtractU32x2 %10, #0x1",
                "SetRegister R3, %12",
            ],
        );
    }

    #[test]
    fn dadd_immediate_high_word() {
        check(
            Dadd,
            RegImm,
            D | A | field(0x3ff0_0000, 32),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPAdd64 %3, #1.0d (flags 0x33)",
                "%5 = UnpackDouble2x32 %4",
                "%6 = CompositeExtractU32x2 %5, #0x0",
                "SetRegister R2, %6",
                "%7 = CompositeExtractU32x2 %5, #0x1",
                "SetRegister R3, %7",
            ],
        );
    }

    #[test]
    fn dadd_odd_pair() {
        rejects(Dadd, RegReg, D | (5 << 24) | C);
    }

    #[test]
    fn dfma_cbuf() {
        check(
            Dfma,
            RegCbuf,
            D | A | C | field(3, 54) | field(0x40, 38) | bit(72),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPNeg64 %3",
                "%5 = GetRegister R8",
                "%6 = GetRegister R9",
                "%7 = CompositeConstructU32x2 %5, %6",
                "%8 = PackDouble2x32 %7",
                "%9 = GetCbufU32 #0x3, #0x40",
                "%10 = GetCbufU32 #0x3, #0x44",
                "%11 = CompositeConstructU32x2 %9, %10",
                "%12 = PackDouble2x32 %11",
                "%13 = FPFma64 %4, %8, %12 (flags 0x33)",
                "%14 = UnpackDouble2x32 %13",
                "%15 = CompositeExtractU32x2 %14, #0x0",
                "SetRegister R2, %15",
                "%16 = CompositeExtractU32x2 %14, #0x1",
                "SetRegister R3, %16",
            ],
        );
    }

    #[test]
    fn dmul_registers() {
        check(
            Dmul,
            RegReg,
            D | A | B | bit(63) | field(3, 78),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R6",
                "%5 = GetRegister R7",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPNeg64 %7",
                "%9 = FPMul64 %3, %8 (flags 0x39)",
                "%10 = UnpackDouble2x32 %9",
                "%11 = CompositeExtractU32x2 %10, #0x0",
                "SetRegister R2, %11",
                "%12 = CompositeExtractU32x2 %10, #0x1",
                "SetRegister R3, %12",
            ],
        );
    }

    #[test]
    fn dsetp_ge() {
        check(
            Dsetp,
            RegReg,
            A | B | field(6, 76) | field(2, 81) | field(3, 84) | field(7, 87),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R6",
                "%5 = GetRegister R7",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPOrdGreaterThanEqual64 %3, %7",
                "%9 = LogicalAnd %8, #true",
                "%10 = LogicalNot %8",
                "%11 = LogicalAnd %10, #true",
                "SetPred P2, %9",
                "SetPred P3, %11",
            ],
        );
    }

    #[test]
    fn f2f_f32_to_f16() {
        check(
            F2f,
            RegReg,
            D | B | field(1, 75) | field(2, 84) | field(3, 78),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = ConvertF16F32 %1 (flags 0x38)",
                "%3 = ConvertF16F32 #0.0f",
                "%4 = CompositeConstructF16x2 %2, %3",
                "%5 = PackFloat2x16 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn f2f_f16_high_to_f32() {
        check(
            F2f,
            RegReg,
            D | B | field(2, 75) | field(1, 84) | field(1, 60) | bit(62) | bit(80),
            &[
                "%0 = GetRegister R6",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x1",
                "%3 = FPAbs16 %2",
                "%4 = ConvertF32F16 %3 (flags 0x12)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn f2f_f32_to_f64() {
        check(
            F2f64,
            RegReg,
            D | B | field(3, 75) | field(2, 84),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = ConvertF64F32 %1 (flags 0x32)",
                "%3 = UnpackDouble2x32 %2",
                "%4 = CompositeExtractU32x2 %3, #0x0",
                "SetRegister R2, %4",
                "%5 = CompositeExtractU32x2 %3, #0x1",
                "SetRegister R3, %5",
            ],
        );
    }

    #[test]
    fn f2f_f64_to_f32() {
        check(
            F2f64,
            RegReg,
            D | (6 << 32) | field(2, 75) | field(3, 84) | field(1, 78),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = ConvertF32F64 %3 (flags 0x34)",
                "%5 = BitCastU32F32 %4",
                "SetRegister R2, %5",
            ],
        );
    }

    #[test]
    fn f2f_64_bit_field_in_32_bit_form() {
        rejects(F2f, RegReg, D | B | field(3, 75) | field(2, 84));
    }

    #[test]
    fn f2f_f16_to_f64() {
        rejects(F2f64, RegReg, D | B | field(3, 75) | field(1, 84));
    }

    #[test]
    fn f2i_f32_to_s32_ntz() {
        check(
            F2i,
            RegReg,
            D | B | bit(72) | field(2, 75) | bit(77) | field(3, 78) | field(2, 84) | bit(80),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPTrunc32 %1 (flags 0x11)",
                "%3 = FPClamp32 %2, #-2147483600.0f, #2147483600.0f",
                "%4 = ConvertS32F32 %3",
                "%5 = FPIsNan32 %1",
                "%6 = SelectU32 %5, #0x0, %4",
                "%7 = FPIsNan32 %1",
                "%8 = SelectU32 %7, #0x0, %6",
                "SetRegister R2, %8",
            ],
        );
    }

    #[test]
    fn f2i_f32_to_u64() {
        check(
            F2i64,
            RegReg,
            D | B | field(3, 75) | field(3, 78) | field(2, 84),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPTrunc32 %1 (flags 0x1)",
                "%3 = FPIsNan32 %1",
                "%4 = SelectF32 %3, #0.0f, %2",
                "%5 = FPClamp32 %4, #0.0f, #1.8446743e19f",
                "%6 = ConvertU64F32 %5",
                "%7 = FPOrdGreaterThanEqual32 %1, #1.8446744e19f",
                "%8 = SelectU64 %7, #0xffffffffffffffffu64, %6",
                "%9 = FPIsNan32 %1",
                "%10 = SelectU64 %9, #0x8000000000000000u64, %8",
                "%11 = UnpackUint2x32 %10",
                "%12 = CompositeExtractU32x2 %11, #0x0",
                "SetRegister R2, %12",
                "%13 = CompositeExtractU32x2 %11, #0x1",
                "SetRegister R3, %13",
            ],
        );
    }

    #[test]
    fn f2i_f16_to_u16_floor() {
        check(
            F2i,
            RegReg,
            D | B | field(1, 75) | field(1, 78) | field(1, 84) | field(1, 60),
            &[
                "%0 = GetRegister R6",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x1",
                "%3 = FPFloor16 %2 (flags 0x1)",
                "%4 = ConvertF16F32 #65535.0f",
                "%5 = ConvertF16F32 #0.0f",
                "%6 = FPClamp16 %3, %5, %4",
                "%7 = ConvertU32F16 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn f2i_f64_cbuf_to_s32() {
        check(
            F2i64,
            CbufReg,
            D | field(1, 54) | field(0x10, 38) | bit(72) | field(2, 75) | field(3, 84),
            &[
                "%0 = GetCbufU32 #0x1, #0x10",
                "%1 = GetCbufU32 #0x1, #0x14",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPRoundEven64 %3 (flags 0x1)",
                "%5 = FPClamp64 %4, #-2147483648.0d, #2147483647.0d",
                "%6 = ConvertS32F64 %5",
                "%7 = FPIsNan64 %3",
                "%8 = SelectU32 %7, #0x80000000, %6",
                "SetRegister R2, %8",
            ],
        );
    }

    #[test]
    fn i2f_u32_to_f32_rz() {
        check(
            I2f,
            RegReg,
            D | B | field(2, 75) | field(3, 78) | field(2, 84),
            &[
                "%0 = GetRegister R6",
                "%1 = INotEqual %0, #0x0",
                "%2 = FindUMsb32 %0",
                "%3 = SelectU32 %1, %2, #0x0",
                "%4 = UMax32 %3, #0x17",
                "%5 = ISub32 %4, #0x17",
                "%6 = ShiftRightLogical32 %0, %5",
                "%7 = IAdd32 %3, #0x7f",
                "%8 = ShiftLeftLogical32 %7, #0x17",
                "%9 = BitwiseAnd32 %6, #0x7fffff",
                "%10 = BitwiseOr32 %8, %9",
                "%11 = BitwiseAnd32 %0, #0xffffff",
                "%12 = ConvertF32U32 %11",
                "%13 = UGreaterThan %3, #0x17",
                "%14 = BitCastF32U32 %10",
                "%15 = SelectF32 %13, %14, %12",
                "%16 = BitCastU32F32 %15",
                "SetRegister R2, %16",
            ],
        );
    }

    #[test]
    fn i2f_s16_high_to_f32() {
        check(
            I2f,
            RegReg,
            D | B | bit(74) | field(2, 75) | field(1, 84) | field(1, 60),
            &[
                "%0 = GetRegister R6",
                "%1 = BitFieldSExtract %0, #0x10, #0x10",
                "%2 = ConvertF32S32 %1 (flags 0x2)",
                "%3 = BitCastU32F32 %2",
                "SetRegister R2, %3",
            ],
        );
    }

    #[test]
    fn i2f_u64_to_f32() {
        check(
            I2f64,
            RegReg,
            D | B | field(2, 75) | field(3, 84),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackUint2x32 %2",
                "%4 = UnpackUint2x32 %3",
                "%5 = CompositeExtractU32x2 %4, #0x0",
                "%6 = UnpackUint2x32 %3",
                "%7 = CompositeExtractU32x2 %6, #0x1",
                "%8 = INotEqual %7, #0x0",
                "%9 = FindUMsb32 %7",
                "%10 = IAdd32 %9, #0x20",
                "%11 = INotEqual %5, #0x0",
                "%12 = FindUMsb32 %5",
                "%13 = SelectU32 %11, %12, #0x0",
                "%14 = SelectU32 %8, %10, %13",
                "%15 = UMax32 %14, #0x17",
                "%16 = ISub32 %15, #0x17",
                "%17 = ShiftRightLogical64 %3, %16",
                "%18 = UnpackUint2x32 %17",
                "%19 = CompositeExtractU32x2 %18, #0x0",
                "%20 = IAdd32 %14, #0x7f",
                "%21 = ShiftLeftLogical32 %20, #0x17",
                "%22 = BitwiseAnd32 %19, #0x7fffff",
                "%23 = BitwiseOr32 %21, %22",
                "%24 = UMax32 %16, #0x1",
                "%25 = ISub32 %24, #0x1",
                "%26 = ShiftRightLogical64 %3, %25",
                "%27 = UnpackUint2x32 %26",
                "%28 = CompositeExtractU32x2 %27, #0x0",
                "%29 = BitwiseAnd32 %28, #0x1",
                "%30 = INotEqual %29, #0x0",
                "%31 = ShiftLeftLogical64 %26, %25",
                "%32 = UnpackUint2x32 %3",
                "%33 = UnpackUint2x32 %31",
                "%34 = CompositeExtractU32x2 %32, #0x0",
                "%35 = CompositeExtractU32x2 %33, #0x0",
                "%36 = IEqual %34, %35",
                "%37 = CompositeExtractU32x2 %32, #0x1",
                "%38 = CompositeExtractU32x2 %33, #0x1",
                "%39 = IEqual %37, %38",
                "%40 = LogicalAnd %36, %39",
                "%41 = LogicalNot %40",
                "%42 = BitwiseAnd32 %19, #0x1",
                "%43 = INotEqual %42, #0x0",
                "%44 = LogicalOr %41, %43",
                "%45 = LogicalAnd %30, %44",
                "%46 = SelectU32 %45, #0x1, #0x0",
                "%47 = IAdd32 %23, %46",
                "%48 = BitwiseAnd32 %5, #0xffffff",
                "%49 = ConvertF32U32 %48",
                "%50 = UGreaterThan %14, #0x17",
                "%51 = BitCastF32U32 %47",
                "%52 = SelectF32 %50, %51, %49",
                "%53 = BitCastU32F32 %52",
                "SetRegister R2, %53",
            ],
        );
    }

    #[test]
    fn i2f_s32_to_f64() {
        check(
            I2f64,
            RegReg,
            D | B | bit(74) | field(3, 75) | field(2, 84),
            &[
                "%0 = GetRegister R6",
                "%1 = ConvertF64S32 %0 (flags 0x2)",
                "%2 = UnpackDouble2x32 %1",
                "%3 = CompositeExtractU32x2 %2, #0x0",
                "SetRegister R2, %3",
                "%4 = CompositeExtractU32x2 %2, #0x1",
                "SetRegister R3, %4",
            ],
        );
    }

    #[test]
    fn i2f_immediate() {
        check(
            I2f,
            ImmReg,
            D | field(7, 32) | field(1, 75) | field(2, 84),
            &[
                "%0 = ConvertF16U32 #0x7 (flags 0x2)",
                "%1 = ConvertF16F32 #0.0f",
                "%2 = CompositeConstructF16x2 %0, %1",
                "%3 = PackFloat2x16 %2",
                "SetRegister R2, %3",
            ],
        );
    }

    #[test]
    fn i2f_negated_source() {
        rejects(I2f, RegReg, D | B | bit(63) | field(2, 75) | field(2, 84));
    }

    #[test]
    fn i2f_u16_selector_out_of_range() {
        rejects(
            I2f,
            RegReg,
            D | B | field(2, 60) | field(2, 75) | field(1, 84),
        );
    }

    #[test]
    fn frnd_f32_floor_ftz() {
        check(
            Frnd,
            RegReg,
            D | B | field(2, 75) | field(2, 84) | field(1, 78) | bit(80) | bit(63),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPNeg32 %1",
                "%3 = FPFloor32 %2 (flags 0x10)",
                "%4 = BitCastU32F32 %3",
                "SetRegister R2, %4",
            ],
        );
    }

    #[test]
    fn frnd_f64_ceil() {
        check(
            Frnd64,
            RegReg,
            D | B | field(3, 75) | field(3, 84) | field(2, 78),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPCeil64 %3 (flags 0x30)",
                "%5 = UnpackDouble2x32 %4",
                "%6 = CompositeExtractU32x2 %5, #0x0",
                "SetRegister R2, %6",
                "%7 = CompositeExtractU32x2 %5, #0x1",
                "SetRegister R3, %7",
            ],
        );
    }

    #[test]
    fn frnd_f16_trunc() {
        check(
            Frnd,
            RegReg,
            D | B | field(1, 75) | field(1, 84) | field(3, 78) | field(1, 60),
            &[
                "%0 = GetRegister R6",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x1",
                "%3 = FPTrunc16 %2 (flags 0x30)",
                "%4 = ConvertF16F32 #0.0f",
                "%5 = CompositeConstructF16x2 %3, %4",
                "%6 = PackFloat2x16 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn frnd_mixed_formats() {
        rejects(Frnd, RegReg, D | B | field(1, 75) | field(2, 84));
    }

    #[test]
    fn f2fp_pack_ab() {
        check(
            F2fp,
            RegReg,
            D | A | B | RZ_C | field(3, 79),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = ConvertF16F32 %1 (flags 0x8)",
                "%3 = GetRegister R4",
                "%4 = BitCastF32U32 %3",
                "%5 = ConvertF16F32 %4 (flags 0x8)",
                "%6 = CompositeConstructF16x2 %2, %5",
                "%7 = PackFloat2x16 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn f2fp_merge_c_high() {
        check(
            F2fp,
            RegReg,
            D | A | B | C | bit(78) | bit(72),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = ConvertF16F32 %1 (flags 0x2)",
                "%3 = GetRegister R8",
                "%4 = UnpackFloat2x16 %3",
                "%5 = CompositeExtractF16x2 %4, #0x1",
                "%6 = CompositeConstructF16x2 %2, %5",
                "%7 = PackFloat2x16 %6",
                "SetRegister R2, %7",
            ],
        );
    }

    #[test]
    fn f2fp_unknown_format() {
        rejects(F2fp, RegReg, D | A | B | RZ_C | bit(73));
    }

    #[test]
    fn uf2fp_uniform() {
        check(
            Uf2fp,
            RegReg,
            D | A | field(5, 32) | RZ_C | bit(91) | field(1, 75) | field(2, 84),
            &[
                "%0 = GetRegister UR5",
                "%1 = BitCastF32U32 %0",
                "%2 = ConvertF16F32 %1 (flags 0x2)",
                "%3 = GetRegister UR4",
                "%4 = BitCastF32U32 %3",
                "%5 = ConvertF16F32 %4 (flags 0x2)",
                "%6 = CompositeConstructF16x2 %2, %5",
                "%7 = PackFloat2x16 %6",
                "SetRegister UR2, %7",
            ],
        );
    }

    #[test]
    fn f2ip_u8_pair() {
        check(
            F2ip,
            RegReg,
            D | A | B | RZ_C | bit(74) | field(3, 78),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = FPTrunc32 %1 (flags 0x31)",
                "%3 = FPIsNan32 %1",
                "%4 = SelectF32 %3, #0.0f, %2",
                "%5 = FPClamp32 %4, #0.0f, #255.0f",
                "%6 = ConvertU32F32 %5",
                "%7 = GetRegister R4",
                "%8 = BitCastF32U32 %7",
                "%9 = FPTrunc32 %8 (flags 0x31)",
                "%10 = FPIsNan32 %8",
                "%11 = SelectF32 %10, #0.0f, %9",
                "%12 = FPClamp32 %11, #0.0f, #255.0f",
                "%13 = ConvertU32F32 %12",
                "%14 = ShiftLeftLogical32 %13, #0x8",
                "%15 = BitwiseOr32 %6, %14",
                "SetRegister R2, %15",
            ],
        );
    }

    #[test]
    fn f2ip_with_c() {
        rejects(F2ip, RegReg, D | A | B | C | bit(74));
    }

    #[test]
    fn i2fp_s32_rp() {
        check(
            I2fp,
            RegReg,
            D | B | bit(74) | field(2, 75) | field(2, 78) | field(2, 84),
            &[
                "%0 = GetRegister R6",
                "%1 = ConvertF32S32 %0",
                "%2 = FPAbs32 %1",
                "%3 = ConvertU32F32 %2",
                "%4 = SLessThan %0, #0x0",
                "%5 = INeg32 %0",
                "%6 = SelectU32 %4, %5, %0",
                "%7 = BitCastU32F32 %1",
                "%8 = ULessThan %3, %6",
                "%9 = IAdd32 %7, #0x1",
                "%10 = SelectU32 %8, %9, %7",
                "%11 = UGreaterThan %3, %6",
                "%12 = ISub32 %7, #0x1",
                "%13 = SelectU32 %11, %12, %7",
                "%14 = SelectU32 %4, %13, %10",
                "%15 = BitCastF32U32 %14",
                "%16 = BitCastU32F32 %15",
                "SetRegister R2, %16",
            ],
        );
    }

    #[test]
    fn i2fp_cbuf() {
        check(
            I2fp,
            CbufReg,
            D | field(1, 54) | field(4, 38) | field(2, 75) | field(2, 84),
            &[
                "%0 = GetCbufU32 #0x1, #0x4",
                "%1 = ConvertF32U32 %0 (flags 0x2)",
                "%2 = BitCastU32F32 %1",
                "SetRegister R2, %2",
            ],
        );
    }

    #[test]
    fn dsetp_immediate_in_c() {
        check(
            Dsetp,
            RegImm,
            A | field(0x4000_0000, 32)
                | field(1, 76)
                | field(2, 74)
                | field(1, 81)
                | field(7, 84)
                | field(3, 87),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPOrdLessThan64 %3, #2.0d",
                "%5 = GetPred P3",
                "%6 = LogicalXor %4, %5",
                "%7 = LogicalNot %4",
                "%8 = LogicalXor %7, %5",
                "SetPred P1, %6",
            ],
        );
    }

    #[test]
    fn hsetp2_lanes() {
        check(
            Hsetp2,
            RegReg,
            A | B | field(3, 76) | field(1, 81) | field(2, 84) | field(7, 87) | field(2, 60),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = GetRegister R6",
                "%5 = UnpackFloat2x16 %4",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = FPOrdLessThanEqual16 %2, %6 (flags 0x30)",
                "%8 = FPOrdLessThanEqual16 %3, %6 (flags 0x30)",
                "%9 = LogicalAnd %7, #true",
                "%10 = LogicalAnd %8, #true",
                "SetPred P1, %9",
                "SetPred P2, %10",
            ],
        );
    }

    #[test]
    fn hsetp2_bf16() {
        rejects(Hsetp2, RegReg, A | B | bit(65));
    }

    #[test]
    fn ffma_immediate_c() {
        check(
            Ffma,
            RegImm,
            D | A | C | field(0x3f80_0000, 32) | bit(77),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R8",
                "%3 = BitCastF32U32 %2",
                "%4 = FPFma32 %1, %3, #1.0f (flags 0x33)",
                "%5 = FPSaturate32 %4",
                "%6 = BitCastU32F32 %5",
                "SetRegister R2, %6",
            ],
        );
    }

    /// A fragment program mixing single, half and double precision with the conversions
    /// translates, optimizes and emits a module spirv-val accepts.
    #[test]
    fn fragment_program_compiles_to_valid_spirv() {
        let r = |n: u128, offset: u32| n << offset;
        let program = [
            encode(Fadd, CbufReg, r(255, 24)),
            encode(Ffma, RegImm, r(1, 16) | field(0x3f00_0000, 32)),
            encode(Mufu, RegReg, r(2, 16) | r(1, 32) | bit(62) | field(5, 74)),
            encode(F2fp, RegReg, r(3, 16) | r(1, 24) | r(2, 32) | RZ_C),
            encode(
                Hfma2,
                RegReg,
                r(3, 16) | r(3, 24) | r(3, 32) | r(3, 64) | bit(79) | field(7, 87),
            ),
            encode(Hadd2, RegReg, r(3, 16) | r(3, 24) | r(255, 32) | bit(78)),
            encode(
                F2f64,
                RegReg,
                r(4, 16) | r(1, 32) | field(3, 75) | field(2, 84),
            ),
            encode(Dfma, RegReg, r(4, 16) | r(4, 24) | r(4, 32) | r(4, 64)),
            encode(
                F2f64,
                RegReg,
                r(1, 16) | r(4, 32) | field(2, 75) | field(3, 84),
            ),
            encode(
                Fsetp,
                RegReg,
                r(1, 32) | field(7, 84) | field(1, 76) | field(7, 87),
            ),
            encode(Fsel, RegReg, r(2, 32)),
            encode(
                F2i,
                RegReg,
                r(2, 16)
                    | r(2, 32)
                    | bit(72)
                    | field(2, 75)
                    | bit(77)
                    | field(3, 78)
                    | field(2, 84),
            ),
            encode(
                I2fp,
                RegReg,
                r(2, 16) | r(2, 32) | bit(74) | field(2, 75) | field(2, 84),
            ),
            encode(Exit, Fixed, field(7, 87)),
        ];
        let mut env = TestEnv::new(&program);
        env.stage = ShaderStage::Fragment;
        env.header[18] = 0xf;
        env.cbuf = vec![0x3f80_0000];
        let host = HostTranslateInfo {
            support_float64: true,
            support_float16: true,
            support_int64: true,
            ..HostTranslateInfo::default()
        };
        let profile = Profile::default();
        let options = TranslateOptions {
            profile: &profile,
            host: &host,
            resolution_scale: ResolutionScale::X1,
        };
        let translated = crate::translate::translate(&env, options).expect("translate");
        assert!(translated.info.stores_frag_color[0]);
        crate::backend::spirv::tests::emit(
            "ampere_float",
            &profile,
            &RuntimeInfo::default(),
            &translated,
        );
    }
}
