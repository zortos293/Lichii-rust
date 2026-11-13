//! HADD2, and the packed-half helpers every HADD2/HMUL2/HFMA2/HSET2/HSETP2 form shares.

use super::floating_point_add::ftz_mode;
use super::{TResult, Translator};
use crate::ir::{Emitter, FmzMode, FpControl, FpRounding, Reg, Type, Value};

/// How a 32-bit operand splits into the two lanes of a packed-half instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Swizzle {
    H1H0,
    /// The whole register is one F32, used by both lanes.
    F32,
    H0H0,
    H1H1,
}

impl Swizzle {
    pub(crate) const fn from_bits(bits: u64) -> Swizzle {
        match bits & 3 {
            0 => Swizzle::H1H0,
            1 => Swizzle::F32,
            2 => Swizzle::H0H0,
            _ => Swizzle::H1H1,
        }
    }
}

/// How the two lane results are written to the destination register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Merge {
    H1H0,
    F32,
    MrgH0,
    MrgH1,
}

impl Merge {
    pub(super) const fn from_bits(bits: u64) -> Merge {
        match bits & 3 {
            0 => Merge::H1H0,
            1 => Merge::F32,
            2 => Merge::MrgH0,
            _ => Merge::MrgH1,
        }
    }
}

/// The 2-bit precision field of HMUL2 and HFMA2: the denormal mode, and whether FMZ (D3D9
/// `x * 0 == 0`) has to be emulated.
pub(crate) const fn half_precision(bits: u64) -> (FmzMode, bool) {
    match bits & 3 {
        0 => (FmzMode::None, false),
        1 => (FmzMode::Ftz, false),
        2 => (FmzMode::Fmz, true),
        _ => (FmzMode::DontCare, false),
    }
}

/// The lanes of `value`: two F16 values, or one F32 value twice.
pub(crate) fn extract(ir: &mut Emitter<'_>, value: Value, swizzle: Swizzle) -> (Value, Value) {
    match swizzle {
        Swizzle::H1H0 => {
            let vector = ir.unpack_float_2x16(value);
            (
                ir.composite_extract(vector, 0),
                ir.composite_extract(vector, 1),
            )
        }
        Swizzle::H0H0 | Swizzle::H1H1 => {
            let vector = ir.unpack_float_2x16(value);
            let scalar = ir.composite_extract(vector, u32::from(swizzle == Swizzle::H1H1));
            (scalar, scalar)
        }
        Swizzle::F32 => {
            let scalar = ir.bitcast_f32(value);
            (scalar, scalar)
        }
    }
}

pub(super) fn is_f16(ir: &Emitter<'_>, v: Value) -> bool {
    ir.ty(v) == Type::F16
}

/// Widens an F16 lane pair to F32; F32 pairs pass through.
pub(crate) fn promote(ir: &mut Emitter<'_>, (lhs, rhs): (Value, Value)) -> (Value, Value) {
    if !is_f16(ir, lhs) {
        return (lhs, rhs);
    }
    let lhs = ir.fp_convert(32, lhs, FpControl::default());
    let rhs = ir.fp_convert(32, rhs, FpControl::default());
    (lhs, rhs)
}

pub(crate) fn abs_neg_pair(
    ir: &mut Emitter<'_>,
    (lhs, rhs): (Value, Value),
    abs: bool,
    neg: bool,
) -> (Value, Value) {
    (ir.fp_abs_neg(lhs, abs, neg), ir.fp_abs_neg(rhs, abs, neg))
}

/// A zero of the same float type as `like`, for the FMZ emulation.
///
/// The C++ always compares against an F32 zero, which fails on unpromoted F16 lanes.
pub(crate) fn zero_like(ir: &mut Emitter<'_>, like: Value) -> Value {
    if is_f16(ir, like) {
        ir.fp_convert(16, Value::F32(0.0), FpControl::default())
    } else {
        Value::F32(0.0)
    }
}

/// Narrows promoted F32 lane results back to F16.
pub(crate) fn demote(ir: &mut Emitter<'_>, (lhs, rhs): (Value, Value)) -> (Value, Value) {
    let lhs = ir.fp_convert(16, lhs, FpControl::default());
    let rhs = ir.fp_convert(16, rhs, FpControl::default());
    (lhs, rhs)
}

/// The destination value of a packed-half instruction.
///
/// `H1H0` narrows F32 lanes (both operands F32) to F16 so the pack stays well typed; the C++
/// packs the F32 pair as is.
pub(super) fn merge_result(
    t: &mut Translator<'_>,
    dest: Reg,
    (lhs, rhs): (Value, Value),
    merge: Merge,
) -> Value {
    match merge {
        Merge::H1H0 => {
            let (lhs, rhs) = demote(&mut t.ir, (lhs, rhs));
            let vector = t.ir.composite_construct(&[lhs, rhs]);
            t.ir.pack_float_2x16(vector)
        }
        Merge::F32 => {
            let value = t.ir.fp_convert(32, lhs, FpControl::default());
            t.ir.bitcast_u32(value)
        }
        Merge::MrgH0 | Merge::MrgH1 => {
            let is_h0 = merge == Merge::MrgH0;
            let old = t.ir.get_reg(dest);
            let vector = t.ir.unpack_float_2x16(old);
            let insert =
                t.ir.fp_convert(16, if is_h0 { lhs } else { rhs }, FpControl::default());
            let vector = t.ir.composite_insert(vector, insert, u32::from(!is_h0));
            t.ir.pack_float_2x16(vector)
        }
    }
}

/// The packed-half immediate of the (imm) forms: two 9-bit halves with their own sign bits.
pub(super) fn packed_imm(t: &Translator<'_>) -> Value {
    let i = t.insn;
    let low = (i.bits(20, 9) as u32) << 6;
    let neg_low = u32::from(i.bit(29)) << 15;
    let high = (i.bits(30, 9) as u32) << 22;
    let neg_high = u32::from(i.bit(56)) << 31;
    Value::U32(low | neg_low | high | neg_high)
}

struct Hadd2 {
    merge: Merge,
    ftz: bool,
    sat: bool,
    abs_a: bool,
    neg_a: bool,
    swizzle_a: Swizzle,
    abs_b: bool,
    neg_b: bool,
    swizzle_b: Swizzle,
}

fn hadd2(t: &mut Translator<'_>, src_b: Value, m: Hadd2) -> TResult {
    let dest = t.insn.dest_reg();
    let src_a = t.x(t.insn.src_a_reg());
    let mut a = extract(&mut t.ir, src_a, m.swizzle_a);
    let mut b = extract(&mut t.ir, src_b, m.swizzle_b);
    let promotion = t.ir.ty(a.0) != t.ir.ty(b.0);
    if promotion {
        a = promote(&mut t.ir, a);
        b = promote(&mut t.ir, b);
    }
    let a = abs_neg_pair(&mut t.ir, a, m.abs_a, m.neg_a);
    let b = abs_neg_pair(&mut t.ir, b, m.abs_b, m.neg_b);
    let control = FpControl {
        no_contraction: true,
        rounding: FpRounding::DontCare,
        fmz_mode: ftz_mode(m.ftz),
    };
    let mut lhs = t.ir.fp_add(a.0, b.0, control);
    let mut rhs = t.ir.fp_add(a.1, b.1, control);
    if m.sat {
        lhs = t.ir.fp_saturate(lhs);
        rhs = t.ir.fp_saturate(rhs);
    }
    let mut result = (lhs, rhs);
    if promotion {
        result = demote(&mut t.ir, result);
    }
    let value = merge_result(t, dest, result, m.merge);
    t.set_x(dest, value);
    Ok(())
}

fn hadd2_common(
    t: &mut Translator<'_>,
    src_b: Value,
    sat: bool,
    (abs_b, neg_b): (bool, bool),
    swizzle_b: Swizzle,
) -> TResult {
    let i = t.insn;
    let m = Hadd2 {
        merge: Merge::from_bits(i.bits(49, 2)),
        ftz: i.bit(39),
        sat,
        abs_a: i.bit(44),
        neg_a: i.bit(43),
        swizzle_a: Swizzle::from_bits(i.bits(47, 2)),
        abs_b,
        neg_b,
        swizzle_b,
    };
    hadd2(t, src_b, m)
}

pub(super) fn hadd2_reg(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = t.get_reg20();
    let swizzle_b = Swizzle::from_bits(i.bits(28, 2));
    hadd2_common(t, src_b, i.bit(32), (i.bit(30), i.bit(31)), swizzle_b)
}

pub(super) fn hadd2_cbuf(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = t.get_cbuf()?;
    hadd2_common(t, src_b, i.bit(52), (i.bit(54), i.bit(56)), Swizzle::F32)
}

pub(super) fn hadd2_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = packed_imm(t);
    hadd2_common(t, src_b, t.insn.bit(52), (false, false), Swizzle::H1H0)
}

pub(super) fn hadd2_32i(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hadd2 {
        merge: Merge::H1H0,
        ftz: i.bit(55),
        sat: i.bit(52),
        abs_a: false,
        neg_a: i.bit(56),
        swizzle_a: Swizzle::from_bits(i.bits(53, 2)),
        abs_b: false,
        neg_b: false,
        swizzle_b: Swizzle::H1H0,
    };
    let src_b = t.get_imm32();
    hadd2(t, src_b, m)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::check;
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn hadd2_reg_h1_h0() {
        check(
            encode(
                Opcode::Hadd2Reg,
                2 | 4 << 8 | 6 << 20 | 1 << 39 | 1 << 32 | 1 << 43,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R4",
                "%2 = UnpackFloat2x16 %1",
                "%3 = CompositeExtractF16x2 %2, #0x0",
                "%4 = CompositeExtractF16x2 %2, #0x1",
                "%5 = UnpackFloat2x16 %0",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = FPNeg16 %3",
                "%9 = FPNeg16 %4",
                "%10 = FPAdd16 %8, %6 (flags 0x11)",
                "%11 = FPAdd16 %9, %7 (flags 0x11)",
                "%12 = FPSaturate16 %10",
                "%13 = FPSaturate16 %11",
                "%14 = CompositeConstructF16x2 %12, %13",
                "%15 = PackFloat2x16 %14",
                "SetRegister R2, %15",
            ],
        );
    }

    #[test]
    fn hadd2_reg_h0_h0_merge_h1() {
        check(
            encode(
                Opcode::Hadd2Reg,
                2 | 4 << 8 | 6 << 20 | 2 << 28 | 1 << 30 | 3 << 49,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R4",
                "%2 = UnpackFloat2x16 %1",
                "%3 = CompositeExtractF16x2 %2, #0x0",
                "%4 = CompositeExtractF16x2 %2, #0x1",
                "%5 = UnpackFloat2x16 %0",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = FPAbs16 %6",
                "%8 = FPAbs16 %6",
                "%9 = FPAdd16 %3, %7 (flags 0x31)",
                "%10 = FPAdd16 %4, %8 (flags 0x31)",
                "%11 = GetRegister R2",
                "%12 = UnpackFloat2x16 %11",
                "%13 = CompositeInsertF16x2 %12, %10, #0x1",
                "%14 = PackFloat2x16 %13",
                "SetRegister R2, %14",
            ],
        );
    }

    #[test]
    fn hadd2_cbuf_promotes_to_f32() {
        check(
            encode(Opcode::Hadd2Cbuf, 2 | 4 << 8 | 1 << 34 | 3 << 20 | 1 << 56),
            &[
                "%0 = GetCbufU32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = UnpackFloat2x16 %1",
                "%3 = CompositeExtractF16x2 %2, #0x0",
                "%4 = CompositeExtractF16x2 %2, #0x1",
                "%5 = BitCastF32U32 %0",
                "%6 = ConvertF32F16 %3",
                "%7 = ConvertF32F16 %4",
                "%8 = FPNeg32 %5",
                "%9 = FPNeg32 %5",
                "%10 = FPAdd32 %6, %8 (flags 0x31)",
                "%11 = FPAdd32 %7, %9 (flags 0x31)",
                "%12 = ConvertF16F32 %10",
                "%13 = ConvertF16F32 %11",
                "%14 = CompositeConstructF16x2 %12, %13",
                "%15 = PackFloat2x16 %14",
                "SetRegister R2, %15",
            ],
        );
    }

    #[test]
    fn hadd2_cbuf_f32_merge_f32() {
        check(
            encode(
                Opcode::Hadd2Cbuf,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 1 << 47 | 1 << 49,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = BitCastF32U32 %1",
                "%3 = BitCastF32U32 %0",
                "%4 = FPAdd32 %2, %3 (flags 0x31)",
                "%5 = FPAdd32 %2, %3 (flags 0x31)",
                "%6 = BitCastU32F32 %4",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn hadd2_imm_packs_halves() {
        check(
            encode(
                Opcode::Hadd2Imm,
                2 | 4 << 8 | 0xf0 << 20 | 1 << 29 | 0xf0 << 30,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = UnpackFloat2x16 #0x3c00bc00",
                "%5 = CompositeExtractF16x2 %4, #0x0",
                "%6 = CompositeExtractF16x2 %4, #0x1",
                "%7 = FPAdd16 %2, %5 (flags 0x31)",
                "%8 = FPAdd16 %3, %6 (flags 0x31)",
                "%9 = CompositeConstructF16x2 %7, %8",
                "%10 = PackFloat2x16 %9",
                "SetRegister R2, %10",
            ],
        );
    }

    #[test]
    fn hadd2_32i() {
        check(
            encode(
                Opcode::Hadd2_32i,
                2 | 4 << 8 | 0x3c00_3c00 << 20 | 1 << 55 | 1 << 56 | 3 << 53,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x1",
                "%3 = UnpackFloat2x16 #0x3c003c00",
                "%4 = CompositeExtractF16x2 %3, #0x0",
                "%5 = CompositeExtractF16x2 %3, #0x1",
                "%6 = FPNeg16 %2",
                "%7 = FPNeg16 %2",
                "%8 = FPAdd16 %6, %4 (flags 0x11)",
                "%9 = FPAdd16 %7, %5 (flags 0x11)",
                "%10 = CompositeConstructF16x2 %8, %9",
                "%11 = PackFloat2x16 %10",
                "SetRegister R2, %11",
            ],
        );
    }
}
