//! HMUL2.

use super::floating_point_fused_multiply_add::any_zero;
use super::half_floating_point_add::{
    Merge, Swizzle, abs_neg_pair, demote, extract, half_precision, merge_result, packed_imm,
    promote, zero_like,
};
use super::{TResult, Translator};
use crate::ir::{FpControl, FpRounding, Value};

struct Hmul2 {
    merge: Merge,
    sat: bool,
    abs_a: bool,
    neg_a: bool,
    swizzle_a: Swizzle,
    abs_b: bool,
    neg_b: bool,
    swizzle_b: Swizzle,
    precision: u64,
}

fn hmul2(t: &mut Translator<'_>, src_b: Value, m: Hmul2) -> TResult {
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
    let (fmz_mode, fmz) = half_precision(m.precision);
    let control = FpControl {
        no_contraction: true,
        rounding: FpRounding::DontCare,
        fmz_mode,
    };
    let mut lhs = t.ir.fp_mul(a.0, b.0, control);
    let mut rhs = t.ir.fp_mul(a.1, b.1, control);
    if fmz && !m.sat {
        let zero = zero_like(&mut t.ir, lhs);
        let lhs_zero = any_zero(&mut t.ir, a.0, b.0, zero);
        lhs = t.ir.select(lhs_zero, zero, lhs);
        let rhs_zero = any_zero(&mut t.ir, a.1, b.1, zero);
        rhs = t.ir.select(rhs_zero, zero, rhs);
    }
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

fn hmul2_common(
    t: &mut Translator<'_>,
    src_b: Value,
    sat: bool,
    (abs_a, neg_a): (bool, bool),
    (abs_b, neg_b): (bool, bool),
    swizzle_b: Swizzle,
) -> TResult {
    let i = t.insn;
    let m = Hmul2 {
        merge: Merge::from_bits(i.bits(49, 2)),
        sat,
        abs_a,
        neg_a,
        swizzle_a: Swizzle::from_bits(i.bits(47, 2)),
        abs_b,
        neg_b,
        swizzle_b,
        precision: i.bits(39, 2),
    };
    hmul2(t, src_b, m)
}

pub(super) fn hmul2_reg(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = t.get_reg20();
    let swizzle_b = Swizzle::from_bits(i.bits(28, 2));
    hmul2_common(
        t,
        src_b,
        i.bit(32),
        (i.bit(44), false),
        (i.bit(30), i.bit(31)),
        swizzle_b,
    )
}

pub(super) fn hmul2_cbuf(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = t.get_cbuf()?;
    hmul2_common(
        t,
        src_b,
        i.bit(52),
        (i.bit(44), i.bit(43)),
        (i.bit(54), false),
        Swizzle::F32,
    )
}

pub(super) fn hmul2_imm(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = packed_imm(t);
    hmul2_common(
        t,
        src_b,
        i.bit(52),
        (i.bit(44), i.bit(43)),
        (false, false),
        Swizzle::H1H0,
    )
}

pub(super) fn hmul2_32i(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hmul2 {
        merge: Merge::H1H0,
        sat: i.bit(52),
        abs_a: false,
        neg_a: false,
        swizzle_a: Swizzle::from_bits(i.bits(53, 2)),
        abs_b: false,
        neg_b: false,
        swizzle_b: Swizzle::H1H0,
        precision: i.bits(55, 2),
    };
    let src_b = t.get_imm32();
    hmul2(t, src_b, m)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::check;
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn hmul2_reg_ftz() {
        check(
            encode(
                Opcode::Hmul2Reg,
                2 | 4 << 8 | 6 << 20 | 1 << 39 | 1 << 44 | 1 << 31,
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
                "%8 = FPAbs16 %3",
                "%9 = FPAbs16 %4",
                "%10 = FPNeg16 %6",
                "%11 = FPNeg16 %7",
                "%12 = FPMul16 %8, %10 (flags 0x11)",
                "%13 = FPMul16 %9, %11 (flags 0x11)",
                "%14 = CompositeConstructF16x2 %12, %13",
                "%15 = PackFloat2x16 %14",
                "SetRegister R2, %15",
            ],
        );
    }

    #[test]
    fn hmul2_reg_fmz_emulation() {
        check(
            encode(Opcode::Hmul2Reg, 2 | 4 << 8 | 6 << 20 | 2 << 39),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R4",
                "%2 = UnpackFloat2x16 %1",
                "%3 = CompositeExtractF16x2 %2, #0x0",
                "%4 = CompositeExtractF16x2 %2, #0x1",
                "%5 = UnpackFloat2x16 %0",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = FPMul16 %3, %6 (flags 0x21)",
                "%9 = FPMul16 %4, %7 (flags 0x21)",
                "%10 = ConvertF16F32 #0.0f",
                "%11 = FPOrdEqual16 %3, %10",
                "%12 = FPOrdEqual16 %6, %10",
                "%13 = LogicalOr %11, %12",
                "%14 = SelectF16 %13, %10, %8",
                "%15 = FPOrdEqual16 %4, %10",
                "%16 = FPOrdEqual16 %7, %10",
                "%17 = LogicalOr %15, %16",
                "%18 = SelectF16 %17, %10, %9",
                "%19 = CompositeConstructF16x2 %14, %18",
                "%20 = PackFloat2x16 %19",
                "SetRegister R2, %20",
            ],
        );
    }

    #[test]
    fn hmul2_cbuf_promoted_fmz() {
        check(
            encode(
                Opcode::Hmul2Cbuf,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 2 << 39 | 1 << 43,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = UnpackFloat2x16 %1",
                "%3 = CompositeExtractF16x2 %2, #0x0",
                "%4 = CompositeExtractF16x2 %2, #0x1",
                "%5 = BitCastF32U32 %0",
                "%6 = ConvertF32F16 %3",
                "%7 = ConvertF32F16 %4",
                "%8 = FPNeg32 %6",
                "%9 = FPNeg32 %7",
                "%10 = FPMul32 %8, %5 (flags 0x21)",
                "%11 = FPMul32 %9, %5 (flags 0x21)",
                "%12 = FPOrdEqual32 %8, #0.0f",
                "%13 = FPOrdEqual32 %5, #0.0f",
                "%14 = LogicalOr %12, %13",
                "%15 = SelectF32 %14, #0.0f, %10",
                "%16 = FPOrdEqual32 %9, #0.0f",
                "%17 = FPOrdEqual32 %5, #0.0f",
                "%18 = LogicalOr %16, %17",
                "%19 = SelectF32 %18, #0.0f, %11",
                "%20 = ConvertF16F32 %15",
                "%21 = ConvertF16F32 %19",
                "%22 = CompositeConstructF16x2 %20, %21",
                "%23 = PackFloat2x16 %22",
                "SetRegister R2, %23",
            ],
        );
    }

    #[test]
    fn hmul2_imm_merge_h0() {
        check(
            encode(Opcode::Hmul2Imm, 2 | 4 << 8 | 0x1e0 << 20 | 2 << 49),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = UnpackFloat2x16 #0x7800",
                "%5 = CompositeExtractF16x2 %4, #0x0",
                "%6 = CompositeExtractF16x2 %4, #0x1",
                "%7 = FPMul16 %2, %5 (flags 0x31)",
                "%8 = FPMul16 %3, %6 (flags 0x31)",
                "%9 = GetRegister R2",
                "%10 = UnpackFloat2x16 %9",
                "%11 = CompositeInsertF16x2 %10, %7, #0x0",
                "%12 = PackFloat2x16 %11",
                "SetRegister R2, %12",
            ],
        );
    }

    #[test]
    fn hmul2_32i_sat() {
        check(
            encode(
                Opcode::Hmul2_32i,
                2 | 4 << 8 | 0x3c00_3c00 << 20 | 1 << 52 | 2 << 55,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = UnpackFloat2x16 #0x3c003c00",
                "%5 = CompositeExtractF16x2 %4, #0x0",
                "%6 = CompositeExtractF16x2 %4, #0x1",
                "%7 = FPMul16 %2, %5 (flags 0x21)",
                "%8 = FPMul16 %3, %6 (flags 0x21)",
                "%9 = FPSaturate16 %7",
                "%10 = FPSaturate16 %8",
                "%11 = CompositeConstructF16x2 %9, %10",
                "%12 = PackFloat2x16 %11",
                "SetRegister R2, %12",
            ],
        );
    }
}
