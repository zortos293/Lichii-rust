//! HFMA2.

use super::floating_point_fused_multiply_add::any_zero;
use super::half_floating_point_add::{
    Merge, Swizzle, demote, extract, half_precision, merge_result, packed_imm, promote, zero_like,
};
use super::{TResult, Translator};
use crate::ir::{FpControl, FpRounding, Value};

struct Hfma2 {
    merge: Merge,
    swizzle_a: Swizzle,
    neg_b: bool,
    neg_c: bool,
    swizzle_b: Swizzle,
    swizzle_c: Swizzle,
    sat: bool,
    precision: u64,
}

fn hfma2(t: &mut Translator<'_>, src_b: Value, src_c: Value, m: Hfma2) -> TResult {
    let dest = t.insn.dest_reg();
    let src_a = t.x(t.insn.src_a_reg());
    let mut a = extract(&mut t.ir, src_a, m.swizzle_a);
    let mut b = extract(&mut t.ir, src_b, m.swizzle_b);
    let mut c = extract(&mut t.ir, src_c, m.swizzle_c);
    let ty_a = t.ir.ty(a.0);
    let promotion = ty_a != t.ir.ty(b.0) || ty_a != t.ir.ty(c.0);
    if promotion {
        a = promote(&mut t.ir, a);
        b = promote(&mut t.ir, b);
        c = promote(&mut t.ir, c);
    }
    let b = (
        t.ir.fp_abs_neg(b.0, false, m.neg_b),
        t.ir.fp_abs_neg(b.1, false, m.neg_b),
    );
    let c = (
        t.ir.fp_abs_neg(c.0, false, m.neg_c),
        t.ir.fp_abs_neg(c.1, false, m.neg_c),
    );
    if m.precision & 3 == 3 {
        crate::fallback::note("HFMA2", "precision_3");
    }
    let (fmz_mode, fmz) = half_precision(m.precision);
    let control = FpControl {
        no_contraction: true,
        rounding: FpRounding::DontCare,
        fmz_mode,
    };
    let mut lhs = t.ir.fp_fma(a.0, b.0, c.0, control);
    let mut rhs = t.ir.fp_fma(a.1, b.1, c.1, control);
    if fmz && !m.sat {
        let zero = zero_like(&mut t.ir, a.0);
        let lhs_zero = any_zero(&mut t.ir, a.0, b.0, zero);
        lhs = t.ir.select(lhs_zero, c.0, lhs);
        let rhs_zero = any_zero(&mut t.ir, a.1, b.1, zero);
        rhs = t.ir.select(rhs_zero, c.1, rhs);
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

struct Operands {
    neg_b: bool,
    neg_c: bool,
    swizzle_b: Swizzle,
    swizzle_c: Swizzle,
    sat: bool,
    precision: u64,
}

fn hfma2_common(t: &mut Translator<'_>, src_b: Value, src_c: Value, o: Operands) -> TResult {
    let i = t.insn;
    let m = Hfma2 {
        merge: Merge::from_bits(i.bits(49, 2)),
        swizzle_a: Swizzle::from_bits(i.bits(47, 2)),
        neg_b: o.neg_b,
        neg_c: o.neg_c,
        swizzle_b: o.swizzle_b,
        swizzle_c: o.swizzle_c,
        sat: o.sat,
        precision: o.precision,
    };
    hfma2(t, src_b, src_c, m)
}

pub(super) fn hfma2_reg(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = t.get_reg20();
    let src_c = t.get_reg39();
    let o = Operands {
        neg_b: i.bit(31),
        neg_c: i.bit(30),
        swizzle_b: Swizzle::from_bits(i.bits(28, 2)),
        swizzle_c: Swizzle::from_bits(i.bits(35, 2)),
        sat: i.bit(32),
        precision: i.bits(37, 2),
    };
    hfma2_common(t, src_b, src_c, o)
}

pub(super) fn hfma2_rc(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = t.get_reg39();
    let src_c = t.get_cbuf()?;
    let o = Operands {
        neg_b: i.bit(56),
        neg_c: i.bit(51),
        swizzle_b: Swizzle::from_bits(i.bits(53, 2)),
        swizzle_c: Swizzle::F32,
        sat: i.bit(52),
        precision: i.bits(57, 2),
    };
    hfma2_common(t, src_b, src_c, o)
}

pub(super) fn hfma2_cr(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = t.get_cbuf()?;
    let src_c = t.get_reg39();
    let o = Operands {
        neg_b: i.bit(56),
        neg_c: i.bit(51),
        swizzle_b: Swizzle::F32,
        swizzle_c: Swizzle::from_bits(i.bits(53, 2)),
        sat: i.bit(52),
        precision: i.bits(57, 2),
    };
    hfma2_common(t, src_b, src_c, o)
}

pub(super) fn hfma2_imm(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let src_b = packed_imm(t);
    let src_c = t.get_reg39();
    let o = Operands {
        neg_b: false,
        neg_c: i.bit(51),
        swizzle_b: Swizzle::H1H0,
        swizzle_c: Swizzle::from_bits(i.bits(53, 2)),
        sat: i.bit(52),
        precision: i.bits(57, 2),
    };
    hfma2_common(t, src_b, src_c, o)
}

pub(super) fn hfma2_32i(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hfma2 {
        merge: Merge::H1H0,
        swizzle_a: Swizzle::from_bits(i.bits(53, 2)),
        neg_b: false,
        neg_c: i.bit(52),
        swizzle_b: Swizzle::H1H0,
        swizzle_c: Swizzle::H1H0,
        sat: false,
        precision: i.bits(55, 2),
    };
    let src_b = t.get_imm32();
    let src_c = t.x(i.dest_reg());
    hfma2(t, src_b, src_c, m)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::check;
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn hfma2_reg() {
        check(
            encode(
                Opcode::Hfma2Reg,
                2 | 4 << 8 | 6 << 20 | 8 << 39 | 1 << 31 | 1 << 37,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R8",
                "%2 = GetRegister R4",
                "%3 = UnpackFloat2x16 %2",
                "%4 = CompositeExtractF16x2 %3, #0x0",
                "%5 = CompositeExtractF16x2 %3, #0x1",
                "%6 = UnpackFloat2x16 %0",
                "%7 = CompositeExtractF16x2 %6, #0x0",
                "%8 = CompositeExtractF16x2 %6, #0x1",
                "%9 = UnpackFloat2x16 %1",
                "%10 = CompositeExtractF16x2 %9, #0x0",
                "%11 = CompositeExtractF16x2 %9, #0x1",
                "%12 = FPNeg16 %7",
                "%13 = FPNeg16 %8",
                "%14 = FPFma16 %4, %12, %10 (flags 0x11)",
                "%15 = FPFma16 %5, %13, %11 (flags 0x11)",
                "%16 = CompositeConstructF16x2 %14, %15",
                "%17 = PackFloat2x16 %16",
                "SetRegister R2, %17",
            ],
        );
    }

    #[test]
    fn hfma2_reg_fmz_emulation() {
        check(
            encode(Opcode::Hfma2Reg, 2 | 4 << 8 | 6 << 20 | 8 << 39 | 2 << 37),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R8",
                "%2 = GetRegister R4",
                "%3 = UnpackFloat2x16 %2",
                "%4 = CompositeExtractF16x2 %3, #0x0",
                "%5 = CompositeExtractF16x2 %3, #0x1",
                "%6 = UnpackFloat2x16 %0",
                "%7 = CompositeExtractF16x2 %6, #0x0",
                "%8 = CompositeExtractF16x2 %6, #0x1",
                "%9 = UnpackFloat2x16 %1",
                "%10 = CompositeExtractF16x2 %9, #0x0",
                "%11 = CompositeExtractF16x2 %9, #0x1",
                "%12 = FPFma16 %4, %7, %10 (flags 0x21)",
                "%13 = FPFma16 %5, %8, %11 (flags 0x21)",
                "%14 = ConvertF16F32 #0.0f",
                "%15 = FPOrdEqual16 %4, %14",
                "%16 = FPOrdEqual16 %7, %14",
                "%17 = LogicalOr %15, %16",
                "%18 = SelectF16 %17, %10, %12",
                "%19 = FPOrdEqual16 %5, %14",
                "%20 = FPOrdEqual16 %8, %14",
                "%21 = LogicalOr %19, %20",
                "%22 = SelectF16 %21, %11, %13",
                "%23 = CompositeConstructF16x2 %18, %22",
                "%24 = PackFloat2x16 %23",
                "SetRegister R2, %24",
            ],
        );
    }

    #[test]
    fn hfma2_rc_promotes() {
        check(
            encode(
                Opcode::Hfma2Rc,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 8 << 39 | 1 << 51,
            ),
            &[
                "%0 = GetRegister R8",
                "%1 = GetCbufU32 #0x1, #0xc",
                "%2 = GetRegister R4",
                "%3 = UnpackFloat2x16 %2",
                "%4 = CompositeExtractF16x2 %3, #0x0",
                "%5 = CompositeExtractF16x2 %3, #0x1",
                "%6 = UnpackFloat2x16 %0",
                "%7 = CompositeExtractF16x2 %6, #0x0",
                "%8 = CompositeExtractF16x2 %6, #0x1",
                "%9 = BitCastF32U32 %1",
                "%10 = ConvertF32F16 %4",
                "%11 = ConvertF32F16 %5",
                "%12 = ConvertF32F16 %7",
                "%13 = ConvertF32F16 %8",
                "%14 = FPNeg32 %9",
                "%15 = FPNeg32 %9",
                "%16 = FPFma32 %10, %12, %14 (flags 0x31)",
                "%17 = FPFma32 %11, %13, %15 (flags 0x31)",
                "%18 = ConvertF16F32 %16",
                "%19 = ConvertF16F32 %17",
                "%20 = CompositeConstructF16x2 %18, %19",
                "%21 = PackFloat2x16 %20",
                "SetRegister R2, %21",
            ],
        );
    }

    #[test]
    fn hfma2_cr() {
        check(
            encode(
                Opcode::Hfma2Cr,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 8 << 39 | 1 << 56,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0xc",
                "%1 = GetRegister R8",
                "%2 = GetRegister R4",
                "%3 = UnpackFloat2x16 %2",
                "%4 = CompositeExtractF16x2 %3, #0x0",
                "%5 = CompositeExtractF16x2 %3, #0x1",
                "%6 = BitCastF32U32 %0",
                "%7 = UnpackFloat2x16 %1",
                "%8 = CompositeExtractF16x2 %7, #0x0",
                "%9 = CompositeExtractF16x2 %7, #0x1",
                "%10 = ConvertF32F16 %4",
                "%11 = ConvertF32F16 %5",
                "%12 = ConvertF32F16 %8",
                "%13 = ConvertF32F16 %9",
                "%14 = FPNeg32 %6",
                "%15 = FPNeg32 %6",
                "%16 = FPFma32 %10, %14, %12 (flags 0x31)",
                "%17 = FPFma32 %11, %15, %13 (flags 0x31)",
                "%18 = ConvertF16F32 %16",
                "%19 = ConvertF16F32 %17",
                "%20 = CompositeConstructF16x2 %18, %19",
                "%21 = PackFloat2x16 %20",
                "SetRegister R2, %21",
            ],
        );
    }

    #[test]
    fn hfma2_imm() {
        check(
            encode(
                Opcode::Hfma2Imm,
                2 | 4 << 8 | 0x1e0 << 20 | 0x1e0 << 30 | 8 << 39 | 1 << 52,
            ),
            &[
                "%0 = GetRegister R8",
                "%1 = GetRegister R4",
                "%2 = UnpackFloat2x16 %1",
                "%3 = CompositeExtractF16x2 %2, #0x0",
                "%4 = CompositeExtractF16x2 %2, #0x1",
                "%5 = UnpackFloat2x16 #0x78007800",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = UnpackFloat2x16 %0",
                "%9 = CompositeExtractF16x2 %8, #0x0",
                "%10 = CompositeExtractF16x2 %8, #0x1",
                "%11 = FPFma16 %3, %6, %9 (flags 0x31)",
                "%12 = FPFma16 %4, %7, %10 (flags 0x31)",
                "%13 = FPSaturate16 %11",
                "%14 = FPSaturate16 %12",
                "%15 = CompositeConstructF16x2 %13, %14",
                "%16 = PackFloat2x16 %15",
                "SetRegister R2, %16",
            ],
        );
    }

    #[test]
    fn hfma2_32i_reads_dest_as_addend() {
        check(
            encode(Opcode::Hfma2_32i, 2 | 4 << 8 | 0x3c00_3c00 << 20 | 1 << 52),
            &[
                "%0 = GetRegister R2",
                "%1 = GetRegister R4",
                "%2 = UnpackFloat2x16 %1",
                "%3 = CompositeExtractF16x2 %2, #0x0",
                "%4 = CompositeExtractF16x2 %2, #0x1",
                "%5 = UnpackFloat2x16 #0x3c003c00",
                "%6 = CompositeExtractF16x2 %5, #0x0",
                "%7 = CompositeExtractF16x2 %5, #0x1",
                "%8 = UnpackFloat2x16 %0",
                "%9 = CompositeExtractF16x2 %8, #0x0",
                "%10 = CompositeExtractF16x2 %8, #0x1",
                "%11 = FPNeg16 %9",
                "%12 = FPNeg16 %10",
                "%13 = FPFma16 %3, %6, %11 (flags 0x31)",
                "%14 = FPFma16 %4, %7, %12 (flags 0x31)",
                "%15 = CompositeConstructF16x2 %13, %14",
                "%16 = PackFloat2x16 %15",
                "SetRegister R2, %16",
            ],
        );
    }
}
