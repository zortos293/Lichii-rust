//! HSETP2.

use super::floating_point_add::ftz_mode;
use super::half_floating_point_add::{Swizzle, abs_neg_pair, extract, packed_imm, promote};
use super::{
    BooleanOp, FpCompareOp, TResult, Translator, floating_point_compare, predicate_combine,
};
use crate::ir::{FpControl, FpRounding, Pred, Value};

struct Hsetp2 {
    neg_b: bool,
    abs_b: bool,
    swizzle_b: Swizzle,
    compare_op: FpCompareOp,
    h_and: bool,
}

fn hsetp2(t: &mut Translator<'_>, src_b: Value, m: Hsetp2) -> TResult {
    let i = t.insn;
    let Some(bop) = BooleanOp::from_bits(i.bits(45, 2)) else {
        return Err(t.unsupported());
    };
    let src_a = t.x(i.src_a_reg());
    let mut a = extract(&mut t.ir, src_a, Swizzle::from_bits(i.bits(47, 2)));
    let mut b = extract(&mut t.ir, src_b, m.swizzle_b);
    if t.ir.ty(a.0) != t.ir.ty(b.0) {
        a = promote(&mut t.ir, a);
        b = promote(&mut t.ir, b);
    }
    let a = abs_neg_pair(&mut t.ir, a, i.bit(44), i.bit(43));
    let b = abs_neg_pair(&mut t.ir, b, m.abs_b, m.neg_b);
    let control = FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: ftz_mode(i.bit(6)),
    };
    let pred = t.ir.get_pred(Pred::p(i.bits(39, 3) as u8), i.bit(42));
    let cmp_lhs = floating_point_compare(&mut t.ir, a.0, b.0, m.compare_op, control);
    let cmp_rhs = floating_point_compare(&mut t.ir, a.1, b.1, m.compare_op, control);
    let bop_lhs = predicate_combine(&mut t.ir, cmp_lhs, pred, bop);
    let bop_rhs = predicate_combine(&mut t.ir, cmp_rhs, pred, bop);
    let dest_a = Pred::p(i.bits(3, 3) as u8);
    let dest_b = Pred::p(i.bits(0, 3) as u8);
    if m.h_and {
        let result = t.ir.logical_and(bop_lhs, bop_rhs);
        t.ir.set_pred(dest_a, result);
        let not_result = t.ir.logical_not(result);
        t.ir.set_pred(dest_b, not_result);
    } else {
        t.ir.set_pred(dest_a, bop_lhs);
        t.ir.set_pred(dest_b, bop_rhs);
    }
    Ok(())
}

pub(super) fn hsetp2_reg(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hsetp2 {
        neg_b: i.bit(31),
        abs_b: i.bit(30),
        swizzle_b: Swizzle::from_bits(i.bits(28, 2)),
        compare_op: FpCompareOp::from_bits(i.bits(35, 4)),
        h_and: i.bit(49),
    };
    let src_b = t.get_reg20();
    hsetp2(t, src_b, m)
}

pub(super) fn hsetp2_cbuf(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hsetp2 {
        neg_b: i.bit(56),
        abs_b: i.bit(54),
        swizzle_b: Swizzle::F32,
        compare_op: FpCompareOp::from_bits(i.bits(49, 4)),
        h_and: i.bit(53),
    };
    let src_b = t.get_cbuf()?;
    hsetp2(t, src_b, m)
}

pub(super) fn hsetp2_imm(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hsetp2 {
        neg_b: false,
        abs_b: false,
        swizzle_b: Swizzle::H1H0,
        compare_op: FpCompareOp::from_bits(i.bits(49, 4)),
        h_and: i.bit(53),
    };
    let src_b = packed_imm(t);
    hsetp2(t, src_b, m)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn hsetp2_reg_h_and() {
        check(
            encode(
                Opcode::Hsetp2Reg,
                2 | 1 << 3 | 4 << 8 | 6 << 20 | 7 << 39 | 1 << 49 | 1 << 35 | 1 << 6,
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
                "%8 = FPOrdLessThan16 %3, %6 (flags 0x10)",
                "%9 = FPOrdLessThan16 %4, %7 (flags 0x10)",
                "%10 = LogicalAnd %8, #true",
                "%11 = LogicalAnd %9, #true",
                "%12 = LogicalAnd %10, %11",
                "SetPred P1, %12",
                "%13 = LogicalNot %12",
                "SetPred P2, %13",
            ],
        );
    }

    #[test]
    fn hsetp2_cbuf() {
        check(
            encode(
                Opcode::Hsetp2Cbuf,
                2 | 1 << 3 | 4 << 8 | 1 << 34 | 3 << 20 | 7 << 39 | 1 << 49,
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
                "%8 = FPOrdLessThan32 %6, %5 (flags 0x30)",
                "%9 = FPOrdLessThan32 %7, %5 (flags 0x30)",
                "%10 = LogicalAnd %8, #true",
                "%11 = LogicalAnd %9, #true",
                "SetPred P1, %10",
                "SetPred P2, %11",
            ],
        );
    }

    #[test]
    fn hsetp2_imm() {
        check(
            encode(
                Opcode::Hsetp2Imm,
                2 | 1 << 3 | 4 << 8 | 0x1e0 << 20 | 7 << 39 | 5 << 49,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = UnpackFloat2x16 #0x7800",
                "%5 = CompositeExtractF16x2 %4, #0x0",
                "%6 = CompositeExtractF16x2 %4, #0x1",
                "%7 = FPOrdNotEqual16 %2, %5 (flags 0x30)",
                "%8 = FPOrdNotEqual16 %3, %6 (flags 0x30)",
                "%9 = LogicalAnd %7, #true",
                "%10 = LogicalAnd %8, #true",
                "SetPred P1, %9",
                "SetPred P2, %10",
            ],
        );
    }

    #[test]
    fn hsetp2_rejects_reserved_bop() {
        rejects(encode(Opcode::Hsetp2Reg, 2 | 4 << 8 | 6 << 20 | 3 << 45));
    }
}
