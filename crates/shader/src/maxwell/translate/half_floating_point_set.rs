//! HSET2.

use super::floating_point_add::ftz_mode;
use super::half_floating_point_add::{Swizzle, abs_neg_pair, extract, packed_imm, promote};
use super::{
    BooleanOp, FpCompareOp, TResult, Translator, floating_point_compare, predicate_combine,
};
use crate::ir::{FpControl, FpRounding, Pred, Value};

struct Hset2 {
    bf: bool,
    ftz: bool,
    neg_b: bool,
    abs_b: bool,
    compare_op: FpCompareOp,
    swizzle_b: Swizzle,
}

fn hset2(t: &mut Translator<'_>, src_b: Value, m: Hset2) -> TResult {
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
        fmz_mode: ftz_mode(m.ftz),
    };
    let pred = t.ir.get_pred(Pred::p(i.bits(39, 3) as u8), i.bit(42));
    let cmp_lhs = floating_point_compare(&mut t.ir, a.0, b.0, m.compare_op, control);
    let cmp_rhs = floating_point_compare(&mut t.ir, a.1, b.1, m.compare_op, control);
    let bop_lhs = predicate_combine(&mut t.ir, cmp_lhs, pred, bop);
    let bop_rhs = predicate_combine(&mut t.ir, cmp_rhs, pred, bop);
    let true_value: u32 = if m.bf { 0x3c00 } else { 0xffff };
    let fail = Value::U32(0);
    let result_lhs = t.ir.select(bop_lhs, Value::U32(true_value), fail);
    let result_rhs = t.ir.select(bop_rhs, Value::U32(true_value << 16), fail);
    let result = t.ir.bitwise_or(result_lhs, result_rhs);
    t.set_x(i.dest_reg(), result);
    Ok(())
}

pub(super) fn hset2_reg(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hset2 {
        bf: i.bit(49),
        ftz: i.bit(50),
        neg_b: i.bit(31),
        abs_b: i.bit(30),
        compare_op: FpCompareOp::from_bits(i.bits(35, 4)),
        swizzle_b: Swizzle::from_bits(i.bits(28, 2)),
    };
    let src_b = t.get_reg20();
    hset2(t, src_b, m)
}

pub(super) fn hset2_cbuf(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hset2 {
        bf: i.bit(53),
        ftz: i.bit(54),
        neg_b: i.bit(56),
        abs_b: false,
        compare_op: FpCompareOp::from_bits(i.bits(49, 4)),
        swizzle_b: Swizzle::F32,
    };
    let src_b = t.get_cbuf()?;
    hset2(t, src_b, m)
}

pub(super) fn hset2_imm(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    let m = Hset2 {
        bf: i.bit(53),
        ftz: i.bit(54),
        neg_b: false,
        abs_b: false,
        compare_op: FpCompareOp::from_bits(i.bits(49, 4)),
        swizzle_b: Swizzle::H1H0,
    };
    let src_b = packed_imm(t);
    hset2(t, src_b, m)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn hset2_reg_bf() {
        check(
            encode(
                Opcode::Hset2Reg,
                2 | 4 << 8 | 6 << 20 | 1 << 39 | 1 << 49 | 1 << 50 | 1 << 35,
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
                "%8 = GetPred P1",
                "%9 = FPOrdLessThan16 %3, %6 (flags 0x10)",
                "%10 = FPOrdLessThan16 %4, %7 (flags 0x10)",
                "%11 = LogicalAnd %9, %8",
                "%12 = LogicalAnd %10, %8",
                "%13 = SelectU32 %11, #0x3c00, #0x0",
                "%14 = SelectU32 %12, #0x3c000000, #0x0",
                "%15 = BitwiseOr32 %13, %14",
                "SetRegister R2, %15",
            ],
        );
    }

    #[test]
    fn hset2_cbuf_promotes() {
        check(
            encode(
                Opcode::Hset2Cbuf,
                2 | 4 << 8 | 1 << 34 | 3 << 20 | 7 << 39 | 1 << 45 | 2 << 49,
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
                "%8 = FPOrdEqual32 %6, %5 (flags 0x30)",
                "%9 = FPOrdEqual32 %7, %5 (flags 0x30)",
                "%10 = LogicalOr %8, #true",
                "%11 = LogicalOr %9, #true",
                "%12 = SelectU32 %10, #0xffff, #0x0",
                "%13 = SelectU32 %11, #0xffff0000, #0x0",
                "%14 = BitwiseOr32 %12, %13",
                "SetRegister R2, %14",
            ],
        );
    }

    #[test]
    fn hset2_imm() {
        check(
            encode(
                Opcode::Hset2Imm,
                2 | 4 << 8 | 0x1e0 << 20 | 7 << 39 | 4 << 49 | 1 << 53,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = UnpackFloat2x16 %0",
                "%2 = CompositeExtractF16x2 %1, #0x0",
                "%3 = CompositeExtractF16x2 %1, #0x1",
                "%4 = UnpackFloat2x16 #0x7800",
                "%5 = CompositeExtractF16x2 %4, #0x0",
                "%6 = CompositeExtractF16x2 %4, #0x1",
                "%7 = FPOrdGreaterThan16 %2, %5 (flags 0x30)",
                "%8 = FPOrdGreaterThan16 %3, %6 (flags 0x30)",
                "%9 = LogicalAnd %7, #true",
                "%10 = LogicalAnd %8, #true",
                "%11 = SelectU32 %9, #0x3c00, #0x0",
                "%12 = SelectU32 %10, #0x3c000000, #0x0",
                "%13 = BitwiseOr32 %11, %12",
                "SetRegister R2, %13",
            ],
        );
    }

    #[test]
    fn hset2_rejects_reserved_bop() {
        rejects(encode(Opcode::Hset2Reg, 2 | 4 << 8 | 6 << 20 | 3 << 45));
    }
}
