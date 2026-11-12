//! DSET.

use super::floating_point_compare_and_set::write_set_result;
use super::{
    BooleanOp, FpCompareOp, TResult, Translator, floating_point_compare, predicate_combine,
};
use crate::ir::{FpControl, Pred, Value};

fn dset(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    let Some(bop) = BooleanOp::from_bits(i.bits(45, 2)) else {
        return Err(t.unsupported());
    };
    let src_a = t.d(i.src_a_reg())?;
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(54), i.bit(43));
    let op_b = t.ir.fp_abs_neg(src_b, i.bit(44), i.bit(53));
    let pred = t.ir.get_pred(Pred::p(i.bits(39, 3) as u8), i.bit(42));
    let compare_op = FpCompareOp::from_bits(i.bits(48, 4));
    let cmp = floating_point_compare(&mut t.ir, op_a, op_b, compare_op, FpControl::default());
    let result = predicate_combine(&mut t.ir, cmp, pred, bop);
    write_set_result(t, result);
    Ok(())
}

pub(super) fn dset_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_cbuf()?;
    dset(t, src_b)
}

pub(super) fn dset_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_reg20()?;
    dset(t, src_b)
}

pub(super) fn dset_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_imm20();
    dset(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn dset_reg_bf_with_condition_codes() {
        check(
            encode(
                Opcode::DsetReg,
                2 | 4 << 8 | 6 << 20 | 1 << 39 | 1 << 47 | 1 << 48 | 1 << 52,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = GetRegister R7",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = GetPred P1",
                "%9 = FPOrdLessThan64 %7, %3",
                "%10 = LogicalAnd %9, %8",
                "%11 = SelectU32 %10, #0x3f800000, #0x0",
                "SetRegister R2, %11",
                "%12 = IEqual %11, #0x0",
                "SetZFlag %12",
                "SetSFlag #false",
                "SetCFlag #false",
                "SetOFlag #false",
            ],
        );
    }

    #[test]
    fn dset_cbuf_mask() {
        check(
            encode(
                Opcode::DsetCbuf,
                2 | 4 << 8 | 1 << 34 | 4 << 20 | 7 << 39 | 1 << 45 | 2 << 48 | 1 << 43 | 1 << 54,
            ),
            &[
                "%0 = GetCbufU32 #0x1, #0x14",
                "%1 = GetCbufU32 #0x1, #0x10",
                "%2 = CompositeConstructU32x2 %1, %0",
                "%3 = PackDouble2x32 %2",
                "%4 = GetRegister R4",
                "%5 = GetRegister R5",
                "%6 = CompositeConstructU32x2 %4, %5",
                "%7 = PackDouble2x32 %6",
                "%8 = FPAbs64 %7",
                "%9 = FPNeg64 %8",
                "%10 = FPOrdEqual64 %9, %3",
                "%11 = LogicalOr %10, #true",
                "%12 = SelectU32 %11, #0xffffffff, #0x0",
                "SetRegister R2, %12",
            ],
        );
    }

    #[test]
    fn dset_imm() {
        check(
            encode(
                Opcode::DsetImm,
                2 | 4 << 8 | 0x3ff00 << 20 | 7 << 39 | 9 << 48,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPUnordLessThan64 %3, #1.0d",
                "%5 = LogicalAnd %4, #true",
                "%6 = SelectU32 %5, #0xffffffff, #0x0",
                "SetRegister R2, %6",
            ],
        );
    }

    #[test]
    fn dset_rejects_reserved_bop() {
        rejects(encode(Opcode::DsetReg, 2 | 4 << 8 | 6 << 20 | 3 << 45));
    }
}
