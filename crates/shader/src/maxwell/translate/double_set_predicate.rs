//! DSETP.

use super::floating_point_set_predicate::write_set_predicates;
use super::{BooleanOp, FpCompareOp, TResult, Translator, floating_point_compare};
use crate::ir::{FpControl, Value};

fn dsetp(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    let Some(bop) = BooleanOp::from_bits(i.bits(45, 2)) else {
        return Err(t.unsupported());
    };
    let src_a = t.d(i.src_a_reg())?;
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(7), i.bit(43));
    let op_b = t.ir.fp_abs_neg(src_b, i.bit(44), i.bit(6));
    let compare_op = FpCompareOp::from_bits(i.bits(48, 4));
    let comparison =
        floating_point_compare(&mut t.ir, op_a, op_b, compare_op, FpControl::default());
    write_set_predicates(t, comparison, bop);
    Ok(())
}

pub(super) fn dsetp_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_cbuf()?;
    dsetp(t, src_b)
}

pub(super) fn dsetp_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_reg20()?;
    dsetp(t, src_b)
}

pub(super) fn dsetp_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_double_imm20();
    dsetp(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::check;
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn dsetp_reg_or_with_negated_source() {
        check(
            encode(
                Opcode::DsetpReg,
                2 | 1 << 3 | 4 << 8 | 6 << 20 | 3 << 39 | 1 << 42 | 1 << 45 | 6 << 48 | 1 << 6,
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
                "%8 = FPNeg64 %3",
                "%9 = FPOrdGreaterThanEqual64 %7, %8",
                "%10 = GetPred P3",
                "%11 = LogicalNot %10",
                "%12 = LogicalOr %9, %11",
                "%13 = LogicalNot %9",
                "%14 = LogicalOr %13, %11",
                "SetPred P1, %12",
                "SetPred P2, %14",
            ],
        );
    }

    #[test]
    fn dsetp_cbuf() {
        check(
            encode(
                Opcode::DsetpCbuf,
                2 | 1 << 3 | 4 << 8 | 1 << 34 | 4 << 20 | 7 << 39 | 1 << 48,
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
                "%8 = FPOrdLessThan64 %7, %3",
                "%9 = LogicalAnd %8, #true",
                "%10 = LogicalNot %8",
                "%11 = LogicalAnd %10, #true",
                "SetPred P1, %9",
                "SetPred P2, %11",
            ],
        );
    }

    #[test]
    fn dsetp_imm() {
        check(
            encode(
                Opcode::DsetpImm,
                2 | 1 << 3 | 4 << 8 | 0x3ff00 << 20 | 7 << 39 | 1 << 48,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = CompositeConstructU32x2 %0, %1",
                "%3 = PackDouble2x32 %2",
                "%4 = FPOrdLessThan64 %3, #1.0d",
                "%5 = LogicalAnd %4, #true",
                "%6 = LogicalNot %4",
                "%7 = LogicalAnd %6, #true",
                "SetPred P1, %5",
                "SetPred P2, %7",
            ],
        );
    }
}
