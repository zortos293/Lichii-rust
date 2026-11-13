//! FSETP.

use super::floating_point_add::ftz_mode;
use super::{
    BooleanOp, FpCompareOp, TResult, Translator, floating_point_compare, predicate_combine,
};
use crate::ir::{FpControl, FpRounding, Pred, Value};

/// Writes the two predicates of FSETP or DSETP: `comparison` and its negation, each combined
/// with the source predicate (bits 39..43) by `bop`.
pub(super) fn write_set_predicates(t: &mut Translator<'_>, comparison: Value, bop: BooleanOp) {
    let i = t.insn;
    let bop_pred = t.ir.get_pred(Pred::p(i.bits(39, 3) as u8), i.bit(42));
    let result_a = predicate_combine(&mut t.ir, comparison, bop_pred, bop);
    let not_comparison = t.ir.logical_not(comparison);
    let result_b = predicate_combine(&mut t.ir, not_comparison, bop_pred, bop);
    t.ir.set_pred(Pred::p(i.bits(3, 3) as u8), result_a);
    t.ir.set_pred(Pred::p(i.bits(0, 3) as u8), result_b);
}

fn fsetp(t: &mut Translator<'_>, src_b: Value) -> TResult {
    let i = t.insn;
    let Some(bop) = BooleanOp::from_bits(i.bits(45, 2)) else {
        return Err(t.unsupported());
    };
    let src_a = t.f(i.src_a_reg());
    let op_a = t.ir.fp_abs_neg(src_a, i.bit(7), i.bit(43));
    let op_b = t.ir.fp_abs_neg(src_b, i.bit(44), i.bit(6));
    let control = FpControl {
        no_contraction: false,
        rounding: FpRounding::DontCare,
        fmz_mode: ftz_mode(i.bit(47)),
    };
    let compare_op = FpCompareOp::from_bits(i.bits(48, 4));
    let comparison = floating_point_compare(&mut t.ir, op_a, op_b, compare_op, control);
    write_set_predicates(t, comparison, bop);
    Ok(())
}

pub(super) fn fsetp_cbuf(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_cbuf()?;
    fsetp(t, src_b)
}

pub(super) fn fsetp_reg(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_reg20();
    fsetp(t, src_b)
}

pub(super) fn fsetp_imm(t: &mut Translator<'_>) -> TResult {
    let src_b = t.get_float_imm20();
    fsetp(t, src_b)
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn fsetp_reg_xor_ftz() {
        check(
            encode(
                Opcode::FsetpReg,
                2 | 1 << 3 | 4 << 8 | 6 << 20 | 3 << 39 | 2 << 45 | 1 << 47 | 3 << 48 | 1 << 7,
            ),
            &[
                "%0 = GetRegister R6",
                "%1 = BitCastF32U32 %0",
                "%2 = GetRegister R4",
                "%3 = BitCastF32U32 %2",
                "%4 = FPAbs32 %3",
                "%5 = FPOrdLessThanEqual32 %4, %1 (flags 0x10)",
                "%6 = GetPred P3",
                "%7 = LogicalXor %5, %6",
                "%8 = LogicalNot %5",
                "%9 = LogicalXor %8, %6",
                "SetPred P1, %7",
                "SetPred P2, %9",
            ],
        );
    }

    #[test]
    fn fsetp_cbuf() {
        check(
            encode(
                Opcode::FsetpCbuf,
                2 | 1 << 3 | 4 << 8 | 1 << 34 | 3 << 20 | 7 << 39 | 1 << 48,
            ),
            &[
                "%0 = GetCbufF32 #0x1, #0xc",
                "%1 = GetRegister R4",
                "%2 = BitCastF32U32 %1",
                "%3 = FPOrdLessThan32 %2, %0 (flags 0x30)",
                "%4 = LogicalAnd %3, #true",
                "%5 = LogicalNot %3",
                "%6 = LogicalAnd %5, #true",
                "SetPred P1, %4",
                "SetPred P2, %6",
            ],
        );
    }

    #[test]
    fn fsetp_imm_unordered() {
        check(
            encode(
                Opcode::FsetpImm,
                2 | 1 << 3 | 4 << 8 | 0x3f800 << 20 | 7 << 39 | 9 << 48,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = BitCastF32U32 %0",
                "%2 = FPUnordLessThan32 %1, #1.0f (flags 0x30)",
                "%3 = LogicalAnd %2, #true",
                "%4 = LogicalNot %2",
                "%5 = LogicalAnd %4, #true",
                "SetPred P1, %3",
                "SetPred P2, %5",
            ],
        );
    }

    #[test]
    fn fsetp_rejects_reserved_bop() {
        rejects(encode(Opcode::FsetpReg, 2 | 4 << 8 | 6 << 20 | 3 << 45));
    }
}
