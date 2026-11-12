//! ISETP: integer compare to predicates, ported from integer_set_predicate.cpp.

use super::{
    BooleanOp, CompareOp, TResult, Translator, extended_integer_compare, integer_compare,
    predicate_combine,
};
use crate::ir::{Pred, Value};

fn isetp(t: &mut Translator<'_>, op_b: Value) -> TResult {
    let insn = t.insn;
    let bop = BooleanOp::from_bits(insn.bits(45, 2)).ok_or_else(|| t.unsupported())?;
    let signed = insn.bit(48);
    let op = CompareOp::from_bits(insn.bits(49, 3));
    let op_a = t.x(insn.src_a_reg());
    let comparison = if insn.bit(43) {
        extended_integer_compare(&mut t.ir, op_a, op_b, op, signed)
    } else {
        integer_compare(&mut t.ir, op_a, op_b, op, signed)
    };
    let bop_pred = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), insn.bit(42));
    let result_a = predicate_combine(&mut t.ir, comparison, bop_pred, bop);
    let not_comparison = t.ir.logical_not(comparison);
    let result_b = predicate_combine(&mut t.ir, not_comparison, bop_pred, bop);
    t.ir.set_pred(Pred::p(insn.bits(3, 3) as u8), result_a);
    t.ir.set_pred(Pred::p(insn.bits(0, 3) as u8), result_b);
    Ok(())
}

pub(super) fn isetp_cbuf(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_cbuf()?;
    isetp(t, op_b)
}

pub(super) fn isetp_reg(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_reg20();
    isetp(t, op_b)
}

pub(super) fn isetp_imm(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_imm20();
    isetp(t, op_b)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn isetp_reg_lt() {
        let insn = encode(
            Opcode::IsetpReg,
            (1 << 3) | 2 | (1 << 8) | (2 << 20) | (7 << 39) | (1 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = ULessThan %1, %0
  %3 = LogicalAnd %2, #true
  %4 = LogicalNot %2
  %5 = LogicalAnd %4, #true
  SetPred P1, %3
  SetPred P2, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn isetp_imm_extended_signed_lt() {
        let insn = encode(
            Opcode::IsetpImm,
            (1 << 3) | 7 | (1 << 8) | (5 << 20) | (7 << 39) | (1 << 43) | (1 << 48) | (1 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetCFlag
  %2 = SelectU32 %1, #0x1, #0x0
  %3 = GetZFlag
  %4 = BitwiseNot32 #0x5
  %5 = IAdd32 %0, %4
  %6 = IAdd32 %5, %2
  %7 = SGreaterThanEqual %6, #0x0
  %8 = SLessThan %6, #0x0
  %9 = SelectU1 #false, %7, %8
  %10 = LogicalAnd %9, #true
  %11 = LogicalNot %9
  %12 = LogicalAnd %11, #true
  SetPred P1, %10
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn isetp_cbuf_ne_or() {
        let insn = encode(
            Opcode::IsetpCbuf,
            (1 << 3)
                | 7
                | (1 << 8)
                | (1 << 34)
                | (4 << 20)
                | (2 << 39)
                | (1 << 42)
                | (1 << 45)
                | (5 << 49),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = INotEqual %1, %0
  %3 = GetPred P2
  %4 = LogicalNot %3
  %5 = LogicalOr %2, %4
  %6 = LogicalNot %2
  %7 = LogicalOr %6, %4
  SetPred P1, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn isetp_reserved_bop() {
        let insn = encode(Opcode::IsetpReg, 3 << 45);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
