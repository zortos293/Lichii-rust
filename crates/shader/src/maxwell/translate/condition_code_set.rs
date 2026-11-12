//! CSET and CSETP: condition-code tests to a register or predicates, ported from
//! condition_code_set.cpp.

use super::{BooleanOp, TResult, Translator, predicate_combine};
use crate::ir::{FlowTest, ICompare, Pred, Value};

pub(super) fn cset(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let bop = BooleanOp::from_bits(insn.bits(45, 2)).ok_or_else(|| t.unsupported())?;
    let bf = insn.bit(44);
    let zero = Value::U32(0);
    let pass_result = Value::U32(if bf { 0x3f80_0000 } else { u32::MAX });
    let cc_test = t.ir.flow_test(FlowTest::from_bits(insn.bits(8, 5) as u32));
    let bop_pred = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), insn.bit(42));
    let pred_result = predicate_combine(&mut t.ir, cc_test, bop_pred, bop);
    let result = t.ir.select(pred_result, pass_result, zero);
    t.set_x(insn.dest_reg(), result);
    if insn.bit(47) {
        let is_zero = t.ir.icompare(ICompare::Equal, result, zero);
        t.ir.set_z_flag(is_zero);
        if bf {
            t.reset_s_flag();
        } else {
            let not_zero = t.ir.logical_not(is_zero);
            t.ir.set_s_flag(not_zero);
        }
        t.reset_o_flag();
        t.reset_c_flag();
    }
    Ok(())
}

pub(super) fn csetp(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let bop = BooleanOp::from_bits(insn.bits(45, 2)).ok_or_else(|| t.unsupported())?;
    let bop_pred = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), insn.bit(42));
    let cc_test = t.ir.flow_test(FlowTest::from_bits(insn.bits(8, 5) as u32));
    let result_a = predicate_combine(&mut t.ir, cc_test, bop_pred, bop);
    let not_cc_test = t.ir.logical_not(cc_test);
    let result_b = predicate_combine(&mut t.ir, not_cc_test, bop_pred, bop);
    t.ir.set_pred(Pred::p(insn.bits(3, 3) as u8), result_a);
    t.ir.set_pred(Pred::p(insn.bits(0, 3) as u8), result_b);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn cset_eq() {
        let insn = encode(Opcode::Cset, (2 << 8) | (7 << 39));
        let expected = "\
{0}:
  %0 = GetSFlag
  %1 = GetZFlag
  %2 = LogicalNot %0
  %3 = LogicalAnd %2, %1
  %4 = LogicalAnd %3, #true
  %5 = SelectU32 %4, #0xffffffff, #0x0
  SetRegister R0, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn cset_float_or_cc() {
        let insn = encode(
            Opcode::Cset,
            (2 << 8) | (1 << 39) | (1 << 42) | (1 << 44) | (1 << 45) | (1 << 47),
        );
        let expected = "\
{0}:
  %0 = GetSFlag
  %1 = GetZFlag
  %2 = LogicalNot %0
  %3 = LogicalAnd %2, %1
  %4 = GetPred P1
  %5 = LogicalNot %4
  %6 = LogicalOr %3, %5
  %7 = SelectU32 %6, #0x3f800000, #0x0
  SetRegister R0, %7
  %8 = IEqual %7, #0x0
  SetZFlag %8
  SetSFlag #false
  SetOFlag #false
  SetCFlag #false
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn cset_reserved_bop() {
        let insn = encode(Opcode::Cset, 3 << 45);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn csetp_ne_xor() {
        let insn = encode(
            Opcode::Csetp,
            (1 << 3) | 7 | (5 << 8) | (2 << 39) | (2 << 45),
        );
        let expected = "\
{0}:
  %0 = GetPred P2
  %1 = GetZFlag
  %2 = LogicalNot %1
  %3 = LogicalXor %2, %0
  %4 = LogicalNot %2
  %5 = LogicalXor %4, %0
  SetPred P1, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
