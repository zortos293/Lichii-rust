//! PSET: combine three predicates into a register mask or float, ported from
//! predicate_set_register.cpp.

use super::{BooleanOp, TResult, Translator, predicate_combine};
use crate::ir::{ICompare, Pred, Value};

pub(super) fn pset(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let (Some(bop_1), Some(bop_2)) = (
        BooleanOp::from_bits(insn.bits(24, 2)),
        BooleanOp::from_bits(insn.bits(45, 2)),
    ) else {
        return Err(t.unsupported());
    };
    let bf = insn.bit(44);
    let pred_a = t.ir.get_pred(Pred::p(insn.bits(12, 3) as u8), insn.bit(15));
    let pred_b = t.ir.get_pred(Pred::p(insn.bits(29, 3) as u8), insn.bit(32));
    let pred_c = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), insn.bit(42));
    let res_1 = predicate_combine(&mut t.ir, pred_a, pred_b, bop_1);
    let res_2 = predicate_combine(&mut t.ir, res_1, pred_c, bop_2);
    let true_result = Value::U32(if bf { 0x3f80_0000 } else { u32::MAX });
    let zero = Value::U32(0);
    let result = t.ir.select(res_2, true_result, zero);
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

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn pset_float_cc() {
        let insn = encode(
            Opcode::Pset,
            (3 << 12) | (4 << 29) | (1 << 32) | (7 << 39) | (1 << 44) | (1 << 47),
        );
        let expected = "\
{0}:
  %0 = GetPred P3
  %1 = GetPred P4
  %2 = LogicalNot %1
  %3 = LogicalAnd %0, %2
  %4 = LogicalAnd %3, #true
  %5 = SelectU32 %4, #0x3f800000, #0x0
  SetRegister R0, %5
  %6 = IEqual %5, #0x0
  SetZFlag %6
  SetSFlag #false
  SetOFlag #false
  SetCFlag #false
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn pset_reserved_bop() {
        let insn = encode(Opcode::Pset, 3 << 45);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
