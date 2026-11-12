//! PSETP: combine three predicates into two, ported from predicate_set_predicate.cpp.

use super::{BooleanOp, TResult, Translator, predicate_combine};
use crate::ir::Pred;

pub(super) fn psetp(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let (Some(bop_1), Some(bop_2)) = (
        BooleanOp::from_bits(insn.bits(24, 2)),
        BooleanOp::from_bits(insn.bits(45, 2)),
    ) else {
        return Err(t.unsupported());
    };
    let pred_a = t.ir.get_pred(Pred::p(insn.bits(12, 3) as u8), insn.bit(15));
    let pred_b = t.ir.get_pred(Pred::p(insn.bits(29, 3) as u8), insn.bit(32));
    let pred_c = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), insn.bit(42));
    let lhs_a = predicate_combine(&mut t.ir, pred_a, pred_b, bop_1);
    let not_a = t.ir.logical_not(pred_a);
    let lhs_b = predicate_combine(&mut t.ir, not_a, pred_b, bop_1);
    let result_a = predicate_combine(&mut t.ir, lhs_a, pred_c, bop_2);
    let result_b = predicate_combine(&mut t.ir, lhs_b, pred_c, bop_2);
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
    fn psetp_or_xor() {
        let insn = encode(
            Opcode::Psetp,
            (1 << 3) | 2 | (3 << 12) | (1 << 15) | (1 << 24) | (4 << 29) | (7 << 39) | (2 << 45),
        );
        let expected = "\
{0}:
  %0 = GetPred P3
  %1 = LogicalNot %0
  %2 = GetPred P4
  %3 = LogicalOr %1, %2
  %4 = LogicalNot %1
  %5 = LogicalOr %4, %2
  %6 = LogicalXor %3, #true
  %7 = LogicalXor %5, #true
  SetPred P1, %6
  SetPred P2, %7
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn psetp_reserved_bop() {
        let insn = encode(Opcode::Psetp, 3 << 24);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
