//! VSETP: video compare to predicates, ported from video_set_predicate.cpp.

use super::video_multiply_add::{VideoWidth, extract_video_operand_value};
use super::{BooleanOp, CompareOp, TResult, Translator, integer_compare, predicate_combine};
use crate::ir::{Pred, Value};

pub(super) fn vsetp(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let compare_op = match insn.bits(43, 5) {
        0 => CompareOp::False,
        1 => CompareOp::LessThan,
        2 => CompareOp::Equal,
        3 => CompareOp::LessThanEqual,
        16 => CompareOp::GreaterThan,
        17 => CompareOp::NotEqual,
        18 => CompareOp::GreaterThanEqual,
        19 => CompareOp::True,
        _ => return Err(t.unsupported()),
    };
    let bop = BooleanOp::from_bits(insn.bits(45, 2)).ok_or_else(|| t.unsupported())?;
    let is_b_imm = !insn.bit(50);
    let src_a = t.get_reg8();
    let src_b = if is_b_imm {
        Value::U32(insn.bits(20, 16) as u32)
    } else {
        t.get_reg20()
    };
    let a_selector = insn.bits(36, 2) as u32;
    let b_selector = insn.bits(28, 2) as u32;
    let a_width = VideoWidth::from_bits(insn.bits(37, 2));
    let b_width = VideoWidth::from_bits(insn.bits(29, 2)).source(is_b_imm);
    let src_b_signed = insn.bit(49);
    let op_a = extract_video_operand_value(&mut t.ir, src_a, a_width, a_selector, insn.bit(48));
    let op_b = extract_video_operand_value(&mut t.ir, src_b, b_width, b_selector, src_b_signed);
    let comparison = integer_compare(&mut t.ir, op_a, op_b, compare_op, src_b_signed);
    let bop_pred = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), insn.bit(42));
    let result_a = predicate_combine(&mut t.ir, comparison, bop_pred, bop);
    let not_comparison = t.ir.logical_not(comparison);
    let result_b = predicate_combine(&mut t.ir, not_comparison, bop_pred, bop);
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
    fn vsetp_words_signed_gt() {
        let insn = encode(
            Opcode::Vsetp,
            (1 << 3)
                | 7
                | (1 << 8)
                | (2 << 20)
                | (3 << 29)
                | (3 << 37)
                | (7 << 39)
                | (16 << 43)
                | (1 << 49)
                | (1 << 50),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = SGreaterThan %0, %1
  %3 = LogicalAnd %2, #true
  %4 = LogicalNot %2
  %5 = LogicalAnd %4, #true
  SetPred P1, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn vsetp_imm_lt_with_pred() {
        let insn = encode(
            Opcode::Vsetp,
            (1 << 3) | 7 | (1 << 8) | (0x10 << 20) | (1 << 39) | (1 << 43),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = BitFieldUExtract %0, #0x0, #0x8
  %2 = BitFieldUExtract #0x10, #0x0, #0x10
  %3 = ULessThan %1, %2
  %4 = GetPred P1
  %5 = LogicalAnd %3, %4
  %6 = LogicalNot %3
  %7 = LogicalAnd %6, %4
  SetPred P1, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn vsetp_reserved_compare() {
        let insn = encode(Opcode::Vsetp, 4 << 43);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
