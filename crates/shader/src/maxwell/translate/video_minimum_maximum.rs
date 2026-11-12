//! VMNMX: video minimum/maximum, ported from video_minimum_maximum.cpp.

use super::video_multiply_add::{VideoWidth, extract_video_operand_value};
use super::{TResult, Translator};
use crate::ir::Value;

const OP_MIN: u64 = 5;
const OP_MAX: u64 = 6;

pub(super) fn vmnmx(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let op = insn.bits(51, 3);
    if insn.bit(47) || insn.bit(55) {
        return Err(t.unsupported());
    }
    let is_b_imm = !insn.bit(50);
    let src_a = t.get_reg8();
    let src_b = if is_b_imm {
        Value::U32(insn.bits(20, 16) as u32)
    } else {
        t.get_reg20()
    };
    let src_c = t.get_reg39();
    let a_width = VideoWidth::from_bits(insn.bits(37, 2));
    let b_width = VideoWidth::from_bits(insn.bits(29, 2)).source(is_b_imm);
    let a_selector = insn.bits(36, 2) as u32;
    let b_selector = if is_b_imm { 0 } else { insn.bits(28, 2) as u32 };
    let src_b_signed = insn.bit(49);
    let op_a = extract_video_operand_value(&mut t.ir, src_a, a_width, a_selector, insn.bit(48));
    let op_b = extract_video_operand_value(&mut t.ir, src_b, b_width, b_selector, src_b_signed);
    let lhs = if insn.bit(56) {
        t.ir.imax(op_a, op_b, src_b_signed)
    } else {
        t.ir.imin(op_a, op_b, src_b_signed)
    };
    let dest_signed = insn.bit(54);
    let result = match op {
        OP_MIN => t.ir.imin(lhs, src_c, dest_signed),
        OP_MAX => t.ir.imax(lhs, src_c, dest_signed),
        _ => return Err(t.unsupported()),
    };
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn vmnmx_words_max_then_signed_min() {
        let insn = encode(
            Opcode::Vmnmx,
            (1 << 8)
                | (2 << 20)
                | (3 << 39)
                | (3 << 29)
                | (3 << 37)
                | (1 << 50)
                | (5 << 51)
                | (1 << 54)
                | (1 << 56),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetRegister R3
  %3 = UMax32 %0, %1
  %4 = SMin32 %3, %2
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn vmnmx_imm_signed_byte_max() {
        let insn = encode(
            Opcode::Vmnmx,
            (1 << 8) | (0x1234 << 20) | (3 << 39) | (1 << 36) | (1 << 48) | (6 << 51),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R3
  %2 = BitFieldSExtract %0, #0x8, #0x8
  %3 = BitFieldUExtract #0x1234, #0x0, #0x10
  %4 = UMin32 %2, %3
  %5 = UMax32 %4, %1
  SetRegister R0, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn vmnmx_merge_op() {
        let insn = encode(Opcode::Vmnmx, (1 << 50) | (4 << 51));
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn vmnmx_sat() {
        let insn = encode(Opcode::Vmnmx, (1 << 55) | (5 << 51));
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
