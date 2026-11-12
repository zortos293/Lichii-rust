//! P2R: pack predicates or condition codes into a register, ported from
//! move_predicate_to_register.cpp.

use super::{TResult, Translator};
use crate::ir::{Pred, Value};

pub(super) fn p2r_cbuf(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn p2r_imm(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let mask = t.imm20();
    let pr_mode = !insn.bit(40);
    let num_items = if pr_mode { 7 } else { 4 };
    let offset = insn.bits(41, 2) as u32 * 8;
    let mut insert = Value::U32(0);
    for index in 0..num_items {
        if (mask >> index) & 1 == 0 {
            continue;
        }
        let cond = if pr_mode {
            t.ir.get_pred(Pred::p(index as u8), false)
        } else {
            match index {
                0 => t.ir.get_z_flag(),
                1 => t.ir.get_s_flag(),
                2 => t.ir.get_c_flag(),
                _ => t.ir.get_o_flag(),
            }
        };
        let bit =
            t.ir.select(cond, Value::U32(1 << (index + offset)), Value::U32(0));
        insert = t.ir.bitwise_or(insert, bit);
    }
    let src = t.x(insn.src_a_reg());
    let masked_out = t.ir.bitwise_and(src, Value::U32(!(mask << offset)));
    let result = t.ir.bitwise_or(masked_out, insert);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn p2r_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn p2r_imm_predicates() {
        let insn = encode(Opcode::P2rImm, (1 << 8) | (0x41 << 20) | (1 << 41));
        let expected = "\
{0}:
  %0 = GetPred P0
  %1 = SelectU32 %0, #0x100, #0x0
  %2 = BitwiseOr32 #0x0, %1
  %3 = GetPred P6
  %4 = SelectU32 %3, #0x4000, #0x0
  %5 = BitwiseOr32 %2, %4
  %6 = GetRegister R1
  %7 = BitwiseAnd32 %6, #0xffffbeff
  %8 = BitwiseOr32 %7, %5
  SetRegister R0, %8
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn p2r_imm_condition_codes() {
        let insn = encode(Opcode::P2rImm, (1 << 8) | (0x6 << 20) | (1 << 40));
        let expected = "\
{0}:
  %0 = GetSFlag
  %1 = SelectU32 %0, #0x2, #0x0
  %2 = BitwiseOr32 #0x0, %1
  %3 = GetCFlag
  %4 = SelectU32 %3, #0x4, #0x0
  %5 = BitwiseOr32 %2, %4
  %6 = GetRegister R1
  %7 = BitwiseAnd32 %6, #0xfffffff9
  %8 = BitwiseOr32 %7, %5
  SetRegister R0, %8
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn p2r_reg() {
        let insn = encode(Opcode::P2rReg, 0);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn p2r_cbuf() {
        let insn = encode(Opcode::P2rCbuf, 0);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
