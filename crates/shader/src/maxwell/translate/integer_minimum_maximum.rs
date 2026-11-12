//! IMNMX: integer minimum or maximum chosen by a predicate, ported from
//! integer_minimum_maximum.cpp.

use super::{TResult, Translator};
use crate::ir::{Pred, Value};

fn imnmx(t: &mut Translator<'_>, op_b: Value) -> TResult {
    let insn = t.insn;
    if insn.bit(47) || insn.bits(43, 2) != 0 {
        return Err(t.unsupported());
    }
    let pred = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), false);
    let op_a = t.x(insn.src_a_reg());
    let signed = insn.bit(48);
    let mut min = t.ir.imin(op_a, op_b, signed);
    let mut max = t.ir.imax(op_a, op_b, signed);
    if insn.bit(42) {
        std::mem::swap(&mut min, &mut max);
    }
    let result = t.ir.select(pred, min, max);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn imnmx_cbuf(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_cbuf()?;
    imnmx(t, op_b)
}

pub(super) fn imnmx_reg(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_reg20();
    imnmx(t, op_b)
}

pub(super) fn imnmx_imm(t: &mut Translator<'_>) -> TResult {
    let op_b = t.get_imm20();
    imnmx(t, op_b)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn imnmx_reg_signed_negated_pred() {
        let insn = encode(
            Opcode::ImnmxReg,
            (1 << 8) | (2 << 20) | (1 << 39) | (1 << 42) | (1 << 48),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetPred P1
  %2 = GetRegister R1
  %3 = SMin32 %2, %0
  %4 = SMax32 %2, %0
  %5 = SelectU32 %1, %4, %3
  SetRegister R0, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn imnmx_imm() {
        let insn = encode(Opcode::ImnmxImm, (1 << 8) | (5 << 20) | (7 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = UMin32 %0, #0x5
  %2 = UMax32 %0, #0x5
  %3 = SelectU32 #true, %1, %2
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn imnmx_cbuf() {
        let insn = encode(
            Opcode::ImnmxCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (7 << 39),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = UMin32 %1, %0
  %3 = UMax32 %1, %0
  %4 = SelectU32 #true, %2, %3
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn imnmx_mode() {
        let insn = encode(Opcode::ImnmxReg, 1 << 43);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
