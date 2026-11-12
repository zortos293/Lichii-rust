//! SEL: select between two sources on a predicate, ported from
//! select_source_with_predicate.cpp.

use super::{TResult, Translator};
use crate::ir::{Pred, Value};

fn sel(t: &mut Translator<'_>, src: Value) -> TResult {
    let insn = t.insn;
    let pred = t.ir.get_pred(Pred::p(insn.bits(39, 3) as u8), false);
    let mut op_a = t.x(insn.src_a_reg());
    let mut op_b = src;
    if insn.bit(42) {
        std::mem::swap(&mut op_a, &mut op_b);
    }
    let result = t.ir.select(pred, op_a, op_b);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn sel_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = t.get_cbuf()?;
    sel(t, src)
}

pub(super) fn sel_reg(t: &mut Translator<'_>) -> TResult {
    let src = t.get_reg20();
    sel(t, src)
}

pub(super) fn sel_imm(t: &mut Translator<'_>) -> TResult {
    let src = t.get_imm20();
    sel(t, src)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn sel_reg() {
        let insn = encode(Opcode::SelReg, (1 << 8) | (2 << 20) | (1 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetPred P1
  %2 = GetRegister R1
  %3 = SelectU32 %1, %2, %0
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn sel_imm_negated() {
        let insn = encode(Opcode::SelImm, (1 << 8) | (5 << 20) | (1 << 39) | (1 << 42));
        let expected = "\
{0}:
  %0 = GetPred P1
  %1 = GetRegister R1
  %2 = SelectU32 %0, #0x5, %1
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn sel_cbuf() {
        let insn = encode(
            Opcode::SelCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (7 << 39),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = SelectU32 #true, %1, %0
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
