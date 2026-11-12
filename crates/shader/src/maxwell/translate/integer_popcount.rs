//! POPC: population count, ported from integer_popcount.cpp.

use super::{TResult, Translator};
use crate::ir::Value;

fn popc(t: &mut Translator<'_>, src: Value) -> TResult {
    let insn = t.insn;
    let operand = if insn.bit(40) {
        t.ir.bitwise_not(src)
    } else {
        src
    };
    let result = t.ir.bit_count(operand);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn popc_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = t.get_cbuf()?;
    popc(t, src)
}

pub(super) fn popc_reg(t: &mut Translator<'_>) -> TResult {
    let src = t.get_reg20();
    popc(t, src)
}

pub(super) fn popc_imm(t: &mut Translator<'_>) -> TResult {
    let src = t.get_imm20();
    popc(t, src)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn popc_reg() {
        let insn = encode(Opcode::PopcReg, 2 << 20);
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = BitCount32 %0
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn popc_imm_inverted() {
        let insn = encode(Opcode::PopcImm, (5 << 20) | (1 << 40));
        let expected = "\
{0}:
  %0 = BitwiseNot32 #0x5
  %1 = BitCount32 %0
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn popc_cbuf() {
        let insn = encode(Opcode::PopcCbuf, (1 << 34) | (4 << 20));
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = BitCount32 %0
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
