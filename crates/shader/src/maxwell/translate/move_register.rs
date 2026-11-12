//! MOV and MOV32I: register moves, ported from move_register.cpp.

use super::{TResult, Translator};
use crate::ir::Value;

fn mov(t: &mut Translator<'_>, src: Value, mask: u64) -> TResult {
    if mask != 0xf && mask != 0x1 {
        tracing::warn!(mask, "(STUBBED) masked MOV");
        return Ok(());
    }
    t.set_x(t.insn.dest_reg(), src);
    Ok(())
}

pub(super) fn mov_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = t.get_cbuf()?;
    mov(t, src, t.insn.bits(39, 4))
}

pub(super) fn mov_reg(t: &mut Translator<'_>) -> TResult {
    let src = t.get_reg20();
    mov(t, src, t.insn.bits(39, 4))
}

pub(super) fn mov32i(t: &mut Translator<'_>) -> TResult {
    let src = t.get_imm32();
    mov(t, src, t.insn.bits(12, 4))
}

pub(super) fn mov_imm(t: &mut Translator<'_>) -> TResult {
    let src = t.get_imm20();
    mov(t, src, t.insn.bits(39, 4))
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn mov_reg() {
        let insn = encode(Opcode::MovReg, (2 << 20) | (0xf << 39));
        let expected = "\
{0}:
  %0 = GetRegister R2
  SetRegister R0, %0
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn mov_cbuf() {
        let insn = encode(Opcode::MovCbuf, (1 << 34) | (4 << 20) | (1 << 39));
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  SetRegister R0, %0
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn mov_imm_masked_is_dropped() {
        let insn = encode(Opcode::MovImm, (5 << 20) | (3 << 39));
        let expected = "\
{0}:
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn mov32i() {
        let insn = encode(Opcode::Mov32i, (0xdead_beef << 20) | (0xf << 12));
        let expected = "\
{0}:
  SetRegister R0, #0xdeadbeef
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
