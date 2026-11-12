//! AL2P: converts an attribute address into a physical attribute address.

use super::{TResult, Translator};
use crate::ir::Value;

pub(super) fn al2p(t: &mut Translator<'_>) -> TResult {
    if t.insn.bits(47, 2) != 0 {
        return Err(t.unsupported());
    }
    let offset = Value::U32(t.insn.signed_bits(20, 11) as u32);
    let index = t.x(t.insn.src_a_reg());
    let result = t.ir.iadd(index, offset);
    t.set_x(t.insn.dest_reg(), result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn al2p_adds_signed_offset() {
        let insn = encode(Opcode::Al2p, (0x7fc << 20) | (5 << 8) | 2);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n  %0 = GetRegister R5\n  %1 = IAdd32 %0, #0xfffffffc\n  SetRegister R2, %1\n"
        );
    }

    #[test]
    fn al2p_wide_sizes_are_unsupported() {
        assert!(translate(&[encode(Opcode::Al2p, 1 << 47)]).is_err());
    }
}
