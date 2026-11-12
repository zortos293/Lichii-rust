//! PIXLD: per-pixel values. Only the sample index is supported.

use super::{TResult, Translator};
use crate::ir::Opcode;

pub(super) fn pixld(t: &mut Translator<'_>) -> TResult {
    const MY_INDEX: u64 = 5;
    let insn = t.insn;
    if insn.bits(45, 3) != 7
        || !insn.src_a_reg().is_zero()
        || insn.bits(20, 8) != 0
        || insn.bits(31, 3) != MY_INDEX
    {
        return Err(t.unsupported());
    }
    let sample = t.ir.inst(Opcode::SampleId, &[]);
    t.set_x(insn.dest_reg(), sample);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    const BASE: u64 = (7 << 45) | (5 << 31) | (0xff << 8);

    #[test]
    fn pixld_my_index() {
        let insn = encode(Opcode::Pixld, BASE | 6);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n  %0 = SampleId\n  SetRegister R6, %0\n"
        );
    }

    #[test]
    fn pixld_other_forms_are_unsupported() {
        let covmask = encode(Opcode::Pixld, (BASE & !(7 << 31)) | (1 << 31));
        let dest_pred = encode(Opcode::Pixld, BASE & !(1 << 45));
        let addr_reg = encode(Opcode::Pixld, BASE & !(1 << 8));
        let addr_offset = encode(Opcode::Pixld, BASE | (1 << 20));
        for insn in [covmask, dest_pred, addr_reg, addr_offset] {
            assert!(translate(&[insn]).is_err(), "{insn:#x}");
        }
    }
}
