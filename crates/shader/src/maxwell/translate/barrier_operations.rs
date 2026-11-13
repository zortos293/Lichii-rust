//! BAR, DEPBAR and MEMBAR.

use super::{TResult, Translator};

pub(super) fn bar(t: &mut Translator<'_>) -> TResult {
    const MODE_MASK: u64 = 0x0000_009b_0000_0000;
    const SYNC: u64 = 0x0000_0080_0000_0000;
    let insn = t.insn;
    let pred_is_true = insn.bits(39, 3) == 7;
    if insn.0 & MODE_MASK != SYNC
        || !insn.bit(43)
        || insn.bits(8, 8) != 0
        || !insn.bit(44)
        || insn.bits(20, 12) != 0
        || (!pred_is_true && insn.bit(42))
    {
        return Err(t.unsupported());
    }
    t.ir.barrier();
    Ok(())
}

pub(super) fn depbar(_: &mut Translator<'_>) -> TResult {
    Ok(())
}

pub(super) fn membar(t: &mut Translator<'_>) -> TResult {
    if t.insn.bits(8, 2) == 0 {
        t.ir.workgroup_memory_barrier();
    } else {
        t.ir.device_memory_barrier();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    const BAR_SYNC: u64 = (7 << 39) | (1 << 43) | (1 << 44);

    #[test]
    fn bar_sync() {
        let insn = encode(Opcode::Bar, BAR_SYNC);
        assert_eq!(translate(&[insn]).unwrap(), "{0}:\n  Barrier\n");
    }

    #[test]
    fn bar_rejects_other_modes_and_operands() {
        let arrive = encode(Opcode::Bar, BAR_SYNC | (1 << 32));
        let reg_a = encode(Opcode::Bar, BAR_SYNC & !(1 << 43));
        let nonzero_b = encode(Opcode::Bar, BAR_SYNC | (1 << 20));
        let negated_pred = encode(Opcode::Bar, (BAR_SYNC & !(7 << 39)) | (1 << 39) | (1 << 42));
        for insn in [arrive, reg_a, nonzero_b, negated_pred] {
            assert!(translate(&[insn]).is_err(), "{insn:#x}");
        }
    }

    #[test]
    fn depbar_is_a_no_op() {
        assert_eq!(translate(&[encode(Opcode::Depbar, 0)]).unwrap(), "{0}:\n");
    }

    #[test]
    fn membar_scopes() {
        assert_eq!(
            translate(&[encode(Opcode::Membar, 0)]).unwrap(),
            "{0}:\n  WorkgroupMemoryBarrier\n"
        );
        assert_eq!(
            translate(&[encode(Opcode::Membar, 1 << 8)]).unwrap(),
            "{0}:\n  DeviceMemoryBarrier\n"
        );
    }
}
