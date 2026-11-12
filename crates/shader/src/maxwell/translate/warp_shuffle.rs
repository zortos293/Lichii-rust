//! SHFL: warp shuffles, with quad operations for fragment derivative patterns.

use super::{TResult, Translator};
use crate::ShaderStage;
use crate::ir::{Opcode, Pred, Value};

/// Clamp 3 with a segment mask of 28: shuffles confined to a quad.
const QUAD_MASK: u32 = (28 << 8) | 3;

pub(super) fn shfl(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let index_is_imm = insn.bit(28);
    let mask_is_imm = insn.bit(29);
    let index_imm = insn.bits(20, 5) as u32;
    let mask_imm = insn.bits(34, 13) as u32;
    let mode = insn.bits(30, 2);
    let pred = Pred::p(insn.bits(48, 3) as u8);
    let index = if index_is_imm {
        Value::U32(index_imm)
    } else {
        t.get_reg20()
    };
    let mask = if mask_is_imm {
        Value::U32(mask_imm)
    } else {
        t.get_reg39()
    };

    let is_quad_candidate = mask_is_imm
        && mask_imm == QUAD_MASK
        && index_is_imm
        && t.env.stage() == ShaderStage::Fragment;
    if is_quad_candidate {
        let quad = match (mode, index_imm) {
            (0, 0..=3) => Some((Opcode::QuadBroadcast, index_imm)),
            (3, 1..=3) => Some((Opcode::QuadSwap, index_imm - 1)),
            _ => None,
        };
        if let Some((op, operand)) = quad {
            let value = t.x(insn.src_a_reg());
            let result = t.ir.inst(op, &[value, Value::U32(operand)]);
            t.set_x(insn.dest_reg(), result);
            t.ir.set_pred(pred, Value::U1(true));
            return Ok(());
        }
    }

    let value = t.x(insn.src_a_reg());
    let clamp =
        t.ir.bit_field_extract(mask, Value::U32(0), Value::U32(5), false);
    let seg_mask =
        t.ir.bit_field_extract(mask, Value::U32(8), Value::U32(5), false);
    let op = match mode {
        0 => Opcode::ShuffleIndex,
        1 => Opcode::ShuffleUp,
        2 => Opcode::ShuffleDown,
        _ => Opcode::ShuffleButterfly,
    };
    let result = t.ir.inst(op, &[value, index, clamp, seg_mask]);
    let in_bounds = t.ir.get_in_bounds_from_op(result);
    t.ir.set_pred(pred, in_bounds);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::exit_program::tests::translate_stage;
    use super::super::tests::{encode, translate};
    use super::QUAD_MASK;
    use crate::ShaderStage;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn shfl_down_registers() {
        let insn = encode(
            Opcode::Shfl,
            (1 << 48) | (2 << 30) | (6 << 39) | (5 << 20) | (4 << 8) | 3,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R5\n\
             \x20 %1 = GetRegister R6\n\
             \x20 %2 = GetRegister R4\n\
             \x20 %3 = BitFieldUExtract %1, #0x0, #0x5\n\
             \x20 %4 = BitFieldUExtract %1, #0x8, #0x5\n\
             \x20 %5 = ShuffleDown %2, %0, %3, %4\n\
             \x20 %6 = GetInBoundsFromOp %5\n\
             \x20 SetPred P1, %6\n\
             \x20 SetRegister R3, %5\n"
        );
    }

    #[test]
    fn shfl_butterfly_immediates() {
        let insn = encode(
            Opcode::Shfl,
            (7 << 48) | (0x1f << 34) | (3 << 30) | (1 << 29) | (1 << 28) | (1 << 20) | (2 << 8) | 2,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = BitFieldUExtract #0x1f, #0x0, #0x5\n\
             \x20 %2 = BitFieldUExtract #0x1f, #0x8, #0x5\n\
             \x20 %3 = ShuffleButterfly %0, #0x1, %1, %2\n\
             \x20 %4 = GetInBoundsFromOp %3\n\
             \x20 SetRegister R2, %3\n"
        );
    }

    fn quad(mode: u64, index: u64) -> u64 {
        encode(
            Opcode::Shfl,
            (u64::from(QUAD_MASK) << 34) | (mode << 30) | (3 << 28) | (index << 20) | (1 << 8) | 2,
        )
    }

    #[test]
    fn shfl_quad_broadcast_and_swap_in_fragment() {
        assert_eq!(
            translate_stage(&[quad(0, 2)], ShaderStage::Fragment, [0; 20]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R1\n\
             \x20 %1 = QuadBroadcast %0, #0x2\n\
             \x20 SetRegister R2, %1\n\
             \x20 SetPred P0, #true\n"
        );
        assert_eq!(
            translate_stage(&[quad(3, 2)], ShaderStage::Fragment, [0; 20]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R1\n\
             \x20 %1 = QuadSwap %0, #0x1\n\
             \x20 SetRegister R2, %1\n\
             \x20 SetPred P0, #true\n"
        );
    }

    #[test]
    fn shfl_quad_mask_outside_fragment_is_a_shuffle() {
        let dump = translate(&[quad(0, 2)]).unwrap();
        assert!(dump.contains("ShuffleIndex %0, #0x2"), "{dump}");
    }
}
