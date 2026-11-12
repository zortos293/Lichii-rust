//! ISBERD (internal stage buffer entry read) and the Switch 2 `ISBE` pseudo instruction.

use super::{TResult, Translator};
use crate::ShaderStage;
use crate::ir::{Attribute, Opcode, Patch, Value};

fn lane_bytes(t: &mut Translator<'_>, shift: u32) -> Value {
    let lane = t.ir.lane_id();
    if shift == 0 {
        return lane;
    }
    t.ir.shift_left_logical(lane, Value::U32(shift))
}

pub(super) fn isberd(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let (dest, src) = (insn.dest_reg(), insn.src_a_reg());
    let imm = Value::U32(insn.bits(24, 8) as u32);
    let skew = insn.bit(31);

    let mut index = if src.is_zero() {
        imm
    } else {
        let shift = match insn.bits(47, 2) {
            0 => 0,
            1 => 1,
            2 => 2,
            _ => return Err(t.unsupported()),
        };
        let value = t.x(src);
        let scaled = if shift == 0 {
            value
        } else {
            t.ir.shift_left_logical(value, Value::U32(shift))
        };
        t.ir.iadd(scaled, imm)
    };

    if insn.bit(32) {
        let (op, lane_shift) = match insn.bits(36, 4) {
            0 => (Opcode::LoadGlobalU8, 0),
            1 => (Opcode::LoadGlobalU16, 1),
            2 | 3 => (Opcode::LoadGlobal32, 2),
            _ => return Err(t.unsupported()),
        };
        if skew {
            let lane = lane_bytes(t, lane_shift);
            index = t.ir.iadd(index, lane);
        }
        let address = t.ir.uconvert(64, index);
        let value = t.ir.inst(op, &[address]);
        t.set_x(dest, value);
        return Ok(());
    }

    let mode = insn.bits(33, 2);
    if mode != 0 {
        if skew {
            let lane = lane_bytes(t, 2);
            index = t.ir.iadd(index, lane);
        }
        let value = match (mode, index) {
            (1, Value::U32(n)) if n < u32::from(Patch::COMPONENT_0.0 + Patch::NUM_COMPONENTS) => {
                t.ir.get_patch(Patch(n as u8))
            }
            (2, Value::U32(n)) if n <= u32::from(u16::MAX) => {
                t.ir.get_attribute(Attribute(n as u16), Value::U32(0))
            }
            (3, _) => t.ir.get_attribute_indexed(index, Value::U32(0)),
            _ => return Err(t.unsupported()),
        };
        let bits = t.ir.bitcast_u32(value);
        t.set_x(dest, bits);
        return Ok(());
    }

    let value = t.x(src);
    let value = if skew {
        let lane = t.ir.lane_id();
        t.ir.iadd(value, lane)
    } else {
        value
    };
    t.set_x(dest, value);
    Ok(())
}

/// Stage buffer access of Switch 2 task and mesh programs, lowered from Ampere ISBERD/ISBEWR.
/// Addresses are linear bytes: reads come from the input attribute region, writes go to the
/// output attribute or map region.
pub(super) fn isbe_nx2(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if !matches!(t.env.stage(), ShaderStage::Task | ShaderStage::Mesh) {
        return Err(t.unsupported());
    }
    let (is_32, attribute, write) = (insn.bit(44), insn.bit(45), insn.bit(48));
    if write && attribute && !is_32 {
        return Err(t.unsupported());
    }
    let base = t.x(insn.src_a_reg());
    let address = t.ir.iadd(base, Value::U32(insn.bits(20, 24) as u32));
    let reg = insn.dest_reg();
    if !write {
        let value = t.ir.inst(Opcode::IsbeRead, &[address]);
        t.set_x(reg, value);
        return Ok(());
    }
    let op = match (attribute, is_32) {
        (true, _) => Opcode::IsbeWriteAttribute,
        (false, true) => Opcode::IsbeWriteMapU32,
        (false, false) => Opcode::IsbeWriteMapU8,
    };
    let value = t.x(reg);
    t.ir.inst(op, &[address, value]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::exit_program::tests::translate_stage;
    use super::super::tests::{encode, translate};
    use crate::ShaderStage;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn isberd_copies_without_mode() {
        let insn = encode(Opcode::Isberd, (2 << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0x0\n\
             \x20 %2 = GetRegister R2\n\
             \x20 SetRegister R1, %2\n"
        );
    }

    #[test]
    fn isberd_skew_adds_lane() {
        let insn = encode(Opcode::Isberd, (1 << 31) | (0xff << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n  %0 = LaneId\n  %1 = IAdd32 #0x0, %0\n  SetRegister R1, %1\n"
        );
    }

    #[test]
    fn isberd_global_u16_scaled_and_skewed() {
        let insn = encode(
            Opcode::Isberd,
            (1 << 47) | (1 << 36) | (1 << 32) | (1 << 31) | (4 << 24) | (3 << 8) | 1,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R3\n\
             \x20 %1 = ShiftLeftLogical32 %0, #0x1\n\
             \x20 %2 = IAdd32 %1, #0x84\n\
             \x20 %3 = LaneId\n\
             \x20 %4 = ShiftLeftLogical32 %3, #0x1\n\
             \x20 %5 = IAdd32 %2, %4\n\
             \x20 %6 = ConvertU64U32 %5\n\
             \x20 %7 = LoadGlobalU16 %6\n\
             \x20 SetRegister R1, %7\n"
        );
    }

    #[test]
    fn isberd_attribute_modes() {
        let prim = encode(Opcode::Isberd, (2 << 33) | (28 << 24) | (0xff << 8) | 1);
        assert_eq!(
            translate(&[prim]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetAttribute PositionX, #0x0\n\
             \x20 %1 = BitCastU32F32 %0\n\
             \x20 SetRegister R1, %1\n"
        );
        let patch = encode(Opcode::Isberd, (1 << 33) | (7 << 24) | (0xff << 8) | 1);
        assert!(translate(&[patch]).unwrap().contains("GetPatch Patch1"));
        let indexed = encode(Opcode::Isberd, (3 << 33) | (2 << 8) | 1);
        assert!(
            translate(&[indexed])
                .unwrap()
                .contains("GetAttributeIndexed %1, #0x0")
        );
    }

    #[test]
    fn isberd_invalid_forms_are_unsupported() {
        let dynamic_patch = encode(Opcode::Isberd, (1 << 33) | (2 << 8) | 1);
        let bad_shift = encode(Opcode::Isberd, (3 << 47) | (2 << 8));
        let bad_size = encode(Opcode::Isberd, (4 << 36) | (1 << 32));
        let patch_out_of_range = encode(Opcode::Isberd, (1 << 33) | (200 << 24) | (0xff << 8));
        for insn in [dynamic_patch, bad_shift, bad_size, patch_out_of_range] {
            assert!(translate(&[insn]).is_err(), "{insn:#x}");
        }
    }

    #[test]
    fn isbe_nx2_read_and_writes() {
        let read = encode(Opcode::IsbeNx2, (0x10 << 20) | (2 << 8) | 1);
        assert_eq!(
            translate_stage(&[read], ShaderStage::Mesh, [0; 20]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0x10\n\
             \x20 %2 = IsbeRead %1\n\
             \x20 SetRegister R1, %2\n"
        );
        let attribute = encode(Opcode::IsbeNx2, (1 << 48) | (3 << 44) | (2 << 8) | 1);
        assert_eq!(
            translate_stage(&[attribute], ShaderStage::Task, [0; 20]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0x0\n\
             \x20 %2 = GetRegister R1\n\
             \x20 IsbeWriteAttribute %1, %2\n"
        );
        let map_u8 = encode(Opcode::IsbeNx2, (1 << 48) | (0xff << 8) | 1);
        assert!(
            translate_stage(&[map_u8], ShaderStage::Mesh, [0; 20])
                .unwrap()
                .contains("IsbeWriteMapU8 %0, %1")
        );
    }

    #[test]
    fn isbe_nx2_invalid_forms_are_unsupported() {
        assert!(translate(&[encode(Opcode::IsbeNx2, 0)]).is_err());
        let u8_attribute = encode(Opcode::IsbeNx2, (1 << 48) | (1 << 45));
        assert!(translate_stage(&[u8_attribute], ShaderStage::Mesh, [0; 20]).is_err());
    }
}
