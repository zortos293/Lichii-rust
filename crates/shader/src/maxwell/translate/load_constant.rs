//! LDC: constant buffer loads with a register offset.

use super::load_store_memory::is_aligned;
use super::{TResult, Translator};
use crate::ir::Value;

pub(super) fn ldc(t: &mut Translator<'_>) -> TResult {
    let dest = t.insn.dest_reg();
    let (bits, signed) = match t.insn.bits(48, 3) {
        0 => (8, false),
        1 => (8, true),
        2 => (16, false),
        3 => (16, true),
        4 => (32, false),
        5 => (64, false),
        _ => return Err(t.unsupported()),
    };
    if bits == 64 && !is_aligned(dest, 2) {
        return Err(t.unsupported());
    }
    let mode = t.insn.bits(44, 2);
    if mode != 0 && mode != 2 {
        return Err(t.unsupported());
    }
    let imm_index = Value::U32(t.insn.bits(36, 5) as u32);
    let imm_offset = Value::U32(t.insn.signed_bits(20, 16) as u32);
    let reg = t.x(t.insn.src_a_reg());
    let address = t.ir.iadd(reg, imm_offset);
    let (index, offset) = if mode == 0 {
        (imm_index, address)
    } else {
        let index =
            t.ir.bit_field_extract(address, Value::U32(16), Value::U32(16), false);
        let offset =
            t.ir.bit_field_extract(address, Value::U32(0), Value::U32(16), false);
        (t.ir.iadd(index, imm_index), offset)
    };
    let value = t.ir.get_cbuf_sized(index, offset, bits, signed);
    if bits != 64 {
        t.set_x(dest, value);
        return Ok(());
    }
    for i in 0..2 {
        let element = t.ir.composite_extract(value, u32::from(i));
        t.set_x(dest.offset(i), element);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn ldc_s8_default_mode() {
        let insn = encode(
            Opcode::Ldc,
            (1 << 48) | (3 << 36) | (0xfffc << 20) | (5 << 8) | 1,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R5\n\
             \x20 %1 = IAdd32 %0, #0xfffffffc\n\
             \x20 %2 = GetCbufS8 #0x3, %1\n\
             \x20 SetRegister R1, %2\n"
        );
    }

    #[test]
    fn ldc_b64_segmented_mode() {
        let insn = encode(
            Opcode::Ldc,
            (5 << 48) | (2 << 44) | (1 << 36) | (8 << 20) | (3 << 8) | 6,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R3\n\
             \x20 %1 = IAdd32 %0, #0x8\n\
             \x20 %2 = BitFieldUExtract %1, #0x10, #0x10\n\
             \x20 %3 = BitFieldUExtract %1, #0x0, #0x10\n\
             \x20 %4 = IAdd32 %2, #0x1\n\
             \x20 %5 = GetCbufU32x2 %4, %3\n\
             \x20 %6 = CompositeExtractU32x2 %5, #0x0\n\
             \x20 SetRegister R6, %6\n\
             \x20 %7 = CompositeExtractU32x2 %5, #0x1\n\
             \x20 SetRegister R7, %7\n"
        );
    }

    #[test]
    fn ldc_invalid_forms_are_unsupported() {
        assert!(translate(&[encode(Opcode::Ldc, 6 << 48)]).is_err());
        assert!(translate(&[encode(Opcode::Ldc, (4 << 48) | (1 << 44))]).is_err());
        assert!(translate(&[encode(Opcode::Ldc, (5 << 48) | 1)]).is_err());
    }
}
