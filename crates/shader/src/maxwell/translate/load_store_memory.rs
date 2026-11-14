//! LDG and STG: global memory loads and stores through 32- or 64-bit pointers.

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{Opcode, Reg, Value};

/// Whether `reg` starts a group of `align` registers; RZ always does.
pub(super) fn is_aligned(reg: Reg, align: u16) -> bool {
    reg.is_zero() || reg.0.is_multiple_of(align)
}

/// The 64-bit address of a LDG/STG: `Ra` (or the pair `Ra:Ra+1` with `.E`) plus a signed
/// 24-bit offset, which is unsigned when `Ra` is RZ.
fn address(t: &mut Translator<'_>) -> Result<Value, TranslateError> {
    let reg = t.insn.src_a_reg();
    let base = if t.insn.bit(45) {
        t.l(reg)?
    } else {
        let low = t.x(reg);
        t.ir.uconvert(64, low)
    };
    let offset = if reg.is_zero() {
        t.insn.bits(20, 24)
    } else {
        t.insn.signed_bits(20, 24) as u64
    };
    Ok(t.ir.iadd(base, Value::U64(offset)))
}

pub(super) fn ldg(t: &mut Translator<'_>) -> TResult {
    let dest = t.insn.dest_reg();
    let (op, count) = match t.insn.bits(48, 3) {
        0 => (Opcode::LoadGlobalU8, 1),
        1 => (Opcode::LoadGlobalS8, 1),
        2 => (Opcode::LoadGlobalU16, 1),
        3 => (Opcode::LoadGlobalS16, 1),
        4 => (Opcode::LoadGlobal32, 1),
        5 => (Opcode::LoadGlobal64, 2),
        6 => (Opcode::LoadGlobal128, 4),
        _ => {
            crate::fallback::note("LDG", "size_7_as_128");
            (Opcode::LoadGlobal128, 4)
        }
    };
    if !is_aligned(dest, count) {
        return Err(t.unsupported());
    }
    let address = address(t)?;
    let value = t.ir.inst(op, &[address]);
    if count == 1 {
        t.set_x(dest, value);
        return Ok(());
    }
    for i in 0..count {
        let element = t.ir.composite_extract(value, u32::from(i));
        t.set_x(dest.offset(i), element);
    }
    Ok(())
}

pub(super) fn stg(t: &mut Translator<'_>) -> TResult {
    let data = t.insn.dest_reg();
    let (op, count) = match t.insn.bits(48, 3) {
        0 => (Opcode::WriteGlobalU8, 1),
        1 => (Opcode::WriteGlobalS8, 1),
        2 => (Opcode::WriteGlobalU16, 1),
        3 => (Opcode::WriteGlobalS16, 1),
        4 => (Opcode::WriteGlobal32, 1),
        5 => (Opcode::WriteGlobal64, 2),
        6 => (Opcode::WriteGlobal128, 4),
        _ => return Err(t.unsupported()),
    };
    if !is_aligned(data, count) {
        return Err(t.unsupported());
    }
    let address = address(t)?;
    let value = match count {
        1 => t.x(data),
        2 => {
            let words = [t.x(data), t.x(data.offset(1))];
            t.ir.composite_construct(&words)
        }
        _ => {
            let words = [
                t.x(data),
                t.x(data.offset(1)),
                t.x(data.offset(2)),
                t.x(data.offset(3)),
            ];
            t.ir.composite_construct(&words)
        }
    };
    t.ir.inst(op, &[address, value]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn ldg_s16_32_bit_pointer() {
        let insn = encode(Opcode::Ldg, (3 << 48) | (0xfffff0 << 20) | (4 << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = ConvertU64U32 %0\n\
             \x20 %2 = IAdd64 %1, #0xfffffffffffffff0u64\n\
             \x20 %3 = LoadGlobalS16 %2\n\
             \x20 SetRegister R1, %3\n"
        );
    }

    #[test]
    fn ldg_b64_extended_pointer() {
        let insn = encode(
            Opcode::Ldg,
            (5 << 48) | (1 << 45) | (8 << 20) | (2 << 8) | 4,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x8u64\n\
             \x20 %5 = LoadGlobal64 %4\n\
             \x20 %6 = CompositeExtractU32x2 %5, #0x0\n\
             \x20 SetRegister R4, %6\n\
             \x20 %7 = CompositeExtractU32x2 %5, #0x1\n\
             \x20 SetRegister R5, %7\n"
        );
    }

    #[test]
    fn ldg_rz_offset_is_absolute() {
        let insn = encode(Opcode::Ldg, (4 << 48) | (0xfffff0 << 20) | (0xff << 8) | 1);
        let dump = translate(&[insn]).unwrap();
        assert!(dump.contains("IAdd64 %0, #0xfffff0u64"), "{dump}");
    }

    #[test]
    fn ldg_misaligned_registers_are_unsupported() {
        assert!(translate(&[encode(Opcode::Ldg, (6 << 48) | (0xff << 8) | 2)]).is_err());
        assert!(translate(&[encode(Opcode::Ldg, (1 << 45) | (3 << 8))]).is_err());
    }

    #[test]
    fn stg_b128() {
        let insn = encode(Opcode::Stg, (6 << 48) | (0x10 << 20) | (0xff << 8) | 8);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = ConvertU64U32 #0x0\n\
             \x20 %1 = IAdd64 %0, #0x10u64\n\
             \x20 %2 = GetRegister R8\n\
             \x20 %3 = GetRegister R9\n\
             \x20 %4 = GetRegister R10\n\
             \x20 %5 = GetRegister R11\n\
             \x20 %6 = CompositeConstructU32x4 %2, %3, %4, %5\n\
             \x20 WriteGlobal128 %1, %6\n"
        );
    }

    #[test]
    fn stg_u8() {
        let insn = encode(Opcode::Stg, (6 << 8) | 7);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R6\n\
             \x20 %1 = ConvertU64U32 %0\n\
             \x20 %2 = IAdd64 %1, #0x0u64\n\
             \x20 %3 = GetRegister R7\n\
             \x20 WriteGlobalU8 %2, %3\n"
        );
    }

    #[test]
    fn stg_reserved_size_is_unsupported() {
        assert!(translate(&[encode(Opcode::Stg, 7 << 48)]).is_err());
    }
}
