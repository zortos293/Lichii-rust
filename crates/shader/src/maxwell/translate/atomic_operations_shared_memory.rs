//! ATOMS: atomic operations on shared memory.

use super::{TResult, Translator};
use crate::ir::{Opcode, Value};

pub(super) fn atoms(t: &mut Translator<'_>) -> TResult {
    use Opcode::*;
    let insn = t.insn;
    let size = insn.bits(28, 2);
    if size == 3 {
        return Err(t.unsupported());
    }
    let (is_64, signed) = (size == 2, size == 1);
    let op = match insn.bits(52, 4) {
        8 if is_64 => SharedAtomicExchange64,
        _ if is_64 => return Err(t.unsupported()),
        0 => SharedAtomicIAdd32,
        1 if signed => SharedAtomicSMin32,
        1 => SharedAtomicUMin32,
        2 if signed => SharedAtomicSMax32,
        2 => SharedAtomicUMax32,
        3 => SharedAtomicInc32,
        4 => SharedAtomicDec32,
        5 => SharedAtomicAnd32,
        6 => SharedAtomicOr32,
        7 => SharedAtomicXor32,
        8 => SharedAtomicExchange32,
        _ => return Err(t.unsupported()),
    };
    let offset_reg = insn.src_a_reg();
    let offset = if offset_reg.is_zero() {
        Value::U32((insn.bits(30, 22) << 2) as u32)
    } else {
        let base = t.x(offset_reg);
        t.ir.iadd(base, Value::U32((insn.signed_bits(30, 22) << 2) as u32))
    };
    let operand = insn.src_b_reg();
    let dest = insn.dest_reg();
    if is_64 {
        let value = t.l(operand)?;
        let result = t.ir.inst(op, &[offset, value]);
        return t.set_l(dest, result);
    }
    let value = t.x(operand);
    let result = t.ir.inst(op, &[offset, value]);
    t.set_x(dest, result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    fn atoms(op: u64, size: u64, fields: u64) -> u64 {
        encode(Opcode::Atoms, (op << 52) | (size << 28) | fields)
    }

    #[test]
    fn atoms_s32_max_relative() {
        let insn = atoms(2, 1, (0x3fffff << 30) | (4 << 20) | (2 << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = IAdd32 %0, #0xfffffffc\n\
             \x20 %2 = GetRegister R4\n\
             \x20 %3 = SharedAtomicSMax32 %1, %2\n\
             \x20 SetRegister R1, %3\n"
        );
    }

    #[test]
    fn atoms_u32_add_absolute() {
        let insn = atoms(0, 0, (3 << 30) | (4 << 20) | (0xff << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = SharedAtomicIAdd32 #0xc, %0\n\
             \x20 SetRegister R1, %1\n"
        );
    }

    #[test]
    fn atoms_u64_exchange() {
        let insn = atoms(8, 2, (4 << 20) | (0xff << 8) | 2);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = SharedAtomicExchange64 #0x0, %3\n\
             \x20 %5 = UnpackUint2x32 %4\n\
             \x20 %6 = CompositeExtractU32x2 %5, #0x0\n\
             \x20 SetRegister R2, %6\n\
             \x20 %7 = CompositeExtractU32x2 %5, #0x1\n\
             \x20 SetRegister R3, %7\n"
        );
    }

    #[test]
    fn atoms_invalid_encodings_are_unsupported() {
        assert!(translate(&[atoms(0, 2, 0)]).is_err());
        assert!(translate(&[atoms(0, 3, 0)]).is_err());
        assert!(translate(&[atoms(9, 0, 0)]).is_err());
        assert!(translate(&[atoms(8, 2, 1 << 20)]).is_err());
    }
}
