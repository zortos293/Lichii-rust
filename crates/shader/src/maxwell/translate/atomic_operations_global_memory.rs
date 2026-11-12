//! ATOM, ATOM.CAS and RED: atomic operations on global memory.

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{FmzMode, FpControl, FpRounding, Opcode, Reg, Type, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomOp {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exch,
    SafeAdd,
}

impl AtomOp {
    const fn from_bits(bits: u64) -> Option<AtomOp> {
        Some(match bits {
            0 => AtomOp::Add,
            1 => AtomOp::Min,
            2 => AtomOp::Max,
            3 => AtomOp::Inc,
            4 => AtomOp::Dec,
            5 => AtomOp::And,
            6 => AtomOp::Or,
            7 => AtomOp::Xor,
            8 => AtomOp::Exch,
            9 => AtomOp::SafeAdd,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomSize {
    U32,
    S32,
    U64,
    F32,
    F16x2,
    S64,
}

impl AtomSize {
    const fn from_bits(bits: u64) -> Option<AtomSize> {
        Some(match bits {
            0 => AtomSize::U32,
            1 => AtomSize::S32,
            2 => AtomSize::U64,
            3 => AtomSize::F32,
            4 => AtomSize::F16x2,
            5 => AtomSize::S64,
            _ => return None,
        })
    }

    const fn is_64(self) -> bool {
        matches!(self, AtomSize::U64 | AtomSize::S64)
    }
}

/// Address of an atomic: `Ra` (or `Ra:Ra+1` with `.E`) plus a signed 20-bit offset, which is
/// unsigned when `Ra` is RZ.
fn atom_offset(t: &mut Translator<'_>) -> Result<Value, TranslateError> {
    let reg = t.insn.src_a_reg();
    let base = if t.insn.bit(48) {
        t.l(reg)?
    } else {
        let low = t.x(reg);
        t.ir.uconvert(64, low)
    };
    let offset = if reg.is_zero() {
        t.insn.bits(28, 20)
    } else {
        t.insn.signed_bits(28, 20) as u64
    };
    Ok(t.ir.iadd(base, Value::U64(offset)))
}

/// Combinations the hardware treats as a plain load of the old value.
fn op_not_applicable(size: AtomSize, op: Option<AtomOp>) -> bool {
    use AtomOp::*;
    match size {
        AtomSize::U32 | AtomSize::S32 | AtomSize::U64 => matches!(op, Some(Inc | Dec)),
        AtomSize::S64 => matches!(op, Some(Add | Inc | Dec)),
        AtomSize::F32 => op != Some(Add),
        AtomSize::F16x2 => !matches!(op, Some(Add | Min | Max)),
    }
}

fn integer_opcode(size: AtomSize, op: Option<AtomOp>) -> Option<Opcode> {
    use Opcode::*;
    let signed = matches!(size, AtomSize::S32 | AtomSize::S64);
    let [narrow, wide] = match op? {
        AtomOp::Add => [GlobalAtomicIAdd32, GlobalAtomicIAdd64],
        AtomOp::Min if signed => [GlobalAtomicSMin32, GlobalAtomicSMin64],
        AtomOp::Min => [GlobalAtomicUMin32, GlobalAtomicUMin64],
        AtomOp::Max if signed => [GlobalAtomicSMax32, GlobalAtomicSMax64],
        AtomOp::Max => [GlobalAtomicUMax32, GlobalAtomicUMax64],
        AtomOp::And => [GlobalAtomicAnd32, GlobalAtomicAnd64],
        AtomOp::Or => [GlobalAtomicOr32, GlobalAtomicOr64],
        AtomOp::Xor => [GlobalAtomicXor32, GlobalAtomicXor64],
        AtomOp::Exch => [GlobalAtomicExchange32, GlobalAtomicExchange64],
        AtomOp::Inc | AtomOp::Dec | AtomOp::SafeAdd => return None,
    };
    Some(if size.is_64() { wide } else { narrow })
}

fn apply(
    t: &mut Translator<'_>,
    operand: Reg,
    offset: Value,
    size: AtomSize,
    op: Option<AtomOp>,
) -> Result<Value, TranslateError> {
    const F16_CONTROL: FpControl = FpControl {
        no_contraction: false,
        rounding: FpRounding::Rn,
        fmz_mode: FmzMode::DontCare,
    };
    const F32_CONTROL: FpControl = FpControl {
        no_contraction: false,
        rounding: FpRounding::Rn,
        fmz_mode: FmzMode::Ftz,
    };
    match size {
        AtomSize::F32 => {
            let value = t.f(operand);
            Ok(t.ir.inst_flags(
                Opcode::GlobalAtomicAddF32,
                F32_CONTROL.to_flags(),
                &[offset, value],
            ))
        }
        AtomSize::F16x2 => {
            let op = match op {
                Some(AtomOp::Add) => Opcode::GlobalAtomicAddF16x2,
                Some(AtomOp::Min) => Opcode::GlobalAtomicMinF16x2,
                _ => Opcode::GlobalAtomicMaxF16x2,
            };
            let packed = t.x(operand);
            let value = t.ir.unpack_float_2x16(packed);
            Ok(t.ir.inst_flags(op, F16_CONTROL.to_flags(), &[offset, value]))
        }
        _ => {
            let Some(opcode) = integer_opcode(size, op) else {
                return Err(t.unsupported());
            };
            let value = if size.is_64() {
                t.l(operand)?
            } else {
                t.x(operand)
            };
            Ok(t.ir.inst(opcode, &[offset, value]))
        }
    }
}

fn global_atomic(
    t: &mut Translator<'_>,
    dest: Reg,
    operand: Reg,
    size_bits: u64,
    op_bits: u64,
) -> TResult {
    let Some(size) = AtomSize::from_bits(size_bits) else {
        return Err(t.unsupported());
    };
    let op = AtomOp::from_bits(op_bits);
    let offset = atom_offset(t)?;
    let result = if !op_not_applicable(size, op) {
        apply(t, operand, offset, size, op)?
    } else if size.is_64() {
        let words = t.ir.inst(Opcode::LoadGlobal64, &[offset]);
        t.ir.pack_uint_2x32(words)
    } else {
        t.ir.inst(Opcode::LoadGlobal32, &[offset])
    };
    if dest.is_zero() {
        return Ok(());
    }
    match t.ir.ty(result) {
        ty if ty == Type::U64 => t.set_l(dest, result)?,
        ty if ty == Type::F32 => t.set_f(dest, result),
        _ => t.set_x(dest, result),
    }
    Ok(())
}

pub(super) fn red(t: &mut Translator<'_>) -> TResult {
    let operand = t.insn.dest_reg();
    let (size, op) = (t.insn.bits(20, 3), t.insn.bits(23, 3));
    global_atomic(t, Reg::RZ, operand, size, op)
}

/// `ATOM.CAS Rd, [Ra + imm], Rb`: compares the word with `Rb` and stores `Rb+1` on a match;
/// `Rd` receives the old word.
pub(super) fn atom_cas(t: &mut Translator<'_>) -> TResult {
    if t.insn.bits(49, 3) != 0 {
        return Err(t.unsupported());
    }
    let compare_reg = t.insn.src_b_reg();
    let offset = atom_offset(t)?;
    let compare = t.x(compare_reg);
    let value = t.x(compare_reg.offset(1));
    let old = t.ir.inst(
        Opcode::GlobalAtomicCompareExchange32,
        &[offset, compare, value],
    );
    t.set_x(t.insn.dest_reg(), old);
    Ok(())
}

pub(super) fn atom(t: &mut Translator<'_>) -> TResult {
    let (dest, operand) = (t.insn.dest_reg(), t.insn.src_b_reg());
    let (size, op) = (t.insn.bits(49, 3), t.insn.bits(52, 4));
    global_atomic(t, dest, operand, size, op)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    fn atom(size: u64, op: u64, fields: u64) -> u64 {
        encode(Opcode::Atom, (op << 52) | (size << 49) | fields)
    }

    #[test]
    fn atom_s32_min() {
        let insn = atom(1, 1, (0xffffc << 28) | (3 << 20) | (2 << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = ConvertU64U32 %0\n\
             \x20 %2 = IAdd64 %1, #0xfffffffffffffffcu64\n\
             \x20 %3 = GetRegister R3\n\
             \x20 %4 = GlobalAtomicSMin32 %2, %3\n\
             \x20 SetRegister R1, %4\n"
        );
    }

    #[test]
    fn atom_u64_exch_extended() {
        let insn = atom(2, 8, (1 << 48) | (6 << 20) | (2 << 8) | 4);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x0u64\n\
             \x20 %5 = GetRegister R6\n\
             \x20 %6 = GetRegister R7\n\
             \x20 %7 = CompositeConstructU32x2 %5, %6\n\
             \x20 %8 = PackUint2x32 %7\n\
             \x20 %9 = GlobalAtomicExchange64 %4, %8\n\
             \x20 %10 = UnpackUint2x32 %9\n\
             \x20 %11 = CompositeExtractU32x2 %10, #0x0\n\
             \x20 SetRegister R4, %11\n\
             \x20 %12 = CompositeExtractU32x2 %10, #0x1\n\
             \x20 SetRegister R5, %12\n"
        );
    }

    #[test]
    fn atom_f32_add() {
        let insn = atom(3, 0, (0x40 << 28) | (5 << 20) | (0xff << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = ConvertU64U32 #0x0\n\
             \x20 %1 = IAdd64 %0, #0x40u64\n\
             \x20 %2 = GetRegister R5\n\
             \x20 %3 = BitCastF32U32 %2\n\
             \x20 %4 = GlobalAtomicAddF32 %1, %3 (flags 0x12)\n\
             \x20 %5 = BitCastU32F32 %4\n\
             \x20 SetRegister R1, %5\n"
        );
    }

    #[test]
    fn atom_f16x2_max() {
        let insn = atom(4, 2, (5 << 20) | (0xff << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = ConvertU64U32 #0x0\n\
             \x20 %1 = IAdd64 %0, #0x0u64\n\
             \x20 %2 = GetRegister R5\n\
             \x20 %3 = UnpackFloat2x16 %2\n\
             \x20 %4 = GlobalAtomicMaxF16x2 %1, %3 (flags 0x2)\n\
             \x20 SetRegister R1, %4\n"
        );
    }

    #[test]
    fn atom_inapplicable_ops_load_the_old_value() {
        let u32_inc = atom(0, 3, (0xff << 8) | 1);
        assert_eq!(
            translate(&[u32_inc]).unwrap(),
            "{0}:\n\
             \x20 %0 = ConvertU64U32 #0x0\n\
             \x20 %1 = IAdd64 %0, #0x0u64\n\
             \x20 %2 = LoadGlobal32 %1\n\
             \x20 SetRegister R1, %2\n"
        );
        let s64_add = atom(5, 0, (0xff << 8) | 2);
        assert_eq!(
            translate(&[s64_add]).unwrap(),
            "{0}:\n\
             \x20 %0 = ConvertU64U32 #0x0\n\
             \x20 %1 = IAdd64 %0, #0x0u64\n\
             \x20 %2 = LoadGlobal64 %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = UnpackUint2x32 %3\n\
             \x20 %5 = CompositeExtractU32x2 %4, #0x0\n\
             \x20 SetRegister R2, %5\n\
             \x20 %6 = CompositeExtractU32x2 %4, #0x1\n\
             \x20 SetRegister R3, %6\n"
        );
        let f32_min = atom(3, 1, (0xff << 8) | 1);
        assert!(
            translate(&[f32_min])
                .unwrap()
                .contains("%2 = LoadGlobal32 %1\n  SetRegister R1, %2")
        );
    }

    #[test]
    fn atom_invalid_encodings_are_unsupported() {
        assert!(translate(&[atom(6, 0, 0)]).is_err());
        assert!(translate(&[atom(0, 9, 0)]).is_err());
        assert!(translate(&[atom(0, 12, 0)]).is_err());
        assert!(translate(&[atom(2, 0, 1 << 20)]).is_err());
    }

    #[test]
    fn red_u32_add_discards_result() {
        let insn = encode(Opcode::Red, (0x10 << 28) | (4 << 8) | 3);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = ConvertU64U32 %0\n\
             \x20 %2 = IAdd64 %1, #0x10u64\n\
             \x20 %3 = GetRegister R3\n\
             \x20 %4 = GlobalAtomicIAdd32 %2, %3\n"
        );
    }

    #[test]
    fn red_u32_xor() {
        let insn = encode(Opcode::Red, (7 << 23) | (0xff << 8) | 3);
        assert!(
            translate(&[insn])
                .unwrap()
                .contains("GlobalAtomicXor32 %1, %2")
        );
    }

    #[test]
    fn atom_cas_u32() {
        let insn = encode(Opcode::AtomCas, (8 << 28) | (6 << 20) | (2 << 8) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = ConvertU64U32 %0\n\
             \x20 %2 = IAdd64 %1, #0x8u64\n\
             \x20 %3 = GetRegister R6\n\
             \x20 %4 = GetRegister R7\n\
             \x20 %5 = GlobalAtomicCompareExchange32 %2, %3, %4\n\
             \x20 SetRegister R1, %5\n"
        );
        assert!(translate(&[encode(Opcode::AtomCas, 1 << 49)]).is_err());
    }
}
