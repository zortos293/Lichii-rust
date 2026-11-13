//! SHF: funnel shift of a 64-bit register pair, ported from integer_funnel_shift.cpp.

use super::{TResult, Translator};
use crate::ir::Value;

const MAX_SHIFT_U32: u64 = 0;
const MAX_SHIFT_UNDEFINED: u64 = 1;
const MAX_SHIFT_S64: u64 = 3;

fn shf(t: &mut Translator<'_>, shift: Value, high_bits: Value, right_shift: bool) -> TResult {
    let insn = t.insn;
    let max_shift_mode = insn.bits(37, 2);
    if insn.bit(47) || insn.bits(48, 2) != 0 || max_shift_mode == MAX_SHIFT_UNDEFINED {
        return Err(t.unsupported());
    }
    let low_bits = t.x(insn.src_a_reg());
    let pair = t.ir.composite_construct(&[low_bits, high_bits]);
    let packed_int = t.ir.pack_uint_2x32(pair);
    let max_shift = Value::U32(if max_shift_mode == MAX_SHIFT_U32 {
        32
    } else {
        63
    });
    let safe_shift = if insn.bit(50) {
        let mask = if max_shift_mode == MAX_SHIFT_U32 {
            31
        } else {
            63
        };
        t.ir.bitwise_and(shift, Value::U32(mask))
    } else {
        t.ir.imin(shift, max_shift, false)
    };
    let shifted_value = if !right_shift {
        t.ir.shift_left_logical(packed_int, safe_shift)
    } else if max_shift_mode == MAX_SHIFT_S64 {
        t.ir.shift_right_arithmetic(packed_int, safe_shift)
    } else {
        t.ir.shift_right_logical(packed_int, safe_shift)
    };
    let unpacked_value = t.ir.unpack_uint_2x32(shifted_value);
    let result =
        t.ir.composite_extract(unpacked_value, if right_shift { 0 } else { 1 });
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn shf_lreg(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_reg20();
    let high_bits = t.get_reg39();
    shf(t, shift, high_bits, false)
}

pub(super) fn shf_rreg(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_reg20();
    let high_bits = t.get_reg39();
    shf(t, shift, high_bits, true)
}

pub(super) fn shf_limm(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_imm20();
    let high_bits = t.get_reg39();
    shf(t, shift, high_bits, false)
}

pub(super) fn shf_rimm(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_imm20();
    let high_bits = t.get_reg39();
    shf(t, shift, high_bits, true)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn shf_l_reg() {
        let insn = encode(Opcode::ShfLReg, (1 << 8) | (2 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = CompositeConstructU32x2 %2, %1
  %4 = PackUint2x32 %3
  %5 = UMin32 %0, #0x20
  %6 = ShiftLeftLogical64 %4, %5
  %7 = UnpackUint2x32 %6
  %8 = CompositeExtractU32x2 %7, #0x1
  SetRegister R0, %8
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shf_r_reg_u64() {
        let insn = encode(
            Opcode::ShfRReg,
            (1 << 8) | (2 << 20) | (3 << 39) | (2 << 37),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = CompositeConstructU32x2 %2, %1
  %4 = PackUint2x32 %3
  %5 = UMin32 %0, #0x3f
  %6 = ShiftRightLogical64 %4, %5
  %7 = UnpackUint2x32 %6
  %8 = CompositeExtractU32x2 %7, #0x0
  SetRegister R0, %8
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shf_l_imm() {
        let insn = encode(Opcode::ShfLImm, (1 << 8) | (5 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetRegister R1
  %2 = CompositeConstructU32x2 %1, %0
  %3 = PackUint2x32 %2
  %4 = UMin32 #0x5, #0x20
  %5 = ShiftLeftLogical64 %3, %4
  %6 = UnpackUint2x32 %5
  %7 = CompositeExtractU32x2 %6, #0x1
  SetRegister R0, %7
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shf_r_imm_s64_wrap() {
        let insn = encode(
            Opcode::ShfRImm,
            (1 << 8) | (5 << 20) | (3 << 39) | (3 << 37) | (1 << 50),
        );
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetRegister R1
  %2 = CompositeConstructU32x2 %1, %0
  %3 = PackUint2x32 %2
  %4 = BitwiseAnd32 #0x60005, #0x3f
  %5 = ShiftRightArithmetic64 %3, %4
  %6 = UnpackUint2x32 %5
  %7 = CompositeExtractU32x2 %6, #0x0
  SetRegister R0, %7
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shf_undefined_max_shift() {
        let insn = encode(Opcode::ShfLReg, 1 << 37);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
