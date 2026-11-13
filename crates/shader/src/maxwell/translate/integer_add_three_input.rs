//! IADD3: three-input integer add, ported from integer_add_three_input.cpp.

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{ICompare, Value};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shift {
    None,
    Right,
    Left,
}

fn integer_half(t: &mut Translator<'_>, value: Value, half: u64) -> Result<Value, TranslateError> {
    match half {
        0 => Ok(value),
        1 => {
            Ok(t.ir.bit_field_extract(value, Value::U32(0), Value::U32(16), false))
        }
        2 => {
            Ok(t.ir.bit_field_extract(value, Value::U32(16), Value::U32(16), false))
        }
        _ => Err(t.unsupported()),
    }
}

fn integer_shift(t: &mut Translator<'_>, value: Value, shift: Shift) -> Value {
    match shift {
        Shift::None => value,
        Shift::Right => {
            let edge_case = t.ir.get_carry_from_op(value);
            let shifted = t.ir.shift_right_logical(value, Value::U32(16));
            let adjusted = t.ir.iadd(shifted, Value::U32(0x10000));
            t.ir.select(edge_case, adjusted, shifted)
        }
        Shift::Left => t.ir.shift_left_logical(value, Value::U32(16)),
    }
}

fn iadd3(t: &mut Translator<'_>, op_a: Value, op_b: Value, op_c: Value, shift: Shift) -> TResult {
    let insn = t.insn;
    let x = insn.bit(48);
    if x && shift == Shift::Right {
        return Err(t.unsupported());
    }
    let op_a = if insn.bit(51) { t.ir.ineg(op_a) } else { op_a };
    let op_b = if insn.bit(50) { t.ir.ineg(op_b) } else { op_b };
    let op_c = if insn.bit(49) { t.ir.ineg(op_c) } else { op_c };
    let mut lhs_1 = t.ir.iadd(op_a, op_b);
    if x {
        let c = t.ir.get_c_flag();
        let carry = t.ir.select(c, Value::U32(1), Value::U32(0));
        lhs_1 = t.ir.iadd(lhs_1, carry);
    }
    let lhs_2 = integer_shift(t, lhs_1, shift);
    let result = t.ir.iadd(lhs_2, op_c);
    t.set_x(insn.dest_reg(), result);
    if insn.bit(47) {
        if x {
            return Err(t.unsupported());
        }
        let z = t.ir.get_zero_from_op(result);
        t.ir.set_z_flag(z);
        let s = t.ir.get_sign_from_op(result);
        t.ir.set_s_flag(s);
        let c = t.ir.get_carry_from_op(result);
        t.ir.set_c_flag(c);
        let of_1 = t.ir.icompare(ICompare::ULessThan, lhs_1, op_a);
        let overflow = t.ir.get_overflow_from_op(result);
        let o = t.ir.logical_or(overflow, of_1);
        t.ir.set_o_flag(o);
    }
    Ok(())
}

pub(super) fn iadd3_cbuf(t: &mut Translator<'_>) -> TResult {
    let op_a = t.get_reg8();
    let op_b = t.get_cbuf()?;
    let op_c = t.get_reg39();
    iadd3(t, op_a, op_b, op_c, Shift::None)
}

pub(super) fn iadd3_reg(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let shift = match insn.bits(37, 2) {
        0 => Shift::None,
        1 => Shift::Right,
        2 => Shift::Left,
        _ => return Err(t.unsupported()),
    };
    let src_a = t.get_reg8();
    let op_a = integer_half(t, src_a, insn.bits(35, 2))?;
    let src_b = t.get_reg20();
    let op_b = integer_half(t, src_b, insn.bits(33, 2))?;
    let src_c = t.get_reg39();
    let op_c = integer_half(t, src_c, insn.bits(31, 2))?;
    iadd3(t, op_a, op_b, op_c, shift)
}

pub(super) fn iadd3_imm(t: &mut Translator<'_>) -> TResult {
    let op_a = t.get_reg8();
    let op_b = t.get_imm20();
    let op_c = t.get_reg39();
    iadd3(t, op_a, op_b, op_c, Shift::None)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn iadd3_reg() {
        let insn = encode(Opcode::Iadd3Reg, (1 << 8) | (2 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetRegister R3
  %3 = IAdd32 %0, %1
  %4 = IAdd32 %3, %2
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd3_reg_rs_halves_cc() {
        let insn = encode(
            Opcode::Iadd3Reg,
            (1 << 8) | (2 << 20) | (3 << 39) | (2 << 33) | (1 << 35) | (1 << 37) | (1 << 47),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = BitFieldUExtract %0, #0x0, #0x10
  %2 = GetRegister R2
  %3 = BitFieldUExtract %2, #0x10, #0x10
  %4 = GetRegister R3
  %5 = IAdd32 %1, %3
  %6 = GetCarryFromOp %5
  %7 = ShiftRightLogical32 %5, #0x10
  %8 = IAdd32 %7, #0x10000
  %9 = SelectU32 %6, %8, %7
  %10 = IAdd32 %9, %4
  SetRegister R0, %10
  %11 = GetZeroFromOp %10
  SetZFlag %11
  %12 = GetSignFromOp %10
  SetSFlag %12
  %13 = GetCarryFromOp %10
  SetCFlag %13
  %14 = ULessThan %5, %1
  %15 = GetOverflowFromOp %10
  %16 = LogicalOr %15, %14
  SetOFlag %16
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd3_cbuf_neg_x() {
        let insn = encode(
            Opcode::Iadd3Cbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39) | (1 << 48) | (1 << 49) | (1 << 51),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetCbufU32 #0x1, #0x10
  %2 = GetRegister R3
  %3 = INeg32 %0
  %4 = INeg32 %2
  %5 = IAdd32 %3, %1
  %6 = GetCFlag
  %7 = SelectU32 %6, #0x1, #0x0
  %8 = IAdd32 %5, %7
  %9 = IAdd32 %8, %4
  SetRegister R0, %9
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd3_imm() {
        let insn = encode(Opcode::Iadd3Imm, (1 << 8) | (5 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R3
  %2 = IAdd32 %0, #0x5
  %3 = IAdd32 %2, %1
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn iadd3_reserved_shift() {
        let insn = encode(Opcode::Iadd3Reg, 3 << 37);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn iadd3_reserved_half() {
        let insn = encode(Opcode::Iadd3Reg, 3 << 35);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn iadd3_rs_x() {
        let insn = encode(Opcode::Iadd3Reg, (1 << 37) | (1 << 48));
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
