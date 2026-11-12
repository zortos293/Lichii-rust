//! SHR: logical or arithmetic shift right, ported from integer_shift_right.cpp.

use super::{TResult, Translator};
use crate::ir::{ICompare, Value};

fn shr(t: &mut Translator<'_>, shift: Value) -> TResult {
    let insn = t.insn;
    if insn.bit(43) || insn.bit(47) {
        return Err(t.unsupported());
    }
    let wrapped = insn.bit(39);
    let signed = insn.bit(48);
    let mut base = t.x(insn.src_a_reg());
    if insn.bit(40) {
        base = t.ir.bit_reverse(base);
    }
    let safe_shift = if wrapped {
        t.ir.bitwise_and(shift, Value::U32(31))
    } else {
        shift
    };
    let mut result = if signed {
        t.ir.shift_right_arithmetic(base, safe_shift)
    } else {
        t.ir.shift_right_logical(base, safe_shift)
    };
    if !wrapped {
        let is_safe = t.ir.icompare(ICompare::ULessThan, shift, Value::U32(32));
        let clamped_value = if signed {
            t.ir.shift_right_arithmetic(base, Value::U32(31))
        } else {
            Value::U32(0)
        };
        result = t.ir.select(is_safe, result, clamped_value);
    }
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn shr_cbuf(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_cbuf()?;
    shr(t, shift)
}

pub(super) fn shr_reg(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_reg20();
    shr(t, shift)
}

pub(super) fn shr_imm(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_imm20();
    shr(t, shift)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn shr_reg_signed_clamped() {
        let insn = encode(Opcode::ShrReg, (1 << 8) | (2 << 20) | (1 << 48));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = ShiftRightArithmetic32 %1, %0
  %3 = ULessThan %0, #0x20
  %4 = ShiftRightArithmetic32 %1, #0x1f
  %5 = SelectU32 %3, %2, %4
  SetRegister R0, %5
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shr_imm_wrapped_brev() {
        let insn = encode(Opcode::ShrImm, (1 << 8) | (5 << 20) | (1 << 39) | (1 << 40));
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = BitReverse32 %0
  %2 = BitwiseAnd32 #0x5, #0x1f
  %3 = ShiftRightLogical32 %1, %2
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shr_cbuf() {
        let insn = encode(Opcode::ShrCbuf, (1 << 8) | (1 << 34) | (4 << 20));
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = ShiftRightLogical32 %1, %0
  %3 = ULessThan %0, #0x20
  %4 = SelectU32 %3, %2, #0x0
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shr_cc() {
        let insn = encode(Opcode::ShrReg, 1 << 47);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
