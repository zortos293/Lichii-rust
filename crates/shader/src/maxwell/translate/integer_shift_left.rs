//! SHL: logical shift left, ported from integer_shift_left.cpp.

use super::{TResult, Translator};
use crate::ir::{ICompare, Value};

fn shl(t: &mut Translator<'_>, unsafe_shift: Value) -> TResult {
    let insn = t.insn;
    if insn.bit(43) || insn.bit(47) {
        return Err(t.unsupported());
    }
    let base = t.x(insn.src_a_reg());
    let result = if insn.bit(39) {
        let shift = t.ir.bitwise_and(unsafe_shift, Value::U32(31));
        t.ir.shift_left_logical(base, shift)
    } else {
        // A clamped shift of 32 or more must read as zero; the shift itself only has an
        // undefined result (not undefined behaviour) in SPIR-V and GLASM, so select it away.
        let is_safe =
            t.ir.icompare(ICompare::ULessThan, unsafe_shift, Value::U32(32));
        let unsafe_result = t.ir.shift_left_logical(base, unsafe_shift);
        t.ir.select(is_safe, unsafe_result, Value::U32(0))
    };
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn shl_cbuf(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_cbuf()?;
    shl(t, shift)
}

pub(super) fn shl_reg(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_reg20();
    shl(t, shift)
}

pub(super) fn shl_imm(t: &mut Translator<'_>) -> TResult {
    let shift = t.get_imm20();
    shl(t, shift)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn shl_reg_clamped() {
        let insn = encode(Opcode::ShlReg, (1 << 8) | (2 << 20));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R1
  %2 = ULessThan %0, #0x20
  %3 = ShiftLeftLogical32 %1, %0
  %4 = SelectU32 %2, %3, #0x0
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shl_imm_wrapped() {
        let insn = encode(Opcode::ShlImm, (1 << 8) | (5 << 20) | (1 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = BitwiseAnd32 #0x5, #0x1f
  %2 = ShiftLeftLogical32 %0, %1
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shl_cbuf_wrapped() {
        let insn = encode(
            Opcode::ShlCbuf,
            (1 << 8) | (1 << 34) | (4 << 20) | (1 << 39),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R1
  %2 = BitwiseAnd32 %0, #0x1f
  %3 = ShiftLeftLogical32 %1, %2
  SetRegister R0, %3
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn shl_x() {
        let insn = encode(Opcode::ShlReg, 1 << 43);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
