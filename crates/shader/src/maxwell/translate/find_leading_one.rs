//! FLO: find leading one, ported from find_leading_one.cpp.

use super::{TResult, Translator};
use crate::ir::{ICompare, Value};

fn flo(t: &mut Translator<'_>, src: Value) -> TResult {
    let insn = t.insn;
    if insn.bit(47) {
        return Err(t.unsupported());
    }
    let src = if insn.bit(40) {
        t.ir.bitwise_not(src)
    } else {
        src
    };
    let mut result = if insn.bit(48) {
        t.ir.find_s_msb(src)
    } else {
        t.ir.find_u_msb(src)
    };
    if insn.bit(41) {
        let not_found = t.ir.icompare(ICompare::Equal, result, Value::U32(u32::MAX));
        let flipped = t.ir.bitwise_xor(result, Value::U32(31));
        result = t.ir.select(not_found, result, flipped);
    }
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn flo_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = t.get_cbuf()?;
    flo(t, src)
}

pub(super) fn flo_reg(t: &mut Translator<'_>) -> TResult {
    let src = t.get_reg20();
    flo(t, src)
}

pub(super) fn flo_imm(t: &mut Translator<'_>) -> TResult {
    let src = t.get_imm20();
    flo(t, src)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn flo_reg_shift() {
        let insn = encode(Opcode::FloReg, (2 << 20) | (1 << 41));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = FindUMsb32 %0
  %2 = IEqual %1, #0xffffffff
  %3 = BitwiseXor32 %1, #0x1f
  %4 = SelectU32 %2, %1, %3
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn flo_imm_signed_inverted() {
        let insn = encode(Opcode::FloImm, (5 << 20) | (1 << 40) | (1 << 48));
        let expected = "\
{0}:
  %0 = BitwiseNot32 #0x5
  %1 = FindSMsb32 %0
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn flo_cbuf() {
        let insn = encode(Opcode::FloCbuf, (1 << 34) | (4 << 20));
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = FindUMsb32 %0
  SetRegister R0, %1
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn flo_cc() {
        let insn = encode(Opcode::FloReg, 1 << 47);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
