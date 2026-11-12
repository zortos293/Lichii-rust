//! ICMP: select between two sources on a comparison against zero, ported from
//! integer_compare.cpp.

use super::{CompareOp, TResult, Translator, integer_compare};
use crate::ir::Value;

fn icmp(t: &mut Translator<'_>, src_a: Value, operand: Value) -> TResult {
    let insn = t.insn;
    let op = CompareOp::from_bits(insn.bits(49, 3));
    let cmp_result = integer_compare(&mut t.ir, operand, Value::U32(0), op, insn.bit(48));
    let src_reg = t.x(insn.src_a_reg());
    let result = t.ir.select(cmp_result, src_reg, src_a);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn icmp_cr(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_cbuf()?;
    let operand = t.get_reg39();
    icmp(t, src_a, operand)
}

pub(super) fn icmp_rc(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_reg39();
    let operand = t.get_cbuf()?;
    icmp(t, src_a, operand)
}

pub(super) fn icmp_reg(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_reg20();
    let operand = t.get_reg39();
    icmp(t, src_a, operand)
}

pub(super) fn icmp_imm(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_imm20();
    let operand = t.get_reg39();
    icmp(t, src_a, operand)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn icmp_reg_signed_gt() {
        let insn = encode(
            Opcode::IcmpReg,
            (1 << 8) | (2 << 20) | (3 << 39) | (1 << 48) | (4 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = SGreaterThan %1, #0x0
  %3 = GetRegister R1
  %4 = SelectU32 %2, %3, %0
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn icmp_rc_eq() {
        let insn = encode(
            Opcode::IcmpRc,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39) | (2 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetCbufU32 #0x1, #0x10
  %2 = IEqual %1, #0x0
  %3 = GetRegister R1
  %4 = SelectU32 %2, %3, %0
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn icmp_cr_eq() {
        let insn = encode(
            Opcode::IcmpCr,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39) | (2 << 49),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R3
  %2 = IEqual %1, #0x0
  %3 = GetRegister R1
  %4 = SelectU32 %2, %3, %0
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn icmp_imm_true() {
        let insn = encode(
            Opcode::IcmpImm,
            (1 << 8) | (5 << 20) | (3 << 39) | (7 << 49),
        );
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetRegister R1
  %2 = SelectU32 #true, %1, #0x5
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
