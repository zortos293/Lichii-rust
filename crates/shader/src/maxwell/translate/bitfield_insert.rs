//! BFI: bitfield insert, ported from bitfield_insert.cpp.

use super::{TResult, Translator};
use crate::ir::{ICompare, Value};

fn bfi(t: &mut Translator<'_>, src_a: Value, base: Value) -> TResult {
    let insn = t.insn;
    let zero = Value::U32(0);
    let offset = t.ir.bit_field_extract(src_a, zero, Value::U32(8), false);
    let unsafe_count =
        t.ir.bit_field_extract(src_a, Value::U32(8), Value::U32(8), false);
    let max_size = Value::U32(32);
    let exceed_offset = t.ir.icompare(ICompare::UGreaterThanEqual, offset, max_size);
    let exceed_count =
        t.ir.icompare(ICompare::UGreaterThan, unsafe_count, max_size);
    let remaining_size = t.ir.isub(max_size, offset);
    let safe_count = t.ir.select(exceed_count, remaining_size, unsafe_count);
    let insert = t.x(insn.src_a_reg());
    let result = t.ir.bit_field_insert(base, insert, offset, safe_count);
    let result = t.ir.select(exceed_offset, base, result);
    t.set_x(insn.dest_reg(), result);
    if insn.bit(47) {
        let z = t.ir.icompare(ICompare::Equal, result, zero);
        t.ir.set_z_flag(z);
        let s = t.ir.icompare(ICompare::SLessThan, result, zero);
        t.ir.set_s_flag(s);
        t.reset_c_flag();
        t.reset_o_flag();
    }
    Ok(())
}

pub(super) fn bfi_cr(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_cbuf()?;
    let base = t.get_reg39();
    bfi(t, src_a, base)
}

pub(super) fn bfi_rc(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_reg39();
    let base = t.get_cbuf()?;
    bfi(t, src_a, base)
}

pub(super) fn bfi_reg(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_reg20();
    let base = t.get_reg39();
    bfi(t, src_a, base)
}

pub(super) fn bfi_imm(t: &mut Translator<'_>) -> TResult {
    let src_a = t.get_imm20();
    let base = t.get_reg39();
    bfi(t, src_a, base)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn bfi_reg_cc() {
        let insn = encode(Opcode::BfiReg, (1 << 8) | (2 << 20) | (3 << 39) | (1 << 47));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = BitFieldUExtract %0, #0x0, #0x8
  %3 = BitFieldUExtract %0, #0x8, #0x8
  %4 = UGreaterThanEqual %2, #0x20
  %5 = UGreaterThan %3, #0x20
  %6 = ISub32 #0x20, %2
  %7 = SelectU32 %5, %6, %3
  %8 = GetRegister R1
  %9 = BitFieldInsert %1, %8, %2, %7
  %10 = SelectU32 %4, %1, %9
  SetRegister R0, %10
  %11 = IEqual %10, #0x0
  SetZFlag %11
  %12 = SLessThan %10, #0x0
  SetSFlag %12
  SetCFlag #false
  SetOFlag #false
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn bfi_rc() {
        let insn = encode(Opcode::BfiRc, (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetCbufU32 #0x1, #0x10
  %2 = BitFieldUExtract %0, #0x0, #0x8
  %3 = BitFieldUExtract %0, #0x8, #0x8
  %4 = UGreaterThanEqual %2, #0x20
  %5 = UGreaterThan %3, #0x20
  %6 = ISub32 #0x20, %2
  %7 = SelectU32 %5, %6, %3
  %8 = GetRegister R1
  %9 = BitFieldInsert %1, %8, %2, %7
  %10 = SelectU32 %4, %1, %9
  SetRegister R0, %10
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn bfi_cr() {
        let insn = encode(Opcode::BfiCr, (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R3
  %2 = BitFieldUExtract %0, #0x0, #0x8
  %3 = BitFieldUExtract %0, #0x8, #0x8
  %4 = UGreaterThanEqual %2, #0x20
  %5 = UGreaterThan %3, #0x20
  %6 = ISub32 #0x20, %2
  %7 = SelectU32 %5, %6, %3
  %8 = GetRegister R1
  %9 = BitFieldInsert %1, %8, %2, %7
  %10 = SelectU32 %4, %1, %9
  SetRegister R0, %10
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn bfi_imm() {
        let insn = encode(Opcode::BfiImm, (1 << 8) | (0x0804 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = BitFieldUExtract #0x804, #0x0, #0x8
  %2 = BitFieldUExtract #0x804, #0x8, #0x8
  %3 = UGreaterThanEqual %1, #0x20
  %4 = UGreaterThan %2, #0x20
  %5 = ISub32 #0x20, %1
  %6 = SelectU32 %4, %5, %2
  %7 = GetRegister R1
  %8 = BitFieldInsert %0, %7, %1, %6
  %9 = SelectU32 %3, %0, %8
  SetRegister R0, %9
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }
}
