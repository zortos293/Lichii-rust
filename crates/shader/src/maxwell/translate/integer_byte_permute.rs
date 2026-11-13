//! PRMT: byte permute, ported from integer_byte_permute.cpp.

use super::{TResult, Translator};
use crate::ir::{ICompare, Value};

fn prmt(t: &mut Translator<'_>, selector: Value, src_c: Value) -> TResult {
    let insn = t.insn;
    if insn.bits(48, 3) != 0 {
        return Err(t.unsupported());
    }
    let src_a = t.x(insn.src_a_reg());
    let mut result = Value::U32(0);
    for byte in 0..4u32 {
        let nibble =
            t.ir.bit_field_extract(selector, Value::U32(byte * 4), Value::U32(4), false);
        let index = t.ir.bitwise_and(nibble, Value::U32(7));
        let from_a = t.ir.icompare(ICompare::ULessThan, index, Value::U32(4));
        let source = t.ir.select(from_a, src_a, src_c);
        let byte_index = t.ir.bitwise_and(index, Value::U32(3));
        let offset = t.ir.shift_left_logical(byte_index, Value::U32(3));
        let value = t.ir.bit_field_extract(source, offset, Value::U32(8), false);
        let sign_offset = t.ir.iadd(offset, Value::U32(7));
        let sign =
            t.ir.bit_field_extract(source, sign_offset, Value::U32(1), true);
        let replicate_bit = t.ir.bitwise_and(nibble, Value::U32(8));
        let replicate =
            t.ir.icompare(ICompare::NotEqual, replicate_bit, Value::U32(0));
        let picked = t.ir.select(replicate, sign, value);
        result =
            t.ir.bit_field_insert(result, picked, Value::U32(byte * 8), Value::U32(8));
    }
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn prmt_cr(t: &mut Translator<'_>) -> TResult {
    let selector = t.get_cbuf()?;
    let src_c = t.get_reg39();
    prmt(t, selector, src_c)
}

pub(super) fn prmt_rc(t: &mut Translator<'_>) -> TResult {
    let selector = t.get_reg39();
    let src_c = t.get_cbuf()?;
    prmt(t, selector, src_c)
}

pub(super) fn prmt_reg(t: &mut Translator<'_>) -> TResult {
    let selector = t.get_reg20();
    let src_c = t.get_reg39();
    prmt(t, selector, src_c)
}

pub(super) fn prmt_imm(t: &mut Translator<'_>) -> TResult {
    let selector = t.get_imm20();
    let src_c = t.get_reg39();
    prmt(t, selector, src_c)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn prmt_reg() {
        let insn = encode(Opcode::PrmtReg, (1 << 8) | (2 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = BitFieldUExtract %0, #0x0, #0x4
  %4 = BitwiseAnd32 %3, #0x7
  %5 = ULessThan %4, #0x4
  %6 = SelectU32 %5, %2, %1
  %7 = BitwiseAnd32 %4, #0x3
  %8 = ShiftLeftLogical32 %7, #0x3
  %9 = BitFieldUExtract %6, %8, #0x8
  %10 = IAdd32 %8, #0x7
  %11 = BitFieldSExtract %6, %10, #0x1
  %12 = BitwiseAnd32 %3, #0x8
  %13 = INotEqual %12, #0x0
  %14 = SelectU32 %13, %11, %9
  %15 = BitFieldInsert #0x0, %14, #0x0, #0x8
  %16 = BitFieldUExtract %0, #0x4, #0x4
  %17 = BitwiseAnd32 %16, #0x7
  %18 = ULessThan %17, #0x4
  %19 = SelectU32 %18, %2, %1
  %20 = BitwiseAnd32 %17, #0x3
  %21 = ShiftLeftLogical32 %20, #0x3
  %22 = BitFieldUExtract %19, %21, #0x8
  %23 = IAdd32 %21, #0x7
  %24 = BitFieldSExtract %19, %23, #0x1
  %25 = BitwiseAnd32 %16, #0x8
  %26 = INotEqual %25, #0x0
  %27 = SelectU32 %26, %24, %22
  %28 = BitFieldInsert %15, %27, #0x8, #0x8
  %29 = BitFieldUExtract %0, #0x8, #0x4
  %30 = BitwiseAnd32 %29, #0x7
  %31 = ULessThan %30, #0x4
  %32 = SelectU32 %31, %2, %1
  %33 = BitwiseAnd32 %30, #0x3
  %34 = ShiftLeftLogical32 %33, #0x3
  %35 = BitFieldUExtract %32, %34, #0x8
  %36 = IAdd32 %34, #0x7
  %37 = BitFieldSExtract %32, %36, #0x1
  %38 = BitwiseAnd32 %29, #0x8
  %39 = INotEqual %38, #0x0
  %40 = SelectU32 %39, %37, %35
  %41 = BitFieldInsert %28, %40, #0x10, #0x8
  %42 = BitFieldUExtract %0, #0xc, #0x4
  %43 = BitwiseAnd32 %42, #0x7
  %44 = ULessThan %43, #0x4
  %45 = SelectU32 %44, %2, %1
  %46 = BitwiseAnd32 %43, #0x3
  %47 = ShiftLeftLogical32 %46, #0x3
  %48 = BitFieldUExtract %45, %47, #0x8
  %49 = IAdd32 %47, #0x7
  %50 = BitFieldSExtract %45, %49, #0x1
  %51 = BitwiseAnd32 %42, #0x8
  %52 = INotEqual %51, #0x0
  %53 = SelectU32 %52, %50, %48
  %54 = BitFieldInsert %41, %53, #0x18, #0x8
  SetRegister R0, %54
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn prmt_rc() {
        let insn = encode(Opcode::PrmtRc, (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39));
        let dump = translate(&[insn]).unwrap();
        let head = "\
{0}:
  %0 = GetRegister R3
  %1 = GetCbufU32 #0x1, #0x10
  %2 = GetRegister R1
  %3 = BitFieldUExtract %0, #0x0, #0x4
  %4 = BitwiseAnd32 %3, #0x7
";
        assert!(dump.starts_with(head), "{dump}");
        assert_eq!(dump.lines().count(), 57);
    }

    #[test]
    fn prmt_cr() {
        let insn = encode(Opcode::PrmtCr, (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39));
        let dump = translate(&[insn]).unwrap();
        let head = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = BitFieldUExtract %0, #0x0, #0x4
  %4 = BitwiseAnd32 %3, #0x7
";
        assert!(dump.starts_with(head), "{dump}");
        assert_eq!(dump.lines().count(), 57);
    }

    #[test]
    fn prmt_imm() {
        let insn = encode(Opcode::PrmtImm, (1 << 8) | (0x3210 << 20) | (3 << 39));
        let dump = translate(&[insn]).unwrap();
        let head = "\
{0}:
  %0 = GetRegister R3
  %1 = GetRegister R1
  %2 = BitFieldUExtract #0x3210, #0x0, #0x4
  %3 = BitwiseAnd32 %2, #0x7
  %4 = ULessThan %3, #0x4
";
        assert!(dump.starts_with(head), "{dump}");
        assert_eq!(dump.lines().count(), 56);
    }

    #[test]
    fn prmt_mode() {
        let insn = encode(Opcode::PrmtImm, 1 << 48);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
