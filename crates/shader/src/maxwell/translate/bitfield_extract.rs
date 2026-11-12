//! BFE: bitfield extract, ported from bitfield_extract.cpp.

use super::{TResult, Translator};
use crate::ir::{ICompare, Value};

fn bfe(t: &mut Translator<'_>, src: Value) -> TResult {
    let insn = t.insn;
    let zero = Value::U32(0);
    let one = Value::U32(1);
    let max_size = Value::U32(32);
    let offset = t.ir.bit_field_extract(src, zero, Value::U32(8), false);
    let count =
        t.ir.bit_field_extract(src, Value::U32(8), Value::U32(8), false);
    let zero_count = t.ir.icompare(ICompare::Equal, count, zero);
    let end = t.ir.iadd(offset, count);
    let exceed_count = t.ir.icompare(ICompare::UGreaterThanEqual, end, max_size);
    let replicate = t.ir.icompare(ICompare::UGreaterThanEqual, offset, max_size);
    let mut base = t.x(insn.src_a_reg());
    if insn.bit(40) {
        base = t.ir.bit_reverse(base);
    }
    let signed = insn.bit(48);
    let mut result = t.ir.bit_field_extract(base, offset, count, signed);
    if signed {
        let is_negative = t.ir.icompare(ICompare::SLessThan, base, zero);
        let replicated_bit = t.ir.select(is_negative, Value::U32(u32::MAX), zero);
        let exceed_bit = t.ir.bit_field_extract(base, Value::U32(31), one, false);
        result = t.ir.select(replicate, replicated_bit, result);
        let exceed_result =
            t.ir.bit_field_insert(result, exceed_bit, Value::U32(31), one);
        result = t.ir.select(exceed_count, exceed_result, result);
    }
    result = t.ir.select(zero_count, zero, result);
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

pub(super) fn bfe_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = t.get_cbuf()?;
    bfe(t, src)
}

pub(super) fn bfe_reg(t: &mut Translator<'_>) -> TResult {
    let src = t.get_reg20();
    bfe(t, src)
}

pub(super) fn bfe_imm(t: &mut Translator<'_>) -> TResult {
    let src = t.get_imm20();
    bfe(t, src)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn bfe_reg() {
        let insn = encode(Opcode::BfeReg, (1 << 8) | (2 << 20));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = BitFieldUExtract %0, #0x0, #0x8
  %2 = BitFieldUExtract %0, #0x8, #0x8
  %3 = IEqual %2, #0x0
  %4 = IAdd32 %1, %2
  %5 = UGreaterThanEqual %4, #0x20
  %6 = UGreaterThanEqual %1, #0x20
  %7 = GetRegister R1
  %8 = BitFieldUExtract %7, %1, %2
  %9 = SelectU32 %3, #0x0, %8
  SetRegister R0, %9
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn bfe_imm_signed_brev_cc() {
        let insn = encode(
            Opcode::BfeImm,
            (1 << 8) | (0x0804 << 20) | (1 << 40) | (1 << 47) | (1 << 48),
        );
        let expected = "\
{0}:
  %0 = BitFieldUExtract #0x804, #0x0, #0x8
  %1 = BitFieldUExtract #0x804, #0x8, #0x8
  %2 = IEqual %1, #0x0
  %3 = IAdd32 %0, %1
  %4 = UGreaterThanEqual %3, #0x20
  %5 = UGreaterThanEqual %0, #0x20
  %6 = GetRegister R1
  %7 = BitReverse32 %6
  %8 = BitFieldSExtract %7, %0, %1
  %9 = SLessThan %7, #0x0
  %10 = SelectU32 %9, #0xffffffff, #0x0
  %11 = BitFieldUExtract %7, #0x1f, #0x1
  %12 = SelectU32 %5, %10, %8
  %13 = BitFieldInsert %12, %11, #0x1f, #0x1
  %14 = SelectU32 %4, %13, %12
  %15 = SelectU32 %2, #0x0, %14
  SetRegister R0, %15
  %16 = IEqual %15, #0x0
  SetZFlag %16
  %17 = SLessThan %15, #0x0
  SetSFlag %17
  SetCFlag #false
  SetOFlag #false
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn bfe_cbuf() {
        let insn = encode(Opcode::BfeCbuf, (1 << 8) | (1 << 34) | (4 << 20));
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = BitFieldUExtract %0, #0x0, #0x8
  %2 = BitFieldUExtract %0, #0x8, #0x8
  %3 = IEqual %2, #0x0
  %4 = IAdd32 %1, %2
  %5 = UGreaterThanEqual %4, #0x20
  %6 = UGreaterThanEqual %1, #0x20
  %7 = GetRegister R1
  %8 = BitFieldUExtract %7, %1, %2
  %9 = SelectU32 %3, #0x0, %8
  SetRegister R0, %9
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn bfe_cbuf_out_of_range_binding() {
        let insn = encode(Opcode::BfeCbuf, 31 << 34);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
