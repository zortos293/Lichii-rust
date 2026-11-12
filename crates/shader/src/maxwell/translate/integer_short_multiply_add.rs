//! XMAD: 16x16-bit multiply with a 32-bit add, ported from integer_short_multiply_add.cpp.

use super::{TResult, Translator};
use crate::ir::Value;

#[derive(Clone, Copy)]
enum SelectMode {
    Default,
    Clo,
    Chi,
    Cbcc,
}

#[derive(Clone, Copy)]
struct Modifiers {
    select_mode: u64,
    half_b_hi: bool,
    psl: bool,
    mrg: bool,
    x: bool,
}

fn extract_half(t: &mut Translator<'_>, src: Value, hi: bool, signed: bool) -> Value {
    let offset = Value::U32(if hi { 16 } else { 0 });
    t.ir.bit_field_extract(src, offset, Value::U32(16), signed)
}

fn xmad(t: &mut Translator<'_>, src_b: Value, src_c: Value, m: Modifiers) -> TResult {
    let insn = t.insn;
    let select_mode = match m.select_mode {
        0 => SelectMode::Default,
        1 => SelectMode::Clo,
        2 => SelectMode::Chi,
        4 => SelectMode::Cbcc,
        _ => return Err(t.unsupported()),
    };
    if m.x || insn.bit(47) {
        return Err(t.unsupported());
    }
    let src_a = t.x(insn.src_a_reg());
    let op_a = extract_half(t, src_a, insn.bit(53), insn.bit(48));
    let op_b = extract_half(t, src_b, m.half_b_hi, insn.bit(49));
    let mut product = t.ir.imul(op_a, op_b);
    if m.psl {
        product = t.ir.shift_left_logical(product, Value::U32(16));
    }
    let op_c = match select_mode {
        SelectMode::Default => src_c,
        SelectMode::Clo => extract_half(t, src_c, false, false),
        SelectMode::Chi => extract_half(t, src_c, true, false),
        SelectMode::Cbcc => {
            let shifted_b = t.ir.shift_left_logical(src_b, Value::U32(16));
            t.ir.iadd(shifted_b, src_c)
        }
    };
    let mut result = t.ir.iadd(product, op_c);
    if m.mrg {
        let lsb_b = extract_half(t, src_b, false, false);
        result =
            t.ir.bit_field_insert(result, lsb_b, Value::U32(16), Value::U32(16));
    }
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

pub(super) fn xmad_reg(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let m = Modifiers {
        select_mode: insn.bits(50, 3),
        half_b_hi: insn.bit(35),
        psl: insn.bit(36),
        mrg: insn.bit(37),
        x: insn.bit(38),
    };
    let src_b = t.get_reg20();
    let src_c = t.get_reg39();
    xmad(t, src_b, src_c, m)
}

pub(super) fn xmad_imm(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let m = Modifiers {
        select_mode: insn.bits(50, 3),
        half_b_hi: false,
        psl: insn.bit(36),
        mrg: insn.bit(37),
        x: insn.bit(38),
    };
    let src_b = Value::U32(insn.bits(20, 16) as u32);
    let src_c = t.get_reg39();
    xmad(t, src_b, src_c, m)
}

pub(super) fn xmad_rc(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let m = Modifiers {
        select_mode: insn.bits(50, 2),
        half_b_hi: insn.bit(52),
        psl: false,
        mrg: false,
        x: insn.bit(54),
    };
    let src_b = t.get_reg39();
    let src_c = t.get_cbuf()?;
    xmad(t, src_b, src_c, m)
}

pub(super) fn xmad_cr(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let m = Modifiers {
        select_mode: insn.bits(50, 2),
        half_b_hi: insn.bit(52),
        psl: insn.bit(55),
        mrg: insn.bit(56),
        x: insn.bit(54),
    };
    let src_b = t.get_cbuf()?;
    let src_c = t.get_reg39();
    xmad(t, src_b, src_c, m)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn xmad_reg() {
        let insn = encode(Opcode::XmadReg, (1 << 8) | (2 << 20) | (3 << 39));
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = BitFieldUExtract %2, #0x0, #0x10
  %4 = BitFieldUExtract %0, #0x0, #0x10
  %5 = IMul32 %3, %4
  %6 = IAdd32 %5, %1
  SetRegister R0, %6
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn xmad_reg_cbcc() {
        let insn = encode(
            Opcode::XmadReg,
            (1 << 8) | (2 << 20) | (3 << 39) | (4 << 50),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = BitFieldUExtract %2, #0x0, #0x10
  %4 = BitFieldUExtract %0, #0x0, #0x10
  %5 = IMul32 %3, %4
  %6 = ShiftLeftLogical32 %0, #0x10
  %7 = IAdd32 %6, %1
  %8 = IAdd32 %5, %7
  SetRegister R0, %8
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn xmad_imm_psl_mrg_signed_high_a() {
        let insn = encode(
            Opcode::XmadImm,
            (1 << 8) | (0x1234 << 20) | (3 << 39) | (1 << 36) | (1 << 37) | (1 << 48) | (1 << 53),
        );
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetRegister R1
  %2 = BitFieldSExtract %1, #0x10, #0x10
  %3 = BitFieldUExtract #0x1234, #0x0, #0x10
  %4 = IMul32 %2, %3
  %5 = ShiftLeftLogical32 %4, #0x10
  %6 = IAdd32 %5, %0
  %7 = BitFieldUExtract #0x1234, #0x0, #0x10
  %8 = BitFieldInsert %6, %7, #0x10, #0x10
  SetRegister R0, %8
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn xmad_rc_clo_high_b() {
        let insn = encode(
            Opcode::XmadRc,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39) | (1 << 50) | (1 << 52),
        );
        let expected = "\
{0}:
  %0 = GetRegister R3
  %1 = GetCbufU32 #0x1, #0x10
  %2 = GetRegister R1
  %3 = BitFieldUExtract %2, #0x0, #0x10
  %4 = BitFieldUExtract %0, #0x10, #0x10
  %5 = IMul32 %3, %4
  %6 = BitFieldUExtract %1, #0x0, #0x10
  %7 = IAdd32 %5, %6
  SetRegister R0, %7
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn xmad_cr_chi() {
        let insn = encode(
            Opcode::XmadCr,
            (1 << 8) | (1 << 34) | (4 << 20) | (3 << 39) | (2 << 50),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = BitFieldUExtract %2, #0x0, #0x10
  %4 = BitFieldUExtract %0, #0x0, #0x10
  %5 = IMul32 %3, %4
  %6 = BitFieldUExtract %1, #0x10, #0x10
  %7 = IAdd32 %5, %6
  SetRegister R0, %7
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn xmad_csfu() {
        let insn = encode(Opcode::XmadReg, 3 << 50);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn xmad_cc() {
        let insn = encode(Opcode::XmadReg, 1 << 47);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
