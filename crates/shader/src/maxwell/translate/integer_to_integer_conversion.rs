//! I2I: integer width and signedness conversion, ported from integer_to_integer_conversion.cpp.

use super::{TResult, Translator};
use crate::ir::Value;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Width {
    Byte,
    Short,
    Word,
}

impl Width {
    const fn from_bits(bits: u64) -> Option<Width> {
        match bits {
            0 => Some(Width::Byte),
            1 => Some(Width::Short),
            2 => Some(Width::Word),
            _ => None,
        }
    }

    const fn bits(self) -> u32 {
        match self {
            Width::Byte => 8,
            Width::Short => 16,
            Width::Word => 32,
        }
    }
}

fn saturate_integer(
    t: &mut Translator<'_>,
    src: Value,
    dst_width: Width,
    dst_signed: bool,
    src_signed: bool,
) -> Value {
    let both_signed = dst_signed && src_signed;
    let (signed_min, signed_max, unsigned_max) = match dst_width {
        Width::Byte => (0xffff_ff80, 0x7f, 0xff),
        Width::Short => (0xffff_8000, 0x7fff, 0xffff),
        Width::Word => (0x8000_0000, 0x7fff_ffff, 0xffff_ffff),
    };
    let min = Value::U32(if both_signed { signed_min } else { 0 });
    let max = Value::U32(if dst_signed { signed_max } else { unsigned_max });
    let value = if !dst_signed && src_signed {
        t.ir.imax(Value::U32(0), src, true)
    } else {
        src
    };
    t.ir.iclamp(value, min, max, both_signed)
}

fn i2i(t: &mut Translator<'_>, src_a: Value) -> TResult {
    let insn = t.insn;
    let selector = insn.bits(41, 3) as u32;
    let (Some(dst_width), Some(src_width)) = (
        Width::from_bits(insn.bits(8, 2)),
        Width::from_bits(insn.bits(10, 2)),
    ) else {
        return Err(t.unsupported());
    };
    let bad_selector = match src_width {
        Width::Byte => false,
        Width::Short => selector == 1 || selector == 3,
        Width::Word => selector != 0,
    };
    if bad_selector {
        return Err(t.unsupported());
    }
    let dst_signed = insn.bit(12);
    let src_signed = insn.bit(13);
    let mut src_values = t.ir.bit_field_extract(
        src_a,
        Value::U32(selector * 8),
        Value::U32(src_width.bits()),
        src_signed,
    );
    if insn.bit(49) {
        src_values = t.ir.iabs(src_values);
    }
    if insn.bit(45) {
        src_values = t.ir.ineg(src_values);
    }
    let result = if insn.bit(50) {
        saturate_integer(t, src_values, dst_width, dst_signed, src_signed)
    } else {
        t.ir.bit_field_extract(
            src_values,
            Value::U32(0),
            Value::U32(dst_width.bits()),
            false,
        )
    };
    t.set_x(insn.dest_reg(), result);
    if insn.bit(47) {
        let z = t.ir.get_zero_from_op(result);
        t.ir.set_z_flag(z);
        let s = t.ir.get_sign_from_op(result);
        t.ir.set_s_flag(s);
        t.reset_c_flag();
        t.reset_o_flag();
    }
    Ok(())
}

pub(super) fn i2i_cbuf(t: &mut Translator<'_>) -> TResult {
    let src = t.get_cbuf()?;
    i2i(t, src)
}

pub(super) fn i2i_reg(t: &mut Translator<'_>) -> TResult {
    let src = t.get_reg20();
    i2i(t, src)
}

pub(super) fn i2i_imm(t: &mut Translator<'_>) -> TResult {
    let src = t.get_imm20();
    i2i(t, src)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn i2i_reg_s8_to_s32() {
        let insn = encode(
            Opcode::I2iReg,
            (2 << 20) | (2 << 8) | (1 << 12) | (1 << 13) | (1 << 41),
        );
        let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = BitFieldSExtract %0, #0x8, #0x8
  %2 = BitFieldUExtract %1, #0x0, #0x20
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn i2i_imm_s16_to_u8_sat_abs_neg_cc() {
        let insn = encode(
            Opcode::I2iImm,
            (5 << 20) | (1 << 10) | (1 << 13) | (1 << 45) | (1 << 47) | (1 << 49) | (1 << 50),
        );
        let expected = "\
{0}:
  %0 = BitFieldSExtract #0x5, #0x0, #0x10
  %1 = IAbs32 %0
  %2 = INeg32 %1
  %3 = SMax32 #0x0, %2
  %4 = UClamp32 %3, #0x0, #0xff
  SetRegister R0, %4
  %5 = GetZeroFromOp %4
  SetZFlag %5
  %6 = GetSignFromOp %4
  SetSFlag %6
  SetCFlag #false
  SetOFlag #false
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn i2i_cbuf_s32_to_s16_sat() {
        let insn = encode(
            Opcode::I2iCbuf,
            (1 << 34) | (4 << 20) | (1 << 8) | (1 << 12) | (2 << 10) | (1 << 13) | (1 << 50),
        );
        let expected = "\
{0}:
  %0 = GetCbufU32 #0x1, #0x10
  %1 = BitFieldSExtract %0, #0x0, #0x20
  %2 = SClamp32 %1, #0xffff8000, #0x7fff
  SetRegister R0, %2
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn i2i_reserved_width() {
        let insn = encode(Opcode::I2iReg, 3 << 8);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn i2i_short_odd_selector() {
        let insn = encode(Opcode::I2iReg, (1 << 10) | (1 << 41));
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
