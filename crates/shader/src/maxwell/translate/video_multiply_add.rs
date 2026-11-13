//! VMAD: video multiply-add, ported from video_multiply_add.cpp. Also holds the operand
//! extraction the C++ keeps in video_helper.cpp, shared by VMNMX and VSETP.

use super::{TResult, Translator};
use crate::ir::{Emitter, Value};

/// Video operand width (2 bits); `Unknown` behaves like `Byte`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum VideoWidth {
    Byte,
    Unknown,
    Short,
    Word,
}

impl VideoWidth {
    pub(super) const fn from_bits(bits: u64) -> VideoWidth {
        match bits & 3 {
            0 => VideoWidth::Byte,
            1 => VideoWidth::Unknown,
            2 => VideoWidth::Short,
            _ => VideoWidth::Word,
        }
    }

    /// Width of operand B: immediates are always 16 bits.
    pub(super) const fn source(self, is_immediate: bool) -> VideoWidth {
        if is_immediate {
            VideoWidth::Short
        } else {
            self
        }
    }
}

/// The byte or half-word of `value` named by `selector`, or all of it for `Word`.
pub(super) fn extract_video_operand_value(
    ir: &mut Emitter<'_>,
    value: Value,
    width: VideoWidth,
    selector: u32,
    signed: bool,
) -> Value {
    match width {
        VideoWidth::Byte => {
            ir.bit_field_extract(value, Value::U32(selector * 8), Value::U32(8), signed)
        }
        VideoWidth::Unknown => {
            crate::fallback::note("VIDEO", "unknown_width_as_byte");
            ir.bit_field_extract(value, Value::U32(selector * 8), Value::U32(8), signed)
        }
        VideoWidth::Short => {
            ir.bit_field_extract(value, Value::U32(selector * 16), Value::U32(16), signed)
        }
        VideoWidth::Word => value,
    }
}

pub(super) fn vmad(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if insn.bit(47) || insn.bit(55) || insn.bits(51, 2) != 0 || insn.bit(53) || insn.bit(54) {
        return Err(t.unsupported());
    }
    let is_b_imm = !insn.bit(50);
    let src_a = t.get_reg8();
    let src_b = if is_b_imm {
        Value::U32(insn.bits(20, 16) as u32)
    } else {
        t.get_reg20()
    };
    let src_c = t.get_reg39();
    let a_selector = insn.bits(36, 2) as u32;
    let b_selector = if is_b_imm { 0 } else { insn.bits(28, 2) as u32 };
    let a_width = VideoWidth::from_bits(insn.bits(37, 2));
    let b_width = VideoWidth::from_bits(insn.bits(29, 2)).source(is_b_imm);
    let op_a = extract_video_operand_value(&mut t.ir, src_a, a_width, a_selector, insn.bit(48));
    let op_b = extract_video_operand_value(&mut t.ir, src_b, b_width, b_selector, insn.bit(49));
    let product = t.ir.imul(op_a, op_b);
    let result = t.ir.iadd(product, src_c);
    t.set_x(insn.dest_reg(), result);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::TranslateError;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn vmad_byte_by_short() {
        let insn = encode(
            Opcode::Vmad,
            (1 << 8)
                | (2 << 20)
                | (3 << 39)
                | (1 << 28)
                | (2 << 29)
                | (1 << 36)
                | (1 << 48)
                | (1 << 50),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetRegister R3
  %3 = BitFieldSExtract %0, #0x8, #0x8
  %4 = BitFieldUExtract %1, #0x10, #0x10
  %5 = IMul32 %3, %4
  %6 = IAdd32 %5, %2
  SetRegister R0, %6
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn vmad_imm() {
        let insn = encode(
            Opcode::Vmad,
            (1 << 8) | (0x42 << 20) | (3 << 39) | (3 << 37),
        );
        let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R3
  %2 = BitFieldUExtract #0x42, #0x0, #0x10
  %3 = IMul32 %0, %2
  %4 = IAdd32 %3, %1
  SetRegister R0, %4
";
        assert_eq!(translate(&[insn]).unwrap(), expected);
    }

    #[test]
    fn vmad_neg() {
        let insn = encode(Opcode::Vmad, 1 << 53);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }

    #[test]
    fn vmad_scale() {
        let insn = encode(Opcode::Vmad, 1 << 51);
        assert!(matches!(
            translate(&[insn]),
            Err(TranslateError::Unsupported { .. })
        ));
    }
}
