//! ALD, AST and IPA: attribute loads, stores and fragment interpolation.

use super::exit_program::program_header;
use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{Attribute, FpControl, Patch, Reg, Value};
use crate::maxwell::header::PixelImap;

/// The patch for word `index`, if it names one.
fn patch(t: &Translator<'_>, index: u32) -> Result<Patch, TranslateError> {
    if index >= u32::from(Patch::COMPONENT_0.0 + Patch::NUM_COMPONENTS) {
        return Err(t.unsupported());
    }
    Ok(Patch(index as u8))
}

/// Word index of the absolute offset (bits 20..30) and the element count (bits 47..49).
fn absolute_access(t: &Translator<'_>) -> Result<(u32, u16), TranslateError> {
    let offset = t.insn.bits(20, 10) as u32;
    if !offset.is_multiple_of(4) {
        return Err(t.unsupported());
    }
    Ok((offset / 4, t.insn.bits(47, 2) as u16 + 1))
}

pub(super) fn ald(t: &mut Translator<'_>) -> TResult {
    let (word, count) = absolute_access(t)?;
    let dest = t.insn.dest_reg();
    let index_reg = t.insn.src_a_reg();
    let is_patch = t.insn.bit(31);
    if is_patch && !index_reg.is_zero() {
        return Err(t.unsupported());
    }
    let vertex = t.x(Reg(t.insn.bits(39, 8) as u16));
    if index_reg.is_zero() {
        for element in 0..count {
            let index = word + u32::from(element);
            let value = if is_patch {
                let patch = patch(t, index)?;
                t.ir.get_patch(patch)
            } else {
                t.ir.get_attribute(Attribute(index as u16), vertex)
            };
            t.set_f(dest.offset(element), value);
        }
        return Ok(());
    }
    let base = t.x(index_reg);
    for element in 0..count {
        let address = if element == 0 {
            base
        } else {
            t.ir.iadd(base, Value::U32(u32::from(element) * 4))
        };
        let value = t.ir.get_attribute_indexed(address, vertex);
        t.set_f(dest.offset(element), value);
    }
    Ok(())
}

pub(super) fn ast(t: &mut Translator<'_>) -> TResult {
    if !t.insn.src_a_reg().is_zero() {
        return Err(t.unsupported());
    }
    let (word, count) = absolute_access(t)?;
    let src = t.insn.dest_reg();
    let is_patch = t.insn.bit(31);
    let vertex = t.x(Reg(t.insn.bits(39, 8) as u16));
    for element in 0..count {
        let index = word + u32::from(element);
        let value = t.f(src.offset(element));
        if is_patch {
            let patch = patch(t, index)?;
            t.ir.set_patch(patch, value);
        } else {
            t.ir.set_attribute(Attribute(index as u16), value, vertex);
        }
    }
    Ok(())
}

pub(super) fn ipa(t: &mut Translator<'_>) -> TResult {
    const MULTIPLY: u64 = 1;
    const SC: u64 = 3;
    let insn = t.insn;
    let attribute = Attribute(insn.bits(30, 8) as u16);
    let interpolation = insn.bits(54, 2);
    if insn.bit(51) && attribute == Attribute::FRONT_FACE {
        return Err(t.unsupported());
    }
    if insn.bits(52, 2) != 0 {
        crate::fallback::note("IPA", "sample_mode");
    }
    let index_reg = insn.src_a_reg();
    let mut value = if insn.bit(38) && !index_reg.is_zero() {
        let index = t.x(index_reg);
        t.ir.get_attribute_indexed(index, Value::U32(0))
    } else {
        t.ir.get_attribute(attribute, Value::U32(0))
    };
    if let Some(generic) = attribute.generic_index() {
        let sph = program_header(t).unwrap_or_else(|| {
            crate::fallback::note("IPA", "no_program_header");
            Default::default()
        });
        let imap = sph
            .ps()
            .generic_input_map(usize::from(generic))
            .into_iter()
            .find(|&imap| imap != PixelImap::Unused)
            .unwrap_or(PixelImap::Unused);
        if matches!(imap, PixelImap::Perspective | PixelImap::Unused) {
            value = multiply_by_position_w(t, value);
        }
    } else if attribute.is_legacy() && interpolation != SC {
        value = multiply_by_position_w(t, value);
    }
    if interpolation == MULTIPLY {
        let multiplier = t.f(insn.src_b_reg());
        value = t.ir.fp_mul(value, multiplier, FpControl::default());
    }
    if insn.bit(51) {
        value = t.ir.fp_saturate(value);
    }
    t.set_f(insn.dest_reg(), value);
    Ok(())
}

fn multiply_by_position_w(t: &mut Translator<'_>, value: Value) -> Value {
    let w = t.ir.get_attribute(Attribute::POSITION_W, Value::U32(0));
    t.ir.fp_mul(value, w, FpControl::default())
}

#[cfg(test)]
mod tests {
    use super::super::exit_program::tests::translate_stage;
    use super::super::tests::{encode, translate};
    use crate::ShaderStage;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn ald_b64_absolute() {
        let insn = encode(
            Opcode::Ald,
            (1 << 47) | (3 << 39) | (0x80 << 20) | (0xff << 8) | 4,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R3\n\
             \x20 %1 = GetAttribute Generic0X, %0\n\
             \x20 %2 = BitCastU32F32 %1\n\
             \x20 SetRegister R4, %2\n\
             \x20 %3 = GetAttribute Generic0Y, %0\n\
             \x20 %4 = BitCastU32F32 %3\n\
             \x20 SetRegister R5, %4\n"
        );
    }

    #[test]
    fn ald_patch_and_indexed() {
        let patch = encode(
            Opcode::Ald,
            (1 << 31) | (0xff << 39) | (0x18 << 20) | (0xff << 8) | 1,
        );
        assert_eq!(
            translate(&[patch]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetPatch Patch0\n\
             \x20 %1 = BitCastU32F32 %0\n\
             \x20 SetRegister R1, %1\n"
        );
        let indexed = encode(Opcode::Ald, (1 << 47) | (0xff << 39) | (2 << 8) | 1);
        assert_eq!(
            translate(&[indexed]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetAttributeIndexed %0, #0x0\n\
             \x20 %2 = BitCastU32F32 %1\n\
             \x20 SetRegister R1, %2\n\
             \x20 %3 = IAdd32 %0, #0x4\n\
             \x20 %4 = GetAttributeIndexed %3, #0x0\n\
             \x20 %5 = BitCastU32F32 %4\n\
             \x20 SetRegister R2, %5\n"
        );
    }

    #[test]
    fn ald_invalid_forms_are_unsupported() {
        let unaligned = encode(Opcode::Ald, 2 << 20);
        let indirect_patch = encode(Opcode::Ald, (1 << 31) | (2 << 8));
        let patch_out_of_range = encode(Opcode::Ald, (1 << 31) | (0x3fc << 20) | (0xff << 8));
        for insn in [unaligned, indirect_patch, patch_out_of_range] {
            assert!(translate(&[insn]).is_err(), "{insn:#x}");
        }
    }

    #[test]
    fn ast_position_and_patch() {
        let position = encode(Opcode::Ast, (0xff << 39) | (0x7c << 20) | (0xff << 8) | 1);
        assert_eq!(
            translate(&[position]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R1\n\
             \x20 %1 = BitCastF32U32 %0\n\
             \x20 SetAttribute PositionW, %1, #0x0\n"
        );
        let patch = encode(
            Opcode::Ast,
            (1 << 31) | (0xff << 39) | (4 << 20) | (0xff << 8) | 2,
        );
        assert_eq!(
            translate(&[patch]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = BitCastF32U32 %0\n\
             \x20 SetPatch TessLod1, %1\n"
        );
        assert!(translate(&[encode(Opcode::Ast, 2 << 8)]).is_err());
    }

    fn ipa(fields: u64) -> u64 {
        encode(Opcode::Ipa, fields | (0xff << 8))
    }

    #[test]
    fn ipa_perspective_generic() {
        let mut header = [0; 20];
        header[6] = 0b10 << 2;
        let insn = ipa((33 << 30) | 1);
        assert_eq!(
            translate_stage(&[insn], ShaderStage::Fragment, header).unwrap(),
            "{0}:\n\
             \x20 %0 = GetAttribute Generic0Y, #0x0\n\
             \x20 %1 = GetAttribute PositionW, #0x0\n\
             \x20 %2 = FPMul32 %0, %1\n\
             \x20 %3 = BitCastU32F32 %2\n\
             \x20 SetRegister R1, %3\n"
        );
    }

    #[test]
    fn ipa_linear_generic_multiply_saturate() {
        let mut header = [0; 20];
        header[6] = 0b11;
        let insn = ipa((1 << 54) | (1 << 51) | (32 << 30) | (5 << 20) | 1);
        assert_eq!(
            translate_stage(&[insn], ShaderStage::Fragment, header).unwrap(),
            "{0}:\n\
             \x20 %0 = GetAttribute Generic0X, #0x0\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = BitCastF32U32 %1\n\
             \x20 %3 = FPMul32 %0, %2\n\
             \x20 %4 = FPSaturate32 %3\n\
             \x20 %5 = BitCastU32F32 %4\n\
             \x20 SetRegister R1, %5\n"
        );
    }

    #[test]
    fn ipa_position_is_not_perspective_corrected() {
        let insn = ipa((28 << 30) | 1);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetAttribute PositionX, #0x0\n\
             \x20 %1 = BitCastU32F32 %0\n\
             \x20 SetRegister R1, %1\n"
        );
    }

    #[test]
    fn ipa_legacy_screen_space_and_indexed() {
        let fog = ipa((3 << 54) | (186 << 30) | 1);
        assert!(!translate(&[fog]).unwrap().contains("PositionW"));
        let color = ipa((160 << 30) | 1);
        assert!(translate(&[color]).unwrap().contains("FPMul32 %0, %1"));
        let indexed = encode(Opcode::Ipa, (1 << 38) | (28 << 30) | (2 << 8) | 1);
        assert!(
            translate(&[indexed])
                .unwrap()
                .starts_with("{0}:\n  %0 = GetRegister R2\n  %1 = GetAttributeIndexed %0, #0x0\n")
        );
    }

    #[test]
    fn ipa_saturated_front_face_is_unsupported() {
        assert!(translate(&[ipa((1 << 51) | (255 << 30))]).is_err());
    }
}
