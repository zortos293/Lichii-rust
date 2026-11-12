//! TLDS: texel fetches with swizzled destinations.

use super::texture_fetch::image;
use super::texture_fetch_swizzled::{check_alignment, store_swizzled};
use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{Reg, TextureInstInfo, TextureType, Value};

fn offset(t: &mut Translator<'_>, reg: Reg) -> Value {
    let packed = t.x(reg);
    let o = [
        t.ir.bit_field_extract(packed, Value::U32(0), Value::U32(4), true),
        t.ir.bit_field_extract(packed, Value::U32(4), Value::U32(4), true),
    ];
    t.ir.composite_construct(&o)
}

fn sample(t: &mut Translator<'_>) -> Result<Value, TranslateError> {
    let insn = t.insn;
    let handle = Value::U32(insn.bits(36, 13) as u32 * 4);
    let a = insn.src_a_reg();
    let b = insn.src_b_reg();
    let encoding = insn.bits(53, 4);
    if matches!(encoding, 4..=7 | 12) {
        check_alignment(t, a, 2)?;
    }
    if matches!(encoding, 8 | 12) {
        check_alignment(t, b, 2)?;
    }
    let pair = |t: &mut Translator<'_>, x: Reg, y: Reg| {
        let c = [t.x(x), t.x(y)];
        t.ir.composite_construct(&c)
    };
    let mut lod = Value::U32(0);
    let mut offsets = Value::Void;
    let mut multisample = Value::Void;
    let (ty, coords) = match encoding {
        0 => (TextureType::Color1D, t.x(a)),
        1 => {
            let coords = t.x(a);
            lod = t.x(b);
            (TextureType::Color1D, coords)
        }
        2 => (TextureType::Color2D, pair(t, a, b)),
        4 => {
            let coords = pair(t, a, a.offset(1));
            offsets = offset(t, b);
            (TextureType::Color2D, coords)
        }
        5 => {
            let coords = pair(t, a, a.offset(1));
            lod = t.x(b);
            (TextureType::Color2D, coords)
        }
        6 => {
            let coords = pair(t, a, a.offset(1));
            multisample = t.x(b);
            (TextureType::Color2D, coords)
        }
        7 => {
            let c = [t.x(a), t.x(a.offset(1)), t.x(b)];
            (TextureType::Color3D, t.ir.composite_construct(&c))
        }
        8 => {
            let packed = t.x(a);
            let layer =
                t.ir.bit_field_extract(packed, Value::U32(0), Value::U32(16), false);
            let c = [t.x(b), t.x(b.offset(1)), layer];
            (TextureType::ColorArray2D, t.ir.composite_construct(&c))
        }
        12 => {
            let coords = pair(t, a, a.offset(1));
            lod = t.x(b);
            offsets = offset(t, b.offset(1));
            (TextureType::Color2D, coords)
        }
        _ => return Err(t.unsupported()),
    };
    let info = TextureInstInfo {
        ty,
        relaxed_precision: !insn.bit(59),
        ..Default::default()
    };
    Ok(image::fetch(
        &mut t.ir,
        handle,
        coords,
        offsets,
        lod,
        multisample,
        info,
    ))
}

pub(super) fn tlds(t: &mut Translator<'_>) -> TResult {
    let sample = sample(t)?;
    store_swizzled(t, sample)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn tlds_2d_lod_offset_f32() {
        let insn = encode(
            Opcode::Tlds,
            (1 << 59) | (12 << 53) | (1 << 50) | (9 << 36) | (255 << 28) | (6 << 20) | (2 << 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = CompositeConstructU32x2 %0, %1
  %3 = GetRegister R6
  %4 = GetRegister R7
  %5 = BitFieldSExtract %4, #0x0, #0x4
  %6 = BitFieldSExtract %4, #0x4, #0x4
  %7 = CompositeConstructU32x2 %5, %6
  %8 = BoundImageFetch #0x24, %2, %7, %3, void (flags 0x8000)
  %9 = CompositeExtractF32x4 %8, #0x1
  %10 = BitCastU32F32 %9
  SetRegister R0, %10
"
        );
    }

    #[test]
    fn tlds_1d_f16_rgb() {
        let insn = encode(Opcode::Tlds, (2 << 28) | (4 << 8));
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R4
  %1 = BoundImageFetch #0x0, %0, void, #0x0, void (flags 0x200000)
  %2 = CompositeExtractF32x4 %1, #0x0
  %3 = CompositeExtractF32x4 %1, #0x1
  %4 = CompositeExtractF32x4 %1, #0x2
  %5 = CompositeConstructF32x2 %2, %3
  %6 = PackHalf2x16 %5
  SetRegister R0, %6
  %7 = CompositeConstructF32x2 %4, #0.0f
  %8 = PackHalf2x16 %7
  SetRegister R2, %8
"
        );
    }

    #[test]
    fn tlds_rejects_bad_encodings() {
        assert!(translate(&[encode(Opcode::Tlds, 3 << 53)]).is_err());
        assert!(translate(&[encode(Opcode::Tlds, (8 << 53) | (1 << 20))]).is_err());
    }
}
