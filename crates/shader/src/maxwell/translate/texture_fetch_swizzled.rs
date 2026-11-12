//! TEXS, plus the swizzled stores TLDS shares.

use super::texture_fetch::{image, sample_component};
use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{Reg, TextureInstInfo, TextureType, Value};

const R: u32 = 1;
const G: u32 = 2;
const B: u32 = 4;
const A: u32 = 8;

const RG_LUT: [u32; 8] = [R, G, B, A, R | G, R | A, G | A, B | A];
const RGBA_LUT: [u32; 5] = [R | G | B, R | G | A, R | B | A, G | B | A, R | G | B | A];

/// Errors unless `reg` is RZ or a multiple of `alignment`.
pub(super) fn check_alignment(t: &Translator<'_>, reg: Reg, alignment: u16) -> TResult {
    if reg.is_zero() || reg.0.is_multiple_of(alignment) {
        Ok(())
    } else {
        Err(t.unsupported())
    }
}

/// Layer index of an array texture: the low 16 bits of `value` as a float.
pub(super) fn read_array(t: &mut Translator<'_>, value: Value) -> Value {
    let layer =
        t.ir.bit_field_extract(value, Value::U32(0), Value::U32(16), false);
    t.ir.convert_i_to_f(32, 16, false, layer, Default::default())
}

fn swizzle(t: &Translator<'_>) -> Result<u32, TranslateError> {
    let encoding = t.insn.bits(50, 3) as usize;
    let lut: &[u32] = if Reg(t.insn.bits(28, 8) as u16).is_zero() {
        &RG_LUT
    } else {
        &RGBA_LUT
    };
    lut.get(encoding).copied().ok_or_else(|| t.unsupported())
}

fn store_component32(t: &Translator<'_>, index: u32) -> Result<Reg, TranslateError> {
    let dest_a = t.insn.dest_reg();
    let dest_b = Reg(t.insn.bits(28, 8) as u16);
    Ok(match index {
        0 => dest_a,
        1 => {
            check_alignment(t, dest_a, 2)?;
            dest_a.offset(1)
        }
        2 => dest_b,
        _ => {
            check_alignment(t, dest_b, 2)?;
            dest_b.offset(1)
        }
    })
}

fn pack(t: &mut Translator<'_>, lhs: Value, rhs: Value) -> Value {
    let v = t.ir.composite_construct(&[lhs, rhs]);
    t.ir.pack_half_2x16(v)
}

/// Stores the components the swizzle at bits 50..53 selects to the destination pair at bits
/// 0..8 and 28..36: one per register when bit 59 selects F32, two packed halves otherwise.
pub(super) fn store_swizzled(t: &mut Translator<'_>, sample: Value) -> TResult {
    let swizzle = swizzle(t)?;
    if t.insn.bit(59) {
        let mut store_index = 0;
        for component in 0..4 {
            if (swizzle >> component) & 1 == 0 {
                continue;
            }
            let dest = store_component32(t, store_index)?;
            let value = sample_component(t, sample, component);
            t.set_f(dest, value);
            store_index += 1;
        }
        return Ok(());
    }
    let mut swizzled = [Value::Void; 4];
    let mut count = 0;
    for component in 0..4 {
        if (swizzle >> component) & 1 == 0 {
            continue;
        }
        swizzled[count] = sample_component(t, sample, component);
        count += 1;
    }
    let zero = Value::F32(0.0);
    let dest_a = t.insn.dest_reg();
    let dest_b = Reg(t.insn.bits(28, 8) as u16);
    let low = pack(t, swizzled[0], if count == 1 { zero } else { swizzled[1] });
    t.set_x(dest_a, low);
    if count > 2 {
        let high = pack(t, swizzled[2], if count == 3 { zero } else { swizzled[3] });
        t.set_x(dest_b, high);
    }
    Ok(())
}

fn sample(t: &mut Translator<'_>) -> Result<Value, TranslateError> {
    let insn = t.insn;
    let handle = Value::U32(insn.bits(36, 13) as u32 * 4);
    let zero = Value::F32(0.0);
    let a = insn.src_a_reg();
    let b = insn.src_b_reg();
    let mut info = TextureInstInfo {
        relaxed_precision: !insn.bit(59),
        ..Default::default()
    };
    let void = Value::Void;
    let pair = |t: &mut Translator<'_>, x: Reg, y: Reg| {
        let c = [t.f(x), t.f(y)];
        t.ir.composite_construct(&c)
    };
    let triple = |t: &mut Translator<'_>| {
        let c = [t.f(a), t.f(a.offset(1)), t.f(b)];
        t.ir.composite_construct(&c)
    };
    let array_2d = |t: &mut Translator<'_>| {
        let x = t.f(a.offset(1));
        let y = t.f(b);
        let layer_reg = t.x(a);
        let layer = read_array(t, layer_reg);
        t.ir.composite_construct(&[x, y, layer])
    };
    let encoding = insn.bits(53, 4);
    if matches!(encoding, 3..=13) {
        check_alignment(t, a, 2)?;
    }
    if matches!(encoding, 5 | 9 | 13) {
        check_alignment(t, b, 2)?;
    }
    Ok(match encoding {
        0 => {
            info.ty = TextureType::Color1D;
            let coords = t.f(a);
            image::sample_explicit_lod(&mut t.ir, handle, coords, zero, void, info)
        }
        1 => {
            info.ty = TextureType::Color2D;
            let coords = pair(t, a, b);
            image::sample_implicit_lod(&mut t.ir, handle, coords, void, void, void, info)
        }
        2 => {
            info.ty = TextureType::Color2D;
            let coords = pair(t, a, b);
            image::sample_explicit_lod(&mut t.ir, handle, coords, zero, void, info)
        }
        3 => {
            info.ty = TextureType::Color2D;
            let coords = pair(t, a, a.offset(1));
            let lod = t.f(b);
            image::sample_explicit_lod(&mut t.ir, handle, coords, lod, void, info)
        }
        4 => {
            info.ty = TextureType::Color2D;
            info.is_depth = true;
            let coords = pair(t, a, a.offset(1));
            let dref = t.f(b);
            image::sample_dref_implicit_lod(&mut t.ir, handle, coords, dref, void, void, void, info)
        }
        5 => {
            info.ty = TextureType::Color2D;
            info.is_depth = true;
            let coords = pair(t, a, a.offset(1));
            let dref = t.f(b.offset(1));
            let lod = t.f(b);
            image::sample_dref_explicit_lod(&mut t.ir, handle, coords, dref, lod, void, info)
        }
        6 => {
            info.ty = TextureType::Color2D;
            info.is_depth = true;
            let coords = pair(t, a, a.offset(1));
            let dref = t.f(b);
            image::sample_dref_explicit_lod(&mut t.ir, handle, coords, dref, zero, void, info)
        }
        7 => {
            info.ty = TextureType::ColorArray2D;
            let coords = array_2d(t);
            image::sample_implicit_lod(&mut t.ir, handle, coords, void, void, void, info)
        }
        8 => {
            info.ty = TextureType::ColorArray2D;
            let coords = array_2d(t);
            image::sample_explicit_lod(&mut t.ir, handle, coords, zero, void, info)
        }
        9 => {
            info.ty = TextureType::ColorArray2D;
            info.is_depth = true;
            let coords = array_2d(t);
            let dref = t.f(b.offset(1));
            image::sample_dref_explicit_lod(&mut t.ir, handle, coords, dref, zero, void, info)
        }
        10 | 11 => {
            info.ty = TextureType::Color3D;
            let coords = triple(t);
            if encoding == 10 {
                image::sample_implicit_lod(&mut t.ir, handle, coords, void, void, void, info)
            } else {
                image::sample_explicit_lod(&mut t.ir, handle, coords, zero, void, info)
            }
        }
        12 => {
            info.ty = TextureType::ColorCube;
            let coords = triple(t);
            image::sample_implicit_lod(&mut t.ir, handle, coords, void, void, void, info)
        }
        13 => {
            info.ty = TextureType::ColorCube;
            let coords = triple(t);
            let lod = t.f(b.offset(1));
            image::sample_explicit_lod(&mut t.ir, handle, coords, lod, void, info)
        }
        _ => return Err(t.unsupported()),
    })
}

pub(super) fn texs(t: &mut Translator<'_>) -> TResult {
    let sample = sample(t)?;
    store_swizzled(t, sample)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn texs_2d_f32_rgba() {
        let insn = encode(
            Opcode::Texs,
            (1 << 59)
                | (1 << 53)
                | (4 << 50)
                | (2 << 36)
                | (6 << 28)
                | (255 << 20)
                | (255 << 8)
                | 4,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = CompositeConstructF32x2 #0.0f, #0.0f
  %1 = BoundImageSampleImplicitLod #0x8, %0, void, void (flags 0x8000)
  %2 = CompositeExtractF32x4 %1, #0x0
  %3 = BitCastU32F32 %2
  SetRegister R4, %3
  %4 = CompositeExtractF32x4 %1, #0x1
  %5 = BitCastU32F32 %4
  SetRegister R5, %5
  %6 = CompositeExtractF32x4 %1, #0x2
  %7 = BitCastU32F32 %6
  SetRegister R6, %7
  %8 = CompositeExtractF32x4 %1, #0x3
  %9 = BitCastU32F32 %8
  SetRegister R7, %9
"
        );
    }

    #[test]
    fn texs_2d_lz_depth_f16_packs_alpha_one() {
        let insn = encode(
            Opcode::Texs,
            (6 << 53) | (5 << 50) | (255 << 28) | (3 << 20) | (2 << 8) | 1,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = BitCastF32U32 %0
  %2 = GetRegister R3
  %3 = BitCastF32U32 %2
  %4 = CompositeConstructF32x2 %1, %3
  %5 = GetRegister R3
  %6 = BitCastF32U32 %5
  %7 = BoundImageSampleDrefExplicitLod #0x0, %4, %6, #0.0f, void (flags 0x248000)
  %8 = CompositeConstructF32x2 %7, #1.0f
  %9 = PackHalf2x16 %8
  SetRegister R1, %9
"
        );
    }

    #[test]
    fn texs_array_2d_reads_layer() {
        let insn = encode(
            Opcode::Texs,
            (1 << 59) | (7 << 53) | (255 << 28) | (255 << 20) | (2 << 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R3
  %1 = BitCastF32U32 %0
  %2 = GetRegister R2
  %3 = BitFieldUExtract %2, #0x0, #0x10
  %4 = ConvertF32U16 %3
  %5 = CompositeConstructF32x3 %1, #0.0f, %4
  %6 = BoundImageSampleImplicitLod #0x0, %5, void, void (flags 0xc000)
  %7 = CompositeExtractF32x4 %6, #0x0
  %8 = BitCastU32F32 %7
  SetRegister R0, %8
"
        );
    }

    #[test]
    fn texs_rejects_bad_encodings() {
        assert!(translate(&[encode(Opcode::Texs, 14 << 53)]).is_err());
        assert!(translate(&[encode(Opcode::Texs, (3 << 53) | (1 << 8))]).is_err());
        assert!(translate(&[encode(Opcode::Texs, (255 << 28) | (5 << 50) | (1 << 53))]).is_ok());
        assert!(translate(&[encode(Opcode::Texs, (4 << 28) | (5 << 50) | (1 << 53))]).is_err());
    }
}
