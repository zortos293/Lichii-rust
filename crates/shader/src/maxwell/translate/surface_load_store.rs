//! SULD and SUST: surface loads and stores, plus the surface helpers SUATOM and SURED share.

use super::texture_fetch::image;
use super::texture_fetch_swizzled::check_alignment;
use super::{TResult, Translator};
use crate::TranslateError;
use crate::ir::{ImageFormat, Reg, TextureInstInfo, TextureType, Value};

/// Surface dimensionality of the 3-bit field at bits 33..36.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SurfaceDim {
    D1,
    Buffer,
    Array1D,
    D2,
    Array2D,
    D3,
}

impl SurfaceDim {
    pub(super) fn decode(t: &Translator<'_>) -> Result<SurfaceDim, TranslateError> {
        Ok(match t.insn.bits(33, 3) {
            0 => SurfaceDim::D1,
            1 => SurfaceDim::Buffer,
            2 => SurfaceDim::Array1D,
            3 => SurfaceDim::D2,
            4 => SurfaceDim::Array2D,
            5 => SurfaceDim::D3,
            _ => return Err(t.unsupported()),
        })
    }

    pub(super) fn ty(self) -> TextureType {
        match self {
            SurfaceDim::D1 => TextureType::Color1D,
            SurfaceDim::Buffer => TextureType::Buffer,
            SurfaceDim::Array1D => TextureType::ColorArray1D,
            SurfaceDim::D2 => TextureType::Color2D,
            SurfaceDim::Array2D => TextureType::ColorArray2D,
            SurfaceDim::D3 => TextureType::Color3D,
        }
    }
}

/// Integer surface coordinates from `reg`; the array layer is the low 16 bits of the register
/// after the coordinates.
pub(super) fn surface_coords(t: &mut Translator<'_>, reg: Reg, dim: SurfaceDim) -> Value {
    let layer = |t: &mut Translator<'_>, index: u16| {
        let v = t.x(reg.offset(index));
        t.ir.bit_field_extract(v, Value::U32(0), Value::U32(16), false)
    };
    match dim {
        SurfaceDim::D1 | SurfaceDim::Buffer => t.x(reg),
        SurfaceDim::Array1D => {
            let c = [t.x(reg), layer(t, 1)];
            t.ir.composite_construct(&c)
        }
        SurfaceDim::D2 => {
            let c = [t.x(reg), t.x(reg.offset(1))];
            t.ir.composite_construct(&c)
        }
        SurfaceDim::Array2D => {
            let c = [t.x(reg), t.x(reg.offset(1)), layer(t, 2)];
            t.ir.composite_construct(&c)
        }
        SurfaceDim::D3 => {
            let c = [t.x(reg), t.x(reg.offset(1)), t.x(reg.offset(2))];
            t.ir.composite_construct(&c)
        }
    }
}

/// Image handle: the immediate bound offset (bits 36..49, in words) or the register at bits
/// 39..47.
pub(super) fn surface_handle(t: &mut Translator<'_>, bound: bool) -> Value {
    if bound {
        Value::U32(t.insn.bits(36, 13) as u32 * 4)
    } else {
        t.x(Reg(t.insn.bits(39, 8) as u16))
    }
}

const MASK: [u32; 16] = [
    0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0x9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf,
];

/// Format and register count of the typed (.D) size at bits 20..23.
fn typed_size(t: &Translator<'_>) -> Result<(ImageFormat, u16), TranslateError> {
    Ok(match t.insn.bits(20, 3) {
        0 => (ImageFormat::R8Uint, 1),
        1 => (ImageFormat::R8Sint, 1),
        2 => (ImageFormat::R16Uint, 1),
        3 => (ImageFormat::R16Sint, 1),
        4 => (ImageFormat::R32Uint, 1),
        5 => (ImageFormat::R32G32Uint, 2),
        6 => (ImageFormat::R32G32B32A32Uint, 4),
        _ => return Err(t.unsupported()),
    })
}

/// Component mask of the untyped (.P) swizzle at bits 20..24.
fn swizzle_mask(t: &Translator<'_>) -> Result<u32, TranslateError> {
    match t.insn.bits(20, 4) {
        0 => Err(t.unsupported()),
        swizzle => Ok(MASK[swizzle as usize]),
    }
}

/// Rejects a clamp mode other than IGN (bits 49..51) and caches other than the first two at
/// bits 24..26 (CA/CG for loads, WB/CG for stores).
fn check_clamp_and_cache(t: &Translator<'_>) -> TResult {
    if t.insn.bits(49, 2) != 0 || t.insn.bits(24, 2) > 1 {
        return Err(t.unsupported());
    }
    Ok(())
}

pub(super) fn suld(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    check_clamp_and_cache(t)?;
    let is_typed = insn.bit(52);
    if is_typed && insn.bit(23) {
        return Err(t.unsupported());
    }
    let (format, num_regs) = if is_typed {
        typed_size(t)?
    } else {
        (ImageFormat::Typeless, 0)
    };
    let dim = SurfaceDim::decode(t)?;
    let coords = surface_coords(t, insn.src_a_reg(), dim);
    let handle = surface_handle(t, insn.bit(51));
    let info = TextureInstInfo {
        ty: dim.ty(),
        image_format: format,
        ..Default::default()
    };
    let result = image::read(&mut t.ir, handle, coords, info);

    let mut dest = insn.dest_reg();
    if is_typed {
        for i in 0..num_regs {
            let value = t.ir.composite_extract(result, u32::from(i));
            t.set_x(dest.offset(i), value);
        }
        return Ok(());
    }
    let mask = swizzle_mask(t)?;
    let bits = mask.count_ones() as u16;
    check_alignment(t, dest, if bits == 3 { 4 } else { bits })?;
    for component in 0..4 {
        if (mask >> component) & 1 == 0 {
            continue;
        }
        let value = t.ir.composite_extract(result, component);
        t.set_x(dest, value);
        dest = dest.offset(1);
    }
    Ok(())
}

pub(super) fn sust(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    check_clamp_and_cache(t)?;
    let is_typed = insn.bit(52);
    if is_typed && insn.bit(23) {
        return Err(t.unsupported());
    }
    let (format, num_regs) = if is_typed {
        typed_size(t)?
    } else {
        let mask = swizzle_mask(t)?;
        if !matches!(mask, 0x1 | 0x3 | 0xf) {
            return Err(t.unsupported());
        }
        (ImageFormat::Typeless, mask.count_ones() as u16)
    };
    let dim = SurfaceDim::decode(t)?;
    let coords = surface_coords(t, insn.src_a_reg(), dim);
    let handle = surface_handle(t, insn.bit(51));
    let info = TextureInstInfo {
        ty: dim.ty(),
        image_format: format,
        ..Default::default()
    };
    let data = insn.dest_reg();
    let mut colors = [Value::U32(0); 4];
    for (i, color) in colors.iter_mut().take(num_regs as usize).enumerate() {
        *color = t.x(data.offset(i as u16));
    }
    let color = t.ir.composite_construct(&colors);
    image::write(&mut t.ir, handle, coords, color, info);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn suld_typed_b64_array_2d_bound() {
        let insn = encode(
            Opcode::Suld,
            (1 << 52) | (1 << 51) | (2 << 36) | (4 << 33) | (5 << 20) | (4 << 8) | 2,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R4
  %1 = GetRegister R5
  %2 = GetRegister R6
  %3 = BitFieldUExtract %2, #0x0, #0x10
  %4 = CompositeConstructU32x3 %0, %1, %3
  %5 = BoundImageRead #0x8, %4 (flags 0x1800c000)
  %6 = CompositeExtractU32x4 %5, #0x0
  SetRegister R2, %6
  %7 = CompositeExtractU32x4 %5, #0x1
  SetRegister R3, %7
"
        );
    }

    #[test]
    fn suld_untyped_bindless_rb() {
        let insn = encode(Opcode::Suld, (9 << 39) | (5 << 20) | (2 << 8) | 4);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R9
  %2 = BindlessImageRead %1, %0
  %3 = CompositeExtractU32x4 %2, #0x0
  SetRegister R4, %3
  %4 = CompositeExtractU32x4 %2, #0x2
  SetRegister R5, %4
"
        );
    }

    #[test]
    fn sust_untyped_rg_buffer() {
        let insn = encode(
            Opcode::Sust,
            (1 << 51) | (1 << 36) | (1 << 33) | (3 << 20) | (2 << 8) | 6,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = GetRegister R2
  %1 = GetRegister R6
  %2 = GetRegister R7
  %3 = CompositeConstructU32x4 %1, %2, #0x0, #0x0
  BoundImageWrite #0x4, %0, %3 (flags 0x1c000)
"
        );
    }

    #[test]
    fn sust_typed_r32_2d() {
        let insn = encode(
            Opcode::Sust,
            (1 << 52) | (1 << 51) | (3 << 33) | (4 << 20) | (255 << 8) | 1,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:
  %0 = CompositeConstructU32x2 #0x0, #0x0
  %1 = GetRegister R1
  %2 = CompositeConstructU32x4 %1, #0x0, #0x0, #0x0
  BoundImageWrite #0x0, %0, %2 (flags 0x14008000)
"
        );
    }

    #[test]
    fn surface_rejects_unsupported_fields() {
        assert!(translate(&[encode(Opcode::Suld, (1 << 49) | (1 << 20))]).is_err());
        assert!(translate(&[encode(Opcode::Suld, (2 << 24) | (1 << 20))]).is_err());
        assert!(translate(&[encode(Opcode::Suld, (1 << 52) | (1 << 23))]).is_err());
        assert!(translate(&[encode(Opcode::Suld, (1 << 52) | (7 << 20))]).is_err());
        assert!(translate(&[encode(Opcode::Suld, 0)]).is_err());
        assert!(translate(&[encode(Opcode::Suld, (3 << 20) | 1)]).is_err());
        assert!(translate(&[encode(Opcode::Suld, (6 << 33) | (1 << 20))]).is_err());
        assert!(translate(&[encode(Opcode::Sust, 7 << 20)]).is_err());
        assert!(translate(&[encode(Opcode::Sust, (1 << 52) | (1 << 23) | (4 << 20))]).is_err());
    }
}
