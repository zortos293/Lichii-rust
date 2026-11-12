//! Render-target pixel conversion for the software blitter: every color render-target format
//! decodes to and encodes from four `f32` channels (R, G, B, A).
//!
//! A format is a list of fields packed from bit 0 upwards, in the order the format's name
//! lists them from its last component to its first (`A8R8G8B8` keeps B in the lowest byte).
//! Missing color channels decode to 0 and a missing or padding alpha to 1. Padding (`X`)
//! fields encode as 0.

use lichii_texture::guest::RenderTargetFormat;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Unorm,
    Snorm,
    Uint,
    Sint,
    Float,
    Srgb,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Channel {
    R,
    G,
    B,
    A,
    Pad,
}

#[derive(Clone, Copy)]
struct Field {
    channel: Channel,
    kind: Kind,
    bits: u32,
}

const fn field(channel: Channel, kind: Kind, bits: u32) -> Field {
    Field {
        channel,
        kind,
        bits,
    }
}

const fn quad(kind: Kind, bits: u32, channels: [Channel; 4]) -> [Field; 4] {
    [
        field(channels[0], kind, bits),
        field(channels[1], kind, bits),
        field(channels[2], kind, bits),
        field(channels[3], kind, bits),
    ]
}

const fn pair(kind: Kind, bits: u32) -> [Field; 2] {
    [field(Channel::R, kind, bits), field(Channel::G, kind, bits)]
}

const fn single(kind: Kind, bits: u32) -> [Field; 1] {
    [field(Channel::R, kind, bits)]
}

use Channel::{A, B, G, Pad, R};
use Kind::{Float, Sint, Snorm, Srgb, Uint, Unorm};

static RGBA32_FLOAT: [Field; 4] = quad(Float, 32, [R, G, B, A]);
static RGBA32_SINT: [Field; 4] = quad(Sint, 32, [R, G, B, A]);
static RGBA32_UINT: [Field; 4] = quad(Uint, 32, [R, G, B, A]);
static RGBX32_FLOAT: [Field; 4] = quad(Float, 32, [R, G, B, Pad]);
static RGBX32_SINT: [Field; 4] = quad(Sint, 32, [R, G, B, Pad]);
static RGBX32_UINT: [Field; 4] = quad(Uint, 32, [R, G, B, Pad]);
static RGBA16_UNORM: [Field; 4] = quad(Unorm, 16, [R, G, B, A]);
static RGBA16_SNORM: [Field; 4] = quad(Snorm, 16, [R, G, B, A]);
static RGBA16_SINT: [Field; 4] = quad(Sint, 16, [R, G, B, A]);
static RGBA16_UINT: [Field; 4] = quad(Uint, 16, [R, G, B, A]);
static RGBA16_FLOAT: [Field; 4] = quad(Float, 16, [R, G, B, A]);
static RGBX16_FLOAT: [Field; 4] = quad(Float, 16, [R, G, B, Pad]);
static RG32_FLOAT: [Field; 2] = pair(Float, 32);
static RG32_SINT: [Field; 2] = pair(Sint, 32);
static RG32_UINT: [Field; 2] = pair(Uint, 32);
static BGRA8_UNORM: [Field; 4] = quad(Unorm, 8, [B, G, R, A]);
static BGRA8_SRGB: [Field; 4] = quad(Srgb, 8, [B, G, R, A]);
static BGRX8_UNORM: [Field; 4] = quad(Unorm, 8, [B, G, R, Pad]);
static BGRX8_SRGB: [Field; 4] = quad(Srgb, 8, [B, G, R, Pad]);
static RGBA8_UNORM: [Field; 4] = quad(Unorm, 8, [R, G, B, A]);
static RGBA8_SRGB: [Field; 4] = quad(Srgb, 8, [R, G, B, A]);
static RGBA8_SNORM: [Field; 4] = quad(Snorm, 8, [R, G, B, A]);
static RGBA8_SINT: [Field; 4] = quad(Sint, 8, [R, G, B, A]);
static RGBA8_UINT: [Field; 4] = quad(Uint, 8, [R, G, B, A]);
static RGBX8_UNORM: [Field; 4] = quad(Unorm, 8, [R, G, B, Pad]);
static RGBX8_SRGB: [Field; 4] = quad(Srgb, 8, [R, G, B, Pad]);
static RGB10A2_UNORM: [Field; 4] = [
    field(R, Unorm, 10),
    field(G, Unorm, 10),
    field(B, Unorm, 10),
    field(A, Unorm, 2),
];
static RGB10A2_UINT: [Field; 4] = [
    field(R, Uint, 10),
    field(G, Uint, 10),
    field(B, Uint, 10),
    field(A, Uint, 2),
];
static BGR10A2_UNORM: [Field; 4] = [
    field(B, Unorm, 10),
    field(G, Unorm, 10),
    field(R, Unorm, 10),
    field(A, Unorm, 2),
];
static RG16_UNORM: [Field; 2] = pair(Unorm, 16);
static RG16_SNORM: [Field; 2] = pair(Snorm, 16);
static RG16_SINT: [Field; 2] = pair(Sint, 16);
static RG16_UINT: [Field; 2] = pair(Uint, 16);
static RG16_FLOAT: [Field; 2] = pair(Float, 16);
static R11G11B10_FLOAT: [Field; 3] = [
    field(R, Float, 11),
    field(G, Float, 11),
    field(B, Float, 10),
];
static R32_SINT: [Field; 1] = single(Sint, 32);
static R32_UINT: [Field; 1] = single(Uint, 32);
static R32_FLOAT: [Field; 1] = single(Float, 32);
static B5G6R5_UNORM: [Field; 3] = [field(B, Unorm, 5), field(G, Unorm, 6), field(R, Unorm, 5)];
static BGR5A1_UNORM: [Field; 4] = [
    field(B, Unorm, 5),
    field(G, Unorm, 5),
    field(R, Unorm, 5),
    field(A, Unorm, 1),
];
static BGR5X1_UNORM: [Field; 4] = [
    field(B, Unorm, 5),
    field(G, Unorm, 5),
    field(R, Unorm, 5),
    field(Pad, Unorm, 1),
];
static RG8_UNORM: [Field; 2] = pair(Unorm, 8);
static RG8_SNORM: [Field; 2] = pair(Snorm, 8);
static RG8_SINT: [Field; 2] = pair(Sint, 8);
static RG8_UINT: [Field; 2] = pair(Uint, 8);
static R16_UNORM: [Field; 1] = single(Unorm, 16);
static R16_SNORM: [Field; 1] = single(Snorm, 16);
static R16_SINT: [Field; 1] = single(Sint, 16);
static R16_UINT: [Field; 1] = single(Uint, 16);
static R16_FLOAT: [Field; 1] = single(Float, 16);
static R8_UNORM: [Field; 1] = single(Unorm, 8);
static R8_SNORM: [Field; 1] = single(Snorm, 8);
static R8_SINT: [Field; 1] = single(Sint, 8);
static R8_UINT: [Field; 1] = single(Uint, 8);

/// How a render-target format packs its channels.
#[derive(Clone, Copy)]
pub struct Layout {
    fields: &'static [Field],
    /// Bytes per pixel.
    pub bytes: usize,
}

impl Layout {
    fn new(fields: &'static [Field]) -> Self {
        let bits: u32 = fields.iter().map(|f| f.bits).sum();
        Self {
            fields,
            bytes: (bits / 8) as usize,
        }
    }

    /// The layout of a color render-target format; `None` for `NONE`.
    pub fn of(format: RenderTargetFormat) -> Option<Self> {
        use RenderTargetFormat as F;
        Some(Self::new(match format {
            F::None => return None,
            F::R32G32B32A32Float => &RGBA32_FLOAT,
            F::R32G32B32A32Sint => &RGBA32_SINT,
            F::R32G32B32A32Uint => &RGBA32_UINT,
            F::R32G32B32X32Float => &RGBX32_FLOAT,
            F::R32G32B32X32Sint => &RGBX32_SINT,
            F::R32G32B32X32Uint => &RGBX32_UINT,
            F::R16G16B16A16Unorm => &RGBA16_UNORM,
            F::R16G16B16A16Snorm => &RGBA16_SNORM,
            F::R16G16B16A16Sint => &RGBA16_SINT,
            F::R16G16B16A16Uint => &RGBA16_UINT,
            F::R16G16B16A16Float => &RGBA16_FLOAT,
            F::R32G32Float => &RG32_FLOAT,
            F::R32G32Sint => &RG32_SINT,
            F::R32G32Uint => &RG32_UINT,
            F::R16G16B16X16Float => &RGBX16_FLOAT,
            F::A8R8G8B8Unorm => &BGRA8_UNORM,
            F::A8R8G8B8Srgb => &BGRA8_SRGB,
            F::A2B10G10R10Unorm => &RGB10A2_UNORM,
            F::A2B10G10R10Uint => &RGB10A2_UINT,
            F::A8B8G8R8Unorm => &RGBA8_UNORM,
            F::A8B8G8R8Srgb => &RGBA8_SRGB,
            F::A8B8G8R8Snorm => &RGBA8_SNORM,
            F::A8B8G8R8Sint => &RGBA8_SINT,
            F::A8B8G8R8Uint => &RGBA8_UINT,
            F::R16G16Unorm => &RG16_UNORM,
            F::R16G16Snorm => &RG16_SNORM,
            F::R16G16Sint => &RG16_SINT,
            F::R16G16Uint => &RG16_UINT,
            F::R16G16Float => &RG16_FLOAT,
            F::A2R10G10B10Unorm => &BGR10A2_UNORM,
            F::B10G11R11Float => &R11G11B10_FLOAT,
            F::R32Sint => &R32_SINT,
            F::R32Uint => &R32_UINT,
            F::R32Float => &R32_FLOAT,
            F::X8R8G8B8Unorm => &BGRX8_UNORM,
            F::X8R8G8B8Srgb => &BGRX8_SRGB,
            F::R5G6B5Unorm => &B5G6R5_UNORM,
            F::A1R5G5B5Unorm => &BGR5A1_UNORM,
            F::R8G8Unorm => &RG8_UNORM,
            F::R8G8Snorm => &RG8_SNORM,
            F::R8G8Sint => &RG8_SINT,
            F::R8G8Uint => &RG8_UINT,
            F::R16Unorm => &R16_UNORM,
            F::R16Snorm => &R16_SNORM,
            F::R16Sint => &R16_SINT,
            F::R16Uint => &R16_UINT,
            F::R16Float => &R16_FLOAT,
            F::R8Unorm => &R8_UNORM,
            F::R8Snorm => &R8_SNORM,
            F::R8Sint => &R8_SINT,
            F::R8Uint => &R8_UINT,
            F::X1R5G5B5Unorm => &BGR5X1_UNORM,
            F::X8B8G8R8Unorm => &RGBX8_UNORM,
            F::X8B8G8R8Srgb => &RGBX8_SRGB,
        }))
    }

    /// Decodes `input` (whole pixels) into `output`, four floats per pixel.
    pub fn decode(&self, input: &[u8], output: &mut [f32]) {
        for (pixel, out) in input
            .chunks_exact(self.bytes)
            .zip(output.as_chunks_mut::<4>().0.iter_mut())
        {
            let mut word = [0u8; 16];
            word[..self.bytes].copy_from_slice(pixel);
            let word = u128::from_le_bytes(word);
            *out = [0.0, 0.0, 0.0, 1.0];
            let mut shift = 0;
            for f in self.fields {
                let raw = ((word >> shift) & ((1u128 << f.bits) - 1)) as u32;
                shift += f.bits;
                if f.channel != Pad {
                    out[f.channel as usize] = decode_field(f, raw);
                }
            }
        }
    }

    /// Encodes four floats per pixel of `input` into `output` (whole pixels).
    pub fn encode(&self, input: &[f32], output: &mut [u8]) {
        for (pixel, out) in input
            .as_chunks::<4>()
            .0
            .iter()
            .zip(output.chunks_exact_mut(self.bytes))
        {
            let mut word = 0u128;
            let mut shift = 0;
            for f in self.fields {
                if f.channel != Pad {
                    word |= u128::from(encode_field(f, pixel[f.channel as usize])) << shift;
                }
                shift += f.bits;
            }
            out.copy_from_slice(&word.to_le_bytes()[..self.bytes]);
        }
    }
}

fn sign_extend(raw: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((raw << shift) as i32) >> shift
}

fn mask(bits: u32) -> u32 {
    if bits >= 32 {
        u32::MAX
    } else {
        (1 << bits) - 1
    }
}

fn decode_field(f: &Field, raw: u32) -> f32 {
    match f.kind {
        Kind::Unorm => raw as f32 / mask(f.bits) as f32,
        Kind::Snorm => (sign_extend(raw, f.bits) as f32 / mask(f.bits - 1) as f32).max(-1.0),
        Kind::Sint => sign_extend(raw, f.bits) as f32,
        Kind::Uint => raw as f32,
        Kind::Float => match f.bits {
            32 => f32::from_bits(raw),
            16 => half_to_f32(raw as u16),
            bits => small_float_to_f32(raw, bits - 5),
        },
        Kind::Srgb => {
            let value = raw as f32 / mask(f.bits) as f32;
            if f.channel == A {
                value
            } else {
                srgb_to_linear(value)
            }
        }
    }
}

fn encode_field(f: &Field, value: f32) -> u32 {
    let max = mask(f.bits);
    match f.kind {
        Kind::Unorm => (value.clamp(0.0, 1.0) * max as f32 + 0.5) as u32,
        Kind::Snorm => {
            let limit = mask(f.bits - 1) as f32;
            ((value.clamp(-1.0, 1.0) * limit).round() as i32 as u32) & max
        }
        Kind::Sint => {
            let limit = mask(f.bits - 1) as f32;
            (value.clamp(-limit - 1.0, limit) as i32 as u32) & max
        }
        Kind::Uint => value.clamp(0.0, max as f32) as u32,
        Kind::Float => match f.bits {
            32 => value.to_bits(),
            16 => u32::from(f32_to_half(value)),
            bits => f32_to_small_float(value, bits - 5),
        },
        Kind::Srgb => {
            let value = if f.channel == A {
                value
            } else {
                linear_to_srgb(value)
            };
            (value.clamp(0.0, 1.0) * max as f32 + 0.5) as u32
        }
    }
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn half_to_f32(half: u16) -> f32 {
    let sign = u32::from(half & 0x8000) << 16;
    let exponent = u32::from((half >> 10) & 0x1f);
    let mantissa = u32::from(half & 0x3ff);
    match exponent {
        0 => {
            let magnitude = mantissa as f32 * (1.0 / 16_777_216.0);
            f32::from_bits(sign | magnitude.to_bits())
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | (mantissa << 13)),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | (mantissa << 13)),
    }
}

fn f32_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let half_exponent = exponent - 112;
    if half_exponent >= 0x1f {
        return sign | 0x7c00;
    }
    let (kept, shift, full) = if half_exponent > 0 {
        (
            ((half_exponent as u32) << 10) | (mantissa >> 13),
            13,
            mantissa,
        )
    } else if half_exponent >= -10 {
        let full = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        (full >> shift, shift, full)
    } else {
        return sign;
    };
    let dropped = full & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    let round_up = dropped > halfway || (dropped == halfway && kept & 1 != 0);
    sign | (kept + u32::from(round_up)) as u16
}

fn small_float_to_f32(raw: u32, mantissa_bits: u32) -> f32 {
    let exponent = raw >> mantissa_bits;
    let mantissa = raw & mask(mantissa_bits);
    let scale = 1.0 / (1u32 << mantissa_bits) as f32;
    match exponent {
        0 => mantissa as f32 * scale * f32::from_bits(0x3880_0000),
        0x1f if mantissa == 0 => f32::INFINITY,
        0x1f => f32::NAN,
        _ => (1.0 + mantissa as f32 * scale) * f32::from_bits((exponent + 112) << 23),
    }
}

fn f32_to_small_float(value: f32, mantissa_bits: u32) -> u32 {
    if value.is_nan() || value <= 0.0 {
        return 0;
    }
    if value.is_infinite() {
        return 0x1f << mantissa_bits;
    }
    let steps = (1u32 << mantissa_bits) as f32;
    let value = value.min((2.0 - 1.0 / steps) * 32768.0);
    let exponent = (value.to_bits() >> 23) as i32 - 127;
    if exponent < -14 {
        let denormal = value * f32::from_bits((127 + 14 + mantissa_bits) << 23);
        return denormal.round() as u32;
    }
    let fraction = value * f32::from_bits(((127 - exponent) as u32) << 23) - 1.0;
    (((exponent + 15) as u32) << mantissa_bits) + (fraction * steps).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_exactly_representable_pattern_round_trips() {
        for &format in RenderTargetFormat::ALL {
            let Some(layout) = Layout::of(format) else {
                continue;
            };
            let lossy = layout.fields.iter().any(|f| {
                f.channel == Pad
                    || f.kind == Snorm
                    || (f.bits == 32 && matches!(f.kind, Sint | Uint))
            });
            if lossy {
                continue;
            }
            let pixel: Vec<u8> = (0..layout.bytes as u8)
                .map(|i| 0x35u8.wrapping_add(i.wrapping_mul(0x11)))
                .collect();
            let mut ir = [0.0f32; 4];
            layout.decode(&pixel, &mut ir);
            if ir.iter().any(|v| v.is_nan()) {
                continue;
            }
            let mut out = vec![0u8; layout.bytes];
            layout.encode(&ir, &mut out);
            assert_eq!(out, pixel, "{format:?}");
        }
    }

    #[test]
    fn bgra_and_rgba_swap_red_and_blue() {
        let layout = Layout::of(RenderTargetFormat::A8B8G8R8Unorm).unwrap();
        let mut ir = [0.0f32; 4];
        layout.decode(&[0x10, 0x20, 0x30, 0x40], &mut ir);
        assert_eq!(ir, [16.0 / 255.0, 32.0 / 255.0, 48.0 / 255.0, 64.0 / 255.0]);
        let bgra = Layout::of(RenderTargetFormat::A8R8G8B8Unorm).unwrap();
        let mut out = [0u8; 4];
        bgra.encode(&ir, &mut out);
        assert_eq!(out, [0x30, 0x20, 0x10, 0x40]);
    }

    #[test]
    fn rgb10a2_packs_red_lowest() {
        let layout = Layout::of(RenderTargetFormat::A2B10G10R10Unorm).unwrap();
        let mut out = [0u8; 4];
        layout.encode(&[1.0, 0.0, 0.0, 1.0], &mut out);
        assert_eq!(u32::from_le_bytes(out), 0xc000_03ff);
        layout.encode(&[0.0, 0.0, 1.0, 0.0], &mut out);
        assert_eq!(u32::from_le_bytes(out), 0x3ff0_0000);
    }

    #[test]
    fn r5g6b5_packs_blue_lowest() {
        let layout = Layout::of(RenderTargetFormat::R5G6B5Unorm).unwrap();
        let mut ir = [0.0f32; 4];
        layout.decode(&0xf800u16.to_le_bytes(), &mut ir);
        assert_eq!(ir, [1.0, 0.0, 0.0, 1.0]);
        layout.decode(&0x001fu16.to_le_bytes(), &mut ir);
        assert_eq!(ir, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn half_and_small_floats_convert() {
        let layout = Layout::of(RenderTargetFormat::R16G16B16A16Float).unwrap();
        let mut bytes = [0u8; 8];
        layout.encode(&[1.0, -2.5, 0.0, 65504.0], &mut bytes);
        assert_eq!(bytes, [0x00, 0x3c, 0x00, 0xc1, 0x00, 0x00, 0xff, 0x7b]);
        let mut ir = [0.0f32; 4];
        layout.decode(&bytes, &mut ir);
        assert_eq!(ir, [1.0, -2.5, 0.0, 65504.0]);
        layout.decode(&[0x01, 0x00, 0, 0, 0, 0, 0, 0], &mut ir);
        assert_eq!(ir[0], 2.0f32.powi(-24));

        let packed = Layout::of(RenderTargetFormat::B10G11R11Float).unwrap();
        let mut word = [0u8; 4];
        packed.encode(&[1.0, 2.0, 0.5, 1.0], &mut word);
        let word = u32::from_le_bytes(word);
        assert_eq!(word & 0x7ff, 0x3c0);
        assert_eq!((word >> 11) & 0x7ff, 0x400);
        assert_eq!(word >> 22, 0x1c0);
        let mut ir = [0.0f32; 4];
        packed.decode(&word.to_le_bytes(), &mut ir);
        assert_eq!(ir, [1.0, 2.0, 0.5, 1.0]);
    }

    #[test]
    fn srgb_decodes_to_linear_and_keeps_alpha_linear() {
        let layout = Layout::of(RenderTargetFormat::A8B8G8R8Srgb).unwrap();
        let mut ir = [0.0f32; 4];
        layout.decode(&[255, 0, 128, 128], &mut ir);
        assert_eq!(ir[0], 1.0);
        assert_eq!(ir[1], 0.0);
        assert!((ir[2] - 0.2158).abs() < 1e-3);
        assert!((ir[3] - 128.0 / 255.0).abs() < 1e-6);
        let pixel = [7, 99, 200, 33];
        layout.decode(&pixel, &mut ir);
        let mut out = [0u8; 4];
        layout.encode(&ir, &mut out);
        assert_eq!(out, pixel);
    }

    #[test]
    fn integer_formats_keep_values_and_saturate() {
        let layout = Layout::of(RenderTargetFormat::R8G8Sint).unwrap();
        let mut ir = [0.0f32; 4];
        layout.decode(&[0xfe, 0x7f], &mut ir);
        assert_eq!(&ir[..2], &[-2.0, 127.0]);
        let mut out = [0u8; 2];
        layout.encode(&[-1000.0, 1000.0, 0.0, 0.0], &mut out);
        assert_eq!(out, [0x80, 0x7f]);
    }

    #[test]
    fn missing_channels_decode_to_zero_and_alpha_to_one() {
        let layout = Layout::of(RenderTargetFormat::R8Unorm).unwrap();
        let mut ir = [9.0f32; 4];
        layout.decode(&[255], &mut ir);
        assert_eq!(ir, [1.0, 0.0, 0.0, 1.0]);
    }
}
