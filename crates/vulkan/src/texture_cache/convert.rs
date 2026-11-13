//! CPU texel work between guest and host layouts: decoding compressed formats the host cannot
//! sample, recompressing ASTC to BC, converting formats that only a fallback host format holds,
//! and splitting depth/stencil texels into the per-aspect planes Vulkan buffer copies use.
//!
//! Guest data here is already deswizzled: tightly packed format blocks, slice after slice.

use ash::vk;
use lichii_texture::{Compression, ImageSizeError, PixelFormat, astc, bcn, etc};

use crate::format::{Conversion, HostFormat};

/// Buffer bytes one host level takes: the colour or depth plane, then the stencil plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HostLayout {
    pub main: usize,
    pub stencil: usize,
}

impl HostLayout {
    pub fn total(self) -> usize {
        self.main + self.stencil
    }
}

/// Bytes per texel of the depth plane and whether a stencil plane follows, for depth formats.
pub(super) fn depth_planes(format: vk::Format) -> Option<(usize, bool)> {
    Some(match format {
        vk::Format::D16_UNORM => (2, false),
        vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => (4, false),
        vk::Format::D16_UNORM_S8_UINT => (2, true),
        vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT => (4, true),
        vk::Format::S8_UINT => (0, true),
        _ => return None,
    })
}

fn texel_bytes(format: vk::Format) -> u32 {
    match format {
        vk::Format::R8_UNORM | vk::Format::R8_SNORM => 1,
        vk::Format::R8G8_UNORM | vk::Format::R8G8_SNORM => 2,
        vk::Format::R16G16B16A16_SFLOAT => 8,
        _ => 4,
    }
}

/// Bytes and block size of one host texel block of a colour image.
fn host_block(format: PixelFormat, host: &HostFormat) -> (u32, u32, u32) {
    match host.conversion {
        Conversion::None => (
            format.bytes_per_block(),
            format.block_width(),
            format.block_height(),
        ),
        Conversion::DecodeToBc(Compression::Bc1) => (8, 4, 4),
        Conversion::DecodeToBc(_) => (16, 4, 4),
        Conversion::Decode | Conversion::Convert => (texel_bytes(host.format), 1, 1),
    }
}

pub(super) fn host_layout(
    format: PixelFormat,
    host: &HostFormat,
    extent: (u32, u32, u32),
) -> HostLayout {
    let (w, h, d) = extent;
    let texels = w as usize * h as usize * d as usize;
    if let Some((depth, stencil)) = depth_planes(host.format) {
        return HostLayout {
            main: texels * depth,
            stencil: if stencil { texels } else { 0 },
        };
    }
    let (bytes, bw, bh) = host_block(format, host);
    HostLayout {
        main: w.div_ceil(bw) as usize * h.div_ceil(bh) as usize * d as usize * bytes as usize,
        stencil: 0,
    }
}

/// Guest bytes of one deswizzled level.
pub(super) fn guest_level_bytes(format: PixelFormat, extent: (u32, u32, u32)) -> usize {
    let (w, h, d) = extent;
    w.div_ceil(format.block_width()) as usize
        * h.div_ceil(format.block_height()) as usize
        * d as usize
        * format.bytes_per_block() as usize
}

/// A texel conversion between a guest format and the fallback host format holding it.
struct ColorConversion {
    guest_bytes: usize,
    host_bytes: usize,
    to_host: fn(u32) -> [u8; 4],
    to_guest: fn([u8; 4]) -> u32,
}

const fn expand(value: u32, bits: u32) -> u8 {
    let max = (1 << bits) - 1;
    ((value * 255 + max / 2) / max) as u8
}

const fn quantize(value: u8, bits: u32) -> u32 {
    let max = (1 << bits) - 1;
    (value as u32 * max + 127) / 255
}

fn color_conversion(format: PixelFormat, host: vk::Format) -> Option<ColorConversion> {
    use PixelFormat::*;
    let rgba8 = host == vk::Format::A8B8G8R8_UNORM_PACK32;
    Some(match format {
        A4B4G4R4Unorm if rgba8 => ColorConversion {
            guest_bytes: 2,
            host_bytes: 4,
            to_host: |v| [0, 4, 8, 12].map(|s| expand((v >> s) & 15, 4)),
            to_guest: |t| (0..4).fold(0, |v, i| v | quantize(t[i], 4) << (4 * i)),
        },
        B5G6R5Unorm if rgba8 => ColorConversion {
            guest_bytes: 2,
            host_bytes: 4,
            to_host: |v| {
                [
                    expand(v & 31, 5),
                    expand((v >> 5) & 63, 6),
                    expand(v >> 11, 5),
                    255,
                ]
            },
            to_guest: |t| quantize(t[0], 5) | quantize(t[1], 6) << 5 | quantize(t[2], 5) << 11,
        },
        A1B5G5R5Unorm if rgba8 => ColorConversion {
            guest_bytes: 2,
            host_bytes: 4,
            to_host: |v| {
                [
                    expand(v & 31, 5),
                    expand((v >> 5) & 31, 5),
                    expand((v >> 10) & 31, 5),
                    expand(v >> 15, 1),
                ]
            },
            to_guest: |t| {
                quantize(t[0], 5)
                    | quantize(t[1], 5) << 5
                    | quantize(t[2], 5) << 10
                    | quantize(t[3], 1) << 15
            },
        },
        A1R5G5B5Unorm if rgba8 => ColorConversion {
            guest_bytes: 2,
            host_bytes: 4,
            to_host: |v| {
                [
                    expand((v >> 10) & 31, 5),
                    expand((v >> 5) & 31, 5),
                    expand(v & 31, 5),
                    expand(v >> 15, 1),
                ]
            },
            to_guest: |t| {
                quantize(t[2], 5)
                    | quantize(t[1], 5) << 5
                    | quantize(t[0], 5) << 10
                    | quantize(t[3], 1) << 15
            },
        },
        A5B5G5R1Unorm if rgba8 => ColorConversion {
            guest_bytes: 2,
            host_bytes: 4,
            to_host: |v| {
                [
                    expand(v & 1, 1),
                    expand((v >> 1) & 31, 5),
                    expand((v >> 6) & 31, 5),
                    expand(v >> 11, 5),
                ]
            },
            to_guest: |t| {
                quantize(t[0], 1)
                    | quantize(t[1], 5) << 1
                    | quantize(t[2], 5) << 6
                    | quantize(t[3], 5) << 11
            },
        },
        G4R4Unorm if host == vk::Format::R8G8_UNORM => ColorConversion {
            guest_bytes: 1,
            host_bytes: 2,
            to_host: |v| [expand(v & 15, 4), expand(v >> 4, 4), 0, 0],
            to_guest: |t| quantize(t[0], 4) | quantize(t[1], 4) << 4,
        },
        _ => return None,
    })
}

fn read_le(bytes: &[u8]) -> u32 {
    bytes.iter().rev().fold(0, |v, &b| (v << 8) | u32::from(b))
}

/// A depth/stencil texel between its guest encoding and the host planes: depth normalized to
/// [0, 1] (or the float itself) and the stencil byte.
fn read_guest_depth(format: PixelFormat, texel: &[u8]) -> (f64, u8) {
    use PixelFormat::*;
    let word = read_le(&texel[..texel.len().min(4)]);
    match format {
        D16Unorm => (f64::from(word) / 65535.0, 0),
        X8D24Unorm => (f64::from(word & 0xff_ffff) / 16_777_215.0, 0),
        D24UnormS8Uint => (
            f64::from(word & 0xff_ffff) / 16_777_215.0,
            (word >> 24) as u8,
        ),
        S8UintD24Unorm => (f64::from(word >> 8) / 16_777_215.0, word as u8),
        D32Float => (f64::from(f32::from_bits(word)), 0),
        D32FloatS8Uint => (f64::from(f32::from_bits(word)), texel[4]),
        S8Uint => (0.0, texel[0]),
        _ => (0.0, 0),
    }
}

fn write_guest_depth(format: PixelFormat, depth: f64, stencil: u8, texel: &mut [u8]) {
    use PixelFormat::*;
    let unorm = |max: f64| (depth.clamp(0.0, 1.0) * max).round() as u32;
    let bits = match format {
        D16Unorm => unorm(65535.0),
        X8D24Unorm | D24UnormS8Uint | S8UintD24Unorm => unorm(16_777_215.0),
        D32Float | D32FloatS8Uint => (depth as f32).to_bits(),
        _ => 0,
    };
    write_guest_depth_bits(format, bits, stencil, texel);
}

/// Writes a depth/stencil texel from depth already in the guest format's encoding: unorm bits
/// for D16 and D24, float bits for D32.
fn write_guest_depth_bits(format: PixelFormat, depth: u32, stencil: u8, texel: &mut [u8]) {
    use PixelFormat::*;
    let depth24 = depth & 0xff_ffff;
    let word = match format {
        D16Unorm => {
            texel.copy_from_slice(&(depth as u16).to_le_bytes());
            return;
        }
        X8D24Unorm => depth24,
        D24UnormS8Uint => depth24 | u32::from(stencil) << 24,
        S8UintD24Unorm => depth24 << 8 | u32::from(stencil),
        D32Float => depth,
        D32FloatS8Uint => {
            texel[..4].copy_from_slice(&depth.to_le_bytes());
            texel[4..].copy_from_slice(&[stencil, 0, 0, 0]);
            return;
        }
        S8Uint => {
            texel[0] = stencil;
            return;
        }
        _ => return,
    };
    texel.copy_from_slice(&word.to_le_bytes());
}

/// Whether the host depth plane stores depth in the guest format's encoding, so a download
/// copies its bits instead of converting through a float.
fn same_depth_encoding(format: PixelFormat, host: vk::Format) -> bool {
    use PixelFormat::*;
    use vk::Format as F;
    match format {
        D16Unorm => matches!(host, F::D16_UNORM | F::D16_UNORM_S8_UINT),
        X8D24Unorm | D24UnormS8Uint | S8UintD24Unorm => {
            matches!(host, F::X8_D24_UNORM_PACK32 | F::D24_UNORM_S8_UINT)
        }
        D32Float | D32FloatS8Uint => matches!(host, F::D32_SFLOAT | F::D32_SFLOAT_S8_UINT),
        S8Uint => host == F::S8_UINT,
        _ => false,
    }
}

fn write_host_depth(host: vk::Format, depth: f64, out: &mut [u8]) {
    match host {
        vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => {
            let v = (depth.clamp(0.0, 1.0) * 65535.0).round() as u16;
            out.copy_from_slice(&v.to_le_bytes());
        }
        vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D24_UNORM_S8_UINT => {
            let v = (depth.clamp(0.0, 1.0) * 16_777_215.0).round() as u32;
            out.copy_from_slice(&v.to_le_bytes());
        }
        _ => out.copy_from_slice(&(depth as f32).to_bits().to_le_bytes()),
    }
}

fn read_host_depth(host: vk::Format, bytes: &[u8]) -> f64 {
    let word = read_le(bytes);
    match host {
        vk::Format::D16_UNORM | vk::Format::D16_UNORM_S8_UINT => f64::from(word) / 65535.0,
        vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D24_UNORM_S8_UINT => {
            f64::from(word & 0xff_ffff) / 16_777_215.0
        }
        _ => f64::from(f32::from_bits(word)),
    }
}

/// Why a level cannot be converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(super) enum ConvertError {
    #[error(transparent)]
    Size(#[from] ImageSizeError),
    #[error("no conversion from {format:?} to {host:?}")]
    Unsupported {
        format: PixelFormat,
        host: vk::Format,
    },
}

/// Converts one deswizzled guest level into the host buffer layout (`host_layout` bytes in
/// `out`). `scratch` holds intermediate texels and keeps its capacity between calls.
pub(super) fn to_host(
    format: PixelFormat,
    host: &HostFormat,
    extent: (u32, u32, u32),
    guest: &[u8],
    out: &mut [u8],
    scratch: &mut Vec<u8>,
) -> Result<(), ConvertError> {
    let unsupported = ConvertError::Unsupported {
        format,
        host: host.format,
    };
    let (w, h, d) = extent;
    let texels = w as usize * h as usize * d as usize;
    if let Some((depth_bytes, stencil)) = depth_planes(host.format) {
        let guest_bytes = format.bytes_per_block() as usize;
        let (depth_out, stencil_out) = out.split_at_mut(texels * depth_bytes);
        for (i, texel) in guest.chunks_exact(guest_bytes).take(texels).enumerate() {
            let (depth, s) = read_guest_depth(format, texel);
            if depth_bytes > 0 {
                write_host_depth(
                    host.format,
                    depth,
                    &mut depth_out[i * depth_bytes..(i + 1) * depth_bytes],
                );
            }
            if stencil {
                stencil_out[i] = s;
            }
        }
        return Ok(());
    }
    match host.conversion {
        Conversion::None => {
            let len = out.len();
            out.copy_from_slice(guest.get(..len).ok_or(ImageSizeError::Input {
                required: len,
                actual: guest.len(),
            })?);
            Ok(())
        }
        Conversion::Decode => decode(format, extent, guest, out, scratch),
        Conversion::DecodeToBc(compression) => {
            if !format.is_astc() {
                return Err(unsupported);
            }
            scratch.resize(texels * 4, 0);
            astc::decode_rgba8(
                guest,
                w,
                h,
                d,
                format.block_width(),
                format.block_height(),
                format.is_srgb(),
                scratch,
            );
            match compression {
                Compression::Bc1 => bcn::encode_bc1(scratch, w, h, d, out)?,
                _ => bcn::encode_bc3(scratch, w, h, d, out)?,
            }
            Ok(())
        }
        Conversion::Convert => {
            let c = color_conversion(format, host.format).ok_or(unsupported)?;
            for (src, dst) in guest
                .chunks_exact(c.guest_bytes)
                .zip(out.chunks_exact_mut(c.host_bytes))
            {
                dst.copy_from_slice(&(c.to_host)(read_le(src))[..c.host_bytes]);
            }
            Ok(())
        }
    }
}

/// Converts one host level read back from the GPU into deswizzled guest texels, the inverse
/// of [`to_host`] for every image the GPU can write (attachable formats).
pub(super) fn to_guest(
    format: PixelFormat,
    host: &HostFormat,
    extent: (u32, u32, u32),
    host_bytes: &[u8],
    guest: &mut [u8],
) -> Result<(), ConvertError> {
    let unsupported = ConvertError::Unsupported {
        format,
        host: host.format,
    };
    let (w, h, d) = extent;
    let texels = w as usize * h as usize * d as usize;
    if let Some((depth_bytes, stencil)) = depth_planes(host.format) {
        let guest_bytes = format.bytes_per_block() as usize;
        let (depth_in, stencil_in) = host_bytes.split_at(texels * depth_bytes);
        if same_depth_encoding(format, host.format) {
            for (i, texel) in guest.chunks_exact_mut(guest_bytes).take(texels).enumerate() {
                let depth = match depth_bytes {
                    2 => u32::from(u16::from_le_bytes([depth_in[2 * i], depth_in[2 * i + 1]])),
                    4 => u32::from_le_bytes(depth_in[4 * i..4 * i + 4].try_into().unwrap()),
                    _ => 0,
                };
                let s = if stencil { stencil_in[i] } else { 0 };
                write_guest_depth_bits(format, depth, s, texel);
            }
            return Ok(());
        }
        for (i, texel) in guest.chunks_exact_mut(guest_bytes).take(texels).enumerate() {
            let depth = if depth_bytes > 0 {
                read_host_depth(
                    host.format,
                    &depth_in[i * depth_bytes..(i + 1) * depth_bytes],
                )
            } else {
                0.0
            };
            let s = if stencil { stencil_in[i] } else { 0 };
            write_guest_depth(format, depth, s, texel);
        }
        return Ok(());
    }
    match host.conversion {
        Conversion::None => {
            let len = guest.len();
            guest.copy_from_slice(host_bytes.get(..len).ok_or(ImageSizeError::Input {
                required: len,
                actual: host_bytes.len(),
            })?);
            Ok(())
        }
        Conversion::Convert => {
            let c = color_conversion(format, host.format).ok_or(unsupported)?;
            for (src, dst) in host_bytes
                .chunks_exact(c.host_bytes)
                .zip(guest.chunks_exact_mut(c.guest_bytes))
            {
                let mut texel = [0; 4];
                texel[..c.host_bytes].copy_from_slice(src);
                dst.copy_from_slice(&(c.to_guest)(texel).to_le_bytes()[..c.guest_bytes]);
            }
            Ok(())
        }
        Conversion::Decode | Conversion::DecodeToBc(_) => Err(unsupported),
    }
}

/// Narrows 16-bit EAC channels to the 8-bit host formats they decode into.
fn narrow16(wide: &[u8], out: &mut [u8], signed: bool) {
    for (&src, dst) in wide.as_chunks::<2>().0.iter().zip(out.iter_mut()) {
        let v = u16::from_le_bytes(src);
        *dst = if signed {
            (f64::from(v as i16) * 127.0 / 32767.0)
                .round()
                .clamp(-127.0, 127.0) as i8 as u8
        } else {
            ((u32::from(v) * 255 + 32767) / 65535) as u8
        };
    }
}

type ImageDecoder = fn(&[u8], u32, u32, u32, &mut [u8]) -> Result<(), ImageSizeError>;

fn decode(
    format: PixelFormat,
    extent: (u32, u32, u32),
    guest: &[u8],
    out: &mut [u8],
    scratch: &mut Vec<u8>,
) -> Result<(), ConvertError> {
    use PixelFormat::*;
    let (w, h, d) = extent;
    if format.is_astc() {
        astc::decode_rgba8(
            guest,
            w,
            h,
            d,
            format.block_width(),
            format.block_height(),
            format.is_srgb(),
            out,
        );
        return Ok(());
    }
    let mut eac = |decoder: ImageDecoder, signed: bool| -> Result<(), ConvertError> {
        scratch.resize(out.len() * 2, 0);
        decoder(guest, w, h, d, scratch)?;
        narrow16(scratch, out, signed);
        Ok(())
    };
    match format {
        Bc1RgbaUnorm | Bc1RgbaSrgb => bcn::decode_bc1(guest, w, h, d, out)?,
        Bc2Unorm | Bc2Srgb => bcn::decode_bc2(guest, w, h, d, out)?,
        Bc3Unorm | Bc3Srgb => bcn::decode_bc3(guest, w, h, d, out)?,
        Bc4Unorm => bcn::decode_bc4_unorm(guest, w, h, d, out)?,
        Bc4Snorm => bcn::decode_bc4_snorm(guest, w, h, d, out)?,
        Bc5Unorm => bcn::decode_bc5_unorm(guest, w, h, d, out)?,
        Bc5Snorm => bcn::decode_bc5_snorm(guest, w, h, d, out)?,
        Bc6hUfloat => bcn::decode_bc6h_ufloat(guest, w, h, d, out)?,
        Bc6hSfloat => bcn::decode_bc6h_sfloat(guest, w, h, d, out)?,
        Bc7Unorm | Bc7Srgb => bcn::decode_bc7(guest, w, h, d, out)?,
        Etc2RgbUnorm | Etc2RgbSrgb => etc::decode_etc2_rgb(guest, w, h, d, out)?,
        Etc2RgbPtaUnorm | Etc2RgbPtaSrgb => etc::decode_etc2_rgb_pta(guest, w, h, d, out)?,
        Etc2RgbaUnorm | Etc2RgbaSrgb => etc::decode_etc2_rgba(guest, w, h, d, out)?,
        EacR11Unorm => eac(etc::decode_eac_r11_unorm, false)?,
        EacR11Snorm => eac(etc::decode_eac_r11_snorm, true)?,
        EacR11G11Unorm => eac(etc::decode_eac_rg11_unorm, false)?,
        EacR11G11Snorm => eac(etc::decode_eac_rg11_snorm, true)?,
        _ => {
            return Err(ConvertError::Unsupported {
                format,
                host: vk::Format::UNDEFINED,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host(format: vk::Format, conversion: Conversion) -> HostFormat {
        HostFormat {
            format,
            conversion,
            swizzle: vk::ComponentMapping::default(),
            attachable: true,
            storage: false,
        }
    }

    #[test]
    fn packed_formats_round_trip_through_rgba8() {
        let rgba8 = host(vk::Format::A8B8G8R8_UNORM_PACK32, Conversion::Convert);
        for format in [
            PixelFormat::A4B4G4R4Unorm,
            PixelFormat::B5G6R5Unorm,
            PixelFormat::A1B5G5R5Unorm,
            PixelFormat::A1R5G5B5Unorm,
            PixelFormat::A5B5G5R1Unorm,
        ] {
            let guest: Vec<u8> = (0..=u16::MAX)
                .step_by(7)
                .flat_map(|v| v.to_le_bytes())
                .collect();
            let texels = guest.len() / 2;
            let extent = (texels as u32, 1, 1);
            let mut out = vec![0; host_layout(format, &rgba8, extent).main];
            to_host(format, &rgba8, extent, &guest, &mut out, &mut Vec::new()).unwrap();
            let mut back = vec![0; guest.len()];
            to_guest(format, &rgba8, extent, &out, &mut back).unwrap();
            assert!(back == guest, "{format:?} does not round-trip");
        }
        let a1b5g5r5 = 0b1_00011_00010_00001u16.to_le_bytes();
        let mut out = [0; 4];
        to_host(
            PixelFormat::A1B5G5R5Unorm,
            &rgba8,
            (1, 1, 1),
            &a1b5g5r5,
            &mut out,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(out, [8, 16, 25, 255]);
    }

    #[test]
    fn downloads_copy_depth_bits_the_host_stores_in_the_guest_encoding() {
        let cases: [(PixelFormat, vk::Format, &[u8], &[u8]); 5] = [
            (
                PixelFormat::D32Float,
                vk::Format::D32_SFLOAT,
                &0.3f32.to_bits().to_le_bytes(),
                &0.3f32.to_bits().to_le_bytes(),
            ),
            (
                PixelFormat::D16Unorm,
                vk::Format::D16_UNORM,
                &[0x34, 0x12],
                &[0x34, 0x12],
            ),
            (
                PixelFormat::S8UintD24Unorm,
                vk::Format::D24_UNORM_S8_UINT,
                &[0x78, 0x56, 0x34, 0xee, 0x12],
                &0x3456_7812u32.to_le_bytes(),
            ),
            (
                PixelFormat::X8D24Unorm,
                vk::Format::X8_D24_UNORM_PACK32,
                &[0x78, 0x56, 0x34, 0xee],
                &0x0034_5678u32.to_le_bytes(),
            ),
            (
                PixelFormat::D32FloatS8Uint,
                vk::Format::D32_SFLOAT_S8_UINT,
                &[0, 0, 0x80, 0x3f, 9],
                &[0, 0, 0x80, 0x3f, 9, 0, 0, 0],
            ),
        ];
        for (format, host_format, planes, expected) in cases {
            assert!(same_depth_encoding(format, host_format));
            let mut out = vec![0; expected.len()];
            to_guest(
                format,
                &host(host_format, Conversion::None),
                (1, 1, 1),
                planes,
                &mut out,
            )
            .unwrap();
            assert_eq!(out, expected, "{format:?} from {host_format:?}");
        }
        assert!(!same_depth_encoding(
            PixelFormat::D24UnormS8Uint,
            vk::Format::D32_SFLOAT_S8_UINT
        ));
    }

    #[test]
    fn depth_texels_split_into_planes_and_back() {
        let d24s8 = host(vk::Format::D24_UNORM_S8_UINT, Conversion::None);
        let guest: Vec<u8> = [0x12_345678u32, 0xff_000001, 0x00_ffffff]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let layout = host_layout(PixelFormat::D24UnormS8Uint, &d24s8, (3, 1, 1));
        assert_eq!(
            layout,
            HostLayout {
                main: 12,
                stencil: 3
            }
        );
        let mut planes = vec![0; layout.total()];
        to_host(
            PixelFormat::D24UnormS8Uint,
            &d24s8,
            (3, 1, 1),
            &guest,
            &mut planes,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(&planes[..4], &0x345678u32.to_le_bytes());
        assert_eq!(&planes[12..], &[0x12, 0xff, 0x00]);
        let mut back = vec![0; guest.len()];
        to_guest(
            PixelFormat::D24UnormS8Uint,
            &d24s8,
            (3, 1, 1),
            &planes,
            &mut back,
        )
        .unwrap();
        assert_eq!(back, guest);

        let z24s8: Vec<u8> = 0x3456_7812u32.to_le_bytes().to_vec();
        let mut planes = vec![0; 5];
        to_host(
            PixelFormat::S8UintD24Unorm,
            &d24s8,
            (1, 1, 1),
            &z24s8,
            &mut planes,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(planes, [0x78, 0x56, 0x34, 0, 0x12]);

        let d32 = host(vk::Format::D32_SFLOAT, Conversion::Convert);
        let d16 = 0x8000u16.to_le_bytes();
        let mut plane = [0; 4];
        to_host(
            PixelFormat::D16Unorm,
            &d32,
            (1, 1, 1),
            &d16,
            &mut plane,
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!(f32::from_le_bytes(plane), (32768.0f64 / 65535.0) as f32);
        let mut back = [0; 2];
        to_guest(PixelFormat::D16Unorm, &d32, (1, 1, 1), &plane, &mut back).unwrap();
        assert_eq!(back, d16);
    }

    #[test]
    fn eac_narrows_to_eight_bits() {
        let mut out = [0; 3];
        narrow16(&[0, 0, 0xff, 0xff, 0x00, 0x80], &mut out, false);
        assert_eq!(out, [0, 255, 128]);
        narrow16(&[0xff, 0x7f, 0x01, 0x80], &mut out[..2], true);
        assert_eq!(out[..2], [127, (-127i8) as u8]);
    }
}
