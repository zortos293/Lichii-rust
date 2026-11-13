//! Texture headers (TIC entries): the 32-byte descriptors shaders sample through.
//!
//! Maxwell and Ampere share the layout (the Ampere 3D class, which NVN2 drives on the Switch 2,
//! reads the same block-linear and pitch header versions), so one decoder serves both consoles;
//! only the GOB layout of the texels differs, and that is the caller's
//! [`GobLayout`](crate::block_linear::GobLayout).

use crate::PixelFormat;
use crate::guest::{ComponentType, TextureFormat};

/// What the texture is, as the header's type field names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureType {
    D1,
    D2,
    D3,
    Cube,
    D1Array,
    D2Array,
    D1Buffer,
    D2NoMipmap,
    CubeArray,
}

impl TextureType {
    const fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::D1,
            1 => Self::D2,
            2 => Self::D3,
            3 => Self::Cube,
            4 => Self::D1Array,
            5 => Self::D2Array,
            6 => Self::D1Buffer,
            7 => Self::D2NoMipmap,
            8 => Self::CubeArray,
            _ => return None,
        })
    }
}

/// Where a sampled component comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SwizzleSource {
    Zero,
    R,
    G,
    B,
    A,
    OneInt,
    OneFloat,
}

impl SwizzleSource {
    /// Reads each component from itself.
    pub const IDENTITY: [Self; 4] = [Self::R, Self::G, Self::B, Self::A];

    const fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::Zero,
            2 => Self::R,
            3 => Self::G,
            4 => Self::B,
            5 => Self::A,
            6 => Self::OneInt,
            7 => Self::OneFloat,
            _ => return None,
        })
    }
}

/// How the texels are laid out in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TextureMemory {
    /// Tiled in GOBs; block dimensions are log2 GOB counts.
    BlockLinear {
        block_height_log2: u32,
        block_depth_log2: u32,
        /// Row alignment in GOBs, log2.
        tile_width_spacing: u32,
    },
    /// Rows `pitch` bytes apart.
    Pitch { pitch: u32 },
    /// A texel buffer (`D1Buffer`); `width` is its texel count.
    Buffer,
}

/// Multisample layout of the texture, as the header encodes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MsaaMode {
    X1,
    X2,
    X4,
    X8,
    X16,
}

impl MsaaMode {
    /// Decodes the hardware's multisample mode value (texture headers and the 3D engine's
    /// `anti_alias_samples_mode` share it).
    pub const fn from_raw(raw: u32) -> Option<Self> {
        Some(match raw {
            0 => Self::X1,
            1 | 5 => Self::X2,
            2 | 8 | 9 => Self::X4,
            3 | 4 | 10 | 11 => Self::X8,
            6 => Self::X16,
            _ => return None,
        })
    }

    pub const fn samples(self) -> u32 {
        match self {
            Self::X1 => 1,
            Self::X2 => 2,
            Self::X4 => 4,
            Self::X8 => 8,
            Self::X16 => 16,
        }
    }

    /// Samples across and down one pixel: guest memory stores the samples as a wider and taller
    /// image of this many texels per pixel.
    pub const fn grid(self) -> (u32, u32) {
        match self {
            Self::X1 => (1, 1),
            Self::X2 => (2, 1),
            Self::X4 => (2, 2),
            Self::X8 => (4, 2),
            Self::X16 => (4, 4),
        }
    }
}

/// Why a header cannot be decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum TicError {
    #[error("unknown texture format {format:#x} with component types {components:?}")]
    Format { format: u32, components: [u32; 4] },
    #[error("unknown texture type {0}")]
    Type(u32),
    #[error("unknown header version {0}")]
    HeaderVersion(u32),
    #[error("unknown swizzle source {0}")]
    Swizzle(u32),
    #[error("unknown multisample mode {0}")]
    Msaa(u32),
}

/// A decoded texture header.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TextureDescriptor {
    /// GPU virtual address of the first texel.
    pub address: u64,
    pub format: PixelFormat,
    /// Source of the sampled R, G, B and A.
    pub swizzle: [SwizzleSource; 4],
    pub texture_type: TextureType,
    pub memory: TextureMemory,
    /// Level 0 size in texels; `depth` is the array size for arrays and cube arrays (in cubes).
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Levels the image holds (`max_mip_level + 1`).
    pub levels: u32,
    /// First array layer the descriptor reads.
    pub base_layer: u32,
    /// Levels a view reads: `[min_level, max_level]` (`res_min_mip_level`, `res_max_mip_level`).
    pub min_level: u32,
    pub max_level: u32,
    pub msaa: MsaaMode,
    pub srgb: bool,
    /// Resident only where mapped (a sparse texture).
    pub sparse: bool,
    /// Sampled as a depth texture (depth comparison).
    pub depth_texture: bool,
    pub normalized_coords: bool,
}

/// A raw 32-byte texture header.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TicEntry(pub [u32; 8]);

const fn bits(word: u32, lo: u32, len: u32) -> u32 {
    (word >> lo) & ((1 << len) - 1)
}

const HEADER_PITCH_COLOR_KEY: u32 = 1;
const HEADER_PITCH: u32 = 2;
const HEADER_BLOCK_LINEAR: u32 = 3;
const HEADER_BLOCK_LINEAR_COLOR_KEY: u32 = 4;
const HEADER_ONE_D_BUFFER: u32 = 0;

impl TicEntry {
    fn header_version(&self) -> u32 {
        bits(self.0[2], 21, 3)
    }

    /// Decodes the header. Component types that force fp16 read as their normalized types.
    pub fn decode(&self) -> Result<TextureDescriptor, TicError> {
        let w = &self.0;
        let raw_format = bits(w[0], 0, 7);
        let raw_components = [
            bits(w[0], 7, 3),
            bits(w[0], 10, 3),
            bits(w[0], 13, 3),
            bits(w[0], 16, 3),
        ];
        let srgb = bits(w[4], 22, 1) != 0;
        let format_error = TicError::Format {
            format: raw_format,
            components: raw_components,
        };
        let texture_format = TextureFormat::try_from(raw_format).map_err(|_| format_error)?;
        let mut components = [ComponentType::Unorm; 4];
        for (c, raw) in components.iter_mut().zip(raw_components) {
            *c = ComponentType::try_from(raw).map_err(|_| format_error)?;
        }
        let format =
            PixelFormat::from_texture(texture_format, components, srgb).ok_or(format_error)?;
        let mut swizzle = [SwizzleSource::Zero; 4];
        for (i, s) in swizzle.iter_mut().enumerate() {
            let raw = bits(w[0], 19 + 3 * i as u32, 3);
            *s = SwizzleSource::from_raw(raw).ok_or(TicError::Swizzle(raw))?;
        }
        let version = self.header_version();
        let memory = match version {
            HEADER_BLOCK_LINEAR | HEADER_BLOCK_LINEAR_COLOR_KEY => TextureMemory::BlockLinear {
                block_height_log2: bits(w[3], 3, 3),
                block_depth_log2: bits(w[3], 6, 3),
                tile_width_spacing: bits(w[3], 10, 3),
            },
            HEADER_PITCH | HEADER_PITCH_COLOR_KEY => TextureMemory::Pitch {
                pitch: bits(w[3], 0, 16) << 5,
            },
            HEADER_ONE_D_BUFFER => TextureMemory::Buffer,
            other => return Err(TicError::HeaderVersion(other)),
        };
        let raw_type = bits(w[4], 23, 4);
        let texture_type = TextureType::from_raw(raw_type).ok_or(TicError::Type(raw_type))?;
        let raw_msaa = bits(w[7], 8, 4);
        let msaa = MsaaMode::from_raw(raw_msaa).ok_or(TicError::Msaa(raw_msaa))?;
        let width = if memory == TextureMemory::Buffer {
            ((bits(w[3], 0, 16) << 16) | bits(w[4], 0, 16)) + 1
        } else {
            bits(w[4], 0, 16) + 1
        };
        let max_mip_level = if memory == TextureMemory::Buffer {
            0
        } else {
            bits(w[3], 28, 4)
        };
        Ok(TextureDescriptor {
            address: (u64::from(bits(w[2], 0, 16)) << 32) | u64::from(w[1]),
            format,
            swizzle,
            texture_type,
            memory,
            width,
            height: bits(w[5], 0, 16) + 1,
            depth: bits(w[5], 16, 14) + 1,
            levels: max_mip_level + 1,
            base_layer: bits(w[4], 16, 3) | bits(w[2], 16, 5) << 3 | bits(w[2], 29, 3) << 8,
            min_level: bits(w[7], 0, 4),
            max_level: bits(w[7], 4, 4),
            msaa,
            srgb,
            sparse: bits(w[5], 30, 1) != 0,
            depth_texture: bits(w[3], 27, 1) != 0,
            normalized_coords: bits(w[5], 31, 1) != 0,
        })
    }
}

/// The header index a texture handle names. Some Switch 2 bindless handles hold the header's
/// byte offset in the pool (index * 32) instead of its index; they are read as offsets only on
/// the Switch 2, when the plain index falls past the pool and the offset lands inside it.
pub const fn texture_header_index(index: u32, limit: u32, switch2: bool) -> u32 {
    if switch2 && index > limit && index.is_multiple_of(32) && index / 32 <= limit {
        index / 32
    } else {
        index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Packs a header from its fields, the inverse of `TicEntry::decode`.
    #[allow(clippy::too_many_arguments)]
    fn header(
        format: u32,
        components: [u32; 4],
        swizzle: [u32; 4],
        address: u64,
        version: u32,
        word3: u32,
        width_minus_one: u32,
        texture_type: u32,
        height_minus_one: u32,
        depth_minus_one: u32,
    ) -> TicEntry {
        let mut w = [0u32; 8];
        w[0] = format
            | components[0] << 7
            | components[1] << 10
            | components[2] << 13
            | components[3] << 16
            | swizzle[0] << 19
            | swizzle[1] << 22
            | swizzle[2] << 25
            | swizzle[3] << 28;
        w[1] = address as u32;
        w[2] = (address >> 32) as u32 | version << 21;
        w[3] = word3;
        w[4] = width_minus_one | texture_type << 23;
        w[5] = height_minus_one | depth_minus_one << 16;
        TicEntry(w)
    }

    #[test]
    fn decodes_a_block_linear_mipmapped_array() {
        let mut tic = header(
            0x08,
            [2, 2, 2, 2],
            [4, 3, 2, 7],
            0x12_3456_7800,
            3,
            4 << 3 | 1 << 10 | 7 << 28,
            255,
            5,
            127,
            5,
        );
        tic.0[4] |= 1 << 22 | 0b101 << 16;
        tic.0[2] |= 0b10 << 16 | 0b1 << 29;
        tic.0[7] = 2 | 6 << 4 | 2 << 8;
        let d = tic.decode().unwrap();
        assert_eq!(d.address, 0x12_3456_7800);
        assert_eq!(d.format, PixelFormat::A8B8G8R8Srgb);
        assert!(d.srgb);
        assert_eq!(
            d.swizzle,
            [
                SwizzleSource::B,
                SwizzleSource::G,
                SwizzleSource::R,
                SwizzleSource::OneFloat
            ]
        );
        assert_eq!(d.texture_type, TextureType::D2Array);
        assert_eq!(
            d.memory,
            TextureMemory::BlockLinear {
                block_height_log2: 4,
                block_depth_log2: 0,
                tile_width_spacing: 1
            }
        );
        assert_eq!((d.width, d.height, d.depth, d.levels), (256, 128, 6, 8));
        assert_eq!(d.base_layer, 1 << 8 | 0b10 << 3 | 0b101);
        assert_eq!((d.min_level, d.max_level), (2, 6));
        assert_eq!(d.msaa, MsaaMode::X4);
        assert_eq!(d.msaa.grid(), (2, 2));
    }

    #[test]
    fn decodes_pitch_buffer_and_compressed_headers() {
        let pitch = header(0x1d, [2; 4], [2, 2, 2, 2], 0x4000, 2, 0x50, 99, 1, 49, 0)
            .decode()
            .unwrap();
        assert_eq!(pitch.format, PixelFormat::R8Unorm);
        assert_eq!(pitch.memory, TextureMemory::Pitch { pitch: 0x50 << 5 });
        assert_eq!((pitch.width, pitch.height, pitch.levels), (100, 50, 1));

        let buffer = header(0x08, [2; 4], [2, 3, 4, 5], 0x8000, 0, 0x3, 0xffff, 6, 0, 0)
            .decode()
            .unwrap();
        assert_eq!(buffer.memory, TextureMemory::Buffer);
        assert_eq!(buffer.width, 0x4_0000);
        assert_eq!(buffer.levels, 1);

        let astc = header(0x44, [2; 4], [2, 3, 4, 5], 0, 3, 3 << 28, 1023, 2, 1023, 31)
            .decode()
            .unwrap();
        assert_eq!(astc.format, PixelFormat::Astc8x8Unorm);
        assert_eq!(astc.texture_type, TextureType::D3);
        assert_eq!(astc.depth, 32);
        assert_eq!(astc.levels, 4);
    }

    #[test]
    fn rejects_unknown_fields() {
        let bad_format = header(0x43, [2; 4], [2, 3, 4, 5], 0, 3, 0, 0, 1, 0, 0);
        assert!(matches!(
            bad_format.decode(),
            Err(TicError::Format { format: 0x43, .. })
        ));
        let bad_type = header(0x08, [2; 4], [2, 3, 4, 5], 0, 3, 0, 0, 9, 0, 0);
        assert_eq!(bad_type.decode(), Err(TicError::Type(9)));
        let bad_swizzle = header(0x08, [2; 4], [1, 3, 4, 5], 0, 3, 0, 0, 1, 0, 0);
        assert_eq!(bad_swizzle.decode(), Err(TicError::Swizzle(1)));
        let bad_version = header(0x08, [2; 4], [2, 3, 4, 5], 0, 5, 0, 0, 1, 0, 0);
        assert_eq!(bad_version.decode(), Err(TicError::HeaderVersion(5)));
    }

    #[test]
    fn switch2_handles_may_hold_byte_offsets() {
        assert_eq!(texture_header_index(64, 10, true), 2);
        assert_eq!(texture_header_index(64, 10, false), 64);
        assert_eq!(texture_header_index(5, 10, true), 5);
        assert_eq!(texture_header_index(66, 10, true), 66);
    }
}
