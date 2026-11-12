//! Per-instruction flags stored in `Inst::flags`.

/// Denormal handling requested by a floating-point instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FmzMode {
    #[default]
    DontCare,
    /// Flush denormals to zero, NaN propagates.
    Ftz,
    /// Flush denormals to zero and x * 0 == 0 (D3D9).
    Fmz,
    /// Keep denormals.
    None,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FpRounding {
    #[default]
    DontCare,
    Rn,
    Rm,
    Rp,
    Rz,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct FpControl {
    pub no_contraction: bool,
    pub rounding: FpRounding,
    pub fmz_mode: FmzMode,
}

impl FpControl {
    pub const fn to_flags(self) -> u32 {
        self.no_contraction as u32 | (self.rounding as u32) << 1 | (self.fmz_mode as u32) << 4
    }

    pub const fn from_flags(flags: u32) -> FpControl {
        FpControl {
            no_contraction: flags & 1 != 0,
            rounding: match (flags >> 1) & 7 {
                1 => FpRounding::Rn,
                2 => FpRounding::Rm,
                3 => FpRounding::Rp,
                4 => FpRounding::Rz,
                _ => FpRounding::DontCare,
            },
            fmz_mode: match (flags >> 4) & 3 {
                1 => FmzMode::Ftz,
                2 => FmzMode::Fmz,
                3 => FmzMode::None,
                _ => FmzMode::DontCare,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum TextureType {
    Color1D,
    ColorArray1D,
    #[default]
    Color2D,
    ColorArray2D,
    Color3D,
    ColorCube,
    ColorArrayCube,
    Buffer,
    Color2DRect,
}

impl TextureType {
    const ALL: [TextureType; 9] = [
        TextureType::Color1D,
        TextureType::ColorArray1D,
        TextureType::Color2D,
        TextureType::ColorArray2D,
        TextureType::Color3D,
        TextureType::ColorCube,
        TextureType::ColorArrayCube,
        TextureType::Buffer,
        TextureType::Color2DRect,
    ];

    pub const fn from_bits(bits: u32) -> TextureType {
        if (bits as usize) < Self::ALL.len() {
            Self::ALL[bits as usize]
        } else {
            TextureType::Color2D
        }
    }
}

/// Storage image format for typed loads and stores; `Typeless` needs formatted access support.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ImageFormat {
    #[default]
    Typeless,
    R8Uint,
    R8Sint,
    R16Uint,
    R16Sint,
    R32Uint,
    R32G32Uint,
    R32G32B32A32Uint,
}

impl ImageFormat {
    pub const fn from_bits(bits: u32) -> ImageFormat {
        match bits {
            1 => ImageFormat::R8Uint,
            2 => ImageFormat::R8Sint,
            3 => ImageFormat::R16Uint,
            4 => ImageFormat::R16Sint,
            5 => ImageFormat::R32Uint,
            6 => ImageFormat::R32G32Uint,
            7 => ImageFormat::R32G32B32A32Uint,
            _ => ImageFormat::Typeless,
        }
    }
}

/// Flags of texture and image instructions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextureInstInfo {
    /// Index into the program's texture or image descriptors, set by the texture pass.
    pub descriptor_index: u16,
    pub ty: TextureType,
    pub is_depth: bool,
    pub has_bias: bool,
    pub has_lod_clamp: bool,
    pub relaxed_precision: bool,
    pub gather_component: u8,
    pub num_derivatives: u8,
    pub image_format: ImageFormat,
    pub ndv_is_active: bool,
    pub is_integer: bool,
}

impl TextureInstInfo {
    /// Largest descriptor index the flags word can hold.
    pub const MAX_DESCRIPTOR_INDEX: u16 = (1 << 14) - 1;

    pub const fn to_flags(self) -> u32 {
        (self.descriptor_index as u32 & 0x3fff)
            | (self.ty as u32) << 14
            | (self.is_depth as u32) << 18
            | (self.has_bias as u32) << 19
            | (self.has_lod_clamp as u32) << 20
            | (self.relaxed_precision as u32) << 21
            | ((self.gather_component as u32) & 3) << 22
            | ((self.num_derivatives as u32) & 3) << 24
            | (self.image_format as u32) << 26
            | (self.ndv_is_active as u32) << 29
            | (self.is_integer as u32) << 30
    }

    pub const fn from_flags(flags: u32) -> TextureInstInfo {
        TextureInstInfo {
            descriptor_index: (flags & 0x3fff) as u16,
            ty: TextureType::from_bits((flags >> 14) & 0xf),
            is_depth: flags & (1 << 18) != 0,
            has_bias: flags & (1 << 19) != 0,
            has_lod_clamp: flags & (1 << 20) != 0,
            relaxed_precision: flags & (1 << 21) != 0,
            gather_component: ((flags >> 22) & 3) as u8,
            num_derivatives: ((flags >> 24) & 3) as u8,
            image_format: ImageFormat::from_bits((flags >> 26) & 7),
            ndv_is_active: flags & (1 << 29) != 0,
            is_integer: flags & (1 << 30) != 0,
        }
    }
}
