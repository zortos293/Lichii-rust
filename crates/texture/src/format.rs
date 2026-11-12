use crate::guest::{ComponentType, DepthFormat, RenderTargetFormat, TextureFormat};

/// Which parts of an image a format holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Aspect {
    Color,
    Depth,
    Stencil,
    DepthStencil,
}

/// How shaders read the format's components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Numeric {
    Unorm,
    Snorm,
    Uint,
    Sint,
    Float,
}

/// Block compression family of a compressed format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Compression {
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    Bc6h,
    Bc7,
    Etc2,
    Eac,
    Astc,
}

/// Static facts about a [`PixelFormat`]. Uncompressed formats have 1x1 blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FormatInfo {
    pub block_width: u32,
    pub block_height: u32,
    pub bytes_per_block: u32,
    pub aspect: Aspect,
    pub numeric: Numeric,
    pub srgb: bool,
    pub compression: Option<Compression>,
}

macro_rules! pixel_formats {
    ($($name:ident: $bw:literal x $bh:literal, $bytes:literal, $aspect:ident, $numeric:ident, $srgb:literal, $compression:expr;)*) => {
        /// Host-independent surface format, after the GPU decoded the guest's format fields.
        ///
        /// Names list components from the most significant bit down, as the guest does
        /// (`A8B8G8R8` keeps R in the lowest byte).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
        #[repr(u8)]
        pub enum PixelFormat {
            $($name,)*
        }

        impl PixelFormat {
            /// Every format, in declaration order (`ALL[f as usize] == f`).
            pub const ALL: &[PixelFormat] = &[$(PixelFormat::$name,)*];

            const INFO: &[FormatInfo] = &[$(FormatInfo {
                block_width: $bw,
                block_height: $bh,
                bytes_per_block: $bytes,
                aspect: Aspect::$aspect,
                numeric: Numeric::$numeric,
                srgb: $srgb,
                compression: $compression,
            },)*];
        }
    };
}

use Compression::*;

pixel_formats! {
    A8B8G8R8Unorm: 1 x 1, 4, Color, Unorm, false, None;
    A8B8G8R8Snorm: 1 x 1, 4, Color, Snorm, false, None;
    A8B8G8R8Sint: 1 x 1, 4, Color, Sint, false, None;
    A8B8G8R8Uint: 1 x 1, 4, Color, Uint, false, None;
    R5G6B5Unorm: 1 x 1, 2, Color, Unorm, false, None;
    B5G6R5Unorm: 1 x 1, 2, Color, Unorm, false, None;
    A1R5G5B5Unorm: 1 x 1, 2, Color, Unorm, false, None;
    A2B10G10R10Unorm: 1 x 1, 4, Color, Unorm, false, None;
    A2B10G10R10Uint: 1 x 1, 4, Color, Uint, false, None;
    A2R10G10B10Unorm: 1 x 1, 4, Color, Unorm, false, None;
    A1B5G5R5Unorm: 1 x 1, 2, Color, Unorm, false, None;
    A5B5G5R1Unorm: 1 x 1, 2, Color, Unorm, false, None;
    R8Unorm: 1 x 1, 1, Color, Unorm, false, None;
    R8Snorm: 1 x 1, 1, Color, Snorm, false, None;
    R8Sint: 1 x 1, 1, Color, Sint, false, None;
    R8Uint: 1 x 1, 1, Color, Uint, false, None;
    R16G16B16A16Float: 1 x 1, 8, Color, Float, false, None;
    R16G16B16A16Unorm: 1 x 1, 8, Color, Unorm, false, None;
    R16G16B16A16Snorm: 1 x 1, 8, Color, Snorm, false, None;
    R16G16B16A16Sint: 1 x 1, 8, Color, Sint, false, None;
    R16G16B16A16Uint: 1 x 1, 8, Color, Uint, false, None;
    B10G11R11Float: 1 x 1, 4, Color, Float, false, None;
    R32G32B32A32Uint: 1 x 1, 16, Color, Uint, false, None;
    Bc1RgbaUnorm: 4 x 4, 8, Color, Unorm, false, Some(Bc1);
    Bc2Unorm: 4 x 4, 16, Color, Unorm, false, Some(Bc2);
    Bc3Unorm: 4 x 4, 16, Color, Unorm, false, Some(Bc3);
    Bc4Unorm: 4 x 4, 8, Color, Unorm, false, Some(Bc4);
    Bc4Snorm: 4 x 4, 8, Color, Snorm, false, Some(Bc4);
    Bc5Unorm: 4 x 4, 16, Color, Unorm, false, Some(Bc5);
    Bc5Snorm: 4 x 4, 16, Color, Snorm, false, Some(Bc5);
    Bc7Unorm: 4 x 4, 16, Color, Unorm, false, Some(Bc7);
    Bc6hUfloat: 4 x 4, 16, Color, Float, false, Some(Bc6h);
    Bc6hSfloat: 4 x 4, 16, Color, Float, false, Some(Bc6h);
    Astc4x4Unorm: 4 x 4, 16, Color, Unorm, false, Some(Astc);
    B8G8R8A8Unorm: 1 x 1, 4, Color, Unorm, false, None;
    R32G32B32A32Float: 1 x 1, 16, Color, Float, false, None;
    R32G32B32A32Sint: 1 x 1, 16, Color, Sint, false, None;
    R32G32Float: 1 x 1, 8, Color, Float, false, None;
    R32G32Sint: 1 x 1, 8, Color, Sint, false, None;
    R32Float: 1 x 1, 4, Color, Float, false, None;
    R16Float: 1 x 1, 2, Color, Float, false, None;
    R16Unorm: 1 x 1, 2, Color, Unorm, false, None;
    R16Snorm: 1 x 1, 2, Color, Snorm, false, None;
    R16Uint: 1 x 1, 2, Color, Uint, false, None;
    R16Sint: 1 x 1, 2, Color, Sint, false, None;
    R16G16Unorm: 1 x 1, 4, Color, Unorm, false, None;
    R16G16Float: 1 x 1, 4, Color, Float, false, None;
    R16G16Uint: 1 x 1, 4, Color, Uint, false, None;
    R16G16Sint: 1 x 1, 4, Color, Sint, false, None;
    R16G16Snorm: 1 x 1, 4, Color, Snorm, false, None;
    R32G32B32Float: 1 x 1, 12, Color, Float, false, None;
    A8B8G8R8Srgb: 1 x 1, 4, Color, Unorm, true, None;
    R8G8Unorm: 1 x 1, 2, Color, Unorm, false, None;
    R8G8Snorm: 1 x 1, 2, Color, Snorm, false, None;
    R8G8Sint: 1 x 1, 2, Color, Sint, false, None;
    R8G8Uint: 1 x 1, 2, Color, Uint, false, None;
    R32G32Uint: 1 x 1, 8, Color, Uint, false, None;
    R16G16B16X16Float: 1 x 1, 8, Color, Float, false, None;
    R32Uint: 1 x 1, 4, Color, Uint, false, None;
    R32Sint: 1 x 1, 4, Color, Sint, false, None;
    Astc8x8Unorm: 8 x 8, 16, Color, Unorm, false, Some(Astc);
    Astc8x5Unorm: 8 x 5, 16, Color, Unorm, false, Some(Astc);
    Astc5x4Unorm: 5 x 4, 16, Color, Unorm, false, Some(Astc);
    B8G8R8A8Srgb: 1 x 1, 4, Color, Unorm, true, None;
    Bc1RgbaSrgb: 4 x 4, 8, Color, Unorm, true, Some(Bc1);
    Bc2Srgb: 4 x 4, 16, Color, Unorm, true, Some(Bc2);
    Bc3Srgb: 4 x 4, 16, Color, Unorm, true, Some(Bc3);
    Bc7Srgb: 4 x 4, 16, Color, Unorm, true, Some(Bc7);
    A4B4G4R4Unorm: 1 x 1, 2, Color, Unorm, false, None;
    G4R4Unorm: 1 x 1, 1, Color, Unorm, false, None;
    Astc4x4Srgb: 4 x 4, 16, Color, Unorm, true, Some(Astc);
    Astc8x8Srgb: 8 x 8, 16, Color, Unorm, true, Some(Astc);
    Astc8x5Srgb: 8 x 5, 16, Color, Unorm, true, Some(Astc);
    Astc5x4Srgb: 5 x 4, 16, Color, Unorm, true, Some(Astc);
    Astc5x5Unorm: 5 x 5, 16, Color, Unorm, false, Some(Astc);
    Astc5x5Srgb: 5 x 5, 16, Color, Unorm, true, Some(Astc);
    Astc10x8Unorm: 10 x 8, 16, Color, Unorm, false, Some(Astc);
    Astc10x8Srgb: 10 x 8, 16, Color, Unorm, true, Some(Astc);
    Astc6x6Unorm: 6 x 6, 16, Color, Unorm, false, Some(Astc);
    Astc6x6Srgb: 6 x 6, 16, Color, Unorm, true, Some(Astc);
    Astc10x6Unorm: 10 x 6, 16, Color, Unorm, false, Some(Astc);
    Astc10x6Srgb: 10 x 6, 16, Color, Unorm, true, Some(Astc);
    Astc10x5Unorm: 10 x 5, 16, Color, Unorm, false, Some(Astc);
    Astc10x5Srgb: 10 x 5, 16, Color, Unorm, true, Some(Astc);
    Astc10x10Unorm: 10 x 10, 16, Color, Unorm, false, Some(Astc);
    Astc10x10Srgb: 10 x 10, 16, Color, Unorm, true, Some(Astc);
    Astc12x10Unorm: 12 x 10, 16, Color, Unorm, false, Some(Astc);
    Astc12x10Srgb: 12 x 10, 16, Color, Unorm, true, Some(Astc);
    Astc12x12Unorm: 12 x 12, 16, Color, Unorm, false, Some(Astc);
    Astc12x12Srgb: 12 x 12, 16, Color, Unorm, true, Some(Astc);
    Astc8x6Unorm: 8 x 6, 16, Color, Unorm, false, Some(Astc);
    Astc8x6Srgb: 8 x 6, 16, Color, Unorm, true, Some(Astc);
    Astc6x5Unorm: 6 x 5, 16, Color, Unorm, false, Some(Astc);
    Astc6x5Srgb: 6 x 5, 16, Color, Unorm, true, Some(Astc);
    E5B9G9R9Float: 1 x 1, 4, Color, Float, false, None;
    Etc2RgbUnorm: 4 x 4, 8, Color, Unorm, false, Some(Etc2);
    Etc2RgbaUnorm: 4 x 4, 16, Color, Unorm, false, Some(Etc2);
    Etc2RgbPtaUnorm: 4 x 4, 8, Color, Unorm, false, Some(Etc2);
    Etc2RgbSrgb: 4 x 4, 8, Color, Unorm, true, Some(Etc2);
    Etc2RgbaSrgb: 4 x 4, 16, Color, Unorm, true, Some(Etc2);
    Etc2RgbPtaSrgb: 4 x 4, 8, Color, Unorm, true, Some(Etc2);
    EacR11Unorm: 4 x 4, 8, Color, Unorm, false, Some(Eac);
    EacR11Snorm: 4 x 4, 8, Color, Snorm, false, Some(Eac);
    EacR11G11Unorm: 4 x 4, 16, Color, Unorm, false, Some(Eac);
    EacR11G11Snorm: 4 x 4, 16, Color, Snorm, false, Some(Eac);
    D32Float: 1 x 1, 4, Depth, Float, false, None;
    D16Unorm: 1 x 1, 2, Depth, Unorm, false, None;
    X8D24Unorm: 1 x 1, 4, Depth, Unorm, false, None;
    S8Uint: 1 x 1, 1, Stencil, Uint, false, None;
    D24UnormS8Uint: 1 x 1, 4, DepthStencil, Unorm, false, None;
    S8UintD24Unorm: 1 x 1, 4, DepthStencil, Unorm, false, None;
    D32FloatS8Uint: 1 x 1, 8, DepthStencil, Float, false, None;
}

impl PixelFormat {
    /// Every static fact about the format, from one table lookup.
    pub const fn info(self) -> &'static FormatInfo {
        &Self::INFO[self as usize]
    }

    /// Texels per block horizontally (1 for uncompressed formats).
    pub const fn block_width(self) -> u32 {
        self.info().block_width
    }

    /// Texels per block vertically (1 for uncompressed formats).
    pub const fn block_height(self) -> u32 {
        self.info().block_height
    }

    /// Bytes per block, which is bytes per texel for uncompressed formats.
    pub const fn bytes_per_block(self) -> u32 {
        self.info().bytes_per_block
    }

    /// Whether the format holds color, depth, stencil or both depth and stencil.
    pub const fn aspect(self) -> Aspect {
        self.info().aspect
    }

    /// How shaders read the components. For combined depth/stencil formats this is the depth
    /// component's type.
    pub const fn numeric(self) -> Numeric {
        self.info().numeric
    }

    /// Whether reads convert from sRGB to linear.
    pub const fn is_srgb(self) -> bool {
        self.info().srgb
    }

    /// The block compression family, `None` for uncompressed formats.
    pub const fn compression(self) -> Option<Compression> {
        self.info().compression
    }

    /// Whether the format is block compressed (BCn, ETC2/EAC or ASTC).
    pub const fn is_compressed(self) -> bool {
        self.info().compression.is_some()
    }

    /// Whether the format is one of the ASTC block sizes.
    pub const fn is_astc(self) -> bool {
        matches!(self.info().compression, Some(Compression::Astc))
    }

    /// Whether the format is a color format read as unnormalized integers (`Uint`/`Sint`).
    /// Stencil formats are excluded even though their numeric type is `Uint`.
    pub const fn is_integer(self) -> bool {
        matches!(self.info().numeric, Numeric::Uint | Numeric::Sint) && self.is_color()
    }

    /// Whether the format holds color rather than depth and/or stencil.
    pub const fn is_color(self) -> bool {
        matches!(self.info().aspect, Aspect::Color)
    }

    /// Bytes of one tightly packed `width` x `height` x `depth` image, rounding partial
    /// blocks up.
    pub const fn size_bytes(self, width: u32, height: u32, depth: u32) -> u64 {
        let info = self.info();
        width.div_ceil(info.block_width) as u64
            * height.div_ceil(info.block_height) as u64
            * depth as u64
            * info.bytes_per_block as u64
    }

    /// The sRGB variant of a linear format with the same layout, if one exists.
    pub const fn to_srgb(self) -> Option<Self> {
        use PixelFormat::*;
        Some(match self {
            A8B8G8R8Unorm => A8B8G8R8Srgb,
            B8G8R8A8Unorm => B8G8R8A8Srgb,
            Bc1RgbaUnorm => Bc1RgbaSrgb,
            Bc2Unorm => Bc2Srgb,
            Bc3Unorm => Bc3Srgb,
            Bc7Unorm => Bc7Srgb,
            Etc2RgbUnorm => Etc2RgbSrgb,
            Etc2RgbaUnorm => Etc2RgbaSrgb,
            Etc2RgbPtaUnorm => Etc2RgbPtaSrgb,
            Astc4x4Unorm => Astc4x4Srgb,
            Astc5x4Unorm => Astc5x4Srgb,
            Astc5x5Unorm => Astc5x5Srgb,
            Astc6x5Unorm => Astc6x5Srgb,
            Astc6x6Unorm => Astc6x6Srgb,
            Astc8x5Unorm => Astc8x5Srgb,
            Astc8x6Unorm => Astc8x6Srgb,
            Astc8x8Unorm => Astc8x8Srgb,
            Astc10x5Unorm => Astc10x5Srgb,
            Astc10x6Unorm => Astc10x6Srgb,
            Astc10x8Unorm => Astc10x8Srgb,
            Astc10x10Unorm => Astc10x10Srgb,
            Astc12x10Unorm => Astc12x10Srgb,
            Astc12x12Unorm => Astc12x12Srgb,
            _ if self.is_srgb() => self,
            _ => return None,
        })
    }

    /// The linear variant of an sRGB format; linear formats return themselves.
    pub const fn to_linear(self) -> Self {
        use PixelFormat::*;
        match self {
            A8B8G8R8Srgb => A8B8G8R8Unorm,
            B8G8R8A8Srgb => B8G8R8A8Unorm,
            Bc1RgbaSrgb => Bc1RgbaUnorm,
            Bc2Srgb => Bc2Unorm,
            Bc3Srgb => Bc3Unorm,
            Bc7Srgb => Bc7Unorm,
            Etc2RgbSrgb => Etc2RgbUnorm,
            Etc2RgbaSrgb => Etc2RgbaUnorm,
            Etc2RgbPtaSrgb => Etc2RgbPtaUnorm,
            Astc4x4Srgb => Astc4x4Unorm,
            Astc5x4Srgb => Astc5x4Unorm,
            Astc5x5Srgb => Astc5x5Unorm,
            Astc6x5Srgb => Astc6x5Unorm,
            Astc6x6Srgb => Astc6x6Unorm,
            Astc8x5Srgb => Astc8x5Unorm,
            Astc8x6Srgb => Astc8x6Unorm,
            Astc8x8Srgb => Astc8x8Unorm,
            Astc10x5Srgb => Astc10x5Unorm,
            Astc10x6Srgb => Astc10x6Unorm,
            Astc10x8Srgb => Astc10x8Unorm,
            Astc10x10Srgb => Astc10x10Unorm,
            Astc12x10Srgb => Astc12x10Unorm,
            Astc12x12Srgb => Astc12x12Unorm,
            _ => self,
        }
    }

    /// Decodes a texture header's format fields. Component types that force fp16 read as their
    /// normalized types, and an sRGB flag on a format without an sRGB variant is ignored, as
    /// the hardware does. `None` for combinations no title is known to use.
    pub fn from_texture(
        format: TextureFormat,
        components: [ComponentType; 4],
        srgb: bool,
    ) -> Option<Self> {
        use ComponentType::{Float as F, Sint as SI, Snorm as SN, Uint as UI, Unorm as UN};
        use PixelFormat::*;
        use TextureFormat as T;
        let components = components.map(ComponentType::without_fp16);
        let [r, g, b, a] = components;
        let uniform = (r == g && g == b && b == a).then_some(r);
        let linear = match (format, uniform) {
            (T::A8B8G8R8, Some(UN)) => A8B8G8R8Unorm,
            (T::A8B8G8R8, Some(SN)) => A8B8G8R8Snorm,
            (T::A8B8G8R8, Some(UI)) => A8B8G8R8Uint,
            (T::A8B8G8R8, Some(SI)) => A8B8G8R8Sint,
            (T::B5G6R5, Some(UN)) => B5G6R5Unorm,
            (T::A2B10G10R10, Some(UN)) => A2B10G10R10Unorm,
            (T::A2B10G10R10, Some(UI)) => A2B10G10R10Uint,
            (T::A1B5G5R5, Some(UN)) => A1B5G5R5Unorm,
            (T::A4B4G4R4, Some(UN)) => A4B4G4R4Unorm,
            (T::G4R4, Some(UN)) => G4R4Unorm,
            (T::A5B5G5R1, Some(UN)) => A5B5G5R1Unorm,
            (T::R8, Some(UN)) => R8Unorm,
            (T::R8, Some(SN)) => R8Snorm,
            (T::R8, Some(UI)) => R8Uint,
            (T::R8, Some(SI)) => R8Sint,
            (T::G8R8, Some(UN)) => R8G8Unorm,
            (T::G8R8, Some(SN)) => R8G8Snorm,
            (T::G8R8, Some(UI)) => R8G8Uint,
            (T::G8R8, Some(SI)) => R8G8Sint,
            (T::R16G16B16A16, Some(F)) => R16G16B16A16Float,
            (T::R16G16B16A16, Some(UN)) => R16G16B16A16Unorm,
            (T::R16G16B16A16, Some(SN)) => R16G16B16A16Snorm,
            (T::R16G16B16A16, Some(UI)) => R16G16B16A16Uint,
            (T::R16G16B16A16, Some(SI)) => R16G16B16A16Sint,
            (T::R16G16, Some(F)) => R16G16Float,
            (T::R16G16, Some(UN)) => R16G16Unorm,
            (T::R16G16, Some(SN)) => R16G16Snorm,
            (T::R16G16, Some(UI)) => R16G16Uint,
            (T::R16G16, Some(SI)) => R16G16Sint,
            (T::R16, Some(F)) => R16Float,
            (T::R16, Some(UN)) => R16Unorm,
            (T::R16, Some(SN)) => R16Snorm,
            (T::R16, Some(UI)) => R16Uint,
            (T::R16, Some(SI)) => R16Sint,
            (T::B10G11R11, Some(F)) => B10G11R11Float,
            (T::R32G32B32A32, Some(F)) => R32G32B32A32Float,
            (T::R32G32B32A32, Some(UI)) => R32G32B32A32Uint,
            (T::R32G32B32A32, Some(SI)) => R32G32B32A32Sint,
            (T::R32G32B32, Some(F)) => R32G32B32Float,
            (T::R32G32, Some(F)) => R32G32Float,
            (T::R32G32, Some(UI)) => R32G32Uint,
            (T::R32G32, Some(SI)) => R32G32Sint,
            (T::R32, Some(F)) => R32Float,
            (T::R32, Some(UI)) => R32Uint,
            (T::R32, Some(SI)) => R32Sint,
            (T::E5B9G9R9, Some(F)) => E5B9G9R9Float,
            (T::Dxt1, Some(UN)) => Bc1RgbaUnorm,
            (T::Dxt23, Some(UN)) => Bc2Unorm,
            (T::Dxt45, Some(UN)) => Bc3Unorm,
            (T::Dxn1, Some(UN)) => Bc4Unorm,
            (T::Dxn1, Some(SN)) => Bc4Snorm,
            (T::Dxn2, Some(UN)) => Bc5Unorm,
            (T::Dxn2, Some(SN)) => Bc5Snorm,
            (T::Bc7U, Some(UN)) => Bc7Unorm,
            (T::Bc6hS16, Some(F)) => Bc6hSfloat,
            (T::Bc6hU16, Some(F)) => Bc6hUfloat,
            (T::Etc2Rgb, Some(UN)) => Etc2RgbUnorm,
            (T::Etc2RgbPta, Some(UN)) => Etc2RgbPtaUnorm,
            (T::Etc2Rgba, Some(UN)) => Etc2RgbaUnorm,
            (T::Eac, Some(UN)) => EacR11Unorm,
            (T::Eac, Some(SN)) => EacR11Snorm,
            (T::EacX2, Some(UN)) => EacR11G11Unorm,
            (T::EacX2, Some(SN)) => EacR11G11Snorm,
            (T::Astc2d4x4, Some(UN)) => Astc4x4Unorm,
            (T::Astc2d5x4, Some(UN)) => Astc5x4Unorm,
            (T::Astc2d5x5, Some(UN)) => Astc5x5Unorm,
            (T::Astc2d6x5, Some(UN)) => Astc6x5Unorm,
            (T::Astc2d6x6, Some(UN)) => Astc6x6Unorm,
            (T::Astc2d8x5, Some(UN)) => Astc8x5Unorm,
            (T::Astc2d8x6, Some(UN)) => Astc8x6Unorm,
            (T::Astc2d8x8, Some(UN)) => Astc8x8Unorm,
            (T::Astc2d10x5, Some(UN)) => Astc10x5Unorm,
            (T::Astc2d10x6, Some(UN)) => Astc10x6Unorm,
            (T::Astc2d10x8, Some(UN)) => Astc10x8Unorm,
            (T::Astc2d10x10, Some(UN)) => Astc10x10Unorm,
            (T::Astc2d12x10, Some(UN)) => Astc12x10Unorm,
            (T::Astc2d12x12, Some(UN)) => Astc12x12Unorm,
            _ => return Self::depth_from_texture(format, components),
        };
        Some(if srgb {
            linear.to_srgb().unwrap_or(linear)
        } else {
            linear
        })
    }

    fn depth_from_texture(format: TextureFormat, components: [ComponentType; 4]) -> Option<Self> {
        use ComponentType::{Float as F, Uint as UI, Unorm as UN};
        use PixelFormat::*;
        use TextureFormat as T;
        Some(match (format, components) {
            (T::Z32, [F, F, F, F] | [F, UI, UI, UI]) => D32Float,
            (T::Z16, [UN, UN, UN, UN] | [UN, UI, UI, UI]) => D16Unorm,
            (T::X8Z24, [UN, UN, UN, UN] | [UN, UI, UI, UI] | [UN, UI, UN, UN]) => X8D24Unorm,
            (T::Z24S8 | T::G24R8, [UI, UN, UN, UN]) | (T::Z24S8, [UI, UN, UI, UI]) => {
                S8UintD24Unorm
            }
            (T::S8Z24 | T::G8R24, [UN, UI, UN, UN] | [UN, UI, UI, UI]) => D24UnormS8Uint,
            (T::Z32X24S8 | T::R32B24G8, [F, UI, UN, UN]) => D32FloatS8Uint,
            _ => return None,
        })
    }

    /// Decodes a color render target's format register. `None` for `NONE` and formats the
    /// renderer cannot back.
    pub const fn from_render_target(format: RenderTargetFormat) -> Option<Self> {
        use PixelFormat::*;
        use RenderTargetFormat as R;
        Some(match format {
            R::R32G32B32A32Float | R::R32G32B32X32Float => R32G32B32A32Float,
            R::R32G32B32A32Sint | R::R32G32B32X32Sint => R32G32B32A32Sint,
            R::R32G32B32A32Uint | R::R32G32B32X32Uint => R32G32B32A32Uint,
            R::R16G16B16A16Unorm => R16G16B16A16Unorm,
            R::R16G16B16A16Snorm => R16G16B16A16Snorm,
            R::R16G16B16A16Sint => R16G16B16A16Sint,
            R::R16G16B16A16Uint => R16G16B16A16Uint,
            R::R16G16B16A16Float => R16G16B16A16Float,
            R::R32G32Float => R32G32Float,
            R::R32G32Sint => R32G32Sint,
            R::R32G32Uint => R32G32Uint,
            R::R16G16B16X16Float => R16G16B16X16Float,
            R::A8R8G8B8Unorm | R::X8R8G8B8Unorm => B8G8R8A8Unorm,
            R::A8R8G8B8Srgb | R::X8R8G8B8Srgb => B8G8R8A8Srgb,
            R::A2B10G10R10Unorm => A2B10G10R10Unorm,
            R::A2B10G10R10Uint => A2B10G10R10Uint,
            R::A2R10G10B10Unorm => A2R10G10B10Unorm,
            R::A8B8G8R8Unorm | R::X8B8G8R8Unorm => A8B8G8R8Unorm,
            R::A8B8G8R8Srgb | R::X8B8G8R8Srgb => A8B8G8R8Srgb,
            R::A8B8G8R8Snorm => A8B8G8R8Snorm,
            R::A8B8G8R8Sint => A8B8G8R8Sint,
            R::A8B8G8R8Uint => A8B8G8R8Uint,
            R::R16G16Unorm => R16G16Unorm,
            R::R16G16Snorm => R16G16Snorm,
            R::R16G16Sint => R16G16Sint,
            R::R16G16Uint => R16G16Uint,
            R::R16G16Float => R16G16Float,
            R::B10G11R11Float => B10G11R11Float,
            R::R32Sint => R32Sint,
            R::R32Uint => R32Uint,
            R::R32Float => R32Float,
            R::R5G6B5Unorm => R5G6B5Unorm,
            R::A1R5G5B5Unorm | R::X1R5G5B5Unorm => A1R5G5B5Unorm,
            R::R8G8Unorm => R8G8Unorm,
            R::R8G8Snorm => R8G8Snorm,
            R::R8G8Sint => R8G8Sint,
            R::R8G8Uint => R8G8Uint,
            R::R16Unorm => R16Unorm,
            R::R16Snorm => R16Snorm,
            R::R16Sint => R16Sint,
            R::R16Uint => R16Uint,
            R::R16Float => R16Float,
            R::R8Unorm => R8Unorm,
            R::R8Snorm => R8Snorm,
            R::R8Sint => R8Sint,
            R::R8Uint => R8Uint,
            R::None => return None,
        })
    }

    /// Decodes the depth/stencil target's format register. `None` for the coverage formats
    /// the renderer cannot back.
    pub const fn from_depth(format: DepthFormat) -> Option<Self> {
        use DepthFormat as D;
        use PixelFormat::*;
        Some(match format {
            D::Z32Float => D32Float,
            D::Z16Unorm => D16Unorm,
            D::Z24UnormS8Uint => S8UintD24Unorm,
            D::X8Z24Unorm => X8D24Unorm,
            D::S8Z24Unorm => D24UnormS8Uint,
            D::S8Uint => S8Uint,
            D::Z32FloatX24S8Uint => D32FloatS8Uint,
            D::V8Z24Unorm => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_is_indexed_by_discriminant() {
        assert_eq!(PixelFormat::ALL.len(), PixelFormat::INFO.len());
        for (i, f) in PixelFormat::ALL.iter().enumerate() {
            assert_eq!(*f as usize, i);
        }
    }

    #[test]
    fn block_sizes_match_their_families() {
        for &f in PixelFormat::ALL {
            let info = f.info();
            match info.compression {
                None => assert_eq!((info.block_width, info.block_height), (1, 1), "{f:?}"),
                Some(Compression::Astc) => assert_eq!(info.bytes_per_block, 16, "{f:?}"),
                Some(Compression::Bc1 | Compression::Bc4) => {
                    assert_eq!(info.bytes_per_block, 8, "{f:?}")
                }
                Some(_) => assert_eq!((info.block_width, info.block_height), (4, 4), "{f:?}"),
            }
            if f.is_astc() {
                let name = format!("{f:?}");
                let dims = format!("{}x{}", info.block_width, info.block_height);
                assert!(name.contains(&dims), "{name} is not {dims}");
            }
        }
    }

    #[test]
    fn srgb_round_trips() {
        for &f in PixelFormat::ALL {
            if let Some(srgb) = f.to_srgb() {
                assert!(srgb.is_srgb(), "{f:?}");
                assert_eq!(srgb.to_linear(), f.to_linear());
                assert_eq!(srgb.info().bytes_per_block, f.info().bytes_per_block);
                assert_eq!(srgb.compression(), f.compression());
            }
            if f.is_srgb() {
                assert_eq!(f.to_linear().to_srgb(), Some(f), "{f:?}");
            }
        }
    }

    #[test]
    fn sizes_round_partial_blocks_up() {
        assert_eq!(
            PixelFormat::A8B8G8R8Unorm.size_bytes(3, 5, 2),
            3 * 5 * 2 * 4
        );
        assert_eq!(PixelFormat::Bc1RgbaUnorm.size_bytes(5, 5, 1), 2 * 2 * 8);
        assert_eq!(
            PixelFormat::Astc10x8Unorm.size_bytes(1280, 720, 1),
            128 * 90 * 16
        );
        assert_eq!(PixelFormat::Astc12x12Srgb.size_bytes(13, 1, 1), 2 * 16);
    }

    #[test]
    fn texture_header_decoding() {
        use ComponentType::*;
        let all = |c| [c; 4];
        assert_eq!(
            PixelFormat::from_texture(TextureFormat::A8B8G8R8, all(Unorm), true),
            Some(PixelFormat::A8B8G8R8Srgb)
        );
        assert_eq!(
            PixelFormat::from_texture(TextureFormat::Astc2d8x8, all(Unorm), true),
            Some(PixelFormat::Astc8x8Srgb)
        );
        assert_eq!(
            PixelFormat::from_texture(TextureFormat::Dxn2, all(UnormForceFp16), false),
            Some(PixelFormat::Bc5Unorm)
        );
        assert_eq!(
            PixelFormat::from_texture(TextureFormat::R16G16, all(Float), true),
            Some(PixelFormat::R16G16Float)
        );
        assert_eq!(
            PixelFormat::from_texture(TextureFormat::Z24S8, [Uint, Unorm, Unorm, Unorm], false),
            Some(PixelFormat::S8UintD24Unorm)
        );
        assert_eq!(
            PixelFormat::from_texture(TextureFormat::R32, [Float, Uint, Uint, Uint], false),
            None
        );
    }

    #[test]
    fn every_render_target_format_has_a_color_format() {
        for &f in RenderTargetFormat::ALL {
            match PixelFormat::from_render_target(f) {
                Some(p) => assert!(p.is_color() && !p.is_compressed(), "{f:?}"),
                None => assert_eq!(f, RenderTargetFormat::None),
            }
        }
        for &f in DepthFormat::ALL {
            if let Some(p) = PixelFormat::from_depth(f) {
                assert!(!p.is_color(), "{f:?}");
            }
        }
    }
}
