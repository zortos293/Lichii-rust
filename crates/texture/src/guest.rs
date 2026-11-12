//! The guest's own format encodings: texture header (TIC) formats and component types,
//! and the render-target and depth-target format registers. Values are the hardware's.

macro_rules! guest_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident = $value:literal,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u32)]
        pub enum $name {
            $($variant = $value,)*
        }

        impl $name {
            pub const ALL: &[$name] = &[$($name::$variant,)*];
        }

        impl TryFrom<u32> for $name {
            type Error = u32;

            fn try_from(value: u32) -> Result<Self, u32> {
                match value {
                    $($value => Ok(Self::$variant),)*
                    _ => Err(value),
                }
            }
        }
    };
}

guest_enum! {
    /// Texture header format field.
    TextureFormat {
        R32G32B32A32 = 0x01,
        R32G32B32 = 0x02,
        R16G16B16A16 = 0x03,
        R32G32 = 0x04,
        R32B24G8 = 0x05,
        Etc2Rgb = 0x06,
        X8B8G8R8 = 0x07,
        A8B8G8R8 = 0x08,
        A2B10G10R10 = 0x09,
        Etc2RgbPta = 0x0a,
        Etc2Rgba = 0x0b,
        R16G16 = 0x0c,
        G8R24 = 0x0d,
        G24R8 = 0x0e,
        R32 = 0x0f,
        Bc6hS16 = 0x10,
        Bc6hU16 = 0x11,
        A4B4G4R4 = 0x12,
        A5B5G5R1 = 0x13,
        A1B5G5R5 = 0x14,
        B5G6R5 = 0x15,
        B6G5R5 = 0x16,
        Bc7U = 0x17,
        G8R8 = 0x18,
        Eac = 0x19,
        EacX2 = 0x1a,
        R16 = 0x1b,
        Y8Video = 0x1c,
        R8 = 0x1d,
        G4R4 = 0x1e,
        R1 = 0x1f,
        E5B9G9R9 = 0x20,
        B10G11R11 = 0x21,
        G8B8G8R8 = 0x22,
        B8G8R8G8 = 0x23,
        Dxt1 = 0x24,
        Dxt23 = 0x25,
        Dxt45 = 0x26,
        Dxn1 = 0x27,
        Dxn2 = 0x28,
        Z24S8 = 0x29,
        X8Z24 = 0x2a,
        S8Z24 = 0x2b,
        X4V4Z24Cov4R4V = 0x2c,
        X4V4Z24Cov8R8V = 0x2d,
        V8Z24Cov4R12V = 0x2e,
        Z32 = 0x2f,
        Z32X24S8 = 0x30,
        X8Z24X20V4S8Cov4R4V = 0x31,
        X8Z24X20V4S8Cov8R8V = 0x32,
        Z32X20V4X8Cov4R4V = 0x33,
        Z32X20V4X8Cov8R8V = 0x34,
        Z32X20V4S8Cov4R4V = 0x35,
        Z32X20V4S8Cov8R8V = 0x36,
        X8Z24X16V8S8Cov4R12V = 0x37,
        Z32X16V8X8Cov4R12V = 0x38,
        Z32X16V8S8Cov4R12V = 0x39,
        Z16 = 0x3a,
        V8Z24Cov8R24V = 0x3b,
        X8Z24X16V8S8Cov8R24V = 0x3c,
        Z32X16V8X8Cov8R24V = 0x3d,
        Z32X16V8S8Cov8R24V = 0x3e,
        Astc2d4x4 = 0x40,
        Astc2d5x5 = 0x41,
        Astc2d6x6 = 0x42,
        Astc2d8x8 = 0x44,
        Astc2d10x10 = 0x45,
        Astc2d12x12 = 0x46,
        Astc2d5x4 = 0x50,
        Astc2d6x5 = 0x51,
        Astc2d8x6 = 0x52,
        Astc2d10x8 = 0x53,
        Astc2d12x10 = 0x54,
        Astc2d8x5 = 0x55,
        Astc2d10x5 = 0x56,
        Astc2d10x6 = 0x57,
    }
}

guest_enum! {
    /// Texture header per-component numeric type.
    ComponentType {
        Snorm = 1,
        Unorm = 2,
        Sint = 3,
        Uint = 4,
        SnormForceFp16 = 5,
        UnormForceFp16 = 6,
        Float = 7,
    }
}

impl ComponentType {
    /// The type with fp16 forcing dropped; precision forcing does not change the format.
    pub const fn without_fp16(self) -> Self {
        match self {
            Self::SnormForceFp16 => Self::Snorm,
            Self::UnormForceFp16 => Self::Unorm,
            other => other,
        }
    }
}

guest_enum! {
    /// Color render target format register (Maxwell 3D and Ampere 3D share the encoding).
    RenderTargetFormat {
        None = 0x0,
        R32G32B32A32Float = 0xc0,
        R32G32B32A32Sint = 0xc1,
        R32G32B32A32Uint = 0xc2,
        R32G32B32X32Float = 0xc3,
        R32G32B32X32Sint = 0xc4,
        R32G32B32X32Uint = 0xc5,
        R16G16B16A16Unorm = 0xc6,
        R16G16B16A16Snorm = 0xc7,
        R16G16B16A16Sint = 0xc8,
        R16G16B16A16Uint = 0xc9,
        R16G16B16A16Float = 0xca,
        R32G32Float = 0xcb,
        R32G32Sint = 0xcc,
        R32G32Uint = 0xcd,
        R16G16B16X16Float = 0xce,
        A8R8G8B8Unorm = 0xcf,
        A8R8G8B8Srgb = 0xd0,
        A2B10G10R10Unorm = 0xd1,
        A2B10G10R10Uint = 0xd2,
        A8B8G8R8Unorm = 0xd5,
        A8B8G8R8Srgb = 0xd6,
        A8B8G8R8Snorm = 0xd7,
        A8B8G8R8Sint = 0xd8,
        A8B8G8R8Uint = 0xd9,
        R16G16Unorm = 0xda,
        R16G16Snorm = 0xdb,
        R16G16Sint = 0xdc,
        R16G16Uint = 0xdd,
        R16G16Float = 0xde,
        A2R10G10B10Unorm = 0xdf,
        B10G11R11Float = 0xe0,
        R32Sint = 0xe3,
        R32Uint = 0xe4,
        R32Float = 0xe5,
        X8R8G8B8Unorm = 0xe6,
        X8R8G8B8Srgb = 0xe7,
        R5G6B5Unorm = 0xe8,
        A1R5G5B5Unorm = 0xe9,
        R8G8Unorm = 0xea,
        R8G8Snorm = 0xeb,
        R8G8Sint = 0xec,
        R8G8Uint = 0xed,
        R16Unorm = 0xee,
        R16Snorm = 0xef,
        R16Sint = 0xf0,
        R16Uint = 0xf1,
        R16Float = 0xf2,
        R8Unorm = 0xf3,
        R8Snorm = 0xf4,
        R8Sint = 0xf5,
        R8Uint = 0xf6,
        X1R5G5B5Unorm = 0xf8,
        X8B8G8R8Unorm = 0xf9,
        X8B8G8R8Srgb = 0xfa,
    }
}

guest_enum! {
    /// Depth/stencil target format register.
    DepthFormat {
        Z32Float = 0x0a,
        Z16Unorm = 0x13,
        Z24UnormS8Uint = 0x14,
        X8Z24Unorm = 0x15,
        S8Z24Unorm = 0x16,
        S8Uint = 0x17,
        V8Z24Unorm = 0x18,
        Z32FloatX24S8Uint = 0x19,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_values_round_trip() {
        for &f in TextureFormat::ALL {
            assert_eq!(TextureFormat::try_from(f as u32), Ok(f));
        }
        for &f in RenderTargetFormat::ALL {
            assert_eq!(RenderTargetFormat::try_from(f as u32), Ok(f));
        }
        assert_eq!(TextureFormat::try_from(0x43), Err(0x43));
        assert_eq!(DepthFormat::try_from(0x14), Ok(DepthFormat::Z24UnormS8Uint));
    }
}
