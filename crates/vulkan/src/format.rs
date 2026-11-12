//! Host formats for guest pixel formats.

use ash::vk;
use lichii_common::settings::AstcRecompression;
use lichii_texture::{Compression, PixelFormat};

use crate::device::Device;

/// How texel data reaches a host image of the format [`host_format`] picked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conversion {
    /// Upload the guest blocks as they are.
    None,
    /// Decode to RGBA8 (or R8/RG8/RGBA16F for single-channel and HDR families).
    Decode,
    /// Decode ASTC, then recompress to BC1 or BC3 to save memory (NX2_NOTES rows 99, 107).
    DecodeToBc(Compression),
}

/// The Vulkan format backing a guest format and what an upload must do to fill it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostFormat {
    pub format: vk::Format,
    pub conversion: Conversion,
    /// Can be a color or depth/stencil attachment.
    pub attachable: bool,
    /// Can be a storage image.
    pub storage: bool,
}

const A: u8 = 1;
const S: u8 = 2;

/// Native Vulkan format and usage (attachable `A`, storage `S`) of every guest format.
const fn native(format: PixelFormat) -> (vk::Format, u8) {
    use PixelFormat::*;
    use vk::Format as F;
    match format {
        A8B8G8R8Unorm => (F::A8B8G8R8_UNORM_PACK32, A | S),
        A8B8G8R8Snorm => (F::A8B8G8R8_SNORM_PACK32, A | S),
        A8B8G8R8Sint => (F::A8B8G8R8_SINT_PACK32, A | S),
        A8B8G8R8Uint => (F::A8B8G8R8_UINT_PACK32, A | S),
        R5G6B5Unorm => (F::R5G6B5_UNORM_PACK16, A),
        B5G6R5Unorm => (F::B5G6R5_UNORM_PACK16, 0),
        A1R5G5B5Unorm => (F::A1R5G5B5_UNORM_PACK16, A),
        A2B10G10R10Unorm => (F::A2B10G10R10_UNORM_PACK32, A | S),
        A2B10G10R10Uint => (F::A2B10G10R10_UINT_PACK32, A | S),
        A2R10G10B10Unorm => (F::A2R10G10B10_UNORM_PACK32, A),
        A1B5G5R5Unorm => (F::A1R5G5B5_UNORM_PACK16, A),
        A5B5G5R1Unorm => (F::R5G5B5A1_UNORM_PACK16, 0),
        R8Unorm => (F::R8_UNORM, A | S),
        R8Snorm => (F::R8_SNORM, A | S),
        R8Sint => (F::R8_SINT, A | S),
        R8Uint => (F::R8_UINT, A | S),
        R16G16B16A16Float => (F::R16G16B16A16_SFLOAT, A | S),
        R16G16B16A16Unorm => (F::R16G16B16A16_UNORM, A | S),
        R16G16B16A16Snorm => (F::R16G16B16A16_SNORM, A | S),
        R16G16B16A16Sint => (F::R16G16B16A16_SINT, A | S),
        R16G16B16A16Uint => (F::R16G16B16A16_UINT, A | S),
        B10G11R11Float => (F::B10G11R11_UFLOAT_PACK32, A | S),
        R32G32B32A32Uint => (F::R32G32B32A32_UINT, A | S),
        Bc1RgbaUnorm => (F::BC1_RGBA_UNORM_BLOCK, 0),
        Bc2Unorm => (F::BC2_UNORM_BLOCK, 0),
        Bc3Unorm => (F::BC3_UNORM_BLOCK, 0),
        Bc4Unorm => (F::BC4_UNORM_BLOCK, 0),
        Bc4Snorm => (F::BC4_SNORM_BLOCK, 0),
        Bc5Unorm => (F::BC5_UNORM_BLOCK, 0),
        Bc5Snorm => (F::BC5_SNORM_BLOCK, 0),
        Bc7Unorm => (F::BC7_UNORM_BLOCK, 0),
        Bc6hUfloat => (F::BC6H_UFLOAT_BLOCK, 0),
        Bc6hSfloat => (F::BC6H_SFLOAT_BLOCK, 0),
        Astc4x4Unorm => (F::ASTC_4X4_UNORM_BLOCK, 0),
        B8G8R8A8Unorm => (F::B8G8R8A8_UNORM, A | S),
        R32G32B32A32Float => (F::R32G32B32A32_SFLOAT, A | S),
        R32G32B32A32Sint => (F::R32G32B32A32_SINT, A | S),
        R32G32Float => (F::R32G32_SFLOAT, A | S),
        R32G32Sint => (F::R32G32_SINT, A | S),
        R32Float => (F::R32_SFLOAT, A | S),
        R16Float => (F::R16_SFLOAT, A | S),
        R16Unorm => (F::R16_UNORM, A | S),
        R16Snorm => (F::R16_SNORM, A | S),
        R16Uint => (F::R16_UINT, A | S),
        R16Sint => (F::R16_SINT, A | S),
        R16G16Unorm => (F::R16G16_UNORM, A | S),
        R16G16Float => (F::R16G16_SFLOAT, A | S),
        R16G16Uint => (F::R16G16_UINT, A | S),
        R16G16Sint => (F::R16G16_SINT, A | S),
        R16G16Snorm => (F::R16G16_SNORM, A | S),
        R32G32B32Float => (F::R32G32B32_SFLOAT, 0),
        A8B8G8R8Srgb => (F::A8B8G8R8_SRGB_PACK32, A),
        R8G8Unorm => (F::R8G8_UNORM, A | S),
        R8G8Snorm => (F::R8G8_SNORM, A | S),
        R8G8Sint => (F::R8G8_SINT, A | S),
        R8G8Uint => (F::R8G8_UINT, A | S),
        R32G32Uint => (F::R32G32_UINT, A | S),
        R16G16B16X16Float => (F::R16G16B16A16_SFLOAT, A | S),
        R32Uint => (F::R32_UINT, A | S),
        R32Sint => (F::R32_SINT, A | S),
        Astc8x8Unorm => (F::ASTC_8X8_UNORM_BLOCK, 0),
        Astc8x5Unorm => (F::ASTC_8X5_UNORM_BLOCK, 0),
        Astc5x4Unorm => (F::ASTC_5X4_UNORM_BLOCK, 0),
        B8G8R8A8Srgb => (F::B8G8R8A8_SRGB, A),
        Bc1RgbaSrgb => (F::BC1_RGBA_SRGB_BLOCK, 0),
        Bc2Srgb => (F::BC2_SRGB_BLOCK, 0),
        Bc3Srgb => (F::BC3_SRGB_BLOCK, 0),
        Bc7Srgb => (F::BC7_SRGB_BLOCK, 0),
        A4B4G4R4Unorm => (F::A4B4G4R4_UNORM_PACK16, 0),
        G4R4Unorm => (F::R4G4_UNORM_PACK8, 0),
        Astc4x4Srgb => (F::ASTC_4X4_SRGB_BLOCK, 0),
        Astc8x8Srgb => (F::ASTC_8X8_SRGB_BLOCK, 0),
        Astc8x5Srgb => (F::ASTC_8X5_SRGB_BLOCK, 0),
        Astc5x4Srgb => (F::ASTC_5X4_SRGB_BLOCK, 0),
        Astc5x5Unorm => (F::ASTC_5X5_UNORM_BLOCK, 0),
        Astc5x5Srgb => (F::ASTC_5X5_SRGB_BLOCK, 0),
        Astc10x8Unorm => (F::ASTC_10X8_UNORM_BLOCK, 0),
        Astc10x8Srgb => (F::ASTC_10X8_SRGB_BLOCK, 0),
        Astc6x6Unorm => (F::ASTC_6X6_UNORM_BLOCK, 0),
        Astc6x6Srgb => (F::ASTC_6X6_SRGB_BLOCK, 0),
        Astc10x6Unorm => (F::ASTC_10X6_UNORM_BLOCK, 0),
        Astc10x6Srgb => (F::ASTC_10X6_SRGB_BLOCK, 0),
        Astc10x5Unorm => (F::ASTC_10X5_UNORM_BLOCK, 0),
        Astc10x5Srgb => (F::ASTC_10X5_SRGB_BLOCK, 0),
        Astc10x10Unorm => (F::ASTC_10X10_UNORM_BLOCK, 0),
        Astc10x10Srgb => (F::ASTC_10X10_SRGB_BLOCK, 0),
        Astc12x10Unorm => (F::ASTC_12X10_UNORM_BLOCK, 0),
        Astc12x10Srgb => (F::ASTC_12X10_SRGB_BLOCK, 0),
        Astc12x12Unorm => (F::ASTC_12X12_UNORM_BLOCK, 0),
        Astc12x12Srgb => (F::ASTC_12X12_SRGB_BLOCK, 0),
        Astc8x6Unorm => (F::ASTC_8X6_UNORM_BLOCK, 0),
        Astc8x6Srgb => (F::ASTC_8X6_SRGB_BLOCK, 0),
        Astc6x5Unorm => (F::ASTC_6X5_UNORM_BLOCK, 0),
        Astc6x5Srgb => (F::ASTC_6X5_SRGB_BLOCK, 0),
        E5B9G9R9Float => (F::E5B9G9R9_UFLOAT_PACK32, 0),
        Etc2RgbUnorm => (F::ETC2_R8G8B8_UNORM_BLOCK, 0),
        Etc2RgbaUnorm => (F::ETC2_R8G8B8A8_UNORM_BLOCK, 0),
        Etc2RgbPtaUnorm => (F::ETC2_R8G8B8A1_UNORM_BLOCK, 0),
        Etc2RgbSrgb => (F::ETC2_R8G8B8_SRGB_BLOCK, 0),
        Etc2RgbaSrgb => (F::ETC2_R8G8B8A8_SRGB_BLOCK, 0),
        Etc2RgbPtaSrgb => (F::ETC2_R8G8B8A1_SRGB_BLOCK, 0),
        EacR11Unorm => (F::EAC_R11_UNORM_BLOCK, 0),
        EacR11Snorm => (F::EAC_R11_SNORM_BLOCK, 0),
        EacR11G11Unorm => (F::EAC_R11G11_UNORM_BLOCK, 0),
        EacR11G11Snorm => (F::EAC_R11G11_SNORM_BLOCK, 0),
        D32Float => (F::D32_SFLOAT, A),
        D16Unorm => (F::D16_UNORM, A),
        X8D24Unorm => (F::X8_D24_UNORM_PACK32, A),
        S8Uint => (F::S8_UINT, A),
        D24UnormS8Uint => (F::D24_UNORM_S8_UINT, A),
        S8UintD24Unorm => (F::D24_UNORM_S8_UINT, A),
        D32FloatS8Uint => (F::D32_SFLOAT_S8_UINT, A),
    }
}

/// The uncompressed format a decoded compressed format lands in.
const fn decoded(format: PixelFormat) -> vk::Format {
    use PixelFormat::*;
    match format {
        Bc4Unorm | EacR11Unorm => vk::Format::R8_UNORM,
        Bc4Snorm | EacR11Snorm => vk::Format::R8_SNORM,
        Bc5Unorm | EacR11G11Unorm => vk::Format::R8G8_UNORM,
        Bc5Snorm | EacR11G11Snorm => vk::Format::R8G8_SNORM,
        Bc6hUfloat | Bc6hSfloat => vk::Format::R16G16B16A16_SFLOAT,
        _ if format.is_srgb() => vk::Format::A8B8G8R8_SRGB_PACK32,
        _ => vk::Format::A8B8G8R8_UNORM_PACK32,
    }
}

fn required_features(attachable: bool, storage: bool, depth: bool) -> vk::FormatFeatureFlags {
    let mut features = vk::FormatFeatureFlags::SAMPLED_IMAGE
        | vk::FormatFeatureFlags::TRANSFER_SRC
        | vk::FormatFeatureFlags::TRANSFER_DST;
    if attachable {
        features |= if depth {
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
        } else {
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
        };
    }
    if storage {
        features |= vk::FormatFeatureFlags::STORAGE_IMAGE;
    }
    features
}

/// Vulkan formats tried, in order, when a depth/stencil format is missing (D24S8 is optional
/// on some AMD drivers) or a packed format is.
fn alternatives(format: vk::Format) -> &'static [vk::Format] {
    match format {
        vk::Format::D24_UNORM_S8_UINT => &[
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D16_UNORM_S8_UINT,
        ],
        vk::Format::X8_D24_UNORM_PACK32 => &[vk::Format::D32_SFLOAT, vk::Format::D24_UNORM_S8_UINT],
        vk::Format::D16_UNORM => &[vk::Format::D32_SFLOAT],
        vk::Format::A4B4G4R4_UNORM_PACK16
        | vk::Format::R4G4_UNORM_PACK8
        | vk::Format::B5G6R5_UNORM_PACK16
        | vk::Format::R5G5B5A1_UNORM_PACK16 => &[vk::Format::A8B8G8R8_UNORM_PACK32],
        _ => &[],
    }
}

/// Picks the host format for a guest format on `device`: the native format when the device
/// supports it with the needed features, otherwise a decode target (compressed families) or
/// the nearest supported alternative.
pub fn host_format(
    device: &Device,
    format: PixelFormat,
    recompression: AstcRecompression,
) -> HostFormat {
    let (native_format, usage) = native(format);
    let attachable = usage & A != 0;
    let storage = usage & S != 0;
    let depth = !format.is_color();
    let supports = |vk_format: vk::Format, attachable: bool, storage: bool| {
        device
            .format_features(vk_format)
            .contains(required_features(attachable, storage, depth))
    };
    if let Some(compression) = format.compression() {
        let native_ok = match compression {
            Compression::Astc => device.features.texture_compression_astc_ldr,
            Compression::Etc2 | Compression::Eac => device.features.texture_compression_etc2,
            _ => device.features.texture_compression_bc,
        } && supports(native_format, false, false);
        if native_ok {
            return HostFormat {
                format: native_format,
                conversion: Conversion::None,
                attachable: false,
                storage: false,
            };
        }
        let srgb = format.is_srgb();
        let recompress = match (
            compression,
            recompression,
            device.features.texture_compression_bc,
        ) {
            (Compression::Astc, AstcRecompression::Bc1, true) => Some(Compression::Bc1),
            (Compression::Astc, AstcRecompression::Bc3, true) => Some(Compression::Bc3),
            _ => None,
        };
        return match recompress {
            Some(Compression::Bc1) => HostFormat {
                format: if srgb {
                    vk::Format::BC1_RGBA_SRGB_BLOCK
                } else {
                    vk::Format::BC1_RGBA_UNORM_BLOCK
                },
                conversion: Conversion::DecodeToBc(Compression::Bc1),
                attachable: false,
                storage: false,
            },
            Some(_) => HostFormat {
                format: if srgb {
                    vk::Format::BC3_SRGB_BLOCK
                } else {
                    vk::Format::BC3_UNORM_BLOCK
                },
                conversion: Conversion::DecodeToBc(Compression::Bc3),
                attachable: false,
                storage: false,
            },
            None => {
                let format = decoded(format);
                HostFormat {
                    format,
                    conversion: Conversion::Decode,
                    attachable: false,
                    storage: !srgb,
                }
            }
        };
    }
    for (attachable, storage) in [(attachable, storage), (attachable, false), (false, false)] {
        if supports(native_format, attachable, storage) {
            return HostFormat {
                format: native_format,
                conversion: Conversion::None,
                attachable,
                storage,
            };
        }
        if let Some(&alternative) = alternatives(native_format)
            .iter()
            .find(|&&f| supports(f, attachable, false))
        {
            return HostFormat {
                format: alternative,
                conversion: Conversion::None,
                attachable,
                storage: false,
            };
        }
    }
    HostFormat {
        format: native_format,
        conversion: Conversion::None,
        attachable: false,
        storage: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_format_has_a_native_mapping_of_the_same_size() {
        for &format in PixelFormat::ALL {
            let (vk_format, usage) = native(format);
            assert_ne!(vk_format, vk::Format::UNDEFINED, "{format:?}");
            if usage & S != 0 {
                assert!(usage & A != 0, "{format:?} is storage but not attachable");
            }
            if format.is_compressed() {
                assert_eq!(usage, 0, "{format:?}");
            }
        }
    }
}
