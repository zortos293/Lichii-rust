//! Guest sampler descriptors (TSC entries) to Vulkan sampler parameters, for the texture cache's
//! sampler cache (the sampler half of `maxwell_to_vk.cpp` and `Sampler::Sampler`).

use ash::vk;
use bytemuck::{Pod, Zeroable};
use lichii_common::settings::Anisotropy;

use crate::device::Device;
use crate::error::{Result, VkResultExt};

macro_rules! tsc_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident = $value:literal,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum $name {
            $($variant = $value,)*
        }

        impl $name {
            pub const ALL: &[$name] = &[$($name::$variant,)*];

            pub fn from_raw(raw: u32) -> Option<Self> {
                match raw {
                    $($value => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

tsc_enum! {
    TextureFilter {
        Nearest = 1,
        Linear = 2,
    }
}

tsc_enum! {
    MipmapFilter {
        None = 1,
        Nearest = 2,
        Linear = 3,
    }
}

tsc_enum! {
    WrapMode {
        Wrap = 0,
        Mirror = 1,
        ClampToEdge = 2,
        Border = 3,
        Clamp = 4,
        MirrorOnceClampToEdge = 5,
        MirrorOnceBorder = 6,
        MirrorOnceClampOgl = 7,
    }
}

tsc_enum! {
    DepthCompareFunc {
        Never = 0,
        Less = 1,
        Equal = 2,
        LessEqual = 3,
        Greater = 4,
        NotEqual = 5,
        GreaterEqual = 6,
        Always = 7,
    }
}

tsc_enum! {
    SamplerReduction {
        WeightedAverage = 0,
        Min = 1,
        Max = 2,
    }
}

/// A guest texture sampler control entry, as read from the sampler pool.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct Tsc(pub [u32; 8]);

impl Tsc {
    fn bits(&self, word: usize, lo: u32, len: u32) -> u32 {
        (self.0[word] >> lo) & ((1 << len) - 1)
    }

    fn wrap(&self, lo: u32) -> WrapMode {
        WrapMode::from_raw(self.bits(0, lo, 3)).expect("every 3-bit wrap mode is defined")
    }

    pub fn wrap_u(&self) -> WrapMode {
        self.wrap(0)
    }

    pub fn wrap_v(&self) -> WrapMode {
        self.wrap(3)
    }

    pub fn wrap_p(&self) -> WrapMode {
        self.wrap(6)
    }

    /// The depth comparison, `None` when comparison is off.
    pub fn depth_compare(&self) -> Option<DepthCompareFunc> {
        (self.bits(0, 9, 1) != 0).then(|| {
            DepthCompareFunc::from_raw(self.bits(0, 10, 3))
                .expect("every 3-bit function is defined")
        })
    }

    pub fn srgb_conversion(&self) -> bool {
        self.bits(0, 13, 1) != 0
    }

    /// The guest's anisotropy as a sample count: 1, 2, 4 ... 128.
    pub fn max_anisotropy(&self) -> u32 {
        1 << self.bits(0, 20, 3)
    }

    pub fn mag_filter(&self) -> Option<TextureFilter> {
        TextureFilter::from_raw(self.bits(1, 0, 2))
    }

    pub fn min_filter(&self) -> Option<TextureFilter> {
        TextureFilter::from_raw(self.bits(1, 4, 2))
    }

    pub fn mipmap_filter(&self) -> Option<MipmapFilter> {
        MipmapFilter::from_raw(self.bits(1, 6, 2))
    }

    pub fn reduction(&self) -> Option<SamplerReduction> {
        SamplerReduction::from_raw(self.bits(1, 10, 2))
    }

    /// Signed 5.8 fixed point.
    pub fn lod_bias(&self) -> f32 {
        let raw = self.bits(1, 12, 13);
        ((raw << 19) as i32 >> 19) as f32 / 256.0
    }

    pub fn min_lod(&self) -> f32 {
        self.bits(2, 0, 12) as f32 / 256.0
    }

    pub fn max_lod(&self) -> f32 {
        self.bits(2, 12, 12) as f32 / 256.0
    }

    pub fn border_color(&self) -> [f32; 4] {
        [4, 5, 6, 7].map(|word| f32::from_bits(self.0[word]))
    }
}

pub fn filter(filter: TextureFilter) -> vk::Filter {
    match filter {
        TextureFilter::Nearest => vk::Filter::NEAREST,
        TextureFilter::Linear => vk::Filter::LINEAR,
    }
}

/// Unmipmapped sampling is nearest mip selection limited to LOD 0.25 ([`SamplerParams::new`]).
pub fn mipmap_mode(filter: MipmapFilter) -> vk::SamplerMipmapMode {
    match filter {
        MipmapFilter::None | MipmapFilter::Nearest => vk::SamplerMipmapMode::NEAREST,
        MipmapFilter::Linear => vk::SamplerMipmapMode::LINEAR,
    }
}

/// GL's `CLAMP` samples the border under linear filtering and the edge under nearest, so
/// it depends on the magnification filter. The mirror-once modes need
/// `samplerMirrorClampToEdge`; without it they mirror without clamping.
pub fn address_mode(
    wrap: WrapMode,
    mag_filter: TextureFilter,
    mirror_clamp_to_edge: bool,
) -> vk::SamplerAddressMode {
    use vk::SamplerAddressMode as A;
    match wrap {
        WrapMode::Wrap => A::REPEAT,
        WrapMode::Mirror => A::MIRRORED_REPEAT,
        WrapMode::ClampToEdge => A::CLAMP_TO_EDGE,
        WrapMode::Border => A::CLAMP_TO_BORDER,
        WrapMode::Clamp => match mag_filter {
            TextureFilter::Nearest => A::CLAMP_TO_EDGE,
            TextureFilter::Linear => A::CLAMP_TO_BORDER,
        },
        WrapMode::MirrorOnceClampToEdge
        | WrapMode::MirrorOnceBorder
        | WrapMode::MirrorOnceClampOgl => {
            if mirror_clamp_to_edge {
                A::MIRROR_CLAMP_TO_EDGE
            } else {
                A::MIRRORED_REPEAT
            }
        }
    }
}

pub fn compare_op(func: DepthCompareFunc) -> vk::CompareOp {
    match func {
        DepthCompareFunc::Never => vk::CompareOp::NEVER,
        DepthCompareFunc::Less => vk::CompareOp::LESS,
        DepthCompareFunc::Equal => vk::CompareOp::EQUAL,
        DepthCompareFunc::LessEqual => vk::CompareOp::LESS_OR_EQUAL,
        DepthCompareFunc::Greater => vk::CompareOp::GREATER,
        DepthCompareFunc::NotEqual => vk::CompareOp::NOT_EQUAL,
        DepthCompareFunc::GreaterEqual => vk::CompareOp::GREATER_OR_EQUAL,
        DepthCompareFunc::Always => vk::CompareOp::ALWAYS,
    }
}

pub fn reduction_mode(reduction: SamplerReduction) -> vk::SamplerReductionMode {
    match reduction {
        SamplerReduction::WeightedAverage => vk::SamplerReductionMode::WEIGHTED_AVERAGE,
        SamplerReduction::Min => vk::SamplerReductionMode::MIN,
        SamplerReduction::Max => vk::SamplerReductionMode::MAX,
    }
}

/// The closest built-in border colour, for devices without custom border colours.
pub fn standard_border(color: [f32; 4]) -> vk::BorderColor {
    let [r, g, b, a] = color;
    if a < 0.5 {
        vk::BorderColor::FLOAT_TRANSPARENT_BLACK
    } else if r + g + b >= 1.5 {
        vk::BorderColor::FLOAT_OPAQUE_WHITE
    } else {
        vk::BorderColor::FLOAT_OPAQUE_BLACK
    }
}

/// Sampler capabilities of the device.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplerCaps {
    pub anisotropy: bool,
    pub max_anisotropy: f32,
    pub max_lod_bias: f32,
    /// Custom border colours without a format (VK_EXT_custom_border_color).
    pub custom_border_color: bool,
    pub minmax: bool,
    pub mirror_clamp_to_edge: bool,
}

impl SamplerCaps {
    pub fn from_device(device: &Device) -> Self {
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut border = vk::PhysicalDeviceCustomBorderColorFeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut v12);
        if device.features.extensions.custom_border_color {
            features = features.push_next(&mut border);
        }
        // SAFETY: `physical` belongs to this instance and the chain only holds structs of
        // extensions the device reports.
        unsafe {
            device
                .instance
                .raw
                .get_physical_device_features2(device.physical, &mut features)
        };
        let limits = &device.properties.limits;
        Self {
            anisotropy: device.features.sampler_anisotropy,
            max_anisotropy: limits.max_sampler_anisotropy,
            max_lod_bias: limits.max_sampler_lod_bias,
            custom_border_color: border.custom_border_colors != vk::FALSE
                && device.features.custom_border_color_without_format,
            minmax: device.features.sampler_filter_minmax,
            mirror_clamp_to_edge: v12.sampler_mirror_clamp_to_edge != vk::FALSE,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Border {
    Standard(vk::BorderColor),
    /// RGBA, unformatted.
    Custom([f32; 4]),
}

/// A host sampler for one TSC entry, already limited to what the device supports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerParams {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address: [vk::SamplerAddressMode; 3],
    pub lod_bias: f32,
    /// `None` disables anisotropic filtering.
    pub anisotropy: Option<f32>,
    pub compare: Option<vk::CompareOp>,
    pub min_lod: f32,
    pub max_lod: f32,
    pub border: Border,
    pub reduction: vk::SamplerReductionMode,
}

impl SamplerParams {
    /// `setting` is the user's anisotropic filtering: `Auto` keeps the guest's value, `X1`
    /// turns it off, and `X2`..`X16` raise trilinear-filtered, fully mipmapped samplers to at
    /// least that many samples. Undefined filter encodings sample nearest.
    pub fn new(tsc: &Tsc, setting: Anisotropy, caps: &SamplerCaps) -> Self {
        let mag = tsc.mag_filter().unwrap_or(TextureFilter::Nearest);
        let min = tsc.min_filter().unwrap_or(TextureFilter::Nearest);
        let mipmap = tsc.mipmap_filter().unwrap_or(MipmapFilter::None);
        let reduction = tsc.reduction().unwrap_or(SamplerReduction::WeightedAverage);
        let compare = tsc.depth_compare().map(compare_op);
        let (min_lod, max_lod) = match mipmap {
            MipmapFilter::None => (0.0, 0.25),
            _ => (tsc.min_lod(), tsc.max_lod().max(tsc.min_lod())),
        };

        let guest = tsc.max_anisotropy() as f32;
        let filtered_mips = mipmap != MipmapFilter::None
            && tsc.min_lod() == 0.0
            && tsc.max_lod() >= 1.0
            && min == TextureFilter::Linear
            && reduction == SamplerReduction::WeightedAverage
            && compare.is_none();
        let forced = match setting {
            Anisotropy::Auto => 1.0,
            Anisotropy::X1 => 0.0,
            Anisotropy::X2 => 2.0,
            Anisotropy::X4 => 4.0,
            Anisotropy::X8 => 8.0,
            Anisotropy::X16 => 16.0,
        };
        let wanted = match setting {
            Anisotropy::X1 => 1.0,
            _ if filtered_mips => guest.max(forced),
            _ => guest,
        };
        let anisotropy = (caps.anisotropy && wanted > 1.0).then(|| wanted.min(caps.max_anisotropy));

        let border_color = tsc.border_color();
        Self {
            mag_filter: filter(mag),
            min_filter: filter(min),
            mipmap_mode: mipmap_mode(mipmap),
            address: [tsc.wrap_u(), tsc.wrap_v(), tsc.wrap_p()]
                .map(|wrap| address_mode(wrap, mag, caps.mirror_clamp_to_edge)),
            lod_bias: tsc.lod_bias().clamp(-caps.max_lod_bias, caps.max_lod_bias),
            anisotropy,
            compare,
            min_lod,
            max_lod,
            border: if caps.custom_border_color {
                Border::Custom(border_color)
            } else {
                Border::Standard(standard_border(border_color))
            },
            reduction: if caps.minmax && compare.is_none() {
                reduction_mode(reduction)
            } else {
                vk::SamplerReductionMode::WEIGHTED_AVERAGE
            },
        }
    }

    /// Creates the sampler; the caller owns and destroys it.
    pub fn create(&self, device: &Device) -> Result<vk::Sampler> {
        let mut info = vk::SamplerCreateInfo::default()
            .mag_filter(self.mag_filter)
            .min_filter(self.min_filter)
            .mipmap_mode(self.mipmap_mode)
            .address_mode_u(self.address[0])
            .address_mode_v(self.address[1])
            .address_mode_w(self.address[2])
            .mip_lod_bias(self.lod_bias)
            .anisotropy_enable(self.anisotropy.is_some())
            .max_anisotropy(self.anisotropy.unwrap_or(1.0))
            .compare_enable(self.compare.is_some())
            .compare_op(self.compare.unwrap_or(vk::CompareOp::NEVER))
            .min_lod(self.min_lod)
            .max_lod(self.max_lod);
        let mut custom = vk::SamplerCustomBorderColorCreateInfoEXT::default();
        let mut reduction =
            vk::SamplerReductionModeCreateInfo::default().reduction_mode(self.reduction);
        match self.border {
            Border::Standard(color) => info = info.border_color(color),
            Border::Custom(float32) => {
                custom = custom.custom_border_color(vk::ClearColorValue { float32 });
                info = info
                    .border_color(vk::BorderColor::FLOAT_CUSTOM_EXT)
                    .push_next(&mut custom);
            }
        }
        if self.reduction != vk::SamplerReductionMode::WEIGHTED_AVERAGE {
            info = info.push_next(&mut reduction);
        }
        // SAFETY: the parameters were limited to the device's capabilities in `new`.
        unsafe {
            device
                .raw
                .create_sampler(&info, None)
                .call("vkCreateSampler")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lichii_common::settings::GraphicsSettings;

    use super::*;
    use crate::{Instance, SurfaceSupport};

    fn caps() -> SamplerCaps {
        SamplerCaps {
            anisotropy: true,
            max_anisotropy: 16.0,
            max_lod_bias: 15.0,
            custom_border_color: true,
            minmax: true,
            mirror_clamp_to_edge: true,
        }
    }

    fn tsc(word0: u32, word1: u32, word2: u32, border: [f32; 4]) -> Tsc {
        let b = border.map(f32::to_bits);
        Tsc([word0, word1, word2, 0, b[0], b[1], b[2], b[3]])
    }

    #[test]
    fn every_guest_value_translates() {
        for raw in 0..8 {
            assert!(WrapMode::from_raw(raw).is_some());
            assert!(DepthCompareFunc::from_raw(raw).is_some());
        }
        assert_eq!(WrapMode::ALL.len(), 8);
        assert_eq!(DepthCompareFunc::ALL.len(), 8);
        for &func in DepthCompareFunc::ALL {
            assert_eq!(
                compare_op(func).as_raw(),
                func as i32,
                "TSC order matches Vulkan's"
            );
        }
        for &f in TextureFilter::ALL {
            for &wrap in WrapMode::ALL {
                address_mode(wrap, f, true);
                address_mode(wrap, f, false);
            }
            filter(f);
        }
        for &m in MipmapFilter::ALL {
            mipmap_mode(m);
        }
        for &r in SamplerReduction::ALL {
            reduction_mode(r);
        }
        assert_eq!(TextureFilter::from_raw(0), None);
        assert_eq!(TextureFilter::from_raw(3), None);
        assert_eq!(MipmapFilter::from_raw(0), None);
        assert_eq!(SamplerReduction::from_raw(3), None);
        assert_eq!(
            address_mode(WrapMode::Clamp, TextureFilter::Linear, true),
            vk::SamplerAddressMode::CLAMP_TO_BORDER
        );
        assert_eq!(
            address_mode(WrapMode::Clamp, TextureFilter::Nearest, true),
            vk::SamplerAddressMode::CLAMP_TO_EDGE
        );
        assert_eq!(
            address_mode(WrapMode::MirrorOnceClampOgl, TextureFilter::Linear, false),
            vk::SamplerAddressMode::MIRRORED_REPEAT
        );
    }

    #[test]
    fn tsc_fields_decode() {
        let entry = tsc(
            4 | 3 << 3 | 5 << 6 | 1 << 9 | 3 << 10 | 1 << 13 | 2 << 20,
            2 | 2 << 4 | 3 << 6 | 1 << 10 | (0x1fff - 127) << 12,
            256 | (13 * 256) << 12,
            [0.25, 0.5, 0.75, 1.0],
        );
        assert_eq!(entry.wrap_u(), WrapMode::Clamp);
        assert_eq!(entry.wrap_v(), WrapMode::Border);
        assert_eq!(entry.wrap_p(), WrapMode::MirrorOnceClampToEdge);
        assert_eq!(entry.depth_compare(), Some(DepthCompareFunc::LessEqual));
        assert!(entry.srgb_conversion());
        assert_eq!(entry.max_anisotropy(), 4);
        assert_eq!(entry.mag_filter(), Some(TextureFilter::Linear));
        assert_eq!(entry.min_filter(), Some(TextureFilter::Linear));
        assert_eq!(entry.mipmap_filter(), Some(MipmapFilter::Linear));
        assert_eq!(entry.reduction(), Some(SamplerReduction::Min));
        assert_eq!(entry.lod_bias(), -0.5);
        assert_eq!(entry.min_lod(), 1.0);
        assert_eq!(entry.max_lod(), 13.0);
        assert_eq!(entry.border_color(), [0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn params_follow_settings_and_caps() {
        let trilinear = tsc(0, 2 | 2 << 4 | 3 << 6, (4 * 256) << 12, [0.0; 4]);
        let at =
            |setting, caps: &SamplerCaps| SamplerParams::new(&trilinear, setting, caps).anisotropy;
        assert_eq!(at(Anisotropy::Auto, &caps()), None, "guest asks for 1x");
        assert_eq!(at(Anisotropy::X8, &caps()), Some(8.0));
        assert_eq!(at(Anisotropy::X1, &caps()), None);
        let low = SamplerCaps {
            max_anisotropy: 4.0,
            ..caps()
        };
        assert_eq!(at(Anisotropy::X16, &low), Some(4.0));
        let none = SamplerCaps {
            anisotropy: false,
            ..caps()
        };
        assert_eq!(at(Anisotropy::X16, &none), None);

        let nearest = tsc(0, 1 | 1 << 4 | 1 << 6, 0, [1.0, 1.0, 1.0, 1.0]);
        let params = SamplerParams::new(&nearest, Anisotropy::X16, &caps());
        assert_eq!(
            params.anisotropy, None,
            "unmipmapped samplers stay isotropic"
        );
        assert_eq!((params.min_lod, params.max_lod), (0.0, 0.25));
        assert_eq!(params.border, Border::Custom([1.0; 4]));
        let fallback = SamplerParams::new(
            &nearest,
            Anisotropy::Auto,
            &SamplerCaps {
                custom_border_color: false,
                ..caps()
            },
        );
        assert_eq!(
            fallback.border,
            Border::Standard(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        );
        assert_eq!(
            standard_border([0.0, 0.0, 0.0, 1.0]),
            vk::BorderColor::FLOAT_OPAQUE_BLACK
        );
        assert_eq!(
            standard_border([1.0, 1.0, 1.0, 0.0]),
            vk::BorderColor::FLOAT_TRANSPARENT_BLACK
        );

        let shadow = tsc(
            1 << 9 | 4 << 10,
            2 | 2 << 4 | 2 << 6 | 2 << 10,
            256 << 12,
            [0.0; 4],
        );
        let params = SamplerParams::new(&shadow, Anisotropy::Auto, &caps());
        assert_eq!(params.compare, Some(vk::CompareOp::GREATER));
        assert_eq!(params.reduction, vk::SamplerReductionMode::WEIGHTED_AVERAGE);
    }

    #[test]
    fn samplers_create_without_validation_errors() {
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let device = match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let caps = SamplerCaps::from_device(&device);
        let entries = [
            tsc(0, 2 | 2 << 4 | 3 << 6, (4 * 256) << 12, [0.0; 4]),
            tsc(
                3 | 3 << 3 | 3 << 6 | 4 << 20,
                2 | 2 << 4 | 3 << 6 | 0x1f00 << 12,
                255 << 12,
                [0.2, 0.4, 0.6, 0.8],
            ),
            tsc(
                1 << 9 | 6 << 10 | 5 | 6 << 3 | 7 << 6,
                2 | 2 << 4 | 2 << 6,
                8 << 12,
                [0.0; 4],
            ),
            tsc(4 | 4 << 3, 2 | 2 << 4 | 1 << 6 | 2 << 10, 0, [1.0; 4]),
            tsc(0, 0, 0, [0.0; 4]),
        ];
        for entry in &entries {
            for setting in [Anisotropy::Auto, Anisotropy::X16] {
                let sampler = SamplerParams::new(entry, setting, &caps)
                    .create(&device)
                    .unwrap();
                // SAFETY: the sampler was never used.
                unsafe { device.raw.destroy_sampler(sampler, None) };
            }
        }
        assert_eq!(device.instance().validation_errors(), 0);
    }
}
