//! Display names and value lists for the settings enums, for combo boxes and cycling hotkeys.

use lichii_common::ConsoleGeneration;
use lichii_common::settings::*;

pub trait Choice: Copy + PartialEq + 'static {
    const ALL: &'static [Self];

    fn label(self) -> &'static str;

    /// The value after this one, wrapping around.
    fn next(self) -> Self {
        let i = Self::ALL.iter().position(|&v| v == self).unwrap_or(0);
        Self::ALL[(i + 1) % Self::ALL.len()]
    }
}

macro_rules! choices {
    ($ty:ty { $($variant:ident => $label:literal),* $(,)? }) => {
        impl Choice for $ty {
            const ALL: &'static [Self] = &[$(<$ty>::$variant),*];

            fn label(self) -> &'static str {
                match self { $(<$ty>::$variant => $label),* }
            }
        }
    };
}

choices!(ConsoleGeneration { Switch1 => "Switch", Switch2 => "Switch 2" });
choices!(Language {
    Japanese => "Japanese (日本語)",
    AmericanEnglish => "American English",
    French => "French (français)",
    German => "German (Deutsch)",
    Italian => "Italian (italiano)",
    Spanish => "Spanish (español)",
    Chinese => "Chinese",
    Korean => "Korean (한국어)",
    Dutch => "Dutch (Nederlands)",
    Portuguese => "Portuguese (português)",
    Russian => "Russian (Русский)",
    Taiwanese => "Taiwanese",
    BritishEnglish => "British English",
    CanadianFrench => "Canadian French",
    LatinAmericanSpanish => "Latin American Spanish",
    SimplifiedChinese => "Simplified Chinese",
    TraditionalChinese => "Traditional Chinese",
    BrazilianPortuguese => "Brazilian Portuguese",
});
choices!(Region {
    Japan => "Japan",
    Usa => "USA",
    Europe => "Europe",
    Australia => "Australia",
    China => "China",
    Korea => "Korea",
    Taiwan => "Taiwan",
});
choices!(MemoryLayout {
    Auto => "Auto (4 GiB Switch, 12 GiB Switch 2)",
    Dram4GiB => "4 GiB",
    Dram6GiB => "6 GiB (unsafe)",
    Dram8GiB => "8 GiB (unsafe)",
    Dram12GiB => "12 GiB",
});
choices!(CpuBackend { Interpreter => "Interpreter (slow)", Jit => "JIT" });
choices!(CpuAccuracy {
    Auto => "Auto",
    Accurate => "Accurate",
    Unsafe => "Unsafe",
    Paranoid => "Paranoid (slow)",
});
choices!(ScalingFilter {
    NearestNeighbor => "Nearest Neighbor",
    Bilinear => "Bilinear",
    Bicubic => "Bicubic",
    Gaussian => "Gaussian",
    Lanczos => "Lanczos",
    ScaleForce => "ScaleForce",
    Fsr => "AMD FidelityFX Super Resolution",
    Area => "Area",
    Mmpx => "MMPX",
});
choices!(AntiAliasing { None => "None", Fxaa => "FXAA", Smaa => "SMAA" });
choices!(AspectRatio {
    R16x9 => "Default (16:9)",
    R4x3 => "Force 4:3",
    R21x9 => "Force 21:9",
    R16x10 => "Force 16:10",
    Stretch => "Stretch to Window",
});
choices!(PresentMode {
    Fifo => "FIFO (VSync)",
    FifoRelaxed => "FIFO Relaxed",
    Mailbox => "Mailbox",
    Immediate => "Immediate (no VSync)",
});
choices!(FullscreenMode { Borderless => "Borderless Windowed", Exclusive => "Exclusive Fullscreen" });
choices!(GpuAccuracy { Low => "Low (fast)", Normal => "Normal", High => "High (slow)" });
choices!(Anisotropy {
    Auto => "Automatic",
    X1 => "Default",
    X2 => "2x",
    X4 => "4x",
    X8 => "8x",
    X16 => "16x",
});
choices!(AstcDecode { Cpu => "CPU", Gpu => "GPU", CpuAsync => "CPU Asynchronously" });
choices!(AstcRecompression {
    Uncompressed => "Uncompressed (best quality)",
    Bc1 => "BC1 (low quality)",
    Bc3 => "BC3 (medium quality)",
});
choices!(VramUsage { Conservative => "Conservative", Aggressive => "Aggressive" });
choices!(OverlayPosition {
    TopLeft => "Top left",
    TopRight => "Top right",
    BottomLeft => "Bottom left",
    BottomRight => "Bottom right",
});
choices!(AudioOutput { Auto => "Auto", Cpal => "cpal", Null => "Null (no sound)" });
choices!(Theme { Dark => "Dark", Light => "Light" });

impl Choice for ResolutionScale {
    const ALL: &'static [Self] = &ResolutionScale::ALL;

    fn label(self) -> &'static str {
        ResolutionScale::label(self)
    }
}

impl Choice for ControllerType {
    const ALL: &'static [Self] = &ControllerType::ALL;

    fn label(self) -> &'static str {
        ControllerType::label(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_wraps() {
        assert_eq!(GpuAccuracy::High.next(), GpuAccuracy::Low);
        assert_eq!(ScalingFilter::Fsr.next(), ScalingFilter::Area);
        assert_eq!(
            ResolutionScale::ALL.len(),
            <ResolutionScale as Choice>::ALL.len()
        );
    }

    #[test]
    fn labels_are_unique() {
        fn check<T: Choice + std::fmt::Debug>() {
            let mut labels: Vec<_> = T::ALL.iter().map(|v| v.label()).collect();
            labels.sort_unstable();
            labels.dedup();
            assert_eq!(labels.len(), T::ALL.len(), "{:?}", T::ALL);
        }
        check::<ScalingFilter>();
        check::<Language>();
        check::<ResolutionScale>();
        check::<ControllerType>();
        check::<Anisotropy>();
    }
}
