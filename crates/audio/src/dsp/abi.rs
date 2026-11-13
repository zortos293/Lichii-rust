//! Guest-visible layouts the DSP consumes: effect parameters (the `specific` bytes of an effect's
//! input parameter), voice biquad parameters, and the aux/capture and limiter statistics blocks.
//!
//! Every struct is `#[repr(C)]` plain data with explicit padding, so it can be read from the raw
//! parameter bytes with [`parse`]. Booleans and enums are stored as their guest bytes; use the
//! accessor methods for typed views. Where version 1 and version 2 of a parameter share a layout,
//! the version 2 name is an alias.

use std::mem::{offset_of, size_of};

use bytemuck::{Pod, Zeroable};

/// Reads a `T` from the start of `bytes`, or `None` if `bytes` is too short.
pub fn parse<T: Pod>(bytes: &[u8]) -> Option<T> {
    bytes
        .get(..size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
}

/// Per-update state of an effect's parameters (`EffectInfoBase::ParameterState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterState {
    Initialized,
    Updating,
    Updated,
}

impl ParameterState {
    pub fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Initialized),
            1 => Some(Self::Updating),
            2 => Some(Self::Updated),
            _ => None,
        }
    }

    pub fn raw(self) -> u8 {
        self as u8
    }
}

macro_rules! parameter_state_accessor {
    ($($ty:ty),*) => {$(
        impl $ty {
            pub fn parameter_state(&self) -> Option<ParameterState> {
                ParameterState::from_raw(self.state)
            }
        }
    )*};
}

/// `BiquadFilterInfo::ParameterVersion1`: Q2.14 coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct BiquadFilterParameterV1 {
    pub inputs: [i8; 6],
    pub outputs: [i8; 6],
    pub b: [i16; 3],
    pub a: [i16; 2],
    pub channel_count: i8,
    pub state: u8,
}

/// `BiquadFilterInfo::ParameterVersion2`: float coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct BiquadFilterParameterV2 {
    pub inputs: [i8; 6],
    pub outputs: [i8; 6],
    pub padding: u32,
    pub b: [f32; 3],
    pub a: [f32; 2],
    pub channel_count: i8,
    pub state: u8,
    pub reserved: u16,
}

/// `DelayInfo::ParameterVersion1`. The rate and gain fields are `FixedPoint<18, 14>` raw values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct DelayParameterV1 {
    pub inputs: [i8; 6],
    pub outputs: [i8; 6],
    pub channel_count_max: u16,
    pub channel_count: u16,
    pub delay_time_max: u32,
    pub delay_time: u32,
    pub sample_rate: i32,
    pub in_gain: i32,
    pub feedback_gain: i32,
    pub wet_gain: i32,
    pub dry_gain: i32,
    pub channel_spread: i32,
    pub lowpass_amount: i32,
    pub state: u8,
    pub padding: [u8; 3],
}

/// `DelayInfo::ParameterVersion2` has the version 1 layout.
pub type DelayParameterV2 = DelayParameterV1;

/// `ReverbInfo::ParameterVersion1`. Gains and times are `FixedPoint<50, 14>`-scaled integers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct ReverbParameterV1 {
    pub inputs: [i8; 6],
    pub outputs: [i8; 6],
    pub channel_count_max: u16,
    pub channel_count: u16,
    pub sample_rate: u32,
    pub early_mode: u32,
    pub early_gain: i32,
    pub pre_delay: i32,
    pub late_mode: i32,
    pub late_gain: i32,
    pub decay_time: i32,
    pub high_freq_decay_ratio: i32,
    pub colouration: i32,
    pub base_gain: i32,
    pub wet_gain: i32,
    pub dry_gain: i32,
    pub state: u8,
    pub padding: [u8; 3],
}

/// `ReverbInfo::ParameterVersion2` has the version 1 layout.
pub type ReverbParameterV2 = ReverbParameterV1;

/// `I3dl2ReverbInfo::ParameterVersion1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct I3dl2ReverbParameterV1 {
    pub inputs: [i8; 6],
    pub outputs: [i8; 6],
    pub channel_count_max: u16,
    pub channel_count: u16,
    pub unk10: [u8; 4],
    pub sample_rate: u32,
    pub room_hf_gain: f32,
    pub reference_hf: f32,
    pub late_reverb_decay_time: f32,
    pub late_reverb_hf_decay_ratio: f32,
    pub room_gain: f32,
    pub reflection_gain: f32,
    pub reverb_gain: f32,
    pub late_reverb_diffusion: f32,
    pub reflection_delay: f32,
    pub late_reverb_delay_time: f32,
    pub late_reverb_density: f32,
    pub dry_gain: f32,
    pub state: u8,
    pub unk49: [u8; 3],
}

/// `I3dl2ReverbInfo::ParameterVersion2` has the version 1 layout.
pub type I3dl2ReverbParameterV2 = I3dl2ReverbParameterV1;

/// `AuxInfo::ParameterVersion1`, also the capture effect's parameter.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct AuxParameterV1 {
    pub inputs: [i8; 24],
    pub outputs: [i8; 24],
    pub mix_buffer_count: u32,
    pub sample_rate: u32,
    pub count_max: u32,
    pub mix_buffer_count_max: u32,
    pub send_buffer_info_address: u64,
    pub send_buffer_address: u64,
    pub return_buffer_info_address: u64,
    pub return_buffer_address: u64,
    pub mix_buffer_sample_size: u32,
    pub sample_count: u32,
    pub mix_buffer_sample_count: u32,
    pub padding: u32,
}

/// `AuxInfo::ParameterVersion2` has the version 1 layout.
pub type AuxParameterV2 = AuxParameterV1;
/// The capture effect reuses the aux parameter layout.
pub type CaptureParameterV1 = AuxParameterV1;
/// The capture effect reuses the aux parameter layout.
pub type CaptureParameterV2 = AuxParameterV1;

/// `LightLimiterInfo::ProcessingMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimiterProcessingMode {
    Mode0,
    Mode1,
}

/// `LightLimiterInfo::ParameterVersion1`.
///
/// `processing_mode` is the byte at 0x43 as the documented layout has it; the C++ struct
/// declares an int-sized enum there, which the compiler pushes to 0x44.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct LightLimiterParameterV1 {
    pub inputs: [i8; 6],
    pub outputs: [i8; 6],
    pub channel_count_max: u16,
    pub channel_count: u16,
    pub sample_rate: u32,
    pub look_ahead_time_max: i32,
    pub attack_time: i32,
    pub release_time: i32,
    pub look_ahead_time: i32,
    pub attack_coeff: f32,
    pub release_coeff: f32,
    pub threshold: f32,
    pub input_gain: f32,
    pub output_gain: f32,
    pub look_ahead_samples_min: i32,
    pub look_ahead_samples_max: i32,
    pub state: u8,
    pub statistics_enabled: u8,
    pub statistics_reset_required: u8,
    pub processing_mode: u8,
}

impl LightLimiterParameterV1 {
    pub fn mode(&self) -> LimiterProcessingMode {
        if self.processing_mode == 1 {
            LimiterProcessingMode::Mode1
        } else {
            LimiterProcessingMode::Mode0
        }
    }
}

/// `LightLimiterInfo::ParameterVersion2` has the version 1 layout.
pub type LightLimiterParameterV2 = LightLimiterParameterV1;

/// `CompressorInfo::ParameterVersion1`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct CompressorParameterV1 {
    pub inputs: [i8; 6],
    pub outputs: [i8; 6],
    pub channel_count_max: i16,
    pub channel_count: i16,
    pub sample_rate: i32,
    pub threshold: f32,
    pub compressor_ratio: f32,
    pub attack_time: i32,
    pub release_time: i32,
    pub unk_24: f32,
    pub unk_28: f32,
    pub unk_2c: f32,
    pub out_gain: f32,
    pub state: u8,
    pub makeup_gain_enabled: u8,
    pub padding: [u8; 2],
}

/// `CompressorInfo::ParameterVersion2` has the version 1 layout.
pub type CompressorParameterV2 = CompressorParameterV1;

parameter_state_accessor!(
    BiquadFilterParameterV1,
    BiquadFilterParameterV2,
    DelayParameterV1,
    ReverbParameterV1,
    I3dl2ReverbParameterV1,
    LightLimiterParameterV1,
    CompressorParameterV1
);

/// `VoiceInfo::BiquadFilterParameter`: one voice biquad stage with Q2.14 coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct VoiceBiquadParameter {
    pub enabled: u8,
    pub padding: u8,
    pub b: [i16; 3],
    pub a: [i16; 2],
}

/// `VoiceInfo::BiquadFilterParameter2`: one voice biquad stage with float coefficients.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct VoiceBiquadParameterFloat {
    pub enabled: u8,
    pub reserved: [u8; 3],
    pub numerator: [f32; 3],
    pub denominator: [f32; 2],
}

/// `AuxInfo::AuxInfoDsp`: one side of an aux ring buffer's bookkeeping.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct AuxInfoDsp {
    pub read_offset: u32,
    pub write_offset: u32,
    pub lost_sample_count: u32,
    pub total_sample_count: u32,
    pub unk10: [u8; 0x30],
}

/// `AuxInfo::AuxBufferInfo`: the CPU and DSP sides of a capture ring buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct AuxBufferInfo {
    pub cpu_info: AuxInfoDsp,
    pub dsp_info: AuxInfoDsp,
}

/// `LightLimiterInfo::StatisticsInternal`, stored at the start of the effect's result state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
#[repr(C)]
pub struct LimiterStatistics {
    pub channel_max_sample: [f32; 6],
    pub channel_compression_gain_min: [f32; 6],
}

/// `EffectResultState`: the per-effect block the DSP writes and the renderer reports back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct EffectResultState {
    pub state: [u8; 0x80],
}

impl Default for AuxInfoDsp {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Default for AuxBufferInfo {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Default for EffectResultState {
    fn default() -> Self {
        Self::zeroed()
    }
}

const _: () = {
    assert!(size_of::<BiquadFilterParameterV1>() == 0x18);
    assert!(offset_of!(BiquadFilterParameterV1, b) == 0x0C);
    assert!(offset_of!(BiquadFilterParameterV1, a) == 0x12);
    assert!(offset_of!(BiquadFilterParameterV1, state) == 0x17);
    assert!(size_of::<BiquadFilterParameterV2>() == 0x28);
    assert!(offset_of!(BiquadFilterParameterV2, b) == 0x10);
    assert!(offset_of!(BiquadFilterParameterV2, a) == 0x1C);
    assert!(offset_of!(BiquadFilterParameterV2, state) == 0x25);
    assert!(size_of::<DelayParameterV1>() == 0x38);
    assert!(offset_of!(DelayParameterV1, sample_rate) == 0x18);
    assert!(offset_of!(DelayParameterV1, lowpass_amount) == 0x30);
    assert!(offset_of!(DelayParameterV1, state) == 0x34);
    assert!(size_of::<ReverbParameterV1>() == 0x44);
    assert!(offset_of!(ReverbParameterV1, sample_rate) == 0x10);
    assert!(offset_of!(ReverbParameterV1, dry_gain) == 0x3C);
    assert!(offset_of!(ReverbParameterV1, state) == 0x40);
    assert!(size_of::<I3dl2ReverbParameterV1>() == 0x4C);
    assert!(offset_of!(I3dl2ReverbParameterV1, sample_rate) == 0x14);
    assert!(offset_of!(I3dl2ReverbParameterV1, dry_gain) == 0x44);
    assert!(offset_of!(I3dl2ReverbParameterV1, state) == 0x48);
    assert!(size_of::<AuxParameterV1>() == 0x70);
    assert!(offset_of!(AuxParameterV1, mix_buffer_count) == 0x30);
    assert!(offset_of!(AuxParameterV1, send_buffer_info_address) == 0x40);
    assert!(offset_of!(AuxParameterV1, return_buffer_address) == 0x58);
    assert!(offset_of!(AuxParameterV1, mix_buffer_sample_count) == 0x68);
    assert!(size_of::<LightLimiterParameterV1>() == 0x44);
    assert!(offset_of!(LightLimiterParameterV1, sample_rate) == 0x10);
    assert!(offset_of!(LightLimiterParameterV1, attack_coeff) == 0x24);
    assert!(offset_of!(LightLimiterParameterV1, look_ahead_samples_max) == 0x3C);
    assert!(offset_of!(LightLimiterParameterV1, state) == 0x40);
    assert!(offset_of!(LightLimiterParameterV1, processing_mode) == 0x43);
    assert!(size_of::<CompressorParameterV1>() == 0x38);
    assert!(offset_of!(CompressorParameterV1, unk_24) == 0x24);
    assert!(offset_of!(CompressorParameterV1, out_gain) == 0x30);
    assert!(offset_of!(CompressorParameterV1, state) == 0x34);
    assert!(offset_of!(CompressorParameterV1, makeup_gain_enabled) == 0x35);
    assert!(size_of::<VoiceBiquadParameter>() == 0x0C);
    assert!(offset_of!(VoiceBiquadParameter, b) == 0x02);
    assert!(offset_of!(VoiceBiquadParameter, a) == 0x08);
    assert!(size_of::<VoiceBiquadParameterFloat>() == 0x18);
    assert!(offset_of!(VoiceBiquadParameterFloat, numerator) == 0x04);
    assert!(offset_of!(VoiceBiquadParameterFloat, denominator) == 0x10);
    assert!(size_of::<AuxInfoDsp>() == 0x40);
    assert!(size_of::<AuxBufferInfo>() == 0x80);
    assert!(size_of::<LimiterStatistics>() == 0x30);
    assert!(size_of::<EffectResultState>() == 0x80);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_delay_parameter_from_specific_bytes() {
        let mut bytes = [0u8; 0xA0];
        bytes[0x0E..0x10].copy_from_slice(&2u16.to_le_bytes());
        bytes[0x14..0x18].copy_from_slice(&250u32.to_le_bytes());
        bytes[0x20..0x24].copy_from_slice(&(1i32 << 13).to_le_bytes());
        bytes[0x34] = 1;
        let parameter: DelayParameterV1 = parse(&bytes).unwrap();
        assert_eq!(parameter.channel_count, 2);
        assert_eq!(parameter.delay_time, 250);
        assert_eq!(parameter.feedback_gain, 8192);
        assert_eq!(parameter.parameter_state(), Some(ParameterState::Updating));
        assert!(parse::<DelayParameterV1>(&bytes[..0x37]).is_none());
    }
}
