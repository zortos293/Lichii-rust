//! The command list the renderer hands to the DSP each frame (`commands.h`).
//!
//! Commands name mix buffers by index and DSP state by handle, so a command list is plain data
//! that the generator can build without borrowing renderer state.

use super::abi::{
    CompressorParameterV2, DelayParameterV1, I3dl2ReverbParameterV1, LightLimiterParameterV2,
    ReverbParameterV2, VoiceBiquadParameter, VoiceBiquadParameterFloat,
};
use super::decode::WaveBuffer;
use super::fixed::{Fx, Q15};
use super::resample::SrcQuality;
use super::state::{EffectResultId, EffectStateId, UpsamplerId, VoiceStateId};
use super::{MAX_BIQUAD_FILTERS, MAX_CHANNELS, MAX_MIX_BUFFERS, MAX_WAVE_BUFFERS};

/// The command type byte (`CommandId`), used by the processing time estimator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CommandId {
    Invalid = 0x00,
    DataSourcePcmInt16Version1 = 0x01,
    DataSourcePcmInt16Version2 = 0x02,
    DataSourcePcmFloatVersion1 = 0x03,
    DataSourcePcmFloatVersion2 = 0x04,
    DataSourceAdpcmVersion1 = 0x05,
    DataSourceAdpcmVersion2 = 0x06,
    Volume = 0x07,
    VolumeRamp = 0x08,
    BiquadFilter = 0x09,
    Mix = 0x0A,
    MixRamp = 0x0B,
    MixRampGrouped = 0x0C,
    DepopPrepare = 0x0D,
    DepopForMixBuffers = 0x0E,
    Delay = 0x0F,
    Upsample = 0x10,
    DownMix6chTo2ch = 0x11,
    Aux = 0x12,
    DeviceSink = 0x13,
    CircularBufferSink = 0x14,
    Reverb = 0x15,
    I3dl2Reverb = 0x16,
    Performance = 0x17,
    ClearMixBuffer = 0x18,
    CopyMixBuffer = 0x19,
    LightLimiterVersion1 = 0x1A,
    LightLimiterVersion2 = 0x1B,
    MultiTapBiquadFilter = 0x1C,
    Capture = 0x1D,
    Compressor = 0x1E,
}

/// One frame's worth of commands plus the frame format they render.
#[derive(Clone, Debug, Default)]
pub struct CommandList {
    pub sample_count: u32,
    pub sample_rate: u32,
    pub buffer_count: u32,
    pub commands: Vec<CommandEntry>,
}

/// A command with the header fields every C++ command carries.
#[derive(Clone, Debug)]
pub struct CommandEntry {
    pub enabled: bool,
    /// Node id of the voice, mix or sink the command was generated for.
    pub node_id: u32,
    pub estimated_process_time: u32,
    pub command: Command,
}

impl CommandEntry {
    pub fn new(node_id: u32, command: Command) -> Self {
        Self {
            enabled: true,
            node_id,
            estimated_process_time: 0,
            command,
        }
    }
}

/// Every command the audio renderer DSP executes.
#[derive(Clone, Debug)]
pub enum Command {
    DataSourcePcmInt16Version1(DataSourceCommand),
    DataSourcePcmInt16Version2(DataSourceCommand),
    DataSourcePcmFloatVersion1(DataSourceCommand),
    DataSourcePcmFloatVersion2(DataSourceCommand),
    DataSourceAdpcmVersion1(DataSourceCommand),
    DataSourceAdpcmVersion2(DataSourceCommand),
    Volume(VolumeCommand),
    VolumeRamp(VolumeRampCommand),
    BiquadFilter(BiquadFilterCommand),
    Mix(MixCommand),
    MixRamp(MixRampCommand),
    MixRampGrouped(Box<MixRampGroupedCommand>),
    DepopPrepare(DepopPrepareCommand),
    DepopForMixBuffers(DepopForMixBuffersCommand),
    Delay(DelayCommand),
    Upsample(UpsampleCommand),
    DownMix6chTo2ch(DownMix6chTo2chCommand),
    Aux(AuxCommand),
    DeviceSink(DeviceSinkCommand),
    CircularBufferSink(CircularBufferSinkCommand),
    Reverb(ReverbCommand),
    I3dl2Reverb(I3dl2ReverbCommand),
    Performance(PerformanceCommand),
    ClearMixBuffer,
    CopyMixBuffer(CopyMixBufferCommand),
    LightLimiterVersion1(LightLimiterCommand),
    LightLimiterVersion2(LightLimiterCommand),
    MultiTapBiquadFilter(MultiTapBiquadFilterCommand),
    Capture(CaptureCommand),
    Compressor(CompressorCommand),
}

impl Command {
    pub fn id(&self) -> CommandId {
        match self {
            Self::DataSourcePcmInt16Version1(_) => CommandId::DataSourcePcmInt16Version1,
            Self::DataSourcePcmInt16Version2(_) => CommandId::DataSourcePcmInt16Version2,
            Self::DataSourcePcmFloatVersion1(_) => CommandId::DataSourcePcmFloatVersion1,
            Self::DataSourcePcmFloatVersion2(_) => CommandId::DataSourcePcmFloatVersion2,
            Self::DataSourceAdpcmVersion1(_) => CommandId::DataSourceAdpcmVersion1,
            Self::DataSourceAdpcmVersion2(_) => CommandId::DataSourceAdpcmVersion2,
            Self::Volume(_) => CommandId::Volume,
            Self::VolumeRamp(_) => CommandId::VolumeRamp,
            Self::BiquadFilter(_) => CommandId::BiquadFilter,
            Self::Mix(_) => CommandId::Mix,
            Self::MixRamp(_) => CommandId::MixRamp,
            Self::MixRampGrouped(_) => CommandId::MixRampGrouped,
            Self::DepopPrepare(_) => CommandId::DepopPrepare,
            Self::DepopForMixBuffers(_) => CommandId::DepopForMixBuffers,
            Self::Delay(_) => CommandId::Delay,
            Self::Upsample(_) => CommandId::Upsample,
            Self::DownMix6chTo2ch(_) => CommandId::DownMix6chTo2ch,
            Self::Aux(_) => CommandId::Aux,
            Self::DeviceSink(_) => CommandId::DeviceSink,
            Self::CircularBufferSink(_) => CommandId::CircularBufferSink,
            Self::Reverb(_) => CommandId::Reverb,
            Self::I3dl2Reverb(_) => CommandId::I3dl2Reverb,
            Self::Performance(_) => CommandId::Performance,
            Self::ClearMixBuffer => CommandId::ClearMixBuffer,
            Self::CopyMixBuffer(_) => CommandId::CopyMixBuffer,
            Self::LightLimiterVersion1(_) => CommandId::LightLimiterVersion1,
            Self::LightLimiterVersion2(_) => CommandId::LightLimiterVersion2,
            Self::MultiTapBiquadFilter(_) => CommandId::MultiTapBiquadFilter,
            Self::Capture(_) => CommandId::Capture,
            Self::Compressor(_) => CommandId::Compressor,
        }
    }
}

/// Decodes a voice's wave buffers into one mix buffer, resampling to the frame rate.
///
/// Version 1 commands treat every looping wave buffer as an infinite whole-buffer loop; version 2
/// honours each buffer's loop range and count. ADPCM always decodes channel 0 of 1.
#[derive(Clone, Debug, Default)]
pub struct DataSourceCommand {
    pub src_quality: SrcQuality,
    pub output_index: i16,
    /// [`DataSourceCommand::FLAG_RESET_PLAYED_AT_LOOP`] and
    /// [`DataSourceCommand::FLAG_SKIP_PITCH_AND_SRC`].
    pub flags: u16,
    pub sample_rate: u32,
    pub pitch: f32,
    pub channel_index: i8,
    pub channel_count: i8,
    pub wave_buffers: [WaveBuffer; MAX_WAVE_BUFFERS],
    pub voice_state: VoiceStateId,
    /// Guest address of the ADPCM coefficient table.
    pub data_address: u64,
    pub data_size: u64,
}

impl DataSourceCommand {
    pub const FLAG_RESET_PLAYED_AT_LOOP: u16 = 1 << 0;
    pub const FLAG_SKIP_PITCH_AND_SRC: u16 = 1 << 1;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VolumeCommand {
    pub precision: u8,
    pub input_index: i16,
    pub output_index: i16,
    pub volume: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VolumeRampCommand {
    pub precision: u8,
    pub input_index: i16,
    pub output_index: i16,
    pub prev_volume: f32,
    pub volume: f32,
}

/// Where a biquad stage keeps its delay elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiquadStateRef {
    /// `VoiceState::biquad_states[filter][0]` of a voice.
    Voice { voice: VoiceStateId, filter: u8 },
    /// Channel `channel` of a biquad filter effect's state.
    Effect { effect: EffectStateId, channel: u8 },
}

impl Default for BiquadStateRef {
    fn default() -> Self {
        Self::Voice {
            voice: VoiceStateId(0),
            filter: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BiquadFilterCommand {
    pub input: i16,
    pub output: i16,
    pub biquad: VoiceBiquadParameter,
    pub biquad_float: VoiceBiquadParameterFloat,
    pub state: BiquadStateRef,
    pub needs_init: bool,
    pub use_float_processing: bool,
    pub use_float_coefficients: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MultiTapBiquadFilterCommand {
    pub input: i16,
    pub output: i16,
    pub biquads: [VoiceBiquadParameter; MAX_BIQUAD_FILTERS],
    pub biquads_float: [VoiceBiquadParameterFloat; MAX_BIQUAD_FILTERS],
    pub states: [BiquadStateRef; MAX_BIQUAD_FILTERS],
    pub needs_init: [bool; MAX_BIQUAD_FILTERS],
    pub filter_tap_count: u8,
    pub use_float_coefficients: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MixCommand {
    pub precision: u8,
    pub input_index: i16,
    pub output_index: i16,
    pub volume: f32,
}

/// `VoiceState::previous_samples[index]` of a voice, where mix ramps leave their last sample.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreviousSampleRef {
    pub voice: VoiceStateId,
    pub index: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MixRampCommand {
    pub precision: u8,
    pub input_index: i16,
    pub output_index: i16,
    pub prev_volume: f32,
    pub volume: f32,
    pub previous_sample: PreviousSampleRef,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MixRampGroupedCommand {
    pub precision: u8,
    pub buffer_count: u32,
    pub inputs: [i16; MAX_MIX_BUFFERS],
    pub outputs: [i16; MAX_MIX_BUFFERS],
    pub prev_volumes: [f32; MAX_MIX_BUFFERS],
    pub volumes: [f32; MAX_MIX_BUFFERS],
    /// The voice whose `previous_samples` receive each buffer's last sample.
    pub previous_samples: VoiceStateId,
}

/// Moves a voice's leftover samples into [`DspState::depop`](super::DspState::depop).
#[derive(Clone, Copy, Debug, Default)]
pub struct DepopPrepareCommand {
    pub inputs: [i16; MAX_MIX_BUFFERS],
    pub previous_samples: VoiceStateId,
    pub buffer_count: u32,
}

/// Fades the depop carry of mix buffers `input..input + count` into those buffers.
#[derive(Clone, Copy, Debug, Default)]
pub struct DepopForMixBuffersCommand {
    pub input: u32,
    pub count: u32,
    pub decay: Q15,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DelayCommand {
    pub inputs: [i16; MAX_CHANNELS],
    pub outputs: [i16; MAX_CHANNELS],
    pub parameter: DelayParameterV1,
    pub state: EffectStateId,
    pub effect_enabled: bool,
}

/// Upsamples the inputs listed in an [`UpsamplerInfo`](super::UpsamplerInfo) into its sample
/// buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct UpsampleCommand {
    pub buffer_count: u32,
    pub unk_20: u32,
    pub source_sample_count: u32,
    pub source_sample_rate: u32,
    pub upsampler: UpsamplerId,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DownMix6chTo2chCommand {
    pub inputs: [i16; MAX_CHANNELS],
    pub outputs: [i16; MAX_CHANNELS],
    /// `FixedPoint<48, 16>` coefficients for front, center, LFE and back.
    pub down_mix_coeff: [Fx<16>; 4],
}

/// Sends a mix buffer to a guest ring buffer and reads the processed samples back.
#[derive(Clone, Copy, Debug, Default)]
pub struct AuxCommand {
    pub input: i16,
    pub output: i16,
    pub send_buffer_info: u64,
    pub return_buffer_info: u64,
    pub send_buffer: u64,
    pub return_buffer: u64,
    pub count_max: u32,
    pub write_offset: u32,
    pub update_count: u32,
    pub effect_enabled: bool,
}

/// Copies a mix buffer into a guest capture ring buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureCommand {
    pub input: i16,
    pub output: i16,
    pub send_buffer_info: u64,
    pub send_buffer: u64,
    pub count_max: u32,
    pub write_offset: u32,
    pub update_count: u32,
    pub effect_enabled: bool,
}

/// Where a device sink reads its channels from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SinkSource {
    /// The frame's mix buffers.
    #[default]
    MixBuffers,
    /// The sample buffer of an upsampler that converted the mix to 48 kHz.
    Upsampler(UpsamplerId),
}

/// Interleaves up to six buffers into the caller's device output as saturated i16.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceSinkCommand {
    pub session_id: i32,
    pub source: SinkSource,
    pub input_count: u32,
    pub inputs: [i16; MAX_CHANNELS],
}

/// Writes saturated i16 channel blocks into a guest ring buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct CircularBufferSinkCommand {
    pub input_count: u32,
    pub inputs: [i16; MAX_CHANNELS],
    pub address: u64,
    pub size: u32,
    pub pos: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ReverbCommand {
    pub inputs: [i16; MAX_CHANNELS],
    pub outputs: [i16; MAX_CHANNELS],
    pub parameter: ReverbParameterV2,
    pub state: EffectStateId,
    pub effect_enabled: bool,
    pub long_size_pre_delay_supported: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct I3dl2ReverbCommand {
    pub inputs: [i16; MAX_CHANNELS],
    pub outputs: [i16; MAX_CHANNELS],
    pub parameter: I3dl2ReverbParameterV1,
    pub state: EffectStateId,
    pub effect_enabled: bool,
}

/// Which edge of a performance entry a performance command stamps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PerformanceState {
    #[default]
    Invalid,
    Start,
    Stop,
}

/// Byte offsets into [`DspState::performance`](super::DspState::performance) of one entry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerformanceEntryAddresses {
    pub translated_address: u32,
    pub entry_start_time_offset: u32,
    pub header_entry_count_offset: u32,
    pub entry_processed_time_offset: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PerformanceCommand {
    pub state: PerformanceState,
    pub entry_address: PerformanceEntryAddresses,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CopyMixBufferCommand {
    pub input_index: i16,
    pub output_index: i16,
}

/// Light limiter; version 2 also records statistics into `result_state` when it is set.
#[derive(Clone, Copy, Debug, Default)]
pub struct LightLimiterCommand {
    pub inputs: [i16; MAX_CHANNELS],
    pub outputs: [i16; MAX_CHANNELS],
    pub parameter: LightLimiterParameterV2,
    pub state: EffectStateId,
    pub result_state: Option<EffectResultId>,
    pub effect_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CompressorCommand {
    pub inputs: [i16; MAX_CHANNELS],
    pub outputs: [i16; MAX_CHANNELS],
    pub parameter: CompressorParameterV2,
    pub state: EffectStateId,
    pub effect_enabled: bool,
}
