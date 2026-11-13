//! DSP-owned state that outlives a single frame, and the handles commands use to reach it.
//!
//! The renderer sizes a [`DspState`] when it opens a session and keeps it next to its voice,
//! mix and effect bookkeeping. The command generator refers to entries by index, so commands
//! carry no borrows; the processor resolves handles while it runs a command list.

use super::abi::EffectResultState;
use super::biquad::BiquadFilterState;
use super::compressor::CompressorState;
use super::delay::DelayState;
use super::fixed::Q15;
use super::i3dl2::I3dl2State;
use super::limiter::LightLimiterState;
use super::reverb::ReverbState;
use super::upsample::UpsamplerInfo;
use super::{MAX_BIQUAD_FILTERS, MAX_CHANNELS, MAX_MIX_BUFFERS, MAX_WAVE_BUFFERS};

/// Index of a [`VoiceState`] in [`DspState::voices`] (one per voice channel resource).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VoiceStateId(pub u32);

/// Index of an [`EffectState`] in [`DspState::effects`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EffectStateId(pub u32);

/// Index of an [`EffectResultState`] in [`DspState::effect_results`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct EffectResultId(pub u32);

/// Index of an [`UpsamplerInfo`] in [`DspState::upsamplers`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct UpsamplerId(pub u32);

/// The ADPCM decoder's predictor state, in the guest's layout (`VoiceState::AdpcmContext`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdpcmContext {
    pub header: u16,
    pub yn0: i16,
    pub yn1: i16,
}

impl AdpcmContext {
    /// Size of the context as the guest stores it next to a wave buffer.
    pub const GUEST_SIZE: usize = 6;

    pub fn from_guest(bytes: [u8; Self::GUEST_SIZE]) -> Self {
        Self {
            header: u16::from_le_bytes([bytes[0], bytes[1]]),
            yn0: i16::from_le_bytes([bytes[2], bytes[3]]),
            yn1: i16::from_le_bytes([bytes[4], bytes[5]]),
        }
    }
}

/// Playback state of one voice channel, read and written by the data source, biquad and mix
/// commands and reported back by the renderer (`VoiceState`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoiceState {
    pub played_sample_count: u64,
    pub offset: u32,
    pub wave_buffer_index: u32,
    pub wave_buffer_valid: [bool; MAX_WAVE_BUFFERS],
    pub wave_buffers_consumed: u32,
    pub sample_history: [i16; MAX_WAVE_BUFFERS * 2],
    pub fraction: Q15,
    pub adpcm_context: AdpcmContext,
    pub biquad_states: [[BiquadFilterState; MAX_BIQUAD_FILTERS]; MAX_BIQUAD_FILTERS],
    pub previous_samples: [i32; MAX_MIX_BUFFERS],
    pub external_context_size: u32,
    pub external_context_enabled: bool,
    pub voice_dropped: bool,
    pub loop_count: i32,
}

/// Per-effect DSP state, created zeroed and shaped by the first command that uses it.
#[derive(Clone, Debug, Default)]
pub enum EffectState {
    #[default]
    Empty,
    BiquadFilter([BiquadFilterState; MAX_CHANNELS]),
    Delay(Box<DelayState>),
    Reverb(Box<ReverbState>),
    I3dl2Reverb(Box<I3dl2State>),
    LightLimiter(Box<LightLimiterState>),
    Compressor(CompressorState),
}

macro_rules! effect_state_accessor {
    ($name:ident, $variant:ident, $ty:ty, $init:expr) => {
        /// Returns this effect's state, replacing any other kind with a zeroed one.
        pub fn $name(&mut self) -> &mut $ty {
            if !matches!(self, Self::$variant(_)) {
                *self = Self::$variant($init);
            }
            match self {
                Self::$variant(state) => state,
                _ => unreachable!(),
            }
        }
    };
}

impl EffectState {
    effect_state_accessor!(
        biquad_filter,
        BiquadFilter,
        [BiquadFilterState; MAX_CHANNELS],
        Default::default()
    );
    effect_state_accessor!(delay, Delay, DelayState, Box::default());
    effect_state_accessor!(reverb, Reverb, ReverbState, Box::default());
    effect_state_accessor!(i3dl2_reverb, I3dl2Reverb, I3dl2State, Box::default());
    effect_state_accessor!(
        light_limiter,
        LightLimiter,
        LightLimiterState,
        Box::default()
    );
    effect_state_accessor!(compressor, Compressor, CompressorState, Default::default());
}

/// All state the command processor reads and writes across frames for one renderer session.
#[derive(Clone, Debug, Default)]
pub struct DspState {
    pub voices: Vec<VoiceState>,
    /// Depop carry per mix buffer, fed by depop prepare and drained by depop for mix buffers.
    pub depop: Vec<i32>,
    pub upsamplers: Vec<UpsamplerInfo>,
    pub effects: Vec<EffectState>,
    pub effect_results: Vec<EffectResultState>,
    /// The performance frame buffer that performance commands stamp times into.
    pub performance: Vec<u8>,
}

impl DspState {
    /// Allocates zeroed state for the given resource counts.
    pub fn new(voices: usize, mix_buffers: usize, upsamplers: usize, effects: usize) -> Self {
        Self {
            voices: vec![VoiceState::default(); voices],
            depop: vec![0; mix_buffers],
            upsamplers: vec![UpsamplerInfo::default(); upsamplers],
            effects: vec![EffectState::default(); effects],
            effect_results: vec![EffectResultState::default(); effects],
            performance: Vec::new(),
        }
    }
}
