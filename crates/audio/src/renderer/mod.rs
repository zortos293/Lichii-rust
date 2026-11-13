//! Audio renderer (audren) state.
//!
//! A game describes its audio graph (voices, mixes, effects, sinks, splitters and the memory
//! pools that back their buffers) through `RequestUpdate`. This module keeps that state, parses
//! every input section, writes every output section and computes the work buffer size the game
//! must allocate. Turning the state into DSP commands is the command generator's job; it walks
//! [`system::RendererSystem`] through the accessors each context exposes.

pub mod behavior;
pub mod device;
pub mod effect;
pub mod memory;
pub mod mix;
pub mod nodes;
pub mod parameter;
pub mod performance;
pub mod result;
pub mod revision;
pub mod sink;
pub mod splitter;
pub mod system;
pub mod updater;
pub mod upsampler;
pub mod voice;
pub mod workbuffer;

pub use behavior::{BehaviorInfo, ErrorInfo};
pub use parameter::{AudioRendererParameterInternal, ExecutionMode};
pub use system::RendererSystem;

/// Number of mix buffers a sink or effect can address.
pub const MAX_MIX_BUFFERS: usize = 24;
/// Channels a voice, sink or effect can carry.
pub const MAX_CHANNELS: usize = 6;
/// Wave buffers queued per voice.
pub const MAX_WAVE_BUFFERS: usize = 4;
/// Biquad filters per voice or splitter destination.
pub const MAX_BIQUAD_FILTERS: usize = 2;
/// Effects a single mix can chain.
pub const MAX_EFFECTS: usize = 256;
/// Samples per rendered frame at the target rate.
pub const TARGET_SAMPLE_COUNT: u32 = 240;
/// Output rate of the renderer.
pub const TARGET_SAMPLE_RATE: u32 = 48_000;
/// Lowest voice priority; voices with this value are dropped first.
pub const LOWEST_VOICE_PRIORITY: i32 = 0xFF;
/// Highest voice priority; these voices are never dropped.
pub const HIGHEST_VOICE_PRIORITY: i32 = 0;
/// Id of the final mix.
pub const FINAL_MIX_ID: i32 = 0;
/// Distance of a mix that does not reach the final mix.
pub const INVALID_DISTANCE_FROM_FINAL_MIX: i32 = i32::MIN;
/// Splitter id of a voice or mix without a splitter.
pub const UNUSED_SPLITTER_ID: i32 = -1;
/// Mix id of a voice, mix or destination that is not connected.
pub const UNUSED_MIX_ID: i32 = i32::MAX;
/// Process order of an effect that is not placed in a mix.
pub const INVALID_PROCESS_ORDER: i32 = -1;

/// Little-endian four character code, as `Common::MakeMagic` builds it.
pub const fn make_magic(a: u8, b: u8, c: u8, d: u8) -> u32 {
    a as u32 | (b as u32) << 8 | (c as u32) << 16 | (d as u32) << 24
}

/// Voice play state requested by the game.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum PlayState {
    #[default]
    Started = 0,
    Stopped = 1,
    Paused = 2,
}

impl PlayState {
    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Started),
            1 => Some(Self::Stopped),
            2 => Some(Self::Paused),
            _ => None,
        }
    }
}

/// Sample rate conversion quality of a voice.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SrcQuality {
    #[default]
    Medium = 0,
    High = 1,
    Low = 2,
}

impl SrcQuality {
    pub const fn from_raw(raw: u8) -> Option<Self> {
        match raw {
            0 => Some(Self::Medium),
            1 => Some(Self::High),
            2 => Some(Self::Low),
            _ => None,
        }
    }
}

/// Sample format of wave buffers and circular buffer sinks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SampleFormat {
    #[default]
    Invalid = 0,
    PcmInt8 = 1,
    PcmInt16 = 2,
    PcmInt24 = 3,
    PcmInt32 = 4,
    PcmFloat = 5,
    Adpcm = 6,
}

impl SampleFormat {
    /// Unknown values decode as `Invalid`, which every consumer treats as unsupported.
    pub const fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::PcmInt8,
            2 => Self::PcmInt16,
            3 => Self::PcmInt24,
            4 => Self::PcmInt32,
            5 => Self::PcmFloat,
            6 => Self::Adpcm,
            _ => Self::Invalid,
        }
    }

    /// Bytes per sample; formats without a fixed size count as 16-bit like the reference.
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::PcmInt8 => 1,
            Self::PcmInt16 => 2,
            Self::PcmInt24 => 3,
            Self::PcmInt32 | Self::PcmFloat => 4,
            _ => 2,
        }
    }
}

/// Channel counts an effect accepts: 1, 2, 4 or 6.
pub const fn is_channel_count_valid(channel_count: u16) -> bool {
    matches!(channel_count, 1 | 2 | 4 | 6)
}
