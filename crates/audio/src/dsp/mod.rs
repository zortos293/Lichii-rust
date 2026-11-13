//! The audio renderer's DSP: executes the command lists the renderer generates each frame.
//!
//! The renderer owns a [`DspState`] per session and a [`CommandListProcessor`]. Each frame it
//! builds a [`CommandList`] whose commands name mix buffers by index and DSP state by handle,
//! then calls [`CommandListProcessor::process`] with the mix buffers, guest memory (through
//! [`DspMemory`]), a clock and a device output buffer. Effect parameters arrive as raw bytes and
//! are read with the layouts in [`abi`].

pub mod abi;
mod aux;
mod biquad;
mod command;
mod compressor;
mod decode;
mod delay;
pub mod fixed;
mod i3dl2;
mod limiter;
mod memory;
mod mix;
mod processor;
mod resample;
mod resample_lut;
mod reverb;
mod sink;
mod state;
mod upsample;

#[cfg(test)]
mod command_tests;
#[cfg(test)]
mod tests;

pub use biquad::BiquadFilterState;
pub use command::*;
pub use compressor::CompressorState;
pub use decode::{TEMP_BUFFER_SIZE, WaveBuffer};
pub use delay::{DelayLine, DelayState};
pub use i3dl2::{I3dl2DelayLine, I3dl2State};
pub use limiter::LightLimiterState;
pub use memory::{DspMemory, VecMemory};
pub use processor::{CommandListProcessor, DspClock, DspFrame, HostClock, ProcessReport};
pub use resample::{SrcQuality, resample};
pub use reverb::{ReverbDelayLine, ReverbState};
pub use sink::DeviceSinkFrame;
pub use state::{
    AdpcmContext, DspState, EffectResultId, EffectState, EffectStateId, UpsamplerId, VoiceState,
    VoiceStateId,
};
pub use upsample::{UpsamplerInfo, UpsamplerState};

/// Channels of a sink, effect or downmix.
pub const MAX_CHANNELS: usize = 6;
/// Mix buffers one voice or mix can address.
pub const MAX_MIX_BUFFERS: usize = 24;
/// Wave buffers queued per voice.
pub const MAX_WAVE_BUFFERS: usize = 4;
/// Biquad stages per voice.
pub const MAX_BIQUAD_FILTERS: usize = 2;
/// Samples per channel of a 48 kHz frame.
pub const TARGET_SAMPLE_COUNT: usize = 240;
/// The device output rate.
pub const TARGET_SAMPLE_RATE: u32 = 48_000;
