//! Renderer open parameters and the work buffer size they need.

use bytemuck::{Pod, Zeroable};
use lichii_common::align_up;

use super::behavior::BehaviorInfo;
use super::nodes::{EdgeMatrix, NodeStates};
use super::performance;
use super::revision::SupportTag;
use super::splitter::SplitterContext;
use super::workbuffer::host_size;
use super::{MAX_CHANNELS, MAX_EFFECTS, MAX_WAVE_BUFFERS, TARGET_SAMPLE_COUNT};

/// Whether the renderer runs on its own (`Auto`) or when the game asks (`Manual`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ExecutionMode {
    #[default]
    Auto = 0,
    Manual = 1,
}

/// Parameters a game opens the renderer with (`OpenAudioRenderer`, `GetWorkBufferSize`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct AudioRendererParameterInternal {
    pub sample_rate: u32,
    pub sample_count: u32,
    pub mixes: u32,
    pub sub_mixes: u32,
    pub voices: u32,
    pub sinks: u32,
    pub effects: u32,
    pub perf_frames: u32,
    pub voice_drop_enabled: u8,
    pub unk_21: u8,
    pub rendering_device: u8,
    pub execution_mode: u8,
    pub splitter_infos: u32,
    pub splitter_destinations: i32,
    pub external_context_size: u32,
    pub revision: u32,
}
const _: () = assert!(size_of::<AudioRendererParameterInternal>() == 0x34);

impl AudioRendererParameterInternal {
    pub fn execution_mode(&self) -> ExecutionMode {
        if self.execution_mode == ExecutionMode::Manual as u8 {
            ExecutionMode::Manual
        } else {
            ExecutionMode::Auto
        }
    }
}

/// Bytes of command list the renderer reserves when the size varies with the parameters
/// (revision 5+).
pub fn command_buffer_size(
    behavior: &BehaviorInfo,
    params: &AudioRendererParameterInternal,
) -> u64 {
    use host_size::command::*;
    let voice_source = if behavior.supports(SupportTag::WaveBufferVer2) {
        ADPCM_DATA_SOURCE_V2
            .max(PCM_INT16_DATA_SOURCE_V2)
            .max(PCM_FLOAT_DATA_SOURCE_V2)
    } else {
        ADPCM_DATA_SOURCE_V1
            .max(PCM_INT16_DATA_SOURCE_V1)
            .max(PCM_FLOAT_DATA_SOURCE_V1)
    };
    let voice_size = voice_source + BIQUAD_FILTER * 2 + VOLUME_RAMP + MIX_RAMP_GROUPED;
    let mix_buffers = super::MAX_MIX_BUFFERS as u64;
    let destinations = params.splitter_destinations as i64 as u64;

    let mut size = params.effects as u64 * host_size::EFFECT_INFO;
    size += params.voices as u64 * (params.splitter_infos as u64 * DEPOP_PREPARE + voice_size);
    size += DEPOP_FOR_MIX_BUFFERS + MIX * mix_buffers * mix_buffers;
    size += DEPOP_FOR_MIX_BUFFERS + VOLUME * mix_buffers;
    size = size.wrapping_add(destinations.wrapping_mul(MIX_RAMP * mix_buffers));
    size += params.sinks as u64 * DEVICE_SINK.max(CIRCULAR_BUFFER_SINK);
    size += (params.effects as u64
        + params.voices as u64
        + params.sinks as u64
        + params.sub_mixes as u64
        + 1
        + performance::MAX_DETAIL_ENTRIES as u64)
        * PERFORMANCE;
    size
}

/// Work buffer bytes a game must hand the renderer for `params`, as the reference computes them.
pub fn work_buffer_size(params: &AudioRendererParameterInternal) -> u64 {
    let behavior = BehaviorInfo::new(params.revision);
    let mixes = params.mixes as u64;
    let sub_mixes = params.sub_mixes as u64;
    let voices = params.voices as u64;
    let sinks = params.sinks as u64;
    let effects = params.effects as u64;
    let effect_v2 = behavior.supports(SupportTag::EffectInfoVer2);

    let mut size = align_up(mixes * 4, 0x40);
    size += sub_mixes * MAX_EFFECTS as u64 * 4;
    size += (sub_mixes + 1) * host_size::MIX_INFO;
    size += voices
        * (host_size::VOICE_INFO + host_size::VOICE_CHANNEL_RESOURCE + host_size::VOICE_STATE);
    size += align_up((sub_mixes + 1) * 8, 0x10);
    size += align_up(voices * 8, 0x10);
    size += align_up(
        ((sinks + sub_mixes) * TARGET_SAMPLE_COUNT as u64 * 4 + params.sample_count as u64 * 4)
            * (mixes + MAX_CHANNELS as u64),
        0x40,
    );
    if behavior.supports(SupportTag::Splitter) {
        let nodes = params.sub_mixes + 1;
        size += align_up(
            NodeStates::work_buffer_size(nodes) + EdgeMatrix::work_buffer_size(nodes),
            0x10,
        );
    }
    size = size.wrapping_add(SplitterContext::work_buffer_size(&behavior, params));
    size += (effects + voices * MAX_WAVE_BUFFERS as u64) * host_size::MEMORY_POOL_INFO;
    if effect_v2 {
        size += effects * host_size::EFFECT_RESULT_STATE;
    }
    size += 0x50;
    size = align_up(size, 0x40);

    size += (sinks + sub_mixes) * host_size::UPSAMPLER_INFO;
    size += effects * host_size::EFFECT_INFO;
    size += align_up(voices * host_size::VOICE_STATE, 0x40);
    size += sinks * host_size::SINK_INFO;
    if effect_v2 {
        size += effects * host_size::EFFECT_RESULT_STATE;
    }
    if params.perf_frames > 0 {
        let frame = performance::frame_size(&behavior, params);
        size += align_up(frame * (params.perf_frames as u64 + 1) + 0xC0, 0x100);
    }
    size = size.wrapping_add(
        if behavior.supports(SupportTag::AudioRendererVariadicCommandBufferSize) {
            command_buffer_size(&behavior, params)
        } else {
            0x18000
        },
    );
    size += (0x40 - 1) * 2;
    align_up(size, 0x1000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::revision::encode_revision;

    #[allow(clippy::too_many_arguments)]
    fn params(
        sample_rate: u32,
        sample_count: u32,
        mixes: u32,
        sub_mixes: u32,
        voices: u32,
        sinks: u32,
        effects: u32,
        perf_frames: u32,
        splitter_infos: u32,
        splitter_destinations: i32,
        revision: u32,
    ) -> AudioRendererParameterInternal {
        AudioRendererParameterInternal {
            sample_rate,
            sample_count,
            mixes,
            sub_mixes,
            voices,
            sinks,
            effects,
            perf_frames,
            splitter_infos,
            splitter_destinations,
            revision,
            ..Default::default()
        }
    }

    #[test]
    fn work_buffer_sizes_match_the_reference() {
        let cases = [
            (
                params(
                    48000,
                    240,
                    24,
                    1,
                    24,
                    1,
                    0,
                    0,
                    0,
                    0,
                    encode_revision(1, false),
                ),
                0x38000,
                0x808,
            ),
            (
                params(
                    48000,
                    240,
                    24,
                    4,
                    96,
                    2,
                    4,
                    0,
                    2,
                    8,
                    encode_revision(4, false),
                ),
                0x74000,
                0xD08,
            ),
            (
                params(
                    48000,
                    240,
                    24,
                    4,
                    96,
                    2,
                    4,
                    1,
                    2,
                    8,
                    encode_revision(5, false),
                ),
                0x83000,
                0x1398,
            ),
            (
                params(
                    48000,
                    240,
                    24,
                    8,
                    128,
                    2,
                    8,
                    2,
                    4,
                    16,
                    encode_revision(9, false),
                ),
                0xC6000,
                0x1758,
            ),
            (
                params(
                    48000,
                    240,
                    24,
                    16,
                    256,
                    2,
                    16,
                    1,
                    16,
                    64,
                    encode_revision(13, false),
                ),
                0x1BD000,
                0x24D8,
            ),
            (
                params(
                    48000,
                    240,
                    24,
                    8,
                    192,
                    2,
                    8,
                    1,
                    8,
                    32,
                    encode_revision(15, false),
                ),
                0x108000,
                0x1D58,
            ),
            (
                params(
                    48000,
                    240,
                    24,
                    8,
                    192,
                    2,
                    8,
                    1,
                    8,
                    32,
                    encode_revision(16, true),
                ),
                0x108000,
                0x1D58,
            ),
            (
                params(
                    32000,
                    160,
                    12,
                    2,
                    64,
                    1,
                    2,
                    0,
                    0,
                    0,
                    encode_revision(15, true),
                ),
                0x41000,
                0x1020,
            ),
        ];
        for (params, size, frame) in cases {
            let behavior = BehaviorInfo::new(params.revision);
            assert_eq!(
                work_buffer_size(&params),
                size,
                "revision {:#x}",
                params.revision
            );
            assert_eq!(
                performance::frame_size(&behavior, &params),
                frame,
                "revision {:#x}",
                params.revision
            );
        }
    }
}
