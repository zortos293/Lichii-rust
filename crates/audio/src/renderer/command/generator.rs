//! The command generator (`command_generator.cpp` and the builders of `command_buffer.cpp`).
//!
//! Commands reach DSP state by handle: a voice channel's [`VoiceStateId`] is its channel resource
//! id, an effect's [`EffectStateId`] and [`EffectResultId`] are its effect index, and a device
//! sink's [`UpsamplerId`] is the upsampler slot it holds. Performance commands name byte offsets
//! into the current performance frame, which [`DspState::performance`] mirrors.

use super::CommandBuffer;
use crate::dsp::abi::{
    self, AuxParameterV1, BiquadFilterParameterV1, BiquadFilterParameterV2, BufferMixerParameterV1,
    CompressorParameterV2, DelayParameterV1, I3dl2ReverbParameterV1, LightLimiterParameterV2,
    ReverbParameterV2, VoiceBiquadParameter, VoiceBiquadParameterFloat,
};
use crate::dsp::fixed::{Fx, Q15};
use crate::dsp::{
    self, AuxCommand, BiquadFilterCommand, BiquadStateRef, CaptureCommand,
    CircularBufferSinkCommand, Command, CompressorCommand, CopyMixBufferCommand, DataSourceCommand,
    DelayCommand, DepopForMixBuffersCommand, DepopPrepareCommand, DeviceSinkCommand,
    DownMix6chTo2chCommand, DspState, EffectResultId, EffectStateId, I3dl2ReverbCommand,
    LightLimiterCommand, MixCommand, MixRampCommand, MixRampGroupedCommand,
    MultiTapBiquadFilterCommand, PerformanceCommand, PerformanceState, PreviousSampleRef,
    ReverbCommand, SinkSource, UpsampleCommand, UpsamplerId, VoiceStateId, VolumeCommand,
    VolumeRampCommand, WaveBuffer,
};
use crate::renderer::behavior::BehaviorInfo;
use crate::renderer::effect::{EffectContext, EffectInfo, EffectType, parameter_state};
use crate::renderer::memory::MemoryPoolInfo;
use crate::renderer::mix::{MixContext, MixInfo};
use crate::renderer::performance::{
    PerformanceEntryAddresses, PerformanceEntryType, PerformanceManager,
};
use crate::renderer::revision::SupportTag;
use crate::renderer::sink::{SinkContext, SinkKind, SinkType};
use crate::renderer::splitter::SplitterContext;
use crate::renderer::upsampler::UpsamplerManager;
use crate::renderer::voice::{VoiceContext, VoiceInfo};
use crate::renderer::{
    FINAL_MIX_ID, MAX_BIQUAD_FILTERS, MAX_CHANNELS, MAX_MIX_BUFFERS, SampleFormat, SrcQuality,
    TARGET_SAMPLE_RATE, UNUSED_MIX_ID, UNUSED_SPLITTER_ID, is_channel_count_valid,
};

/// Node id of commands that belong to no voice, mix or sink.
pub const INVALID_NODE_ID: u32 = 0xF000_0000;

mod detail {
    pub const PCM_INT16: u8 = 1;
    pub const ADPCM: u8 = 2;
    pub const VOLUME: u8 = 3;
    pub const BIQUAD_FILTER: u8 = 4;
    pub const MIX: u8 = 5;
    pub const DELAY: u8 = 6;
    pub const AUX: u8 = 7;
    pub const REVERB: u8 = 8;
    pub const I3DL2_REVERB: u8 = 9;
    pub const PCM_FLOAT: u8 = 10;
    pub const LIGHT_LIMITER: u8 = 11;
    pub const CAPTURE: u8 = 12;
    pub const COMPRESSOR: u8 = 13;
}

/// Where performance commands stamp a claimed entry, as the DSP addresses it.
fn dsp_addresses(addresses: PerformanceEntryAddresses) -> dsp::PerformanceEntryAddresses {
    dsp::PerformanceEntryAddresses {
        translated_address: 0,
        entry_start_time_offset: addresses.entry_start_time_offset as u32,
        header_entry_count_offset: addresses.header_entry_count_offset as u32,
        entry_processed_time_offset: addresses.entry_processed_time_offset as u32,
    }
}

/// The command buffer plus the performance manager whose entries bracket commands.
pub(crate) struct Output<'a> {
    pub buffer: CommandBuffer<'a>,
    pub performance: Option<&'a mut PerformanceManager>,
}

impl Output<'_> {
    fn push(&mut self, node_id: u32, command: Command) {
        self.buffer.push(node_id, command);
    }

    fn performance(
        &mut self,
        node_id: u32,
        state: PerformanceState,
        entry: PerformanceEntryAddresses,
    ) {
        self.push(
            node_id,
            Command::Performance(PerformanceCommand {
                state,
                entry_address: dsp_addresses(entry),
            }),
        );
    }

    /// `EntryAspect`: claims an entry and starts timing it.
    fn entry(
        &mut self,
        entry_type: PerformanceEntryType,
        node_id: u32,
    ) -> Option<PerformanceEntryAddresses> {
        let entry = self
            .performance
            .as_mut()?
            .next_entry(entry_type, node_id as i32)?;
        self.performance(node_id, PerformanceState::Start, entry);
        Some(entry)
    }

    /// `DetailAspect`: claims a detail when `node_id` is the game's detail target.
    fn detail(
        &mut self,
        entry_type: PerformanceEntryType,
        node_id: u32,
        detail_type: u8,
    ) -> Option<PerformanceEntryAddresses> {
        let performance = self.performance.as_mut()?;
        if !performance.is_detail_target(node_id) {
            return None;
        }
        let entry = performance.next_detail(detail_type, entry_type, node_id as i32)?;
        self.performance(node_id, PerformanceState::Start, entry);
        Some(entry)
    }

    fn stop(&mut self, node_id: u32, entry: Option<PerformanceEntryAddresses>) {
        if let Some(entry) = entry {
            self.performance(node_id, PerformanceState::Stop, entry);
        }
    }
}

/// Walks one session's renderer state and appends its commands. Every field borrows a piece of
/// [`RendererSystem`](crate::renderer::RendererSystem).
pub(crate) struct CommandGenerator<'a> {
    pub(crate) out: Output<'a>,
    pub(crate) behavior: &'a BehaviorInfo,
    pub(crate) pools: &'a [MemoryPoolInfo],
    pub(crate) voices: &'a mut VoiceContext,
    pub(crate) mixes: &'a MixContext,
    pub(crate) effects: &'a mut EffectContext,
    pub(crate) sinks: &'a mut SinkContext,
    pub(crate) splitters: &'a mut SplitterContext,
    pub(crate) upsamplers: &'a mut UpsamplerManager,
    pub(crate) dsp: &'a mut DspState,
    pub(crate) mix_buffer_count: i16,
    /// Channels of the output device; the 6 to 2 channel downmix runs only for stereo devices.
    pub(crate) channels: u32,
    pub(crate) session_id: i32,
    pub(crate) sample_count: u32,
    pub(crate) sample_rate: u32,
    pub(crate) buffer_count: u32,
}

fn to_q14(value: f32) -> i16 {
    (value * 16384.0).clamp(-32768.0, 32767.0) as i16
}

fn dsp_src_quality(quality: SrcQuality) -> dsp::SrcQuality {
    match quality {
        SrcQuality::Medium => dsp::SrcQuality::Medium,
        SrcQuality::High => dsp::SrcQuality::High,
        SrcQuality::Low => dsp::SrcQuality::Low,
    }
}

/// `UseOldChannelMapping`: before revision 11, 6 channel effects order center and LFE after the
/// back channels.
fn use_old_channel_mapping(inputs: &mut [i16; MAX_CHANNELS], outputs: &mut [i16; MAX_CHANNELS]) {
    inputs.swap(2, 4);
    inputs.swap(3, 5);
    outputs.swap(2, 4);
    outputs.swap(3, 5);
}

/// Effect channel routing: `offset + inputs[ch]` for the first `count` channels.
fn route(
    offset: i16,
    inputs: &[i8],
    outputs: &[i8],
    count: usize,
) -> ([i16; MAX_CHANNELS], [i16; MAX_CHANNELS]) {
    let mut routed = ([0; MAX_CHANNELS], [0; MAX_CHANNELS]);
    for ch in 0..count.min(MAX_CHANNELS) {
        routed.0[ch] = offset.wrapping_add(inputs[ch] as i16);
        routed.1[ch] = offset.wrapping_add(outputs[ch] as i16);
    }
    routed
}

/// The read-only pieces of the generator, copied out so they can be used while a context is
/// mutably borrowed.
#[derive(Clone, Copy)]
struct Shared<'s> {
    behavior: &'s BehaviorInfo,
    pools: &'s [MemoryPoolInfo],
    mixes: &'s MixContext,
    mix_buffer_count: i16,
}

impl Shared<'_> {
    fn supports(&self, tag: SupportTag) -> bool {
        self.behavior.supports(tag)
    }

    fn precision(&self) -> u8 {
        if self.supports(SupportTag::VolumeMixParameterPrecisionQ23) {
            23
        } else {
            15
        }
    }

    /// The configured mix a splitter destination or voice routes to, if it exists.
    fn mix(&self, mix_id: i32) -> Option<&MixInfo> {
        if (mix_id as u32) < self.mixes.count() {
            Some(self.mixes.info(mix_id as usize))
        } else {
            None
        }
    }
}

fn depop_prepare(out: &mut Output<'_>, voice: &VoiceInfo, state: VoiceStateId, mix: &MixInfo) {
    let mut inputs = [0; MAX_MIX_BUFFERS];
    for (i, input) in inputs.iter_mut().enumerate() {
        *input = mix.buffer_offset.wrapping_add(i as i16);
    }
    out.buffer.push_with(
        voice.node_id,
        voice.was_playing,
        Command::DepopPrepare(DepopPrepareCommand {
            inputs,
            previous_samples: state,
            buffer_count: mix.buffer_count as u32,
        }),
    );
}

/// `WaveBuffer::Copy`: both command versions carry the loop fields; version 1 decoding ignores
/// them.
fn wave_buffers(voice: &VoiceInfo, pools: &[MemoryPoolInfo]) -> [WaveBuffer; 4] {
    voice.wave_buffers.map(|source| {
        let (context, context_size) = if source.context_address.cpu_address() != 0 {
            (
                source.context_address.reference(pools, true),
                source.context_address.size(),
            )
        } else {
            (0, 0)
        };
        WaveBuffer {
            buffer: source.buffer_address.reference(pools, true),
            buffer_size: source.buffer_address.size(),
            context,
            context_size,
            start_offset: source.start_offset,
            end_offset: source.end_offset,
            loop_start_offset: source.loop_start_offset,
            loop_end_offset: source.loop_end_offset,
            loop_count: source.loop_count,
            looping: source.looping,
            stream_ended: source.stream_ended,
        }
    })
}

/// `GenerateDataSourceCommand`: depop prepares for every mix the voice feeds, then the decode.
fn data_source(
    out: &mut Output<'_>,
    shared: Shared<'_>,
    splitters: &SplitterContext,
    voice: &VoiceInfo,
    state: VoiceStateId,
    channel: usize,
) {
    if voice.mix_id == UNUSED_MIX_ID as u32 {
        if voice.splitter_id != UNUSED_SPLITTER_ID {
            let mut n = 0;
            while let Some(destination) = splitters.destination_data(voice.splitter_id, n) {
                if destination.is_configured()
                    && let Some(mix) = shared.mix(destination.mix_id())
                {
                    depop_prepare(out, voice, state, mix);
                }
                n += 1;
            }
        }
    } else if let Some(mix) = shared.mixes.infos().get(voice.mix_id as usize) {
        depop_prepare(out, voice, state, mix);
    }

    if voice.was_playing {
        return;
    }

    let version2 = shared.supports(SupportTag::WaveBufferVer2);
    let make = match (voice.sample_format, version2) {
        (SampleFormat::PcmInt16, false) => Command::DataSourcePcmInt16Version1,
        (SampleFormat::PcmInt16, true) => Command::DataSourcePcmInt16Version2,
        (SampleFormat::PcmFloat, false) => Command::DataSourcePcmFloatVersion1,
        (SampleFormat::PcmFloat, true) => Command::DataSourcePcmFloatVersion2,
        (SampleFormat::Adpcm, false) => Command::DataSourceAdpcmVersion1,
        (SampleFormat::Adpcm, true) => Command::DataSourceAdpcmVersion2,
        (format, _) => {
            tracing::error!(?format, "voice sample format has no data source command");
            return;
        }
    };
    let mut command = DataSourceCommand {
        src_quality: dsp_src_quality(voice.src_quality),
        output_index: shared.mix_buffer_count.wrapping_add(channel as i16),
        flags: voice.flags & 3,
        sample_rate: voice.sample_rate,
        pitch: voice.pitch,
        channel_index: channel as i8,
        channel_count: voice.channel_count,
        wave_buffers: wave_buffers(voice, shared.pools),
        voice_state: state,
        data_address: 0,
        data_size: 0,
    };
    if voice.sample_format == SampleFormat::Adpcm {
        if !version2 {
            command.channel_index = 0;
            command.channel_count = 0;
        }
        command.data_address = voice.data_address.reference(shared.pools, true);
        command.data_size = voice.data_address.size();
    }
    out.push(voice.node_id, make(command));
}

fn voice_biquad_parameter(voice: &VoiceInfo, index: usize) -> VoiceBiquadParameter {
    let biquad = voice.biquads[index];
    VoiceBiquadParameter {
        enabled: biquad.enabled,
        padding: 0,
        b: biquad.b,
        a: biquad.a,
    }
}

fn voice_biquad_float(voice: &VoiceInfo, index: usize) -> VoiceBiquadParameterFloat {
    if !voice.use_float_biquads {
        return VoiceBiquadParameterFloat::default();
    }
    let biquad = voice.biquads_float[index];
    VoiceBiquadParameterFloat {
        enabled: biquad.enabled,
        reserved: [0; 3],
        numerator: biquad.numerator,
        denominator: biquad.denominator,
    }
}

/// `GenerateBiquadFilterCommandForVoice`.
fn voice_biquads(
    out: &mut Output<'_>,
    shared: Shared<'_>,
    voice: &VoiceInfo,
    state: VoiceStateId,
    input: i16,
) {
    let enabled = voice.biquads.map(|biquad| biquad.enabled != 0);
    let use_float_processing = shared.supports(SupportTag::BiquadFilterFloatProcessing);
    if enabled == [true; MAX_BIQUAD_FILTERS]
        && shared.supports(SupportTag::MultiTapBiquadFilterProcessing)
        && use_float_processing
    {
        out.push(
            voice.node_id,
            Command::MultiTapBiquadFilter(MultiTapBiquadFilterCommand {
                input,
                output: input,
                biquads: [
                    voice_biquad_parameter(voice, 0),
                    voice_biquad_parameter(voice, 1),
                ],
                biquads_float: [voice_biquad_float(voice, 0), voice_biquad_float(voice, 1)],
                states: [0, 1].map(|filter| BiquadStateRef::Voice {
                    voice: state,
                    filter,
                }),
                needs_init: voice.biquad_initialized.map(|initialized| !initialized),
                filter_tap_count: MAX_BIQUAD_FILTERS as u8,
                use_float_coefficients: voice.use_float_biquads,
            }),
        );
        return;
    }
    for index in 0..MAX_BIQUAD_FILTERS {
        if !enabled[index] {
            continue;
        }
        out.push(
            voice.node_id,
            Command::BiquadFilter(BiquadFilterCommand {
                input,
                output: input,
                biquad: voice_biquad_parameter(voice, index),
                biquad_float: voice_biquad_float(voice, index),
                state: BiquadStateRef::Voice {
                    voice: state,
                    filter: index as u8,
                },
                needs_init: !voice.biquad_initialized[index],
                use_float_processing,
                use_float_coefficients: voice.use_float_biquads,
            }),
        );
    }
}

/// `GenerateVoiceMixCommand`: one grouped ramp for wide mixes, else one ramp per buffer.
#[allow(clippy::too_many_arguments)]
fn voice_mix(
    out: &mut Output<'_>,
    volumes: &[f32; MAX_MIX_BUFFERS],
    prev_volumes: &[f32; MAX_MIX_BUFFERS],
    state: VoiceStateId,
    mix: &MixInfo,
    input: i16,
    node_id: u32,
    precision: u8,
) {
    let count = (mix.buffer_count.max(0) as usize).min(MAX_MIX_BUFFERS);
    if mix.buffer_count > 8 {
        let mut command = MixRampGroupedCommand {
            precision,
            buffer_count: mix.buffer_count as u32,
            previous_samples: state,
            ..MixRampGroupedCommand::default()
        };
        for i in 0..count {
            command.inputs[i] = input;
            command.outputs[i] = mix.buffer_offset.wrapping_add(i as i16);
            command.prev_volumes[i] = prev_volumes[i];
            command.volumes[i] = volumes[i];
        }
        out.push(node_id, Command::MixRampGrouped(command));
        return;
    }
    for i in 0..count {
        if volumes[i] == 0.0 && prev_volumes[i] == 0.0 {
            continue;
        }
        out.push(
            node_id,
            Command::MixRamp(MixRampCommand {
                precision,
                input_index: input,
                output_index: mix.buffer_offset.wrapping_add(i as i16),
                prev_volume: prev_volumes[i],
                volume: volumes[i],
                previous_sample: PreviousSampleRef {
                    voice: state,
                    index: i as u8,
                },
            }),
        );
    }
}

fn depop_for_mix_buffers(out: &mut Output<'_>, mix: &MixInfo) {
    let decay = if mix.sample_rate == TARGET_SAMPLE_RATE {
        0.962_188_7
    } else {
        0.943_695_07
    };
    out.push(
        mix.node_id as u32,
        Command::DepopForMixBuffers(DepopForMixBuffersCommand {
            input: mix.buffer_offset as u32,
            count: mix.buffer_count as u32,
            decay: Q15::from_f32(decay),
        }),
    );
}

impl<'a> CommandGenerator<'a> {
    fn shared(&self) -> Shared<'a> {
        Shared {
            behavior: self.behavior,
            pools: self.pools,
            mixes: self.mixes,
            mix_buffer_count: self.mix_buffer_count,
        }
    }

    /// The command every list starts with.
    pub fn generate_clear_mix_command(&mut self) {
        self.out.push(INVALID_NODE_ID, Command::ClearMixBuffer);
    }

    /// Voices in render order, then the splitter volume carry-over.
    pub fn generate_voice_commands(&mut self) {
        for i in 0..self.voices.count() as usize {
            let index = self.voices.sorted()[i] as usize;
            if self.voices.info(index).should_skip()
                || !self.voices.update_for_command_generation(
                    index,
                    self.pools,
                    &mut self.dsp.voices,
                )
            {
                continue;
            }
            let node_id = self.voices.info(index).node_id;
            let entry = self.out.entry(PerformanceEntryType::Voice, node_id);
            self.generate_voice_command(index);
            self.out.stop(node_id, entry);
        }
        self.splitters.update_internal_state();
    }

    fn generate_voice_command(&mut self, index: usize) {
        let shared = self.shared();
        let precision = shared.precision();
        let channel_count = self.voices.info(index).channel_count.max(0) as usize;
        for channel in 0..channel_count.min(MAX_CHANNELS) {
            let (voice, resources) = self.voices.info_and_resources_mut(index);
            let resource_id = voice.channel_resource_ids[channel] as usize;
            let state = VoiceStateId(resource_id as u32);
            let node_id = voice.node_id;
            let input = shared.mix_buffer_count.wrapping_add(channel as i16);

            let detail_type = match voice.sample_format {
                SampleFormat::PcmInt16 => detail::PCM_INT16,
                SampleFormat::PcmFloat => detail::PCM_FLOAT,
                _ => detail::ADPCM,
            };
            let entry = self
                .out
                .detail(PerformanceEntryType::Voice, node_id, detail_type);
            data_source(&mut self.out, shared, self.splitters, voice, state, channel);
            self.out.stop(node_id, entry);

            if voice.was_playing {
                voice.prev_volume = 0.0;
                continue;
            }
            if !voice.has_any_connection() {
                continue;
            }

            let entry =
                self.out
                    .detail(PerformanceEntryType::Voice, node_id, detail::BIQUAD_FILTER);
            voice_biquads(&mut self.out, shared, voice, state, input);
            self.out.stop(node_id, entry);

            let entry = self
                .out
                .detail(PerformanceEntryType::Voice, node_id, detail::VOLUME);
            self.out.push(
                node_id,
                Command::VolumeRamp(VolumeRampCommand {
                    precision,
                    input_index: input,
                    output_index: input,
                    prev_volume: voice.prev_volume,
                    volume: voice.volume,
                }),
            );
            self.out.stop(node_id, entry);
            voice.prev_volume = voice.volume;

            if voice.mix_id == UNUSED_MIX_ID as u32 {
                if voice.splitter_id != UNUSED_SPLITTER_ID {
                    let mut n = channel as u32;
                    while let Some(destination) =
                        self.splitters.destination_index(voice.splitter_id, n)
                    {
                        let Some(data) = self.splitters.data(destination) else {
                            break;
                        };
                        if data.is_configured()
                            && let Some(mix) = shared.mix(data.mix_id())
                        {
                            voice_mix(
                                &mut self.out,
                                data.mix_volumes(),
                                data.prev_mix_volumes(),
                                state,
                                mix,
                                input,
                                node_id,
                                precision,
                            );
                            if let Some(data) = self.splitters.data_mut(destination) {
                                data.mark_as_need_to_update_internal_state();
                            }
                        }
                        n += voice.channel_count as u32;
                    }
                }
            } else if let Some(mix) = shared.mixes.infos().get(voice.mix_id as usize) {
                let entry = self
                    .out
                    .detail(PerformanceEntryType::Voice, node_id, detail::VOLUME);
                let resource = &mut resources[resource_id];
                voice_mix(
                    &mut self.out,
                    &resource.mix_volumes,
                    &resource.prev_mix_volumes,
                    state,
                    mix,
                    input,
                    node_id,
                    precision,
                );
                self.out.stop(node_id, entry);
                resource.prev_mix_volumes = resource.mix_volumes;
            }
            voice.biquad_initialized = voice.biquads.map(|biquad| biquad.enabled != 0);
        }
    }

    /// Sub mixes in render order: depop, effects, then the sends to their destinations.
    pub fn generate_sub_mix_commands(&mut self) {
        let mixes = self.mixes;
        for &index in mixes.sorted() {
            let mix = mixes.info(index as usize);
            if !mix.in_use || mix.mix_id == FINAL_MIX_ID {
                continue;
            }
            let node_id = mix.node_id as u32;
            let entry = self.out.entry(PerformanceEntryType::SubMix, node_id);
            depop_for_mix_buffers(&mut self.out, mix);
            self.generate_effect_commands(mix, index as usize);
            let detail = self
                .out
                .detail(PerformanceEntryType::SubMix, node_id, detail::MIX);
            self.generate_mix_commands(mix);
            self.out.stop(node_id, detail);
            self.out.stop(node_id, entry);
        }
    }

    /// The final mix: depop, effects, then its volume on every buffer.
    pub fn generate_final_mix_commands(&mut self) {
        let mix = self.mixes.final_mix();
        let node_id = mix.node_id as u32;
        let precision = self.shared().precision();
        let entry = self.out.entry(PerformanceEntryType::FinalMix, node_id);
        depop_for_mix_buffers(&mut self.out, mix);
        self.generate_effect_commands(mix, 0);
        for i in 0..mix.buffer_count.max(0) {
            let detail = self
                .out
                .detail(PerformanceEntryType::FinalMix, node_id, detail::VOLUME);
            let index = mix.buffer_offset.wrapping_add(i);
            self.out.push(
                node_id,
                Command::Volume(VolumeCommand {
                    precision,
                    input_index: index,
                    output_index: index,
                    volume: mix.volume,
                }),
            );
            self.out.stop(node_id, detail);
        }
        self.out.stop(node_id, entry);
    }

    /// `GenerateMixCommands`: sends a sub mix to its destination mix or splitter.
    fn generate_mix_commands(&mut self, mix: &MixInfo) {
        let shared = self.shared();
        let precision = shared.precision();
        if !mix.has_any_connection() {
            return;
        }
        let node_id = mix.node_id as u32;
        if mix.dst_mix_id == UNUSED_MIX_ID {
            if mix.dst_splitter_id == UNUSED_SPLITTER_ID {
                return;
            }
            let mut n = 0u32;
            while let Some(destination) = self.splitters.destination_data(mix.dst_splitter_id, n) {
                if destination.is_configured()
                    && let Some(target) = shared.mix(destination.mix_id())
                {
                    let input = mix
                        .buffer_offset
                        .wrapping_add((n as i16).checked_rem(mix.buffer_count).unwrap_or(0));
                    let count = (target.buffer_count.max(0) as usize).min(MAX_MIX_BUFFERS);
                    for i in 0..count {
                        let volume = mix.volume * destination.mix_volumes()[i];
                        if volume != 0.0 {
                            self.out.push(
                                node_id,
                                Command::Mix(MixCommand {
                                    precision,
                                    input_index: input,
                                    output_index: target.buffer_offset.wrapping_add(i as i16),
                                    volume,
                                }),
                            );
                        }
                    }
                }
                n += 1;
            }
            return;
        }
        let Some(target) = shared.mixes.infos().get(mix.dst_mix_id as usize) else {
            return;
        };
        let count = (mix.buffer_count.max(0) as usize).min(MAX_MIX_BUFFERS);
        let target_count = (target.buffer_count.max(0) as usize).min(MAX_MIX_BUFFERS);
        for i in 0..count {
            for j in 0..target_count {
                let volume = mix.volume * mix.mix_volumes[i][j];
                if volume != 0.0 {
                    self.out.push(
                        node_id,
                        Command::Mix(MixCommand {
                            precision,
                            input_index: mix.buffer_offset.wrapping_add(i as i16),
                            output_index: target.buffer_offset.wrapping_add(j as i16),
                            volume,
                        }),
                    );
                }
            }
        }
    }

    /// The effects of the mix at `mix_index`, by process order.
    fn generate_effect_commands(&mut self, mix: &MixInfo, mix_index: usize) {
        let mixes = self.mixes;
        let entry_type = if mix.mix_id == FINAL_MIX_ID {
            PerformanceEntryType::FinalMix
        } else {
            PerformanceEntryType::SubMix
        };
        let node_id = mix.node_id as u32;
        for &effect_index in mixes.effect_order(mix_index) {
            let Ok(index) = usize::try_from(effect_index) else {
                break;
            };
            let effect = self.effects.info(index);
            if effect.should_skip() {
                continue;
            }
            let detail_type = match effect.effect_type {
                EffectType::Mix => detail::MIX,
                EffectType::Aux => detail::AUX,
                EffectType::Delay => detail::DELAY,
                EffectType::Reverb => detail::REVERB,
                EffectType::I3dl2Reverb => detail::I3DL2_REVERB,
                EffectType::BiquadFilter => detail::BIQUAD_FILTER,
                EffectType::LightLimiter => detail::LIGHT_LIMITER,
                EffectType::Capture => detail::CAPTURE,
                EffectType::Compressor => detail::COMPRESSOR,
                EffectType::Invalid => {
                    tracing::error!(index, "effect without a type in a mix's effect order");
                    self.effects.info_mut(index).update_for_command_generation();
                    continue;
                }
            };
            let detail = self.out.detail(entry_type, node_id, detail_type);
            self.generate_effect_command(index, mix.buffer_offset, node_id);
            self.out.stop(node_id, detail);
            self.effects.info_mut(index).update_for_command_generation();
        }
    }

    fn generate_effect_command(&mut self, index: usize, offset: i16, node_id: u32) {
        let shared = self.shared();
        let effect = self.effects.info(index);
        let state = EffectStateId(index as u32);
        let out = &mut self.out;
        match effect.effect_type {
            EffectType::Mix => buffer_mixer(out, shared, effect, offset, node_id),
            EffectType::Aux => aux(out, shared, effect, offset, node_id, self.sample_count),
            EffectType::Capture => capture(out, shared, effect, offset, node_id, self.sample_count),
            EffectType::Delay => {
                let parameter: DelayParameterV1 = parameter(effect);
                let mut command = DelayCommand::default();
                if is_channel_count_valid(parameter.channel_count) {
                    (command.inputs, command.outputs) = route(
                        offset,
                        &parameter.inputs,
                        &parameter.outputs,
                        parameter.channel_count as usize,
                    );
                    if !shared.supports(SupportTag::DelayChannelMappingChange)
                        && parameter.channel_count == 6
                    {
                        use_old_channel_mapping(&mut command.inputs, &mut command.outputs);
                    }
                    command.parameter = parameter;
                    command.effect_enabled = effect.enabled;
                    command.state = state;
                    effect.workbuffer(0, shared.pools);
                }
                out.push(node_id, Command::Delay(command));
            }
            EffectType::Reverb => {
                let parameter: ReverbParameterV2 = parameter(effect);
                let mut command = ReverbCommand::default();
                if is_channel_count_valid(parameter.channel_count) {
                    (command.inputs, command.outputs) = route(
                        offset,
                        &parameter.inputs,
                        &parameter.outputs,
                        parameter.channel_count as usize,
                    );
                    if !shared.supports(SupportTag::ReverbChannelMappingChange)
                        && parameter.channel_count == 6
                    {
                        use_old_channel_mapping(&mut command.inputs, &mut command.outputs);
                    }
                    command.parameter = parameter;
                    command.effect_enabled = effect.enabled;
                    command.state = state;
                    effect.workbuffer(0, shared.pools);
                    command.long_size_pre_delay_supported =
                        shared.supports(SupportTag::LongSizePreDelay);
                }
                out.push(node_id, Command::Reverb(command));
            }
            EffectType::I3dl2Reverb => {
                let parameter: I3dl2ReverbParameterV1 = parameter(effect);
                let mut command = I3dl2ReverbCommand::default();
                if is_channel_count_valid(parameter.channel_count) {
                    (command.inputs, command.outputs) = route(
                        offset,
                        &parameter.inputs,
                        &parameter.outputs,
                        parameter.channel_count as usize,
                    );
                    if !shared.supports(SupportTag::I3dl2ReverbChannelMappingChange)
                        && parameter.channel_count == 6
                    {
                        use_old_channel_mapping(&mut command.inputs, &mut command.outputs);
                    }
                    command.parameter = parameter;
                    command.effect_enabled = effect.enabled;
                    command.state = state;
                    effect.workbuffer(0, shared.pools);
                }
                out.push(node_id, Command::I3dl2Reverb(command));
            }
            EffectType::BiquadFilter => {
                biquad_filter_effect(out, shared, effect, state, offset, node_id)
            }
            EffectType::LightLimiter => {
                effect.workbuffer(0, shared.pools);
                let version2 = shared.supports(SupportTag::EffectInfoVer2);
                let parameter: LightLimiterParameterV2 = parameter(effect);
                let mut command = LightLimiterCommand::default();
                if is_channel_count_valid(parameter.channel_count) {
                    (command.inputs, command.outputs) = route(
                        offset,
                        &parameter.inputs,
                        &parameter.outputs,
                        parameter.channel_count as usize,
                    );
                    command.parameter = parameter;
                    command.effect_enabled = effect.enabled;
                    command.state = state;
                    if version2 && parameter.statistics_enabled != 0 {
                        command.result_state = Some(EffectResultId(index as u32));
                    }
                }
                out.push(
                    node_id,
                    if version2 {
                        Command::LightLimiterVersion2(command)
                    } else {
                        Command::LightLimiterVersion1(command)
                    },
                );
            }
            EffectType::Compressor => {
                let parameter: CompressorParameterV2 = parameter(effect);
                let mut command = CompressorCommand::default();
                if is_channel_count_valid(parameter.channel_count as u16) {
                    (command.inputs, command.outputs) = route(
                        offset,
                        &parameter.inputs,
                        &parameter.outputs,
                        parameter.channel_count as usize,
                    );
                    command.parameter = parameter;
                    command.state = state;
                    command.effect_enabled = effect.enabled;
                }
                out.push(node_id, Command::Compressor(command));
            }
            EffectType::Invalid => {}
        }
    }

    /// Device sinks, then circular buffer sinks, each reading the final mix.
    pub fn generate_sink_commands(&mut self) {
        let offset = self.mixes.final_mix().buffer_offset;
        for kind in [SinkType::DeviceSink, SinkType::CircularBufferSink] {
            for index in 0..self.sinks.count() as usize {
                let sink = self.sinks.info_mut(index);
                if !sink.in_use || sink.sink_type() != kind {
                    continue;
                }
                if let SinkKind::Device { state, .. } = &mut sink.kind
                    && self.sample_rate != TARGET_SAMPLE_RATE
                    && state.upsampler.is_none()
                {
                    state.upsampler = self.upsamplers.allocate();
                }
                let node_id = sink.node_id;
                let entry = self.out.entry(PerformanceEntryType::Sink, node_id);
                self.generate_sink_command(index, offset);
                self.out.stop(node_id, entry);
            }
        }
    }

    fn generate_sink_command(&mut self, index: usize, offset: i16) {
        let sink = self.sinks.info(index);
        if sink.should_skip() {
            return;
        }
        match sink.kind {
            SinkKind::Device { parameter, state } => {
                if self.channels == 2 && parameter.downmix_enabled != 0 {
                    let (inputs, _) =
                        route(offset, &parameter.inputs, &parameter.inputs, MAX_CHANNELS);
                    self.out.push(
                        INVALID_NODE_ID,
                        Command::DownMix6chTo2ch(DownMix6chTo2chCommand {
                            inputs,
                            outputs: inputs,
                            down_mix_coeff: parameter.downmix_coeff.map(Fx::from_f32),
                        }),
                    );
                }
                let input_count = parameter.input_count.min(MAX_CHANNELS as u32);
                let (inputs, _) = route(
                    offset,
                    &parameter.inputs,
                    &parameter.inputs,
                    input_count as usize,
                );
                let mut source = SinkSource::MixBuffers;
                if let Some(slot) = state.upsampler {
                    if let Some(info) = self.dsp.upsamplers.get_mut(slot as usize) {
                        info.input_count = parameter.input_count;
                        info.inputs[..input_count as usize]
                            .copy_from_slice(&inputs[..input_count as usize]);
                    }
                    self.out.push(
                        INVALID_NODE_ID,
                        Command::Upsample(UpsampleCommand {
                            buffer_count: self.buffer_count,
                            unk_20: 0,
                            source_sample_count: self.sample_count,
                            source_sample_rate: self.sample_rate,
                            upsampler: UpsamplerId(slot),
                        }),
                    );
                    source = SinkSource::Upsampler(UpsamplerId(slot));
                }
                self.out.push(
                    INVALID_NODE_ID,
                    Command::DeviceSink(DeviceSinkCommand {
                        session_id: self.session_id,
                        source,
                        input_count: parameter.input_count,
                        inputs,
                    }),
                );
            }
            SinkKind::CircularBuffer { parameter, state } => {
                let input_count = parameter.input_count.min(MAX_CHANNELS as u32) as usize;
                let (inputs, _) = route(offset, &parameter.inputs, &parameter.inputs, input_count);
                self.out.push(
                    sink.node_id,
                    Command::CircularBufferSink(CircularBufferSinkCommand {
                        input_count: parameter.input_count,
                        inputs,
                        address: state.address_info.reference(self.pools, true),
                        size: parameter.size,
                        pos: state.current_pos as u32,
                    }),
                );
            }
            SinkKind::Invalid => {}
        }
        self.sinks.info_mut(index).update_for_command_generation();
    }
}

fn parameter<T: bytemuck::Pod + Default>(effect: &EffectInfo) -> T {
    abi::parse(&effect.parameter).unwrap_or_default()
}

/// `GenerateBufferMixerCommand`: the mix effect as plain mix commands.
fn buffer_mixer(
    out: &mut Output<'_>,
    shared: Shared<'_>,
    effect: &EffectInfo,
    offset: i16,
    node_id: u32,
) {
    if !effect.enabled {
        return;
    }
    let parameter: BufferMixerParameterV1 = parameter(effect);
    let precision = shared.precision();
    for i in 0..(parameter.mix_count as usize).min(MAX_MIX_BUFFERS) {
        if parameter.volumes[i] == 0.0 {
            continue;
        }
        out.push(
            node_id,
            Command::Mix(MixCommand {
                precision,
                input_index: offset.wrapping_add(parameter.inputs[i] as i16),
                output_index: offset.wrapping_add(parameter.outputs[i] as i16),
                volume: parameter.volumes[i],
            }),
        );
    }
}

/// The ring buffer bookkeeping aux and capture share: every channel but the last carries an
/// update count of 0, and write offsets advance a frame per channel.
fn ring_channels(
    parameter: &AuxParameterV1,
    sample_count: u32,
    mut emit: impl FnMut(usize, u32, u32),
) {
    let mut channel_index = parameter.mix_buffer_count.wrapping_sub(1);
    let mut write_offset = 0u32;
    for i in 0..(parameter.mix_buffer_count as usize).min(MAX_MIX_BUFFERS) {
        let new_update_count = sample_count.wrapping_add(write_offset);
        let update_count = if channel_index > 0 {
            0
        } else {
            new_update_count
        };
        emit(i, update_count, write_offset);
        write_offset = new_update_count;
        channel_index = channel_index.wrapping_sub(1);
    }
}

fn aux(
    out: &mut Output<'_>,
    shared: Shared<'_>,
    effect: &EffectInfo,
    offset: i16,
    node_id: u32,
    sample_count: u32,
) {
    if effect.enabled {
        effect.workbuffer(0, shared.pools);
        effect.workbuffer(1, shared.pools);
    }
    if effect.send_buffer == 0 || effect.return_buffer == 0 {
        return;
    }
    let parameter: AuxParameterV1 = parameter(effect);
    ring_channels(&parameter, sample_count, |i, update_count, write_offset| {
        out.push(
            node_id,
            Command::Aux(AuxCommand {
                input: offset.wrapping_add(parameter.inputs[i] as i16),
                output: offset.wrapping_add(parameter.outputs[i] as i16),
                send_buffer_info: effect.send_buffer_info,
                return_buffer_info: effect.return_buffer_info,
                send_buffer: effect.send_buffer,
                return_buffer: effect.return_buffer,
                count_max: parameter.count_max,
                write_offset,
                update_count,
                effect_enabled: effect.enabled,
            }),
        );
    });
}

fn capture(
    out: &mut Output<'_>,
    shared: Shared<'_>,
    effect: &EffectInfo,
    offset: i16,
    node_id: u32,
    sample_count: u32,
) {
    if effect.enabled {
        effect.workbuffer(0, shared.pools);
    }
    if effect.send_buffer == 0 {
        return;
    }
    let parameter: AuxParameterV1 = parameter(effect);
    ring_channels(&parameter, sample_count, |i, update_count, write_offset| {
        out.push(
            node_id,
            Command::Capture(CaptureCommand {
                input: offset.wrapping_add(parameter.inputs[i] as i16),
                output: offset.wrapping_add(parameter.outputs[i] as i16),
                send_buffer_info: effect.send_buffer_info,
                send_buffer: effect.send_buffer,
                count_max: parameter.count_max,
                write_offset,
                update_count,
                effect_enabled: effect.enabled,
            }),
        );
    });
}

/// `GenerateBiquadFilterEffectCommand`: a biquad per channel, or plain copies when disabled.
fn biquad_filter_effect(
    out: &mut Output<'_>,
    shared: Shared<'_>,
    effect: &EffectInfo,
    state: EffectStateId,
    offset: i16,
    node_id: u32,
) {
    let (parameter_state, channel_count, inputs, outputs, biquad) =
        if shared.supports(SupportTag::EffectInfoVer2) {
            let p: BiquadFilterParameterV2 = parameter(effect);
            let biquad = VoiceBiquadParameter {
                b: p.b.map(to_q14),
                a: p.a.map(to_q14),
                ..VoiceBiquadParameter::default()
            };
            (p.state, p.channel_count, p.inputs, p.outputs, biquad)
        } else {
            let p: BiquadFilterParameterV1 = parameter(effect);
            let biquad = VoiceBiquadParameter {
                b: p.b,
                a: p.a,
                ..VoiceBiquadParameter::default()
            };
            (p.state, p.channel_count, p.inputs, p.outputs, biquad)
        };
    if channel_count <= 0 || parameter_state > parameter_state::UPDATED {
        return;
    }
    let channels = (channel_count as usize).min(MAX_CHANNELS);
    let (inputs, outputs) = route(offset, &inputs, &outputs, channels);
    if !effect.enabled {
        for ch in 0..channels {
            out.push(
                node_id,
                Command::CopyMixBuffer(CopyMixBufferCommand {
                    input_index: inputs[ch],
                    output_index: outputs[ch],
                }),
            );
        }
        return;
    }
    let needs_init = match parameter_state {
        parameter_state::INITIALIZED => true,
        parameter_state::UPDATING => {
            !shared.supports(SupportTag::BiquadFilterEffectStateClearBugFix)
        }
        _ => false,
    };
    let use_float_processing = shared.supports(SupportTag::BiquadFilterFloatProcessing);
    for ch in 0..channels {
        out.push(
            node_id,
            Command::BiquadFilter(BiquadFilterCommand {
                input: inputs[ch],
                output: outputs[ch],
                biquad,
                biquad_float: VoiceBiquadParameterFloat::default(),
                state: BiquadStateRef::Effect {
                    effect: state,
                    channel: ch as u8,
                },
                needs_init,
                use_float_processing,
                use_float_coefficients: false,
            }),
        );
    }
}
