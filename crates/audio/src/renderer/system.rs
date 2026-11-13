//! One renderer session's state: every context \`RequestUpdate\` fills and the command generator
//! reads.

use lichii_common::HleResult;

use super::behavior::BehaviorInfo;
use super::command::{
    CommandBuffer, CommandGenerator, EstimatorVersion, Output, ProcessingTimeEstimator,
};
use super::effect::EffectContext;
use super::memory::{self, MemoryPoolInfo, PoolLocation};
use super::mix::MixContext;
use super::parameter::{AudioRendererParameterInternal, ExecutionMode, work_buffer_size};
use super::performance::{self, PerformanceManager};
use super::result::{INSUFFICIENT_BUFFER, INVALID_REVISION};
use super::revision::{SupportTag, is_valid_revision};
use super::sink::SinkContext;
use super::splitter::SplitterContext;
use super::updater::InfoUpdater;
use super::upsampler::UpsamplerManager;
use super::voice::VoiceContext;
use super::{HIGHEST_VOICE_PRIORITY, MAX_CHANNELS, MAX_WAVE_BUFFERS, TARGET_SAMPLE_COUNT};
use crate::dsp::{CommandId, CommandList, DspClock, DspFrame, DspMemory, DspState, UpsamplerInfo};

/// Renderer state for one \`IAudioRenderer\` session.
pub struct RendererSystem {
    pub behavior: BehaviorInfo,
    pub sample_rate: u32,
    pub sample_count: u32,
    /// Mix buffers, read as the reference's signed 16-bit count.
    pub mix_buffer_count: i16,
    pub voice_channels: u32,
    pub render_device: u8,
    pub execution_mode: ExecutionMode,
    pub memory_pools: Vec<MemoryPoolInfo>,
    pub voices: VoiceContext,
    pub mixes: MixContext,
    pub effects: EffectContext,
    pub sinks: SinkContext,
    pub splitters: SplitterContext,
    pub upsamplers: UpsamplerManager,
    pub performance: Option<PerformanceManager>,
    pub render_time_limit_percent: u32,
    pub drop_voice: bool,
    /// Scales every estimate the voice drop pass reads (`SetVoiceDropParameter`).
    pub voice_drop_parameter: f32,
    pub active: bool,
    pub frames_elapsed: u64,
    pub times_updated: u64,
    pub session_id: i32,
    /// State the DSP keeps across frames, sized when the session opens.
    pub dsp: DspState,
    /// `buffer_count * sample_count` samples: the mix buffers, then one buffer per voice channel.
    pub mix_buffers: Vec<i32>,
    /// The list the last [`Self::generate_command`] built; its capacity carries over.
    pub command_list: CommandList,
    pub estimator: ProcessingTimeEstimator,
    /// Voices the last frame dropped, reported in the next performance frame.
    pub voices_dropped: u32,
}

impl RendererSystem {
    /// \`System::Initialize\`: validates the parameters against the transfer memory the game handed
    /// over and builds every context.
    pub fn new(
        params: &AudioRendererParameterInternal,
        transfer_memory_size: u64,
    ) -> HleResult<Self> {
        if !is_valid_revision(params.revision) {
            return Err(INVALID_REVISION);
        }
        if work_buffer_size(params) > transfer_memory_size {
            return Err(INSUFFICIENT_BUFFER);
        }
        if params.voices == 0 || params.sinks == 0 {
            return Err(INSUFFICIENT_BUFFER);
        }

        let behavior = BehaviorInfo::new(params.revision);
        let memory_pool_count = params.effects + params.voices * MAX_WAVE_BUFFERS as u32;
        let execution_mode = params.execution_mode();
        let mix_buffer_count = params.mixes as i16;
        let buffer_count = (MAX_CHANNELS as u32).wrapping_add(mix_buffer_count as i32 as u32);
        let upsampler_count = params.sinks + params.sub_mixes;
        let mut dsp = DspState::new(
            params.voices as usize,
            (mix_buffer_count.max(0) as usize).next_multiple_of(0x40),
            upsampler_count as usize,
            params.effects as usize,
        );
        for upsampler in &mut dsp.upsamplers {
            *upsampler = UpsamplerInfo {
                sample_count: TARGET_SAMPLE_COUNT,
                samples: vec![0; buffer_count as usize * TARGET_SAMPLE_COUNT as usize],
                ..UpsamplerInfo::default()
            };
        }
        if params.perf_frames > 0 {
            dsp.performance = vec![0; performance::frame_size(&behavior, params) as usize];
        }
        Ok(Self {
            sample_rate: params.sample_rate,
            sample_count: params.sample_count,
            mix_buffer_count,
            voice_channels: MAX_CHANNELS as u32,
            render_device: params.rendering_device,
            execution_mode,
            memory_pools: vec![MemoryPoolInfo::new(PoolLocation::Dsp); memory_pool_count as usize],
            voices: VoiceContext::new(params.voices),
            mixes: MixContext::new(params.sub_mixes + 1, params.effects, &behavior),
            effects: EffectContext::new(
                params.effects,
                behavior.supports(SupportTag::EffectInfoVer2),
            ),
            sinks: SinkContext::new(params.sinks),
            splitters: SplitterContext::new(&behavior, params),
            upsamplers: UpsamplerManager::new(upsampler_count),
            performance: (params.perf_frames > 0)
                .then(|| PerformanceManager::new(&behavior, params)),
            render_time_limit_percent: 100,
            drop_voice: params.voice_drop_enabled != 0 && execution_mode == ExecutionMode::Auto,
            voice_drop_parameter: 1.0,
            active: false,
            frames_elapsed: 0,
            times_updated: 0,
            session_id: 0,
            dsp,
            mix_buffers: vec![0; buffer_count as usize * params.sample_count as usize],
            command_list: CommandList {
                sample_count: params.sample_count,
                sample_rate: params.sample_rate,
                buffer_count,
                commands: Vec::new(),
            },
            estimator: ProcessingTimeEstimator::new(
                EstimatorVersion::for_behavior(&behavior),
                params.sample_count,
                mix_buffer_count as i32 as u32,
            ),
            voices_dropped: 0,
            behavior,
        })
    }

    pub fn start(&mut self) {
        self.frames_elapsed = 0;
        self.active = true;
    }

    pub fn stop(&mut self) {
        self.active = false;
    }

    /// \`RequestUpdate\`: applies \`input\` and writes the matching \`output\` and \`performance\`
    /// buffers. \`output\` is zeroed first, so a failed update leaves the sections written so far.
    pub fn update(
        &mut self,
        input: &[u8],
        performance: &mut [u8],
        output: &mut [u8],
    ) -> HleResult<()> {
        output.fill(0);
        let mut updater = InfoUpdater::new(input, output, &self.behavior)?;
        updater.update_behavior_info(&mut self.behavior)?;
        updater.update_memory_pools(&mut self.memory_pools)?;
        updater.update_voice_channel_resources(&mut self.voices)?;
        updater.update_voices(&mut self.voices, &self.memory_pools, &mut self.behavior)?;
        updater.update_effects(
            &mut self.effects,
            &mut self.dsp.effect_results,
            self.active,
            &self.memory_pools,
            &mut self.behavior,
        )?;
        if self.behavior.supports(SupportTag::Splitter) {
            updater.update_splitter_info(&mut self.splitters)?;
        }
        updater.update_mixes(
            &mut self.mixes,
            self.mix_buffer_count as i32 as u32,
            &self.effects,
            &self.splitters,
            &self.behavior,
        )?;
        updater.update_sinks(
            &mut self.sinks,
            &mut self.upsamplers,
            &self.memory_pools,
            &mut self.behavior,
        )?;
        updater.update_performance_buffer(performance, self.performance.as_mut())?;
        updater.update_error_info(&self.behavior)?;
        if self.behavior.supports(SupportTag::ElapsedFrameCount) {
            updater.update_renderer_info(self.frames_elapsed)?;
        }
        updater.check_consumed_size()?;
        self.times_updated += 1;
        Ok(())
    }
    /// `System::GenerateCommand`: builds this frame's command list from the state the last update
    /// left, runs the voice drop pass when the session allows it, and copies the DSP-shared states
    /// back. `device_channels` is the output device's channel count, which only
    /// [`ExecutionMode::Auto`] sessions use. Reuses the list's capacity, so a frame allocates
    /// nothing once the list has held its largest frame.
    pub fn generate_command(&mut self, device_channels: u32) -> &CommandList {
        memory::clear_use_state(&self.memory_pools);
        if let Some(performance) = self.performance.as_mut() {
            let frame = self.dsp.performance.len();
            performance.buffer_mut()[..frame].copy_from_slice(&self.dsp.performance);
            performance.tap_frame(false, self.voices_dropped, 0);
            self.voices_dropped = 0;
        }
        let channels = if self.execution_mode == ExecutionMode::Auto {
            device_channels
        } else {
            2
        };

        self.voices.sort_info();
        let list = &mut self.command_list;
        let (sample_count, sample_rate, buffer_count) =
            (list.sample_count, list.sample_rate, list.buffer_count);
        let mut generator = CommandGenerator {
            out: Output {
                buffer: CommandBuffer::new(list, &self.estimator),
                performance: self.performance.as_mut(),
            },
            behavior: &self.behavior,
            pools: &self.memory_pools,
            voices: &mut self.voices,
            mixes: &self.mixes,
            effects: &mut self.effects,
            sinks: &mut self.sinks,
            splitters: &mut self.splitters,
            upsamplers: &mut self.upsamplers,
            dsp: &mut self.dsp,
            mix_buffer_count: self.mix_buffer_count,
            channels,
            session_id: self.session_id,
            sample_count,
            sample_rate,
            buffer_count,
        };
        generator.generate_clear_mix_command();
        generator.generate_voice_commands();
        let start_estimated_time =
            self.voice_drop_parameter * generator.out.buffer.estimated_process_time() as f32;
        generator.generate_sub_mix_commands();
        generator.generate_final_mix_commands();
        generator.generate_sink_commands();
        let end_estimated_time =
            self.voice_drop_parameter * generator.out.buffer.estimated_process_time() as f32;

        if self.drop_voice {
            let dsp_time_limit = ((self.behavior.processing_time_limit_percent() / 100.0)
                * 2_880_000.0)
                * (self.render_time_limit_percent as f32 / 100.0);
            let time_limit =
                (dsp_time_limit + (start_estimated_time - end_estimated_time)).max(0.0) as u32;
            self.voices_dropped = drop_voices(
                &mut self.command_list,
                &mut self.voices,
                self.voice_drop_parameter,
                start_estimated_time as u32,
                time_limit,
            );
        }

        if let Some(performance) = self.performance.as_ref() {
            let frame = self.dsp.performance.len();
            self.dsp
                .performance
                .copy_from_slice(&performance.buffer()[..frame]);
        }
        self.voices.update_state_by_dsp_shared(&self.dsp.voices);
        if self.behavior.supports(SupportTag::EffectInfoVer2) {
            self.effects
                .update_state_by_dsp_shared(&self.dsp.effect_results);
        }
        self.frames_elapsed += 1;
        &self.command_list
    }

    /// The last generated list and everything [`CommandListProcessor::process`] needs to run it.
    ///
    /// [`CommandListProcessor::process`]: crate::dsp::CommandListProcessor::process
    pub fn dsp_frame<'a>(
        &'a mut self,
        memory: &'a mut dyn DspMemory,
        clock: &'a dyn DspClock,
        device_output: &'a mut [i16],
    ) -> (&'a CommandList, DspFrame<'a>) {
        (
            &self.command_list,
            DspFrame {
                mix_buffers: &mut self.mix_buffers,
                state: &mut self.dsp,
                memory,
                clock,
                device_output,
            },
        )
    }
}

fn is_voice_drop_start(id: CommandId) -> bool {
    matches!(
        id,
        CommandId::Performance
            | CommandId::DataSourcePcmInt16Version1
            | CommandId::DataSourcePcmInt16Version2
            | CommandId::DataSourcePcmFloatVersion1
            | CommandId::DataSourcePcmFloatVersion2
            | CommandId::DataSourceAdpcmVersion1
            | CommandId::DataSourceAdpcmVersion2
    )
}

/// `System::DropVoices`: walks the voice commands from the first data source or performance
/// command, lowest priority first, disabling whole voices until the estimate fits `time_limit`.
/// Depop prepares stay enabled so a dropped voice fades out. Returns the voices dropped.
///
/// The walk starts at the first command of that voice. The reference starts at the data source
/// itself, which skips the first voice's depop prepares when performance entries are off.
///
/// The reference advances its loop counter once more than its command cursor per dropped voice,
/// so the pass stops that many commands before the end of the list; this keeps that bound.
fn drop_voices(
    list: &mut CommandList,
    voices: &mut VoiceContext,
    voice_drop_parameter: f32,
    mut estimated_process_time: u32,
    time_limit: u32,
) -> u32 {
    let commands = &mut list.commands;
    let count = commands.len();
    let Some(start) = commands
        .iter()
        .position(|entry| is_voice_drop_start(entry.command.id()))
    else {
        return 0;
    };
    let mut cursor = commands[..start]
        .iter()
        .rposition(|entry| entry.node_id != commands[start].node_id)
        .map_or(0, |index| index + 1);
    let mut counter = cursor;
    let mut dropped = 0;
    while counter < count {
        let node_id = commands[cursor].node_id;
        if estimated_process_time <= time_limit || node_id >> 28 != 1 {
            break;
        }
        let index = ((node_id >> 16) & 0xFFF) as usize;
        if index >= voices.count() as usize {
            break;
        }
        let voice = voices.info_mut(index);
        if voice.priority == HIGHEST_VOICE_PRIORITY {
            break;
        }
        dropped += 1;
        voice.voice_dropped = true;
        while counter < count && commands[cursor].node_id == node_id {
            let entry = &mut commands[cursor];
            if entry.command.id() == CommandId::DepopPrepare {
                entry.enabled = true;
            } else if entry.enabled && entry.command.id() != CommandId::Performance {
                entry.enabled = false;
                estimated_process_time = estimated_process_time.wrapping_sub(
                    (voice_drop_parameter * entry.estimated_process_time as f32) as u32,
                );
            }
            cursor += 1;
            counter += 1;
        }
        counter += 1;
    }
    dropped
}
