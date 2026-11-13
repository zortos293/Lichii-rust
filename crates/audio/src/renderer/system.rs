//! One renderer session's state: every context \`RequestUpdate\` fills and the command generator
//! reads.

use lichii_common::HleResult;

use super::behavior::BehaviorInfo;
use super::effect::EffectContext;
use super::memory::{MemoryPoolInfo, PoolLocation};
use super::mix::MixContext;
use super::parameter::{AudioRendererParameterInternal, ExecutionMode, work_buffer_size};
use super::performance::PerformanceManager;
use super::result::{INSUFFICIENT_BUFFER, INVALID_REVISION};
use super::revision::{SupportTag, is_valid_revision};
use super::sink::SinkContext;
use super::splitter::SplitterContext;
use super::updater::InfoUpdater;
use super::upsampler::UpsamplerManager;
use super::voice::VoiceContext;
use super::{MAX_CHANNELS, MAX_WAVE_BUFFERS};

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
    pub active: bool,
    pub frames_elapsed: u64,
    pub times_updated: u64,
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
        Ok(Self {
            sample_rate: params.sample_rate,
            sample_count: params.sample_count,
            mix_buffer_count: params.mixes as i16,
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
            upsamplers: UpsamplerManager::new(params.sinks + params.sub_mixes),
            performance: (params.perf_frames > 0)
                .then(|| PerformanceManager::new(&behavior, params)),
            render_time_limit_percent: 100,
            drop_voice: params.voice_drop_enabled != 0 && execution_mode == ExecutionMode::Auto,
            active: false,
            frames_elapsed: 0,
            times_updated: 0,
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
}
