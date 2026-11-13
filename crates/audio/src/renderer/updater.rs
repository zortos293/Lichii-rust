//! `RequestUpdate`: parses each input section into the renderer state and writes the matching
//! output section.
//!
//! Both buffers start with an [`UpdateDataHeader`] giving each section's size. Sections follow
//! in a fixed order; every parser checks that it consumed exactly the size the header names,
//! and the update fails unless both buffers are consumed to the byte. Where the reference reads
//! outside the game's buffers (indices past a table, sections shorter than their count), this
//! port fails the update with `ResultInvalidUpdateInfo` instead.

use bytemuck::{Pod, Zeroable};
use lichii_common::HleResult;

use super::behavior::{BehaviorInParameter, BehaviorInfo, BehaviorOutStatus, ErrorInfo};
use super::effect::{
    EffectContext, EffectInParameter, EffectInfo, EffectOutStatusV1, EffectOutStatusV2,
    EffectResultState, EffectType,
};
use super::memory::{MemoryPoolInParameter, MemoryPoolInfo, MemoryPoolOutStatus, PoolMapper};
use super::mix::{MixContext, MixInDirtyParameter, MixInParameter};
use super::performance::{PerformanceInParameter, PerformanceManager, PerformanceOutStatus};
use super::result::INVALID_UPDATE_INFO;
use super::revision::{SupportTag, is_valid_revision, revision_number};
use super::sink::{SinkContext, SinkInParameter, SinkOutStatus};
use super::splitter::SplitterContext;
use super::upsampler::UpsamplerManager;
use super::voice::{
    VoiceChannelResourceInParameter, VoiceContext, VoiceInParameter, VoiceInParameterV2,
    VoiceOutStatus, VoiceState,
};
use super::{FINAL_MIX_ID, MAX_CHANNELS, MAX_WAVE_BUFFERS, UNUSED_MIX_ID};

/// Header of both the input and the output buffer.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct UpdateDataHeader {
    pub revision: u32,
    pub behavior_size: u32,
    pub memory_pool_size: u32,
    pub voices_size: u32,
    pub voice_resources_size: u32,
    pub effects_size: u32,
    pub mix_size: u32,
    pub sinks_size: u32,
    pub performance_buffer_size: u32,
    pub splitter_size: u32,
    pub render_info_size: u32,
    pub _unk_2c: [u8; 0x10],
    pub size: u32,
}
const _: () = assert!(size_of::<UpdateDataHeader>() == 0x40);

/// Renderer info output section (revision 5+).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct RenderInfoOutStatus {
    pub elapsed_frames: u64,
    pub _unk_08: [u8; 8],
}
const _: () = assert!(size_of::<RenderInfoOutStatus>() == 0x10);

const HEADER_SIZE: usize = size_of::<UpdateDataHeader>();

/// Walks one `RequestUpdate`.
pub struct InfoUpdater<'a> {
    input: &'a [u8],
    in_offset: usize,
    output: &'a mut [u8],
    out_offset: usize,
    in_header: UpdateDataHeader,
    out_header: UpdateDataHeader,
}

fn invalid<T>(section: &str, header: u32, consumed: usize) -> HleResult<T> {
    tracing::error!(
        section,
        header,
        consumed,
        "audio update section size mismatch"
    );
    Err(INVALID_UPDATE_INFO)
}

impl<'a> InfoUpdater<'a> {
    /// Starts an update and writes the output header. `output` must already be zeroed.
    pub fn new(input: &'a [u8], output: &'a mut [u8], behavior: &BehaviorInfo) -> HleResult<Self> {
        if input.len() < HEADER_SIZE || output.len() < HEADER_SIZE {
            return Err(INVALID_UPDATE_INFO);
        }
        let mut updater = Self {
            in_header: bytemuck::pod_read_unaligned(&input[..HEADER_SIZE]),
            input,
            in_offset: HEADER_SIZE,
            output,
            out_offset: HEADER_SIZE,
            out_header: UpdateDataHeader {
                revision: behavior.process_revision(),
                size: HEADER_SIZE as u32,
                ..Default::default()
            },
        };
        updater.write_header();
        Ok(updater)
    }

    fn write_header(&mut self) {
        self.output[..HEADER_SIZE].copy_from_slice(bytemuck::bytes_of(&self.out_header));
    }

    fn read<T: Pod>(&self, offset: usize) -> HleResult<T> {
        let start = self
            .in_offset
            .checked_add(offset)
            .ok_or(INVALID_UPDATE_INFO)?;
        let end = start
            .checked_add(size_of::<T>())
            .ok_or(INVALID_UPDATE_INFO)?;
        self.input
            .get(start..end)
            .map(bytemuck::pod_read_unaligned)
            .ok_or(INVALID_UPDATE_INFO)
    }

    fn ensure_output(&self, size: usize) -> HleResult<()> {
        if self.out_offset + size > self.output.len() {
            return Err(INVALID_UPDATE_INFO);
        }
        Ok(())
    }

    fn write<T: Pod>(&mut self, offset: usize, value: &T) {
        let start = self.out_offset + offset;
        self.output[start..start + size_of::<T>()].copy_from_slice(bytemuck::bytes_of(value));
    }

    fn finish_output(&mut self, size: usize) {
        self.out_offset += size;
        self.out_header.size += size as u32;
        self.write_header();
    }

    pub fn update_behavior_info(&mut self, behavior: &mut BehaviorInfo) -> HleResult<()> {
        let params: BehaviorInParameter = self.read(0)?;
        if !is_valid_revision(params.revision) {
            tracing::error!(
                revision = format_args!("{:#x}", params.revision),
                "invalid update revision"
            );
            return Err(INVALID_UPDATE_INFO);
        }
        if revision_number(params.revision) != revision_number(behavior.user_revision()) {
            tracing::error!(
                update = format_args!("{:#x}", params.revision),
                renderer = format_args!("{:#x}", behavior.user_revision()),
                "update revision does not match the renderer's"
            );
            return Err(INVALID_UPDATE_INFO);
        }
        behavior.clear_error();
        behavior.update_flags(params.flags);
        let consumed = size_of::<BehaviorInParameter>();
        if self.in_header.behavior_size as usize != consumed {
            return invalid("behavior", self.in_header.behavior_size, consumed);
        }
        self.in_offset += consumed;
        Ok(())
    }

    pub fn update_memory_pools(&mut self, pools: &mut [MemoryPoolInfo]) -> HleResult<()> {
        let in_size = size_of::<MemoryPoolInParameter>();
        let out_size = size_of::<MemoryPoolOutStatus>();
        self.ensure_output(pools.len() * out_size)?;
        for (i, pool) in pools.iter_mut().enumerate() {
            let params: MemoryPoolInParameter = self.read(i * in_size)?;
            let mut out = MemoryPoolOutStatus::zeroed();
            pool.update(&params, &mut out);
            self.write(i * out_size, &out);
        }
        let consumed = pools.len() * in_size;
        if self.in_header.memory_pool_size as usize != consumed {
            return invalid("memory pools", self.in_header.memory_pool_size, consumed);
        }
        self.in_offset += consumed;
        self.out_header.memory_pool_size = (pools.len() * out_size) as u32;
        self.finish_output(pools.len() * out_size);
        Ok(())
    }

    pub fn update_voice_channel_resources(&mut self, voices: &mut VoiceContext) -> HleResult<()> {
        let in_size = size_of::<VoiceChannelResourceInParameter>();
        let count = voices.count() as usize;
        for i in 0..count {
            let params: VoiceChannelResourceInParameter = self.read(i * in_size)?;
            let resource = voices.channel_resource_mut(i);
            resource.in_use = params.in_use != 0;
            if resource.in_use {
                resource.mix_volumes = params.mix_volumes;
            }
        }
        let consumed = count * in_size;
        if self.in_header.voice_resources_size as usize != consumed {
            return invalid(
                "voice resources",
                self.in_header.voice_resources_size,
                consumed,
            );
        }
        self.in_offset += consumed;
        Ok(())
    }

    pub fn update_voices(
        &mut self,
        voices: &mut VoiceContext,
        pools: &[MemoryPoolInfo],
        behavior: &mut BehaviorInfo,
    ) -> HleResult<()> {
        let mapper = PoolMapper::new(pools, behavior.is_memory_force_mapping_enabled());
        let count = voices.count() as usize;
        let use_v2 = behavior.supports(SupportTag::VoiceInParameterV2);
        let stride = if use_v2 {
            size_of::<VoiceInParameterV2>()
        } else {
            size_of::<VoiceInParameter>()
        };
        let out_size = size_of::<VoiceOutStatus>();
        self.ensure_output(count * out_size)?;

        let (infos, states) = voices.split_for_update();
        for info in infos.iter_mut() {
            info.in_use = false;
        }
        let mut new_voice_count = 0u32;
        for i in 0..count {
            let (params, float_biquads) = if use_v2 {
                let v2: VoiceInParameterV2 = self.read(i * stride)?;
                if behavior.user_revision_num() >= 16 && v2.in_use != 0 && v2.crossfade_curve != 0 {
                    tracing::error!(
                        curve = v2.crossfade_curve,
                        "unsupported voice biquad crossfade"
                    );
                    return Err(INVALID_UPDATE_INFO);
                }
                let (v1, float_biquads) = v2.to_v1();
                (v1, Some(float_biquads))
            } else {
                (self.read::<VoiceInParameter>(i * stride)?, None)
            };
            if params.in_use == 0 {
                continue;
            }

            let channel_count = params.channel_count as usize;
            let ids = &params.channel_resource_ids[..channel_count.min(MAX_CHANNELS)];
            if params.id as usize >= count
                || channel_count == 0
                || channel_count > MAX_CHANNELS
                || ids.iter().any(|&id| id as usize >= states.len())
            {
                tracing::error!(
                    id = params.id,
                    channel_count,
                    "voice outside the renderer's tables"
                );
                return Err(INVALID_UPDATE_INFO);
            }

            let info = &mut infos[params.id as usize];
            if params.is_new != 0 {
                info.initialize();
                for &id in ids {
                    states[id as usize] = VoiceState::default();
                }
            }

            let mut update_error = ErrorInfo::default();
            info.update_parameters(&mut update_error, &params, &mapper, behavior);
            info.use_float_biquads = float_biquads.is_some();
            if let Some(float_biquads) = float_biquads {
                info.biquads_float = float_biquads;
            }
            if update_error.is_error() {
                behavior.append_error(update_error);
            }

            let mut wave_errors = [[ErrorInfo::default(); 2]; MAX_WAVE_BUFFERS];
            info.update_wave_buffers(&mut wave_errors, &params, states, ids, &mapper, behavior);
            for error in wave_errors.iter().flatten() {
                if error.is_error() {
                    behavior.append_error(*error);
                }
            }

            let mut out = VoiceOutStatus::zeroed();
            info.write_out_status(&mut out, &params, &states[ids[0] as usize]);
            self.write(i * out_size, &out);
            new_voice_count += params.channel_count;
        }

        let consumed = count * stride;
        if self.in_header.voices_size as usize != consumed {
            return invalid("voices", self.in_header.voices_size, consumed);
        }
        self.out_header.voices_size = (count * out_size) as u32;
        self.finish_output(count * out_size);
        self.in_offset += consumed;
        voices.set_active_count(new_voice_count);
        Ok(())
    }

    pub fn update_effects(
        &mut self,
        effects: &mut EffectContext,
        dsp_states: &mut [EffectResultState],
        renderer_active: bool,
        pools: &[MemoryPoolInfo],
        behavior: &mut BehaviorInfo,
    ) -> HleResult<()> {
        let mapper = PoolMapper::new(pools, behavior.is_memory_force_mapping_enabled());
        let version2 = behavior.supports(SupportTag::EffectInfoVer2);
        let in_size = size_of::<EffectInParameter>();
        let out_size = if version2 {
            size_of::<EffectOutStatusV2>()
        } else {
            size_of::<EffectOutStatusV1>()
        };
        let count = effects.count() as usize;
        self.ensure_output(count * out_size)?;

        let (infos, cpu_states) = effects.parts_mut();
        for (i, info) in infos.iter_mut().enumerate() {
            let params: EffectInParameter = self.read(i * in_size)?;
            let effect_type = EffectType::from_raw(params.effect_type);
            if Some(info.effect_type) != effect_type {
                *info = EffectInfo::new(effect_type.unwrap_or_default());
            }

            let mut error = ErrorInfo::default();
            info.update(&mut error, &params, &mapper, pools, version2);
            if error.is_error() {
                behavior.append_error(error);
            }

            let state = info.store_status(renderer_active);
            if !version2 {
                let out = EffectOutStatusV1 {
                    state,
                    ..Default::default()
                };
                self.write(i * out_size, &out);
                continue;
            }
            let (Some(cpu_state), Some(dsp_state)) = (cpu_states.get_mut(i), dsp_states.get_mut(i))
            else {
                return Err(INVALID_UPDATE_INFO);
            };
            if params.is_new != 0 {
                info.initialize_result_state(dsp_state);
                info.initialize_result_state(cpu_state);
            }
            let mut out = EffectOutStatusV2::zeroed();
            out.state = state;
            info.update_result_state(&mut out.result_state, cpu_state);
            self.write(i * out_size, &out);
        }

        let consumed = count * in_size;
        if self.in_header.effects_size as usize != consumed {
            return invalid("effects", self.in_header.effects_size, consumed);
        }
        self.out_header.effects_size = (count * out_size) as u32;
        self.finish_output(count * out_size);
        self.in_offset += consumed;
        Ok(())
    }

    pub fn update_splitter_info(&mut self, splitters: &mut SplitterContext) -> HleResult<()> {
        let input = self.input.get(self.in_offset..).unwrap_or_default();
        if !splitters.update(input) {
            return Err(INVALID_UPDATE_INFO);
        }
        self.in_offset = self
            .in_offset
            .saturating_add(self.in_header.splitter_size as usize);
        Ok(())
    }

    pub fn update_mixes(
        &mut self,
        mixes: &mut MixContext,
        mix_buffer_count: u32,
        effects: &EffectContext,
        splitters: &SplitterContext,
        behavior: &BehaviorInfo,
    ) -> HleResult<()> {
        let dirty_only = behavior.supports(SupportTag::MixInParameterDirtyOnlyUpdate);
        let mut consumed = 0usize;
        let mix_count = if dirty_only {
            let dirty: MixInDirtyParameter = self.read(0)?;
            if !(0..=0x100).contains(&dirty.count) {
                tracing::error!(count = dirty.count, "invalid dirty mix count");
                return Err(INVALID_UPDATE_INFO);
            }
            consumed += size_of::<MixInDirtyParameter>();
            self.in_offset += size_of::<MixInDirtyParameter>();
            dirty.count as usize
        } else {
            mixes.count() as usize
        };
        let in_size = size_of::<MixInParameter>();
        consumed += mix_count * in_size;

        if mix_buffer_count == 0 {
            return Err(INVALID_UPDATE_INFO);
        }

        let mut total_buffer_count = 0u32;
        for i in 0..mix_count {
            let params: MixInParameter = self.read(i * in_size)?;
            if params.in_use == 0 {
                continue;
            }
            total_buffer_count = total_buffer_count.wrapping_add(params.buffer_count);
            if params.dest_mix_id > mixes.count() as i32
                && params.dest_mix_id != UNUSED_MIX_ID
                && params.mix_id != FINAL_MIX_ID
            {
                return Err(INVALID_UPDATE_INFO);
            }
        }
        if total_buffer_count > mix_buffer_count {
            return Err(INVALID_UPDATE_INFO);
        }

        let mut mix_dirty = false;
        for i in 0..mix_count {
            let params: MixInParameter = self.read(i * in_size)?;
            let mix_id = if dirty_only {
                params.mix_id as usize
            } else {
                i
            };
            if mix_id >= mixes.count() as usize {
                tracing::error!(mix_id = params.mix_id, "mix outside the renderer's table");
                return Err(INVALID_UPDATE_INFO);
            }
            let in_use = params.in_use != 0;
            let info = mixes.info_mut(mix_id);
            if info.in_use != in_use {
                info.in_use = in_use;
                if !in_use {
                    mixes.clear_effect_order(mix_id);
                }
                mix_dirty = true;
            }
            if in_use {
                mix_dirty |= mixes.update(mix_id, &params, effects, splitters, behavior);
            }
        }

        if mix_dirty {
            if behavior.supports(SupportTag::Splitter) && splitters.using_splitter() {
                if !mixes.tsort_info(splitters) {
                    return Err(INVALID_UPDATE_INFO);
                }
            } else {
                mixes.sort_info();
            }
        }

        if self.in_header.mix_size as usize != consumed {
            return invalid("mixes", self.in_header.mix_size, consumed);
        }
        self.in_offset += mix_count * in_size;
        Ok(())
    }

    pub fn update_sinks(
        &mut self,
        sinks: &mut SinkContext,
        upsamplers: &mut UpsamplerManager,
        pools: &[MemoryPoolInfo],
        behavior: &mut BehaviorInfo,
    ) -> HleResult<()> {
        let mapper = PoolMapper::new(pools, behavior.is_memory_force_mapping_enabled());
        let in_size = size_of::<SinkInParameter>();
        let out_size = size_of::<SinkOutStatus>();
        let count = sinks.count() as usize;
        self.ensure_output(count * out_size)?;
        for i in 0..count {
            let params: SinkInParameter = self.read(i * in_size)?;
            let info = sinks.info_mut(i);
            if info.sink_type() as u8 != params.sink_type {
                info.reset(params.sink_type, upsamplers);
            }
            let mut error = ErrorInfo::default();
            let mut out = SinkOutStatus::default();
            info.update(&mut error, &mut out, &params, &mapper);
            self.write(i * out_size, &out);
            if error.is_error() {
                behavior.append_error(error);
            }
        }
        let consumed = count * in_size;
        if self.in_header.sinks_size as usize != consumed {
            return invalid("sinks", self.in_header.sinks_size, consumed);
        }
        self.in_offset += consumed;
        self.out_header.sinks_size = (count * out_size) as u32;
        self.finish_output(count * out_size);
        Ok(())
    }

    /// Writes the performance history into `performance` and its size into the output.
    pub fn update_performance_buffer(
        &mut self,
        performance: &mut [u8],
        manager: Option<&mut PerformanceManager>,
    ) -> HleResult<()> {
        let params: PerformanceInParameter = self.read(0)?;
        let out_size = size_of::<PerformanceOutStatus>();
        self.ensure_output(out_size)?;
        let mut out = PerformanceOutStatus::zeroed();
        if let Some(manager) = manager {
            out.history_size = manager.copy_histories(performance) as i32;
            manager.set_detail_target(params.target_node_id as u32);
        }
        self.write(0, &out);
        let consumed = size_of::<PerformanceInParameter>();
        if self.in_header.performance_buffer_size as usize != consumed {
            return invalid(
                "performance",
                self.in_header.performance_buffer_size,
                consumed,
            );
        }
        self.in_offset += consumed;
        self.out_header.performance_buffer_size = out_size as u32;
        self.finish_output(out_size);
        Ok(())
    }

    pub fn update_error_info(&mut self, behavior: &BehaviorInfo) -> HleResult<()> {
        let out_size = size_of::<BehaviorOutStatus>();
        self.ensure_output(out_size)?;
        let mut out = BehaviorOutStatus::zeroed();
        behavior.copy_error_info(&mut out);
        self.write(0, &out);
        self.out_header.behavior_size = out_size as u32;
        self.finish_output(out_size);
        Ok(())
    }

    pub fn update_renderer_info(&mut self, elapsed_frames: u64) -> HleResult<()> {
        let out_size = size_of::<RenderInfoOutStatus>();
        self.ensure_output(out_size)?;
        let out = RenderInfoOutStatus {
            elapsed_frames,
            ..Default::default()
        };
        self.write(0, &out);
        self.out_header.render_info_size = out_size as u32;
        self.finish_output(out_size);
        Ok(())
    }

    /// Both buffers must be consumed exactly.
    pub fn check_consumed_size(&self) -> HleResult<()> {
        if self.in_offset != self.input.len() || self.out_offset != self.output.len() {
            tracing::error!(
                input = self.in_offset,
                input_size = self.input.len(),
                output = self.out_offset,
                output_size = self.output.len(),
                "audio update buffers not consumed exactly"
            );
            return Err(INVALID_UPDATE_INFO);
        }
        Ok(())
    }
}
