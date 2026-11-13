//! Effects as the renderer tracks them: type, routing, usage state, mapped work buffers and the
//! game's parameter bytes.
//!
//! The parameter stays the raw bytes the game sent; the DSP side owns the typed layouts. The
//! update logic here reads only the few fields it must act on (channel counts, the parameter
//! state byte, aux buffer addresses), at the offsets the reference layouts give them.

use bytemuck::{Pod, Zeroable};

use super::behavior::ErrorInfo;
use super::memory::{AddressInfo, MemoryPoolInfo, PoolMapper};
use super::{INVALID_PROCESS_ORDER, UNUSED_MIX_ID, is_channel_count_valid};

/// Bytes of the effect-specific parameter block.
pub const EFFECT_PARAMETER_SIZE: usize = 0xA0;
/// Bytes of DSP state an effect keeps between frames.
pub const EFFECT_STATE_SIZE: usize = 0x500;
/// Bytes of a result state (revision 9+).
pub const EFFECT_RESULT_STATE_SIZE: usize = 0x80;

/// Effect type tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum EffectType {
    #[default]
    Invalid = 0,
    Mix = 1,
    Aux = 2,
    Delay = 3,
    Reverb = 4,
    I3dl2Reverb = 5,
    BiquadFilter = 6,
    LightLimiter = 7,
    Capture = 8,
    Compressor = 9,
}

impl EffectType {
    pub const fn from_raw(raw: u8) -> Option<Self> {
        Some(match raw {
            0 => Self::Invalid,
            1 => Self::Mix,
            2 => Self::Aux,
            3 => Self::Delay,
            4 => Self::Reverb,
            5 => Self::I3dl2Reverb,
            6 => Self::BiquadFilter,
            7 => Self::LightLimiter,
            8 => Self::Capture,
            9 => Self::Compressor,
            _ => return None,
        })
    }
}

/// Whether the DSP has picked up the effect.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum UsageState {
    #[default]
    Invalid,
    New,
    Enabled,
    Disabled,
}

/// Status reported back to the game.
pub mod out_status {
    pub const INVALID: u8 = 0;
    pub const NEW: u8 = 1;
    pub const INITIALIZED: u8 = 2;
    pub const USED: u8 = 3;
    pub const REMOVED: u8 = 4;
}

/// Values of the parameter state byte.
pub mod parameter_state {
    pub const INITIALIZED: u8 = 0;
    pub const UPDATING: u8 = 1;
    pub const UPDATED: u8 = 2;
}

/// Effect input. Versions 1 and 2 share this layout; they differ in the output section.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EffectInParameter {
    pub effect_type: u8,
    pub is_new: u8,
    pub enabled: u8,
    pub _pad: u8,
    pub mix_id: u32,
    pub workbuffer: u64,
    pub workbuffer_size: u64,
    pub process_order: u32,
    pub _unk_1c: [u8; 4],
    pub specific: [u8; EFFECT_PARAMETER_SIZE],
}
const _: () = assert!(size_of::<EffectInParameter>() == 0xC0);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EffectOutStatusV1 {
    pub state: u8,
    pub _unk_01: [u8; 0xF],
}
const _: () = assert!(size_of::<EffectOutStatusV1>() == 0x10);

/// Result state bytes; the light limiter keeps its statistics here.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct EffectResultState {
    pub state: [u8; EFFECT_RESULT_STATE_SIZE],
}

impl Default for EffectResultState {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EffectOutStatusV2 {
    pub state: u8,
    pub _unk_01: [u8; 0xF],
    pub result_state: EffectResultState,
}
const _: () = assert!(size_of::<EffectOutStatusV2>() == 0x90);

/// Offsets inside the parameter block of the fields the update logic reads.
mod layout {
    pub const CHANNEL_COUNT_MAX: usize = 0x0C;
    pub const CHANNEL_COUNT: usize = 0x0E;
    pub const DELAY_STATE: usize = 0x34;
    pub const REVERB_STATE: usize = 0x40;
    pub const I3DL2_STATE: usize = 0x48;
    pub const LIMITER_STATE: usize = 0x40;
    pub const LIMITER_STATISTICS_RESET_REQUIRED: usize = 0x42;
    pub const BIQUAD_V1_STATE: usize = 0x17;
    pub const BIQUAD_V2_STATE: usize = 0x25;
    pub const COMPRESSOR_STATE: usize = 0x34;
    pub const AUX_COUNT_MAX: usize = 0x38;
    pub const AUX_SEND_BUFFER_INFO: usize = 0x40;
    pub const AUX_RETURN_BUFFER_INFO: usize = 0x50;

    pub const AUX_PARAMETER_SIZE: usize = 0x70;
    pub const MIX_PARAMETER_SIZE: usize = 0x94;
    pub const DELAY_PARAMETER_SIZE: usize = 0x38;
    pub const REVERB_PARAMETER_SIZE: usize = 0x44;
    pub const I3DL2_PARAMETER_SIZE: usize = 0x4C;
    pub const LIMITER_PARAMETER_SIZE: usize = 0x44;
    pub const BIQUAD_V1_PARAMETER_SIZE: usize = 0x18;
    pub const BIQUAD_V2_PARAMETER_SIZE: usize = 0x28;
    pub const COMPRESSOR_PARAMETER_SIZE: usize = 0x38;

    /// `AuxInfoDsp`: read/write offsets the DSP and the game share.
    pub const AUX_INFO_DSP_SIZE: u64 = 0x40;
    /// `AuxBufferInfo`: the CPU and DSP `AuxInfoDsp` pair before the sample ring.
    pub const AUX_BUFFER_INFO_SIZE: u64 = 0x80;
}

fn u16_at(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

/// One effect.
#[derive(Clone, Debug)]
pub struct EffectInfo {
    pub effect_type: EffectType,
    pub enabled: bool,
    pub buffer_unmapped: bool,
    pub usage_state: UsageState,
    pub mix_id: i32,
    pub process_order: i32,
    pub workbuffers: [AddressInfo; 2],
    pub send_buffer_info: u64,
    pub send_buffer: u64,
    pub return_buffer_info: u64,
    pub return_buffer: u64,
    /// The game's effect-specific parameter bytes, with the renderer's state byte updates.
    pub parameter: [u8; EFFECT_PARAMETER_SIZE],
    /// DSP state carried between frames, laid out by the effect's DSP implementation.
    pub state: [u8; EFFECT_STATE_SIZE],
}

impl Default for EffectInfo {
    fn default() -> Self {
        Self::new(EffectType::Invalid)
    }
}

impl EffectInfo {
    pub fn new(effect_type: EffectType) -> Self {
        Self {
            effect_type,
            enabled: false,
            buffer_unmapped: false,
            usage_state: UsageState::Invalid,
            mix_id: UNUSED_MIX_ID,
            process_order: INVALID_PROCESS_ORDER,
            workbuffers: [AddressInfo::default(); 2],
            send_buffer_info: 0,
            send_buffer: 0,
            return_buffer_info: 0,
            return_buffer: 0,
            parameter: [0; EFFECT_PARAMETER_SIZE],
            state: [0; EFFECT_STATE_SIZE],
        }
    }

    pub fn should_skip(&self) -> bool {
        self.buffer_unmapped
    }

    pub(crate) fn store_status(&self, renderer_active: bool) -> u8 {
        let used = if renderer_active {
            self.usage_state != UsageState::Disabled
        } else {
            self.usage_state == UsageState::New
        };
        if used {
            out_status::USED
        } else {
            out_status::REMOVED
        }
    }

    fn copy_parameter(&mut self, params: &EffectInParameter, size: usize) {
        self.parameter[..size].copy_from_slice(&params.specific[..size]);
        self.mix_id = params.mix_id as i32;
        self.process_order = params.process_order as i32;
        self.enabled = params.enabled != 0;
    }

    /// Delay, reverb and I3DL2 reverb share this update; only the state byte moves.
    fn update_channel_effect(
        &mut self,
        error: &mut ErrorInfo,
        params: &EffectInParameter,
        mapper: &PoolMapper,
        size: usize,
        state_offset: usize,
    ) {
        let specific = &params.specific;
        if is_channel_count_valid(u16_at(specific, layout::CHANNEL_COUNT_MAX)) {
            let old_state = self.parameter[state_offset];
            self.copy_parameter(params, size);
            let channel_count_valid =
                is_channel_count_valid(u16_at(specific, layout::CHANNEL_COUNT));
            if !channel_count_valid {
                let max = [
                    specific[layout::CHANNEL_COUNT_MAX],
                    specific[layout::CHANNEL_COUNT_MAX + 1],
                ];
                self.parameter[layout::CHANNEL_COUNT..layout::CHANNEL_COUNT + 2]
                    .copy_from_slice(&max);
            }
            if !channel_count_valid || old_state != parameter_state::UPDATED {
                self.parameter[state_offset] = old_state;
            }
            if self.buffer_unmapped || params.is_new != 0 {
                self.usage_state = UsageState::New;
                self.parameter[state_offset] = parameter_state::INITIALIZED;
                self.buffer_unmapped = !mapper.try_attach_buffer(
                    error,
                    &mut self.workbuffers[0],
                    params.workbuffer,
                    params.workbuffer_size,
                );
                return;
            }
        }
        error.set_success();
    }

    fn update_aux(
        &mut self,
        error: &mut ErrorInfo,
        params: &EffectInParameter,
        mapper: &PoolMapper,
        pools: &[MemoryPoolInfo],
        has_return: bool,
    ) {
        self.copy_parameter(params, layout::AUX_PARAMETER_SIZE);
        if !(self.buffer_unmapped || params.is_new != 0) {
            error.set_success();
            return;
        }
        let specific = &params.specific;
        let buffer_size =
            layout::AUX_BUFFER_INFO_SIZE + u32_at(specific, layout::AUX_COUNT_MAX) as u64 * 4;
        let send_unmapped = !mapper.try_attach_buffer(
            error,
            &mut self.workbuffers[0],
            u64_at(specific, layout::AUX_SEND_BUFFER_INFO),
            buffer_size,
        );
        let return_unmapped = has_return
            && !mapper.try_attach_buffer(
                error,
                &mut self.workbuffers[1],
                u64_at(specific, layout::AUX_RETURN_BUFFER_INFO),
                buffer_size,
            );
        self.buffer_unmapped = send_unmapped || return_unmapped;
        if self.buffer_unmapped {
            return;
        }
        let send = self.workbuffers[0].reference(pools, false);
        self.send_buffer_info = send + layout::AUX_INFO_DSP_SIZE;
        self.send_buffer = send + layout::AUX_BUFFER_INFO_SIZE;
        if has_return {
            let ret = self.workbuffers[1].reference(pools, false);
            self.return_buffer_info = ret + layout::AUX_INFO_DSP_SIZE;
            self.return_buffer = ret + layout::AUX_BUFFER_INFO_SIZE;
        } else {
            self.return_buffer_info = 0;
            self.return_buffer = 0;
        }
    }

    /// Applies the game's input. `version2` selects the revision 9+ parameter layouts.
    pub fn update(
        &mut self,
        error: &mut ErrorInfo,
        params: &EffectInParameter,
        mapper: &PoolMapper,
        pools: &[MemoryPoolInfo],
        version2: bool,
    ) {
        use layout::*;
        match self.effect_type {
            EffectType::Invalid => error.set_success(),
            EffectType::Mix => {
                self.copy_parameter(params, MIX_PARAMETER_SIZE);
                error.set_success();
            }
            EffectType::Aux => self.update_aux(error, params, mapper, pools, true),
            EffectType::Capture => self.update_aux(error, params, mapper, pools, false),
            EffectType::Delay => {
                self.update_channel_effect(error, params, mapper, DELAY_PARAMETER_SIZE, DELAY_STATE)
            }
            EffectType::Reverb => self.update_channel_effect(
                error,
                params,
                mapper,
                REVERB_PARAMETER_SIZE,
                REVERB_STATE,
            ),
            EffectType::I3dl2Reverb => {
                self.update_channel_effect(error, params, mapper, I3DL2_PARAMETER_SIZE, I3DL2_STATE)
            }
            EffectType::BiquadFilter => {
                let size = if version2 {
                    BIQUAD_V2_PARAMETER_SIZE
                } else {
                    BIQUAD_V1_PARAMETER_SIZE
                };
                self.copy_parameter(params, size);
                error.set_success();
            }
            EffectType::LightLimiter => {
                self.copy_parameter(params, LIMITER_PARAMETER_SIZE);
                if self.buffer_unmapped || params.is_new != 0 {
                    self.usage_state = UsageState::New;
                    self.parameter[LIMITER_STATE] = parameter_state::INITIALIZED;
                    self.buffer_unmapped = !mapper.try_attach_buffer(
                        error,
                        &mut self.workbuffers[0],
                        params.workbuffer,
                        params.workbuffer_size,
                    );
                } else {
                    error.set_success();
                }
            }
            EffectType::Compressor => {
                if version2 {
                    self.copy_parameter(params, COMPRESSOR_PARAMETER_SIZE);
                    error.set_success();
                }
            }
        }
    }

    /// Marks the effect picked up by a command list and its parameters consumed.
    pub fn update_for_command_generation(&mut self) {
        use layout::*;
        if self.effect_type == EffectType::Invalid {
            return;
        }
        self.usage_state = if self.enabled {
            UsageState::Enabled
        } else {
            UsageState::Disabled
        };
        match self.effect_type {
            EffectType::Delay => self.parameter[DELAY_STATE] = parameter_state::UPDATED,
            EffectType::Reverb => self.parameter[REVERB_STATE] = parameter_state::UPDATED,
            EffectType::I3dl2Reverb => self.parameter[I3DL2_STATE] = parameter_state::UPDATED,
            EffectType::Compressor => self.parameter[COMPRESSOR_STATE] = parameter_state::UPDATED,
            EffectType::LightLimiter => {
                self.parameter[LIMITER_STATE] = parameter_state::UPDATED;
                self.parameter[LIMITER_STATISTICS_RESET_REQUIRED] = 0;
            }
            EffectType::BiquadFilter => {
                if self.parameter[BIQUAD_V1_STATE] <= parameter_state::UPDATED {
                    self.parameter[BIQUAD_V1_STATE] = parameter_state::UPDATED;
                } else if self.parameter[BIQUAD_V2_STATE] <= parameter_state::UPDATED {
                    self.parameter[BIQUAD_V2_STATE] = parameter_state::UPDATED;
                } else {
                    self.parameter[BIQUAD_V1_STATE] = parameter_state::UPDATED;
                    self.parameter[BIQUAD_V2_STATE] = parameter_state::UPDATED;
                }
            }
            _ => {}
        }
    }

    /// Clears a result state for a new effect. Only the light limiter keeps one: per-channel
    /// peaks at 0 and minimum compression gains at 1.0.
    pub fn initialize_result_state(&self, result_state: &mut EffectResultState) {
        if self.effect_type != EffectType::LightLimiter {
            return;
        }
        result_state.state[..0x18].fill(0);
        for gain in result_state.state[0x18..0x30].as_chunks_mut::<4>().0 {
            *gain = 1.0f32.to_le_bytes();
        }
    }

    /// Copies the light limiter statistics from `source` into `target`.
    pub fn update_result_state(&self, target: &mut EffectResultState, source: &EffectResultState) {
        if self.effect_type == EffectType::LightLimiter {
            target.state[..0x30].copy_from_slice(&source.state[..0x30]);
        }
    }

    fn single_buffer(&self, pools: &[MemoryPoolInfo]) -> u64 {
        if self.enabled {
            return self.workbuffers[0].reference(pools, true);
        }
        0
    }

    /// DSP address of work buffer `index`, marking its pool used.
    pub fn workbuffer(&self, index: usize, pools: &[MemoryPoolInfo]) -> u64 {
        match self.effect_type {
            EffectType::Aux | EffectType::Capture => self.workbuffers[index].reference(pools, true),
            EffectType::Delay
            | EffectType::Reverb
            | EffectType::I3dl2Reverb
            | EffectType::LightLimiter
            | EffectType::Compressor => self.single_buffer(pools),
            EffectType::Invalid | EffectType::Mix | EffectType::BiquadFilter => 0,
        }
    }
}

/// All effects and their result states.
#[derive(Clone, Debug, Default)]
pub struct EffectContext {
    infos: Vec<EffectInfo>,
    result_states_cpu: Vec<EffectResultState>,
    result_states_dsp: Vec<EffectResultState>,
}

impl EffectContext {
    /// `count` effects; result states exist only from revision 9 (`with_result_states`).
    pub fn new(count: u32, with_result_states: bool) -> Self {
        let states = if with_result_states {
            count as usize
        } else {
            0
        };
        Self {
            infos: (0..count).map(|_| EffectInfo::default()).collect(),
            result_states_cpu: vec![EffectResultState::default(); states],
            result_states_dsp: vec![EffectResultState::default(); states],
        }
    }

    pub fn count(&self) -> u32 {
        self.infos.len() as u32
    }

    pub fn info(&self, index: usize) -> &EffectInfo {
        &self.infos[index]
    }

    pub fn info_mut(&mut self, index: usize) -> &mut EffectInfo {
        &mut self.infos[index]
    }

    pub fn infos(&self) -> &[EffectInfo] {
        &self.infos
    }

    pub fn result_state(&self, index: usize) -> &EffectResultState {
        &self.result_states_cpu[index]
    }

    /// Result states the DSP writes while rendering.
    pub fn dsp_result_states_mut(&mut self) -> &mut [EffectResultState] {
        &mut self.result_states_dsp
    }

    pub(crate) fn parts_mut(
        &mut self,
    ) -> (
        &mut [EffectInfo],
        &mut [EffectResultState],
        &mut [EffectResultState],
    ) {
        (
            &mut self.infos,
            &mut self.result_states_cpu,
            &mut self.result_states_dsp,
        )
    }

    /// Copies DSP result states into the host copies after a command list.
    pub fn update_state_by_dsp_shared(&mut self) {
        for ((info, cpu), dsp) in self
            .infos
            .iter()
            .zip(&mut self.result_states_cpu)
            .zip(&self.result_states_dsp)
        {
            info.update_result_state(cpu, dsp);
        }
    }
}
