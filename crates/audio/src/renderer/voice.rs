//! Voices: wave buffer sources the game plays, their per-channel DSP state and the channel
//! resources that hold their mix volumes.

use bytemuck::{Pod, Zeroable};

use super::behavior::{BehaviorInfo, ErrorInfo};
use super::memory::{AddressInfo, MemoryPoolInfo, PoolMapper};
use super::result::INVALID_UPDATE_INFO;
use super::revision::SupportTag;
use super::{
    LOWEST_VOICE_PRIORITY, MAX_BIQUAD_FILTERS, MAX_CHANNELS, MAX_MIX_BUFFERS, MAX_WAVE_BUFFERS,
    PlayState, SampleFormat, SrcQuality, UNUSED_MIX_ID, UNUSED_SPLITTER_ID,
};

/// Wave buffer as the game describes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct WaveBufferInternal {
    pub address: u64,
    pub size: u64,
    pub start_offset: i32,
    pub end_offset: i32,
    pub looping: u8,
    pub stream_ended: u8,
    pub sent_to_dsp: u8,
    pub _pad: u8,
    pub loop_count: i32,
    pub context_address: u64,
    pub context_size: u64,
    pub loop_start: u32,
    pub loop_end: u32,
}
const _: () = assert!(size_of::<WaveBufferInternal>() == 0x38);

/// Fixed point (Q2.14) biquad coefficients.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BiquadFilterParameter {
    pub enabled: u8,
    pub _pad: u8,
    pub b: [i16; 3],
    pub a: [i16; 2],
}
const _: () = assert!(size_of::<BiquadFilterParameter>() == 0xC);

/// Float biquad coefficients (revision 12 splitters, revision 15 voices). `a0` is 1.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct BiquadFilterParameter2 {
    pub enabled: u8,
    pub _reserved: [u8; 3],
    pub numerator: [f32; 3],
    pub denominator: [f32; 2],
}
const _: () = assert!(size_of::<BiquadFilterParameter2>() == 0x18);

/// Bit 0: reset the played sample count at loop points. Bit 1: skip pitch and SRC.
pub mod voice_flags {
    pub const PLAYED_SAMPLE_COUNT_RESET_AT_LOOP_POINT: u16 = 1;
    pub const PITCH_AND_SRC_SKIPPED: u16 = 2;
}

/// Voice input before revision 15.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VoiceInParameter {
    pub id: u32,
    pub node_id: u32,
    pub is_new: u8,
    pub in_use: u8,
    pub play_state: u8,
    pub sample_format: u8,
    pub sample_rate: u32,
    pub priority: i32,
    pub sort_order: i32,
    pub channel_count: u32,
    pub pitch: f32,
    pub volume: f32,
    pub biquads: [BiquadFilterParameter; MAX_BIQUAD_FILTERS],
    pub wave_buffer_count: u32,
    pub wave_buffer_index: u16,
    pub _unk_042: [u8; 6],
    pub src_data_address: u64,
    pub src_data_size: u64,
    pub mix_id: u32,
    pub splitter_id: u32,
    pub wave_buffers: [WaveBufferInternal; MAX_WAVE_BUFFERS],
    pub channel_resource_ids: [u32; MAX_CHANNELS],
    pub clear_voice_drop: u8,
    pub flush_buffer_count: u8,
    pub _unk_15a: [u8; 2],
    pub flags: u8,
    pub _unk_15d: u8,
    pub src_quality: u8,
    pub _unk_15f: [u8; 0x11],
}
const _: () = assert!(size_of::<VoiceInParameter>() == 0x170);

/// Voice input from revision 15: float biquads and a wider wave buffer index.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VoiceInParameterV2 {
    pub id: u32,
    pub node_id: u32,
    pub is_new: u8,
    pub in_use: u8,
    pub play_state: u8,
    pub sample_format: u8,
    pub sample_rate: u32,
    pub priority: u32,
    pub sort_order: u32,
    pub channel_count: u32,
    pub pitch: f32,
    pub volume: f32,
    pub biquads: [BiquadFilterParameter2; MAX_BIQUAD_FILTERS],
    pub wave_buffer_count: u32,
    pub wave_buffer_index: u32,
    pub _reserved1: u32,
    pub src_data_address: u64,
    pub src_data_size: u64,
    pub mix_id: i32,
    pub splitter_id: u32,
    pub wave_buffers: [WaveBufferInternal; MAX_WAVE_BUFFERS],
    pub channel_resource_ids: [u32; MAX_CHANNELS],
    pub clear_voice_drop: u8,
    pub flush_buffer_count: u8,
    pub _reserved2: u16,
    pub flags: u8,
    pub _reserved3: u8,
    pub src_quality: u8,
    pub crossfade_curve: u8,
    pub external_context: u32,
    pub external_context_size: u32,
    pub _reserved4: [u32; 2],
}
const _: () = assert!(size_of::<VoiceInParameterV2>() == 0x188);
const _: () = assert!(std::mem::offset_of!(VoiceInParameterV2, crossfade_curve) == 0x177);

fn to_q14(value: f32) -> i16 {
    (value * 16384.0).clamp(-32768.0, 32767.0) as i16
}

impl VoiceInParameterV2 {
    /// The pre-revision 15 layout with Q2.14 biquads, plus the float biquads.
    pub fn to_v1(
        &self,
    ) -> (
        VoiceInParameter,
        [BiquadFilterParameter2; MAX_BIQUAD_FILTERS],
    ) {
        let mut v1 = VoiceInParameter::zeroed();
        v1.id = self.id;
        v1.node_id = self.node_id;
        v1.is_new = self.is_new;
        v1.in_use = self.in_use;
        v1.play_state = self.play_state;
        v1.sample_format = self.sample_format;
        v1.sample_rate = self.sample_rate;
        v1.priority = self.priority as i32;
        v1.sort_order = self.sort_order as i32;
        v1.channel_count = self.channel_count;
        v1.pitch = self.pitch;
        v1.volume = self.volume;
        for (dst, src) in v1.biquads.iter_mut().zip(&self.biquads) {
            dst.enabled = src.enabled;
            dst.b = src.numerator.map(to_q14);
            dst.a = src.denominator.map(to_q14);
        }
        v1.wave_buffer_count = self.wave_buffer_count;
        v1.wave_buffer_index = self.wave_buffer_index as u16;
        v1.src_data_address = self.src_data_address;
        v1.src_data_size = self.src_data_size;
        v1.mix_id = self.mix_id as u32;
        v1.splitter_id = self.splitter_id;
        v1.wave_buffers = self.wave_buffers;
        v1.channel_resource_ids = self.channel_resource_ids;
        v1.clear_voice_drop = self.clear_voice_drop;
        v1.flush_buffer_count = self.flush_buffer_count;
        v1.flags = self.flags;
        v1.src_quality = self.src_quality;
        (v1, self.biquads)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct VoiceOutStatus {
    pub played_sample_count: u64,
    pub wave_buffers_consumed: u32,
    pub voice_dropped: u8,
    pub _pad: [u8; 3],
}
const _: () = assert!(size_of::<VoiceOutStatus>() == 0x10);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VoiceChannelResourceInParameter {
    pub id: u32,
    pub mix_volumes: [f32; MAX_MIX_BUFFERS],
    pub in_use: u8,
    pub _unk_65: [u8; 0xB],
}
const _: () = assert!(size_of::<VoiceChannelResourceInParameter>() == 0x70);

/// Mix volumes of one voice channel.
#[derive(Clone, Copy, Debug, Default)]
pub struct VoiceChannelResource {
    pub id: u32,
    pub mix_volumes: [f32; MAX_MIX_BUFFERS],
    pub prev_mix_volumes: [f32; MAX_MIX_BUFFERS],
    pub in_use: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BiquadFilterState {
    pub s0: i64,
    pub s1: i64,
    pub s2: i64,
    pub s3: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AdpcmContext {
    pub header: u16,
    pub yn0: i16,
    pub yn1: i16,
}

/// Per-channel decode state. The DSP updates the shared copy while rendering; the renderer
/// copies it back after every command list.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VoiceState {
    pub played_sample_count: u64,
    pub offset: u32,
    pub wave_buffer_index: u32,
    pub wave_buffer_valid: [bool; MAX_WAVE_BUFFERS],
    pub wave_buffers_consumed: u32,
    pub sample_history: [i16; MAX_WAVE_BUFFERS * 2],
    /// Resampling position, Q49.15.
    pub fraction: i64,
    pub adpcm_context: AdpcmContext,
    pub biquad_states: [[BiquadFilterState; MAX_BIQUAD_FILTERS]; MAX_BIQUAD_FILTERS],
    pub previous_samples: [i32; MAX_MIX_BUFFERS],
    pub external_context_size: u32,
    pub external_context_enabled: bool,
    pub voice_dropped: bool,
    pub loop_count: i32,
}

/// Play state as the renderer tracks it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ServerPlayState {
    Started,
    #[default]
    Stopped,
    RequestStop,
    Paused,
}

/// A wave buffer the renderer holds for a voice.
#[derive(Clone, Copy, Debug, Default)]
pub struct WaveBuffer {
    pub buffer_address: AddressInfo,
    pub context_address: AddressInfo,
    pub start_offset: u32,
    pub end_offset: u32,
    pub looping: bool,
    pub stream_ended: bool,
    pub sent_to_dsp: bool,
    pub loop_start_offset: u32,
    pub loop_end_offset: u32,
    pub loop_count: i32,
}

impl WaveBuffer {
    fn initial() -> Self {
        Self {
            sent_to_dsp: true,
            ..Self::default()
        }
    }
}

/// Byte span of `offset` samples of ADPCM: 8-byte frames of 14 samples, a header byte, then
/// two samples per byte.
fn adpcm_byte_offset(offset: i32) -> i64 {
    let frame = (offset / 14) as i64;
    let rest = offset % 14;
    let extra = if rest == 0 {
        0
    } else {
        (rest / 2 + 1 + rest % 2) as i64
    };
    frame * 8 + extra
}

/// One voice.
#[derive(Clone, Debug)]
pub struct VoiceInfo {
    pub in_use: bool,
    pub is_new: bool,
    pub was_playing: bool,
    pub sample_format: SampleFormat,
    pub sample_rate: u32,
    pub channel_count: i8,
    pub id: u32,
    pub node_id: u32,
    pub mix_id: u32,
    pub current_play_state: ServerPlayState,
    pub last_play_state: ServerPlayState,
    pub priority: i32,
    pub sort_order: i32,
    pub pitch: f32,
    pub volume: f32,
    pub prev_volume: f32,
    pub biquads: [BiquadFilterParameter; MAX_BIQUAD_FILTERS],
    pub biquads_float: [BiquadFilterParameter2; MAX_BIQUAD_FILTERS],
    pub use_float_biquads: bool,
    pub wave_buffer_count: u32,
    pub wave_buffer_index: u16,
    /// See [`voice_flags`].
    pub flags: u16,
    pub data_address: AddressInfo,
    pub wave_buffers: [WaveBuffer; MAX_WAVE_BUFFERS],
    pub channel_resource_ids: [u32; MAX_CHANNELS],
    pub splitter_id: i32,
    pub src_quality: SrcQuality,
    pub voice_dropped: bool,
    pub data_unmapped: bool,
    pub buffer_unmapped: bool,
    pub biquad_initialized: [bool; MAX_BIQUAD_FILTERS],
    pub flush_buffer_count: u8,
}

impl Default for VoiceInfo {
    fn default() -> Self {
        Self {
            in_use: false,
            is_new: false,
            was_playing: false,
            sample_format: SampleFormat::Invalid,
            sample_rate: 0,
            channel_count: 0,
            id: 0,
            node_id: 0,
            mix_id: UNUSED_MIX_ID as u32,
            current_play_state: ServerPlayState::Stopped,
            last_play_state: ServerPlayState::Started,
            priority: LOWEST_VOICE_PRIORITY,
            sort_order: 0,
            pitch: 0.0,
            volume: 0.0,
            prev_volume: 0.0,
            biquads: Default::default(),
            biquads_float: Default::default(),
            use_float_biquads: false,
            wave_buffer_count: 0,
            wave_buffer_index: 0,
            flags: 0,
            data_address: AddressInfo::default(),
            wave_buffers: [WaveBuffer::initial(); MAX_WAVE_BUFFERS],
            channel_resource_ids: [0; MAX_CHANNELS],
            splitter_id: UNUSED_SPLITTER_ID,
            src_quality: SrcQuality::Medium,
            voice_dropped: false,
            data_unmapped: false,
            buffer_unmapped: false,
            biquad_initialized: [false; MAX_BIQUAD_FILTERS],
            flush_buffer_count: 0,
        }
    }
}

impl VoiceInfo {
    /// Resets a voice the game marks new. Fields the reference keeps across a reset (flags,
    /// sort order, play history, float biquads) survive here too.
    pub fn initialize(&mut self) {
        self.in_use = false;
        self.is_new = false;
        self.id = 0;
        self.node_id = 0;
        self.current_play_state = ServerPlayState::Stopped;
        self.src_quality = SrcQuality::Medium;
        self.priority = LOWEST_VOICE_PRIORITY;
        self.sample_format = SampleFormat::Invalid;
        self.sample_rate = 0;
        self.channel_count = 0;
        self.wave_buffer_count = 0;
        self.wave_buffer_index = 0;
        self.pitch = 0.0;
        self.volume = 0.0;
        self.prev_volume = 0.0;
        self.mix_id = UNUSED_MIX_ID as u32;
        self.splitter_id = UNUSED_SPLITTER_ID;
        self.biquads = Default::default();
        self.biquad_initialized = [false; MAX_BIQUAD_FILTERS];
        self.voice_dropped = false;
        self.data_unmapped = false;
        self.buffer_unmapped = false;
        self.flush_buffer_count = 0;
        self.data_address = AddressInfo::default();
        self.wave_buffers = [WaveBuffer::initial(); MAX_WAVE_BUFFERS];
    }

    fn should_update_parameters(&self, params: &VoiceInParameter) -> bool {
        self.data_address.cpu_address() != params.src_data_address
            || self.data_address.size() != params.src_data_size
            || self.data_unmapped
    }

    pub fn update_parameters(
        &mut self,
        error: &mut ErrorInfo,
        params: &VoiceInParameter,
        mapper: &PoolMapper,
        behavior: &BehaviorInfo,
    ) {
        self.in_use = params.in_use != 0;
        self.id = params.id;
        self.node_id = params.node_id;
        self.update_play_state(params.play_state);
        if let Some(quality) = SrcQuality::from_raw(params.src_quality) {
            self.src_quality = quality;
        }
        self.priority = params.priority;
        self.sort_order = params.sort_order;
        self.sample_rate = params.sample_rate;
        self.sample_format = SampleFormat::from_raw(params.sample_format);
        self.channel_count = params.channel_count as i8;
        self.pitch = params.pitch;
        self.volume = params.volume;
        self.biquads = params.biquads;
        self.wave_buffer_count = params.wave_buffer_count;
        self.wave_buffer_index = params.wave_buffer_index;
        if behavior.supports(SupportTag::FlushVoiceWaveBuffers) {
            self.flush_buffer_count = self
                .flush_buffer_count
                .wrapping_add(params.flush_buffer_count);
        }
        self.mix_id = params.mix_id;
        self.splitter_id = if behavior.supports(SupportTag::Splitter) {
            params.splitter_id as i32
        } else {
            UNUSED_SPLITTER_ID
        };
        self.channel_resource_ids = params.channel_resource_ids;

        self.flags &= !0b11;
        if behavior.supports(SupportTag::VoicePlayedSampleCountResetAtLoopPoint) {
            self.flags |=
                params.flags as u16 & voice_flags::PLAYED_SAMPLE_COUNT_RESET_AT_LOOP_POINT;
        }
        if behavior.supports(SupportTag::VoicePitchAndSrcSkipped) {
            self.flags |= params.flags as u16 & voice_flags::PITCH_AND_SRC_SKIPPED;
        }

        if params.clear_voice_drop != 0 {
            self.voice_dropped = false;
        }

        if self.should_update_parameters(params) {
            self.data_unmapped = !mapper.try_attach_buffer(
                error,
                &mut self.data_address,
                params.src_data_address,
                params.src_data_size,
            );
        } else {
            error.set_success();
        }
    }

    fn update_play_state(&mut self, raw: u8) {
        self.last_play_state = self.current_play_state;
        match PlayState::from_raw(raw) {
            Some(PlayState::Started) => self.current_play_state = ServerPlayState::Started,
            Some(PlayState::Stopped) => {
                if self.current_play_state != ServerPlayState::Stopped {
                    self.current_play_state = ServerPlayState::RequestStop;
                }
            }
            Some(PlayState::Paused) => self.current_play_state = ServerPlayState::Paused,
            None => {}
        }
    }

    /// Updates the wave buffers. `ids` are the voice's channel resources, already checked
    /// against `states`.
    pub fn update_wave_buffers(
        &mut self,
        errors: &mut [[ErrorInfo; 2]; MAX_WAVE_BUFFERS],
        params: &VoiceInParameter,
        states: &mut [VoiceState],
        ids: &[u32],
        mapper: &PoolMapper,
        behavior: &BehaviorInfo,
    ) {
        if params.is_new != 0 {
            self.wave_buffers = [WaveBuffer::initial(); MAX_WAVE_BUFFERS];
            for &id in ids {
                states[id as usize].wave_buffer_valid = [false; MAX_WAVE_BUFFERS];
            }
        }
        let sample_format = SampleFormat::from_raw(params.sample_format);
        for i in 0..MAX_WAVE_BUFFERS {
            let valid = states[ids[0] as usize].wave_buffer_valid[i];
            self.update_wave_buffer(
                &mut errors[i],
                i,
                &params.wave_buffers[i],
                sample_format,
                valid,
                mapper,
                behavior,
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn update_wave_buffer(
        &mut self,
        errors: &mut [ErrorInfo; 2],
        index: usize,
        internal: &WaveBufferInternal,
        sample_format: SampleFormat,
        valid: bool,
        mapper: &PoolMapper,
        behavior: &BehaviorInfo,
    ) {
        let wave_buffer = &mut self.wave_buffers[index];
        if !valid && wave_buffer.sent_to_dsp && wave_buffer.buffer_address.cpu_address() != 0 {
            wave_buffer.buffer_address.setup(0, 0);
        }

        if internal.sent_to_dsp != 0 && !self.buffer_unmapped {
            return;
        }

        let out_of_range = match sample_format {
            SampleFormat::PcmInt16 | SampleFormat::PcmFloat => {
                let byte_size = sample_format.byte_size();
                (internal.start_offset as i64 as u64).wrapping_mul(byte_size) > internal.size
                    || (internal.end_offset as i64 as u64).wrapping_mul(byte_size) > internal.size
            }
            SampleFormat::Adpcm => {
                adpcm_byte_offset(internal.start_offset) > internal.size as i64
                    || adpcm_byte_offset(internal.end_offset) > internal.size as i64
            }
            _ => false,
        };
        if out_of_range || internal.start_offset < 0 || internal.end_offset < 0 {
            errors[0].set(INVALID_UPDATE_INFO, internal.address);
            return;
        }

        wave_buffer.start_offset = internal.start_offset as u32;
        wave_buffer.end_offset = internal.end_offset as u32;
        wave_buffer.looping = internal.looping != 0;
        wave_buffer.stream_ended = internal.stream_ended != 0;
        wave_buffer.sent_to_dsp = false;
        wave_buffer.loop_start_offset = internal.loop_start;
        wave_buffer.loop_end_offset = internal.loop_end;
        wave_buffer.loop_count = internal.loop_count;

        let [buffer_error, context_error] = errors;
        let mut unmapped = !mapper.try_attach_buffer(
            buffer_error,
            &mut wave_buffer.buffer_address,
            internal.address,
            internal.size,
        );
        if sample_format == SampleFormat::Adpcm
            && behavior.supports(SupportTag::AdpcmLoopContextBugFix)
            && internal.context_address != 0
        {
            unmapped = !mapper.try_attach_buffer(
                context_error,
                &mut wave_buffer.context_address,
                internal.context_address,
                internal.context_size,
            ) || self.data_unmapped;
        } else {
            wave_buffer.context_address.setup(0, 0);
        }
        self.buffer_unmapped = unmapped;
    }

    pub fn write_out_status(
        &mut self,
        out: &mut VoiceOutStatus,
        params: &VoiceInParameter,
        first_state: &VoiceState,
    ) {
        if params.is_new != 0 {
            self.is_new = true;
        }
        if self.is_new {
            *out = VoiceOutStatus::default();
        } else {
            out.played_sample_count = first_state.played_sample_count;
            out.wave_buffers_consumed = first_state.wave_buffers_consumed;
            out.voice_dropped = self.voice_dropped as u8;
        }
    }

    pub fn should_skip(&self) -> bool {
        !self.in_use
            || self.wave_buffer_count == 0
            || self.data_unmapped
            || self.buffer_unmapped
            || self.voice_dropped
    }

    pub fn has_any_connection(&self) -> bool {
        self.mix_id != UNUSED_MIX_ID as u32 || self.splitter_id != UNUSED_SPLITTER_ID
    }

    fn channel_ids(&self) -> &[u32] {
        let count = (self.channel_count.max(0) as usize).min(MAX_CHANNELS);
        &self.channel_resource_ids[..count]
    }

    fn flush_wave_buffers(&mut self, flush_count: u8, states: &mut [VoiceState]) {
        let mut wave_index = self.wave_buffer_index as usize % MAX_WAVE_BUFFERS;
        for _ in 0..flush_count {
            self.wave_buffers[wave_index].sent_to_dsp = true;
            for &id in self.channel_ids() {
                let state = &mut states[id as usize];
                if state.wave_buffer_index as usize == wave_index {
                    state.wave_buffer_index =
                        (state.wave_buffer_index + 1) % MAX_WAVE_BUFFERS as u32;
                    state.wave_buffers_consumed += 1;
                }
                state.wave_buffer_valid[wave_index] = false;
            }
            wave_index = (wave_index + 1) % MAX_WAVE_BUFFERS;
        }
    }

    /// Applies flushes and play state changes to the DSP-shared states before a command list
    /// is built. Returns whether the voice was playing (for depop).
    fn update_parameters_for_command_generation(
        &mut self,
        states: &mut [VoiceState],
        pools: &[MemoryPoolInfo],
    ) -> bool {
        if self.flush_buffer_count > 0 {
            self.flush_wave_buffers(self.flush_buffer_count, states);
            self.flush_buffer_count = 0;
        }

        match self.current_play_state {
            ServerPlayState::Started => {
                for i in 0..MAX_WAVE_BUFFERS {
                    if !self.wave_buffers[i].sent_to_dsp {
                        for &id in self.channel_ids() {
                            states[id as usize].wave_buffer_valid[i] = true;
                        }
                        self.wave_buffers[i].sent_to_dsp = true;
                    }
                }
                self.was_playing = false;
                let first = self.channel_ids().first().map(|&id| &states[id as usize]);
                if first.is_some_and(|state| state.wave_buffer_valid.contains(&true)) {
                    return true;
                }
            }
            ServerPlayState::Stopped | ServerPlayState::Paused => {
                for wave_buffer in &self.wave_buffers {
                    if !wave_buffer.sent_to_dsp {
                        wave_buffer.buffer_address.reference(pools, true);
                        wave_buffer.context_address.reference(pools, true);
                    }
                }
                if self.sample_format == SampleFormat::Adpcm && self.data_address.cpu_address() != 0
                {
                    self.data_address.reference(pools, true);
                }
                self.was_playing = self.last_play_state == ServerPlayState::Started;
            }
            ServerPlayState::RequestStop => {
                for i in 0..MAX_WAVE_BUFFERS {
                    self.wave_buffers[i].sent_to_dsp = true;
                    for &id in self.channel_ids() {
                        let state = &mut states[id as usize];
                        if state.wave_buffer_valid[i] {
                            state.wave_buffer_index =
                                (state.wave_buffer_index + 1) % MAX_WAVE_BUFFERS as u32;
                            state.wave_buffers_consumed += 1;
                        }
                        state.wave_buffer_valid[i] = false;
                    }
                }
                for &id in self.channel_ids() {
                    let state = &mut states[id as usize];
                    state.offset = 0;
                    state.played_sample_count = 0;
                    state.adpcm_context = AdpcmContext::default();
                    state.sample_history = [0; MAX_WAVE_BUFFERS * 2];
                    state.fraction = 0;
                }
                self.current_play_state = ServerPlayState::Stopped;
                self.was_playing = self.last_play_state == ServerPlayState::Started;
            }
        }
        self.was_playing
    }
}

/// All voices of a renderer.
#[derive(Clone, Debug)]
pub struct VoiceContext {
    infos: Vec<VoiceInfo>,
    sorted: Vec<u32>,
    channel_resources: Vec<VoiceChannelResource>,
    cpu_states: Vec<VoiceState>,
    dsp_states: Vec<VoiceState>,
    active_count: u32,
}

impl VoiceContext {
    pub fn new(count: u32) -> Self {
        let count = count as usize;
        Self {
            infos: vec![VoiceInfo::default(); count],
            sorted: (0..count as u32).collect(),
            channel_resources: (0..count as u32)
                .map(|id| VoiceChannelResource {
                    id,
                    ..Default::default()
                })
                .collect(),
            cpu_states: vec![VoiceState::default(); count],
            dsp_states: vec![VoiceState::default(); count],
            active_count: 0,
        }
    }

    pub fn count(&self) -> u32 {
        self.infos.len() as u32
    }

    pub fn info(&self, index: usize) -> &VoiceInfo {
        &self.infos[index]
    }

    pub fn info_mut(&mut self, index: usize) -> &mut VoiceInfo {
        &mut self.infos[index]
    }

    pub fn infos(&self) -> &[VoiceInfo] {
        &self.infos
    }

    /// Voice indices in render order, highest priority value first. Valid after [`Self::sort_info`].
    pub fn sorted(&self) -> &[u32] {
        &self.sorted
    }

    pub fn channel_resource(&self, index: usize) -> &VoiceChannelResource {
        &self.channel_resources[index]
    }

    pub fn channel_resources(&self) -> &[VoiceChannelResource] {
        &self.channel_resources
    }

    pub fn channel_resource_mut(&mut self, index: usize) -> &mut VoiceChannelResource {
        &mut self.channel_resources[index]
    }

    /// Host copy of the voice states, read when answering updates.
    pub fn states(&self) -> &[VoiceState] {
        &self.cpu_states
    }

    /// DSP-shared voice states the command list decodes into.
    pub fn dsp_states(&self) -> &[VoiceState] {
        &self.dsp_states
    }

    pub fn dsp_states_mut(&mut self) -> &mut [VoiceState] {
        &mut self.dsp_states
    }

    pub fn active_count(&self) -> u32 {
        self.active_count
    }

    pub fn set_active_count(&mut self, count: u32) {
        self.active_count = count;
    }

    /// Orders voices by descending priority value, then descending sort order. Ties keep the
    /// voice index order.
    pub fn sort_info(&mut self) {
        let infos = &self.infos;
        for (slot, index) in self.sorted.iter_mut().zip(0..) {
            *slot = index;
        }
        self.sorted.sort_unstable_by(|&a, &b| {
            let (va, vb) = (&infos[a as usize], &infos[b as usize]);
            vb.priority
                .cmp(&va.priority)
                .then(vb.sort_order.cmp(&va.sort_order))
                .then(a.cmp(&b))
        });
    }

    /// Copies the DSP-shared states back to the host copy after a command list.
    pub fn update_state_by_dsp_shared(&mut self) {
        self.cpu_states.copy_from_slice(&self.dsp_states);
    }

    /// Prepares voice `index` for command generation: resets new voices and applies flushes
    /// and play state changes. Returns whether the voice was playing.
    pub fn update_for_command_generation(
        &mut self,
        index: usize,
        pools: &[MemoryPoolInfo],
    ) -> bool {
        let info = &mut self.infos[index];
        if info.is_new {
            for &id in info.channel_ids() {
                self.dsp_states[id as usize] = VoiceState::default();
                let resource = &mut self.channel_resources[id as usize];
                resource.prev_mix_volumes = resource.mix_volumes;
            }
            info.prev_volume = info.volume;
            info.is_new = false;
        }
        info.update_parameters_for_command_generation(&mut self.dsp_states, pools)
    }

    pub(crate) fn split_for_update(&mut self) -> (&mut [VoiceInfo], &mut [VoiceState]) {
        (&mut self.infos, &mut self.cpu_states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adpcm_offsets_count_frame_headers() {
        assert_eq!(adpcm_byte_offset(0), 0);
        assert_eq!(adpcm_byte_offset(14), 8);
        assert_eq!(adpcm_byte_offset(1), 2);
        assert_eq!(adpcm_byte_offset(2), 2);
        assert_eq!(adpcm_byte_offset(3), 3);
        assert_eq!(adpcm_byte_offset(15), 10);
    }

    #[test]
    fn v2_biquads_convert_to_q14() {
        let mut v2 = VoiceInParameterV2::zeroed();
        v2.biquads[0] = BiquadFilterParameter2 {
            enabled: 1,
            numerator: [1.0, -0.5, 4.0],
            denominator: [-2.5, 0.25],
            ..Default::default()
        };
        v2.wave_buffer_index = 0x1_0003;
        v2.mix_id = -1;
        let (v1, float) = v2.to_v1();
        assert_eq!(v1.biquads[0].b, [16384, -8192, 32767]);
        assert_eq!(v1.biquads[0].a, [-32768, 4096]);
        assert_eq!(v1.wave_buffer_index, 3);
        assert_eq!(v1.mix_id, u32::MAX);
        assert_eq!(float[0].numerator, [1.0, -0.5, 4.0]);
    }

    #[test]
    fn sorting_is_priority_then_order_then_index() {
        let mut voices = VoiceContext::new(4);
        for (i, (priority, order)) in [(1, 0), (5, 0), (1, 3), (5, 0)].into_iter().enumerate() {
            voices.info_mut(i).priority = priority;
            voices.info_mut(i).sort_order = order;
        }
        voices.sort_info();
        assert_eq!(voices.sorted(), &[1, 3, 2, 0]);
    }
}
