//! Sinks: where the final mix goes, either the audio device or a guest circular buffer.

use bytemuck::{Pod, Zeroable};

use super::behavior::ErrorInfo;
use super::memory::{AddressInfo, PoolMapper};
use super::upsampler::UpsamplerManager;
use super::{MAX_CHANNELS, SampleFormat};

/// Sink type tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum SinkType {
    #[default]
    Invalid = 0,
    DeviceSink = 1,
    CircularBufferSink = 2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DeviceInParameter {
    pub name: [u8; 0x100],
    pub input_count: u32,
    pub inputs: [i8; MAX_CHANNELS],
    pub _unk_10a: u8,
    pub downmix_enabled: u8,
    pub downmix_coeff: [f32; 4],
}
const _: () = assert!(size_of::<DeviceInParameter>() == 0x11C);

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CircularBufferInParameter {
    pub cpu_address: u64,
    pub size: u32,
    pub input_count: u32,
    pub sample_count: u32,
    pub previous_pos: u32,
    pub format: u8,
    pub _pad: [u8; 3],
    pub inputs: [i8; MAX_CHANNELS],
    pub in_use: u8,
    pub _unk_23: [u8; 5],
}
const _: () = assert!(size_of::<CircularBufferInParameter>() == 0x28);

/// Sink input; `specific` holds a [`DeviceInParameter`] or a [`CircularBufferInParameter`].
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SinkInParameter {
    pub sink_type: u8,
    pub in_use: u8,
    pub _pad: [u8; 2],
    pub node_id: u32,
    pub _unk_08: [u8; 0x18],
    pub specific: [u8; 0x120],
}
const _: () = assert!(size_of::<SinkInParameter>() == 0x140);

impl SinkInParameter {
    pub fn device(&self) -> DeviceInParameter {
        bytemuck::pod_read_unaligned(&self.specific[..size_of::<DeviceInParameter>()])
    }

    pub fn circular_buffer(&self) -> CircularBufferInParameter {
        bytemuck::pod_read_unaligned(&self.specific[..size_of::<CircularBufferInParameter>()])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct SinkOutStatus {
    pub write_offset: u32,
    pub _unk_04: [u8; 0x1C],
}
const _: () = assert!(size_of::<SinkOutStatus>() == 0x20);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceState {
    /// Upsampler slot, taken by the command generator when the sink needs one.
    pub upsampler: Option<u32>,
    /// Downmix coefficients, Q16.16.
    pub downmix_coeff: [i32; 4],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CircularBufferState {
    pub last_pos2: u32,
    pub current_pos: i32,
    pub last_pos: u32,
    pub address_info: AddressInfo,
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(
    clippy::large_enum_variant,
    reason = "sinks live in a fixed table, so every slot pays for the largest variant anyway"
)]
pub enum SinkKind {
    #[default]
    Invalid,
    Device {
        parameter: DeviceInParameter,
        state: DeviceState,
    },
    CircularBuffer {
        parameter: CircularBufferInParameter,
        state: CircularBufferState,
    },
}

/// One sink.
#[derive(Clone, Copy, Debug, Default)]
pub struct SinkInfo {
    pub in_use: bool,
    pub buffer_unmapped: bool,
    pub node_id: u32,
    pub kind: SinkKind,
}

fn to_q16(value: f32) -> i32 {
    (value * 65536.0) as i32
}

impl SinkInfo {
    pub fn sink_type(&self) -> SinkType {
        match self.kind {
            SinkKind::Invalid => SinkType::Invalid,
            SinkKind::Device { .. } => SinkType::DeviceSink,
            SinkKind::CircularBuffer { .. } => SinkType::CircularBufferSink,
        }
    }

    pub fn should_skip(&self) -> bool {
        self.buffer_unmapped
    }

    /// Replaces the sink with a fresh one of `raw_type`, releasing a device sink's upsampler.
    /// Unknown types become invalid sinks.
    pub fn reset(&mut self, raw_type: u8, upsamplers: &mut UpsamplerManager) {
        if let SinkKind::Device { state, .. } = &self.kind
            && let Some(upsampler) = state.upsampler
        {
            upsamplers.free(upsampler);
        }
        let kind = match raw_type {
            1 => SinkKind::Device {
                parameter: DeviceInParameter::zeroed(),
                state: DeviceState::default(),
            },
            2 => SinkKind::CircularBuffer {
                parameter: CircularBufferInParameter::zeroed(),
                state: CircularBufferState::default(),
            },
            _ => {
                if raw_type != 0 {
                    tracing::error!(raw_type, "invalid audio sink type");
                }
                SinkKind::Invalid
            }
        };
        *self = Self {
            kind,
            ..Self::default()
        };
    }

    pub fn update(
        &mut self,
        error: &mut ErrorInfo,
        out: &mut SinkOutStatus,
        params: &SinkInParameter,
        mapper: &PoolMapper,
    ) {
        match &mut self.kind {
            SinkKind::Invalid => {
                *out = SinkOutStatus::default();
                error.set_success();
            }
            SinkKind::Device { parameter, state } => {
                let input = params.device();
                if self.in_use == (params.in_use != 0) {
                    parameter.downmix_enabled = input.downmix_enabled;
                    parameter.downmix_coeff = input.downmix_coeff;
                } else {
                    self.in_use = params.in_use != 0;
                    self.node_id = params.node_id;
                    *parameter = input;
                }
                state.downmix_coeff = parameter.downmix_coeff.map(to_q16);
                *out = SinkOutStatus::default();
                error.set_success();
            }
            SinkKind::CircularBuffer { parameter, state } => {
                let input = params.circular_buffer();
                if self.in_use == (input.in_use != 0) && !self.buffer_unmapped {
                    error.set_success();
                    out.write_offset = state.last_pos2;
                    return;
                }
                self.node_id = params.node_id;
                self.in_use = params.in_use != 0;
                if self.in_use {
                    self.buffer_unmapped = !mapper.try_attach_buffer(
                        error,
                        &mut state.address_info,
                        input.cpu_address,
                        input.size as u64,
                    );
                }
                *parameter = input;
                out.write_offset = state.last_pos2;
            }
        }
    }

    /// Advances a circular buffer sink's write position by one frame.
    pub fn update_for_command_generation(&mut self) {
        let SinkKind::CircularBuffer { parameter, state } = &mut self.kind else {
            return;
        };
        if !self.in_use {
            return;
        }
        let pos = state.current_pos;
        state.last_pos2 = state.last_pos;
        state.last_pos = pos as u32;
        let advance = (parameter.input_count.wrapping_mul(parameter.sample_count) as u64)
            .wrapping_mul(SampleFormat::PcmInt16.byte_size());
        state.current_pos = state.current_pos.wrapping_add(advance as i32);
        if parameter.size > 0 {
            state.current_pos = (state.current_pos as u32 % parameter.size) as i32;
        }
    }
}

/// All sinks of a renderer.
#[derive(Clone, Debug, Default)]
pub struct SinkContext {
    infos: Vec<SinkInfo>,
}

impl SinkContext {
    pub fn new(count: u32) -> Self {
        Self {
            infos: vec![SinkInfo::default(); count as usize],
        }
    }

    pub fn count(&self) -> u32 {
        self.infos.len() as u32
    }

    pub fn info(&self, index: usize) -> &SinkInfo {
        &self.infos[index]
    }

    pub fn info_mut(&mut self, index: usize) -> &mut SinkInfo {
        &mut self.infos[index]
    }

    pub fn infos(&self) -> &[SinkInfo] {
        &self.infos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circular_position_wraps_like_unsigned_modulo() {
        let mut sink = SinkInfo::default();
        sink.reset(2, &mut UpsamplerManager::new(0));
        sink.in_use = true;
        if let SinkKind::CircularBuffer { parameter, state } = &mut sink.kind {
            parameter.input_count = 2;
            parameter.sample_count = 240;
            parameter.size = 0x1000;
            state.current_pos = 0xD00;
        }
        sink.update_for_command_generation();
        let SinkKind::CircularBuffer { state, .. } = sink.kind else {
            unreachable!()
        };
        assert_eq!(state.last_pos, 0xD00);
        assert_eq!(state.current_pos, 0xD00 + 960 - 0x1000);
    }

    #[test]
    fn device_reset_frees_its_upsampler() {
        let mut upsamplers = UpsamplerManager::new(2);
        let mut sink = SinkInfo::default();
        sink.reset(1, &mut upsamplers);
        if let SinkKind::Device { state, .. } = &mut sink.kind {
            state.upsampler = upsamplers.allocate();
        }
        assert!(upsamplers.is_enabled(0));
        sink.reset(2, &mut upsamplers);
        assert!(!upsamplers.is_enabled(0));
        assert_eq!(sink.sink_type(), SinkType::CircularBufferSink);
    }
}
