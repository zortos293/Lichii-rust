//! 32 kHz/16 kHz/8 kHz to 48 kHz upsampling of the final mix (`upsample.cpp`).

#![allow(clippy::excessive_precision)]

use super::MAX_CHANNELS;
use super::command::UpsampleCommand;
use super::fixed::Fx32;
use super::processor::Frame;
use super::state::DspState;

const WINDOW_SIZE: usize = 10;
const HISTORY_SIZE: u16 = 20;

/// Per-channel polyphase upsampler state (`UpsamplerState`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpsamplerState {
    /// `FixedPoint<16, 16>` output samples per input sample.
    pub ratio: Fx32<16>,
    /// `FixedPoint<24, 8>` input history ring.
    pub history: [Fx32<8>; HISTORY_SIZE as usize],
    pub window_size: u16,
    pub history_output_index: u16,
    pub history_input_index: u16,
    pub history_start_index: u16,
    pub history_end_index: u16,
    pub initialized: bool,
    pub sample_index: u8,
}

/// An upsampler owned by a device sink (`UpsamplerInfo`): which mix buffers it converts and the
/// 48 kHz samples it produces, `sample_count` per mix buffer index.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UpsamplerInfo {
    pub states: [UpsamplerState; MAX_CHANNELS],
    pub samples: Vec<i32>,
    pub sample_count: u32,
    pub input_count: u32,
    pub enabled: bool,
    pub inputs: [i16; MAX_CHANNELS],
}

fn sinc(values: [f32; WINDOW_SIZE]) -> [Fx32<15>; WINDOW_SIZE] {
    values.map(Fx32::from_f32)
}

struct Windows {
    sinc1: [Fx32<15>; WINDOW_SIZE],
    sinc2: [Fx32<15>; WINDOW_SIZE],
    sinc3: [Fx32<15>; WINDOW_SIZE],
    sinc4: [Fx32<15>; WINDOW_SIZE],
    sinc5: [Fx32<15>; WINDOW_SIZE],
}

impl Windows {
    fn new() -> Self {
        Self {
            sinc1: sinc([
                0.95376587,
                -0.12872314,
                0.060028076,
                -0.032470703,
                0.017669678,
                -0.009124756,
                0.004272461,
                -0.001739502,
                0.000579834,
                -0.000091552734,
            ]),
            sinc2: sinc([
                0.8230896,
                -0.19161987,
                0.093444824,
                -0.05090332,
                0.027557373,
                -0.014038086,
                0.0064697266,
                -0.002532959,
                0.00079345703,
                -0.00012207031,
            ]),
            sinc3: sinc([
                0.6298828,
                -0.19274902,
                0.09725952,
                -0.05319214,
                0.028625488,
                -0.014373779,
                0.006500244,
                -0.0024719238,
                0.0007324219,
                -0.000091552734,
            ]),
            sinc4: sinc([
                0.4057312,
                -0.1468811,
                0.07601929,
                -0.041656494,
                0.022216797,
                -0.011016846,
                0.004852295,
                -0.0017700195,
                0.00048828125,
                -0.000030517578,
            ]),
            sinc5: sinc([
                0.1854248,
                -0.075164795,
                0.03967285,
                -0.021728516,
                0.011474609,
                -0.005584717,
                0.0024108887,
                -0.0008239746,
                0.00021362305,
                0.0,
            ]),
        }
    }
}

impl UpsamplerState {
    fn initialize(&mut self, source_sample_count: u32) {
        let ratio = match source_sample_count {
            40 => 6.0,
            80 => 3.0,
            160 => 1.5,
            count => {
                tracing::error!(count, "invalid upsampling source sample count");
                1.5
            }
        };
        self.window_size = WINDOW_SIZE as u16;
        self.ratio = Fx32::from_f32(ratio);
        self.history = Default::default();
        self.history_input_index = 0;
        self.history_output_index = 9;
        self.history_start_index = 0;
        self.history_end_index = HISTORY_SIZE - 1;
        self.initialized = true;
    }

    fn push(&mut self, sample: i32) {
        self.history[self.history_input_index as usize] = Fx32::from_int(sample);
        self.history_input_index = (self.history_input_index + 1) % HISTORY_SIZE;
        self.history_output_index = (self.history_output_index + 1) % HISTORY_SIZE;
    }

    fn current(&self) -> i32 {
        self.history[self.history_output_index as usize].to_int_floor()
    }

    fn interpolate(
        &self,
        backward: &[Fx32<15>; WINDOW_SIZE],
        forward: &[Fx32<15>; WINDOW_SIZE],
    ) -> i32 {
        let mut result = 0i64;
        let mut index = self.history_output_index;
        for coefficient in backward {
            result = result.wrapping_add(
                (self.history[index as usize].raw() as i64).wrapping_mul(coefficient.raw() as i64),
            );
            index = if index == self.history_start_index {
                self.history_end_index
            } else {
                index - 1
            };
        }
        index = (self.history_output_index + 1) % HISTORY_SIZE;
        for coefficient in forward {
            result = result.wrapping_add(
                (self.history[index as usize].raw() as i64).wrapping_mul(coefficient.raw() as i64),
            );
            index = if index == self.history_end_index {
                self.history_start_index
            } else {
                index + 1
            };
        }
        (result >> (8 + 15)) as i32
    }

    fn process(
        &mut self,
        output: &mut [i32],
        input: &[i32],
        source_sample_count: u32,
        windows: &Windows,
    ) {
        if !self.initialized {
            self.initialize(source_sample_count);
        }
        let mut input = input.iter().copied();
        let mut next = || input.next().unwrap_or(0);
        let w = windows;
        let phases: u8 = match self.ratio.to_int_floor() {
            6 => 6,
            _ => 3,
        };
        for out in output.iter_mut() {
            *out = match (self.ratio.to_int_floor(), self.sample_index) {
                (_, 0) => {
                    self.push(next());
                    self.current()
                }
                (6, 1) => self.interpolate(&w.sinc1, &w.sinc5),
                (6, 2) => self.interpolate(&w.sinc2, &w.sinc4),
                (6, 3) => self.interpolate(&w.sinc3, &w.sinc3),
                (6, 4) => self.interpolate(&w.sinc4, &w.sinc2),
                (6, 5) => self.interpolate(&w.sinc5, &w.sinc1),
                (3, 1) => self.interpolate(&w.sinc2, &w.sinc4),
                (3, 2) => self.interpolate(&w.sinc4, &w.sinc2),
                (_, 1) => self.interpolate(&w.sinc4, &w.sinc2),
                _ => {
                    self.push(next());
                    self.interpolate(&w.sinc2, &w.sinc4)
                }
            };
            self.sample_index = (self.sample_index + 1) % phases;
        }
    }
}

pub(super) fn upsample(command: &UpsampleCommand, frame: &Frame<'_>, state: &mut DspState) {
    let Some(info) = state.upsamplers.get_mut(command.upsampler.0 as usize) else {
        return;
    };
    let windows = Windows::new();
    let input_count = info
        .input_count
        .min(command.buffer_count)
        .min(MAX_CHANNELS as u32) as usize;
    let sample_count = info.sample_count as usize;
    for i in 0..input_count {
        let channel = info.inputs[i];
        if channel < 0 || channel as u32 >= frame.buffer_count {
            continue;
        }
        let Some(input) = frame.range(channel) else {
            continue;
        };
        let start = sample_count * channel as usize;
        let Some(output) = info.samples.get_mut(start..start + sample_count) else {
            continue;
        };
        info.states[i].process(
            output,
            &frame.mix[input],
            command.source_sample_count,
            &windows,
        );
    }
}
