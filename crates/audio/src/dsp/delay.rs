//! The delay effect (`delay.cpp`).

#![allow(clippy::excessive_precision)]

use super::MAX_CHANNELS;
use super::abi::{DelayParameterV1, ParameterState};
use super::command::DelayCommand;
use super::fixed::{Fx, Q14, Q14x32};
use super::processor::{Frame, bypass, channel_buffers, is_channel_count_valid};
use super::state::DspState;

/// One channel's delay line (`DelayInfo::DelayLine`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DelayLine {
    pub sample_count_max: i32,
    pub sample_count: i32,
    pub buffer: Vec<Q14>,
    pub buffer_pos: u32,
    pub decay_rate: Q14x32,
}

impl DelayLine {
    fn read(&self) -> Q14 {
        self.buffer
            .get(self.buffer_pos as usize)
            .copied()
            .unwrap_or_default()
    }

    fn write(&mut self, value: Q14) {
        if self.buffer.is_empty() {
            return;
        }
        self.buffer[self.buffer_pos as usize] = value;
        self.buffer_pos = ((self.buffer_pos as usize + 1) % self.buffer.len()) as u32;
    }
}

/// Delay effect state (`DelayInfo::State`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DelayState {
    pub delay_lines: [DelayLine; MAX_CHANNELS],
    pub feedback_gain: Q14x32,
    pub delay_feedback_gain: Q14x32,
    pub delay_feedback_cross_gain: Q14x32,
    pub lowpass_gain: Q14x32,
    pub lowpass_feedback_gain: Q14x32,
    pub lowpass_z: [Q14; MAX_CHANNELS],
}

fn set_parameters(params: &DelayParameterV1, state: &mut DelayState) {
    let one = Q14x32::from_f32(1.0);
    let mut channel_spread = Q14x32::from_raw(params.channel_spread);
    state.feedback_gain = Q14x32::from_raw(params.feedback_gain) * Q14x32::from_f32(0.97998046875);
    state.delay_feedback_gain = state.feedback_gain * (one - channel_spread);
    if params.channel_count == 4 || params.channel_count == 6 {
        channel_spread = Q14x32::from_raw(channel_spread.raw() >> 1);
    }
    state.delay_feedback_cross_gain = channel_spread * state.feedback_gain;
    state.lowpass_feedback_gain =
        Q14x32::from_raw(params.lowpass_amount) * Q14x32::from_f32(0.949951171875);
    state.lowpass_gain = one - state.lowpass_feedback_gain;
}

fn initialize(params: &DelayParameterV1, state: &mut DelayState) {
    let mut buffers = std::mem::take(&mut state.delay_lines).map(|line| line.buffer);
    *state = DelayState::default();
    let sample_rate = Q14x32::from_raw(params.sample_rate);
    let channels = (params.channel_count as usize).min(MAX_CHANNELS);
    for (line, buffer) in state
        .delay_lines
        .iter_mut()
        .zip(&mut buffers)
        .take(channels)
    {
        let samples_max_factor =
            (sample_rate.to_int_floor() as u32).wrapping_mul(params.delay_time_max);
        let sample_count_max = Fx::<32>::from_f32(0.064) * Fx::from_int(samples_max_factor as i64);
        let delay_time =
            Q14x32::from_int(params.delay_time as i32) * (sample_rate / Q14x32::from_int(1000));
        let sample_count =
            Fx::<32>::from_raw((delay_time.raw() as i64) << 18).min(sample_count_max);
        line.sample_count_max = sample_count_max.to_int_floor();
        line.sample_count = sample_count.to_int_floor();
        let mut buffer = std::mem::take(buffer);
        buffer.clear();
        buffer.resize(line.sample_count.max(1) as usize, Q14::default());
        line.buffer = buffer;
        line.buffer_pos = 0;
        line.decay_rate = Q14x32::from_f32(1.0);
    }
    set_parameters(params, state);
}

fn feedback_matrix<const N: usize>(params: &DelayParameterV1, state: &DelayState) -> [[Q14; N]; N] {
    let g = state.delay_feedback_gain.widen();
    let c = state.delay_feedback_cross_gain.widen();
    let z = Q14::default();
    let rows: [[Q14; 6]; 6] = match N {
        1 => [
            [state.feedback_gain.widen(), z, z, z, z, z],
            [z; 6],
            [z; 6],
            [z; 6],
            [z; 6],
            [z; 6],
        ],
        2 => [
            [g, c, z, z, z, z],
            [c, g, z, z, z, z],
            [z; 6],
            [z; 6],
            [z; 6],
            [z; 6],
        ],
        4 => [
            [g, c, c, z, z, z],
            [c, g, z, c, z, z],
            [c, z, g, c, z, z],
            [z, c, c, g, z, z],
            [z; 6],
            [z; 6],
        ],
        _ => [
            [g, z, c, z, c, z],
            [z, g, c, z, z, c],
            [c, c, g, z, z, z],
            [
                z,
                z,
                z,
                Q14x32::from_raw(params.feedback_gain).widen(),
                z,
                z,
            ],
            [c, z, z, z, g, c],
            [z, c, z, z, c, g],
        ],
    };
    std::array::from_fn(|row| std::array::from_fn(|column| rows[row][column]))
}

fn apply<const N: usize>(
    params: &DelayParameterV1,
    state: &mut DelayState,
    mix: &mut [i32],
    inputs: &[usize; MAX_CHANNELS],
    outputs: &[usize; MAX_CHANNELS],
    count: usize,
) {
    let matrix = feedback_matrix::<N>(params, state);
    let in_gain = Q14x32::from_raw(params.in_gain).widen();
    let dry_gain = Q14x32::from_raw(params.dry_gain).widen();
    let wet_gain = Q14x32::from_raw(params.wet_gain).widen();
    let lowpass_gain = state.lowpass_gain.widen();
    let lowpass_feedback_gain = state.lowpass_feedback_gain.widen();
    for i in 0..count {
        let input: [Q14; N] =
            std::array::from_fn(|ch| Q14::from_int(mix[inputs[ch] + i].wrapping_mul(64) as i64));
        let delayed: [Q14; N] = std::array::from_fn(|ch| state.delay_lines[ch].read());
        for ch in 0..N {
            let mut feedback = Q14::default();
            for (row, &sample) in matrix.iter().zip(&delayed) {
                feedback += sample * row[ch];
            }
            let gained = input[ch] * in_gain + feedback;
            state.lowpass_z[ch] =
                gained * lowpass_gain + state.lowpass_z[ch] * lowpass_feedback_gain;
        }
        for ch in 0..N {
            let lowpassed = state.lowpass_z[ch];
            state.delay_lines[ch].write(lowpassed);
        }
        for ch in 0..N {
            mix[outputs[ch] + i] =
                (input[ch] * dry_gain + delayed[ch] * wet_gain).to_int_floor() / 64;
        }
    }
}

pub(super) fn delay(command: &DelayCommand, frame: &mut Frame<'_>, state: &mut DspState) {
    let params = &command.parameter;
    let channels = params.channel_count as usize;
    let Some((inputs, outputs)) =
        channel_buffers(frame, &command.inputs, &command.outputs, channels)
    else {
        return;
    };
    let Some(effect) = state.effects.get_mut(command.state.0 as usize) else {
        return;
    };
    let delay_state = effect.delay();
    if command.effect_enabled {
        match params.parameter_state() {
            Some(ParameterState::Updating) => set_parameters(params, delay_state),
            Some(ParameterState::Initialized) => initialize(params, delay_state),
            _ => {}
        }
    }
    if !is_channel_count_valid(params.channel_count) {
        tracing::error!(
            channels = params.channel_count,
            "invalid delay channel count"
        );
        return;
    }
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    if !command.effect_enabled {
        bypass(mix, &inputs, &outputs, channels, count);
        return;
    }
    match channels {
        1 => apply::<1>(params, delay_state, mix, &inputs, &outputs, count),
        2 => apply::<2>(params, delay_state, mix, &inputs, &outputs, count),
        4 => apply::<4>(params, delay_state, mix, &inputs, &outputs, count),
        _ => apply::<6>(params, delay_state, mix, &inputs, &outputs, count),
    }
}
