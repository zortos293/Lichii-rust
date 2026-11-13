//! The compressor effect (`compressor.cpp`).

use super::abi::{CompressorParameterV2, ParameterState};
use super::command::CompressorCommand;
use super::fixed::Q15;
use super::processor::{Frame, bypass, channel_buffers};
use super::state::DspState;

#[allow(clippy::approx_constant)]
const LOG2_10: f32 = 3.3219;
#[allow(clippy::approx_constant)]
const LN_2: f32 = 0.69315;

/// Compressor state (`CompressorInfo::State`); the unknown fields keep the C++ names.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CompressorState {
    pub unk_00: f32,
    pub unk_04: f32,
    pub unk_08: f32,
    pub unk_0c: f32,
    pub unk_10: f32,
    pub unk_14: f32,
    pub unk_18: f32,
    pub makeup_gain: f32,
    pub unk_20: f32,
}

fn set_parameters(params: &CompressorParameterV2, state: &mut CompressorState) {
    let ratio = 1.0 / params.compressor_ratio;
    let makeup_gain = if params.makeup_gain_enabled != 0 {
        (params.threshold * 0.5) * (ratio - 1.0) - 3.0
    } else {
        0.0
    };
    state.makeup_gain = makeup_gain;
    state.unk_18 = params.unk_28;
    let a = (params.out_gain + makeup_gain) / 20.0 * LOG2_10;
    let b = (a - a.trunc()) * LN_2;
    state.unk_0c = (1.0 - ratio) / 6.0;
    state.unk_14 = params.threshold + 1.5;
    state.unk_10 = params.threshold - 1.5;
    state.unk_20 = 2.0f32.powf(b);
}

fn initialize(params: &CompressorParameterV2, state: &mut CompressorState) {
    *state = CompressorState {
        unk_04: 1.0,
        unk_08: 1.0,
        ..Default::default()
    };
    set_parameters(params, state);
}

fn apply(
    params: &CompressorParameterV2,
    state: &mut CompressorState,
    mix: &mut [i32],
    inputs: &[usize; 6],
    outputs: &[usize; 6],
    count: usize,
) {
    let channels = params.channel_count.clamp(0, 6) as usize;
    let (mut s00, mut s04, mut s08, mut s18) =
        (state.unk_00, state.unk_04, state.unk_08, state.unk_18);
    for i in 0..count {
        let mut a = 0.0f32;
        for &input in &inputs[..channels] {
            let sample = Q15::from_int(mix[input + i] as i64);
            a += (sample * sample).to_f32();
        }
        s00 += params.unk_24 * ((a / params.channel_count as f32) - state.unk_00);
        let mut b = -100.0f32;
        let mut c = 0.0f32;
        if s00 as f64 >= 1.0e-10 {
            b = s00.log10() * 10.0;
            c = 1.0;
        }
        if b >= state.unk_10 {
            let d = if b >= state.unk_14 {
                ((1.0 / params.compressor_ratio) - 1.0) * (b - params.threshold)
            } else {
                (b - state.unk_10) * (b - state.unk_10) * -state.unk_0c
            };
            let e = d / 20.0 * LOG2_10;
            let f = (e - e.trunc()) * LN_2;
            c = 2.0f32.powf(f);
        }
        s18 = params.unk_28;
        let mut next = c;
        if (s04 - c) <= 0.08 {
            s18 = params.unk_2c;
            if (s04 - c) >= -0.08 && (s08 - c).abs() >= 0.001 {
                next = s04;
            }
        }
        s04 = next;
        s08 += (c - s08) * s18;
        for ch in 0..channels {
            mix[outputs[ch] + i] = (mix[inputs[ch] + i] as f32 * s08 * state.unk_20) as i32;
        }
    }
    state.unk_00 = s00;
    state.unk_04 = s04;
    state.unk_08 = s08;
    state.unk_18 = s18;
}

pub(super) fn compressor(command: &CompressorCommand, frame: &mut Frame<'_>, state: &mut DspState) {
    let params = &command.parameter;
    let channels = params.channel_count.max(0) as usize;
    let Some((inputs, outputs)) =
        channel_buffers(frame, &command.inputs, &command.outputs, channels)
    else {
        return;
    };
    let Some(effect) = state.effects.get_mut(command.state.0 as usize) else {
        return;
    };
    let compressor_state = effect.compressor();
    if command.effect_enabled {
        match params.parameter_state() {
            Some(ParameterState::Updating) => set_parameters(params, compressor_state),
            Some(ParameterState::Initialized) => initialize(params, compressor_state),
            _ => {}
        }
    }
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    if command.effect_enabled {
        apply(params, compressor_state, mix, &inputs, &outputs, count);
    } else {
        bypass(mix, &inputs, &outputs, channels, count);
    }
}
