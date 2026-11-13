//! The reverb effect (`reverb.cpp`).

#![allow(clippy::excessive_precision)]

use super::abi::{ParameterState, ReverbParameterV2};
use super::command::ReverbCommand;
use super::fixed::Q14;
use super::processor::{Frame, bypass, channel_buffers};
use super::state::DspState;

const MAX_DELAY_LINES: usize = 4;
const MAX_DELAY_TAPS: usize = 10;

const FDN_MAX_DELAY_LINE_TIMES: [f32; MAX_DELAY_LINES] = [
    53.9532470703125,
    79.19256591796875,
    116.23876953125,
    170.61529541015625,
];
const DECAY_MAX_DELAY_LINE_TIMES: [f32; MAX_DELAY_LINES] = [7.0, 9.0, 13.0, 17.0];

#[rustfmt::skip]
const EARLY_DELAY_TIMES: [[f32; MAX_DELAY_TAPS + 1]; 5] = [
    [0.000000, 3.500000, 2.799988, 3.899963, 2.699951, 13.399963, 7.899963, 8.399963, 9.899963, 12.000000, 12.500000],
    [0.000000, 11.799988, 5.500000, 11.199951, 10.399963, 38.099976, 22.199951, 29.599976, 21.199951, 24.799988, 40.000000],
    [0.000000, 41.500000, 20.500000, 41.299988, 0.000000, 29.500000, 33.799988, 45.199951, 46.799988, 0.000000, 50.000000],
    [33.099976, 43.299988, 22.799988, 37.899963, 14.899963, 35.299988, 17.899963, 34.199951, 0.000000, 43.299988, 50.000000],
    [0.0; MAX_DELAY_TAPS + 1],
];

#[rustfmt::skip]
const EARLY_DELAY_GAINS: [[f32; MAX_DELAY_TAPS]; 5] = [
    [0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993, 0.679993],
    [0.699951, 0.679993, 0.699951, 0.679993, 0.699951, 0.679993, 0.679993, 0.679993, 0.679993, 0.679993],
    [0.500000, 0.699951, 0.699951, 0.679993, 0.500000, 0.679993, 0.679993, 0.699951, 0.679993, 0.000000],
    [0.929993, 0.919983, 0.869995, 0.859985, 0.939941, 0.809998, 0.799988, 0.769958, 0.759949, 0.649963],
    [0.0; MAX_DELAY_TAPS],
];

const FDN_DELAY_TIMES: [[f32; MAX_DELAY_LINES]; 5] = [
    [53.953247, 79.192566, 116.238770, 130.615295],
    [53.953247, 79.192566, 116.238770, 170.615295],
    [5.000000, 10.000000, 5.000000, 10.000000],
    [47.029968, 71.000000, 103.000000, 170.000000],
    [53.953247, 79.192566, 116.238770, 170.615295],
];

const DECAY_DELAY_TIMES: [[f32; MAX_DELAY_LINES]; 5] = [
    [7.0, 9.0, 13.0, 17.0],
    [7.0, 9.0, 13.0, 17.0],
    [1.0, 1.0, 1.0, 1.0],
    [7.0, 7.0, 13.0, 9.0],
    [7.0, 9.0, 13.0, 17.0],
];

/// A reverb delay line (`ReverbInfo::ReverbDelayLine`), with the C++ pointers as indices.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReverbDelayLine {
    pub sample_count: i32,
    pub sample_count_max: i32,
    pub buffer: Vec<Q14>,
    /// Wrap index; can sit below zero after a parameter update shrinks the line to nothing.
    pub buffer_end: i64,
    pub input: usize,
    pub output: usize,
    pub decay: Q14,
}

impl ReverbDelayLine {
    fn initialize(&mut self, delay_time: i32, decay_rate: f32) {
        self.buffer.clear();
        self.buffer
            .resize((delay_time.max(0) as usize) + 1, Q14::default());
        self.buffer_end = delay_time as i64;
        self.output = 0;
        self.decay = Q14::from_f32(decay_rate);
        self.sample_count_max = delay_time;
        self.set_delay(delay_time);
    }

    fn set_delay(&mut self, delay_time: i32) {
        if self.sample_count_max < delay_time {
            return;
        }
        self.sample_count = delay_time;
        self.input = 0;
    }

    fn tick(&mut self, sample: Q14) -> Q14 {
        let out = self.read();
        self.output += 1;
        if self.output as i64 >= self.buffer_end {
            self.output = 0;
        }
        self.write(sample);
        out
    }

    fn read(&self) -> Q14 {
        self.buffer.get(self.output).copied().unwrap_or_default()
    }

    fn write(&mut self, sample: Q14) {
        if let Some(slot) = self.buffer.get_mut(self.input) {
            *slot = sample;
        }
        self.input += 1;
        if self.input as i64 >= self.buffer_end {
            self.input = 0;
        }
    }

    fn tap_out(&self, index: i32) -> Q14 {
        let mut out = self.input as i64 - (index as i64 + 1);
        if out < 0 {
            out += self.sample_count as i64;
        }
        usize::try_from(out)
            .ok()
            .and_then(|out| self.buffer.get(out))
            .copied()
            .unwrap_or_default()
    }
}

/// Reverb effect state (`ReverbInfo::State`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReverbState {
    pub pre_delay_line: ReverbDelayLine,
    pub center_delay_line: ReverbDelayLine,
    pub early_delay_times: [i32; MAX_DELAY_TAPS],
    pub early_gains: [Q14; MAX_DELAY_TAPS],
    pub pre_delay_time: i32,
    pub decay_delay_lines: [ReverbDelayLine; MAX_DELAY_LINES],
    pub fdn_delay_lines: [ReverbDelayLine; MAX_DELAY_LINES],
    pub hf_decay_gain: [Q14; MAX_DELAY_LINES],
    pub hf_decay_prev_gain: [Q14; MAX_DELAY_LINES],
    pub prev_feedback_output: [Q14; MAX_DELAY_LINES],
    /// High-frequency cosine term. The C++ computes it once per process from the first
    /// reverb's sample rate; here it is computed once per effect and survives re-initialisation.
    pub hf_cosine: Option<Q14>,
}

pub(super) fn pow_10(value: f32) -> f32 {
    if value >= 0.0 {
        1.0
    } else if value <= -5.3 {
        0.0
    } else {
        10.0f32.powf(value)
    }
}

pub(super) fn cos_degrees(degrees: f32) -> f32 {
    (degrees * std::f32::consts::PI / 180.0).cos()
}

fn clamp_delay(target: i32, max: i32) -> i32 {
    let target = target.max(0);
    if target > max { max } else { target }
}

fn clamp_unit(value: Q14) -> Q14 {
    if value.to_f32() < 0.0 {
        Q14::from_f32(0.0)
    } else if value.to_f32() > 1.0 {
        Q14::from_f32(1.0)
    } else {
        value
    }
}

fn update_parameters(params: &ReverbParameterV2, state: &mut ReverbState) {
    let four = Q14::from_int(4);
    let mut sample_rate = Q14::from_raw(params.sample_rate as i64);
    if sample_rate.to_f32() < 1.0 {
        sample_rate = Q14::from_f32(1.0);
    }
    let mut pre_delay_time = Q14::from_raw(params.pre_delay as i64);
    if pre_delay_time.to_f32() < 0.0 {
        pre_delay_time = Q14::from_f32(0.0);
    }
    let early_mode = (params.early_mode as usize).min(EARLY_DELAY_TIMES.len() - 1);
    let early_gain = Q14::from_raw(params.early_gain as i64);
    let pre_delay_max = state.pre_delay_line.sample_count_max;
    for i in 0..MAX_DELAY_TAPS {
        let time = pre_delay_time + Q14::from_f32(EARLY_DELAY_TIMES[early_mode][i]);
        let target_delay = clamp_delay((time * sample_rate).to_int(), pre_delay_max);
        let old_delay = state.early_delay_times[i] - 1;
        let smooth_delay = old_delay + (target_delay - old_delay) / 4;
        state.early_delay_times[i] = smooth_delay + 1;
        let target_gain = clamp_unit(early_gain * Q14::from_f32(EARLY_DELAY_GAINS[early_mode][i]));
        state.early_gains[i] = state.early_gains[i] + (target_gain - state.early_gains[i]) / four;
    }
    if params.channel_count == 2 {
        state.early_gains[4] *= Q14::from_f32(0.5);
        state.early_gains[5] *= Q14::from_f32(0.5);
    }
    let pre_time = pre_delay_time + Q14::from_f32(EARLY_DELAY_TIMES[early_mode][MAX_DELAY_TAPS]);
    let target_pre_time = clamp_delay((pre_time * sample_rate).to_int(), pre_delay_max);
    state.pre_delay_time += (target_pre_time - state.pre_delay_time) / 4;

    let hf_cosine = *state.hf_cosine.get_or_insert_with(|| {
        Q14::from_f32(cos_degrees((Q14::from_f32(1280.0) / sample_rate).to_f32()))
    });

    let late_mode = (params.late_mode.max(0) as usize).min(FDN_DELAY_TIMES.len() - 1);
    let mut colouration = Q14::from_raw(params.colouration as i64);
    colouration = clamp_unit(colouration);
    let mut decay_time = Q14::from_raw(params.decay_time as i64);
    if decay_time.to_f32() <= 0.0 {
        decay_time = Q14::from_f32(0.0001);
    }
    let mut hf_decay_ratio = Q14::from_raw(params.high_freq_decay_ratio as i64);
    if hf_decay_ratio.to_f32() < 0.001 {
        hf_decay_ratio = Q14::from_f32(0.001);
    }
    if hf_decay_ratio.to_f32() > 0.999 {
        hf_decay_ratio = Q14::from_f32(0.999);
    }
    for i in 0..MAX_DELAY_LINES {
        let fdn = &mut state.fdn_delay_lines[i];
        let target_fdn = clamp_delay(
            (Q14::from_f32(FDN_DELAY_TIMES[late_mode][i]) * sample_rate).to_int(),
            fdn.sample_count_max,
        );
        fdn.sample_count += (target_fdn - fdn.sample_count) / 4;
        fdn.buffer_end = fdn.sample_count as i64 - 1;

        let decay = &mut state.decay_delay_lines[i];
        let target_decay = clamp_delay(
            (Q14::from_f32(DECAY_DELAY_TIMES[late_mode][i]) * sample_rate).to_int(),
            decay.sample_count_max,
        );
        decay.sample_count += (target_decay - decay.sample_count) / 4;
        decay.buffer_end = decay.sample_count as i64 - 1;
        let target_coefficient =
            Q14::from_f32(0.5999755859375) * (Q14::from_f32(1.0) - colouration);
        decay.decay = decay.decay + (target_coefficient - decay.decay) / four;

        let a = (Q14::from_int(state.fdn_delay_lines[i].sample_count_max as i64)
            + Q14::from_int(state.decay_delay_lines[i].sample_count_max as i64))
            * Q14::from_int(-3);
        let b = a / (decay_time * sample_rate);
        let (c, d) = if hf_decay_ratio.to_f32() > 0.9949 {
            (Q14::from_f32(0.0), Q14::from_f32(1.0))
        } else {
            let exponent = ((1.0 / hf_decay_ratio.to_f32()) - 1.0) * 2.0 / 100.0;
            let e = pow_10((Q14::from_f32(exponent) * (b / Q14::from_int(10))).to_f32())
                .clamp(0.0001, 1.0);
            let f = (1.0 - e).max(0.0001);
            let g = 2.0 - (hf_cosine.to_f32() * e * 2.0);
            let h = (g * g - f * f * 4.0).max(0.0).sqrt();
            let c = Q14::from_f32((g - h) / (f * 2.0));
            (c, Q14::from_f32(1.0) - c)
        };
        state.hf_decay_prev_gain[i] =
            state.hf_decay_prev_gain[i] + (c - state.hf_decay_prev_gain[i]) / four;
        let target_gain = Q14::from_f32(pow_10((b / Q14::from_int(1000)).to_f32()))
            * d
            * Q14::from_f32(0.70709228515625);
        state.hf_decay_gain[i] =
            state.hf_decay_gain[i] + (target_gain - state.hf_decay_gain[i]) / four;
        state.prev_feedback_output[i] = Q14::default();
    }
}

fn initialize(params: &ReverbParameterV2, state: &mut ReverbState, long_pre_delay: bool) {
    let hf_cosine = state.hf_cosine;
    let mut old = std::mem::take(state);
    state.hf_cosine = hf_cosine;
    let delay = Q14::from_raw(params.sample_rate as i64);
    for i in 0..MAX_DELAY_LINES {
        state.fdn_delay_lines[i].buffer = std::mem::take(&mut old.fdn_delay_lines[i].buffer);
        let fdn_time = (Q14::from_f32(FDN_MAX_DELAY_LINE_TIMES[i]) * delay).to_uint_floor();
        state.fdn_delay_lines[i].initialize(fdn_time as i32, 1.0);
        state.decay_delay_lines[i].buffer = std::mem::take(&mut old.decay_delay_lines[i].buffer);
        let decay_time = (Q14::from_f32(DECAY_MAX_DELAY_LINE_TIMES[i]) * delay).to_uint_floor();
        state.decay_delay_lines[i].initialize(decay_time as i32, 0.0);
    }
    let pre_delay = if long_pre_delay { 350.0 } else { 150.0 };
    state.pre_delay_line.buffer = old.pre_delay_line.buffer;
    state.pre_delay_line.initialize(
        (Q14::from_f32(pre_delay) * delay).to_uint_floor() as i32,
        1.0,
    );
    state.center_delay_line.buffer = old.center_delay_line.buffer;
    state
        .center_delay_line
        .initialize((Q14::from_int(5) * delay).to_uint_floor() as i32, 1.0);
    update_parameters(params, state);
}

fn all_pass_tick(decay: &mut ReverbDelayLine, fdn: &mut ReverbDelayLine, mix: Q14) -> Q14 {
    let value = decay.read();
    let mixed = mix - value * decay.decay;
    let out = decay.tick(mixed) + mixed * decay.decay;
    fdn.tick(out);
    out
}

const OUT_TAP_INDEXES_1CH: [usize; MAX_DELAY_TAPS] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const OUT_TAP_INDEXES_2CH: [usize; MAX_DELAY_TAPS] = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1];
const OUT_TAP_INDEXES_4CH: [usize; MAX_DELAY_TAPS] = [0, 0, 1, 1, 0, 1, 2, 2, 3, 3];
const OUT_TAP_INDEXES_6CH: [usize; MAX_DELAY_TAPS] = [0, 0, 1, 1, 2, 2, 4, 4, 5, 5];
const LFE: usize = 3;
const CENTER: usize = 2;

fn apply<const N: usize>(
    params: &ReverbParameterV2,
    state: &mut ReverbState,
    mix: &mut [i32],
    inputs: &[usize; 6],
    outputs: &[usize; 6],
    count: usize,
) {
    let tap_indexes = match N {
        1 => &OUT_TAP_INDEXES_1CH,
        2 => &OUT_TAP_INDEXES_2CH,
        4 => &OUT_TAP_INDEXES_4CH,
        _ => &OUT_TAP_INDEXES_6CH,
    };
    let base_gain = Q14::from_raw(params.base_gain as i64);
    let late_gain = Q14::from_raw(params.late_gain as i64);
    let dry_gain = Q14::from_raw(params.dry_gain as i64);
    let wet_gain = Q14::from_raw(params.wet_gain as i64);
    for i in 0..count {
        let mut output_samples = [Q14::default(); N];
        for (tap, &target) in tap_indexes.iter().enumerate() {
            let sample =
                state.pre_delay_line.tap_out(state.early_delay_times[tap]) * state.early_gains[tap];
            output_samples[target] += sample;
            if N == 6 {
                output_samples[LFE] += sample;
            }
        }
        if N == 6 {
            output_samples[LFE] *= Q14::from_f32(0.2);
        }

        let mut input_sample = Q14::default();
        for &input in &inputs[..N] {
            input_sample += Q14::from_int(mix[input + i] as i64);
        }
        input_sample *= Q14::from_int(64);
        input_sample *= base_gain;
        state.pre_delay_line.write(input_sample);

        for line in 0..MAX_DELAY_LINES {
            state.prev_feedback_output[line] = state.prev_feedback_output[line]
                * state.hf_decay_prev_gain[line]
                + state.fdn_delay_lines[line].read() * state.hf_decay_gain[line];
        }
        let pre_delay_sample = state.pre_delay_line.tap_out(state.pre_delay_time) * late_gain;
        let p = state.prev_feedback_output;
        let mix_matrix = [
            p[2] + p[1] + pre_delay_sample,
            -p[0] - p[3] + pre_delay_sample,
            p[0] - p[3] + pre_delay_sample,
            p[1] - p[2] + pre_delay_sample,
        ];
        let mut allpass = [Q14::default(); MAX_DELAY_LINES];
        for line in 0..MAX_DELAY_LINES {
            allpass[line] = all_pass_tick(
                &mut state.decay_delay_lines[line],
                &mut state.fdn_delay_lines[line],
                mix_matrix[line],
            );
        }

        let allpass_outputs = if N == 6 {
            [
                allpass[0],
                allpass[1],
                allpass[2] - allpass[3],
                allpass[3],
                allpass[2],
                allpass[3],
            ]
        } else {
            [
                allpass[0],
                allpass[1],
                allpass[2],
                allpass[3],
                Q14::default(),
                Q14::default(),
            ]
        };
        for ch in 0..N {
            let in_sample = Q14::from_int(mix[inputs[ch] + i] as i64) * dry_gain;
            let wet = if N == 6 && ch == CENTER {
                state
                    .center_delay_line
                    .tick(allpass_outputs[ch] * Q14::from_f32(0.5))
            } else {
                allpass_outputs[ch]
            };
            let out_sample = ((output_samples[ch] + wet) * wet_gain) / Q14::from_int(64);
            mix[outputs[ch] + i] = (in_sample + out_sample).to_int();
        }
    }
}

pub(super) fn reverb(command: &ReverbCommand, frame: &mut Frame<'_>, state: &mut DspState) {
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
    let reverb_state = effect.reverb();
    if command.effect_enabled {
        match params.parameter_state() {
            Some(ParameterState::Updating) => update_parameters(params, reverb_state),
            Some(ParameterState::Initialized) => {
                initialize(params, reverb_state, command.long_size_pre_delay_supported)
            }
            _ => {}
        }
    }
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    match (command.effect_enabled, channels) {
        (true, 0) => {}
        (true, 1) => apply::<1>(params, reverb_state, mix, &inputs, &outputs, count),
        (true, 2) => apply::<2>(params, reverb_state, mix, &inputs, &outputs, count),
        (true, 4) => apply::<4>(params, reverb_state, mix, &inputs, &outputs, count),
        (true, 6) => apply::<6>(params, reverb_state, mix, &inputs, &outputs, count),
        _ => bypass(mix, &inputs, &outputs, channels, count),
    }
}
