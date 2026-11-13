//! The I3DL2 reverb effect (`i3dl2_reverb.cpp`).

#![allow(clippy::excessive_precision)]

use super::abi::{I3dl2ReverbParameterV1, ParameterState};
use super::command::I3dl2ReverbCommand;
use super::fixed::Q14;
use super::processor::{Frame, bypass, channel_buffers};
use super::reverb::{cos_degrees, pow_10};
use super::state::DspState;

const MAX_DELAY_LINES: usize = 4;
const MAX_DELAY_TAPS: usize = 20;

#[allow(clippy::approx_constant)]
const FRAC_1_SQRT_2: f32 = 0.7071;

const MIN_DELAY_LINE_TIMES: [f32; MAX_DELAY_LINES] = [5.0, 6.0, 13.0, 14.0];
const MAX_DELAY_LINE_TIMES: [f32; MAX_DELAY_LINES] =
    [45.7042007446, 82.7817001343, 149.938293457, 271.575805664];
const DECAY0_MAX_DELAY_LINE_TIMES: [f32; MAX_DELAY_LINES] = [17.0, 13.0, 9.0, 7.0];
const DECAY1_MAX_DELAY_LINE_TIMES: [f32; MAX_DELAY_LINES] = [19.0, 11.0, 10.0, 6.0];

#[rustfmt::skip]
const EARLY_TAP_TIMES: [f32; MAX_DELAY_TAPS] = [
    0.0171360000968, 0.0591540001333, 0.161733001471, 0.390186011791, 0.425262004137,
    0.455410987139, 0.689737021923, 0.74590998888, 0.833844006062, 0.859502017498,
    0.0, 0.0750240013003, 0.168788000941, 0.299901008606, 0.337442994118,
    0.371903002262, 0.599011003971, 0.716741025448, 0.817858994007, 0.85166400671,
];

#[rustfmt::skip]
const EARLY_GAINS: [f32; MAX_DELAY_TAPS] = [
    0.67096, 0.61027, 1.0, 0.3568, 0.68361, 0.65978, 0.51939, 0.24712, 0.45945, 0.45021,
    0.64196, 0.54879, 0.92925, 0.3827, 0.72867, 0.69794, 0.5464, 0.24563, 0.45214, 0.44042,
];

/// An I3DL2 delay line (`I3dl2ReverbInfo::I3dl2DelayLine`), with the C++ pointers as indices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct I3dl2DelayLine {
    pub buffer: Vec<Q14>,
    pub buffer_end: usize,
    pub max_delay: i32,
    pub input: usize,
    pub output: usize,
    pub delay: i32,
    pub wet_gain: f32,
}

impl I3dl2DelayLine {
    fn initialize(&mut self, delay_time: i32) {
        self.max_delay = delay_time;
        self.buffer.clear();
        self.buffer
            .resize(delay_time.max(0) as usize + 1, Q14::default());
        self.buffer_end = delay_time.max(0) as usize;
        self.output = 0;
        self.set_delay(delay_time);
        self.wet_gain = 0.0;
    }

    fn set_delay(&mut self, delay_time: i32) {
        if self.max_delay < delay_time {
            return;
        }
        self.delay = delay_time;
        let length = self.max_delay as i64 + 1;
        self.input = (self.output as i64 + delay_time as i64).rem_euclid(length) as usize;
    }

    fn tick(&mut self, sample: Q14) -> Q14 {
        self.write(sample);
        let out = self.read();
        self.output += 1;
        if self.output >= self.buffer_end {
            self.output = 0;
        }
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
        if self.input >= self.buffer_end {
            self.input = 0;
        }
    }

    fn tap_out(&self, index: i32) -> Q14 {
        let mut out = self.input as i64 - (index as i64 + 1);
        if out < 0 {
            out += self.max_delay as i64 + 1;
        }
        usize::try_from(out)
            .ok()
            .and_then(|out| self.buffer.get(out))
            .copied()
            .unwrap_or_default()
    }
}

/// I3DL2 reverb effect state (`I3dl2ReverbInfo::State`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct I3dl2State {
    pub lowpass_0: f32,
    pub lowpass_1: f32,
    pub lowpass_2: f32,
    pub early_delay_line: I3dl2DelayLine,
    pub early_tap_steps: [i32; MAX_DELAY_TAPS],
    pub early_gain: f32,
    pub late_gain: f32,
    pub early_to_late_taps: i32,
    pub fdn_delay_lines: [I3dl2DelayLine; MAX_DELAY_LINES],
    pub decay_delay_lines0: [I3dl2DelayLine; MAX_DELAY_LINES],
    pub decay_delay_lines1: [I3dl2DelayLine; MAX_DELAY_LINES],
    pub last_reverb_echo: f32,
    pub center_delay_line: I3dl2DelayLine,
    pub lowpass_coeff: [[f32; 3]; MAX_DELAY_LINES],
    pub shelf_filter: [f32; MAX_DELAY_LINES],
    pub dry_gain: f32,
}

/// `std::min` on floats: the first argument wins unless the second is strictly smaller.
fn min_f32(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
}

fn sin_degrees(degrees: f32) -> f32 {
    (degrees * std::f32::consts::PI / 180.0).sin()
}

fn update_parameters(params: &I3dl2ReverbParameterV1, state: &mut I3dl2State, reset: bool) {
    let sample_rate = params.sample_rate as f32;
    let delay = Q14::from_f32(sample_rate / 1000.0);
    state.dry_gain = params.dry_gain;
    let early_gain =
        Q14::from_f32(min_f32(params.room_gain + params.reflection_gain, 5000.0) / 2000.0);
    state.early_gain = pow_10(early_gain.to_f32());
    let late_gain = Q14::from_f32(min_f32(params.room_gain + params.reverb_gain, 5000.0) / 2000.0);
    state.late_gain = pow_10(late_gain.to_f32());
    let hf_gain = Q14::from_f32(pow_10(params.room_hf_gain / 2000.0));
    if hf_gain >= Q14::from_f32(1.0) {
        state.lowpass_1 = 0.0;
        state.lowpass_2 = 1.0;
    } else {
        let reference_hf = (params.reference_hf * 256.0) / sample_rate;
        let a = Q14::from_f32(1.0 - hf_gain.to_f32());
        let b = Q14::from_f32(2.0)
            + Q14::from_f32(-cos_degrees(reference_hf)) * (hf_gain * Q14::from_f32(2.0));
        let (bf, af) = (b.to_f32(), a.to_f32());
        let c = Q14::from_f32((bf * bf + (af * af) * -4.0).sqrt());
        state.lowpass_1 = min_f32(((b - c) / (a * Q14::from_f32(2.0))).to_f32(), 0.99723);
        state.lowpass_2 = 1.0 - state.lowpass_1;
    }
    state.early_to_late_taps =
        (Q14::from_f32((params.reflection_delay + params.late_reverb_delay_time) * 1000.0) * delay)
            .to_int();
    state.last_reverb_echo = params.late_reverb_diffusion * 0.6 * 0.01;
    let cutoff = ((params.reference_hf * 0.5) * 128.0) / sample_rate;
    for i in 0..MAX_DELAY_LINES {
        let length = MIN_DELAY_LINE_TIMES[i]
            + (params.late_reverb_density / 100.0)
                * (MAX_DELAY_LINE_TIMES[i] - MIN_DELAY_LINE_TIMES[i]);
        state.fdn_delay_lines[i].set_delay((Q14::from_f32(length) * delay).to_int());
        let total_delay = state.fdn_delay_lines[i].delay
            + state.decay_delay_lines0[i].delay
            + state.decay_delay_lines1[i].delay;
        let a = (total_delay as f32 * -60.0) / (params.late_reverb_decay_time * sample_rate);
        let b = a / params.late_reverb_hf_decay_ratio;
        let c = cos_degrees(cutoff) / sin_degrees(cutoff);
        let d = pow_10((b - a) / 40.0);
        let e = pow_10((b + a) / 40.0) * FRAC_1_SQRT_2;
        state.lowpass_coeff[i][0] = ((c * d + 1.0) * e) / (c + d);
        state.lowpass_coeff[i][1] = ((1.0 - (c * d)) * e) / (c + d);
        state.lowpass_coeff[i][2] = (c - d) / (c + d);
        state.decay_delay_lines0[i].wet_gain = state.last_reverb_echo;
        state.decay_delay_lines1[i].wet_gain = state.last_reverb_echo * -0.9;
    }
    if reset {
        state.shelf_filter = [0.0; MAX_DELAY_LINES];
        state.lowpass_0 = 0.0;
        let lines = state
            .fdn_delay_lines
            .iter_mut()
            .chain(&mut state.decay_delay_lines0)
            .chain(&mut state.decay_delay_lines1)
            .chain([&mut state.center_delay_line, &mut state.early_delay_line]);
        for line in lines {
            line.buffer.fill(Q14::default());
        }
    }
    let reflection_time = (params.late_reverb_delay_time * 0.9998 + 0.02) * 1000.0;
    let reflection_delay = params.reflection_delay * 1000.0;
    for (step, &tap_time) in state.early_tap_steps.iter_mut().zip(&EARLY_TAP_TIMES) {
        let length =
            (Q14::from_f32(reflection_delay + reflection_time * tap_time) * delay).to_int();
        *step = length.min(state.early_delay_line.max_delay);
    }
}

fn initialize(params: &I3dl2ReverbParameterV1, state: &mut I3dl2State) {
    let mut old = std::mem::take(state);
    let delay = Q14::from_f32(params.sample_rate as f32 / 1000.0);
    let line = |buffer: &mut Vec<Q14>, time: f32| {
        let mut line = I3dl2DelayLine {
            buffer: std::mem::take(buffer),
            ..Default::default()
        };
        line.initialize((Q14::from_f32(time) * delay).to_uint_floor() as i32);
        line
    };
    for i in 0..MAX_DELAY_LINES {
        state.fdn_delay_lines[i] =
            line(&mut old.fdn_delay_lines[i].buffer, MAX_DELAY_LINE_TIMES[i]);
        state.decay_delay_lines0[i] = line(
            &mut old.decay_delay_lines0[i].buffer,
            DECAY0_MAX_DELAY_LINE_TIMES[i],
        );
        state.decay_delay_lines1[i] = line(
            &mut old.decay_delay_lines1[i].buffer,
            DECAY1_MAX_DELAY_LINE_TIMES[i],
        );
    }
    state.center_delay_line = line(&mut old.center_delay_line.buffer, 5.0);
    state.early_delay_line = line(&mut old.early_delay_line.buffer, 400.0);
    update_parameters(params, state, true);
}

fn all_pass_tick(
    decay0: &mut I3dl2DelayLine,
    decay1: &mut I3dl2DelayLine,
    fdn: &mut I3dl2DelayLine,
    mix: Q14,
) -> Q14 {
    let gain0 = Q14::from_f32(decay0.wet_gain);
    let value = decay0.read();
    let mixed = mix - value * gain0;
    let out = decay0.tick(mixed) + mixed * gain0;
    let gain1 = Q14::from_f32(decay1.wet_gain);
    let value = decay1.read();
    let mixed = out - value * gain1;
    let out = decay1.tick(mixed) + mixed * gain1;
    fdn.tick(out);
    out
}

#[rustfmt::skip]
const OUT_TAP_INDEXES: [[usize; MAX_DELAY_TAPS]; 4] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 3, 3, 3],
    [2, 0, 0, 1, 1, 1, 1, 4, 4, 4, 1, 1, 1, 0, 0, 0, 0, 5, 5, 5],
];
const LFE: usize = 3;
const CENTER: usize = 2;

fn apply<const N: usize>(
    state: &mut I3dl2State,
    mix: &mut [i32],
    inputs: &[usize; 6],
    outputs: &[usize; 6],
    count: usize,
) {
    let tap_indexes = match N {
        1 => &OUT_TAP_INDEXES[0],
        2 => &OUT_TAP_INDEXES[1],
        4 => &OUT_TAP_INDEXES[2],
        _ => &OUT_TAP_INDEXES[3],
    };
    for i in 0..count {
        let early_to_late_tap = state.early_delay_line.tap_out(state.early_to_late_taps);
        let mut output_samples = [Q14::default(); N];
        for (tap, &target) in tap_indexes.iter().enumerate() {
            let sample = state.early_delay_line.tap_out(state.early_tap_steps[tap])
                * Q14::from_f32(EARLY_GAINS[tap]);
            output_samples[target] += sample;
            if N == 6 {
                output_samples[LFE] += sample;
            }
        }
        let mut current_sample = Q14::default();
        for &input in &inputs[..N] {
            current_sample += Q14::from_int(mix[input + i] as i64);
        }
        state.lowpass_0 = (current_sample * Q14::from_f32(state.lowpass_2)
            + Q14::from_f32(state.lowpass_0 * state.lowpass_1))
        .to_f32();
        state.early_delay_line.tick(Q14::from_f32(state.lowpass_0));
        for sample in &mut output_samples {
            *sample *= Q14::from_f32(state.early_gain);
        }

        let mut filtered = [Q14::default(); MAX_DELAY_LINES];
        for line in 0..MAX_DELAY_LINES {
            let coeff = state.lowpass_coeff[line];
            let fdn_out = state.fdn_delay_lines[line].read();
            filtered[line] =
                fdn_out * Q14::from_f32(coeff[0]) + Q14::from_f32(state.shelf_filter[line]);
            state.shelf_filter[line] = (filtered[line] * Q14::from_f32(coeff[2])
                + fdn_out * Q14::from_f32(coeff[1]))
            .to_f32();
        }
        let late = early_to_late_tap * Q14::from_f32(state.late_gain);
        let f = filtered;
        let mix_matrix = [
            f[1] + f[2] + late,
            -f[0] - f[3] + late,
            f[0] - f[3] + late,
            f[1] - f[2] + late,
        ];
        let mut allpass = [Q14::default(); MAX_DELAY_LINES];
        for line in 0..MAX_DELAY_LINES {
            allpass[line] = all_pass_tick(
                &mut state.decay_delay_lines0[line],
                &mut state.decay_delay_lines1[line],
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
            let wet = if N == 6 && ch == CENTER {
                state
                    .center_delay_line
                    .tick(allpass_outputs[ch] * Q14::from_f32(0.5))
            } else {
                allpass_outputs[ch]
            };
            let dry = Q14::from_f32(state.dry_gain * mix[inputs[ch] + i] as f32);
            let out_sample = output_samples[ch] + wet + dry;
            mix[outputs[ch] + i] = out_sample.to_f32().clamp(-8388600.0, 8388600.0) as i32;
        }
    }
}

pub(super) fn i3dl2_reverb(
    command: &I3dl2ReverbCommand,
    frame: &mut Frame<'_>,
    state: &mut DspState,
) {
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
    let i3dl2_state = effect.i3dl2_reverb();
    if command.effect_enabled {
        match params.parameter_state() {
            Some(ParameterState::Updating) => update_parameters(params, i3dl2_state, false),
            Some(ParameterState::Initialized) => initialize(params, i3dl2_state),
            _ => {}
        }
    }
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    match (command.effect_enabled, channels) {
        (true, 0) => {}
        (true, 1) => apply::<1>(i3dl2_state, mix, &inputs, &outputs, count),
        (true, 2) => apply::<2>(i3dl2_state, mix, &inputs, &outputs, count),
        (true, 4) => apply::<4>(i3dl2_state, mix, &inputs, &outputs, count),
        (true, 6) => apply::<6>(i3dl2_state, mix, &inputs, &outputs, count),
        _ => bypass(mix, &inputs, &outputs, channels, count),
    }
}
