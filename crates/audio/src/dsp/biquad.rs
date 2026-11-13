//! Biquad filters for voices and the biquad filter effect (`biquad_filter.cpp`,
//! `multi_tap_biquad_filter.cpp`).

use super::MAX_BIQUAD_FILTERS;
use super::command::{BiquadFilterCommand, BiquadStateRef, MultiTapBiquadFilterCommand};
use super::fixed::Fx;
use super::processor::Frame;
use super::state::DspState;

/// A biquad's delay elements. The integer path keeps two Q14 accumulators in `s0`/`s1`; the float
/// paths keep four `f64` values bit-cast into all four fields.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BiquadFilterState {
    pub s0: i64,
    pub s1: i64,
    pub s2: i64,
    pub s3: i64,
}

fn run_float(
    mix: &mut [i32],
    output: usize,
    input: usize,
    b: [f64; 3],
    a: [f64; 2],
    state: &mut BiquadFilterState,
    count: usize,
) {
    let mut s = [state.s0, state.s1, state.s2, state.s3].map(|v| f64::from_bits(v as u64));
    for i in 0..count {
        let in_sample = mix[input + i] as f64;
        let sample = in_sample * b[0] + s[0] * b[1] + s[1] * b[2] + s[2] * a[0] + s[3] * a[1];
        mix[output + i] = sample.clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        s[1] = s[0];
        s[0] = in_sample;
        s[3] = s[2];
        s[2] = sample;
    }
    [state.s0, state.s1, state.s2, state.s3] = s.map(|v| v.to_bits() as i64);
}

/// Float processing with Q2.14 coefficients (`ApplyBiquadFilterFloat`).
fn run_float_q14(
    mix: &mut [i32],
    output: usize,
    input: usize,
    b: [i16; 3],
    a: [i16; 2],
    state: &mut BiquadFilterState,
    count: usize,
) {
    let q14 = |v: i16| Fx::<14>::from_raw(v as i64).to_f64();
    run_float(mix, output, input, b.map(q14), a.map(q14), state, count);
}

/// Float processing with float coefficients (`ApplyBiquadFilterFloat2`).
fn run_float_f32(
    mix: &mut [i32],
    output: usize,
    input: usize,
    b: [f32; 3],
    a: [f32; 2],
    state: &mut BiquadFilterState,
    count: usize,
) {
    run_float(
        mix,
        output,
        input,
        b.map(f64::from),
        a.map(f64::from),
        state,
        count,
    );
}

fn run_int(
    mix: &mut [i32],
    output: usize,
    input: usize,
    b: [i16; 3],
    a: [i16; 2],
    state: &mut BiquadFilterState,
    count: usize,
) {
    for i in 0..count {
        let in_sample = mix[input + i] as i64;
        let sample = in_sample.wrapping_mul(b[0] as i64).wrapping_add(state.s0);
        let out_sample =
            (sample.wrapping_add(1 << 13) >> 14).clamp(i32::MIN as i64, i32::MAX as i64);
        mix[output + i] = out_sample as i32;
        state.s0 = state
            .s1
            .wrapping_add((b[1] as i64).wrapping_mul(in_sample))
            .wrapping_add((a[0] as i64).wrapping_mul(out_sample));
        state.s1 = (b[2] as i64)
            .wrapping_mul(in_sample)
            .wrapping_add((a[1] as i64).wrapping_mul(out_sample));
    }
}

fn resolve(state: &mut DspState, reference: BiquadStateRef) -> Option<&mut BiquadFilterState> {
    match reference {
        BiquadStateRef::Voice { voice, filter } => state
            .voices
            .get_mut(voice.0 as usize)?
            .biquad_states
            .get_mut(filter as usize)
            .map(|filters| &mut filters[0]),
        BiquadStateRef::Effect { effect, channel } => state
            .effects
            .get_mut(effect.0 as usize)?
            .biquad_filter()
            .get_mut(channel as usize),
    }
}

pub(super) fn biquad_filter(
    command: &BiquadFilterCommand,
    frame: &mut Frame<'_>,
    state: &mut DspState,
) {
    let (Some(input), Some(output)) = (frame.range(command.input), frame.range(command.output))
    else {
        return;
    };
    let Some(filter) = resolve(state, command.state) else {
        return;
    };
    if command.needs_init {
        *filter = BiquadFilterState::default();
    }
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    let (b, a) = (command.biquad.b, command.biquad.a);
    if !command.use_float_processing {
        run_int(mix, output.start, input.start, b, a, filter, count);
    } else if command.use_float_coefficients {
        let float = &command.biquad_float;
        run_float_f32(
            mix,
            output.start,
            input.start,
            float.numerator,
            float.denominator,
            filter,
            count,
        );
    } else {
        run_float_q14(mix, output.start, input.start, b, a, filter, count);
    }
}

pub(super) fn multi_tap_biquad_filter(
    command: &MultiTapBiquadFilterCommand,
    frame: &mut Frame<'_>,
    state: &mut DspState,
) {
    let (Some(input), Some(output)) = (frame.range(command.input), frame.range(command.output))
    else {
        return;
    };
    let taps = (command.filter_tap_count as usize).min(MAX_BIQUAD_FILTERS);
    for tap in 0..taps {
        let Some(filter) = resolve(state, command.states[tap]) else {
            continue;
        };
        if command.needs_init[tap] {
            *filter = BiquadFilterState::default();
        }
        let (mix, count) = (&mut *frame.mix, frame.sample_count);
        if command.use_float_coefficients {
            let float = &command.biquads_float[tap];
            run_float_f32(
                mix,
                output.start,
                input.start,
                float.numerator,
                float.denominator,
                filter,
                count,
            );
        } else {
            let fixed = &command.biquads[tap];
            run_float_q14(
                mix,
                output.start,
                input.start,
                fixed.b,
                fixed.a,
                filter,
                count,
            );
        }
    }
}
