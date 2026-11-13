//! Mix buffer arithmetic: volume, mixing, ramps, depop and the 5.1 downmix (`command/mix`,
//! `downmix_6ch_to_2ch.cpp`).

use std::ops::Range;

use super::MAX_MIX_BUFFERS;
use super::command::{
    CopyMixBufferCommand, DepopForMixBuffersCommand, DepopPrepareCommand, DownMix6chTo2chCommand,
    MixCommand, MixRampCommand, MixRampGroupedCommand, VolumeCommand, VolumeRampCommand,
};
use super::fixed::{Fx, Q15};
use super::processor::Frame;
use super::state::DspState;

fn apply_mix<const Q: u32>(
    mix: &mut [i32],
    output: usize,
    input: usize,
    volume: f32,
    count: usize,
) {
    let volume = Fx::<Q>::from_f32(volume);
    for i in 0..count {
        let sum = Fx::<Q>::from_int(mix[output + i] as i64)
            + Fx::from_int(mix[input + i] as i64) * volume;
        mix[output + i] = sum.to_int();
    }
}

pub(super) fn apply_mix_ramp<const Q: u32>(
    mix: &mut [i32],
    output: usize,
    input: usize,
    volume: f32,
    ramp: f32,
    count: usize,
) -> i32 {
    let mut volume = Fx::<Q>::from_f32(volume);
    let ramp = Fx::<Q>::from_f32(ramp);
    let mut sample = Fx::<Q>::default();
    for i in 0..count {
        sample = Fx::from_int(mix[input + i] as i64) * volume;
        mix[output + i] = (Fx::<Q>::from_int(mix[output + i] as i64) + sample).to_int();
        volume += ramp;
    }
    sample.to_int()
}

fn apply_gain<const Q: u32>(
    mix: &mut [i32],
    output: usize,
    input: usize,
    volume: f32,
    count: usize,
) {
    let gain = Fx::<Q>::from_f32(volume);
    for i in 0..count {
        mix[output + i] = (Fx::<Q>::from_int(mix[input + i] as i64) * gain).to_int();
    }
}

pub(super) fn apply_linear_envelope<const Q: u32>(
    mix: &mut [i32],
    output: usize,
    input: usize,
    volume: f32,
    ramp: f32,
    count: usize,
) {
    if volume == 0.0 && ramp == 0.0 {
        mix[output..output + count].fill(0);
    } else if volume == 1.0 && ramp == 0.0 {
        mix.copy_within(input..input + count, output);
    } else {
        let mut gain = Fx::<Q>::from_f32(volume);
        let ramp = Fx::<Q>::from_f32(ramp);
        for i in 0..count {
            mix[output + i] = (Fx::<Q>::from_int(mix[input + i] as i64) * gain).to_int();
            gain += ramp;
        }
    }
}

fn invalid_precision(precision: u8) {
    tracing::error!(precision, "invalid mix precision");
}

fn io(frame: &Frame<'_>, input: i16, output: i16) -> Option<(Range<usize>, Range<usize>)> {
    Some((frame.range(input)?, frame.range(output)?))
}

pub(super) fn volume(command: &VolumeCommand, frame: &mut Frame<'_>) {
    if command.input_index == command.output_index && command.volume == 1.0 {
        return;
    }
    let Some((input, output)) = io(frame, command.input_index, command.output_index) else {
        return;
    };
    let count = frame.sample_count;
    match command.precision {
        15 | 23 if command.volume == 1.0 => frame.mix.copy_within(input, output.start),
        15 => apply_gain::<15>(frame.mix, output.start, input.start, command.volume, count),
        23 => apply_gain::<23>(frame.mix, output.start, input.start, command.volume, count),
        precision => invalid_precision(precision),
    }
}

pub(super) fn volume_ramp(command: &VolumeRampCommand, frame: &mut Frame<'_>) {
    let ramp = (command.volume - command.prev_volume) / frame.sample_count as f32;
    if command.input_index == command.output_index && command.prev_volume == 1.0 && ramp == 0.0 {
        return;
    }
    let Some((input, output)) = io(frame, command.input_index, command.output_index) else {
        return;
    };
    let (mix, count, volume) = (&mut *frame.mix, frame.sample_count, command.prev_volume);
    match command.precision {
        15 => apply_linear_envelope::<15>(mix, output.start, input.start, volume, ramp, count),
        23 => apply_linear_envelope::<23>(mix, output.start, input.start, volume, ramp, count),
        precision => invalid_precision(precision),
    }
}

pub(super) fn mix(command: &MixCommand, frame: &mut Frame<'_>) {
    let Some((input, output)) = io(frame, command.input_index, command.output_index) else {
        return;
    };
    if command.volume == 0.0 {
        return;
    }
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    match command.precision {
        15 => apply_mix::<15>(mix, output.start, input.start, command.volume, count),
        23 => apply_mix::<23>(mix, output.start, input.start, command.volume, count),
        precision => invalid_precision(precision),
    }
}

/// Runs one ramped mix, returning the new previous sample, or `None` for an invalid precision.
fn mix_ramp_buffer(
    frame: &mut Frame<'_>,
    precision: u8,
    input: Range<usize>,
    output: Range<usize>,
    prev_volume: f32,
    ramp: f32,
) -> Option<i32> {
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    match precision {
        15 => Some(apply_mix_ramp::<15>(
            mix,
            output.start,
            input.start,
            prev_volume,
            ramp,
            count,
        )),
        23 => Some(apply_mix_ramp::<23>(
            mix,
            output.start,
            input.start,
            prev_volume,
            ramp,
            count,
        )),
        precision => {
            invalid_precision(precision);
            None
        }
    }
}

pub(super) fn mix_ramp(command: &MixRampCommand, frame: &mut Frame<'_>, state: &mut DspState) {
    let Some((input, output)) = io(frame, command.input_index, command.output_index) else {
        return;
    };
    let reference = command.previous_sample;
    let Some(previous) = state
        .voices
        .get_mut(reference.voice.0 as usize)
        .and_then(|voice| voice.previous_samples.get_mut(reference.index as usize))
    else {
        return;
    };
    let ramp = (command.volume - command.prev_volume) / frame.sample_count as f32;
    if command.prev_volume == 0.0 && ramp == 0.0 {
        *previous = 0;
        return;
    }
    if let Some(last) = mix_ramp_buffer(
        frame,
        command.precision,
        input,
        output,
        command.prev_volume,
        ramp,
    ) {
        *previous = last;
    }
}

pub(super) fn mix_ramp_grouped(
    command: &MixRampGroupedCommand,
    frame: &mut Frame<'_>,
    state: &mut DspState,
) {
    let Some(voice) = state.voices.get_mut(command.previous_samples.0 as usize) else {
        return;
    };
    let count = (command.buffer_count as usize).min(MAX_MIX_BUFFERS);
    for i in 0..count {
        let (prev_volume, volume) = (command.prev_volumes[i], command.volumes[i]);
        let mut last_sample = 0;
        if prev_volume != 0.0 || volume != 0.0 {
            let Some((input, output)) = io(frame, command.inputs[i], command.outputs[i]) else {
                continue;
            };
            let ramp = (volume - prev_volume) / frame.sample_count as f32;
            if prev_volume == 0.0 && ramp == 0.0 {
                voice.previous_samples[i] = 0;
                continue;
            }
            if let Some(last) =
                mix_ramp_buffer(frame, command.precision, input, output, prev_volume, ramp)
            {
                last_sample = last;
            }
        }
        voice.previous_samples[i] = last_sample;
    }
}

pub(super) fn clear_mix_buffers(frame: &mut Frame<'_>) {
    frame.mix.fill(0);
}

pub(super) fn copy_mix_buffer(command: &CopyMixBufferCommand, frame: &mut Frame<'_>) {
    if let Some((input, output)) = io(frame, command.input_index, command.output_index) {
        frame.mix.copy_within(input, output.start);
    }
}

pub(super) fn depop_prepare(command: &DepopPrepareCommand, state: &mut DspState) {
    let Some(voice) = state.voices.get_mut(command.previous_samples.0 as usize) else {
        return;
    };
    let count = (command.buffer_count as usize).min(MAX_MIX_BUFFERS);
    for (sample, &input) in voice.previous_samples[..count]
        .iter_mut()
        .zip(&command.inputs)
    {
        if *sample == 0 {
            continue;
        }
        if let Some(depop) = usize::try_from(input)
            .ok()
            .and_then(|i| state.depop.get_mut(i))
        {
            *depop = depop.wrapping_add(*sample);
        }
        *sample = 0;
    }
}

fn apply_depop_mix(output: &mut [i32], depop_sample: i32, decay: Q15) -> i32 {
    let mut sample = depop_sample.wrapping_abs();
    let decay = decay.raw();
    if depop_sample <= 0 {
        for out in output {
            sample = ((sample as i64 * decay) >> 15) as i32;
            *out = out.wrapping_sub(sample);
        }
        -sample
    } else {
        for out in output {
            sample = ((sample as i64 * decay) >> 15) as i32;
            *out = out.wrapping_add(sample);
        }
        sample
    }
}

pub(super) fn depop_for_mix_buffers(
    command: &DepopForMixBuffersCommand,
    frame: &mut Frame<'_>,
    state: &mut DspState,
) {
    let end = frame
        .buffer_count
        .min(command.input.saturating_add(command.count))
        .min(state.depop.len() as u32);
    for index in command.input..end {
        let depop_sample = state.depop[index as usize];
        if depop_sample == 0 {
            continue;
        }
        let Some(buffer) = frame.range(index as i64) else {
            continue;
        };
        state.depop[index as usize] =
            apply_depop_mix(&mut frame.mix[buffer], depop_sample, command.decay);
    }
}

pub(super) fn downmix_6ch_to_2ch(command: &DownMix6chTo2chCommand, frame: &mut Frame<'_>) {
    let mut inputs = [0usize; 6];
    let mut outputs = [0usize; 6];
    for channel in 0..6 {
        let (Some(input), Some(output)) = (
            frame.range(command.inputs[channel]),
            frame.range(command.outputs[channel]),
        ) else {
            return;
        };
        inputs[channel] = input.start;
        outputs[channel] = output.start;
    }
    let [front, center, lfe, back] = command.down_mix_coeff;
    let mix = &mut *frame.mix;
    let at = |mix: &[i32], buffer: usize, i: usize| Fx::<16>::from_int(mix[buffer + i] as i64);
    for i in 0..frame.sample_count {
        let shared = at(mix, inputs[2], i) * center + at(mix, inputs[3], i) * lfe;
        let left = at(mix, inputs[0], i) * front + shared + at(mix, inputs[4], i) * back;
        let right = at(mix, inputs[1], i) * front + shared + at(mix, inputs[5], i) * back;
        mix[outputs[0] + i] = left.to_int();
        mix[outputs[1] + i] = right.to_int();
    }
    for &output in &outputs[2..] {
        mix[output..output + frame.sample_count].fill(0);
    }
}
