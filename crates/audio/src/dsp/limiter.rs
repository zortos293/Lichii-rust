//! The light limiter effect, with optional statistics (`light_limiter.cpp`).

use super::MAX_CHANNELS;
use super::abi::{
    LightLimiterParameterV2, LimiterProcessingMode, LimiterStatistics, ParameterState,
};
use super::command::LightLimiterCommand;
use super::fixed::Q15;
use super::processor::{Frame, bypass, channel_buffers};
use super::state::DspState;

/// Light limiter state (`LightLimiterInfo::State`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LightLimiterState {
    pub samples_average: [Q15; MAX_CHANNELS],
    pub compression_gain: [Q15; MAX_CHANNELS],
    pub look_ahead_sample_offsets: [i32; MAX_CHANNELS],
    pub look_ahead_sample_buffers: [Vec<Q15>; MAX_CHANNELS],
}

fn initialize(params: &LightLimiterParameterV2, state: &mut LightLimiterState) {
    let mut buffers = std::mem::take(&mut state.look_ahead_sample_buffers);
    *state = LightLimiterState::default();
    state.samples_average = [Q15::from_f32(0.0); MAX_CHANNELS];
    state.compression_gain = [Q15::from_f32(1.0); MAX_CHANNELS];
    let channels = (params.channel_count as usize).min(MAX_CHANNELS);
    for (slot, buffer) in state
        .look_ahead_sample_buffers
        .iter_mut()
        .zip(&mut buffers)
        .take(channels)
    {
        let mut buffer = std::mem::take(buffer);
        buffer.clear();
        buffer.resize(
            params.look_ahead_samples_max.max(0) as usize,
            Q15::from_f32(0.0),
        );
        *slot = buffer;
    }
}

fn recip_estimate(a: f64) -> f64 {
    let q = (a * 512.0) as i32;
    let r = 1.0 / ((q as f64 + 0.5) / 512.0);
    let s = (256.0 * r + 0.5) as i32;
    s as f64 / 256.0
}

fn apply(
    params: &LightLimiterParameterV2,
    state: &mut LightLimiterState,
    mix: &mut [i32],
    inputs: &[usize; MAX_CHANNELS],
    outputs: &[usize; MAX_CHANNELS],
    count: usize,
    mut statistics: Option<&mut LimiterStatistics>,
) {
    let channels = (params.channel_count as usize).min(MAX_CHANNELS);
    if params.statistics_reset_required != 0
        && let Some(statistics) = statistics.as_deref_mut()
    {
        statistics.channel_compression_gain_min[..channels].fill(1.0);
        statistics.channel_max_sample[..channels].fill(0.0);
    }
    let one = Q15::from_int(Q15::ONE);
    let input_gain = Q15::from_f32(params.input_gain);
    let output_gain = Q15::from_f32(params.output_gain);
    let threshold = Q15::from_f32(params.threshold);
    let two = Q15::from_f64(2.0);
    for i in 0..count {
        for ch in 0..channels {
            let sample = (Q15::from_int(mix[inputs[ch] + i] as i64) / one) * input_gain;
            let abs_sample = if sample < Q15::from_f32(0.0) {
                -sample
            } else {
                sample
            };
            let average = &mut state.samples_average[ch];
            let coeff = if abs_sample > *average {
                params.attack_coeff
            } else {
                params.release_coeff
            };
            *average += Q15::from_f32(((abs_sample - *average) * Q15::from_f32(coeff)).to_f32());
            let mut new_average = Q15::from_f64(recip_estimate(average.to_f64()));
            if params.mode() != LimiterProcessingMode::Mode1 {
                let temp = two - *average * new_average;
                new_average = two - *average * temp;
            }
            let attenuation = if *average > threshold {
                threshold * new_average
            } else {
                Q15::from_f32(1.0)
            };
            let gain = &mut state.compression_gain[ch];
            let coeff = if attenuation < *gain {
                params.attack_coeff
            } else {
                params.release_coeff
            };
            *gain += (attenuation - *gain) * Q15::from_f32(coeff);

            let offset = &mut state.look_ahead_sample_offsets[ch];
            let buffer = &mut state.look_ahead_sample_buffers[ch];
            let lookahead = match buffer.get_mut(*offset as usize) {
                Some(slot) => std::mem::replace(slot, sample),
                None => Q15::default(),
            };
            *offset = (*offset + 1)
                .checked_rem(params.look_ahead_samples_min)
                .unwrap_or(0);
            let output = (lookahead * *gain * output_gain * one).to_long();
            mix[outputs[ch] + i] = output.clamp(i32::MIN as i64, i32::MAX as i64) as i32;

            if let Some(statistics) = statistics.as_deref_mut() {
                let max = &mut statistics.channel_max_sample[ch];
                if *max < abs_sample.to_f32() {
                    *max = abs_sample.to_f32();
                }
                let min = &mut statistics.channel_compression_gain_min[ch];
                if gain.to_f32() < *min {
                    *min = gain.to_f32();
                }
            }
        }
    }
}

pub(super) fn light_limiter(
    command: &LightLimiterCommand,
    with_statistics: bool,
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
    let limiter = effect.light_limiter();
    if command.effect_enabled && params.parameter_state() == Some(ParameterState::Initialized) {
        initialize(params, limiter);
    }
    let (mix, count) = (&mut *frame.mix, frame.sample_count);
    if !command.effect_enabled {
        bypass(mix, &inputs, &outputs, channels, count);
        return;
    }
    let result = command
        .result_state
        .filter(|_| with_statistics)
        .and_then(|id| state.effect_results.get_mut(id.0 as usize));
    match result {
        Some(result) => {
            let size = size_of::<LimiterStatistics>();
            let mut statistics: LimiterStatistics =
                bytemuck::pod_read_unaligned(&result.state[..size]);
            apply(
                params,
                limiter,
                mix,
                &inputs,
                &outputs,
                count,
                Some(&mut statistics),
            );
            result.state[..size].copy_from_slice(bytemuck::bytes_of(&statistics));
        }
        None => apply(params, limiter, mix, &inputs, &outputs, count, None),
    }
}
