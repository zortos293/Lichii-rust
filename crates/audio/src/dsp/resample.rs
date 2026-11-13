//! Sample-rate conversion of decoded voice samples (`resample.cpp`).

use super::fixed::{Fx, Q15};
use super::resample_lut::{HIGH_LUT0, HIGH_LUT1, HIGH_LUT2, NORMAL_LUT0, NORMAL_LUT1, NORMAL_LUT2};

/// Sample-rate conversion quality of a voice (`SrcQuality`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SrcQuality {
    #[default]
    Medium = 0,
    High = 1,
    Low = 2,
}

impl SrcQuality {
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            1 => Self::High,
            2 => Self::Low,
            _ => Self::Medium,
        }
    }

    /// Samples of history the resampler needs before the first new sample.
    pub(super) fn history_len(self) -> usize {
        match self {
            Self::High => 8,
            Self::Medium | Self::Low => 4,
        }
    }
}

/// Converts `samples_to_write` output samples from `input` at `ratio` input samples per output
/// sample, carrying the sub-sample position in `fraction`.
pub fn resample(
    output: &mut [i32],
    input: &[i16],
    ratio: Q15,
    fraction: &mut Q15,
    samples_to_write: usize,
    quality: SrcQuality,
) {
    match quality {
        SrcQuality::Low => resample_low(output, input, ratio, fraction, samples_to_write),
        SrcQuality::Medium => {
            let lut = select_lut(ratio, &NORMAL_LUT0, &NORMAL_LUT1, &NORMAL_LUT2);
            resample_fir::<4>(output, input, ratio, fraction, samples_to_write, lut);
        }
        SrcQuality::High => {
            let lut = select_lut(ratio, &HIGH_LUT0, &HIGH_LUT1, &HIGH_LUT2);
            resample_fir::<8>(output, input, ratio, fraction, samples_to_write, lut);
        }
    }
}

fn resample_low(
    output: &mut [i32],
    input: &[i16],
    ratio: Q15,
    fraction: &mut Q15,
    samples_to_write: usize,
) {
    let half = Q15::from_f32(0.5);
    let mut read_index = 0usize;
    for out in &mut output[..samples_to_write] {
        *out = input[read_index + usize::from(*fraction >= half)] as i32;
        *fraction += ratio;
        read_index += fraction.to_int_floor() as u32 as usize;
        *fraction = fraction.clear_int();
    }
}

fn select_lut<'a>(ratio: Q15, downsample: &'a [f32], mild: &'a [f32], up: &'a [f32]) -> &'a [f32] {
    if ratio <= Q15::from_f32(1.0) {
        up
    } else if ratio < Q15::from_f32(1.3) {
        mild
    } else {
        downsample
    }
}

fn resample_fir<const TAPS: usize>(
    output: &mut [i32],
    input: &[i16],
    ratio: Q15,
    fraction: &mut Q15,
    samples_to_write: usize,
    lut: &[f32],
) {
    let shift = TAPS.trailing_zeros();
    let mask = lut.len() - 1;
    let mut read_index = 0usize;
    for out in &mut output[..samples_to_write] {
        let lut_index = (((fraction.frac() >> 8) as usize) << shift) & mask;
        let taps = &input[read_index..read_index + TAPS];
        let coefficients = &lut[lut_index..lut_index + TAPS];
        let mut sum = Fx::<8>::default();
        for (&sample, &coefficient) in taps.iter().zip(coefficients) {
            sum += Fx::<8>::from_f32(sample as f32 * coefficient);
        }
        *out = sum.to_int_floor();
        *fraction += ratio;
        read_index += fraction.to_int_floor() as u32 as usize;
        *fraction = fraction.clear_int();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize) -> Vec<i16> {
        (0..len).map(|i| (i as i16 - 16) * 1000).collect()
    }

    #[test]
    fn low_quality_picks_nearest_sample() {
        let input = ramp(32);
        let mut output = [0; 6];
        let mut fraction = Q15::default();
        resample(
            &mut output,
            &input,
            Q15::from_f32(1.5),
            &mut fraction,
            6,
            SrcQuality::Low,
        );
        assert_eq!(output, [-16000, -14000, -13000, -11000, -10000, -8000]);
        assert_eq!(fraction, Q15::default());
    }

    /// At a zero fraction the upsampling tables are a near-identity kernel centred on tap 1
    /// (medium) and tap 3 (high). Expected values come from an independent Python model of the
    /// C++ float-times-table, truncate-to-Q8 arithmetic.
    #[test]
    fn unity_ratio_selects_the_center_tap() {
        let input = ramp(32);
        let mut medium = [0; 4];
        let mut fraction = Q15::default();
        resample(
            &mut medium,
            &input,
            Q15::from_f32(1.0),
            &mut fraction,
            4,
            SrcQuality::Medium,
        );
        assert_eq!(medium, [-15003, -14003, -13003, -12002]);
        let mut high = [0; 4];
        let mut fraction = Q15::default();
        resample(
            &mut high,
            &input,
            Q15::from_f32(1.0),
            &mut fraction,
            4,
            SrcQuality::High,
        );
        assert_eq!(high, [-12983, -11984, -10985, -9986]);
    }

    /// 32 kHz to 48 kHz at every quality; the snapshot matches the Python model above.
    #[test]
    fn golden_outputs_per_quality() {
        let input: Vec<i16> = (0..64).map(|i| ((i * 2731) % 8191 - 4096) as i16).collect();
        let ratio = Q15::from_f32(32000.0 / 48000.0);
        let mut snapshots = Vec::new();
        for quality in [SrcQuality::Low, SrcQuality::Medium, SrcQuality::High] {
            let mut output = [0; 12];
            let mut fraction = Q15::from_raw(1234);
            resample(&mut output, &input, ratio, &mut fraction, 12, quality);
            snapshots.push((output, fraction.raw()));
        }
        assert_eq!(snapshots, GOLDEN_RESAMPLE);
    }

    #[rustfmt::skip]
    const GOLDEN_RESAMPLE: [([i32; 12], i64); 3] = [
        ([-4096, -1365, -1365, 1366, -4094, -4094, -1363, 1368, 1368, -4092, -1361, -1361], 1230),
        ([-1261, 524, -705, -3330, -2227, -272, 498, -2389, -3114, -1257, 528, -701], 1230),
        ([-4189, -3010, 660, 1257, -2724, -4252, -1152, 1650, -491, -4185, -3006, 664], 1230),
    ];
}
