//! Golden tests against the C++ renderer.
//!
//! \`golden/cref.txt\` is the output of a standalone harness that compiles the C++ DSP functions
//! (\`renderer/command/**\` of Lichii \`nx2\`) and runs them on the synthetic inputs built here: the
//! same tone, parameters, wave buffers and guest memory image. Each test runs the same case
//! through [\`CommandListProcessor\`], prints it in the harness's line format and compares it line
//! by line with the C++ output, so a mismatch shows the first differing sample and its hash.

use std::fmt::Write;

use super::abi::{
    CompressorParameterV2, DelayParameterV1, I3dl2ReverbParameterV1, LightLimiterParameterV2,
    LimiterStatistics, ParameterState, ReverbParameterV2,
};
use super::fixed::Q15;
use super::*;

const SC: usize = 240;
const GOLDEN: &str = include_str!("golden/cref.txt");

fn tone(i: usize, ch: usize) -> i32 {
    (((i * 37 + ch * 101) % 211) as i32 - 105) * 1200
}

fn fnv(samples: &[i32], mut hash: u64) -> u64 {
    for &sample in samples {
        for byte in (sample as u32).to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
    }
    hash
}

const FNV_OFFSET: u64 = 1469598103934665603;

fn list(sample_count: usize, commands: Vec<Command>) -> CommandList {
    CommandList {
        sample_count: sample_count as u32,
        sample_rate: TARGET_SAMPLE_RATE,
        buffer_count: MAX_MIX_BUFFERS as u32,
        commands: commands
            .into_iter()
            .map(|command| CommandEntry::new(0, command))
            .collect(),
    }
}

struct FixedClock;

impl DspClock for FixedClock {
    fn now_us(&self) -> u64 {
        0
    }
}

struct Rig {
    processor: CommandListProcessor,
    state: DspState,
    memory: VecMemory,
    mix: Vec<i32>,
    device: Vec<i16>,
    sample_count: usize,
}

impl Rig {
    fn new(sample_count: usize) -> Self {
        Self {
            processor: CommandListProcessor::new(),
            state: DspState::new(2, MAX_MIX_BUFFERS, 1, 1),
            memory: VecMemory::new(0, 0x10000),
            mix: vec![0; MAX_MIX_BUFFERS * sample_count],
            device: vec![0; SC * MAX_CHANNELS],
            sample_count,
        }
    }

    fn buffer(&mut self, index: usize) -> &mut [i32] {
        &mut self.mix[index * self.sample_count..(index + 1) * self.sample_count]
    }

    fn fill_tone(&mut self, frame: usize, buffer: usize, ch: usize) {
        let sc = self.sample_count;
        for (i, sample) in self.buffer(buffer).iter_mut().enumerate() {
            *sample = tone(frame * sc + i, ch);
        }
    }

    fn run(&mut self, commands: Vec<Command>) -> ProcessReport {
        let list = list(self.sample_count, commands);
        self.processor.process(
            &list,
            DspFrame {
                mix_buffers: &mut self.mix,
                state: &mut self.state,
                memory: &mut self.memory,
                clock: &FixedClock,
                device_output: &mut self.device,
            },
        )
    }

    fn report(&self, out: &mut String, name: &str, frame: usize, buffers: &[usize]) {
        let sc = self.sample_count;
        let hash = buffers.iter().fold(FNV_OFFSET, |hash, &b| {
            fnv(&self.mix[b * sc..(b + 1) * sc], hash)
        });
        let first = &self.mix[buffers[0] * sc..][..8];
        writeln!(
            out,
            "{name} frame={frame} hash=0x{hash:016x} first={first:?}"
        )
        .unwrap();
    }
}

fn expected(names: &[&str]) -> Vec<&'static str> {
    GOLDEN
        .lines()
        .filter(|line| names.contains(&line.split(' ').next().unwrap_or("")))
        .collect()
}

fn check(names: &[&str], actual: &str) {
    let expected = expected(names);
    assert!(!expected.is_empty(), "no golden lines for {names:?}");
    let actual: Vec<&str> = actual.lines().collect();
    for (i, (want, got)) in expected.iter().zip(&actual).enumerate() {
        assert_eq!(got, want, "line {i} differs from the C++ reference");
    }
    assert_eq!(actual.len(), expected.len(), "line count differs");
}

fn channel_indices(channels: &[i16]) -> [i16; MAX_CHANNELS] {
    std::array::from_fn(|i| channels.get(i).copied().unwrap_or(0))
}

fn param_channels(channels: &[i16]) -> [i8; MAX_CHANNELS] {
    channel_indices(channels).map(|c| c as i8)
}

fn state_for(frame: usize, states: &[ParameterState]) -> u8 {
    states[frame.min(states.len() - 1)].raw()
}

const INIT_THEN_UPDATED: &[ParameterState] =
    &[ParameterState::Initialized, ParameterState::Updated];
const INIT_UPDATING_UPDATED: &[ParameterState] = &[
    ParameterState::Initialized,
    ParameterState::Updating,
    ParameterState::Updated,
];

#[test]
fn delay_matches_cpp() {
    let mut out = String::new();
    let mut rig = Rig::new(SC);
    let mut p = DelayParameterV1 {
        inputs: param_channels(&[0, 1]),
        outputs: param_channels(&[2, 3]),
        channel_count_max: 2,
        channel_count: 2,
        delay_time_max: 10,
        delay_time: 1,
        sample_rate: 48000 << 14,
        in_gain: 12288,
        feedback_gain: 8192,
        wet_gain: 9830,
        dry_gain: 6553,
        channel_spread: 4096,
        lowpass_amount: 4915,
        ..Default::default()
    };
    for frame in 0..2 {
        p.state = state_for(frame, INIT_THEN_UPDATED);
        rig.fill_tone(frame, 0, 0);
        rig.fill_tone(frame, 1, 1);
        rig.run(vec![Command::Delay(DelayCommand {
            inputs: channel_indices(&[0, 1]),
            outputs: channel_indices(&[2, 3]),
            parameter: p,
            state: EffectStateId(0),
            effect_enabled: true,
        })]);
        rig.report(&mut out, "delay2", frame, &[2, 3]);
    }

    let mut rig = Rig::new(SC);
    let six = [0, 1, 2, 3, 4, 5];
    let p = DelayParameterV1 {
        inputs: param_channels(&six),
        outputs: param_channels(&six),
        channel_count_max: 6,
        channel_count: 6,
        delay_time_max: 10,
        delay_time: 2,
        sample_rate: 48000 << 14,
        in_gain: 16384,
        feedback_gain: 10000,
        wet_gain: 8192,
        dry_gain: 8192,
        channel_spread: 8192,
        lowpass_amount: 2000,
        state: ParameterState::Initialized.raw(),
        ..Default::default()
    };
    for ch in 0..6 {
        rig.fill_tone(0, ch, ch);
    }
    rig.run(vec![Command::Delay(DelayCommand {
        inputs: six,
        outputs: six,
        parameter: p,
        state: EffectStateId(0),
        effect_enabled: true,
    })]);
    rig.report(&mut out, "delay6", 0, &[0, 1, 2, 3, 4, 5]);
    check(&["delay2", "delay6"], &out);
}

#[test]
fn reverb_matches_cpp() {
    let mut out = String::new();
    let mut rig = Rig::new(SC);
    let mut p = ReverbParameterV2 {
        inputs: param_channels(&[0, 1]),
        outputs: param_channels(&[2, 3]),
        channel_count_max: 2,
        channel_count: 2,
        sample_rate: 48 << 14,
        early_mode: 1,
        early_gain: 13107,
        pre_delay: 10 << 14,
        late_mode: 1,
        late_gain: 11468,
        decay_time: 32768,
        high_freq_decay_ratio: 8192,
        colouration: 4915,
        base_gain: 8192,
        wet_gain: 9830,
        dry_gain: 8192,
        ..Default::default()
    };
    for frame in 0..3 {
        p.state = state_for(frame, INIT_UPDATING_UPDATED);
        rig.fill_tone(frame, 0, 0);
        rig.fill_tone(frame, 1, 1);
        rig.run(vec![Command::Reverb(ReverbCommand {
            inputs: channel_indices(&[0, 1]),
            outputs: channel_indices(&[2, 3]),
            parameter: p,
            state: EffectStateId(0),
            effect_enabled: true,
            long_size_pre_delay_supported: false,
        })]);
        rig.report(&mut out, "reverb2", frame, &[2, 3]);
    }

    let mut rig = Rig::new(SC);
    let six = [0, 1, 2, 3, 4, 5];
    let mut p = ReverbParameterV2 {
        inputs: param_channels(&six),
        outputs: param_channels(&six),
        channel_count_max: 6,
        channel_count: 6,
        sample_rate: 48 << 14,
        early_mode: 3,
        early_gain: 16000,
        pre_delay: 3 << 14,
        late_mode: 3,
        late_gain: 9000,
        decay_time: 20000,
        high_freq_decay_ratio: 16300,
        colouration: 12000,
        base_gain: 12000,
        wet_gain: 12000,
        dry_gain: 4000,
        ..Default::default()
    };
    for frame in 0..2 {
        p.state = state_for(frame, INIT_THEN_UPDATED);
        for ch in 0..6 {
            rig.fill_tone(frame, ch, ch);
        }
        rig.run(vec![Command::Reverb(ReverbCommand {
            inputs: six,
            outputs: six,
            parameter: p,
            state: EffectStateId(0),
            effect_enabled: true,
            long_size_pre_delay_supported: true,
        })]);
        rig.report(&mut out, "reverb6", frame, &six.map(|c| c as usize));
    }
    check(&["reverb2", "reverb6"], &out);
}

#[test]
fn i3dl2_reverb_matches_cpp() {
    let mut out = String::new();
    let mut rig = Rig::new(SC);
    let mut p = I3dl2ReverbParameterV1 {
        inputs: param_channels(&[0, 1]),
        outputs: param_channels(&[2, 3]),
        channel_count_max: 2,
        channel_count: 2,
        sample_rate: 48000,
        room_hf_gain: -500.0,
        reference_hf: 5000.0,
        late_reverb_decay_time: 1.5,
        late_reverb_hf_decay_ratio: 0.6,
        room_gain: -1000.0,
        reflection_gain: -800.0,
        reverb_gain: -600.0,
        late_reverb_diffusion: 80.0,
        reflection_delay: 0.01,
        late_reverb_delay_time: 0.02,
        late_reverb_density: 70.0,
        dry_gain: 0.8,
        ..Default::default()
    };
    for frame in 0..3 {
        p.state = state_for(frame, INIT_UPDATING_UPDATED);
        rig.fill_tone(frame, 0, 0);
        rig.fill_tone(frame, 1, 1);
        rig.run(vec![Command::I3dl2Reverb(I3dl2ReverbCommand {
            inputs: channel_indices(&[0, 1]),
            outputs: channel_indices(&[2, 3]),
            parameter: p,
            state: EffectStateId(0),
            effect_enabled: true,
        })]);
        rig.report(&mut out, "i3dl2_2", frame, &[2, 3]);
    }

    let mut rig = Rig::new(SC);
    let six = [0, 1, 2, 3, 4, 5];
    let p = I3dl2ReverbParameterV1 {
        inputs: param_channels(&six),
        outputs: param_channels(&six),
        channel_count_max: 6,
        channel_count: 6,
        sample_rate: 48000,
        room_hf_gain: 0.0,
        reference_hf: 3000.0,
        late_reverb_decay_time: 2.5,
        late_reverb_hf_decay_ratio: 0.9,
        room_gain: 0.0,
        reflection_gain: -200.0,
        reverb_gain: 0.0,
        late_reverb_diffusion: 100.0,
        reflection_delay: 0.005,
        late_reverb_delay_time: 0.01,
        late_reverb_density: 100.0,
        dry_gain: 0.5,
        state: ParameterState::Initialized.raw(),
        ..Default::default()
    };
    for ch in 0..6 {
        rig.fill_tone(0, ch, ch);
    }
    rig.run(vec![Command::I3dl2Reverb(I3dl2ReverbCommand {
        inputs: six,
        outputs: six,
        parameter: p,
        state: EffectStateId(0),
        effect_enabled: true,
    })]);
    rig.report(&mut out, "i3dl2_6", 0, &six.map(|c| c as usize));
    check(&["i3dl2_2", "i3dl2_6"], &out);
}

#[test]
fn light_limiter_with_statistics_matches_cpp() {
    let mut out = String::new();
    let mut rig = Rig::new(SC);
    let mut p = LightLimiterParameterV2 {
        inputs: param_channels(&[0, 1]),
        outputs: param_channels(&[2, 3]),
        channel_count_max: 2,
        channel_count: 2,
        sample_rate: 48000,
        look_ahead_samples_min: 48,
        look_ahead_samples_max: 96,
        attack_coeff: 0.1,
        release_coeff: 0.01,
        threshold: 0.25,
        input_gain: 1.0,
        output_gain: 0.9,
        statistics_enabled: 1,
        statistics_reset_required: 1,
        ..Default::default()
    };
    for frame in 0..2 {
        p.state = state_for(frame, INIT_THEN_UPDATED);
        p.processing_mode = frame as u8;
        rig.fill_tone(frame, 0, 0);
        rig.fill_tone(frame, 1, 1);
        rig.run(vec![Command::LightLimiterVersion2(LightLimiterCommand {
            inputs: channel_indices(&[0, 1]),
            outputs: channel_indices(&[2, 3]),
            parameter: p,
            state: EffectStateId(0),
            result_state: Some(EffectResultId(0)),
            effect_enabled: true,
        })]);
        rig.report(&mut out, "limiter2", frame, &[2, 3]);
        let stats: LimiterStatistics = bytemuck::pod_read_unaligned(
            &rig.state.effect_results[0].state[..size_of::<LimiterStatistics>()],
        );
        let bits = [
            stats.channel_max_sample[0],
            stats.channel_max_sample[1],
            stats.channel_compression_gain_min[0],
            stats.channel_compression_gain_min[1],
        ]
        .map(|value| format!("0x{:08x}", value.to_bits()));
        writeln!(out, "limiter2 frame={frame} stats=[{}]", bits.join(", ")).unwrap();
    }
    check(&["limiter2"], &out);
}

#[test]
fn compressor_matches_cpp() {
    let mut out = String::new();
    let mut rig = Rig::new(SC);
    let mut p = CompressorParameterV2 {
        inputs: param_channels(&[0, 1]),
        outputs: param_channels(&[2, 3]),
        channel_count_max: 2,
        channel_count: 2,
        sample_rate: 48000,
        threshold: -20.0,
        compressor_ratio: 4.0,
        unk_24: 0.1,
        unk_28: 0.05,
        unk_2c: 0.2,
        out_gain: 2.0,
        makeup_gain_enabled: 1,
        ..Default::default()
    };
    for frame in 0..2 {
        p.state = state_for(frame, INIT_THEN_UPDATED);
        rig.fill_tone(frame, 0, 0);
        rig.fill_tone(frame, 1, 1);
        rig.run(vec![Command::Compressor(CompressorCommand {
            inputs: channel_indices(&[0, 1]),
            outputs: channel_indices(&[2, 3]),
            parameter: p,
            state: EffectStateId(0),
            effect_enabled: true,
        })]);
        rig.report(&mut out, "compressor2", frame, &[2, 3]);
    }
    check(&["compressor2"], &out);
}

#[test]
fn biquad_filters_match_cpp() {
    let mut out = String::new();
    let mut rig = Rig::new(SC);
    let fixed = abi::VoiceBiquadParameter {
        enabled: 1,
        b: [6000, 4000, 2000],
        a: [9000, -3000],
        ..Default::default()
    };
    let float = abi::VoiceBiquadParameterFloat {
        enabled: 1,
        numerator: [0.3, 0.25, 0.1],
        denominator: [0.6, -0.2],
        ..Default::default()
    };
    let filter = |output, state, use_float_processing, use_float_coefficients| {
        Command::BiquadFilter(BiquadFilterCommand {
            input: 0,
            output,
            biquad: fixed,
            biquad_float: float,
            state,
            needs_init: false,
            use_float_processing,
            use_float_coefficients,
        })
    };
    let voice = |voice, filter| BiquadStateRef::Voice {
        voice: VoiceStateId(voice),
        filter,
    };
    for frame in 0..2 {
        rig.fill_tone(frame, 0, 0);
        rig.run(vec![
            filter(1, voice(0, 0), false, false),
            filter(2, voice(0, 1), true, false),
            filter(3, voice(1, 0), true, true),
        ]);
        rig.report(&mut out, "biquad_int", frame, &[1]);
        rig.report(&mut out, "biquad_float", frame, &[2]);
        rig.report(&mut out, "biquad_float2", frame, &[3]);
    }
    check(&["biquad_int", "biquad_float", "biquad_float2"], &out);
}

#[test]
fn upsampler_matches_cpp() {
    let mut out = String::new();
    let mut shared = None;
    for (k, source) in [160usize, 160, 80, 40].into_iter().enumerate() {
        let mut rig = Rig::new(source);
        let upsampler = match (k, shared.take()) {
            (1, Some(upsampler)) => upsampler,
            _ => UpsamplerInfo {
                samples: vec![0; SC],
                sample_count: SC as u32,
                input_count: 1,
                enabled: true,
                ..Default::default()
            },
        };
        rig.state.upsamplers[0] = upsampler;
        for (i, sample) in rig.buffer(0).iter_mut().enumerate() {
            *sample = tone(k * 160 + i, 0);
        }
        rig.run(vec![Command::Upsample(UpsampleCommand {
            buffer_count: 1,
            source_sample_count: source as u32,
            source_sample_rate: source as u32 * 200,
            upsampler: UpsamplerId(0),
            ..Default::default()
        })]);
        let samples = &rig.state.upsamplers[0].samples;
        let hash = fnv(samples, FNV_OFFSET);
        let first = &samples[..8];
        writeln!(
            out,
            "upsample{source} k={k} hash=0x{hash:016x} first={first:?}"
        )
        .unwrap();
        if k == 0 {
            shared = Some(rig.state.upsamplers[0].clone());
        }
    }
    check(&["upsample160", "upsample80", "upsample40"], &out);
}

#[test]
fn mix_helpers_match_cpp() {
    let mut out = String::new();
    let mut rig = Rig::new(SC);
    rig.fill_tone(0, 0, 0);
    rig.fill_tone(0, 1, 1);
    let mix = |precision, volume| {
        Command::Mix(MixCommand {
            precision,
            input_index: 0,
            output_index: 1,
            volume,
        })
    };
    rig.run(vec![mix(15, 0.37)]);
    rig.report(&mut out, "mix15", 0, &[1]);
    rig.fill_tone(0, 1, 1);
    rig.run(vec![mix(23, -1.21)]);
    rig.report(&mut out, "mix23", 0, &[1]);

    rig.fill_tone(0, 1, 1);
    let last = mix::apply_mix_ramp::<15>(&mut rig.mix, SC, 0, 0.2, 0.003, SC);
    rig.report(&mut out, "mixramp15", 0, &[1]);
    writeln!(out, "mixramp15 last={last}").unwrap();
    rig.fill_tone(0, 1, 1);
    let last = mix::apply_mix_ramp::<23>(&mut rig.mix, SC, 0, 0.9, -0.001, SC);
    rig.report(&mut out, "mixramp23", 0, &[1]);
    writeln!(out, "mixramp23 last={last}").unwrap();

    rig.run(vec![Command::Volume(VolumeCommand {
        precision: 15,
        input_index: 0,
        output_index: 1,
        volume: 0.61,
    })]);
    rig.report(&mut out, "gain15", 0, &[1]);
    mix::apply_linear_envelope::<23>(&mut rig.mix, SC, 0, 0.1, 0.0025, SC);
    rig.report(&mut out, "envelope23", 0, &[1]);

    let decay = Q15::from_f32(0.9621887);
    rig.buffer(1).fill(5);
    rig.state.depop[1] = -12345;
    rig.run(vec![Command::DepopForMixBuffers(
        DepopForMixBuffersCommand {
            input: 1,
            count: 1,
            decay,
        },
    )]);
    rig.report(&mut out, "depop", 0, &[1]);
    writeln!(
        out,
        "depop last={} decay_raw={}",
        rig.state.depop[1],
        decay.raw()
    )
    .unwrap();
    check(
        &[
            "mix15",
            "mix23",
            "mixramp15",
            "mixramp23",
            "gain15",
            "envelope23",
            "depop",
        ],
        &out,
    );
}

struct DecodeCase {
    name: &'static str,
    wave_buffers: [WaveBuffer; MAX_WAVE_BUFFERS],
    valid: usize,
    version1: bool,
    format: u8,
    channel: i8,
    channels: i8,
    quality: SrcQuality,
    pitch: f32,
    rate: u32,
    flags: u16,
    frames: usize,
    data: u64,
    data_size: u64,
}

#[allow(clippy::too_many_arguments)]
fn wave_buffer(
    buffer: u64,
    context: u64,
    buffer_size: u64,
    context_size: u64,
    offsets: [u32; 4],
    loop_count: i32,
    looping: bool,
    stream_ended: bool,
) -> WaveBuffer {
    WaveBuffer {
        buffer,
        context,
        buffer_size,
        context_size,
        start_offset: offsets[0],
        end_offset: offsets[1],
        loop_start_offset: offsets[2],
        loop_end_offset: offsets[3],
        loop_count,
        looping,
        stream_ended,
    }
}

fn decode_memory() -> VecMemory {
    let mut memory = VecMemory::new(0, 0x10000);
    let put16 = |memory: &mut VecMemory, address: u64, value: i16| {
        memory.write(address, &value.to_le_bytes());
    };
    for i in 0..300 {
        put16(&mut memory, 0x1000 + 2 * i as u64, (tone(i, 0) / 40) as i16);
    }
    for i in 0..400 {
        put16(
            &mut memory,
            0x2000 + 2 * i as u64,
            (tone(i + 1000, 1) / 40) as i16,
        );
    }
    for i in 0..1000 {
        let value = tone(i, i & 1) as f32 / 40000.0;
        memory.write(0x3000 + 4 * i as u64, &value.to_le_bytes());
    }
    let coefficients: [i16; 16] = [
        2048, 0, 4096, -2048, 3000, -1000, 1500, 400, 0, 0, 1024, 1024, -1024, 2048, 2500, -1300,
    ];
    for (i, &c) in coefficients.iter().enumerate() {
        put16(&mut memory, 0x5000 + 2 * i as u64, c);
    }
    for frame in 0..40usize {
        let base = 0x4000 + frame as u64 * 8;
        memory.write(base, &[(((frame % 8) << 4) | ((frame * 3) % 12)) as u8]);
        for b in 1..8 {
            memory.write(base + b as u64, &[((frame * 8 + b) * 73 + 11) as u8]);
        }
    }
    put16(&mut memory, 0x5100, 0x23);
    put16(&mut memory, 0x5102, 100);
    put16(&mut memory, 0x5104, -50);
    memory
}

fn run_decode(case: &DecodeCase, out: &mut String) {
    let mut rig = Rig::new(SC);
    rig.memory = decode_memory();
    for valid in &mut rig.state.voices[0].wave_buffer_valid[..case.valid] {
        *valid = true;
    }
    let command = DataSourceCommand {
        src_quality: case.quality,
        output_index: 0,
        flags: case.flags,
        sample_rate: case.rate,
        pitch: case.pitch,
        channel_index: case.channel,
        channel_count: case.channels,
        wave_buffers: case.wave_buffers,
        voice_state: VoiceStateId(0),
        data_address: case.data,
        data_size: case.data_size,
    };
    let command = match (case.format, case.version1) {
        (0, false) => Command::DataSourcePcmInt16Version2(command),
        (0, true) => Command::DataSourcePcmInt16Version1(command),
        (1, false) => Command::DataSourcePcmFloatVersion2(command),
        (1, true) => Command::DataSourcePcmFloatVersion1(command),
        (_, false) => Command::DataSourceAdpcmVersion2(command),
        (_, true) => Command::DataSourceAdpcmVersion1(command),
    };
    for frame in 0..case.frames {
        rig.buffer(0).fill(0x1234);
        rig.run(vec![command.clone()]);
        let v = rig.state.voices[0];
        let samples = &rig.mix[..SC];
        let valid = v.wave_buffer_valid.map(|valid| valid as u8);
        let ctx = v.adpcm_context;
        writeln!(
            out,
            "{} frame={frame} hash=0x{:016x} first={:?} played={} offset={} index={} consumed={} \
             fraction={} loop_count={} valid=[{},{},{},{}] ctx=[{},{},{}] history={:?}",
            case.name,
            fnv(samples, FNV_OFFSET),
            &samples[..6],
            v.played_sample_count,
            v.offset,
            v.wave_buffer_index,
            v.wave_buffers_consumed,
            v.fraction.raw(),
            v.loop_count,
            valid[0],
            valid[1],
            valid[2],
            valid[3],
            ctx.header,
            ctx.yn0,
            ctx.yn1,
            v.sample_history,
        )
        .unwrap();
    }
}

#[test]
fn decode_matches_cpp() {
    let none = WaveBuffer::default();
    let pcm0 = wave_buffer(0x1000, 0, 600, 0, [0, 300, 0, 0], 0, false, false);
    let pcm1 = wave_buffer(0x2000, 0, 800, 0, [0, 400, 0, 0], 0, false, true);
    let base = DecodeCase {
        name: "",
        wave_buffers: [pcm0, pcm1, none, none],
        valid: 2,
        version1: false,
        format: 0,
        channel: 0,
        channels: 1,
        quality: SrcQuality::Medium,
        pitch: 1.0,
        rate: 32000,
        flags: 0,
        frames: 3,
        data: 0,
        data_size: 0,
    };
    let loop1 = wave_buffer(0x2000, 0, 800, 0, [10, 110, 20, 60], 1, true, false);
    let loop_v1 = wave_buffer(0x1000, 0, 600, 0, [0, 100, 0, 0], 0, true, false);
    let float = wave_buffer(0x3000, 0, 4000, 0, [0, 500, 0, 0], 0, false, false);
    let adpcm = wave_buffer(0x4000, 0x5100, 320, 6, [0, 200, 30, 180], 2, true, false);
    let adpcm_mid = wave_buffer(0x4000, 0, 320, 0, [5, 250, 0, 0], 0, false, false);
    let cases = [
        DecodeCase {
            name: "pcm16_medium",
            ..base
        },
        DecodeCase {
            name: "pcm16_high",
            quality: SrcQuality::High,
            ..base
        },
        DecodeCase {
            name: "pcm16_low",
            quality: SrcQuality::Low,
            ..base
        },
        DecodeCase {
            name: "pcm16_loop_skip",
            wave_buffers: [pcm0, loop1, none, none],
            rate: 48000,
            flags: DataSourceCommand::FLAG_SKIP_PITCH_AND_SRC,
            frames: 2,
            ..base
        },
        DecodeCase {
            name: "pcm16_v1_loop_reset",
            wave_buffers: [loop_v1, none, none, none],
            valid: 1,
            version1: true,
            quality: SrcQuality::High,
            pitch: 0.8,
            rate: 44100,
            flags: DataSourceCommand::FLAG_RESET_PLAYED_AT_LOOP,
            ..base
        },
        DecodeCase {
            name: "float_stereo_ch1",
            wave_buffers: [float, none, none, none],
            valid: 1,
            format: 1,
            channel: 1,
            channels: 2,
            quality: SrcQuality::Low,
            rate: 48000,
            ..base
        },
        DecodeCase {
            name: "adpcm_loop_high",
            wave_buffers: [adpcm, none, none, none],
            valid: 1,
            format: 2,
            quality: SrcQuality::High,
            pitch: 1.25,
            rate: 48000,
            frames: 4,
            data: 0x5000,
            data_size: 32,
            ..base
        },
        DecodeCase {
            name: "adpcm_skip",
            wave_buffers: [adpcm_mid, none, none, none],
            valid: 1,
            format: 2,
            rate: 48000,
            flags: DataSourceCommand::FLAG_SKIP_PITCH_AND_SRC,
            frames: 2,
            data: 0x5000,
            data_size: 32,
            ..base
        },
    ];
    let mut out = String::new();
    let mut names = Vec::new();
    for case in &cases {
        run_decode(case, &mut out);
        names.push(case.name);
    }
    check(&names, &out);
}
