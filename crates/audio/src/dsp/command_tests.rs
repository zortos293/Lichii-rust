//! Hand-checked tests for the commands whose results follow directly from their inputs: guest
//! ring buffers, sinks, routing, ramps and bookkeeping, plus an end-to-end command list. The
//! arithmetic these commands share (mix ramps, linear envelopes, biquads, decode) is checked
//! bit-exactly against the C++ renderer in [\`super::tests\`].

use std::cell::Cell;

use super::abi::{AuxBufferInfo, AuxInfoDsp, DelayParameterV1, ParameterState};
use super::fixed::Fx;
use super::tests::{Rig, SC, tone};
use super::*;

fn read_i32s(memory: &VecMemory, address: u64, count: usize) -> Vec<i32> {
    let mut bytes = vec![0u8; count * 4];
    memory.read(address, &mut bytes);
    bytemuck::cast_slice(&bytes).to_vec()
}

fn write_i32s(memory: &mut VecMemory, address: u64, values: &[i32]) {
    memory.write(address, bytemuck::cast_slice(values));
}

fn read_pod<T: bytemuck::Pod>(memory: &VecMemory, address: u64) -> T {
    let mut value = T::zeroed();
    memory.read(address, bytemuck::bytes_of_mut(&mut value));
    value
}

fn buffer(rig: &Rig, index: usize) -> Vec<i32> {
    rig.mix[index * rig.sample_count..(index + 1) * rig.sample_count].to_vec()
}

const SEND_INFO: u64 = 0x100;
const RETURN_INFO: u64 = 0x200;
const SEND_BUFFER: u64 = 0x1000;
const RETURN_BUFFER: u64 = 0x3000;
const RING: u32 = 300;

#[test]
fn aux_sends_and_returns_through_wrapping_rings() {
    let mut rig = Rig::new(SC);
    rig.fill_tone(0, 0, 0);
    let returned: Vec<i32> = (0..RING as i32).map(|i| i * 3 + 1).collect();
    write_i32s(&mut rig.memory, RETURN_BUFFER, &returned);
    let send = AuxInfoDsp {
        write_offset: 200,
        lost_sample_count: 9,
        total_sample_count: 77,
        ..AuxInfoDsp::default()
    };
    let ret = AuxInfoDsp {
        read_offset: 100,
        ..AuxInfoDsp::default()
    };
    rig.memory.write(SEND_INFO, bytemuck::bytes_of(&send));
    rig.memory.write(RETURN_INFO, bytemuck::bytes_of(&ret));
    let aux = |effect_enabled| {
        Command::Aux(AuxCommand {
            input: 0,
            output: 1,
            send_buffer_info: SEND_INFO,
            return_buffer_info: RETURN_INFO,
            send_buffer: SEND_BUFFER,
            return_buffer: RETURN_BUFFER,
            count_max: RING,
            write_offset: 10,
            update_count: SC as u32,
            effect_enabled,
        })
    };
    rig.run(vec![aux(true)]);

    let input = buffer(&rig, 0);
    let ring = read_i32s(&rig.memory, SEND_BUFFER, RING as usize);
    for (i, &sample) in input.iter().enumerate() {
        assert_eq!(ring[(210 + i) % RING as usize], sample, "send sample {i}");
    }
    let output = buffer(&rig, 1);
    for (i, &sample) in output.iter().enumerate() {
        assert_eq!(
            sample,
            returned[(110 + i) % RING as usize],
            "return sample {i}"
        );
    }
    let send: AuxInfoDsp = read_pod(&rig.memory, SEND_INFO);
    let ret: AuxInfoDsp = read_pod(&rig.memory, RETURN_INFO);
    assert_eq!((send.write_offset, send.total_sample_count), (140, 77));
    assert_eq!(ret.read_offset, 40);

    rig.run(vec![aux(false)]);
    assert_eq!(buffer(&rig, 1), input);
    let send: AuxInfoDsp = read_pod(&rig.memory, SEND_INFO);
    assert_eq!(
        (
            send.read_offset,
            send.write_offset,
            send.lost_sample_count,
            send.total_sample_count
        ),
        (0, 0, 9, 0)
    );
}

#[test]
fn aux_zeroes_the_output_when_the_return_offset_is_past_the_ring() {
    let mut rig = Rig::new(SC);
    rig.fill_tone(0, 0, 0);
    rig.buffer(1).fill(7);
    let ret = AuxInfoDsp {
        read_offset: RING,
        ..AuxInfoDsp::default()
    };
    rig.memory.write(RETURN_INFO, bytemuck::bytes_of(&ret));
    rig.run(vec![Command::Aux(AuxCommand {
        input: 0,
        output: 1,
        send_buffer_info: SEND_INFO,
        return_buffer_info: RETURN_INFO,
        send_buffer: SEND_BUFFER,
        return_buffer: RETURN_BUFFER,
        count_max: RING,
        write_offset: 1,
        update_count: SC as u32,
        effect_enabled: true,
    })]);
    assert!(buffer(&rig, 1).iter().all(|&s| s == 0));
}

#[test]
fn capture_tracks_lost_and_total_samples() {
    let mut rig = Rig::new(SC);
    rig.fill_tone(0, 2, 1);
    let info = AuxBufferInfo {
        cpu_info: AuxInfoDsp {
            read_offset: 5,
            lost_sample_count: 2,
            total_sample_count: 1000,
            ..AuxInfoDsp::default()
        },
        dsp_info: AuxInfoDsp {
            write_offset: 250,
            lost_sample_count: 5,
            total_sample_count: 1300,
            ..AuxInfoDsp::default()
        },
    };
    rig.memory.write(SEND_INFO, bytemuck::bytes_of(&info));
    let capture = |effect_enabled| {
        Command::Capture(CaptureCommand {
            input: 2,
            output: 2,
            send_buffer_info: SEND_INFO,
            send_buffer: SEND_BUFFER,
            count_max: RING,
            write_offset: 0,
            update_count: SC as u32,
            effect_enabled,
        })
    };
    rig.run(vec![capture(true)]);

    let ring = read_i32s(&rig.memory, SEND_BUFFER, RING as usize);
    for (i, &sample) in buffer(&rig, 2).iter().enumerate() {
        assert_eq!(
            ring[(250 + i) % RING as usize],
            sample,
            "capture sample {i}"
        );
    }
    let info: AuxBufferInfo = read_pod(&rig.memory, SEND_INFO);
    assert_eq!(info.dsp_info.lost_sample_count, 245);
    assert_eq!(info.dsp_info.write_offset, 190);
    assert_eq!(info.dsp_info.total_sample_count, 1540);

    rig.run(vec![capture(false)]);
    let info: AuxBufferInfo = read_pod(&rig.memory, SEND_INFO);
    assert_eq!(
        (
            info.cpu_info.read_offset,
            info.cpu_info.write_offset,
            info.cpu_info.total_sample_count,
            info.cpu_info.lost_sample_count
        ),
        (0, 0, 0, 2)
    );
    assert_eq!(info.dsp_info.write_offset, 190);
}

#[test]
fn circular_buffer_sink_saturates_and_wraps_per_channel() {
    let mut rig = Rig::new(SC);
    for (i, sample) in rig.buffer(3).iter_mut().enumerate() {
        *sample = (i as i32 - 120) * 400;
    }
    rig.fill_tone(0, 4, 1);
    let address = 0x8000u64;
    rig.run(vec![Command::CircularBufferSink(
        CircularBufferSinkCommand {
            input_count: 2,
            inputs: [3, 4, 0, 0, 0, 0],
            address,
            size: 960,
            pos: 480,
        },
    )]);
    let mut bytes = vec![0u8; 960];
    rig.memory.read(address, &mut bytes);
    let ring: &[i16] = bytemuck::cast_slice(&bytes);
    let saturate = |s: i32| s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let first: Vec<i16> = buffer(&rig, 3).into_iter().map(saturate).collect();
    let second: Vec<i16> = buffer(&rig, 4).into_iter().map(saturate).collect();
    assert_eq!(&ring[240..], &first[..]);
    assert_eq!(&ring[..240], &second[..]);
    assert_eq!((first[0], first[239]), (i16::MIN, i16::MAX));
}

#[test]
fn device_sink_interleaves_mix_buffers_or_upsampler_samples() {
    let mut rig = Rig::new(SC);
    rig.fill_tone(0, 3, 0);
    for (i, sample) in rig.buffer(1).iter_mut().enumerate() {
        *sample = i as i32 * 300 - 36000;
    }
    let report = rig.run(vec![Command::DeviceSink(DeviceSinkCommand {
        session_id: 4,
        source: SinkSource::MixBuffers,
        input_count: 2,
        inputs: [3, 1, 0, 0, 0, 0],
    })]);
    assert_eq!(
        report.device_sink,
        Some(DeviceSinkFrame {
            session_id: 4,
            channel_count: 2,
            frame_count: SC as u32
        })
    );
    let saturate = |s: i32| s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let (left, right) = (buffer(&rig, 3), buffer(&rig, 1));
    for i in 0..SC {
        assert_eq!(rig.device[2 * i], saturate(left[i]), "left {i}");
        assert_eq!(rig.device[2 * i + 1], saturate(right[i]), "right {i}");
    }

    let mut rig = Rig::new(160);
    rig.state.upsamplers[0].samples = (0..SC as i32 * 2).map(|i| i - 100).collect();
    rig.run(vec![Command::DeviceSink(DeviceSinkCommand {
        session_id: 0,
        source: SinkSource::Upsampler(UpsamplerId(0)),
        input_count: 1,
        inputs: [1, 0, 0, 0, 0, 0],
    })]);
    let expected: Vec<i16> = (0..SC as i16).map(|i| i + SC as i16 - 100).collect();
    assert_eq!(&rig.device[..SC], &expected[..]);
}

#[test]
fn downmix_folds_six_channels_into_front_left_and_right() {
    let mut rig = Rig::new(SC);
    for (buffer, value) in [1000, 2000, 400, 800, -600, 300].into_iter().enumerate() {
        rig.buffer(buffer).fill(value);
    }
    for buffer in 6..12 {
        rig.buffer(buffer).fill(55);
    }
    let coefficient = |raw| Fx::<16>::from_raw(raw);
    rig.run(vec![Command::DownMix6chTo2ch(DownMix6chTo2chCommand {
        inputs: [0, 1, 2, 3, 4, 5],
        outputs: [6, 7, 8, 9, 10, 11],
        down_mix_coeff: [
            coefficient(0x10000),
            coefficient(0x8000),
            coefficient(0x4000),
            coefficient(0x8000),
        ],
    })]);
    assert!(buffer(&rig, 6).iter().all(|&s| s == 1000 + 200 + 200 - 300));
    assert!(buffer(&rig, 7).iter().all(|&s| s == 2000 + 200 + 200 + 150));
    for index in 8..12 {
        assert!(buffer(&rig, index).iter().all(|&s| s == 0));
    }
}

#[test]
fn multi_tap_biquad_runs_its_taps_like_chained_float_biquads() {
    let fixed = [
        abi::VoiceBiquadParameter {
            enabled: 1,
            b: [6000, 4000, 2000],
            a: [9000, -3000],
            ..Default::default()
        },
        abi::VoiceBiquadParameter {
            enabled: 1,
            b: [8000, -2000, 1000],
            a: [-4000, 1500],
            ..Default::default()
        },
    ];
    let float = [
        abi::VoiceBiquadParameterFloat {
            enabled: 1,
            numerator: [0.3, 0.25, 0.1],
            denominator: [0.6, -0.2],
            ..Default::default()
        },
        abi::VoiceBiquadParameterFloat {
            enabled: 1,
            numerator: [0.5, -0.1, 0.05],
            denominator: [-0.3, 0.1],
            ..Default::default()
        },
    ];
    let state = |voice, filter| BiquadStateRef::Voice {
        voice: VoiceStateId(voice),
        filter,
    };
    for use_float_coefficients in [false, true] {
        let mut multi = Rig::new(SC);
        let mut chained = Rig::new(SC);
        for frame in 0..2 {
            multi.fill_tone(frame, 0, 0);
            chained.fill_tone(frame, 0, 0);
            multi.run(vec![Command::MultiTapBiquadFilter(
                MultiTapBiquadFilterCommand {
                    input: 0,
                    output: 0,
                    biquads: fixed,
                    biquads_float: float,
                    states: [state(0, 0), state(0, 1)],
                    needs_init: [frame == 0; 2],
                    filter_tap_count: 2,
                    use_float_coefficients,
                },
            )]);
            chained.run(
                (0..2)
                    .map(|tap| {
                        Command::BiquadFilter(BiquadFilterCommand {
                            input: 0,
                            output: 0,
                            biquad: fixed[tap],
                            biquad_float: float[tap],
                            state: state(1, tap as u8),
                            needs_init: frame == 0,
                            use_float_processing: true,
                            use_float_coefficients,
                        })
                    })
                    .collect(),
            );
            assert_eq!(buffer(&multi, 0), buffer(&chained, 0));
            assert_ne!(buffer(&multi, 0)[..8], [0; 8]);
        }
    }
}

#[test]
fn mix_ramps_record_their_last_sample_and_feed_depop() {
    let mut rig = Rig::new(SC);
    rig.fill_tone(0, 0, 0);
    rig.fill_tone(0, 1, 1);
    let mut expected = rig.mix.clone();
    let ramp = (0.9f32 - 0.2) / SC as f32;
    let last = mix::apply_mix_ramp::<15>(&mut expected, SC, 0, 0.2, ramp, SC);
    rig.run(vec![
        Command::MixRamp(MixRampCommand {
            precision: 15,
            input_index: 0,
            output_index: 1,
            prev_volume: 0.2,
            volume: 0.9,
            previous_sample: PreviousSampleRef {
                voice: VoiceStateId(1),
                index: 3,
            },
        }),
        Command::MixRamp(MixRampCommand {
            precision: 23,
            input_index: 0,
            output_index: 2,
            prev_volume: 0.0,
            volume: 0.0,
            previous_sample: PreviousSampleRef {
                voice: VoiceStateId(1),
                index: 4,
            },
        }),
    ]);
    assert_eq!(rig.mix, expected);
    assert_eq!(rig.state.voices[1].previous_samples[3], last);
    assert_eq!(rig.state.voices[1].previous_samples[4], 0);

    let mut grouped = MixRampGroupedCommand {
        precision: 23,
        buffer_count: 2,
        previous_samples: VoiceStateId(0),
        ..Default::default()
    };
    grouped.inputs[..2].copy_from_slice(&[0, 0]);
    grouped.outputs[..2].copy_from_slice(&[5, 6]);
    grouped.prev_volumes[..2].copy_from_slice(&[1.0, 0.0]);
    grouped.volumes[..2].copy_from_slice(&[0.5, 0.0]);
    rig.state.voices[0].previous_samples[1] = 99;
    let mut expected = rig.mix.clone();
    let last = mix::apply_mix_ramp::<23>(&mut expected, 5 * SC, 0, 1.0, -0.5 / SC as f32, SC);
    rig.run(vec![Command::MixRampGrouped(grouped)]);
    assert_eq!(rig.mix, expected);
    assert_eq!(rig.state.voices[0].previous_samples[..2], [last, 0]);

    rig.state.depop[7] = 10;
    rig.run(vec![Command::DepopPrepare(DepopPrepareCommand {
        inputs: [
            7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        previous_samples: VoiceStateId(0),
        buffer_count: 2,
    })]);
    assert_eq!(rig.state.depop[7], 10 + last);
    assert_eq!(rig.state.depop[8], 0);
    assert_eq!(rig.state.voices[0].previous_samples[..2], [0, 0]);
}

#[test]
fn volume_ramp_clear_and_copy_route_buffers() {
    let mut rig = Rig::new(SC);
    rig.fill_tone(0, 0, 0);
    let mut expected = rig.mix.clone();
    let ramp = (0.75f32 - 0.25) / SC as f32;
    mix::apply_linear_envelope::<23>(&mut expected, SC, 0, 0.25, ramp, SC);
    let volume_ramp = |output, prev_volume, volume| {
        Command::VolumeRamp(VolumeRampCommand {
            precision: 23,
            input_index: 0,
            output_index: output,
            prev_volume,
            volume,
        })
    };
    rig.run(vec![volume_ramp(1, 0.25, 0.75)]);
    assert_eq!(rig.mix, expected);

    rig.buffer(2).fill(3);
    rig.run(vec![volume_ramp(2, 0.0, 0.0), volume_ramp(3, 1.0, 1.0)]);
    assert!(buffer(&rig, 2).iter().all(|&s| s == 0));
    assert_eq!(buffer(&rig, 3), buffer(&rig, 0));

    rig.run(vec![Command::CopyMixBuffer(CopyMixBufferCommand {
        input_index: 1,
        output_index: 4,
    })]);
    assert_eq!(buffer(&rig, 4), buffer(&rig, 1));
    rig.run(vec![Command::ClearMixBuffer]);
    assert!(rig.mix.iter().all(|&s| s == 0));
}

struct SteppingClock(Cell<u64>);

impl DspClock for SteppingClock {
    fn now_us(&self) -> u64 {
        let now = self.0.get() + 10;
        self.0.set(now);
        now
    }
}

#[test]
fn performance_commands_stamp_times_relative_to_the_list_start() {
    let mut state = DspState::new(0, 0, 0, 0);
    state.performance = vec![0; 0x40];
    let entry = PerformanceEntryAddresses {
        translated_address: 0x10,
        entry_start_time_offset: 4,
        header_entry_count_offset: 0x20,
        entry_processed_time_offset: 8,
    };
    let performance = |state| {
        CommandEntry::new(
            0,
            Command::Performance(PerformanceCommand {
                state,
                entry_address: entry,
            }),
        )
    };
    let mut list = CommandList {
        sample_count: SC as u32,
        sample_rate: TARGET_SAMPLE_RATE,
        buffer_count: 0,
        commands: vec![
            performance(PerformanceState::Start),
            CommandEntry::new(0, Command::ClearMixBuffer),
            performance(PerformanceState::Stop),
        ],
    };
    list.commands[1].enabled = false;
    let mut processor = CommandListProcessor::new();
    let clock = SteppingClock(Cell::new(1000));
    let report = processor.process(
        &list,
        DspFrame {
            mix_buffers: &mut [],
            state: &mut state,
            memory: &mut VecMemory::default(),
            clock: &clock,
            device_output: &mut [],
        },
    );
    let word = |offset: usize| {
        u32::from_le_bytes(state.performance[offset..offset + 4].try_into().unwrap())
    };
    let (start, processed) = (word(0x14), word(0x18));
    assert!(
        start > 0 && processed > start,
        "start {start} processed {processed}"
    );
    assert_eq!(word(0x30), 1);
    assert_eq!(report.processed_command_count, 3);
    assert_eq!(report.remaining_command_count, 0);
    assert_eq!(processor.command_times().len(), 3);
    assert_eq!(processor.command_times()[1], 0);
}

#[test]
fn end_to_end_list_decodes_mixes_delays_and_sinks() {
    let mut rig = Rig::new(SC);
    for i in 0..480u64 {
        let left = (tone(i as usize, 0) / 40) as i16;
        let right = (tone(i as usize, 1) / 50) as i16;
        rig.memory.write(0x1000 + 2 * i, &left.to_le_bytes());
        rig.memory.write(0x2000 + 2 * i, &right.to_le_bytes());
    }
    let voice = |index: u32, address: u64, output_index| {
        let mut wave_buffers = [WaveBuffer::default(); MAX_WAVE_BUFFERS];
        wave_buffers[0] = WaveBuffer {
            buffer: address,
            buffer_size: 960,
            end_offset: 480,
            ..WaveBuffer::default()
        };
        Command::DataSourcePcmInt16Version2(DataSourceCommand {
            src_quality: SrcQuality::High,
            output_index,
            sample_rate: 32000,
            pitch: 1.0,
            channel_index: 0,
            channel_count: 1,
            wave_buffers,
            voice_state: VoiceStateId(index),
            ..DataSourceCommand::default()
        })
    };
    let mix_into = |input_index, output_index, volume| {
        Command::Mix(MixCommand {
            precision: 15,
            input_index,
            output_index,
            volume,
        })
    };
    let delay = DelayParameterV1 {
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
        state: ParameterState::Initialized.raw(),
        ..Default::default()
    };
    let commands = vec![
        Command::ClearMixBuffer,
        voice(0, 0x1000, 0),
        voice(1, 0x2000, 1),
        mix_into(0, 2, 0.8),
        mix_into(1, 2, 0.2),
        mix_into(0, 3, 0.2),
        mix_into(1, 3, 0.8),
        Command::Delay(DelayCommand {
            inputs: [2, 3, 0, 0, 0, 0],
            outputs: [4, 5, 0, 0, 0, 0],
            parameter: delay,
            state: EffectStateId(0),
            effect_enabled: true,
        }),
        Command::DeviceSink(DeviceSinkCommand {
            session_id: 0,
            source: SinkSource::MixBuffers,
            input_count: 2,
            inputs: [4, 5, 0, 0, 0, 0],
        }),
    ];
    for valid in [0, 1] {
        rig.state.voices[valid].wave_buffer_valid[0] = true;
    }
    let report = rig.run(commands.clone());
    assert_eq!(report.processed_command_count, commands.len() as u32);
    assert!(report.device_sink.is_some());
    for voice in &rig.state.voices {
        assert_eq!(voice.played_sample_count, 159);
    }

    let mut staged = Rig::new(SC);
    staged.memory = rig.memory.clone();
    for valid in [0, 1] {
        staged.state.voices[valid].wave_buffer_valid[0] = true;
    }
    let mut mixed = None;
    for command in commands.iter().take(7) {
        staged.run(vec![command.clone()]);
        mixed = Some((buffer(&staged, 2), buffer(&staged, 3)));
    }
    let (left, right) = mixed.unwrap();
    assert_ne!(left, right);
    staged.run(vec![commands[7].clone(), commands[8].clone()]);
    assert_eq!(staged.device, rig.device);
    let saturate = |s: i32| s.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    let (out_left, out_right) = (buffer(&rig, 4), buffer(&rig, 5));
    for i in 0..SC {
        assert_eq!(rig.device[2 * i], saturate(out_left[i]));
        assert_eq!(rig.device[2 * i + 1], saturate(out_right[i]));
    }
    assert!(out_left.iter().any(|&s| s != 0));
}
