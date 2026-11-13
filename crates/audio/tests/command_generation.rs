//! Replays command list fixtures produced by the C++ reference.
//!
//! `golden/commands.cpp` opens a renderer per case, applies three synthetic updates and runs a
//! copy of `System::GenerateCommand` after each, recording every command (id, node id, enabled
//! flag, estimated processing time and the fields the DSP reads), the voices the drop pass
//! dropped and a digest of the renderer state generation leaves behind. These tests rebuild the
//! same words from `RendererSystem::generate_command` and compare them command by command.

mod support;

use lichii_audio::dsp::abi::{VoiceBiquadParameter, VoiceBiquadParameterFloat};
use lichii_audio::dsp::{
    BiquadStateRef, Command, CommandEntry, CommandListProcessor, DataSourceCommand, HostClock,
    SinkSource, VecMemory, WaveBuffer,
};
use lichii_audio::renderer::RendererSystem;
use lichii_audio::renderer::parameter::{AudioRendererParameterInternal, work_buffer_size};
use support::{Reader, digest, first_difference_words, reference_voice_flags};

#[derive(Default)]
struct Words(Vec<u32>);

impl Words {
    fn u(&mut self, value: impl Into<u32>) {
        self.0.push(value.into());
    }

    fn s(&mut self, value: impl Into<i32>) {
        self.0.push(value.into() as u32);
    }

    fn f(&mut self, value: f32) {
        self.0.push(value.to_bits());
    }

    fn q(&mut self, value: u64) {
        self.0.push(value as u32);
        self.0.push((value >> 32) as u32);
    }

    fn indices(&mut self, values: &[i16]) {
        values.iter().for_each(|&v| self.s(v));
    }

    fn bytes(&mut self, value: &impl bytemuck::Pod) {
        for word in bytemuck::bytes_of(value).as_chunks::<4>().0 {
            self.u(u32::from_le_bytes(*word));
        }
    }

    fn biquad(&mut self, b: &VoiceBiquadParameter) {
        self.u(b.enabled);
        b.b.iter().for_each(|&x| self.s(x));
        b.a.iter().for_each(|&x| self.s(x));
    }

    fn biquad_float(&mut self, b: &VoiceBiquadParameterFloat) {
        self.u(b.enabled);
        b.numerator.iter().for_each(|&x| self.f(x));
        b.denominator.iter().for_each(|&x| self.f(x));
    }

    fn biquad_state(&mut self, state: BiquadStateRef) {
        match state {
            BiquadStateRef::Voice { voice, filter } => {
                self.u(0u32);
                self.u(voice.0);
                self.u(filter);
            }
            BiquadStateRef::Effect { effect, channel } => {
                self.u(1u32);
                self.u(effect.0);
                self.u(channel);
            }
        }
    }

    fn wave_buffers(&mut self, wave_buffers: &[WaveBuffer]) {
        for w in wave_buffers {
            self.q(w.buffer);
            self.q(w.buffer_size);
            self.u(w.start_offset);
            self.u(w.end_offset);
            self.u(w.looping);
            self.u(w.stream_ended);
            self.q(w.context);
            self.q(w.context_size);
            self.u(w.loop_start_offset);
            self.u(w.loop_end_offset);
            self.s(w.loop_count);
        }
    }

    fn data_source(&mut self, c: &DataSourceCommand, adpcm: bool) {
        self.u(c.src_quality as u32);
        self.s(c.output_index);
        self.u(reference_voice_flags(c.flags));
        self.u(c.sample_rate);
        self.f(c.pitch);
        self.s(c.channel_index);
        self.s(c.channel_count);
        self.u(c.voice_state.0);
        self.wave_buffers(&c.wave_buffers);
        if adpcm {
            self.q(c.data_address);
            self.q(c.data_size);
        }
    }
}

/// Mirrors `Dump::Command` in `golden/commands.cpp`: the fields of one command.
fn fields(command: &Command) -> Vec<u32> {
    let mut w = Words::default();
    match command {
        Command::DataSourcePcmInt16Version1(c)
        | Command::DataSourcePcmInt16Version2(c)
        | Command::DataSourcePcmFloatVersion1(c)
        | Command::DataSourcePcmFloatVersion2(c) => w.data_source(c, false),
        Command::DataSourceAdpcmVersion1(c) | Command::DataSourceAdpcmVersion2(c) => {
            w.data_source(c, true)
        }
        Command::Volume(c) => {
            w.u(c.precision);
            w.s(c.input_index);
            w.s(c.output_index);
            w.f(c.volume);
        }
        Command::VolumeRamp(c) => {
            w.u(c.precision);
            w.s(c.input_index);
            w.s(c.output_index);
            w.f(c.prev_volume);
            w.f(c.volume);
        }
        Command::BiquadFilter(c) => {
            w.s(c.input);
            w.s(c.output);
            w.biquad(&c.biquad);
            w.biquad_float(&c.biquad_float);
            w.biquad_state(c.state);
            w.u(c.needs_init);
            w.u(c.use_float_processing);
            w.u(c.use_float_coefficients);
        }
        Command::MultiTapBiquadFilter(c) => {
            w.s(c.input);
            w.s(c.output);
            for i in 0..c.biquads.len() {
                w.biquad(&c.biquads[i]);
                w.biquad_float(&c.biquads_float[i]);
                w.biquad_state(c.states[i]);
                w.u(c.needs_init[i]);
            }
            w.u(c.filter_tap_count);
            w.u(c.use_float_coefficients);
        }
        Command::Mix(c) => {
            w.u(c.precision);
            w.s(c.input_index);
            w.s(c.output_index);
            w.f(c.volume);
        }
        Command::MixRamp(c) => {
            w.u(c.precision);
            w.s(c.input_index);
            w.s(c.output_index);
            w.f(c.prev_volume);
            w.f(c.volume);
            w.u(c.previous_sample.voice.0);
            w.u(c.previous_sample.index);
        }
        Command::MixRampGrouped(c) => {
            w.u(c.precision);
            w.u(c.buffer_count);
            w.u(c.previous_samples.0);
            for i in 0..c.buffer_count as usize {
                w.s(c.inputs[i]);
                w.s(c.outputs[i]);
                w.f(c.prev_volumes[i]);
                w.f(c.volumes[i]);
            }
        }
        Command::DepopPrepare(c) => {
            w.indices(&c.inputs);
            w.u(c.previous_samples.0);
            w.u(c.buffer_count);
        }
        Command::DepopForMixBuffers(c) => {
            w.u(c.input);
            w.u(c.count);
            w.q(c.decay.0 as u64);
        }
        Command::Delay(c) => {
            w.indices(&c.inputs);
            w.indices(&c.outputs);
            w.bytes(&c.parameter);
            w.u(c.state.0);
            w.u(c.effect_enabled);
        }
        Command::Reverb(c) => {
            w.indices(&c.inputs);
            w.indices(&c.outputs);
            w.bytes(&c.parameter);
            w.u(c.state.0);
            w.u(c.effect_enabled);
            w.u(c.long_size_pre_delay_supported);
        }
        Command::I3dl2Reverb(c) => {
            w.indices(&c.inputs);
            w.indices(&c.outputs);
            w.bytes(&c.parameter);
            w.u(c.state.0);
            w.u(c.effect_enabled);
        }
        Command::LightLimiterVersion1(c) | Command::LightLimiterVersion2(c) => {
            w.indices(&c.inputs);
            w.indices(&c.outputs);
            w.bytes(&c.parameter);
            w.u(c.state.0);
            w.u(c.effect_enabled);
            if matches!(command, Command::LightLimiterVersion2(_)) {
                w.u(c.result_state.map_or(u32::MAX, |id| id.0));
            }
        }
        Command::Compressor(c) => {
            w.indices(&c.inputs);
            w.indices(&c.outputs);
            w.bytes(&c.parameter);
            w.u(c.state.0);
        }
        Command::Upsample(c) => {
            w.u(c.buffer_count);
            w.u(c.unk_20);
            w.u(c.source_sample_count);
            w.u(c.source_sample_rate);
            w.u(c.upsampler.0);
        }
        Command::DownMix6chTo2ch(c) => {
            w.indices(&c.inputs);
            w.indices(&c.outputs);
            c.down_mix_coeff
                .iter()
                .for_each(|coeff| w.q(coeff.0 as u64));
        }
        Command::Aux(c) => {
            w.s(c.input);
            w.s(c.output);
            w.q(c.send_buffer_info);
            w.q(c.return_buffer_info);
            w.q(c.send_buffer);
            w.q(c.return_buffer);
            w.u(c.count_max);
            w.u(c.write_offset);
            w.u(c.update_count);
            w.u(c.effect_enabled);
        }
        Command::Capture(c) => {
            w.s(c.input);
            w.s(c.output);
            w.q(c.send_buffer_info);
            w.q(c.send_buffer);
            w.u(c.count_max);
            w.u(c.write_offset);
            w.u(c.update_count);
            w.u(c.effect_enabled);
        }
        Command::DeviceSink(c) => {
            w.s(c.session_id);
            w.u(c.input_count);
            w.indices(&c.inputs[..c.input_count as usize]);
            w.u(match c.source {
                SinkSource::MixBuffers => u32::MAX,
                SinkSource::Upsampler(id) => id.0,
            });
        }
        Command::CircularBufferSink(c) => {
            w.u(c.input_count);
            w.indices(&c.inputs[..c.input_count as usize]);
            w.q(c.address);
            w.u(c.size);
            w.u(c.pos);
        }
        Command::Performance(c) => {
            w.u(c.state as u32);
            w.u(c.entry_address.entry_start_time_offset);
            w.u(c.entry_address.header_entry_count_offset);
            w.u(c.entry_address.entry_processed_time_offset);
        }
        Command::ClearMixBuffer => {}
        Command::CopyMixBuffer(c) => {
            w.s(c.input_index);
            w.s(c.output_index);
        }
    }
    w.0
}

/// One command as `[id, node id, enabled, estimate, field count, fields...]`. The reference
/// stores the compressor effect's enable in the command's enabled flag; the port keeps the command
/// enabled and carries the effect's enable in `effect_enabled`, so the flag folds that back in.
fn command_words(entry: &CommandEntry) -> Vec<u32> {
    let fields = fields(&entry.command);
    let enabled = match &entry.command {
        Command::Compressor(c) => entry.enabled && c.effect_enabled,
        _ => entry.enabled,
    };
    let mut words = vec![
        entry.command.id() as u32,
        entry.node_id,
        enabled as u32,
        entry.estimated_process_time,
        fields.len() as u32,
    ];
    words.extend(fields);
    words
}

fn split_commands(mut words: &[u32]) -> Vec<&[u32]> {
    let mut commands = Vec::new();
    while !words.is_empty() {
        let (command, rest) = words.split_at(5 + words[4] as usize);
        commands.push(command);
        words = rest;
    }
    commands
}

struct Case {
    params: AudioRendererParameterInternal,
    active: bool,
    channels: u32,
    drop_voice: bool,
    render_time_limit_percent: u32,
}

impl Case {
    fn read(reader: &mut Reader<'_>) -> Self {
        assert_eq!(reader.bytes(4), b"LRCG");
        assert_eq!(reader.u32(), 1);
        Self {
            params: bytemuck::pod_read_unaligned(
                reader.bytes(size_of::<AudioRendererParameterInternal>()),
            ),
            active: reader.u32() != 0,
            channels: reader.u32(),
            drop_voice: reader.u32() != 0,
            render_time_limit_percent: reader.u32(),
        }
    }

    fn open(&self) -> RendererSystem {
        let mut system = RendererSystem::new(&self.params, work_buffer_size(&self.params))
            .unwrap_or_else(|code| panic!("open failed with {code:?}"));
        if self.active {
            system.start();
        }
        system.drop_voice = self.drop_voice;
        system.render_time_limit_percent = self.render_time_limit_percent;
        system
    }
}

struct Step<'a> {
    input: &'a [u8],
    output_len: usize,
    perf_len: usize,
    command_count: u32,
    voices_dropped: u32,
    commands: Vec<u32>,
    digest: Vec<u32>,
}

fn words(reader: &mut Reader<'_>) -> Vec<u32> {
    let len = reader.u32() as usize;
    reader
        .bytes(len * 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

impl<'a> Step<'a> {
    fn read(reader: &mut Reader<'a>) -> Self {
        let input_len = reader.u32() as usize;
        let output_len = reader.u32() as usize;
        let perf_len = reader.u32() as usize;
        Self {
            input: reader.bytes(input_len),
            output_len,
            perf_len,
            command_count: reader.u32(),
            voices_dropped: reader.u32(),
            commands: words(reader),
            digest: words(reader),
        }
    }

    fn update(&self, system: &mut RendererSystem) {
        let mut output = vec![0; self.output_len];
        let mut perf = vec![0; self.perf_len];
        system
            .update(self.input, &mut perf, &mut output)
            .unwrap_or_else(|code| panic!("update failed with {code:?}"));
    }
}

fn fixture(name: &str) -> &'static [u8] {
    match name {
        "commands_rev1" => include_bytes!("golden/commands_rev1.bin"),
        "commands_rev5_perf" => include_bytes!("golden/commands_rev5_perf.bin"),
        "commands_rev9_effects_v2" => include_bytes!("golden/commands_rev9_effects_v2.bin"),
        "commands_rev10_6ch" => include_bytes!("golden/commands_rev10_6ch.bin"),
        "commands_rev13_splitter_v2a" => include_bytes!("golden/commands_rev13_splitter_v2a.bin"),
        "commands_rev15_current" => include_bytes!("golden/commands_rev15_current.bin"),
        "commands_rev15_32k" => include_bytes!("golden/commands_rev15_32k.bin"),
        "commands_rev15_native" => include_bytes!("golden/commands_rev15_native.bin"),
        "commands_rev16_native" => include_bytes!("golden/commands_rev16_native.bin"),
        "commands_rev16_voice_drop" => include_bytes!("golden/commands_rev16_voice_drop.bin"),
        "commands_rev5_voice_drop" => include_bytes!("golden/commands_rev5_voice_drop.bin"),
        _ => panic!("no fixture {name}"),
    }
}

fn replay(name: &str) {
    let data = fixture(name);
    let mut reader = Reader { data, offset: 0 };
    let case = Case::read(&mut reader);
    let steps = reader.u32();
    let mut system = case.open();
    for step in 0..steps {
        let expected = Step::read(&mut reader);
        expected.update(&mut system);
        let list = system.generate_command(case.channels);
        let actual: Vec<u32> = list.commands.iter().flat_map(command_words).collect();

        let actual_commands = split_commands(&actual);
        let expected_commands = split_commands(&expected.commands);
        for (i, (a, e)) in actual_commands.iter().zip(&expected_commands).enumerate() {
            assert_eq!(
                a, e,
                "{name} step {step}: command {i} differs (id, node, enabled, estimate, len, fields...)"
            );
        }
        assert_eq!(
            actual_commands.len(),
            expected.command_count as usize,
            "{name} step {step}: command count"
        );
        assert_eq!(expected_commands.len(), expected.command_count as usize);
        assert_eq!(
            system.voices_dropped, expected.voices_dropped,
            "{name} step {step}: voices dropped"
        );

        let actual_digest = digest(&system);
        if let Some(at) = first_difference_words(&actual_digest, &expected.digest) {
            let end = (at + 8).min(actual_digest.len()).min(expected.digest.len());
            panic!(
                "{name} step {step}: state digest differs at word {at}\n  actual   {:08x?}\n  expected {:08x?}",
                &actual_digest[at..end],
                &expected.digest[at..end]
            );
        }
    }
    assert_eq!(reader.offset, data.len(), "{name}: trailing bytes");
}

macro_rules! golden {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                replay(stringify!($name));
            }
        )*
    };
}

golden!(
    commands_rev1,
    commands_rev5_perf,
    commands_rev9_effects_v2,
    commands_rev10_6ch,
    commands_rev13_splitter_v2a,
    commands_rev15_current,
    commands_rev15_32k,
    commands_rev15_native,
    commands_rev16_native,
    commands_rev16_voice_drop,
    commands_rev5_voice_drop,
);

/// The first update of a fixture, applied to a fresh renderer.
fn opened(name: &str, drop_voice: Option<u32>) -> RendererSystem {
    let mut reader = Reader {
        data: fixture(name),
        offset: 0,
    };
    let case = Case::read(&mut reader);
    reader.u32();
    let mut system = case.open();
    if let Some(limit) = drop_voice {
        system.drop_voice = true;
        system.render_time_limit_percent = limit;
    }
    Step::read(&mut reader).update(&mut system);
    system
}

const POOL_BASE: u64 = 0x100_0000;
const SINE_VOICE_BUFFER: usize = 0x5000;

fn device_peak(memory: &mut VecMemory) -> (Option<u32>, i16) {
    let mut system = opened("commands_rev15_current", None);
    system.generate_command(2);
    let clock = HostClock::default();
    let mut device = vec![0i16; 240 * 6];
    let mut processor = CommandListProcessor::new();
    let (list, frame) = system.dsp_frame(memory, &clock, &mut device);
    let report = processor.process(list, frame);
    let peak = device.iter().map(|s| s.saturating_abs()).max().unwrap_or(0);
    (report.device_sink.map(|sink| sink.frame_count), peak)
}

/// Update, generate and process one frame end to end: voice 3 of the fixture plays a PCM16
/// buffer straight into the final mix, which the device sink reads.
#[test]
fn generated_list_plays_a_pcm16_sine_on_the_device_sink() {
    let mut silent = VecMemory::new(POOL_BASE, 0x10_0000);
    let (frames, peak) = device_peak(&mut silent);
    assert_eq!(frames, Some(240));
    assert_eq!(peak, 0, "silent wave buffers must give a silent device");

    let mut memory = VecMemory::new(POOL_BASE, 0x10_0000);
    for i in 0..0x200 {
        let sample = (f32::sin(i as f32 * std::f32::consts::TAU / 48.0) * 12000.0) as i16;
        let at = SINE_VOICE_BUFFER + i * 2;
        memory.bytes[at..at + 2].copy_from_slice(&sample.to_le_bytes());
    }
    let (frames, peak) = device_peak(&mut memory);
    assert_eq!(frames, Some(240));
    assert!(peak > 1000, "device peak {peak} is too quiet for the sine");
}

/// A disabled compressor stays executable with `effect_enabled` clear, so the DSP copies its
/// inputs to its outputs instead of skipping the command.
#[test]
fn a_disabled_compressor_keeps_its_bypass_copy() {
    let data = fixture("commands_rev15_current");
    let mut reader = Reader { data, offset: 0 };
    let case = Case::read(&mut reader);
    let steps: Vec<Step<'_>> = (0..reader.u32()).map(|_| Step::read(&mut reader)).collect();
    let mut system = case.open();
    let mut effect_enabled = Vec::new();
    for step in &steps {
        step.update(&mut system);
        system.generate_command(case.channels);
        for entry in &system.command_list.commands {
            if let Command::Compressor(command) = &entry.command {
                assert!(entry.enabled);
                effect_enabled.push(command.effect_enabled);
            }
        }
    }
    assert_eq!(effect_enabled, [true, true, false]);
}

/// The fixture's playing voices from lowest to highest priority; voice 3 has the highest
/// priority and is never dropped.
const DROP_ORDER: [usize; 3] = [0, 1, 2];
const HIGHEST_PRIORITY_VOICE: usize = 3;

#[test]
fn voice_drop_removes_lowest_priority_voices_first() {
    let mut previous = 0;
    for limit in (0..=100).rev() {
        let mut system = opened("commands_rev16_native", Some(limit));
        system.generate_command(2);
        let dropped: Vec<usize> = (0..system.voices.count() as usize)
            .filter(|&i| system.voices.info(i).voice_dropped)
            .collect();
        assert_eq!(
            dropped,
            DROP_ORDER[..dropped.len()],
            "limit {limit}%: dropped voices must be the lowest priority ones"
        );
        assert_eq!(system.voices_dropped as usize, dropped.len());
        assert!(
            dropped.len() >= previous,
            "limit {limit}%: a lower limit dropped fewer voices"
        );
        previous = dropped.len();

        for entry in &system.command_list.commands {
            let voice = (entry.node_id >> 16 & 0xFFF) as usize;
            if entry.node_id >> 28 != 1 || !dropped.contains(&voice) {
                continue;
            }
            let kept = matches!(
                entry.command,
                Command::DepopPrepare(_) | Command::Performance(_)
            );
            assert!(
                entry.enabled == kept || matches!(entry.command, Command::Performance(_)),
                "limit {limit}%: dropped voice {voice} kept {:?}",
                entry.command.id()
            );
        }
        assert!(!system.voices.info(HIGHEST_PRIORITY_VOICE).voice_dropped);
    }
    assert_eq!(
        previous,
        DROP_ORDER.len(),
        "a 0% limit drops every droppable voice"
    );

    let mut system = opened("commands_rev16_native", Some(100));
    system.generate_command(2);
    assert!(
        system
            .voices
            .infos()
            .iter()
            .all(|voice| !voice.voice_dropped)
    );
}

/// Allocations the current thread makes while `f` runs; tests running alongside on other threads
/// do not add to the count.
fn allocations_of(f: impl FnOnce()) -> u64 {
    allocation_counter::measure(f).count_total
}

/// Generation reuses the command list, so once a frame has sized it, later frames allocate
/// nothing, with performance entries and the voice drop pass active.
#[test]
fn generating_a_frame_after_the_first_allocates_nothing() {
    let data = fixture("commands_rev16_voice_drop");
    let mut reader = Reader { data, offset: 0 };
    let case = Case::read(&mut reader);
    let steps: Vec<Step<'_>> = (0..reader.u32()).map(|_| Step::read(&mut reader)).collect();
    assert!(allocations_of(|| drop(std::hint::black_box(vec![0u8; 16]))) > 0);
    let mut system = case.open();
    steps[0].update(&mut system);
    system.generate_command(case.channels);
    for _ in 0..3 {
        assert_eq!(
            allocations_of(|| _ = system.generate_command(case.channels)),
            0
        );
    }
    for step in &steps[1..] {
        step.update(&mut system);
        assert_eq!(
            allocations_of(|| _ = system.generate_command(case.channels)),
            0
        );
    }
}
