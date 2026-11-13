//! Runs one frame's command list (`command_list_processor.cpp`).

use std::ops::Range;
use std::time::Instant;

use super::command::{Command, CommandList, PerformanceCommand, PerformanceState};
use super::decode::{DecodeScratch, SampleFormat, TEMP_BUFFER_SIZE, process_data_source};
use super::memory::DspMemory;
use super::sink::DeviceSinkFrame;
use super::state::DspState;
use super::{
    MAX_CHANNELS, aux, biquad, compressor, delay, i3dl2, limiter, mix, reverb, sink, upsample,
};

/// The time source performance commands and per-command timings read, in microseconds.
pub trait DspClock {
    fn now_us(&self) -> u64;
}

/// Wall-clock time since the clock was created.
#[derive(Clone, Copy, Debug)]
pub struct HostClock {
    origin: Instant,
}

impl Default for HostClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl DspClock for HostClock {
    fn now_us(&self) -> u64 {
        self.origin.elapsed().as_micros() as u64
    }
}

/// Everything a command list reads and writes while it runs.
pub struct DspFrame<'a> {
    /// `buffer_count * sample_count` samples, one block per mix buffer.
    pub mix_buffers: &'a mut [i32],
    pub state: &'a mut DspState,
    pub memory: &'a mut dyn DspMemory,
    pub clock: &'a dyn DspClock,
    /// Interleaved i16 device output; device sinks write up to 240 frames of up to six channels.
    pub device_output: &'a mut [i16],
}

/// The outcome of running a command list.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProcessReport {
    pub processed_command_count: u32,
    pub remaining_command_count: u32,
    pub elapsed_us: u64,
    /// Set when a device sink wrote into [`DspFrame::device_output`].
    pub device_sink: Option<DeviceSinkFrame>,
}

/// The frame format and mix buffers one command sees.
pub(super) struct Frame<'a> {
    pub sample_count: usize,
    pub target_sample_rate: u32,
    pub buffer_count: u32,
    pub mix: &'a mut [i32],
}

impl Frame<'_> {
    /// The samples of mix buffer `index`, if it lies inside the mix buffers.
    pub fn range(&self, index: impl Into<i64>) -> Option<Range<usize>> {
        let start = usize::try_from(index.into())
            .ok()?
            .checked_mul(self.sample_count)?;
        let end = start.checked_add(self.sample_count)?;
        (end <= self.mix.len()).then_some(start..end)
    }
}

/// Resolves the first `channels` input and output buffers of an effect to sample offsets.
pub(super) fn channel_buffers(
    frame: &Frame<'_>,
    inputs: &[i16; MAX_CHANNELS],
    outputs: &[i16; MAX_CHANNELS],
    channels: usize,
) -> Option<([usize; MAX_CHANNELS], [usize; MAX_CHANNELS])> {
    let mut input_offsets = [0; MAX_CHANNELS];
    let mut output_offsets = [0; MAX_CHANNELS];
    for ch in 0..channels.min(MAX_CHANNELS) {
        input_offsets[ch] = frame.range(inputs[ch])?.start;
        output_offsets[ch] = frame.range(outputs[ch])?.start;
    }
    Some((input_offsets, output_offsets))
}

/// Copies each input to its output, as disabled effects do.
pub(super) fn bypass(
    mix: &mut [i32],
    inputs: &[usize; MAX_CHANNELS],
    outputs: &[usize; MAX_CHANNELS],
    channels: usize,
    count: usize,
) {
    for ch in 0..channels.min(MAX_CHANNELS) {
        if inputs[ch] != outputs[ch] {
            mix.copy_within(inputs[ch]..inputs[ch] + count, outputs[ch]);
        }
    }
}

pub(super) fn is_channel_count_valid(count: u16) -> bool {
    matches!(count, 1 | 2 | 4 | 6)
}

/// Executes command lists. Holds the decode scratch and timing storage so a frame allocates
/// nothing once the processor has seen its largest command list.
pub struct CommandListProcessor {
    scratch: Box<DecodeScratch>,
    command_times: Vec<u32>,
}

impl Default for CommandListProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandListProcessor {
    pub fn new() -> Self {
        Self {
            scratch: vec![0; TEMP_BUFFER_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            command_times: Vec::new(),
        }
    }

    /// Microseconds each command of the last list took; disabled commands read 0.
    pub fn command_times(&self) -> &[u32] {
        &self.command_times
    }

    /// Runs every command of `list` against `frame`.
    pub fn process(&mut self, list: &CommandList, frame: DspFrame<'_>) -> ProcessReport {
        let DspFrame {
            mix_buffers,
            state,
            memory,
            clock,
            device_output,
        } = frame;
        let start = clock.now_us();
        let mut frame = Frame {
            sample_count: list.sample_count as usize,
            target_sample_rate: list.sample_rate,
            buffer_count: list.buffer_count,
            mix: mix_buffers,
        };
        self.command_times.clear();
        self.command_times.resize(list.commands.len(), 0);
        let mut report = ProcessReport::default();
        for (entry, time) in list.commands.iter().zip(&mut self.command_times) {
            report.processed_command_count += 1;
            if !entry.enabled {
                continue;
            }
            let command_start = clock.now_us();
            let scratch = &mut *self.scratch;
            match &entry.command {
                Command::DataSourcePcmInt16Version1(c) => process_data_source(
                    c,
                    SampleFormat::PcmInt16,
                    true,
                    &mut frame,
                    state,
                    memory,
                    scratch,
                ),
                Command::DataSourcePcmInt16Version2(c) => process_data_source(
                    c,
                    SampleFormat::PcmInt16,
                    false,
                    &mut frame,
                    state,
                    memory,
                    scratch,
                ),
                Command::DataSourcePcmFloatVersion1(c) => process_data_source(
                    c,
                    SampleFormat::PcmFloat,
                    true,
                    &mut frame,
                    state,
                    memory,
                    scratch,
                ),
                Command::DataSourcePcmFloatVersion2(c) => process_data_source(
                    c,
                    SampleFormat::PcmFloat,
                    false,
                    &mut frame,
                    state,
                    memory,
                    scratch,
                ),
                Command::DataSourceAdpcmVersion1(c) => process_data_source(
                    c,
                    SampleFormat::Adpcm,
                    true,
                    &mut frame,
                    state,
                    memory,
                    scratch,
                ),
                Command::DataSourceAdpcmVersion2(c) => process_data_source(
                    c,
                    SampleFormat::Adpcm,
                    false,
                    &mut frame,
                    state,
                    memory,
                    scratch,
                ),
                Command::Volume(c) => mix::volume(c, &mut frame),
                Command::VolumeRamp(c) => mix::volume_ramp(c, &mut frame),
                Command::BiquadFilter(c) => biquad::biquad_filter(c, &mut frame, state),
                Command::Mix(c) => mix::mix(c, &mut frame),
                Command::MixRamp(c) => mix::mix_ramp(c, &mut frame, state),
                Command::MixRampGrouped(c) => mix::mix_ramp_grouped(c, &mut frame, state),
                Command::DepopPrepare(c) => mix::depop_prepare(c, state),
                Command::DepopForMixBuffers(c) => mix::depop_for_mix_buffers(c, &mut frame, state),
                Command::Delay(c) => delay::delay(c, &mut frame, state),
                Command::Upsample(c) => upsample::upsample(c, &frame, state),
                Command::DownMix6chTo2ch(c) => mix::downmix_6ch_to_2ch(c, &mut frame),
                Command::Aux(c) => aux::aux(c, &mut frame, memory),
                Command::DeviceSink(c) => {
                    report.device_sink = Some(sink::device_sink(c, &frame, state, device_output));
                }
                Command::CircularBufferSink(c) => sink::circular_buffer_sink(c, &frame, memory),
                Command::Reverb(c) => reverb::reverb(c, &mut frame, state),
                Command::I3dl2Reverb(c) => i3dl2::i3dl2_reverb(c, &mut frame, state),
                Command::Performance(c) => performance(c, state, clock.now_us() - start),
                Command::ClearMixBuffer => mix::clear_mix_buffers(&mut frame),
                Command::CopyMixBuffer(c) => mix::copy_mix_buffer(c, &mut frame),
                Command::LightLimiterVersion1(c) => {
                    limiter::light_limiter(c, false, &mut frame, state)
                }
                Command::LightLimiterVersion2(c) => {
                    limiter::light_limiter(c, true, &mut frame, state)
                }
                Command::MultiTapBiquadFilter(c) => {
                    biquad::multi_tap_biquad_filter(c, &mut frame, state)
                }
                Command::Capture(c) => aux::capture(c, &frame, memory),
                Command::Compressor(c) => compressor::compressor(c, &mut frame, state),
            }
            *time = clock.now_us().saturating_sub(command_start) as u32;
        }
        report.remaining_command_count =
            list.commands.len() as u32 - report.processed_command_count;
        report.elapsed_us = clock.now_us().saturating_sub(start);
        report
    }
}

fn write_u32(buffer: &mut [u8], offset: u64, update: impl FnOnce(u32) -> u32) {
    let Ok(offset) = usize::try_from(offset) else {
        return;
    };
    if let Some(bytes) = buffer.get_mut(offset..offset.saturating_add(4)) {
        let value = u32::from_le_bytes(bytes.try_into().unwrap());
        bytes.copy_from_slice(&update(value).to_le_bytes());
    }
}

fn performance(command: &PerformanceCommand, state: &mut DspState, elapsed_us: u64) {
    let entry = command.entry_address;
    let base = entry.translated_address as u64;
    match command.state {
        PerformanceState::Start => {
            let offset = base + entry.entry_start_time_offset as u64;
            write_u32(&mut state.performance, offset, |_| elapsed_us as u32);
        }
        PerformanceState::Stop => {
            let offset = base + entry.entry_processed_time_offset as u64;
            write_u32(&mut state.performance, offset, |_| elapsed_us as u32);
            let count = base + entry.header_entry_count_offset as u64;
            write_u32(&mut state.performance, count, |count| count.wrapping_add(1));
        }
        PerformanceState::Invalid => {}
    }
}
