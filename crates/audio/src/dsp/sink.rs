//! Device and circular buffer sinks (`sink/device.cpp`, `sink/circular_buffer.cpp`).

use super::command::{CircularBufferSinkCommand, DeviceSinkCommand, SinkSource};
use super::memory::DspMemory;
use super::processor::Frame;
use super::state::DspState;
use super::{MAX_CHANNELS, TARGET_SAMPLE_COUNT};

/// What a device sink wrote into the caller's output buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceSinkFrame {
    pub session_id: i32,
    pub channel_count: u32,
    pub frame_count: u32,
}

fn saturate(sample: i32) -> i16 {
    sample.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

/// Interleaves the sink's inputs into `output`, always [`TARGET_SAMPLE_COUNT`] frames as the
/// C++ does (the 32 kHz path reaches the sink through a 48 kHz upsampler).
pub(super) fn device_sink(
    command: &DeviceSinkCommand,
    frame: &Frame<'_>,
    state: &DspState,
    output: &mut [i16],
) -> DeviceSinkFrame {
    let frames = TARGET_SAMPLE_COUNT;
    let channels = (command.input_count as usize).min(MAX_CHANNELS);
    let source: &[i32] = match command.source {
        SinkSource::MixBuffers => frame.mix,
        SinkSource::Upsampler(id) => state
            .upsamplers
            .get(id.0 as usize)
            .map_or(&[], |upsampler| upsampler.samples.as_slice()),
    };
    for (channel, &input) in command.inputs[..channels].iter().enumerate() {
        let offset = usize::try_from(input).unwrap_or(usize::MAX / 2) * frames;
        for index in 0..frames {
            let sample = source.get(offset + index).copied().unwrap_or(0);
            if let Some(slot) = output.get_mut(index * channels + channel) {
                *slot = saturate(sample);
            }
        }
    }
    DeviceSinkFrame {
        session_id: command.session_id,
        channel_count: channels as u32,
        frame_count: frames as u32,
    }
}

pub(super) fn circular_buffer_sink(
    command: &CircularBufferSinkCommand,
    frame: &Frame<'_>,
    memory: &mut dyn DspMemory,
) {
    let mut block = [0i16; TARGET_SAMPLE_COUNT * MAX_CHANNELS];
    let count = frame.sample_count.min(block.len());
    let mut pos = command.pos;
    for &input in &command.inputs[..(command.input_count as usize).min(MAX_CHANNELS)] {
        let Some(range) = frame.range(input) else {
            continue;
        };
        for (slot, &sample) in block.iter_mut().zip(&frame.mix[range][..count]) {
            *slot = saturate(sample);
        }
        memory.write(
            command.address.wrapping_add(pos as u64),
            bytemuck::cast_slice(&block[..count]),
        );
        pos = pos.wrapping_add((count * 2) as u32);
        if pos >= command.size {
            pos = 0;
        }
    }
}
