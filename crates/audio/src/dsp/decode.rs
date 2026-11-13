//! Voice decoding: PCM16, PCM float and ADPCM wave buffers with looping (`decode.cpp`).

use super::MAX_WAVE_BUFFERS;
use super::command::DataSourceCommand;
use super::fixed::{Fx32, Q15};
use super::memory::DspMemory;
use super::processor::Frame;
use super::resample::{SrcQuality, resample};
use super::state::{AdpcmContext, DspState, VoiceState};

/// Samples of decode scratch per data source command (`TempBufferSize`).
pub const TEMP_BUFFER_SIZE: usize = 0x3F00;

/// Scratch the decoder reuses across commands, so decoding allocates nothing per frame.
pub type DecodeScratch = [i16; TEMP_BUFFER_SIZE];

/// A wave buffer as the DSP consumes it (`WaveBufferVersion2`): a guest sample range plus
/// loop range and count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaveBuffer {
    pub buffer: u64,
    pub context: u64,
    pub buffer_size: u64,
    pub context_size: u64,
    pub start_offset: u32,
    pub end_offset: u32,
    pub loop_start_offset: u32,
    pub loop_end_offset: u32,
    /// Loops left to play; negative loops forever.
    pub loop_count: i32,
    pub looping: bool,
    pub stream_ended: bool,
}

/// The sample encoding of a data source command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    PcmInt16,
    PcmFloat,
    Adpcm,
}

/// Executes a data source command: decodes into its output mix buffer and advances the voice.
pub(super) fn process_data_source(
    command: &DataSourceCommand,
    format: SampleFormat,
    version1: bool,
    frame: &mut Frame<'_>,
    state: &mut DspState,
    memory: &dyn DspMemory,
    scratch: &mut DecodeScratch,
) {
    let Some(output) = frame.range(command.output_index) else {
        return;
    };
    let Some(voice) = state.voices.get_mut(command.voice_state.0 as usize) else {
        return;
    };
    let mut wave_buffers = command.wave_buffers;
    if version1 {
        for wave_buffer in &mut wave_buffers {
            wave_buffer.loop_start_offset = wave_buffer.start_offset;
            wave_buffer.loop_end_offset = wave_buffer.end_offset;
            wave_buffer.loop_count = if wave_buffer.looping { -1 } else { 0 };
        }
    }
    let (channel, channel_count) = match format {
        SampleFormat::Adpcm => (0, 1),
        _ => (command.channel_index, command.channel_count),
    };
    let args = DecodeArgs {
        format,
        wave_buffers: &wave_buffers,
        channel,
        channel_count,
        src_quality: command.src_quality,
        pitch: command.pitch,
        source_sample_rate: command.sample_rate,
        target_sample_rate: frame.target_sample_rate,
        sample_count: frame.sample_count as u32,
        data_address: command.data_address,
        data_size: command.data_size,
        reset_played_at_loop: command.flags & DataSourceCommand::FLAG_RESET_PLAYED_AT_LOOP != 0,
        skip_pitch_and_src: command.flags & DataSourceCommand::FLAG_SKIP_PITCH_AND_SRC != 0,
    };
    decode_from_wave_buffers(memory, &mut frame.mix[output], voice, &args, scratch);
}

struct DecodeArgs<'a> {
    format: SampleFormat,
    wave_buffers: &'a [WaveBuffer; MAX_WAVE_BUFFERS],
    channel: i8,
    channel_count: i8,
    src_quality: SrcQuality,
    pitch: f32,
    source_sample_rate: u32,
    target_sample_rate: u32,
    sample_count: u32,
    data_address: u64,
    data_size: u64,
    reset_played_at_loop: bool,
    skip_pitch_and_src: bool,
}

struct DecodeRequest<'a> {
    buffer: u64,
    buffer_size: u64,
    start_offset: u32,
    end_offset: u32,
    channel_count: i8,
    coefficients: &'a [i16; 16],
    target_channel: i8,
    offset: u32,
    samples_to_read: u32,
}

/// Buffered sequential access to a guest byte range, so per-sample decoding does not cost one
/// memory call per sample.
struct GuestReader<'a> {
    memory: &'a dyn DspMemory,
    address: u64,
    limit: u64,
    window_start: u64,
    window_len: usize,
    window: [u8; 512],
}

impl<'a> GuestReader<'a> {
    fn new(memory: &'a dyn DspMemory, address: u64, limit: u64) -> Self {
        Self {
            memory,
            address,
            limit,
            window_start: 0,
            window_len: 0,
            window: [0; 512],
        }
    }

    fn bytes<const N: usize>(&mut self, offset: u64) -> [u8; N] {
        let window_end = self.window_start + self.window_len as u64;
        if offset < self.window_start || offset + N as u64 > window_end {
            let wanted = self.limit.saturating_sub(offset).clamp(N as u64, 512);
            self.window_start = offset;
            self.window_len = wanted as usize;
            self.memory.read(
                self.address.wrapping_add(offset),
                &mut self.window[..self.window_len],
            );
        }
        let start = (offset - self.window_start) as usize;
        self.window[start..start + N].try_into().unwrap()
    }
}

trait PcmSample: Copy {
    const SIZE: u64;
    fn decode(bytes: &mut GuestReader<'_>, offset: u64) -> i16;
}

impl PcmSample for i16 {
    const SIZE: u64 = 2;
    fn decode(bytes: &mut GuestReader<'_>, offset: u64) -> i16 {
        i16::from_le_bytes(bytes.bytes(offset))
    }
}

impl PcmSample for f32 {
    const SIZE: u64 = 4;
    fn decode(bytes: &mut GuestReader<'_>, offset: u64) -> i16 {
        let sample = f32::from_le_bytes(bytes.bytes(offset));
        ((sample * i16::MAX as f32) as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }
}

fn decode_pcm<T: PcmSample>(memory: &dyn DspMemory, out: &mut [i16], req: &DecodeRequest) -> u32 {
    if req.buffer == 0 || req.buffer_size == 0 || req.start_offset >= req.end_offset {
        return 0;
    }
    let samples = req.samples_to_read.min(
        req.end_offset
            .wrapping_sub(req.start_offset)
            .wrapping_sub(req.offset),
    );
    let channel_count = req.channel_count as i32 as u32;
    if req.channel_count == 1 && req.target_channel != 0 {
        tracing::error!(
            channel = req.target_channel,
            "invalid target channel for mono PCM"
        );
        return 0;
    }
    if req.channel_count <= 0 || req.target_channel < 0 {
        return 0;
    }
    let first = (req.start_offset.wrapping_add(req.offset) as u64) * channel_count as u64;
    let source = req.buffer.wrapping_add(first * T::SIZE);
    let span = samples as u64 * channel_count as u64 * T::SIZE;
    let mut reader = GuestReader::new(memory, source, span);
    let stride = channel_count as u64 * T::SIZE;
    let channel = req.target_channel as u64 * T::SIZE;
    for (i, sample) in out.iter_mut().take(samples as usize).enumerate() {
        *sample = T::decode(&mut reader, i as u64 * stride + channel);
    }
    samples
}

fn decode_adpcm(
    memory: &dyn DspMemory,
    out: &mut [i16],
    req: &DecodeRequest,
    context: &mut AdpcmContext,
) -> u32 {
    const SAMPLES_PER_FRAME: u32 = 14;
    const NIBBLES_PER_FRAME: u32 = 16;

    if req.buffer == 0 || req.buffer_size == 0 || req.end_offset < req.start_offset {
        return 0;
    }
    let mut end = (req.end_offset % SAMPLES_PER_FRAME)
        + NIBBLES_PER_FRAME * (req.end_offset / SAMPLES_PER_FRAME);
    end += if req.end_offset.is_multiple_of(SAMPLES_PER_FRAME) {
        1
    } else {
        3
    };
    if req.buffer_size < (end / 2) as u64 {
        return 0;
    }
    let start_pos = req.start_offset.wrapping_add(req.offset);
    let samples_to_process = req
        .end_offset
        .wrapping_sub(start_pos)
        .min(req.samples_to_read);
    if samples_to_process == 0 {
        return 0;
    }
    let mut samples_to_read = samples_to_process;
    let remaining_in_frame = start_pos % SAMPLES_PER_FRAME;
    let mut position = (start_pos / SAMPLES_PER_FRAME) * NIBBLES_PER_FRAME + remaining_in_frame;
    if remaining_in_frame != 0 {
        position += 2;
    }
    let span = (samples_to_process as u64).div_ceil(SAMPLES_PER_FRAME as u64) * 8 + 16;
    let mut reader = GuestReader::new(memory, req.buffer.wrapping_add((position / 2) as u64), span);
    let mut byte = |index: u32| reader.bytes::<1>(index as u64)[0];

    let mut header = context.header;
    let mut scale = (header & 0xF) as u32;
    let mut coeff_index = ((header >> 4) & 0x7) as usize;
    let mut coeff0 = req.coefficients[coeff_index * 2] as i32;
    let mut coeff1 = req.coefficients[coeff_index * 2 + 1] as i32;
    let mut yn0 = context.yn0;
    let mut yn1 = context.yn1;

    let step = |nibble: u8| -> i32 { ((nibble as i32) << 28) >> 28 };
    let mut decode_sample = |code: i32, scale: u32, coeff0: i32, coeff1: i32| -> i16 {
        let xn = code.wrapping_mul(1 << scale);
        let prediction = coeff0 * yn0 as i32 + coeff1 * yn1 as i32;
        let sample = (xn
            .wrapping_shl(11)
            .wrapping_add(0x400)
            .wrapping_add(prediction))
            >> 11;
        yn1 = yn0;
        yn0 = sample.clamp(-0x8000, 0x7FFF) as i16;
        yn0
    };

    let mut read_index = 0u32;
    let mut write_index = 0usize;
    while samples_to_read > 0 && write_index < out.len() {
        if position.is_multiple_of(NIBBLES_PER_FRAME) {
            header = byte(read_index) as u16;
            read_index += 1;
            scale = (header & 0xF) as u32;
            coeff_index = ((header >> 4) & 0x7) as usize;
            coeff0 = req.coefficients[coeff_index * 2] as i32;
            coeff1 = req.coefficients[coeff_index * 2 + 1] as i32;
            position += 2;
            if samples_to_read >= SAMPLES_PER_FRAME {
                for i in 0..SAMPLES_PER_FRAME / 2 {
                    let packed = byte(read_index + i);
                    for nibble in [packed >> 4, packed & 0xF] {
                        let sample = decode_sample(step(nibble), scale, coeff0, coeff1);
                        if let Some(slot) = out.get_mut(write_index) {
                            *slot = sample;
                        }
                        write_index += 1;
                    }
                }
                read_index += SAMPLES_PER_FRAME / 2;
                position += SAMPLES_PER_FRAME;
                samples_to_read -= SAMPLES_PER_FRAME;
                continue;
            }
        }
        let packed = byte(read_index);
        let nibble = if position & 1 != 0 {
            read_index += 1;
            packed & 0xF
        } else {
            packed >> 4
        };
        out[write_index] = decode_sample(step(nibble), scale, coeff0, coeff1);
        write_index += 1;
        position += 1;
        samples_to_read -= 1;
    }

    *context = AdpcmContext { header, yn0, yn1 };
    samples_to_process
}

struct WaveBufferCursor<'a> {
    index: &'a mut u32,
    played: &'a mut u64,
    consumed: &'a mut u32,
}

impl WaveBufferCursor<'_> {
    fn end(&mut self, voice: &mut VoiceState, wave_buffer: &WaveBuffer) {
        voice.wave_buffer_valid[*self.index as usize] = false;
        voice.loop_count = 0;
        if wave_buffer.stream_ended {
            *self.played = 0;
        }
        *self.index = (*self.index + 1) % MAX_WAVE_BUFFERS as u32;
        *self.consumed += 1;
    }
}

fn decode_from_wave_buffers(
    memory: &dyn DspMemory,
    output: &mut [i32],
    voice: &mut VoiceState,
    args: &DecodeArgs,
    temp: &mut DecodeScratch,
) {
    let mut remaining = args.sample_count;
    let mut fraction = voice.fraction;
    let ratio =
        Q15::from_f32(args.source_sample_rate as f32 / args.target_sample_rate as f32 * args.pitch);
    let size_required = fraction + Q15::from_int(remaining as i64) * ratio;
    if size_required.raw() < 0 {
        return;
    }
    let history = args.src_quality.history_len();
    if (history as u32).wrapping_add(size_required.to_int_floor() as u32) > TEMP_BUFFER_SIZE as u32
    {
        return;
    }
    let temp_capacity = Fx32::<15>::from_int(TEMP_BUFFER_SIZE as i32).widen();
    let max_remaining = if ratio.raw() == 0 {
        remaining
    } else {
        ((temp_capacity - fraction) / ratio)
            .to_uint_floor()
            .min(remaining)
    };

    let mut coefficients = [0i16; 16];
    if args.format == SampleFormat::Adpcm {
        let mut bytes = [0u8; 32];
        let len = (args.data_size as usize).min(bytes.len());
        memory.read(args.data_address, &mut bytes[..len]);
        for (coefficient, pair) in coefficients.iter_mut().zip(bytes.as_chunks::<2>().0) {
            *coefficient = i16::from_le_bytes(*pair);
        }
    }

    let mut consumed = voice.wave_buffers_consumed;
    let mut index = voice.wave_buffer_index;
    let mut played = voice.played_sample_count;
    let mut offset = voice.offset;
    let mut starved = false;
    let mut out_pos = 0usize;

    while remaining > 0 {
        let samples_to_write = remaining.min(max_remaining);
        if samples_to_write == 0 {
            break;
        }
        let samples_to_read =
            (fraction + Q15::from_int(samples_to_write as i64) * ratio).to_uint_floor();
        let mut temp_pos = 0usize;
        if !args.skip_pitch_and_src {
            temp[..history].copy_from_slice(&voice.sample_history[..history]);
            temp_pos = history;
        }

        let mut samples_read = 0u32;
        while samples_read < samples_to_read {
            if index as usize >= MAX_WAVE_BUFFERS {
                tracing::error!(index, "invalid wave buffer index");
                index = 0;
                consumed = MAX_WAVE_BUFFERS as u32;
                voice.wave_buffer_valid = [false; MAX_WAVE_BUFFERS];
            }
            if !voice.wave_buffer_valid[index as usize] {
                starved = true;
                break;
            }
            let wave_buffer = &args.wave_buffers[index as usize];
            if offset == 0 && args.format == SampleFormat::Adpcm && wave_buffer.context != 0 {
                let mut bytes = [0u8; AdpcmContext::GUEST_SIZE];
                let ctx = voice.adpcm_context;
                bytes[..2].copy_from_slice(&ctx.header.to_le_bytes());
                bytes[2..4].copy_from_slice(&ctx.yn0.to_le_bytes());
                bytes[4..].copy_from_slice(&ctx.yn1.to_le_bytes());
                let len = (wave_buffer.context_size as usize).min(AdpcmContext::GUEST_SIZE);
                memory.read(wave_buffer.context, &mut bytes[..len]);
                voice.adpcm_context = AdpcmContext::from_guest(bytes);
            }

            let (start_offset, end_offset) = if wave_buffer.looping
                && voice.loop_count > 0
                && wave_buffer.loop_start_offset <= wave_buffer.loop_end_offset
            {
                (wave_buffer.loop_start_offset, wave_buffer.loop_end_offset)
            } else {
                (wave_buffer.start_offset, wave_buffer.end_offset)
            };
            let request = DecodeRequest {
                buffer: wave_buffer.buffer,
                buffer_size: wave_buffer.buffer_size,
                start_offset,
                end_offset,
                channel_count: args.channel_count,
                coefficients: &coefficients,
                target_channel: args.channel,
                offset,
                samples_to_read: samples_to_read - samples_read,
            };
            let out = &mut temp[temp_pos..];
            let decoded = match args.format {
                SampleFormat::PcmInt16 => decode_pcm::<i16>(memory, out, &request),
                SampleFormat::PcmFloat => decode_pcm::<f32>(memory, out, &request),
                SampleFormat::Adpcm => {
                    decode_adpcm(memory, out, &request, &mut voice.adpcm_context)
                }
            };

            played = played.wrapping_add(decoded as u64);
            samples_read += decoded;
            temp_pos += decoded as usize;
            offset = offset.wrapping_add(decoded);
            if decoded != 0 && offset < end_offset.wrapping_sub(start_offset) {
                continue;
            }

            offset = 0;
            let mut finished = WaveBufferCursor {
                index: &mut index,
                played: &mut played,
                consumed: &mut consumed,
            };
            if wave_buffer.looping {
                voice.loop_count += 1;
                if wave_buffer.loop_count >= 0
                    && (voice.loop_count > wave_buffer.loop_count || decoded == 0)
                {
                    finished.end(voice, wave_buffer);
                }
                if decoded == 0 {
                    starved = true;
                    break;
                }
                if args.reset_played_at_loop {
                    played = 0;
                }
            } else {
                finished.end(voice, wave_buffer);
            }
        }

        let out_start = out_pos.min(output.len());
        let out = &mut output[out_start..];
        if args.skip_pitch_and_src {
            for (slot, &sample) in out.iter_mut().zip(&temp[..samples_read as usize]) {
                *slot = sample as i32;
            }
        } else {
            let zero_end = (temp_pos + (samples_to_read - samples_read) as usize).min(temp.len());
            temp[temp_pos..zero_end].fill(0);
            resample(
                out,
                &temp[..],
                ratio,
                &mut fraction,
                samples_to_write as usize,
                args.src_quality,
            );
            let history_start = samples_to_read as usize;
            voice.sample_history[..history]
                .copy_from_slice(&temp[history_start..history_start + history]);
        }

        remaining -= samples_to_write;
        if remaining != 0 && starved {
            tracing::error!("samples remaining but the voice is starving");
            break;
        }
        out_pos += samples_to_write as usize;
    }

    voice.wave_buffers_consumed = consumed;
    voice.played_sample_count = played;
    voice.wave_buffer_index = index;
    voice.offset = offset;
    voice.fraction = fraction;
}
