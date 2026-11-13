//! `hwopus`: `IHardwareOpusDecoderManager` and `IHardwareOpusDecoder`, decoding with the
//! pure-Rust `opus1` codec.
//!
//! The work buffer sizes are the ones Eden reports for a libopus built without the neural
//! features, since games size their transfer memory from them. The transfer memory itself is
//! not used: the decoder state lives on the host.

use std::sync::Arc;
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use opus1::multistream::MultistreamDecoder;
use opus1::{Channels, ErrorKind, OpusDecoder, SampleRate};
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const MODULE: u32 = 111;
const LIB_OPUS_BAD_ARG: ResultCode = ResultCode::new(MODULE, 2);
const BUFFER_TOO_SMALL: ResultCode = ResultCode::new(MODULE, 3);
const LIB_OPUS_INTERNAL_ERROR: ResultCode = ResultCode::new(MODULE, 4);
const LIB_OPUS_UNIMPLEMENTED: ResultCode = ResultCode::new(MODULE, 5);
const LIB_OPUS_INVALID_STATE: ResultCode = ResultCode::new(MODULE, 6);
const LIB_OPUS_ALLOC_FAIL: ResultCode = ResultCode::new(MODULE, 7);
const INPUT_DATA_TOO_SMALL: ResultCode = ResultCode::new(MODULE, 8);
const LIB_OPUS_INVALID_PACKET: ResultCode = ResultCode::new(MODULE, 17);
const OUT_OF_OPUS_DECODERS: ResultCode = ResultCode::new(MODULE, 385);
const INVALID_OPUS_SAMPLE_RATE: ResultCode = ResultCode::new(MODULE, 1001);
const INVALID_OPUS_CHANNEL_COUNT: ResultCode = ResultCode::new(MODULE, 1002);

/// Decoders the DSP holds at once.
const MAX_DECODERS: usize = 24;
const OPUS_STREAM_COUNT_MAX: u32 = 255;
/// `sizeof(OpusDecodeObject)` (and the multistream one) on the 64-bit DSP side.
const DECODE_OBJECT_SIZE: u32 = 24;
/// `opus_decoder_get_size` of libopus without the neural features, mono and stereo.
const DECODER_SIZE: [u32; 2] = [18468, 27236];
/// `opus_multistream_decoder_get_size`'s layout header and per-stream decoders (8-aligned).
const MULTI_STREAM_HEADER_SIZE: u32 = 272;
const ALIGNED_MONO_DECODER_SIZE: u32 = 18472;
const ALIGNED_STEREO_DECODER_SIZE: u32 = 27240;
/// The input area of a single-stream decoder's work buffer.
const SINGLE_STREAM_INPUT_SIZE: usize = 0x600;

/// `OpusParameters`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct OpusParameters {
    sample_rate: u32,
    channel_count: u32,
}
const _: () = assert!(size_of::<OpusParameters>() == 0x8);

/// `OpusParametersEx`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct OpusParametersEx {
    sample_rate: u32,
    channel_count: u32,
    use_large_frame_size: u8,
    padding: [u8; 7],
}
const _: () = assert!(size_of::<OpusParametersEx>() == 0x10);

/// `OpusMultiStreamParameters`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct OpusMultiStreamParameters {
    sample_rate: u32,
    channel_count: u32,
    total_stream_count: u32,
    stereo_stream_count: u32,
    mappings: [u8; 256],
}
const _: () = assert!(size_of::<OpusMultiStreamParameters>() == 0x110);

/// `OpusMultiStreamParametersEx`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct OpusMultiStreamParametersEx {
    sample_rate: u32,
    channel_count: u32,
    total_stream_count: u32,
    stereo_stream_count: u32,
    use_large_frame_size: u8,
    padding: [u8; 7],
    mappings: [u8; 256],
}
const _: () = assert!(size_of::<OpusMultiStreamParametersEx>() == 0x118);

impl From<OpusParameters> for OpusParametersEx {
    fn from(params: OpusParameters) -> Self {
        Self {
            sample_rate: params.sample_rate,
            channel_count: params.channel_count,
            ..Self::zeroed()
        }
    }
}

impl From<OpusMultiStreamParameters> for OpusMultiStreamParametersEx {
    fn from(params: OpusMultiStreamParameters) -> Self {
        Self {
            sample_rate: params.sample_rate,
            channel_count: params.channel_count,
            total_stream_count: params.total_stream_count,
            stereo_stream_count: params.stereo_stream_count,
            mappings: params.mappings,
            ..Self::zeroed()
        }
    }
}

fn is_valid_sample_rate(sample_rate: u32) -> bool {
    matches!(sample_rate, 8_000 | 12_000 | 16_000 | 24_000 | 48_000)
}

fn frame_size(large: bool) -> u32 {
    if large { 5760 } else { 1920 }
}

/// The samples of one decoded frame at `sample_rate`, before alignment.
fn frame_samples(large: bool, channel_count: u32, sample_rate: u32) -> u32 {
    frame_size(large) * channel_count / (48_000 / sample_rate)
}

fn work_buffer_size(params: &OpusParametersEx) -> HleResult<u32> {
    if !matches!(params.channel_count, 1 | 2) {
        return Err(INVALID_OPUS_CHANNEL_COUNT);
    }
    if !is_valid_sample_rate(params.sample_rate) {
        return Err(INVALID_OPUS_SAMPLE_RATE);
    }
    let decoder = DECODE_OBJECT_SIZE + DECODER_SIZE[params.channel_count as usize - 1];
    let output = frame_samples(
        params.use_large_frame_size != 0,
        params.channel_count,
        params.sample_rate,
    )
    .next_multiple_of(64);
    Ok(decoder + output + SINGLE_STREAM_INPUT_SIZE as u32)
}

fn multi_stream_work_buffer_size(params: &OpusMultiStreamParametersEx) -> HleResult<u32> {
    let (channels, total, stereo) = (
        params.channel_count,
        params.total_stream_count,
        params.stereo_stream_count,
    );
    if channels == 0 || channels > OPUS_STREAM_COUNT_MAX {
        return Err(INVALID_OPUS_CHANNEL_COUNT);
    }
    if !is_valid_sample_rate(params.sample_rate) {
        return Err(INVALID_OPUS_SAMPLE_RATE);
    }
    if total == 0 || stereo > total || total + stereo > channels {
        return Err(INVALID_OPUS_SAMPLE_RATE);
    }
    let decoder = DECODE_OBJECT_SIZE
        + MULTI_STREAM_HEADER_SIZE
        + stereo * ALIGNED_STEREO_DECODER_SIZE
        + (total - stereo) * ALIGNED_MONO_DECODER_SIZE;
    let input = (1500 * total).next_multiple_of(64);
    let output = frame_samples(
        params.use_large_frame_size != 0,
        channels,
        params.sample_rate,
    )
    .next_multiple_of(64);
    Ok(decoder + input + output)
}

fn opus_result(err: opus1::Error) -> ResultCode {
    match err.kind() {
        ErrorKind::BadArg => LIB_OPUS_BAD_ARG,
        ErrorKind::BufferTooSmall => BUFFER_TOO_SMALL,
        ErrorKind::InvalidPacket => LIB_OPUS_INVALID_PACKET,
        ErrorKind::Unimplemented => LIB_OPUS_UNIMPLEMENTED,
        ErrorKind::InvalidState => LIB_OPUS_INVALID_STATE,
        ErrorKind::AllocFail => LIB_OPUS_ALLOC_FAIL,
        _ => LIB_OPUS_INTERNAL_ERROR,
    }
}

pub(super) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let open = Arc::new(Mutex::new(0));
    context.register("hwopus", "hwopus", move |context| HwOpus {
        context: context.clone(),
        open: open.clone(),
    })
}

struct HwOpus {
    context: Arc<ServiceContext>,
    open: Arc<Mutex<usize>>,
}

/// One of the DSP's decoder slots, freed when the decoder closes.
struct Slot(Arc<Mutex<usize>>);

impl Slot {
    fn take(open: &Arc<Mutex<usize>>) -> HleResult<Self> {
        let mut count = open.lock();
        if *count == MAX_DECODERS {
            return Err(OUT_OF_OPUS_DECODERS);
        }
        *count += 1;
        Ok(Self(open.clone()))
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        *self.0.lock() -= 1;
    }
}

/// The large parameter struct passed in an X (pointer) buffer.
fn read_pointer_params<T: Pod>(ctx: &HleRequestContext) -> HleResult<T> {
    let buffer = ctx
        .readable_buffer(0)
        .ok_or(lichii_kernel::result::INVALID_POINTER)?;
    let mut bytes = vec![0; size_of::<T>()];
    let len = bytes.len().min(buffer.size as usize);
    ctx.memory()
        .read_bytes(buffer.address, &mut bytes[..len])
        .map_err(|_| lichii_kernel::result::INVALID_CURRENT_MEMORY)?;
    Ok(bytemuck::pod_read_unaligned(&bytes))
}

impl Service for HwOpus {
    fn name(&self) -> &str {
        "hwopus"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            0 => {
                let params: OpusParameters = ctx.pop();
                self.open_single(ctx, params.into())
            }
            1 => {
                let params: OpusParameters = ctx.pop();
                work_buffer_size(&params.into()).map(|size| ctx.push(size))
            }
            2 => read_pointer_params::<OpusMultiStreamParameters>(ctx)
                .and_then(|params| self.open_multi(ctx, params.into())),
            3 => read_pointer_params::<OpusMultiStreamParameters>(ctx)
                .and_then(|params| multi_stream_work_buffer_size(&params.into()))
                .map(|size| ctx.push(size)),
            4 => {
                let params: OpusParametersEx = ctx.pop();
                self.open_single(ctx, params)
            }
            5 | 8 => {
                let params: OpusParametersEx = ctx.pop();
                work_buffer_size(&params).map(|size| ctx.push(size))
            }
            6 => read_pointer_params::<OpusMultiStreamParametersEx>(ctx)
                .and_then(|params| self.open_multi(ctx, params)),
            7 | 9 => read_pointer_params::<OpusMultiStreamParametersEx>(ctx)
                .and_then(|params| multi_stream_work_buffer_size(&params))
                .map(|size| ctx.push(size)),
            _ => return None,
        };
        Some(result.into())
    }
}

impl HwOpus {
    /// `OpenHardwareOpusDecoder` and `OpenHardwareOpusDecoderEx`.
    fn open_single(&self, ctx: &mut HleRequestContext, params: OpusParametersEx) -> HleResult<()> {
        let tmem_size: u32 = ctx.pop();
        tracing::debug!(
            sample_rate = params.sample_rate,
            channels = params.channel_count,
            tmem_size,
            "opening an opus decoder"
        );
        let slot = Slot::take(&self.open)?;
        let sample_rate = SampleRate::try_from(params.sample_rate).map_err(opus_result)?;
        let channels = match params.channel_count {
            1 => Channels::Mono,
            2 => Channels::Stereo,
            _ => return Err(LIB_OPUS_BAD_ARG),
        };
        let decoder = OpusDecoder::new(sample_rate, channels).map_err(opus_result)?;
        let state = DecoderState::new(
            Decoder::Single(Box::new(decoder)),
            params.channel_count,
            params.sample_rate,
            params.use_large_frame_size != 0,
            SINGLE_STREAM_INPUT_SIZE,
        );
        ctx.push_interface(self.context.handler(IHardwareOpusDecoder {
            state: Mutex::new(state),
            _slot: slot,
        }));
        Ok(())
    }

    /// `OpenHardwareOpusDecoderForMultiStream` and its `Ex` variant.
    fn open_multi(
        &self,
        ctx: &mut HleRequestContext,
        params: OpusMultiStreamParametersEx,
    ) -> HleResult<()> {
        let tmem_size: u32 = ctx.pop();
        tracing::debug!(
            sample_rate = params.sample_rate,
            channels = params.channel_count,
            streams = params.total_stream_count,
            stereo_streams = params.stereo_stream_count,
            tmem_size,
            "opening a multistream opus decoder"
        );
        let slot = Slot::take(&self.open)?;
        let sample_rate = SampleRate::try_from(params.sample_rate).map_err(opus_result)?;
        let channels = params.channel_count as usize;
        if channels == 0 || channels > params.mappings.len() {
            return Err(LIB_OPUS_BAD_ARG);
        }
        let decoder = MultistreamDecoder::new(
            sample_rate,
            channels,
            params.total_stream_count as usize,
            params.stereo_stream_count as usize,
            &params.mappings[..channels],
        )
        .map_err(opus_result)?;
        let input_size = (1500 * params.total_stream_count as usize).next_multiple_of(64);
        let state = DecoderState::new(
            Decoder::Multi(Box::new(decoder)),
            params.channel_count,
            params.sample_rate,
            params.use_large_frame_size != 0,
            input_size,
        );
        ctx.push_interface(self.context.handler(IHardwareOpusDecoder {
            state: Mutex::new(state),
            _slot: slot,
        }));
        Ok(())
    }
}

enum Decoder {
    Single(Box<OpusDecoder>),
    Multi(Box<MultistreamDecoder>),
}

struct DecoderState {
    decoder: Decoder,
    channels: usize,
    /// The most frames one call decodes: the size of the work buffer's output area, which Eden
    /// passes to libopus as the frame count.
    max_frames: usize,
    input_size: usize,
    output: Vec<i16>,
}

impl DecoderState {
    fn new(
        decoder: Decoder,
        channel_count: u32,
        sample_rate: u32,
        large_frames: bool,
        input_size: usize,
    ) -> Self {
        let max_frames =
            frame_samples(large_frames, channel_count, sample_rate).next_multiple_of(16) as usize;
        let channels = channel_count as usize;
        Self {
            decoder,
            channels,
            max_frames,
            input_size,
            output: vec![0; max_frames * channels],
        }
    }

    /// Decodes one packet from `input` (an 8-byte big-endian size and final range header, then
    /// the packet) into at most `max_frames` frames. Returns the bytes consumed and the frames
    /// decoded into `self.output`.
    fn decode(
        &mut self,
        input: &[u8],
        max_frames: usize,
        multi_stream: bool,
        reset: bool,
    ) -> HleResult<(u32, usize)> {
        if input.len() <= 8 {
            return Err(INPUT_DATA_TOO_SMALL);
        }
        let size = u32::from_be_bytes([input[0], input[1], input[2], input[3]]) as usize;
        if self.input_size < size || size + 8 > input.len() {
            return Err(BUFFER_TOO_SMALL);
        }
        let packet = &input[8..8 + size];
        let max_frames = max_frames.min(self.max_frames);
        if max_frames == 0 {
            return Err(BUFFER_TOO_SMALL);
        }
        let frames = match (&mut self.decoder, multi_stream) {
            (Decoder::Single(decoder), false) => {
                if reset {
                    decoder.reset_state();
                }
                decoder.decode_i16(packet, max_frames, &mut self.output)
            }
            (Decoder::Multi(decoder), true) => {
                if reset {
                    decoder.reset_state();
                }
                decoder.decode_i16(packet, max_frames, &mut self.output)
            }
            _ => return Err(LIB_OPUS_INVALID_STATE),
        }
        .map_err(opus_result)?;
        Ok((size as u32 + 8, frames))
    }
}

struct IHardwareOpusDecoder {
    state: Mutex<DecoderState>,
    _slot: Slot,
}

impl Service for IHardwareOpusDecoder {
    fn name(&self) -> &str {
        "IHardwareOpusDecoder"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let (multi_stream, perf, reset) = match ctx.command() {
            0 => (false, false, false),
            2 => (true, false, false),
            4 => (false, true, false),
            5 => (true, true, false),
            6 | 8 => (false, true, ctx.pop::<u8>() != 0),
            7 | 9 => (true, true, ctx.pop::<u8>() != 0),
            1 | 3 => return Some(ResultCode::SUCCESS),
            _ => return None,
        };
        Some(self.decode(ctx, multi_stream, perf, reset).into())
    }
}

impl IHardwareOpusDecoder {
    /// Every `DecodeInterleaved` variant: out `u32` bytes consumed, `u32` frames decoded and,
    /// with `perf`, `u64` microseconds taken. The packet must fit the receive buffer, else the
    /// call fails with `BUFFER_TOO_SMALL` and decodes nothing.
    fn decode(
        &self,
        ctx: &mut HleRequestContext,
        multi_stream: bool,
        perf: bool,
        reset: bool,
    ) -> HleResult<()> {
        let input = ctx.read_buffer(0)?;
        let output_size = ctx
            .writable_buffer(0)
            .map_or(0, |buffer| buffer.size as usize);
        let mut state = self.state.lock();
        let max_frames = output_size / (size_of::<i16>() * state.channels);
        let started = Instant::now();
        let (consumed, frames) = state.decode(&input, max_frames, multi_stream, reset)?;
        let taken = started.elapsed().as_micros() as u64;
        let samples = &state.output[..frames * state.channels];
        ctx.write_buffer(0, bytemuck::cast_slice(samples))?;
        ctx.push(consumed);
        ctx.push(frames as u32);
        if perf {
            ctx.push(taken);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_buffer_sizes_match_eden() {
        let stereo = OpusParametersEx {
            sample_rate: 48_000,
            channel_count: 2,
            ..Zeroable::zeroed()
        };
        assert_eq!(work_buffer_size(&stereo), Ok(24 + 27236 + 3840 + 0x600));
        let mono_16k = OpusParametersEx {
            sample_rate: 16_000,
            channel_count: 1,
            use_large_frame_size: 1,
            ..Zeroable::zeroed()
        };
        assert_eq!(work_buffer_size(&mono_16k), Ok(24 + 18468 + 1920 + 0x600));
        let bad_rate = OpusParametersEx {
            sample_rate: 44_100,
            ..stereo
        };
        assert_eq!(work_buffer_size(&bad_rate), Err(INVALID_OPUS_SAMPLE_RATE));
        let bad_channels = OpusParametersEx {
            channel_count: 3,
            ..stereo
        };
        assert_eq!(
            work_buffer_size(&bad_channels),
            Err(INVALID_OPUS_CHANNEL_COUNT)
        );
    }

    #[test]
    fn multi_stream_work_buffer_sizes_match_eden() {
        let surround = OpusMultiStreamParametersEx {
            sample_rate: 48_000,
            channel_count: 6,
            total_stream_count: 4,
            stereo_stream_count: 2,
            ..Zeroable::zeroed()
        };
        let decoders = 24 + 272 + 2 * 27240 + 2 * 18472;
        assert_eq!(
            multi_stream_work_buffer_size(&surround),
            Ok(decoders + 6016 + 11520)
        );
        let too_many_streams = OpusMultiStreamParametersEx {
            total_stream_count: 5,
            ..surround
        };
        assert_eq!(
            multi_stream_work_buffer_size(&too_many_streams),
            Err(INVALID_OPUS_SAMPLE_RATE)
        );
    }
}
