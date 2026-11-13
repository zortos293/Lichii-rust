//! A software decoder for one compressed stream, on FFmpeg.
//!
//! The types here are what a later host GPU decoder (d3d11va, Vulkan video) would also produce:
//! [`Frame`] holds a decoded picture and [`Picture`] is a backend-neutral view of its planes,
//! so callers never see FFmpeg.

use std::ffi::{CStr, c_int, c_void};
use std::ptr;

use crate::ffmpeg::{
    self, AVBufferPool, AVCodecContext, AVERROR_EAGAIN, AVERROR_EOF, AVFrame, AVPacket, Api,
    PADDING,
};

/// A compressed video format the video engines decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264 (AVC), Annex B byte stream.
    H264,
    Vp8,
    Vp9,
}

impl Codec {
    /// The codec a Switch 2 `nn::codec` MIME type names (`"video/vp9"`, `"video/avc"`, ...).
    pub fn from_mime(mime: &str) -> Option<Self> {
        let mime = mime.to_ascii_lowercase();
        if mime.contains("vp9") {
            Some(Self::Vp9)
        } else if mime.contains("vp8") {
            Some(Self::Vp8)
        } else if mime.contains("avc") || mime.contains("264") {
            Some(Self::H264)
        } else {
            None
        }
    }

    fn decoder_name(self) -> &'static CStr {
        match self {
            Self::H264 => c"h264",
            Self::Vp8 => c"vp8",
            Self::Vp9 => c"vp9",
        }
    }
}

/// When a decoded picture comes out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Latency {
    /// Pictures come out as soon as the stream's reordering allows, with no frame-thread delay:
    /// a VP8 or VP9 picture leaves the packet that completes it (NVDEC, whose guest reads the
    /// picture once the engine reports the job done). Slice and tile threads only.
    Immediate,
    /// Frames come out in display order, some packets later, decoded on frame threads as well
    /// (a movie player that queues input ahead).
    Buffered,
}

/// Why a decoder call failed.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum VideoError {
    /// No supported FFmpeg is installed; callers show black frames.
    #[error("FFmpeg is not available")]
    Unavailable,
    /// The installed FFmpeg was built without this codec's decoder.
    #[error("FFmpeg has no {0:?} decoder")]
    NoDecoder(Codec),
    /// [`Decoder::send`] while decoded pictures wait; take them with [`Decoder::receive`] first.
    #[error("the decoder has output waiting; receive it before sending more")]
    Full,
    /// An FFmpeg call failed with this AVERROR code: corrupt input, or out of memory.
    #[error("FFmpeg error {0}")]
    Ffmpeg(i32),
}

fn check(code: c_int) -> Result<c_int, VideoError> {
    if code < 0 {
        Err(VideoError::Ffmpeg(code))
    } else {
        Ok(code)
    }
}

/// Packet buffers start at this size and grow to the next power of two that holds a packet.
const MIN_PACKET_BUFFER: usize = 64 * 1024;

/// One open decoder. Packets are copied into pooled buffers, so steady-state decoding allocates
/// nothing on this side.
pub struct Decoder {
    api: &'static Api,
    codec: Codec,
    context: *mut AVCodecContext,
    packet: *mut AVPacket,
    pool: *mut AVBufferPool,
    pool_size: usize,
}

// SAFETY: the decoder's FFmpeg objects are owned by it alone and only used through `&mut self`.
unsafe impl Send for Decoder {}

impl Decoder {
    /// Opens a decoder for `codec`. Fails with [`VideoError::Unavailable`] when FFmpeg is not
    /// installed (logged once per process) and [`VideoError::NoDecoder`] when it lacks the codec.
    pub fn new(codec: Codec, latency: Latency) -> Result<Self, VideoError> {
        let api = ffmpeg::api().ok_or(VideoError::Unavailable)?;
        // SAFETY: plain FFmpeg allocation calls; every pointer is checked before use and owned by
        // the returned decoder, whose Drop frees it.
        unsafe {
            let found = (api.avcodec_find_decoder_by_name)(codec.decoder_name().as_ptr());
            if found.is_null() {
                return Err(VideoError::NoDecoder(codec));
            }
            let decoder = Self {
                api,
                codec,
                context: (api.avcodec_alloc_context3)(found),
                packet: (api.av_packet_alloc)(),
                pool: ptr::null_mut(),
                pool_size: 0,
            };
            if decoder.context.is_null() || decoder.packet.is_null() {
                return Err(VideoError::Ffmpeg(-12));
            }
            let context = decoder.context.cast::<c_void>();
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(16));
            (api.av_opt_set_int)(context, c"threads".as_ptr(), threads as i64, 0);
            let thread_type = match latency {
                Latency::Immediate => c"slice",
                Latency::Buffered => c"frame+slice",
            };
            (api.av_opt_set)(context, c"thread_type".as_ptr(), thread_type.as_ptr(), 0);
            if latency == Latency::Immediate {
                (api.av_opt_set)(context, c"flags".as_ptr(), c"+low_delay".as_ptr(), 0);
            }
            check((api.avcodec_open2)(decoder.context, found, ptr::null_mut()))?;
            Ok(decoder)
        }
    }

    /// The codec this decoder was opened for.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Queues one compressed frame. A corrupt frame fails here or decodes damaged, and the decoder
    /// goes on with the next one.
    pub fn send(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        let api = self.api;
        let needed = data.len() + PADDING;
        // SAFETY: the pool hands out buffers of `pool_size >= needed` bytes, written only within
        // that size; the packet takes over the buffer reference and FFmpeg keeps its own.
        unsafe {
            if needed > self.pool_size {
                (api.av_buffer_pool_uninit)(&mut self.pool);
                self.pool_size = needed.next_power_of_two().max(MIN_PACKET_BUFFER);
                self.pool = (api.av_buffer_pool_init)(self.pool_size, ptr::null());
                if self.pool.is_null() {
                    self.pool_size = 0;
                    return Err(VideoError::Ffmpeg(-12));
                }
            }
            let buffer = (api.av_buffer_pool_get)(self.pool);
            if buffer.is_null() {
                return Err(VideoError::Ffmpeg(-12));
            }
            let bytes = (*buffer).data;
            ptr::copy_nonoverlapping(data.as_ptr(), bytes, data.len());
            ptr::write_bytes(bytes.add(data.len()), 0, PADDING);
            let packet = &mut *self.packet;
            packet.buf = buffer;
            packet.data = bytes;
            packet.size = data.len() as c_int;
            packet.pts = pts;
            let result = (api.avcodec_send_packet)(self.context, self.packet);
            (api.av_packet_unref)(self.packet);
            match result {
                AVERROR_EAGAIN => Err(VideoError::Full),
                code => check(code).map(drop),
            }
        }
    }

    /// Ends the stream: the pictures the decoder still holds come out of [`Self::receive`], and
    /// the decoder then starts over.
    pub fn end_of_stream(&mut self) {
        // SAFETY: a null packet is FFmpeg's drain request.
        unsafe { (self.api.avcodec_send_packet)(self.context, ptr::null()) };
    }

    /// Takes the next decoded picture into `frame`, replacing what it held. Returns false when
    /// the decoder needs more input (or finished draining, after which it takes a new stream).
    pub fn receive(&mut self, frame: &mut Frame) -> Result<bool, VideoError> {
        let api = self.api;
        // SAFETY: both objects are live FFmpeg allocations owned by their wrappers.
        unsafe {
            match (api.avcodec_receive_frame)(self.context, frame.raw) {
                AVERROR_EAGAIN => Ok(false),
                AVERROR_EOF => {
                    (api.avcodec_flush_buffers)(self.context);
                    Ok(false)
                }
                code => {
                    check(code)?;
                    frame.color = self.color();
                    Ok(true)
                }
            }
        }
    }

    /// Drops every queued frame, for a seek.
    pub fn flush(&mut self) {
        // SAFETY: the context is open.
        unsafe { (self.api.avcodec_flush_buffers)(self.context) };
    }

    /// The stream's colour signalling, which the decoder keeps on its context.
    fn color(&self) -> Color {
        let get = |name: &CStr| {
            let mut value = 0i64;
            // SAFETY: an AVOptions read of a named integer field of the codec context.
            unsafe {
                (self.api.av_opt_get_int)(self.context.cast(), name.as_ptr(), 0, &mut value);
            }
            value
        };
        let matrix = match get(c"colorspace") {
            1 => Some(Matrix::Bt709),
            5 | 6 => Some(Matrix::Bt601),
            9 | 10 => Some(Matrix::Bt2020),
            _ => None,
        };
        Color {
            matrix,
            full_range: get(c"color_range") == 2,
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: each pointer is owned here; the free functions accept null.
        unsafe {
            (self.api.avcodec_free_context)(&mut self.context);
            (self.api.av_packet_free)(&mut self.packet);
            (self.api.av_buffer_pool_uninit)(&mut self.pool);
        }
    }
}

/// The YUV to RGB matrix a stream signals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020,
}

/// Colour signalling of a decoded picture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Color {
    /// `None` when the stream does not say.
    pub matrix: Option<Matrix>,
    /// Samples use the full 0-255 range rather than 16-235 (luma) and 16-240 (chroma).
    pub full_range: bool,
}

/// Storage for one decoded picture. Reuse frames: receiving into a frame recycles its buffers.
pub struct Frame {
    api: &'static Api,
    raw: *mut AVFrame,
    color: Color,
}

// SAFETY: the frame's buffers are reference counted by FFmpeg and owned by this frame alone.
unsafe impl Send for Frame {}

impl Frame {
    /// An empty frame. Fails with [`VideoError::Unavailable`] when FFmpeg is not installed.
    pub fn new() -> Result<Self, VideoError> {
        let api = ffmpeg::api().ok_or(VideoError::Unavailable)?;
        // SAFETY: a plain allocation, checked.
        let raw = unsafe { (api.av_frame_alloc)() };
        if raw.is_null() {
            return Err(VideoError::Ffmpeg(-12));
        }
        Ok(Self {
            api,
            raw,
            color: Color::default(),
        })
    }

    /// Releases the picture, keeping the frame for reuse.
    pub fn clear(&mut self) {
        // SAFETY: the frame is a live allocation.
        unsafe { (self.api.av_frame_unref)(self.raw) };
    }

    /// The picture this frame holds, or `None` when it is empty or in a pixel format the converters
    /// do not read.
    pub fn picture(&self) -> Option<Picture<'_>> {
        // SAFETY: the frame is a live allocation; a decoded frame's planes hold `linesize` bytes per
        // row for each of their rows, and stay valid while the frame is borrowed.
        unsafe {
            let raw = &*self.raw;
            if raw.data[0].is_null() || raw.width <= 0 || raw.height <= 0 {
                return None;
            }
            let formats = &self.api.formats;
            let (format, plane_count) = match raw.format {
                f if f == formats.yuv420p || f == formats.yuvj420p => (PixelFormat::Yuv420, 3),
                f if f == formats.nv12 => (PixelFormat::Nv12, 2),
                f if f == formats.yuv420p10le => (PixelFormat::Yuv420P10, 3),
                _ => return None,
            };
            let (width, height) = (raw.width as u32, raw.height as u32);
            let mut planes = [Plane::EMPTY; 3];
            for (index, plane) in planes.iter_mut().enumerate().take(plane_count) {
                let stride = usize::try_from(raw.linesize[index]).ok()?;
                let rows = if index == 0 {
                    height
                } else {
                    height.div_ceil(2)
                } as usize;
                if raw.data[index].is_null() || stride == 0 {
                    return None;
                }
                *plane = Plane {
                    data: std::slice::from_raw_parts(raw.data[index], stride * rows),
                    stride,
                };
            }
            let mut color = self.color;
            color.full_range |= raw.format == formats.yuvj420p;
            Some(Picture {
                width,
                height,
                pts: raw.pts,
                format,
                planes,
                color,
            })
        }
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        // SAFETY: the frame is owned here.
        unsafe { (self.api.av_frame_free)(&mut self.raw) };
    }
}

/// The plane layouts a [`Picture`] comes in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit Y, U and V planes, chroma at half width and height.
    Yuv420,
    /// 8-bit Y plane and an interleaved UV plane at half height.
    Nv12,
    /// Like [`Self::Yuv420`] with 10-bit samples in little-endian u16s.
    Yuv420P10,
}

/// One plane of a picture: rows `stride` bytes apart.
#[derive(Clone, Copy, Debug)]
pub struct Plane<'a> {
    /// `stride` bytes for each of the plane's rows.
    pub data: &'a [u8],
    /// Bytes from one row to the next.
    pub stride: usize,
}

impl Plane<'_> {
    const EMPTY: Self = Plane {
        data: &[],
        stride: 0,
    };

    /// Row `y`, `stride` bytes including padding past the picture's width. Panics when `y` is
    /// at or past the plane's height.
    pub fn row(&self, y: usize) -> &[u8] {
        &self.data[y * self.stride..(y + 1) * self.stride]
    }
}

/// A decoded picture.
#[derive(Clone, Copy, Debug)]
pub struct Picture<'a> {
    /// Displayed width in pixels.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
    /// The pts the frame was sent with.
    pub pts: i64,
    pub format: PixelFormat,
    /// Y, then U and V or the interleaved UV plane; unused planes are empty.
    pub planes: [Plane<'a>; 3],
    pub color: Color,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mime_types_name_their_codecs() {
        assert_eq!(Codec::from_mime("video/vp9"), Some(Codec::Vp9));
        assert_eq!(Codec::from_mime("video/x-vnd.on2.vp8"), Some(Codec::Vp8));
        assert_eq!(Codec::from_mime("video/avc"), Some(Codec::H264));
        assert_eq!(Codec::from_mime("video/H264"), Some(Codec::H264));
        assert_eq!(Codec::from_mime("video/hevc"), None);
    }
}
