//! A decoder for one compressed stream: on the host GPU's video engine when the renderer
//! registered one for the codec ([`set_gpu_video`]), else in software on FFmpeg.
//!
//! [`Frame`] holds a decoded picture and [`Picture`] is a backend-neutral view of its planes,
//! so callers never see which decoder made it.

use std::ffi::{CStr, c_int, c_void};
use std::ptr;
use std::sync::{Arc, RwLock};

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
    /// AV1, as temporal units of OBUs. Decoded on the host GPU only.
    Av1,
}

impl Codec {
    /// The codec a Switch 2 `nn::codec` MIME type names (`"video/vp9"`, `"video/avc"`, ...).
    pub fn from_mime(mime: &str) -> Option<Self> {
        let mime = mime.to_ascii_lowercase();
        if mime.contains("vp9") {
            Some(Self::Vp9)
        } else if mime.contains("av1") || mime.contains("av01") {
            Some(Self::Av1)
        } else if mime.contains("vp8") {
            Some(Self::Vp8)
        } else if mime.contains("avc") || mime.contains("264") {
            Some(Self::H264)
        } else {
            None
        }
    }

    /// The FFmpeg decoder, `None` for codecs only the host GPU decodes.
    fn decoder_name(self) -> Option<&'static CStr> {
        match self {
            Self::H264 => Some(c"h264"),
            Self::Vp8 => Some(c"vp8"),
            Self::Vp9 => Some(c"vp9"),
            Self::Av1 => None,
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
    /// The host GPU decoder failed: a Vulkan error code, or 0 for input it cannot decode.
    #[error("GPU video decoder error {0}")]
    Gpu(i32),
}

/// A picture the host GPU decoded, brought to CPU memory as NV12: `width` x `height` luma
/// bytes, then interleaved UV rows of `width` bytes for `height.div_ceil(2)` rows.
#[derive(Clone, Debug, Default)]
pub struct OwnedPicture {
    pub width: u32,
    pub height: u32,
    pub pts: i64,
    pub data: Vec<u8>,
    pub color: Color,
}

impl OwnedPicture {
    /// The stride of both planes: the width rounded up to whole chroma samples.
    pub fn stride(&self) -> usize {
        self.width.next_multiple_of(2) as usize
    }

    /// Bytes the NV12 picture takes.
    pub fn size(width: u32, height: u32) -> usize {
        let stride = width.next_multiple_of(2) as usize;
        stride * height as usize + stride * height.div_ceil(2) as usize
    }
}

/// One stream decoding on the host GPU, with the [`Decoder`] calls.
pub trait GpuStream: Send {
    fn send(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError>;
    /// The stream ends: pictures still held come out of [`GpuStream::receive`].
    fn end_of_stream(&mut self);
    /// The next picture in display order into `out`; false when none is ready.
    fn receive(&mut self, out: &mut OwnedPicture) -> Result<bool, VideoError>;
    /// Drops queued input and held pictures, for a seek.
    fn flush(&mut self);
}

/// The host GPU's video decoder, as the renderer registers it.
pub trait GpuVideo: Send + Sync {
    /// A stream for `codec`, or `None` when the GPU does not decode it.
    fn open(&self, codec: Codec) -> Option<Box<dyn GpuStream>>;
}

static GPU_VIDEO: RwLock<Option<Arc<dyn GpuVideo>>> = RwLock::new(None);

/// Registers the host GPU's video decoder (or none): decoders opened from now on use it for the
/// codecs it decodes.
pub fn set_gpu_video(video: Option<Arc<dyn GpuVideo>>) {
    *GPU_VIDEO.write().unwrap_or_else(|e| e.into_inner()) = video;
}

fn gpu_stream(codec: Codec) -> Option<Box<dyn GpuStream>> {
    let video = GPU_VIDEO
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()?;
    video.open(codec)
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

/// One open decoder, on the host GPU or on FFmpeg.
pub struct Decoder {
    codec: Codec,
    inner: Inner,
}

enum Inner {
    Gpu(Box<dyn GpuStream>),
    Ffmpeg(FfmpegDecoder),
}

impl Decoder {
    /// Opens a decoder for `codec`: the host GPU's when it decodes the codec, else FFmpeg's.
    /// Fails with [`VideoError::Unavailable`] when FFmpeg is needed but not installed (logged
    /// once per process) and [`VideoError::NoDecoder`] when no decoder takes the codec.
    pub fn new(codec: Codec, latency: Latency) -> Result<Self, VideoError> {
        let inner = match gpu_stream(codec) {
            Some(stream) => Inner::Gpu(stream),
            None => Inner::Ffmpeg(FfmpegDecoder::new(codec, latency)?),
        };
        Ok(Self { codec, inner })
    }

    /// The codec this decoder was opened for.
    pub fn codec(&self) -> Codec {
        self.codec
    }

    /// Whether the host GPU decodes this stream.
    pub fn on_gpu(&self) -> bool {
        matches!(self.inner, Inner::Gpu(_))
    }

    /// Queues one compressed frame. A corrupt frame fails here or decodes damaged, and the decoder
    /// goes on with the next one.
    pub fn send(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        match &mut self.inner {
            Inner::Gpu(stream) => stream.send(data, pts),
            Inner::Ffmpeg(decoder) => decoder.send(data, pts),
        }
    }

    /// Ends the stream: the pictures the decoder still holds come out of [`Self::receive`], and
    /// the decoder then starts over.
    pub fn end_of_stream(&mut self) {
        match &mut self.inner {
            Inner::Gpu(stream) => stream.end_of_stream(),
            Inner::Ffmpeg(decoder) => decoder.end_of_stream(),
        }
    }

    /// Takes the next decoded picture into `frame`, replacing what it held. Returns false when
    /// the decoder needs more input (or finished draining, after which it takes a new stream).
    pub fn receive(&mut self, frame: &mut Frame) -> Result<bool, VideoError> {
        match &mut self.inner {
            Inner::Gpu(stream) => {
                frame.clear();
                let got = stream.receive(&mut frame.owned)?;
                if got {
                    frame.source = Source::Owned;
                    frame.color = frame.owned.color;
                }
                Ok(got)
            }
            Inner::Ffmpeg(decoder) => decoder.receive(frame),
        }
    }

    /// Drops every queued frame, for a seek.
    pub fn flush(&mut self) {
        match &mut self.inner {
            Inner::Gpu(stream) => stream.flush(),
            Inner::Ffmpeg(decoder) => decoder.flush(),
        }
    }
}

/// FFmpeg's decoder. Packets are copied into pooled buffers, so steady-state decoding allocates
/// nothing on this side.
struct FfmpegDecoder {
    api: &'static Api,
    context: *mut AVCodecContext,
    packet: *mut AVPacket,
    pool: *mut AVBufferPool,
    pool_size: usize,
}

// SAFETY: the decoder's FFmpeg objects are owned by it alone and only used through `&mut self`.
unsafe impl Send for FfmpegDecoder {}

impl FfmpegDecoder {
    fn new(codec: Codec, latency: Latency) -> Result<Self, VideoError> {
        let name = codec.decoder_name().ok_or(VideoError::NoDecoder(codec))?;
        let api = ffmpeg::api().ok_or(VideoError::Unavailable)?;
        // SAFETY: plain FFmpeg allocation calls; every pointer is checked before use and owned by
        // the returned decoder, whose Drop frees it.
        unsafe {
            let found = (api.avcodec_find_decoder_by_name)(name.as_ptr());
            if found.is_null() {
                return Err(VideoError::NoDecoder(codec));
            }
            let decoder = Self {
                api,
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

    fn send(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
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

    fn end_of_stream(&mut self) {
        // SAFETY: a null packet is FFmpeg's drain request.
        unsafe { (self.api.avcodec_send_packet)(self.context, ptr::null()) };
    }

    fn receive(&mut self, frame: &mut Frame) -> Result<bool, VideoError> {
        let api = self.api;
        frame.source = Source::Empty;
        let raw = match &frame.ff {
            Some(ff) => ff.raw,
            None => frame.ff.insert(FfFrame::new(api)?).raw,
        };
        // SAFETY: both objects are live FFmpeg allocations owned by their wrappers.
        unsafe {
            match (api.avcodec_receive_frame)(self.context, raw) {
                AVERROR_EAGAIN => Ok(false),
                AVERROR_EOF => {
                    (api.avcodec_flush_buffers)(self.context);
                    Ok(false)
                }
                code => {
                    check(code)?;
                    frame.color = self.color();
                    frame.source = Source::Ffmpeg;
                    Ok(true)
                }
            }
        }
    }

    fn flush(&mut self) {
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

impl Drop for FfmpegDecoder {
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

/// FFmpeg's frame inside a [`Frame`], allocated on the first FFmpeg picture.
struct FfFrame {
    api: &'static Api,
    raw: *mut AVFrame,
}

impl FfFrame {
    fn new(api: &'static Api) -> Result<Self, VideoError> {
        // SAFETY: a plain allocation, checked.
        let raw = unsafe { (api.av_frame_alloc)() };
        if raw.is_null() {
            return Err(VideoError::Ffmpeg(-12));
        }
        Ok(Self { api, raw })
    }
}

impl Drop for FfFrame {
    fn drop(&mut self) {
        // SAFETY: the frame is owned here.
        unsafe { (self.api.av_frame_free)(&mut self.raw) };
    }
}

/// Which decoder filled a [`Frame`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Empty,
    Ffmpeg,
    Owned,
}

/// Storage for one decoded picture. Reuse frames: receiving into a frame recycles its buffers.
pub struct Frame {
    ff: Option<FfFrame>,
    owned: OwnedPicture,
    source: Source,
    color: Color,
}

// SAFETY: the frame's buffers are reference counted by FFmpeg and owned by this frame alone.
unsafe impl Send for Frame {}
// SAFETY: a shared frame is only read; every call that changes it takes `&mut self`.
unsafe impl Sync for Frame {}

impl Frame {
    /// An empty frame. It takes FFmpeg's buffers only once FFmpeg decodes into it.
    pub fn new() -> Result<Self, VideoError> {
        Ok(Self {
            ff: None,
            owned: OwnedPicture::default(),
            source: Source::Empty,
            color: Color::default(),
        })
    }

    /// Releases the picture, keeping the frame for reuse.
    pub fn clear(&mut self) {
        if let Some(ff) = &self.ff {
            // SAFETY: the frame is a live allocation.
            unsafe { (ff.api.av_frame_unref)(ff.raw) };
        }
        self.source = Source::Empty;
    }

    /// The picture this frame holds, or `None` when it is empty or in a pixel format the converters
    /// do not read.
    pub fn picture(&self) -> Option<Picture<'_>> {
        match self.source {
            Source::Empty => None,
            Source::Owned => self.owned_picture(),
            Source::Ffmpeg => self.ffmpeg_picture(),
        }
    }

    fn owned_picture(&self) -> Option<Picture<'_>> {
        let o = &self.owned;
        let stride = o.stride();
        let luma = stride * o.height as usize;
        if o.width == 0 || o.height == 0 || o.data.len() < OwnedPicture::size(o.width, o.height) {
            return None;
        }
        let (y, uv) = o.data.split_at(luma);
        let mut planes = [Plane::EMPTY; 3];
        planes[0] = Plane { data: y, stride };
        planes[1] = Plane {
            data: &uv[..stride * o.height.div_ceil(2) as usize],
            stride,
        };
        Some(Picture {
            width: o.width,
            height: o.height,
            pts: o.pts,
            format: PixelFormat::Nv12,
            planes,
            color: self.color,
        })
    }

    fn ffmpeg_picture(&self) -> Option<Picture<'_>> {
        let ff = self.ff.as_ref()?;
        // SAFETY: the frame is a live allocation; a decoded frame's planes hold `linesize` bytes per
        // row for each of their rows, and stay valid while the frame is borrowed.
        unsafe {
            let raw = &*ff.raw;
            if raw.data[0].is_null() || raw.width <= 0 || raw.height <= 0 {
                return None;
            }
            let formats = &ff.api.formats;
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
        assert_eq!(Codec::from_mime("video/av1"), Some(Codec::Av1));
        assert_eq!(Codec::from_mime("video/AV01"), Some(Codec::Av1));
    }

    struct FakeStream {
        sent: Vec<i64>,
    }

    impl GpuStream for FakeStream {
        fn send(&mut self, _data: &[u8], pts: i64) -> Result<(), VideoError> {
            self.sent.push(pts);
            Ok(())
        }

        fn end_of_stream(&mut self) {}

        fn receive(&mut self, out: &mut OwnedPicture) -> Result<bool, VideoError> {
            let Some(pts) = self.sent.pop() else {
                return Ok(false);
            };
            *out = OwnedPicture {
                width: 3,
                height: 3,
                pts,
                data: (0..OwnedPicture::size(3, 3) as u8).collect(),
                color: Color::default(),
            };
            Ok(true)
        }

        fn flush(&mut self) {
            self.sent.clear();
        }
    }

    struct FakeGpu;

    impl GpuVideo for FakeGpu {
        fn open(&self, codec: Codec) -> Option<Box<dyn GpuStream>> {
            (codec == Codec::Av1).then(|| Box::new(FakeStream { sent: Vec::new() }) as _)
        }
    }

    #[test]
    fn av1_decodes_on_the_registered_gpu_into_nv12_pictures() {
        set_gpu_video(None);
        assert_eq!(
            Decoder::new(Codec::Av1, Latency::Immediate).err(),
            Some(VideoError::NoDecoder(Codec::Av1))
        );
        set_gpu_video(Some(Arc::new(FakeGpu)));
        let mut decoder = Decoder::new(Codec::Av1, Latency::Immediate).unwrap();
        assert!(decoder.on_gpu());
        let mut frame = Frame::new().unwrap();
        assert!(frame.picture().is_none());
        decoder.send(&[1, 2, 3], 42).unwrap();
        assert_eq!(decoder.receive(&mut frame), Ok(true));
        let picture = frame.picture().unwrap();
        assert_eq!((picture.width, picture.height, picture.pts), (3, 3, 42));
        assert_eq!(picture.format, PixelFormat::Nv12);
        assert_eq!(picture.planes[0].stride, 4);
        assert_eq!(picture.planes[0].row(2), &[8, 9, 10, 11]);
        assert_eq!(picture.planes[1].row(1), &[16, 17, 18, 19]);
        assert_eq!(decoder.receive(&mut frame), Ok(false));
        assert!(frame.picture().is_none());
        set_gpu_video(None);
    }
}
