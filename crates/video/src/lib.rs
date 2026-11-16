//! Video decoding for the Switch's video engines: the host GPU's video decoder when the renderer
//! registers one, FFmpeg (libavcodec, loaded at runtime) otherwise, behind a small decoder API;
//! the AV1 header reader the GPU decoder needs, the NVDEC codec state builders, and the pixel
//! conversions guests' output images need.
//!
//! FFmpeg is optional at runtime: without it [`Decoder::new`] fails with
//! [`VideoError::Unavailable`] after one warning, and callers show black frames.

pub mod av1;
pub mod convert;
mod decoder;
mod ffmpeg;
pub mod nvdec;
pub mod vp9;

pub use decoder::{
    Codec, Color, Decoder, Frame, GpuStream, GpuVideo, Latency, Matrix, OwnedPicture, Picture,
    PixelFormat, Plane, VideoError, set_gpu_video,
};

/// Whether a supported FFmpeg is installed. The first call loads it, or logs that it is missing.
pub fn available() -> bool {
    ffmpeg::api().is_some()
}
