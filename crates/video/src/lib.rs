//! Video decoding for the Switch's video engines: FFmpeg (libavcodec, loaded at runtime) behind a
//! small decoder API, the NVDEC codec state builders, and the pixel conversions guests' output
//! images need.
//!
//! FFmpeg is optional at runtime: without it [`Decoder::new`] fails with
//! [`VideoError::Unavailable`] after one warning, and callers show black frames.

pub mod convert;
mod decoder;
mod ffmpeg;
pub mod nvdec;

pub use decoder::{
    Codec, Color, Decoder, Frame, Latency, Matrix, Picture, PixelFormat, Plane, VideoError,
};

/// Whether a supported FFmpeg is installed. The first call loads it, or logs that it is missing.
pub fn available() -> bool {
    ffmpeg::api().is_some()
}
