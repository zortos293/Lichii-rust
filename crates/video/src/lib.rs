//! Video decoding for the Switch's video engines: FFmpeg (libavcodec) behind a small decoder API,
//! the NVDEC codec state builders, and the pixel conversions guests' output images need.
//!
//! The Linux release links a decoder-only FFmpeg statically (`LICHII_FFMPEG_STATIC_DIR`, see
//! `build.rs`). Other builds load it at runtime, where it is optional: without it
//! [`Decoder::new`] fails with [`VideoError::Unavailable`] after one warning, and callers show
//! black frames.

// The runtime loader is these dependencies' only user, and a statically linked build leaves it out.
// Cargo cannot see build.rs's cfg to make them conditional, so mark them used here instead.
#[cfg(ffmpeg_static)]
use {libloading as _, lichii_common as _};

pub mod convert;
mod decoder;
mod ffmpeg;
pub mod nvdec;

pub use decoder::{
    Codec, Color, Decoder, Frame, Latency, Matrix, Picture, PixelFormat, Plane, VideoError,
};

/// Whether FFmpeg is linked in or a supported one is installed. The first call loads it, or logs
/// that it is missing.
pub fn available() -> bool {
    ffmpeg::api().is_some()
}
