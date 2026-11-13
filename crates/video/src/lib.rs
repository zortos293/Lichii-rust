//! Video decoding for the Switch's video engines: FFmpeg (libavcodec, loaded at runtime) behind a
//! small decoder API, the NVDEC codec state builders, and the pixel conversions the guests' output
//! images need.

pub mod nvdec;
