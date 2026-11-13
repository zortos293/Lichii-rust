//! libavcodec and libavutil, loaded at runtime.
//!
//! Loading at runtime keeps FFmpeg out of the build: the emulator builds without FFmpeg headers,
//! starts without the libraries, and picks up whichever supported FFmpeg (4.x to 8.x) the system
//! or the release folder holds. Only ABI-stable parts are touched: functions, the leading fields of
//! `AVFrame` and `AVPacket` (unchanged since FFmpeg 3), and `AVCodecContext` through AVOptions,
//! never its layout.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

use libloading::Library;

/// libavcodec majors this binding supports, newest first, each with the libavutil major it ships
/// with.
const MAJORS: [(u32, u32); 5] = [(62, 60), (61, 59), (60, 58), (59, 57), (58, 56)];
/// Zero bytes FFmpeg may read past the end of a packet (`AV_INPUT_BUFFER_PADDING_SIZE`).
pub(crate) const PADDING: usize = 64;
pub(crate) const AVERROR_EAGAIN: c_int = -11;
/// `AVERROR_EOF`: `-MKTAG('E', 'O', 'F', ' ')`.
pub(crate) const AVERROR_EOF: c_int = -0x2046_4f45;
const AV_LOG_QUIET: c_int = -8;
const AV_LOG_ERROR: c_int = 16;

#[repr(C)]
pub(crate) struct AVCodec {
    _opaque: [u8; 0],
}

#[repr(C)]
pub(crate) struct AVCodecContext {
    _opaque: [u8; 0],
}

#[repr(C)]
pub(crate) struct AVBufferPool {
    _opaque: [u8; 0],
}

#[repr(C)]
pub(crate) struct AVBufferRef {
    _buffer: *mut c_void,
    pub data: *mut u8,
    pub size: usize,
}

/// The leading fields of `AVPacket`.
#[repr(C)]
pub(crate) struct AVPacket {
    pub buf: *mut AVBufferRef,
    pub pts: i64,
    pub dts: i64,
    pub data: *mut u8,
    pub size: c_int,
}

/// The leading fields of `AVFrame`. FFmpeg 8 dropped `key_frame` before `pict_type`, but `pts`
/// stays at offset 136 because of its 8-byte alignment.
#[repr(C)]
pub(crate) struct AVFrame {
    pub data: [*mut u8; 8],
    pub linesize: [c_int; 8],
    _extended_data: *mut *mut u8,
    pub width: c_int,
    pub height: c_int,
    _nb_samples: c_int,
    pub format: c_int,
    _key_frame_pict_type: [c_int; 2],
    _sample_aspect_ratio: [c_int; 2],
    pub pts: i64,
}

const _: () = assert!(std::mem::offset_of!(AVFrame, width) == 104);
const _: () = assert!(std::mem::offset_of!(AVFrame, format) == 116);
const _: () = assert!(std::mem::offset_of!(AVFrame, pts) == 136);
const _: () = assert!(std::mem::offset_of!(AVPacket, data) == 24);
const _: () = assert!(std::mem::offset_of!(AVPacket, size) == 32);

/// `AVPixelFormat` values of the formats the converters read, looked up by name because the enum
/// is only append-stable.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PixelFormats {
    pub yuv420p: c_int,
    pub yuvj420p: c_int,
    pub nv12: c_int,
    pub yuv420p10le: c_int,
}

macro_rules! api {
    ($($library:ident { $($name:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)* })*) => {
        /// The FFmpeg functions in use, resolved once.
        pub(crate) struct Api {
            $($(pub $name: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*)*
            pub formats: PixelFormats,
            pub version: (u32, u32),
            _libraries: Vec<Library>,
        }

        impl Api {
            fn resolve($($library: Library),*) -> Result<Self, String> {
                // SAFETY: each symbol is declared with the signature FFmpeg's public headers give
                // it, which every supported major keeps.
                let mut api = unsafe {
                    Self {
                        $($(
                            $name: *$library
                                .get::<unsafe extern "C" fn($($arg),*) $(-> $ret)?>(
                                    concat!(stringify!($name), "\0").as_bytes(),
                                )
                                .map_err(|err| format!("{}: {err}", stringify!($name)))?,
                        )*)*
                        formats: PixelFormats::default(),
                        version: (0, 0),
                        _libraries: vec![$($library),*],
                    }
                };
                // SAFETY: version queries and pixel format lookups by NUL-terminated name.
                unsafe {
                    let format = |name: &CStr| (api.av_get_pix_fmt)(name.as_ptr());
                    api.formats = PixelFormats {
                        yuv420p: format(c"yuv420p"),
                        yuvj420p: format(c"yuvj420p"),
                        nv12: format(c"nv12"),
                        yuv420p10le: format(c"yuv420p10le"),
                    };
                    api.version = ((api.avcodec_version)() >> 16, (api.avutil_version)() >> 16);
                }
                Ok(api)
            }
        }
    };
}

api! {
    avcodec {
        avcodec_version: fn() -> u32;
        avcodec_find_decoder_by_name: fn(*const c_char) -> *const AVCodec;
        avcodec_alloc_context3: fn(*const AVCodec) -> *mut AVCodecContext;
        avcodec_open2: fn(*mut AVCodecContext, *const AVCodec, *mut *mut c_void) -> c_int;
        avcodec_free_context: fn(*mut *mut AVCodecContext);
        avcodec_send_packet: fn(*mut AVCodecContext, *const AVPacket) -> c_int;
        avcodec_receive_frame: fn(*mut AVCodecContext, *mut AVFrame) -> c_int;
        avcodec_flush_buffers: fn(*mut AVCodecContext);
        av_packet_alloc: fn() -> *mut AVPacket;
        av_packet_free: fn(*mut *mut AVPacket);
        av_packet_unref: fn(*mut AVPacket);
    }
    avutil {
        avutil_version: fn() -> u32;
        av_frame_alloc: fn() -> *mut AVFrame;
        av_frame_free: fn(*mut *mut AVFrame);
        av_frame_unref: fn(*mut AVFrame);
        av_buffer_pool_init: fn(usize, *const c_void) -> *mut AVBufferPool;
        av_buffer_pool_get: fn(*mut AVBufferPool) -> *mut AVBufferRef;
        av_buffer_pool_uninit: fn(*mut *mut AVBufferPool);
        av_opt_set: fn(*mut c_void, *const c_char, *const c_char, c_int) -> c_int;
        av_opt_set_int: fn(*mut c_void, *const c_char, i64, c_int) -> c_int;
        av_opt_get_int: fn(*mut c_void, *const c_char, c_int, *mut i64) -> c_int;
        av_get_pix_fmt: fn(*const c_char) -> c_int;
        av_log_set_level: fn(c_int);
    }
}

/// The loaded FFmpeg, or `None` (after one warning) when no supported FFmpeg is installed.
pub(crate) fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(|| match load() {
        Ok(api) => {
            let level = if tracing::enabled!(tracing::Level::DEBUG) {
                AV_LOG_ERROR
            } else {
                AV_LOG_QUIET
            };
            // SAFETY: a plain setter of FFmpeg's global log level.
            unsafe { (api.av_log_set_level)(level) };
            tracing::info!(
                "video: FFmpeg libavcodec {} / libavutil {} loaded",
                api.version.0,
                api.version.1
            );
            Some(api)
        }
        Err(tried) => {
            tracing::warn!(
                "video: FFmpeg not found ({tried}); videos play as black frames. Install the \
                 FFmpeg 4 to 8 shared libraries, or put the avcodec and avutil libraries next to \
                 the executable"
            );
            None
        }
    })
    .as_ref()
}

fn library_name(name: &str, major: u32) -> String {
    if cfg!(windows) {
        format!("{name}-{major}.dll")
    } else if cfg!(target_os = "macos") {
        format!("lib{name}.{major}.dylib")
    } else {
        format!("lib{name}.so.{major}")
    }
}

/// Opens `name` from `LICHII_FFMPEG_DIR` when set, then from next to the executable, then from
/// the system's library search path. avutil loads before avcodec, so avcodec's import of it
/// resolves to the copy already loaded, wherever that came from.
fn open(name: &str) -> Option<Library> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(dir) = std::env::var_os("LICHII_FFMPEG_DIR") {
        candidates.push(PathBuf::from(dir).join(name));
    }
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(PathBuf::from))
    {
        candidates.push(dir.join(name));
    }
    candidates.push(name.into());
    candidates.into_iter().find_map(|path| {
        // SAFETY: loading FFmpeg runs only its library initialisers, which have no preconditions.
        unsafe { Library::new(&path) }.ok()
    })
}

fn load() -> Result<Api, String> {
    let mut tried = Vec::new();
    for (codec_major, util_major) in MAJORS {
        let (codec_name, util_name) = (
            library_name("avcodec", codec_major),
            library_name("avutil", util_major),
        );
        let Some(avutil) = open(&util_name) else {
            tried.push(util_name);
            continue;
        };
        let Some(avcodec) = open(&codec_name) else {
            tried.push(codec_name);
            continue;
        };
        let api = Api::resolve(avcodec, avutil)?;
        if api.version != (codec_major, util_major) {
            return Err(format!(
                "{codec_name} reports libavcodec {} / libavutil {}",
                api.version.0, api.version.1
            ));
        }
        return Ok(api);
    }
    Err(format!("tried {}", tried.join(", ")))
}
