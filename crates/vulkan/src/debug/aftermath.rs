//! Nsight Aftermath GPU crash dumps (NVIDIA), ported from the reference's
//! `nsight_aftermath_tracker.cpp`. With `debug.nsight_aftermath` on, `GFSDK_Aftermath_Lib` is
//! loaded from `LICHII_AFTERMATH_DIR`, the executable's folder or the library path and crash
//! dumps are enabled before the device exists; the device then turns on
//! `VK_NV_device_diagnostics_config` and `VK_NV_device_diagnostic_checkpoints`. When the device
//! is lost, the dump (`.nv-gpudmp`, plus a decoded `.json`), shader debug info (`.nvdbg`) and
//! the SPIR-V of every shader built (`.spv`) are in a new folder under the crash dump folder
//! (`LICHII_AFTERMATH_DUMP_DIR` overrides it).

use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use libloading::Library;

use lichii_shader::ShaderModule;

use super::Program;
use super::library;

#[cfg(windows)]
const LIBRARY: &str = "GFSDK_Aftermath_Lib.x64.dll";
#[cfg(not(windows))]
const LIBRARY: &str = "libGFSDK_Aftermath_Lib.x64.so";

/// `GFSDK_Aftermath_Version_API` of SDK 2026.3 (API 2.27).
const VERSION_API: u32 = 0x21b;
const WATCHED_API_VULKAN: u32 = 0x2;
const FEATURE_FLAGS_DEFAULT: u32 = 0x0;
const DESCRIPTION_APPLICATION_NAME: u32 = 0x1;
const DECODER_ALL_INFO: u32 = 0x3fff;
const FORMATTER_NONE: u32 = 0x0;
const STATUS_COLLECTING_DATA_FAILED: u32 = 2;
const STATUS_FINISHED: u32 = 4;
/// How long a lost device waits for Aftermath to write its dump.
const DUMP_WAIT: Duration = Duration::from_secs(15);

type AddDescription = unsafe extern "C" fn(u32, *const c_char);
type CrashDumpCallback = unsafe extern "C" fn(*const c_void, u32, *mut c_void);
type DescriptionCallback = unsafe extern "C" fn(AddDescription, *mut c_void);
type Decoder = *mut c_void;

#[repr(C)]
struct SpirvCode {
    data: *const c_void,
    size: u32,
}

/// Aftermath's result codes are failures when their top 12 bits are `0xBAD`.
fn succeeded(result: u32) -> bool {
    result & 0xfff0_0000 != 0xbad0_0000
}

macro_rules! entry_points {
    ($($name:ident: fn($($arg:ty),*);)*) => {
        #[allow(non_snake_case)]
        struct Api {
            $($name: unsafe extern "C" fn($($arg),*) -> u32,)*
        }

        impl Api {
            fn resolve(library: &Library) -> Result<Self, String> {
                Ok(Self {
                    $($name: {
                        let symbol = concat!("GFSDK_Aftermath_", stringify!($name), "\0");
                        // SAFETY: the type is the entry point's signature in the SDK headers.
                        *unsafe { library.get(symbol.as_bytes()) }.map_err(|error| {
                            format!("{LIBRARY} lacks {}: {error}", symbol.trim_end_matches('\0'))
                        })?
                    },)*
                })
            }
        }
    };
}

entry_points! {
    EnableGpuCrashDumps: fn(
        u32, u32, u32,
        CrashDumpCallback, CrashDumpCallback, DescriptionCallback,
        Option<unsafe extern "C" fn()>, *mut c_void
    );
    GetCrashDumpStatus: fn(*mut u32);
    GetShaderDebugInfoIdentifier: fn(u32, *const c_void, u32, *mut [u64; 2]);
    GetShaderHashSpirv: fn(u32, *const SpirvCode, *mut u64);
    GpuCrashDump_CreateDecoder: fn(u32, *const c_void, u32, *mut Decoder);
    GpuCrashDump_DestroyDecoder: fn(Decoder);
    GpuCrashDump_GenerateJSON: fn(
        Decoder, u32, u32,
        Option<unsafe extern "C" fn()>, Option<unsafe extern "C" fn()>,
        Option<unsafe extern "C" fn()>, *mut c_void, *mut u32
    );
    GpuCrashDump_GetJSON: fn(Decoder, u32, *mut c_char);
}

pub(crate) struct Aftermath {
    api: Api,
    dir: PathBuf,
    enabled: AtomicBool,
    dumps: AtomicU32,
    _library: Library,
}

static AFTERMATH: OnceLock<Aftermath> = OnceLock::new();

/// Aftermath, when crash dumps are enabled.
pub(crate) fn get() -> Option<&'static Aftermath> {
    AFTERMATH
        .get()
        .filter(|aftermath| aftermath.enabled.load(Ordering::Relaxed))
}

pub(super) fn init(crash_dumps: &Path) {
    let root = std::env::var_os("LICHII_AFTERMATH_DUMP_DIR")
        .map_or_else(|| crash_dumps.to_path_buf(), PathBuf::from);
    let aftermath = match load(&root) {
        Ok(aftermath) => AFTERMATH.get_or_init(|| aftermath),
        Err(error) => {
            tracing::error!("Nsight Aftermath: {error}");
            return;
        }
    };
    // SAFETY: the callbacks match Aftermath's signatures and their user data is the static
    // tracker, which outlives every callback.
    let result = unsafe {
        (aftermath.api.EnableGpuCrashDumps)(
            VERSION_API,
            WATCHED_API_VULKAN,
            FEATURE_FLAGS_DEFAULT,
            crash_dump_callback,
            shader_debug_info_callback,
            description_callback,
            None,
            std::ptr::from_ref(aftermath).cast_mut().cast(),
        )
    };
    if !succeeded(result) {
        tracing::error!("Nsight Aftermath: enabling GPU crash dumps failed ({result:#x})");
        return;
    }
    aftermath.enabled.store(true, Ordering::Relaxed);
    tracing::info!(dir = ?aftermath.dir, "Nsight Aftermath GPU crash dumps enabled");
}

fn load(root: &Path) -> Result<Aftermath, String> {
    let dirs = library::search_dirs("LICHII_AFTERMATH_DIR", None);
    let (library, _) = library::open(LIBRARY, &dirs).ok_or_else(|| {
        format!("{LIBRARY} not found next to the executable or on the library path; put it there or name its folder in LICHII_AFTERMATH_DIR")
    })?;
    let api = Api::resolve(&library)?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let dir = root.join(format!("{stamp}-{}", std::process::id()));
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("creating {}: {error}", dir.display()))?;
    Ok(Aftermath {
        api,
        dir,
        enabled: AtomicBool::new(false),
        dumps: AtomicU32::new(0),
        _library: library,
    })
}

impl Aftermath {
    /// Keeps `program`'s SPIR-V beside the dumps, named by Aftermath's hash, so a dump can
    /// show the shader that faulted, and logs which program the hash stands for.
    pub(crate) fn save_shader(&self, program: &ShaderModule) {
        let spirv = &program.spirv[..];
        let code = SpirvCode {
            data: spirv.as_ptr().cast(),
            size: size_of_val(spirv) as u32,
        };
        let mut hash = 0;
        // SAFETY: `code` points to `spirv`, which lives across the call.
        let result = unsafe { (self.api.GetShaderHashSpirv)(VERSION_API, &code, &mut hash) };
        if !succeeded(result) {
            tracing::warn!("Nsight Aftermath: hashing a SPIR-V module failed ({result:#x})");
            return;
        }
        let path = self.dir.join(format!("source_{hash:016x}.spv"));
        if path.exists() {
            return;
        }
        match std::fs::write(&path, bytemuck::cast_slice::<u32, u8>(spirv)) {
            Ok(()) => tracing::info!(
                program = %Program(program),
                aftermath = format_args!("{hash:016x}"),
                "Nsight Aftermath: shader saved"
            ),
            Err(error) => tracing::warn!(?path, %error, "Nsight Aftermath: saving a shader failed"),
        }
    }

    fn on_crash_dump(&self, dump: &[u8]) {
        let id = self.dumps.fetch_add(1, Ordering::Relaxed);
        let path = match id {
            0 => self.dir.join("crash.nv-gpudmp"),
            id => self.dir.join(format!("crash_{id}.nv-gpudmp")),
        };
        tracing::error!(?path, "Nsight Aftermath: writing the GPU crash dump");
        if let Err(error) = std::fs::write(&path, dump) {
            tracing::error!(?path, %error, "Nsight Aftermath: writing the dump failed");
            return;
        }
        match self.decode(dump) {
            Ok(json) => {
                if let Err(error) = std::fs::write(path.with_extension("nv-gpudmp.json"), json) {
                    tracing::error!(%error, "Nsight Aftermath: writing the decoded dump failed");
                }
            }
            Err(error) => tracing::error!("Nsight Aftermath: decoding the dump failed: {error}"),
        }
    }

    fn decode(&self, dump: &[u8]) -> Result<Vec<u8>, String> {
        let mut decoder = std::ptr::null_mut();
        // SAFETY: `dump` is the buffer Aftermath handed to the callback.
        let result = unsafe {
            (self.api.GpuCrashDump_CreateDecoder)(
                VERSION_API,
                dump.as_ptr().cast(),
                dump.len() as u32,
                &mut decoder,
            )
        };
        if !succeeded(result) {
            return Err(format!("creating the decoder ({result:#x})"));
        }
        let mut size = 0;
        // SAFETY: a live decoder and plain out-parameters; no lookup callbacks.
        let result = unsafe {
            (self.api.GpuCrashDump_GenerateJSON)(
                decoder,
                DECODER_ALL_INFO,
                FORMATTER_NONE,
                None,
                None,
                None,
                std::ptr::null_mut(),
                &mut size,
            )
        };
        let mut json = vec![0u8; size as usize];
        let result = if succeeded(result) {
            // SAFETY: `json` holds the `size` bytes GenerateJSON reported.
            unsafe { (self.api.GpuCrashDump_GetJSON)(decoder, size, json.as_mut_ptr().cast()) }
        } else {
            result
        };
        // SAFETY: the decoder is live and not used after this.
        unsafe { (self.api.GpuCrashDump_DestroyDecoder)(decoder) };
        if !succeeded(result) {
            return Err(format!("generating JSON ({result:#x})"));
        }
        if json.last() == Some(&0) {
            json.pop();
        }
        Ok(json)
    }

    fn on_shader_debug_info(&self, info: &[u8]) {
        let mut id = [0u64; 2];
        // SAFETY: `info` is the buffer Aftermath handed to the callback.
        let result = unsafe {
            (self.api.GetShaderDebugInfoIdentifier)(
                VERSION_API,
                info.as_ptr().cast(),
                info.len() as u32,
                &mut id,
            )
        };
        if !succeeded(result) {
            tracing::error!("Nsight Aftermath: reading a shader debug info id failed");
            return;
        }
        let path = self
            .dir
            .join(format!("shader_{:016x}{:016x}.nvdbg", id[0], id[1]));
        if let Err(error) = std::fs::write(&path, info) {
            tracing::error!(?path, %error, "Nsight Aftermath: writing shader debug info failed");
        }
    }

    /// Waits until Aftermath finished the dump of a lost device, at most [`DUMP_WAIT`].
    pub(super) fn wait_for_dump(&self) {
        let deadline = Instant::now() + DUMP_WAIT;
        let mut status = 0;
        while Instant::now() < deadline {
            // SAFETY: a plain out-parameter.
            let result = unsafe { (self.api.GetCrashDumpStatus)(&mut status) };
            if !succeeded(result)
                || status == STATUS_FINISHED
                || status == STATUS_COLLECTING_DATA_FAILED
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        tracing::error!(
            dir = ?self.dir,
            status,
            "Nsight Aftermath: crash dump collection ended"
        );
    }
}

/// # Safety
///
/// Aftermath calls it with a dump of `size` bytes and the tracker as `user`.
unsafe extern "C" fn crash_dump_callback(dump: *const c_void, size: u32, user: *mut c_void) {
    // SAFETY: as documented.
    let (aftermath, dump) = unsafe {
        (
            &*(user as *const Aftermath),
            std::slice::from_raw_parts(dump.cast::<u8>(), size as usize),
        )
    };
    aftermath.on_crash_dump(dump);
}

/// # Safety
///
/// Aftermath calls it with debug info of `size` bytes and the tracker as `user`.
unsafe extern "C" fn shader_debug_info_callback(info: *const c_void, size: u32, user: *mut c_void) {
    // SAFETY: as documented.
    let (aftermath, info) = unsafe {
        (
            &*(user as *const Aftermath),
            std::slice::from_raw_parts(info.cast::<u8>(), size as usize),
        )
    };
    aftermath.on_shader_debug_info(info);
}

/// # Safety
///
/// Aftermath calls it with its own `add` function.
unsafe extern "C" fn description_callback(add: AddDescription, _user: *mut c_void) {
    const NAME: &CStr = c"Lichii";
    // SAFETY: `add` copies the NUL-terminated value.
    unsafe { add(DESCRIPTION_APPLICATION_NAME, NAME.as_ptr()) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failures_are_the_0xbad_results() {
        assert!(succeeded(0x1));
        assert!(succeeded(0x2));
        assert!(!succeeded(0xbad0_0000));
        assert!(!succeeded(0xbad0_0001));
    }
}
