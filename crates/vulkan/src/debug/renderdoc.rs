//! RenderDoc's in-application API. Loaded only when `debug.renderdoc_hotkey` is on or
//! `LICHII_RENDERDOC_CAPTURE_FRAME` lists guest frames to capture: from the running process
//! when RenderDoc launched Lichii, else from `LICHII_RENDERDOC_DIR`, the executable's folder or
//! RenderDoc's install folder. Captures span whole guest frames: one starts after a guest present
//! and ends after a later one, so it holds every command that drew the frames between.
//! `LICHII_RENDERDOC_CAPTURE_DIR` names the folder capture files go to.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use libloading::Library;

use super::library;

#[cfg(windows)]
const LIBRARY: &str = "renderdoc.dll";
#[cfg(not(windows))]
const LIBRARY: &str = "librenderdoc.so";
const LAYER_MANIFESTS: [&str; 2] = ["renderdoc.json", "renderdoc_capture.json"];
const API_VERSION_1_6_0: c_int = 10600;

type GetApi = unsafe extern "C" fn(c_int, *mut *mut c_void) -> c_int;
type Unused = Option<unsafe extern "C" fn()>;

/// The start of `RENDERDOC_API_1_6_0`, up to the last entry used.
#[repr(C)]
struct Api {
    get_api_version: unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int),
    _capture_options: [Unused; 4],
    _keys: [Unused; 2],
    _overlay: [Unused; 2],
    _remove_hooks: Unused,
    _unload_crash_handler: Unused,
    set_capture_file_path_template: unsafe extern "C" fn(*const c_char),
    _get_capture_file_path_template: Unused,
    get_num_captures: unsafe extern "C" fn() -> u32,
    get_capture: unsafe extern "C" fn(u32, *mut c_char, *mut u32, *mut u64) -> u32,
    _trigger_capture: Unused,
    _is_target_control_connected: Unused,
    _launch_replay_ui: Unused,
    _set_active_window: Unused,
    start_frame_capture: unsafe extern "C" fn(*mut c_void, *mut c_void),
    _is_frame_capturing: Unused,
    end_frame_capture: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32,
}

/// The loaded RenderDoc.
pub struct RenderDoc {
    api: &'static Api,
    /// Guest frames to capture, 1-based, sorted.
    frames: Vec<u64>,
    toggle: AtomicBool,
    /// The renderer whose capture is running, 0 for none: a renderer still tearing down never
    /// ends the capture of a newer one.
    owner: AtomicU64,
    _library: Library,
}

static RENDERDOC: OnceLock<RenderDoc> = OnceLock::new();

/// RenderDoc, when it is loaded.
pub fn get() -> Option<&'static RenderDoc> {
    RENDERDOC.get()
}

/// Starts a capture at the next guest present, or ends the running one there. False when
/// RenderDoc is not loaded.
pub fn request_toggle() -> bool {
    get().is_some_and(|renderdoc| {
        renderdoc.toggle.store(true, Ordering::Relaxed);
        true
    })
}

/// # Safety
///
/// No other thread may run: see [`super::init`].
pub(super) unsafe fn init(hotkey: bool) {
    let frames = std::env::var("LICHII_RENDERDOC_CAPTURE_FRAME")
        .map(|text| parse_frames(&text))
        .unwrap_or_default();
    if !hotkey && frames.is_empty() {
        return;
    }
    // SAFETY: forwarded from the caller.
    match unsafe { load(frames) } {
        Ok(renderdoc) => {
            let _ = RENDERDOC.set(renderdoc);
        }
        Err(error) => tracing::warn!("RenderDoc: {error}"),
    }
}

/// # Safety
///
/// As [`init`].
unsafe fn load(frames: Vec<u64>) -> Result<RenderDoc, String> {
    let (library, source) = match injected() {
        Some(library) => (library, "attached to this process".to_string()),
        None => {
            let program_files = cfg!(windows)
                .then(|| std::env::var_os("ProgramFiles"))
                .flatten()
                .map(|dir| PathBuf::from(dir).join("RenderDoc"));
            let dirs = library::search_dirs("LICHII_RENDERDOC_DIR", program_files);
            let (library, dir) = library::open(LIBRARY, &dirs).ok_or_else(|| {
                let dirs = dirs
                    .iter()
                    .map(|dir| dir.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{LIBRARY} is in none of {dirs} nor on the library path; put RenderDoc there \
                     or name its folder in LICHII_RENDERDOC_DIR"
                )
            })?;
            if let Some(dir) = &dir {
                // SAFETY: forwarded from the caller.
                unsafe { add_layer_path(dir) };
            }
            let source = dir.map_or_else(
                || "from the library path".to_string(),
                |dir| format!("from {}", dir.display()),
            );
            (library, source)
        }
    };
    // SAFETY: every RenderDoc release exports RENDERDOC_GetAPI with this signature.
    let get_api = unsafe { library.get::<GetApi>(b"RENDERDOC_GetAPI\0") }
        .map_err(|error| format!("{LIBRARY} has no RENDERDOC_GetAPI: {error}"))?;
    let mut table = std::ptr::null_mut();
    // SAFETY: RENDERDOC_GetAPI stores a pointer to its API table of the version asked for.
    if unsafe { get_api(API_VERSION_1_6_0, &mut table) } != 1 || table.is_null() {
        return Err(format!("{LIBRARY} does not offer API 1.6"));
    }
    // SAFETY: the table is RenderDoc's static RENDERDOC_API_1_6_0, which starts like `Api` and
    // lives as long as the library, which is never unloaded.
    let api = unsafe { &*(table as *const Api) };
    let (mut major, mut minor, mut patch) = (0, 0, 0);
    // SAFETY: plain out-parameters.
    unsafe { (api.get_api_version)(&mut major, &mut minor, &mut patch) };
    let capture_dir = std::env::var_os("LICHII_RENDERDOC_CAPTURE_DIR").map(PathBuf::from);
    if let Some(dir) = &capture_dir {
        set_capture_dir(api, dir)?;
    }
    tracing::info!(
        ?frames,
        ?capture_dir,
        "RenderDoc API {major}.{minor}.{patch} loaded {source}"
    );
    Ok(RenderDoc {
        api,
        frames,
        toggle: AtomicBool::new(false),
        owner: AtomicU64::new(0),
        _library: library,
    })
}

#[cfg(windows)]
fn injected() -> Option<Library> {
    libloading::os::windows::Library::open_already_loaded(LIBRARY)
        .ok()
        .map(Library::from)
}

#[cfg(not(windows))]
fn injected() -> Option<Library> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_NOW};
    const RTLD_NOLOAD: std::ffi::c_int = 4;
    // SAFETY: RTLD_NOLOAD only finds a library that is already loaded; nothing initialises.
    unsafe { UnixLibrary::open(Some(LIBRARY), RTLD_NOW | RTLD_NOLOAD) }
        .ok()
        .map(Library::from)
}

/// Lets the Vulkan loader find RenderDoc's layer in `dir`, which holds its manifest when
/// RenderDoc was unpacked rather than installed. The layer only works as the implicit layer its
/// manifest describes, which `ENABLE_VULKAN_RENDERDOC_CAPTURE` turns on: enabled by name it fails
/// instance creation.
///
/// # Safety
///
/// No other thread may run.
unsafe fn add_layer_path(dir: &Path) {
    if !LAYER_MANIFESTS.iter().any(|name| dir.join(name).is_file()) {
        return;
    }
    let existing = std::env::var_os("VK_ADD_IMPLICIT_LAYER_PATH");
    let paths =
        std::iter::once(dir.to_path_buf()).chain(existing.iter().flat_map(std::env::split_paths));
    let Ok(joined) = std::env::join_paths(paths) else {
        return;
    };
    // SAFETY: the caller guarantees no other thread reads the environment.
    unsafe {
        std::env::set_var("VK_ADD_IMPLICIT_LAYER_PATH", joined);
        std::env::set_var("ENABLE_VULKAN_RENDERDOC_CAPTURE", "1");
    }
}

fn set_capture_dir(api: &Api, dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir)
        .map_err(|error| format!("capture folder {} unusable: {error}", dir.display()))?;
    let template = CString::new(dir.join("lichii").to_string_lossy().into_owned())
        .map_err(|_| format!("capture folder {} has a NUL in its name", dir.display()))?;
    // SAFETY: RenderDoc copies the NUL-terminated template.
    unsafe { (api.set_capture_file_path_template)(template.as_ptr()) };
    Ok(())
}

fn parse_frames(text: &str) -> Vec<u64> {
    let mut frames: Vec<u64> = text
        .split(',')
        .filter_map(|frame| frame.trim().parse().ok())
        .filter(|&frame| frame > 0)
        .collect();
    frames.sort_unstable();
    frames.dedup();
    frames
}

impl RenderDoc {
    fn start(&self, owner: u64) -> bool {
        if self
            .owner
            .compare_exchange(0, owner, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            tracing::warn!(
                "RenderDoc: a stopping renderer still holds a capture; skipped this one"
            );
            return false;
        }
        // SAFETY: null device and window capture RenderDoc's active ones.
        unsafe { (self.api.start_frame_capture)(std::ptr::null_mut(), std::ptr::null_mut()) };
        true
    }

    fn end(&self, owner: u64, frames: (u64, u64)) {
        if self
            .owner
            .compare_exchange(owner, 0, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        // SAFETY: as in `start`; a capture is running.
        let captured =
            unsafe { (self.api.end_frame_capture)(std::ptr::null_mut(), std::ptr::null_mut()) };
        if captured != 1 {
            tracing::warn!(?frames, "RenderDoc: the capture of guest frames failed");
            return;
        }
        // SAFETY: plain queries; the second call fills a buffer of the length the first gave.
        let path = unsafe {
            let index = (self.api.get_num_captures)().saturating_sub(1);
            let mut length = 0;
            (self.api.get_capture)(
                index,
                std::ptr::null_mut(),
                &mut length,
                std::ptr::null_mut(),
            );
            let mut path = vec![0u8; length as usize + 1];
            (self.api.get_capture)(
                index,
                path.as_mut_ptr().cast(),
                &mut length,
                std::ptr::null_mut(),
            );
            CStr::from_bytes_until_nul(&path)
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        tracing::info!(
            "RenderDoc: captured guest frames {}-{} to {path}",
            frames.0,
            frames.1
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trigger {
    /// A frame `LICHII_RENDERDOC_CAPTURE_FRAME` lists: ends at the next present.
    Listed,
    /// Ends when the hotkey is pressed again.
    Hotkey,
}

/// What a guest present does to the capture.
#[derive(Debug, PartialEq, Eq)]
struct Step {
    end: bool,
    start: Option<Trigger>,
}

fn step(running: Option<Trigger>, toggled: bool, next_listed: bool) -> Step {
    let end = match running {
        Some(Trigger::Listed) => true,
        Some(Trigger::Hotkey) => toggled,
        None => false,
    };
    let start = match running {
        Some(Trigger::Hotkey) => None,
        _ if toggled => Some(Trigger::Hotkey),
        _ if next_listed => Some(Trigger::Listed),
        _ => None,
    };
    Step { end, start }
}

/// The renderer's side of captures: counts guest presents and starts and ends captures there.
pub(crate) struct FrameCapture {
    renderdoc: &'static RenderDoc,
    id: u64,
    /// Guest frames presented so far.
    frame: u64,
    running: Option<(Trigger, u64)>,
}

impl FrameCapture {
    pub(crate) fn new(renderdoc: &'static RenderDoc) -> Self {
        static RENDERERS: AtomicU64 = AtomicU64::new(0);
        let mut capture = Self {
            renderdoc,
            id: RENDERERS.fetch_add(1, Ordering::Relaxed) + 1,
            frame: 0,
            running: None,
        };
        if renderdoc.frames.first() == Some(&1) && renderdoc.start(capture.id) {
            capture.running = Some((Trigger::Listed, 1));
        }
        capture
    }

    /// Call after each guest present; `flush` submits the commands recorded so far, so an
    /// ending capture holds them.
    pub(crate) fn presented(&mut self, flush: impl FnOnce()) {
        self.frame += 1;
        let toggled = self.renderdoc.toggle.swap(false, Ordering::Relaxed);
        let next = self.frame + 1;
        let next_listed = self.renderdoc.frames.binary_search(&next).is_ok();
        let step = step(
            self.running.map(|(trigger, _)| trigger),
            toggled,
            next_listed,
        );
        if step.end
            && let Some((_, first)) = self.running.take()
        {
            flush();
            self.renderdoc.end(self.id, (first, self.frame));
        }
        if let Some(trigger) = step.start
            && self.renderdoc.start(self.id)
        {
            self.running = Some((trigger, next));
        }
    }

    /// Ends a running capture when the renderer stops, once its work finished, so the capture
    /// file is written and the next session starts with none running.
    pub(crate) fn finish(&mut self) {
        if let Some((_, first)) = self.running.take() {
            self.renderdoc.end(self.id, (first, self.frame.max(first)));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_lists_parse_sorted_without_zero_or_junk() {
        assert_eq!(parse_frames("300"), [300]);
        assert_eq!(parse_frames(" 450, 300,x,0,300 "), [300, 450]);
        assert!(parse_frames("").is_empty());
    }

    #[test]
    fn listed_frames_end_at_the_next_present() {
        let idle = step(None, false, false);
        assert_eq!(
            idle,
            Step {
                end: false,
                start: None
            }
        );
        assert_eq!(step(None, false, true).start, Some(Trigger::Listed));
        let listed = step(Some(Trigger::Listed), false, false);
        assert_eq!(
            listed,
            Step {
                end: true,
                start: None
            }
        );
        let back_to_back = step(Some(Trigger::Listed), false, true);
        assert_eq!(
            back_to_back,
            Step {
                end: true,
                start: Some(Trigger::Listed)
            }
        );
    }

    #[test]
    fn the_hotkey_runs_a_capture_until_pressed_again() {
        assert_eq!(step(None, true, false).start, Some(Trigger::Hotkey));
        let running = step(Some(Trigger::Hotkey), false, true);
        assert_eq!(
            running,
            Step {
                end: false,
                start: None
            }
        );
        let stopped = step(Some(Trigger::Hotkey), true, false);
        assert_eq!(
            stopped,
            Step {
                end: true,
                start: None
            }
        );
        let after_listed = step(Some(Trigger::Listed), true, false);
        assert_eq!(
            after_listed,
            Step {
                end: true,
                start: Some(Trigger::Hotkey)
            }
        );
    }
}
