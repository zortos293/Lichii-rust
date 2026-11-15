//! Graphics debugging tools, all off by default and free when off: RenderDoc frame captures
//! ([`renderdoc`]), Nsight Aftermath GPU crash dumps and Vulkan debug names and labels.
//! [`init`] sets them up once from the settings, before the first [`crate::Instance`].

pub(crate) mod aftermath;
mod labels;
mod library;
pub mod renderdoc;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use lichii_common::settings::DebugSettings;

pub(crate) use labels::{Program, Programs};

static LABELS: AtomicBool = AtomicBool::new(false);

/// Sets up the tools `settings` turns on: loads RenderDoc (also when
/// `LICHII_RENDERDOC_CAPTURE_FRAME` asks for captures), enables Aftermath crash dumps into a new
/// folder under `crash_dumps` and asks the next instance and devices for debug labels.
///
/// # Safety
///
/// Call it once, before the first Vulkan instance and while no other thread runs: loading
/// RenderDoc from a folder adds its layer to `VK_ADD_IMPLICIT_LAYER_PATH` in this process's environment.
pub unsafe fn init(settings: &DebugSettings, crash_dumps: &Path) {
    LABELS.store(settings.vulkan_labels, Ordering::Relaxed);
    // SAFETY: the caller guarantees no other thread runs.
    unsafe { renderdoc::init(settings.renderdoc_hotkey) };
    if settings.nsight_aftermath {
        aftermath::init(crash_dumps);
    }
}

pub(crate) fn labels_requested() -> bool {
    LABELS.load(Ordering::Relaxed)
}
