//! Graphics debugging tools, all off by default and free when off: RenderDoc frame captures
//! ([`renderdoc`]), Nsight Aftermath GPU crash dumps, GPU fault reports, shader dumps and Vulkan
//! debug names and labels.
//! [`init`] sets them up once from the settings, before the first [`crate::Instance`].

pub(crate) mod aftermath;
pub(crate) mod fault;
mod labels;
mod library;
pub mod renderdoc;
pub(crate) mod shader_dump;

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use lichii_common::settings::DebugSettings;

pub(crate) use labels::{Program, Programs};

static LABELS: AtomicBool = AtomicBool::new(false);
static FAULT_REPORT: AtomicBool = AtomicBool::new(false);
static DEVICE_LOST: AtomicBool = AtomicBool::new(false);

/// Sets up the tools `settings` turns on: loads RenderDoc (also when
/// `LICHII_RENDERDOC_CAPTURE_FRAME` asks for captures), enables Aftermath crash dumps into a new
/// folder under `crash_dumps`, asks the next devices for fault reports (vendor binaries go to
/// `crash_dumps`), dumps translated shaders next to the fault reports and asks the next
/// instance and devices for debug labels.
///
/// # Safety
///
/// Call it once, before the first Vulkan instance and while no other thread runs: loading
/// RenderDoc from a folder adds its layer to `VK_ADD_IMPLICIT_LAYER_PATH` in this process's environment.
pub unsafe fn init(settings: &DebugSettings, crash_dumps: &Path) {
    LABELS.store(settings.vulkan_labels, Ordering::Relaxed);
    FAULT_REPORT.store(settings.gpu_fault_report, Ordering::Relaxed);
    let reports = fault::init(crash_dumps);
    if settings.dump_shaders {
        shader_dump::init(reports.join("shaders"));
    }
    // SAFETY: the caller guarantees no other thread runs.
    unsafe { renderdoc::init(settings.renderdoc_hotkey) };
    if settings.nsight_aftermath {
        aftermath::init(crash_dumps);
    }
}

pub(crate) fn labels_requested() -> bool {
    LABELS.load(Ordering::Relaxed)
}

pub(crate) fn fault_report_requested() -> bool {
    FAULT_REPORT.load(Ordering::Relaxed)
}

/// Logs the loss of the device once, reports the fault and, with Aftermath on, waits for its
/// crash dump.
pub(crate) fn device_lost() {
    if DEVICE_LOST.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("the Vulkan device was lost");
    fault::report();
    if let Some(aftermath) = aftermath::get() {
        aftermath.wait_for_dump();
    }
}
