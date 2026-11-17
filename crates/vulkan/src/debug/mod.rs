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
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use ash::vk;

use lichii_common::settings::DebugSettings;

pub(crate) use labels::{Program, Programs};

use crate::error::VkResultExt;

static LABELS: AtomicBool = AtomicBool::new(false);
static FAULT_REPORT: AtomicBool = AtomicBool::new(false);
static DEVICE: AtomicU8 = AtomicU8::new(HEALTHY);

const HEALTHY: u8 = 0;
const LOST: u8 = 1;
const REPORTED: u8 = 2;

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

/// Marks the device lost, so the renderer records, submits and reads back nothing more, then
/// logs the loss once, reports the fault and, with Aftermath on, waits for its crash dump.
pub(crate) fn lose_device() {
    if DEVICE
        .compare_exchange(HEALTHY, LOST, Ordering::AcqRel, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    tracing::error!("the Vulkan device was lost");
    fault::report();
    if let Some(aftermath) = aftermath::get() {
        aftermath.wait_for_dump();
    }
    DEVICE.store(REPORTED, Ordering::Release);
}

/// Whether a Vulkan call returned `VK_ERROR_DEVICE_LOST`, even while its fault report is being
/// written.
pub(crate) fn device_is_lost() -> bool {
    DEVICE.load(Ordering::Relaxed) != HEALTHY
}

/// Whether the Vulkan device was lost and its fault report and crash dump are written. The
/// renderer has stopped recording, submitting and reading back by then; the caller ends the
/// session, since nothing it draws or reads from the GPU is valid any more.
pub fn device_lost() -> bool {
    DEVICE.load(Ordering::Acquire) == REPORTED
}

/// Takes the path a `VK_ERROR_DEVICE_LOST` result from any Vulkan call takes, for tests.
#[doc(hidden)]
pub fn inject_device_lost() {
    let _ = Err::<(), _>(vk::Result::ERROR_DEVICE_LOST).call("injected device loss");
}
