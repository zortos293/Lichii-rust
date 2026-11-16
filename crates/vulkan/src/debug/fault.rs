//! GPU fault reports, on with `debug.gpu_fault_report`. When the device is lost, logs what
//! `VK_EXT_device_fault` reports (the fault, faulting addresses and vendor records) and writes
//! the vendor's binary dump next to the crash dumps. With `VK_NV_device_diagnostic_checkpoints`
//! every label (and every command buffer's start) also leaves a checkpoint, so the report names
//! the last draw or dispatch each pipeline stage passed and the work still in flight. Labels are
//! kept in a fixed ring of [`SLOTS`] entries, so a checkpoint older than that shows by number only.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use ash::{ext, nv, vk};
use parking_lot::Mutex;

use super::labels::Text;

/// Labels kept for the report, about 13 MiB: a hung GPU lets the host record a few seconds of
/// work (some 60k draws in MKW) past the last checkpoint the GPU passed.
const SLOTS: usize = 1 << 16;
/// Most in-flight labels listed in one report.
const IN_FLIGHT_LISTED: u64 = 48;

static DIR: OnceLock<PathBuf> = OnceLock::new();
static REPORTER: StdMutex<Option<Arc<Reporter>>> = StdMutex::new(None);

pub(super) fn init(crash_dumps: &Path) {
    let _ = DIR.set(crash_dumps.to_path_buf());
}

/// What the device turned on for fault reports.
pub(crate) struct Reporter {
    device: vk::Device,
    queue: vk::Queue,
    fault: Option<ext::device_fault::Device>,
    vendor_binary: bool,
    checkpoints: Option<nv::device_diagnostic_checkpoints::Device>,
    ring: Mutex<Ring>,
}

impl Reporter {
    /// The reporter of a device made with `fault` and `checkpoints` (when they are on); it
    /// reports when any Vulkan call returns `VK_ERROR_DEVICE_LOST` until [`Self::unregister`].
    pub(crate) fn register(
        instance: &ash::Instance,
        device: &ash::Device,
        queue: vk::Queue,
        fault: Option<bool>,
        checkpoints: bool,
    ) -> Arc<Self> {
        let reporter = Arc::new(Self {
            device: device.handle(),
            queue,
            fault: fault.map(|_| ext::device_fault::Device::new(instance, device)),
            vendor_binary: fault == Some(true),
            checkpoints: checkpoints
                .then(|| nv::device_diagnostic_checkpoints::Device::new(instance, device)),
            ring: Mutex::new(Ring::new()),
        });
        *REPORTER.lock().unwrap_or_else(|e| e.into_inner()) = Some(reporter.clone());
        reporter
    }

    pub(crate) fn unregister(self: &Arc<Self>) {
        let mut current = REPORTER.lock().unwrap_or_else(|e| e.into_inner());
        if current.as_ref().is_some_and(|r| Arc::ptr_eq(r, self)) {
            *current = None;
        }
    }

    /// Leaves a checkpoint named `text` in `cmd`.
    pub(super) fn checkpoint(&self, cmd: vk::CommandBuffer, text: Text) {
        let Some(checkpoints) = &self.checkpoints else {
            return;
        };
        let seq = self.ring.lock().push(text);
        // SAFETY: `cmd` is recording; the marker is an integer the driver hands back as is.
        unsafe { checkpoints.cmd_set_checkpoint(cmd, std::ptr::without_provenance(seq as usize)) };
    }

    fn report(&self) {
        if let Some(fault) = &self.fault {
            self.report_fault(fault);
        }
        if let Some(checkpoints) = &self.checkpoints {
            self.report_checkpoints(checkpoints);
        }
    }

    fn report_fault(&self, fault: &ext::device_fault::Device) {
        let get = fault.fp().get_device_fault_info_ext;
        let mut counts = vk::DeviceFaultCountsEXT::default();
        // SAFETY: the device is lost and has VK_EXT_device_fault; a null info asks for counts.
        let result = unsafe { get(self.device, &mut counts, std::ptr::null_mut()) };
        if result != vk::Result::SUCCESS {
            tracing::error!("GPU fault: vkGetDeviceFaultInfoEXT counts failed: {result}");
            return;
        }
        if !self.vendor_binary {
            counts.vendor_binary_size = 0;
        }
        let mut addresses =
            vec![vk::DeviceFaultAddressInfoEXT::default(); counts.address_info_count as usize];
        let mut vendors =
            vec![vk::DeviceFaultVendorInfoEXT::default(); counts.vendor_info_count as usize];
        let mut binary = vec![0u8; counts.vendor_binary_size as usize];
        let mut info = vk::DeviceFaultInfoEXT {
            p_address_infos: addresses.as_mut_ptr(),
            p_vendor_infos: vendors.as_mut_ptr(),
            p_vendor_binary_data: binary.as_mut_ptr().cast::<c_void>(),
            ..Default::default()
        };
        // SAFETY: every array holds as many elements as `counts` says.
        let result = unsafe { get(self.device, &mut counts, &mut info) };
        if result != vk::Result::SUCCESS && result != vk::Result::INCOMPLETE {
            tracing::error!("GPU fault: vkGetDeviceFaultInfoEXT failed: {result}");
            return;
        }
        addresses.truncate(counts.address_info_count as usize);
        vendors.truncate(counts.vendor_info_count as usize);
        binary.truncate(counts.vendor_binary_size as usize);
        tracing::error!(
            description = ?info.description_as_c_str().unwrap_or_default(),
            addresses = addresses.len(),
            vendor_records = vendors.len(),
            vendor_binary_bytes = binary.len(),
            "GPU fault"
        );
        for address in &addresses {
            let (start, end) = address_range(address.reported_address, address.address_precision);
            tracing::error!(
                kind = ?address.address_type,
                reported = format_args!("{:#x}", address.reported_address),
                range = format_args!("{start:#x}..={end:#x}"),
                "GPU fault address"
            );
        }
        for vendor in &vendors {
            tracing::error!(
                description = ?vendor.description_as_c_str().unwrap_or_default(),
                code = format_args!("{:#x}", vendor.vendor_fault_code),
                data = format_args!("{:#x}", vendor.vendor_fault_data),
                "GPU fault vendor record"
            );
        }
        if !binary.is_empty() {
            self.save_binary(&binary);
        }
    }

    fn save_binary(&self, binary: &[u8]) {
        let Some(dir) = DIR.get() else {
            return;
        };
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        let path = dir.join(format!("device-fault-{secs}.bin"));
        match std::fs::create_dir_all(dir).and_then(|()| std::fs::write(&path, binary)) {
            Ok(()) => tracing::error!(?path, "GPU fault: vendor binary written"),
            Err(error) => tracing::error!(
                ?path,
                "GPU fault: writing the vendor binary failed: {error}"
            ),
        }
    }

    fn report_checkpoints(&self, checkpoints: &nv::device_diagnostic_checkpoints::Device) {
        let get = checkpoints.fp().get_queue_checkpoint_data_nv;
        let mut count = 0;
        // SAFETY: the queue belongs to the lost device; a null array asks for the count.
        unsafe { get(self.queue, &mut count, std::ptr::null_mut()) };
        let mut data = vec![vk::CheckpointDataNV::default(); count as usize];
        // SAFETY: `data` holds `count` elements.
        unsafe { get(self.queue, &mut count, data.as_mut_ptr()) };
        data.truncate(count as usize);
        let ring = self.ring.lock();
        let stages: Vec<_> = data
            .iter()
            .map(|d| (d.stage, d.p_checkpoint_marker as usize as u64))
            .collect();
        for &(stage, seq) in &stages {
            tracing::error!(
                ?stage,
                checkpoint = seq,
                label = ring.label(seq).unwrap_or("(gone from the ring)"),
                "GPU fault: last checkpoint this stage passed"
            );
        }
        let Some((first, last)) = in_flight(&stages, ring.last()) else {
            tracing::error!(
                recorded = ring.last(),
                "GPU fault: no checkpoint was reached"
            );
            return;
        };
        tracing::error!(
            first,
            last,
            listed = (last - first + 1).min(IN_FLIGHT_LISTED),
            "GPU fault: work after the oldest stage's checkpoint, which holds the fault"
        );
        for seq in first..=last.min(first + IN_FLIGHT_LISTED - 1) {
            tracing::error!(
                checkpoint = seq,
                label = ring.label(seq).unwrap_or("(gone from the ring)"),
                "GPU fault: in flight"
            );
        }
    }
}

/// Reports the fault of the registered device, if any.
pub(super) fn report() {
    let reporter = REPORTER.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(reporter) = reporter {
        reporter.report();
    }
}

/// The bytes a fault address may stand for: its `precision` (a power of two) aligned block.
fn address_range(reported: u64, precision: u64) -> (u64, u64) {
    let mask = precision.max(1).next_power_of_two() - 1;
    (reported & !mask, reported | mask)
}

/// The checkpoints from the oldest one a stage passed up to the newest one recorded: the work
/// that may not have finished. `None` when no stage passed any.
fn in_flight(stages: &[(vk::PipelineStageFlags, u64)], recorded: u64) -> Option<(u64, u64)> {
    let oldest = stages
        .iter()
        .map(|&(_, seq)| seq)
        .filter(|&seq| seq != 0)
        .min()?;
    Some((oldest, recorded.max(oldest)))
}

struct Slot {
    seq: u64,
    text: Text,
}

/// The last [`SLOTS`] checkpoint labels, by sequence number (from 1).
struct Ring {
    last: u64,
    slots: Box<[Slot]>,
}

impl Ring {
    fn new() -> Self {
        Self {
            last: 0,
            slots: (0..SLOTS)
                .map(|_| Slot {
                    seq: 0,
                    text: Text::EMPTY,
                })
                .collect(),
        }
    }

    fn push(&mut self, text: Text) -> u64 {
        self.last += 1;
        self.slots[self.last as usize % SLOTS] = Slot {
            seq: self.last,
            text,
        };
        self.last
    }

    fn last(&self) -> u64 {
        self.last
    }

    fn label(&self, seq: u64) -> Option<&str> {
        let slot = &self.slots[seq as usize % SLOTS];
        (seq != 0 && slot.seq == seq).then(|| slot.text.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_keeps_the_last_slots_labels() {
        let mut ring = Ring::new();
        let first = ring.push(Text::from(format_args!("draw vs 1 fs 2")));
        assert_eq!(first, 1);
        assert_eq!(ring.label(1), Some("draw vs 1 fs 2"));
        assert_eq!(ring.label(0), None);
        for i in 0..SLOTS {
            ring.push(Text::from(format_args!("dispatch {i}")));
        }
        assert_eq!(ring.label(1), None);
        assert_eq!(ring.last(), SLOTS as u64 + 1);
        assert_eq!(
            ring.label(ring.last()),
            Some(format!("dispatch {}", SLOTS - 1).as_str())
        );
    }

    #[test]
    fn in_flight_spans_the_oldest_stage_to_the_newest_recorded() {
        let top = vk::PipelineStageFlags::TOP_OF_PIPE;
        let bottom = vk::PipelineStageFlags::BOTTOM_OF_PIPE;
        assert_eq!(in_flight(&[(top, 90), (bottom, 75)], 120), Some((75, 120)));
        assert_eq!(in_flight(&[(top, 0), (bottom, 0)], 120), None);
        assert_eq!(in_flight(&[(bottom, 40)], 0), Some((40, 40)));
    }

    #[test]
    fn address_range_covers_the_precision_block() {
        assert_eq!(
            address_range(0x1234_5678, 0x1000),
            (0x1234_5000, 0x1234_5fff)
        );
        assert_eq!(address_range(0x1234_5678, 0), (0x1234_5678, 0x1234_5678));
        assert_eq!(address_range(0x10, 1), (0x10, 0x10));
    }
}
