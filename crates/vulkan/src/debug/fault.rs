//! GPU fault reports, on with `debug.gpu_fault_report`. When the device is lost, logs what
//! `VK_EXT_device_fault` reports (the fault, faulting addresses and vendor records) and writes
//! the vendor's binary dump to `LICHII_GPU_FAULT_DIR` or else next to the crash dumps. Every label
//! (and every command buffer's start) also leaves a numbered checkpoint: an NVIDIA checkpoint
//! (`VK_NV_device_diagnostic_checkpoints`) and a pair of buffer markers (`VK_AMD_buffer_marker`)
//! that record the newest checkpoint the top and the bottom of the pipe passed, in host memory
//! that outlives the device. The report names the last draw or dispatch each stage passed and the
//! work between them, which holds the fault. Labels are kept in a fixed ring of [`SLOTS`] entries,
//! so a checkpoint older than that shows by number only.

use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use ash::{amd, ext, nv, vk};
use parking_lot::Mutex;

use super::labels::Text;

/// Labels kept for the report, about 13 MiB: a hung GPU lets the host record a few seconds of
/// work (some 60k draws in MKW) past the last checkpoint the GPU passed.
const SLOTS: usize = 1 << 16;
/// Most in-flight labels listed in one report.
const IN_FLIGHT_LISTED: u64 = 48;
/// Byte offsets of the newest checkpoint the top and the bottom of the pipe passed.
const TOP_MARKER: vk::DeviceSize = 0;
const BOTTOM_MARKER: vk::DeviceSize = 4;

static DIR: OnceLock<PathBuf> = OnceLock::new();
static REPORTER: StdMutex<Option<Arc<Reporter>>> = StdMutex::new(None);

pub(super) fn init(crash_dumps: &Path) {
    let dir = std::env::var_os("LICHII_GPU_FAULT_DIR")
        .map_or_else(|| crash_dumps.to_path_buf(), PathBuf::from);
    let _ = DIR.set(dir);
}

/// The fault-report extensions a device turned on.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaultSupport {
    /// `VK_EXT_device_fault`, with whether it returns a vendor binary.
    pub fault: Option<bool>,
    pub checkpoints: bool,
    pub buffer_marker: bool,
}

/// What the device turned on for fault reports.
pub(crate) struct Reporter {
    device: vk::Device,
    queue: vk::Queue,
    fault: Option<ext::device_fault::Device>,
    vendor_binary: bool,
    checkpoints: Option<nv::device_diagnostic_checkpoints::Device>,
    /// `vkGetQueueCheckpointData2NV`, when the driver has it.
    checkpoints2: Option<vk::PFN_vkGetQueueCheckpointData2NV>,
    markers: Option<Markers>,
    ring: Mutex<Ring>,
}

impl Reporter {
    /// The reporter of a device made with the extensions `support` names; it reports when any
    /// Vulkan call returns `VK_ERROR_DEVICE_LOST` until [`Self::destroy`].
    pub(crate) fn register(
        instance: &ash::Instance,
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
        queue: vk::Queue,
        support: FaultSupport,
    ) -> Arc<Self> {
        let markers = support
            .buffer_marker
            .then(|| Markers::new(instance, device, memory))
            .and_then(|markers| {
                markers
                    .inspect_err(|error| {
                        tracing::warn!("GPU fault report: no buffer markers: {error}")
                    })
                    .ok()
            });
        let reporter = Arc::new(Self {
            device: device.handle(),
            queue,
            fault: support
                .fault
                .map(|_| ext::device_fault::Device::new(instance, device)),
            vendor_binary: support.fault == Some(true),
            checkpoints: support
                .checkpoints
                .then(|| nv::device_diagnostic_checkpoints::Device::new(instance, device)),
            checkpoints2: support
                .checkpoints
                .then(|| {
                    // SAFETY: a device-level entry point of a live device; the type is its
                    // signature in the Vulkan headers.
                    unsafe {
                        instance
                            .get_device_proc_addr(
                                device.handle(),
                                c"vkGetQueueCheckpointData2NV".as_ptr(),
                            )
                            .map(|f| {
                                std::mem::transmute::<_, vk::PFN_vkGetQueueCheckpointData2NV>(f)
                            })
                    }
                })
                .flatten(),
            markers,
            ring: Mutex::new(Ring::new()),
        });
        *REPORTER.lock().unwrap_or_else(|e| e.into_inner()) = Some(reporter.clone());
        reporter
    }

    /// Stops reporting for this device and frees the marker buffer; call it before the device
    /// is destroyed.
    pub(crate) fn destroy(self: &Arc<Self>, device: &ash::Device) {
        let mut current = REPORTER.lock().unwrap_or_else(|e| e.into_inner());
        if current.as_ref().is_some_and(|r| Arc::ptr_eq(r, self)) {
            *current = None;
        }
        if let Some(markers) = &self.markers {
            // SAFETY: the device is idle, so no command buffer writes the markers any more.
            unsafe {
                device.destroy_buffer(markers.buffer, None);
                device.free_memory(markers.memory, None);
            }
        }
    }

    /// Leaves a checkpoint named `text` in `cmd`.
    pub(super) fn checkpoint(&self, cmd: vk::CommandBuffer, text: Text) {
        if self.checkpoints.is_none() && self.markers.is_none() {
            return;
        }
        let seq = self.ring.lock().push(text);
        if let Some(checkpoints) = &self.checkpoints {
            // SAFETY: `cmd` is recording; the marker is an integer the driver hands back as is.
            unsafe {
                checkpoints.cmd_set_checkpoint(cmd, std::ptr::without_provenance(seq as usize))
            };
        }
        if let Some(markers) = &self.markers {
            let write = markers.write.fp().cmd_write_buffer_marker_amd;
            // SAFETY: `cmd` is recording and the marker buffer lives as long as the device.
            unsafe {
                write(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    markers.buffer,
                    TOP_MARKER,
                    seq as u32,
                );
                write(
                    cmd,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    markers.buffer,
                    BOTTOM_MARKER,
                    seq as u32,
                );
            }
        }
    }

    fn report(&self) {
        if let Some(fault) = &self.fault {
            self.report_fault(fault);
        }
        if self.checkpoints.is_some() || self.markers.is_some() {
            self.report_progress();
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
            save_binary(&binary);
        }
    }

    fn report_progress(&self) {
        let ring = self.ring.lock();
        let mut passed = Vec::new();
        if let Some(checkpoints) = &self.checkpoints {
            for (stage, marker, top) in self.queue_checkpoints(checkpoints) {
                passed.push((format!("checkpoint {stage}"), marker as usize as u64, top));
            }
        }
        if let Some(markers) = &self.markers {
            for (stage, offset, top) in [
                ("top of pipe", TOP_MARKER, true),
                ("bottom of pipe", BOTTOM_MARKER, false),
            ] {
                passed.push((
                    format!("buffer marker {stage}"),
                    ring.widen(markers.read(offset)),
                    top,
                ));
            }
        }
        for (source, seq, _) in &passed {
            tracing::error!(
                source = source.as_str(),
                checkpoint = seq,
                label = ring.label(*seq).unwrap_or("(gone from the ring)"),
                "GPU fault: last checkpoint passed"
            );
        }
        let seqs: Vec<(u64, bool)> = passed.iter().map(|&(_, seq, top)| (seq, top)).collect();
        let Some((first, last)) = in_flight(&seqs, ring.last()) else {
            tracing::error!(
                recorded = ring.last(),
                "GPU fault: no checkpoint was reached"
            );
            return;
        };
        tracing::error!(
            first,
            last,
            recorded = ring.last(),
            listed = (last - first + 1).min(IN_FLIGHT_LISTED),
            "GPU fault: work from the oldest checkpoint passed to the newest one the top of the pipe passed (or the newest recorded), which holds the fault"
        );
        for seq in first..=last.min(first + IN_FLIGHT_LISTED - 1) {
            tracing::error!(
                checkpoint = seq,
                label = ring.label(seq).unwrap_or("(gone from the ring)"),
                "GPU fault: in flight"
            );
        }
    }

    /// Each stage's last checkpoint marker, from `vkGetQueueCheckpointData2NV` or else the
    /// original query.
    fn queue_checkpoints(
        &self,
        checkpoints: &nv::device_diagnostic_checkpoints::Device,
    ) -> Vec<(String, *mut c_void, bool)> {
        let fp = checkpoints.fp();
        let mut count = 0;
        if let Some(get) = self.checkpoints2 {
            // SAFETY: the queue belongs to the lost device; a null array asks for the count.
            unsafe { get(self.queue, &mut count, std::ptr::null_mut()) };
            if count > 0 {
                let mut data = vec![vk::CheckpointData2NV::default(); count as usize];
                // SAFETY: `data` holds `count` elements.
                unsafe { get(self.queue, &mut count, data.as_mut_ptr()) };
                data.truncate(count as usize);
                return data
                    .iter()
                    .map(|d| {
                        let top = d.stage == vk::PipelineStageFlags2::TOP_OF_PIPE;
                        (format!("{:?}", d.stage), d.p_checkpoint_marker, top)
                    })
                    .collect();
            }
        }
        // SAFETY: the queue belongs to the lost device; a null array asks for the count.
        unsafe { (fp.get_queue_checkpoint_data_nv)(self.queue, &mut count, std::ptr::null_mut()) };
        let mut data = vec![vk::CheckpointDataNV::default(); count as usize];
        // SAFETY: `data` holds `count` elements.
        unsafe { (fp.get_queue_checkpoint_data_nv)(self.queue, &mut count, data.as_mut_ptr()) };
        data.truncate(count as usize);
        data.iter()
            .map(|d| {
                let top = d.stage == vk::PipelineStageFlags::TOP_OF_PIPE;
                (format!("{:?}", d.stage), d.p_checkpoint_marker, top)
            })
            .collect()
    }
}

/// Reports the fault of the registered device, if any.
pub(super) fn report() {
    let reporter = REPORTER.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(reporter) = reporter {
        reporter.report();
    }
}

fn save_binary(binary: &[u8]) {
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

/// Two 32-bit buffer markers in host-coherent memory that stays readable after the device is
/// lost.
struct Markers {
    write: amd::buffer_marker::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *const u32,
}

// SAFETY: `mapped` points into memory the reporter owns and keeps mapped; it is only read,
// with volatile loads, after the device was lost.
unsafe impl Send for Markers {}
// SAFETY: as above.
unsafe impl Sync for Markers {}

impl Markers {
    fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        memory: &vk::PhysicalDeviceMemoryProperties,
    ) -> Result<Self, vk::Result> {
        let info = vk::BufferCreateInfo::default()
            .size(BOTTOM_MARKER + 4)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: a valid create info on a live device.
        let buffer = unsafe { device.create_buffer(&info, None)? };
        // SAFETY: `buffer` is live.
        let needs = unsafe { device.get_buffer_memory_requirements(buffer) };
        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let Some(index) = (0..memory.memory_type_count).find(|&i| {
            needs.memory_type_bits & (1 << i) != 0
                && memory.memory_types[i as usize]
                    .property_flags
                    .contains(wanted)
        }) else {
            // SAFETY: nothing uses the buffer yet.
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(vk::Result::ERROR_FEATURE_NOT_PRESENT);
        };
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(needs.size)
            .memory_type_index(index);
        // SAFETY: the buffer and memory are fresh; each failure frees what was made before it.
        unsafe {
            let memory = device.allocate_memory(&allocate, None).inspect_err(|_| {
                device.destroy_buffer(buffer, None);
            })?;
            let mapped = device
                .bind_buffer_memory(buffer, memory, 0)
                .and_then(|()| {
                    device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                })
                .inspect_err(|_| {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                })?
                .cast::<u32>();
            mapped.write_bytes(0, 2);
            Ok(Self {
                write: amd::buffer_marker::Device::new(instance, device),
                buffer,
                memory,
                mapped,
            })
        }
    }

    fn read(&self, offset: vk::DeviceSize) -> u32 {
        // SAFETY: `offset` is one of the two markers inside the mapped buffer.
        unsafe { self.mapped.add(offset as usize / 4).read_volatile() }
    }
}

/// The bytes a fault address may stand for: its `precision` (a power of two) aligned block.
fn address_range(reported: u64, precision: u64) -> (u64, u64) {
    let mask = precision.max(1).next_power_of_two() - 1;
    (reported & !mask, reported | mask)
}

/// The work that may not have finished: from the oldest checkpoint any stage passed to the
/// newest one the top of the pipe passed, or to the newest `recorded` when no top-of-pipe
/// checkpoint bounds it. `passed` holds each stage's checkpoint and whether it is the top of
/// the pipe. `None` when no stage passed any.
fn in_flight(passed: &[(u64, bool)], recorded: u64) -> Option<(u64, u64)> {
    let first = passed
        .iter()
        .map(|&(seq, _)| seq)
        .filter(|&seq| seq != 0)
        .min()?;
    let last = passed
        .iter()
        .filter(|&&(seq, top)| top && seq != 0)
        .map(|&(seq, _)| seq)
        .max()
        .unwrap_or(recorded);
    Some((first, last.max(first)))
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

    /// The newest sequence number recorded whose low 32 bits are `low` (what a marker holds).
    fn widen(&self, low: u32) -> u64 {
        if low == 0 {
            return 0;
        }
        self.last
            .saturating_sub(u64::from((self.last as u32).wrapping_sub(low)))
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
    fn markers_widen_to_the_newest_matching_sequence_number() {
        let mut ring = Ring::new();
        assert_eq!(ring.widen(0), 0);
        ring.last = (5 << 32) + 10;
        assert_eq!(ring.widen(10), (5 << 32) + 10);
        assert_eq!(ring.widen(7), (5 << 32) + 7);
        assert_eq!(ring.widen(u32::MAX), (4 << 32) + u64::from(u32::MAX));
        ring.last = 3;
        assert_eq!(ring.widen(9), 0);
    }

    #[test]
    fn in_flight_ends_at_the_top_of_pipe_or_else_the_newest_recorded() {
        assert_eq!(in_flight(&[(90, true), (75, false)], 120), Some((75, 90)));
        assert_eq!(in_flight(&[(40, true), (40, false)], 120), Some((40, 40)));
        assert_eq!(in_flight(&[(75, false)], 120), Some((75, 120)));
        assert_eq!(in_flight(&[(0, true), (75, false)], 120), Some((75, 120)));
        assert_eq!(in_flight(&[(40, false)], 0), Some((40, 40)));
        assert_eq!(in_flight(&[(0, true), (0, false)], 120), None);
        assert_eq!(in_flight(&[], 120), None);
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
