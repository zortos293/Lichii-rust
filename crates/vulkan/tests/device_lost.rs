//! A lost device stops the renderer: after `VK_ERROR_DEVICE_LOST`, nothing records, submits or
//! waits on the GPU any more, and guest reads of GPU-written pages stop trapping. The loss is
//! process-wide, so this binary holds this one test.

use std::sync::Arc;
use std::time::Instant;

use ash::vk;
use lichii_common::settings::GraphicsSettings;
use lichii_gpu::activity::Activity;
use lichii_gpu::backend::{GpuHandles, GpuWrittenPages, Rasterizer};
use lichii_gpu::clock::{GpuTimestamp, GuestTime};
use lichii_gpu::host1x::Syncpoints;
use lichii_gpu::lazy::LazyDeviceRanges;
use lichii_memory::{AddressSpace, DeviceAddressSpace, MemoryPermission, PhysicalMemory};
use lichii_texture::block_linear::GobLayout;
use lichii_vulkan::present::FrameMailbox;
use lichii_vulkan::rasterizer::VulkanRasterizer;
use lichii_vulkan::scheduler::Scheduler;
use lichii_vulkan::{Device, Error, Instance, SurfaceSupport};

const DRAM: u64 = 16 << 20;
const DADDR: u64 = 0x1_0000_0000;
const CPU_VA: u64 = 0x1000_0000;

struct HostTime(Instant);

impl GuestTime for HostTime {
    fn global_time_ns(&self) -> u64 {
        self.0.elapsed().as_nanos() as u64 + 1
    }
}

fn device() -> Option<Arc<Device>> {
    let instance = match Instance::new(true, SurfaceSupport::None) {
        Ok(instance) => Arc::new(instance),
        Err(error) => {
            eprintln!("skipping: {error}");
            return None;
        }
    };
    match Device::new(instance, &GraphicsSettings::default(), None) {
        Ok(device) => Some(device),
        Err(error) => {
            eprintln!("skipping: {error}");
            None
        }
    }
}

#[test]
fn a_lost_device_stops_gpu_work_and_flush_traps() {
    let Some(device) = device() else { return };
    let physical = Arc::new(PhysicalMemory::new(DRAM).unwrap());
    physical.commit(0, DRAM).unwrap();
    let memory = Arc::new(DeviceAddressSpace::new(physical.clone(), 36));
    memory.map(DADDR, 0, DRAM);
    let space = Arc::new(AddressSpace::new(physical, 39));
    space.map(CPU_VA, 0, 0x10000, MemoryPermission::READ_WRITE);
    memory.attach_process(space);
    let handles = GpuHandles {
        device: memory.clone(),
        syncpoints: Arc::new(Syncpoints::new()),
        timestamp: Arc::new(GpuTimestamp::new(Arc::new(HostTime(Instant::now())), false)),
        written_pages: Arc::new(GpuWrittenPages::new(36)),
        lazy: Arc::new(LazyDeviceRanges::new(memory.clone())),
        activity: Arc::new(Activity::new()),
    };
    let mut rasterizer = VulkanRasterizer::new(
        &device,
        handles,
        Arc::new(FrameMailbox::default()),
        GobLayout::Tegra,
    )
    .unwrap();
    let mut scheduler = Scheduler::new(&device).unwrap();
    scheduler.record().unwrap();
    let done = scheduler.flush().unwrap();
    scheduler.wait(done).unwrap();
    memory.arm_flush_traps(DADDR, 0x2000);
    memory.take_flush_trap_stats();
    assert!(!lichii_vulkan::debug::device_lost());

    scheduler.record().unwrap();
    lichii_vulkan::debug::inject_device_lost();
    assert!(lichii_vulkan::debug::device_lost());

    let pending = scheduler.tick();
    assert!(matches!(scheduler.record(), Err(Error::DeviceLost)));
    assert!(matches!(scheduler.flush(), Err(Error::DeviceLost)));
    assert_eq!(scheduler.tick(), pending, "a submission went out");
    assert!(matches!(scheduler.wait(pending), Err(Error::DeviceLost)));
    assert!(matches!(scheduler.finish(), Err(Error::DeviceLost)));
    let mut recorded = false;
    assert!(matches!(
        scheduler.run_aside(|_| recorded = true),
        Err(Error::DeviceLost)
    ));
    assert!(!recorded, "work was recorded on a lost device");
    assert!(matches!(
        device.submit(&[vk::SubmitInfo2::default()], vk::Fence::null()),
        Err(Error::DeviceLost)
    ));
    scheduler.wait(done).unwrap();

    rasterizer.flush_for_cpu_read(DADDR, 0x2000);
    let stats = memory.take_flush_trap_stats();
    assert_eq!(
        (stats.arms, stats.disarms),
        (0, 1),
        "a CPU read after the loss must end its trap instead of rearming it: {stats:?}"
    );
}
