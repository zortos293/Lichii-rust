//! A lost device stops the renderer: after `VK_ERROR_DEVICE_LOST`, nothing records, submits or
//! waits on the GPU any more. The loss is process-wide, so this binary holds this one test.

use std::sync::Arc;

use ash::vk;
use lichii_common::settings::GraphicsSettings;
use lichii_vulkan::scheduler::Scheduler;
use lichii_vulkan::{Device, Error, Instance, SurfaceSupport};

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
fn a_lost_device_stops_recording_submitting_and_waiting() {
    let Some(device) = device() else { return };
    let mut scheduler = Scheduler::new(&device).unwrap();
    scheduler.record().unwrap();
    let done = scheduler.flush().unwrap();
    scheduler.wait(done).unwrap();
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
}
