//! host1x: the SMMU and the syncpoints the GPU shares with the other Tegra engines.

mod syncpoints;

use std::sync::Arc;

use lichii_memory::PhysicalMemory;

pub use syncpoints::{
    ActionHandle, InvalidSyncpoint, SYNCPOINT_COUNT, SyncpointBank, SyncpointWaitError, Syncpoints,
    is_reached,
};

use crate::device_memory::DeviceMemory;

/// What every host1x client shares: device memory and syncpoints.
#[derive(Clone)]
pub struct Host1x {
    pub memory: Arc<DeviceMemory>,
    pub syncpoints: Arc<Syncpoints>,
}

impl Host1x {
    pub fn new(physical: Arc<PhysicalMemory>) -> Self {
        Self {
            memory: Arc::new(DeviceMemory::new(physical)),
            syncpoints: Arc::new(Syncpoints::new()),
        }
    }
}
