//! host1x: the SMMU and the syncpoints the GPU shares with the other Tegra engines.

mod syncpoints;

use std::sync::Arc;

use lichii_memory::{DeviceAddressSpace, PhysicalMemory};

pub use syncpoints::{
    ActionHandle, InvalidSyncpoint, SYNCPOINT_COUNT, SyncpointBank, SyncpointWaitError, Syncpoints,
    is_reached,
};

use crate::device_memory::DeviceAllocator;

/// What every host1x client shares: the device address space, its allocator and syncpoints.
#[derive(Clone)]
pub struct Host1x {
    pub memory: Arc<DeviceAddressSpace>,
    pub allocator: Arc<DeviceAllocator>,
    pub syncpoints: Arc<Syncpoints>,
}

impl Host1x {
    /// `address_space_bits` is the SMMU width: 34 on Switch 1; Switch 2 titles need 36 (64 GiB,
    /// NX2_NOTES row 85).
    pub fn new(physical: Arc<PhysicalMemory>, address_space_bits: u32) -> Self {
        Self {
            memory: Arc::new(DeviceAddressSpace::new(physical, address_space_bits)),
            allocator: Arc::new(DeviceAllocator::new(address_space_bits)),
            syncpoints: Arc::new(Syncpoints::new()),
        }
    }
}
