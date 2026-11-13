//! host1x: the SMMU and the syncpoints the GPU shares with the other Tegra engines.

mod syncpoints;

use std::sync::Arc;

use lichii_memory::{DeviceAddressSpace, PhysicalMemory};

pub use syncpoints::{
    ActionHandle, InvalidSyncpoint, SYNCPOINT_COUNT, SyncpointBank, SyncpointWaitError, Syncpoints,
    is_reached,
};

use crate::backend::GpuWrittenPages;
use crate::device_memory::DeviceAllocator;
use crate::lazy::LazyDeviceRanges;

/// What every host1x client shares: the device address space, its allocator and syncpoints.
#[derive(Clone)]
pub struct Host1x {
    pub memory: Arc<DeviceAddressSpace>,
    pub allocator: Arc<DeviceAllocator>,
    pub syncpoints: Arc<Syncpoints>,
    pub written_pages: Arc<GpuWrittenPages>,
    /// Device ranges that get memory as they are written (Switch 2 GPU-only regions).
    pub lazy: Arc<LazyDeviceRanges>,
}

impl Host1x {
    /// `address_space_bits` is the SMMU width: 34 on Switch 1; Switch 2 titles need 36 (64 GiB,
    /// NX2_NOTES row 85).
    pub fn new(physical: Arc<PhysicalMemory>, address_space_bits: u32) -> Self {
        let memory = Arc::new(DeviceAddressSpace::new(physical, address_space_bits));
        Self {
            lazy: Arc::new(LazyDeviceRanges::new(memory.clone())),
            memory,
            allocator: Arc::new(DeviceAllocator::new(address_space_bits)),
            syncpoints: Arc::new(Syncpoints::new()),
            written_pages: Arc::new(GpuWrittenPages::new(address_space_bits)),
        }
    }
}
