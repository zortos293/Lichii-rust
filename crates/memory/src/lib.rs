//! Guest memory: DRAM backing, per-process address spaces, device (SMMU) address spaces and the
//! exclusive monitor.
//!
//! [`PhysicalMemory`] reserves guest DRAM as a shareable host object and commits it on demand.
//! [`AddressSpace`] translates guest virtual addresses through a lock-free two-level table of
//! host pointers and can mirror the guest layout into a [`Fastmem`] arena for the JIT.
//! [`DeviceAddressSpace`] is the same table keyed by device address, for the GPU.

mod address_space;
mod device;
mod exclusive;
mod fault;
mod host;
mod page_table;
mod physical;

pub use address_space::{
    AddressSpace, Fastmem, FastmemError, FaultKind, MemoryFault, MemoryPermission,
};
pub use device::DeviceAddressSpace;
pub use exclusive::ExclusiveMonitor;
pub use fault::{FaultHandler, HostFault};
pub use physical::{PhysicalMemory, PhysicalMemoryError};

#[cfg(test)]
mod fastmem_tests;
