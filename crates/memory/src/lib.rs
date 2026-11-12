//! Guest memory: DRAM backing, per-process address spaces and the exclusive monitor.
//!
//! The first implementation keeps a two-level table of host pointers per guest page. The kernel
//! crew replaces the backing with reserve-and-commit host memory and adds fastmem views without
//! changing this API.

mod address_space;
mod exclusive;
mod physical;

pub use address_space::{AddressSpace, FaultKind, MemoryFault, MemoryPermission};
pub use exclusive::ExclusiveMonitor;
pub use physical::{PhysicalMemory, PhysicalMemoryError};
