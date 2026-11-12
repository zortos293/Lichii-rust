//! Kernel memory management: physical pools, the block map and process page tables.

pub mod block;
pub mod manager;
pub mod page_table;

pub use block::{MemoryInfo, MemoryState, attribute};
pub use manager::{MemoryManager, PageGroup, Pool, PoolLayout};
pub use page_table::{LayoutParams, PageTable, Region, Regions};
