//! Types shared by every Lichii crate.

pub mod paths;
pub mod perf;
pub mod result;
pub mod settings;

pub use result::{HleResult, ResultCode};

/// Guest virtual address (process address space).
pub type VAddr = u64;
/// Guest physical address (DRAM offset as seen by the kernel).
pub type PAddr = u64;
/// GPU virtual address (GMMU address space of a channel).
pub type GpuVAddr = u64;
/// Device address (SMMU address space shared by the GPU and other devices).
pub type DAddr = u64;

pub const PAGE_BITS: u32 = 12;
pub const PAGE_SIZE: u64 = 1 << PAGE_BITS;
pub const PAGE_MASK: u64 = PAGE_SIZE - 1;

#[inline]
pub const fn align_down(value: u64, align: u64) -> u64 {
    value & !(align - 1)
}

#[inline]
pub const fn align_up(value: u64, align: u64) -> u64 {
    (value + align - 1) & !(align - 1)
}

/// Which console a title targets. Switch 2 titles have program ids starting with `04`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ConsoleGeneration {
    Switch1,
    Switch2,
}

impl ConsoleGeneration {
    pub fn from_program_id(program_id: u64) -> Self {
        if program_id >> 56 == 0x04 {
            Self::Switch2
        } else {
            Self::Switch1
        }
    }

    /// Number of CPU cores visible to applications.
    pub fn core_count(self) -> usize {
        match self {
            Self::Switch1 => 4,
            Self::Switch2 => 6,
        }
    }
}

/// Installs the global tracing subscriber. `filter` uses `EnvFilter` syntax; `RUST_LOG` wins when set.
pub fn init_logging(filter: &str) {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_from_program_id() {
        assert_eq!(
            ConsoleGeneration::from_program_id(0x0400C3F00006E000),
            ConsoleGeneration::Switch2
        );
        assert_eq!(
            ConsoleGeneration::from_program_id(0x0100152000022000),
            ConsoleGeneration::Switch1
        );
    }

    #[test]
    fn alignment() {
        assert_eq!(align_up(0x1001, PAGE_SIZE), 0x2000);
        assert_eq!(align_down(0x1fff, PAGE_SIZE), 0x1000);
    }
}
