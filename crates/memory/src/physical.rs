use std::io;

use lichii_common::PAddr;

use crate::host::Backing;

#[derive(Debug, thiserror::Error)]
pub enum PhysicalMemoryError {
    #[error("reserving {size:#x} bytes of guest DRAM: {source}")]
    Reserve { size: u64, source: io::Error },
    #[error("committing {size:#x} bytes of guest DRAM at {pa:#x}: {source}")]
    Commit {
        pa: PAddr,
        size: u64,
        source: io::Error,
    },
}

/// Guest DRAM. Physical addresses are offsets into it.
///
/// Creating it only reserves host address space; host memory is committed as the kernel hands
/// out pages ([`PhysicalMemory::commit`], also done by every map), so a 12 GiB Switch 2 layout
/// costs only what the title touches (NX2_NOTES row 128).
pub struct PhysicalMemory {
    backing: Backing,
}

impl PhysicalMemory {
    pub fn new(size: u64) -> Result<Self, PhysicalMemoryError> {
        let backing = Backing::new(size as usize)
            .map_err(|source| PhysicalMemoryError::Reserve { size, source })?;
        Ok(Self { backing })
    }

    pub fn size(&self) -> u64 {
        self.backing.size() as u64
    }

    /// Host pointer to physical address `pa`. Panics if `pa` is outside DRAM.
    pub fn pointer(&self, pa: PAddr) -> *mut u8 {
        assert!(pa < self.size(), "physical address {pa:#x} outside DRAM");
        // SAFETY: pa is inside the mapping.
        unsafe { self.backing.ptr().add(pa as usize) }
    }

    /// Physical address of a host pointer into DRAM.
    pub fn physical_address(&self, ptr: *const u8) -> Option<PAddr> {
        let offset = (ptr as usize).checked_sub(self.backing.ptr() as usize)?;
        (offset < self.backing.size()).then_some(offset as PAddr)
    }

    /// Backs `[pa, pa + size)` with host memory. Idempotent and cheap once committed.
    pub fn commit(&self, pa: PAddr, size: u64) -> Result<(), PhysicalMemoryError> {
        assert!(pa + size <= self.size(), "commit past the end of DRAM");
        self.backing
            .commit(pa as usize, size as usize)
            .map_err(|source| PhysicalMemoryError::Commit { pa, size, source })
    }

    /// Zeroes `[pa, pa + size)` and gives its host pages back to the OS where the host allows.
    pub fn discard(&self, pa: PAddr, size: u64) {
        assert!(pa + size <= self.size(), "discard past the end of DRAM");
        self.backing.discard(pa as usize, size as usize);
    }

    /// Whether pages can also be mapped as views elsewhere (fastmem).
    pub fn supports_views(&self) -> bool {
        self.backing.supports_views()
    }

    pub(crate) fn backing(&self) -> &Backing {
        &self.backing
    }
}
