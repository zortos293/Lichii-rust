use lichii_common::PAddr;

#[derive(Debug, thiserror::Error)]
pub enum PhysicalMemoryError {
    #[error("reserving {size:#x} bytes of guest DRAM: {source}")]
    Reserve { size: u64, source: std::io::Error },
}

/// Guest DRAM. Physical addresses are offsets into it.
pub struct PhysicalMemory {
    map: memmap2::MmapRaw,
}

impl PhysicalMemory {
    pub fn new(size: u64) -> Result<Self, PhysicalMemoryError> {
        let map = memmap2::MmapOptions::new()
            .len(size as usize)
            .map_anon()
            .map(memmap2::MmapRaw::from)
            .map_err(|source| PhysicalMemoryError::Reserve { size, source })?;
        Ok(Self { map })
    }

    pub fn size(&self) -> u64 {
        self.map.len() as u64
    }

    /// Host pointer to physical address \`pa\`. Panics if \`pa\` is outside DRAM.
    pub fn pointer(&self, pa: PAddr) -> *mut u8 {
        assert!(pa < self.size(), "physical address {pa:#x} outside DRAM");
        // SAFETY: pa is inside the mapping.
        unsafe { self.map.as_mut_ptr().add(pa as usize) }
    }
}

// SAFETY: the mapping lives as long as the struct; guest accesses are unsynchronised by design,
// like the hardware they model.
unsafe impl Send for PhysicalMemory {}
unsafe impl Sync for PhysicalMemory {}
