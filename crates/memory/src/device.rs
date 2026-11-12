use std::sync::Arc;

use lichii_common::{DAddr, PAGE_MASK, PAGE_SIZE, PAddr, VAddr};
use parking_lot::Mutex;

use crate::page_table::PageTable;
use crate::{AddressSpace, MemoryFault, MemoryPermission, PhysicalMemory};

/// A device (SMMU) address space: guest memory as the GPU and other DMA engines see it.
///
/// Device addresses map to DRAM pages, usually pinned from a process address space
/// ([`DeviceAddressSpace::map_process`]). Translation is a lock-free table walk, safe to call from
/// the GPU thread while a service thread maps or unmaps; there is no translation cache to
/// invalidate.
pub struct DeviceAddressSpace {
    table: PageTable,
    physical: Arc<PhysicalMemory>,
    update: Mutex<()>,
}

impl DeviceAddressSpace {
    /// `address_space_bits` is 34 on Switch 1 (Tegra X1 SMMU) and 35 on Switch 2.
    pub fn new(physical: Arc<PhysicalMemory>, address_space_bits: u32) -> Self {
        Self {
            table: PageTable::new(address_space_bits),
            physical,
            update: Mutex::new(()),
        }
    }

    /// Width of the device address space in bits.
    pub fn address_space_bits(&self) -> u32 {
        self.table.bits()
    }

    /// The DRAM this space maps.
    pub fn physical(&self) -> &Arc<PhysicalMemory> {
        &self.physical
    }

    /// Maps `size` bytes of DRAM at `pa` to device address `daddr`.
    pub fn map(&self, daddr: DAddr, pa: PAddr, size: u64) {
        assert_eq!(
            (daddr | pa | size) & PAGE_MASK,
            0,
            "unaligned device map {daddr:#x} {pa:#x} {size:#x}"
        );
        assert!(
            pa + size <= self.physical.size(),
            "device map past the end of DRAM"
        );
        if let Err(err) = self.physical.commit(pa, size) {
            panic!("{err}");
        }
        let _update = self.update.lock();
        for page in 0..size / PAGE_SIZE {
            let host = self.physical.pointer(pa + page * PAGE_SIZE);
            self.table
                .set(daddr + page * PAGE_SIZE, host, MemoryPermission::READ_WRITE);
        }
    }

    /// Maps the DRAM pages behind `[va, va + size)` of `process` to `daddr`. Fails without
    /// changing anything when a page of the range is unmapped. `process` must use the same
    /// [`PhysicalMemory`] as this device space.
    pub fn map_process(
        &self,
        daddr: DAddr,
        process: &AddressSpace,
        va: VAddr,
        size: u64,
    ) -> Result<(), MemoryFault> {
        assert_eq!(
            (daddr | va | size) & PAGE_MASK,
            0,
            "unaligned device map {daddr:#x} {va:#x} {size:#x}"
        );
        assert!(
            Arc::ptr_eq(process.physical(), &self.physical),
            "device and process address spaces use different DRAM"
        );
        let hosts = (0..size / PAGE_SIZE)
            .map(|page| {
                let addr = va + page * PAGE_SIZE;
                process
                    .translate(addr)
                    .map(|(host, _)| host)
                    .ok_or(MemoryFault::unmapped(addr))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let _update = self.update.lock();
        for (page, host) in hosts.into_iter().enumerate() {
            self.table.set(
                daddr + page as u64 * PAGE_SIZE,
                host,
                MemoryPermission::READ_WRITE,
            );
        }
        Ok(())
    }

    /// Removes the mappings of `[daddr, daddr + size)`.
    pub fn unmap(&self, daddr: DAddr, size: u64) {
        assert_eq!(
            (daddr | size) & PAGE_MASK,
            0,
            "unaligned device unmap {daddr:#x} {size:#x}"
        );
        let _update = self.update.lock();
        for page in 0..size / PAGE_SIZE {
            self.table.clear(daddr + page * PAGE_SIZE);
        }
    }

    /// Host pointer for the byte at `daddr`.
    #[inline]
    pub fn translate(&self, daddr: DAddr) -> Option<*mut u8> {
        self.table.translate(daddr).map(|(ptr, _)| ptr)
    }

    /// Host pointer for `[daddr, daddr + size)` when it is mapped to contiguous DRAM.
    pub fn contiguous_pointer(&self, daddr: DAddr, size: u64) -> Option<*mut u8> {
        self.table.contiguous(daddr, size)
    }

    /// Physical address of the byte at `daddr`.
    pub fn physical_address(&self, daddr: DAddr) -> Option<PAddr> {
        self.physical.physical_address(self.translate(daddr)?)
    }

    /// Whether the page holding `daddr` is mapped.
    pub fn is_mapped(&self, daddr: DAddr) -> bool {
        self.translate(daddr).is_some()
    }

    /// Reads a value; it may straddle pages. Fails on the first unmapped page.
    #[inline]
    pub fn read<T: bytemuck::Pod>(&self, daddr: DAddr) -> Result<T, MemoryFault> {
        self.table.read(daddr)
    }

    /// Writes a value; it may straddle pages. Fails on the first unmapped page, after writing the
    /// bytes before it.
    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, daddr: DAddr, value: T) -> Result<(), MemoryFault> {
        self.table.write(daddr, value)
    }

    /// Copies `out.len()` bytes from `daddr`. Fails on the first unmapped page.
    pub fn read_bytes(&self, daddr: DAddr, out: &mut [u8]) -> Result<(), MemoryFault> {
        self.table.read_bytes(daddr, out)
    }

    /// Copies `data` to `daddr`. Fails on the first unmapped page, after writing the bytes
    /// before it.
    pub fn write_bytes(&self, daddr: DAddr, data: &[u8]) -> Result<(), MemoryFault> {
        self.table.write_bytes(daddr, data)
    }

    /// Sets `size` bytes at `daddr` to `byte`. Fails on the first unmapped page.
    pub fn fill(&self, daddr: DAddr, size: u64, byte: u8) -> Result<(), MemoryFault> {
        self.table.fill(daddr, size, byte)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_sees_process_writes() {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let process = AddressSpace::new(physical.clone(), 39);
        process.map(0x8000_0000, 0x10_0000, 0x3000, MemoryPermission::READ_WRITE);
        let device = DeviceAddressSpace::new(physical, 35);
        device
            .map_process(0x4_0000_0000, &process, 0x8000_0000, 0x3000)
            .unwrap();
        process
            .write(0x8000_1ff8, 0xabcd_ef01_2345_6789u64)
            .unwrap();
        assert_eq!(
            device.read::<u64>(0x4_0000_1ff8).unwrap(),
            0xabcd_ef01_2345_6789
        );
        assert_eq!(device.physical_address(0x4_0000_2000), Some(0x10_2000));
        device.write(0x4_0000_0010, 5u32).unwrap();
        assert_eq!(process.read::<u32>(0x8000_0010).unwrap(), 5);
        device.unmap(0x4_0000_0000, 0x3000);
        assert!(device.read::<u32>(0x4_0000_0010).is_err());
    }

    #[test]
    fn map_process_rejects_holes() {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let process = AddressSpace::new(physical.clone(), 39);
        process.map(0x1000, 0, 0x1000, MemoryPermission::READ_WRITE);
        let device = DeviceAddressSpace::new(physical, 34);
        assert_eq!(
            device.map_process(0, &process, 0x1000, 0x2000),
            Err(MemoryFault::unmapped(0x2000))
        );
        assert!(!device.is_mapped(0));
    }

    #[test]
    fn concurrent_map_and_translate() {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let device = Arc::new(DeviceAddressSpace::new(physical, 35));
        device.map(0, 0, 0x10_0000);
        let reader = {
            let device = device.clone();
            std::thread::spawn(move || {
                for i in 0..200_000u64 {
                    let addr = (i * 0x1000) % 0x10_0000;
                    if let Some(ptr) = device.translate(addr) {
                        let pa = device.physical().physical_address(ptr).unwrap();
                        assert_eq!(pa & PAGE_MASK, 0);
                    }
                }
            })
        };
        for round in 0..200u64 {
            device.unmap(0x8_0000, 0x8_0000);
            device.map(0x8_0000, (round % 8) * 0x8_0000, 0x8_0000);
        }
        reader.join().unwrap();
    }
}
