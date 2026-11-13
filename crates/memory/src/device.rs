use std::sync::{Arc, OnceLock};

use lichii_common::{DAddr, PAGE_MASK, PAGE_SIZE, PAddr, VAddr};
use parking_lot::Mutex;

use crate::page_table::PageTable;
use crate::tracking::{CachedPages, ReverseMap};
use crate::{AddressSpace, MemoryFault, MemoryPermission, PhysicalMemory};

/// A device (SMMU) address space: guest memory as the GPU and other DMA engines see it.
///
/// Device addresses map to DRAM pages, usually pinned from a process address space
/// ([`DeviceAddressSpace::map_process`]). Translation is a lock-free table walk, safe to call from
/// the GPU thread while a service thread maps or unmaps; there is no translation cache to
/// invalidate. GPU caches learn about CPU writes through it ([`crate::tracking`]).
pub struct DeviceAddressSpace {
    table: PageTable,
    physical: Arc<PhysicalMemory>,
    /// Serialises map and unmap; the device pages mapping each physical page.
    update: Mutex<ReverseMap>,
    process: OnceLock<Arc<AddressSpace>>,
}

impl DeviceAddressSpace {
    /// `address_space_bits` is 34 on Switch 1 (Tegra X1 SMMU) and 35 on Switch 2.
    pub fn new(physical: Arc<PhysicalMemory>, address_space_bits: u32) -> Self {
        Self {
            table: PageTable::new(address_space_bits),
            update: Mutex::new(ReverseMap::new(physical.size())),
            physical,
            process: OnceLock::new(),
        }
    }

    /// Names the process whose CPU writes GPU caches over this space must see. Called once.
    pub fn attach_process(&self, process: Arc<AddressSpace>) {
        assert!(
            Arc::ptr_eq(process.physical(), &self.physical),
            "device and process address spaces use different DRAM"
        );
        assert!(
            self.process.set(process).is_ok(),
            "a process is already attached to this device address space"
        );
    }

    /// Tracks the physical pages behind the mapped pages of `[daddr, daddr + size)` for CPU writes
    /// until the guard is dropped. A cache takes it before it reads the bytes it copies, and a
    /// new one for every upload; without an attached process nothing is tracked.
    pub fn cache_pages(&self, daddr: DAddr, size: u64) -> CachedPages {
        let mut runs: Vec<(PAddr, u64)> = Vec::new();
        let end = daddr.saturating_add(size);
        let mut page = daddr & !PAGE_MASK;
        while page < end {
            if let Some(pa) = self.physical_address(page) {
                match runs.last_mut() {
                    Some((start, len)) if *start + *len == pa => *len += PAGE_SIZE,
                    _ => runs.push((pa, PAGE_SIZE)),
                }
            }
            page += PAGE_SIZE;
        }
        CachedPages::new(self.process.get().cloned(), runs)
    }

    /// Whether the attached process wrote a tracked page since the last
    /// [`DeviceAddressSpace::take_cpu_writes`]. One atomic load.
    #[inline]
    pub fn has_cpu_writes(&self) -> bool {
        self.process.get().is_some_and(|p| p.has_cpu_writes())
    }

    /// Calls `f` with each device range (address, size) the attached process wrote while it was
    /// tracked, and forgets them. Pages no device address maps are left out.
    pub fn take_cpu_writes(&self, mut f: impl FnMut(DAddr, u64)) {
        let Some(process) = self.process.get() else {
            return;
        };
        let mut pages: Vec<DAddr> = Vec::new();
        process.take_cpu_writes(|pa, size| {
            let reverse = self.update.lock();
            for page in (pa..pa + size).step_by(PAGE_SIZE as usize) {
                reverse.for_each(page, |daddr| pages.push(daddr));
            }
        });
        pages.sort_unstable();
        pages.dedup();
        let mut run: Option<(DAddr, u64)> = None;
        for daddr in pages {
            match &mut run {
                Some((start, len)) if *start + *len == daddr => *len += PAGE_SIZE,
                _ => {
                    if let Some((start, len)) = run.take() {
                        f(start, len);
                    }
                    run = Some((daddr, PAGE_SIZE));
                }
            }
        }
        if let Some((start, len)) = run {
            f(start, len);
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
        let mut reverse = self.update.lock();
        for page in 0..size / PAGE_SIZE {
            self.set_page(
                &mut reverse,
                daddr + page * PAGE_SIZE,
                pa + page * PAGE_SIZE,
            );
        }
    }

    fn set_page(&self, reverse: &mut ReverseMap, daddr: DAddr, pa: PAddr) {
        if let Some(old) = self.physical_address(daddr) {
            reverse.remove(old, daddr);
        }
        self.table.set(
            daddr,
            self.physical.pointer(pa),
            MemoryPermission::READ_WRITE,
        );
        reverse.insert(pa, daddr);
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
        let pages = (0..size / PAGE_SIZE)
            .map(|page| {
                let addr = va + page * PAGE_SIZE;
                process
                    .physical_address(addr)
                    .ok_or(MemoryFault::unmapped(addr))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut reverse = self.update.lock();
        for (page, pa) in pages.into_iter().enumerate() {
            self.set_page(&mut reverse, daddr + page as u64 * PAGE_SIZE, pa);
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
        let mut reverse = self.update.lock();
        for page in 0..size / PAGE_SIZE {
            let addr = daddr + page * PAGE_SIZE;
            if let Some(old) = self.physical_address(addr) {
                reverse.remove(old, addr);
            }
            self.table.clear(addr);
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
        self.table.read(daddr, MemoryPermission::NONE)
    }

    /// Writes a value; it may straddle pages. Fails on the first unmapped page, after writing the
    /// bytes before it.
    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, daddr: DAddr, value: T) -> Result<(), MemoryFault> {
        self.table.write(daddr, value, MemoryPermission::NONE)
    }

    /// Copies `out.len()` bytes from `daddr`. Fails on the first unmapped page.
    pub fn read_bytes(&self, daddr: DAddr, out: &mut [u8]) -> Result<(), MemoryFault> {
        self.table.read_bytes(daddr, out, MemoryPermission::NONE)
    }

    /// Copies `data` to `daddr`. Fails on the first unmapped page, after writing the bytes
    /// before it.
    pub fn write_bytes(&self, daddr: DAddr, data: &[u8]) -> Result<(), MemoryFault> {
        self.table.write_bytes(daddr, data, MemoryPermission::NONE)
    }

    /// Sets `size` bytes at `daddr` to `byte`. Fails on the first unmapped page.
    pub fn fill(&self, daddr: DAddr, size: u64, byte: u8) -> Result<(), MemoryFault> {
        self.table.fill(daddr, size, byte, MemoryPermission::NONE)
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
