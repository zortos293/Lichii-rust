//! The SMMU: one device address space shared by the GPU and the other host1x clients.
//!
//! Device pages point straight at guest DRAM. Every page entry is an atomic host pointer, so
//! translation never takes a lock and never goes through a shared cache: a service thread can
//! map or unmap while the GPU thread reads (NX2_NOTES row 125). Page tables and the DRAM they
//! point into live as long as the [`DeviceMemory`], so a pointer read just before an unmap still
//! points at valid host memory.

use std::collections::BTreeMap;
use std::ops::Bound;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use lichii_common::{DAddr, PAGE_BITS, PAGE_MASK, PAGE_SIZE, VAddr, align_up};
use lichii_memory::{AddressSpace, PhysicalMemory};
use parking_lot::Mutex;

/// Width of the device address space: 64 GiB, enough for Switch 2 titles that map most of
/// their memory twice plus GPU-only regions (NX2_NOTES rows 42 and 85).
pub const DEVICE_ADDRESS_BITS: u32 = 36;
/// Size of the device address space in bytes.
pub const DEVICE_ADDRESS_SPACE_SIZE: u64 = 1 << DEVICE_ADDRESS_BITS;

const L2_BITS: u32 = 12;
const L2_ENTRIES: usize = 1 << L2_BITS;
const L1_ENTRIES: usize = 1 << (DEVICE_ADDRESS_BITS - PAGE_BITS - L2_BITS);

type L2Table = [AtomicUsize; L2_ENTRIES];

/// Why a device memory map, unmap or free was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DeviceMemoryError {
    #[error("device range {daddr:#x}+{size:#x} is unaligned, empty or outside the device space")]
    InvalidRange { daddr: DAddr, size: u64 },
    #[error("process address {va:#x} is unaligned")]
    UnalignedSource { va: VAddr },
    #[error("process address {va:#x} is not mapped")]
    SourceUnmapped { va: VAddr },
    #[error("the address space is backed by other guest DRAM than the device")]
    ForeignPhysicalMemory,
    #[error("device range {daddr:#x}+{size:#x} is not allocated")]
    NotAllocated { daddr: DAddr, size: u64 },
}

/// A device access that touched an unmapped page; carries the first such address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("device address {0:#x} is not mapped")]
pub struct DeviceFault(pub DAddr);

/// The SMMU shared by the GPU and other devices: device address pages to host pointers.
pub struct DeviceMemory {
    l1: Box<[AtomicPtr<L2Table>]>,
    grow: Mutex<()>,
    free: Mutex<BTreeMap<DAddr, u64>>,
    physical: Arc<PhysicalMemory>,
    generation: AtomicU64,
}

impl DeviceMemory {
    /// A device space over `physical`; only address spaces backed by it can be mapped.
    pub fn new(physical: Arc<PhysicalMemory>) -> Self {
        let l1 = (0..L1_ENTRIES)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self {
            l1,
            grow: Mutex::new(()),
            free: Mutex::new(BTreeMap::from([(
                PAGE_SIZE,
                DEVICE_ADDRESS_SPACE_SIZE - PAGE_SIZE,
            )])),
            physical,
            generation: AtomicU64::new(0),
        }
    }

    /// Reserves `size` bytes (rounded up to pages) of device addresses aligned to `align`, a
    /// power of two of at least a page. Freed ranges are reused. Never returns 0.
    pub fn allocate(&self, size: u64, align: u64) -> Option<DAddr> {
        assert!(
            align.is_power_of_two() && align >= PAGE_SIZE,
            "bad device alignment {align:#x}"
        );
        let size = align_up(size, PAGE_SIZE);
        if size == 0 {
            return None;
        }
        let mut free = self.free.lock();
        let (start, len, base) = free.iter().find_map(|(&start, &len)| {
            let base = start.checked_next_multiple_of(align)?;
            (base.checked_add(size)? <= start + len).then_some((start, len, base))
        })?;
        free.remove(&start);
        if base > start {
            free.insert(start, base - start);
        }
        let end = base + size;
        if end < start + len {
            free.insert(end, start + len - end);
        }
        Some(base)
    }

    /// Returns a range [`allocate`](Self::allocate) handed out (or part of one) for reuse.
    pub fn free(&self, daddr: DAddr, size: u64) -> Result<(), DeviceMemoryError> {
        let size = align_up(size, PAGE_SIZE);
        Self::check_range(daddr, size)?;
        let end = daddr + size;
        let mut free = self.free.lock();
        let prev = free.range(..=daddr).next_back().map(|(&s, &l)| (s, l));
        let next = free
            .range((Bound::Excluded(daddr), Bound::Unbounded))
            .next()
            .map(|(&s, &l)| (s, l));
        if prev.is_some_and(|(s, l)| s + l > daddr) || next.is_some_and(|(s, _)| s < end) {
            return Err(DeviceMemoryError::NotAllocated { daddr, size });
        }
        let mut start = daddr;
        let mut stop = end;
        if let Some((s, l)) = prev
            && s + l == daddr
        {
            free.remove(&s);
            start = s;
        }
        if let Some((s, l)) = next
            && s == end
        {
            free.remove(&s);
            stop = s + l;
        }
        free.insert(start, stop - start);
        Ok(())
    }

    /// Bytes of device address space currently allocated.
    pub fn allocated_bytes(&self) -> u64 {
        let free: u64 = self.free.lock().values().sum();
        DEVICE_ADDRESS_SPACE_SIZE - PAGE_SIZE - free
    }

    /// Points device pages [`daddr`, +`size`) at the DRAM behind process memory [`va`, +`size`).
    /// Every source page must be mapped; nothing changes otherwise.
    pub fn map(
        &self,
        daddr: DAddr,
        space: &AddressSpace,
        va: VAddr,
        size: u64,
    ) -> Result<(), DeviceMemoryError> {
        Self::check_range(daddr, size)?;
        if va & PAGE_MASK != 0 {
            return Err(DeviceMemoryError::UnalignedSource { va });
        }
        if !Arc::ptr_eq(space.physical(), &self.physical) {
            return Err(DeviceMemoryError::ForeignPhysicalMemory);
        }
        let hosts = (0..size / PAGE_SIZE)
            .map(|page| {
                let source = va + page * PAGE_SIZE;
                space
                    .translate(source)
                    .map(|(host, _)| host as usize)
                    .ok_or(DeviceMemoryError::SourceUnmapped { va: source })
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (page, host) in hosts.into_iter().enumerate() {
            self.entry_for_write(daddr + page as u64 * PAGE_SIZE)
                .store(host, Ordering::Release);
        }
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }

    pub fn unmap(&self, daddr: DAddr, size: u64) -> Result<(), DeviceMemoryError> {
        Self::check_range(daddr, size)?;
        for page in 0..size / PAGE_SIZE {
            if let Some(entry) = self.entry(daddr + page * PAGE_SIZE) {
                entry.store(0, Ordering::Release);
            }
        }
        self.generation.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Changes after every map and unmap; a host pointer from [`translate`](Self::translate)
    /// shows the current mapping only while this stays the same.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Host pointer of the byte at `daddr`.
    #[inline]
    pub fn translate(&self, daddr: DAddr) -> Option<*mut u8> {
        let host = self.entry(daddr)?.load(Ordering::Acquire);
        if host == 0 {
            return None;
        }
        // SAFETY: the offset stays inside the mapped page.
        Some(unsafe { (host as *mut u8).add((daddr & PAGE_MASK) as usize) })
    }

    #[inline]
    pub fn read<T: bytemuck::Pod>(&self, daddr: DAddr) -> Result<T, DeviceFault> {
        if (daddr & PAGE_MASK) + size_of::<T>() as u64 <= PAGE_SIZE {
            let ptr = self.translate(daddr).ok_or(DeviceFault(daddr))?;
            // SAFETY: the whole value is inside one mapped page.
            return Ok(unsafe { std::ptr::read_unaligned(ptr as *const T) });
        }
        let mut value = T::zeroed();
        self.read_block(daddr, bytemuck::bytes_of_mut(&mut value))?;
        Ok(value)
    }

    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, daddr: DAddr, value: T) -> Result<(), DeviceFault> {
        if (daddr & PAGE_MASK) + size_of::<T>() as u64 <= PAGE_SIZE {
            let ptr = self.translate(daddr).ok_or(DeviceFault(daddr))?;
            // SAFETY: the whole value is inside one mapped page.
            unsafe { std::ptr::write_unaligned(ptr as *mut T, value) };
            return Ok(());
        }
        self.write_block(daddr, bytemuck::bytes_of(&value))
    }

    /// Reads `out.len()` bytes. Unmapped pages read as zero and the first one is the error.
    pub fn read_block(&self, daddr: DAddr, out: &mut [u8]) -> Result<(), DeviceFault> {
        self.for_each_page(daddr, out.len() as u64, |addr, host, offset, len| {
            let chunk = &mut out[offset..offset + len];
            let Some(ptr) = host else {
                chunk.fill(0);
                return Err(DeviceFault(addr));
            };
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(ptr, chunk.as_mut_ptr(), len) };
            Ok(())
        })
    }

    /// Writes `data`. Writes to unmapped pages are dropped and the first one is the error.
    pub fn write_block(&self, daddr: DAddr, data: &[u8]) -> Result<(), DeviceFault> {
        self.for_each_page(daddr, data.len() as u64, |addr, host, offset, len| {
            let ptr = host.ok_or(DeviceFault(addr))?;
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(data[offset..].as_ptr(), ptr, len) };
            Ok(())
        })
    }

    /// Sets `size` bytes to `byte`, skipping unmapped pages like [`write_block`](Self::write_block).
    pub fn fill(&self, daddr: DAddr, size: u64, byte: u8) -> Result<(), DeviceFault> {
        self.for_each_page(daddr, size, |addr, host, _, len| {
            let ptr = host.ok_or(DeviceFault(addr))?;
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::write_bytes(ptr, byte, len) };
            Ok(())
        })
    }

    /// Copies `size` bytes from `src` to `dst`. Unmapped source pages copy zeros, unmapped
    /// destination pages drop the bytes; the first unmapped address is the error.
    pub fn copy_block(&self, dst: DAddr, src: DAddr, size: u64) -> Result<(), DeviceFault> {
        let mut fault = None;
        let mut done = 0;
        while done < size {
            let (from, to) = (src + done, dst + done);
            let len = (PAGE_SIZE - (from & PAGE_MASK))
                .min(PAGE_SIZE - (to & PAGE_MASK))
                .min(size - done) as usize;
            match (self.translate(from), self.translate(to)) {
                // SAFETY: both pointers are valid for len bytes inside their mapped pages.
                (Some(s), Some(d)) => unsafe { std::ptr::copy(s, d, len) },
                (None, Some(d)) => {
                    // SAFETY: d is valid for len bytes inside its mapped page.
                    unsafe { std::ptr::write_bytes(d, 0, len) };
                    fault.get_or_insert(DeviceFault(from));
                }
                (_, None) => {
                    fault.get_or_insert(DeviceFault(to));
                }
            }
            done += len as u64;
        }
        fault.map_or(Ok(()), Err)
    }

    /// Calls `f(daddr, host, len)` for each run of [`daddr`, +`size`) that is contiguous in host
    /// memory, or unmapped throughout with `host` `None`, in order.
    pub fn for_each_host_span(
        &self,
        daddr: DAddr,
        size: u64,
        mut f: impl FnMut(DAddr, Option<*mut u8>, usize),
    ) {
        let mut run: Option<(DAddr, Option<*mut u8>, usize)> = None;
        let _ = self.for_each_page(daddr, size, |addr, host, _, len| {
            if let Some((_, run_host, run_len)) = &mut run {
                let extends = match (*run_host, host) {
                    (Some(a), Some(b)) => a.wrapping_add(*run_len) == b,
                    (None, None) => true,
                    _ => false,
                };
                if extends {
                    *run_len += len;
                    return Ok(());
                }
            }
            if let Some((a, h, l)) = run.replace((addr, host, len)) {
                f(a, h, l);
            }
            Ok(())
        });
        if let Some((a, h, l)) = run {
            f(a, h, l);
        }
    }

    fn for_each_page(
        &self,
        daddr: DAddr,
        size: u64,
        mut f: impl FnMut(DAddr, Option<*mut u8>, usize, usize) -> Result<(), DeviceFault>,
    ) -> Result<(), DeviceFault> {
        let mut fault = None;
        let mut done = 0;
        while done < size {
            let addr = daddr.wrapping_add(done);
            let len = (PAGE_SIZE - (addr & PAGE_MASK)).min(size - done);
            if let Err(e) = f(addr, self.translate(addr), done as usize, len as usize) {
                fault.get_or_insert(e);
            }
            done += len;
        }
        fault.map_or(Ok(()), Err)
    }

    fn check_range(daddr: DAddr, size: u64) -> Result<(), DeviceMemoryError> {
        let fits = daddr
            .checked_add(size)
            .is_some_and(|end| end <= DEVICE_ADDRESS_SPACE_SIZE);
        if (daddr | size) & PAGE_MASK != 0 || size == 0 || daddr == 0 || !fits {
            return Err(DeviceMemoryError::InvalidRange { daddr, size });
        }
        Ok(())
    }

    #[inline]
    fn entry(&self, daddr: DAddr) -> Option<&AtomicUsize> {
        let page = (daddr >> PAGE_BITS) as usize;
        let table = self.l1.get(page >> L2_BITS)?.load(Ordering::Acquire);
        // SAFETY: tables are never freed while the device memory lives.
        unsafe { table.as_ref() }.map(|t| &t[page & (L2_ENTRIES - 1)])
    }

    fn entry_for_write(&self, daddr: DAddr) -> &AtomicUsize {
        let page = (daddr >> PAGE_BITS) as usize;
        let slot = &self.l1[page >> L2_BITS];
        if slot.load(Ordering::Acquire).is_null() {
            let _guard = self.grow.lock();
            if slot.load(Ordering::Acquire).is_null() {
                let table: Box<L2Table> = Box::new(std::array::from_fn(|_| AtomicUsize::new(0)));
                slot.store(Box::into_raw(table), Ordering::Release);
            }
        }
        // SAFETY: just ensured non-null; tables live as long as self.
        let table = unsafe { &*slot.load(Ordering::Acquire) };
        &table[page & (L2_ENTRIES - 1)]
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        for slot in self.l1.iter() {
            let table = slot.load(Ordering::Acquire);
            if !table.is_null() {
                // SAFETY: created by Box::into_raw in entry_for_write and dropped once here.
                drop(unsafe { Box::from_raw(table) });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_memory::MemoryPermission;

    fn setup() -> (AddressSpace, DeviceMemory) {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        (
            AddressSpace::new(physical.clone(), 39),
            DeviceMemory::new(physical),
        )
    }

    #[test]
    fn maps_process_memory_and_reads_through_it() {
        let (space, device) = setup();
        space.map(0x10_0000, 0x3000, 0x1000, MemoryPermission::READ_WRITE);
        space.map(0x10_1000, 0x8000, 0x1000, MemoryPermission::READ_WRITE);
        let daddr = device.allocate(0x2000, 0x20000).unwrap();
        device.map(daddr, &space, 0x10_0000, 0x2000).unwrap();
        space.write(0x10_0ffc, 0x1122_3344_5566_7788u64).unwrap();
        assert_eq!(
            device.read::<u64>(daddr + 0xffc).unwrap(),
            0x1122_3344_5566_7788
        );
        device.write(daddr + 0x1004, 0xabcdu32).unwrap();
        assert_eq!(space.read::<u32>(0x10_1004).unwrap(), 0xabcd);
        let mut spans = Vec::new();
        device.for_each_host_span(daddr, 0x3000, |a, h, l| spans.push((a, h.is_some(), l)));
        assert_eq!(
            spans,
            [
                (daddr, true, 0x1000),
                (daddr + 0x1000, true, 0x1000),
                (daddr + 0x2000, false, 0x1000)
            ]
        );
        device.unmap(daddr, 0x2000).unwrap();
        assert!(device.translate(daddr).is_none());
        let mut out = [0xffu8; 8];
        assert_eq!(device.read_block(daddr, &mut out), Err(DeviceFault(daddr)));
        assert_eq!(out, [0; 8]);
    }

    #[test]
    fn map_refuses_unmapped_or_foreign_sources() {
        let (space, device) = setup();
        space.map(0x10_0000, 0, 0x1000, MemoryPermission::READ_WRITE);
        assert_eq!(
            device.map(0x2_0000, &space, 0x10_0000, 0x2000),
            Err(DeviceMemoryError::SourceUnmapped { va: 0x10_1000 })
        );
        assert!(device.translate(0x2_0000).is_none());
        let other = AddressSpace::new(Arc::new(PhysicalMemory::new(1 << 16).unwrap()), 39);
        assert_eq!(
            device.map(0x2_0000, &other, 0, 0x1000),
            Err(DeviceMemoryError::ForeignPhysicalMemory)
        );
    }

    #[test]
    fn allocator_reuses_and_coalesces_freed_ranges() {
        let (_, device) = setup();
        let a = device.allocate(0x1_0000, 0x2_0000).unwrap();
        let b = device.allocate(0x1_0000, 0x2_0000).unwrap();
        assert_ne!(a, 0);
        assert!(b >= a + 0x1_0000);
        device.free(a, 0x1_0000).unwrap();
        assert_eq!(device.allocate(0x1_0000, 0x2_0000), Some(a));
        assert_eq!(
            device.free(b + 0x10_0000, 0x1000),
            Err(DeviceMemoryError::NotAllocated {
                daddr: b + 0x10_0000,
                size: 0x1000
            })
        );
        device.free(a, 0x1_0000).unwrap();
        device.free(b, 0x1_0000).unwrap();
        assert_eq!(device.allocated_bytes(), 0);
        let whole = device.allocate(DEVICE_ADDRESS_SPACE_SIZE - PAGE_SIZE, PAGE_SIZE);
        assert_eq!(whole, Some(PAGE_SIZE));
        assert_eq!(device.allocate(PAGE_SIZE, PAGE_SIZE), None);
    }

    #[test]
    fn copy_block_crosses_pages() {
        let (space, device) = setup();
        space.map(0x10_0000, 0x10000, 0x4000, MemoryPermission::READ_WRITE);
        device.map(0x10_0000, &space, 0x10_0000, 0x4000).unwrap();
        let data: Vec<u8> = (0..0x1800u32).map(|i| i as u8).collect();
        device.write_block(0x10_0800, &data).unwrap();
        device.copy_block(0x10_2400, 0x10_0800, 0x1800).unwrap();
        let mut back = vec![0; 0x1800];
        device.read_block(0x10_2400, &mut back).unwrap();
        assert_eq!(back, data);
        device.fill(0x10_0ff0, 0x20, 0xaa).unwrap();
        assert_eq!(device.read::<u32>(0x10_0ffe).unwrap(), 0xaaaa_aaaa);
    }
}
