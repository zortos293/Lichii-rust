use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use lichii_common::{PAGE_BITS, PAGE_MASK, PAGE_SIZE, PAddr, VAddr};
use parking_lot::Mutex;

use crate::PhysicalMemory;

const L2_BITS: u32 = 14;
const L2_ENTRIES: usize = 1 << L2_BITS;
const PERM_MASK: usize = 0xfff;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct MemoryPermission(pub u8);

impl MemoryPermission {
    pub const NONE: Self = Self(0);
    pub const READ: Self = Self(1);
    pub const WRITE: Self = Self(2);
    pub const EXECUTE: Self = Self(4);
    pub const READ_WRITE: Self = Self(3);
    pub const READ_EXECUTE: Self = Self(5);

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultKind {
    Unmapped,
    Permission,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("guest memory fault at {addr:#x}: {kind:?}")]
pub struct MemoryFault {
    pub addr: VAddr,
    pub kind: FaultKind,
}

type L2Table = [AtomicUsize; L2_ENTRIES];

/// A process's guest virtual address space.
pub struct AddressSpace {
    bits: u32,
    l1: Box<[AtomicPtr<L2Table>]>,
    grow: Mutex<()>,
    physical: Arc<PhysicalMemory>,
}

impl AddressSpace {
    /// \`address_space_bits\` is 32, 36 or 39 for Horizon processes.
    pub fn new(physical: Arc<PhysicalMemory>, address_space_bits: u32) -> Self {
        let l1_len = 1usize << address_space_bits.saturating_sub(PAGE_BITS + L2_BITS);
        let l1 = (0..l1_len)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self {
            bits: address_space_bits,
            l1,
            grow: Mutex::new(()),
            physical,
        }
    }

    pub fn address_space_bits(&self) -> u32 {
        self.bits
    }

    pub fn physical(&self) -> &Arc<PhysicalMemory> {
        &self.physical
    }

    /// Maps \`size\` bytes of DRAM at \`pa\` to \`va\`. Both must be page aligned.
    pub fn map(&self, va: VAddr, pa: PAddr, size: u64, perm: MemoryPermission) {
        assert_eq!(
            (va | pa | size) & PAGE_MASK,
            0,
            "unaligned map {va:#x} {pa:#x} {size:#x}"
        );
        assert!(
            pa + size <= self.physical.size(),
            "map past the end of DRAM"
        );
        for page in 0..size / PAGE_SIZE {
            let host = self.physical.pointer(pa + page * PAGE_SIZE) as usize;
            self.entry_for_write(va + page * PAGE_SIZE)
                .store(host | perm.0 as usize, Ordering::Release);
        }
    }

    pub fn unmap(&self, va: VAddr, size: u64) {
        assert_eq!(
            (va | size) & PAGE_MASK,
            0,
            "unaligned unmap {va:#x} {size:#x}"
        );
        for page in 0..size / PAGE_SIZE {
            if let Some(entry) = self.entry(va + page * PAGE_SIZE) {
                entry.store(0, Ordering::Release);
            }
        }
    }

    pub fn protect(&self, va: VAddr, size: u64, perm: MemoryPermission) {
        for page in 0..size.div_ceil(PAGE_SIZE) {
            if let Some(entry) = self.entry(va + page * PAGE_SIZE) {
                let old = entry.load(Ordering::Acquire);
                if old != 0 {
                    entry.store((old & !PERM_MASK) | perm.0 as usize, Ordering::Release);
                }
            }
        }
    }

    /// Host pointer and permission for the byte at \`va\`.
    #[inline]
    pub fn translate(&self, va: VAddr) -> Option<(*mut u8, MemoryPermission)> {
        let raw = self.entry(va)?.load(Ordering::Acquire);
        if raw == 0 {
            return None;
        }
        let host = (raw & !PERM_MASK) as *mut u8;
        // SAFETY: the offset stays inside the mapped page.
        let ptr = unsafe { host.add((va & PAGE_MASK) as usize) };
        Some((ptr, MemoryPermission(raw as u8 & 7)))
    }

    pub fn is_mapped(&self, va: VAddr) -> bool {
        self.translate(va).is_some()
    }

    #[inline]
    pub fn read<T: bytemuck::Pod>(&self, va: VAddr) -> Result<T, MemoryFault> {
        let size = std::mem::size_of::<T>() as u64;
        if (va & PAGE_MASK) + size <= PAGE_SIZE {
            let (ptr, _) = self.translate(va).ok_or(MemoryFault::unmapped(va))?;
            // SAFETY: the whole value is inside one mapped page.
            return Ok(unsafe { std::ptr::read_unaligned(ptr as *const T) });
        }
        let mut value = T::zeroed();
        self.read_bytes(va, bytemuck::bytes_of_mut(&mut value))?;
        Ok(value)
    }

    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, va: VAddr, value: T) -> Result<(), MemoryFault> {
        let size = std::mem::size_of::<T>() as u64;
        if (va & PAGE_MASK) + size <= PAGE_SIZE {
            let (ptr, _) = self.translate(va).ok_or(MemoryFault::unmapped(va))?;
            // SAFETY: the whole value is inside one mapped page.
            unsafe { std::ptr::write_unaligned(ptr as *mut T, value) };
            return Ok(());
        }
        self.write_bytes(va, bytemuck::bytes_of(&value))
    }

    pub fn read_bytes(&self, va: VAddr, out: &mut [u8]) -> Result<(), MemoryFault> {
        self.for_each_chunk(va, out.len() as u64, |ptr, offset, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(ptr, out[offset..].as_mut_ptr(), len) };
        })
    }

    pub fn write_bytes(&self, va: VAddr, data: &[u8]) -> Result<(), MemoryFault> {
        self.for_each_chunk(va, data.len() as u64, |ptr, offset, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(data[offset..].as_ptr(), ptr, len) };
        })
    }

    pub fn fill(&self, va: VAddr, size: u64, byte: u8) -> Result<(), MemoryFault> {
        self.for_each_chunk(va, size, |ptr, _, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::write_bytes(ptr, byte, len) };
        })
    }

    /// Reads a NUL-terminated string of at most \`max_len\` bytes.
    pub fn read_cstring(&self, va: VAddr, max_len: usize) -> Result<String, MemoryFault> {
        let mut bytes = Vec::new();
        for i in 0..max_len as u64 {
            let b: u8 = self.read(va + i)?;
            if b == 0 {
                break;
            }
            bytes.push(b);
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn for_each_chunk(
        &self,
        va: VAddr,
        size: u64,
        mut f: impl FnMut(*mut u8, usize, usize),
    ) -> Result<(), MemoryFault> {
        let mut done = 0u64;
        while done < size {
            let addr = va + done;
            let len = (PAGE_SIZE - (addr & PAGE_MASK)).min(size - done);
            let (ptr, _) = self.translate(addr).ok_or(MemoryFault::unmapped(addr))?;
            f(ptr, done as usize, len as usize);
            done += len;
        }
        Ok(())
    }

    #[inline]
    fn entry(&self, va: VAddr) -> Option<&AtomicUsize> {
        let page = (va >> PAGE_BITS) as usize;
        let table = self.l1.get(page >> L2_BITS)?.load(Ordering::Acquire);
        // SAFETY: tables are never freed while the address space lives.
        unsafe { table.as_ref() }.map(|t| &t[page & (L2_ENTRIES - 1)])
    }

    fn entry_for_write(&self, va: VAddr) -> &AtomicUsize {
        let page = (va >> PAGE_BITS) as usize;
        let slot = self
            .l1
            .get(page >> L2_BITS)
            .unwrap_or_else(|| panic!("{va:#x} outside the {}-bit address space", self.bits));
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

impl MemoryFault {
    pub const fn unmapped(addr: VAddr) -> Self {
        Self {
            addr,
            kind: FaultKind::Unmapped,
        }
    }
}

impl Drop for AddressSpace {
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

    fn space() -> AddressSpace {
        AddressSpace::new(Arc::new(PhysicalMemory::new(1 << 24).unwrap()), 39)
    }

    #[test]
    fn reads_back_writes_across_pages() {
        let mem = space();
        mem.map(0x8000_0000, 0x1000, 0x3000, MemoryPermission::READ_WRITE);
        mem.write(0x8000_0ffe, 0x1122_3344_5566_7788u64).unwrap();
        assert_eq!(mem.read::<u64>(0x8000_0ffe).unwrap(), 0x1122_3344_5566_7788);
        assert_eq!(mem.read::<u8>(0x8000_1000).unwrap(), 0x66);
    }

    #[test]
    fn unmapped_access_faults() {
        let mem = space();
        assert_eq!(
            mem.read::<u32>(0x1000).unwrap_err(),
            MemoryFault::unmapped(0x1000)
        );
    }

    #[test]
    fn aliases_share_physical_pages() {
        let mem = space();
        mem.map(0x10_0000, 0x4000, 0x1000, MemoryPermission::READ_WRITE);
        mem.map(0x20_0000, 0x4000, 0x1000, MemoryPermission::READ);
        mem.write(0x10_0010, 7u32).unwrap();
        assert_eq!(mem.read::<u32>(0x20_0010).unwrap(), 7);
    }
}
