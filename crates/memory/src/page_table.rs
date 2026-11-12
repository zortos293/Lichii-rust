use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use lichii_common::{PAGE_BITS, PAGE_MASK, PAGE_SIZE};
use parking_lot::Mutex;

use crate::{MemoryFault, MemoryPermission};

const L2_BITS: u32 = 14;
const L2_ENTRIES: usize = 1 << L2_BITS;
const PERM_MASK: usize = 0xfff;

type L2Table = [AtomicUsize; L2_ENTRIES];

/// Two-level table from page number to `host pointer | permission`.
///
/// Lookups are lock-free atomic loads, so any thread can translate while another maps or unmaps:
/// a reader sees either the old or the new entry, never a torn one, and the host pages behind an
/// entry stay valid for the life of the DRAM backing. No per-thread translation cache is needed,
/// which removes the invalidation race of NX2_NOTES row 125.
pub(crate) struct PageTable {
    bits: u32,
    l1: Box<[AtomicPtr<L2Table>]>,
    grow: Mutex<()>,
}

impl PageTable {
    pub(crate) fn new(bits: u32) -> Self {
        let l1_len = 1usize << bits.saturating_sub(PAGE_BITS + L2_BITS);
        let l1 = (0..l1_len)
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self {
            bits,
            l1,
            grow: Mutex::new(()),
        }
    }

    pub(crate) fn bits(&self) -> u32 {
        self.bits
    }

    pub(crate) fn set(&self, addr: u64, host: *mut u8, perm: MemoryPermission) {
        self.entry_for_write(addr)
            .store(host as usize | perm.0 as usize, Ordering::Release);
    }

    pub(crate) fn clear(&self, addr: u64) {
        if let Some(entry) = self.entry(addr) {
            entry.store(0, Ordering::Release);
        }
    }

    /// Changes the permission of a mapped page. Returns whether the page was mapped.
    pub(crate) fn set_permission(&self, addr: u64, perm: MemoryPermission) -> bool {
        let Some(entry) = self.entry(addr) else {
            return false;
        };
        let old = entry.load(Ordering::Acquire);
        if old == 0 {
            return false;
        }
        entry.store((old & !PERM_MASK) | perm.0 as usize, Ordering::Release);
        true
    }

    #[inline]
    pub(crate) fn translate(&self, addr: u64) -> Option<(*mut u8, MemoryPermission)> {
        let raw = self.entry(addr)?.load(Ordering::Acquire);
        if raw == 0 {
            return None;
        }
        let host = (raw & !PERM_MASK) as *mut u8;
        // SAFETY: the offset stays inside the mapped page.
        let ptr = unsafe { host.add((addr & PAGE_MASK) as usize) };
        Some((ptr, MemoryPermission(raw as u8 & 7)))
    }

    /// Host pointer for `[addr, addr + size)` when every page is mapped and the host pages are
    /// contiguous, so the whole range can be accessed as one slice.
    pub(crate) fn contiguous(&self, addr: u64, size: u64) -> Option<*mut u8> {
        let (first, _) = self.translate(addr)?;
        if size == 0 {
            return Some(first);
        }
        let start_page = addr & !PAGE_MASK;
        let last_page = (addr + size - 1) & !PAGE_MASK;
        let mut page = start_page + PAGE_SIZE;
        while page <= last_page {
            let (ptr, _) = self.translate(page)?;
            // SAFETY: pointer arithmetic only, compared and never dereferenced.
            let expected = unsafe { first.add((page - addr) as usize) };
            if ptr != expected {
                return None;
            }
            page += PAGE_SIZE;
        }
        Some(first)
    }

    #[inline]
    pub(crate) fn read<T: bytemuck::Pod>(&self, addr: u64) -> Result<T, MemoryFault> {
        let size = std::mem::size_of::<T>() as u64;
        if (addr & PAGE_MASK) + size <= PAGE_SIZE {
            let (ptr, _) = self.translate(addr).ok_or(MemoryFault::unmapped(addr))?;
            // SAFETY: the whole value is inside one mapped page.
            return Ok(unsafe { std::ptr::read_unaligned(ptr as *const T) });
        }
        let mut value = T::zeroed();
        self.read_bytes(addr, bytemuck::bytes_of_mut(&mut value))?;
        Ok(value)
    }

    #[inline]
    pub(crate) fn write<T: bytemuck::Pod>(&self, addr: u64, value: T) -> Result<(), MemoryFault> {
        let size = std::mem::size_of::<T>() as u64;
        if (addr & PAGE_MASK) + size <= PAGE_SIZE {
            let (ptr, _) = self.translate(addr).ok_or(MemoryFault::unmapped(addr))?;
            // SAFETY: the whole value is inside one mapped page.
            unsafe { std::ptr::write_unaligned(ptr as *mut T, value) };
            return Ok(());
        }
        self.write_bytes(addr, bytemuck::bytes_of(&value))
    }

    pub(crate) fn read_bytes(&self, addr: u64, out: &mut [u8]) -> Result<(), MemoryFault> {
        self.for_each_chunk(addr, out.len() as u64, |ptr, offset, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(ptr, out[offset..].as_mut_ptr(), len) };
        })
    }

    pub(crate) fn write_bytes(&self, addr: u64, data: &[u8]) -> Result<(), MemoryFault> {
        self.for_each_chunk(addr, data.len() as u64, |ptr, offset, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(data[offset..].as_ptr(), ptr, len) };
        })
    }

    pub(crate) fn fill(&self, addr: u64, size: u64, byte: u8) -> Result<(), MemoryFault> {
        self.for_each_chunk(addr, size, |ptr, _, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::write_bytes(ptr, byte, len) };
        })
    }

    fn for_each_chunk(
        &self,
        addr: u64,
        size: u64,
        mut f: impl FnMut(*mut u8, usize, usize),
    ) -> Result<(), MemoryFault> {
        let mut done = 0u64;
        while done < size {
            let at = addr + done;
            let len = (PAGE_SIZE - (at & PAGE_MASK)).min(size - done);
            let (ptr, _) = self.translate(at).ok_or(MemoryFault::unmapped(at))?;
            f(ptr, done as usize, len as usize);
            done += len;
        }
        Ok(())
    }

    #[inline]
    fn entry(&self, addr: u64) -> Option<&AtomicUsize> {
        let page = (addr >> PAGE_BITS) as usize;
        let table = self.l1.get(page >> L2_BITS)?.load(Ordering::Acquire);
        // SAFETY: tables are never freed while the page table lives.
        unsafe { table.as_ref() }.map(|t| &t[page & (L2_ENTRIES - 1)])
    }

    fn entry_for_write(&self, addr: u64) -> &AtomicUsize {
        let page = (addr >> PAGE_BITS) as usize;
        let slot = self
            .l1
            .get(page >> L2_BITS)
            .unwrap_or_else(|| panic!("{addr:#x} outside the {}-bit address space", self.bits));
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

impl Drop for PageTable {
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
