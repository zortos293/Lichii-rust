use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use lichii_common::{PAGE_BITS, PAGE_MASK, PAGE_SIZE};
use parking_lot::Mutex;

use crate::{FaultKind, MemoryFault, MemoryPermission};

const L2_BITS: u32 = 14;
const L2_ENTRIES: usize = 1 << L2_BITS;
const PERM_MASK: usize = 0xfff;
/// Set on process pages whose physical page a GPU cache holds ([`crate::tracking`]).
pub(crate) const TRACKED: usize = 0x800;
/// Set on process pages whose physical page holds GPU-modified data not written back yet: every
/// host access through the table calls its trap first ([`crate::tracking`]).
pub(crate) const FLUSH_TRAP: usize = 0x400;
const FLAGS: usize = TRACKED | FLUSH_TRAP;

/// The trap callback of accesses that never meet a flush trap, such as device accesses.
#[inline]
pub(crate) fn no_trap(_addr: u64) {}

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

    /// Maps `addr` to `host` with `perm` and the [`TRACKED`] and [`FLUSH_TRAP`] bits of `flags`.
    pub(crate) fn set(&self, addr: u64, host: *mut u8, perm: MemoryPermission, flags: usize) {
        self.entry_for_write(addr)
            .store(host as usize | perm.0 as usize | flags, Ordering::Release);
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
        entry.store(
            (old & (!PERM_MASK | FLAGS)) | perm.0 as usize,
            Ordering::Release,
        );
        true
    }

    /// Sets the [`TRACKED`] and [`FLUSH_TRAP`] flags of a mapped page to `flags`. Returns the
    /// page's permission when they changed.
    pub(crate) fn set_flags(&self, addr: u64, flags: usize) -> Option<MemoryPermission> {
        let entry = self.entry(addr)?;
        let old = entry.load(Ordering::Acquire);
        if old == 0 || old & FLAGS == flags {
            return None;
        }
        entry.store((old & !FLAGS) | flags, Ordering::Release);
        Some(MemoryPermission(old as u8 & 7))
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

    /// Host pointer for an access to `addr` that needs `need`, ignoring flush traps.
    #[inline]
    pub(crate) fn access(&self, addr: u64, need: MemoryPermission) -> Result<*mut u8, MemoryFault> {
        self.access_with(addr, need, no_trap)
    }

    /// Host pointer for an access to `addr` that needs `need`. On a page with a flush trap it
    /// calls `trap` with `addr` first, then accesses the page whatever the trap did, unless `addr`
    /// was remapped meanwhile to another page with a flush trap, which traps in turn.
    #[inline]
    pub(crate) fn access_with(
        &self,
        addr: u64,
        need: MemoryPermission,
        trap: impl Fn(u64) + Copy,
    ) -> Result<*mut u8, MemoryFault> {
        let entry = self.entry(addr).ok_or(MemoryFault::unmapped(addr))?;
        let raw = entry.load(Ordering::Acquire);
        let need_bits = need.0 as usize;
        if raw & (FLUSH_TRAP | need_bits) == need_bits && raw != 0 {
            return Ok(Self::pointer(raw, addr));
        }
        Self::access_slow(entry, raw, addr, need, trap)
    }

    /// [`PageTable::access_with`] for an entry that is unmapped, lacks `need` or has a flush
    /// trap.
    #[cold]
    fn access_slow(
        entry: &AtomicUsize,
        mut raw: usize,
        addr: u64,
        need: MemoryPermission,
        trap: impl Fn(u64) + Copy,
    ) -> Result<*mut u8, MemoryFault> {
        while raw & FLUSH_TRAP != 0 {
            trap(addr);
            let trapped = raw;
            raw = entry.load(Ordering::Acquire);
            if raw & !PERM_MASK == trapped & !PERM_MASK {
                break;
            }
        }
        if raw == 0 {
            return Err(MemoryFault::unmapped(addr));
        }
        if !MemoryPermission(raw as u8 & 7).contains(need) {
            return Err(MemoryFault {
                addr,
                kind: FaultKind::Permission,
            });
        }
        Ok(Self::pointer(raw, addr))
    }

    #[inline]
    fn pointer(raw: usize, addr: u64) -> *mut u8 {
        let host = (raw & !PERM_MASK) as *mut u8;
        // SAFETY: the offset stays inside the mapped page.
        unsafe { host.add((addr & PAGE_MASK) as usize) }
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
    pub(crate) fn read<T: bytemuck::Pod>(
        &self,
        addr: u64,
        need: MemoryPermission,
        trap: impl Fn(u64) + Copy,
    ) -> Result<T, MemoryFault> {
        let size = std::mem::size_of::<T>() as u64;
        if (addr & PAGE_MASK) + size <= PAGE_SIZE {
            let ptr = self.access_with(addr, need, trap)?;
            // SAFETY: the whole value is inside one mapped page.
            return Ok(unsafe { std::ptr::read_unaligned(ptr as *const T) });
        }
        self.read_split(addr, need, trap)
    }

    /// [`PageTable::read`] of a value that crosses a page boundary.
    #[cold]
    #[inline(never)]
    fn read_split<T: bytemuck::Pod>(
        &self,
        addr: u64,
        need: MemoryPermission,
        trap: impl Fn(u64) + Copy,
    ) -> Result<T, MemoryFault> {
        let mut value = T::zeroed();
        self.read_bytes(addr, bytemuck::bytes_of_mut(&mut value), need, trap)?;
        Ok(value)
    }

    #[inline]
    pub(crate) fn write<T: bytemuck::Pod>(
        &self,
        addr: u64,
        value: T,
        need: MemoryPermission,
    ) -> Result<(), MemoryFault> {
        let size = std::mem::size_of::<T>() as u64;
        if (addr & PAGE_MASK) + size <= PAGE_SIZE {
            let ptr = self.access(addr, need)?;
            // SAFETY: the whole value is inside one mapped page.
            unsafe { std::ptr::write_unaligned(ptr as *mut T, value) };
            return Ok(());
        }
        self.write_bytes(addr, bytemuck::bytes_of(&value), need)
    }

    #[inline]
    pub(crate) fn read_bytes(
        &self,
        addr: u64,
        out: &mut [u8],
        need: MemoryPermission,
        trap: impl Fn(u64) + Copy,
    ) -> Result<(), MemoryFault> {
        self.for_each_chunk(addr, out.len() as u64, need, trap, |ptr, offset, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(ptr, out[offset..].as_mut_ptr(), len) };
        })
    }

    pub(crate) fn write_bytes(
        &self,
        addr: u64,
        data: &[u8],
        need: MemoryPermission,
    ) -> Result<(), MemoryFault> {
        self.for_each_chunk(
            addr,
            data.len() as u64,
            need,
            no_trap,
            |ptr, offset, len| {
                // SAFETY: ptr is valid for len bytes inside one mapped page.
                unsafe { std::ptr::copy_nonoverlapping(data[offset..].as_ptr(), ptr, len) };
            },
        )
    }

    pub(crate) fn fill(
        &self,
        addr: u64,
        size: u64,
        byte: u8,
        need: MemoryPermission,
    ) -> Result<(), MemoryFault> {
        self.for_each_chunk(addr, size, need, no_trap, |ptr, _, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::write_bytes(ptr, byte, len) };
        })
    }

    #[inline]
    pub(crate) fn for_each_chunk(
        &self,
        addr: u64,
        size: u64,
        need: MemoryPermission,
        trap: impl Fn(u64) + Copy,
        mut f: impl FnMut(*mut u8, usize, usize),
    ) -> Result<(), MemoryFault> {
        let mut done = 0u64;
        while done < size {
            let at = addr + done;
            let len = (PAGE_SIZE - (at & PAGE_MASK)).min(size - done);
            f(
                self.access_with(at, need, trap)?,
                done as usize,
                len as usize,
            );
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[repr(align(4096))]
    struct Page([u8; 4096]);

    const ADDR: u64 = 0x8000_0000;
    const RW: MemoryPermission = MemoryPermission::READ_WRITE;

    #[test]
    fn an_access_remapped_to_another_trapped_page_during_its_trap_traps_again() {
        let table = PageTable::new(39);
        let mut pages = [Page([0; 4096]), Page([0; 4096])];
        let [a, b] = pages.each_mut().map(|page| page.0.as_mut_ptr());
        table.set(ADDR, a, RW, FLUSH_TRAP);
        let traps = Cell::new(0);
        let remap = |addr: u64| {
            assert_eq!(addr, ADDR + 0x10);
            traps.set(traps.get() + 1);
            let flags = if traps.get() == 1 { FLUSH_TRAP } else { 0 };
            table.set(ADDR, b, RW, flags);
        };
        let ptr = table
            .access_with(ADDR + 0x10, MemoryPermission::READ, remap)
            .unwrap();
        assert_eq!(traps.get(), 2, "the page it was remapped to trapped too");
        assert_eq!(ptr, b.wrapping_add(0x10));

        table.set(ADDR, a, RW, FLUSH_TRAP);
        traps.set(0);
        let ignore = |_: u64| traps.set(traps.get() + 1);
        let ptr = table
            .access_with(ADDR + 0x10, MemoryPermission::READ, ignore)
            .unwrap();
        assert_eq!(
            traps.get(),
            1,
            "a page its trap left trapped is accessed anyway"
        );
        assert_eq!(ptr, a.wrapping_add(0x10));
    }
}
