//! GPU virtual memory: one GMMU address space per channel or nvhost-as object.
//!
//! Page tables are atomic entries in tables that are never freed while the [`Gmmu`] lives, so
//! the GPU thread translates without a lock while a service thread maps (NX2_NOTES row 125).
//! Each big page has one entry; mapping part of one splits it into a small-page table, and a
//! small-page table that becomes uniform again collapses back. Mapping a whole big page with
//! small pages therefore costs the same as a big page, which keeps huge sparse regions cheap
//! (NX2_NOTES row 81).

use std::sync::atomic::{AtomicPtr, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use lichii_common::{DAddr, GpuVAddr, PAGE_BITS, PAGE_MASK, PAGE_SIZE};
use lichii_memory::DeviceAddressSpace;
use parking_lot::Mutex;

use crate::host1x::Syncpoints;
use crate::lazy::LazyDeviceRanges;

/// Width of a GPU virtual address space.
pub const GPU_VA_BITS: u32 = 40;
/// Size of a GPU virtual address space in bytes.
pub const GPU_VA_SIZE: u64 = 1 << GPU_VA_BITS;
/// PTE kind of pitch-linear memory, the kind plain mappings get.
pub const PTE_KIND_PITCH: u8 = 0x00;
/// PTE kind of sparse mappings until a kind is set.
pub const PTE_KIND_INVALID: u8 = 0xff;

const TABLE_BITS: u32 = 9;
const TABLE_ENTRIES: usize = 1 << TABLE_BITS;
const MAX_SMALL_PER_BIG: usize = 32;

const STATE_MASK: u64 = 0x7;
const FREE: u64 = 0;
const SPARSE: u64 = 1;
const MAPPED: u64 = 2;
const SHIM: u64 = 3;
const SPLIT: u64 = 4;
const KIND_SHIFT: u32 = 8;
const ADDR_SHIFT: u32 = 16;

/// Size of the GMMU's big pages, chosen per address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BigPageSize {
    Size64K,
    Size128K,
}

impl BigPageSize {
    pub const fn bytes(self) -> u64 {
        1 << self.bits()
    }

    const fn bits(self) -> u32 {
        match self {
            Self::Size64K => 16,
            Self::Size128K => 17,
        }
    }
}

/// Why a GMMU map or unmap was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum GmmuError {
    #[error("GPU range {gpu_va:#x}+{size:#x} is not aligned to {align:#x}")]
    Unaligned {
        gpu_va: GpuVAddr,
        size: u64,
        align: u64,
    },
    #[error("GPU range {gpu_va:#x}+{size:#x} is outside the 40-bit address space")]
    OutOfRange { gpu_va: GpuVAddr, size: u64 },
    #[error("device range {daddr:#x}+{size:#x} is unaligned or outside the device space")]
    InvalidDeviceAddress { daddr: DAddr, size: u64 },
    #[error("the syncpoint shim is already mapped at {0:#x}")]
    ShimAlreadyMapped(GpuVAddr),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFaultKind {
    /// No GMMU mapping.
    Unmapped,
    /// A GMMU mapping to a device page that is not mapped.
    DeviceUnmapped,
    /// A write to the read-only syncpoint shim.
    ReadOnly,
}

/// A GPU access that touched memory it cannot use; carries the first such address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("GPU access fault at {addr:#x}: {kind:?}")]
pub struct GpuFault {
    pub addr: GpuVAddr,
    pub kind: GpuFaultKind,
}

/// What a GPU address range reaches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RangeTarget {
    /// Device memory starting at this address, contiguous for the whole range.
    Device(DAddr),
    /// Reserved: reads return zero, writes are dropped.
    Sparse,
    Unmapped,
    /// The read-only syncpoint page (NX2_NOTES row 46).
    SyncpointShim,
}

/// A run of GPU addresses with one [`RangeTarget`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuRange {
    pub gpu_va: GpuVAddr,
    pub size: u64,
    pub target: RangeTarget,
}

/// A GPU range that was mapped to contiguous device memory, as map and unmap report it so the
/// GPU can tell the renderer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappedRange {
    pub gpu_va: GpuVAddr,
    pub daddr: DAddr,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pte(u64);

impl Pte {
    const FREE: Self = Self(FREE);
    const SPLIT: Self = Self(SPLIT);
    const SHIM: Self = Self(SHIM);

    const fn new(state: u64, kind: u8, daddr: DAddr) -> Self {
        Self(state | (kind as u64) << KIND_SHIFT | (daddr >> PAGE_BITS) << ADDR_SHIFT)
    }

    const fn state(self) -> u64 {
        self.0 & STATE_MASK
    }

    const fn kind(self) -> u8 {
        (self.0 >> KIND_SHIFT) as u8
    }

    const fn daddr(self) -> DAddr {
        (self.0 >> ADDR_SHIFT) << PAGE_BITS
    }

    const fn with_kind(self, kind: u8) -> Self {
        Self(self.0 & !(0xff << KIND_SHIFT) | (kind as u64) << KIND_SHIFT)
    }

    const fn advanced(self, by: u64) -> Self {
        if self.state() == MAPPED {
            Self(self.0 + ((by >> PAGE_BITS) << ADDR_SHIFT))
        } else {
            self
        }
    }

    const fn target(self, offset: u64) -> RangeTarget {
        match self.state() {
            MAPPED => RangeTarget::Device(self.daddr() + offset),
            SPARSE => RangeTarget::Sparse,
            SHIM => RangeTarget::SyncpointShim,
            _ => RangeTarget::Unmapped,
        }
    }
}

type SmallTable = [AtomicU64; MAX_SMALL_PER_BIG];

struct BigTable {
    big: [AtomicU64; TABLE_ENTRIES],
    small: [AtomicPtr<SmallTable>; TABLE_ENTRIES],
}

/// One GPU virtual address space: 40-bit, 4 KiB small pages and 64 or 128 KiB big pages.
pub struct Gmmu {
    device: Arc<DeviceAddressSpace>,
    big_bits: u32,
    directory: Box<[AtomicPtr<BigTable>]>,
    writer: Mutex<()>,
    shim: OnceLock<(GpuVAddr, Arc<Syncpoints>)>,
    /// Device ranges that get memory as they are written, set for Switch 2 address spaces.
    lazy: OnceLock<Arc<LazyDeviceRanges>>,
    generation: AtomicU64,
    table_bytes: AtomicU64,
}

impl Gmmu {
    pub fn new(device: Arc<DeviceAddressSpace>, big_page_size: BigPageSize) -> Self {
        let big_bits = big_page_size.bits();
        let directory = (0..1usize << (GPU_VA_BITS - big_bits - TABLE_BITS))
            .map(|_| AtomicPtr::new(std::ptr::null_mut()))
            .collect();
        Self {
            device,
            big_bits,
            directory,
            writer: Mutex::new(()),
            shim: OnceLock::new(),
            lazy: OnceLock::new(),
            generation: AtomicU64::new(0),
            table_bytes: AtomicU64::new(0),
        }
    }

    pub fn device(&self) -> &Arc<DeviceAddressSpace> {
        &self.device
    }

    pub fn big_page_size(&self) -> u64 {
        1 << self.big_bits
    }

    /// Changes after every map, unmap and kind change: a translation cached by a reader stays
    /// valid only while this stays the same.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Heap bytes the page tables use.
    pub fn table_bytes(&self) -> u64 {
        self.table_bytes.load(Ordering::Relaxed)
    }

    /// Maps [`gpu_va`, +`size`) to device memory at `daddr` with PTE kind `kind`. With `big`, the range
    /// must be aligned to big pages, otherwise to 4 KiB. Returns the mappings it replaced.
    pub fn map(
        &self,
        gpu_va: GpuVAddr,
        daddr: DAddr,
        size: u64,
        kind: u8,
        big: bool,
    ) -> Result<Vec<MappedRange>, GmmuError> {
        self.check(gpu_va, size, big)?;
        let fits = daddr
            .checked_add(size)
            .is_some_and(|end| end <= 1 << self.device.address_space_bits());
        if daddr & PAGE_MASK != 0 || !fits {
            return Err(GmmuError::InvalidDeviceAddress { daddr, size });
        }
        Ok(self.modify(gpu_va, size, true, |va, _| {
            Pte::new(MAPPED, kind, daddr + (va - gpu_va))
        }))
    }

    /// Reserves [`gpu_va`, +`size`) as sparse: reads return zero and writes are dropped until it is
    /// mapped. Commits no device memory. Returns the mappings it replaced.
    pub fn map_sparse(
        &self,
        gpu_va: GpuVAddr,
        size: u64,
        big: bool,
    ) -> Result<Vec<MappedRange>, GmmuError> {
        self.check(gpu_va, size, big)?;
        let sparse = Pte::new(SPARSE, PTE_KIND_INVALID, 0);
        Ok(self.modify(gpu_va, size, true, |_, _| sparse))
    }

    /// Frees [`gpu_va`, +`size`), sparse pages included. Returns the mappings it removed.
    pub fn unmap(&self, gpu_va: GpuVAddr, size: u64) -> Result<Vec<MappedRange>, GmmuError> {
        self.check(gpu_va, size, false)?;
        Ok(self.modify(gpu_va, size, false, |_, _| Pte::FREE))
    }

    /// Sets the PTE kind of the mapped and sparse pages in [`gpu_va`, +`size`).
    pub fn set_kind(&self, gpu_va: GpuVAddr, size: u64, kind: u8) -> Result<(), GmmuError> {
        self.check(gpu_va, size, false)?;
        self.modify(gpu_va, size, false, |_, old| match old.state() {
            MAPPED | SPARSE => old.with_kind(kind),
            _ => old,
        });
        Ok(())
    }

    /// Writes to device pages without memory give lazy device ranges their memory from now on.
    pub(crate) fn set_lazy(&self, lazy: Arc<LazyDeviceRanges>) {
        let _ = self.lazy.set(lazy);
    }

    /// Maps the read-only syncpoint shim page at `gpu_va`: a u32 per syncpoint id with its guest
    /// value (NX2_NOTES row 46). One shim per address space.
    pub fn map_syncpoint_shim(
        &self,
        gpu_va: GpuVAddr,
        syncpoints: Arc<Syncpoints>,
    ) -> Result<Vec<MappedRange>, GmmuError> {
        self.check(gpu_va, PAGE_SIZE, false)?;
        if let Err((_, _)) = self.shim.set((gpu_va, syncpoints)) {
            let (existing, _) = self.shim.get().expect("set failed because it is set");
            return Err(GmmuError::ShimAlreadyMapped(*existing));
        }
        Ok(self.modify(gpu_va, PAGE_SIZE, true, |_, _| Pte::SHIM))
    }

    /// Device address of the byte at `gpu_va`.
    #[inline]
    pub fn translate(&self, gpu_va: GpuVAddr) -> Option<DAddr> {
        let (pte, page) = self.lookup(gpu_va);
        match pte.target(gpu_va & (page - 1)) {
            RangeTarget::Device(daddr) => Some(daddr),
            _ => None,
        }
    }

    /// PTE kind at `gpu_va`; `None` unless it is mapped or sparse.
    pub fn kind(&self, gpu_va: GpuVAddr) -> Option<u8> {
        let (pte, _) = self.lookup(gpu_va);
        matches!(pte.state(), MAPPED | SPARSE).then_some(pte.kind())
    }

    /// Reads a value. On a fault the caller should use zero, as unmapped GPU memory reads.
    #[inline]
    pub fn read<T: bytemuck::Pod>(&self, gpu_va: GpuVAddr) -> Result<T, GpuFault> {
        let (pte, page) = self.lookup(gpu_va);
        let offset = gpu_va & (page - 1);
        if pte.state() == MAPPED
            && offset + size_of::<T>() as u64 <= page
            && let Ok(value) = self.device.read(pte.daddr() + offset)
        {
            return Ok(value);
        }
        let mut value = T::zeroed();
        self.read_block(gpu_va, bytemuck::bytes_of_mut(&mut value))?;
        Ok(value)
    }

    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, gpu_va: GpuVAddr, value: T) -> Result<(), GpuFault> {
        let (pte, page) = self.lookup(gpu_va);
        let offset = gpu_va & (page - 1);
        if pte.state() == MAPPED
            && offset + size_of::<T>() as u64 <= page
            && self.device.write(pte.daddr() + offset, value).is_ok()
        {
            return Ok(());
        }
        self.write_block(gpu_va, bytemuck::bytes_of(&value))
    }

    /// Reads `out.len()` bytes across any number of mappings. Sparse and unmapped memory reads
    /// zero; the first unmapped address is the error, with `out` still filled.
    pub fn read_block(&self, gpu_va: GpuVAddr, out: &mut [u8]) -> Result<(), GpuFault> {
        let mut fault = None;
        let mut done = 0;
        self.for_each_range(gpu_va, out.len() as u64, |range| {
            let chunk = &mut out[done..done + range.size as usize];
            done += range.size as usize;
            let result = match range.target {
                RangeTarget::Device(daddr) => self
                    .read_device(daddr, chunk)
                    .map_err(|d| device_fault(range.gpu_va + (d - daddr))),
                RangeTarget::Sparse => {
                    chunk.fill(0);
                    Ok(())
                }
                RangeTarget::Unmapped => {
                    chunk.fill(0);
                    Err(GpuFault {
                        addr: range.gpu_va,
                        kind: GpuFaultKind::Unmapped,
                    })
                }
                RangeTarget::SyncpointShim => {
                    self.read_shim(range.gpu_va, chunk);
                    Ok(())
                }
            };
            if let Err(e) = result {
                fault.get_or_insert(e);
            }
        });
        fault.map_or(Ok(()), Err)
    }

    /// Writes `data` across any number of mappings. Writes to sparse memory are dropped; the
    /// first unmapped or read-only address is the error, with the rest still written.
    pub fn write_block(&self, gpu_va: GpuVAddr, data: &[u8]) -> Result<(), GpuFault> {
        let mut fault = None;
        let mut done = 0;
        self.for_each_range(gpu_va, data.len() as u64, |range| {
            let chunk = &data[done..done + range.size as usize];
            done += range.size as usize;
            let kind = match range.target {
                RangeTarget::Device(daddr) => {
                    if let Err(d) = self.write_device(daddr, chunk) {
                        fault.get_or_insert(device_fault(range.gpu_va + (d - daddr)));
                    }
                    return;
                }
                RangeTarget::Sparse => return,
                RangeTarget::Unmapped => GpuFaultKind::Unmapped,
                RangeTarget::SyncpointShim => GpuFaultKind::ReadOnly,
            };
            fault.get_or_insert(GpuFault {
                addr: range.gpu_va,
                kind,
            });
        });
        fault.map_or(Ok(()), Err)
    }

    /// Calls `f` for each maximal run of [`gpu_va`, +`size`) with one target, device runs split
    /// only where the device addresses stop being contiguous. For DMA copies.
    pub fn for_each_range(&self, gpu_va: GpuVAddr, size: u64, mut f: impl FnMut(GpuRange)) {
        let end = gpu_va.saturating_add(size);
        let mut run: Option<GpuRange> = None;
        let mut va = gpu_va;
        while va < end {
            let (target, len) = if va >= GPU_VA_SIZE {
                (RangeTarget::Unmapped, end - va)
            } else {
                let (pte, page) = self.lookup(va);
                let offset = va & (page - 1);
                (pte.target(offset), (page - offset).min(end - va))
            };
            match &mut run {
                Some(r) if r.continues_with(target) => r.size += len,
                _ => {
                    if let Some(r) = run.replace(GpuRange {
                        gpu_va: va,
                        size: len,
                        target,
                    }) {
                        f(r);
                    }
                }
            }
            va += len;
        }
        if let Some(r) = run {
            f(r);
        }
    }

    /// Whether [`gpu_va`, +`size`) is mapped to one contiguous device range.
    pub fn is_continuous(&self, gpu_va: GpuVAddr, size: u64) -> bool {
        let mut runs = 0;
        let mut device = true;
        self.for_each_range(gpu_va, size, |range| {
            runs += 1;
            device &= matches!(range.target, RangeTarget::Device(_));
        });
        runs == 1 && device
    }

    /// Copies device memory a page at a time; unmapped pages read zero. Returns the first
    /// unmapped device address outside lazy device ranges, whose pages without memory are zero.
    fn read_device(&self, daddr: DAddr, out: &mut [u8]) -> Result<(), DAddr> {
        let mut fault = None;
        let mut done = 0;
        while done < out.len() {
            let addr = daddr + done as u64;
            let len = ((PAGE_SIZE - (addr & PAGE_MASK)) as usize).min(out.len() - done);
            let chunk = &mut out[done..done + len];
            match self.device.translate(addr) {
                // SAFETY: the pointer is valid for len bytes inside one mapped page.
                Some(ptr) => unsafe { std::ptr::copy_nonoverlapping(ptr, chunk.as_mut_ptr(), len) },
                None => {
                    chunk.fill(0);
                    let lazy = self.lazy.get();
                    if !lazy.is_some_and(|lazy| lazy.is_lazy(addr, len as u64)) {
                        fault.get_or_insert(addr);
                    }
                }
            }
            done += len;
        }
        fault.map_or(Ok(()), Err)
    }

    /// Writes device memory a page at a time, skipping unmapped pages; pages of lazy device
    /// ranges get memory instead. Returns the first unmapped device address.
    fn write_device(&self, daddr: DAddr, data: &[u8]) -> Result<(), DAddr> {
        let result = self.write_mapped(daddr, data);
        match (result, self.lazy.get()) {
            (Err(_), Some(lazy)) => lazy.write(daddr, data).map_err(|fault| fault.addr),
            (result, _) => result,
        }
    }

    fn write_mapped(&self, daddr: DAddr, data: &[u8]) -> Result<(), DAddr> {
        let mut fault = None;
        let mut done = 0;
        while done < data.len() {
            let addr = daddr + done as u64;
            let len = ((PAGE_SIZE - (addr & PAGE_MASK)) as usize).min(data.len() - done);
            match self.device.translate(addr) {
                // SAFETY: the pointer is valid for len bytes inside one mapped page.
                Some(ptr) => unsafe {
                    std::ptr::copy_nonoverlapping(data[done..].as_ptr(), ptr, len)
                },
                None => {
                    fault.get_or_insert(addr);
                }
            }
            done += len;
        }
        fault.map_or(Ok(()), Err)
    }

    fn read_shim(&self, gpu_va: GpuVAddr, out: &mut [u8]) {
        let Some((_, syncpoints)) = self.shim.get() else {
            out.fill(0);
            return;
        };
        let mut done = 0;
        while done < out.len() {
            let offset = (gpu_va + done as u64) & PAGE_MASK;
            let first = (offset % 4) as usize;
            let len = (4 - first).min(out.len() - done);
            let value = syncpoints.guest().value((offset / 4) as u32).unwrap_or(0);
            out[done..done + len].copy_from_slice(&value.to_le_bytes()[first..first + len]);
            done += len;
        }
    }

    fn check(&self, gpu_va: GpuVAddr, size: u64, big: bool) -> Result<(), GmmuError> {
        let align = if big { self.big_page_size() } else { PAGE_SIZE };
        if (gpu_va | size) & (align - 1) != 0 {
            return Err(GmmuError::Unaligned {
                gpu_va,
                size,
                align,
            });
        }
        if gpu_va.checked_add(size).is_none_or(|end| end > GPU_VA_SIZE) {
            return Err(GmmuError::OutOfRange { gpu_va, size });
        }
        Ok(())
    }

    fn small_per_big(&self) -> usize {
        1 << (self.big_bits - PAGE_BITS)
    }

    /// The entry covering `va` and the size of the page (or empty table span) it describes.
    #[inline]
    fn lookup(&self, va: GpuVAddr) -> (Pte, u64) {
        let big_index = (va >> self.big_bits) as usize;
        let Some(slot) = self.directory.get(big_index >> TABLE_BITS) else {
            return (Pte::FREE, PAGE_SIZE);
        };
        // SAFETY: tables are never freed while the GMMU lives.
        let Some(table) = (unsafe { slot.load(Ordering::Acquire).as_ref() }) else {
            return (Pte::FREE, self.big_page_size() << TABLE_BITS);
        };
        let index = big_index & (TABLE_ENTRIES - 1);
        let big = Pte(table.big[index].load(Ordering::Acquire));
        if big.state() != SPLIT {
            return (big, self.big_page_size());
        }
        // SAFETY: a split entry is published after its small table, which is never freed.
        let small = unsafe { &*table.small[index].load(Ordering::Acquire) };
        let sub = (va >> PAGE_BITS) as usize & (self.small_per_big() - 1);
        (Pte(small[sub].load(Ordering::Acquire)), PAGE_SIZE)
    }

    fn modify(
        &self,
        gpu_va: GpuVAddr,
        size: u64,
        allocate: bool,
        entry: impl FnMut(GpuVAddr, Pte) -> Pte,
    ) -> Vec<MappedRange> {
        let _writer = self.writer.lock();
        let mut replaced = Vec::new();
        self.for_each_range(gpu_va, size, |range| {
            if let RangeTarget::Device(daddr) = range.target {
                replaced.push(MappedRange {
                    gpu_va: range.gpu_va,
                    daddr,
                    size: range.size,
                });
            }
        });
        self.update(gpu_va, size, allocate, entry);
        self.generation.fetch_add(1, Ordering::Release);
        replaced
    }

    /// Rewrites every page of the range with `entry(page address, old entry)`, a whole big page at
    /// once where the range covers it. Without `allocate`, empty table spans are skipped.
    fn update(
        &self,
        gpu_va: GpuVAddr,
        size: u64,
        allocate: bool,
        mut entry: impl FnMut(GpuVAddr, Pte) -> Pte,
    ) {
        let big = self.big_page_size();
        let end = gpu_va + size;
        let mut va = gpu_va;
        while va < end {
            let slot = &self.directory[(va >> (self.big_bits + TABLE_BITS)) as usize];
            if !allocate && slot.load(Ordering::Acquire).is_null() {
                va = ((va >> (self.big_bits + TABLE_BITS)) + 1) << (self.big_bits + TABLE_BITS);
                continue;
            }
            let table = self.table_for_write(slot);
            let index = (va >> self.big_bits) as usize & (TABLE_ENTRIES - 1);
            let big_start = va & !(big - 1);
            let old = Pte(table.big[index].load(Ordering::Relaxed));
            if old.state() != SPLIT && va == big_start && end - va >= big {
                table.big[index].store(entry(va, old).0, Ordering::Release);
                va += big;
                continue;
            }
            let small = self.split(table, index, old);
            let stop = (big_start + big).min(end);
            while va < stop {
                let sub = (va >> PAGE_BITS) as usize & (self.small_per_big() - 1);
                let old = Pte(small[sub].load(Ordering::Relaxed));
                small[sub].store(entry(va, old).0, Ordering::Release);
                va += PAGE_SIZE;
            }
            self.collapse(table, index, small);
        }
    }

    fn table_for_write<'a>(&self, slot: &'a AtomicPtr<BigTable>) -> &'a BigTable {
        let mut ptr = slot.load(Ordering::Acquire);
        if ptr.is_null() {
            ptr = Box::into_raw(Box::new(BigTable {
                big: std::array::from_fn(|_| AtomicU64::new(FREE)),
                small: std::array::from_fn(|_| AtomicPtr::new(std::ptr::null_mut())),
            }));
            slot.store(ptr, Ordering::Release);
            self.table_bytes
                .fetch_add(size_of::<BigTable>() as u64, Ordering::Relaxed);
        }
        // SAFETY: tables are never freed while the GMMU lives.
        unsafe { &*ptr }
    }

    /// The small-page table of a big page, filled from `old` and published first if the big page
    /// is not split yet.
    fn split<'a>(&self, table: &'a BigTable, index: usize, old: Pte) -> &'a SmallTable {
        let mut ptr = table.small[index].load(Ordering::Acquire);
        if ptr.is_null() {
            ptr = Box::into_raw(Box::new(std::array::from_fn(|_| AtomicU64::new(FREE))));
            table.small[index].store(ptr, Ordering::Release);
            self.table_bytes
                .fetch_add(size_of::<SmallTable>() as u64, Ordering::Relaxed);
        }
        // SAFETY: small tables are never freed while the GMMU lives.
        let small = unsafe { &*ptr };
        if old.state() != SPLIT {
            for (i, e) in small.iter().take(self.small_per_big()).enumerate() {
                e.store(old.advanced(i as u64 * PAGE_SIZE).0, Ordering::Release);
            }
            table.big[index].store(Pte::SPLIT.0, Ordering::Release);
        }
        small
    }

    fn collapse(&self, table: &BigTable, index: usize, small: &SmallTable) {
        let first = Pte(small[0].load(Ordering::Relaxed));
        if first.state() == SHIM {
            return;
        }
        let uniform = (1..self.small_per_big())
            .all(|i| small[i].load(Ordering::Relaxed) == first.advanced(i as u64 * PAGE_SIZE).0);
        if uniform {
            table.big[index].store(first.0, Ordering::Release);
        }
    }
}

impl GpuRange {
    fn continues_with(&self, next: RangeTarget) -> bool {
        match (self.target, next) {
            (RangeTarget::Device(start), RangeTarget::Device(daddr)) => start + self.size == daddr,
            (a, b) => a == b,
        }
    }
}

fn device_fault(addr: GpuVAddr) -> GpuFault {
    GpuFault {
        addr,
        kind: GpuFaultKind::DeviceUnmapped,
    }
}

impl Drop for Gmmu {
    fn drop(&mut self) {
        for slot in self.directory.iter() {
            let table = slot.load(Ordering::Acquire);
            if table.is_null() {
                continue;
            }
            // SAFETY: created by Box::into_raw in table_for_write and dropped once here.
            let table = unsafe { Box::from_raw(table) };
            for small in table.small.iter() {
                let small = small.load(Ordering::Acquire);
                if !small.is_null() {
                    // SAFETY: created by Box::into_raw in split and dropped once here.
                    drop(unsafe { Box::from_raw(small) });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_memory::{AddressSpace, MemoryPermission, PhysicalMemory};
    use std::sync::atomic::AtomicBool;

    const CPU_VA: u64 = 0x1000_0000;
    const CPU_SIZE: u64 = 0x40_0000;

    struct Fixture {
        space: AddressSpace,
        device: Arc<DeviceAddressSpace>,
        daddr: DAddr,
    }

    fn fixture() -> Fixture {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let space = AddressSpace::new(physical.clone(), 39);
        space.map(CPU_VA, 0x10_0000, CPU_SIZE, MemoryPermission::READ_WRITE);
        let device = Arc::new(DeviceAddressSpace::new(physical, 36));
        let daddr = 0x4_0000_0000;
        device.map_process(daddr, &space, CPU_VA, CPU_SIZE).unwrap();
        Fixture {
            space,
            device,
            daddr,
        }
    }

    impl Fixture {
        fn gmmu(&self, big: BigPageSize) -> Gmmu {
            Gmmu::new(self.device.clone(), big)
        }
    }

    #[test]
    fn big_and_small_pages_mix() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size64K);
        let d = f.daddr;
        assert!(
            gmmu.map(0x10_0000, d, 0x2_0000, PTE_KIND_PITCH, true)
                .unwrap()
                .is_empty()
        );
        assert_eq!(gmmu.translate(0x10_1234), Some(d + 0x1234));
        assert_eq!(gmmu.translate(0x11_fffc), Some(d + 0x1_fffc));
        let replaced = gmmu
            .map(0x10_1000, d + 0x8000, 0x1000, 0xfe, false)
            .unwrap();
        assert_eq!(
            replaced,
            [MappedRange {
                gpu_va: 0x10_1000,
                daddr: d + 0x1000,
                size: 0x1000
            }]
        );
        assert_eq!(gmmu.translate(0x10_1010), Some(d + 0x8010));
        assert_eq!(gmmu.translate(0x10_2000), Some(d + 0x2000));
        assert_eq!(gmmu.translate(0x10_0000), Some(d));
        assert_eq!(gmmu.kind(0x10_1000), Some(0xfe));
        assert_eq!(gmmu.kind(0x10_0000), Some(PTE_KIND_PITCH));
        assert_eq!(gmmu.lookup(0x11_0000).1, 0x1_0000);
        assert_eq!(gmmu.lookup(0x10_0000).1, PAGE_SIZE);
        f.space.write(CPU_VA + 0x8010, 0x5566_7788u32).unwrap();
        assert_eq!(gmmu.read::<u32>(0x10_1010), Ok(0x5566_7788));
        assert_eq!(
            gmmu.map(0x10_1000, d, 0x1_0000, PTE_KIND_PITCH, true),
            Err(GmmuError::Unaligned {
                gpu_va: 0x10_1000,
                size: 0x1_0000,
                align: 0x1_0000
            })
        );
    }

    #[test]
    fn small_pages_covering_a_big_page_collapse() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size128K);
        for page in 0..32 {
            let offset = page * PAGE_SIZE;
            gmmu.map(0x20_0000 + offset, f.daddr + offset, PAGE_SIZE, 7, false)
                .unwrap();
        }
        assert_eq!(gmmu.lookup(0x20_0000).1, 0x2_0000);
        assert_eq!(gmmu.translate(0x21_f004), Some(f.daddr + 0x1_f004));
        gmmu.unmap(0x20_0000, 0x2_0000).unwrap();
        assert_eq!(gmmu.lookup(0x20_0000), (Pte::FREE, 0x2_0000));
    }

    #[test]
    fn unmap_reports_what_was_mapped() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size64K);
        let d = f.daddr;
        gmmu.map(0x20_0000, d, 0x3000, PTE_KIND_PITCH, false)
            .unwrap();
        gmmu.map_sparse(0x20_3000, 0x2000, false).unwrap();
        gmmu.map(0x20_5000, d + 0x10_0000, 0x1000, PTE_KIND_PITCH, false)
            .unwrap();
        let removed = gmmu.unmap(0x20_0000, 0x8000).unwrap();
        assert_eq!(
            removed,
            [
                MappedRange {
                    gpu_va: 0x20_0000,
                    daddr: d,
                    size: 0x3000
                },
                MappedRange {
                    gpu_va: 0x20_5000,
                    daddr: d + 0x10_0000,
                    size: 0x1000
                }
            ]
        );
        assert_eq!(gmmu.translate(0x20_0000), None);
        assert_eq!(gmmu.kind(0x20_3000), None);
        assert!(gmmu.unmap(0x20_0000, 0x8000).unwrap().is_empty());
    }

    #[test]
    fn sparse_reads_zero_and_drops_writes() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size64K);
        gmmu.map_sparse(0x40_0000, 0x4_0000, true).unwrap();
        assert_eq!(gmmu.write(0x40_1000, u64::MAX), Ok(()));
        assert_eq!(gmmu.read::<u64>(0x40_1000), Ok(0));
        assert_eq!(gmmu.translate(0x40_1000), None);
        assert_eq!(gmmu.kind(0x40_1000), Some(PTE_KIND_INVALID));
        gmmu.map(0x40_2000, f.daddr, 0x1000, PTE_KIND_PITCH, false)
            .unwrap();
        gmmu.write(0x40_2ffe, 0xaabbu16).unwrap();
        let mut out = [0xffu8; 4];
        gmmu.read_block(0x40_2ffe, &mut out).unwrap();
        assert_eq!(out, [0xbb, 0xaa, 0, 0]);
        let mut ranges = Vec::new();
        gmmu.for_each_range(0x40_0000, 0x4_0000, |r| ranges.push(r.target));
        assert_eq!(
            ranges,
            [
                RangeTarget::Sparse,
                RangeTarget::Device(f.daddr),
                RangeTarget::Sparse
            ]
        );
    }

    #[test]
    fn blocks_cross_mappings() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size128K);
        let d = f.daddr;
        gmmu.map(0x30_0000, d + 0x1000, 0x1000, PTE_KIND_PITCH, false)
            .unwrap();
        gmmu.map(0x30_1000, d + 0x5000, 0x2000, PTE_KIND_PITCH, false)
            .unwrap();
        let data: Vec<u8> = (0..0x2000u32).map(|i| (i * 7) as u8).collect();
        gmmu.write_block(0x30_0800, &data).unwrap();
        let mut back = vec![0; 0x2000];
        gmmu.read_block(0x30_0800, &mut back).unwrap();
        assert_eq!(back, data);
        assert_eq!(f.device.read::<u8>(d + 0x1fff), Ok(data[0x7ff]));
        assert_eq!(f.device.read::<u8>(d + 0x5000), Ok(data[0x800]));
        assert_eq!(f.space.read::<u8>(CPU_VA + 0x5001), Ok(data[0x801]));
        let mut ranges = Vec::new();
        gmmu.for_each_range(0x30_0000, 0x4000, |r| ranges.push(r));
        assert_eq!(
            ranges,
            [
                GpuRange {
                    gpu_va: 0x30_0000,
                    size: 0x1000,
                    target: RangeTarget::Device(d + 0x1000)
                },
                GpuRange {
                    gpu_va: 0x30_1000,
                    size: 0x2000,
                    target: RangeTarget::Device(d + 0x5000)
                },
                GpuRange {
                    gpu_va: 0x30_3000,
                    size: 0x1000,
                    target: RangeTarget::Unmapped
                },
            ]
        );
        assert!(gmmu.is_continuous(0x30_1000, 0x2000));
        assert!(!gmmu.is_continuous(0x30_0000, 0x2000));
        assert_eq!(
            gmmu.read::<u64>(0x30_0ffc),
            Ok(u64::from_le_bytes(back[0x7fc..0x804].try_into().unwrap()))
        );
    }

    #[test]
    fn unmapped_reads_zero_and_fault() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size64K);
        gmmu.map(0x50_0000, f.daddr, 0x1000, PTE_KIND_PITCH, false)
            .unwrap();
        gmmu.write(0x50_0ff8, u64::MAX).unwrap();
        let mut out = [0x11u8; 16];
        assert_eq!(
            gmmu.read_block(0x50_0ff8, &mut out),
            Err(GpuFault {
                addr: 0x50_1000,
                kind: GpuFaultKind::Unmapped
            })
        );
        assert_eq!(
            out,
            [
                0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        );
        assert_eq!(
            gmmu.read::<u32>(GPU_VA_SIZE + 4),
            Err(GpuFault {
                addr: GPU_VA_SIZE + 4,
                kind: GpuFaultKind::Unmapped
            })
        );
        assert!(gmmu.write(0x60_0000, 1u32).is_err());
        assert_eq!(
            gmmu.map(GPU_VA_SIZE - 0x1000, f.daddr, 0x2000, 0, false),
            Err(GmmuError::OutOfRange {
                gpu_va: GPU_VA_SIZE - 0x1000,
                size: 0x2000
            })
        );
        f.device.unmap(f.daddr, 0x1000);
        assert_eq!(
            gmmu.read::<u32>(0x50_0010),
            Err(GpuFault {
                addr: 0x50_0010,
                kind: GpuFaultKind::DeviceUnmapped
            })
        );
    }

    #[test]
    fn syncpoint_shim_reads_guest_values() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size64K);
        let syncpoints = Arc::new(Syncpoints::new());
        syncpoints.guest().increment(3).unwrap();
        syncpoints.guest().increment(3).unwrap();
        syncpoints.host().increment(4).unwrap();
        syncpoints.guest().increment(0x3ff).unwrap();
        let shim = GPU_VA_SIZE - PAGE_SIZE;
        gmmu.map_syncpoint_shim(shim, syncpoints.clone()).unwrap();
        assert_eq!(gmmu.read::<u32>(shim + 12), Ok(2));
        assert_eq!(gmmu.read::<u32>(shim + 16), Ok(0));
        assert_eq!(gmmu.read::<u32>(shim + 0xffc), Ok(1));
        assert_eq!(gmmu.read::<u32>(shim + 10), Ok(2 << 16));
        syncpoints.guest().increment(3).unwrap();
        let mut words = [0u32; 4];
        gmmu.read_block(shim, bytemuck::cast_slice_mut(&mut words))
            .unwrap();
        assert_eq!(words, [0, 0, 0, 3]);
        assert_eq!(
            gmmu.write(shim + 12, 9u32),
            Err(GpuFault {
                addr: shim + 12,
                kind: GpuFaultKind::ReadOnly
            })
        );
        assert_eq!(gmmu.translate(shim), None);
        assert_eq!(
            gmmu.map_syncpoint_shim(0x1000, syncpoints),
            Err(GmmuError::ShimAlreadyMapped(shim))
        );
    }

    #[test]
    fn huge_sparse_mapping_costs_few_tables() {
        let f = fixture();
        let gmmu = f.gmmu(BigPageSize::Size64K);
        let base = 0x81_0000_0000;
        let size = 20 << 30;
        gmmu.map_sparse(base, size, false).unwrap();
        assert!(gmmu.table_bytes() < 6 << 20, "{}", gmmu.table_bytes());
        assert_eq!(gmmu.read::<u32>(base + size - 4), Ok(0));
        assert_eq!(gmmu.kind(base + (7 << 30)), Some(PTE_KIND_INVALID));
        let before = gmmu.table_bytes();
        gmmu.unmap(0, 0x80_0000_0000).unwrap();
        gmmu.unmap(base, size).unwrap();
        assert_eq!(gmmu.table_bytes(), before);
        assert_eq!(gmmu.kind(base), None);
    }

    #[test]
    fn concurrent_map_while_reading() {
        let f = fixture();
        let gmmu = Arc::new(f.gmmu(BigPageSize::Size64K));
        let d = f.daddr;
        f.device.fill(d, 0x2_0000, 0xaa).unwrap();
        f.device.fill(d + 0x2_0000, 0x2_0000, 0xbb).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let reads = Arc::new(AtomicU64::new(0));
        let reader = {
            let gmmu = gmmu.clone();
            let stop = stop.clone();
            let reads = reads.clone();
            std::thread::spawn(move || {
                let mut block = [0u32; 64];
                while !stop.load(Ordering::Relaxed) {
                    for offset in (0..0x2_0000).step_by(0x1f04) {
                        let value = gmmu.read::<u32>(0x100_0000 + offset).unwrap_or(0);
                        assert!(matches!(value, 0 | 0xaaaa_aaaa | 0xbbbb_bbbb), "{value:#x}");
                    }
                    let _ = gmmu.read_block(0x100_0f80, bytemuck::cast_slice_mut(&mut block));
                    for word in block {
                        assert!(matches!(word, 0 | 0xaaaa_aaaa | 0xbbbb_bbbb), "{word:#x}");
                    }
                    reads.fetch_add(1, Ordering::Relaxed);
                }
            })
        };
        let mut round = 0u64;
        while round < 2000 || reads.load(Ordering::Relaxed) < 2000 {
            match round % 4 {
                0 => gmmu.map(0x100_0000, d, 0x2_0000, 0, true),
                1 => gmmu.map(0x100_1000, d + 0x2_1000, 0x1000, 0, false),
                2 => gmmu.map_sparse(0x100_0000, 0x1_0000, true),
                _ => gmmu.unmap(0x100_0000, 0x2_0000),
            }
            .unwrap();
            round += 1;
        }
        stop.store(true, Ordering::Relaxed);
        reader.join().unwrap();
    }
}
