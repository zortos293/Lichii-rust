use std::sync::Arc;

use lichii_common::{PAGE_MASK, PAGE_SIZE, PAddr, VAddr};

use crate::PhysicalMemory;
use crate::fault::{FaultHandler, Registration};
use crate::host::{Arena, HostProtection};
use crate::page_table::PageTable;

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

    fn host(self) -> HostProtection {
        if self.contains(Self::WRITE) {
            HostProtection::ReadWrite
        } else if self.contains(Self::READ) {
            HostProtection::Read
        } else {
            HostProtection::None
        }
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

impl MemoryFault {
    pub const fn unmapped(addr: VAddr) -> Self {
        Self {
            addr,
            kind: FaultKind::Unmapped,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum FastmemError {
    #[error("the DRAM backing cannot be mapped as views on this host")]
    NoViews,
    #[error("host fault handling for fastmem is unsupported on this host")]
    NoFaultHandler,
    #[error("reserving the fastmem arena: {0}")]
    Reserve(std::io::Error),
}

/// A host range of `2^address_space_bits` bytes that mirrors the guest address space: guest page
/// `va` is mapped at `base + va` with host protection matching the guest permission, and
/// unmapped pages are inaccessible. The JIT emits direct host loads and stores against it.
pub struct Fastmem {
    arena: Arena,
    registration: Registration,
}

impl Fastmem {
    /// Host address of guest address 0.
    pub fn base(&self) -> *mut u8 {
        self.arena.base()
    }

    pub fn size(&self) -> usize {
        self.arena.size()
    }

    /// Sets the handler for host faults inside this arena (unmapped or protected guest pages).
    /// `data` is passed back to the handler. See [`FaultHandler`] for its restrictions.
    pub fn set_fault_handler(&self, handler: Option<FaultHandler>, data: usize) {
        self.registration.set_handler(handler, data);
    }
}

/// A process's guest virtual address space.
pub struct AddressSpace {
    table: PageTable,
    physical: Arc<PhysicalMemory>,
    fastmem: Option<Fastmem>,
}

impl AddressSpace {
    /// `address_space_bits` is 32, 36 or 39 for Horizon processes.
    pub fn new(physical: Arc<PhysicalMemory>, address_space_bits: u32) -> Self {
        Self {
            table: PageTable::new(address_space_bits),
            physical,
            fastmem: None,
        }
    }

    /// Like [`AddressSpace::new`], with a [`Fastmem`] arena kept in sync with every map.
    pub fn with_fastmem(
        physical: Arc<PhysicalMemory>,
        address_space_bits: u32,
    ) -> Result<Self, FastmemError> {
        if !physical.supports_views() {
            return Err(FastmemError::NoViews);
        }
        let arena = Arena::reserve(1usize << address_space_bits).map_err(FastmemError::Reserve)?;
        let registration = Registration::new(arena.base() as usize, arena.size())
            .ok_or(FastmemError::NoFaultHandler)?;
        Ok(Self {
            table: PageTable::new(address_space_bits),
            physical,
            fastmem: Some(Fastmem {
                arena,
                registration,
            }),
        })
    }

    pub fn address_space_bits(&self) -> u32 {
        self.table.bits()
    }

    pub fn physical(&self) -> &Arc<PhysicalMemory> {
        &self.physical
    }

    pub fn fastmem(&self) -> Option<&Fastmem> {
        self.fastmem.as_ref()
    }

    /// Maps `size` bytes of DRAM at `pa` to `va`. Both must be page aligned.
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
        if let Err(err) = self.physical.commit(pa, size) {
            panic!("{err}");
        }
        for page in 0..size / PAGE_SIZE {
            let host = self.physical.pointer(pa + page * PAGE_SIZE);
            self.table.set(va + page * PAGE_SIZE, host, perm);
        }
        if let Some(fastmem) = &self.fastmem {
            fastmem
                .arena
                .map(
                    self.physical.backing(),
                    va as usize,
                    pa as usize,
                    size as usize,
                    perm.host(),
                )
                .unwrap_or_else(|err| panic!("fastmem map {va:#x}+{size:#x}: {err}"));
        }
    }

    pub fn unmap(&self, va: VAddr, size: u64) {
        assert_eq!(
            (va | size) & PAGE_MASK,
            0,
            "unaligned unmap {va:#x} {size:#x}"
        );
        for page in 0..size / PAGE_SIZE {
            self.table.clear(va + page * PAGE_SIZE);
        }
        if let Some(fastmem) = &self.fastmem {
            fastmem
                .arena
                .unmap(self.physical.backing(), va as usize, size as usize)
                .unwrap_or_else(|err| panic!("fastmem unmap {va:#x}+{size:#x}: {err}"));
        }
    }

    pub fn protect(&self, va: VAddr, size: u64, perm: MemoryPermission) {
        let start = va & !PAGE_MASK;
        let pages = (va + size).div_ceil(PAGE_SIZE) - start / PAGE_SIZE;
        let mut run: Option<(VAddr, u64)> = None;
        for page in 0..pages {
            let addr = start + page * PAGE_SIZE;
            if self.table.set_permission(addr, perm) {
                run = Some(run.map_or((addr, PAGE_SIZE), |(s, len)| (s, len + PAGE_SIZE)));
                continue;
            }
            if let Some((s, len)) = run.take() {
                self.protect_host(s, len, perm);
            }
        }
        if let Some((s, len)) = run {
            self.protect_host(s, len, perm);
        }
    }

    fn protect_host(&self, va: VAddr, size: u64, perm: MemoryPermission) {
        if let Some(fastmem) = &self.fastmem {
            fastmem
                .arena
                .protect(
                    self.physical.backing(),
                    va as usize,
                    size as usize,
                    perm.host(),
                )
                .unwrap_or_else(|err| panic!("fastmem protect {va:#x}+{size:#x}: {err}"));
        }
    }

    /// Host pointer and permission for the byte at `va`.
    #[inline]
    pub fn translate(&self, va: VAddr) -> Option<(*mut u8, MemoryPermission)> {
        self.table.translate(va)
    }

    /// Host pointer for `[va, va + size)` when it is mapped to contiguous DRAM.
    pub fn contiguous_pointer(&self, va: VAddr, size: u64) -> Option<*mut u8> {
        self.table.contiguous(va, size)
    }

    /// Physical address of the byte at `va`.
    pub fn physical_address(&self, va: VAddr) -> Option<PAddr> {
        let (ptr, _) = self.table.translate(va)?;
        self.physical.physical_address(ptr)
    }

    pub fn is_mapped(&self, va: VAddr) -> bool {
        self.translate(va).is_some()
    }

    #[inline]
    pub fn read<T: bytemuck::Pod>(&self, va: VAddr) -> Result<T, MemoryFault> {
        self.table.read(va)
    }

    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, va: VAddr, value: T) -> Result<(), MemoryFault> {
        self.table.write(va, value)
    }

    pub fn read_bytes(&self, va: VAddr, out: &mut [u8]) -> Result<(), MemoryFault> {
        self.table.read_bytes(va, out)
    }

    pub fn write_bytes(&self, va: VAddr, data: &[u8]) -> Result<(), MemoryFault> {
        self.table.write_bytes(va, data)
    }

    pub fn fill(&self, va: VAddr, size: u64, byte: u8) -> Result<(), MemoryFault> {
        self.table.fill(va, size, byte)
    }

    /// Reads a NUL-terminated string of at most `max_len` bytes.
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

    #[test]
    fn contiguous_pointer_needs_contiguous_dram() {
        let mem = space();
        mem.map(0x10_0000, 0x4000, 0x2000, MemoryPermission::READ_WRITE);
        mem.map(0x10_2000, 0x8000, 0x1000, MemoryPermission::READ_WRITE);
        assert!(mem.contiguous_pointer(0x10_0800, 0x1800).is_some());
        assert!(mem.contiguous_pointer(0x10_0800, 0x2000).is_none());
        assert_eq!(mem.physical_address(0x10_1004), Some(0x5004));
    }

    #[test]
    fn protect_changes_only_mapped_pages() {
        let mem = space();
        mem.map(0x10_0000, 0, 0x2000, MemoryPermission::READ_WRITE);
        mem.protect(0x10_0000, 0x4000, MemoryPermission::READ);
        assert_eq!(mem.translate(0x10_1000).unwrap().1, MemoryPermission::READ);
        assert!(!mem.is_mapped(0x10_2000));
    }
}
