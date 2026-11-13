//! Shared, transfer and code memory objects.

use std::sync::Arc;

use lichii_common::{HleResult, PAddr, VAddr};
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use crate::memory::{MemoryManager, MemoryState, PageGroup, Pool};
use crate::process::Process;
use crate::resource_limit::{LimitableResource, Reservation};
use crate::result;

/// Memory shared between processes or with HLE services (`KSharedMemory`). The pages are
/// physically contiguous, so services can access them as one host slice.
pub struct SharedMemory {
    memory: Arc<MemoryManager>,
    owner: Option<u64>,
    base: PAddr,
    size: u64,
    owner_permission: MemoryPermission,
    user_permission: MemoryPermission,
}

impl SharedMemory {
    /// `owner` is the id of the process that gets `owner_permission`; everyone else gets
    /// `user_permission`. HLE services create shared memory without a guest owner.
    pub fn new(
        memory: Arc<MemoryManager>,
        owner: Option<u64>,
        pool: Pool,
        size: u64,
        owner_permission: MemoryPermission,
        user_permission: MemoryPermission,
    ) -> HleResult<Self> {
        let size = lichii_common::align_up(size, lichii_common::PAGE_SIZE);
        let base = memory.allocate_contiguous(pool, size, lichii_common::PAGE_SIZE)?;
        Ok(Self {
            memory,
            owner,
            base,
            size,
            owner_permission,
            user_permission,
        })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn physical_address(&self) -> PAddr {
        self.base
    }

    /// Host pointer to the first byte. Valid for `size()` bytes while the object lives.
    pub fn host_pointer(&self) -> *mut u8 {
        self.memory.physical().pointer(self.base)
    }

    pub fn read(&self, offset: u64, out: &mut [u8]) {
        assert!(
            offset + out.len() as u64 <= self.size,
            "shared memory read out of range"
        );
        // SAFETY: in range of the contiguous allocation, which lives as long as self.
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.host_pointer().add(offset as usize),
                out.as_mut_ptr(),
                out.len(),
            )
        };
    }

    pub fn write(&self, offset: u64, data: &[u8]) {
        assert!(
            offset + data.len() as u64 <= self.size,
            "shared memory write out of range"
        );
        // SAFETY: as in read.
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.host_pointer().add(offset as usize),
                data.len(),
            )
        };
    }

    fn group(&self) -> PageGroup {
        PageGroup {
            runs: vec![(self.base, self.size)],
        }
    }

    /// `svcMapSharedMemory`. The process keeps the object (and so its pages) alive until the
    /// mapping is removed or the process is destroyed.
    pub fn map(
        self: &Arc<Self>,
        process: &Process,
        addr: VAddr,
        size: u64,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        if size != self.size {
            return Err(result::INVALID_SIZE);
        }
        let allowed = if self.owner == Some(process.id()) {
            self.owner_permission
        } else {
            self.user_permission
        };
        if perm.0 & !allowed.0 != 0 || perm == MemoryPermission::NONE {
            return Err(result::INVALID_NEW_MEMORY_PERMISSION);
        }
        let mut mappings = process.shared_mappings.lock();
        process
            .page_table()
            .map_group(addr, &self.group(), MemoryState::SHARED, perm)?;
        mappings.push((addr, self.clone()));
        Ok(())
    }

    pub fn unmap(self: &Arc<Self>, process: &Process, addr: VAddr, size: u64) -> HleResult<()> {
        if size != self.size {
            return Err(result::INVALID_SIZE);
        }
        let mut mappings = process.shared_mappings.lock();
        let index = mappings
            .iter()
            .position(|(at, shared)| *at == addr && Arc::ptr_eq(shared, self))
            .ok_or(result::INVALID_MEMORY_REGION)?;
        process
            .page_table()
            .unmap_group(addr, size, MemoryState::SHARED)?;
        mappings.swap_remove(index);
        Ok(())
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        self.memory.free(self.base, self.size);
    }
}

/// A range of one process's memory lent to another (`KTransferMemory`). The owner loses access
/// (or keeps the given permission) until the object is closed.
pub struct TransferMemory {
    owner: Arc<Process>,
    address: VAddr,
    size: u64,
    permission: MemoryPermission,
    _reservation: Reservation,
}

impl TransferMemory {
    /// `svcCreateTransferMemory`. Counts against the owner's transfer memory limit until it is
    /// destroyed.
    pub fn new(
        owner: Arc<Process>,
        address: VAddr,
        size: u64,
        permission: MemoryPermission,
    ) -> HleResult<Self> {
        if ![
            MemoryPermission::NONE,
            MemoryPermission::READ,
            MemoryPermission::READ_WRITE,
        ]
        .contains(&permission)
        {
            return Err(result::INVALID_NEW_MEMORY_PERMISSION);
        }
        let reservation = owner
            .resource_limit()
            .reserve_owned(LimitableResource::TransferMemories, 1)?;
        owner
            .page_table()
            .lock_for_transfer(address, size, permission)?;
        Ok(Self {
            owner,
            address,
            size,
            permission,
            _reservation: reservation,
        })
    }

    pub fn owner(&self) -> &Arc<Process> {
        &self.owner
    }

    /// Address of the memory in the owner's address space.
    pub fn address(&self) -> VAddr {
        self.address
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn permission(&self) -> MemoryPermission {
        self.permission
    }

    /// Reads from the lent memory (services use this instead of mapping it).
    pub fn read(&self, offset: u64, out: &mut [u8]) -> HleResult<()> {
        if offset + out.len() as u64 > self.size {
            return Err(result::OUT_OF_RANGE);
        }
        self.owner
            .memory()
            .read_bytes(self.address + offset, out)
            .map_err(|_| result::INVALID_CURRENT_MEMORY)
    }

    pub fn write(&self, offset: u64, data: &[u8]) -> HleResult<()> {
        if offset + data.len() as u64 > self.size {
            return Err(result::OUT_OF_RANGE);
        }
        self.owner
            .memory()
            .write_bytes(self.address + offset, data)
            .map_err(|_| result::INVALID_CURRENT_MEMORY)
    }
}

impl Drop for TransferMemory {
    fn drop(&mut self) {
        let _ = self
            .owner
            .page_table()
            .unlock_for_transfer(self.address, self.size);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodeMemoryOperation {
    MapOwner = 0,
    MapSlave = 1,
    UnmapOwner = 2,
    UnmapSlave = 3,
}

impl CodeMemoryOperation {
    pub fn from_raw(raw: u32) -> HleResult<Self> {
        Ok(match raw {
            0 => Self::MapOwner,
            1 => Self::MapSlave,
            2 => Self::UnmapOwner,
            3 => Self::UnmapSlave,
            _ => return Err(result::INVALID_ENUM_VALUE),
        })
    }
}

/// Memory a process turns into code (`KCodeMemory`): mapped writable at an owner alias and
/// executable at a slave alias.
pub struct CodeMemory {
    owner: Arc<Process>,
    address: VAddr,
    group: PageGroup,
    mapped: Mutex<(Option<VAddr>, Option<VAddr>)>,
}

impl CodeMemory {
    pub fn new(owner: Arc<Process>, address: VAddr, size: u64) -> HleResult<Self> {
        let group = owner.page_table().lock_for_code_memory(address, size)?;
        Ok(Self {
            owner,
            address,
            group,
            mapped: Mutex::new((None, None)),
        })
    }

    pub fn size(&self) -> u64 {
        self.group.size()
    }

    /// `svcControlCodeMemory`.
    pub fn control(
        &self,
        op: CodeMemoryOperation,
        addr: VAddr,
        size: u64,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        if size != self.size() {
            return Err(result::INVALID_SIZE);
        }
        let table = self.owner.page_table();
        let mut mapped = self.mapped.lock();
        match op {
            CodeMemoryOperation::MapOwner => {
                if mapped.0.is_some() {
                    return Err(result::INVALID_STATE);
                }
                if perm != MemoryPermission::READ_WRITE {
                    return Err(result::INVALID_NEW_MEMORY_PERMISSION);
                }
                table.map_group(addr, &self.group, MemoryState::GENERATED_CODE, perm)?;
                mapped.0 = Some(addr);
            }
            CodeMemoryOperation::MapSlave => {
                if mapped.1.is_some() {
                    return Err(result::INVALID_STATE);
                }
                if perm != MemoryPermission::READ && perm != MemoryPermission::READ_EXECUTE {
                    return Err(result::INVALID_NEW_MEMORY_PERMISSION);
                }
                table.map_group(addr, &self.group, MemoryState::CODE_OUT, perm)?;
                mapped.1 = Some(addr);
            }
            CodeMemoryOperation::UnmapOwner => {
                if mapped.0 != Some(addr) {
                    return Err(result::INVALID_STATE);
                }
                table.unmap_group(addr, size, MemoryState::GENERATED_CODE)?;
                mapped.0 = None;
            }
            CodeMemoryOperation::UnmapSlave => {
                if mapped.1 != Some(addr) {
                    return Err(result::INVALID_STATE);
                }
                table.unmap_group(addr, size, MemoryState::CODE_OUT)?;
                mapped.1 = None;
            }
        }
        Ok(())
    }
}

impl Drop for CodeMemory {
    fn drop(&mut self) {
        let size = self.size();
        let table = self.owner.page_table();
        let (owner, slave) = *self.mapped.get_mut();
        if let Some(addr) = owner {
            let _ = table.unmap_group(addr, size, MemoryState::GENERATED_CODE);
        }
        if let Some(addr) = slave {
            let _ = table.unmap_group(addr, size, MemoryState::CODE_OUT);
        }
        let _ = table.unlock_for_code_memory(self.address, size);
    }
}
