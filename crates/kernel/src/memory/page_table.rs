//! A process's page table: the address-space layout, the block map and every mapping operation.

use std::sync::Arc;

use lichii_common::{HleResult, PAGE_SIZE, PAddr, VAddr, align_down, align_up};
use lichii_memory::{AddressSpace, MemoryPermission};
use parking_lot::Mutex;

use crate::memory::block::{BlockMap, Expect, MemoryInfo, MemoryState, attribute};
use crate::memory::manager::{MemoryManager, PageGroup, Pool};
use crate::result;
use crate::thread::TLS_SIZE;
use crate::util::Rng;

const GIB: u64 = 1 << 30;
const MIB: u64 = 1 << 20;
const REGION_ALIGNMENT: u64 = 2 * MIB;
/// `SetHeapSize` granularity.
pub const HEAP_ALIGNMENT: u64 = 2 * MIB;

/// A half-open address range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub start: VAddr,
    pub end: VAddr,
}

impl Region {
    pub const fn new(start: VAddr, size: u64) -> Self {
        Self {
            start,
            end: start + size,
        }
    }

    pub const fn size(self) -> u64 {
        self.end - self.start
    }

    pub const fn contains(self, addr: VAddr, size: u64) -> bool {
        addr >= self.start
            && size <= self.end - self.start
            && addr - self.start <= self.size() - size
    }

    pub const fn overlaps(self, addr: VAddr, size: u64) -> bool {
        addr < self.end && addr + size > self.start
    }
}

/// Inputs of the address-space layout (`KPageTableBase::InitializeForProcess`).
#[derive(Clone, Copy, Debug)]
pub struct LayoutParams {
    /// 32, 36 or 39.
    pub address_space_bits: u32,
    /// 32-bit processes without an alias region fold it into the heap.
    pub without_alias: bool,
    pub code_address: VAddr,
    pub code_size: u64,
    pub aslr: bool,
    /// Firmware 18+: grow the alias region by an eighth of the address space.
    pub alias_extra: bool,
}

/// The regions of a process address space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Regions {
    pub address_space_bits: u32,
    pub code: Region,
    /// The ASLR region (`GetInfo` 12/13).
    pub alias_code: Region,
    pub alias: Region,
    pub heap: Region,
    pub stack: Region,
    pub kernel_map: Region,
    pub alias_extra_size: u64,
}

impl Regions {
    pub fn compute(params: LayoutParams, rng: &mut Rng) -> HleResult<Self> {
        let bits = params.address_space_bits;
        let end = 1u64 << bits;
        let (mut alias_size, mut heap_size) = match bits {
            32 => (GIB, GIB),
            36 => (6 * GIB, 8 * GIB),
            39 => (64 * GIB, 16 * GIB),
            _ => return Err(result::INVALID_ENUM_VALUE),
        };
        if params.without_alias {
            heap_size += alias_size;
            alias_size = 0;
        }
        let (code, alias_code, stack_size, kmap_size, code_start, code_end);
        if bits == 39 {
            code = Region::new(128 * MIB, 512 * GIB - 128 * MIB);
            alias_code = code;
            stack_size = 2 * GIB;
            kmap_size = 64 * GIB;
            code_start = align_down(params.code_address, REGION_ALIGNMENT);
            code_end = align_up(params.code_address + params.code_size, REGION_ALIGNMENT);
        } else {
            let (small, large) = if bits == 32 {
                (
                    Region::new(2 * MIB, GIB - 2 * MIB),
                    Region::new(GIB, 3 * GIB),
                )
            } else {
                (
                    Region::new(128 * MIB, 2 * GIB - 128 * MIB),
                    Region::new(2 * GIB, 62 * GIB),
                )
            };
            code = small;
            alias_code = Region {
                start: small.start,
                end: large.end,
            };
            stack_size = 0;
            kmap_size = 0;
            code_start = code.start;
            code_end = code.end;
        }

        let (alloc_start, alloc_size) = if code_start - code.start >= end - code_end {
            (code.start, code_start - code.start)
        } else {
            (code_end, end - code_end)
        };
        let mut alias_extra_size = 0;
        if bits != 32 && params.alias_extra && alias_size != 0 {
            alias_extra_size = end >> 3;
            alias_size += alias_extra_size;
        }
        let needed = alias_size + heap_size + stack_size + kmap_size;
        if alloc_size < needed {
            return Err(result::OUT_OF_MEMORY);
        }
        let remaining = alloc_size - needed;
        let mut random = || {
            if params.aslr {
                rng.below(remaining / REGION_ALIGNMENT + 1) * REGION_ALIGNMENT
            } else {
                0
            }
        };
        let (alias_rnd, heap_rnd, stack_rnd, kmap_rnd) = (random(), random(), random(), random());

        let mut alias = Region::new(alloc_start + alias_rnd, alias_size);
        let mut heap = Region::new(alloc_start + heap_rnd, heap_size);
        let shift = |r: &mut Region, by: u64| {
            r.start += by;
            r.end += by;
        };
        if alias_rnd <= heap_rnd {
            shift(&mut heap, alias_size);
        } else {
            shift(&mut alias, heap_size);
        }
        let mut stack = code;
        if stack_size != 0 {
            stack = Region::new(alloc_start + stack_rnd, stack_size);
            if alias_rnd < stack_rnd {
                shift(&mut stack, alias_size);
            } else {
                shift(&mut alias, stack_size);
            }
            if heap_rnd < stack_rnd {
                shift(&mut stack, heap_size);
            } else {
                shift(&mut heap, stack_size);
            }
        }
        let mut kernel_map = code;
        if kmap_size != 0 {
            kernel_map = Region::new(alloc_start + kmap_rnd, kmap_size);
            if alias_rnd < kmap_rnd {
                shift(&mut kernel_map, alias_size);
            } else {
                shift(&mut alias, kmap_size);
            }
            if heap_rnd < kmap_rnd {
                shift(&mut kernel_map, heap_size);
            } else {
                shift(&mut heap, kmap_size);
            }
            if stack_size != 0 {
                if stack_rnd < kmap_rnd {
                    shift(&mut kernel_map, stack_size);
                } else {
                    shift(&mut stack, kmap_size);
                }
            }
        }
        Ok(Self {
            address_space_bits: bits,
            code,
            alias_code,
            alias,
            heap,
            stack,
            kernel_map,
            alias_extra_size,
        })
    }
}

struct TlsPage {
    addr: VAddr,
    used: u8,
}

struct State {
    blocks: BlockMap,
    heap_end: VAddr,
    physical_mapped: u64,
    tls_pages: Vec<TlsPage>,
}

/// Called with each range whose executable pages an operation unmapped or reprotected, so the
/// cores drop the code they translated from it.
pub type CodeInvalidator = Box<dyn Fn(VAddr, u64) + Send + Sync>;

/// The process page table. Every operation validates against the block map, then updates the
/// block map and the [`AddressSpace`] together under one lock.
pub struct PageTable {
    space: Arc<AddressSpace>,
    invalidate_code: CodeInvalidator,
    memory: Arc<MemoryManager>,
    pool: Pool,
    regions: Regions,
    state: Mutex<State>,
}

const USER_RW: MemoryPermission = MemoryPermission::READ_WRITE;

fn has_code(blocks: &BlockMap, addr: VAddr, size: u64) -> bool {
    blocks
        .iter_range(addr, size)
        .any(|(_, block)| block.perm.contains(MemoryPermission::EXECUTE))
}

impl PageTable {
    pub fn new(
        space: Arc<AddressSpace>,
        memory: Arc<MemoryManager>,
        pool: Pool,
        regions: Regions,
        invalidate_code: CodeInvalidator,
    ) -> Self {
        let heap_end = regions.heap.start;
        Self {
            invalidate_code,
            state: Mutex::new(State {
                blocks: BlockMap::new(space.address_space_bits()),
                heap_end,
                physical_mapped: 0,
                tls_pages: Vec::new(),
            }),
            space,
            memory,
            pool,
            regions,
        }
    }

    /// Unmaps a range whose blocks `blocks` still describes, dropping translated code from it.
    fn unmap_space(&self, blocks: &BlockMap, addr: VAddr, size: u64) {
        let code = has_code(blocks, addr, size);
        self.space.unmap(addr, size);
        if code {
            (self.invalidate_code)(addr, size);
        }
    }

    /// Reprotects a range whose blocks `blocks` still describes, dropping translated code from it.
    fn protect_space(&self, blocks: &BlockMap, addr: VAddr, size: u64, perm: MemoryPermission) {
        let code = has_code(blocks, addr, size);
        self.space.protect(addr, size, perm);
        if code {
            (self.invalidate_code)(addr, size);
        }
    }

    pub fn address_space(&self) -> &Arc<AddressSpace> {
        &self.space
    }

    pub fn memory_manager(&self) -> &Arc<MemoryManager> {
        &self.memory
    }

    pub fn pool(&self) -> Pool {
        self.pool
    }

    pub fn regions(&self) -> &Regions {
        &self.regions
    }

    pub fn contains(&self, addr: VAddr, size: u64) -> bool {
        Region::new(0, 1 << self.regions.address_space_bits).contains(addr, size)
    }

    /// `CanContain(addr, size, AliasCode)`: inside the ASLR region and clear of the heap and
    /// alias regions.
    pub fn can_contain_alias_code(&self, addr: VAddr, size: u64) -> bool {
        let r = &self.regions;
        r.alias_code.contains(addr, size)
            && !r.heap.overlaps(addr, size)
            && !r.alias.overlaps(addr, size)
    }

    pub fn query(&self, addr: VAddr) -> MemoryInfo {
        self.state.lock().blocks.query(addr)
    }

    /// Bytes of heap plus memory mapped with `MapPhysicalMemory`.
    pub fn normal_memory_size(&self) -> u64 {
        let s = self.state.lock();
        s.heap_end - self.regions.heap.start + s.physical_mapped
    }

    pub fn heap_size(&self) -> u64 {
        self.state.lock().heap_end - self.regions.heap.start
    }

    fn map_runs(&self, addr: VAddr, group: &PageGroup, perm: MemoryPermission) {
        let mut va = addr;
        for &(pa, size) in &group.runs {
            self.space.map(va, pa, size, perm);
            va += size;
        }
    }

    /// The physical runs behind a mapped range.
    fn physical_runs(&self, addr: VAddr, size: u64) -> PageGroup {
        let mut runs: Vec<(PAddr, u64)> = Vec::new();
        for page in 0..size / PAGE_SIZE {
            let pa = self
                .space
                .physical_address(addr + page * PAGE_SIZE)
                .expect("mapped page");
            match runs.last_mut() {
                Some((start, len)) if *start + *len == pa => *len += PAGE_SIZE,
                _ => runs.push((pa, PAGE_SIZE)),
            }
        }
        PageGroup { runs }
    }

    fn check_args(&self, addr: VAddr, size: u64) -> HleResult<()> {
        if !addr.is_multiple_of(PAGE_SIZE) {
            return Err(result::INVALID_ADDRESS);
        }
        if size == 0 || !size.is_multiple_of(PAGE_SIZE) {
            return Err(result::INVALID_SIZE);
        }
        if !self.contains(addr, size) {
            return Err(result::INVALID_CURRENT_MEMORY);
        }
        Ok(())
    }

    /// Allocates fresh pages from the process pool and maps them at a fixed, free address.
    pub fn map_new(
        &self,
        addr: VAddr,
        size: u64,
        state: MemoryState,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        s.blocks
            .check(addr, size, Expect::state(MemoryState::FREE, 0))?;
        let group = self.memory.allocate(self.pool, size)?;
        self.map_runs(addr, &group, perm);
        s.blocks.set(addr, size, state, perm, 0);
        Ok(())
    }

    /// Allocates fresh pages and maps them at a free spot of `region`.
    pub fn map_new_in(
        &self,
        region: Region,
        size: u64,
        state: MemoryState,
        perm: MemoryPermission,
    ) -> HleResult<VAddr> {
        self.map_new_from(self.pool, region, size, state, perm)
    }

    /// [`PageTable::map_new_in`] with pages from `pool` instead of the process pool, for memory
    /// a service keeps in the process (the graphics server's GPU-only regions). Freeing returns
    /// the pages to `pool`.
    pub fn map_new_from(
        &self,
        pool: Pool,
        region: Region,
        size: u64,
        state: MemoryState,
        perm: MemoryPermission,
    ) -> HleResult<VAddr> {
        let mut s = self.state.lock();
        let addr = s
            .blocks
            .find_free(region.start, region.end, size, PAGE_SIZE, PAGE_SIZE)
            .ok_or(result::OUT_OF_MEMORY)?;
        let group = self.memory.allocate(pool, size)?;
        self.map_runs(addr, &group, perm);
        s.blocks.set(addr, size, state, perm, 0);
        Ok(addr)
    }

    /// Maps existing pages (shared memory, code memory) at a fixed, free address.
    pub fn map_group(
        &self,
        addr: VAddr,
        group: &PageGroup,
        state: MemoryState,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        let size = group.size();
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        s.blocks
            .check(addr, size, Expect::state(MemoryState::FREE, 0))?;
        self.map_runs(addr, group, perm);
        s.blocks.set(addr, size, state, perm, 0);
        Ok(())
    }

    /// Unmaps a range mapped with [`PageTable::map_group`] in `state`, keeping the pages.
    pub fn unmap_group(&self, addr: VAddr, size: u64, state: MemoryState) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        s.blocks.check(
            addr,
            size,
            Expect::state(
                state,
                attribute::LOCKED | attribute::IPC_LOCKED | attribute::DEVICE_SHARED,
            ),
        )?;
        self.unmap_space(&s.blocks, addr, size);
        s.blocks
            .set(addr, size, MemoryState::FREE, MemoryPermission::NONE, 0);
        Ok(())
    }

    /// Unmaps a range in `state` and frees its pages.
    pub fn unmap_and_free(&self, addr: VAddr, size: u64, state: MemoryState) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        s.blocks.check(
            addr,
            size,
            Expect::state(
                state,
                attribute::LOCKED | attribute::IPC_LOCKED | attribute::DEVICE_SHARED,
            ),
        )?;
        let group = self.physical_runs(addr, size);
        self.unmap_space(&s.blocks, addr, size);
        self.memory.free_group(&group);
        s.blocks
            .set(addr, size, MemoryState::FREE, MemoryPermission::NONE, 0);
        Ok(())
    }

    /// Changes permission and state of a range without remapping (used to reprotect loaded
    /// code segments).
    pub fn set_process_memory_permission(
        &self,
        addr: VAddr,
        size: u64,
        state: MemoryState,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        s.blocks
            .check_each(addr, size, Expect::flag(MemoryState::FLAG_MAPPED, 0, 0, 0))?;
        self.protect_space(&s.blocks, addr, size, perm);
        s.blocks.set(addr, size, state, perm, 0);
        Ok(())
    }

    /// `MapCodeMemory` (`svcMapProcessCodeMemory`, ro): aliases the heap pages at `src` as
    /// code at `dst` in the ASLR region. Both stay inaccessible to the process until it
    /// reprotects `dst` with [`PageTable::set_code_permission`]; `src` stays locked until
    /// [`PageTable::unmap_code_memory`].
    pub fn map_code_memory(&self, dst: VAddr, src: VAddr, size: u64) -> HleResult<()> {
        if !self.can_contain_alias_code(dst, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        self.check_args(src, size)?;
        self.check_args(dst, size)?;
        let mut s = self.state.lock();
        let (src_state, _, _) = s.blocks.check(
            src,
            size,
            Expect {
                state_mask: u32::MAX,
                state: MemoryState::NORMAL.0,
                perm_mask: 7,
                perm: USER_RW.0,
                attr_mask: u32::MAX,
                attr: 0,
            },
        )?;
        s.blocks
            .check(dst, size, Expect::state(MemoryState::FREE, u32::MAX))?;
        let group = self.physical_runs(src, size);
        self.protect_space(&s.blocks, src, size, MemoryPermission::NONE);
        self.map_runs(dst, &group, MemoryPermission::NONE);
        s.blocks.set(
            src,
            size,
            src_state,
            MemoryPermission::NONE,
            attribute::LOCKED,
        );
        s.blocks.set(
            dst,
            size,
            MemoryState::ALIAS_CODE,
            MemoryPermission::NONE,
            0,
        );
        Ok(())
    }

    /// `UnmapCodeMemory`: undoes [`PageTable::map_code_memory`] and gives `src` back to the
    /// process as read-write heap.
    pub fn unmap_code_memory(&self, dst: VAddr, src: VAddr, size: u64) -> HleResult<()> {
        if !self.can_contain_alias_code(dst, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        self.check_args(src, size)?;
        self.check_args(dst, size)?;
        let mut s = self.state.lock();
        s.blocks.check(
            src,
            size,
            Expect {
                state_mask: u32::MAX,
                state: MemoryState::NORMAL.0,
                perm_mask: 0,
                perm: 0,
                attr_mask: u32::MAX,
                attr: attribute::LOCKED,
            },
        )?;
        s.blocks.check_each(
            dst,
            size,
            Expect::flag(
                MemoryState::FLAG_CAN_CODE_ALIAS,
                0,
                0,
                !attribute::PERMISSION_LOCKED,
            ),
        )?;
        if self.physical_runs(dst, size) != self.physical_runs(src, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        self.unmap_space(&s.blocks, dst, size);
        self.protect_space(&s.blocks, src, size, USER_RW);
        s.blocks
            .set(dst, size, MemoryState::FREE, MemoryPermission::NONE, 0);
        s.blocks.set(src, size, MemoryState::NORMAL, USER_RW, 0);
        Ok(())
    }

    /// `SetProcessMemoryPermission` on code: a writable permission turns `Code` into
    /// `CodeData` (and `AliasCode` into `AliasCodeData`). Making pages executable drops the
    /// code translated from them, so the process runs what it wrote there.
    pub fn set_code_permission(
        &self,
        addr: VAddr,
        size: u64,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        let (old_state, old_perm, _) = s.blocks.check(
            addr,
            size,
            Expect::flag(MemoryState::FLAG_CODE, 0, 0, u32::MAX),
        )?;
        let state = if perm.contains(MemoryPermission::WRITE) {
            match old_state {
                MemoryState::CODE => MemoryState::CODE_DATA,
                MemoryState::ALIAS_CODE => MemoryState::ALIAS_CODE_DATA,
                _ => return Err(result::INVALID_STATE),
            }
        } else {
            old_state
        };
        if old_perm == perm && old_state == state {
            return Ok(());
        }
        self.protect_space(&s.blocks, addr, size, perm);
        s.blocks.set(addr, size, state, perm, 0);
        if perm.contains(MemoryPermission::EXECUTE) && !old_perm.contains(MemoryPermission::EXECUTE)
        {
            (self.invalidate_code)(addr, size);
        }
        Ok(())
    }

    /// `svcSetHeapSize`. Returns the heap base.
    pub fn set_heap_size(&self, size: u64) -> HleResult<VAddr> {
        if !size.is_multiple_of(HEAP_ALIGNMENT) || size > self.regions.heap.size() {
            return Err(result::INVALID_SIZE);
        }
        let start = self.regions.heap.start;
        let mut s = self.state.lock();
        let current = s.heap_end - start;
        if size > current {
            let grow = size - current;
            s.blocks
                .check(s.heap_end, grow, Expect::state(MemoryState::FREE, 0))?;
            let group = self.memory.allocate(self.pool, grow)?;
            self.map_runs(s.heap_end, &group, USER_RW);
            let heap_end = s.heap_end;
            s.blocks
                .set(heap_end, grow, MemoryState::NORMAL, USER_RW, 0);
        } else if size < current {
            let shrink_start = start + size;
            let shrink = current - size;
            s.blocks.check(
                shrink_start,
                shrink,
                Expect {
                    state_mask: u32::MAX,
                    state: MemoryState::NORMAL.0,
                    perm_mask: 7,
                    perm: USER_RW.0,
                    attr_mask: u32::MAX,
                    attr: 0,
                },
            )?;
            let group = self.physical_runs(shrink_start, shrink);
            self.unmap_space(&s.blocks, shrink_start, shrink);
            self.memory.free_group(&group);
            s.blocks.set(
                shrink_start,
                shrink,
                MemoryState::FREE,
                MemoryPermission::NONE,
                0,
            );
        }
        s.heap_end = start + size;
        Ok(start)
    }

    /// `svcSetMemoryPermission`.
    pub fn set_memory_permission(
        &self,
        addr: VAddr,
        size: u64,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        self.check_args(addr, size)?;
        if ![MemoryPermission::NONE, MemoryPermission::READ, USER_RW].contains(&perm) {
            return Err(result::INVALID_NEW_MEMORY_PERMISSION);
        }
        let mut s = self.state.lock();
        let (state, old, attr) = s.blocks.check(
            addr,
            size,
            Expect::flag(MemoryState::FLAG_CAN_REPROTECT, 0, 0, u32::MAX),
        )?;
        if old != perm {
            self.protect_space(&s.blocks, addr, size, perm);
            s.blocks.set(addr, size, state, perm, attr);
        }
        Ok(())
    }

    /// `svcSetMemoryAttribute` and the Switch 2 `svcSetGpuMemoryAttribute` (0x80).
    pub fn set_memory_attribute(
        &self,
        addr: VAddr,
        size: u64,
        mask: u32,
        attr: u32,
    ) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        let ignore = attribute::SET_MASK | attribute::DEVICE_SHARED;
        s.blocks.check_ignoring(
            addr,
            size,
            Expect {
                state_mask: MemoryState::FLAG_CAN_CHANGE_ATTRIBUTE,
                state: MemoryState::FLAG_CAN_CHANGE_ATTRIBUTE,
                perm_mask: 0,
                perm: 0,
                attr_mask: !ignore,
                attr: 0,
            },
            ignore,
        )?;
        s.blocks
            .update(addr, size, |b| b.attr = (b.attr & !mask) | (attr & mask));
        Ok(())
    }

    /// `svcMapMemory`: aliases heap pages at `dst` in the stack region.
    pub fn map_memory(&self, dst: VAddr, src: VAddr, size: u64) -> HleResult<()> {
        self.check_args(src, size)?;
        self.check_args(dst, size)?;
        if !self.regions.stack.contains(dst, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        let mut s = self.state.lock();
        let (src_state, _, _) = s.blocks.check(
            src,
            size,
            Expect::flag(MemoryState::FLAG_CAN_ALIAS, 7, USER_RW.0, u32::MAX),
        )?;
        s.blocks
            .check(dst, size, Expect::state(MemoryState::FREE, 0))?;
        let group = self.physical_runs(src, size);
        self.protect_space(&s.blocks, src, size, MemoryPermission::NONE);
        s.blocks.set(
            src,
            size,
            src_state,
            MemoryPermission::NONE,
            attribute::LOCKED,
        );
        self.map_runs(dst, &group, USER_RW);
        s.blocks.set(dst, size, MemoryState::STACK, USER_RW, 0);
        Ok(())
    }

    /// `svcUnmapMemory`.
    pub fn unmap_memory(&self, dst: VAddr, src: VAddr, size: u64) -> HleResult<()> {
        self.check_args(src, size)?;
        self.check_args(dst, size)?;
        if !self.regions.stack.contains(dst, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        let mut s = self.state.lock();
        let (src_state, _, _) = s.blocks.check(
            src,
            size,
            Expect {
                state_mask: MemoryState::FLAG_CAN_ALIAS,
                state: MemoryState::FLAG_CAN_ALIAS,
                perm_mask: 7,
                perm: 0,
                attr_mask: u32::MAX,
                attr: attribute::LOCKED,
            },
        )?;
        s.blocks
            .check(dst, size, Expect::state(MemoryState::STACK, u32::MAX))?;
        if self.physical_runs(src, size) != self.physical_runs(dst, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        self.unmap_space(&s.blocks, dst, size);
        s.blocks
            .set(dst, size, MemoryState::FREE, MemoryPermission::NONE, 0);
        self.protect_space(&s.blocks, src, size, USER_RW);
        s.blocks.set(src, size, src_state, USER_RW, 0);
        Ok(())
    }

    /// `svcMapPhysicalMemory`: backs the unmapped parts of an alias-region range with new pages.
    pub fn map_physical_memory(&self, addr: VAddr, size: u64) -> HleResult<()> {
        self.check_args(addr, size)?;
        if !self.regions.alias.contains(addr, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        let mut s = self.state.lock();
        let parts: Vec<(VAddr, u64, MemoryState)> = s
            .blocks
            .iter_range(addr, size)
            .map(|(start, b)| (start, b.size, b.state))
            .collect();
        if parts
            .iter()
            .any(|&(_, _, st)| st != MemoryState::FREE && st != MemoryState::NORMAL)
        {
            return Err(result::INVALID_CURRENT_MEMORY);
        }
        let free: u64 = parts
            .iter()
            .filter(|p| p.2 == MemoryState::FREE)
            .map(|p| p.1)
            .sum();
        let group = self.memory.allocate(self.pool, free)?;
        let mut runs = group.runs.iter().copied();
        let mut run = None;
        for &(start, len, st) in &parts {
            if st != MemoryState::FREE {
                continue;
            }
            let mut va = start;
            while va < start + len {
                let (pa, size) = run
                    .take()
                    .or_else(|| runs.next())
                    .expect("allocated enough pages");
                let take = size.min(start + len - va);
                self.space.map(va, pa, take, USER_RW);
                if take < size {
                    run = Some((pa + take, size - take));
                }
                va += take;
            }
            s.blocks.set(start, len, MemoryState::NORMAL, USER_RW, 0);
        }
        s.physical_mapped += free;
        Ok(())
    }

    /// `svcUnmapPhysicalMemory`.
    pub fn unmap_physical_memory(&self, addr: VAddr, size: u64) -> HleResult<()> {
        self.check_args(addr, size)?;
        if !self.regions.alias.contains(addr, size) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        let mut s = self.state.lock();
        let parts: Vec<(VAddr, u64, MemoryState, u32)> = s
            .blocks
            .iter_range(addr, size)
            .map(|(start, b)| (start, b.size, b.state, b.attr))
            .collect();
        if parts.iter().any(|&(_, _, st, attr)| {
            (st != MemoryState::FREE && st != MemoryState::NORMAL) || attr != 0
        }) {
            return Err(result::INVALID_CURRENT_MEMORY);
        }
        let mut freed = 0;
        for &(start, len, st, _) in &parts {
            if st != MemoryState::NORMAL {
                continue;
            }
            let group = self.physical_runs(start, len);
            self.unmap_space(&s.blocks, start, len);
            self.memory.free_group(&group);
            s.blocks
                .set(start, len, MemoryState::FREE, MemoryPermission::NONE, 0);
            freed += len;
        }
        s.physical_mapped = s.physical_mapped.saturating_sub(freed);
        Ok(())
    }

    /// Locks a range for a transfer memory: the owner keeps `perm` (or loses access entirely
    /// when it is `NONE`) until [`PageTable::unlock_for_transfer`].
    pub fn lock_for_transfer(
        &self,
        addr: VAddr,
        size: u64,
        perm: MemoryPermission,
    ) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        let (state, _, _) = s.blocks.check(
            addr,
            size,
            Expect::flag(MemoryState::FLAG_CAN_TRANSFER, 7, USER_RW.0, u32::MAX),
        )?;
        self.protect_space(&s.blocks, addr, size, perm);
        s.blocks.set(addr, size, state, perm, attribute::LOCKED);
        Ok(())
    }

    pub fn unlock_for_transfer(&self, addr: VAddr, size: u64) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        let (state, _, _) = s.blocks.check(
            addr,
            size,
            Expect {
                state_mask: MemoryState::FLAG_CAN_TRANSFER,
                state: MemoryState::FLAG_CAN_TRANSFER,
                perm_mask: 0,
                perm: 0,
                attr_mask: u32::MAX,
                attr: attribute::LOCKED,
            },
        )?;
        self.protect_space(&s.blocks, addr, size, USER_RW);
        s.blocks.set(addr, size, state, USER_RW, 0);
        Ok(())
    }

    /// Locks a range for code memory (`svcCreateCodeMemory`) and returns its pages.
    pub fn lock_for_code_memory(&self, addr: VAddr, size: u64) -> HleResult<PageGroup> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        let (state, _, _) = s.blocks.check(
            addr,
            size,
            Expect::flag(MemoryState::FLAG_CAN_CODE_MEMORY, 7, USER_RW.0, u32::MAX),
        )?;
        let group = self.physical_runs(addr, size);
        self.protect_space(&s.blocks, addr, size, MemoryPermission::NONE);
        s.blocks
            .set(addr, size, state, MemoryPermission::NONE, attribute::LOCKED);
        Ok(group)
    }

    pub fn unlock_for_code_memory(&self, addr: VAddr, size: u64) -> HleResult<()> {
        self.check_args(addr, size)?;
        let mut s = self.state.lock();
        let (state, _, _) = s.blocks.check(
            addr,
            size,
            Expect {
                state_mask: MemoryState::FLAG_CAN_CODE_MEMORY,
                state: MemoryState::FLAG_CAN_CODE_MEMORY,
                perm_mask: 0,
                perm: 0,
                attr_mask: u32::MAX,
                attr: attribute::LOCKED,
            },
        )?;
        self.protect_space(&s.blocks, addr, size, USER_RW);
        s.blocks.set(addr, size, state, USER_RW, 0);
        Ok(())
    }

    /// Reserves a 0x200-byte TLS slot, mapping a new thread-local page when all are full.
    pub fn allocate_tls(&self) -> HleResult<VAddr> {
        let mut s = self.state.lock();
        let per_page = (PAGE_SIZE / TLS_SIZE) as u32;
        for page in &mut s.tls_pages {
            if page.used.count_ones() < per_page {
                let slot = (!page.used).trailing_zeros();
                page.used |= 1 << slot;
                return Ok(page.addr + slot as u64 * TLS_SIZE);
            }
        }
        let region = self.regions.kernel_map;
        let addr = s
            .blocks
            .find_free(region.start, region.end, PAGE_SIZE, PAGE_SIZE, 0)
            .ok_or(result::OUT_OF_MEMORY)?;
        let group = self.memory.allocate(self.pool, PAGE_SIZE)?;
        self.map_runs(addr, &group, USER_RW);
        s.blocks
            .set(addr, PAGE_SIZE, MemoryState::THREAD_LOCAL, USER_RW, 0);
        s.tls_pages.push(TlsPage { addr, used: 1 });
        Ok(addr)
    }

    pub fn free_tls(&self, addr: VAddr) {
        let mut s = self.state.lock();
        let page_addr = addr & !(PAGE_SIZE - 1);
        let Some(index) = s.tls_pages.iter().position(|p| p.addr == page_addr) else {
            return;
        };
        let slot = (addr - page_addr) / TLS_SIZE;
        s.tls_pages[index].used &= !(1 << slot);
        if s.tls_pages[index].used == 0 {
            s.tls_pages.swap_remove(index);
            let group = self.physical_runs(page_addr, PAGE_SIZE);
            self.unmap_space(&s.blocks, page_addr, PAGE_SIZE);
            self.memory.free_group(&group);
            s.blocks.set(
                page_addr,
                PAGE_SIZE,
                MemoryState::FREE,
                MemoryPermission::NONE,
                0,
            );
        }
    }

    /// Frees every page the process owns (on process destruction). Stacks are not included:
    /// `MapMemory` aliases share the stack state, so the process frees its main stack itself.
    pub(crate) fn release_all(&self) {
        let mut s = self.state.lock();
        let mapped: Vec<(VAddr, u64, bool)> = s
            .blocks
            .iter_range(0, s.blocks.end())
            .filter(|(_, b)| {
                !matches!(
                    b.state,
                    MemoryState::FREE | MemoryState::INACCESSIBLE | MemoryState::KERNEL
                )
            })
            .map(|(start, b)| {
                let owned = matches!(
                    b.state,
                    MemoryState::NORMAL
                        | MemoryState::CODE
                        | MemoryState::CODE_DATA
                        | MemoryState::THREAD_LOCAL
                );
                (start, b.size, owned)
            })
            .collect();
        let (mut runs_t, mut free_t, mut unmap_t) = Default::default();
        let (blocks, mut bytes, mut owned_bytes) = (mapped.len(), 0u64, 0u64);
        for (start, size, owned) in mapped {
            bytes += size;
            if owned {
                owned_bytes += size;
                let t = std::time::Instant::now();
                let group = self.physical_runs(start, size);
                runs_t += t.elapsed();
                let t = std::time::Instant::now();
                self.memory.free_group(&group);
                free_t += t.elapsed();
            }
            let t = std::time::Instant::now();
            self.unmap_space(&s.blocks, start, size);
            unmap_t += t.elapsed();
        }
        let (runs_t, free_t, unmap_t): (
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ) = (runs_t, free_t, unmap_t);
        tracing::info!(
            blocks,
            bytes,
            owned_bytes,
            ?runs_t,
            ?free_t,
            ?unmap_t,
            "diag exit: release_all"
        );
        let end = s.blocks.end();
        s.blocks
            .set(0, end, MemoryState::FREE, MemoryPermission::NONE, 0);
        s.tls_pages.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(bits: u32) -> LayoutParams {
        LayoutParams {
            address_space_bits: bits,
            without_alias: false,
            code_address: 0x800_0000,
            code_size: 0x100_0000,
            aslr: false,
            alias_extra: false,
        }
    }

    fn disjoint(a: Region, b: Region) -> bool {
        a.end <= b.start || b.end <= a.start
    }

    #[test]
    fn layout_39_bit_without_aslr() {
        let r = Regions::compute(params(39), &mut Rng::new(1)).unwrap();
        assert_eq!(r.code, Region::new(0x800_0000, 512 * GIB - 0x800_0000));
        assert_eq!(r.alias.size(), 64 * GIB);
        assert_eq!(r.heap.size(), 16 * GIB);
        assert_eq!(r.stack.size(), 2 * GIB);
        assert_eq!(r.kernel_map.size(), 64 * GIB);
        assert_eq!(r.alias.start, 0x900_0000 + 2 * GIB + 64 * GIB);
        assert!(
            disjoint(r.alias, r.heap)
                && disjoint(r.heap, r.stack)
                && disjoint(r.stack, r.kernel_map)
        );
    }

    #[test]
    fn layout_regions_never_overlap_with_aslr() {
        let mut rng = Rng::new(7);
        for bits in [32, 36, 39] {
            for _ in 0..50 {
                let mut p = params(bits);
                p.aslr = true;
                p.alias_extra = bits == 39;
                p.code_address = if bits == 32 { 0x20_0000 } else { 0x800_0000 };
                let r = Regions::compute(p, &mut rng).unwrap();
                let end = 1u64 << bits;
                for region in [r.alias, r.heap, r.stack, r.kernel_map] {
                    assert!(region.end <= end);
                }
                assert!(disjoint(r.alias, r.heap));
                if bits == 39 {
                    assert!(disjoint(r.alias, r.stack) && disjoint(r.heap, r.kernel_map));
                    assert_eq!(r.alias_extra_size, end >> 3);
                }
            }
        }
    }
}
