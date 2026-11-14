use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};

use lichii_common::{PAGE_BITS, PAGE_MASK, PAGE_SIZE, PAddr, VAddr};
use parking_lot::{Mutex, MutexGuard, RwLock};

use crate::PhysicalMemory;
use crate::fault::{FaultHandler, Registration};
use crate::host::{Arena, HostProtection};
use crate::page_table::{FLUSH_TRAP, PageTable, TRACKED};
use crate::tracking::{DirtyPages, PageFlags, ReverseMap};

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
    /// Declared before `arena` so it unregisters first: once the arena is unmapped the host can
    /// hand the range to a new arena, and a stale registration listed ahead of the new one would
    /// take its faults.
    registration: Registration,
    arena: Arena,
}

impl Fastmem {
    #[cfg(all(test, windows))]
    pub(crate) fn arena(&self) -> &Arena {
        &self.arena
    }

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

/// Called on a flush trap with the physical page it hit; returns false when no device address
/// maps the page any more, so nothing can write it back.
pub(crate) type FlushSink = Arc<dyn Fn(PAddr) -> bool + Send + Sync>;

/// A process's guest virtual address space.
///
/// Translation and accesses are lock-free. `map`, `unmap`, `protect`, write tracking and flush
/// traps ([`crate::tracking`]) are serialised so the table, the fastmem arena and the reverse map
/// always agree. Writes from host code report themselves to write tracking: use the write methods
/// here, or [`AddressSpace::write_with`] for a write through a host pointer. Host code that reads
/// through a pointer from [`AddressSpace::translate`] calls [`AddressSpace::flush_gpu_writes`]
/// first.
pub struct AddressSpace {
    table: PageTable,
    physical: Arc<PhysicalMemory>,
    fastmem: Option<Fastmem>,
    update: Mutex<Tracking>,
    /// Physical pages a cache holds that no CPU write untracked since: the flag writes check.
    tracked: PageFlags,
    /// Physical pages with a flush trap: GPU-modified data not written back yet.
    trapped: PageFlags,
    dirty: DirtyPages,
    flush_sink: RwLock<Option<FlushSink>>,
    /// Calls that armed or disarmed flush traps under the lock.
    #[cfg(test)]
    pub(crate) trap_updates: std::sync::atomic::AtomicU64,
}

/// Write-tracking state that changes with the mappings.
struct Tracking {
    /// The virtual pages mapping each physical page.
    reverse: ReverseMap,
    /// How many [`crate::CachedPages`] guards hold each physical page.
    cached: Box<[u16]>,
}

impl Tracking {
    fn new(physical_size: u64) -> Self {
        Self {
            reverse: ReverseMap::new(physical_size),
            cached: vec![0; (physical_size / PAGE_SIZE) as usize].into_boxed_slice(),
        }
    }
}

impl AddressSpace {
    /// `address_space_bits` is 32, 36 or 39 for Horizon processes.
    pub fn new(physical: Arc<PhysicalMemory>, address_space_bits: u32) -> Self {
        Self {
            table: PageTable::new(address_space_bits),
            update: Mutex::new(Tracking::new(physical.size())),
            tracked: PageFlags::new(physical.size()),
            trapped: PageFlags::new(physical.size()),
            dirty: DirtyPages::new(physical.size()),
            flush_sink: RwLock::new(None),
            #[cfg(test)]
            trap_updates: Default::default(),
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
            update: Mutex::new(Tracking::new(physical.size())),
            tracked: PageFlags::new(physical.size()),
            trapped: PageFlags::new(physical.size()),
            dirty: DirtyPages::new(physical.size()),
            flush_sink: RwLock::new(None),
            #[cfg(test)]
            trap_updates: Default::default(),
            physical,
            fastmem: Some(Fastmem {
                registration,
                arena,
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
        let mut tracking = self.lock_for_rebuild(va, size, Some(pa));
        let mut caps = Vec::new();
        for page in 0..size / PAGE_SIZE {
            let (addr, page_pa) = (va + page * PAGE_SIZE, pa + page * PAGE_SIZE);
            if let Some(old) = self.physical_address(addr) {
                tracking.reverse.remove(old, addr);
            }
            let (flags, cap) = self.page_state(page_pa);
            self.table
                .set(addr, self.physical.pointer(page_pa), perm, flags);
            tracking.reverse.insert(page_pa, addr);
            if let Some(cap) = cap {
                caps.push((addr as usize, cap));
            }
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
                    &caps,
                )
                .unwrap_or_else(|err| panic!("fastmem map {va:#x}+{size:#x}: {err}"));
        }
        self.publish_exposed();
        drop(tracking);
    }

    pub fn unmap(&self, va: VAddr, size: u64) {
        let mut tracking = self.lock_for_rebuild(va, size, None);
        self.clear_pages(&mut tracking, va, size);
        if let Some(fastmem) = &self.fastmem {
            fastmem
                .arena
                .unmap(self.physical.backing(), va as usize, size as usize)
                .unwrap_or_else(|err| panic!("fastmem unmap {va:#x}+{size:#x}: {err}"));
        }
        self.publish_exposed();
        drop(tracking);
    }

    /// Unmaps `[va, va + size)` like [`AddressSpace::unmap`] but leaves its fastmem views for
    /// [`AddressSpace::release_fastmem`]. Only for an address space that runs no more guest code,
    /// since the arena still shows the old pages.
    pub fn unmap_retired(&self, va: VAddr, size: u64) {
        let mut tracking = self.update.lock();
        self.clear_pages(&mut tracking, va, size);
    }

    /// Removes every fastmem view. The host then frees the page tables of every page the title
    /// touched, which takes most of a second for 10 GiB on Windows, so a process exit does it
    /// after reporting the exit.
    pub fn release_fastmem(&self) {
        let _tracking = self.update.lock();
        if let Some(fastmem) = &self.fastmem {
            fastmem
                .arena
                .clear()
                .unwrap_or_else(|err| panic!("fastmem release: {err}"));
        }
    }

    fn clear_pages(&self, tracking: &mut Tracking, va: VAddr, size: u64) {
        assert_eq!(
            (va | size) & PAGE_MASK,
            0,
            "unaligned unmap {va:#x} {size:#x}"
        );
        for page in 0..size / PAGE_SIZE {
            let addr = va + page * PAGE_SIZE;
            if let Some(old) = self.physical_address(addr) {
                tracking.reverse.remove(old, addr);
            }
            self.table.clear(addr);
        }
    }

    /// Changes the permission of the mapped pages of `[va, va + size)`. Write-tracked pages
    /// stay write-protected in the fastmem arena, and pages with a flush trap inaccessible.
    pub fn protect(&self, va: VAddr, size: u64, perm: MemoryPermission) {
        let _update = self.lock_for_rebuild(va, size, None);
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
        self.publish_exposed();
    }

    /// Locks `update` for a change to the fastmem views over `[va, va + size)`, which maps the
    /// physical pages from `mapped` there if given. Where the host maps a view accessible before
    /// it can protect it (Windows), the GPU first writes back the trapped pages the change would
    /// expose: those of views it rebuilds, and those it maps. Pages armed again meanwhile are
    /// written back on the next attempt; the handler disarms what it writes back, so only a GPU
    /// rewriting the same pages throughout outlasts the attempts, and the change then goes ahead.
    fn lock_for_rebuild(
        &self,
        va: VAddr,
        size: u64,
        mapped: Option<PAddr>,
    ) -> MutexGuard<'_, Tracking> {
        const ATTEMPTS: usize = 16;
        for _ in 1..ATTEMPTS {
            let tracking = self.update.lock();
            let mut exposed: Vec<PAddr> = Vec::new();
            if let Some(fastmem) = &self.fastmem {
                exposed.extend(
                    fastmem
                        .arena
                        .trapped_in_rebuilt_views(va as usize, size as usize)
                        .into_iter()
                        .filter_map(|page| self.physical_address(page as VAddr)),
                );
                if let Some(pa) = mapped.filter(|_| fastmem.arena.exposes_new_views()) {
                    exposed.extend(
                        (pa..pa + size)
                            .step_by(PAGE_SIZE as usize)
                            .filter(|&page| self.trapped.test(page)),
                    );
                }
            }
            if exposed.is_empty() {
                return tracking;
            }
            drop(tracking);
            for page in exposed {
                self.flush_physical(page);
            }
        }
        self.update.lock()
    }

    /// The table flags and fastmem cap a virtual page mapping physical page `pa` gets: a tracked
    /// page's view is write-protected, a trapped page's view inaccessible.
    fn page_state(&self, pa: PAddr) -> (usize, Option<HostProtection>) {
        match (self.tracked.test(pa), self.trapped.test(pa)) {
            (tracked, true) => (
                FLUSH_TRAP | if tracked { TRACKED } else { 0 },
                Some(HostProtection::None),
            ),
            (true, false) => (TRACKED, Some(HostProtection::Read)),
            (false, false) => (0, None),
        }
    }

    /// Publishes as written the tracked pages whose fastmem views had to be mapped again (and so
    /// were briefly writable). Called with `update` held, after the views are protected again.
    fn publish_exposed(&self) {
        let Some(fastmem) = &self.fastmem else {
            return;
        };
        for va in fastmem.arena.take_exposed() {
            if let Some(pa) = self.physical_address(va as u64) {
                self.dirty.mark(pa);
            }
        }
    }

    /// Brings the table flags and fastmem views of `pages` in line with the tracked and trapped
    /// flags of the physical pages they map, in runs: a tracked page's view is write-protected, a
    /// trapped page's view inaccessible. Called with `update` held.
    fn sync_pages(&self, pages: &mut [VAddr]) {
        type Run = (VAddr, u64, MemoryPermission, Option<HostProtection>);
        pages.sort_unstable();
        let mut run: Option<Run> = None;
        let flush = |run: Option<Run>| {
            if let (Some((va, size, perm, cap)), Some(fastmem)) = (run, &self.fastmem) {
                fastmem
                    .arena
                    .restrict(va as usize, size as usize, perm.host(), cap)
                    .unwrap_or_else(|err| panic!("fastmem tracking {va:#x}+{size:#x}: {err}"));
            }
        };
        for &va in pages.iter() {
            let Some(pa) = self.physical_address(va) else {
                continue;
            };
            let (flags, cap) = self.page_state(pa);
            let Some(perm) = self.table.set_flags(va, flags) else {
                continue;
            };
            match &mut run {
                Some((start, size, p, c)) if *start + *size == va && *p == perm && *c == cap => {
                    *size += PAGE_SIZE
                }
                _ => {
                    flush(run.take());
                    run = Some((va, PAGE_SIZE, perm, cap));
                }
            }
        }
        flush(run);
    }

    /// Adds a cache hold to every physical page of `runs` and tracks every virtual page that maps
    /// one, including pages a CPU write untracked since an earlier hold.
    pub(crate) fn cache_physical(&self, runs: &[(PAddr, u64)]) {
        let mut tracking = self.update.lock();
        let mut pages = Vec::new();
        for &(pa, size) in runs {
            for page in (pa..pa + size).step_by(PAGE_SIZE as usize) {
                let count = &mut tracking.cached[(page >> PAGE_BITS) as usize];
                *count = count
                    .checked_add(1)
                    .expect("too many cache holds on one page");
                self.tracked.set(page);
                tracking.reverse.for_each(page, |va| pages.push(va));
            }
        }
        self.sync_pages(&mut pages);
        drop(tracking);
        fence(Ordering::SeqCst);
    }

    /// Releases a hold [`AddressSpace::cache_physical`] added; pages nothing holds any more stop
    /// being tracked.
    pub(crate) fn uncache_physical(&self, runs: &[(PAddr, u64)]) {
        let mut tracking = self.update.lock();
        let mut pages = Vec::new();
        for &(pa, size) in runs {
            for page in (pa..pa + size).step_by(PAGE_SIZE as usize) {
                let count = &mut tracking.cached[(page >> PAGE_BITS) as usize];
                *count = count.checked_sub(1).expect("cache hold released twice");
                if *count == 0 {
                    self.tracked.clear(page);
                    tracking.reverse.for_each(page, |va| pages.push(va));
                }
            }
        }
        self.sync_pages(&mut pages);
    }

    /// Whether the physical page holding `pa` has a flush trap.
    #[inline]
    pub(crate) fn is_trapped(&self, pa: PAddr) -> bool {
        self.trapped.test(pa & !PAGE_MASK)
    }

    /// Sets a flush trap on every physical page of `runs`, in every virtual page that maps one.
    pub(crate) fn arm_physical(&self, runs: &[(PAddr, u64)]) {
        let tracking = self.update.lock();
        #[cfg(test)]
        self.trap_updates.fetch_add(1, Ordering::Relaxed);
        let mut pages = Vec::new();
        for &(pa, size) in runs {
            for page in (pa..pa + size).step_by(PAGE_SIZE as usize) {
                if !self.trapped.test(page) {
                    self.trapped.set(page);
                    tracking.reverse.for_each(page, |va| pages.push(va));
                }
            }
        }
        self.sync_pages(&mut pages);
    }

    /// Removes the flush traps of the physical pages of `runs`.
    pub(crate) fn disarm_physical(&self, runs: &[(PAddr, u64)]) {
        let tracking = self.update.lock();
        #[cfg(test)]
        self.trap_updates.fetch_add(1, Ordering::Relaxed);
        let mut pages = Vec::new();
        for &(pa, size) in runs {
            for page in (pa..pa + size).step_by(PAGE_SIZE as usize) {
                if self.trapped.test(page) {
                    self.trapped.clear(page);
                    tracking.reverse.for_each(page, |va| pages.push(va));
                }
            }
        }
        self.sync_pages(&mut pages);
    }

    /// Installs what a flush trap calls; without one a trap only removes itself.
    pub(crate) fn set_flush_sink(&self, sink: Option<FlushSink>) {
        *self.flush_sink.write() = sink;
    }

    /// A host access met the flush trap of the page holding `va`: has the GPU-modified data
    /// written back first. The page is accessed afterwards whatever the sink did; one it left
    /// trapped traps again on the next access.
    #[cold]
    #[inline(never)]
    fn flush_trap(&self, va: VAddr) {
        if let Some(pa) = self.physical_address(va) {
            self.flush_physical(pa & !PAGE_MASK);
        }
    }

    /// Has the GPU-modified data of physical page `page` written back, if it is trapped.
    #[cold]
    fn flush_physical(&self, page: PAddr) {
        if !self.trapped.test(page) {
            return;
        }
        let sink = self.flush_sink.read().clone();
        if !sink.is_some_and(|sink| sink(page)) {
            self.disarm_physical(&[(page, PAGE_SIZE)]);
        }
    }

    /// Has the GPU-modified data in `[va, va + size)` written back to guest memory, for host
    /// code about to read it through a pointer from [`AddressSpace::translate`] or
    /// [`AddressSpace::contiguous_pointer`]. The read and write methods here do it themselves.
    pub fn flush_gpu_writes(&self, va: VAddr, size: u64) {
        let trap = |at| self.flush_trap(at);
        let start = va & !PAGE_MASK;
        for page in (start..va.saturating_add(size)).step_by(PAGE_SIZE as usize) {
            let _ = self.table.access_with(page, MemoryPermission::NONE, trap);
        }
    }

    /// Reports a write host code made through `ptr`, a pointer into a page that
    /// [`AddressSpace::translate`] or [`AddressSpace::access`] returned, once its bytes landed.
    /// The write methods here report their own writes.
    #[inline]
    pub fn report_write(&self, ptr: *const u8) {
        fence(Ordering::SeqCst);
        let Some(pa) = self.physical.physical_address(ptr) else {
            return;
        };
        let pa = pa & !PAGE_MASK;
        if self.tracked.test(pa) {
            self.cpu_wrote(pa);
        }
    }

    /// The CPU wrote tracked physical page `pa`: untracks it on every alias, then publishes it.
    #[cold]
    fn cpu_wrote(&self, pa: PAddr) {
        let tracking = self.update.lock();
        if self.tracked.test(pa) {
            self.tracked.clear(pa);
            let mut pages = Vec::new();
            tracking.reverse.for_each(pa, |alias| pages.push(alias));
            self.sync_pages(&mut pages);
        }
        drop(tracking);
        self.dirty.mark(pa);
    }

    /// Writes `[va, va + size)` page by page with `copy`, reporting each page after its bytes.
    fn write_pages(
        &self,
        va: VAddr,
        size: u64,
        need: MemoryPermission,
        mut copy: impl FnMut(*mut u8, usize, usize),
    ) -> Result<(), MemoryFault> {
        let trap = |at| self.flush_trap(at);
        self.table
            .for_each_chunk(va, size, need, trap, |ptr, offset, len| {
                copy(ptr, offset, len);
                self.report_write(ptr);
            })
    }

    #[inline]
    fn write_value<T: bytemuck::Pod>(
        &self,
        va: VAddr,
        value: T,
        need: MemoryPermission,
    ) -> Result<(), MemoryFault> {
        let size = std::mem::size_of::<T>();
        if (va & PAGE_MASK) as usize + size <= PAGE_SIZE as usize {
            let ptr = self.table.access_with(va, need, |at| self.flush_trap(at))?;
            // SAFETY: the whole value is inside one mapped page.
            unsafe { std::ptr::write_unaligned(ptr as *mut T, value) };
            self.report_write(ptr);
            return Ok(());
        }
        let bytes = bytemuck::bytes_of(&value);
        self.write_pages(va, size as u64, need, |ptr, offset, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::copy_nonoverlapping(bytes[offset..].as_ptr(), ptr, len) };
        })
    }

    /// Whether a CPU write to a tracked page is waiting for [`AddressSpace::take_cpu_writes`].
    pub(crate) fn has_cpu_writes(&self) -> bool {
        self.dirty.pending()
    }

    /// Calls `f` with each run of physical pages the CPU wrote while they were tracked, as
    /// (address, size), and forgets them.
    pub(crate) fn take_cpu_writes(&self, f: impl FnMut(PAddr, u64)) {
        self.dirty.take(f);
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

    /// Host pointer for an access to `va` that needs `need` (atomics, exclusives): a permission
    /// fault when the page is mapped without it. GPU-modified data of the page is written back
    /// first.
    #[inline]
    pub fn access(&self, va: VAddr, need: MemoryPermission) -> Result<*mut u8, MemoryFault> {
        self.table.access_with(va, need, |at| self.flush_trap(at))
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

    /// Reads a value, whatever the page permissions (the kernel's and the services' view).
    #[inline]
    pub fn read<T: bytemuck::Pod>(&self, va: VAddr) -> Result<T, MemoryFault> {
        self.table
            .read(va, MemoryPermission::NONE, |at| self.flush_trap(at))
    }

    /// Writes a value, whatever the page permissions (loading code, HLE writes).
    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, va: VAddr, value: T) -> Result<(), MemoryFault> {
        self.write_value(va, value, MemoryPermission::NONE)
    }

    #[inline]
    pub fn read_bytes(&self, va: VAddr, out: &mut [u8]) -> Result<(), MemoryFault> {
        self.table
            .read_bytes(va, out, MemoryPermission::NONE, |at| self.flush_trap(at))
    }

    pub fn write_bytes(&self, va: VAddr, data: &[u8]) -> Result<(), MemoryFault> {
        self.write_pages(
            va,
            data.len() as u64,
            MemoryPermission::NONE,
            |ptr, offset, len| {
                // SAFETY: ptr is valid for len bytes inside one mapped page.
                unsafe { std::ptr::copy_nonoverlapping(data[offset..].as_ptr(), ptr, len) };
            },
        )
    }

    pub fn fill(&self, va: VAddr, size: u64, byte: u8) -> Result<(), MemoryFault> {
        self.write_pages(va, size, MemoryPermission::NONE, |ptr, _, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::write_bytes(ptr, byte, len) };
        })
    }

    /// Runs `write` on the host pointer for `va`, which needs `need`, then reports the write
    /// to write tracking. For atomics, exclusive stores and other writes through a host pointer;
    /// `write` must stay inside the page of `va`.
    pub fn write_with<R>(
        &self,
        va: VAddr,
        need: MemoryPermission,
        write: impl FnOnce(*mut u8) -> R,
    ) -> Result<R, MemoryFault> {
        let ptr = self.table.access_with(va, need, |at| self.flush_trap(at))?;
        let result = write(ptr);
        self.report_write(ptr);
        Ok(result)
    }

    /// A guest load: the page must be readable, else a permission fault.
    #[inline]
    pub fn guest_read<T: bytemuck::Pod>(&self, va: VAddr) -> Result<T, MemoryFault> {
        self.table
            .read(va, MemoryPermission::READ, |at| self.flush_trap(at))
    }

    /// A guest store: a page without write permission is a permission fault and is left
    /// unchanged. The fastmem arena protects such pages on the host too, so the JIT's direct
    /// stores fault into its slow path, which stores through here.
    #[inline]
    pub fn guest_write<T: bytemuck::Pod>(&self, va: VAddr, value: T) -> Result<(), MemoryFault> {
        self.write_value(va, value, MemoryPermission::WRITE)
    }

    /// A guest store of `size` copies of `byte` (`DC ZVA`), checked like
    /// [`AddressSpace::guest_write`]; pages before a faulting one are written.
    pub fn guest_fill(&self, va: VAddr, size: u64, byte: u8) -> Result<(), MemoryFault> {
        self.write_pages(va, size, MemoryPermission::WRITE, |ptr, _, len| {
            // SAFETY: ptr is valid for len bytes inside one mapped page.
            unsafe { std::ptr::write_bytes(ptr, byte, len) };
        })
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

    /// Cost of guest loads and stores through the page table on pages without traps. Run with
    /// `cargo test --release -p lichii-memory guest_access_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn guest_access_cost() {
        let mem = space();
        mem.map(0x8000_0000, 0, 0x10_0000, MemoryPermission::READ_WRITE);
        let rounds = 50_000_000u64;
        let start = std::time::Instant::now();
        let mut sum = 0u64;
        for i in 0..rounds {
            let va = 0x8000_0000 + (i.wrapping_mul(0x9e37_79b9) & 0xf_fff8);
            sum = sum.wrapping_add(u64::from(mem.guest_read::<u32>(va).unwrap()));
        }
        let read = start.elapsed();
        let start = std::time::Instant::now();
        for i in 0..rounds {
            let va = 0x8000_0000 + (i.wrapping_mul(0x9e37_79b9) & 0xf_fff8);
            mem.guest_write(va, i as u32).unwrap();
        }
        let write = start.elapsed();
        println!(
            "guest_read {:.2} ns, guest_write {:.2} ns ({sum})",
            read.as_nanos() as f64 / rounds as f64,
            write.as_nanos() as f64 / rounds as f64
        );
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
