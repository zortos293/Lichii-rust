//! Host virtual memory: the guest DRAM backing and reserved regions that map views of it.
//!
//! The backing is shareable memory of the full DRAM size: a memfd on Linux, one pagefile section
//! per 2 MiB chunk on Windows. Nothing is committed up front and freed memory goes back: Linux
//! allocates pages on first touch and punches out the pages the kernel frees, and Windows creates
//! a chunk's section when the kernel first hands out its pages and closes it once they are all
//! free. Arenas are reserved, inaccessible host ranges into which pages of the backing are mapped
//! as views, so one physical page can appear at several host addresses.

use std::collections::BTreeMap;
use std::io;
use std::ops::Range;

use lichii_common::PAGE_SIZE;
use parking_lot::Mutex;

const PAGE: usize = PAGE_SIZE as usize;

/// Host page protection, ordered from least to most access.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum HostProtection {
    None,
    Read,
    ReadWrite,
}

/// Pages whose views get at most a cap, by offset: write-tracked pages are capped at read, pages
/// with a flush trap at no access.
pub(crate) type Restricted = BTreeMap<usize, HostProtection>;

/// Runs of restricted pages inside `[start, end)` with the same cap, as (offset, length, cap).
fn restricted_runs(
    restricted: &Restricted,
    start: usize,
    end: usize,
) -> Vec<(usize, usize, HostProtection)> {
    let mut runs: Vec<(usize, usize, HostProtection)> = Vec::new();
    for (&page, &cap) in restricted.range(start..end) {
        match runs.last_mut() {
            Some((first, len, c)) if *first + *len == page && *c == cap => *len += PAGE,
            _ => runs.push((page, PAGE, cap)),
        }
    }
    runs
}

/// `[start, end)` split into runs that get `prot` and runs of restricted pages that get it
/// capped, so a protection change never raises a restricted page past its cap, even briefly.
fn protection_runs(
    restricted: &Restricted,
    start: usize,
    end: usize,
    prot: HostProtection,
) -> Vec<(usize, usize, HostProtection)> {
    let mut runs: Vec<(usize, usize, HostProtection)> = Vec::new();
    let mut push = |at: usize, len: usize, prot: HostProtection| match runs.last_mut() {
        Some((first, l, p)) if *first + *l == at && *p == prot => *l += len,
        _ => runs.push((at, len, prot)),
    };
    let mut at = start;
    for (first, len, cap) in restricted_runs(restricted, start, end) {
        if first > at {
            push(at, first - at, prot);
        }
        push(first, len, prot.min(cap));
        at = first + len;
    }
    if at < end {
        push(at, end - at, prot);
    }
    runs
}

/// The DRAM backing object and its main read-write view.
pub(crate) struct Backing {
    ptr: *mut u8,
    size: usize,
    sys: sys::Backing,
}

impl Backing {
    pub(crate) fn new(size: usize) -> io::Result<Self> {
        let (sys, ptr) = sys::Backing::create(size)?;
        Ok(Self { ptr, size, sys })
    }

    pub(crate) fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Makes `[offset, offset + len)` accessible. Idempotent.
    pub(crate) fn commit(&self, offset: usize, len: usize) -> io::Result<()> {
        self.sys.commit(self.ptr, offset, len)
    }

    /// Zeroes `[offset, offset + len)` and returns its host pages to the OS where possible.
    pub(crate) fn discard(&self, offset: usize, len: usize) {
        self.sys.discard(self.ptr, offset, len)
    }

    /// Gives the host memory behind `[offset, offset + len)`, which holds nothing anyone
    /// needs, back to the OS, in whole [`Self::release_granule`]s. Returns the part given back,
    /// which reads as zero once committed again.
    pub(crate) fn release(&self, offset: usize, len: usize) -> Range<usize> {
        self.sys.release(self.ptr, offset, len)
    }

    /// The unit [`Self::release`] gives memory back in.
    pub(crate) fn release_granule(&self) -> usize {
        self.sys.release_granule()
    }

    /// Host memory backing guest DRAM now, in bytes: committed on Windows, allocated in the
    /// shared memory object on Unix.
    pub(crate) fn host_bytes(&self) -> u64 {
        self.sys.host_bytes()
    }

    pub(crate) fn supports_views(&self) -> bool {
        self.sys.supports_views()
    }

    /// The data [`charge_touched`] takes to commit this backing's memory on first touch, or
    /// None when it needs no touch hook. Valid while the backing lives.
    pub(crate) fn touch_target(&self) -> Option<usize> {
        self.sys.touch_target()
    }

    /// The part of the backing the host has touched, through its own view or through views in
    /// `arenas`, as (bytes of touched pages, bytes of the [`Self::release_granule`]s holding
    /// one): resident pages on Windows, allocated pages on Unix.
    pub(crate) fn touched(&self, arenas: &[&Arena]) -> (u64, u64) {
        let arenas: Vec<(&sys::Arena, *mut u8)> = arenas.iter().map(|a| (&a.sys, a.base)).collect();
        self.sys.touched(self.ptr, &arenas)
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        self.sys.destroy(self.ptr, self.size);
    }
}

/// Commits the memory behind backing `offset` for a [`Backing::touch_target`] `target` when the
/// access at `host_addr`, a view of it, faulted because it was not committed yet. Called from a
/// fault hook; returns whether the access can be retried.
pub(crate) fn charge_touched(target: usize, offset: usize, host_addr: usize) -> bool {
    sys::charge_touched(target, offset, host_addr)
}

/// A reserved host range that holds views of a [`Backing`].
pub(crate) struct Arena {
    base: *mut u8,
    size: usize,
    sys: sys::Arena,
    /// Restricted pages: their views stay capped whatever `map` and `protect` give them, until
    /// `restrict` lifts the cap or they are mapped or unmapped again.
    restricted: Mutex<Restricted>,
}

impl Arena {
    pub(crate) fn reserve(size: usize) -> io::Result<Self> {
        let (sys, base) = sys::Arena::reserve(size)?;
        Ok(Self {
            base,
            size,
            sys,
            restricted: Mutex::default(),
        })
    }

    fn forget_restricted(restricted: &mut Restricted, offset: usize, len: usize) {
        let pages: Vec<usize> = restricted
            .range(offset..offset + len)
            .map(|(&page, _)| page)
            .collect();
        for page in pages {
            restricted.remove(&page);
        }
    }

    pub(crate) fn base(&self) -> *mut u8 {
        self.base
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Maps `len` bytes of `backing` at `backing_offset` to `offset` in the arena, replacing any
    /// view there. `caps` restricts pages of the new view, by offset, from the start.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn map(
        &self,
        backing: &Backing,
        offset: usize,
        backing_offset: usize,
        len: usize,
        prot: HostProtection,
        caps: &[(usize, HostProtection)],
    ) -> io::Result<()> {
        assert!(offset + len <= self.size && backing_offset + len <= backing.size);
        let mut restricted = self.restricted.lock();
        Self::forget_restricted(&mut restricted, offset, len);
        restricted.extend(caps.iter().copied());
        self.sys.map(
            self.base,
            &backing.sys,
            offset,
            backing_offset,
            len,
            prot,
            &restricted,
        )
    }

    /// Removes views from `[offset, offset + len)`; accesses there fault again.
    pub(crate) fn unmap(&self, backing: &Backing, offset: usize, len: usize) -> io::Result<()> {
        assert!(offset + len <= self.size);
        let mut restricted = self.restricted.lock();
        Self::forget_restricted(&mut restricted, offset, len);
        self.sys
            .unmap(self.base, &backing.sys, offset, len, &restricted)
    }

    /// Removes every view; the whole arena faults again.
    pub(crate) fn clear(&self) -> io::Result<()> {
        self.restricted.lock().clear();
        self.sys.clear(self.base, self.size)
    }

    /// Changes the protection of mapped pages in `[offset, offset + len)`.
    pub(crate) fn protect(
        &self,
        backing: &Backing,
        offset: usize,
        len: usize,
        prot: HostProtection,
    ) -> io::Result<()> {
        assert!(offset + len <= self.size);
        let restricted = self.restricted.lock();
        self.sys
            .protect(self.base, &backing.sys, offset, len, prot, &restricted)
    }

    /// Caps the views of the mapped pages of `[offset, offset + len)`, whose protection is
    /// `prot`, at `cap`, or gives them `prot` back with `None`.
    pub(crate) fn restrict(
        &self,
        backing: &Backing,
        offset: usize,
        len: usize,
        prot: HostProtection,
        cap: Option<HostProtection>,
    ) -> io::Result<()> {
        assert!(offset + len <= self.size);
        let mut restricted = self.restricted.lock();
        for page in (offset..offset + len).step_by(PAGE) {
            match cap {
                Some(cap) => restricted.insert(page, cap),
                None => restricted.remove(&page),
            };
        }
        self.sys
            .restrict(self.base, &backing.sys, offset, len, prot, cap)
    }

    /// Whether a new view is accessible for a moment before its caps apply.
    pub(crate) fn exposes_new_views(&self) -> bool {
        sys::EXPOSES_NEW_VIEWS
    }

    /// Offsets of the trapped pages (capped at no access) that a change to the views over
    /// `[offset, offset + len)` would map accessible for a moment: none where views can be
    /// protected as they are mapped.
    pub(crate) fn trapped_in_rebuilt_views(&self, offset: usize, len: usize) -> Vec<usize> {
        let restricted = self.restricted.lock();
        let mut trapped = Vec::new();
        for (start, end) in self.sys.rebuilt_views(offset, len) {
            trapped.extend(
                restricted
                    .range(start..end)
                    .filter(|(_, cap)| **cap == HostProtection::None)
                    .map(|(&page, _)| page),
            );
        }
        trapped
    }

    /// Offsets of restricted pages whose views had to be mapped again since the last call.
    /// A view is mapped read-write before it is protected, so a store may have landed there
    /// unseen; the caller publishes those pages as written.
    pub(crate) fn take_exposed(&self) -> Vec<usize> {
        self.sys.take_exposed()
    }

    #[cfg(all(test, windows))]
    pub(crate) fn sys(&self) -> &sys::Arena {
        &self.sys
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        self.sys.release(self.base, self.size);
    }
}

// SAFETY: the raw pointers name mappings owned by the struct for its whole life; the OS calls are
// thread-safe and the Windows view bookkeeping sits behind a mutex.
unsafe impl Send for Backing {}
unsafe impl Sync for Backing {}
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

#[cfg(unix)]
mod sys {
    /// Views are mapped with their final protection.
    pub const EXPOSES_NEW_VIEWS: bool = false;

    use std::io;
    use std::ops::Range;

    use super::{HostProtection, PAGE, Restricted, protection_runs};

    fn check(ret: libc::c_int) -> io::Result<()> {
        if ret == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    fn prot_flags(prot: HostProtection) -> libc::c_int {
        match prot {
            HostProtection::None => libc::PROT_NONE,
            HostProtection::Read => libc::PROT_READ,
            HostProtection::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
        }
    }

    pub struct Backing {
        fd: libc::c_int,
    }

    /// Frees and zeroes `[offset, offset + len)` of a shared memory object in every view.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn punch_hole(fd: libc::c_int, offset: usize, len: usize) -> bool {
        // SAFETY: plain system call on our own descriptor.
        unsafe {
            libc::fallocate(
                fd,
                libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                offset as libc::off_t,
                len as libc::off_t,
            ) == 0
        }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn punch_hole(_fd: libc::c_int, _offset: usize, _len: usize) -> bool {
        false
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn create_shared_fd() -> libc::c_int {
        // SAFETY: plain system call with a NUL-terminated name.
        unsafe { libc::memfd_create(c"lichii-dram".as_ptr(), libc::MFD_CLOEXEC) }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn create_shared_fd() -> libc::c_int {
        let name = format!("/lichii-dram-{}\0", std::process::id());
        // SAFETY: name is NUL-terminated; the object is unlinked right away so it lives only as
        // long as the descriptor.
        unsafe {
            let fd = libc::shm_open(
                name.as_ptr().cast(),
                libc::O_RDWR | libc::O_CREAT | libc::O_EXCL,
                0o600,
            );
            if fd >= 0 {
                libc::shm_unlink(name.as_ptr().cast());
            }
            fd
        }
    }

    impl Backing {
        pub fn create(size: usize) -> io::Result<(Self, *mut u8)> {
            let fd = create_shared_fd();
            // SAFETY: fd is checked before use and closed on every failure path.
            unsafe {
                if fd >= 0 && libc::ftruncate(fd, size as libc::off_t) == 0 {
                    let ptr = libc::mmap(
                        std::ptr::null_mut(),
                        size,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_SHARED | libc::MAP_NORESERVE,
                        fd,
                        0,
                    );
                    if ptr != libc::MAP_FAILED {
                        return Ok((Self { fd }, ptr.cast()));
                    }
                }
                if fd >= 0 {
                    libc::close(fd);
                }
                let ptr = libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                );
                if ptr == libc::MAP_FAILED {
                    return Err(io::Error::last_os_error());
                }
                Ok((Self { fd: -1 }, ptr.cast()))
            }
        }

        pub fn supports_views(&self) -> bool {
            self.fd >= 0
        }

        pub fn commit(&self, _ptr: *mut u8, _offset: usize, _len: usize) -> io::Result<()> {
            Ok(())
        }

        pub fn release_granule(&self) -> usize {
            PAGE
        }

        pub fn host_bytes(&self) -> u64 {
            if self.fd < 0 {
                return 0;
            }
            // SAFETY: fstat of our own descriptor into a zeroed buffer.
            unsafe {
                let mut st: libc::stat = std::mem::zeroed();
                if libc::fstat(self.fd, &mut st) != 0 {
                    return 0;
                }
                st.st_blocks as u64 * 512
            }
        }

        pub fn touch_target(&self) -> Option<usize> {
            None
        }

        pub fn touched(&self, _ptr: *mut u8, _arenas: &[(&Arena, *mut u8)]) -> (u64, u64) {
            let bytes = self.host_bytes();
            (bytes, bytes)
        }

        pub fn release(&self, ptr: *mut u8, offset: usize, len: usize) -> Range<usize> {
            let released = if self.fd >= 0 {
                punch_hole(self.fd, offset, len)
            } else {
                // SAFETY: the range is inside our private anonymous mapping, whose pages read as
                // zero again after MADV_DONTNEED.
                unsafe { libc::madvise(ptr.add(offset).cast(), len, libc::MADV_DONTNEED) == 0 }
            };
            offset..if released { offset + len } else { offset }
        }

        pub fn discard(&self, ptr: *mut u8, offset: usize, len: usize) {
            if self.fd >= 0 && punch_hole(self.fd, offset, len) {
                return;
            }
            // SAFETY: the range is inside the backing view.
            unsafe { std::ptr::write_bytes(ptr.add(offset), 0, len) };
        }

        pub fn destroy(&mut self, ptr: *mut u8, size: usize) {
            // SAFETY: ptr/size is the mapping created in create; fd is ours.
            unsafe {
                libc::munmap(ptr.cast(), size);
                if self.fd >= 0 {
                    libc::close(self.fd);
                }
            }
        }
    }

    pub fn charge_touched(_target: usize, _offset: usize, _host_addr: usize) -> bool {
        false
    }

    pub struct Arena;

    fn reserve_at(addr: *mut u8, len: usize, fixed: bool) -> *mut libc::c_void {
        let flags = libc::MAP_PRIVATE
            | libc::MAP_ANONYMOUS
            | libc::MAP_NORESERVE
            | if fixed { libc::MAP_FIXED } else { 0 };
        // SAFETY: anonymous PROT_NONE mapping; with MAP_FIXED the caller owns the range.
        unsafe { libc::mmap(addr.cast(), len, libc::PROT_NONE, flags, -1, 0) }
    }

    impl Arena {
        pub fn reserve(size: usize) -> io::Result<(Self, *mut u8)> {
            let ptr = reserve_at(std::ptr::null_mut(), size, false);
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok((Self, ptr.cast()))
        }

        #[allow(clippy::too_many_arguments)]
        pub fn map(
            &self,
            base: *mut u8,
            backing: &Backing,
            offset: usize,
            backing_offset: usize,
            len: usize,
            prot: HostProtection,
            restricted: &Restricted,
        ) -> io::Result<()> {
            for (start, run, prot) in protection_runs(restricted, offset, offset + len, prot) {
                // SAFETY: the run lies inside the arena reservation this struct owns.
                let ptr = unsafe {
                    libc::mmap(
                        base.add(start).cast(),
                        run,
                        prot_flags(prot),
                        libc::MAP_SHARED | libc::MAP_FIXED,
                        backing.fd,
                        (backing_offset + (start - offset)) as libc::off_t,
                    )
                };
                if ptr == libc::MAP_FAILED {
                    return Err(io::Error::last_os_error());
                }
            }
            Ok(())
        }

        pub fn rebuilt_views(&self, _offset: usize, _len: usize) -> Vec<(usize, usize)> {
            Vec::new()
        }

        pub fn unmap(
            &self,
            base: *mut u8,
            _backing: &Backing,
            offset: usize,
            len: usize,
            _restricted: &Restricted,
        ) -> io::Result<()> {
            // SAFETY: the range lies inside the arena reservation.
            let ptr = reserve_at(unsafe { base.add(offset) }, len, true);
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub fn clear(&self, base: *mut u8, size: usize) -> io::Result<()> {
            if reserve_at(base, size, true) == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub fn protect(
            &self,
            base: *mut u8,
            _backing: &Backing,
            offset: usize,
            len: usize,
            prot: HostProtection,
            restricted: &Restricted,
        ) -> io::Result<()> {
            for (start, run, prot) in protection_runs(restricted, offset, offset + len, prot) {
                // SAFETY: the run lies inside the arena reservation.
                check(unsafe { libc::mprotect(base.add(start).cast(), run, prot_flags(prot)) })?;
            }
            Ok(())
        }

        pub fn take_exposed(&self) -> Vec<usize> {
            Vec::new()
        }

        pub fn restrict(
            &self,
            base: *mut u8,
            _backing: &Backing,
            offset: usize,
            len: usize,
            prot: HostProtection,
            cap: Option<HostProtection>,
        ) -> io::Result<()> {
            let prot = cap.map_or(prot, |cap| prot.min(cap));
            // SAFETY: the range lies inside the arena reservation.
            check(unsafe { libc::mprotect(base.add(offset).cast(), len, prot_flags(prot)) })
        }

        pub fn release(&mut self, base: *mut u8, size: usize) {
            // SAFETY: base/size is the reservation made in reserve.
            unsafe { libc::munmap(base.cast(), size) };
        }
    }
}

#[cfg(windows)]
mod sys {
    /// Views are mapped read-write and protected afterwards.
    pub const EXPOSES_NEW_VIEWS: bool = true;

    use std::collections::BTreeMap;
    use std::io;
    use std::ops::Range;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use parking_lot::Mutex;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, MEM_COMMIT, MEM_MAPPED, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
        MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS,
        PAGE_READONLY, PAGE_READWRITE, SEC_COMMIT, SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc,
        VirtualAlloc2, VirtualFree, VirtualProtect, VirtualQuery,
    };
    use windows_sys::Win32::System::ProcessStatus::{
        K32QueryWorkingSetEx, PSAPI_WORKING_SET_EX_INFORMATION,
    };
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, Sleep};

    use super::{HostProtection, PAGE, Restricted, protection_runs, restricted_runs};
    use crate::fault::Registration;

    const MEM_COALESCE_PLACEHOLDERS: u32 = 0x1;
    const MEM_SPLIT_PLACEHOLDER: u32 = 0x2;
    const CHUNK_BITS: u32 = 21;
    const CHUNK: usize = 1 << CHUNK_BITS;

    fn page_flags(prot: HostProtection) -> PAGE_PROTECTION_FLAGS {
        match prot {
            HostProtection::None => PAGE_NOACCESS,
            HostProtection::Read => PAGE_READONLY,
            HostProtection::ReadWrite => PAGE_READWRITE,
        }
    }

    fn check(ok: i32) -> io::Result<()> {
        if ok != 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    enum Slot {
        /// Part of the reserved range, never committed.
        Reserved,
        Committed(HANDLE),
        /// Freed whole: its section is closed and the scratch section shows in its place.
        Scratch,
    }

    /// Which chunks have a section (created) and which have their pages committed (charged).
    /// Boxed so a fault hook can keep its address; it locks and allocates nothing.
    struct Chunks {
        ptr: usize,
        size: usize,
        created: Box<[AtomicU64]>,
        charged: Box<[AtomicU64]>,
        charged_chunks: AtomicUsize,
    }

    fn bit(map: &[AtomicU64], chunk: usize) -> bool {
        map[chunk / 64].load(Ordering::Acquire) & (1 << (chunk % 64)) != 0
    }

    impl Chunks {
        fn len(&self, chunk: usize) -> usize {
            CHUNK.min(self.size - chunk * CHUNK)
        }

        fn mark_charged(&self, chunk: usize) {
            let old = self.charged[chunk / 64].fetch_or(1 << (chunk % 64), Ordering::Release);
            if old & (1 << (chunk % 64)) == 0 {
                self.charged_chunks.fetch_add(1, Ordering::Relaxed);
            }
        }

        /// Commits a created chunk's pages through the backing view, which commits them in
        /// every view of its section. Retries for a while when the host has no commit left, since
        /// the pagefile may still grow. Returns false for a chunk without a section.
        fn charge(&self, chunk: usize) -> bool {
            if bit(&self.charged, chunk) {
                return true;
            }
            if !bit(&self.created, chunk) {
                return false;
            }
            for _ in 0..CHARGE_RETRIES {
                // SAFETY: the range is the chunk's view of its reserved section.
                let at = unsafe {
                    VirtualAlloc(
                        (self.ptr + chunk * CHUNK) as *const _,
                        self.len(chunk),
                        MEM_COMMIT,
                        PAGE_READWRITE,
                    )
                };
                if !at.is_null() {
                    self.mark_charged(chunk);
                    return true;
                }
                // SAFETY: plain sleep.
                unsafe { Sleep(CHARGE_RETRY_MS) };
            }
            false
        }
    }

    /// The touch hook of the backing view: commits the chunk a host access faulted in.
    fn charge_backing(data: usize, offset: u64, host_addr: usize) -> bool {
        charge_touched(data, offset as usize, host_addr)
    }

    /// Commits the chunk holding backing `offset` for the [`Backing::touch_target`] `target`
    /// when the view page at `host_addr` is not committed. A committed read-write page means
    /// another thread committed it after this access faulted, so the access is retried too; any
    /// other committed page faulted for its protection, which the fault handler owns.
    pub fn charge_touched(target: usize, offset: usize, host_addr: usize) -> bool {
        // SAFETY: the target is a live backing's boxed chunk state (see touch_target).
        let chunks = unsafe { &*(target as *const Chunks) };
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: queries our own address space into a local struct.
        let queried = unsafe {
            VirtualQuery(
                host_addr as *const _,
                &mut info,
                size_of::<MEMORY_BASIC_INFORMATION>(),
            )
        };
        if queried == 0 || info.Type != MEM_MAPPED || offset >= chunks.size {
            return false;
        }
        match info.State {
            MEM_RESERVE => chunks.charge(offset >> CHUNK_BITS),
            MEM_COMMIT => info.Protect == PAGE_READWRITE,
            _ => false,
        }
    }

    /// Whether the host has `len` bytes of commit to spare beyond the headroom.
    fn commit_available(len: usize) -> bool {
        let mut status = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: fills our own struct, whose length field is set.
        if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
            return true;
        }
        status.ullAvailPageFile >= len as u64 + COMMIT_HEADROOM
    }

    /// DRAM as one pagefile-backed section per chunk, each mapped into its slot of one reserved
    /// range when the kernel first hands out its pages. Sections are only reserved: a chunk's
    /// pages are committed when the host or the guest first touches it, through a fault hook on
    /// the backing view and on every fastmem arena, so the commit a title costs follows what it
    /// uses rather than what it was handed. A chunk the kernel frees whole closes its section,
    /// which gives its commit and RAM back, and maps the shared scratch section in its place, so
    /// a late host access to freed memory lands somewhere harmless instead of faulting.
    pub struct Backing {
        size: usize,
        chunks: Box<Chunks>,
        slots: Mutex<Box<[Slot]>>,
        scratch: HANDLE,
        /// The backing view's touch hook. None when no fault slot was free: sections are then
        /// committed whole as they are created.
        touch: Option<Registration>,
    }

    /// Commit a new chunk must leave free on the host, so a title runs out of memory as a guest
    /// error instead of failing to commit a page it already holds.
    const COMMIT_HEADROOM: u64 = 512 << 20;
    /// How long a first touch retries a commit the host refused, while the pagefile grows.
    const CHARGE_RETRIES: u32 = 50;
    const CHARGE_RETRY_MS: u32 = 20;

    /// A pagefile-backed section of `len` bytes: committed whole, or only reserved, so its pages
    /// are committed through a view as they are first touched.
    fn create_section(len: usize, reserve: bool) -> io::Result<HANDLE> {
        let len = len as u64;
        let commit = if reserve { SEC_RESERVE } else { SEC_COMMIT };
        // SAFETY: creates an unnamed pagefile-backed section; the caller owns the handle.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | commit,
                (len >> 32) as u32,
                len as u32,
                std::ptr::null(),
            )
        };
        if section.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(section)
    }

    /// Maps `section` over the free placeholder at `at`.
    ///
    /// # Safety
    /// `[at, at + len)` must be exactly one free placeholder of ours.
    unsafe fn map_over(section: HANDLE, at: *mut u8, len: usize) -> io::Result<()> {
        // SAFETY: the caller guarantees [at, at + len) is one free placeholder.
        let view = unsafe {
            MapViewOfFile3(
                section,
                GetCurrentProcess(),
                at.cast(),
                0,
                len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        if view.Value.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }

    /// Unmaps the view at `at`, leaving its placeholder.
    ///
    /// # Safety
    /// `at` must be the base of a view we mapped over a placeholder.
    unsafe fn unmap_view(at: *mut u8) -> io::Result<()> {
        // SAFETY: the caller guarantees at is one of our views.
        check(unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS { Value: at.cast() },
                MEM_PRESERVE_PLACEHOLDER,
            )
        })
    }

    impl Backing {
        pub fn create(size: usize) -> io::Result<(Self, *mut u8)> {
            let scratch = create_section(CHUNK, false)?;
            // SAFETY: reserves one placeholder covering all of DRAM; chunks are split out of it
            // as they are committed.
            let base = unsafe {
                VirtualAlloc2(
                    GetCurrentProcess(),
                    std::ptr::null(),
                    size,
                    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                    PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if base.is_null() {
                let err = io::Error::last_os_error();
                // SAFETY: scratch is our own handle.
                unsafe { CloseHandle(scratch) };
                return Err(err);
            }
            let count = size.div_ceil(CHUNK);
            let bitmap = || (0..count.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
            let chunks = Box::new(Chunks {
                ptr: base as usize,
                size,
                created: bitmap(),
                charged: bitmap(),
                charged_chunks: AtomicUsize::new(0),
            });
            let touch = Registration::new(base as usize, size);
            if let Some(touch) = &touch {
                touch.set_touch(charge_backing, &*chunks as *const Chunks as usize);
            }
            Ok((
                Self {
                    size,
                    chunks,
                    slots: Mutex::new((0..count).map(|_| Slot::Reserved).collect()),
                    scratch,
                    touch,
                },
                base.cast(),
            ))
        }

        /// The data [`charge_touched`] takes to commit this backing's chunks on first touch, or
        /// None when they are committed as they are created. Valid while the backing lives.
        pub fn touch_target(&self) -> Option<usize> {
            self.touch
                .as_ref()
                .map(|_| &*self.chunks as *const Chunks as usize)
        }

        /// Commits the chunks under `[offset, offset + len)` that have a section, before their
        /// views are protected: protection applies only to committed pages.
        pub fn charge_range(&self, offset: usize, len: usize) -> io::Result<()> {
            if len == 0 {
                return Ok(());
            }
            for chunk in offset >> CHUNK_BITS..=(offset + len - 1) >> CHUNK_BITS {
                if bit(&self.chunks.created, chunk) && !self.chunks.charge(chunk) {
                    return Err(io::Error::from(io::ErrorKind::OutOfMemory));
                }
            }
            Ok(())
        }

        pub fn supports_views(&self) -> bool {
            true
        }

        pub fn host_bytes(&self) -> u64 {
            (self.chunks.charged_chunks.load(Ordering::Relaxed) * CHUNK) as u64
        }

        pub fn release_granule(&self) -> usize {
            CHUNK
        }

        fn is_committed(&self, chunk: usize) -> bool {
            bit(&self.chunks.created, chunk)
        }

        fn chunk_len(&self, chunk: usize) -> usize {
            CHUNK.min(self.size - chunk * CHUNK)
        }

        /// The section whose view belongs at `chunk`: its own once committed, else scratch.
        fn section(&self, chunk: usize) -> HANDLE {
            match self.slots.lock()[chunk] {
                Slot::Committed(section) => section,
                Slot::Reserved | Slot::Scratch => self.scratch,
            }
        }

        pub fn commit(&self, ptr: *mut u8, offset: usize, len: usize) -> io::Result<()> {
            if len == 0 {
                return Ok(());
            }
            let first = offset >> CHUNK_BITS;
            let last = (offset + len - 1) >> CHUNK_BITS;
            if (first..=last).all(|c| self.is_committed(c)) {
                return Ok(());
            }
            let mut slots = self.slots.lock();
            for chunk in first..=last {
                if self.is_committed(chunk) {
                    continue;
                }
                let (len, at) = (self.chunk_len(chunk), ptr.wrapping_add(chunk * CHUNK));
                let lazy = self.touch.is_some();
                if lazy && !commit_available(len) {
                    return Err(io::Error::from(io::ErrorKind::OutOfMemory));
                }
                let section = create_section(len, lazy)?;
                // SAFETY: the slot holds the scratch view, which unmapping turns back into the
                // chunk's own placeholder, or lies in the reserved range, from which the split
                // carves it out (and fails harmlessly once it is split). Either way [at, at + len)
                // is then one free placeholder.
                let mapped = unsafe {
                    match slots[chunk] {
                        Slot::Scratch => unmap_view(at),
                        _ => {
                            VirtualFree(at.cast(), len, MEM_RELEASE | MEM_SPLIT_PLACEHOLDER);
                            Ok(())
                        }
                    }
                    .and_then(|()| map_over(section, at, len))
                };
                if let Err(err) = mapped {
                    slots[chunk] = Slot::Reserved;
                    // SAFETY: the section is ours and mapped nowhere.
                    unsafe { CloseHandle(section) };
                    return Err(err);
                }
                slots[chunk] = Slot::Committed(section);
                self.chunks.created[chunk / 64].fetch_or(1 << (chunk % 64), Ordering::Release);
                if !lazy {
                    self.chunks.mark_charged(chunk);
                }
            }
            Ok(())
        }

        /// Marks the resident pages of the views in `views` (host address, backing offset,
        /// length) in `seen`, one bit per backing page.
        fn mark_resident(views: impl Iterator<Item = (usize, usize, usize)>, seen: &mut [u64]) {
            const BATCH: usize = 512;
            let mut entries = [PSAPI_WORKING_SET_EX_INFORMATION::default(); BATCH];
            for (host, backing_offset, len) in views {
                let mut at = 0;
                while at < len {
                    let n = ((len - at) / PAGE).min(BATCH);
                    for (i, entry) in entries[..n].iter_mut().enumerate() {
                        entry.VirtualAddress = (host + at + i * PAGE) as *mut _;
                    }
                    // SAFETY: queries our own process into a buffer of n entries.
                    let ok = unsafe {
                        K32QueryWorkingSetEx(
                            GetCurrentProcess(),
                            entries.as_mut_ptr().cast(),
                            (n * size_of::<PSAPI_WORKING_SET_EX_INFORMATION>()) as u32,
                        )
                    };
                    if ok != 0 {
                        for (i, entry) in entries[..n].iter().enumerate() {
                            // SAFETY: every bit pattern is a valid usize.
                            if unsafe { entry.VirtualAttributes.Flags } & 1 != 0 {
                                let page = (backing_offset + at) / PAGE + i;
                                seen[page / 64] |= 1 << (page % 64);
                            }
                        }
                    }
                    at += n * PAGE;
                }
            }
        }

        pub fn touched(&self, ptr: *mut u8, arenas: &[(&Arena, *mut u8)]) -> (u64, u64) {
            let chunks = self.size.div_ceil(CHUNK);
            let mut seen = vec![0u64; (self.size / PAGE).div_ceil(64)];
            let own = (0..chunks)
                .filter(|&chunk| self.is_committed(chunk))
                .map(|chunk| {
                    (
                        ptr as usize + chunk * CHUNK,
                        chunk * CHUNK,
                        self.chunk_len(chunk),
                    )
                });
            Self::mark_resident(own, &mut seen);
            for &(arena, base) in arenas {
                let views: Vec<(usize, usize, usize)> = arena
                    .views
                    .lock()
                    .iter()
                    .map(|(&offset, view)| (base as usize + offset, view.backing_offset, view.len))
                    .collect();
                Self::mark_resident(views.into_iter(), &mut seen);
            }
            let (mut pages, mut granules) = (0u64, 0u64);
            for chunk in (0..chunks).filter(|&chunk| self.is_committed(chunk)) {
                let len = self.chunk_len(chunk);
                let first = chunk * CHUNK / PAGE;
                let touched: u64 = (first..first + len / PAGE)
                    .map(|page| (seen[page / 64] >> (page % 64)) & 1)
                    .sum();
                pages += touched;
                if touched > 0 {
                    granules += len as u64;
                }
            }
            (pages * PAGE as u64, granules)
        }

        pub fn release(&self, ptr: *mut u8, offset: usize, len: usize) -> Range<usize> {
            let first = offset.div_ceil(CHUNK);
            let last = ((offset + len) / CHUNK).max(first);
            let mut slots = self.slots.lock();
            for chunk in first..last {
                let Slot::Committed(section) = slots[chunk] else {
                    continue;
                };
                let (len, at) = (self.chunk_len(chunk), ptr.wrapping_add(chunk * CHUNK));
                // SAFETY: at is the chunk's own view; unmapping it leaves the chunk's placeholder.
                if unsafe { unmap_view(at) }.is_err() {
                    return offset..offset;
                }
                self.chunks.created[chunk / 64].fetch_and(!(1 << (chunk % 64)), Ordering::Release);
                let charged = self.chunks.charged[chunk / 64]
                    .fetch_and(!(1 << (chunk % 64)), Ordering::Release);
                if charged & (1 << (chunk % 64)) != 0 {
                    self.chunks.charged_chunks.fetch_sub(1, Ordering::Relaxed);
                }
                // SAFETY: the section is ours; views elsewhere keep it alive until they are
                // unmapped too. [at, at + len) is the free placeholder the unmap left.
                slots[chunk] = unsafe {
                    CloseHandle(section);
                    match map_over(self.scratch, at, len) {
                        Ok(()) => Slot::Scratch,
                        Err(_) => Slot::Reserved,
                    }
                };
            }
            first * CHUNK..(last * CHUNK).min(self.size).max(first * CHUNK)
        }

        pub fn discard(&self, ptr: *mut u8, offset: usize, len: usize) {
            let mut done = 0;
            while done < len {
                let at = offset + done;
                let step = (CHUNK - at % CHUNK).min(len - done);
                if bit(&self.chunks.charged, at >> CHUNK_BITS) {
                    // SAFETY: the range is committed and inside the backing view; a chunk not
                    // committed yet reads as zero once it is.
                    unsafe { std::ptr::write_bytes(ptr.add(at), 0, step) };
                }
                done += step;
            }
        }

        pub fn destroy(&mut self, ptr: *mut u8, size: usize) {
            self.touch = None;
            for (chunk, slot) in self.slots.get_mut().iter().enumerate() {
                let at = ptr.wrapping_add(chunk * CHUNK);
                // SAFETY: committed and scratch slots hold our views; unmapping keeps their
                // placeholders, which the coalesce below joins so one release frees the range.
                unsafe {
                    match *slot {
                        Slot::Committed(section) => {
                            let _ = unmap_view(at);
                            CloseHandle(section);
                        }
                        Slot::Scratch => {
                            let _ = unmap_view(at);
                        }
                        Slot::Reserved => {}
                    }
                }
            }
            // SAFETY: every view is gone, so the range is placeholders only.
            unsafe {
                VirtualFree(ptr.cast(), size, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS);
                VirtualFree(ptr.cast(), 0, MEM_RELEASE);
                CloseHandle(self.scratch);
            }
        }
    }

    #[derive(Clone, Copy)]
    struct View {
        len: usize,
        backing_offset: usize,
        prot: HostProtection,
    }

    /// Placeholder bookkeeping. Invariant: every maximal gap between views is exactly one
    /// placeholder, so a new view can always be carved out of the gap that contains it.
    pub struct Arena {
        size: usize,
        views: Mutex<BTreeMap<usize, View>>,
        /// Restricted pages `place` mapped read-write again, for `take_exposed`.
        exposed: Mutex<Vec<usize>>,
        #[cfg(test)]
        pub calls: OsCalls,
    }

    /// Counts of the placeholder and view calls an arena made, for tests.
    #[cfg(test)]
    #[derive(Default)]
    pub struct OsCalls {
        pub map: std::sync::atomic::AtomicUsize,
        pub unmap: std::sync::atomic::AtomicUsize,
        pub coalesce: std::sync::atomic::AtomicUsize,
    }

    #[cfg(test)]
    impl OsCalls {
        pub fn take(&self) -> (usize, usize, usize) {
            (
                self.map.swap(0, Ordering::Relaxed),
                self.unmap.swap(0, Ordering::Relaxed),
                self.coalesce.swap(0, Ordering::Relaxed),
            )
        }
    }

    impl Arena {
        pub fn reserve(size: usize) -> io::Result<(Self, *mut u8)> {
            // SAFETY: reserves one placeholder covering the whole arena.
            let base = unsafe {
                VirtualAlloc2(
                    GetCurrentProcess(),
                    std::ptr::null(),
                    size,
                    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                    PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if base.is_null() {
                return Err(io::Error::last_os_error());
            }
            Ok((
                Self {
                    size,
                    views: Mutex::new(BTreeMap::new()),
                    exposed: Mutex::default(),
                    #[cfg(test)]
                    calls: OsCalls::default(),
                },
                base.cast(),
            ))
        }

        fn gap_around(views: &BTreeMap<usize, View>, offset: usize, size: usize) -> (usize, usize) {
            let start = views
                .range(..offset)
                .next_back()
                .map_or(0, |(s, v)| s + v.len);
            let end = views.range(offset..).next().map_or(size, |(s, _)| *s);
            (start, end)
        }

        fn coalesce(&self, base: *mut u8, views: &BTreeMap<usize, View>, offset: usize) {
            let (start, end) = Self::gap_around(views, offset, self.size);
            if start < end {
                #[cfg(test)]
                self.calls.coalesce.fetch_add(1, Ordering::Relaxed);
                // SAFETY: [start, end) is a run of free placeholders inside the arena; the call
                // fails harmlessly when it already is a single placeholder.
                unsafe {
                    VirtualFree(
                        base.add(start).cast(),
                        end - start,
                        MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
                    )
                };
            }
        }

        /// Maps `view` at `offset` as one view per backing chunk it spans, since each chunk is a
        /// section of its own.
        fn place(
            &self,
            base: *mut u8,
            backing: &Backing,
            views: &mut BTreeMap<usize, View>,
            mut offset: usize,
            mut view: View,
            restricted: &Restricted,
        ) -> io::Result<()> {
            loop {
                let in_chunk = CHUNK - view.backing_offset % CHUNK;
                if view.len <= in_chunk {
                    return self.place_one(base, backing, views, offset, view, restricted);
                }
                let head = View {
                    len: in_chunk,
                    ..view
                };
                self.place_one(base, backing, views, offset, head, restricted)?;
                offset += in_chunk;
                view.len -= in_chunk;
                view.backing_offset += in_chunk;
            }
        }

        fn place_one(
            &self,
            base: *mut u8,
            backing: &Backing,
            views: &mut BTreeMap<usize, View>,
            offset: usize,
            view: View,
            restricted: &Restricted,
        ) -> io::Result<()> {
            // SAFETY: [offset, offset + len) lies in one free placeholder (the gap invariant), so
            // splitting it out and replacing it with a view of our section is valid. The split
            // fails harmlessly when the range already is the whole placeholder. The view is mapped
            // read-write because its protection at map time caps what VirtualProtect may set later.
            #[cfg(test)]
            self.calls.map.fetch_add(1, Ordering::Relaxed);
            unsafe {
                VirtualFree(
                    base.add(offset).cast(),
                    view.len,
                    MEM_RELEASE | MEM_SPLIT_PLACEHOLDER,
                );
                let mapped = MapViewOfFile3(
                    backing.section(view.backing_offset >> CHUNK_BITS),
                    GetCurrentProcess(),
                    base.add(offset).cast(),
                    (view.backing_offset % CHUNK) as u64,
                    view.len,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                );
                if mapped.Value.is_null() {
                    return Err(io::Error::last_os_error());
                }
            }
            views.insert(
                offset,
                View {
                    prot: HostProtection::ReadWrite,
                    ..view
                },
            );
            let reduced_runs: Vec<(usize, usize, HostProtection)> =
                restricted_runs(restricted, offset, offset + view.len);
            if view.prot != HostProtection::ReadWrite
                || reduced_runs
                    .iter()
                    .any(|&(_, _, cap)| view.prot.min(cap) != view.prot)
            {
                backing.charge_range(view.backing_offset, view.len)?;
            }
            if view.prot != HostProtection::ReadWrite {
                let mut old = 0;
                // SAFETY: [offset, offset + len) is the view just mapped.
                check(unsafe {
                    VirtualProtect(
                        base.add(offset).cast(),
                        view.len,
                        page_flags(view.prot),
                        &mut old,
                    )
                })?;
                views.insert(offset, view);
            }
            let mut exposed = self.exposed.lock();
            for (start, len, cap) in reduced_runs {
                let reduced = view.prot.min(cap);
                if reduced != view.prot {
                    let mut old = 0;
                    // SAFETY: [start, start + len) lies inside the view at offset.
                    check(unsafe {
                        VirtualProtect(base.add(start).cast(), len, page_flags(reduced), &mut old)
                    })?;
                }
                exposed.extend((start..start + len).step_by(super::PAGE));
            }
            Ok(())
        }

        pub fn take_exposed(&self) -> Vec<usize> {
            std::mem::take(&mut *self.exposed.lock())
        }

        /// The views overlapping `[offset, offset + len)` without lying inside it, as (start, end):
        /// a map, unmap or protect there unmaps them and maps them again.
        pub fn rebuilt_views(&self, offset: usize, len: usize) -> Vec<(usize, usize)> {
            let end = offset + len;
            self.views
                .lock()
                .range(..end)
                .rev()
                .take_while(|(s, v)| **s + v.len > offset)
                .filter(|(s, v)| **s < offset || **s + v.len > end)
                .map(|(s, v)| (*s, *s + v.len))
                .collect()
        }

        #[cfg(test)]
        pub fn view_count(&self) -> usize {
            self.views.lock().len()
        }

        /// Unmaps the views overlapping `[offset, offset + len)`, mapping the parts outside it
        /// again. Returns whether any view was unmapped, which leaves placeholders to coalesce.
        fn remove_range(
            &self,
            base: *mut u8,
            backing: &Backing,
            views: &mut BTreeMap<usize, View>,
            offset: usize,
            len: usize,
            restricted: &Restricted,
        ) -> io::Result<bool> {
            let end = offset + len;
            let mut removed = false;
            while let Some((start, view)) = views
                .range(..end)
                .next_back()
                .filter(|(s, v)| **s + v.len > offset)
                .map(|(s, v)| (*s, *v))
            {
                removed = true;
                #[cfg(test)]
                self.calls.unmap.fetch_add(1, Ordering::Relaxed);
                // SAFETY: start is the base address of a view we mapped.
                let ok = unsafe {
                    UnmapViewOfFile2(
                        GetCurrentProcess(),
                        MEMORY_MAPPED_VIEW_ADDRESS {
                            Value: base.add(start).cast(),
                        },
                        MEM_PRESERVE_PLACEHOLDER,
                    )
                };
                check(ok)?;
                views.remove(&start);
                if start < offset {
                    let head = View {
                        len: offset - start,
                        ..view
                    };
                    self.place(base, backing, views, start, head, restricted)?;
                }
                let view_end = start + view.len;
                if view_end > end {
                    let tail = View {
                        len: view_end - end,
                        backing_offset: view.backing_offset + (end - start),
                        prot: view.prot,
                    };
                    self.place(base, backing, views, end, tail, restricted)?;
                }
            }
            Ok(removed)
        }

        #[allow(clippy::too_many_arguments)]
        pub fn map(
            &self,
            base: *mut u8,
            backing: &Backing,
            offset: usize,
            backing_offset: usize,
            len: usize,
            prot: HostProtection,
            restricted: &Restricted,
        ) -> io::Result<()> {
            let mut views = self.views.lock();
            if self.remove_range(base, backing, &mut views, offset, len, restricted)? {
                self.coalesce(base, &views, offset);
            }
            let view = View {
                len,
                backing_offset,
                prot,
            };
            self.place(base, backing, &mut views, offset, view, restricted)
        }

        pub fn unmap(
            &self,
            base: *mut u8,
            backing: &Backing,
            offset: usize,
            len: usize,
            restricted: &Restricted,
        ) -> io::Result<()> {
            let mut views = self.views.lock();
            if self.remove_range(base, backing, &mut views, offset, len, restricted)? {
                self.coalesce(base, &views, offset);
            }
            Ok(())
        }

        pub fn protect(
            &self,
            base: *mut u8,
            backing: &Backing,
            offset: usize,
            len: usize,
            prot: HostProtection,
            restricted: &Restricted,
        ) -> io::Result<()> {
            let mut views = self.views.lock();
            let end = offset + len;
            let overlapping: Vec<(usize, View)> = views
                .range(..end)
                .rev()
                .take_while(|(s, v)| *s + v.len > offset)
                .map(|(s, v)| (*s, *v))
                .collect();
            for (start, view) in overlapping {
                let a = start.max(offset);
                let b = (start + view.len).min(end);
                if a == start && b == start + view.len {
                    backing.charge_range(view.backing_offset, view.len)?;
                    for (run, len, prot) in protection_runs(restricted, a, b, prot) {
                        let mut old = 0;
                        // SAFETY: [run, run + len) lies inside the view at start.
                        check(unsafe {
                            VirtualProtect(base.add(run).cast(), len, page_flags(prot), &mut old)
                        })?;
                    }
                    views.insert(start, View { prot, ..view });
                } else {
                    self.remove_range(base, backing, &mut views, a, b - a, restricted)?;
                    let part = View {
                        len: b - a,
                        backing_offset: view.backing_offset + (a - start),
                        prot,
                    };
                    self.place(base, backing, &mut views, a, part, restricted)?;
                }
            }
            Ok(())
        }

        pub fn restrict(
            &self,
            base: *mut u8,
            backing: &Backing,
            offset: usize,
            len: usize,
            _prot: HostProtection,
            cap: Option<HostProtection>,
        ) -> io::Result<()> {
            let views = self.views.lock();
            let end = offset + len;
            for (&start, view) in views
                .range(..end)
                .rev()
                .take_while(|(s, v)| **s + v.len > offset)
            {
                let (a, b) = (start.max(offset), (start + view.len).min(end));
                let prot = cap.map_or(view.prot, |cap| view.prot.min(cap));
                backing.charge_range(view.backing_offset + (a - start), b - a)?;
                let mut old = 0;
                // SAFETY: [a, b) lies inside one of our views.
                check(unsafe {
                    VirtualProtect(base.add(a).cast(), b - a, page_flags(prot), &mut old)
                })?;
            }
            Ok(())
        }

        /// Unmaps every view and coalesces the arena back into one placeholder.
        pub fn clear(&self, base: *mut u8, _size: usize) -> io::Result<()> {
            let mut views = self.views.lock();
            if views.is_empty() {
                return Ok(());
            }
            for start in std::mem::take(&mut *views).into_keys() {
                #[cfg(test)]
                self.calls.unmap.fetch_add(1, Ordering::Relaxed);
                // SAFETY: start is the base address of a view we mapped.
                check(unsafe {
                    UnmapViewOfFile2(
                        GetCurrentProcess(),
                        MEMORY_MAPPED_VIEW_ADDRESS {
                            Value: base.add(start).cast(),
                        },
                        MEM_PRESERVE_PLACEHOLDER,
                    )
                })?;
            }
            self.exposed.lock().clear();
            self.coalesce(base, &views, 0);
            Ok(())
        }

        pub fn release(&mut self, base: *mut u8, size: usize) {
            let _ = self.clear(base, size);
            // SAFETY: the arena is one free placeholder again; this frees the reservation.
            unsafe { VirtualFree(base.cast(), 0, MEM_RELEASE) };
        }
    }
}
