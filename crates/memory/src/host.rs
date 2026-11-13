//! Host virtual memory: the guest DRAM backing and reserved regions that map views of it.
//!
//! The backing is a shareable memory object (a memfd on Linux, a pagefile section on Windows) of
//! the full DRAM size. Nothing is committed up front: Linux allocates pages on first touch, and
//! Windows reserves the section (`SEC_RESERVE`) and commits 2 MiB chunks when the kernel first
//! maps them. Arenas are reserved, inaccessible host ranges into which pages of the backing are
//! mapped as views, so one physical page can appear at several host addresses.

use std::collections::BTreeSet;
use std::io;

use lichii_common::PAGE_SIZE;
use parking_lot::Mutex;

const PAGE: usize = PAGE_SIZE as usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostProtection {
    None,
    Read,
    ReadWrite,
}

impl HostProtection {
    /// What a write-tracked page with this protection gets.
    fn without_write(self) -> Self {
        match self {
            Self::ReadWrite => Self::Read,
            other => other,
        }
    }
}

/// Runs of write-tracked pages inside `[start, end)`, as (offset, length).
fn tracked_runs(tracked: &BTreeSet<usize>, start: usize, end: usize) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &page in tracked.range(start..end) {
        match runs.last_mut() {
            Some((first, len)) if *first + *len == page => *len += PAGE,
            _ => runs.push((page, PAGE)),
        }
    }
    runs
}

/// `[start, end)` split into runs that get `prot` and runs of write-tracked pages that get it
/// without write, so a protection change never makes a tracked page writable, even briefly.
fn protection_runs(
    tracked: &BTreeSet<usize>,
    start: usize,
    end: usize,
    prot: HostProtection,
) -> Vec<(usize, usize, HostProtection)> {
    let reduced = prot.without_write();
    if reduced == prot {
        return vec![(start, end - start, prot)];
    }
    let mut runs = Vec::new();
    let mut at = start;
    for (first, len) in tracked_runs(tracked, start, end) {
        if first > at {
            runs.push((at, first - at, prot));
        }
        runs.push((first, len, reduced));
        at = first + len;
    }
    if at < end {
        runs.push((at, end - at, prot));
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

    pub(crate) fn supports_views(&self) -> bool {
        self.sys.supports_views()
    }
}

impl Drop for Backing {
    fn drop(&mut self) {
        self.sys.release(self.ptr, self.size);
    }
}

/// A reserved host range that holds views of a [`Backing`].
pub(crate) struct Arena {
    base: *mut u8,
    size: usize,
    sys: sys::Arena,
    /// Offsets of write-tracked pages: their views stay write-protected whatever `map` and
    /// `protect` give them, until `track` releases them or they are mapped or unmapped again.
    tracked: Mutex<BTreeSet<usize>>,
}

impl Arena {
    pub(crate) fn reserve(size: usize) -> io::Result<Self> {
        let (sys, base) = sys::Arena::reserve(size)?;
        Ok(Self {
            base,
            size,
            sys,
            tracked: Mutex::default(),
        })
    }

    fn forget_tracked(tracked: &mut BTreeSet<usize>, offset: usize, len: usize) {
        let pages: Vec<usize> = tracked.range(offset..offset + len).copied().collect();
        for page in pages {
            tracked.remove(&page);
        }
    }

    pub(crate) fn base(&self) -> *mut u8 {
        self.base
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    /// Maps `len` bytes of `backing` at `backing_offset` to `offset` in the arena, replacing any
    /// view there.
    pub(crate) fn map(
        &self,
        backing: &Backing,
        offset: usize,
        backing_offset: usize,
        len: usize,
        prot: HostProtection,
    ) -> io::Result<()> {
        assert!(offset + len <= self.size && backing_offset + len <= backing.size);
        let mut tracked = self.tracked.lock();
        Self::forget_tracked(&mut tracked, offset, len);
        self.sys.map(
            self.base,
            &backing.sys,
            offset,
            backing_offset,
            len,
            prot,
            &tracked,
        )
    }

    /// Removes views from `[offset, offset + len)`; accesses there fault again.
    pub(crate) fn unmap(&self, backing: &Backing, offset: usize, len: usize) -> io::Result<()> {
        assert!(offset + len <= self.size);
        let mut tracked = self.tracked.lock();
        Self::forget_tracked(&mut tracked, offset, len);
        self.sys
            .unmap(self.base, &backing.sys, offset, len, &tracked)
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
        let tracked = self.tracked.lock();
        self.sys
            .protect(self.base, &backing.sys, offset, len, prot, &tracked)
    }

    /// Write-protects the mapped pages of `[offset, offset + len)`, whose protection is `prot`,
    /// for write tracking, or gives them `prot` back.
    pub(crate) fn track(
        &self,
        offset: usize,
        len: usize,
        prot: HostProtection,
        tracked: bool,
    ) -> io::Result<()> {
        assert!(offset + len <= self.size);
        let mut set = self.tracked.lock();
        for page in (offset..offset + len).step_by(PAGE) {
            if tracked {
                set.insert(page);
            } else {
                set.remove(&page);
            }
        }
        self.sys.track(self.base, offset, len, prot, tracked)
    }

    /// Offsets of write-tracked pages whose views had to be mapped again since the last call.
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
    use std::collections::BTreeSet;
    use std::io;

    use super::{HostProtection, protection_runs};

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

        pub fn discard(&self, ptr: *mut u8, offset: usize, len: usize) {
            #[cfg(any(target_os = "linux", target_os = "android"))]
            if self.fd >= 0 {
                // SAFETY: punching a hole in our own memfd frees and zeroes those pages in every
                // view.
                let ret = unsafe {
                    libc::fallocate(
                        self.fd,
                        libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE,
                        offset as libc::off_t,
                        len as libc::off_t,
                    )
                };
                if ret == 0 {
                    return;
                }
            }
            // SAFETY: the range is inside the backing view.
            unsafe { std::ptr::write_bytes(ptr.add(offset), 0, len) };
        }

        pub fn release(&mut self, ptr: *mut u8, size: usize) {
            // SAFETY: ptr/size is the mapping created in create; fd is ours.
            unsafe {
                libc::munmap(ptr.cast(), size);
                if self.fd >= 0 {
                    libc::close(self.fd);
                }
            }
        }
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
            _tracked: &BTreeSet<usize>,
        ) -> io::Result<()> {
            // SAFETY: the target range lies inside the arena reservation this struct owns.
            let ptr = unsafe {
                libc::mmap(
                    base.add(offset).cast(),
                    len,
                    prot_flags(prot),
                    libc::MAP_SHARED | libc::MAP_FIXED,
                    backing.fd,
                    backing_offset as libc::off_t,
                )
            };
            if ptr == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        }

        pub fn unmap(
            &self,
            base: *mut u8,
            _backing: &Backing,
            offset: usize,
            len: usize,
            _tracked: &BTreeSet<usize>,
        ) -> io::Result<()> {
            // SAFETY: the range lies inside the arena reservation.
            let ptr = reserve_at(unsafe { base.add(offset) }, len, true);
            if ptr == libc::MAP_FAILED {
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
            tracked: &BTreeSet<usize>,
        ) -> io::Result<()> {
            for (start, run, prot) in protection_runs(tracked, offset, offset + len, prot) {
                // SAFETY: the run lies inside the arena reservation.
                check(unsafe { libc::mprotect(base.add(start).cast(), run, prot_flags(prot)) })?;
            }
            Ok(())
        }

        pub fn take_exposed(&self) -> Vec<usize> {
            Vec::new()
        }

        pub fn track(
            &self,
            base: *mut u8,
            offset: usize,
            len: usize,
            prot: HostProtection,
            tracked: bool,
        ) -> io::Result<()> {
            let prot = if tracked { prot.without_write() } else { prot };
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::io;
    use std::sync::atomic::{AtomicU64, Ordering};

    use parking_lot::Mutex;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_ALL_ACCESS, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
        MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS,
        MapViewOfFile, MapViewOfFile3, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY,
        PAGE_READWRITE, SEC_RESERVE, UnmapViewOfFile, UnmapViewOfFile2, VirtualAlloc,
        VirtualAlloc2, VirtualFree, VirtualProtect,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    use super::{HostProtection, protection_runs, tracked_runs};

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

    pub struct Backing {
        section: HANDLE,
        size: usize,
        committed: Box<[AtomicU64]>,
        commit_lock: Mutex<()>,
    }

    impl Backing {
        pub fn create(size: usize) -> io::Result<(Self, *mut u8)> {
            let size64 = size as u64;
            // SAFETY: creates a pagefile-backed section that is only reserved, never committed as
            // a whole, then maps one view of it.
            unsafe {
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE | SEC_RESERVE,
                    (size64 >> 32) as u32,
                    size64 as u32,
                    std::ptr::null(),
                );
                if section.is_null() {
                    return Err(io::Error::last_os_error());
                }
                let view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
                if view.Value.is_null() {
                    let err = io::Error::last_os_error();
                    CloseHandle(section);
                    return Err(err);
                }
                let chunks = size.div_ceil(CHUNK);
                let committed = (0..chunks.div_ceil(64))
                    .map(|_| AtomicU64::new(0))
                    .collect();
                Ok((
                    Self {
                        section,
                        size,
                        committed,
                        commit_lock: Mutex::new(()),
                    },
                    view.Value.cast(),
                ))
            }
        }

        pub fn supports_views(&self) -> bool {
            true
        }

        fn is_committed(&self, chunk: usize) -> bool {
            self.committed[chunk / 64].load(Ordering::Acquire) & (1 << (chunk % 64)) != 0
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
            let _guard = self.commit_lock.lock();
            for chunk in first..=last {
                if self.is_committed(chunk) {
                    continue;
                }
                let start = chunk * CHUNK;
                let len = CHUNK.min(self.size - start);
                // SAFETY: commits pages of our own reserved section view.
                let ret =
                    unsafe { VirtualAlloc(ptr.add(start).cast(), len, MEM_COMMIT, PAGE_READWRITE) };
                if ret.is_null() {
                    return Err(io::Error::last_os_error());
                }
                self.committed[chunk / 64].fetch_or(1 << (chunk % 64), Ordering::Release);
            }
            Ok(())
        }

        pub fn discard(&self, ptr: *mut u8, offset: usize, len: usize) {
            let mut done = 0;
            while done < len {
                let at = offset + done;
                let step = (CHUNK - at % CHUNK).min(len - done);
                if self.is_committed(at >> CHUNK_BITS) {
                    // SAFETY: the range is committed and inside the backing view.
                    unsafe { std::ptr::write_bytes(ptr.add(at), 0, step) };
                }
                done += step;
            }
        }

        pub fn release(&mut self, ptr: *mut u8, _size: usize) {
            // SAFETY: unmaps the view made in create and closes our section handle.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: ptr.cast() });
                CloseHandle(self.section);
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
        /// Tracked pages `place` mapped read-write again, for `take_exposed`.
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

        fn place(
            &self,
            base: *mut u8,
            backing: &Backing,
            views: &mut BTreeMap<usize, View>,
            offset: usize,
            view: View,
            tracked: &BTreeSet<usize>,
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
                    backing.section,
                    GetCurrentProcess(),
                    base.add(offset).cast(),
                    view.backing_offset as u64,
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
            let reduced = view.prot.without_write();
            let mut exposed = self.exposed.lock();
            for (start, len) in tracked_runs(tracked, offset, offset + view.len) {
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
            tracked: &BTreeSet<usize>,
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
                    self.place(base, backing, views, start, head, tracked)?;
                }
                let view_end = start + view.len;
                if view_end > end {
                    let tail = View {
                        len: view_end - end,
                        backing_offset: view.backing_offset + (end - start),
                        prot: view.prot,
                    };
                    self.place(base, backing, views, end, tail, tracked)?;
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
            tracked: &BTreeSet<usize>,
        ) -> io::Result<()> {
            let mut views = self.views.lock();
            if self.remove_range(base, backing, &mut views, offset, len, tracked)? {
                self.coalesce(base, &views, offset);
            }
            let view = View {
                len,
                backing_offset,
                prot,
            };
            self.place(base, backing, &mut views, offset, view, tracked)
        }

        pub fn unmap(
            &self,
            base: *mut u8,
            backing: &Backing,
            offset: usize,
            len: usize,
            tracked: &BTreeSet<usize>,
        ) -> io::Result<()> {
            let mut views = self.views.lock();
            if self.remove_range(base, backing, &mut views, offset, len, tracked)? {
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
            tracked: &BTreeSet<usize>,
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
                    for (run, len, prot) in protection_runs(tracked, a, b, prot) {
                        let mut old = 0;
                        // SAFETY: [run, run + len) lies inside the view at start.
                        check(unsafe {
                            VirtualProtect(base.add(run).cast(), len, page_flags(prot), &mut old)
                        })?;
                    }
                    views.insert(start, View { prot, ..view });
                } else {
                    self.remove_range(base, backing, &mut views, a, b - a, tracked)?;
                    let part = View {
                        len: b - a,
                        backing_offset: view.backing_offset + (a - start),
                        prot,
                    };
                    self.place(base, backing, &mut views, a, part, tracked)?;
                }
            }
            Ok(())
        }

        pub fn track(
            &self,
            base: *mut u8,
            offset: usize,
            len: usize,
            _prot: HostProtection,
            tracked: bool,
        ) -> io::Result<()> {
            let views = self.views.lock();
            let end = offset + len;
            for (&start, view) in views
                .range(..end)
                .rev()
                .take_while(|(s, v)| **s + v.len > offset)
            {
                let (a, b) = (start.max(offset), (start + view.len).min(end));
                let prot = if tracked {
                    view.prot.without_write()
                } else {
                    view.prot
                };
                let mut old = 0;
                // SAFETY: [a, b) lies inside one of our views.
                check(unsafe {
                    VirtualProtect(base.add(a).cast(), b - a, page_flags(prot), &mut old)
                })?;
            }
            Ok(())
        }

        pub fn release(&mut self, base: *mut u8, _size: usize) {
            let views = std::mem::take(self.views.get_mut());
            // SAFETY: unmaps every view we made, then frees the placeholders.
            unsafe {
                for start in views.keys() {
                    UnmapViewOfFile2(
                        GetCurrentProcess(),
                        MEMORY_MAPPED_VIEW_ADDRESS {
                            Value: base.add(*start).cast(),
                        },
                        MEM_PRESERVE_PLACEHOLDER,
                    );
                }
                VirtualFree(
                    base.cast(),
                    self.size,
                    MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS,
                );
                VirtualFree(base.cast(), 0, MEM_RELEASE);
            }
        }
    }
}
