//! Translated code and the fault-site tables the fastmem fault handler searches.

use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

use lichii_common::{PAGE_SIZE, VAddr};
use lichii_memory::HostFault;

use super::x64::Compiled;
use super::{JitState, x64};
use crate::ThreadContext;
use crate::decoder::Inst;

const CODE_SIZE: usize = 64 << 20;
const FAST_BITS: u32 = 14;
const EMPTY: (u64, u32) = (u64::MAX, u32::MAX);
const REGIONS: usize = 256;

pub(super) type Trampoline =
    unsafe extern "sysv64" fn(*mut ThreadContext, *mut JitState, *mut u8, *const u8) -> u64;

/// How the dispatcher runs the guest code at a block's start.
#[derive(Clone, Copy)]
pub(super) enum Code {
    /// Compiled host code.
    Native(*const u8),
    /// A lone instruction the translator rejects, for the interpreter.
    Interpret(Inst),
}

/// One translated guest block.
#[derive(Clone, Copy)]
pub(super) struct Entry {
    pub code: Code,
    /// Guest bytes covered.
    len: u64,
}

/// Code memory for one cache: one shared-memory object mapped twice, so blocks run from a
/// read-execute view and are written and patched through a read-write one, never through
/// `mprotect`.
struct CodeMemory {
    exec: *mut u8,
    write: *mut u8,
    used: usize,
}

impl CodeMemory {
    fn new() -> Option<Self> {
        let (write, exec) = sys::map(CODE_SIZE)?;
        Some(Self {
            exec,
            write,
            used: 0,
        })
    }

    fn push(&mut self, code: &[u8]) -> Option<*const u8> {
        let start = self.used.next_multiple_of(16);
        if start + code.len() > CODE_SIZE {
            return None;
        }
        // SAFETY: `[start, start + len)` is inside the mapping, and no code runs while the cache
        // is mutated.
        unsafe { std::ptr::copy_nonoverlapping(code.as_ptr(), self.write.add(start), code.len()) };
        self.used = start + code.len();
        // SAFETY: as above.
        Some(unsafe { self.exec.add(start) })
    }

    /// Points the `jmp rel32` at executable address `site` to `target`.
    ///
    /// # Safety
    /// `site` must be a link site of a block in this memory, and none of its code may be
    /// running on another thread.
    unsafe fn patch(&mut self, site: usize, target: usize) {
        let rel = target.wrapping_sub(site + 5) as i32;
        // SAFETY: guaranteed by the caller; the operand follows the one-byte opcode, at the same
        // offset in the writable view.
        unsafe {
            let operand = self.write.add(site + 1 - self.exec as usize);
            std::ptr::write_unaligned(operand.cast::<i32>(), rel);
        }
    }
}

impl Drop for CodeMemory {
    fn drop(&mut self) {
        // SAFETY: both views came from one `sys::map(CODE_SIZE)`.
        unsafe {
            sys::unmap(self.exec, CODE_SIZE);
            sys::unmap(self.write, CODE_SIZE);
        }
    }
}

/// Host addresses of a cache's fastmem accesses and their slow paths, sorted by access.
struct SiteTable(Vec<(usize, usize)>);

/// A process-wide slot naming one cache's code range and site table.
struct Region {
    start: AtomicUsize,
    end: AtomicUsize,
    sites: AtomicPtr<SiteTable>,
}

static REGION_TABLE: [Region; REGIONS] = [const {
    Region {
        start: AtomicUsize::new(0),
        end: AtomicUsize::new(0),
        sites: AtomicPtr::new(std::ptr::null_mut()),
    }
}; REGIONS];
static REGISTRY: Mutex<()> = Mutex::new(());

/// The fastmem fault handler shared by every core: moves a faulting access to its slow path.
pub(super) fn on_fault(_data: usize, _guest_addr: u64, fault: &mut HostFault) -> bool {
    let pc = fault.pc() as usize;
    for region in &REGION_TABLE {
        let start = region.start.load(Ordering::Acquire);
        if start == 0 || pc < start || pc >= region.end.load(Ordering::Acquire) {
            continue;
        }
        // SAFETY: a fault at `pc` runs on the thread that owns this region's cache, which only
        // changes the table between runs of its own code, and clears the slot before freeing it.
        let sites = unsafe { &*region.sites.load(Ordering::Acquire) };
        return match sites.0.binary_search_by_key(&pc, |&(site, _)| site) {
            Ok(i) => {
                fault.set_pc(sites.0[i].1 as u64);
                true
            }
            Err(_) => false,
        };
    }
    false
}

/// Translated blocks of one core, keyed by guest address, with the code memory they live in.
/// Blocks never cross a guest page.
pub(super) struct JitCache {
    memory: CodeMemory,
    trampoline: Trampoline,
    materialize: usize,
    code_start: usize,
    entries: Vec<Entry>,
    free: Vec<u32>,
    by_start: BTreeMap<u64, u32>,
    fast: Box<[(u64, u32)]>,
    sites: Box<SiteTable>,
    region: Option<usize>,
    /// The table indirect branches probe, indexed like `fast` but with `TABLE_BITS` bits.
    table: Box<[(u64, usize)]>,
    /// The value of an empty table slot: a key and the miss stub.
    empty: (u64, usize),
    /// Host addresses of the linked `jmp`s into each guest block.
    links: HashMap<u64, Vec<usize>>,
    /// Bumped by every `clear`, which discards all code and so every pending link site.
    generation: u64,
}

fn table_slot(pc: u64) -> usize {
    ((pc >> 2) as usize) & ((1 << x64::TABLE_BITS) - 1)
}

#[inline]
fn fast_slot(pc: u64) -> usize {
    ((pc >> 2) as usize) & ((1 << FAST_BITS) - 1)
}

impl JitCache {
    /// `profile` builds the materialize stub that counts its write-backs.
    pub fn new(profile: bool) -> Option<Self> {
        let mut memory = CodeMemory::new()?;
        let code = memory.push(&x64::trampoline())?;
        // SAFETY: the bytes at `code` are the trampoline, which has this signature.
        let trampoline = unsafe { std::mem::transmute::<*const u8, Trampoline>(code) };
        let empty = (u64::MAX, memory.push(&x64::miss_stub())? as usize);
        let materialize = memory.push(&x64::materialize_stub(profile))? as usize;
        let mut sites = Box::new(SiteTable(Vec::new()));
        let region = {
            let _guard = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
            let slot = REGION_TABLE
                .iter()
                .position(|r| r.start.load(Ordering::Acquire) == 0);
            if let Some(slot) = slot {
                let region = &REGION_TABLE[slot];
                region.sites.store(&mut *sites, Ordering::Release);
                region
                    .end
                    .store(memory.exec as usize + CODE_SIZE, Ordering::Release);
                region.start.store(memory.exec as usize, Ordering::Release);
            }
            slot
        };
        Some(Self {
            code_start: memory.used,
            memory,
            trampoline,
            materialize,
            entries: Vec::new(),
            free: Vec::new(),
            by_start: BTreeMap::new(),
            fast: vec![EMPTY; 1 << FAST_BITS].into_boxed_slice(),
            sites,
            region,
            table: vec![empty; 1 << x64::TABLE_BITS].into_boxed_slice(),
            empty,
            links: HashMap::new(),
            generation: 0,
        })
    }

    pub fn trampoline(&self) -> Trampoline {
        self.trampoline
    }

    /// The address of `x64::materialize_stub`.
    pub fn materialize(&self) -> usize {
        self.materialize
    }

    pub fn table(&self) -> *const (u64, usize) {
        self.table.as_ptr()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Makes `code`, the block at guest `pc`, reachable from indirect branches.
    #[inline]
    pub fn publish(&mut self, pc: u64, code: *const u8) {
        self.table[table_slot(pc)] = (pc, code as usize);
    }

    /// Points the link `site`, recorded by a block exit in this generation, at `code`, the block
    /// at guest `target`.
    pub fn link(&mut self, site: usize, target: u64, code: *const u8) {
        // SAFETY: the site belongs to code of the current generation, and this core's code only
        // runs on the thread that owns the cache.
        unsafe { self.memory.patch(site, code as usize) };
        self.links.entry(target).or_default().push(site);
    }

    /// Whether the fault handler can find this cache's sites, so its code may use fastmem.
    pub fn handles_faults(&self) -> bool {
        self.region.is_some()
    }

    #[inline]
    pub fn lookup(&mut self, pc: u64) -> Option<Entry> {
        let slot = fast_slot(pc);
        let (start, index) = self.fast[slot];
        if start == pc && index != EMPTY.1 {
            return Some(self.entries[index as usize]);
        }
        let index = *self.by_start.get(&pc)?;
        self.fast[slot] = (pc, index);
        Some(self.entries[index as usize])
    }

    /// Adds a block covering guest `[start, start + len)`.
    pub fn insert(&mut self, start: u64, len: u64, body: Result<Compiled, Inst>) -> Entry {
        let code = match body {
            Err(inst) => Code::Interpret(inst),
            Ok(compiled) => {
                let code = match self.memory.push(&compiled.code) {
                    Some(code) => code,
                    None => {
                        self.clear();
                        self.memory
                            .push(&compiled.code)
                            .expect("one block fits in an empty code cache")
                    }
                };
                let base = code as usize;
                self.sites.0.extend(
                    compiled
                        .sites
                        .iter()
                        .map(|&(site, stub)| (base + site as usize, base + stub as usize)),
                );
                Code::Native(code)
            }
        };
        let entry = Entry { code, len };
        let index = match self.free.pop() {
            Some(index) => {
                self.entries[index as usize] = entry;
                index
            }
            None => {
                self.entries.push(entry);
                (self.entries.len() - 1) as u32
            }
        };
        self.by_start.insert(start, index);
        self.fast[fast_slot(start)] = (start, index);
        entry
    }

    /// Drops every block that overlaps `[addr, addr + size)`. Their code stays in memory, unused,
    /// until the next `clear`.
    pub fn invalidate(&mut self, addr: VAddr, size: u64) {
        if size == 0 {
            return;
        }
        let end = addr.saturating_add(size);
        let first = addr.saturating_sub(PAGE_SIZE);
        let doomed: Vec<(u64, u32)> = self
            .by_start
            .range(first..end)
            .filter(|&(&start, &index)| start + self.entries[index as usize].len > addr)
            .map(|(&start, &index)| (start, index))
            .collect();
        for (start, index) in doomed {
            for site in self.links.remove(&start).unwrap_or_default() {
                // SAFETY: link sites stay in code memory until the next `clear`, which forgets
                // them, and no code of this cache runs while it is changed.
                unsafe { self.memory.patch(site, site + 5) };
            }
            let slot = table_slot(start);
            if self.table[slot].0 == start {
                self.table[slot] = self.empty;
            }
            self.by_start.remove(&start);
            let slot = fast_slot(start);
            if self.fast[slot].0 == start {
                self.fast[slot] = EMPTY;
            }
            self.free.push(index);
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.free.clear();
        self.by_start.clear();
        self.fast.fill(EMPTY);
        self.sites.0.clear();
        self.table.fill(self.empty);
        self.links.clear();
        self.generation += 1;
        self.memory.used = self.code_start;
    }
}

impl Drop for JitCache {
    fn drop(&mut self) {
        if let Some(slot) = self.region {
            let _guard = REGISTRY.lock().unwrap_or_else(|e| e.into_inner());
            REGION_TABLE[slot].start.store(0, Ordering::Release);
        }
    }
}

#[cfg(unix)]
mod sys {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn create_fd() -> libc::c_int {
        // SAFETY: plain system call with a NUL-terminated name.
        unsafe { libc::memfd_create(c"lichii-jit".as_ptr(), libc::MFD_CLOEXEC) }
    }

    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    fn create_fd() -> libc::c_int {
        static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let name = format!("/lichii-jit-{}-{id}\0", std::process::id());
        // SAFETY: the name is NUL-terminated; the object is unlinked right away so it lives only
        // as long as the descriptor.
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

    /// Maps one shared-memory object of `size` bytes twice: read-write, then read-execute.
    pub fn map(size: usize) -> Option<(*mut u8, *mut u8)> {
        let fd = create_fd();
        if fd < 0 {
            return None;
        }
        let view = |prot| {
            // SAFETY: a fresh shared mapping of a descriptor this function owns.
            let ptr =
                unsafe { libc::mmap(std::ptr::null_mut(), size, prot, libc::MAP_SHARED, fd, 0) };
            (ptr != libc::MAP_FAILED).then_some(ptr.cast::<u8>())
        };
        // SAFETY: `fd` is open and owned here.
        let sized = unsafe { libc::ftruncate(fd, size as libc::off_t) } == 0;
        let write = sized
            .then(|| view(libc::PROT_READ | libc::PROT_WRITE))
            .flatten();
        let exec = write.and_then(|_| view(libc::PROT_READ | libc::PROT_EXEC));
        // SAFETY: the mappings keep the object alive without the descriptor.
        unsafe { libc::close(fd) };
        match (write, exec) {
            (Some(write), Some(exec)) => Some((write, exec)),
            (Some(write), None) => {
                // SAFETY: `write` is the mapping made above.
                unsafe { unmap(write, size) };
                None
            }
            _ => None,
        }
    }

    /// # Safety
    /// `ptr` and `size` must name one view from `map`, and no code in it may run again.
    pub unsafe fn unmap(ptr: *mut u8, size: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe { libc::munmap(ptr.cast(), size) };
    }
}

#[cfg(windows)]
mod sys {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP, FILE_MAP_EXECUTE, FILE_MAP_READ, FILE_MAP_WRITE,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_EXECUTE_READWRITE, UnmapViewOfFile,
    };

    /// Maps one pagefile-backed section of `size` bytes twice: read-write, then read-execute.
    pub fn map(size: usize) -> Option<(*mut u8, *mut u8)> {
        // SAFETY: a fresh anonymous section has no preconditions.
        let section = unsafe {
            CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_EXECUTE_READWRITE,
                (size as u64 >> 32) as u32,
                size as u32,
                std::ptr::null(),
            )
        };
        if section.is_null() {
            return None;
        }
        let view = |access: FILE_MAP| {
            // SAFETY: `section` is a live section of `size` bytes.
            let ptr = unsafe { MapViewOfFile(section, access, 0, 0, size) }.Value;
            (!ptr.is_null()).then_some(ptr.cast::<u8>())
        };
        let write = view(FILE_MAP_WRITE);
        let exec = write.and_then(|_| view(FILE_MAP_READ | FILE_MAP_EXECUTE));
        // SAFETY: the views keep the section alive without the handle.
        unsafe { CloseHandle(section) };
        match (write, exec) {
            (Some(write), Some(exec)) => Some((write, exec)),
            (Some(write), None) => {
                // SAFETY: `write` is the view made above.
                unsafe { unmap(write, size) };
                None
            }
            _ => None,
        }
    }

    /// # Safety
    /// `ptr` must be one view from `map`, and no code in it may run again.
    pub unsafe fn unmap(ptr: *mut u8, _size: usize) {
        // SAFETY: guaranteed by the caller.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: ptr.cast() }) };
    }
}
