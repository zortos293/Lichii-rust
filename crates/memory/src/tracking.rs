//! CPU write tracking for pages the GPU caches.
//!
//! A GPU cache that copies guest memory (a texture, a vertex buffer, shader code) takes a
//! [`CachedPages`] guard over it with [`DeviceAddressSpace::cache_pages`] before it reads the
//! bytes. Every physical page under a guard is *tracked* in the attached process: a per-physical
//! page flag is set, and the fastmem views of every virtual page mapping it are write-protected.
//! The first CPU write to a tracked page lands, clears the flag, releases the views on every
//! alias, and only then marks the physical page dirty. The GPU thread drains the dirty pages with
//! [`DeviceAddressSpace::take_cpu_writes`] and invalidates what it cached there; a cache that
//! uploads again takes a new guard, which tracks the pages again.
//!
//! Invariant: every CPU write to a page that was tracked when the write began is either seen by
//! the upload that follows the tracking, or published as dirty after its bytes landed.
//! - Writes from host code go through the page table to the DRAM backing's own view, which is
//!   never protected, so they never fault. After each page's bytes they issue a sequentially
//!   consistent fence and read the flag of the physical page they wrote, found from the host
//!   pointer, never by translating the address again, so a concurrent remap cannot hide it.
//!   Tracking sets the flag and issues the same fence before the cache reads. Either the cache
//!   sees the bytes or the writer sees the flag.
//! - JIT stores to a write-protected fastmem view fault into their slow path, which writes
//!   through the page table as above. Protection changes never raise a tracked page's view, not
//!   even briefly; where a view has to be mapped again (a new alias of a held page, a Windows
//!   view split), the view is writable for a moment, so those pages are published as dirty once
//!   they are protected again.
//! - The dirty bit is set with release ordering after the write and read with acquire ordering,
//!   so a drain that sees it also sees the bytes.
//!
//! Pages that are not tracked cost nothing on fastmem accesses; the first write to a tracked page
//! costs one fault, once per cache cycle.
//!
//! [`DeviceAddressSpace::cache_pages`]: crate::DeviceAddressSpace::cache_pages
//! [`DeviceAddressSpace::take_cpu_writes`]: crate::DeviceAddressSpace::take_cpu_writes

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use lichii_common::{PAGE_BITS, PAGE_SIZE, PAddr};

use crate::AddressSpace;

const ALIASED: u32 = 1 << 31;

/// Physical page → the pages of one address space (virtual or device) that map it. Most
/// physical pages have one mapping, kept inline; aliases spill into a map.
pub(crate) struct ReverseMap {
    first: Box<[u32]>,
    more: HashMap<u32, Vec<u32>>,
}

impl ReverseMap {
    pub(crate) fn new(physical_size: u64) -> Self {
        Self {
            first: vec![0; (physical_size / PAGE_SIZE) as usize].into_boxed_slice(),
            more: HashMap::new(),
        }
    }

    pub(crate) fn insert(&mut self, pa: PAddr, addr: u64) {
        let (index, page) = Self::key(pa, addr);
        let slot = &mut self.first[index];
        if *slot == 0 {
            *slot = page + 1;
        } else if *slot & !ALIASED != page + 1 {
            self.more.entry(index as u32).or_default().push(page);
            *slot |= ALIASED;
        }
    }

    pub(crate) fn remove(&mut self, pa: PAddr, addr: u64) {
        let (index, page) = Self::key(pa, addr);
        let key = index as u32;
        let slot = &mut self.first[index];
        if *slot & !ALIASED == page + 1 {
            let next = self.more.get_mut(&key).and_then(Vec::pop);
            *slot = next.map_or(0, |next| next + 1);
        } else if let Some(more) = self.more.get_mut(&key) {
            more.retain(|&other| other != page);
        }
        if self.more.get(&key).is_some_and(Vec::is_empty) {
            self.more.remove(&key);
        }
        *slot &= !ALIASED;
        if self.more.contains_key(&key) {
            *slot |= ALIASED;
        }
    }

    /// Calls `f` with the address of every page that maps the physical page holding `pa`.
    pub(crate) fn for_each(&self, pa: PAddr, mut f: impl FnMut(u64)) {
        let index = (pa >> PAGE_BITS) as usize;
        let slot = self.first[index];
        if slot & !ALIASED != 0 {
            f(u64::from((slot & !ALIASED) - 1) << PAGE_BITS);
        }
        if slot & ALIASED != 0 {
            for &page in &self.more[&(index as u32)] {
                f(u64::from(page) << PAGE_BITS);
            }
        }
    }

    fn key(pa: PAddr, addr: u64) -> (usize, u32) {
        let page = addr >> PAGE_BITS;
        assert!(
            page < u64::from(ALIASED - 1),
            "page {addr:#x} too high to track"
        );
        ((pa >> PAGE_BITS) as usize, page as u32)
    }
}

/// One flag per physical page, set and read with sequentially consistent ordering.
pub(crate) struct PageFlags(Box<[AtomicU64]>);

impl PageFlags {
    pub(crate) fn new(physical_size: u64) -> Self {
        let words = (physical_size / PAGE_SIZE).div_ceil(64);
        Self((0..words).map(|_| AtomicU64::new(0)).collect())
    }

    fn word(&self, pa: PAddr) -> (&AtomicU64, u64) {
        let page = pa >> PAGE_BITS;
        (&self.0[(page / 64) as usize], 1 << (page % 64))
    }

    pub(crate) fn set(&self, pa: PAddr) {
        let (word, bit) = self.word(pa);
        word.fetch_or(bit, Ordering::SeqCst);
    }

    pub(crate) fn clear(&self, pa: PAddr) {
        let (word, bit) = self.word(pa);
        word.fetch_and(!bit, Ordering::SeqCst);
    }

    #[inline]
    pub(crate) fn test(&self, pa: PAddr) -> bool {
        let (word, bit) = self.word(pa);
        word.load(Ordering::SeqCst) & bit != 0
    }
}

/// Physical pages the CPU wrote while they were tracked, until the GPU thread drains them.
/// Lock-free: one bit per page, a summary bit per word of bits, and a pending flag.
pub(crate) struct DirtyPages {
    words: Box<[AtomicU64]>,
    summary: Box<[AtomicU64]>,
    pending: AtomicBool,
}

impl DirtyPages {
    pub(crate) fn new(physical_size: u64) -> Self {
        let words = (physical_size / PAGE_SIZE).div_ceil(64);
        Self {
            words: (0..words).map(|_| AtomicU64::new(0)).collect(),
            summary: (0..words.div_ceil(64)).map(|_| AtomicU64::new(0)).collect(),
            pending: AtomicBool::new(false),
        }
    }

    /// Marks the page holding `pa`. Call it after the write landed.
    pub(crate) fn mark(&self, pa: PAddr) {
        let page = (pa >> PAGE_BITS) as usize;
        let word = page / 64;
        self.words[word].fetch_or(1 << (page % 64), Ordering::Release);
        self.summary[word / 64].fetch_or(1 << (word % 64), Ordering::Release);
        self.pending.store(true, Ordering::Release);
    }

    pub(crate) fn pending(&self) -> bool {
        self.pending.load(Ordering::Acquire)
    }

    /// Clears the marks and calls `f` with each run of marked physical pages, as (address, size).
    pub(crate) fn take(&self, mut f: impl FnMut(PAddr, u64)) {
        if !self.pending.swap(false, Ordering::AcqRel) {
            return;
        }
        let mut run: Option<(u64, u64)> = None;
        let mut flush = |run: &mut Option<(u64, u64)>| {
            if let Some((first, count)) = run.take() {
                f(first << PAGE_BITS, count << PAGE_BITS);
            }
        };
        for (s, summary) in self.summary.iter().enumerate() {
            let mut words = summary.swap(0, Ordering::AcqRel);
            while words != 0 {
                let word = s * 64 + words.trailing_zeros() as usize;
                words &= words - 1;
                let mut bits = self.words[word].swap(0, Ordering::AcqRel);
                while bits != 0 {
                    let page = (word * 64) as u64 + u64::from(bits.trailing_zeros());
                    bits &= bits - 1;
                    match &mut run {
                        Some((first, count)) if *first + *count == page => *count += 1,
                        _ => {
                            flush(&mut run);
                            run = Some((page, 1));
                        }
                    }
                }
            }
        }
        flush(&mut run);
    }
}

/// Physical pages a GPU cache copied, tracked for CPU writes until the guard is dropped. Take it
/// before reading the bytes, and take a new one for every upload.
#[must_use = "the pages stop being tracked when the guard is dropped"]
pub struct CachedPages {
    process: Option<Arc<AddressSpace>>,
    runs: Vec<(PAddr, u64)>,
}

impl CachedPages {
    pub(crate) fn new(process: Option<Arc<AddressSpace>>, runs: Vec<(PAddr, u64)>) -> Self {
        if let Some(process) = &process {
            process.cache_physical(&runs);
        }
        Self { process, runs }
    }

    /// The physical runs held, as (address, size).
    pub fn runs(&self) -> &[(PAddr, u64)] {
        &self.runs
    }
}

impl Drop for CachedPages {
    fn drop(&mut self) {
        if let Some(process) = &self.process {
            process.uncache_physical(&self.runs);
        }
    }
}

impl std::fmt::Debug for CachedPages {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedPages")
            .field("runs", &self.runs)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use crate::{
        AddressSpace, DeviceAddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory,
    };

    const VA: u64 = 0x8000_0000;
    const PA: u64 = 0x10_0000;
    const DADDR: u64 = 0x4_0000_0000;

    fn spaces() -> (Arc<AddressSpace>, DeviceAddressSpace) {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let process = Arc::new(AddressSpace::new(physical.clone(), 39));
        process.map(VA, PA, 0x4000, MemoryPermission::READ_WRITE);
        let device = DeviceAddressSpace::new(physical, 36);
        device.map_process(DADDR, &process, VA, 0x4000).unwrap();
        device.attach_process(process.clone());
        (process, device)
    }

    fn writes(device: &DeviceAddressSpace) -> Vec<(u64, u64)> {
        let mut found = Vec::new();
        device.take_cpu_writes(|daddr, size| found.push((daddr, size)));
        found
    }

    #[test]
    fn a_cpu_write_to_a_cached_page_is_reported_once_per_cache_cycle() {
        let (process, device) = spaces();
        process.write(VA + 0x1000, 1u32).unwrap();
        assert!(!device.has_cpu_writes(), "nothing is cached yet");

        let cached = device.cache_pages(DADDR + 0x1000, 0x2000);
        assert_eq!(cached.runs(), [(PA + 0x1000, 0x2000)]);
        process.write(VA + 0x1008, 2u32).unwrap();
        assert!(device.has_cpu_writes());
        assert_eq!(writes(&device), [(DADDR + 0x1000, 0x1000)]);
        process.write(VA + 0x1010, 3u32).unwrap();
        assert!(
            writes(&device).is_empty(),
            "the page is untracked until it is cached again"
        );
        process.write(VA + 0x3000, 4u32).unwrap();
        assert!(writes(&device).is_empty(), "the page was never cached");

        drop(cached);
        let again = device.cache_pages(DADDR + 0x1000, 0x1000);
        assert_eq!(device.read::<u32>(DADDR + 0x1010).unwrap(), 3);
        process.write(VA + 0x1010, 5u32).unwrap();
        assert_eq!(
            writes(&device),
            [(DADDR + 0x1000, 0x1000)],
            "a write after the cache uploaded again is reported again"
        );
        drop(again);
        process.write(VA + 0x1010, 6u32).unwrap();
        assert!(
            writes(&device).is_empty(),
            "nothing holds the page any more"
        );
    }

    #[test]
    fn every_host_write_path_reports() {
        let (process, device) = spaces();
        let monitor = ExclusiveMonitor::new(1);
        let check = |what: &str, write: &dyn Fn()| {
            let _cached = device.cache_pages(DADDR, 0x4000);
            write();
            assert!(!writes(&device).is_empty(), "{what}");
        };
        check("write", &|| process.write(VA + 4, 1u8).unwrap());
        check("guest_write", &|| {
            process.guest_write(VA + 4, 1u16).unwrap()
        });
        check("write_bytes across pages", &|| {
            process.write_bytes(VA + 0xffe, &[1, 2, 3, 4]).unwrap()
        });
        check("fill", &|| process.fill(VA + 0x2000, 0x10, 7).unwrap());
        check("guest_fill", &|| {
            process.guest_fill(VA + 0x3000, 0x40, 0).unwrap()
        });
        check("write_with", &|| {
            process
                .write_with(VA + 8, MemoryPermission::WRITE, |ptr| {
                    // SAFETY: the pointer is valid for the page holding VA + 8.
                    unsafe { ptr.write(9) }
                })
                .unwrap()
        });
        check("store-exclusive", &|| {
            monitor.read(0, &process, VA + 0x10, 8).unwrap();
            assert!(monitor.write(0, &process, VA + 0x10, 8, 5).unwrap());
        });
    }

    #[test]
    fn aliases_of_a_cached_page_are_tracked_too() {
        let (process, device) = spaces();
        process.map(0x9000_0000, PA, 0x1000, MemoryPermission::READ_WRITE);
        let cached = device.cache_pages(DADDR, 0x1000);
        process.map(0xa000_0000, PA, 0x1000, MemoryPermission::READ_WRITE);
        process.write(0xa000_0000, 1u32).unwrap();
        assert_eq!(
            writes(&device),
            [(DADDR, 0x1000)],
            "an alias mapped after caching"
        );
        process.write(0x9000_0000, 2u32).unwrap();
        process.write(VA, 3u32).unwrap();
        assert!(writes(&device).is_empty(), "one write untracks every alias");

        drop(cached);
        let _again = device.cache_pages(DADDR, 0x1000);
        process.write(0x9000_0004, 4u32).unwrap();
        assert_eq!(
            writes(&device),
            [(DADDR, 0x1000)],
            "an alias write after the cache uploaded again"
        );
        process.unmap(0xa000_0000, 0x1000);
        process.unmap(0x9000_0000, 0x1000);
    }

    #[test]
    fn a_write_reports_the_physical_page_it_wrote_even_after_a_remap() {
        let (process, device) = spaces();
        let _cached = device.cache_pages(DADDR, 0x1000);
        let (ptr, _) = process.translate(VA).unwrap();
        process.unmap(VA, 0x1000);
        process.map(VA, 0x20_0000, 0x1000, MemoryPermission::READ_WRITE);
        // SAFETY: the DRAM page behind ptr stays valid after its mapping is gone.
        unsafe { ptr.write(7) };
        process.report_write(ptr);
        assert_eq!(writes(&device), [(DADDR, 0x1000)]);
        process.write(VA, 1u8).unwrap();
        assert!(
            writes(&device).is_empty(),
            "the page mapped there now is not cached"
        );
    }

    #[test]
    fn protect_keeps_pages_tracked() {
        let (process, device) = spaces();
        let _cached = device.cache_pages(DADDR, 0x1000);
        process.protect(VA, 0x1000, MemoryPermission::READ);
        process.protect(VA, 0x1000, MemoryPermission::READ_WRITE);
        process.guest_write(VA, 1u32).unwrap();
        assert_eq!(writes(&device), [(DADDR, 0x1000)]);
    }

    #[test]
    fn guards_outlive_device_unmaps() {
        let (process, device) = spaces();
        let cached = device.cache_pages(DADDR, 0x2000);
        device.unmap(DADDR, 0x1000);
        process.write(VA, 1u32).unwrap();
        process.write(VA + 0x1000, 1u32).unwrap();
        assert_eq!(
            writes(&device),
            [(DADDR + 0x1000, 0x1000)],
            "a page no device address maps is left out"
        );
        drop(cached);
        let _again = device.cache_pages(DADDR + 0x1000, 0x1000);
        process.write(VA + 0x1000, 2u32).unwrap();
        assert_eq!(writes(&device), [(DADDR + 0x1000, 0x1000)]);
    }

    #[test]
    fn a_cache_that_uploads_while_the_cpu_writes_ends_with_the_last_bytes() {
        let (process, device) = spaces();
        let device = Arc::new(device);
        let done = Arc::new(AtomicBool::new(false));
        let last = Arc::new(AtomicU64::new(0));
        let writer = {
            let (process, done, last) = (process.clone(), done.clone(), last.clone());
            std::thread::spawn(move || {
                for value in 1..=200_000u64 {
                    process.write(VA + 0x2000, value).unwrap();
                    last.store(value, Ordering::Relaxed);
                }
                done.store(true, Ordering::Release);
            })
        };
        let mut cached = device.cache_pages(DADDR + 0x2000, 0x1000);
        let mut copy: u64 = device.read(DADDR + 0x2000).unwrap();
        loop {
            let finished = done.load(Ordering::Acquire);
            if device.has_cpu_writes() {
                writes(&device);
                drop(cached);
                cached = device.cache_pages(DADDR + 0x2000, 0x1000);
                copy = device.read(DADDR + 0x2000).unwrap();
            }
            if finished && !device.has_cpu_writes() {
                break;
            }
        }
        writer.join().unwrap();
        drop(cached);
        assert_eq!(
            copy,
            last.load(Ordering::Relaxed),
            "the copy missed a write"
        );
    }
}
