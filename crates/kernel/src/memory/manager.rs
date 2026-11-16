//! Physical page allocation from the DRAM pools.

use std::collections::BTreeMap;
use std::sync::Arc;

use lichii_common::{ConsoleGeneration, HleResult, PAGE_SIZE, PAddr, align_down, align_up};
use lichii_memory::PhysicalMemory;
use parking_lot::Mutex;

use crate::result;

const MIB: u64 = 1 << 20;

/// DRAM Switch 2 consoles get beyond their own for the memory the graphics server keeps itself
/// (nx2:gpu 214 GPU-only regions, NX2_NOTES row 57). It joins the non-secure system pool, so the
/// application pool a title sizes its heap from stays whole.
pub const NX2_GRAPHICS_SERVER_MEMORY: u64 = 2 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pool {
    Application = 0,
    Applet = 1,
    System = 2,
    SystemNonSecure = 3,
}

impl Pool {
    pub const ALL: [Pool; 4] = [
        Pool::Application,
        Pool::Applet,
        Pool::System,
        Pool::SystemNonSecure,
    ];

    pub fn from_raw(raw: u64) -> Option<Self> {
        Self::ALL.get(raw as usize).copied()
    }
}

/// How DRAM is split into pools, in bytes. The system pool takes what is left.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolLayout {
    pub application: u64,
    pub applet: u64,
    pub system_non_secure: u64,
}

impl PoolLayout {
    /// The DRAM to emulate for a console with `dram` bytes, and its pools: Horizon's split for
    /// the console's arrangement (4/6/8/12 GiB), proportional for any other size. Switch 2 adds
    /// [`NX2_GRAPHICS_SERVER_MEMORY`] to the non-secure system pool.
    pub fn for_console(dram: u64, console: ConsoleGeneration) -> (u64, Self) {
        let mut layout = Self::for_dram(dram);
        if console != ConsoleGeneration::Switch2 {
            return (dram, layout);
        }
        layout.system_non_secure += NX2_GRAPHICS_SERVER_MEMORY;
        (dram + NX2_GRAPHICS_SERVER_MEMORY, layout)
    }

    fn for_dram(dram: u64) -> Self {
        let application = match dram >> 30 {
            4 => 3285 * MIB,
            6 => 4916 * MIB,
            8 => 6547 * MIB,
            10 => 8178 * MIB,
            12 => 9809 * MIB,
            _ => align_up(dram / 4 * 3, 2 * MIB).min(dram),
        };
        let rest = dram - application;
        let applet = (507 * MIB).min(rest / 2);
        let system_non_secure = align_up(0x2c0_4000, 2 * MIB).min((rest - applet) / 2);
        Self {
            application,
            applet,
            system_non_secure,
        }
    }
}

struct FreeList {
    ranges: BTreeMap<PAddr, u64>,
    free: u64,
    /// Freed ranges that still hold their old contents, freed since the last
    /// [`MemoryManager::trim`]. They are zeroed when allocated again, so memory nobody reuses is
    /// never written.
    dirty: BTreeMap<PAddr, u64>,
    /// Dirty ranges freed before the last trim. The next trim gives their host memory back.
    aged: BTreeMap<PAddr, u64>,
}

impl FreeList {
    fn take(&mut self, start: PAddr, size: u64) {
        let (&range_start, &range_size) = self
            .ranges
            .range(..=start)
            .next_back()
            .expect("taking from a free range");
        debug_assert!(start + size <= range_start + range_size);
        self.ranges.remove(&range_start);
        if start > range_start {
            self.ranges.insert(range_start, start - range_start);
        }
        let end = start + size;
        if end < range_start + range_size {
            self.ranges.insert(end, range_start + range_size - end);
        }
        self.free -= size;
    }

    fn give(&mut self, start: PAddr, size: u64) {
        insert_coalescing(&mut self.ranges, start, size);
        insert_coalescing(&mut self.dirty, start, size);
        self.free += size;
    }

    /// Removes the dirty parts of `[start, start + size)` and appends them to `out`.
    fn take_dirty(&mut self, start: PAddr, size: u64, out: &mut Vec<(PAddr, u64)>) {
        cut(&mut self.dirty, start, size, |run| out.push(run));
        cut(&mut self.aged, start, size, |run| out.push(run));
    }
}

/// Removes `[start, start + size)` from `ranges`, passing each part it held to `found`.
fn cut(
    ranges: &mut BTreeMap<PAddr, u64>,
    start: PAddr,
    size: u64,
    mut found: impl FnMut((PAddr, u64)),
) {
    let end = start + size;
    while let Some((&run, &len)) = ranges
        .range(..end)
        .next_back()
        .filter(|&(&run, &len)| run + len > start)
    {
        ranges.remove(&run);
        if run < start {
            ranges.insert(run, start - run);
        }
        if run + len > end {
            ranges.insert(end, run + len - end);
        }
        let from = run.max(start);
        found((from, (run + len).min(end) - from));
    }
}

fn insert_coalescing(ranges: &mut BTreeMap<PAddr, u64>, mut start: PAddr, mut size: u64) {
    if let Some((&prev, &prev_size)) = ranges.range(..start).next_back()
        && prev + prev_size == start
    {
        ranges.remove(&prev);
        start = prev;
        size += prev_size;
    }
    if let Some(next_size) = ranges.remove(&(start + size)) {
        size += next_size;
    }
    ranges.insert(start, size);
}

struct PoolAllocator {
    start: PAddr,
    size: u64,
    free: Mutex<FreeList>,
}

/// A set of physical page runs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PageGroup {
    pub runs: Vec<(PAddr, u64)>,
}

impl PageGroup {
    pub fn size(&self) -> u64 {
        self.runs.iter().map(|(_, size)| size).sum()
    }
}

/// Hands out zeroed, committed DRAM pages from the pools.
pub struct MemoryManager {
    physical: Arc<PhysicalMemory>,
    pools: [PoolAllocator; 4],
}

impl MemoryManager {
    pub fn new(physical: Arc<PhysicalMemory>, layout: PoolLayout) -> Self {
        let total = physical.size();
        let system = total - layout.application - layout.applet - layout.system_non_secure;
        let sizes = [
            layout.application,
            layout.applet,
            system,
            layout.system_non_secure,
        ];
        let mut start = 0;
        let pools = sizes.map(|size| {
            let mut ranges = BTreeMap::new();
            if size > 0 {
                ranges.insert(start, size);
            }
            let pool = PoolAllocator {
                start,
                size,
                free: Mutex::new(FreeList {
                    ranges,
                    free: size,
                    dirty: BTreeMap::new(),
                    aged: BTreeMap::new(),
                }),
            };
            start += size;
            pool
        });
        Self { physical, pools }
    }

    pub fn physical(&self) -> &Arc<PhysicalMemory> {
        &self.physical
    }

    pub fn size(&self, pool: Pool) -> u64 {
        self.pools[pool as usize].size
    }

    pub fn free_size(&self, pool: Pool) -> u64 {
        self.pools[pool as usize].free.lock().free
    }

    /// Allocates `size` bytes, contiguous when possible.
    pub fn allocate(&self, pool: Pool, size: u64) -> HleResult<PageGroup> {
        assert_eq!(size % PAGE_SIZE, 0);
        let mut list = self.pools[pool as usize].free.lock();
        if list.free < size {
            return Err(result::OUT_OF_MEMORY);
        }
        let mut runs = Vec::new();
        if let Some((&start, _)) = list.ranges.iter().find(|(_, s)| **s >= size) {
            runs.push((start, size));
        } else {
            let mut left = size;
            for (&start, &run) in &list.ranges {
                let take = run.min(left);
                runs.push((start, take));
                left -= take;
                if left == 0 {
                    break;
                }
            }
        }
        let mut dirty = Vec::new();
        for &(start, run) in &runs {
            list.take(start, run);
            list.take_dirty(start, run, &mut dirty);
        }
        drop(list);
        self.commit(&runs)?;
        self.zero(&dirty);
        Ok(PageGroup { runs })
    }

    /// Allocates `size` physically contiguous bytes aligned to `align`.
    pub fn allocate_contiguous(&self, pool: Pool, size: u64, align: u64) -> HleResult<PAddr> {
        assert_eq!(size % PAGE_SIZE, 0);
        let align = align.max(PAGE_SIZE);
        let mut list = self.pools[pool as usize].free.lock();
        let found = list.ranges.iter().find_map(|(&start, &run)| {
            let aligned = align_up(start, align);
            (aligned + size <= start + run).then_some(aligned)
        });
        let start = found.ok_or(result::OUT_OF_MEMORY)?;
        list.take(start, size);
        let mut dirty = Vec::new();
        list.take_dirty(start, size, &mut dirty);
        drop(list);
        self.commit(&[(start, size)])?;
        self.zero(&dirty);
        Ok(start)
    }

    fn commit(&self, runs: &[(PAddr, u64)]) -> HleResult<()> {
        for &(start, size) in runs {
            if let Err(err) = self.physical.commit(start, size) {
                tracing::error!("{err}");
                for &(start, size) in runs {
                    self.free(start, size);
                }
                return Err(result::OUT_OF_MEMORY);
            }
        }
        Ok(())
    }

    fn zero(&self, runs: &[(PAddr, u64)]) {
        for &(start, size) in runs {
            self.physical.discard(start, size);
        }
    }

    /// Returns pages to their pool. They are zeroed when allocated again, and their host memory
    /// goes back to the OS at the second [`Self::trim`] that finds them still free.
    pub fn free(&self, start: PAddr, size: u64) {
        if size == 0 {
            return;
        }
        let pool = self
            .pools
            .iter()
            .find(|p| start >= p.start && start + size <= p.start + p.size)
            .unwrap_or_else(|| panic!("freeing {start:#x}+{size:#x} outside every pool"));
        pool.free.lock().give(start, size);
    }

    pub fn free_group(&self, group: &PageGroup) {
        for &(start, size) in &group.runs {
            self.free(start, size);
        }
    }

    /// Gives the host memory of pages that stayed free since the previous trim back to the OS,
    /// in whole release granules. The kernel trims about once a second, so pages a title frees
    /// and soon takes again (MKW maps and unmaps physical memory all the time) keep their host
    /// memory, and pages that stay free stop costing any.
    pub fn trim(&self) {
        let granule = self.physical.release_granule();
        for pool in &self.pools {
            let mut list = pool.free.lock();
            let mut kept = BTreeMap::new();
            for (start, size) in std::mem::take(&mut list.aged) {
                let (&run, &run_size) = list
                    .ranges
                    .range(..=start)
                    .next_back()
                    .expect("dirty pages are free");
                let from = align_down(start, granule).max(run);
                let to = align_up(start + size, granule).min(run + run_size);
                let released = self.physical.release(from, to - from);
                let end = start + size;
                if start < released.start {
                    insert_coalescing(&mut kept, start, end.min(released.start) - start);
                }
                if released.end < end {
                    let at = start.max(released.end);
                    insert_coalescing(&mut kept, at, end - at);
                }
                cut(
                    &mut list.dirty,
                    released.start,
                    released.end - released.start,
                    |_| {},
                );
            }
            for (start, size) in std::mem::take(&mut list.dirty) {
                insert_coalescing(&mut kept, start, size);
            }
            list.aged = kept;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> MemoryManager {
        let physical = Arc::new(PhysicalMemory::new(64 * MIB).unwrap());
        MemoryManager::new(
            physical,
            PoolLayout {
                application: 32 * MIB,
                applet: 8 * MIB,
                system_non_secure: 8 * MIB,
            },
        )
    }

    #[test]
    fn allocations_coalesce_when_freed() {
        let mm = manager();
        let a = mm.allocate(Pool::Application, 0x3000).unwrap();
        let b = mm.allocate(Pool::Application, 0x1000).unwrap();
        assert_eq!(a.runs, vec![(0, 0x3000)]);
        assert_eq!(b.runs, vec![(0x3000, 0x1000)]);
        mm.free_group(&a);
        let c = mm.allocate(Pool::Application, 0x4000).unwrap();
        assert_eq!(c.runs, vec![(0x4000, 0x4000)]);
        mm.free_group(&b);
        mm.free_group(&c);
        assert_eq!(mm.free_size(Pool::Application), 32 * MIB);
        let d = mm.allocate(Pool::Application, 32 * MIB).unwrap();
        assert_eq!(d.runs, vec![(0, 32 * MIB)]);
    }

    #[test]
    fn fragmented_allocation_spans_runs_and_pools_are_separate() {
        let mm = manager();
        let a = mm.allocate(Pool::Application, 0x1000).unwrap();
        let _b = mm.allocate(Pool::Application, 0x1000).unwrap();
        mm.free_group(&a);
        let big = mm.allocate(Pool::Application, 32 * MIB - 0x1000).unwrap();
        assert_eq!(big.runs.len(), 2);
        assert_eq!(big.size(), 32 * MIB - 0x1000);
        assert_eq!(
            mm.allocate(Pool::Application, 0x1000),
            Err(result::OUT_OF_MEMORY)
        );
        let sys = mm
            .allocate_contiguous(Pool::System, 0x2000, 0x10000)
            .unwrap();
        assert_eq!(sys, 40 * MIB);
    }

    fn fill(mm: &MemoryManager, pa: PAddr, size: u64, byte: u8) {
        // SAFETY: the range is allocated and committed DRAM inside the backing.
        unsafe { std::ptr::write_bytes(mm.physical().pointer(pa), byte, size as usize) };
    }

    fn byte_at(mm: &MemoryManager, pa: PAddr) -> u8 {
        // SAFETY: pa is committed DRAM inside the backing.
        unsafe { *mm.physical().pointer(pa) }
    }

    #[test]
    fn freed_pages_are_zeroed_only_when_allocated_again() {
        let mm = manager();
        let a = mm.allocate(Pool::Application, 0x3000).unwrap();
        fill(&mm, 0, 0x3000, 0xaa);
        mm.free_group(&a);
        assert_eq!(byte_at(&mm, 0x2fff), 0xaa);
        assert_eq!(
            mm.allocate(Pool::Application, 0x1000).unwrap().runs,
            vec![(0, 0x1000)]
        );
        assert_eq!(byte_at(&mm, 0), 0);
        assert_eq!(byte_at(&mm, 0x1000), 0xaa);
        let sys = mm
            .allocate_contiguous(Pool::Application, 0x1000, 0x2000)
            .unwrap();
        assert_eq!(sys, 0x2000);
        assert_eq!(byte_at(&mm, 0x2fff), 0);
        assert_eq!(byte_at(&mm, 0x1000), 0xaa);
        assert_eq!(
            mm.allocate(Pool::Application, 0x1000).unwrap().runs,
            vec![(0x1000, 0x1000)]
        );
        assert_eq!(byte_at(&mm, 0x1fff), 0);
    }

    #[test]
    fn host_memory_goes_back_once_pages_stay_free_for_a_trim() {
        let mm = manager();
        let host = || mm.physical().host_bytes();
        assert_eq!(host(), 0, "nothing is backed before the first allocation");

        let a = mm.allocate(Pool::Application, 8 * MIB).unwrap();
        fill(&mm, 0, 8 * MIB, 0x55);
        assert!(host() >= 8 * MIB, "touched pages are backed: {:#x}", host());
        let b = mm.allocate(Pool::Applet, PAGE_SIZE).unwrap();
        fill(&mm, b.runs[0].0, PAGE_SIZE, 0x66);

        mm.trim();
        mm.free_group(&a);
        mm.trim();
        assert!(
            host() >= 8 * MIB,
            "pages freed since the last trim keep their host memory"
        );
        let again = mm.allocate(Pool::Application, 4 * MIB).unwrap();
        assert_eq!(byte_at(&mm, 0), 0, "reused pages are zeroed");
        mm.free_group(&again);

        mm.trim();
        mm.trim();
        assert!(
            host() <= 2 * MIB,
            "pages free for a whole trim go back: {:#x}",
            host()
        );
        assert_eq!(byte_at(&mm, b.runs[0].0), 0x66);
        mm.free_group(&b);
        mm.trim();
        mm.trim();
        assert_eq!(host(), 0, "freeing everything gives everything back");

        let c = mm.allocate(Pool::Application, 8 * MIB).unwrap();
        assert_eq!(c.runs, vec![(0, 8 * MIB)]);
        assert!(
            (0..8 * MIB)
                .step_by(PAGE_SIZE as usize)
                .all(|pa| byte_at(&mm, pa) == 0)
        );
    }

    #[test]
    fn pages_freed_beside_live_ones_are_zeroed_when_reused_and_released_once_free_whole() {
        let mm = manager();
        let granule = mm.physical().release_granule().max(2 * MIB);
        let keep = mm.allocate(Pool::Application, PAGE_SIZE).unwrap();
        let rest = mm.allocate(Pool::Application, granule - PAGE_SIZE).unwrap();
        fill(&mm, 0, granule, 0x77);
        mm.free_group(&rest);
        mm.trim();
        mm.trim();
        assert_eq!(byte_at(&mm, 0), 0x77, "live pages keep their contents");
        let again = mm.allocate(Pool::Application, granule - PAGE_SIZE).unwrap();
        assert_eq!(again.runs, rest.runs);
        assert!(
            (PAGE_SIZE..granule)
                .step_by(PAGE_SIZE as usize)
                .all(|pa| byte_at(&mm, pa) == 0)
        );
        mm.free_group(&again);
        mm.trim();
        mm.free_group(&keep);
        mm.trim();
        mm.trim();
        assert_eq!(mm.physical().host_bytes(), 0);
    }

    #[test]
    fn horizon_pool_sizes() {
        let (dram, layout) = PoolLayout::for_console(4 << 30, ConsoleGeneration::Switch1);
        assert_eq!(dram, 4 << 30);
        assert_eq!(layout.application, 3285 * MIB);
        assert_eq!(
            layout.system_non_secure,
            0x2c0_4000_u64.next_multiple_of(2 * MIB)
        );
    }

    #[test]
    fn switch2_graphics_server_memory_is_outside_the_application_pool() {
        let (dram, layout) = PoolLayout::for_console(12 << 30, ConsoleGeneration::Switch2);
        let base = 0x2c0_4000_u64.next_multiple_of(2 * MIB);
        assert_eq!(dram, 14 << 30);
        assert_eq!(layout.application, 9809 * MIB);
        assert_eq!(layout.applet, 507 * MIB);
        assert_eq!(layout.system_non_secure, base + NX2_GRAPHICS_SERVER_MEMORY);
        let system = dram - layout.application - layout.applet - layout.system_non_secure;
        assert_eq!(system, (12 << 30) - 9809 * MIB - 507 * MIB - base);
    }
}
