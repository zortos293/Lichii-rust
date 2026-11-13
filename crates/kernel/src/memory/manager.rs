//! Physical page allocation from the DRAM pools.

use std::collections::BTreeMap;
use std::sync::Arc;

use lichii_common::{ConsoleGeneration, HleResult, PAGE_SIZE, PAddr, align_up};
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

    fn give(&mut self, mut start: PAddr, mut size: u64) {
        if let Some((&prev, &prev_size)) = self.ranges.range(..start).next_back()
            && prev + prev_size == start
        {
            self.ranges.remove(&prev);
            start = prev;
            size += prev_size;
        }
        if let Some(&next_size) = self.ranges.get(&(start + size)) {
            self.ranges.remove(&(start + size));
            size += next_size;
        }
        self.ranges.insert(start, size);
    }
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
                free: Mutex::new(FreeList { ranges, free: size }),
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
        for &(start, run) in &runs {
            list.take(start, run);
        }
        drop(list);
        self.commit(&runs)?;
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
        drop(list);
        self.commit(&[(start, size)])?;
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

    /// Returns pages to their pool, zeroed.
    pub fn free(&self, start: PAddr, size: u64) {
        if size == 0 {
            return;
        }
        self.physical.discard(start, size);
        let pool = self
            .pools
            .iter()
            .find(|p| start >= p.start && start + size <= p.start + p.size)
            .unwrap_or_else(|| panic!("freeing {start:#x}+{size:#x} outside every pool"));
        let mut list = pool.free.lock();
        list.give(start, size);
        list.free += size;
    }

    pub fn free_group(&self, group: &PageGroup) {
        for &(start, size) in &group.runs {
            self.free(start, size);
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
