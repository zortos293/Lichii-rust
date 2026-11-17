//! [`ShaderCache::invalidate`]'s chunk reject against the segment scan it guards.

use std::collections::BTreeSet;
use std::sync::Arc;

use lichii_gpu::gmmu::BigPageSize;
use lichii_memory::{DeviceAddressSpace, PhysicalMemory};

use super::*;
use crate::chunk_mask::chunks_of;

impl ShaderCache {
    /// Registers the `len` program bytes at `start` as a translation would.
    pub(crate) fn track_program(&mut self, gmmu: &Gmmu, start: GpuVAddr, len: u64) {
        let key = gmmu.translate(start).expect("a mapped program");
        self.track(gmmu, start, len, key);
    }
}

/// The program of each segment the scan [`ShaderCache::invalidate`] ran before the chunk
/// reject marks for a write of `[daddr, +size)`, in scan order.
fn scanned(cache: &ShaderCache, daddr: DAddr, size: u64) -> Vec<DAddr> {
    let end = daddr.saturating_add(size);
    let from = (daddr.saturating_sub(cache.longest), 0);
    cache
        .segments
        .range(from..(end, 0))
        .filter(|&(&(start, key), &len)| start + len > daddr && cache.programs.contains_key(&key))
        .map(|(&(_, key), _)| key)
        .collect()
}

/// A seeded random stream of program registrations, re-registrations over remapped pages (which
/// drop segments) and guest writes: after every write the programs made stale and the generation
/// count are the scan's, the stale set is a scan over every program's own segments, and the
/// chunk mask holds exactly the chunks of the registered segments.
#[test]
fn writes_mark_the_programs_the_unguarded_scan_marks() {
    const DADDR: DAddr = 0x1_0000_0000;
    const DRAM: u64 = 16 << 20;
    const SPAN: u64 = 128 << 20;
    const GPU_VA: GpuVAddr = 0x20_0000_0000;
    let physical = Arc::new(PhysicalMemory::new(DRAM).unwrap());
    physical.commit(0, DRAM).unwrap();
    let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
    for alias in 0..SPAN / DRAM {
        memory.map(DADDR + alias * DRAM, 0, DRAM);
    }
    let gmmu = Gmmu::new(memory, BigPageSize::Size64K);
    gmmu.map(GPU_VA, DADDR, SPAN, 0, false).unwrap();
    let mut cache = ShaderCache::new(Profile::default(), HostTranslateInfo::default());
    let mut seed = 0x0123_4567_89ab_cdefu64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut registered: Vec<(GpuVAddr, u64)> = Vec::new();
    let mut hits = 0;
    for step in 0..6000 {
        match next() % 10 {
            0..=2 => {
                let start = GPU_VA + ((next() % (SPAN - (64 << 10))) & !0xf);
                let len = match next() % 4 {
                    0 => 0x80 + (next() % 0x400),
                    1 => 0x1000 - (start & 0xfff) + 0x40,
                    2 => (64 << 10) - (start & 0xffff) + 0x100,
                    _ => next() % 0x8000 + 1,
                };
                cache.track_program(&gmmu, start, len);
                registered.push((start, len));
            }
            3 => {
                let page = GPU_VA + ((next() % (SPAN >> 12)) << 12);
                let target = DADDR + ((next() % (SPAN >> 12)) << 12);
                gmmu.map(page, target, 0x1000, 0, false).unwrap();
            }
            4 if !registered.is_empty() => {
                let (start, len) = registered[(next() % registered.len() as u64) as usize];
                let len = if next() % 2 == 0 { len } else { len / 2 + 1 };
                cache.track_program(&gmmu, start, len);
            }
            _ => {
                let daddr = match next() % 4 {
                    0 => DADDR + ((next() % (SPAN >> 16)) << 16) - (next() % 0x40),
                    1 if !registered.is_empty() => {
                        let (start, len) = registered[(next() % registered.len() as u64) as usize];
                        gmmu.translate(start + next() % len.max(1)).unwrap_or(DADDR)
                    }
                    _ => DADDR + next() % SPAN,
                };
                let size = match next() % 6 {
                    0 => 0,
                    1 => next() % (4 << 20),
                    _ => 64 + next() % 193,
                };
                for program in cache.programs.values_mut() {
                    program.stale = false;
                }
                let expected = scanned(&cache, daddr, size);
                let end = daddr.saturating_add(size);
                let brute: BTreeSet<DAddr> = cache
                    .programs
                    .iter()
                    .filter(|(_, p)| {
                        p.segments
                            .iter()
                            .any(|&(d, len)| d < end && d + len > daddr)
                    })
                    .map(|(&key, _)| key)
                    .collect();
                let generation = cache.generation;
                cache.invalidate(daddr, size);
                let stale: BTreeSet<DAddr> = cache
                    .programs
                    .iter()
                    .filter(|(_, p)| p.stale)
                    .map(|(&key, _)| key)
                    .collect();
                let at = format!("step {step}: [{daddr:#x}, +{size:#x})");
                assert_eq!(stale, expected.iter().copied().collect(), "{at}");
                assert_eq!(stale, brute, "{at}");
                assert_eq!(cache.generation - generation, expected.len() as u64, "{at}");
                assert!(
                    stale
                        .iter()
                        .all(|key| cache.programs[key].tracked.is_empty()),
                    "{at}"
                );
                hits += usize::from(!stale.is_empty());
            }
        }
        let occupied: BTreeSet<DAddr> = cache
            .segments
            .iter()
            .flat_map(|(&(daddr, _), &len)| chunks_of(daddr, len))
            .collect();
        assert_eq!(cache.chunks.marked(), occupied, "step {step}");
    }
    assert!(hits > 200, "the stream hits programs: {hits}");
}
