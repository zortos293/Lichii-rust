//! Fastmem placement cost: mapping and unmapping guest memory the way `svcMapPhysicalMemory`
//! and `svcUnmapPhysicalMemory` do, one 4 KiB page per call or one call per physical run.
//!
//! `cargo bench -p lichii-memory --bench placement`; `LICHII_BENCH_BLOCKS` sets how many 2 MiB
//! blocks each pattern maps (default 64, zero is ignored).

use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_memory::{AddressSpace, MemoryPermission, PhysicalMemory};

const BLOCK: u64 = 2 << 20;
const PAGE: u64 = 0x1000;
const VA: u64 = 0x10_0000_0000;

struct Times {
    map: Duration,
    touch: Duration,
    unmap: Duration,
}

fn run(space: &AddressSpace, blocks: u64, per_page: bool) -> Times {
    let base = space.fastmem().expect("fastmem").base();
    let start = Instant::now();
    for block in 0..blocks {
        let (va, pa) = (VA + block * BLOCK, block * BLOCK);
        if per_page {
            for page in (0..BLOCK).step_by(PAGE as usize) {
                space.map(va + page, pa + page, PAGE, MemoryPermission::READ_WRITE);
            }
        } else {
            space.map(va, pa, BLOCK, MemoryPermission::READ_WRITE);
        }
    }
    let map = start.elapsed();
    let start = Instant::now();
    for page in (0..blocks * BLOCK).step_by(PAGE as usize) {
        // SAFETY: the page was just mapped read-write in the fastmem arena.
        unsafe { base.add((VA + page) as usize).write_volatile(1) };
    }
    let touch = start.elapsed();
    let start = Instant::now();
    for block in 0..blocks {
        space.unmap(VA + block * BLOCK, BLOCK);
    }
    Times {
        map,
        touch,
        unmap: start.elapsed(),
    }
}

fn main() {
    let blocks = std::env::var("LICHII_BENCH_BLOCKS")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|&n: &u64| n > 0)
        .unwrap_or(64);
    let physical = Arc::new(PhysicalMemory::new(blocks * BLOCK).expect("backing"));
    let space = AddressSpace::with_fastmem(physical, 39).expect("fastmem arena");
    run(&space, blocks, false);
    println!(
        "{:<12} {:>14} {:>14} {:>14} {:>14}",
        "pattern", "map us/2MiB", "touch us/2MiB", "unmap us/2MiB", "total us/2MiB"
    );
    for (name, per_page) in [("per page", true), ("per run", false)] {
        let t = run(&space, blocks, per_page);
        let us = |d: Duration| d.as_secs_f64() * 1e6 / blocks as f64;
        println!(
            "{name:<12} {:>14.1} {:>14.1} {:>14.1} {:>14.1}",
            us(t.map),
            us(t.touch),
            us(t.unmap),
            us(t.map + t.touch + t.unmap)
        );
    }
}
