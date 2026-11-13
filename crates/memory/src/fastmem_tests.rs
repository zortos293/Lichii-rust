use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{
    AddressSpace, DeviceAddressSpace, FastmemError, HostFault, MemoryPermission, PhysicalMemory,
};

fn fastmem_space() -> Option<AddressSpace> {
    let physical = Arc::new(PhysicalMemory::new(1 << 26).unwrap());
    match AddressSpace::with_fastmem(physical, 39) {
        Ok(space) => Some(space),
        Err(FastmemError::NoFaultHandler | FastmemError::NoViews) => None,
        Err(err) => panic!("{err}"),
    }
}

fn host_read<T: Copy>(space: &AddressSpace, va: u64) -> T {
    let base = space.fastmem().unwrap().base();
    // SAFETY: the test only reads pages it mapped readable.
    unsafe { std::ptr::read_volatile(base.add(va as usize).cast::<T>()) }
}

#[test]
fn fastmem_mirrors_guest_mappings() {
    let Some(space) = fastmem_space() else {
        return;
    };
    space.map(0x8000_0000, 0x20_0000, 0x4000, MemoryPermission::READ_WRITE);
    space.map(0x9000_0000, 0x20_1000, 0x1000, MemoryPermission::READ);
    space.write(0x8000_1008, 0x1234_5678u32).unwrap();
    assert_eq!(host_read::<u32>(&space, 0x8000_1008), 0x1234_5678);
    assert_eq!(host_read::<u32>(&space, 0x9000_0008), 0x1234_5678);

    space.unmap(0x8000_1000, 0x1000);
    space.map(0x8000_1000, 0x30_0000, 0x1000, MemoryPermission::READ_WRITE);
    space.write(0x8000_1000, 0xaau8).unwrap();
    assert_eq!(host_read::<u8>(&space, 0x8000_1000), 0xaa);
    space.write(0x8000_3000, 0x55u8).unwrap();
    assert_eq!(host_read::<u8>(&space, 0x8000_3000), 0x55);

    space.protect(0x8000_0000, 0x4000, MemoryPermission::READ);
    assert_eq!(host_read::<u8>(&space, 0x8000_3000), 0x55);
}

#[test]
fn protections_lower_and_raise_again_within_and_across_views() {
    let Some(space) = fastmem_space() else {
        return;
    };
    let host_write = |va: u64, value: u8| {
        let base = space.fastmem().unwrap().base();
        // SAFETY: the test only writes pages it made writable.
        unsafe { std::ptr::write_volatile(base.add(va as usize), value) }
    };
    space.map(
        0x8000_0000,
        0x40_0000,
        0x1_0000,
        MemoryPermission::READ_WRITE,
    );
    space.map(0x8001_0000, 0x50_0000, 0x1_0000, MemoryPermission::READ);

    space.protect(0x8000_3000, 0x2000, MemoryPermission::NONE);
    space.protect(0x8000_3000, 0x2000, MemoryPermission::READ_WRITE);
    host_write(0x8000_3000, 1);

    space.protect(0x8000_8000, 0x1_0000, MemoryPermission::NONE);
    space.protect(0x8000_c000, 0x8000, MemoryPermission::READ);
    space.protect(0x8000_0000, 0x2_0000, MemoryPermission::READ_WRITE);
    host_write(0x8000_8000, 2);
    host_write(0x8001_f000, 3);

    assert_eq!(space.read::<u8>(0x8000_3000).unwrap(), 1);
    assert_eq!(space.read::<u8>(0x8000_8000).unwrap(), 2);
    assert_eq!(space.read::<u8>(0x8001_f000).unwrap(), 3);
    assert_eq!(host_read::<u8>(&space, 0x8001_f000), 3);

    space.unmap(0x8000_4000, 0x1_8000);
    space.map(0x8000_4000, 0x60_0000, 0x1_8000, MemoryPermission::READ);
    space.protect(0x8000_0000, 0x2_0000, MemoryPermission::READ_WRITE);
    host_write(0x8000_4000, 4);
    assert_eq!(space.read::<u8>(0x8000_4000).unwrap(), 4);
    space.unmap(0x8000_0000, 0x2_0000);
}

#[test]
fn physical_memory_is_committed_lazily() {
    let physical = PhysicalMemory::new(12 << 30).unwrap();
    physical.commit(0x1000_0000, 0x10_0000).unwrap();
    // SAFETY: the page was just committed.
    unsafe { physical.pointer(0x1000_0000).write(1) };
    physical.discard(0x1000_0000, 0x1000);
    // SAFETY: as above.
    assert_eq!(unsafe { physical.pointer(0x1000_0000).read() }, 0);
}

#[cfg(target_arch = "x86_64")]
mod faults {
    use super::*;

    static LAST_FAULT: AtomicU64 = AtomicU64::new(0);

    fn skip_load(data: usize, guest_addr: u64, fault: &mut HostFault) -> bool {
        LAST_FAULT.store(guest_addr, Ordering::SeqCst);
        skip_load_quietly(data, guest_addr, fault)
    }

    /// Like `skip_load`, without touching `LAST_FAULT`, which another test asserts on.
    fn skip_load_quietly(data: usize, _guest_addr: u64, fault: &mut HostFault) -> bool {
        fault.set_gpr(0, data as u64);
        fault.set_pc(fault.pc() + 3);
        true
    }

    fn load(base: *mut u8, va: u64) -> u64 {
        let value: u64;
        // SAFETY: `mov rax, [rdi]` (3 bytes); a fault on it is handled by skip_load, which
        // writes rax and steps over the instruction.
        unsafe {
            std::arch::asm!(
                ".byte 0x48, 0x8b, 0x07",
                in("rdi") base.add(va as usize),
                out("rax") value,
                options(nostack),
            );
        }
        value
    }

    #[test]
    fn unmapped_guest_page_reaches_the_fault_handler() {
        let Some(space) = fastmem_space() else {
            return;
        };
        let fastmem = space.fastmem().unwrap();
        fastmem.set_fault_handler(Some(skip_load), 0xdead);
        space.map(0x4000_0000, 0, 0x1000, MemoryPermission::READ_WRITE);
        space.write(0x4000_0000, 77u64).unwrap();

        assert_eq!(load(fastmem.base(), 0x4000_0000), 77);
        assert_eq!(load(fastmem.base(), 0x4000_1010), 0xdead);
        assert_eq!(LAST_FAULT.load(Ordering::SeqCst), 0x4000_1010);

        space.unmap(0x4000_0000, 0x1000);
        assert_eq!(load(fastmem.base(), 0x4000_0000), 0xdead);
        assert_eq!(LAST_FAULT.load(Ordering::SeqCst), 0x4000_0000);
    }

    #[test]
    fn a_new_arena_at_a_dropped_arenas_range_gets_its_own_faults() {
        if fastmem_space().is_none() {
            return;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let churn = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let physical = Arc::new(PhysicalMemory::new(1 << 26).unwrap());
                    drop(AddressSpace::with_fastmem(physical, 36));
                }
            })
        };
        for _ in 0..20_000 {
            let physical = Arc::new(PhysicalMemory::new(1 << 26).unwrap());
            let space = AddressSpace::with_fastmem(physical, 36).unwrap();
            let fastmem = space.fastmem().unwrap();
            fastmem.set_fault_handler(Some(skip_load_quietly), 0xdead);
            assert_eq!(load(fastmem.base(), 0x1000), 0xdead);
        }
        stop.store(true, Ordering::Relaxed);
        churn.join().unwrap();
    }

    fn count_store(data: usize, _guest_addr: u64, fault: &mut HostFault) -> bool {
        // SAFETY: `data` points at the test's counter, which outlives the arena.
        unsafe { &*(data as *const AtomicU64) }.fetch_add(1, Ordering::SeqCst);
        fault.set_pc(fault.pc() + 3);
        true
    }

    fn store(base: *mut u8, va: u64, value: u64) {
        // SAFETY: `mov [rdi], rsi` (3 bytes); a fault on it is handled by count_store, which
        // steps over the instruction.
        unsafe {
            std::arch::asm!(
                ".byte 0x48, 0x89, 0x37",
                in("rdi") base.add(va as usize),
                in("rsi") value,
                options(nostack),
            );
        }
    }

    fn tracked_space() -> Option<(Arc<AddressSpace>, DeviceAddressSpace)> {
        let space = Arc::new(fastmem_space()?);
        let device = DeviceAddressSpace::new(space.physical().clone(), 36);
        device.attach_process(space.clone());
        Some((space, device))
    }

    #[test]
    fn stores_to_cached_pages_fault_until_a_cpu_write_untracks_them() {
        let faults = AtomicU64::new(0);
        let Some((space, device)) = tracked_space() else {
            return;
        };
        let fastmem = space.fastmem().unwrap();
        fastmem.set_fault_handler(Some(count_store), &faults as *const AtomicU64 as usize);
        let base = fastmem.base();
        space.map(0x4000_0000, 0x10_0000, 0x2000, MemoryPermission::READ_WRITE);
        device
            .map_process(0x1_0000_0000, &space, 0x4000_0000, 0x2000)
            .unwrap();

        let cached = device.cache_pages(0x1_0000_0000, 0x1000);
        store(base, 0x4000_0008, 1);
        assert_eq!(
            faults.load(Ordering::SeqCst),
            1,
            "the cached page is write-protected"
        );
        assert_eq!(host_read::<u64>(&space, 0x4000_0008), 0, "reads stay fast");
        store(base, 0x4000_1008, 2);
        assert_eq!(
            faults.load(Ordering::SeqCst),
            1,
            "the page beside it is not cached"
        );

        space.guest_write(0x4000_0008, 3u64).unwrap();
        assert!(device.has_cpu_writes());
        store(base, 0x4000_0010, 4);
        assert_eq!(
            faults.load(Ordering::SeqCst),
            1,
            "the CPU write untracked the page"
        );

        let mut found = Vec::new();
        device.take_cpu_writes(|daddr, size| found.push((daddr, size)));
        assert_eq!(found, [(0x1_0000_0000, 0x1000)]);
        drop(cached);
        let _again = device.cache_pages(0x1_0000_0000, 0x1000);
        store(base, 0x4000_0018, 5);
        assert_eq!(
            faults.load(Ordering::SeqCst),
            2,
            "caching again protects the page again"
        );
    }

    #[test]
    fn tracked_pages_stay_protected_through_protect_and_new_aliases() {
        let faults = AtomicU64::new(0);
        let Some((space, device)) = tracked_space() else {
            return;
        };
        let fastmem = space.fastmem().unwrap();
        fastmem.set_fault_handler(Some(count_store), &faults as *const AtomicU64 as usize);
        let base = fastmem.base();
        space.map(0x4000_0000, 0x10_0000, 0x4000, MemoryPermission::READ_WRITE);
        device
            .map_process(0x1_0000_0000, &space, 0x4000_0000, 0x4000)
            .unwrap();
        let _cached = device.cache_pages(0x1_0000_1000, 0x1000);

        space.protect(0x4000_0000, 0x4000, MemoryPermission::READ);
        space.protect(0x4000_0000, 0x4000, MemoryPermission::READ_WRITE);
        store(base, 0x4000_1000, 1);
        assert_eq!(
            faults.load(Ordering::SeqCst),
            1,
            "protect keeps the page tracked"
        );
        store(base, 0x4000_2000, 2);
        store(base, 0x4000_0000, 3);
        assert_eq!(
            faults.load(Ordering::SeqCst),
            1,
            "the views around it are writable"
        );

        space.map(0x5000_0000, 0x10_1000, 0x1000, MemoryPermission::READ_WRITE);
        store(base, 0x5000_0000, 4);
        assert_eq!(
            faults.load(Ordering::SeqCst),
            2,
            "a new alias of the page is protected"
        );
        space.unmap(0x5000_0000, 0x1000);
    }

    #[test]
    fn many_scattered_pages_track_and_untrack() {
        let faults = AtomicU64::new(0);
        let Some((space, device)) = tracked_space() else {
            return;
        };
        let fastmem = space.fastmem().unwrap();
        fastmem.set_fault_handler(Some(count_store), &faults as *const AtomicU64 as usize);
        let base = fastmem.base();
        const PAGES: u64 = 0x2000;
        let va = 0x4000_0000;
        space.map(va, 0, PAGES * 0x1000, MemoryPermission::READ_WRITE);
        device
            .map_process(0x1_0000_0000, &space, va, PAGES * 0x1000)
            .unwrap();
        let page = |i: u64| 0x1_0000_0000 + i * 0x1000;

        let mut guards: Vec<_> = (0..PAGES)
            .step_by(3)
            .map(|i| (i, device.cache_pages(page(i), 0x1000)))
            .collect();
        space.protect(va + 0x10_0000, 0x40_0000, MemoryPermission::READ);
        space.protect(va, PAGES * 0x1000, MemoryPermission::READ_WRITE);
        guards.retain(|(i, _)| i % 2 == 0);
        for i in (0..PAGES).step_by(7) {
            guards.push((i, device.cache_pages(page(i), 0x1000)));
        }
        let tracked: std::collections::BTreeSet<u64> = guards.iter().map(|(i, _)| *i).collect();
        for i in 0..PAGES {
            let before = faults.load(Ordering::SeqCst);
            store(base, va + i * 0x1000, i);
            let faulted = faults.load(Ordering::SeqCst) != before;
            assert_eq!(faulted, tracked.contains(&i), "page {i}");
            assert_eq!(
                host_read::<u64>(&space, va + i * 0x1000),
                if faulted { 0 } else { i }
            );
        }
        guards.clear();
        for i in 0..PAGES {
            store(base, va + i * 0x1000, i + 1);
        }
        assert_eq!(
            faults.load(Ordering::SeqCst),
            tracked.len() as u64,
            "every page is writable again"
        );
    }
}
