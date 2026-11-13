use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::{AddressSpace, FastmemError, HostFault, MemoryPermission, PhysicalMemory};

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
}
