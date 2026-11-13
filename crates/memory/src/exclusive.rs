use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

use lichii_common::VAddr;
use portable_atomic::AtomicU128;

use crate::{AddressSpace, MemoryFault};

/// One core's reservation, laid out so translated code can take and check it inline, on its own
/// cache line so cores never share one.
///
/// Only the thread running the core touches its slot, so every access is a relaxed atomic, and
/// translated code may use plain loads and stores at these offsets. `size` is zero when the
/// core holds no reservation; `value` is the memory value the load-exclusive observed,
/// zero-extended to 128 bits and split into 64-bit halves.
#[repr(C, align(64))]
pub struct ReservationSlot {
    pub value_lo: AtomicU64,
    pub value_hi: AtomicU64,
    pub addr: AtomicU64,
    pub size: AtomicU64,
}

/// Global exclusive monitor for LDXR/STXR (and LDXP/STXP) across all emulated cores.
///
/// A store-exclusive succeeds when the core still holds a reservation for the address and guest
/// memory still holds the value the load observed. The swap is a host compare-and-swap, so it is
/// atomic against plain stores from other cores.
pub struct ExclusiveMonitor {
    slots: Box<[ReservationSlot]>,
}

impl ExclusiveMonitor {
    pub fn new(core_count: usize) -> Self {
        Self {
            slots: (0..core_count)
                .map(|_| ReservationSlot {
                    value_lo: AtomicU64::new(0),
                    value_hi: AtomicU64::new(0),
                    addr: AtomicU64::new(0),
                    size: AtomicU64::new(0),
                })
                .collect(),
        }
    }

    /// The reservation of `core`, for translated code running that core to use inline.
    pub fn slot(&self, core: usize) -> &ReservationSlot {
        &self.slots[core]
    }

    /// Load-exclusive of \`size\` bytes (1, 2, 4, 8 or 16).
    pub fn read(
        &self,
        core: usize,
        mem: &AddressSpace,
        addr: VAddr,
        size: u8,
    ) -> Result<u128, MemoryFault> {
        let value = match size {
            1 => mem.read::<u8>(addr)? as u128,
            2 => mem.read::<u16>(addr)? as u128,
            4 => mem.read::<u32>(addr)? as u128,
            8 => mem.read::<u64>(addr)? as u128,
            16 => {
                let (ptr, _) = mem.translate(addr).ok_or(MemoryFault::unmapped(addr))?;
                // SAFETY: exclusive pairs are 16-byte aligned, so the value is aligned and lies in
                // one mapped page.
                unsafe { AtomicU128::from_ptr(ptr.cast()) }.load(Ordering::SeqCst)
            }
            _ => unreachable!("exclusive size {size}"),
        };
        let slot = &self.slots[core];
        slot.value_lo.store(value as u64, Ordering::Relaxed);
        slot.value_hi.store((value >> 64) as u64, Ordering::Relaxed);
        slot.addr.store(addr, Ordering::Relaxed);
        slot.size.store(size.into(), Ordering::Relaxed);
        Ok(value)
    }

    /// Store-exclusive. Returns \`Ok(true)\` when the store happened.
    pub fn write(
        &self,
        core: usize,
        mem: &AddressSpace,
        addr: VAddr,
        size: u8,
        value: u128,
    ) -> Result<bool, MemoryFault> {
        let slot = &self.slots[core];
        if slot.size.swap(0, Ordering::Relaxed) != u64::from(size)
            || slot.addr.load(Ordering::Relaxed) != addr
        {
            return Ok(false);
        }
        let (ptr, _) = mem.translate(addr).ok_or(MemoryFault::unmapped(addr))?;
        let expected = u128::from(slot.value_lo.load(Ordering::Relaxed))
            | u128::from(slot.value_hi.load(Ordering::Relaxed)) << 64;
        // SAFETY: exclusive accesses are naturally aligned, so ptr is aligned for the atomic type
        // and the value lies in one page.
        let stored = unsafe {
            match size {
                1 => cas::<AtomicU8, u8>(ptr, expected as u8, value as u8),
                2 => cas::<AtomicU16, u16>(ptr, expected as u16, value as u16),
                4 => cas::<AtomicU32, u32>(ptr, expected as u32, value as u32),
                8 => cas::<AtomicU64, u64>(ptr, expected as u64, value as u64),
                16 => AtomicU128::from_ptr(ptr.cast())
                    .compare_exchange(expected, value, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok(),
                _ => unreachable!("exclusive size {size}"),
            }
        };
        Ok(stored)
    }

    pub fn clear(&self, core: usize) {
        self.slots[core].size.store(0, Ordering::Relaxed);
    }
}

trait HostAtomic<T> {
    fn cas(&self, current: T, new: T) -> bool;
}

macro_rules! host_atomic {
    ($atomic:ty, $int:ty) => {
        impl HostAtomic<$int> for $atomic {
            fn cas(&self, current: $int, new: $int) -> bool {
                self.compare_exchange(current, new, Ordering::SeqCst, Ordering::SeqCst)
                    .is_ok()
            }
        }
    };
}
host_atomic!(AtomicU8, u8);
host_atomic!(AtomicU16, u16);
host_atomic!(AtomicU32, u32);
host_atomic!(AtomicU64, u64);

/// # Safety
/// \`ptr\` must be valid and aligned for \`A\`.
unsafe fn cas<A: HostAtomic<T>, T>(ptr: *mut u8, current: T, new: T) -> bool {
    // SAFETY: guaranteed by the caller.
    unsafe { &*(ptr as *const A) }.cas(current, new)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{MemoryPermission, PhysicalMemory};

    #[test]
    fn store_fails_after_intervening_write() {
        let mem = AddressSpace::new(Arc::new(PhysicalMemory::new(1 << 20).unwrap()), 39);
        mem.map(0x1000, 0, 0x1000, MemoryPermission::READ_WRITE);
        let monitor = ExclusiveMonitor::new(2);
        monitor.read(0, &mem, 0x1000, 8).unwrap();
        mem.write(0x1000, 5u64).unwrap();
        assert!(!monitor.write(0, &mem, 0x1000, 8, 9).unwrap());
        monitor.read(0, &mem, 0x1000, 8).unwrap();
        assert!(monitor.write(0, &mem, 0x1000, 8, 9).unwrap());
        assert_eq!(mem.read::<u64>(0x1000).unwrap(), 9);
    }

    #[test]
    fn slots_hold_the_reservation_until_taken_or_cleared() {
        let mem = AddressSpace::new(Arc::new(PhysicalMemory::new(1 << 20).unwrap()), 39);
        mem.map(0x1000, 0, 0x1000, MemoryPermission::READ_WRITE);
        mem.write(0x1010, u128::MAX - 1).unwrap();
        let monitor = ExclusiveMonitor::new(2);
        monitor.read(1, &mem, 0x1010, 16).unwrap();
        let slot = monitor.slot(1);
        assert_eq!(slot.addr.load(Ordering::Relaxed), 0x1010);
        assert_eq!(slot.size.load(Ordering::Relaxed), 16);
        assert_eq!(slot.value_lo.load(Ordering::Relaxed), u64::MAX - 1);
        assert_eq!(slot.value_hi.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(monitor.slot(0).size.load(Ordering::Relaxed), 0);
        assert!(!monitor.write(1, &mem, 0x1018, 8, 0).unwrap());
        assert_eq!(
            slot.size.load(Ordering::Relaxed),
            0,
            "a failed store takes the reservation"
        );
        monitor.read(1, &mem, 0x1010, 16).unwrap();
        monitor.clear(1);
        assert!(!monitor.write(1, &mem, 0x1010, 16, 7).unwrap());
        monitor.read(1, &mem, 0x1010, 16).unwrap();
        assert!(monitor.write(1, &mem, 0x1010, 16, 7).unwrap());
        assert_eq!(mem.read::<u128>(0x1010).unwrap(), 7);
    }
}
