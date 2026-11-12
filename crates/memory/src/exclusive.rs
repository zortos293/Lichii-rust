use std::sync::atomic::{AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering};

use lichii_common::VAddr;
use parking_lot::Mutex;
use portable_atomic::AtomicU128;

use crate::{AddressSpace, MemoryFault};

#[derive(Clone, Copy)]
struct Reservation {
    addr: VAddr,
    value: u128,
    size: u8,
}

/// Global exclusive monitor for LDXR/STXR (and LDXP/STXP) across all emulated cores.
///
/// A store-exclusive succeeds when the core still holds a reservation for the address and guest
/// memory still holds the value the load observed. The swap is a host compare-and-swap, so it is
/// atomic against plain stores from other cores.
pub struct ExclusiveMonitor {
    reservations: Box<[Mutex<Option<Reservation>>]>,
}

impl ExclusiveMonitor {
    pub fn new(core_count: usize) -> Self {
        Self {
            reservations: (0..core_count).map(|_| Mutex::new(None)).collect(),
        }
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
        *self.reservations[core].lock() = Some(Reservation { addr, value, size });
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
        let Some(reservation) = self.reservations[core].lock().take() else {
            return Ok(false);
        };
        if reservation.addr != addr || reservation.size != size {
            return Ok(false);
        }
        let (ptr, _) = mem.translate(addr).ok_or(MemoryFault::unmapped(addr))?;
        let expected = reservation.value;
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
        *self.reservations[core].lock() = None;
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
}
