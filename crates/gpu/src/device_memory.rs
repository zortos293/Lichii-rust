//! Device (SMMU) address allocation. Translation and access live in
//! [`lichii_memory::DeviceAddressSpace`]; this hands out the device ranges the GPU, nvmap and the
//! Switch 2 memory registry map there.

use std::collections::BTreeMap;
use std::ops::Bound;

use lichii_common::{DAddr, PAGE_MASK, PAGE_SIZE, align_up};
use parking_lot::Mutex;

/// A free or allocated range that cannot be returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum DeviceAllocError {
    #[error("device range {daddr:#x}+{size:#x} is unaligned, empty or outside the device space")]
    InvalidRange { daddr: DAddr, size: u64 },
    #[error("device range {daddr:#x}+{size:#x} is not allocated")]
    NotAllocated { daddr: DAddr, size: u64 },
}

/// First-fit allocator over a device address space; freed ranges are coalesced and reused
/// (NX2_NOTES rows 42 and 85). Address 0 is never handed out.
pub struct DeviceAllocator {
    size: u64,
    free: Mutex<BTreeMap<DAddr, u64>>,
}

impl DeviceAllocator {
    /// Allocates from a device space of `address_space_bits`, as
    /// [`DeviceAddressSpace::address_space_bits`](lichii_memory::DeviceAddressSpace::address_space_bits)
    /// reports it.
    pub fn new(address_space_bits: u32) -> Self {
        let size = 1u64 << address_space_bits;
        Self {
            size,
            free: Mutex::new(BTreeMap::from([(PAGE_SIZE, size - PAGE_SIZE)])),
        }
    }

    /// Reserves `size` bytes (rounded up to pages) aligned to `align`, a power of two of at least a
    /// page.
    pub fn allocate(&self, size: u64, align: u64) -> Option<DAddr> {
        assert!(
            align.is_power_of_two() && align >= PAGE_SIZE,
            "bad device alignment {align:#x}"
        );
        let size = align_up(size, PAGE_SIZE);
        if size == 0 {
            return None;
        }
        let mut free = self.free.lock();
        let (start, len, base) = free.iter().find_map(|(&start, &len)| {
            let base = start.checked_next_multiple_of(align)?;
            (base.checked_add(size)? <= start + len).then_some((start, len, base))
        })?;
        free.remove(&start);
        if base > start {
            free.insert(start, base - start);
        }
        let end = base + size;
        if end < start + len {
            free.insert(end, start + len - end);
        }
        Some(base)
    }

    /// Returns a range [`allocate`](Self::allocate) handed out, or part of one, for reuse.
    pub fn free(&self, daddr: DAddr, size: u64) -> Result<(), DeviceAllocError> {
        let size = align_up(size, PAGE_SIZE);
        let fits = daddr.checked_add(size).is_some_and(|end| end <= self.size);
        if daddr & PAGE_MASK != 0 || size == 0 || daddr == 0 || !fits {
            return Err(DeviceAllocError::InvalidRange { daddr, size });
        }
        let end = daddr + size;
        let mut free = self.free.lock();
        let prev = free.range(..=daddr).next_back().map(|(&s, &l)| (s, l));
        let next = free
            .range((Bound::Excluded(daddr), Bound::Unbounded))
            .next()
            .map(|(&s, &l)| (s, l));
        if prev.is_some_and(|(s, l)| s + l > daddr) || next.is_some_and(|(s, _)| s < end) {
            return Err(DeviceAllocError::NotAllocated { daddr, size });
        }
        let mut start = daddr;
        let mut stop = end;
        if let Some((s, l)) = prev
            && s + l == daddr
        {
            free.remove(&s);
            start = s;
        }
        if let Some((s, l)) = next
            && s == end
        {
            free.remove(&s);
            stop = s + l;
        }
        free.insert(start, stop - start);
        Ok(())
    }

    /// Bytes currently allocated.
    pub fn allocated_bytes(&self) -> u64 {
        let free: u64 = self.free.lock().values().sum();
        self.size - PAGE_SIZE - free
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_and_coalesces_freed_ranges() {
        let allocator = DeviceAllocator::new(36);
        let a = allocator.allocate(0x1_0000, 0x2_0000).unwrap();
        let b = allocator.allocate(0x1_0000, 0x2_0000).unwrap();
        assert_ne!(a, 0);
        assert!(b >= a + 0x1_0000);
        allocator.free(a, 0x1_0000).unwrap();
        assert_eq!(allocator.allocate(0x1_0000, 0x2_0000), Some(a));
        assert_eq!(
            allocator.free(b + 0x10_0000, 0x1000),
            Err(DeviceAllocError::NotAllocated {
                daddr: b + 0x10_0000,
                size: 0x1000
            })
        );
        assert_eq!(
            allocator.free(0, 0x1000),
            Err(DeviceAllocError::InvalidRange {
                daddr: 0,
                size: 0x1000
            })
        );
        allocator.free(a, 0x1_0000).unwrap();
        allocator.free(b, 0x1_0000).unwrap();
        assert_eq!(allocator.allocated_bytes(), 0);
        let whole = allocator.allocate((1 << 36) - PAGE_SIZE, PAGE_SIZE);
        assert_eq!(whole, Some(PAGE_SIZE));
        assert_eq!(allocator.allocate(PAGE_SIZE, PAGE_SIZE), None);
    }
}
