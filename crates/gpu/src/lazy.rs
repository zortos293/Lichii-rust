//! Lazy device ranges: device address ranges with no memory behind them until something writes
//! there. Switch 2 titles ask nx2:gpu 214 for GPU-only regions far larger than what they fill
//! (Mario Kart World asks for about 9 GiB), so, as in the reference, each region gets a device
//! range of its own and a [`LAZY_BLOCK_SIZE`] block gets memory only when a write lands in it.
//!
//! A lazy range is ordinary device memory to its readers: GPU mappings point at it, the texture
//! cache places images over it, and pages without memory read zero. Writers go through
//! [`LazyDeviceRanges::write`], which only does more than a plain device write when that write
//! faults: it backs the blocks the write touches through the services' [`LazyBacking`] and writes
//! again. Zeros written over a block that has no memory are left out, since it reads zero already.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lichii_common::DAddr;
use lichii_memory::{DeviceAddressSpace, MemoryFault};
use parking_lot::Mutex;

/// Granularity at which a lazy range gets memory. Lazy ranges start on a multiple of it.
pub const LAZY_BLOCK_SIZE: u64 = 4 << 20;
const BLOCK_BITS: u32 = LAZY_BLOCK_SIZE.trailing_zeros();

/// Gives a block of a lazy range its memory; installed by the services.
pub trait LazyBacking: Send + Sync {
    /// Maps memory that reads zero behind `[daddr, daddr + size)` of the device address space,
    /// and returns whether it did. Runs on GPU threads with the ranges' lock held, so it must take
    /// the memory from a reserve that a services or host thread refills: it never allocates from
    /// the kernel and never waits. Without memory ready it returns false, and the write that
    /// needed the block is dropped and counted.
    fn back(&self, daddr: DAddr, size: u64) -> bool;
}

/// A block of a lazy range that [`LazyBacking::back`] gave memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LazyBlock {
    pub daddr: DAddr,
    pub size: u64,
}

/// Every lazy range of one device address space.
pub struct LazyDeviceRanges {
    device: Arc<DeviceAddressSpace>,
    /// A bit per [`LAZY_BLOCK_SIZE`] of device space, set where a lazy range lies: readers check
    /// it without the lock, which only backing changes take.
    blocks: Box<[AtomicU64]>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    backing: Option<Arc<dyn LazyBacking>>,
    regions: BTreeMap<DAddr, Region>,
}

struct Region {
    size: u64,
    backed: Vec<bool>,
    /// Writes dropped because no memory was ready for their block.
    dropped: u64,
}

impl Region {
    fn block(&self, start: DAddr, index: usize) -> LazyBlock {
        let offset = index as u64 * LAZY_BLOCK_SIZE;
        LazyBlock {
            daddr: start + offset,
            size: LAZY_BLOCK_SIZE.min(self.size - offset),
        }
    }
}

impl LazyDeviceRanges {
    pub fn new(device: Arc<DeviceAddressSpace>) -> Self {
        let blocks = (1u64 << device.address_space_bits().saturating_sub(BLOCK_BITS)).div_ceil(64);
        Self {
            device,
            blocks: (0..blocks).map(|_| AtomicU64::new(0)).collect(),
            state: Mutex::default(),
        }
    }

    /// Installs what gives blocks memory. Until then every write that needs a block is dropped.
    pub fn set_backing(&self, backing: Arc<dyn LazyBacking>) {
        self.state.lock().backing = Some(backing);
    }

    /// Makes `[daddr, daddr + size)`, which must have no memory mapped, a lazy range. `daddr` is a
    /// multiple of [`LAZY_BLOCK_SIZE`], and nothing else lies in the rest of its last block.
    pub(crate) fn add(&self, daddr: DAddr, size: u64) {
        assert!(
            daddr.is_multiple_of(LAZY_BLOCK_SIZE),
            "unaligned lazy range {daddr:#x}"
        );
        let blocks = size.div_ceil(LAZY_BLOCK_SIZE) as usize;
        self.state.lock().regions.insert(
            daddr,
            Region {
                size,
                backed: vec![false; blocks],
                dropped: 0,
            },
        );
        self.mark(daddr, size, true);
    }

    fn mark(&self, daddr: DAddr, size: u64, lazy: bool) {
        for block in daddr >> BLOCK_BITS..(daddr + size).div_ceil(LAZY_BLOCK_SIZE) {
            let bit = 1 << (block % 64);
            let word = &self.blocks[(block / 64) as usize];
            if lazy {
                word.fetch_or(bit, Ordering::Release);
            } else {
                word.fetch_and(!bit, Ordering::Release);
            }
        }
    }

    /// Ends the lazy range at `daddr` and returns the blocks that got memory, which stay mapped
    /// in the device address space.
    pub(crate) fn remove(&self, daddr: DAddr) -> Vec<LazyBlock> {
        let Some(region) = self.state.lock().regions.remove(&daddr) else {
            return Vec::new();
        };
        self.mark(daddr, region.size, false);
        if region.dropped != 0 {
            tracing::warn!(
                daddr = format_args!("{daddr:#x}"),
                size = format_args!("{:#x}", region.size),
                dropped = region.dropped,
                "lazy device range destroyed after dropping writes that had no memory ready"
            );
        }
        (0..region.backed.len())
            .filter(|&index| region.backed[index])
            .map(|index| region.block(daddr, index))
            .collect()
    }

    /// Whether `[daddr, daddr + size)` overlaps a lazy range. Takes no lock.
    pub fn is_lazy(&self, daddr: DAddr, size: u64) -> bool {
        if size == 0 {
            return false;
        }
        let (first, last) = (daddr >> BLOCK_BITS, (daddr + size - 1) >> BLOCK_BITS);
        (first..=last).any(|block| {
            self.blocks
                .get((block / 64) as usize)
                .is_some_and(|word| word.load(Ordering::Acquire) & (1 << (block % 64)) != 0)
        })
    }

    /// Writes `data` to device memory at `daddr`. Blocks of lazy ranges the write touches get
    /// memory first, unless only zeros land in them; outside lazy ranges this is
    /// [`DeviceAddressSpace::write_bytes`]. Fails on a page outside lazy ranges with no memory,
    /// after writing the rest; writes dropped for want of block memory are counted, not failed.
    pub fn write(&self, daddr: DAddr, data: &[u8]) -> Result<(), MemoryFault> {
        let fault = match self.device.write_bytes(daddr, data) {
            Ok(()) => return Ok(()),
            Err(fault) => fault,
        };
        if !self.is_lazy(daddr, data.len() as u64) {
            return Err(fault);
        }
        let end = daddr + data.len() as u64;
        let mut state = self.state.lock();
        let mut result = Ok(());
        let mut at = daddr;
        while at < end {
            let (stop, write) = match state.block_at(at) {
                Some((start, index)) => {
                    let block = state.regions[&start].block(start, index);
                    let stop = end.min(block.daddr + block.size);
                    let chunk = &data[(at - daddr) as usize..(stop - daddr) as usize];
                    let write = state.regions[&start].backed[index]
                        || (chunk.iter().any(|&byte| byte != 0) && state.back_block(start, index));
                    (stop, write)
                }
                None => (
                    state.next_region(at).map_or(end, |next| next.min(end)),
                    true,
                ),
            };
            if write
                && let Err(fault) = self
                    .device
                    .write_bytes(at, &data[(at - daddr) as usize..(stop - daddr) as usize])
            {
                result = result.and(Err(fault));
            }
            at = stop;
        }
        result
    }

    /// Gives memory to the blocks of lazy ranges in `[daddr, daddr + size)`, for an access that
    /// needs it in place (an atomic). Whether every such block has memory now.
    pub fn back(&self, daddr: DAddr, size: u64) -> bool {
        if !self.is_lazy(daddr, size) {
            return true;
        }
        let end = daddr + size;
        let mut state = self.state.lock();
        let mut backed = true;
        let mut at = daddr;
        while at < end {
            let Some((start, index)) = state.block_at(at) else {
                at = state.next_region(at).map_or(end, |next| next.min(end));
                continue;
            };
            let block = state.regions[&start].block(start, index);
            backed &= state.regions[&start].backed[index] || state.back_block(start, index);
            at = block.daddr + block.size;
        }
        backed
    }
}

impl State {
    /// The region and block holding `daddr`.
    fn block_at(&self, daddr: DAddr) -> Option<(DAddr, usize)> {
        let (&start, region) = self.regions.range(..=daddr).next_back()?;
        (daddr < start + region.size).then(|| (start, ((daddr - start) / LAZY_BLOCK_SIZE) as usize))
    }

    fn next_region(&self, daddr: DAddr) -> Option<DAddr> {
        self.regions.range(daddr..).next().map(|(&start, _)| start)
    }

    /// Asks the backing for a block's memory; counts and reports a write it drops.
    fn back_block(&mut self, start: DAddr, index: usize) -> bool {
        let backing = self.backing.clone();
        let region = self.regions.get_mut(&start).expect("the caller found it");
        let block = region.block(start, index);
        if backing.is_some_and(|backing| backing.back(block.daddr, block.size)) {
            region.backed[index] = true;
            return true;
        }
        region.dropped += 1;
        if region.dropped == 1 {
            tracing::warn!(
                region = format_args!("{start:#x}"),
                size = format_args!("{:#x}", region.size),
                block = format_args!("{:#x}", block.daddr),
                "lazy device range: no memory ready for a block, so its writes are dropped; \
                 the count is reported when the region is destroyed"
            );
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_memory::PhysicalMemory;
    use std::sync::atomic::AtomicUsize;

    const BASE: DAddr = 0x1_0000_0000;

    /// Backs blocks from a small reserve of DRAM, as the services do.
    struct Reserve {
        device: Arc<DeviceAddressSpace>,
        next: Mutex<Vec<u64>>,
        calls: AtomicUsize,
    }

    impl LazyBacking for Reserve {
        fn back(&self, daddr: DAddr, size: u64) -> bool {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let Some(pa) = self.next.lock().pop() else {
                return false;
            };
            self.device.map(daddr, pa, size);
            true
        }
    }

    fn fixture(blocks: u64) -> (Arc<DeviceAddressSpace>, LazyDeviceRanges, Arc<Reserve>) {
        let physical = Arc::new(PhysicalMemory::new(LAZY_BLOCK_SIZE * 4).unwrap());
        let device = Arc::new(DeviceAddressSpace::new(physical, 36));
        let ranges = LazyDeviceRanges::new(device.clone());
        let reserve = Arc::new(Reserve {
            device: device.clone(),
            next: Mutex::new((0..blocks).map(|block| block * LAZY_BLOCK_SIZE).collect()),
            calls: AtomicUsize::new(0),
        });
        ranges.set_backing(reserve.clone());
        (device, ranges, reserve)
    }

    #[test]
    fn unbacked_blocks_read_zero_and_get_memory_on_their_first_write() {
        let (device, ranges, reserve) = fixture(2);
        ranges.add(BASE, 3 * LAZY_BLOCK_SIZE);
        assert!(device.read::<u32>(BASE + 0x10).is_err());
        ranges.write(BASE + 0x10, &[0; 64]).unwrap();
        assert_eq!(
            reserve.calls.load(Ordering::SeqCst),
            0,
            "zeros need no memory"
        );
        let straddling = BASE + LAZY_BLOCK_SIZE - 2;
        ranges.write(straddling, &[1, 2, 3, 4]).unwrap();
        assert_eq!(device.read::<u32>(straddling), Ok(0x0403_0201));
        assert_eq!(reserve.calls.load(Ordering::SeqCst), 2);
        assert_eq!(device.read::<u32>(BASE), Ok(0), "a backed block reads zero");
        ranges.write(BASE + 0x20, &[5]).unwrap();
        assert_eq!(reserve.calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            ranges.remove(BASE),
            [
                LazyBlock {
                    daddr: BASE,
                    size: LAZY_BLOCK_SIZE
                },
                LazyBlock {
                    daddr: BASE + LAZY_BLOCK_SIZE,
                    size: LAZY_BLOCK_SIZE
                }
            ]
        );
    }

    #[test]
    fn writes_without_memory_ready_are_dropped_and_counted() {
        let (device, ranges, _) = fixture(0);
        ranges.add(BASE, LAZY_BLOCK_SIZE / 2);
        ranges.write(BASE, &[7]).unwrap();
        ranges.write(BASE + 1, &[7]).unwrap();
        assert!(device.read::<u8>(BASE).is_err());
        assert_eq!(ranges.state.lock().regions[&BASE].dropped, 2);
        assert!(ranges.remove(BASE).is_empty());
    }

    #[test]
    fn writes_outside_lazy_ranges_still_fault() {
        let (_, ranges, _) = fixture(1);
        ranges.add(BASE, LAZY_BLOCK_SIZE);
        assert!(ranges.write(BASE + LAZY_BLOCK_SIZE, &[1]).is_err());
        assert!(ranges.write(BASE + LAZY_BLOCK_SIZE - 1, &[1, 1]).is_err());
        assert_eq!(
            ranges.remove(BASE).len(),
            1,
            "the lazy part was still backed"
        );
        assert!(!ranges.is_lazy(BASE, LAZY_BLOCK_SIZE));
    }

    #[test]
    fn back_gives_memory_for_an_atomic() {
        let (device, ranges, _) = fixture(1);
        ranges.add(BASE, LAZY_BLOCK_SIZE);
        assert!(ranges.back(BASE + 8, 4));
        assert!(device.translate(BASE + 8).is_some());
        assert!(ranges.back(BASE + LAZY_BLOCK_SIZE * 8, 4), "not lazy");
    }
}
