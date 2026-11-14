//! Host semaphore releases the front has queued. A macro DMA read of their words sees the values in
//! stream order from here, without draining the back or waiting for the host GPU to finish the
//! work before them: a release writes a value the front already knows.

use lichii_common::GpuVAddr;

use crate::backend::ReleaseSize;
use crate::gmmu::Gmmu;

const SLOTS: usize = 64;

#[derive(Clone, Copy)]
struct Release {
    address: GpuVAddr,
    payload: u64,
    words: usize,
    /// What guest memory held under the release when it was queued, until memory shows something
    /// else: the release landed, or the guest CPU wrote there since.
    before: Option<[u32; 4]>,
}

impl Release {
    fn end(&self) -> GpuVAddr {
        self.address + 4 * self.words as u64
    }
}

/// The last [`SLOTS`] addresses host semaphore releases were queued to on one channel.
pub(crate) struct PendingReleases {
    slots: [Option<Release>; SLOTS],
    next: usize,
}

impl PendingReleases {
    pub(crate) fn new() -> Self {
        Self {
            slots: [None; SLOTS],
            next: 0,
        }
    }

    /// The channel's GPU addresses now map elsewhere.
    pub(crate) fn clear(&mut self) {
        self.slots = [None; SLOTS];
    }

    /// A host semaphore release of `payload` to `address` was queued for the back.
    pub(crate) fn release(
        &mut self,
        memory: &Gmmu,
        address: GpuVAddr,
        payload: u64,
        size: ReleaseSize,
    ) {
        let words = match size {
            ReleaseSize::Four => 1,
            ReleaseSize::Eight => 2,
            ReleaseSize::Sixteen => 4,
        };
        let mut before = [0u32; 4];
        if !address.is_multiple_of(4)
            || memory
                .read_block(address, bytemuck::cast_slice_mut(&mut before[..words]))
                .is_err()
        {
            self.forget(address, 4 * words as u64);
            return;
        }
        let release = Release {
            address,
            payload,
            words,
            before: Some(before),
        };
        if let Some(slot) = self
            .slots
            .iter_mut()
            .find(|s| s.is_some_and(|s| s.address == address))
        {
            *slot = Some(release);
            return;
        }
        self.forget(address, 4 * words as u64);
        self.slots[self.next] = Some(release);
        self.next = (self.next + 1) % SLOTS;
    }

    /// Queued work writes `[address, address + size)` with values the front does not know.
    pub(crate) fn forget(&mut self, address: GpuVAddr, size: u64) {
        let end = address.saturating_add(size);
        for slot in &mut self.slots {
            if slot.is_some_and(|s| s.address < end && address < s.end()) {
                *slot = None;
            }
        }
    }

    /// Reads `out.len()` words at `gpu_va` as the queued work leaves them: guest memory, with the
    /// releases that have not landed yet over it, their timestamps taken from `now`. Returns false
    /// when no release queued here covers any of the words, or guest memory can't be read.
    pub(crate) fn read(
        &mut self,
        memory: &Gmmu,
        gpu_va: GpuVAddr,
        out: &mut [u32],
        now: impl Fn() -> u64,
    ) -> bool {
        let end = gpu_va + 4 * out.len() as u64;
        let covers = |s: &Release| s.address < end && gpu_va < s.end();
        if !gpu_va.is_multiple_of(4) || !self.slots.iter().flatten().any(covers) {
            return false;
        }
        if memory
            .read_block(gpu_va, bytemuck::cast_slice_mut(out))
            .is_err()
        {
            return false;
        }
        for release in self.slots.iter_mut().flatten() {
            if !covers(release) {
                continue;
            }
            let Some(before) = release.before else {
                continue;
            };
            let mut current = [0u32; 4];
            let words = release.words;
            if memory
                .read_block(
                    release.address,
                    bytemuck::cast_slice_mut(&mut current[..words]),
                )
                .is_err()
                || current[..words] != before[..words]
            {
                release.before = None;
                continue;
            }
            let timestamp = if words == 4 { now() } else { 0 };
            let value = [
                release.payload as u32,
                (release.payload >> 32) as u32,
                timestamp as u32,
                (timestamp >> 32) as u32,
            ];
            for (i, &word) in value[..words].iter().enumerate() {
                let at = release.address + 4 * i as u64;
                if (gpu_va..end).contains(&at) {
                    out[((at - gpu_va) / 4) as usize] = word;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use lichii_memory::{AddressSpace, DeviceAddressSpace, MemoryPermission, PhysicalMemory};

    use super::*;
    use crate::gmmu::BigPageSize;

    const VA: GpuVAddr = 0x20_0000;

    fn gmmu() -> Gmmu {
        let physical = Arc::new(PhysicalMemory::new(1 << 22).unwrap());
        let space = AddressSpace::new(physical.clone(), 39);
        space.map(
            0x1000_0000,
            0x10_0000,
            0x10_0000,
            MemoryPermission::READ_WRITE,
        );
        let device = Arc::new(DeviceAddressSpace::new(physical, 36));
        device
            .map_process(0x4_0000_0000, &space, 0x1000_0000, 0x10_0000)
            .unwrap();
        let gmmu = Gmmu::new(device, BigPageSize::Size64K);
        gmmu.map(VA, 0x4_0000_0000, 0x1_0000, 0, false).unwrap();
        gmmu
    }

    fn write(gmmu: &Gmmu, address: GpuVAddr, words: &[u32]) {
        gmmu.write_block(address, bytemuck::cast_slice(words))
            .unwrap();
    }

    #[test]
    fn a_read_sees_a_queued_release_before_it_lands() {
        let gmmu = gmmu();
        write(&gmmu, VA, &[1, 0, 0, 0, 0xaa, 0xbb]);
        let mut releases = PendingReleases::new();
        releases.release(&gmmu, VA, 0, ReleaseSize::Sixteen);
        let mut out = [0u32; 6];
        assert!(releases.read(&gmmu, VA, &mut out, || 0x7_1234_5678));
        assert_eq!(out, [0, 0, 0x1234_5678, 7, 0xaa, 0xbb]);
    }

    #[test]
    fn guest_memory_wins_once_the_release_landed_or_was_overwritten() {
        let gmmu = gmmu();
        write(&gmmu, VA, &[1, 0, 0, 0]);
        let mut releases = PendingReleases::new();
        releases.release(&gmmu, VA, 0, ReleaseSize::Sixteen);
        write(&gmmu, VA, &[0, 0, 5, 0]);
        let mut out = [0u32; 4];
        assert!(releases.read(&gmmu, VA, &mut out, || 9));
        assert_eq!(out, [0, 0, 5, 0]);
        write(&gmmu, VA, &[1, 0, 0, 0]);
        assert!(releases.read(&gmmu, VA, &mut out, || 9));
        assert_eq!(
            out,
            [1, 0, 0, 0],
            "a CPU write after the landing stays visible"
        );
    }

    #[test]
    fn reads_away_from_releases_and_forgotten_releases_are_not_served() {
        let gmmu = gmmu();
        let mut releases = PendingReleases::new();
        releases.release(&gmmu, VA + 0x20, 3, ReleaseSize::Four);
        let mut out = [0u32; 4];
        assert!(!releases.read(&gmmu, VA, &mut out, || 0));
        assert!(releases.read(&gmmu, VA + 0x1c, &mut out[..2], || 0));
        assert_eq!(out[1], 3);
        releases.forget(VA + 0x20, 8);
        assert!(!releases.read(&gmmu, VA + 0x20, &mut out, || 0));
    }

    #[test]
    fn a_later_release_to_the_same_address_replaces_the_earlier_one() {
        let gmmu = gmmu();
        let mut releases = PendingReleases::new();
        releases.release(&gmmu, VA, 1, ReleaseSize::Four);
        releases.release(&gmmu, VA, 2, ReleaseSize::Four);
        let mut out = [0u32; 1];
        assert!(releases.read(&gmmu, VA, &mut out, || 0));
        assert_eq!(out, [2]);
    }
}
