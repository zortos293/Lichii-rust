use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::device::Device;
use crate::error::Result;
use crate::host_budget::Oversized;
use crate::memory::{Buffer, LiveMemory};
use crate::scheduler::Scheduler;

/// Host-visible upload memory handed out for one transfer.
pub struct StagingSlice<'a> {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub data: &'a mut [u8],
}

/// Staging memory feeds transfers and the compute passes that decode uploads on the GPU.
const STAGING_USAGE: vk::BufferUsageFlags = vk::BufferUsageFlags::from_raw(
    vk::BufferUsageFlags::TRANSFER_SRC.as_raw() | vk::BufferUsageFlags::STORAGE_BUFFER.as_raw(),
);

/// A ring of host-visible memory for uploads. Each allocation is tagged with the scheduler tick
/// that consumes it and reused once that tick completed, so steady-state uploads allocate
/// nothing. Requests larger than a quarter of the ring get a buffer of their own, freed the same
/// way, at most `oversized_cap` bytes of them in flight ([`Oversized`]).
pub struct StagingRing {
    device: Arc<Device>,
    ring: Buffer,
    capacity: u64,
    head: u64,
    /// In-flight regions in ring order: (start, end, tick).
    regions: VecDeque<(u64, u64, u64)>,
    oversized: Oversized,
}

impl StagingRing {
    pub fn new(device: &Arc<Device>, capacity: u64, oversized_cap: u64) -> Result<Self> {
        let ring = Buffer::new(
            device,
            capacity,
            STAGING_USAGE,
            MemoryLocation::CpuToGpu,
            "staging ring",
        )?;
        Ok(Self {
            device: device.clone(),
            ring,
            capacity,
            head: 0,
            regions: VecDeque::new(),
            oversized: Oversized::new(oversized_cap),
        })
    }

    /// The oversized buffers not freed yet.
    pub fn oversized(&self) -> LiveMemory {
        self.oversized.live()
    }

    /// `size` bytes aligned to `align` (a power of two), valid until the work recorded in
    /// `scheduler` right now completes.
    pub fn alloc(
        &mut self,
        scheduler: &mut Scheduler,
        size: u64,
        align: u64,
    ) -> Result<StagingSlice<'_>> {
        if size > self.capacity / 4 {
            self.reclaim(scheduler);
            let buffer = self.oversized.alloc(
                &self.device,
                scheduler,
                size,
                STAGING_USAGE,
                "staging (oversized)",
            )?;
            return Ok(StagingSlice {
                buffer: buffer.raw(),
                offset: 0,
                data: buffer.mapped_mut().expect("staging memory is host visible"),
            });
        }
        let start = loop {
            self.reclaim(scheduler);
            let aligned = self.head.next_multiple_of(align);
            let start = if aligned + size <= self.capacity {
                aligned
            } else {
                0
            };
            if self.fits(start, size) {
                break start;
            }
            let (_, _, oldest) = *self.regions.front().expect("a full ring has regions");
            scheduler.wait(oldest)?;
        };
        let tick = scheduler.tick();
        let end = start + size;
        match self.regions.back_mut() {
            Some(last) if last.2 == tick && last.1 <= start => last.1 = end,
            _ => self.regions.push_back((start, end, tick)),
        }
        self.head = end;
        let buffer = self.ring.raw();
        let data = &mut self
            .ring
            .mapped_mut()
            .expect("staging memory is host visible")[start as usize..end as usize];
        Ok(StagingSlice {
            buffer,
            offset: start,
            data,
        })
    }

    fn fits(&self, start: u64, size: u64) -> bool {
        let end = start + size;
        self.regions.iter().all(|&(s, e, _)| end <= s || start >= e)
    }

    fn reclaim(&mut self, scheduler: &mut Scheduler) {
        while let Some(&(_, _, tick)) = self.regions.front() {
            if !scheduler.is_complete(tick) {
                break;
            }
            self.regions.pop_front();
        }
        if self.regions.is_empty() {
            self.head = 0;
        }
        self.oversized.reclaim(scheduler);
    }

    /// Whether the work being recorded at `tick` holds half the oversized cap or more
    /// ([`Oversized::should_submit`]).
    pub(crate) fn should_submit(&self, tick: u64) -> bool {
        self.oversized.should_submit(tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::host_budget::testing::{device, flood};

    #[test]
    fn oversized_staging_floods_stay_within_their_cap_on_a_lagging_gpu() {
        let Some(device) = device() else { return };
        const CAP: u64 = 8 << 20;
        const SIZE: u64 = 3 << 19;
        let mut ring = StagingRing::new(&device, 1 << 20, CAP).unwrap();
        let ops = 64;
        let result = flood(
            &device,
            &mut ring,
            ops,
            |ring, scheduler| {
                let slice = ring.alloc(scheduler, SIZE, 16).unwrap();
                assert_eq!(slice.data.len() as u64, SIZE);
            },
            |ring| ring.oversized().bytes,
            StagingRing::should_submit,
        );
        assert!(
            u64::from(ops) * SIZE > 8 * CAP,
            "the flood is many times the cap"
        );
        assert!(result.lagged > 0, "allocations ran ahead of the GPU");
        assert!(
            result.peak > CAP / 2,
            "the cap bound the flood: {}",
            result.peak
        );
        assert!(
            result.peak <= CAP,
            "{} in flight passed the cap",
            result.peak
        );
    }
}
