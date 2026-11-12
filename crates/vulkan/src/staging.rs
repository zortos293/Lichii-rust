use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use gpu_allocator::MemoryLocation;

use crate::device::Device;
use crate::error::Result;
use crate::memory::Buffer;
use crate::scheduler::Scheduler;

/// Host-visible upload memory handed out for one transfer.
pub struct StagingSlice<'a> {
    pub buffer: vk::Buffer,
    pub offset: u64,
    pub data: &'a mut [u8],
}

/// A ring of host-visible memory for uploads. Each allocation is tagged with the scheduler tick
/// that consumes it and reused once that tick completed, so steady-state uploads allocate
/// nothing. Requests larger than a quarter of the ring get a buffer of their own, freed the same
/// way.
pub struct StagingRing {
    device: Arc<Device>,
    ring: Buffer,
    capacity: u64,
    head: u64,
    /// In-flight regions in ring order: (start, end, tick).
    regions: VecDeque<(u64, u64, u64)>,
    oversized: VecDeque<(u64, Buffer)>,
}

impl StagingRing {
    pub fn new(device: &Arc<Device>, capacity: u64) -> Result<Self> {
        let ring = Buffer::new(
            device,
            capacity,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging ring",
        )?;
        Ok(Self {
            device: device.clone(),
            ring,
            capacity,
            head: 0,
            regions: VecDeque::new(),
            oversized: VecDeque::new(),
        })
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
            let buffer = Buffer::new(
                &self.device,
                size,
                vk::BufferUsageFlags::TRANSFER_SRC,
                MemoryLocation::CpuToGpu,
                "staging (oversized)",
            )?;
            self.oversized.push_back((scheduler.tick(), buffer));
            let (_, buffer) = self.oversized.back_mut().expect("just pushed");
            return Ok(StagingSlice {
                buffer: buffer.raw(),
                offset: 0,
                data: buffer.mapped_mut().expect("staging memory is host visible"),
            });
        }
        let tick = scheduler.tick();
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
        while self
            .oversized
            .front()
            .is_some_and(|(tick, _)| scheduler.is_complete(*tick))
        {
            self.oversized.pop_front();
        }
    }
}
