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
///
/// The ring asks the GPU which ticks completed once per scheduler tick, and again only when an
/// allocation does not fit. A tick's newest region stays until the tick completes, so within one
/// tick a reclaim can neither empty the ring nor move where the next allocation lands: it only
/// frees space an allocation that fits does not need. Allocations land where reclaiming before
/// each of them would put them.
pub struct StagingRing {
    device: Arc<Device>,
    ring: Buffer,
    capacity: u64,
    head: u64,
    /// In-flight regions in ring order: (start, end, tick).
    regions: VecDeque<(u64, u64, u64)>,
    /// The tick being recorded when the ring last reclaimed before an allocation in it; that
    /// allocation's region holds the tick.
    polled: u64,
    usage: vk::BufferUsageFlags,
    oversized_name: &'static str,
    oversized: Oversized,
}

impl StagingRing {
    pub fn new(device: &Arc<Device>, capacity: u64, oversized_cap: u64) -> Result<Self> {
        Self::with_usage(
            device,
            capacity,
            oversized_cap,
            STAGING_USAGE,
            "staging ring",
            "staging (oversized)",
        )
    }

    /// A ring of buffers made for `usage`, named `name` and `oversized_name`.
    pub(crate) fn with_usage(
        device: &Arc<Device>,
        capacity: u64,
        oversized_cap: u64,
        usage: vk::BufferUsageFlags,
        name: &'static str,
        oversized_name: &'static str,
    ) -> Result<Self> {
        let ring = Buffer::new(device, capacity, usage, MemoryLocation::CpuToGpu, name)?;
        Ok(Self {
            device: device.clone(),
            ring,
            capacity,
            head: 0,
            regions: VecDeque::new(),
            polled: u64::MAX,
            usage,
            oversized_name,
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
        if !self.shares_ring(size) {
            self.reclaim(scheduler);
            let buffer = self.oversized.alloc(
                &self.device,
                scheduler,
                size,
                self.usage,
                self.oversized_name,
            )?;
            return Ok(StagingSlice {
                buffer: buffer.raw(),
                offset: 0,
                data: buffer.mapped_mut().expect("staging memory is host visible"),
            });
        }
        let mut stale = self.polled != scheduler.tick();
        let start = loop {
            if stale {
                self.reclaim(scheduler);
                self.polled = scheduler.tick();
            }
            let aligned = self.head.next_multiple_of(align);
            let start = if aligned + size <= self.capacity {
                aligned
            } else {
                0
            };
            if self.fits(start, size) {
                break start;
            }
            if stale {
                let (_, _, oldest) = *self.regions.front().expect("a full ring has regions");
                scheduler.wait(oldest)?;
            }
            stale = true;
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

    /// Whether a request of `size` bytes shares the ring instead of getting an [`Oversized`]
    /// buffer of its own.
    pub(crate) fn shares_ring(&self, size: u64) -> bool {
        size <= self.capacity / 4
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
    use crate::scheduler::SemaphoreWait;

    /// A seeded random schedule of ring and oversized allocations, submissions held back by a
    /// host-signalled gate, and gate releases between allocations: every allocation lands where
    /// it lands in a ring that reclaims before each one.
    #[test]
    fn allocations_land_where_reclaiming_before_each_would_put_them() {
        let Some(device) = device() else { return };
        const CAPACITY: u64 = 64 << 10;
        let mut lazy = StagingRing::new(&device, CAPACITY, 1 << 20).unwrap();
        let mut eager = StagingRing::new(&device, CAPACITY, 1 << 20).unwrap();
        let mut scheduler = Scheduler::new(&device).unwrap();
        let mut kind = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let info = vk::SemaphoreCreateInfo::default().push_next(&mut kind);
        // SAFETY: valid create info on a live device.
        let gate = unsafe { device.raw.create_semaphore(&info, None) }.unwrap();
        let mut gated = Vec::new();
        let mut released = 0;
        let release = |scheduler: &mut Scheduler, released: &mut usize, gated: &[u64], value| {
            if value <= *released {
                return;
            }
            *released = value;
            let info = vk::SemaphoreSignalInfo::default()
                .semaphore(gate)
                .value(value as u64);
            // SAFETY: the gate is a live timeline semaphore below `value`.
            unsafe { device.raw.signal_semaphore(&info) }.unwrap();
            scheduler.wait(gated[value - 1]).unwrap();
        };
        let mut seed = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (mut allocs, mut reused) = (0, 0);
        for step in 0..20_000 {
            match next() % 16 {
                0 => {
                    let wait = SemaphoreWait {
                        semaphore: gate,
                        value: gated.len() as u64 + 1,
                        stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                    };
                    gated.push(scheduler.flush_with(&[wait], &[]).unwrap());
                }
                1 | 2 => {
                    let value = match next() % 4 {
                        0 => gated.len(),
                        _ => released + (next() % 3) as usize,
                    };
                    release(
                        &mut scheduler,
                        &mut released,
                        &gated,
                        value.min(gated.len()),
                    );
                }
                3 | 4 => {
                    let size = CAPACITY / 4 + 1 + next() % CAPACITY;
                    eager.reclaim(&mut scheduler);
                    if eager.oversized().bytes + 2 * size > 1 << 20 {
                        release(&mut scheduler, &mut released, &gated, gated.len());
                    }
                    eager.alloc(&mut scheduler, size, 16).unwrap();
                    lazy.alloc(&mut scheduler, size, 16).unwrap();
                }
                _ => {
                    let size = 1 + next() % (CAPACITY / 4);
                    let align = [1, 4, 16, 256][(next() % 4) as usize];
                    eager.reclaim(&mut scheduler);
                    let aligned = eager.head.next_multiple_of(align);
                    let start = if aligned + size <= CAPACITY {
                        aligned
                    } else {
                        0
                    };
                    if !eager.fits(start, size) {
                        release(&mut scheduler, &mut released, &gated, gated.len());
                    }
                    reused += u64::from(start < eager.head);
                    let expected = eager.alloc(&mut scheduler, size, align).unwrap().offset;
                    let offset = lazy.alloc(&mut scheduler, size, align).unwrap().offset;
                    assert_eq!(
                        offset, expected,
                        "step {step}: {size:#x} aligned to {align}"
                    );
                    allocs += 1;
                }
            }
        }
        assert!(
            allocs > 10_000 && reused > 100,
            "{allocs} allocations, {reused} wraps"
        );
        release(&mut scheduler, &mut released, &gated, gated.len());
        scheduler.finish().unwrap();
        // SAFETY: every submission waiting on the gate completed.
        unsafe { device.raw.destroy_semaphore(gate, None) };
    }

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
