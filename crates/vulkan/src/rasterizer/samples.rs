//! The Z-pass pixel counter: while the guest enables it, every draw runs inside a host occlusion
//! query, and a SamplesPassed report sums the queries recorded since the counter's last reset.
//!
//! Queries live in one pool used as a ring. Draws take slots in recording order and reset them
//! a block at a time from the command buffer, so recording never waits on the host GPU; reports
//! read their results once the work before them completed.

use ash::vk;

use crate::device::Device;
use crate::error::{Result, VkResultExt};

const SLOTS: u64 = 16 * 1024;
/// Slots reset by one command; a slot is taken only once every query of its block was read.
const BLOCK: u64 = 64;

/// The queries one SamplesPassed report sums: sequence numbers `from..to`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct SampleRange {
    from: u64,
    to: u64,
}

pub(super) struct SampleCounter {
    pool: vk::QueryPool,
    flags: vk::QueryControlFlags,
    /// Sequence number of the next query; its slot is `next % SLOTS`.
    next: u64,
    /// First query the counter sums since its last reset.
    reset_at: u64,
    /// Queries below this were read or are summed by no report; their slots may be reused.
    released: u64,
    summed_from: u64,
    summed_to: u64,
    sum: u64,
    results: Vec<u64>,
}

impl SampleCounter {
    pub(super) fn new(device: &Device) -> Result<Self> {
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(SLOTS as u32);
        // SAFETY: valid create info.
        let pool = unsafe {
            device
                .raw
                .create_query_pool(&info, None)
                .call("vkCreateQueryPool")?
        };
        let flags = match device.features.occlusion_query_precise {
            true => vk::QueryControlFlags::PRECISE,
            false => vk::QueryControlFlags::empty(),
        };
        Ok(Self {
            pool,
            flags,
            next: 0,
            reset_at: 0,
            released: 0,
            summed_from: 0,
            summed_to: 0,
            sum: 0,
            results: Vec::new(),
        })
    }

    /// Destroys the pool; the device must be idle.
    pub(super) fn destroy(&self, device: &Device) {
        // SAFETY: the caller waited for the device, so no pending work uses the pool.
        unsafe { device.raw.destroy_query_pool(self.pool, None) };
    }

    /// Takes the slot of the next counted draw, recording a reset of its block into `cmd`
    /// (outside any rendering) when the draw starts one. `None` when the block still holds
    /// queries no report has read; the draw then goes uncounted.
    pub(super) fn reserve(&mut self, device: &Device, cmd: vk::CommandBuffer) -> Option<u32> {
        let slot = (self.next % SLOTS) as u32;
        if self.next.is_multiple_of(BLOCK) {
            if self.next + BLOCK - self.released > SLOTS {
                return None;
            }
            // SAFETY: `cmd` is recording outside rendering, and the host read every query of
            // the block before releasing it.
            unsafe {
                device
                    .raw
                    .cmd_reset_query_pool(cmd, self.pool, slot, BLOCK as u32)
            };
        }
        self.next += 1;
        Some(slot)
    }

    /// Begins the query of a reserved `slot` inside the draw's rendering.
    pub(super) fn begin(&self, device: &Device, cmd: vk::CommandBuffer, slot: u32) {
        // SAFETY: `slot` was reset before this draw and is not active.
        unsafe { device.raw.cmd_begin_query(cmd, self.pool, slot, self.flags) };
    }

    /// Ends the query of `slot` in the rendering that began it.
    pub(super) fn end(&self, device: &Device, cmd: vk::CommandBuffer, slot: u32) {
        // SAFETY: `begin` started `slot` in the same rendering of `cmd`.
        unsafe { device.raw.cmd_end_query(cmd, self.pool, slot) };
    }

    /// CLEAR_REPORT_VALUE: later reports count only draws recorded from now on.
    pub(super) fn reset(&mut self) {
        self.reset_at = self.next;
    }

    /// What a report recorded now sums.
    pub(super) fn report(&self) -> SampleRange {
        SampleRange {
            from: self.reset_at,
            to: self.next,
        }
    }

    /// The counter value at `range`. Call it for reports in recording order, once the work
    /// recorded before each completed.
    pub(super) fn resolve(&mut self, device: &Device, range: SampleRange) -> u64 {
        if range.from != self.summed_from {
            self.summed_from = range.from;
            self.summed_to = range.from;
            self.sum = 0;
        }
        while self.summed_to < range.to {
            let first = self.summed_to % SLOTS;
            let count = (range.to - self.summed_to).min(SLOTS - first);
            self.results.resize(count as usize, 0);
            // SAFETY: every query in the range ended in work that completed, and no reset
            // reaches its slot before `released` passes it.
            let read = unsafe {
                device.raw.get_query_pool_results(
                    self.pool,
                    first as u32,
                    &mut self.results,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
            match read {
                Ok(()) => self.sum += self.results.iter().sum::<u64>(),
                Err(error) => tracing::error!(%error, "reading occlusion query results failed"),
            }
            self.summed_to += count;
        }
        self.released = range.to;
        self.sum
    }
}
