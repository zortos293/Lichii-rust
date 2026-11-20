//! GPU channels as the CPU side sees them: how a channel is created and what a submission
//! carries.
//!
//! A channel is one nvhost-gpu channel or NVN2 queue: a GPU address space, a host class, eight
//! subchannels and a queue of GPFIFO entries. Submissions go to the GPU's front thread in order
//! (see [`crate::gpu::Gpu`]).

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use lichii_common::DAddr;
use lichii_memory::DeviceAddressSpace;
use lichii_texture::block_linear::GobLayout;
use parking_lot::Mutex;

pub use crate::backend::ChannelId;
use crate::gmmu::Gmmu;
use crate::gpu::{Console, Inbox};
use crate::host::{AcquireOp, HostClass};
pub use crate::pushbuffer::GpEntry;
use crate::queue_trace::{Event, QueueTrace};

/// How a channel is set up.
#[derive(Clone)]
pub struct ChannelConfig {
    /// The GPU address space the channel's commands and engines use (several channels may share
    /// one).
    pub memory: Arc<Gmmu>,
    /// B06F on Switch 1, C56F and later on Switch 2.
    pub host_class: HostClass,
    /// GOB byte order of the channel's block-linear surfaces.
    pub gob_layout: GobLayout,
}

impl ChannelConfig {
    /// A channel of `host_class` with that console's GOB layout: Switch 2 channels use the
    /// native swizzle.
    pub fn new(memory: Arc<Gmmu>, host_class: HostClass) -> Self {
        Self {
            memory,
            host_class,
            gob_layout: match host_class {
                HostClass::Maxwell => GobLayout::Tegra,
                HostClass::Ampere => GobLayout::Switch2,
            },
        }
    }

    /// The channel a console's driver creates: Maxwell B06F with Tegra GOBs on Switch 1, Ampere
    /// C56F with the native GOB swizzle on Switch 2 (NX2_NOTES row 48).
    pub fn for_console(memory: Arc<Gmmu>, console: Console) -> Self {
        Self::new(
            memory,
            match console {
                Console::Switch1 => HostClass::Maxwell,
                Console::Switch2 => HostClass::Ampere,
            },
        )
    }
}

/// A host1x syncpoint threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncpointFence {
    /// Syncpoint id.
    pub id: u32,
    /// Value the syncpoint must reach (compared as a wrapping counter).
    pub value: u32,
}

/// Syncpoint increments the driver appends after a submission's entries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncpointIncrement {
    /// Syncpoint id.
    pub id: u32,
    /// Increments, each counted once the work before it finished.
    pub count: u32,
    /// Wait for the channel's earlier work to go idle first.
    pub wait_for_idle: bool,
}

impl SyncpointIncrement {
    /// Switch 1 `fence_increment`: two increments, after a wait for idle unless the submit
    /// suppressed it.
    pub fn fence_increment(id: u32, suppress_wfi: bool) -> Self {
        Self {
            id,
            count: 2,
            wait_for_idle: !suppress_wfi,
        }
    }

    /// Switch 2 `increment_value` (NX2_NOTES row 32): the pushbuffers signal completion with
    /// semaphore releases only, but the driver raised the syncpoint's maximum by the fence value
    /// and waits for it, so `value` increments follow the work, after a wait for idle.
    pub fn increment_value(id: u32, value: u32) -> Self {
        Self {
            id,
            count: value,
            wait_for_idle: true,
        }
    }
}

/// One GPFIFO submission. [`Gpu::submit`](crate::gpu::Gpu::submit) copies the entries, so the
/// caller keeps and reuses its buffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct Submission<'a> {
    /// Pushbuffer segments, read from GPU memory when the front thread reaches them.
    pub entries: &'a [GpEntry],
    /// `fence_wait`: hold the entries until this syncpoint value is reached.
    pub wait: Option<SyncpointFence>,
    /// Increments after the entries.
    pub increment: Option<SyncpointIncrement>,
}

impl<'a> Submission<'a> {
    /// A submission of `entries` without fences.
    pub fn new(entries: &'a [GpEntry]) -> Self {
        Self {
            entries,
            wait: None,
            increment: None,
        }
    }
}

/// Channels suspended on a semaphore acquire, shared by the GPU and the renderer through
/// [`Syncpoints::semaphore_waiters`](crate::host1x::Syncpoints::semaphore_waiters). Every semaphore write a
/// [`FenceAction`](crate::backend::FenceAction) performs is checked against them, once it landed
/// in memory: the ones it meets are latched and the GPU's front looks at them again, so a waiter
/// cannot miss a value that is overwritten before the front reads the word (NX2_NOTES row 70).
#[derive(Default)]
pub struct SemaphoreWaiters {
    count: AtomicUsize,
    list: Mutex<Vec<Waiter>>,
    front: OnceLock<Weak<Inbox>>,
    trace: OnceLock<Arc<QueueTrace>>,
    published: AtomicU64,
    current_dependency: AtomicU64,
    wake_after_publish: AtomicBool,
}

struct Waiter {
    channel: ChannelId,
    address: SemaphoreAddress,
    payload: u64,
    op: AcquireOp,
    wide: bool,
    latched: Option<Latch>,
    mode: WaitMode,
    trace_id: u64,
}

pub(crate) struct Acquire {
    pub address: SemaphoreAddress,
    pub payload: u64,
    pub op: AcquireOp,
    pub wide: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct SemaphoreAddress {
    pub low: Option<DAddr>,
    pub high: Option<DAddr>,
}

impl SemaphoreAddress {
    pub(crate) fn contiguous(self, wide: bool) -> Option<DAddr> {
        self.low.filter(|low| !wide || self.high == Some(low + 4))
    }
}

pub(crate) enum SemaphoreStatus {
    Waiting,
    Memory,
    Met { dependency: u64 },
}

#[derive(Clone, Copy)]
enum Latch {
    Queued(u64),
    Written(u64),
}

enum WaitMode {
    Memory,
    Unarmed,
    Ordered { first: u64, last: u64 },
}

impl Waiter {
    fn accepts(&self, fence: u64, published: u64) -> bool {
        match self.mode {
            WaitMode::Memory => true,
            WaitMode::Unarmed => false,
            WaitMode::Ordered { last, .. } if fence == 0 => last <= published,
            WaitMode::Ordered { first, .. } => first <= fence,
        }
    }
}

impl SemaphoreWaiters {
    pub(crate) fn attach_trace(&self, trace: Arc<QueueTrace>) {
        let _ = self.trace.set(trace);
    }

    /// Wakes `inbox` whenever a write latches a waiter.
    pub(crate) fn attach(&self, inbox: &Arc<Inbox>) {
        if self.front.set(Arc::downgrade(inbox)).is_err() {
            tracing::error!("semaphore waiters attached to a second GPU; it is never woken");
        }
    }

    /// Registers `channel` as waiting until the word at `address` meets the acquire.
    pub(crate) fn add(&self, channel: ChannelId, acquire: Acquire, ordered: bool, trace_id: u64) {
        let mut list = self.list.lock();
        list.push(Waiter {
            channel,
            address: acquire.address,
            payload: acquire.payload,
            op: acquire.op,
            wide: acquire.wide,
            latched: None,
            mode: if ordered {
                WaitMode::Unarmed
            } else {
                WaitMode::Memory
            },
            trace_id,
        });
        self.count.store(list.len(), Ordering::Release);
    }

    /// Drops `channel`'s registration.
    pub(crate) fn remove(&self, channel: ChannelId) {
        let mut list = self.list.lock();
        let Some(index) = list.iter().position(|w| w.channel == channel) else {
            return;
        };
        list.swap_remove(index);
        self.count.store(list.len(), Ordering::Release);
    }

    pub(crate) fn status(&self, channel: ChannelId) -> SemaphoreStatus {
        let list = self.list.lock();
        let published = self.published.load(Ordering::Acquire);
        let Some(waiter) = list.iter().find(|w| w.channel == channel) else {
            return SemaphoreStatus::Memory;
        };
        match (waiter.latched, &waiter.mode) {
            (Some(Latch::Queued(fence)), _) => SemaphoreStatus::Met {
                dependency: if fence > published { fence } else { 0 },
            },
            (Some(Latch::Written(fence)), _) if fence <= published => {
                SemaphoreStatus::Met { dependency: 0 }
            }
            (None, WaitMode::Memory) => SemaphoreStatus::Memory,
            (None, WaitMode::Ordered { last, .. }) if *last <= published => SemaphoreStatus::Memory,
            _ => SemaphoreStatus::Waiting,
        }
    }

    /// The back reached the acquire. A pending write at that point sets the oldest fence that
    /// can satisfy it and the earliest point at which a read of guest memory is valid.
    pub(crate) fn arm(&self, channel: ChannelId, fence: u64, value: Option<u64>) {
        let mut list = self.list.lock();
        let Some(waiter) = list.iter_mut().find(|w| w.channel == channel) else {
            return;
        };
        waiter.mode = WaitMode::Ordered {
            first: fence,
            last: fence,
        };
        if let Some(value) =
            value.filter(|&value| waiter.op.is_met(value, waiter.payload, waiter.wide))
        {
            self.trace_latch(waiter, fence, value, true);
            waiter.latched = Some(Latch::Queued(fence));
        }
        drop(list);
        self.wake_front();
    }

    /// A renderer queue entry has finished all its writes, including any mirror.
    pub(crate) fn publish(&self, fence: u64) {
        self.published.store(fence, Ordering::Release);
        if self.wake_after_publish.load(Ordering::Relaxed) {
            self.wake_after_publish.store(false, Ordering::Relaxed);
            self.wake_front();
        }
    }

    pub(crate) fn select_dependency(&self, fence: u64) {
        self.current_dependency.store(fence, Ordering::Relaxed);
    }

    /// The current GPU channel observed a queued release that hasn't reached guest memory yet.
    /// Called on the back thread, which also sets the channel's dependency.
    pub fn has_unpublished_dependency(&self) -> bool {
        let dependency = self.current_dependency.load(Ordering::Relaxed);
        dependency != 0 && dependency > self.published.load(Ordering::Acquire)
    }

    /// A write newly queued by the renderer can satisfy existing acquires without publishing
    /// the write to guest memory. An unknown value still bars a read of old RAM.
    pub(crate) fn queued(&self, fence: u64, observe: impl Fn(DAddr, bool) -> Option<Option<u64>>) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut latched = false;
        for waiter in self.list.lock().iter_mut() {
            let WaitMode::Ordered { first, last } = &mut waiter.mode else {
                continue;
            };
            if waiter.latched.is_some() || *first > fence {
                continue;
            }
            let observation = if let Some(address) = waiter.address.contiguous(waiter.wide) {
                observe(address, waiter.wide)
            } else if [waiter.address.low, waiter.address.high]
                .into_iter()
                .flatten()
                .any(|address| observe(address, false).is_some())
            {
                Some(None)
            } else {
                None
            };
            if let Some(value) = observation {
                *last = fence;
                if let Some(value) =
                    value.filter(|&value| waiter.op.is_met(value, waiter.payload, waiter.wide))
                {
                    self.trace_latch(waiter, fence, value, true);
                    waiter.latched = Some(Latch::Queued(fence));
                    latched = true;
                }
            }
        }
        if latched {
            self.wake_front();
        }
    }

    fn wake_front(&self) {
        if let Some(front) = self.front.get().and_then(Weak::upgrade) {
            front.wake();
        }
    }

    fn trace_latch(&self, waiter: &Waiter, fence: u64, value: u64, queued: bool) {
        if waiter.trace_id != 0
            && let Some(trace) = self.trace.get()
        {
            trace.record(Event::Latch(waiter.trace_id, fence, value, queued));
        }
    }

    fn wake_at(&self, fence: u64) {
        if fence == 0 {
            self.wake_front();
        } else {
            self.wake_after_publish.store(true, Ordering::Relaxed);
        }
    }

    /// An acquire of the 32-bit word at `address` now sees `value`, which memory does not hold
    /// (a mirrored Switch 2 queue semaphore value kept aside). Latches the narrow waiters on that
    /// word it meets.
    pub(crate) fn met_by(&self, address: DAddr, value: u32, fence: u64) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut latched = false;
        for waiter in self.list.lock().iter_mut() {
            if waiter.latched.is_some()
                || waiter.wide
                || waiter.address.low != Some(address)
                || !waiter.accepts(fence, self.published.load(Ordering::Relaxed))
            {
                continue;
            }
            if waiter.op.is_met(u64::from(value), waiter.payload, false) {
                self.trace_latch(waiter, fence, u64::from(value), false);
                waiter.latched = Some(Latch::Written(fence));
                latched = true;
            }
        }
        if latched {
            self.wake_at(fence);
        }
    }

    /// `[address, address + size)` of `device` was written. Latches the waiters on a word it
    /// overlaps that the word now meets, compared at the waiter's width: a 4-byte write never
    /// stands in for the 64-bit word a wide acquire reads.
    #[inline]
    pub(crate) fn written(
        &self,
        device: &DeviceAddressSpace,
        address: DAddr,
        size: u64,
        fence: u64,
    ) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut latched = false;
        for waiter in self.list.lock().iter_mut() {
            let Some(at) = waiter.address.contiguous(waiter.wide) else {
                continue;
            };
            let width = if waiter.wide { 8 } else { 4 };
            if waiter.latched.is_some()
                || at >= address.saturating_add(size)
                || address >= at.saturating_add(width)
                || !waiter.accepts(fence, self.published.load(Ordering::Relaxed))
            {
                continue;
            }
            let current = if waiter.wide {
                device.read::<u64>(at)
            } else {
                device.read::<u32>(at).map(u64::from)
            };
            if let Ok(current) = current
                && waiter.op.is_met(current, waiter.payload, waiter.wide)
            {
                self.trace_latch(waiter, fence, current, false);
                waiter.latched = Some(Latch::Written(fence));
                latched = true;
            }
        }
        if latched {
            self.wake_at(fence);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_write_does_not_resume_a_waiter_until_publication() {
        let waiters = SemaphoreWaiters::default();
        let channel = ChannelId(1);
        let word = 0x2000;
        waiters.add(
            channel,
            Acquire {
                address: SemaphoreAddress {
                    low: Some(word),
                    high: None,
                },
                payload: 1,
                op: AcquireOp::Equal,
                wide: false,
            },
            true,
            0,
        );
        waiters.arm(channel, 2, None);
        waiters.met_by(word, 1, 1);
        waiters.publish(1);
        assert!(matches!(waiters.status(channel), SemaphoreStatus::Waiting));
        waiters.met_by(word, 1, 2);
        assert!(matches!(waiters.status(channel), SemaphoreStatus::Waiting));
        waiters.publish(2);
        assert!(matches!(
            waiters.status(channel),
            SemaphoreStatus::Met { dependency: 0 }
        ));
    }
}
