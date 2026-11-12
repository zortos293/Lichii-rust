//! GPU channels as the CPU side sees them: how a channel is created and what a submission
//! carries.
//!
//! A channel is one nvhost-gpu channel or NVN2 queue: a GPU address space, a host class, eight
//! subchannels and a queue of GPFIFO entries. Submissions go to the GPU's front thread in order
//! (see [`crate::gpu::Gpu`]).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use lichii_common::DAddr;
use lichii_memory::DeviceAddressSpace;
use lichii_texture::block_linear::GobLayout;
use parking_lot::Mutex;

pub use crate::backend::ChannelId;
use crate::gmmu::Gmmu;
use crate::gpu::Inbox;
use crate::host::{AcquireOp, HostClass};
pub use crate::pushbuffer::GpEntry;

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

/// One GPFIFO submission.
#[derive(Clone, Debug, Default)]
pub struct Submission {
    /// Pushbuffer segments, read from GPU memory when the front thread reaches them.
    pub entries: Vec<GpEntry>,
    /// `fence_wait`: hold the entries until this syncpoint value is reached.
    pub wait: Option<SyncpointFence>,
    /// Increments after the entries.
    pub increment: Option<SyncpointIncrement>,
}

impl Submission {
    /// A submission of `entries` without fences.
    pub fn new(entries: Vec<GpEntry>) -> Self {
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
}

struct Waiter {
    channel: ChannelId,
    address: DAddr,
    payload: u64,
    op: AcquireOp,
    wide: bool,
    latched: bool,
}

impl SemaphoreWaiters {
    /// Wakes `inbox` whenever a write latches a waiter.
    pub(crate) fn attach(&self, inbox: &Arc<Inbox>) {
        if self.front.set(Arc::downgrade(inbox)).is_err() {
            tracing::error!("semaphore waiters attached to a second GPU; it is never woken");
        }
    }

    /// Registers `channel` as waiting until the word at `address` meets the acquire.
    pub(crate) fn add(
        &self,
        channel: ChannelId,
        address: DAddr,
        payload: u64,
        op: AcquireOp,
        wide: bool,
    ) {
        let mut list = self.list.lock();
        list.push(Waiter {
            channel,
            address,
            payload,
            op,
            wide,
            latched: false,
        });
        self.count.store(list.len(), Ordering::Release);
    }

    /// Drops `channel`'s registration; returns whether a write latched it.
    pub(crate) fn remove(&self, channel: ChannelId) -> bool {
        let mut list = self.list.lock();
        let Some(index) = list.iter().position(|w| w.channel == channel) else {
            return false;
        };
        let latched = list.swap_remove(index).latched;
        self.count.store(list.len(), Ordering::Release);
        latched
    }

    /// Whether a write latched `channel`.
    pub(crate) fn latched(&self, channel: ChannelId) -> bool {
        self.list
            .lock()
            .iter()
            .any(|w| w.channel == channel && w.latched)
    }

    /// `[address, address + size)` of `device` was written. Latches the waiters on a word it
    /// overlaps that the word now meets, compared at the waiter's width: a 4-byte write never
    /// stands in for the 64-bit word a wide acquire reads.
    #[inline]
    pub(crate) fn written(&self, device: &DeviceAddressSpace, address: DAddr, size: u64) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut latched = false;
        for waiter in self.list.lock().iter_mut() {
            let width = if waiter.wide { 8 } else { 4 };
            if waiter.latched
                || waiter.address >= address.saturating_add(size)
                || address >= waiter.address.saturating_add(width)
            {
                continue;
            }
            let current = if waiter.wide {
                device.read::<u64>(waiter.address)
            } else {
                device.read::<u32>(waiter.address).map(u64::from)
            };
            if current.is_ok_and(|current| waiter.op.is_met(current, waiter.payload, waiter.wide)) {
                waiter.latched = true;
                latched = true;
            }
        }
        if latched && let Some(front) = self.front.get().and_then(Weak::upgrade) {
            front.wake();
        }
    }
}
