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
use crate::backend::ReleaseSize;
use crate::gmmu::Gmmu;
use crate::gpu::{Console, Inbox};
use crate::host::{AcquireOp, HostClass};
pub use crate::pushbuffer::GpEntry;
use crate::semaphore_mirror::{self, SemaphoreMirror};

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
///
/// They also hold the releases the back executed that have not landed yet: acquires see those
/// values at once, as syncpoint waits see the host bank, so a channel gated on another channel's
/// release never waits for the host GPU to finish the work before it.
#[derive(Default)]
pub struct SemaphoreWaiters {
    count: AtomicUsize,
    list: Mutex<Vec<Waiter>>,
    front: OnceLock<Weak<Inbox>>,
    in_flight: AtomicUsize,
    executed: Mutex<Executed>,
}

const IN_FLIGHT: usize = 32;

/// Executed releases on their way to memory, at most one per byte. A full table evicts, which
/// only sends an acquire back to memory.
#[derive(Default)]
struct Executed {
    entries: [Option<Release>; IN_FLIGHT],
    next: usize,
}

#[derive(Clone, Copy)]
struct Release {
    address: DAddr,
    value: u64,
    bytes: u64,
}

impl Executed {
    fn record(&mut self, address: DAddr, value: Option<u64>, bytes: u64) {
        for slot in &mut self.entries {
            if slot.is_some_and(|e| e.address < address + bytes && address < e.address + e.bytes) {
                *slot = None;
            }
        }
        let Some(value) = value else {
            return;
        };
        let index = self
            .entries
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| {
                let index = self.next;
                self.next = (index + 1) % IN_FLIGHT;
                index
            });
        self.entries[index] = Some(Release {
            address,
            value,
            bytes,
        });
    }

    fn len(&self) -> usize {
        self.entries.iter().flatten().count()
    }
}

fn release_bytes(size: ReleaseSize) -> u64 {
    match size {
        ReleaseSize::Four => 4,
        ReleaseSize::Eight => 8,
        ReleaseSize::Sixteen => 16,
    }
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

    /// The back executed a write of `size` to `address` that lands once the host GPU finished
    /// the work before it: a release of `value`, or, when `None`, a write whose value is known
    /// only then (a reduction, a counter). Until it lands, acquires see the release's value and
    /// the waiters it meets resume now: the GPU work a resumed channel queues next follows it on
    /// the host GPU, as after a syncpoint wait. The guest CPU still sees it only once it lands. A
    /// release into a Switch 2 queue semaphore's second page counts for its first page too.
    pub(crate) fn executed(
        &self,
        mirror: &SemaphoreMirror,
        address: DAddr,
        value: Option<u64>,
        size: ReleaseSize,
    ) {
        let bytes = release_bytes(size);
        let first_page = mirror
            .mirror_of(address)
            .is_some()
            .then(|| address - semaphore_mirror::PAGE);
        {
            let mut executed = self.executed.lock();
            executed.record(address, value, bytes);
            if let Some(first_page) = first_page {
                executed.record(first_page, value, bytes);
            }
            self.in_flight.store(executed.len(), Ordering::Release);
        }
        if let Some(value) = value {
            self.latch_released(address, value, size);
            if let Some(first_page) = first_page {
                self.latch_released(first_page, value, size);
            }
        }
    }

    /// A release of `value` to `address` landed: memory holds what acquires saw until now.
    pub(crate) fn landed(&self, address: DAddr, value: u64) {
        if self.in_flight.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut executed = self.executed.lock();
        for slot in &mut executed.entries {
            if slot.is_some_and(|e| e.address == address && e.value == value) {
                *slot = None;
            }
        }
        self.in_flight.store(executed.len(), Ordering::Release);
    }

    /// The value an acquire of the word (or with `wide`, the 64-bit word) at `address` sees from
    /// an executed release that has not landed yet, if one wrote all of it.
    pub(crate) fn in_flight(&self, address: DAddr, wide: bool) -> Option<u64> {
        if self.in_flight.load(Ordering::Acquire) == 0 {
            return None;
        }
        let width = if wide { 8 } else { 4 };
        let executed = self.executed.lock();
        let release = executed
            .entries
            .iter()
            .flatten()
            .find(|e| e.address < address + width && address < e.address + e.bytes)?;
        if release.address == address && (!wide || release.bytes >= 8) {
            Some(if wide {
                release.value
            } else {
                release.value & 0xffff_ffff
            })
        } else if !wide && release.address + 4 == address && release.bytes >= 8 {
            Some(release.value >> 32)
        } else {
            None
        }
    }

    fn latch_released(&self, address: DAddr, value: u64, size: ReleaseSize) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let wide_release = size != ReleaseSize::Four;
        let mut latched = false;
        for waiter in self.list.lock().iter_mut() {
            let current = if waiter.wide {
                (wide_release && waiter.address == address).then_some(value)
            } else if waiter.address & !3 == address {
                Some(value & 0xffff_ffff)
            } else if wide_release && waiter.address & !3 == address + 4 {
                Some(value >> 32)
            } else {
                None
            };
            if !waiter.latched
                && current
                    .is_some_and(|current| waiter.op.is_met(current, waiter.payload, waiter.wide))
            {
                waiter.latched = true;
                latched = true;
            }
        }
        if latched && let Some(front) = self.front.get().and_then(Weak::upgrade) {
            front.wake();
        }
    }

    /// An acquire of the 32-bit word at `address` now sees `value`, which memory does not hold
    /// (a mirrored Switch 2 queue semaphore value kept aside). Latches the narrow waiters on that
    /// word it meets.
    pub(crate) fn met_by(&self, address: DAddr, value: u32) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut latched = false;
        for waiter in self.list.lock().iter_mut() {
            if waiter.latched || waiter.wide || waiter.address & !3 != address & !3 {
                continue;
            }
            if waiter.op.is_met(u64::from(value), waiter.payload, false) {
                waiter.latched = true;
                latched = true;
            }
        }
        if latched && let Some(front) = self.front.get().and_then(Weak::upgrade) {
            front.wake();
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    const WORD: DAddr = 0x1000_0000;

    #[test]
    fn acquires_see_an_executed_release_until_it_lands_with_its_value() {
        let waiters = SemaphoreWaiters::default();
        let mirror = SemaphoreMirror::default();
        waiters.executed(&mirror, WORD, Some(7 << 32 | 3), ReleaseSize::Eight);
        assert_eq!(waiters.in_flight(WORD, false), Some(3));
        assert_eq!(waiters.in_flight(WORD + 4, false), Some(7));
        assert_eq!(waiters.in_flight(WORD, true), Some(7 << 32 | 3));
        waiters.landed(WORD, 3);
        assert_eq!(waiters.in_flight(WORD, true), Some(7 << 32 | 3));
        waiters.landed(WORD, 7 << 32 | 3);
        assert_eq!(waiters.in_flight(WORD, false), None);
    }

    #[test]
    fn a_later_write_replaces_the_releases_it_overlaps() {
        let waiters = SemaphoreWaiters::default();
        let mirror = SemaphoreMirror::default();
        waiters.executed(&mirror, WORD, Some(1), ReleaseSize::Eight);
        waiters.executed(&mirror, WORD + 4, Some(2), ReleaseSize::Four);
        assert_eq!(waiters.in_flight(WORD, true), None);
        assert_eq!(waiters.in_flight(WORD, false), None);
        assert_eq!(waiters.in_flight(WORD + 4, false), Some(2));
        waiters.executed(&mirror, WORD + 4, None, ReleaseSize::Four);
        assert_eq!(waiters.in_flight(WORD + 4, false), None);
    }

    #[test]
    fn a_full_table_forgets_its_oldest_release() {
        let waiters = SemaphoreWaiters::default();
        let mirror = SemaphoreMirror::default();
        for index in 0..=IN_FLIGHT as u64 {
            waiters.executed(&mirror, WORD + 4 * index, Some(index), ReleaseSize::Four);
        }
        assert_eq!(waiters.in_flight(WORD, false), None);
        assert_eq!(waiters.in_flight(WORD + 4, false), Some(1));
        let last = IN_FLIGHT as u64;
        assert_eq!(waiters.in_flight(WORD + 4 * last, false), Some(last));
    }
}
