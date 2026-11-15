//! host1x syncpoints: 32-bit counters the CPU and the GPU wait on.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::channel::SemaphoreWaiters;
use crate::semaphore_mirror::SemaphoreMirror;

/// Syncpoint ids per bank. Switch 2 ids are below 0x400; Switch 1 uses the first 192.
pub const SYNCPOINT_COUNT: usize = 1024;

/// A syncpoint id at or above [`SYNCPOINT_COUNT`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("syncpoint id {0:#x} is out of range")]
pub struct InvalidSyncpoint(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SyncpointWaitError {
    #[error(transparent)]
    InvalidId(#[from] InvalidSyncpoint),
    #[error("syncpoint wait timed out at value {value:#x}")]
    TimedOut { value: u32 },
}

/// Identifies a registered action for [`SyncpointBank::cancel`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionHandle {
    id: u32,
    serial: u64,
}

/// Identifies a mirror for [`SyncpointBank::clear_mirror`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MirrorHandle {
    serial: u64,
    count: u32,
}

type Mirror = Arc<dyn Fn(u32, u32) + Send + Sync>;

/// Both syncpoint banks, as yuzu's SyncpointManager keeps them, and the maximum value each id is
/// expected to reach: the one record nvhost-ctrl, nvhost-gpu submits and the Switch 2 syncpoint
/// sessions share.
pub struct Syncpoints {
    guest: SyncpointBank,
    host: SyncpointBank,
    max: Box<[AtomicU32]>,
    semaphores: Arc<SemaphoreWaiters>,
    mirror: Arc<SemaphoreMirror>,
}

impl Syncpoints {
    pub fn new() -> Self {
        Self {
            guest: SyncpointBank::new(),
            host: SyncpointBank::new(),
            max: (0..SYNCPOINT_COUNT).map(|_| AtomicU32::new(0)).collect(),
            semaphores: Arc::default(),
            mirror: Arc::default(),
        }
    }

    /// The value `id` reaches once every increment reserved so far ran.
    #[inline]
    pub fn max(&self, id: u32) -> Result<u32, InvalidSyncpoint> {
        Ok(self.max_slot(id)?.load(Ordering::Acquire))
    }

    /// Reserves `count` increments of `id` and returns the new maximum (wrapping), the value a
    /// fence for them waits for.
    #[inline]
    pub fn increase_max(&self, id: u32, count: u32) -> Result<u32, InvalidSyncpoint> {
        Ok(self
            .max_slot(id)?
            .fetch_add(count, Ordering::AcqRel)
            .wrapping_add(count))
    }

    fn max_slot(&self, id: u32) -> Result<&AtomicU32, InvalidSyncpoint> {
        self.max.get(id as usize).ok_or(InvalidSyncpoint(id))
    }

    /// GPU channels waiting on semaphores in memory: semaphore writes that land through
    /// [`FenceAction::execute`](crate::backend::FenceAction::execute) latch and wake them.
    pub fn semaphore_waiters(&self) -> &Arc<SemaphoreWaiters> {
        &self.semaphores
    }

    /// Switch 2 queue semaphore regions, whose second-page releases also land in the first page.
    pub fn semaphore_mirror(&self) -> &Arc<SemaphoreMirror> {
        &self.mirror
    }

    /// Values the guest sees: incremented once the work before an increment has completed on
    /// the host GPU. nvhost-ctrl waits, fences and the Switch 2 syncpoint shim read these.
    pub fn guest(&self) -> &SyncpointBank {
        &self.guest
    }

    /// Values the command processor has reached: incremented when it executes an increment,
    /// before the host GPU work behind it has finished.
    pub fn host(&self) -> &SyncpointBank {
        &self.host
    }
}

impl Default for Syncpoints {
    fn default() -> Self {
        Self::new()
    }
}

/// [`SYNCPOINT_COUNT`] atomic counters. An increment takes no lock unless an action or a waiter
/// is registered on that id, and then only that id's lock.
pub struct SyncpointBank {
    slots: Box<[Slot]>,
    next_serial: AtomicU64,
}

struct Slot {
    value: AtomicU32,
    interest: AtomicU32,
    actions: Mutex<Vec<PendingAction>>,
    changed: Condvar,
    mirror: Mutex<Option<(u64, Mirror)>>,
}

struct PendingAction {
    serial: u64,
    threshold: u32,
    run: Box<dyn FnOnce() + Send>,
}

/// Whether `value` has reached `threshold`, comparing as host1x does so that counters can wrap.
#[inline]
pub const fn is_reached(value: u32, threshold: u32) -> bool {
    value.wrapping_sub(threshold) as i32 >= 0
}

impl SyncpointBank {
    fn new() -> Self {
        Self {
            slots: (0..SYNCPOINT_COUNT)
                .map(|_| Slot {
                    value: AtomicU32::new(0),
                    interest: AtomicU32::new(0),
                    actions: Mutex::new(Vec::new()),
                    changed: Condvar::new(),
                    mirror: Mutex::new(None),
                })
                .collect(),
            next_serial: AtomicU64::new(0),
        }
    }

    #[inline]
    pub fn value(&self, id: u32) -> Result<u32, InvalidSyncpoint> {
        Ok(self.slot(id)?.value.load(Ordering::Acquire))
    }

    #[inline]
    pub fn is_reached(&self, id: u32, threshold: u32) -> Result<bool, InvalidSyncpoint> {
        Ok(is_reached(self.value(id)?, threshold))
    }

    /// Adds one and returns the new value. Actions it satisfies run on this thread.
    #[inline]
    pub fn increment(&self, id: u32) -> Result<u32, InvalidSyncpoint> {
        let slot = self.slot(id)?;
        let value = slot.value.fetch_add(1, Ordering::SeqCst).wrapping_add(1);
        if slot.interest.load(Ordering::SeqCst) != 0 {
            slot.wake(id, value);
        }
        Ok(value)
    }

    /// Blocks until `id` reaches `threshold` or `timeout` passes; returns the value seen.
    pub fn wait(
        &self,
        id: u32,
        threshold: u32,
        timeout: Option<Duration>,
    ) -> Result<u32, SyncpointWaitError> {
        let slot = self.slot(id)?;
        let value = slot.value.load(Ordering::Acquire);
        if is_reached(value, threshold) {
            return Ok(value);
        }
        let deadline = timeout.map(|t| Instant::now() + t);
        slot.interest.fetch_add(1, Ordering::SeqCst);
        let mut guard = slot.actions.lock();
        let result = loop {
            let value = slot.value.load(Ordering::SeqCst);
            if is_reached(value, threshold) {
                break Ok(value);
            }
            match deadline {
                None => slot.changed.wait(&mut guard),
                Some(deadline) => {
                    if slot.changed.wait_until(&mut guard, deadline).timed_out() {
                        let value = slot.value.load(Ordering::SeqCst);
                        break if is_reached(value, threshold) {
                            Ok(value)
                        } else {
                            Err(SyncpointWaitError::TimedOut { value })
                        };
                    }
                }
            }
        };
        drop(guard);
        slot.interest.fetch_sub(1, Ordering::SeqCst);
        result
    }

    /// Runs `action` once `id` reaches `threshold`, on whichever thread increments it there.
    /// Runs it right away and returns `None` when the value is already there.
    pub fn register_action(
        &self,
        id: u32,
        threshold: u32,
        action: impl FnOnce() + Send + 'static,
    ) -> Result<Option<ActionHandle>, InvalidSyncpoint> {
        let slot = self.slot(id)?;
        if is_reached(slot.value.load(Ordering::Acquire), threshold) {
            slot.publish(id);
            action();
            return Ok(None);
        }
        slot.interest.fetch_add(1, Ordering::SeqCst);
        let mut actions = slot.actions.lock();
        if is_reached(slot.value.load(Ordering::SeqCst), threshold) {
            drop(actions);
            slot.interest.fetch_sub(1, Ordering::SeqCst);
            slot.publish(id);
            action();
            return Ok(None);
        }
        let serial = self.next_serial.fetch_add(1, Ordering::Relaxed);
        actions.push(PendingAction {
            serial,
            threshold,
            run: Box::new(action),
        });
        Ok(Some(ActionHandle { id, serial }))
    }

    /// Calls `mirror(id, value)` after each increment of every id below `count`, with the value
    /// the id holds then, before the actions that increment satisfies run, and again before an
    /// action registered on an already reached value runs. Calls for one id never overlap.
    /// Replaces the mirror those ids had.
    pub fn set_mirror(
        &self,
        count: u32,
        mirror: impl Fn(u32, u32) + Send + Sync + 'static,
    ) -> MirrorHandle {
        let serial = self.next_serial.fetch_add(1, Ordering::Relaxed);
        let mirror: Mirror = Arc::new(mirror);
        for slot in self.slots.iter().take(count as usize) {
            if slot
                .mirror
                .lock()
                .replace((serial, mirror.clone()))
                .is_none()
            {
                slot.interest.fetch_add(1, Ordering::SeqCst);
            }
        }
        MirrorHandle { serial, count }
    }

    /// Runs `id`'s mirror, if it has one, with the value it holds now. It runs in turn with the
    /// mirror runs of increments, so a value read before an increment never lands after it.
    pub fn publish(&self, id: u32) -> Result<(), InvalidSyncpoint> {
        self.slot(id)?.publish(id);
        Ok(())
    }

    /// Drops the mirror `handle` names from the ids no later mirror replaced it on.
    pub fn clear_mirror(&self, handle: MirrorHandle) {
        for slot in self.slots.iter().take(handle.count as usize) {
            let mut mirror = slot.mirror.lock();
            if mirror
                .as_ref()
                .is_some_and(|(serial, _)| *serial == handle.serial)
            {
                *mirror = None;
                slot.interest.fetch_sub(1, Ordering::SeqCst);
            }
        }
    }

    /// The threshold of the registered action on `id` that is nearest ahead of its value.
    pub fn next_action_threshold(&self, id: u32) -> Result<Option<u32>, InvalidSyncpoint> {
        let slot = self.slot(id)?;
        if slot.interest.load(Ordering::Acquire) == 0 {
            return Ok(None);
        }
        let actions = slot.actions.lock();
        let value = slot.value.load(Ordering::Acquire);
        Ok(actions
            .iter()
            .map(|a| a.threshold)
            .min_by_key(|threshold| threshold.wrapping_sub(value)))
    }

    /// Drops a registered action. False when it already ran (or is running).
    pub fn cancel(&self, handle: ActionHandle) -> bool {
        let Ok(slot) = self.slot(handle.id) else {
            return false;
        };
        let mut actions = slot.actions.lock();
        let Some(index) = actions.iter().position(|a| a.serial == handle.serial) else {
            return false;
        };
        actions.swap_remove(index);
        slot.interest.fetch_sub(1, Ordering::SeqCst);
        true
    }

    #[inline]
    fn slot(&self, id: u32) -> Result<&Slot, InvalidSyncpoint> {
        self.slots.get(id as usize).ok_or(InvalidSyncpoint(id))
    }
}

impl Slot {
    fn publish(&self, id: u32) {
        if let Some((_, mirror)) = &*self.mirror.lock() {
            mirror(id, self.value.load(Ordering::SeqCst));
        }
    }

    #[cold]
    fn wake(&self, id: u32, value: u32) {
        self.publish(id);
        let ready: Vec<PendingAction> = {
            let mut actions = self.actions.lock();
            self.changed.notify_all();
            actions
                .extract_if(.., |a| is_reached(value, a.threshold))
                .collect()
        };
        for action in ready {
            self.interest.fetch_sub(1, Ordering::SeqCst);
            (action.run)();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn increments_and_reads_back() {
        let syncpoints = Syncpoints::new();
        assert_eq!(syncpoints.guest().increment(5), Ok(1));
        assert_eq!(syncpoints.guest().increment(5), Ok(2));
        assert_eq!(syncpoints.guest().value(5), Ok(2));
        assert_eq!(syncpoints.host().value(5), Ok(0));
        assert_eq!(syncpoints.guest().value(0x3ff), Ok(0));
        assert_eq!(
            syncpoints.guest().increment(0x400),
            Err(InvalidSyncpoint(0x400))
        );
    }

    #[test]
    fn max_values_are_reserved_per_id() {
        let syncpoints = Syncpoints::new();
        assert_eq!(syncpoints.increase_max(3, 2), Ok(2));
        assert_eq!(syncpoints.increase_max(3, 1), Ok(3));
        assert_eq!(syncpoints.max(3), Ok(3));
        assert_eq!(syncpoints.max(4), Ok(0));
        assert_eq!(syncpoints.guest().value(3), Ok(0));
        assert_eq!(
            syncpoints.increase_max(0x400, 1),
            Err(InvalidSyncpoint(0x400))
        );
        assert_eq!(syncpoints.max(0x400), Err(InvalidSyncpoint(0x400)));
    }

    #[test]
    fn comparison_wraps() {
        assert!(is_reached(5, 5));
        assert!(!is_reached(4, 5));
        assert!(is_reached(2, u32::MAX - 1));
        assert!(!is_reached(u32::MAX - 1, 2));
    }

    #[test]
    fn wait_returns_when_another_thread_increments() {
        let syncpoints = Arc::new(Syncpoints::new());
        let incrementer = {
            let syncpoints = syncpoints.clone();
            std::thread::spawn(move || {
                for _ in 0..3 {
                    std::thread::sleep(Duration::from_millis(5));
                    syncpoints.guest().increment(1).unwrap();
                }
            })
        };
        let value = syncpoints
            .guest()
            .wait(1, 3, Some(Duration::from_secs(10)))
            .unwrap();
        assert!(value >= 3);
        incrementer.join().unwrap();
        assert_eq!(syncpoints.guest().wait(1, 2, None), Ok(3));
    }

    #[test]
    fn wait_times_out() {
        let syncpoints = Syncpoints::new();
        syncpoints.host().increment(2).unwrap();
        assert_eq!(
            syncpoints
                .host()
                .wait(2, 2, Some(Duration::from_millis(10))),
            Err(SyncpointWaitError::TimedOut { value: 1 })
        );
        assert_eq!(
            syncpoints.host().wait(0x500, 0, None),
            Err(SyncpointWaitError::InvalidId(InvalidSyncpoint(0x500)))
        );
    }

    #[test]
    fn actions_run_at_threshold_and_can_be_cancelled() {
        let bank = Syncpoints::new();
        let bank = bank.guest();
        let ran = Arc::new(AtomicUsize::new(0));
        let counter = |n| {
            let ran = ran.clone();
            move || {
                ran.fetch_add(n, Ordering::SeqCst);
            }
        };
        let first = bank.register_action(7, 2, counter(1)).unwrap().unwrap();
        let cancelled = bank.register_action(7, 2, counter(100)).unwrap().unwrap();
        bank.register_action(7, 3, counter(10)).unwrap().unwrap();
        assert!(bank.cancel(cancelled));
        bank.increment(7).unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 0);
        bank.increment(7).unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 1);
        assert!(!bank.cancel(first));
        bank.increment(7).unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 11);
        assert_eq!(bank.register_action(7, 3, counter(1000)), Ok(None));
        assert_eq!(ran.load(Ordering::SeqCst), 1011);
    }

    #[test]
    fn actions_race_with_increments() {
        let syncpoints = Arc::new(Syncpoints::new());
        let ran = Arc::new(AtomicUsize::new(0));
        let incrementer = {
            let syncpoints = syncpoints.clone();
            std::thread::spawn(move || {
                for _ in 0..2000 {
                    syncpoints.host().increment(3).unwrap();
                }
            })
        };
        for threshold in 1..=2000 {
            let ran = ran.clone();
            syncpoints
                .host()
                .register_action(3, threshold, move || {
                    ran.fetch_add(1, Ordering::SeqCst);
                })
                .unwrap();
        }
        incrementer.join().unwrap();
        assert_eq!(ran.load(Ordering::SeqCst), 2000);
    }

    #[test]
    fn mirrors_see_each_increment_before_its_actions() {
        let bank = Syncpoints::new();
        let bank = bank.guest();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let record = |tag: &'static str| {
            let seen = seen.clone();
            move |id, value| seen.lock().push((tag, id, value))
        };
        let first = bank.set_mirror(8, record("first"));
        let action = seen.clone();
        bank.register_action(3, 2, move || action.lock().push(("action", 3, 2)))
            .unwrap();
        bank.increment(3).unwrap();
        bank.increment(3).unwrap();
        bank.increment(8).unwrap();
        assert_eq!(
            *seen.lock(),
            [("first", 3, 1), ("first", 3, 2), ("action", 3, 2)],
            "ids past the count are not mirrored"
        );

        seen.lock().clear();
        let second = bank.set_mirror(4, record("second"));
        bank.clear_mirror(first);
        bank.increment(3).unwrap();
        bank.increment(5).unwrap();
        assert_eq!(*seen.lock(), [("second", 3, 3)]);

        bank.clear_mirror(second);
        bank.increment(3).unwrap();
        assert_eq!(seen.lock().len(), 1);
        assert_eq!(bank.next_action_threshold(3), Ok(None));
    }

    #[test]
    fn an_action_on_a_reached_value_runs_after_the_mirror() {
        let syncpoints = Syncpoints::new();
        let bank = syncpoints.guest();
        bank.increment(4).unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mirror = seen.clone();
        bank.set_mirror(8, move |id, value| mirror.lock().push((id, value)));
        let action = seen.clone();
        bank.register_action(4, 1, move || action.lock().push((99, 0)))
            .unwrap();
        assert_eq!(
            *seen.lock(),
            [(4, 1), (99, 0)],
            "the incrementing thread may not have mirrored yet"
        );
    }
}
