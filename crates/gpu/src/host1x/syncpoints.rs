//! host1x syncpoints: 32-bit counters the CPU and the GPU wait on.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::channel::SemaphoreWaiters;

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

/// Both syncpoint banks, as yuzu's SyncpointManager keeps them.
pub struct Syncpoints {
    guest: SyncpointBank,
    host: SyncpointBank,
    semaphores: Arc<SemaphoreWaiters>,
}

impl Syncpoints {
    pub fn new() -> Self {
        Self {
            guest: SyncpointBank::new(),
            host: SyncpointBank::new(),
            semaphores: Arc::default(),
        }
    }

    /// GPU channels waiting on semaphores in memory: semaphore writes that land through
    /// [`FenceAction::execute`](crate::backend::FenceAction::execute) latch and wake them.
    pub fn semaphore_waiters(&self) -> &Arc<SemaphoreWaiters> {
        &self.semaphores
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
            slot.wake(value);
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
            action();
            return Ok(None);
        }
        slot.interest.fetch_add(1, Ordering::SeqCst);
        let mut actions = slot.actions.lock();
        if is_reached(slot.value.load(Ordering::SeqCst), threshold) {
            drop(actions);
            slot.interest.fetch_sub(1, Ordering::SeqCst);
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
    #[cold]
    fn wake(&self, value: u32) {
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
}
