//! Events (`KEvent` with its readable and writable ends).

use crate::resource_limit::Reservation;
use crate::sync::SyncState;

/// An event. Services hold an `Arc<Event>` and signal it from any host thread; the guest waits
/// on its readable end.
#[derive(Default)]
pub struct Event {
    sync: SyncState,
    _reservation: Option<Reservation>,
}

impl Event {
    pub fn new() -> Self {
        Self::default()
    }

    /// An event counted against a process's resource limit until it is destroyed.
    pub(crate) fn reserved(reservation: Reservation) -> Self {
        Self {
            sync: SyncState::default(),
            _reservation: Some(reservation),
        }
    }

    /// Signals the event, waking every waiter. It stays signaled until cleared.
    pub fn signal(&self) {
        self.sync.signal();
    }

    /// Clears the signal (`ClearEvent`).
    pub fn clear(&self) {
        self.sync.reset();
    }

    /// Clears the signal and reports whether it was set (`ResetSignal`).
    pub fn reset(&self) -> bool {
        self.sync.reset()
    }

    pub fn is_signaled(&self) -> bool {
        self.sync.is_signaled()
    }

    pub(crate) fn sync_state(&self) -> &SyncState {
        &self.sync
    }
}
