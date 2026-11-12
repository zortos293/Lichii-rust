//! Events (`KEvent` with its readable and writable ends).

use crate::sync::SyncState;

/// An event. Services hold an `Arc<Event>` and signal it from any host thread; the guest waits
/// on its readable end.
#[derive(Default)]
pub struct Event {
    sync: SyncState,
}

impl Event {
    pub fn new() -> Self {
        Self::default()
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
