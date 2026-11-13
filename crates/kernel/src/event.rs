//! Events (`KEvent` with its readable and writable ends).

use crate::resource_limit::Reservation;
use crate::sync::SyncState;

/// An event. Services hold an `Arc<Event>` and signal it from any host thread; the guest waits
/// on its readable end.
#[derive(Default)]
pub struct Event {
    sync: SyncState,
    observer: Option<Box<dyn Fn() + Send + Sync>>,
    _reservation: Option<Reservation>,
}

impl Event {
    pub fn new() -> Self {
        Self::default()
    }

    /// An event that runs `observer` on the signalling thread after every signal, for HLE servers
    /// that serve requests the guest posts by signalling an event it holds the writable end of.
    pub fn observed(observer: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            observer: Some(Box::new(observer)),
            ..Self::default()
        }
    }

    /// An event counted against a process's resource limit until it is destroyed.
    pub(crate) fn reserved(reservation: Reservation) -> Self {
        Self {
            _reservation: Some(reservation),
            ..Self::default()
        }
    }

    /// Signals the event, waking every waiter. It stays signaled until cleared.
    pub fn signal(&self) {
        self.sync.signal();
        if let Some(observer) = &self.observer {
            observer();
        }
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
