//! Kernel objects as handle-table entries.

use std::sync::Arc;

use crate::event::Event;
use crate::ipc::ClientSession;
use crate::memory_objects::{CodeMemory, SharedMemory, TransferMemory};
use crate::process::Process;
use crate::resource_limit::ResourceLimit;
use crate::sync::SyncState;
use crate::thread::Thread;

/// Anything a handle can refer to.
#[derive(Clone)]
pub enum KObject {
    Process(Arc<Process>),
    Thread(Arc<Thread>),
    /// The writable end of an event (`SignalEvent`).
    WritableEvent(Arc<Event>),
    /// The readable end of an event (waitable, `ClearEvent`/`ResetSignal`).
    ReadableEvent(Arc<Event>),
    SharedMemory(Arc<SharedMemory>),
    TransferMemory(Arc<TransferMemory>),
    CodeMemory(Arc<CodeMemory>),
    ResourceLimit(Arc<ResourceLimit>),
    /// The client end of a session to an HLE service.
    ClientSession(Arc<ClientSession>),
}

impl KObject {
    /// The wait state of objects `WaitSynchronization` accepts.
    pub(crate) fn sync(&self) -> Option<&SyncState> {
        match self {
            KObject::Process(p) => Some(p.sync_state()),
            KObject::Thread(t) => Some(&t.sync),
            KObject::ReadableEvent(e) => Some(e.sync_state()),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            KObject::Process(_) => "KProcess",
            KObject::Thread(_) => "KThread",
            KObject::WritableEvent(_) => "KEvent",
            KObject::ReadableEvent(_) => "KReadableEvent",
            KObject::SharedMemory(_) => "KSharedMemory",
            KObject::TransferMemory(_) => "KTransferMemory",
            KObject::CodeMemory(_) => "KCodeMemory",
            KObject::ResourceLimit(_) => "KResourceLimit",
            KObject::ClientSession(_) => "KClientSession",
        }
    }
}

/// Typed access to handle-table entries.
pub trait FromObject: Sized {
    fn from_object(object: &KObject) -> Option<Arc<Self>>;
}

macro_rules! from_object {
    ($ty:ty, $($variant:ident)|+) => {
        impl FromObject for $ty {
            fn from_object(object: &KObject) -> Option<Arc<Self>> {
                match object {
                    $(KObject::$variant(inner) => Some(inner.clone()),)+
                    _ => None,
                }
            }
        }
    };
}

from_object!(Process, Process);
from_object!(Thread, Thread);
from_object!(SharedMemory, SharedMemory);
from_object!(TransferMemory, TransferMemory);
from_object!(CodeMemory, CodeMemory);
from_object!(ResourceLimit, ResourceLimit);
from_object!(ClientSession, ClientSession);
