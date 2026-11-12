use std::sync::Arc;

use lichii_common::HleResult;

use super::{Flow, SvcContext, deadline};
use crate::arbiter::{ArbitrationType, SignalType, WaitOutcome};
use crate::event::Event;
use crate::object::KObject;
use crate::resource_limit::LimitableResource;
use crate::result;
use crate::sync::Waiter;
use crate::thread::{Resume, Thread};

const MAX_WAIT_OBJECTS: usize = 64;

/// `svcSignalEvent(handle w0)`: the writable end.
pub(super) fn signal_event(svc: &mut SvcContext) -> Flow {
    let result = match svc.object(svc.arg32(0)) {
        Some(KObject::WritableEvent(event)) => {
            event.signal();
            Ok(())
        }
        _ => Err(result::INVALID_HANDLE),
    };
    svc.finish(result)
}

/// `svcClearEvent(handle w0)`: either end.
pub(super) fn clear_event(svc: &mut SvcContext) -> Flow {
    let result = match svc.object(svc.arg32(0)) {
        Some(KObject::WritableEvent(event) | KObject::ReadableEvent(event)) => {
            event.clear();
            Ok(())
        }
        _ => Err(result::INVALID_HANDLE),
    };
    svc.finish(result)
}

/// `svcResetSignal(handle w0)`: readable events and processes; fails when not signaled.
pub(super) fn reset_signal(svc: &mut SvcContext) -> Flow {
    let reset = match svc.object(svc.arg32(0)) {
        Some(KObject::ReadableEvent(event)) => Ok(event.reset()),
        Some(KObject::Process(process)) => Ok(process.sync_state().reset()),
        _ => Err(result::INVALID_HANDLE),
    };
    let result = reset.and_then(|was| {
        if was {
            Ok(())
        } else {
            Err(result::INVALID_STATE)
        }
    });
    svc.finish(result)
}

/// `svcCloseHandle(handle w0)`.
pub(super) fn close_handle(svc: &mut SvcContext) -> Flow {
    let removed = svc.process().handles().write().remove(svc.arg32(0));
    let result = if removed.is_some() {
        Ok(())
    } else {
        Err(result::INVALID_HANDLE)
    };
    drop(removed);
    svc.finish(result)
}

/// `svcCreateEvent` -> `w1` writable, `w2` readable.
pub(super) fn create_event(svc: &mut SvcContext) -> Flow {
    let result = (|| {
        let process = svc.process();
        process
            .resource_limit()
            .reserve(LimitableResource::Events, 1)?;
        let event = Arc::new(Event::new());
        let mut handles = process.handles().write();
        let writable = handles.add(KObject::WritableEvent(event.clone()))?;
        let readable = match handles.add(KObject::ReadableEvent(event)) {
            Ok(handle) => handle,
            Err(code) => {
                handles.remove(writable);
                return Err(code);
            }
        };
        Ok((writable, readable))
    })();
    match result {
        Ok((writable, readable)) => {
            svc.set(1, writable as u64);
            svc.set(2, readable as u64);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcWaitSynchronization(handles x1, count w2, timeout x3)` -> `w1` index.
pub(super) fn wait_synchronization(svc: &mut SvcContext) -> Flow {
    let (list, count, timeout) = (svc.arg(1), svc.arg32(2) as i32, svc.arg(3) as i64);
    svc.set(1, u32::MAX as u64);
    if !(0..=MAX_WAIT_OBJECTS as i32).contains(&count) {
        return svc.finish(Err(result::OUT_OF_RANGE));
    }
    let count = count as usize;
    let mut objects: [Option<KObject>; MAX_WAIT_OBJECTS] = [const { None }; MAX_WAIT_OBJECTS];
    for (i, slot) in objects.iter_mut().enumerate().take(count) {
        let Ok(handle) = svc.process().memory().read::<u32>(list + 4 * i as u64) else {
            return svc.finish(Err(result::INVALID_POINTER));
        };
        match svc.object(handle) {
            Some(object) if object.sync().is_some() => *slot = Some(object),
            _ => return svc.finish(Err(result::INVALID_HANDLE)),
        }
    }
    let objects = &objects[..count];
    let sync = |i: usize| {
        objects[i]
            .as_ref()
            .and_then(KObject::sync)
            .expect("waitable")
    };
    if timeout == 0 {
        let signaled = (0..count).find(|&i| sync(i).is_signaled());
        return match signaled {
            Some(i) => {
                svc.set(1, i as u64);
                svc.finish(Ok(()))
            }
            None => svc.finish(Err(result::TIMED_OUT)),
        };
    }
    let token = match svc.thread.begin_wait(Resume::ResultAndIndex, true) {
        Ok(token) => token,
        Err(code) => return svc.finish(Err(code)),
    };
    for i in 0..count {
        let waiter = Waiter {
            thread: svc.thread.clone(),
            token,
            index: i as i32,
        };
        if sync(i).add_waiter_or_signaled(waiter) {
            svc.thread.cancel_wait(token);
            svc.set(1, i as u64);
            return svc.finish(Ok(()));
        }
    }
    svc.block(token, deadline(timeout), false)
}

/// `svcCancelSynchronization(thread handle w0)`.
pub(super) fn cancel_synchronization(svc: &mut SvcContext) -> Flow {
    let result = svc
        .typed::<Thread>(svc.arg32(0))
        .map(|thread| thread.cancel_synchronization());
    svc.finish(result)
}

fn after_wait(svc: &mut SvcContext, outcome: HleResult<WaitOutcome>, timeout: i64) -> Flow {
    match outcome {
        Ok(WaitOutcome::Done(code)) => {
            svc.set_result(code);
            Flow::Continue
        }
        Ok(WaitOutcome::Block(token)) => svc.block(token, deadline(timeout), false),
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcArbitrateLock(owner w0, addr x1, tag w2)`.
pub(super) fn arbitrate_lock(svc: &mut SvcContext) -> Flow {
    let outcome =
        svc.kernel
            .arbiter
            .arbitrate_lock(svc.thread, svc.arg32(0), svc.arg(1), svc.arg32(2));
    after_wait(svc, outcome, -1)
}

/// `svcArbitrateUnlock(addr x0)`.
pub(super) fn arbitrate_unlock(svc: &mut SvcContext) -> Flow {
    let result = svc
        .kernel
        .arbiter
        .arbitrate_unlock(svc.process(), svc.arg(0));
    svc.finish(result)
}

/// `svcWaitProcessWideKeyAtomic(addr x0, key x1, tag w2, timeout x3)`.
pub(super) fn wait_process_wide_key_atomic(svc: &mut SvcContext) -> Flow {
    let timeout = svc.arg(3) as i64;
    let outcome = svc.kernel.arbiter.wait_condvar(
        svc.thread,
        svc.arg(0),
        svc.arg(1),
        svc.arg32(2),
        timeout == 0,
    );
    after_wait(svc, outcome, timeout)
}

/// `svcSignalProcessWideKey(key x0, count w1)`.
pub(super) fn signal_process_wide_key(svc: &mut SvcContext) -> Flow {
    svc.kernel
        .arbiter
        .signal_condvar(svc.process(), svc.arg(0), svc.arg32(1) as i32);
    svc.finish(Ok(()))
}

/// `svcWaitForAddress(addr x0, type w1, value x2, timeout x3)`.
pub(super) fn wait_for_address(svc: &mut SvcContext) -> Flow {
    let timeout = svc.arg(3) as i64;
    let outcome = ArbitrationType::from_raw(svc.arg32(1)).and_then(|kind| {
        let value = if kind == ArbitrationType::WaitIfEqual64 {
            svc.arg(2) as i64
        } else {
            svc.arg32(2) as i32 as i64
        };
        svc.kernel
            .arbiter
            .wait_for_address(svc.thread, svc.arg(0), kind, value, timeout == 0)
    });
    after_wait(svc, outcome, timeout)
}

/// `svcSignalToAddress(addr x0, type w1, value w2, count w3)`.
pub(super) fn signal_to_address(svc: &mut SvcContext) -> Flow {
    let result = SignalType::from_raw(svc.arg32(1)).and_then(|kind| {
        svc.kernel.arbiter.signal_to_address(
            svc.process(),
            svc.arg(0),
            kind,
            svc.arg32(2) as i32,
            svc.arg32(3) as i32,
        )
    });
    svc.finish(result)
}
