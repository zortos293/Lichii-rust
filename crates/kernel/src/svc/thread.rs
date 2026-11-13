use std::sync::Arc;

use lichii_common::HleResult;
use lichii_cpu::ThreadContext;

use super::{Flow, SvcContext, deadline};
use crate::object::KObject;
use crate::resource_limit::LimitableResource;
use crate::result;
use crate::thread::{LOWEST_PRIORITY, Resume, Thread, ThreadState};

const IDEAL_CORE_DONT_CARE: i32 = -1;
const IDEAL_CORE_USE_PROCESS_VALUE: i32 = -2;
const IDEAL_CORE_NO_UPDATE: i32 = -3;

/// `svcCreateThread(entry x1, arg x2, stack_top x3, priority w4, core w5)` -> `w1` handle.
pub(super) fn create_thread(svc: &mut SvcContext) -> Flow {
    let (entry, arg, stack_top) = (svc.arg(1), svc.arg(2), svc.arg(3));
    let priority = svc.arg32(4);
    let core = svc.arg32(5) as i32;
    match svc
        .kernel
        .create_thread(svc.process(), entry, arg, stack_top, priority, core)
    {
        Ok(thread) => {
            let handle = svc
                .process()
                .handles()
                .write()
                .add(KObject::Thread(thread.clone()));
            match handle {
                Ok(handle) => {
                    svc.set(1, handle as u64);
                    svc.finish(Ok(()))
                }
                Err(code) => {
                    svc.process().remove_thread(&thread);
                    release_thread_resources(&thread);
                    svc.finish(Err(code))
                }
            }
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcStartThread(handle w0)`.
pub(super) fn start_thread(svc: &mut SvcContext) -> Flow {
    let result = svc.typed::<Thread>(svc.arg32(0)).and_then(|thread| {
        let mut s = thread.sched.lock();
        if s.state != ThreadState::Created {
            return Err(result::INVALID_STATE);
        }
        if s.terminate_requested {
            return Err(result::TERMINATION_REQUESTED);
        }
        thread.scheduler.enqueue(&thread, &mut s, false);
        Ok(())
    });
    svc.finish(result)
}

/// `svcSleepThread(ns x0)`: 0, -1 and -2 are the three yield flavours.
pub(super) fn sleep_thread(svc: &mut SvcContext) -> Flow {
    let ns = svc.arg(0) as i64;
    if ns <= 0 {
        return Flow::Yield;
    }
    let token = match svc.thread.begin_wait(Resume::Nothing, false) {
        Ok(token) => token,
        Err(_) => return Flow::Continue,
    };
    svc.block(token, deadline(ns), true)
}

/// `svcGetThreadPriority(handle w1)` -> `w1`.
pub(super) fn get_thread_priority(svc: &mut SvcContext) -> Flow {
    match svc.typed::<Thread>(svc.arg32(1)) {
        Ok(thread) => {
            svc.set(1, thread.priority() as u64);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcSetThreadPriority(handle w0, priority w1)`.
pub(super) fn set_thread_priority(svc: &mut SvcContext) -> Flow {
    let priority = svc.arg32(1);
    let result = (|| {
        let caps = svc.process().capabilities();
        if priority > LOWEST_PRIORITY as u32
            || (caps.priority_mask != 0 && caps.priority_mask & (1 << priority) == 0)
        {
            return Err(result::INVALID_PRIORITY);
        }
        let thread = svc.typed::<Thread>(svc.arg32(0))?;
        thread.set_base_priority(priority as u8);
        Ok(())
    })();
    svc.finish(result)
}

/// `svcGetThreadCoreMask(handle w2)` -> `w1` ideal core, `x2` affinity mask.
pub(super) fn get_thread_core_mask(svc: &mut SvcContext) -> Flow {
    match svc.typed::<Thread>(svc.arg32(2)) {
        Ok(thread) => {
            let (ideal, mask) = thread.core_mask();
            svc.set(1, ideal as u32 as u64);
            svc.set(2, mask);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcSetThreadCoreMask(handle w0, core w1, mask x2)`.
pub(super) fn set_thread_core_mask(svc: &mut SvcContext) -> Flow {
    let result = (|| -> HleResult<()> {
        let mut core = svc.arg32(1) as i32;
        let mut mask = svc.arg(2);
        let process = svc.process();
        let process_mask = process.capabilities().core_mask & svc.kernel.scheduler.all_cores_mask();
        let process_mask = if process_mask == 0 {
            svc.kernel.scheduler.all_cores_mask()
        } else {
            process_mask
        };
        if core == IDEAL_CORE_USE_PROCESS_VALUE {
            core = process.params().main_thread_core as i32;
            mask = 1 << core;
        } else {
            if mask == 0 {
                return Err(result::INVALID_COMBINATION);
            }
            if mask & !process_mask != 0 {
                return Err(result::INVALID_CORE_ID);
            }
            if core >= 0 {
                if core as usize >= svc.kernel.scheduler.core_count() {
                    return Err(result::INVALID_CORE_ID);
                }
                if mask & (1 << core) == 0 {
                    return Err(result::INVALID_COMBINATION);
                }
            } else if core != IDEAL_CORE_NO_UPDATE && core != IDEAL_CORE_DONT_CARE {
                return Err(result::INVALID_CORE_ID);
            }
        }
        let thread = svc.typed::<Thread>(svc.arg32(0))?;
        let mut s = thread.sched.lock();
        if core == IDEAL_CORE_NO_UPDATE {
            core = s.ideal_core;
            if core >= 0 && mask & (1 << core) == 0 {
                return Err(result::INVALID_COMBINATION);
            }
        }
        s.ideal_core = core;
        s.affinity = mask;
        match s.state {
            ThreadState::Runnable if mask & (1 << s.active_core) == 0 => {
                thread.scheduler.dequeue(&thread, &s);
                if core >= 0 {
                    s.active_core = core as usize;
                }
                thread.scheduler.enqueue(&thread, &mut s, false);
            }
            ThreadState::Running(running) if mask & (1 << running) == 0 => {
                if core >= 0 {
                    s.active_core = core as usize;
                }
                svc.kernel.scheduler.request_reschedule(running);
            }
            _ => {
                if core >= 0 && !matches!(s.state, ThreadState::Runnable | ThreadState::Running(_))
                {
                    s.active_core = core as usize;
                }
            }
        }
        Ok(())
    })();
    svc.finish(result)
}

/// `svcGetThreadId(handle w1)` -> `x1`.
pub(super) fn get_thread_id(svc: &mut SvcContext) -> Flow {
    match svc.typed::<Thread>(svc.arg32(1)) {
        Ok(thread) => {
            svc.set(1, thread.id());
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcSetThreadActivity(handle w0, activity w1)`: 0 runs, 1 pauses.
pub(super) fn set_thread_activity(svc: &mut SvcContext) -> Flow {
    let activity = svc.arg32(1);
    let result = (|| {
        if activity > 1 {
            return Err(result::INVALID_ENUM_VALUE);
        }
        let thread = svc.typed::<Thread>(svc.arg32(0))?;
        if Arc::ptr_eq(&thread, svc.thread) {
            return Err(result::BUSY);
        }
        if !Arc::ptr_eq(thread.process(), svc.process()) {
            return Err(result::INVALID_HANDLE);
        }
        let mut s = thread.sched.lock();
        if activity == 1 {
            if s.pause_requested {
                return Err(result::INVALID_STATE);
            }
            s.pause_requested = true;
            match s.state {
                ThreadState::Runnable => {
                    thread.scheduler.dequeue(&thread, &s);
                    s.state = ThreadState::Suspended;
                }
                ThreadState::Running(core) => svc.kernel.scheduler.request_reschedule(core),
                _ => {}
            }
        } else {
            if !s.pause_requested {
                return Err(result::INVALID_STATE);
            }
            s.pause_requested = false;
            if s.state == ThreadState::Suspended {
                thread.scheduler.enqueue(&thread, &mut s, false);
            }
        }
        Ok(())
    })();
    svc.finish(result)
}

/// Horizon's `ThreadContext` as `svcGetThreadContext3` writes it (0x320 bytes).
fn horizon_context(ctx: &ThreadContext) -> Vec<u8> {
    let mut out = Vec::with_capacity(0x320);
    for x in &ctx.x[..29] {
        out.extend_from_slice(&x.to_le_bytes());
    }
    for v in [ctx.x[29], ctx.x[30], ctx.sp, ctx.pc] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&ctx.pstate.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for v in &ctx.v {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&ctx.fpcr.to_le_bytes());
    out.extend_from_slice(&ctx.fpsr.to_le_bytes());
    out.extend_from_slice(&ctx.tpidr_el0.to_le_bytes());
    out
}

/// `svcGetThreadContext3(out x0, handle w1)`: the thread must be paused.
pub(super) fn get_thread_context3(svc: &mut SvcContext) -> Flow {
    let out = svc.arg(0);
    let result = (|| {
        let thread = svc.typed::<Thread>(svc.arg32(1))?;
        if Arc::ptr_eq(&thread, svc.thread) || !Arc::ptr_eq(thread.process(), svc.process()) {
            return Err(result::INVALID_HANDLE);
        }
        if thread.state() != ThreadState::Suspended {
            return Err(result::INVALID_STATE);
        }
        let bytes = horizon_context(&thread.context());
        svc.process()
            .memory()
            .write_bytes(out, &bytes)
            .map_err(|_| result::INVALID_POINTER)
    })();
    svc.finish(result)
}

pub(crate) fn release_thread_resources(thread: &Arc<Thread>) {
    let process = thread.process();
    process.page_table().free_tls(thread.tls_address());
    process
        .resource_limit()
        .release(LimitableResource::Threads, 1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_layout_is_0x320_bytes() {
        let mut ctx = ThreadContext::default();
        ctx.x[29] = 0xf0;
        ctx.pc = 0x1234;
        let bytes = horizon_context(&ctx);
        assert_eq!(bytes.len(), 0x320);
        assert_eq!(&bytes[29 * 8..30 * 8], &0xf0u64.to_le_bytes());
        assert_eq!(&bytes[32 * 8..33 * 8], &0x1234u64.to_le_bytes());
    }
}
