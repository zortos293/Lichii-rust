//! Supervisor call dispatch.
//!
//! Arguments arrive in `x0..x7` and results go back in the same registers (64-bit ABI). A call
//! that blocks saves the context, commits its wait and returns [`Flow::Switch`]; the result
//! registers are written when the thread resumes on whichever core picks it up.

mod info;
mod memory;
mod numbers;
mod process;
mod sync;
mod thread;

use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_common::{HleResult, ResultCode};
use lichii_cpu::ThreadContext;

pub use numbers::Svc;
pub(crate) use thread::release_thread_resources;

use crate::KernelInner;
use crate::handle_table::{CURRENT_PROCESS, CURRENT_THREAD};
use crate::object::{FromObject, KObject};
use crate::process::Process;
use crate::result;
use crate::thread::{Resume, Thread, ThreadState, WakeResult};

/// What the core does after a supervisor call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flow {
    /// Keep running the thread.
    Continue,
    /// Keep running unless an equal or higher priority thread is ready (yield).
    Yield,
    /// The thread blocked; its context is saved and the core picks another.
    Switch,
    /// The thread exits.
    ExitThread,
}

pub(crate) struct SvcContext<'a> {
    pub kernel: &'a Arc<KernelInner>,
    pub core: usize,
    pub thread: &'a Arc<Thread>,
    pub ctx: &'a mut ThreadContext,
}

pub(crate) fn apply_resume(ctx: &mut ThreadContext, resume: Resume, wake: WakeResult) {
    match resume {
        Resume::Nothing => {}
        Resume::Result => ctx.x[0] = wake.result.0 as u64,
        Resume::ResultAndIndex => {
            ctx.x[0] = wake.result.0 as u64;
            ctx.x[1] = wake.index as u32 as u64;
        }
    }
}

/// How long a blocking call may wait. Negative is forever, as in Horizon.
pub(crate) fn deadline(timeout_ns: i64) -> Option<Instant> {
    (timeout_ns >= 0).then(|| {
        Instant::now()
            .checked_add(Duration::from_nanos(timeout_ns as u64))
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(1 << 32))
    })
}

impl SvcContext<'_> {
    pub fn arg(&self, index: usize) -> u64 {
        self.ctx.x[index]
    }

    pub fn arg32(&self, index: usize) -> u32 {
        self.ctx.x[index] as u32
    }

    pub fn set(&mut self, index: usize, value: u64) {
        self.ctx.x[index] = value;
    }

    pub fn set_result(&mut self, result: ResultCode) {
        self.ctx.x[0] = result.0 as u64;
    }

    /// Writes `Ok(())`/`Err` to `x0` and continues.
    pub fn finish(&mut self, result: HleResult<()>) -> Flow {
        self.set_result(ResultCode::from(result));
        Flow::Continue
    }

    pub fn process(&self) -> &Arc<Process> {
        self.thread.process()
    }

    /// Resolves a handle in the calling process, including the pseudo-handles.
    pub fn object(&self, handle: u32) -> Option<KObject> {
        match handle {
            CURRENT_THREAD => Some(KObject::Thread(self.thread.clone())),
            CURRENT_PROCESS => Some(KObject::Process(self.process().clone())),
            _ => self.process().handles().read().get(handle),
        }
    }

    pub fn typed<T: FromObject>(&self, handle: u32) -> HleResult<Arc<T>> {
        self.object(handle)
            .and_then(|object| T::from_object(&object))
            .ok_or(result::INVALID_HANDLE)
    }

    /// Commits a wait begun with `begin_wait`. Saves the context first, so that a waker on
    /// another core can resume the thread as soon as it is visible as waiting. If the thread
    /// was already woken, the result is applied here and it keeps running.
    pub fn block(
        &mut self,
        token: u64,
        deadline: Option<Instant>,
        timeout_is_success: bool,
    ) -> Flow {
        *self.thread.context.lock() = self.ctx.clone();
        let mut s = self.thread.sched.lock();
        if s.wait != token {
            let resume = std::mem::take(&mut s.resume);
            let wake = s.wake;
            drop(s);
            apply_resume(self.ctx, resume, wake);
            return Flow::Continue;
        }
        s.state = ThreadState::Waiting;
        if let Some(deadline) = deadline {
            if timeout_is_success {
                self.kernel
                    .timer
                    .schedule_wakeup(deadline, self.thread, token);
            } else {
                self.kernel
                    .timer
                    .schedule_timeout(deadline, self.thread, token);
            }
        }
        drop(s);
        self.kernel.scheduler.clear_running(self.core);
        Flow::Switch
    }
}

pub(crate) fn dispatch(svc: &mut SvcContext, imm: u32) -> Flow {
    let caps = svc.process().capabilities();
    if caps.svc_mask != [0; 4] && !caps.permits_svc(imm) {
        tracing::warn!(
            svc = imm,
            "supervisor call not permitted by the process capabilities"
        );
        svc.set_result(result::NOT_IMPLEMENTED);
        return Flow::Continue;
    }
    match imm {
        0x01 => memory::set_heap_size(svc),
        0x02 => memory::set_memory_permission(svc),
        0x03 => memory::set_memory_attribute(svc),
        0x04 => memory::map_memory(svc),
        0x05 => memory::unmap_memory(svc),
        0x06 => memory::query_memory(svc),
        0x07 => process::exit_process(svc),
        0x08 => thread::create_thread(svc),
        0x09 => thread::start_thread(svc),
        0x0a => Flow::ExitThread,
        0x0b => thread::sleep_thread(svc),
        0x0c => thread::get_thread_priority(svc),
        0x0d => thread::set_thread_priority(svc),
        0x0e => thread::get_thread_core_mask(svc),
        0x0f => thread::set_thread_core_mask(svc),
        0x10 => {
            svc.set(0, svc.core as u64);
            Flow::Continue
        }
        0x11 => sync::signal_event(svc),
        0x12 => sync::clear_event(svc),
        0x13 => memory::map_shared_memory(svc),
        0x14 => memory::unmap_shared_memory(svc),
        0x15 => memory::create_transfer_memory(svc),
        0x16 => sync::close_handle(svc),
        0x17 => sync::reset_signal(svc),
        0x18 => sync::wait_synchronization(svc),
        0x19 => sync::cancel_synchronization(svc),
        0x1a => sync::arbitrate_lock(svc),
        0x1b => sync::arbitrate_unlock(svc),
        0x1c => sync::wait_process_wide_key_atomic(svc),
        0x1d => sync::signal_process_wide_key(svc),
        0x1e => {
            svc.set(0, svc.kernel.clock.ticks());
            Flow::Continue
        }
        0x24 => process::get_process_id(svc),
        0x25 => thread::get_thread_id(svc),
        0x26 => process::break_(svc),
        0x27 => process::output_debug_string(svc),
        0x28 => process::return_from_exception(svc),
        0x29 => info::get_info(svc),
        0x2a | 0x2b | 0x5d | 0x5e | 0x5f => svc.finish(Ok(())),
        0x2c => memory::map_physical_memory(svc),
        0x2d => memory::unmap_physical_memory(svc),
        0x30 => info::get_resource_limit_value(svc, info::LimitQuery::Limit),
        0x31 => info::get_resource_limit_value(svc, info::LimitQuery::Current),
        0x37 => info::get_resource_limit_value(svc, info::LimitQuery::Peak),
        0x32 => thread::set_thread_activity(svc),
        0x33 => thread::get_thread_context3(svc),
        0x34 => sync::wait_for_address(svc),
        0x35 => sync::signal_to_address(svc),
        0x36 => svc.finish(Ok(())),
        0x45 => sync::create_event(svc),
        0x4b => memory::create_code_memory(svc),
        0x4c => memory::control_code_memory(svc),
        0x6f => info::get_system_info(svc),
        0x80 => memory::set_gpu_memory_attribute(svc),
        _ => {
            tracing::warn!(svc = imm, "unimplemented supervisor call");
            svc.set_result(result::NOT_IMPLEMENTED);
            Flow::Continue
        }
    }
}
