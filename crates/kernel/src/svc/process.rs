use super::{Flow, SvcContext};
use crate::object::KObject;
use crate::result;

/// `svcExitProcess`: every other thread is asked to terminate; this one exits now.
pub(super) fn exit_process(svc: &mut SvcContext) -> Flow {
    svc.kernel.terminate_process(svc.process());
    Flow::ExitThread
}

/// `svcGetProcessId(handle w1)` -> `x1`; thread handles name their process.
pub(super) fn get_process_id(svc: &mut SvcContext) -> Flow {
    let id = match svc.object(svc.arg32(1)) {
        Some(KObject::Process(process)) => Ok(process.id()),
        Some(KObject::Thread(thread)) => Ok(thread.process().id()),
        _ => Err(result::INVALID_HANDLE),
    };
    match id {
        Ok(id) => {
            svc.set(1, id);
            svc.finish(Ok(()))
        }
        Err(code) => svc.finish(Err(code)),
    }
}

/// `svcBreak(reason w0, info x1, size x2)`. Bit 31 of the reason marks a notification; any
/// other break is fatal for the process.
pub(super) fn break_(svc: &mut SvcContext) -> Flow {
    let (reason, info, size) = (svc.arg32(0), svc.arg(1), svc.arg(2));
    let notification = reason & 0x8000_0000 != 0;
    svc.kernel
        .events()
        .on_break(svc.process(), reason, info, size);
    if notification {
        return svc.finish(Ok(()));
    }
    tracing::error!(
        process = svc.process().name(),
        reason = format_args!("{reason:#x}"),
        info = format_args!("{info:#x}"),
        size = format_args!("{size:#x}"),
        pc = format_args!("{:#x}", svc.ctx.pc),
        lr = format_args!("{:#x}", svc.ctx.x[30]),
        "svcBreak"
    );
    svc.kernel.terminate_process(svc.process());
    Flow::ExitThread
}

/// `svcOutputDebugString(ptr x0, len x1)`.
pub(super) fn output_debug_string(svc: &mut SvcContext) -> Flow {
    let (ptr, len) = (svc.arg(0), svc.arg(1).min(0x1000) as usize);
    let mut bytes = vec![0u8; len];
    let result = svc
        .process()
        .memory()
        .read_bytes(ptr, &mut bytes)
        .map_err(|_| result::INVALID_CURRENT_MEMORY)
        .map(|()| {
            let text = String::from_utf8_lossy(&bytes);
            svc.kernel.events().on_debug_output(svc.process(), &text);
        });
    svc.finish(result)
}

/// `svcReturnFromException`: without a user exception handler to return to, the process ends.
pub(super) fn return_from_exception(svc: &mut SvcContext) -> Flow {
    tracing::error!(
        process = svc.process().name(),
        result = svc.arg32(0),
        "svcReturnFromException"
    );
    svc.kernel.terminate_process(svc.process());
    Flow::ExitThread
}
