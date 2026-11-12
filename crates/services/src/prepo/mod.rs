//! `prepo:a`, `prepo:a2`, `prepo:m`, `prepo:s` and `prepo:u`: play reports are accepted and
//! logged, never stored or sent.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    for name in ["prepo:a", "prepo:a2", "prepo:m", "prepo:s", "prepo:u"] {
        context.register(name, "prepo", move |_| PlayReport { name })?;
    }
    Ok(())
}

struct PlayReport {
    name: &'static str,
}

impl PlayReport {
    fn log(&self, ctx: &HleRequestContext, kind: &str, user: Option<[u64; 2]>, id: u64) {
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let event = ctx
            .buffers_a()
            .first()
            .filter(|buffer| buffer.size != 0)
            .and_then(|buffer| ctx.memory().read_cstring(buffer.address, buffer.size.min(0x40) as usize).ok())
            .unwrap_or_default();
        let size = |buffers: &[lichii_kernel::ipc::BufferDescriptor]| buffers.first().map_or(0, |b| b.size);
        tracing::debug!(
            service = self.name,
            kind,
            event,
            user = format_args!("{:016x}{:016x}", user.unwrap_or_default()[1], user.unwrap_or_default()[0]),
            id = format_args!("{id:#x}"),
            a = size(ctx.buffers_a()),
            x = size(ctx.buffers_x()),
            "play report"
        );
    }
}

impl Service for PlayReport {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            10100 | 10102 | 10104 | 10106 => {
                let pid: u64 = ctx.pop();
                self.log(ctx, "application", None, pid);
            }
            10101 | 10103 | 10105 | 10107 => {
                let user: [u64; 2] = ctx.pop();
                let pid: u64 = ctx.pop();
                self.log(ctx, "application", Some(user), pid);
            }
            10200 => {}
            10300 => ctx.push(0i32),
            10400 => ctx.push(0u64),
            20100 => {
                let title: u64 = ctx.pop();
                self.log(ctx, "system", None, title);
            }
            20101 => {
                let user: [u64; 2] = ctx.pop();
                let title: u64 = ctx.pop();
                self.log(ctx, "system", Some(user), title);
            }
            20102 => {
                let _field: u64 = ctx.pop();
                let title: u64 = ctx.pop();
                self.log(ctx, "system", None, title);
            }
            20103 => {
                let _field: u64 = ctx.pop();
                let user: [u64; 2] = ctx.pop();
                let title: u64 = ctx.pop();
                self.log(ctx, "system", Some(user), title);
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ConsoleGeneration;

    use crate::testing::{Guest, Request, settings};

    #[test]
    fn reports_are_accepted() {
        let guest = Guest::new(ConsoleGeneration::Switch1, settings(), super::register);
        let prepo = guest.connect("prepo:u");
        let event = guest.buffer_with(b"test_event\0");
        let data = guest.buffer_with(&[0x80]);
        for command in [10100, 10102, 10104, 10106] {
            let request = Request::new(command).arg(0u64).pid().a(event, 11).x(data, 1);
            guest.call(prepo, &request).ok();
        }
        let with_user = Request::new(10107).arg([1u64, 2]).arg(0u64).pid().a(event, 11).x(data, 1);
        guest.call(prepo, &with_user).ok();
        guest.call(prepo, &Request::new(20103).arg(0u64).arg([1u64, 2]).arg(0x0100u64)).ok();
        guest.call(prepo, &Request::new(10200)).ok();
        assert_eq!(guest.call(prepo, &Request::new(10300)).ok().get::<i32>(0), 0);
        assert_eq!(guest.call(prepo, &Request::new(10400)).ok().get::<u64>(0), 0);
    }
}
