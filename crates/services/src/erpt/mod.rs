//! `erpt:c` and `erpt:r`: error reports are accepted and dropped.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("erpt:c", "erpt", |_| Context)?;
    context.register("erpt:r", "erpt", |_| Session)
}

/// `IContext`.
struct Context;

impl Service for Context {
    fn name(&self) -> &str {
        "erpt:c"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 | 5 => {}
            1 | 11 | 12 => {
                let report_type: u32 = ctx.pop();
                tracing::debug!(report_type, "erpt: error report dropped");
            }
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `ISession`. Titles never open reports.
struct Session;

impl Service for Session {
    fn name(&self) -> &str {
        "erpt:r"
    }

    fn handle(&self, _ctx: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ConsoleGeneration;

    use crate::testing::{Guest, Request, settings};

    #[test]
    fn reports_are_accepted() {
        let guest = Guest::new(ConsoleGeneration::Switch1, settings(), super::register);
        let erpt = guest.connect("erpt:c");
        for command in [0, 1, 5, 11, 12] {
            guest.call(erpt, &Request::new(command).arg(2u32)).ok();
        }
        guest.connect("erpt:r");
    }
}
