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

    use lichii_common::settings::Settings;
    use lichii_kernel::testing::Request;

    use crate::testing::{ReplyExt, SWITCH1_TITLE, booted};

    #[test]
    fn reports_are_accepted() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let erpt = client.connect("erpt:c").unwrap();
        for command in [0, 1, 5, 11, 12] {
            client
                .send(erpt, &Request::new(command).arg(2u32))
                .unwrap()
                .ok();
        }
        client.connect("erpt:r").unwrap();
    }
}
