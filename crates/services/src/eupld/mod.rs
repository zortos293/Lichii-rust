//! `eupld:c` and `eupld:r`: error uploads are accepted and dropped.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("eupld:c", "eupld", |_| Upload::Context)?;
    context.register("eupld:r", "eupld", |_| Upload::Request)
}

enum Upload {
    Context,
    Request,
}

impl Service for Upload {
    fn name(&self) -> &str {
        match self {
            Self::Context => "eupld:c",
            Self::Request => "eupld:r",
        }
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match (self, ctx.command()) {
            (Self::Context, 0..=3) | (Self::Request, 0..=2 | 4) => {}
            (Self::Context, 4) => ctx.push(0u8),
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
    fn uploads_are_accepted() {
        let guest = Guest::new(ConsoleGeneration::Switch1, settings(), super::register);
        let context = guest.connect("eupld:c");
        for command in 0..4 {
            guest.call(context, &Request::new(command)).ok();
        }
        assert_eq!(guest.call(context, &Request::new(4)).ok().get::<u8>(0), 0);
        let request = guest.connect("eupld:r");
        for command in [0, 1, 2, 4] {
            guest.call(request, &Request::new(command)).ok();
        }
    }
}
