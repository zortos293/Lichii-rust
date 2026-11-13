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

    use lichii_common::settings::Settings;
    use lichii_kernel::testing::Request;

    use crate::testing::{ReplyExt, SWITCH1_TITLE, booted};

    #[test]
    fn uploads_are_accepted() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let context = client.connect("eupld:c").unwrap();
        for command in 0..4 {
            client.send(context, &Request::new(command)).unwrap().ok();
        }
        assert_eq!(
            client
                .send(context, &Request::new(4))
                .unwrap()
                .ok()
                .get::<u8>(0),
            0
        );
        let request = client.connect("eupld:r").unwrap();
        for command in [0, 1, 2, 4] {
            client.send(request, &Request::new(command)).unwrap().ok();
        }
    }
}
