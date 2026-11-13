//! `npns:u`: push notifications for applications. Nothing is ever pushed, so the receive event
//! never fires, as the reference answers.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let receive = Arc::new(Event::new());
    context.register("npns:u", "services", move |_| Npns {
        receive: receive.clone(),
    })
}

struct Npns {
    receive: Arc<Event>,
}

impl Service for Npns {
    fn name(&self) -> &str {
        "npns:u"
    }

    /// `GetReceiveEvent` (5) and `ListenToByName` (8), which takes the name in buffer 0.
    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            5 => ctx.push_copy_object(KObject::ReadableEvent(self.receive.clone())),
            8 => {}
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;
    use lichii_kernel::ipc::UNKNOWN_COMMAND_ID;
    use lichii_kernel::object::KObject;

    use crate::testing::{Request, SWITCH1_TITLE, SWITCH2_TITLE, booted};

    #[test]
    fn listening_succeeds_and_nothing_is_received() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let npns = client.connect("npns:u").unwrap();
        let name = client.alloc(b"topic");
        let listen = Request::new(8).buffer_a(name, 5);
        assert_eq!(
            client.send(npns, &listen).unwrap().result,
            ResultCode::SUCCESS
        );
        let reply = client.send(npns, &Request::new(5)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert!(matches!(
            client.object(reply.copy_handles[0]),
            Some(KObject::ReadableEvent(event)) if !event.is_signaled()
        ));
    }

    #[test]
    fn unhandled_commands_are_stubbed_only_for_switch2_titles() {
        for (title, expected) in [
            (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
            (SWITCH2_TITLE, ResultCode::SUCCESS),
        ] {
            let client = booted(title, Settings::default(), super::register);
            let npns = client.connect("npns:u").unwrap();
            assert_eq!(
                client.send(npns, &Request::new(1)).unwrap().result,
                expected
            );
        }
    }
}
