//! `chat:u`, Switch 2 GameChat. Newer SDKs call `nn::chat::Initialize` at boot and wait in `sm:`
//! until the service exists (NX2_NOTES row 123). Command 0 opens a session; its command 1 returns
//! an event the title waits on, which nothing signals yet. Both are `nx2:` interfaces, so the
//! commands nobody knows yet get the Switch 2 auto-stub.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("chat:u", "services", |context| ChatService {
        context: context.clone(),
    })
}

struct ChatService {
    context: Arc<ServiceContext>,
}

impl Service for ChatService {
    fn name(&self) -> &str {
        "nx2:chat:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                tracing::warn!("(stubbed) chat:u OpenSession");
                ctx.push_interface(self.context.handler(ChatSession {
                    event: Arc::new(Event::new()),
                }));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

struct ChatSession {
    event: Arc<Event>,
}

impl Service for ChatSession {
    fn name(&self) -> &str {
        "nx2:chat:u::0"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.event.clone()));
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;
    use lichii_kernel::object::KObject;
    use lichii_kernel::testing::Request;

    use crate::testing::{SWITCH1_TITLE, context};

    #[test]
    fn open_session_returns_a_session_whose_command_1_returns_an_event() {
        let (context, client) = context(SWITCH1_TITLE, Settings::default());
        super::register(&context).unwrap();
        let chat = client.connect("chat:u").expect("chat:u registered");
        let reply = client.send(chat, &Request::new(0)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let session = reply.move_handles[0];
        let reply = client.send(session, &Request::new(1)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        match client.object(reply.copy_handles[0]) {
            Some(KObject::ReadableEvent(event)) => assert!(!event.is_signaled()),
            _ => panic!("command 1 did not return an event"),
        }
        let stub = client.send(session, &Request::new(7)).unwrap();
        assert_eq!(
            stub.result,
            ResultCode::SUCCESS,
            "nx2: interfaces auto-stub"
        );
        assert_eq!(stub.data(), [0; 128]);
    }
}
