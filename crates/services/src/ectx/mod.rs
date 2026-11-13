//! `ectx`: the error context service. Applications open a registrar on `ectx:aw` and hand it
//! the context of an error they are about to report; the reference keeps nothing and answers
//! success, and `ectx:w` and `ectx:r` have no commands.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::dispatch::Commandless;
use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("ectx:w", "services", |_| Commandless("ectx:w"))?;
    context.register("ectx:r", "services", |_| Commandless("ectx:r"))?;
    context.register("ectx:aw", "services", |context| Ectx {
        context: context.clone(),
    })
}

struct Ectx {
    context: Arc<ServiceContext>,
}

impl Service for Ectx {
    fn name(&self) -> &str {
        "ectx:aw"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push_interface(self.context.handler(ContextRegistrar)),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `nn::err::context::IContextRegistrar`.
struct ContextRegistrar;

impl Service for ContextRegistrar {
    fn name(&self) -> &str {
        "IContextRegistrar"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push(0u32),
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

    use crate::testing::{Request, SWITCH1_TITLE, SWITCH2_TITLE, booted};

    #[test]
    fn registrar_completes_a_context() {
        for title in [SWITCH1_TITLE, SWITCH2_TITLE] {
            let client = booted(title, Settings::default(), super::register);
            let ectx = client.connect("ectx:aw").unwrap();
            let reply = client.send(ectx, &Request::new(0)).unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            let registrar = reply.move_handles[0];
            let context = client.alloc(&[0xab; 0x40]);
            let mut reply = client
                .send(
                    registrar,
                    &Request::new(0).arg(7u32).buffer_a(context, 0x40),
                )
                .unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            assert_eq!(reply.pop::<u32>(), 0);
        }
    }

    #[test]
    fn commands_without_handlers_are_stubbed_only_for_switch2_titles() {
        for (title, stubbed) in [(SWITCH1_TITLE, false), (SWITCH2_TITLE, true)] {
            let client = booted(title, Settings::default(), super::register);
            for (name, command) in [("ectx:w", 0), ("ectx:r", 1), ("ectx:aw", 1)] {
                let service = client.connect(name).unwrap();
                let reply = client.send(service, &Request::new(command)).unwrap();
                let expected = if stubbed {
                    ResultCode::SUCCESS
                } else {
                    UNKNOWN_COMMAND_ID
                };
                assert_eq!(reply.result, expected, "{name} {command}");
            }
        }
    }
}
