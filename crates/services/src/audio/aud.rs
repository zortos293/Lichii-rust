//! Switch 2 `aud:u` (`nn::audio::CreateUserClientSessionByHipc`): the SDK reaches the Switch 1
//! audio managers through it. Command 0 is `audin:u`, 1 `audout:u` and 2 `audren:u`, each
//! backed by the same shared state as the registered service.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use super::{Audio, audin, audout, audren};
use crate::{Service, ServiceContext};

pub(super) fn register(context: &Arc<ServiceContext>, audio: &Arc<Audio>) -> HleResult<()> {
    let audio = audio.clone();
    context.register("aud:u", "audio", move |context| AudU {
        context: context.clone(),
        audio: audio.clone(),
    })
}

struct AudU {
    context: Arc<ServiceContext>,
    audio: Arc<Audio>,
}

impl Service for AudU {
    fn name(&self) -> &str {
        "nx2:aud:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let (context, audio) = (&self.context, &self.audio);
        let handler = match ctx.command() {
            0 => context.handler(audin::manager(context, audio)),
            1 => {
                let _: u32 = ctx.pop();
                context.handler(audout::manager(context, audio))
            }
            2 => {
                let _: u32 = ctx.pop();
                context.handler(audren::AudRenU::new(context, audio))
            }
            _ => return None,
        };
        ctx.push_interface(handler);
        Some(ResultCode::SUCCESS)
    }
}
