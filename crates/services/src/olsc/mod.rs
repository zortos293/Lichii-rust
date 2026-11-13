//! `olsc:u`: save data cloud backup for applications. There is no cloud, so backup stays off,
//! as the reference answers.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("olsc:u", "services", |_| Olsc)
}

struct Olsc;

impl Service for Olsc {
    fn name(&self) -> &str {
        "olsc:u"
    }

    /// `Initialize` (0) takes the process id and `SetSaveDataBackupSettingEnabled` (14) a flag
    /// and a user; neither is kept.
    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 | 14 => {}
            13 => ctx.push(0u8),
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
    fn backup_is_off() {
        let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
        let olsc = client.connect("olsc:u").unwrap();
        let initialize = Request::new(0).send_pid().arg(0u64);
        assert_eq!(
            client.send(olsc, &initialize).unwrap().result,
            ResultCode::SUCCESS
        );
        let mut reply = client.send(olsc, &Request::new(13)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u8>(), 0);
        let enable = Request::new(14).arg(1u8).arg([0u8; 7]).arg([0x11u8; 16]);
        assert_eq!(
            client.send(olsc, &enable).unwrap().result,
            ResultCode::SUCCESS
        );
    }

    #[test]
    fn commands_without_handlers_are_stubbed_only_for_switch2_titles() {
        for (title, expected) in [
            (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
            (SWITCH2_TITLE, ResultCode::SUCCESS),
        ] {
            let client = booted(title, Settings::default(), super::register);
            let olsc = client.connect("olsc:u").unwrap();
            assert_eq!(
                client.send(olsc, &Request::new(10)).unwrap().result,
                expected
            );
        }
    }
}
