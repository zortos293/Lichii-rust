//! `nfp:user`: amiibo. `CreateUserInterface` opens an `IUser` on the NFC device manager
//! ([`crate::nfc::common`]), which finds no amiibo: `Mount` and every application-area and
//! info command answer `WrongDeviceState`.

use std::sync::Arc;

use lichii_common::HleResult;

use crate::ServiceContext;
use crate::nfc::common::{self, Backend, Command, UserKind};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    common::register(
        context,
        UserKind {
            service: "nfp:user",
            interface: "nfp:IUser",
            backend: Backend::Nfp,
            commands: user,
        },
    )
}

/// `nn::nfp::IUser`; `StartDetectionWithFilter` (25) is `StartDetection`.
fn user(command: u32) -> Option<Command> {
    Some(match command {
        0 => Command::Initialize,
        1 => Command::Finalize,
        2 => Command::ListDevices,
        3 | 25 => Command::StartDetection,
        4 => Command::StopDetection,
        5 => Command::Mount,
        6..=16 | 24 => Command::NeedsTag,
        17 => Command::AttachActivateEvent,
        18 => Command::AttachDeactivateEvent,
        19 => Command::GetState,
        20 => Command::GetDeviceState,
        21 => Command::GetNpadId,
        22 => Command::GetApplicationAreaSize,
        23 => Command::AttachAvailabilityChangeEvent,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::ControllerType;
    use lichii_kernel::ipc::UNKNOWN_COMMAND_ID;

    use crate::nfc::common::testing::{booted, is_quiet_event, list_devices, user};
    use crate::testing::{Request, SWITCH1_TITLE, SWITCH2_TITLE};

    const NFP_DEVICE_NOT_FOUND: ResultCode = ResultCode::new(115, 64);
    const NFP_INVALID_ARGUMENT: ResultCode = ResultCode::new(115, 65);
    const NFP_WRONG_DEVICE_STATE: ResultCode = ResultCode::new(115, 73);
    const NFP_NFC_DISABLED: ResultCode = ResultCode::new(115, 80);

    #[test]
    fn nfp_user_initializes_lists_player_one_and_finds_no_amiibo() {
        let (_context, client) = booted(
            SWITCH1_TITLE,
            Some(ControllerType::ProController),
            super::register,
        );
        let user = user(&client, "nfp:user");
        let call = |request: Request| client.send(user, &request).unwrap();
        let on = |command: u32| Request::new(command).arg(0u64);

        assert_eq!(call(Request::new(19)).pop::<u32>(), 0, "not initialized");
        assert_eq!(call(on(5).arg(0u32).arg(3u32)).result, NFP_NFC_DISABLED);
        assert_eq!(list_devices(&client, user, 2).0.result, NFP_NFC_DISABLED);
        assert_eq!(
            call(on(20)).pop::<u32>(),
            5,
            "unavailable before Initialize"
        );

        assert_eq!(call(Request::new(0)).result, ResultCode::SUCCESS);
        assert_eq!(call(Request::new(19)).pop::<u32>(), 1, "initialized");
        let (reply, handles) = list_devices(&client, user, 2);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(handles, [0], "player 1");
        assert_eq!(call(on(20)).pop::<u32>(), 0, "device initialized");
        assert_eq!(call(on(21)).pop::<u32>(), 0, "player 1 npad");

        assert_eq!(call(on(3)).result, ResultCode::SUCCESS);
        assert_eq!(call(on(20)).pop::<u32>(), 1, "searching for a tag");
        assert_eq!(
            call(on(3)).result,
            NFP_WRONG_DEVICE_STATE,
            "already searching"
        );
        assert_eq!(
            call(on(5).arg(0u32).arg(3u32)).result,
            NFP_WRONG_DEVICE_STATE
        );
        assert_eq!(call(on(5).arg(1u32).arg(3u32)).result, NFP_INVALID_ARGUMENT);
        for command in [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24] {
            assert_eq!(
                call(on(command)).result,
                NFP_WRONG_DEVICE_STATE,
                "{command}"
            );
        }
        assert_eq!(call(on(4)).result, ResultCode::SUCCESS);
        assert_eq!(call(on(20)).pop::<u32>(), 0);
        assert_eq!(call(on(25)).result, ResultCode::SUCCESS, "with filter");
        assert_eq!(call(on(4)).result, ResultCode::SUCCESS);

        let mut size = call(on(22));
        assert_eq!(size.result, ResultCode::SUCCESS);
        assert_eq!(size.pop::<u32>(), 0xd8);
        for command in [17, 18, 23] {
            let reply = call(on(command));
            assert_eq!(reply.result, ResultCode::SUCCESS, "{command}");
            assert!(is_quiet_event(&client, &reply), "event {command}");
        }
        assert_eq!(
            call(Request::new(17).arg(1u64)).result,
            NFP_DEVICE_NOT_FOUND
        );
        assert_eq!(
            call(Request::new(20).arg(0x99u64)).pop::<u32>(),
            6,
            "no such device"
        );
        assert_eq!(
            call(Request::new(21).arg(0x99u64)).result,
            NFP_DEVICE_NOT_FOUND
        );

        assert_eq!(call(Request::new(1)).result, ResultCode::SUCCESS);
        assert_eq!(call(Request::new(19)).pop::<u32>(), 0, "finalized");
    }

    #[test]
    fn nfp_user_lists_handheld_and_nothing_without_a_reader() {
        let (context, client) = booted(
            SWITCH1_TITLE,
            Some(ControllerType::Handheld),
            super::register,
        );
        let user = user(&client, "nfp:user");
        client.send(user, &Request::new(0)).unwrap();
        assert_eq!(list_devices(&client, user, 2).1, [0x20]);

        crate::nfc::common::testing::connect(&context, ControllerType::LeftJoycon);
        assert_eq!(
            list_devices(&client, user, 2).0.result,
            NFP_DEVICE_NOT_FOUND
        );
        let reply = client.send(user, &Request::new(20).arg(0x20u64)).unwrap();
        assert_eq!(reply.data()[0], 5, "unavailable");

        crate::nfc::common::testing::connect(&context, ControllerType::Switch2Handheld);
        assert_eq!(list_devices(&client, user, 2).1, [0x20], "reconnected");
        let empty = client.send(user, &Request::new(2)).unwrap();
        assert_eq!(empty.result, NFP_INVALID_ARGUMENT, "no room for a handle");
    }

    #[test]
    fn nfp_user_auto_stubs_unknown_commands_only_for_switch2() {
        for (title, expected) in [
            (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
            (SWITCH2_TITLE, ResultCode::SUCCESS),
        ] {
            let (_context, client) = booted(title, None, super::register);
            let user = user(&client, "nfp:user");
            let reply = client.send(user, &Request::new(100)).unwrap();
            assert_eq!(reply.result, expected, "{title:#x}");
        }
    }
}
