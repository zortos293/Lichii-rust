//! `nfc:user` and `nfc:mf:u`: NFC tags and Mifare cards (Skylanders). Both open an `IUser` on
//! the shared device manager in [`common`], which finds no tag.

pub(crate) mod common;

use std::sync::Arc;

use lichii_common::HleResult;

use crate::ServiceContext;
use common::{Backend, Command, UserKind};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    common::register(context, NFC_USER)?;
    common::register(context, MIFARE_USER)
}

const NFC_USER: UserKind = UserKind {
    service: "nfc:user",
    interface: "nfc:IUser",
    backend: Backend::Nfc,
    commands: nfc_user,
};

const MIFARE_USER: UserKind = UserKind {
    service: "nfc:mf:u",
    interface: "nfc:mf:IUser",
    backend: Backend::Mifare,
    commands: mifare_user,
};

/// `nn::nfc::IUser`; 0-3 are the pre-4.0.0 ids of 400-403.
fn nfc_user(command: u32) -> Option<Command> {
    Some(match command {
        0 | 400 => Command::Initialize,
        1 | 401 => Command::Finalize,
        2 | 402 => Command::GetState,
        3 | 403 => Command::IsNfcEnabled,
        404 => Command::ListDevices,
        405 => Command::GetDeviceState,
        406 => Command::GetNpadId,
        407 => Command::AttachAvailabilityChangeEvent,
        408 => Command::StartDetection,
        409 => Command::StopDetection,
        410 | 1000 | 1001 => Command::NeedsTag,
        411 => Command::AttachActivateEvent,
        412 => Command::AttachDeactivateEvent,
        1300 => Command::SendCommandByPassThrough,
        _ => return None,
    })
}

/// `nn::nfc::mifare::IUser`.
fn mifare_user(command: u32) -> Option<Command> {
    Some(match command {
        0 => Command::Initialize,
        1 => Command::Finalize,
        2 => Command::ListDevices,
        3 => Command::StartDetection,
        4 => Command::StopDetection,
        5..=7 => Command::NeedsTag,
        8 => Command::AttachActivateEvent,
        9 => Command::AttachDeactivateEvent,
        10 => Command::GetState,
        11 => Command::GetDeviceState,
        12 => Command::GetNpadId,
        13 => Command::AttachAvailabilityChangeEvent,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::ControllerType;
    use lichii_kernel::ipc::UNKNOWN_COMMAND_ID;

    use std::time::{Duration, Instant};

    use lichii_kernel::ipc::Handle;
    use lichii_kernel::object::KObject;

    use super::common::testing::{
        booted, booted_with, connect, disconnect, is_quiet_event, list_devices, user,
    };
    use crate::testing::{Request, SWITCH1_TITLE, SWITCH2_TITLE, TempDir};

    const NFC_DEVICE_NOT_FOUND: ResultCode = ResultCode::new(127, 64);
    const NFC_WRONG_DEVICE_STATE: ResultCode = ResultCode::new(127, 73);
    const NFC_NOT_INITIALIZED: ResultCode = ResultCode::new(127, 77);
    const NFC_DISABLED: ResultCode = ResultCode::new(127, 80);
    const NFP_WRONG_DEVICE_STATE: ResultCode = ResultCode::new(115, 73);
    const NFP_NFC_DISABLED: ResultCode = ResultCode::new(115, 80);

    #[test]
    fn nfc_user_detects_no_tag_and_reports_nfc_errors() {
        let (context, client) = booted(
            SWITCH1_TITLE,
            Some(ControllerType::DualJoycons),
            super::register,
        );
        let user = user(&client, "nfc:user");
        let call = |request: Request| client.send(user, &request).unwrap();
        let on = |command: u32| Request::new(command).arg(0u64);

        assert_eq!(call(on(410)).result, NFC_NOT_INITIALIZED);
        for (initialize, get_state) in [(0, 2), (400, 402)] {
            assert_eq!(call(Request::new(initialize)).result, ResultCode::SUCCESS);
            assert_eq!(call(Request::new(get_state)).pop::<u32>(), 1);
        }
        for command in [3, 403] {
            assert_eq!(call(Request::new(command)).pop::<u8>(), 1, "nfc enabled");
        }
        assert_eq!(list_devices(&client, user, 404).1, [0]);
        assert_eq!(call(on(405)).pop::<u32>(), 0);
        assert_eq!(call(on(406)).pop::<u32>(), 0);
        assert_eq!(call(on(408).arg(1u32)).result, ResultCode::SUCCESS);
        assert_eq!(call(on(405)).pop::<u32>(), 1, "searching");
        for command in [410, 1000, 1001] {
            assert_eq!(
                call(on(command)).result,
                NFC_WRONG_DEVICE_STATE,
                "{command}"
            );
        }
        for command in [407, 411, 412] {
            let reply = call(on(command));
            assert_eq!(reply.result, ResultCode::SUCCESS, "{command}");
            assert!(is_quiet_event(&client, &reply), "event {command}");
        }
        let out = client.alloc(&[0xff; 4]);
        let mut reply = call(on(1300).arg(0i64).buffer_b(out, 4));
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u32>(), 1);
        assert_eq!(client.read(out, 1), [0]);
        assert_eq!(call(on(409)).result, ResultCode::SUCCESS);

        disconnect(&context);
        assert_eq!(
            list_devices(&client, user, 404).0.result,
            NFC_DEVICE_NOT_FOUND
        );
        assert_eq!(call(on(408)).result, NFC_WRONG_DEVICE_STATE, "unavailable");
        assert_eq!(call(on(411)).result, NFC_DEVICE_NOT_FOUND);
        assert_eq!(call(Request::new(401)).result, ResultCode::SUCCESS);
        assert_eq!(call(Request::new(402)).pop::<u32>(), 0);
    }

    #[test]
    fn mifare_user_detects_no_card() {
        let (_context, client) = booted(
            SWITCH1_TITLE,
            Some(ControllerType::RightJoycon),
            super::register,
        );
        let user = user(&client, "nfc:mf:u");
        let call = |request: Request| client.send(user, &request).unwrap();
        let on = |command: u32| Request::new(command).arg(0u64);

        assert_eq!(
            call(on(5)).result,
            NFP_NFC_DISABLED,
            "nfp's module, as the reference"
        );
        assert_eq!(call(Request::new(0)).result, ResultCode::SUCCESS);
        assert_eq!(call(Request::new(10)).pop::<u32>(), 1);
        assert_eq!(list_devices(&client, user, 2).1, [0]);
        assert_eq!(call(on(3)).result, ResultCode::SUCCESS);
        assert_eq!(call(on(11)).pop::<u32>(), 1);
        assert_eq!(call(on(12)).pop::<u32>(), 0);
        for command in [5, 6, 7] {
            assert_eq!(
                call(on(command)).result,
                NFP_WRONG_DEVICE_STATE,
                "{command}"
            );
        }
        for command in [8, 9, 13] {
            let reply = call(on(command));
            assert_eq!(reply.result, ResultCode::SUCCESS, "{command}");
            assert!(is_quiet_event(&client, &reply), "event {command}");
        }
        assert_eq!(call(on(4)).result, ResultCode::SUCCESS);
        assert_eq!(call(Request::new(1)).result, ResultCode::SUCCESS);
        assert_eq!(call(Request::new(10)).pop::<u32>(), 0);
    }

    /// Waits for the hid updater to signal `event`, with no NFC command in between.
    fn wait_signaled(client: &lichii_kernel::testing::IpcClient, event: Handle) {
        let Some(KObject::ReadableEvent(event)) = client.object(event) else {
            panic!("not a readable event");
        };
        let deadline = Instant::now() + Duration::from_secs(5);
        while !event.is_signaled() {
            assert!(Instant::now() < deadline, "availability event never fired");
            std::thread::sleep(Duration::from_millis(1));
        }
        event.clear();
    }

    #[test]
    fn controller_changes_signal_the_availability_event() {
        let (context, client) = booted(SWITCH1_TITLE, None, |context| {
            crate::hid::register(context)?;
            super::register(context)
        });
        let hid = client.connect("hid").unwrap();
        let resource = client
            .send(hid, &Request::new(0).send_pid().arg(0x51u64))
            .unwrap()
            .move_handles[0];
        assert_eq!(
            client.send(resource, &Request::new(0)).unwrap().result,
            ResultCode::SUCCESS,
            "hid shared memory starts the updater"
        );

        let user = user(&client, "nfc:user");
        client.send(user, &Request::new(400)).unwrap();
        let reply = client.send(user, &Request::new(407)).unwrap();
        assert!(is_quiet_event(&client, &reply));
        let event = reply.copy_handles[0];
        assert_eq!(
            list_devices(&client, user, 404).0.result,
            NFC_DEVICE_NOT_FOUND
        );

        connect(&context, ControllerType::ProController);
        wait_signaled(&client, event);
        assert_eq!(list_devices(&client, user, 404).1, [0], "connected");

        disconnect(&context);
        wait_signaled(&client, event);
        assert_eq!(
            list_devices(&client, user, 404).0.result,
            NFC_DEVICE_NOT_FOUND
        );
    }

    #[test]
    fn disabling_nfc_in_set_sys_disables_every_interface() {
        let dir = TempDir::new("nfc-disabled");
        let (_context, client) = booted_with(
            SWITCH1_TITLE,
            dir.settings(),
            Some(ControllerType::ProController),
            |context| {
                crate::set::register(context)?;
                super::register(context)?;
                crate::nfp::register(context)
            },
        );
        let set_sys = client.connect("set:sys").unwrap();
        let reply = client.send(set_sys, &Request::new(70).arg(0u8)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);

        let nfc = user(&client, "nfc:user");
        assert_eq!(client.send(nfc, &Request::new(403)).unwrap().pop::<u8>(), 0);
        client.send(nfc, &Request::new(400)).unwrap();
        assert_eq!(list_devices(&client, nfc, 404).0.result, NFC_DISABLED);
        for command in [406, 408, 410, 411] {
            let reply = client.send(nfc, &Request::new(command).arg(0u64)).unwrap();
            assert_eq!(reply.result, NFC_DISABLED, "nfc {command}");
        }

        for (service, initialize, list, mount) in [("nfp:user", 0, 2, 5), ("nfc:mf:u", 0, 2, 5)] {
            let user = user(&client, service);
            client.send(user, &Request::new(initialize)).unwrap();
            assert_eq!(
                list_devices(&client, user, list).0.result,
                NFP_NFC_DISABLED,
                "{service}"
            );
            let reply = client.send(user, &Request::new(mount).arg(0u64)).unwrap();
            assert_eq!(reply.result, NFP_NFC_DISABLED, "{service} {mount}");
        }

        client.send(set_sys, &Request::new(70).arg(1u8)).unwrap();
        assert_eq!(list_devices(&client, nfc, 404).1, [0], "enabled again");
    }

    #[test]
    fn nfc_interfaces_auto_stub_unknown_commands_only_for_switch2() {
        for (title, expected) in [
            (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
            (SWITCH2_TITLE, ResultCode::SUCCESS),
        ] {
            let (_context, client) = booted(title, None, super::register);
            let nfc = user(&client, "nfc:user");
            let reply = client.send(nfc, &Request::new(1301)).unwrap();
            assert_eq!(
                reply.result, expected,
                "KeepPassThroughSession on {title:#x}"
            );
            let mifare = user(&client, "nfc:mf:u");
            let reply = client.send(mifare, &Request::new(14)).unwrap();
            assert_eq!(reply.result, expected, "mifare 14 on {title:#x}");
        }
    }
}
