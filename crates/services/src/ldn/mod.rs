//! `ldn:u`: local wireless play. Lichii has no host network for it, so the communication
//! service is never initialized: `Initialize` fails with airplane mode, the state reads as an
//! error and network queries fail with a bad state, as the reference answers without a network
//! interface.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::{Service, ServiceContext};

const MODULE: u32 = 203;
const ADVERTISE_DATA_TOO_LARGE: ResultCode = ResultCode::new(MODULE, 10);
const AIRPLANE_MODE_ENABLED: ResultCode = ResultCode::new(MODULE, 23);
const BAD_STATE: ResultCode = ResultCode::new(MODULE, 32);
const NO_IP_ADDRESS: ResultCode = ResultCode::new(MODULE, 33);
const INVALID_BUFFER_COUNT: ResultCode = ResultCode::new(MODULE, 50);
const BAD_INPUT: ResultCode = ResultCode::new(MODULE, 96);

/// `State::Error`, what an uninitialized service reports.
const STATE_ERROR: u32 = 6;
const NETWORK_INFO_SIZE: u64 = 0x480;
const NODE_LATEST_UPDATE_SIZE: u64 = 0x8;
const NODE_COUNT_MAX: u64 = 8;
const ADVERTISE_DATA_SIZE_MAX: usize = 384;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    context.register("ldn:u", "services", |context| UserServiceCreator {
        context: context.clone(),
    })
}

struct UserServiceCreator {
    context: Arc<ServiceContext>,
}

impl Service for UserServiceCreator {
    fn name(&self) -> &str {
        "ldn:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => ctx.push_interface(self.context.handler(LocalCommunication {
                state_change: Arc::new(Event::new()),
            })),
            1 => ctx.push_interface(self.context.handler(ClientProcessMonitor)),
            _ => return None,
        }
        Some(ResultCode::SUCCESS)
    }
}

/// `IClientProcessMonitor`: `RegisterClient` (0) takes the process id.
struct ClientProcessMonitor;

impl Service for ClientProcessMonitor {
    fn name(&self) -> &str {
        "IClientProcessMonitor"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        (ctx.command() == 0).then_some(ResultCode::SUCCESS)
    }
}

/// `IUserLocalCommunicationService` with no network. The reference's `DestroyNetwork` and
/// `Disconnect` succeed and signal the state change in any state; they do so here too, but the
/// state stays uninitialized, since no network can be created or joined.
struct LocalCommunication {
    state_change: Arc<Event>,
}

impl Service for LocalCommunication {
    fn name(&self) -> &str {
        "IUserLocalCommunicationService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let output_count = |ctx: &HleRequestContext, index: usize, size: u64| {
            ctx.writable_buffer(index)
                .map_or(0, |buffer| buffer.size / size)
        };
        Some(match ctx.command() {
            0 => {
                ctx.push(STATE_ERROR);
                ResultCode::SUCCESS
            }
            1 | 4 | 5 | 200 | 201 | 202 | 203 | 300 | 301 | 302 => BAD_STATE,
            2 => NO_IP_ADDRESS,
            3 => {
                ctx.push(0i16);
                ResultCode::SUCCESS
            }
            100 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.state_change.clone()));
                ResultCode::SUCCESS
            }
            101 => match output_count(ctx, 1, NODE_LATEST_UPDATE_SIZE) {
                0 => BAD_INPUT,
                count if count > NODE_COUNT_MAX => INVALID_BUFFER_COUNT,
                _ => BAD_STATE,
            },
            102 if output_count(ctx, 0, NETWORK_INFO_SIZE) == 0 => BAD_INPUT,
            103 if output_count(ctx, 0, NETWORK_INFO_SIZE) != 0 => BAD_INPUT,
            102 | 103 => {
                ctx.push(0i16);
                ResultCode::SUCCESS
            }
            104 | 106 | 207 | 208 | 401 => ResultCode::SUCCESS,
            204 | 304 => {
                self.state_change.signal();
                ResultCode::SUCCESS
            }
            206 => match ctx.read_buffer(0) {
                Ok(data) if data.len() > ADVERTISE_DATA_SIZE_MAX => ADVERTISE_DATA_TOO_LARGE,
                Ok(_) => ResultCode::SUCCESS,
                Err(code) => code,
            },
            400 | 402 => AIRPLANE_MODE_ENABLED,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;
    use lichii_kernel::ipc::{Handle, UNKNOWN_COMMAND_ID};
    use lichii_kernel::object::KObject;
    use lichii_kernel::testing::IpcClient;

    use super::*;
    use crate::testing::{Request, SWITCH1_TITLE, SWITCH2_TITLE, booted};

    fn communication(title: u64) -> (IpcClient, Handle) {
        let client = booted(title, Settings::default(), super::register);
        let ldn = client.connect("ldn:u").unwrap();
        let reply = client.send(ldn, &Request::new(0)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let service = reply.move_handles[0];
        (client, service)
    }

    #[test]
    fn without_a_network_initialize_reports_airplane_mode() {
        let (client, service) = communication(SWITCH1_TITLE);
        let send = |request: Request| client.send(service, &request).unwrap();
        assert_eq!(
            send(Request::new(400).send_pid().arg(0u64)).result,
            AIRPLANE_MODE_ENABLED
        );
        assert_eq!(
            send(Request::new(402).send_pid().arg(0u32).arg(0u64)).result,
            AIRPLANE_MODE_ENABLED
        );
        let mut reply = send(Request::new(0));
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u32>(), STATE_ERROR);
        assert_eq!(send(Request::new(2)).result, NO_IP_ADDRESS);
        let mut reply = send(Request::new(3));
        assert_eq!(reply.pop::<i16>(), 0, "DisconnectReason::None");
        for command in [1, 4, 5, 200, 202, 300, 302] {
            assert_eq!(send(Request::new(command)).result, BAD_STATE, "{command}");
        }
        assert_eq!(send(Request::new(401)).result, ResultCode::SUCCESS);
    }

    #[test]
    fn scans_find_nothing_and_buffers_are_checked() {
        let (client, service) = communication(SWITCH1_TITLE);
        let send = |request: Request| client.send(service, &request).unwrap();
        let networks = client.alloc(&[0; 0x900]);
        let scan = |command: u32, size: u64| {
            Request::new(command)
                .arg(0i16)
                .arg([0u8; 0x60])
                .buffer_b(networks, size)
        };
        let mut reply = send(scan(102, 0x900));
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<i16>(), 0);
        assert_eq!(send(scan(102, 0)).result, BAD_INPUT);
        assert_eq!(send(scan(103, 0x900)).result, BAD_INPUT);
        assert_eq!(send(scan(103, 0)).result, ResultCode::SUCCESS);

        let info = client.alloc(&[0; 0x480]);
        let updates = client.alloc(&[0; 0x48]);
        let latest = |count: u64| {
            Request::new(101)
                .buffer_c(info, 0x480)
                .buffer_c(updates, count * 8)
        };
        assert_eq!(send(latest(0)).result, BAD_INPUT);
        assert_eq!(send(latest(9)).result, INVALID_BUFFER_COUNT);
        assert_eq!(send(latest(8)).result, BAD_STATE);

        let data = client.alloc(&[0; 385]);
        let advertise = |size: u64| Request::new(206).buffer_a(data, size);
        assert_eq!(send(advertise(384)).result, ResultCode::SUCCESS);
        assert_eq!(send(advertise(385)).result, ADVERTISE_DATA_TOO_LARGE);
    }

    #[test]
    fn disconnecting_signals_the_state_change() {
        let (client, service) = communication(SWITCH1_TITLE);
        let reply = client.send(service, &Request::new(100)).unwrap();
        let Some(KObject::ReadableEvent(event)) = client.object(reply.copy_handles[0]) else {
            panic!("state change event");
        };
        assert!(!event.is_signaled());
        let reply = client.send(service, &Request::new(304)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert!(event.is_signaled());
    }

    #[test]
    fn the_monitor_registers_clients_and_unhandled_commands_stub_only_for_switch2() {
        for (title, expected) in [
            (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
            (SWITCH2_TITLE, ResultCode::SUCCESS),
        ] {
            let client = booted(title, Settings::default(), super::register);
            let ldn = client.connect("ldn:u").unwrap();
            let monitor = client.send(ldn, &Request::new(1)).unwrap().move_handles[0];
            let register = Request::new(0).send_pid().arg(0u64);
            assert_eq!(
                client.send(monitor, &register).unwrap().result,
                ResultCode::SUCCESS
            );
            let (client, service) = communication(title);
            assert_eq!(
                client.send(service, &Request::new(205)).unwrap().result,
                expected
            );
        }
    }
}
