//! nifm:a/s/u: network interface management with no network. Requests are never accepted,
//! profiles are empty and scans find nothing, as the C++ answers without a host connection.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;

use crate::ns::Placeholder;
use crate::{Service, ServiceContext};

const PENDING_CONNECTION: ResultCode = ResultCode::new(110, 111);
const INVALID_INPUT: ResultCode = ResultCode::new(110, 112);
const NETWORK_COMMUNICATION_DISABLED: ResultCode = ResultCode::new(110, 1111);
/// `RequestState`: 1 is both NotSubmitted and Invalid.
const STATE_NOT_SUBMITTED: u32 = 1;
const STATE_ON_HOLD: u32 = 2;
const SF_NETWORK_PROFILE_SIZE: usize = 0x17c;
const NIFM_NETWORK_PROFILE_SIZE: usize = 0x18e;
const ACCESS_POINT_SIZE: usize = 52;
/// `IpAddressSetting` and `DnsSetting` (0xd + 0x9 bytes), padded to words.
const IP_CONFIG_INFO_WORDS: usize = 6;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let wireless = Arc::new(AtomicBool::new(true));
    for name in ["nifm:a", "nifm:s", "nifm:u"] {
        let wireless = wireless.clone();
        context.register(name, "nifm", move |context| NetworkInterface {
            name,
            context: context.clone(),
            wireless: wireless.clone(),
        })?;
    }
    Ok(())
}

struct NetworkInterface {
    name: &'static str,
    context: Arc<ServiceContext>,
    wireless: Arc<AtomicBool>,
}

impl Service for NetworkInterface {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if !matches!(ctx.command(), 4 | 5) {
            return None;
        }
        let general = GeneralService {
            context: self.context.clone(),
            wireless: self.wireless.clone(),
        };
        ctx.push_interface(self.context.handler(general));
        Some(ResultCode::SUCCESS)
    }
}

fn write_zeroed(ctx: &mut HleRequestContext, len: usize) -> ResultCode {
    match ctx.write_buffer(0, &vec![0; len]) {
        Ok(_) => ResultCode::SUCCESS,
        Err(code) => code,
    }
}

/// IGeneralService.
struct GeneralService {
    context: Arc<ServiceContext>,
    /// Whether wireless communication is on (the C++ airplane mode, inverted).
    wireless: Arc<AtomicBool>,
}

impl Service for GeneralService {
    fn name(&self) -> &str {
        "IGeneralService"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => {
                ctx.push(1u64);
                ResultCode::SUCCESS
            }
            2 => {
                ctx.push_interface(self.context.handler(ScanRequest::new()));
                ResultCode::SUCCESS
            }
            4 => {
                ctx.push_interface(self.context.handler(Request::new()));
                ResultCode::SUCCESS
            }
            5 | 8 => write_zeroed(ctx, SF_NETWORK_PROFILE_SIZE),
            6 | 7 | 11 | 35 | 47 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            9 => {
                let len = ctx.readable_buffer(0).map_or(0, |b| b.size as usize);
                if len < NIFM_NETWORK_PROFILE_SIZE {
                    INVALID_INPUT
                } else {
                    ResultCode::SUCCESS
                }
            }
            10 | 19 | 33 | 34 => ResultCode::SUCCESS,
            12 => {
                ctx.push([0u8; 4]);
                ResultCode::SUCCESS
            }
            14 => {
                let profile = ctx.read_buffer(0).unwrap_or_default();
                let mut uuid = [0u8; 16];
                if let Some(bytes) = profile.get(8..24) {
                    uuid.copy_from_slice(bytes);
                }
                ctx.push_interface(self.context.handler(Placeholder("INetworkProfile")));
                ctx.push(uuid);
                ResultCode::SUCCESS
            }
            15 => {
                ctx.push([0u32; IP_CONFIG_INFO_WORDS]);
                ResultCode::SUCCESS
            }
            16 => {
                let enable: u8 = ctx.pop();
                self.wireless.store(enable != 0, Ordering::Relaxed);
                ResultCode::SUCCESS
            }
            17 => {
                ctx.push(u8::from(self.wireless.load(Ordering::Relaxed)));
                ResultCode::SUCCESS
            }
            18 => {
                ctx.push([1u8, 0, 0]);
                ResultCode::SUCCESS
            }
            20..=22 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            25 => {
                ctx.push(0u32);
                ResultCode::SUCCESS
            }
            36 => write_zeroed(ctx, ACCESS_POINT_SIZE),
            _ => return None,
        })
    }
}

/// IScanRequest: a scan finds no access point and finishes at once.
struct ScanRequest {
    complete: Arc<Event>,
    processing: Arc<Event>,
    result: Mutex<ResultCode>,
}

impl ScanRequest {
    fn new() -> Self {
        Self {
            complete: Arc::new(Event::new()),
            processing: Arc::new(Event::new()),
            result: Mutex::new(PENDING_CONNECTION),
        }
    }
}

impl Service for ScanRequest {
    fn name(&self) -> &str {
        "IScanRequest"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                self.processing.signal();
                *self.result.lock().unwrap() = PENDING_CONNECTION;
                self.complete.signal();
                ResultCode::SUCCESS
            }
            1 => {
                ctx.push(0u8);
                ResultCode::SUCCESS
            }
            2 => *self.result.lock().unwrap(),
            3 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.complete.clone()));
                ctx.push_copy_object(KObject::ReadableEvent(self.processing.clone()));
                ResultCode::SUCCESS
            }
            4 => ResultCode::SUCCESS,
            _ => return None,
        })
    }
}

/// IRequest: a request for the network that is never accepted.
struct Request {
    state: AtomicU32,
    event1: Arc<Event>,
    event2: Arc<Event>,
}

impl Request {
    fn new() -> Self {
        Self {
            state: AtomicU32::new(STATE_NOT_SUBMITTED),
            event1: Arc::new(Event::new()),
            event2: Arc::new(Event::new()),
        }
    }

    fn update(&self, state: u32) {
        self.state.store(state, Ordering::Relaxed);
        self.event1.signal();
    }
}

impl Service for Request {
    fn name(&self) -> &str {
        "IRequest"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => {
                ctx.push(self.state.load(Ordering::Relaxed));
                ResultCode::SUCCESS
            }
            1 => match self.state.load(Ordering::Relaxed) {
                STATE_NOT_SUBMITTED => NETWORK_COMMUNICATION_DISABLED,
                STATE_ON_HOLD => {
                    self.update(STATE_NOT_SUBMITTED);
                    PENDING_CONNECTION
                }
                _ => ResultCode::SUCCESS,
            },
            2 => {
                ctx.push_copy_object(KObject::ReadableEvent(self.event1.clone()));
                ctx.push_copy_object(KObject::ReadableEvent(self.event2.clone()));
                ResultCode::SUCCESS
            }
            3 | 6 | 11 => ResultCode::SUCCESS,
            4 => {
                if self.state.load(Ordering::Relaxed) == STATE_NOT_SUBMITTED {
                    self.update(STATE_ON_HOLD);
                }
                ResultCode::SUCCESS
            }
            9 => {
                let ssid_length: u32 = ctx.pop();
                if ssid_length > 0x20 {
                    INVALID_INPUT
                } else {
                    ResultCode(1)
                }
            }
            21 => {
                let len = ctx.writable_buffer(0).map_or(0, |b| b.size as usize);
                if len != 0 && write_zeroed(ctx, len) != ResultCode::SUCCESS {
                    return Some(write_zeroed(ctx, len));
                }
                ctx.push([0u32; 3]);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::Settings;

    use super::*;
    use crate::testing::{self, Guest, ReplyExt, Request as Ipc};

    fn general() -> (Guest, u32) {
        let guest = Guest::new(testing::image(), Settings::default(), |context| {
            register(context).unwrap()
        });
        guest.service("nifm:a");
        guest.service("nifm:s");
        let nifm = guest.service("nifm:u");
        guest.call(nifm, Ipc::new(4)).ok().interface();
        let general = guest.call(nifm, Ipc::new(5)).ok().interface();
        (guest, general)
    }

    #[test]
    fn requests_are_never_accepted() {
        let (guest, general) = general();
        let request = guest.call(general, Ipc::new(4)).ok().interface();
        let state = |guest: &Guest| guest.call(request, Ipc::new(0)).ok().get::<u32>(0);
        assert_eq!(state(&guest), STATE_NOT_SUBMITTED);
        assert_eq!(
            guest.call(request, Ipc::new(1)).result,
            NETWORK_COMMUNICATION_DISABLED
        );
        let events = guest.call(request, Ipc::new(2)).ok();
        assert_eq!(events.copy_handles.len(), 2);
        guest.call(request, Ipc::new(6).arg(0x2000u32)).ok();
        guest.call(request, Ipc::new(4)).ok();
        assert!(guest.is_signaled(events.copy_handles[0]));
        assert_eq!(state(&guest), STATE_ON_HOLD);
        assert_eq!(guest.call(request, Ipc::new(1)).result, PENDING_CONNECTION);
        assert_eq!(state(&guest), STATE_NOT_SUBMITTED);
        assert_eq!(
            guest.call(request, Ipc::new(1)).result,
            NETWORK_COMMUNICATION_DISABLED
        );
        assert_eq!(
            guest.call(request, Ipc::new(9).arg(0x21u32)).result,
            INVALID_INPUT
        );
        assert_eq!(
            guest.call(request, Ipc::new(9).arg(4u32)).result,
            ResultCode(1)
        );

        let out = guest.heap;
        guest.write(out, &[0xaa; 0x40]);
        let info = guest.call(request, Ipc::new(21).buffer_b(out, 0x40)).ok();
        assert_eq!(info.get::<[u32; 3]>(0), [0; 3]);
        assert!(guest.read(out, 0x40).iter().all(|&b| b == 0));
    }

    #[test]
    fn the_network_is_not_connected() {
        let (guest, general) = general();
        assert_eq!(guest.call(general, Ipc::new(1)).ok().get::<u64>(0), 1);
        let out = guest.heap;
        for (command, size) in [
            (5, SF_NETWORK_PROFILE_SIZE),
            (8, SF_NETWORK_PROFILE_SIZE),
            (36, ACCESS_POINT_SIZE),
        ] {
            guest.write(out, &[0xaa; 0x200]);
            guest
                .call(general, Ipc::new(command).buffer_b(out, size as u64))
                .ok();
            assert!(
                guest.read(out, size).iter().all(|&b| b == 0),
                "command {command}"
            );
        }
        for command in [6, 7, 11, 25, 35, 47] {
            let reply = guest
                .call(general, Ipc::new(command).buffer_b(out, 0x100))
                .ok();
            assert_eq!(reply.get::<u32>(0), 0, "command {command}");
        }
        for command in [20, 21, 22] {
            assert_eq!(guest.call(general, Ipc::new(command)).ok().get::<u8>(0), 0);
        }
        assert_eq!(
            guest.call(general, Ipc::new(12)).ok().get::<[u8; 4]>(0),
            [0; 4]
        );
        let config = guest.call(general, Ipc::new(15)).ok();
        assert_eq!(config.get::<[u8; 22]>(0), [0; 22]);
        let status = guest.call(general, Ipc::new(18)).ok();
        assert_eq!(status.get::<[u8; 3]>(0), [1, 0, 0]);

        assert_eq!(guest.call(general, Ipc::new(17)).ok().get::<u8>(0), 1);
        guest.call(general, Ipc::new(16).arg(0u8)).ok();
        assert_eq!(guest.call(general, Ipc::new(17)).ok().get::<u8>(0), 0);

        let profile = guest.heap + 0x1000;
        let mut data = vec![0u8; NIFM_NETWORK_PROFILE_SIZE];
        data[8..24].copy_from_slice(&[9; 16]);
        guest.write(profile, &data);
        assert_eq!(
            guest
                .call(general, Ipc::new(9).buffer_a(profile, 0x100))
                .result,
            INVALID_INPUT
        );
        guest
            .call(
                general,
                Ipc::new(9).buffer_a(profile, NIFM_NETWORK_PROFILE_SIZE as u64),
            )
            .ok();
        let temporary = guest
            .call(
                general,
                Ipc::new(14).buffer_a(profile, SF_NETWORK_PROFILE_SIZE as u64),
            )
            .ok();
        temporary.interface();
        assert_eq!(temporary.get::<[u8; 16]>(0), [9; 16]);
    }

    #[test]
    fn scans_find_nothing() {
        let (guest, general) = general();
        let scan = guest.call(general, Ipc::new(2)).ok().interface();
        let events = guest.call(scan, Ipc::new(3)).ok();
        assert_eq!(events.copy_handles.len(), 2);
        assert_eq!(guest.call(scan, Ipc::new(2)).result, PENDING_CONNECTION);
        guest.call(scan, Ipc::new(0)).ok();
        assert!(guest.is_signaled(events.copy_handles[0]));
        assert_eq!(guest.call(scan, Ipc::new(1)).ok().get::<u8>(0), 0);
        assert_eq!(guest.call(scan, Ipc::new(2)).result, PENDING_CONNECTION);
    }
}
