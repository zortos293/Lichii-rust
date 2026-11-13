//! nifm:a/s/u: network interface management. With `settings.system.network_enabled` off (the
//! default), or wireless communication turned off, there is no network: requests are never
//! accepted and profiles are empty, as the C++ answers without a host connection. With it on,
//! nifm reports what the C++ does for a wired host connection: requests are accepted and the
//! current profile and IP configuration carry the host's IPv4 address. Scans find nothing either
//! way, and bsd still creates no host sockets.

use std::net::{IpAddr, Ipv4Addr, UdpSocket};
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
const STATE_ACCEPTED: u32 = 3;
/// `NetworkInterfaceType::Ethernet`.
const INTERFACE_ETHERNET: u8 = 2;
/// `InternetConnectionStatus::Connected`.
const STATUS_CONNECTED: u8 = 4;
const ETHERNET_BARS: u8 = 3;
const PRIMARY_DNS: [u8; 4] = [1, 1, 1, 1];
const PROFILE_SECONDARY_DNS: [u8; 4] = [1, 0, 0, 1];
const CONFIG_SECONDARY_DNS: [u8; 4] = [8, 8, 8, 8];
const PROFILE_UUID_OFFSET: usize = 0xc2;
const PROFILE_TYPE_OFFSET: usize = 0x112;
const SF_NETWORK_PROFILE_SIZE: usize = 0x17c;
const NIFM_NETWORK_PROFILE_SIZE: usize = 0x18e;
const ACCESS_POINT_SIZE: usize = 52;
/// `IpAddressSetting` and `DnsSetting` (0xd + 0x9 bytes), padded to words.
const IP_CONFIG_INFO_WORDS: usize = 6;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let host = context.settings.system.network_enabled.then(Host::discover);
    let network = Arc::new(Network {
        host,
        wireless: AtomicBool::new(true),
    });
    for name in ["nifm:a", "nifm:s", "nifm:u"] {
        let network = network.clone();
        context.register(name, "nifm", move |context| NetworkInterface {
            name,
            context: context.clone(),
            network: network.clone(),
        })?;
    }
    Ok(())
}

/// The host's IPv4 configuration, reported as a wired connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Host {
    ip: Ipv4Addr,
    mask: Ipv4Addr,
    gateway: Ipv4Addr,
}

impl Host {
    /// The address the host routes outgoing traffic from (found by connecting a UDP socket,
    /// which sends nothing), on a /24 whose `.1` is the gateway. Without a route it is the
    /// loopback address, so an enabled network still reads as connected.
    fn discover() -> Self {
        let routed = UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
            .and_then(|socket| {
                socket.connect((Ipv4Addr::from(PRIMARY_DNS), 53))?;
                socket.local_addr()
            })
            .ok()
            .and_then(|address| match address.ip() {
                IpAddr::V4(ip) if !ip.is_unspecified() => Some(ip),
                _ => None,
            });
        let ip = routed.unwrap_or_else(|| {
            tracing::warn!("the host has no IPv4 route; nifm reports the loopback address");
            Ipv4Addr::LOCALHOST
        });
        let [a, b, c, _] = ip.octets();
        Self {
            ip,
            mask: Ipv4Addr::new(255, 255, 255, 0),
            gateway: Ipv4Addr::new(a, b, c, 1),
        }
    }

    /// `IpAddressSetting`: automatic, address, mask, gateway.
    fn ip_setting(&self) -> [u8; 0xd] {
        let mut setting = [0; 0xd];
        setting[0] = 1;
        setting[1..5].copy_from_slice(&self.ip.octets());
        setting[5..9].copy_from_slice(&self.mask.octets());
        setting[9..13].copy_from_slice(&self.gateway.octets());
        setting
    }
}

/// `DnsSetting`: automatic, primary, secondary.
fn dns_setting(secondary: [u8; 4]) -> [u8; 9] {
    let mut setting = [0; 9];
    setting[0] = 1;
    setting[1..5].copy_from_slice(&PRIMARY_DNS);
    setting[5..9].copy_from_slice(&secondary);
    setting
}

/// The profile uuid the C++ derives from the SSID, which a wired connection does not have.
fn profile_uuid(ssid: &[u8]) -> [u8; 16] {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x100_0000_01b3;
    let (mut h1, mut h2) = (OFFSET, OFFSET ^ 0x9e37_79b9_7f4a_7c15);
    for &c in ssid {
        h1 = (h1 ^ u64::from(c)).wrapping_mul(PRIME);
        h2 = (h2 ^ u64::from(c ^ 0x5a)).wrapping_mul(PRIME);
    }
    if h1 == 0 && h2 == 0 {
        h1 = 1;
    }
    let mut uuid = [0; 16];
    uuid[..8].copy_from_slice(&h1.to_le_bytes());
    uuid[8..].copy_from_slice(&h2.to_le_bytes());
    uuid
}

/// What every nifm session shares.
struct Network {
    /// The host connection, when `settings.system.network_enabled`.
    host: Option<Host>,
    /// Whether wireless communication is on (the C++ airplane mode, inverted).
    wireless: AtomicBool,
}

impl Network {
    /// The connection titles see: the host's, unless the network or wireless is off.
    fn connection(&self) -> Option<Host> {
        self.host.filter(|_| self.wireless.load(Ordering::Relaxed))
    }

    /// `SfNetworkProfileData` of the current network: zeroed without one.
    fn current_profile(&self) -> [u8; SF_NETWORK_PROFILE_SIZE] {
        let mut profile = [0; SF_NETWORK_PROFILE_SIZE];
        if let Some(host) = self.connection() {
            profile[..0xd].copy_from_slice(&host.ip_setting());
            profile[0xd..0x16].copy_from_slice(&dns_setting(PROFILE_SECONDARY_DNS));
            profile[PROFILE_UUID_OFFSET..][..16].copy_from_slice(&profile_uuid(b""));
            profile[PROFILE_TYPE_OFFSET + 1] = INTERFACE_ETHERNET;
        }
        profile
    }

    /// `IpAddressSetting` and `DnsSetting`, padded to words: zeroed without a network.
    fn ip_config_info(&self) -> [u8; IP_CONFIG_INFO_WORDS * 4] {
        let mut info = [0; IP_CONFIG_INFO_WORDS * 4];
        if let Some(host) = self.connection() {
            info[..0xd].copy_from_slice(&host.ip_setting());
            info[0xd..0x16].copy_from_slice(&dns_setting(CONFIG_SECONDARY_DNS));
        }
        info
    }
}

struct NetworkInterface {
    name: &'static str,
    context: Arc<ServiceContext>,
    network: Arc<Network>,
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
            network: self.network.clone(),
        };
        ctx.push_interface(self.context.handler(general));
        Some(ResultCode::SUCCESS)
    }
}

fn write_zeroed(ctx: &mut HleRequestContext, len: usize) -> ResultCode {
    write(ctx, &vec![0; len])
}

fn write(ctx: &mut HleRequestContext, data: &[u8]) -> ResultCode {
    match ctx.write_buffer(0, data) {
        Ok(_) => ResultCode::SUCCESS,
        Err(code) => code,
    }
}

/// IGeneralService.
struct GeneralService {
    context: Arc<ServiceContext>,
    network: Arc<Network>,
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
                ctx.push_interface(self.context.handler(Request::new(self.network.clone())));
                ResultCode::SUCCESS
            }
            5 => write(ctx, &self.network.current_profile()),
            8 => write_zeroed(ctx, SF_NETWORK_PROFILE_SIZE),
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
                let ip = self
                    .network
                    .connection()
                    .map_or([0; 4], |host| host.ip.octets());
                ctx.push(ip);
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
                ctx.push(self.network.ip_config_info());
                ResultCode::SUCCESS
            }
            16 => {
                let enable: u8 = ctx.pop();
                self.network.wireless.store(enable != 0, Ordering::Relaxed);
                ResultCode::SUCCESS
            }
            17 => {
                ctx.push(u8::from(self.network.wireless.load(Ordering::Relaxed)));
                ResultCode::SUCCESS
            }
            18 => {
                let status = match self.network.connection() {
                    Some(_) => [INTERFACE_ETHERNET, ETHERNET_BARS, STATUS_CONNECTED],
                    None => [1, 0, 0],
                };
                ctx.push(status);
                ResultCode::SUCCESS
            }
            20 | 21 => {
                ctx.push(u8::from(self.network.connection().is_some()));
                ResultCode::SUCCESS
            }
            22 => {
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

/// IRequest: a request for the network, accepted once submitted when there is a connection.
struct Request {
    network: Arc<Network>,
    state: AtomicU32,
    event1: Arc<Event>,
    event2: Arc<Event>,
}

impl Request {
    fn new(network: Arc<Network>) -> Self {
        Self {
            network,
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
            1 => {
                let connected = self.network.connection().is_some();
                match self.state.load(Ordering::Relaxed) {
                    STATE_NOT_SUBMITTED if connected => ResultCode::SUCCESS,
                    STATE_NOT_SUBMITTED => NETWORK_COMMUNICATION_DISABLED,
                    STATE_ON_HOLD => {
                        self.update(if connected {
                            STATE_ACCEPTED
                        } else {
                            STATE_NOT_SUBMITTED
                        });
                        PENDING_CONNECTION
                    }
                    _ => ResultCode::SUCCESS,
                }
            }
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
        general_with(false)
    }

    fn general_with(network_enabled: bool) -> (Guest, u32) {
        let mut settings = Settings::default();
        settings.system.network_enabled = network_enabled;
        let guest = Guest::new(testing::image(), settings, |context| {
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

    #[test]
    fn an_enabled_network_is_a_wired_host_connection() {
        let (guest, general) = general_with(true);
        let request = guest.call(general, Ipc::new(4)).ok().interface();
        let state = |guest: &Guest| guest.call(request, Ipc::new(0)).ok().get::<u32>(0);
        guest.call(request, Ipc::new(1)).ok();
        guest.call(request, Ipc::new(4)).ok();
        assert_eq!(state(&guest), STATE_ON_HOLD);
        assert_eq!(guest.call(request, Ipc::new(1)).result, PENDING_CONNECTION);
        assert_eq!(state(&guest), STATE_ACCEPTED);
        guest.call(request, Ipc::new(1)).ok();

        let status = guest.call(general, Ipc::new(18)).ok();
        assert_eq!(status.get::<[u8; 3]>(0), [2, 3, 4]);
        for command in [20, 21] {
            assert_eq!(guest.call(general, Ipc::new(command)).ok().get::<u8>(0), 1);
        }
        assert_eq!(guest.call(general, Ipc::new(22)).ok().get::<u8>(0), 0);

        let ip = guest.call(general, Ipc::new(12)).ok().get::<[u8; 4]>(0);
        assert_ne!(ip, [0; 4]);
        let config = guest.call(general, Ipc::new(15)).ok().get::<[u8; 22]>(0);
        assert_eq!(config[0], 1);
        assert_eq!(config[1..5], ip);
        assert_eq!(config[5..9], [255, 255, 255, 0]);
        assert_eq!(config[9..13], [ip[0], ip[1], ip[2], 1]);
        assert_eq!(config[13..], [1, 1, 1, 1, 1, 8, 8, 8, 8]);

        let out = guest.heap;
        guest
            .call(
                general,
                Ipc::new(5).buffer_b(out, SF_NETWORK_PROFILE_SIZE as u64),
            )
            .ok();
        let profile = guest.read(out, SF_NETWORK_PROFILE_SIZE);
        assert_eq!(profile[..0xd], config[..0xd]);
        assert_eq!(profile[0xd..0x16], [1, 1, 1, 1, 1, 1, 0, 0, 1]);
        assert_eq!(profile[PROFILE_UUID_OFFSET..][..16], profile_uuid(b""));
        assert_eq!(profile[PROFILE_TYPE_OFFSET..][..2], [0, INTERFACE_ETHERNET]);

        guest.call(general, Ipc::new(16).arg(0u8)).ok();
        assert_eq!(guest.call(general, Ipc::new(21)).ok().get::<u8>(0), 0);
        assert_eq!(
            guest.call(general, Ipc::new(12)).ok().get::<[u8; 4]>(0),
            [0; 4]
        );
        let offline = guest.call(general, Ipc::new(4)).ok().interface();
        assert_eq!(
            guest.call(offline, Ipc::new(1)).result,
            NETWORK_COMMUNICATION_DISABLED
        );
    }

    #[test]
    fn the_profile_uuid_hashes_the_ssid_like_the_cpp() {
        let uuid = profile_uuid(b"");
        assert_eq!(uuid[..8], 0xcbf2_9ce4_8422_2325u64.to_le_bytes());
        assert_eq!(
            uuid[8..],
            (0xcbf2_9ce4_8422_2325u64 ^ 0x9e37_79b9_7f4a_7c15).to_le_bytes()
        );
        assert_ne!(profile_uuid(b"home"), uuid);
    }
}
