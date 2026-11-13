//! The NFC device manager behind every NFC `IUser` (`nfc:user`, `nfc:mf:u` and `nfp:user`), as
//! the reference's `nfc/common` shares it: the npads with a reader, their device state and the
//! events titles wait on.
//!
//! There is no amiibo or Mifare backend, so no tag is ever found: detection starts and stops,
//! every command that needs a tag answers `WrongDeviceState`, and the activate, deactivate and
//! availability events never fire. Player 1 and handheld have the reference's virtual reader when
//! their controller has one; the other npads are devices that stay unavailable.

use std::sync::Arc;

use lichii_common::settings::ControllerType;
use lichii_common::{HleResult, ResultCode};
use lichii_input::NpadHub;
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::object::KObject;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const NFC_MODULE: u32 = 127;
const NFP_MODULE: u32 = 115;
/// `nfp`'s `NfcDisabled`, which nfp answers for nfc's `NfcNotInitialized`.
const NFP_NFC_DISABLED: u32 = 80;

const DEVICE_COUNT: usize = 10;
const PLAYER_1: u32 = 0;
const HANDHELD: u32 = 0x20;
/// The `NpadIdType` of each device, which is also its handle, in the reference's order
/// (`HID::IndexToNpadIdType`).
const NPAD_IDS: [u32; DEVICE_COUNT] = [PLAYER_1, 1, 2, 3, 4, 5, 6, 7, HANDHELD, 0x10];
/// `sizeof(nn::nfp::ApplicationArea)`.
const APPLICATION_AREA_SIZE: u32 = 0xd8;

/// The service an `IUser` belongs to, which picks the module its errors are reported in.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Backend {
    Nfc,
    Nfp,
    /// `nfc:mf:u`, whose errors the reference reports in nfp's module, like nfp's own.
    Mifare,
}

/// The `nn::nfc` errors a reader without tags answers, by description.
#[derive(Clone, Copy)]
enum Error {
    DeviceNotFound = 64,
    InvalidArgument = 65,
    WrongDeviceState = 73,
    NfcNotInitialized = 77,
}

impl Backend {
    /// `TranslateResultToServiceError`.
    fn result(self, error: Error) -> ResultCode {
        match (self, error) {
            (Self::Nfc, error) => ResultCode::new(NFC_MODULE, error as u32),
            (_, Error::NfcNotInitialized) => ResultCode::new(NFP_MODULE, NFP_NFC_DISABLED),
            (_, error) => ResultCode::new(NFP_MODULE, error as u32),
        }
    }
}

/// `nn::nfc::DeviceState`, the values a device without tags takes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
enum DeviceState {
    Initialized = 0,
    SearchingForTag = 1,
    Unavailable = 5,
    /// What `GetDeviceState` answers for a handle that names no device.
    Finalized = 6,
}

/// The commands every NFC `IUser` shares; each interface maps its command ids to them.
#[derive(Clone, Copy)]
pub(crate) enum Command {
    Initialize,
    Finalize,
    GetState,
    IsNfcEnabled,
    ListDevices,
    GetDeviceState,
    GetNpadId,
    AttachAvailabilityChangeEvent,
    StartDetection,
    StopDetection,
    AttachActivateEvent,
    AttachDeactivateEvent,
    /// nfp's `Mount`, which checks the model type before it looks for the tag.
    Mount,
    /// A command that reads or writes the tag: `GetTagInfo`, `ReadMifare`, `WriteMifare` and
    /// nfp's application area and info commands.
    NeedsTag,
    SendCommandByPassThrough,
    GetApplicationAreaSize,
}

/// One kind of `IUser`: the service that opens it, its name in logs, its backend and its
/// command table.
#[derive(Clone, Copy)]
pub(crate) struct UserKind {
    pub service: &'static str,
    pub interface: &'static str,
    pub backend: Backend,
    pub commands: fn(u32) -> Option<Command>,
}

/// Registers `kind.service`, whose command 0 (`CreateUserInterface` or
/// `CreateUserNfcInterface`) opens a new `IUser` of `kind`.
pub(crate) fn register(context: &Arc<ServiceContext>, kind: UserKind) -> HleResult<()> {
    context.register(kind.service, "services", move |context| UserManager {
        kind,
        context: context.clone(),
    })
}

struct UserManager {
    kind: UserKind,
    context: Arc<ServiceContext>,
}

impl Service for UserManager {
    fn name(&self) -> &str {
        self.kind.service
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if ctx.command() != 0 {
            return None;
        }
        let user = User::new(self.kind, self.context.npad.clone());
        ctx.push_interface(self.context.handler(user));
        Some(ResultCode::SUCCESS)
    }
}

/// An `IUser` and its own device manager, as each reference interface has one.
struct User {
    kind: UserKind,
    hub: Arc<NpadHub>,
    availability_change: Arc<Event>,
    activate: [Arc<Event>; DEVICE_COUNT],
    deactivate: [Arc<Event>; DEVICE_COUNT],
    state: Mutex<State>,
}

struct State {
    initialized: bool,
    devices: [DeviceState; DEVICE_COUNT],
}

impl State {
    const FINALIZED: Self = Self {
        initialized: false,
        devices: [DeviceState::Unavailable; DEVICE_COUNT],
    };
}

/// The npad id whose reader the host's player 1 controller provides (`EmulatedController::HasNfc`):
/// handheld, or player 1 on a style with a right Joy-Con or Pro Controller reader.
fn reader_npad(controller: ControllerType) -> Option<u32> {
    use ControllerType::*;
    match controller {
        ProController | Switch2ProController | DualJoycons | Switch2DualJoycons | RightJoycon
        | Switch2RightJoycon => Some(PLAYER_1),
        Handheld | Switch2Handheld => Some(HANDHELD),
        LeftJoycon | Switch2LeftJoycon | GameCube => None,
    }
}

fn device_index(handle: u64) -> Option<usize> {
    NPAD_IDS.iter().position(|&id| u64::from(id) == handle)
}

impl User {
    fn new(kind: UserKind, hub: Arc<NpadHub>) -> Self {
        let event = |_| Arc::new(Event::new());
        Self {
            kind,
            hub,
            availability_change: Arc::new(Event::new()),
            activate: std::array::from_fn(event),
            deactivate: std::array::from_fn(event),
            state: Mutex::new(State::FINALIZED),
        }
    }

    fn error(&self, error: Error) -> ResultCode {
        self.kind.backend.result(error)
    }

    /// Follows controller changes as the reference's npad callbacks do: a device whose npad lost
    /// its reader becomes unavailable, and one that got it back starts over as initialized.
    fn sync(&self, state: &mut State) {
        if !state.initialized {
            return;
        }
        let host = self.hub.read(0);
        let reader = host
            .connected
            .then(|| reader_npad(host.controller))
            .flatten();
        for (device, &npad_id) in state.devices.iter_mut().zip(&NPAD_IDS) {
            if reader != Some(npad_id) {
                *device = DeviceState::Unavailable;
            } else if *device == DeviceState::Unavailable {
                *device = DeviceState::Initialized;
            }
        }
    }

    /// The device `handle` names, once the manager is initialized (`GetDeviceHandle`).
    fn device(&self, state: &State, handle: u64) -> HleResult<usize> {
        if !state.initialized {
            return Err(self.error(Error::NfcNotInitialized));
        }
        device_index(handle).ok_or_else(|| self.error(Error::DeviceNotFound))
    }

    /// The device `handle` names among the listed ones, for the activate and deactivate events.
    fn listed_device(&self, state: &State, handle: u64) -> HleResult<usize> {
        let index = self.device(state, handle)?;
        if state.devices[index] == DeviceState::Unavailable {
            return Err(self.error(Error::DeviceNotFound));
        }
        Ok(index)
    }

    fn run(&self, command: Command, ctx: &mut HleRequestContext) -> HleResult<()> {
        let mut state = self.state.lock();
        self.sync(&mut state);
        match command {
            Command::Initialize => {
                state.initialized = true;
                state.devices = [DeviceState::Unavailable; DEVICE_COUNT];
                self.sync(&mut state);
            }
            Command::Finalize => *state = State::FINALIZED,
            Command::GetState => ctx.push(u32::from(state.initialized)),
            Command::IsNfcEnabled => ctx.push(1u8),
            Command::ListDevices => {
                let capacity = ctx.writable_buffer(0).map_or(0, |buffer| buffer.size / 8);
                if capacity == 0 {
                    return Err(self.error(Error::InvalidArgument));
                }
                if !state.initialized {
                    return Err(self.error(Error::NfcNotInitialized));
                }
                let mut handles = [0u8; DEVICE_COUNT * 8];
                let mut count = 0;
                for (&device, &npad_id) in state.devices.iter().zip(&NPAD_IDS) {
                    if device != DeviceState::Unavailable && (count as u64) < capacity {
                        handles[count * 8..][..8]
                            .copy_from_slice(&u64::from(npad_id).to_le_bytes());
                        count += 1;
                    }
                }
                if count == 0 {
                    return Err(self.error(Error::DeviceNotFound));
                }
                ctx.write_buffer(0, &handles[..count * 8])?;
                ctx.push(count as i32);
            }
            Command::GetDeviceState => {
                let device = device_index(ctx.pop())
                    .map_or(DeviceState::Finalized, |index| state.devices[index]);
                ctx.push(device as u32);
            }
            Command::GetNpadId => {
                let index = self.device(&state, ctx.pop())?;
                ctx.push(NPAD_IDS[index]);
            }
            Command::AttachAvailabilityChangeEvent => {
                ctx.push_copy_object(KObject::ReadableEvent(self.availability_change.clone()));
            }
            Command::StartDetection => {
                let index = self.device(&state, ctx.pop())?;
                if state.devices[index] != DeviceState::Initialized {
                    return Err(self.error(Error::WrongDeviceState));
                }
                state.devices[index] = DeviceState::SearchingForTag;
            }
            Command::StopDetection => {
                let index = self.device(&state, ctx.pop())?;
                match state.devices[index] {
                    DeviceState::Initialized => {}
                    DeviceState::SearchingForTag => state.devices[index] = DeviceState::Initialized,
                    _ => return Err(self.error(Error::WrongDeviceState)),
                }
            }
            Command::AttachActivateEvent => {
                let index = self.listed_device(&state, ctx.pop())?;
                ctx.push_copy_object(KObject::ReadableEvent(self.activate[index].clone()));
            }
            Command::AttachDeactivateEvent => {
                let index = self.listed_device(&state, ctx.pop())?;
                ctx.push_copy_object(KObject::ReadableEvent(self.deactivate[index].clone()));
            }
            Command::Mount => {
                let handle = ctx.pop::<u64>();
                let model_type = ctx.pop::<u32>();
                self.device(&state, handle)?;
                let error = match model_type {
                    0 => Error::WrongDeviceState,
                    _ => Error::InvalidArgument,
                };
                return Err(self.error(error));
            }
            Command::NeedsTag => {
                self.device(&state, ctx.pop())?;
                return Err(self.error(Error::WrongDeviceState));
            }
            Command::SendCommandByPassThrough => {
                self.device(&state, ctx.pop())?;
                ctx.write_buffer(0, &[0])?;
                ctx.push(1u32);
            }
            Command::GetApplicationAreaSize => ctx.push(APPLICATION_AREA_SIZE),
        }
        Ok(())
    }
}

impl Service for User {
    fn name(&self) -> &str {
        self.kind.interface
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command = (self.kind.commands)(ctx.command())?;
        Some(self.run(command, ctx).err().unwrap_or(ResultCode::SUCCESS))
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::sync::Arc;

    use lichii_common::settings::{ControllerType, Settings};
    use lichii_input::NpadState;
    use lichii_kernel::ipc::Handle;
    use lichii_kernel::object::KObject;
    use lichii_kernel::testing::IpcClient;

    use crate::ServiceContext;
    use crate::testing::{Reply, Request, context};

    /// A test title with `register`'s services and the host's player 1 on `controller`.
    pub fn booted(
        program_id: u64,
        controller: Option<ControllerType>,
        register: fn(&Arc<ServiceContext>) -> lichii_common::HleResult<()>,
    ) -> (Arc<ServiceContext>, IpcClient) {
        let (context, client) = context(program_id, Settings::default());
        register(&context).unwrap();
        if let Some(controller) = controller {
            connect(&context, controller);
        }
        (context, client)
    }

    pub fn connect(context: &ServiceContext, controller: ControllerType) {
        let state = NpadState {
            connected: true,
            controller,
            ..NpadState::default()
        };
        context.npad.publish(0, &state);
    }

    pub fn disconnect(context: &ServiceContext) {
        context.npad.publish(0, &NpadState::default());
    }

    /// Opens `service`'s `IUser` with command 0.
    pub fn user(client: &IpcClient, service: &str) -> Handle {
        let manager = client.connect(service).unwrap();
        client.send(manager, &Request::new(0)).unwrap().move_handles[0]
    }

    /// `ListDevices` into room for four handles: the result and the handles.
    pub fn list_devices(client: &IpcClient, user: Handle, command: u32) -> (Reply, Vec<u64>) {
        let out = client.alloc(&[0xff; 32]);
        let mut reply = client
            .send(user, &Request::new(command).buffer_b(out, 32))
            .unwrap();
        if reply.result != lichii_common::ResultCode::SUCCESS {
            return (reply, Vec::new());
        }
        let count = reply.pop::<i32>() as usize;
        let handles = client
            .read(out, count * 8)
            .chunks(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        (reply, handles)
    }

    /// Whether the reply carries one readable event that is not signalled.
    pub fn is_quiet_event(client: &IpcClient, reply: &Reply) -> bool {
        reply.copy_handles.len() == 1
            && matches!(
                client.object(reply.copy_handles[0]),
                Some(KObject::ReadableEvent(event)) if !event.is_signaled()
            )
    }
}
