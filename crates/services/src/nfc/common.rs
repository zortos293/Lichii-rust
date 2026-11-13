//! The NFC device manager behind every NFC `IUser` (`nfc:user`, `nfc:mf:u` and `nfp:user`), as
//! the reference's `nfc/common` shares it: the npads with a reader, their device state and the
//! events titles wait on.
//!
//! There is no amiibo or Mifare backend, so no tag is ever found: detection starts and stops,
//! every command that needs a tag answers `WrongDeviceState`, and the activate and deactivate
//! events never fire. Player 1 and handheld have the reference's virtual reader when their
//! controller has one, and the availability event fires when that changes; the other npads are
//! devices that stay unavailable. `set:sys`'s NFC enable flag gates every device command.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use lichii_common::settings::ControllerType;
use lichii_common::{HleResult, ResultCode};
use lichii_input::{NpadHub, NpadState};
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
/// What [`Readers`] holds while no npad has a reader.
const NO_READER: u32 = u32::MAX;
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
    NfcDisabled = 80,
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
        let user = User::new(self.kind, self.context.clone());
        ctx.push_interface(self.context.handler(user));
        Some(ResultCode::SUCCESS)
    }
}

/// An `IUser` and its own device manager, as each reference interface has one.
struct User {
    kind: UserKind,
    context: Arc<ServiceContext>,
    availability_change: Arc<Event>,
    activate: [Arc<Event>; DEVICE_COUNT],
    deactivate: [Arc<Event>; DEVICE_COUNT],
    state: Mutex<State>,
}

struct State {
    initialized: bool,
    /// The [`Readers`] change count `devices` follow.
    generation: u64,
    devices: [DeviceState; DEVICE_COUNT],
}

impl State {
    const FINALIZED: Self = Self {
        initialized: false,
        generation: 0,
        devices: [DeviceState::Unavailable; DEVICE_COUNT],
    };
}

/// The npad id whose reader the host's player 1 controller provides (`EmulatedController::HasNfc`):
/// handheld, or player 1 on a style with a right Joy-Con or Pro Controller reader.
fn reader_of(host: &NpadState) -> u32 {
    use ControllerType::*;
    if !host.connected {
        return NO_READER;
    }
    match host.controller {
        ProController | Switch2ProController | DualJoycons | Switch2DualJoycons | RightJoycon
        | Switch2RightJoycon => PLAYER_1,
        Handheld | Switch2Handheld => HANDHELD,
        LeftJoycon | Switch2LeftJoycon | GameCube => NO_READER,
    }
}

/// Which npad has a reader, shared by every `IUser`. hid's updater and every NFC command poll
/// it; a change signals each interface's availability event, as the reference's npad callbacks
/// do on connection and disconnection.
pub(crate) struct Readers {
    /// The reader's npad id, or [`NO_READER`].
    reader: AtomicU32,
    /// Counts changes, so an interface can tell a reconnection from no change.
    generation: AtomicU64,
    availability_events: Mutex<Vec<Weak<Event>>>,
}

impl Readers {
    pub(crate) fn new(hub: &NpadHub) -> Self {
        Self {
            reader: AtomicU32::new(reader_of(&hub.read(0))),
            generation: AtomicU64::new(0),
            availability_events: Mutex::new(Vec::new()),
        }
    }

    /// Samples the host's player 1; when the reader moved, signals every availability event.
    pub(crate) fn poll(&self, hub: &NpadHub) {
        let reader = reader_of(&hub.read(0));
        let old = self.reader.load(Ordering::Acquire);
        if old == reader
            || self
                .reader
                .compare_exchange(old, reader, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
        {
            return;
        }
        self.generation.fetch_add(1, Ordering::Release);
        self.availability_events
            .lock()
            .retain(|event| event.upgrade().inspect(|event| event.signal()).is_some());
    }

    /// The change count and the reader's npad id.
    fn current(&self) -> (u64, u32) {
        let generation = self.generation.load(Ordering::Acquire);
        (generation, self.reader.load(Ordering::Acquire))
    }

    fn watch(&self, event: &Arc<Event>) {
        self.availability_events.lock().push(Arc::downgrade(event));
    }
}

fn device_index(handle: u64) -> Option<usize> {
    NPAD_IDS.iter().position(|&id| u64::from(id) == handle)
}

impl User {
    fn new(kind: UserKind, context: Arc<ServiceContext>) -> Self {
        let event = |_| Arc::new(Event::new());
        let availability_change = Arc::new(Event::new());
        context.nfc_readers.poll(&context.npad);
        context.nfc_readers.watch(&availability_change);
        Self {
            kind,
            context,
            availability_change,
            activate: std::array::from_fn(event),
            deactivate: std::array::from_fn(event),
            state: Mutex::new(State::FINALIZED),
        }
    }

    fn error(&self, error: Error) -> ResultCode {
        self.kind.backend.result(error)
    }

    fn nfc_enabled(&self) -> bool {
        self.context
            .nfc_enabled
            .get()
            .is_none_or(|enabled| enabled())
    }

    /// Starts every device over from the current reader: the one with it is initialized, the
    /// rest unavailable, as the reference's `Initialize` and npad callbacks leave them.
    fn reset(&self, state: &mut State) {
        let (generation, reader) = self.context.nfc_readers.current();
        state.generation = generation;
        state.devices = NPAD_IDS.map(|npad_id| {
            if npad_id == reader {
                DeviceState::Initialized
            } else {
                DeviceState::Unavailable
            }
        });
    }

    /// Follows controller changes since the last command.
    fn sync(&self, state: &mut State) {
        let readers = &self.context.nfc_readers;
        readers.poll(&self.context.npad);
        if state.initialized && state.generation != readers.current().0 {
            self.reset(state);
        }
    }

    /// The device `handle` names, once NFC is enabled and the manager initialized
    /// (`GetDeviceHandle`).
    fn device(&self, state: &State, handle: u64) -> HleResult<usize> {
        if !self.nfc_enabled() {
            return Err(self.error(Error::NfcDisabled));
        }
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
                self.reset(&mut state);
            }
            Command::Finalize => *state = State::FINALIZED,
            Command::GetState => ctx.push(u32::from(state.initialized)),
            Command::IsNfcEnabled => ctx.push(u8::from(self.nfc_enabled())),
            Command::ListDevices => {
                let capacity = ctx.writable_buffer(0).map_or(0, |buffer| buffer.size / 8);
                if capacity == 0 {
                    return Err(self.error(Error::InvalidArgument));
                }
                if !self.nfc_enabled() {
                    return Err(self.error(Error::NfcDisabled));
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
        booted_with(program_id, Settings::default(), controller, register)
    }

    pub fn booted_with(
        program_id: u64,
        settings: Settings,
        controller: Option<ControllerType>,
        register: fn(&Arc<ServiceContext>) -> lichii_common::HleResult<()>,
    ) -> (Arc<ServiceContext>, IpcClient) {
        let (context, client) = context(program_id, settings);
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
