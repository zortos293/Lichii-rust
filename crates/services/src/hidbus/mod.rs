//! `hidbus`: accessories on the Joy-Con rails (Ring-Con, Starlink). No accessory is attached, so
//! every bus gets the reference's stub device: it is never connected, reports device id 0xFF and
//! answers commands with empty replies.

use std::sync::Arc;

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::Pool;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::{SharedMemory, TransferMemory};
use lichii_kernel::object::KObject;
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

const UNKNOWN: ResultCode = ResultCode(u32::MAX);

const SHARED_MEMORY_SIZE: u64 = 0x1000;
const MAX_HANDLES: usize = 0x13;
/// `HidbusStatusManagerEntry`.
const ENTRY_SIZE: usize = 0x80;
/// The stub device's `GetDeviceId`.
const STUB_DEVICE_ID: u32 = 0xff;
/// The `BusHandle` fields a lookup compares: pad id, internal index, player, bus type, valid.
const HANDLE_MASK: u64 = (1 << 57) - 1;
const HANDLE_VALID: u64 = 1 << 56;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let memory = Arc::new(SharedMemory::new(
        context.kernel.memory_manager().clone(),
        None,
        Pool::System,
        SHARED_MEMORY_SIZE,
        MemoryPermission::NONE,
        MemoryPermission::READ,
    )?);
    let bus = Arc::new(Hidbus {
        memory,
        state: Mutex::new(HidbusState {
            enabled: false,
            devices: std::array::from_fn(|_| Device::default()),
            entries: [Entry::default(); MAX_HANDLES],
        }),
    });
    context.register("hidbus", "hid", move |_| HidbusServer { bus: bus.clone() })
}

struct Hidbus {
    memory: Arc<SharedMemory>,
    state: Mutex<HidbusState>,
}

struct HidbusState {
    /// Set by the first `Initialize`; from then on device state reaches the shared memory.
    enabled: bool,
    devices: [Device; MAX_HANDLES],
    entries: [Entry; MAX_HANDLES],
}

/// One bus slot and the stub device `Initialize` attaches to it.
#[derive(Default)]
struct Device {
    handle: u64,
    initialized: bool,
    device_enabled: bool,
    polling_enabled: bool,
    polling_mode: u32,
    send_command_event: Arc<Event>,
}

/// `HidbusStatusManagerEntry`, the shared memory record of one bus.
#[derive(Clone, Copy, Default)]
struct Entry {
    is_enabled: bool,
    is_in_focus: bool,
    is_polling_mode: bool,
    polling_mode: u32,
}

impl Entry {
    fn bytes(&self) -> [u8; ENTRY_SIZE] {
        let mut raw = [0; ENTRY_SIZE];
        raw[8] = self.is_enabled.into();
        raw[9] = self.is_in_focus.into();
        raw[10] = self.is_polling_mode.into();
        raw[12..16].copy_from_slice(&self.polling_mode.to_le_bytes());
        raw
    }
}

impl HidbusState {
    fn index(&self, handle: u64) -> Result<usize, ResultCode> {
        self.devices
            .iter()
            .position(|device| device.handle & HANDLE_MASK == handle & HANDLE_MASK)
            .ok_or(UNKNOWN)
    }
}

impl Hidbus {
    fn write_entry(&self, state: &HidbusState, index: usize) {
        self.memory
            .write((index * ENTRY_SIZE) as u64, &state.entries[index].bytes());
    }

    /// The reference's 15 ms update: an initialized device's enable and polling state reach its
    /// entry once the bus is enabled.
    fn update(&self, state: &mut HidbusState, index: usize) {
        let device = &state.devices[index];
        if !state.enabled || !device.initialized {
            return;
        }
        let entry = &mut state.entries[index];
        entry.is_enabled = device.device_enabled;
        entry.is_polling_mode = device.polling_enabled;
        entry.polling_mode = device.polling_mode;
        self.write_entry(state, index);
    }

    /// `GetBusHandle`: the slot already made for this npad and bus, else a new one.
    fn bus_handle(&self, npad_id: u32, bus_type: u64) -> (bool, u64) {
        let mut state = self.state.lock();
        let bus_type = bus_type & 0xff;
        if let Some(device) = state.devices.iter().find(|device| {
            device.handle & HANDLE_VALID != 0
                && (device.handle >> 40) & 0xff == u64::from(npad_id)
                && (device.handle >> 48) & 0xff == bus_type
        }) {
            return (true, device.handle);
        }
        let Some(index) = state
            .devices
            .iter()
            .position(|device| device.handle & HANDLE_VALID == 0)
        else {
            tracing::error!("hidbus: no free bus handle");
            return (false, state.devices[0].handle);
        };
        let handle = HANDLE_VALID
            | (bus_type << 48)
            | (u64::from(npad_id as u8) << 40)
            | ((index as u64) << 32)
            | index as u64;
        state.devices[index].handle = handle;
        (true, handle)
    }

    fn initialize(&self, handle: u64) -> HleResult<()> {
        let mut state = self.state.lock();
        state.enabled = true;
        let index = state.index(handle)?;
        state.devices[index] = Device {
            handle: state.devices[index].handle,
            initialized: true,
            ..Device::default()
        };
        self.reset_entry(&mut state, index);
        Ok(())
    }

    fn finalize(&self, handle: u64) -> HleResult<()> {
        let mut state = self.state.lock();
        let index = state.index(handle)?;
        state.devices[index].initialized = false;
        self.reset_entry(&mut state, index);
        Ok(())
    }

    /// Marks bus `index` in focus, disconnected, disabled and not polling, and copies the whole
    /// status to the shared memory.
    fn reset_entry(&self, state: &mut HidbusState, index: usize) {
        state.entries[index] = Entry {
            is_in_focus: true,
            polling_mode: state.entries[index].polling_mode,
            ..Entry::default()
        };
        for index in 0..MAX_HANDLES {
            self.write_entry(state, index);
        }
    }

    fn with_device<T>(
        &self,
        handle: u64,
        change: impl FnOnce(&mut Device) -> T,
    ) -> Result<T, ResultCode> {
        let mut state = self.state.lock();
        let index = state.index(handle)?;
        let value = change(&mut state.devices[index]);
        self.update(&mut state, index);
        Ok(value)
    }
}

struct HidbusServer {
    bus: Arc<Hidbus>,
}

impl HidbusServer {
    fn command(&self, ctx: &mut HleRequestContext) -> Option<HleResult<()>> {
        let bus = &self.bus;
        Some(match ctx.command() {
            1 => {
                let npad_id: u32 = ctx.pop();
                let bus_type: u64 = ctx.pop();
                let (valid, handle) = bus.bus_handle(npad_id, bus_type);
                ctx.push(u8::from(valid));
                ctx.push(handle);
                Ok(())
            }
            2 => {
                let handle: u64 = ctx.pop();
                bus.with_device(handle, |_| ()).map(|()| ctx.push(0u8))
            }
            3 => bus.initialize(ctx.pop()),
            4 => bus.finalize(ctx.pop()),
            5 => {
                let enable = ctx.pop::<u8>() != 0;
                let handle: u64 = ctx.pop();
                bus.with_device(handle, |device| device.device_enabled = enable)
            }
            6 => {
                let handle: u64 = ctx.pop();
                bus.with_device(handle, |_| ())
                    .map(|()| ctx.push(STUB_DEVICE_ID))
            }
            7 => bus.with_device(ctx.pop(), |_| ()),
            8 => {
                let handle: u64 = ctx.pop();
                bus.with_device(handle, |_| ()).map(|()| ctx.push(0u64))
            }
            9 => {
                let handle: u64 = ctx.pop();
                bus.with_device(handle, |device| device.send_command_event.clone())
                    .map(|event| ctx.push_copy_object(KObject::ReadableEvent(event)))
            }
            10 => {
                ctx.push_copy_object(KObject::SharedMemory(bus.memory.clone()));
                Ok(())
            }
            11 => {
                let _size: u32 = ctx.pop();
                let mode: u32 = ctx.pop();
                let handle: u64 = ctx.pop();
                ctx.copy_handle::<TransferMemory>(0).and_then(|_| {
                    bus.with_device(handle, |device| {
                        device.polling_enabled = true;
                        device.polling_mode = mode;
                    })
                })
            }
            12 => bus.with_device(ctx.pop(), |device| device.polling_enabled = false),
            14 => Ok(()),
            _ => return None,
        })
    }
}

impl Service for HidbusServer {
    fn name(&self) -> &str {
        "hidbus"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(self.command(ctx)?.err().unwrap_or(ResultCode::SUCCESS))
    }
}

#[cfg(test)]
mod tests;
