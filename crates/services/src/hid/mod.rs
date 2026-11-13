//! `hid`: controllers, six-axis sensors and rumble. `CreateAppletResource` makes the HID
//! shared memory and starts the updater that fills it from `NpadHub`; the commands configure
//! which npads and styles the title accepts and forward rumble to the hub.

mod npad;
mod shared_memory;

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_common::{HleResult, ResultCode};
use lichii_input::VibrationValue;
use lichii_kernel::Event;
use lichii_kernel::Pool;
use lichii_kernel::ipc::{BufferDescriptor, HleRequestContext};
use lichii_kernel::memory_objects::SharedMemory;
use lichii_kernel::object::KObject;
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};
pub(crate) use npad::Hid;
use npad::{
    DEFAULT_FUSION_PARAMETERS, DEFAULT_VIBRATION, HANDHELD, Halves, HidState, OTHER,
    STANDARD_DRIFT_MODE, npad_index,
};
use shared_memory::NPAD_COUNT;

const fn hid_result(description: u32) -> ResultCode {
    ResultCode::new(202, description)
}

const NPAD_DEVICE_INDEX_OUT_OF_RANGE: ResultCode = hid_result(107);
const VIBRATION_INVALID_STYLE_INDEX: ResultCode = hid_result(122);
const VIBRATION_INVALID_NPAD_ID: ResultCode = hid_result(123);
const VIBRATION_DEVICE_INDEX_OUT_OF_RANGE: ResultCode = hid_result(124);
const VIBRATION_ARRAY_SIZE_MISMATCH: ResultCode = hid_result(131);
const INVALID_NPAD_ID: ResultCode = hid_result(709);
const INVALID_ARRAY_SIZE: ResultCode = hid_result(715);

/// Most vibration values one `SendVibrationValues` call carries; titles send one per motor.
const MAX_VIBRATIONS: usize = 32;
/// `NpadJoyDeviceType::Right`.
const JOY_DEVICE_RIGHT: u64 = 1;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let hid = Arc::new(Hid {
        hub: context.npad.clone(),
        state: Mutex::new(HidState::default()),
        nfc_readers: context.nfc_readers.clone(),
    });
    let _ = context.hid.set(hid.clone());
    context.register("hid", "hid", move |context| HidServer {
        context: context.clone(),
        hid: hid.clone(),
    })
}

/// `nn::hid::SixAxisSensorHandle` and `nn::hid::VibrationDeviceHandle`: style index, npad id,
/// device index (left, right, none).
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
struct DeviceHandle {
    style_index: u8,
    npad_id: u8,
    device_index: u8,
    reserved: u8,
}

impl DeviceHandle {
    fn sixaxis_npad(self) -> HleResult<usize> {
        let index = npad_index(u32::from(self.npad_id)).ok_or(INVALID_NPAD_ID)?;
        if self.device_index > 2 {
            return Err(NPAD_DEVICE_INDEX_OUT_OF_RANGE);
        }
        Ok(index)
    }

    fn vibration_npad(self) -> HleResult<usize> {
        if !matches!(self.style_index, 3..=8 | 13 | 32 | 33) {
            return Err(VIBRATION_INVALID_STYLE_INDEX);
        }
        let index = npad_index(u32::from(self.npad_id)).ok_or(VIBRATION_INVALID_NPAD_ID)?;
        if self.device_index > 2 {
            return Err(VIBRATION_DEVICE_INDEX_OUT_OF_RANGE);
        }
        Ok(index)
    }

    /// `nn::hid::VibrationDeviceInfo`: device type and position.
    fn vibration_info(self) -> [u32; 2] {
        match (self.style_index, self.device_index) {
            (3..=7, 0) => [1, 1],
            (3..=7, 1) => [1, 2],
            (3..=7, _) => [1, 0],
            (8, _) => [2, 0],
            (13, _) => [3, 0],
            _ => [0, 0],
        }
    }

    /// The motor slot (left or right) a vibration for this handle goes to.
    fn motor(self) -> usize {
        usize::from(self.device_index == 1)
    }
}

/// The one rumble the hub carries for a host controller's motors: each band from the motor that
/// drives it harder.
fn strongest(left: VibrationValue, right: VibrationValue) -> VibrationValue {
    let low = if right.low_amplitude > left.low_amplitude {
        right
    } else {
        left
    };
    let high = if right.high_amplitude > left.high_amplitude {
        right
    } else {
        left
    };
    VibrationValue {
        low_amplitude: low.low_amplitude,
        low_frequency: low.low_frequency,
        high_amplitude: high.high_amplitude,
        high_frequency: high.high_frequency,
    }
}

/// `nn::hid::VibrationValue` as it travels in requests.
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
#[repr(C)]
struct RawVibration([f32; 4]);

impl From<RawVibration> for VibrationValue {
    fn from(
        RawVibration([low_amplitude, low_frequency, high_amplitude, high_frequency]): RawVibration,
    ) -> Self {
        Self {
            low_amplitude,
            low_frequency,
            high_amplitude,
            high_frequency,
        }
    }
}

impl From<VibrationValue> for RawVibration {
    fn from(value: VibrationValue) -> Self {
        Self([
            value.low_amplitude,
            value.low_frequency,
            value.high_amplitude,
            value.high_frequency,
        ])
    }
}

struct HidServer {
    context: Arc<ServiceContext>,
    hid: Arc<Hid>,
}

impl HidServer {
    fn state(&self) -> parking_lot::MutexGuard<'_, HidState> {
        self.hid.state.lock()
    }

    /// The HID shared memory, made (and the updater started) on first use.
    fn shared_memory(&self) -> HleResult<Arc<SharedMemory>> {
        let mut state = self.state();
        if let Some(memory) = &state.memory {
            return Ok(memory.clone());
        }
        let memory = Arc::new(SharedMemory::new(
            self.context.kernel.memory_manager().clone(),
            None,
            Pool::System,
            shared_memory::SIZE,
            MemoryPermission::NONE,
            MemoryPermission::READ,
        )?);
        shared_memory::initialize(&memory, state.hold_type);
        state.memory = Some(memory.clone());
        drop(state);
        npad::spawn_updater(&self.hid);
        Ok(memory)
    }

    /// Sends one vibration to the hub (the frontend rumbles the player's controller).
    fn send_vibration(&self, handle: DeviceHandle, value: VibrationValue) -> HleResult<()> {
        let index = handle.vibration_npad()?;
        let mut state = self.state();
        if !state.vibration_permitted {
            return Ok(());
        }
        state.npads[index].vibration[handle.motor()] = value;
        let Some(player) = state.host_player(index) else {
            return Ok(());
        };
        let rumble = (0..NPAD_COUNT)
            .filter(|&entry| {
                entry == index
                    || (state.npads[entry].style.is_some()
                        && state.host_player(entry) == Some(player))
            })
            .flat_map(|entry| state.npads[entry].vibration)
            .reduce(strongest)
            .unwrap_or(value);
        self.hid.hub.request_vibration(player, rumble);
        Ok(())
    }

    /// Reads up to `out.len()` values from input buffer `index`; returns how many it read.
    fn read_array<T: Pod>(
        ctx: &HleRequestContext,
        index: usize,
        out: &mut [T],
    ) -> HleResult<usize> {
        let Some(BufferDescriptor { address, size }) = ctx.readable_buffer(index) else {
            return Ok(0);
        };
        let count = (size as usize / size_of::<T>()).min(out.len());
        ctx.memory()
            .read_bytes(address, bytemuck::cast_slice_mut(&mut out[..count]))
            .map_err(|_| lichii_kernel::result::INVALID_CURRENT_MEMORY)?;
        Ok(count)
    }

    /// The vibration handles and values of 206 and 214 (X buffers 0 and 1).
    fn vibration_arrays(
        ctx: &HleRequestContext,
        handles: &mut [DeviceHandle; MAX_VIBRATIONS],
        values: &mut [RawVibration; MAX_VIBRATIONS],
    ) -> HleResult<(usize, usize)> {
        Ok((
            Self::read_array(ctx, 0, handles)?,
            Self::read_array(ctx, 1, values)?,
        ))
    }

    fn npad_command(&self, ctx: &mut HleRequestContext) -> Option<HleResult<()>> {
        Some(match ctx.command() {
            100 => {
                let styles: u32 = ctx.pop();
                let mut state = self.state();
                state.supported_styles = Some(styles);
                state.update_npads(&self.hid.hub);
                Ok(())
            }
            101 => {
                let styles = self.state().supported_styles.unwrap_or(0);
                ctx.push(styles);
                Ok(())
            }
            102 => {
                let mut ids = [0u32; NPAD_COUNT + 1];
                Self::read_array(ctx, 0, &mut ids).and_then(|count| {
                    if count > NPAD_COUNT {
                        return Err(INVALID_ARRAY_SIZE);
                    }
                    let mut mask = 0u16;
                    for &id in &ids[..count] {
                        mask |= 1 << npad_index(id).ok_or(INVALID_NPAD_ID)?;
                    }
                    let mut state = self.state();
                    state.supported_ids = mask;
                    state.update_npads(&self.hid.hub);
                    Ok(())
                })
            }
            103 | 109 => {
                let mut state = self.state();
                state.revision = if ctx.command() == 109 { ctx.pop() } else { 0 };
                state.npad_active = true;
                Ok(())
            }
            104 => Ok(()),
            106 => {
                let npad_id: u32 = ctx.pop();
                npad_index(npad_id).ok_or(INVALID_NPAD_ID).map(|index| {
                    let event = self.state().npads[index]
                        .style_event
                        .get_or_insert_with(|| Arc::new(Event::new()))
                        .clone();
                    event.signal();
                    ctx.push_copy_object(KObject::ReadableEvent(event));
                })
            }
            107 => {
                let npad_id: u32 = ctx.pop();
                npad_index(npad_id)
                    .ok_or(INVALID_NPAD_ID)
                    .map(|index| self.state().disconnect_npad(index))
            }
            108 => {
                let npad_id: u32 = ctx.pop();
                let pattern: u64 = match npad_id {
                    0 => 0b0001,
                    1 => 0b0011,
                    2 => 0b0111,
                    3 => 0b1111,
                    4 => 0b1001,
                    5 => 0b0101,
                    6 => 0b1101,
                    7 => 0b0110,
                    _ => 0,
                };
                ctx.push(pattern);
                Ok(())
            }
            120 => {
                let _aruid: u64 = ctx.pop();
                let hold_type: u64 = ctx.pop();
                self.state().set_hold_type(hold_type);
                Ok(())
            }
            121 => {
                let hold_type = self.state().hold_type;
                ctx.push(hold_type);
                Ok(())
            }
            122..=124 | 133 => {
                let command = ctx.command();
                let npad_id: u32 = ctx.pop();
                let _aruid: u64 = ctx.pop();
                let kept = match command {
                    123 | 133 if ctx.pop::<u64>() == JOY_DEVICE_RIGHT => Halves::RIGHT,
                    _ => Halves::LEFT,
                };
                npad_index(npad_id).ok_or(INVALID_NPAD_ID).map(|index| {
                    let moved = self.state().set_assignment_mode(
                        &self.hid.hub,
                        index,
                        command != 124,
                        kept,
                    );
                    if command == 133 {
                        ctx.push(u8::from(moved.is_some()));
                        ctx.push(moved.map_or(npad_id, |index| index as u32));
                    }
                })
            }
            125 => {
                let first: u32 = ctx.pop();
                let second: u32 = ctx.pop();
                match (npad_index(first), npad_index(second)) {
                    (Some(first), Some(second)) => {
                        self.state().merge_joy_cons(&self.hid.hub, first, second);
                        Ok(())
                    }
                    _ => Err(INVALID_NPAD_ID),
                }
            }
            126 | 127 | 130 | 134..=136 => Ok(()),
            128 => {
                let _aruid: u64 = ctx.pop();
                let mode: u64 = ctx.pop();
                self.state().handheld_activation_mode = mode;
                Ok(())
            }
            129 => {
                let mode = self.state().handheld_activation_mode;
                ctx.push(mode);
                Ok(())
            }
            131 => {
                let npad_id: u32 = ctx.pop();
                npad_index(npad_id).ok_or(INVALID_NPAD_ID).map(|index| {
                    let enabled = self.state().npads[index].home_protection;
                    ctx.push(u8::from(enabled));
                })
            }
            132 => {
                let enabled = ctx.pop::<u8>() != 0;
                let npad_id: u32 = ctx.pop();
                npad_index(npad_id)
                    .ok_or(INVALID_NPAD_ID)
                    .map(|index| self.state().npads[index].home_protection = enabled)
            }
            _ => return None,
        })
    }

    fn sixaxis_command(&self, ctx: &mut HleRequestContext) -> Option<HleResult<()>> {
        let command = ctx.command();
        match command {
            60 | 61 | 63 | 64 | 89 | 300..=307 | 310 => return Some(Ok(())),
            66..=72 | 79..=85 => {}
            _ => return None,
        }
        let flag = matches!(command, 69 | 84) && ctx.pop::<u8>() != 0;
        let handle: DeviceHandle = ctx.pop();
        let index = match handle.sixaxis_npad() {
            Ok(index) => index,
            Err(code) => return Some(Err(code)),
        };
        let player = match index {
            HANDHELD => Some(0),
            OTHER => None,
            player => Some(player),
        };
        let mut state = self.state();
        let npad = &mut state.npads[index];
        match command {
            66 | 67 => npad.sixaxis_enabled = command == 66,
            68 => ctx.push(u8::from(npad.fusion_enabled)),
            69 => npad.fusion_enabled = flag,
            70 => npad.fusion_parameters = ctx.pop(),
            71 => ctx.push(npad.fusion_parameters),
            72 => {
                npad.fusion_parameters = DEFAULT_FUSION_PARAMETERS;
                npad.fusion_enabled = true;
            }
            79 => npad.drift_mode = ctx.pop(),
            80 => ctx.push(npad.drift_mode),
            81 => npad.drift_mode = STANDARD_DRIFT_MODE,
            82 => {
                let at_rest = player
                    .and_then(|player| self.hid.hub.read(player).motion)
                    .is_none_or(|motion| motion.gyro.iter().all(|rate| rate.abs() < 0.01));
                ctx.push(u8::from(at_rest));
            }
            83 => ctx.push(0u8),
            84 => npad.unaltered_passthrough = flag,
            _ => ctx.push(u8::from(npad.unaltered_passthrough)),
        }
        Some(Ok(()))
    }

    fn vibration_command(&self, ctx: &mut HleRequestContext) -> Option<HleResult<()>> {
        Some(match ctx.command() {
            200 => {
                let handle: DeviceHandle = ctx.pop();
                handle
                    .vibration_npad()
                    .map(|_| ctx.push(handle.vibration_info()))
            }
            201 => {
                let handle: DeviceHandle = ctx.pop();
                let value: RawVibration = ctx.pop();
                self.send_vibration(handle, value.into())
            }
            202 => {
                let handle: DeviceHandle = ctx.pop();
                handle.vibration_npad().map(|index| {
                    let value = self.state().npads[index].vibration[handle.motor()];
                    ctx.push(RawVibration::from(value));
                })
            }
            203 => {
                ctx.push_interface(self.context.handler(ActiveVibrationDeviceList));
                Ok(())
            }
            204 => {
                let permitted = ctx.pop::<u8>() != 0;
                self.state().vibration_permitted = permitted;
                Ok(())
            }
            205 => {
                let permitted = self.state().vibration_permitted;
                ctx.push(u8::from(permitted));
                Ok(())
            }
            206 => {
                let mut handles = [DeviceHandle::default(); MAX_VIBRATIONS];
                let mut values = [RawVibration::default(); MAX_VIBRATIONS];
                Self::vibration_arrays(ctx, &mut handles, &mut values).and_then(|(h, v)| {
                    if h != v {
                        return Err(VIBRATION_ARRAY_SIZE_MISMATCH);
                    }
                    for (&handle, &value) in handles.iter().zip(&values).take(h) {
                        self.send_vibration(handle, value.into())?;
                    }
                    Ok(())
                })
            }
            207 => {
                let handle: DeviceHandle = ctx.pop();
                let _aruid: u64 = ctx.pop();
                let command: u64 = ctx.pop();
                handle.vibration_npad().and_then(|index| {
                    self.state().npads[index].gc_erm_command = command;
                    let amplitude = if command == 1 { 1.0 } else { 0.0 };
                    let value = VibrationValue {
                        low_amplitude: amplitude,
                        high_amplitude: amplitude,
                        ..DEFAULT_VIBRATION
                    };
                    self.send_vibration(handle, value)
                })
            }
            208 => {
                let handle: DeviceHandle = ctx.pop();
                handle.vibration_npad().map(|index| {
                    let command = self.state().npads[index].gc_erm_command;
                    ctx.push(command);
                })
            }
            209 | 210 | 212 => Ok(()),
            211 | 220 => {
                let handle: DeviceHandle = ctx.pop();
                match handle.vibration_npad() {
                    Ok(index) => {
                        let mounted = self.state().npads[index].style.is_some();
                        ctx.push(u8::from(mounted));
                        Ok(())
                    }
                    Err(_) if ctx.command() == 220 => {
                        ctx.push(0u8);
                        Ok(())
                    }
                    Err(code) => Err(code),
                }
            }
            213 => {
                let raw: [u8; 0x18] = ctx.pop();
                let value: RawVibration = bytemuck::pod_read_unaligned(&raw[8..0x18]);
                let finite = value.0.iter().all(|v| v.is_finite());
                let handle = [4usize, 0]
                    .into_iter()
                    .map(|at| bytemuck::pod_read_unaligned::<DeviceHandle>(&raw[at..at + 4]))
                    .find(|handle| handle.vibration_npad().is_ok());
                if let (true, Some(handle)) = (finite, handle) {
                    let _ = self.send_vibration(handle, value.into());
                }
                Ok(())
            }
            214 => {
                let mut handles = [DeviceHandle::default(); MAX_VIBRATIONS];
                let mut values = [RawVibration::default(); MAX_VIBRATIONS];
                if let Ok((h, v)) = Self::vibration_arrays(ctx, &mut handles, &mut values) {
                    for (&handle, &value) in handles.iter().zip(&values).take(h.min(v)) {
                        let _ = self.send_vibration(handle, value.into());
                    }
                }
                Ok(())
            }
            _ => return None,
        })
    }
}

impl Service for HidServer {
    fn name(&self) -> &str {
        "hid"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            0 => self.shared_memory().map(|memory| {
                ctx.push_interface(self.context.handler(AppletResource { memory }));
            }),
            1 | 11 | 21 | 31 => {
                let mut state = self.state();
                match ctx.command() {
                    1 => state.debug_pad_active = true,
                    11 => state.touch_screen_active = true,
                    21 => state.mouse_active = true,
                    _ => state.keyboard_active = true,
                }
                Ok(())
            }
            26 | 32 | 91 | 92 => Ok(()),
            400 => {
                ctx.push(0u8);
                Ok(())
            }
            _ => self
                .npad_command(ctx)
                .or_else(|| self.sixaxis_command(ctx))
                .or_else(|| self.vibration_command(ctx))?,
        };
        Some(result.err().unwrap_or(ResultCode::SUCCESS))
    }
}

/// `IAppletResource`: hands out the HID shared memory.
struct AppletResource {
    memory: Arc<SharedMemory>,
}

impl Service for AppletResource {
    fn name(&self) -> &str {
        "IAppletResource"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => {
                ctx.push_copy_object(KObject::SharedMemory(self.memory.clone()));
                Some(ResultCode::SUCCESS)
            }
            _ => None,
        }
    }
}

/// `IActiveVibrationDeviceList`: activating a device checks its handle.
struct ActiveVibrationDeviceList;

impl Service for ActiveVibrationDeviceList {
    fn name(&self) -> &str {
        "IActiveVibrationDeviceList"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => {
                let handle: DeviceHandle = ctx.pop();
                Some(handle.vibration_npad().err().unwrap_or(ResultCode::SUCCESS))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
