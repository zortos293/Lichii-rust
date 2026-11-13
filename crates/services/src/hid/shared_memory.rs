//! The HID shared memory (`nn::hid::detail::SharedMemoryFormat`, 0x40000 bytes): where each
//! section lives and the ring-buffer lifos titles read input from.

use std::marker::PhantomData;
use std::sync::atomic::{Ordering, fence};

use bytemuck::{Pod, Zeroable};
use lichii_kernel::memory_objects::SharedMemory;

pub(super) const SIZE: u64 = 0x4_0000;
/// Entries in every lifo; at most `ENTRY_COUNT - 1` count as valid.
const ENTRY_COUNT: u64 = 17;
const LIFO_HEADER_SIZE: u64 = 0x20;

/// Npad entries: players 1-8, Other and Handheld, in `NpadIdType` index order.
pub(super) const NPAD_COUNT: usize = 10;
const NPAD_BASE: u64 = 0x9a00;
const NPAD_ENTRY_SIZE: u64 = 0x5000;
const NPAD_CONDITION: u64 = 0x3_e200;

/// A state stored in a lifo. Every state carries its sampling number.
pub(super) trait LifoState: Pod {
    /// Offset of the sampling number in the state.
    const SAMPLING_OFFSET: u64;
    fn set_sampling_number(&mut self, sampling_number: i64);
}

/// One lifo: a header (timestamp, total entries, tail, valid entries) and 17
/// `{ sampling_number, state }` entries. The stored sampling number is the state's shifted left
/// by one, as the reference writes it.
pub(super) struct Lifo<S> {
    offset: u64,
    state: PhantomData<S>,
}

impl<S> Clone for Lifo<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for Lifo<S> {}

impl<S: LifoState> Lifo<S> {
    pub const SIZE: u64 = LIFO_HEADER_SIZE + ENTRY_COUNT * (8 + size_of::<S>() as u64);

    pub const fn at(offset: u64) -> Self {
        Self {
            offset,
            state: PhantomData,
        }
    }

    #[cfg(test)]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    fn entry(&self, index: u64) -> u64 {
        self.offset + LIFO_HEADER_SIZE + index * (8 + size_of::<S>() as u64)
    }

    /// Writes the header of an empty lifo.
    pub fn init(&self, memory: &SharedMemory) {
        put(memory, self.offset + 8, ENTRY_COUNT as i64);
    }

    /// Stores `state` as the newest entry, numbered one past the current newest. The entry is
    /// written before the header points at it.
    pub fn push(&self, memory: &SharedMemory, mut state: S) {
        let tail: i64 = get(memory, self.offset + 0x10);
        let count: i64 = get(memory, self.offset + 0x18);
        let tail = tail as u64 % ENTRY_COUNT;
        let previous: i64 = get(memory, self.entry(tail) + 8 + S::SAMPLING_OFFSET);
        let sampling_number = previous + 1;
        state.set_sampling_number(sampling_number);
        let next = (tail + 1) % ENTRY_COUNT;
        put(memory, self.entry(next), sampling_number << 1);
        put(memory, self.entry(next) + 8, state);
        fence(Ordering::Release);
        put(
            memory,
            self.offset + 0x18,
            (count + 1).min(ENTRY_COUNT as i64 - 1),
        );
        put(memory, self.offset + 0x10, next as i64);
    }
}

pub(super) fn put<T: Pod>(memory: &SharedMemory, offset: u64, value: T) {
    memory.write(offset, bytemuck::bytes_of(&value));
}

pub(super) fn get<T: Pod>(memory: &SharedMemory, offset: u64) -> T {
    let mut value = T::zeroed();
    memory.read(offset, bytemuck::bytes_of_mut(&mut value));
    value
}

/// `nn::hid::NpadFullKeyState` and its siblings (handheld, Joy-Con, Palma, system ext).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub(super) struct NpadGenericState {
    pub sampling_number: i64,
    pub buttons: u64,
    pub left_stick: [i32; 2],
    pub right_stick: [i32; 2],
    pub attributes: u32,
    pub reserved: u32,
}

impl LifoState for NpadGenericState {
    const SAMPLING_OFFSET: u64 = 0;
    fn set_sampling_number(&mut self, sampling_number: i64) {
        self.sampling_number = sampling_number;
    }
}

/// `nn::hid::SixAxisSensorState`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(super) struct SixAxisSensorState {
    pub delta_time: i64,
    pub sampling_number: i64,
    pub accel: [f32; 3],
    pub gyro: [f32; 3],
    pub rotation: [f32; 3],
    pub orientation: [[f32; 3]; 3],
    pub attributes: u32,
    pub reserved: u32,
}

impl LifoState for SixAxisSensorState {
    const SAMPLING_OFFSET: u64 = 8;
    fn set_sampling_number(&mut self, sampling_number: i64) {
        self.sampling_number = sampling_number;
    }
}

/// `nn::hid::server::NpadGcTriggerState`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub(super) struct GcTriggerState {
    pub sampling_number: i64,
    pub left: i32,
    pub right: i32,
}

impl LifoState for GcTriggerState {
    const SAMPLING_OFFSET: u64 = 0;
    fn set_sampling_number(&mut self, sampling_number: i64) {
        self.sampling_number = sampling_number;
    }
}

/// States that start with their sampling number and are otherwise left zero: the devices the
/// host has no input for yet (debug pad, touch screen, mouse, keyboard, system buttons).
macro_rules! blank_states {
    ($($name:ident: $size:literal),* $(,)?) => {$(
        #[repr(C)]
        #[derive(Clone, Copy, Pod, Zeroable)]
        pub(super) struct $name {
            sampling_number: i64,
            rest: [u8; $size - 8],
        }

        impl Default for $name {
            fn default() -> Self {
                Self::zeroed()
            }
        }

        impl LifoState for $name {
            const SAMPLING_OFFSET: u64 = 0;
            fn set_sampling_number(&mut self, sampling_number: i64) {
                self.sampling_number = sampling_number;
            }
        }
    )*};
}

blank_states! {
    DebugPadState: 0x20,
    TouchScreenState: 0x290,
    MouseState: 0x28,
    KeyboardState: 0x30,
    ButtonState: 0x10,
    GestureState: 0x60,
}

pub(super) const DEBUG_PAD: Lifo<DebugPadState> = Lifo::at(0x0);
pub(super) const TOUCH_SCREEN: Lifo<TouchScreenState> = Lifo::at(0x400);
pub(super) const MOUSE: Lifo<MouseState> = Lifo::at(0x3400);
pub(super) const KEYBOARD: Lifo<KeyboardState> = Lifo::at(0x3800);
pub(super) const HOME_BUTTON: Lifo<ButtonState> = Lifo::at(0x4c00);
pub(super) const SLEEP_BUTTON: Lifo<ButtonState> = Lifo::at(0x4e00);
pub(super) const CAPTURE_BUTTON: Lifo<ButtonState> = Lifo::at(0x5000);
pub(super) const GESTURE: Lifo<GestureState> = Lifo::at(0x3_ba00);
pub(super) const DEBUG_MOUSE: Lifo<MouseState> = Lifo::at(0x3_dc00);

const _: () = {
    assert!(Lifo::<DebugPadState>::SIZE == 0x2c8);
    assert!(Lifo::<TouchScreenState>::SIZE == 0x2c38);
    assert!(Lifo::<MouseState>::SIZE == 0x350);
    assert!(Lifo::<KeyboardState>::SIZE == 0x3d8);
    assert!(Lifo::<ButtonState>::SIZE + 0x48 == 0x200);
    assert!(Lifo::<GestureState>::SIZE == 0x708);
    assert!(NPAD_BASE + NPAD_COUNT as u64 * NPAD_ENTRY_SIZE == 0x3_ba00);
};

/// The generic lifos of an npad entry, in layout order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PadLifo {
    FullKey,
    Handheld,
    JoyDual,
    JoyLeft,
    JoyRight,
    Palma,
    SystemExt,
}

/// The six-axis lifos of an npad entry, in layout order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SixAxisLifo {
    FullKey,
    Handheld,
    DualLeft,
    DualRight,
    Left,
    Right,
}

impl SixAxisLifo {
    pub const ALL: [Self; 6] = [
        Self::FullKey,
        Self::Handheld,
        Self::DualLeft,
        Self::DualRight,
        Self::Left,
        Self::Right,
    ];
}

/// `nn::hid::detail::NpadInternalState` of one npad: field offsets within its entry.
#[derive(Clone, Copy, Debug)]
pub(super) struct NpadEntry(u64);

impl NpadEntry {
    pub const STYLE_SET: u64 = 0x0;
    pub const ASSIGNMENT_MODE: u64 = 0x4;
    pub const FULL_KEY_COLOR: u64 = 0x8;
    pub const JOY_COLOR: u64 = 0x14;
    const PAD_LIFOS: u64 = 0x28;
    const SIXAXIS_LIFOS: u64 = 0x1758;
    pub const DEVICE_TYPE: u64 = 0x4188;
    pub const SYSTEM_PROPERTIES: u64 = 0x4190;
    pub const BUTTON_PROPERTIES: u64 = 0x4198;
    pub const BATTERY_DUAL: u64 = 0x419c;
    pub const BATTERY_LEFT: u64 = 0x41a0;
    pub const BATTERY_RIGHT: u64 = 0x41a4;
    pub const FOOTER_TYPE: u64 = 0x41ac;
    const GC_TRIGGER_LIFO: u64 = 0x4228;
    const LARK_TYPES: u64 = 0x43e0;
    pub const SIXAXIS_PROPERTIES: u64 = 0x43f0;
    const INTERNAL_STATE_SIZE: u64 = 0x43f8;

    pub fn new(index: usize) -> Self {
        Self(NPAD_BASE + index as u64 * NPAD_ENTRY_SIZE)
    }

    /// Absolute offset of the field at `field`.
    pub fn at(self, field: u64) -> u64 {
        self.0 + field
    }

    pub fn pad(self, lifo: PadLifo) -> Lifo<NpadGenericState> {
        Lifo::at(self.0 + Self::PAD_LIFOS + lifo as u64 * Lifo::<NpadGenericState>::SIZE)
    }

    pub fn sixaxis(self, lifo: SixAxisLifo) -> Lifo<SixAxisSensorState> {
        Lifo::at(self.0 + Self::SIXAXIS_LIFOS + lifo as u64 * Lifo::<SixAxisSensorState>::SIZE)
    }

    pub fn gc_trigger(self) -> Lifo<GcTriggerState> {
        Lifo::at(self.0 + Self::GC_TRIGGER_LIFO)
    }
}

const _: () = {
    assert!(size_of::<NpadGenericState>() == 0x28);
    assert!(size_of::<SixAxisSensorState>() == 0x60);
    assert!(size_of::<GcTriggerState>() == 0x10);
    assert!(Lifo::<NpadGenericState>::SIZE == 0x350);
    assert!(NpadEntry::PAD_LIFOS + 7 * Lifo::<NpadGenericState>::SIZE == NpadEntry::SIXAXIS_LIFOS);
    assert!(
        NpadEntry::SIXAXIS_LIFOS + 6 * Lifo::<SixAxisSensorState>::SIZE == NpadEntry::DEVICE_TYPE
    );
    assert!(NpadEntry::GC_TRIGGER_LIFO + Lifo::<GcTriggerState>::SIZE == NpadEntry::LARK_TYPES);
    assert!(NpadEntry::SIXAXIS_PROPERTIES + 8 == NpadEntry::INTERNAL_STATE_SIZE);
    assert!(NpadEntry::INTERNAL_STATE_SIZE <= NPAD_ENTRY_SIZE);
};

/// `ColorAttribute::NoController`.
pub(super) const NO_CONTROLLER: u32 = 2;

/// Zeroes the memory and writes what an untouched shared memory holds: every lifo header, npad
/// entries with no controller, and the npad condition.
pub(super) fn initialize(memory: &SharedMemory, hold_type: u64) {
    let zero = [0u8; 0x1000];
    for offset in (0..SIZE).step_by(zero.len()) {
        memory.write(offset, &zero);
    }
    DEBUG_PAD.init(memory);
    TOUCH_SCREEN.init(memory);
    MOUSE.init(memory);
    KEYBOARD.init(memory);
    HOME_BUTTON.init(memory);
    SLEEP_BUTTON.init(memory);
    CAPTURE_BUTTON.init(memory);
    GESTURE.init(memory);
    DEBUG_MOUSE.init(memory);
    for index in 0..NPAD_COUNT {
        let entry = NpadEntry::new(index);
        for lifo in [
            PadLifo::FullKey,
            PadLifo::Handheld,
            PadLifo::JoyDual,
            PadLifo::JoyLeft,
            PadLifo::JoyRight,
            PadLifo::Palma,
            PadLifo::SystemExt,
        ] {
            entry.pad(lifo).init(memory);
        }
        for lifo in SixAxisLifo::ALL {
            entry.sixaxis(lifo).init(memory);
        }
        entry.gc_trigger().init(memory);
        put(memory, entry.at(NpadEntry::FULL_KEY_COLOR), NO_CONTROLLER);
        put(memory, entry.at(NpadEntry::JOY_COLOR), NO_CONTROLLER);
    }
    set_hold_type(memory, hold_type);
}

/// `nn::hid::NpadCondition`: initialized, the hold type, valid.
pub(super) fn set_hold_type(memory: &SharedMemory, hold_type: u64) {
    put(memory, NPAD_CONDITION, [0, 1, hold_type as u32, 1]);
}
