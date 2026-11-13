//! Npad state and the periodic updater that copies `NpadHub` into the shared memory: npads every
//! 1 ms (the reference's overclocked rate), six-axis sensors every 5 ms, the debug pad, touch
//! screen and system buttons every 4 ms, mouse and keyboard every 8 ms.

use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use lichii_common::settings::{ControllerType, MAX_PLAYERS};
use lichii_input::{NpadButtons, NpadHub, NpadState, VibrationValue};
use lichii_kernel::Event;
use lichii_kernel::memory_objects::SharedMemory;

use crate::applets::ControllerSetup;

use super::shared_memory::{
    self as layout, CAPTURE_BUTTON, DEBUG_PAD, GcTriggerState, HOME_BUTTON, KEYBOARD, MOUSE,
    NO_CONTROLLER, NPAD_COUNT, NpadEntry, NpadGenericState, PadLifo, SLEEP_BUTTON, SixAxisLifo,
    SixAxisSensorState, TOUCH_SCREEN, put,
};

const NPAD_PERIOD: Duration = Duration::from_millis(1);
const MOTION_TICKS: u64 = 5;
const DEFAULT_TICKS: u64 = 4;
const MOUSE_KEYBOARD_TICKS: u64 = 8;
const MOTION_PERIOD_NS: i64 = 5_000_000;

/// Index of `NpadIdType::Other` and `NpadIdType::Handheld` among the npad entries.
pub(super) const OTHER: usize = 8;
pub(super) const HANDHELD: usize = 9;

/// `NpadJoyHoldType::Vertical`.
const HOLD_VERTICAL: u64 = 0;
/// `nn::hid::NpadBatteryLevel::Full`.
const BATTERY_FULL: u32 = 4;

/// The npad entry index of an `NpadIdType`.
pub(super) fn npad_index(npad_id: u32) -> Option<usize> {
    match npad_id {
        0..=7 => Some(npad_id as usize),
        0x10 => Some(OTHER),
        0x20 => Some(HANDHELD),
        _ => None,
    }
}

/// The npad styles hid reports for the host's controllers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Style {
    FullKey,
    Handheld,
    JoyDual,
    JoyLeft,
    JoyRight,
    Gc,
}

impl Style {
    fn of(controller: ControllerType) -> Self {
        match controller {
            ControllerType::ProController | ControllerType::Switch2ProController => Self::FullKey,
            ControllerType::DualJoycons | ControllerType::Switch2DualJoycons => Self::JoyDual,
            ControllerType::LeftJoycon | ControllerType::Switch2LeftJoycon => Self::JoyLeft,
            ControllerType::RightJoycon | ControllerType::Switch2RightJoycon => Self::JoyRight,
            ControllerType::Handheld | ControllerType::Switch2Handheld => Self::Handheld,
            ControllerType::GameCube => Self::Gc,
        }
    }

    /// The `NpadStyleSet` bit.
    pub fn tag(self) -> u32 {
        match self {
            Self::FullKey => 1 << 0,
            Self::Handheld => 1 << 1,
            Self::JoyDual => 1 << 2,
            Self::JoyLeft => 1 << 3,
            Self::JoyRight => 1 << 4,
            Self::Gc => 1 << 5,
        }
    }

    fn pad_lifo(self) -> PadLifo {
        match self {
            Self::FullKey | Self::Gc => PadLifo::FullKey,
            Self::Handheld => PadLifo::Handheld,
            Self::JoyDual => PadLifo::JoyDual,
            Self::JoyLeft => PadLifo::JoyLeft,
            Self::JoyRight => PadLifo::JoyRight,
        }
    }

    /// `NpadAttribute`: connected, wired, left/right connected and wired.
    fn attributes(self) -> u32 {
        match self {
            Self::FullKey | Self::Gc => 0b11,
            Self::Handheld => 0x3f,
            Self::JoyDual => 0x15,
            Self::JoyLeft => 0x5,
            Self::JoyRight => 0x11,
        }
    }
}

/// The Joy-Cons of a host controller an npad entry shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Halves {
    pub left: bool,
    pub right: bool,
}

impl Halves {
    pub const BOTH: Self = Self {
        left: true,
        right: true,
    };
    pub const LEFT: Self = Self {
        left: true,
        right: false,
    };
    pub const RIGHT: Self = Self {
        left: false,
        right: true,
    };

    /// The Joy-Cons a host controller has, `None` for one that is not made of Joy-Cons.
    fn of(controller: ControllerType) -> Option<Self> {
        match Style::of(controller) {
            Style::JoyDual => Some(Self::BOTH),
            Style::JoyLeft => Some(Self::LEFT),
            Style::JoyRight => Some(Self::RIGHT),
            _ => None,
        }
    }

    fn contains(self, other: Self) -> bool {
        (self.left || !other.left) && (self.right || !other.right)
    }
}

/// An npad entry the title reassigned (`SetNpadMode`, `MergeSingleJoyAsDualJoy`): it shows
/// `halves` of host `player`'s Joy-Cons instead of the player's own controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Route {
    pub player: usize,
    pub halves: Halves,
}

impl Route {
    /// The style of a reassigned entry: a whole pair is dual, one Joy-Con is single only in the
    /// single assignment mode.
    fn style(self, single: bool) -> Style {
        match (self.halves, single) {
            (Halves::LEFT, true) => Style::JoyLeft,
            (Halves::RIGHT, true) => Style::JoyRight,
            _ => Style::JoyDual,
        }
    }
}

/// Controller colors: `{ body, buttons }` as RGBA bytes.
const FULL_KEY_COLORS: [[u8; 4]; 2] = [[0x2d, 0x2d, 0x2d, 0xff], [0xe6, 0xe6, 0xe6, 0xff]];
const LEFT_COLORS: [[u8; 4]; 2] = [[0x0a, 0xb9, 0xe6, 0xff], [0x1e, 0x1e, 0x1e, 0xff]];
const RIGHT_COLORS: [[u8; 4]; 2] = [[0xff, 0x3c, 0x28, 0xff], [0x1e, 0x1e, 0x1e, 0xff]];

const RIGHT_BUTTONS: NpadButtons = NpadButtons::A
    .union(NpadButtons::B)
    .union(NpadButtons::X)
    .union(NpadButtons::Y)
    .union(NpadButtons::STICK_R)
    .union(NpadButtons::R)
    .union(NpadButtons::ZR)
    .union(NpadButtons::PLUS)
    .union(NpadButtons::STICK_R_LEFT)
    .union(NpadButtons::STICK_R_UP)
    .union(NpadButtons::STICK_R_RIGHT)
    .union(NpadButtons::STICK_R_DOWN);
const LEFT_BUTTONS: NpadButtons = NpadButtons::LEFT
    .union(NpadButtons::UP)
    .union(NpadButtons::RIGHT)
    .union(NpadButtons::DOWN)
    .union(NpadButtons::STICK_L)
    .union(NpadButtons::L)
    .union(NpadButtons::ZL)
    .union(NpadButtons::MINUS)
    .union(NpadButtons::STICK_L_LEFT)
    .union(NpadButtons::STICK_L_UP)
    .union(NpadButtons::STICK_L_RIGHT)
    .union(NpadButtons::STICK_L_DOWN);

/// The guest npad state of a host state for `style`: buttons the style does not have are
/// dropped, and a GameCube controller's L/R/Z are moved where the guest expects them.
fn pad_state(style: Style, halves: Halves, state: &NpadState) -> NpadGenericState {
    let raw = state.buttons;
    let mut buttons = NpadButtons::empty();
    let mut pad = NpadGenericState {
        attributes: style.attributes(),
        ..NpadGenericState::default()
    };
    if style != Style::JoyLeft && halves.right {
        buttons |= raw & RIGHT_BUTTONS;
        pad.right_stick = [state.right_stick.x, state.right_stick.y];
    }
    if style != Style::JoyRight && halves.left {
        buttons |= raw & LEFT_BUTTONS;
        pad.left_stick = [state.left_stick.x, state.left_stick.y];
    }
    if matches!(style, Style::JoyLeft | Style::JoyDual) && halves.left {
        buttons |= raw & (NpadButtons::LEFT_SL | NpadButtons::LEFT_SR);
    }
    if matches!(style, Style::JoyRight | Style::JoyDual) && halves.right {
        buttons |= raw & (NpadButtons::RIGHT_SL | NpadButtons::RIGHT_SR);
    }
    if !halves.left {
        pad.attributes &= !0b1100;
    }
    if !halves.right {
        pad.attributes &= !0b11_0000;
    }
    if style == Style::Gc {
        buttons.remove(NpadButtons::ZL | NpadButtons::ZR | NpadButtons::L | NpadButtons::R);
        buttons.set(NpadButtons::ZR, raw.contains(NpadButtons::R));
        buttons.set(NpadButtons::L, raw.contains(NpadButtons::ZL));
        buttons.set(NpadButtons::R, raw.contains(NpadButtons::ZR));
    }
    pad.buttons = buttons.bits();
    pad
}

/// Host-side state of one npad entry.
pub(super) struct Npad {
    /// The style the shared memory shows as connected.
    pub style: Option<Style>,
    /// The Joy-Cons the connected controller has.
    pub halves: Halves,
    /// `NpadJoyAssignmentMode::Single`.
    pub single: bool,
    /// Disconnected by the title (`DisconnectNpad`): stays off until the controller goes away.
    pub suppressed: bool,
    pub style_event: Option<Arc<Event>>,
    pub sixaxis_enabled: bool,
    pub fusion_enabled: bool,
    pub fusion_parameters: [f32; 2],
    pub drift_mode: u32,
    pub unaltered_passthrough: bool,
    pub home_protection: bool,
    /// The last vibration sent to the left and right devices.
    pub vibration: [VibrationValue; 2],
    pub gc_erm_command: u64,
    /// Integrated gyro: orientation quaternion (w, x, y, z) and rotation in revolutions.
    orientation: [f32; 4],
    rotation: [f32; 3],
}

impl Default for Npad {
    fn default() -> Self {
        Self {
            style: None,
            halves: Halves::BOTH,
            single: false,
            suppressed: false,
            style_event: None,
            sixaxis_enabled: false,
            fusion_enabled: true,
            fusion_parameters: DEFAULT_FUSION_PARAMETERS,
            drift_mode: STANDARD_DRIFT_MODE,
            unaltered_passthrough: false,
            home_protection: true,
            vibration: [DEFAULT_VIBRATION; 2],
            gc_erm_command: 0,
            orientation: [1.0, 0.0, 0.0, 0.0],
            rotation: [0.0; 3],
        }
    }
}

pub(super) const DEFAULT_VIBRATION: VibrationValue = VibrationValue {
    low_amplitude: 0.0,
    low_frequency: 160.0,
    high_amplitude: 0.0,
    high_frequency: 320.0,
};
pub(super) const DEFAULT_FUSION_PARAMETERS: [f32; 2] = [0.03, 0.4];
/// `GyroscopeZeroDriftMode::Standard`.
pub(super) const STANDARD_DRIFT_MODE: u32 = 1;

impl Npad {
    fn signal_style_change(&self) {
        if let Some(event) = &self.style_event {
            event.signal();
        }
    }
}

/// Everything the hid commands configure and the updater reads, behind one lock.
pub(super) struct HidState {
    pub memory: Option<Arc<SharedMemory>>,
    pub npad_active: bool,
    /// `None` until the title sets it; npads stay disconnected until then.
    pub supported_styles: Option<u32>,
    /// Bit `i` set when npad entry `i` is supported.
    pub supported_ids: u16,
    pub hold_type: u64,
    pub handheld_activation_mode: u64,
    pub revision: u32,
    pub vibration_permitted: bool,
    pub debug_pad_active: bool,
    pub touch_screen_active: bool,
    pub mouse_active: bool,
    pub keyboard_active: bool,
    pub npads: [Npad; NPAD_COUNT],
    pub routes: [Option<Route>; NPAD_COUNT],
}

impl Default for HidState {
    fn default() -> Self {
        Self {
            memory: None,
            npad_active: false,
            supported_styles: None,
            supported_ids: (1 << NPAD_COUNT) - 1,
            hold_type: HOLD_VERTICAL,
            handheld_activation_mode: 0,
            revision: 0,
            vibration_permitted: true,
            debug_pad_active: false,
            touch_screen_active: false,
            mouse_active: false,
            keyboard_active: false,
            npads: Default::default(),
            routes: [None; NPAD_COUNT],
        }
    }
}

impl HidState {
    fn supports(&self, index: usize, style: Style) -> bool {
        self.supported_ids & 1 << index != 0
            && self.supported_styles.unwrap_or(0) & style.tag() != 0
    }

    /// Where a controller of `style` on npad entry `index` connects, given what the title
    /// supports: its own style, else the closest one it accepts. A handheld the title does not
    /// accept becomes player 1.
    fn placement(&self, index: usize, style: Style) -> Option<(usize, Style)> {
        if index == HANDHELD {
            if self.supports(HANDHELD, Style::Handheld) {
                return Some((HANDHELD, Style::Handheld));
            }
            return [Style::JoyDual, Style::FullKey]
                .into_iter()
                .find(|&style| self.supports(0, style))
                .map(|style| (0, style));
        }
        [style, Style::FullKey, Style::JoyDual]
            .into_iter()
            .find(|&candidate| candidate != Style::Handheld && self.supports(index, candidate))
            .map(|style| (index, style))
    }

    /// One npad update: connects, disconnects and samples every npad from the hub.
    pub fn update_npads(&mut self, hub: &NpadHub) {
        let Some(memory) = self.memory.clone() else {
            return;
        };
        if !self.npad_active || self.supported_styles.is_none() {
            return;
        }
        let states: [NpadState; MAX_PLAYERS] = std::array::from_fn(|player| hub.read(player));
        for route in &mut self.routes {
            let valid = route.is_some_and(|route| {
                let state = &states[route.player];
                state.connected
                    && Halves::of(state.controller).is_some_and(|host| host.contains(route.halves))
            });
            if !valid {
                *route = None;
            }
        }
        let mut wanted: [Option<(Style, Halves, NpadState)>; NPAD_COUNT] = [None; NPAD_COUNT];
        for (index, route) in self.routes.iter().enumerate() {
            let Some(route) = route else { continue };
            let style = route.style(self.npads[index].single);
            if let Some((index, style)) = self.placement(index, style) {
                wanted[index].get_or_insert((style, route.halves, states[route.player]));
            }
        }
        for (player, state) in states.into_iter().enumerate() {
            let routed = self
                .routes
                .iter()
                .flatten()
                .any(|route| route.player == player);
            if !state.connected || routed {
                continue;
            }
            let style = Style::of(state.controller);
            let index = match style {
                Style::Handheld if player == 0 => HANDHELD,
                _ => player,
            };
            let style = if style == Style::Handheld && index != HANDHELD {
                Style::JoyDual
            } else {
                style
            };
            if let Some((index, style)) = self.placement(index, style) {
                wanted[index].get_or_insert((style, Halves::BOTH, state));
            }
        }
        for (index, wanted) in wanted.into_iter().enumerate() {
            let npad = &mut self.npads[index];
            if npad.suppressed {
                if wanted.is_some() {
                    continue;
                }
                npad.suppressed = false;
            }
            let shown = npad.style.map(|style| (style, npad.halves));
            let wanted_style = wanted.map(|(style, halves, _)| (style, halves));
            if shown.is_some() && shown != wanted_style {
                disconnect(&memory, index, npad);
            }
            let Some((style, halves, state)) = wanted else {
                continue;
            };
            if npad.style.is_none() {
                connect(&memory, index, npad, style, halves);
            }
            let entry = NpadEntry::new(index);
            let pad = pad_state(style, halves, &state);
            entry.pad(style.pad_lifo()).push(&memory, pad);
            entry.pad(PadLifo::SystemExt).push(&memory, pad);
            if style == Style::Gc {
                entry.gc_trigger().push(
                    &memory,
                    GcTriggerState {
                        left: state.left_trigger as i32,
                        right: state.right_trigger as i32,
                        ..GcTriggerState::default()
                    },
                );
            }
        }
    }

    /// `SetNpadMode`: sets the Joy-Con assignment mode of entry `index` and applies it. A dual
    /// pair set to single keeps the `kept` Joy-Con and moves the other one to the first free
    /// entry, which it returns; a single Joy-Con set to dual becomes a pair with one side.
    pub fn set_assignment_mode(
        &mut self,
        hub: &NpadHub,
        index: usize,
        single: bool,
        kept: Halves,
    ) -> Option<usize> {
        let player = match self.routes[index] {
            Some(route) => Some(route.player),
            None => (index < MAX_PLAYERS).then_some(index),
        };
        let npad = &self.npads[index];
        let splits = single
            && npad.style == Some(Style::JoyDual)
            && npad.halves == Halves::BOTH
            && player.is_some_and(|player| {
                Halves::of(hub.read(player).controller) == Some(Halves::BOTH)
            });
        let free = if splits {
            Some((0..MAX_PLAYERS).find(|&other| {
                other != index
                    && self.supported_ids & 1 << other != 0
                    && self.npads[other].style.is_none()
                    && self.routes[other].is_none()
            })?)
        } else {
            None
        };
        if let Some(memory) = &self.memory {
            put(
                memory,
                NpadEntry::new(index).at(NpadEntry::ASSIGNMENT_MODE),
                u32::from(single),
            );
        }
        self.npads[index].single = single;
        let player = player?;
        let npad = &self.npads[index];
        let halves = npad.halves;
        match (npad.style, single, free) {
            (Some(Style::JoyLeft | Style::JoyRight), false, _) => {
                let halves = if npad.style == Some(Style::JoyLeft) {
                    Halves::LEFT
                } else {
                    Halves::RIGHT
                };
                self.routes[index] = Some(Route { player, halves });
            }
            (Some(Style::JoyDual), true, _) if halves != Halves::BOTH => {
                self.routes[index] = Some(Route { player, halves });
            }
            (_, _, Some(free)) => {
                let other = Halves {
                    left: !kept.left,
                    right: !kept.right,
                };
                self.routes[index] = Some(Route {
                    player,
                    halves: kept,
                });
                self.routes[free] = Some(Route {
                    player,
                    halves: other,
                });
                self.npads[free].single = false;
            }
            _ => return None,
        }
        self.update_npads(hub);
        free
    }

    /// The host player whose controller npad entry `index` shows: its route's player, player 1
    /// for the handheld entry, and none for `Other`.
    pub fn host_player(&self, index: usize) -> Option<usize> {
        match (self.routes[index], index) {
            (Some(route), _) => Some(route.player),
            (None, HANDHELD) => Some(0),
            (None, OTHER) => None,
            (None, player) => Some(player),
        }
    }

    /// `MergeSingleJoyAsDualJoy`: joins the two Joy-Cons of one host pair, shown on entries
    /// `first` and `second`, back into a dual pair on `first`.
    pub fn merge_joy_cons(&mut self, hub: &NpadHub, first: usize, second: usize) {
        let (Some(a), Some(b)) = (self.routes[first], self.routes[second]) else {
            return;
        };
        let complementary = a.halves.left != b.halves.left && a.halves.right != b.halves.right;
        if first == second || a.player != b.player || !complementary {
            return;
        }
        self.routes[first] = Some(Route {
            player: a.player,
            halves: Halves::BOTH,
        });
        self.routes[second] = None;
        self.update_npads(hub);
    }

    /// Disconnects npad entry `index` now and keeps it off until its controller goes away.
    pub fn disconnect_npad(&mut self, index: usize) {
        let npad = &mut self.npads[index];
        if let (Some(memory), Some(_)) = (&self.memory, npad.style) {
            disconnect(memory, index, npad);
            npad.suppressed = true;
        }
    }

    /// One six-axis update for every connected npad.
    pub fn update_motion(&mut self, hub: &NpadHub) {
        let Some(memory) = self.memory.clone() else {
            return;
        };
        if !self.npad_active {
            return;
        }
        let players: [Option<usize>; NPAD_COUNT] =
            std::array::from_fn(|index| self.host_player(index));
        for ((index, npad), player) in self.npads.iter_mut().enumerate().zip(players) {
            let Some(style) = npad.style else { continue };
            let motion = player
                .filter(|_| npad.sixaxis_enabled)
                .and_then(|player| hub.read(player).motion);
            let active = match motion {
                Some(sample) => npad.integrate(sample.accel, sample.gyro),
                None => SixAxisSensorState {
                    delta_time: MOTION_PERIOD_NS,
                    accel: [0.0, 0.0, -1.0],
                    orientation: IDENTITY,
                    attributes: 1,
                    ..SixAxisSensorState::default()
                },
            };
            let entry = NpadEntry::new(index);
            for lifo in SixAxisLifo::ALL {
                let skipped = if index == HANDHELD {
                    SixAxisLifo::FullKey
                } else {
                    SixAxisLifo::Handheld
                };
                if lifo == skipped {
                    continue;
                }
                let used = match style {
                    Style::FullKey => lifo == SixAxisLifo::FullKey,
                    Style::Handheld => lifo == SixAxisLifo::Handheld,
                    Style::JoyDual => {
                        matches!(lifo, SixAxisLifo::DualLeft | SixAxisLifo::DualRight)
                    }
                    Style::JoyLeft => lifo == SixAxisLifo::Left,
                    Style::JoyRight => lifo == SixAxisLifo::Right,
                    Style::Gc => false,
                };
                let state = if used {
                    active
                } else {
                    SixAxisSensorState::default()
                };
                entry.sixaxis(lifo).push(&memory, state);
            }
        }
    }

    /// The devices without host input: empty samples, so their sampling numbers advance.
    pub fn update_other_devices(&mut self, mouse_and_keyboard: bool) {
        let Some(memory) = &self.memory else {
            return;
        };
        HOME_BUTTON.push(memory, Default::default());
        SLEEP_BUTTON.push(memory, Default::default());
        CAPTURE_BUTTON.push(memory, Default::default());
        if self.debug_pad_active {
            DEBUG_PAD.push(memory, Default::default());
        }
        if self.touch_screen_active {
            TOUCH_SCREEN.push(memory, Default::default());
        }
        if mouse_and_keyboard && self.mouse_active {
            MOUSE.push(memory, Default::default());
        }
        if mouse_and_keyboard && self.keyboard_active {
            KEYBOARD.push(memory, Default::default());
        }
    }

    pub fn tick(&mut self, hub: &NpadHub, tick: u64) {
        self.update_npads(hub);
        if tick.is_multiple_of(MOTION_TICKS) {
            self.update_motion(hub);
        }
        if tick.is_multiple_of(DEFAULT_TICKS) {
            self.update_other_devices(tick.is_multiple_of(MOUSE_KEYBOARD_TICKS));
        }
    }

    pub fn set_hold_type(&mut self, hold_type: u64) {
        self.hold_type = hold_type;
        if let Some(memory) = &self.memory {
            layout::set_hold_type(memory, hold_type);
        }
    }
}

const IDENTITY: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

impl Npad {
    /// Advances the orientation by one motion period of `gyro` (revolutions per second) and
    /// returns the sample.
    fn integrate(&mut self, accel: [f32; 3], gyro: [f32; 3]) -> SixAxisSensorState {
        let dt = MOTION_PERIOD_NS as f32 / 1e9;
        for (rotation, rate) in self.rotation.iter_mut().zip(gyro) {
            *rotation += rate * dt;
        }
        let half = std::f32::consts::PI * dt;
        let [x, y, z] = gyro.map(|rate| rate * half);
        let [w0, x0, y0, z0] = self.orientation;
        let q = [
            w0 - x0 * x - y0 * y - z0 * z,
            x0 + w0 * x + y0 * z - z0 * y,
            y0 + w0 * y + z0 * x - x0 * z,
            z0 + w0 * z + x0 * y - y0 * x,
        ];
        let norm = q.iter().map(|v| v * v).sum::<f32>().sqrt();
        self.orientation = q.map(|v| v / norm);
        let [w, x, y, z] = self.orientation;
        SixAxisSensorState {
            delta_time: MOTION_PERIOD_NS,
            accel,
            gyro,
            rotation: self.rotation,
            orientation: [
                [
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y - w * z),
                    2.0 * (x * z + w * y),
                ],
                [
                    2.0 * (x * y + w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z - w * x),
                ],
                [
                    2.0 * (x * z - w * y),
                    2.0 * (y * z + w * x),
                    1.0 - 2.0 * (x * x + y * y),
                ],
            ],
            attributes: 1,
            ..SixAxisSensorState::default()
        }
    }
}

/// Writes a newly connected controller into its npad entry (the reference's
/// `InitNewlyAddedController`), starting with its rumble off, and signals the style-set event.
fn connect(memory: &SharedMemory, index: usize, npad: &mut Npad, style: Style, halves: Halves) {
    let entry = NpadEntry::new(index);
    clear_device(memory, entry);
    let joy_colors = |left: bool, right: bool| {
        let mut colors = [0u32; 5];
        if left {
            colors[1] = u32::from_le_bytes(LEFT_COLORS[0]);
            colors[2] = u32::from_le_bytes(LEFT_COLORS[1]);
        }
        if right {
            colors[3] = u32::from_le_bytes(RIGHT_COLORS[0]);
            colors[4] = u32::from_le_bytes(RIGHT_COLORS[1]);
        }
        colors
    };
    let full_key = |colors: [[u8; 4]; 2]| {
        [
            0,
            u32::from_le_bytes(colors[0]),
            u32::from_le_bytes(colors[1]),
        ]
    };
    const IS_VERTICAL: u64 = 1 << 11;
    const IS_HORIZONTAL: u64 = 1 << 12;
    const USE_PLUS: u64 = 1 << 13;
    const USE_MINUS: u64 = 1 << 14;
    const USE_DIRECTIONAL_BUTTONS: u64 = 1 << 15;
    let (device_type, properties, footer, sixaxis): (u32, u64, u8, &[SixAxisLifo]) = match style {
        Style::FullKey => {
            put(
                memory,
                entry.at(NpadEntry::FULL_KEY_COLOR),
                full_key(FULL_KEY_COLORS),
            );
            put(memory, entry.at(NpadEntry::BATTERY_DUAL), BATTERY_FULL);
            (
                1 << 0,
                IS_VERTICAL | USE_PLUS | USE_MINUS,
                12,
                &[SixAxisLifo::FullKey],
            )
        }
        Style::Handheld => {
            put(
                memory,
                entry.at(NpadEntry::FULL_KEY_COLOR),
                full_key(FULL_KEY_COLORS),
            );
            put(
                memory,
                entry.at(NpadEntry::JOY_COLOR),
                joy_colors(true, true),
            );
            (
                1 << 2 | 1 << 3,
                IS_VERTICAL | USE_PLUS | USE_MINUS | USE_DIRECTIONAL_BUTTONS,
                4,
                &[SixAxisLifo::Handheld],
            )
        }
        Style::JoyDual => {
            put(
                memory,
                entry.at(NpadEntry::FULL_KEY_COLOR),
                full_key(LEFT_COLORS),
            );
            put(
                memory,
                entry.at(NpadEntry::JOY_COLOR),
                joy_colors(halves.left, halves.right),
            );
            for battery in [
                NpadEntry::BATTERY_DUAL,
                NpadEntry::BATTERY_LEFT,
                NpadEntry::BATTERY_RIGHT,
            ] {
                put(memory, entry.at(battery), BATTERY_FULL);
            }
            (
                1 << 4 | 1 << 5,
                IS_VERTICAL | USE_PLUS | USE_MINUS | USE_DIRECTIONAL_BUTTONS,
                5,
                &[SixAxisLifo::DualLeft, SixAxisLifo::DualRight],
            )
        }
        Style::JoyLeft => {
            put(
                memory,
                entry.at(NpadEntry::FULL_KEY_COLOR),
                full_key(LEFT_COLORS),
            );
            put(
                memory,
                entry.at(NpadEntry::JOY_COLOR),
                joy_colors(true, false),
            );
            put(memory, entry.at(NpadEntry::BATTERY_DUAL), BATTERY_FULL);
            (1 << 4, IS_HORIZONTAL | USE_MINUS, 8, &[SixAxisLifo::Left])
        }
        Style::JoyRight => {
            put(
                memory,
                entry.at(NpadEntry::FULL_KEY_COLOR),
                full_key(RIGHT_COLORS),
            );
            put(
                memory,
                entry.at(NpadEntry::JOY_COLOR),
                joy_colors(false, true),
            );
            put(memory, entry.at(NpadEntry::BATTERY_RIGHT), BATTERY_FULL);
            (1 << 5, IS_HORIZONTAL | USE_PLUS, 10, &[SixAxisLifo::Right])
        }
        Style::Gc => (1 << 0, IS_VERTICAL | USE_PLUS, 0, &[]),
    };
    put(memory, entry.at(NpadEntry::STYLE_SET), style.tag());
    put(memory, entry.at(NpadEntry::DEVICE_TYPE), device_type);
    put(memory, entry.at(NpadEntry::SYSTEM_PROPERTIES), properties);
    put(memory, entry.at(NpadEntry::FOOTER_TYPE), footer);
    for &lifo in sixaxis {
        put(
            memory,
            entry.at(NpadEntry::SIXAXIS_PROPERTIES + lifo as u64),
            1u8,
        );
    }
    npad.single = matches!(style, Style::JoyLeft | Style::JoyRight);
    put(
        memory,
        entry.at(NpadEntry::ASSIGNMENT_MODE),
        u32::from(npad.single),
    );
    npad.style = Some(style);
    npad.halves = halves;
    npad.vibration = [DEFAULT_VIBRATION; 2];
    npad.orientation = [1.0, 0.0, 0.0, 0.0];
    npad.rotation = [0.0; 3];
    npad.signal_style_change();
    write_empty_entries(memory, entry);
}

/// Clears the controller from its npad entry (the reference's `DisconnectNpad`; the assignment
/// mode stays), forgets its rumble and signals the style-set event.
fn disconnect(memory: &SharedMemory, index: usize, npad: &mut Npad) {
    let entry = NpadEntry::new(index);
    clear_device(memory, entry);
    put(memory, entry.at(NpadEntry::BUTTON_PROPERTIES), 0u32);
    put(memory, entry.at(NpadEntry::SIXAXIS_PROPERTIES), [0u8; 6]);
    npad.style = None;
    npad.vibration = [DEFAULT_VIBRATION; 2];
    npad.signal_style_change();
    write_empty_entries(memory, entry);
}

fn clear_device(memory: &SharedMemory, entry: NpadEntry) {
    put(memory, entry.at(NpadEntry::STYLE_SET), 0u32);
    put(memory, entry.at(NpadEntry::DEVICE_TYPE), 0u32);
    put(memory, entry.at(NpadEntry::SYSTEM_PROPERTIES), 0u64);
    put(
        memory,
        entry.at(NpadEntry::FULL_KEY_COLOR),
        [NO_CONTROLLER, 0, 0],
    );
    put(
        memory,
        entry.at(NpadEntry::JOY_COLOR),
        [NO_CONTROLLER, 0, 0, 0, 0],
    );
    put(memory, entry.at(NpadEntry::BATTERY_DUAL), [0u32; 3]);
    put(memory, entry.at(NpadEntry::FOOTER_TYPE), 0u8);
}

/// One empty sample in every generic lifo of the entry, as a (dis)connection writes.
fn write_empty_entries(memory: &SharedMemory, entry: NpadEntry) {
    for lifo in [
        PadLifo::FullKey,
        PadLifo::Handheld,
        PadLifo::JoyDual,
        PadLifo::JoyLeft,
        PadLifo::JoyRight,
        PadLifo::Palma,
        PadLifo::SystemExt,
    ] {
        entry.pad(lifo).push(memory, NpadGenericState::default());
    }
    entry.gc_trigger().push(memory, GcTriggerState::default());
}

/// The hid state shared by every hid session, and the npad hub it samples.
pub(crate) struct Hid {
    pub(super) hub: Arc<NpadHub>,
    pub(super) state: parking_lot::Mutex<HidState>,
}

impl Hid {
    /// The npads the title sees connected, as the controller applet reports them: how many of
    /// players 1-8 and handheld are connected, and the first connected npad (player 1 when none
    /// is).
    pub(crate) fn controller_setup(&self) -> ControllerSetup {
        let state = self.state.lock();
        let connected = |index: &usize| state.npads[*index].style.is_some();
        let npad_id = |index: usize| match index {
            OTHER => 0x10,
            HANDHELD => 0x20,
            player => player as u32,
        };
        ControllerSetup {
            player_count: (0..NPAD_COUNT)
                .filter(|&index| index != OTHER)
                .filter(connected)
                .count() as i8,
            npad_id: (0..NPAD_COUNT).find(connected).map_or(0, npad_id),
        }
    }
}

/// Runs the updater on its own host thread until the last `Hid` reference is gone.
pub(super) fn spawn_updater(hid: &Arc<Hid>) {
    let weak: Weak<Hid> = Arc::downgrade(hid);
    let spawned = std::thread::Builder::new()
        .name("hid updater".into())
        .spawn(move || {
            let mut next = Instant::now();
            for tick in 0u64.. {
                next += NPAD_PERIOD;
                match next.checked_duration_since(Instant::now()) {
                    Some(wait) => std::thread::sleep(wait),
                    None => next = Instant::now(),
                }
                let Some(hid) = weak.upgrade() else { return };
                hid.state.lock().tick(&hid.hub, tick);
            }
        });
    if let Err(err) = spawned {
        tracing::error!("spawning the hid updater: {err}");
    }
}
