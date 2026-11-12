//! Turns host input into npad state through each player's `PlayerInput` bindings.
//!
//! Bindings are compiled once (names parsed, the device looked up) whenever settings or the set of
//! connected gamepads change. Resolving a frame then reads only flat arrays: no allocation, no
//! string compares, no locks.

use std::f32::consts::TAU;

use lichii_common::settings::{
    ControlSettings, ControllerType, InputBinding, InputDevice, MAX_PLAYERS, NpadButton,
    PlayerInput, StickBinding,
};

use crate::gamepad::{Gamepad, GamepadState, PadAxis, PadButton};
use crate::keys::{Key, KeySet, MouseButton};
use crate::npad::{MotionSample, NpadButtons, NpadState, STICK_MAX, StickPosition, TRIGGER_MAX};
use crate::stick;

const STANDARD_GRAVITY: f32 = 9.80665;

/// Everything the host reports, updated from window events and the SDL driver.
#[derive(Clone, Debug, Default)]
pub struct HostInput {
    pub keys: KeySet,
    /// One bit per [`MouseButton`].
    pub mouse_buttons: u8,
    pub gamepads: Vec<Gamepad>,
}

impl HostInput {
    pub fn set_mouse_button(&mut self, button: MouseButton, down: bool) {
        if down {
            self.mouse_buttons |= button.bit();
        } else {
            self.mouse_buttons &= !button.bit();
        }
    }

    pub fn mouse_button(&self, button: MouseButton) -> bool {
        self.mouse_buttons & button.bit() != 0
    }

    /// The connected gamepad a setting names, by GUID and per-GUID index.
    pub fn find_gamepad(&self, guid: &str, index: u32) -> Option<usize> {
        self.gamepads
            .iter()
            .position(|g| g.guid == guid && g.index == index)
    }
}

/// One host input, parsed from an `InputBinding`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Source {
    #[default]
    None,
    Key(Key),
    Mouse(MouseButton),
    Pad(PadButton),
    /// Pressed past `threshold` in the direction of `sign` (+1 or -1).
    Axis {
        axis: PadAxis,
        threshold: f32,
        sign: f32,
    },
}

impl Source {
    /// Parses a binding; names this build does not know resolve to `None` with a warning.
    pub fn compile(binding: &InputBinding) -> Self {
        let parsed = match binding {
            InputBinding::None => return Self::None,
            InputBinding::Key(name) => Key::from_name(name).map(Self::Key),
            InputBinding::MouseButton(name) => MouseButton::from_name(name).map(Self::Mouse),
            InputBinding::GamepadButton { button } => PadButton::from_name(button).map(Self::Pad),
            InputBinding::GamepadAxis {
                axis,
                threshold,
                direction,
            } => PadAxis::from_name(axis).map(|axis| Self::Axis {
                axis,
                threshold: threshold.clamp(0.05, 0.95),
                sign: if *direction < 0 { -1.0 } else { 1.0 },
            }),
        };
        parsed.unwrap_or_else(|| {
            tracing::warn!("unknown input binding {binding:?}; leaving it unbound");
            Self::None
        })
    }

    /// The binding this source came from, for writing back to settings.
    pub fn to_binding(self) -> InputBinding {
        match self {
            Self::None => InputBinding::None,
            Self::Key(key) => InputBinding::Key(key.name().into()),
            Self::Mouse(button) => InputBinding::MouseButton(button.name().into()),
            Self::Pad(button) => InputBinding::GamepadButton {
                button: button.name().into(),
            },
            Self::Axis {
                axis,
                threshold,
                sign,
            } => InputBinding::GamepadAxis {
                axis: axis.name().into(),
                threshold,
                direction: if sign < 0.0 { -1 } else { 1 },
            },
        }
    }

    /// How far the input is pressed, 0..=1. Keys, mouse and pad buttons are 0 or 1.
    pub fn analog(self, host: &HostInput, pad: Option<&GamepadState>, keyboard: bool) -> f32 {
        let on = |b: bool| if b { 1.0 } else { 0.0 };
        match self {
            Self::None => 0.0,
            Self::Key(key) => on(keyboard && host.keys.is_down(key)),
            Self::Mouse(button) => on(keyboard && host.mouse_button(button)),
            Self::Pad(button) => on(pad.is_some_and(|p| p.button(button))),
            Self::Axis { axis, sign, .. } => {
                pad.map_or(0.0, |p| (p.axis(axis) * sign).clamp(0.0, 1.0))
            }
        }
    }

    pub fn pressed(self, host: &HostInput, pad: Option<&GamepadState>, keyboard: bool) -> bool {
        match self {
            Self::Axis { threshold, .. } => self.analog(host, pad, keyboard) > threshold,
            _ => self.analog(host, pad, keyboard) > 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct StickMapping {
    analog: Option<(PadAxis, PadAxis)>,
    invert_x: bool,
    invert_y: bool,
    up: Source,
    down: Source,
    left: Source,
    right: Source,
    modifier: Source,
    modifier_scale: f32,
    deadzone: f32,
    range: f32,
}

impl StickMapping {
    fn compile(stick: &StickBinding) -> Self {
        let axis = |name: &Option<String>| {
            let name = name.as_deref()?;
            let parsed = PadAxis::from_name(name);
            if parsed.is_none() {
                tracing::warn!("unknown stick axis {name:?}; leaving it unbound");
            }
            parsed
        };
        Self {
            analog: axis(&stick.analog_x).zip(axis(&stick.analog_y)),
            invert_x: stick.invert_x,
            invert_y: stick.invert_y,
            up: Source::compile(&stick.up),
            down: Source::compile(&stick.down),
            left: Source::compile(&stick.left),
            right: Source::compile(&stick.right),
            modifier: Source::compile(&stick.modifier),
            modifier_scale: stick.modifier_scale,
            deadzone: stick.deadzone,
            range: stick.range,
        }
    }

    /// Host units, +y up.
    fn resolve(&self, host: &HostInput, pad: Option<&GamepadState>, keyboard: bool) -> (f32, f32) {
        let (mut x, mut y) = match (self.analog, pad) {
            (Some((ax, ay)), Some(pad)) => {
                stick::shape_analog(pad.axis(ax), -pad.axis(ay), self.deadzone, self.range)
            }
            _ => (0.0, 0.0),
        };
        let held = |s: Source| s.pressed(host, pad, keyboard);
        let scale = if held(self.modifier) {
            self.modifier_scale
        } else {
            1.0
        };
        let (dx, dy) = stick::digital(
            held(self.up),
            held(self.down),
            held(self.left),
            held(self.right),
            scale,
        );
        (x, y) = stick::clamp_unit(x + dx, y + dy);
        if self.invert_x {
            x = -x;
        }
        if self.invert_y {
            y = -y;
        }
        (x, y)
    }
}

/// One player's bindings, compiled.
#[derive(Clone, Debug)]
pub struct PlayerMapping {
    connected: bool,
    controller: ControllerType,
    /// Index into `HostInput::gamepads` of the player's device.
    gamepad: Option<usize>,
    buttons: Vec<(NpadButtons, Source)>,
    zl: Source,
    zr: Source,
    left: StickMapping,
    right: StickMapping,
    motion: bool,
    rumble_strength: f32,
}

impl PlayerMapping {
    pub fn compile(player: &PlayerInput, host: &HostInput) -> Self {
        let gamepad = match &player.device {
            InputDevice::Gamepad { guid, index, .. } => host.find_gamepad(guid, *index),
            InputDevice::None | InputDevice::Keyboard => None,
        };
        let controller = player.controller_type;
        let source = |b: NpadButton| player.buttons.get(&b).map_or(Source::None, Source::compile);
        let bound = |b: NpadButton| {
            if controller.has_button(b) {
                source(b)
            } else {
                Source::None
            }
        };
        let buttons = NpadButton::ALL
            .into_iter()
            .map(|b| (NpadButtons::from_button(b), bound(b)))
            .filter(|(_, source)| *source != Source::None)
            .collect();
        let stick = |has: bool, s: &StickBinding| {
            if has {
                StickMapping::compile(s)
            } else {
                StickMapping::default()
            }
        };
        Self {
            connected: player.connected,
            controller,
            gamepad,
            buttons,
            zl: source(NpadButton::ZL),
            zr: source(NpadButton::ZR),
            left: stick(controller.has_left_stick(), &player.left_stick),
            right: stick(controller.has_right_stick(), &player.right_stick),
            motion: player.motion_enabled,
            rumble_strength: if player.vibration_enabled {
                f32::from(player.vibration_strength.min(100)) / 100.0
            } else {
                0.0
            },
        }
    }

    /// The device index rumble for this player goes to, and its strength (0 when disabled).
    pub fn rumble_target(&self) -> Option<(usize, f32)> {
        self.gamepad
            .filter(|_| self.rumble_strength > 0.0)
            .map(|g| (g, self.rumble_strength))
    }

    pub fn resolve(&self, host: &HostInput, keyboard: bool) -> NpadState {
        if !self.connected {
            return NpadState {
                controller: self.controller,
                ..NpadState::default()
            };
        }
        let pad = self
            .gamepad
            .and_then(|i| host.gamepads.get(i))
            .map(|g| &g.state);
        let mut buttons = NpadButtons::empty();
        for &(bit, source) in &self.buttons {
            if source.pressed(host, pad, keyboard) {
                buttons |= bit;
            }
        }
        let (lx, ly) = self.left.resolve(host, pad, keyboard);
        let (rx, ry) = self.right.resolve(host, pad, keyboard);
        let left_stick = StickPosition {
            x: stick::to_switch(lx),
            y: stick::to_switch(ly),
        };
        let right_stick = StickPosition {
            x: stick::to_switch(rx),
            y: stick::to_switch(ry),
        };
        buttons |= stick_directions(left_stick, true) | stick_directions(right_stick, false);
        let trigger = |s: Source| (s.analog(host, pad, keyboard) * TRIGGER_MAX as f32) as u32;
        NpadState {
            connected: true,
            controller: self.controller,
            buttons,
            left_stick,
            right_stick,
            left_trigger: trigger(self.zl),
            right_trigger: trigger(self.zr),
            motion: pad
                .filter(|_| self.motion)
                .and_then(|p| p.gyro.zip(p.accel))
                .map(|(gyro, accel)| motion_from_sdl(gyro, accel)),
        }
    }
}

/// Converts SDL sensor readings to Switch six-axis units and axes, as Lichii's SDL driver does.
pub fn motion_from_sdl(gyro: [f32; 3], accel: [f32; 3]) -> MotionSample {
    MotionSample {
        accel: [
            -accel[0] / STANDARD_GRAVITY,
            accel[2] / STANDARD_GRAVITY,
            -accel[1] / STANDARD_GRAVITY,
        ],
        gyro: [gyro[0] / TAU, -gyro[2] / TAU, gyro[1] / TAU],
    }
}

/// The pseudo-buttons hid reports when a stick passes half its travel.
fn stick_directions(stick: StickPosition, left: bool) -> NpadButtons {
    const HALF: i32 = STICK_MAX / 2;
    let [l, u, r, d] = if left {
        [
            NpadButtons::STICK_L_LEFT,
            NpadButtons::STICK_L_UP,
            NpadButtons::STICK_L_RIGHT,
            NpadButtons::STICK_L_DOWN,
        ]
    } else {
        [
            NpadButtons::STICK_R_LEFT,
            NpadButtons::STICK_R_UP,
            NpadButtons::STICK_R_RIGHT,
            NpadButtons::STICK_R_DOWN,
        ]
    };
    let mut out = NpadButtons::empty();
    out.set(l, stick.x < -HALF);
    out.set(r, stick.x > HALF);
    out.set(u, stick.y > HALF);
    out.set(d, stick.y < -HALF);
    out
}

/// All players' compiled bindings. Rebuild it with [`InputMapper::new`] when `ControlSettings`
/// change or a gamepad connects or disconnects.
#[derive(Clone, Debug)]
pub struct InputMapper {
    players: Vec<PlayerMapping>,
    keyboard_enabled: bool,
}

impl InputMapper {
    pub fn new(controls: &ControlSettings, host: &HostInput) -> Self {
        Self {
            players: controls
                .players
                .iter()
                .take(MAX_PLAYERS)
                .map(|p| PlayerMapping::compile(p, host))
                .collect(),
            keyboard_enabled: controls.keyboard_enabled,
        }
    }

    pub fn players(&self) -> &[PlayerMapping] {
        &self.players
    }

    pub fn resolve(&self, player: usize, host: &HostInput) -> NpadState {
        self.players
            .get(player)
            .map_or_else(NpadState::default, |p| {
                p.resolve(host, self.keyboard_enabled)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::settings::Settings;

    fn key(name: &str) -> Key {
        Key::from_name(name).unwrap()
    }

    fn pad_player() -> (PlayerInput, HostInput) {
        let player = PlayerInput::default_gamepad(InputDevice::Gamepad {
            guid: "g".into(),
            index: 1,
            name: "Pad".into(),
        });
        let mut host = HostInput::default();
        for index in 0..2 {
            host.gamepads.push(Gamepad {
                guid: "g".into(),
                index,
                name: "Pad".into(),
                has_rumble: true,
                state: GamepadState::default(),
            });
        }
        (player, host)
    }

    #[test]
    fn keyboard_player_one() {
        let settings = Settings::default();
        let mut host = HostInput::default();
        let mapper = InputMapper::new(&settings.controls, &host);
        assert_eq!(mapper.resolve(0, &host).buttons, NpadButtons::empty());
        host.keys.set(key("KeyC"), true);
        host.keys.set(key("ArrowUp"), true);
        host.keys.set(key("KeyW"), true);
        let state = mapper.resolve(0, &host);
        assert!(state.connected);
        assert_eq!(
            state.buttons,
            NpadButtons::A | NpadButtons::UP | NpadButtons::STICK_L_UP
        );
        assert_eq!(state.left_stick, StickPosition { x: 0, y: STICK_MAX });
        assert!(!mapper.resolve(1, &host).connected);
    }

    #[test]
    fn keyboard_modifier_walks() {
        let settings = Settings::default();
        let mut host = HostInput::default();
        let mapper = InputMapper::new(&settings.controls, &host);
        host.keys.set(key("KeyD"), true);
        host.keys.set(key("ShiftLeft"), true);
        let state = mapper.resolve(0, &host);
        assert_eq!(
            state.left_stick,
            StickPosition {
                x: stick::to_switch(0.5),
                y: 0
            }
        );
    }

    #[test]
    fn disabled_keyboard_reads_nothing() {
        let mut settings = Settings::default();
        settings.controls.keyboard_enabled = false;
        let mut host = HostInput::default();
        host.keys.set(key("KeyC"), true);
        let mapper = InputMapper::new(&settings.controls, &host);
        assert_eq!(mapper.resolve(0, &host).buttons, NpadButtons::empty());
    }

    #[test]
    fn gamepad_by_position_and_index() {
        let (player, mut host) = pad_player();
        let mapping = PlayerMapping::compile(&player, &host);
        host.gamepads[0].state.set_button(PadButton::East, true);
        assert_eq!(mapping.resolve(&host, true).buttons, NpadButtons::empty());
        let pad = &mut host.gamepads[1].state;
        pad.set_button(PadButton::East, true);
        pad.set_button(PadButton::South, true);
        pad.set_axis(PadAxis::RightTrigger, 0.8);
        pad.set_axis(PadAxis::LeftTrigger, 0.3);
        let state = mapping.resolve(&host, true);
        assert_eq!(
            state.buttons,
            NpadButtons::A | NpadButtons::B | NpadButtons::ZR
        );
        assert_eq!(state.right_trigger, (0.8 * TRIGGER_MAX as f32) as u32);
        assert_eq!(state.left_trigger, (0.3 * TRIGGER_MAX as f32) as u32);
    }

    #[test]
    fn analog_stick_flips_sdl_y_and_applies_deadzone() {
        let (player, mut host) = pad_player();
        let mapping = PlayerMapping::compile(&player, &host);
        let pad = &mut host.gamepads[1].state;
        pad.set_axis(PadAxis::LeftY, -1.0);
        pad.set_axis(PadAxis::RightX, 0.1);
        let state = mapping.resolve(&host, true);
        assert_eq!(state.left_stick, StickPosition { x: 0, y: STICK_MAX });
        assert_eq!(state.right_stick, StickPosition::default());
        assert!(state.buttons.contains(NpadButtons::STICK_L_UP));
    }

    #[test]
    fn inverted_axis() {
        let (mut player, mut host) = pad_player();
        player.left_stick.invert_x = true;
        let mapping = PlayerMapping::compile(&player, &host);
        host.gamepads[1].state.set_axis(PadAxis::LeftX, 1.0);
        assert_eq!(mapping.resolve(&host, true).left_stick.x, -STICK_MAX);
    }

    #[test]
    fn missing_device_reads_neutral() {
        let (player, mut host) = pad_player();
        host.gamepads.truncate(1);
        let mapping = PlayerMapping::compile(&player, &host);
        host.gamepads[0].state.set_button(PadButton::East, true);
        let state = mapping.resolve(&host, true);
        assert!(state.connected);
        assert_eq!(state.buttons, NpadButtons::empty());
    }

    #[test]
    fn buttons_the_controller_lacks_are_masked() {
        let (mut player, mut host) = pad_player();
        player.controller_type = ControllerType::LeftJoycon;
        let mapping = PlayerMapping::compile(&player, &host);
        let pad = &mut host.gamepads[1].state;
        pad.set_button(PadButton::East, true);
        pad.set_button(PadButton::DPadUp, true);
        pad.set_axis(PadAxis::RightX, 1.0);
        let state = mapping.resolve(&host, true);
        assert_eq!(state.buttons, NpadButtons::UP);
        assert_eq!(state.right_stick, StickPosition::default());
    }

    #[test]
    fn gamecube_triggers_are_analog_only() {
        let (mut player, mut host) = pad_player();
        player.controller_type = ControllerType::GameCube;
        let mapping = PlayerMapping::compile(&player, &host);
        let pad = &mut host.gamepads[1].state;
        pad.set_axis(PadAxis::LeftTrigger, 1.0);
        pad.set_axis(PadAxis::RightTrigger, 0.25);
        let state = mapping.resolve(&host, true);
        assert_eq!(state.left_trigger, TRIGGER_MAX);
        assert_eq!(state.right_trigger, (0.25 * TRIGGER_MAX as f32) as u32);
        assert_eq!(state.buttons, NpadButtons::empty());
    }

    #[test]
    fn switch2_buttons() {
        let (mut player, mut host) = pad_player();
        player.controller_type = ControllerType::Switch2ProController;
        let mapping = PlayerMapping::compile(&player, &host);
        let pad = &mut host.gamepads[1].state;
        pad.set_button(PadButton::Touchpad, true);
        pad.set_button(PadButton::LeftPaddle1, true);
        pad.set_button(PadButton::RightPaddle1, true);
        let state = mapping.resolve(&host, true);
        assert_eq!(
            state.buttons,
            NpadButtons::C | NpadButtons::GL | NpadButtons::GR
        );
        player.controller_type = ControllerType::ProController;
        let mapping = PlayerMapping::compile(&player, &host);
        assert_eq!(mapping.resolve(&host, true).buttons, NpadButtons::empty());
    }

    #[test]
    fn mixed_keyboard_and_pad_bindings() {
        let (mut player, mut host) = pad_player();
        player
            .buttons
            .insert(NpadButton::Home, InputBinding::Key("KeyH".into()));
        player.buttons.insert(
            NpadButton::Capture,
            InputBinding::MouseButton("Back".into()),
        );
        let mapping = PlayerMapping::compile(&player, &host);
        host.keys.set(key("KeyH"), true);
        host.set_mouse_button(MouseButton::Back, true);
        assert_eq!(
            mapping.resolve(&host, true).buttons,
            NpadButtons::HOME | NpadButtons::CAPTURE
        );
    }

    #[test]
    fn axis_bindings_respect_direction_and_threshold() {
        let (mut player, mut host) = pad_player();
        player.buttons.insert(
            NpadButton::DPadUp,
            InputBinding::GamepadAxis {
                axis: "righty".into(),
                threshold: 0.5,
                direction: -1,
            },
        );
        let mapping = PlayerMapping::compile(&player, &host);
        host.gamepads[1].state.set_axis(PadAxis::RightY, -0.4);
        assert!(
            !mapping
                .resolve(&host, true)
                .buttons
                .contains(NpadButtons::UP)
        );
        host.gamepads[1].state.set_axis(PadAxis::RightY, -0.6);
        assert!(
            mapping
                .resolve(&host, true)
                .buttons
                .contains(NpadButtons::UP)
        );
        host.gamepads[1].state.set_axis(PadAxis::RightY, 0.9);
        assert!(
            !mapping
                .resolve(&host, true)
                .buttons
                .contains(NpadButtons::UP)
        );
    }

    #[test]
    fn unknown_names_are_unbound() {
        assert_eq!(
            Source::compile(&InputBinding::Key("Nope".into())),
            Source::None
        );
        assert_eq!(
            Source::compile(&InputBinding::GamepadButton {
                button: "zz".into()
            }),
            Source::None
        );
    }

    #[test]
    fn sources_round_trip_to_bindings() {
        for binding in [
            InputBinding::None,
            InputBinding::Key("KeyQ".into()),
            InputBinding::MouseButton("Middle".into()),
            InputBinding::GamepadButton {
                button: "dpleft".into(),
            },
            InputBinding::GamepadAxis {
                axis: "lefttrigger".into(),
                threshold: 0.5,
                direction: 1,
            },
        ] {
            assert_eq!(Source::compile(&binding).to_binding(), binding);
        }
    }

    #[test]
    fn motion_needs_both_sensors_and_the_toggle() {
        let (mut player, mut host) = pad_player();
        host.gamepads[1].state.gyro = Some([TAU, 0.0, 0.0]);
        let mapping = PlayerMapping::compile(&player, &host);
        assert_eq!(mapping.resolve(&host, true).motion, None);
        host.gamepads[1].state.accel = Some([0.0, -STANDARD_GRAVITY, 0.0]);
        let motion = mapping.resolve(&host, true).motion.unwrap();
        assert_eq!(motion.gyro, [1.0, 0.0, 0.0]);
        assert_eq!(motion.accel, [0.0, 0.0, 1.0]);
        player.motion_enabled = false;
        let mapping = PlayerMapping::compile(&player, &host);
        assert_eq!(mapping.resolve(&host, true).motion, None);
    }

    #[test]
    fn rumble_target_follows_the_toggle() {
        let (mut player, host) = pad_player();
        player.vibration_strength = 40;
        let mapping = PlayerMapping::compile(&player, &host);
        assert_eq!(mapping.rumble_target(), Some((1, 0.4)));
        player.vibration_enabled = false;
        assert_eq!(PlayerMapping::compile(&player, &host).rumble_target(), None);
    }
}
