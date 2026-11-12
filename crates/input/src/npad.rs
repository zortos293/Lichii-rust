//! The per-player state the hid service consumes: Switch button bits, sticks in Switch range and
//! motion in Switch units.

use bitflags::bitflags;
use lichii_common::settings::{ControllerType, NpadButton};

/// Full stick deflection on the Switch (`HID_JOYSTICK_MAX`).
pub const STICK_MAX: i32 = 0x7fff;
/// Full analog trigger travel (GameCube ZL/ZR).
pub const TRIGGER_MAX: u32 = 0x7fff;

bitflags! {
    /// `nn::hid::NpadButton` bits. Bits 0-34 match the guest layout exactly so hid can copy
    /// them. C, GL and GR (bits 40-42) are host-side: Switch 2's guest layout for them is not
    /// documented yet, so the hid service maps them.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct NpadButtons: u64 {
        const A = 1 << 0;
        const B = 1 << 1;
        const X = 1 << 2;
        const Y = 1 << 3;
        const STICK_L = 1 << 4;
        const STICK_R = 1 << 5;
        const L = 1 << 6;
        const R = 1 << 7;
        const ZL = 1 << 8;
        const ZR = 1 << 9;
        const PLUS = 1 << 10;
        const MINUS = 1 << 11;
        const LEFT = 1 << 12;
        const UP = 1 << 13;
        const RIGHT = 1 << 14;
        const DOWN = 1 << 15;
        const STICK_L_LEFT = 1 << 16;
        const STICK_L_UP = 1 << 17;
        const STICK_L_RIGHT = 1 << 18;
        const STICK_L_DOWN = 1 << 19;
        const STICK_R_LEFT = 1 << 20;
        const STICK_R_UP = 1 << 21;
        const STICK_R_RIGHT = 1 << 22;
        const STICK_R_DOWN = 1 << 23;
        const LEFT_SL = 1 << 24;
        const LEFT_SR = 1 << 25;
        const RIGHT_SL = 1 << 26;
        const RIGHT_SR = 1 << 27;
        const PALMA = 1 << 28;
        const VERIFICATION = 1 << 29;
        const HANDHELD_LEFT_B = 1 << 30;
        const LAGON_C_LEFT = 1 << 31;
        const LAGON_C_UP = 1 << 32;
        const LAGON_C_RIGHT = 1 << 33;
        const LAGON_C_DOWN = 1 << 34;
        const C = 1 << 40;
        const GL = 1 << 41;
        const GR = 1 << 42;
        /// Home and Capture are not npad bits on the guest; hid routes them to the system
        /// button service. They are carried here so hotkeys and the applet layer can see them.
        const HOME = 1 << 48;
        const CAPTURE = 1 << 49;
    }
}

impl NpadButtons {
    pub const fn from_button(button: NpadButton) -> Self {
        match button {
            NpadButton::A => Self::A,
            NpadButton::B => Self::B,
            NpadButton::X => Self::X,
            NpadButton::Y => Self::Y,
            NpadButton::LStick => Self::STICK_L,
            NpadButton::RStick => Self::STICK_R,
            NpadButton::L => Self::L,
            NpadButton::R => Self::R,
            NpadButton::ZL => Self::ZL,
            NpadButton::ZR => Self::ZR,
            NpadButton::Plus => Self::PLUS,
            NpadButton::Minus => Self::MINUS,
            NpadButton::DPadLeft => Self::LEFT,
            NpadButton::DPadUp => Self::UP,
            NpadButton::DPadRight => Self::RIGHT,
            NpadButton::DPadDown => Self::DOWN,
            NpadButton::LeftSL => Self::LEFT_SL,
            NpadButton::LeftSR => Self::LEFT_SR,
            NpadButton::RightSL => Self::RIGHT_SL,
            NpadButton::RightSR => Self::RIGHT_SR,
            NpadButton::Home => Self::HOME,
            NpadButton::Capture => Self::CAPTURE,
            NpadButton::C => Self::C,
            NpadButton::GL => Self::GL,
            NpadButton::GR => Self::GR,
        }
    }

    /// Whether every button of `combo` is held; an empty combo never matches.
    pub fn holds(self, combo: &[NpadButton]) -> bool {
        !combo.is_empty() && combo.iter().all(|&b| self.contains(Self::from_button(b)))
    }
}

/// Stick position in Switch units, each axis in `-STICK_MAX..=STICK_MAX`, +y up.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StickPosition {
    pub x: i32,
    pub y: i32,
}

/// One six-axis sample in Switch units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MotionSample {
    /// Acceleration in g.
    pub accel: [f32; 3],
    /// Angular velocity in revolutions per second (1.0 = 360°/s).
    pub gyro: [f32; 3],
}

/// Everything hid needs for one player at one sampling point. `Copy`, fixed size, no heap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NpadState {
    pub connected: bool,
    pub controller: ControllerType,
    pub buttons: NpadButtons,
    pub left_stick: StickPosition,
    pub right_stick: StickPosition,
    /// Analog trigger travel from the ZL/ZR bindings, `0..=TRIGGER_MAX`; a digital press reads
    /// as full travel. On a GameCube controller these are its L/R triggers and the ZL/ZR bits
    /// stay clear.
    pub left_trigger: u32,
    pub right_trigger: u32,
    /// `None` when the player's device has no motion sensors or motion is disabled.
    pub motion: Option<MotionSample>,
}

impl Default for NpadState {
    fn default() -> Self {
        Self {
            connected: false,
            controller: ControllerType::ProController,
            buttons: NpadButtons::empty(),
            left_stick: StickPosition::default(),
            right_stick: StickPosition::default(),
            left_trigger: 0,
            right_trigger: 0,
            motion: None,
        }
    }
}

/// A rumble request from the guest, `nn::hid::VibrationValue`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VibrationValue {
    pub low_amplitude: f32,
    pub low_frequency: f32,
    pub high_amplitude: f32,
    pub high_frequency: f32,
}

impl VibrationValue {
    /// Low- and high-frequency motor strengths for SDL rumble, scaled by `strength` (0..=1).
    pub fn to_rumble(self, strength: f32) -> (u16, u16) {
        let motor = |amp: f32| (amp.clamp(0.0, 1.0) * strength.clamp(0.0, 1.0) * 65535.0) as u16;
        (motor(self.low_amplitude), motor(self.high_amplitude))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_button_has_a_distinct_bit() {
        let mut all = NpadButtons::empty();
        for button in NpadButton::ALL {
            let bit = NpadButtons::from_button(button);
            assert_eq!(bit.bits().count_ones(), 1);
            assert!(!all.intersects(bit), "{button:?}");
            all |= bit;
        }
    }

    #[test]
    fn guest_bits_match_nn_hid() {
        assert_eq!(NpadButtons::ZR.bits(), 1 << 9);
        assert_eq!(NpadButtons::DOWN.bits(), 1 << 15);
        assert_eq!(NpadButtons::RIGHT_SR.bits(), 1 << 27);
        assert_eq!(NpadButtons::LAGON_C_DOWN.bits(), 1 << 34);
    }

    #[test]
    fn combos() {
        use NpadButton::*;
        let held = NpadButtons::HOME | NpadButtons::PLUS | NpadButtons::A;
        assert!(held.holds(&[Home, Plus]));
        assert!(!held.holds(&[Home, Minus]));
        assert!(!held.holds(&[]));
    }

    #[test]
    fn rumble_scales() {
        let v = VibrationValue {
            low_amplitude: 1.0,
            high_amplitude: 0.5,
            ..Default::default()
        };
        assert_eq!(v.to_rumble(1.0), (65535, 32767));
        assert_eq!(v.to_rumble(0.0), (0, 0));
    }
}
