//! Gamepad state in SDL's terms. The SDL driver fills it; the mapper reads it. Kept free of SDL
//! types so mapping is testable without a device.

macro_rules! named_enum {
    ($(#[$meta:meta])* $vis:vis enum $ty:ident { $($variant:ident => $name:literal),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        $vis enum $ty { $($variant),* }

        impl $ty {
            pub const ALL: &'static [Self] = &[$(Self::$variant),*];

            /// SDL's name, as stored in `InputBinding`.
            pub fn name(self) -> &'static str {
                match self { $(Self::$variant => $name),* }
            }

            pub fn from_name(name: &str) -> Option<Self> {
                Self::ALL.iter().copied().find(|v| v.name() == name)
            }

            /// SDL's enum value for this input.
            pub fn from_sdl_index(index: usize) -> Option<Self> {
                Self::ALL.get(index).copied()
            }
        }
    };
}

named_enum! {
    /// SDL3 gamepad buttons in `SDL_GamepadButton` order. `a`/`b`/`x`/`y` are the south, east,
    /// west and north face buttons, whatever the controller prints on them.
    pub enum PadButton {
        South => "a",
        East => "b",
        West => "x",
        North => "y",
        Back => "back",
        Guide => "guide",
        Start => "start",
        LeftStick => "leftstick",
        RightStick => "rightstick",
        LeftShoulder => "leftshoulder",
        RightShoulder => "rightshoulder",
        DPadUp => "dpup",
        DPadDown => "dpdown",
        DPadLeft => "dpleft",
        DPadRight => "dpright",
        Misc1 => "misc1",
        RightPaddle1 => "rightpaddle1",
        LeftPaddle1 => "leftpaddle1",
        RightPaddle2 => "rightpaddle2",
        LeftPaddle2 => "leftpaddle2",
        Touchpad => "touchpad",
        Misc2 => "misc2",
        Misc3 => "misc3",
        Misc4 => "misc4",
        Misc5 => "misc5",
        Misc6 => "misc6",
    }
}

named_enum! {
    /// SDL3 gamepad axes in `SDL_GamepadAxis` order.
    pub enum PadAxis {
        LeftX => "leftx",
        LeftY => "lefty",
        RightX => "rightx",
        RightY => "righty",
        LeftTrigger => "lefttrigger",
        RightTrigger => "righttrigger",
    }
}

/// One gamepad's inputs at the last poll.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GamepadState {
    /// One bit per [`PadButton`].
    pub buttons: u32,
    /// Sticks in -1..=1 with SDL's convention (+y is down); triggers in 0..=1.
    pub axes: [f32; 6],
    /// Angular velocity in radians per second, SDL sensor axes.
    pub gyro: Option<[f32; 3]>,
    /// Acceleration in m/s², SDL sensor axes.
    pub accel: Option<[f32; 3]>,
}

impl GamepadState {
    pub fn set_button(&mut self, button: PadButton, down: bool) {
        let bit = 1u32 << button as u32;
        if down {
            self.buttons |= bit;
        } else {
            self.buttons &= !bit;
        }
    }

    pub fn button(&self, button: PadButton) -> bool {
        self.buttons & (1 << button as u32) != 0
    }

    pub fn axis(&self, axis: PadAxis) -> f32 {
        self.axes[axis as usize]
    }

    pub fn set_axis(&mut self, axis: PadAxis, value: f32) {
        self.axes[axis as usize] = value;
    }
}

/// A connected gamepad. `guid` plus `index` (the n-th connected pad with that GUID) is how
/// settings name it, so two identical controllers stay distinct across reconnects.
#[derive(Clone, Debug, PartialEq)]
pub struct Gamepad {
    pub guid: String,
    pub index: u32,
    pub name: String,
    pub has_rumble: bool,
    pub state: GamepadState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_sdl() {
        assert_eq!(PadButton::from_sdl_index(1), Some(PadButton::East));
        assert_eq!(PadButton::from_name("dpup"), Some(PadButton::DPadUp));
        assert_eq!(PadButton::ALL.len(), 26);
        assert_eq!(
            PadAxis::from_name("righttrigger"),
            Some(PadAxis::RightTrigger)
        );
    }

    #[test]
    fn button_bits() {
        let mut pad = GamepadState::default();
        pad.set_button(PadButton::Misc6, true);
        assert!(pad.button(PadButton::Misc6));
        pad.set_button(PadButton::Misc6, false);
        assert_eq!(pad.buttons, 0);
    }
}
