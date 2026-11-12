//! Host input: keyboards, mice and gamepads, mapped through `ControlSettings` into per-player
//! npad state that the hid service copies into guest shared memory.
//!
//! Data flow: window events and the SDL driver ([`SdlInput`]) update a [`HostInput`]; an
//! [`InputMapper`] compiled from the settings turns it into one [`NpadState`] per player; the
//! frontend publishes those into an [`NpadHub`] that the hid service samples, and reads rumble
//! requests back out.

pub mod capture;
pub mod gamepad;
pub mod hub;
pub mod keys;
pub mod mapping;
pub mod npad;
pub mod profiles;
pub mod sdl;
pub mod stick;

pub use gamepad::{Gamepad, GamepadState, PadAxis, PadButton};
pub use hub::NpadHub;
pub use keys::{Key, KeyChord, KeySet, MouseButton};
pub use mapping::{HostInput, InputMapper, PlayerMapping, Source};
pub use npad::{
    MotionSample, NpadButtons, NpadState, STICK_MAX, StickPosition, TRIGGER_MAX, VibrationValue,
};
pub use profiles::{ProfileError, ProfileStore};
pub use sdl::{SdlError, SdlInput};
