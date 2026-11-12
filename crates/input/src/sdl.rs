//! SDL3 gamepads: hotplug, buttons, axes, gyro and accelerometer, rumble.
//!
//! SDL is built from source as a static, video-less library, so neither CI nor users need a system
//! SDL. [`SdlInput`] is not `Send`: create it on the thread that polls it (the frontend's input
//! thread). The first SDL context in a process claims its thread as SDL's main thread.

use sdl3::event::Event;
use sdl3::gamepad::{Axis, Button};
use sdl3::joystick::JoystickId;
use sdl3::sensor::SensorType;

use crate::gamepad::{Gamepad, GamepadState, PadAxis, PadButton};
use crate::mapping::HostInput;

/// SDL failed to initialise its gamepad subsystem or event pump; the message is SDL's.
#[derive(Debug, thiserror::Error)]
#[error("SDL: {0}")]
pub struct SdlError(String);

/// SDL's gamepad subsystem and the pads it has open. Not `Send`: create, poll and rumble it on
/// one thread, the first in the process to touch SDL.
pub struct SdlInput {
    _sdl: sdl3::Sdl,
    subsystem: sdl3::GamepadSubsystem,
    events: sdl3::EventPump,
    /// Open pads in the same order as `HostInput::gamepads`.
    open: Vec<(JoystickId, sdl3::gamepad::Gamepad)>,
}

impl SdlInput {
    /// Sets SDL's hints (HIDAPI drivers for Switch and Switch 2 controllers, background events,
    /// no signal handlers) and opens the gamepad subsystem. Pads already connected arrive as
    /// hotplug events on the first [`Self::poll`].
    pub fn new() -> Result<Self, SdlError> {
        for (hint, value) in [
            ("SDL_NO_SIGNAL_HANDLERS", "1"),
            ("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1"),
            ("SDL_JOYSTICK_HIDAPI", "1"),
            ("SDL_JOYSTICK_HIDAPI_SWITCH", "1"),
            ("SDL_JOYSTICK_HIDAPI_SWITCH2", "1"),
            ("SDL_JOYSTICK_HIDAPI_JOY_CONS", "1"),
            ("SDL_JOYSTICK_HIDAPI_COMBINE_JOY_CONS", "1"),
            ("SDL_JOYSTICK_HIDAPI_SWITCH_PLAYER_LED", "1"),
        ] {
            sdl3::hint::set(hint, value);
        }
        let err = |e: sdl3::Error| SdlError(e.to_string());
        let sdl = sdl3::init().map_err(err)?;
        let subsystem = sdl.gamepad().map_err(err)?;
        let events = sdl.event_pump().map_err(err)?;
        Ok(Self {
            _sdl: sdl,
            subsystem,
            events,
            open: Vec::new(),
        })
    }

    /// Pumps SDL events into `host.gamepads`. Returns true when a pad connected or disconnected,
    /// so the caller recompiles its `InputMapper`.
    pub fn poll(&mut self, host: &mut HostInput) -> bool {
        let mut changed = false;
        while let Some(event) = self.events.poll_event() {
            match event {
                Event::GamepadAdded { which, .. } => changed |= self.add(which, host),
                Event::GamepadRemoved { which, .. } => {
                    if let Some(i) = self.slot(which) {
                        self.open.remove(i);
                        let pad = host.gamepads.remove(i);
                        tracing::info!("gamepad disconnected: {} #{}", pad.name, pad.index);
                        changed = true;
                    }
                }
                Event::GamepadButtonDown { which, button, .. } => {
                    self.set_button(host, which, button, true)
                }
                Event::GamepadButtonUp { which, button, .. } => {
                    self.set_button(host, which, button, false)
                }
                Event::GamepadAxisMotion {
                    which, axis, value, ..
                } => {
                    if let (Some(i), Some(axis)) = (self.slot(which), pad_axis(axis)) {
                        host.gamepads[i]
                            .state
                            .set_axis(axis, normalize_axis(axis, value));
                    }
                }
                Event::GamepadSensorUpdated {
                    which,
                    sensor,
                    data,
                    ..
                } => {
                    if let Some(i) = self.slot(which) {
                        let state = &mut host.gamepads[i].state;
                        match sensor {
                            SensorType::Gyroscope => state.gyro = Some(data),
                            SensorType::Accelerometer => state.accel = Some(data),
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        changed
    }

    /// Rumbles `host.gamepads[index]` with motor strengths 0..=65535 for `duration_ms`.
    pub fn rumble(&mut self, index: usize, low: u16, high: u16, duration_ms: u32) {
        if let Some((_, pad)) = self.open.get_mut(index)
            && let Err(e) = pad.set_rumble(low, high, duration_ms)
        {
            tracing::debug!("rumble failed: {e}");
        }
    }

    fn slot(&self, which: JoystickId) -> Option<usize> {
        self.open.iter().position(|(id, _)| *id == which)
    }

    fn set_button(&self, host: &mut HostInput, which: JoystickId, button: Button, down: bool) {
        if let (Some(i), Some(button)) = (self.slot(which), pad_button(button)) {
            host.gamepads[i].state.set_button(button, down);
        }
    }

    fn add(&mut self, which: JoystickId, host: &mut HostInput) -> bool {
        if self.slot(which).is_some() {
            return false;
        }
        let pad = match self.subsystem.open(which) {
            Ok(pad) => pad,
            Err(e) => {
                tracing::warn!("cannot open gamepad: {e}");
                return false;
            }
        };
        for sensor in [SensorType::Gyroscope, SensorType::Accelerometer] {
            let _ = pad.sensor_set_enabled(sensor, true);
        }
        let guid = self.subsystem.guid_for_id(which).string();
        let index = free_index(&host.gamepads, &guid);
        let name = pad.name().unwrap_or_else(|| "Gamepad".into());
        // SAFETY: `pad` is open; the call only reads its property set.
        let has_rumble = unsafe { pad.has_rumble() };
        tracing::info!("gamepad connected: {name} #{index} ({guid}), rumble: {has_rumble}");
        host.gamepads.push(Gamepad {
            guid,
            index,
            name,
            has_rumble,
            state: GamepadState::default(),
        });
        self.open.push((which, pad));
        true
    }
}

fn pad_button(button: Button) -> Option<PadButton> {
    PadButton::from_sdl_index(usize::try_from(button.to_ll().0).ok()?)
}

fn pad_axis(axis: Axis) -> Option<PadAxis> {
    PadAxis::from_sdl_index(usize::try_from(axis.to_ll().0).ok()?)
}

/// SDL's i16 axis value in host units: sticks -1..=1 (+y down, as SDL), triggers 0..=1.
pub fn normalize_axis(axis: PadAxis, value: i16) -> f32 {
    let v = (f32::from(value) / 32767.0).clamp(-1.0, 1.0);
    match axis {
        PadAxis::LeftTrigger | PadAxis::RightTrigger => v.max(0.0),
        _ => v,
    }
}

/// The lowest index not used by a connected pad with the same GUID, so identical controllers
/// keep stable, distinct indices.
pub fn free_index(pads: &[Gamepad], guid: &str) -> u32 {
    (0..)
        .find(|i| !pads.iter().any(|p| p.guid == guid && p.index == *i))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pad(guid: &str, index: u32) -> Gamepad {
        Gamepad {
            guid: guid.into(),
            index,
            name: "Pad".into(),
            has_rumble: false,
            state: GamepadState::default(),
        }
    }

    #[test]
    fn indices_fill_the_lowest_gap() {
        assert_eq!(free_index(&[], "a"), 0);
        assert_eq!(free_index(&[pad("a", 0), pad("b", 0)], "a"), 1);
        assert_eq!(free_index(&[pad("a", 1)], "a"), 0);
        assert_eq!(free_index(&[pad("a", 0), pad("a", 2)], "a"), 1);
    }

    #[test]
    fn axes_normalize() {
        assert_eq!(normalize_axis(PadAxis::LeftX, 32767), 1.0);
        assert_eq!(normalize_axis(PadAxis::LeftX, -32768), -1.0);
        assert_eq!(normalize_axis(PadAxis::RightY, 0), 0.0);
        assert_eq!(normalize_axis(PadAxis::LeftTrigger, -5), 0.0);
        assert!((normalize_axis(PadAxis::RightTrigger, 16384) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn sdl_enums_line_up_with_ours() {
        assert_eq!(pad_button(Button::East), Some(PadButton::East));
        assert_eq!(pad_button(Button::DPadRight), Some(PadButton::DPadRight));
        assert_eq!(pad_button(Button::Misc6), Some(PadButton::Misc6));
        assert_eq!(pad_axis(Axis::TriggerRight), Some(PadAxis::RightTrigger));
        for button in PadButton::ALL {
            assert_eq!(
                Button::from_string(button.name()).and_then(pad_button),
                Some(*button)
            );
        }
        for axis in PadAxis::ALL {
            assert_eq!(
                Axis::from_string(axis.name()).and_then(pad_axis),
                Some(*axis)
            );
        }
    }
}
