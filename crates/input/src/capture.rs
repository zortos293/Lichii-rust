//! Click-to-bind: find what the user pressed between two snapshots of the host input.

use crate::gamepad::{GamepadState, PadAxis, PadButton};
use crate::keys::{Key, MouseButton};
use crate::mapping::{HostInput, Source};

const AXIS_PRESS: f32 = 0.5;

/// The first newly pressed input, ready to store with [`Source::to_binding`], and the pad it came
/// from. With `pad` set only that gamepad counts; with `None` any gamepad does. Keys and mouse
/// buttons always count, modifiers included (Shift is a common stick modifier). A newly pressed
/// button wins over an axis that crossed half travel in the same interval.
pub fn detect_press(
    before: &HostInput,
    after: &HostInput,
    pad: Option<usize>,
) -> Option<(Source, Option<usize>)> {
    if let Some(key) = Key::all().find(|&k| after.keys.is_down(k) && !before.keys.is_down(k)) {
        return Some((Source::Key(key), None));
    }
    if let Some(button) = MouseButton::ALL
        .into_iter()
        .find(|&b| after.mouse_button(b) && !before.mouse_button(b))
    {
        return Some((Source::Mouse(button), None));
    }
    let pads = pad_pairs(before, after, pad);
    for &(i, old, new) in &pads {
        if let Some(button) = PadButton::ALL
            .iter()
            .copied()
            .find(|&b| new.button(b) && !old.button(b))
        {
            return Some((Source::Pad(button), Some(i)));
        }
    }
    for &(i, old, new) in &pads {
        for &axis in PadAxis::ALL {
            let (was, now) = (old.axis(axis), new.axis(axis));
            if now.abs() > AXIS_PRESS && was.abs() <= AXIS_PRESS {
                let sign = if is_trigger(axis) || now > 0.0 {
                    1.0
                } else {
                    -1.0
                };
                let source = Source::Axis {
                    axis,
                    threshold: AXIS_PRESS,
                    sign,
                };
                return Some((source, Some(i)));
            }
        }
    }
    None
}

/// For binding an analog stick: the stick (x and y axis) pushed furthest past half travel.
pub fn detect_stick(
    before: &HostInput,
    after: &HostInput,
    pad: Option<usize>,
) -> Option<((PadAxis, PadAxis), usize)> {
    let sticks = [
        (PadAxis::LeftX, PadAxis::LeftY),
        (PadAxis::RightX, PadAxis::RightY),
    ];
    pad_pairs(before, after, pad)
        .into_iter()
        .flat_map(|(i, old, new)| {
            sticks.into_iter().filter_map(move |(x, y)| {
                let now = new.axis(x).hypot(new.axis(y));
                let was = old.axis(x).hypot(old.axis(y));
                (now > AXIS_PRESS && was <= AXIS_PRESS).then_some(((x, y), i, now))
            })
        })
        .max_by(|a, b| a.2.total_cmp(&b.2))
        .map(|(stick, i, _)| (stick, i))
}

fn is_trigger(axis: PadAxis) -> bool {
    matches!(axis, PadAxis::LeftTrigger | PadAxis::RightTrigger)
}

/// Each gamepad present in both snapshots (matched by GUID and index), filtered to `pad`.
fn pad_pairs<'a>(
    before: &'a HostInput,
    after: &'a HostInput,
    pad: Option<usize>,
) -> Vec<(usize, &'a GamepadState, &'a GamepadState)> {
    after
        .gamepads
        .iter()
        .enumerate()
        .filter(|(i, _)| pad.is_none_or(|p| p == *i))
        .filter_map(|(i, new)| {
            let old = before
                .gamepads
                .iter()
                .find(|g| g.guid == new.guid && g.index == new.index)?;
            Some((i, &old.state, &new.state))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gamepad::Gamepad;

    fn host(pads: usize) -> HostInput {
        HostInput {
            gamepads: (0..pads)
                .map(|i| Gamepad {
                    guid: "g".into(),
                    index: i as u32,
                    name: "Pad".into(),
                    has_rumble: false,
                    state: GamepadState::default(),
                })
                .collect(),
            ..HostInput::default()
        }
    }

    #[test]
    fn nothing_pressed() {
        let a = host(1);
        assert_eq!(detect_press(&a, &a.clone(), None), None);
        assert_eq!(detect_stick(&a, &a.clone(), None), None);
    }

    #[test]
    fn keys_and_mouse() {
        let before = host(0);
        let mut after = before.clone();
        let shift = Key::from_name("ShiftLeft").unwrap();
        after.keys.set(shift, true);
        assert_eq!(
            detect_press(&before, &after, None),
            Some((Source::Key(shift), None))
        );
        let mut after = before.clone();
        after.set_mouse_button(MouseButton::Right, true);
        assert_eq!(
            detect_press(&before, &after, None),
            Some((Source::Mouse(MouseButton::Right), None))
        );
    }

    #[test]
    fn held_keys_are_not_new_presses() {
        let mut before = host(0);
        before.keys.set(Key::from_name("KeyA").unwrap(), true);
        assert_eq!(detect_press(&before, &before.clone(), None), None);
    }

    #[test]
    fn pad_button_beats_axis() {
        let before = host(2);
        let mut after = before.clone();
        after.gamepads[1].state.set_axis(PadAxis::LeftX, 0.9);
        after.gamepads[1].state.set_button(PadButton::North, true);
        assert_eq!(
            detect_press(&before, &after, None),
            Some((Source::Pad(PadButton::North), Some(1)))
        );
    }

    #[test]
    fn axis_direction_and_triggers() {
        let before = host(1);
        let mut after = before.clone();
        after.gamepads[0].state.set_axis(PadAxis::RightY, -0.8);
        assert_eq!(
            detect_press(&before, &after, None),
            Some((
                Source::Axis {
                    axis: PadAxis::RightY,
                    threshold: 0.5,
                    sign: -1.0
                },
                Some(0)
            ))
        );
        let mut after = before.clone();
        after.gamepads[0].state.set_axis(PadAxis::LeftTrigger, 1.0);
        assert!(matches!(
            detect_press(&before, &after, None),
            Some((
                Source::Axis {
                    axis: PadAxis::LeftTrigger,
                    sign: 1.0,
                    ..
                },
                _
            ))
        ));
    }

    #[test]
    fn pad_filter() {
        let before = host(2);
        let mut after = before.clone();
        after.gamepads[0].state.set_button(PadButton::South, true);
        assert_eq!(detect_press(&before, &after, Some(1)), None);
        assert!(detect_press(&before, &after, Some(0)).is_some());
    }

    #[test]
    fn sticks() {
        let before = host(1);
        let mut after = before.clone();
        after.gamepads[0].state.set_axis(PadAxis::LeftX, 0.6);
        after.gamepads[0].state.set_axis(PadAxis::RightY, 0.9);
        assert_eq!(
            detect_stick(&before, &after, None),
            Some(((PadAxis::RightX, PadAxis::RightY), 0))
        );
    }
}
