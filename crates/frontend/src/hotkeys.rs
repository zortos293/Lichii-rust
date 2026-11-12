//! Keyboard chords and controller combos from `UiSettings::hotkeys`.

use std::collections::BTreeMap;

use lichii_common::settings::{Hotkey, HotkeyAction, NpadButton};
use lichii_input::KeyChord;

/// Chord lookup for the window's key events.
#[derive(Clone, Debug, Default)]
pub struct HotkeyMap {
    chords: Vec<(KeyChord, HotkeyAction)>,
}

impl HotkeyMap {
    pub fn new(hotkeys: &BTreeMap<HotkeyAction, Hotkey>) -> Self {
        let chords = hotkeys
            .iter()
            .filter_map(|(&action, hotkey)| {
                let chord = KeyChord::parse(&hotkey.keys);
                if chord.is_none() && !hotkey.keys.is_empty() {
                    tracing::warn!("hotkey {:?}: cannot parse {:?}", action, hotkey.keys);
                }
                chord.map(|c| (c, action))
            })
            .collect();
        Self { chords }
    }

    pub fn action(&self, chord: KeyChord) -> Option<HotkeyAction> {
        self.chords
            .iter()
            .find(|(c, _)| *c == chord)
            .map(|&(_, action)| action)
    }
}

/// Controller combos, for the input thread.
pub fn combos(hotkeys: &BTreeMap<HotkeyAction, Hotkey>) -> Vec<(HotkeyAction, Vec<NpadButton>)> {
    hotkeys
        .iter()
        .filter(|(_, h)| !h.controller.is_empty())
        .map(|(&a, h)| (a, h.controller.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::settings::UiSettings;
    use lichii_input::{Key, KeySet};

    #[test]
    fn default_chords_resolve() {
        let map = HotkeyMap::new(&UiSettings::default().hotkeys);
        let mut held = KeySet::default();
        let f11 = KeyChord::from_press(Key::from_name("F11").unwrap(), &held).unwrap();
        assert_eq!(map.action(f11), Some(HotkeyAction::Fullscreen));
        held.set(Key::from_name("ControlLeft").unwrap(), true);
        let ctrl_v = KeyChord::from_press(Key::from_name("KeyV").unwrap(), &held).unwrap();
        assert_eq!(map.action(ctrl_v), Some(HotkeyAction::ToggleFrameView));
        let plain_v = KeyChord::parse("KeyV").unwrap();
        assert_eq!(map.action(plain_v), None);
    }

    #[test]
    fn combos_skip_unbound() {
        let list = combos(&UiSettings::default().hotkeys);
        assert!(list.iter().all(|(_, c)| !c.is_empty()));
        assert!(list.contains(&(
            HotkeyAction::ContinuePause,
            vec![NpadButton::Home, NpadButton::Plus]
        )));
    }
}
