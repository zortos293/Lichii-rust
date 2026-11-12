//! Keyboard and mouse identifiers. Keys are named by their `winit` `KeyCode` variant, the same
//! names `InputBinding::Key` and hotkey chords store.

use winit::keyboard::KeyCode;

macro_rules! key_table {
    ($($name:ident),* $(,)?) => {
        const NAMES: &[&str] = &[$(stringify!($name)),*];

        fn index_of(code: KeyCode) -> Option<u16> {
            let mut index = 0u16;
            $(
                if code == KeyCode::$name {
                    return Some(index);
                }
                index += 1;
            )*
            let _ = index;
            None
        }
    };
}

key_table! {
    KeyA, KeyB, KeyC, KeyD, KeyE, KeyF, KeyG, KeyH, KeyI, KeyJ, KeyK, KeyL, KeyM, KeyN, KeyO,
    KeyP, KeyQ, KeyR, KeyS, KeyT, KeyU, KeyV, KeyW, KeyX, KeyY, KeyZ,
    Digit0, Digit1, Digit2, Digit3, Digit4, Digit5, Digit6, Digit7, Digit8, Digit9,
    Backquote, Backslash, BracketLeft, BracketRight, Comma, Equal, IntlBackslash, IntlRo,
    IntlYen, Minus, Period, Quote, Semicolon, Slash,
    AltLeft, AltRight, Backspace, CapsLock, ContextMenu, ControlLeft, ControlRight, Enter,
    SuperLeft, SuperRight, ShiftLeft, ShiftRight, Space, Tab,
    Delete, End, Home, Insert, PageDown, PageUp,
    ArrowDown, ArrowLeft, ArrowRight, ArrowUp,
    NumLock, Numpad0, Numpad1, Numpad2, Numpad3, Numpad4, Numpad5, Numpad6, Numpad7, Numpad8,
    Numpad9, NumpadAdd, NumpadDecimal, NumpadDivide, NumpadEnter, NumpadEqual, NumpadMultiply,
    NumpadSubtract,
    Escape, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12, F13, F14, F15, F16, F17, F18,
    F19, F20, F21, F22, F23, F24, PrintScreen, ScrollLock, Pause,
}

/// A keyboard key Lichii can bind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key(u16);

impl Key {
    pub fn from_winit(code: KeyCode) -> Option<Self> {
        index_of(code).map(Self)
    }

    pub fn from_name(name: &str) -> Option<Self> {
        NAMES.iter().position(|n| *n == name).map(|i| Self(i as u16))
    }

    /// The `KeyCode` name stored in settings, e.g. `KeyZ`.
    pub fn name(self) -> &'static str {
        NAMES[self.0 as usize]
    }

    /// A short name for the UI, e.g. `Z`, `1`, `Up`.
    pub fn label(self) -> &'static str {
        let name = self.name();
        name.strip_prefix("Key")
            .or_else(|| name.strip_prefix("Digit"))
            .or_else(|| name.strip_prefix("Arrow"))
            .unwrap_or(name)
    }

    pub fn all() -> impl Iterator<Item = Self> {
        (0..NAMES.len() as u16).map(Self)
    }

    fn is_modifier(self) -> bool {
        matches!(
            self.name(),
            "ControlLeft"
                | "ControlRight"
                | "ShiftLeft"
                | "ShiftRight"
                | "AltLeft"
                | "AltRight"
                | "SuperLeft"
                | "SuperRight"
        )
    }
}

/// Which keys are held, one bit per [`Key`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeySet([u64; 4]);

impl KeySet {
    pub fn set(&mut self, key: Key, down: bool) {
        let (word, bit) = (key.0 as usize / 64, key.0 % 64);
        if down {
            self.0[word] |= 1 << bit;
        } else {
            self.0[word] &= !(1 << bit);
        }
    }

    pub fn is_down(&self, key: Key) -> bool {
        self.0[key.0 as usize / 64] & (1 << (key.0 % 64)) != 0
    }

    pub fn clear(&mut self) {
        self.0 = [0; 4];
    }

    fn any_down(&self, names: &[&str]) -> bool {
        names
            .iter()
            .filter_map(|n| Key::from_name(n))
            .any(|k| self.is_down(k))
    }

    pub fn modifiers(&self) -> Modifiers {
        Modifiers {
            ctrl: self.any_down(&["ControlLeft", "ControlRight"]),
            shift: self.any_down(&["ShiftLeft", "ShiftRight"]),
            alt: self.any_down(&["AltLeft", "AltRight"]),
            logo: self.any_down(&["SuperLeft", "SuperRight"]),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Modifiers {
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
    pub logo: bool,
}

/// A hotkey chord: modifiers plus one key, written `Ctrl+Shift+F11`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyChord {
    pub modifiers: Modifiers,
    pub key: Key,
}

impl KeyChord {
    /// Parses `Ctrl+Alt+KeyX`. Empty or unknown text yields `None`.
    pub fn parse(text: &str) -> Option<Self> {
        let mut modifiers = Modifiers::default();
        let mut parts = text.split('+').map(str::trim).peekable();
        while let Some(part) = parts.next() {
            if parts.peek().is_none() {
                return Key::from_name(part).map(|key| Self { modifiers, key });
            }
            match part {
                "Ctrl" => modifiers.ctrl = true,
                "Shift" => modifiers.shift = true,
                "Alt" => modifiers.alt = true,
                "Super" => modifiers.logo = true,
                _ => return None,
            }
        }
        None
    }

    /// The chord as stored in settings.
    pub fn to_settings_string(self) -> String {
        self.format(Key::name)
    }

    /// The chord for display, e.g. `Ctrl+O`.
    pub fn label(self) -> String {
        self.format(Key::label)
    }

    fn format(self, key_name: fn(Key) -> &'static str) -> String {
        let m = self.modifiers;
        let mut out = String::new();
        for (on, name) in [
            (m.ctrl, "Ctrl+"),
            (m.shift, "Shift+"),
            (m.alt, "Alt+"),
            (m.logo, "Super+"),
        ] {
            if on {
                out.push_str(name);
            }
        }
        out.push_str(key_name(self.key));
        out
    }

    /// The chord a key press forms with the held modifiers; `None` for a bare modifier press.
    pub fn from_press(key: Key, held: &KeySet) -> Option<Self> {
        (!key.is_modifier()).then(|| Self {
            modifiers: held.modifiers(),
            key,
        })
    }
}

/// Mouse buttons by the names `InputBinding::MouseButton` stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}

impl MouseButton {
    pub const ALL: [Self; 5] = [
        Self::Left,
        Self::Right,
        Self::Middle,
        Self::Back,
        Self::Forward,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Middle => "Middle",
            Self::Back => "Back",
            Self::Forward => "Forward",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.name() == name)
    }

    pub fn from_winit(button: winit::event::MouseButton) -> Option<Self> {
        use winit::event::MouseButton as W;
        match button {
            W::Left => Some(Self::Left),
            W::Right => Some(Self::Right),
            W::Middle => Some(Self::Middle),
            W::Back => Some(Self::Back),
            W::Forward => Some(Self::Forward),
            W::Other(_) => None,
        }
    }

    pub(crate) fn bit(self) -> u8 {
        1 << self as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_names_round_trip() {
        for key in Key::all() {
            assert_eq!(Key::from_name(key.name()), Some(key));
        }
        assert_eq!(Key::from_winit(KeyCode::KeyZ), Key::from_name("KeyZ"));
        assert_eq!(Key::from_winit(KeyCode::Pause).unwrap().name(), "Pause");
        assert_eq!(Key::from_name("ArrowUp").unwrap().label(), "Up");
        assert!(Key::from_name("NotAKey").is_none());
    }

    #[test]
    fn key_set_tracks_presses() {
        let mut keys = KeySet::default();
        let f11 = Key::from_name("F11").unwrap();
        keys.set(f11, true);
        assert!(keys.is_down(f11));
        keys.set(f11, false);
        assert!(!keys.is_down(f11));
    }

    #[test]
    fn chords_parse_and_format() {
        let chord = KeyChord::parse("Ctrl+Shift+KeyO").unwrap();
        assert!(chord.modifiers.ctrl && chord.modifiers.shift && !chord.modifiers.alt);
        assert_eq!(chord.to_settings_string(), "Ctrl+Shift+KeyO");
        assert_eq!(chord.label(), "Ctrl+Shift+O");
        assert_eq!(KeyChord::parse("F11").unwrap().key.name(), "F11");
        assert!(KeyChord::parse("").is_none());
        assert!(KeyChord::parse("Hyper+KeyA").is_none());
    }

    #[test]
    fn chord_from_press_uses_held_modifiers() {
        let mut held = KeySet::default();
        held.set(Key::from_name("ControlRight").unwrap(), true);
        let comma = Key::from_name("Comma").unwrap();
        assert_eq!(
            KeyChord::from_press(comma, &held),
            KeyChord::parse("Ctrl+Comma")
        );
        assert_eq!(
            KeyChord::from_press(Key::from_name("ShiftLeft").unwrap(), &held),
            None
        );
    }
}
