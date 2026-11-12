//! The hotkeys page: a keyboard chord and a controller combo per action.

use std::collections::BTreeMap;

use egui::{Color32, RichText, Ui};
use lichii_common::settings::{Hotkey, HotkeyAction, NpadButton};
use lichii_input::KeyChord;

pub fn ui(
    ui: &mut Ui,
    hotkeys: &mut BTreeMap<HotkeyAction, Hotkey>,
    capture: &mut Option<HotkeyAction>,
) {
    super::section(ui, "Hotkeys");
    super::note(
        ui,
        "Click a key binding and press the new chord. Controller combos fire while every listed button is held on player 1.",
    );
    egui::Grid::new("hotkeys")
        .num_columns(4)
        .striped(true)
        .spacing([16.0, 6.0])
        .show(ui, |ui| {
            for heading in ["Action", "Keyboard", "", "Controller"] {
                ui.label(RichText::new(heading).strong());
            }
            ui.end_row();
            for action in HotkeyAction::ALL {
                let hotkey = hotkeys.entry(action).or_default();
                ui.label(action.label());
                let text = if *capture == Some(action) {
                    "Press a key…".to_string()
                } else {
                    KeyChord::parse(&hotkey.keys).map_or_else(|| "—".into(), KeyChord::label)
                };
                if ui
                    .add_sized([160.0, 20.0], egui::Button::new(text))
                    .clicked()
                {
                    *capture = Some(action);
                }
                if ui.small_button("Clear").clicked() {
                    hotkey.keys.clear();
                    if *capture == Some(action) {
                        *capture = None;
                    }
                }
                combo_editor(ui, action, &mut hotkey.controller);
                ui.end_row();
            }
        });
    ui.add_space(8.0);
    for (a, b) in conflicts(hotkeys) {
        ui.label(
            RichText::new(format!(
                "\"{}\" and \"{}\" share a key; only \"{}\" fires.",
                a.label(),
                b.label(),
                a.label()
            ))
            .color(Color32::from_rgb(230, 170, 60)),
        );
    }
    if ui.button("Restore Defaults").clicked() {
        *hotkeys = HotkeyAction::defaults().collect();
        *capture = None;
    }
}

fn combo_editor(ui: &mut Ui, action: HotkeyAction, combo: &mut Vec<NpadButton>) {
    let text = if combo.is_empty() {
        "—".to_string()
    } else {
        combo
            .iter()
            .map(|b| b.label())
            .collect::<Vec<_>>()
            .join(" + ")
    };
    ui.menu_button(text, |ui| {
        ui.push_id(action, |ui| {
            for button in NpadButton::ALL {
                let mut held = combo.contains(&button);
                if ui.checkbox(&mut held, button.label()).changed() {
                    if held {
                        combo.push(button);
                    } else {
                        combo.retain(|b| *b != button);
                    }
                }
            }
            if ui.button("Clear").clicked() {
                combo.clear();
            }
        });
    });
}

/// Pairs of actions bound to the same chord; only the first of each pair fires.
pub fn conflicts(hotkeys: &BTreeMap<HotkeyAction, Hotkey>) -> Vec<(HotkeyAction, HotkeyAction)> {
    let parsed: Vec<_> = hotkeys
        .iter()
        .filter_map(|(&a, h)| KeyChord::parse(&h.keys).map(|c| (a, c)))
        .collect();
    let mut out = Vec::new();
    for (i, (a, ca)) in parsed.iter().enumerate() {
        for (b, cb) in &parsed[i + 1..] {
            if ca == cb {
                out.push((*a, *b));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::settings::UiSettings;

    #[test]
    fn defaults_have_no_conflicts() {
        assert!(conflicts(&UiSettings::default().hotkeys).is_empty());
        let mut hotkeys = UiSettings::default().hotkeys;
        hotkeys.get_mut(&HotkeyAction::Stop).unwrap().keys = "F4".into();
        assert_eq!(
            conflicts(&hotkeys),
            [(HotkeyAction::ContinuePause, HotkeyAction::Stop)]
        );
    }
}
