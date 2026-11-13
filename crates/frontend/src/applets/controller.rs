//! The controller applet, after Lichii's QtControllerSelectorDialog: the players the title
//! supports, each with a controller type the title accepts, set in the controls settings.

use crossbeam_channel::Sender;
use egui::{Color32, Key, RichText};
use lichii_common::settings::{ControlSettings, ControllerType, MAX_PLAYERS, PlayerInput};
use lichii_services::applets::{ControllerAnswer, ControllerRequest, ControllerSetup};

const HANDHELD_NPAD: u32 = 0x20;
/// `ControllerSupportMode` 2; 3 is key remapping.
const FIRMWARE_UPDATE: u8 = 2;
const HANDHELD_STYLE: u32 = 1 << 1;

/// The `NpadStyleSet` bit hid reports for `controller`.
fn style(controller: ControllerType) -> u32 {
    match controller {
        ControllerType::ProController | ControllerType::Switch2ProController => 1 << 0,
        ControllerType::Handheld | ControllerType::Switch2Handheld => HANDHELD_STYLE,
        ControllerType::DualJoycons | ControllerType::Switch2DualJoycons => 1 << 2,
        ControllerType::LeftJoycon | ControllerType::Switch2LeftJoycon => 1 << 3,
        ControllerType::RightJoycon | ControllerType::Switch2RightJoycon => 1 << 4,
        ControllerType::GameCube => 1 << 5,
    }
}

pub(super) struct ControllerDialog {
    request: ControllerRequest,
    pub reply: Sender<ControllerAnswer>,
    original: ControlSettings,
    controls: ControlSettings,
    /// Confirmed without asking: a firmware update or key remapping, which this frontend does
    /// not offer, or a single-player request the setup already meets.
    confirm_on_open: bool,
}

impl ControllerDialog {
    pub fn new(
        request: ControllerRequest,
        reply: Sender<ControllerAnswer>,
        controls: &ControlSettings,
    ) -> Self {
        let mut dialog = Self {
            request,
            reply,
            original: controls.clone(),
            controls: controls.clone(),
            confirm_on_open: false,
        };
        dialog.confirm_on_open =
            request.mode >= FIRMWARE_UPDATE || (request.single_mode && dialog.valid());
        if dialog.confirm_on_open {
            return dialog;
        }
        let first = ControllerType::ALL
            .into_iter()
            .find(|&controller| dialog.accepts(controller));
        let shown = dialog.shown();
        for (index, player) in dialog.controls.players.iter_mut().enumerate() {
            if index >= shown || !request.keep_connected {
                player.connected = false;
            }
            if player.connected
                && let Some(first) = first
                && !accepts(&request, player.controller_type)
            {
                player.controller_type = first;
            }
        }
        dialog
    }

    fn accepts(&self, controller: ControllerType) -> bool {
        accepts(&self.request, controller)
    }

    /// Players the title can use: one in single-player mode, else up to its maximum.
    fn shown(&self) -> usize {
        let max = if self.request.single_mode {
            1
        } else {
            self.request.max_players.clamp(1, MAX_PLAYERS as i8) as usize
        };
        max.min(self.controls.players.len())
    }

    fn min_players(&self) -> usize {
        if self.request.single_mode {
            1
        } else {
            (self.request.min_players.max(0) as usize).min(self.shown())
        }
    }

    fn connected(&self) -> impl Iterator<Item = (usize, ControllerType)> + '_ {
        self.controls.players[..self.shown()]
            .iter()
            .enumerate()
            .filter(|(_, player)| player.connected)
            .map(|(index, player)| (index, player.controller_type))
    }

    /// Whether the setup has a player count the title supports, every player on a controller
    /// it accepts.
    fn valid(&self) -> bool {
        let count = self.setup().player_count.max(0) as usize;
        (self.min_players()..=self.shown()).contains(&count)
            && self
                .connected()
                .all(|(_, controller)| self.accepts(controller))
    }

    /// What the title sees after the applet. hid's own count knows how the title split Joy-Con
    /// pairs, so it is kept and moved by the players the user connected or disconnected. In
    /// single-player mode am reports one player whatever hid shows, so only the player the
    /// dialog shows counts.
    fn setup(&self) -> ControllerSetup {
        if self.request.single_mode {
            return host_setup(&self.controls.players[..self.shown()]);
        }
        let (before, after) = (
            host_setup(&self.original.players),
            host_setup(&self.controls.players),
        );
        if after.player_count == 0 {
            return after;
        }
        let current = self.request.current;
        ControllerSetup {
            player_count: (current.player_count + after.player_count - before.player_count)
                .max(after.player_count),
            npad_id: if before.player_count > 0 && before.npad_id == after.npad_id {
                current.npad_id
            } else {
                after.npad_id
            },
        }
    }

    /// The answer once the user decided, and the controls when they changed.
    pub fn show(
        &mut self,
        ctx: &egui::Context,
    ) -> Option<(ControllerAnswer, Option<ControlSettings>)> {
        if self.request.mode >= FIRMWARE_UPDATE {
            return Some((ControllerAnswer::Confirmed(self.request.current), None));
        }
        if self.confirm_on_open {
            return Some((ControllerAnswer::Confirmed(self.setup()), None));
        }
        let (enter, escape) =
            ctx.input(|i| (i.key_pressed(Key::Enter), i.key_pressed(Key::Escape)));
        let (shown, min) = (self.shown(), self.min_players());
        let accepted: Vec<_> = ControllerType::ALL
            .into_iter()
            .filter(|&controller| self.accepts(controller))
            .collect();
        let mut clicked = None;
        egui::Modal::new(egui::Id::unique("controller applet")).show(ctx, |ui| {
            ui.set_width(420.0);
            ui.heading("Controller Applet");
            ui.label(if min == shown {
                format!("Players: {shown}")
            } else {
                format!("Players: {min} - {shown}")
            });
            ui.add_space(6.0);
            for (index, player) in self.controls.players[..shown].iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut player.connected, format!("Player {}", index + 1));
                    egui::ComboBox::from_id_salt(("applet controller", index))
                        .selected_text(player.controller_type.label())
                        .show_ui(ui, |ui| {
                            for &controller in &accepted {
                                ui.selectable_value(
                                    &mut player.controller_type,
                                    controller,
                                    controller.label(),
                                );
                            }
                        });
                });
            }
            let valid = self.valid();
            if !valid {
                ui.label(
                    RichText::new(
                        "Connect a supported number of players, each on a controller this title \
                         accepts.",
                    )
                    .color(Color32::from_rgb(220, 120, 80)),
                );
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.button("Cancel").clicked() {
                    clicked = Some(false);
                }
                if ui.add_enabled(valid, egui::Button::new("OK")).clicked() {
                    clicked = Some(true);
                }
            });
        });
        let ok = clicked
            .or((enter && self.valid()).then_some(true))
            .or(escape.then_some(false))?;
        if !ok {
            return Some((ControllerAnswer::Cancelled, None));
        }
        let changed = (self.controls != self.original).then(|| self.controls.clone());
        Some((ControllerAnswer::Confirmed(self.setup()), changed))
    }
}

/// The npads `players` connect: how many players, and the first one (`0x20` for handheld).
fn host_setup(players: &[PlayerInput]) -> ControllerSetup {
    let mut connected = players
        .iter()
        .enumerate()
        .filter(|(_, player)| player.connected);
    let npad_id = connected.clone().next().map_or(0, |(index, player)| {
        if style(player.controller_type) == HANDHELD_STYLE {
            HANDHELD_NPAD
        } else {
            index as u32
        }
    });
    ControllerSetup {
        player_count: connected.by_ref().count() as i8,
        npad_id,
    }
}

/// Whether the title takes `controller`: its style is in the request's style set (any style
/// when the set is empty), and handheld only in single-player mode, as in Lichii.
fn accepts(request: &ControllerRequest, controller: ControllerType) -> bool {
    let style = style(controller);
    (request.style_set == 0 || request.style_set & style != 0)
        && (style != HANDHELD_STYLE || request.single_mode)
}

#[cfg(test)]
mod tests {
    use std::thread::JoinHandle;

    use super::super::testing::Harness;
    use super::*;

    const FULL_KEY: u32 = 1 << 0;
    const JOY_DUAL: u32 = 1 << 2;

    fn configure(ui: &Harness, request: ControllerRequest) -> JoinHandle<ControllerAnswer> {
        ui.ask(move |frontend| frontend.configure_controllers(&request))
    }

    fn request(min: i8, max: i8, style_set: u32) -> ControllerRequest {
        ControllerRequest {
            min_players: min,
            max_players: max,
            style_set,
            keep_connected: true,
            ..ControllerRequest::default()
        }
    }

    fn harness(players: &[(bool, ControllerType)]) -> Harness {
        let mut ui = Harness::new();
        for (player, &(connected, controller)) in ui.controls.players.iter_mut().zip(players) {
            player.connected = connected;
            player.controller_type = controller;
        }
        ui
    }

    #[test]
    fn enter_confirms_a_setup_the_title_supports() {
        let mut ui = harness(&[
            (true, ControllerType::ProController),
            (true, ControllerType::ProController),
        ]);
        let answer = configure(
            &ui,
            ControllerRequest {
                current: ControllerSetup {
                    player_count: 2,
                    npad_id: 0,
                },
                ..request(1, 4, FULL_KEY | JOY_DUAL)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            ControllerAnswer::Confirmed(ControllerSetup {
                player_count: 2,
                npad_id: 0
            })
        );
        assert_eq!(ui.applied, None, "nothing changed");
    }

    #[test]
    fn players_past_the_maximum_are_disconnected() {
        let mut ui = harness(&[
            (true, ControllerType::ProController),
            (true, ControllerType::ProController),
            (true, ControllerType::ProController),
        ]);
        let answer = configure(&ui, request(1, 2, FULL_KEY));
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            ControllerAnswer::Confirmed(ControllerSetup {
                player_count: 2,
                npad_id: 0
            })
        );
        let applied = ui.applied.expect("player 3 was disconnected");
        let connected: Vec<_> = applied.players.iter().map(|p| p.connected).collect();
        assert_eq!(connected[..3], [true, true, false]);
    }

    #[test]
    fn controllers_the_title_rejects_switch_to_one_it_accepts() {
        let mut ui = harness(&[(true, ControllerType::DualJoycons)]);
        let answer = configure(&ui, request(1, 1, FULL_KEY));
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            ControllerAnswer::Confirmed(ControllerSetup {
                player_count: 1,
                npad_id: 0
            })
        );
        let applied = ui.applied.expect("the controller type changed");
        assert_eq!(
            applied.players[0].controller_type,
            ControllerType::ProController
        );
    }

    #[test]
    fn without_keep_connected_players_start_disconnected() {
        let mut ui = harness(&[(true, ControllerType::ProController)]);
        let answer = configure(
            &ui,
            ControllerRequest {
                keep_connected: false,
                current: ControllerSetup {
                    player_count: 1,
                    npad_id: 0,
                },
                ..request(1, 2, FULL_KEY)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert!(ui.dialogs.is_open(), "no player is connected");
        ui.key(Key::Escape);
        assert_eq!(answer.join().unwrap(), ControllerAnswer::Cancelled);
        assert_eq!(ui.applied, None, "cancel keeps the controls");
    }

    #[test]
    fn a_single_player_setup_that_already_fits_is_confirmed_at_once() {
        let mut ui = harness(&[(true, ControllerType::Handheld)]);
        let handheld = ControllerSetup {
            player_count: 1,
            npad_id: HANDHELD_NPAD,
        };
        let answer = configure(
            &ui,
            ControllerRequest {
                single_mode: true,
                keep_connected: false,
                current: handheld,
                ..request(1, 4, FULL_KEY | HANDHELD_STYLE)
            },
        );
        ui.frame(Vec::new());
        assert_eq!(
            answer.join().unwrap(),
            ControllerAnswer::Confirmed(handheld)
        );
        assert!(ui.applied.is_none());
    }

    #[test]
    fn firmware_update_and_key_remapping_keep_the_setup() {
        for mode in [2, 3] {
            let mut ui = harness(&[(true, ControllerType::ProController)]);
            let current = ControllerSetup {
                player_count: 1,
                npad_id: 0,
            };
            let answer = configure(
                &ui,
                ControllerRequest {
                    mode,
                    keep_connected: false,
                    current,
                    ..request(0, 0, 0)
                },
            );
            ui.frame(Vec::new());
            assert_eq!(answer.join().unwrap(), ControllerAnswer::Confirmed(current));
            assert!(
                ui.applied.is_none(),
                "mode {mode} leaves the controls alone"
            );
        }
    }

    #[test]
    fn an_unchanged_setup_reports_what_hid_shows() {
        let mut ui = harness(&[(true, ControllerType::DualJoycons)]);
        let split = ControllerSetup {
            player_count: 2,
            npad_id: 0,
        };
        let answer = configure(
            &ui,
            ControllerRequest {
                current: split,
                ..request(2, 4, JOY_DUAL | (1 << 3) | (1 << 4))
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            ControllerAnswer::Confirmed(split),
            "the title split the pair into two npads"
        );
    }

    #[test]
    fn a_changed_setup_is_answered_once_input_applied_it() {
        let mut ui = harness(&[(true, ControllerType::DualJoycons)]);
        ui.apply_at_once = false;
        let answer = configure(&ui, request(1, 1, FULL_KEY));
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        let change = ui.pending.take().expect("the type changed");
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(!answer.is_finished(), "the title waits for input");
        (change.applied)();
        assert_eq!(
            answer.join().unwrap(),
            ControllerAnswer::Confirmed(ControllerSetup {
                player_count: 1,
                npad_id: 0
            })
        );
    }

    #[test]
    fn single_player_mode_opens_the_dialog_when_no_player_is_connected() {
        let mut ui = harness(&[(false, ControllerType::ProController)]);
        let answer = configure(
            &ui,
            ControllerRequest {
                single_mode: true,
                current: ControllerSetup {
                    player_count: 1,
                    npad_id: 0,
                },
                ..request(1, 4, FULL_KEY)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert!(
            ui.dialogs.is_open(),
            "am's forced count of one is not a player"
        );
        ui.key(Key::Escape);
        assert_eq!(answer.join().unwrap(), ControllerAnswer::Cancelled);
    }

    #[test]
    fn single_player_mode_counts_only_the_player_it_shows() {
        let mut ui = harness(&[
            (true, ControllerType::ProController),
            (true, ControllerType::ProController),
        ]);
        let one = ControllerSetup {
            player_count: 1,
            npad_id: 0,
        };
        let answer = configure(
            &ui,
            ControllerRequest {
                single_mode: true,
                current: one,
                ..request(1, 4, FULL_KEY)
            },
        );
        ui.frame(Vec::new());
        assert_eq!(answer.join().unwrap(), ControllerAnswer::Confirmed(one));
        assert!(ui.applied.is_none(), "player 2 stays connected in settings");
    }

    #[test]
    fn an_edited_setup_keeps_the_npads_hid_split() {
        let mut ui = harness(&[
            (true, ControllerType::DualJoycons),
            (false, ControllerType::ProController),
            (true, ControllerType::ProController),
        ]);
        let answer = configure(
            &ui,
            ControllerRequest {
                current: ControllerSetup {
                    player_count: 3,
                    npad_id: 0,
                },
                ..request(1, 2, FULL_KEY | JOY_DUAL)
            },
        );
        ui.frame(Vec::new());
        ui.key(Key::Enter);
        assert_eq!(
            answer.join().unwrap(),
            ControllerAnswer::Confirmed(ControllerSetup {
                player_count: 2,
                npad_id: 0
            }),
            "the split pair stays two npads once player 3 is disconnected"
        );
    }

    #[test]
    fn handheld_is_accepted_in_single_player_mode_only() {
        let multi = request(1, 4, HANDHELD_STYLE | FULL_KEY);
        assert!(!accepts(&multi, ControllerType::Handheld));
        assert!(accepts(&multi, ControllerType::Switch2ProController));
        let single = ControllerRequest {
            single_mode: true,
            ..multi
        };
        assert!(accepts(&single, ControllerType::Switch2Handheld));
        assert!(!accepts(&single, ControllerType::GameCube));
        assert!(accepts(&request(1, 4, 0), ControllerType::GameCube));
    }
}
