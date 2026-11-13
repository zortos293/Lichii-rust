//! The main window's egui layout: menu bar, game list or guest area, status bar and dialogs.

use std::time::Instant;

use egui::{Align, Color32, Layout, RichText, Ui};
use lichii_common::settings::HotkeyAction;
use lichii_vulkan::present::Viewport;

use crate::app::{Action, App};
use crate::game_list::format_size;

impl App {
    pub(crate) fn draw(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let immersive = self.is_fullscreen() && self.session.is_some();
        if !immersive {
            egui::Panel::top("menu").show(ui, |ui| {
                egui::MenuBar::new().ui(ui, |ui| self.menus(ui, actions));
            });
            if self.settings.ui.show_status_bar {
                egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui));
            }
        }
        if self.session.is_some() {
            egui::CentralPanel::no_frame().show(ui, |ui| self.guest_area(ui));
        } else {
            egui::CentralPanel::default().show(ui, |ui| self.game_list(ui, actions));
        }
        self.dialogs(ui.ctx(), actions);
        self.show_config(ui.ctx());
        if let Some((_, at)) = &self.ui.toast {
            let left = crate::app::TOAST_TIME.saturating_sub(at.elapsed());
            if left.is_zero() {
                self.ui.toast = None;
            } else {
                ui.ctx().request_repaint_after(left);
            }
        }
    }

    fn menu_item(&self, ui: &mut Ui, label: &str, action: HotkeyAction, enabled: bool) -> bool {
        ui.add_enabled(
            enabled,
            egui::Button::new(label).shortcut_text(self.shortcut(action)),
        )
        .clicked()
    }

    fn menus(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let running = self.session.is_some();
        ui.menu_button("File", |ui| {
            if self.menu_item(ui, "Load File…", HotkeyAction::LoadFile, true) {
                actions.push(Action::Hotkey(HotkeyAction::LoadFile));
            }
            if ui.button("Add Game Directory…").clicked() {
                actions.push(Action::AddGameDir);
            }
            if ui.button("Refresh Game List").clicked() {
                actions.push(Action::RefreshGames);
            }
            ui.menu_button("Recent Files", |ui| {
                if self.settings.ui.recent_files.is_empty() {
                    ui.label(RichText::new("None").weak());
                }
                for path in &self.settings.ui.recent_files {
                    if ui.button(path.display().to_string()).clicked() {
                        actions.push(Action::Boot(path.clone()));
                    }
                }
                ui.separator();
                if ui.button("Clear Recent Files").clicked() {
                    actions.push(Action::ClearRecent);
                }
            });
            ui.separator();
            if self.menu_item(ui, "Exit", HotkeyAction::Exit, true) {
                actions.push(Action::Hotkey(HotkeyAction::Exit));
            }
        });
        ui.menu_button("Emulation", |ui| {
            let paused = self.session.as_ref().is_some_and(|s| s.is_paused());
            let label = if paused { "Continue" } else { "Pause" };
            if self.menu_item(ui, label, HotkeyAction::ContinuePause, running) {
                actions.push(Action::Hotkey(HotkeyAction::ContinuePause));
            }
            if self.menu_item(ui, "Stop", HotkeyAction::Stop, running) {
                actions.push(Action::Hotkey(HotkeyAction::Stop));
            }
            if self.menu_item(ui, "Restart", HotkeyAction::Restart, running) {
                actions.push(Action::Hotkey(HotkeyAction::Restart));
            }
            ui.separator();
            if self.menu_item(ui, "Configure…", HotkeyAction::Configure, true) {
                actions.push(Action::Hotkey(HotkeyAction::Configure));
            }
        });
        ui.menu_button("View", |ui| {
            let toggles = [
                ("Fullscreen", self.is_fullscreen(), HotkeyAction::Fullscreen),
                (
                    "Frame View",
                    self.settings.frame_view.enabled,
                    HotkeyAction::ToggleFrameView,
                ),
                (
                    "Status Bar",
                    self.settings.ui.show_status_bar,
                    HotkeyAction::ToggleStatusBar,
                ),
                (
                    "Docked Mode",
                    self.settings.system.docked,
                    HotkeyAction::ToggleDocked,
                ),
            ];
            for (label, on, action) in toggles {
                let text = if on {
                    format!("✔ {label}")
                } else {
                    format!("   {label}")
                };
                if self.menu_item(ui, &text, action, true) {
                    actions.push(Action::Hotkey(action));
                }
            }
        });
        ui.menu_button("Help", |ui| {
            if ui.button("About Lichii").clicked() {
                actions.push(Action::About);
            }
        });
    }

    fn status_bar(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            let message = match (self.current_toast(), &self.session) {
                (Some(toast), _) => toast.to_string(),
                (None, Some(s)) if s.is_paused() => format!("{}: paused", s.title),
                (None, Some(s)) => s.title.clone(),
                (None, None) if self.games.is_scanning() => "Scanning game directories…".into(),
                (None, None) => format!("{} games", self.games.entries.len()),
            };
            ui.label(message);
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if self.session.is_none() {
                    if let Some(name) = self.adapter_name() {
                        ui.label(RichText::new(name).weak());
                    }
                    return;
                }
                let snap = self
                    .perf
                    .snapshot(Instant::now(), std::time::Duration::from_secs(1));
                let g = &self.settings.graphics;
                ui.label(format!("Frame: {:.2} ms", snap.frame_time_ms));
                ui.separator();
                ui.label(format!("Game: {:.0} FPS", snap.game_fps));
                ui.separator();
                let speed = snap.emulation_speed * 100.0;
                ui.label(match g.speed_limit {
                    Some(limit) => format!("Speed: {speed:.0}% / {limit}%"),
                    None => format!("Speed: {speed:.0}%"),
                });
                ui.separator();
                let (num, den) = g.resolution_scale.ratio();
                let scale = num as f32 / den as f32;
                ui.label(format!("Scale: {scale}x"));
                ui.separator();
                ui.label(if self.settings.system.docked {
                    "Docked"
                } else {
                    "Handheld"
                });
            });
        });
    }

    fn game_list(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        ui.horizontal(|ui| {
            ui.label("Filter:");
            ui.add(
                egui::TextEdit::singleline(&mut self.games.filter)
                    .hint_text("Name or path")
                    .desired_width(260.0),
            );
            if ui.button("Add Game Directory…").clicked() {
                actions.push(Action::AddGameDir);
            }
            if ui
                .add_enabled(!self.games.is_scanning(), egui::Button::new("Refresh"))
                .clicked()
            {
                actions.push(Action::RefreshGames);
            }
            if self.games.is_scanning() {
                ui.spinner();
            }
        });
        ui.separator();
        if self.games.entries.is_empty() {
            ui.centered_and_justified(|ui| {
                let text = if self.settings.paths.game_dirs.is_empty() {
                    "No game directories yet.\nUse Add Game Directory, or drop a game onto this window."
                } else if self.games.is_scanning() {
                    "Scanning…"
                } else {
                    "No NSP, XCI, NCA, NRO, NSO or ExeFS titles found in the game directories."
                };
                ui.label(RichText::new(text).size(16.0).weak());
            });
            return;
        }
        egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            egui::Grid::new("games")
                .num_columns(4)
                .striped(true)
                .spacing([24.0, 6.0])
                .show(ui, |ui| {
                    for heading in ["Name", "Type", "Size", "Path"] {
                        ui.label(RichText::new(heading).strong());
                    }
                    ui.end_row();
                    for entry in self.games.visible() {
                        let selected = self.ui.selected.as_ref() == Some(&entry.path);
                        let row = ui.selectable_label(selected, &entry.name);
                        if row.clicked() {
                            self.ui.selected = Some(entry.path.clone());
                        }
                        if row.double_clicked() {
                            actions.push(Action::Boot(entry.path.clone()));
                        }
                        row.on_hover_text("Double-click to boot");
                        ui.label(entry.kind.label());
                        ui.label(if entry.size > 0 {
                            format_size(entry.size)
                        } else {
                            String::new()
                        });
                        ui.label(RichText::new(entry.path.display().to_string()).weak());
                        ui.end_row();
                    }
                });
        });
    }

    fn guest_area(&mut self, ui: &mut Ui) {
        let rect = ui.max_rect();
        let ppp = ui.ctx().pixels_per_point();
        let area = Viewport {
            x: (rect.min.x * ppp).round() as i32,
            y: (rect.min.y * ppp).round() as i32,
            width: (rect.width() * ppp).round().max(1.0) as u32,
            height: (rect.height() * ppp).round().max(1.0) as u32,
        };
        let aspect = self.settings.graphics.aspect_ratio;
        let fitted = Viewport::fit(aspect, area.x, area.y, area.width, area.height);
        self.viewport = Some(fitted);
        let guest = egui::Rect::from_min_size(
            egui::pos2(fitted.x as f32 / ppp, fitted.y as f32 / ppp),
            egui::vec2(fitted.width as f32 / ppp, fitted.height as f32 / ppp),
        );
        let painter = ui.painter();
        painter.rect_stroke(
            guest,
            0.0,
            egui::Stroke::new(1.0, Color32::from_gray(40)),
            egui::StrokeKind::Inside,
        );
        let Some(session) = &self.session else { return };
        let fv = &self.settings.frame_view;
        if fv.enabled {
            let window = std::time::Duration::from_secs_f32(fv.graph_seconds.max(0.1));
            let snap = self.perf.snapshot(Instant::now(), window);
            crate::frame_view::draw(painter, guest, fv, &snap, &self.settings.graphics);
        }
        if !session.is_paused() {
            return;
        }
        painter.text(
            guest.center(),
            egui::Align2::CENTER_CENTER,
            format!("{}\nPaused", session.title),
            egui::FontId::proportional(18.0),
            Color32::from_gray(150),
        );
    }

    fn dialogs(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        if let Some(error) = self.ui.error.clone() {
            egui::Modal::new(egui::Id::unique("error")).show(ctx, |ui| {
                ui.heading("Lichii");
                ui.label(error);
                if ui.button("OK").clicked() {
                    self.ui.error = None;
                }
            });
        }
        if self.ui.confirm_stop {
            let title = self.session.as_ref().map_or("", |s| s.title.as_str());
            let mut answer = None;
            egui::Modal::new(egui::Id::unique("confirm stop")).show(ctx, |ui| {
                ui.heading("Stop emulation?");
                ui.label(format!("{title} will close and unsaved progress is lost."));
                ui.horizontal(|ui| {
                    if ui.button("Stop").clicked() {
                        answer = Some(true);
                    }
                    if ui.button("Cancel").clicked() {
                        answer = Some(false);
                    }
                });
            });
            if let Some(stop) = answer {
                self.ui.confirm_stop = false;
                if stop {
                    actions.push(Action::StopConfirmed);
                }
            }
        }
        let mut about = self.ui.about;
        egui::Window::new("About Lichii")
            .open(&mut about)
            .collapsible(false)
            .resizable(false)
            .show(ctx, |ui| {
                ui.heading(format!("Lichii {}", env!("CARGO_PKG_VERSION")));
                ui.label("An experimental Nintendo Switch 2 and Switch emulator, written in Rust.");
                ui.label("Derived from Lichii, Eglantine, Eden, yuzu and Sudachi.");
                if let Some(name) = self.adapter_name() {
                    ui.label(format!("GPU: {name}"));
                }
                ui.label("License: GPL-3.0-or-later");
                ui.hyperlink("https://github.com/zortos293/lichii-rust");
            });
        self.ui.about = about;
    }
}
