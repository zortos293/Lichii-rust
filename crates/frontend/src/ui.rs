//! The main window's egui layout: menu bar, game list or guest area, status bar and dialogs.

use std::time::Instant;

use egui::{Align, Color32, Layout, RichText, Ui};
use lichii_common::settings::{AntiAliasing, HotkeyAction, ScalingFilter};

use lichii_vulkan::present::Viewport;

use crate::app::{Action, App};
use crate::choices::Choice;
use crate::game_list::{GameEntry, format_size};

/// The game list's icon size, in points.
const GAME_ICON: f32 = 40.0;

/// A game's right-click menu, as in Lichii's game list.
fn game_menu(ui: &mut Ui, entry: &GameEntry, actions: &mut Vec<Action>) {
    let path = &entry.path;
    let program_id = entry.meta.as_ref().map_or(0, |m| m.program_id);
    let mut pick = |ui: &mut Ui, enabled: bool, label: &str, action: Action| {
        if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
            actions.push(action);
            ui.close();
        }
    };
    pick(ui, true, "Start Game", Action::Boot(path.clone()));
    pick(
        ui,
        true,
        "Start Game without Custom Configuration",
        Action::BootWithoutCustom(path.clone()),
    );
    ui.separator();
    let known = entry.meta.is_some();
    pick(
        ui,
        known,
        "Properties…",
        Action::GameProperties {
            path: path.clone(),
            program_id,
            title: entry.title().to_owned(),
        },
    );
    pick(
        ui,
        known,
        "Remove Custom Configuration",
        Action::RemoveGameConfig {
            path: path.clone(),
            program_id,
        },
    );
    if !known {
        ui.label(
            RichText::new("Properties need the title's metadata; set up the keys first.")
                .weak()
                .small(),
        );
    }
}

/// A game list cell, centred on the icon's height.
fn centered_cell(ui: &mut Ui, text: egui::WidgetText) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.allocate_exact_size(egui::vec2(0.0, GAME_ICON), egui::Sense::hover());
        ui.add(egui::Label::new(text).extend());
    });
}

/// The title, with the publisher and program id below it, as Lichii's game list shows them.
fn game_name(ui: &Ui, entry: &GameEntry) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let format = |size: f32, color: Color32| egui::TextFormat {
        font_id: egui::FontId::proportional(size),
        color,
        ..Default::default()
    };
    job.append(entry.title(), 0.0, format(14.0, ui.visuals().text_color()));
    let Some(meta) = &entry.meta else {
        return job;
    };
    let id = (meta.program_id != 0).then(|| format!("{:016X}", meta.program_id));
    let details: Vec<&str> = [Some(meta.publisher.as_str()), id.as_deref()]
        .into_iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .collect();
    if !details.is_empty() {
        job.append(
            &format!("\n{}", details.join("  \u{b7}  ")),
            0.0,
            format(11.5, ui.visuals().weak_text_color()),
        );
    }
    job
}

impl App {
    pub(crate) fn draw(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        let immersive = self.is_fullscreen() && self.session.is_some();
        if !immersive {
            egui::Panel::top("menu").show(ui, |ui| {
                egui::MenuBar::new().ui(ui, |ui| self.menus(ui, actions));
            });
            if self.settings.ui.show_status_bar {
                egui::Panel::bottom("status").show(ui, |ui| self.status_bar(ui, actions));
            }
        }
        if self.session.is_some() {
            egui::CentralPanel::no_frame().show(ui, |ui| self.guest_area(ui));
        } else {
            egui::CentralPanel::default().show(ui, |ui| self.game_list(ui, actions));
        }
        self.dialogs(ui.ctx(), actions);
        if let Some(session) = &mut self.session
            && let Some(change) = session.show_applets(ui.ctx(), &self.settings.controls)
        {
            self.apply_controls(change);
        }
        self.show_config(ui.ctx());
        self.show_setup(ui.ctx());
        self.show_game_config(ui.ctx());
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
            if ui
                .add_enabled(
                    self.install.is_none(),
                    egui::Button::new("Install Files to NAND…"),
                )
                .clicked()
            {
                actions.push(Action::InstallFiles);
            }
            let idle = !running && self.install.is_none() && self.firmware.is_none();
            ui.add_enabled_ui(idle, |ui| {
                ui.menu_button("Install Firmware", |ui| {
                    if ui.button("From Folder…").clicked() {
                        actions.push(Action::InstallFirmware { zip: false });
                    }
                    if ui.button("From ZIP…").clicked() {
                        actions.push(Action::InstallFirmware { zip: true });
                    }
                });
            });
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
            if ui.button("Keys and Firmware…").clicked() {
                actions.push(Action::KeysAndFirmware);
            }
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
            if ui
                .add_enabled(running, egui::Button::new("Configure Current Game…"))
                .clicked()
            {
                actions.push(Action::ConfigureCurrentGame);
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
                    self.current().system.docked,
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

    fn status_bar(&self, ui: &mut Ui, actions: &mut Vec<Action>) {
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
                let g = &self.current().graphics;
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
                ui.label(if self.current().system.docked {
                    "Docked"
                } else {
                    "Handheld"
                });
                ui.separator();
                let aa = match g.anti_aliasing {
                    AntiAliasing::None => "NO AA".to_owned(),
                    aa => aa.label().to_uppercase(),
                };
                let aa = ui
                    .button(aa)
                    .on_hover_text("Anti-aliasing. Click to cycle, right-click to pick.");
                if aa.clicked() {
                    actions.push(Action::SetAntiAliasing(g.anti_aliasing.next()));
                }
                pick_menu(&aa, g.anti_aliasing, actions, Action::SetAntiAliasing);
                let filter = ui
                    .button(match g.scaling_filter {
                        ScalingFilter::NearestNeighbor => "NEAREST".to_owned(),
                        ScalingFilter::Fsr => "FSR".to_owned(),
                        filter => filter.label().to_uppercase(),
                    })
                    .on_hover_text("Scaling filter. Click to cycle, right-click to pick.");
                if filter.clicked() {
                    actions.push(Action::SetScalingFilter(g.scaling_filter.next()));
                }
                pick_menu(&filter, g.scaling_filter, actions, Action::SetScalingFilter);
                let (num, den) = g.resolution_scale.ratio();
                let scale = num as f32 / den as f32;
                ui.menu_button(format!("Scale: {scale}x"), |ui| {
                    pick_list(ui, g.resolution_scale, actions, Action::SetResolution);
                })
                .response
                .on_hover_text("The resolution scaling multiplier; applies to the running game.");
            });
        });
    }

    fn game_list(&mut self, ui: &mut Ui, actions: &mut Vec<Action>) {
        ui.horizontal(|ui| {
            ui.label("Filter:");
            ui.add(
                egui::TextEdit::singleline(&mut self.games.filter)
                    .hint_text("Name, publisher, title ID or path")
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
                    "No NSP, XCI, NCA, NUP, NRO, NSO or ExeFS titles found in the game directories."
                };
                ui.label(RichText::new(text).size(16.0).weak());
            });
            return;
        }
        self.games.upload_icons(ui.ctx());
        egui::ScrollArea::both().auto_shrink(false).show(ui, |ui| {
            egui::Grid::new("games")
                .num_columns(6)
                .striped(true)
                .spacing([24.0, 6.0])
                .show(ui, |ui| {
                    for heading in ["Name", "System", "Type", "Version", "Size", "Path"] {
                        ui.label(RichText::new(heading).strong());
                    }
                    ui.end_row();
                    for entry in self.games.visible() {
                        let selected = self.ui.selected.as_ref() == Some(&entry.path);
                        ui.horizontal(|ui| {
                            let size = egui::vec2(GAME_ICON, GAME_ICON);
                            match self.games.icons.get(&entry.path) {
                                Some(icon) => {
                                    ui.add(egui::Image::new((icon.id(), size)).corner_radius(6));
                                }
                                None => {
                                    let (rect, _) =
                                        ui.allocate_exact_size(size, egui::Sense::hover());
                                    ui.painter()
                                        .rect_filled(rect, 6, ui.visuals().faint_bg_color);
                                }
                            }
                            let row = ui.selectable_label(selected, game_name(ui, entry));
                            if row.clicked() {
                                self.ui.selected = Some(entry.path.clone());
                            }
                            if row.double_clicked() {
                                actions.push(Action::Boot(entry.path.clone()));
                            }
                            row.on_hover_text("Double-click to boot; right-click for more")
                                .context_menu(|ui| game_menu(ui, entry, actions));
                        });
                        let meta = entry.meta.as_ref();
                        let size = if entry.size > 0 {
                            format_size(entry.size)
                        } else {
                            String::new()
                        };
                        let path = RichText::new(entry.path.display().to_string()).weak();
                        for text in [
                            meta.map_or("", |m| m.system_label()).into(),
                            entry.kind.label().into(),
                            meta.map_or("", |m| m.version.as_str()).into(),
                            size.into(),
                            path.into(),
                        ] {
                            centered_cell(ui, text);
                        }
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
        let aspect = self.current().graphics.aspect_ratio;
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
            crate::frame_view::draw(painter, guest, fv, &snap, &self.current().graphics);
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
        if let Some((title, text)) = self.ui.info.clone() {
            egui::Modal::new(egui::Id::unique("info")).show(ctx, |ui| {
                ui.heading(title);
                ui.label(text);
                if ui.button("OK").clicked() {
                    self.ui.info = None;
                }
            });
        }
        if let Some(job) = &self.install {
            let (index, count, file, share) = job.progress();
            let name = file
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
            egui::Modal::new(egui::Id::unique("install")).show(ctx, |ui| {
                ui.heading("Installing to NAND");
                ui.label(format!(
                    "Installing file \"{name}\" ({} of {count})…",
                    index + 1
                ));
                ui.add(
                    egui::ProgressBar::new(share)
                        .show_percentage()
                        .desired_width(360.0),
                );
            });
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        if let Some(job) = &self.firmware {
            let share = job.progress().share();
            egui::Modal::new(egui::Id::unique("install firmware")).show(ctx, |ui| {
                ui.heading("Installing Firmware...");
                ui.add(
                    egui::ProgressBar::new(share)
                        .show_percentage()
                        .desired_width(360.0),
                );
                if ui.button("Cancel").clicked() {
                    actions.push(Action::CancelFirmwareInstall);
                }
            });
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
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

/// A right-click menu on `response` listing every value of a setting, the current one marked.
fn pick_menu<T: Choice>(
    response: &egui::Response,
    current: T,
    actions: &mut Vec<Action>,
    action: fn(T) -> Action,
) {
    response.context_menu(|ui| pick_list(ui, current, actions, action));
}

fn pick_list<T: Choice>(
    ui: &mut Ui,
    current: T,
    actions: &mut Vec<Action>,
    action: fn(T) -> Action,
) {
    for &value in T::ALL {
        if ui
            .selectable_label(value == current, value.label())
            .clicked()
        {
            actions.push(action(value));
            ui.close();
        }
    }
}
