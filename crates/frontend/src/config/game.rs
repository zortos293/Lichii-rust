//! The per-game Properties window: settings a game overrides, each with a Custom toggle; the
//! rest follow the global configuration.

use std::collections::BTreeSet;

use egui::{RichText, Ui};
use lichii_common::settings::{ScalingFilter, Settings};

use super::{combo, docked_toggle, note, section, speed_limit, time_zone};
use crate::per_game::{GameConfig, copy_key};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    System,
    Cpu,
    Graphics,
    Audio,
}

impl Page {
    const ALL: [Self; 4] = [Self::System, Self::Cpu, Self::Graphics, Self::Audio];

    fn label(self) -> &'static str {
        match self {
            Self::System => "System",
            Self::Cpu => "CPU",
            Self::Graphics => "Graphics",
            Self::Audio => "Audio",
        }
    }
}

pub struct GameConfigWindow {
    pub title: String,
    pub program_id: u64,
    pub config: GameConfig,
    /// The settings the game would run with: global, with the custom ones edited.
    draft: Settings,
    custom: BTreeSet<&'static str>,
    page: Page,
}

#[derive(Default)]
pub struct GameConfigOutput {
    /// The configuration file was written; a running game picks it up.
    pub saved: bool,
    pub close: bool,
}

impl GameConfigWindow {
    pub fn new(title: String, program_id: u64, config: GameConfig, global: &Settings) -> Self {
        Self {
            title,
            program_id,
            draft: config.apply(global),
            custom: config.overridden(),
            config,
            page: Page::System,
        }
    }

    pub fn show(
        &mut self,
        ctx: &egui::Context,
        global: &Settings,
        running: bool,
    ) -> GameConfigOutput {
        let mut out = GameConfigOutput::default();
        let mut open = true;
        egui::Window::new(format!("Properties: {}", self.title))
            .open(&mut open)
            .default_size([720.0, 560.0])
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Panel::bottom("game config buttons").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            out.saved = self.save();
                            out.close = true;
                        }
                        if ui.button("Apply").clicked() {
                            out.saved = self.save();
                        }
                        if ui.button("Cancel").clicked() {
                            out.close = true;
                        }
                        ui.separator();
                        if ui
                            .add_enabled(!self.custom.is_empty(), egui::Button::new("Use Global for All"))
                            .clicked()
                        {
                            self.custom.clear();
                            self.draft = global.clone();
                        }
                    });
                });
                egui::CentralPanel::default().show(ui, |ui| {
                    let id = match self.program_id {
                        0 => "no title ID".to_owned(),
                        id => format!("title ID {id:016X}"),
                    };
                    let file = self.config.file().display().to_string();
                    ui.label(RichText::new(format!("{id} \u{b7} {file}")).weak().small());
                    ui.horizontal(|ui| {
                        for page in Page::ALL {
                            ui.selectable_value(&mut self.page, page, page.label());
                        }
                    });
                    ui.separator();
                    egui::ScrollArea::vertical()
                        .auto_shrink(false)
                        .show(ui, |ui| self.page_ui(ui, global));
                    if running {
                        note(
                            ui,
                            "Graphics changes reach the running game on Apply; the rest apply the next time it starts.",
                        );
                    }
                });
            });
        if !open {
            out.close = true;
        }
        out
    }

    fn save(&mut self) -> bool {
        self.config.set_overrides(&self.draft, &self.custom);
        match self.config.save() {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!("{}: {e}", self.config.file().display());
                false
            }
        }
    }

    fn page_ui(&mut self, ui: &mut Ui, global: &Settings) {
        let mut rows = Rows {
            custom: &mut self.custom,
            draft: &mut self.draft,
            global,
        };
        match self.page {
            Page::System => {
                section(ui, "Console");
                rows.grid(ui, "game system", |ui, r| {
                    r.row(
                        ui,
                        "system.preferred_console",
                        "Preferred console",
                        |ui, d| {
                            combo(ui, "game console", &mut d.system.preferred_console);
                        },
                    );
                    r.row(ui, "system.docked", "Console mode", |ui, d| {
                        docked_toggle(ui, &mut d.system.docked);
                    });
                    r.row(ui, "system.language", "Language", |ui, d| {
                        combo(ui, "game language", &mut d.system.language);
                    });
                    r.row(ui, "system.region", "Region", |ui, d| {
                        combo(ui, "game region", &mut d.system.region);
                    });
                    r.row(ui, "system.memory_layout", "Memory layout", |ui, d| {
                        combo(ui, "game memory", &mut d.system.memory_layout);
                    });
                    r.row(ui, "system.time_zone", "Time zone", |ui, d| {
                        time_zone(ui, &mut d.system.time_zone);
                    });
                    r.row(ui, "system.network_enabled", "Network", |ui, d| {
                        ui.checkbox(&mut d.system.network_enabled, "Allow the game to go online");
                    });
                });
            }
            Page::Cpu => {
                section(ui, "CPU");
                rows.grid(ui, "game cpu", |ui, r| {
                    r.row(ui, "cpu.backend", "Backend", |ui, d| {
                        combo(ui, "game cpu backend", &mut d.cpu.backend);
                    });
                    r.row(ui, "cpu.accuracy", "Accuracy", |ui, d| {
                        combo(ui, "game cpu accuracy", &mut d.cpu.accuracy);
                    });
                    r.row(ui, "cpu.fastmem", "Fastmem", |ui, d| {
                        ui.checkbox(&mut d.cpu.fastmem, "Map guest memory into the host");
                    });
                });
            }
            Page::Graphics => {
                section(ui, "Image");
                rows.grid(ui, "game graphics", |ui, r| {
                    r.row(ui, "graphics.resolution_scale", "Resolution", |ui, d| {
                        combo(ui, "game resolution", &mut d.graphics.resolution_scale);
                    });
                    r.row(ui, "graphics.scaling_filter", "Scaling filter", |ui, d| {
                        combo(ui, "game scaling filter", &mut d.graphics.scaling_filter);
                    });
                    r.row(ui, "graphics.fsr_sharpness", "FSR sharpness", |ui, d| {
                        ui.add_enabled(
                            d.graphics.scaling_filter == ScalingFilter::Fsr,
                            egui::Slider::new(&mut d.graphics.fsr_sharpness, 0..=100).suffix("%"),
                        );
                    });
                    r.row(ui, "graphics.anti_aliasing", "Anti-aliasing", |ui, d| {
                        combo(ui, "game aa", &mut d.graphics.anti_aliasing);
                    });
                    r.row(ui, "graphics.aspect_ratio", "Aspect ratio", |ui, d| {
                        combo(ui, "game aspect", &mut d.graphics.aspect_ratio);
                    });
                    r.row(ui, "graphics.speed_limit", "Speed", |ui, d| {
                        speed_limit(ui, &mut d.graphics.speed_limit);
                    });
                });
                section(ui, "Accuracy and textures");
                rows.grid(ui, "game advanced", |ui, r| {
                    r.row(ui, "graphics.gpu_accuracy", "GPU accuracy", |ui, d| {
                        combo(ui, "game gpu accuracy", &mut d.graphics.gpu_accuracy);
                    });
                    r.row(
                        ui,
                        "graphics.anisotropic_filtering",
                        "Anisotropic filtering",
                        |ui, d| {
                            combo(ui, "game aniso", &mut d.graphics.anisotropic_filtering);
                        },
                    );
                    r.row(ui, "graphics.astc_decode", "ASTC decoding", |ui, d| {
                        combo(ui, "game astc decode", &mut d.graphics.astc_decode);
                    });
                    r.row(
                        ui,
                        "graphics.astc_recompression",
                        "ASTC recompression",
                        |ui, d| {
                            combo(
                                ui,
                                "game astc recompression",
                                &mut d.graphics.astc_recompression,
                            );
                        },
                    );
                    r.row(ui, "graphics.vram_usage", "VRAM usage", |ui, d| {
                        combo(ui, "game vram", &mut d.graphics.vram_usage);
                    });
                    r.row(
                        ui,
                        "graphics.async_shader_compilation",
                        "Shaders",
                        |ui, d| {
                            ui.checkbox(
                                &mut d.graphics.async_shader_compilation,
                                "Build shaders asynchronously",
                            );
                        },
                    );
                    r.row(ui, "graphics.disk_shader_cache", "Shader cache", |ui, d| {
                        ui.checkbox(
                            &mut d.graphics.disk_shader_cache,
                            "Use the disk shader cache",
                        );
                    });
                });
            }
            Page::Audio => {
                section(ui, "Audio");
                rows.grid(ui, "game audio", |ui, r| {
                    r.row(ui, "audio.volume", "Volume", |ui, d| {
                        ui.add(egui::Slider::new(&mut d.audio.volume, 0..=200).suffix("%"));
                    });
                    r.row(ui, "audio.muted", "Mute", |ui, d| {
                        ui.checkbox(&mut d.audio.muted, "Mute this game");
                    });
                });
            }
        }
    }
}

/// Rows of overridable settings: a label, a Custom toggle, and the setting, editable only while
/// it is custom. Turning Custom off shows the global value again.
struct Rows<'a> {
    custom: &'a mut BTreeSet<&'static str>,
    draft: &'a mut Settings,
    global: &'a Settings,
}

impl Rows<'_> {
    fn grid(&mut self, ui: &mut Ui, id: &str, add: impl FnOnce(&mut Ui, &mut Self)) {
        egui::Grid::new(id)
            .num_columns(3)
            .spacing([16.0, 8.0])
            .min_col_width(80.0)
            .show(ui, |ui| add(ui, self));
    }

    fn row(
        &mut self,
        ui: &mut Ui,
        key: &'static str,
        label: &str,
        add: impl FnOnce(&mut Ui, &mut Settings),
    ) {
        ui.horizontal(|ui| {
            ui.set_min_width(150.0);
            ui.label(label);
        });
        let mut on = self.custom.contains(key);
        let toggle = ui
            .checkbox(&mut on, "Custom")
            .on_hover_text("Off: use the global setting");
        if toggle.changed() {
            if on {
                self.custom.insert(key);
            } else {
                self.custom.remove(key);
                copy_key(self.global, self.draft, key);
            }
        }
        ui.add_enabled_ui(on, |ui| ui.horizontal(|ui| add(ui, self.draft)));
        ui.end_row();
    }
}
