//! The configuration window: every field of `Settings`, organised like Lichii's configuration
//! dialog. It edits a draft; OK and Apply hand the draft back to the app.

mod controls;
mod hotkeys;

use std::path::PathBuf;

use egui::{RichText, Ui};
use lichii_common::paths::UserPaths;
use lichii_common::settings::*;
use lichii_input::{Gamepad, HostInput, Key, KeyChord, MouseButton};

use crate::choices::Choice;

pub use controls::ControlsPage;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Page {
    General,
    System,
    Cpu,
    Graphics,
    GraphicsAdvanced,
    FrameView,
    Controls,
    Hotkeys,
    Audio,
    Filesystem,
    Debug,
}

impl Page {
    const ALL: [Self; 11] = [
        Self::General,
        Self::System,
        Self::Cpu,
        Self::Graphics,
        Self::GraphicsAdvanced,
        Self::FrameView,
        Self::Controls,
        Self::Hotkeys,
        Self::Audio,
        Self::Filesystem,
        Self::Debug,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::General => "General",
            Self::System => "System",
            Self::Cpu => "CPU",
            Self::Graphics => "Graphics",
            Self::GraphicsAdvanced => "Advanced Graphics",
            Self::FrameView => "Frame View",
            Self::Controls => "Controls",
            Self::Hotkeys => "Hotkeys",
            Self::Audio => "Audio",
            Self::Filesystem => "Filesystem",
            Self::Debug => "Debug",
        }
    }
}

/// What the app gives the window each frame.
pub struct ConfigEnv<'a> {
    pub devices: &'a [Gamepad],
    /// Live host input while the window is open, for click-to-bind and previews.
    pub host: Option<&'a HostInput>,
    pub adapter: Option<&'a str>,
    pub paths: &'a UserPaths,
    pub running: bool,
}

/// What the window asks of the app after a frame.
#[derive(Default)]
pub struct ConfigOutput {
    /// Settings to apply (OK or Apply).
    pub apply: Option<Settings>,
    pub close: bool,
    pub test_rumble: Option<usize>,
}

pub struct ConfigWindow {
    draft: Settings,
    page: Page,
    controls: ControlsPage,
    hotkey_capture: Option<HotkeyAction>,
}

impl ConfigWindow {
    pub fn new(settings: &Settings, paths: &UserPaths) -> Self {
        Self {
            draft: settings.clone(),
            page: Page::General,
            controls: ControlsPage::new(paths.input_profiles_dir()),
            hotkey_capture: None,
        }
    }

    /// Whether the window wants the next key press as a hotkey chord.
    pub fn wants_chord(&self) -> bool {
        self.hotkey_capture.is_some()
    }

    /// A key press while the window is open; true when the controls page used it.
    pub fn on_key(&mut self, key: Key) -> bool {
        self.page == Page::Controls && self.controls.on_key(key, &mut self.draft)
    }

    /// A mouse press while the window is open; true when it bound a control, in which case the
    /// press must not reach the UI.
    pub fn on_mouse(&mut self, button: MouseButton) -> bool {
        self.page == Page::Controls && self.controls.on_mouse(button, &mut self.draft)
    }

    /// A key chord pressed while [`Self::wants_chord`].
    pub fn on_chord(&mut self, chord: KeyChord) {
        if let Some(action) = self.hotkey_capture.take() {
            self.draft.ui.hotkeys.entry(action).or_default().keys = chord.to_settings_string();
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, env: &ConfigEnv) -> ConfigOutput {
        let mut out = ConfigOutput::default();
        let mut open = true;
        egui::Window::new("Configure")
            .open(&mut open)
            .default_size([980.0, 680.0])
            .min_size([760.0, 480.0])
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Panel::bottom("config buttons").show(ui, |ui| {
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.button("OK").clicked() {
                            out.apply = Some(self.draft.clone());
                            out.close = true;
                        }
                        if ui.button("Apply").clicked() {
                            out.apply = Some(self.draft.clone());
                        }
                        if ui.button("Cancel").clicked() {
                            out.close = true;
                        }
                    });
                });
                egui::Panel::left("config pages")
                    .resizable(false)
                    .default_size(150.0)
                    .show(ui, |ui| {
                        for page in Page::ALL {
                            if ui
                                .selectable_label(self.page == page, page.label())
                                .clicked()
                                && self.page != page
                            {
                                self.page = page;
                                self.hotkey_capture = None;
                                self.controls.cancel_capture();
                            }
                        }
                    });
                egui::CentralPanel::default().show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .auto_shrink(false)
                        .show(ui, |ui| self.page_ui(ui, env, &mut out));
                });
            });
        if !open {
            out.close = true;
        }
        if out.close {
            self.hotkey_capture = None;
        }
        out
    }

    fn page_ui(&mut self, ui: &mut Ui, env: &ConfigEnv, out: &mut ConfigOutput) {
        let d = &mut self.draft;
        match self.page {
            Page::General => {
                section(ui, "Interface");
                grid(ui, "general", |ui| {
                    row(ui, "Theme", |ui| combo(ui, "theme", &mut d.ui.theme));
                    row(ui, "", |ui| {
                        ui.checkbox(&mut d.ui.confirm_stop, "Confirm before stopping emulation")
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(
                            &mut d.ui.pause_when_unfocused,
                            "Pause when the window loses focus",
                        )
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(&mut d.ui.show_status_bar, "Show the status bar")
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(
                            &mut d.ui.hide_cursor_when_idle,
                            "Hide the mouse cursor while idle",
                        )
                    });
                });
                section(ui, "Speed");
                speed_limit(ui, &mut d.graphics.speed_limit);
            }
            Page::System => {
                section(ui, "Console");
                grid(ui, "system", |ui| {
                    row(ui, "Preferred console", |ui| {
                        combo(ui, "console", &mut d.system.preferred_console)
                    });
                    row(ui, "Console mode", |ui| {
                        docked_toggle(ui, &mut d.system.docked)
                    });
                    row(ui, "Language", |ui| {
                        combo(ui, "language", &mut d.system.language)
                    });
                    row(ui, "Region", |ui| combo(ui, "region", &mut d.system.region));
                    row(ui, "Memory layout", |ui| {
                        combo(ui, "memory", &mut d.system.memory_layout)
                    });
                    row(ui, "Username", |ui| {
                        ui.add(egui::TextEdit::singleline(&mut d.system.username).char_limit(32))
                    });
                    row(ui, "RNG seed", |ui| {
                        optional_number(ui, &mut d.system.rng_seed, 0)
                    });
                });
                note(
                    ui,
                    "Switch 2 titles always run as Switch 2; the preferred console applies to titles that run on both.",
                );
            }
            Page::Cpu => {
                section(ui, "CPU");
                grid(ui, "cpu", |ui| {
                    row(ui, "Backend", |ui| {
                        combo(ui, "cpu backend", &mut d.cpu.backend)
                    });
                    row(ui, "Accuracy", |ui| {
                        combo(ui, "cpu accuracy", &mut d.cpu.accuracy)
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(
                            &mut d.cpu.fastmem,
                            "Fastmem (map guest memory into the host)",
                        )
                    });
                });
            }
            Page::Graphics => {
                section(ui, "Device");
                grid(ui, "device", |ui| {
                    row(ui, "Vulkan device", |ui| {
                        let mut auto = d.graphics.device.is_none();
                        let resp = ui.checkbox(&mut auto, "Automatic");
                        if resp.changed() {
                            d.graphics.device =
                                (!auto).then(|| env.adapter.unwrap_or_default().to_owned());
                        }
                        if let Some(name) = &mut d.graphics.device {
                            ui.text_edit_singleline(name);
                        } else if let Some(name) = env.adapter {
                            ui.label(RichText::new(name).weak());
                        }
                    });
                });
                section(ui, "Image");
                grid(ui, "graphics", |ui| {
                    row(ui, "Resolution", |ui| {
                        combo(ui, "resolution", &mut d.graphics.resolution_scale)
                    });
                    row(ui, "Scaling filter", |ui| {
                        combo(ui, "scaling filter", &mut d.graphics.scaling_filter)
                    });
                    row(ui, "FSR sharpness", |ui| {
                        ui.add_enabled(
                            d.graphics.scaling_filter == ScalingFilter::Fsr,
                            egui::Slider::new(&mut d.graphics.fsr_sharpness, 0..=100).suffix("%"),
                        )
                    });
                    row(ui, "Anti-aliasing", |ui| {
                        combo(ui, "aa", &mut d.graphics.anti_aliasing)
                    });
                    row(ui, "Aspect ratio", |ui| {
                        combo(ui, "aspect", &mut d.graphics.aspect_ratio)
                    });
                    row(ui, "Present mode", |ui| {
                        combo(ui, "present", &mut d.graphics.present_mode)
                    });
                    row(ui, "Fullscreen mode", |ui| {
                        combo(ui, "fullscreen", &mut d.graphics.fullscreen)
                    });
                });
                let (num, den) = d.graphics.resolution_scale.ratio();
                note(
                    ui,
                    &format!(
                        "Renders at {}x{} docked, {}x{} handheld (Switch 2: {}x{} / {}x{}).",
                        num * 1920 / den,
                        num * 1080 / den,
                        num * 1280 / den,
                        num * 720 / den,
                        num * 3840 / den,
                        num * 2160 / den,
                        num * 1920 / den,
                        num * 1080 / den,
                    ),
                );
                section(ui, "Speed");
                speed_limit(ui, &mut d.graphics.speed_limit);
            }
            Page::GraphicsAdvanced => {
                section(ui, "Accuracy and textures");
                grid(ui, "advanced", |ui| {
                    row(ui, "GPU accuracy", |ui| {
                        combo(ui, "gpu accuracy", &mut d.graphics.gpu_accuracy)
                    });
                    row(ui, "Anisotropic filtering", |ui| {
                        combo(ui, "aniso", &mut d.graphics.anisotropic_filtering)
                    });
                    row(ui, "ASTC decoding", |ui| {
                        combo(ui, "astc decode", &mut d.graphics.astc_decode)
                    });
                    row(ui, "ASTC recompression", |ui| {
                        combo(ui, "astc recompression", &mut d.graphics.astc_recompression)
                    });
                    row(ui, "VRAM usage", |ui| {
                        combo(ui, "vram", &mut d.graphics.vram_usage)
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(
                            &mut d.graphics.async_shader_compilation,
                            "Build shaders asynchronously",
                        )
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(
                            &mut d.graphics.disk_shader_cache,
                            "Use the disk shader cache",
                        )
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(
                            &mut d.graphics.validation_layers,
                            "Vulkan validation layers (slow)",
                        )
                    });
                });
                note(
                    ui,
                    "Large Switch 2 titles fit in 6-8 GB of VRAM with ASTC recompression set to BC3 and decoding on the CPU asynchronously.",
                );
            }
            Page::FrameView => {
                let fv = &mut d.frame_view;
                section(ui, "Frame view");
                ui.checkbox(&mut fv.enabled, "Show the frame view");
                section(ui, "Show");
                ui.columns(2, |cols| {
                    cols[0].checkbox(&mut fv.show_fps, "Frames per second");
                    cols[0].checkbox(&mut fv.show_frame_time, "Frame time");
                    cols[0].checkbox(&mut fv.show_frame_time_graph, "Frame time graph");
                    cols[0].checkbox(&mut fv.show_emulation_speed, "Emulation speed");
                    cols[1].checkbox(&mut fv.show_cpu_gpu_time, "GPU time");
                    cols[1].checkbox(&mut fv.show_vram_usage, "VRAM usage");
                    cols[1].checkbox(&mut fv.show_shader_compilation, "Shaders building");
                    cols[1].checkbox(&mut fv.show_resolution, "Render resolution");
                });
                section(ui, "Layout");
                grid(ui, "frame view", |ui| {
                    row(ui, "Position", |ui| {
                        combo(ui, "overlay position", &mut fv.position)
                    });
                    row(ui, "Graph length", |ui| {
                        ui.add(egui::Slider::new(&mut fv.graph_seconds, 1.0..=30.0).suffix(" s"))
                    });
                    row(ui, "Opacity", |ui| {
                        ui.add(egui::Slider::new(&mut fv.opacity, 0.1..=1.0))
                    });
                    row(ui, "Scale", |ui| {
                        ui.add(egui::Slider::new(&mut fv.scale, 0.5..=3.0))
                    });
                });
                section(ui, "Preview");
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(640.0, 360.0), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, 0.0, egui::Color32::from_gray(24));
                let sample = crate::frame_view::sample_snapshot();
                crate::frame_view::draw(ui.painter(), rect, fv, &sample, &d.graphics);
                note(
                    ui,
                    "Sample numbers; in a game the overlay sits in this corner of the picture.",
                );
            }
            Page::Controls => self.controls.ui(ui, d, env, out),
            Page::Hotkeys => hotkeys::ui(ui, &mut d.ui.hotkeys, &mut self.hotkey_capture),
            Page::Audio => {
                section(ui, "Audio");
                grid(ui, "audio", |ui| {
                    row(ui, "Output engine", |ui| {
                        combo(ui, "audio output", &mut d.audio.output)
                    });
                    row(ui, "Output device", |ui| {
                        optional_text(ui, &mut d.audio.output_device, "Default")
                    });
                    row(ui, "Volume", |ui| {
                        ui.add(egui::Slider::new(&mut d.audio.volume, 0..=200).suffix("%"))
                    });
                    row(ui, "", |ui| ui.checkbox(&mut d.audio.muted, "Mute"));
                });
            }
            Page::Filesystem => {
                section(ui, "Game directories");
                let mut remove = None;
                for (i, dir) in d.paths.game_dirs.iter().enumerate() {
                    ui.horizontal(|ui| {
                        if ui.small_button("Remove").clicked() {
                            remove = Some(i);
                        }
                        ui.label(dir.display().to_string());
                    });
                }
                if let Some(i) = remove {
                    d.paths.game_dirs.remove(i);
                }
                if ui.button("Add Game Directory…").clicked()
                    && let Some(dir) = rfd::FileDialog::new().pick_folder()
                    && !d.paths.game_dirs.contains(&dir)
                {
                    d.paths.game_dirs.push(dir);
                }
                section(ui, "Profile directories");
                grid(ui, "paths", |ui| {
                    row(ui, "Keys", |ui| {
                        dir_override(ui, &mut d.paths.keys_dir, env.paths.keys_dir())
                    });
                    row(ui, "NAND", |ui| {
                        dir_override(ui, &mut d.paths.nand_dir, env.paths.nand_dir())
                    });
                    row(ui, "SD card", |ui| {
                        dir_override(ui, &mut d.paths.sdmc_dir, env.paths.sdmc_dir())
                    });
                    row(ui, "Mods", |ui| {
                        dir_override(ui, &mut d.paths.mods_dir, env.paths.data_dir.join("load"))
                    });
                });
            }
            Page::Debug => {
                section(ui, "Logging");
                grid(ui, "debug", |ui| {
                    row(ui, "Log filter", |ui| {
                        ui.text_edit_singleline(&mut d.debug.log_filter)
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(&mut d.debug.svc_trace, "Trace every SVC")
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(&mut d.debug.ipc_trace, "Trace every IPC command")
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(
                            &mut d.debug.auto_stub,
                            "Answer unknown service commands with success",
                        )
                    });
                    row(ui, "", |ui| {
                        ui.checkbox(&mut d.debug.dump_shaders, "Dump shaders")
                    });
                    row(ui, "GDB stub port", |ui| {
                        optional_number(ui, &mut d.debug.gdb_stub_port, 6543)
                    });
                });
                note(
                    ui,
                    "The log filter takes effect on the next start. It uses tracing's EnvFilter syntax, e.g. \"info,lichii_kernel=debug\".",
                );
            }
        }
        if env.running && matches!(self.page, Page::System | Page::Cpu) {
            note(ui, "Changes here apply to the next boot.");
        }
    }
}

fn section(ui: &mut Ui, title: &str) {
    ui.add_space(8.0);
    ui.label(RichText::new(title).strong().size(15.0));
    ui.separator();
}

fn note(ui: &mut Ui, text: &str) {
    ui.add_space(4.0);
    ui.label(RichText::new(text).weak().small());
}

fn grid(ui: &mut Ui, id: &str, add: impl FnOnce(&mut Ui)) {
    egui::Grid::new(id)
        .num_columns(2)
        .spacing([16.0, 8.0])
        .min_col_width(160.0)
        .show(ui, add);
}

fn row<R>(ui: &mut Ui, label: &str, add: impl FnOnce(&mut Ui) -> R) {
    ui.label(label);
    ui.horizontal(|ui| add(ui));
    ui.end_row();
}

/// A combo box over every value of `T`; true when the value changed.
pub fn combo<T: Choice>(ui: &mut Ui, id: &str, value: &mut T) -> bool {
    let mut changed = false;
    egui::ComboBox::from_id_salt(id)
        .selected_text(value.label())
        .width(280.0)
        .show_ui(ui, |ui| {
            for &v in T::ALL {
                changed |= ui.selectable_value(value, v, v.label()).changed();
            }
        });
    changed
}

fn docked_toggle(ui: &mut Ui, docked: &mut bool) {
    ui.selectable_value(docked, true, "Docked");
    ui.selectable_value(docked, false, "Handheld");
}

fn speed_limit(ui: &mut Ui, limit: &mut Option<u16>) {
    ui.horizontal(|ui| {
        let mut on = limit.is_some();
        if ui.checkbox(&mut on, "Limit speed to").changed() {
            *limit = on.then_some(100);
        }
        let mut value = limit.unwrap_or(100);
        let slider = egui::Slider::new(&mut value, 1..=MAX_SPEED_LIMIT)
            .suffix("%")
            .logarithmic(true);
        if ui.add_enabled(on, slider).changed() {
            *limit = Some(value);
        }
    });
}

fn optional_number<T: egui::emath::Numeric>(ui: &mut Ui, value: &mut Option<T>, default: T) {
    let mut on = value.is_some();
    if ui.checkbox(&mut on, "").changed() {
        *value = on.then_some(default);
    }
    if let Some(v) = value {
        ui.add(egui::DragValue::new(v));
    }
}

fn optional_text(ui: &mut Ui, value: &mut Option<String>, none: &str) {
    let mut on = value.is_some();
    if ui.checkbox(&mut on, "Custom").changed() {
        *value = on.then(String::new);
    }
    match value {
        Some(text) => {
            ui.text_edit_singleline(text);
        }
        None => {
            ui.label(RichText::new(none).weak());
        }
    }
}

fn dir_override(ui: &mut Ui, value: &mut Option<PathBuf>, default: PathBuf) {
    let shown = value.clone().unwrap_or(default);
    ui.label(shown.display().to_string());
    if ui.small_button("Browse…").clicked()
        && let Some(dir) = rfd::FileDialog::new().set_directory(&shown).pick_folder()
    {
        *value = Some(dir);
    }
    if value.is_some() && ui.small_button("Reset").clicked() {
        *value = None;
    }
}
