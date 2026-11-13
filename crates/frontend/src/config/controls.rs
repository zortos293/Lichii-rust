//! The controls page, like Lichii's player configuration: players 1-8, controller type, device,
//! a live controller diagram with click-to-bind, sticks, motion, vibration and profiles.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, Response, RichText, Sense, Stroke, Ui, vec2};
use lichii_common::settings::{
    ButtonOptions, ControllerType, InputBinding, InputDevice, NpadButton, PlayerInput, Settings,
    StickBinding,
};
use lichii_input::capture::{detect_press, detect_stick};
use lichii_input::{
    HostInput, Key, MouseButton, NpadButtons, NpadState, PadAxis, PadButton, PlayerMapping,
    ProfileStore, STICK_MAX, Source, UPDATE_PERIOD,
};

use super::controller_layout::{self, Body};
use super::{ConfigEnv, ConfigOutput, combo, docked_toggle, grid, note, row, section};

const CAPTURE_TIMEOUT: Duration = Duration::from_secs(5);
const ACCENT: Color32 = Color32::from_rgb(230, 60, 60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dir {
    Up,
    Down,
    Left,
    Right,
    Modifier,
}

impl Dir {
    const ALL: [Self; 5] = [
        Self::Up,
        Self::Down,
        Self::Left,
        Self::Right,
        Self::Modifier,
    ];

    fn label(self) -> &'static str {
        match self {
            Self::Up => "Up",
            Self::Down => "Down",
            Self::Left => "Left",
            Self::Right => "Right",
            Self::Modifier => "Modifier",
        }
    }

    fn binding(self, stick: &mut StickBinding) -> &mut InputBinding {
        match self {
            Self::Up => &mut stick.up,
            Self::Down => &mut stick.down,
            Self::Left => &mut stick.left,
            Self::Right => &mut stick.right,
            Self::Modifier => &mut stick.modifier,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    Button(NpadButton),
    Stick { right: bool, dir: Dir },
    Analog { right: bool },
    Motion,
}

struct Capture {
    target: Target,
    baseline: Option<HostInput>,
    started: Instant,
}

pub struct ControlsPage {
    player: usize,
    profiles: ProfileStore,
    profile_names: Option<Vec<String>>,
    profile_choice: String,
    message: Option<String>,
    capture: Option<Capture>,
    preview: Option<Preview>,
}

/// The page's own compiled mapping for the live diagram, kept across frames and resolved at the
/// input thread's rate so toggle and turbo buttons show as they will in game.
struct Preview {
    player: usize,
    input: PlayerInput,
    pad: Option<usize>,
    mapping: PlayerMapping,
    started: Instant,
    updates: u64,
}

impl ControlsPage {
    pub fn new(profiles_dir: PathBuf) -> Self {
        Self {
            player: 0,
            profiles: ProfileStore::new(profiles_dir),
            profile_names: None,
            profile_choice: String::new(),
            message: None,
            capture: None,
            preview: None,
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, d: &mut Settings, env: &ConfigEnv, out: &mut ConfigOutput) {
        let first = &d.controls.players[0];
        let handheld = first.connected && first.controller_type.is_handheld();
        if handheld {
            d.system.docked = false;
        }
        section(ui, "Console");
        grid(ui, "console mode", |ui| {
            row(ui, "Console mode", |ui| {
                ui.add_enabled_ui(!handheld, |ui| docked_toggle(ui, &mut d.system.docked))
                    .response
                    .on_disabled_hover_text("Player 1 is handheld, so the console is undocked")
            });
            row(ui, "", |ui| {
                ui.checkbox(&mut d.controls.keyboard_enabled, "Keyboard and mouse")
            });
            row(ui, "", |ui| {
                ui.checkbox(&mut d.controls.touchscreen_enabled, "Touchscreen (mouse)")
            });
            row(ui, "Mouse panning", |ui| {
                ui.checkbox(&mut d.controls.mouse_panning, "");
                ui.add_enabled(
                    d.controls.mouse_panning,
                    egui::Slider::new(&mut d.controls.mouse_panning_sensitivity, 1.0..=100.0)
                        .text("sensitivity"),
                )
            });
        });

        section(ui, "Players");
        let mut connect = None;
        ui.horizontal_wrapped(|ui| {
            for (i, player) in d.controls.players.iter().enumerate() {
                let mut connected = player.connected;
                if ui
                    .checkbox(&mut connected, "")
                    .on_hover_text("Connected")
                    .changed()
                {
                    connect = Some((i, connected));
                }
                let text = RichText::new(format!("Player {}", i + 1));
                let text = if player.connected {
                    text.strong()
                } else {
                    text.weak()
                };
                if ui.selectable_label(self.player == i, text).clicked() {
                    self.player = i;
                    self.capture = None;
                }
                ui.add_space(6.0);
            }
        });
        ui.separator();

        let keyboard = d.controls.keyboard_enabled;
        let docked = d.system.docked;
        let index = self.player;
        let player = &mut d.controls.players[index];
        self.poll_capture(player, env.host);

        ui.horizontal(|ui| {
            let mut connected = player.connected;
            if ui.checkbox(&mut connected, "Connected").changed() {
                connect = Some((index, connected));
            }
            ui.label("Controller");
            combo(ui, "controller type", &mut player.controller_type);
            ui.label("Device");
            device_combo(ui, player, env);
        });
        if index == 0 && !docked && !player.controller_type.is_handheld() {
            note(
                ui,
                "In handheld mode the console reads player 1 as the attached Joy-Cons; pick Handheld for player 1.",
            );
        }
        self.profiles_row(ui, player);

        let pad = pad_index(player, env.host);
        let state = env
            .host
            .map(|host| self.preview(player, host, pad, keyboard));

        ui.add_space(6.0);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(250.0);
                self.stick_ui(ui, player, false, state.as_ref());
                self.buttons_ui(ui, player, &LEFT_BUTTONS);
            });
            ui.vertical(|ui| {
                ui.set_width(380.0);
                self.diagram(ui, player, state.as_ref());
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.checkbox(&mut player.motion_enabled, "Motion");
                    let text = match state.and_then(|s| s.motion) {
                        Some(m) => format!(
                            "gyro {:+.2} {:+.2} {:+.2} rev/s",
                            m.gyro[0], m.gyro[1], m.gyro[2]
                        ),
                        None if pad.is_some() => "no motion sensors reported".into(),
                        None => "needs a gyro or a motion button".into(),
                    };
                    ui.label(RichText::new(text).weak().small());
                });
                ui.horizontal(|ui| {
                    ui.label("Motion button")
                        .on_hover_text("Without a gyro the controller lies at rest; holding this input shakes it");
                    self.bind_button(ui, &mut player.motion_button, None, true, Target::Motion);
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut player.vibration_enabled, "Vibration");
                    ui.add_enabled(
                        player.vibration_enabled,
                        egui::Slider::new(&mut player.vibration_strength, 0..=100).suffix("%"),
                    );
                    if ui
                        .add_enabled(pad.is_some(), egui::Button::new("Test"))
                        .on_hover_text("Uses the applied settings")
                        .clicked()
                    {
                        out.test_rumble = Some(self.player);
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Restore Defaults").clicked() {
                        restore_defaults(player);
                    }
                    if ui.button("Clear All").clicked() {
                        player.buttons.clear();
                        player.button_options.clear();
                        player.left_stick = StickBinding::default();
                        player.right_stick = StickBinding::default();
                        player.motion_button = InputBinding::None;
                    }
                });
                if let Some(cap) = &self.capture {
                    let left = CAPTURE_TIMEOUT.saturating_sub(cap.started.elapsed());
                    ui.label(
                        RichText::new(format!(
                            "Press the input for {} ({}s)…",
                            target_label(cap.target),
                            left.as_secs() + 1
                        ))
                        .color(ACCENT),
                    );
                    note(ui, "Press Esc to cancel.");
                }
                if env.host.is_none() {
                    note(ui, "Waiting for the input thread…");
                }
                note(
                    ui,
                    "Click a button or stick to bind it; right-click for clear, invert, toggle, turbo and threshold.",
                );
            });
            ui.vertical(|ui| {
                ui.set_width(250.0);
                self.stick_ui(ui, player, true, state.as_ref());
                self.buttons_ui(ui, player, &RIGHT_BUTTONS);
            });
        });
        if let Some((i, connected)) = connect {
            set_connected(&mut d.controls.players, i, connected);
        }
        ui.ctx().request_repaint_after(Duration::from_millis(16));
    }

    /// The live diagram's state. The mapping is compiled again only when the player's settings or
    /// device change, and runs the input-thread updates due since the last frame.
    fn preview(
        &mut self,
        player: &PlayerInput,
        host: &HostInput,
        pad: Option<usize>,
        keyboard: bool,
    ) -> NpadState {
        let index = self.player;
        if self
            .preview
            .as_ref()
            .is_some_and(|p| p.player != index || p.pad != pad || p.input != *player)
        {
            self.preview = None;
        }
        let preview = self.preview.get_or_insert_with(|| Preview {
            player: index,
            input: player.clone(),
            pad,
            mapping: PlayerMapping::compile(player, host),
            started: Instant::now(),
            updates: 0,
        });
        let due = (preview.started.elapsed().as_nanos() / UPDATE_PERIOD.as_nanos()) as u64;
        let steps = due.saturating_sub(preview.updates).clamp(1, 64);
        preview.updates = due.max(preview.updates + 1);
        let mut state = preview.mapping.resolve(host, keyboard);
        for _ in 1..steps {
            state = preview.mapping.resolve(host, keyboard);
        }
        state
    }

    fn poll_capture(&mut self, player: &mut PlayerInput, host: Option<&HostInput>) {
        let Some(cap) = &mut self.capture else { return };
        if cap.started.elapsed() > CAPTURE_TIMEOUT {
            self.capture = None;
            return;
        }
        let Some(host) = host else { return };
        let Some(baseline) = &cap.baseline else {
            cap.baseline = Some(host.clone());
            return;
        };
        let pad = match &player.device {
            InputDevice::Gamepad { .. } => pad_index(player, Some(host)).or(Some(usize::MAX)),
            InputDevice::None | InputDevice::Keyboard => Some(usize::MAX),
        };
        let target = cap.target;
        if let Target::Analog { right } = target {
            if let Some(((x, y), _)) = detect_stick(baseline, host, pad) {
                let stick = if right {
                    &mut player.right_stick
                } else {
                    &mut player.left_stick
                };
                stick.analog_x = Some(x.name().into());
                stick.analog_y = Some(y.name().into());
                self.capture = None;
            }
            return;
        }
        let Some((source, _)) = detect_press(baseline, host, pad) else {
            return;
        };
        if matches!(source, Source::Key(_) | Source::Mouse(_)) {
            cap.baseline = Some(host.clone());
            return;
        }
        bind(player, target, source.to_binding());
        self.capture = None;
    }

    /// A key press the window saw. While capturing, Esc cancels and any other key is bound; keys
    /// and mouse buttons come straight from the window because a quick tap can start and end
    /// between two host-input snapshots. True when the press was used.
    pub fn on_key(&mut self, key: Key, d: &mut Settings) -> bool {
        if key.name() == "Escape" && self.capture.is_some() {
            self.capture = None;
            return true;
        }
        self.bind_source(Source::Key(key), d)
    }

    /// A mouse press the window saw; while capturing it is bound (left button included) instead
    /// of reaching the UI. True when it was bound.
    pub fn on_mouse(&mut self, button: MouseButton, d: &mut Settings) -> bool {
        self.bind_source(Source::Mouse(button), d)
    }

    pub fn cancel_capture(&mut self) {
        self.capture = None;
    }

    fn bind_source(&mut self, source: Source, d: &mut Settings) -> bool {
        let Some(cap) = &self.capture else {
            return false;
        };
        if matches!(cap.target, Target::Analog { .. }) {
            return false;
        }
        bind(
            &mut d.controls.players[self.player],
            cap.target,
            source.to_binding(),
        );
        self.capture = None;
        true
    }

    fn start(&mut self, target: Target) {
        self.capture = Some(Capture {
            target,
            baseline: None,
            started: Instant::now(),
        });
    }

    fn capturing(&self, target: Target) -> bool {
        self.capture.as_ref().is_some_and(|c| c.target == target)
    }

    /// A button showing a binding: click starts capture, right-click opens Lichii's binding menu.
    fn bind_button(
        &mut self,
        ui: &mut Ui,
        binding: &mut InputBinding,
        options: Option<&mut ButtonOptions>,
        threshold: bool,
        target: Target,
    ) {
        let text = if self.capturing(target) {
            RichText::new("…").color(ACCENT)
        } else {
            let prefix = options_prefix(options.as_deref().copied().unwrap_or_default());
            RichText::new(format!("{prefix}{}", binding_label(binding)))
        };
        let response = ui.add_sized([130.0, 20.0], egui::Button::new(text));
        if response.clicked() {
            self.start(target);
        }
        binding_menu(&response, binding, options, threshold);
    }

    fn buttons_ui(&mut self, ui: &mut Ui, player: &mut PlayerInput, buttons: &[NpadButton]) {
        let controller = player.controller_type;
        egui::Grid::new(("buttons", buttons[0]))
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for &button in buttons {
                    if !controller.has_button(button)
                        && !controller_layout::analog_trigger(controller, button)
                    {
                        continue;
                    }
                    ui.label(button_label(controller, button));
                    let digital = !controller_layout::analog_trigger(controller, button);
                    edit_button(player, button, |binding, options| {
                        let options = digital.then_some(options);
                        self.bind_button(ui, binding, options, digital, Target::Button(button));
                    });
                    ui.end_row();
                }
            });
    }

    fn stick_ui(
        &mut self,
        ui: &mut Ui,
        player: &mut PlayerInput,
        right: bool,
        state: Option<&NpadState>,
    ) {
        let controller = player.controller_type;
        let has = if right {
            controller.has_right_stick()
        } else {
            controller.has_left_stick()
        };
        let name = match (controller, right) {
            (ControllerType::GameCube, false) => "Control Stick",
            (ControllerType::GameCube, true) => "C-Stick",
            (_, false) => "Left Stick",
            (_, true) => "Right Stick",
        };
        ui.label(RichText::new(name).strong());
        if !has {
            ui.label(RichText::new("Not on this controller").weak());
            return;
        }
        let position = state.map(|s| if right { s.right_stick } else { s.left_stick });
        let stick = if right {
            &mut player.right_stick
        } else {
            &mut player.left_stick
        };
        ui.horizontal(|ui| {
            stick_preview(ui, stick, position.map(|p| (p.x, p.y)));
            egui::Grid::new(("stick", right))
                .num_columns(2)
                .spacing([6.0, 3.0])
                .show(ui, |ui| {
                    for dir in Dir::ALL {
                        ui.label(dir.label());
                        self.bind_button(
                            ui,
                            dir.binding(stick),
                            None,
                            true,
                            Target::Stick { right, dir },
                        );
                        ui.end_row();
                    }
                });
        });
        ui.horizontal(|ui| {
            let analog = match (&stick.analog_x, &stick.analog_y) {
                (Some(x), Some(y)) => format!("Analog: {x} / {y}"),
                _ => "Analog: none".into(),
            };
            let text = if self.capturing(Target::Analog { right }) {
                RichText::new("Move the stick…").color(ACCENT)
            } else {
                RichText::new(analog)
            };
            let response = ui
                .button(text)
                .on_hover_text("Click, then move the stick. Right-click to clear.");
            if response.clicked() {
                self.start(Target::Analog { right });
            }
            response.context_menu(|ui| {
                if ui.button("Clear").clicked() {
                    stick.analog_x = None;
                    stick.analog_y = None;
                    ui.close();
                }
            });
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut stick.invert_x, "Invert X");
            ui.checkbox(&mut stick.invert_y, "Invert Y");
        });
        percent_slider(ui, &mut stick.deadzone, 0.0..=0.9, "Deadzone");
        percent_slider(ui, &mut stick.range, 0.5..=1.5, "Range");
        percent_slider(ui, &mut stick.modifier_scale, 0.0..=1.0, "Modifier");
        ui.add_space(8.0);
    }

    fn profiles_row(&mut self, ui: &mut Ui, player: &mut PlayerInput) {
        if self.profile_names.is_none() {
            self.profile_names = Some(self.profiles.list().unwrap_or_default());
        }
        let names = self.profile_names.clone().unwrap_or_default();
        ui.horizontal(|ui| {
            ui.label("Profile");
            egui::ComboBox::from_id_salt("profile")
                .selected_text(if self.profile_choice.is_empty() {
                    player.profile.clone().unwrap_or_else(|| "—".into())
                } else {
                    self.profile_choice.clone()
                })
                .width(180.0)
                .show_ui(ui, |ui| {
                    for name in names.iter() {
                        ui.selectable_value(&mut self.profile_choice, name.clone(), name);
                    }
                });
            ui.add(
                egui::TextEdit::singleline(&mut self.profile_choice)
                    .hint_text("Profile name")
                    .desired_width(140.0),
            );
            let name = self.profile_choice.trim().to_owned();
            let exists = names.contains(&name);
            let mut refresh = false;
            if ui.add_enabled(exists, egui::Button::new("Load")).clicked() {
                self.message = Some(match self.profiles.load_into(&name, player) {
                    Ok(()) => format!("Loaded \"{name}\""),
                    Err(e) => e.to_string(),
                });
            }
            if ui
                .add_enabled(!name.is_empty(), egui::Button::new("Save"))
                .clicked()
            {
                self.message = Some(match self.profiles.save(&name, player) {
                    Ok(()) => {
                        player.profile = Some(name.clone());
                        refresh = true;
                        format!("Saved \"{name}\"")
                    }
                    Err(e) => e.to_string(),
                });
            }
            if ui
                .add_enabled(exists, egui::Button::new("Delete"))
                .clicked()
            {
                self.message = Some(match self.profiles.delete(&name) {
                    Ok(()) => {
                        refresh = true;
                        if player.profile.as_deref() == Some(&name) {
                            player.profile = None;
                        }
                        format!("Deleted \"{name}\"")
                    }
                    Err(e) => e.to_string(),
                });
            }
            if refresh {
                self.profile_names = None;
            }
            if let Some(message) = &self.message {
                ui.label(RichText::new(message).weak());
            }
        });
    }

    fn diagram(&mut self, ui: &mut Ui, player: &mut PlayerInput, state: Option<&NpadState>) {
        let controller = player.controller_type;
        let layout = controller_layout::layout(controller);
        let (response, painter) =
            ui.allocate_painter(controller_layout::SIZE.into(), Sense::hover());
        let offset = response.rect.min.to_vec2();
        let visuals = ui.visuals().clone();
        let outline = Stroke::new(1.5, visuals.widgets.inactive.fg_stroke.color);
        for body in &layout.bodies {
            let (rect, radius, fill) = match *body {
                Body::Shell(rect, radius) => (rect, radius, visuals.widgets.inactive.bg_fill),
                Body::Screen(rect) => (rect, 4.into(), visuals.extreme_bg_color),
            };
            painter.rect(
                rect.translate(offset),
                radius,
                fill,
                outline,
                egui::StrokeKind::Inside,
            );
        }
        let held = state.map_or(NpadButtons::empty(), |s| s.buttons);
        let mut clicked = None;
        for &(button, rect) in &layout.buttons {
            let rect = rect.translate(offset);
            let id = ui.scope_id().with(("diagram", button));
            let hit = ui.interact(rect, id, Sense::click());
            let pressed = held.contains(NpadButtons::from_button(button))
                || state.is_some_and(|s| trigger_pressed(controller, button, s));
            let fill = if pressed || self.capturing(Target::Button(button)) {
                ACCENT
            } else if hit.hovered() {
                visuals.widgets.hovered.bg_fill
            } else {
                visuals.widgets.inactive.weak_bg_fill
            };
            let round = if rect.width() == rect.height() {
                rect.width() / 2.0
            } else {
                4.0
            };
            painter.rect(rect, round, fill, outline, egui::StrokeKind::Inside);
            let ink = visuals.text_color();
            let center = rect.center();
            let arrow = match button {
                NpadButton::DPadUp => Some(vec2(0.0, -1.0)),
                NpadButton::DPadDown => Some(vec2(0.0, 1.0)),
                NpadButton::DPadLeft => Some(vec2(-1.0, 0.0)),
                NpadButton::DPadRight => Some(vec2(1.0, 0.0)),
                _ => None,
            };
            if let Some(dir) = arrow {
                let side = vec2(-dir.y, dir.x);
                let points = vec![
                    center + dir * 4.0,
                    center - dir * 3.0 + side * 4.0,
                    center - dir * 3.0 - side * 4.0,
                ];
                painter.add(egui::Shape::convex_polygon(points, ink, Stroke::NONE));
            } else if button == NpadButton::Capture {
                painter.circle_filled(center, 3.5, ink);
            } else {
                painter.text(
                    center,
                    egui::Align2::CENTER_CENTER,
                    short_label(controller, button),
                    egui::FontId::proportional(10.0),
                    ink,
                );
            }
            let binding = player.buttons.get(&button).cloned().unwrap_or_default();
            let options = player
                .button_options
                .get(&button)
                .copied()
                .unwrap_or_default();
            let hit = hit.on_hover_text(format!(
                "{}: {}{}",
                button_label(controller, button),
                options_prefix(options),
                binding_label(&binding)
            ));
            if hit.clicked() {
                clicked = Some(Target::Button(button));
            }
            let digital = !controller_layout::analog_trigger(controller, button);
            edit_button(player, button, |binding, options| {
                binding_menu(&hit, binding, digital.then_some(options), digital);
            });
        }
        for stick in &layout.sticks {
            let right = stick.right;
            let center = stick.center + offset;
            let rect = Rect::from_center_size(center, vec2(stick.radius, stick.radius) * 2.0);
            let hit = ui.interact(rect, ui.scope_id().with(("stick", right)), Sense::click());
            let ring = if self.capturing(Target::Analog { right }) || hit.hovered() {
                ACCENT
            } else {
                outline.color
            };
            painter.circle(
                center,
                stick.radius,
                visuals.widgets.inactive.weak_bg_fill,
                Stroke::new(1.5, ring),
            );
            let p = state.map(|s| if right { s.right_stick } else { s.left_stick });
            let (dx, dy) = p.map_or((0.0, 0.0), |p| {
                (p.x as f32 / STICK_MAX as f32, p.y as f32 / STICK_MAX as f32)
            });
            let travel = stick.radius * 2.0 / 3.0;
            painter.circle_filled(
                center + vec2(dx, -dy) * travel,
                stick.radius / 3.0,
                outline.color,
            );
            let hit = hit.on_hover_text("Click, then move the stick to bind its axes");
            if hit.clicked() {
                clicked = Some(Target::Analog { right });
            }
        }
        if let Some(target) = clicked {
            self.start(target);
        }
    }
}

const LEFT_BUTTONS: [NpadButton; 12] = {
    use NpadButton::*;
    [
        ZL, L, Minus, Capture, LStick, DPadUp, DPadDown, DPadLeft, DPadRight, LeftSL, LeftSR, GL,
    ]
};

const RIGHT_BUTTONS: [NpadButton; 13] = {
    use NpadButton::*;
    [
        ZR, R, Plus, Home, RStick, A, B, X, Y, C, RightSL, RightSR, GR,
    ]
};

/// Stores a digital binding for a button or stick direction.
fn bind(player: &mut PlayerInput, target: Target, binding: InputBinding) {
    match target {
        Target::Button(button) => {
            player.buttons.insert(button, binding);
        }
        Target::Stick { right, dir } => {
            let stick = if right {
                &mut player.right_stick
            } else {
                &mut player.left_stick
            };
            *dir.binding(stick) = binding;
        }
        Target::Motion => player.motion_button = binding,
        Target::Analog { .. } => {}
    }
}

/// Edits a button's binding and options together, storing neither while it is unset.
fn edit_button(
    player: &mut PlayerInput,
    button: NpadButton,
    edit: impl FnOnce(&mut InputBinding, &mut ButtonOptions),
) {
    let mut binding = player.buttons.get(&button).cloned().unwrap_or_default();
    let mut options = player
        .button_options
        .get(&button)
        .copied()
        .unwrap_or_default();
    edit(&mut binding, &mut options);
    if binding == InputBinding::None {
        player.buttons.remove(&button);
    } else if player.buttons.get(&button) != Some(&binding) {
        player.buttons.insert(button, binding);
    }
    if options == ButtonOptions::default() {
        player.button_options.remove(&button);
    } else {
        player.button_options.insert(button, options);
    }
}

/// Lichii's binding context menu: clear, invert, toggle and turbo for digital buttons, and the
/// press threshold of an analog axis bound as a digital input. GameCube's analog triggers read
/// their binding directly, so they get neither.
fn binding_menu(
    response: &Response,
    binding: &mut InputBinding,
    options: Option<&mut ButtonOptions>,
    threshold: bool,
) {
    response.context_menu(|ui| {
        let cleared = ui.button("Clear").clicked();
        if let Some(options) = options {
            ui.checkbox(&mut options.inverted, "Invert button");
            ui.checkbox(&mut options.toggle, "Toggle button");
            ui.checkbox(&mut options.turbo, "Turbo button");
            if cleared {
                *options = ButtonOptions::default();
            }
        }
        if let (true, InputBinding::GamepadAxis { threshold, .. }) = (threshold, &mut *binding) {
            ui.add(egui::Slider::new(threshold, 0.05..=0.95).text("Threshold"));
        }
        if cleared {
            *binding = InputBinding::None;
            ui.close();
        }
    });
}

/// Lichii's markers before a binding: `$` turbo, `~` toggle, `!` inverted.
fn options_prefix(options: ButtonOptions) -> String {
    [
        (options.turbo, '$'),
        (options.toggle, '~'),
        (options.inverted, '!'),
    ]
    .into_iter()
    .filter_map(|(on, mark)| on.then_some(mark))
    .collect()
}

/// Lichii keeps players connected in order: connecting a player connects every player before it,
/// and disconnecting one disconnects every player after it. Unchecking a player while later ones
/// are connected keeps it as the last connected player.
fn set_connected(players: &mut [PlayerInput], index: usize, connected: bool) {
    let keep = !connected && players.get(index + 1).is_some_and(|p| p.connected);
    for (i, player) in players.iter_mut().enumerate() {
        if connected && i <= index {
            player.connected = true;
        } else if !connected && i >= index {
            player.connected = i == index && keep;
        }
    }
}

fn trigger_pressed(controller: ControllerType, button: NpadButton, state: &NpadState) -> bool {
    controller_layout::analog_trigger(controller, button)
        && match button {
            NpadButton::ZL => state.left_trigger > 0,
            _ => state.right_trigger > 0,
        }
}

/// Lichii renames buttons for the GameCube layout.
fn button_label(controller: ControllerType, button: NpadButton) -> &'static str {
    match (controller, button) {
        (ControllerType::GameCube, NpadButton::ZL) => "L (trigger)",
        (ControllerType::GameCube, NpadButton::ZR) => "R (trigger)",
        (ControllerType::GameCube, NpadButton::R) => "Z",
        (ControllerType::GameCube, NpadButton::Plus) => "Start / Pause",
        _ => button.label(),
    }
}

fn short_label(controller: ControllerType, button: NpadButton) -> &'static str {
    use NpadButton::*;
    match button {
        Minus => "−",
        Plus => "+",
        Home => "H",
        LStick | RStick => "press",
        LeftSL | RightSL => "SL",
        LeftSR | RightSR => "SR",
        _ => button_label(controller, button),
    }
}

fn target_label(target: Target) -> String {
    match target {
        Target::Button(b) => b.label().into(),
        Target::Stick { right, dir } => {
            format!(
                "{} stick {}",
                if right { "right" } else { "left" },
                dir.label()
            )
        }
        Target::Analog { right } => format!("{} stick", if right { "right" } else { "left" }),
        Target::Motion => "Motion".into(),
    }
}

/// How a binding reads in the UI.
pub fn binding_label(binding: &InputBinding) -> String {
    match binding {
        InputBinding::None => "—".into(),
        InputBinding::Key(name) => Key::from_name(name).map_or(name.clone(), |k| k.label().into()),
        InputBinding::MouseButton(name) => format!("Mouse {name}"),
        InputBinding::GamepadButton { button } => {
            let pretty = match PadButton::parse(button) {
                Some(PadButton::South) => "South",
                Some(PadButton::East) => "East",
                Some(PadButton::West) => "West",
                Some(PadButton::North) => "North",
                _ => button.as_str(),
            };
            format!("Pad {pretty}")
        }
        InputBinding::GamepadAxis {
            axis, direction, ..
        } => {
            let trigger = matches!(
                PadAxis::from_name(axis),
                Some(PadAxis::LeftTrigger | PadAxis::RightTrigger)
            );
            if trigger {
                format!("Pad {axis}")
            } else {
                format!("Pad {axis}{}", if *direction < 0 { "−" } else { "+" })
            }
        }
    }
}

fn percent_slider(ui: &mut Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, text: &str) {
    let mut percent = (*value * 100.0).round();
    let range = (range.start() * 100.0)..=(range.end() * 100.0);
    if ui
        .add(
            egui::Slider::new(&mut percent, range)
                .suffix("%")
                .text(text),
        )
        .changed()
    {
        *value = percent / 100.0;
    }
}

fn stick_preview(ui: &mut Ui, stick: &StickBinding, position: Option<(i32, i32)>) {
    let (response, painter) = ui.allocate_painter(vec2(90.0, 90.0), Sense::hover());
    let center = response.rect.center();
    let radius = 40.0;
    let stroke = ui.visuals().widgets.inactive.fg_stroke;
    painter.circle_stroke(center, radius, stroke);
    painter.circle_stroke(
        center,
        radius * stick.deadzone,
        Stroke::new(1.0, Color32::from_gray(90)),
    );
    painter.line_segment(
        [center - vec2(radius, 0.0), center + vec2(radius, 0.0)],
        Stroke::new(0.5, Color32::from_gray(70)),
    );
    painter.line_segment(
        [center - vec2(0.0, radius), center + vec2(0.0, radius)],
        Stroke::new(0.5, Color32::from_gray(70)),
    );
    if let Some((x, y)) = position {
        let p: Pos2 = center
            + vec2(
                x as f32 / STICK_MAX as f32 * radius,
                -(y as f32) / STICK_MAX as f32 * radius,
            );
        painter.circle_filled(p, 4.0, ACCENT);
    }
}

fn pad_index(player: &PlayerInput, host: Option<&HostInput>) -> Option<usize> {
    match &player.device {
        InputDevice::Gamepad { guid, index, .. } => host?.find_gamepad(guid, *index),
        InputDevice::None | InputDevice::Keyboard => None,
    }
}

fn device_label(device: &InputDevice) -> String {
    match device {
        InputDevice::None => "None".into(),
        InputDevice::Keyboard => "Keyboard / Mouse".into(),
        InputDevice::Gamepad { name, index, .. } => format!("{name} #{}", index + 1),
    }
}

fn device_combo(ui: &mut Ui, player: &mut PlayerInput, env: &ConfigEnv) {
    let mut choices = vec![InputDevice::None, InputDevice::Keyboard];
    choices.extend(env.devices.iter().map(|g| InputDevice::Gamepad {
        guid: g.guid.clone(),
        index: g.index,
        name: g.name.clone(),
    }));
    let current_connected = choices.iter().any(|c| same_device(c, &player.device));
    let mut selected = player.device.clone();
    let mut text = device_label(&player.device);
    if !current_connected {
        text.push_str(" (disconnected)");
    }
    egui::ComboBox::from_id_salt("device")
        .selected_text(text)
        .width(240.0)
        .show_ui(ui, |ui| {
            for choice in choices {
                let label = device_label(&choice);
                let on = same_device(&choice, &selected);
                if ui.selectable_label(on, label).clicked() {
                    selected = choice;
                }
            }
        });
    if !same_device(&selected, &player.device) {
        player.device = selected;
        restore_defaults(player);
    }
}

fn same_device(a: &InputDevice, b: &InputDevice) -> bool {
    match (a, b) {
        (
            InputDevice::Gamepad {
                guid: ga,
                index: ia,
                ..
            },
            InputDevice::Gamepad {
                guid: gb,
                index: ib,
                ..
            },
        ) => ga == gb && ia == ib,
        _ => a == b,
    }
}

/// Resets the mapping to the device's defaults, keeping the player's other choices (as Lichii
/// does when the device changes).
fn restore_defaults(player: &mut PlayerInput) {
    let defaults = match &player.device {
        InputDevice::Keyboard => PlayerInput::default_keyboard(),
        InputDevice::Gamepad { .. } => PlayerInput::default_gamepad(player.device.clone()),
        InputDevice::None => PlayerInput::default(),
    };
    player.buttons = defaults.buttons;
    player.left_stick = defaults.left_stick;
    player.right_stick = defaults.right_stick;
    player.button_options = defaults.button_options;
    player.motion_button = defaults.motion_button;
    player.profile = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(binding_label(&InputBinding::Key("KeyZ".into())), "Z");
        assert_eq!(
            binding_label(&InputBinding::GamepadButton { button: "b".into() }),
            "Pad East"
        );
        assert_eq!(
            binding_label(&InputBinding::GamepadAxis {
                axis: "lefty".into(),
                threshold: 0.5,
                direction: -1
            }),
            "Pad lefty−"
        );
        assert_eq!(button_label(ControllerType::GameCube, NpadButton::R), "Z");
    }

    #[test]
    fn device_change_restores_defaults_but_keeps_choices() {
        let mut player = PlayerInput::default_keyboard();
        player.controller_type = ControllerType::Switch2ProController;
        player.vibration_strength = 30;
        player.device = InputDevice::Gamepad {
            guid: "g".into(),
            index: 0,
            name: "Pad".into(),
        };
        restore_defaults(&mut player);
        assert_eq!(
            player.buttons.get(&NpadButton::A),
            Some(&InputBinding::GamepadButton { button: "b".into() })
        );
        assert_eq!(player.left_stick.analog_x.as_deref(), Some("leftx"));
        assert_eq!(player.controller_type, ControllerType::Switch2ProController);
        assert_eq!(player.vibration_strength, 30);
    }

    #[test]
    fn every_button_is_listed_on_one_side() {
        for button in NpadButton::ALL {
            assert!(
                LEFT_BUTTONS.contains(&button) ^ RIGHT_BUTTONS.contains(&button),
                "{button:?}"
            );
        }
    }

    #[test]
    fn players_connect_in_order() {
        let mut players = vec![PlayerInput::default(); 4];
        let connected = |players: &[PlayerInput]| -> Vec<bool> {
            players.iter().map(|p| p.connected).collect()
        };
        players.iter_mut().for_each(|p| p.connected = false);
        set_connected(&mut players, 2, true);
        assert_eq!(connected(&players), [true, true, true, false]);
        set_connected(&mut players, 1, false);
        assert_eq!(connected(&players), [true, true, false, false]);
        set_connected(&mut players, 1, false);
        assert_eq!(connected(&players), [true, false, false, false]);
    }

    #[test]
    fn button_options_are_stored_only_while_set() {
        let mut player = PlayerInput::default_keyboard();
        edit_button(&mut player, NpadButton::A, |_, options| {
            options.turbo = true
        });
        edit_button(&mut player, NpadButton::B, |_, options| {
            options.toggle = true;
            options.inverted = true;
        });
        assert_eq!(options_prefix(player.button_options[&NpadButton::A]), "$");
        assert_eq!(options_prefix(player.button_options[&NpadButton::B]), "~!");
        edit_button(&mut player, NpadButton::A, |binding, options| {
            *binding = InputBinding::None;
            *options = ButtonOptions::default();
        });
        assert!(!player.buttons.contains_key(&NpadButton::A));
        assert!(!player.button_options.contains_key(&NpadButton::A));
        player.motion_button = InputBinding::Key("Digit7".into());
        restore_defaults(&mut player);
        assert!(player.button_options.is_empty());
        assert_eq!(player.motion_button, InputBinding::None);
    }
}
