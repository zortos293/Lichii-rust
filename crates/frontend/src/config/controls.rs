//! The controls page, like Lichii's player configuration: players 1-8, controller type, device,
//! a live controller diagram with click-to-bind, sticks, motion, vibration and profiles.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use egui::{Color32, Pos2, Rect, RichText, Sense, Stroke, Ui, vec2};
use lichii_common::settings::{
    ControllerType, InputBinding, InputDevice, NpadButton, PlayerInput, Settings, StickBinding,
};
use lichii_input::capture::{detect_press, detect_stick};
use lichii_input::{
    HostInput, Key, MouseButton, NpadButtons, NpadState, PadAxis, PadButton, PlayerMapping,
    ProfileStore, STICK_MAX, Source,
};

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
        }
    }

    pub fn ui(&mut self, ui: &mut Ui, d: &mut Settings, env: &ConfigEnv, out: &mut ConfigOutput) {
        section(ui, "Console");
        grid(ui, "console mode", |ui| {
            row(ui, "Console mode", |ui| {
                docked_toggle(ui, &mut d.system.docked)
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
        ui.horizontal_wrapped(|ui| {
            for (i, player) in d.controls.players.iter().enumerate() {
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
            }
        });
        ui.separator();

        let keyboard = d.controls.keyboard_enabled;
        let docked = d.system.docked;
        let player = &mut d.controls.players[self.player];
        self.poll_capture(player, env.host);

        ui.horizontal(|ui| {
            ui.checkbox(&mut player.connected, "Connected");
            ui.label("Controller");
            combo(ui, "controller type", &mut player.controller_type);
            ui.label("Device");
            device_combo(ui, player, env);
        });
        if self.player == 0 && !docked && !player.controller_type.is_handheld() {
            note(
                ui,
                "In handheld mode the console reads player 1 as the attached Joy-Cons; pick Handheld for player 1.",
            );
        }
        self.profiles_row(ui, player);

        let live = env.host.map(|host| {
            let state = PlayerMapping::compile(player, host).resolve(host, keyboard);
            (state, host)
        });
        let state = live.map(|(s, _)| s);
        let pad = pad_index(player, env.host);

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
                        None => "needs a gamepad with a gyro".into(),
                    };
                    ui.label(RichText::new(text).weak().small());
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
                        player.left_stick = StickBinding::default();
                        player.right_stick = StickBinding::default();
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
                    if ui.button("Cancel binding").clicked() {
                        self.capture = None;
                    }
                }
                if env.host.is_none() {
                    note(ui, "Waiting for the input thread…");
                }
                note(
                    ui,
                    "Click a button or stick to bind it, right-click to clear it.",
                );
            });
            ui.vertical(|ui| {
                ui.set_width(250.0);
                self.stick_ui(ui, player, true, state.as_ref());
                self.buttons_ui(ui, player, &RIGHT_BUTTONS);
            });
        });
        ui.ctx().request_repaint_after(Duration::from_millis(16));
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
        if source == Source::Mouse(MouseButton::Left) {
            cap.baseline = Some(host.clone());
            return;
        }
        bind(player, target, source.to_binding());
        self.capture = None;
    }

    /// A key press the window saw while capturing. Keys come straight from the window because a
    /// quick tap can start and end between two host-input snapshots. True when it was bound.
    pub fn on_key(&mut self, key: Key, d: &mut Settings) -> bool {
        let Some(cap) = &self.capture else {
            return false;
        };
        if matches!(cap.target, Target::Analog { .. }) {
            return false;
        }
        bind(
            &mut d.controls.players[self.player],
            cap.target,
            Source::Key(key).to_binding(),
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

    /// A button showing a binding; click starts capture, right-click clears. True when cleared.
    fn bind_button(&mut self, ui: &mut Ui, binding: &InputBinding, target: Target) -> bool {
        let text = if self.capturing(target) {
            RichText::new("…").color(ACCENT)
        } else {
            RichText::new(binding_label(binding))
        };
        let response = ui.add_sized([130.0, 20.0], egui::Button::new(text));
        if response.clicked() {
            self.start(target);
        }
        response.secondary_clicked()
    }

    fn buttons_ui(&mut self, ui: &mut Ui, player: &mut PlayerInput, buttons: &[NpadButton]) {
        let controller = player.controller_type;
        egui::Grid::new(("buttons", buttons[0]))
            .num_columns(2)
            .spacing([8.0, 4.0])
            .show(ui, |ui| {
                for &button in buttons {
                    if !controller.has_button(button) && !trigger_on(controller, button) {
                        continue;
                    }
                    ui.label(button_label(controller, button));
                    let binding = player.buttons.get(&button).cloned().unwrap_or_default();
                    if self.bind_button(ui, &binding, Target::Button(button)) {
                        player.buttons.remove(&button);
                    }
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
        let mut clear = None;
        {
            let stick = if right {
                &player.right_stick
            } else {
                &player.left_stick
            };
            let snapshot = stick.clone();
            ui.horizontal(|ui| {
                stick_preview(ui, &snapshot, position.map(|p| (p.x, p.y)));
                egui::Grid::new(("stick", right))
                    .num_columns(2)
                    .spacing([6.0, 3.0])
                    .show(ui, |ui| {
                        for dir in Dir::ALL {
                            ui.label(dir.label());
                            let mut copy = snapshot.clone();
                            let binding = dir.binding(&mut copy).clone();
                            if self.bind_button(ui, &binding, Target::Stick { right, dir }) {
                                clear = Some(dir);
                            }
                            ui.end_row();
                        }
                    });
            });
        }
        let stick = if right {
            &mut player.right_stick
        } else {
            &mut player.left_stick
        };
        if let Some(dir) = clear {
            *dir.binding(stick) = InputBinding::None;
        }
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
                .on_hover_text("Click, then move the stick. Right-click clears.");
            if response.clicked() {
                self.start(Target::Analog { right });
            }
            if response.secondary_clicked() {
                stick.analog_x = None;
                stick.analog_y = None;
            }
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
        let size = vec2(380.0, 250.0);
        let (response, painter) = ui.allocate_painter(size, Sense::hover());
        let origin = response.rect.min;
        let at = |x: f32, y: f32| origin + vec2(x, y);
        let visuals = ui.visuals().clone();
        let body = visuals.widgets.inactive.bg_fill;
        let outline = Stroke::new(1.5, visuals.widgets.inactive.fg_stroke.color);
        painter.rect(
            Rect::from_min_max(at(30.0, 46.0), at(350.0, 210.0)),
            70.0,
            body,
            outline,
            egui::StrokeKind::Inside,
        );
        let controller = player.controller_type;
        let held = state.map_or(NpadButtons::empty(), |s| s.buttons);
        let mut clicked = None;
        let mut cleared = None;
        for &(button, x, y, w, h) in LAYOUT {
            let shown = controller.has_button(button) || trigger_on(controller, button);
            if !shown {
                continue;
            }
            let rect = Rect::from_center_size(at(x, y), vec2(w, h));
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
            let round = if w == h { w / 2.0 } else { 4.0 };
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
            let hit = hit.on_hover_text(format!(
                "{}: {}",
                button_label(controller, button),
                binding_label(&binding)
            ));
            if hit.clicked() {
                clicked = Some(Target::Button(button));
            }
            if hit.secondary_clicked() {
                cleared = Some(button);
            }
        }
        for (right, x, y) in [(false, 105.0, 95.0), (true, 245.0, 160.0)] {
            let has = if right {
                controller.has_right_stick()
            } else {
                controller.has_left_stick()
            };
            if !has {
                continue;
            }
            let center = at(x, y);
            let rect = Rect::from_center_size(center, vec2(56.0, 56.0));
            let hit = ui.interact(rect, ui.scope_id().with(("stick", right)), Sense::click());
            let ring = if self.capturing(Target::Analog { right }) || hit.hovered() {
                ACCENT
            } else {
                outline.color
            };
            painter.circle(
                center,
                27.0,
                visuals.widgets.inactive.weak_bg_fill,
                Stroke::new(1.5, ring),
            );
            let p = state.map(|s| if right { s.right_stick } else { s.left_stick });
            let (dx, dy) = p.map_or((0.0, 0.0), |p| {
                (p.x as f32 / STICK_MAX as f32, p.y as f32 / STICK_MAX as f32)
            });
            painter.circle_filled(center + vec2(dx * 18.0, -dy * 18.0), 9.0, outline.color);
            let hit = hit.on_hover_text("Click, then move the stick to bind its axes");
            if hit.clicked() {
                clicked = Some(Target::Analog { right });
            }
        }
        if let Some(target) = clicked {
            self.start(target);
        }
        if let Some(button) = cleared {
            player.buttons.remove(&button);
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

/// Button positions on the diagram: center x, y and size.
const LAYOUT: &[(NpadButton, f32, f32, f32, f32)] = {
    use NpadButton::*;
    &[
        (ZL, 85.0, 12.0, 70.0, 16.0),
        (L, 85.0, 32.0, 70.0, 16.0),
        (ZR, 295.0, 12.0, 70.0, 16.0),
        (R, 295.0, 32.0, 70.0, 16.0),
        (Minus, 160.0, 75.0, 18.0, 18.0),
        (Plus, 220.0, 75.0, 18.0, 18.0),
        (Capture, 160.0, 105.0, 16.0, 16.0),
        (Home, 220.0, 105.0, 20.0, 20.0),
        (C, 190.0, 135.0, 18.0, 18.0),
        (DPadUp, 135.0, 140.0, 18.0, 18.0),
        (DPadDown, 135.0, 180.0, 18.0, 18.0),
        (DPadLeft, 115.0, 160.0, 18.0, 18.0),
        (DPadRight, 155.0, 160.0, 18.0, 18.0),
        (X, 285.0, 72.0, 22.0, 22.0),
        (B, 285.0, 118.0, 22.0, 22.0),
        (Y, 262.0, 95.0, 22.0, 22.0),
        (A, 308.0, 95.0, 22.0, 22.0),
        (LStick, 105.0, 132.0, 30.0, 10.0),
        (RStick, 245.0, 197.0, 30.0, 10.0),
        (LeftSL, 22.0, 100.0, 12.0, 30.0),
        (LeftSR, 22.0, 150.0, 12.0, 30.0),
        (RightSL, 358.0, 150.0, 12.0, 30.0),
        (RightSR, 358.0, 100.0, 12.0, 30.0),
        (GL, 120.0, 232.0, 44.0, 14.0),
        (GR, 260.0, 232.0, 44.0, 14.0),
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
        Target::Analog { .. } => {}
    }
}

/// GameCube ZL/ZR are analog triggers: bindable and shown, but never digital buttons.
fn trigger_on(controller: ControllerType, button: NpadButton) -> bool {
    controller.has_analog_triggers() && matches!(button, NpadButton::ZL | NpadButton::ZR)
}

fn trigger_pressed(controller: ControllerType, button: NpadButton, state: &NpadState) -> bool {
    trigger_on(controller, button)
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
    fn layout_covers_every_button() {
        for button in NpadButton::ALL {
            assert!(LAYOUT.iter().any(|l| l.0 == button), "{button:?}");
            assert!(
                LEFT_BUTTONS.contains(&button) ^ RIGHT_BUTTONS.contains(&button),
                "{button:?}"
            );
        }
    }
}
