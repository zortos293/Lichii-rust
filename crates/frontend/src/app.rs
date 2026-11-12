//! The application: window, egui, emulation session, input thread and hotkeys.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_common::paths::UserPaths;
use lichii_common::perf::PerfStats;
use lichii_common::settings::{FullscreenMode, HotkeyAction, Settings, Theme};
use lichii_input::{Key, KeyChord, KeySet, MouseButton};
use lichii_vulkan::present::{UiFrame, Viewport};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::keyboard::PhysicalKey;
use winit::window::{Fullscreen, Window, WindowId};

use crate::choices::Choice;
use crate::cli::Cli;
use crate::game_list::GameList;
use crate::hotkeys::{self, HotkeyMap};
use crate::input_worker::{InputMsg, InputWorker};
use crate::presenter::Presenter;
use crate::session::Session;

/// Events other threads send to the UI thread.
#[derive(Debug)]
pub enum UserEvent {
    /// A controller hotkey combo was pressed; combos act only while a session exists, so menu
    /// navigation on a controller never exits or toggles settings.
    Hotkey(HotkeyAction),
    /// Background work finished (game list scan); redraw.
    Wake,
}

/// Something the UI or a hotkey asked for, applied after the frame is drawn.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    Hotkey(HotkeyAction),
    Boot(PathBuf),
    StopConfirmed,
    AddGameDir,
    RefreshGames,
    ClearRecent,
    About,
}

/// Dialogs and transient messages.
#[derive(Default)]
pub struct UiState {
    pub error: Option<String>,
    pub about: bool,
    pub confirm_stop: bool,
    pub toast: Option<(String, Instant)>,
    pub selected: Option<PathBuf>,
}

pub const TOAST_TIME: Duration = Duration::from_secs(3);

pub struct App {
    pub settings: Settings,
    /// Where settings are saved; None when an unreadable file could not be backed up.
    save_to: Option<PathBuf>,
    proxy: EventLoopProxy<UserEvent>,
    window: Option<Arc<Window>>,
    presenter: Option<Presenter>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    pub games: GameList,
    pub session: Option<Session>,
    input: InputWorker,
    hotkeys: HotkeyMap,
    held_keys: KeySet,
    pub perf: Arc<PerfStats>,
    pub ui: UiState,
    /// Where the renderer draws the guest frame, in physical pixels.
    pub viewport: Option<Viewport>,
    pending_boot: Option<PathBuf>,
    start_fullscreen: bool,
    /// The speed limit to restore when the speed limit hotkey turns it back on.
    saved_speed_limit: u16,
    paused_for_focus: bool,
    exit_requested: bool,
    /// What the input thread was last told about guest input.
    guest_input: bool,
}

impl App {
    pub fn new(
        paths: &UserPaths,
        settings: Settings,
        save_to: Option<PathBuf>,
        cli: Cli,
        proxy: EventLoopProxy<UserEvent>,
    ) -> Self {
        let hotkey_proxy = proxy.clone();
        let input = InputWorker::spawn(
            settings.controls.clone(),
            hotkeys::combos(&settings.ui.hotkeys),
            move |action| {
                let _ = hotkey_proxy.send_event(UserEvent::Hotkey(action));
            },
        );
        tracing::debug!(
            "input profiles live in {}",
            paths.input_profiles_dir().display()
        );
        let mut app = Self {
            hotkeys: HotkeyMap::new(&settings.ui.hotkeys),
            saved_speed_limit: settings.graphics.speed_limit.unwrap_or(100),
            settings,
            save_to,
            proxy,
            window: None,
            presenter: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            games: GameList::default(),
            session: None,
            input,
            held_keys: KeySet::default(),
            perf: Arc::new(PerfStats::default()),
            ui: UiState::default(),
            viewport: None,
            pending_boot: cli.game,
            start_fullscreen: cli.fullscreen,
            paused_for_focus: false,
            exit_requested: false,
            guest_input: false,
        };
        app.refresh_games();
        app
    }

    pub fn adapter_name(&self) -> Option<&str> {
        self.presenter.as_ref().map(Presenter::adapter_name)
    }

    pub fn is_fullscreen(&self) -> bool {
        self.window
            .as_ref()
            .is_some_and(|w| w.fullscreen().is_some())
    }

    /// The hotkey's chord for menu labels, e.g. `Ctrl+O`.
    pub fn shortcut(&self, action: HotkeyAction) -> String {
        self.settings
            .ui
            .hotkeys
            .get(&action)
            .and_then(|h| KeyChord::parse(&h.keys))
            .map(KeyChord::label)
            .unwrap_or_default()
    }

    pub fn toast(&mut self, text: impl Into<String>) {
        self.ui.toast = Some((text.into(), Instant::now()));
    }

    pub fn current_toast(&self) -> Option<&str> {
        self.ui
            .toast
            .as_ref()
            .filter(|(_, at)| at.elapsed() < TOAST_TIME)
            .map(|(text, _)| text.as_str())
    }

    fn refresh_games(&mut self) {
        let proxy = self.proxy.clone();
        self.games
            .refresh(self.settings.paths.game_dirs.clone(), move || {
                let _ = proxy.send_event(UserEvent::Wake);
            });
    }

    fn save_settings(&self) {
        let Some(path) = &self.save_to else {
            return;
        };
        if let Err(e) = self.settings.save(path) {
            tracing::error!("saving settings: {e}");
        }
    }

    /// Re-applies settings that live outside `self.settings`: input bindings and hotkeys.
    fn reconfigure_input(&mut self) {
        self.hotkeys = HotkeyMap::new(&self.settings.ui.hotkeys);
        self.input.send(InputMsg::Configure {
            controls: Box::new(self.settings.controls.clone()),
            combos: hotkeys::combos(&self.settings.ui.hotkeys),
        });
    }

    fn update_title(&self) {
        let Some(window) = &self.window else { return };
        let version = env!("CARGO_PKG_VERSION");
        let title = match &self.session {
            Some(s) if s.is_paused() => format!("Lichii {version} | {} (paused)", s.title),
            Some(s) => format!("Lichii {version} | {}", s.title),
            None => format!("Lichii {version}"),
        };
        window.set_title(&title);
    }

    fn set_fullscreen(&mut self, on: bool) {
        let Some(window) = &self.window else { return };
        let mode = on.then(|| match self.settings.graphics.fullscreen {
            FullscreenMode::Borderless => Fullscreen::Borderless(None),
            FullscreenMode::Exclusive => window
                .current_monitor()
                .and_then(|m| {
                    m.video_modes().max_by_key(|v| {
                        (
                            v.size().width * v.size().height,
                            v.refresh_rate_millihertz(),
                        )
                    })
                })
                .map_or(Fullscreen::Borderless(None), Fullscreen::Exclusive),
        });
        window.set_fullscreen(mode);
    }

    fn boot(&mut self, path: PathBuf) {
        self.session = None;
        match Session::boot(&path) {
            Ok(session) => {
                self.session = Some(session);
                self.settings.ui.push_recent(path);
                self.save_settings();
            }
            Err(e) => self.ui.error = Some(format!("Cannot boot: {e}")),
        }
        self.input.send(InputMsg::ReleaseAll);
        self.update_title();
    }

    fn stop(&mut self) {
        self.session = None;
        self.viewport = None;
        self.input.send(InputMsg::ReleaseAll);
        if self.is_fullscreen() {
            self.set_fullscreen(false);
        }
        self.update_title();
    }

    pub fn apply(&mut self, action: Action) {
        match action {
            Action::Hotkey(hotkey) => self.hotkey(hotkey),
            Action::Boot(path) => self.boot(path),
            Action::StopConfirmed => self.stop(),
            Action::AddGameDir => {
                if let Some(dir) = rfd::FileDialog::new()
                    .set_title("Add game directory")
                    .pick_folder()
                {
                    if !self.settings.paths.game_dirs.contains(&dir) {
                        self.settings.paths.game_dirs.push(dir);
                        self.save_settings();
                    }
                    self.refresh_games();
                }
            }
            Action::RefreshGames => self.refresh_games(),
            Action::ClearRecent => {
                self.settings.ui.recent_files.clear();
                self.save_settings();
            }
            Action::About => self.ui.about = true,
        }
    }

    fn hotkey(&mut self, hotkey: HotkeyAction) {
        match hotkey {
            HotkeyAction::ContinuePause => {
                if let Some(session) = &mut self.session {
                    session.toggle_pause();
                    self.paused_for_focus = false;
                    self.input.send(InputMsg::ReleaseAll);
                    self.update_title();
                }
            }
            HotkeyAction::Stop => {
                if self.session.is_some() {
                    if self.settings.ui.confirm_stop {
                        self.ui.confirm_stop = true;
                    } else {
                        self.stop();
                    }
                }
            }
            HotkeyAction::Restart => {
                if let Some(path) = self.session.as_ref().map(|s| s.path.clone()) {
                    self.boot(path);
                }
            }
            HotkeyAction::LoadFile => {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Load file")
                    .add_filter("Switch titles", &["nsp", "xci", "nca", "nro", "nso"])
                    .add_filter("All files", &["*"])
                    .pick_file()
                {
                    self.boot(path);
                }
            }
            HotkeyAction::Configure => {
                self.toast("The configuration window is not built yet");
            }
            HotkeyAction::Fullscreen => self.set_fullscreen(!self.is_fullscreen()),
            HotkeyAction::ExitFullscreen => {
                if self.is_fullscreen() {
                    self.set_fullscreen(false);
                }
            }
            HotkeyAction::ToggleDocked => {
                let docked = !self.settings.system.docked;
                self.settings.system.docked = docked;
                self.toast(if docked {
                    "Docked mode"
                } else {
                    "Handheld mode"
                });
                self.save_settings();
            }
            HotkeyAction::ToggleSpeedLimit => {
                let g = &mut self.settings.graphics;
                g.speed_limit = match g.speed_limit {
                    Some(limit) => {
                        self.saved_speed_limit = limit;
                        None
                    }
                    None => Some(self.saved_speed_limit),
                };
                let text = match g.speed_limit {
                    Some(limit) => format!("Speed limit {limit}%"),
                    None => "Speed limit off".into(),
                };
                self.toast(text);
                self.save_settings();
            }
            HotkeyAction::ToggleFrameView => {
                self.settings.frame_view.enabled ^= true;
                self.save_settings();
            }
            HotkeyAction::ToggleStatusBar => {
                self.settings.ui.show_status_bar ^= true;
                self.save_settings();
            }
            HotkeyAction::ToggleMousePanning => {
                let on = !self.settings.controls.mouse_panning;
                self.settings.controls.mouse_panning = on;
                self.toast(if on {
                    "Mouse panning on"
                } else {
                    "Mouse panning off"
                });
                self.reconfigure_input();
                self.save_settings();
            }
            HotkeyAction::CycleScalingFilter => {
                let g = &mut self.settings.graphics;
                g.scaling_filter = g.scaling_filter.next();
                let text = format!("Scaling filter: {}", g.scaling_filter.label());
                self.toast(text);
                self.save_settings();
            }
            HotkeyAction::CycleGpuAccuracy => {
                let g = &mut self.settings.graphics;
                g.gpu_accuracy = g.gpu_accuracy.next();
                let text = format!("GPU accuracy: {}", g.gpu_accuracy.label());
                self.toast(text);
                self.save_settings();
            }
            HotkeyAction::Exit => self.exit_requested = true,
        }
    }

    /// Tells the input thread whether the guest should see input: only while a title runs.
    fn sync_guest_input(&mut self) {
        let on = self.session.as_ref().is_some_and(|s| !s.is_paused());
        if on != self.guest_input {
            self.guest_input = on;
            self.input.send(InputMsg::GuestInput(on));
        }
    }

    fn on_key(&mut self, event: &KeyEvent) {
        let PhysicalKey::Code(code) = event.physical_key else {
            return;
        };
        let Some(key) = Key::from_winit(code) else {
            return;
        };
        let down = event.state == ElementState::Pressed;
        self.held_keys.set(key, down);
        if down
            && !event.repeat
            && !self.egui_ctx.egui_wants_keyboard_input()
            && let Some(chord) = KeyChord::from_press(key, &self.held_keys)
            && let Some(action) = self.hotkeys.action(chord)
        {
            self.apply(Action::Hotkey(action));
            return;
        }
        if !down || !self.egui_ctx.egui_wants_keyboard_input() {
            self.input.send(InputMsg::Key(key, down));
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(mut egui_state)) = (self.window.clone(), self.egui_state.take())
        else {
            return;
        };
        self.games.poll();
        let raw = egui_state.take_egui_input(&window);
        let ctx = self.egui_ctx.clone();
        let mut actions = Vec::new();
        let output = ctx.run_ui(raw, |ui| self.draw(ui, &mut actions));
        egui_state.handle_platform_output(&window, output.platform_output);
        self.egui_state = Some(egui_state);

        let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
        if let Some(presenter) = &mut self.presenter {
            presenter.present(UiFrame {
                primitives,
                textures_delta: output.textures_delta,
                pixels_per_point: output.pixels_per_point,
            });
        }
        for action in actions {
            self.apply(action);
        }
        self.sync_guest_input();

        let running = self.session.as_ref().is_some_and(|s| !s.is_paused());
        let delay = output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map_or(Duration::MAX, |v| v.repaint_delay);
        if running || delay.is_zero() {
            event_loop.set_control_flow(ControlFlow::Wait);
            window.request_redraw();
        } else if let Some(at) = Instant::now().checked_add(delay) {
            event_loop.set_control_flow(ControlFlow::WaitUntil(at));
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    fn apply_theme(&self) {
        self.egui_ctx.set_theme(match self.settings.ui.theme {
            Theme::Dark => egui::ThemePreference::Dark,
            Theme::Light => egui::ThemePreference::Light,
        });
    }

    fn create_window(&mut self, event_loop: &ActiveEventLoop) -> Result<(), String> {
        let ui = &self.settings.ui;
        let attributes = Window::default_attributes()
            .with_title(format!("Lichii {}", env!("CARGO_PKG_VERSION")))
            .with_inner_size(winit::dpi::LogicalSize::new(
                ui.window_width.max(640),
                ui.window_height.max(400),
            ))
            .with_min_inner_size(winit::dpi::LogicalSize::new(640, 400))
            .with_maximized(ui.window_maximized);
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .map_err(|e| format!("creating the window: {e}"))?,
        );
        let presenter = Presenter::new(
            window.clone(),
            self.settings.graphics.present_mode,
            self.perf.clone(),
        )?;
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(8192),
        ));
        self.apply_theme();
        self.presenter = Some(presenter);
        self.window = Some(window);
        Ok(())
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if let StartCause::ResumeTimeReached { .. } = cause {
            event_loop.set_control_flow(ControlFlow::Wait);
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        if let Err(e) = self.create_window(event_loop) {
            tracing::error!("{e}");
            rfd::MessageDialog::new()
                .set_title("Lichii")
                .set_description(&e)
                .set_level(rfd::MessageLevel::Error)
                .show();
            event_loop.exit();
            return;
        }
        if let Some(path) = self.pending_boot.take() {
            self.boot(path);
        }
        if self.start_fullscreen {
            self.set_fullscreen(true);
        }
        self.update_title();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        let response = self
            .egui_state
            .as_mut()
            .map(|s| s.on_window_event(&window, &event));
        if response.as_ref().is_some_and(|r| r.repaint) {
            window.request_redraw();
        }
        let consumed = response.is_some_and(|r| r.consumed);
        match event {
            WindowEvent::CloseRequested => self.exit_requested = true,
            WindowEvent::Resized(size) => {
                if let Some(presenter) = &mut self.presenter {
                    presenter.resize(size.width, size.height);
                }
                let ui = &mut self.settings.ui;
                ui.window_maximized = window.is_maximized();
                if !ui.window_maximized && window.fullscreen().is_none() {
                    let logical = size.to_logical::<u32>(window.scale_factor());
                    ui.window_width = logical.width;
                    ui.window_height = logical.height;
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            WindowEvent::KeyboardInput { event, .. } => self.on_key(&event),
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(button) = MouseButton::from_winit(button) {
                    let down = state == ElementState::Pressed;
                    if !down || !consumed {
                        self.input.send(InputMsg::Mouse(button, down));
                    }
                }
            }
            WindowEvent::Focused(focused) => {
                if !focused {
                    self.held_keys.clear();
                    self.input.send(InputMsg::ReleaseAll);
                }
                if self.settings.ui.pause_when_unfocused
                    && let Some(session) = &mut self.session
                {
                    if !focused && !session.is_paused() {
                        session.pause();
                        self.paused_for_focus = true;
                    } else if focused && self.paused_for_focus {
                        session.resume();
                        self.paused_for_focus = false;
                    }
                    self.update_title();
                }
            }
            WindowEvent::DroppedFile(path) => self.boot(path),
            _ => {}
        }
        if self.exit_requested {
            event_loop.exit();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Hotkey(action) if self.session.is_some() => {
                self.apply(Action::Hotkey(action));
            }
            UserEvent::Hotkey(_) => {}
            UserEvent::Wake => self.games.poll(),
        }
        self.sync_guest_input();
        if let Some(window) = &self.window {
            window.request_redraw();
        }
        if self.exit_requested {
            event_loop.exit();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.session = None;
        self.save_settings();
    }
}
