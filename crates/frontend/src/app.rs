//! The application: window, egui, emulation session, input thread and hotkeys.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_common::paths::UserPaths;
use lichii_common::perf::PerfStats;
use lichii_common::settings::{
    AntiAliasing, FullscreenMode, GraphicsSettings, HotkeyAction, ResolutionScale, ScalingFilter,
    Settings, Theme,
};
use lichii_core::{ExitReason, Host};
use lichii_crypto::KeySet as ConsoleKeys;
use lichii_input::{Key, KeyChord, KeySet, MouseButton};
use lichii_loader::LoadOptions;
use lichii_vulkan::present::{UiFrame, Viewport};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoopProxy};
use winit::keyboard::PhysicalKey;
use winit::window::{Fullscreen, Window, WindowId};

use crate::choices::Choice;
use crate::cli::Cli;
use crate::config::game::GameConfigWindow;
use crate::config::{ConfigEnv, ConfigWindow};
use crate::display::Display;
use crate::game_list::{self, GameList, ScanSources};
use crate::hotkeys::{self, HotkeyMap};
use crate::input_worker::{InputEvent, InputMsg, InputWorker};
use crate::install::{self, InstallJob};
use crate::per_game::{self, GameConfig};
use crate::session::Session;
use crate::setup::{self, SetupWindow};

/// Events other threads send to the UI thread.
#[derive(Debug)]
pub enum UserEvent {
    /// A controller hotkey combo was pressed; combos act only while a session exists and the
    /// configuration window is closed, so menu navigation and binding never exit or toggle.
    Hotkey(HotkeyAction),
    /// Background work finished (game list scan) or gamepads changed; redraw.
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
    SetResolution(ResolutionScale),
    SetScalingFilter(ScalingFilter),
    SetAntiAliasing(AntiAliasing),
    KeysAndFirmware,
    InstallFiles,
    /// Lichii's "Start Game without Custom Configuration".
    BootWithoutCustom(PathBuf),
    GameProperties {
        path: PathBuf,
        program_id: u64,
        title: String,
    },
    ConfigureCurrentGame,
    RemoveGameConfig {
        path: PathBuf,
        program_id: u64,
    },
}

/// The configuration of the running game, when it has one.
struct RunningConfig {
    config: GameConfig,
    /// The global settings with the game's overrides applied.
    settings: Settings,
}

/// Dialogs and transient messages.
#[derive(Default)]
pub struct UiState {
    pub error: Option<String>,
    /// A message with a title, such as the install results.
    pub info: Option<(String, String)>,
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
    display: Option<Display>,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    pub games: GameList,
    pub session: Option<Session>,
    input: InputWorker,
    hotkeys: HotkeyMap,
    held_keys: KeySet,
    pub perf: Arc<PerfStats>,
    pub ui: UiState,
    paths: UserPaths,
    /// The configuration window while it is open.
    config: Option<ConfigWindow>,
    /// The keys and firmware window while it is open.
    setup: Option<SetupWindow>,
    /// The per-game Properties window while it is open.
    game_config: Option<GameConfigWindow>,
    /// The running game's own configuration.
    running: Option<RunningConfig>,
    /// The running game was started without its custom configuration; a restart keeps that.
    without_custom: bool,
    /// File → Install Files to NAND while it runs.
    pub install: Option<InstallJob>,
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
            move |event| {
                let _ = hotkey_proxy.send_event(match event {
                    InputEvent::Hotkey(action) => UserEvent::Hotkey(action),
                    InputEvent::DevicesChanged => UserEvent::Wake,
                });
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
            display: None,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            games: GameList::default(),
            session: None,
            input,
            held_keys: KeySet::default(),
            perf: Arc::new(PerfStats::default()),
            ui: UiState::default(),
            paths: paths.clone(),
            config: None,
            setup: None,
            game_config: None,
            running: None,
            without_custom: false,
            install: None,
            viewport: None,
            pending_boot: cli.game,
            start_fullscreen: cli.fullscreen,
            paused_for_focus: false,
            exit_requested: false,
            guest_input: false,
        };
        app.refresh_games();
        if app.pending_boot.is_none() && !app.settings.ui.setup_done {
            let window = SetupWindow::new(&app.settings.paths, &app.paths);
            if !window.keys_usable() {
                app.setup = Some(window);
            }
        }
        app
    }

    pub fn adapter_name(&self) -> Option<&str> {
        self.display.as_ref().map(Display::adapter_name)
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
        let sources = ScanSources {
            dirs: self.settings.paths.game_dirs.clone(),
            keys_dir: setup::keys_dir(&self.settings.paths, &self.paths),
            cache_dir: self.paths.cache_dir.join("game_list"),
            language: game_list::nacp_language(self.settings.system.language),
            installed: self.installed_content(),
        };
        self.games.refresh(sources, move || {
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

    /// Draws the configuration window when it is open and applies what it returns.
    pub(crate) fn show_config(&mut self, ctx: &egui::Context) {
        let Some(config) = &mut self.config else {
            return;
        };
        let (devices, host) = {
            let shared = self
                .input
                .shared()
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            (shared.devices.clone(), shared.host.clone())
        };
        let env = ConfigEnv {
            devices: &devices,
            host: host.as_ref(),
            adapter: self.display.as_ref().map(Display::adapter_name),
            paths: &self.paths,
            running: self.session.is_some(),
        };
        let out = config.show(ctx, &env);
        if let Some(player) = out.test_rumble {
            self.input.send(InputMsg::TestRumble(player));
        }
        if let Some(settings) = out.apply {
            self.apply_settings(settings);
        }
        if out.close {
            self.config = None;
            self.input.send(InputMsg::Mirror(false));
        }
    }

    /// Replaces every setting (the configuration window's OK or Apply) and saves.
    /// The settings in effect: the running game's, when it has its own configuration.
    pub fn current(&self) -> &Settings {
        self.running
            .as_ref()
            .map_or(&self.settings, |r| &r.settings)
    }

    /// Where per-game configuration files go: `custom/` beside the settings file.
    fn config_dir(&self) -> PathBuf {
        self.save_to
            .as_deref()
            .and_then(Path::parent)
            .map_or_else(|| self.paths.config_dir.clone(), Path::to_path_buf)
    }

    fn keys(&self) -> Arc<ConsoleKeys> {
        setup::KeyStatus::check(setup::keys_dir(&self.settings.paths, &self.paths)).keys
    }

    /// The NAND's user content, where installed updates and DLC live.
    fn installed_content(&self) -> Vec<PathBuf> {
        vec![setup::nand_dir(&self.settings.paths, &self.paths).join("user/Contents/registered")]
    }

    /// File → Install Files to NAND: installs the chosen packages in the background.
    fn install_files(&mut self) {
        if self.install.is_some() {
            return;
        }
        let Some(files) = rfd::FileDialog::new()
            .set_title("Install Files to NAND")
            .add_filter("Nintendo Submission Package", &["nsp"])
            .pick_files()
            .filter(|files| !files.is_empty())
        else {
            return;
        };
        let registered = self.installed_content().remove(0);
        let proxy = self.proxy.clone();
        self.install = Some(InstallJob::start(
            files,
            registered,
            self.keys(),
            move || {
                let _ = proxy.send_event(UserEvent::Wake);
            },
        ));
    }

    /// Takes a finished install: shows its results and rescans the game list, whose versions
    /// and names now include the installed updates.
    fn poll_install(&mut self) {
        let Some(results) = self.install.as_ref().and_then(InstallJob::poll) else {
            return;
        };
        self.install = None;
        self.ui.info = Some(("Install Results".into(), install::summary(&results)));
        self.refresh_games();
    }

    /// Recomputes the running game's settings after the global ones or its configuration changed,
    /// and hands the graphics settings to the presenter.
    fn refresh_running(&mut self) {
        if let Some(running) = &mut self.running {
            running.settings = running.config.apply(&self.settings);
        }
        let graphics = self.current().graphics.clone();
        if let Some(display) = &mut self.display {
            display.set_settings(&graphics);
        }
    }

    fn apply_settings(&mut self, settings: Settings) {
        let rescan = settings.paths.game_dirs != self.settings.paths.game_dirs
            || settings.paths.keys_dir != self.settings.paths.keys_dir
            || settings.system.language != self.settings.system.language;
        let window_before = self.game_window_settings();
        self.settings = settings;
        self.refresh_running();
        self.follow_into_game_window(window_before);
        if let Some(session) = &self.session {
            session.set_audio_volume(&self.current().audio);
        }
        self.reconfigure_input();
        self.apply_theme();
        if rescan {
            self.refresh_games();
        }
        self.save_settings();
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

    /// Boots the title at `path`, with its custom configuration unless `without_custom`.
    fn boot(&mut self, path: PathBuf, without_custom: bool) {
        self.end_session();
        let config_dir = self.config_dir();
        let mut running = None;
        let configure = |program_id, title| {
            let config = GameConfig::load(GameConfig::path(&config_dir, program_id, &path));
            if !without_custom && !config.is_empty() {
                running = Some(RunningConfig {
                    settings: config.apply(&self.settings),
                    config,
                });
            }
            let settings = running.as_ref().map_or(&self.settings, |r| &r.settings);
            let host = Host {
                renderer: match &self.display {
                    Some(display) => display.renderer(
                        title,
                        &settings.graphics,
                        crate::display::shader_cache_dir(&self.paths, program_id),
                    ),
                    None => lichii_core::headless_renderer(),
                },
                npad: self.input.hub().clone(),
                perf: self.perf.clone(),
            };
            (Arc::new(settings.clone()), host)
        };
        let options = LoadOptions {
            keys: Some(self.keys()),
            installed: self.installed_content(),
            ..LoadOptions::default()
        };
        match Session::boot(&path, options, configure) {
            Ok(session) => {
                self.session = Some(session);
                self.without_custom = without_custom;
                if let Some(running) = &running {
                    let overridden = running.config.overridden();
                    tracing::info!(
                        file = %running.config.file().display(),
                        "custom configuration: {}",
                        overridden.into_iter().collect::<Vec<_>>().join(", ")
                    );
                    self.toast("Using this game's custom configuration");
                }
                self.running = running;
                self.refresh_running();
                self.settings.ui.push_recent(path);
                self.save_settings();
            }
            Err(e) => match e.missing_key() {
                Some(key) => {
                    self.ui.error = Some(format!(
                        "Lichii cannot decrypt this title: {key}.

Dump prod.keys and title.keys \
                         from your own console, then choose them in the Keys and Firmware window."
                    ));
                    self.open_setup();
                }
                None => self.ui.error = Some(format!("Cannot boot: {e}")),
            },
        }
        self.input.send(InputMsg::ReleaseAll);
        self.update_title();
    }

    /// Ends the session once its title has exited, saying why when it crashed.
    fn check_exit(&mut self) {
        let Some(reason) = self.session.as_ref().and_then(Session::exit_reason) else {
            return;
        };
        match reason {
            ExitReason::Exited => self.toast("The title exited"),
            ExitReason::Break { reason } => {
                self.ui.error = Some(format!(
                    "The title stopped with a break (reason {reason:#x})"
                ))
            }
        }
        self.stop();
    }

    /// Drops the session, which stops its System, then the frame it last drew.
    fn end_session(&mut self) {
        self.session = None;
        if self.running.take().is_some() {
            self.refresh_running();
        }
        if let Some(display) = &self.display {
            display.clear_frame();
        }
    }

    fn stop(&mut self) {
        self.end_session();
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
            Action::Boot(path) => self.boot(path, false),
            Action::BootWithoutCustom(path) => self.boot(path, true),
            Action::GameProperties {
                path,
                program_id,
                title,
            } => self.open_game_config(&path, program_id, title),
            Action::ConfigureCurrentGame => {
                if let Some(s) = &self.session {
                    let (path, id, title) = (s.path.clone(), s.program_id, s.title.clone());
                    self.open_game_config(&path, id, title);
                }
            }
            Action::RemoveGameConfig { path, program_id } => {
                let file = GameConfig::path(&self.config_dir(), program_id, &path);
                match std::fs::remove_file(&file) {
                    Ok(()) => self.toast("Removed the custom configuration"),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                        self.toast("This game has no custom configuration")
                    }
                    Err(e) => {
                        self.ui.error = Some(format!("Cannot remove {}: {e}", file.display()))
                    }
                }
                if self
                    .running
                    .as_ref()
                    .is_some_and(|r| r.config.file() == file)
                {
                    self.running = None;
                    self.refresh_running();
                }
            }
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
            Action::KeysAndFirmware => self.open_setup(),
            Action::InstallFiles => self.install_files(),
            Action::SetResolution(scale) => {
                self.set_graphics(|g| g.resolution_scale = scale);
                self.toast(format!("Resolution: {}", scale.label()));
            }
            Action::SetScalingFilter(filter) => {
                self.set_graphics(|g| g.scaling_filter = filter);
                self.toast(format!("Scaling filter: {}", filter.label()));
            }
            Action::SetAntiAliasing(aa) => {
                self.set_graphics(|g| g.anti_aliasing = aa);
                self.toast(format!("Anti-aliasing: {}", aa.label()));
            }
        }
    }

    fn open_game_config(&mut self, path: &Path, program_id: u64, title: String) {
        let config = GameConfig::load(GameConfig::path(&self.config_dir(), program_id, path));
        self.game_config = Some(GameConfigWindow::new(
            title,
            program_id,
            config,
            &self.settings,
        ));
    }

    /// Draws the per-game Properties window when it is open; a saved configuration reaches the
    /// game at once when it is the one running.
    pub(crate) fn show_game_config(&mut self, ctx: &egui::Context) {
        let Some(window) = &mut self.game_config else {
            return;
        };
        let out = window.show(ctx, &self.settings, self.session.is_some());
        let saved = out.saved.then(|| window.config.clone());
        if out.close {
            self.game_config = None;
        }
        let Some(config) = saved else {
            return;
        };
        let running_this = !self.without_custom
            && self.session.as_ref().is_some_and(|s| {
                GameConfig::path(&self.config_dir(), s.program_id, &s.path) == config.file()
            });
        if running_this {
            self.running = (!config.is_empty()).then(|| RunningConfig {
                settings: config.apply(&self.settings),
                config,
            });
            self.refresh_running();
        }
    }

    fn open_setup(&mut self) {
        self.setup = Some(SetupWindow::new(&self.settings.paths, &self.paths));
    }

    /// Draws the keys and firmware window when it is open; a changed folder is saved, and
    /// follows into an open configuration window's draft. New keys rescan the game list, since
    /// encrypted titles only show their names once they can be opened.
    pub(crate) fn show_setup(&mut self, ctx: &egui::Context) {
        let Some(window) = &mut self.setup else {
            return;
        };
        let out = window.show(ctx, &mut self.settings.paths, &self.paths);
        if out.keys_changed || out.nand_changed {
            if let Some(config) = &mut self.config {
                let draft = config.draft_paths();
                if out.keys_changed {
                    draft.keys_dir.clone_from(&self.settings.paths.keys_dir);
                }
                if out.nand_changed {
                    draft.nand_dir.clone_from(&self.settings.paths.nand_dir);
                }
            }
            self.save_settings();
        }
        if out.keys_changed {
            self.refresh_games();
        }
        if out.closed {
            self.setup = None;
            self.settings.ui.setup_done = true;
            self.save_settings();
        }
    }

    /// Changes graphics settings from the main window; see [`Self::edit_settings`].
    fn set_graphics(&mut self, edit: impl Fn(&mut GraphicsSettings)) {
        self.edit_settings(|s| edit(&mut s.graphics));
    }

    /// Changes settings from the main window (status bar, hotkeys). While a game with its own
    /// configuration runs, settings it overrides change in its file and the rest globally. The
    /// presenter and a running title's renderer pick graphics changes up at once, an open
    /// configuration window's draft follows the global changes, and everything is saved.
    fn edit_settings(&mut self, edit: impl Fn(&mut Settings)) {
        let before = self.settings.clone();
        let window_before = self.game_window_settings();
        match &mut self.running {
            Some(running) => {
                running.config.edit(&mut self.settings, &edit);
                if let Err(e) = running.config.save() {
                    tracing::warn!("{}: {e}", running.config.file().display());
                }
            }
            None => edit(&mut self.settings),
        }
        if let Some(config) = &mut self.config {
            per_game::copy_changes(&before, &self.settings, config.draft());
        }
        self.refresh_running();
        self.follow_into_game_window(window_before);
        self.save_settings();
    }

    /// The settings the game in the Properties window runs with now.
    fn game_window_settings(&self) -> Option<Settings> {
        let window = self.game_config.as_ref()?;
        Some(match &self.running {
            Some(running) if running.config.file() == window.config.file() => {
                running.settings.clone()
            }
            _ => window.config.apply(&self.settings),
        })
    }

    /// Carries a change made outside the Properties window into its draft, so its next Apply
    /// does not undo it.
    fn follow_into_game_window(&mut self, before: Option<Settings>) {
        let (Some(before), Some(after)) = (before, self.game_window_settings()) else {
            return;
        };
        if let Some(window) = &mut self.game_config {
            window.follow(&before, &after);
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
                    self.boot(path, self.without_custom);
                }
            }
            HotkeyAction::LoadFile => {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Load file")
                    .add_filter("Switch titles", &["nsp", "xci", "nca", "nup", "nro", "nso"])
                    .add_filter("All files", &["*"])
                    .pick_file()
                {
                    self.boot(path, false);
                }
            }
            HotkeyAction::Configure => {
                if self.config.is_none() {
                    self.config = Some(ConfigWindow::new(&self.settings, &self.paths));
                    self.input.send(InputMsg::Mirror(true));
                }
            }
            HotkeyAction::Fullscreen => self.set_fullscreen(!self.is_fullscreen()),
            HotkeyAction::ExitFullscreen => {
                if self.is_fullscreen() {
                    self.set_fullscreen(false);
                }
            }
            HotkeyAction::ToggleDocked => {
                let docked = !self.current().system.docked;
                self.edit_settings(|s| s.system.docked = docked);
                self.toast(if docked {
                    "Docked mode"
                } else {
                    "Handheld mode"
                });
            }
            HotkeyAction::ToggleSpeedLimit => {
                let limit = match self.current().graphics.speed_limit {
                    Some(limit) => {
                        self.saved_speed_limit = limit;
                        None
                    }
                    None => Some(self.saved_speed_limit),
                };
                self.set_graphics(|g| g.speed_limit = limit);
                self.toast(match limit {
                    Some(limit) => format!("Speed limit {limit}%"),
                    None => "Speed limit off".into(),
                });
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
                let filter = self.current().graphics.scaling_filter.next();
                self.apply(Action::SetScalingFilter(filter));
            }
            HotkeyAction::CycleGpuAccuracy => {
                let accuracy = self.current().graphics.gpu_accuracy.next();
                self.set_graphics(|g| g.gpu_accuracy = accuracy);
                self.toast(format!("GPU accuracy: {}", accuracy.label()));
            }
            HotkeyAction::Exit => self.exit_requested = true,
        }
    }

    /// Tells the input thread whether the guest should see input: only while a title runs and
    /// the configuration window is closed (its keys bind controls, not play).
    fn sync_guest_input(&mut self) {
        let on = self.config.is_none() && self.session.as_ref().is_some_and(|s| !s.is_paused());
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
            && let Some(config) = &mut self.config
            && config.wants_chord()
        {
            if let Some(chord) = KeyChord::from_press(key, &self.held_keys) {
                config.on_chord(chord);
            }
            return;
        }
        if down
            && !event.repeat
            && let Some(config) = &mut self.config
            && config.on_key(key)
        {
            return;
        }
        if down
            && !event.repeat
            && self.config.is_none()
            && !self.egui_ctx.egui_wants_keyboard_input()
            && let Some(chord) = KeyChord::from_press(key, &self.held_keys)
            && let Some(action) = self.hotkeys.action(chord)
        {
            self.apply(Action::Hotkey(action));
            return;
        }
        if !down || self.config.is_some() || !self.egui_ctx.egui_wants_keyboard_input() {
            self.input.send(InputMsg::Key(key, down));
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let (Some(window), Some(mut egui_state)) = (self.window.clone(), self.egui_state.take())
        else {
            return;
        };
        self.games.poll();
        self.poll_install();
        self.check_exit();
        let raw = egui_state.take_egui_input(&window);
        let ctx = self.egui_ctx.clone();
        let mut actions = Vec::new();
        let output = ctx.run_ui(raw, |ui| self.draw(ui, &mut actions));
        egui_state.handle_platform_output(&window, output.platform_output);
        self.egui_state = Some(egui_state);

        let primitives = ctx.tessellate(output.shapes, output.pixels_per_point);
        if let Some(display) = &mut self.display {
            let size = window.inner_size();
            let viewport = self.viewport.unwrap_or(Viewport {
                x: 0,
                y: 0,
                width: size.width,
                height: size.height,
            });
            display.draw(
                viewport,
                UiFrame {
                    primitives,
                    textures_delta: output.textures_delta,
                    pixels_per_point: output.pixels_per_point,
                },
            );
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
        let display = Display::new(window.clone(), &self.settings.graphics, self.perf.clone())?;
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(8192),
        ));
        self.apply_theme();
        self.display = Some(display);
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
            self.boot(path, false);
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
        if let WindowEvent::MouseInput {
            state: ElementState::Pressed,
            button,
            ..
        } = event
            && let Some(button) = MouseButton::from_winit(button)
            && let Some(config) = &mut self.config
            && config.on_mouse(button)
        {
            window.request_redraw();
            return;
        }
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
            WindowEvent::DroppedFile(path) => self.boot(path, false),
            _ => {}
        }
        if self.exit_requested {
            event_loop.exit();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Hotkey(action) if self.session.is_some() && self.config.is_none() => {
                self.apply(Action::Hotkey(action));
            }
            UserEvent::Hotkey(_) => {}
            UserEvent::Wake => {
                self.games.poll();
                self.poll_install();
            }
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
