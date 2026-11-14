//! The settings model. Every crate reads it; the frontend edits it and saves it as TOML.
//!
//! Loading is forgiving: a file from an older version is migrated, a field that no longer parses
//! falls back to its default with a warning instead of discarding the whole file, and values out
//! of range are clamped. Saving writes the current [`SETTINGS_VERSION`].

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::ConsoleGeneration;

/// Version written to new files. Bump it with a step in `migrate` when a field changes shape.
pub const SETTINGS_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    /// Format version of the file this was loaded from; [`SETTINGS_VERSION`] once saved.
    pub version: u32,
    pub system: SystemSettings,
    pub cpu: CpuSettings,
    pub graphics: GraphicsSettings,
    pub frame_view: FrameViewSettings,
    pub controls: ControlSettings,
    pub audio: AudioSettings,
    pub paths: PathSettings,
    pub debug: DebugSettings,
    pub ui: UiSettings,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            version: SETTINGS_VERSION,
            system: SystemSettings::default(),
            cpu: CpuSettings::default(),
            graphics: GraphicsSettings::default(),
            frame_view: FrameViewSettings::default(),
            controls: ControlSettings::default(),
            audio: AudioSettings::default(),
            paths: PathSettings::default(),
            debug: DebugSettings::default(),
            ui: UiSettings::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("reading {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("parsing {path}: {source}")]
    Parse {
        path: PathBuf,
        source: Box<toml::de::Error>,
    },
    #[error("serializing settings: {0}")]
    Serialize(#[from] toml::ser::Error),
}

impl Settings {
    /// Loads settings from `path`; a missing file yields the defaults. Fields that fail to parse
    /// are reset to their defaults and logged; only a file that is not TOML at all is an error.
    pub fn load(path: &Path) -> Result<Self, SettingsError> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(source) => {
                return Err(SettingsError::Io {
                    path: path.to_path_buf(),
                    source,
                });
            }
        };
        let (settings, warnings) =
            Self::from_toml(&text).map_err(|source| SettingsError::Parse {
                path: path.to_path_buf(),
                source: Box::new(source),
            })?;
        for warning in warnings {
            tracing::warn!("{}: {warning}", path.display());
        }
        Ok(settings)
    }

    /// Parses, migrates and normalizes a settings file. The second value lists every field that
    /// was reset to its default because it no longer parses.
    pub fn from_toml(text: &str) -> Result<(Self, Vec<String>), toml::de::Error> {
        let mut table: toml::Table = toml::from_str(text)?;
        let mut warnings = Vec::new();
        let version = table
            .get("version")
            .and_then(toml::Value::as_integer)
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(0);
        if version > SETTINGS_VERSION {
            warnings.push(format!(
                "written by a newer Lichii (settings version {version}); unknown fields are ignored"
            ));
        }
        migrate(&mut table, version);
        let mut settings = Self {
            version: SETTINGS_VERSION,
            system: section(&mut table, "system", &mut warnings),
            cpu: section(&mut table, "cpu", &mut warnings),
            graphics: section(&mut table, "graphics", &mut warnings),
            frame_view: section(&mut table, "frame_view", &mut warnings),
            controls: section(&mut table, "controls", &mut warnings),
            audio: section(&mut table, "audio", &mut warnings),
            paths: section(&mut table, "paths", &mut warnings),
            debug: section(&mut table, "debug", &mut warnings),
            ui: section(&mut table, "ui", &mut warnings),
        };
        settings.normalize();
        Ok((settings, warnings))
    }

    /// Saves atomically: the file is written beside `path` and renamed over it, so a crash never
    /// leaves a half-written config.
    pub fn save(&self, path: &Path) -> Result<(), SettingsError> {
        let mut text = toml::to_string_pretty(&Self {
            version: SETTINGS_VERSION,
            ..self.clone()
        })?;
        text.insert_str(
            0,
            "# Lichii settings. Unknown or invalid fields fall back to defaults.\n",
        );
        write_atomic(path, text.as_bytes()).map_err(|source| SettingsError::Io {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Clamps every value to the range the UI offers and fills in missing players and hotkeys.
    pub fn normalize(&mut self) {
        let g = &mut self.graphics;
        g.fsr_sharpness = g.fsr_sharpness.min(100);
        g.speed_limit = g
            .speed_limit
            .filter(|&v| v > 0)
            .map(|v| v.min(MAX_SPEED_LIMIT));

        let fv = &mut self.frame_view;
        fv.graph_seconds = clamp_or(fv.graph_seconds, 1.0, 30.0, 5.0);
        fv.opacity = clamp_or(fv.opacity, 0.1, 1.0, 0.85);
        fv.scale = clamp_or(fv.scale, 0.5, 3.0, 1.0);

        let controls = &mut self.controls;
        controls.players.truncate(MAX_PLAYERS);
        while controls.players.len() < MAX_PLAYERS {
            controls.players.push(PlayerInput::default());
        }
        controls.mouse_panning_sensitivity =
            clamp_or(controls.mouse_panning_sensitivity, 1.0, 100.0, 50.0);
        for player in &mut controls.players {
            player.vibration_strength = player.vibration_strength.min(100);
            for stick in [&mut player.left_stick, &mut player.right_stick] {
                stick.deadzone = clamp_or(stick.deadzone, 0.0, 0.9, 0.15);
                stick.range = clamp_or(stick.range, 0.5, 1.5, 1.0);
                stick.modifier_scale = clamp_or(stick.modifier_scale, 0.0, 1.0, 0.5);
            }
        }

        self.audio.volume = self.audio.volume.min(200);

        let zone = self.system.time_zone.trim();
        self.system.time_zone = if zone.is_empty() || zone.eq_ignore_ascii_case(TIME_ZONE_AUTO) {
            TIME_ZONE_AUTO.into()
        } else {
            zone.into()
        };

        for (action, hotkey) in HotkeyAction::defaults() {
            self.ui.hotkeys.entry(action).or_insert(hotkey);
        }
    }
}

/// Highest speed limit the UI offers, in percent.
pub const MAX_SPEED_LIMIT: u16 = 1000;

fn clamp_or(value: f32, min: f32, max: f32, fallback: f32) -> f32 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

/// Rewrites `table` from format `from` to [`SETTINGS_VERSION`], one version at a time.
fn migrate(table: &mut toml::Table, from: u32) {
    for version in from..SETTINGS_VERSION {
        match version {
            // Version 1 writes an unlimited speed as `speed_limit = 0`; version 0 dropped the key,
            // so those files load the default limit, which is all version 0 could express.
            0 => {}
            _ => unreachable!("no migration from settings version {version}"),
        }
    }
    table.insert(
        "version".into(),
        toml::Value::Integer(SETTINGS_VERSION.into()),
    );
}

/// Deserializes one top-level section. When it fails as a whole, every field is tried on its own
/// (each section is `#[serde(default)]`), the fields that fail are dropped with a warning, and the
/// rest are kept.
fn section<T: DeserializeOwned + Default>(
    table: &mut toml::Table,
    key: &str,
    warnings: &mut Vec<String>,
) -> T {
    let Some(value) = table.remove(key) else {
        return T::default();
    };
    let toml::Value::Table(mut fields) = value else {
        warnings.push(format!("[{key}] is not a table; using defaults"));
        return T::default();
    };
    if let Ok(parsed) = T::deserialize(toml::Value::Table(fields.clone())) {
        return parsed;
    }
    fields.retain(|name, field| {
        let single = toml::Table::from_iter([(name.to_string(), field.clone())]);
        match T::deserialize(toml::Value::Table(single)) {
            Ok(_) => true,
            Err(e) => {
                warnings.push(format!("{key}.{name}: {}; using the default", e.message()));
                false
            }
        }
    });
    T::deserialize(toml::Value::Table(fields)).unwrap_or_else(|e| {
        warnings.push(format!("[{key}]: {}; using defaults", e.message()));
        T::default()
    })
}

/// Writes `bytes` to a sibling temporary file and renames it over `path`.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SystemSettings {
    /// Console to present to titles that run on both; Switch 2 titles always get Switch 2.
    pub preferred_console: ConsoleGeneration,
    pub docked: bool,
    pub language: Language,
    pub region: Region,
    pub memory_layout: MemoryLayout,
    pub username: String,
    pub rng_seed: Option<u32>,
    /// The console's time zone: [`TIME_ZONE_AUTO`] follows the host, anything else is an IANA
    /// name such as `Europe/Amsterdam`.
    pub time_zone: String,
    /// Whether titles may go online; when false the network services report no connection.
    pub network_enabled: bool,
    /// Refuse services a title's NPDM does not list. When false, sm only logs them.
    pub enforce_service_access: bool,
    /// How long closing waits, in milliseconds, for a title that holds the exit lock
    /// (`LockExit`) to save and exit after it gets the exit request; 0 stops it at once.
    pub exit_timeout_ms: u32,
}

/// The [`SystemSettings::time_zone`] value that follows the host's time zone.
pub const TIME_ZONE_AUTO: &str = "auto";

impl Default for SystemSettings {
    fn default() -> Self {
        Self {
            preferred_console: ConsoleGeneration::Switch2,
            docked: true,
            language: Language::AmericanEnglish,
            region: Region::Usa,
            memory_layout: MemoryLayout::Auto,
            username: "lichii".into(),
            rng_seed: None,
            time_zone: TIME_ZONE_AUTO.into(),
            network_enabled: false,
            enforce_service_access: false,
            exit_timeout_ms: 2000,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Language {
    Japanese,
    #[default]
    AmericanEnglish,
    French,
    German,
    Italian,
    Spanish,
    Chinese,
    Korean,
    Dutch,
    Portuguese,
    Russian,
    Taiwanese,
    BritishEnglish,
    CanadianFrench,
    LatinAmericanSpanish,
    SimplifiedChinese,
    TraditionalChinese,
    BrazilianPortuguese,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Region {
    Japan,
    #[default]
    Usa,
    Europe,
    Australia,
    China,
    Korea,
    Taiwan,
}

/// Guest DRAM size. `Auto` picks 4 GiB for Switch 1 and 12 GiB for Switch 2.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum MemoryLayout {
    #[default]
    Auto,
    Dram4GiB,
    Dram6GiB,
    Dram8GiB,
    Dram12GiB,
}

impl MemoryLayout {
    pub fn dram_size(self, console: ConsoleGeneration) -> u64 {
        const GIB: u64 = 1 << 30;
        match (self, console) {
            (Self::Auto, ConsoleGeneration::Switch1) | (Self::Dram4GiB, _) => 4 * GIB,
            (Self::Auto, ConsoleGeneration::Switch2) | (Self::Dram12GiB, _) => 12 * GIB,
            (Self::Dram6GiB, _) => 6 * GIB,
            (Self::Dram8GiB, _) => 8 * GIB,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct CpuSettings {
    pub backend: CpuBackend,
    pub accuracy: CpuAccuracy,
    /// Map guest memory directly into the host address space for JIT loads and stores.
    pub fastmem: bool,
}

impl Default for CpuSettings {
    fn default() -> Self {
        Self {
            backend: CpuBackend::Jit,
            accuracy: CpuAccuracy::Auto,
            fastmem: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum CpuBackend {
    Interpreter,
    #[default]
    Jit,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum CpuAccuracy {
    #[default]
    Auto,
    Accurate,
    Unsafe,
    Paranoid,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct GraphicsSettings {
    /// Vulkan physical device name; `None` picks the first discrete GPU.
    pub device: Option<String>,
    pub resolution_scale: ResolutionScale,
    pub scaling_filter: ScalingFilter,
    /// FSR sharpening, 0 (none) to 100 (most).
    pub fsr_sharpness: u8,
    pub anti_aliasing: AntiAliasing,
    pub aspect_ratio: AspectRatio,
    pub present_mode: PresentMode,
    pub fullscreen: FullscreenMode,
    /// Speed limit in percent of real time; `None` runs unlimited (written as `0`).
    #[serde(with = "speed_limit_serde")]
    pub speed_limit: Option<u16>,
    pub gpu_accuracy: GpuAccuracy,
    pub anisotropic_filtering: Anisotropy,
    pub astc_decode: AstcDecode,
    pub astc_recompression: AstcRecompression,
    pub async_shader_compilation: bool,
    pub disk_shader_cache: bool,
    pub vram_usage: VramUsage,
    pub validation_layers: bool,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self {
            device: None,
            resolution_scale: ResolutionScale::X1,
            scaling_filter: ScalingFilter::Fsr,
            fsr_sharpness: 25,
            anti_aliasing: AntiAliasing::None,
            aspect_ratio: AspectRatio::R16x9,
            present_mode: PresentMode::Mailbox,
            fullscreen: FullscreenMode::Borderless,
            speed_limit: Some(100),
            gpu_accuracy: GpuAccuracy::Normal,
            anisotropic_filtering: Anisotropy::Auto,
            astc_decode: AstcDecode::Gpu,
            astc_recompression: AstcRecompression::Uncompressed,
            async_shader_compilation: true,
            disk_shader_cache: true,
            vram_usage: VramUsage::Conservative,
            validation_layers: false,
        }
    }
}

/// Internal render resolution relative to the console's native resolution.
#[derive(
    Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub enum ResolutionScale {
    X1_4,
    X1_2,
    X3_4,
    #[default]
    X1,
    X5_4,
    X3_2,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    X8,
}

impl ResolutionScale {
    pub const ALL: [Self; 13] = [
        Self::X1_4,
        Self::X1_2,
        Self::X3_4,
        Self::X1,
        Self::X5_4,
        Self::X3_2,
        Self::X2,
        Self::X3,
        Self::X4,
        Self::X5,
        Self::X6,
        Self::X7,
        Self::X8,
    ];

    /// Scale as numerator / denominator, so integer sizes scale exactly.
    pub const fn ratio(self) -> (u32, u32) {
        match self {
            Self::X1_4 => (1, 4),
            Self::X1_2 => (1, 2),
            Self::X3_4 => (3, 4),
            Self::X1 => (1, 1),
            Self::X5_4 => (5, 4),
            Self::X3_2 => (3, 2),
            Self::X2 => (2, 1),
            Self::X3 => (3, 1),
            Self::X4 => (4, 1),
            Self::X5 => (5, 1),
            Self::X6 => (6, 1),
            Self::X7 => (7, 1),
            Self::X8 => (8, 1),
        }
    }

    pub const fn scale(self, size: u32) -> u32 {
        let (num, den) = self.ratio();
        let scaled = (size as u64 * num as u64) / den as u64;
        if scaled == 0 { 1 } else { scaled as u32 }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::X1_4 => "0.25x (180p/270p)",
            Self::X1_2 => "0.5x (360p/540p)",
            Self::X3_4 => "0.75x (540p/810p)",
            Self::X1 => "1x (720p/1080p)",
            Self::X5_4 => "1.25x (900p/1350p)",
            Self::X3_2 => "1.5x (1080p/1620p)",
            Self::X2 => "2x (1440p/2160p)",
            Self::X3 => "3x (2160p/3240p)",
            Self::X4 => "4x (2880p/4320p)",
            Self::X5 => "5x (3600p/5400p)",
            Self::X6 => "6x (4320p/6480p)",
            Self::X7 => "7x (5040p/7560p)",
            Self::X8 => "8x (5760p/8640p)",
        }
    }
}

/// Filter used to scale the final frame to the window.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScalingFilter {
    NearestNeighbor,
    Bilinear,
    Bicubic,
    Gaussian,
    Lanczos,
    ScaleForce,
    #[default]
    Fsr,
    Area,
    Mmpx,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum AntiAliasing {
    #[default]
    None,
    Fxaa,
    Smaa,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum AspectRatio {
    #[default]
    R16x9,
    R4x3,
    R21x9,
    R16x10,
    Stretch,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum PresentMode {
    /// VSync.
    Fifo,
    FifoRelaxed,
    #[default]
    Mailbox,
    Immediate,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum FullscreenMode {
    #[default]
    Borderless,
    Exclusive,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum GpuAccuracy {
    Low,
    #[default]
    Normal,
    High,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Anisotropy {
    #[default]
    Auto,
    X1,
    X2,
    X4,
    X8,
    X16,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum AstcDecode {
    Cpu,
    #[default]
    Gpu,
    CpuAsync,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum AstcRecompression {
    #[default]
    Uncompressed,
    Bc1,
    Bc3,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum VramUsage {
    #[default]
    Conservative,
    Aggressive,
}

/// The frame view: an overlay with frame pacing and performance numbers.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct FrameViewSettings {
    pub enabled: bool,
    pub show_fps: bool,
    pub show_frame_time: bool,
    pub show_frame_time_graph: bool,
    pub show_emulation_speed: bool,
    pub show_cpu_gpu_time: bool,
    pub show_vram_usage: bool,
    pub show_shader_compilation: bool,
    pub show_resolution: bool,
    pub position: OverlayPosition,
    /// Seconds of history in the frame-time graph.
    pub graph_seconds: f32,
    pub opacity: f32,
    pub scale: f32,
}

impl Default for FrameViewSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            show_fps: true,
            show_frame_time: true,
            show_frame_time_graph: true,
            show_emulation_speed: true,
            show_cpu_gpu_time: false,
            show_vram_usage: false,
            show_shader_compilation: true,
            show_resolution: true,
            position: OverlayPosition::TopLeft,
            graph_seconds: 5.0,
            opacity: 0.85,
            scale: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum OverlayPosition {
    #[default]
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

pub const MAX_PLAYERS: usize = 8;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ControlSettings {
    /// Players 1-8. Handheld mode uses player 1's mapping.
    pub players: Vec<PlayerInput>,
    pub keyboard_enabled: bool,
    pub mouse_panning: bool,
    pub mouse_panning_sensitivity: f32,
    pub touchscreen_enabled: bool,
}

impl Default for ControlSettings {
    fn default() -> Self {
        let mut players: Vec<PlayerInput> =
            (0..MAX_PLAYERS).map(|_| PlayerInput::default()).collect();
        players[0] = PlayerInput::default_keyboard();
        Self {
            players,
            keyboard_enabled: true,
            mouse_panning: false,
            mouse_panning_sensitivity: 50.0,
            touchscreen_enabled: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ControllerType {
    #[default]
    ProController,
    DualJoycons,
    LeftJoycon,
    RightJoycon,
    Handheld,
    GameCube,
    /// Switch 2 Pro Controller (adds C, GL and GR).
    Switch2ProController,
    /// Switch 2 Joy-Con 2 pair (adds C and mouse sensors).
    Switch2DualJoycons,
    Switch2Handheld,
    Switch2LeftJoycon,
    /// Right Joy-Con 2 (has the C button).
    Switch2RightJoycon,
}

impl ControllerType {
    /// Every controller type. The order is part of the contract: `lichii-input` packs a type
    /// as its index here, so new types are appended.
    pub const ALL: [Self; 11] = [
        Self::ProController,
        Self::DualJoycons,
        Self::LeftJoycon,
        Self::RightJoycon,
        Self::Handheld,
        Self::GameCube,
        Self::Switch2ProController,
        Self::Switch2DualJoycons,
        Self::Switch2Handheld,
        Self::Switch2LeftJoycon,
        Self::Switch2RightJoycon,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ProController => "Pro Controller",
            Self::DualJoycons => "Dual Joy-Cons",
            Self::LeftJoycon => "Left Joy-Con",
            Self::RightJoycon => "Right Joy-Con",
            Self::Handheld => "Handheld",
            Self::GameCube => "GameCube Controller",
            Self::Switch2ProController => "Switch 2 Pro Controller",
            Self::Switch2DualJoycons => "Dual Joy-Con 2",
            Self::Switch2Handheld => "Switch 2 Handheld",
            Self::Switch2LeftJoycon => "Left Joy-Con 2",
            Self::Switch2RightJoycon => "Right Joy-Con 2",
        }
    }

    /// Switch 2 controllers (C button, and GL/GR on the Pro Controller 2).
    pub fn is_switch2(self) -> bool {
        matches!(
            self,
            Self::Switch2ProController
                | Self::Switch2DualJoycons
                | Self::Switch2Handheld
                | Self::Switch2LeftJoycon
                | Self::Switch2RightJoycon
        )
    }

    /// Joy-Cons attached to the console; hid reports these on the handheld npad id.
    pub fn is_handheld(self) -> bool {
        matches!(self, Self::Handheld | Self::Switch2Handheld)
    }

    /// Whether ZL/ZR are analog triggers rather than buttons (GameCube).
    pub fn has_analog_triggers(self) -> bool {
        self == Self::GameCube
    }

    /// Whether the controller has a left stick (every type but a single right Joy-Con).
    pub fn has_left_stick(self) -> bool {
        !matches!(self, Self::RightJoycon | Self::Switch2RightJoycon)
    }

    /// Whether the controller has a right stick (every type but a single left Joy-Con).
    pub fn has_right_stick(self) -> bool {
        !matches!(self, Self::LeftJoycon | Self::Switch2LeftJoycon)
    }

    /// Whether the controller has `button` as a digital button. The input mapper masks out the
    /// rest. A GameCube controller's ZL/ZR bindings drive its analog L/R triggers instead
    /// (see [`Self::has_analog_triggers`]) and its Z button is R, as in Lichii.
    pub fn has_button(self, button: NpadButton) -> bool {
        use NpadButton::*;
        let left_half = matches!(
            button,
            DPadLeft | DPadUp | DPadRight | DPadDown | L | ZL | Minus | LStick | Capture
        );
        let right_half = matches!(button, A | B | X | Y | R | ZR | Plus | RStick | Home);
        let rails = matches!(button, LeftSL | LeftSR | RightSL | RightSR);
        let left_rail = matches!(button, LeftSL | LeftSR);
        let right_rail = matches!(button, RightSL | RightSR);
        match self {
            Self::ProController | Self::Handheld => left_half || right_half,
            Self::DualJoycons => left_half || right_half || rails,
            Self::LeftJoycon => left_half || left_rail,
            Self::RightJoycon => right_half || right_rail,
            Self::GameCube => matches!(
                button,
                A | B | X | Y | R | Plus | DPadLeft | DPadUp | DPadRight | DPadDown
            ),
            Self::Switch2ProController => left_half || right_half || matches!(button, C | GL | GR),
            Self::Switch2Handheld => left_half || right_half || button == C,
            Self::Switch2DualJoycons => left_half || right_half || rails || button == C,
            Self::Switch2LeftJoycon => left_half || left_rail,
            Self::Switch2RightJoycon => right_half || right_rail || button == C,
        }
    }
}

/// Npad buttons, including the Switch 2 additions (C, GL, GR).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NpadButton {
    A,
    B,
    X,
    Y,
    LStick,
    RStick,
    L,
    R,
    ZL,
    ZR,
    Plus,
    Minus,
    DPadLeft,
    DPadUp,
    DPadRight,
    DPadDown,
    LeftSL,
    LeftSR,
    RightSL,
    RightSR,
    Home,
    Capture,
    C,
    GL,
    GR,
}

impl NpadButton {
    /// Every button, in `nn::hid::NpadButton` bit order followed by the Switch 2 additions.
    pub const ALL: [Self; 25] = {
        use NpadButton::*;
        [
            A, B, X, Y, LStick, RStick, L, R, ZL, ZR, Plus, Minus, DPadLeft, DPadUp, DPadRight,
            DPadDown, LeftSL, LeftSR, RightSL, RightSR, Home, Capture, C, GL, GR,
        ]
    };

    /// The button's name as printed on a Switch controller.
    pub fn label(self) -> &'static str {
        use NpadButton::*;
        match self {
            A => "A",
            B => "B",
            X => "X",
            Y => "Y",
            LStick => "Left Stick",
            RStick => "Right Stick",
            L => "L",
            R => "R",
            ZL => "ZL",
            ZR => "ZR",
            Plus => "Plus",
            Minus => "Minus",
            DPadLeft => "Left",
            DPadUp => "Up",
            DPadRight => "Right",
            DPadDown => "Down",
            LeftSL => "SL (Left)",
            LeftSR => "SR (Left)",
            RightSL => "SL (Right)",
            RightSR => "SR (Right)",
            Home => "Home",
            Capture => "Capture",
            C => "C",
            GL => "GL",
            GR => "GR",
        }
    }
}

/// One host input bound to one guest input. Keyboard and mouse bindings read the host keyboard
/// and mouse; gamepad bindings read the player's `InputDevice`.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum InputBinding {
    #[default]
    None,
    /// A keyboard key, by its `winit` `KeyCode` name (e.g. `KeyZ`, `ArrowUp`).
    Key(String),
    /// A gamepad button by SDL name (e.g. `a`, `leftshoulder`, `dpup`).
    GamepadButton { button: String },
    /// A mouse button: `Left`, `Right`, `Middle`, `Back` or `Forward`.
    MouseButton(String),
    /// A gamepad axis past `threshold` (0..1) in `direction` (+1 or -1).
    GamepadAxis {
        axis: String,
        threshold: f32,
        direction: i8,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct StickBinding {
    /// Analog source: an SDL axis pair (`leftx`/`lefty`) on a gamepad.
    pub analog_x: Option<String>,
    pub analog_y: Option<String>,
    pub invert_x: bool,
    pub invert_y: bool,
    /// Digital sources for keyboards.
    pub up: InputBinding,
    pub down: InputBinding,
    pub left: InputBinding,
    pub right: InputBinding,
    /// Held to scale digital input by `modifier_scale`.
    pub modifier: InputBinding,
    pub modifier_scale: f32,
    /// Radial deadzone, 0..=0.9 of the stick's travel.
    pub deadzone: f32,
    /// 0.5..=1.5: the deflection that maps to full output. Below 1 lets a worn stick reach the
    /// edge; above 1 needs more than full travel and so caps the output.
    pub range: f32,
}

impl Default for StickBinding {
    fn default() -> Self {
        Self {
            analog_x: None,
            analog_y: None,
            invert_x: false,
            invert_y: false,
            up: InputBinding::None,
            down: InputBinding::None,
            left: InputBinding::None,
            right: InputBinding::None,
            modifier: InputBinding::None,
            modifier_scale: 0.5,
            deadzone: 0.15,
            range: 1.0,
        }
    }
}

/// How a bound button behaves, as set from Lichii's binding context menu.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct ButtonOptions {
    /// Reports the opposite of the host input.
    pub inverted: bool,
    /// Each press flips the button between held and released.
    pub toggle: bool,
    /// While held, the button alternates between pressed and released.
    pub turbo: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum InputDevice {
    #[default]
    None,
    Keyboard,
    /// A gamepad by SDL GUID; `index` separates identical controllers.
    Gamepad {
        guid: String,
        index: u32,
        name: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PlayerInput {
    pub connected: bool,
    pub controller_type: ControllerType,
    pub device: InputDevice,
    pub buttons: BTreeMap<NpadButton, InputBinding>,
    /// Inverted, toggle and turbo buttons; absent buttons behave normally.
    pub button_options: BTreeMap<NpadButton, ButtonOptions>,
    pub left_stick: StickBinding,
    pub right_stick: StickBinding,
    pub motion_enabled: bool,
    /// Motion without a gyro, as in Lichii: while bound the controller reports lying at rest,
    /// and holding the input shakes it. The device's own sensors take precedence.
    pub motion_button: InputBinding,
    pub vibration_enabled: bool,
    /// 0..=100 percent.
    pub vibration_strength: u8,
    /// Name of the saved profile this mapping came from, if any.
    pub profile: Option<String>,
}

impl Default for PlayerInput {
    fn default() -> Self {
        Self {
            connected: false,
            controller_type: ControllerType::ProController,
            device: InputDevice::None,
            buttons: BTreeMap::new(),
            button_options: BTreeMap::new(),
            left_stick: StickBinding::default(),
            right_stick: StickBinding::default(),
            motion_enabled: true,
            motion_button: InputBinding::None,
            vibration_enabled: true,
            vibration_strength: 100,
            profile: None,
        }
    }
}

impl PlayerInput {
    /// Player 1 on the keyboard, with yuzu's default layout.
    pub fn default_keyboard() -> Self {
        use NpadButton::*;
        let key = |k: &str| InputBinding::Key(k.to_string());
        let buttons = [
            (A, "KeyC"),
            (B, "KeyX"),
            (X, "KeyV"),
            (Y, "KeyZ"),
            (LStick, "KeyF"),
            (RStick, "KeyG"),
            (L, "KeyQ"),
            (R, "KeyE"),
            (ZL, "KeyR"),
            (ZR, "KeyT"),
            (Plus, "KeyM"),
            (Minus, "KeyN"),
            (DPadLeft, "ArrowLeft"),
            (DPadUp, "ArrowUp"),
            (DPadRight, "ArrowRight"),
            (DPadDown, "ArrowDown"),
            (Home, "Backspace"),
            (Capture, "Backslash"),
            (C, "KeyB"),
        ]
        .into_iter()
        .map(|(button, k)| (button, key(k)))
        .collect();
        let stick = |up: &str, down: &str, left: &str, right: &str, modifier: &str| StickBinding {
            up: key(up),
            down: key(down),
            left: key(left),
            right: key(right),
            modifier: key(modifier),
            ..StickBinding::default()
        };
        Self {
            connected: true,
            controller_type: ControllerType::ProController,
            device: InputDevice::Keyboard,
            buttons,
            left_stick: stick("KeyW", "KeyS", "KeyA", "KeyD", "ShiftLeft"),
            right_stick: stick("KeyI", "KeyK", "KeyJ", "KeyL", "ShiftRight"),
            ..Self::default()
        }
    }

    /// A gamepad mapped by position, the way Lichii maps SDL controllers: Switch A is the east
    /// face button, B south, X north, Y west.
    pub fn default_gamepad(device: InputDevice) -> Self {
        use NpadButton::*;
        let pad = |b: &str| InputBinding::GamepadButton {
            button: b.to_string(),
        };
        let trigger = |axis: &str| InputBinding::GamepadAxis {
            axis: axis.to_string(),
            threshold: 0.5,
            direction: 1,
        };
        let mut buttons: BTreeMap<NpadButton, InputBinding> = [
            (A, "b"),
            (B, "a"),
            (X, "y"),
            (Y, "x"),
            (LStick, "leftstick"),
            (RStick, "rightstick"),
            (L, "leftshoulder"),
            (R, "rightshoulder"),
            (Plus, "start"),
            (Minus, "back"),
            (DPadLeft, "dpleft"),
            (DPadUp, "dpup"),
            (DPadRight, "dpright"),
            (DPadDown, "dpdown"),
            (Home, "guide"),
            (Capture, "misc1"),
            (C, "touchpad"),
            (GL, "paddle2"),
            (GR, "paddle1"),
            (LeftSL, "paddle4"),
            (RightSR, "paddle3"),
        ]
        .into_iter()
        .map(|(button, name)| (button, pad(name)))
        .collect();
        buttons.insert(ZL, trigger("lefttrigger"));
        buttons.insert(ZR, trigger("righttrigger"));
        let stick = |x: &str, y: &str| StickBinding {
            analog_x: Some(x.to_string()),
            analog_y: Some(y.to_string()),
            ..StickBinding::default()
        };
        Self {
            connected: true,
            device,
            buttons,
            left_stick: stick("leftx", "lefty"),
            right_stick: stick("rightx", "righty"),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct AudioSettings {
    pub output: AudioOutput,
    pub output_device: Option<String>,
    /// 0..=200 percent.
    pub volume: u8,
    pub muted: bool,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            output: AudioOutput::Auto,
            output_device: None,
            volume: 100,
            muted: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum AudioOutput {
    #[default]
    Auto,
    Cpal,
    Null,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct PathSettings {
    pub game_dirs: Vec<PathBuf>,
    /// Overrides for the profile directories; `None` uses `UserPaths`.
    pub keys_dir: Option<PathBuf>,
    pub nand_dir: Option<PathBuf>,
    pub sdmc_dir: Option<PathBuf>,
    pub mods_dir: Option<PathBuf>,
}

/// Window, game list and hotkey preferences of the desktop frontend.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct UiSettings {
    pub theme: Theme,
    /// Ask before stopping a running title.
    pub confirm_stop: bool,
    pub pause_when_unfocused: bool,
    pub show_status_bar: bool,
    pub hide_cursor_when_idle: bool,
    /// Most recently booted files, newest first.
    pub recent_files: Vec<PathBuf>,
    pub window_width: u32,
    pub window_height: u32,
    pub window_maximized: bool,
    pub hotkeys: BTreeMap<HotkeyAction, Hotkey>,
    /// The keys and firmware setup was closed once; it no longer opens by itself at startup.
    pub setup_done: bool,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            theme: Theme::Dark,
            confirm_stop: true,
            pause_when_unfocused: false,
            show_status_bar: true,
            hide_cursor_when_idle: true,
            recent_files: Vec::new(),
            window_width: 1280,
            window_height: 760,
            window_maximized: false,
            hotkeys: HotkeyAction::defaults().collect(),
            setup_done: false,
        }
    }
}

/// Most recent files kept in [`UiSettings::recent_files`].
pub const MAX_RECENT_FILES: usize = 10;

impl UiSettings {
    /// Moves `path` to the front of the recent files.
    pub fn push_recent(&mut self, path: PathBuf) {
        self.recent_files.retain(|p| *p != path);
        self.recent_files.insert(0, path);
        self.recent_files.truncate(MAX_RECENT_FILES);
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum Theme {
    #[default]
    Dark,
    Light,
}

/// A frontend action triggered by a keyboard chord or a held controller button combination.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HotkeyAction {
    ContinuePause,
    Stop,
    Restart,
    LoadFile,
    Configure,
    Fullscreen,
    ExitFullscreen,
    ToggleDocked,
    ToggleSpeedLimit,
    ToggleFrameView,
    ToggleStatusBar,
    ToggleMousePanning,
    CycleScalingFilter,
    CycleGpuAccuracy,
    Exit,
}

/// One hotkey. `keys` is a chord such as `Ctrl+Shift+F11` (modifiers `Ctrl`, `Shift`, `Alt`,
/// `Super`, then a `winit` `KeyCode` name); empty means unbound. `controller` triggers it while
/// every listed button is held on player 1; empty means unbound.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct Hotkey {
    pub keys: String,
    pub controller: Vec<NpadButton>,
}

impl HotkeyAction {
    pub const ALL: [Self; 15] = [
        Self::ContinuePause,
        Self::Stop,
        Self::Restart,
        Self::LoadFile,
        Self::Configure,
        Self::Fullscreen,
        Self::ExitFullscreen,
        Self::ToggleDocked,
        Self::ToggleSpeedLimit,
        Self::ToggleFrameView,
        Self::ToggleStatusBar,
        Self::ToggleMousePanning,
        Self::CycleScalingFilter,
        Self::CycleGpuAccuracy,
        Self::Exit,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::ContinuePause => "Continue/Pause Emulation",
            Self::Stop => "Stop Emulation",
            Self::Restart => "Restart Emulation",
            Self::LoadFile => "Load File",
            Self::Configure => "Configure",
            Self::Fullscreen => "Fullscreen",
            Self::ExitFullscreen => "Exit Fullscreen",
            Self::ToggleDocked => "Change Docked Mode",
            Self::ToggleSpeedLimit => "Toggle Speed Limit",
            Self::ToggleFrameView => "Toggle Frame View",
            Self::ToggleStatusBar => "Toggle Status Bar",
            Self::ToggleMousePanning => "Toggle Mouse Panning",
            Self::CycleScalingFilter => "Change Scaling Filter",
            Self::CycleGpuAccuracy => "Change GPU Accuracy",
            Self::Exit => "Exit Lichii",
        }
    }

    /// Lichii's default bindings.
    pub fn defaults() -> impl Iterator<Item = (Self, Hotkey)> {
        use NpadButton::*;
        Self::ALL.into_iter().map(|action| {
            let (keys, controller): (&str, &[NpadButton]) = match action {
                Self::ContinuePause => ("F4", &[Home, Plus]),
                Self::Stop => ("F5", &[L, Plus, Minus]),
                Self::Restart => ("F6", &[R, Plus, Minus]),
                Self::LoadFile => ("Ctrl+KeyO", &[]),
                Self::Configure => ("Ctrl+Comma", &[]),
                Self::Fullscreen => ("F11", &[Home, B]),
                Self::ExitFullscreen => ("Escape", &[]),
                Self::ToggleDocked => ("F10", &[Home, X]),
                Self::ToggleSpeedLimit => ("Ctrl+KeyU", &[Home, Y]),
                Self::ToggleFrameView => ("Ctrl+KeyV", &[]),
                Self::ToggleStatusBar => ("Ctrl+KeyS", &[]),
                Self::ToggleMousePanning => ("Ctrl+F9", &[]),
                Self::CycleScalingFilter => ("F8", &[Home, L]),
                Self::CycleGpuAccuracy => ("F9", &[Home, R]),
                Self::Exit => ("Ctrl+KeyQ", &[Home, Minus]),
            };
            (
                action,
                Hotkey {
                    keys: keys.to_string(),
                    controller: controller.to_vec(),
                },
            )
        })
    }
}

mod speed_limit_serde {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(limit: &Option<u16>, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u16(limit.unwrap_or(0))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<u16>, D::Error> {
        Ok(Some(u16::deserialize(d)?).filter(|&v| v > 0))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct DebugSettings {
    pub log_filter: String,
    pub svc_trace: bool,
    pub ipc_trace: bool,
    /// Answer unknown service commands with an empty success instead of stopping.
    pub auto_stub: bool,
    pub gdb_stub_port: Option<u16>,
    pub dump_shaders: bool,
    /// Count every place shader translation approximates or stubs a guest mode, and log each
    /// (opcode, variant) once plus a census of them. Read at launch.
    pub shader_fallbacks: bool,
}

impl Default for DebugSettings {
    fn default() -> Self {
        Self {
            log_filter: "info".into(),
            svc_trace: false,
            ipc_trace: false,
            auto_stub: false,
            gdb_stub_port: None,
            dump_shaders: false,
            shader_fallbacks: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load(text: &str) -> (Settings, Vec<String>) {
        Settings::from_toml(text).unwrap()
    }

    #[test]
    fn time_zone_and_network() {
        let (settings, _) = load("");
        assert_eq!(settings.system.time_zone, TIME_ZONE_AUTO);
        assert!(!settings.system.network_enabled);
        assert!(!settings.system.enforce_service_access);
        let text = "[system]\ntime_zone = \" Europe/Amsterdam \"\nnetwork_enabled = true\n";
        let (settings, _) = load(text);
        assert_eq!(settings.system.time_zone, "Europe/Amsterdam");
        assert!(settings.system.network_enabled);
        let zone = |text: &str| load(text).0.system.time_zone;
        assert_eq!(zone("[system]\ntime_zone = \"\"\n"), TIME_ZONE_AUTO);
        assert_eq!(zone("[system]\ntime_zone = \"AUTO\"\n"), TIME_ZONE_AUTO);
    }

    #[test]
    fn round_trips_through_toml() {
        let mut settings = Settings::default();
        settings.graphics.resolution_scale = ResolutionScale::X3_2;
        settings.frame_view.enabled = true;
        settings.controls.players[3] = PlayerInput::default_gamepad(InputDevice::Gamepad {
            guid: "abc".into(),
            index: 1,
            name: "Pad".into(),
        });
        settings.ui.push_recent("/games/a.nsp".into());
        let text = toml::to_string_pretty(&settings).unwrap();
        let (back, warnings) = load(&text);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(back, settings);
    }

    #[test]
    fn partial_file_fills_defaults() {
        let (back, _) = load("[graphics]\nresolution_scale = \"X2\"\n");
        assert_eq!(back.graphics.resolution_scale, ResolutionScale::X2);
        assert_eq!(back.controls.players.len(), MAX_PLAYERS);
        assert_eq!(back.ui.hotkeys.len(), HotkeyAction::ALL.len());
    }

    #[test]
    fn unlimited_speed_round_trips() {
        let mut settings = Settings::default();
        settings.graphics.speed_limit = None;
        let text = toml::to_string_pretty(&settings).unwrap();
        assert!(text.contains("speed_limit = 0"), "{text}");
        assert_eq!(load(&text).0.graphics.speed_limit, None);
    }

    #[test]
    fn migrates_version_0_files() {
        let (settings, warnings) = load(include_str!("testdata/settings_v0.toml"));
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(settings.version, SETTINGS_VERSION);
        assert_eq!(settings.graphics.resolution_scale, ResolutionScale::X2);
        assert_eq!(settings.graphics.speed_limit, Some(150));
        assert_eq!(settings.paths.game_dirs, [PathBuf::from("/games/switch")]);
        assert_eq!(
            settings.controls.players[0],
            PlayerInput::default_keyboard()
        );
        assert!(matches!(
            &settings.controls.players[1].device,
            InputDevice::Gamepad { name, .. } if name == "8BitDo Pro 2"
        ));
        assert_eq!(settings.ui, UiSettings::default());
    }

    #[test]
    fn a_bad_field_keeps_the_rest_of_its_section() {
        let text = "[graphics]\nresolution_scale = \"X9000\"\nfsr_sharpness = 70\n\
                    [system]\ndocked = false\n";
        let (settings, warnings) = load(text);
        assert_eq!(settings.graphics.resolution_scale, ResolutionScale::X1);
        assert_eq!(settings.graphics.fsr_sharpness, 70);
        assert!(!settings.system.docked);
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].starts_with("graphics.resolution_scale"));
    }

    #[test]
    fn a_section_of_the_wrong_type_is_reset() {
        let (settings, warnings) = load("audio = 3\n[cpu]\nfastmem = false\n");
        assert_eq!(settings.audio, AudioSettings::default());
        assert!(!settings.cpu.fastmem);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn newer_files_load_with_a_warning() {
        let (settings, warnings) = load("version = 99\nfuture = 1\n[cpu]\nfastmem = false\n");
        assert!(!settings.cpu.fastmem);
        assert_eq!(settings.version, SETTINGS_VERSION);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn not_toml_is_an_error() {
        assert!(Settings::from_toml("[graphics").is_err());
    }

    #[test]
    fn normalize_clamps_and_pads() {
        let mut settings = Settings::default();
        settings.controls.players.truncate(2);
        settings.controls.players[1].left_stick.deadzone = 4.0;
        settings.controls.players[1].right_stick.range = f32::NAN;
        settings.controls.players[1].vibration_strength = 250;
        settings.graphics.fsr_sharpness = 200;
        settings.graphics.speed_limit = Some(0);
        settings.frame_view.opacity = -1.0;
        settings.ui.hotkeys.clear();
        settings.normalize();
        let p = &settings.controls.players;
        assert_eq!(p.len(), MAX_PLAYERS);
        assert_eq!(p[1].left_stick.deadzone, 0.9);
        assert_eq!(p[1].right_stick.range, 1.0);
        assert_eq!(p[1].vibration_strength, 100);
        assert_eq!(settings.graphics.fsr_sharpness, 100);
        assert_eq!(settings.graphics.speed_limit, None);
        assert_eq!(settings.frame_view.opacity, 0.1);
        assert_eq!(settings.ui.hotkeys.len(), HotkeyAction::ALL.len());
    }

    #[test]
    fn user_hotkeys_survive_and_missing_ones_are_added() {
        let text = "[ui.hotkeys.Fullscreen]\nkeys = \"Alt+Enter\"\ncontroller = []\n";
        let (settings, warnings) = load(text);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            settings.ui.hotkeys[&HotkeyAction::Fullscreen].keys,
            "Alt+Enter"
        );
        assert_eq!(settings.ui.hotkeys[&HotkeyAction::ContinuePause].keys, "F4");
    }

    #[test]
    fn save_then_load() {
        let dir = std::env::temp_dir().join(format!("lichii-settings-{}", std::process::id()));
        let path = dir.join("config.toml");
        let mut settings = Settings::default();
        settings.system.docked = false;
        settings.graphics.speed_limit = None;
        settings.save(&path).unwrap();
        assert!(!dir.join("config.toml.tmp").exists());
        assert_eq!(Settings::load(&path).unwrap(), settings);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(Settings::load(&path).unwrap(), Settings::default());
    }

    #[test]
    fn recent_files_are_unique_and_bounded() {
        let mut ui = UiSettings::default();
        for i in 0..15 {
            ui.push_recent(format!("/g/{i}.nsp").into());
        }
        ui.push_recent("/g/12.nsp".into());
        assert_eq!(ui.recent_files.len(), MAX_RECENT_FILES);
        assert_eq!(ui.recent_files[0], PathBuf::from("/g/12.nsp"));
        assert_eq!(ui.recent_files[1], PathBuf::from("/g/14.nsp"));
    }

    #[test]
    fn controller_buttons() {
        use NpadButton::*;
        assert!(ControllerType::Switch2ProController.has_button(GL));
        assert!(!ControllerType::ProController.has_button(C));
        assert!(ControllerType::Switch2RightJoycon.has_button(C));
        assert!(!ControllerType::Switch2LeftJoycon.has_button(C));
        assert!(!ControllerType::LeftJoycon.has_button(A));
        assert!(ControllerType::DualJoycons.has_button(RightSR));
        assert!(!ControllerType::RightJoycon.has_left_stick());
        assert!(!ControllerType::GameCube.has_button(ZL));
        assert!(!ControllerType::GameCube.has_button(ZR));
        assert!(ControllerType::GameCube.has_analog_triggers());
    }

    #[test]
    fn resolution_scale_is_exact() {
        assert_eq!(ResolutionScale::X3_2.scale(1080), 1620);
        assert_eq!(ResolutionScale::X1_4.scale(720), 180);
    }
}
