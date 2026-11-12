//! The settings model. Every crate reads it; the frontend edits it and saves it as TOML.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::ConsoleGeneration;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Settings {
    pub system: SystemSettings,
    pub cpu: CpuSettings,
    pub graphics: GraphicsSettings,
    pub frame_view: FrameViewSettings,
    pub controls: ControlSettings,
    pub audio: AudioSettings,
    pub paths: PathSettings,
    pub debug: DebugSettings,
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
    /// Loads settings from \`path\`; a missing file yields the defaults.
    pub fn load(path: &Path) -> Result<Self, SettingsError> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|source| SettingsError::Parse {
                path: path.to_path_buf(),
                source: Box::new(source),
            }),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(SettingsError::Io {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), SettingsError> {
        let text = toml::to_string_pretty(self)?;
        let io = |source| SettingsError::Io {
            path: path.to_path_buf(),
            source,
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(io)?;
        }
        std::fs::write(path, text).map_err(io)
    }
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
}

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

/// Guest DRAM size. \`Auto\` picks 4 GiB for Switch 1 and 12 GiB for Switch 2.
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
    /// Vulkan physical device name; \`None\` picks the first discrete GPU.
    pub device: Option<String>,
    pub resolution_scale: ResolutionScale,
    pub scaling_filter: ScalingFilter,
    /// FSR sharpening, 0 (none) to 100 (most).
    pub fsr_sharpness: u8,
    pub anti_aliasing: AntiAliasing,
    pub aspect_ratio: AspectRatio,
    pub present_mode: PresentMode,
    pub fullscreen: FullscreenMode,
    /// Speed limit in percent of real time; \`None\` runs unlimited.
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

/// One host input bound to one guest input.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum InputBinding {
    #[default]
    None,
    /// A keyboard key, by its \`winit\` \`KeyCode\` name (e.g. \`KeyZ\`, \`ArrowUp\`).
    Key(String),
    /// A gamepad button by SDL name (e.g. \`a\`, \`leftshoulder\`, \`dpup\`).
    GamepadButton { button: String },
    /// A gamepad axis past \`threshold\` (0..1) in \`direction\` (+1 or -1).
    GamepadAxis {
        axis: String,
        threshold: f32,
        direction: i8,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct StickBinding {
    /// Analog source: an SDL axis pair (\`leftx\`/\`lefty\`) on a gamepad.
    pub analog_x: Option<String>,
    pub analog_y: Option<String>,
    pub invert_x: bool,
    pub invert_y: bool,
    /// Digital sources for keyboards.
    pub up: InputBinding,
    pub down: InputBinding,
    pub left: InputBinding,
    pub right: InputBinding,
    /// Held to scale digital input by \`modifier_scale\`.
    pub modifier: InputBinding,
    pub modifier_scale: f32,
    /// 0..1.
    pub deadzone: f32,
    /// 0..1.5, how far the output reaches before clamping.
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

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum InputDevice {
    #[default]
    None,
    Keyboard,
    /// A gamepad by SDL GUID; \`index\` separates identical controllers.
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
    pub left_stick: StickBinding,
    pub right_stick: StickBinding,
    pub motion_enabled: bool,
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
            left_stick: StickBinding::default(),
            right_stick: StickBinding::default(),
            motion_enabled: true,
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
    /// Overrides for the profile directories; \`None\` uses \`UserPaths\`.
    pub keys_dir: Option<PathBuf>,
    pub nand_dir: Option<PathBuf>,
    pub sdmc_dir: Option<PathBuf>,
    pub mods_dir: Option<PathBuf>,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_toml() {
        let mut settings = Settings::default();
        settings.graphics.resolution_scale = ResolutionScale::X3_2;
        settings.frame_view.enabled = true;
        let text = toml::to_string_pretty(&settings).unwrap();
        let back: Settings = toml::from_str(&text).unwrap();
        assert_eq!(back, settings);
    }

    #[test]
    fn partial_file_fills_defaults() {
        let back: Settings = toml::from_str("[graphics]\nresolution_scale = \"X2\"\n").unwrap();
        assert_eq!(back.graphics.resolution_scale, ResolutionScale::X2);
        assert_eq!(back.controls.players.len(), MAX_PLAYERS);
    }

    #[test]
    fn resolution_scale_is_exact() {
        assert_eq!(ResolutionScale::X3_2.scale(1080), 1620);
        assert_eq!(ResolutionScale::X1_4.scale(720), 180);
    }
}
