//! Per-game configuration, kept as Lichii keeps it: `custom/<TITLE ID>.toml` beside the global
//! settings (named after the file when a title has no program id) holds only the settings a game
//! overrides. They are merged over the global settings when the game boots and while it runs.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use lichii_common::settings::{Settings, write_atomic};

/// The settings a game can override, as `section.key`.
pub const KEYS: &[&str] = &[
    "system.preferred_console",
    "system.docked",
    "system.language",
    "system.region",
    "system.memory_layout",
    "system.time_zone",
    "system.network_enabled",
    "cpu.backend",
    "cpu.accuracy",
    "cpu.fastmem",
    "graphics.resolution_scale",
    "graphics.scaling_filter",
    "graphics.fsr_sharpness",
    "graphics.anti_aliasing",
    "graphics.aspect_ratio",
    "graphics.speed_limit",
    "graphics.gpu_accuracy",
    "graphics.anisotropic_filtering",
    "graphics.astc_decode",
    "graphics.astc_recompression",
    "graphics.vram_usage",
    "graphics.async_shader_compilation",
    "graphics.disk_shader_cache",
    "audio.volume",
    "audio.muted",
];

/// One game's overrides and the file they live in.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GameConfig {
    path: PathBuf,
    overrides: toml::Table,
}

impl GameConfig {
    /// Where the configuration of the title at `title` with `program_id` lives.
    pub fn path(config_dir: &Path, program_id: u64, title: &Path) -> PathBuf {
        let name = match program_id {
            0 => title
                .file_name()
                .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
            id => format!("{id:016X}"),
        };
        config_dir.join("custom").join(format!("{name}.toml"))
    }

    /// Reads the overrides at `path`; a missing file has none. Settings a game cannot override,
    /// and values that do not parse, are dropped with a warning.
    pub fn load(path: PathBuf) -> Self {
        let parsed = match std::fs::read_to_string(&path) {
            Ok(text) => toml::from_str::<toml::Table>(&text)
                .inspect_err(|e| tracing::warn!("{}: {e}", path.display()))
                .unwrap_or_default(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => toml::Table::new(),
            Err(e) => {
                tracing::warn!("{}: {e}", path.display());
                toml::Table::new()
            }
        };
        let base = table(&Settings::default());
        let mut overrides = toml::Table::new();
        for key in KEYS {
            let Some(value) = get(&parsed, key) else {
                continue;
            };
            let mut probe = base.clone();
            set(&mut probe, key, value.clone());
            match parse(&probe) {
                Some((_, warnings)) if warnings.is_empty() => {
                    set(&mut overrides, key, value.clone())
                }
                _ => tracing::warn!("{}: {key} = {value} is not a valid value", path.display()),
            }
        }
        Self { path, overrides }
    }

    pub fn file(&self) -> &Path {
        &self.path
    }

    pub fn is_empty(&self) -> bool {
        self.overridden().is_empty()
    }

    /// The settings this game overrides.
    pub fn overridden(&self) -> BTreeSet<&'static str> {
        KEYS.iter()
            .copied()
            .filter(|key| get(&self.overrides, key).is_some())
            .collect()
    }

    /// The global settings with this game's overrides applied.
    pub fn apply(&self, global: &Settings) -> Settings {
        let mut merged = table(global);
        for key in KEYS {
            if let Some(value) = get(&self.overrides, key) {
                set(&mut merged, key, value.clone());
            }
        }
        parse(&merged).map_or_else(|| global.clone(), |(settings, _)| settings)
    }

    /// Makes `keys` the overridden settings, with their values from `settings`.
    pub fn set_overrides(&mut self, settings: &Settings, keys: &BTreeSet<&'static str>) {
        let source = table(settings);
        self.overrides = toml::Table::new();
        for key in keys {
            if let Some(value) = get(&source, key) {
                set(&mut self.overrides, key, value.clone());
            }
        }
    }

    /// Applies a change made while the game runs (the status bar, a hotkey): settings this game
    /// overrides change here, every other one in `global`.
    pub fn edit(&mut self, global: &mut Settings, edit: impl Fn(&mut Settings)) {
        let keys = self.overridden();
        let mut effective = self.apply(global);
        edit(&mut effective);
        let before = table(global);
        edit(global);
        let mut after = table(global);
        for key in &keys {
            if let Some(value) = get(&before, key) {
                set(&mut after, key, value.clone());
            }
        }
        if let Some((settings, _)) = parse(&after) {
            *global = settings;
        }
        self.set_overrides(&effective, &keys);
    }

    /// Writes the overrides, or removes the file when there are none.
    pub fn save(&self) -> std::io::Result<()> {
        if self.is_empty() {
            return match std::fs::remove_file(&self.path) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
                _ => Ok(()),
            };
        }
        let text = toml::to_string_pretty(&self.overrides).map_err(std::io::Error::other)?;
        write_atomic(&self.path, text.as_bytes())
    }
}

/// Copies one of [`KEYS`] from `from` into `to`.
pub fn copy_key(from: &Settings, to: &mut Settings, key: &str) {
    let Some(value) = get(&table(from), key).cloned() else {
        return;
    };
    let mut merged = table(to);
    set(&mut merged, key, value);
    if let Some((settings, _)) = parse(&merged) {
        *to = settings;
    }
}

/// Copies every setting that differs between `before` and `after` into `to`, leaving the rest
/// of `to` as it is.
pub fn copy_changes(before: &Settings, after: &Settings, to: &mut Settings) {
    let (before, after) = (table(before), table(after));
    let mut merged = table(to);
    let mut changed = false;
    for (name, section) in &after {
        let Some(section) = section.as_table() else {
            continue;
        };
        for (field, value) in section {
            let key = format!("{name}.{field}");
            if get(&before, &key) != Some(value) {
                set(&mut merged, &key, value.clone());
                changed = true;
            }
        }
    }
    if changed && let Some((settings, _)) = parse(&merged) {
        *to = settings;
    }
}

fn table(settings: &Settings) -> toml::Table {
    match toml::Value::try_from(settings) {
        Ok(toml::Value::Table(table)) => table,
        _ => toml::Table::new(),
    }
}

fn parse(table: &toml::Table) -> Option<(Settings, Vec<String>)> {
    Settings::from_toml(&toml::to_string(table).ok()?).ok()
}

fn get<'a>(table: &'a toml::Table, key: &str) -> Option<&'a toml::Value> {
    let (section, name) = key.split_once('.')?;
    table.get(section)?.as_table()?.get(name)
}

fn set(table: &mut toml::Table, key: &str, value: toml::Value) {
    let Some((section, name)) = key.split_once('.') else {
        return;
    };
    let section = table
        .entry(section)
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    if let Some(section) = section.as_table_mut() {
        section.insert(name.to_owned(), value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::settings::{CpuAccuracy, ResolutionScale};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("lichii-per-game-{name}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn files_are_named_after_the_title_id_or_the_file() {
        let dir = Path::new("/config");
        assert_eq!(
            GameConfig::path(dir, 0x0100_0000_0001_0000, Path::new("/g/Game.nsp")),
            Path::new("/config/custom/0100000000010000.toml")
        );
        assert_eq!(
            GameConfig::path(dir, 0, Path::new("/g/hbmenu.nro")),
            Path::new("/config/custom/hbmenu.nro.toml")
        );
    }

    #[test]
    fn overrides_are_saved_sparse_and_applied_over_global() {
        let tmp = TempDir::new("save");
        let path = GameConfig::path(&tmp.0, 0x0100_0000_0001_0000, Path::new("Game.nsp"));
        let mut global = Settings::default();
        global.graphics.resolution_scale = ResolutionScale::X2;
        global.system.docked = true;

        let mut game = global.clone();
        game.graphics.resolution_scale = ResolutionScale::X1;
        game.cpu.accuracy = CpuAccuracy::Paranoid;
        game.system.docked = false;
        let mut config = GameConfig::load(path.clone());
        assert!(config.is_empty());
        config.set_overrides(
            &game,
            &BTreeSet::from(["graphics.resolution_scale", "cpu.accuracy"]),
        );
        config.save().unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("resolution_scale") && text.contains("accuracy"));
        assert!(
            !text.contains("docked"),
            "only overridden settings are written: {text}"
        );

        global.audio.volume = 40;
        let effective = GameConfig::load(path.clone()).apply(&global);
        assert_eq!(effective.graphics.resolution_scale, ResolutionScale::X1);
        assert_eq!(effective.cpu.accuracy, CpuAccuracy::Paranoid);
        assert!(effective.system.docked, "not overridden, so global");
        assert_eq!(
            effective.audio.volume, 40,
            "global changes still reach the game"
        );

        config.set_overrides(&game, &BTreeSet::new());
        config.save().unwrap();
        assert!(!path.exists(), "a game without overrides has no file");
    }

    #[test]
    fn unknown_keys_and_invalid_values_are_dropped() {
        let tmp = TempDir::new("invalid");
        let path = tmp.0.join("custom/game.toml");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            "[graphics]\nresolution_scale = \"Huge\"\nanti_aliasing = \"Smaa\"\n\n[paths]\nnand_dir = \"/elsewhere\"\n",
        )
        .unwrap();
        let config = GameConfig::load(path);
        assert_eq!(
            config.overridden(),
            BTreeSet::from(["graphics.anti_aliasing"])
        );
    }

    #[test]
    fn live_edits_change_overrides_or_global() {
        let mut global = Settings::default();
        global.graphics.resolution_scale = ResolutionScale::X2;
        let mut game = global.clone();
        game.graphics.resolution_scale = ResolutionScale::X1;
        let mut config = GameConfig::default();
        config.set_overrides(&game, &BTreeSet::from(["graphics.resolution_scale"]));

        config.edit(&mut global, |s| {
            s.graphics.resolution_scale = ResolutionScale::X3;
            s.system.docked = !s.system.docked;
        });
        let docked = !Settings::default().system.docked;
        assert_eq!(global.graphics.resolution_scale, ResolutionScale::X2);
        assert_eq!(global.system.docked, docked);
        let effective = config.apply(&global);
        assert_eq!(effective.graphics.resolution_scale, ResolutionScale::X3);
        assert_eq!(effective.system.docked, docked);
    }

    #[test]
    fn only_changed_settings_follow_into_a_draft() {
        let before = Settings::default();
        let mut after = before.clone();
        after.graphics.resolution_scale = ResolutionScale::X3;
        let mut draft = before.clone();
        draft.cpu.accuracy = CpuAccuracy::Paranoid;
        copy_changes(&before, &after, &mut draft);
        assert_eq!(draft.graphics.resolution_scale, ResolutionScale::X3);
        assert_eq!(
            draft.cpu.accuracy,
            CpuAccuracy::Paranoid,
            "the draft's own edit stays"
        );
    }

    #[test]
    fn copying_a_key_reverts_it_to_global() {
        let global = Settings::default();
        let mut draft = global.clone();
        draft.graphics.resolution_scale = ResolutionScale::X3;
        draft.cpu.fastmem = !global.cpu.fastmem;
        copy_key(&global, &mut draft, "graphics.resolution_scale");
        assert_eq!(
            draft.graphics.resolution_scale,
            global.graphics.resolution_scale
        );
        assert_ne!(draft.cpu.fastmem, global.cpu.fastmem);
    }
}
