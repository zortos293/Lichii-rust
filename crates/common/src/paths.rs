use std::path::{Path, PathBuf};

/// Directories Lichii reads and writes, laid out like the reference emulator's: a `user/`
/// directory next to the executable makes the install portable and wins; otherwise everything
/// lives under `%APPDATA%\lichii` on Windows, and under `lichii` in the XDG data, config and
/// cache directories elsewhere.
#[derive(Clone, Debug)]
pub struct UserPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
    /// Where earlier Lichii builds kept their data (`%APPDATA%\lichii\data` on Windows). Its
    /// `keys`, `nand` and `sdmc` folders are used when `data_dir` has none of its own.
    pub legacy_data_dir: Option<PathBuf>,
}

impl UserPaths {
    pub fn discover() -> Self {
        if let Some(portable) = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("user")))
            .filter(|dir| dir.is_dir())
        {
            return Self::portable(portable);
        }
        let base =
            directories::BaseDirs::new().expect("no home directory to place Lichii's profile in");
        let paths = if cfg!(windows) {
            Self::windows(base.config_dir())
        } else {
            let xdg = |var: &str, fallback: &str| {
                std::env::var_os(var)
                    .map(PathBuf::from)
                    .filter(|dir| dir.is_absolute())
                    .unwrap_or_else(|| base.home_dir().join(fallback))
            };
            Self::xdg(
                &xdg("XDG_DATA_HOME", ".local/share"),
                &xdg("XDG_CONFIG_HOME", ".config"),
                &xdg("XDG_CACHE_HOME", ".cache"),
            )
        };
        let legacy = directories::ProjectDirs::from("", "", "lichii")
            .map(|dirs| dirs.data_dir().to_path_buf());
        paths.with_legacy_data_dir(legacy)
    }

    /// A portable profile: everything under `root`.
    pub fn portable(root: PathBuf) -> Self {
        Self {
            config_dir: root.join("config"),
            cache_dir: root.join("cache"),
            data_dir: root,
            legacy_data_dir: None,
        }
    }

    /// The Windows profile under the roaming application data folder `app_data`.
    pub fn windows(app_data: &Path) -> Self {
        Self::portable(app_data.join("lichii"))
    }

    /// The profile in the XDG base directories.
    pub fn xdg(data_home: &Path, config_home: &Path, cache_home: &Path) -> Self {
        Self {
            config_dir: config_home.join("lichii"),
            data_dir: data_home.join("lichii"),
            cache_dir: cache_home.join("lichii"),
            legacy_data_dir: None,
        }
    }

    /// Keeps `legacy` as the fallback data folder, unless it is the data folder itself.
    pub fn with_legacy_data_dir(mut self, legacy: Option<PathBuf>) -> Self {
        self.legacy_data_dir = legacy.filter(|dir| *dir != self.data_dir);
        self
    }

    pub fn settings_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    pub fn keys_dir(&self) -> PathBuf {
        self.data_item("keys")
    }

    pub fn nand_dir(&self) -> PathBuf {
        self.data_item("nand")
    }

    pub fn sdmc_dir(&self) -> PathBuf {
        self.data_item("sdmc")
    }

    pub fn shader_cache_dir(&self) -> PathBuf {
        self.cache_dir.join("shader")
    }

    pub fn input_profiles_dir(&self) -> PathBuf {
        self.config_dir.join("input")
    }

    /// `name` in the data folder, or in the legacy data folder when only that one has it.
    fn data_item(&self, name: &str) -> PathBuf {
        let current = self.data_dir.join(name);
        match &self.legacy_data_dir {
            Some(legacy) if !current.exists() && legacy.join(name).exists() => legacy.join(name),
            _ => current,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("lichii-paths-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn windows_keeps_everything_under_app_data_lichii_like_the_reference() {
        let app_data = PathBuf::from("C:/Users/u/AppData/Roaming");
        let root = app_data.join("lichii");
        let paths = UserPaths::windows(&app_data);
        assert_eq!(paths.data_dir, root);
        assert_eq!(paths.keys_dir(), root.join("keys"));
        assert_eq!(paths.nand_dir(), root.join("nand"));
        assert_eq!(paths.sdmc_dir(), root.join("sdmc"));
        assert_eq!(
            paths.settings_file(),
            root.join("config").join("config.toml")
        );
        assert_eq!(paths.shader_cache_dir(), root.join("cache").join("shader"));
    }

    #[test]
    fn xdg_splits_data_config_and_cache_like_the_reference() {
        let home = PathBuf::from("/home/u");
        let paths = UserPaths::xdg(
            &home.join(".local/share"),
            &home.join(".config"),
            &home.join(".cache"),
        );
        assert_eq!(paths.keys_dir(), home.join(".local/share/lichii/keys"));
        assert_eq!(
            paths.settings_file(),
            home.join(".config/lichii/config.toml")
        );
        assert_eq!(paths.shader_cache_dir(), home.join(".cache/lichii/shader"));
        let same = paths
            .clone()
            .with_legacy_data_dir(Some(home.join(".local/share/lichii")));
        assert_eq!(
            same.legacy_data_dir, None,
            "a legacy folder equal to the data folder"
        );
    }

    #[test]
    fn portable_profiles_keep_everything_in_user() {
        let user = PathBuf::from("/opt/lichii/user");
        let paths = UserPaths::portable(user.clone());
        assert_eq!(paths.keys_dir(), user.join("keys"));
        assert_eq!(
            paths.settings_file(),
            user.join("config").join("config.toml")
        );
        assert_eq!(paths.shader_cache_dir(), user.join("cache").join("shader"));
    }

    #[test]
    fn the_old_data_folder_serves_what_the_reference_location_lacks() {
        let app_data = temp("legacy");
        let root = app_data.join("lichii");
        let legacy = root.join("data");
        std::fs::create_dir_all(legacy.join("keys")).unwrap();
        std::fs::create_dir_all(legacy.join("nand")).unwrap();
        std::fs::create_dir_all(root.join("nand")).unwrap();
        let paths = UserPaths::windows(&app_data).with_legacy_data_dir(Some(legacy.clone()));
        assert_eq!(
            paths.keys_dir(),
            legacy.join("keys"),
            "only the old folder has keys"
        );
        assert_eq!(
            paths.nand_dir(),
            root.join("nand"),
            "the reference location wins"
        );
        assert_eq!(
            paths.sdmc_dir(),
            root.join("sdmc"),
            "neither has one: the new location"
        );
        std::fs::create_dir_all(root.join("keys")).unwrap();
        assert_eq!(paths.keys_dir(), root.join("keys"));
        std::fs::remove_dir_all(&app_data).unwrap();
    }
}
