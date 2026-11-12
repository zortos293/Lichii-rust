use std::path::PathBuf;

/// Directories Lichii reads and writes. A `user/` directory next to the executable makes the
/// install portable and wins over the platform directories.
#[derive(Clone, Debug)]
pub struct UserPaths {
    pub config_dir: PathBuf,
    pub data_dir: PathBuf,
    pub cache_dir: PathBuf,
}

impl UserPaths {
    pub fn discover() -> Self {
        if let Some(portable) = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(|dir| dir.join("user")))
            .filter(|dir| dir.is_dir())
        {
            return Self {
                config_dir: portable.join("config"),
                data_dir: portable.clone(),
                cache_dir: portable.join("cache"),
            };
        }
        let dirs = directories::ProjectDirs::from("", "", "lichii")
            .expect("no home directory to place Lichii's profile in");
        Self {
            config_dir: dirs.config_dir().to_path_buf(),
            data_dir: dirs.data_dir().to_path_buf(),
            cache_dir: dirs.cache_dir().to_path_buf(),
        }
    }

    pub fn settings_file(&self) -> PathBuf {
        self.config_dir.join("config.toml")
    }

    pub fn keys_dir(&self) -> PathBuf {
        self.data_dir.join("keys")
    }

    pub fn nand_dir(&self) -> PathBuf {
        self.data_dir.join("nand")
    }

    pub fn sdmc_dir(&self) -> PathBuf {
        self.data_dir.join("sdmc")
    }

    pub fn shader_cache_dir(&self) -> PathBuf {
        self.cache_dir.join("shader")
    }

    pub fn input_profiles_dir(&self) -> PathBuf {
        self.config_dir.join("input")
    }
}
