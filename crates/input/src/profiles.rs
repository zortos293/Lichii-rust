//! Saved input profiles: one TOML file per profile under `UserPaths::input_profiles_dir`.

use std::path::{Path, PathBuf};

use lichii_common::settings::{PlayerInput, write_atomic};

#[derive(Debug, thiserror::Error)]
pub enum ProfileError {
    #[error(
        "profile names may not be empty or contain path separators, dots or control characters"
    )]
    InvalidName,
    #[error("profile {0:?} does not exist")]
    NotFound(String),
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {source}")]
    Parse {
        path: PathBuf,
        source: Box<toml::de::Error>,
    },
    #[error("serializing profile: {0}")]
    Serialize(#[from] toml::ser::Error),
}

/// The profiles directory. Loading a profile keeps the player's `connected` flag and records the
/// profile name in `PlayerInput::profile`.
pub struct ProfileStore {
    dir: PathBuf,
}

impl ProfileStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Profile names, sorted. A missing directory has none.
    pub fn list(&self) -> Result<Vec<String>, ProfileError> {
        let entries = match std::fs::read_dir(&self.dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(source) => return Err(self.io(&self.dir, source)),
        };
        let mut names: Vec<String> = entries
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "toml"))
            .filter_map(|p| p.file_stem()?.to_str().map(str::to_owned))
            .filter(|name| valid_name(name))
            .collect();
        names.sort_by_key(|n| n.to_lowercase());
        Ok(names)
    }

    pub fn save(&self, name: &str, player: &PlayerInput) -> Result<(), ProfileError> {
        let path = self.path(name)?;
        let profile = PlayerInput {
            profile: None,
            ..player.clone()
        };
        let text = toml::to_string_pretty(&profile)?;
        write_atomic(&path, text.as_bytes()).map_err(|e| self.io(&path, e))
    }

    /// Applies profile `name` to `player`.
    pub fn load_into(&self, name: &str, player: &mut PlayerInput) -> Result<(), ProfileError> {
        let path = self.path(name)?;
        let text = std::fs::read_to_string(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ProfileError::NotFound(name.to_owned())
            } else {
                self.io(&path, e)
            }
        })?;
        let loaded: PlayerInput = toml::from_str(&text).map_err(|source| ProfileError::Parse {
            path: path.clone(),
            source: Box::new(source),
        })?;
        *player = PlayerInput {
            connected: player.connected,
            profile: Some(name.to_owned()),
            ..loaded
        };
        Ok(())
    }

    pub fn delete(&self, name: &str) -> Result<(), ProfileError> {
        let path = self.path(name)?;
        std::fs::remove_file(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ProfileError::NotFound(name.to_owned())
            } else {
                self.io(&path, e)
            }
        })
    }

    fn path(&self, name: &str) -> Result<PathBuf, ProfileError> {
        if !valid_name(name) {
            return Err(ProfileError::InvalidName);
        }
        Ok(self.dir.join(format!("{name}.toml")))
    }

    fn io(&self, path: &Path, source: std::io::Error) -> ProfileError {
        ProfileError::Io {
            path: path.to_path_buf(),
            source,
        }
    }
}

/// Names become file names on every OS, so keep them to what Windows and Linux both accept.
pub fn valid_name(name: &str) -> bool {
    let trimmed = name.trim();
    !trimmed.is_empty()
        && trimmed == name
        && name.len() <= 64
        && !name.chars().any(|c| {
            c.is_control()
                || matches!(
                    c,
                    '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '.'
                )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::settings::{InputBinding, InputDevice, NpadButton};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("lichii-profiles-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn save_list_load_delete() {
        let tmp = TempDir::new("cycle");
        let store = ProfileStore::new(tmp.0.join("input"));
        assert!(store.list().unwrap().is_empty());

        let mut saved = PlayerInput::default_gamepad(InputDevice::Gamepad {
            guid: "g".into(),
            index: 0,
            name: "Pad".into(),
        });
        saved
            .buttons
            .insert(NpadButton::A, InputBinding::Key("KeyJ".into()));
        saved.left_stick.deadzone = 0.3;
        store.save("Racing", &saved).unwrap();
        store
            .save("arcade stick", &PlayerInput::default_keyboard())
            .unwrap();
        assert_eq!(store.list().unwrap(), ["arcade stick", "Racing"]);

        let mut player = PlayerInput {
            connected: false,
            ..PlayerInput::default()
        };
        store.load_into("Racing", &mut player).unwrap();
        assert!(!player.connected);
        assert_eq!(player.profile.as_deref(), Some("Racing"));
        assert_eq!(player.left_stick.deadzone, 0.3);
        assert_eq!(player.buttons, saved.buttons);
        assert_eq!(player.device, saved.device);

        store.delete("Racing").unwrap();
        assert_eq!(store.list().unwrap(), ["arcade stick"]);
        assert!(matches!(
            store.load_into("Racing", &mut player),
            Err(ProfileError::NotFound(_))
        ));
        assert!(matches!(
            store.delete("Racing"),
            Err(ProfileError::NotFound(_))
        ));
    }

    #[test]
    fn names_cannot_escape_the_directory() {
        let store = ProfileStore::new("unused");
        for bad in ["", " x", "../x", "a/b", "a\\b", "con:", "x.toml", "tab\t"] {
            assert!(
                matches!(
                    store.save(bad, &PlayerInput::default()),
                    Err(ProfileError::InvalidName)
                ),
                "{bad:?}"
            );
        }
        assert!(valid_name("Player 1 - Pro"));
    }

    #[test]
    fn a_corrupt_profile_is_a_parse_error() {
        let tmp = TempDir::new("corrupt");
        std::fs::create_dir_all(&tmp.0).unwrap();
        std::fs::write(tmp.0.join("bad.toml"), "connected = [").unwrap();
        let store = ProfileStore::new(&tmp.0);
        assert!(matches!(
            store.load_into("bad", &mut PlayerInput::default()),
            Err(ProfileError::Parse { .. })
        ));
    }
}
