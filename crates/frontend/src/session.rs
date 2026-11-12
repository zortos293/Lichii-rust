//! One emulation session: the title being run and whether it is paused.
//!
//! `lichii-core` has no `System` yet, so a session validates the title and tracks its run state
//! while the renderer shows nothing. When `System` lands it is owned here and driven by
//! `pause`, `resume` and `Drop` (stop).

use std::path::{Path, PathBuf};

use crate::game_list::GameKind;

#[derive(Debug, PartialEq, Eq)]
pub enum BootError {
    NotFound(PathBuf),
    Unsupported(PathBuf),
}

impl std::fmt::Display for BootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(path) => write!(f, "{} does not exist", path.display()),
            Self::Unsupported(path) => write!(
                f,
                "{} is not an NSP, XCI, NCA, NRO, NSO or ExeFS directory",
                path.display()
            ),
        }
    }
}

pub struct Session {
    pub path: PathBuf,
    pub title: String,
    paused: bool,
}

impl Session {
    pub fn boot(path: &Path) -> Result<Self, BootError> {
        if !path.exists() {
            return Err(BootError::NotFound(path.to_path_buf()));
        }
        let kind = GameKind::detect(path).ok_or_else(|| BootError::Unsupported(path.into()))?;
        let name = if kind == GameKind::ExeFs && path.ends_with("exefs") {
            path.parent().unwrap_or(path)
        } else {
            path
        };
        let title = match kind {
            GameKind::ExeFs => name.file_name(),
            _ => name.file_stem(),
        }
        .map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into(),
        );
        tracing::info!("booting {} ({})", path.display(), kind.label());
        Ok(Self {
            path: path.to_path_buf(),
            title,
            paused: false,
        })
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boots_only_existing_known_titles() {
        let missing = Path::new("/definitely/not/here.nsp");
        assert_eq!(
            Session::boot(missing).err(),
            Some(BootError::NotFound(missing.into()))
        );
        let dir = std::env::temp_dir().join(format!("lichii-session-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("Title/exefs")).unwrap();
        let text = dir.join("notes.txt");
        std::fs::write(&text, "x").unwrap();
        assert!(matches!(
            Session::boot(&text),
            Err(BootError::Unsupported(_))
        ));
        let game = dir.join("Some Game.nro");
        std::fs::write(&game, "x").unwrap();
        let mut session = Session::boot(&game).unwrap();
        assert_eq!(session.title, "Some Game");
        session.toggle_pause();
        assert!(session.is_paused());
        session.resume();
        assert!(!session.is_paused());
        std::fs::write(dir.join("Title/exefs/main"), "x").unwrap();
        std::fs::write(dir.join("Title/exefs/main.npdm"), "x").unwrap();
        assert_eq!(
            Session::boot(&dir.join("Title/exefs")).unwrap().title,
            "Title"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
