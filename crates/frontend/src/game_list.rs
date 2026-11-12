//! The game list: files and loose ExeFS directories under `PathSettings::game_dirs`, scanned on a
//! background thread. Titles show their file names until the loader crate provides metadata.

use std::path::{Path, PathBuf};

use crossbeam_channel::Receiver;

/// How deep below a game directory to look.
const MAX_DEPTH: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GameKind {
    Nsp,
    Xci,
    Nca,
    Nro,
    Nso,
    /// A directory holding `main` and `main.npdm` (or an `exefs/` subdirectory with them).
    ExeFs,
}

impl GameKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Nsp => "NSP",
            Self::Xci => "XCI",
            Self::Nca => "NCA",
            Self::Nro => "NRO",
            Self::Nso => "NSO",
            Self::ExeFs => "ExeFS",
        }
    }

    /// What `path` is, if Lichii can boot it.
    pub fn detect(path: &Path) -> Option<Self> {
        if path.is_dir() {
            return exefs_root(path).map(|_| Self::ExeFs);
        }
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "nsp" => Some(Self::Nsp),
            "xci" => Some(Self::Xci),
            "nca" => Some(Self::Nca),
            "nro" => Some(Self::Nro),
            "nso" => Some(Self::Nso),
            _ => None,
        }
    }
}

/// The directory that holds `main` and `main.npdm`: `dir` itself or `dir/exefs`.
pub fn exefs_root(dir: &Path) -> Option<PathBuf> {
    [dir.to_path_buf(), dir.join("exefs")]
        .into_iter()
        .find(|d| d.join("main").is_file() && d.join("main.npdm").is_file())
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameEntry {
    pub path: PathBuf,
    pub name: String,
    pub kind: GameKind,
    /// File size in bytes; 0 for directories.
    pub size: u64,
}

/// Every bootable file or ExeFS directory under `dirs`, sorted by name.
pub fn scan(dirs: &[PathBuf]) -> Vec<GameEntry> {
    let mut out = Vec::new();
    for dir in dirs {
        scan_dir(dir, 0, &mut out);
    }
    out.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.path.cmp(&b.path))
    });
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn scan_dir(dir: &Path, depth: usize, out: &mut Vec<GameEntry>) {
    if exefs_root(dir).is_some() {
        out.push(entry(dir, GameKind::ExeFs, 0));
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for item in entries.filter_map(Result::ok) {
        let path = item.path();
        let Ok(meta) = item.metadata() else {
            continue;
        };
        if meta.is_dir() {
            if depth < MAX_DEPTH {
                scan_dir(&path, depth + 1, out);
            }
        } else if let Some(kind) = GameKind::detect(&path) {
            out.push(entry(&path, kind, meta.len()));
        }
    }
}

fn entry(path: &Path, kind: GameKind, size: u64) -> GameEntry {
    let name = if kind == GameKind::ExeFs && path.file_name().is_some_and(|n| n == "exefs") {
        path.parent().unwrap_or(path)
    } else {
        path
    };
    let name = match kind {
        GameKind::ExeFs => name.file_name(),
        _ => name.file_stem(),
    };
    GameEntry {
        path: path.to_path_buf(),
        name: name.map_or_else(
            || path.display().to_string(),
            |n| n.to_string_lossy().into_owned(),
        ),
        kind,
        size,
    }
}

/// The list the UI shows, refreshed in the background.
#[derive(Default)]
pub struct GameList {
    pub entries: Vec<GameEntry>,
    pub filter: String,
    pending: Option<Receiver<Vec<GameEntry>>>,
}

impl GameList {
    /// Starts a rescan; `wake` runs on the scanning thread when it finishes.
    pub fn refresh(&mut self, dirs: Vec<PathBuf>, wake: impl FnOnce() + Send + 'static) {
        let (tx, rx) = crossbeam_channel::bounded(1);
        std::thread::Builder::new()
            .name("game list scan".into())
            .spawn(move || {
                let _ = tx.send(scan(&dirs));
                wake();
            })
            .expect("spawning the game list scanner");
        self.pending = Some(rx);
    }

    pub fn is_scanning(&self) -> bool {
        self.pending.is_some()
    }

    /// Takes a finished scan, if any.
    pub fn poll(&mut self) {
        if let Some(entries) = self.pending.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.entries = entries;
            self.pending = None;
        }
    }

    /// Entries whose name or path contains the filter, case-insensitively.
    pub fn visible(&self) -> impl Iterator<Item = &GameEntry> {
        let filter = self.filter.trim().to_lowercase();
        self.entries.iter().filter(move |e| {
            filter.is_empty()
                || e.name.to_lowercase().contains(&filter)
                || e.path.to_string_lossy().to_lowercase().contains(&filter)
        })
    }
}

pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("lichii-games-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn file(&self, rel: &str, len: usize) {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, vec![0u8; len]).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn finds_files_and_exefs_directories() {
        let tmp = TempDir::new("scan");
        tmp.file("Zelda.NSP", 10);
        tmp.file("sub/Kart.xci", 20);
        tmp.file("sub/readme.txt", 1);
        tmp.file("homebrew/app.nro", 5);
        tmp.file("Title1/exefs/main", 1);
        tmp.file("Title1/exefs/main.npdm", 1);
        tmp.file("Title1/romfs/data.bin", 1);
        tmp.file("Loose/main", 1);
        tmp.file("Loose/main.npdm", 1);
        tmp.file("Loose/rtld", 1);
        let games = scan(std::slice::from_ref(&tmp.0));
        let summary: Vec<(&str, GameKind, u64)> = games
            .iter()
            .map(|g| (g.name.as_str(), g.kind, g.size))
            .collect();
        assert_eq!(
            summary,
            [
                ("app", GameKind::Nro, 5),
                ("Kart", GameKind::Xci, 20),
                ("Loose", GameKind::ExeFs, 0),
                ("Title1", GameKind::ExeFs, 0),
                ("Zelda", GameKind::Nsp, 10),
            ]
        );
    }

    #[test]
    fn missing_directories_are_skipped_and_duplicates_merged() {
        let tmp = TempDir::new("dupes");
        tmp.file("a.nso", 1);
        let games = scan(&[tmp.0.clone(), tmp.0.join("missing"), tmp.0.clone()]);
        assert_eq!(games.len(), 1);
    }

    #[test]
    fn detects_kinds() {
        let tmp = TempDir::new("detect");
        tmp.file("x/exefs/main", 1);
        tmp.file("x/exefs/main.npdm", 1);
        assert_eq!(GameKind::detect(&tmp.0.join("x")), Some(GameKind::ExeFs));
        assert_eq!(
            GameKind::detect(&tmp.0.join("x/exefs")),
            Some(GameKind::ExeFs)
        );
        assert_eq!(GameKind::detect(Path::new("a.Nca")), Some(GameKind::Nca));
        assert_eq!(GameKind::detect(Path::new("a.zip")), None);
        assert_eq!(GameKind::detect(&tmp.0), None);
    }

    #[test]
    fn filter_matches_name_or_path() {
        let entry = |name: &str, path: &str| GameEntry {
            path: path.into(),
            name: name.into(),
            kind: GameKind::Nsp,
            size: 0,
        };
        let list = GameList {
            entries: vec![
                entry("Kart", "/g/racing/kart.nsp"),
                entry("Zelda", "/g/zelda.nsp"),
            ],
            filter: "RACING".into(),
            pending: None,
        };
        let names: Vec<_> = list.visible().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["Kart"]);
    }

    #[test]
    fn sizes() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(3 * 1024 * 1024 * 1024 / 2), "1.5 GiB");
    }
}
