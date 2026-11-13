//! The game list: files and loose ExeFS directories under `PathSettings::game_dirs`, scanned on a
//! background thread. The file list arrives first; each title's name, publisher, version,
//! program id and icon follow as the loader reads them from its NACP and control data. Results
//! for files are cached under `<cache>/game_list`, keyed by path, size and modification time, so a
//! rescan only opens titles that changed.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crossbeam_channel::{Receiver, TryRecvError};
use lichii_common::ConsoleGeneration;
use lichii_common::settings::Language;
use lichii_fs::nacp::NacpLanguage;
use lichii_loader::LoadOptions;
use serde::{Deserialize, Serialize};

/// How deep below a game directory to look.
const MAX_DEPTH: usize = 4;
/// Icons are kept as `ICON_SIZE` square RGBA images.
pub const ICON_SIZE: usize = 64;

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

/// What a title's control data says about it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Metadata {
    pub title: String,
    pub publisher: String,
    pub version: String,
    /// 0 for homebrew and bare NSOs, which carry none.
    pub program_id: u64,
    pub generation: ConsoleGeneration,
}

impl Metadata {
    pub fn system_label(&self) -> &'static str {
        match self.generation {
            ConsoleGeneration::Switch1 => "Switch 1",
            ConsoleGeneration::Switch2 => "Switch 2",
        }
    }
}

/// An icon scaled to `ICON_SIZE` square, as RGBA bytes.
pub struct Icon(pub Vec<u8>);

impl Icon {
    /// Decodes the JPEG (or PNG/BMP) icon a title ships.
    fn decode(bytes: &[u8]) -> Option<Self> {
        let image = image::load_from_memory(bytes).ok()?;
        let size = ICON_SIZE as u32;
        let scaled = image.resize_exact(size, size, image::imageops::FilterType::Triangle);
        Some(Self(scaled.to_rgba8().into_raw()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameEntry {
    pub path: PathBuf,
    /// The file or folder name, shown until the metadata arrives or when there is none.
    pub name: String,
    pub kind: GameKind,
    /// File size in bytes; 0 for directories.
    pub size: u64,
    pub meta: Option<Metadata>,
}

impl GameEntry {
    /// The NACP title, or the file name.
    pub fn title(&self) -> &str {
        self.meta
            .as_ref()
            .map(|m| m.title.as_str())
            .filter(|t| !t.is_empty())
            .unwrap_or(&self.name)
    }
}

/// Every bootable file or ExeFS directory under `dirs`, sorted by name.
pub fn scan(dirs: &[PathBuf]) -> Vec<GameEntry> {
    let mut out = Vec::new();
    for dir in dirs {
        scan_dir(dir, 0, &mut out);
    }
    sort(&mut out);
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn sort(entries: &mut [GameEntry]) {
    entries.sort_by(|a, b| {
        a.title()
            .to_lowercase()
            .cmp(&b.title().to_lowercase())
            .then_with(|| a.path.cmp(&b.path))
    });
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
        meta: None,
    }
}

/// The NACP title language for the console language, as Lichii's game list picks it.
pub fn nacp_language(language: Language) -> NacpLanguage {
    match language {
        Language::Japanese => NacpLanguage::Japanese,
        Language::AmericanEnglish => NacpLanguage::AmericanEnglish,
        Language::French => NacpLanguage::French,
        Language::German => NacpLanguage::German,
        Language::Italian => NacpLanguage::Italian,
        Language::Spanish => NacpLanguage::Spanish,
        Language::Chinese | Language::SimplifiedChinese => NacpLanguage::SimplifiedChinese,
        Language::Korean => NacpLanguage::Korean,
        Language::Dutch => NacpLanguage::Dutch,
        Language::Portuguese => NacpLanguage::Portuguese,
        Language::Russian => NacpLanguage::Russian,
        Language::Taiwanese | Language::TraditionalChinese => NacpLanguage::TraditionalChinese,
        Language::BritishEnglish => NacpLanguage::BritishEnglish,
        Language::CanadianFrench => NacpLanguage::CanadianFrench,
        Language::LatinAmericanSpanish => NacpLanguage::LatinAmericanSpanish,
        Language::BrazilianPortuguese => NacpLanguage::BrazilianPortuguese,
    }
}

/// Where metadata is cached, and which keys it was read with.
pub struct MetadataCache {
    dir: PathBuf,
    /// A hash of what titles are read with besides their own file: the keys folder and the size
    /// and modification time of its key files (never their contents), and the names and sizes
    /// of the installed content. A title read before its keys were fixed can have a name but no
    /// control data, and an installed update changes its name and version, so either change
    /// reads every title again.
    keys: u64,
}

impl MetadataCache {
    pub fn new(dir: PathBuf, keys_dir: &Path, installed: &[PathBuf]) -> Self {
        let mut bytes = keys_dir.to_string_lossy().into_owned().into_bytes();
        for name in ["prod.keys", "console.keys", "title.keys"] {
            let (size, modified) = file_time(&keys_dir.join(name)).unwrap_or_default();
            bytes.extend(size.to_le_bytes());
            bytes.extend(modified.to_le_bytes());
        }
        for dir in installed {
            let mut entries: Vec<(String, u64)> = std::fs::read_dir(dir)
                .into_iter()
                .flatten()
                .filter_map(Result::ok)
                .map(|e| {
                    let size = e.metadata().map_or(0, |m| m.len());
                    (e.file_name().to_string_lossy().into_owned(), size)
                })
                .collect();
            entries.sort();
            for (name, size) in entries {
                bytes.extend(name.into_bytes());
                bytes.extend(size.to_le_bytes());
            }
        }
        Self {
            dir,
            keys: fnv1a(bytes),
        }
    }
}

/// A file's size and modification time in nanoseconds since the Unix epoch.
fn file_time(path: &Path) -> Option<(u64, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let modified = meta.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
    Some((meta.len(), u64::try_from(modified.as_nanos()).ok()?))
}

/// FNV-1a, stable across runs and Rust versions.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Reads a title's metadata and icon, from `cache` when neither the file nor the keys have
/// changed since. Titles that do not open (no keys yet, a damaged file) have none and are not
/// cached, so they are read again once the keys are set up.
pub fn read_metadata(
    entry: &GameEntry,
    options: &LoadOptions,
    cache: Option<&MetadataCache>,
) -> Option<(Metadata, Option<Icon>)> {
    let cache = cache
        .filter(|_| entry.kind != GameKind::ExeFs)
        .and_then(|cache| {
            Some((
                cache.dir.as_path(),
                Stamp::of(&entry.path, options.language, cache.keys)?,
            ))
        });
    if let Some(hit) = cache.as_ref().and_then(|(dir, stamp)| stamp.load(dir)) {
        return Some(hit);
    }
    let info = lichii_loader::scan(&entry.path, options)
        .inspect_err(|e| tracing::debug!(path = %entry.path.display(), "no title metadata: {e}"))
        .ok()?;
    let meta = Metadata {
        title: info.name,
        publisher: info.publisher,
        version: info.version,
        program_id: info.program_id,
        generation: info.generation,
    };
    let icon = info.icon.as_deref().and_then(Icon::decode);
    if let Some((dir, stamp)) = &cache {
        stamp.store(dir, &meta, icon.as_ref());
    }
    Some((meta, icon))
}

/// What identifies one version of a file, read in one language with one set of keys, in the
/// cache.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Stamp {
    path: PathBuf,
    size: u64,
    modified_ns: u64,
    language: u8,
    keys: String,
}

#[derive(Serialize, Deserialize)]
struct CacheRecord {
    stamp: Stamp,
    meta: Metadata,
}

impl Stamp {
    fn of(path: &Path, language: NacpLanguage, keys: u64) -> Option<Self> {
        let (size, modified_ns) = file_time(path)?;
        Some(Self {
            path: path.to_path_buf(),
            size,
            modified_ns,
            language: language as u8,
            keys: format!("{keys:016x}"),
        })
    }

    /// The cache file stem: a hash of the path.
    fn key(&self) -> String {
        format!("{:016x}", fnv1a(self.path.to_string_lossy().bytes()))
    }

    fn load(&self, dir: &Path) -> Option<(Metadata, Option<Icon>)> {
        let key = self.key();
        let text = std::fs::read_to_string(dir.join(format!("{key}.toml"))).ok()?;
        let record: CacheRecord = toml::from_str(&text).ok()?;
        if record.stamp != *self {
            return None;
        }
        let icon = std::fs::read(dir.join(format!("{key}.rgba")))
            .ok()
            .filter(|rgba| rgba.len() == ICON_SIZE * ICON_SIZE * 4)
            .map(Icon);
        Some((record.meta, icon))
    }

    fn store(&self, dir: &Path, meta: &Metadata, icon: Option<&Icon>) {
        let key = self.key();
        let record = CacheRecord {
            stamp: self.clone(),
            meta: meta.clone(),
        };
        let written = std::fs::create_dir_all(dir).and_then(|()| {
            let icon_path = dir.join(format!("{key}.rgba"));
            match icon {
                Some(icon) => std::fs::write(&icon_path, &icon.0)?,
                None => {
                    let _ = std::fs::remove_file(&icon_path);
                }
            }
            let text = toml::to_string(&record).map_err(std::io::Error::other)?;
            std::fs::write(dir.join(format!("{key}.toml")), text)
        });
        if let Err(e) = written {
            tracing::debug!(dir = %dir.display(), "game list cache not written: {e}");
        }
    }
}

/// What the scanning thread reports, in order.
enum ScanEvent {
    Listed(Vec<GameEntry>),
    Metadata(usize, Metadata, Option<Icon>),
    Done,
}

/// Where a scan reads titles from and caches what it learns.
pub struct ScanSources {
    pub dirs: Vec<PathBuf>,
    /// The folder with `prod.keys` and `title.keys`.
    pub keys_dir: PathBuf,
    pub cache_dir: PathBuf,
    /// The language titles are shown in, when the NACP has it.
    pub language: NacpLanguage,
    /// Registered content folders whose installed updates and DLC apply.
    pub installed: Vec<PathBuf>,
}

/// The list the UI shows, refreshed in the background.
#[derive(Default)]
pub struct GameList {
    pub entries: Vec<GameEntry>,
    pub filter: String,
    /// Icon textures by title path, uploaded as their metadata arrives.
    pub icons: HashMap<PathBuf, egui::TextureHandle>,
    new_icons: Vec<(PathBuf, Icon)>,
    pending: Option<Receiver<ScanEvent>>,
}

impl GameList {
    /// Starts a rescan; `wake` runs on the scanning thread each time it has news.
    pub fn refresh(&mut self, sources: ScanSources, wake: impl Fn() + Send + 'static) {
        let (tx, rx) = crossbeam_channel::unbounded();
        std::thread::Builder::new()
            .name("game list scan".into())
            .spawn(move || {
                let entries = scan(&sources.dirs);
                let listed = entries.clone();
                if tx.send(ScanEvent::Listed(listed)).is_err() {
                    return;
                }
                wake();
                let cache =
                    MetadataCache::new(sources.cache_dir, &sources.keys_dir, &sources.installed);
                let options = LoadOptions {
                    keys: Some(crate::setup::KeyStatus::check(sources.keys_dir).keys),
                    language: sources.language,
                    installed: sources.installed,
                    ..LoadOptions::default()
                };
                for (index, entry) in entries.iter().enumerate() {
                    let Some((meta, icon)) = read_metadata(entry, &options, Some(&cache)) else {
                        continue;
                    };
                    if tx.send(ScanEvent::Metadata(index, meta, icon)).is_err() {
                        return;
                    }
                    wake();
                }
                let _ = tx.send(ScanEvent::Done);
                wake();
            })
            .expect("spawning the game list scanner");
        self.pending = Some(rx);
    }

    pub fn is_scanning(&self) -> bool {
        self.pending.is_some()
    }

    /// Takes what the scan found so far. The list is sorted by title once the scan finishes.
    pub fn poll(&mut self) {
        let Some(rx) = &self.pending else {
            return;
        };
        let mut finished = false;
        loop {
            match rx.try_recv() {
                Ok(ScanEvent::Listed(entries)) => {
                    self.entries = entries;
                    self.icons.clear();
                    self.new_icons.clear();
                }
                Ok(ScanEvent::Metadata(index, meta, icon)) => {
                    let Some(entry) = self.entries.get_mut(index) else {
                        continue;
                    };
                    entry.meta = Some(meta);
                    if let Some(icon) = icon {
                        self.new_icons.push((entry.path.clone(), icon));
                    }
                }
                Ok(ScanEvent::Done) | Err(TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
                Err(TryRecvError::Empty) => break,
            }
        }
        if finished {
            sort(&mut self.entries);
            self.pending = None;
        }
    }

    /// Uploads the icons that arrived since the last frame.
    pub fn upload_icons(&mut self, ctx: &egui::Context) {
        for (path, icon) in self.new_icons.drain(..) {
            let image = egui::ColorImage::from_rgba_unmultiplied([ICON_SIZE, ICON_SIZE], &icon.0);
            let texture = ctx.load_texture(
                path.display().to_string(),
                image,
                egui::TextureOptions::LINEAR,
            );
            self.icons.insert(path, texture);
        }
    }

    /// Entries whose title, publisher, program id or path contains the filter,
    /// case-insensitively.
    pub fn visible(&self) -> impl Iterator<Item = &GameEntry> {
        let filter = self.filter.trim().to_lowercase();
        self.entries.iter().filter(move |e| {
            let meta = e.meta.as_ref();
            filter.is_empty()
                || e.title().to_lowercase().contains(&filter)
                || e.path.to_string_lossy().to_lowercase().contains(&filter)
                || meta.is_some_and(|m| m.publisher.to_lowercase().contains(&filter))
                || meta.is_some_and(|m| {
                    m.program_id != 0 && format!("{:016x}", m.program_id).contains(&filter)
                })
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
            meta: None,
        };
        let mut list = GameList {
            entries: vec![
                entry("Kart", "/g/racing/kart.nsp"),
                entry("Zelda", "/g/zelda.nsp"),
            ],
            filter: "RACING".into(),
            ..GameList::default()
        };
        let names: Vec<_> = list.visible().map(|e| e.name.as_str()).collect();
        assert_eq!(names, ["Kart"]);

        list.entries[1].meta = Some(Metadata {
            title: "Breath".into(),
            publisher: "Big N".into(),
            version: "1.0.0".into(),
            program_id: 0x0100_0000_0001_0000,
            generation: ConsoleGeneration::Switch1,
        });
        for filter in ["big n", "breath", "0100000000010000"] {
            list.filter = filter.into();
            let titles: Vec<_> = list.visible().map(GameEntry::title).collect();
            assert_eq!(titles, ["Breath"], "{filter}");
        }
    }

    fn nacp(name: &str, publisher: &str, version: &str) -> Vec<u8> {
        let mut nacp = vec![0u8; 0x4000];
        nacp[..name.len()].copy_from_slice(name.as_bytes());
        nacp[0x200..0x200 + publisher.len()].copy_from_slice(publisher.as_bytes());
        nacp[0x3060..0x3060 + version.len()].copy_from_slice(version.as_bytes());
        nacp
    }

    fn png_icon() -> Vec<u8> {
        let image = image::RgbaImage::from_pixel(8, 8, image::Rgba([200, 30, 30, 255]));
        let mut png = std::io::Cursor::new(Vec::new());
        image.write_to(&mut png, image::ImageFormat::Png).unwrap();
        png.into_inner()
    }

    /// A minimal NRO with an ASET section holding `assets` (icon, NACP, RomFS).
    fn nro(assets: [&[u8]; 3]) -> Vec<u8> {
        let mut image = vec![0u8; 0x3000];
        image[0x10..0x14].copy_from_slice(b"NRO0");
        image[0x18..0x1c].copy_from_slice(&0x3000u32.to_le_bytes());
        for (i, (offset, size)) in [(0u32, 0x1000u32), (0x1000, 0x1000), (0x2000, 0x800)]
            .into_iter()
            .enumerate()
        {
            image[0x20 + i * 8..0x24 + i * 8].copy_from_slice(&offset.to_le_bytes());
            image[0x24 + i * 8..0x28 + i * 8].copy_from_slice(&size.to_le_bytes());
        }
        image[0x38..0x3c].copy_from_slice(&0x1800u32.to_le_bytes());
        let mut header = b"ASET".to_vec();
        header.extend([0; 4]);
        let mut payload = Vec::new();
        for asset in assets {
            header.extend((0x38 + payload.len() as u64).to_le_bytes());
            header.extend((asset.len() as u64).to_le_bytes());
            payload.extend_from_slice(asset);
        }
        image.extend(header);
        image.extend(payload);
        image
    }

    #[test]
    fn metadata_is_read_from_the_title_and_cached() {
        let tmp = TempDir::new("meta");
        let path = tmp.0.join("hb/app.nro");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let control = nacp("Synthetic Game", "Lichii Tests", "1.2.3");
        std::fs::write(&path, nro([&png_icon(), &control, b""])).unwrap();
        let keys = tmp.0.join("keys");
        let cache = MetadataCache::new(tmp.0.join("cache"), &keys, &[]);
        let game = scan(std::slice::from_ref(&tmp.0)).remove(0);

        let (meta, icon) = read_metadata(&game, &LoadOptions::default(), Some(&cache)).unwrap();
        assert_eq!(
            meta,
            Metadata {
                title: "Synthetic Game".into(),
                publisher: "Lichii Tests".into(),
                version: "1.2.3".into(),
                program_id: 0,
                generation: ConsoleGeneration::Switch1,
            }
        );
        let icon = icon.expect("the icon decodes");
        assert_eq!(icon.0.len(), ICON_SIZE * ICON_SIZE * 4);
        assert_eq!(&icon.0[..4], [200, 30, 30, 255]);

        let record = std::fs::read_dir(&cache.dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| p.extension().is_some_and(|e| e == "toml"))
            .unwrap();
        let text = std::fs::read_to_string(&record).unwrap();
        std::fs::write(&record, text.replace("Synthetic Game", "From Cache")).unwrap();
        let (cached, icon) = read_metadata(&game, &LoadOptions::default(), Some(&cache)).unwrap();
        assert_eq!(cached.title, "From Cache");
        assert!(icon.is_some(), "the cached icon is used");
        std::fs::create_dir_all(&keys).unwrap();
        std::fs::write(keys.join("prod.keys"), "").unwrap();
        let new_keys = MetadataCache::new(tmp.0.join("cache"), &keys, &[]);
        let (rekeyed, _) = read_metadata(&game, &LoadOptions::default(), Some(&new_keys)).unwrap();
        assert_eq!(
            rekeyed.title, "Synthetic Game",
            "new keys read the title again"
        );
        let british = LoadOptions {
            language: NacpLanguage::BritishEnglish,
            ..LoadOptions::default()
        };
        let (other, _) = read_metadata(&game, &british, Some(&cache)).unwrap();
        assert_eq!(
            other.title, "Synthetic Game",
            "another language is read again"
        );

        let control = nacp("Synthetic Game II", "Lichii Tests", "2.0.0");
        std::fs::write(&path, nro([b"", &control, b"pad"])).unwrap();
        let (fresh, icon) = read_metadata(&game, &LoadOptions::default(), Some(&cache)).unwrap();
        assert_eq!(
            (fresh.title.as_str(), fresh.version.as_str()),
            ("Synthetic Game II", "2.0.0")
        );
        assert!(icon.is_none());
    }

    #[test]
    fn titles_that_do_not_open_are_not_cached() {
        let tmp = TempDir::new("locked");
        tmp.file("Locked.nsp", 0x400);
        let cache = MetadataCache::new(tmp.0.join("cache"), &tmp.0.join("keys"), &[]);
        let game = scan(std::slice::from_ref(&tmp.0)).remove(0);
        assert!(read_metadata(&game, &LoadOptions::default(), Some(&cache)).is_none());
        assert!(!cache.dir.exists());
        assert_eq!(game.title(), "Locked");
    }

    #[test]
    fn a_scan_lists_files_then_fills_in_metadata_sorted_by_title() {
        let tmp = TempDir::new("refresh");
        let games = tmp.0.join("games");
        std::fs::create_dir_all(&games).unwrap();
        std::fs::write(games.join("a.nro"), nro([b"", &nacp("Zeta", "", ""), b""])).unwrap();
        std::fs::write(games.join("b.nro"), nro([b"", &nacp("Alpha", "", ""), b""])).unwrap();
        let mut list = GameList::default();
        let (tx, rx) = crossbeam_channel::unbounded();
        list.refresh(
            ScanSources {
                dirs: vec![games],
                keys_dir: tmp.0.join("keys"),
                cache_dir: tmp.0.join("cache"),
                language: NacpLanguage::AmericanEnglish,
                installed: Vec::new(),
            },
            move || {
                let _ = tx.send(());
            },
        );
        while list.is_scanning() {
            rx.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
            list.poll();
        }
        let titles: Vec<_> = list.entries.iter().map(GameEntry::title).collect();
        assert_eq!(titles, ["Alpha", "Zeta"]);
    }

    #[test]
    fn sizes() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(3 * 1024 * 1024 * 1024 / 2), "1.5 GiB");
    }
}
