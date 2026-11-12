//! Finding titles on disk, reading what the game list shows, and loading them into a
//! [`ProgramImage`].

use std::io::Read;
use std::path::{Path, PathBuf};

use lichii_common::ConsoleGeneration;
use lichii_fs::host::{HostDir, HostFile};
use lichii_fs::nacp::{Nacp, NacpLanguage};
use lichii_fs::romfs::build_romfs;
use lichii_fs::{DirRef, Directory, DirectoryExt, EntryKind, FileExt, FileRef};

use crate::error::{LoaderError, LoaderResult};
use crate::exefs::load_exefs;
use crate::npdm::Npdm;
use crate::nro::load_nro;
use crate::nso::{HashCheck, load_nso};
use crate::program::ProgramImage;

const ICON_LANGUAGES: [&str; 18] = [
    "AmericanEnglish",
    "BritishEnglish",
    "Japanese",
    "French",
    "German",
    "LatinAmericanSpanish",
    "Spanish",
    "Italian",
    "Dutch",
    "CanadianFrench",
    "Portuguese",
    "Russian",
    "Korean",
    "TraditionalChinese",
    "SimplifiedChinese",
    "BrazilianPortuguese",
    "Polish",
    "Thai",
];

/// How a title is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TitleFormat {
    /// An extracted ExeFS directory (`main`, `main.npdm`, ...), usually with a sibling `romfs/`.
    LooseExefs,
    Nso,
    Nro,
}

/// What the game list shows for a title.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TitleInfo {
    pub path: PathBuf,
    pub format: TitleFormat,
    pub program_id: u64,
    pub generation: ConsoleGeneration,
    pub name: String,
    pub publisher: String,
    pub version: String,
    /// The JPEG (or PNG/BMP sidecar) icon, when there is one.
    pub icon: Option<Vec<u8>>,
}

/// Options that change how a title is loaded.
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub hashes: HashCheck,
    pub language: NacpLanguage,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            hashes: HashCheck::Skip,
            language: NacpLanguage::AmericanEnglish,
        }
    }
}

/// An extracted title: its ExeFS directory and the directory that holds it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LooseTitle {
    pub exefs: PathBuf,
}

impl LooseTitle {
    /// Accepts the ExeFS's `main`, the ExeFS directory itself, or a title directory with an
    /// `exefs/` inside.
    pub fn locate(path: &Path) -> Option<Self> {
        let candidates = [
            path.parent()
                .filter(|_| path.is_file())
                .map(Path::to_path_buf),
            Some(path.to_path_buf()),
            Some(path.join("exefs")),
        ];
        candidates
            .into_iter()
            .flatten()
            .find(|dir| dir.join("main.npdm").is_file() && dir.join("main").is_file())
            .map(|exefs| Self { exefs })
    }

    /// Directories that may carry `control.nacp` and icon sidecars: the ExeFS and the folders
    /// above it up to the game folder (`<game>/exefs`, `<game>/base/exefs`,
    /// `<game>/updates/<version>/exefs`).
    fn sidecar_dirs(&self) -> Vec<PathBuf> {
        let mut dirs = vec![self.exefs.clone()];
        let mut current = self.exefs.as_path();
        while dirs.len() < 4 {
            let Some(parent) = current.parent() else {
                break;
            };
            dirs.push(parent.to_path_buf());
            let name = parent
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let storage_level = matches!(name.as_str(), "base" | "update" | "updates")
                || (!name.is_empty() && name.bytes().all(|b| b.is_ascii_hexdigit()));
            if !storage_level {
                break;
            }
            current = parent;
        }
        dirs
    }

    /// The folder name the game list shows when there is no NACP.
    fn folder_name(&self) -> String {
        let dirs = self.sidecar_dirs();
        let named = if dirs.len() > 1 { dirs.last() } else { None };
        named
            .and_then(|d| d.file_name())
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    fn nacp(&self) -> Option<Nacp> {
        self.sidecar_dirs().iter().find_map(|dir| {
            let file = std::fs::read_dir(dir)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .find(|p| {
                    p.is_file()
                        && (p.file_name().is_some_and(|n| n == "control.nacp")
                            || p.extension()
                                .is_some_and(|e| e.eq_ignore_ascii_case("nacp")))
                })?;
            Nacp::read(HostFile::open(&file).ok()?.as_ref()).ok()
        })
    }

    fn icon(&self) -> Option<Vec<u8>> {
        let dirs = self.sidecar_dirs();
        let named = dirs.iter().find_map(|dir| {
            ICON_LANGUAGES
                .iter()
                .find_map(|lang| std::fs::read(dir.join(format!("icon_{lang}.dat"))).ok())
        });
        named.or_else(|| {
            dirs.iter().find_map(|dir| {
                let image = std::fs::read_dir(dir)
                    .ok()?
                    .flatten()
                    .map(|e| e.path())
                    .find(|p| {
                        p.extension().and_then(|e| e.to_str()).is_some_and(|e| {
                            ["png", "jpg", "jpeg", "bmp"].contains(&e.to_ascii_lowercase().as_str())
                        })
                    })?;
                std::fs::read(image).ok()
            })
        })
    }

    /// The RomFS directory the title uses: a `romfs/` next to the ExeFS, else (for an update
    /// ExeFS without one) the base's, else (for a base dump whose data all lives in an update)
    /// the newest update's when that update's `main` is the same build.
    fn romfs_dir(&self) -> Option<PathBuf> {
        let title = self.exefs.parent()?;
        if let Some(dir) = find_romfs_dir(title) {
            return Some(dir);
        }
        let parent = title.parent()?;
        if parent
            .file_name()
            .is_some_and(|n| n.eq_ignore_ascii_case("updates"))
        {
            let game = parent.parent()?;
            return find_romfs_dir(game).or_else(|| find_romfs_dir(&game.join("base")));
        }
        let main_build_id = |exefs: &Path| -> Option<[u8; 32]> {
            let mut header = [0u8; 0x60];
            std::fs::File::open(exefs.join("main"))
                .ok()?
                .read_exact(&mut header)
                .ok()?;
            (&header[..4] == b"NSO0").then(|| header[0x40..].try_into().unwrap())
        };
        [title.join("updates"), parent.join("updates")].iter().find_map(|updates| {
            let newest = std::fs::read_dir(updates)
                .ok()?
                .flatten()
                .filter_map(|e| Some((e.file_name().to_str()?.parse::<u64>().ok()?, e.path())))
                .max_by_key(|(version, _)| *version)?
                .1;
            let romfs = find_romfs_dir(&newest)?;
            match (main_build_id(&self.exefs), main_build_id(&newest.join("exefs"))) {
                (Some(base), Some(update)) if base == update => Some(romfs),
                _ => {
                    tracing::error!(
                        "{}: the base has no RomFS and the newest update's main is not the same build; load the update's exefs",
                        self.exefs.display()
                    );
                    None
                }
            }
        })
    }

    fn romfs(&self) -> LoaderResult<Option<FileRef>> {
        let in_exefs = std::fs::read_dir(&self.exefs)?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.is_file() && p.extension().is_some_and(|e| e == "romfs"));
        if let Some(file) = in_exefs {
            return Ok(Some(HostFile::open(&file)?));
        }
        match self.romfs_dir() {
            Some(dir) => Ok(Some(build_romfs(HostDir::open(dir)?.as_ref())?)),
            None => Ok(None),
        }
    }

    /// Reads the game-list entry without decompressing any module.
    pub fn info(&self) -> LoaderResult<TitleInfo> {
        let npdm = Npdm::parse(&std::fs::read(self.exefs.join("main.npdm"))?)?;
        let nacp = self.nacp();
        let title = nacp
            .as_ref()
            .and_then(|n| n.preferred_title(NacpLanguage::AmericanEnglish));
        let version = nacp
            .as_ref()
            .map(Nacp::display_version)
            .filter(|v| !v.is_empty())
            .or_else(|| {
                self.romfs_dir()
                    .and_then(|d| HostDir::open(d).ok())
                    .and_then(|d| romfs_product_version(d.as_ref()))
            })
            .unwrap_or_default();
        Ok(TitleInfo {
            path: self.exefs.clone(),
            format: TitleFormat::LooseExefs,
            program_id: npdm.program_id,
            generation: ConsoleGeneration::from_program_id(npdm.program_id),
            name: title
                .map(|t| t.name.clone())
                .unwrap_or_else(|| self.folder_name()),
            publisher: title.map(|t| t.publisher.clone()).unwrap_or_default(),
            version,
            icon: self.icon(),
        })
    }

    /// Lays out every module and builds the RomFS view the title will mount.
    pub fn load(&self, options: &LoadOptions) -> LoaderResult<ProgramImage> {
        let exefs = HostDir::open(&self.exefs)?;
        let (npdm, modules) = load_exefs(exefs.as_ref(), options.hashes)?;
        Ok(ProgramImage::new(npdm, modules, self.romfs()?, self.nacp()))
    }
}

fn find_romfs_dir(parent: &Path) -> Option<PathBuf> {
    let exact = parent.join("romfs");
    if exact.is_dir() {
        return Some(exact);
    }
    std::fs::read_dir(parent)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| {
            p.is_dir()
                && p.file_name()
                    .is_some_and(|n| n.eq_ignore_ascii_case("romfs"))
        })
}

fn newest_product_tag(dir: Option<DirRef>, prefix: &str) -> Option<String> {
    let entries = dir?.entries().ok()?;
    entries
        .iter()
        .filter(|e| e.kind == EntryKind::File)
        .filter_map(|e| {
            let digits = e.name.strip_prefix(prefix)?.get(..4)?;
            let (digits, dot) = digits.split_at(3);
            (dot == "." && digits.bytes().all(|b| b.is_ascii_digit())).then(|| digits.to_owned())
        })
        .max()
}

/// The display version of a dump without control data, from the resource tables Nintendo
/// titles name after it (`System/Resource/ResourceSizeTable.Product.180.*` is 1.8.0).
pub fn romfs_product_version(romfs: &dyn Directory) -> Option<String> {
    let tables = newest_product_tag(
        romfs.open_dir_path("System/Resource").ok().flatten(),
        "ResourceSizeTable.Product.",
    );
    let game_data = newest_product_tag(
        romfs.open_dir("GameData").ok().flatten(),
        "GameDataList.Product.",
    );
    let best = tables.into_iter().chain(game_data).max()?;
    let digits: Vec<char> = best.chars().collect();
    Some(format!("{}.{}.{}", digits[0], digits[1], digits[2]))
}

/// Recognises a title at `path`.
pub fn identify(path: &Path) -> Option<TitleFormat> {
    if LooseTitle::locate(path).is_some() {
        return Some(TitleFormat::LooseExefs);
    }
    let magic = std::fs::File::open(path)
        .ok()
        .and_then(|f| HostFile::open(path).ok().map(|h| (f, h)))?;
    let head = magic.1.read_array::<0x14>(0).ok()?;
    match (&head[..4], &head[0x10..0x14]) {
        (b"NSO0", _) => Some(TitleFormat::Nso),
        (_, b"NRO0") => Some(TitleFormat::Nro),
        _ => None,
    }
}

/// Reads the game-list entry of the title at `path`.
pub fn scan(path: &Path) -> LoaderResult<TitleInfo> {
    let not_a_title = || LoaderError::NotATitle(path.display().to_string());
    match identify(path).ok_or_else(not_a_title)? {
        TitleFormat::LooseExefs => LooseTitle::locate(path).ok_or_else(not_a_title)?.info(),
        format @ (TitleFormat::Nso | TitleFormat::Nro) => {
            let (nacp, icon) = if format == TitleFormat::Nro {
                let (_, assets) = load_nro(&HostFile::open(path)?, "nro")?;
                let nacp = assets.nacp.and_then(|f| Nacp::read(f.as_ref()).ok());
                (nacp, assets.icon.and_then(|f| f.read_all().ok()))
            } else {
                (None, None)
            };
            let title = nacp
                .as_ref()
                .and_then(|n| n.preferred_title(NacpLanguage::AmericanEnglish));
            let stem = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            Ok(TitleInfo {
                path: path.to_path_buf(),
                format,
                program_id: 0,
                generation: ConsoleGeneration::Switch1,
                name: title.map_or(stem, |t| t.name.clone()),
                publisher: title.map(|t| t.publisher.clone()).unwrap_or_default(),
                version: nacp.as_ref().map(Nacp::display_version).unwrap_or_default(),
                icon,
            })
        }
    }
}

/// Loads the title at `path` into a [`ProgramImage`].
pub fn load(path: &Path, options: &LoadOptions) -> LoaderResult<ProgramImage> {
    let not_a_title = || LoaderError::NotATitle(path.display().to_string());
    match identify(path).ok_or_else(not_a_title)? {
        TitleFormat::LooseExefs => LooseTitle::locate(path)
            .ok_or_else(not_a_title)?
            .load(options),
        TitleFormat::Nso => {
            let module = load_nso(HostFile::open(path)?.as_ref(), "main", options.hashes)?;
            Ok(ProgramImage::new(
                Npdm::homebrew(0),
                vec![module],
                None,
                None,
            ))
        }
        TitleFormat::Nro => {
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let (module, assets) = load_nro(&HostFile::open(path)?, &name)?;
            let nacp = assets.nacp.and_then(|f| Nacp::read(f.as_ref()).ok());
            Ok(ProgramImage::new(
                Npdm::homebrew(0),
                vec![module],
                assets.romfs,
                nacp,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npdm::AcidLayout;
    use crate::npdm::tests::{NpdmSpec, build_npdm};
    use crate::nso::tests::{Compression, build_nso, segments};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("lichii-loader-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn write_exefs(dir: &Path, program_id: u64, main_seed: u8) {
        std::fs::create_dir_all(dir).unwrap();
        let spec = NpdmSpec {
            program_id,
            layout: AcidLayout::Switch2,
            ..NpdmSpec::default()
        };
        std::fs::write(dir.join("main.npdm"), build_npdm(&spec)).unwrap();
        std::fs::write(
            dir.join("rtld"),
            build_nso(&segments(9), Compression::Lz4, 0, true),
        )
        .unwrap();
        std::fs::write(
            dir.join("main"),
            build_nso(&segments(main_seed), Compression::Lz4, 0x8000, true),
        )
        .unwrap();
        std::fs::write(
            dir.join("sdk"),
            build_nso(&segments(7), Compression::Lz4, 0x2000, true),
        )
        .unwrap();
    }

    #[test]
    fn loose_switch2_exefs_boots_with_sibling_romfs() {
        let temp = TempDir::new("loose");
        let game = temp.0.join("first switch game");
        write_exefs(&game.join("exefs"), 0x0400_c3f0_0006_e000, 1);
        std::fs::create_dir_all(game.join("romfs/System/Resource")).unwrap();
        std::fs::write(
            game.join("romfs/System/Resource/ResourceSizeTable.Product.180.rsizetable.zs"),
            b"t",
        )
        .unwrap();
        std::fs::write(game.join("romfs/data.bin"), [5u8; 33]).unwrap();

        let main = game.join("exefs/main");
        assert_eq!(identify(&main), Some(TitleFormat::LooseExefs));
        assert_eq!(identify(&game), Some(TitleFormat::LooseExefs));
        let info = scan(&main).unwrap();
        assert_eq!(info.program_id, 0x0400_c3f0_0006_e000);
        assert_eq!(info.generation, ConsoleGeneration::Switch2);
        assert_eq!(info.name, "first switch game");
        assert_eq!(info.version, "1.8.0");
        assert!(info.icon.is_none());

        let image = load(
            &main,
            &LoadOptions {
                hashes: HashCheck::Verify,
                ..LoadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(image.generation, ConsoleGeneration::Switch2);
        let names: Vec<_> = image.modules.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, ["rtld", "main", "sdk"]);
        assert_eq!(image.modules[0].offset, 0);
        assert_eq!(image.modules[1].offset, image.modules[0].size);
        assert_eq!(
            image.modules[2].offset,
            image.modules[1].offset + image.modules[1].size
        );
        assert_eq!(
            image.code_size,
            image.modules.iter().map(|m| m.size).sum::<u64>()
        );
        assert_eq!(image.entry_offset, 0);
        assert_eq!(image.main_thread.stack_size, 0x10_0000);
        assert_eq!(image.capabilities.thread_info.unwrap().core_mask(), 0x3f);
        assert_eq!(image.pseudo_device_seed, 0x0400_c3f0_0006_e000);
        let romfs = lichii_fs::romfs::open_romfs(image.romfs.as_ref().unwrap()).unwrap();
        assert_eq!(
            romfs
                .open_file("data.bin")
                .unwrap()
                .unwrap()
                .read_all()
                .unwrap(),
            [5u8; 33]
        );
    }

    #[test]
    fn sidecars_and_update_layouts() {
        let temp = TempDir::new("updates");
        let game = temp.0.join("Game");
        write_exefs(&game.join("base/exefs"), 0x0100_0000_0000_0000, 1);
        write_exefs(&game.join("updates/65536/exefs"), 0x0100_0000_0000_0000, 1);
        write_exefs(&game.join("updates/131072/exefs"), 0x0100_0000_0000_0000, 1);
        std::fs::create_dir_all(game.join("updates/131072/romfs")).unwrap();
        std::fs::write(game.join("updates/131072/romfs/new.bin"), b"v2").unwrap();
        let mut nacp = vec![0u8; lichii_fs::nacp::NACP_SIZE];
        nacp[..10].copy_from_slice(b"Named Game");
        nacp[0x200..0x203].copy_from_slice(b"Pub");
        nacp[0x3060..0x3065].copy_from_slice(b"2.0.0");
        nacp[0x30f8..0x3100].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        std::fs::write(game.join("control.nacp"), nacp).unwrap();
        std::fs::write(game.join("icon_AmericanEnglish.dat"), b"jpeg").unwrap();

        let base = LooseTitle::locate(&game.join("base")).unwrap();
        let info = base.info().unwrap();
        assert_eq!(
            (
                info.name.as_str(),
                info.publisher.as_str(),
                info.version.as_str()
            ),
            ("Named Game", "Pub", "2.0.0")
        );
        assert_eq!(info.icon.as_deref(), Some(&b"jpeg"[..]));
        assert_eq!(info.generation, ConsoleGeneration::Switch1);

        let image = base.load(&LoadOptions::default()).unwrap();
        let romfs = lichii_fs::romfs::open_romfs(image.romfs.as_ref().unwrap()).unwrap();
        assert!(romfs.open_file("new.bin").unwrap().is_some());
        assert_eq!(image.pseudo_device_seed, 0x1234_5678);

        let update = LooseTitle::locate(&game.join("updates/65536/exefs/main")).unwrap();
        assert!(
            update
                .load(&LoadOptions::default())
                .unwrap()
                .romfs
                .is_none()
        );

        let mut other_build = std::fs::read(game.join("base/exefs/main")).unwrap();
        other_build[0x40] ^= 1;
        std::fs::write(game.join("updates/131072/exefs/main"), other_build).unwrap();
        assert!(base.load(&LoadOptions::default()).unwrap().romfs.is_none());

        std::fs::remove_file(game.join("updates/131072/exefs/main")).unwrap();
        assert!(base.load(&LoadOptions::default()).unwrap().romfs.is_none());
    }

    #[test]
    fn standalone_nso_and_nro() {
        let temp = TempDir::new("single");
        let nso = temp.0.join("app.nso");
        std::fs::write(&nso, build_nso(&segments(3), Compression::Lz4, 0, false)).unwrap();
        assert_eq!(identify(&nso), Some(TitleFormat::Nso));
        let image = load(&nso, &LoadOptions::default()).unwrap();
        assert_eq!(image.modules.len(), 1);
        assert!(image.capabilities.allows_svc(0x7f));
        assert_eq!(scan(&nso).unwrap().name, "app");

        let nro = temp.0.join("hello.nro");
        std::fs::write(&nro, crate::nro::tests::build_nro(&[b"icon", b"", b""])).unwrap();
        assert_eq!(identify(&nro), Some(TitleFormat::Nro));
        let info = scan(&nro).unwrap();
        assert_eq!(
            (info.name.as_str(), info.icon.as_deref()),
            ("hello", Some(&b"icon"[..]))
        );
        assert_eq!(
            load(&nro, &LoadOptions::default()).unwrap().modules[0].name,
            "hello.nro"
        );

        let junk = temp.0.join("junk.bin");
        std::fs::write(&junk, [0u8; 64]).unwrap();
        assert_eq!(identify(&junk), None);
        assert!(matches!(scan(&junk), Err(LoaderError::NotATitle(_))));
    }
}
