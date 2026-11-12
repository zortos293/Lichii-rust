//! Mods for one title: ExeFS file replacement and IPS/IPSwitch patches matched by build id,
//! and LayeredFS RomFS replacement with `romfs_ext` stubs and patches.
//!
//! Mods live in `<mods>/<title id>/<mod name>/` with `exefs/`, `romfs/`, `romfslite/` and
//! `romfs_ext/` inside, and in Atmosphere's `<sdmc>/atmosphere/contents/<title id>/`, which
//! the add-on list calls `SDMC`. Mods apply in name order; the first one wins a conflict.

use std::path::PathBuf;

use lichii_fs::host::HostDir;
use lichii_fs::ips::{IpSwitch, Ips, parse_build_id};
use lichii_fs::romfs::{RomfsExt, build_romfs, open_romfs};
use lichii_fs::{DirRef, Directory, DirectoryExt, EntryKind, FileExt, FileRef, LayeredDir};

use crate::error::LoaderResult;
use crate::program::Module;

const NSO_HEADER_SIZE: usize = 0x100;

/// Where mods are installed and which ones the user turned off.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModOptions {
    /// `<mods>`: one directory per title id, one directory per mod inside.
    pub load_root: Option<PathBuf>,
    /// The emulated SD card; its `atmosphere/contents/<title id>/` is the `SDMC` mod.
    pub sdmc_root: Option<PathBuf>,
    /// Mod directory names to skip, `SDMC` included.
    pub disabled: Vec<String>,
}

enum ExefsPatch {
    Ips(Ips),
    IpSwitch(IpSwitch),
}

struct ModDir {
    name: String,
    dir: DirRef,
}

/// The enabled mods for one title, in the order they apply.
pub struct Mods {
    mods: Vec<ModDir>,
}

impl Mods {
    /// Finds the title's mods. Missing roots and title directories mean no mods.
    pub fn find(options: &ModOptions, program_id: u64) -> LoaderResult<Self> {
        let title = format!("{program_id:016X}");
        let mut found: Vec<(String, ModDir)> = Vec::new();
        if let Some(root) = options.load_root.as_deref().filter(|p| p.is_dir())
            && let Some(title_dir) = find_caseless(HostDir::open(root)?.as_ref(), &title)?
        {
            for entry in title_dir.entries()? {
                if entry.kind != EntryKind::Directory {
                    continue;
                }
                if let Some(dir) = title_dir.open_dir(&entry.name)? {
                    found.push((
                        entry.name.clone(),
                        ModDir {
                            name: entry.name,
                            dir,
                        },
                    ));
                }
            }
        }
        if let Some(sdmc) = options.sdmc_root.as_deref() {
            let contents = sdmc.join("atmosphere").join("contents");
            if contents.is_dir()
                && let Some(dir) = find_caseless(HostDir::open(&contents)?.as_ref(), &title)?
            {
                found.push((
                    title.clone(),
                    ModDir {
                        name: "SDMC".to_owned(),
                        dir,
                    },
                ));
            }
        }
        found.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Self {
            mods: found
                .into_iter()
                .map(|(_, m)| m)
                .filter(|m| !options.disabled.contains(&m.name))
                .collect(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.mods.is_empty()
    }

    fn subdirs(&self, kind: &str) -> LoaderResult<Vec<DirRef>> {
        let mut out = Vec::new();
        for m in &self.mods {
            if let Some(dir) = find_caseless(m.dir.as_ref(), kind)? {
                tracing::info!("mod {} ({kind})", m.name);
                out.push(dir);
            }
        }
        Ok(out)
    }

    /// LayeredExeFS: files in a mod's `exefs/` replace the title's.
    pub fn layer_exefs(&self, exefs: DirRef) -> LoaderResult<DirRef> {
        let mut layers = self.subdirs("exefs")?;
        if layers.is_empty() {
            return Ok(exefs);
        }
        layers.push(exefs);
        Ok(LayeredDir::from_layers(layers))
    }

    /// Applies the `.ips` files named after a module's build id and the `.pchtxt` files whose
    /// `@nsobid` matches it, from every mod's `exefs/`. Offsets count the 0x100-byte NSO
    /// header, as mods are written against the NSO file. A patch that fails to parse or apply
    /// is logged and skipped.
    pub fn patch_modules(&self, modules: &mut [Module]) -> LoaderResult<()> {
        let mut patches: Vec<(String, [u8; 32], ExefsPatch)> = Vec::new();
        for exefs in self.subdirs("exefs")? {
            let mut entries = exefs.entries()?;
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            for entry in entries {
                let lower = entry.name.to_ascii_lowercase();
                let is_ips = lower.ends_with(".ips");
                if !is_ips && !lower.ends_with(".pchtxt") {
                    continue;
                }
                let Some(file) = exefs.open_file(&entry.name)? else {
                    continue;
                };
                let data = file.read_all()?;
                if !is_ips {
                    let patch = IpSwitch::parse(&data);
                    patches.push((entry.name, *patch.build_id(), ExefsPatch::IpSwitch(patch)));
                    continue;
                }
                let Some(build_id) = parse_build_id(lower.split('.').next().unwrap_or_default())
                else {
                    continue;
                };
                match Ips::parse(&data) {
                    Ok(patch) => patches.push((entry.name, build_id, ExefsPatch::Ips(patch))),
                    Err(error) => tracing::warn!("{}: {error}", entry.name),
                }
            }
        }
        for module in modules {
            let mut target: Option<Vec<u8>> = None;
            for (name, _, patch) in patches.iter().filter(|p| p.1 == module.build_id) {
                tracing::info!("{}: applying {name}", module.name);
                let target = target.get_or_insert_with(|| {
                    let mut t = vec![0u8; NSO_HEADER_SIZE + module.image.len()];
                    t[NSO_HEADER_SIZE..].copy_from_slice(&module.image);
                    t
                });
                match patch {
                    ExefsPatch::Ips(ips) => {
                        if let Err(error) = ips.apply(target) {
                            tracing::warn!("{name}: {error}");
                        }
                    }
                    ExefsPatch::IpSwitch(patch) => patch.apply(target),
                }
            }
            if let Some(target) = target {
                module.image.copy_from_slice(&target[NSO_HEADER_SIZE..]);
            }
        }
        Ok(())
    }

    /// LayeredFS: rebuilds the RomFS with every mod's `romfs/` and `romfslite/` over the
    /// base and `romfs_ext/` applied. Without a base RomFS or any RomFS mod, returns the base.
    pub fn layer_romfs(&self, romfs: Option<FileRef>) -> LoaderResult<Option<FileRef>> {
        let Some(base) = romfs else {
            return Ok(None);
        };
        let mut layers = Vec::new();
        let mut ext = Vec::new();
        for m in &self.mods {
            for kind in ["romfs", "romfslite"] {
                if let Some(dir) = find_caseless(m.dir.as_ref(), kind)? {
                    tracing::info!("mod {} ({kind})", m.name);
                    layers.push(dir);
                }
            }
            if let Some(dir) = find_caseless(m.dir.as_ref(), "romfs_ext")? {
                tracing::info!("mod {} (romfs_ext)", m.name);
                ext.push(dir);
            }
        }
        if layers.is_empty() && ext.is_empty() {
            return Ok(Some(base));
        }
        layers.push(open_romfs(&base)?);
        let mut root = LayeredDir::from_layers(layers);
        if !ext.is_empty() {
            root = RomfsExt::layer(root, LayeredDir::from_layers(ext));
        }
        Ok(Some(build_romfs(root.as_ref())?))
    }
}

fn find_caseless(dir: &dyn Directory, name: &str) -> LoaderResult<Option<DirRef>> {
    if let Some(found) = dir.open_dir(name)? {
        return Ok(Some(found));
    }
    let Some(entry) = dir
        .entries()?
        .into_iter()
        .find(|e| e.kind == EntryKind::Directory && e.name.eq_ignore_ascii_case(name))
    else {
        return Ok(None);
    };
    Ok(dir.open_dir(&entry.name)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exefs::load_exefs;
    use crate::npdm::tests::{NpdmSpec, build_npdm};
    use crate::nso::HashCheck;
    use crate::nso::tests::{Compression, build_nso, segments};
    use lichii_fs::MemDir;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("lichii-mods-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn write(&self, path: &str, data: &[u8]) {
            let path = self.0.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, data).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    const TITLE: u64 = 0x0100_ABCD_0000_0000;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02X}")).collect()
    }

    fn options(temp: &TempDir, disabled: &[&str]) -> ModOptions {
        ModOptions {
            load_root: Some(temp.0.join("mods")),
            sdmc_root: Some(temp.0.join("sdmc")),
            disabled: disabled.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn exefs() -> DirRef {
        let mut main = build_nso(&segments(1), Compression::Lz4, 0, false);
        main[0x40..0x60].fill(0);
        main[0x40..0x43].copy_from_slice(&[0x12, 0x34, 0x50]);
        MemDir::new()
            .with_bytes("main", main)
            .with_bytes("main.npdm", build_npdm(&NpdmSpec::default()))
            .into_ref()
    }

    #[test]
    fn exefs_patches_match_build_id_and_skip_disabled() {
        let (_, modules) = load_exefs(exefs().as_ref(), HashCheck::Skip).unwrap();
        let main = &modules[0];
        let id = hex(&main.build_id);
        let short = id.trim_end_matches('0').to_ascii_lowercase();
        assert_eq!(short, "12345");
        let temp = TempDir::new("exefs");
        let mut ips = b"PATCH".to_vec();
        ips.extend_from_slice(&[0, 1, 0, 0, 2, 0xde, 0xad]);
        ips.extend_from_slice(b"EOF");
        temp.write(&format!("mods/{TITLE:016X}/a/exefs/{short}.ips"), &ips);
        temp.write(
            &format!("mods/{TITLE:016X}/b/ExeFS/fix.pchtxt"),
            format!("@nsobid-{id}\n@enabled\n00000104 BEEF\n").as_bytes(),
        );
        temp.write(
            &format!("mods/{TITLE:016X}/c/exefs/other.pchtxt"),
            b"@nsobid-FF\n@enabled\n00000108 11\n",
        );
        temp.write(
            &format!("mods/{TITLE:016X}/off/exefs/{id}.ips"),
            &ips.iter().map(|_| 0).collect::<Vec<_>>(),
        );

        let mut patched = modules.clone();
        Mods::find(&options(&temp, &["off"]), TITLE)
            .unwrap()
            .patch_modules(&mut patched)
            .unwrap();
        let mut expected = main.image.clone();
        expected[0..2].copy_from_slice(&[0xde, 0xad]);
        expected[4..6].copy_from_slice(&[0xbe, 0xef]);
        assert_eq!(patched[0].image, expected);

        let mut untouched = modules.clone();
        Mods::find(&options(&temp, &["a", "b", "c", "off"]), TITLE)
            .unwrap()
            .patch_modules(&mut untouched)
            .unwrap();
        assert_eq!(untouched[0].image, main.image);
    }

    #[test]
    fn exefs_files_and_romfs_layer_in_name_order() {
        let temp = TempDir::new("layers");
        let replacement = build_nso(&segments(5), Compression::None, 0, false);
        temp.write(&format!("mods/{TITLE:016X}/b/exefs/main"), &replacement);
        temp.write(&format!("mods/{TITLE:016X}/a/romfs/data/new.bin"), b"a");
        temp.write(
            &format!("mods/{TITLE:016X}/a/romfs/data/shared.bin"),
            b"from a",
        );
        temp.write(
            &format!("mods/{TITLE:016X}/b/romfs/data/shared.bin"),
            b"from b",
        );
        temp.write(
            &format!("mods/{TITLE:016X}/b/romfs_ext/data/gone.bin.stub"),
            b"",
        );
        temp.write(
            &format!("sdmc/atmosphere/contents/{TITLE:016x}/romfslite/sd.bin"),
            b"sd",
        );
        let mods = Mods::find(&options(&temp, &[]), TITLE).unwrap();

        let layered = mods.layer_exefs(exefs()).unwrap();
        let (_, modules) = load_exefs(layered.as_ref(), HashCheck::Skip).unwrap();
        let (_, expected) = load_exefs(
            &MemDir::new()
                .with_bytes("main", replacement)
                .with_bytes("main.npdm", build_npdm(&NpdmSpec::default())),
            HashCheck::Skip,
        )
        .unwrap();
        assert_eq!(modules[0].image, expected[0].image);

        let base = build_romfs(
            &MemDir::new().with_dir(
                "data",
                MemDir::new()
                    .with_bytes("shared.bin", b"base".to_vec())
                    .with_bytes("gone.bin", b"x".to_vec())
                    .with_bytes("kept.bin", b"kept".to_vec()),
            ),
        )
        .unwrap();
        let rebuilt = open_romfs(&mods.layer_romfs(Some(base.clone())).unwrap().unwrap()).unwrap();
        let read = |path: &str| {
            rebuilt
                .open_file_path(path)
                .unwrap()
                .map(|f| f.read_all().unwrap())
        };
        assert_eq!(read("data/shared.bin").as_deref(), Some(&b"from a"[..]));
        assert_eq!(read("data/new.bin").as_deref(), Some(&b"a"[..]));
        assert_eq!(read("data/kept.bin").as_deref(), Some(&b"kept"[..]));
        assert_eq!(read("sd.bin").as_deref(), Some(&b"sd"[..]));
        assert!(read("data/gone.bin").is_none());

        let none = Mods::find(&options(&temp, &["a", "b", "SDMC"]), TITLE).unwrap();
        assert!(none.is_empty());
        let same = none.layer_romfs(Some(base.clone())).unwrap().unwrap();
        assert!(std::sync::Arc::ptr_eq(&same, &base));
        assert!(
            Mods::find(&ModOptions::default(), TITLE)
                .unwrap()
                .is_empty()
        );
    }
}
