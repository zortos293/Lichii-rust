//! NUP (NX Package): one file holding the loose ExeFS and RomFS trees of a base program, its
//! updates and its add-on content.
//!
//! Layout, every integer little-endian:
//! - a 0x80-byte superblock (`NUP\x01`) holding the SHA-256 of the table of contents;
//! - the table of contents (`TOC\x01`): package, directory and file records, the children
//!   array and a string pool;
//! - the blob region: one 16-byte aligned extent per file, stored raw or as one zstd frame,
//!   with the SHA-256 of the file's contents in its record;
//! - one block of TLV metadata per package (NPDM, names, icons, display version).
//!
//! Nothing is encrypted. Raw files are windows into the container. Zstd files decode a block
//! at a time into a cache the container shares between its files, with a memory budget, so a
//! pack of tens of gigabytes is never read whole.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use lichii_common::ConsoleGeneration;
use lichii_crypto::{Sha256, Sha256Hash, sha256};
use lichii_fs::host::HostFile;
use lichii_fs::nacp::{NACP_SIZE, Nacp, NacpLanguage};
use lichii_fs::romfs::build_romfs;
use lichii_fs::{
    DirRef, Directory, Entry, File, FileExt, FileRef, LayeredDir, MemDir, MemFile, slice,
};

use crate::error::{LoaderError, LoaderResult};
use crate::exefs::load_exefs;
use crate::patch::Mods;
use crate::program::{AddOnContent, ProgramImage};
use crate::title::{LoadOptions, TitleFormat, TitleInfo};
use crate::zstd::{FrameStream, Input, Variant, ZstdError};

pub const MAGIC: [u8; 4] = *b"NUP\x01";
pub(crate) const TOC_MAGIC: [u8; 4] = *b"TOC\x01";
pub(crate) const SUPERBLOCK_SIZE: usize = 0x80;
pub(crate) const TOC_HEADER_SIZE: usize = 0x30;
pub(crate) const PACKAGE_RECORD_SIZE: usize = 0x40;
pub(crate) const DIR_RECORD_SIZE: usize = 0x18;
pub(crate) const FILE_RECORD_SIZE: usize = 0x50;
pub(crate) const EXTENT_ALIGNMENT: u64 = 0x10;
pub(crate) const NO_DIR: u32 = u32::MAX;
pub(crate) const CODEC_RAW: u8 = 0;
pub(crate) const CODEC_ZSTD: u8 = 1;
pub(crate) const FLAG_STANDALONE_UPDATE: u8 = 0x01;
const TLV_HEADER_SIZE: usize = 8;
const MAX_DIR_DEPTH: usize = 128;
const MAX_TOC_SIZE: u64 = 64 << 20;
const MAX_META_SIZE: u32 = 64 << 20;
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// Types of the records in a package's metadata block.
pub const META_NPDM: u16 = 0x0001;
pub const META_ICON: u16 = 0x0003;
pub const META_DISPLAY_VERSION: u16 = 0x0004;
pub const META_TITLE_NAME: u16 = 0x0005;
pub const META_PUBLISHER: u16 = 0x0006;
pub const META_SOURCE_DUMP: u16 = 0x0008;
pub const META_LOCALIZED_TITLES: u16 = 0x0009;
pub const META_LOCALIZED_ICONS: u16 = 0x000a;

/// Decoded zstd files a container keeps in memory at once; the least recently read go first.
const DECODE_BUDGET: u64 = 512 << 20;
/// Zstd files larger than this are refused. The reference packer stores every file above
/// 256 MiB raw.
const MAX_ZSTD_FILE: u64 = DECODE_BUDGET / 2;
/// A read ending at most this far past a file's decoded prefix decodes only up to its end; one
/// further in decodes the whole file, so a random reader waits once per file.
pub(crate) const NEAR_DECODE: u64 = 1 << 20;
/// Compressed bytes read from the container per refill.
const INPUT_CHUNK: usize = 128 << 10;
/// Bytes hashed per read when verifying.
const VERIFY_CHUNK: usize = 1 << 20;

/// BCP-47 tags of localized titles and icons, in [`NacpLanguage`] order.
const LANGUAGE_TAGS: [&str; 18] = [
    "en-us", "en-gb", "ja", "fr", "de", "es-419", "es", "it", "nl", "fr-ca", "pt", "ru", "ko",
    "zh-hant", "zh-hans", "pt-br", "pl", "th",
];
/// Language slots of the uncompressed NACP title area.
const NACP_LANGUAGES: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackageKind {
    Base,
    Update,
    AddOn,
}

/// What a package's metadata block says about it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackageMeta {
    pub npdm: Vec<u8>,
    pub title_name: String,
    pub publisher: String,
    /// The name of the dump the package was packed from, the title of last resort.
    pub source_dump: String,
    /// `(BCP-47 tag, name)` pairs.
    pub localized_titles: Vec<(String, String)>,
    pub icon: Vec<u8>,
    /// `(BCP-47 tag, JPEG)` pairs.
    pub localized_icons: Vec<(String, Vec<u8>)>,
    pub display_version: String,
}

#[derive(Clone, Debug)]
pub struct Package {
    pub title_id: u64,
    pub version: u32,
    pub kind: PackageKind,
    /// An update that carries the complete patched tree and boots without its base.
    pub standalone_update: bool,
    pub meta: PackageMeta,
    root_dir: u32,
    exefs_dir: Option<u32>,
    romfs_dir: Option<u32>,
    meta_extent: (u64, u32),
}

struct DirNode {
    name: String,
    first_child: u32,
    child_count: u32,
}

struct FileNode {
    name: String,
    size: u64,
    sha256: Sha256Hash,
    extent_offset: u64,
    extent_size: u64,
    codec: u8,
}

/// An opened NUP container. Opening checks the table of contents against its SHA-256 and every
/// record against the file; [`Nup::verify`] checks the file contents.
pub struct Nup {
    file: FileRef,
    dirs: Vec<DirNode>,
    files: Vec<FileNode>,
    children: Vec<u32>,
    packages: Vec<Package>,
    cache: Arc<DecodeCache>,
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
}

fn corrupt(what: &'static str) -> LoaderError {
    LoaderError::Corrupt("NUP", what)
}

fn lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

impl Nup {
    /// Whether `file` starts with the NUP magic.
    pub fn is_nup(file: &dyn File) -> bool {
        file.size() >= SUPERBLOCK_SIZE as u64 && file.read_array::<4>(0).is_ok_and(|m| m == MAGIC)
    }

    pub fn open(file: FileRef) -> LoaderResult<Self> {
        let superblock: [u8; SUPERBLOCK_SIZE] = file
            .read_array(0)
            .map_err(|_| corrupt("shorter than its superblock"))?;
        if superblock[..4] != MAGIC {
            return Err(LoaderError::BadMagic("NUP"));
        }
        let toc_offset = u64_at(&superblock, 0x08);
        let toc_size = u64_at(&superblock, 0x10);
        let blob_offset = u64_at(&superblock, 0x38);
        let file_size = u64_at(&superblock, 0x40);
        if usize::from(u16_at(&superblock, 0x04)) < SUPERBLOCK_SIZE
            || u32_at(&superblock, 0x4c) != 0
        {
            return Err(corrupt("unsupported superblock"));
        }
        if file_size != file.size()
            || toc_offset < SUPERBLOCK_SIZE as u64
            || blob_offset > file_size
            || toc_offset > blob_offset
            || toc_size > blob_offset - toc_offset
            || !blob_offset.is_multiple_of(EXTENT_ALIGNMENT)
        {
            return Err(corrupt("superblock regions do not fit the file"));
        }
        if toc_size > MAX_TOC_SIZE {
            return Err(corrupt("table of contents larger than 64 MiB"));
        }
        let toc = file.read_vec(toc_offset, toc_size as usize)?;
        if sha256(&toc)[..] != superblock[0x18..0x38] {
            return Err(corrupt("table of contents does not match its SHA-256"));
        }
        let mut nup = Self::parse_toc(file, &toc, u32_at(&superblock, 0x48), blob_offset)?;
        for index in 0..nup.packages.len() {
            nup.packages[index].meta = nup.read_meta(nup.packages[index].meta_extent)?;
        }
        Ok(nup)
    }

    fn parse_toc(
        file: FileRef,
        toc: &[u8],
        package_count: u32,
        blob_offset: u64,
    ) -> LoaderResult<Self> {
        if toc.len() < TOC_HEADER_SIZE || toc[..4] != TOC_MAGIC {
            return Err(LoaderError::BadMagic("NUP table of contents"));
        }
        if u32_at(toc, 0x04) != package_count || package_count == 0 {
            return Err(corrupt("package count does not match the superblock"));
        }
        let dir_count = u64_at(toc, 0x08);
        let file_count = u64_at(toc, 0x10);
        let pool_offset = u64_at(toc, 0x18);
        let pool_size = u64_at(toc, 0x20);
        let len = toc.len() as u64;
        if dir_count > len / DIR_RECORD_SIZE as u64 || file_count > len / FILE_RECORD_SIZE as u64 {
            return Err(corrupt("more records than the table of contents holds"));
        }
        let records_end = TOC_HEADER_SIZE as u64
            + u64::from(package_count) * PACKAGE_RECORD_SIZE as u64
            + dir_count * DIR_RECORD_SIZE as u64
            + file_count * FILE_RECORD_SIZE as u64;
        if records_end > len
            || pool_offset < records_end
            || pool_offset > len
            || pool_size > len - pool_offset
            || !(pool_offset - records_end).is_multiple_of(4)
        {
            return Err(corrupt(
                "records and string pool do not fit the table of contents",
            ));
        }
        let (dir_count, file_count) = (dir_count as usize, file_count as usize);
        let file_size = file.size();
        let pool = &toc[pool_offset as usize..(pool_offset + pool_size) as usize];
        let name = |record: &[u8]| -> LoaderResult<String> {
            let start = u32_at(record, 0) as usize;
            let end = start + u32_at(record, 4) as usize;
            pool.get(start..end)
                .map(lossy)
                .ok_or(corrupt("name past the string pool"))
        };
        let dir_index = |value: u32| -> LoaderResult<Option<u32>> {
            match value {
                NO_DIR => Ok(None),
                dir if (dir as usize) < dir_count => Ok(Some(dir)),
                _ => Err(corrupt("package directory out of range")),
            }
        };
        let mut pos = TOC_HEADER_SIZE;
        let mut next = |size: usize| {
            let record = &toc[pos..pos + size];
            pos += size;
            record
        };

        let mut packages = Vec::with_capacity(package_count as usize);
        for _ in 0..package_count {
            let record = next(PACKAGE_RECORD_SIZE);
            let kind = match record[0x18] {
                0 => PackageKind::Base,
                1 => PackageKind::Update,
                2 => PackageKind::AddOn,
                _ => return Err(corrupt("unknown package kind")),
            };
            let flags = record[0x19];
            if flags & !FLAG_STANDALONE_UPDATE != 0 {
                return Err(corrupt("unknown package flags"));
            }
            let (meta_offset, meta_size) = (u64_at(record, 0x20), u32_at(record, 0x28));
            let meta_in_file = meta_size > 0
                && meta_offset != 0
                && meta_offset <= file_size
                && u64::from(meta_size) <= file_size - meta_offset;
            packages.push(Package {
                title_id: u64_at(record, 0x00),
                version: u32_at(record, 0x14),
                kind,
                standalone_update: flags & FLAG_STANDALONE_UPDATE != 0,
                meta: PackageMeta::default(),
                root_dir: dir_index(u32_at(record, 0x08))?
                    .ok_or(corrupt("package without a root directory"))?,
                exefs_dir: dir_index(u32_at(record, 0x0c))?,
                romfs_dir: dir_index(u32_at(record, 0x10))?,
                meta_extent: if meta_in_file {
                    (meta_offset, meta_size)
                } else {
                    (0, 0)
                },
            });
        }

        let child_slots = (pool_offset - records_end) / 4;
        let mut dirs = Vec::with_capacity(dir_count);
        for _ in 0..dir_count {
            let record = next(DIR_RECORD_SIZE);
            let (first_child, child_count) = (u32_at(record, 0x08), u32_at(record, 0x0c));
            if u64::from(first_child) + u64::from(child_count) > child_slots {
                return Err(corrupt("directory children past the children array"));
            }
            dirs.push(DirNode {
                name: name(record)?,
                first_child,
                child_count,
            });
        }

        let mut files = Vec::with_capacity(file_count);
        for _ in 0..file_count {
            let record = next(FILE_RECORD_SIZE);
            let size = u64_at(record, 0x08);
            let extent_offset = u64_at(record, 0x10);
            let extent_size = u64_at(record, 0x18);
            let codec = record[0x40];
            let extent_fits = if size == 0 {
                extent_offset == 0 && extent_size == 0
            } else {
                extent_size != 0
                    && extent_offset >= blob_offset
                    && extent_offset.is_multiple_of(EXTENT_ALIGNMENT)
                    && extent_offset <= file_size
                    && extent_size <= file_size - extent_offset
                    && (codec != CODEC_RAW || extent_size == size)
            };
            if codec > CODEC_ZSTD || record[0x41] != 0 || !extent_fits {
                return Err(corrupt("file record does not fit the file"));
            }
            files.push(FileNode {
                name: name(record)?,
                size,
                sha256: record[0x20..0x40].try_into().unwrap(),
                extent_offset,
                extent_size,
                codec,
            });
        }

        let children: Vec<u32> = toc[records_end as usize..pool_offset as usize]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| u32::from_le_bytes(c))
            .collect();
        if children
            .iter()
            .any(|&child| child as usize >= dir_count + file_count)
        {
            return Err(corrupt("child out of range"));
        }
        let nup = Self {
            file,
            dirs,
            files,
            children,
            packages,
            cache: Arc::default(),
        };
        nup.validate_tree()?;
        Ok(nup)
    }

    fn slots(&self, dir: u32) -> &[u32] {
        let node = &self.dirs[dir as usize];
        &self.children[node.first_child as usize..][..node.child_count as usize]
    }

    /// Directories have at most one parent, and every tree a package names is at most
    /// `MAX_DIR_DEPTH` deep, so a cycle reachable from one is caught too.
    fn validate_tree(&self) -> LoaderResult<()> {
        let mut has_parent = vec![false; self.dirs.len()];
        for &child in &self.children {
            if let Some(seen) = has_parent.get_mut(child as usize)
                && std::mem::replace(seen, true)
            {
                return Err(corrupt("a directory has two parents"));
            }
        }
        let roots = self.packages.iter().flat_map(|p| {
            [Some(p.root_dir), p.exefs_dir, p.romfs_dir]
                .into_iter()
                .flatten()
        });
        for root in roots {
            let mut stack = vec![(root, 1)];
            while let Some((dir, depth)) = stack.pop() {
                if depth > MAX_DIR_DEPTH {
                    return Err(corrupt("directory tree nested too deeply or cyclic"));
                }
                stack.extend(
                    self.slots(dir)
                        .iter()
                        .filter(|&&child| (child as usize) < self.dirs.len())
                        .map(|&child| (child, depth + 1)),
                );
            }
        }
        Ok(())
    }

    fn read_meta(&self, (offset, size): (u64, u32)) -> LoaderResult<PackageMeta> {
        let mut meta = PackageMeta::default();
        if size == 0 {
            return Ok(meta);
        }
        if size > MAX_META_SIZE {
            return Err(corrupt("package metadata larger than 64 MiB"));
        }
        let block = self.file.read_vec(offset, size as usize)?;
        let mut pos = 0;
        while pos + TLV_HEADER_SIZE <= block.len() {
            let kind = u16_at(&block, pos);
            let len = u32_at(&block, pos + 4) as usize;
            pos += TLV_HEADER_SIZE;
            let payload = block
                .get(pos..pos + len)
                .ok_or(corrupt("metadata record past its block"))?;
            pos += len;
            match kind {
                META_NPDM => meta.npdm = payload.to_vec(),
                META_ICON => meta.icon = payload.to_vec(),
                META_TITLE_NAME => meta.title_name = lossy(payload),
                META_PUBLISHER => meta.publisher = lossy(payload),
                META_SOURCE_DUMP => meta.source_dump = lossy(payload),
                META_DISPLAY_VERSION => meta.display_version = lossy(payload),
                META_LOCALIZED_TITLES => meta.localized_titles.extend(
                    localized(payload)?
                        .into_iter()
                        .map(|(tag, name)| (tag, lossy(name))),
                ),
                META_LOCALIZED_ICONS => meta.localized_icons.extend(
                    localized(payload)?
                        .into_iter()
                        .map(|(tag, icon)| (tag, icon.to_vec())),
                ),
                _ => {}
            }
        }
        Ok(meta)
    }

    pub fn packages(&self) -> &[Package] {
        &self.packages
    }

    /// A package's ExeFS, when it has one with anything in it.
    pub fn exefs(&self, package: usize) -> LoaderResult<Option<DirRef>> {
        let Some(dir) = self.packages[package].exefs_dir else {
            return Ok(None);
        };
        let exefs = self.build_dir(dir, false)?;
        Ok((!exefs.entries()?.is_empty()).then(|| exefs.into_ref()))
    }

    /// A package's RomFS tree. A zstd-packed `.zs` file that does not decode to a zstd frame
    /// (a damaged dump) is replaced by its closest sound sibling, as the reference does.
    pub fn romfs(&self, package: usize) -> LoaderResult<Option<DirRef>> {
        self.packages[package]
            .romfs_dir
            .map(|dir| Ok(self.build_dir(dir, true)?.into_ref()))
            .transpose()
    }

    /// Hashes every file against the SHA-256 in its record, a chunk at a time.
    pub fn verify(&self) -> LoaderResult<()> {
        let mut buf = vec![0; VERIFY_CHUNK];
        for (index, node) in self.files.iter().enumerate() {
            let file = self.open_file(index);
            let mut hash = Sha256::new();
            let mut offset = 0;
            while offset < node.size {
                let len = (node.size - offset).min(VERIFY_CHUNK as u64) as usize;
                file.read_exact_at(offset, &mut buf[..len])?;
                hash.update(&buf[..len]);
                offset += len as u64;
            }
            if hash.finalize() != node.sha256 {
                return Err(LoaderError::FileHashMismatch(node.name.clone()));
            }
        }
        Ok(())
    }

    fn open_file(&self, index: usize) -> FileRef {
        let node = &self.files[index];
        match (node.size, node.codec) {
            (0, _) => MemFile::from_bytes(Vec::new()),
            (_, CODEC_RAW) => slice(&self.file, node.extent_offset, node.size),
            _ => Arc::new(ZstdFile {
                container: self.file.clone(),
                name: node.name.clone(),
                index,
                size: node.size,
                extent_offset: node.extent_offset,
                extent_size: node.extent_size,
                cache: self.cache.clone(),
            }),
        }
    }

    fn build_dir(&self, dir: u32, stand_ins: bool) -> LoaderResult<MemDir> {
        let mut out = MemDir::new();
        let mut files = Vec::new();
        let mut damaged = Vec::new();
        for &child in self.slots(dir) {
            let Some(index) = (child as usize).checked_sub(self.dirs.len()) else {
                let sub = self.build_dir(child, stand_ins)?;
                out.insert(
                    self.dirs[child as usize].name.clone(),
                    Entry::Dir(Arc::new(sub)),
                );
                continue;
            };
            files.push(index);
            if stand_ins && self.is_damaged_zs(index) {
                damaged.push(index);
            } else {
                out.insert(
                    self.files[index].name.clone(),
                    Entry::File(self.open_file(index)),
                );
            }
        }
        for index in damaged {
            let name = &self.files[index].name;
            let served = match self.stand_in(index, &files) {
                Some(stand_in) => {
                    tracing::warn!(
                        "NUP: {name} has no zstd header, so this dump is damaged; serving {} in its place, dump the game again to get the original",
                        self.files[stand_in].name
                    );
                    stand_in
                }
                None => {
                    tracing::warn!(
                        "NUP: {name} has no zstd header, so this dump is damaged and the game cannot load it; dump the game again"
                    );
                    index
                }
            };
            out.insert(name.clone(), Entry::File(self.open_file(served)));
        }
        Ok(out)
    }

    /// A game's `.zs` is zstd already, so the packer only recompresses one that shrinks, which
    /// a zeroed range does.
    fn is_damaged_zs(&self, index: usize) -> bool {
        let node = &self.files[index];
        node.codec == CODEC_ZSTD
            && node.name.ends_with(".zs")
            && self.has_zstd_magic(index) == Some(false)
    }

    /// Whether a file starts with the zstd frame magic; nothing when it cannot be read.
    fn has_zstd_magic(&self, index: usize) -> Option<bool> {
        if self.files[index].size < ZSTD_MAGIC.len() as u64 {
            return Some(false);
        }
        self.open_file(index)
            .read_array::<4>(0)
            .ok()
            .map(|head| head == ZSTD_MAGIC)
    }

    /// The sibling to serve for a damaged file: the same last name token (`Nrm.bntx.zs`), the
    /// longest common name prefix, then the closest size, so a texture of the same kind and
    /// often the same resolution.
    fn stand_in(&self, damaged: usize, siblings: &[usize]) -> Option<usize> {
        let name = &self.files[damaged].name;
        let suffix = &name[name.rfind('_')? + 1..];
        let size = self.files[damaged].size;
        let mut best: Option<(usize, usize, u64)> = None;
        for &candidate in siblings {
            let other = &self.files[candidate].name;
            if candidate == damaged
                || other.len() <= suffix.len()
                || !other.ends_with(suffix)
                || other.as_bytes()[other.len() - suffix.len() - 1] != b'_'
            {
                continue;
            }
            let prefix = name
                .bytes()
                .zip(other.bytes())
                .take_while(|(a, b)| a == b)
                .count();
            let gap = self.files[candidate].size.abs_diff(size);
            if best.is_some_and(|(_, p, g)| prefix < p || (prefix == p && gap >= g)) {
                continue;
            }
            if self.has_zstd_magic(candidate) == Some(true) {
                best = Some((candidate, prefix, gap));
            }
        }
        best.map(|(index, _, _)| index)
    }

    fn base(&self) -> Option<usize> {
        self.packages
            .iter()
            .position(|p| p.kind == PackageKind::Base)
    }

    /// The update with the highest version, the first of several with the same one.
    fn newest_update(&self) -> Option<usize> {
        self.packages
            .iter()
            .enumerate()
            .filter(|(_, p)| p.kind == PackageKind::Update)
            .rev()
            .max_by_key(|(_, p)| p.version)
            .map(|(index, _)| index)
    }

    /// The program comes from the base with the newest update's ExeFS over it; a standalone
    /// update carries the complete patched tree and boots on its own.
    fn select(&self, path: &Path) -> LoaderResult<Selection> {
        let base = self.base();
        let update = self.newest_update();
        let exefs = |package: Option<usize>| -> LoaderResult<Option<DirRef>> {
            Ok(match package {
                Some(package) => self.exefs(package)?,
                None => None,
            })
        };
        let (program, exefs) = match (base, exefs(base)?, update, exefs(update)?) {
            (Some(base), Some(base_exefs), _, update_exefs) => {
                (base, update_exefs.unwrap_or(base_exefs))
            }
            (_, _, Some(update), Some(update_exefs)) => {
                tracing::info!(
                    "{} is a standalone update; booting the update's program",
                    path.display()
                );
                (update, update_exefs)
            }
            _ => return Err(LoaderError::MissingNupProgram(path.display().to_string())),
        };
        Ok(Selection {
            program,
            exefs,
            base,
            update,
        })
    }

    /// The program's metadata, with the base's names, publisher and icons when the program is
    /// a standalone update.
    fn metadata(&self, selection: &Selection) -> PackageMeta {
        let mut meta = self.packages[selection.program].meta.clone();
        let Some(base) = selection.base.filter(|&b| b != selection.program) else {
            return meta;
        };
        let base = &self.packages[base].meta;
        let prefer = |field: &mut String, value: &String| {
            if !value.is_empty() {
                field.clone_from(value);
            }
        };
        prefer(&mut meta.title_name, &base.title_name);
        prefer(&mut meta.source_dump, &base.source_dump);
        prefer(&mut meta.publisher, &base.publisher);
        if !base.localized_titles.is_empty() {
            meta.localized_titles.clone_from(&base.localized_titles);
        }
        if !base.icon.is_empty() || !base.localized_icons.is_empty() {
            meta.icon.clone_from(&base.icon);
            meta.localized_icons.clone_from(&base.localized_icons);
        }
        meta
    }

    /// The newest update's display version, else the program's.
    fn display_version(&self, selection: &Selection) -> String {
        let package = &self.packages[selection.update.unwrap_or(selection.program)];
        if !package.meta.display_version.is_empty() {
            return package.meta.display_version.clone();
        }
        match package.version {
            0 => String::new(),
            v => format!("{}.{}.{}", v >> 16, (v >> 8) & 0xff, v & 0xff),
        }
    }

    /// The base's RomFS with the newest update's files over it.
    fn merged_romfs(&self, selection: &Selection) -> LoaderResult<Option<FileRef>> {
        let mut layers = Vec::new();
        for package in [selection.update, selection.base].into_iter().flatten() {
            layers.extend(self.romfs(package)?);
        }
        if layers.is_empty() {
            return Ok(None);
        }
        Ok(Some(build_romfs(LayeredDir::from_layers(layers).as_ref())?))
    }

    /// The newest version of each add-on, by title id.
    fn add_ons(&self) -> Vec<AddOnContent> {
        let mut newest: BTreeMap<u64, usize> = BTreeMap::new();
        for (index, package) in self.packages.iter().enumerate() {
            if package.kind != PackageKind::AddOn {
                continue;
            }
            newest
                .entry(package.title_id)
                .and_modify(|best| {
                    if self.packages[*best].version < package.version {
                        *best = index;
                    }
                })
                .or_insert(index);
        }
        newest
            .into_iter()
            .filter_map(|(title_id, index)| {
                let romfs = self
                    .romfs(index)
                    .and_then(|dir| dir.map(|d| Ok(build_romfs(d.as_ref())?)).transpose());
                match romfs {
                    Ok(Some(romfs)) => Some(AddOnContent {
                        title_id,
                        version: self.packages[index].version,
                        romfs,
                    }),
                    Ok(None) => {
                        tracing::warn!("NUP add-on {title_id:016x} has no RomFS");
                        None
                    }
                    Err(error) => {
                        tracing::warn!("skipping NUP add-on {title_id:016x}: {error}");
                        None
                    }
                }
            })
            .collect()
    }
}

/// The records of a localized titles or icons block: a one-byte tag length, the tag, a 32-bit
/// data length and the data.
fn localized(payload: &[u8]) -> LoaderResult<Vec<(String, &[u8])>> {
    let past = || corrupt("localized metadata past its record");
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < payload.len() {
        let tag_len = usize::from(payload[pos]);
        pos += 1;
        let tag = payload.get(pos..pos + tag_len).ok_or_else(past)?;
        pos += tag_len;
        let len = payload.get(pos..pos + 4).ok_or_else(past)?;
        pos += 4;
        let len = u32_at(len, 0) as usize;
        let data = payload.get(pos..pos + len).ok_or_else(past)?;
        pos += len;
        out.push((lossy(tag), data));
    }
    Ok(out)
}

/// The packages a NUP title boots from.
struct Selection {
    program: usize,
    exefs: DirRef,
    base: Option<usize>,
    update: Option<usize>,
}

fn put_c_string(raw: &mut [u8], at: usize, capacity: usize, text: &str) {
    let mut end = text.len().min(capacity - 1);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    raw[at..at + end].copy_from_slice(&text.as_bytes()[..end]);
    raw[at + end] = 0;
}

/// A NACP holding what the metadata says: titles per language (the default title in American
/// English when it has no localized one), the publisher, the display version and the save data
/// owner. Nothing when the metadata has none of them.
fn build_nacp(meta: &PackageMeta, title_id: u64, version: &str) -> Option<Nacp> {
    let title = if meta.title_name.is_empty() {
        &meta.source_dump
    } else {
        &meta.title_name
    };
    if title.is_empty()
        && meta.localized_titles.is_empty()
        && meta.publisher.is_empty()
        && version.is_empty()
    {
        return None;
    }
    let mut raw = vec![0u8; NACP_SIZE];
    for (tag, name) in &meta.localized_titles {
        if let Some(slot) = LANGUAGE_TAGS[..NACP_LANGUAGES]
            .iter()
            .position(|t| t == tag)
        {
            put_c_string(&mut raw, slot * 0x300, 0x200, name);
        }
    }
    if !title.is_empty() && raw[0] == 0 {
        put_c_string(&mut raw, 0, 0x200, title);
    }
    if !meta.publisher.is_empty() {
        for slot in 0..NACP_LANGUAGES {
            put_c_string(&mut raw, slot * 0x300 + 0x200, 0x100, &meta.publisher);
        }
    }
    put_c_string(&mut raw, 0x3060, 0x10, version);
    raw[0x302c..0x3030].copy_from_slice(&0x3_ffffu32.to_le_bytes());
    raw[0x3078..0x3080].copy_from_slice(&title_id.to_le_bytes());
    Nacp::parse(&raw).ok()
}

/// The default icon, else the one for `language`, else American English, else the first.
fn icon(meta: &PackageMeta, language: NacpLanguage) -> Option<Vec<u8>> {
    if !meta.icon.is_empty() {
        return Some(meta.icon.clone());
    }
    let find = |tag: &str| meta.localized_icons.iter().find(|(t, _)| t == tag);
    find(LANGUAGE_TAGS[language as usize])
        .or_else(|| find(LANGUAGE_TAGS[0]))
        .or_else(|| meta.localized_icons.first())
        .map(|(_, icon)| icon.clone())
}

/// Reads the game-list entry of a NUP title from its metadata, without opening any module.
pub(crate) fn scan(path: &Path, options: &LoadOptions) -> LoaderResult<TitleInfo> {
    let nup = Nup::open(HostFile::open(path)?)?;
    let selection = nup.select(path)?;
    let title_id = nup.packages[selection.program].title_id;
    let meta = nup.metadata(&selection);
    let version = nup.display_version(&selection);
    let nacp = build_nacp(&meta, title_id, &version);
    let title = nacp
        .as_ref()
        .and_then(|n| n.preferred_title(options.language));
    Ok(TitleInfo {
        path: path.to_path_buf(),
        format: TitleFormat::Nup,
        program_id: title_id,
        generation: ConsoleGeneration::from_program_id(title_id),
        name: title.map_or_else(
            || {
                path.file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            },
            |t| t.name.clone(),
        ),
        publisher: meta.publisher.clone(),
        version,
        icon: icon(&meta, options.language),
    })
}

/// Loads a NUP title: the base program with its newest update and mods applied, the merged
/// RomFS, and the newest version of every add-on.
pub(crate) fn load(path: &Path, options: &LoadOptions) -> LoaderResult<ProgramImage> {
    let nup = Nup::open(HostFile::open(path)?)?;
    let selection = nup.select(path)?;
    let program = &nup.packages[selection.program];
    let mods = Mods::find(&options.mods, program.title_id)?;
    let exefs = mods.layer_exefs(selection.exefs.clone())?;
    let (npdm, mut modules) = load_exefs(exefs.as_ref(), options.hashes)?;
    mods.patch_modules(&mut modules)?;
    let meta = nup.metadata(&selection);
    let nacp = build_nacp(&meta, program.title_id, &nup.display_version(&selection));
    let romfs = mods.layer_romfs(nup.merged_romfs(&selection)?)?;
    let mut image = ProgramImage::new(npdm, modules, romfs, nacp);
    image.icon = icon(&meta, options.language);
    image.version = nup.packages[selection.update.unwrap_or(selection.program)].version;
    image.add_ons = nup.add_ons();
    Ok(image)
}

/// A zstd-packed file, decoded from its start as reads need it.
struct ZstdFile {
    container: FileRef,
    name: String,
    index: usize,
    size: u64,
    extent_offset: u64,
    extent_size: u64,
    cache: Arc<DecodeCache>,
}

impl ZstdFile {
    fn decode(&self, member: &mut Member, target: usize) -> Result<(), ZstdError> {
        if member.failed {
            return Err(ZstdError::Corrupt(
                "an earlier read failed to decode this file",
            ));
        }
        let mut frame = match member.stream.take() {
            Some(frame) => frame,
            None => {
                let input = ExtentInput {
                    file: self.container.clone(),
                    next: self.extent_offset,
                    end: self.extent_offset + self.extent_size,
                    buf: Vec::new(),
                    start: 0,
                    filled: 0,
                };
                let frame = FrameStream::new(Variant::Zstd, input)?;
                if frame.content_size().is_some_and(|size| size != self.size) {
                    return Err(ZstdError::Corrupt(
                        "the frame's content size is not the file size",
                    ));
                }
                member.data = vec![0; self.size as usize];
                frame
            }
        };
        member.decoded = frame.decode_to(&mut member.data, target)?;
        if !frame.is_finished() {
            member.stream = Some(frame);
            return Ok(());
        }
        if member.decoded as u64 != self.size || frame.input().remaining() != 0 {
            return Err(ZstdError::Corrupt("the frame does not end with the file"));
        }
        Ok(())
    }
}

impl File for ZstdFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= self.size || buf.is_empty() {
            return Ok(0);
        }
        if self.size > MAX_ZSTD_FILE {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!(
                    "{}: a zstd-packed file of {} bytes is larger than the 256 MiB Lichii decodes",
                    self.name, self.size
                ),
            ));
        }
        let len = buf.len().min((self.size - offset) as usize);
        let end = offset + len as u64;
        let member = self.cache.member(self.index, self.size);
        let mut member = member.lock().unwrap_or_else(PoisonError::into_inner);
        let decoded = member.decoded as u64;
        if decoded < end {
            let target = if end - decoded <= NEAR_DECODE {
                end
            } else {
                self.size
            };
            if let Err(error) = self.decode(&mut member, target as usize) {
                member.failed = true;
                member.stream = None;
                tracing::error!("NUP: decoding {} failed: {error}", self.name);
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}: {error}", self.name),
                ));
            }
        }
        buf[..len].copy_from_slice(&member.data[offset as usize..end as usize]);
        Ok(len)
    }
}

/// The decoded prefix of one zstd file, and the decoder that grows it.
#[derive(Default)]
struct Member {
    data: Vec<u8>,
    decoded: usize,
    stream: Option<FrameStream<ExtentInput>>,
    failed: bool,
}

/// The zstd files of one container that were read recently, charged their full size against
/// `DECODE_BUDGET`. An evicted file stays alive while a read holds it and decodes again on
/// its next read.
#[derive(Default)]
struct DecodeCache {
    state: Mutex<CacheState>,
}

#[derive(Default)]
struct CacheState {
    members: HashMap<usize, Cached>,
    by_use: BTreeMap<u64, usize>,
    bytes: u64,
    clock: u64,
}

struct Cached {
    member: Arc<Mutex<Member>>,
    size: u64,
    used: u64,
}

impl DecodeCache {
    fn member(&self, index: usize, size: u64) -> Arc<Mutex<Member>> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let state = &mut *state;
        state.clock += 1;
        let clock = state.clock;
        if let Some(cached) = state.members.get_mut(&index) {
            state.by_use.remove(&cached.used);
            cached.used = clock;
            state.by_use.insert(clock, index);
            return cached.member.clone();
        }
        while state.bytes + size > DECODE_BUDGET
            && let Some((_, victim)) = state.by_use.pop_first()
        {
            if let Some(evicted) = state.members.remove(&victim) {
                state.bytes -= evicted.size;
            }
        }
        let member = Arc::<Mutex<Member>>::default();
        state.bytes += size;
        state.by_use.insert(clock, index);
        state.members.insert(
            index,
            Cached {
                member: member.clone(),
                size,
                used: clock,
            },
        );
        member
    }
}

/// A file's compressed extent, read `INPUT_CHUNK` bytes at a time.
struct ExtentInput {
    file: FileRef,
    next: u64,
    end: u64,
    buf: Vec<u8>,
    start: usize,
    filled: usize,
}

impl ExtentInput {
    fn remaining(&self) -> u64 {
        (self.filled - self.start) as u64 + (self.end - self.next)
    }
}

impl Input for ExtentInput {
    fn take(&mut self, len: usize) -> Result<&[u8], ZstdError> {
        let have = self.filled - self.start;
        if have < len {
            let unread = self.end - self.next;
            if (len - have) as u64 > unread {
                return Err(ZstdError::Truncated);
            }
            self.buf.copy_within(self.start..self.filled, 0);
            let read = ((len.max(INPUT_CHUNK) - have) as u64).min(unread) as usize;
            if self.buf.len() < have + read {
                self.buf.resize(have + read, 0);
            }
            self.file
                .read_exact_at(self.next, &mut self.buf[have..have + read])?;
            self.next += read as u64;
            self.start = 0;
            self.filled = have + read;
        }
        let bytes = &self.buf[self.start..self.start + len];
        self.start += len;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nso::tests::{Compression, build_nso, segments};
    use crate::testing::{
        AcidLayout, Codec, NpdmSpec, NupPackage, ZSTD_TEST_BLOCK, build_npdm, build_nup, zstd_frame,
    };
    use crate::title::tests::TempDir;
    use lichii_fs::romfs::open_romfs;
    use lichii_fs::{DirectoryExt, EntryKind};

    const BASE_ID: u64 = 0x0100_0000_0001_0000;
    const BLOCK: u64 = ZSTD_TEST_BLOCK as u64;

    /// Bytes that do not repeat within a block, with every third block a single byte, so a frame
    /// holds both raw and RLE blocks.
    fn content(len: usize, seed: u32) -> Vec<u8> {
        (0..len)
            .map(|i| match i / ZSTD_TEST_BLOCK {
                block if block % 3 == 1 => (block as u8).wrapping_add(seed as u8),
                _ => ((i as u32).wrapping_mul(2_654_435_761).wrapping_add(seed) >> 13) as u8,
            })
            .collect()
    }

    fn open(bytes: Vec<u8>) -> LoaderResult<Nup> {
        Nup::open(MemFile::from_bytes(bytes))
    }

    fn read(dir: &DirRef, path: &str) -> Vec<u8> {
        dir.open_file_path(path)
            .unwrap()
            .unwrap_or_else(|| panic!("{path} is missing"))
            .read_all()
            .unwrap()
    }

    fn listing(dir: &DirRef) -> Vec<(String, EntryKind, u64)> {
        dir.entries()
            .unwrap()
            .into_iter()
            .map(|e| (e.name, e.kind, e.size))
            .collect()
    }

    fn decoded(nup: &Nup, name: &str) -> usize {
        let index = nup.files.iter().position(|f| f.name == name).unwrap();
        let state = nup.cache.state.lock().unwrap();
        state
            .members
            .get(&index)
            .map_or(0, |c| c.member.lock().unwrap().decoded)
    }

    #[test]
    fn round_trips_packages_trees_and_file_data() {
        let big = content(3 * ZSTD_TEST_BLOCK + 123, 1);
        let raw = content(5000, 2);
        let base = NupPackage::new(PackageKind::Base, BASE_ID, 0)
            .file("exefs/main", b"main code".to_vec(), Codec::Raw)
            .file("exefs/main.npdm", b"npdm bytes".to_vec(), Codec::Zstd)
            .file("romfs/Data/big.bin", big.clone(), Codec::Zstd)
            .file("romfs/Data/raw.bin", raw.clone(), Codec::Raw)
            .file("romfs/Data/Deep/er/leaf.txt", b"leaf".to_vec(), Codec::Zstd)
            .file("romfs/empty.txt", Vec::new(), Codec::Raw)
            .meta(META_TITLE_NAME, "Synthetic Kart")
            .meta(META_PUBLISHER, "Lichii Tests")
            .meta(META_NPDM, b"npdm bytes".to_vec())
            .localized(
                META_LOCALIZED_TITLES,
                &[("ja", "シンセ".as_bytes()), ("fr", b"Kart FR")],
            )
            .localized(META_LOCALIZED_ICONS, &[("en-us", b"icon")]);
        let update = NupPackage::new(PackageKind::Update, BASE_ID | 0x800, 0x0001_0200)
            .file("romfs/Data/raw.bin", b"patched".to_vec(), Codec::Raw)
            .meta(META_DISPLAY_VERSION, "1.2.0");
        let dlc = NupPackage::new(PackageKind::AddOn, BASE_ID + 0x1001, 3).file(
            "romfs/dlc.bin",
            b"dlc".to_vec(),
            Codec::Zstd,
        );
        let nup = open(build_nup(&[base, update, dlc])).unwrap();

        let summary: Vec<_> = nup
            .packages()
            .iter()
            .map(|p| (p.kind, p.title_id, p.version))
            .collect();
        assert_eq!(
            summary,
            [
                (PackageKind::Base, BASE_ID, 0),
                (PackageKind::Update, BASE_ID | 0x800, 0x0001_0200),
                (PackageKind::AddOn, BASE_ID + 0x1001, 3),
            ]
        );
        let meta = &nup.packages()[0].meta;
        assert_eq!(meta.title_name, "Synthetic Kart");
        assert_eq!(meta.publisher, "Lichii Tests");
        assert_eq!(meta.npdm, b"npdm bytes");
        assert_eq!(
            meta.localized_titles,
            [
                ("ja".to_string(), "シンセ".to_string()),
                ("fr".to_string(), "Kart FR".to_string())
            ]
        );
        assert_eq!(
            meta.localized_icons,
            [("en-us".to_string(), b"icon".to_vec())]
        );
        assert_eq!(nup.packages()[1].meta.display_version, "1.2.0");

        let exefs = nup.exefs(0).unwrap().unwrap();
        assert_eq!(
            listing(&exefs),
            [
                ("main".into(), EntryKind::File, 9),
                ("main.npdm".into(), EntryKind::File, 10),
            ]
        );
        assert_eq!(read(&exefs, "main"), b"main code");
        assert_eq!(read(&exefs, "main.npdm"), b"npdm bytes");
        assert!(nup.exefs(1).unwrap().is_none());
        assert!(nup.exefs(2).unwrap().is_none());

        let romfs = nup.romfs(0).unwrap().unwrap();
        assert_eq!(
            listing(&romfs),
            [
                ("Data".into(), EntryKind::Directory, 0),
                ("empty.txt".into(), EntryKind::File, 0),
            ]
        );
        let data = romfs.open_dir("Data").unwrap().unwrap();
        assert_eq!(
            listing(&data),
            [
                ("Deep".into(), EntryKind::Directory, 0),
                ("big.bin".into(), EntryKind::File, big.len() as u64),
                ("raw.bin".into(), EntryKind::File, raw.len() as u64),
            ]
        );
        assert_eq!(read(&romfs, "Data/big.bin"), big);
        assert_eq!(read(&romfs, "Data/raw.bin"), raw);
        assert_eq!(read(&romfs, "Data/Deep/er/leaf.txt"), b"leaf");
        assert_eq!(read(&romfs, "empty.txt"), b"");
        assert_eq!(
            read(&nup.romfs(1).unwrap().unwrap(), "Data/raw.bin"),
            b"patched"
        );
        assert_eq!(read(&nup.romfs(2).unwrap().unwrap(), "dlc.bin"), b"dlc");
        nup.verify().unwrap();
    }

    #[test]
    fn rejects_a_table_of_contents_or_file_that_does_not_match_its_hash() {
        let main = content(300, 4);
        let zipped = content(70_000, 5);
        let bytes = build_nup(&[NupPackage::new(PackageKind::Base, BASE_ID, 0)
            .file("exefs/main", main.clone(), Codec::Raw)
            .file("romfs/a.bin", zipped.clone(), Codec::Zstd)]);
        open(bytes.clone()).unwrap().verify().unwrap();
        let toc_hash_mismatch = |result: LoaderResult<Nup>| matches!(result, Err(LoaderError::Corrupt("NUP", what)) if what.contains("SHA-256"));

        let mut toc = bytes.clone();
        toc[SUPERBLOCK_SIZE + TOC_HEADER_SIZE] ^= 1;
        assert!(toc_hash_mismatch(open(toc)));
        let mut stored_hash = bytes.clone();
        stored_hash[0x18] ^= 0x80;
        assert!(toc_hash_mismatch(open(stored_hash)));

        let find = |needle: &[u8]| {
            bytes
                .windows(needle.len())
                .position(|w| w == needle)
                .unwrap()
        };
        let mut raw = bytes.clone();
        raw[find(&main) + 17] ^= 1;
        let nup = open(raw).unwrap();
        assert!(matches!(nup.verify(), Err(LoaderError::FileHashMismatch(name)) if name == "main"));

        let mut frame = bytes.clone();
        frame[find(&zipped[..64]) + 1000] ^= 1;
        let nup = open(frame).unwrap();
        assert!(
            matches!(nup.verify(), Err(LoaderError::FileHashMismatch(name)) if name == "a.bin")
        );

        let mut truncated = bytes.clone();
        truncated.truncate(bytes.len() - 1);
        assert!(open(truncated).is_err());
        let mut magic = bytes;
        magic[3] = 2;
        assert!(matches!(open(magic), Err(LoaderError::BadMagic("NUP"))));
    }

    #[test]
    fn random_reads_cross_zstd_block_boundaries() {
        let zipped = content(40 * ZSTD_TEST_BLOCK + 777, 9);
        let raw = content(3 * ZSTD_TEST_BLOCK + 5, 10);
        assert!(zipped.len() as u64 > 2 * NEAR_DECODE);
        let bytes = build_nup(&[NupPackage::new(PackageKind::Base, BASE_ID, 0)
            .file("romfs/z.bin", zipped.clone(), Codec::Zstd)
            .file("romfs/r.bin", raw.clone(), Codec::Raw)]);
        let check = |file: &FileRef, expected: &[u8], offset: u64, len: usize| {
            let mut buf = vec![0xee; len];
            let n = file.read_at(offset, &mut buf).unwrap();
            let start = (offset as usize).min(expected.len());
            let end = (start + len).min(expected.len());
            assert_eq!(n, end - start, "read {len} at {offset}");
            assert!(buf[..n] == expected[start..end], "read {len} at {offset}");
        };

        let nup = open(bytes.clone()).unwrap();
        let romfs = nup.romfs(0).unwrap().unwrap();
        let file = romfs.open_file("z.bin").unwrap().unwrap();
        assert_eq!(decoded(&nup, "z.bin"), 0, "listing decodes nothing");
        check(&file, &zipped, 0, 10);
        assert_eq!(
            decoded(&nup, "z.bin"),
            ZSTD_TEST_BLOCK,
            "a short read decodes one block"
        );
        check(&file, &zipped, BLOCK - 3, 2 * ZSTD_TEST_BLOCK);
        assert_eq!(decoded(&nup, "z.bin"), 3 * ZSTD_TEST_BLOCK);
        check(&file, &zipped, 30 * BLOCK - 1, 2);
        assert_eq!(
            decoded(&nup, "z.bin"),
            zipped.len(),
            "a far read decodes the rest"
        );

        let mut seed = 0x1234_5678u64;
        let mut next = move |bound: u64| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (seed >> 33) % bound
        };
        for _ in 0..3 {
            let nup = open(bytes.clone()).unwrap();
            let romfs = nup.romfs(0).unwrap().unwrap();
            for (name, expected) in [("z.bin", &zipped), ("r.bin", &raw)] {
                let file = romfs.open_file(name).unwrap().unwrap();
                let size = expected.len() as u64;
                assert_eq!(file.size(), size);
                for round in 0..300 {
                    let (offset, len) = if round % 2 == 0 {
                        let boundary = (next(size / BLOCK) + 1) * BLOCK;
                        (boundary - next(300) - 1, next(600) as usize + 2)
                    } else {
                        (next(size), next(3 * BLOCK) as usize + 1)
                    };
                    check(&file, expected, offset, len);
                }
                check(&file, expected, size - 3, 8);
                check(&file, expected, size, 4);
                check(&file, expected, size + 100, 4);
            }
        }

        let nup = open(bytes).unwrap();
        let romfs = nup.romfs(0).unwrap().unwrap();
        let file = romfs.open_file("z.bin").unwrap().unwrap();
        std::thread::scope(|scope| {
            for thread in 0..4u64 {
                let file = file.clone();
                let zipped = &zipped;
                scope.spawn(move || {
                    for step in 0..50 {
                        let offset = (thread * 9 + step * 7) % 40 * BLOCK + step * 13;
                        check(&file, zipped, offset, 3 * ZSTD_TEST_BLOCK);
                    }
                });
            }
        });
    }

    #[test]
    fn damaged_zs_files_are_served_from_a_sound_sibling() {
        let sound = zstd_frame(b"a sound texture");
        let near = zstd_frame(b"the closest sound texture");
        let bytes = build_nup(&[NupPackage::new(PackageKind::Base, BASE_ID, 0)
            .file("romfs/Tex/Kart_Nrm.bntx.zs", vec![0; 4096], Codec::Zstd)
            .file("romfs/Tex/Kart2_Nrm.bntx.zs", near.clone(), Codec::Raw)
            .file("romfs/Tex/Bike_Nrm.bntx.zs", sound.clone(), Codec::Zstd)
            .file("romfs/Tex/Kart_Alb.bntx.zs", sound.clone(), Codec::Raw)
            .file("romfs/Tex/Lone.zs", vec![0; 64], Codec::Zstd)
            .file("romfs/Tex/plain.bin", vec![0; 64], Codec::Zstd)]);
        let nup = open(bytes).unwrap();
        let romfs = nup.romfs(0).unwrap().unwrap();
        assert_eq!(read(&romfs, "Tex/Kart_Nrm.bntx.zs"), near);
        assert_eq!(read(&romfs, "Tex/Bike_Nrm.bntx.zs"), sound);
        assert_eq!(read(&romfs, "Tex/Lone.zs"), [0; 64]);
        assert_eq!(read(&romfs, "Tex/plain.bin"), [0; 64]);
        nup.verify().unwrap();
    }

    fn exefs_files(package: NupPackage, main_seed: u8, rtld: bool) -> NupPackage {
        let npdm = build_npdm(&NpdmSpec {
            program_id: BASE_ID,
            layout: AcidLayout::Switch1,
            ..NpdmSpec::default()
        });
        let package = package.file("exefs/main.npdm", npdm, Codec::Zstd).file(
            "exefs/main",
            build_nso(&segments(main_seed), Compression::Lz4, 0x1000, true),
            Codec::Zstd,
        );
        if rtld {
            return package.file(
                "exefs/rtld",
                build_nso(&segments(9), Compression::None, 0, false),
                Codec::Raw,
            );
        }
        package
    }

    #[test]
    fn nup_titles_scan_and_load_with_their_newest_update_and_dlc() {
        let tmp = TempDir::new("nup");
        let path = tmp.0.join("Kart.nup");
        let base = exefs_files(NupPackage::new(PackageKind::Base, BASE_ID, 0), 1, true)
            .file("romfs/a.txt", b"base a".to_vec(), Codec::Zstd)
            .file("romfs/b.txt", b"base b".to_vec(), Codec::Raw)
            .meta(META_TITLE_NAME, "Synthetic Kart")
            .meta(META_PUBLISHER, "Lichii Tests")
            .meta(META_ICON, b"JPEG icon".to_vec())
            .localized(META_LOCALIZED_TITLES, &[("ja", "シンセカート".as_bytes())]);
        let old_update = exefs_files(
            NupPackage::new(PackageKind::Update, BASE_ID | 0x800, 0x0000_0100),
            3,
            false,
        );
        let update = exefs_files(
            NupPackage::new(PackageKind::Update, BASE_ID | 0x800, 0x0001_0000),
            2,
            false,
        )
        .file("romfs/b.txt", b"update b".to_vec(), Codec::Zstd);
        let dlc = |version: u32, data: &[u8]| {
            NupPackage::new(PackageKind::AddOn, BASE_ID + 0x1001, version).file(
                "romfs/dlc.bin",
                data.to_vec(),
                Codec::Raw,
            )
        };
        let empty_dlc = NupPackage::new(PackageKind::AddOn, BASE_ID + 0x1002, 0).file(
            "other/readme",
            b"x".to_vec(),
            Codec::Raw,
        );
        std::fs::write(
            &path,
            build_nup(&[
                dlc(2, b"dlc v2"),
                base,
                update,
                old_update,
                dlc(1, b"dlc v1"),
                empty_dlc,
            ]),
        )
        .unwrap();

        assert_eq!(crate::identify(&path, None), Some(TitleFormat::Nup));
        let info = crate::scan(&path, &LoadOptions::default()).unwrap();
        assert_eq!(
            (
                info.format,
                info.program_id,
                info.generation,
                info.name.as_str(),
                info.publisher.as_str(),
                info.version.as_str(),
            ),
            (
                TitleFormat::Nup,
                BASE_ID,
                ConsoleGeneration::Switch1,
                "Synthetic Kart",
                "Lichii Tests",
                "1.0.0",
            )
        );
        assert_eq!(info.icon.as_deref(), Some(&b"JPEG icon"[..]));
        let japanese = LoadOptions {
            language: NacpLanguage::Japanese,
            ..LoadOptions::default()
        };
        assert_eq!(crate::scan(&path, &japanese).unwrap().name, "シンセカート");

        let image = crate::load(&path, &LoadOptions::default()).unwrap();
        assert_eq!(image.program_id, BASE_ID);
        let names: Vec<_> = image.modules.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(
            names,
            ["main"],
            "the newest update's ExeFS replaces the base's"
        );
        let update_main = build_nso(&segments(2), Compression::Lz4, 0x1000, true);
        let expected = crate::nso::load_nso(
            MemFile::from_bytes(update_main).as_ref(),
            "main",
            crate::nso::HashCheck::Skip,
        )
        .unwrap();
        assert_eq!(image.modules[0].build_id, expected.build_id);
        assert_eq!(image.version, 0x0001_0000);
        let nacp = image.nacp.as_ref().unwrap();
        assert_eq!(nacp.display_version(), "1.0.0");
        assert_eq!(nacp.save_data_owner_id(), BASE_ID);
        assert_eq!(image.icon.as_deref(), Some(&b"JPEG icon"[..]));
        let romfs = open_romfs(image.romfs.as_ref().unwrap()).unwrap();
        assert_eq!(read(&romfs, "a.txt"), b"base a");
        assert_eq!(read(&romfs, "b.txt"), b"update b");
        let add_ons: Vec<_> = image
            .add_ons
            .iter()
            .map(|a| (a.title_id, a.version))
            .collect();
        assert_eq!(add_ons, [(BASE_ID + 0x1001, 2)]);
        let dlc_romfs = open_romfs(&image.add_ons[0].romfs).unwrap();
        assert_eq!(read(&dlc_romfs, "dlc.bin"), b"dlc v2");
    }

    #[test]
    fn a_standalone_update_boots_with_its_base_names() {
        let tmp = TempDir::new("nup-standalone");
        let path = tmp.0.join("Update.nup");
        let base = NupPackage::new(PackageKind::Base, BASE_ID, 0)
            .meta(META_TITLE_NAME, "Base Name")
            .file("control/readme", b"x".to_vec(), Codec::Raw);
        let mut update = exefs_files(
            NupPackage::new(PackageKind::Update, BASE_ID, 0x0002_0000),
            4,
            true,
        )
        .meta(META_TITLE_NAME, "Update Name")
        .meta(META_DISPLAY_VERSION, "2.0.0");
        update.standalone_update = true;
        std::fs::write(&path, build_nup(&[base, update])).unwrap();
        let info = crate::scan(&path, &LoadOptions::default()).unwrap();
        assert_eq!(
            (info.name.as_str(), info.version.as_str()),
            ("Base Name", "2.0.0")
        );
        let image = crate::load(&path, &LoadOptions::default()).unwrap();
        assert_eq!(image.modules.len(), 2);
        assert!(image.romfs.is_none());

        let no_program = tmp.0.join("Empty.nup");
        std::fs::write(
            &no_program,
            build_nup(&[NupPackage::new(PackageKind::Base, BASE_ID, 0).file(
                "romfs/a",
                b"a".to_vec(),
                Codec::Raw,
            )]),
        )
        .unwrap();
        assert!(matches!(
            crate::load(&no_program, &LoadOptions::default()),
            Err(LoaderError::MissingNupProgram(_))
        ));
    }
}
