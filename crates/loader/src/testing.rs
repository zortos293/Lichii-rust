//! Synthetic inputs for the tests of this crate and the crates above it: NUP containers in the
//! reference packer's layout, and NPDMs. Nothing here is real game data.

use std::collections::{BTreeMap, HashMap};

use lichii_crypto::sha256;

pub use crate::npdm::AcidLayout;
pub use crate::npdm::tests::{NpdmSpec, build_npdm, svc_mask, thread_info};
use crate::nup::{
    CODEC_RAW, CODEC_ZSTD, DIR_RECORD_SIZE, EXTENT_ALIGNMENT, FILE_RECORD_SIZE,
    FLAG_STANDALONE_UPDATE, MAGIC, NO_DIR, PACKAGE_RECORD_SIZE, PackageKind, SUPERBLOCK_SIZE,
    TOC_HEADER_SIZE, TOC_MAGIC,
};

/// Bytes per block of the zstd frames [`zstd_frame`] writes.
pub const ZSTD_TEST_BLOCK: usize = 0x1_0000;

/// How [`build_nup`] stores a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    Raw,
    /// One zstd frame from [`zstd_frame`].
    Zstd,
}

/// One package of a synthetic NUP: files by path below its root (`exefs/main`,
/// `romfs/dir/file`) and metadata records.
pub struct NupPackage {
    pub kind: PackageKind,
    pub title_id: u64,
    pub version: u32,
    pub standalone_update: bool,
    pub files: Vec<(String, Vec<u8>, Codec)>,
    pub meta: Vec<(u16, Vec<u8>)>,
}

impl NupPackage {
    pub fn new(kind: PackageKind, title_id: u64, version: u32) -> Self {
        Self {
            kind,
            title_id,
            version,
            standalone_update: false,
            files: Vec::new(),
            meta: Vec::new(),
        }
    }

    pub fn file(mut self, path: &str, data: impl Into<Vec<u8>>, codec: Codec) -> Self {
        self.files.push((path.into(), data.into(), codec));
        self
    }

    pub fn meta(mut self, kind: u16, payload: impl Into<Vec<u8>>) -> Self {
        self.meta.push((kind, payload.into()));
        self
    }

    /// A localized titles or icons record from `(tag, data)` pairs.
    pub fn localized(self, kind: u16, records: &[(&str, &[u8])]) -> Self {
        let mut payload = Vec::new();
        for (tag, data) in records {
            payload.push(tag.len() as u8);
            payload.extend(tag.as_bytes());
            payload.extend((data.len() as u32).to_le_bytes());
            payload.extend(*data);
        }
        self.meta(kind, payload)
    }
}

/// A zstd frame holding `data` in raw blocks, and RLE blocks where a block repeats one byte,
/// of [`ZSTD_TEST_BLOCK`] bytes each. It declares its content size and has no checksum, like
/// the reference packer's frames.
pub fn zstd_frame(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x28, 0xb5, 0x2f, 0xfd, 0xe0];
    out.extend((data.len() as u64).to_le_bytes());
    let blocks: Vec<&[u8]> = data.chunks(ZSTD_TEST_BLOCK).collect();
    for (index, block) in blocks.iter().enumerate() {
        let rle = block.iter().all(|&b| b == block[0]);
        let header =
            u32::from(index + 1 == blocks.len()) | u32::from(rle) << 1 | (block.len() as u32) << 3;
        out.extend(&header.to_le_bytes()[..3]);
        if rle {
            out.push(block[0]);
        } else {
            out.extend_from_slice(block);
        }
    }
    out
}

enum Child {
    Dir(TreeDir),
    File((usize, usize)),
}

#[derive(Default)]
struct TreeDir {
    name: String,
    children: BTreeMap<String, Child>,
}

const FILE_TAG: u64 = 1 << 32;

#[derive(Default)]
struct Layout {
    dirs: Vec<(String, u32, u32)>,
    files: Vec<(String, (usize, usize))>,
    children: Vec<u64>,
}

/// Directories in pre-order, each one's child slots contiguous and sorted by name, its files
/// numbered in slot order before its subdirectories are visited, as the reference packs them.
fn assign(dir: &TreeDir, layout: &mut Layout) -> u32 {
    let id = layout.dirs.len() as u32;
    let base = layout.children.len();
    layout
        .dirs
        .push((dir.name.clone(), base as u32, dir.children.len() as u32));
    layout.children.resize(base + dir.children.len(), 0);
    for (slot, (name, child)) in dir.children.iter().enumerate() {
        if let Child::File(source) = child {
            layout.children[base + slot] = FILE_TAG | layout.files.len() as u64;
            layout.files.push((name.clone(), *source));
        }
    }
    for (slot, child) in dir.children.values().enumerate() {
        if let Child::Dir(sub) = child {
            layout.children[base + slot] = u64::from(assign(sub, layout));
        }
    }
    id
}

fn put(out: &mut [u8], at: usize, bytes: &[u8]) {
    out[at..at + bytes.len()].copy_from_slice(bytes);
}

fn align(value: usize) -> usize {
    value.next_multiple_of(EXTENT_ALIGNMENT as usize)
}

/// A NUP container holding `packages`, laid out as the reference packer writes one.
pub fn build_nup(packages: &[NupPackage]) -> Vec<u8> {
    let mut layout = Layout::default();
    let mut package_dirs = Vec::new();
    for (index, package) in packages.iter().enumerate() {
        let mut root = TreeDir::default();
        for (file, (path, _, _)) in package.files.iter().enumerate() {
            let mut parts: Vec<&str> = path.split('/').collect();
            let leaf = parts.pop().unwrap();
            let mut dir = &mut root;
            for part in parts {
                let child = dir.children.entry(part.into()).or_insert_with(|| {
                    Child::Dir(TreeDir {
                        name: part.into(),
                        ..TreeDir::default()
                    })
                });
                let Child::Dir(sub) = child else {
                    panic!("{path}: a file and a folder share a name");
                };
                dir = sub;
            }
            dir.children.insert(leaf.into(), Child::File((index, file)));
        }
        let root_id = assign(&root, &mut layout);
        let first_slot = layout.dirs[root_id as usize].1 as usize;
        let named = |name: &str| {
            root.children
                .iter()
                .position(|(n, child)| n == name && matches!(child, Child::Dir(_)))
                .map_or(NO_DIR, |slot| layout.children[first_slot + slot] as u32)
        };
        package_dirs.push((root_id, named("exefs"), named("romfs")));
    }

    let mut pool = Vec::new();
    let mut interned: HashMap<String, u32> = HashMap::new();
    let names: Vec<String> = layout
        .dirs
        .iter()
        .map(|d| d.0.clone())
        .chain(layout.files.iter().map(|f| f.0.clone()))
        .collect();
    for name in names {
        interned.entry(name.clone()).or_insert_with(|| {
            let offset = pool.len() as u32;
            pool.extend(name.as_bytes());
            offset
        });
    }

    let records_end = TOC_HEADER_SIZE
        + packages.len() * PACKAGE_RECORD_SIZE
        + layout.dirs.len() * DIR_RECORD_SIZE
        + layout.files.len() * FILE_RECORD_SIZE;
    let pool_offset = records_end + layout.children.len() * 4;
    let toc_size = align(pool_offset + pool.len());
    let blob_offset = align(SUPERBLOCK_SIZE + toc_size);

    let mut out = vec![0u8; blob_offset];
    let mut extents = Vec::new();
    for (_, (package, file)) in &layout.files {
        let (_, data, codec) = &packages[*package].files[*file];
        if data.is_empty() {
            extents.push((0, 0, CODEC_RAW, sha256(data)));
            continue;
        }
        out.resize(align(out.len()), 0);
        let (stored, codec) = match codec {
            Codec::Raw => (data.clone(), CODEC_RAW),
            Codec::Zstd => (zstd_frame(data), CODEC_ZSTD),
        };
        extents.push((out.len() as u64, stored.len() as u64, codec, sha256(data)));
        out.extend(stored);
    }
    let mut meta_extents = Vec::new();
    for package in packages {
        let mut records = package.meta.clone();
        records.sort_by_key(|(kind, _)| *kind);
        let start = out.len() as u64;
        for (kind, payload) in records {
            out.extend(kind.to_le_bytes());
            out.extend([0, 0]);
            out.extend((payload.len() as u32).to_le_bytes());
            out.extend(payload);
        }
        let size = out.len() as u64 - start;
        meta_extents.push(if size == 0 {
            (0, 0)
        } else {
            (start, size as u32)
        });
    }

    let mut toc = vec![0u8; toc_size];
    put(&mut toc, 0, &TOC_MAGIC);
    put(&mut toc, 0x04, &(packages.len() as u32).to_le_bytes());
    put(&mut toc, 0x08, &(layout.dirs.len() as u64).to_le_bytes());
    put(&mut toc, 0x10, &(layout.files.len() as u64).to_le_bytes());
    put(&mut toc, 0x18, &(pool_offset as u64).to_le_bytes());
    put(&mut toc, 0x20, &(pool.len() as u64).to_le_bytes());
    let mut pos = TOC_HEADER_SIZE;
    for ((package, (root, exefs, romfs)), (meta_offset, meta_size)) in
        packages.iter().zip(&package_dirs).zip(&meta_extents)
    {
        put(&mut toc, pos, &package.title_id.to_le_bytes());
        put(&mut toc, pos + 0x08, &root.to_le_bytes());
        put(&mut toc, pos + 0x0c, &exefs.to_le_bytes());
        put(&mut toc, pos + 0x10, &romfs.to_le_bytes());
        put(&mut toc, pos + 0x14, &package.version.to_le_bytes());
        toc[pos + 0x18] = match package.kind {
            PackageKind::Base => 0,
            PackageKind::Update => 1,
            PackageKind::AddOn => 2,
        };
        toc[pos + 0x19] = if package.standalone_update {
            FLAG_STANDALONE_UPDATE
        } else {
            0
        };
        put(&mut toc, pos + 0x20, &meta_offset.to_le_bytes());
        put(&mut toc, pos + 0x28, &meta_size.to_le_bytes());
        pos += PACKAGE_RECORD_SIZE;
    }
    for (name, first_child, child_count) in &layout.dirs {
        put(&mut toc, pos, &interned[name].to_le_bytes());
        put(&mut toc, pos + 0x04, &(name.len() as u32).to_le_bytes());
        put(&mut toc, pos + 0x08, &first_child.to_le_bytes());
        put(&mut toc, pos + 0x0c, &child_count.to_le_bytes());
        pos += DIR_RECORD_SIZE;
    }
    for ((name, (package, file)), (offset, stored, codec, hash)) in
        layout.files.iter().zip(&extents)
    {
        put(&mut toc, pos, &interned[name].to_le_bytes());
        put(&mut toc, pos + 0x04, &(name.len() as u32).to_le_bytes());
        let size = packages[*package].files[*file].1.len() as u64;
        put(&mut toc, pos + 0x08, &size.to_le_bytes());
        put(&mut toc, pos + 0x10, &offset.to_le_bytes());
        put(&mut toc, pos + 0x18, &stored.to_le_bytes());
        put(&mut toc, pos + 0x20, hash);
        toc[pos + 0x40] = *codec;
        pos += FILE_RECORD_SIZE;
    }
    let dir_count = layout.dirs.len() as u64;
    for child in &layout.children {
        let id = if child & FILE_TAG != 0 {
            dir_count + (child & !FILE_TAG)
        } else {
            *child
        };
        put(&mut toc, pos, &(id as u32).to_le_bytes());
        pos += 4;
    }
    put(&mut toc, pos, &pool);

    put(&mut out, 0, &MAGIC);
    put(&mut out, 0x04, &(SUPERBLOCK_SIZE as u16).to_le_bytes());
    put(&mut out, 0x06, &1u16.to_le_bytes());
    put(&mut out, 0x08, &(SUPERBLOCK_SIZE as u64).to_le_bytes());
    put(&mut out, 0x10, &(toc_size as u64).to_le_bytes());
    put(&mut out, 0x18, &sha256(&toc));
    put(&mut out, 0x38, &(blob_offset as u64).to_le_bytes());
    let file_size = out.len() as u64;
    put(&mut out, 0x40, &file_size.to_le_bytes());
    put(&mut out, 0x48, &(packages.len() as u32).to_le_bytes());
    put(&mut out, SUPERBLOCK_SIZE, &toc);
    out
}
