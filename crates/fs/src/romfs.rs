//! RomFS: reading an image as a directory tree, and laying out a new image over any directory
//! tree (used by LayeredFS, so a modded RomFS is served without copying file data).

use std::io;
use std::sync::Arc;

use crate::bytes::{u32_at, u64_at};
use crate::error::{FsResult, corrupt};
use crate::vfs::{
    ConcatFile, DirEntry, DirRef, Directory, Entry, EntryKind, FileExt, FileRef, MemFile, ZeroFile,
    slice,
};

const FORMAT: &str = "RomFS";
const HEADER_SIZE: u64 = 0x50;
const DATA_OFFSET: u64 = 0x200;
const EMPTY: u32 = u32::MAX;
const DIR_ENTRY_SIZE: usize = 0x18;
const FILE_ENTRY_SIZE: usize = 0x20;
const MAX_META_SIZE: u64 = 512 << 20;

/// The hash the RomFS lookup tables use for a name under the directory at `parent`.
pub fn path_hash(parent: u32, name: &[u8]) -> u32 {
    name.iter().fold(parent ^ 123_456_789, |hash, &c| {
        hash.rotate_right(5) ^ u32::from(c)
    })
}

fn hash_table_len(entries: usize) -> usize {
    if entries < 3 {
        return 3;
    }
    if entries < 19 {
        return entries | 1;
    }
    let mut count = entries;
    while [2, 3, 5, 7, 11, 13, 17]
        .iter()
        .any(|p| count.is_multiple_of(*p))
    {
        count += 1;
    }
    count
}

struct Image {
    data: FileRef,
    dir_hash: Vec<u32>,
    dir_meta: Vec<u8>,
    file_hash: Vec<u32>,
    file_meta: Vec<u8>,
}

struct DirRecord<'a> {
    sibling: u32,
    child_dir: u32,
    child_file: u32,
    parent: u32,
    hash_next: u32,
    name: &'a [u8],
}

struct FileRecord<'a> {
    parent: u32,
    sibling: u32,
    offset: u64,
    size: u64,
    hash_next: u32,
    name: &'a [u8],
}

fn name_at(meta: &[u8], offset: usize, fixed: usize) -> io::Result<&[u8]> {
    let len = u32_at(meta, offset + fixed - 4) as usize;
    meta.get(offset + fixed..offset + fixed + len)
        .ok_or_else(|| corrupt(FORMAT, "name past the metadata table").into())
}

impl Image {
    fn dir(&self, offset: u32) -> io::Result<DirRecord<'_>> {
        let o = offset as usize;
        if o.checked_add(DIR_ENTRY_SIZE)
            .is_none_or(|end| end > self.dir_meta.len())
        {
            return Err(corrupt(FORMAT, "directory entry past the table").into());
        }
        let m = &self.dir_meta;
        Ok(DirRecord {
            parent: u32_at(m, o),
            sibling: u32_at(m, o + 4),
            child_dir: u32_at(m, o + 8),
            child_file: u32_at(m, o + 12),
            hash_next: u32_at(m, o + 16),
            name: name_at(m, o, DIR_ENTRY_SIZE)?,
        })
    }

    fn file(&self, offset: u32) -> io::Result<FileRecord<'_>> {
        let o = offset as usize;
        if o.checked_add(FILE_ENTRY_SIZE)
            .is_none_or(|end| end > self.file_meta.len())
        {
            return Err(corrupt(FORMAT, "file entry past the table").into());
        }
        let m = &self.file_meta;
        Ok(FileRecord {
            parent: u32_at(m, o),
            sibling: u32_at(m, o + 4),
            offset: u64_at(m, o + 8),
            size: u64_at(m, o + 16),
            hash_next: u32_at(m, o + 24),
            name: name_at(m, o, FILE_ENTRY_SIZE)?,
        })
    }

    fn open_file(&self, record: &FileRecord<'_>) -> io::Result<FileRef> {
        if record
            .offset
            .checked_add(record.size)
            .is_none_or(|end| end > self.data.size())
        {
            return Err(corrupt(FORMAT, "file data past the end of the image").into());
        }
        Ok(slice(&self.data, record.offset, record.size))
    }

    fn max_steps(&self) -> usize {
        self.dir_meta.len() / DIR_ENTRY_SIZE + self.file_meta.len() / FILE_ENTRY_SIZE + 1
    }
}

/// A directory inside a RomFS image.
pub struct RomFsDir {
    image: Arc<Image>,
    offset: u32,
}

/// Opens a RomFS image and returns its root directory. Only the metadata tables are read;
/// file data stays in `file`.
pub fn open_romfs(file: &FileRef) -> FsResult<DirRef> {
    let header: [u8; HEADER_SIZE as usize] = file.read_array(0)?;
    let field = |index: usize| u64_at(&header, index * 8);
    if field(0) != HEADER_SIZE {
        return Err(corrupt(FORMAT, format!("header size {:#x}", field(0))));
    }
    let table = |index: usize| -> FsResult<Vec<u8>> {
        let (offset, size) = (field(index), field(index + 1));
        if size > MAX_META_SIZE || offset.checked_add(size).is_none_or(|end| end > file.size()) {
            return Err(corrupt(FORMAT, "metadata table past the end of the image"));
        }
        Ok(file.read_vec(offset, size as usize)?)
    };
    let words = |bytes: Vec<u8>| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    };
    let data_offset = field(9);
    if data_offset > file.size() {
        return Err(corrupt(FORMAT, "data offset past the end of the image"));
    }
    let image = Image {
        data: slice(file, data_offset, file.size() - data_offset),
        dir_hash: words(table(1)?),
        dir_meta: table(3)?,
        file_hash: words(table(5)?),
        file_meta: table(7)?,
    };
    image.dir(0)?;
    Ok(Arc::new(RomFsDir {
        image: Arc::new(image),
        offset: 0,
    }))
}

impl RomFsDir {
    fn find_dir(&self, name: &[u8]) -> io::Result<Option<u32>> {
        let image = &self.image;
        if image.dir_hash.is_empty() {
            return self.walk_dirs(|record| record.name == name);
        }
        let bucket = path_hash(self.offset, name) as usize % image.dir_hash.len();
        let mut next = image.dir_hash[bucket];
        for _ in 0..image.max_steps() {
            if next == EMPTY {
                return Ok(None);
            }
            let record = image.dir(next)?;
            if record.parent == self.offset && record.name == name {
                return Ok(Some(next));
            }
            next = record.hash_next;
        }
        Err(corrupt(FORMAT, "directory hash chain loops").into())
    }

    fn find_file(&self, name: &[u8]) -> io::Result<Option<u32>> {
        let image = &self.image;
        if image.file_hash.is_empty() {
            return self.walk_files(|record| record.name == name);
        }
        let bucket = path_hash(self.offset, name) as usize % image.file_hash.len();
        let mut next = image.file_hash[bucket];
        for _ in 0..image.max_steps() {
            if next == EMPTY {
                return Ok(None);
            }
            let record = image.file(next)?;
            if record.parent == self.offset && record.name == name {
                return Ok(Some(next));
            }
            next = record.hash_next;
        }
        Err(corrupt(FORMAT, "file hash chain loops").into())
    }

    fn walk_dirs(&self, mut stop: impl FnMut(&DirRecord<'_>) -> bool) -> io::Result<Option<u32>> {
        let mut next = self.image.dir(self.offset)?.child_dir;
        for _ in 0..self.image.max_steps() {
            if next == EMPTY {
                return Ok(None);
            }
            let record = self.image.dir(next)?;
            if stop(&record) {
                return Ok(Some(next));
            }
            next = record.sibling;
        }
        Err(corrupt(FORMAT, "directory sibling chain loops").into())
    }

    fn walk_files(&self, mut stop: impl FnMut(&FileRecord<'_>) -> bool) -> io::Result<Option<u32>> {
        let mut next = self.image.dir(self.offset)?.child_file;
        for _ in 0..self.image.max_steps() {
            if next == EMPTY {
                return Ok(None);
            }
            let record = self.image.file(next)?;
            if stop(&record) {
                return Ok(Some(next));
            }
            next = record.sibling;
        }
        Err(corrupt(FORMAT, "file sibling chain loops").into())
    }
}

impl Directory for RomFsDir {
    fn entries(&self) -> io::Result<Vec<DirEntry>> {
        let mut out = Vec::new();
        let name = |bytes: &[u8]| String::from_utf8_lossy(bytes).into_owned();
        self.walk_dirs(|record| {
            out.push(DirEntry {
                name: name(record.name),
                kind: EntryKind::Directory,
                size: 0,
            });
            false
        })?;
        self.walk_files(|record| {
            out.push(DirEntry {
                name: name(record.name),
                kind: EntryKind::File,
                size: record.size,
            });
            false
        })?;
        Ok(out)
    }

    fn open(&self, name: &str) -> io::Result<Option<Entry>> {
        if let Some(offset) = self.find_dir(name.as_bytes())? {
            return Ok(Some(Entry::Dir(Arc::new(RomFsDir {
                image: self.image.clone(),
                offset,
            }))));
        }
        match self.find_file(name.as_bytes())? {
            Some(offset) => Ok(Some(Entry::File(
                self.image.open_file(&self.image.file(offset)?)?,
            ))),
            None => Ok(None),
        }
    }
}

struct BuildDir {
    name: Vec<u8>,
    parent: usize,
    dirs: Vec<usize>,
    files: Vec<usize>,
    offset: u32,
}

struct BuildFile {
    name: Vec<u8>,
    parent: usize,
    file: FileRef,
    offset: u32,
    data_offset: u64,
}

fn entry_size(fixed: usize, name: &[u8]) -> usize {
    fixed + name.len().next_multiple_of(4)
}

/// Lays out a RomFS image over `root`. Only the header and metadata tables are built in
/// memory; file contents are referenced, not copied.
pub fn build_romfs(root: &dyn Directory) -> io::Result<FileRef> {
    let mut dirs = vec![BuildDir {
        name: Vec::new(),
        parent: 0,
        dirs: Vec::new(),
        files: Vec::new(),
        offset: 0,
    }];
    let mut files: Vec<BuildFile> = Vec::new();
    let mut pending: Vec<(usize, DirRef)> = Vec::new();
    let mut queue = std::collections::VecDeque::new();
    queue.push_back((0usize, None::<DirRef>));
    while let Some((index, handle)) = queue.pop_front() {
        let dir: &dyn Directory = match &handle {
            Some(dir) => dir.as_ref(),
            None => root,
        };
        let mut entries = dir.entries()?;
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        for entry in entries {
            match dir.open(&entry.name)? {
                Some(Entry::Dir(child)) => {
                    let child_index = dirs.len();
                    dirs.push(BuildDir {
                        name: entry.name.into_bytes(),
                        parent: index,
                        dirs: Vec::new(),
                        files: Vec::new(),
                        offset: 0,
                    });
                    dirs[index].dirs.push(child_index);
                    pending.push((child_index, child));
                }
                Some(Entry::File(file)) => {
                    dirs[index].files.push(files.len());
                    files.push(BuildFile {
                        name: entry.name.into_bytes(),
                        parent: index,
                        file,
                        offset: 0,
                        data_offset: 0,
                    });
                }
                None => {}
            }
        }
        queue.extend(pending.drain(..).map(|(i, d)| (i, Some(d))));
    }

    let mut dir_meta_size = 0usize;
    for dir in &mut dirs {
        dir.offset = dir_meta_size as u32;
        dir_meta_size += entry_size(DIR_ENTRY_SIZE, &dir.name);
    }
    let mut file_meta_size = 0usize;
    let mut data_size = 0u64;
    let mut data_parts: Vec<FileRef> = Vec::new();
    for file in &mut files {
        file.offset = file_meta_size as u32;
        file_meta_size += entry_size(FILE_ENTRY_SIZE, &file.name);
        let aligned = data_size.next_multiple_of(0x10);
        if aligned > data_size {
            data_parts.push(Arc::new(ZeroFile(aligned - data_size)));
        }
        file.data_offset = aligned;
        data_size = aligned + file.file.size();
        data_parts.push(file.file.clone());
    }

    let mut dir_siblings = vec![EMPTY; dirs.len()];
    let mut file_siblings = vec![EMPTY; files.len()];
    for dir in &dirs {
        for pair in dir.dirs.windows(2) {
            dir_siblings[pair[0]] = dirs[pair[1]].offset;
        }
        for pair in dir.files.windows(2) {
            file_siblings[pair[0]] = files[pair[1]].offset;
        }
    }
    let mut dir_hash = vec![EMPTY; hash_table_len(dirs.len())];
    let mut file_hash = vec![EMPTY; hash_table_len(files.len())];
    let mut dir_meta = Vec::with_capacity(dir_meta_size);
    for (index, dir) in dirs.iter().enumerate() {
        let parent = dirs[dir.parent].offset;
        let bucket = path_hash(parent, &dir.name) as usize % dir_hash.len();
        let hash_next = std::mem::replace(&mut dir_hash[bucket], dir.offset);
        for word in [
            parent,
            dir_siblings[index],
            dir.dirs.first().map_or(EMPTY, |&d| dirs[d].offset),
            dir.files.first().map_or(EMPTY, |&f| files[f].offset),
            hash_next,
            dir.name.len() as u32,
        ] {
            dir_meta.extend(word.to_le_bytes());
        }
        dir_meta.extend(&dir.name);
        dir_meta.resize(
            dir.offset as usize + entry_size(DIR_ENTRY_SIZE, &dir.name),
            0,
        );
    }
    let mut file_meta = Vec::with_capacity(file_meta_size);
    for (index, file) in files.iter().enumerate() {
        let parent = dirs[file.parent].offset;
        let bucket = path_hash(parent, &file.name) as usize % file_hash.len();
        let hash_next = std::mem::replace(&mut file_hash[bucket], file.offset);
        file_meta.extend(parent.to_le_bytes());
        file_meta.extend(file_siblings[index].to_le_bytes());
        file_meta.extend(file.data_offset.to_le_bytes());
        file_meta.extend(file.file.size().to_le_bytes());
        file_meta.extend(hash_next.to_le_bytes());
        file_meta.extend((file.name.len() as u32).to_le_bytes());
        file_meta.extend(&file.name);
        file_meta.resize(
            file.offset as usize + entry_size(FILE_ENTRY_SIZE, &file.name),
            0,
        );
    }

    let meta_offset = (DATA_OFFSET + data_size).next_multiple_of(4);
    let mut meta = Vec::new();
    let mut fields = vec![HEADER_SIZE];
    for table in [
        dir_hash
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
        dir_meta,
        file_hash.iter().flat_map(|w| w.to_le_bytes()).collect(),
        file_meta,
    ] {
        fields.push(meta_offset + meta.len() as u64);
        fields.push(table.len() as u64);
        meta.extend(table);
    }
    fields.push(DATA_OFFSET);
    let mut header: Vec<u8> = fields.iter().flat_map(|f| f.to_le_bytes()).collect();
    header.resize(DATA_OFFSET as usize, 0);

    let parts = std::iter::once(MemFile::from_bytes(header))
        .chain(data_parts)
        .chain(std::iter::once(
            Arc::new(ZeroFile(meta_offset - DATA_OFFSET - data_size)) as FileRef,
        ))
        .chain(std::iter::once(MemFile::from_bytes(meta)));
    Ok(ConcatFile::from_parts(parts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::{DirectoryExt, LayeredDir, MemDir};

    fn sample() -> MemDir {
        let mut many = MemDir::new();
        for i in 0..40u8 {
            many = many.with_bytes(format!("file{i:02}.bin"), vec![i; usize::from(i) * 3 + 1]);
        }
        MemDir::new()
            .with_bytes("root.txt", b"hello romfs".to_vec())
            .with_bytes("empty", Vec::new())
            .with_dir(
                "data",
                MemDir::new()
                    .with_dir("many", many)
                    .with_dir("emptydir", MemDir::new())
                    .with_bytes("odd name.dat", vec![9; 17]),
            )
    }

    fn assert_same_tree(a: &dyn Directory, b: &dyn Directory) {
        let mut left = a.entries().unwrap();
        let mut right = b.entries().unwrap();
        left.sort_by(|x, y| x.name.cmp(&y.name));
        right.sort_by(|x, y| x.name.cmp(&y.name));
        assert_eq!(left, right);
        for entry in left {
            match (
                a.open(&entry.name).unwrap().unwrap(),
                b.open(&entry.name).unwrap().unwrap(),
            ) {
                (Entry::File(x), Entry::File(y)) => {
                    assert_eq!(
                        x.read_all().unwrap(),
                        y.read_all().unwrap(),
                        "{}",
                        entry.name
                    )
                }
                (Entry::Dir(x), Entry::Dir(y)) => assert_same_tree(x.as_ref(), y.as_ref()),
                _ => panic!("{} changed kind", entry.name),
            }
        }
    }

    #[test]
    fn round_trips_a_tree() {
        let source = sample();
        let image = build_romfs(&source).unwrap();
        let root = open_romfs(&image).unwrap();
        assert_same_tree(&source, root.as_ref());
        let file = root
            .open_file_path("data/many/file39.bin")
            .unwrap()
            .unwrap();
        assert_eq!(file.read_all().unwrap(), vec![39; 118]);
        assert!(root.open_path("data/many/nope").unwrap().is_none());
        assert!(root.open_path("root.txt/x").unwrap().is_none());

        let bytes = image.read_all().unwrap();
        assert_eq!(bytes.len() as u64, image.size());
        let reparsed = open_romfs(&MemFile::from_bytes(bytes)).unwrap();
        assert_same_tree(&source, reparsed.as_ref());
    }

    #[test]
    fn layered_mods_rebuild_into_an_image() {
        let base = build_romfs(&sample()).unwrap();
        let base_root = open_romfs(&base).unwrap();
        let mods = MemDir::new()
            .with_bytes("root.txt", b"modded".to_vec())
            .with_dir("data", MemDir::new().with_bytes("new.bin", vec![1, 2, 3]))
            .into_ref();
        let layered = LayeredDir::from_layers(vec![mods, base_root]);
        let rebuilt = open_romfs(&build_romfs(layered.as_ref()).unwrap()).unwrap();
        assert_same_tree(layered.as_ref(), rebuilt.as_ref());
        let read = |path: &str| {
            rebuilt
                .open_file_path(path)
                .unwrap()
                .unwrap()
                .read_all()
                .unwrap()
        };
        assert_eq!(read("root.txt"), b"modded");
        assert_eq!(read("data/new.bin"), [1, 2, 3]);
        assert_eq!(read("data/odd name.dat"), [9; 17]);
    }

    #[test]
    fn lookups_work_without_hash_tables() {
        let image = build_romfs(&sample()).unwrap().read_all().unwrap();
        let mut stripped = image.clone();
        for size_field in [2usize, 6] {
            stripped[size_field * 8..size_field * 8 + 8].fill(0);
        }
        let root = open_romfs(&MemFile::from_bytes(stripped)).unwrap();
        assert_same_tree(&sample(), root.as_ref());
    }

    #[test]
    fn rejects_corrupt_images() {
        let image = build_romfs(&sample()).unwrap().read_all().unwrap();
        let mut bad_header = image.clone();
        bad_header[0] = 0x40;
        assert!(open_romfs(&MemFile::from_bytes(bad_header)).is_err());
        let mut huge_table = image.clone();
        huge_table[0x38..0x40].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(open_romfs(&MemFile::from_bytes(huge_table)).is_err());
        assert!(open_romfs(&MemFile::from_bytes(image[..0x40].to_vec())).is_err());

        let header = &image[..0x50];
        let file_meta = u64::from_le_bytes(header[0x38..0x40].try_into().unwrap()) as usize;
        let mut looping = image.clone();
        looping[file_meta + 4..file_meta + 8].copy_from_slice(&0u32.to_le_bytes());
        let root = open_romfs(&MemFile::from_bytes(looping)).unwrap();
        let _ = root.entries();
    }

    #[test]
    fn hash_matches_reference_values() {
        assert_eq!(path_hash(0, b""), 123_456_789);
        assert_eq!(path_hash(0, b"abc"), 0x926a0ed7);
        assert_eq!(path_hash(5, b"main"), 0xec4b0018);
        assert_eq!(hash_table_len(0), 3);
        assert_eq!(hash_table_len(10), 11);
        assert_eq!(hash_table_len(19), 19);
        assert_eq!(hash_table_len(20), 23);
    }
}
