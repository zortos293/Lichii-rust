//! Random-access files and directories. Containers hand out cheap views into their backing file
//! instead of copying, so a multi-gigabyte image is never read whole.

use std::collections::BTreeMap;
use std::io;
use std::sync::Arc;

/// A random-access file. Reads are positional and take `&self`, so one file can serve many
/// threads without a seek lock. Writes are optional; read-only files return
/// [`io::ErrorKind::PermissionDenied`].
pub trait File: Send + Sync {
    fn size(&self) -> u64;

    /// Reads up to `buf.len()` bytes at `offset`. A short count only happens at end of file.
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize>;

    fn write_at(&self, _offset: u64, _data: &[u8]) -> io::Result<usize> {
        Err(read_only())
    }

    fn set_size(&self, _size: u64) -> io::Result<()> {
        Err(read_only())
    }

    fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    fn is_writable(&self) -> bool {
        false
    }

    /// When this file is a window into another file: that file and the window's start. Lets
    /// [`slice`] collapse nested views into one.
    fn window(&self) -> Option<(&FileRef, u64)> {
        None
    }
}

pub type FileRef = Arc<dyn File>;

pub fn read_only() -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, "read-only file system")
}

pub fn not_found(name: &str) -> io::Error {
    io::Error::new(io::ErrorKind::NotFound, format!("{name}: not found"))
}

/// Convenience reads on any [`File`].
pub trait FileExt: File {
    /// Fills `buf` from `offset`, failing with `UnexpectedEof` past the end.
    fn read_exact_at(&self, mut offset: u64, mut buf: &mut [u8]) -> io::Result<()> {
        while !buf.is_empty() {
            match self.read_at(offset, buf)? {
                0 => return Err(io::ErrorKind::UnexpectedEof.into()),
                n => {
                    offset += n as u64;
                    buf = &mut buf[n..];
                }
            }
        }
        Ok(())
    }

    /// Reads `len` bytes at `offset` into a new buffer.
    fn read_vec(&self, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        let mut out = vec![0; len];
        self.read_exact_at(offset, &mut out)?;
        Ok(out)
    }

    fn read_array<const N: usize>(&self, offset: u64) -> io::Result<[u8; N]> {
        let mut out = [0; N];
        self.read_exact_at(offset, &mut out)?;
        Ok(out)
    }

    /// Reads the whole file. Only for files known to be small (metadata, executables).
    fn read_all(&self) -> io::Result<Vec<u8>> {
        let len = usize::try_from(self.size())
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        self.read_vec(0, len)
    }
}

impl<T: File + ?Sized> FileExt for T {}

/// A window `[offset, offset + size)` into `file`. Windows of windows collapse into one.
pub fn slice(file: &FileRef, offset: u64, size: u64) -> FileRef {
    let (base, start) = match file.window() {
        Some((inner, start)) => (inner.clone(), start + offset),
        None => (file.clone(), offset),
    };
    Arc::new(OffsetFile {
        inner: base,
        offset: start,
        size,
    })
}

struct OffsetFile {
    inner: FileRef,
    offset: u64,
    size: u64,
}

impl File for OffsetFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        if offset >= self.size {
            return Ok(0);
        }
        let len = buf.len().min((self.size - offset) as usize);
        self.inner.read_at(self.offset + offset, &mut buf[..len])
    }

    fn window(&self) -> Option<(&FileRef, u64)> {
        Some((&self.inner, self.offset))
    }
}

/// A file held in memory.
pub struct MemFile(Arc<[u8]>);

impl MemFile {
    pub fn from_bytes(data: impl Into<Arc<[u8]>>) -> FileRef {
        Arc::new(Self(data.into()))
    }
}

impl File for MemFile {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let Some(rest) = usize::try_from(offset).ok().and_then(|o| self.0.get(o..)) else {
            return Ok(0);
        };
        let len = buf.len().min(rest.len());
        buf[..len].copy_from_slice(&rest[..len]);
        Ok(len)
    }
}

/// `size` zero bytes, used for padding in synthesized images.
pub struct ZeroFile(pub u64);

impl File for ZeroFile {
    fn size(&self) -> u64 {
        self.0
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len().min(self.0.saturating_sub(offset) as usize);
        buf[..len].fill(0);
        Ok(len)
    }
}

/// Several files read back to back as one.
pub struct ConcatFile {
    starts: Vec<u64>,
    parts: Vec<FileRef>,
    size: u64,
}

impl ConcatFile {
    pub fn from_parts(parts: impl IntoIterator<Item = FileRef>) -> FileRef {
        let mut starts = Vec::new();
        let mut kept = Vec::new();
        let mut size = 0;
        for part in parts {
            if part.size() == 0 {
                continue;
            }
            starts.push(size);
            size += part.size();
            kept.push(part);
        }
        Arc::new(Self {
            starts,
            parts: kept,
            size,
        })
    }
}

impl File for ConcatFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, mut offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut done = 0;
        let mut index = self
            .starts
            .partition_point(|&start| start <= offset)
            .saturating_sub(1);
        while done < buf.len() && index < self.parts.len() && offset < self.size {
            let part = &self.parts[index];
            let read = part.read_at(offset - self.starts[index], &mut buf[done..])?;
            if read == 0 {
                index += 1;
                continue;
            }
            done += read;
            offset += read as u64;
            if offset >= self.starts[index] + part.size() {
                index += 1;
            }
        }
        Ok(done)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Directory,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirEntry {
    pub name: String,
    pub kind: EntryKind,
    /// File size; zero for directories.
    pub size: u64,
}

#[derive(Clone)]
pub enum Entry {
    File(FileRef),
    Dir(DirRef),
}

impl Entry {
    pub fn kind(&self) -> EntryKind {
        match self {
            Self::File(_) => EntryKind::File,
            Self::Dir(_) => EntryKind::Directory,
        }
    }
}

/// A directory. Names are single path components; lookups are case-sensitive. Mutations are
/// optional and fail with [`io::ErrorKind::PermissionDenied`] on read-only containers.
pub trait Directory: Send + Sync {
    fn entries(&self) -> io::Result<Vec<DirEntry>>;

    fn open(&self, name: &str) -> io::Result<Option<Entry>>;

    fn create_file(&self, _name: &str, _size: u64) -> io::Result<FileRef> {
        Err(read_only())
    }

    fn create_dir(&self, _name: &str) -> io::Result<DirRef> {
        Err(read_only())
    }

    /// Removes a file or an empty directory.
    fn remove(&self, _name: &str) -> io::Result<()> {
        Err(read_only())
    }

    /// Removes a directory and everything below it.
    fn remove_all(&self, _name: &str) -> io::Result<()> {
        Err(read_only())
    }

    fn rename(&self, _from: &str, _to: &str) -> io::Result<()> {
        Err(read_only())
    }
}

pub type DirRef = Arc<dyn Directory>;

/// Path helpers on any [`Directory`]. Paths use `/`; empty components and `.` are skipped.
pub trait DirectoryExt: Directory {
    fn open_file(&self, name: &str) -> io::Result<Option<FileRef>> {
        Ok(match self.open(name)? {
            Some(Entry::File(file)) => Some(file),
            _ => None,
        })
    }

    fn open_dir(&self, name: &str) -> io::Result<Option<DirRef>> {
        Ok(match self.open(name)? {
            Some(Entry::Dir(dir)) => Some(dir),
            _ => None,
        })
    }

    fn open_path(&self, path: &str) -> io::Result<Option<Entry>> {
        let mut parts = path.split('/').filter(|p| !p.is_empty() && *p != ".");
        let Some(first) = parts.next() else {
            return Ok(None);
        };
        let mut entry = self.open(first)?;
        for part in parts {
            entry = match entry {
                Some(Entry::Dir(dir)) => dir.open(part)?,
                _ => return Ok(None),
            };
        }
        Ok(entry)
    }

    fn open_file_path(&self, path: &str) -> io::Result<Option<FileRef>> {
        Ok(match self.open_path(path)? {
            Some(Entry::File(file)) => Some(file),
            _ => None,
        })
    }

    fn open_dir_path(&self, path: &str) -> io::Result<Option<DirRef>> {
        Ok(match self.open_path(path)? {
            Some(Entry::Dir(dir)) => Some(dir),
            _ => None,
        })
    }
}

impl<T: Directory + ?Sized> DirectoryExt for T {}

/// A read-only directory assembled in memory, for synthesized trees and tests.
#[derive(Clone, Default)]
pub struct MemDir {
    entries: BTreeMap<String, Entry>,
}

impl MemDir {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, name: impl Into<String>, entry: Entry) -> &mut Self {
        self.entries.insert(name.into(), entry);
        self
    }

    pub fn with_file(mut self, name: impl Into<String>, file: FileRef) -> Self {
        self.insert(name, Entry::File(file));
        self
    }

    pub fn with_bytes(self, name: impl Into<String>, data: impl Into<Arc<[u8]>>) -> Self {
        self.with_file(name, MemFile::from_bytes(data))
    }

    pub fn with_dir(mut self, name: impl Into<String>, dir: impl Directory + 'static) -> Self {
        self.insert(name, Entry::Dir(Arc::new(dir)));
        self
    }

    pub fn into_ref(self) -> DirRef {
        Arc::new(self)
    }
}

impl Directory for MemDir {
    fn entries(&self) -> io::Result<Vec<DirEntry>> {
        Ok(self
            .entries
            .iter()
            .map(|(name, entry)| DirEntry {
                name: name.clone(),
                kind: entry.kind(),
                size: match entry {
                    Entry::File(file) => file.size(),
                    Entry::Dir(_) => 0,
                },
            })
            .collect())
    }

    fn open(&self, name: &str) -> io::Result<Option<Entry>> {
        Ok(self.entries.get(name).cloned())
    }
}

/// Directories stacked on top of each other: a name resolves to the first layer that has it,
/// and directories with the same name in several layers merge. This is how LayeredFS puts mod
/// files over a game's RomFS.
pub struct LayeredDir {
    layers: Vec<DirRef>,
}

impl LayeredDir {
    /// `layers[0]` has the highest priority.
    pub fn from_layers(layers: Vec<DirRef>) -> DirRef {
        if layers.len() == 1 {
            return layers.into_iter().next().unwrap();
        }
        Arc::new(Self { layers })
    }
}

impl Directory for LayeredDir {
    fn entries(&self) -> io::Result<Vec<DirEntry>> {
        let mut merged = BTreeMap::new();
        for layer in &self.layers {
            for entry in layer.entries()? {
                merged.entry(entry.name.clone()).or_insert(entry);
            }
        }
        Ok(merged.into_values().collect())
    }

    fn open(&self, name: &str) -> io::Result<Option<Entry>> {
        let mut dirs = Vec::new();
        for layer in &self.layers {
            match layer.open(name)? {
                Some(Entry::File(file)) if dirs.is_empty() => return Ok(Some(Entry::File(file))),
                Some(Entry::Dir(dir)) => dirs.push(dir),
                _ => {}
            }
        }
        Ok((!dirs.is_empty()).then(|| Entry::Dir(LayeredDir::from_layers(dirs))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(range: std::ops::Range<u8>) -> Vec<u8> {
        range.collect()
    }

    #[test]
    fn slices_collapse_and_clamp() {
        let base = MemFile::from_bytes(bytes(0..100));
        let outer = slice(&base, 10, 50);
        let inner = slice(&outer, 5, 10);
        let (root, start) = inner.window().unwrap();
        assert_eq!(start, 15);
        assert_eq!(root.size(), 100);
        assert_eq!(inner.read_vec(0, 10).unwrap(), bytes(15..25));
        let mut buf = [0u8; 20];
        assert_eq!(inner.read_at(4, &mut buf).unwrap(), 6);
        assert_eq!(inner.read_at(10, &mut buf).unwrap(), 0);
        assert!(inner.read_vec(5, 6).is_err());
    }

    #[test]
    fn concat_reads_across_parts() {
        let file = ConcatFile::from_parts([
            MemFile::from_bytes(bytes(0..10)),
            Arc::new(ZeroFile(0)) as FileRef,
            Arc::new(ZeroFile(5)),
            MemFile::from_bytes(bytes(20..30)),
        ]);
        assert_eq!(file.size(), 25);
        let mut expected = bytes(0..10);
        expected.extend([0; 5]);
        expected.extend(bytes(20..30));
        assert_eq!(file.read_all().unwrap(), expected);
        for start in 0..25 {
            for len in 0..=(25 - start) {
                assert_eq!(
                    file.read_vec(start as u64, len).unwrap(),
                    expected[start..start + len]
                );
            }
        }
        let mut buf = [0u8; 4];
        assert_eq!(file.read_at(23, &mut buf).unwrap(), 2);
    }

    #[test]
    fn layered_dirs_prefer_top_and_merge_dirs() {
        let base = MemDir::new()
            .with_bytes("a", vec![1])
            .with_bytes("b", vec![2])
            .with_dir(
                "sub",
                MemDir::new()
                    .with_bytes("x", vec![3])
                    .with_bytes("y", vec![4]),
            )
            .into_ref();
        let top = MemDir::new()
            .with_bytes("b", vec![20])
            .with_dir("sub", MemDir::new().with_bytes("y", vec![40]))
            .into_ref();
        let layered = LayeredDir::from_layers(vec![top, base]);
        let names: Vec<_> = layered
            .entries()
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, ["a", "b", "sub"]);
        let read = |path: &str| {
            layered
                .open_file_path(path)
                .unwrap()
                .unwrap()
                .read_all()
                .unwrap()
        };
        assert_eq!(read("a"), [1]);
        assert_eq!(read("b"), [20]);
        assert_eq!(read("sub/x"), [3]);
        assert_eq!(read("/sub//y"), [40]);
        assert!(layered.open_path("sub/z").unwrap().is_none());
        assert!(layered.open_path("a/z").unwrap().is_none());
        assert!(layered.create_dir("new").is_err());
    }
}
