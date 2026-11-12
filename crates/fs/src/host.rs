//! Host directories and files: the backing for NAND, SD card and save data, and for loose
//! title directories on disk.

use std::collections::VecDeque;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::vfs::{DirEntry, DirRef, Directory, Entry, EntryKind, File, FileRef, read_only};

/// A host file read and written at explicit offsets, without a shared cursor.
pub struct HostFile {
    file: std::fs::File,
    size: AtomicU64,
    writable: bool,
}

impl HostFile {
    pub fn open(path: &Path) -> io::Result<FileRef> {
        Self::open_with(path, false)
    }

    pub fn open_writable(path: &Path) -> io::Result<FileRef> {
        Self::open_with(path, true)
    }

    fn open_with(path: &Path, writable: bool) -> io::Result<FileRef> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(writable)
            .open(path)?;
        let size = file.metadata()?.len();
        Ok(Arc::new(Self {
            file,
            size: AtomicU64::new(size),
            writable,
        }))
    }
}

#[cfg(unix)]
fn pread(file: &std::fs::File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buf, offset)
}

#[cfg(unix)]
fn pwrite(file: &std::fs::File, offset: u64, data: &[u8]) -> io::Result<usize> {
    std::os::unix::fs::FileExt::write_at(file, data, offset)
}

#[cfg(windows)]
fn pread(file: &std::fs::File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buf, offset)
}

#[cfg(windows)]
fn pwrite(file: &std::fs::File, offset: u64, data: &[u8]) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_write(file, data, offset)
}

impl File for HostFile {
    fn size(&self) -> u64 {
        self.size.load(Ordering::Relaxed)
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut done = 0;
        while done < buf.len() {
            match pread(&self.file, offset + done as u64, &mut buf[done..]) {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(done)
    }

    fn write_at(&self, offset: u64, data: &[u8]) -> io::Result<usize> {
        if !self.writable {
            return Err(read_only());
        }
        let mut done = 0;
        while done < data.len() {
            match pwrite(&self.file, offset + done as u64, &data[done..]) {
                Ok(0) => return Err(io::ErrorKind::WriteZero.into()),
                Ok(n) => done += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        self.size.fetch_max(offset + done as u64, Ordering::Relaxed);
        Ok(done)
    }

    fn set_size(&self, size: u64) -> io::Result<()> {
        if !self.writable {
            return Err(read_only());
        }
        self.file.set_len(size)?;
        self.size.store(size, Ordering::Relaxed);
        Ok(())
    }

    fn flush(&self) -> io::Result<()> {
        if self.writable {
            self.file.sync_data()?;
        }
        Ok(())
    }

    fn is_writable(&self) -> bool {
        self.writable
    }
}

const OPEN_HANDLE_LIMIT: usize = 256;

/// Open handles of the read-only files below one [`HostDir`] root, most recently used first.
/// A loose RomFS can hold hundreds of thousands of files; they are opened on first read and the
/// oldest handle closes once [`OPEN_HANDLE_LIMIT`] are open, so the process never runs out.
#[derive(Default)]
struct HandleCache {
    next_id: AtomicU64,
    open: Mutex<VecDeque<(u64, Arc<std::fs::File>)>>,
}

impl HandleCache {
    fn get(&self, id: u64, path: &Path) -> io::Result<Arc<std::fs::File>> {
        let mut open = self.open.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(index) = open.iter().position(|(key, _)| *key == id) {
            let entry = open.remove(index).unwrap();
            let file = entry.1.clone();
            open.push_front(entry);
            return Ok(file);
        }
        let file = Arc::new(std::fs::File::open(path)?);
        if open.len() == OPEN_HANDLE_LIMIT {
            open.pop_back();
        }
        open.push_front((id, file.clone()));
        Ok(file)
    }
}

/// A read-only host file that holds a path, not a handle, until it is read.
struct LazyHostFile {
    id: u64,
    path: PathBuf,
    size: u64,
    cache: Arc<HandleCache>,
}

impl File for LazyHostFile {
    fn size(&self) -> u64 {
        self.size
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len().min(self.size.saturating_sub(offset) as usize);
        let file = self.cache.get(self.id, &self.path)?;
        let mut done = 0;
        while done < len {
            match pread(&file, offset + done as u64, &mut buf[done..len]) {
                Ok(0) => break,
                Ok(n) => done += n,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        Ok(done)
    }
}

/// A host directory. Names are checked to be single components, so a guest path can never
/// escape the root. Files of a read-only directory are opened lazily (see [`HandleCache`]);
/// files of a writable one are opened when looked up.
pub struct HostDir {
    path: PathBuf,
    writable: bool,
    cache: Arc<HandleCache>,
}

impl HostDir {
    pub fn open(path: impl Into<PathBuf>) -> io::Result<DirRef> {
        Self::open_with(path.into(), false)
    }

    /// Opens `path` read-write, creating it if needed.
    pub fn open_writable(path: impl Into<PathBuf>) -> io::Result<DirRef> {
        let path = path.into();
        std::fs::create_dir_all(&path)?;
        Self::open_with(path, true)
    }

    fn open_with(path: PathBuf, writable: bool) -> io::Result<DirRef> {
        Self::open_in(path, writable, Arc::default())
    }

    fn open_in(path: PathBuf, writable: bool, cache: Arc<HandleCache>) -> io::Result<DirRef> {
        if !path.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("{}: not a directory", path.display()),
            ));
        }
        Ok(Arc::new(Self {
            path,
            writable,
            cache,
        }))
    }

    fn child(&self, name: &str) -> io::Result<PathBuf> {
        let valid = !name.is_empty()
            && name != "."
            && name != ".."
            && !name.contains(['/', '\\', '\0'])
            && !(cfg!(windows) && name.contains(':'));
        if !valid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid name {name:?}"),
            ));
        }
        Ok(self.path.join(name))
    }

    fn require_writable(&self) -> io::Result<()> {
        if self.writable {
            Ok(())
        } else {
            Err(read_only())
        }
    }
}

impl Directory for HostDir {
    fn entries(&self) -> io::Result<Vec<DirEntry>> {
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&self.path)? {
            let entry = entry?;
            let Ok(name) = entry.file_name().into_string() else {
                continue;
            };
            let meta = entry.metadata()?;
            let kind = if meta.is_dir() {
                EntryKind::Directory
            } else {
                EntryKind::File
            };
            let size = if meta.is_dir() { 0 } else { meta.len() };
            out.push(DirEntry { name, kind, size });
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    fn open(&self, name: &str) -> io::Result<Option<Entry>> {
        let path = self.child(name)?;
        let meta = match std::fs::metadata(&path) {
            Ok(meta) => meta,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        Ok(Some(if meta.is_dir() {
            Entry::Dir(Self::open_in(path, self.writable, self.cache.clone())?)
        } else if self.writable {
            Entry::File(HostFile::open_with(&path, true)?)
        } else {
            Entry::File(Arc::new(LazyHostFile {
                id: self.cache.next_id.fetch_add(1, Ordering::Relaxed),
                path,
                size: meta.len(),
                cache: self.cache.clone(),
            }))
        }))
    }

    fn create_file(&self, name: &str, size: u64) -> io::Result<FileRef> {
        self.require_writable()?;
        let path = self.child(name)?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.set_len(size)?;
        drop(file);
        HostFile::open_with(&path, true)
    }

    fn create_dir(&self, name: &str) -> io::Result<DirRef> {
        self.require_writable()?;
        let path = self.child(name)?;
        std::fs::create_dir(&path)?;
        Self::open_with(path, true)
    }

    fn remove(&self, name: &str) -> io::Result<()> {
        self.require_writable()?;
        let path = self.child(name)?;
        if path.is_dir() {
            std::fs::remove_dir(path)
        } else {
            std::fs::remove_file(path)
        }
    }

    fn remove_all(&self, name: &str) -> io::Result<()> {
        self.require_writable()?;
        let path = self.child(name)?;
        if path.is_dir() {
            std::fs::remove_dir_all(path)
        } else {
            std::fs::remove_file(path)
        }
    }

    fn rename(&self, from: &str, to: &str) -> io::Result<()> {
        self.require_writable()?;
        std::fs::rename(self.child(from)?, self.child(to)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::{DirectoryExt, FileExt};

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!("lichii-fs-{tag}-{}", std::process::id()));
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

    #[test]
    fn read_write_and_list() {
        let temp = TempDir::new("host");
        let root = HostDir::open_writable(&temp.0).unwrap();
        let save = root.create_dir("save").unwrap();
        let file = save.create_file("data.bin", 4).unwrap();
        file.write_at(2, b"hello").unwrap();
        assert_eq!(file.size(), 7);
        assert_eq!(file.read_all().unwrap(), b"\0\0hello");
        file.set_size(3).unwrap();
        assert_eq!(file.read_all().unwrap(), b"\0\0h");

        let entries = root.entries().unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].kind, EntryKind::Directory);
        let reopened = root.open_file_path("save/data.bin").unwrap().unwrap();
        assert_eq!(reopened.read_all().unwrap(), b"\0\0h");

        save.rename("data.bin", "moved.bin").unwrap();
        assert!(save.open("data.bin").unwrap().is_none());
        assert!(root.remove("save").is_err());
        root.remove_all("save").unwrap();
        assert!(root.entries().unwrap().is_empty());
    }

    #[test]
    fn read_only_trees_never_hold_more_than_the_handle_limit() {
        let temp = TempDir::new("lazy");
        let count = OPEN_HANDLE_LIMIT + 50;
        for i in 0..count {
            std::fs::write(temp.0.join(format!("{i:04}")), (i as u32).to_le_bytes()).unwrap();
        }
        let root = HostDir::open(&temp.0).unwrap();
        let files: Vec<_> = (0..count)
            .map(|i| root.open_file(&format!("{i:04}")).unwrap().unwrap())
            .collect();
        for _ in 0..2 {
            for (i, file) in files.iter().enumerate() {
                assert_eq!(file.read_all().unwrap(), (i as u32).to_le_bytes());
            }
        }
        let mut buf = [0u8; 8];
        assert_eq!(files[0].read_at(2, &mut buf).unwrap(), 2);
        assert_eq!(files[0].read_at(9, &mut buf).unwrap(), 0);

        let cache = HandleCache::default();
        for i in 0..count {
            cache
                .get(i as u64, &temp.0.join(format!("{i:04}")))
                .unwrap();
        }
        cache
            .get(count as u64 - 1, &temp.0.join("missing"))
            .unwrap();
        let open = cache.open.lock().unwrap();
        assert_eq!(open.len(), OPEN_HANDLE_LIMIT);
        assert_eq!(open.front().unwrap().0, count as u64 - 1);
    }

    #[test]
    fn names_cannot_escape_and_read_only_refuses_writes() {
        let temp = TempDir::new("escape");
        std::fs::write(temp.0.join("f"), b"x").unwrap();
        let root = HostDir::open(&temp.0).unwrap();
        for name in ["..", ".", "", "a/b", "a\\b"] {
            assert!(root.open(name).is_err(), "{name:?}");
        }
        assert!(root.create_file("g", 0).is_err());
        let file = root.open_file("f").unwrap().unwrap();
        assert!(!file.is_writable());
        assert!(file.write_at(0, b"y").is_err());
    }
}
