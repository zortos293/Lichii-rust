//! `IFileSystem` over a [`Directory`]: the SD card, save data and read-only RomFS mounts.

use std::path::PathBuf;
use std::sync::{Arc, Weak};
use std::time::UNIX_EPOCH;

use lichii_common::{HleResult, ResultCode};
use lichii_fs::{DirRef, Entry, File};
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::result::INVALID_CURRENT_MEMORY;
use parking_lot::Mutex;

use super::file::{
    IDirectory, IFile, OPEN_ALLOW_APPEND, OPEN_DIRECTORIES, OPEN_FILES, OPEN_READ, OPEN_WRITE,
};
use super::result::{self, *};
use super::transfer::{copy_file, copy_tree};
use super::{FsShared, StorageSpace};
use crate::Service;

/// `nn::fs::EntryNameLengthMax` plus the terminator: the size of `nn::fssrv::sf::Path`.
const PATH_SIZE: usize = 0x301;
const OPEN_DIRECTORY_NOT_REQUIRE_FILE_SIZE: u32 = 1 << 31;
const ENTRY_TYPE_DIRECTORY: u32 = 0;
const ENTRY_TYPE_FILE: u32 = 1;

/// A guest path read from a pointer buffer without allocating. Components split on `/` and
/// `\`; empty and `.` components are skipped.
struct GuestPath {
    bytes: [u8; PATH_SIZE],
    len: usize,
}

impl GuestPath {
    fn read(ctx: &HleRequestContext, index: usize) -> HleResult<Self> {
        let buffer = ctx.readable_buffer(index).ok_or(NULLPTR_ARGUMENT)?;
        let mut bytes = [0u8; PATH_SIZE];
        let len = (buffer.size as usize).min(PATH_SIZE);
        ctx.memory()
            .read_bytes(buffer.address, &mut bytes[..len])
            .map_err(|_| INVALID_CURRENT_MEMORY)?;
        let len = bytes[..len].iter().position(|&b| b == 0).unwrap_or(len);
        std::str::from_utf8(&bytes[..len]).map_err(|_| INVALID_CHARACTER)?;
        Ok(Self { bytes, len })
    }

    fn as_str(&self) -> &str {
        std::str::from_utf8(&self.bytes[..self.len]).unwrap_or_default()
    }

    fn components(&self) -> impl Iterator<Item = &str> {
        self.as_str()
            .split(['/', '\\'])
            .filter(|c| !c.is_empty() && *c != ".")
    }

    fn parent_components(&self) -> impl Iterator<Item = &str> {
        let count = self.components().count();
        self.components().take(count.saturating_sub(1))
    }

    fn name(&self) -> Option<&str> {
        self.components().last()
    }
}

/// `IFileSystem`. Writes go straight to the backend, so `Commit` only flushes the files the
/// guest opened.
pub(super) struct IFileSystem {
    root: DirRef,
    /// The host directory behind `root`, for timestamps.
    host: Option<PathBuf>,
    space: StorageSpace,
    shared: Arc<FsShared>,
    open_files: Mutex<Vec<Weak<dyn File>>>,
}

fn io(error: std::io::Error) -> ResultCode {
    result::from_io(&error, UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE_SYSTEM)
}

impl IFileSystem {
    pub(super) fn new(
        root: DirRef,
        host: Option<PathBuf>,
        space: StorageSpace,
        shared: Arc<FsShared>,
    ) -> Self {
        Self {
            root,
            host,
            space,
            shared,
            open_files: Mutex::new(Vec::new()),
        }
    }

    fn dir_at<'a>(&self, parts: impl Iterator<Item = &'a str>) -> HleResult<Option<DirRef>> {
        let mut dir = self.root.clone();
        for part in parts {
            dir = match dir.open(part).map_err(io)? {
                Some(Entry::Dir(next)) => next,
                _ => return Ok(None),
            };
        }
        Ok(Some(dir))
    }

    /// The parent directory and last component of `path`; `None` for the root.
    fn locate<'p>(&self, path: &'p GuestPath) -> HleResult<Option<(DirRef, &'p str)>> {
        let Some(name) = path.name() else {
            return Ok(None);
        };
        let parent = self
            .dir_at(path.parent_components())?
            .ok_or(PATH_NOT_FOUND)?;
        Ok(Some((parent, name)))
    }

    fn create_file(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let _option: i32 = ctx.pop();
        let size: i64 = ctx.pop();
        if size < 0 {
            return Err(OUT_OF_RANGE);
        }
        let (parent, name) = self.locate(&path)?.ok_or(PATH_ALREADY_EXISTS)?;
        if parent.open(name).map_err(io)?.is_some() {
            return Err(PATH_ALREADY_EXISTS);
        }
        parent.create_file(name, size as u64).map_err(io)?;
        Ok(())
    }

    fn delete_file(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let Some((parent, name)) = self.locate(&path)? else {
            return Ok(());
        };
        match parent.open(name).map_err(io)? {
            Some(Entry::File(_)) => parent.remove(name).map_err(io),
            _ => Err(PATH_NOT_FOUND),
        }
    }

    /// Creates every missing component, as the C++ does (UMM needs it).
    fn create_directory(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        if self.dir_at(path.components())?.is_some() {
            return Err(PATH_ALREADY_EXISTS);
        }
        let mut dir = self.root.clone();
        for part in path.components() {
            dir = match dir.open(part).map_err(io)? {
                Some(Entry::Dir(next)) => next,
                Some(Entry::File(_)) => return Err(PATH_ALREADY_EXISTS),
                None => dir.create_dir(part).map_err(io)?,
            };
        }
        Ok(())
    }

    fn directory_target(&self, path: &GuestPath) -> HleResult<(DirRef, DirRef, String)> {
        let (parent, name) = self.locate(path)?.ok_or(PATH_NOT_FOUND)?;
        match parent.open(name).map_err(io)? {
            Some(Entry::Dir(dir)) => Ok((parent, dir, name.to_owned())),
            _ => Err(PATH_NOT_FOUND),
        }
    }

    fn delete_directory(&self, ctx: &mut HleRequestContext, recursive: bool) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let (parent, dir, name) = self.directory_target(&path)?;
        if recursive {
            return parent.remove_all(&name).map_err(io);
        }
        if !dir.entries().map_err(io)?.is_empty() {
            return Err(DIRECTORY_NOT_EMPTY);
        }
        parent.remove(&name).map_err(io)
    }

    fn clean_directory_recursively(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let dir = self.dir_at(path.components())?.ok_or(PATH_NOT_FOUND)?;
        for entry in dir.entries().map_err(io)? {
            dir.remove_all(&entry.name).map_err(io)?;
        }
        Ok(())
    }

    fn rename(&self, ctx: &mut HleRequestContext, directory: bool) -> HleResult<()> {
        let from = GuestPath::read(ctx, 0)?;
        let to = GuestPath::read(ctx, 1)?;
        let (src_parent, src_name) = self.locate(&from)?.ok_or(PATH_NOT_FOUND)?;
        let source = src_parent.open(src_name).map_err(io)?;
        match (&source, directory) {
            (Some(Entry::File(_)), false) | (Some(Entry::Dir(_)), true) => {}
            _ => return Err(PATH_NOT_FOUND),
        }
        let (dst_parent, dst_name) = self.locate(&to)?.ok_or(PATH_ALREADY_EXISTS)?;
        if dst_parent.open(dst_name).map_err(io)?.is_some() {
            return Err(PATH_ALREADY_EXISTS);
        }
        if from.parent_components().eq(to.parent_components()) {
            return src_parent.rename(src_name, dst_name).map_err(io);
        }
        match source {
            Some(Entry::File(file)) => {
                let copy = dst_parent.create_file(dst_name, 0).map_err(io)?;
                copy_file(file.as_ref(), copy.as_ref()).map_err(io)?;
                src_parent.remove(src_name).map_err(io)
            }
            Some(Entry::Dir(dir)) => {
                copy_tree(dir.as_ref(), &dst_parent, dst_name).map_err(io)?;
                src_parent.remove_all(src_name).map_err(io)
            }
            None => Err(PATH_NOT_FOUND),
        }
    }

    fn entry(&self, path: &GuestPath) -> HleResult<Option<Entry>> {
        match self.locate(path)? {
            Some((parent, name)) => parent.open(name).map_err(io),
            None => Ok(Some(Entry::Dir(self.root.clone()))),
        }
    }

    fn get_entry_type(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let kind = match self.entry(&path)?.ok_or(PATH_NOT_FOUND)? {
            Entry::Dir(_) => ENTRY_TYPE_DIRECTORY,
            Entry::File(_) => ENTRY_TYPE_FILE,
        };
        ctx.push(kind);
        Ok(())
    }

    fn open_file(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let mode: u32 = ctx.pop();
        let all = OPEN_READ | OPEN_WRITE | OPEN_ALLOW_APPEND;
        if mode & (OPEN_READ | OPEN_WRITE) == 0 || mode & !all != 0 {
            return Err(INVALID_OPEN_MODE);
        }
        let Some(Entry::File(file)) = self.entry(&path)? else {
            return Err(PATH_NOT_FOUND);
        };
        if mode & OPEN_WRITE != 0 {
            let mut open_files = self.open_files.lock();
            open_files.retain(|f| f.strong_count() > 0);
            open_files.push(Arc::downgrade(&file));
        }
        ctx.push_interface(self.shared.context.handler(IFile::new(file, mode)));
        Ok(())
    }

    fn open_directory(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let mode: u32 = ctx.pop();
        let all = OPEN_DIRECTORIES | OPEN_FILES;
        if mode & all == 0 || mode & !(all | OPEN_DIRECTORY_NOT_REQUIRE_FILE_SIZE) != 0 {
            return Err(INVALID_OPEN_MODE);
        }
        let dir = self.dir_at(path.components())?.ok_or(PATH_NOT_FOUND)?;
        let directory = IDirectory::new(dir.as_ref(), mode).map_err(io)?;
        ctx.push_interface(self.shared.context.handler(directory));
        Ok(())
    }

    fn commit(&self) -> HleResult<()> {
        let mut open_files = self.open_files.lock();
        open_files.retain(|f| f.strong_count() > 0);
        for file in open_files.iter().filter_map(Weak::upgrade) {
            file.flush().map_err(io)?;
        }
        Ok(())
    }

    fn space_size(&self, ctx: &mut HleRequestContext, total: bool) -> HleResult<()> {
        GuestPath::read(ctx, 0)?;
        let size = if total {
            self.space.total(&self.shared)
        } else {
            self.space.free(&self.shared)
        };
        ctx.push(size as i64);
        Ok(())
    }

    /// `FileTimeStampRaw`: created, accessed and modified POSIX seconds; zero for backends
    /// without host files.
    fn get_file_time_stamp_raw(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        self.entry(&path)?.ok_or(PATH_NOT_FOUND)?;
        let mut stamps = [0u64; 4];
        if let Some(host) = &self.host {
            let host = path.components().fold(host.clone(), |p, c| p.join(c));
            if let Ok(meta) = std::fs::symlink_metadata(host) {
                let seconds = |time: std::io::Result<std::time::SystemTime>| {
                    time.ok()
                        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                        .map_or(0, |d| d.as_secs())
                };
                let modified = seconds(meta.modified());
                stamps = [
                    meta.created().map_or(modified, |t| seconds(Ok(t))),
                    seconds(meta.accessed()),
                    modified,
                    0,
                ];
            }
        }
        ctx.push(stamps);
        Ok(())
    }

    /// `QueryEntry(out buffer, in buffer, s32 query, path)`. Host files need no concatenation
    /// attribute or MAC, and there is no signed system partition on the SD card.
    fn query_entry(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let path = GuestPath::read(ctx, 0)?;
        let query: i32 = ctx.pop();
        match query {
            0 => {
                self.entry(&path)?.ok_or(PATH_NOT_FOUND)?;
                Ok(())
            }
            1 => Ok(()),
            2 => {
                ctx.write_buffer(0, &[0])?;
                Ok(())
            }
            3 => Err(NOT_IMPLEMENTED),
            _ => Err(INVALID_ARGUMENT),
        }
    }
}

/// `nn::fs::FileSystemAttribute` as the C++ reports it for every file system: entry names up
/// to 0x80 bytes, paths up to 0x300.
fn file_system_attribute() -> [u8; 0xc0] {
    const NAME: i32 = 0x80;
    const PATH: i32 = 0x300;
    let mut raw = [0u8; 0xc0];
    raw[..13].fill(1);
    let limits = [
        NAME, NAME, PATH, PATH, PATH, PATH, PATH, PATH, PATH, NAME, NAME, PATH, PATH,
    ];
    for (i, limit) in limits.into_iter().enumerate() {
        raw[0x28 + 4 * i..0x2c + 4 * i].copy_from_slice(&limit.to_le_bytes());
    }
    raw
}

impl Service for IFileSystem {
    fn name(&self) -> &str {
        "IFileSystem"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(
            match ctx.command() {
                0 => self.create_file(ctx),
                1 => self.delete_file(ctx),
                2 => self.create_directory(ctx),
                3 => self.delete_directory(ctx, false),
                4 => self.delete_directory(ctx, true),
                5 => self.rename(ctx, false),
                6 => self.rename(ctx, true),
                7 => self.get_entry_type(ctx),
                8 => self.open_file(ctx),
                9 => self.open_directory(ctx),
                10 => self.commit(),
                11 => self.space_size(ctx, false),
                12 => self.space_size(ctx, true),
                13 => self.clean_directory_recursively(ctx),
                14 => self.get_file_time_stamp_raw(ctx),
                15 => self.query_entry(ctx),
                16 => {
                    ctx.push_bytes(&file_system_attribute());
                    Ok(())
                }
                _ => return None,
            }
            .into(),
        )
    }
}
