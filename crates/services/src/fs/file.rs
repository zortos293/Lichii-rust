//! `IFile`, `IDirectory` and `IStorage`.

use std::time::{Duration, Instant};

use lichii_common::{ConsoleGeneration, HleResult, ResultCode};
use lichii_fs::save::SAVE_DATA_SIZE_FILE_NAME;
use lichii_fs::{DirEntry, Directory, EntryKind, FileRef};
use lichii_kernel::ipc::HleRequestContext;
use lichii_loader::program::ProgramImage;
use parking_lot::Mutex;

use super::result::{self, *};
use super::transfer::{read_to_guest, write_from_guest};
use crate::Service;

pub(super) const OPEN_READ: u32 = 1;
pub(super) const OPEN_WRITE: u32 = 2;
pub(super) const OPEN_ALLOW_APPEND: u32 = 4;
const WRITE_OPTION_FLUSH: u32 = 1;
const OPERATION_INVALIDATE: u32 = 2;
const OPERATION_QUERY_RANGE: u32 = 3;
const QUERY_RANGE_INFO_SIZE: usize = 0x40;

/// `IFile`: a file opened with an `nn::fs::OpenMode`.
pub(super) struct IFile {
    file: FileRef,
    mode: u32,
}

impl IFile {
    pub(super) fn new(file: FileRef, mode: u32) -> Self {
        Self { file, mode }
    }

    fn io(error: std::io::Error) -> ResultCode {
        result::from_io(&error, UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE)
    }

    fn read(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let _option: u32 = ctx.pop();
        let offset: i64 = ctx.pop();
        let size: i64 = ctx.pop();
        if size == 0 {
            ctx.push(0i64);
            return Ok(());
        }
        if self.mode & OPEN_READ == 0 {
            return Err(READ_NOT_PERMITTED);
        }
        if offset < 0 || size < 0 || offset.checked_add(size).is_none() {
            return Err(OUT_OF_RANGE);
        }
        let buffer = ctx.writable_buffer(0).ok_or(NULLPTR_ARGUMENT)?;
        let file_size = self.file.size();
        let offset = offset as u64;
        if offset > file_size {
            return Err(OUT_OF_RANGE);
        }
        let len = (size as u64).min(buffer.size).min(file_size - offset);
        let read = read_to_guest(
            self.file.as_ref(),
            offset,
            ctx.memory(),
            buffer.address,
            len,
            UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE,
        )?;
        ctx.push(read as i64);
        Ok(())
    }

    fn write(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let option: u32 = ctx.pop();
        let offset: i64 = ctx.pop();
        let size: i64 = ctx.pop();
        if size == 0 {
            if option & WRITE_OPTION_FLUSH != 0 {
                self.file.flush().map_err(Self::io)?;
            }
            return Ok(());
        }
        if self.mode & OPEN_WRITE == 0 {
            return Err(WRITE_NOT_PERMITTED);
        }
        let Some(end) = offset.checked_add(size).filter(|_| offset >= 0 && size > 0) else {
            return Err(OUT_OF_RANGE);
        };
        if end as u64 > self.file.size() && self.mode & OPEN_ALLOW_APPEND == 0 {
            return Err(FILE_EXTENSION_WITHOUT_OPEN_MODE_ALLOW_APPEND);
        }
        let buffer = ctx.readable_buffer(0).ok_or(NULLPTR_ARGUMENT)?;
        write_from_guest(
            self.file.as_ref(),
            offset as u64,
            ctx.memory(),
            buffer.address,
            (size as u64).min(buffer.size),
            UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE,
        )?;
        if option & WRITE_OPTION_FLUSH != 0 {
            self.file.flush().map_err(Self::io)?;
        }
        Ok(())
    }

    fn set_size(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let size: i64 = ctx.pop();
        if self.mode & OPEN_WRITE == 0 {
            return Err(WRITE_NOT_PERMITTED);
        }
        if size < 0 {
            return Err(OUT_OF_RANGE);
        }
        self.file.set_size(size as u64).map_err(Self::io)
    }
}

/// Closing a writable file flushes it, so a later `Commit` covers writes made through handles
/// that are already gone.
impl Drop for IFile {
    fn drop(&mut self) {
        if self.mode & OPEN_WRITE != 0
            && let Err(error) = self.file.flush()
        {
            tracing::warn!("fsp-srv: flushing a closed file failed: {error}");
        }
    }
}

impl Service for IFile {
    fn name(&self) -> &str {
        "IFile"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(
            match ctx.command() {
                0 => self.read(ctx),
                1 => self.write(ctx),
                2 => self.file.flush().map_err(Self::io),
                3 => self.set_size(ctx),
                4 => {
                    ctx.push(self.file.size() as i64);
                    Ok(())
                }
                5 => operate_range(ctx, UNSUPPORTED_OPERATE_RANGE_FOR_FILE),
                _ => return None,
            }
            .into(),
        )
    }
}

/// `OperateRange(u32 operation, s64 offset, s64 size)`: invalidation is a no-op and a range
/// query reports no encryption.
fn operate_range(ctx: &mut HleRequestContext, unsupported: ResultCode) -> HleResult<()> {
    let operation: u32 = ctx.pop();
    match operation {
        OPERATION_INVALIDATE => Ok(()),
        OPERATION_QUERY_RANGE => {
            ctx.push_bytes(&[0; QUERY_RANGE_INFO_SIZE]);
            Ok(())
        }
        _ => Err(unsupported),
    }
}

const DIRECTORY_ENTRY_SIZE: usize = 0x310;
const ENTRY_NAME_MAX: usize = 0x300;
pub(super) const OPEN_DIRECTORIES: u32 = 1;
pub(super) const OPEN_FILES: u32 = 2;

/// `IDirectory`: the entries of a directory when it was opened, directories first.
pub(super) struct IDirectory {
    entries: Vec<DirEntry>,
    next: Mutex<usize>,
}

impl IDirectory {
    pub(super) fn new(dir: &dyn Directory, mode: u32) -> std::io::Result<Self> {
        let all = dir.entries()?;
        let mut entries = Vec::with_capacity(all.len());
        if mode & OPEN_DIRECTORIES != 0 {
            entries.extend(
                all.iter()
                    .filter(|e| e.kind == EntryKind::Directory)
                    .cloned(),
            );
        }
        if mode & OPEN_FILES != 0 {
            entries.extend(
                all.into_iter()
                    .filter(|e| e.kind == EntryKind::File && e.name != SAVE_DATA_SIZE_FILE_NAME),
            );
        }
        Ok(Self {
            entries,
            next: Mutex::new(0),
        })
    }

    fn read(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let buffer = ctx.writable_buffer(0).unwrap_or_default();
        let capacity = buffer.size as usize / DIRECTORY_ENTRY_SIZE;
        let mut next = self.next.lock();
        let count = capacity.min(self.entries.len() - *next);
        let mut raw = [0u8; DIRECTORY_ENTRY_SIZE];
        for (i, entry) in self.entries[*next..*next + count].iter().enumerate() {
            raw.fill(0);
            let name = &entry.name.as_bytes()[..entry.name.len().min(ENTRY_NAME_MAX)];
            raw[..name.len()].copy_from_slice(name);
            raw[0x304] = match entry.kind {
                EntryKind::Directory => 0,
                EntryKind::File => 1,
            };
            raw[0x308..].copy_from_slice(&(entry.size as i64).to_le_bytes());
            ctx.memory()
                .write_bytes(buffer.address + (i * DIRECTORY_ENTRY_SIZE) as u64, &raw)
                .map_err(|_| lichii_kernel::result::INVALID_CURRENT_MEMORY)?;
        }
        *next += count;
        ctx.push(count as i64);
        Ok(())
    }
}

impl Service for IDirectory {
    fn name(&self) -> &str {
        "IDirectory"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => self.read(ctx).into(),
            1 => {
                ctx.push((self.entries.len() - *self.next.lock()) as i64);
                ResultCode::SUCCESS
            }
            _ => return None,
        })
    }
}

/// The 40 MB/s a Switch 1 game card reads at (Mario Strikers needs the delay in its intro).
const SWITCH1_BYTES_PER_SECOND: u64 = 40_000_000;
/// Hyrule Warriors: Age of Imprisonment follows wild pointers when reads complete at once; it
/// boots at 100 and 200 MB/s.
const AGE_OF_IMPRISONMENT: u64 = 0x0400_3F40_232D_E000;
const AGE_OF_IMPRISONMENT_BYTES_PER_SECOND: u64 = 100_000_000;
const MAX_CHARGED_READ: u64 = 64 * 1024;

/// Paces storage reads to a card's speed: each read delays the next one.
struct Throttle {
    bytes_per_second: u64,
    next_read: Mutex<Instant>,
}

impl Throttle {
    /// The pace for a title: Switch 1 titles read at card speed, Switch 2 titles unthrottled
    /// (a Switch 2 movie streams hundreds of reads a second) except Age of Imprisonment.
    fn for_program(program: &ProgramImage) -> Option<Self> {
        let bytes_per_second = if program.generation == ConsoleGeneration::Switch1 {
            SWITCH1_BYTES_PER_SECOND
        } else if program.program_id & !0xfff == AGE_OF_IMPRISONMENT {
            AGE_OF_IMPRISONMENT_BYTES_PER_SECOND
        } else {
            return None;
        };
        Some(Self {
            bytes_per_second,
            next_read: Mutex::new(Instant::now()),
        })
    }

    fn charge(&self, length: u64) {
        let mut next_read = self.next_read.lock();
        let now = Instant::now();
        if now < *next_read {
            std::thread::sleep(*next_read - now);
        }
        let nanos = length.min(MAX_CHARGED_READ) * 1_000_000_000 / self.bytes_per_second;
        *next_read = Instant::now() + Duration::from_nanos(nanos);
    }
}

/// `IStorage`: a raw image (RomFS, add-on content, system data).
pub(super) struct IStorage {
    file: FileRef,
    throttle: Option<Throttle>,
}

impl IStorage {
    pub(super) fn new(file: FileRef, program: &ProgramImage) -> Self {
        Self {
            file,
            throttle: Throttle::for_program(program),
        }
    }

    fn io(error: std::io::Error) -> ResultCode {
        result::from_io(&error, UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE)
    }

    fn read(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let offset: i64 = ctx.pop();
        let length: i64 = ctx.pop();
        if length < 0 {
            return Err(INVALID_SIZE);
        }
        if offset < 0 {
            return Err(INVALID_OFFSET);
        }
        if let Some(throttle) = &self.throttle {
            throttle.charge(length as u64);
        }
        let buffer = ctx.writable_buffer(0).ok_or(NULLPTR_ARGUMENT)?;
        read_to_guest(
            self.file.as_ref(),
            offset as u64,
            ctx.memory(),
            buffer.address,
            (length as u64).min(buffer.size),
            UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE,
        )?;
        Ok(())
    }

    fn write(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let offset: i64 = ctx.pop();
        let length: i64 = ctx.pop();
        if length < 0 {
            return Err(INVALID_SIZE);
        }
        if offset < 0 {
            return Err(INVALID_OFFSET);
        }
        let buffer = ctx.readable_buffer(0).ok_or(NULLPTR_ARGUMENT)?;
        write_from_guest(
            self.file.as_ref(),
            offset as u64,
            ctx.memory(),
            buffer.address,
            (length as u64).min(buffer.size),
            UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE,
        )
    }

    fn set_size(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let size: i64 = ctx.pop();
        if size < 0 {
            return Err(INVALID_SIZE);
        }
        self.file.set_size(size as u64).map_err(Self::io)
    }
}

impl Service for IStorage {
    fn name(&self) -> &str {
        "IStorage"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(
            match ctx.command() {
                0 => self.read(ctx),
                1 => self.write(ctx),
                2 => self.file.flush().map_err(Self::io),
                3 => self.set_size(ctx),
                4 => {
                    ctx.push(self.file.size());
                    Ok(())
                }
                5 => operate_range(ctx, UNSUPPORTED_OPERATE_RANGE_FOR_STORAGE),
                _ => return None,
            }
            .into(),
        )
    }
}
