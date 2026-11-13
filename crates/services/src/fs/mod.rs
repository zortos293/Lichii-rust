//! `fsp-srv`, the file system proxy, on its own service thread, plus `fsp-ldr` and `fsp-pr`,
//! which the C++ registers without commands.
//!
//! The title's RomFS, add-on content, SD card and save data come from [`ServiceContext`]: the
//! RomFS and add-ons from the [`ProgramImage`](lichii_loader::program::ProgramImage), the NAND and SD card roots from
//! `settings.paths` (laid out by [`lichii_fs::save`]). System data comes from the NAND's
//! registered content, or is synthesised ([`archive`]).

mod archive;
mod file;
mod filesystem;
mod result;
mod save;
#[cfg(test)]
mod tests;
mod transfer;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use lichii_common::paths::UserPaths;
use lichii_common::settings::Settings;
use lichii_common::{HleResult, ResultCode};
use lichii_crypto::KeySet;
use lichii_fs::romfs::open_romfs;
use lichii_fs::save::{Nand, SaveDataFactory, SaveDataSpaceId, SdCard};
use lichii_fs::{ContentIndex, FileRef};
use lichii_kernel::ipc::HleRequestContext;

use self::file::IStorage;
use self::filesystem::IFileSystem;
use self::result::*;
use self::save::{ISaveDataInfoReader, RawSaveDataAttribute, find_saves};
use crate::{Service, ServiceContext};

const THREAD: &str = "fs";
const FILE_SYSTEM_PROXY_TYPE_MANUAL: u8 = 4;
const ACCESS_LOG_VERSION_LATEST: u32 = 2;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let shared = Arc::new(FsShared::new(context.clone()));
    context.register("fsp-srv", THREAD, move |_| FspSrv::new(shared.clone()))?;
    context.register("fsp-ldr", THREAD, |_| Commandless("fsp-ldr"))?;
    context.register("fsp-pr", THREAD, |_| Commandless("fsp-pr"))
}

/// The NAND root: `settings.paths.nand_dir`, else the user's data directory.
pub(crate) fn nand_dir(settings: &Settings) -> PathBuf {
    settings
        .paths
        .nand_dir
        .clone()
        .unwrap_or_else(|| UserPaths::discover().nand_dir())
}

/// The RomFS of system data `title_id` (or another installed data title): the Data NCA of the
/// NAND's registered content, else the archive [`archive::synthesize`] builds. fsp-srv and the
/// services that read system archives themselves (time) share this one lookup.
pub(crate) fn system_data(context: &ServiceContext, title_id: u64) -> Option<FileRef> {
    let installed = context
        .installed_content
        .get_or_init(|| index_nand(&context.settings))
        .as_ref()
        .and_then(|index| {
            index
                .data_romfs(title_id)
                .inspect_err(|e| tracing::error!("opening the RomFS of {title_id:#018x}: {e}"))
                .ok()
                .flatten()
        });
    installed.or_else(|| archive::synthesize(title_id))
}

/// The titles registered in the NAND's system and user content.
fn index_nand(settings: &Settings) -> Option<ContentIndex> {
    let nand_dir = nand_dir(settings);
    let nand = Nand::open(&nand_dir)
        .inspect_err(|e| tracing::error!("opening the NAND at {}: {e}", nand_dir.display()))
        .ok()?;
    let keys_dir = settings
        .paths
        .keys_dir
        .clone()
        .unwrap_or_else(|| UserPaths::discover().keys_dir());
    let mut keys = KeySet::load_dir(&keys_dir)
        .map(|(keys, _)| keys)
        .unwrap_or_else(|_| KeySet::new());
    let mut index = ContentIndex::default();
    for dir in [nand.system_contents(), nand.user_contents()] {
        match dir
            .map_err(Into::into)
            .and_then(|d| ContentIndex::from_registered(d.as_ref(), &mut keys))
        {
            Ok(found) => index.merge(found),
            Err(e) => tracing::warn!("indexing NAND content: {e}"),
        }
    }
    Some(index)
}

/// What every `fsp-srv` session shares: the host roots.
struct FsShared {
    context: Arc<ServiceContext>,
    nand: Option<(Nand, PathBuf)>,
    sdmc: Option<(SdCard, PathBuf)>,
    saves: Option<SaveDataFactory>,
}

impl FsShared {
    fn new(context: Arc<ServiceContext>) -> Self {
        let nand_dir = nand_dir(&context.settings);
        let sdmc_dir = context
            .settings
            .paths
            .sdmc_dir
            .clone()
            .unwrap_or_else(|| UserPaths::discover().sdmc_dir());
        let nand = Nand::open(&nand_dir)
            .inspect_err(|e| tracing::error!("opening the NAND at {}: {e}", nand_dir.display()))
            .ok()
            .map(|nand| (nand, nand_dir));
        let sdmc = SdCard::open(&sdmc_dir)
            .inspect_err(|e| tracing::error!("opening the SD card at {}: {e}", sdmc_dir.display()))
            .ok()
            .map(|sdmc| (sdmc, sdmc_dir));
        let saves = nand
            .as_ref()
            .map(|(nand, _)| SaveDataFactory::new(nand.root().clone(), context.program.program_id));
        Self {
            context,
            nand,
            sdmc,
            saves,
        }
    }
}

/// Which storage's sizes `GetFreeSpaceSize` and `GetTotalSpaceSize` report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageSpace {
    NandSystem,
    NandUser,
    SdCard,
}

impl StorageSpace {
    fn for_save(space: SaveDataSpaceId) -> Self {
        match space {
            SaveDataSpaceId::User | SaveDataSpaceId::Temporary => Self::NandUser,
            SaveDataSpaceId::SdSystem | SaveDataSpaceId::SdUser => Self::SdCard,
            _ => Self::NandSystem,
        }
    }

    fn free(self, shared: &FsShared) -> u64 {
        match self {
            Self::NandSystem => shared
                .nand
                .as_ref()
                .map_or(0, |(n, _)| n.system_free_space()),
            Self::NandUser => shared.nand.as_ref().map_or(0, |(n, _)| n.user_free_space()),
            Self::SdCard => shared.sdmc.as_ref().map_or(0, |(s, _)| s.free_space()),
        }
    }

    fn total(self, shared: &FsShared) -> u64 {
        match self {
            Self::NandSystem => shared
                .nand
                .as_ref()
                .map_or(0, |(n, _)| n.system_total_space()),
            Self::NandUser => shared
                .nand
                .as_ref()
                .map_or(0, |(n, _)| n.user_total_space()),
            Self::SdCard => shared.sdmc.as_ref().map_or(0, |(s, _)| s.total_space()),
        }
    }
}

/// A service or interface the C++ registers with no commands.
struct Commandless(&'static str);

impl Service for Commandless {
    fn name(&self) -> &str {
        self.0
    }

    fn handle(&self, _ctx: &mut HleRequestContext) -> Option<ResultCode> {
        None
    }
}

/// One `fsp-srv` session.
struct FspSrv {
    shared: Arc<FsShared>,
    program_id: AtomicU64,
    access_log_mode: AtomicU32,
}

impl FspSrv {
    fn new(shared: Arc<FsShared>) -> Self {
        let program_id = shared.context.program.program_id;
        Self {
            shared,
            program_id: AtomicU64::new(program_id),
            access_log_mode: AtomicU32::new(0),
        }
    }

    fn context(&self) -> &Arc<ServiceContext> {
        &self.shared.context
    }

    fn program_id(&self) -> u64 {
        self.program_id.load(Ordering::Relaxed)
    }

    fn saves(&self) -> HleResult<&SaveDataFactory> {
        self.shared.saves.as_ref().ok_or(TARGET_NOT_FOUND)
    }

    fn push_storage(&self, ctx: &mut HleRequestContext, file: FileRef) {
        let storage = IStorage::new(file, &self.context().program);
        ctx.push_interface(self.context().handler(storage));
    }

    fn push_file_system(
        &self,
        ctx: &mut HleRequestContext,
        root: lichii_fs::DirRef,
        host: Option<PathBuf>,
        space: StorageSpace,
    ) {
        let fs = IFileSystem::new(root, host, space, self.shared.clone());
        ctx.push_interface(self.context().handler(fs));
    }

    /// `SetCurrentProcess(u64, pid)`: the application process is the only one registered.
    fn set_current_process(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let reserved: u64 = ctx.pop();
        let pid = ctx.pid().unwrap_or(reserved);
        if pid != self.context().process.id() {
            tracing::warn!("SetCurrentProcess for unregistered process {pid}");
            return Err(TARGET_NOT_FOUND);
        }
        self.program_id
            .store(self.context().program.program_id, Ordering::Relaxed);
        Ok(())
    }

    fn romfs(&self) -> HleResult<FileRef> {
        self.context().program.romfs.clone().ok_or_else(|| {
            tracing::error!("the title has no RomFS");
            UNKNOWN
        })
    }

    fn open_data_file_system_by_current_process(
        &self,
        ctx: &mut HleRequestContext,
    ) -> HleResult<()> {
        let root = open_romfs(&self.romfs()?).map_err(|e| {
            tracing::error!("extracting the RomFS of the current process: {e}");
            UNKNOWN
        })?;
        self.push_file_system(ctx, root, None, StorageSpace::NandUser);
        Ok(())
    }

    /// `OpenFileSystemWithPatch(u8 type, u64 program_id)`: only the manual is asked for, and
    /// titles carry no HTML document content.
    fn open_file_system_with_patch(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let kind: u8 = ctx.pop();
        let program_id: u64 = ctx.pop();
        if kind != FILE_SYSTEM_PROXY_TYPE_MANUAL {
            tracing::error!("OpenFileSystemWithPatch type {kind} for {program_id:#018x}");
        }
        Err(TARGET_NOT_FOUND)
    }

    fn open_sd_card_file_system(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let (sdmc, path) = self.shared.sdmc.as_ref().ok_or(PORT_SD_CARD_NO_DEVICE)?;
        self.push_file_system(
            ctx,
            sdmc.root().clone(),
            Some(path.clone()),
            StorageSpace::SdCard,
        );
        Ok(())
    }

    fn create_save(
        &self,
        space: SaveDataSpaceId,
        attribute: &RawSaveDataAttribute,
    ) -> HleResult<()> {
        self.saves()?
            .create(space, &attribute.attribute())
            .map(drop)
            .map_err(|e| {
                tracing::error!("creating save data: {e}");
                TARGET_NOT_FOUND
            })
    }

    /// `CreateSaveDataFileSystem(attribute, creation info, meta info)` in the user space. The
    /// attribute comes first, as in libnx; the C++ reads the creation info there.
    fn create_save_data_file_system(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let attribute: RawSaveDataAttribute = ctx.pop();
        self.create_save(SaveDataSpaceId::User, &attribute)
    }

    fn create_save_data_file_system_by_system_save_data_id(
        &self,
        ctx: &mut HleRequestContext,
    ) -> HleResult<()> {
        let attribute: RawSaveDataAttribute = ctx.pop();
        self.create_save(SaveDataSpaceId::System, &attribute)
    }

    /// `SaveDataCreationInfo2`: version, attribute at 8, the space id at 0x6c.
    fn create_save_data_file_system_with_creation_info2(
        &self,
        ctx: &mut HleRequestContext,
    ) -> HleResult<()> {
        let _version: u32 = ctx.pop();
        let attribute: RawSaveDataAttribute = ctx.pop();
        let _sizes: [i64; 3] = ctx.pop();
        let _owner_id: u64 = ctx.pop();
        let _flags: u32 = ctx.pop();
        let space = SaveDataSpaceId::from_raw(ctx.pop());
        self.create_save(space, &attribute)
    }

    /// `OpenSaveDataFileSystem(u8 space, attribute)` and its `BySystemSaveDataId` form, which
    /// needs a system save id and defaults the program to the current one.
    fn open_save_data_file_system(
        &self,
        ctx: &mut HleRequestContext,
        by_system_save_data_id: bool,
    ) -> HleResult<()> {
        let space = SaveDataSpaceId::from_raw(ctx.pop());
        let raw: RawSaveDataAttribute = ctx.pop();
        let mut attribute = raw.attribute();
        if by_system_save_data_id {
            if attribute.system_save_data_id == 0 {
                return Err(INVALID_ARGUMENT);
            }
            if attribute.program_id == 0 {
                attribute.program_id = self.program_id();
            }
        }
        let saves = self.saves()?;
        let dir = saves
            .open(space, &attribute)
            .map_err(|e| {
                tracing::error!("opening save data: {e}");
                TARGET_NOT_FOUND
            })?
            .ok_or(TARGET_NOT_FOUND)?;
        let host = self
            .shared
            .nand
            .as_ref()
            .map(|(_, root)| root.join(saves.path(space, &attribute).trim_start_matches('/')));
        self.push_file_system(ctx, dir, host, StorageSpace::for_save(space));
        Ok(())
    }

    /// The save data extra data reads: the C++ keeps no extra data and answers zeroes.
    fn read_extra_data(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        if let Some(buffer) = ctx.writable_buffer(0) {
            ctx.memory()
                .fill(buffer.address, buffer.size, 0)
                .map_err(|_| lichii_kernel::result::INVALID_CURRENT_MEMORY)?;
        }
        Ok(())
    }

    fn open_save_data_info_reader(
        &self,
        ctx: &mut HleRequestContext,
        spaces: &[SaveDataSpaceId],
    ) -> HleResult<()> {
        let saves = self.saves()?;
        let info = spaces
            .iter()
            .flat_map(|&space| find_saves(saves, space))
            .collect();
        ctx.push_interface(self.context().handler(ISaveDataInfoReader::new(info)));
        Ok(())
    }

    fn open_data_storage_by_current_process(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let romfs = self.romfs()?;
        self.push_storage(ctx, romfs);
        Ok(())
    }

    fn add_on(&self, title_id: u64) -> Option<FileRef> {
        self.context()
            .program
            .add_ons
            .iter()
            .find(|aoc| aoc.title_id == title_id)
            .map(|aoc| aoc.romfs.clone())
    }

    fn open_data_storage_by_program_id(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let program_id: u64 = ctx.pop();
        let storage = if program_id == self.program_id() {
            self.romfs()?
        } else {
            self.add_on(program_id).ok_or(TARGET_NOT_FOUND)?
        };
        self.push_storage(ctx, storage);
        Ok(())
    }

    /// `OpenDataStorageByDataId(u8 storage, u64 data_id)`: add-on content, then installed
    /// system data, then a synthesised archive.
    fn open_data_storage_by_data_id(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let storage_id: u8 = ctx.pop();
        let title_id: u64 = ctx.pop();
        let data = self
            .add_on(title_id)
            .or_else(|| system_data(self.context(), title_id))
            .ok_or_else(|| {
                tracing::error!("no data storage {title_id:#018x} (storage {storage_id:#x})");
                UNKNOWN
            })?;
        self.push_storage(ctx, data);
        Ok(())
    }

    /// `OpenDataStorageWithProgramIndex(u8 index)`: the program of a multi-program title. Only
    /// the running program's own RomFS is loaded.
    fn open_data_storage_with_program_index(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let index: u8 = ctx.pop();
        let program_id = self.program_id();
        if (program_id & !0xfff) + u64::from(index) != program_id {
            tracing::error!("no storage for program index {index}");
            return Err(UNKNOWN);
        }
        let romfs = self.romfs()?;
        self.push_storage(ctx, romfs);
        Ok(())
    }

    fn output_access_log_to_sd_card(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let message = ctx.read_buffer(0)?;
        let end = message
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(message.len());
        tracing::info!(target: "fs_access_log", "{}", String::from_utf8_lossy(&message[..end]));
        Ok(())
    }
}

impl Service for FspSrv {
    fn name(&self) -> &str {
        "fsp-srv"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            1 | 210 => self.set_current_process(ctx),
            2 => self.open_data_file_system_by_current_process(ctx),
            7 => self.open_file_system_with_patch(ctx),
            18 => self.open_sd_card_file_system(ctx),
            22 => self.create_save_data_file_system(ctx),
            23 => self.create_save_data_file_system_by_system_save_data_id(ctx),
            27 | 631 => {
                ctx.push(1u8);
                Ok(())
            }
            32 | 59 | 70 | 1016 => Ok(()),
            34 => {
                ctx.push([0i64; 2]);
                Ok(())
            }
            37 => self.create_save_data_file_system_with_creation_info2(ctx),
            51 | 53 => self.open_save_data_file_system(ctx, false),
            52 => self.open_save_data_file_system(ctx, true),
            57 | 58 | 69 | 71 => self.read_extra_data(ctx),
            60 => self
                .open_save_data_info_reader(ctx, &[SaveDataSpaceId::User, SaveDataSpaceId::System]),
            61 => {
                let space = SaveDataSpaceId::from_raw(ctx.pop());
                self.open_save_data_info_reader(ctx, &[space])
            }
            62 => self.open_save_data_info_reader(ctx, &[SaveDataSpaceId::Temporary]),
            67 => Err(TARGET_NOT_FOUND),
            83 => {
                ctx.push_interface(
                    self.context()
                        .handler(Commandless("ISaveDataTransferProhibiter")),
                );
                Ok(())
            }
            200 => self.open_data_storage_by_current_process(ctx),
            201 => self.open_data_storage_by_program_id(ctx),
            202 => self.open_data_storage_by_data_id(ctx),
            203 => Err(TARGET_NOT_FOUND),
            205 => self.open_data_storage_with_program_index(ctx),
            1003 => {
                if let Some(saves) = &self.shared.saves {
                    saves.set_auto_create(false);
                }
                Ok(())
            }
            1004 => {
                self.access_log_mode.store(ctx.pop(), Ordering::Relaxed);
                Ok(())
            }
            1005 => {
                ctx.push(self.access_log_mode.load(Ordering::Relaxed));
                Ok(())
            }
            1006 => self.output_access_log_to_sd_card(ctx),
            1011 => {
                ctx.push([ACCESS_LOG_VERSION_LATEST, 0]);
                Ok(())
            }
            1200 => {
                ctx.push_interface(self.context().handler(MultiCommitManager));
                Ok(())
            }
            _ => return None,
        };
        Some(result.into())
    }
}

/// `IMultiCommitManager`: save writes reach the host at once, so `Add` and `Commit` succeed.
struct MultiCommitManager;

impl Service for MultiCommitManager {
    fn name(&self) -> &str {
        "IMultiCommitManager"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            1 | 2 => Some(ResultCode::SUCCESS),
            _ => None,
        }
    }
}
