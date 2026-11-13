//! Where NAND, SD card and save data live on the host: the BIS, SDMC and save data layouts of
//! the C++ emulator, so existing user directories keep working.
//!
//! [`Nand`] and [`SdCard`] open their host roots with [`HostDir`]; [`SaveDataFactory`] lays
//! save data out under the NAND root (`/user/save/...`, `/system/save/...`, `/temp/...`).

use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::host::HostDir;
use crate::vfs::{DirRef, Directory, DirectoryExt, Entry, EntryKind};

/// The file that records a save's normal and journal sizes; directory listings hide it.
pub const SAVE_DATA_SIZE_FILE_NAME: &str = ".yuzu_save_size";

const NAND_USER_SIZE: u64 = 0x6_8000_0000;
const NAND_SYSTEM_SIZE: u64 = 0xA000_0000;
const NAND_TOTAL_SIZE: u64 = 0x7_47C0_0000;
const SD_STEP: u64 = 4 << 30;

/// The emulated NAND: its system and user partitions as host directories.
pub struct Nand {
    root: DirRef,
}

impl Nand {
    /// Opens (and creates) the NAND root at `path`.
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        Ok(Self {
            root: HostDir::open_writable(path)?,
        })
    }

    /// The NAND root, writable.
    pub fn root(&self) -> &DirRef {
        &self.root
    }

    /// The registered content of the system partition (`/system/Contents/registered`).
    pub fn system_contents(&self) -> io::Result<DirRef> {
        create_dir_path(&self.root, "system/Contents/registered")
    }

    /// The registered content of the user partition (`/user/Contents/registered`).
    pub fn user_contents(&self) -> io::Result<DirRef> {
        create_dir_path(&self.root, "user/Contents/registered")
    }

    /// The size of the system partition.
    pub fn system_total_space(&self) -> u64 {
        NAND_SYSTEM_SIZE
    }

    /// The total minus what `/system` holds.
    pub fn system_free_space(&self) -> u64 {
        let used = match self.root.open_dir("system") {
            Ok(Some(dir)) => dir_size(dir.as_ref()).unwrap_or(0),
            _ => 0,
        };
        NAND_SYSTEM_SIZE.saturating_sub(used)
    }

    /// The size of the user partition.
    pub fn user_total_space(&self) -> u64 {
        NAND_USER_SIZE
    }

    /// One MiB less than the total: some titles (BioShock) reject exactly the total.
    pub fn user_free_space(&self) -> u64 {
        NAND_USER_SIZE - 0x10_0000
    }

    /// The size of the whole NAND.
    pub fn full_total_space(&self) -> u64 {
        NAND_TOTAL_SIZE
    }
}

/// The emulated SD card.
pub struct SdCard {
    root: DirRef,
}

impl SdCard {
    /// Opens (and creates) the SD card root at `path`.
    pub fn open(path: impl Into<PathBuf>) -> io::Result<Self> {
        Ok(Self {
            root: HostDir::open_writable(path)?,
        })
    }

    /// The SD card root, writable.
    pub fn root(&self) -> &DirRef {
        &self.root
    }

    /// The card grows with its contents: the used size rounded up to the next 4 GiB step, so
    /// at least 4 GiB stay free.
    pub fn total_space(&self) -> u64 {
        total_for(dir_size(self.root.as_ref()).unwrap_or(0))
    }

    /// [`SdCard::total_space`] minus what the card holds.
    pub fn free_space(&self) -> u64 {
        let used = dir_size(self.root.as_ref()).unwrap_or(0);
        total_for(used) - used
    }
}

fn total_for(used: u64) -> u64 {
    ((used + SD_STEP) / SD_STEP * SD_STEP).next_multiple_of(512)
}

/// `nn::fs::SaveDataSpaceId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveDataSpaceId {
    System,
    User,
    SdSystem,
    Temporary,
    SdUser,
    ProperSystem,
    SafeMode,
    Other(u8),
}

impl SaveDataSpaceId {
    /// Decodes the guest's raw value; unknown values are kept as `Other`.
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::System,
            1 => Self::User,
            2 => Self::SdSystem,
            3 => Self::Temporary,
            4 => Self::SdUser,
            100 => Self::ProperSystem,
            101 => Self::SafeMode,
            other => Self::Other(other),
        }
    }

    /// The guest's raw value.
    pub fn raw(self) -> u8 {
        match self {
            Self::System => 0,
            Self::User => 1,
            Self::SdSystem => 2,
            Self::Temporary => 3,
            Self::SdUser => 4,
            Self::ProperSystem => 100,
            Self::SafeMode => 101,
            Self::Other(raw) => raw,
        }
    }

    fn path(self) -> &'static str {
        match self {
            Self::System => "/system/",
            Self::User | Self::SdUser => "/user/",
            Self::Temporary => "/temp/",
            _ => "/unrecognized/",
        }
    }
}

/// `nn::fs::SaveDataType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SaveDataType {
    System,
    Account,
    Bcat,
    Device,
    Temporary,
    Cache,
    SystemBcat,
    Other(u8),
}

impl SaveDataType {
    /// Decodes the guest's raw value; unknown values are kept as `Other`.
    pub fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::System,
            1 => Self::Account,
            2 => Self::Bcat,
            3 => Self::Device,
            4 => Self::Temporary,
            5 => Self::Cache,
            6 => Self::SystemBcat,
            other => Self::Other(other),
        }
    }

    /// The guest's raw value.
    pub fn raw(self) -> u8 {
        match self {
            Self::System => 0,
            Self::Account => 1,
            Self::Bcat => 2,
            Self::Device => 3,
            Self::Temporary => 4,
            Self::Cache => 5,
            Self::SystemBcat => 6,
            Self::Other(raw) => raw,
        }
    }
}

/// The fields of `nn::fs::SaveDataAttribute` that pick a save's directory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveDataAttribute {
    /// The owning program; `0` means the current program for account and device saves.
    pub program_id: u64,
    /// The account: the guest's 16 bytes read little-endian.
    pub user_id: u128,
    pub system_save_data_id: u64,
    pub save_type: SaveDataType,
}

/// Creates and finds the save data directories of one program under the save root (the NAND
/// root), the way `FileSys::SaveDataFactory` does.
pub struct SaveDataFactory {
    root: DirRef,
    program_id: u64,
    auto_create: AtomicBool,
}

impl SaveDataFactory {
    /// Saves live under `root` (the NAND root). `program_id` is the running program, which
    /// account and device saves with program id 0 belong to.
    pub fn new(root: DirRef, program_id: u64) -> Self {
        Self {
            root,
            program_id,
            auto_create: AtomicBool::new(true),
        }
    }

    /// Whether [`SaveDataFactory::open`] creates missing temporary, cache and current-program
    /// account and device saves (`DisableAutoSaveDataCreation` turns it off).
    pub fn set_auto_create(&self, enabled: bool) {
        self.auto_create.store(enabled, Ordering::Relaxed);
    }

    /// Creates the save's directory, or returns the existing one.
    pub fn create(&self, space: SaveDataSpaceId, attr: &SaveDataAttribute) -> io::Result<DirRef> {
        create_dir_path(&self.root, &self.path(space, attr))
    }

    /// The save's directory. A missing temporary, cache or current-program save is created.
    pub fn open(
        &self,
        space: SaveDataSpaceId,
        attr: &SaveDataAttribute,
    ) -> io::Result<Option<DirRef>> {
        if let Some(dir) = self.root.open_dir_path(&self.path(space, attr))? {
            return Ok(Some(dir));
        }
        if self.auto_create.load(Ordering::Relaxed) && is_implicit(space, attr) {
            return self.create(space, attr).map(Some);
        }
        Ok(None)
    }

    /// The directory of a whole save data space, when it exists.
    pub fn space_dir(&self, space: SaveDataSpaceId) -> io::Result<Option<DirRef>> {
        self.root.open_dir_path(space.path())
    }

    /// The save's directory below the root, for example
    /// `/user/save/0000000000000000/<user>/<program>` for an account save.
    pub fn path(&self, space: SaveDataSpaceId, attr: &SaveDataAttribute) -> String {
        let mut title_id = attr.program_id;
        if matches!(attr.save_type, SaveDataType::Account | SaveDataType::Device) && title_id == 0 {
            title_id = self.program_id;
        }
        if let Some(future) = future_path(space, attr.save_type, title_id & !0xff, attr.user_id)
            && matches!(self.root.open_dir_path(&future), Ok(Some(_)))
        {
            tracing::info!("using the save at its new location {future}");
            return future;
        }
        let out = space.path();
        let user = format!(
            "{:016X}{:016X}",
            (attr.user_id >> 64) as u64,
            attr.user_id as u64
        );
        match attr.save_type {
            SaveDataType::System => {
                format!("{out}save/{:016X}/{user}", attr.system_save_data_id)
            }
            SaveDataType::Account | SaveDataType::Device => {
                format!("{out}save/{:016X}/{user}/{title_id:016X}", 0)
            }
            SaveDataType::Temporary => format!("{out}{:016X}/{user}/{title_id:016X}", 0),
            SaveDataType::Cache => format!("{out}save/cache/{title_id:016X}"),
            other => format!("{out}save/unknown_{:X}/{title_id:016X}", other.raw()),
        }
    }
}

fn is_implicit(space: SaveDataSpaceId, attr: &SaveDataAttribute) -> bool {
    matches!(
        attr.save_type,
        SaveDataType::Cache | SaveDataType::Temporary
    ) || (space == SaveDataSpaceId::User
        && matches!(attr.save_type, SaveDataType::Account | SaveDataType::Device)
        && attr.program_id == 0
        && attr.system_save_data_id == 0)
}

/// The newer layout (`/user/save/account/<uuid>/<program>/0`), preferred when it exists.
fn future_path(
    space: SaveDataSpaceId,
    save_type: SaveDataType,
    title_id: u64,
    user_id: u128,
) -> Option<String> {
    if space != SaveDataSpaceId::User {
        return None;
    }
    match save_type {
        SaveDataType::Account => {
            let uuid: String = user_id
                .to_le_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            Some(format!("/user/save/account/{uuid}/{title_id:016X}/0"))
        }
        SaveDataType::Device => Some(format!("/user/save/device/{title_id:016X}/0")),
        _ => None,
    }
}

/// Opens the directory at `path` below `root`, creating every missing component.
pub fn create_dir_path(root: &DirRef, path: &str) -> io::Result<DirRef> {
    let mut dir = root.clone();
    for part in path.split('/').filter(|p| !p.is_empty() && *p != ".") {
        dir = match dir.open(part)? {
            Some(Entry::Dir(next)) => next,
            Some(Entry::File(_)) => {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!("{part}: not a directory"),
                ));
            }
            None => dir.create_dir(part)?,
        };
    }
    Ok(dir)
}

/// The size of every file below `dir`.
pub fn dir_size(dir: &dyn Directory) -> io::Result<u64> {
    let mut total = 0;
    for entry in dir.entries()? {
        total += match entry.kind {
            EntryKind::File => entry.size,
            EntryKind::Directory => match dir.open_dir(&entry.name)? {
                Some(sub) => dir_size(sub.as_ref())?,
                None => 0,
            },
        };
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("lichii-save-{tag}-{}", std::process::id()));
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

    const USER: u128 = 0x1122_3344_5566_7788_99aa_bbcc_ddee_ff00;
    const USER_DIR: &str = "112233445566778899AABBCCDDEEFF00";
    const PROGRAM: u64 = 0x0100_0000_0000_1000;

    fn attr(
        save_type: SaveDataType,
        program_id: u64,
        system_save_data_id: u64,
    ) -> SaveDataAttribute {
        SaveDataAttribute {
            program_id,
            user_id: USER,
            system_save_data_id,
            save_type,
        }
    }

    #[test]
    fn paths_follow_the_cpp_layout() {
        let temp = TempDir::new("paths");
        let nand = Nand::open(&temp.0).unwrap();
        let saves = SaveDataFactory::new(nand.root().clone(), PROGRAM);
        let path = |space, save_type, program, system| {
            saves.path(space, &attr(save_type, program, system))
        };
        assert_eq!(
            path(SaveDataSpaceId::User, SaveDataType::Account, 0, 0),
            format!("/user/save/0000000000000000/{USER_DIR}/0100000000001000")
        );
        assert_eq!(
            path(
                SaveDataSpaceId::User,
                SaveDataType::Device,
                0x0100_0000_0000_2000,
                0
            ),
            format!("/user/save/0000000000000000/{USER_DIR}/0100000000002000")
        );
        assert_eq!(
            path(
                SaveDataSpaceId::System,
                SaveDataType::System,
                0,
                0x8000_0000_0000_0010
            ),
            format!("/system/save/8000000000000010/{USER_DIR}")
        );
        assert_eq!(
            path(
                SaveDataSpaceId::Temporary,
                SaveDataType::Temporary,
                PROGRAM,
                0
            ),
            format!("/temp/0000000000000000/{USER_DIR}/0100000000001000")
        );
        assert_eq!(
            path(SaveDataSpaceId::User, SaveDataType::Cache, PROGRAM, 0),
            "/user/save/cache/0100000000001000"
        );
        assert_eq!(
            path(SaveDataSpaceId::SdUser, SaveDataType::Bcat, PROGRAM, 0),
            "/user/save/unknown_2/0100000000001000"
        );
        assert_eq!(
            path(SaveDataSpaceId::Other(7), SaveDataType::Cache, PROGRAM, 0),
            "/unrecognized/save/cache/0100000000001000"
        );
    }

    #[test]
    fn the_new_layout_wins_when_it_exists() {
        let temp = TempDir::new("future");
        let nand = Nand::open(&temp.0).unwrap();
        let saves = SaveDataFactory::new(nand.root().clone(), PROGRAM);
        let future = "/user/save/account/00ffeeddccbbaa998877665544332211/0100000000001000/0";
        create_dir_path(nand.root(), future).unwrap();
        assert_eq!(
            saves.path(SaveDataSpaceId::User, &attr(SaveDataType::Account, 0, 0)),
            future
        );
        create_dir_path(nand.root(), "/user/save/device/0100000000001000/0").unwrap();
        assert_eq!(
            saves.path(
                SaveDataSpaceId::User,
                &attr(SaveDataType::Device, 0x0100_0000_0000_1001, 0)
            ),
            "/user/save/device/0100000000001000/0"
        );
    }

    #[test]
    fn open_creates_only_implicit_saves() {
        let temp = TempDir::new("open");
        let nand = Nand::open(&temp.0).unwrap();
        let saves = SaveDataFactory::new(nand.root().clone(), PROGRAM);
        let system = attr(SaveDataType::System, 0, 0x10);
        assert!(
            saves
                .open(SaveDataSpaceId::System, &system)
                .unwrap()
                .is_none()
        );
        saves.create(SaveDataSpaceId::System, &system).unwrap();
        assert!(
            saves
                .open(SaveDataSpaceId::System, &system)
                .unwrap()
                .is_some()
        );

        let account = attr(SaveDataType::Account, 0, 0);
        assert!(
            saves
                .open(SaveDataSpaceId::User, &account)
                .unwrap()
                .is_some()
        );
        assert!(
            temp.0
                .join("user/save/0000000000000000")
                .join(USER_DIR)
                .is_dir()
        );
        let other = attr(SaveDataType::Account, 0x0100_0000_0000_2000, 0);
        assert!(saves.open(SaveDataSpaceId::User, &other).unwrap().is_none());

        let cache = attr(SaveDataType::Cache, PROGRAM, 0);
        saves.set_auto_create(false);
        assert!(saves.open(SaveDataSpaceId::User, &cache).unwrap().is_none());
        saves.set_auto_create(true);
        assert!(saves.open(SaveDataSpaceId::User, &cache).unwrap().is_some());
        assert!(saves.space_dir(SaveDataSpaceId::User).unwrap().is_some());
        assert!(
            saves
                .space_dir(SaveDataSpaceId::Temporary)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn space_sizes() {
        let temp = TempDir::new("sizes");
        let nand = Nand::open(temp.0.join("nand")).unwrap();
        let sd = SdCard::open(temp.0.join("sdmc")).unwrap();
        assert_eq!(nand.user_free_space(), nand.user_total_space() - 0x10_0000);
        assert_eq!(nand.system_free_space(), nand.system_total_space());
        let dir = create_dir_path(nand.root(), "system/save").unwrap();
        dir.create_file("f", 0x1000).unwrap();
        assert_eq!(nand.system_free_space(), nand.system_total_space() - 0x1000);
        assert!(
            nand.system_contents()
                .unwrap()
                .entries()
                .unwrap()
                .is_empty()
        );
        nand.user_contents().unwrap();
        assert!(temp.0.join("nand/user/Contents/registered").is_dir());

        assert_eq!(sd.total_space(), SD_STEP);
        sd.root().create_file("big", 0x2000).unwrap();
        assert_eq!(sd.total_space(), SD_STEP);
        assert_eq!(sd.free_space(), SD_STEP - 0x2000);
        assert!(create_dir_path(sd.root(), "big/x").is_err());
    }
}
