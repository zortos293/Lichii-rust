//! Save data parameter layouts and `ISaveDataInfoReader`.

use bytemuck::{Pod, Zeroable};
use lichii_common::{HleResult, ResultCode};
use lichii_fs::save::{
    SaveDataAttribute, SaveDataFactory, SaveDataSpaceId, SaveDataType, dir_size,
};
use lichii_fs::{DirRef, Directory, DirectoryExt, EntryKind};
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::result::INVALID_CURRENT_MEMORY;
use parking_lot::Mutex;

use crate::Service;

/// `nn::fs::SaveDataAttribute` (0x40 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct RawSaveDataAttribute {
    pub program_id: u64,
    pub user_id: [u64; 2],
    pub system_save_data_id: u64,
    pub save_type: u8,
    pub rank: u8,
    pub index: u16,
    pub reserved: [u8; 0x1c],
}

impl RawSaveDataAttribute {
    pub(super) fn attribute(&self) -> SaveDataAttribute {
        SaveDataAttribute {
            program_id: self.program_id,
            user_id: u128::from(self.user_id[0]) | u128::from(self.user_id[1]) << 64,
            system_save_data_id: self.system_save_data_id,
            save_type: SaveDataType::from_raw(self.save_type),
        }
    }
}

/// `SaveDataInfo` as `ReadSaveDataInfo` returns it (0x60 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq)]
pub(super) struct SaveDataInfo {
    pub save_id_unknown: u64,
    pub space: u8,
    pub save_type: u8,
    pub padding: [u8; 6],
    pub user_id: [u8; 0x10],
    pub save_id: u64,
    pub title_id: u64,
    pub save_image_size: u64,
    pub index: u16,
    pub rank: u8,
    pub reserved: [u8; 0x25],
}

impl SaveDataInfo {
    fn new(space: SaveDataSpaceId, save_type: SaveDataType, user_id: u128) -> Self {
        Self {
            space: space.raw(),
            save_type: save_type.raw(),
            user_id: user_id.to_le_bytes(),
            ..Self::zeroed()
        }
    }
}

/// A 16-digit hex directory name, or 0.
fn hex_id(name: &str) -> u64 {
    if name.len() != 16 {
        return 0;
    }
    u64::from_str_radix(name, 16).unwrap_or(0)
}

fn subdirs(dir: &dyn Directory) -> Vec<(String, DirRef)> {
    let Ok(entries) = dir.entries() else {
        return Vec::new();
    };
    entries
        .into_iter()
        .filter(|e| e.kind == EntryKind::Directory)
        .filter_map(|e| Some((e.name.clone(), dir.open_dir(&e.name).ok()??)))
        .collect()
}

/// Every save in `space`, found by walking the save layout the way the C++ reader does.
pub(super) fn find_saves(saves: &SaveDataFactory, space: SaveDataSpaceId) -> Vec<SaveDataInfo> {
    let mut out = Vec::new();
    let Ok(Some(root)) = saves.space_dir(space) else {
        tracing::error!("the save root of space {space:?} does not exist");
        return out;
    };
    for (name, kind) in subdirs(root.as_ref()) {
        if name == "save" {
            normal_saves(space, kind.as_ref(), &mut out);
        } else if space == SaveDataSpaceId::Temporary {
            temporary_saves(space, &name, kind.as_ref(), &mut out);
        }
    }
    out
}

fn user_dirs(dir: &dyn Directory) -> impl Iterator<Item = (u128, DirRef)> {
    subdirs(dir)
        .into_iter()
        .filter(|(name, _)| name.len() == 0x20)
        .map(|(name, dir)| (u128::from_str_radix(&name, 16).unwrap_or(0), dir))
}

fn normal_saves(space: SaveDataSpaceId, root: &dyn Directory, out: &mut Vec<SaveDataInfo>) {
    for (save_id_name, save_id_dir) in subdirs(root) {
        let save_id = hex_id(&save_id_name);
        for (user, user_dir) in user_dirs(save_id_dir.as_ref()) {
            if save_id != 0 {
                out.push(SaveDataInfo {
                    save_id,
                    save_image_size: dir_size(user_dir.as_ref()).unwrap_or(0),
                    ..SaveDataInfo::new(space, SaveDataType::System, user)
                });
                continue;
            }
            let save_type = if user == 0 {
                SaveDataType::Device
            } else {
                SaveDataType::Account
            };
            for (title, title_dir) in subdirs(user_dir.as_ref()) {
                out.push(SaveDataInfo {
                    title_id: hex_id(&title),
                    save_image_size: dir_size(title_dir.as_ref()).unwrap_or(0),
                    ..SaveDataInfo::new(space, save_type, user)
                });
            }
        }
    }
}

fn temporary_saves(
    space: SaveDataSpaceId,
    name: &str,
    root: &dyn Directory,
    out: &mut Vec<SaveDataInfo>,
) {
    for (user, user_dir) in user_dirs(root) {
        for (title, title_dir) in subdirs(user_dir.as_ref()) {
            if title_dir.entries().is_ok_and(|e| !e.is_empty()) {
                out.push(SaveDataInfo {
                    save_id: hex_id(name),
                    title_id: hex_id(&title),
                    save_image_size: dir_size(title_dir.as_ref()).unwrap_or(0),
                    ..SaveDataInfo::new(space, SaveDataType::Temporary, user)
                });
            }
        }
    }
}

/// `ISaveDataInfoReader`: the saves found when it was opened.
pub(super) struct ISaveDataInfoReader {
    info: Vec<SaveDataInfo>,
    next: Mutex<usize>,
}

impl ISaveDataInfoReader {
    pub(super) fn new(info: Vec<SaveDataInfo>) -> Self {
        Self {
            info,
            next: Mutex::new(0),
        }
    }

    fn read(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let buffer = ctx.writable_buffer(0).unwrap_or_default();
        let capacity = buffer.size as usize / size_of::<SaveDataInfo>();
        let mut next = self.next.lock();
        let entries = &self.info[*next..];
        let entries = &entries[..capacity.min(entries.len())];
        ctx.memory()
            .write_bytes(buffer.address, bytemuck::cast_slice(entries))
            .map_err(|_| INVALID_CURRENT_MEMORY)?;
        *next += entries.len();
        ctx.push(entries.len() as u64);
        Ok(())
    }
}

impl Service for ISaveDataInfoReader {
    fn name(&self) -> &str {
        "ISaveDataInfoReader"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            0 => self.read(ctx).into(),
            _ => return None,
        })
    }
}
