//! Virtual file system and Nintendo container formats.
//!
//! [`vfs`] defines random-access files and directories with cheap offset views, concatenation
//! and layering; [`host`] backs them with host directories for NAND, SD card and save data.
//! The containers parse into those traits: PFS0/HFS0 ([`pfs`]), RomFS ([`romfs`]), and the
//! metadata formats NACP ([`nacp`]) and CNMT ([`cnmt`]). [`ips`] applies IPS and IPSwitch
//! patches to executables and RomFS files.

mod bytes;
pub mod cnmt;
mod error;
pub mod host;
pub mod ips;
pub mod nacp;
pub mod pfs;
pub mod romfs;
pub mod vfs;

pub use crate::error::{FsError, FsResult};
pub use crate::vfs::{
    ConcatFile, DirEntry, DirRef, Directory, DirectoryExt, Entry, EntryKind, File, FileExt,
    FileRef, LayeredDir, MemDir, MemFile, ZeroFile, slice,
};
