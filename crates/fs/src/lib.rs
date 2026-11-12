//! Virtual file system and Nintendo container formats.
//!
//! [`vfs`] defines random-access files and directories with cheap offset views, concatenation
//! and layering; [`host`] backs them with host directories for NAND, SD card and save data.
//! The containers parse into those traits: PFS0/HFS0 ([`pfs`]), RomFS ([`romfs`]), NCA
//! archives ([`nca`]) with BKTR update patching ([`bktr`]), NSP ([`nsp`]) and XCI ([`xci`])
//! packages indexed into titles ([`content`]), and the metadata formats NACP ([`nacp`]) and
//! CNMT ([`cnmt`]).

pub mod bktr;
mod bytes;
pub mod cnmt;
pub mod content;
mod error;
pub mod host;
pub mod nacp;
pub mod nca;
pub mod nsp;
pub mod pfs;
pub mod romfs;
#[cfg(test)]
mod testutil;
pub mod vfs;
pub mod xci;

pub use crate::bktr::open_patched_romfs;
pub use crate::content::{ContentIndex, Title, TitleContent};
pub use crate::error::{FsError, FsResult};
pub use crate::nca::{Nca, NcaContentType, NcaHeader, NcaSection};
pub use crate::nsp::Nsp;
pub use crate::vfs::{
    ConcatFile, DirEntry, DirRef, Directory, DirectoryExt, Entry, EntryKind, File, FileExt,
    FileRef, LayeredDir, MemDir, MemFile, ZeroFile, slice,
};
pub use crate::xci::{Xci, XciPartition};
