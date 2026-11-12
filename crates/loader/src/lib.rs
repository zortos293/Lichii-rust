//! Turns a title on disk into a [`ProgramImage`] the kernel can start.
//!
//! NSO (LZ4 and Switch 2 ZBIC segments, MOD0), NRO with assets, NPDM in the Switch 1 and
//! Switch 2 ACID layouts, loose ExeFS directories with their sibling `romfs/`, and the title
//! metadata (program id, name, icon, version) the game list shows. [`patch`] applies mods:
//! ExeFS replacement and IPS/IPSwitch patches, and LayeredFS RomFS replacement.

mod error;
pub mod exefs;
pub mod npdm;
pub mod nro;
pub mod nso;
pub mod patch;
pub mod program;
pub mod title;
pub mod zstd;

pub use crate::error::{LoaderError, LoaderResult};
pub use crate::npdm::Npdm;
pub use crate::patch::{ModOptions, Mods};
pub use crate::program::{
    AddOnContent, MainThread, Mod0, Module, Permission, ProgramImage, Segment,
};
pub use crate::title::{LoadOptions, LooseTitle, TitleFormat, TitleInfo, identify, load, scan};
