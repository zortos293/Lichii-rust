//! Turns a title on disk into a program image the kernel can start.
//!
//! Owned by the file-formats crew: NSO (LZ4 segments, MOD0), NRO, NPDM (including the Switch 2
//! ACID layout with its 0x300-byte signature area), loose ExeFS directories with a sibling
//! `romfs/`, NSP/XCI/NCA titles with updates and DLC, and the title metadata (program id,
//! name, icon, version) the game list shows.
