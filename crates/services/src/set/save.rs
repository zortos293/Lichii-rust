//! The settings `set:sys` keeps across boots, in the files of the C++ settings sysmodule: a
//! `yuzu_set` header, then the raw `SystemSettings` or `PrivateSettings`, at
//! `system/save/8000000000000050/settings.dat` and `system/save/8000000000000052/settings.dat` of
//! the NAND. Values Lichii does not serve keep the bytes the file had.

use std::path::{Path, PathBuf};

use lichii_common::settings::write_atomic;

const MAGIC: [u8; 8] = *b"yuzu_set";
const VERSION: u32 = 4;
const HEADER_SIZE: usize = 0x10;
const FILE_NAME: &str = "settings.dat";
/// `SystemSettings::version` and `flags` of a 14.0.0+ console, as the C++ defaults them.
const SYSTEM_VERSION: u32 = 0x14_0000;
const SYSTEM_FLAGS: u32 = 7;
const MII_AUTHOR_ID_OFFSET: usize = 0x94b0;

/// One of the settings structures.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Part {
    /// `nn::settings::system::SystemSettings`.
    System,
    /// `PrivateSettings`.
    Private,
}

impl Part {
    fn dir(self) -> &'static str {
        match self {
            Self::System => "system/save/8000000000000050",
            Self::Private => "system/save/8000000000000052",
        }
    }

    fn size(self) -> usize {
        match self {
            Self::System => 0x336a0,
            Self::Private => 0xd8,
        }
    }

    /// The structure before any value is set: zeroes and the defaults of fields Lichii does
    /// not serve.
    fn default_bytes(self) -> Vec<u8> {
        let mut bytes = vec![0; self.size()];
        if self == Self::System {
            bytes[..4].copy_from_slice(&SYSTEM_VERSION.to_le_bytes());
            bytes[4..8].copy_from_slice(&SYSTEM_FLAGS.to_le_bytes());
            bytes[MII_AUTHOR_ID_OFFSET..][..16].copy_from_slice(super::system::MII_AUTHOR_ID);
        }
        bytes
    }
}

/// The two settings files of one NAND, as raw structures.
pub(super) struct SettingsFiles {
    nand: PathBuf,
    system: Vec<u8>,
    private: Vec<u8>,
    loaded: [bool; 2],
    /// Per part, whether its file may be written: not when it exists but could not be read.
    writable: [bool; 2],
}

impl SettingsFiles {
    /// Reads both files under `nand`. A missing or malformed one starts from the defaults; one
    /// that cannot be read starts from them too and is never written.
    pub(super) fn open(nand: PathBuf) -> Self {
        let read = |part: Part| match read_part(&nand, part) {
            Ok(bytes) => (bytes, true),
            Err(e) => {
                let path = nand.join(part.dir()).join(FILE_NAME);
                tracing::error!("reading {}: {e}; it is not saved", path.display());
                (None, false)
            }
        };
        let ((system, system_writable), (private, private_writable)) =
            (read(Part::System), read(Part::Private));
        Self {
            loaded: [system.is_some(), private.is_some()],
            writable: [system_writable, private_writable],
            system: system.unwrap_or_else(|| Part::System.default_bytes()),
            private: private.unwrap_or_else(|| Part::Private.default_bytes()),
            nand,
        }
    }

    /// Whether `part` came from its file rather than the defaults.
    pub(super) fn loaded(&self, part: Part) -> bool {
        self.loaded[part as usize]
    }

    pub(super) fn bytes(&self, part: Part) -> &[u8] {
        match part {
            Part::System => &self.system,
            Part::Private => &self.private,
        }
    }

    pub(super) fn bytes_mut(&mut self, part: Part) -> &mut [u8] {
        match part {
            Part::System => &mut self.system,
            Part::Private => &mut self.private,
        }
    }

    /// Writes both files, each to a temporary file renamed over the old one.
    pub(super) fn store(&self) {
        for part in [Part::System, Part::Private] {
            if !self.writable[part as usize] {
                continue;
            }
            let path = self.nand.join(part.dir()).join(FILE_NAME);
            let mut file = Vec::with_capacity(HEADER_SIZE + part.size());
            file.extend_from_slice(&MAGIC);
            file.extend_from_slice(&VERSION.to_le_bytes());
            file.extend_from_slice(&0u32.to_le_bytes());
            file.extend_from_slice(self.bytes(part));
            if let Err(e) = write_atomic(&path, &file) {
                tracing::error!("saving {}: {e}", path.display());
            }
        }
    }
}

/// The structure in `part`'s file, if it exists with the C++ header, version and size. An
/// error is a file that exists but cannot be read.
fn read_part(nand: &Path, part: Part) -> std::io::Result<Option<Vec<u8>>> {
    let path = nand.join(part.dir()).join(FILE_NAME);
    let file = match std::fs::read(&path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let valid = file.len() == HEADER_SIZE + part.size()
        && file[..8] == MAGIC
        && u32::from_le_bytes(file[8..12].try_into().unwrap()) >= VERSION;
    if !valid {
        tracing::warn!("{} is not a settings file; using defaults", path.display());
        return Ok(None);
    }
    Ok(Some(file[HEADER_SIZE..].to_vec()))
}
