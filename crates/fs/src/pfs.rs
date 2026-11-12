//! PFS0 (ExeFS, NSP, meta sections) and HFS0 (game card partitions).

use std::io;

use lichii_crypto::Sha256Hash;

use crate::bytes::{array_at, u32_at, u64_at};
use crate::error::{FsResult, corrupt};
use crate::vfs::{
    ConcatFile, DirEntry, Directory, Entry, EntryKind, FileExt, FileRef, MemFile, slice,
};

const HEADER_SIZE: usize = 0x10;
const MAX_HEADER_SIZE: u64 = 64 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionKind {
    Pfs0,
    Hfs0,
}

impl PartitionKind {
    fn magic(self) -> &'static [u8; 4] {
        match self {
            Self::Pfs0 => b"PFS0",
            Self::Hfs0 => b"HFS0",
        }
    }

    fn entry_size(self) -> usize {
        match self {
            Self::Pfs0 => 0x18,
            Self::Hfs0 => 0x40,
        }
    }
}

/// The SHA-256 an HFS0 entry records over the first `hashed_size` bytes of its file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HfsHash {
    pub hashed_size: u32,
    pub sha256: Sha256Hash,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionEntry {
    pub name: String,
    /// Offset from the end of the header.
    pub offset: u64,
    pub size: u64,
    pub hash: Option<HfsHash>,
}

/// A parsed PFS0/HFS0. Files are views into the backing file.
pub struct PartitionFs {
    kind: PartitionKind,
    entries: Vec<PartitionEntry>,
    data: FileRef,
    header_size: u64,
}

impl PartitionFs {
    pub fn open(file: &FileRef) -> FsResult<Self> {
        let head: [u8; HEADER_SIZE] = file.read_array(0)?;
        let kind = match &head[..4] {
            b"PFS0" => PartitionKind::Pfs0,
            b"HFS0" => PartitionKind::Hfs0,
            _ => {
                return Err(crate::FsError::BadMagic {
                    format: "PFS0/HFS0",
                    found: array_at(&head, 0),
                });
            }
        };
        let format = if kind == PartitionKind::Pfs0 {
            "PFS0"
        } else {
            "HFS0"
        };
        let count = u64::from(u32_at(&head, 4));
        let strings = u64::from(u32_at(&head, 8));
        let header_size = HEADER_SIZE as u64 + count * kind.entry_size() as u64 + strings;
        if header_size > file.size() || header_size > MAX_HEADER_SIZE {
            return Err(corrupt(format, "header is larger than the file"));
        }
        let table = file.read_vec(
            HEADER_SIZE as u64,
            (header_size - HEADER_SIZE as u64) as usize,
        )?;
        let (records, names) = table.split_at(count as usize * kind.entry_size());
        let data_size = file.size() - header_size;
        let entries = records
            .chunks_exact(kind.entry_size())
            .map(|record| {
                let offset = u64_at(record, 0);
                let size = u64_at(record, 8);
                let name_offset = u32_at(record, 16) as usize;
                let name = names
                    .get(name_offset..)
                    .map(crate::bytes::c_string)
                    .ok_or_else(|| corrupt(format, "name offset past the string table"))?;
                if offset.checked_add(size).is_none_or(|end| end > data_size) {
                    return Err(corrupt(
                        format,
                        format!("{name} extends past the end of the file"),
                    ));
                }
                let hash = (kind == PartitionKind::Hfs0).then(|| HfsHash {
                    hashed_size: u32_at(record, 20),
                    sha256: array_at(record, 32),
                });
                Ok(PartitionEntry {
                    name,
                    offset,
                    size,
                    hash,
                })
            })
            .collect::<FsResult<Vec<_>>>()?;
        Ok(Self {
            kind,
            entries,
            data: slice(file, header_size, data_size),
            header_size,
        })
    }

    pub fn kind(&self) -> PartitionKind {
        self.kind
    }

    pub fn entries(&self) -> &[PartitionEntry] {
        &self.entries
    }

    /// Size of the header, which is where file data starts in the backing file.
    pub fn header_size(&self) -> u64 {
        self.header_size
    }

    pub fn file(&self, name: &str) -> Option<FileRef> {
        self.entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| self.entry_file(e))
    }

    pub fn entry_file(&self, entry: &PartitionEntry) -> FileRef {
        slice(&self.data, entry.offset, entry.size)
    }

    /// Checks an HFS0 entry's recorded hash. Always true for PFS0 entries.
    pub fn verify(&self, entry: &PartitionEntry) -> io::Result<bool> {
        let Some(hash) = entry.hash else {
            return Ok(true);
        };
        let len = u64::from(hash.hashed_size).min(entry.size) as usize;
        let data = self.entry_file(entry).read_vec(0, len)?;
        Ok(lichii_crypto::sha256(&data) == hash.sha256)
    }
}

impl Directory for PartitionFs {
    fn entries(&self) -> io::Result<Vec<DirEntry>> {
        Ok(self
            .entries
            .iter()
            .map(|e| DirEntry {
                name: e.name.clone(),
                kind: EntryKind::File,
                size: e.size,
            })
            .collect())
    }

    fn open(&self, name: &str) -> io::Result<Option<Entry>> {
        Ok(self.file(name).map(Entry::File))
    }
}

/// Lays out a PFS0/HFS0 image over `files` without copying them: the header is built in memory
/// and the files are concatenated behind it.
pub fn build_partition(kind: PartitionKind, files: &[(String, FileRef)]) -> io::Result<FileRef> {
    let mut names = Vec::new();
    let mut records = Vec::new();
    let mut offset = 0u64;
    for (name, file) in files {
        let mut record = vec![0u8; kind.entry_size()];
        record[0..8].copy_from_slice(&offset.to_le_bytes());
        record[8..16].copy_from_slice(&file.size().to_le_bytes());
        record[16..20].copy_from_slice(&(names.len() as u32).to_le_bytes());
        if kind == PartitionKind::Hfs0 {
            let hashed = file.size().min(0x200);
            record[20..24].copy_from_slice(&(hashed as u32).to_le_bytes());
            record[32..64]
                .copy_from_slice(&lichii_crypto::sha256(&file.read_vec(0, hashed as usize)?));
        }
        records.extend(record);
        names.extend(name.as_bytes());
        names.push(0);
        offset += file.size();
    }
    let unpadded = HEADER_SIZE + records.len() + names.len();
    names.resize(
        names.len() + (unpadded.next_multiple_of(0x20) - unpadded),
        0,
    );
    let mut header = Vec::with_capacity(HEADER_SIZE + records.len() + names.len());
    header.extend(kind.magic());
    header.extend((files.len() as u32).to_le_bytes());
    header.extend((names.len() as u32).to_le_bytes());
    header.extend([0; 4]);
    header.extend(records);
    header.extend(names);
    let parts = std::iter::once(MemFile::from_bytes(header))
        .chain(files.iter().map(|(_, file)| file.clone()));
    Ok(ConcatFile::from_parts(parts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vfs::DirectoryExt;

    fn files() -> Vec<(String, FileRef)> {
        vec![
            (
                "main".to_owned(),
                MemFile::from_bytes((0..=255u8).cycle().take(0x300).collect::<Vec<_>>()),
            ),
            ("main.npdm".to_owned(), MemFile::from_bytes(vec![7u8; 5])),
            ("empty".to_owned(), MemFile::from_bytes(Vec::new())),
        ]
    }

    #[test]
    fn round_trips_both_kinds() {
        for kind in [PartitionKind::Pfs0, PartitionKind::Hfs0] {
            let image = build_partition(kind, &files()).unwrap();
            let pfs = PartitionFs::open(&image).unwrap();
            assert_eq!(pfs.kind(), kind);
            assert_eq!(pfs.header_size() % 0x20, 0);
            let names: Vec<_> = pfs.entries().iter().map(|e| e.name.as_str()).collect();
            assert_eq!(names, ["main", "main.npdm", "empty"]);
            for (name, file) in files() {
                let read = pfs.open_file(&name).unwrap().unwrap();
                assert_eq!(read.read_all().unwrap(), file.read_all().unwrap(), "{name}");
            }
            for entry in pfs.entries() {
                assert!(pfs.verify(entry).unwrap());
                assert_eq!(entry.hash.is_some(), kind == PartitionKind::Hfs0);
            }
            assert!(pfs.file("missing").is_none());
        }
    }

    #[test]
    fn hfs0_detects_bad_hash_and_rejects_corrupt_headers() {
        let image = build_partition(PartitionKind::Hfs0, &files()).unwrap();
        let mut bytes = image.read_all().unwrap();
        let header_size = PartitionFs::open(&image).unwrap().header_size() as usize;
        bytes[header_size] ^= 1;
        let pfs = PartitionFs::open(&MemFile::from_bytes(bytes.clone())).unwrap();
        assert!(!pfs.verify(&pfs.entries()[0]).unwrap());

        let mut huge = bytes.clone();
        huge[4..8].copy_from_slice(&0x1000_0000u32.to_le_bytes());
        assert!(PartitionFs::open(&MemFile::from_bytes(huge)).is_err());

        let mut past_end = bytes.clone();
        past_end[0x18..0x20].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(PartitionFs::open(&MemFile::from_bytes(past_end)).is_err());

        assert!(
            PartitionFs::open(&MemFile::from_bytes(
                b"NOPE\0\0\0\0\0\0\0\0\0\0\0\0".to_vec()
            ))
            .is_err()
        );
        assert!(PartitionFs::open(&MemFile::from_bytes(vec![0u8; 3])).is_err());
        assert_eq!(image.size(), bytes.len() as u64);
    }
}
