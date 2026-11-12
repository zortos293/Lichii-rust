//! XCI game card images: the card header, the root HFS0 and its update/normal/secure/logo
//! partitions. The secure partition holds the game and is indexed like an NSP.

use lichii_crypto::KeySet;

use crate::bytes::{array_at, u32_at, u64_at};
use crate::content::ContentIndex;
use crate::error::{FsError, FsResult, corrupt};
use crate::pfs::{PartitionFs, PartitionKind};
use crate::vfs::{FileExt, FileRef, slice};

const FORMAT: &str = "XCI";
const HEADER_SIZE: usize = 0x200;
const KEY_AREA_SIZE: u64 = 0x1000;

/// The card header fields (offsets relative to the image without its key area).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct XciHeader {
    pub secure_area_start: u32,
    pub backup_area_start: u32,
    pub kek_index: u8,
    pub rom_size: u8,
    pub version: u8,
    pub flags: u8,
    pub package_id: u64,
    pub valid_data_end: u64,
    pub info_iv: [u8; 16],
    /// Offset of the root HFS0.
    pub root_offset: u64,
    /// Size of the root HFS0's header (the hashed part); its partitions follow it.
    pub root_header_size: u64,
    pub root_header_hash: [u8; 32],
    pub initial_data_hash: [u8; 32],
    pub secure_mode: u32,
    pub title_key_flag: u32,
    pub key_flag: u32,
    pub normal_area_end: u32,
}

impl XciHeader {
    fn parse(raw: &[u8]) -> Self {
        Self {
            secure_area_start: u32_at(raw, 0x104),
            backup_area_start: u32_at(raw, 0x108),
            kek_index: raw[0x10c],
            rom_size: raw[0x10d],
            version: raw[0x10e],
            flags: raw[0x10f],
            package_id: u64_at(raw, 0x110),
            valid_data_end: u64_at(raw, 0x118),
            info_iv: array_at(raw, 0x120),
            root_offset: u64_at(raw, 0x130),
            root_header_size: u64_at(raw, 0x138),
            root_header_hash: array_at(raw, 0x140),
            initial_data_hash: array_at(raw, 0x160),
            secure_mode: u32_at(raw, 0x180),
            title_key_flag: u32_at(raw, 0x184),
            key_flag: u32_at(raw, 0x188),
            normal_area_end: u32_at(raw, 0x18c),
        }
    }
}

/// The partitions of the root HFS0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XciPartition {
    Update,
    Normal,
    Secure,
    Logo,
}

impl XciPartition {
    pub const ALL: [Self; 4] = [Self::Update, Self::Normal, Self::Secure, Self::Logo];

    /// The partition's file name in the root HFS0.
    pub fn name(self) -> &'static str {
        match self {
            Self::Update => "update",
            Self::Normal => "normal",
            Self::Secure => "secure",
            Self::Logo => "logo",
        }
    }
}

/// An opened card image.
pub struct Xci {
    header: XciHeader,
    has_key_area: bool,
    root: PartitionFs,
    partitions: [Option<PartitionFs>; 4],
    index: ContentIndex,
}

fn has_card_magic(file: &FileRef, offset: u64) -> bool {
    file.read_array::<4>(offset + 0x100)
        .is_ok_and(|magic| &magic == b"HEAD")
}

impl Xci {
    /// Opens a card image, with or without the 0x1000-byte key area dumps sometimes keep in
    /// front of the header, imports the secure partition's tickets into `keys` and indexes its
    /// titles.
    pub fn open(file: &FileRef, keys: &mut KeySet) -> FsResult<Self> {
        let (image, has_key_area) = if has_card_magic(file, 0) {
            (file.clone(), false)
        } else if has_card_magic(file, KEY_AREA_SIZE) {
            (
                slice(file, KEY_AREA_SIZE, file.size() - KEY_AREA_SIZE),
                true,
            )
        } else {
            return Err(FsError::BadMagic {
                format: FORMAT,
                found: file.read_array(0x100).unwrap_or_default(),
            });
        };
        let raw: [u8; HEADER_SIZE] = image.read_array(0)?;
        let header = XciHeader::parse(&raw);
        if header.root_offset >= image.size() {
            return Err(corrupt(FORMAT, "root partition past the end of the image"));
        }
        let root = PartitionFs::open(&slice(
            &image,
            header.root_offset,
            image.size() - header.root_offset,
        ))?;
        if root.kind() != PartitionKind::Hfs0 {
            return Err(corrupt(FORMAT, "root partition is not an HFS0"));
        }
        let mut partitions: [Option<PartitionFs>; 4] = Default::default();
        for (slot, partition) in partitions.iter_mut().zip(XciPartition::ALL) {
            *slot = root
                .file(partition.name())
                .map(|file| PartitionFs::open(&file))
                .transpose()?;
        }
        let secure = partitions[XciPartition::Secure as usize]
            .as_ref()
            .ok_or_else(|| corrupt(FORMAT, "no secure partition"))?;
        let index = ContentIndex::from_dir(secure, keys)?;
        Ok(Self {
            header,
            has_key_area,
            root,
            partitions,
            index,
        })
    }

    pub fn header(&self) -> &XciHeader {
        &self.header
    }

    /// Whether the dump starts with the card's 0x1000-byte key area.
    pub fn has_key_area(&self) -> bool {
        self.has_key_area
    }

    pub fn root(&self) -> &PartitionFs {
        &self.root
    }

    pub fn partition(&self, partition: XciPartition) -> Option<&PartitionFs> {
        self.partitions[partition as usize].as_ref()
    }

    /// The titles of the secure partition.
    pub fn index(&self) -> &ContentIndex {
        &self.index
    }

    pub fn into_index(self) -> ContentIndex {
        self.index
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::content::tests::check_application;
    use crate::pfs::build_partition;
    use crate::testutil::*;
    use crate::vfs::{ConcatFile, ZeroFile};

    fn hfs0(files: &[(String, FileRef)]) -> FileRef {
        build_partition(PartitionKind::Hfs0, files).unwrap()
    }

    fn card(partitions: &[(&str, FileRef)], key_area: bool) -> FileRef {
        let named: Vec<_> = partitions
            .iter()
            .map(|(name, file)| ((*name).to_owned(), file.clone()))
            .collect();
        let root = hfs0(&named);
        let root_offset = 0x400u64;
        let mut header = vec![0u8; root_offset as usize];
        header[0x100..0x104].copy_from_slice(b"HEAD");
        header[0x10d] = 0xfa;
        header[0x110..0x118].copy_from_slice(&0x1234_5678_9abc_def0u64.to_le_bytes());
        header[0x130..0x138].copy_from_slice(&root_offset.to_le_bytes());
        let root_header = PartitionFs::open(&root).unwrap().header_size();
        header[0x138..0x140].copy_from_slice(&root_header.to_le_bytes());
        let mut parts: Vec<FileRef> = Vec::new();
        if key_area {
            parts.push(Arc::new(ZeroFile(KEY_AREA_SIZE)));
        }
        parts.push(file(header));
        parts.push(root);
        ConcatFile::from_parts(parts)
    }

    fn game_card(key_area: bool) -> FileRef {
        let empty = hfs0(&[]);
        card(
            &[
                ("update", empty.clone()),
                ("normal", empty),
                ("secure", hfs0(&application_package(true))),
            ],
            key_area,
        )
    }

    #[test]
    fn opens_cards_with_and_without_a_key_area() {
        for key_area in [false, true] {
            let mut keys = key_set();
            let xci = Xci::open(&game_card(key_area), &mut keys).unwrap();
            assert_eq!(xci.has_key_area(), key_area);
            assert_eq!(xci.header().root_offset, 0x400);
            assert_eq!(xci.header().package_id, 0x1234_5678_9abc_def0);
            assert_eq!(xci.header().rom_size, 0xfa);
            assert_eq!(xci.root().entries().len(), 3);
            assert!(
                xci.partition(XciPartition::Update)
                    .unwrap()
                    .entries()
                    .is_empty()
            );
            assert!(xci.partition(XciPartition::Logo).is_none());
            let secure = xci.partition(XciPartition::Secure).unwrap();
            assert_eq!(secure.kind(), PartitionKind::Hfs0);
            assert!(secure.entries().iter().all(|e| secure.verify(e).unwrap()));
            check_application(xci.index());
        }
    }

    #[test]
    fn rejects_broken_cards() {
        let mut keys = key_set();
        let image = game_card(false).read_all().unwrap();
        assert!(matches!(
            Xci::open(&file(vec![0u8; 0x2000]), &mut keys),
            Err(FsError::BadMagic { .. })
        ));
        for cut in [0, 0x103, 0x1ff, 0x400, 0x500, image.len() / 2] {
            assert!(Xci::open(&file(image[..cut].to_vec()), &mut keys).is_err());
        }
        let mut past_end = image.clone();
        past_end[0x130..0x138].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(Xci::open(&file(past_end), &mut keys).is_err());

        let empty = hfs0(&[]);
        let no_secure = card(&[("update", empty.clone()), ("normal", empty)], false);
        assert!(Xci::open(&no_secure, &mut keys).is_err());

        let named = vec![("secure".to_owned(), hfs0(&application_package(true)))];
        let pfs0_root = build_partition(PartitionKind::Pfs0, &named).unwrap();
        let mut header = vec![0u8; 0x400];
        header[0x100..0x104].copy_from_slice(b"HEAD");
        header[0x130..0x138].copy_from_slice(&0x400u64.to_le_bytes());
        let wrong_kind = ConcatFile::from_parts([file(header), pfs0_root]);
        assert!(Xci::open(&wrong_kind, &mut keys).is_err());
    }
}
