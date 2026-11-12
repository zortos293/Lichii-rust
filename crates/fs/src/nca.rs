//! NCA content archives: header decryption (NCA3, NCA2 and plaintext `DNCA`), section keys
//! from the key area or a title key, decrypting section views, hash layers, compressed and
//! sparse sections, and the ExeFS/RomFS/logo sections a loader needs.
//!
//! Section files read straight from the backing file into the caller's buffer and decrypt in
//! place; a section is never read whole.

use std::fmt;
use std::io;
use std::sync::Arc;

use lichii_crypto::{
    AesCtr, AesEcb, AesXts, Key128, Key256, KeyAreaKeyType, KeyError, KeySet, RightsId, Sha256Hash,
    XtsTweak, master_key_revision, sha256,
};

use crate::bktr::{AesCtrExFile, BucketTable, CompressedFile, IndirectFile, bucket_entry_count};
use crate::bytes::{array_at, u16_at, u32_at, u64_at};
use crate::error::{FsError, FsResult, corrupt};
use crate::pfs::PartitionFs;
use crate::vfs::{File, FileExt, FileRef, ZeroFile, slice};

const FORMAT: &str = "NCA";
/// Size of the encrypted NCA header: the archive header and four FS headers.
pub const HEADER_SIZE: usize = 0xC00;
/// Number of section slots an NCA header has.
pub const MAX_SECTIONS: usize = 4;
const SECTOR_SIZE: usize = 0x200;
const SECTOR: u64 = SECTOR_SIZE as u64;
const FS_HEADERS: usize = 0x400;
const MAX_VERIFY_BLOCK: u64 = 16 << 20;
const MAX_HASH_LEVEL: u64 = 256 << 20;

/// Which header format the archive uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaMagic {
    /// `NCA2`: each FS header is encrypted on its own as XTS sector 0.
    Nca2,
    /// `NCA3`: the whole 0xC00-byte header is XTS sectors 0 to 5.
    Nca3,
    /// `DNCA`: header and sections stored decrypted by a converter.
    Decrypted,
}

/// Where the content was distributed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DistributionType {
    Download,
    GameCard,
    Unknown(u8),
}

/// The NCA header's content type (numbered differently from CNMT content records).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NcaContentType {
    Program,
    Meta,
    Control,
    Manual,
    Data,
    PublicData,
    Unknown(u8),
}

/// What a section holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FsType {
    RomFs,
    PartitionFs,
    Unknown(u8),
}

/// How a section's data is hashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashType {
    Auto,
    None,
    HierarchicalSha256,
    HierarchicalIntegrity,
    AutoSha3,
    HierarchicalSha3256,
    HierarchicalIntegritySha3,
    Unknown(u8),
}

/// How a section is encrypted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncryptionType {
    Auto,
    None,
    AesXts,
    AesCtr,
    AesCtrEx,
    AesCtrSkipLayerHash,
    AesCtrExSkipLayerHash,
    Unknown(u8),
}

impl DistributionType {
    fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Download,
            1 => Self::GameCard,
            other => Self::Unknown(other),
        }
    }
}

impl NcaContentType {
    fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Program,
            1 => Self::Meta,
            2 => Self::Control,
            3 => Self::Manual,
            4 => Self::Data,
            5 => Self::PublicData,
            other => Self::Unknown(other),
        }
    }
}

impl FsType {
    fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::RomFs,
            1 => Self::PartitionFs,
            other => Self::Unknown(other),
        }
    }
}

impl HashType {
    fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Auto,
            1 => Self::None,
            2 => Self::HierarchicalSha256,
            3 => Self::HierarchicalIntegrity,
            4 => Self::AutoSha3,
            5 => Self::HierarchicalSha3256,
            6 => Self::HierarchicalIntegritySha3,
            other => Self::Unknown(other),
        }
    }
}

impl EncryptionType {
    fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Auto,
            1 => Self::None,
            2 => Self::AesXts,
            3 => Self::AesCtr,
            4 => Self::AesCtrEx,
            5 => Self::AesCtrSkipLayerHash,
            6 => Self::AesCtrExSkipLayerHash,
            other => Self::Unknown(other),
        }
    }
}

/// A byte range inside a section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region {
    pub offset: u64,
    pub size: u64,
}

/// One level of an IVFC (hierarchical integrity) tree; the last level is the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegrityLevel {
    pub offset: u64,
    pub size: u64,
    pub block_order: u32,
}

/// A section's hash tree description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HashData {
    /// PFS0 sections: SHA-256 (or SHA3-256) hash layers; the last layer is the data.
    HierarchicalSha256 {
        master_hash: Sha256Hash,
        block_size: u32,
        layers: Vec<Region>,
    },
    /// RomFS sections: IVFC levels; the last level is the data.
    HierarchicalIntegrity {
        master_hash: Sha256Hash,
        levels: Vec<IntegrityLevel>,
    },
    None,
}

/// Where a bucket tree table lives in a section, and its `BKTR` header.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BucketInfo {
    pub offset: u64,
    pub size: u64,
    pub header: [u8; 16],
}

/// The BKTR tables of an update section.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PatchInfo {
    pub indirect: BucketInfo,
    pub aes_ctr_ex: BucketInfo,
}

/// The upper half of a section's AES-CTR counter (FS header 0x140).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AesCtrUpperIv {
    pub generation: u32,
    pub secure_value: u32,
}

impl AesCtrUpperIv {
    /// The counter block for archive offset 0: the upper half big-endian, the lower half zero.
    /// [`AesCtr::apply`] adds `offset / 16` for each byte.
    pub fn iv(self) -> [u8; 16] {
        let upper = (u64::from(self.secure_value) << 32) | u64::from(self.generation);
        let mut iv = [0u8; 16];
        iv[..8].copy_from_slice(&upper.to_be_bytes());
        iv
    }
}

/// The sparse layer of a section: only some of its virtual range is stored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SparseInfo {
    pub bucket: BucketInfo,
    pub physical_offset: u64,
    pub generation: u16,
}

/// A parsed FS header (one per section).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FsHeader {
    pub version: u16,
    pub fs_type: FsType,
    pub hash_type: HashType,
    pub encryption_type: EncryptionType,
    pub meta_data_hash_type: u8,
    pub hash_data: HashData,
    pub patch_info: PatchInfo,
    pub upper_iv: AesCtrUpperIv,
    pub sparse_info: SparseInfo,
    pub compression_info: BucketInfo,
}

fn bucket_at(raw: &[u8], at: usize) -> BucketInfo {
    BucketInfo {
        offset: u64_at(raw, at),
        size: u64_at(raw, at + 8),
        header: array_at(raw, at + 16),
    }
}

impl FsHeader {
    fn parse(raw: &[u8], index: usize) -> FsResult<Self> {
        let bad = |reason: String| corrupt(FORMAT, format!("section {index}: {reason}"));
        let hash_type = HashType::from_raw(raw[3]);
        let hash_data = match hash_type {
            HashType::HierarchicalSha256 | HashType::HierarchicalSha3256 => {
                let block_size = u32_at(raw, 0x28);
                let count = u32_at(raw, 0x2c) as usize;
                if !(1..=5).contains(&count) {
                    return Err(bad(format!("{count} hash layers")));
                }
                if !block_size.is_power_of_two() || block_size < 0x20 {
                    return Err(bad(format!("hash block size {block_size:#x}")));
                }
                HashData::HierarchicalSha256 {
                    master_hash: array_at(raw, 0x08),
                    block_size,
                    layers: (0..count)
                        .map(|i| Region {
                            offset: u64_at(raw, 0x30 + i * 0x10),
                            size: u64_at(raw, 0x38 + i * 0x10),
                        })
                        .collect(),
                }
            }
            HashType::HierarchicalIntegrity | HashType::HierarchicalIntegritySha3 => {
                if &raw[0x08..0x0c] != b"IVFC" {
                    return Err(FsError::BadMagic {
                        format: "IVFC",
                        found: array_at(raw, 0x08),
                    });
                }
                let max_layers = u32_at(raw, 0x14) as usize;
                if !(2..=7).contains(&max_layers) {
                    return Err(bad(format!("{max_layers} IVFC layers")));
                }
                let levels = (0..max_layers - 1)
                    .map(|i| {
                        let at = 0x18 + i * 0x18;
                        let level = IntegrityLevel {
                            offset: u64_at(raw, at),
                            size: u64_at(raw, at + 8),
                            block_order: u32_at(raw, at + 16),
                        };
                        if !(5..32).contains(&level.block_order) {
                            return Err(bad(format!("IVFC block order {}", level.block_order)));
                        }
                        Ok(level)
                    })
                    .collect::<FsResult<_>>()?;
                HashData::HierarchicalIntegrity {
                    master_hash: array_at(raw, 0xc8),
                    levels,
                }
            }
            _ => HashData::None,
        };
        Ok(Self {
            version: u16_at(raw, 0),
            fs_type: FsType::from_raw(raw[2]),
            hash_type,
            encryption_type: EncryptionType::from_raw(raw[4]),
            meta_data_hash_type: raw[5],
            hash_data,
            patch_info: PatchInfo {
                indirect: bucket_at(raw, 0x100),
                aes_ctr_ex: bucket_at(raw, 0x120),
            },
            upper_iv: AesCtrUpperIv {
                generation: u32_at(raw, 0x140),
                secure_value: u32_at(raw, 0x144),
            },
            sparse_info: SparseInfo {
                bucket: bucket_at(raw, 0x148),
                physical_offset: u64_at(raw, 0x168),
                generation: u16_at(raw, 0x170),
            },
            compression_info: bucket_at(raw, 0x178),
        })
    }

    /// Whether this is an update section that patches a base NCA's section (BKTR).
    pub fn is_patch(&self) -> bool {
        self.patch_info.indirect.size != 0
    }

    /// The data region inside the decrypted section: the last hash layer or IVFC level. `None`
    /// for unhashed sections, whose data is the whole section.
    pub fn hash_target(&self) -> FsResult<Option<Region>> {
        match &self.hash_data {
            HashData::HierarchicalSha256 { layers, .. } => Ok(layers.last().copied()),
            HashData::HierarchicalIntegrity { levels, .. } => Ok(levels.last().map(|l| Region {
                offset: l.offset,
                size: l.size,
            })),
            HashData::None if self.hash_type == HashType::None => Ok(None),
            HashData::None => Err(corrupt(FORMAT, format!("hash type {:?}", self.hash_type))),
        }
    }
}

/// The fields of the archive header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcaHeader {
    pub magic: NcaMagic,
    /// False when the header was stored decrypted (`DNCA`, or a plaintext NCA3/NCA2 header).
    pub header_encrypted: bool,
    pub distribution_type: DistributionType,
    pub content_type: NcaContentType,
    /// The larger of the two key generation fields.
    pub key_generation: u8,
    pub key_index: u8,
    pub content_size: u64,
    pub program_id: u64,
    pub content_index: u32,
    pub sdk_addon_version: u32,
    pub signature_key_generation: u8,
    pub rights_id: RightsId,
}

/// One present section of an NCA.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NcaSection {
    /// The FS entry slot (0-3).
    pub index: usize,
    /// Offset of the section in the NCA file.
    pub offset: u64,
    pub size: u64,
    pub header: FsHeader,
    /// Whether the archive header's SHA-256 of this FS header matches.
    pub header_hash_valid: bool,
}

#[derive(Clone)]
struct ContentKeys {
    ctr: Key128,
    xts: Option<Key256>,
}

/// An opened NCA. Content keys are resolved when it is opened; a missing key is reported (as
/// [`FsError::Key`], naming the `prod.keys` entry or rights id) when an encrypted section is
/// opened, so the header of an NCA whose keys are missing can still be inspected.
#[derive(Clone)]
pub struct Nca {
    file: FileRef,
    header: NcaHeader,
    sections: [Option<NcaSection>; MAX_SECTIONS],
    keys: Result<ContentKeys, KeyError>,
}

impl fmt::Debug for Nca {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nca")
            .field("header", &self.header)
            .field("sections", &self.sections)
            .finish_non_exhaustive()
    }
}

fn magic_of(raw: &[u8]) -> Option<NcaMagic> {
    match &raw[0x200..0x204] {
        b"NCA3" => Some(NcaMagic::Nca3),
        b"NCA2" => Some(NcaMagic::Nca2),
        b"DNCA" => Some(NcaMagic::Decrypted),
        _ => None,
    }
}

fn decrypt_header(raw: &mut [u8], key: &Key256) {
    let xts = AesXts::new(key, XtsTweak::Nintendo);
    xts.decrypt(&mut raw[..FS_HEADERS], SECTOR_SIZE, 0);
    if magic_of(raw) == Some(NcaMagic::Nca2) {
        for fs_header in raw[FS_HEADERS..].as_chunks_mut::<SECTOR_SIZE>().0 {
            xts.decrypt(fs_header, SECTOR_SIZE, 0);
        }
    } else {
        xts.decrypt(&mut raw[FS_HEADERS..], SECTOR_SIZE, 2);
    }
}

fn content_keys(
    header: &NcaHeader,
    key_area: &[u8],
    keys: &KeySet,
) -> FsResult<Result<ContentKeys, KeyError>> {
    let revision = master_key_revision(header.key_generation);
    if !header.rights_id.is_zero() {
        return Ok(keys
            .title_key(&header.rights_id, revision)
            .map(|ctr| ContentKeys { ctr, xts: None }));
    }
    let kind = KeyAreaKeyType::from_index(header.key_index)
        .ok_or_else(|| corrupt(FORMAT, format!("key area key index {}", header.key_index)))?;
    Ok(keys.key_area_key(revision, kind).map(|kek| {
        let mut area: [u8; 0x40] = key_area.try_into().unwrap();
        AesEcb::new(&kek).decrypt(&mut area);
        ContentKeys {
            ctr: array_at(&area, 0x20),
            xts: Some(array_at(&area, 0)),
        }
    }))
}

fn within(region: Region, size: u64) -> bool {
    region
        .offset
        .checked_add(region.size)
        .is_some_and(|end| end <= size)
}

impl Nca {
    /// Parses the header of `file` and resolves its content keys from `keys`. Fails when the
    /// header cannot be decrypted (`header_key` missing or wrong) or is malformed.
    pub fn open(file: FileRef, keys: &KeySet) -> FsResult<Self> {
        if file.size() < HEADER_SIZE as u64 {
            return Err(corrupt(FORMAT, "shorter than its header"));
        }
        let mut raw = vec![0u8; HEADER_SIZE];
        file.read_exact_at(0, &mut raw)?;
        let header_encrypted = magic_of(&raw).is_none();
        if header_encrypted {
            decrypt_header(&mut raw, keys.header_key()?);
        }
        let magic = magic_of(&raw).ok_or_else(|| match &raw[0x200..0x204] {
            b"NCA0" | b"NCA1" => FsError::Unsupported(
                "NCA0/NCA1 archives (pre-release SDK) are not supported".into(),
            ),
            _ => FsError::BadMagic {
                format: "NCA (wrong header_key?)",
                found: array_at(&raw, 0x200),
            },
        })?;
        let header = NcaHeader {
            magic,
            header_encrypted,
            distribution_type: DistributionType::from_raw(raw[0x204]),
            content_type: NcaContentType::from_raw(raw[0x205]),
            key_generation: raw[0x206].max(raw[0x220]),
            key_index: raw[0x207],
            content_size: u64_at(&raw, 0x208),
            program_id: u64_at(&raw, 0x210),
            content_index: u32_at(&raw, 0x218),
            sdk_addon_version: u32_at(&raw, 0x21c),
            signature_key_generation: raw[0x221],
            rights_id: RightsId(array_at(&raw, 0x230)),
        };

        let mut sections: [Option<NcaSection>; MAX_SECTIONS] = Default::default();
        for (index, slot) in sections.iter_mut().enumerate() {
            let start = u32_at(&raw, 0x240 + index * 0x10);
            let end = u32_at(&raw, 0x244 + index * 0x10);
            if start == 0 && end == 0 {
                continue;
            }
            if end <= start {
                return Err(corrupt(
                    FORMAT,
                    format!("section {index} ends before it starts"),
                ));
            }
            let fs_raw = &raw[FS_HEADERS + index * SECTOR_SIZE..][..SECTOR_SIZE];
            *slot = Some(NcaSection {
                index,
                offset: u64::from(start) * SECTOR,
                size: u64::from(end - start) * SECTOR,
                header: FsHeader::parse(fs_raw, index)?,
                header_hash_valid: sha256(fs_raw) == array_at::<32>(&raw, 0x280 + index * 0x20),
            });
        }

        let keys = if magic == NcaMagic::Decrypted {
            Ok(ContentKeys {
                ctr: [0; 16],
                xts: None,
            })
        } else {
            content_keys(&header, &raw[0x300..0x340], keys)?
        };
        Ok(Self {
            file,
            header,
            sections,
            keys,
        })
    }

    pub fn header(&self) -> &NcaHeader {
        &self.header
    }

    /// The backing file.
    pub fn file(&self) -> &FileRef {
        &self.file
    }

    /// The present sections, in slot order.
    pub fn sections(&self) -> impl Iterator<Item = &NcaSection> {
        self.sections.iter().flatten()
    }

    pub fn section(&self, index: usize) -> Option<&NcaSection> {
        self.sections.get(index)?.as_ref()
    }

    /// Whether any section patches a base NCA (an update's Program NCA).
    pub fn is_patch(&self) -> bool {
        self.sections().any(|s| s.header.is_patch())
    }

    /// The decrypted section as stored: hash layers and data, after sparse expansion and before
    /// BKTR patching.
    pub fn raw_section_file(&self, index: usize) -> FsResult<FileRef> {
        self.decrypted_section(self.require(index)?)
    }

    /// The section's data (the hash target, decompressed if the section is compressed): a
    /// PFS0 or RomFS image. Update sections need their base; see [`Nca::patched_section_file`].
    pub fn section_file(&self, index: usize) -> FsResult<FileRef> {
        let section = self.require(index)?;
        if section.header.is_patch() {
            return Err(FsError::Unsupported(format!(
                "NCA section {index} is an update (BKTR) section; open it over its base NCA \
                 with open_patched_romfs"
            )));
        }
        self.data_view(section, self.decrypted_section(section)?)
    }

    /// Section `index` of this update NCA patched over the matching section of `base`. A
    /// section without patch tables is returned as [`Nca::section_file`] returns it.
    pub fn patched_section_file(&self, index: usize, base: &Nca) -> FsResult<FileRef> {
        let section = self.require(index)?;
        let indirect = &section.header.patch_info.indirect;
        if indirect.size == 0 {
            return self.section_file(index);
        }
        let patch = self.decrypted_section(section)?;
        let table_region = Region {
            offset: indirect.offset,
            size: indirect.size,
        };
        if !indirect.offset.is_multiple_of(SECTOR) || !within(table_region, patch.size()) {
            return Err(corrupt(FORMAT, "indirect table outside the patch data"));
        }
        let meta = slice(&patch, indirect.offset, indirect.size);
        let table = BucketTable::parse("indirect storage", &indirect.header, &meta)?;
        let fs_type = section.header.fs_type;
        let original = base
            .section(index)
            .filter(|s| s.header.fs_type == fs_type)
            .or_else(|| base.sections().find(|s| s.header.fs_type == fs_type))
            .ok_or_else(|| corrupt(FORMAT, "base NCA has no section to patch"))?;
        if original.header.is_patch() {
            return Err(corrupt(FORMAT, "base NCA section is itself an update"));
        }
        let storages = [
            base.decrypted_section(original)?,
            slice(&patch, 0, indirect.offset),
        ];
        self.data_view(section, Arc::new(IndirectFile::new(table, storages)))
    }

    /// Section `index` opened as a PFS0.
    pub fn section_pfs(&self, index: usize) -> FsResult<PartitionFs> {
        PartitionFs::open(&self.section_file(index)?)
    }

    /// The ExeFS: the PFS0 section holding `main.npdm`.
    pub fn exefs(&self) -> FsResult<Option<PartitionFs>> {
        self.find_partition(|pfs| pfs.file("main.npdm").is_some())
    }

    /// The logo partition: the PFS0 section holding `NintendoLogo.png`/`StartupMovie.gif`.
    pub fn logo(&self) -> FsResult<Option<PartitionFs>> {
        self.find_partition(|pfs| {
            pfs.file("NintendoLogo.png").is_some() || pfs.file("StartupMovie.gif").is_some()
        })
    }

    /// The raw RomFS image of the first RomFS section (what fsp-srv serves as an IStorage).
    /// Fails for update NCAs, whose RomFS needs [`crate::bktr::open_patched_romfs`].
    pub fn romfs(&self) -> FsResult<Option<FileRef>> {
        self.sections()
            .find(|s| s.header.fs_type == FsType::RomFs)
            .map(|s| self.section_file(s.index))
            .transpose()
    }

    /// Checks section `index` against the archive header's hash of its FS header and its own
    /// hash tree. Reads the whole section.
    pub fn verify_section(&self, index: usize) -> FsResult<bool> {
        let section = self.require(index)?;
        if !section.header_hash_valid {
            return Ok(false);
        }
        if section.header.is_patch() {
            return Err(FsError::Unsupported(
                "verifying update (BKTR) sections".into(),
            ));
        }
        if matches!(
            section.header.hash_type,
            HashType::HierarchicalSha3256 | HashType::HierarchicalIntegritySha3
        ) {
            return Err(FsError::Unsupported("SHA3-256 hash verification".into()));
        }
        let raw = self.decrypted_section(section)?;
        let level = |region: Region| -> FsResult<FileRef> {
            if !within(region, raw.size()) {
                return Err(corrupt(FORMAT, "hash layer past the end of the section"));
            }
            Ok(slice(&raw, region.offset, region.size))
        };
        match &section.header.hash_data {
            HashData::HierarchicalSha256 {
                master_hash,
                block_size,
                layers,
            } => {
                let mut hashes = read_level(&level(layers[0])?)?;
                if sha256(&hashes) != *master_hash {
                    return Ok(false);
                }
                for (position, layer) in layers.iter().enumerate().skip(1) {
                    let data = level(*layer)?;
                    if !hash_blocks(&data, &hashes, u64::from(*block_size), false)? {
                        return Ok(false);
                    }
                    if position + 1 < layers.len() {
                        hashes = read_level(&data)?;
                    }
                }
                Ok(true)
            }
            HashData::HierarchicalIntegrity {
                master_hash,
                levels,
            } => {
                let mut hashes = master_hash.to_vec();
                for (position, info) in levels.iter().enumerate() {
                    let data = level(Region {
                        offset: info.offset,
                        size: info.size,
                    })?;
                    if !hash_blocks(&data, &hashes, 1 << info.block_order, true)? {
                        return Ok(false);
                    }
                    if position + 1 < levels.len() {
                        hashes = read_level(&data)?;
                    }
                }
                Ok(true)
            }
            HashData::None => Ok(true),
        }
    }

    fn find_partition(
        &self,
        matches: impl Fn(&PartitionFs) -> bool,
    ) -> FsResult<Option<PartitionFs>> {
        for section in self
            .sections()
            .filter(|s| s.header.fs_type == FsType::PartitionFs)
        {
            let pfs = self.section_pfs(section.index)?;
            if matches(&pfs) {
                return Ok(Some(pfs));
            }
        }
        Ok(None)
    }

    fn require(&self, index: usize) -> FsResult<&NcaSection> {
        self.section(index)
            .ok_or_else(|| corrupt(FORMAT, format!("no section {index}")))
    }

    fn content_keys(&self) -> FsResult<&ContentKeys> {
        self.keys.as_ref().map_err(|e| FsError::Key(e.clone()))
    }

    fn body(&self, offset: u64, size: u64) -> FsResult<FileRef> {
        if !within(Region { offset, size }, self.file.size()) {
            return Err(corrupt(FORMAT, "section extends past the end of the file"));
        }
        Ok(slice(&self.file, offset, size))
    }

    fn ctr_file(
        &self,
        inner: FileRef,
        upper_iv: AesCtrUpperIv,
        counter_offset: u64,
        plain_prefix: u64,
    ) -> FsResult<FileRef> {
        Ok(Arc::new(CtrFile {
            inner,
            cipher: AesCtr::new(&self.content_keys()?.ctr),
            iv: upper_iv.iv(),
            counter_offset,
            plain_prefix,
        }))
    }

    pub(crate) fn decrypted_section(&self, section: &NcaSection) -> FsResult<FileRef> {
        let header = &section.header;
        let body = if header.sparse_info.generation != 0 {
            self.sparse_body(section)?
        } else {
            self.body(section.offset, section.size)?
        };
        let encryption = if self.header.magic == NcaMagic::Decrypted {
            EncryptionType::None
        } else {
            header.encryption_type
        };
        if header.patch_info.aes_ctr_ex.size != 0 {
            return self.aes_ctr_ex_section(section, body, encryption);
        }
        match encryption {
            EncryptionType::None => Ok(body),
            EncryptionType::AesCtr => self.ctr_file(body, header.upper_iv, section.offset, 0),
            EncryptionType::AesCtrSkipLayerHash => {
                let hashed = header.hash_target()?.map_or(0, |data| data.offset);
                self.ctr_file(body, header.upper_iv, section.offset, hashed)
            }
            EncryptionType::AesXts => {
                let key = self.content_keys()?.xts.ok_or_else(|| {
                    FsError::Unsupported("AES-XTS section in an NCA with a rights id".into())
                })?;
                Ok(Arc::new(XtsFile {
                    inner: body,
                    xts: AesXts::new(&key, XtsTweak::Nintendo),
                    first_sector: section.offset / SECTOR,
                }))
            }
            other => Err(corrupt(
                FORMAT,
                format!("section {} encryption type {other:?}", section.index),
            )),
        }
    }

    fn sparse_body(&self, section: &NcaSection) -> FsResult<FileRef> {
        let sparse = &section.header.sparse_info;
        if self.header.magic == NcaMagic::Decrypted {
            return Err(corrupt(FORMAT, "sparse section in a decrypted NCA"));
        }
        if bucket_entry_count("sparse storage", &sparse.bucket.header)? == 0 {
            return Ok(Arc::new(ZeroFile(section.size)));
        }
        let physical_size = sparse
            .bucket
            .offset
            .checked_add(sparse.bucket.size)
            .ok_or_else(|| corrupt(FORMAT, "sparse table past the end of the file"))?;
        let physical = self.body(sparse.physical_offset, physical_size)?;
        let meta_iv = AesCtrUpperIv {
            generation: u32::from(sparse.generation) << 16,
            ..section.header.upper_iv
        };
        let meta = self.ctr_file(
            slice(&physical, sparse.bucket.offset, sparse.bucket.size),
            meta_iv,
            sparse.physical_offset + sparse.bucket.offset,
            0,
        )?;
        let table = BucketTable::parse("sparse storage", &sparse.bucket.header, &meta)?;
        Ok(Arc::new(IndirectFile::new(
            table,
            [physical, Arc::new(ZeroFile(u64::MAX))],
        )))
    }

    fn aes_ctr_ex_section(
        &self,
        section: &NcaSection,
        body: FileRef,
        encryption: EncryptionType,
    ) -> FsResult<FileRef> {
        let PatchInfo {
            indirect,
            aes_ctr_ex,
        } = section.header.patch_info;
        let indirect_end = indirect.offset.checked_add(indirect.size);
        if indirect.size == 0 || indirect_end.is_none_or(|end| end > aes_ctr_ex.offset) {
            return Err(corrupt(FORMAT, "patch tables overlap or are missing"));
        }
        let table_size = aes_ctr_ex.size.checked_next_multiple_of(SECTOR);
        let table_region = table_size.map(|size| Region {
            offset: aes_ctr_ex.offset,
            size,
        });
        if !table_region.is_some_and(|region| within(region, body.size())) {
            return Err(corrupt(
                FORMAT,
                "AesCtrEx table past the end of the section",
            ));
        }
        if encryption == EncryptionType::None {
            return Ok(slice(&body, 0, aes_ctr_ex.offset));
        }
        if !matches!(
            encryption,
            EncryptionType::AesCtrEx | EncryptionType::AesCtrExSkipLayerHash
        ) {
            return Err(corrupt(
                FORMAT,
                format!("AesCtrEx table in a section encrypted with {encryption:?}"),
            ));
        }
        let upper_iv = section.header.upper_iv;
        let meta = self.ctr_file(
            slice(&body, aes_ctr_ex.offset, table_size.unwrap()),
            upper_iv,
            section.offset + aes_ctr_ex.offset,
            0,
        )?;
        let table = BucketTable::parse("AesCtrEx storage", &aes_ctr_ex.header, &meta)?;
        Ok(Arc::new(AesCtrExFile::new(
            slice(&body, 0, aes_ctr_ex.offset),
            table,
            AesCtr::new(&self.content_keys()?.ctr),
            upper_iv.secure_value,
            section.offset,
        )?))
    }

    fn data_view(&self, section: &NcaSection, raw: FileRef) -> FsResult<FileRef> {
        let header = &section.header;
        let data = match header.hash_target()? {
            Some(region) if within(region, raw.size()) => slice(&raw, region.offset, region.size),
            Some(_) => {
                return Err(corrupt(
                    FORMAT,
                    format!("section {} data past the end of the section", section.index),
                ));
            }
            None => raw,
        };
        let compression = &header.compression_info;
        if compression.offset == 0 || compression.size == 0 {
            return Ok(data);
        }
        let table_region = Region {
            offset: compression.offset,
            size: compression.size,
        };
        if !within(table_region, data.size()) {
            return Err(corrupt(
                FORMAT,
                "compression table past the end of the data",
            ));
        }
        let meta = slice(&data, compression.offset, compression.size);
        let table = BucketTable::parse("compressed storage", &compression.header, &meta)?;
        Ok(Arc::new(CompressedFile::new(
            slice(&data, 0, compression.offset),
            table,
        )?))
    }
}

fn read_level(file: &FileRef) -> FsResult<Vec<u8>> {
    if file.size() > MAX_HASH_LEVEL {
        return Err(corrupt(FORMAT, "hash level is too large"));
    }
    Ok(file.read_all()?)
}

fn hash_blocks(data: &FileRef, hashes: &[u8], block_size: u64, pad: bool) -> FsResult<bool> {
    if block_size > MAX_VERIFY_BLOCK {
        return Err(FsError::Unsupported(format!(
            "verifying hash blocks of {block_size:#x} bytes"
        )));
    }
    let blocks = data.size().div_ceil(block_size);
    if blocks.saturating_mul(32) > hashes.len() as u64 {
        return Ok(false);
    }
    let mut block = vec![0u8; block_size as usize];
    for (index, expected) in hashes
        .as_chunks::<32>()
        .0
        .iter()
        .take(blocks as usize)
        .enumerate()
    {
        let start = index as u64 * block_size;
        let len = (data.size() - start).min(block_size) as usize;
        data.read_exact_at(start, &mut block[..len])?;
        let hashed = if pad {
            block[len..].fill(0);
            &block[..]
        } else {
            &block[..len]
        };
        if sha256(hashed) != *expected {
            return Ok(false);
        }
    }
    Ok(true)
}

/// AES-CTR over a section: counters are `iv + (counter_offset + offset) / 16`, so they count
/// from the start of the NCA. The first `plain_prefix` bytes (skip-layer-hash sections) are
/// stored unencrypted.
struct CtrFile {
    inner: FileRef,
    cipher: AesCtr,
    iv: [u8; 16],
    counter_offset: u64,
    plain_prefix: u64,
}

impl File for CtrFile {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read_at(offset, buf)?;
        let plain = (self.plain_prefix.saturating_sub(offset) as usize).min(read);
        self.cipher.apply(
            &self.iv,
            self.counter_offset + offset + plain as u64,
            &mut buf[plain..read],
        );
        Ok(read)
    }
}

/// AES-XTS over a section, 0x200-byte sectors numbered from the start of the NCA. Whole sectors
/// decrypt in the caller's buffer; partial ones at the edges of a read go through one stack
/// sector.
struct XtsFile {
    inner: FileRef,
    xts: AesXts,
    first_sector: u64,
}

impl File for XtsFile {
    fn size(&self) -> u64 {
        self.inner.size()
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
        let size = self.size();
        let len = buf.len().min(size.saturating_sub(offset) as usize);
        let mut done = 0;
        while done < len {
            let position = offset + done as u64;
            let sector = position / SECTOR;
            let skip = (position % SECTOR) as usize;
            let remaining = len - done;
            if skip == 0 && remaining >= SECTOR_SIZE {
                let whole = remaining / SECTOR_SIZE * SECTOR_SIZE;
                let out = &mut buf[done..done + whole];
                self.inner.read_exact_at(position, out)?;
                self.xts
                    .decrypt(out, SECTOR_SIZE, self.first_sector + sector);
                done += whole;
                continue;
            }
            let mut block = [0u8; SECTOR_SIZE];
            let stored = (size - sector * SECTOR).min(SECTOR) as usize;
            self.inner
                .read_exact_at(sector * SECTOR, &mut block[..stored])?;
            self.xts
                .decrypt(&mut block, SECTOR_SIZE, self.first_sector + sector);
            let take = (SECTOR_SIZE - skip).min(remaining);
            buf[done..done + take].copy_from_slice(&block[skip..skip + take]);
            done += take;
        }
        Ok(len)
    }
}

#[cfg(test)]
mod tests {
    use lichii_crypto::{Ticket, TitleKey};

    use super::*;
    use crate::bktr::open_patched_romfs;
    use crate::bktr::tests::{bucket_tree, compressed_entry, ctr_ex_entry, indirect_entry};
    use crate::romfs::open_romfs;
    use crate::testutil::*;
    use crate::vfs::{DirectoryExt, MemDir};

    fn romfs_tree() -> MemDir {
        MemDir::new()
            .with_bytes("a.bin", bytes(0x9000, 10))
            .with_bytes("b.bin", bytes(0x5123, 11))
            .with_dir("sub", MemDir::new().with_bytes("c.txt", b"hello".to_vec()))
    }

    fn logo() -> Vec<u8> {
        pfs(&[
            ("NintendoLogo.png", bytes(0x300, 4)),
            ("StartupMovie.gif", bytes(0x700, 5)),
        ])
    }

    fn program(magic: &[u8; 4]) -> (NcaSpec, Vec<u8>) {
        let image = romfs_image(&romfs_tree());
        let mut spec = NcaSpec::new(
            PROGRAM,
            APP,
            vec![
                pfs_section(&exefs()),
                ivfc_section(&image),
                pfs_section(&logo()).with_crypt(Crypt::Xts),
            ],
        );
        spec.magic = *magic;
        (spec, image)
    }

    fn open(data: Vec<u8>, keys: &KeySet) -> FsResult<Nca> {
        Nca::open(file(data), keys)
    }

    fn check_program(nca: &Nca, image: &[u8]) {
        let exefs = nca.exefs().unwrap().unwrap();
        assert_eq!(
            exefs.file("main").unwrap().read_all().unwrap(),
            bytes(0x2345, 1)
        );
        assert_eq!(
            exefs.file("main.npdm").unwrap().read_all().unwrap(),
            bytes(0x80, 2)
        );
        let romfs = nca.romfs().unwrap().unwrap();
        assert_eq!(romfs.read_all().unwrap(), image);
        let root = open_romfs(&romfs).unwrap();
        let text = root.open_file_path("sub/c.txt").unwrap().unwrap();
        assert_eq!(text.read_all().unwrap(), b"hello");
        let logo = nca.logo().unwrap().unwrap();
        assert_eq!(
            logo.file("StartupMovie.gif").unwrap().read_all().unwrap(),
            bytes(0x700, 5)
        );
        for section in nca.sections() {
            assert!(
                nca.verify_section(section.index).unwrap(),
                "{}",
                section.index
            );
        }
    }

    #[test]
    fn opens_nca3_nca2_and_decrypted_archives() {
        let keys = key_set();
        for (magic, kind) in [
            (b"NCA3", NcaMagic::Nca3),
            (b"NCA2", NcaMagic::Nca2),
            (b"DNCA", NcaMagic::Decrypted),
        ] {
            let (spec, image) = program(magic);
            let data = spec.build();
            assert_ne!(&data[0x200..0x204], b"NCA3");
            let nca = open(data.clone(), &keys).unwrap();
            let header = nca.header();
            assert_eq!(header.magic, kind);
            assert_eq!(header.header_encrypted, kind != NcaMagic::Decrypted);
            assert_eq!(header.content_type, NcaContentType::Program);
            assert_eq!(header.distribution_type, DistributionType::Download);
            assert_eq!(header.program_id, APP);
            assert_eq!(header.key_generation, KEY_GENERATION);
            assert_eq!(header.sdk_addon_version, 0x000c_1100);
            assert_eq!(header.content_size, data.len() as u64);
            assert!(header.rights_id.is_zero());
            let types: Vec<_> = nca.sections().map(|s| s.header.fs_type).collect();
            assert_eq!(
                types,
                [FsType::PartitionFs, FsType::RomFs, FsType::PartitionFs]
            );
            let encryption: Vec<_> = nca.sections().map(|s| s.header.encryption_type).collect();
            if kind == NcaMagic::Decrypted {
                assert_eq!(encryption, [EncryptionType::None; 3]);
            } else {
                assert_eq!(
                    encryption,
                    [
                        EncryptionType::AesCtr,
                        EncryptionType::AesCtr,
                        EncryptionType::AesXts
                    ]
                );
            }
            let romfs = nca.section(1).unwrap();
            assert_eq!(romfs.header.hash_type, HashType::HierarchicalIntegrity);
            assert_eq!(
                romfs.header.upper_iv,
                AesCtrUpperIv {
                    generation: GENERATION,
                    secure_value: SECURE_VALUE
                }
            );
            assert!(!nca.is_patch());
            check_program(&nca, &image);
        }

        let (spec, image) = program(b"DNCA");
        check_program(&open(spec.build(), &KeySet::new()).unwrap(), &image);

        let (spec, image) = program(b"NCA3");
        let plain_header = open(spec.build_plain(), &keys).unwrap();
        assert!(!plain_header.header().header_encrypted);
        check_program(&plain_header, &image);
    }

    #[test]
    fn ctr_and_xts_reads_at_any_offset_stop_at_the_section_end() {
        let keys = key_set();
        let (spec, _) = program(b"NCA3");
        let plain: Vec<Vec<u8>> = spec.sections.iter().map(|s| s.data.clone()).collect();
        let nca = open(spec.build(), &keys).unwrap();
        for (index, expected) in plain.iter().enumerate() {
            let raw = nca.raw_section_file(index).unwrap();
            assert_eq!(raw.size(), expected.len() as u64);
            assert_eq!(&raw.read_all().unwrap(), expected);
            let len = expected.len();
            for (start, size) in [
                (0, 1),
                (1, 15),
                (15, 2),
                (17, 0x1ff),
                (0x1ff, 2),
                (0x1f0, 0x220),
                (3, 0x600),
                (0x201, 0x3ff),
                (len - 0x11, 0x11),
                (len - 0x201, 0x201),
            ] {
                assert_eq!(
                    raw.read_vec(start as u64, size).unwrap(),
                    expected[start..start + size],
                    "section {index} at {start:#x}+{size:#x}"
                );
            }
            for start in (0..len).step_by(0x1f3) {
                let size = (len - start).min(0x35 + start % 0x400);
                assert_eq!(
                    raw.read_vec(start as u64, size).unwrap(),
                    expected[start..start + size]
                );
            }
            let mut buf = vec![0u8; 0x400];
            assert_eq!(raw.read_at(len as u64 - 0x10, &mut buf).unwrap(), 0x10);
            assert_eq!(&buf[..0x10], &expected[len - 0x10..]);
            assert_eq!(raw.read_at(len as u64, &mut buf).unwrap(), 0);
            assert!(raw.read_vec(len as u64 - 4, 8).is_err());
        }
    }

    #[test]
    fn title_keys_and_missing_keys_name_the_key() {
        let image = romfs_image(&romfs_tree());
        let rights = control_rights();
        let mut spec = NcaSpec::new(CONTROL, APP, vec![ivfc_section(&image)]);
        spec.rights_id = Some(rights);
        let data = spec.build();

        let error = open(data.clone(), &KeySet::new()).unwrap_err();
        assert!(matches!(&error, FsError::Key(KeyError::Missing(name)) if name == "header_key"));
        assert!(error.to_string().contains("header_key"));

        let mut keys = key_set();
        let locked = open(data.clone(), &keys).unwrap();
        assert_eq!(locked.header().rights_id, rights);
        let error = locked.romfs().err().unwrap();
        assert!(matches!(error, FsError::Key(KeyError::MissingTitleKey(id)) if id == rights));
        assert!(
            locked
                .romfs()
                .err()
                .unwrap()
                .to_string()
                .contains(&rights.to_string())
        );

        let ticket = Ticket::parse(&ticket(rights)).unwrap();
        let TitleKey::Common(key) = ticket.title_key else {
            panic!("common ticket expected");
        };
        keys.add_title_key(ticket.rights_id, key);
        let unlocked = open(data.clone(), &keys).unwrap();
        assert_eq!(
            unlocked.romfs().unwrap().unwrap().read_all().unwrap(),
            image
        );

        let header_only = format!("header_key = {}\n", hex(&header_key()));
        let mut no_titlekek = KeySet::new();
        no_titlekek.load_prod_keys(&header_only);
        no_titlekek.add_title_key(rights, encrypted_title_key());
        let error = open(data, &no_titlekek).unwrap().romfs().err().unwrap();
        assert!(error.to_string().contains("titlekek_02"), "{error}");

        let mut no_key_area_key = KeySet::new();
        no_key_area_key.load_prod_keys(&header_only);
        let nca = open(program(b"NCA3").0.build(), &no_key_area_key).unwrap();
        let error = nca.exefs().err().unwrap();
        assert!(
            error.to_string().contains("key_area_key_application_02"),
            "{error}"
        );

        let mut wrong = KeySet::new();
        wrong.load_prod_keys(&format!("header_key = {}\n", hex(&pattern::<32>(0x99))));
        assert!(matches!(
            open(program(b"NCA3").0.build(), &wrong),
            Err(FsError::BadMagic { .. })
        ));
    }

    fn bktr_pair() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let image_a = romfs_image(&romfs_tree());
        let mut modded = bytes(0x9000, 10);
        for byte in &mut modded[0x3000..0x3400] {
            *byte ^= 0x5a;
        }
        let tree_b = MemDir::new()
            .with_bytes("a.bin", modded)
            .with_bytes("b.bin", bytes(0x5123, 11))
            .with_dir(
                "sub",
                MemDir::new()
                    .with_bytes("c.txt", b"hello".to_vec())
                    .with_bytes("new.bin", bytes(0x2100, 12)),
            );
        let image_b = romfs_image(&tree_b);

        let base_section = ivfc_section(&image_a);
        let base_raw = base_section.data.clone();
        let base = NcaSpec::new(PROGRAM, APP, vec![pfs_section(&exefs()), base_section]).build();

        let target = ivfc_section(&image_b);
        let mut runs: Vec<(u64, u64, u32, u64)> = Vec::new();
        let mut patch = Vec::new();
        for (block_index, block) in target.data.chunks(0x200).enumerate() {
            let at = block_index * 0x200;
            let original = base_raw.get(at..at + block.len()) == Some(block);
            let (storage, physical) = if original {
                (0, at as u64)
            } else {
                patch.extend(block);
                (1, (patch.len() - block.len()) as u64)
            };
            match runs.last_mut() {
                Some(run) if run.2 == storage && run.1 + run.3 == physical => {
                    run.3 += block.len() as u64
                }
                _ => runs.push((at as u64, physical, storage, block.len() as u64)),
            }
        }
        assert!(runs.iter().any(|r| r.2 == 0) && runs.iter().any(|r| r.2 == 1));
        patch.resize(patch.len().next_multiple_of(0x200), 0);
        let indirect: Vec<_> = runs
            .iter()
            .map(|run| indirect_entry(run.0, run.1, run.2))
            .collect();
        let (indirect_header, indirect_table) = bucket_tree(&indirect, target.data.len() as u64);
        let indirect_offset = patch.len();
        let ctr_ex_offset = indirect_offset + indirect_table.len();
        let split = (patch.len() / 2) & !0xf;
        let regions = vec![
            (0..split, Some(7u32)),
            (split..patch.len(), None),
            (patch.len()..ctr_ex_offset, Some(GENERATION)),
        ];
        let ctr_ex: Vec<_> = regions
            .iter()
            .map(|(range, generation)| {
                ctr_ex_entry(
                    range.start as u64,
                    generation.is_some(),
                    generation.unwrap_or(0),
                )
            })
            .collect();
        let (ctr_ex_header, ctr_ex_table) = bucket_tree(&ctr_ex, ctr_ex_offset as u64);

        let mut data = patch;
        data.extend(&indirect_table);
        data.extend(&ctr_ex_table);
        let total = data.len();
        let mut header = target.fs_header;
        header[0x100..0x108].copy_from_slice(&(indirect_offset as u64).to_le_bytes());
        header[0x108..0x110].copy_from_slice(&(indirect_table.len() as u64).to_le_bytes());
        header[0x110..0x120].copy_from_slice(&indirect_header);
        header[0x120..0x128].copy_from_slice(&(ctr_ex_offset as u64).to_le_bytes());
        header[0x128..0x130].copy_from_slice(&(ctr_ex_table.len() as u64).to_le_bytes());
        header[0x130..0x140].copy_from_slice(&ctr_ex_header);
        let encrypt: Encrypt = Box::new(move |data, offset, cipher| {
            for (range, generation) in &regions {
                if let Some(generation) = generation {
                    ctr_range(cipher, data, offset, range.clone(), *generation);
                }
            }
            ctr_range(cipher, data, offset, ctr_ex_offset..total, GENERATION);
        });
        let section = Section {
            fs_header: header,
            data,
            crypt: Crypt::Custom(4, encrypt),
            size: None,
        };
        let update = NcaSpec::new(PROGRAM, APP, vec![pfs_section(&exefs()), section]).build();
        (base, update, image_b)
    }

    #[test]
    fn bktr_update_patches_the_base_romfs() {
        let keys = key_set();
        let (base, update, expected) = bktr_pair();
        let base = open(base, &keys).unwrap();
        let update = open(update, &keys).unwrap();
        assert!(update.is_patch() && !base.is_patch());
        assert_eq!(
            update.section(1).unwrap().header.encryption_type,
            EncryptionType::AesCtrEx
        );
        assert!(matches!(update.romfs(), Err(FsError::Unsupported(_))));
        assert!(update.exefs().unwrap().is_some());

        let patched = open_patched_romfs(&update, &base).unwrap();
        assert_eq!(patched.size(), expected.len() as u64);
        assert_eq!(patched.read_all().unwrap(), expected);
        for start in (0..expected.len()).step_by(0x3e1) {
            let size = (expected.len() - start).min(0x1234);
            assert_eq!(
                patched.read_vec(start as u64, size).unwrap(),
                expected[start..start + size]
            );
        }
        let root = open_romfs(&patched).unwrap();
        let added = root.open_file_path("sub/new.bin").unwrap().unwrap();
        assert_eq!(added.read_all().unwrap(), bytes(0x2100, 12));
        let modded = root
            .open_file_path("a.bin")
            .unwrap()
            .unwrap()
            .read_all()
            .unwrap();
        assert_eq!(modded[0x3000], bytes(0x9000, 10)[0x3000] ^ 0x5a);

        let unpatched = update.patched_section_file(0, &base).unwrap();
        assert_eq!(unpatched.read_all().unwrap(), exefs());
    }

    #[test]
    fn corrupt_bktr_tables_fail_cleanly() {
        let keys = key_set();
        let (base_data, update, _) = bktr_pair();
        let base = open(base_data.clone(), &keys).unwrap();
        for position in (0xc00..update.len()).step_by(0xfd) {
            let mut bad = update.clone();
            bad[position] ^= 0xa5;
            if let Ok(nca) = open(bad, &keys)
                && let Ok(patched) = open_patched_romfs(&nca, &base)
            {
                let _ = patched.read_all();
            }
        }
        let update = open(update, &keys).unwrap();
        let truncated = open(base_data[..0x4000].to_vec(), &keys).unwrap();
        assert!(open_patched_romfs(&update, &truncated).is_err());
        assert!(open_patched_romfs(&update, &update).is_err());
    }

    #[test]
    fn compressed_romfs_section() {
        let keys = key_set();
        let image = romfs_image(&romfs_tree());
        let mut data = Vec::new();
        let mut entries = Vec::new();
        for (index, block) in image.chunks(0x4000).enumerate() {
            data.resize(data.len().next_multiple_of(0x10), 0);
            let at = data.len() as u64;
            let virtual_offset = (index * 0x4000) as u64;
            if index == 1 {
                entries.push(compressed_entry(virtual_offset, at, 0, 0));
                data.extend(block);
            } else {
                let packed = lz4_flex::block::compress(block);
                entries.push(compressed_entry(virtual_offset, at, 3, packed.len() as u32));
                data.extend(packed);
            }
        }
        data.resize(data.len().next_multiple_of(0x10), 0);
        let table_offset = data.len() as u64;
        let (header, table) = bucket_tree(&entries, image.len() as u64);
        data.extend(&table);
        let mut section = ivfc_section(&data);
        section.fs_header[0x178..0x180].copy_from_slice(&table_offset.to_le_bytes());
        section.fs_header[0x180..0x188].copy_from_slice(&(table.len() as u64).to_le_bytes());
        section.fs_header[0x188..0x198].copy_from_slice(&header);
        let nca = open(NcaSpec::new(DATA, APP, vec![section]).build(), &keys).unwrap();
        let romfs = nca.romfs().unwrap().unwrap();
        assert_eq!(romfs.read_all().unwrap(), image);
        assert_eq!(romfs.read_vec(0x3ff8, 0x10).unwrap(), image[0x3ff8..0x4008]);
        let root = open_romfs(&romfs).unwrap();
        let text = root.open_file_path("sub/c.txt").unwrap().unwrap();
        assert_eq!(text.read_all().unwrap(), b"hello");
        assert!(nca.verify_section(0).unwrap());
    }

    #[test]
    fn sparse_section_reads_stored_ranges() {
        let keys = key_set();
        let virtual_data = bytes(0x6000, 77);
        let sparse_generation = 3u16;
        let entries = vec![
            indirect_entry(0, 0, 0),
            indirect_entry(0x2000, 0, 1),
            indirect_entry(0x4000, 0x2000, 0),
        ];
        let (header, table) = bucket_tree(&entries, 0x6000);
        let mut physical = virtual_data[..0x2000].to_vec();
        physical.extend(&virtual_data[0x4000..]);
        physical.extend(&table);
        let table_len = physical.len();
        let mut fs = fs_header(0, 1);
        fs[0x148..0x150].copy_from_slice(&0x4000u64.to_le_bytes());
        fs[0x150..0x158].copy_from_slice(&(table.len() as u64).to_le_bytes());
        fs[0x158..0x168].copy_from_slice(&header);
        fs[0x168..0x170].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
        fs[0x170..0x172].copy_from_slice(&sparse_generation.to_le_bytes());
        let encrypt: Encrypt = Box::new(move |data, offset, cipher| {
            ctr_range(cipher, data, offset, 0..0x2000, GENERATION);
            let mut tail = data[0x2000..0x4000].to_vec();
            cipher.apply(&counter_iv(GENERATION), offset + 0x4000, &mut tail);
            data[0x2000..0x4000].copy_from_slice(&tail);
            let meta_generation = u32::from(sparse_generation) << 16;
            ctr_range(cipher, data, offset, 0x4000..table_len, meta_generation);
        });
        let section = Section {
            fs_header: fs,
            data: physical,
            crypt: Crypt::Custom(3, encrypt),
            size: Some(0x6000),
        };
        let nca = open(NcaSpec::new(DATA, APP, vec![section]).build(), &keys).unwrap();
        let data = nca.section_file(0).unwrap();
        assert_eq!(data.size(), 0x6000);
        assert_eq!(data.read_vec(0, 0x2000).unwrap(), virtual_data[..0x2000]);
        assert_eq!(
            data.read_vec(0x4000, 0x2000).unwrap(),
            virtual_data[0x4000..]
        );
        assert_eq!(data.read_vec(0x1ff3, 0x2020).unwrap().len(), 0x2020);
    }

    fn exercise(nca: &Nca) {
        for index in 0..MAX_SECTIONS {
            if let Ok(raw) = nca.raw_section_file(index) {
                let _ = raw.read_vec(0, raw.size().min(0x4_0000) as usize);
            }
            if let Ok(data) = nca.section_file(index) {
                let _ = data.read_vec(0, data.size().min(0x4_0000) as usize);
            }
            let _ = nca.verify_section(index);
        }
        let _ = nca.exefs();
        let _ = nca.logo();
        if let Ok(Some(romfs)) = nca.romfs()
            && let Ok(root) = open_romfs(&romfs)
        {
            let _ = root.entries();
        }
    }

    #[test]
    fn corrupt_and_truncated_archives_fail_cleanly() {
        let keys = key_set();
        for magic in [b"NCA3", b"DNCA"] {
            let data = program(magic).0.build();
            for cut in [0, 0x1ff, 0x400, 0xbff, 0xc00, 0xc01, 0x1000, data.len() / 2] {
                if let Ok(nca) = open(data[..cut].to_vec(), &keys) {
                    exercise(&nca);
                }
            }
            for position in (0..HEADER_SIZE).step_by(7) {
                for value in [0x00, 0xff, 0x81] {
                    let mut bad = data.clone();
                    bad[position] = value;
                    if let Ok(nca) = open(bad, &keys) {
                        exercise(&nca);
                    }
                }
            }
        }
    }
}
