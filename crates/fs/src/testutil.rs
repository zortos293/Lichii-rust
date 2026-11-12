//! Synthetic NCAs, tickets and packages built from made-up keys for the container tests.

use std::ops::Range;

use lichii_crypto::{AesCtr, AesEcb, AesXts, KeySet, RightsId, XtsTweak, sha256};

use crate::cnmt::{Cnmt, ContentId, ContentMetaType, ContentRecord, ContentType, ExtendedHeader};
use crate::pfs::{PartitionKind, build_partition};
use crate::romfs::build_romfs;
use crate::vfs::{FileExt, FileRef, MemDir, MemFile};

/// Key generation of every synthetic NCA: master key revision 2.
pub const KEY_GENERATION: u8 = 3;
pub const GENERATION: u32 = 0x0000_0002;
pub const SECURE_VALUE: u32 = 0x5eed_0001;
pub const PROGRAM: u8 = 0;
pub const META: u8 = 1;
pub const CONTROL: u8 = 2;
pub const DATA: u8 = 4;

pub fn pattern<const N: usize>(seed: u8) -> [u8; N] {
    std::array::from_fn(|i| seed.wrapping_add((i as u8).wrapping_mul(29)) ^ (i as u8 >> 3))
}

pub fn bytes(len: usize, seed: u32) -> Vec<u8> {
    (0..len as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 13) as u8)
        .collect()
}

pub fn header_key() -> [u8; 32] {
    pattern(0x31)
}

pub fn key_area_key() -> [u8; 16] {
    pattern(0x52)
}

pub fn titlekek() -> [u8; 16] {
    pattern(0x73)
}

pub fn title_key() -> [u8; 16] {
    pattern(0x94)
}

pub fn key_area() -> [u8; 0x40] {
    pattern(0xb5)
}

pub fn hex(data: &[u8]) -> String {
    data.iter().map(|b| format!("{b:02x}")).collect()
}

/// A key set holding the header key, key area key and titlekek the builders use.
pub fn key_set() -> KeySet {
    let mut keys = KeySet::new();
    let report = keys.load_prod_keys(&format!(
        "header_key = {}\nkey_area_key_application_02 = {}\ntitlekek_02 = {}\n",
        hex(&header_key()),
        hex(&key_area_key()),
        hex(&titlekek())
    ));
    assert_eq!(report.loaded, 3);
    keys
}

pub fn encrypted_title_key() -> [u8; 16] {
    let mut key = title_key();
    AesEcb::new(&titlekek()).encrypt(&mut key);
    key
}

/// A common ticket for `rights_id` carrying [`title_key`].
pub fn ticket(rights_id: RightsId) -> Vec<u8> {
    let mut raw = vec![0u8; 0x2c0];
    raw[..4].copy_from_slice(&0x10004u32.to_le_bytes());
    let body = &mut raw[0x140..];
    body[0x40..0x50].copy_from_slice(&encrypted_title_key());
    body[0x145] = KEY_GENERATION;
    body[0x160..0x170].copy_from_slice(&rights_id.0);
    raw
}

pub type Encrypt = Box<dyn Fn(&mut [u8], u64, &AesCtr)>;

/// How the builder encrypts a section.
pub enum Crypt {
    Ctr,
    Xts,
    /// Caller-provided encryption, given the data, the section's offset in the NCA and the
    /// content key's cipher; the FS header's encryption type is set to the given value.
    Custom(u8, Encrypt),
}

pub struct Section {
    pub fs_header: Vec<u8>,
    pub data: Vec<u8>,
    pub crypt: Crypt,
    /// Overrides the section size recorded in the header (sparse sections).
    pub size: Option<u64>,
}

impl Section {
    pub fn with_crypt(mut self, crypt: Crypt) -> Self {
        self.crypt = crypt;
        self
    }
}

pub fn fs_header(fs_type: u8, hash_type: u8) -> Vec<u8> {
    let mut header = vec![0u8; 0x200];
    header[0..2].copy_from_slice(&2u16.to_le_bytes());
    header[2] = fs_type;
    header[3] = hash_type;
    header[0x140..0x144].copy_from_slice(&GENERATION.to_le_bytes());
    header[0x144..0x148].copy_from_slice(&SECURE_VALUE.to_le_bytes());
    header
}

fn pad(data: &mut Vec<u8>) {
    data.resize(data.len().next_multiple_of(0x200), 0);
}

/// A PFS0 section with a two-layer SHA-256 tree over 0x1000-byte blocks.
pub fn pfs_section(pfs: &[u8]) -> Section {
    let table: Vec<u8> = pfs.chunks(0x1000).flat_map(sha256).collect();
    let data_offset = table.len().next_multiple_of(0x200);
    let mut data = table.clone();
    data.resize(data_offset, 0);
    data.extend(pfs);
    pad(&mut data);
    let mut header = fs_header(1, 2);
    header[0x08..0x28].copy_from_slice(&sha256(&table));
    header[0x28..0x2c].copy_from_slice(&0x1000u32.to_le_bytes());
    header[0x2c..0x30].copy_from_slice(&2u32.to_le_bytes());
    header[0x38..0x40].copy_from_slice(&(table.len() as u64).to_le_bytes());
    header[0x40..0x48].copy_from_slice(&(data_offset as u64).to_le_bytes());
    header[0x48..0x50].copy_from_slice(&(pfs.len() as u64).to_le_bytes());
    Section {
        fs_header: header,
        data,
        crypt: Crypt::Ctr,
        size: None,
    }
}

/// A RomFS section: six IVFC levels over 0x4000-byte blocks, the last one `image`.
pub fn ivfc_section(image: &[u8]) -> Section {
    const BLOCK: usize = 0x4000;
    let hash_level = |below: &[u8]| -> Vec<u8> {
        below
            .chunks(BLOCK)
            .flat_map(|chunk| {
                let mut block = chunk.to_vec();
                block.resize(BLOCK, 0);
                sha256(&block)
            })
            .collect()
    };
    let mut levels = vec![image.to_vec()];
    for _ in 0..5 {
        let next = hash_level(levels.last().unwrap());
        levels.push(next);
    }
    levels.reverse();
    let master = hash_level(&levels[0]);
    let mut header = fs_header(0, 3);
    header[0x08..0x0c].copy_from_slice(b"IVFC");
    header[0x0c..0x10].copy_from_slice(&0x20000u32.to_le_bytes());
    header[0x10..0x14].copy_from_slice(&0x20u32.to_le_bytes());
    header[0x14..0x18].copy_from_slice(&7u32.to_le_bytes());
    header[0xc8..0xe8].copy_from_slice(&master);
    let mut data = Vec::new();
    for (index, level) in levels.iter().enumerate() {
        let at = 0x18 + index * 0x18;
        header[at..at + 8].copy_from_slice(&(data.len() as u64).to_le_bytes());
        header[at + 8..at + 16].copy_from_slice(&(level.len() as u64).to_le_bytes());
        header[at + 16..at + 20].copy_from_slice(&14u32.to_le_bytes());
        data.extend(level);
        pad(&mut data);
    }
    Section {
        fs_header: header,
        data,
        crypt: Crypt::Ctr,
        size: None,
    }
}

/// The counter block the FS header's stored upper IV (generation, then secure value, both
/// little-endian) gives, derived the way hactool does: the eight stored bytes reversed.
pub fn counter_iv(generation: u32) -> [u8; 16] {
    let mut stored = [0u8; 8];
    stored[..4].copy_from_slice(&generation.to_le_bytes());
    stored[4..].copy_from_slice(&SECURE_VALUE.to_le_bytes());
    let mut iv = [0u8; 16];
    for (slot, byte) in iv.iter_mut().zip(stored.iter().rev()) {
        *slot = *byte;
    }
    iv
}

/// Encrypts `range` of a section at `section_offset` with the counter generation `generation`.
pub fn ctr_range(
    cipher: &AesCtr,
    data: &mut [u8],
    section_offset: u64,
    range: Range<usize>,
    generation: u32,
) {
    let iv = counter_iv(generation);
    let start = range.start as u64;
    cipher.apply(&iv, section_offset + start, &mut data[range]);
}

pub struct NcaSpec {
    pub magic: [u8; 4],
    pub content_type: u8,
    pub program_id: u64,
    pub rights_id: Option<RightsId>,
    pub sections: Vec<Section>,
}

impl NcaSpec {
    pub fn new(content_type: u8, program_id: u64, sections: Vec<Section>) -> Self {
        Self {
            magic: *b"NCA3",
            content_type,
            program_id,
            rights_id: None,
            sections,
        }
    }

    /// The NCA with its header still in plaintext.
    pub fn build_plain(&self) -> Vec<u8> {
        let plaintext = &self.magic == b"DNCA";
        let mut out = vec![0u8; 0xc00];
        let (ctr_key, xts_key) = match self.rights_id {
            Some(_) => (title_key(), [0u8; 32]),
            None => {
                let area = key_area();
                (
                    area[0x20..0x30].try_into().unwrap(),
                    area[..0x20].try_into().unwrap(),
                )
            }
        };
        let cipher = AesCtr::new(&ctr_key);
        let mut headers = vec![0u8; 0x800];
        for (index, section) in self.sections.iter().enumerate() {
            let offset = out.len() as u64;
            let mut fs = section.fs_header.clone();
            let mut data = section.data.clone();
            fs[4] = if plaintext {
                1
            } else {
                match &section.crypt {
                    Crypt::Xts => 2,
                    Crypt::Ctr => 3,
                    Crypt::Custom(kind, _) => *kind,
                }
            };
            if !plaintext {
                let iv = counter_iv(GENERATION);
                match &section.crypt {
                    Crypt::Ctr => cipher.apply(&iv, offset, &mut data),
                    Crypt::Xts => AesXts::new(&xts_key, XtsTweak::Nintendo).encrypt(
                        &mut data,
                        0x200,
                        offset / 0x200,
                    ),
                    Crypt::Custom(_, encrypt) => encrypt(&mut data, offset, &cipher),
                }
            }
            let start = (offset / 0x200) as u32;
            let sectors = section.size.unwrap_or(data.len() as u64).div_ceil(0x200) as u32;
            let info = 0x240 + index * 0x10;
            out[info..info + 4].copy_from_slice(&start.to_le_bytes());
            out[info + 4..info + 8].copy_from_slice(&(start + sectors).to_le_bytes());
            out[info + 8..info + 12].copy_from_slice(&1u32.to_le_bytes());
            let hash = 0x280 + index * 0x20;
            out[hash..hash + 0x20].copy_from_slice(&sha256(&fs));
            headers[index * 0x200..][..0x200].copy_from_slice(&fs);
            out.extend(data);
        }
        out[0x400..0xc00].copy_from_slice(&headers);
        out[0x200..0x204].copy_from_slice(&self.magic);
        out[0x205] = self.content_type;
        out[0x206] = 2;
        out[0x220] = KEY_GENERATION;
        let size = out.len() as u64;
        out[0x208..0x210].copy_from_slice(&size.to_le_bytes());
        out[0x210..0x218].copy_from_slice(&self.program_id.to_le_bytes());
        out[0x21c..0x220].copy_from_slice(&0x000c_1100u32.to_le_bytes());
        match self.rights_id {
            Some(rights_id) => out[0x230..0x240].copy_from_slice(&rights_id.0),
            None => {
                let mut area = key_area();
                AesEcb::new(&key_area_key()).encrypt(&mut area);
                out[0x300..0x340].copy_from_slice(&area);
            }
        }
        out
    }

    pub fn build(&self) -> Vec<u8> {
        let mut out = self.build_plain();
        seal(&mut out);
        out
    }
}

/// Encrypts a plaintext header built by [`NcaSpec::build_plain`] according to its magic.
pub fn seal(nca: &mut [u8]) {
    let xts = AesXts::new(&header_key(), XtsTweak::Nintendo);
    match &nca[0x200..0x204] {
        b"NCA3" => xts.encrypt(&mut nca[..0xc00], 0x200, 0),
        b"NCA2" => {
            xts.encrypt(&mut nca[..0x400], 0x200, 0);
            for fs in nca[0x400..0xc00].as_chunks_mut::<0x200>().0 {
                xts.encrypt(fs, 0x200, 0);
            }
        }
        _ => {}
    }
}

pub fn file(data: Vec<u8>) -> FileRef {
    MemFile::from_bytes(data)
}

pub fn pfs(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
    let files: Vec<_> = files
        .iter()
        .map(|(name, data)| ((*name).to_owned(), file(data.clone())))
        .collect();
    build_partition(PartitionKind::Pfs0, &files)
        .unwrap()
        .read_all()
        .unwrap()
}

pub fn exefs() -> Vec<u8> {
    pfs(&[
        ("main", bytes(0x2345, 1)),
        ("main.npdm", bytes(0x80, 2)),
        ("rtld", bytes(0x1000, 3)),
    ])
}

pub const APP: u64 = 0x0100_0000_0001_0000;

pub fn romfs_image(tree: &MemDir) -> Vec<u8> {
    build_romfs(tree).unwrap().read_all().unwrap()
}

pub fn control_rights() -> RightsId {
    let mut id = [0u8; 16];
    id[..8].copy_from_slice(&APP.to_be_bytes());
    id[15] = KEY_GENERATION;
    RightsId(id)
}

pub fn record(content_type: ContentType, id: u8, data: &[u8]) -> ContentRecord {
    ContentRecord {
        hash: sha256(data),
        content_id: ContentId([id; 16]),
        size: data.len() as u64,
        attributes: 0,
        content_type,
        id_offset: 0,
    }
}

pub fn cnmt(
    title_id: u64,
    version: u32,
    extended: ExtendedHeader,
    contents: Vec<ContentRecord>,
) -> Cnmt {
    let meta_type = match extended {
        ExtendedHeader::Application { .. } => ContentMetaType::Application,
        ExtendedHeader::Patch { .. } => ContentMetaType::Patch,
        ExtendedHeader::AddOnContent { .. } => ContentMetaType::AddOnContent,
        ExtendedHeader::Other(_) => ContentMetaType::SystemData,
    };
    Cnmt {
        title_id,
        version,
        meta_type,
        attributes: 0,
        required_download_system_version: 0,
        extended,
        contents,
        meta: Vec::new(),
    }
}

/// A Meta NCA whose PFS0 holds `cnmt`, and its file name for content id `id`.
pub fn meta_nca(cnmt: &Cnmt, id: u8) -> (String, FileRef) {
    let name = format!("{:?}_{:016x}.cnmt", cnmt.meta_type, cnmt.title_id);
    let image = pfs(&[(&name, crate::cnmt::tests::encode(cnmt))]);
    let nca = NcaSpec::new(META, cnmt.title_id, vec![pfs_section(&image)]).build();
    (format!("{}.cnmt.nca", ContentId([id; 16])), file(nca))
}

pub fn content_file(id: u8, data: Vec<u8>) -> (String, FileRef) {
    (ContentId([id; 16]).file_name(), file(data))
}

/// An application package: program NCA (ExeFS + RomFS), control NCA behind a title key, the
/// Meta NCA, the control NCA's ticket and a certificate. The CNMT also lists a delta fragment
/// and a manual that are not in the package.
pub fn application_package(with_ticket: bool) -> Vec<(String, FileRef)> {
    let data = MemDir::new().with_bytes("data.bin", bytes(0x1800, 20));
    let program = NcaSpec::new(
        PROGRAM,
        APP,
        vec![pfs_section(&exefs()), ivfc_section(&romfs_image(&data))],
    )
    .build();
    let control_tree = MemDir::new()
        .with_bytes(
            "control.nacp",
            crate::nacp::tests::synthetic_nacp("Lichii Test", "Lichii", "1.0.0"),
        )
        .with_bytes("icon_AmericanEnglish.dat", bytes(0x400, 21));
    let mut control = NcaSpec::new(
        CONTROL,
        APP,
        vec![ivfc_section(&romfs_image(&control_tree))],
    );
    control.rights_id = Some(control_rights());
    let control = control.build();
    let meta = cnmt(
        APP,
        0,
        ExtendedHeader::Application {
            patch_id: APP | 0x800,
            required_system_version: 0,
        },
        vec![
            record(ContentType::Program, 0xa1, &program),
            record(ContentType::Control, 0xa2, &control),
            record(ContentType::DeltaFragment, 0xa9, &[]),
            record(ContentType::HtmlDocument, 0xa3, &[]),
        ],
    );
    let rights = control_rights();
    let mut files = vec![
        content_file(0xa1, program),
        content_file(0xa2, control),
        meta_nca(&meta, 0xa0),
        (format!("{rights}.cert"), file(bytes(0x700, 22))),
    ];
    if with_ticket {
        files.push((format!("{rights}.tik"), file(ticket(rights))));
    }
    files
}

/// An update package: a patch for [`APP`] and one of its add-ons.
pub fn update_package() -> Vec<(String, FileRef)> {
    let program = NcaSpec::new(PROGRAM, APP, vec![pfs_section(&exefs())]).build();
    let dlc_tree = MemDir::new().with_bytes("dlc.bin", bytes(0x900, 23));
    let dlc = NcaSpec::new(
        DATA,
        APP + 0x1001,
        vec![ivfc_section(&romfs_image(&dlc_tree))],
    )
    .build();
    let patch = cnmt(
        APP | 0x800,
        0x10000,
        ExtendedHeader::Patch {
            application_id: APP,
            required_system_version: 0,
        },
        vec![record(ContentType::Program, 0xb1, &program)],
    );
    let add_on = cnmt(
        APP + 0x1001,
        0,
        ExtendedHeader::AddOnContent {
            application_id: APP,
            required_application_version: 0,
        },
        vec![record(ContentType::Data, 0xc1, &dlc)],
    );
    vec![
        content_file(0xb1, program),
        content_file(0xc1, dlc),
        meta_nca(&patch, 0xb0),
        meta_nca(&add_on, 0xc0),
    ]
}
