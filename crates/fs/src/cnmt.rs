//! CNMT: packaged content meta, the list of NCAs that make up a title, update or add-on.

use std::fmt;

use crate::bytes::{array_at, u16_at, u32_at, u64_at};
use crate::error::{FsResult, corrupt};

const FORMAT: &str = "CNMT";
const HEADER_SIZE: usize = 0x20;
const CONTENT_RECORD_SIZE: usize = 0x38;
const META_RECORD_SIZE: usize = 0x10;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContentMetaType {
    SystemProgram,
    SystemData,
    SystemUpdate,
    BootImagePackage,
    BootImagePackageSafe,
    Application,
    Patch,
    AddOnContent,
    Delta,
    DataPatch,
    Unknown(u8),
}

impl ContentMetaType {
    pub fn from_raw(value: u8) -> Self {
        match value {
            0x01 => Self::SystemProgram,
            0x02 => Self::SystemData,
            0x03 => Self::SystemUpdate,
            0x04 => Self::BootImagePackage,
            0x05 => Self::BootImagePackageSafe,
            0x80 => Self::Application,
            0x81 => Self::Patch,
            0x82 => Self::AddOnContent,
            0x83 => Self::Delta,
            0x84 => Self::DataPatch,
            other => Self::Unknown(other),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContentType {
    Meta,
    Program,
    Data,
    Control,
    HtmlDocument,
    LegalInformation,
    DeltaFragment,
    Unknown(u8),
}

impl ContentType {
    pub fn from_raw(value: u8) -> Self {
        match value {
            0 => Self::Meta,
            1 => Self::Program,
            2 => Self::Data,
            3 => Self::Control,
            4 => Self::HtmlDocument,
            5 => Self::LegalInformation,
            6 => Self::DeltaFragment,
            other => Self::Unknown(other),
        }
    }
}

/// An NCA's content id: the first half of its SHA-256 and its file name in hex.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ContentId(pub [u8; 16]);

impl ContentId {
    /// The name the NCA is stored under: lowercase hex plus `.nca`.
    pub fn file_name(&self) -> String {
        format!("{self}.nca")
    }
}

impl fmt::Display for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl fmt::Debug for ContentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ContentId({self})")
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentRecord {
    pub hash: [u8; 32],
    pub content_id: ContentId,
    pub size: u64,
    pub attributes: u8,
    pub content_type: ContentType,
    pub id_offset: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContentMetaRecord {
    pub title_id: u64,
    pub version: u32,
    pub meta_type: ContentMetaType,
    pub attributes: u8,
}

/// The per-type extended header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExtendedHeader {
    Application {
        patch_id: u64,
        required_system_version: u32,
    },
    Patch {
        application_id: u64,
        required_system_version: u32,
    },
    AddOnContent {
        application_id: u64,
        required_application_version: u32,
    },
    Other(Vec<u8>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cnmt {
    pub title_id: u64,
    pub version: u32,
    pub meta_type: ContentMetaType,
    pub attributes: u8,
    pub required_download_system_version: u32,
    pub extended: ExtendedHeader,
    pub contents: Vec<ContentRecord>,
    pub meta: Vec<ContentMetaRecord>,
}

impl Cnmt {
    pub fn parse(data: &[u8]) -> FsResult<Self> {
        if data.len() < HEADER_SIZE {
            return Err(corrupt(FORMAT, "shorter than its header"));
        }
        let meta_type = ContentMetaType::from_raw(data[0x0c]);
        let extended_size = usize::from(u16_at(data, 0x0e));
        let content_count = usize::from(u16_at(data, 0x10));
        let meta_count = usize::from(u16_at(data, 0x12));
        let contents_start = HEADER_SIZE + extended_size;
        let meta_start = contents_start + content_count * CONTENT_RECORD_SIZE;
        if meta_start + meta_count * META_RECORD_SIZE > data.len() {
            return Err(corrupt(FORMAT, "records past the end of the data"));
        }
        let ext = &data[HEADER_SIZE..contents_start];
        let id_and_version = || (ext.len() >= 12).then(|| (u64_at(ext, 0), u32_at(ext, 8)));
        let extended = match (meta_type, id_and_version()) {
            (ContentMetaType::Application, Some((patch_id, required_system_version))) => {
                ExtendedHeader::Application {
                    patch_id,
                    required_system_version,
                }
            }
            (ContentMetaType::Patch, Some((application_id, required_system_version))) => {
                ExtendedHeader::Patch {
                    application_id,
                    required_system_version,
                }
            }
            (
                ContentMetaType::AddOnContent,
                Some((application_id, required_application_version)),
            ) => ExtendedHeader::AddOnContent {
                application_id,
                required_application_version,
            },
            _ => ExtendedHeader::Other(ext.to_vec()),
        };
        let contents = data[contents_start..meta_start]
            .as_chunks::<CONTENT_RECORD_SIZE>()
            .0
            .iter()
            .map(|r| ContentRecord {
                hash: array_at(r, 0),
                content_id: ContentId(array_at(r, 0x20)),
                size: u64::from(u32_at(r, 0x30)) | (u64::from(r[0x34]) << 32),
                attributes: r[0x35],
                content_type: ContentType::from_raw(r[0x36]),
                id_offset: r[0x37],
            })
            .collect();
        let meta = data[meta_start..meta_start + meta_count * META_RECORD_SIZE]
            .as_chunks::<META_RECORD_SIZE>()
            .0
            .iter()
            .map(|r| ContentMetaRecord {
                title_id: u64_at(r, 0),
                version: u32_at(r, 8),
                meta_type: ContentMetaType::from_raw(r[12]),
                attributes: r[13],
            })
            .collect();
        Ok(Self {
            title_id: u64_at(data, 0),
            version: u32_at(data, 8),
            meta_type,
            attributes: data[0x14],
            required_download_system_version: u32_at(data, 0x18),
            extended,
            contents,
            meta,
        })
    }

    /// The first content of `content_type`, preferring id offset 0 (the main program of a
    /// multi-program title).
    pub fn content(&self, content_type: ContentType) -> Option<&ContentRecord> {
        let mut matching = self
            .contents
            .iter()
            .filter(|c| c.content_type == content_type);
        let first = matching.clone().next();
        matching.find(|c| c.id_offset == 0).or(first)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn encode(cnmt: &Cnmt) -> Vec<u8> {
        let meta_type = match cnmt.meta_type {
            ContentMetaType::Application => 0x80,
            ContentMetaType::Patch => 0x81,
            ContentMetaType::AddOnContent => 0x82,
            _ => 0,
        };
        let extended = match &cnmt.extended {
            ExtendedHeader::Application {
                patch_id: id,
                required_system_version: v,
            }
            | ExtendedHeader::Patch {
                application_id: id,
                required_system_version: v,
            }
            | ExtendedHeader::AddOnContent {
                application_id: id,
                required_application_version: v,
            } => {
                let mut out = id.to_le_bytes().to_vec();
                out.extend(v.to_le_bytes());
                out.extend([0; 4]);
                out
            }
            ExtendedHeader::Other(bytes) => bytes.clone(),
        };
        let mut out = cnmt.title_id.to_le_bytes().to_vec();
        out.extend(cnmt.version.to_le_bytes());
        out.extend([meta_type, 0]);
        out.extend((extended.len() as u16).to_le_bytes());
        out.extend((cnmt.contents.len() as u16).to_le_bytes());
        out.extend((cnmt.meta.len() as u16).to_le_bytes());
        out.extend([cnmt.attributes, 0, 0, 0]);
        out.extend(cnmt.required_download_system_version.to_le_bytes());
        out.extend([0; 4]);
        out.extend(extended);
        for c in &cnmt.contents {
            out.extend(c.hash);
            out.extend(c.content_id.0);
            out.extend(&c.size.to_le_bytes()[..5]);
            let content_type = match c.content_type {
                ContentType::Meta => 0,
                ContentType::Program => 1,
                ContentType::Data => 2,
                ContentType::Control => 3,
                ContentType::HtmlDocument => 4,
                ContentType::LegalInformation => 5,
                ContentType::DeltaFragment => 6,
                ContentType::Unknown(v) => v,
            };
            out.extend([c.attributes, content_type, c.id_offset]);
        }
        for m in &cnmt.meta {
            out.extend(m.title_id.to_le_bytes());
            out.extend(m.version.to_le_bytes());
            out.extend([0x80, m.attributes, 0, 0]);
        }
        out
    }

    pub fn record(content_type: ContentType, id: u8, size: u64) -> ContentRecord {
        ContentRecord {
            hash: [id; 32],
            content_id: ContentId([id; 16]),
            size,
            attributes: 0,
            content_type,
            id_offset: 0,
        }
    }

    #[test]
    fn round_trips_application_patch_and_add_on() {
        let headers = [
            (
                ContentMetaType::Application,
                ExtendedHeader::Application {
                    patch_id: 0x0100_0000_0000_0800,
                    required_system_version: 7,
                },
            ),
            (
                ContentMetaType::Patch,
                ExtendedHeader::Patch {
                    application_id: 0x0100_0000_0000_0000,
                    required_system_version: 9,
                },
            ),
            (
                ContentMetaType::AddOnContent,
                ExtendedHeader::AddOnContent {
                    application_id: 0x0100_0000_0000_0000,
                    required_application_version: 0x10000,
                },
            ),
        ];
        for (meta_type, extended) in headers {
            let mut second = record(ContentType::Program, 3, 0x12_3456_789a);
            second.id_offset = 1;
            let cnmt = Cnmt {
                title_id: 0x0400_c3f0_0006_e000,
                version: 0x0001_0000,
                meta_type,
                attributes: 0,
                required_download_system_version: 0,
                extended,
                contents: vec![
                    second,
                    record(ContentType::Program, 1, 0x4000),
                    record(ContentType::Control, 2, 0x200),
                ],
                meta: vec![],
            };
            let parsed = Cnmt::parse(&encode(&cnmt)).unwrap();
            assert_eq!(parsed, cnmt);
            assert_eq!(
                parsed.content(ContentType::Program).unwrap().content_id,
                ContentId([1; 16])
            );
            assert_eq!(parsed.content(ContentType::Control).unwrap().size, 0x200);
            assert!(parsed.content(ContentType::Meta).is_none());
        }
        assert_eq!(
            ContentId([0xab; 16]).file_name(),
            format!("{}.nca", "ab".repeat(16))
        );
        assert!(Cnmt::parse(&[0; 0x10]).is_err());
        let mut truncated = encode(&Cnmt {
            title_id: 1,
            version: 0,
            meta_type: ContentMetaType::Application,
            attributes: 0,
            required_download_system_version: 0,
            extended: ExtendedHeader::Other(vec![]),
            contents: vec![record(ContentType::Program, 1, 1)],
            meta: vec![],
        });
        truncated.pop();
        assert!(Cnmt::parse(&truncated).is_err());
    }
}
