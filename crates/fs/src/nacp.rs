//! NACP: the application control property (`control.nacp`) with names, publisher, version,
//! save data sizes and platform.

use crate::bytes::{array_at, c_string, u16_at, u32_at, u64_at};
use crate::error::{FsResult, corrupt};
use crate::vfs::{File, FileExt};

pub const NACP_SIZE: usize = 0x4000;
const LANGUAGE_ENTRY_SIZE: usize = 0x300;
const LANGUAGE_AREA: usize = 0x3000;
const MAX_LANGUAGES: usize = 32;

/// Languages in NACP title order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NacpLanguage {
    AmericanEnglish = 0,
    BritishEnglish = 1,
    Japanese = 2,
    French = 3,
    German = 4,
    LatinAmericanSpanish = 5,
    Spanish = 6,
    Italian = 7,
    Dutch = 8,
    CanadianFrench = 9,
    Portuguese = 10,
    Russian = 11,
    Korean = 12,
    TraditionalChinese = 13,
    SimplifiedChinese = 14,
    BrazilianPortuguese = 15,
    Polish = 16,
    Thai = 17,
}

/// Which console the application presents itself as (`ApparentPlatform`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApparentPlatform {
    Switch1,
    Switch2,
    Unknown(u8),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LanguageEntry {
    pub name: String,
    pub publisher: String,
}

/// A parsed NACP. The raw bytes stay available for services that hand them to the guest.
#[derive(Clone)]
pub struct Nacp {
    raw: Box<[u8; NACP_SIZE]>,
    titles: Vec<LanguageEntry>,
}

impl Nacp {
    pub fn read(file: &dyn File) -> FsResult<Self> {
        let len = file.size().min(NACP_SIZE as u64) as usize;
        Self::parse(&file.read_vec(0, len)?)
    }

    /// Parses NACP bytes. Shorter input is zero-extended to the full 0x4000 bytes; titles in the
    /// compressed format (raw deflate, used by titles with more than 16 languages) are inflated.
    pub fn parse(bytes: &[u8]) -> FsResult<Self> {
        let mut raw = Box::new([0u8; NACP_SIZE]);
        let len = bytes.len().min(NACP_SIZE);
        raw[..len].copy_from_slice(&bytes[..len]);
        let compressed = raw[0x3215] == 1;
        let titles = if compressed {
            let size = usize::from(u16_at(&raw[..], 0));
            let payload = raw
                .get(2..2 + size)
                .filter(|_| size <= LANGUAGE_AREA - 2)
                .ok_or_else(|| corrupt("NACP", "compressed title size past the title area"))?;
            let inflated = miniz_oxide::inflate::decompress_to_vec_with_limit(
                payload,
                LANGUAGE_ENTRY_SIZE * MAX_LANGUAGES,
            )
            .map_err(|e| corrupt("NACP", format!("compressed titles: {e:?}")))?;
            parse_titles(&inflated)
        } else {
            parse_titles(&raw[..LANGUAGE_AREA])
        };
        Ok(Self { raw, titles })
    }

    pub fn raw(&self) -> &[u8; NACP_SIZE] {
        &self.raw
    }

    /// Every language slot, empty ones included, in [`NacpLanguage`] order.
    pub fn titles(&self) -> &[LanguageEntry] {
        &self.titles
    }

    /// The entry for `language` if it has a name.
    pub fn title(&self, language: NacpLanguage) -> Option<&LanguageEntry> {
        self.titles
            .get(language as usize)
            .filter(|t| !t.name.is_empty())
    }

    /// The entry for `language`, falling back to American English and then to the first
    /// language with a name.
    pub fn preferred_title(&self, language: NacpLanguage) -> Option<&LanguageEntry> {
        self.title(language)
            .or_else(|| self.title(NacpLanguage::AmericanEnglish))
            .or_else(|| self.titles.iter().find(|t| !t.name.is_empty()))
    }

    pub fn display_version(&self) -> String {
        c_string(&self.raw[0x3060..0x3070])
    }

    pub fn startup_user_account(&self) -> u8 {
        self.raw[0x3025]
    }

    /// Whether the title forbids switching the user account while it runs (non-zero locks it).
    pub fn user_account_switch_lock(&self) -> u8 {
        self.raw[0x3026]
    }

    pub fn supported_languages(&self) -> u32 {
        u32_at(&self.raw[..], 0x302c)
    }

    pub fn parental_control(&self) -> u32 {
        u32_at(&self.raw[..], 0x3030)
    }

    pub fn screenshot(&self) -> u8 {
        self.raw[0x3034]
    }

    pub fn video_capture(&self) -> u8 {
        self.raw[0x3035]
    }

    pub fn presence_group_id(&self) -> u64 {
        u64_at(&self.raw[..], 0x3038)
    }

    pub fn rating_age(&self) -> [u8; 0x20] {
        array_at(&self.raw[..], 0x3040)
    }

    pub fn add_on_content_base_id(&self) -> u64 {
        u64_at(&self.raw[..], 0x3070)
    }

    pub fn save_data_owner_id(&self) -> u64 {
        u64_at(&self.raw[..], 0x3078)
    }

    pub fn user_account_save_data_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x3080)
    }

    pub fn user_account_save_data_journal_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x3088)
    }

    pub fn device_save_data_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x3090)
    }

    pub fn device_save_data_journal_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x3098)
    }

    pub fn bcat_delivery_cache_storage_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x30a0)
    }

    pub fn seed_for_pseudo_device_id(&self) -> u64 {
        u64_at(&self.raw[..], 0x30f8)
    }

    pub fn temporary_storage_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x3168)
    }

    pub fn cache_storage_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x3170)
    }

    pub fn cache_storage_journal_size(&self) -> u64 {
        u64_at(&self.raw[..], 0x3178)
    }

    pub fn cache_storage_max_index(&self) -> u16 {
        u16_at(&self.raw[..], 0x3188)
    }

    pub fn program_index(&self) -> u8 {
        self.raw[0x3212]
    }

    pub fn apparent_platform(&self) -> ApparentPlatform {
        match self.raw[0x3217] {
            0 => ApparentPlatform::Switch1,
            1 => ApparentPlatform::Switch2,
            other => ApparentPlatform::Unknown(other),
        }
    }
}

fn parse_titles(area: &[u8]) -> Vec<LanguageEntry> {
    area.as_chunks::<LANGUAGE_ENTRY_SIZE>()
        .0
        .iter()
        .map(|entry| LanguageEntry {
            name: c_string(&entry[..0x200]),
            publisher: c_string(&entry[0x200..]),
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::vfs::MemFile;

    pub fn synthetic_nacp(name: &str, publisher: &str, version: &str) -> Vec<u8> {
        let mut raw = vec![0u8; NACP_SIZE];
        for (lang, entry) in raw[..LANGUAGE_AREA]
            .as_chunks_mut::<LANGUAGE_ENTRY_SIZE>()
            .0
            .iter_mut()
            .enumerate()
        {
            if lang == NacpLanguage::AmericanEnglish as usize
                || lang == NacpLanguage::Japanese as usize
            {
                entry[..name.len()].copy_from_slice(name.as_bytes());
                entry[0x200..0x200 + publisher.len()].copy_from_slice(publisher.as_bytes());
            }
        }
        raw[0x3060..0x3060 + version.len()].copy_from_slice(version.as_bytes());
        raw[0x3070..0x3078].copy_from_slice(&0x0100_0000_0000_1000u64.to_le_bytes());
        raw[0x3078..0x3080].copy_from_slice(&0x0100_0000_0000_0000u64.to_le_bytes());
        raw[0x3080..0x3088].copy_from_slice(&0x40_0000u64.to_le_bytes());
        raw[0x30f8..0x3100].copy_from_slice(&0x1234_5678u64.to_le_bytes());
        raw
    }

    #[test]
    fn reads_fields_and_falls_back_between_languages() {
        let raw = synthetic_nacp("Squid Game", "Lichii", "1.2.3");
        let nacp = Nacp::read(MemFile::from_bytes(raw).as_ref()).unwrap();
        assert_eq!(nacp.titles().len(), 16);
        assert_eq!(
            nacp.title(NacpLanguage::Japanese).unwrap().name,
            "Squid Game"
        );
        assert!(nacp.title(NacpLanguage::German).is_none());
        let fallback = nacp.preferred_title(NacpLanguage::German).unwrap();
        assert_eq!(fallback.publisher, "Lichii");
        assert_eq!(nacp.display_version(), "1.2.3");
        assert_eq!(nacp.add_on_content_base_id(), 0x0100_0000_0000_1000);
        assert_eq!(nacp.user_account_save_data_size(), 0x40_0000);
        assert_eq!(nacp.seed_for_pseudo_device_id(), 0x1234_5678);
        assert_eq!(nacp.apparent_platform(), ApparentPlatform::Switch1);
    }

    #[test]
    fn inflates_compressed_titles() {
        let mut titles = vec![0u8; LANGUAGE_ENTRY_SIZE * 18];
        let thai = &mut titles[NacpLanguage::Thai as usize * LANGUAGE_ENTRY_SIZE..];
        thai[..4].copy_from_slice(b"Thai");
        let compressed = miniz_oxide::deflate::compress_to_vec(&titles, 9);
        let mut raw = vec![0u8; NACP_SIZE];
        raw[..2].copy_from_slice(&(compressed.len() as u16).to_le_bytes());
        raw[2..2 + compressed.len()].copy_from_slice(&compressed);
        raw[0x3215] = 1;
        raw[0x3217] = 1;
        let nacp = Nacp::parse(&raw).unwrap();
        assert_eq!(nacp.titles().len(), 18);
        assert_eq!(nacp.title(NacpLanguage::Thai).unwrap().name, "Thai");
        assert_eq!(
            nacp.preferred_title(NacpLanguage::French).unwrap().name,
            "Thai"
        );
        assert_eq!(nacp.apparent_platform(), ApparentPlatform::Switch2);

        raw[2] ^= 0xff;
        raw[3] ^= 0xff;
        assert!(Nacp::parse(&raw).is_err());
        raw[..2].copy_from_slice(&0xffffu16.to_le_bytes());
        assert!(Nacp::parse(&raw).is_err());
    }
}
