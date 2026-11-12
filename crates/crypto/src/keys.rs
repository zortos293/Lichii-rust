use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use crate::aes::AesEcb;
use crate::hex::parse_hex;
use crate::{Key128, Key256};

/// Number of key revisions (`master_key_00` .. `master_key_1f`) a key set can hold.
pub const KEY_REVISIONS: usize = 0x20;

const FIRST_DERIVED_MASTER_KEY_REVISION: usize = 6;
const LAST_TSEC_ROOT_KEY_REVISION: usize = 8;
const TSEC_ROOT_KEYS: usize = LAST_TSEC_ROOT_KEY_REVISION - FIRST_DERIVED_MASTER_KEY_REVISION + 1;

/// Which key area key decrypts an NCA's key area (NCA header byte 0x207).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyAreaKeyType {
    Application = 0,
    Ocean = 1,
    System = 2,
}

impl KeyAreaKeyType {
    pub const ALL: [Self; 3] = [Self::Application, Self::Ocean, Self::System];

    /// The key type an NCA header's key index byte selects; `None` for unknown indices.
    pub fn from_index(index: u8) -> Option<Self> {
        Self::ALL.get(usize::from(index)).copied()
    }

    /// The name `prod.keys` uses, as in `key_area_key_<name>_XX`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Application => "application",
            Self::Ocean => "ocean",
            Self::System => "system",
        }
    }
}

/// The SD card keys `sd_card_save_key` and `sd_card_nca_key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SdKeyType {
    Save = 0,
    Nca = 1,
}

/// A ticket's rights id: the program id (big-endian) followed by the key generation.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RightsId(pub [u8; 16]);

impl RightsId {
    /// An all-zero rights id means the NCA uses its own key area instead of a title key.
    pub fn is_zero(&self) -> bool {
        self.0 == [0; 16]
    }
}

impl fmt::Display for RightsId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|b| write!(f, "{b:02x}"))
    }
}

impl fmt::Debug for RightsId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RightsId({self})")
    }
}

/// A key the caller asked for that the loaded files neither listed nor let us derive. The
/// message names the key the way `prod.keys`/`title.keys` do, so it can be shown to the user.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum KeyError {
    #[error("missing key `{0}`")]
    Missing(String),
    #[error("missing title key for rights id {0}")]
    MissingTitleKey(RightsId),
}

/// A line of a key file that names a known key but could not be used.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MalformedLine {
    pub line: usize,
    pub name: String,
    pub reason: &'static str,
}

/// What loading one key file did. Unknown key names are not errors: `prod.keys` dumps carry
/// many keys an emulator never needs.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub loaded: usize,
    pub ignored: Vec<String>,
    pub malformed: Vec<MalformedLine>,
}

/// Console keys loaded from `prod.keys`/`title.keys`, plus everything derivable from them.
#[derive(Clone, Default)]
pub struct KeySet {
    header_key: Option<Key256>,
    header_kek_source: Option<Key128>,
    header_key_source: Option<Key256>,
    aes_kek_generation_source: Option<Key128>,
    aes_key_generation_source: Option<Key128>,
    master_key_source: Option<Key128>,
    titlekek_source: Option<Key128>,
    key_area_key_sources: [Option<Key128>; 3],
    mariko_kek: Option<Key128>,
    tsec_root_keys: [Option<Key128>; TSEC_ROOT_KEYS],
    master_kek_sources: [Option<Key128>; KEY_REVISIONS],
    mariko_master_kek_sources: [Option<Key128>; KEY_REVISIONS],
    master_keys: [Option<Key128>; KEY_REVISIONS],
    titlekeks: [Option<Key128>; KEY_REVISIONS],
    key_area_keys: [[Option<Key128>; 3]; KEY_REVISIONS],
    sd_card_kek_source: Option<Key128>,
    sd_seed: Option<Key128>,
    sd_card_key_sources: [Option<Key256>; 2],
    sd_card_keys: [Option<Key256>; 2],
    bis_keys: [[Option<Key128>; 2]; 4],
    title_keys: HashMap<RightsId, Key128>,
}

enum Slot<'a> {
    Key128(&'a mut Option<Key128>),
    Key256(&'a mut Option<Key256>),
    Split(&'a mut [Option<Key128>; 2]),
}

fn revision(name: &str, prefix: &str) -> Option<usize> {
    let digits = name.strip_prefix(prefix)?;
    if digits.len() != 2 {
        return None;
    }
    usize::from_str_radix(digits, 16).ok()
}

fn generate_kek(source: &Key128, master: &Key128, kek_seed: &Key128, key_seed: &Key128) -> Key128 {
    let kek = AesEcb::new(master).decrypt_key(kek_seed);
    let source_key = AesEcb::new(&kek).decrypt_key(source);
    if *key_seed == [0; 16] {
        return source_key;
    }
    AesEcb::new(&source_key).decrypt_key(key_seed)
}

fn missing(name: impl Into<String>) -> KeyError {
    KeyError::Missing(name.into())
}

impl KeySet {
    /// An empty key set; every accessor reports its key as missing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Loads `prod.keys`, `title.keys` and `console.keys` from `dir` (each optional), then
    /// derives what the loaded keys allow. Returns the per-file reports of the files that exist.
    pub fn load_dir(dir: &Path) -> std::io::Result<(Self, Vec<(String, LoadReport)>)> {
        let mut keys = Self::new();
        let mut reports = Vec::new();
        for (file, title) in [
            ("prod.keys", false),
            ("console.keys", false),
            ("title.keys", true),
        ] {
            let text = match std::fs::read_to_string(dir.join(file)) {
                Ok(text) => text,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            let report = if title {
                keys.load_title_keys(&text)
            } else {
                keys.load_prod_keys(&text)
            };
            if !report.malformed.is_empty() {
                tracing::warn!(file, malformed = ?report.malformed, "unusable key lines");
            }
            reports.push((file.to_owned(), report));
        }
        keys.derive();
        Ok((keys, reports))
    }

    /// Parses a `prod.keys`-style file (`name = hex` per line). Later entries win. Does not
    /// derive; call [`KeySet::derive`] after loading every file.
    pub fn load_prod_keys(&mut self, text: &str) -> LoadReport {
        let mut report = LoadReport::default();
        for (index, name, value) in key_lines(text) {
            let name = name.to_ascii_lowercase();
            let Some(slot) = self.slot(&name) else {
                report.ignored.push(name);
                continue;
            };
            let stored = match slot {
                Slot::Key128(key) => parse_hex(value).map(|v| *key = Some(v)),
                Slot::Key256(key) => parse_hex(value).map(|v| *key = Some(v)),
                Slot::Split(halves) => parse_hex::<32>(value).map(|v| {
                    halves[0] = Some(v[..16].try_into().unwrap());
                    halves[1] = Some(v[16..].try_into().unwrap());
                }),
            };
            match stored {
                Some(()) => report.loaded += 1,
                None => report.malformed.push(MalformedLine {
                    line: index + 1,
                    name,
                    reason: "value is not hex of the key's length",
                }),
            }
        }
        report
    }

    /// Parses a `title.keys` file: `rights_id = encrypted_title_key`, both 16 bytes of hex.
    pub fn load_title_keys(&mut self, text: &str) -> LoadReport {
        let mut report = LoadReport::default();
        for (index, name, value) in key_lines(text) {
            match (parse_hex::<16>(name), parse_hex::<16>(value)) {
                (Some(rights_id), Some(key)) => {
                    self.title_keys.insert(RightsId(rights_id), key);
                    report.loaded += 1;
                }
                _ => report.malformed.push(MalformedLine {
                    line: index + 1,
                    name: name.to_owned(),
                    reason: "rights id and title key must be 16 bytes of hex each",
                }),
            }
        }
        report
    }

    fn slot(&mut self, name: &str) -> Option<Slot<'_>> {
        let slot = match name {
            "header_key" => Slot::Key256(&mut self.header_key),
            "header_kek_source" => Slot::Key128(&mut self.header_kek_source),
            "header_key_source" => Slot::Key256(&mut self.header_key_source),
            "aes_kek_generation_source" => Slot::Key128(&mut self.aes_kek_generation_source),
            "aes_key_generation_source" => Slot::Key128(&mut self.aes_key_generation_source),
            "master_key_source" => Slot::Key128(&mut self.master_key_source),
            "titlekek_source" => Slot::Key128(&mut self.titlekek_source),
            "key_area_key_application_source" => Slot::Key128(&mut self.key_area_key_sources[0]),
            "key_area_key_ocean_source" => Slot::Key128(&mut self.key_area_key_sources[1]),
            "key_area_key_system_source" => Slot::Key128(&mut self.key_area_key_sources[2]),
            "mariko_kek" => Slot::Key128(&mut self.mariko_kek),
            "sd_card_kek_source" => Slot::Key128(&mut self.sd_card_kek_source),
            "sd_seed" => Slot::Key128(&mut self.sd_seed),
            "sd_card_save_key_source" => Slot::Key256(&mut self.sd_card_key_sources[0]),
            "sd_card_nca_key_source" => Slot::Key256(&mut self.sd_card_key_sources[1]),
            "sd_card_save_key" => Slot::Key256(&mut self.sd_card_keys[0]),
            "sd_card_nca_key" => Slot::Key256(&mut self.sd_card_keys[1]),
            _ => return self.indexed_slot(name),
        };
        Some(slot)
    }

    fn indexed_slot(&mut self, name: &str) -> Option<Slot<'_>> {
        if let Some(rest) = name.strip_prefix("bis_key_") {
            return match rest.as_bytes() {
                [b'0', p @ b'0'..=b'3'] => {
                    Some(Slot::Split(&mut self.bis_keys[usize::from(p - b'0')]))
                }
                [p @ b'0'..=b'3', b'_', ..] => {
                    let half = match &rest[2..] {
                        "crypt" => 0,
                        "tweak" => 1,
                        _ => return None,
                    };
                    Some(Slot::Key128(
                        &mut self.bis_keys[usize::from(p - b'0')][half],
                    ))
                }
                _ => None,
            };
        }
        if let Some(rev) = revision(name, "tsec_root_key_") {
            return self.tsec_root_keys.get_mut(rev).map(Slot::Key128);
        }
        for (index, kind) in KeyAreaKeyType::ALL.iter().enumerate() {
            let prefix = format!("key_area_key_{}_", kind.name());
            if let Some(rev) = revision(name, &prefix) {
                return self
                    .key_area_keys
                    .get_mut(rev)
                    .map(|keys| Slot::Key128(&mut keys[index]));
            }
        }
        let table: [(&str, &mut [Option<Key128>; KEY_REVISIONS]); 4] = [
            ("master_key_", &mut self.master_keys),
            ("titlekek_", &mut self.titlekeks),
            ("master_kek_source_", &mut self.master_kek_sources),
            (
                "mariko_master_kek_source_",
                &mut self.mariko_master_kek_sources,
            ),
        ];
        for (prefix, keys) in table {
            if let Some(rev) = revision(name, prefix) {
                return keys.get_mut(rev).map(Slot::Key128);
            }
        }
        None
    }

    /// Fills in every key the loaded files lack but can be computed from, the way the console
    /// does: master keys from TSEC root or Mariko keks, then key area keys and titlekeks per
    /// revision, then the header key and SD card keys. Keys the files listed are never replaced.
    pub fn derive(&mut self) {
        if let Some(master_source) = self.master_key_source {
            for rev in FIRST_DERIVED_MASTER_KEY_REVISION..KEY_REVISIONS {
                if self.master_keys[rev].is_some() {
                    continue;
                }
                let root = self.tsec_root_keys
                    [rev.min(LAST_TSEC_ROOT_KEY_REVISION) - FIRST_DERIVED_MASTER_KEY_REVISION];
                let kek = root
                    .zip(self.master_kek_sources[rev])
                    .or(self.mariko_kek.zip(self.mariko_master_kek_sources[rev]))
                    .map(|(kek_key, source)| AesEcb::new(&kek_key).decrypt_key(&source));
                if let Some(kek) = kek {
                    self.master_keys[rev] = Some(AesEcb::new(&kek).decrypt_key(&master_source));
                    tracing::debug!(revision = rev, "derived master key");
                }
            }
        }

        let generation_sources = self
            .aes_kek_generation_source
            .zip(self.aes_key_generation_source);
        for rev in 0..KEY_REVISIONS {
            let Some(master) = self.master_keys[rev] else {
                continue;
            };
            if let Some((kek_seed, key_seed)) = generation_sources {
                for (slot, source) in self.key_area_keys[rev]
                    .iter_mut()
                    .zip(&self.key_area_key_sources)
                {
                    if let (None, Some(source)) = (&slot, source) {
                        *slot = Some(generate_kek(source, &master, &kek_seed, &key_seed));
                    }
                }
            }
            if let (None, Some(source)) = (self.titlekeks[rev], self.titlekek_source) {
                self.titlekeks[rev] = Some(AesEcb::new(&master).decrypt_key(&source));
            }
        }

        let Some(((kek_seed, key_seed), master)) = generation_sources.zip(self.master_keys[0])
        else {
            return;
        };
        if let (None, Some(kek_source), Some(mut key)) = (
            self.header_key,
            self.header_kek_source,
            self.header_key_source,
        ) {
            let header_kek = generate_kek(&kek_source, &master, &kek_seed, &key_seed);
            AesEcb::new(&header_kek).decrypt(&mut key);
            self.header_key = Some(key);
        }
        if let (Some(kek_source), Some(seed)) = (self.sd_card_kek_source, self.sd_seed) {
            let sd_kek = AesEcb::new(&generate_kek(&kek_source, &master, &kek_seed, &key_seed));
            for (slot, source) in self.sd_card_keys.iter_mut().zip(&self.sd_card_key_sources) {
                if let (None, Some(source)) = (&slot, source) {
                    let mut key: Key256 = std::array::from_fn(|i| source[i] ^ seed[i % 16]);
                    sd_kek.decrypt(&mut key);
                    *slot = Some(key);
                }
            }
        }
    }

    /// The NCA header key (`header_key`).
    pub fn header_key(&self) -> Result<&Key256, KeyError> {
        self.header_key
            .as_ref()
            .ok_or_else(|| missing("header_key"))
    }

    /// The master key of key `revision` (`master_key_XX`, zero-based). An NCA's key generation
    /// maps to a revision through [`master_key_revision`].
    pub fn master_key(&self, revision: usize) -> Result<Key128, KeyError> {
        self.master_keys
            .get(revision)
            .copied()
            .flatten()
            .ok_or_else(|| missing(format!("master_key_{revision:02x}")))
    }

    /// The key area key for `revision` (an NCA's key generation minus one, see
    /// [`master_key_revision`]).
    pub fn key_area_key(&self, revision: usize, kind: KeyAreaKeyType) -> Result<Key128, KeyError> {
        self.key_area_keys
            .get(revision)
            .and_then(|keys| keys[kind as usize])
            .ok_or_else(|| missing(format!("key_area_key_{}_{revision:02x}", kind.name())))
    }

    /// The titlekek of key `revision` (`titlekek_XX`, zero-based), which decrypts title keys of
    /// NCAs with that revision. An NCA's key generation maps to a revision through
    /// [`master_key_revision`].
    pub fn titlekek(&self, revision: usize) -> Result<Key128, KeyError> {
        self.titlekeks
            .get(revision)
            .copied()
            .flatten()
            .ok_or_else(|| missing(format!("titlekek_{revision:02x}")))
    }

    /// The encrypted title key for `rights_id`, from `title.keys` or an added ticket.
    pub fn encrypted_title_key(&self, rights_id: &RightsId) -> Result<Key128, KeyError> {
        self.title_keys
            .get(rights_id)
            .copied()
            .ok_or(KeyError::MissingTitleKey(*rights_id))
    }

    /// The decrypted title key that opens an NCA with `rights_id`, using the titlekek of
    /// `revision`.
    pub fn title_key(&self, rights_id: &RightsId, revision: usize) -> Result<Key128, KeyError> {
        let encrypted = self.encrypted_title_key(rights_id)?;
        Ok(AesEcb::new(&self.titlekek(revision)?).decrypt_key(&encrypted))
    }

    /// Records an encrypted title key, for example from a ticket shipped in an NSP.
    pub fn add_title_key(&mut self, rights_id: RightsId, encrypted_key: Key128) {
        self.title_keys.insert(rights_id, encrypted_key);
    }

    /// The XTS key for SD card save data or NCAs, from the file or derived from `sd_seed`.
    pub fn sd_card_key(&self, kind: SdKeyType) -> Result<Key256, KeyError> {
        self.sd_card_keys[kind as usize].ok_or_else(|| match kind {
            SdKeyType::Save => missing("sd_card_save_key"),
            SdKeyType::Nca => missing("sd_card_nca_key"),
        })
    }

    /// The XTS key of BIS partition `index` (0-3): crypt key followed by tweak key.
    pub fn bis_key(&self, index: usize) -> Result<Key256, KeyError> {
        let [crypt, tweak] = self
            .bis_keys
            .get(index)
            .copied()
            .ok_or_else(|| missing(format!("bis_key_{index:02x}")))?;
        let crypt = crypt.ok_or_else(|| missing(format!("bis_key_{index}_crypt")))?;
        let tweak = tweak.ok_or_else(|| missing(format!("bis_key_{index}_tweak")))?;
        let mut key = [0; 32];
        key[..16].copy_from_slice(&crypt);
        key[16..].copy_from_slice(&tweak);
        Ok(key)
    }
}

impl fmt::Debug for KeySet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = |keys: &[Option<Key128>]| keys.iter().flatten().count();
        f.debug_struct("KeySet")
            .field("header_key", &self.header_key.is_some())
            .field("master_keys", &count(&self.master_keys))
            .field("titlekeks", &count(&self.titlekeks))
            .field(
                "key_area_keys",
                &self.key_area_keys.iter().map(|k| count(k)).sum::<usize>(),
            )
            .field("title_keys", &self.title_keys.len())
            .finish_non_exhaustive()
    }
}

/// Maps an NCA header's key generation to the revision of the keys that open it: generations 0
/// and 1 both use revision 0.
pub fn master_key_revision(key_generation: u8) -> usize {
    usize::from(key_generation.saturating_sub(1))
}

fn key_lines(text: &str) -> impl Iterator<Item = (usize, &str, &str)> {
    text.lines().enumerate().filter_map(|(index, line)| {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            return None;
        }
        let (name, value) = line.split_once(['=', ','])?;
        Some((index, name.trim(), value.trim()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(byte: u8) -> String {
        format!("{byte:02x}").repeat(16)
    }

    fn encrypt(key: &Key128, value: &Key128) -> Key128 {
        let mut out = *value;
        AesEcb::new(key).encrypt(&mut out);
        out
    }

    #[test]
    fn parses_reports_and_tolerates_unknown_keys() {
        let mut keys = KeySet::new();
        let report = keys.load_prod_keys(&format!(
            "# comment\n\
             MASTER_KEY_00 = {}\n\
             titlekek_01={}\n\
             keyblob_00 = 1234\n\
             key_area_key_ocean_02 = {}\n\
             header_key = {}{}\n\
             master_key_01 = 12345\n\
             bis_key_00 = {}{}\n\
             bis_key_1_crypt = {}\n",
            h(1),
            h(2),
            h(3),
            h(4),
            h(5),
            h(6),
            h(7),
            h(8)
        ));
        assert_eq!(report.loaded, 6);
        assert_eq!(report.ignored, ["keyblob_00"]);
        assert_eq!(report.malformed.len(), 1);
        assert_eq!(report.malformed[0].line, 7);
        assert_eq!(report.malformed[0].name, "master_key_01");

        assert_eq!(keys.master_key(0).unwrap(), [1; 16]);
        assert_eq!(keys.titlekek(1).unwrap(), [2; 16]);
        assert_eq!(
            keys.key_area_key(2, KeyAreaKeyType::Ocean).unwrap(),
            [3; 16]
        );
        let mut header = [4; 32];
        header[16..].fill(5);
        assert_eq!(keys.header_key().unwrap(), &header);
        let mut bis = [6; 32];
        bis[16..].fill(7);
        assert_eq!(keys.bis_key(0).unwrap(), bis);
        assert_eq!(
            keys.bis_key(1),
            Err(KeyError::Missing("bis_key_1_tweak".into()))
        );
        assert_eq!(
            keys.key_area_key(0x13, KeyAreaKeyType::System),
            Err(KeyError::Missing("key_area_key_system_13".into()))
        );
        assert_eq!(
            keys.master_key(1),
            Err(KeyError::Missing("master_key_01".into()))
        );
    }

    #[test]
    fn title_keys_decrypt_with_titlekek() {
        let mut keys = KeySet::new();
        let titlekek = [0x11; 16];
        let title_key = [0x22; 16];
        let rights = RightsId([0xab; 16]);
        let report = keys.load_title_keys(&format!(
            "{} = {}\nnot hex = 00\n",
            rights,
            crate::tests::hex_string(&encrypt(&titlekek, &title_key))
        ));
        assert_eq!(report.loaded, 1);
        assert_eq!(report.malformed.len(), 1);
        keys.load_prod_keys(&format!("titlekek_04 = {}", h(0x11)));
        assert_eq!(keys.title_key(&rights, 4).unwrap(), title_key);
        assert_eq!(
            keys.title_key(&rights, 3),
            Err(KeyError::Missing("titlekek_03".into()))
        );
        assert_eq!(
            keys.title_key(&RightsId([1; 16]), 4),
            Err(KeyError::MissingTitleKey(RightsId([1; 16])))
        );
    }

    #[test]
    fn derives_the_nca_key_chain_from_sources() {
        let master0 = [0x10; 16];
        let mariko_kek = [0x20; 16];
        let master_source = [0x30; 16];
        let kek_seed = [0x40; 16];
        let key_seed = [0x50; 16];
        let header_kek_source = [0x60; 16];
        let kak_source = [0x70; 16];
        let titlekek_source = [0x80; 16];
        let mariko_source_0a = [0x90; 16];

        let text = format!(
            "master_key_00 = {}\nmariko_kek = {}\nmaster_key_source = {}\n\
             aes_kek_generation_source = {}\naes_key_generation_source = {}\n\
             header_kek_source = {}\nheader_key_source = {}{}\n\
             key_area_key_application_source = {}\ntitlekek_source = {}\n\
             mariko_master_kek_source_0a = {}\nkey_area_key_application_0a = {}\n",
            h(0x10),
            h(0x20),
            h(0x30),
            h(0x40),
            h(0x50),
            h(0x60),
            h(0xa1),
            h(0xa2),
            h(0x70),
            h(0x80),
            h(0x90),
            h(0xee),
        );
        let mut keys = KeySet::new();
        assert!(keys.load_prod_keys(&text).malformed.is_empty());
        keys.derive();

        let dec = |key: &Key128, value: &Key128| AesEcb::new(key).decrypt_key(value);
        let kek = |source: &Key128, master: &Key128| {
            dec(&dec(&dec(master, &kek_seed), source), &key_seed)
        };

        let master_0a = dec(&dec(&mariko_kek, &mariko_source_0a), &master_source);
        assert_eq!(keys.master_key(0x0a).unwrap(), master_0a);
        assert!(keys.master_key(0x0b).is_err());

        assert_eq!(
            keys.key_area_key(0, KeyAreaKeyType::Application).unwrap(),
            kek(&kak_source, &master0)
        );
        assert_eq!(
            keys.key_area_key(0x0a, KeyAreaKeyType::Application)
                .unwrap(),
            [0xee; 16]
        );
        assert!(keys.key_area_key(0, KeyAreaKeyType::Ocean).is_err());
        assert_eq!(keys.titlekek(0).unwrap(), dec(&master0, &titlekek_source));
        assert_eq!(
            keys.titlekek(0x0a).unwrap(),
            dec(&master_0a, &titlekek_source)
        );

        let header_kek = kek(&header_kek_source, &master0);
        let mut header = [0xa1; 32];
        header[16..].fill(0xa2);
        AesEcb::new(&header_kek).decrypt(&mut header);
        assert_eq!(keys.header_key().unwrap(), &header);
    }

    #[test]
    fn tsec_root_keys_cover_revisions_six_and_up() {
        let mut keys = KeySet::new();
        keys.load_prod_keys(&format!(
            "master_key_source = {}\ntsec_root_key_02 = {}\nmaster_kek_source_0c = {}\n\
             tsec_root_key_00 = {}\nmaster_kek_source_06 = {}\n",
            h(1),
            h(2),
            h(3),
            h(4),
            h(5)
        ));
        keys.derive();
        let dec = |key: &Key128, value: &Key128| AesEcb::new(key).decrypt_key(value);
        assert_eq!(
            keys.master_key(0x0c).unwrap(),
            dec(&dec(&[2; 16], &[3; 16]), &[1; 16])
        );
        assert_eq!(
            keys.master_key(6).unwrap(),
            dec(&dec(&[4; 16], &[5; 16]), &[1; 16])
        );
        assert!(keys.master_key(7).is_err());
    }

    #[test]
    fn key_generation_to_revision() {
        assert_eq!(master_key_revision(0), 0);
        assert_eq!(master_key_revision(1), 0);
        assert_eq!(master_key_revision(2), 1);
        assert_eq!(master_key_revision(0x14), 0x13);
    }

    #[test]
    fn load_dir_reads_present_files() {
        let dir = std::env::temp_dir().join(format!("lichii-keys-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("prod.keys"), format!("master_key_00 = {}\n", h(9))).unwrap();
        let (keys, reports) = KeySet::load_dir(&dir).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].0, "prod.keys");
        assert_eq!(keys.master_key(0).unwrap(), [9; 16]);
    }
}
