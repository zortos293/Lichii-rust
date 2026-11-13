//! A persistent shader cache: translated modules with a record of every question their
//! translation asked the guest, so a module is reused only when the live guest answers all of
//! them the same way (`handoffs/shader-cache.md` in the Captain's drive is the contract).
//!
//! A file is a [`header`] followed by records the renderer appends as it translates. Each
//! record is a length, a checksum and a postcard payload: a module, stored once whatever number
//! of entries share it, or an entry, which names its module by hash and follows it. A header from another format or
//! translator build makes the whole file stale. A damaged record ends the readable part, so a
//! crash while appending loses only that record.

mod record;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

pub use self::record::CodeCheck;
use self::record::{EnvRecord, Logger};
use crate::backend::Bindings;
use crate::hash::StableHasher;
pub use crate::translate::TRANSLATOR_ID;
use crate::translate::{TranslateOptions, header_size};
use crate::{RuntimeInfo, ShaderEnvironment, ShaderModule, ShaderStage, TranslateError};

/// Bump on any change to the header or record layout.
pub const FORMAT_VERSION: u32 = 4;
/// Bytes of [`header`], the start of every cache file.
pub const HEADER_LEN: usize = 32;
const MAGIC: [u8; 8] = *b"LICHISHC";
const RECORD_HEADER_LEN: usize = 12;

/// The header that starts a new cache file.
pub fn header() -> [u8; HEADER_LEN] {
    let mut header = [0; HEADER_LEN];
    header[..8].copy_from_slice(&MAGIC);
    header[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[16..24].copy_from_slice(&TRANSLATOR_ID.to_le_bytes());
    header
}

/// Why a cache file cannot be used; delete it and start a new one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum StaleCache {
    #[error("not a shader cache file")]
    NotACache,
    #[error("shader cache format {0}, expected {FORMAT_VERSION}")]
    Format(u32),
    #[error("shader cache written by another translator build")]
    Translator,
}

/// The record bytes of `file`, after its header.
pub fn records(file: &[u8]) -> Result<&[u8], StaleCache> {
    let (head, records) = file
        .split_at_checked(HEADER_LEN)
        .ok_or(StaleCache::NotACache)?;
    if head[..8] != MAGIC {
        return Err(StaleCache::NotACache);
    }
    let format = u32::from_le_bytes(head[8..12].try_into().expect("4 bytes"));
    if format != FORMAT_VERSION {
        return Err(StaleCache::Format(format));
    }
    if head[16..24] != TRANSLATOR_ID.to_le_bytes() {
        return Err(StaleCache::Translator);
    }
    Ok(records)
}

/// The entries of the record bytes [`records`] returned, in order, up to the first damaged
/// record.
pub fn entries(records: &[u8]) -> Entries<'_> {
    Entries {
        records,
        read: 0,
        modules: HashMap::new(),
    }
}

/// Iterator over the [`CacheEntry`]s of a file's records; [`entries`] makes one.
pub struct Entries<'a> {
    records: &'a [u8],
    read: usize,
    /// The modules read so far, by hash.
    modules: HashMap<u64, Arc<ShaderModule>>,
}

impl Entries<'_> {
    /// Bytes of records decoded so far; once the iterator ends, the length of the readable
    /// part, where the next record should be appended.
    pub fn valid_len(&self) -> usize {
        self.read
    }
}

impl Iterator for Entries<'_> {
    type Item = CacheEntry;

    fn next(&mut self) -> Option<CacheEntry> {
        loop {
            let rest = &self.records[self.read..];
            let (head, rest) = rest.split_at_checked(RECORD_HEADER_LEN)?;
            let len = u32::from_le_bytes(head[..4].try_into().expect("4 bytes")) as usize;
            let checksum = u64::from_le_bytes(head[4..].try_into().expect("8 bytes"));
            let payload = rest.get(..len)?;
            if checksum_of(payload) != checksum {
                return None;
            }
            let entry = match postcard::from_bytes(payload).ok()? {
                Stored::Module(module) => {
                    self.modules.insert(module.hash, Arc::from(module));
                    None
                }
                Stored::Entry(stored) => {
                    let module = self.modules.get(&stored.module)?.clone();
                    Some(CacheEntry::from_stored(stored, module)?)
                }
            };
            self.read += RECORD_HEADER_LEN + len;
            if entry.is_some() {
                return entry;
            }
        }
    }
}

fn push_record(payload: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(&checksum_of(payload).to_le_bytes());
    out.extend_from_slice(payload);
}

fn checksum_of(payload: &[u8]) -> u64 {
    let mut hasher = StableHasher::new();
    hasher.write(payload);
    hasher.finish()
}

/// The translation inputs that are not guest state: the host profile, the resolution scale,
/// the pipeline's runtime info and the bindings the module starts at.
pub fn options_key(
    options: TranslateOptions<'_>,
    runtime: &RuntimeInfo,
    bindings: &Bindings,
) -> u64 {
    let mut hasher = StableHasher::new();
    format!(
        "{:?}{:?}{:?}{runtime:?}",
        options.profile, options.host, options.resolution_scale
    )
    .hash(&mut hasher);
    bindings.hash(&mut hasher);
    hasher.finish()
}

/// Bytes after a graphics program's header that the probe covers, and a compute program's.
const PROBE_GRAPHICS_CODE: u64 = 0x40;
const PROBE_COMPUTE_CODE: u64 = 0x80;

/// A key from live state that every entry for this program shares, computed without
/// translating: the stage, `options_key` and the program's first bytes (a graphics program's
/// header and first instructions). Entries are only candidates; [`CacheEntry::matches`] decides.
pub fn probe_key(
    a: Option<&dyn ShaderEnvironment>,
    env: &dyn ShaderEnvironment,
    options_key: u64,
) -> u64 {
    let mut hasher = StableHasher::new();
    options_key.hash(&mut hasher);
    for env in a.into_iter().chain([env]) {
        let stage = env.stage();
        let start = env.start_address();
        let (first, end) = if stage == ShaderStage::Compute {
            (start, start + PROBE_COMPUTE_CODE)
        } else {
            (
                start.wrapping_sub(header_size(env)),
                start + PROBE_GRAPHICS_CODE,
            )
        };
        stage.hash(&mut hasher);
        (start % 32).hash(&mut hasher);
        for address in (first..end).step_by(8) {
            env.read_u64(address).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// [`crate::compile`], also returning the entry to store.
pub fn compile(
    env: &dyn ShaderEnvironment,
    options: TranslateOptions<'_>,
    runtime: &RuntimeInfo,
    bindings: &mut Bindings,
) -> Result<CacheEntry, TranslateError> {
    let key = options_key(options, runtime, bindings);
    let probe = probe_key(None, env, key);
    let logger = Logger::new(env);
    let module = crate::compile(&logger, options, runtime, bindings)?;
    if module.info.reads_start_address {
        logger.pin_start_address();
    }
    Ok(CacheEntry {
        module: Arc::new(module),
        records: vec![logger.into_record()],
        options_key: key,
        probe_key: probe,
        bindings_after: *bindings,
    })
}

/// [`crate::compile_dual_vertex`], also returning the entry to store.
pub fn compile_dual_vertex(
    env_a: &dyn ShaderEnvironment,
    env_b: &dyn ShaderEnvironment,
    options: TranslateOptions<'_>,
    runtime: &RuntimeInfo,
    bindings: &mut Bindings,
) -> Result<CacheEntry, TranslateError> {
    let key = options_key(options, runtime, bindings);
    let probe = probe_key(Some(env_a), env_b, key);
    let (logger_a, logger_b) = (Logger::new(env_a), Logger::new(env_b));
    let module = crate::compile_dual_vertex(&logger_a, &logger_b, options, runtime, bindings)?;
    if module.info.reads_start_address {
        logger_a.pin_start_address();
        logger_b.pin_start_address();
    }
    Ok(CacheEntry {
        module: Arc::new(module),
        records: vec![logger_a.into_record(), logger_b.into_record()],
        options_key: key,
        probe_key: probe,
        bindings_after: *bindings,
    })
}

/// A translated module and the record that proves when it may be reused.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    module: Arc<ShaderModule>,
    /// VertexA's then the program's, or the program's alone.
    records: Vec<EnvRecord>,
    options_key: u64,
    probe_key: u64,
    bindings_after: Bindings,
}

#[derive(Serialize)]
enum StoredRef<'a> {
    Module(&'a ShaderModule),
    Entry(StoredEntryRef<'a>),
}

#[derive(Deserialize)]
enum Stored {
    Module(Box<ShaderModule>),
    Entry(StoredEntry),
}

#[derive(Serialize)]
struct StoredEntryRef<'a> {
    module: u64,
    records: &'a [EnvRecord],
    options_key: u64,
    probe_key: u64,
    bindings_after: Bindings,
}

#[derive(Deserialize)]
struct StoredEntry {
    /// The hash of a module stored before the entry.
    module: u64,
    records: Vec<EnvRecord>,
    options_key: u64,
    probe_key: u64,
    bindings_after: Bindings,
}

/// Appends `module` as one record, for the entries that follow to name.
pub fn encode_module(module: &ShaderModule, out: &mut Vec<u8>) {
    let payload =
        postcard::to_stdvec(&StoredRef::Module(module)).expect("serializing to a Vec cannot fail");
    push_record(&payload, out);
}

impl CacheEntry {
    fn from_stored(stored: StoredEntry, module: Arc<ShaderModule>) -> Option<CacheEntry> {
        matches!(stored.records.len(), 1 | 2).then(|| CacheEntry {
            module,
            records: stored.records,
            options_key: stored.options_key,
            probe_key: stored.probe_key,
            bindings_after: stored.bindings_after,
        })
    }

    /// The translated module to use on a hit.
    pub fn module(&self) -> &Arc<ShaderModule> {
        &self.module
    }

    /// The [`probe_key`] of the live state this entry was translated from, to index entries by.
    pub fn probe_key(&self) -> u64 {
        self.probe_key
    }

    /// The bindings after this module; on a hit, the caller's bindings become these.
    pub fn bindings_after(&self) -> Bindings {
        self.bindings_after
    }

    /// The program bytes translation read, per program (`[VertexA, program]`, `(0, 0)` without
    /// VertexA): the first byte's offset from the program's first instruction, negative for its
    /// header, and the length.
    pub fn code_ranges(&self) -> [(i64, u64); 2] {
        match self.records.as_slice() {
            [a, program] => [a.code_range(), program.code_range()],
            [program] => [(0, 0), program.code_range()],
            _ => unreachable!("an entry has one or two records"),
        }
    }

    /// Whether translating now would give this module: the options match and the live
    /// environments answer every recorded question, program memory included, the same way.
    /// `code` hashes each range of live program memory once across the entries a lookup tries.
    pub fn matches(
        &self,
        a: Option<&dyn ShaderEnvironment>,
        env: &dyn ShaderEnvironment,
        options_key: u64,
        code: &mut CodeCheck,
    ) -> bool {
        self.replay(a, env, options_key, Some(code))
    }

    /// [`Self::matches`] without program memory, for a caller that proved the code equal
    /// itself.
    pub fn matches_state(
        &self,
        a: Option<&dyn ShaderEnvironment>,
        env: &dyn ShaderEnvironment,
        options_key: u64,
    ) -> bool {
        self.replay(a, env, options_key, None)
    }

    fn replay(
        &self,
        a: Option<&dyn ShaderEnvironment>,
        env: &dyn ShaderEnvironment,
        options_key: u64,
        mut code: Option<&mut CodeCheck>,
    ) -> bool {
        if options_key != self.options_key {
            return false;
        }
        match (self.records.as_slice(), a) {
            ([record_a, record], Some(a)) => {
                record_a.matches(a, 0, code.as_deref_mut()) && record.matches(env, 1, code)
            }
            ([record], None) => record.matches(env, 1, code),
            _ => false,
        }
    }

    pub fn options_key(&self) -> u64 {
        self.options_key
    }

    /// Diagnostic: why this entry does not match, or None when it does.
    pub fn explain_mismatch(
        &self,
        a: Option<&dyn ShaderEnvironment>,
        env: &dyn ShaderEnvironment,
        options_key: u64,
    ) -> Option<String> {
        if options_key != self.options_key {
            return Some("options key".into());
        }
        let envs: Vec<&dyn ShaderEnvironment> = match (self.records.as_slice(), a) {
            ([_, _], Some(a)) => vec![a, env],
            ([_], None) => vec![env],
            _ => return Some("vertex A presence".into()),
        };
        for (side, (record, env)) in self.records.iter().zip(envs).enumerate() {
            if let Some(reason) = record.explain(env) {
                return Some(format!("side {side}: {reason}"));
            }
        }
        None
    }

    /// Appends this entry as one record. Its module must be in the file already, before it:
    /// [`encode_module`] writes it.
    pub fn encode(&self, out: &mut Vec<u8>) {
        let payload = postcard::to_stdvec(&StoredRef::Entry(StoredEntryRef {
            module: self.module.hash,
            records: &self.records,
            options_key: self.options_key,
            probe_key: self.probe_key,
            bindings_after: self.bindings_after,
        }))
        .expect("serializing to a Vec cannot fail");
        push_record(&payload, out);
    }
}
