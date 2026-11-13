//! A persistent shader cache: translated modules with a record of every question their
//! translation asked the guest, so a module is reused only when the live guest answers all of
//! them the same way (`handoffs/shader-cache.md` in the Captain's drive is the contract).
//!
//! A file is a [`header`] followed by records the renderer appends as it translates. Each
//! record is a length, a checksum and a postcard payload. A header from another format or
//! translator build makes the whole file stale. A damaged record ends the readable part, so a
//! crash while appending loses only that record.

mod record;
#[cfg(test)]
mod tests;

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use self::record::{EnvRecord, Logger};
use crate::backend::Bindings;
use crate::hash::StableHasher;
pub use crate::translate::TRANSLATOR_ID;
use crate::translate::{TranslateOptions, header_size};
use crate::{RuntimeInfo, ShaderEnvironment, ShaderModule, ShaderStage, TranslateError};

/// Bump on any change to the header or record layout.
pub const FORMAT_VERSION: u32 = 1;
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
    Entries { records, read: 0 }
}

pub struct Entries<'a> {
    records: &'a [u8],
    read: usize,
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
        let rest = &self.records[self.read..];
        let (head, rest) = rest.split_at_checked(RECORD_HEADER_LEN)?;
        let len = u32::from_le_bytes(head[..4].try_into().expect("4 bytes")) as usize;
        let checksum = u64::from_le_bytes(head[4..].try_into().expect("8 bytes"));
        let payload = rest.get(..len)?;
        if checksum_of(payload) != checksum {
            return None;
        }
        let stored: Stored = postcard::from_bytes(payload).ok()?;
        let entry = CacheEntry::from_stored(stored)?;
        self.read += RECORD_HEADER_LEN + len;
        Some(entry)
    }
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
struct StoredRef<'a> {
    module: &'a ShaderModule,
    records: &'a [EnvRecord],
    options_key: u64,
    probe_key: u64,
    bindings_after: Bindings,
}

#[derive(Deserialize)]
struct Stored {
    module: ShaderModule,
    records: Vec<EnvRecord>,
    options_key: u64,
    probe_key: u64,
    bindings_after: Bindings,
}

impl CacheEntry {
    fn from_stored(stored: Stored) -> Option<CacheEntry> {
        matches!(stored.records.len(), 1 | 2).then(|| CacheEntry {
            module: Arc::new(stored.module),
            records: stored.records,
            options_key: stored.options_key,
            probe_key: stored.probe_key,
            bindings_after: stored.bindings_after,
        })
    }

    pub fn module(&self) -> &Arc<ShaderModule> {
        &self.module
    }

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
    pub fn matches(
        &self,
        a: Option<&dyn ShaderEnvironment>,
        env: &dyn ShaderEnvironment,
        options_key: u64,
    ) -> bool {
        self.replay(a, env, options_key, true)
    }

    /// [`Self::matches`] without program memory, for a caller that proved the code equal
    /// itself.
    pub fn matches_state(
        &self,
        a: Option<&dyn ShaderEnvironment>,
        env: &dyn ShaderEnvironment,
        options_key: u64,
    ) -> bool {
        self.replay(a, env, options_key, false)
    }

    fn replay(
        &self,
        a: Option<&dyn ShaderEnvironment>,
        env: &dyn ShaderEnvironment,
        options_key: u64,
        code: bool,
    ) -> bool {
        if options_key != self.options_key {
            return false;
        }
        match (self.records.as_slice(), a) {
            ([record_a, record], Some(a)) => record_a.matches(a, code) && record.matches(env, code),
            ([record], None) => record.matches(env, code),
            _ => false,
        }
    }

    /// Appends this entry as one record.
    pub fn encode(&self, out: &mut Vec<u8>) {
        let payload = postcard::to_stdvec(&StoredRef {
            module: &self.module,
            records: &self.records,
            options_key: self.options_key,
            probe_key: self.probe_key,
            bindings_after: self.bindings_after,
        })
        .expect("serializing to a Vec cannot fail");
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&checksum_of(&payload).to_le_bytes());
        out.extend_from_slice(&payload);
    }
}
