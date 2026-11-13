//! Translated shaders by guest program.
//!
//! A module is keyed by the bytes of its program (the header and every instruction the
//! translator read), the register and pipeline state it was translated for, and the descriptor
//! bindings it starts at; the texture queries the translator made are replayed on every hit, so
//! a texture changing type under an unchanged program translates it again. Never by the module
//! hash, which costs a full translation to compute.
//!
//! Hashing a program's bytes on every draw would cost more than the draw, so the hash is
//! remembered per device address and recomputed only after a guest write touched it. A hit
//! is a page table lookup, two map lookups and the replay; it allocates nothing and takes no
//! lock. Failed translations are cached too: their draws are skipped without translating again.
//! A failure no texture header can have caused ([`TranslateError::depends_on_textures`]) is
//! remembered without the reads that led to it, so a constant buffer word that changes from
//! draw to draw (a per-material bindless handle) does not translate the failing program again
//! on every draw; one that changes a jump table does not either. A failure a texture header may
//! have caused keeps its reads and translates again under other headers. Each key keeps at most
//! [`MAX_VARIANTS`] variants.
//!
//! With the title's disk cache open, a miss first replays the stored entries that share the
//! program's probe key (`lichii_shader::cache`) and reuses a module whose every recorded answer
//! still holds; new translations are appended to the file at the next flush.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_common::settings::ResolutionScale;
use lichii_common::{DAddr, GpuVAddr};
use lichii_gpu::gmmu::{Gmmu, RangeTarget};
use lichii_memory::CachedPages;
use lichii_shader::backend::Bindings;
use lichii_shader::cache;
use lichii_shader::profile::HostTranslateInfo;
use lichii_shader::translate::TranslateOptions;
use lichii_shader::{
    Profile, RuntimeInfo, ShaderEnvironment, ShaderModule, TranslateError, compile,
    compile_dual_vertex,
};

use super::environment::{EnvRead, ProgramEnv};
use super::shader_disk::ShaderDisk;
use super::{FxBuildHasher, FxHasher};

/// Counts since the cache was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderCacheStats {
    /// Programs translated, successfully or not.
    pub translated: u64,
    /// Translations that failed; their draws are skipped.
    pub failed: u64,
    pub hits: u64,
    /// Programs the disk cache served instead of translating them.
    pub disk_hits: u64,
}

#[derive(Debug)]
struct ProgramBytes {
    len: u64,
    hash: u64,
    /// A guest write touched the bytes since they were hashed.
    stale: bool,
    /// The device ranges backing the bytes, which need not be contiguous.
    segments: Vec<(DAddr, u64)>,
    /// Write tracking over `segments`, taken before the bytes were hashed and dropped once a
    /// guest write makes them stale.
    tracked: Vec<CachedPages>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ModuleKey {
    /// Hashes of the VertexA program (0 without one) and of the program.
    code: [u64; 2],
    state: u64,
}

/// The shortest time between two INFO lines of [`ShaderCacheStats`].
const STATS_INTERVAL: Duration = Duration::from_secs(5);

/// Variants kept per key; the oldest goes first.
const MAX_VARIANTS: usize = 32;

struct Variant {
    reads: [Vec<EnvRead>; 2],
    bindings: Bindings,
    module: Option<Arc<ShaderModule>>,
}

/// The renderer's translated modules, owned by its GPU thread.
pub struct ShaderCache {
    profile: Profile,
    host: HostTranslateInfo,
    scale: ResolutionScale,
    /// By the device address of the program's first byte.
    programs: HashMap<DAddr, ProgramBytes, FxBuildHasher>,
    /// Every device range backing a program, keyed by its start and the program's key, to
    /// its length.
    segments: BTreeMap<(DAddr, DAddr), u64>,
    longest: u64,
    modules: HashMap<ModuleKey, Vec<Variant>, FxBuildHasher>,
    disk: Option<ShaderDisk>,
    bytes: Vec<u8>,
    stats: ShaderCacheStats,
    /// The counts the last INFO line showed, and when it was written.
    logged: (ShaderCacheStats, Option<Instant>),
}

impl ShaderCache {
    pub fn new(profile: Profile, host: HostTranslateInfo) -> Self {
        Self {
            profile,
            host,
            scale: ResolutionScale::X1,
            programs: HashMap::default(),
            segments: BTreeMap::new(),
            longest: 0,
            modules: HashMap::default(),
            disk: None,
            bytes: Vec::new(),
            stats: ShaderCacheStats::default(),
            logged: (ShaderCacheStats::default(), None),
        }
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    pub fn stats(&self) -> ShaderCacheStats {
        self.stats
    }

    /// Writes the counts as an INFO line when they changed since the last one and at least
    /// [`STATS_INTERVAL`] has passed; returns whether it did.
    pub fn log_stats(&mut self, now: Instant) -> bool {
        let (logged, at) = self.logged;
        if self.stats == logged || at.is_some_and(|at| now.duration_since(at) < STATS_INTERVAL) {
            return false;
        }
        let ShaderCacheStats {
            translated,
            failed,
            hits,
            disk_hits,
        } = self.stats;
        tracing::info!(translated, failed, hits, disk_hits, "shader cache");
        self.logged = (self.stats, Some(now));
        true
    }

    /// Keeps translated programs in the per-title file at `path` from now on: programs it
    /// holds are reused instead of translated, and new translations are appended at
    /// [`Self::flush_disk`].
    pub fn open_disk(&mut self, path: PathBuf) -> io::Result<()> {
        self.disk = Some(ShaderDisk::open(path)?);
        Ok(())
    }

    /// Appends the programs translated since the last call to the disk cache.
    pub fn flush_disk(&mut self) {
        if let Some(disk) = &mut self.disk
            && let Err(error) = disk.flush()
        {
            tracing::warn!("writing the shader cache: {error}");
        }
    }

    /// Every module the disk cache holds, by module hash.
    pub(crate) fn disk_modules(&self) -> HashMap<u64, Arc<ShaderModule>, FxBuildHasher> {
        self.disk
            .as_ref()
            .map(ShaderDisk::modules)
            .unwrap_or_default()
    }

    /// Programs translate for `scale` from now on (resolution rescaling patches them).
    pub fn set_resolution_scale(&mut self, scale: ResolutionScale) {
        self.scale = scale;
    }

    /// The guest wrote `[daddr, +size)`: programs any of whose device ranges it touches are
    /// hashed again before their next use.
    pub fn invalidate(&mut self, daddr: DAddr, size: u64) {
        let end = daddr.saturating_add(size);
        let from = (daddr.saturating_sub(self.longest), 0);
        for (&(start, key), &len) in self.segments.range(from..(end, 0)) {
            if start + len > daddr
                && let Some(program) = self.programs.get_mut(&key)
            {
                program.stale = true;
                program.tracked.clear();
            }
        }
    }

    /// The module for the program `env` describes (VertexA `a` and VertexB `env` make one
    /// module), translated with the runtime info `runtime` builds, which must depend only on
    /// what `state` hashes. `bindings` carries the next free binding from stage to stage.
    /// `None` when the program does not translate: skip the draw.
    pub fn module(
        &mut self,
        a: Option<&ProgramEnv<'_>>,
        env: &ProgramEnv<'_>,
        state: u64,
        runtime: impl FnOnce() -> RuntimeInfo,
        bindings: &mut Bindings,
    ) -> Option<Arc<ShaderModule>> {
        let mut hasher = FxHasher::default();
        (state, *bindings, self.scale, env.state_words()).hash(&mut hasher);
        let state = hasher.finish();
        let code = [a.map_or(Some(0), |a| self.known(a)), self.known(env)];
        if let [Some(a_code), Some(code)] = code {
            let key = ModuleKey {
                code: [a_code, code],
                state,
            };
            let found = self.modules.get(&key).and_then(|variants| {
                variants
                    .iter()
                    .find(|v| a.is_none_or(|a| a.replay(&v.reads[0])) && env.replay(&v.reads[1]))
            });
            if let Some(variant) = found {
                self.stats.hits += 1;
                *bindings = variant.bindings;
                return variant.module.clone();
            }
            if let Some(variants) = self.modules.get(&key) {
                static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 400 {
                    let v = variants.last().expect("a variant");
                    let miss = |reads: &[EnvRead], e: &ProgramEnv<'_>| {
                        reads
                            .iter()
                            .find(|r| !e.replay(std::slice::from_ref(r)))
                            .copied()
                    };
                    let a_miss = a.and_then(|a| miss(&v.reads[0], a));
                    let miss = miss(&v.reads[1], env);
                    tracing::info!(target: "diag", "DIAG variant miss code={:x?} variants={} reads={} a_miss={a_miss:?} miss={miss:?} now={:?}", key.code, variants.len(), v.reads[1].len(), miss.map(|r| match r { EnvRead::Cbuf { index, offset, .. } => env.cbuf_now(index, offset), _ => 0 }));
                }
            }
        }

        let first = *bindings;
        let mut wildcard = false;
        let options = TranslateOptions {
            profile: &self.profile,
            host: &self.host,
            resolution_scale: self.scale,
        };
        let runtime = runtime();
        let a_env = a.map(|a| a as &dyn ShaderEnvironment);
        let stored = self.disk.as_ref().and_then(|disk| {
            let key = cache::options_key(options, &runtime, bindings);
            let probe = cache::probe_key(a_env, env, key);
            let forget_reads = || {
                if let Some(a) = a {
                    a.take_reads();
                }
                env.take_reads();
            };
            let found = disk.candidates(probe).iter().find(|entry| {
                forget_reads();
                entry.matches(a_env, env, key)
            });
            if found.is_none() {
                forget_reads();
            }
            found.map(|entry| (entry.module().clone(), entry.bindings_after()))
        });
        let module = match stored {
            Some((module, after)) => {
                self.stats.disk_hits += 1;
                *bindings = after;
                Some(module)
            }
            None => {
                let result = match (&mut self.disk, a) {
                    (Some(disk), a) => match a {
                        Some(a) => cache::compile_dual_vertex(a, env, options, &runtime, bindings),
                        None => cache::compile(env, options, &runtime, bindings),
                    }
                    .map(|entry| {
                        disk.append(&entry);
                        entry.module().clone()
                    }),
                    (None, Some(a)) => {
                        compile_dual_vertex(a, env, options, &runtime, bindings).map(Arc::new)
                    }
                    (None, None) => compile(env, options, &runtime, bindings).map(Arc::new),
                };
                self.stats.translated += 1;
                match result {
                    Ok(module) => Some(module),
                    Err(error) => {
                        self.stats.failed += 1;
                        *bindings = first;
                        warn_failed(env, &error);
                        wildcard = !error.depends_on_textures();
                        None
                    }
                }
            }
        };
        let key = ModuleKey {
            code: [a.map_or(0, |a| self.learn(a)), self.learn(env)],
            state,
        };
        let mut reads = [
            a.map(ProgramEnv::take_reads).unwrap_or_default(),
            env.take_reads(),
        ];
        if wildcard {
            reads = Default::default();
        }
        let variants = self.modules.entry(key).or_default();
        if variants.len() == MAX_VARIANTS {
            variants.remove(0);
        }
        variants.push(Variant {
            reads,
            bindings: *bindings,
            module: module.clone(),
        });
        self.log_stats(Instant::now());
        module
    }

    /// The hash of the program at `env`'s address, when its extent is known.
    fn known(&mut self, env: &ProgramEnv<'_>) -> Option<u64> {
        let daddr = env.gmmu().translate(env.code_start())?;
        let program = self.programs.get(&daddr)?;
        if program.stale {
            let len = program.len;
            return Some(self.track(env.gmmu(), env.code_start(), len, daddr));
        }
        Some(program.hash)
    }

    /// Hashes the bytes the last translation on `env` read and remembers them by address.
    fn learn(&mut self, env: &ProgramEnv<'_>) -> u64 {
        let (start, len) = env.code_range();
        match env.gmmu().translate(start) {
            Some(daddr) => self.track(env.gmmu(), start, len, daddr),
            None => self.hash_bytes(env.gmmu(), start, len),
        }
    }

    /// Hashes the `len` program bytes at `start` (device address `key`) and registers every
    /// device range now backing them.
    fn track(&mut self, gmmu: &Gmmu, start: GpuVAddr, len: u64, key: DAddr) -> u64 {
        let mut segments = Vec::new();
        gmmu.for_each_range(start, len, |range| {
            if let RangeTarget::Device(daddr) = range.target {
                segments.push((daddr, range.size));
            }
        });
        let tracked = segments
            .iter()
            .map(|&(daddr, size)| gmmu.device().cache_pages(daddr, size))
            .collect();
        let hash = self.hash_bytes(gmmu, start, len);
        for &(daddr, size) in &segments {
            self.longest = self.longest.max(size);
            self.segments.insert((daddr, key), size);
        }
        let program = ProgramBytes {
            len,
            hash,
            stale: false,
            segments,
            tracked,
        };
        if let Some(old) = self.programs.insert(key, program) {
            let current = &self.programs[&key].segments;
            for &(daddr, _) in &old.segments {
                if !current.iter().any(|&(d, _)| d == daddr) {
                    self.segments.remove(&(daddr, key));
                }
            }
        }
        hash
    }

    fn hash_bytes(&mut self, gmmu: &Gmmu, start: GpuVAddr, len: u64) -> u64 {
        self.bytes.resize(len as usize, 0);
        let _ = gmmu.read_block(start, &mut self.bytes);
        hash_code(&self.bytes)
    }
}

fn hash_code(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(bytes);
    hasher.write_u64(bytes.len() as u64);
    hasher.finish()
}

fn warn_failed(env: &ProgramEnv<'_>, error: &TranslateError) {
    let (stage, address) = (env.stage(), env.start_address());
    match error {
        TranslateError::Emit(_) | TranslateError::UnsupportedProgram { .. } => {
            tracing::warn!(?stage, address, "skipping draws of a program: {error}");
        }
        _ => tracing::error!(?stage, address, "translating a program failed: {error}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_are_logged_when_they_change_at_most_every_five_seconds() {
        let mut cache = ShaderCache::new(Profile::default(), HostTranslateInfo::default());
        let start = Instant::now();
        assert!(!cache.log_stats(start), "nothing to log yet");
        cache.stats.translated = 1;
        assert!(cache.log_stats(start));
        cache.stats.hits = 1;
        assert!(!cache.log_stats(start + Duration::from_secs(4)));
        assert!(cache.log_stats(start + STATS_INTERVAL));
        assert!(
            !cache.log_stats(start + Duration::from_secs(60)),
            "nothing changed since"
        );
    }
}
