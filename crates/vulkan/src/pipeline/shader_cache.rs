//! Translated shaders by guest program.
//!
//! A module is keyed by the bytes of its program (the header and every instruction the
//! translator read), the register and pipeline state it was translated for, and the descriptor
//! bindings it starts at; the queries the translator made of other guest state are asked again
//! on every hit, so a texture changing type under an unchanged program translates it again.
//! Never by the module hash, which costs a full translation to compute.
//!
//! Hashing a program's bytes on every draw would cost more than the draw, so the hash is
//! remembered per device address and recomputed only after a guest write touched it. A key's
//! variants are grouped by the questions they asked (nearly always one group) and found by
//! their answers in a map, so a hit asks each question once however many variants the key
//! holds: a page table lookup, the questions and three map lookups. It allocates nothing and
//! takes no lock. Failed translations are cached too: their draws are skipped without translating again.
//! A failure no texture header can have caused ([`TranslateError::depends_on_textures`]) is
//! remembered with only its storage pointer questions, so a constant buffer word that changes
//! from draw to draw (a per-material bindless handle) does not translate the failing program
//! again on every draw; one that changes a jump table does not either. A Switch 2 pointer that
//! was null when the program failed translates it again once it is mapped. A failure a texture header may
//! have caused keeps its reads and translates again under other headers. Each key keeps at most
//! [`MAX_VARIANTS`] variants.
//!
//! With the title's disk cache open, a miss first replays the stored entries that share the
//! program's probe key (`lichii_shader::cache`) and reuses a module whose every recorded answer
//! still holds; new translations serve later misses of the same run and are appended to the
//! file at the next flush.
//!
//! With asynchronous translation on, a graphics miss the disk cannot serve queues the program
//! for the workers (`shader_workers`) and skips the draw; the back thread never translates a
//! graphics program itself. Compute programs translate in place, since a skipped dispatch would
//! change what the guest computes.

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
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
    Profile, RuntimeInfo, ShaderEnvironment, ShaderModule, ShaderStage, TranslateError, compile,
    compile_dual_vertex,
};

use super::environment::{EnvRead, ProgramEnv, TextureAnswers};
use super::shader_disk::ShaderDisk;
use super::shader_workers::{Job, Workers};
use super::{FxBuildHasher, FxHasher};
use crate::debug::shader_dump;

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

/// What [`ShaderCache::module`] finds a module under, besides the answers of its variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ModuleKey {
    /// Hashes of the VertexA program (0 without one) and of the program.
    code: [u64; 2],
    state: u64,
}

/// The shortest time between two INFO lines of [`ShaderCacheStats`].
const STATS_INTERVAL: Duration = Duration::from_secs(5);

/// Variants kept per key; the oldest goes first.
const MAX_VARIANTS: usize = 32;

struct Variant {
    bindings: Bindings,
    module: Option<Arc<ShaderModule>>,
    /// Insertion order within the key, for eviction.
    age: u64,
}

/// Variants of one key whose translations asked the same questions, by their answers.
struct Shape {
    /// The VertexA program's questions and the program's, with the first variant's answers.
    questions: [Vec<EnvRead>; 2],
    by_answers: HashMap<Box<[u32]>, Variant, FxBuildHasher>,
}

#[derive(Default)]
struct Variants {
    shapes: Vec<Shape>,
    len: usize,
    next_age: u64,
}

impl Variants {
    /// The variant whose recorded answers `answer` gives now, asking each question once.
    /// `answer` takes 0 for a VertexA question and 1 for the program's. `answers` is scratch.
    fn find(
        &self,
        answers: &mut Vec<u32>,
        mut answer: impl FnMut(usize, &EnvRead) -> u32,
    ) -> Option<&Variant> {
        self.shapes.iter().find_map(|shape| {
            answers.clear();
            for (side, questions) in shape.questions.iter().enumerate() {
                answers.extend(questions.iter().map(|read| answer(side, read)));
            }
            shape.by_answers.get(answers.as_slice())
        })
    }

    /// Adds a variant translated after `reads`, dropping the oldest when the key is full.
    fn insert(
        &mut self,
        reads: [Vec<EnvRead>; 2],
        bindings: Bindings,
        module: Option<Arc<ShaderModule>>,
    ) {
        let answers = reads
            .iter()
            .flatten()
            .map(EnvRead::answer)
            .collect::<Box<[u32]>>();
        let variant = Variant {
            bindings,
            module,
            age: self.next_age,
        };
        self.next_age += 1;
        let same = |shape: &&mut Shape| {
            shape
                .questions
                .iter()
                .zip(&reads)
                .all(|(questions, reads)| {
                    questions.len() == reads.len()
                        && questions.iter().zip(reads).all(|(q, r)| q.same_question(r))
                })
        };
        let replaced = match self.shapes.iter_mut().find(same) {
            Some(shape) => shape.by_answers.insert(answers, variant).is_some(),
            None => {
                let mut by_answers = HashMap::default();
                by_answers.insert(answers, variant);
                self.shapes.push(Shape {
                    questions: reads,
                    by_answers,
                });
                false
            }
        };
        if replaced {
            return;
        }
        self.len += 1;
        if self.len > MAX_VARIANTS {
            self.evict_oldest();
        }
    }

    fn evict_oldest(&mut self) {
        let oldest = self
            .shapes
            .iter()
            .enumerate()
            .flat_map(|(i, shape)| shape.by_answers.iter().map(move |(k, v)| (v.age, i, k)))
            .min_by_key(|&(age, ..)| age)
            .map(|(_, i, k)| (i, k.clone()));
        let Some((i, answers)) = oldest else { return };
        self.shapes[i].by_answers.remove(&answers);
        if self.shapes[i].by_answers.is_empty() {
            self.shapes.remove(i);
        }
        self.len -= 1;
    }
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
    /// Bumped by every guest write to a known program's bytes and every resolution scale change.
    generation: u64,
    modules: HashMap<ModuleKey, Variants, FxBuildHasher>,
    disk: Option<ShaderDisk>,
    bytes: Vec<u8>,
    answers: Vec<u32>,
    /// What the VertexA program's and the program's questions found of their textures.
    textures: [TextureAnswers; 2],
    workers: Option<Workers>,
    /// Translations queued or running, by request key, with the last frame a draw asked.
    pending: HashMap<u64, Arc<AtomicU64>, FxBuildHasher>,
    code_check: cache::CodeCheck,
    stats: ShaderCacheStats,
    /// Lookups that returned no module because its translation was still queued or running.
    waits: u64,
    /// Graphics lookups that translated in place because they could not be skipped.
    blocked: u64,
    /// Render-thread time spent translating programs in place.
    stalled: Duration,
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
            generation: 0,
            modules: HashMap::default(),
            disk: None,
            bytes: Vec::new(),
            answers: Vec::new(),
            textures: Default::default(),
            workers: None,
            pending: HashMap::default(),
            code_check: cache::CodeCheck::default(),
            stats: ShaderCacheStats::default(),
            waits: 0,
            blocked: 0,
            stalled: Duration::ZERO,
            logged: (ShaderCacheStats::default(), None),
        }
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    pub fn stats(&self) -> ShaderCacheStats {
        self.stats
    }

    /// Lookups since the cache was created that found the module still translating.
    pub fn waits(&self) -> u64 {
        self.waits
    }

    /// Lookups since the cache was created that translated in place a graphics program they
    /// were not allowed to skip, while workers translate the others.
    pub fn blocked(&self) -> u64 {
        self.blocked
    }

    /// Render-thread time since the cache was created spent translating programs in place.
    pub fn stalled(&self) -> Duration {
        self.stalled
    }

    /// Translations queued or running.
    pub fn translating(&self) -> usize {
        self.pending.len()
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
        self.generation += 1;
    }

    /// Changes whenever a guest write touches the bytes of a program the cache knows, or the
    /// resolution scale changes: a [`ModuleKey`] [`Self::module`] returned is still the key
    /// it would compute for the same registers while this stays the same.
    pub fn generation(&self) -> u64 {
        self.generation
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
                self.generation += 1;
            }
        }
    }

    /// The module for the program `env` describes (VertexA `a` and VertexB `env` make one
    /// module), translated with the runtime info `runtime` builds, which must depend only on
    /// what `state` hashes. `bindings` carries the next free binding from stage to stage.
    /// With workers, a graphics program not translated yet is queued for them and `None`
    /// returned when `may_skip`, else translated in place. `None` also when the program does
    /// not translate: skip the draw. The key returned finds the module again through
    /// [`Self::find`].
    pub fn module(
        &mut self,
        a: Option<&ProgramEnv<'_>>,
        env: &ProgramEnv<'_>,
        state: u64,
        runtime: impl FnOnce() -> RuntimeInfo,
        bindings: &mut Bindings,
        may_skip: bool,
    ) -> Option<(ModuleKey, Arc<ShaderModule>)> {
        self.drain();
        let mut hasher = FxHasher::default();
        (state, *bindings, self.scale, env.state_words()).hash(&mut hasher);
        let state = hasher.finish();
        let code = [a.map_or(Some(0), |a| self.known(a)), self.known(env)];
        if let [Some(a_code), Some(code)] = code {
            let key = ModuleKey {
                code: [a_code, code],
                state,
            };
            if let Some(variant) = self.variant(&key, a, env) {
                *bindings = variant.bindings;
                return variant.module.clone().map(|module| (key, module));
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
            let code = &mut self.code_check;
            code.clear();
            let found = disk.candidates(probe).iter().find(|entry| {
                forget_reads();
                entry.matches(a_env, env, key, code)
            });
            if found.is_none() {
                forget_reads();
            }
            found.map(|entry| {
                (
                    entry.module().clone(),
                    entry.bindings_after(),
                    entry.code_ranges(),
                )
            })
        });
        let module = match stored {
            Some((module, after, ranges)) => {
                self.stats.disk_hits += 1;
                *bindings = after;
                dump(&module, a, env, ranges);
                Some(module)
            }
            None if may_skip && self.workers.is_some() && env.stage() != ShaderStage::Compute => {
                self.waits += 1;
                self.request(a, env, state, runtime, first);
                return None;
            }
            None => {
                if self.workers.is_some() && env.stage() != ShaderStage::Compute {
                    self.blocked += 1;
                }
                let start = Instant::now();
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
                self.stalled += start.elapsed();
                match result {
                    Ok(module) => {
                        dump(&module, a, env, translated_ranges(a, env));
                        Some(module)
                    }
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
            keep_failure_questions(&mut reads);
        }
        self.remember(key, reads, *bindings, module.clone());
        module.map(|module| (key, module))
    }

    /// The module [`Self::module`] returned under `key` if the environments still answer
    /// its translation's queries as they did, else `None`.
    pub(crate) fn find(
        &mut self,
        key: &ModuleKey,
        a: Option<&ProgramEnv<'_>>,
        env: &ProgramEnv<'_>,
    ) -> Option<Arc<ShaderModule>> {
        self.variant(key, a, env)?.module.clone()
    }

    fn variant(
        &mut self,
        key: &ModuleKey,
        a: Option<&ProgramEnv<'_>>,
        env: &ProgramEnv<'_>,
    ) -> Option<&Variant> {
        let textures = &mut self.textures;
        textures.iter_mut().for_each(TextureAnswers::clear);
        let variant = self
            .modules
            .get(key)?
            .find(&mut self.answers, |side, read| match (side, a) {
                (0, Some(a)) => a.answer(read, &mut textures[0]),
                _ => env.answer(read, &mut textures[1]),
            })?;
        self.stats.hits += 1;
        Some(variant)
    }

    fn remember(
        &mut self,
        key: ModuleKey,
        reads: [Vec<EnvRead>; 2],
        bindings: Bindings,
        module: Option<Arc<ShaderModule>>,
    ) {
        self.answers.clear();
        self.answers.reserve(reads[0].len() + reads[1].len());
        self.modules
            .entry(key)
            .or_default()
            .insert(reads, bindings, module);
        self.log_stats(Instant::now());
    }

    /// Queues the translation a draw missed, once while it is queued or running.
    fn request(
        &mut self,
        a: Option<&ProgramEnv<'_>>,
        env: &ProgramEnv<'_>,
        state: u64,
        runtime: RuntimeInfo,
        bindings: Bindings,
    ) {
        let Some(workers) = &self.workers else {
            return;
        };
        let mut hasher = FxHasher::default();
        (state, a.map(ProgramEnv::code_start), env.code_start()).hash(&mut hasher);
        let key = hasher.finish();
        let frame = workers.frame();
        if let Some(seen) = self.pending.get(&key) {
            seen.store(frame, Ordering::Relaxed);
            return;
        }
        let seen = Arc::new(AtomicU64::new(frame));
        self.pending.insert(key, seen.clone());
        let job = Job {
            key,
            state,
            vertex_a: a.map(ProgramEnv::snapshot),
            program: env.snapshot(),
            runtime,
            bindings,
            scale: self.scale,
            disk: self.disk.is_some(),
        };
        workers.push(seen, job);
    }

    /// Takes in what the workers translated. A result whose program bytes changed since is
    /// dropped; the next draw of the program asks again.
    fn drain(&mut self) {
        while let Some(done) = self.workers.as_ref().and_then(Workers::try_recv) {
            self.pending.remove(&done.key);
            let Some(translated) = done.translated else {
                continue;
            };
            let mut code = [0; 2];
            for (hash, program) in code.iter_mut().zip(&translated.code) {
                let Some(program) = program else {
                    continue;
                };
                let (gmmu, start, len) = (&program.gmmu, program.start, program.len);
                *hash = match gmmu.translate(start) {
                    Some(daddr) => self.track(gmmu, start, len, daddr),
                    None => self.hash_bytes(gmmu, start, len),
                };
            }
            let changed = code
                .iter()
                .zip(&translated.code)
                .any(|(&hash, program)| program.as_ref().is_some_and(|p| p.hash != hash));
            if changed {
                continue;
            }
            self.stats.translated += 1;
            if translated.module.is_err() {
                self.stats.failed += 1;
            }
            if let (Some(disk), Some(entry)) = (&mut self.disk, &translated.entry) {
                disk.append(entry);
            }
            let key = ModuleKey {
                code,
                state: done.state,
            };
            self.remember(
                key,
                translated.reads,
                translated.bindings,
                translated.module.ok(),
            );
        }
    }

    /// Translates graphics programs on worker threads, so a draw whose program is not
    /// translated yet is skipped instead of waiting; compute programs translate in place.
    pub fn set_async(&mut self, on: bool) {
        if on == self.workers.is_some() {
            return;
        }
        self.workers = on.then(|| Workers::new(&self.profile, &self.host));
        self.pending.clear();
    }

    /// Ages the queued translations: one no draw asked for in the last few presents is dropped.
    pub fn end_frame(&self) {
        if let Some(workers) = &self.workers {
            workers.end_frame();
        }
    }

    /// Waits until every queued translation is taken in.
    #[cfg(test)]
    pub(crate) fn finish_translations(&mut self) {
        let start = Instant::now();
        while !self.pending.is_empty() {
            assert!(
                start.elapsed() < Duration::from_secs(60),
                "translations never finished"
            );
            std::thread::sleep(Duration::from_millis(1));
            self.drain();
        }
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
        hash_program(gmmu, start, len, &mut self.bytes)
    }
}

/// The hash of the `len` program bytes at `start`, read into `bytes`.
pub(super) fn hash_program(gmmu: &Gmmu, start: GpuVAddr, len: u64, bytes: &mut Vec<u8>) -> u64 {
    bytes.resize(len as usize, 0);
    let _ = gmmu.read_block(start, bytes);
    hash_code(bytes)
}

/// With `dump_shaders` on, dumps `module` with the guest code it came from: `ranges` holds the
/// bytes of VertexA `a` and of `env` as offsets from each program's first instruction and
/// lengths, as [`lichii_shader::cache::CacheEntry::code_ranges`] gives them.
fn dump(
    module: &ShaderModule,
    a: Option<&ProgramEnv<'_>>,
    env: &ProgramEnv<'_>,
    ranges: [(i64, u64); 2],
) {
    if !shader_dump::enabled() {
        return;
    }
    let read = |env: &ProgramEnv<'_>, (offset, len): (i64, u64)| {
        let mut bytes = vec![0; len as usize];
        let _ = env
            .gmmu()
            .read_block(env.start_address().wrapping_add_signed(offset), &mut bytes);
        bytes
    };
    let vertex_a = a.map(|a| read(a, ranges[0]));
    shader_dump::write(
        module.hash,
        &module.spirv,
        &read(env, ranges[1]),
        vertex_a.as_deref(),
    );
}

/// The ranges [`dump`] takes for programs just translated in place: what translation read.
fn translated_ranges(a: Option<&ProgramEnv<'_>>, env: &ProgramEnv<'_>) -> [(i64, u64); 2] {
    let range = |env: &ProgramEnv<'_>| {
        let (start, len) = env.code_range();
        (start.wrapping_sub(env.start_address()) as i64, len)
    };
    [a.map_or((0, 0), range), range(env)]
}

fn hash_code(bytes: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(bytes);
    hasher.write_u64(bytes.len() as u64);
    hasher.finish()
}

/// Drops the questions a failure no texture header caused is not remembered by: all but whether
/// its storage pointers were mapped.
pub(super) fn keep_failure_questions(reads: &mut [Vec<EnvRead>; 2]) {
    reads
        .iter_mut()
        .for_each(|reads| reads.retain(EnvRead::is_storage_pointer));
}

pub(super) fn warn_failed(env: &ProgramEnv<'_>, error: &TranslateError) {
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

    fn word(offset: u32, value: u32) -> EnvRead {
        EnvRead::Cbuf {
            index: 3,
            offset,
            value,
        }
    }

    fn reads(values: [u32; 2]) -> [Vec<EnvRead>; 2] {
        [Vec::new(), vec![word(0, values[0]), word(4, values[1])]]
    }

    fn insert(variants: &mut Variants, values: [u32; 2]) {
        let bindings = Bindings {
            uniform_buffer: values[0],
            ..Bindings::default()
        };
        variants.insert(reads(values), bindings, None);
    }

    fn find(variants: &Variants, words: [u32; 2]) -> (Option<u32>, usize) {
        let mut asked = 0;
        let found = variants.find(&mut Vec::new(), |side, read| {
            assert_eq!(side, 1);
            asked += 1;
            let EnvRead::Cbuf { offset, .. } = *read else {
                unreachable!()
            };
            words[offset as usize / 4]
        });
        (found.map(|v| v.bindings.uniform_buffer), asked)
    }

    #[test]
    fn a_hit_asks_each_question_once_however_many_variants_the_key_holds() {
        let mut variants = Variants::default();
        for value in 0..MAX_VARIANTS as u32 {
            insert(&mut variants, [value, 7]);
        }
        assert_eq!(variants.shapes.len(), 1);
        assert_eq!(find(&variants, [31, 7]), (Some(31), 2));
        assert_eq!(find(&variants, [0, 7]), (Some(0), 2));
        assert_eq!(find(&variants, [5, 8]), (None, 2));
    }

    #[test]
    fn a_failure_keeps_only_its_pointer_questions() {
        let null = EnvRead::StoragePointer {
            index: 0,
            offset: 0x40,
            mapped: false,
        };
        let mut failed = reads([1, 2]);
        failed[1].push(null);
        keep_failure_questions(&mut failed);
        assert_eq!(failed, [Vec::new(), vec![null]]);

        let mut variants = Variants::default();
        variants.insert(failed, Bindings::default(), None);
        let lookup = |mapped: u32| {
            variants
                .find(&mut Vec::new(), |_, read| {
                    assert!(read.is_storage_pointer());
                    mapped
                })
                .is_some()
        };
        assert!(lookup(0), "the failure is reused while the pointer is null");
        assert!(!lookup(1), "a mapped pointer translates again");
    }

    #[test]
    fn a_full_key_drops_its_oldest_variant() {
        let mut variants = Variants::default();
        for value in 0..MAX_VARIANTS as u32 {
            insert(&mut variants, [value, 7]);
        }
        insert(&mut variants, [0, 7]);
        assert_eq!(
            variants.len, MAX_VARIANTS,
            "the same answers replace a variant"
        );
        insert(&mut variants, [100, 7]);
        assert_eq!(variants.len, MAX_VARIANTS);
        assert_eq!(find(&variants, [1, 7]).0, None, "the oldest went");
        assert_eq!(find(&variants, [0, 7]).0, Some(0), "replaced, so younger");
        assert_eq!(find(&variants, [100, 7]).0, Some(100));
    }

    #[test]
    fn variants_that_asked_other_questions_are_found_too() {
        let mut variants = Variants::default();
        insert(&mut variants, [1, 2]);
        variants.insert(Default::default(), Bindings::default(), None);
        assert_eq!(variants.shapes.len(), 2);
        assert_eq!(find(&variants, [1, 2]), (Some(1), 2));
        assert_eq!(
            find(&variants, [9, 9]),
            (Some(0), 2),
            "a variant without questions matches anything"
        );
    }

    /// The back thread's cost of re-checking the variants of three stages whose translations
    /// asked 8 to 12 questions each, nearly all of texture headers:
    /// `cargo test --release -p lichii-vulkan --lib variant_check_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn variant_check_cost_of_three_texturing_stages() {
        use std::hint::black_box;

        use lichii_gpu::backend::{ChannelId, ComputeContext, ConstBufferBinding, QmdVersion};
        use lichii_gpu::gmmu::BigPageSize;
        use lichii_memory::{DeviceAddressSpace, PhysicalMemory};
        use lichii_shader::info::HandleSource;
        use lichii_texture::tic::TicEntry;

        use crate::pipeline::environment::TexturePool;

        fn tic(format: u32, component: u32, ty: u32, normalized: bool) -> TicEntry {
            let mut w = [0u32; 8];
            w[0] = format | (component * 0o1111) << 7 | 0o5432 << 19;
            w[2] = 3 << 21;
            w[4] = 63 | ty << 23;
            w[5] = 63 | u32::from(normalized) << 31;
            TicEntry(w)
        }

        const SIZE: u64 = 1 << 20;
        const DADDR: u64 = 0x1_0000_0000;
        const GPU_VA: GpuVAddr = 0x4_0000_0000;
        const TIC: GpuVAddr = GPU_VA + 0x1_0000;
        const CBUF: GpuVAddr = GPU_VA + 0x2_0000;
        const LIMIT: u32 = 63;
        let physical = Arc::new(PhysicalMemory::new(SIZE).unwrap());
        physical.commit(0, SIZE).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(DADDR, 0, SIZE);
        let gmmu = Arc::new(Gmmu::new(memory, BigPageSize::Size64K));
        gmmu.map(GPU_VA, DADDR, SIZE, 0, false).unwrap();
        let headers = [
            tic(0x08, 2, 1, true),
            tic(0x08, 4, 1, true),
            tic(0x24, 2, 1, true),
            tic(0x17, 2, 3, true),
            tic(0x0f, 7, 5, true),
            tic(0x1d, 4, 2, true),
            tic(0x08, 2, 1, false),
            tic(0x00, 2, 1, true),
        ];
        for i in 0..=LIMIT {
            gmmu.write(TIC + u64::from(i) * 32, headers[i as usize % headers.len()])
                .unwrap();
        }
        let textures = [3usize, 5, 4];
        let mut const_buffers = [[ConstBufferBinding::default(); 8]; 3];
        for (stage, &count) in textures.iter().enumerate() {
            let cbuf = CBUF + stage as u64 * 0x1000;
            const_buffers[stage][0] = ConstBufferBinding {
                address: cbuf,
                size: 0x1000,
                enabled: true,
            };
            for t in 0..count as u32 {
                let index = 1 + stage as u32 * 8 + t;
                let handle = if t == 1 { index * 32 } else { index | 3 << 20 };
                gmmu.write(cbuf + u64::from(t) * 8, handle).unwrap();
            }
        }
        let qmd = [0u32; 64];
        let ctx = |stage: usize| ComputeContext {
            channel: ChannelId(0),
            memory: &gmmu,
            program: GPU_VA,
            grid: [1; 3],
            block: [1; 3],
            shared_memory_size: 0,
            local_memory_size: 0,
            const_buffers: &const_buffers[stage],
            texture_headers: TIC,
            texture_header_limit: LIMIT,
            samplers: 0,
            sampler_limit: 0,
            linked_samplers: false,
            qmd_version: QmdVersion::V3,
            qmd: &qmd,
            indirect_grid: None,
            indirect: None,
        };
        let source = |t: u32| HandleSource {
            cbuf_index: 0,
            cbuf_offset: t * 8,
            shift_left: 0,
            secondary: (t == 2).then_some((0, t * 8 + 4, 0)),
        };
        let mut cache = ShaderCache::new(Profile::default(), HostTranslateInfo::default());
        let mut keys = Vec::new();
        let mut questions = 0;
        for (stage, &count) in textures.iter().enumerate() {
            let ctx = ctx(stage);
            let env = ProgramEnv::compute(&ctx);
            for t in 0..count as u32 {
                env.texture_type_of(source(t));
                env.is_texture_pixel_format_integer_of(source(t));
                if t % 3 == 0 {
                    env.texture_pixel_format_of(source(t));
                }
            }
            if stage == 0 {
                env.read_cbuf(0, 0x100);
            }
            if stage == 2 {
                env.texture_type(60);
                env.texture_type(100);
            }
            let reads = env.take_reads();
            assert!((8..=12).contains(&reads.len()), "{}", reads.len());
            questions += reads.len();
            let key = ModuleKey {
                code: [0, stage as u64],
                state: 0,
            };
            cache.remember(key, [Vec::new(), reads], Bindings::default(), None);
            keys.push(key);
        }
        let contexts = [ctx(0), ctx(1), ctx(2)];
        let check = |cache: &mut ShaderCache| {
            contexts.iter().zip(&keys).all(|(ctx, key)| {
                let env = ProgramEnv::compute(black_box(ctx));
                cache.variant(key, None, &env).is_some()
            })
        };
        assert!(check(&mut cache), "every stage finds its variant");
        let time = |what: &str, calls: u32, run: &mut dyn FnMut()| {
            let mut times = Vec::new();
            for _ in 0..201 {
                let start = Instant::now();
                run();
                times.push(start.elapsed() / calls);
            }
            times.sort();
            eprintln!(
                "{what}: median {:?}, min {:?}, p90 {:?}",
                times[times.len() / 2],
                times[0],
                times[times.len() * 9 / 10]
            );
        };
        const CHECKS: u32 = 2000;
        time(
            &format!("variant check of 3 stages, {questions} questions"),
            CHECKS,
            &mut || {
                for _ in 0..CHECKS {
                    assert!(black_box(check(&mut cache)));
                }
            },
        );
        let pool = TexturePool {
            address: TIC,
            limit: LIMIT,
            via_header_index: false,
            switch2: true,
        };
        let handles = 0..LIMIT + 8;
        let rounds = 500;
        time(
            &format!("TexturePool::header over {} handles", handles.len()),
            rounds * handles.len() as u32,
            &mut || {
                for _ in 0..rounds {
                    for handle in handles.clone() {
                        black_box(pool.header(&gmmu, black_box(handle)));
                    }
                }
            },
        );
    }

    #[test]
    fn dump_shaders_writes_a_translated_program_with_its_guest_code_once() {
        use lichii_gpu::backend::{ChannelId, ComputeContext, ConstBufferBinding, QmdVersion};
        use lichii_gpu::gmmu::BigPageSize;
        use lichii_memory::{DeviceAddressSpace, PhysicalMemory};

        const SIZE: u64 = 1 << 16;
        const GPU_VA: GpuVAddr = 0x4_0000_0000;
        const EXIT: [u64; 2] = [0x0000_0000_0000_794d, 0x000f_ea00_0380_0000];
        let physical = Arc::new(PhysicalMemory::new(SIZE).unwrap());
        physical.commit(0, SIZE).unwrap();
        let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
        memory.map(0x1_0000_0000, 0, SIZE);
        let gmmu = Arc::new(Gmmu::new(memory, BigPageSize::Size64K));
        gmmu.map(GPU_VA, 0x1_0000_0000, SIZE, 0, false).unwrap();
        gmmu.write(GPU_VA, EXIT).unwrap();
        let dir = std::env::temp_dir().join(format!("lichii-dump-shaders-{}", std::process::id()));
        shader_dump::init(dir.clone());
        let const_buffers = [ConstBufferBinding::default(); 8];
        let qmd = [0u32; 64];
        let ctx = ComputeContext {
            channel: ChannelId(0),
            memory: &gmmu,
            program: GPU_VA,
            grid: [1; 3],
            block: [1; 3],
            shared_memory_size: 0,
            local_memory_size: 0,
            const_buffers: &const_buffers,
            texture_headers: 0,
            texture_header_limit: 0,
            samplers: 0,
            sampler_limit: 0,
            linked_samplers: false,
            qmd_version: QmdVersion::V3,
            qmd: &qmd,
            indirect_grid: None,
            indirect: None,
        };
        let mut cache = ShaderCache::new(Profile::default(), HostTranslateInfo::default());
        let env = ProgramEnv::compute(&ctx);
        let (_, module) = cache
            .module(
                None,
                &env,
                0,
                RuntimeInfo::default,
                &mut Bindings::default(),
                false,
            )
            .expect("EXIT translates");
        let read = |suffix: &str| std::fs::read(dir.join(format!("{:016x}.{suffix}", module.hash)));
        let code = read("b.bin").unwrap();
        assert_eq!(code[..8], EXIT[0].to_le_bytes());
        assert_eq!(code[8..16], EXIT[1].to_le_bytes());
        let spirv: Vec<u8> = module
            .spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        assert_eq!(read("spv").unwrap(), spirv);
        assert!(read("a.bin").is_err(), "a compute program has no VertexA");
        std::fs::write(dir.join(format!("{:016x}.b.bin", module.hash)), b"kept").unwrap();
        dump(&module, None, &env, translated_ranges(None, &env));
        assert_eq!(read("b.bin").unwrap(), b"kept", "a module is written once");
    }
}
