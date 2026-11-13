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

use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use lichii_common::DAddr;
use lichii_common::settings::ResolutionScale;
use lichii_shader::backend::Bindings;
use lichii_shader::profile::HostTranslateInfo;
use lichii_shader::translate::TranslateOptions;
use lichii_shader::{
    Profile, RuntimeInfo, ShaderEnvironment, ShaderModule, TranslateError, compile,
    compile_dual_vertex,
};

use super::environment::{ProgramEnv, TextureRead};
use super::{FxBuildHasher, FxHasher};

/// Counts since the cache was created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderCacheStats {
    /// Programs translated, successfully or not.
    pub translated: u64,
    /// Translations that failed; their draws are skipped.
    pub failed: u64,
    pub hits: u64,
}

#[derive(Clone, Copy, Debug)]
struct ProgramBytes {
    len: u64,
    hash: u64,
    /// A guest write touched the bytes since they were hashed.
    stale: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ModuleKey {
    /// Hashes of the VertexA program (0 without one) and of the program.
    code: [u64; 2],
    state: u64,
}

struct Variant {
    reads: [Vec<TextureRead>; 2],
    bindings: Bindings,
    module: Option<Arc<ShaderModule>>,
}

/// The renderer's translated modules, owned by its GPU thread.
pub struct ShaderCache {
    profile: Profile,
    host: HostTranslateInfo,
    scale: ResolutionScale,
    programs: BTreeMap<DAddr, ProgramBytes>,
    longest: u64,
    modules: HashMap<ModuleKey, Vec<Variant>, FxBuildHasher>,
    bytes: Vec<u8>,
    stats: ShaderCacheStats,
}

impl ShaderCache {
    pub fn new(profile: Profile, host: HostTranslateInfo) -> Self {
        Self {
            profile,
            host,
            scale: ResolutionScale::X1,
            programs: BTreeMap::new(),
            longest: 0,
            modules: HashMap::default(),
            bytes: Vec::new(),
            stats: ShaderCacheStats::default(),
        }
    }

    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    pub fn stats(&self) -> ShaderCacheStats {
        self.stats
    }

    /// Programs translate for `scale` from now on (resolution rescaling patches them).
    pub fn set_resolution_scale(&mut self, scale: ResolutionScale) {
        self.scale = scale;
    }

    /// The guest wrote `[daddr, +size)`: programs there are hashed again before their next use.
    pub fn invalidate(&mut self, daddr: DAddr, size: u64) {
        let end = daddr.saturating_add(size);
        for (&start, program) in self
            .programs
            .range_mut(daddr.saturating_sub(self.longest)..end)
        {
            if start + program.len > daddr {
                program.stale = true;
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
        }

        let first = *bindings;
        let options = TranslateOptions {
            profile: &self.profile,
            host: &self.host,
            resolution_scale: self.scale,
        };
        let runtime = runtime();
        let result = match a {
            Some(a) => compile_dual_vertex(a, env, options, &runtime, bindings),
            None => compile(env, options, &runtime, bindings),
        };
        self.stats.translated += 1;
        let module = match result {
            Ok(module) => Some(Arc::new(module)),
            Err(error) => {
                self.stats.failed += 1;
                *bindings = first;
                warn_failed(env, &error);
                None
            }
        };
        let key = ModuleKey {
            code: [a.map_or(0, |a| self.learn(a)), self.learn(env)],
            state,
        };
        self.modules.entry(key).or_default().push(Variant {
            reads: [
                a.map(ProgramEnv::take_reads).unwrap_or_default(),
                env.take_reads(),
            ],
            bindings: *bindings,
            module: module.clone(),
        });
        module
    }

    /// The hash of the program at `env`'s address, when its extent is known.
    fn known(&mut self, env: &ProgramEnv<'_>) -> Option<u64> {
        let daddr = env.gmmu().translate(env.code_start())?;
        let program = self.programs.get_mut(&daddr)?;
        if program.stale {
            self.bytes.resize(program.len as usize, 0);
            env.gmmu()
                .read_block(env.code_start(), &mut self.bytes)
                .ok()?;
            program.hash = hash_code(&self.bytes);
            program.stale = false;
        }
        Some(program.hash)
    }

    /// Hashes the bytes the last translation on `env` read and remembers them by address.
    fn learn(&mut self, env: &ProgramEnv<'_>) -> u64 {
        let (start, len) = env.code_range();
        self.bytes.resize(len as usize, 0);
        let _ = env.gmmu().read_block(start, &mut self.bytes);
        let hash = hash_code(&self.bytes);
        if let Some(daddr) = env.gmmu().translate(start) {
            self.longest = self.longest.max(len);
            self.programs.insert(
                daddr,
                ProgramBytes {
                    len,
                    hash,
                    stale: false,
                },
            );
        }
        hash
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
