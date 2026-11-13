//! Shader translation off the GPU back thread. Jobs wait on a stack, so the programs the latest
//! draws asked for go first, and a fixed pool of workers translates them. A job no draw asked
//! for in the last [`STALE_FRAMES`] presents is dropped unstarted. Workers read guest memory
//! through the channel's `Gmmu` as it is when they run; the back thread checks the program
//! bytes again before it uses what they translated.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::JoinHandle;

use lichii_common::GpuVAddr;
use lichii_common::settings::ResolutionScale;
use lichii_gpu::gmmu::Gmmu;
use lichii_shader::backend::Bindings;
use lichii_shader::cache::{self, CacheEntry};
use lichii_shader::profile::HostTranslateInfo;
use lichii_shader::translate::TranslateOptions;
use lichii_shader::{
    Profile, RuntimeInfo, ShaderModule, TranslateError, compile, compile_dual_vertex,
};
use parking_lot::{Condvar, Mutex};

use super::environment::{EnvRead, EnvSnapshot, ProgramEnv};
use super::shader_cache::{hash_program, warn_failed};

/// Presents a queued job may go unrequested before workers drop it.
pub(crate) const STALE_FRAMES: u64 = 2;
/// Host threads the emulator keeps busy: four guest cores and the GPU front and back threads.
const EMULATION_THREADS: usize = 6;
const MAX_WORKERS: usize = 4;

/// One program, or a VertexA and VertexB pair, to translate.
pub(crate) struct Job {
    pub key: u64,
    /// The module key's state hash, handed back in [`Done`].
    pub state: u64,
    pub vertex_a: Option<EnvSnapshot>,
    pub program: EnvSnapshot,
    pub runtime: RuntimeInfo,
    pub bindings: Bindings,
    pub scale: ResolutionScale,
    /// Whether to build the disk cache entry too.
    pub disk: bool,
}

/// A finished job: `translated` is `None` when it went stale before a worker took it, or the
/// guest rewrote the program while it translated.
pub(crate) struct Done {
    pub key: u64,
    pub state: u64,
    pub translated: Option<Translated>,
}

pub(crate) struct Translated {
    pub module: Result<Arc<ShaderModule>, TranslateError>,
    pub entry: Option<CacheEntry>,
    /// What translation asked, `[VertexA, program]`; none for a failure no texture caused.
    pub reads: [Vec<EnvRead>; 2],
    /// The program bytes translation read, `[VertexA, program]`.
    pub code: [Option<ProgramCode>; 2],
    /// The bindings after the module, or the incoming ones when it failed.
    pub bindings: Bindings,
}

/// A range of program memory and the hash of the bytes a worker translated from it.
pub(crate) struct ProgramCode {
    pub gmmu: Arc<Gmmu>,
    pub start: GpuVAddr,
    pub len: u64,
    pub hash: u64,
}

/// Jobs by recency, each with the last frame a draw asked for it.
struct Queue<T> {
    jobs: Vec<(Arc<AtomicU64>, T)>,
}

impl<T> Queue<T> {
    /// The newest job, and whether no draw asked for it within [`STALE_FRAMES`] of `frame`.
    fn pop(&mut self, frame: u64) -> Option<(T, bool)> {
        let (seen, job) = self.jobs.pop()?;
        let stale = frame.saturating_sub(seen.load(Ordering::Relaxed)) > STALE_FRAMES;
        Some((job, stale))
    }
}

struct Shared {
    queue: Mutex<Queue<Job>>,
    ready: Condvar,
    stop: AtomicBool,
    frame: AtomicU64,
}

/// The translation workers, owned by the back thread's shader cache.
pub(crate) struct Workers {
    shared: Arc<Shared>,
    done: mpsc::Receiver<Done>,
    threads: Vec<JoinHandle<()>>,
}

impl Workers {
    /// One worker per host thread the emulator leaves free, at least one and at most
    /// [`MAX_WORKERS`].
    pub(crate) fn new(profile: &Profile, host: &HostTranslateInfo) -> Self {
        let count = std::thread::available_parallelism()
            .map_or(1, |n| n.get().saturating_sub(EMULATION_THREADS))
            .clamp(1, MAX_WORKERS);
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue { jobs: Vec::new() }),
            ready: Condvar::new(),
            stop: AtomicBool::new(false),
            frame: AtomicU64::new(0),
        });
        let (sender, done) = mpsc::channel();
        let threads = (0..count)
            .map(|i| {
                let (shared, sender) = (shared.clone(), sender.clone());
                let (profile, host) = (profile.clone(), host.clone());
                std::thread::Builder::new()
                    .name(format!("shader worker {i}"))
                    .spawn(move || work(&shared, &sender, &profile, &host))
                    .expect("spawning a shader worker")
            })
            .collect();
        Self {
            shared,
            done,
            threads,
        }
    }

    /// The present count jobs are aged by.
    pub(crate) fn frame(&self) -> u64 {
        self.shared.frame.load(Ordering::Relaxed)
    }

    pub(crate) fn end_frame(&self) {
        self.shared.frame.fetch_add(1, Ordering::Relaxed);
    }

    /// Queues `job`; a draw that asks for it again stores the frame in `seen`.
    pub(crate) fn push(&self, seen: Arc<AtomicU64>, job: Job) {
        self.shared.queue.lock().jobs.push((seen, job));
        self.shared.ready.notify_one();
    }

    pub(crate) fn try_recv(&self) -> Option<Done> {
        self.done.try_recv().ok()
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Relaxed);
        {
            let _queue = self.shared.queue.lock();
            self.shared.ready.notify_all();
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

fn work(shared: &Shared, done: &mpsc::Sender<Done>, profile: &Profile, host: &HostTranslateInfo) {
    loop {
        let (job, stale) = {
            let mut queue = shared.queue.lock();
            loop {
                if shared.stop.load(Ordering::Relaxed) {
                    return;
                }
                if let Some(popped) = queue.pop(shared.frame.load(Ordering::Relaxed)) {
                    break popped;
                }
                shared.ready.wait(&mut queue);
            }
        };
        let translated = if stale {
            None
        } else {
            translate(&job, profile, host)
        };
        let finished = Done {
            key: job.key,
            state: job.state,
            translated,
        };
        if done.send(finished).is_err() {
            return;
        }
    }
}

/// `None` when the guest rewrote program words translation had read: the module would not
/// match the bytes now there.
fn translate(job: &Job, profile: &Profile, host: &HostTranslateInfo) -> Option<Translated> {
    let options = TranslateOptions {
        profile,
        host,
        resolution_scale: job.scale,
    };
    let a = job.vertex_a.as_ref().map(ProgramEnv::from_snapshot);
    let env = ProgramEnv::from_snapshot(&job.program);
    let mut bindings = job.bindings;
    let runtime = &job.runtime;
    let result = match (&a, job.disk) {
        (Some(a), true) => cache::compile_dual_vertex(a, &env, options, runtime, &mut bindings)
            .map(|entry| (entry.module().clone(), Some(entry))),
        (None, true) => cache::compile(&env, options, runtime, &mut bindings)
            .map(|entry| (entry.module().clone(), Some(entry))),
        (Some(a), false) => compile_dual_vertex(a, &env, options, runtime, &mut bindings)
            .map(|module| (Arc::new(module), None)),
        (None, false) => {
            compile(&env, options, runtime, &mut bindings).map(|module| (Arc::new(module), None))
        }
    };
    let mut bytes = Vec::new();
    let mut code = |env: &ProgramEnv<'_>| {
        let (start, len) = env.code_range();
        let hash = hash_program(env.gmmu(), start, len, &mut bytes);
        env.consumed_unchanged(start, &bytes).then(|| ProgramCode {
            gmmu: env.memory().clone(),
            start,
            len,
            hash,
        })
    };
    let vertex_a = match &a {
        Some(a) => Some(code(a)?),
        None => None,
    };
    let code = [vertex_a, Some(code(&env)?)];
    let mut reads = [
        a.as_ref().map(ProgramEnv::take_reads).unwrap_or_default(),
        env.take_reads(),
    ];
    let (module, entry) = match result {
        Ok((module, entry)) => (Ok(module), entry),
        Err(error) => {
            warn_failed(&env, &error);
            bindings = job.bindings;
            if !error.depends_on_textures() {
                reads = Default::default();
            }
            (Err(error), None)
        }
    };
    Some(Translated {
        module,
        entry,
        reads,
        code,
        bindings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn queue(jobs: &[(u64, u32)]) -> Queue<u32> {
        let jobs = jobs
            .iter()
            .map(|&(seen, job)| (Arc::new(AtomicU64::new(seen)), job))
            .collect();
        Queue { jobs }
    }

    #[test]
    fn the_newest_job_goes_first() {
        let mut queue = queue(&[(5, 1), (5, 2), (5, 3)]);
        let order: Vec<_> = std::iter::from_fn(|| queue.pop(5)).collect();
        assert_eq!(order, [(3, false), (2, false), (1, false)]);
    }

    #[test]
    fn jobs_no_draw_asked_for_lately_are_stale() {
        let mut queue = queue(&[(1, 1), (3, 2)]);
        assert_eq!(queue.pop(3 + STALE_FRAMES), Some((2, false)));
        assert_eq!(queue.pop(3 + STALE_FRAMES), Some((1, true)));
    }

    #[test]
    fn a_draw_asking_again_keeps_a_job_fresh() {
        let seen = Arc::new(AtomicU64::new(1));
        let mut queue = Queue {
            jobs: vec![(seen.clone(), 7)],
        };
        seen.store(10, Ordering::Relaxed);
        assert_eq!(queue.pop(10 + STALE_FRAMES), Some((7, false)));
    }
}
