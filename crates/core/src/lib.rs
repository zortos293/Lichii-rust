//! `System`: owns every subsystem for one emulation session and boots a title.
//!
//! Boot order: settings, then the loader's [`ProgramImage`], then the kernel (DRAM sized by
//! [`MemoryLayout`], 4 or 6 cores), then the title's process, then the HLE services, then the
//! GPU, then the main thread.
//!
//! [`MemoryLayout`]: lichii_common::settings::MemoryLayout

mod timing;

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use lichii_common::settings::Settings;
use lichii_common::{ConsoleGeneration, ResultCode, VAddr};
use lichii_gpu::host1x::Host1x;
use lichii_gpu::nx2_memory::Nx2GpuMemory;
use lichii_input::hub::NpadHub;
use lichii_kernel::process::Process;
use lichii_kernel::{Kernel, KernelConfig, KernelError, KernelEvents};
use lichii_loader::{LoadOptions, LoaderError, ProgramImage};
use lichii_services::{ServiceContext, ServiceParts};
use parking_lot::{Condvar, Mutex};

pub use crate::timing::CoreTiming;
pub use lichii_common::perf::PerfStats;

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error("loading the title: {0}")]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error("{stage}: {code:?}")]
    Boot {
        stage: &'static str,
        code: ResultCode,
    },
}

/// Why the title stopped running.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExitReason {
    /// The process called `svcExitProcess` or its last thread exited.
    Exited,
    /// The process called `svcBreak` with a non-notification reason (an abort or a panic).
    Break { reason: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Booted,
    Running,
    Paused,
    Stopped,
}

/// The GPU-side pieces the services share. The GPU orchestrator will own these and its
/// front/back threads; [`System::start_gpu`] is where it gets built.
struct GpuParts {
    host1x: Host1x,
    nx2_memory: Option<Arc<Nx2GpuMemory>>,
}

/// One emulation session: a booted title and every subsystem it runs on.
pub struct System {
    settings: Arc<Settings>,
    timing: Arc<CoreTiming>,
    kernel: Kernel,
    program: Arc<ProgramImage>,
    process: Arc<Process>,
    services: Arc<ServiceContext>,
    gpu: GpuParts,
    npad: Arc<NpadHub>,
    perf: Arc<PerfStats>,
    exit: Arc<ExitSignal>,
    state: State,
}

impl System {
    /// Loads the title at `path` and boots it up to the point where [`System::run`] starts it.
    pub fn boot(settings: Arc<Settings>, path: &Path) -> Result<Self, SystemError> {
        tracing::info!(path = %path.display(), "loading title");
        let image = lichii_loader::load(path, &LoadOptions::default())?;
        Self::from_image(settings, image)
    }

    /// Boots an already loaded title.
    pub fn from_image(settings: Arc<Settings>, image: ProgramImage) -> Result<Self, SystemError> {
        let program = Arc::new(image);
        let console = match program.generation {
            ConsoleGeneration::Switch2 => ConsoleGeneration::Switch2,
            ConsoleGeneration::Switch1 => settings.system.preferred_console,
        };
        let dram_size = settings.system.memory_layout.dram_size(console);
        tracing::info!(
            program_id = format_args!("{:016x}", program.program_id),
            name = %program.name,
            title = ?program.generation,
            ?console,
            dram_gib = dram_size >> 30,
            modules = program.modules.len(),
            "booting"
        );

        let timing = Arc::new(CoreTiming::new());
        let exit = Arc::new(ExitSignal::default());
        let backend = settings.cpu.backend;
        let kernel = Kernel::new(KernelConfig {
            console,
            dram_size,
            pool_layout: None,
            fastmem: settings.cpu.fastmem,
            aslr: false,
            rng_seed: settings.system.rng_seed.map_or_else(entropy, u64::from),
            clock: timing.clone(),
            cpu: Arc::new(move |config| lichii_cpu::new_core(backend, config)),
            events: Some(exit.clone()),
        })?;
        tracing::info!(cores = kernel.core_count(), "kernel up");

        let boot = |stage| move |code| SystemError::Boot { stage, code };
        let process = kernel
            .create_process(&program)
            .map_err(boot("creating the process"))?;
        tracing::info!(
            process = process.name(),
            entry = format_args!("{:#x}", process.entry()),
            "process created"
        );

        let gpu = Self::start_gpu(&kernel, &program);
        let npad = Arc::new(NpadHub::default());
        let perf = Arc::new(PerfStats::default());
        let services = ServiceContext::new(ServiceParts {
            kernel: kernel.clone(),
            settings: settings.clone(),
            program: program.clone(),
            process: process.clone(),
            host1x: gpu.host1x.clone(),
            nx2_memory: gpu.nx2_memory.clone(),
            npad: npad.clone(),
            perf: perf.clone(),
        });
        if let Err(code) = lichii_services::register_all(&services) {
            kernel.shutdown();
            return Err(boot("registering services")(code));
        }
        tracing::info!("services registered");

        Ok(Self {
            settings,
            timing,
            kernel,
            program,
            process,
            services,
            gpu,
            npad,
            perf,
            exit,
            state: State::Booted,
        })
    }

    /// Builds the GPU for the title: the host1x device space (34-bit SMMU on Switch 1, 36-bit on
    /// Switch 2) and, for Switch 2 titles, the GPU memory registry the gsv sessions share.
    fn start_gpu(kernel: &Kernel, program: &ProgramImage) -> GpuParts {
        let address_space_bits = match kernel.console() {
            ConsoleGeneration::Switch1 => 34,
            ConsoleGeneration::Switch2 => 36,
        };
        let host1x = Host1x::new(kernel.physical_memory().clone(), address_space_bits);
        let nx2_memory = (program.generation == ConsoleGeneration::Switch2).then(|| {
            Arc::new(Nx2GpuMemory::new(
                host1x.memory.clone(),
                host1x.allocator.clone(),
            ))
        });
        tracing::info!(address_space_bits, nx2 = nx2_memory.is_some(), "gpu up");
        GpuParts { host1x, nx2_memory }
    }

    /// Starts the main thread, the CPU cores and the service threads, or resumes a paused session.
    pub fn run(&mut self) -> Result<(), SystemError> {
        match self.state {
            State::Booted => {
                let boot = |code| SystemError::Boot {
                    stage: "starting the main thread",
                    code,
                };
                self.kernel.start_process(&self.process).map_err(boot)?;
                self.kernel.run()?;
                self.state = State::Running;
                tracing::info!("running");
            }
            State::Paused => self.resume(),
            State::Running | State::Stopped => {}
        }
        Ok(())
    }

    /// Stops guest code and guest time until [`System::resume`].
    pub fn pause(&mut self) {
        if self.state != State::Running {
            return;
        }
        self.kernel.pause();
        self.timing.pause();
        self.state = State::Paused;
        tracing::info!("paused");
    }

    pub fn resume(&mut self) {
        if self.state != State::Paused {
            return;
        }
        self.timing.resume();
        self.kernel.resume();
        self.state = State::Running;
        tracing::info!("resumed");
    }

    /// Stops every thread and releases the title. The session cannot run again.
    pub fn stop(&mut self) {
        if self.state == State::Stopped {
            return;
        }
        self.kernel.shutdown();
        self.state = State::Stopped;
        tracing::info!("stopped");
    }

    pub fn is_paused(&self) -> bool {
        self.state == State::Paused
    }

    /// Blocks until the title exits or breaks, or until `timeout` passes.
    pub fn wait_for_exit(&self, timeout: Option<Duration>) -> Option<ExitReason> {
        self.exit.wait(timeout)
    }

    pub fn exit_reason(&self) -> Option<ExitReason> {
        *self.exit.reason.lock()
    }

    pub fn settings(&self) -> &Arc<Settings> {
        &self.settings
    }

    pub fn console(&self) -> ConsoleGeneration {
        self.kernel.console()
    }

    pub fn timing(&self) -> &Arc<CoreTiming> {
        &self.timing
    }

    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    pub fn program(&self) -> &Arc<ProgramImage> {
        &self.program
    }

    pub fn process(&self) -> &Arc<Process> {
        &self.process
    }

    pub fn services(&self) -> &Arc<ServiceContext> {
        &self.services
    }

    pub fn host1x(&self) -> &Host1x {
        &self.gpu.host1x
    }

    /// The Switch 2 GPU memory registry; `None` for Switch 1 titles.
    pub fn nx2_memory(&self) -> Option<&Arc<Nx2GpuMemory>> {
        self.gpu.nx2_memory.as_ref()
    }

    /// The input state the frontend publishes and hid reads.
    pub fn npad(&self) -> &Arc<NpadHub> {
        &self.npad
    }

    pub fn perf(&self) -> &Arc<PerfStats> {
        &self.perf
    }
}

impl Drop for System {
    fn drop(&mut self) {
        self.stop();
    }
}

fn entropy() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

#[derive(Default)]
struct ExitSignal {
    reason: Mutex<Option<ExitReason>>,
    changed: Condvar,
}

impl ExitSignal {
    fn set(&self, reason: ExitReason) {
        let mut slot = self.reason.lock();
        if slot.is_none() {
            *slot = Some(reason);
            self.changed.notify_all();
        }
    }

    fn wait(&self, timeout: Option<Duration>) -> Option<ExitReason> {
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut slot = self.reason.lock();
        while slot.is_none() {
            match deadline {
                Some(deadline) => {
                    if self.changed.wait_until(&mut slot, deadline).timed_out() {
                        break;
                    }
                }
                None => self.changed.wait(&mut slot),
            }
        }
        *slot
    }
}

impl KernelEvents for ExitSignal {
    fn on_break(&self, process: &Process, reason: u32, info: VAddr, size: u64) {
        if reason & 0x8000_0000 != 0 {
            tracing::info!(process = process.name(), reason, "break notification");
            return;
        }
        tracing::error!(
            process = process.name(),
            reason = format_args!("{reason:#x}"),
            info = format_args!("{info:#x}"),
            size,
            "guest break"
        );
        self.set(ExitReason::Break { reason });
    }

    fn on_process_exit(&self, process: &Process) {
        tracing::info!(process = process.name(), "process exited");
        self.set(ExitReason::Exited);
    }
}

#[cfg(test)]
mod tests;
