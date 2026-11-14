//! `System`: owns every subsystem for one emulation session and boots a title.
//!
//! Boot order: settings, then the loader's [`ProgramImage`], then the kernel (DRAM sized by
//! [`MemoryLayout`], 4 or 6 cores), then the title's process, then the GPU and the audio sink,
//! then the HLE services, then the main thread.
//!
//! [`MemoryLayout`]: lichii_common::settings::MemoryLayout

mod timing;

use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::RecvTimeoutError;
use std::time::{Duration, Instant, SystemTime};

use lichii_audio::sink::Sink;
use lichii_common::settings::{AudioSettings, Settings};
use lichii_common::{ConsoleGeneration, ResultCode, VAddr};
use lichii_gpu::backend::{GpuHandles, NullRasterizer, Rasterizer};
use lichii_gpu::host1x::Host1x;
use lichii_gpu::{Console, Gpu, GpuConfig};
use lichii_input::hub::NpadHub;
use lichii_kernel::process::Process;
use lichii_kernel::{Kernel, KernelConfig, KernelError, KernelEvents};
use lichii_loader::{LoadOptions, LoaderError, ProgramImage};
use lichii_services::{ServiceContext, ServiceParts, SessionEvents};
use parking_lot::{Condvar, Mutex};

pub use crate::timing::CoreTiming;
pub use lichii_services::Handover;

/// The bits of a program id that name the program within its application.
pub const PROGRAM_INDEX_MASK: u64 = 0x7ff;
pub use lichii_common::perf::PerfStats;

#[derive(Debug, thiserror::Error)]
pub enum SystemError {
    #[error("loading the title: {0}")]
    Loader(#[from] LoaderError),
    #[error(transparent)]
    Kernel(#[from] KernelError),
    #[error("the session has stopped")]
    Stopped,
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
    /// The title asked to start program `program_index` of its application in its place
    /// (`ExecuteProgram`, `CreateApplicationAndRequestToStart`). Boot it with
    /// `LoadOptions::program_index`, and hand it [`System::take_handover`] through
    /// [`System::set_handover`] before [`System::run`].
    ExecuteProgram { program_index: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Booted,
    Running,
    Paused,
    Stopped,
}

/// Builds the renderer once the GPU knows the handles fences and queries need. The frontend
/// passes its Vulkan renderer here; headless runs pass [`headless_renderer`].
pub type RendererFactory = Box<dyn FnOnce(GpuHandles) -> Box<dyn Rasterizer> + Send>;

/// A [`RendererFactory`] that draws nothing: [`NullRasterizer`] runs fences and releases at once.
pub fn headless_renderer() -> RendererFactory {
    Box::new(|handles| Box::new(NullRasterizer::new(handles)))
}

/// What the application running a session provides: the renderer, and the state it shares with
/// the guest.
pub struct Host {
    pub renderer: RendererFactory,
    /// Controller state the frontend's input thread publishes; hid reads it.
    pub npad: Arc<NpadHub>,
    /// Counters the frontend's overlay shows; the presenter and the services record into it.
    pub perf: Arc<PerfStats>,
}

impl Host {
    /// A host without a window: [`headless_renderer`], no controller input, fresh counters.
    pub fn headless() -> Self {
        Self {
            renderer: headless_renderer(),
            npad: Arc::new(NpadHub::default()),
            perf: Arc::new(PerfStats::default()),
        }
    }
}

/// One emulation session: a booted title and every subsystem it runs on.
pub struct System {
    settings: Arc<Settings>,
    timing: Arc<CoreTiming>,
    kernel: Kernel,
    program: Arc<ProgramImage>,
    process: Arc<Process>,
    services: Arc<ServiceContext>,
    gpu: Arc<Gpu>,
    npad: Arc<NpadHub>,
    perf: Arc<PerfStats>,
    audio: Arc<Sink>,
    exit: Arc<ExitSignal>,
    state: State,
}

impl System {
    /// Loads the title at `path` and boots it up to the point where [`System::run`] starts it.
    pub fn boot(settings: Arc<Settings>, path: &Path, host: Host) -> Result<Self, SystemError> {
        tracing::info!(path = %path.display(), "loading title");
        let image = lichii_loader::load(path, &LoadOptions::default())?;
        Self::from_image(settings, image, host)
    }

    /// Boots an already loaded title.
    pub fn from_image(
        settings: Arc<Settings>,
        image: ProgramImage,
        host: Host,
    ) -> Result<Self, SystemError> {
        let Host {
            renderer,
            npad,
            perf,
        } = host;
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

        let timing = Arc::new(CoreTiming::new(program.generation));
        let exit = Arc::new(ExitSignal::default());
        let backend = settings.cpu.backend;
        let kernel = Kernel::new(KernelConfig {
            console,
            dram_size,
            pool_layout: None,
            fastmem: settings.cpu.fastmem,
            cpu_accuracy: settings.cpu.accuracy,
            aslr: false,
            rng_seed: settings.system.rng_seed.map_or_else(entropy, u64::from),
            enforce_service_access: settings.system.enforce_service_access,
            svc_trace: settings.debug.svc_trace,
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

        let gpu = Self::start_gpu(&kernel, &program, &timing, renderer);
        gpu.host1x().memory.attach_process(process.memory().clone());
        let audio = Arc::new(Sink::new(&settings.audio));
        let services = ServiceContext::new(ServiceParts {
            kernel: kernel.clone(),
            settings: settings.clone(),
            program: program.clone(),
            process: process.clone(),
            gpu: gpu.clone(),
            npad: npad.clone(),
            perf: perf.clone(),
            audio: audio.clone(),
            session: exit.clone(),
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
            audio,
            exit,
            state: State::Booted,
        })
    }

    /// Builds the GPU for the title, with its front and back threads: the host1x device space
    /// (34-bit SMMU for Switch 1 titles, 36-bit for Switch 2 titles), the console's channel classes
    /// and, for Switch 2 titles, the GPU memory registry the gsv sessions share. The GPU follows the
    /// title, not the emulated console: a Switch 1 title on a Switch 2 console still gets Maxwell.
    fn start_gpu(
        kernel: &Kernel,
        program: &ProgramImage,
        timing: &Arc<CoreTiming>,
        renderer: RendererFactory,
    ) -> Arc<Gpu> {
        let (console, address_space_bits) = match program.generation {
            ConsoleGeneration::Switch1 => (Console::Switch1, 34),
            ConsoleGeneration::Switch2 => (Console::Switch2, 36),
        };
        let host1x = Host1x::new(kernel.physical_memory().clone(), address_space_bits);
        let gpu = Gpu::new(host1x, timing.clone(), GpuConfig::new(console), renderer);
        tracing::info!(?console, address_space_bits, "gpu up");
        Arc::new(gpu)
    }

    /// Starts the main thread, the CPU cores and the service threads, or resumes a paused session.
    /// A start that fails stops the session.
    pub fn run(&mut self) -> Result<(), SystemError> {
        match self.state {
            State::Booted => {
                let started = match self.kernel.start_process(&self.process) {
                    Ok(_) => self.kernel.run().map_err(SystemError::from),
                    Err(code) => Err(SystemError::Boot {
                        stage: "starting the main thread",
                        code,
                    }),
                };
                if let Err(err) = started {
                    self.stop();
                    return Err(err);
                }
                self.state = State::Running;
                tracing::info!("running");
            }
            State::Paused => self.resume(),
            State::Running => {}
            State::Stopped => return Err(SystemError::Stopped),
        }
        Ok(())
    }

    /// Stops guest code, guest time and audio output until [`System::resume`].
    pub fn pause(&mut self) {
        if self.state != State::Running {
            return;
        }
        self.kernel.pause();
        self.gpu.pause(true);
        self.audio.set_paused(true);
        self.timing.pause();
        self.state = State::Paused;
        tracing::info!("paused");
    }

    /// Lets guest code and guest time run again after [`System::pause`].
    pub fn resume(&mut self) {
        if self.state != State::Paused {
            return;
        }
        self.timing.resume();
        self.audio.set_paused(false);
        self.gpu.pause(false);
        self.kernel.resume();
        self.state = State::Running;
        tracing::info!("resumed");
    }

    /// Stops every thread and releases the title. The session cannot run again.
    pub fn stop(&mut self) {
        if self.state == State::Stopped {
            return;
        }
        self.let_the_title_exit();
        self.kernel.pause();
        noting_slow("stopping the GPU", || self.gpu.shutdown());
        self.audio.set_paused(true);
        noting_slow("stopping the kernel", || self.kernel.shutdown());
        self.state = State::Stopped;
        tracing::info!("stopped");
    }

    /// Sends a running title that holds the exit lock the exit request and waits up to
    /// `system.exit_timeout_ms` for it to save and exit; past that, [`System::stop`] stops it.
    fn let_the_title_exit(&self) {
        let timeout = Duration::from_millis(self.settings.system.exit_timeout_ms.into());
        if timeout.is_zero() || self.state != State::Running || !self.services.request_exit() {
            return;
        }
        let started = Instant::now();
        match self.exit.wait(Some(timeout)) {
            Some(_) => tracing::info!(took = ?started.elapsed(), "the title exited on request"),
            None => tracing::warn!(?timeout, "the title did not exit on request; stopping it"),
        }
    }

    /// Whether the session is paused.
    pub fn is_paused(&self) -> bool {
        self.state == State::Paused
    }

    /// Blocks until the title exits or breaks, or until `timeout` passes.
    pub fn wait_for_exit(&self, timeout: Option<Duration>) -> Option<ExitReason> {
        self.exit.wait(timeout)
    }

    /// Why the title stopped, once it has.
    pub fn exit_reason(&self) -> Option<ExitReason> {
        *self.exit.reason.lock()
    }

    /// The settings the session booted with.
    pub fn settings(&self) -> &Arc<Settings> {
        &self.settings
    }

    /// The console being emulated: Switch 2 for Switch 2 titles, the preferred console otherwise.
    pub fn console(&self) -> ConsoleGeneration {
        self.kernel.console()
    }

    /// The session's guest clock.
    pub fn timing(&self) -> &Arc<CoreTiming> {
        &self.timing
    }

    /// The HLE kernel running the title.
    pub fn kernel(&self) -> &Kernel {
        &self.kernel
    }

    /// The loaded title.
    pub fn program(&self) -> &Arc<ProgramImage> {
        &self.program
    }

    /// The title's process.
    pub fn process(&self) -> &Arc<Process> {
        &self.process
    }

    /// State shared by every HLE service.
    pub fn services(&self) -> &Arc<ServiceContext> {
        &self.services
    }

    /// The GPU, with host1x and (for Switch 2 titles) the GPU memory registry.
    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }

    /// The input state the frontend publishes and hid reads.
    /// The host audio sink the audio services play on.
    pub fn audio(&self) -> &Arc<Sink> {
        &self.audio
    }

    /// Applies the volume and mute of `audio` to every stream at once. The output device is
    /// chosen at boot, so changing it takes a new session.
    pub fn set_audio_volume(&self, audio: &AudioSettings) {
        self.audio.set_host_volume(audio.volume, audio.muted);
    }

    pub fn npad(&self) -> &Arc<NpadHub> {
        &self.npad
    }

    /// What the title leaves the program it asked to start ([`ExitReason::ExecuteProgram`]): its
    /// program index and its user channel, which is empty unless the title asked, and after the
    /// first call.
    pub fn take_handover(&self) -> Handover {
        Handover {
            previous_program_index: (self.program.program_id & PROGRAM_INDEX_MASK) as u8,
            user_channel: std::mem::take(&mut *self.exit.user_channel.lock()),
        }
    }

    /// Gives the title what the program before it handed over. Call it before the first
    /// [`System::run`].
    pub fn set_handover(&self, handover: Handover) {
        self.services.receive_handover(handover);
    }

    /// Frame and performance counters for the frontend overlay.
    pub fn perf(&self) -> &Arc<PerfStats> {
        &self.perf
    }
}

impl Drop for System {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Runs one step of [`System::stop`], warning when it takes longer than [`SLOW_STOP`] and
/// again when it ends, so a stop that hangs says which step.
fn noting_slow(step: &'static str, run: impl FnOnce()) {
    const SLOW_STOP: Duration = Duration::from_secs(5);
    let (done, finished) = std::sync::mpsc::channel::<()>();
    let watchdog = std::thread::Builder::new()
        .name("stop watchdog".into())
        .spawn(move || {
            if finished.recv_timeout(SLOW_STOP) == Err(RecvTimeoutError::Timeout) {
                tracing::warn!("{step} takes longer than {SLOW_STOP:?}");
                let _ = finished.recv();
                tracing::warn!("{step} finished");
            }
        });
    run();
    drop(done);
    if let Ok(watchdog) = watchdog {
        let _ = watchdog.join();
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
    user_channel: Mutex<Vec<Vec<u8>>>,
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

impl SessionEvents for ExitSignal {
    fn execute_program(&self, program_index: u8, user_channel: Vec<Vec<u8>>) {
        if self.reason.lock().is_some() {
            return;
        }
        *self.user_channel.lock() = user_channel;
        self.set(ExitReason::ExecuteProgram { program_index });
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
