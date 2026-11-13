//! What services share: the kernel, settings, the title, the GPU, input and the audio sink.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use lichii_audio::sink::Sink;
use lichii_common::HleResult;
use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_fs::ContentIndex;
use lichii_gpu::Gpu;
use lichii_gpu::host1x::{SyncpointBank, SyncpointWaitError};
use lichii_input::NpadHub;
use lichii_kernel::Kernel;
use lichii_kernel::ipc::{ServiceThread, SessionHandler};
use lichii_kernel::process::Process;
use lichii_loader::program::ProgramImage;
use parking_lot::{Mutex, RwLock};

use crate::Service;
use crate::applets::{AppletFrontend, HeadlessApplets};
use crate::dispatch::{Dispatch, UnknownCommands};

/// The parts `core::System` hands to the services.
pub struct ServiceParts {
    pub kernel: Kernel,
    pub settings: Arc<Settings>,
    /// The running title, including its RomFS and NACP.
    pub program: Arc<ProgramImage>,
    /// The title's process.
    pub process: Arc<Process>,
    /// The GPU: channels, address spaces, host1x (SMMU, device allocator, syncpoints) and, for
    /// Switch 2 titles, the GPU memory registry.
    pub gpu: Arc<Gpu>,
    pub npad: Arc<NpadHub>,
    pub perf: Arc<PerfStats>,
    /// The host audio sink, built by `core::System` from the audio settings.
    pub audio: Arc<Sink>,
    /// What the title asks of the session that runs it.
    pub session: Arc<dyn SessionEvents>,
}

/// What a program leaves the program it starts: its own index, which `GetPreviousProgramIndex`
/// returns, and its user channel.
pub struct Handover {
    pub previous_program_index: u8,
    pub user_channel: Vec<Vec<u8>>,
}

/// Requests from the title that only the session running it can carry out.
pub trait SessionEvents: Send + Sync {
    /// The title asked to stop and start program `program_index` of its application
    /// (`ExecuteProgram`, `CreateApplicationAndRequestToStart`), handing `user_channel`
    /// over to it.
    fn execute_program(&self, program_index: u8, user_channel: Vec<Vec<u8>>);
}

/// Shared state of every service. Services keep an `Arc<ServiceContext>`; the fields are the
/// [`ServiceParts`] it was built from, shared with the rest of the emulator.
pub struct ServiceContext {
    /// The kernel the services are registered with.
    pub kernel: Kernel,
    /// Settings as they were when the title booted.
    pub settings: Arc<Settings>,
    /// The running title, including its RomFS and NACP.
    pub program: Arc<ProgramImage>,
    /// The title's process.
    pub process: Arc<Process>,
    /// The GPU: channels, address spaces, host1x (SMMU, device allocator, syncpoints) and, for
    /// Switch 2 titles, the GPU memory registry.
    pub gpu: Arc<Gpu>,
    /// Controller state from the frontend's input thread; hid reads it, and writes vibration.
    pub npad: Arc<NpadHub>,
    /// Counters the overlay shows; vi/gsv record guest frames here.
    pub perf: Arc<PerfStats>,
    /// The host audio sink every audio service plays on. `core::System` owns it and pauses it
    /// and sets its host volume; the services set the device volume.
    pub audio: Arc<Sink>,
    pub(crate) session: Arc<dyn SessionEvents>,
    pub(crate) unknown: UnknownCommands,
    /// The application's applet state, once am registered: a restarted program's user channel
    /// goes there.
    pub(crate) am: OnceLock<Arc<crate::am::Applet>>,
    /// The nvdrv state, once nvdrv registered: gsv opens `nx2:nvsrv` sessions with it.
    pub(crate) nvdrv: OnceLock<Arc<crate::nvdrv::Nvdrv>>,
    /// The memory behind nx2:gpu's lazy GPU-only regions, once the title made one.
    pub(crate) gpu_reserve: OnceLock<Arc<crate::gsv::Reserve>>,
    /// The hid state, once hid registered: am's controller applet reports its connected npads.
    pub(crate) hid: OnceLock<Arc<crate::hid::Hid>>,
    /// The selected user's id, once acc registered: am's preselected-user launch parameter names
    /// it.
    pub(crate) user: OnceLock<[u8; 16]>,
    /// The vi state, once vi registered: am creates the application's layer in it.
    pub(crate) vi: OnceLock<Arc<crate::vi::Vi>>,
    /// Which npad has an NFC reader; hid's updater polls it so NFC sees controller changes.
    pub(crate) nfc_readers: Arc<crate::nfc::common::Readers>,
    /// Reads `set:sys`'s NFC enable flag, once set registered.
    pub(crate) nfc_enabled: OnceLock<Box<dyn Fn() -> bool + Send + Sync>>,
    /// The NAND's registered content, indexed on first use by [`crate::fs::system_data`].
    pub(crate) installed_content: OnceLock<Option<ContentIndex>>,
    applets: RwLock<Arc<dyn AppletFrontend>>,
    threads: Mutex<HashMap<String, ServiceThread>>,
}

impl ServiceContext {
    /// Builds the shared state. `core::System` calls this once per boot, after creating the
    /// title's process, and passes the result to [`crate::register_all`].
    pub fn new(parts: ServiceParts) -> Arc<Self> {
        let nfc_readers = Arc::new(crate::nfc::common::Readers::new(&parts.npad));
        Arc::new(Self {
            kernel: parts.kernel,
            settings: parts.settings,
            program: parts.program,
            process: parts.process,
            gpu: parts.gpu,
            npad: parts.npad,
            perf: parts.perf,
            audio: parts.audio,
            session: parts.session,
            unknown: UnknownCommands::default(),
            am: OnceLock::new(),
            nvdrv: OnceLock::new(),
            gpu_reserve: OnceLock::new(),
            hid: OnceLock::new(),
            user: OnceLock::new(),
            vi: OnceLock::new(),
            nfc_readers,
            nfc_enabled: OnceLock::new(),
            installed_content: OnceLock::new(),
            applets: RwLock::new(Arc::new(HeadlessApplets)),
            threads: Mutex::new(HashMap::new()),
        })
    }

    /// Answers library applets (error, profile selector, controllers, software keyboard) with
    /// `frontend` from now on, instead of [`HeadlessApplets`].
    pub fn set_applet_frontend(&self, frontend: Arc<dyn AppletFrontend>) {
        *self.applets.write() = frontend;
    }

    /// Gives the title what the program before it handed over, before the title starts.
    pub fn receive_handover(&self, handover: Handover) {
        match self.am.get() {
            Some(applet) => applet.receive_handover(handover),
            None => tracing::warn!("am is not registered; the handover is lost"),
        }
    }

    /// The frontend that answers library applets.
    pub(crate) fn applet_frontend(&self) -> Arc<dyn AppletFrontend> {
        self.applets.read().clone()
    }

    /// The service thread called `name`, created on first use. Services that may block (file
    /// I/O, GPU waits) use their own thread; the rest share `"services"`.
    pub fn service_thread(&self, name: &str) -> HleResult<ServiceThread> {
        let mut threads = self.threads.lock();
        if let Some(thread) = threads.get(name) {
            return Ok(thread.clone());
        }
        let thread = self.kernel.create_service_thread(name).map_err(|err| {
            tracing::error!("creating service thread {name}: {err}");
            lichii_kernel::result::OUT_OF_RESOURCE
        })?;
        threads.insert(name.into(), thread.clone());
        Ok(thread)
    }

    /// Wraps a service or sub-interface for the kernel (unknown commands and IPC tracing).
    pub fn handler<S: Service>(self: &Arc<Self>, service: S) -> Arc<dyn SessionHandler> {
        Arc::new(Dispatch {
            service,
            context: self.clone(),
        })
    }

    /// Registers `name` with `sm:`, served by service thread `thread`. Every session gets a new
    /// service from `make`.
    pub fn register<S: Service>(
        self: &Arc<Self>,
        name: &str,
        thread: &str,
        make: impl Fn(&Arc<Self>) -> S + Send + Sync + 'static,
    ) -> HleResult<()> {
        let thread = self.service_thread(thread)?;
        let context = self.clone();
        self.kernel.service_manager().register(
            name,
            &thread,
            Arc::new(move || context.handler(make(&context))),
        )
    }
}

/// Waits until guest syncpoint `id` reaches `threshold` or `timeout` passes, like
/// [`SyncpointBank::wait`], and times out once the kernel shuts down: the GPU stops first, so the
/// value may never come, and a service thread still waiting would hold up the shutdown.
pub(crate) fn wait_guest_syncpoint(
    kernel: &Kernel,
    bank: &SyncpointBank,
    id: u32,
    threshold: u32,
    timeout: Option<Duration>,
) -> Result<u32, SyncpointWaitError> {
    const SHUTDOWN_CHECK: Duration = Duration::from_millis(100);
    let deadline = timeout.map(|timeout| Instant::now() + timeout);
    loop {
        let left = deadline.map_or(SHUTDOWN_CHECK, |deadline| {
            deadline.saturating_duration_since(Instant::now())
        });
        match bank.wait(id, threshold, Some(left.min(SHUTDOWN_CHECK))) {
            Err(SyncpointWaitError::TimedOut { .. })
                if !kernel.is_shutting_down()
                    && deadline.is_none_or(|deadline| Instant::now() < deadline) => {}
            result => return result,
        }
    }
}
