//! What services share: the kernel, settings, the title, the GPU and input.

use std::collections::HashMap;
use std::sync::Arc;

use lichii_common::HleResult;
use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_gpu::Gpu;
use lichii_input::NpadHub;
use lichii_kernel::Kernel;
use lichii_kernel::ipc::{ServiceThread, SessionHandler};
use lichii_kernel::process::Process;
use lichii_loader::program::ProgramImage;
use parking_lot::Mutex;

use crate::Service;
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
    pub(crate) unknown: UnknownCommands,
    threads: Mutex<HashMap<String, ServiceThread>>,
}

impl ServiceContext {
    /// Builds the shared state. `core::System` calls this once per boot, after creating the
    /// title's process, and passes the result to [`crate::register_all`].
    pub fn new(parts: ServiceParts) -> Arc<Self> {
        Arc::new(Self {
            kernel: parts.kernel,
            settings: parts.settings,
            program: parts.program,
            process: parts.process,
            gpu: parts.gpu,
            npad: parts.npad,
            perf: parts.perf,
            unknown: UnknownCommands::default(),
            threads: Mutex::new(HashMap::new()),
        })
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
