//! HLE Horizon kernel: processes, threads, the multi-core scheduler, synchronisation objects,
//! memory management and supervisor calls.
//!
//! [`Kernel::new`] sets up DRAM, the pools and the scheduler; [`Kernel::create_process`] and
//! [`Kernel::start_process`] load and start a title; [`Kernel::run`] starts one host thread per
//! emulated core (4 for Switch 1, 6 for Switch 2), each running [`lichii_cpu::CpuCore::run`] and
//! handling what it halts for.

pub mod arbiter;
mod cpu_manager;
pub mod event;
pub mod handle_table;
pub mod ipc;
pub mod memory;
pub mod memory_objects;
pub mod object;
pub mod process;
pub mod resource_limit;
pub mod result;
pub mod scheduler;
pub mod sm;
pub mod svc;
mod sync;
pub mod thread;
mod timer;
mod util;

#[cfg(any(test, feature = "testing"))]
pub mod testing;
#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use lichii_common::settings::CpuAccuracy;
use lichii_common::{ConsoleGeneration, HleResult, PAGE_SIZE, VAddr, align_up};
use lichii_cpu::{CpuConfig, CpuCore};
use lichii_loader::program::ProgramImage;
use lichii_memory::{
    AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory, PhysicalMemoryError,
};
use parking_lot::Mutex;

use crate::arbiter::Arbiter;
use crate::ipc::{ClientSession, ServiceThread, SessionFactory};
use crate::memory::{MemoryManager, MemoryState, PoolLayout, Region, Regions};
use crate::object::KObject;
use crate::process::{NewProcess, Process, ProcessId, ProcessParams, ProcessState};
use crate::resource_limit::{LimitableResource, ResourceLimit};
use crate::scheduler::Scheduler;
use crate::sm::ServiceManager;
use crate::thread::{LOWEST_PRIORITY, NewThread, Thread};
use crate::timer::Timer;
use crate::util::Rng;

pub use crate::event::Event;
pub use crate::memory::Pool;
pub use lichii_cpu::GuestClock;

/// A range of translated code to drop: `(process, address, size)`.
pub(crate) type CodeRange = (ProcessId, VAddr, u64);

/// Builds the CPU core that runs one process's threads on one emulated core.
pub type CpuFactory = Arc<dyn Fn(CpuConfig) -> Box<dyn CpuCore> + Send + Sync>;

/// Things the kernel reports to the rest of the emulator. Every method has a logging default.
pub trait KernelEvents: Send + Sync {
    fn on_debug_output(&self, process: &Process, text: &str) {
        tracing::info!(target: "guest", process = process.name(), "{}", text.trim_end());
    }

    fn on_break(&self, _process: &Process, _reason: u32, _info: VAddr, _size: u64) {}

    fn on_process_exit(&self, process: &Process) {
        tracing::info!(process = process.name(), "process exited");
    }
}

struct LogEvents;
impl KernelEvents for LogEvents {}

pub struct KernelConfig {
    pub console: ConsoleGeneration,
    pub dram_size: u64,
    /// Pool split; `None` uses Horizon's split for `dram_size`.
    pub pool_layout: Option<PoolLayout>,
    /// Give process address spaces a fastmem arena for the JIT.
    pub fastmem: bool,
    /// The accuracy every CPU core is created with.
    pub cpu_accuracy: CpuAccuracy,
    /// Slide the code region base of new processes (ASLR).
    pub aslr: bool,
    /// Seed for ASLR and random entropy.
    pub rng_seed: u64,
    pub clock: Arc<dyn GuestClock>,
    pub cpu: CpuFactory,
    pub events: Option<Arc<dyn KernelEvents>>,
}

#[derive(Debug, thiserror::Error)]
pub enum KernelError {
    #[error(transparent)]
    Memory(#[from] PhysicalMemoryError),
    #[error("spawning a kernel thread: {0}")]
    Spawn(std::io::Error),
}

pub(crate) struct KernelInner {
    pub console: ConsoleGeneration,
    pub physical: Arc<PhysicalMemory>,
    pub memory: Arc<MemoryManager>,
    pub scheduler: Arc<Scheduler>,
    pub timer: Timer,
    pub arbiter: Arbiter,
    pub monitor: Arc<ExclusiveMonitor>,
    pub clock: Arc<dyn GuestClock>,
    cpu_factory: CpuFactory,
    fastmem: bool,
    cpu_accuracy: CpuAccuracy,
    aslr: bool,
    events: Arc<dyn KernelEvents>,
    rng: Mutex<Rng>,
    next_process_id: AtomicU64,
    next_thread_id: AtomicU64,
    processes: Mutex<Vec<Arc<Process>>>,
    /// Counts process exits, so cores know when to drop the CPU state of exited processes.
    process_exits: AtomicU64,
    invalidations: Box<[Mutex<Vec<CodeRange>>]>,
    host_threads: Mutex<Vec<JoinHandle<()>>>,
    application_limit: Arc<ResourceLimit>,
    services: Mutex<ServiceThreads>,
    named_ports: Mutex<Vec<(String, ServiceThread, SessionFactory)>>,
    service_manager: Arc<ServiceManager>,
}

#[derive(Default)]
struct ServiceThreads {
    threads: Vec<ServiceThread>,
    joins: Vec<JoinHandle<()>>,
    running: bool,
}

/// The kernel. Cloning it is cheap; every clone refers to the same kernel.
#[derive(Clone)]
pub struct Kernel(Arc<KernelInner>);

impl Kernel {
    pub fn new(config: KernelConfig) -> Result<Self, KernelError> {
        let physical = Arc::new(PhysicalMemory::new(config.dram_size)?);
        let layout = config
            .pool_layout
            .unwrap_or_else(|| PoolLayout::for_dram(config.dram_size, config.console));
        let memory = Arc::new(MemoryManager::new(physical.clone(), layout));
        let cores = config.console.core_count();
        let service_manager = Arc::new(ServiceManager::default());
        let sm_thread = ServiceThread::new("sm");
        let named_ports = vec![(
            "sm:".to_string(),
            sm_thread.clone(),
            service_manager.session_factory(),
        )];
        Ok(Self(Arc::new(KernelInner {
            services: Mutex::new(ServiceThreads {
                threads: vec![sm_thread],
                ..ServiceThreads::default()
            }),
            named_ports: Mutex::new(named_ports),
            service_manager,
            console: config.console,
            application_limit: Arc::new(ResourceLimit::application(memory.size(Pool::Application))),
            physical,
            memory,
            scheduler: Arc::new(Scheduler::new(cores)),
            timer: Timer::default(),
            arbiter: Arbiter::default(),
            monitor: Arc::new(ExclusiveMonitor::new(cores)),
            clock: config.clock,
            cpu_factory: config.cpu,
            fastmem: config.fastmem,
            cpu_accuracy: config.cpu_accuracy,
            aslr: config.aslr,
            events: config.events.unwrap_or_else(|| Arc::new(LogEvents)),
            rng: Mutex::new(Rng::new(config.rng_seed)),
            next_process_id: AtomicU64::new(0x51),
            next_thread_id: AtomicU64::new(1),
            processes: Mutex::new(Vec::new()),
            process_exits: AtomicU64::new(0),
            invalidations: (0..cores).map(|_| Mutex::new(Vec::new())).collect(),
            host_threads: Mutex::new(Vec::new()),
        })))
    }

    pub fn console(&self) -> ConsoleGeneration {
        self.0.console
    }

    pub fn core_count(&self) -> usize {
        self.0.scheduler.core_count()
    }

    /// The guest system counter the cores read (`CNTPCT_EL0`, `CNTFRQ_EL0`); services take guest
    /// ticks and time from it.
    pub fn clock(&self) -> &Arc<dyn GuestClock> {
        &self.0.clock
    }

    pub fn physical_memory(&self) -> &Arc<PhysicalMemory> {
        &self.0.physical
    }

    pub fn memory_manager(&self) -> &Arc<MemoryManager> {
        &self.0.memory
    }

    pub fn exclusive_monitor(&self) -> &Arc<ExclusiveMonitor> {
        &self.0.monitor
    }

    pub fn processes(&self) -> Vec<Arc<Process>> {
        self.0.processes.lock().clone()
    }

    /// Creates the process of a title. The code region base follows `image.address_space`
    /// (slid when [`KernelConfig::aslr`] is set); every module is mapped at `base +
    /// module.offset` with its segment permissions, its image copied in and the rest of its size
    /// zeroed. The raw ACI0 descriptors are validated here. Start it with
    /// [`Kernel::start_process`].
    pub fn create_process(&self, image: &ProgramImage) -> HleResult<Arc<Process>> {
        self.0.create_process(image)
    }

    /// Maps the main thread's stack (`main_thread.stack_size`) in the stack region and starts it
    /// at `base + entry_offset` with `main_thread`'s priority and core; `x1` holds its handle.
    pub fn start_process(&self, process: &Arc<Process>) -> HleResult<Arc<Thread>> {
        self.0.start_process(process)
    }

    /// The `sm:` registry HLE services register with.
    pub fn service_manager(&self) -> &Arc<ServiceManager> {
        &self.0.service_manager
    }

    /// A new host thread for HLE sessions. It starts with [`Kernel::run`] (at once when the
    /// kernel already runs); requests queue until then.
    pub fn create_service_thread(&self, name: &str) -> Result<ServiceThread, KernelError> {
        let thread = ServiceThread::new(name);
        let mut services = self.0.services.lock();
        if services.running {
            let join = thread.spawn().map_err(KernelError::Spawn)?;
            services.joins.push(join);
        }
        services.threads.push(thread.clone());
        Ok(thread)
    }

    /// Makes `name` reachable with `svcConnectToNamedPort`; each connection gets a new session
    /// with a handler from `factory`, served by `thread`. `sm:` is registered by the kernel.
    pub fn register_named_port(&self, name: &str, thread: &ServiceThread, factory: SessionFactory) {
        let mut ports = self.0.named_ports.lock();
        ports.retain(|(port, _, _)| port != name);
        ports.push((name.into(), thread.clone(), factory));
    }

    /// Starts the core threads, the timer thread and the service threads.
    pub fn run(&self) -> Result<(), KernelError> {
        self.spawn_service_threads()?;
        let mut threads = self.0.host_threads.lock();
        let kernel = self.0.clone();
        threads.push(
            std::thread::Builder::new()
                .name("kernel timer".into())
                .spawn(move || kernel.timer.run(&kernel.scheduler))
                .map_err(KernelError::Spawn)?,
        );
        for core in 0..self.core_count() {
            let kernel = self.0.clone();
            threads.push(
                std::thread::Builder::new()
                    .name(format!("cpu core {core}"))
                    .spawn(move || cpu_manager::run_core(kernel, core))
                    .map_err(KernelError::Spawn)?,
            );
        }
        Ok(())
    }

    pub(crate) fn spawn_service_threads(&self) -> Result<(), KernelError> {
        let mut services = self.0.services.lock();
        if !services.running {
            services.running = true;
            let joins = services
                .threads
                .iter()
                .map(ServiceThread::spawn)
                .collect::<Result<Vec<_>, _>>()
                .map_err(KernelError::Spawn)?;
            services.joins = joins;
        }
        Ok(())
    }

    /// Stops every core and kernel thread and releases all processes.
    pub fn shutdown(&self) {
        self.0.scheduler.shutdown();
        self.0.timer.shutdown();
        let threads = std::mem::take(&mut *self.0.host_threads.lock());
        for thread in threads {
            let _ = thread.join();
        }
        let services = std::mem::take(&mut *self.0.services.lock());
        for thread in &services.threads {
            thread.stop();
        }
        for join in services.joins {
            let _ = join.join();
        }
        self.0.service_manager.clear();
        self.0.named_ports.lock().clear();
        self.0.scheduler.clear();
        let processes = std::mem::take(&mut *self.0.processes.lock());
        for process in processes {
            process.threads.lock().clear();
            if process.state() != ProcessState::Exited {
                process.finalize();
            }
        }
    }

    /// Stops running guest code: every core puts its thread back in its run queue and idles until
    /// [`Kernel::resume`]. Service threads, waits and timers keep going. Returns once every core
    /// has left guest code, so a clock paused after this sees no more guest counter reads.
    pub fn pause(&self) {
        self.0.scheduler.pause();
    }

    /// Lets the cores pick threads again after [`Kernel::pause`].
    pub fn resume(&self) {
        self.0.scheduler.resume();
    }

    /// Whether [`Kernel::pause`] is in effect.
    pub fn is_paused(&self) -> bool {
        self.0.scheduler.is_paused()
    }

    /// Asks every thread of `process` to terminate.
    pub fn terminate_process(&self, process: &Arc<Process>) {
        self.0.terminate_process(process);
    }
}

impl KernelInner {
    pub(crate) fn events(&self) -> &dyn KernelEvents {
        &*self.events
    }

    pub(crate) fn connect_named_port(&self, name: &str) -> Option<Arc<ClientSession>> {
        let ports = self.named_ports.lock();
        let (_, thread, factory) = ports.iter().find(|(port, _, _)| port == name)?;
        Some(thread.session(factory()))
    }

    fn code_address(&self, address_space_bits: u32, aslr: bool) -> VAddr {
        let base = match address_space_bits {
            32 => 0x20_0000,
            _ => 0x800_0000,
        };
        if aslr && address_space_bits == 39 {
            base + self.rng.lock().below(0x40) * 0x20_0000
        } else {
            base
        }
    }

    fn create_process(self: &Arc<Self>, image: &ProgramImage) -> HleResult<Arc<Process>> {
        let params = ProcessParams::from_image(image, self.aslr)?;
        let base = self.code_address(params.address_space_bits, params.aslr);
        let code = Region::new(base, align_up(image.code_size.max(PAGE_SIZE), PAGE_SIZE));
        let entry = base + image.entry_offset;
        if !code.contains(entry, 4) {
            return Err(result::INVALID_ARGUMENT);
        }
        let regions = Regions::compute(Process::layout(&params, code), &mut self.rng.lock())?;
        if !regions.code.contains(code.start, code.size()) {
            return Err(result::INVALID_MEMORY_REGION);
        }
        let space = if self.fastmem {
            match AddressSpace::with_fastmem(self.physical.clone(), params.address_space_bits) {
                Ok(space) => space,
                Err(err) => {
                    tracing::warn!("fastmem unavailable: {err}");
                    AddressSpace::new(self.physical.clone(), params.address_space_bits)
                }
            }
        } else {
            AddressSpace::new(self.physical.clone(), params.address_space_bits)
        };
        let random_entropy = {
            let mut rng = self.rng.lock();
            std::array::from_fn(|_| rng.next_u64())
        };
        let resource_limit = if params.is_application {
            self.application_limit.clone()
        } else {
            Arc::new(ResourceLimit::new([i64::MAX, 1024, 1024, 1024, 1024]))
        };
        let process = Process::new(NewProcess {
            id: self.next_process_id.fetch_add(1, Ordering::Relaxed),
            params,
            kernel: Arc::downgrade(self),
            space: Arc::new(space),
            regions,
            code,
            entry,
            modules: &image.modules,
            random_entropy,
            resource_limit,
            memory: self.memory.clone(),
        })?;
        self.processes.lock().push(process.clone());
        Ok(process)
    }

    pub(crate) fn create_thread(
        &self,
        process: &Arc<Process>,
        entry: VAddr,
        arg: u64,
        stack_top: VAddr,
        priority: u32,
        core: i32,
    ) -> HleResult<Arc<Thread>> {
        let caps = process.capabilities();
        let core = if core == -2 {
            process.params().main_thread_core as i32
        } else {
            core
        };
        let cores = self.scheduler.core_count();
        let core_mask = if caps.core_mask == 0 {
            self.scheduler.all_cores_mask()
        } else {
            caps.core_mask
        };
        if core < 0 || core as usize >= cores || core_mask & (1 << core) == 0 {
            return Err(result::INVALID_CORE_ID);
        }
        if priority > LOWEST_PRIORITY as u32
            || (caps.priority_mask != 0 && caps.priority_mask & (1 << priority) == 0)
        {
            return Err(result::INVALID_PRIORITY);
        }
        if process.state() == ProcessState::Exiting || process.state() == ProcessState::Exited {
            return Err(result::TERMINATION_REQUESTED);
        }
        process
            .resource_limit()
            .reserve(LimitableResource::Threads, 1)
            .map_err(|_| result::LIMIT_REACHED)?;
        let tls = match process.page_table().allocate_tls() {
            Ok(tls) => tls,
            Err(code) => {
                process
                    .resource_limit()
                    .release(LimitableResource::Threads, 1);
                return Err(code);
            }
        };
        if process.memory().fill(tls, thread::TLS_SIZE, 0).is_err() {
            return Err(result::OUT_OF_MEMORY);
        }
        let thread = Thread::new(NewThread {
            id: self.next_thread_id.fetch_add(1, Ordering::Relaxed),
            process: process.clone(),
            scheduler: self.scheduler.clone(),
            entry,
            arg,
            stack_top,
            tls,
            priority: priority as u8,
            ideal_core: core,
            affinity: 1 << core,
        });
        process.add_thread(thread.clone());
        Ok(thread)
    }

    fn start_process(&self, process: &Arc<Process>) -> HleResult<Arc<Thread>> {
        let params = process.params().clone();
        let stack_size = align_up(params.main_thread_stack_size.max(PAGE_SIZE), PAGE_SIZE);
        let stack_region = process.page_table().regions().stack;
        let stack = process.page_table().map_new_in(
            stack_region,
            stack_size,
            MemoryState::STACK,
            MemoryPermission::READ_WRITE,
        )?;
        process.set_main_stack(stack, stack_size);
        let thread = self.create_thread(
            process,
            process.entry(),
            0,
            stack + stack_size,
            params.main_thread_priority as u32,
            params.main_thread_core as i32,
        )?;
        let handle = process
            .handles()
            .write()
            .add(KObject::Thread(thread.clone()))?;
        thread.context.lock().x[1] = handle as u64;
        process.set_state(ProcessState::Running);
        let mut s = thread.sched.lock();
        self.scheduler.enqueue(&thread, &mut s, false);
        drop(s);
        Ok(thread)
    }

    pub(crate) fn terminate_process(&self, process: &Arc<Process>) {
        process.set_state(ProcessState::Exiting);
        for thread in process.threads() {
            thread.request_termination();
        }
    }

    pub(crate) fn invalidate_code(&self, process: ProcessId, addr: VAddr, size: u64) {
        for (core, list) in self.invalidations.iter().enumerate() {
            list.lock().push((process, addr, size));
            self.scheduler.request_reschedule(core);
        }
    }

    pub(crate) fn take_invalidations(&self, core: usize) -> Vec<CodeRange> {
        let mut list = self.invalidations[core].lock();
        if list.is_empty() {
            Vec::new()
        } else {
            std::mem::take(&mut *list)
        }
    }

    pub(crate) fn new_cpu(&self, process: &Process, core: usize) -> Box<dyn CpuCore> {
        (self.cpu_factory)(CpuConfig {
            memory: process.memory().clone(),
            monitor: self.monitor.clone(),
            clock: self.clock.clone(),
            core_index: core,
            ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
            dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
            accuracy: self.cpu_accuracy,
        })
    }

    /// Called on the core that ran the thread when it exits.
    pub(crate) fn exit_thread(&self, core: usize, thread: &Arc<Thread>) {
        thread.mark_terminated();
        self.scheduler.clear_running(core);
        thread.sync.signal();
        svc::release_thread_resources(thread);
        let process = thread.process();
        if process.remove_thread(thread) {
            process.finalize();
            self.events.on_process_exit(process);
            self.processes.lock().retain(|p| !Arc::ptr_eq(p, process));
            self.process_exits.fetch_add(1, Ordering::Release);
        }
    }
}
