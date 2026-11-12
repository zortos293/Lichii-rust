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
pub mod svc;
mod sync;
pub mod thread;
mod timer;
mod util;

#[cfg(test)]
mod tests;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread::JoinHandle;

use lichii_common::{ConsoleGeneration, HleResult, PAGE_SIZE, VAddr, align_up};
use lichii_cpu::{CpuConfig, CpuCore, GuestClock};
use lichii_memory::{
    AddressSpace, ExclusiveMonitor, MemoryPermission, PhysicalMemory, PhysicalMemoryError,
};
use parking_lot::Mutex;

use crate::arbiter::Arbiter;
use crate::memory::{MemoryManager, MemoryState, PoolLayout, Region, Regions};
use crate::object::KObject;
use crate::process::{NewProcess, Process, ProcessId, ProcessParams, ProcessState};
use crate::resource_limit::{LimitableResource, ResourceLimit};
use crate::scheduler::Scheduler;
use crate::thread::{LOWEST_PRIORITY, NewThread, Thread};
use crate::timer::Timer;
use crate::util::Rng;

pub use crate::event::Event;
pub use crate::memory::Pool;

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
    events: Arc<dyn KernelEvents>,
    rng: Mutex<Rng>,
    next_process_id: AtomicU64,
    next_thread_id: AtomicU64,
    processes: Mutex<Vec<Arc<Process>>>,
    invalidations: Box<[Mutex<Vec<CodeRange>>]>,
    host_threads: Mutex<Vec<JoinHandle<()>>>,
    application_limit: Arc<ResourceLimit>,
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
        Ok(Self(Arc::new(KernelInner {
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
            events: config.events.unwrap_or_else(|| Arc::new(LogEvents)),
            rng: Mutex::new(Rng::new(config.rng_seed)),
            next_process_id: AtomicU64::new(0x51),
            next_thread_id: AtomicU64::new(1),
            processes: Mutex::new(Vec::new()),
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

    /// Creates a process whose code occupies `code` (normally right after the start of the
    /// code region, see [`Kernel::default_code_address`]). Load its modules with
    /// [`Process::load_module`], then call [`Kernel::start_process`].
    pub fn create_process(&self, params: ProcessParams, code: Region) -> HleResult<Arc<Process>> {
        self.0.create_process(params, code)
    }

    /// Where Horizon puts the code of a process with this address-space width (plus ASLR slide
    /// when `aslr` is set).
    pub fn default_code_address(&self, address_space_bits: u32, aslr: bool) -> VAddr {
        let base = match address_space_bits {
            32 => 0x20_0000,
            _ => 0x800_0000,
        };
        if aslr && address_space_bits == 39 {
            base + self.0.rng.lock().below(0x40) * 0x20_0000
        } else {
            base
        }
    }

    /// Creates the main thread (stack in the stack region, `x1` = its handle) and makes it
    /// runnable.
    pub fn start_process(&self, process: &Arc<Process>, entry: VAddr) -> HleResult<Arc<Thread>> {
        self.0.start_process(process, entry)
    }

    /// Starts the core threads and the timer thread.
    pub fn run(&self) -> Result<(), KernelError> {
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

    /// Stops every core and kernel thread and releases all processes.
    pub fn shutdown(&self) {
        self.0.scheduler.shutdown();
        self.0.timer.shutdown();
        let threads = std::mem::take(&mut *self.0.host_threads.lock());
        for thread in threads {
            let _ = thread.join();
        }
        self.0.scheduler.clear();
        let processes = std::mem::take(&mut *self.0.processes.lock());
        for process in processes {
            process.threads.lock().clear();
            if process.state() != ProcessState::Exited {
                process.finalize();
            }
        }
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

    fn create_process(
        self: &Arc<Self>,
        params: ProcessParams,
        code: Region,
    ) -> HleResult<Arc<Process>> {
        let code = Region {
            start: code.start,
            end: align_up(code.end, PAGE_SIZE),
        };
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

    fn start_process(&self, process: &Arc<Process>, entry: VAddr) -> HleResult<Arc<Thread>> {
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
            entry,
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
        })
    }

    /// Called on the core that ran the thread when it exits.
    pub(crate) fn exit_thread(&self, core: usize, thread: &Arc<Thread>) {
        {
            let mut s = thread.sched.lock();
            s.state = thread::ThreadState::Terminated;
            s.wait = 0;
        }
        self.scheduler.clear_running(core);
        thread.sync.signal();
        svc::release_thread_resources(thread);
        let process = thread.process();
        if process.remove_thread(thread) {
            process.finalize();
            self.events.on_process_exit(process);
            self.processes.lock().retain(|p| !Arc::ptr_eq(p, process));
        }
    }
}
