//! Kernel tests driven by a scripted mock `CpuCore`: each guest "program" is a list of Rust
//! steps indexed by the program counter, so the tests exercise the real scheduler, waits and
//! SVC paths without depending on the interpreter.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender};
use lichii_common::settings::CpuAccuracy;
use lichii_common::{ConsoleGeneration, VAddr};
use lichii_cpu::{CpuConfig, CpuCore, GuestClock, HaltReason, InterruptHandle, ThreadContext};
use lichii_loader::npdm::{AddressSpaceType, KernelCapabilities, PoolPartition, ServiceAccess};
use lichii_loader::program::{MainThread, Module, Permission, ProgramImage, Segment};
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use crate::memory::PoolLayout;
use crate::process::Process;
use crate::{Kernel, KernelConfig, KernelEvents};

pub(crate) struct Env<'a> {
    pub memory: &'a AddressSpace,
    pub core: usize,
}

pub(crate) enum Action {
    Next,
    Stay,
    Jump(usize),
    Svc(u32),
}

type Step = Box<dyn Fn(&mut ThreadContext, &Env) -> Action + Send + Sync>;

/// A guest program: step `i` lives at `base + 4 * i`.
#[derive(Default)]
pub(crate) struct Script {
    steps: Vec<Step>,
}

impl Script {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.steps.len()
    }

    pub fn step(
        mut self,
        f: impl Fn(&mut ThreadContext, &Env) -> Action + Send + Sync + 'static,
    ) -> Self {
        self.steps.push(Box::new(f));
        self
    }

    /// Runs host code, then continues.
    pub fn then(self, f: impl Fn(&mut ThreadContext, &Env) + Send + Sync + 'static) -> Self {
        self.step(move |ctx, env| {
            f(ctx, env);
            Action::Next
        })
    }

    /// Sets up registers with `f` and issues `SVC #imm`.
    pub fn svc(
        self,
        imm: u32,
        f: impl Fn(&mut ThreadContext, &Env) + Send + Sync + 'static,
    ) -> Self {
        self.step(move |ctx, env| {
            f(ctx, env);
            Action::Svc(imm)
        })
    }

    /// Spins (interruptibly) until `f` holds.
    pub fn spin_until(
        self,
        f: impl Fn(&ThreadContext, &Env) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.step(move |ctx, env| {
            if f(ctx, env) {
                Action::Next
            } else {
                Action::Stay
            }
        })
    }

    /// Jumps to step `target` while `f` holds.
    pub fn jump_if(
        self,
        target: usize,
        f: impl Fn(&mut ThreadContext, &Env) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.step(move |ctx, env| {
            if f(ctx, env) {
                Action::Jump(target)
            } else {
                Action::Next
            }
        })
    }

    pub fn exit(self) -> Self {
        self.svc(0x0a, |_, _| {})
    }
}

#[derive(Default)]
pub(crate) struct Scripts {
    programs: Mutex<Vec<(VAddr, Arc<Script>)>>,
}

impl Scripts {
    fn find(&self, pc: VAddr) -> (Arc<Script>, usize) {
        let programs = self.programs.lock();
        let (base, script) = programs
            .iter()
            .find(|(base, script)| pc >= *base && pc < base + 4 * script.len() as u64)
            .unwrap_or_else(|| panic!("pc {pc:#x} is outside every script"));
        (script.clone(), ((pc - base) / 4) as usize)
    }
}

struct MockCore {
    scripts: Arc<Scripts>,
    memory: Arc<AddressSpace>,
    core: usize,
    interrupt: InterruptHandle,
}

impl CpuCore for MockCore {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        let (mut script, mut base_index) = self.scripts.find(ctx.pc);
        let mut base = ctx.pc - 4 * base_index as u64;
        let env = Env {
            memory: &self.memory,
            core: self.core,
        };
        loop {
            if self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            let index = ((ctx.pc - base) / 4) as usize;
            if index >= script.len() {
                (script, base_index) = self.scripts.find(ctx.pc);
                base = ctx.pc - 4 * base_index as u64;
                continue;
            }
            match (script.steps[index])(ctx, &env) {
                Action::Next => ctx.pc += 4,
                Action::Stay => std::thread::yield_now(),
                Action::Jump(target) => ctx.pc = base + 4 * target as u64,
                Action::Svc(imm) => {
                    ctx.pc += 4;
                    return HaltReason::SupervisorCall(imm);
                }
            }
        }
    }

    fn step(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        self.run(ctx)
    }

    fn interrupt_handle(&self) -> InterruptHandle {
        self.interrupt.clone()
    }

    fn invalidate_code(&mut self, _addr: VAddr, _size: u64) {}

    fn clear_code_cache(&mut self) {}

    fn clear_exclusive(&mut self) {}
}

struct HostClock(Instant);

impl GuestClock for HostClock {
    fn ticks(&self) -> u64 {
        (self.0.elapsed().as_nanos() * 192 / 10_000) as u64
    }
}

struct ExitEvents(Sender<u64>);

impl KernelEvents for ExitEvents {
    fn on_process_exit(&self, process: &Process) {
        let _ = self.0.send(process.id());
    }
}

pub(crate) const CODE_BASE: VAddr = 0x800_0000;
pub(crate) const CODE_SIZE: u64 = 0x100_0000;

pub(crate) struct TestKernel {
    pub kernel: Kernel,
    pub scripts: Arc<Scripts>,
    exits: Receiver<u64>,
    next_script: AtomicU64,
}

impl TestKernel {
    pub fn new(console: ConsoleGeneration) -> Self {
        Self::with_fastmem(console, false)
    }

    /// A test kernel whose process address spaces get fastmem arenas when `fastmem` is set.
    pub fn with_fastmem(console: ConsoleGeneration, fastmem: bool) -> Self {
        let scripts = Arc::new(Scripts::default());
        let factory_scripts = scripts.clone();
        let (tx, exits) = crossbeam_channel::unbounded();
        let kernel = Kernel::new(KernelConfig {
            console,
            dram_size: 256 << 20,
            pool_layout: Some(PoolLayout {
                application: 192 << 20,
                applet: 16 << 20,
                system_non_secure: 16 << 20,
            }),
            fastmem,
            cpu_accuracy: CpuAccuracy::Auto,
            aslr: false,
            rng_seed: 1,
            enforce_service_access: true,
            clock: Arc::new(HostClock(Instant::now())),
            cpu: Arc::new(move |config: CpuConfig| {
                Box::new(MockCore {
                    scripts: factory_scripts.clone(),
                    memory: config.memory,
                    core: config.core_index,
                    interrupt: InterruptHandle::default(),
                }) as Box<dyn CpuCore>
            }),
            events: Some(Arc::new(ExitEvents(tx))),
        })
        .unwrap();
        Self {
            kernel,
            scripts,
            exits,
            next_script: AtomicU64::new(CODE_BASE),
        }
    }

    /// Registers a program and returns its entry address.
    pub fn program(&self, script: Script) -> VAddr {
        let base = self.next_script.fetch_add(0x1_0000, Ordering::Relaxed);
        self.scripts.programs.lock().push((base, Arc::new(script)));
        base
    }

    /// A 64-bit application with a 39-bit address space and one 16 MiB module of zeroes at the
    /// code base, where the test scripts live.
    pub fn image(&self) -> ProgramImage {
        ProgramImage {
            program_id: 0x0100_0000_0000_1000,
            generation: self.kernel.console(),
            name: "test".into(),
            is_64bit: true,
            address_space: AddressSpaceType::Is39Bit,
            main_thread: MainThread {
                priority: 44,
                core: 0,
                stack_size: 0x10000,
            },
            system_resource_size: 0,
            pool_partition: PoolPartition::Application,
            kernel_capabilities: Vec::new(),
            capabilities: KernelCapabilities::default(),
            services: ServiceAccess {
                client: vec!["*".into()],
                server: vec!["*".into()],
            },
            filesystem_permissions: 0,
            modules: vec![Module {
                name: "main".into(),
                offset: 0,
                image: vec![0; 0x1000],
                size: CODE_SIZE,
                segments: [
                    Segment {
                        offset: 0,
                        size: 0x1000,
                        permission: Permission::ReadExecute,
                    },
                    Segment {
                        offset: 0x1000,
                        size: 0x1000,
                        permission: Permission::Read,
                    },
                    Segment {
                        offset: 0x2000,
                        size: CODE_SIZE - 0x2000,
                        permission: Permission::ReadWrite,
                    },
                ],
                build_id: [0; 32],
                mod0: None,
            }],
            code_size: CODE_SIZE,
            entry_offset: 0,
            romfs: None,
            nacp: None,
            pseudo_device_seed: 0,
            icon: None,
            version: 0,
            add_ons: Vec::new(),
        }
    }

    /// Creates the process of `image` with a 2 MiB heap and starts `main` as its main thread.
    pub fn start(&self, mut image: ProgramImage, main: VAddr) -> Arc<Process> {
        image.entry_offset = main - CODE_BASE;
        let process = self.kernel.create_process(&image).unwrap();
        assert_eq!(process.code_region().start, CODE_BASE);
        process.page_table().set_heap_size(2 << 20).unwrap();
        self.kernel.start_process(&process).unwrap();
        process
    }

    pub fn run_until_exit(&self, timeout: Duration) {
        self.kernel.run().unwrap();
        self.wait_for_exit(timeout);
    }

    /// Waits for a process of an already running kernel to exit, then shuts the kernel down.
    pub fn wait_for_exit(&self, timeout: Duration) {
        let exited = self.exits.recv_timeout(timeout);
        self.kernel.shutdown();
        exited.expect("process did not exit in time");
    }
}

/// Atomic view of a guest word.
pub(crate) fn word(env: &Env, addr: VAddr) -> &'static AtomicU32 {
    let (ptr, _) = env.memory.translate(addr).expect("mapped test word");
    // SAFETY: test words are aligned heap words that outlive the kernel run.
    unsafe { &*(ptr as *const AtomicU32) }
}

mod ipc;
mod memory;
mod scheduling;
mod sync;
