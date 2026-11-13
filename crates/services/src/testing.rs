//! Drives services through the real kernel the way a title does: a guest thread on a scripted
//! core writes HIPC messages to its TLS and calls `SendSyncRequest`.

use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use bytemuck::Pod;
use lichii_common::settings::Settings;
use lichii_common::{ConsoleGeneration, HleResult, VAddr};
use lichii_cpu::{CpuConfig, CpuCore, GuestClock, HaltReason, InterruptHandle, ThreadContext};
use lichii_gpu::host1x::Host1x;
use lichii_kernel::{Kernel, KernelConfig};
use lichii_loader::npdm::{AddressSpaceType, KernelCapabilities, PoolPartition, ServiceAccess};
use lichii_loader::program::{MainThread, Module, Permission, ProgramImage, Segment};
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use crate::{ServiceContext, ServiceParts};

type Job = Box<dyn FnOnce(&mut ThreadContext, &AddressSpace) -> Option<u32> + Send>;

const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
const CODE_SIZE: u64 = 0x100_0000;
const SCRATCH: u64 = 0x10_0000;
const TIMEOUT: Duration = Duration::from_secs(10);

struct MockCore {
    jobs: Arc<Mutex<mpsc::Receiver<Job>>>,
    memory: Arc<AddressSpace>,
    interrupt: InterruptHandle,
}

impl CpuCore for MockCore {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        loop {
            if self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            let job = self.jobs.lock().recv_timeout(Duration::from_millis(1));
            match job {
                Ok(job) => {
                    if let Some(svc) = job(ctx, &self.memory) {
                        return HaltReason::SupervisorCall(svc);
                    }
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    std::thread::sleep(Duration::from_millis(1))
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
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

/// A request as a title encodes it.
#[derive(Clone, Default)]
pub(crate) struct Request {
    command: u32,
    data: Vec<u8>,
    pid: bool,
    a: Vec<(u64, u64)>,
    b: Vec<(u64, u64)>,
    x: Vec<(u64, u64)>,
}

impl Request {
    pub fn new(command: u32) -> Self {
        Self {
            command,
            ..Self::default()
        }
    }

    /// Appends a parameter, naturally aligned like a CMIF parameter struct.
    pub fn arg<T: Pod>(mut self, value: T) -> Self {
        let start = self
            .data
            .len()
            .next_multiple_of(std::mem::align_of::<T>().min(8));
        self.data.resize(start, 0);
        self.data.extend_from_slice(bytemuck::bytes_of(&value));
        self
    }

    pub fn pid(mut self) -> Self {
        self.pid = true;
        self
    }

    pub fn a(mut self, address: u64, size: u64) -> Self {
        self.a.push((address, size));
        self
    }

    pub fn b(mut self, address: u64, size: u64) -> Self {
        self.b.push((address, size));
        self
    }

    pub fn x(mut self, address: u64, size: u64) -> Self {
        self.x.push((address, size));
        self
    }

    fn encode(&self) -> Vec<u32> {
        let mut w = vec![
            4 | (self.x.len() as u32) << 16
                | (self.a.len() as u32) << 20
                | (self.b.len() as u32) << 24,
            0,
        ];
        if self.pid {
            w.extend([1, 0, 0]);
        }
        for &(addr, size) in &self.x {
            w.push(
                (size as u32) << 16
                    | ((addr >> 36) as u32 & 0x7) << 6
                    | ((addr >> 32) as u32 & 0xf) << 12,
            );
            w.push(addr as u32);
        }
        for &(addr, size) in self.a.iter().chain(&self.b) {
            w.push(size as u32);
            w.push(addr as u32);
            w.push(
                ((addr >> 32) as u32 & 0xf) << 28
                    | ((addr >> 36) as u32 & 0x7) << 2
                    | ((size >> 32) as u32 & 0xf) << 24,
            );
        }
        let raw_start = w.len();
        while !w.len().is_multiple_of(4) {
            w.push(0);
        }
        w.extend([SFCI, 0, self.command, 0]);
        let mut data = self.data.clone();
        data.resize(data.len().next_multiple_of(4), 0);
        w.extend(
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c)),
        );
        w[1] = (w.len() - raw_start) as u32 | u32::from(self.pid) << 31;
        w
    }
}

/// A decoded CMIF reply.
#[derive(Debug)]
pub(crate) struct Reply {
    pub result: u32,
    pub data: Vec<u8>,
    pub copies: Vec<u32>,
    pub moves: Vec<u32>,
}

impl Reply {
    fn decode(words: &[u32]) -> Self {
        let raw = (words[1] & 0x3ff) as usize;
        let mut at = 2;
        let (mut copies, mut moves) = (Vec::new(), Vec::new());
        if words[1] >> 31 != 0 {
            let descriptor = words[2];
            at = 3 + if descriptor & 1 != 0 { 2 } else { 0 };
            let copy_count = ((descriptor >> 1) & 0xf) as usize;
            let move_count = ((descriptor >> 5) & 0xf) as usize;
            copies.extend_from_slice(&words[at..at + copy_count]);
            moves.extend_from_slice(&words[at + copy_count..at + copy_count + move_count]);
            at += copy_count + move_count;
        }
        let header = at.next_multiple_of(4);
        let data_words = raw - 8;
        let data =
            bytemuck::cast_slice::<u32, u8>(&words[header + 4..header + 4 + data_words]).to_vec();
        Self {
            result: words[header + 2],
            data,
            copies,
            moves,
        }
    }

    /// Asserts success and returns the reply.
    pub fn ok(self) -> Self {
        assert_eq!(self.result, 0, "request failed with {:#x}", self.result);
        self
    }

    /// The value at byte `offset` of the reply data.
    pub fn get<T: Pod>(&self, offset: usize) -> T {
        bytemuck::pod_read_unaligned(&self.data[offset..offset + std::mem::size_of::<T>()])
    }

    /// The first move handle: a returned interface.
    pub fn interface(&self) -> u32 {
        self.moves[0]
    }
}

/// A running kernel with one guest thread that sends requests on behalf of the test.
pub(crate) struct Guest {
    kernel: Kernel,
    jobs: Option<mpsc::Sender<Job>>,
    pub context: Arc<ServiceContext>,
    scratch: VAddr,
    next: Mutex<u64>,
    sm: u32,
}

impl Guest {
    /// Boots a kernel for `console`, registers services with `register` and connects to `sm:`.
    pub fn new(
        console: ConsoleGeneration,
        settings: Settings,
        register: impl FnOnce(&Arc<ServiceContext>) -> HleResult<()>,
    ) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let kernel = Kernel::new(KernelConfig {
            console,
            dram_size: 256 << 20,
            pool_layout: Some(lichii_kernel::memory::PoolLayout {
                application: 192 << 20,
                applet: 16 << 20,
                system_non_secure: 16 << 20,
            }),
            fastmem: false,
            aslr: false,
            rng_seed: 1,
            clock: Arc::new(HostClock(Instant::now())),
            cpu: Arc::new(move |config: CpuConfig| {
                Box::new(MockCore {
                    jobs: rx.clone(),
                    memory: config.memory,
                    interrupt: InterruptHandle::default(),
                }) as Box<dyn CpuCore>
            }),
            events: None,
        })
        .unwrap();
        let program = Arc::new(image(console));
        let process = kernel.create_process(&program).unwrap();
        let context = ServiceContext::new(ServiceParts {
            kernel: kernel.clone(),
            settings: Arc::new(settings),
            program,
            process: process.clone(),
            host1x: Host1x::new(kernel.physical_memory().clone(), 34),
            nx2_memory: None,
            npad: Arc::default(),
            perf: Arc::default(),
        });
        register(&context).unwrap();
        kernel.start_process(&process).unwrap();
        kernel.run().unwrap();
        let mut guest = Self {
            kernel,
            jobs: Some(tx),
            context,
            scratch: process.code_region().start + SCRATCH,
            next: Mutex::new(0x1000),
            sm: 0,
        };
        let name = guest.scratch;
        guest.write(name, b"sm:\0");
        let regs = guest.svc(0x1f, &[(1, name)]);
        assert_eq!(regs[0], 0, "ConnectToNamedPort(sm:)");
        guest.sm = regs[1] as u32;
        guest.call(guest.sm, &Request::new(0).arg(0u64).pid()).ok();
        guest
    }

    /// Runs `f` on the guest thread and returns its result.
    fn on_guest<R: Send + 'static>(
        &self,
        svc: Option<u32>,
        f: impl FnOnce(&mut ThreadContext, &AddressSpace) -> R + Send + 'static,
    ) -> R {
        let (tx, rx) = mpsc::channel();
        let jobs = self.jobs.as_ref().unwrap();
        jobs.send(Box::new(move |ctx, memory| {
            tx.send(f(ctx, memory)).unwrap();
            svc
        }))
        .unwrap();
        rx.recv_timeout(TIMEOUT).expect("guest thread stalled")
    }

    /// Issues `SVC #imm` with registers `regs` and returns x0..x7 afterwards.
    pub fn svc(&self, imm: u32, regs: &[(usize, u64)]) -> [u64; 8] {
        let regs = regs.to_vec();
        self.on_guest(Some(imm), move |ctx, _| {
            for (index, value) in regs {
                ctx.x[index] = value;
            }
        });
        self.on_guest(None, |ctx, _| std::array::from_fn(|i| ctx.x[i]))
    }

    pub fn write(&self, address: VAddr, data: &[u8]) {
        let data = data.to_vec();
        self.on_guest(None, move |_, memory| {
            memory.write_bytes(address, &data).unwrap()
        });
    }

    pub fn read(&self, address: VAddr, len: usize) -> Vec<u8> {
        self.on_guest(None, move |_, memory| {
            let mut data = vec![0; len];
            memory.read_bytes(address, &mut data).unwrap();
            data
        })
    }

    pub fn read_value<T: Pod + Send>(&self, address: VAddr) -> T {
        bytemuck::pod_read_unaligned(&self.read(address, std::mem::size_of::<T>()))
    }

    /// A zeroed guest buffer of `size` bytes.
    pub fn buffer(&self, size: u64) -> VAddr {
        let mut next = self.next.lock();
        let address = self.scratch + *next;
        *next = (*next + size).next_multiple_of(0x1000);
        assert!(*next < CODE_SIZE - SCRATCH, "test scratch exhausted");
        self.write(address, &vec![0; size as usize]);
        address
    }

    /// A guest buffer holding `data`.
    pub fn buffer_with(&self, data: &[u8]) -> VAddr {
        let address = self.buffer(data.len() as u64);
        self.write(address, data);
        address
    }

    /// Sends `request` on `handle` and waits for the reply.
    pub fn call(&self, handle: u32, request: &Request) -> Reply {
        let words = request.encode();
        self.on_guest(Some(0x21), move |ctx, memory| {
            memory
                .write_bytes(ctx.tpidrro_el0, bytemuck::cast_slice(&words))
                .unwrap();
            ctx.x[0] = u64::from(handle);
        });
        let (status, words) = self.on_guest(None, |ctx, memory| {
            let mut words = vec![0u32; 64];
            memory
                .read_bytes(ctx.tpidrro_el0, bytemuck::cast_slice_mut(&mut words))
                .unwrap();
            (ctx.x[0], words)
        });
        assert_eq!(status, 0, "SendSyncRequest failed");
        Reply::decode(&words)
    }

    /// Opens a session to service `name`.
    pub fn connect(&self, name: &str) -> u32 {
        let mut raw = [0u8; 8];
        raw[..name.len()].copy_from_slice(name.as_bytes());
        let reply = self
            .call(self.sm, &Request::new(1).arg(u64::from_le_bytes(raw)))
            .ok();
        reply.interface()
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        self.kernel.shutdown();
        self.jobs = None;
    }
}

/// Default settings, as a fresh install has them.
pub(crate) fn settings() -> Settings {
    Settings::default()
}

fn image(console: ConsoleGeneration) -> ProgramImage {
    ProgramImage {
        program_id: match console {
            ConsoleGeneration::Switch1 => 0x0100_0000_0000_1000,
            ConsoleGeneration::Switch2 => 0x0400_0000_0000_1000,
        },
        generation: console,
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
        services: ServiceAccess::default(),
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
        add_ons: Vec::new(),
    }
}
