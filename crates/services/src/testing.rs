//! A guest that drives services through the real kernel: a mock CPU core runs one guest thread
//! whose every step is a supervisor call the test asks for, so requests take the same IPC path
//! (TLS message, `SendSyncRequest`, service thread, reply) as a title's.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_common::{ConsoleGeneration, ResultCode, VAddr};
use lichii_cpu::{CpuConfig, CpuCore, GuestClock, HaltReason, InterruptHandle, ThreadContext};
use lichii_fs::nacp::Nacp;
use lichii_gpu::host1x::Host1x;
use lichii_input::NpadHub;
use lichii_kernel::memory::PoolLayout;
use lichii_kernel::process::Process;
use lichii_kernel::{Kernel, KernelConfig};
use lichii_loader::npdm::{AddressSpaceType, KernelCapabilities, PoolPartition, ServiceAccess};
use lichii_loader::program::{MainThread, Module, Permission, ProgramImage, Segment};
use lichii_memory::AddressSpace;

use crate::{ServiceContext, ServiceParts};

const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
const SFCO: u32 = u32::from_le_bytes(*b"SFCO");
const CODE_SIZE: u64 = 0x10_0000;
const HEAP_SIZE: u64 = 2 << 20;
/// Program id of [`image`].
pub const PROGRAM_ID: u64 = 0x0100_0000_0001_0000;

/// A 64-bit application with one zero-filled module; tests add a NACP or add-ons.
pub fn image() -> ProgramImage {
    ProgramImage {
        program_id: PROGRAM_ID,
        generation: ConsoleGeneration::Switch1,
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

/// A NACP with `name` in every language slot, the supported-language mask and the add-on
/// base id.
pub fn nacp(name: &str, supported_languages: u32, add_on_base: u64) -> Nacp {
    let mut raw = vec![0u8; 0x4000];
    for entry in raw[..0x3000].chunks_mut(0x300) {
        entry[..name.len()].copy_from_slice(name.as_bytes());
        entry[0x200..0x206].copy_from_slice(b"Lichii");
    }
    raw[0x302c..0x3030].copy_from_slice(&supported_languages.to_le_bytes());
    raw[0x3060..0x3065].copy_from_slice(b"1.2.3");
    raw[0x3070..0x3078].copy_from_slice(&add_on_base.to_le_bytes());
    Nacp::parse(&raw).unwrap()
}

/// A request as a title encodes it.
#[derive(Clone, Default)]
pub struct Request {
    command: u32,
    pid: bool,
    copy: Vec<u32>,
    a: Vec<(u64, u64)>,
    b: Vec<(u64, u64)>,
    data: Vec<u8>,
}

impl Request {
    pub fn new(command: u32) -> Self {
        Self {
            command,
            ..Self::default()
        }
    }

    /// Appends a value, naturally aligned in the input data.
    pub fn arg<T: bytemuck::Pod>(mut self, value: T) -> Self {
        let align = std::mem::align_of::<T>().min(8);
        self.data.resize(self.data.len().next_multiple_of(align), 0);
        self.data.extend_from_slice(bytemuck::bytes_of(&value));
        self
    }

    pub fn pid(mut self) -> Self {
        self.pid = true;
        self
    }

    pub fn copy(mut self, handle: u32) -> Self {
        self.copy.push(handle);
        self
    }

    /// Adds an A (input) buffer.
    pub fn input(mut self, address: VAddr, size: u64) -> Self {
        self.a.push((address, size));
        self
    }

    /// Adds a B (output) buffer.
    pub fn output(mut self, address: VAddr, size: u64) -> Self {
        self.b.push((address, size));
        self
    }

    fn encode(&self) -> Vec<u32> {
        let mut w = vec![
            4 | (self.a.len() as u32) << 20 | (self.b.len() as u32) << 24,
            0,
        ];
        let has_handles = self.pid || !self.copy.is_empty();
        if has_handles {
            w.push(u32::from(self.pid) | (self.copy.len() as u32) << 1);
            if self.pid {
                w.extend([0, 0]);
            }
            w.extend(&self.copy);
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
        w.resize(w.len().next_multiple_of(4), 0);
        w.extend([SFCI, 0, self.command, 0]);
        let mut data = self.data.clone();
        data.resize(data.len().next_multiple_of(4), 0);
        w.extend(
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c)),
        );
        w[1] = (w.len() - raw_start) as u32 | u32::from(has_handles) << 31;
        w
    }
}

/// A decoded CMIF reply.
#[derive(Debug)]
pub struct Reply {
    pub result: ResultCode,
    pub copy: Vec<u32>,
    pub moves: Vec<u32>,
    data: Vec<u8>,
}

impl Reply {
    fn decode(words: &[u32]) -> Self {
        let mut at = 2;
        let (mut copy, mut moves) = (Vec::new(), Vec::new());
        if words[1] >> 31 != 0 {
            let desc = words[2];
            at = 3 + if desc & 1 != 0 { 2 } else { 0 };
            let copies = ((desc >> 1) & 0xf) as usize;
            let move_count = ((desc >> 5) & 0xf) as usize;
            copy.extend_from_slice(&words[at..at + copies]);
            moves.extend_from_slice(&words[at + copies..at + copies + move_count]);
            at += copies + move_count;
        }
        let end = (at + (words[1] & 0x3ff) as usize).min(words.len());
        let start = at.next_multiple_of(4);
        assert_eq!(
            words[start], SFCO,
            "reply without a CMIF header: {words:x?}"
        );
        Self {
            result: ResultCode(words[start + 2]),
            copy,
            moves,
            data: bytemuck::cast_slice::<u32, u8>(&words[start + 4..end.max(start + 4)]).to_vec(),
        }
    }

    pub fn ok(self) -> Self {
        assert_eq!(self.result, ResultCode::SUCCESS, "request failed");
        self
    }

    /// The value at byte `offset` of the reply data.
    pub fn at<T: bytemuck::Pod>(&self, offset: usize) -> T {
        bytemuck::pod_read_unaligned(&self.data[offset..offset + std::mem::size_of::<T>()])
    }

    /// The one sub-interface (move handle) of the reply.
    pub fn interface(&self) -> u32 {
        assert_eq!(self.moves.len(), 1, "expected one interface");
        self.moves[0]
    }

    /// The one copy handle of the reply.
    pub fn handle(&self) -> u32 {
        assert_eq!(self.copy.len(), 1, "expected one copy handle");
        self.copy[0]
    }
}

enum Call {
    Svc {
        imm: u32,
        x: [u64; 4],
        tls: Vec<u32>,
    },
    Exit,
}

struct Done {
    x: [u64; 2],
    tls: Vec<u32>,
}

struct Shared {
    entry: AtomicU64,
    calls: Mutex<Receiver<Call>>,
    done: Mutex<Sender<Done>>,
}

struct GuestCore {
    shared: Arc<Shared>,
    memory: Arc<AddressSpace>,
    interrupt: InterruptHandle,
}

impl CpuCore for GuestCore {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        let entry = self.shared.entry.load(Ordering::Acquire);
        if ctx.pc == entry + 4 {
            let mut tls = vec![0u32; 64];
            self.memory
                .read_bytes(ctx.tpidrro_el0, bytemuck::cast_slice_mut(&mut tls))
                .unwrap();
            let done = Done {
                x: [ctx.x[0], ctx.x[1]],
                tls,
            };
            self.shared.done.lock().unwrap().send(done).unwrap();
            ctx.pc = entry;
        }
        loop {
            if self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            let call = self
                .shared
                .calls
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_millis(1));
            match call {
                Ok(Call::Svc { imm, x, tls }) => {
                    self.memory
                        .write_bytes(ctx.tpidrro_el0, bytemuck::cast_slice(&tls))
                        .unwrap();
                    ctx.x[..4].copy_from_slice(&x);
                    ctx.pc = entry + 4;
                    return HaltReason::SupervisorCall(imm);
                }
                Ok(Call::Exit) | Err(RecvTimeoutError::Disconnected) => {
                    ctx.pc = entry + 8;
                    return HaltReason::SupervisorCall(0x0a);
                }
                Err(RecvTimeoutError::Timeout) => {}
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

/// The title's process on a running kernel, with the services `register` added.
pub struct Guest {
    pub kernel: Kernel,
    pub process: Arc<Process>,
    _context: Arc<ServiceContext>,
    /// 1 MiB of heap for request buffers.
    pub heap: VAddr,
    calls: Sender<Call>,
    done: Receiver<Done>,
    sm: u32,
}

impl Guest {
    pub fn new(
        program: ProgramImage,
        settings: Settings,
        register: impl FnOnce(&Arc<ServiceContext>),
    ) -> Self {
        let (calls, call_rx) = mpsc::channel();
        let (done_tx, done) = mpsc::channel();
        let shared = Arc::new(Shared {
            entry: AtomicU64::new(0),
            calls: Mutex::new(call_rx),
            done: Mutex::new(done_tx),
        });
        let factory_shared = shared.clone();
        let kernel = Kernel::new(KernelConfig {
            console: program.generation,
            dram_size: 256 << 20,
            pool_layout: Some(PoolLayout {
                application: 192 << 20,
                applet: 16 << 20,
                system_non_secure: 16 << 20,
            }),
            fastmem: false,
            aslr: false,
            rng_seed: 1,
            clock: Arc::new(HostClock(Instant::now())),
            cpu: Arc::new(move |config: CpuConfig| {
                Box::new(GuestCore {
                    shared: factory_shared.clone(),
                    memory: config.memory,
                    interrupt: InterruptHandle::default(),
                }) as Box<dyn CpuCore>
            }),
            events: None,
        })
        .unwrap();
        let process = kernel.create_process(&program).unwrap();
        let heap = process.page_table().set_heap_size(HEAP_SIZE).unwrap();
        shared
            .entry
            .store(process.code_region().start, Ordering::Release);
        let context = ServiceContext::new(ServiceParts {
            host1x: Host1x::new(kernel.physical_memory().clone(), 34),
            kernel: kernel.clone(),
            settings: Arc::new(settings),
            program: Arc::new(program),
            process: process.clone(),
            nx2_memory: None,
            npad: Arc::new(NpadHub::default()),
            perf: Arc::new(PerfStats::default()),
        });
        register(&context);
        kernel.start_process(&process).unwrap();
        kernel.run().unwrap();
        let mut guest = Self {
            kernel,
            process,
            _context: context,
            heap,
            calls,
            done,
            sm: 0,
        };
        let name = heap + 0x10_0000;
        guest.write(name, b"sm:\0");
        let (result, handle) = guest.svc(0x1f, [0, name, 0, 0]);
        assert_eq!(result, 0, "ConnectToNamedPort(sm:)");
        guest.sm = handle as u32;
        guest.call(guest.sm, Request::new(0).pid().arg(0u64)).ok();
        guest
    }

    /// Opens a session to service `name`.
    pub fn service(&self, name: &str) -> u32 {
        let mut raw = [0u8; 8];
        raw[..name.len()].copy_from_slice(name.as_bytes());
        self.call(self.sm, Request::new(1).arg(u64::from_le_bytes(raw)))
            .ok()
            .interface()
    }

    /// Issues `SVC #imm` with x0..x3 and returns x0 and x1 after it.
    pub fn svc(&self, imm: u32, x: [u64; 4]) -> (u64, u64) {
        let done = self.exchange(imm, x, Vec::new());
        (done.x[0], done.x[1])
    }

    /// Sends `request` on session `handle` and decodes the reply.
    pub fn call(&self, handle: u32, request: Request) -> Reply {
        let done = self.exchange(0x21, [u64::from(handle), 0, 0, 0], request.encode());
        assert_eq!(done.x[0], 0, "SendSyncRequest failed");
        Reply::decode(&done.tls)
    }

    fn exchange(&self, imm: u32, x: [u64; 4], tls: Vec<u32>) -> Done {
        self.calls.send(Call::Svc { imm, x, tls }).unwrap();
        self.done
            .recv_timeout(Duration::from_secs(10))
            .expect("guest call timed out")
    }

    pub fn write(&self, address: VAddr, data: &[u8]) {
        self.process.memory().write_bytes(address, data).unwrap();
    }

    pub fn read(&self, address: VAddr, len: usize) -> Vec<u8> {
        let mut data = vec![0; len];
        self.process
            .memory()
            .read_bytes(address, &mut data)
            .unwrap();
        data
    }

    /// Whether the readable event behind `handle` is signaled.
    pub fn is_signaled(&self, handle: u32) -> bool {
        let list = self.heap + 0x10_0100;
        self.write(list, &handle.to_le_bytes());
        self.svc(0x18, [0, list, 1, 0]).0 == 0
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        let _ = self.calls.send(Call::Exit);
        self.kernel.shutdown();
    }
}
