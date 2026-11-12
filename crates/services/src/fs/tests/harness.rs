//! A guest thread on a mock core that a test drives one SVC at a time, so services run through
//! the real kernel path: `sm:`, sessions, HIPC parsing and replies, guest buffers.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use bytemuck::Pod;
use lichii_common::settings::Settings;
use lichii_common::{ConsoleGeneration, ResultCode, VAddr};
use lichii_cpu::{CpuConfig, CpuCore, GuestClock, HaltReason, InterruptHandle, ThreadContext};
use lichii_gpu::host1x::Host1x;
use lichii_input::NpadHub;
use lichii_kernel::memory::PoolLayout;
use lichii_kernel::{Kernel, KernelConfig};
use lichii_loader::npdm::{AddressSpaceType, KernelCapabilities, PoolPartition, ServiceAccess};
use lichii_loader::program::{MainThread, Module, Permission, ProgramImage, Segment};
use lichii_memory::AddressSpace;
use parking_lot::{Condvar, Mutex};

use crate::{ServiceContext, ServiceParts};

const SFCI: u32 = u32::from_le_bytes(*b"SFCI");
const SFCO: u32 = u32::from_le_bytes(*b"SFCO");
const PENDING: u64 = 0x5ead;
const CODE_SIZE: u64 = 0x10_0000;
const TIMEOUT: Duration = Duration::from_secs(20);

pub(crate) const SWITCH1_PROGRAM: u64 = 0x0100_0000_0000_1000;
pub(crate) const SWITCH2_PROGRAM: u64 = 0x0400_0000_0000_1000;

struct Op {
    imm: u32,
    x: [u64; 8],
    message: Vec<u32>,
}

struct Reply {
    x: [u64; 8],
    words: [u32; 64],
}

#[derive(Default)]
struct Mail {
    op: Option<Op>,
    reply: Option<Reply>,
}

#[derive(Default)]
struct Mailbox {
    mail: Mutex<Mail>,
    changed: Condvar,
}

/// Runs the one guest thread: each `run` first reports the result of the SVC it issued last,
/// then waits for the test's next SVC.
struct MockCore {
    mailbox: Arc<Mailbox>,
    memory: Arc<AddressSpace>,
    interrupt: InterruptHandle,
}

impl CpuCore for MockCore {
    fn run(&mut self, ctx: &mut ThreadContext) -> HaltReason {
        if ctx.x[28] == PENDING {
            ctx.x[28] = 0;
            let mut words = [0u32; 64];
            self.memory
                .read_bytes(ctx.tpidrro_el0, bytemuck::cast_slice_mut(&mut words))
                .unwrap();
            let mut x = [0; 8];
            x.copy_from_slice(&ctx.x[..8]);
            self.mailbox.mail.lock().reply = Some(Reply { x, words });
            self.mailbox.changed.notify_all();
        }
        let mut mail = self.mailbox.mail.lock();
        loop {
            if self.interrupt.take() {
                return HaltReason::Interrupted;
            }
            if let Some(op) = mail.op.take() {
                drop(mail);
                self.memory
                    .write_bytes(ctx.tpidrro_el0, bytemuck::cast_slice(&op.message))
                    .unwrap();
                ctx.x[..8].copy_from_slice(&op.x);
                ctx.x[28] = PENDING;
                ctx.pc += 4;
                return HaltReason::SupervisorCall(op.imm);
            }
            self.mailbox
                .changed
                .wait_for(&mut mail, Duration::from_millis(1));
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

/// A CMIF request as a guest encodes it.
#[derive(Default)]
pub(crate) struct Request {
    command: u32,
    pid: bool,
    data: Vec<u8>,
    x: Vec<(u64, u64)>,
    a: Vec<(u64, u64)>,
    b: Vec<(u64, u64)>,
}

impl Request {
    pub fn new(command: u32) -> Self {
        Self {
            command,
            ..Self::default()
        }
    }

    pub fn pid(mut self) -> Self {
        self.pid = true;
        self
    }

    /// Appends a parameter, naturally aligned from the start of the data.
    pub fn arg<T: Pod>(mut self, value: T) -> Self {
        let start = self.data.len().next_multiple_of(align_of::<T>().min(8));
        self.data.resize(start, 0);
        self.data.extend_from_slice(bytemuck::bytes_of(&value));
        self
    }

    /// A pointer (X) buffer.
    pub fn x(mut self, buffer: (u64, u64)) -> Self {
        self.x.push(buffer);
        self
    }

    /// An input map (A) buffer.
    pub fn a(mut self, buffer: (u64, u64)) -> Self {
        self.a.push(buffer);
        self
    }

    /// An output map (B) buffer.
    pub fn b(mut self, buffer: (u64, u64)) -> Self {
        self.b.push(buffer);
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
        for (i, &(addr, size)) in self.x.iter().enumerate() {
            w.push(
                (size as u32) << 16
                    | ((addr >> 32) as u32 & 0xf) << 12
                    | ((addr >> 36) as u32 & 0x7) << 6
                    | i as u32 & 0x3f,
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
        assert!(w.len() <= 64, "request does not fit in TLS");
        w
    }
}

/// A parsed CMIF reply.
pub(crate) struct Response {
    pub result: ResultCode,
    data: Vec<u8>,
    pub moves: Vec<u32>,
}

impl Response {
    fn parse(words: &[u32; 64]) -> Self {
        let w1 = words[1];
        let mut i = 2;
        let mut moves = Vec::new();
        if w1 >> 31 != 0 {
            let descriptor = words[2];
            i = 3 + ((descriptor >> 1) & 0xf) as usize;
            for _ in 0..(descriptor >> 5) & 0xf {
                moves.push(words[i]);
                i += 1;
            }
        }
        let data_words = (w1 & 0x3ff) as usize - 8;
        let j = i.next_multiple_of(4);
        assert_eq!(words[j], SFCO, "not a CMIF reply");
        Self {
            result: ResultCode(words[j + 2]),
            data: bytemuck::cast_slice(&words[j + 4..j + 4 + data_words]).to_vec(),
            moves,
        }
    }

    /// Asserts success.
    #[track_caller]
    pub fn ok(self) -> Self {
        assert_eq!(self.result, ResultCode::SUCCESS);
        self
    }

    /// The reply value at byte `offset` of the data.
    #[track_caller]
    pub fn get<T: Pod>(&self, offset: usize) -> T {
        bytemuck::pod_read_unaligned(&self.data[offset..offset + size_of::<T>()])
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The first move handle (a sub-interface session).
    #[track_caller]
    pub fn handle(&self) -> u32 {
        assert_eq!(self.result, ResultCode::SUCCESS);
        self.moves[0]
    }
}

static NEXT_ROOT: AtomicU32 = AtomicU32::new(0);

/// A kernel with the fs services registered and one guest thread the test drives.
pub(crate) struct Guest {
    kernel: Kernel,
    pub memory: Arc<AddressSpace>,
    pub heap: u64,
    /// Host directory holding `nand` and `sdmc`.
    pub root: PathBuf,
    mailbox: Arc<Mailbox>,
    sm: u32,
}

impl Guest {
    pub fn new(console: ConsoleGeneration, configure: impl FnOnce(&mut ProgramImage)) -> Self {
        let root = std::env::temp_dir().join(format!(
            "lichii-fsp-{}-{}",
            std::process::id(),
            NEXT_ROOT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        let mut settings = Settings::default();
        settings.paths.nand_dir = Some(root.join("nand"));
        settings.paths.sdmc_dir = Some(root.join("sdmc"));
        settings.paths.keys_dir = Some(root.join("keys"));

        let mailbox = Arc::new(Mailbox::default());
        let cores = mailbox.clone();
        let kernel = Kernel::new(KernelConfig {
            console,
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
                Box::new(MockCore {
                    mailbox: cores.clone(),
                    memory: config.memory,
                    interrupt: InterruptHandle::default(),
                }) as Box<dyn CpuCore>
            }),
            events: None,
        })
        .unwrap();

        let mut image = image(console);
        configure(&mut image);
        let process = kernel.create_process(&image).unwrap();
        let heap = process.page_table().set_heap_size(2 << 20).unwrap();
        kernel.start_process(&process).unwrap();
        let memory = process.memory().clone();
        let context = ServiceContext::new(ServiceParts {
            kernel: kernel.clone(),
            settings: Arc::new(settings),
            program: Arc::new(image),
            process,
            host1x: Host1x::new(kernel.physical_memory().clone(), 34),
            nx2_memory: None,
            npad: Arc::new(NpadHub::default()),
            perf: Arc::default(),
        });
        crate::fs::register(&context).unwrap();
        drop(context);
        kernel.run().unwrap();

        let mut guest = Self {
            kernel,
            memory,
            heap,
            root,
            mailbox,
            sm: 0,
        };
        guest.memory.write_bytes(heap, b"sm:\0").unwrap();
        let x = guest.svc(0x1f, [0, heap, 0, 0, 0, 0, 0, 0], &[]);
        assert_eq!(x[0], 0, "ConnectToNamedPort(sm:)");
        guest.sm = x[1] as u32;
        guest.call(guest.sm, Request::new(0).pid().arg(0u64)).ok();
        guest
    }

    fn svc(&self, imm: u32, x: [u64; 8], message: &[u32]) -> [u64; 8] {
        self.exchange(imm, x, message).x
    }

    fn exchange(&self, imm: u32, x: [u64; 8], message: &[u32]) -> Reply {
        let mut mail = self.mailbox.mail.lock();
        mail.op = Some(Op {
            imm,
            x,
            message: message.to_vec(),
        });
        self.mailbox.changed.notify_all();
        let deadline = Instant::now() + TIMEOUT;
        loop {
            if let Some(reply) = mail.reply.take() {
                return reply;
            }
            assert!(Instant::now() < deadline, "the guest did not answer");
            self.mailbox
                .changed
                .wait_for(&mut mail, Duration::from_millis(10));
        }
    }

    /// Sends `request` on session `handle` with `SendSyncRequest`.
    pub fn call(&self, handle: u32, request: Request) -> Response {
        let reply = self.exchange(
            0x21,
            [u64::from(handle), 0, 0, 0, 0, 0, 0, 0],
            &request.encode(),
        );
        assert_eq!(reply.x[0], 0, "SendSyncRequest failed");
        Response::parse(&reply.words)
    }

    /// A session to the service `name`.
    pub fn service(&self, name: &str) -> u32 {
        let mut raw = [0u8; 8];
        raw[..name.len()].copy_from_slice(name.as_bytes());
        self.call(self.sm, Request::new(1).arg(u64::from_le_bytes(raw)))
            .handle()
    }

    /// `bytes` written to guest memory at `offset` into the heap.
    pub fn put(&self, offset: u64, bytes: &[u8]) -> (u64, u64) {
        self.memory.write_bytes(self.heap + offset, bytes).unwrap();
        (self.heap + offset, bytes.len() as u64)
    }

    /// A NUL-terminated path in path slot `slot`, sized like `nn::fssrv::sf::Path`.
    pub fn path(&self, slot: u64, path: &str) -> (u64, u64) {
        let mut raw = vec![0u8; 0x301];
        raw[..path.len()].copy_from_slice(path.as_bytes());
        self.put(0x1000 + slot * 0x400, &raw)
    }

    /// A buffer of `size` bytes of `fill` at `offset` into the heap.
    pub fn buffer(&self, offset: u64, size: u64, fill: u8) -> (u64, u64) {
        self.memory.fill(self.heap + offset, size, fill).unwrap();
        (self.heap + offset, size)
    }

    pub fn read(&self, (address, size): (u64, u64)) -> Vec<u8> {
        let mut out = vec![0; size as usize];
        self.memory.read_bytes(address, &mut out).unwrap();
        out
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        self.kernel.shutdown();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// A 64-bit application with one module; the mock core never fetches its code.
fn image(console: ConsoleGeneration) -> ProgramImage {
    ProgramImage {
        program_id: match console {
            ConsoleGeneration::Switch1 => SWITCH1_PROGRAM,
            ConsoleGeneration::Switch2 => SWITCH2_PROGRAM,
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
