//! A host-side IPC client for service tests, behind the `testing` feature.
//!
//! [`IpcClient`] owns a guest thread that never runs code: it writes CMIF requests into that
//! thread's TLS, hands them to the session's service thread exactly as `svcSendSyncRequest` does,
//! and reads the reply back, so handles and buffers go through the real kernel paths.

use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use bytemuck::Pod;
use lichii_common::settings::{CpuAccuracy, CpuBackend};
use lichii_common::{ConsoleGeneration, ResultCode, VAddr};
use lichii_cpu::{CpuConfig, CpuCore, GuestClock, HaltReason, ThreadContext};
use lichii_loader::npdm::{AddressSpaceType, KernelCapabilities, PoolPartition, ServiceAccess};
use lichii_loader::{MainThread, Module, Permission, ProgramImage, Segment};
use parking_lot::Mutex;

use crate::ipc::{ClientSession, Handle, SessionHandler, TLS_MESSAGE_SIZE, message};
use crate::memory::PoolLayout;
use crate::object::KObject;
use crate::process::Process;
use crate::thread::{Resume, Thread};
use crate::{Kernel, KernelConfig};

const CODE_SIZE: u64 = 0x100_0000;
const SCRATCH_OFFSET: u64 = 0x10_0000;

struct HostClock {
    start: Instant,
    hz: u64,
}

impl GuestClock for HostClock {
    fn ticks(&self) -> u64 {
        (self.start.elapsed().as_nanos() * u128::from(self.hz) / 1_000_000_000) as u64
    }

    fn frequency(&self) -> u64 {
        self.hz
    }
}

/// A kernel with 256 MiB of DRAM whose cores never run (tests drive it from the host). Its clock
/// ticks at the generation's counter rate, as `core::System` sets it for a title.
pub fn kernel(console: ConsoleGeneration) -> Kernel {
    Kernel::new(config(console)).expect("test kernel")
}

/// The configuration [`kernel`] uses; sm enforces NPDM service lists.
pub fn config(console: ConsoleGeneration) -> KernelConfig {
    KernelConfig {
        console,
        dram_size: 256 << 20,
        pool_layout: Some(PoolLayout {
            application: 192 << 20,
            applet: 16 << 20,
            system_non_secure: 16 << 20,
        }),
        fastmem: false,
        cpu_accuracy: CpuAccuracy::Auto,
        aslr: false,
        rng_seed: 1,
        enforce_service_access: true,
        clock: Arc::new(HostClock {
            start: Instant::now(),
            hz: match console {
                ConsoleGeneration::Switch1 => 19_200_000,
                ConsoleGeneration::Switch2 => 31_250_000,
            },
        }),
        cpu: Arc::new(|config| lichii_cpu::new_core(CpuBackend::Interpreter, config)),
        events: None,
    }
}

/// A 64-bit application image with one 16 MiB module of zeroes; its writable part holds the
/// buffers [`IpcClient::alloc`] hands out.
pub fn program_image(program_id: u64) -> ProgramImage {
    ProgramImage {
        program_id,
        generation: ConsoleGeneration::from_program_id(program_id),
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
        icon: None,
        version: 0,
        pseudo_device_seed: program_id,
        add_ons: Vec::new(),
    }
}

/// A CMIF (or TIPC) request under construction.
#[derive(Clone, Debug, Default)]
pub struct Request {
    command: u32,
    tipc: bool,
    control: bool,
    object: Option<u32>,
    input_objects: Vec<u32>,
    data: Vec<u8>,
    send_pid: bool,
    copies: Vec<Handle>,
    moves: Vec<Handle>,
    x: Vec<(VAddr, u64)>,
    a: Vec<(VAddr, u64)>,
    b: Vec<(VAddr, u64)>,
    c: Vec<(VAddr, u64)>,
}

impl Request {
    /// A CMIF request for `command`.
    pub fn new(command: u32) -> Self {
        Self {
            command,
            ..Self::default()
        }
    }

    /// A CMIF control request (`ConvertCurrentObjectToDomain` is 0).
    pub fn control(command: u32) -> Self {
        Self {
            command,
            control: true,
            ..Self::default()
        }
    }

    /// Sends the request to domain object `id`, with a domain header.
    pub fn to_object(mut self, id: u32) -> Self {
        self.object = Some(id);
        self
    }

    /// Passes domain object `id` as the next input object.
    pub fn input_object(mut self, id: u32) -> Self {
        self.input_objects.push(id);
        self
    }

    /// A TIPC request for `command`.
    pub fn tipc(command: u32) -> Self {
        Self {
            command,
            tipc: true,
            ..Self::default()
        }
    }

    /// Appends a raw argument, aligned like [`HleRequestContext::pop`] reads it (to its size, at
    /// most 8 bytes).
    ///
    /// [`HleRequestContext::pop`]: crate::ipc::HleRequestContext::pop
    pub fn arg<T: Pod>(mut self, value: T) -> Self {
        let align = std::mem::align_of::<T>().min(8);
        self.data.resize(self.data.len().next_multiple_of(align), 0);
        self.data.extend_from_slice(bytemuck::bytes_of(&value));
        self
    }

    /// Asks the kernel to pass the client's process id.
    pub fn send_pid(mut self) -> Self {
        self.send_pid = true;
        self
    }

    /// Passes one of the client's handles; the client keeps it.
    pub fn copy_handle(mut self, handle: Handle) -> Self {
        self.copies.push(handle);
        self
    }

    /// Passes one of the client's handles; the kernel removes it from the client's table.
    pub fn move_handle(mut self, handle: Handle) -> Self {
        self.moves.push(handle);
        self
    }

    /// An X (pointer) buffer.
    pub fn buffer_x(mut self, address: VAddr, size: u64) -> Self {
        self.x.push((address, size));
        self
    }

    /// An A (mapped, client to server) buffer.
    pub fn buffer_a(mut self, address: VAddr, size: u64) -> Self {
        self.a.push((address, size));
        self
    }

    /// A B (mapped, server to client) buffer.
    pub fn buffer_b(mut self, address: VAddr, size: u64) -> Self {
        self.b.push((address, size));
        self
    }

    /// A C (receive list) buffer.
    pub fn buffer_c(mut self, address: VAddr, size: u64) -> Self {
        self.c.push((address, size));
        self
    }

    fn encode(&self) -> Vec<u32> {
        let ty = match (self.tipc, self.control) {
            (true, _) => 16 + self.command,
            (false, true) => 5,
            (false, false) => 4,
        };
        let mut words = vec![
            ty | (self.x.len() as u32) << 16
                | (self.a.len() as u32) << 20
                | (self.b.len() as u32) << 24,
            0,
        ];
        let special = self.send_pid || !self.copies.is_empty() || !self.moves.is_empty();
        if special {
            words.push(
                u32::from(self.send_pid)
                    | (self.copies.len() as u32) << 1
                    | (self.moves.len() as u32) << 5,
            );
            if self.send_pid {
                words.extend([0, 0]);
            }
            words.extend(&self.copies);
            words.extend(&self.moves);
        }
        for (index, &(address, size)) in self.x.iter().enumerate() {
            let index = index as u32;
            words.push(
                (index & 0x3f)
                    | ((address >> 36) as u32 & 0x7) << 6
                    | ((index >> 9) & 0x7) << 9
                    | ((address >> 32) as u32 & 0xf) << 12
                    | (size as u32) << 16,
            );
            words.push(address as u32);
        }
        for &(address, size) in self.a.iter().chain(&self.b) {
            words.push(size as u32);
            words.push(address as u32);
            words.push(
                ((address >> 36) as u32 & 0x7) << 2
                    | ((size >> 32) as u32 & 0xf) << 24
                    | ((address >> 32) as u32 & 0xf) << 28,
            );
        }
        let raw_start = words.len();
        let payload = self.data.len().div_ceil(4);
        if self.tipc {
            words.resize(raw_start + payload, 0);
        } else {
            let header = raw_start.next_multiple_of(4);
            words.resize(header, 0);
            if let Some(object) = self.object {
                let size = (16 + payload * 4) as u32;
                let count = self.input_objects.len() as u32;
                words.extend([1 | count << 8 | size << 16, object, 0, 0]);
            }
            words.extend([message::SFCI, 0, self.command, 0]);
            words.resize(words.len() + payload, 0);
        }
        let start = words.len() - payload;
        bytemuck::cast_slice_mut::<u32, u8>(&mut words[start..])[..self.data.len()]
            .copy_from_slice(&self.data);
        words.extend(&self.input_objects);
        let raw = if self.tipc {
            payload
        } else {
            words.len() - raw_start + 4
        };
        words.resize(raw_start + raw, 0);
        for &(address, size) in &self.c {
            words.push(address as u32);
            words.push(((address >> 32) as u32 & 0xffff) | (size as u32) << 16);
        }
        let c_flags = match self.c.len() {
            0 => 0,
            1 => 2,
            n => n as u32 + 2,
        };
        words[1] = raw as u32 | c_flags << 10 | u32::from(special) << 31;
        words
    }
}

/// A decoded reply.
#[derive(Clone, Debug)]
pub struct Reply {
    /// The result the service answered with (the kernel-level send error is [`IpcClient::send`]'s
    /// `Err`).
    pub result: ResultCode,
    /// Handles the service copied to the client, already in the client's handle table.
    pub copy_handles: Vec<Handle>,
    /// Handles the service moved to the client (sub-interface sessions among them), already in the
    /// client's handle table.
    pub move_handles: Vec<Handle>,
    /// Domain object ids of the sub-interfaces a domain reply returned.
    pub objects: Vec<u32>,
    data: Vec<u8>,
    cursor: usize,
}

impl Reply {
    fn decode(words: &[u32], tipc: bool, domain: bool) -> Self {
        let mut i = 2;
        let (mut copy_handles, mut move_handles) = (Vec::new(), Vec::new());
        if words[1] >> 31 != 0 {
            let descriptor = words[i];
            i += 1;
            copy_handles.extend(&words[i..i + ((descriptor >> 1) & 0xf) as usize]);
            i += copy_handles.len();
            move_handles.extend(&words[i..i + ((descriptor >> 5) & 0xf) as usize]);
            i += move_handles.len();
        }
        let raw = (words[1] & 0x3ff) as usize;
        let mut object_count = 0;
        let (result, start, data_words) = if tipc {
            (ResultCode(words[i]), i + 1, raw.saturating_sub(1))
        } else {
            let mut header = i.next_multiple_of(4);
            if domain {
                object_count = words[header] as usize;
                header += 4;
            }
            assert_eq!(words[header], message::SFCO, "reply without SFCO");
            (
                ResultCode(words[header + 2]),
                header + 4,
                raw.saturating_sub(8 + if domain { 4 + object_count } else { 0 }),
            )
        };
        let raw_end = words.len().min(start + data_words);
        Self {
            result,
            copy_handles,
            move_handles,
            objects: words[raw_end..raw_end + object_count].to_vec(),
            data: bytemuck::cast_slice(&words[start.min(raw_end)..raw_end]).to_vec(),
            cursor: 0,
        }
    }

    /// Reads the next output value, aligned like [`HleRequestContext::push`] writes it.
    ///
    /// [`HleRequestContext::push`]: crate::ipc::HleRequestContext::push
    pub fn pop<T: Pod>(&mut self) -> T {
        self.cursor = self
            .cursor
            .next_multiple_of(std::mem::align_of::<T>().min(8));
        let size = std::mem::size_of::<T>();
        let value = bytemuck::pod_read_unaligned(&self.data[self.cursor..self.cursor + size]);
        self.cursor += size;
        value
    }

    /// The reply's output data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }
}

/// Next free scratch address of each process, shared by every client in it.
static SCRATCH: Mutex<Vec<(Weak<Process>, VAddr)>> = Mutex::new(Vec::new());

/// Sends requests to HLE sessions as a thread of `process`. Requests from one client run one at a
/// time.
pub struct IpcClient {
    kernel: Kernel,
    process: Arc<Process>,
    thread: Arc<Thread>,
    in_flight: Mutex<()>,
}

impl IpcClient {
    /// Creates the client thread and starts the kernel's service threads (the cores stay off).
    pub fn new(kernel: &Kernel, process: &Arc<Process>) -> Self {
        let base = process.entry();
        let thread = kernel
            .0
            .create_thread(process, base, 0, base + SCRATCH_OFFSET, 44, -2)
            .expect("client thread");
        kernel.spawn_service_threads().expect("service threads");
        Self {
            kernel: kernel.clone(),
            process: process.clone(),
            thread,
            in_flight: Mutex::new(()),
        }
    }

    /// The client's process.
    pub fn process(&self) -> &Arc<Process> {
        &self.process
    }

    /// Opens a session to the service `name` registered with `sm:`.
    pub fn connect(&self, name: &str) -> Option<Handle> {
        let session = self.kernel.service_manager().open(name)?;
        Some(self.insert(session))
    }

    /// Opens a session to `handler`, served by `thread`.
    pub fn connect_handler(
        &self,
        thread: &crate::ipc::ServiceThread,
        handler: Arc<dyn SessionHandler>,
    ) -> Handle {
        self.insert(thread.session(handler))
    }

    fn insert(&self, session: Arc<ClientSession>) -> Handle {
        self.process
            .handles()
            .write()
            .add(KObject::ClientSession(session))
            .expect("handle table full")
    }

    /// The object behind one of the process's handles (an event, a session, shared memory, ...).
    pub fn object(&self, handle: Handle) -> Option<KObject> {
        self.process.handles().read().get(handle)
    }

    /// Copies `bytes` into fresh guest memory and returns its address.
    pub fn alloc(&self, bytes: &[u8]) -> VAddr {
        let mut scratch = SCRATCH.lock();
        scratch.retain(|(owner, _)| owner.strong_count() > 0);
        let owner = Arc::downgrade(&self.process);
        let index = match scratch.iter().position(|(o, _)| o.ptr_eq(&owner)) {
            Some(index) => index,
            None => {
                scratch.push((owner, self.process.entry() + SCRATCH_OFFSET));
                scratch.len() - 1
            }
        };
        let address = scratch[index].1;
        scratch[index].1 = (address + bytes.len().max(1) as u64).next_multiple_of(0x100);
        drop(scratch);
        self.process
            .memory()
            .write_bytes(address, bytes)
            .expect("scratch memory");
        address
    }

    /// Reads `size` bytes of guest memory.
    pub fn read(&self, address: VAddr, size: usize) -> Vec<u8> {
        let mut out = vec![0; size];
        self.process
            .memory()
            .read_bytes(address, &mut out)
            .expect("guest memory");
        out
    }

    /// Sends `request` on `session` and waits for the reply. `Err` is the result the kernel
    /// returns from `svcSendSyncRequest` itself (a closed session, a bad handle).
    pub fn send(&self, session: Handle, request: &Request) -> Result<Reply, ResultCode> {
        let client = match self.object(session) {
            Some(KObject::ClientSession(client)) => client,
            _ => return Err(crate::result::INVALID_HANDLE),
        };
        let words = request.encode();
        if words.len() * 4 > TLS_MESSAGE_SIZE {
            return Err(crate::result::MESSAGE_TOO_LARGE);
        }
        let _in_flight = self.in_flight.lock();
        let message = self.thread.tls_address();
        self.process
            .memory()
            .write_bytes(message, bytemuck::cast_slice(&words))
            .expect("tls");
        let token = self.thread.begin_wait(Resume::Result, false)?;
        client.send(self.thread.clone(), token, message, TLS_MESSAGE_SIZE);
        let deadline = Instant::now() + Duration::from_secs(10);
        while self.thread.is_waiting_on(token) {
            assert!(
                Instant::now() < deadline,
                "no reply to command {}",
                request.command
            );
            std::thread::sleep(Duration::from_micros(100));
        }
        let wake = self.thread.sched.lock().wake;
        if !wake.result.is_success() {
            return Err(wake.result);
        }
        let mut reply = [0u32; TLS_MESSAGE_SIZE / 4];
        self.process
            .memory()
            .read_bytes(message, bytemuck::cast_slice_mut(&mut reply))
            .expect("tls");
        Ok(Reply::decode(
            &reply,
            request.tipc,
            request.object.is_some(),
        ))
    }
}

/// A JIT core that runs a process's code from the host. Like the kernel's cores, it drops the code
/// the kernel invalidated before every run.
pub struct JitCore {
    kernel: Kernel,
    process: Arc<Process>,
    cpu: Box<dyn CpuCore>,
}

impl JitCore {
    /// A JIT core over `process`'s address space, sharing `kernel`'s exclusive monitor, clock and
    /// CPU accuracy. It takes core 0's invalidations, so use it only on a kernel whose cores never
    /// run, such as [`kernel`].
    pub fn new(kernel: &Kernel, process: &Arc<Process>) -> Self {
        let cpu = lichii_cpu::new_core(
            CpuBackend::Jit,
            CpuConfig {
                memory: process.memory().clone(),
                monitor: kernel.0.monitor.clone(),
                clock: kernel.0.clock.clone(),
                core_index: 0,
                ctr_el0: CpuConfig::DEFAULT_CTR_EL0,
                dczid_el0: CpuConfig::DEFAULT_DCZID_EL0,
                accuracy: kernel.0.cpu_accuracy,
            },
        );
        Self {
            kernel: kernel.clone(),
            process: process.clone(),
            cpu,
        }
    }

    /// Runs from `pc` until the core halts, and returns why with `x0`.
    pub fn run(&mut self, pc: VAddr) -> (HaltReason, u64) {
        for (pid, start, size) in self.kernel.0.take_invalidations(0) {
            if pid == self.process.id() {
                self.cpu.invalidate_code(start, size);
            }
        }
        let mut ctx = ThreadContext {
            pc,
            ..Default::default()
        };
        let halt = self.cpu.run(&mut ctx);
        (halt, ctx.x[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;
    use crate::ipc::HleRequestContext;

    struct Echo(Arc<Event>);

    impl SessionHandler for Echo {
        fn name(&self) -> &str {
            "echo"
        }

        fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode {
            match ctx.command() {
                0 => {
                    let (a, b): (u32, u64) = (ctx.pop(), ctx.pop());
                    ctx.push(a + 1);
                    ctx.push(b * 2);
                    ctx.push_copy_object(KObject::ReadableEvent(self.0.clone()));
                }
                2 => {
                    let (a, b): (u64, u128) = (ctx.pop(), ctx.pop());
                    ctx.push(b + u128::from(a));
                }
                1 => {
                    let mut data = ctx.read_buffer(0).unwrap();
                    data.reverse();
                    ctx.write_buffer(0, &data).unwrap();
                    ctx.push(ctx.pid().unwrap_or(0));
                }
                _ => return ResultCode(0x1234),
            }
            ResultCode::SUCCESS
        }
    }

    #[test]
    fn round_trips_requests_handles_and_buffers() {
        let kernel = kernel(ConsoleGeneration::Switch1);
        let process = kernel
            .create_process(&program_image(0x0100_0000_0000_2000))
            .unwrap();
        let thread = kernel.create_service_thread("echo").unwrap();
        let event = Arc::new(Event::new());
        let shared = event.clone();
        kernel
            .service_manager()
            .register(
                "echo",
                &thread,
                Arc::new(move || Arc::new(Echo(shared.clone())) as Arc<dyn SessionHandler>),
            )
            .unwrap();
        let client = IpcClient::new(&kernel, &process);
        let session = client.connect("echo").unwrap();
        assert!(client.connect("nope").is_none());

        let mut reply = client
            .send(session, &Request::new(0).arg(41u32).arg(21u64))
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.data().len(), 16);
        assert_eq!((reply.pop::<u32>(), reply.pop::<u64>()), (42, 42));
        let Some(KObject::ReadableEvent(got)) = client.object(reply.copy_handles[0]) else {
            panic!("no event handle");
        };
        assert!(Arc::ptr_eq(&got, &event));

        let input = client.alloc(b"abcd");
        let output = client.alloc(&[0; 4]);
        let mut reply = client
            .send(
                session,
                &Request::new(1)
                    .send_pid()
                    .buffer_a(input, 4)
                    .buffer_b(output, 4),
            )
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(client.read(output, 4), b"dcba");
        assert_eq!(reply.pop::<u64>(), process.id());

        let mut reply = client
            .send(session, &Request::new(2).arg(1u64).arg(2u128))
            .unwrap();
        assert_eq!(reply.pop::<u128>(), 3);

        let reply = client.send(session, &Request::new(9)).unwrap();
        assert_eq!(reply.result, ResultCode(0x1234));

        let oversized = Request::new(0).arg([0u64; 32]);
        assert_eq!(
            client.send(session, &oversized).unwrap_err(),
            crate::result::MESSAGE_TOO_LARGE
        );

        let other = IpcClient::new(&kernel, &process);
        let (first, second) = (client.alloc(b"first"), other.alloc(b"other"));
        assert_ne!(first, second);
        assert_eq!(client.read(first, 5), b"first");
        kernel.shutdown();
    }
    struct Root;

    struct Leaf(u32);

    impl SessionHandler for Leaf {
        fn name(&self) -> &str {
            "leaf"
        }

        fn handle_request(&self, _: &mut HleRequestContext) -> ResultCode {
            ResultCode::SUCCESS
        }

        fn as_any(&self) -> Option<&dyn std::any::Any> {
            Some(self)
        }
    }

    impl SessionHandler for Root {
        fn name(&self) -> &str {
            "root"
        }

        fn handle_request(&self, ctx: &mut HleRequestContext) -> ResultCode {
            match ctx.command() {
                0 => ctx.push_interface(Arc::new(Leaf(42))),
                1 => {
                    let Some(object) = ctx.input_object(0) else {
                        return ResultCode(0x1234);
                    };
                    let Some(leaf) = object.as_any().and_then(|any| any.downcast_ref::<Leaf>())
                    else {
                        return ResultCode(0x5678);
                    };
                    ctx.push(leaf.0);
                }
                _ => return ResultCode(0x1234),
            }
            ResultCode::SUCCESS
        }
    }

    #[test]
    fn input_objects_resolve_to_their_handlers() {
        let kernel = kernel(ConsoleGeneration::Switch1);
        let process = kernel
            .create_process(&program_image(0x0100_0000_0000_2000))
            .unwrap();
        let thread = kernel.create_service_thread("objects").unwrap();
        kernel
            .service_manager()
            .register(
                "objects",
                &thread,
                Arc::new(|| Arc::new(Root) as Arc<dyn SessionHandler>),
            )
            .unwrap();
        let client = IpcClient::new(&kernel, &process);

        let session = client.connect("objects").unwrap();
        let reply = client.send(session, &Request::new(0)).unwrap();
        let leaf = reply.move_handles[0];
        let mut reply = client
            .send(session, &Request::new(1).move_handle(leaf))
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u32>(), 42, "a session moved as a handle");
        let reply = client.send(session, &Request::new(1)).unwrap();
        assert_eq!(reply.result, ResultCode(0x1234));

        let domain = client.connect("objects").unwrap();
        let mut reply = client.send(domain, &Request::control(0)).unwrap();
        let root = reply.pop::<u32>();
        let reply = client
            .send(domain, &Request::new(0).to_object(root))
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let [leaf] = reply.objects[..] else {
            panic!("one domain object, got {:?}", reply.objects);
        };
        let mut reply = client
            .send(domain, &Request::new(1).to_object(root).input_object(leaf))
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u32>(), 42, "a domain object");
        let reply = client
            .send(domain, &Request::new(1).to_object(root).input_object(root))
            .unwrap();
        assert_eq!(
            reply.result,
            ResultCode(0x5678),
            "another type does not downcast"
        );
        let reply = client
            .send(domain, &Request::new(1).to_object(root).input_object(99))
            .unwrap();
        assert_eq!(
            reply.result,
            ResultCode(0x1234),
            "an unknown id resolves to nothing"
        );
        kernel.shutdown();
    }
}
