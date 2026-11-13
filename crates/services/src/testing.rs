//! Builds a [`ServiceContext`] on the kernel's host-side IPC client for service tests.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::Pod;
use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_common::{ConsoleGeneration, HleResult, ResultCode, VAddr};
use lichii_fs::nacp::Nacp;
use lichii_gpu::backend::NullRasterizer;
use lichii_gpu::host1x::Host1x;
use lichii_gpu::{Console, Gpu, GpuConfig};
use lichii_input::hub::NpadHub;
use lichii_kernel::ipc::Handle;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::process::Process;
use lichii_kernel::testing::{self, IpcClient};
pub use lichii_kernel::testing::{Reply, Request};
use lichii_loader::program::ProgramImage;
use lichii_memory::MemoryPermission;

use crate::{ServiceContext, ServiceParts};

/// Program id of a Switch 1 test title.
pub const SWITCH1_TITLE: u64 = 0x0100_0000_0000_1000;
/// Program id of a Switch 2 test title.
pub const SWITCH2_TITLE: u64 = 0x0400_0000_0000_1000;

/// A context for a test title, with `settings` and an IPC client in the title's process.
pub fn context(program_id: u64, settings: Settings) -> (Arc<ServiceContext>, IpcClient) {
    context_with(testing::program_image(program_id), settings)
}

/// A context for `program`, for tests that give the title a RomFS, NACP or add-on content.
pub fn context_with(program: ProgramImage, settings: Settings) -> (Arc<ServiceContext>, IpcClient) {
    let generation = ConsoleGeneration::from_program_id(program.program_id);
    let kernel = testing::kernel(generation);
    let program = Arc::new(program);
    let process = kernel.create_process(&program).unwrap();
    let (console, bits) = match generation {
        ConsoleGeneration::Switch1 => (Console::Switch1, 34),
        ConsoleGeneration::Switch2 => (Console::Switch2, 36),
    };
    let host1x = Host1x::new(kernel.physical_memory().clone(), bits);
    let gpu = Gpu::new(
        host1x,
        Arc::new(Clock),
        GpuConfig::new(console),
        |handles| Box::new(NullRasterizer::new(handles)),
    );
    let client = IpcClient::new(&kernel, &process);
    let context = ServiceContext::new(ServiceParts {
        kernel,
        settings: Arc::new(settings),
        program,
        process,
        gpu: Arc::new(gpu),
        npad: Arc::new(NpadHub::default()),
        perf: Arc::new(PerfStats::default()),
        audio: Arc::new(lichii_audio::sink::Sink::null()),
    });
    (context, client)
}

/// An IPC client in a test title whose services `register` has registered.
pub fn booted(
    program_id: u64,
    settings: Settings,
    register: impl FnOnce(&Arc<ServiceContext>) -> HleResult<()>,
) -> IpcClient {
    let (context, client) = context(program_id, settings);
    register(&context).unwrap();
    client
}

static NEXT_DIR: AtomicU32 = AtomicU32::new(0);

/// A host directory of its own for one test, removed when dropped.
pub struct TempDir(PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "lichii-{tag}-{}-{}",
            std::process::id(),
            NEXT_DIR.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    /// Default settings with the NAND, SD card and keys in this directory.
    pub fn settings(&self) -> Settings {
        let mut settings = Settings::default();
        settings.paths.nand_dir = Some(self.0.join("nand"));
        settings.paths.sdmc_dir = Some(self.0.join("sdmc"));
        settings.paths.keys_dir = Some(self.0.join("keys"));
        settings
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Reads a value of guest memory.
pub fn read_value<T: Pod>(client: &IpcClient, address: VAddr) -> T {
    bytemuck::pod_read_unaligned(&client.read(address, std::mem::size_of::<T>()))
}

/// Program id of [`image`].
pub const PROGRAM_ID: u64 = 0x0100_0000_0001_0000;

/// The kernel's test image for [`PROGRAM_ID`]; tests add a NACP or add-ons.
pub fn image() -> ProgramImage {
    testing::program_image(PROGRAM_ID)
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

/// A title's process with services registered and an [`IpcClient`] in it.
pub struct Guest {
    pub client: IpcClient,
    pub process: Arc<Process>,
    context: Arc<ServiceContext>,
    /// 1 MiB of page-aligned guest memory for request buffers.
    pub heap: VAddr,
}

impl Guest {
    pub fn new(
        program: ProgramImage,
        settings: Settings,
        register: impl FnOnce(&Arc<ServiceContext>),
    ) -> Self {
        let (context, client) = context_with(program, settings);
        register(&context);
        let heap = client.alloc(&vec![0; 0x10_1000]).next_multiple_of(0x1000);
        Self {
            process: client.process().clone(),
            client,
            context,
            heap,
        }
    }

    /// Another client thread in the title's process; it shares the process's handles.
    pub fn other_client(&self) -> IpcClient {
        IpcClient::new(&self.context.kernel, &self.process)
    }

    /// Opens a session to the registered service `name`.
    pub fn service(&self, name: &str) -> Handle {
        self.client
            .connect(name)
            .unwrap_or_else(|| panic!("{name} is not registered"))
    }

    /// Sends `request` on `session`; the kernel itself must accept it.
    pub fn call(&self, session: Handle, request: Request) -> Reply {
        self.client
            .send(session, &request)
            .expect("SendSyncRequest")
    }

    pub fn write(&self, address: VAddr, data: &[u8]) {
        self.process.memory().write_bytes(address, data).unwrap();
    }

    pub fn read(&self, address: VAddr, len: usize) -> Vec<u8> {
        self.client.read(address, len)
    }

    /// Whether the readable event behind `handle` is signaled.
    pub fn is_signaled(&self, handle: Handle) -> bool {
        match self.client.object(handle) {
            Some(KObject::ReadableEvent(event)) => event.is_signaled(),
            _ => panic!("handle {handle:#x} is not a readable event"),
        }
    }

    /// `svcCreateTransferMemory` over `size` bytes at `address`, read-write.
    pub fn transfer_memory(&self, address: VAddr, size: u64) -> Handle {
        let memory = TransferMemory::new(
            self.process.clone(),
            address,
            size,
            MemoryPermission::READ_WRITE,
        )
        .unwrap();
        self.process
            .handles()
            .write()
            .add(KObject::TransferMemory(Arc::new(memory)))
            .unwrap()
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        self.context.kernel.shutdown();
    }
}

/// Shorthands for checking a [`Reply`].
pub trait ReplyExt {
    /// Asserts the service answered success.
    fn ok(self) -> Self;
    /// The value at byte `offset` of the reply data.
    fn get<T: Pod>(&self, offset: usize) -> T;
    /// The reply's one sub-interface (move handle).
    fn interface(&self) -> Handle;
    /// The reply's one copy handle.
    fn handle(&self) -> Handle;
}

impl ReplyExt for Reply {
    #[track_caller]
    fn ok(self) -> Self {
        assert_eq!(self.result, ResultCode::SUCCESS);
        self
    }

    fn get<T: Pod>(&self, offset: usize) -> T {
        bytemuck::pod_read_unaligned(&self.data()[offset..offset + std::mem::size_of::<T>()])
    }

    fn interface(&self) -> Handle {
        assert_eq!(self.move_handles.len(), 1, "expected one interface");
        self.move_handles[0]
    }

    fn handle(&self) -> Handle {
        assert_eq!(self.copy_handles.len(), 1, "expected one copy handle");
        self.copy_handles[0]
    }
}

struct Clock;

impl lichii_gpu::clock::GuestTime for Clock {
    fn global_time_ns(&self) -> u64 {
        0
    }
}
