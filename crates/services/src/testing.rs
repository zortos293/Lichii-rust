//! Builds a [`ServiceContext`] on the kernel's host-side IPC client for service tests.

use std::sync::Arc;

use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_common::{ConsoleGeneration, ResultCode, VAddr};
use lichii_fs::nacp::Nacp;
use lichii_gpu::host1x::Host1x;
use lichii_input::hub::NpadHub;
use lichii_kernel::ipc::Handle;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::process::Process;
use lichii_kernel::testing::{self, IpcClient};
pub use lichii_kernel::testing::{Reply, Request};
use lichii_loader::ProgramImage;
use lichii_memory::MemoryPermission;

use crate::{ServiceContext, ServiceParts};

/// Program id of a Switch 1 test title.
pub const SWITCH1_TITLE: u64 = 0x0100_0000_0000_1000;
/// Program id of a Switch 2 test title.
pub const SWITCH2_TITLE: u64 = 0x0400_0000_0000_1000;

/// A context for a test title, with `settings` and an IPC client in the title's process.
pub fn context(program_id: u64, settings: Settings) -> (Arc<ServiceContext>, IpcClient) {
    context_for(testing::program_image(program_id), settings)
}

/// A context for `program` (a NACP or add-ons on top of [`testing::program_image`]).
pub fn context_for(program: ProgramImage, settings: Settings) -> (Arc<ServiceContext>, IpcClient) {
    let generation = program.generation;
    let kernel = testing::kernel(generation);
    let program = Arc::new(program);
    let process = kernel.create_process(&program).unwrap();
    let bits = match generation {
        ConsoleGeneration::Switch1 => 34,
        ConsoleGeneration::Switch2 => 36,
    };
    let host1x = Host1x::new(kernel.physical_memory().clone(), bits);
    let client = IpcClient::new(&kernel, &process);
    let context = ServiceContext::new(ServiceParts {
        kernel,
        settings: Arc::new(settings),
        program,
        process,
        host1x,
        nx2_memory: None,
        npad: Arc::new(NpadHub::default()),
        perf: Arc::new(PerfStats::default()),
    });
    (context, client)
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
        let (context, client) = context_for(program, settings);
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
pub trait ReplyExt: Sized {
    /// Asserts the service answered success.
    fn ok(self) -> Self;
    /// The value at byte `offset` of the reply data.
    fn at<T: bytemuck::Pod>(&self, offset: usize) -> T;
    /// The reply's one sub-interface (move handle).
    fn interface(&self) -> Handle;
    /// The reply's one copy handle.
    fn handle(&self) -> Handle;
}

impl ReplyExt for Reply {
    fn ok(self) -> Self {
        assert_eq!(self.result, ResultCode::SUCCESS, "request failed");
        self
    }

    fn at<T: bytemuck::Pod>(&self, offset: usize) -> T {
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
