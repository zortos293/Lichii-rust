//! Builds a [`ServiceContext`] on the kernel's host-side IPC client for service tests.

use std::sync::Arc;

use bytemuck::Pod;
use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_common::{ConsoleGeneration, HleResult, ResultCode, VAddr};
use lichii_gpu::backend::NullRasterizer;
use lichii_gpu::host1x::Host1x;
use lichii_gpu::{Console, Gpu, GpuConfig};
use lichii_input::hub::NpadHub;
use lichii_kernel::ipc::Handle;
use lichii_kernel::testing::{self, IpcClient, Reply};
use lichii_loader::ProgramImage;

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

/// Reads a value of guest memory.
pub fn read_value<T: Pod>(client: &IpcClient, address: VAddr) -> T {
    bytemuck::pod_read_unaligned(&client.read(address, std::mem::size_of::<T>()))
}

/// Checks and reads of replies.
pub trait ReplyExt {
    /// Asserts the service answered success.
    fn ok(self) -> Self;
    /// The value at byte `offset` of the reply data.
    fn get<T: Pod>(&self, offset: usize) -> T;
    /// The first moved handle: a returned interface.
    fn interface(&self) -> Handle;
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
        self.move_handles[0]
    }
}

struct Clock;

impl lichii_gpu::clock::GuestTime for Clock {
    fn global_time_ns(&self) -> u64 {
        0
    }
}
