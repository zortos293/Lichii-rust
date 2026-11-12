//! Builds a [`ServiceContext`] on the kernel's host-side IPC client for service tests.

use std::sync::Arc;

use lichii_common::ConsoleGeneration;
use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_gpu::host1x::Host1x;
use lichii_input::hub::NpadHub;
use lichii_kernel::testing::{self, IpcClient};

use crate::{ServiceContext, ServiceParts};

/// Program id of a Switch 1 test title.
pub const SWITCH1_TITLE: u64 = 0x0100_0000_0000_1000;
/// Program id of a Switch 2 test title.
pub const SWITCH2_TITLE: u64 = 0x0400_0000_0000_1000;

/// A context for a test title, with `settings` and an IPC client in the title's process.
pub fn context(program_id: u64, settings: Settings) -> (Arc<ServiceContext>, IpcClient) {
    let generation = ConsoleGeneration::from_program_id(program_id);
    let kernel = testing::kernel(generation);
    let program = Arc::new(testing::program_image(program_id));
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
