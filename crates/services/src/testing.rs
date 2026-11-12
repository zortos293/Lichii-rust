//! Builds a [`ServiceContext`] on the kernel's host-side IPC client for service tests.

use std::sync::Arc;

use lichii_common::ConsoleGeneration;
use lichii_common::DAddr;
use lichii_common::perf::PerfStats;
use lichii_common::settings::Settings;
use lichii_gpu::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles,
    GraphicsContext, IndirectDraw, Query, Rasterizer,
};
use lichii_gpu::gmmu::Gmmu;
use lichii_gpu::host1x::Host1x;
use lichii_gpu::{Console, Gpu, GpuConfig};
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
    let (console, bits) = match generation {
        ConsoleGeneration::Switch1 => (Console::Switch1, 34),
        ConsoleGeneration::Switch2 => (Console::Switch2, 36),
    };
    let host1x = Host1x::new(kernel.physical_memory().clone(), bits);
    let gpu = Gpu::new(
        host1x,
        Arc::new(Clock),
        GpuConfig::new(console),
        |handles| Box::new(NullRasterizer(handles)),
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

struct Clock;

impl lichii_gpu::clock::GuestTime for Clock {
    fn global_time_ns(&self) -> u64 {
        0
    }
}

/// Draws nothing; fence actions run at once.
struct NullRasterizer(GpuHandles);

impl Rasterizer for NullRasterizer {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}
    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}
    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}
    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}
    fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, _query: &Query) {}
    fn signal_fence(&mut self, action: FenceAction) {
        action.execute(&self.0);
    }
    fn release_fences(&mut self, _wait: bool) {}
    fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}
    fn flush_region(&mut self, _addr: DAddr, _size: u64) {}
    fn wait_for_idle(&mut self) {}
    fn present(&mut self, _framebuffer: &FramebufferConfig) {}
}
