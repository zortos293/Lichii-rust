//! A GPU memory and a recording renderer for engine tests.

use std::sync::Arc;

use lichii_common::{DAddr, GpuVAddr};
use lichii_memory::{AddressSpace, DeviceAddressSpace, MemoryPermission, PhysicalMemory};
use lichii_texture::block_linear::GobLayout;

use super::EngineContext;
use super::kepler_compute::pcas::QmdScheduler;
use crate::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuWrittenPages,
    GraphicsContext, IndirectDraw, Query, Rasterizer,
};
use crate::gmmu::{BigPageSize, Gmmu};
use crate::host1x::Syncpoints;

const CPU_VA: u64 = 0x1000_0000;
/// Bytes of GPU memory mapped at GPU VA 0.
pub const MAPPED: u64 = 0x40_0000;

/// Renderer calls, recorded.
/// The activity counters of engines under test.
pub static ACTIVITY: crate::activity::Activity = crate::activity::Activity::new();

#[derive(Default)]
pub struct Recorder {
    pub inline_writes: Vec<(GpuVAddr, u64)>,
    pub flushes: Vec<(DAddr, u64)>,
}

impl Rasterizer for Recorder {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}
    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}
    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}
    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}
    fn inline_written(
        &mut self,
        _channel: ChannelId,
        _memory: &Gmmu,
        address: GpuVAddr,
        size: u64,
    ) {
        self.inline_writes.push((address, size));
    }
    fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, _query: &Query) {}
    fn signal_fence(&mut self, _action: FenceAction) {}
    fn release_fences(&mut self, _wait: bool) {}
    fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}
    fn flush_region(&mut self, addr: DAddr, size: u64) {
        self.flushes.push((addr, size));
    }
    fn wait_for_idle(&mut self) {}
    fn present(&mut self, _framebuffer: &FramebufferConfig) {}
}

/// [`MAPPED`] bytes of zeroed GPU memory at GPU VA 0, and a recording renderer.
pub struct Harness {
    _space: AddressSpace,
    pub memory: Gmmu,
    pub rasterizer: Recorder,
    pub syncpoints: Syncpoints,
    pub gob_layout: GobLayout,
    pub qmd_scheduler: QmdScheduler,
    pub written_pages: GpuWrittenPages,
}

impl Harness {
    pub fn new() -> Self {
        let physical = Arc::new(PhysicalMemory::new(1 << 24).unwrap());
        let space = AddressSpace::new(physical.clone(), 39);
        space.map(CPU_VA, 0x10_0000, MAPPED, MemoryPermission::READ_WRITE);
        let device = Arc::new(DeviceAddressSpace::new(physical, 36));
        let daddr = 0x4_0000_0000;
        device.map_process(daddr, &space, CPU_VA, MAPPED).unwrap();
        let memory = Gmmu::new(device, BigPageSize::Size64K);
        memory.map(0, daddr, MAPPED, 0, false).unwrap();
        Self {
            _space: space,
            memory,
            rasterizer: Recorder::default(),
            syncpoints: Syncpoints::new(),
            gob_layout: GobLayout::Tegra,
            qmd_scheduler: QmdScheduler::new(),
            written_pages: GpuWrittenPages::new(36),
        }
    }

    pub fn context(&mut self) -> EngineContext<'_> {
        EngineContext {
            channel: ChannelId(0),
            memory: &self.memory,
            rasterizer: &mut self.rasterizer,
            syncpoints: &self.syncpoints,
            gob_layout: self.gob_layout,
            qmd_scheduler: &mut self.qmd_scheduler,
            written_pages: &self.written_pages,
            activity: &crate::engines::testing::ACTIVITY,
        }
    }
}
