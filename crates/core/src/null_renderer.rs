//! A renderer that draws nothing, for headless runs and tests.

use lichii_common::DAddr;
use lichii_gpu::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles,
    GraphicsContext, IndirectDraw, Query, Rasterizer,
};
use lichii_gpu::gmmu::Gmmu;

use crate::RendererFactory;

/// Draws nothing and runs every fence action as soon as it is signalled, so syncpoints and
/// semaphores still advance. Query reports are not written.
pub struct NullRasterizer {
    handles: GpuHandles,
}

impl NullRasterizer {
    /// A [`RendererFactory`] for [`crate::System::boot`].
    pub fn factory() -> RendererFactory {
        Box::new(|handles| Box::new(NullRasterizer { handles }) as Box<dyn Rasterizer>)
    }
}

impl Rasterizer for NullRasterizer {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}

    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}

    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}

    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}

    fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, _query: &Query) {}

    fn signal_fence(&mut self, action: FenceAction) {
        action.execute(&self.handles);
    }

    fn release_fences(&mut self, _wait: bool) {}

    fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}

    fn flush_region(&mut self, _addr: DAddr, _size: u64) {}

    fn wait_for_idle(&mut self) {}

    fn present(&mut self, _framebuffer: &FramebufferConfig) {}
}
