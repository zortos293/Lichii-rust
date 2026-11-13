//! `/dev/nvdisp_disp0` and the display flip the compositor (vi) presents through.

use std::time::Duration;

use lichii_gpu::backend::FramebufferConfig;

use super::{NvFence, NvResult, Nvdrv};

impl Nvdrv {
    /// Presents nvmap buffer `handle`, from `offset` bytes in, once the guest reached `fences`
    /// (ids below zero are unused slots). `framebuffer.address` is replaced by the buffer's device
    /// address; the buffer must be pinned, as a buffer the title rendered to is. Each fence is
    /// waited for at most a second.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "vi presents through it once it exists")
    )]
    pub(crate) fn flip(
        &self,
        handle: u32,
        offset: u64,
        mut framebuffer: FramebufferConfig,
        fences: &[NvFence],
    ) -> NvResult {
        let daddr = self.nvmap.device_address(handle);
        if daddr == 0 {
            tracing::error!(handle, "flip of an nvmap buffer that is not pinned");
            return NvResult::BadValue;
        }
        const FENCE_TIMEOUT: Duration = Duration::from_secs(1);
        let guest = self.syncpoints.syncpoints().guest();
        for fence in fences.iter().filter(|fence| fence.id >= 0) {
            if let Err(error) = guest.wait(fence.id as u32, fence.value, Some(FENCE_TIMEOUT)) {
                tracing::warn!(%error, "presenting before an acquire fence");
            }
        }
        framebuffer.address = daddr + offset;
        self.gpu.present(framebuffer);
        NvResult::Success
    }
}
