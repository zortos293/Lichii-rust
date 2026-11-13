//! `/dev/nvdisp_disp0` and the display flip the compositor (vi) presents through.

use lichii_gpu::backend::FramebufferConfig;

use super::{NvFence, NvResult, Nvdrv};

impl Nvdrv {
    /// Presents nvmap buffer `handle`, from `offset` bytes in, once the guest reached `fences`
    /// (ids below zero are unused slots), without waiting for them: the compositor thread that
    /// calls it also ticks vsync. `framebuffer.address` is replaced by the buffer's device
    /// address; the buffer must be pinned, as a buffer the title rendered to is. `presented` runs
    /// once the present is queued, or at once when the flip fails.
    pub(crate) fn flip(
        &self,
        handle: u32,
        offset: u64,
        mut framebuffer: FramebufferConfig,
        fences: &[NvFence],
        presented: impl FnOnce() + Send + 'static,
    ) -> NvResult {
        let daddr = self.nvmap.device_address(handle);
        if daddr == 0 {
            tracing::error!(handle, "flip of an nvmap buffer that is not pinned");
            presented();
            return NvResult::BadValue;
        }
        framebuffer.address = daddr + offset;
        crate::context::present_when_reached(
            &self.gpu,
            fences
                .iter()
                .filter(|fence| fence.id >= 0)
                .map(|fence| (fence.id as u32, fence.value)),
            framebuffer,
            presented,
        );
        NvResult::Success
    }

    /// Holds nvmap buffer `handle` for a vi buffer queue: a reference keeps it, and its device
    /// mapping, after the guest frees it, and a pin keeps it in device space. False when the
    /// handle cannot be held.
    pub(crate) fn pin_buffer(&self, handle: u32) -> bool {
        let held =
            self.nvmap
                .duplicate_internal(handle)
                .and_then(|()| match self.nvmap.pin(handle) {
                    Ok(_) => Ok(()),
                    Err(error) => {
                        self.nvmap.free_internal(handle);
                        Err(error)
                    }
                });
        if let Err(error) = held {
            tracing::warn!(handle, ?error, "vi: holding a preallocated buffer");
        }
        held.is_ok()
    }

    /// Lets go of a buffer [`Nvdrv::pin_buffer`] held.
    pub(crate) fn unpin_buffer(&self, handle: u32) {
        self.nvmap.unpin(handle);
        self.nvmap.free_internal(handle);
    }
}
