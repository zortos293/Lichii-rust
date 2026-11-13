//! Which host1x syncpoints nvdrv handed out, and fence checks against host1x's values.
//!
//! The values and maximums are host1x's ([`Syncpoints`]); a syncpoint's minimum is its guest
//! value, read when needed.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use lichii_gpu::host1x::Syncpoints;
use parking_lot::Mutex;

use super::NvFence;

/// Syncpoints nvdrv manages, as the Switch 1 driver does.
pub(crate) const MAX_SYNCPOINTS: u32 = 192;

/// Display vblank syncpoints, managed by the display in continuous mode.
const VBLANK_SYNCPOINTS: [u32; 2] = [26, 27];
/// Fixed syncpoints of the VIC, NVDEC and NVJPG channels.
const ENGINE_SYNCPOINTS: [u32; 3] = [0xC, 0x36, 0x37];

pub(crate) struct SyncpointTable {
    syncpoints: Arc<Syncpoints>,
    reserved: [AtomicBool; MAX_SYNCPOINTS as usize],
    client_managed: [AtomicBool; MAX_SYNCPOINTS as usize],
    allocation: Mutex<()>,
}

impl SyncpointTable {
    pub fn new(syncpoints: Arc<Syncpoints>) -> Self {
        let table = Self {
            syncpoints,
            reserved: std::array::from_fn(|_| AtomicBool::new(false)),
            client_managed: std::array::from_fn(|_| AtomicBool::new(false)),
            allocation: Mutex::new(()),
        };
        for id in VBLANK_SYNCPOINTS {
            table.reserve(id, true);
        }
        for id in ENGINE_SYNCPOINTS {
            table.reserve(id, false);
        }
        table
    }

    pub fn syncpoints(&self) -> &Arc<Syncpoints> {
        &self.syncpoints
    }

    fn reserve(&self, id: u32, client_managed: bool) {
        self.reserved[id as usize].store(true, Ordering::Release);
        self.client_managed[id as usize].store(client_managed, Ordering::Release);
    }

    /// Reserves the first free syncpoint for a channel. 0 (logged) when every one is taken.
    pub fn allocate(&self) -> u32 {
        let _allocation = self.allocation.lock();
        let Some(id) = (1..MAX_SYNCPOINTS).find(|&id| !self.is_allocated(id)) else {
            tracing::error!("no free syncpoint");
            return 0;
        };
        self.reserve(id, false);
        id
    }

    pub fn free(&self, id: u32) {
        if let Some(reserved) = self.reserved.get(id as usize) {
            reserved.store(false, Ordering::Release);
        }
    }

    pub fn is_allocated(&self, id: u32) -> bool {
        self.reserved
            .get(id as usize)
            .is_some_and(|reserved| reserved.load(Ordering::Acquire))
    }

    /// The guest value: what the work submitted so far has reached.
    pub fn min(&self, id: u32) -> u32 {
        self.syncpoints.guest().value(id).unwrap_or(0)
    }

    pub fn max(&self, id: u32) -> u32 {
        self.syncpoints.max(id).unwrap_or(0)
    }

    /// Whether `fence` was reached. A fence on a syncpoint nvdrv did not hand out never is; for
    /// channel syncpoints a threshold past the maximum counts as reached, as host1x compares.
    pub fn is_signalled(&self, fence: NvFence) -> bool {
        let id = fence.id as u32;
        if !self.is_allocated(id) {
            return false;
        }
        let min = self.min(id);
        if self.client_managed[id as usize].load(Ordering::Acquire) {
            min.wrapping_sub(fence.value) as i32 >= 0
        } else {
            self.max(id).wrapping_sub(fence.value) >= min.wrapping_sub(fence.value)
        }
    }

    /// The fence the syncpoint reaches once every increment reserved so far ran.
    pub fn fence(&self, id: u32) -> NvFence {
        NvFence {
            id: id as i32,
            value: self.max(id),
        }
    }
}
