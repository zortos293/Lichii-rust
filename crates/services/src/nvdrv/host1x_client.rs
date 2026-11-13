//! `/dev/nvhost-nvdec`, `/dev/nvhost-vic` and `/dev/nvhost-nvjpg`: host1x engine channels.
//!
//! There are no video engines yet, so these are stubs: a submission's syncpoint increments are
//! reserved and counted at once, and command buffers are not executed. Map-buffer pins return the
//! nvmap device address, as there is no separate host1x IOVA space.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;

use super::ioctl::{Args, Variant, items, read_struct, write_struct};
use super::{NvResult, Nvdrv};

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Submit {
    cmd_buffer_count: u32,
    relocation_count: u32,
    syncpoint_count: u32,
    fence_count: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SyncptIncr {
    id: u32,
    increments: u32,
    _unknown: [u32; 3],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MapBuffer {
    num_entries: u32,
    data_address: u32,
    attach_host_ch_das: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct MapBufferEntry {
    handle: u32,
    address: u32,
}

const COMMAND_BUFFER_SIZE: usize = 12;
const RELOC_SIZE: usize = 16;
const CLOCK_RATE: u32 = 614_400_000;

pub(crate) struct Host1xClient {
    nvdrv: Arc<Nvdrv>,
    name: &'static str,
    syncpoint: u32,
    pinned: Mutex<Vec<u32>>,
}

impl Host1xClient {
    pub fn new(nvdrv: Arc<Nvdrv>, path: &str) -> Self {
        let name = match path {
            "/dev/nvhost-nvdec" => "/dev/nvhost-nvdec",
            "/dev/nvhost-vic" => "/dev/nvhost-vic",
            _ => "/dev/nvhost-nvjpg",
        };
        tracing::info!(device = name, "host1x engine channel opened (stub)");
        Self {
            syncpoint: nvdrv.syncpoints.allocate(),
            nvdrv,
            name,
            pinned: Mutex::default(),
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn ioctl(&self, args: &mut Args) -> Option<NvResult> {
        if args.variant != Variant::One {
            return None;
        }
        Some(match args.number() {
            (0, 0x1) => self.submit(args),
            (0, 0x2) => args.fixed(|p: &mut [u32; 2]| {
                p[1] = self.syncpoint;
                NvResult::Success
            }),
            (0, 0x3) => args.fixed(|p: &mut [u32; 2]| {
                p[1] = 0;
                NvResult::Success
            }),
            (0, 0x7) => NvResult::Success,
            (0, 0x9) => self.map_buffer(args),
            (0, 0xa) => self.unmap_buffer(args),
            (0, 0x23) => args.fixed(|p: &mut [u32; 2]| {
                *p = [CLOCK_RATE, 0];
                NvResult::Success
            }),
            (b'H', 0x1) => NvResult::Success,
            _ => return None,
        })
    }

    /// Reserves each syncpoint increment and counts it at once, and writes the thresholds back.
    fn submit(&self, args: &mut Args) -> NvResult {
        let params: Submit = read_struct(args.input);
        args.echo();
        let syncpoints = self.nvdrv.syncpoints.syncpoints();
        let increments_at = size_of::<Submit>()
            + params.cmd_buffer_count as usize * COMMAND_BUFFER_SIZE
            + params.relocation_count as usize * (RELOC_SIZE + size_of::<u32>());
        let thresholds_at =
            increments_at + params.syncpoint_count as usize * size_of::<SyncptIncr>();
        let increments = args
            .input
            .get(increments_at..thresholds_at)
            .unwrap_or_default();
        for (index, increment) in items::<SyncptIncr>(increments).enumerate() {
            let Ok(threshold) = syncpoints.increase_max(increment.id, increment.increments) else {
                return NvResult::BadParameter;
            };
            for _ in 0..increment.increments {
                let _ = syncpoints.host().increment(increment.id);
                let _ = syncpoints.guest().increment(increment.id);
            }
            if index < params.fence_count as usize {
                let at = thresholds_at + index * size_of::<u32>();
                write_struct(args.output.get_mut(at..).unwrap_or_default(), &threshold);
            }
        }
        tracing::debug!(device = self.name, "(stubbed) host1x submission");
        NvResult::Success
    }

    fn map_buffer(&self, args: &mut Args) -> NvResult {
        let params: MapBuffer = read_struct(args.input);
        args.echo();
        let entries = args.input.get(size_of::<MapBuffer>()..).unwrap_or_default();
        let mut pinned = self.pinned.lock();
        for (index, entry) in items::<MapBufferEntry>(entries)
            .take(params.num_entries as usize)
            .enumerate()
        {
            let address = match self.nvdrv.nvmap.pin(entry.handle) {
                Ok(daddr) => {
                    pinned.push(entry.handle);
                    u32::try_from(daddr).unwrap_or_else(|_| {
                        tracing::warn!(daddr, "host1x pin above 4 GiB truncated");
                        daddr as u32
                    })
                }
                Err(_) => 0,
            };
            let at = size_of::<MapBuffer>() + index * size_of::<MapBufferEntry>();
            let entry = MapBufferEntry {
                handle: entry.handle,
                address,
            };
            write_struct(args.output.get_mut(at..).unwrap_or_default(), &entry);
        }
        NvResult::Success
    }

    fn unmap_buffer(&self, args: &mut Args) -> NvResult {
        let params: MapBuffer = read_struct(args.input);
        let entries = args.input.get(size_of::<MapBuffer>()..).unwrap_or_default();
        let mut pinned = self.pinned.lock();
        for entry in items::<MapBufferEntry>(entries).take(params.num_entries as usize) {
            if let Some(index) = pinned.iter().position(|&h| h == entry.handle) {
                pinned.swap_remove(index);
                self.nvdrv.nvmap.unpin(entry.handle);
            }
        }
        NvResult::Success
    }
}

impl Drop for Host1xClient {
    fn drop(&mut self) {
        for handle in self.pinned.get_mut().drain(..) {
            self.nvdrv.nvmap.unpin(handle);
        }
        self.nvdrv.syncpoints.free(self.syncpoint);
    }
}
