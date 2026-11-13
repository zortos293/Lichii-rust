//! `/dev/nvhost-nvdec`, `/dev/nvhost-vic` and `/dev/nvhost-nvjpg`: host1x engine channels.
//!
//! A submission reserves its syncpoint increments and hands its command buffers to the channel's
//! [`EngineChannel`], which counts the increments as its stream reaches them. Map-buffer pins
//! return the nvmap device address, as there is no separate host1x IOVA space.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_gpu::host1x::{EngineChannel, VideoEngine};
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
struct CommandBuffer {
    memory_id: u32,
    offset: u32,
    word_count: u32,
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

const RELOC_SIZE: usize = 16;
const CLOCK_RATE: u32 = 614_400_000;

pub(crate) struct Host1xClient {
    nvdrv: Arc<Nvdrv>,
    name: &'static str,
    syncpoint: u32,
    pinned: Mutex<Vec<u32>>,
    channel: EngineChannel,
}

impl Host1xClient {
    pub fn new(nvdrv: Arc<Nvdrv>, path: &str) -> Self {
        let (name, engine) = match path {
            "/dev/nvhost-nvdec" => ("/dev/nvhost-nvdec", VideoEngine::NvDec),
            "/dev/nvhost-vic" => ("/dev/nvhost-vic", VideoEngine::Vic),
            _ => ("/dev/nvhost-nvjpg", VideoEngine::NvJpg),
        };
        tracing::info!(device = name, "host1x engine channel opened");
        Self {
            syncpoint: nvdrv.syncpoints.allocate(),
            channel: EngineChannel::open(engine, nvdrv.gpu.host1x().syncpoints.clone()),
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

    /// Reserves each syncpoint increment, writes the thresholds back and queues the command
    /// buffers on the channel.
    fn submit(&self, args: &mut Args) -> NvResult {
        let params: Submit = read_struct(args.input);
        args.echo();
        let syncpoints = self.nvdrv.syncpoints.syncpoints();
        let buffers_at = size_of::<Submit>();
        let relocations_at =
            buffers_at + params.cmd_buffer_count as usize * size_of::<CommandBuffer>();
        let increments_at =
            relocations_at + params.relocation_count as usize * (RELOC_SIZE + size_of::<u32>());
        let thresholds_at =
            increments_at + params.syncpoint_count as usize * size_of::<SyncptIncr>();
        let buffers = args
            .input
            .get(buffers_at..relocations_at)
            .unwrap_or_default();
        let mut command_buffers = Vec::new();
        for buffer in items::<CommandBuffer>(buffers) {
            let mut words = vec![0u32; buffer.word_count as usize];
            if let Err(error) = self.nvdrv.nvmap.read(
                buffer.memory_id,
                u64::from(buffer.offset),
                bytemuck::cast_slice_mut(&mut words),
            ) {
                tracing::warn!(
                    device = self.name,
                    ?error,
                    "host1x command buffer unreadable"
                );
                return NvResult::InvalidState;
            }
            command_buffers.push(words);
        }
        let increments = args
            .input
            .get(increments_at..thresholds_at)
            .unwrap_or_default();
        for (index, increment) in items::<SyncptIncr>(increments).enumerate() {
            let Ok(threshold) = syncpoints.increase_max(increment.id, increment.increments) else {
                return NvResult::BadParameter;
            };
            if index < params.fence_count as usize {
                let at = thresholds_at + index * size_of::<u32>();
                write_struct(args.output.get_mut(at..).unwrap_or_default(), &threshold);
            }
        }
        for words in command_buffers {
            self.channel.submit(words);
        }
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
        self.channel.close();
        for handle in self.pinned.get_mut().drain(..) {
            self.nvdrv.nvmap.unpin(handle);
        }
        self.nvdrv.syncpoints.retire(self.syncpoint);
    }
}
