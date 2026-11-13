//! `/dev/nvhost-gpu`: one GPU channel. GPFIFO submissions go to [`Gpu::submit`](lichii_gpu::Gpu),
//! which reserves their syncpoint increments in host1x and returns the fence they reach.
//!
//! The GPU channel is created once the channel has a GPFIFO and an address space, in whichever
//! order the client sets them up. Switch 2 NvRm calls ioctls Switch 1 never had (0x1F, 0x20 and
//! more, NX2_NOTES row 26): they succeed and return their input.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_gpu::backend::ChannelId;
use lichii_gpu::channel::{ChannelConfig, GpEntry, Submission, SyncpointFence, SyncpointIncrement};
use lichii_gpu::gmmu::Gmmu;
use lichii_kernel::Event;
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use super::ioctl::{Args, Variant, items, read_struct, write_struct};
use super::{Client, NvFence, NvResult, Nvdrv};

const FLAG_FENCE_WAIT: u32 = 1 << 0;
const FLAG_FENCE_INCREMENT: u32 = 1 << 1;
const FLAG_SUPPRESS_WFI: u32 = 1 << 4;
const FLAG_INCREMENT_VALUE: u32 = 1 << 8;

const CLASS_2D: u32 = 0x902D;
const CLASS_3D: u32 = 0xB197;
const CLASS_COMPUTE: u32 = 0xB1C0;
const CLASS_KEPLER: u32 = 0xA140;
const CLASS_DMA: u32 = 0xB0B5;
const CLASS_GPFIFO: u32 = 0xB06F;
/// Object context slots, by Maxwell class.
const OBJECT_CLASSES: [u32; 6] = [
    CLASS_2D,
    CLASS_3D,
    CLASS_COMPUTE,
    CLASS_KEPLER,
    CLASS_DMA,
    CLASS_GPFIFO,
];

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SubmitGpfifo {
    address: u64,
    num_entries: u32,
    flags: u32,
    fence: NvFence,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct AllocGpfifoEx {
    num_entries: u32,
    num_jobs: u32,
    flags: u32,
    fence_out: NvFence,
    reserved: [u32; 3],
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct AllocObjCtx {
    class_num: u32,
    flags: u32,
    obj_id: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ZcullBind {
    gpu_va: u64,
    mode: u32,
    _pad: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct SetErrorNotifier {
    offset: u64,
    size: u64,
    mem: u32,
    _pad: u32,
}

#[derive(Default)]
struct ChannelState {
    user_data: u64,
    objects: [bool; OBJECT_CLASSES.len()],
    gmmu: Option<Arc<Gmmu>>,
    gpfifo: bool,
    channel: Option<ChannelId>,
}

pub(crate) struct GpuChannel {
    nvdrv: Arc<Nvdrv>,
    memory: Arc<AddressSpace>,
    syncpoint: u32,
    state: Mutex<ChannelState>,
    sm_exception_breakpoint_int: Arc<Event>,
    sm_exception_breakpoint_pause: Arc<Event>,
    error_notifier: Arc<Event>,
    nx2_events: Mutex<HashMap<u32, Arc<Event>>>,
}

impl GpuChannel {
    pub fn new(nvdrv: Arc<Nvdrv>, client: &Client) -> Self {
        Self {
            syncpoint: nvdrv.syncpoints.allocate(),
            nvdrv,
            memory: client.memory().clone(),
            state: Mutex::default(),
            sm_exception_breakpoint_int: Arc::new(Event::new()),
            sm_exception_breakpoint_pause: Arc::new(Event::new()),
            error_notifier: Arc::new(Event::new()),
            nx2_events: Mutex::default(),
        }
    }

    pub fn ioctl(&self, args: &mut Args) -> Option<NvResult> {
        Some(match (args.variant, args.number()) {
            (Variant::One, (0, 0x3)) => args.fixed(|waitbase: &mut [u32; 2]| {
                waitbase[1] = 0;
                NvResult::Success
            }),
            (Variant::One, (b'H', 0x1)) => args.fixed(|fd: &mut i32| {
                tracing::debug!(fd, "channel nvmap fd");
                NvResult::Success
            }),
            (Variant::One, (b'H', 0x3)) => args.fixed(|timeout: &mut u32| {
                tracing::debug!(timeout, "channel timeout");
                NvResult::Success
            }),
            (Variant::One, (b'H', 0x8)) => {
                let entries = args
                    .input
                    .get(size_of::<SubmitGpfifo>()..)
                    .unwrap_or_default();
                self.submit(args, Some(entries))
            }
            (Variant::One, (b'H', 0x9)) => args.fixed(|p| self.alloc_obj_ctx(p)),
            (Variant::One, (b'H', 0xb)) => args.fixed(|zcull: &mut ZcullBind| {
                tracing::debug!(gpu_va = zcull.gpu_va, mode = zcull.mode, "zcull bind");
                NvResult::Success
            }),
            (Variant::One, (b'H', 0xc)) => args.fixed(|p: &mut SetErrorNotifier| {
                tracing::warn!(
                    offset = p.offset,
                    size = p.size,
                    mem = p.mem,
                    "(stubbed) SetErrorNotifier"
                );
                NvResult::Success
            }),
            (Variant::One, (b'H', 0xd)) => {
                args.fixed(|priority: &mut u32| self.set_priority(*priority))
            }
            (Variant::One, (b'H', 0x18 | 0x1a)) => args.fixed(|p| self.alloc_gpfifo(p)),
            (Variant::One, (b'H', 0x1b)) => self.submit(args, None),
            (Variant::Two, (b'H', 0x1b)) => {
                let entries = args.inline_input;
                self.submit(args, Some(entries))
            }
            (Variant::One, (b'H', 0x1d)) => {
                args.fixed(|timeslice: &mut u32| self.set_timeslice(*timeslice))
            }
            (Variant::One, (b'G', 0x14)) => args.fixed(|data: &mut u64| {
                self.state.lock().user_data = *data;
                NvResult::Success
            }),
            (Variant::One, (b'G', 0x15)) => args.fixed(|data: &mut u64| {
                *data = self.state.lock().user_data;
                NvResult::Success
            }),
            _ if self.nvdrv.nx2 => {
                tracing::warn!(
                    ioctl = format_args!("{:#010x}", args.ioctl.0),
                    variant = ?args.variant,
                    "(stubbed) Switch 2 nvhost-gpu ioctl returns its input"
                );
                args.echo();
                NvResult::Success
            }
            _ => return None,
        })
    }

    /// Uses `gmmu` for the work submitted from now on (nvhost-as-gpu BindChannel).
    pub fn bind_address_space(&self, gmmu: Arc<Gmmu>) {
        let mut state = self.state.lock();
        if let Some(channel) = state.channel {
            self.nvdrv.gpu.bind_address_space(channel, gmmu.clone());
        }
        state.gmmu = Some(gmmu);
        if state.gpfifo {
            self.channel(&mut state);
        }
    }

    /// The GPU channel, created if the channel has an address space.
    fn channel(&self, state: &mut ChannelState) -> Option<ChannelId> {
        if state.channel.is_none() {
            let gmmu = state.gmmu.clone()?;
            let config = ChannelConfig::for_console(gmmu, self.nvdrv.console());
            state.channel = Some(self.nvdrv.gpu.create_channel(config));
        }
        state.channel
    }

    fn alloc_gpfifo(&self, params: &mut AllocGpfifoEx) -> NvResult {
        let mut state = self.state.lock();
        if state.channel.is_some() {
            return NvResult::AlreadyAllocated;
        }
        state.gpfifo = true;
        self.channel(&mut state);
        params.fence_out = self.nvdrv.syncpoints.fence(self.syncpoint);
        NvResult::Success
    }

    /// Submits GPFIFO entries, from `entries` or, without them, from the client's memory at the
    /// address the arguments give.
    fn submit(&self, args: &mut Args, entries: Option<&[u8]>) -> NvResult {
        let mut params: SubmitGpfifo = read_struct(args.input);
        let count = params.num_entries as usize;
        let entries: Vec<GpEntry> = match entries {
            Some(bytes) if bytes.len() / size_of::<u64>() < count => {
                tracing::error!(count, "GPFIFO submission with fewer entries than it claims");
                return NvResult::InvalidSize;
            }
            Some(bytes) => items::<u64>(bytes).take(count).map(GpEntry).collect(),
            None => {
                let mut entries = Vec::with_capacity(count);
                let mut chunk = [0u64; 64];
                let mut address = params.address;
                while entries.len() < count {
                    let words = &mut chunk[..(count - entries.len()).min(64)];
                    if let Err(fault) = self
                        .memory
                        .read_bytes(address, bytemuck::cast_slice_mut(words))
                    {
                        tracing::error!(%fault, "GPFIFO entries are not readable");
                        return NvResult::InvalidAddress;
                    }
                    address += size_of_val(words) as u64;
                    entries.extend(words.iter().copied().map(GpEntry));
                }
                entries
            }
        };
        let result = self.submit_entries(&mut params, entries);
        args.echo();
        write_struct(args.output, &params);
        result
    }

    fn submit_entries(&self, params: &mut SubmitGpfifo, entries: Vec<GpEntry>) -> NvResult {
        let table = &self.nvdrv.syncpoints;
        let flags = params.flags;
        let increment_value = flags & FLAG_INCREMENT_VALUE != 0;
        let fence_increment = flags & FLAG_FENCE_INCREMENT != 0;
        let mut state = self.state.lock();
        let Some(channel) = self.channel(&mut state) else {
            tracing::error!("GPFIFO submission before an address space was bound");
            return NvResult::InvalidState;
        };
        let mut submission = Submission::new(entries);
        if flags & FLAG_FENCE_WAIT != 0 {
            if increment_value {
                return NvResult::BadParameter;
            }
            if !table.is_signalled(params.fence) {
                submission.wait = Some(SyncpointFence {
                    id: params.fence.id as u32,
                    value: params.fence.value,
                });
            }
        }
        let id = self.syncpoint;
        let value = if increment_value {
            params.fence.value
        } else {
            0
        };
        if self.nvdrv.nx2 && !fence_increment {
            submission.increment =
                (value > 0).then(|| SyncpointIncrement::increment_value(id, value));
        } else {
            if value > 0 && table.syncpoints().increase_max(id, value).is_err() {
                return NvResult::BadParameter;
            }
            if fence_increment {
                let suppress_wfi = flags & FLAG_SUPPRESS_WFI != 0;
                submission.increment = Some(SyncpointIncrement::fence_increment(id, suppress_wfi));
            }
        }
        let fence = self.nvdrv.gpu.submit(channel, submission);
        params.fence = NvFence {
            id: id as i32,
            value: fence.map_or_else(|| table.max(id), |fence| fence.value),
        };
        params.flags = 0;
        NvResult::Success
    }

    fn alloc_obj_ctx(&self, params: &mut AllocObjCtx) -> NvResult {
        let mut state = self.state.lock();
        if params.flags != 0 {
            tracing::warn!(
                flags = params.flags,
                class = params.class_num,
                "object context flags ignored"
            );
            params.flags = 0;
        }
        if self.nvdrv.nx2 {
            let mapped = match params.class_num {
                0xC797 | 0xC697 | 0xC597 => CLASS_3D,
                0xC7C0 | 0xC6C0 | 0xC5C0 => CLASS_COMPUTE,
                0xC7B5 | 0xC6B5 | 0xC5B5 => CLASS_DMA,
                0xC56F | 0xC76F => CLASS_GPFIFO,
                0 => [CLASS_3D, CLASS_COMPUTE, CLASS_DMA, CLASS_2D, CLASS_KEPLER]
                    .into_iter()
                    .find(|&class| !state.objects[slot(class).expect("Maxwell class")])
                    .unwrap_or(0),
                class => class,
            };
            if mapped != params.class_num {
                tracing::debug!(
                    from = params.class_num,
                    to = mapped,
                    "Switch 2 object class"
                );
                params.class_num = mapped;
            }
        }
        let Some(index) = slot(params.class_num) else {
            tracing::error!(
                class = params.class_num,
                "object context of an unknown class"
            );
            return NvResult::BadParameter;
        };
        if std::mem::replace(&mut state.objects[index], true) {
            return NvResult::AlreadyAllocated;
        }
        NvResult::Success
    }

    /// Channel priorities map to timeslices; the GPU does not schedule by them.
    fn set_priority(&self, priority: u32) -> NvResult {
        let timeslice = match priority {
            0x32 => 1300,
            0x64 => 2600,
            0x96 => 5200,
            _ => return NvResult::BadParameter,
        };
        tracing::debug!(priority, timeslice, "channel priority");
        NvResult::Success
    }

    fn set_timeslice(&self, timeslice: u32) -> NvResult {
        if !(1000..=5000).contains(&timeslice) {
            return NvResult::BadParameter;
        }
        tracing::debug!(timeslice, "channel timeslice");
        NvResult::Success
    }

    pub fn query_event(&self, id: u32) -> Option<Arc<Event>> {
        match id {
            1 => Some(self.sm_exception_breakpoint_int.clone()),
            2 => Some(self.sm_exception_breakpoint_pause.clone()),
            3 => Some(self.error_notifier.clone()),
            _ if self.nvdrv.nx2 => {
                tracing::warn!(id, "(stubbed) Switch 2 channel event, never signaled");
                Some(self.nx2_events.lock().entry(id).or_default().clone())
            }
            _ => None,
        }
    }
}

fn slot(class: u32) -> Option<usize> {
    OBJECT_CLASSES.iter().position(|&c| c == class)
}

impl Drop for GpuChannel {
    fn drop(&mut self) {
        if let Some(channel) = self.state.get_mut().channel {
            self.nvdrv.gpu.destroy_channel(channel);
        }
        self.nvdrv.syncpoints.free(self.syncpoint);
    }
}
