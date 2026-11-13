//! `nn::gpu::sdk`'s session (gsv 497): job-slot events, graphics device ids, the GPU VA
//! partitions, and the Switch 2 GPU memory requests (sub-session commands 101-216) on the GPU's
//! [`Nx2GpuMemory`] registry.
//!
//! Requests name their memory by inclusive [first, last] ranges and complete through a job slot:
//! the SDK waits on the slot's event (12) after the reply. Maps complete at once; unmaps and
//! destroys complete once the GPU has run everything queued before them, as the console's
//! graphics server does, because the title reuses the memory right after.

use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_gpu::Gpu;
use lichii_gpu::nx2_memory::{GpuMemoryHandle, Nx2GpuMemory, Nx2MemoryError, Unmapped};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use super::{handshake, push_readable, push_session};
use crate::{Service, ServiceContext};

/// Job slot event ids are taken modulo this; 11 reports [0, `SLOTS` - 1].
const SLOTS: u32 = 32;
/// Graphics device ids (41) are 1..`DEVICES`.
const DEVICES: usize = 64;
/// The GPU VA range NVN reserves for the user partition.
const VA_START: u64 = 0x800_0000;
const VA_END: u64 = 0xc0_0000_0000;
/// `nn::gpu::sdk`'s SparseMapRequest (216), 40 bytes.
const SPARSE_REQUEST_SIZE: usize = 40;

/// gsv 497, and the sub-session its command 22 returns, which shares the job slots.
pub(super) struct GpuSession {
    context: Arc<ServiceContext>,
    state: Arc<State>,
}

struct State {
    slots: [Arc<Event>; SLOTS as usize],
    devices: Mutex<[bool; DEVICES]>,
}

impl GpuSession {
    pub fn new(context: &Arc<ServiceContext>) -> Self {
        Self {
            context: context.clone(),
            state: Arc::new(State {
                slots: std::array::from_fn(|_| Arc::new(Event::new())),
                devices: Mutex::new([false; DEVICES]),
            }),
        }
    }

    fn slot(&self, id: u32) -> &Arc<Event> {
        &self.state.slots[(id % SLOTS) as usize]
    }

    fn memory(&self) -> Option<&Nx2GpuMemory> {
        self.context.gpu.nx2_memory()
    }

    /// Signals a job slot once the GPU has run everything queued so far.
    fn signal_after_gpu(&self, id: u32) {
        let slot = self.slot(id).clone();
        self.context.gpu.run_in_order(move |_| slot.signal());
    }

    fn create_device(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let mut devices = self.state.devices.lock();
        let Some(id) = (1..DEVICES).find(|&id| !devices[id]) else {
            return lichii_kernel::result::OUT_OF_RESOURCE;
        };
        devices[id] = true;
        ctx.push(id as u32);
        ResultCode::SUCCESS
    }

    /// 101(first u64, last u64, flags): makes process memory GPU accessible at GPU VA == CPU VA.
    /// Reply word 3 is the GpuMemoryHandle later requests name.
    fn create_memory(&self, ctx: &mut HleRequestContext, memory: &Nx2GpuMemory) -> ResultCode {
        let words = Words::command(ctx);
        let (first, last) = (words.u64(2), words.u64(4));
        let Some(size) = last.checked_sub(first).map(|n| n + 1) else {
            tracing::warn!("nx2:gpu 101: empty range [{first:#x}, {last:#x}]");
            return lichii_kernel::result::INVALID_SIZE;
        };
        let space: &AddressSpace = self.context.process.memory();
        match memory.create_gpu_accessible_memory(space, first, size) {
            Ok((handle, unmapped)) => {
                unmap(&self.context.gpu, &unmapped);
                ctx.push(handle.raw());
                ResultCode::SUCCESS
            }
            Err(err) => {
                tracing::warn!("nx2:gpu 101 [{first:#x}, {last:#x}]: {err}");
                lichii_kernel::result::OUT_OF_RESOURCE
            }
        }
    }

    /// 216: an A buffer of SparseMapRequests, a B buffer that gets the kind applied to each.
    /// A request naming memory maps it; memory handle 0 makes the range read zero again.
    fn sparse_map(&self, ctx: &mut HleRequestContext, memory: &Nx2GpuMemory) -> ResultCode {
        let slot = Words::command(ctx).u32(14);
        let requests = ctx.read_buffer(0).unwrap_or_default();
        let count = requests.len() / SPARSE_REQUEST_SIZE;
        let mut kinds = Vec::with_capacity(count);
        for request in requests.as_chunks::<SPARSE_REQUEST_SIZE>().0 {
            let words = Words::of(request);
            let (va, size) = (words.u64(0), words.u64(2));
            let kind = request[19];
            kinds.push(kind);
            let Some(last) = size.checked_sub(1).and_then(|n| va.checked_add(n)) else {
                continue;
            };
            let result = match GpuMemoryHandle::new(words.u32(6)) {
                Some(handle) => memory.normal_map_4k(va, last, handle, words.u64(8)),
                None => memory.unmap_4k(va, last),
            };
            self.finish("216", result);
        }
        if count != 0 && ctx.writable_buffer(0).is_some() {
            let _ = ctx.write_buffer(0, &kinds);
        }
        self.signal_after_gpu(slot);
        ctx.push(count as u64);
        ResultCode::SUCCESS
    }

    /// Passes a request's unmapped ranges to the GPU, or logs why it failed. Failures still
    /// complete the job, as the reference does: the SDK aborts on an error reply.
    fn finish(&self, request: &str, result: Result<Vec<Unmapped>, Nx2MemoryError>) {
        match result {
            Ok(unmapped) => unmap(&self.context.gpu, &unmapped),
            Err(err) => tracing::warn!("nx2:gpu {request}: {err}"),
        }
    }
}

fn unmap(gpu: &Gpu, unmapped: &[Unmapped]) {
    for range in unmapped {
        gpu.unmap_region(range.range.daddr, range.range.size);
    }
}

/// Little-endian words, indexed from `first`. Missing words read as zero.
struct Words<'a> {
    data: &'a [u8],
    first: usize,
}

impl<'a> Words<'a> {
    /// A TIPC request's data, indexed as the reference's command buffer is: from word 2.
    fn command(ctx: &'a HleRequestContext) -> Self {
        Self {
            data: ctx.input(),
            first: 2,
        }
    }

    fn of(data: &'a [u8]) -> Self {
        Self { data, first: 0 }
    }

    fn u32(&self, index: usize) -> u32 {
        let at = (index - self.first) * 4;
        self.data
            .get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn u64(&self, index: usize) -> u64 {
        u64::from(self.u32(index)) | (u64::from(self.u32(index + 1)) << 32)
    }
}

impl Service for GpuSession {
    fn name(&self) -> &str {
        "nx2:gpu"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command = ctx.command();
        let words = Words::command(ctx);
        Some(match command {
            0 => handshake(ctx),
            11 => {
                ctx.push([0, SLOTS - 1]);
                ResultCode::SUCCESS
            }
            12 => {
                let slot = self.slot(words.u32(2)).clone();
                push_readable(ctx, &slot)
            }
            21 => {
                self.slot(words.u32(2)).signal();
                ResultCode::SUCCESS
            }
            22 => push_session(
                &self.context,
                ctx,
                GpuSession {
                    context: self.context.clone(),
                    state: self.state.clone(),
                },
            ),
            33 => {
                ctx.push(1u32);
                ResultCode::SUCCESS
            }
            41 => self.create_device(ctx),
            42 => {
                let id = (words.u32(2) & 0xff) as usize;
                let existed = id < DEVICES && std::mem::take(&mut self.state.devices.lock()[id]);
                if existed {
                    ResultCode::SUCCESS
                } else {
                    lichii_kernel::result::NOT_FOUND
                }
            }
            201 => {
                ctx.push([VA_START, VA_END - VA_START]);
                ResultCode::SUCCESS
            }
            202 => {
                ctx.push([u64::MAX, 0]);
                ResultCode::SUCCESS
            }
            203 => {
                ctx.push([1, VA_START, VA_END - VA_START]);
                ResultCode::SUCCESS
            }
            101 | 102 | 211..=216 => {
                let Some(memory) = self.memory() else {
                    tracing::warn!("nx2:gpu {command}: no Switch 2 GPU memory for this title");
                    return Some(lichii_kernel::result::NOT_FOUND);
                };
                let range = (words.u64(3), words.u64(5));
                match command {
                    101 => self.create_memory(ctx, memory),
                    102 => {
                        match GpuMemoryHandle::new(words.u32(2)) {
                            Some(handle) => self.finish("102", memory.destroy(handle)),
                            None => tracing::warn!("nx2:gpu 102: memory handle 0"),
                        }
                        self.signal_after_gpu(words.u32(3));
                        ResultCode::SUCCESS
                    }
                    211 => {
                        match GpuMemoryHandle::new(words.u32(7)) {
                            Some(handle) => self.finish(
                                "211",
                                memory.normal_map_4k(range.0, range.1, handle, words.u64(8)),
                            ),
                            None => tracing::warn!("nx2:gpu 211: memory handle 0"),
                        }
                        self.slot(words.u32(17)).signal();
                        ResultCode::SUCCESS
                    }
                    212 => {
                        let wanted = (words.u32(7) & 0xff) as u8;
                        if let Err(err) = memory.modify_kind(range.0, range.1, wanted) {
                            tracing::warn!("nx2:gpu 212: {err}");
                        }
                        self.slot(words.u32(10)).signal();
                        ctx.push(u32::from(wanted));
                        ResultCode::SUCCESS
                    }
                    213 => {
                        self.finish("213", memory.unmap_4k(range.0, range.1));
                        self.signal_after_gpu(words.u32(8));
                        ResultCode::SUCCESS
                    }
                    214 => {
                        let size = range.1.wrapping_sub(range.0).wrapping_add(1);
                        self.finish("214", memory.create_sparse_region(range.0, size));
                        self.slot(words.u32(11)).signal();
                        ResultCode::SUCCESS
                    }
                    215 => {
                        let size = range.1.wrapping_sub(range.0).wrapping_add(1);
                        let result = match memory.destroy_sparse_region(range.0, size) {
                            Err(Nx2MemoryError::NoSuchRegion { .. }) => {
                                memory.unmap_4k(range.0, range.1)
                            }
                            result => result,
                        };
                        self.finish("215", result);
                        self.signal_after_gpu(words.u32(8));
                        ResultCode::SUCCESS
                    }
                    _ => self.sparse_map(ctx, memory),
                }
            }
            _ => return None,
        })
    }
}
