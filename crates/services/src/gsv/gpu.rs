//! `nn::gpu::sdk`'s session (gsv 497): job-slot events, graphics device ids, the GPU VA
//! partitions, and the Switch 2 GPU memory requests (sub-session commands 101-216) on the GPU's
//! [`Nx2GpuMemory`] registry.
//!
//! Requests name their memory by inclusive [first, last] ranges and complete through a job slot:
//! the SDK waits on the slot's event (12) after the reply. Every request changes the registry at
//! once, in the order the title sent them, as the reference does. An unmap or destroy hands what it
//! removed to the renderer after the GPU work queued before it, and completes its slot once the
//! host GPU finished that work, because the title reuses the memory right after.
//!
//! A region 214 creates without the sparse flag is GPU-only storage the client renders into
//! (NX2_NOTES row 57): the server backs it with memory of its own in the alias code region, made
//! GPU accessible and mapped at the region. Tiles mapped into it and unmapped again show that
//! memory again.

use std::collections::BTreeMap;
use std::sync::Arc;

use lichii_common::{DAddr, GpuVAddr, PAGE_MASK, ResultCode, VAddr};
use lichii_gpu::Gpu;
use lichii_gpu::backend::Rasterizer;
use lichii_gpu::gmmu::MappedRange;
use lichii_gpu::nx2_memory::{GpuMemoryHandle, Nx2GpuMemory, Nx2MemoryError, Unmapped};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory::MemoryState;
use lichii_memory::{AddressSpace, MemoryPermission};
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
/// Word 7 of 214: the region is sparse (texture arrays, tile pools) rather than GPU-only storage.
const SPARSE_REGION: u32 = 0x200;

/// gsv 497, and the sub-session its command 22 returns, which shares the job slots.
pub(super) struct GpuSession {
    context: Arc<ServiceContext>,
    state: Arc<State>,
}

struct State {
    slots: [Arc<Event>; SLOTS as usize],
    devices: Mutex<[bool; DEVICES]>,
    /// GPU-only regions by GPU address, and the memory backing each.
    gpu_only: Mutex<BTreeMap<GpuVAddr, Backing>>,
    /// Backings no GPU range reaches any more, once the GPU is past them. The service thread
    /// frees them: kernel memory calls never run on a GPU thread.
    released: Mutex<Vec<Backing>>,
}

#[derive(Clone, Copy, Debug)]
struct Backing {
    handle: GpuMemoryHandle,
    cpu_va: VAddr,
    size: u64,
}

impl State {
    /// Maps the parts of [`va`, +`size`) that lie in GPU-only regions back to their backing, and
    /// takes out of `unmapped` what that maps back to the same device memory: the renderer keeps
    /// what the GPU wrote there.
    fn restore_gpu_only(
        &self,
        memory: &Nx2GpuMemory,
        va: GpuVAddr,
        size: u64,
        unmapped: &mut Vec<Unmapped>,
    ) {
        let end = va.saturating_add(size);
        for (&start, backing) in self.gpu_only.lock().range(..end) {
            let (lo, hi) = (va.max(start), end.min(start + backing.size));
            if lo >= hi {
                continue;
            }
            let Some(info) = memory.memory_info(backing.handle) else {
                continue;
            };
            match memory.normal_map_4k(lo, hi - 1, backing.handle, lo - start) {
                Ok(replaced) => {
                    keep_changed(unmapped, lo, hi, info.daddr + (lo - start));
                    unmapped.extend(replaced);
                }
                Err(err) => tracing::warn!("nx2:gpu: restoring GPU-only memory at {lo:#x}: {err}"),
            }
        }
    }
}

/// Removes from `unmapped` the parts of [`lo`, `hi`) that map to device memory from `daddr` on.
pub(super) fn keep_changed(unmapped: &mut Vec<Unmapped>, lo: GpuVAddr, hi: GpuVAddr, daddr: DAddr) {
    let mut kept = Vec::with_capacity(unmapped.len());
    for entry in unmapped.drain(..) {
        let MappedRange {
            gpu_va,
            daddr: from,
            size,
        } = entry.range;
        let (a, b) = (gpu_va.max(lo), (gpu_va + size).min(hi));
        if a >= b || from + (a - gpu_va) != daddr + (a - lo) {
            kept.push(entry);
            continue;
        }
        let piece = |start: GpuVAddr, end: GpuVAddr| Unmapped {
            space: entry.space,
            range: MappedRange {
                gpu_va: start,
                daddr: from + (start - gpu_va),
                size: end - start,
            },
        };
        if gpu_va < a {
            kept.push(piece(gpu_va, a));
        }
        if b < gpu_va + size {
            kept.push(piece(b, gpu_va + size));
        }
    }
    *unmapped = kept;
}

impl GpuSession {
    pub fn new(context: &Arc<ServiceContext>) -> Self {
        Self {
            context: context.clone(),
            state: Arc::new(State {
                slots: std::array::from_fn(|_| Arc::new(Event::new())),
                devices: Mutex::new([false; DEVICES]),
                gpu_only: Mutex::default(),
                released: Mutex::default(),
            }),
        }
    }

    fn slot(&self, id: u32) -> &Arc<Event> {
        &self.state.slots[(id % SLOTS) as usize]
    }

    fn memory(&self) -> Option<&Nx2GpuMemory> {
        self.context.gpu.nx2_memory()
    }

    /// After the GPU work queued before it, hands `unmapped` to the renderer, waits until the host
    /// GPU finished everything submitted, then runs `then`.
    fn after_gpu(&self, unmapped: Vec<Unmapped>, then: impl FnOnce(&State) + Send + 'static) {
        let state = self.state.clone();
        self.context
            .gpu
            .run_in_order(move |rasterizer: &mut dyn Rasterizer| {
                for range in &unmapped {
                    rasterizer.unmap_region(range.range.daddr, range.range.size);
                }
                rasterizer.wait_for_idle();
                then(&state);
            });
    }

    /// Completes a destructive request: signals job slot `slot` once the GPU is past `unmapped`.
    fn complete(&self, slot: u32, unmapped: Vec<Unmapped>) {
        let event = self.slot(slot).clone();
        self.after_gpu(unmapped, move |_| event.signal());
    }

    /// Frees the backings of destroyed GPU-only regions the GPU is past.
    fn free_released(&self) {
        let released = std::mem::take(&mut *self.state.released.lock());
        let page_table = self.context.process.page_table();
        for backing in released {
            if let Err(code) =
                page_table.unmap_and_free(backing.cpu_va, backing.size, MemoryState::IO_MEMORY)
            {
                tracing::warn!(
                    "nx2:gpu: freeing GPU-only memory at {:#x}: {code:?}",
                    backing.cpu_va
                );
            }
        }
    }

    /// 214 without the sparse flag: backs [`start`, +`size`) with server memory.
    fn create_gpu_only(&self, memory: &Nx2GpuMemory, start: GpuVAddr, size: u64) {
        let page_table = self.context.process.page_table();
        let region = page_table.regions().alias_code;
        let cpu_va = match page_table.map_new_in(
            region,
            size,
            MemoryState::IO_MEMORY,
            MemoryPermission::READ_WRITE,
        ) {
            Ok(cpu_va) => cpu_va,
            Err(code) => {
                tracing::error!(
                    "nx2:gpu 214: no memory for GPU-only region {start:#x}+{size:#x}: {code:?}"
                );
                return;
            }
        };
        let space: &AddressSpace = self.context.process.memory();
        let (handle, mut unmapped) = match memory.create_gpu_accessible_memory(space, cpu_va, size)
        {
            Ok(created) => created,
            Err(err) => {
                tracing::error!("nx2:gpu 214: GPU-only region {start:#x}+{size:#x}: {err}");
                let _ = page_table.unmap_and_free(cpu_va, size, MemoryState::IO_MEMORY);
                return;
            }
        };
        match memory.normal_map_4k(start, start + size - 1, handle, 0) {
            Ok(replaced) => unmapped.extend(replaced),
            Err(err) => {
                tracing::error!("nx2:gpu 214: GPU-only region {start:#x}+{size:#x}: {err}");
                unmapped.extend(logged("214", memory.destroy(handle)));
                unmap(&self.context.gpu, &unmapped);
                let _ = page_table.unmap_and_free(cpu_va, size, MemoryState::IO_MEMORY);
                return;
            }
        }
        unmap(&self.context.gpu, &unmapped);
        tracing::info!("nx2:gpu 214: GPU-only region {start:#x}+{size:#x} backed at {cpu_va:#x}");
        let backing = Backing {
            handle,
            cpu_va,
            size,
        };
        let old = self.state.gpu_only.lock().insert(start, backing);
        if let Some(old) = old {
            self.release(memory, old);
        }
    }

    /// Destroys a backing's GPU memory, and queues the backing for freeing once the GPU is past it.
    fn release(&self, memory: &Nx2GpuMemory, backing: Backing) {
        let unmapped = logged("releasing GPU-only memory", memory.destroy(backing.handle));
        self.after_gpu(unmapped, move |state| state.released.lock().push(backing));
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
    /// Requests apply in order: one naming memory maps it with the wanted kind; memory handle 0
    /// makes the range read zero again, or show its GPU-only region's memory.
    /// The payload follows the two buffer descriptors, so the job slot (the reference's word 14)
    /// is payload word 6.
    fn sparse_map(&self, ctx: &mut HleRequestContext, memory: &Nx2GpuMemory) -> ResultCode {
        let slot = Words::of(ctx.input()).u32(6);
        let requests = ctx.read_buffer(0).unwrap_or_default();
        let count = requests.len() / SPARSE_REQUEST_SIZE;
        let mut kinds = Vec::with_capacity(count);
        let mut unmapped = Vec::new();
        for request in requests.as_chunks::<SPARSE_REQUEST_SIZE>().0 {
            let words = Words::of(request);
            let (va, size) = (words.u64(0), words.u64(2));
            let kind = request[19];
            kinds.push(kind);
            let Some(last) = size.checked_sub(1).and_then(|n| va.checked_add(n)) else {
                continue;
            };
            match GpuMemoryHandle::new(words.u32(6)) {
                Some(handle) => match memory.normal_map_4k(va, last, handle, words.u64(8)) {
                    Ok(replaced) => {
                        unmapped.extend(replaced);
                        if let Err(err) = memory.modify_kind(va, last, kind) {
                            tracing::warn!("nx2:gpu 216: kind {kind:#x} at {va:#x}: {err}");
                        }
                    }
                    Err(err) => tracing::warn!("nx2:gpu 216: {err}"),
                },
                None => {
                    let mut removed = logged("216", memory.unmap_4k(va, last));
                    self.state.restore_gpu_only(memory, va, size, &mut removed);
                    unmapped.extend(removed);
                }
            }
        }
        if count != 0 && ctx.writable_buffer(0).is_some() {
            let _ = ctx.write_buffer(0, &kinds);
        }
        self.complete(slot, unmapped);
        ctx.push(count as u64);
        ResultCode::SUCCESS
    }
}

/// A request's unmapped ranges, or none after logging why it failed. Failures still complete the
/// job, as the reference does: the SDK aborts on an error reply.
fn logged(request: &str, result: Result<Vec<Unmapped>, Nx2MemoryError>) -> Vec<Unmapped> {
    result.unwrap_or_else(|err| {
        tracing::warn!("nx2:gpu {request}: {err}");
        Vec::new()
    })
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
                self.free_released();
                let range = (words.u64(3), words.u64(5));
                match command {
                    101 => self.create_memory(ctx, memory),
                    102 => {
                        let mut unmapped = match GpuMemoryHandle::new(words.u32(2)) {
                            Some(handle) => logged("102", memory.destroy(handle)),
                            None => {
                                tracing::warn!("nx2:gpu 102: memory handle 0");
                                Vec::new()
                            }
                        };
                        let mut spans: Vec<(GpuVAddr, u64)> = unmapped
                            .iter()
                            .map(|u| (u.range.gpu_va, u.range.size))
                            .collect();
                        spans.sort_unstable();
                        spans.dedup();
                        for (va, size) in spans {
                            self.state.restore_gpu_only(memory, va, size, &mut unmapped);
                        }
                        self.complete(words.u32(3), unmapped);
                        ResultCode::SUCCESS
                    }
                    211 => {
                        match GpuMemoryHandle::new(words.u32(7)) {
                            Some(handle) => unmap(
                                &self.context.gpu,
                                &logged(
                                    "211",
                                    memory.normal_map_4k(range.0, range.1, handle, words.u64(8)),
                                ),
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
                        let (first, last) = range;
                        let mut unmapped = logged("213", memory.unmap_4k(first, last));
                        if last >= first {
                            self.state.restore_gpu_only(
                                memory,
                                first,
                                last - first + 1,
                                &mut unmapped,
                            );
                        }
                        self.complete(words.u32(8), unmapped);
                        ResultCode::SUCCESS
                    }
                    214 => {
                        let size = range.1.wrapping_sub(range.0).wrapping_add(1);
                        if words.u32(7) & SPARSE_REGION != 0 {
                            unmap(
                                &self.context.gpu,
                                &logged("214", memory.create_sparse_region(range.0, size)),
                            );
                        } else if range.1 > range.0 && (range.0 | size) & PAGE_MASK == 0 {
                            self.create_gpu_only(memory, range.0, size);
                        } else {
                            tracing::warn!(
                                "nx2:gpu 214: unexpected region [{:#x}, {:#x}]",
                                range.0,
                                range.1
                            );
                        }
                        self.slot(words.u32(11)).signal();
                        ResultCode::SUCCESS
                    }
                    215 => {
                        let (first, last) = range;
                        let size = last.wrapping_sub(first).wrapping_add(1);
                        let backing = self.state.gpu_only.lock().remove(&first);
                        if let Some(backing) = backing {
                            self.release(memory, backing);
                        }
                        let result = match memory.destroy_sparse_region(first, size) {
                            Err(Nx2MemoryError::NoSuchRegion { .. }) => {
                                memory.unmap_4k(first, last)
                            }
                            result => result,
                        };
                        self.complete(words.u32(8), logged("215", result));
                        ResultCode::SUCCESS
                    }
                    _ => self.sparse_map(ctx, memory),
                }
            }
            _ => return None,
        })
    }
}
