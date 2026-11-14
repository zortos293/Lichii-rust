//! `nn::gpu::sdk`'s session (gsv 497): job-slot events, graphics device ids, the GPU VA
//! partitions, and the Switch 2 GPU memory requests (sub-session commands 101-216) on the GPU's
//! [`Nx2GpuMemory`] registry.
//!
//! Requests name their memory by inclusive [first, last] ranges and complete through a job slot:
//! the SDK waits on the slot's event (12) after the reply. Every request changes the registry on
//! the GPU's back thread, so it keeps its order both with the other requests and with the GPU
//! work: work queued before it still sees the old mappings, work queued after it the new ones.
//! An unmap or destroy completes its slot once the host GPU finished the work before it, because
//! the title reuses the memory right after. The back thread does not wait for that: it signals a
//! fence on a syncpoint of the session's own and moves on, and the slot completes when the host GPU
//! passes the fence. 101 reserves the memory handle up front and replies with it at once; the
//! memory is made in order like the other requests, so GPU work submitted after the reply sees it.
//!
//! A region 214 creates without the sparse flag is GPU-only storage the client renders into
//! (NX2_NOTES row 57), often far larger than what it fills (Mario Kart World asks for about
//! 9 GiB). It becomes a lazy region: a device range of its own that reads zero, where a 4 MiB
//! block gets memory when a write lands in it. That memory comes from the graphics server's
//! carve-out, the non-secure system pool, so the application pool the title sizes its heap from
//! stays whole ([`Reserve`]). Tiles mapped into the region and unmapped again show what was
//! written there again, and 215 gives the blocks back.

use std::sync::{Arc, OnceLock, Weak, mpsc};

use lichii_common::{GpuVAddr, ResultCode};
use lichii_gpu::Gpu;
use lichii_gpu::backend::{FenceAction, Rasterizer};
use lichii_gpu::host1x::Syncpoints;
use lichii_gpu::lazy::LazyBlock;
use lichii_gpu::nx2_memory::{GpuMemoryHandle, Nx2GpuMemory, Nx2MemoryError, Unmapped};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_memory::AddressSpace;
use parking_lot::Mutex;

use super::{Reserve, Words, handshake, push_readable, push_session};
use crate::nvdrv::Nvdrv;
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
    /// Made on the first request that needs one; none before nvdrv, which allocates syncpoints,
    /// is registered.
    fence: OnceLock<Option<Arc<Fence>>>,
    pending: Mutex<Pending>,
}

/// The GPU VA ranges of requests queued for the back thread whose change has not landed yet.
#[derive(Default)]
struct Pending {
    last_id: u64,
    ranges: Vec<(u64, GpuVAddr, GpuVAddr)>,
}

impl Pending {
    fn add(&mut self, first: GpuVAddr, last: GpuVAddr) -> u64 {
        self.last_id += 1;
        self.ranges.push((self.last_id, first, last));
        self.last_id
    }

    fn remove(&mut self, id: u64) {
        if let Some(index) = self.ranges.iter().position(|&(other, ..)| other == id) {
            self.ranges.swap_remove(index);
        }
    }

    fn overlaps(&self, first: GpuVAddr, last: GpuVAddr) -> bool {
        self.ranges
            .iter()
            .any(|&(_, start, end)| start <= last && first <= end)
    }
}

/// A syncpoint the session counts the requests on whose job slots wait for the host GPU.
struct Fence {
    nvdrv: Arc<Nvdrv>,
    syncpoints: Arc<Syncpoints>,
    id: u32,
}

impl Fence {
    fn new(context: &ServiceContext) -> Option<Arc<Self>> {
        let nvdrv = context.nvdrv.get()?.clone();
        let id = nvdrv.allocate_syncpoint();
        if id == 0 {
            return None;
        }
        Some(Arc::new(Self {
            nvdrv,
            syncpoints: context.gpu.host1x().syncpoints.clone(),
            id,
        }))
    }

    /// Runs `action` once the host GPU finished the work the back thread submitted so far,
    /// without waiting for it. The action keeps the fence, so the syncpoint is retired only once
    /// no increment of it is still on the way.
    fn after_gpu(
        self: &Arc<Self>,
        rasterizer: &mut dyn Rasterizer,
        action: impl FnOnce() + Send + 'static,
    ) {
        let Ok(threshold) = self.syncpoints.increase_max(self.id, 1) else {
            rasterizer.wait_for_idle();
            action();
            return;
        };
        rasterizer.signal_fence(FenceAction::IncrementSyncpoint(self.id));
        rasterizer.flush_commands();
        let fence = self.clone();
        let action = move || {
            action();
            drop(fence);
        };
        if self
            .syncpoints
            .guest()
            .register_action(self.id, threshold, action)
            .is_err()
        {
            tracing::error!(id = self.id, "nx2:gpu fence on an invalid syncpoint");
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        self.nvdrv.retire_syncpoint(self.id);
    }
}

impl GpuSession {
    pub fn new(context: &Arc<ServiceContext>) -> Self {
        Self {
            context: context.clone(),
            state: Arc::new(State {
                slots: std::array::from_fn(|_| Arc::new(Event::new())),
                devices: Mutex::new([false; DEVICES]),
                fence: OnceLock::new(),
                pending: Mutex::default(),
            }),
        }
    }

    fn slot(&self, id: u32) -> &Arc<Event> {
        &self.state.slots[(id % SLOTS) as usize]
    }

    fn memory(&self) -> Option<&Nx2GpuMemory> {
        self.context.gpu.nx2_memory()
    }

    /// Runs `change` on the GPU's back thread, after the GPU work queued before it and in order
    /// with the other requests, and hands what it unmapped to the renderer. Then it gives what the
    /// change released to the [`Reserve`] and signals job slot `slot`. When a `destructive`
    /// change unmapped something, or the change released blocks, that waits until the host GPU
    /// finished the work before it, since the title or the pool reuses that memory right after.
    /// `range` is the inclusive GPU VA range the change may map or unmap, which 101 checks
    /// until the change landed.
    fn in_order(
        &self,
        slot: Option<u32>,
        destructive: bool,
        range: Option<(GpuVAddr, GpuVAddr)>,
        change: impl FnOnce(&Nx2GpuMemory, &mut Change) + Send + 'static,
    ) {
        let gpu: Weak<Gpu> = Arc::downgrade(&self.context.gpu);
        let reserve = self.context.gpu_reserve.get().cloned();
        let event = slot.map(|slot| self.slot(slot).clone());
        let fence = destructive.then(|| self.fence()).flatten();
        let (diag_cmd, diag_t) = (diag::current(), std::time::Instant::now());
        let pending = range.map(|(first, last)| self.state.pending.lock().add(first, last));
        let state = self.state.clone();
        self.context
            .gpu
            .run_in_order(move |rasterizer: &mut dyn Rasterizer| {
                let mut done = Change::default();
                if let Some(memory) = gpu.upgrade().as_deref().and_then(Gpu::nx2_memory) {
                    change(memory, &mut done);
                    for range in &done.unmapped {
                        rasterizer.unmap_region(range.range.daddr, range.range.size);
                    }
                }
                if let Some(id) = pending {
                    state.pending.lock().remove(id);
                }
                let wait = (destructive && !done.unmapped.is_empty()) || !done.released.is_empty();
                let detached = reserve
                    .as_ref()
                    .map(|reserve| reserve.detach(&done.released))
                    .unwrap_or_default();
                let complete = move || {
                    if let Some(reserve) = reserve {
                        reserve.free(detached);
                    }
                    if let Some(event) = event {
                        diag::record(diag_cmd, diag_t.elapsed());
                        event.signal();
                    }
                };
                match fence {
                    Some(fence) if wait => fence.after_gpu(rasterizer, complete),
                    None if wait => {
                        rasterizer.wait_for_idle();
                        complete();
                    }
                    _ => complete(),
                }
            });
    }

    fn fence(&self) -> Option<Arc<Fence>> {
        self.state
            .fence
            .get_or_init(|| Fence::new(&self.context))
            .clone()
    }

    /// 214 without the sparse flag: makes [`start`, +`size`) a lazy region, in order. The reserve
    /// its blocks come from is started on the first one and topped up here, on the service
    /// thread, so the region's first writes find memory ready.
    fn create_gpu_only(&self, start: GpuVAddr, size: u64, slot: u32) {
        let reserve = self
            .context
            .gpu_reserve
            .get_or_init(|| Reserve::start(&self.context));
        reserve.refill(&self.context.process);
        let range = (start, start.saturating_add(size.saturating_sub(1)));
        self.in_order(
            Some(slot),
            false,
            Some(range),
            move |memory, done| match memory.create_lazy_region(start, size) {
                Ok((daddr, replaced)) => {
                    tracing::info!(
                        "nx2:gpu 214: GPU-only region {start:#x}+{size:#x} at device {daddr:#x}, \
                         backed as it is written"
                    );
                    done.unmapped = replaced;
                }
                Err(err) => {
                    tracing::warn!("nx2:gpu 214: GPU-only region {start:#x}+{size:#x}: {err}")
                }
            },
        );
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
    /// Reply word 3 is the GpuMemoryHandle later requests name. Memory over GPU VAs nothing maps
    /// and no queued request changes is made here, so the front thread and other services find it
    /// as soon as the title has the handle. Otherwise it would change mappings queued GPU work may
    /// still use, so it is made in order and the reply waits for the back thread.
    fn create_memory(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let t = std::time::Instant::now();
        let result = self.create_memory_inner(ctx);
        diag::record(101, t.elapsed());
        result
    }

    fn create_memory_inner(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let words = Words::command(ctx);
        let (first, last) = (words.u64(2), words.u64(4));
        let size = last.checked_sub(first).and_then(|n| n.checked_add(1));
        let Some(size) = size.filter(|&size| first.checked_add(size).is_some()) else {
            tracing::warn!("nx2:gpu 101: bad range [{first:#x}, {last:#x}]");
            return lichii_kernel::result::INVALID_SIZE;
        };
        let Some(memory) = self.memory() else {
            return lichii_kernel::result::NOT_FOUND;
        };
        let Ok(handle) = memory.reserve_handle() else {
            tracing::warn!("nx2:gpu 101 [{first:#x}, {last:#x}]: no memory handle left");
            return lichii_kernel::result::OUT_OF_RESOURCE;
        };
        let pending = self.state.pending.lock();
        let created = if pending.overlaps(first, last) || memory.in_use(first, size) {
            drop(pending);
            let (sender, created) = mpsc::sync_channel(1);
            let process = self.context.process.clone();
            self.in_order(None, false, Some((first, last)), move |memory, done| {
                let space: &AddressSpace = process.memory();
                let result = memory.create_gpu_accessible_memory_as(handle, space, first, size);
                let _ = sender.send(result.map(|replaced| done.unmapped = replaced));
            });
            let Ok(created) = created.recv() else {
                tracing::warn!("nx2:gpu 101: the GPU stopped");
                return lichii_kernel::result::OUT_OF_RESOURCE;
            };
            created
        } else {
            let result = memory.create_gpu_accessible_memory_as(
                handle,
                self.context.process.memory(),
                first,
                size,
            );
            drop(pending);
            result.map(|replaced| debug_assert!(replaced.is_empty()))
        };
        match created {
            Ok(()) => {
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
    fn sparse_map(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let slot = Words::of(ctx.input()).u32(6);
        let data = ctx.read_buffer(0).unwrap_or_default();
        let requests: Vec<SparseMap> = data
            .as_chunks::<SPARSE_REQUEST_SIZE>()
            .0
            .iter()
            .map(|request| {
                let words = Words::of(request);
                SparseMap {
                    va: words.u64(0),
                    size: words.u64(2),
                    handle: GpuMemoryHandle::new(words.u32(6)),
                    offset: words.u64(8),
                    kind: request[19],
                }
            })
            .collect();
        let count = requests.len();
        if count != 0 && ctx.writable_buffer(0).is_some() {
            let kinds: Vec<u8> = requests.iter().map(|r| r.kind).collect();
            let _ = ctx.write_buffer(0, &kinds);
        }
        let hull = requests
            .iter()
            .filter_map(|r| {
                let last = r.size.checked_sub(1).and_then(|n| r.va.checked_add(n))?;
                Some((r.va, last))
            })
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)));
        self.in_order(Some(slot), true, hull, move |memory, done| {
            for SparseMap {
                va,
                size,
                handle,
                offset,
                kind,
            } in requests
            {
                let Some(last) = size.checked_sub(1).and_then(|n| va.checked_add(n)) else {
                    continue;
                };
                match handle {
                    Some(handle) => match memory.normal_map_4k(va, last, handle, offset) {
                        Ok(replaced) => {
                            done.unmapped.extend(replaced);
                            if let Err(err) = memory.modify_kind(va, last, kind) {
                                tracing::warn!("nx2:gpu 216: kind {kind:#x} at {va:#x}: {err}");
                            }
                        }
                        Err(err) => tracing::warn!("nx2:gpu 216: {err}"),
                    },
                    None => done
                        .unmapped
                        .extend(logged("216", memory.unmap_4k(va, last))),
                }
            }
        });
        ctx.push(count as u64);
        ResultCode::SUCCESS
    }
}

/// What a request changed, as [`GpuSession::in_order`] passes it on.
#[derive(Default)]
struct Change {
    unmapped: Vec<Unmapped>,
    /// Lazy blocks whose memory goes back to the pool once the host GPU is past them.
    released: Vec<LazyBlock>,
}

/// One of 216's SparseMapRequests.
struct SparseMap {
    va: GpuVAddr,
    size: u64,
    handle: Option<GpuMemoryHandle>,
    offset: u64,
    kind: u8,
}

/// A request's unmapped ranges, or none after logging why it failed. Failures still complete the
/// job, as the reference does: the SDK aborts on an error reply.
fn logged(request: &str, result: Result<Vec<Unmapped>, Nx2MemoryError>) -> Vec<Unmapped> {
    result.unwrap_or_else(|err| {
        tracing::warn!("nx2:gpu {request}: {err}");
        Vec::new()
    })
}

impl Service for GpuSession {
    fn name(&self) -> &str {
        "nx2:gpu"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let command = ctx.command();
        diag::set_current(command);
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
                if self.memory().is_none() {
                    tracing::warn!("nx2:gpu {command}: no Switch 2 GPU memory for this title");
                    return Some(lichii_kernel::result::NOT_FOUND);
                }
                let range = (words.u64(3), words.u64(5));
                match command {
                    101 => self.create_memory(ctx),
                    102 => {
                        let handle = GpuMemoryHandle::new(words.u32(2));
                        self.in_order(Some(words.u32(3)), true, None, move |memory, done| {
                            match handle {
                                Some(handle) => {
                                    done.unmapped = logged("102", memory.destroy(handle))
                                }
                                None => tracing::warn!("nx2:gpu 102: memory handle 0"),
                            }
                        });
                        ResultCode::SUCCESS
                    }
                    211 => {
                        let (handle, offset) = (GpuMemoryHandle::new(words.u32(7)), words.u64(8));
                        self.in_order(
                            Some(words.u32(17)),
                            false,
                            Some(range),
                            move |memory, done| match handle {
                                Some(handle) => {
                                    done.unmapped = logged(
                                        "211",
                                        memory.normal_map_4k(range.0, range.1, handle, offset),
                                    );
                                }
                                None => tracing::warn!("nx2:gpu 211: memory handle 0"),
                            },
                        );
                        ResultCode::SUCCESS
                    }
                    212 => {
                        let wanted = (words.u32(7) & 0xff) as u8;
                        self.in_order(Some(words.u32(10)), false, Some(range), move |memory, _| {
                            if let Err(err) = memory.modify_kind(range.0, range.1, wanted) {
                                tracing::warn!("nx2:gpu 212: {err}");
                            }
                        });
                        ctx.push(u32::from(wanted));
                        ResultCode::SUCCESS
                    }
                    213 => {
                        let (first, last) = range;
                        self.in_order(
                            Some(words.u32(8)),
                            true,
                            Some((first, last)),
                            move |memory, done| {
                                done.unmapped = logged("213", memory.unmap_4k(first, last));
                            },
                        );
                        ResultCode::SUCCESS
                    }
                    214 => {
                        let (first, last) = range;
                        let size = last.wrapping_sub(first).wrapping_add(1);
                        let slot = words.u32(11);
                        if words.u32(7) & SPARSE_REGION != 0 {
                            self.in_order(Some(slot), false, Some(range), move |memory, done| {
                                done.unmapped =
                                    logged("214", memory.create_sparse_region(first, size));
                            });
                        } else {
                            self.create_gpu_only(first, size, slot);
                        }
                        ResultCode::SUCCESS
                    }
                    215 => {
                        let (first, last) = range;
                        let size = last.wrapping_sub(first).wrapping_add(1);
                        self.in_order(
                            Some(words.u32(8)),
                            true,
                            Some((first, last)),
                            move |memory, done| {
                                let result = match memory.destroy_lazy_region(first, size) {
                                    Ok(destroyed) => {
                                        done.released = destroyed.backed;
                                        Ok(destroyed.unmapped)
                                    }
                                    Err(Nx2MemoryError::NoSuchRegion { .. }) => {
                                        match memory.destroy_sparse_region(first, size) {
                                            Err(Nx2MemoryError::NoSuchRegion { .. }) => {
                                                memory.unmap_4k(first, last)
                                            }
                                            result => result,
                                        }
                                    }
                                    Err(err) => Err(err),
                                };
                                done.unmapped.extend(logged("215", result));
                            },
                        );
                        ResultCode::SUCCESS
                    }
                    _ => self.sparse_map(ctx),
                }
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod pending_tests {
    use super::Pending;

    #[test]
    fn pending_ranges_overlap_inclusively_until_removed() {
        let mut pending = Pending::default();
        let a = pending.add(0x1000, 0x1fff);
        let b = pending.add(0x4000, 0x4fff);
        assert!(pending.overlaps(0x1fff, 0x2fff));
        assert!(!pending.overlaps(0x2000, 0x3fff));
        assert!(pending.overlaps(0x0, 0x1000));
        pending.remove(a);
        assert!(!pending.overlaps(0x1000, 0x1fff));
        assert!(pending.overlaps(0x4800, 0x4800));
        pending.remove(b);
        assert!(!pending.overlaps(0, u64::MAX));
    }
}

mod diag {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    use std::time::{Duration, Instant};

    use parking_lot::Mutex;

    const COMMANDS: [u32; 6] = [101, 102, 213, 214, 215, 216];
    const ZERO: AtomicU64 = AtomicU64::new(0);
    static STATS: [AtomicU64; 18] = [ZERO; 18];
    static LAST: Mutex<Option<Instant>> = Mutex::new(None);

    thread_local! {
        static CURRENT: Cell<u32> = const { Cell::new(0) };
    }

    pub fn set_current(command: u32) {
        CURRENT.set(command);
    }

    pub fn current() -> u32 {
        CURRENT.get()
    }

    pub fn record(command: u32, took: Duration) {
        if let Some(i) = COMMANDS.iter().position(|&c| c == command) {
            let ns = took.as_nanos() as u64;
            STATS[i * 3].fetch_add(1, Relaxed);
            STATS[i * 3 + 1].fetch_add(ns, Relaxed);
            STATS[i * 3 + 2].fetch_max(ns, Relaxed);
        }
        let mut last = LAST.lock();
        let now = Instant::now();
        let start = *last.get_or_insert(now);
        if now - start < Duration::from_secs(5) {
            return;
        }
        *last = Some(now);
        drop(last);
        let line: Vec<String> = COMMANDS
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let n = STATS[i * 3].swap(0, Relaxed);
                let total = STATS[i * 3 + 1].swap(0, Relaxed);
                let max = STATS[i * 3 + 2].swap(0, Relaxed);
                format!(
                    "{c}: n={n} total={:.1}ms max={:.1}ms",
                    total as f64 / 1e6,
                    max as f64 / 1e6
                )
            })
            .collect();
        tracing::info!("diag nx2:gpu waits {}", line.join(" | "));
    }
}
