//! nvmap: handles to guest memory, pinned into host1x's device (SMMU) space for the GPU, and
//! `/dev/nvmap`.
//!
//! A handle is created with a size, then allocated over a range of the client's memory. Pinning
//! gives it a device range mapped to that memory; unpinned handles keep their range until it is
//! needed for another pin or the handle is freed.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};
use lichii_common::{DAddr, PAGE_MASK, PAGE_SIZE, VAddr, align_up};
use lichii_gpu::Gpu;
use lichii_memory::AddressSpace;
use parking_lot::{Mutex, RwLock};

use super::ioctl::Args;
use super::{Client, NvResult, Nvdrv};

/// Device ranges are reserved in multiples of this (yuzu's nvmap big page).
const DEVICE_GRANULE: u64 = 0x1_0000;
const HANDLE_ID_STEP: u32 = 4;
/// `Flags::map_uncached`.
const FLAG_MAP_UNCACHED: u32 = 1 << 0;
/// `Flags::keep_uncached_after_free`, meaningless for handles with an address.
const FLAG_KEEP_UNCACHED_AFTER_FREE: u32 = 1 << 2;

/// A handle as `free` leaves it.
pub(crate) struct FreeInfo {
    pub address: VAddr,
    pub size: u64,
    pub was_uncached: bool,
    /// The handle is gone, not only one reference to it.
    pub can_unlock: bool,
}

pub(crate) struct Handle {
    id: u32,
    state: Mutex<HandleState>,
}

struct HandleState {
    /// Page-aligned size.
    size: u64,
    /// `size` aligned to `align`.
    aligned_size: u64,
    /// The size the handle was created with.
    orig_size: u64,
    align: u64,
    kind: u8,
    flags: u32,
    address: VAddr,
    memory: Option<Arc<AddressSpace>>,
    allocated: bool,
    dupes: i32,
    internal_dupes: i32,
    pins: i64,
    daddr: DAddr,
    /// Has an entry in the queue of device ranges a pin may reclaim (checked for pins when it is
    /// taken).
    queued: bool,
}

impl HandleState {
    fn reserved_size(&self) -> u64 {
        align_up(self.aligned_size, DEVICE_GRANULE)
    }
}

/// Every nvmap handle of the console.
pub(crate) struct NvMap {
    gpu: Arc<Gpu>,
    handles: RwLock<HashMap<u32, Arc<Handle>>>,
    next_id: AtomicU32,
    unmap_queue: Mutex<VecDeque<Arc<Handle>>>,
    /// The references each session holds through Create and FromId, by handle.
    session_refs: Mutex<HashMap<u64, HashMap<u32, u32>>>,
}

impl NvMap {
    pub fn new(gpu: Arc<Gpu>) -> Self {
        Self {
            gpu,
            handles: RwLock::new(HashMap::new()),
            next_id: AtomicU32::new(HANDLE_ID_STEP),
            unmap_queue: Mutex::new(VecDeque::new()),
            session_refs: Mutex::default(),
        }
    }

    fn create(&self, size: u64) -> Result<u32, NvResult> {
        if size == 0 {
            return Err(NvResult::BadValue);
        }
        let id = self.next_id.fetch_add(HANDLE_ID_STEP, Ordering::Relaxed);
        let aligned = align_up(size, PAGE_SIZE);
        let handle = Arc::new(Handle {
            id,
            state: Mutex::new(HandleState {
                size: aligned,
                aligned_size: aligned,
                orig_size: size,
                align: 0,
                kind: 0,
                flags: 0,
                address: 0,
                memory: None,
                allocated: false,
                dupes: 1,
                internal_dupes: 0,
                pins: 0,
                daddr: 0,
                queued: false,
            }),
        });
        self.handles.write().insert(id, handle);
        Ok(id)
    }

    fn get(&self, id: u32) -> Option<Arc<Handle>> {
        self.handles.read().get(&id).cloned()
    }

    /// The device address handle `id` is pinned at; 0 when it is not mapped.
    pub fn device_address(&self, id: u32) -> DAddr {
        self.get(id).map_or(0, |handle| handle.state.lock().daddr)
    }

    /// The size the handle was created with.
    pub fn size(&self, id: u32) -> Option<u64> {
        self.get(id).map(|handle| handle.state.lock().orig_size)
    }

    /// The size aligned to the handle's alignment: how far into the handle a mapping may reach.
    pub fn aligned_size(&self, id: u32) -> Option<u64> {
        self.get(id).map(|handle| handle.state.lock().aligned_size)
    }

    #[cfg(test)]
    pub fn pins(&self, id: u32) -> i64 {
        self.get(id).map_or(0, |handle| handle.state.lock().pins)
    }

    /// The alignment the handle was allocated with.
    pub fn align(&self, id: u32) -> Option<u64> {
        self.get(id).map(|handle| handle.state.lock().align)
    }

    /// Maps the handle into device space if it is not (pins are counted) and returns its device
    /// address.
    pub fn pin(&self, id: u32) -> Result<DAddr, NvResult> {
        let handle = self.get(id).ok_or(NvResult::BadValue)?;
        let mut state = handle.state.lock();
        if state.daddr == 0 {
            self.map(&handle, &mut state)?;
        }
        state.pins += 1;
        Ok(state.daddr)
    }

    /// Drops a pin; a mapped handle without pins joins the queue of device ranges a later pin may
    /// reclaim.
    pub fn unpin(&self, id: u32) {
        let Some(handle) = self.get(id) else {
            return;
        };
        let mut state = handle.state.lock();
        state.pins -= 1;
        if state.pins < 0 {
            tracing::warn!(id, "nvmap pin count imbalance");
            state.pins = 0;
        }
        if state.pins == 0 && state.daddr != 0 && !state.queued {
            state.queued = true;
            self.unmap_queue.lock().push_back(handle.clone());
        }
    }

    fn map(&self, handle: &Handle, state: &mut HandleState) -> Result<(), NvResult> {
        let Some(memory) = state.memory.clone() else {
            tracing::error!(id = handle.id, "pinning an unallocated nvmap handle");
            return Err(NvResult::BadValue);
        };
        if state.address == 0 || state.address & PAGE_MASK != 0 {
            tracing::error!(
                id = handle.id,
                address = format_args!("{:#x}", state.address),
                "nvmap handle without a page-aligned CPU address cannot be pinned"
            );
            return Err(NvResult::InvalidAddress);
        }
        let host1x = self.gpu.host1x();
        let reserved = state.reserved_size();
        let align = state.align.max(DEVICE_GRANULE).next_power_of_two();
        let daddr = loop {
            if let Some(daddr) = host1x.allocator.allocate(reserved, align) {
                break daddr;
            }
            if !self.reclaim_one(handle, state) {
                tracing::error!(size = reserved, "device address space exhausted");
                return Err(NvResult::InsufficientMemory);
            }
        };
        if let Err(fault) = host1x
            .memory
            .map_process(daddr, &memory, state.address, state.size)
        {
            tracing::error!(id = handle.id, %fault, "nvmap handle memory is not mapped");
            self.free_device_range(daddr, reserved);
            return Err(NvResult::InvalidAddress);
        }
        state.daddr = daddr;
        Ok(())
    }

    /// Unmaps the oldest queued handle that has no pins. `pinning` (locked by the caller as
    /// `pinning_state`) leaves the queue if it is met. False when nothing could be unmapped.
    fn reclaim_one(&self, pinning: &Handle, pinning_state: &mut HandleState) -> bool {
        let mut busy = 0;
        loop {
            let mut queue = self.unmap_queue.lock();
            if busy >= queue.len() {
                return false;
            }
            let victim = queue.pop_front().expect("the queue is not empty");
            drop(queue);
            if std::ptr::eq(&*victim, pinning) {
                pinning_state.queued = false;
                continue;
            }
            let Some(mut state) = victim.state.try_lock() else {
                self.unmap_queue.lock().push_back(victim.clone());
                busy += 1;
                continue;
            };
            state.queued = false;
            if state.pins == 0 && state.daddr != 0 {
                self.unmap(&mut state);
                return true;
            }
        }
    }

    fn unmap(&self, state: &mut HandleState) {
        let host1x = self.gpu.host1x();
        host1x.memory.unmap(state.daddr, state.size);
        self.gpu.unmap_region(state.daddr, state.size);
        self.free_device_range(state.daddr, state.reserved_size());
        state.daddr = 0;
    }

    fn free_device_range(&self, daddr: DAddr, size: u64) {
        if let Err(error) = self.gpu.host1x().allocator.free(daddr, size) {
            tracing::error!(%error, "freeing an nvmap device range");
        }
    }

    /// Drops one reference. The last client reference unmaps the handle; the handle goes away
    /// with the last reference of any kind. `None` for an unknown handle.
    fn free(&self, id: u32) -> Option<FreeInfo> {
        let handle = self.get(id)?;
        let mut state = handle.state.lock();
        state.dupes -= 1;
        if state.dupes < 0 {
            tracing::warn!(id, "nvmap duplicate count imbalance");
        } else if state.dupes == 0 && state.internal_dupes <= 0 {
            self.release_mapping(&handle, &mut state);
        }
        let removed = state.dupes <= 0 && state.internal_dupes <= 0;
        if removed {
            self.handles.write().remove(&id);
        }
        Some(FreeInfo {
            address: state.address,
            size: state.size,
            was_uncached: state.flags & FLAG_MAP_UNCACHED != 0,
            can_unlock: removed,
        })
    }

    /// Takes a reference on the emulator's behalf (vi holding a buffer): the handle and its
    /// device mapping stay until [`NvMap::free_internal`], even after the guest frees it.
    pub fn duplicate_internal(&self, id: u32) -> Result<(), NvResult> {
        let handle = self.get(id).ok_or(NvResult::BadValue)?;
        handle.state.lock().internal_dupes += 1;
        Ok(())
    }

    /// Drops a reference [`NvMap::duplicate_internal`] took; the last reference of any kind
    /// unmaps and removes the handle.
    pub fn free_internal(&self, id: u32) {
        let Some(handle) = self.get(id) else {
            return;
        };
        let mut state = handle.state.lock();
        state.internal_dupes -= 1;
        if state.internal_dupes < 0 {
            tracing::warn!(id, "nvmap internal duplicate count imbalance");
        }
        if state.dupes <= 0 && state.internal_dupes <= 0 {
            self.release_mapping(&handle, &mut state);
            self.handles.write().remove(&id);
        }
    }

    fn release_mapping(&self, handle: &Arc<Handle>, state: &mut HandleState) {
        if std::mem::take(&mut state.queued) {
            self.unmap_queue
                .lock()
                .retain(|queued| !Arc::ptr_eq(queued, handle));
        }
        if state.daddr != 0 {
            self.unmap(state);
        }
        state.pins = 0;
    }

    fn add_session_ref(&self, session: u64, id: u32) {
        *self
            .session_refs
            .lock()
            .entry(session)
            .or_default()
            .entry(id)
            .or_default() += 1;
    }

    /// Drops one reference `session` holds and the handle reference behind it.
    fn free_session_ref(&self, session: u64, id: u32) -> Option<FreeInfo> {
        {
            let mut sessions = self.session_refs.lock();
            let refs = sessions.get_mut(&session)?;
            let count = refs.get_mut(&id)?;
            *count -= 1;
            if *count == 0 {
                refs.remove(&id);
            }
        }
        self.free(id)
    }

    /// Drops every reference a closing session still holds, allocated or not.
    pub fn release_session(&self, session: u64) {
        let Some(refs) = self.session_refs.lock().remove(&session) else {
            return;
        };
        for (id, count) in refs {
            for _ in 0..count {
                self.free(id);
            }
        }
    }
}

/// `/dev/nvmap`, opened by one client.
pub(crate) struct NvMapDevice {
    nvdrv: Arc<Nvdrv>,
    memory: Arc<AddressSpace>,
    session: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct CreateParams {
    size: u32,
    handle: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FromIdParams {
    id: u32,
    handle: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct AllocParams {
    handle: u32,
    heap_mask: u32,
    flags: u32,
    align: u32,
    kind: u8,
    _pad: [u8; 7],
    address: u64,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct FreeParams {
    handle: u32,
    _pad: u32,
    address: u64,
    size: u32,
    flags: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ParamParams {
    handle: u32,
    param: u32,
    result: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GetIdParams {
    id: u32,
    handle: u32,
}

const PARAM_SIZE: u32 = 1;
const PARAM_ALIGNMENT: u32 = 2;
const PARAM_BASE: u32 = 3;
const PARAM_HEAP: u32 = 4;
const PARAM_KIND: u32 = 5;
const PARAM_IS_SHARED_MEM_MAPPED: u32 = 6;
const EINVAL: i32 = -22;

impl NvMapDevice {
    pub fn new(nvdrv: Arc<Nvdrv>, client: &Client) -> Self {
        Self {
            nvdrv,
            memory: client.memory().clone(),
            session: client.session,
        }
    }

    pub fn ioctl(&self, args: &mut Args) -> Option<NvResult> {
        Some(match args.number() {
            (1, 0x1) => args.fixed(|p| self.create(p)),
            (1, 0x3) => args.fixed(|p| self.reference_by_id(p)),
            (1, 0x4) => args.fixed(|p| self.alloc(p)),
            (1, 0x5) => args.fixed(|p| self.free(p)),
            (1, 0x9) => args.fixed(|p| self.param(p)),
            (1, 0xe) => args.fixed(|p| self.get_id(p)),
            _ => return None,
        })
    }

    fn nvmap(&self) -> &NvMap {
        &self.nvdrv.nvmap
    }

    fn create(&self, params: &mut CreateParams) -> NvResult {
        match self.nvmap().create(u64::from(params.size)) {
            Ok(id) => {
                self.nvmap().add_session_ref(self.session, id);
                params.handle = id;
                NvResult::Success
            }
            Err(result) => result,
        }
    }

    fn reference_by_id(&self, params: &mut FromIdParams) -> NvResult {
        if params.id == 0 {
            return NvResult::BadValue;
        }
        let Some(handle) = self.nvmap().get(params.id) else {
            return NvResult::BadValue;
        };
        let mut state = handle.state.lock();
        if !state.allocated {
            return NvResult::BadValue;
        }
        state.dupes += 1;
        drop(state);
        self.nvmap().add_session_ref(self.session, handle.id);
        params.handle = handle.id;
        NvResult::Success
    }

    fn alloc(&self, params: &mut AllocParams) -> NvResult {
        if params.handle == 0 || !(params.align == 0 || params.align.is_power_of_two()) {
            return NvResult::BadValue;
        }
        let align = u64::from(params.align).max(PAGE_SIZE);
        let Some(handle) = self.nvmap().get(params.handle) else {
            return NvResult::BadValue;
        };
        let mut state = handle.state.lock();
        if state.allocated {
            return NvResult::InsufficientMemory;
        }
        state.flags = params.flags;
        if params.address != 0 {
            state.flags &= !FLAG_KEEP_UNCACHED_AFTER_FREE;
        } else {
            tracing::error!(
                id = handle.id,
                "nvmap handles without a CPU address are not supported"
            );
        }
        state.kind = params.kind;
        state.align = align;
        state.aligned_size = align_up(state.size, align);
        state.address = params.address;
        state.memory = Some(self.memory.clone());
        state.allocated = true;
        params.align = align as u32;
        NvResult::Success
    }

    fn free(&self, params: &mut FreeParams) -> NvResult {
        if params.handle == 0 {
            return NvResult::Success;
        }
        if let Some(info) = self.nvmap().free_session_ref(self.session, params.handle) {
            params.address = info.address;
            params.size = info.size as u32;
            params.flags = if info.was_uncached {
                FLAG_MAP_UNCACHED
            } else {
                0
            };
            if !info.can_unlock {
                tracing::debug!(handle = params.handle, "nvmap handle still referenced");
            }
        }
        NvResult::Success
    }

    fn param(&self, params: &mut ParamParams) -> NvResult {
        let Some(handle) = (params.handle != 0)
            .then(|| self.nvmap().get(params.handle))
            .flatten()
        else {
            return NvResult::BadValue;
        };
        let state = handle.state.lock();
        params.result = match params.param {
            PARAM_SIZE => state.orig_size as u32,
            PARAM_ALIGNMENT => state.align as u32,
            PARAM_BASE => EINVAL as u32,
            PARAM_HEAP if state.allocated => 0x4000_0000,
            PARAM_HEAP => 0,
            PARAM_KIND => u32::from(state.kind),
            PARAM_IS_SHARED_MEM_MAPPED => 0,
            _ => return NvResult::BadValue,
        };
        NvResult::Success
    }

    fn get_id(&self, params: &mut GetIdParams) -> NvResult {
        if params.handle == 0 {
            return NvResult::BadValue;
        }
        match self.nvmap().get(params.handle) {
            Some(handle) => {
                params.id = handle.id;
                NvResult::Success
            }
            None => NvResult::AccessDenied,
        }
    }
}
