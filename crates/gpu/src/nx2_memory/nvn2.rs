use lichii_common::{PAGE_SIZE, PAddr};
use lichii_memory::PhysicalMemory;

use super::*;

#[cfg(any(test, feature = "testing"))]
const MAX_TARGET_BYTES: u64 = 256 << 10;

/// Owns this exact physical reservation for as long as the value lives. The GSV test
/// producer supplies a kernel SharedMemory it allocated from the process's own manager.
pub trait Nvn2BackingOwner: Send + Sync {
    /// The DRAM and contiguous span this owner actually retains, not a translation snapshot.
    fn physical_range(&self) -> (&Arc<PhysicalMemory>, PAddr, u64);
}

/// A native operation that can retire the single replay target.
#[derive(Clone, Copy)]
pub enum Nvn2Mutation {
    /// Remove the handle and its mappings.
    Destroy(GpuMemoryHandle),
    /// Replace the entire target mapping with another handle's pages.
    Remap {
        /// First byte, inclusive.
        first: GpuVAddr,
        /// Last byte, inclusive.
        last: GpuVAddr,
        /// Replacement handle.
        handle: GpuMemoryHandle,
        /// Offset into the replacement handle.
        offset: u64,
    },
}

/// How the service must dispatch a proposed mutation.
pub enum Nvn2RetirementAdmission {
    /// The mutation does not touch replay state. Use the existing Standard path.
    Untracked,
    /// The tracked target cannot accept this mutation. Do not start or complete it.
    Refused,
    /// Admission is closed. Schedule the mutation only when the ticket is quiescent.
    Deferred(Nvn2Retirement),
}

type Ready = Box<dyn FnOnce(Arc<Target>) + Send>;

#[cfg_attr(not(any(test, feature = "testing")), allow(dead_code))]
enum Use {
    Available,
    Held,
    Retiring {
        mutation: Nvn2Mutation,
        ready: Option<Ready>,
    },
    Quiescent(Nvn2Mutation),
    Failed,
}

pub(super) enum Registration {
    #[cfg_attr(not(any(test, feature = "testing")), allow(dead_code))]
    Mapped(Arc<Target>),
    Remapped(Arc<Target>),
}

impl Registration {
    pub(super) fn target(&self) -> &Target {
        match self {
            Self::Mapped(target) | Self::Remapped(target) => target,
        }
    }
}

pub(super) struct Target {
    pub(super) handle: GpuMemoryHandle,
    va: GpuVAddr,
    size: u64,
    daddr: DAddr,
    pa: PAddr,
    _owner: Arc<dyn Nvn2BackingOwner>,
    usage: Mutex<Use>,
    #[cfg(any(test, feature = "testing"))]
    memory: Arc<Gmmu>,
}

impl Target {
    fn overlaps(&self, va: GpuVAddr, size: u64) -> bool {
        size != 0 && va < self.va + self.size && self.va < va.saturating_add(size)
    }

    pub(super) fn busy(&self) -> Nx2MemoryError {
        Nx2MemoryError::Nvn2TargetInUse {
            va: self.va,
            size: self.size,
        }
    }

    pub(super) fn aliases(&self, device: &DeviceAddressSpace, daddr: DAddr, size: u64) -> bool {
        aliases(device, daddr, size, self.pa, self.size)
    }

    pub(super) fn reserves_remap_source(&self, raw: u32) -> bool {
        let usage = self.usage.lock();
        matches!(*usage,
            Use::Retiring { mutation: Nvn2Mutation::Remap { handle, .. }, .. } |
            Use::Quiescent(Nvn2Mutation::Remap { handle, .. }) if handle.raw() == raw)
    }

    fn release(self: Arc<Self>) {
        let ready = {
            let mut usage = self.usage.lock();
            match &mut *usage {
                Use::Held => {
                    *usage = Use::Available;
                    None
                }
                Use::Retiring { mutation, ready } => {
                    let ready = ready.take();
                    *usage = Use::Quiescent(*mutation);
                    ready
                }
                _ => None,
            }
        };
        if let Some(ready) = ready {
            ready(self);
        }
    }
}

fn aliases(device: &DeviceAddressSpace, daddr: DAddr, size: u64, pa: PAddr, len: u64) -> bool {
    (0..size).step_by(PAGE_SIZE as usize).any(|offset| {
        device
            .physical_address(daddr + offset)
            .is_some_and(|page| (pa..pa + len).contains(&page))
    })
}

/// A single queued or in-flight use. Keep it through host retirement if any work was recorded.
pub struct Nvn2TargetLease(Option<Arc<Target>>);

impl Drop for Nvn2TargetLease {
    fn drop(&mut self) {
        if let Some(target) = self.0.take() {
            target.release();
        }
    }
}

#[cfg(any(test, feature = "testing"))]
impl Nvn2TargetLease {
    pub(crate) fn mapping(&self) -> (&Arc<Gmmu>, GpuVAddr) {
        let target = self.0.as_ref().expect("queued lease owns target");
        (&target.memory, target.va)
    }

    pub(crate) fn fail(mut self) {
        let target = self.0.take().expect("queued lease owns target");
        let old = std::mem::replace(&mut *target.usage.lock(), Use::Failed);
        drop(old);
    }

    pub(crate) fn after_fence(
        mut self,
        syncpoints: &crate::host1x::Syncpoints,
        id: u32,
        value: u32,
    ) {
        let target = self.0.take().expect("queued lease owns target");
        let target = Arc::downgrade(&target);
        syncpoints
            .guest()
            .register_action(id, value, move || {
                if let Some(target) = target.upgrade() {
                    target.release();
                }
            })
            .expect("the fixture already validated this syncpoint");
    }
}

/// Owns a retiring mapping and its physical backing until the renderer confirms writeback.
pub struct Nvn2Retirement {
    target: Arc<Target>,
}

impl Nvn2Retirement {
    /// Runs `ready` outside the locks once queued or in-flight use is gone. It must schedule
    /// retirement, not wait for the GPU from a queue or fence callback.
    pub(crate) fn when_quiescent(self, ready: Box<dyn FnOnce(Nvn2Retirement) + Send>) {
        let Self { target } = self;
        {
            let mut usage = target.usage.lock();
            match &mut *usage {
                Use::Retiring {
                    ready: callback, ..
                } => {
                    *callback = Some(Box::new(move |target| ready(Self { target })));
                    return;
                }
                Use::Quiescent(_) => {}
                _ => return,
            }
        }
        ready(Self { target });
    }
}

impl State {
    pub(super) fn guard_nvn2(
        &self,
        handle: Option<u32>,
        va: GpuVAddr,
        size: u64,
    ) -> Result<(), Nx2MemoryError> {
        if !cfg!(any(test, feature = "testing")) {
            return Ok(());
        }
        if let Some(registered) = &self.nvn2 {
            let target = registered.target();
            if handle == Some(target.handle.raw())
                || (matches!(registered, Registration::Mapped(_)) && target.overlaps(va, size))
            {
                return Err(target.busy());
            }
        }
        Ok(())
    }
}

impl Nx2GpuMemory {
    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn register_nvn2_test_space(&self, gmmu: Arc<Gmmu>) -> Option<AddressSpaceId> {
        let mut state = self.state.lock();
        if state.last_space != 0 {
            return None;
        }
        state.last_space += 1;
        let id = AddressSpaceId(state.last_space);
        state.nvn2_space = Some(id);
        self.register_locked(&mut state, id, gmmu);
        Some(id)
    }

    /// Whether an ordinary mapping request would touch the protected target or introduce
    /// an alias of it. The registry also checks this when the mutation runs.
    pub fn nvn2_mapping_is_protected(
        &self,
        handle: Option<GpuMemoryHandle>,
        va: GpuVAddr,
        size: u64,
    ) -> bool {
        self.nvn2_mappings_are_protected([(handle, va, size)])
    }

    /// Checks a native batch against one registry snapshot before any subrequest runs.
    pub fn nvn2_mappings_are_protected(
        &self,
        mappings: impl IntoIterator<Item = (Option<GpuMemoryHandle>, GpuVAddr, u64)>,
    ) -> bool {
        if !cfg!(any(test, feature = "testing")) {
            return false;
        }
        let state = self.state.lock();
        state.nvn2.is_some()
            && mappings.into_iter().any(|(handle, va, size)| {
                state
                    .guard_nvn2(handle.map(GpuMemoryHandle::raw), va, size)
                    .is_err()
            })
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn is_nvn2_space(&self, memory: &Arc<Gmmu>) -> bool {
        let state = self.state.lock();
        matches!(state.spaces.as_slice(), [(id, registered)] if state.nvn2_space == Some(*id)
            && Arc::ptr_eq(registered, memory))
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn owns_nvn2_lease(&self, lease: &Nvn2TargetLease) -> bool {
        let state = self.state.lock();
        matches!((&state.nvn2, &lease.0),
            (Some(Registration::Mapped(registered)), Some(queued)) if Arc::ptr_eq(registered, queued))
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn try_lease_nvn2(
        &self,
        gmmu: &Arc<Gmmu>,
        handle: GpuMemoryHandle,
        va: GpuVAddr,
        size: u64,
        owner: Arc<dyn Nvn2BackingOwner>,
    ) -> Option<Nvn2TargetLease> {
        if size > MAX_TARGET_BYTES || check_range(va, size).is_err() {
            return None;
        }
        let (dram, physical, owned) = owner.physical_range();
        if !Arc::ptr_eq(dram, self.device.physical())
            || physical & PAGE_MASK != 0
            || owned != size
            || physical.checked_add(size)? > dram.size()
        {
            return None;
        }
        let mut state = self.state.lock();
        if matches!(state.nvn2, Some(Registration::Remapped(_))) {
            return None;
        }
        let [(space, registered)] = state.spaces.as_slice() else {
            return None;
        };
        if Some(*space) != state.nvn2_space || !Arc::ptr_eq(registered, gmmu) {
            return None;
        }
        let info = state.memories.get(&handle.raw())?;
        let range = state.ranges.get(&va)?;
        if info.cpu_va != va
            || info.size != size
            || range.size != size
            || range.kind != PTE_KIND_PITCH
            || range.backing != (Backing::Memory { handle: handle.raw(), offset: 0 })
            || state.regions.range(..va + size).next_back().is_some_and(|(&start, r)| start + r.size > va)
            || state.ranges.values().filter(|r| matches!(r.backing, Backing::Memory { handle: raw, .. } if raw == handle.raw())).count() != 1
        {
            return None;
        }
        let daddr = info.daddr;
        if state.memories.iter().any(|(&raw, info)| {
            raw != handle.raw() && aliases(&self.device, info.daddr, info.size, physical, size)
        }) {
            return None;
        }
        for offset in (0..size).step_by(PAGE_SIZE as usize) {
            if gmmu.translate(va + offset) != Some(daddr + offset)
                || gmmu.kind(va + offset) != Some(PTE_KIND_PITCH)
                || self.device.physical_address(daddr + offset) != Some(physical + offset)
            {
                return None;
            }
        }
        if let Some(Registration::Mapped(target)) = &state.nvn2 {
            if target.handle != handle || target.va != va || !Arc::ptr_eq(&target._owner, &owner) {
                return None;
            }
            let mut usage = target.usage.lock();
            if !matches!(*usage, Use::Available) {
                return None;
            }
            *usage = Use::Held;
            return Some(Nvn2TargetLease(Some(target.clone())));
        }
        let target = Arc::new(Target {
            handle,
            va,
            size,
            daddr,
            pa: physical,
            _owner: owner,
            usage: Mutex::new(Use::Held),
            memory: gmmu.clone(),
        });
        state.nvn2 = Some(Registration::Mapped(target.clone()));
        Some(Nvn2TargetLease(Some(target)))
    }

    /// Closes admission before a tracked mutation enters the queue. An overlapping mutation
    /// that isn't a whole-target destroy or replacement cannot use the Standard path.
    pub fn begin_nvn2_retirement(&self, mutation: Nvn2Mutation) -> Nvn2RetirementAdmission {
        if !cfg!(any(test, feature = "testing")) {
            return Nvn2RetirementAdmission::Untracked;
        }
        let state = self.state.lock();
        let target = match (&state.nvn2, mutation) {
            (Some(Registration::Mapped(target)), _) => target,
            (Some(Registration::Remapped(target)), Nvn2Mutation::Remap { handle, .. })
                if handle == target.handle =>
            {
                return Nvn2RetirementAdmission::Refused;
            }
            _ => return Nvn2RetirementAdmission::Untracked,
        };
        match mutation {
            Nvn2Mutation::Destroy(handle) if handle != target.handle => {
                return if target.reserves_remap_source(handle.raw()) {
                    Nvn2RetirementAdmission::Refused
                } else {
                    Nvn2RetirementAdmission::Untracked
                };
            }
            Nvn2Mutation::Destroy(_) => {}
            Nvn2Mutation::Remap {
                first,
                last,
                handle,
                offset,
            } => {
                let size = last.saturating_sub(first).saturating_add(1);
                if handle != target.handle && !target.overlaps(first, size) {
                    return Nvn2RetirementAdmission::Untracked;
                }
                if first != target.va
                    || size != target.size
                    || handle == target.handle
                    || offset & PAGE_MASK != 0
                    || !state.memories.get(&handle.raw()).is_some_and(|info| {
                        offset
                            .checked_add(target.size)
                            .is_some_and(|end| end <= info.size)
                    })
                {
                    return Nvn2RetirementAdmission::Refused;
                }
            }
        }
        let mut usage = target.usage.lock();
        match *usage {
            Use::Available => *usage = Use::Quiescent(mutation),
            Use::Held => {
                *usage = Use::Retiring {
                    mutation,
                    ready: None,
                }
            }
            _ => return Nvn2RetirementAdmission::Refused,
        }
        Nvn2RetirementAdmission::Deferred(Nvn2Retirement {
            target: target.clone(),
        })
    }

    pub(crate) fn nvn2_retirement_range(&self, ticket: &Nvn2Retirement) -> Option<(DAddr, u64)> {
        let state = self.state.lock();
        match &state.nvn2 {
            Some(Registration::Mapped(target)) if Arc::ptr_eq(target, &ticket.target) => {
                Some((target.daddr, target.size))
            }
            _ => None,
        }
    }

    /// Applies a quiescent mutation after writeback and renderer invalidation succeeded.
    /// On error the old target stays protected; the caller must not signal the job.
    pub(crate) fn finish_nvn2_retirement(
        &self,
        ready: Nvn2Retirement,
    ) -> Result<(), Nx2MemoryError> {
        let mut state = self.state.lock();
        if !matches!(&state.nvn2, Some(Registration::Mapped(target)) if Arc::ptr_eq(target, &ready.target))
        {
            return Err(ready.target.busy());
        }
        let Use::Quiescent(mutation) = *ready.target.usage.lock() else {
            return Err(ready.target.busy());
        };
        state.nvn2 = match mutation {
            Nvn2Mutation::Destroy(handle) => {
                self.destroy_locked(&mut state, handle)?;
                None
            }
            Nvn2Mutation::Remap {
                first,
                handle,
                offset,
                ..
            } => {
                Self::map_locked(&mut state, first, ready.target.size, handle, offset)?;
                Some(Registration::Remapped(ready.target.clone()))
            }
        };
        drop(state);
        drop(ready);
        Ok(())
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn has_nvn2_backing(&self) -> bool {
        let state = self.state.lock();
        state.nvn2.is_some()
    }

    #[cfg(any(test, feature = "testing"))]
    pub(crate) fn shutdown_nvn2(&self) {
        let target = {
            let mut state = self.state.lock();
            let Some(target) = state.nvn2.take() else {
                return;
            };
            let result = self.destroy_locked(&mut state, target.target().handle);
            if let Err(error) = result {
                tracing::error!(%error, "releasing NVN2 backing during shutdown");
            }
            target
        };
        let old = std::mem::replace(&mut *target.target().usage.lock(), Use::Failed);
        drop(old);
    }
}
