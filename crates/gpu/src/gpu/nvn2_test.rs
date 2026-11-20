use lichii_common::GpuVAddr;

use super::*;
use crate::backend::{FenceAction, Nvn2Submission, ReplayAdmission, ReplayError};
use crate::nx2_memory::{GpuMemoryHandle, Nvn2BackingOwner, Nvn2TargetLease};

/// The native-only GMMU of a synthetic GPU. It cannot be passed to an as_gpu session.
pub struct Nvn2TestSpace(Arc<Gmmu>);

impl Nvn2TestSpace {
    /// Inspects the fixture's native identity mapping without exposing its GMMU to writers.
    pub fn translate(&self, address: GpuVAddr) -> Option<DAddr> {
        self.0.translate(address)
    }
}

type Standard = Box<dyn FnOnce(&mut dyn Rasterizer, &Gmmu) -> Result<(), ReplayError> + Send>;

/// A synthetic clear and its optional same-target Standard fallback.
pub struct Nvn2TestWork {
    /// An admitted lease from this GPU.
    pub lease: Nvn2TargetLease,
    /// Clear value in RGBA order.
    pub color: [f32; 4],
    /// RGBA write bits (red is bit zero).
    pub color_mask: u8,
    /// A syncpoint reserved exclusively for this fixture's host completion.
    pub retirement_syncpoint: u32,
    /// Standard work, or no work when the renderer declines before recording. An error stops
    /// the stream because Standard may already have recorded commands using the backing.
    pub standard: Option<Standard>,
}

impl Gpu {
    /// Creates the fixture's sole native address space before any regular space has registered.
    /// Ordinary as_gpu spaces cannot register into this fixture's native mapping broadcast.
    pub fn create_nvn2_test_space(&self) -> Option<Nvn2TestSpace> {
        let memory = self.nx2_memory()?;
        let gmmu = Arc::new(Gmmu::new(self.host1x.memory.clone(), BigPageSize::Size64K));
        memory.register_nvn2_test_space(gmmu.clone())?;
        Some(Nvn2TestSpace(gmmu))
    }

    /// Binds a synthetic channel without handing out a writable GMMU.
    pub fn create_nvn2_test_channel(&self, space: &Nvn2TestSpace) -> Option<ChannelId> {
        if !self.nx2_memory()?.is_nvn2_space(&space.0) {
            return None;
        }
        Some(self.create_channel(ChannelConfig::for_console(
            space.0.clone(),
            Console::Switch2,
        )))
    }

    /// Verifies every page of the exact native identity mapping before lending it to a packet.
    pub fn try_lease_nvn2_for_test(
        &self,
        space: &Nvn2TestSpace,
        handle: GpuMemoryHandle,
        address: GpuVAddr,
        size: u64,
        owner: Arc<dyn Nvn2BackingOwner>,
    ) -> Option<Nvn2TargetLease> {
        if self.stopping.load(Ordering::Acquire) {
            return None;
        }
        self.nx2_memory()?
            .try_lease_nvn2(&space.0, handle, address, size, owner)
    }

    /// Enqueues the lease on this channel, including while it waits on an earlier acquire.
    /// The returned receiver reports recording disposition, not host completion.
    pub fn submit_nvn2_for_test(
        &self,
        channel: ChannelId,
        work: Nvn2TestWork,
    ) -> mpsc::Receiver<Result<ReplayAdmission, ReplayError>> {
        let (done, outcome) = mpsc::channel();
        if self
            .nx2_memory()
            .is_none_or(|memory| !memory.owns_nvn2_lease(&work.lease))
            || self
                .host1x
                .syncpoints
                .guest()
                .value(work.retirement_syncpoint)
                .is_err()
        {
            let _ = done.send(Err(ReplayError));
            return outcome;
        }
        let (memory, target) = work.lease.mapping();
        let packet = Nvn2Submission {
            memory: memory.clone(),
            target,
            mapping_generation: memory.generation(),
            color: work.color,
            color_mask: work.color_mask,
            fence: FenceAction::IncrementSyncpoint(work.retirement_syncpoint),
        };
        let memory = memory.clone();
        self.inbox.push(CpuCommand::SubmitNvn2(
            channel,
            Box::new(crate::back::Nvn2Work {
                packet,
                standard: work.standard.map(|action| {
                    Box::new(move |r: &mut dyn Rasterizer| action(r, &memory))
                        as crate::back::Nvn2Fallback
                }),
                lease: Some((work.lease, work.retirement_syncpoint)),
                done,
            }),
        ));
        outcome
    }
}

#[derive(Clone)]
pub(super) struct Worker(Arc<Teardown>);

impl Worker {
    pub(super) fn new(memory: Arc<Nx2GpuMemory>) -> Self {
        Self(Arc::new(Teardown { memory }))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if std::thread::panicking() && self.0.memory.has_nvn2_backing() {
            tracing::error!("a GPU worker panicked; retaining native NVN2 backing");
            std::mem::forget(self.0.clone());
        }
    }
}

struct Teardown {
    memory: Arc<Nx2GpuMemory>,
}

impl Drop for Teardown {
    fn drop(&mut self) {
        self.memory.shutdown_nvn2();
    }
}
