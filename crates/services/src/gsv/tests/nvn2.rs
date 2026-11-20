use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Weak, mpsc};

use lichii_common::{DAddr, PAGE_SIZE, PAddr};
use lichii_gpu::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles,
    GraphicsContext, IndirectDraw, Nvn2Submission, Query, Rasterizer, ReplayAdmission, ReplayError,
};
use lichii_gpu::channel::{Submission, SyncpointFence};
use lichii_gpu::gmmu::{BigPageSize, Gmmu};
use lichii_gpu::gpu::{Nvn2TestSpace, Nvn2TestWork};
use lichii_gpu::nx2_memory::{GpuMemoryHandle, Nvn2BackingOwner, Nvn2TargetLease, Nx2MemoryError};
use lichii_kernel::memory_objects::SharedMemory;
use lichii_kernel::result::BUSY;
use lichii_memory::PhysicalMemory;

use super::*;
use crate::testing::{RecordedPrograms, context_with_renderer};

const SIZE: u64 = 2 * PAGE_SIZE;
const PIXEL: u32 = 0xff00_ff00;
const TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
enum Mode {
    Direct,
    Standard,
    Terminal,
    TerminalFallback,
}

struct Renderer {
    handles: GpuHandles,
    mode: Mode,
    pending: Vec<FenceAction>,
    host_done: Arc<AtomicBool>,
    fallback_count: Arc<AtomicU32>,
    writeback: mpsc::Sender<(DAddr, PAddr, u32)>,
    allow_writeback: mpsc::Receiver<()>,
    writeback_fails: bool,
    teardown: Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>,
    recorded_target: Option<DAddr>,
}

impl Rasterizer for Renderer {
    fn try_replay_nvn2(&mut self, packet: &Nvn2Submission) -> Result<ReplayAdmission, ReplayError> {
        if matches!(self.mode, Mode::Standard | Mode::TerminalFallback) {
            return Ok(ReplayAdmission::UseStandard);
        }
        self.recorded_target = packet.memory.translate(packet.target);
        assert!(self.recorded_target.is_some());
        if matches!(self.mode, Mode::Terminal) {
            return Err(ReplayError);
        }
        self.signal_fence(packet.fence);
        Ok(ReplayAdmission::Recorded)
    }

    fn accelerate_buffer_clear(
        &mut self,
        _: ChannelId,
        memory: &Gmmu,
        dst: u64,
        size: u64,
        value: u32,
    ) -> bool {
        assert_eq!((size, value), (SIZE, PIXEL));
        self.recorded_target = memory.translate(dst);
        assert!(self.recorded_target.is_some());
        self.fallback_count.fetch_add(1, Ordering::Release);
        !matches!(self.mode, Mode::TerminalFallback)
    }

    fn finish_nvn2_backing(&mut self, addr: DAddr, size: u64) -> Result<(), ReplayError> {
        assert_eq!(size, SIZE);
        if let Some(target) = self.recorded_target {
            assert_eq!(target, addr);
            assert!(self.host_done.load(Ordering::Acquire));
        }
        let device = &self.handles.device;
        let pa = device.physical_address(addr).unwrap();
        if !self.writeback_fails && self.recorded_target.is_some() {
            device.write(addr, PIXEL).unwrap();
        }
        self.writeback
            .send((addr, pa, device.read(addr).unwrap()))
            .unwrap();
        self.allow_writeback
            .recv_timeout(TIMEOUT)
            .map_err(|_| ReplayError)?;
        if self.writeback_fails {
            return Err(ReplayError);
        }
        self.recorded_target = None;
        Ok(())
    }

    fn signal_fence(&mut self, action: FenceAction) {
        self.pending.push(action);
    }

    fn release_fences(&mut self, _: bool) {
        if self.host_done.load(Ordering::Acquire) {
            for action in self.pending.drain(..) {
                action.execute(&self.handles);
            }
        }
    }

    fn has_pending_fences(&self) -> bool {
        !self.pending.is_empty()
    }

    fn has_unsubmitted_fences(&self) -> bool {
        false
    }

    fn draw(&mut self, _: &mut GraphicsContext<'_>, _: &DrawParams<'_>) {}
    fn draw_indirect(&mut self, _: &mut GraphicsContext<'_>, _: &IndirectDraw) {}
    fn clear(&mut self, _: &mut GraphicsContext<'_>, _: u32) {}
    fn dispatch_compute(&mut self, _: &ComputeContext<'_>) {}
    fn query(&mut self, _: ChannelId, _: &Gmmu, _: &Query) {}
    fn invalidate_region(&mut self, _: DAddr, _: u64) {}
    fn flush_region(&mut self, _: DAddr, _: u64) {}
    fn wait_for_idle(&mut self) {}
    fn present(&mut self, _: &FramebufferConfig) {}
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if let Some((entered, release)) = &self.teardown {
            let _ = entered.send(());
            let _ = release.recv_timeout(TIMEOUT);
        }
    }
}

struct Owner {
    dram: Arc<PhysicalMemory>,
    shared: Arc<SharedMemory>,
}

impl Nvn2BackingOwner for Owner {
    fn physical_range(&self) -> (&Arc<PhysicalMemory>, PAddr, u64) {
        (
            &self.dram,
            self.shared.physical_address(),
            self.shared.size(),
        )
    }
}

struct Native {
    handle: GpuMemoryHandle,
    owner: Arc<Owner>,
    device: DAddr,
}

struct Fixture {
    context: Arc<ServiceContext>,
    client: IpcClient,
    session: Handle,
    space: Nvn2TestSpace,
    channel: ChannelId,
    syncpoint: u32,
    base: VAddr,
    heap: VAddr,
    host_done: Arc<AtomicBool>,
    fallback_count: Arc<AtomicU32>,
    writeback: mpsc::Receiver<(DAddr, PAddr, u32)>,
    allow_writeback: mpsc::Sender<()>,
    teardown: Option<(mpsc::Receiver<()>, mpsc::Sender<()>)>,
}

impl Fixture {
    fn new(mode: Mode, writeback_fails: bool, hold_teardown: bool) -> Self {
        let host_done = Arc::new(AtomicBool::new(false));
        let fallback_count = Arc::new(AtomicU32::new(0));
        let (writeback_seen, writeback) = mpsc::channel();
        let (allow_writeback, writeback_gate) = mpsc::channel();
        let (entered, teardown_entered) = mpsc::channel();
        let (allow_teardown, teardown_gate) = mpsc::channel();
        let (context, client) = context_with_renderer(
            lichii_kernel::testing::program_image(SWITCH2_TITLE),
            Settings::default(),
            Arc::new(RecordedPrograms::default()),
            |handles| {
                Box::new(Renderer {
                    handles,
                    mode,
                    pending: Vec::new(),
                    host_done: host_done.clone(),
                    fallback_count: fallback_count.clone(),
                    writeback: writeback_seen,
                    allow_writeback: writeback_gate,
                    writeback_fails,
                    teardown: hold_teardown.then_some((entered, teardown_gate)),
                    recorded_target: None,
                })
            },
        );
        let space = context.gpu.create_nvn2_test_space().unwrap();
        let channel = context.gpu.create_nvn2_test_channel(&space).unwrap();
        let gsv = connect_gsv(&context, &client);
        let session = session(&client, gsv, 497);
        crate::nvdrv::register(&context).unwrap();
        let syncpoint = context.nvdrv.get().unwrap().allocate_syncpoint();
        assert_ne!(syncpoint, 0);
        let heap = client
            .process()
            .page_table()
            .set_heap_size(2 << 20)
            .unwrap();
        Self {
            context,
            client,
            session,
            space,
            channel,
            syncpoint,
            base: heap + (16 << 20),
            heap,
            host_done,
            fallback_count,
            writeback,
            allow_writeback,
            teardown: hold_teardown.then_some((teardown_entered, allow_teardown)),
        }
    }

    fn owner(&self) -> Arc<Owner> {
        let table = self.client.process().page_table();
        let manager = table.memory_manager();
        assert!(Arc::ptr_eq(manager, self.context.kernel.memory_manager()));
        let shared = Arc::new(
            SharedMemory::new(
                manager.clone(),
                None,
                table.pool(),
                SIZE,
                MemoryPermission::READ_WRITE,
                MemoryPermission::READ_WRITE,
            )
            .unwrap(),
        );
        shared.write(0, &0u32.to_le_bytes());
        Arc::new(Owner {
            dram: manager.physical().clone(),
            shared,
        })
    }

    fn native(&self) -> Native {
        let owner = self.owner();
        owner
            .shared
            .map(
                self.client.process(),
                self.base,
                SIZE,
                MemoryPermission::READ_WRITE,
            )
            .unwrap();
        let handle = self.create(self.base).unwrap();
        let device = self.space.translate(self.base).unwrap();
        Native {
            handle,
            owner,
            device,
        }
    }

    fn create(&self, first: VAddr) -> Result<GpuMemoryHandle, ResultCode> {
        let mut reply = self
            .client
            .send(
                self.session,
                &words(
                    Request::tipc(101),
                    &[split(first), split(first + SIZE - 1), [0, 0]].concat(),
                ),
            )
            .unwrap();
        if reply.result != ResultCode::SUCCESS {
            return Err(reply.result);
        }
        Ok(GpuMemoryHandle::new(reply.pop::<u32>()).unwrap())
    }

    fn lease(&self, target: &Native) -> Option<Nvn2TargetLease> {
        self.context.gpu.try_lease_nvn2_for_test(
            &self.space,
            target.handle,
            self.base,
            SIZE,
            target.owner.clone(),
        )
    }

    fn queue(
        &self,
        lease: Nvn2TargetLease,
        fallback: bool,
    ) -> mpsc::Receiver<Result<ReplayAdmission, ReplayError>> {
        let channel = self.channel;
        let dst = self.base;
        self.context.gpu.submit_nvn2_for_test(
            channel,
            Nvn2TestWork {
                lease,
                color: [0., 1., 0., 1.],
                color_mask: 0xf,
                retirement_syncpoint: self.syncpoint,
                standard: if fallback {
                    Some(Box::new(move |renderer, memory| {
                        renderer
                            .accelerate_buffer_clear(channel, memory, dst, SIZE, PIXEL)
                            .then_some(())
                            .ok_or(ReplayError)
                    }))
                } else {
                    None
                },
            },
        )
    }

    fn park(&self) -> u32 {
        let gpu = &self.context.gpu;
        let id = self.context.nvdrv.get().unwrap().allocate_syncpoint();
        assert_ne!(id, self.syncpoint);
        let value = gpu.host1x().syncpoints.increase_max(id, 1).unwrap();
        gpu.submit(
            self.channel,
            Submission {
                wait: Some(SyncpointFence { id, value }),
                ..Submission::default()
            },
        );
        id
    }

    fn request(&self, request: Request) -> ResultCode {
        self.client.send(self.session, &request).unwrap().result
    }

    fn destroy(&self, handle: GpuMemoryHandle, slot: u32) -> ResultCode {
        self.request(Request::tipc(102).arg(handle.raw()).arg(slot))
    }

    fn slot(&self, id: u32) -> Arc<Event> {
        let reply = self
            .client
            .send(self.session, &Request::tipc(12).arg(id))
            .unwrap();
        event(&self.client, reply.copy_handles[0])
    }

    fn free(&self) -> u64 {
        let table = self.client.process().page_table();
        table.memory_manager().free_size(table.pool())
    }

    fn remove_cpu_mapping(&self, native: Native) -> Weak<SharedMemory> {
        let weak = Arc::downgrade(&native.owner.shared);
        native
            .owner
            .shared
            .unmap(self.client.process(), self.base, SIZE)
            .unwrap();
        assert_eq!(
            self.client.process().memory().physical_address(self.base),
            None
        );
        weak
    }
}

fn until(check: impl Fn() -> bool) {
    let end = Instant::now() + TIMEOUT;
    while !check() {
        assert!(Instant::now() < end, "GPU worker did not finish in time");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn check_ordered_retirement(mode: Mode, expected: ReplayAdmission) {
    let f = Fixture::new(mode, false, false);
    let baseline = f.free();
    let target = f.native();
    let (handle, daddr, pa) = (
        target.handle,
        target.device,
        target.owner.shared.physical_address(),
    );
    let gate = f.park();
    let outcome = f.queue(f.lease(&target).unwrap(), true);
    let slot = f.slot(3);
    let gpu = &f.context.gpu;
    let host = gpu.host1x();
    gpu.synchronize();
    assert_eq!(outcome.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert_eq!(f.destroy(handle, 3), ResultCode::SUCCESS);
    assert_eq!(f.destroy(handle, 4), BUSY);
    assert!(f.lease(&target).is_none());
    let weak = f.remove_cpu_mapping(target);
    let still_pinned = || {
        assert!(!slot.is_signaled());
        assert_eq!(f.space.translate(f.base), Some(daddr));
        assert_eq!(
            host.memory.physical_address(daddr + PAGE_SIZE),
            Some(pa + PAGE_SIZE)
        );
        assert!(weak.upgrade().is_some());
        assert_eq!(f.free(), baseline - SIZE);
    };
    still_pinned();
    let ordinary = f.create(f.heap).unwrap();
    assert_ne!(f.space.translate(f.heap), Some(daddr));
    let ordinary_job = f.slot(6);
    assert_eq!(f.destroy(ordinary, 6), ResultCode::SUCCESS);
    gpu.synchronize();
    assert_eq!(
        f.space.translate(f.heap),
        None,
        "untracked 102 still unmaps before its host fence"
    );
    assert!(!ordinary_job.is_signaled());
    let paused_slot = f.slot(5);
    assert_eq!(
        f.request(range_words(213, f.base, f.base + PAGE_SIZE - 1, &[0, 5])),
        BUSY
    );
    assert!(!paused_slot.is_signaled());

    host.syncpoints.host().increment(gate).unwrap();
    gpu.wake();
    assert_eq!(outcome.recv_timeout(TIMEOUT).unwrap(), Ok(expected));
    assert_eq!(
        f.fallback_count.load(Ordering::Acquire),
        u32::from(matches!(mode, Mode::Standard))
    );
    host.syncpoints.host().increment(f.syncpoint).unwrap();
    gpu.synchronize();
    assert_eq!(host.syncpoints.guest().value(f.syncpoint), Ok(0));
    assert_eq!(f.writeback.try_recv(), Err(mpsc::TryRecvError::Empty));
    still_pinned();

    f.host_done.store(true, Ordering::Release);
    assert_eq!(
        f.writeback.recv_timeout(TIMEOUT).unwrap(),
        (daddr, pa, PIXEL)
    );
    assert!(
        ordinary_job.is_signaled(),
        "untracked 102 still completes on its existing fence"
    );
    still_pinned();
    let mut bytes = [0; 4];
    weak.upgrade().unwrap().read(0, &mut bytes);
    assert_eq!(u32::from_le_bytes(bytes), PIXEL);
    f.allow_writeback.send(()).unwrap();
    until(|| slot.is_signaled());
    assert_eq!(host.syncpoints.guest().value(f.syncpoint), Ok(1));
    assert_eq!(f.space.translate(f.base), None);
    assert_eq!(host.memory.physical_address(daddr), None);
    assert!(weak.upgrade().is_none());
    assert_eq!(f.free(), baseline);
    f.create(f.heap + SIZE).unwrap();
    assert_eq!(f.space.translate(f.heap + SIZE), Some(daddr));
    assert_ne!(host.memory.physical_address(daddr), Some(pa));
}

#[test]
fn direct_and_standard_keep_the_same_backing_from_park_to_gsv_retirement() {
    check_ordered_retirement(Mode::Direct, ReplayAdmission::Recorded);
    check_ordered_retirement(Mode::Standard, ReplayAdmission::UseStandard);
}

#[test]
fn no_work_decline_and_parked_channel_cancel_release_the_queue_lease() {
    let f = Fixture::new(Mode::Standard, false, false);
    let target = f.native();
    let outcome = f.queue(f.lease(&target).unwrap(), false);
    assert_eq!(
        outcome.recv_timeout(TIMEOUT).unwrap(),
        Ok(ReplayAdmission::UseStandard)
    );
    assert_eq!(
        f.context.gpu.host1x().syncpoints.guest().value(f.syncpoint),
        Ok(0)
    );
    let lease = f
        .lease(&target)
        .expect("a decline recorded nothing and released the lease");
    f.park();
    let outcome = f.queue(lease, false);
    let slot = f.slot(3);
    let gpu = &f.context.gpu;
    gpu.synchronize();
    assert_eq!(outcome.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert_eq!(f.destroy(target.handle, 3), ResultCode::SUCCESS);
    gpu.destroy_channel(f.channel);
    assert_eq!(
        outcome.recv_timeout(TIMEOUT),
        Err(mpsc::RecvTimeoutError::Disconnected)
    );
    assert_eq!(f.writeback.recv_timeout(TIMEOUT).unwrap().2, 0);
    assert!(!slot.is_signaled());
    f.allow_writeback.send(()).unwrap();
    until(|| slot.is_signaled());
    assert_eq!(f.space.translate(f.base), None);
    assert_eq!(gpu.host1x().syncpoints.guest().value(f.syncpoint), Ok(0));
}

#[test]
fn shutdown_drops_unconsumed_lease_without_publishing_its_job() {
    let f = Fixture::new(Mode::Direct, false, false);
    let target = f.native();
    let (handle, daddr) = (target.handle, target.device);
    let (gpu, slot) = (&f.context.gpu, f.slot(3));
    gpu.pause(true);
    let outcome = f.queue(f.lease(&target).unwrap(), true);
    let weak = f.remove_cpu_mapping(target);
    assert_eq!(f.destroy(handle, 3), ResultCode::SUCCESS);
    gpu.shutdown();
    assert_eq!(
        outcome.recv_timeout(TIMEOUT),
        Err(mpsc::RecvTimeoutError::Disconnected)
    );
    assert!(weak.upgrade().is_none());
    assert_eq!(f.space.translate(f.base), None);
    assert_eq!(gpu.host1x().memory.physical_address(daddr), None);
    assert!(!slot.is_signaled());
    assert_eq!(gpu.host1x().syncpoints.guest().value(f.syncpoint), Ok(0));
}

fn check_terminal_recording_failure(mode: Mode) {
    let f = Fixture::new(mode, false, false);
    let target = f.native();
    let (handle, daddr, pa) = (
        target.handle,
        target.device,
        target.owner.shared.physical_address(),
    );
    let gate = f.park();
    let outcome = f.queue(f.lease(&target).unwrap(), true);
    let weak = f.remove_cpu_mapping(target);
    let slot = f.slot(3);
    assert_eq!(f.destroy(handle, 3), ResultCode::SUCCESS);
    let gpu = &f.context.gpu;
    let host = gpu.host1x();
    host.syncpoints.host().increment(gate).unwrap();
    gpu.wake();
    assert_eq!(outcome.recv_timeout(TIMEOUT).unwrap(), Err(ReplayError));
    assert_eq!(
        f.fallback_count.load(Ordering::Acquire),
        u32::from(matches!(mode, Mode::TerminalFallback))
    );
    assert!(!slot.is_signaled());
    assert_eq!(host.syncpoints.guest().value(f.syncpoint), Ok(0));
    assert_eq!(f.writeback.try_recv(), Err(mpsc::TryRecvError::Empty));
    assert!(weak.upgrade().is_some());
    assert_eq!(f.space.translate(f.base), Some(daddr));
    assert_eq!(host.memory.physical_address(daddr), Some(pa));
    gpu.shutdown();
    assert!(weak.upgrade().is_none());
    assert_eq!(f.space.translate(f.base), None);
    assert_eq!(host.memory.physical_address(daddr), None);
    assert!(!slot.is_signaled());
    assert_eq!(host.syncpoints.guest().value(f.syncpoint), Ok(0));
}

#[test]
fn terminal_recording_failure_does_not_retire_or_signal_success() {
    check_terminal_recording_failure(Mode::Terminal);
    check_terminal_recording_failure(Mode::TerminalFallback);
}

#[test]
fn failed_writeback_keeps_backing_when_shutdown_detaches_the_renderer() {
    let f = Fixture::new(Mode::Direct, true, true);
    let target = f.native();
    let (handle, daddr, pa) = (
        target.handle,
        target.device,
        target.owner.shared.physical_address(),
    );
    let outcome = f.queue(f.lease(&target).unwrap(), false);
    let weak = f.remove_cpu_mapping(target);
    let slot = f.slot(3);
    assert_eq!(f.destroy(handle, 3), ResultCode::SUCCESS);
    assert_eq!(
        outcome.recv_timeout(TIMEOUT).unwrap(),
        Ok(ReplayAdmission::Recorded)
    );
    f.host_done.store(true, Ordering::Release);
    assert_eq!(f.writeback.recv_timeout(TIMEOUT).unwrap(), (daddr, pa, 0));
    f.allow_writeback.send(()).unwrap();
    let (destructor, allow_destructor) = f.teardown.as_ref().unwrap();
    destructor.recv_timeout(TIMEOUT).unwrap();
    let gpu = &f.context.gpu;
    let device = &gpu.host1x().memory;
    gpu.shutdown();
    assert!(!slot.is_signaled());
    assert!(weak.upgrade().is_some());
    assert_eq!(device.physical_address(daddr), Some(pa));
    assert_eq!(f.space.translate(f.base), Some(daddr));
    f.create(f.heap).unwrap();
    assert_ne!(f.space.translate(f.heap), Some(daddr));
    allow_destructor.send(()).unwrap();
    until(|| weak.upgrade().is_none());
    assert_eq!(f.space.translate(f.base), None);
    assert_eq!(device.physical_address(daddr), None);
    assert!(!slot.is_signaled());
}

#[test]
fn admission_requires_one_dedicated_exact_unaliased_shared_mapping() {
    let (regular, _, _) = gsv();
    let ordinary = regular.gpu.create_address_space(BigPageSize::Size64K);
    regular.gpu.destroy_address_space(&ordinary);
    assert!(regular.gpu.create_nvn2_test_space().is_none());

    let f = Fixture::new(Mode::Standard, false, false);
    let gpu = &f.context.gpu;
    let target = f.native();
    let owner = target.owner.clone();
    let addr = f.base;
    let last = addr + SIZE - 1;
    let alias = addr + SIZE * 2;
    let registry = gpu.nx2_memory().unwrap();
    let ordinary = gpu.create_address_space(BigPageSize::Size64K);
    assert_eq!(
        ordinary.translate(addr),
        None,
        "as_gpu's factory cannot share the native fixture"
    );
    assert!(gpu.create_nvn2_test_space().is_none());
    assert!(
        registry
            .unregister_address_space(lichii_gpu::nx2_memory::AddressSpaceId(1))
            .is_none()
    );
    for (handle, va, size, candidate) in [
        (target.handle, addr, SIZE, f.owner()),
        (target.handle, addr + 1, SIZE, owner.clone()),
        (target.handle, addr, PAGE_SIZE, owner.clone()),
        (
            GpuMemoryHandle::new(u32::MAX).unwrap(),
            addr,
            SIZE,
            owner.clone(),
        ),
    ] {
        assert!(
            gpu.try_lease_nvn2_for_test(&f.space, handle, va, size, candidate)
                .is_none()
        );
    }
    let device = &gpu.host1x().memory;
    let second = target.device + PAGE_SIZE;
    device.unmap(second, PAGE_SIZE);
    assert!(
        f.lease(&target).is_none(),
        "the first page alone is insufficient"
    );
    device.map(
        second,
        owner.shared.physical_address() + PAGE_SIZE,
        PAGE_SIZE,
    );
    registry
        .normal_map_4k(alias, alias + SIZE - 1, target.handle, 0)
        .unwrap();
    assert!(
        f.lease(&target).is_none(),
        "a native GPU VA alias disqualifies the handle"
    );
    registry.unmap_4k(alias, alias + SIZE - 1).unwrap();
    owner
        .shared
        .map(
            f.client.process(),
            alias,
            SIZE,
            MemoryPermission::READ_WRITE,
        )
        .unwrap();
    let physical_alias = f.create(alias).unwrap();
    assert!(
        f.lease(&target).is_none(),
        "a different native handle can alias the same physical pages"
    );
    registry.destroy(physical_alias).unwrap();
    assert!(f.lease(&target).is_some());
    assert_eq!(f.create(alias), Err(BUSY));
    assert_eq!(f.create(addr), Err(BUSY));
    assert!(matches!(
        registry.unmap_4k(addr, last),
        Err(Nx2MemoryError::Nvn2TargetInUse { .. })
    ));
    assert!(matches!(
        registry.destroy(target.handle),
        Err(Nx2MemoryError::Nvn2TargetInUse { .. })
    ));
    gpu.shutdown();
    assert_eq!(f.space.translate(addr), None);
}

#[test]
fn protected_range_mutations_fail_closed_but_whole_remap_waits() {
    let f = Fixture::new(Mode::Direct, false, false);
    let target = f.native();
    let (handle, daddr, pa) = (
        target.handle,
        target.device,
        target.owner.shared.physical_address(),
    );
    let other = f.create(f.heap).unwrap();
    let other_daddr = f.space.translate(f.heap).unwrap();
    let mut remap = vec![other.raw(), 0, 0];
    remap.resize(10, 0);
    remap.push(3);
    let mut alias = remap.clone();
    alias[0] = handle.raw();
    for (command, args) in [
        (212, vec![0xdb, 0, 0, 5]),
        (213, vec![0, 5]),
        (214, vec![0x200, 0, 0, 0, 5]),
        (214, vec![0, 0, 0, 0, 5]),
        (215, vec![0, 5]),
    ] {
        let lease = f.lease(&target).unwrap();
        assert_eq!(
            f.request(range_words(command, f.base, f.base + PAGE_SIZE - 1, &args)),
            BUSY
        );
        drop(lease);
    }
    let gate = f.park();
    let outcome = f.queue(f.lease(&target).unwrap(), true);
    assert_eq!(
        f.request(range_words(211, f.base, f.base + PAGE_SIZE - 1, &remap)),
        BUSY
    );
    assert_eq!(
        f.request(range_words(
            211,
            f.heap + SIZE,
            f.heap + 2 * SIZE - 1,
            &alias
        )),
        BUSY
    );
    let list = [
        sparse_request(f.heap + SIZE, SIZE, other.raw(), 0, 0),
        sparse_request(f.heap + 2 * SIZE, SIZE, handle.raw(), 0, 0),
    ]
    .concat();
    let input = f.client.alloc(&list);
    let batch = words(
        Request::tipc(216).buffer_a(input, list.len() as u64),
        &[0, 0, 0, 0, 0, 0, 5],
    );
    assert_eq!(f.request(batch), BUSY);
    assert_eq!(f.space.translate(f.heap + SIZE), None);
    assert!(!f.slot(5).is_signaled());
    assert_eq!(
        f.request(range_words(211, f.base, f.base + SIZE - 1, &remap)),
        ResultCode::SUCCESS
    );
    assert_eq!(f.destroy(handle, 5), BUSY);
    let refused = f.slot(6);
    assert_eq!(f.destroy(other, 6), BUSY);
    assert!(!refused.is_signaled());
    let weak = f.remove_cpu_mapping(target);
    let slot = f.slot(3);
    assert!(!slot.is_signaled());
    let gpu = &f.context.gpu;
    let registry = gpu.nx2_memory().unwrap();
    assert!(matches!(
        registry.destroy(other),
        Err(Nx2MemoryError::Nvn2TargetInUse { .. })
    ));
    assert!(matches!(
        registry.create_gpu_accessible_memory_as(
            other,
            f.client.process().memory(),
            f.heap + SIZE,
            SIZE,
        ),
        Err(Nx2MemoryError::Nvn2TargetInUse { .. })
    ));
    gpu.host1x().syncpoints.host().increment(gate).unwrap();
    gpu.wake();
    assert_eq!(
        outcome.recv_timeout(TIMEOUT).unwrap(),
        Ok(ReplayAdmission::Recorded)
    );
    assert!(!slot.is_signaled());
    assert_eq!(f.space.translate(f.base), Some(daddr));

    f.host_done.store(true, Ordering::Release);
    assert_eq!(
        f.writeback.recv_timeout(TIMEOUT).unwrap(),
        (daddr, pa, PIXEL)
    );
    assert!(!slot.is_signaled());
    assert_eq!(f.destroy(other, 6), BUSY);
    f.allow_writeback.send(()).unwrap();
    until(|| slot.is_signaled());
    assert_eq!(f.space.translate(f.base), Some(other_daddr));
    assert!(
        weak.upgrade().is_some(),
        "211 leaves the old native handle alive"
    );
    assert_eq!(gpu.host1x().memory.physical_address(daddr), Some(pa));
    assert_eq!(f.destroy(handle, 4), ResultCode::SUCCESS);
    let destroyed = f.slot(4);
    until(|| destroyed.is_signaled());
    assert!(weak.upgrade().is_none());
    assert_eq!(gpu.host1x().memory.physical_address(daddr), None);
    assert_eq!(f.space.translate(f.base), Some(other_daddr));
    assert_eq!(f.destroy(other, 7), ResultCode::SUCCESS);
    let replacement_destroyed = f.slot(7);
    until(|| replacement_destroyed.is_signaled());
    assert_eq!(f.space.translate(f.base), None);
    assert!(!refused.is_signaled());
}
