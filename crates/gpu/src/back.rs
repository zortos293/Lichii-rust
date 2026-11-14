//! The GPU's back half: executes the stream in order on each channel's engines and runs the host
//! actions that need the renderer. The only code that touches the [`Rasterizer`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc;

use lichii_common::DAddr;
use lichii_memory::DeviceAddressSpace;
use lichii_texture::block_linear::GobLayout;

use crate::activity::{Activity, RendererCall, Wait};
use crate::backend::{
    ChannelId, FenceAction, FramebufferConfig, GpuHandles, GpuWrittenPages, Rasterizer, ReleaseSize,
};
use crate::clock::GpuTimestamp;
use crate::engines::kepler_compute::pcas::QmdScheduler;
use crate::engines::{Engine, EngineContext, EngineFactory, EngineKind};
use crate::gmmu::{Gmmu, RangeTarget};
use crate::gpu::{MAX_POLL, MIN_POLL};
use crate::host1x::Syncpoints;
use crate::ring::{Consumer, WRAP};
use crate::stream::{FetchSync, FetchSyncs, MULTI_MAX_WORDS, REDUCTIONS, RELEASE_SIZES, tag};
use crate::throttle::Throttle;

/// The back reports progress at least this often while it executes.
const RELEASE_SLOTS: u64 = 1024;

/// What a channel needs on the back.
pub(crate) struct BackChannelSetup {
    pub id: ChannelId,
    pub memory: Arc<Gmmu>,
    pub gob_layout: GobLayout,
    pub reference: Arc<AtomicU32>,
}

/// Work the CPU side runs on the back thread with the renderer.
pub(crate) type RasterizerAction = Box<dyn FnOnce(&mut dyn Rasterizer) + Send>;

/// A CPU-side command forwarded through the stream, so it keeps its order with the methods.
pub(crate) enum SideCommand {
    CreateChannel(BackChannelSetup),
    DestroyChannel(ChannelId),
    BindAddressSpace(ChannelId, Arc<Gmmu>),
    FlushRegion(DAddr, u64),
    UnmapRegion(DAddr, u64),
    Present(FramebufferConfig),
    Run(RasterizerAction),
    /// Reports that everything before it executed.
    Signal(mpsc::SyncSender<()>),
}

struct BackChannel {
    setup: BackChannelSetup,
    engines: [Option<Box<dyn Engine>>; EngineKind::COUNT],
}

/// Executes the stream.
pub(crate) struct Back {
    rasterizer: Box<dyn Rasterizer>,
    device: Arc<DeviceAddressSpace>,
    syncpoints: Arc<Syncpoints>,
    timestamp: Arc<GpuTimestamp>,
    engines: EngineFactory,
    side: mpsc::Receiver<SideCommand>,
    channels: Vec<Option<BackChannel>>,
    current: Option<usize>,
    /// The engine the last method went to (channel index, kind index), until it is flushed.
    last: Option<(usize, u8)>,
    words: Vec<u32>,
    qmd_scheduler: QmdScheduler,
    fetch_syncs: Arc<FetchSyncs>,
    written_pages: Arc<GpuWrittenPages>,
    activity: Arc<Activity>,
    /// Set by [`crate::Gpu::shutdown`]: the back drops what is left instead of executing it.
    stopping: Arc<AtomicBool>,
}

impl Back {
    pub(crate) fn new(
        rasterizer: Box<dyn Rasterizer>,
        handles: &GpuHandles,
        engines: EngineFactory,
        side: mpsc::Receiver<SideCommand>,
        fetch_syncs: Arc<FetchSyncs>,
        activity: Arc<Activity>,
        stopping: Arc<AtomicBool>,
    ) -> Self {
        Self {
            rasterizer,
            device: handles.device.clone(),
            syncpoints: handles.syncpoints.clone(),
            timestamp: handles.timestamp.clone(),
            engines,
            side,
            channels: Vec::new(),
            current: None,
            last: None,
            words: Vec::with_capacity(MULTI_MAX_WORDS + 1),
            qmd_scheduler: QmdScheduler::new(),
            fetch_syncs,
            written_pages: handles.written_pages.clone(),
            activity,
            stopping,
        }
    }

    /// Executes everything published, until the ring is empty.
    pub(crate) fn run(&mut self, consumer: &mut Consumer) {
        while consumer.refill() {
            self.drain_cpu_writes();
            while consumer.has_slots() {
                if self.stopping.load(Ordering::Relaxed) {
                    self.abandon(consumer);
                    return;
                }
                let header = consumer.slot(0);
                if header == WRAP {
                    consumer.wrap();
                    continue;
                }
                consumer.begin(header);
                let len = self.execute(header, consumer);
                consumer.advance(len);
                if consumer.unreleased() >= RELEASE_SLOTS {
                    consumer.release();
                }
            }
            consumer.release();
        }
    }

    /// Skips every published record and drops every queued side command unexecuted, so a
    /// [`crate::Gpu::run_in_order`] action or a synchronize still queued drops its sender and its
    /// waiter returns.
    fn abandon(&mut self, consumer: &mut Consumer) {
        consumer.skip_published();
        consumer.release();
        while self.side.try_recv().is_ok() {}
    }

    /// Drops the renderer's copies of device memory the guest CPU wrote while a cache held it.
    /// Every write made before the work just published is seen here, ahead of that work.
    #[inline]
    fn drain_cpu_writes(&mut self) {
        if self.device.has_cpu_writes() {
            let rasterizer = &mut self.rasterizer;
            self.device
                .take_cpu_writes(|daddr, size| rasterizer.cpu_written(daddr, size));
        }
    }

    /// The back thread: runs until the front closes the stream.
    pub(crate) fn run_thread(mut self, mut consumer: Consumer) {
        let _liveness = consumer.liveness();
        let mut poll = MIN_POLL;
        loop {
            self.run(&mut consumer);
            let fences = !self.stopping.load(Ordering::Relaxed) && self.poll_fences();
            let idle = self.activity.waiting(Wait::BackIdle);
            let more = if fences {
                consumer.wait_more_for(poll)
            } else {
                Some(consumer.wait_more())
            };
            drop(idle);
            match more {
                Some(true) => poll = MIN_POLL,
                Some(false) => break,
                None => poll = (poll * 2).min(MAX_POLL),
            }
        }
    }

    /// The stream is empty: submits the recorded host GPU work and runs the actions of fences
    /// whose work finished, so that a syncpoint or semaphore the guest CPU waits on lands without
    /// more GPU work behind it. Whether fences are still pending, and the back should look again.
    pub(crate) fn poll_fences(&mut self) -> bool {
        if !self.rasterizer.has_pending_fences() {
            return false;
        }
        self.rasterizer.flush_commands();
        self.rasterizer.release_fences(false);
        self.rasterizer.has_pending_fences()
    }

    /// Executes the record whose header is `header`; returns its length in slots.
    #[inline]
    fn execute(&mut self, header: u64, consumer: &Consumer) -> usize {
        let engine = (header >> 8) as u8;
        let method = (header >> 16) as u32 & 0xffff;
        let argument = (header >> 32) as u32;
        let record = header as u8;
        if matches!(record, tag::METHOD | tag::MULTI | tag::UPLOAD_SOURCE) {
            let target = self.current.map(|index| (index, engine));
            if self.last != target {
                self.flush_engine();
                self.last = target;
            }
        } else {
            self.flush_engine();
        }
        match record {
            tag::METHOD => {
                self.with_engine(engine, |engine, ctx| {
                    engine.call_method(ctx, method, argument)
                });
                1
            }
            tag::MULTI => {
                let count = argument as usize;
                let slots = count.div_ceil(2);
                self.words.clear();
                for index in 0..slots {
                    let slot = consumer.slot(1 + index);
                    self.words.push(slot as u32);
                    self.words.push((slot >> 32) as u32);
                }
                self.words.truncate(count);
                let words = std::mem::take(&mut self.words);
                self.with_engine(engine, |engine, ctx| engine.call_multi(ctx, method, &words));
                self.words = words;
                1 + slots
            }
            tag::UPLOAD_SOURCE => {
                let address = consumer.slot(1);
                self.with_engine(engine, |engine, _| engine.upload_source(address));
                2
            }
            tag::CHANNEL => {
                let index = argument as usize;
                self.current = self
                    .channels
                    .get(index)
                    .is_some_and(Option::is_some)
                    .then_some(index);
                1
            }
            tag::BIND => {
                self.bind(engine, argument);
                1
            }
            tag::RELEASE => {
                let size = RELEASE_SIZES[method as usize % RELEASE_SIZES.len()];
                self.release(consumer.slot(1), consumer.slot(2), size);
                3
            }
            tag::REDUCE => {
                let op = REDUCTIONS[method as usize & 7];
                self.reduce(consumer.slot(1), argument, op, method & 0x10 != 0);
                2
            }
            tag::SYNCPOINT_INCREMENT => {
                if self.syncpoints.host().increment(argument).is_err() {
                    tracing::warn!(id = argument, "GPU increments an invalid syncpoint");
                } else {
                    self.rasterizer
                        .signal_fence(FenceAction::IncrementSyncpoint(argument));
                }
                1
            }
            tag::WAIT_FOR_IDLE => {
                let waiting = self.activity.waiting(Wait::HostGpu);
                self.rasterizer.wait_for_idle();
                drop(waiting);
                self.rasterizer.release_fences(false);
                1
            }
            tag::INVALIDATE_CACHES => {
                self.rasterizer.invalidate_gpu_caches();
                1
            }
            tag::SET_REFERENCE => {
                if let Some(channel) = self.channel() {
                    channel.setup.reference.store(argument, Ordering::Release);
                }
                1
            }
            tag::RELEASE_FENCES => {
                self.rasterizer.release_fences(false);
                1
            }
            tag::FLUSH_COMMANDS => {
                self.rasterizer.flush_commands();
                1
            }
            tag::SYNCHRONIZE_READ => {
                self.synchronize_read(consumer.slot(1), consumer.slot(2));
                3
            }
            tag::SYNCHRONIZE_FETCH => {
                let (address, size) = (consumer.slot(1), consumer.slot(2));
                let host = match method {
                    m if m == FetchSync::Host as u32 => true,
                    m if m == FetchSync::HostIfWritten as u32 => self.written(address, size),
                    _ => false,
                };
                if host {
                    self.synchronize_read(address, size);
                }
                self.fetch_syncs.complete();
                3
            }
            tag::SIDE => {
                match self.side.try_recv() {
                    Ok(command) => self.side(command),
                    Err(_) => tracing::error!("GPU side command missing from its queue"),
                }
                1
            }
            other => unreachable!("GPU stream record with tag {other}"),
        }
    }

    /// Lets the engine that ran the last method finish what it holds back.
    #[inline]
    fn flush_engine(&mut self) {
        if let Some((index, engine)) = self.last.take() {
            self.with_engine_of(index, engine, |engine, ctx| engine.flush(ctx));
        }
    }

    #[inline]
    fn with_engine(
        &mut self,
        engine: u8,
        call: impl FnOnce(&mut dyn Engine, &mut EngineContext<'_>),
    ) {
        if let Some(index) = self.current {
            self.with_engine_of(index, engine, call);
        }
    }

    #[inline]
    fn with_engine_of(
        &mut self,
        index: usize,
        engine: u8,
        call: impl FnOnce(&mut dyn Engine, &mut EngineContext<'_>),
    ) {
        let Some(channel) = self.channels.get_mut(index).and_then(Option::as_mut) else {
            return;
        };
        let Some(instance) = channel.engines[usize::from(engine)].as_deref_mut() else {
            return;
        };
        let mut ctx = EngineContext {
            channel: channel.setup.id,
            memory: &channel.setup.memory,
            rasterizer: &mut *self.rasterizer,
            syncpoints: &self.syncpoints,
            gob_layout: channel.setup.gob_layout,
            qmd_scheduler: &mut self.qmd_scheduler,
            written_pages: &self.written_pages,
            activity: &self.activity,
        };
        call(instance, &mut ctx);
    }

    fn channel(&mut self) -> Option<&mut BackChannel> {
        self.current.and_then(|index| self.channels[index].as_mut())
    }

    fn bind(&mut self, engine: u8, class: u32) {
        let factory = self.engines.clone();
        let Some(channel) = self.channel() else {
            return;
        };
        let kind = EngineKind::ALL[usize::from(engine)];
        let slot = &mut channel.engines[kind.index()];
        if slot.is_none() {
            *slot = factory(kind, class);
            if slot.is_none() {
                tracing::warn!(
                    ?kind,
                    class,
                    "no engine for a bound class; its methods are dropped"
                );
            }
        }
    }

    /// A semaphore release lands once the work before it completed; the fence action that writes
    /// it also resumes the channels its value meets.
    fn release(&mut self, address: u64, payload: u64, size: ReleaseSize) {
        let Some(index) = self.current else {
            return;
        };
        let Some(channel) = self.channels[index].as_ref() else {
            return;
        };
        let Some(daddr) = channel.setup.memory.translate(address) else {
            static UNMAPPED: Throttle = Throttle::new();
            if let Some(dropped) = UNMAPPED.allow() {
                tracing::warn!(address, dropped, "semaphore release to unmapped GPU memory");
            }
            return;
        };
        self.rasterizer.signal_fence(FenceAction::WriteSemaphore {
            address: daddr,
            value: payload,
            size,
        });
    }

    /// The front is about to read `[address, address + size)`: the GPU work before it completes,
    /// its fences and queries land, and GPU-written data in the range goes back to guest memory.
    fn synchronize_read(&mut self, address: u64, size: u64) {
        let _waiting = self.activity.waiting(Wait::HostGpu);
        let Some(channel) = self.current.and_then(|index| self.channels[index].as_ref()) else {
            self.rasterizer.release_fences(true);
            return;
        };
        let rasterizer = &mut *self.rasterizer;
        channel
            .setup
            .memory
            .for_each_range(address, size, |range| match range.target {
                RangeTarget::Device(daddr) => rasterizer.synchronize_region(daddr, range.size),
                _ => rasterizer.release_fences(true),
            });
    }

    /// Whether GPU work may have written a page of the current channel's `[address, +size)`.
    fn written(&self, address: u64, size: u64) -> bool {
        let Some(channel) = self.current.and_then(|index| self.channels[index].as_ref()) else {
            return false;
        };
        let mut written = false;
        channel.setup.memory.for_each_range(address, size, |range| {
            if let RangeTarget::Device(daddr) = range.target {
                written |= self.written_pages.may_be_written(daddr, range.size);
            }
        });
        written
    }

    fn reduce(&mut self, address: u64, payload: u32, op: crate::host::Reduction, signed: bool) {
        let Some(index) = self.current else {
            return;
        };
        let Some(channel) = self.channels[index].as_ref() else {
            return;
        };
        let Some(daddr) = channel.setup.memory.translate(address) else {
            tracing::warn!(address, "semaphore reduction on unmapped GPU memory");
            return;
        };
        self.rasterizer.signal_fence(FenceAction::ReduceSemaphore {
            address: daddr,
            op,
            signed,
            payload,
        });
    }

    fn side(&mut self, command: SideCommand) {
        match command {
            SideCommand::CreateChannel(setup) => {
                let index = setup.id.0 as usize;
                if self.channels.len() <= index {
                    self.channels.resize_with(index + 1, || None);
                }
                self.rasterizer.bind_channel(setup.id);
                self.channels[index] = Some(BackChannel {
                    setup,
                    engines: Default::default(),
                });
            }
            SideCommand::DestroyChannel(id) => {
                let index = id.0 as usize;
                if self
                    .channels
                    .get_mut(index)
                    .and_then(Option::take)
                    .is_some()
                {
                    self.rasterizer.release_channel(id);
                }
                if self.current == Some(index) {
                    self.current = None;
                }
            }
            SideCommand::BindAddressSpace(id, memory) => {
                if let Some(channel) = self
                    .channels
                    .get_mut(id.0 as usize)
                    .and_then(Option::as_mut)
                {
                    channel.setup.memory = memory;
                }
            }
            SideCommand::FlushRegion(addr, size) => self.rasterizer.flush_region(addr, size),
            SideCommand::UnmapRegion(addr, size) => self.rasterizer.unmap_region(addr, size),
            SideCommand::Present(framebuffer) => {
                let start = self.activity.begin_call(RendererCall::Present);
                self.rasterizer.present(&framebuffer);
                self.activity.renderer_call(RendererCall::Present, start);
                self.timestamp.frame_presented();
            }
            SideCommand::Run(action) => action(&mut *self.rasterizer),
            SideCommand::Signal(done) => {
                let _ = done.send(());
            }
        }
    }
}
