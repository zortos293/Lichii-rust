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
    ChannelId, FenceAction, FramebufferConfig, GpuHandles, GpuWrittenPages, PendingSemaphore,
    Rasterizer, ReleaseSize, RenderCondition,
};
use crate::channel::SemaphoreAddress;
use crate::clock::GpuTimestamp;
use crate::cpu_reads::{CpuReads, QueuedRead};
use crate::engines::kepler_compute::pcas::QmdScheduler;
use crate::engines::maxwell_3d::{MACRO_REGISTERS_START, RENDER_ENABLE_MODE};
use crate::engines::{Engine, EngineContext, EngineFactory, EngineKind};
use crate::gmmu::{Gmmu, RangeTarget};
use crate::gpu::{MAX_POLL, MIN_POLL};
use crate::host1x::Syncpoints;
use crate::macros::{MacroDataRam, MacroEngine, MacroHost, ReadCondition, is_load_mme};
use crate::ring::{Consumer, WRAP};
use crate::stream::{
    FetchSync, FetchSyncs, MULTI_MAX_WORDS, OVERLAY_WORDS, REDUCTIONS, RELEASE_SIZES, tag,
};
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
    /// The macro code the current channel's [`tag::MACRO`] records run.
    MacroCode(Box<MacroEngine>),
    /// Puts the current channel's 3D registers into [`FetchSyncs::registers`]
    /// (see [`Stream::fetch_registers`](crate::stream::Stream::fetch_registers)).
    Registers,
    /// Reports that everything before it executed.
    Signal(mpsc::SyncSender<()>),
}

struct BackChannel {
    setup: BackChannelSetup,
    engines: [Option<Box<dyn Engine>>; EngineKind::COUNT],
    macros: Option<Box<MacroEngine>>,
    semaphore_dependency: u64,
}

/// How much of the device memory around a GPFIFO entry one fetch sync writes back. Switch 2
/// titles fetch command lists GPU work wrote moments before, often hundreds of small ones per
/// frame side by side (DKB: about 1,500 entries of 20 bytes in eleven 64 KiB blocks), so one
/// write-back covers the entries after it until GPU work writes the block again.
const FETCH_SYNC_WINDOW: u64 = 64 * 1024;

/// The most consecutive [`tag::METHOD`] records to one engine that run as one call. Guest CPU
/// reads are served between runs, so this bounds how long one waits.
const METHOD_RUN_MAX: usize = 256;

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
    /// Scratch for a run of [`tag::METHOD`] records to one engine.
    methods: Vec<(u32, u32)>,
    qmd_scheduler: QmdScheduler,
    fetch_syncs: Arc<FetchSyncs>,
    written_pages: Arc<GpuWrittenPages>,
    activity: Arc<Activity>,
    /// Set by [`crate::Gpu::shutdown`]: the back drops what is left instead of executing it.
    stopping: Arc<AtomicBool>,
    cpu_reads: Arc<CpuReads>,
    /// Scratch for [`CpuReads::serve`].
    cpu_read_ranges: Vec<QueuedRead>,
    /// The data RAM of macros the back runs: the front sends only calls that never use it.
    data_ram: MacroDataRam,
    /// The writes of a macro's two condition runs ([`run_as_condition`]), kept for their
    /// capacity.
    condition_writes: [Vec<(u32, u32)>; 2],
}

impl Drop for Back {
    fn drop(&mut self) {
        self.cpu_reads.close();
    }
}

impl Back {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        rasterizer: Box<dyn Rasterizer>,
        handles: &GpuHandles,
        engines: EngineFactory,
        side: mpsc::Receiver<SideCommand>,
        fetch_syncs: Arc<FetchSyncs>,
        activity: Arc<Activity>,
        stopping: Arc<AtomicBool>,
        cpu_reads: Arc<CpuReads>,
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
            methods: Vec::with_capacity(METHOD_RUN_MAX),
            qmd_scheduler: QmdScheduler::new(),
            fetch_syncs,
            written_pages: handles.written_pages.clone(),
            activity,
            stopping,
            cpu_reads,
            cpu_read_ranges: Vec::new(),
            data_ram: MacroDataRam::new(),
            condition_writes: [Vec::new(), Vec::new()],
        }
    }

    /// Executes everything published, until the ring is empty. Guest CPU reads are served first
    /// and between records (or runs of single-method records to one engine).
    pub(crate) fn run(&mut self, consumer: &mut Consumer) {
        self.serve_cpu_reads();
        while consumer.refill() {
            self.drain_cpu_writes();
            while consumer.has_slots() {
                if self.stopping.load(Ordering::Relaxed) {
                    self.abandon(consumer);
                    return;
                }
                self.serve_cpu_reads();
                let header = consumer.slot(0);
                if header == WRAP {
                    consumer.wrap();
                    continue;
                }
                consumer.begin(header);
                let len = self.execute(header, consumer);
                consumer.advance(len);
                if consumer.unreleased() >= RELEASE_SLOTS {
                    self.rasterizer.poll_semaphore_acquires();
                    consumer.release();
                }
            }
            self.rasterizer.poll_semaphore_acquires();
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

    /// Writes back the ranges guest CPU threads wait for, from the work executed so far.
    #[inline]
    fn serve_cpu_reads(&mut self) {
        if self.cpu_reads.pending() {
            self.serve_cpu_reads_now();
        }
    }

    #[cold]
    fn serve_cpu_reads_now(&mut self) {
        self.flush_engine();
        self.cpu_reads
            .serve(&mut self.cpu_read_ranges, &mut *self.rasterizer);
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

    /// The stream is empty: submits the recorded host GPU work a fence waits for and runs the
    /// actions of fences whose work finished, so that a syncpoint or semaphore the guest CPU waits
    /// on lands without more GPU work behind it. Whether fences are still pending, and the back
    /// should look again.
    pub(crate) fn poll_fences(&mut self) -> bool {
        if !self.rasterizer.has_pending_fences() {
            return false;
        }
        if self.rasterizer.has_unsubmitted_fences() {
            self.rasterizer.flush_commands();
        }
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
        if matches!(
            record,
            tag::METHOD | tag::MULTI | tag::UPLOAD_SOURCE | tag::MACRO
        ) {
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
                let run = consumer.available().min(METHOD_RUN_MAX);
                let len = (1..run)
                    .find(|&index| consumer.slot(index) & 0xffff != header & 0xffff)
                    .unwrap_or(run);
                let mut methods = std::mem::take(&mut self.methods);
                methods.clear();
                methods.extend((0..len).map(|index| {
                    let slot = consumer.slot(index);
                    ((slot >> 16) as u32 & 0xffff, (slot >> 32) as u32)
                }));
                self.with_engine(engine, |engine, ctx| engine.call_methods(ctx, &methods));
                self.methods = methods;
                len
            }
            tag::MULTI => {
                let slots = self.collect_words(argument as usize, consumer);
                let words = std::mem::take(&mut self.words);
                self.with_engine(engine, |engine, ctx| engine.call_multi(ctx, method, &words));
                self.words = words;
                1 + slots
            }
            tag::MACRO => {
                let slots = self.collect_words(argument as usize, consumer);
                let words = std::mem::take(&mut self.words);
                self.run_macro(engine, method, &words);
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
                let dependency = self.current.map_or(0, |index| {
                    self.channels[index].as_ref().unwrap().semaphore_dependency
                });
                self.syncpoints
                    .semaphore_waiters()
                    .select_dependency(dependency);
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
                self.rasterizer.release_fences(false);
                self.rasterizer.barrier();
                1
            }
            tag::INVALIDATE_CACHES => {
                self.rasterizer.invalidate_gpu_caches();
                1
            }
            tag::SET_REFERENCE => {
                if self
                    .syncpoints
                    .semaphore_waiters()
                    .has_unpublished_dependency()
                {
                    let _waiting = self.activity.waiting(Wait::HostGpu);
                    self.rasterizer.release_fences(true);
                }
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
                let (address, size) = (consumer.slot(1), consumer.slot(2));
                let pending = self.pending_releases(address, size);
                let waits = pending.is_none();
                self.fetch_syncs
                    .read_overlay
                    .set(pending.unwrap_or_default());
                if waits {
                    self.synchronize_read(address, size, 1);
                }
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
                    self.synchronize_read(address, size, FETCH_SYNC_WINDOW);
                }
                self.fetch_syncs.complete();
                3
            }
            tag::ACQUIRE => {
                self.acquire(consumer.slot(1), argument != 0);
                2
            }
            tag::SEMAPHORE_DEPENDENCY => {
                if let Some(channel) = self.channel() {
                    let dependency = channel.semaphore_dependency.max(consumer.slot(1));
                    channel.semaphore_dependency = dependency;
                    self.syncpoints
                        .semaphore_waiters()
                        .select_dependency(dependency);
                }
                2
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

    /// Copies the `count` words after the header into `words`; returns the slots they took.
    #[inline]
    fn collect_words(&mut self, count: usize, consumer: &Consumer) -> usize {
        let slots = count.div_ceil(2);
        self.words.clear();
        for index in 0..slots {
            let slot = consumer.slot(1 + index);
            self.words.push(slot as u32);
            self.words.push((slot >> 32) as u32);
        }
        self.words.truncate(count);
        slots
    }

    /// Runs a macro call the front sent (`tag::MACRO`) against the current channel's `engine`.
    fn run_macro(&mut self, engine: u8, start: u32, parameters: &[u32]) {
        let Some(channel) = self.current.and_then(|index| self.channels[index].as_mut()) else {
            return;
        };
        let Some(code) = channel.macros.as_deref_mut() else {
            tracing::error!(start, "macro call on the back before its code");
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
        if run_as_condition(
            code,
            &mut self.data_ram,
            instance,
            &mut ctx,
            start,
            parameters,
            &mut self.condition_writes,
        ) {
            return;
        }
        let mut host = BackMacroHost {
            engine: instance,
            ctx: &mut ctx,
        };
        if let Err(error) = code.call(start, parameters, &mut self.data_ram, &mut host) {
            tracing::warn!(%error, "macro call on the back failed");
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

    fn acquire(&mut self, address: u64, wide: bool) {
        let Some(channel) = self.current.and_then(|index| self.channels[index].as_ref()) else {
            return;
        };
        let memory = &channel.setup.memory;
        let at = SemaphoreAddress {
            low: memory.translate(address),
            high: wide.then(|| memory.translate(address + 4)).flatten(),
        };
        let pending = if let Some(address) = at.contiguous(wide) {
            self.rasterizer.pending_semaphore(address, wide)
        } else {
            [at.low, at.high]
                .into_iter()
                .flatten()
                .filter_map(|address| self.rasterizer.pending_semaphore(address, false))
                .map(|write| write.fence)
                .max()
                .map(|fence| PendingSemaphore { fence, value: None })
        };
        let PendingSemaphore { fence, value } = pending.unwrap_or(PendingSemaphore {
            fence: 0,
            value: None,
        });
        self.syncpoints
            .semaphore_waiters()
            .arm(channel.setup.id, fence, value);
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
        self.signal_host_semaphore(
            channel.setup.id,
            daddr,
            FenceAction::WriteSemaphore {
                address: daddr,
                value: payload,
                size,
            },
        );
    }

    /// The front is about to read `[address, address + size)`: the GPU work before it completes,
    /// its fences and queries land, and GPU-written data goes back to guest memory over the
    /// device range widened to whole `window`-aligned blocks (a power of two; 1 is the range).
    fn synchronize_read(&mut self, address: u64, size: u64, window: u64) {
        let _waiting = self.activity.waiting(Wait::HostGpu);
        let Some(channel) = self.current.and_then(|index| self.channels[index].as_ref()) else {
            self.rasterizer.release_fences(true);
            return;
        };
        synchronize_range(
            &channel.setup.memory,
            &mut *self.rasterizer,
            address,
            size,
            window,
        );
    }

    /// The words of `[address, address + size)` that semaphore releases still on their way write,
    /// by index, when GPU work never wrote its pages and the renderer holds no other pending write
    /// there: the read takes guest memory with these over it and needs no wait. None when the
    /// read must synchronize.
    fn pending_releases(
        &mut self,
        address: u64,
        size: u64,
    ) -> Option<[Option<u32>; OVERLAY_WORDS]> {
        let channel = self
            .current
            .and_then(|index| self.channels[index].as_ref())?;
        pending_words(
            &channel.setup.memory,
            &mut *self.rasterizer,
            &self.written_pages,
            address,
            size,
        )
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
        self.signal_host_semaphore(
            channel.setup.id,
            daddr,
            FenceAction::ReduceSemaphore {
                address: daddr,
                op,
                signed,
                payload,
            },
        );
    }

    fn signal_host_semaphore(&mut self, channel: ChannelId, address: DAddr, action: FenceAction) {
        if let Some(trace) = &self.activity.queue_trace {
            trace.track_producer(&mut *self.rasterizer, channel, address, |rasterizer| {
                rasterizer.signal_fence(action);
            });
        } else {
            self.rasterizer.signal_fence(action);
        }
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
                    macros: None,
                    semaphore_dependency: 0,
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
                    self.syncpoints.semaphore_waiters().select_dependency(0);
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
            SideCommand::MacroCode(code) => {
                if let Some(channel) = self.channel() {
                    channel.macros = Some(code);
                }
            }
            SideCommand::Registers => {
                let fetch_syncs = self.fetch_syncs.clone();
                let state = self
                    .channel()
                    .and_then(|channel| channel.engines[EngineKind::Maxwell3d.index()].as_deref())
                    .and_then(|engine| engine.graphics_state());
                if let Some(state) = state {
                    for (method, word) in fetch_syncs.registers.lock().iter_mut() {
                        *word = state.word(*method);
                    }
                }
            }
            SideCommand::Signal(done) => {
                let _ = done.send(());
            }
        }
    }
}

/// Has the GPU work before it complete and GPU-written data in `[address, address + size)` reach
/// guest memory, over the device range widened to whole `window`-aligned blocks (a power of two;
/// 1 is the range).
fn synchronize_range(
    memory: &Gmmu,
    rasterizer: &mut dyn Rasterizer,
    address: u64,
    size: u64,
    window: u64,
) {
    memory.for_each_range(address, size, |range| match range.target {
        RangeTarget::Device(daddr) => {
            let start = daddr & !(window - 1);
            let end = (daddr + range.size).next_multiple_of(window);
            rasterizer.synchronize_region(start, end - start);
        }
        _ => rasterizer.release_fences(true),
    });
}

/// The words of `[address, address + size)` that semaphore releases still on their way write,
/// by index, when GPU work never wrote its pages and the renderer holds no other pending write
/// there. None when a read must synchronize.
fn pending_words(
    memory: &Gmmu,
    rasterizer: &mut dyn Rasterizer,
    written_pages: &GpuWrittenPages,
    address: u64,
    size: u64,
) -> Option<[Option<u32>; OVERLAY_WORDS]> {
    if !address.is_multiple_of(4) || size > 4 * OVERLAY_WORDS as u64 {
        return None;
    }
    let mut words = [None; OVERLAY_WORDS];
    let mut known = true;
    memory.for_each_range(address, size, |range| {
        let RangeTarget::Device(daddr) = range.target else {
            known = false;
            return;
        };
        if !known || written_pages.may_be_written(daddr, range.size) {
            known = false;
            return;
        }
        let base = range.gpu_va - address;
        known = rasterizer.pending_semaphore_words(daddr, range.size, &mut |at, word| {
            words[((base + at - daddr) / 4) as usize] = Some(word);
        });
    });
    (known && words.iter().any(Option::is_some)).then_some(words)
}

/// Runs the macro at `start` without waiting for the SamplesPassed report it reads, when the
/// only use it makes of the report is whether its count is zero, to write RENDER_ENABLE_MODE:
/// its two condition runs, one for a zero and one for a nonzero count, write the same but for
/// that mode, which becomes a condition the renderer tests on the host GPU. False when the call
/// must run as usual.
fn run_as_condition(
    code: &mut MacroEngine,
    data_ram: &mut MacroDataRam,
    engine: &mut dyn Engine,
    ctx: &mut EngineContext<'_>,
    start: u32,
    parameters: &[u32],
    writes: &mut [Vec<(u32, u32)>; 2],
) -> bool {
    if !code.may_run_as_condition(start) {
        return false;
    }
    engine.flush(ctx);
    let shape = condition_shape(code, data_ram, engine, ctx, start, parameters, writes);
    code.note_condition_shape(start, shape.is_some());
    let Some((count, tests)) = shape else {
        return false;
    };
    let [zero, nonzero] = &*writes;
    let predicate = match tests {
        false => None,
        true => {
            let Some(predicate) = ctx
                .memory
                .translate(count)
                .and_then(|daddr| ctx.rasterizer.samples_predicate(ctx.channel, daddr))
            else {
                return false;
            };
            Some(predicate)
        }
    };
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if predicate.is_some() && !LOGGED.swap(true, Ordering::Relaxed) {
        tracing::info!(start, "3D macro's occlusion test runs on the host GPU");
    }
    for (&(method, value), &(_, nonzero_value)) in zero.iter().zip(nonzero) {
        if is_load_mme(method) || method >= MACRO_REGISTERS_START {
            continue;
        }
        match predicate {
            Some(predicate) if method == RENDER_ENABLE_MODE && value != nonzero_value => {
                let condition = RenderCondition {
                    predicate,
                    inverted: value == 1,
                };
                engine.set_render_condition(ctx, nonzero_value, condition);
            }
            _ => engine.call_method(ctx, method, value),
        }
    }
    true
}

/// The two condition runs of [`run_as_condition`] into `writes`: the count address they test
/// and whether their writes differ in RENDER_ENABLE_MODE alone. None when the call does not have
/// that shape.
fn condition_shape(
    code: &mut MacroEngine,
    data_ram: &mut MacroDataRam,
    engine: &dyn Engine,
    ctx: &mut EngineContext<'_>,
    start: u32,
    parameters: &[u32],
    writes: &mut [Vec<(u32, u32)>; 2],
) -> Option<(u64, bool)> {
    let mut count = None;
    for (differs, writes) in writes.iter_mut().enumerate() {
        writes.clear();
        let mut condition = ReadCondition::new(differs == 1);
        let mut host = ConditionHost {
            engine,
            ctx: &mut *ctx,
            writes,
        };
        if code
            .condition_execute(start, parameters, data_ram, &mut host, &mut condition)
            .is_err()
        {
            return None;
        }
        let tested = condition.tested()?;
        let mut words = [None; 2];
        for (address, expected) in tested {
            let first = *words[0].get_or_insert(address);
            let slot = match address.wrapping_sub(first) {
                0 => 0,
                4 => 1,
                _ => return None,
            };
            if expected != 0 {
                return None;
            }
            words[slot] = Some(address);
        }
        count = words[0];
    }
    let count = count?;
    let [zero, nonzero] = &*writes;
    if zero.len() != nonzero.len() {
        return None;
    }
    let mut tests = false;
    for (&(method, a), &(other, b)) in zero.iter().zip(nonzero) {
        if method != other {
            return None;
        }
        if a == b {
            continue;
        }
        if method != RENDER_ENABLE_MODE || !matches!((a, b), (0, 1) | (1, 0)) {
            return None;
        }
        tests = true;
    }
    Some((count, tests))
}

/// What a macro's condition run on the back ([`run_as_condition`]) runs against: the engine's
/// registers under the run's own writes, which it records instead of executing.
struct ConditionHost<'a, 'c> {
    engine: &'a dyn Engine,
    ctx: &'a mut EngineContext<'c>,
    writes: &'a mut Vec<(u32, u32)>,
}

impl MacroHost for ConditionHost<'_, '_> {
    fn read_register(&mut self, method: u32) -> u32 {
        if let Some(&(_, value)) = self.writes.iter().rev().find(|&&(m, _)| m == method) {
            return value;
        }
        self.engine
            .graphics_state()
            .and_then(|state| state.regs.words().get(method as usize).copied())
            .unwrap_or(0)
    }

    fn write_method(&mut self, method: u32, argument: u32) {
        self.writes.push((method, argument));
    }

    fn dma_read(&mut self, _gpu_va: u64, out: &mut [u32]) {
        out.fill(0);
    }

    /// The words no GPU work wrote and no pending write of the renderer other than a semaphore
    /// release reaches: guest memory under those releases' values.
    fn dma_peek(&mut self, gpu_va: u64, out: &mut [u32]) -> u64 {
        let memory = self.ctx.memory;
        let mut known = 0;
        for (index, word) in out.iter_mut().enumerate().take(64) {
            let address = gpu_va + 4 * index as u64;
            let Some(daddr) = memory.translate(address) else {
                continue;
            };
            if self.ctx.written_pages.may_be_written(daddr, 4) {
                continue;
            }
            let mut released = None;
            if !self
                .ctx
                .rasterizer
                .pending_semaphore_words(daddr, 4, &mut |_, value| released = Some(value))
            {
                continue;
            }
            let Some(value) = released.or_else(|| memory.read::<u32>(address).ok()) else {
                continue;
            };
            *word = value;
            known |= 1 << index;
        }
        known
    }
}

/// What a macro the back runs (`tag::MACRO`) runs against: the channel's engine itself.
struct BackMacroHost<'a, 'c> {
    engine: &'a mut dyn Engine,
    ctx: &'a mut EngineContext<'c>,
}

impl MacroHost for BackMacroHost<'_, '_> {
    fn read_register(&mut self, method: u32) -> u32 {
        self.engine
            .graphics_state()
            .and_then(|state| state.regs.words().get(method as usize).copied())
            .unwrap_or(0)
    }

    fn write_method(&mut self, method: u32, argument: u32) {
        if is_load_mme(method) || method >= MACRO_REGISTERS_START {
            return;
        }
        self.engine.call_method(self.ctx, method, argument);
    }

    /// Reads memory as the front would after a drain: the engine submits what it holds, and the
    /// words wait for the host GPU unless semaphore releases on their way hold them.
    fn dma_read(&mut self, gpu_va: u64, out: &mut [u32]) {
        self.engine.flush(self.ctx);
        let size = size_of_val(out) as u64;
        let memory = self.ctx.memory;
        let pending = pending_words(
            memory,
            &mut *self.ctx.rasterizer,
            self.ctx.written_pages,
            gpu_va,
            size,
        );
        if pending.is_none() {
            let _waiting = self.ctx.activity.waiting(Wait::HostGpu);
            synchronize_range(memory, &mut *self.ctx.rasterizer, gpu_va, size, 1);
        }
        if let Err(fault) = memory.read_block(gpu_va, bytemuck::cast_slice_mut(out)) {
            tracing::warn!(?fault, "MME DMA read from unmapped GPU memory");
            out.fill(0);
        }
        for (slot, word) in out.iter_mut().zip(pending.unwrap_or_default()) {
            if let Some(word) = word {
                *slot = word;
            }
        }
    }
}

#[cfg(test)]
mod tests;
