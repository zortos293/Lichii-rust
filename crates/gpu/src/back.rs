//! The GPU's back half: executes the stream in order on each channel's engines and runs the host
//! actions that need the renderer. The only code that touches the [`Rasterizer`].

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;

use lichii_common::DAddr;
use lichii_texture::block_linear::GobLayout;

use crate::backend::{
    ChannelId, FenceAction, FramebufferConfig, Query, QueryKind, Rasterizer, ReleaseSize,
};
use crate::channel::Waiters;
use crate::engines::{Engine, EngineContext, EngineFactory, EngineKind};
use crate::gmmu::Gmmu;
use crate::gpu::Inbox;
use crate::host1x::Syncpoints;
use crate::ring::{Consumer, WRAP};
use crate::stream::{MULTI_MAX_WORDS, REDUCTIONS, RELEASE_SIZES, tag};

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
    InvalidateRegion(DAddr, u64),
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
    syncpoints: Arc<Syncpoints>,
    engines: EngineFactory,
    side: mpsc::Receiver<SideCommand>,
    waiters: Arc<Waiters>,
    inbox: Arc<Inbox>,
    channels: Vec<Option<BackChannel>>,
    current: Option<usize>,
    /// The engine the last method went to (channel index, kind index), until it is flushed.
    last: Option<(usize, u8)>,
    words: Vec<u32>,
}

impl Back {
    pub(crate) fn new(
        rasterizer: Box<dyn Rasterizer>,
        syncpoints: Arc<Syncpoints>,
        engines: EngineFactory,
        side: mpsc::Receiver<SideCommand>,
        waiters: Arc<Waiters>,
        inbox: Arc<Inbox>,
    ) -> Self {
        Self {
            rasterizer,
            syncpoints,
            engines,
            side,
            waiters,
            inbox,
            channels: Vec::new(),
            current: None,
            last: None,
            words: Vec::with_capacity(MULTI_MAX_WORDS + 1),
        }
    }

    /// Executes everything published, until the ring is empty.
    pub(crate) fn run(&mut self, consumer: &mut Consumer) {
        while consumer.refill() {
            while consumer.has_slots() {
                let header = consumer.slot(0);
                if header == WRAP {
                    consumer.wrap();
                    continue;
                }
                let len = self.execute(header, consumer);
                consumer.advance(len);
                if consumer.unreleased() >= RELEASE_SLOTS {
                    consumer.release();
                }
            }
            consumer.release();
        }
    }

    /// The back thread: runs until the front closes the stream.
    pub(crate) fn run_thread(mut self, mut consumer: Consumer) {
        let _liveness = consumer.liveness();
        loop {
            self.run(&mut consumer);
            if !consumer.wait_more() {
                break;
            }
        }
    }

    /// Executes the record whose header is `header`; returns its length in slots.
    #[inline]
    fn execute(&mut self, header: u64, consumer: &Consumer) -> usize {
        let engine = (header >> 8) as u8;
        let method = (header >> 16) as u32 & 0xffff;
        let argument = (header >> 32) as u32;
        let record = header as u8;
        if record == tag::METHOD || record == tag::MULTI {
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
                self.rasterizer.wait_for_idle();
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

    fn release(&mut self, address: u64, payload: u64, size: ReleaseSize) {
        let Some(index) = self.current else {
            return;
        };
        let Some(channel) = self.channels[index].as_ref() else {
            return;
        };
        let memory = &channel.setup.memory;
        self.rasterizer.query(
            channel.setup.id,
            memory,
            &Query {
                address,
                kind: QueryKind::Payload,
                payload,
                size,
                after_work: true,
            },
        );
        let value = match size {
            ReleaseSize::Four => payload & 0xffff_ffff,
            ReleaseSize::Eight | ReleaseSize::Sixteen => payload,
        };
        if let Some(daddr) = memory.translate(address)
            && self.waiters.released(daddr, value)
        {
            self.inbox.wake();
        }
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
        self.rasterizer
            .signal_fence(crate::backend::FenceAction::ReduceSemaphore {
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
            SideCommand::InvalidateRegion(addr, size) => {
                self.rasterizer.invalidate_region(addr, size)
            }
            SideCommand::FlushRegion(addr, size) => self.rasterizer.flush_region(addr, size),
            SideCommand::UnmapRegion(addr, size) => self.rasterizer.unmap_region(addr, size),
            SideCommand::Present(framebuffer) => self.rasterizer.present(&framebuffer),
            SideCommand::Run(action) => action(&mut *self.rasterizer),
            SideCommand::Signal(done) => {
                let _ = done.send(());
            }
        }
    }
}
