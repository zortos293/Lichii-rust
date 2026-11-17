//! Simulated frames through an inline back: one that drains the stream between bursts of work,
//! as a back faster than its front does, counting the host GPU submissions it costs, and a
//! microbenchmark of the per-method cost of 3D state writes.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use lichii_common::DAddr;
use lichii_memory::{DeviceAddressSpace, PhysicalMemory};
use lichii_texture::block_linear::GobLayout;

use super::{Back, BackChannelSetup, METHOD_RUN_MAX, SideCommand};
use crate::activity::Activity;
use crate::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles,
    GpuWrittenPages, GraphicsContext, IndirectDraw, Query, Rasterizer,
};
use crate::clock::{GpuTimestamp, GuestTime};
use crate::cpu_reads::CpuReads;
use crate::engines::maxwell_3d::Regs;
use crate::engines::{Engine, EngineContext, EngineFactory, EngineKind, create_engine, testing};
use crate::gmmu::{BigPageSize, Gmmu};
use crate::gpu::Inbox;
use crate::host1x::Syncpoints;
use crate::stream::{FetchSyncs, Link, MIN_RING_SLOTS, Stream};
use crate::{reg_index, ring};

/// A host GPU that finishes submitted work at once, tracked the way the Vulkan renderer tracks
/// it: the work recorded since the last submission completes at `tick`, and each fence waits for
/// the tick it was signalled in.
struct HostQueue {
    handles: GpuHandles,
    tick: u64,
    recorded: bool,
    fences: VecDeque<(u64, FenceAction)>,
    submits: Arc<AtomicUsize>,
}

impl Rasterizer for HostQueue {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {
        self.recorded = true;
    }
    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {
        self.recorded = true;
    }
    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {
        self.recorded = true;
    }
    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {
        self.recorded = true;
    }
    fn barrier(&mut self) {
        self.recorded = true;
    }
    fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, _query: &Query) {}
    fn signal_fence(&mut self, action: FenceAction) {
        self.fences.push_back((self.tick, action));
        self.recorded = true;
    }
    fn flush_commands(&mut self) {
        if self.recorded {
            self.recorded = false;
            self.tick += 1;
            self.submits.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn release_fences(&mut self, wait: bool) {
        if wait {
            self.flush_commands();
        }
        while let Some(&(tick, action)) = self.fences.front() {
            if tick >= self.tick {
                break;
            }
            self.fences.pop_front();
            action.execute(&self.handles);
        }
    }
    fn has_pending_fences(&self) -> bool {
        !self.fences.is_empty()
    }
    fn has_unsubmitted_fences(&self) -> bool {
        self.fences
            .back()
            .is_some_and(|&(tick, _)| tick >= self.tick)
    }
    fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}
    fn flush_region(&mut self, _addr: DAddr, _size: u64) {}
    fn wait_for_idle(&mut self) {
        self.release_fences(true);
    }
    fn present(&mut self, _framebuffer: &FramebufferConfig) {
        self.flush_commands();
    }
}

struct Boot;

impl GuestTime for Boot {
    fn global_time_ns(&self) -> u64 {
        0
    }
}

/// GPFIFO submissions in a simulated frame.
const SUBMISSIONS: u32 = 4;
/// Times the back drains the stream within one submission.
const GAPS: u32 = 16;
/// The syncpoint each submission ends with.
const FRAME: u32 = 1;
/// The syncpoint a release in the middle of each submission increments.
const RELEASE: u32 = 2;

fn handles(syncpoints: &Arc<Syncpoints>) -> GpuHandles {
    let physical = Arc::new(PhysicalMemory::new(1 << 16).unwrap());
    let device = Arc::new(DeviceAddressSpace::new(physical, 36));
    GpuHandles {
        lazy: Arc::new(crate::lazy::LazyDeviceRanges::new(device.clone())),
        device,
        syncpoints: syncpoints.clone(),
        timestamp: Arc::new(GpuTimestamp::new(Arc::new(Boot), false)),
        written_pages: Arc::new(GpuWrittenPages::new(36)),
        activity: Arc::new(Activity::new()),
    }
}

/// A front stream of `slots` ring slots whose back runs inline on `rasterizer` and `engines`.
fn inline_stream(
    rasterizer: Box<dyn Rasterizer>,
    engines: EngineFactory,
    handles: &GpuHandles,
    slots: usize,
) -> Stream {
    let (producer, consumer) = ring::ring(slots, MIN_RING_SLOTS);
    let (side, side_receiver) = mpsc::channel();
    let fetch_syncs = Arc::new(FetchSyncs::new(Arc::new(Inbox::new())));
    let back = Back::new(
        rasterizer,
        handles,
        engines,
        side_receiver,
        fetch_syncs.clone(),
        Arc::new(Activity::new()),
        Arc::default(),
        Arc::new(CpuReads::new(Box::new(|| {}))),
    );
    Stream::new(
        producer,
        Link::Inline(Box::new((back, consumer))),
        side,
        fetch_syncs,
        Arc::new(Activity::new()),
    )
}

#[test]
fn the_idle_back_submits_only_the_work_a_fence_waits_for() {
    let syncpoints = Arc::new(Syncpoints::new());
    let handles = handles(&syncpoints);
    let submits = Arc::new(AtomicUsize::new(0));
    let host = HostQueue {
        handles: handles.clone(),
        tick: 0,
        recorded: false,
        fences: VecDeque::new(),
        submits: submits.clone(),
    };
    let mut stream = inline_stream(
        Box::new(host),
        Arc::new(create_engine),
        &handles,
        MIN_RING_SLOTS,
    );
    let guest = |id| syncpoints.guest().value(id).unwrap();
    let frames = 8;
    for frame in 0..frames {
        for submission in 0..SUBMISSIONS {
            let done = frame * SUBMISSIONS + submission;
            for gap in 0..GAPS {
                stream.wait_for_idle();
                if gap == GAPS / 2 {
                    stream.syncpoint_increment(RELEASE);
                }
                stream.idle();
                let released = done + u32::from(gap >= GAPS / 2);
                assert_eq!(
                    guest(RELEASE),
                    released,
                    "a release lands in the idle poll after it, not at the end of its submission"
                );
            }
            stream.syncpoint_increment(FRAME);
            stream.flush_commands();
            assert!(!stream.idle(), "every fence ran once the submission ended");
            assert_eq!(guest(FRAME), done + 1);
        }
    }
    let per_frame = submits.load(Ordering::Relaxed) as f64 / f64::from(frames);
    println!("host GPU submissions per simulated frame: {per_frame}");
    assert_eq!(
        per_frame,
        f64::from(2 * SUBMISSIONS),
        "one submission for each mid-submission release and one for each submission's end"
    );
}

/// State words a simulated draw writes, each to its own method.
const STATE_WORDS: u32 = 40;
/// Simulated draws per frame (MKW on the PC: about 2,700).
const DRAWS_PER_FRAME: u32 = 2_700;

#[test]
#[ignore = "microbenchmark: cargo test --release -p lichii-gpu --lib -- --ignored --nocapture state_methods"]
fn state_methods_per_frame() {
    let syncpoints = Arc::new(Syncpoints::new());
    let handles = handles(&syncpoints);
    let slots = (2 * DRAWS_PER_FRAME * STATE_WORDS).next_power_of_two() as usize;
    let mut stream = inline_stream(
        Box::new(testing::Recorder::default()),
        Arc::new(create_engine),
        &handles,
        slots,
    );
    stream.side(SideCommand::CreateChannel(BackChannelSetup {
        id: ChannelId(0),
        memory: Arc::new(Gmmu::new(handles.device.clone(), BigPageSize::Size64K)),
        gob_layout: GobLayout::Tegra,
        reference: Arc::default(),
    }));
    stream.select(ChannelId(0));
    stream.bind(EngineKind::Maxwell3d, 0xb197);
    let first = reg_index!(Regs, viewport_transform);
    let frames = 200;
    let mut back = 0.0;
    for frame in 0..frames {
        for draw in 0..DRAWS_PER_FRAME {
            for word in 0..STATE_WORDS {
                stream.engine_word(EngineKind::Maxwell3d, first + word, frame ^ draw ^ word);
            }
        }
        stream.flush_pending();
        let start = Instant::now();
        stream.idle();
        back += start.elapsed().as_secs_f64();
    }
    let methods = f64::from(frames * DRAWS_PER_FRAME * STATE_WORDS);
    println!(
        "{:.2} ns per state method on the back, {:.3} ms per frame of {} methods",
        back * 1e9 / methods,
        back * 1e3 / f64::from(frames),
        DRAWS_PER_FRAME * STATE_WORDS
    );
}

/// Each write an engine got, and how many calls brought them.
#[derive(Default)]
struct Writes {
    writes: Vec<(EngineKind, u32, u32)>,
    calls: usize,
}

struct Logger(EngineKind, Arc<Mutex<Writes>>);

impl Engine for Logger {
    fn call_method(&mut self, _ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        let mut log = self.1.lock().unwrap();
        log.writes.push((self.0, method, argument));
        log.calls += 1;
    }
    fn call_methods(&mut self, _ctx: &mut EngineContext<'_>, methods: &[(u32, u32)]) {
        let mut log = self.1.lock().unwrap();
        log.writes.extend(
            methods
                .iter()
                .map(|&(method, argument)| (self.0, method, argument)),
        );
        log.calls += 1;
    }
    fn call_multi(&mut self, _ctx: &mut EngineContext<'_>, method: u32, arguments: &[u32]) {
        let mut log = self.1.lock().unwrap();
        log.writes
            .extend(arguments.iter().map(|&argument| (self.0, method, argument)));
        log.calls += 1;
    }
}

#[test]
fn runs_of_single_method_records_reach_their_engine_in_order() {
    let syncpoints = Arc::new(Syncpoints::new());
    let handles = handles(&syncpoints);
    let log = Arc::new(Mutex::new(Writes::default()));
    let engines: EngineFactory = {
        let log = log.clone();
        Arc::new(move |kind, _| Some(Box::new(Logger(kind, log.clone())) as Box<dyn Engine>))
    };
    let mut stream = inline_stream(
        Box::new(testing::Recorder::default()),
        engines,
        &handles,
        1 << 12,
    );
    stream.side(SideCommand::CreateChannel(BackChannelSetup {
        id: ChannelId(0),
        memory: Arc::new(Gmmu::new(handles.device.clone(), BigPageSize::Size64K)),
        gob_layout: GobLayout::Tegra,
        reference: Arc::default(),
    }));
    stream.select(ChannelId(0));
    stream.bind(EngineKind::Maxwell3d, 0xb197);
    stream.bind(EngineKind::Dma, 0xb0b5);
    let long = METHOD_RUN_MAX as u32 + 10;
    let mut sent = Vec::new();
    let mut send = |stream: &mut Stream, kind, method, argument| {
        stream.engine_word(kind, method, argument);
        sent.push((kind, method, argument));
    };
    for i in 0..long {
        send(&mut stream, EngineKind::Maxwell3d, 0x100 + i, i);
    }
    send(&mut stream, EngineKind::Dma, 0x100, 7);
    send(&mut stream, EngineKind::Dma, 0x101, 8);
    for i in 0..3 {
        send(&mut stream, EngineKind::Maxwell3d, 0x200, i);
    }
    send(&mut stream, EngineKind::Maxwell3d, 0x201, 9);
    stream.syncpoint_increment(0);
    send(&mut stream, EngineKind::Maxwell3d, 0x202, 10);
    stream.idle();
    let log = log.lock().unwrap();
    assert_eq!(log.writes, sent);
    assert_eq!(
        log.calls,
        2 + 1 + 1 + 1 + 1,
        "a run longer than the cap takes two calls, then one each for the DMA pair, the \
         multi-word record, the single write before the increment and the one after it"
    );
}
