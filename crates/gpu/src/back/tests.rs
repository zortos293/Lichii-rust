//! A simulated frame that drains the stream between bursts of work, as a back faster than its
//! front does, counting the host GPU submissions it costs.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;

use lichii_common::DAddr;
use lichii_memory::{DeviceAddressSpace, PhysicalMemory};

use super::Back;
use crate::activity::Activity;
use crate::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GpuHandles,
    GpuWrittenPages, GraphicsContext, IndirectDraw, Query, Rasterizer,
};
use crate::clock::{GpuTimestamp, GuestTime};
use crate::cpu_reads::CpuReads;
use crate::engines::create_engine;
use crate::gmmu::Gmmu;
use crate::gpu::Inbox;
use crate::host1x::Syncpoints;
use crate::ring;
use crate::stream::{FetchSyncs, Link, MIN_RING_SLOTS, Stream};

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

#[test]
fn the_idle_back_submits_only_the_work_a_fence_waits_for() {
    let syncpoints = Arc::new(Syncpoints::new());
    let physical = Arc::new(PhysicalMemory::new(1 << 16).unwrap());
    let device = Arc::new(DeviceAddressSpace::new(physical, 36));
    let handles = GpuHandles {
        lazy: Arc::new(crate::lazy::LazyDeviceRanges::new(device.clone())),
        device,
        syncpoints: syncpoints.clone(),
        timestamp: Arc::new(GpuTimestamp::new(Arc::new(Boot), false)),
        written_pages: Arc::new(GpuWrittenPages::new(36)),
        activity: Arc::new(Activity::new()),
    };
    let submits = Arc::new(AtomicUsize::new(0));
    let host = HostQueue {
        handles: handles.clone(),
        tick: 0,
        recorded: false,
        fences: VecDeque::new(),
        submits: submits.clone(),
    };
    let (producer, consumer) = ring::ring(MIN_RING_SLOTS, MIN_RING_SLOTS);
    let (side, side_receiver) = mpsc::channel();
    let fetch_syncs = Arc::new(FetchSyncs::new(Arc::new(Inbox::new())));
    let back = Back::new(
        Box::new(host),
        &handles,
        Arc::new(create_engine),
        side_receiver,
        fetch_syncs.clone(),
        Arc::new(Activity::new()),
        Arc::default(),
        Arc::new(CpuReads::new(Box::new(|| {}))),
    );
    let mut stream = Stream::new(
        producer,
        Link::Inline(Box::new((back, consumer))),
        side,
        fetch_syncs,
        Arc::new(Activity::new()),
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
