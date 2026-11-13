//! The Switch 2 video decoder (`nn::codec::video::detail::ipc`). gsv 902, 905, 906 and 1002 hand
//! out a [`VideoDecoder`], whose TIPC 1 (`CreateVideoDecoderService`) hands out a
//! [`VideoDecoderService`].
//!
//! Nothing is decoded yet: every submitted input becomes one frame without a picture, and the
//! output image keeps what it held. The command flow is the reference's, so a movie plays at its
//! pts pace and ends. 311 and 314 hand an output image to the decoder and reply the
//! NativeSyncHandle `{~syncpoint, mode 0, u64 value}` that is reached once a frame is in it, on a
//! syncpoint the service owns. A worker thread fills the images in hand-out order (the order the
//! client acquires them from its own ring), each frame once it is due by its pts, and increments
//! the syncpoint; 213 then reports the frame. An end-of-stream input completes as an empty
//! end-of-stream frame, and Stop and Finalize complete every fence handed out.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use lichii_common::ResultCode;
use lichii_gpu::host1x::Syncpoints;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::{Event, GuestClock};
use parking_lot::{Condvar, Mutex};

use super::{UNKNOWN, push_readable, push_session};
use crate::nvdrv::Nvdrv;
use crate::{Service, ServiceContext};

/// `nn::codec` (module 111, shared with hwopus): the decoder cannot take another input now; the
/// client retries after the ready event.
const BUSY: ResultCode = ResultCode::new(111, 3603);
/// No finished frame: the code the client's own acquire uses for an empty ring.
const NO_OUTPUT: ResultCode = ResultCode::new(111, 3604);

/// Queued inputs below which the ready event is signalled.
const READY_WATERMARK: usize = 8;
/// Queued inputs from which inputs other than end of stream are refused. Well above the
/// watermark: a client may hold a ready signal when the event stops.
const REFUSE_AT: usize = READY_WATERMARK * 3;
/// The client clears the ready event after each wait, so while there is room it is signalled
/// again this often.
const READY_INTERVAL: Duration = Duration::from_millis(4);
/// A frame further than this (us) from the movie clock restarts it: a seek or a loop.
const MAX_DRIFT: u64 = 1_000_000;

/// gsv 902, 905, 906 and 1002. Without the decoder service its TIPC 1 creates,
/// `InitializeVp9Decoder` fails with InvalidHandle and the movie player aborts.
pub(super) struct VideoDecoder {
    pub context: Arc<ServiceContext>,
    pub name: &'static str,
}

impl Service for VideoDecoder {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => match VideoDecoderService::new(&self.context) {
                Ok(service) => push_session(&self.context, ctx, service),
                Err(code) => code,
            },
            _ => return None,
        })
    }
}

/// One decoder. Its syncpoint counts the frames put into output images.
pub(super) struct VideoDecoderService {
    nvdrv: Arc<Nvdrv>,
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

/// What the service thread and the worker share.
struct Shared {
    /// The guest clock, which stands still while emulation is paused: the movie clock runs on it.
    clock: Arc<dyn GuestClock>,
    syncpoints: Arc<Syncpoints>,
    syncpoint: u32,
    ready: Arc<Event>,
    state: Mutex<State>,
    wake: Condvar,
}

#[derive(Clone, Copy)]
struct Input {
    pts: u64,
    size: u64,
    end: bool,
}

struct Output {
    image: u32,
    width: u32,
    height: u32,
    pts: u64,
    end: bool,
}

struct State {
    width: u32,
    height: u32,
    inputs: VecDeque<Input>,
    /// The last refused input, queued ahead of the client's next one unless that is the same
    /// input sent again: some clients retry a refused input, others ignore the result.
    held: Option<Input>,
    /// Output images handed out and not filled yet, in the order the client acquires them. `None`
    /// is an image unregistered before it took a frame: its fence value is still counted in turn.
    free: VecDeque<Option<u32>>,
    /// Filled images, for 213.
    outputs: VecDeque<Output>,
    clock: MovieClock,
    submitted: u64,
    /// The client fetched the ready event (3).
    armed: bool,
    /// From creation or Start until Stop or Finalize.
    running: bool,
    end_queued: bool,
    /// The end-of-stream frame is out and nothing was submitted since.
    stream_ended: bool,
    quit: bool,
}

impl State {
    fn has_room(&self) -> bool {
        !self.stream_ended && !self.end_queued && self.inputs.len() < READY_WATERMARK
    }

    fn submit(&mut self, input: Input) -> bool {
        if let Some(held) = self.held.take()
            && (held.pts, held.size) != (input.pts, input.size)
        {
            self.inputs.push_back(held);
        }
        if !input.end && self.inputs.len() >= REFUSE_AT {
            self.held = Some(input);
            return false;
        }
        self.inputs.push_back(input);
        self.stream_ended = false;
        self.end_queued |= input.end;
        true
    }

    fn flush(&mut self) {
        self.inputs.clear();
        self.held = None;
        self.stream_ended = false;
        self.end_queued = false;
        self.clock = MovieClock::default();
    }
}

/// Maps a stream's pts to guest time (us). It is anchored at the first frame after a reset, so the
/// frames come out at their pts pace, and stands still while emulation is paused, as the movie's
/// audio does.
#[derive(Default)]
struct MovieClock(Option<(u64, u64)>);

impl MovieClock {
    /// When the frame at `pts` is due, in guest us.
    fn due(&mut self, now: u64, pts: u64) -> u64 {
        if let Some((start, base)) = self.0
            && let Some(due) = pts
                .checked_sub(base)
                .and_then(|offset| start.checked_add(offset))
            && due.abs_diff(now) <= MAX_DRIFT
        {
            return due;
        }
        self.0 = Some((now, pts));
        now
    }
}

impl VideoDecoderService {
    /// Fails until nvdrv, which allocates syncpoints, is registered, and when the worker thread
    /// cannot start.
    fn new(context: &ServiceContext) -> Result<Self, ResultCode> {
        let Some(nvdrv) = context.nvdrv.get().cloned() else {
            tracing::error!("nx2:vdec 1: nvdrv is not registered");
            return Err(UNKNOWN);
        };
        let syncpoint = nvdrv.allocate_syncpoint();
        tracing::info!("nx2:vdecsvc counts its frames on syncpoint {syncpoint}");
        let shared = Arc::new(Shared {
            clock: context.kernel.clock().clone(),
            syncpoints: context.gpu.host1x().syncpoints.clone(),
            syncpoint,
            ready: Arc::new(Event::new()),
            state: Mutex::new(State {
                width: 0,
                height: 0,
                inputs: VecDeque::new(),
                held: None,
                free: VecDeque::new(),
                outputs: VecDeque::new(),
                clock: MovieClock::default(),
                submitted: 0,
                armed: false,
                running: true,
                end_queued: false,
                stream_ended: false,
                quit: false,
            }),
            wake: Condvar::new(),
        });
        let worker = shared.clone();
        match std::thread::Builder::new()
            .name("nx2:vdec".into())
            .spawn(move || worker.run())
        {
            Ok(worker) => Ok(Self {
                nvdrv,
                shared,
                worker: Some(worker),
            }),
            Err(err) => {
                tracing::error!("spawning the nx2:vdec thread: {err}");
                nvdrv.retire_syncpoint(syncpoint);
                Err(UNKNOWN)
            }
        }
    }

    /// 311(u32 image) and 314(u32 image, u32, NativeSyncHandle release): queues the image for the
    /// next frame and replies the NativeSyncHandle reached once that frame is in it. The release
    /// fence guards the client's reads of the image, which nothing here writes.
    fn hand_out(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let image: u32 = ctx.pop();
        let shared = &self.shared;
        let value = {
            let mut state = shared.state.lock();
            let Ok(value) = shared.syncpoints.increase_max(shared.syncpoint, 1) else {
                return UNKNOWN;
            };
            state.free.push_back(Some(image));
            value
        };
        shared.wake.notify_one();
        ctx.push(!shared.syncpoint);
        ctx.push(0u32);
        ctx.push(u64::from(value));
        ResultCode::SUCCESS
    }

    /// 109(A: frame; u64 pts in us, u8 end of stream) and 400 (a frame information block in a
    /// second A buffer): replies the input's number, or busy while the queue is full.
    fn submit(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let input = Input {
            size: ctx.readable_buffer(0).map_or(0, |buffer| buffer.size),
            pts: ctx.pop(),
            end: ctx.pop::<u8>() != 0,
        };
        let id = {
            let mut state = self.shared.state.lock();
            if !state.submit(input) {
                return BUSY;
            }
            state.submitted += 1;
            state.submitted
        };
        self.shared.wake.notify_one();
        ctx.push(id);
        ResultCode::SUCCESS
    }

    /// 101(A: codec name; u32 width, u32 height, u32 rate, u32).
    fn configure(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let name = ctx.read_buffer(0).unwrap_or_default();
        let codec = name.split(|&b| b == 0).next().unwrap_or_default();
        let (width, height, rate) = (ctx.pop::<u32>(), ctx.pop::<u32>(), ctx.pop::<u32>());
        tracing::info!(
            "nx2:vdecsvc configure \"{}\" {width}x{height} rate {rate}; frames are not decoded",
            String::from_utf8_lossy(codec)
        );
        let mut state = self.shared.state.lock();
        state.width = width;
        state.height = height;
        state.clock = MovieClock::default();
        ResultCode::SUCCESS
    }

    /// 213: the next filled image (16 words): u32 image, u32 width, u32 height, u32 width,
    /// u32 height, u32 colour space, u64 pts in ns, u32 end of stream.
    fn output_info(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let Some(output) = self.shared.state.lock().outputs.pop_front() else {
            return NO_OUTPUT;
        };
        for word in [
            output.image,
            output.width,
            output.height,
            output.width,
            output.height,
            0,
        ] {
            ctx.push(word);
        }
        ctx.push(output.pts * 1000);
        ctx.push(u32::from(output.end));
        ctx.push_bytes(&[0; 28]);
        ResultCode::SUCCESS
    }

    /// 8 (Flush), 207 (Stop) and 115 (Finalize) drop queued input. Flush leaves the images handed
    /// out in place, so the fence values stay in step with the client's ring. Stop and Finalize
    /// complete their fences instead (the client acquires each as a frame and hands it back),
    /// and stop signalling the ready event.
    fn stop(&self, command: u32) -> ResultCode {
        let shared = &self.shared;
        let mut state = shared.state.lock();
        state.flush();
        if command == 8 {
            drop(state);
            shared.wake.notify_one();
            return ResultCode::SUCCESS;
        }
        state.running = false;
        shared.ready.clear();
        state.free.clear();
        let guest = shared.syncpoints.guest();
        let target = shared.syncpoints.max(shared.syncpoint).unwrap_or(0);
        while guest.is_reached(shared.syncpoint, target) == Ok(false) {
            let _ = guest.increment(shared.syncpoint);
        }
        ResultCode::SUCCESS
    }

    /// 6 (Start) and 3 (GetEvent): signals the ready event while there is room for input.
    fn signal_ready(&self, command: u32) {
        let mut state = self.shared.state.lock();
        if command == 6 {
            state.running = true;
        } else {
            state.armed = true;
        }
        if state.running && state.has_room() {
            self.shared.ready.signal();
        }
        self.shared.wake.notify_one();
    }
}

impl Shared {
    fn now_us(&self) -> u64 {
        (u128::from(self.clock.ticks()) * 1_000_000 / u128::from(self.clock.frequency())) as u64
    }

    fn increment(&self) {
        if self.syncpoints.guest().increment(self.syncpoint).is_err() {
            tracing::warn!("nx2:vdecsvc: syncpoint {} is out of range", self.syncpoint);
        }
    }

    /// The worker: puts each queued input, once it is due, into the next free image, and counts
    /// unregistered images off in their turn.
    fn run(&self) {
        let mut state = self.state.lock();
        while !state.quit {
            if state.free.front() == Some(&None) {
                state.free.pop_front();
                self.increment();
                continue;
            }
            let (now, host_now) = (self.now_us(), Instant::now());
            let mut wake = None;
            let next = state
                .inputs
                .front()
                .copied()
                .zip(state.free.front().copied().flatten());
            if let Some((input, image)) = next {
                let due = if input.end {
                    now
                } else {
                    state.clock.due(now, input.pts)
                };
                if due <= now {
                    state.inputs.pop_front();
                    state.free.pop_front();
                    let (width, height) = (state.width, state.height);
                    state.outputs.push_back(Output {
                        image,
                        width,
                        height,
                        pts: input.pts,
                        end: input.end,
                    });
                    if input.end {
                        state.stream_ended = true;
                        state.end_queued = false;
                        state.clock = MovieClock::default();
                    }
                    self.increment();
                    continue;
                }
                wake = Some(host_now + Duration::from_micros(due - now));
            }
            if state.armed && state.running && state.has_room() {
                self.ready.signal();
                let next = host_now + READY_INTERVAL;
                wake = Some(wake.map_or(next, |due: Instant| due.min(next)));
            }
            match wake {
                Some(at) => {
                    self.wake.wait_until(&mut state, at);
                }
                None => self.wake.wait(&mut state),
            }
        }
    }
}

impl Drop for VideoDecoderService {
    fn drop(&mut self) {
        self.shared.state.lock().quit = true;
        self.shared.wake.notify_one();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.nvdrv.retire_syncpoint(self.shared.syncpoint);
    }
}

impl Service for VideoDecoderService {
    fn name(&self) -> &str {
        "nx2:vdecsvc"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            3 => {
                self.signal_ready(3);
                push_readable(ctx, &self.shared.ready)
            }
            6 => {
                self.signal_ready(6);
                ResultCode::SUCCESS
            }
            command @ (8 | 115 | 207) => self.stop(command),
            101 => self.configure(ctx),
            109 | 400 => self.submit(ctx),
            200 | 201 | 202 | 210 => ResultCode::SUCCESS,
            212 => {
                let image: u32 = ctx.pop();
                let mut state = self.shared.state.lock();
                for free in state.free.iter_mut().filter(|free| **free == Some(image)) {
                    *free = None;
                }
                drop(state);
                self.shared.wake.notify_one();
                ResultCode::SUCCESS
            }
            213 => self.output_info(ctx),
            311 | 314 => self.hand_out(ctx),
            _ => return None,
        })
    }
}
