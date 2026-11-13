//! The Switch 2 video decoder (`nn::codec::video::detail::ipc`). gsv 902, 905, 906 and 1002 hand
//! out a [`VideoDecoder`], whose TIPC 1 (`CreateVideoDecoderService`) hands out a
//! [`VideoDecoderService`].
//!
//! 101 names the codec (`"video/vp9"`, ...) and 109 submits demuxed frames, which a worker thread
//! decodes with FFmpeg ([`lichii_video`]) a few frames ahead. 311 and 314 hand an output image (a
//! gsv 911 pool image) to the decoder and reply the NativeSyncHandle `{~syncpoint, mode 0, u64
//! value}` that is reached once a frame is in it, on a syncpoint the service owns. The worker
//! writes each decoded picture, once it is due by its pts, into the next image in hand-out order
//! (the order the client acquires them from its own ring) in the layout gsv 961 describes, drops
//! the renderer's cached copies of it (the title may have drawn into the image, which write
//! tracking leaves alone), and increments the syncpoint; 213 then reports the frame. An end-of-stream input completes after
//! the pictures the decoder still holds, and Stop and Finalize complete every fence handed out.
//!
//! Without FFmpeg, or for a codec it cannot decode, every input becomes one frame without a
//! picture, so a movie still plays at its pace (black) and ends.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use lichii_common::ResultCode;
use lichii_gpu::Gpu;
use lichii_gpu::backend::Rasterizer;
use lichii_gpu::gmmu::{BigPageSize, Gmmu, RangeTarget};
use lichii_gpu::host1x::Syncpoints;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::{Event, GuestClock};
use lichii_memory::AddressSpace;
use lichii_video::convert::{ColorOrder, write_nv12, write_rgba};
use lichii_video::{Codec, Decoder, Frame, Latency, VideoError};
use parking_lot::{Condvar, Mutex};

use super::native::{DecoderFormat, NativeObjects};
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
/// Decoded pictures kept ahead of the images they go into.
const DECODE_AHEAD: usize = 4;
/// How long a write waits for the client's reads of an image it gave back (314) to pass.
const RELEASE_WAIT: Duration = Duration::from_millis(100);

/// gsv 902, 905, 906 and 1002. Without the decoder service its TIPC 1 creates,
/// `InitializeVp9Decoder` fails with InvalidHandle and the movie player aborts.
pub(super) struct VideoDecoder {
    pub context: Arc<ServiceContext>,
    pub objects: Arc<Mutex<NativeObjects>>,
    pub name: &'static str,
}

impl Service for VideoDecoder {
    fn name(&self) -> &str {
        self.name
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        Some(match ctx.command() {
            1 => match VideoDecoderService::new(&self.context, &self.objects) {
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
    memory: Arc<AddressSpace>,
    gpu: Arc<Gpu>,
    /// Translates image addresses (GPU VA == CPU VA) to the device ranges the renderer caches.
    gmmu: Arc<Gmmu>,
    objects: Arc<Mutex<NativeObjects>>,
    ready: Arc<Event>,
    state: Mutex<State>,
    wake: Condvar,
}

struct Input {
    pts: u64,
    size: u64,
    end: bool,
    data: Vec<u8>,
}

/// An output image handed out, and the fence the client's reads of it end at (314).
#[derive(Clone, Copy, PartialEq, Eq)]
struct Handout {
    image: u32,
    release: Option<(u32, u32)>,
}

/// A decoded frame waiting for an image; `frame` is `None` for a frame without a picture.
struct Decoded {
    frame: Option<Frame>,
    pts: u64,
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
    codec: Option<Codec>,
    width: u32,
    height: u32,
    inputs: VecDeque<Input>,
    /// The last refused input, queued ahead of the client's next one unless that is the same
    /// input sent again: some clients retry a refused input, others ignore the result.
    held: Option<Input>,
    /// Input buffers the worker is done with, for the next submissions.
    spare: Vec<Vec<u8>>,
    decoded: VecDeque<Decoded>,
    /// Frames for the worker to decode into again.
    frames: Vec<Frame>,
    /// A flush came: the worker drops what its decoder holds.
    reset: bool,
    /// Output images handed out and not filled yet, in the order the client acquires them. `None`
    /// is an image unregistered before it took a frame: its fence value is still counted in turn.
    free: VecDeque<Option<Handout>>,
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

    /// Queues `input`; false when the queue is full.
    fn submit(&mut self, input: Input) -> bool {
        if let Some(held) = self.held.take() {
            if (held.pts, held.size) != (input.pts, input.size) {
                self.inputs.push_back(held);
            } else {
                self.spare.push(held.data);
            }
        }
        if !input.end && self.inputs.len() >= REFUSE_AT {
            self.held = Some(input);
            return false;
        }
        self.stream_ended = false;
        self.end_queued |= input.end;
        self.inputs.push_back(input);
        true
    }

    fn flush(&mut self) {
        let inputs: Vec<_> = self.inputs.drain(..).chain(self.held.take()).collect();
        self.spare
            .extend(inputs.into_iter().map(|input| input.data));
        let frames: Vec<_> = self.decoded.drain(..).filter_map(|d| d.frame).collect();
        self.frames.extend(frames);
        self.reset = true;
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
    fn new(
        context: &ServiceContext,
        objects: &Arc<Mutex<NativeObjects>>,
    ) -> Result<Self, ResultCode> {
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
            memory: context.process.memory().clone(),
            gpu: context.gpu.clone(),
            gmmu: context.gpu.create_address_space(BigPageSize::Size64K),
            objects: objects.clone(),
            ready: Arc::new(Event::new()),
            state: Mutex::new(State {
                codec: None,
                width: 0,
                height: 0,
                inputs: VecDeque::new(),
                held: None,
                spare: Vec::new(),
                decoded: VecDeque::new(),
                frames: Vec::new(),
                reset: false,
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
    /// next frame and replies the NativeSyncHandle reached once that frame is in it. A frame is
    /// written into a 314 image once the release fence guarding the client's reads has passed.
    fn hand_out(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let image: u32 = ctx.pop();
        let release = (ctx.command() == 314).then(|| {
            let _: u32 = ctx.pop();
            let (counter, _mode, value) = (ctx.pop::<u32>(), ctx.pop::<u32>(), ctx.pop::<u64>());
            (!counter, value as u32)
        });
        let shared = &self.shared;
        let value = {
            let mut state = shared.state.lock();
            let Ok(value) = shared.syncpoints.increase_max(shared.syncpoint, 1) else {
                return UNKNOWN;
            };
            state.free.push_back(Some(Handout { image, release }));
            value
        };
        tracing::debug!(
            "nx2:vdecsvc {} image {image} release {release:?} -> fence {value}",
            ctx.command()
        );
        shared.wake.notify_one();
        ctx.push(!shared.syncpoint);
        ctx.push(0u32);
        ctx.push(u64::from(value));
        ResultCode::SUCCESS
    }

    /// 109(A: frame; u64 pts in us, u8 end of stream) and 400 (a frame information block in a
    /// second A buffer): replies the input's number, or busy while the queue is full.
    fn submit(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let buffer = ctx.readable_buffer(0);
        let (pts, end) = (ctx.pop::<u64>(), ctx.pop::<u8>() != 0);
        let size = buffer.map_or(0, |buffer| buffer.size);
        let mut data = self.shared.state.lock().spare.pop().unwrap_or_default();
        data.clear();
        if let Some(buffer) = buffer {
            data.resize(buffer.size as usize, 0);
            if ctx.memory().read_bytes(buffer.address, &mut data).is_err() {
                tracing::warn!("nx2:vdecsvc: input at {:#x} is unreadable", buffer.address);
                data.clear();
            }
        }
        let input = Input {
            pts,
            size,
            end,
            data,
        };
        let id = {
            let mut state = self.shared.state.lock();
            if !state.submit(input) {
                return BUSY;
            }
            state.submitted += 1;
            state.submitted
        };
        tracing::debug!("nx2:vdecsvc input {id}: {size} bytes, pts {pts}, end {end}");
        self.shared.wake.notify_one();
        ctx.push(id);
        ResultCode::SUCCESS
    }

    /// 101(A: codec name; u32 width, u32 height, u32 rate, u32).
    fn configure(&self, ctx: &mut HleRequestContext) -> ResultCode {
        let name = ctx.read_buffer(0).unwrap_or_default();
        let mime = String::from_utf8_lossy(name.split(|&b| b == 0).next().unwrap_or_default());
        let (width, height, rate) = (ctx.pop::<u32>(), ctx.pop::<u32>(), ctx.pop::<u32>());
        let codec = Codec::from_mime(&mime);
        tracing::info!("nx2:vdecsvc configure \"{mime}\" {width}x{height} rate {rate}: {codec:?}");
        let mut state = self.shared.state.lock();
        state.codec = codec;
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
        drop(state);
        shared.wake.notify_one();
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

/// The worker's own decoding state.
struct Media {
    decoder: Option<Decoder>,
    /// The codec `decoder` was opened for, or failed to open for.
    codec: Option<Codec>,
    /// Pictures one input completed.
    decoded: Vec<Decoded>,
    /// One output image in its guest layout, written to guest memory in one go.
    staging: Vec<u8>,
    write_failed: bool,
    /// `LICHII_VDEC_DUMP`: a folder that gets the luma of every 120th frame, at a quarter of its
    /// size, as a PGM.
    dump: Option<std::path::PathBuf>,
    frames_out: u64,
    pictures: u64,
    decodes: u64,
    decode_time: Duration,
}

impl Media {
    /// Decodes one input into `self.decoded`, taking frames from and returning them to `frames`.
    fn decode(&mut self, codec: Option<Codec>, input: &Input, frames: &mut Vec<Frame>) {
        if codec != self.codec {
            self.codec = codec;
            self.decoder = codec.and_then(|codec| match Decoder::new(codec, Latency::Immediate) {
                Ok(decoder) => Some(decoder),
                Err(VideoError::Unavailable) => None,
                Err(err) => {
                    tracing::error!("nx2:vdecsvc: opening the {codec:?} decoder: {err}");
                    None
                }
            });
        }
        let Some(decoder) = &mut self.decoder else {
            self.decoded.push(Decoded {
                frame: None,
                pts: input.pts,
                end: input.end,
            });
            return;
        };
        let started = Instant::now();
        let sent = match decoder.send(&input.data, input.pts as i64) {
            _ if input.data.is_empty() => true,
            Ok(()) => true,
            Err(err) => {
                tracing::debug!(
                    "nx2:vdecsvc: input at pts {} does not decode: {err}",
                    input.pts
                );
                false
            }
        };
        if input.end {
            decoder.end_of_stream();
        }
        loop {
            let mut frame = match frames.pop().map(Ok).unwrap_or_else(Frame::new) {
                Ok(frame) => frame,
                Err(_) => break,
            };
            if decoder.receive(&mut frame) != Ok(true) {
                frames.push(frame);
                break;
            }
            let pts = frame.picture().map_or(input.pts, |p| p.pts.max(0) as u64);
            self.decoded.push(Decoded {
                frame: Some(frame),
                pts,
                end: false,
            });
        }
        self.decodes += 1;
        self.decode_time += started.elapsed();
        self.pictures += self.decoded.iter().filter(|d| d.frame.is_some()).count() as u64;
        if !sent && self.decoded.is_empty() {
            self.decoded.push(Decoded {
                frame: None,
                pts: input.pts,
                end: false,
            });
        }
        if input.end {
            match self.decoded.last_mut() {
                Some(last) => last.end = true,
                None => self.decoded.push(Decoded {
                    frame: None,
                    pts: input.pts,
                    end: true,
                }),
            }
        }
    }

    fn drop_queued(&mut self) {
        if let Some(decoder) = &mut self.decoder {
            decoder.flush();
        }
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

    /// Writes `frame`'s picture into `image`; returns the size the frame reports.
    fn write(
        &self,
        media: &mut Media,
        handout: Handout,
        frame: &Frame,
        size: (u32, u32),
    ) -> (u32, u32) {
        let Some(picture) = frame.picture() else {
            return size;
        };
        let Some(target) = self.objects.lock().decoder_image(handout.image) else {
            return (picture.width, picture.height);
        };
        if let Some((syncpoint, value)) = handout.release {
            let _ = self
                .syncpoints
                .guest()
                .wait(syncpoint, value, Some(RELEASE_WAIT));
        }
        let layout = target.layout;
        let (width, height) = (
            picture.width.min(layout.width),
            picture.height.min(layout.height),
        );
        let pitch = layout.pitch as usize;
        let rows = pitch * height as usize;
        media.staging.resize(layout.size as usize, 0);
        let staging = &mut media.staging;
        let written = match layout.format {
            DecoderFormat::Nv12 { chroma_offset } => {
                let chroma = chroma_offset as usize;
                write_nv12(&picture, staging, pitch, chroma, width, height)
                    && self
                        .memory
                        .write_bytes(target.address, &staging[..rows])
                        .is_ok()
                    && self
                        .memory
                        .write_bytes(
                            target.address + chroma_offset,
                            &staging[chroma..chroma + pitch * height.div_ceil(2) as usize],
                        )
                        .is_ok()
            }
            DecoderFormat::Rgba | DecoderFormat::Bgra => {
                let order = if layout.format == DecoderFormat::Rgba {
                    ColorOrder::Rgba
                } else {
                    ColorOrder::Bgra
                };
                write_rgba(&picture, staging, pitch, width, height, order)
                    && self
                        .memory
                        .write_bytes(target.address, &staging[..rows])
                        .is_ok()
            }
        };
        if written {
            let (gmmu, address, size) = (self.gmmu.clone(), target.address, layout.size);
            self.gpu
                .run_in_order(move |rasterizer: &mut dyn Rasterizer| {
                    gmmu.for_each_range(address, size, |range| {
                        if let RangeTarget::Device(daddr) = range.target {
                            rasterizer.invalidate_region(daddr, range.size);
                        }
                    });
                });
        }
        if media.frames_out == 0 {
            tracing::info!(
                "nx2:vdecsvc writes image {} at {:#x}: {layout:?}, written {written}",
                handout.image,
                target.address
            );
        }
        if written
            && media.frames_out.is_multiple_of(120)
            && let Some(dir) = &media.dump
        {
            let (w, h) = (width as usize / 4, height as usize / 4);
            let mut pgm = format!("P5 {w} {h} 255\n").into_bytes();
            pgm.extend(
                (0..h)
                    .flat_map(|y| (0..w).map(move |x| (y, x)))
                    .map(|(y, x)| staging[y * 4 * pitch + x * 4]),
            );
            let path = dir.join(format!("vdec-{:05}.pgm", media.frames_out));
            if let Err(err) = std::fs::write(&path, pgm) {
                tracing::warn!("nx2:vdecsvc: dumping {}: {err}", path.display());
            }
        }
        if !written && !media.write_failed {
            media.write_failed = true;
            tracing::warn!(
                "nx2:vdecsvc: cannot write a {width}x{height} frame into image {} at {:#x}",
                handout.image,
                target.address
            );
        }
        (width, height)
    }

    /// The worker: decodes queued input a few frames ahead, puts each decoded frame, once it is
    /// due, into the next free image, and counts unregistered images off in their turn.
    fn run(&self) {
        let mut media = Media {
            decoder: None,
            codec: None,
            decoded: Vec::new(),
            staging: Vec::new(),
            write_failed: false,
            dump: std::env::var_os("LICHII_VDEC_DUMP").map(Into::into),
            frames_out: 0,
            pictures: 0,
            decodes: 0,
            decode_time: Duration::ZERO,
        };
        let mut state = self.state.lock();
        while !state.quit {
            if std::mem::take(&mut state.reset) {
                media.drop_queued();
                continue;
            }
            if state.free.front() == Some(&None) {
                state.free.pop_front();
                self.increment();
                continue;
            }
            let (now, host_now) = (self.now_us(), Instant::now());
            let mut wake = None;
            if let (Some(decoded), Some(&Some(handout))) =
                (state.decoded.front(), state.free.front())
            {
                let due = if decoded.end && decoded.frame.is_none() {
                    now
                } else {
                    let pts = decoded.pts;
                    state.clock.due(now, pts)
                };
                if due <= now {
                    let decoded = state.decoded.pop_front().expect("checked above");
                    state.free.pop_front();
                    let size = (state.width, state.height);
                    let (width, height) = match &decoded.frame {
                        Some(frame) => {
                            drop(state);
                            let size = self.write(&mut media, handout, frame, size);
                            state = self.state.lock();
                            size
                        }
                        None => size,
                    };
                    state.frames.extend(decoded.frame);
                    state.outputs.push_back(Output {
                        image: handout.image,
                        width,
                        height,
                        pts: decoded.pts,
                        end: decoded.end,
                    });
                    media.frames_out += 1;
                    if media.frames_out == 1 || media.frames_out.is_multiple_of(600) || decoded.end
                    {
                        tracing::info!(
                            "nx2:vdecsvc frame {} ({width}x{height}, pts {} us{}) in image {}, \
                             {} decoded, {:.2} ms per decode",
                            media.frames_out,
                            decoded.pts,
                            if decoded.end { ", end of stream" } else { "" },
                            handout.image,
                            media.pictures,
                            media.decode_time.as_secs_f64() * 1000.0 / media.decodes.max(1) as f64,
                        );
                    }
                    if decoded.end {
                        state.stream_ended = true;
                        state.end_queued = false;
                        state.clock = MovieClock::default();
                    }
                    self.increment();
                    continue;
                }
                wake = Some(host_now + Duration::from_micros(due - now));
            }
            if state.decoded.len() < DECODE_AHEAD
                && let Some(input) = state.inputs.pop_front()
            {
                let codec = state.codec;
                let mut frames = std::mem::take(&mut state.frames);
                drop(state);
                media.decode(codec, &input, &mut frames);
                state = self.state.lock();
                state.frames.append(&mut frames);
                state.spare.push(input.data);
                if std::mem::take(&mut state.reset) {
                    media.drop_queued();
                    let dropped = media.decoded.drain(..).filter_map(|d| d.frame);
                    state.frames.extend(dropped);
                } else {
                    state.decoded.extend(media.decoded.drain(..));
                }
                continue;
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
        self.shared.gpu.destroy_address_space(&self.shared.gmmu);
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
                for free in state.free.iter_mut() {
                    if free.is_some_and(|handout| handout.image == image) {
                        *free = None;
                    }
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
