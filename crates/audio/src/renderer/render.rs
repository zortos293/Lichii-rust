//! The render loop (`audio_render_manager.cpp`, `system_manager.cpp` and the ADSP
//! `AudioRenderer` app): up to two renderer sessions, and one thread that, every 5 ms frame,
//! generates each active session's command list, runs it and appends the device sink's output to
//! the session's host stream.
//!
//! The thread is paced by the host streams: before a frame it waits until the session's stream
//! has room ([`SinkStream::wait_free_space`]), so the null sink and a host device both pull frames
//! at 48 kHz. A frame with no device sink appends silence, which keeps that clock running. Nothing
//! is allocated per frame once a session's command list has held its largest frame.
//!
//! The device sink interleaves as many channels as the guest's sink has inputs, so each frame is
//! spread into a six-channel block before it is appended; the stream downmixes that to the host's
//! channel count.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

use lichii_common::HleResult;
use lichii_memory::AddressSpace;
use parking_lot::{Condvar, MappedMutexGuard, Mutex, MutexGuard};

use super::parameter::{AudioRendererParameterInternal, ExecutionMode, work_buffer_size};
use super::result::{INVALID_REVISION, OUT_OF_SESSIONS};
use super::revision::is_valid_revision;
use super::{RendererSystem, TARGET_SAMPLE_COUNT};
use crate::dsp::{CommandListProcessor, DspMemory, HostClock};
use crate::sink::{MAX_CHANNELS, Sink, SinkStream, StreamType};

/// Renderer sessions open at once (`AudioCore::MaxRendererSessions`).
pub const MAX_RENDERER_SESSIONS: usize = 2;

/// Buffers a session's host stream queues before the loop waits (the ADSP's ring size).
const RING_SIZE: u32 = 4;

/// How long the loop sleeps when no session rendered a frame.
const IDLE_WAIT: Duration = Duration::from_millis(5);

const FRAME_SAMPLES: usize = TARGET_SAMPLE_COUNT as usize * MAX_CHANNELS;

/// Called after each rendered frame: signals the session's `IAudioRenderer` system event.
pub type RenderedCallback = Box<dyn Fn() + Send + Sync>;

/// `GetWorkBufferSize`: the transfer memory a session with `params` needs.
pub fn work_buffer_size_for(params: &AudioRendererParameterInternal) -> HleResult<u64> {
    if !is_valid_revision(params.revision) {
        return Err(INVALID_REVISION);
    }
    Ok(work_buffer_size(params))
}

struct Shared {
    sink: Arc<Sink>,
    slots: Mutex<[Option<Weak<RendererSession>>; MAX_RENDERER_SESSIONS]>,
    wake: Condvar,
    shutdown: AtomicBool,
}

/// The renderer sessions of one boot and the thread that renders them.
pub struct RenderManager {
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

impl RenderManager {
    /// A manager whose sessions play on `sink`. The render thread starts with the first session.
    pub fn new(sink: Arc<Sink>) -> Self {
        Self {
            shared: Arc::new(Shared {
                sink,
                slots: Mutex::new(Default::default()),
                wake: Condvar::new(),
                shutdown: AtomicBool::new(false),
            }),
            thread: Mutex::new(None),
        }
    }

    /// Sessions open now.
    pub fn session_count(&self) -> usize {
        self.shared.slots.lock().iter().flatten().count()
    }

    /// The open session with id `id`.
    pub fn session(&self, id: usize) -> Option<Arc<RendererSession>> {
        self.shared.slots.lock().get(id)?.as_ref()?.upgrade()
    }

    /// `OpenAudioRenderer`: builds a session from `params` in `transfer_memory_size` bytes of work
    /// buffer. Voices, sinks and memory pools address `memory`. Automatic sessions render every
    /// frame while started and call `rendered` after each; manual ones never render.
    pub fn open(
        &self,
        params: &AudioRendererParameterInternal,
        transfer_memory_size: u64,
        memory: Arc<AddressSpace>,
        rendered: RenderedCallback,
    ) -> HleResult<Arc<RendererSession>> {
        let mut slots = self.shared.slots.lock();
        let id = slots
            .iter()
            .position(Option::is_none)
            .ok_or(OUT_OF_SESSIONS)?;
        let mut system = RendererSystem::new(params, transfer_memory_size)?;
        system.session_id = id as i32;
        let stream = self.shared.sink.open_stream(
            MAX_CHANNELS,
            &format!("AudioRenderer-{id}"),
            StreamType::Render,
        );
        stream.set_ring_size(RING_SIZE);
        stream.start();
        let session = Arc::new(RendererSession {
            id,
            execution_mode: system.execution_mode,
            state: Mutex::new(SessionState {
                system,
                memory: GuestMemory(memory),
                processor: CommandListProcessor::new(),
                device: vec![0; FRAME_SAMPLES].into_boxed_slice(),
                frame: vec![0; FRAME_SAMPLES].into_boxed_slice(),
            }),
            stream,
            rendered,
            frames_rendered: AtomicU64::new(0),
            audible_frames: AtomicU64::new(0),
            shared: self.shared.clone(),
        });
        slots[id] = Some(Arc::downgrade(&session));
        drop(slots);
        self.start_thread();
        self.shared.wake.notify_all();
        Ok(session)
    }

    fn start_thread(&self) {
        let mut thread = self.thread.lock();
        if thread.is_some() {
            return;
        }
        let shared = self.shared.clone();
        match std::thread::Builder::new()
            .name("AudioRenderer".into())
            .spawn(move || render_loop(&shared))
        {
            Ok(handle) => *thread = Some(handle),
            Err(err) => tracing::error!("starting the audio render thread: {err}"),
        }
    }
}

impl Drop for RenderManager {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        self.shared.wake.notify_all();
        if let Some(thread) = self.thread.get_mut().take() {
            let _ = thread.join();
        }
    }
}

fn render_loop(shared: &Shared) {
    let clock = HostClock::default();
    let mut sessions: [Option<Arc<RendererSession>>; MAX_RENDERER_SESSIONS] = Default::default();
    while !shared.shutdown.load(Ordering::Acquire) {
        {
            let slots = shared.slots.lock();
            for (session, slot) in sessions.iter_mut().zip(slots.iter()) {
                *session = slot.as_ref().and_then(Weak::upgrade);
            }
        }
        let mut rendered = false;
        if !shared.sink.is_paused() {
            for session in sessions.iter().flatten() {
                rendered |= session.render_frame(&clock);
            }
        }
        sessions.iter_mut().for_each(|session| *session = None);
        if !rendered {
            let mut slots = shared.slots.lock();
            shared.wake.wait_for(&mut slots, IDLE_WAIT);
        }
    }
}

struct SessionState {
    system: RendererSystem,
    memory: GuestMemory,
    processor: CommandListProcessor,
    /// What the device sink wrote, interleaved by the guest sink's channel count.
    device: Box<[i16]>,
    /// The frame as the stream takes it: six channels per sample frame.
    frame: Box<[i16]>,
}

/// One `IAudioRenderer` session. Dropping it frees its session id and closes its stream.
pub struct RendererSession {
    id: usize,
    execution_mode: ExecutionMode,
    state: Mutex<SessionState>,
    stream: SinkStream,
    rendered: RenderedCallback,
    frames_rendered: AtomicU64,
    audible_frames: AtomicU64,
    shared: Arc<Shared>,
}

impl RendererSession {
    /// The session id (0 or 1).
    pub fn id(&self) -> usize {
        self.id
    }

    /// Whether the session renders every frame by itself (`Auto`) or only when the guest asks
    /// (`Manual`, which this port never renders). Fixed when the session opens.
    pub fn execution_mode(&self) -> ExecutionMode {
        self.execution_mode
    }

    /// The renderer state, for `RequestUpdate` and the getters. The render thread waits while it
    /// is held.
    pub fn system(&self) -> MappedMutexGuard<'_, RendererSystem> {
        MutexGuard::map(self.state.lock(), |state| &mut state.system)
    }

    /// Frames the loop rendered and appended to the host stream.
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered.load(Ordering::Acquire)
    }

    /// Rendered frames with a nonzero sample on any channel.
    pub fn audible_frames(&self) -> u64 {
        self.audible_frames.load(Ordering::Acquire)
    }

    /// The host stream the session plays on.
    pub fn stream(&self) -> &SinkStream {
        &self.stream
    }

    /// Renders one frame if the session is automatic and started. Returns whether it did.
    fn render_frame(&self, clock: &HostClock) -> bool {
        if self.execution_mode != ExecutionMode::Auto || !self.state.lock().system.active {
            return false;
        }
        self.stream.wait_free_space();
        let mut guard = self.state.lock();
        let state = &mut *guard;
        if !state.system.active {
            return false;
        }
        state
            .system
            .generate_command(self.stream.device_channels() as u32);
        let (list, frame) = state
            .system
            .dsp_frame(&mut state.memory, clock, &mut state.device);
        let report = state.processor.process(list, frame);
        let channels = report
            .device_sink
            .map_or(0, |sink| (sink.channel_count as usize).min(MAX_CHANNELS));
        to_stream_layout(&state.device, channels, &mut state.frame);
        let audible = state
            .frame
            .as_chunks::<MAX_CHANNELS>()
            .0
            .iter()
            .filter(|frame| frame.iter().any(|&sample| sample != 0))
            .count();
        self.stream
            .append_buffer(u64::from(TARGET_SAMPLE_COUNT), &state.frame);
        drop(guard);
        self.frames_rendered
            .fetch_add(u64::from(TARGET_SAMPLE_COUNT), Ordering::AcqRel);
        self.audible_frames
            .fetch_add(audible as u64, Ordering::AcqRel);
        (self.rendered)();
        true
    }
}

/// Spreads `device`, one frame of `channels`-channel samples, into six-channel `frame`, the
/// missing channels silent. No channels (no device sink ran) gives a silent frame.
fn to_stream_layout(device: &[i16], channels: usize, frame: &mut [i16]) {
    frame.fill(0);
    if channels == 0 {
        return;
    }
    let (out, _) = frame.as_chunks_mut::<MAX_CHANNELS>();
    for (out, samples) in out.iter_mut().zip(device.chunks_exact(channels)) {
        out[..channels].copy_from_slice(samples);
    }
}

impl Drop for RendererSession {
    fn drop(&mut self) {
        self.shared.slots.lock()[self.id] = None;
    }
}

/// Guest memory as the DSP sees it: the identity mapping of the client's address space. Reads
/// of unmapped memory give zeroes and writes to it are dropped.
struct GuestMemory(Arc<AddressSpace>);

impl DspMemory for GuestMemory {
    fn read(&self, address: u64, out: &mut [u8]) {
        if self.0.read_bytes(address, out).is_err() {
            out.fill(0);
        }
    }

    fn write(&mut self, address: u64, data: &[u8]) {
        let _ = self.0.write_bytes(address, data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_frames_spread_into_six_channel_blocks() {
        let mut frame = [7i16; FRAME_SAMPLES];
        let stereo: Vec<i16> = (0..TARGET_SAMPLE_COUNT as i16 * 2).collect();
        to_stream_layout(&stereo, 2, &mut frame);
        assert_eq!(
            frame[..MAX_CHANNELS * 2],
            [0, 1, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0]
        );
        let last = &frame[FRAME_SAMPLES - MAX_CHANNELS..];
        assert_eq!(last, [478, 479, 0, 0, 0, 0]);

        let surround: Vec<i16> = (0..FRAME_SAMPLES as i16).collect();
        to_stream_layout(&surround, MAX_CHANNELS, &mut frame);
        assert_eq!(frame[..], surround[..]);

        to_stream_layout(&surround, 0, &mut frame);
        assert!(frame.iter().all(|&sample| sample == 0));
    }
}
