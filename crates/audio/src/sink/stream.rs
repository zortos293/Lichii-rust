//! One host stream: the buffer queue the emulator fills and the callback that drains it.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::{
    MAX_CHANNELS, SAMPLE_RATE, SinkShared, TARGET_SAMPLE_COUNT, load_f32, null, store_f32,
};

/// Samples (across all channels) a stream's ring starts with: four 6-channel buffers of 10k
/// frames. Larger buffers grow it, up to [`MAX_RING_CAPACITY`].
const RING_CAPACITY: usize = 1 << 18;

/// The most samples a ring grows to (32 MiB). A buffer that does not fit is cut short, and only
/// the frames kept are queued, so play positions stay in step with the samples.
const MAX_RING_CAPACITY: usize = 1 << 24;

/// Buffers a stream may queue before `append_buffer` drops the oldest one.
const QUEUE_CAPACITY: usize = 64;

/// How many frames ahead of the real play position [`SinkStream::expected_played_sample_count`]
/// reports, so the guest gets buffers back before the host runs dry (25 ms, as Eden does).
const PLAYED_LEAD: u64 = TARGET_SAMPLE_COUNT as u64 * 5;

/// What a stream carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StreamType {
    /// The audio renderer's final mix.
    Render,
    /// An `IAudioOut` session.
    Out,
    /// An `IAudioIn` session (captures silence).
    In,
}

#[derive(Clone, Copy, Debug, Default)]
struct SinkBuffer {
    frames: u64,
    frames_played: u64,
    consumed: bool,
}

/// Fixed-size ring of interleaved samples.
struct SampleRing {
    data: Box<[i16]>,
    read: usize,
    len: usize,
}

impl SampleRing {
    fn new() -> Self {
        Self {
            data: vec![0; RING_CAPACITY].into_boxed_slice(),
            read: 0,
            len: 0,
        }
    }

    /// Grows the ring so `additional` more samples fit, within [`MAX_RING_CAPACITY`], and
    /// returns how many fit now. Allocates only when a buffer outgrows the ring.
    fn reserve(&mut self, additional: usize) -> usize {
        let needed = self.len + additional;
        if needed > self.data.len() && self.data.len() < MAX_RING_CAPACITY {
            let mut data = vec![0; needed.next_power_of_two().min(MAX_RING_CAPACITY)];
            let len = self.len;
            self.pop_into(&mut data[..len]);
            self.data = data.into_boxed_slice();
            self.read = 0;
            self.len = len;
        }
        self.data.len() - self.len
    }

    fn push(&mut self, sample: i16) {
        debug_assert!(self.len < self.data.len());
        let write = (self.read + self.len) % self.data.len();
        self.data[write] = sample;
        self.len += 1;
    }

    /// Fills `out` from the ring; samples the ring does not have read as silence.
    fn pop_into(&mut self, out: &mut [i16]) {
        let take = out.len().min(self.len);
        let first = take.min(self.data.len() - self.read);
        out[..first].copy_from_slice(&self.data[self.read..self.read + first]);
        out[first..take].copy_from_slice(&self.data[..take - first]);
        out[take..].fill(0);
        self.read = (self.read + take) % self.data.len();
        self.len -= take;
    }

    fn clear(&mut self) {
        self.read = 0;
        self.len = 0;
    }
}

struct StreamInner {
    ring: SampleRing,
    queue: VecDeque<SinkBuffer>,
    playing: SinkBuffer,
    last_frame: [i16; MAX_CHANNELS],
    appended_frames: u64,
    min_played: u64,
    max_played: u64,
    last_update: Instant,
    overflow_logged: bool,
}

/// A host stream and the null clock it falls back to when the device refuses to start.
pub(crate) struct HostStream {
    pub(crate) stream: cpal::Stream,
    pub(crate) fallback: null::PacerHandle,
}

pub(crate) struct StreamState {
    ty: StreamType,
    name: String,
    system_channels: usize,
    device_channels: usize,
    sink: Arc<SinkShared>,
    paused: AtomicBool,
    queued_buffers: AtomicU32,
    max_queue_size: AtomicU32,
    system_volume: AtomicU32,
    host_failed: AtomicBool,
    inner: Mutex<StreamInner>,
    released: Condvar,
}

impl StreamState {
    pub(crate) fn new(
        ty: StreamType,
        name: &str,
        system_channels: usize,
        device_channels: usize,
        sink: Arc<SinkShared>,
    ) -> Self {
        Self {
            ty,
            name: name.into(),
            system_channels,
            device_channels,
            sink,
            paused: AtomicBool::new(true),
            queued_buffers: AtomicU32::new(0),
            max_queue_size: AtomicU32::new(0),
            system_volume: AtomicU32::new(1.0f32.to_bits()),
            host_failed: AtomicBool::new(false),
            inner: Mutex::new(StreamInner {
                ring: SampleRing::new(),
                queue: VecDeque::with_capacity(QUEUE_CAPACITY),
                playing: SinkBuffer::default(),
                last_frame: [0; MAX_CHANNELS],
                appended_frames: 0,
                min_played: 0,
                max_played: 0,
                last_update: Instant::now(),
                overflow_logged: false,
            }),
            released: Condvar::new(),
        }
    }

    pub(crate) fn ty(&self) -> StreamType {
        self.ty
    }

    pub(crate) fn device_channels(&self) -> usize {
        self.device_channels
    }

    pub(crate) fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    fn silent(&self) -> bool {
        self.is_paused() || self.sink.paused.load(Ordering::Acquire)
    }

    /// The host output callback: fills `output` (interleaved, `device_channels` per frame) from
    /// the queued buffers. An underrun repeats the last frame, as Eden's sink does, so a late
    /// buffer does not click. Allocates nothing.
    pub(crate) fn process_output(&self, output: &mut [i16]) {
        let channels = self.device_channels;
        let frames = output.len() / channels;
        if frames == 0 {
            return;
        }
        if self.silent() {
            output.fill(0);
            return;
        }
        let mut inner = self.inner.lock();
        let mut written = 0;
        let mut played = 0;
        while written < frames {
            if inner.playing.consumed || inner.playing.frames == 0 {
                let Some(next) = inner.queue.pop_front() else {
                    let last = inner.last_frame;
                    for frame in
                        output[written * channels..frames * channels].chunks_exact_mut(channels)
                    {
                        frame.copy_from_slice(&last[..channels]);
                    }
                    written = frames;
                    continue;
                };
                inner.playing = next;
                self.queued_buffers.fetch_sub(1, Ordering::AcqRel);
                self.released.notify_one();
            }
            let available = ((inner.playing.frames - inner.playing.frames_played) as usize)
                .min(frames - written);
            inner
                .ring
                .pop_into(&mut output[written * channels..(written + available) * channels]);
            written += available;
            played += available as u64;
            inner.playing.frames_played += available as u64;
            if inner.playing.frames_played >= inner.playing.frames {
                inner.playing.consumed = true;
            }
        }
        inner.last_frame[..channels]
            .copy_from_slice(&output[(frames - 1) * channels..frames * channels]);
        inner.min_played = inner.max_played;
        inner.max_played += played;
        inner.last_update = Instant::now();
    }

    /// The capture callback of an input stream: `frames` frames of silence arrived. Input
    /// buffers are released by time alone, so this only advances the play position.
    pub(crate) fn process_input(&self, frames: usize) {
        if self.silent() {
            return;
        }
        let mut inner = self.inner.lock();
        inner.min_played = inner.max_played;
        inner.max_played += frames as u64;
        inner.last_update = Instant::now();
    }
}

/// A stream acquired from a [`Sink`](super::Sink). The emulator appends buffers of interleaved
/// `i16` samples with [`append_buffer`](Self::append_buffer); the host callback (or the null
/// sink's real-time clock) plays them in order. Dropping the stream closes it.
pub struct SinkStream {
    pub(crate) state: Arc<StreamState>,
    pub(crate) host: Option<HostStream>,
}

impl SinkStream {
    /// What the stream carries.
    pub fn stream_type(&self) -> StreamType {
        self.state.ty
    }

    /// The stream's name in logs.
    pub fn name(&self) -> &str {
        &self.state.name
    }

    /// Channels of the samples the emulator appends (1 to 6).
    pub fn system_channels(&self) -> usize {
        self.state.system_channels
    }

    /// Channels of the host stream (2 or 6).
    pub fn device_channels(&self) -> usize {
        self.state.device_channels
    }

    /// Whether a host device plays the stream (otherwise the null sink's clock consumes it).
    pub fn is_host_stream(&self) -> bool {
        self.host.is_some() && !self.state.host_failed.load(Ordering::Acquire)
    }

    /// Starts (or resumes) playback. A host stream the device refuses to start moves to the
    /// null sink's clock for good, so its buffers still play out and get released.
    pub fn start(&self) {
        self.state.paused.store(false, Ordering::Release);
        let Some(host) = &self.host else {
            return;
        };
        if self.state.host_failed.load(Ordering::Acquire) {
            return;
        }
        if let Err(err) = cpal::traits::StreamTrait::play(&host.stream) {
            tracing::warn!(stream = %self.state.name, "starting the host stream, using the null clock: {err}");
            self.state.host_failed.store(true, Ordering::Release);
            host.fallback.add(&self.state);
        }
    }

    /// Pauses playback; queued buffers stay queued.
    pub fn stop(&self) {
        if let Some(host) = &self.host
            && !self.state.host_failed.load(Ordering::Acquire)
            && let Err(err) = cpal::traits::StreamTrait::pause(&host.stream)
        {
            tracing::warn!(stream = %self.state.name, "pausing the host stream: {err}");
        }
        let _inner = self.state.inner.lock();
        self.state.paused.store(true, Ordering::Release);
        self.state.released.notify_all();
    }

    /// Whether the stream is stopped (it starts stopped).
    pub fn is_paused(&self) -> bool {
        self.state.is_paused()
    }

    /// The volume of the guest system using this stream (`SetAudioOutVolume`).
    pub fn system_volume(&self) -> f32 {
        load_f32(&self.state.system_volume)
    }

    /// Sets the system volume, applied to samples appended from now on.
    pub fn set_system_volume(&self, volume: f32) {
        store_f32(&self.state.system_volume, volume);
    }

    /// Buffers queued and not yet started.
    pub fn queue_size(&self) -> u32 {
        self.state.queued_buffers.load(Ordering::Acquire)
    }

    /// How many buffers [`wait_free_space`](Self::wait_free_space) lets the producer queue.
    pub fn set_ring_size(&self, buffers: u32) {
        self.state.max_queue_size.store(buffers, Ordering::Release);
    }

    /// Whether samples are still waiting to play, the rest of the playing buffer included.
    pub fn has_pending_audio(&self) -> bool {
        self.state.inner.lock().ring.len != 0
    }

    /// Queues `frames` frames of `samples` (interleaved, `system_channels` per frame) to play
    /// after the buffers already queued, and returns the stream position (in frames) at which
    /// the buffer will have finished playing. Only frames `samples` holds are queued, and a
    /// buffer too large for the stream's ring is cut short. The samples are scaled by the stream, device and
    /// host volumes and converted to the device's channel count: 6 to 2 channels downmixes as
    /// Eden does, fewer channels than the device fill the first ones. Allocates nothing.
    pub fn append_buffer(&self, frames: u64, samples: &[i16]) -> u64 {
        let state = &self.state;
        let mut inner = state.inner.lock();
        if state.ty == StreamType::In {
            inner.appended_frames += frames;
            return inner.appended_frames;
        }
        let volume = load_f32(&state.system_volume)
            * load_f32(&state.sink.device_volume)
            * load_f32(&state.sink.host_volume);
        let (system, device) = (state.system_channels, state.device_channels);
        let wanted = (frames as usize).min(samples.len() / system);
        let fits = inner.ring.reserve(wanted * device) / device;
        if fits < wanted && !inner.overflow_logged {
            inner.overflow_logged = true;
            tracing::warn!(stream = %state.name, "sample ring full, cutting a buffer short");
        }
        let frames = wanted.min(fits);
        let samples = &samples[..frames * system];
        if system > device {
            const MIX: [f32; 4] = [1.0, 0.596, 0.354, 0.707];
            for frame in samples.chunks_exact(system) {
                let mut c = [0.0f32; MAX_CHANNELS];
                for (to, &from) in c.iter_mut().zip(frame) {
                    *to = f32::from(from);
                }
                let [fl, fr, center, lfe, bl, br] = c;
                let left = fl * MIX[0] + center * MIX[1] + lfe * MIX[2] + bl * MIX[3];
                let right = fr * MIX[0] + center * MIX[1] + lfe * MIX[2] + br * MIX[3];
                inner.ring.push(scale(left, volume));
                inner.ring.push(scale(right, volume));
            }
        } else {
            for frame in samples.chunks_exact(system) {
                for &sample in frame {
                    inner.ring.push(scale(f32::from(sample), volume));
                }
                for _ in system..device {
                    inner.ring.push(0);
                }
            }
        }
        let frames = frames as u64;
        if inner.queue.len() == QUEUE_CAPACITY {
            tracing::warn!(stream = %state.name, "buffer queue full, dropping the oldest buffer");
            inner.queue.pop_front();
            state.queued_buffers.fetch_sub(1, Ordering::AcqRel);
        }
        inner.queue.push_back(SinkBuffer {
            frames,
            frames_played: 0,
            consumed: false,
        });
        state.queued_buffers.fetch_add(1, Ordering::AcqRel);
        inner.appended_frames += frames;
        inner.appended_frames
    }

    /// Drops every queued sample. The dropped frames count as played, so positions returned by
    /// [`append_buffer`](Self::append_buffer) stay comparable with the play position.
    pub fn clear_queue(&self) {
        let mut inner = self.state.inner.lock();
        inner.ring.clear();
        inner.queue.clear();
        inner.playing = SinkBuffer {
            consumed: true,
            ..SinkBuffer::default()
        };
        inner.min_played = inner.appended_frames;
        inner.max_played = inner.appended_frames;
        self.state.queued_buffers.store(0, Ordering::Release);
        self.state.released.notify_all();
    }

    /// The stream position (in frames) the guest should see: the last callback's position plus
    /// the time since, capped at what the host has taken, plus a 25 ms lead.
    pub fn expected_played_sample_count(&self) -> u64 {
        let inner = self.state.inner.lock();
        let elapsed = inner.last_update.elapsed().as_nanos() as u64;
        let expected = inner.min_played + u64::from(SAMPLE_RATE) * elapsed / 1_000_000_000;
        expected.min(inner.max_played) + PLAYED_LEAD
    }

    /// Blocks the producer (the audio renderer) while the queue is full: up to 7 ms, and longer
    /// only while it is more than three buffers over its ring size. Returns at once when the
    /// stream or the sink is paused.
    pub fn wait_free_space(&self) {
        let state = &self.state;
        let max = state.max_queue_size.load(Ordering::Acquire);
        let queued = || state.queued_buffers.load(Ordering::Acquire);
        let mut inner = state.inner.lock();
        if state.silent() || queued() < max {
            return;
        }
        state
            .released
            .wait_for(&mut inner, Duration::from_millis(7));
        while !state.silent() && queued() > max + 3 {
            state
                .released
                .wait_for(&mut inner, Duration::from_millis(7));
        }
    }
}

fn scale(sample: f32, volume: f32) -> i16 {
    ((sample * volume) as i32).clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(system: usize, device: usize) -> SinkStream {
        let shared = Arc::new(SinkShared::new(1.0));
        let state = Arc::new(StreamState::new(
            StreamType::Out,
            "test",
            system,
            device,
            shared,
        ));
        let stream = SinkStream { state, host: None };
        stream.start();
        stream
    }

    #[test]
    fn plays_buffers_in_order_and_repeats_the_last_frame_on_underrun() {
        let stream = stream(2, 2);
        assert_eq!(stream.append_buffer(2, &[1, 2, 3, 4]), 2);
        assert_eq!(stream.append_buffer(1, &[5, 6]), 3);
        assert_eq!(stream.queue_size(), 2);
        let mut out = [9i16; 8];
        stream.state.process_output(&mut out);
        assert_eq!(out, [1, 2, 3, 4, 5, 6, 0, 0]);
        assert_eq!(stream.queue_size(), 0);
        stream.state.process_output(&mut out);
        assert_eq!(out, [0, 0, 0, 0, 0, 0, 0, 0]);
        stream.append_buffer(1, &[7, 8]);
        stream.state.process_output(&mut out[..2]);
        stream.state.process_output(&mut out);
        assert_eq!(out, [7, 8, 7, 8, 7, 8, 7, 8]);
        assert!(!stream.has_pending_audio());
        assert_eq!(stream.expected_played_sample_count(), 4 + PLAYED_LEAD);
    }

    #[test]
    fn applies_volume_with_clamping() {
        let stream = stream(2, 2);
        stream.set_system_volume(2.0);
        stream.append_buffer(2, &[20000, -20000, 100, -100]);
        let mut out = [0i16; 4];
        stream.state.process_output(&mut out);
        assert_eq!(out, [i16::MAX, i16::MIN, 200, -200]);
    }

    #[test]
    fn downmixes_six_channels_like_eden() {
        let stream = stream(6, 2);
        stream.append_buffer(1, &[1000, 2000, 3000, 4000, 5000, 6000]);
        let mut out = [0i16; 2];
        stream.state.process_output(&mut out);
        let left = 1000.0f32 + 3000.0 * 0.596 + 4000.0 * 0.354 + 5000.0 * 0.707;
        let right = 2000.0f32 + 3000.0 * 0.596 + 4000.0 * 0.354 + 6000.0 * 0.707;
        assert_eq!(out, [left as i16, right as i16]);
    }

    #[test]
    fn upmixes_stereo_into_the_front_channels() {
        let stream = stream(2, 6);
        stream.append_buffer(1, &[7, 8]);
        let mut out = [9i16; 6];
        stream.state.process_output(&mut out);
        assert_eq!(out, [7, 8, 0, 0, 0, 0]);
    }

    #[test]
    fn paused_streams_output_silence_and_keep_their_queue() {
        let stream = stream(2, 2);
        stream.append_buffer(1, &[3, 4]);
        stream.stop();
        let mut out = [1i16; 2];
        stream.state.process_output(&mut out);
        assert_eq!(out, [0, 0]);
        assert_eq!(stream.queue_size(), 1);
        stream.start();
        stream.state.process_output(&mut out);
        assert_eq!(out, [3, 4]);
    }

    #[test]
    fn clearing_the_queue_counts_dropped_frames_as_played() {
        let stream = stream(2, 2);
        let end = stream.append_buffer(100, &[0; 200]);
        stream.clear_queue();
        assert_eq!(stream.queue_size(), 0);
        assert!(stream.expected_played_sample_count() >= end);
    }

    #[test]
    fn wait_free_space_returns_once_the_queue_drains() {
        let stream = stream(2, 2);
        stream.set_ring_size(1);
        stream.append_buffer(1, &[0, 0]);
        let started = Instant::now();
        stream.wait_free_space();
        assert!(started.elapsed() >= Duration::from_millis(3));
        stream.state.process_output(&mut [0; 2]);
        let started = Instant::now();
        stream.wait_free_space();
        assert!(started.elapsed() < Duration::from_millis(5));
    }

    #[test]
    fn ring_wraps_and_pads_missing_samples() {
        let mut ring = SampleRing::new();
        ring.read = RING_CAPACITY - 1;
        ring.push(1);
        ring.push(2);
        let mut out = [9; 3];
        ring.pop_into(&mut out);
        assert_eq!(out, [1, 2, 0]);
        assert_eq!(ring.len, 0);
    }

    #[test]
    fn buffers_larger_than_the_ring_play_in_full() {
        let stream = stream(6, 6);
        let frames = 48_000u64;
        let samples: Vec<i16> = (0..frames as usize * 6)
            .map(|i| (i % 6) as i16 + 1)
            .collect();
        assert_eq!(stream.append_buffer(frames, &samples), frames);
        assert_eq!(stream.append_buffer(frames, &samples), 2 * frames);
        let mut out = vec![0i16; 1024 * 6];
        for _ in 0..(2 * frames as usize).div_ceil(1024) - 1 {
            stream.state.process_output(&mut out);
            assert!(
                out.as_chunks::<6>()
                    .0
                    .iter()
                    .all(|f| *f == [1, 2, 3, 4, 5, 6])
            );
        }
        assert!(stream.has_pending_audio());
        stream.state.process_output(&mut out);
        assert!(!stream.has_pending_audio());
        assert_eq!(stream.queue_size(), 0);
    }

    #[test]
    fn buffers_past_the_ring_limit_queue_only_the_frames_kept() {
        let stream = stream(2, 2);
        let frames = MAX_RING_CAPACITY as u64 / 2 + 100;
        let end = stream.append_buffer(frames, &vec![1; frames as usize * 2]);
        assert_eq!(end, MAX_RING_CAPACITY as u64 / 2);
        assert_eq!(stream.append_buffer(10, &[1; 20]), end);
    }
}
