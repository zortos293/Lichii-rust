//! What audio out and audio in share: the guest's ring of buffers and the device session that
//! plays them on a sink stream.

use std::sync::Arc;

use lichii_common::{HleResult, VAddr};
use lichii_memory::AddressSpace;

use super::result;
use super::{Sink, SinkStream, StreamType};

/// Buffers a session can hold: appended, registered with the sink and released together.
pub const BUFFER_COUNT: usize = 32;
/// Buffers registered with the sink at once.
pub const BUFFER_APPEND_LIMIT: usize = 4;
/// The largest guest buffer a session reads, in samples (8 MiB).
const MAX_BUFFER_SAMPLES: usize = 4 << 20;

/// A guest buffer of PCM16 samples.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AudioBuffer {
    /// Guest address of the samples.
    pub samples: VAddr,
    /// Size of the samples in bytes.
    pub size: u64,
    /// The guest's identifier (the address of its `AudioOutBuffer`).
    pub tag: u64,
    /// Stream position at which the buffer has played; set when it is registered.
    end: u64,
}

impl AudioBuffer {
    pub fn new(samples: VAddr, size: u64, tag: u64) -> Self {
        Self {
            samples,
            size,
            tag,
            end: 0,
        }
    }
}

/// The ring of a session's buffers, oldest first: released (played, waiting for the guest to
/// collect them), registered (queued on the sink) and appended (waiting to be registered).
pub struct AudioBuffers {
    buffers: [AudioBuffer; BUFFER_COUNT],
    start: usize,
    released: usize,
    registered: usize,
    appended: usize,
}

impl Default for AudioBuffers {
    fn default() -> Self {
        Self {
            buffers: [AudioBuffer::default(); BUFFER_COUNT],
            start: 0,
            released: 0,
            registered: 0,
            appended: 0,
        }
    }
}

impl AudioBuffers {
    fn slot(&self, index: usize) -> usize {
        (self.start + index) % BUFFER_COUNT
    }

    /// Buffers in the ring, of every section.
    pub fn total(&self) -> usize {
        self.released + self.registered + self.appended
    }

    /// Appended and registered buffers (`GetAudioOutBufferCount`).
    pub fn active(&self) -> usize {
        self.registered + self.appended
    }

    pub fn registered(&self) -> usize {
        self.registered
    }

    /// Adds a buffer at the end of the ring; `false` when the ring is full.
    pub fn append(&mut self, buffer: AudioBuffer) -> bool {
        if self.total() == BUFFER_COUNT {
            return false;
        }
        let slot = self.slot(self.total());
        self.buffers[slot] = buffer;
        self.appended += 1;
        true
    }

    /// Moves appended buffers to registered until `BUFFER_APPEND_LIMIT` are registered, passing
    /// each to `register` in order. Returns how many moved.
    pub fn register(&mut self, mut register: impl FnMut(&mut AudioBuffer)) -> usize {
        let count = self
            .appended
            .min(BUFFER_APPEND_LIMIT.saturating_sub(self.registered));
        for index in 0..count {
            let slot = self.slot(self.released + self.registered + index);
            register(&mut self.buffers[slot]);
        }
        self.registered += count;
        self.appended -= count;
        count
    }

    /// Releases registered buffers, oldest first, while `consumed` says they have played (all
    /// of them with `force`), passing each to `release`. Returns whether the guest should be
    /// signalled: a buffer was released or none is registered any more.
    pub fn release(
        &mut self,
        force: bool,
        consumed: impl Fn(&AudioBuffer) -> bool,
        mut release: impl FnMut(&AudioBuffer),
    ) -> bool {
        let mut released = false;
        while self.registered > 0 {
            let buffer = &self.buffers[self.slot(self.released)];
            if !force && !consumed(buffer) {
                break;
            }
            release(buffer);
            self.registered -= 1;
            self.released += 1;
            released = true;
        }
        released || self.registered == 0
    }

    /// Hands released buffers back to the guest: writes their tags to `tags` and returns how
    /// many it took. As on Eden, a released buffer with tag 0 ends the list, and an empty
    /// `tags` still takes one buffer and reports it.
    pub fn take_released(&mut self, tags: &mut [u64]) -> u32 {
        let mut count = 0;
        while self.released > 0 {
            let slot = self.slot(0);
            let tag = self.buffers[slot].tag;
            self.buffers[slot] = AudioBuffer::default();
            self.start = (self.start + 1) % BUFFER_COUNT;
            self.released -= 1;
            if tag == 0 {
                break;
            }
            if let Some(out) = tags.get_mut(count) {
                *out = tag;
            }
            count += 1;
            if count >= tags.len() {
                break;
            }
        }
        count as u32
    }

    /// Releases every registered and appended buffer without playing it. Returns how many.
    pub fn flush(&mut self) -> usize {
        let flushed = self.registered + self.appended;
        self.released += flushed;
        self.registered = 0;
        self.appended = 0;
        flushed
    }

    /// Whether a buffer with `tag` is in the ring.
    pub fn contains(&self, tag: u64) -> bool {
        (0..self.total()).any(|index| self.buffers[self.slot(index)].tag == tag)
    }
}

/// A session's link to the sink: one stream, opened on the first start and kept until the
/// session closes, and the guest memory its buffers live in.
pub struct DeviceSession {
    name: String,
    ty: StreamType,
    channels: usize,
    memory: Arc<AddressSpace>,
    stream: Option<SinkStream>,
    samples: Vec<i16>,
}

impl DeviceSession {
    pub fn new(name: String, ty: StreamType, channels: usize, memory: Arc<AddressSpace>) -> Self {
        Self {
            name,
            ty,
            channels,
            memory,
            stream: None,
            samples: Vec::new(),
        }
    }

    pub fn memory(&self) -> &Arc<AddressSpace> {
        &self.memory
    }

    pub fn stream(&self) -> Option<&SinkStream> {
        self.stream.as_ref()
    }

    /// Opens the stream if this is the first start, then starts it.
    pub fn start(&mut self, sink: &Sink) {
        self.stream
            .get_or_insert_with(|| sink.open_stream(self.channels, &self.name, self.ty))
            .start();
    }

    pub fn stop(&self) {
        if let Some(stream) = &self.stream {
            stream.stop();
        }
    }

    pub fn clear_buffers(&self) {
        if let Some(stream) = &self.stream {
            stream.clear_queue();
        }
    }

    pub fn set_volume(&self, volume: f32) {
        if let Some(stream) = &self.stream {
            stream.set_system_volume(volume);
        }
    }

    pub fn set_ring_size(&self, buffers: usize) {
        if let Some(stream) = &self.stream {
            stream.set_ring_size(buffers as u32);
        }
    }

    /// Queues `buffer` on the stream: reads its samples from guest memory (output sessions)
    /// and records the stream position at which it will have played.
    pub fn append(&mut self, buffer: &mut AudioBuffer) -> HleResult<()> {
        let stream = self.stream.as_ref().ok_or(result::OPERATION_FAILED)?;
        let count = ((buffer.size / 2) as usize).min(MAX_BUFFER_SAMPLES);
        let frames = (count / self.channels) as u64;
        if self.ty == StreamType::In {
            buffer.end = stream.append_buffer(frames, &[]);
            return Ok(());
        }
        if self.samples.len() < count {
            self.samples.resize(count, 0);
        }
        let samples = &mut self.samples[..count];
        if let Err(err) = self
            .memory
            .read_bytes(buffer.samples, bytemuck::cast_slice_mut(samples))
        {
            tracing::warn!(session = %self.name, "reading an audio buffer: {err:?}");
            samples.fill(0);
        }
        buffer.end = stream.append_buffer(frames, samples);
        Ok(())
    }

    /// Hands a played buffer back: capture sessions fill it with what they heard (silence).
    pub fn release(&self, buffer: &AudioBuffer) {
        if self.ty == StreamType::In
            && let Err(err) = self.memory.fill(buffer.samples, buffer.size, 0)
        {
            tracing::warn!(session = %self.name, "writing an audio in buffer: {err:?}");
        }
    }

    /// Whether the stream has played `buffer`.
    pub fn is_consumed(&self, buffer: &AudioBuffer) -> bool {
        self.played_sample_count() >= buffer.end
    }

    /// The stream position the guest sees (`GetAudioOutPlayedSampleCount`), 0 before the first
    /// start.
    pub fn played_sample_count(&self) -> u64 {
        self.stream
            .as_ref()
            .map_or(0, SinkStream::expected_played_sample_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buffer(tag: u64) -> AudioBuffer {
        AudioBuffer::new(0x1000 * tag, 0x100, tag)
    }

    #[test]
    fn registers_at_most_four_buffers() {
        let mut ring = AudioBuffers::default();
        for tag in 1..=6 {
            assert!(ring.append(buffer(tag)));
        }
        let mut seen = Vec::new();
        assert_eq!(ring.register(|b| seen.push(b.tag)), 4);
        assert_eq!(seen, [1, 2, 3, 4]);
        assert_eq!(ring.register(|_| {}), 0);
        assert_eq!(ring.active(), 6);
    }

    #[test]
    fn releases_in_order_and_hands_back_tags() {
        let mut ring = AudioBuffers::default();
        for tag in 1..=3 {
            ring.append(buffer(tag));
        }
        ring.register(|_| {});
        assert!(ring.release(false, |b| b.tag == 1, |_| {}));
        assert!(!ring.release(false, |_| false, |_| {}));
        let mut tags = [0; 4];
        assert_eq!(ring.take_released(&mut tags), 1);
        assert_eq!(tags[0], 1);
        assert!(ring.release(true, |_| false, |_| {}));
        assert_eq!(ring.take_released(&mut tags), 2);
        assert_eq!(tags[..2], [2, 3]);
        assert_eq!(ring.total(), 0);
    }

    #[test]
    fn an_empty_tag_list_still_takes_one_buffer() {
        let mut ring = AudioBuffers::default();
        ring.append(buffer(1));
        ring.append(buffer(2));
        ring.flush();
        assert_eq!(ring.take_released(&mut []), 1);
        assert_eq!(ring.total(), 1);
    }

    #[test]
    fn holds_at_most_buffer_count_and_wraps() {
        let mut ring = AudioBuffers::default();
        for tag in 1..=BUFFER_COUNT as u64 {
            assert!(ring.append(buffer(tag)));
        }
        assert!(!ring.append(buffer(99)));
        assert!(ring.contains(BUFFER_COUNT as u64));
        assert_eq!(ring.flush(), BUFFER_COUNT);
        let mut tags = [0; BUFFER_COUNT];
        assert_eq!(ring.take_released(&mut tags[..3]), 3);
        assert!(ring.append(buffer(100)));
        assert!(ring.contains(100) && !ring.contains(1));
    }
}
