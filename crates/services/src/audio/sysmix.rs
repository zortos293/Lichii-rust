//! The Switch 2 sound mixer streams (`nx2_sysmix.cpp`): PCM rings in guest memory that
//! `audsmx:u` creates and `auddmg:u` attaches. A pump thread moves one 240-frame block per stream
//! every 5 ms between the ring and a host sink stream, and signals the stream's data event when
//! the free output space (or the captured input) reaches the guest's threshold.
//!
//! Ring layout, from the start of the guest buffer: the read cursor at 0x44, the write cursor at
//! 0x1004 and the frames at 0x1040, in `frames + 1` slots. The guest owns one cursor (the write
//! cursor of an output stream, the read cursor of an input stream) and the service the other.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use lichii_audio::renderer::SampleFormat;
use lichii_audio::renderer::result::{
    INSUFFICIENT_BUFFER, INVALID_ADDRESS_INFO, INVALID_ARGUMENT, INVALID_HANDLE, NOT_FOUND,
    NOT_SUPPORTED, OPERATION_FAILED,
};
use lichii_audio::sink::{SAMPLE_RATE, Sink, SinkStream, StreamType, TARGET_SAMPLE_COUNT};
use lichii_common::{HleResult, PAGE_SIZE};
use lichii_kernel::Event;
use lichii_kernel::process::Process;
use lichii_memory::{AddressSpace, MemoryPermission};
use parking_lot::Mutex;

const READ_OFFSET: u64 = 0x44;
const WRITE_OFFSET: u64 = 0x1004;
const PAYLOAD_OFFSET: u64 = 0x1040;
const BLOCK_FRAMES: u32 = TARGET_SAMPLE_COUNT;
const PUMP_INTERVAL: Duration = Duration::from_millis(5);
/// Output blocks the host stream may queue before a pump backs off.
const HOST_QUEUE_LIMIT: u32 = 4;
/// Ticks between status logs (2 s).
const STATUS_TICKS: u64 = 400;
/// Attempts `PurgeStream` makes while the guest moves its read cursor.
const PURGE_ATTEMPTS: u32 = 8;

/// Device functionality of an output (playback) stream.
pub(super) const FUNCTIONALITY_OUTPUT: u32 = 1;
/// Device functionality of an input (capture) stream.
pub(super) const FUNCTIONALITY_INPUT: u32 = 2;

/// What a stream has moved since it opened.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counters {
    pub ticks: u64,
    pub copied_frames: u64,
    pub accepted_frames: u64,
    pub retried_frames: u64,
    pub padded_frames: u64,
    pub nonzero_frames: u64,
    pub admitted_blocks: u64,
    pub backpressure: u64,
    pub deferred: u64,
    pub faults: u64,
}

/// Whether an output stream can play `sample_type` at `channels` and `rate`: 48 kHz stereo,
/// S16 or F32.
pub(super) fn is_format_supported(sample_type: u32, channels: u32, rate: u32) -> bool {
    (sample_type == SampleFormat::PcmInt16 as u32 || sample_type == SampleFormat::PcmFloat as u32)
        && channels == 2
        && rate == SAMPLE_RATE
}

/// Whether an input stream can capture the format: like output, with one or two channels.
pub(super) fn is_input_format_supported(sample_type: u32, channels: u32, rate: u32) -> bool {
    (channels == 1 || channels == 2) && is_format_supported(sample_type, 2, rate)
}

/// Checks the creation flags of a direction and that the ring is page aligned, at least two pages
/// and mapped in `memory`.
pub(super) fn validate_region(
    memory: &AddressSpace,
    buffer: u64,
    size: u64,
    creation_flags: u32,
    input: bool,
) -> HleResult<()> {
    if creation_flags != if input { 2 } else { 0x24 } {
        return Err(NOT_SUPPORTED);
    }
    let page = PAGE_SIZE;
    let valid = buffer != 0
        && buffer.is_multiple_of(page)
        && size.is_multiple_of(page)
        && size >= 2 * page
        && buffer.checked_add(size).is_some()
        && (buffer..buffer + size)
            .step_by(PAGE_SIZE as usize)
            .all(|address| memory.is_mapped(address));
    if !valid {
        return Err(INVALID_ADDRESS_INFO);
    }
    Ok(())
}

/// Every sysmix stream of the boot, by id, and the thread that pumps them.
pub(crate) struct SysmixStreams {
    sink: Arc<Sink>,
    streams: Mutex<BTreeMap<u32, Arc<SysmixStream>>>,
    next_id: AtomicU32,
    pump: Mutex<bool>,
}

impl SysmixStreams {
    pub fn new(sink: Arc<Sink>) -> Arc<Self> {
        Arc::new(Self {
            sink,
            streams: Mutex::new(BTreeMap::new()),
            next_id: AtomicU32::new(1),
            pump: Mutex::new(false),
        })
    }

    /// Creates a stream on the ring at `buffer`, owned by `process`, and starts the pump.
    pub fn create(
        self: &Arc<Self>,
        process: &Arc<Process>,
        buffer: u64,
        size: u64,
        creation_flags: u32,
        input: bool,
    ) -> Arc<SysmixStream> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let stream = Arc::new(SysmixStream {
            id,
            input,
            process: process.clone(),
            memory: process.memory().clone(),
            buffer,
            size,
            creation_flags,
            sink: self.sink.clone(),
            state_event: Arc::new(Event::new()),
            data_event: Arc::new(Event::new()),
            inner: Mutex::new(Inner::default()),
        });
        self.streams.lock().insert(id, stream.clone());
        self.start_pump();
        stream
    }

    #[cfg(test)]
    pub fn get(&self, id: u32) -> Option<Arc<SysmixStream>> {
        self.streams.lock().get(&id).cloned()
    }

    /// Forgets stream `id` and closes it (its session closed).
    pub fn remove(&self, id: u32) {
        let stream = self.streams.lock().remove(&id);
        if let Some(stream) = stream {
            stream.close();
        }
    }

    fn owned(&self, id: u32, process: &Arc<Process>) -> HleResult<Arc<SysmixStream>> {
        let stream = self.streams.lock().get(&id).cloned().ok_or(NOT_FOUND)?;
        if !Arc::ptr_eq(&stream.process, process) {
            return Err(INVALID_HANDLE);
        }
        Ok(stream)
    }

    /// Engages or disengages stream `id` of `process`.
    pub fn set_engaged(&self, id: u32, engaged: bool, process: &Arc<Process>) -> HleResult<()> {
        self.owned(id, process)?.set_engaged(engaged)
    }

    /// `AttachStream`: engages stream `id` when its direction matches `functionality`.
    pub fn attach(&self, id: u32, functionality: u32, process: &Arc<Process>) -> HleResult<()> {
        let stream = self.owned(id, process)?;
        let direction = if stream.input {
            FUNCTIONALITY_INPUT
        } else {
            FUNCTIONALITY_OUTPUT
        };
        if functionality != direction {
            return Err(NOT_SUPPORTED);
        }
        stream.set_engaged(true)
    }

    /// One pump tick for every stream. Skipped while emulation is paused (the sink is) or another
    /// thread holds the stream list.
    pub fn pump(&self) {
        if self.sink.is_paused() {
            return;
        }
        if let Some(streams) = self.streams.try_lock() {
            streams.values().for_each(|stream| stream.pump());
        }
    }

    fn start_pump(self: &Arc<Self>) {
        let mut started = self.pump.lock();
        if *started {
            return;
        }
        let streams = Arc::downgrade(self);
        match std::thread::Builder::new()
            .name("sysmix".into())
            .spawn(move || pump_loop(&streams))
        {
            Ok(_) => *started = true,
            Err(err) => tracing::error!("starting the sysmix pump: {err}"),
        }
    }
}

/// Pumps every 5 ms until the streams are dropped, skipping ticks it fell behind on.
fn pump_loop(streams: &Weak<SysmixStreams>) {
    let mut next = Instant::now() + PUMP_INTERVAL;
    loop {
        std::thread::sleep(next.saturating_duration_since(Instant::now()));
        let Some(streams) = streams.upgrade() else {
            return;
        };
        streams.pump();
        next += PUMP_INTERVAL;
        let now = Instant::now();
        if next < now {
            next = now + PUMP_INTERVAL;
        }
    }
}

#[derive(Default)]
struct Inner {
    host: Option<SinkStream>,
    float: bool,
    frame_bytes: u32,
    slots: u32,
    consumer: u32,
    data_mode: u32,
    threshold: u32,
    gain: f32,
    open: bool,
    engaged: bool,
    faulted: bool,
    block: Vec<u8>,
    output: Vec<i16>,
    counters: Counters,
}

/// The ring's cursors: `(read, write, filled)`.
type Occupancy = (u32, u32, u32);

/// One sysmix stream.
pub(crate) struct SysmixStream {
    id: u32,
    input: bool,
    process: Arc<Process>,
    memory: Arc<AddressSpace>,
    buffer: u64,
    size: u64,
    creation_flags: u32,
    sink: Arc<Sink>,
    state_event: Arc<Event>,
    data_event: Arc<Event>,
    inner: Mutex<Inner>,
}

impl SysmixStream {
    pub fn id(&self) -> u32 {
        self.id
    }

    pub fn state_event(&self) -> &Arc<Event> {
        &self.state_event
    }

    pub fn data_event(&self) -> &Arc<Event> {
        &self.data_event
    }

    /// Whether a started host stream is playing the ring.
    #[cfg(test)]
    pub fn is_playing(&self) -> bool {
        self.inner
            .lock()
            .host
            .as_ref()
            .is_some_and(|host| !host.is_paused())
    }

    #[cfg(test)]
    pub fn counters(&self) -> Counters {
        self.inner.lock().counters
    }

    /// `OpenStream`: checks the format and ring layout and opens the host stream, disengaged.
    pub fn open(
        &self,
        sample_type: u32,
        channels: u32,
        rate: u32,
        frames: u32,
        prefill: u32,
        flags: u32,
    ) -> HleResult<()> {
        let mut p = self.inner.lock();
        if p.open {
            return Err(OPERATION_FAILED);
        }
        let supported = if self.input {
            is_input_format_supported(sample_type, channels, rate)
        } else {
            is_format_supported(sample_type, channels, rate)
        };
        if !supported || prefill != 0 || flags != 0 {
            return Err(NOT_SUPPORTED);
        }
        if frames == 0 || frames >= i32::MAX as u32 {
            return Err(INVALID_ARGUMENT);
        }
        validate_region(
            &self.memory,
            self.buffer,
            self.size,
            self.creation_flags,
            self.input,
        )?;
        let slots = u64::from(frames) + 1;
        let float = sample_type == SampleFormat::PcmFloat as u32;
        let frame_bytes = channels * if float { 4 } else { 2 };
        let timestamp_bytes = if self.input { 0 } else { slots.min(16) * 8 };
        let minimum = (0x1000
            + (0x40 + timestamp_bytes).next_multiple_of(64)
            + (slots * u64::from(frame_bytes)).next_multiple_of(64))
        .next_multiple_of(0x1000);
        if self.size < minimum || PAYLOAD_OFFSET + slots * u64::from(frame_bytes) > self.size {
            return Err(INSUFFICIENT_BUFFER);
        }
        let (Some(read), Some(write)) = (self.cursor(READ_OFFSET), self.cursor(WRITE_OFFSET))
        else {
            return Err(INVALID_ADDRESS_INFO);
        };
        if u64::from(read) >= slots || u64::from(write) >= slots {
            return Err(INVALID_ADDRESS_INFO);
        }
        let ty = if self.input {
            StreamType::In
        } else {
            StreamType::Out
        };
        let host = self
            .sink
            .open_stream(channels as usize, &format!("sysmix-{}", self.id), ty);
        let slots = slots as u32;
        *p = Inner {
            host: Some(host),
            float,
            frame_bytes,
            slots,
            consumer: if self.input { write } else { read },
            gain: 1.0,
            open: true,
            block: vec![0; BLOCK_FRAMES as usize * 2 * 4],
            output: vec![0; BLOCK_FRAMES as usize * 2],
            ..Inner::default()
        };
        self.state_event.signal();
        self.notify(
            &p,
            (u64::from(write) + u64::from(slots) - u64::from(read)) as u32 % slots,
        );
        tracing::info!(
            stream = self.id,
            input = self.input,
            sample_type,
            channels,
            slots,
            "sysmix stream open"
        );
        Ok(())
    }

    /// `CloseStream`: stops admission and closes the host stream.
    pub fn close(&self) {
        let mut p = self.inner.lock();
        p.open = false;
        p.engaged = false;
        p.data_mode = 0;
        p.host = None;
        self.state_event.signal();
    }

    pub fn set_engaged(&self, engaged: bool) -> HleResult<()> {
        let mut p = self.inner.lock();
        if !p.open || p.faulted {
            return Err(OPERATION_FAILED);
        }
        let Some(host) = &p.host else {
            return Err(OPERATION_FAILED);
        };
        if p.engaged == engaged {
            return Ok(());
        }
        if engaged {
            host.start();
        } else {
            host.stop();
        }
        p.engaged = engaged;
        self.state_event.signal();
        Ok(())
    }

    /// `AcquireStreamDataEvent` (mode 2 with a threshold) and its release (mode 0).
    pub fn set_data_notification(&self, mode: u32, threshold: i32) -> HleResult<()> {
        let mut p = self.inner.lock();
        if mode != 0 && mode != 2 {
            return Err(NOT_SUPPORTED);
        }
        if !p.open || p.faulted || threshold < 0 || threshold as u32 >= p.slots {
            return Err(INVALID_ARGUMENT);
        }
        let Some((_, _, filled)) = self.occupancy(&p) else {
            return Err(self.fault(&mut p, INVALID_ADDRESS_INFO));
        };
        p.data_mode = mode;
        p.threshold = threshold as u32;
        self.notify(&p, filled);
        Ok(())
    }

    /// `SetStreamGain`: `gain` from 0 to 128, applied at once (ramps are not supported).
    pub fn set_gain(&self, gain: f32, ramp_ns: i64) -> HleResult<()> {
        if ramp_ns != 0 {
            return Err(NOT_SUPPORTED);
        }
        if !gain.is_finite() || !(0.0..=128.0).contains(&gain) {
            return Err(INVALID_ARGUMENT);
        }
        self.inner.lock().gain = gain;
        Ok(())
    }

    pub fn gain(&self) -> f32 {
        self.inner.lock().gain
    }

    /// 0 closed (or faulted), 1 open, 2 engaged.
    pub fn state(&self) -> u32 {
        let p = self.inner.lock();
        match (p.open && !p.faulted, p.engaged) {
            (false, _) => 0,
            (true, false) => 1,
            (true, true) => 2,
        }
    }

    /// `PurgeStream`: drops up to `frames` captured frames (all of them for 0) by moving the
    /// guest's read cursor, and returns how many it dropped.
    pub fn purge(&self, frames: u32) -> HleResult<u32> {
        let mut p = self.inner.lock();
        if !self.input || !p.open || p.faulted {
            return Err(OPERATION_FAILED);
        }
        for _ in 0..PURGE_ATTEMPTS {
            let Some((read, _, filled)) = self.occupancy(&p) else {
                return Err(self.fault(&mut p, INVALID_ADDRESS_INFO));
            };
            let count = if frames == 0 {
                filled
            } else {
                frames.min(filled)
            };
            match self.swap_cursor(READ_OFFSET, read, (read + count) % p.slots) {
                None => return Err(self.fault(&mut p, INVALID_ADDRESS_INFO)),
                Some(false) => continue,
                Some(true) => {}
            }
            let Some((_, _, filled)) = self.occupancy(&p) else {
                return Err(self.fault(&mut p, INVALID_ADDRESS_INFO));
            };
            self.notify(&p, filled);
            return Ok(count);
        }
        Err(OPERATION_FAILED)
    }

    /// One tick: moves at most one block between the ring and the host stream.
    pub fn pump(&self) {
        let Some(mut guard) = self.inner.try_lock() else {
            return;
        };
        let p = &mut *guard;
        if !p.open || p.faulted {
            return;
        }
        p.counters.ticks += 1;
        let Some((read, write, filled)) = self.occupancy(p) else {
            self.fault(p, INVALID_ADDRESS_INFO);
            return;
        };
        if !p.engaged {
            self.notify(p, filled);
            return;
        }
        if self.input {
            self.pump_input(p, write, filled);
        } else {
            self.pump_output(p, read, filled);
        }
    }

    /// Captures one block of silence (the host has no microphone) into the ring.
    fn pump_input(&self, p: &mut Inner, write: u32, filled: u32) {
        let taken = (p.slots - 1 - filled).min(BLOCK_FRAMES);
        if taken == 0 {
            p.counters.backpressure += 1;
            self.notify(p, filled);
            return;
        }
        let bytes = (taken * p.frame_bytes) as usize;
        p.block[..bytes].fill(0);
        let first = taken.min(p.slots - write);
        let split = (first * p.frame_bytes) as usize;
        let next = (write + taken) % p.slots;
        let written = self.write_payload(write, &p.block[..split], p.frame_bytes)
            && self.write_payload(0, &p.block[split..bytes], p.frame_bytes)
            && self.swap_cursor(WRITE_OFFSET, write, next) == Some(true);
        if !written {
            self.fault(p, INVALID_ADDRESS_INFO);
            return;
        }
        p.consumer = next;
        p.counters.copied_frames += u64::from(taken);
        p.counters.accepted_frames += u64::from(taken);
        p.counters.admitted_blocks += 1;
        match self.occupancy(p) {
            Some((_, _, filled)) => self.notify(p, filled),
            None => {
                self.fault(p, INVALID_ADDRESS_INFO);
            }
        }
    }

    /// Plays one block from the ring, padded with silence when the guest is short of a block and
    /// the host has nothing left to play.
    fn pump_output(&self, p: &mut Inner, read: u32, filled: u32) {
        let Some(host) = &p.host else {
            return;
        };
        let taken = filled.min(BLOCK_FRAMES);
        if taken < BLOCK_FRAMES && host.has_pending_audio() {
            p.counters.deferred += 1;
            self.notify(p, filled);
            return;
        }
        if host.queue_size() >= HOST_QUEUE_LIMIT {
            p.counters.backpressure += 1;
            p.counters.retried_frames += u64::from(taken);
            self.notify(p, filled);
            return;
        }
        let first = taken.min(p.slots - read);
        let split = (first * p.frame_bytes) as usize;
        let bytes = (taken * p.frame_bytes) as usize;
        let (head, tail) = p.block[..bytes].split_at_mut(split);
        if !(self.read_payload(read, head, p.frame_bytes)
            && self.read_payload(0, tail, p.frame_bytes))
        {
            self.fault(p, INVALID_ADDRESS_INFO);
            return;
        }
        p.counters.copied_frames += u64::from(taken);
        p.output.fill(0);
        let mut nonzero = 0;
        for frame in 0..taken as usize {
            let mut audible = false;
            for channel in 0..2 {
                let index = frame * 2 + channel;
                let value = if p.float {
                    f32::from_le_bytes(p.block[index * 4..index * 4 + 4].try_into().unwrap())
                } else {
                    f32::from(i16::from_le_bytes(
                        p.block[index * 2..index * 2 + 2].try_into().unwrap(),
                    )) / 32768.0
                };
                if value.is_finite() {
                    let sample = ((value * p.gain).clamp(-1.0, 1.0) * 32768.0) as i32;
                    p.output[index] = sample.clamp(-32768, 32767) as i16;
                    audible |= p.output[index] != 0;
                }
            }
            nonzero += u64::from(audible);
        }
        host.append_buffer(u64::from(BLOCK_FRAMES), &p.output);
        p.counters.admitted_blocks += 1;
        p.counters.padded_frames += u64::from(BLOCK_FRAMES - taken);
        let next = (read + taken) % p.slots;
        if self.swap_cursor(READ_OFFSET, read, next) != Some(true) {
            self.fault(p, INVALID_ADDRESS_INFO);
            return;
        }
        p.consumer = next;
        p.counters.accepted_frames += u64::from(taken);
        p.counters.nonzero_frames += nonzero;
        let Some((_, _, filled)) = self.occupancy(p) else {
            self.fault(p, INVALID_ADDRESS_INFO);
            return;
        };
        self.notify(p, filled);
        if p.counters.admitted_blocks <= 3 || p.counters.ticks.is_multiple_of(STATUS_TICKS) {
            self.log_status(p, filled);
        }
    }

    /// Logs what the stream moved, so a stall or a fast or slow guest clock shows in the log.
    fn log_status(&self, p: &Inner, filled: u32) {
        let c = &p.counters;
        tracing::debug!(
            stream = self.id,
            ticks = c.ticks,
            copied = c.copied_frames,
            accepted = c.accepted_frames,
            nonzero = c.nonzero_frames,
            blocks = c.admitted_blocks,
            full = c.backpressure,
            retried = c.retried_frames,
            padded = c.padded_frames,
            deferred = c.deferred,
            faults = c.faults,
            filled,
            "sysmix output"
        );
    }

    /// The ring cursor at `offset` as a host atomic on the guest's memory, the word the guest's
    /// own exclusive loads and stores see, or `None` unless its page is mapped read-write.
    fn cursor_word(&self, offset: u64) -> Option<&AtomicU32> {
        let ptr = self
            .memory
            .access(self.buffer + offset, MemoryPermission::READ_WRITE)
            .ok()?;
        debug_assert_eq!(ptr as usize % align_of::<AtomicU32>(), 0);
        // SAFETY: the ring is page aligned and both cursors sit at 4-byte aligned offsets of its
        // first two pages, so the word is aligned and inside one mapped page, which stays valid
        // host memory for the life of the address space (the kernel's address arbiter relies on
        // the same).
        Some(unsafe { AtomicU32::from_ptr(ptr.cast()) })
    }

    fn cursor(&self, offset: u64) -> Option<u32> {
        Some(self.cursor_word(offset)?.load(Ordering::Acquire))
    }

    /// Atomically stores `next` in the cursor at `offset` if it still holds `expected` (the
    /// reference's `WriteExclusive32`): `None` when the memory is gone, `Some(false)` when the
    /// guest moved the cursor first.
    fn swap_cursor(&self, offset: u64, expected: u32, next: u32) -> Option<bool> {
        let word = self.cursor_word(offset)?;
        let swapped = word
            .compare_exchange(expected, next, Ordering::AcqRel, Ordering::Acquire)
            .is_ok();
        self.memory.report_write(word.as_ptr().cast());
        Some(swapped)
    }

    fn read_payload(&self, slot: u32, out: &mut [u8], frame_bytes: u32) -> bool {
        out.is_empty()
            || self
                .memory
                .read_bytes(self.payload(slot, frame_bytes), out)
                .is_ok()
    }

    fn write_payload(&self, slot: u32, data: &[u8], frame_bytes: u32) -> bool {
        data.is_empty()
            || self
                .memory
                .write_bytes(self.payload(slot, frame_bytes), data)
                .is_ok()
    }

    fn payload(&self, slot: u32, frame_bytes: u32) -> u64 {
        self.buffer + PAYLOAD_OFFSET + u64::from(slot) * u64::from(frame_bytes)
    }

    /// The ring's cursors, or `None` when they are unreadable, out of range, or the guest moved
    /// the cursor the service owns.
    fn occupancy(&self, p: &Inner) -> Option<Occupancy> {
        let read = self.cursor(READ_OFFSET)?;
        let write = self.cursor(WRITE_OFFSET)?;
        let owned = if self.input { write } else { read };
        if owned != p.consumer || read >= p.slots || write >= p.slots {
            return None;
        }
        Some((read, write, (write + p.slots - read) % p.slots))
    }

    /// Signals the data event when the captured input or free output space reaches the threshold.
    fn notify(&self, p: &Inner, filled: u32) {
        let available = if self.input {
            filled
        } else {
            p.slots - 1 - filled
        };
        if p.data_mode == 2 && available >= p.threshold {
            self.data_event.signal();
        }
    }

    /// Stops the stream after bad guest memory, closing its host stream so nothing queued keeps
    /// playing, and tells the guest through the state event.
    fn fault(&self, p: &mut Inner, result: lichii_common::ResultCode) -> lichii_common::ResultCode {
        p.faulted = true;
        p.engaged = false;
        p.host = None;
        p.counters.faults += 1;
        self.state_event.signal();
        tracing::error!(
            stream = self.id,
            copied = p.counters.copied_frames,
            accepted = p.counters.accepted_frames,
            "sysmix stream fault {:#x}",
            result.0
        );
        result
    }
}
