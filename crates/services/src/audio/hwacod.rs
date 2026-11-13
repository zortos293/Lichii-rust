//! `IClientSession` (hwopus command 100) and `IHardwareAudioCodecManagerSession`: the batch
//! Opus decoder that Switch 2 titles drive through shared memory.
//!
//! The game lays out a manager region, the decoder instances and a task queue in transfer
//! memory, and the I/O slots and a completion ring in its own memory. `Execute` takes a list of
//! tasks, decodes each task's packet from its input slot into its output slot, and publishes a
//! completion by advancing the ring's producer index. The game reports how far it has read by
//! moving the consumer index, which frees the I/O slots of the completions it has seen.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use bytemuck::{Pod, Zeroable};
use lichii_audio::sink::result::{
    BUFFER_COUNT_REACHED, INVALID_ADDRESS_INFO, INVALID_ARGUMENT, INVALID_HANDLE, INVALID_REVISION,
    NOT_SUPPORTED, OPERATION_FAILED, OUT_OF_SESSIONS,
};
use lichii_common::{HleResult, ResultCode};
use lichii_kernel::Event;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory::MemoryState;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::process::{Process, ProcessState};
use lichii_memory::{AddressSpace, MemoryPermission};
use opus1::multistream::MultistreamDecoder;
use opus1::{Channels, OpusDecoder, SampleRate};
use parking_lot::Mutex;

use super::hwopus::{
    ALIGNED_MONO_DECODER_SIZE, ALIGNED_STEREO_DECODER_SIZE, BUFFER_TOO_SMALL, DECODE_OBJECT_SIZE,
    DECODER_SIZE, INPUT_DATA_TOO_SMALL, INVALID_OPUS_CHANNEL_COUNT, INVALID_OPUS_SAMPLE_RATE,
    LIB_OPUS_BAD_ARG, LIB_OPUS_INVALID_PACKET, MULTI_STREAM_HEADER_SIZE, opus_result,
};
use crate::{Service, ServiceContext};

/// Size of the codec parameter block `GetInstanceSize` and `InitializeInstance` read.
const PARAMETER_SIZE: u64 = 0x120;
/// Where the decoder object starts in an instance: after the config, aligned to 64 bytes.
const DECODER_OFFSET: u64 = 0x140;
const MAX_IO_COUNT: u32 = 65536;
const PAGE: u64 = 4096;
/// Offsets in the completion ring: the producer index, the consumer index and the entries.
const RING_PRODUCER: u64 = 0x4;
const RING_CONSUMER: u64 = 0x44;
const RING_ENTRIES: u64 = 0x80;
/// `sizeof(ManagerState)` and `sizeof(IoLease)`, which `GetManagerSize` counts.
const MANAGER_STATE_SIZE: u64 = 64;
const IO_LEASE_SIZE: u64 = 16;

/// `Hwacod::SharedMemoryWire`: where each region lives in transfer and process memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct SharedMemoryWire {
    version: u32,
    manager_size: u32,
    manager_offset: u32,
    instances_size: u32,
    instances_offset: u32,
    transfer_queue_size: u32,
    transfer_queue_offset: u32,
    transfer_size: u32,
    io_size: u32,
    io_offset: u32,
    completion_size: u32,
    completion_offset: u32,
    process_size: u32,
    instance_stride: u32,
    instance_count: u32,
    input_stride: u32,
    output_stride: u32,
    io_count: u32,
    reserved: [u32; 6],
}
const _: () = assert!(size_of::<SharedMemoryWire>() == 0x60);
const _: () = assert!(std::mem::offset_of!(SharedMemoryWire, completion_offset) == 0x2c);
const _: () = assert!(std::mem::offset_of!(SharedMemoryWire, instance_stride) == 0x34);

/// One decode request.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct Task {
    tag: u16,
    instance: u16,
    io: u16,
    sequence: u16,
    input_bytes: u32,
    reset: u8,
    notify: u8,
    reserved: u16,
}
const _: () = assert!(size_of::<Task>() == 0x10);
const _: () = assert!(std::mem::offset_of!(Task, reset) == 12);

/// One entry of the completion ring.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Completion {
    tag: u16,
    instance: u16,
    io: u16,
    sequence: u16,
    duration_ns: i64,
    output_bytes: u32,
    result: u32,
    reserved: [u8; 40],
}
const _: () = assert!(size_of::<Completion>() == 0x40);
const _: () = assert!(std::mem::offset_of!(Completion, result) == 0x14);

impl Completion {
    fn new(task: &Task, result: HleResult<u32>, duration_ns: i64) -> Self {
        let (output_bytes, result) = match result {
            Ok(bytes) => (bytes, ResultCode::SUCCESS),
            Err(code) => (0, code),
        };
        Self {
            tag: task.tag,
            instance: task.instance,
            io: task.io,
            sequence: task.sequence,
            duration_ns: duration_ns.max(0),
            output_bytes,
            result: result.0,
            reserved: [0; 40],
        }
    }
}

/// `GetManagerSize`.
fn manager_size(io_count: u32) -> HleResult<u32> {
    if io_count == 0 || io_count > MAX_IO_COUNT {
        return Err(INVALID_ARGUMENT);
    }
    Ok((MANAGER_STATE_SIZE + u64::from(io_count) * IO_LEASE_SIZE).next_multiple_of(64) as u32)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodecKind {
    Opus,
    Multistream,
}

/// `AudioCore::OpusDecoder::CodecConfig`.
#[derive(Clone, Copy, Debug)]
struct CodecConfig {
    kind: CodecKind,
    rate: u32,
    channels: u32,
    streams: u32,
    coupled: u32,
    frame_samples: u32,
    mapping: [u8; 255],
}

impl CodecConfig {
    /// `Hwacod::ParseParameters`: reads and validates the codec parameter block.
    fn parse(bytes: &[u8]) -> HleResult<Self> {
        if bytes.len() as u64 != PARAMETER_SIZE {
            return Err(INVALID_ARGUMENT);
        }
        let word =
            |offset: usize| i32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let kind = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let (rate, channels) = (word(8), word(12));
        let duration = i64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let (streams, coupled) = (word(24), word(28));
        let kind = match kind {
            0 => return Err(NOT_SUPPORTED),
            1 => CodecKind::Opus,
            2 => CodecKind::Multistream,
            _ => return Err(INVALID_ARGUMENT),
        };
        if !matches!(rate, 8_000 | 12_000 | 16_000 | 24_000 | 48_000) {
            return Err(INVALID_OPUS_SAMPLE_RATE);
        }
        let max_channels = if kind == CodecKind::Opus { 2 } else { 255 };
        if channels <= 0 || channels > max_channels {
            return Err(INVALID_OPUS_CHANNEL_COUNT);
        }
        if !matches!(duration / 1000, 2_500 | 5_000 | 10_000 | 20_000) {
            return Err(INVALID_ARGUMENT);
        }
        let mut config = Self {
            kind,
            rate: rate as u32,
            channels: channels as u32,
            streams: 0,
            coupled: 0,
            frame_samples: (rate as u64 * duration as u64).div_ceil(1_000_000_000) as u32,
            mapping: [0; 255],
        };
        if kind == CodecKind::Multistream {
            let streams_valid = streams > 0
                && streams <= 255
                && coupled >= 0
                && coupled <= streams
                && streams + coupled <= channels;
            if !streams_valid {
                return Err(INVALID_ARGUMENT);
            }
            config.streams = streams as u32;
            config.coupled = coupled as u32;
            config.mapping.copy_from_slice(&bytes[0x20..0x20 + 255]);
            let outputs = config.streams + config.coupled;
            let mapping = &config.mapping[..config.channels as usize];
            if mapping.iter().any(|&c| c != 255 && u32::from(c) >= outputs) {
                return Err(INVALID_ARGUMENT);
            }
        }
        config.required_bytes()?;
        Ok(config)
    }

    /// `HostDecoder::RequiredBytes`: the instance size `GetInstanceSize` reports, the config
    /// followed by the DSP's decoder object.
    fn required_bytes(&self) -> HleResult<u32> {
        let frames = [
            self.rate / 400,
            self.rate / 200,
            self.rate / 100,
            self.rate / 50,
        ];
        if !frames
            .iter()
            .any(|&size| self.frame_samples == size || self.frame_samples == size + 1)
        {
            return Err(LIB_OPUS_BAD_ARG);
        }
        let decoder = match self.kind {
            CodecKind::Opus => DECODE_OBJECT_SIZE + DECODER_SIZE[self.channels as usize - 1],
            CodecKind::Multistream => {
                DECODE_OBJECT_SIZE
                    + MULTI_STREAM_HEADER_SIZE
                    + self.coupled * ALIGNED_STEREO_DECODER_SIZE
                    + (self.streams - self.coupled) * ALIGNED_MONO_DECODER_SIZE
            }
        };
        let required = DECODER_OFFSET + u64::from(decoder).next_multiple_of(64);
        u32::try_from(required).map_err(|_| BUFFER_TOO_SMALL)
    }
}

/// One decoder instance (`HostDecoder`).
struct Instance {
    config: CodecConfig,
    decoder: Decoder,
}

enum Decoder {
    Single(Box<OpusDecoder>),
    Multi(Box<MultistreamDecoder>),
}

impl Instance {
    fn new(config: CodecConfig) -> HleResult<Self> {
        let rate = SampleRate::try_from(config.rate).map_err(opus_result)?;
        let decoder = match config.kind {
            CodecKind::Opus => {
                let channels = if config.channels == 1 {
                    Channels::Mono
                } else {
                    Channels::Stereo
                };
                Decoder::Single(Box::new(
                    OpusDecoder::new(rate, channels).map_err(opus_result)?,
                ))
            }
            CodecKind::Multistream => Decoder::Multi(Box::new(
                MultistreamDecoder::new(
                    rate,
                    config.channels as usize,
                    config.streams as usize,
                    config.coupled as usize,
                    &config.mapping[..config.channels as usize],
                )
                .map_err(opus_result)?,
            )),
        };
        Ok(Self { config, decoder })
    }

    /// Decodes one framed packet (a big-endian payload size and the final range, then the
    /// payload) into `output`. Returns the samples decoded per channel.
    fn decode(&mut self, framed: &[u8], output: &mut [i16], reset: bool) -> HleResult<u32> {
        if reset {
            match &mut self.decoder {
                Decoder::Single(decoder) => decoder.reset_state(),
                Decoder::Multi(decoder) => decoder.reset_state(),
            }
        }
        if framed.len() < 8 {
            return Err(INPUT_DATA_TOO_SMALL);
        }
        let payload = u32::from_be_bytes(framed[..4].try_into().unwrap()) as usize;
        if payload == 0 {
            return Err(LIB_OPUS_INVALID_PACKET);
        }
        if payload > framed.len() - 8 || payload > i32::MAX as usize {
            return Err(BUFFER_TOO_SMALL);
        }
        let frames =
            (self.config.frame_samples as usize).min(output.len() / self.config.channels as usize);
        if frames == 0 {
            return Err(BUFFER_TOO_SMALL);
        }
        let packet = &framed[8..8 + payload];
        let decoded = match &mut self.decoder {
            Decoder::Single(decoder) => decoder.decode_i16(packet, frames, output),
            Decoder::Multi(decoder) => decoder.decode_i16(packet, frames, output),
        }
        .map_err(opus_result)?;
        Ok(decoded as u32)
    }
}

/// The tags that tell managers apart in tasks and completions, shared by every client session
/// of the `hwopus` service.
pub(super) struct CodecTags(Mutex<TagState>);

struct TagState {
    live: Vec<u64>,
    next: u16,
}

impl CodecTags {
    pub(super) fn new() -> Self {
        Self(Mutex::new(TagState {
            live: vec![0; 65536 / 64],
            next: 1,
        }))
    }

    fn acquire(self: &Arc<Self>) -> HleResult<Tag> {
        let mut state = self.0.lock();
        for _ in 0..u16::MAX {
            let candidate = state.next;
            state.next = if candidate == u16::MAX {
                1
            } else {
                candidate + 1
            };
            let (word, bit) = (usize::from(candidate) / 64, candidate % 64);
            if state.live[word] & (1 << bit) == 0 {
                state.live[word] |= 1 << bit;
                return Ok(Tag(self.clone(), candidate));
            }
        }
        Err(OUT_OF_SESSIONS)
    }
}

/// A live tag, released when its manager closes.
struct Tag(Arc<CodecTags>, u16);

impl Drop for Tag {
    fn drop(&mut self) {
        let (word, bit) = (usize::from(self.1) / 64, self.1 % 64);
        self.0.0.lock().live[word] &= !(1 << bit);
    }
}

/// The input buffer a command takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum InputBuffer {
    None,
    /// The codec parameter block, in an X buffer.
    Parameter,
    /// A task list, in an X or an A buffer but not both.
    Tasks,
}

/// `GuardHardwareAudioCodecRequest`: rejects requests whose buffers, handles or data do not
/// have the command's shape. Returns the input buffer's address and size, when the command
/// takes one.
fn guard(
    ctx: &HleRequestContext,
    scalar_bytes: usize,
    buffer: InputBuffer,
) -> HleResult<Option<(u64, u64)>> {
    if !ctx.buffers_b().is_empty()
        || !ctx.buffers_w().is_empty()
        || !ctx.buffers_c().is_empty()
        || ctx.input().len() < scalar_bytes
    {
        return Err(INVALID_ARGUMENT);
    }
    if ctx.pid().is_some() {
        return Err(INVALID_HANDLE);
    }
    let (x, a) = (ctx.buffers_x(), ctx.buffers_a());
    let input = match buffer {
        InputBuffer::None if x.is_empty() && a.is_empty() => return Ok(None),
        InputBuffer::Parameter if x.len() == 1 && a.is_empty() && x[0].size == PARAMETER_SIZE => {
            x[0]
        }
        InputBuffer::Tasks if x.len() <= 1 && a.len() <= 1 => {
            match (x.first().copied(), a.first().copied()) {
                (Some(x), Some(a)) if x.size != 0 && a.size != 0 => return Err(INVALID_ARGUMENT),
                (_, Some(a)) if a.size != 0 => a,
                (Some(x), _) => x,
                _ => return Err(INVALID_ARGUMENT),
            }
        }
        _ => return Err(INVALID_ARGUMENT),
    };
    let task = size_of::<Task>() as u64;
    if buffer == InputBuffer::Tasks
        && (!input.size.is_multiple_of(task) || input.size > u64::from(MAX_IO_COUNT) * task)
    {
        return Err(INVALID_ARGUMENT);
    }
    if input.address.checked_add(input.size).is_none() {
        return Err(INVALID_ARGUMENT);
    }
    Ok(Some((input.address, input.size)))
}

fn read_input(ctx: &HleRequestContext, input: Option<(u64, u64)>) -> HleResult<Vec<u8>> {
    let (address, size) = input.ok_or(INVALID_ARGUMENT)?;
    let mut bytes = vec![0; size as usize];
    ctx.memory()
        .read_bytes(address, &mut bytes)
        .map_err(|_| INVALID_ARGUMENT)?;
    Ok(bytes)
}

/// `IClientSession`, which `hwopus` command 100 returns.
pub(super) struct ClientSession {
    pub(super) context: Arc<ServiceContext>,
    pub(super) tags: Arc<CodecTags>,
}

impl Service for ClientSession {
    fn name(&self) -> &str {
        "IClientSession"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            0 => guard(ctx, 0, InputBuffer::Parameter)
                .and_then(|input| read_input(ctx, input))
                .and_then(|bytes| CodecConfig::parse(&bytes)?.required_bytes())
                .map(|size| ctx.push(size)),
            1 => guard(ctx, 4, InputBuffer::None)
                .and_then(|_| manager_size(ctx.pop()))
                .map(|size| ctx.push(size)),
            10 => guard(ctx, 0x80, InputBuffer::None).and_then(|_| self.open_manager(ctx)),
            _ => return None,
        };
        Some(result.into())
    }
}

impl ClientSession {
    /// `OpenManager`: takes over the game's layout and returns the manager's tag, its
    /// completion event and the manager session.
    fn open_manager(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let layout: SharedMemoryWire = ctx.pop();
        let region = Region {
            transfer_size: ctx.pop(),
            process_address: ctx.pop(),
            process_size: ctx.pop(),
            properties: ctx.pop(),
        };
        let transfer = ctx
            .copy_handle::<TransferMemory>(0)
            .map_err(|_| INVALID_HANDLE)?;
        let process = ctx.copy_handle::<Process>(1).map_err(|_| INVALID_HANDLE)?;
        let manager = Manager::open(&self.tags, &layout, &region, transfer, process)?;
        ctx.push(u32::from(manager.tag.1));
        ctx.push_copy_object(KObject::ReadableEvent(manager.event.clone()));
        ctx.push_interface(self.context.handler(manager));
        Ok(())
    }
}

/// The scalar arguments of `OpenManager` besides the layout.
struct Region {
    transfer_size: u64,
    process_address: u64,
    process_size: u64,
    properties: u64,
}

fn contains(total: u64, offset: u64, size: u64) -> bool {
    offset <= total && size <= total - offset
}

fn overlaps(a: u64, a_size: u64, b: u64, b_size: u64) -> bool {
    if a <= b {
        b - a < a_size
    } else {
        a - b < b_size
    }
}

/// `Hwacod::ValidateLayout`.
fn validate_layout(l: &SharedMemoryWire, r: &Region) -> HleResult<()> {
    if l.version != 1 {
        return Err(INVALID_REVISION);
    }
    if l.reserved.iter().any(|&v| v != 0) || r.properties & 7 == 0 {
        return Err(INVALID_ARGUMENT);
    }
    let strides_valid = l.instance_count > 0
        && l.instance_count <= MAX_IO_COUNT
        && l.instance_stride > 0
        && l.instance_stride.is_multiple_of(64)
        && l.input_stride >= 0x600
        && l.input_stride.is_multiple_of(64)
        && l.output_stride > 0
        && l.output_stride.is_multiple_of(64);
    if !strides_valid {
        return Err(INVALID_ARGUMENT);
    }
    let manager_bytes = manager_size(l.io_count)?;
    let (layout_transfer, layout_process) = (u64::from(l.transfer_size), u64::from(l.process_size));
    let sizes_valid = r.transfer_size > 0
        && r.transfer_size.is_multiple_of(PAGE)
        && r.process_size > 0
        && r.process_size.is_multiple_of(PAGE)
        && r.process_address.is_multiple_of(PAGE)
        && r.process_size <= u64::MAX - r.process_address
        && layout_transfer <= r.transfer_size
        && layout_process <= r.process_size
        && layout_transfer.is_multiple_of(PAGE)
        && layout_process.is_multiple_of(PAGE);
    if !sizes_valid {
        return Err(INVALID_ADDRESS_INFO);
    }
    let queue_bytes = (0xc0 + u64::from(l.io_count) * 0x40).next_multiple_of(PAGE);
    let io_stride = u64::from(l.input_stride) + u64::from(l.output_stride);
    let regions_fit = l.manager_size >= manager_bytes
        && u64::from(l.instances_size)
            >= u64::from(l.instance_count) * u64::from(l.instance_stride)
        && u64::from(l.io_size) >= u64::from(l.io_count) * io_stride
        && u64::from(l.transfer_queue_size) >= queue_bytes
        && u64::from(l.completion_size) >= queue_bytes;
    if !regions_fit {
        return Err(BUFFER_TOO_SMALL);
    }
    let transfer_regions = [
        (l.manager_offset, l.manager_size, 64),
        (l.instances_offset, l.instances_size, 64),
        (l.transfer_queue_offset, l.transfer_queue_size, PAGE),
    ];
    for (i, &(offset, size, alignment)) in transfer_regions.iter().enumerate() {
        let (offset, size) = (u64::from(offset), u64::from(size));
        if !offset.is_multiple_of(alignment) || !contains(layout_transfer, offset, size) {
            return Err(INVALID_ADDRESS_INFO);
        }
        if transfer_regions[..i]
            .iter()
            .any(|&(o, s, _)| overlaps(offset, size, o.into(), s.into()))
        {
            return Err(INVALID_ADDRESS_INFO);
        }
    }
    let io = (u64::from(l.io_offset), u64::from(l.io_size));
    let completion = (u64::from(l.completion_offset), u64::from(l.completion_size));
    let process_regions_valid = io.0.is_multiple_of(64)
        && completion.0.is_multiple_of(PAGE)
        && contains(layout_process, io.0, io.1)
        && contains(layout_process, completion.0, completion.1)
        && !overlaps(io.0, io.1, completion.0, completion.1);
    if !process_regions_valid {
        return Err(INVALID_ADDRESS_INFO);
    }
    Ok(())
}

/// `Hwacod::IoLease`: an I/O slot a committed task holds until the game reads its completion.
#[derive(Clone, Copy, Default)]
struct IoLease {
    occupied: bool,
    completion_slot: u32,
    instance: u16,
}

/// `Hwacod::BatchState`: the ring positions and the I/O slots in use.
struct BatchState {
    tag: u16,
    write_index: u32,
    reclaimed_read_index: u32,
    outstanding: u32,
    leases: Vec<IoLease>,
    seen: Vec<bool>,
}

impl BatchState {
    fn new(io_count: u32, tag: u16) -> Self {
        Self {
            tag,
            write_index: 0,
            reclaimed_read_index: 0,
            outstanding: 0,
            leases: vec![IoLease::default(); io_count as usize],
            seen: vec![false; io_count as usize],
        }
    }

    fn capacity(&self) -> u32 {
        self.leases.len() as u32 + 1
    }

    /// Frees the I/O slots of the completions the game has read up to `consumer`.
    fn reclaim(&mut self, consumer: u32, producer: u32) -> HleResult<()> {
        let capacity = self.capacity();
        if consumer >= capacity || producer != self.write_index {
            return Err(INVALID_ARGUMENT);
        }
        let read = self.reclaimed_read_index;
        let consumed = (consumer + capacity - read) % capacity;
        if consumed > self.outstanding {
            return Err(INVALID_ARGUMENT);
        }
        let is_read = |lease: &IoLease| {
            lease.occupied && (lease.completion_slot + capacity - read) % capacity < consumed
        };
        if self.leases.iter().filter(|l| is_read(l)).count() as u32 != consumed {
            return Err(INVALID_ARGUMENT);
        }
        for lease in self.leases.iter_mut().filter(|l| is_read(l)) {
            *lease = IoLease::default();
        }
        self.reclaimed_read_index = consumer;
        self.outstanding -= consumed;
        Ok(())
    }

    fn validate(
        &mut self,
        tasks: &[Task],
        instances: &[Option<Instance>],
        input_stride: u32,
    ) -> HleResult<()> {
        if tasks.len() > self.leases.len() - self.outstanding as usize {
            return Err(BUFFER_COUNT_REACHED);
        }
        self.seen.fill(false);
        for task in tasks {
            let io = usize::from(task.io);
            let valid = task.tag == self.tag
                && instances
                    .get(usize::from(task.instance))
                    .is_some_and(Option::is_some)
                && io < self.leases.len()
                && task.input_bytes <= input_stride
                && task.reset <= 1
                && task.notify <= 1
                && task.reserved == 0;
            if !valid {
                return Err(INVALID_ARGUMENT);
            }
            if self.leases[io].occupied || self.seen[io] {
                return Err(BUFFER_COUNT_REACHED);
            }
            self.seen[io] = true;
        }
        Ok(())
    }

    fn commit(&mut self, task: &Task) {
        self.leases[usize::from(task.io)] = IoLease {
            occupied: true,
            completion_slot: self.write_index,
            instance: task.instance,
        };
        self.write_index = (self.write_index + 1) % self.capacity();
        self.outstanding += 1;
    }

    fn has_outstanding(&self, instance: u16) -> bool {
        self.leases
            .iter()
            .any(|l| l.occupied && l.instance == instance)
    }
}

/// A guest word the game and the manager share, as a host atomic.
fn guest_word(memory: &AddressSpace, address: u64) -> HleResult<&AtomicU32> {
    if !address.is_multiple_of(4) {
        return Err(INVALID_ADDRESS_INFO);
    }
    let ptr = memory
        .access(address, MemoryPermission::READ_WRITE)
        .map_err(|_| INVALID_ADDRESS_INFO)?;
    // SAFETY: the address is 4-byte aligned and lies in one mapped, writable guest page, which
    // stays mapped while the manager keeps the region locked.
    Ok(unsafe { AtomicU32::from_ptr(ptr.cast()) })
}

/// The game's memory region, locked like transfer memory while the manager uses it.
struct LockedRegion {
    process: Arc<Process>,
    address: u64,
    size: u64,
}

impl Drop for LockedRegion {
    fn drop(&mut self) {
        if let Err(err) = self
            .process
            .page_table()
            .unlock_for_transfer(self.address, self.size)
        {
            tracing::error!(?err, "hwacod could not unlock its process region");
        }
    }
}

/// `IHardwareAudioCodecManagerSession`.
struct Manager {
    tag: Tag,
    event: Arc<Event>,
    region: LockedRegion,
    _transfer: Arc<TransferMemory>,
    layout: SharedMemoryWire,
    ring: u64,
    state: Mutex<ManagerState>,
}

struct ManagerState {
    batch: BatchState,
    instances: Vec<Option<Instance>>,
    faulted: bool,
    tasks: Vec<Task>,
    packets: Vec<u8>,
    pcm: Vec<i16>,
}

impl Manager {
    fn open(
        tags: &Arc<CodecTags>,
        layout: &SharedMemoryWire,
        region: &Region,
        transfer: Arc<TransferMemory>,
        process: Arc<Process>,
    ) -> HleResult<Self> {
        if matches!(
            process.state(),
            ProcessState::Exiting | ProcessState::Exited
        ) {
            return Err(INVALID_HANDLE);
        }
        validate_layout(layout, region)?;
        let (address, size) = (region.process_address, region.process_size);
        if region.transfer_size > transfer.size() || !process.page_table().contains(address, size) {
            return Err(INVALID_ADDRESS_INFO);
        }
        if Arc::ptr_eq(transfer.owner(), &process)
            && overlaps(transfer.address(), transfer.size(), address, size)
        {
            return Err(INVALID_ADDRESS_INFO);
        }
        process
            .page_table()
            .lock_for_transfer(address, size, MemoryPermission::READ_WRITE)?;
        let region = LockedRegion {
            process,
            address,
            size,
        };
        let mut current = address;
        while current < address + size {
            let info = region.process.page_table().query(current);
            let valid = info.state == MemoryState::NORMAL.svc_value()
                && info.size > 0
                && info.base_address <= current
                && current - info.base_address < info.size
                && info.size <= u64::MAX - info.base_address;
            if !valid {
                return Err(INVALID_ADDRESS_INFO);
            }
            current = info.base_address + info.size;
        }
        let tag = tags.acquire()?;
        let memory = region.process.memory();
        let ring = address + u64::from(layout.completion_offset);
        if guest_word(memory, ring + RING_CONSUMER)?.load(Ordering::Acquire) != 0 {
            return Err(INVALID_ARGUMENT);
        }
        let write = |offset: u64, bytes: &[u8]| {
            memory
                .write_bytes(ring + offset, bytes)
                .map_err(|_| INVALID_ADDRESS_INFO)
        };
        write(0, &[0; 64])?;
        let empty = Completion::zeroed();
        for i in 0..=u64::from(layout.io_count) {
            let offset = RING_ENTRIES + i * size_of::<Completion>() as u64;
            write(offset, bytemuck::bytes_of(&empty))?;
        }
        tracing::info!(
            tag = tag.1,
            instances = layout.instance_count,
            io = layout.io_count,
            input_stride = layout.input_stride,
            output_stride = layout.output_stride,
            "hwacod opened a manager"
        );
        Ok(Self {
            state: Mutex::new(ManagerState {
                batch: BatchState::new(layout.io_count, tag.1),
                instances: (0..layout.instance_count).map(|_| None).collect(),
                faulted: false,
                tasks: Vec::new(),
                packets: Vec::new(),
                pcm: vec![0; layout.output_stride as usize / 2],
            }),
            tag,
            event: Arc::new(Event::new()),
            region,
            _transfer: transfer,
            layout: *layout,
            ring,
        })
    }

    fn memory(&self) -> &AddressSpace {
        self.region.process.memory()
    }

    fn reclaim(&self, state: &mut ManagerState) -> HleResult<()> {
        if state.faulted {
            return Err(OPERATION_FAILED);
        }
        let memory = self.memory();
        let consumer = guest_word(memory, self.ring + RING_CONSUMER)?.load(Ordering::Acquire);
        let producer = guest_word(memory, self.ring + RING_PRODUCER)?.load(Ordering::Acquire);
        state.batch.reclaim(consumer, producer)
    }

    /// Address of I/O slot `io`: its input area, then its output area.
    fn io_slot(&self, io: u16) -> u64 {
        let stride = u64::from(self.layout.input_stride) + u64::from(self.layout.output_stride);
        self.region.address + u64::from(self.layout.io_offset) + u64::from(io) * stride
    }

    fn initialize_instance(&self, id: u32, parameters: &[u8]) -> HleResult<()> {
        let mut state = self.state.lock();
        if state.faulted {
            return Err(OPERATION_FAILED);
        }
        if state.instances.get(id as usize).is_none_or(Option::is_some) {
            return Err(INVALID_ARGUMENT);
        }
        let config = CodecConfig::parse(parameters)?;
        let output =
            (u64::from(config.frame_samples) * u64::from(config.channels) * 2).next_multiple_of(64);
        if output > u64::from(self.layout.output_stride)
            || config.required_bytes()? > self.layout.instance_stride
        {
            return Err(BUFFER_TOO_SMALL);
        }
        state.instances[id as usize] = Some(Instance::new(config)?);
        Ok(())
    }

    fn finalize_instance(&self, id: u32) -> HleResult<()> {
        let mut state = self.state.lock();
        self.reclaim(&mut state)?;
        if state.instances.get(id as usize).is_none_or(Option::is_none) {
            return Err(INVALID_ARGUMENT);
        }
        if state.batch.has_outstanding(id as u16) {
            return Err(BUFFER_COUNT_REACHED);
        }
        state.instances[id as usize] = None;
        Ok(())
    }

    /// `Execute`: decodes every task and publishes its completion, in order.
    fn execute(&self, input: &[u8]) -> HleResult<()> {
        let mut guard = self.state.lock();
        let state = &mut *guard;
        self.reclaim(state)?;
        let task_size = size_of::<Task>();
        if input.len() / task_size > self.layout.io_count as usize {
            return Err(INVALID_ARGUMENT);
        }
        state.tasks.clear();
        state.tasks.extend(
            input
                .chunks_exact(task_size)
                .map(bytemuck::pod_read_unaligned::<Task>),
        );
        let input_stride = self.layout.input_stride;
        state
            .batch
            .validate(&state.tasks, &state.instances, input_stride)?;
        if state.tasks.is_empty() {
            return Ok(());
        }
        let stride = input_stride as usize;
        state.packets.resize(state.tasks.len() * stride, 0);
        for (task, packet) in state
            .tasks
            .iter()
            .zip(state.packets.chunks_exact_mut(stride))
        {
            self.memory()
                .read_bytes(
                    self.io_slot(task.io),
                    &mut packet[..task.input_bytes as usize],
                )
                .map_err(|_| INVALID_ADDRESS_INFO)?;
        }
        self.reclaim(state)?;
        state
            .batch
            .validate(&state.tasks, &state.instances, input_stride)?;

        let capacity = self.layout.io_count + 1;
        for (task, packet) in state.tasks.iter().zip(state.packets.chunks_exact(stride)) {
            let instance = state.instances[usize::from(task.instance)]
                .as_mut()
                .expect("validated instance");
            let start = Instant::now();
            let result = instance
                .decode(
                    &packet[..task.input_bytes as usize],
                    &mut state.pcm,
                    task.reset != 0,
                )
                .map(|samples| samples * instance.config.channels * 2);
            let completion = Completion::new(task, result, start.elapsed().as_nanos() as i64);
            let current = state.batch.write_index;
            let pcm = &state.pcm[..completion.output_bytes as usize / 2];
            let output = self.io_slot(task.io) + u64::from(input_stride);
            let entry =
                self.ring + RING_ENTRIES + u64::from(current) * size_of::<Completion>() as u64;
            let memory = self.memory();
            let published = memory
                .write_bytes(output, bytemuck::cast_slice(pcm))
                .is_ok()
                && memory
                    .write_bytes(entry, bytemuck::bytes_of(&completion))
                    .is_ok()
                && guest_word(memory, self.ring + RING_PRODUCER).is_ok_and(|producer| {
                    let swapped = producer
                        .compare_exchange(
                            current,
                            (current + 1) % capacity,
                            Ordering::Release,
                            Ordering::Relaxed,
                        )
                        .is_ok();
                    memory.report_write(producer.as_ptr().cast());
                    swapped
                });
            if !published {
                state.faulted = true;
                self.event.signal();
                tracing::error!(tag = self.tag.1, "hwacod lost a completion");
                return Err(OPERATION_FAILED);
            }
            state.batch.commit(task);
            if task.notify != 0 {
                self.event.signal();
            }
            if completion.result != ResultCode::SUCCESS.0 {
                tracing::debug!(
                    tag = self.tag.1,
                    instance = task.instance,
                    io = task.io,
                    result = completion.result,
                    "hwacod decode error"
                );
            }
        }
        Ok(())
    }
}

impl Service for Manager {
    fn name(&self) -> &str {
        "IHardwareAudioCodecManagerSession"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            0 => guard(ctx, 4, InputBuffer::Parameter).and_then(|input| {
                let id: u32 = ctx.pop();
                self.initialize_instance(id, &read_input(ctx, input)?)
            }),
            1 => guard(ctx, 4, InputBuffer::None).and_then(|_| self.finalize_instance(ctx.pop())),
            10 => guard(ctx, 0, InputBuffer::Tasks)
                .and_then(|input| self.execute(&read_input(ctx, input)?)),
            _ => return None,
        };
        Some(result.into())
    }
}

#[cfg(test)]
#[path = "hwacod_tests.rs"]
mod tests;
