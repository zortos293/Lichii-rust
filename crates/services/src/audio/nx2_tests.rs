//! The renderer, `IAudioDevice` and the Switch 2 sound mixer driven through the kernel's
//! host-side IPC client by a Switch 2 title, with synthetic audio only.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use lichii_audio::renderer::behavior::{BehaviorInParameter, BehaviorOutStatus};
use lichii_audio::renderer::memory::{MemoryPoolInParameter, MemoryPoolOutStatus, pool_state};
use lichii_audio::renderer::mix::{MixInDirtyParameter, MixInParameter};
use lichii_audio::renderer::performance::{PerformanceInParameter, PerformanceOutStatus};
use lichii_audio::renderer::result as audio_result;
use lichii_audio::renderer::revision::encode_revision;
use lichii_audio::renderer::sink::{DeviceInParameter, SinkInParameter, SinkOutStatus};
use lichii_audio::renderer::updater::{RenderInfoOutStatus, UpdateDataHeader};
use lichii_audio::renderer::voice::{
    VoiceChannelResourceInParameter, VoiceInParameterV2, VoiceOutStatus, WaveBufferInternal,
};
use lichii_audio::renderer::{AudioRendererParameterInternal, UNUSED_MIX_ID};
use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::handle_table::CURRENT_PROCESS;
use lichii_kernel::ipc::Handle;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Reply, Request};
use lichii_kernel::{Event, Kernel};
use lichii_memory::{DeviceAddressSpace, MemoryPermission};

use super::{Audio, DEVICE_NAME_SIZE};

const REVISION: u32 = encode_revision(16, true);
const HEAP_SIZE: u64 = 8 << 20;
/// Where the renderer's memory pool (and the voice's samples) live, from the heap start.
const POOL_OFFSET: u64 = 0x40_0000;
const POOL_SIZE: u64 = 0x10_000;
/// Samples of the voice's sine: 64 frames of 240.
const SINE_SAMPLES: usize = 240 * 64;
/// Where the sysmix rings live, from the heap start.
const RING_OFFSET: u64 = 0x60_0000;
const RING_SIZE: u64 = 0x4000;
const RING_FRAMES: u32 = 960;

struct Harness {
    kernel: Kernel,
    audio: Arc<Audio>,
    client: IpcClient,
    device: Arc<DeviceAddressSpace>,
    heap: u64,
    transfer_end: AtomicU64,
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.kernel.shutdown();
    }
}

fn harness() -> Harness {
    let (context, client) =
        crate::testing::context(crate::testing::SWITCH2_TITLE, Settings::default());
    let audio = super::start(&context).unwrap();
    let heap = client
        .process()
        .page_table()
        .set_heap_size(HEAP_SIZE)
        .unwrap();
    Harness {
        kernel: context.kernel.clone(),
        audio,
        device: context.gpu.host1x().memory.clone(),
        client,
        heap,
        transfer_end: AtomicU64::new(0),
    }
}

fn wait_for(what: &str, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timed out waiting for {what}"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

impl Harness {
    fn connect(&self, name: &str) -> Handle {
        self.client.connect(name).unwrap()
    }

    fn send(&self, session: Handle, request: Request) -> Reply {
        self.client.send(session, &request).unwrap()
    }

    fn ok(&self, session: Handle, request: Request) -> Reply {
        let reply = self.send(session, request);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        reply
    }

    fn interface(&self, session: Handle, request: Request) -> Handle {
        self.ok(session, request).move_handles[0]
    }

    fn event(&self, handle: Handle) -> Arc<Event> {
        match self.client.object(handle) {
            Some(KObject::ReadableEvent(event)) => event,
            _ => panic!("handle {handle:#x} is not an event"),
        }
    }

    fn write(&self, address: u64, bytes: &[u8]) {
        self.client
            .process()
            .memory()
            .write_bytes(address, bytes)
            .unwrap();
    }

    fn read_u32(&self, address: u64) -> u32 {
        self.client.process().memory().read(address).unwrap()
    }

    fn name_buffer(&self, name: &str) -> u64 {
        let mut bytes = [0u8; DEVICE_NAME_SIZE];
        bytes[..name.len()].copy_from_slice(name.as_bytes());
        self.client.alloc(&bytes)
    }

    fn read_name(&self, address: u64) -> String {
        let bytes = self.client.read(address, DEVICE_NAME_SIZE);
        let len = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        String::from_utf8_lossy(&bytes[..len]).into_owned()
    }

    /// The renderer manager `aud:u` command 2 returns.
    fn renderer_manager(&self) -> Handle {
        self.interface(self.connect("aud:u"), Request::new(2).arg(0u32))
    }

    /// A transfer memory handle over the next `size` bytes (rounded up to a page) of the heap.
    fn transfer_memory(&self, size: u64) -> Handle {
        let process = self.client.process().clone();
        let size = size.next_multiple_of(0x1000);
        let offset = self.transfer_end.fetch_add(size, Ordering::Relaxed);
        assert!(
            offset + size <= POOL_OFFSET,
            "transfer memory overlaps the pool"
        );
        let transfer = TransferMemory::new(
            process.clone(),
            self.heap + offset,
            size,
            MemoryPermission::READ_WRITE,
        )
        .unwrap();
        process
            .handles()
            .write()
            .add(KObject::TransferMemory(Arc::new(transfer)))
            .unwrap()
    }

    fn work_buffer_size(&self, manager: Handle, params: AudioRendererParameterInternal) -> u64 {
        self.ok(manager, Request::new(1).arg(params)).pop()
    }

    fn open_renderer(
        &self,
        manager: Handle,
        params: AudioRendererParameterInternal,
        size: u64,
    ) -> Reply {
        let transfer = self.transfer_memory(size);
        let request = Request::new(0)
            .arg(params)
            .arg(size)
            .arg(0u64)
            .send_pid()
            .copy_handle(transfer)
            .copy_handle(CURRENT_PROCESS);
        self.send(manager, request)
    }

    /// Sends `input` as a `RequestUpdate` with an output buffer of `output_size` bytes.
    fn request_update(&self, renderer: Handle, input: &[u8], output_size: usize) -> Reply {
        let input = self.client.alloc(input);
        let output = self.client.alloc(&vec![0; output_size]);
        let performance = self.client.alloc(&[0; 0x10]);
        let request = Request::new(4)
            .buffer_b(output, output_size as u64)
            .buffer_b(performance, 0)
            .buffer_a(input, self.update_size() as u64);
        self.send(renderer, request)
    }

    fn update_size(&self) -> usize {
        update_input(0).len()
    }
}

fn renderer_params() -> AudioRendererParameterInternal {
    AudioRendererParameterInternal {
        sample_rate: 48_000,
        sample_count: 240,
        mixes: 2,
        voices: 1,
        sinks: 1,
        revision: REVISION,
        ..AudioRendererParameterInternal::default()
    }
}

fn push<T: Pod>(out: &mut Vec<u8>, value: &T) -> u32 {
    out.extend_from_slice(bytemuck::bytes_of(value));
    size_of::<T>() as u32
}

/// One update for [`renderer_params`]: a pool over the samples at `pool`, a mono PCM16 voice
/// playing them into both channels of the final mix, and a stereo device sink. The voice's
/// unused wave buffers are marked as already sent, as the SDK sends them.
fn update_input(pool: u64) -> Vec<u8> {
    let mut header = UpdateDataHeader {
        revision: REVISION,
        ..UpdateDataHeader::default()
    };
    let mut body = Vec::new();
    header.behavior_size = push(
        &mut body,
        &BehaviorInParameter {
            revision: REVISION,
            ..BehaviorInParameter::default()
        },
    );
    let pools = [MemoryPoolInParameter {
        address: pool,
        size: POOL_SIZE,
        state: pool_state::REQUEST_ATTACH,
        ..MemoryPoolInParameter::default()
    }];
    for i in 0..4 {
        header.memory_pool_size += push(&mut body, pools.get(i).unwrap_or(&Zeroable::zeroed()));
    }
    let mut resource = VoiceChannelResourceInParameter::zeroed();
    resource.in_use = 1;
    resource.mix_volumes[..2].fill(1.0);
    header.voice_resources_size = push(&mut body, &resource);
    let mut voice = VoiceInParameterV2::zeroed();
    voice.in_use = 1;
    voice.is_new = 1;
    voice.sample_format = 2;
    voice.sample_rate = 48_000;
    voice.channel_count = 1;
    voice.pitch = 1.0;
    voice.volume = 1.0;
    voice.wave_buffer_count = 1;
    voice.mix_id = 0;
    voice.splitter_id = u32::MAX;
    for unused in &mut voice.wave_buffers[1..] {
        unused.sent_to_dsp = 1;
    }
    voice.wave_buffers[0] = WaveBufferInternal {
        address: pool,
        size: (SINE_SAMPLES * 2) as u64,
        end_offset: SINE_SAMPLES as i32,
        ..WaveBufferInternal::default()
    };
    header.voices_size = push(&mut body, &voice);
    let mut dirty = MixInDirtyParameter::zeroed();
    dirty.count = 1;
    header.mix_size = push(&mut body, &dirty);
    let mut mix = MixInParameter::zeroed();
    mix.volume = 1.0;
    mix.sample_rate = 48_000;
    mix.buffer_count = 2;
    mix.in_use = 1;
    mix.is_dirty = 1;
    mix.dest_mix_id = UNUSED_MIX_ID;
    mix.dest_splitter_id = -1;
    header.mix_size += push(&mut body, &mix);
    let mut device = DeviceInParameter::zeroed();
    device.name[..12].copy_from_slice(b"MainAudioOut");
    device.input_count = 2;
    device.inputs[..2].copy_from_slice(&[0, 1]);
    let mut sink = SinkInParameter::zeroed();
    sink.sink_type = 1;
    sink.in_use = 1;
    sink.specific[..size_of::<DeviceInParameter>()].copy_from_slice(bytemuck::bytes_of(&device));
    header.sinks_size = push(&mut body, &sink);
    header.performance_buffer_size = push(&mut body, &PerformanceInParameter::default());
    header.size = (size_of::<UpdateDataHeader>() + body.len()) as u32;
    let mut input = bytemuck::bytes_of(&header).to_vec();
    input.extend(body);
    input
}

/// The output buffer size a [`update_input`] update fills to the byte.
fn update_output_size() -> usize {
    size_of::<UpdateDataHeader>()
        + 4 * size_of::<MemoryPoolOutStatus>()
        + size_of::<VoiceOutStatus>()
        + size_of::<SinkOutStatus>()
        + size_of::<PerformanceOutStatus>()
        + size_of::<BehaviorOutStatus>()
        + size_of::<RenderInfoOutStatus>()
}

#[test]
fn a_switch2_title_renders_a_voice_through_aud_u() {
    let h = harness();
    let manager = h.renderer_manager();
    let params = renderer_params();
    let size = h.work_buffer_size(manager, params);
    assert!(size > 0);
    let mut reply = h.open_renderer(manager, params, size);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let renderer = reply.move_handles[0];
    let mut sample_rate = h.ok(renderer, Request::new(0));
    assert_eq!(sample_rate.pop::<u32>(), 48_000);
    assert_eq!(h.ok(renderer, Request::new(3)).pop::<u32>(), 1, "stopped");
    reply = h.ok(renderer, Request::new(7));
    let rendered = h.event(reply.copy_handles[0]);

    let pool = h.heap + POOL_OFFSET;
    let sine: Vec<i16> = (0..SINE_SAMPLES)
        .map(|i| (f32::sin(i as f32 * std::f32::consts::TAU / 48.0) * 12_000.0) as i16)
        .collect();
    h.write(pool, bytemuck::cast_slice(&sine));
    let reply = h.request_update(renderer, &update_input(pool), update_output_size());
    assert_eq!(reply.result, ResultCode::SUCCESS);

    let session = h.audio.renderer.session(0).unwrap();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(
        session.frames_rendered(),
        0,
        "a stopped session renders nothing"
    );
    h.ok(renderer, Request::new(5));
    assert_eq!(h.ok(renderer, Request::new(3)).pop::<u32>(), 0, "started");
    rendered.clear();
    wait_for("audible frames", || session.audible_frames() >= 240 * 4);
    assert!(
        rendered.is_signaled(),
        "the rendered event fires each frame"
    );
    assert!(session.stream().expected_played_sample_count() > 0);
    h.ok(renderer, Request::new(6));
}

#[test]
fn audio_device_names_volume_and_channels() {
    let h = harness();
    let manager = h.renderer_manager();
    let device = h.interface(manager, Request::new(4).arg(REVISION).arg(0u64).send_pid());

    let names = h.client.alloc(&[0; DEVICE_NAME_SIZE * 4]);
    let mut reply = h.ok(
        device,
        Request::new(0).buffer_b(names, (DEVICE_NAME_SIZE * 4) as u64),
    );
    assert_eq!(reply.pop::<u32>(), 4);
    assert_eq!(
        h.read_name(names + 2 * DEVICE_NAME_SIZE as u64),
        "AudioTvOutput"
    );
    assert_eq!(
        h.read_name(names + 3 * DEVICE_NAME_SIZE as u64),
        "AudioUsbDeviceOutput"
    );

    let tv = h.name_buffer("AudioTvOutput");
    let size = DEVICE_NAME_SIZE as u64;
    h.ok(device, Request::new(1).arg(0.5f32).buffer_a(tv, size));
    assert_eq!(h.audio.sink.device_volume(), 0.5);
    let mut reply = h.ok(device, Request::new(2).buffer_a(tv, size));
    assert_eq!(reply.pop::<f32>(), 0.5);
    let jack = h.name_buffer("AudioStereoJackOutput");
    let mut reply = h.ok(device, Request::new(2).buffer_a(jack, size));
    assert_eq!(reply.pop::<f32>(), 1.0);

    let out = h.client.alloc(&[0xff; DEVICE_NAME_SIZE]);
    h.ok(device, Request::new(3).buffer_b(out, size));
    assert_eq!(h.read_name(out), "AudioTvOutput");
    let mut reply = h.ok(device, Request::new(5));
    assert_eq!(reply.pop::<u32>(), 2);
    let reply = h.ok(device, Request::new(4));
    assert!(h.event(reply.copy_handles[0]).is_signaled());

    assert_eq!(
        h.send(device, Request::new(2)).result,
        audio_result::INSUFFICIENT_BUFFER
    );
    assert_eq!(
        h.send(device, Request::new(3)).result,
        audio_result::INSUFFICIENT_BUFFER
    );
}

/// `audsmx:u` user session for the client process.
fn user_session(h: &Harness) -> Handle {
    h.interface(
        h.connect("audsmx:u"),
        Request::new(0).copy_handle(CURRENT_PROCESS),
    )
}

fn create_stream(h: &Harness, session: Handle, device: u32, flags: u32, ring: u64) -> Reply {
    let name = h.name_buffer("synthetic");
    let request = Request::new(1)
        .arg(device)
        .arg(flags)
        .arg(0u64)
        .arg(ring)
        .arg(RING_SIZE)
        .buffer_a(name, DEVICE_NAME_SIZE as u64);
    h.send(session, request)
}

fn open_stream(h: &Harness, stream: Handle, sample_type: u32, channels: u32) -> ResultCode {
    let args = [sample_type, channels, 48_000, RING_FRAMES, 0, 0];
    h.send(stream, Request::new(0).arg(args)).result
}

#[test]
fn sysmix_pumps_a_stereo_s16_ring_into_the_host_stream() {
    let h = harness();
    let session = user_session(&h);
    let ring = h.heap + RING_OFFSET;
    let mut reply = create_stream(&h, session, 1, 0x24, ring);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let id: u32 = reply.pop();
    let stream = reply.move_handles[0];

    const QUEUED: u32 = 720;
    let frames: Vec<i16> = (0..QUEUED * 2).map(|i| 1000 + (i % 7) as i16).collect();
    h.write(ring + 0x1040, bytemuck::cast_slice(&frames));
    h.write(ring + 0x1004, &QUEUED.to_le_bytes());
    assert_eq!(open_stream(&h, stream, 2, 2), ResultCode::SUCCESS);
    assert_eq!(h.ok(stream, Request::new(2)).pop::<u32>(), 1, "open");
    const DADDR: u64 = 0x1_0000_0000;
    h.device
        .map_process(DADDR, h.client.process().memory(), ring, 0x1000)
        .unwrap();
    let cursors = h.device.cache_pages(DADDR, 0x1000);

    let threshold = 480i32;
    let reply = h.ok(stream, Request::new(7).arg(2u32).arg(threshold));
    let data = h.event(reply.copy_handles[0]);
    assert!(
        !data.is_signaled(),
        "240 free frames are below the threshold"
    );

    let devices = h.interface(
        h.connect("auddmg:u"),
        Request::new(0).arg([0u64; 2]).send_pid(),
    );
    let name = h.name_buffer("AudioVirtualOutput0");
    let attach = |functionality: u32| {
        let request = Request::new(700)
            .arg(id)
            .arg(functionality)
            .buffer_a(name, DEVICE_NAME_SIZE as u64);
        h.send(devices, request).result
    };
    assert_eq!(attach(2), audio_result::NOT_SUPPORTED);
    assert_eq!(attach(1), ResultCode::SUCCESS);
    assert_eq!(h.ok(stream, Request::new(2)).pop::<u32>(), 2, "engaged");

    let sysmix = h.audio.sysmix.get(id).unwrap();
    wait_for("the ring to drain", || {
        sysmix.counters().accepted_frames == u64::from(QUEUED)
    });
    let counters = sysmix.counters();
    assert_eq!(counters.nonzero_frames, u64::from(QUEUED));
    assert_eq!(counters.admitted_blocks, 3);
    assert_eq!(counters.padded_frames, 0);
    assert_eq!(counters.faults, 0);
    assert_eq!(
        h.read_u32(ring + 0x44),
        QUEUED,
        "the service moved the read cursor"
    );
    let mut reported = Vec::new();
    h.device
        .take_cpu_writes(|daddr, size| reported.push((daddr, size)));
    assert_eq!(reported, [(DADDR, 0x1000)], "the cursor write was tracked");
    drop(cursors);
    assert!(data.is_signaled(), "the free space reached the threshold");

    assert_eq!(
        h.send(stream, Request::new(4).arg(200.0f32).arg(0u32).arg(0i64))
            .result,
        audio_result::INVALID_ARGUMENT
    );
    assert_eq!(
        h.send(stream, Request::new(4).arg(0.5f32).arg(0u32).arg(1i64))
            .result,
        audio_result::NOT_SUPPORTED
    );
    h.ok(stream, Request::new(4).arg(0.5f32).arg(0u32).arg(0i64));
    assert_eq!(h.ok(stream, Request::new(3)).pop::<f32>(), 0.5);
    h.ok(devices, Request::new(800).arg(id));
    assert_eq!(h.ok(stream, Request::new(2)).pop::<u32>(), 1, "detached");
}

#[test]
fn sysmix_input_captures_silence_and_purges_it() {
    let h = harness();
    let session = user_session(&h);
    let ring = h.heap + RING_OFFSET;
    let mut reply = create_stream(&h, session, 2, 0x2, ring);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let id: u32 = reply.pop();
    let stream = reply.move_handles[0];
    h.write(ring + 0x1040, &[0xff; 240 * 4]);
    assert_eq!(open_stream(&h, stream, 2, 2), ResultCode::SUCCESS);
    let ids = h.client.alloc(&id.to_le_bytes());
    h.ok(session, Request::new(2).buffer_a(ids, 4));

    wait_for("captured frames", || h.read_u32(ring + 0x1004) >= 480);
    assert!(
        h.client
            .read(ring + 0x1040, 240 * 4)
            .iter()
            .all(|&b| b == 0),
        "the host has no microphone, so the ring captures silence"
    );
    let purged: u32 = h.ok(stream, Request::new(11).arg(0u32)).pop();
    assert!(purged >= 480, "purged {purged} frames");
    assert_eq!(
        h.read_u32(ring + 0x44),
        purged,
        "purge moves the guest's read cursor"
    );
}

#[test]
fn renderer_error_paths_use_the_reference_results() {
    let h = harness();
    let manager = h.renderer_manager();
    let params = renderer_params();
    let size = h.work_buffer_size(manager, params);

    let bad_revision = AudioRendererParameterInternal {
        revision: 0x1234_5678,
        ..params
    };
    assert_eq!(
        h.send(manager, Request::new(1).arg(bad_revision)).result,
        audio_result::INVALID_REVISION
    );
    assert_eq!(
        h.open_renderer(manager, bad_revision, size).result,
        audio_result::INVALID_REVISION
    );
    assert_eq!(
        h.open_renderer(manager, params, size / 2).result,
        audio_result::INSUFFICIENT_BUFFER
    );
    let transfer = h.transfer_memory(size);
    let no_process = Request::new(0)
        .arg(params)
        .arg(size)
        .arg(0u64)
        .copy_handle(transfer);
    assert_eq!(
        h.send(manager, no_process).result,
        audio_result::INVALID_HANDLE
    );

    let manual = AudioRendererParameterInternal {
        execution_mode: 1,
        ..params
    };
    let first = h.open_renderer(manager, manual, size);
    assert_eq!(first.result, ResultCode::SUCCESS);
    assert_eq!(
        h.send(first.move_handles[0], Request::new(7)).result,
        audio_result::NOT_SUPPORTED
    );
    assert_eq!(
        h.open_renderer(manager, params, size).result,
        ResultCode::SUCCESS
    );
    assert_eq!(
        h.open_renderer(manager, params, size).result,
        audio_result::OUT_OF_SESSIONS
    );
    let reply = h.request_update(first.move_handles[0], &[0; 0x10], update_output_size());
    assert_eq!(reply.result, audio_result::INVALID_UPDATE_INFO);
}

#[test]
fn sysmix_error_paths_use_the_reference_results() {
    let h = harness();
    assert_eq!(
        h.send(h.connect("audsmx:u"), Request::new(0)).result,
        audio_result::INVALID_HANDLE
    );
    let session = user_session(&h);
    let ring = h.heap + RING_OFFSET;
    assert_eq!(
        create_stream(&h, session, 3, 0x24, ring).result,
        audio_result::INVALID_ARGUMENT
    );
    assert_eq!(
        create_stream(&h, session, 1, 0x2, ring).result,
        audio_result::NOT_SUPPORTED
    );
    assert_eq!(
        create_stream(&h, session, 1, 0x24, ring + 0x10).result,
        audio_result::INVALID_ADDRESS_INFO
    );
    assert_eq!(
        create_stream(&h, session, 1, 0x24, 0x7000_0000_0000).result,
        audio_result::INVALID_ADDRESS_INFO
    );

    let mut reply = create_stream(&h, session, 1, 0x24, ring);
    let id: u32 = reply.pop();
    let stream = reply.move_handles[0];
    let ids = h.client.alloc(&id.to_le_bytes());
    assert_eq!(
        h.send(session, Request::new(2).buffer_a(ids, 4)).result,
        audio_result::OPERATION_FAILED,
        "a stream engages only once open"
    );
    let unknown = h.client.alloc(&99u32.to_le_bytes());
    assert_eq!(
        h.send(session, Request::new(2).buffer_a(unknown, 4)).result,
        audio_result::NOT_FOUND
    );
    assert_eq!(open_stream(&h, stream, 2, 6), audio_result::NOT_SUPPORTED);
    assert_eq!(open_stream(&h, stream, 2, 2), ResultCode::SUCCESS);
    assert_eq!(
        open_stream(&h, stream, 2, 2),
        audio_result::OPERATION_FAILED
    );
    assert_eq!(
        h.send(stream, Request::new(7).arg(1u32).arg(0i32)).result,
        audio_result::NOT_SUPPORTED
    );
    assert_eq!(
        h.send(
            stream,
            Request::new(7).arg(2u32).arg(RING_FRAMES as i32 + 1)
        )
        .result,
        audio_result::INVALID_ARGUMENT
    );
    assert_eq!(
        h.send(stream, Request::new(11).arg(0u32)).result,
        audio_result::OPERATION_FAILED,
        "only input streams purge"
    );
    h.ok(session, Request::new(2).buffer_a(ids, 4));
    let sysmix = h.audio.sysmix.get(id).unwrap();
    assert!(sysmix.is_playing());
    h.write(ring + 0x44, &5u32.to_le_bytes());
    wait_for("the fault", || sysmix.counters().faults == 1);
    assert!(
        !sysmix.is_playing(),
        "a faulted stream stops its host stream"
    );
    assert_eq!(
        h.ok(stream, Request::new(2)).pop::<u32>(),
        0,
        "a moved cursor faults"
    );
}
