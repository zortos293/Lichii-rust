//! The audio services driven through the kernel's host-side IPC client.

use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_audio::sink::result;
use lichii_common::settings::{AudioOutput, Settings};
use lichii_common::ResultCode;
use lichii_kernel::handle_table::CURRENT_PROCESS;
use lichii_kernel::ipc::Handle;
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Reply, Request};
use lichii_kernel::{Event, Kernel};

use super::{Audio, DEVICE_NAME_SIZE};

struct Harness {
    kernel: Kernel,
    audio: Arc<Audio>,
    client: IpcClient,
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.kernel.shutdown();
    }
}

fn harness() -> Harness {
    let mut settings = Settings::default();
    settings.audio.output = AudioOutput::Null;
    let (context, client) = crate::testing::context(crate::testing::SWITCH1_TITLE, settings);
    let audio = super::start(&context).unwrap();
    Harness {
        kernel: context.kernel.clone(),
        audio,
        client,
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

    fn event(&self, handle: Handle) -> Arc<Event> {
        match self.client.object(handle) {
            Some(KObject::ReadableEvent(event)) => event,
            _ => panic!("handle {handle:#x} is not an event"),
        }
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

    /// `OpenAudioOut`/`OpenAudioIn` (`command` on `service`) for `device`: the reply and the
    /// device name the service wrote back.
    fn open(&self, service: &str, command: u32, prefix: Request, device: &str) -> (Reply, String) {
        let name = self.name_buffer(device);
        let out = self.client.alloc(&[0xff; DEVICE_NAME_SIZE]);
        let size = DEVICE_NAME_SIZE as u64;
        let request = prefix
            .copy_handle(CURRENT_PROCESS)
            .buffer_a(name, size)
            .buffer_b(out, size);
        let _ = command;
        let reply = self.send(self.connect(service), request);
        (reply, self.read_name(out))
    }

    fn open_audio_out(&self, device: &str, sample_rate: u32, channels: u32) -> (Reply, String) {
        let request = Request::new(1).arg([sample_rate, channels]).arg(0u64);
        self.open("audout:u", 1, request, device)
    }

    /// An `AudioOutBuffer` for `samples` (copied to guest memory): its address, which the tests
    /// use as the tag, and the samples' address.
    fn audio_buffer(&self, samples: &[i16]) -> (u64, u64) {
        let data = self.client.alloc(bytemuck::cast_slice(samples));
        let size = std::mem::size_of_val(samples) as u64;
        let header: [u64; 5] = [0, data, size, size, 0];
        (self.client.alloc(bytemuck::cast_slice(&header)), data)
    }

    fn read_u64s(&self, address: u64, count: usize) -> Vec<u64> {
        bytemuck::pod_collect_to_vec(&self.client.read(address, count * 8))
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

#[test]
fn audout_lists_the_default_device() {
    let h = harness();
    let audout = h.connect("audout:u");
    let out = h.client.alloc(&[0; DEVICE_NAME_SIZE]);
    let mut reply = h.ok(
        audout,
        Request::new(0).buffer_b(out, DEVICE_NAME_SIZE as u64),
    );
    assert_eq!(reply.pop::<u32>(), 1);
    assert_eq!(h.read_name(out), "DeviceOut");
    let mut reply = h.ok(audout, Request::new(2).buffer_c(out, 0x10));
    assert_eq!(reply.pop::<u32>(), 0);
}

#[test]
fn audout_rejects_bad_configurations() {
    let h = harness();
    assert_eq!(
        h.open_audio_out("Speaker", 48_000, 2).0.result,
        result::NOT_FOUND
    );
    assert_eq!(
        h.open_audio_out("", 44_100, 2).0.result,
        result::INVALID_SAMPLE_RATE
    );
    assert_eq!(
        h.open_audio_out("", 48_000, 4).0.result,
        result::INVALID_CHANNEL_COUNT
    );
    let no_buffers = Request::new(1)
        .arg([48_000u32, 2])
        .arg(0u64)
        .copy_handle(CURRENT_PROCESS);
    assert_eq!(
        h.send(h.connect("audout:u"), no_buffers).result,
        result::UNKNOWN
    );
    let mut sessions = Vec::new();
    for _ in 0..lichii_audio::out::MAX_SESSIONS {
        let (reply, _) = h.open_audio_out("", 0, 0);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        sessions.push(reply);
    }
    assert_eq!(h.audio.out.free_sessions(), 0);
    assert_eq!(h.open_audio_out("", 0, 0).0.result, result::OUT_OF_SESSIONS);
}

#[test]
fn audout_plays_buffers_and_releases_them_in_real_time() {
    let h = harness();
    let (mut reply, name) = h.open_audio_out("", 0, 0);
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(name, "DeviceOut");
    assert_eq!(reply.pop::<[u32; 4]>(), [48_000, 2, 2, 1]);
    let out = reply.move_handles[0];

    let event = h.event(h.ok(out, Request::new(4)).copy_handles[0]);
    let samples = vec![1000i16; 480 * 2];
    let (first, _) = h.audio_buffer(&samples);
    let (second, _) = h.audio_buffer(&samples);
    h.ok(out, Request::new(3).arg(first).buffer_a(first, 0x28));
    h.ok(out, Request::new(7).arg(second).buffer_x(second, 0x28));
    assert_eq!(h.ok(out, Request::new(9)).pop::<u32>(), 2);
    assert_eq!(h.ok(out, Request::new(6).arg(first)).pop::<u8>(), 1);
    assert_eq!(h.ok(out, Request::new(6).arg(0x1234u64)).pop::<u8>(), 0);
    assert_eq!(h.ok(out, Request::new(10)).pop::<u64>(), 0);

    h.ok(out, Request::new(12).arg(0.5f32));
    assert_eq!(h.ok(out, Request::new(13)).pop::<f32>(), 0.5);
    h.ok(out, Request::new(1));
    assert_eq!(h.ok(out, Request::new(0)).pop::<u32>(), 0);
    assert_eq!(
        h.send(out, Request::new(1)).result,
        result::OPERATION_FAILED
    );

    let tags = h.client.alloc(&[0; 8 * 4]);
    let mut released = Vec::new();
    wait_for("both buffers to be released", || {
        if event.reset() {
            let mut reply = h.ok(out, Request::new(5).buffer_b(tags, 8 * 4));
            let count = reply.pop::<u32>() as usize;
            released.extend_from_slice(&h.read_u64s(tags, count));
        }
        released.len() == 2
    });
    assert_eq!(released, [first, second]);
    assert_eq!(h.ok(out, Request::new(9)).pop::<u32>(), 0);

    let (long, _) = h.audio_buffer(&vec![0i16; 4800 * 2]);
    let appended = Instant::now();
    h.ok(out, Request::new(3).arg(long).buffer_a(long, 0x28));
    wait_for("the 100 ms buffer", || {
        h.ok(out, Request::new(5).buffer_b(tags, 8)).pop::<u32>() == 1
    });
    let elapsed = appended.elapsed();
    assert!(
        elapsed >= Duration::from_millis(40),
        "100 ms released after {elapsed:?}"
    );
    assert!(h.ok(out, Request::new(10)).pop::<u64>() >= 960 + 4800);

    let (third, _) = h.audio_buffer(&vec![0i16; 48_000 * 2]);
    h.ok(out, Request::new(3).arg(third).buffer_a(third, 0x28));
    assert_eq!(h.ok(out, Request::new(11)).pop::<u8>(), 1);
    let mut reply = h.ok(out, Request::new(8).buffer_c(tags, 8 * 4));
    assert_eq!(reply.pop::<u32>(), 1);
    assert_eq!(h.read_u64s(tags, 1), [third]);

    h.ok(out, Request::new(2));
    assert_eq!(h.ok(out, Request::new(0)).pop::<u32>(), 1);
    assert_eq!(h.ok(out, Request::new(11)).pop::<u8>(), 0);
    let tiny = h.client.alloc(&[0; 8]);
    let reply = h.send(out, Request::new(3).arg(1u64).buffer_a(tiny, 8));
    assert_eq!(reply.result, result::INSUFFICIENT_BUFFER);
}

#[test]
fn audout_reports_a_full_buffer_ring() {
    let h = harness();
    let (mut reply, _) = h.open_audio_out("DeviceOut", 48_000, 6);
    assert_eq!(reply.pop::<[u32; 4]>()[1], 6);
    let out = reply.move_handles[0];
    let (buffer, _) = h.audio_buffer(&[0; 6]);
    for tag in 1..=32u64 {
        h.ok(out, Request::new(3).arg(tag).buffer_a(buffer, 0x28));
    }
    let reply = h.send(out, Request::new(3).arg(99u64).buffer_a(buffer, 0x28));
    assert_eq!(reply.result, result::BUFFER_COUNT_REACHED);
    assert_eq!(h.ok(out, Request::new(9)).pop::<u32>(), 32);
}

#[test]
fn audin_records_silence() {
    let h = harness();
    let audin = h.connect("audin:u");
    let names = h.client.alloc(&[0; DEVICE_NAME_SIZE]);
    let mut reply = h.ok(
        audin,
        Request::new(4).buffer_c(names, DEVICE_NAME_SIZE as u64),
    );
    assert_eq!(reply.pop::<u32>(), 1);
    assert_eq!(h.read_name(names), "Uac");

    let request = Request::new(1).arg([48_000u32, 2]).arg(0u64);
    let (mut reply, name) = h.open("audin:u", 1, request, "");
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.pop::<[u32; 4]>(), [48_000, 2, 2, 1]);
    assert_eq!(name, "DeviceIn");
    let input = reply.move_handles[0];

    let event = h.event(h.ok(input, Request::new(4)).copy_handles[0]);
    let (buffer, data) = h.audio_buffer(&[0x5555; 480]);
    h.ok(input, Request::new(3).arg(buffer).buffer_a(buffer, 0x28));
    h.ok(input, Request::new(12).arg(0.25f32));
    assert_eq!(h.ok(input, Request::new(13)).pop::<f32>(), 0.25);
    assert_eq!(h.ok(input, Request::new(11)).pop::<u32>(), 1);
    h.ok(input, Request::new(1));
    let tags = h.client.alloc(&[0; 8]);
    wait_for("the capture buffer", || {
        event.reset() && h.ok(input, Request::new(5).buffer_b(tags, 8)).pop::<u32>() == 1
    });
    assert_eq!(h.read_u64s(tags, 1), [buffer]);
    assert!(h.client.read(data, 960).iter().all(|&b| b == 0));

    let request = Request::new(5).arg([1u32, 0]).arg([0u32, 2]).arg(0u64);
    let (reply, name) = h.open("audin:u", 5, request, "Uac");
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(name, "Uac");
    let (reply, name) = h.open(
        "audin:u",
        3,
        Request::new(3).arg([0u32, 2]).arg(0u64),
        "Uac",
    );
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(name, "UacIn");
    let request = Request::new(1).arg([48_000u32, 2]).arg(0u64);
    assert_eq!(
        h.open("audin:u", 1, request, "Mic").0.result,
        result::NOT_FOUND
    );
}

#[test]
fn audctl_target_volume_drives_the_sink() {
    let h = harness();
    let audctl = h.connect("audctl");
    let sink = &h.audio.sink;
    assert_eq!(h.ok(audctl, Request::new(2)).pop::<i32>(), 0);
    assert_eq!(h.ok(audctl, Request::new(3)).pop::<i32>(), 15);
    assert_eq!(h.ok(audctl, Request::new(32)).pop::<u32>(), 2);
    assert_eq!(h.ok(audctl, Request::new(0).arg(2u32)).pop::<i32>(), 15);
    let reply = h.send(audctl, Request::new(0).arg(6u32));
    assert_eq!(reply.result, result::INVALID_ARGUMENT);

    h.ok(audctl, Request::new(1).arg(2u32).arg(30i32));
    assert_eq!(h.ok(audctl, Request::new(0).arg(2u32)).pop::<i32>(), 15);
    h.ok(audctl, Request::new(1).arg(2u32).arg(5i32));
    assert_eq!(sink.device_volume(), 5.0 / 15.0);
    h.ok(audctl, Request::new(1).arg(3u32).arg(1i32));
    assert_eq!(sink.device_volume(), 5.0 / 15.0);

    h.ok(audctl, Request::new(5).arg(1u8).arg(2u32));
    assert_eq!(h.ok(audctl, Request::new(4).arg(2u32)).pop::<u8>(), 1);
    assert_eq!(sink.device_volume(), 0.0);
    h.ok(audctl, Request::new(5).arg(0u8).arg(2u32));
    assert_eq!(sink.device_volume(), 5.0 / 15.0);

    assert_eq!(h.ok(audctl, Request::new(9).arg(1u32)).pop::<u32>(), 1);
    h.ok(audctl, Request::new(10).arg(1u32).arg(2u32));
    assert_eq!(h.ok(audctl, Request::new(9).arg(1u32)).pop::<u32>(), 2);
    assert_eq!(h.ok(audctl, Request::new(13).arg(2u32)).pop::<u32>(), 3);
    assert_eq!(h.ok(audctl, Request::new(12)).pop::<u32>(), 0);
    h.ok(audctl, Request::new(30).arg(1u8));
    assert_eq!(h.ok(audctl, Request::new(31)).pop::<u8>(), 1);
    h.event(h.ok(audctl, Request::new(34)).copy_handles[0]);

    let clone = h.ok(audctl, Request::new(5000)).move_handles[0];
    assert_eq!(h.ok(clone, Request::new(0).arg(2u32)).pop::<i32>(), 5);
    assert_eq!(h.ok(h.connect("audctl"), Request::new(31)).pop::<u8>(), 1);
}

const HWOPUS: u32 = 111;

/// 20 ms frames of a 440 Hz stereo tone at 48 kHz, each Opus-encoded and wrapped in the
/// big-endian size and final range header hwopus expects.
fn opus_packets(frames: usize) -> (Vec<Vec<i16>>, Vec<Vec<u8>>) {
    let mut encoder = opus1::OpusEncoder::new(
        opus1::SampleRate::Hz48000,
        opus1::Channels::Stereo,
        opus1::Application::Audio,
    )
    .unwrap();
    let mut pcm = Vec::new();
    let mut packets = Vec::new();
    for frame in 0..frames {
        let samples: Vec<i16> = (0..960 * 2)
            .map(|i| {
                let t = (frame * 960 + i / 2) as f32 / 48_000.0;
                ((t * 440.0 * std::f32::consts::TAU).sin() * 12_000.0) as i16
            })
            .collect();
        let mut packet = vec![0u8; 1500];
        let len = encoder.encode_i16(&samples, 960, &mut packet).unwrap();
        let mut framed = (len as u32).to_be_bytes().to_vec();
        framed.extend_from_slice(&encoder.final_range().to_be_bytes());
        framed.extend_from_slice(&packet[..len]);
        pcm.push(samples);
        packets.push(framed);
    }
    (pcm, packets)
}

fn rms(samples: &[i16]) -> f64 {
    (samples.iter().map(|&s| f64::from(s).powi(2)).sum::<f64>() / samples.len() as f64).sqrt()
}

#[test]
fn hwopus_reports_eden_work_buffer_sizes() {
    let h = harness();
    let hwopus = h.connect("hwopus");
    assert_eq!(
        h.ok(hwopus, Request::new(1).arg([48_000u32, 2]))
            .pop::<u32>(),
        32636
    );
    let reply = h.send(hwopus, Request::new(1).arg([44_100u32, 2]));
    assert_eq!(reply.result, ResultCode::new(HWOPUS, 1001));
    let ex = Request::new(8).arg([48_000u32, 1, 1, 0]);
    assert_eq!(h.ok(hwopus, ex).pop::<u32>(), 24 + 18468 + 5760 + 0x600);
    let mut params = vec![0u8; 0x110];
    params[..16].copy_from_slice(bytemuck::cast_slice(&[48_000u32, 2, 1, 1]));
    let address = h.client.alloc(&params);
    let reply = h.ok(hwopus, Request::new(3).buffer_x(address, 0x110));
    assert_eq!(reply.clone().pop::<u32>(), 24 + 272 + 27240 + 1536 + 3840);
}

#[test]
fn hwopus_decodes_a_synthetic_packet() {
    let h = harness();
    let hwopus = h.connect("hwopus");
    let open = h.ok(hwopus, Request::new(0).arg([48_000u32, 2]).arg(0x8000u32));
    let decoder = open.move_handles[0];
    let (pcm, packets) = opus_packets(3);
    let output = h.client.alloc(&[0; 960 * 4]);
    let mut last = Vec::new();
    for (index, packet) in packets.iter().enumerate() {
        let input = h.client.alloc(packet);
        let command = if index == 0 { 0 } else { 4 };
        let mut reply = h.ok(
            decoder,
            Request::new(command)
                .buffer_a(input, packet.len() as u64)
                .buffer_b(output, 960 * 4),
        );
        assert_eq!(reply.pop::<u32>(), packet.len() as u32);
        assert_eq!(reply.pop::<u32>(), 960);
        if command == 4 {
            reply.pop::<u64>();
        }
        last = bytemuck::pod_collect_to_vec(&h.client.read(output, 960 * 4));
    }
    let (expected, decoded) = (rms(&pcm[2]), rms(&last));
    assert!(
        (decoded - expected).abs() < expected * 0.25,
        "decoded rms {decoded}, source {expected}"
    );

    let input = h.client.alloc(&packets[0]);
    let size = packets[0].len() as u64;
    let mut reply = h.ok(
        decoder,
        Request::new(8)
            .arg(1u8)
            .buffer_a(input, size)
            .buffer_b(output, 960 * 4),
    );
    assert_eq!((reply.pop::<u32>(), reply.pop::<u32>()), (size as u32, 960));
    let reply = h.send(
        decoder,
        Request::new(2)
            .buffer_a(input, size)
            .buffer_b(output, 960 * 4),
    );
    assert_eq!(reply.result, ResultCode::new(HWOPUS, 6));
    let reply = h.send(
        decoder,
        Request::new(0).buffer_a(input, 8).buffer_b(output, 960 * 4),
    );
    assert_eq!(reply.result, ResultCode::new(HWOPUS, 8));
    let reply = h.send(
        decoder,
        Request::new(0)
            .buffer_a(input, 20)
            .buffer_b(output, 960 * 4),
    );
    assert_eq!(reply.result, ResultCode::new(HWOPUS, 3));
    h.ok(decoder, Request::new(1).buffer_a(input, 0));
}

#[test]
fn hwopus_multistream_decoder_plays_a_single_coupled_stream() {
    let h = harness();
    let hwopus = h.connect("hwopus");
    let mut params = vec![0u8; 0x118];
    params[..16].copy_from_slice(bytemuck::cast_slice(&[48_000u32, 2, 1, 1]));
    params[0x18..0x1a].copy_from_slice(&[0, 1]);
    let address = h.client.alloc(&params);
    let open = h.ok(
        hwopus,
        Request::new(6).arg(0x10000u32).buffer_x(address, 0x118),
    );
    let decoder = open.move_handles[0];
    let (_, packets) = opus_packets(1);
    let input = h.client.alloc(&packets[0]);
    let output = h.client.alloc(&[0; 960 * 4]);
    let mut reply = h.ok(
        decoder,
        Request::new(9)
            .arg(0u8)
            .buffer_a(input, packets[0].len() as u64)
            .buffer_b(output, 960 * 4),
    );
    assert_eq!(reply.pop::<u32>(), packets[0].len() as u32);
    assert_eq!(reply.pop::<u32>(), 960);
    let size = packets[0].len() as u64;
    let reply = h.send(
        decoder,
        Request::new(0)
            .buffer_a(input, size)
            .buffer_b(output, 960 * 4),
    );
    assert_eq!(reply.result, ResultCode::new(HWOPUS, 6));
}

#[test]
fn hwopus_holds_at_most_24_decoders() {
    let h = harness();
    let hwopus = h.connect("hwopus");
    let mut decoders = Vec::new();
    for _ in 0..24 {
        decoders.push(h.ok(hwopus, Request::new(0).arg([24_000u32, 1]).arg(0u32)));
    }
    let reply = h.send(hwopus, Request::new(0).arg([24_000u32, 1]).arg(0u32));
    assert_eq!(reply.result, ResultCode::new(HWOPUS, 385));
    let reply = h.send(hwopus, Request::new(4).arg([48_000u32, 3, 0, 0]).arg(0u32));
    assert_eq!(reply.result, ResultCode::new(HWOPUS, 385));
}
