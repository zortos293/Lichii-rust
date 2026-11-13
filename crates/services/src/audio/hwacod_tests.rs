//! The batch decoder driven through the kernel's host-side IPC client, with a layout built the
//! way a Switch 2 title lays it out.

use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::handle_table::CURRENT_PROCESS;
use lichii_kernel::ipc::Handle;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Request};
use lichii_kernel::{Event, Kernel};
use lichii_memory::MemoryPermission;

use super::*;

const IO_COUNT: u32 = 4;
const INPUT_STRIDE: u32 = 0x600;
/// One 20 ms stereo frame at 48 kHz.
const OUTPUT_STRIDE: u32 = 960 * 2 * 2;
const INSTANCE_STRIDE: u32 = 0x6c00;
const TRANSFER_SIZE: u64 = 0x10000;
const PROCESS_SIZE: u64 = 0x8000;
const COMPLETION_OFFSET: u64 = 0x6000;

struct Harness {
    kernel: Kernel,
    client: IpcClient,
    transfer: Handle,
    process_address: u64,
}

impl Drop for Harness {
    fn drop(&mut self) {
        self.kernel.shutdown();
    }
}

impl Harness {
    fn new() -> Self {
        let (context, client) =
            crate::testing::context(crate::testing::SWITCH2_TITLE, Settings::default());
        super::super::start(&context).unwrap();
        let process = client.process().clone();
        let heap = process.page_table().set_heap_size(2 << 20).unwrap();
        let transfer = TransferMemory::new(
            process.clone(),
            heap,
            TRANSFER_SIZE,
            MemoryPermission::READ_WRITE,
        )
        .unwrap();
        let transfer = process
            .handles()
            .write()
            .add(KObject::TransferMemory(Arc::new(transfer)))
            .unwrap();
        Self {
            kernel: context.kernel.clone(),
            client,
            transfer,
            process_address: heap + TRANSFER_SIZE,
        }
    }

    fn send(&self, session: Handle, request: Request) -> lichii_kernel::testing::Reply {
        self.client.send(session, &request).unwrap()
    }

    fn ok(&self, session: Handle, request: Request) -> lichii_kernel::testing::Reply {
        let reply = self.send(session, request);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        reply
    }

    fn client_session(&self) -> Handle {
        let hwopus = self.client.connect("hwopus").unwrap();
        self.ok(hwopus, Request::new(100)).move_handles[0]
    }

    fn write(&self, address: u64, bytes: &[u8]) {
        self.client
            .process()
            .memory()
            .write_bytes(address, bytes)
            .unwrap();
    }

    fn word(&self, offset: u64) -> u32 {
        self.client
            .process()
            .memory()
            .read(self.process_address + offset)
            .unwrap()
    }

    fn open_request(&self, layout: SharedMemoryWire) -> Request {
        Request::new(10)
            .arg(layout)
            .arg(TRANSFER_SIZE)
            .arg(self.process_address)
            .arg(PROCESS_SIZE)
            .arg(1u64)
            .copy_handle(self.transfer)
            .copy_handle(CURRENT_PROCESS)
    }
}

fn layout() -> SharedMemoryWire {
    SharedMemoryWire {
        version: 1,
        manager_size: 0x80,
        manager_offset: 0,
        instances_size: 2 * INSTANCE_STRIDE,
        instances_offset: 0x100,
        transfer_queue_size: 0x1000,
        transfer_queue_offset: 0xe000,
        transfer_size: TRANSFER_SIZE as u32,
        io_size: IO_COUNT * (INPUT_STRIDE + OUTPUT_STRIDE),
        io_offset: 0,
        completion_size: 0x1000,
        completion_offset: COMPLETION_OFFSET as u32,
        process_size: PROCESS_SIZE as u32,
        instance_stride: INSTANCE_STRIDE,
        instance_count: 2,
        input_stride: INPUT_STRIDE,
        output_stride: OUTPUT_STRIDE,
        io_count: IO_COUNT,
        reserved: [0; 6],
    }
}

/// The 0x120-byte codec parameter block.
fn parameters(kind: u64, rate: i32, channels: i32, duration_ns: i64) -> Vec<u8> {
    let mut bytes = vec![0u8; PARAMETER_SIZE as usize];
    bytes[..8].copy_from_slice(&kind.to_le_bytes());
    bytes[8..12].copy_from_slice(&rate.to_le_bytes());
    bytes[12..16].copy_from_slice(&channels.to_le_bytes());
    bytes[16..24].copy_from_slice(&duration_ns.to_le_bytes());
    bytes
}

/// A 20 ms stereo sine frame, framed with the big-endian size and final range header.
fn framed_packet() -> Vec<u8> {
    let mut encoder = opus1::OpusEncoder::new(
        opus1::SampleRate::Hz48000,
        opus1::Channels::Stereo,
        opus1::Application::Audio,
    )
    .unwrap();
    let pcm: Vec<i16> = (0..960 * 2)
        .map(|i| ((i / 2) as f32 / 48_000.0 * 440.0 * std::f32::consts::TAU).sin() * 12_000.0)
        .map(|s| s as i16)
        .collect();
    let mut packet = vec![0u8; 1500];
    let len = encoder.encode_i16(&pcm, 960, &mut packet).unwrap();
    let mut framed = (len as u32).to_be_bytes().to_vec();
    framed.extend_from_slice(&encoder.final_range().to_be_bytes());
    framed.extend_from_slice(&packet[..len]);
    framed
}

fn task(tag: u16, io: u16, sequence: u16, input_bytes: usize) -> Task {
    Task {
        tag,
        instance: 0,
        io,
        sequence,
        input_bytes: input_bytes as u32,
        reset: 0,
        notify: 1,
        reserved: 0,
    }
}

fn tasks(h: &Harness, tasks: &[Task]) -> Request {
    let address = h.client.alloc(bytemuck::cast_slice(tasks));
    Request::new(10).buffer_a(address, size_of_val(tasks) as u64)
}

#[test]
fn sizes_match_the_reference() {
    let h = Harness::new();
    let session = h.client_session();
    let params = h.client.alloc(&parameters(1, 48_000, 2, 20_000_000));
    let mut reply = h.ok(session, Request::new(0).buffer_x(params, PARAMETER_SIZE));
    assert_eq!(
        reply.pop::<u32>(),
        0x140 + (24u32 + 27236).next_multiple_of(64)
    );
    let mut reply = h.ok(session, Request::new(1).arg(IO_COUNT));
    assert_eq!(reply.pop::<u32>(), 128);

    let cases = [
        (parameters(0, 48_000, 2, 20_000_000), NOT_SUPPORTED),
        (
            parameters(1, 44_100, 2, 20_000_000),
            INVALID_OPUS_SAMPLE_RATE,
        ),
        (
            parameters(1, 48_000, 3, 20_000_000),
            INVALID_OPUS_CHANNEL_COUNT,
        ),
        (parameters(1, 48_000, 2, 15_000_000), INVALID_ARGUMENT),
    ];
    for (bytes, expected) in cases {
        let params = h.client.alloc(&bytes);
        let reply = h.send(session, Request::new(0).buffer_x(params, PARAMETER_SIZE));
        assert_eq!(reply.result, expected);
    }
    assert_eq!(
        h.send(session, Request::new(1).arg(0u32)).result,
        INVALID_ARGUMENT
    );
    let short = h.client.alloc(&[0; 0x100]);
    assert_eq!(
        h.send(session, Request::new(0).buffer_x(short, 0x100))
            .result,
        INVALID_ARGUMENT
    );
}

#[test]
fn bad_layouts_are_refused() {
    let h = Harness::new();
    let session = h.client_session();
    let cases = [
        (
            SharedMemoryWire {
                version: 2,
                ..layout()
            },
            INVALID_REVISION,
        ),
        (
            SharedMemoryWire {
                input_stride: 0x400,
                ..layout()
            },
            INVALID_ARGUMENT,
        ),
        (
            SharedMemoryWire {
                io_size: 0x100,
                ..layout()
            },
            BUFFER_TOO_SMALL,
        ),
        (
            SharedMemoryWire {
                instances_offset: 0x40,
                ..layout()
            },
            INVALID_ADDRESS_INFO,
        ),
        (
            SharedMemoryWire {
                completion_offset: 0x1000,
                ..layout()
            },
            INVALID_ADDRESS_INFO,
        ),
    ];
    for (layout, expected) in cases {
        assert_eq!(h.send(session, h.open_request(layout)).result, expected);
    }
}

#[test]
fn execute_decodes_into_the_io_slot_and_publishes_a_completion() {
    let h = Harness::new();
    let session = h.client_session();
    let mut open = h.ok(session, h.open_request(layout()));
    let tag = open.pop::<u32>() as u16;
    assert_ne!(tag, 0);
    let event = match h.client.object(open.copy_handles[0]) {
        Some(KObject::ReadableEvent(event)) => event,
        _ => panic!("OpenManager returns the completion event"),
    };
    let manager = open.move_handles[0];
    let params = h.client.alloc(&parameters(1, 48_000, 2, 20_000_000));
    h.ok(
        manager,
        Request::new(0).arg(0u32).buffer_x(params, PARAMETER_SIZE),
    );

    let packet = framed_packet();
    h.write(h.process_address, &packet);
    h.ok(manager, tasks(&h, &[task(tag, 0, 7, packet.len())]));

    assert_eq!(h.word(COMPLETION_OFFSET + 4), 1, "the producer index moved");
    let entry = h
        .client
        .read(h.process_address + COMPLETION_OFFSET + 0x80, 0x40);
    let completion: Completion = bytemuck::pod_read_unaligned(&entry);
    assert_eq!(
        (completion.tag, completion.io, completion.sequence),
        (tag, 0, 7)
    );
    assert_eq!(completion.result, ResultCode::SUCCESS.0);
    assert_eq!(completion.output_bytes, OUTPUT_STRIDE);
    assert!(Event::is_signaled(&event));

    let mut reference = OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();
    let mut expected = vec![0i16; 960 * 2];
    reference
        .decode_i16(&packet[8..], 960, &mut expected)
        .unwrap();
    let output = h.client.read(
        h.process_address + u64::from(INPUT_STRIDE),
        OUTPUT_STRIDE as usize,
    );
    assert_eq!(
        bytemuck::cast_slice::<u8, i16>(&output),
        expected.as_slice()
    );
    assert!(expected.iter().any(|&s| s.unsigned_abs() > 1000));

    let reused = h.send(manager, tasks(&h, &[task(tag, 0, 8, packet.len())]));
    assert_eq!(
        reused.result, BUFFER_COUNT_REACHED,
        "the slot is held until it is read"
    );
    assert_eq!(
        h.send(manager, Request::new(1).arg(0u32)).result,
        BUFFER_COUNT_REACHED
    );

    h.write(
        h.process_address + COMPLETION_OFFSET + 0x44,
        &1u32.to_le_bytes(),
    );
    h.ok(manager, tasks(&h, &[task(tag, 0, 8, packet.len())]));
    assert_eq!(h.word(COMPLETION_OFFSET + 4), 2);
    h.write(
        h.process_address + COMPLETION_OFFSET + 0x44,
        &2u32.to_le_bytes(),
    );
    h.ok(manager, Request::new(1).arg(0u32));
    assert_eq!(
        h.send(manager, tasks(&h, &[task(tag, 1, 9, packet.len())]))
            .result,
        INVALID_ARGUMENT,
        "a finalized instance takes no tasks"
    );
}

#[test]
fn a_bad_packet_completes_with_its_error() {
    let h = Harness::new();
    let session = h.client_session();
    let mut open = h.ok(session, h.open_request(layout()));
    let tag = open.pop::<u32>() as u16;
    let manager = open.move_handles[0];
    let params = h.client.alloc(&parameters(1, 48_000, 2, 20_000_000));
    h.ok(
        manager,
        Request::new(0).arg(0u32).buffer_x(params, PARAMETER_SIZE),
    );
    h.write(h.process_address, &[0; 16]);
    h.ok(manager, tasks(&h, &[task(tag, 0, 1, 16)]));
    let entry = h
        .client
        .read(h.process_address + COMPLETION_OFFSET + 0x80, 0x40);
    let completion: Completion = bytemuck::pod_read_unaligned(&entry);
    assert_eq!(completion.result, LIB_OPUS_INVALID_PACKET.0);
    assert_eq!(completion.output_bytes, 0);

    let wrong_tag = h.send(manager, tasks(&h, &[task(tag.wrapping_add(1), 1, 2, 16)]));
    assert_eq!(wrong_tag.result, INVALID_ARGUMENT);
}

#[test]
fn tags_are_unique_while_live_and_reused_after_release() {
    let tags = Arc::new(CodecTags::new());
    let first = tags.acquire().unwrap();
    let second = tags.acquire().unwrap();
    assert_eq!((first.1, second.1), (1, 2));
    drop(first);
    let held: Vec<_> = (0..u16::MAX - 1).map(|_| tags.acquire().unwrap()).collect();
    assert!(held.iter().all(|t| t.1 != 0 && t.1 != 2));
    assert_eq!(
        held.last().unwrap().1,
        1,
        "the search wraps past the last tag"
    );
    assert_eq!(tags.acquire().err(), Some(OUT_OF_SESSIONS));
    drop(held);
    assert!(tags.acquire().is_ok());
}
