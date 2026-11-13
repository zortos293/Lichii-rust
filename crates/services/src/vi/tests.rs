use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::ipc::Handle;
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Request};

use super::parcel::{ParcelReader, ParcelWriter};
use crate::ServiceContext;
use crate::testing::{SWITCH1_TITLE, context};

const CURRENT_PROCESS: Handle = 0xffff_8001;
const REPLY_SIZE: u64 = 0x400;

fn display_name(name: &str) -> [u8; 0x40] {
    let mut out = [0; 0x40];
    out[..name.len()].copy_from_slice(name.as_bytes());
    out
}

struct Title {
    context: Arc<ServiceContext>,
    client: IpcClient,
    display: Handle,
    relay: Handle,
    aruid: u64,
}

impl Title {
    fn boot() -> Self {
        let (context, client) = context(SWITCH1_TITLE, Settings::default());
        let nvdrv = crate::nvdrv::register(&context).unwrap();
        super::register(&context, nvdrv).unwrap();
        crate::am::register(&context).unwrap();
        let vi = client.connect("vi:u").unwrap();
        let display = client
            .send(vi, &Request::new(0).arg(0u32))
            .unwrap()
            .move_handles[0];
        let relay = client
            .send(display, &Request::new(100))
            .unwrap()
            .move_handles[0];
        let aruid = context.process.id();
        Self {
            context,
            client,
            display,
            relay,
            aruid,
        }
    }

    /// The layer am's ISelfController::CreateManagedDisplayLayer makes.
    fn managed_layer(&self) -> u64 {
        let client = &self.client;
        let service = client.connect("appletOE").unwrap();
        let open = Request::new(0)
            .send_pid()
            .arg(0u64)
            .copy_handle(CURRENT_PROCESS);
        let proxy = client.send(service, &open).unwrap().move_handles[0];
        let controller = client.send(proxy, &Request::new(1)).unwrap().move_handles[0];
        let mut reply = client.send(controller, &Request::new(40)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        reply.pop()
    }

    /// Opens `layer` and returns its producer's binder id from the native window parcel.
    fn open_layer(&self, layer: u64, aruid: u64) -> Result<i32, ResultCode> {
        let out = self.client.alloc(&[0; 0x100]);
        let request = Request::new(2020)
            .arg(display_name("Default"))
            .arg(layer)
            .arg(aruid)
            .send_pid()
            .buffer_b(out, 0x100);
        let mut reply = self.client.send(self.display, &request).unwrap();
        if reply.result != ResultCode::SUCCESS {
            return Err(reply.result);
        }
        let size: u64 = reply.pop();
        let window = self.client.read(out, size as usize);
        let data: &[u8] = &window[16..];
        assert_eq!(&data[0..8], &[2, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&data[24..32], b"dispdrv\0");
        Ok(i32::from_le_bytes(data[8..12].try_into().unwrap()))
    }

    /// Runs binder transaction `code` with `payload` and returns the reply parcel's data.
    fn transact(&self, binder: i32, code: u32, payload: impl FnOnce(&mut ParcelWriter)) -> Vec<u8> {
        let mut parcel = ParcelWriter::default();
        parcel.write(0x100u32);
        parcel.write(1u32);
        parcel.write([u16::from(b'a'), 0]);
        payload(&mut parcel);
        let input = parcel.serialize();
        let address = self.client.alloc(&input);
        let out = self.client.alloc(&[0; REPLY_SIZE as usize]);
        let request = Request::new(0)
            .arg(binder)
            .arg(code)
            .arg(0u32)
            .buffer_a(address, input.len() as u64)
            .buffer_b(out, REPLY_SIZE);
        let reply = self.client.send(self.relay, &request).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let reply = self.client.read(out, REPLY_SIZE as usize);
        let (size, offset) = (word(&reply, 0) as usize, word(&reply, 4) as usize);
        reply[offset..offset + size].to_vec()
    }
}

fn word(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn status(reply: &[u8]) -> i32 {
    word(reply, reply.len() - 4) as i32
}

fn graphic_buffer(width: u32, height: u32) -> [u32; 91] {
    let mut buffer = [0u32; 91];
    buffer[1] = width;
    buffer[2] = height;
    buffer[3] = width;
    buffer[4] = 1;
    buffer
}

fn connect_with_buffers(title: &Title, binder: i32, count: u32) {
    let reply = title.transact(binder, 10, |p| {
        p.write(0u32);
        p.write(2i32);
        p.write(0u32);
    });
    assert_eq!(status(&reply), 0, "connect");
    for slot in 0..count {
        let reply = title.transact(binder, 14, |p| {
            p.write(slot as i32);
            p.write(1u32);
            p.write(0x16ci64);
            p.write(graphic_buffer(1280, 720));
        });
        assert_eq!(status(&reply), 0, "preallocate {slot}");
    }
}

/// DequeueBuffer: the slot and its status.
fn dequeue(title: &Title, binder: i32) -> (i32, i32) {
    let reply = title.transact(binder, 3, |p| {
        p.write(0u32);
        p.write([1280u32, 720, 1, 0xb00]);
    });
    (word(&reply, 0) as i32, status(&reply))
}

/// QueueBuffer with swap interval 1 and no fence; returns the pending count.
fn queue(title: &Title, binder: i32, slot: i32) -> u32 {
    let reply = title.transact(binder, 7, |p| {
        p.write(slot);
        p.write(84i64);
        let mut input = [0u32; 21];
        input[7] = 1;
        input[11] = 1;
        input[12] = 0;
        input[13..21].copy_from_slice(&[u32::MAX, 0, u32::MAX, 0, u32::MAX, 0, u32::MAX, 0]);
        p.write(input);
    });
    assert_eq!(status(&reply), 0, "queue slot {slot}");
    word(&reply, 12)
}

#[test]
fn parcels_round_trip_with_their_header_and_token() {
    let mut parcel = ParcelWriter::default();
    parcel.write(0u32);
    parcel.write(3u32);
    parcel.write([u16::from(b'a'), u16::from(b'b'), u16::from(b'c'), 0]);
    parcel.write(7u8);
    parcel.write(-5i32);
    parcel.write_object(Some(&0x1234_5678u32));
    parcel.write_object::<u32>(None);
    let bytes = parcel.serialize();
    assert_eq!(word(&bytes, 4), 16);
    let mut reader = ParcelReader::new(&bytes);
    assert_eq!(reader.read::<u8>(), 7);
    assert_eq!(reader.read::<i32>(), -5);
    assert_eq!(reader.read_object::<u32>(), Some(0x1234_5678));
    assert_eq!(reader.read_object::<u32>(), None);
    assert_eq!(reader.read::<u32>(), 0, "reads past the end are zero");
}

#[test]
fn display_service_opens_displays_and_hands_out_one_vsync_event_each() {
    let title = Title::boot();
    let client = &title.client;
    let mut reply = client
        .send(
            title.display,
            &Request::new(1010).arg(display_name("Default")),
        )
        .unwrap();
    assert_eq!(reply.pop::<u64>(), 0);
    let reply = client
        .send(
            title.display,
            &Request::new(1010).arg(display_name("Bogus")),
        )
        .unwrap();
    assert_eq!(reply.result, super::OPERATION_FAILED);
    let reply = client
        .send(title.display, &Request::new(5202).arg(0u64))
        .unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let Some(KObject::ReadableEvent(vsync)) = client.object(reply.copy_handles[0]) else {
        panic!("no vsync event");
    };
    let deadline = Instant::now() + Duration::from_secs(2);
    while !vsync.is_signaled() {
        assert!(Instant::now() < deadline, "vsync never fired");
        std::thread::sleep(Duration::from_millis(2));
    }
    let reply = client
        .send(title.display, &Request::new(5202).arg(0u64))
        .unwrap();
    assert_eq!(reply.result, super::PERMISSION_DENIED);
}

#[test]
fn managed_layers_open_only_for_their_owner() {
    let title = Title::boot();
    let layer = title.managed_layer();
    assert_eq!(
        title.open_layer(layer, title.aruid + 1),
        Err(super::PERMISSION_DENIED)
    );
    let binder = title.open_layer(layer, title.aruid).unwrap();
    assert!(binder > 0);
    assert_eq!(
        title.open_layer(layer, title.aruid),
        Err(super::OPERATION_FAILED)
    );
}

#[test]
fn title_presents_frames_paced_by_the_compositor() {
    let title = Title::boot();
    let binder = title
        .open_layer(title.managed_layer(), title.aruid)
        .unwrap();
    connect_with_buffers(&title, binder, 2);

    let query = title.transact(binder, 9, |p| p.write(0i32));
    assert_eq!((word(&query, 0), status(&query)), (1280, 0));

    let (first, status0) = dequeue(&title, binder);
    let (second, status1) = dequeue(&title, binder);
    assert_eq!((status0, status1), (0, 0));
    assert_ne!(first, second);
    let request = title.transact(binder, 1, |p| p.write(first));
    assert_eq!(
        word(&request, 0),
        1,
        "RequestBuffer returns the preallocated buffer"
    );
    assert_eq!(word(&request, 12 + 4), 1280);

    queue(&title, binder, first);
    queue(&title, binder, second);
    let (third, status2) = dequeue(&title, binder);
    assert_eq!(status2, 0);
    assert!(third == first || third == second);

    let native = title
        .client
        .send(title.relay, &Request::new(2).arg(binder).arg(0x0fu32));
    let native = native.unwrap();
    assert_eq!(native.result, ResultCode::SUCCESS);
    let Some(KObject::ReadableEvent(released)) = title.client.object(native.copy_handles[0]) else {
        panic!("no native handle");
    };
    assert!(released.is_signaled(), "the compositor released a buffer");

    let perf = title
        .context
        .perf
        .snapshot(Instant::now(), Duration::from_secs(5));
    assert_eq!(
        perf.game_frame_times_ms.len(),
        1,
        "two QueueBuffers, one interval"
    );
}
