use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::ipc::{Handle, UNKNOWN_COMMAND_ID};
use lichii_kernel::memory_objects::{SharedMemory, TransferMemory};
use lichii_kernel::object::KObject;
use lichii_kernel::testing::IpcClient;
use lichii_memory::MemoryPermission;

use super::{UNKNOWN, register};
use crate::testing::{ReplyExt, Request, SWITCH1_TITLE, SWITCH2_TITLE, booted, context};

const ARUID: u64 = 0x51;
const VALID: u64 = 1 << 56;

fn bus_handle(client: &IpcClient, bus: Handle, npad_id: u32, bus_type: u64) -> u64 {
    let request = Request::new(1)
        .arg(npad_id)
        .arg(bus_type)
        .arg(ARUID)
        .send_pid();
    let reply = client.send(bus, &request).unwrap().ok();
    assert_eq!(reply.get::<u8>(0), 1, "valid");
    reply.get::<u64>(8)
}

fn shared_memory(client: &IpcClient, bus: Handle) -> Arc<SharedMemory> {
    let reply = client.send(bus, &Request::new(10)).unwrap().ok();
    match client.object(reply.handle()) {
        Some(KObject::SharedMemory(memory)) => memory,
        _ => panic!("not shared memory"),
    }
}

fn entry(memory: &SharedMemory, index: u64) -> [u8; 16] {
    let mut raw = [0; 16];
    memory.read(index * 0x80, &mut raw);
    raw
}

#[test]
fn bus_handles_are_made_once_per_npad_and_bus() {
    let client = booted(SWITCH1_TITLE, Settings::default(), register);
    let bus = client.connect("hidbus").unwrap();
    let right_rail = bus_handle(&client, bus, 0, 1);
    assert_eq!(right_rail, VALID | (1 << 48));
    assert_eq!(bus_handle(&client, bus, 0, 1), right_rail);
    let handheld = bus_handle(&client, bus, 0x20, 0);
    assert_eq!(handheld, VALID | (0x20 << 40) | (1 << 32) | 1);

    let other = client.connect("hidbus").unwrap();
    assert_eq!(
        bus_handle(&client, other, 0x20, 0),
        handheld,
        "one bus state"
    );
}

#[test]
fn unknown_handles_fail() {
    let client = booted(SWITCH1_TITLE, Settings::default(), register);
    let bus = client.connect("hidbus").unwrap();
    let handle = VALID | 0x1234;
    for command in [2, 4, 6, 9, 12] {
        let reply = client
            .send(bus, &Request::new(command).arg(handle))
            .unwrap();
        assert_eq!(reply.result, UNKNOWN, "command {command}");
    }
}

#[test]
fn the_stub_device_is_never_connected_and_reports_its_state() {
    let (context, client) = context(SWITCH1_TITLE, Settings::default());
    register(&context).unwrap();
    let bus = client.connect("hidbus").unwrap();
    let memory = shared_memory(&client, bus);
    assert_eq!(memory.size(), 0x1000);
    let handle = bus_handle(&client, bus, 0, 1);
    let call = |request: Request| client.send(bus, &request).unwrap().ok();

    call(Request::new(3).arg(handle).arg(ARUID).send_pid());
    assert_eq!(
        entry(&memory, 0)[..12],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    );
    assert_eq!(
        call(Request::new(2).arg(handle)).get::<u8>(0),
        0,
        "not connected"
    );
    assert_eq!(call(Request::new(6).arg(handle)).get::<u32>(0), 0xff);

    call(
        Request::new(5)
            .arg(1u64)
            .arg(handle)
            .arg(0u64)
            .arg(ARUID)
            .send_pid(),
    );
    assert_eq!(entry(&memory, 0)[8], 1, "enabled");

    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    let transfer =
        TransferMemory::new(process.clone(), heap, 0x1000, MemoryPermission::NONE).unwrap();
    let transfer = process
        .handles()
        .write()
        .add(KObject::TransferMemory(Arc::new(transfer)))
        .unwrap();
    call(
        Request::new(11)
            .arg(0x1000u32)
            .arg(2u32)
            .arg(handle)
            .copy_handle(transfer),
    );
    let polling = entry(&memory, 0);
    assert_eq!((polling[10], polling[12]), (1, 2), "button-only polling");
    call(Request::new(12).arg(handle));
    let polling = entry(&memory, 0);
    assert_eq!((polling[10], polling[12]), (0, 2));

    let command = client.alloc(&[1, 2, 3, 4]);
    call(Request::new(7).arg(handle).buffer_a(command, 4));
    let reply = call(Request::new(8).arg(handle).buffer_b(heap + 0x1000, 0x30));
    assert_eq!(reply.get::<u64>(0), 0, "no reply data");
    let reply = call(Request::new(9).arg(handle));
    match client.object(reply.handle()) {
        Some(KObject::ReadableEvent(event)) => assert!(!event.is_signaled()),
        _ => panic!("not an event"),
    }

    call(Request::new(4).arg(handle).arg(ARUID).send_pid());
    assert_eq!(entry(&memory, 0)[8..10], [0, 1], "disabled, in focus");
    call(Request::new(14).arg(1u32));
}

#[test]
fn get_polling_data_is_auto_stubbed_only_for_switch_2_titles() {
    for (title, expected) in [
        (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
        (SWITCH2_TITLE, ResultCode::SUCCESS),
    ] {
        let client = booted(title, Settings::default(), register);
        let bus = client.connect("hidbus").unwrap();
        let reply = client.send(bus, &Request::new(13).arg(0u64)).unwrap();
        assert_eq!(reply.result, expected, "title {title:#x}");
    }
}
