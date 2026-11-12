use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::ipc::{Handle, UNKNOWN_COMMAND_ID};
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Request};

use super::*;
use crate::testing::{SWITCH1_TITLE, SWITCH2_TITLE, context};

const CURRENT_PROCESS: Handle = 0xffff_8001;

fn open(client: &IpcClient, open_command: u32) -> Handle {
    let service = client.connect("appletOE").expect("appletOE registered");
    let request = Request::new(open_command)
        .send_pid()
        .arg(0u64)
        .copy_handle(CURRENT_PROCESS);
    let reply = client.send(service, &request).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    reply.move_handles[0]
}

fn interface(client: &IpcClient, session: Handle, command: u32) -> Handle {
    let reply = client.send(session, &Request::new(command)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS, "command {command}");
    reply.move_handles[0]
}

fn booted(program_id: u64) -> (Arc<ServiceContext>, IpcClient, Handle) {
    let mut settings = Settings::default();
    settings.system.docked = true;
    let (context, client) = context(program_id, settings);
    register(&context).unwrap();
    let proxy = open(&client, 0);
    (context, client, proxy)
}

fn event_signaled(client: &IpcClient, handle: Handle) -> bool {
    match client.object(handle) {
        Some(KObject::ReadableEvent(event)) => event.is_signaled(),
        _ => panic!("handle {handle:#x} is not a readable event"),
    }
}

#[test]
fn first_message_is_focus_state_changed_and_the_event_follows_the_queue() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let state = interface(&client, proxy, 0);
    let reply = client.send(state, &Request::new(0)).unwrap();
    let event = reply.copy_handles[0];
    assert!(event_signaled(&client, event));

    let mut reply = client.send(state, &Request::new(1)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.pop::<u32>(), AppletMessage::FocusStateChanged as u32);
    assert!(!event_signaled(&client, event));
    assert_eq!(
        client.send(state, &Request::new(1)).unwrap().result,
        NO_MESSAGES
    );

    let mut reply = client.send(state, &Request::new(9)).unwrap();
    assert_eq!(reply.pop::<u8>(), IN_FOCUS);
    let mut reply = client.send(state, &Request::new(5)).unwrap();
    assert_eq!(reply.pop::<u8>(), 1, "docked");
    let mut reply = client.send(state, &Request::new(60)).unwrap();
    assert_eq!((reply.pop::<i32>(), reply.pop::<i32>()), (1920, 1080));

    assert_eq!(
        client.send(state, &Request::new(130)).unwrap().result,
        ResultCode::SUCCESS
    );
    assert!(event_signaled(&client, event));
    let mut reply = client.send(state, &Request::new(1)).unwrap();
    assert_eq!(
        reply.pop::<u32>(),
        AppletMessage::StartupLogoDisappeared as u32
    );
}

#[test]
fn focus_notification_can_be_turned_off_before_the_first_message() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let controller = interface(&client, proxy, 1);
    let request = Request::new(13).arg(0u8).arg(0u8).arg(1u8);
    assert_eq!(
        client.send(controller, &request).unwrap().result,
        ResultCode::SUCCESS
    );
    let state = interface(&client, proxy, 0);
    assert_eq!(
        client.send(state, &Request::new(1)).unwrap().result,
        NO_MESSAGES
    );
}

#[test]
fn switch2_proxy_getters_and_auto_stubs() {
    let (_context, client, _) = booted(SWITCH2_TITLE);
    let proxy = open(&client, 10);

    let scale = interface(&client, proxy, 60);
    let mut reply = client.send(scale, &Request::new(10)).unwrap();
    assert_eq!(reply.pop::<[u32; 3]>(), [10, 0, 0]);

    let stub = interface(&client, proxy, 40);
    let reply = client.send(stub, &Request::new(1234).arg(7u64)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.data(), [0; 128]);

    let functions = interface(&client, proxy, 20);
    let mut reply = client.send(functions, &Request::new(320)).unwrap();
    assert_eq!(reply.pop::<u16>(), 0);
    let reply = client.send(functions, &Request::new(9999)).unwrap();
    assert_eq!(
        reply.result,
        ResultCode::SUCCESS,
        "Switch 2 titles are auto-stubbed"
    );
}

#[test]
fn switch1_unknown_commands_are_reported() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let functions = interface(&client, proxy, 20);
    let reply = client.send(functions, &Request::new(9999)).unwrap();
    assert_eq!(reply.result, UNKNOWN_COMMAND_ID);
}

#[test]
fn application_functions_answer_from_the_title_and_settings() {
    let (context, client, proxy) = booted(SWITCH1_TITLE);
    let functions = interface(&client, proxy, 20);
    let mut reply = client.send(functions, &Request::new(21)).unwrap();
    assert_eq!(reply.pop::<u64>(), u64::from_le_bytes(*b"en-US\0\0\0"));
    let mut reply = client.send(functions, &Request::new(23)).unwrap();
    assert_eq!(&reply.pop::<[u8; 16]>()[..6], b"1.0.0\0");
    let mut reply = client.send(functions, &Request::new(50)).unwrap();
    let id = reply.pop::<[u8; 16]>();
    assert_eq!(
        id,
        functions::pseudo_device_id(context.program.pseudo_device_seed)
    );
    let reply = client.send(functions, &Request::new(1).arg(1u32)).unwrap();
    assert_eq!(reply.result, NO_DATA_IN_CHANNEL);

    let window = interface(&client, proxy, 2);
    let mut reply = client.send(window, &Request::new(1)).unwrap();
    assert_eq!(reply.pop::<u64>(), context.process.id());
}

#[test]
fn storages_read_and_write_within_bounds() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let creator = interface(&client, proxy, 11);
    let reply = client.send(creator, &Request::new(10).arg(16i64)).unwrap();
    let storage = reply.move_handles[0];
    let accessor = interface(&client, storage, 0);

    let mut reply = client.send(accessor, &Request::new(0)).unwrap();
    assert_eq!(reply.pop::<i64>(), 16);
    let input = client.alloc(b"abcd");
    let write = Request::new(10).arg(4i64).buffer_a(input, 4);
    assert_eq!(
        client.send(accessor, &write).unwrap().result,
        ResultCode::SUCCESS
    );
    let output = client.alloc(&[0xff; 8]);
    let read = Request::new(11).arg(2i64).buffer_b(output, 8);
    assert_eq!(
        client.send(accessor, &read).unwrap().result,
        ResultCode::SUCCESS
    );
    assert_eq!(client.read(output, 8), b"\0\0abcd\0\0");
    let past_end = Request::new(10).arg(14i64).buffer_a(input, 4);
    assert_eq!(
        client.send(accessor, &past_end).unwrap().result,
        INVALID_OFFSET
    );
    assert_eq!(
        client
            .send(creator, &Request::new(10).arg(0i64))
            .unwrap()
            .result,
        ResultCode(u32::MAX)
    );
}

#[test]
fn fatal_sections_must_balance() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let controller = interface(&client, proxy, 1);
    assert_eq!(
        client.send(controller, &Request::new(4)).unwrap().result,
        FATAL_SECTION_COUNT_IMBALANCE
    );
    assert_eq!(
        client.send(controller, &Request::new(3)).unwrap().result,
        ResultCode::SUCCESS
    );
    assert_eq!(
        client.send(controller, &Request::new(4)).unwrap().result,
        ResultCode::SUCCESS
    );
}

#[test]
fn lock_accessors_start_free_with_their_event_signaled() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let state = interface(&client, proxy, 0);
    let lock = interface(&client, state, 30);
    let reply = client.send(lock, &Request::new(3)).unwrap();
    assert!(event_signaled(&client, reply.copy_handles[0]));
    let mut reply = client.send(lock, &Request::new(1).arg(0u8)).unwrap();
    assert_eq!(reply.pop::<u8>(), 1, "acquired");
    let mut reply = client.send(lock, &Request::new(1).arg(0u8)).unwrap();
    assert_eq!(reply.pop::<u8>(), 0, "already held");
}
