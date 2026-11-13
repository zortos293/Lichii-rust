use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::ipc::{Handle, UNKNOWN_COMMAND_ID};
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Reply, Request};

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
fn later_firmware_state_and_controller_commands_answer_like_the_reference() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let call = |session: Handle, request: Request| {
        let reply = client.send(session, &request).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        reply
    };
    let state = interface(&client, proxy, 0);
    call(state, Request::new(610).arg(0u64));
    call(state, Request::new(611).arg(1u8));
    assert_eq!(call(state, Request::new(1002)).pop::<u8>(), 0);
    call(state, Request::new(1000));
    assert_eq!(
        call(state, Request::new(1002)).pop::<u8>(),
        1,
        "3D VR mode on"
    );
    call(state, Request::new(1001));
    assert_eq!(call(state, Request::new(1002)).pop::<u8>(), 0);
    assert_eq!(
        call(state, Request::new(1003)).pop::<[i32; 4]>(),
        [0, 0, 1280, 720]
    );
    assert_eq!(
        call(state, Request::new(1004)).pop::<[f32; 2]>(),
        [137.25, 77.2]
    );
    assert_eq!(
        call(state, Request::new(1005)).pop::<[i32; 2]>(),
        [1280, 720]
    );

    let controller = interface(&client, proxy, 1);
    assert_eq!(
        call(controller, Request::new(230).arg(7u32)).pop::<u16>(),
        0
    );
    call(controller, Request::new(240).arg(7u32));
}

#[test]
fn an_unpopped_storage_is_the_next_launch_parameter() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let creator = interface(&client, proxy, 11);
    let functions = interface(&client, proxy, 20);
    let storage = |data: &[u8]| {
        let request = Request::new(10).arg(data.len() as i64);
        let storage = client.send(creator, &request).unwrap().move_handles[0];
        let accessor = interface(&client, storage, 0);
        let input = client.alloc(data);
        let write = Request::new(10)
            .arg(0i64)
            .buffer_a(input, data.len() as u64);
        assert_eq!(
            client.send(accessor, &write).unwrap().result,
            ResultCode::SUCCESS
        );
        storage
    };

    let unpop = Request::new(122).move_handle(storage(b"launch"));
    assert_eq!(
        client.send(functions, &unpop).unwrap().result,
        ResultCode::SUCCESS
    );
    let popped = client
        .send(functions, &Request::new(1).arg(1u32))
        .unwrap()
        .move_handles[0];
    let accessor = interface(&client, popped, 0);
    let output = client.alloc(&[0; 6]);
    let read = Request::new(11).arg(0i64).buffer_b(output, 6);
    assert_eq!(
        client.send(accessor, &read).unwrap().result,
        ResultCode::SUCCESS
    );
    assert_eq!(client.read(output, 6), b"launch");
    let reply = client.send(functions, &Request::new(1).arg(1u32)).unwrap();
    assert_eq!(reply.result, NO_DATA_IN_CHANNEL);
    assert_eq!(
        client.send(functions, &Request::new(122)).unwrap().result,
        UNKNOWN
    );

    let state = interface(&client, proxy, 0);
    let push = Request::new(20).move_handle(storage(b"general"));
    assert_eq!(
        client.send(state, &push).unwrap().result,
        ResultCode::SUCCESS
    );
}

#[test]
fn unpopped_storages_are_charged_to_the_live_storage_budget() {
    let (_context, client, _) = booted(SWITCH1_TITLE);
    let am = Domain::open(&client);
    let functions = am.object(am.proxy, Request::new(20));
    let quarter = storage::LIVE_STORAGE_BUDGET / 4;
    let storage = am.object(am.creator, Request::new(10).arg(quarter as i64));
    let unpop = || {
        am.call(functions, Request::new(122).input_object(storage))
            .result
    };
    for copy in 1..=3 {
        assert_eq!(unpop(), ResultCode::SUCCESS, "copy {copy}");
    }
    assert_eq!(
        unpop(),
        UNKNOWN,
        "the original and three copies fill the budget"
    );
}

#[test]
fn cache_storage_limits_come_from_the_nacp() {
    let mut image = crate::testing::image();
    let mut raw = crate::testing::nacp("Cache", 1, 0).raw().to_vec();
    raw[0x3180..0x3188].copy_from_slice(&0x40_0000u64.to_le_bytes());
    raw[0x3188..0x318a].copy_from_slice(&3u16.to_le_bytes());
    image.nacp = Some(lichii_fs::nacp::Nacp::parse(&raw).unwrap());
    let (context, client) = crate::testing::context_with(image, Settings::default());
    register(&context).unwrap();
    let functions = interface(&client, open(&client, 0), 20);
    let mut reply = client.send(functions, &Request::new(29)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!((reply.pop::<u32>(), reply.pop::<u64>()), (3, 0x40_0000));
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
fn live_storage_is_charged_until_freed() {
    let budget = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let full = storage::Charge::take(&budget, storage::LIVE_STORAGE_BUDGET).unwrap();
    assert!(storage::Charge::take(&budget, 1).is_none());
    drop(full);
    let half = storage::LIVE_STORAGE_BUDGET / 2;
    let first = storage::Charge::take(&budget, half).unwrap();
    let _second = storage::Charge::take(&budget, half).unwrap();
    assert!(storage::Charge::take(&budget, 1).is_none());
    drop(first);
    assert!(storage::Charge::take(&budget, half).is_some());
    assert!(storage::Charge::take(&budget, usize::MAX).is_none());
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

#[test]
fn launch_parameter_preselects_the_acc_user_whose_save_ensure_save_data_creates() {
    let dir = crate::testing::TempDir::new("am-preselected");
    let settings = dir.settings();
    let nand = settings.paths.nand_dir.clone().unwrap();
    let (context, client) = context(SWITCH1_TITLE, settings);
    crate::acc::register(&context).unwrap();
    register(&context).unwrap();
    let proxy = open(&client, 0);
    let functions = interface(&client, proxy, 20);
    let user = *context.user.get().unwrap();

    let reply = client.send(functions, &Request::new(1).arg(2u32)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let accessor = interface(&client, reply.move_handles[0], 0);
    let output = client.alloc(&[0xff; 0x88]);
    let read = Request::new(11).arg(0i64).buffer_b(output, 0x88);
    assert_eq!(
        client.send(accessor, &read).unwrap().result,
        ResultCode::SUCCESS
    );
    let data = client.read(output, 0x88);
    assert_eq!(&data[0..4], &0xc794_97cau32.to_le_bytes());
    assert_eq!(&data[4..8], &1u32.to_le_bytes());
    assert_eq!(&data[8..24], &user);
    assert!(data[24..].iter().all(|&b| b == 0));
    let again = client.send(functions, &Request::new(1).arg(2u32)).unwrap();
    assert_eq!(again.result, NO_DATA_IN_CHANNEL);

    let mut reply = client.send(functions, &Request::new(20).arg(user)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.pop::<u64>(), 0);
    let id = u128::from_le_bytes(user);
    let save = nand.join(format!(
        "user/save/0000000000000000/{:016X}{:016X}/{SWITCH1_TITLE:016X}",
        (id >> 64) as u64,
        id as u64
    ));
    assert!(save.is_dir(), "{} was not created", save.display());
}

/// An `appletOE` proxy session converted to a domain, as titles use it.
struct Domain<'a> {
    client: &'a IpcClient,
    session: Handle,
    proxy: u32,
    creator: u32,
}

impl<'a> Domain<'a> {
    fn open(client: &'a IpcClient) -> Self {
        let session = open(client, 0);
        let mut reply = client.send(session, &Request::control(0)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let proxy = reply.pop::<u32>();
        let mut domain = Self {
            client,
            session,
            proxy,
            creator: 0,
        };
        domain.creator = domain.object(proxy, Request::new(11));
        domain
    }

    fn call(&self, object: u32, request: Request) -> Reply {
        self.client
            .send(self.session, &request.to_object(object))
            .unwrap()
    }

    #[track_caller]
    fn object(&self, object: u32, request: Request) -> u32 {
        let reply = self.call(object, request);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.objects.len(), 1, "expected one object");
        reply.objects[0]
    }

    fn storage(&self, data: &[u8]) -> u32 {
        let storage = self.object(self.creator, Request::new(10).arg(data.len() as i64));
        let accessor = self.object(storage, Request::new(0));
        let input = self.client.alloc(data);
        let write = Request::new(10)
            .arg(0i64)
            .buffer_a(input, data.len() as u64);
        assert_eq!(self.call(accessor, write).result, ResultCode::SUCCESS);
        storage
    }

    fn read(&self, storage: u32) -> Vec<u8> {
        let accessor = self.object(storage, Request::new(0));
        let size = self.call(accessor, Request::new(0)).pop::<i64>() as usize;
        let output = self.client.alloc(&vec![0xff; size]);
        let read = Request::new(11).arg(0i64).buffer_b(output, size as u64);
        assert_eq!(self.call(accessor, read).result, ResultCode::SUCCESS);
        self.client.read(output, size)
    }

    /// Creates applet `id`, pushes the common arguments and `arguments`, and starts it.
    fn run(&self, id: u32, mode: u32, library_version: u32, arguments: &[&[u8]]) -> u32 {
        let applet = self.object(self.creator, Request::new(0).arg(id).arg(mode));
        for data in
            std::iter::once(&common_arguments(library_version)[..]).chain(arguments.iter().copied())
        {
            let storage = self.storage(data);
            let push = Request::new(100).input_object(storage);
            assert_eq!(self.call(applet, push).result, ResultCode::SUCCESS);
        }
        assert_eq!(
            self.call(applet, Request::new(10)).result,
            ResultCode::SUCCESS
        );
        applet
    }

    fn completed(&self, applet: u32) -> bool {
        let state_changed = self.call(applet, Request::new(0)).copy_handles[0];
        let completed = self.call(applet, Request::new(1)).pop::<u8>() != 0;
        assert_eq!(event_signaled(self.client, state_changed), completed);
        completed
    }

    fn pop_out_data(&self, applet: u32) -> Vec<u8> {
        let event = self.call(applet, Request::new(105)).copy_handles[0];
        assert!(event_signaled(self.client, event), "pop-out-data event");
        let storage = self.object(applet, Request::new(101));
        assert!(
            !event_signaled(self.client, event),
            "out-data channel is empty"
        );
        self.read(storage)
    }
}

/// `CommonArguments` version 1 with `library_version`.
fn common_arguments(library_version: u32) -> Vec<u8> {
    let mut data = vec![0; 0x20];
    data[0..4].copy_from_slice(&1u32.to_le_bytes());
    data[4..8].copy_from_slice(&0x20u32.to_le_bytes());
    data[8..12].copy_from_slice(&library_version.to_le_bytes());
    data
}

/// A Switch 1 8.0.0 software keyboard config (`SwkbdConfigCommon` and `SwkbdConfigNew`).
fn keyboard_config(text_check: bool) -> Vec<u8> {
    let mut config = vec![0; 0x4c8];
    config[0x3ac..0x3b0].copy_from_slice(&16u32.to_le_bytes());
    config[0x3d0] = u8::from(text_check);
    config
}

const KEYBOARD_VERSION: u32 = 0x8000d;

fn booted_with_acc() -> (crate::testing::TempDir, Arc<ServiceContext>, IpcClient) {
    let dir = crate::testing::TempDir::new("am-applets");
    let (context, client) = context(SWITCH1_TITLE, dir.settings());
    crate::acc::register(&context).unwrap();
    register(&context).unwrap();
    (dir, context, client)
}

#[test]
fn profile_selector_returns_the_acc_user_through_a_domain() {
    let (_dir, context, client) = booted_with_acc();
    let am = Domain::open(&client);
    let user = *context.user.get().unwrap();

    let applet = am.run(library_applet::PROFILE_SELECT, 0, 0x20000, &[&[0; 0xa0]]);
    assert!(am.completed(applet));
    let mut info = am.call(applet, Request::new(120));
    assert_eq!(info.pop::<[u32; 2]>(), [0x10, 0]);
    assert_eq!(
        am.call(applet, Request::new(30)).result,
        ResultCode::SUCCESS
    );

    let out = am.pop_out_data(applet);
    assert_eq!(out.len(), 0x18);
    assert_eq!(&out[..8], &0u64.to_le_bytes());
    assert_eq!(&out[8..], &user);
    assert_eq!(
        am.call(applet, Request::new(101)).result,
        NO_DATA_IN_CHANNEL
    );
}

#[test]
fn profile_selector_cancels_without_a_user() {
    let (_context, client, _) = booted(SWITCH1_TITLE);
    let am = Domain::open(&client);
    let applet = am.run(library_applet::PROFILE_SELECT, 0, 1, &[&[0; 0x98]]);
    let out = am.pop_out_data(applet);
    let cancelled = u64::from(ResultCode::new(124, 1).0);
    assert_eq!(&out[..8], &cancelled.to_le_bytes());
    assert_eq!(&out[8..], &[0; 16]);
}

#[test]
fn software_keyboard_returns_cancel() {
    let (_context, client, _) = booted(SWITCH1_TITLE);
    let am = Domain::open(&client);
    let applet = am.run(
        library_applet::SOFTWARE_KEYBOARD,
        0,
        KEYBOARD_VERSION,
        &[&keyboard_config(false), &[0; 0x1000]],
    );
    assert!(am.completed(applet));
    let out = am.pop_out_data(applet);
    assert_eq!(out.len(), 4 + 0x7d4);
    assert_eq!(&out[..4], &1u32.to_le_bytes(), "SwkbdResult::Cancel");
    assert!(out[4..].iter().all(|&byte| byte == 0));
}

#[test]
fn inline_software_keyboard_completes_with_decided_cancel() {
    let (_context, client, _) = booted(SWITCH1_TITLE);
    let am = Domain::open(&client);
    let applet = am.run(
        library_applet::SOFTWARE_KEYBOARD,
        1,
        KEYBOARD_VERSION,
        &[&[0; 8]],
    );
    assert!(am.completed(applet));
    let event = am.call(applet, Request::new(106)).copy_handles[0];
    assert!(event_signaled(&client, event));
    let reply = am.object(applet, Request::new(104));
    let mut expected = 1u32.to_le_bytes().to_vec();
    expected.extend(6u32.to_le_bytes());
    assert_eq!(am.read(reply), expected);
    assert_eq!(
        am.call(applet, Request::new(101)).result,
        NO_DATA_IN_CHANNEL
    );
}

/// Types "Lichi", fixes it to "Lichii" when rejected, and on a confirmation submits the text
/// unchanged, or "Lichii!" when it edits.
struct Typist {
    edits_on_confirm: bool,
}

impl applets::AppletFrontend for Typist {
    fn show_error(&self, _request: &applets::ErrorRequest) {}

    fn select_profile(&self, _: &applets::ProfileSelectRequest) -> applets::ProfileSelectAnswer {
        applets::ProfileSelectAnswer::Cancelled
    }

    fn configure_controllers(&self, _: &applets::ControllerRequest) -> applets::ControllerAnswer {
        applets::ControllerAnswer::Cancelled
    }

    fn software_keyboard(
        &self,
        request: &applets::SoftwareKeyboardRequest,
    ) -> applets::SoftwareKeyboardAnswer {
        let text = match &request.text_check {
            None => "Lichi",
            Some(applets::TextCheckPrompt::Rejected(message)) => {
                assert_eq!(message, "no");
                "Lichii"
            }
            Some(applets::TextCheckPrompt::Confirm(_)) if self.edits_on_confirm => "Lichii!",
            Some(applets::TextCheckPrompt::Confirm(_)) => &request.initial_text,
        };
        applets::SoftwareKeyboardAnswer::Submitted(text.into())
    }
}

/// Runs the keyboard with `Typist` and a text check, answers each text the keyboard sends for
/// checking with the given `SwkbdTextCheckResult`, and returns the text the applet exits with.
fn type_with_text_check(edits_on_confirm: bool, checks: &[(u32, &str)]) -> String {
    let (context, client, _) = booted(SWITCH1_TITLE);
    context.set_applet_frontend(Arc::new(Typist { edits_on_confirm }));
    let am = Domain::open(&client);
    let applet = am.run(
        library_applet::SOFTWARE_KEYBOARD,
        0,
        KEYBOARD_VERSION,
        &[&keyboard_config(true), &[0; 0x1000]],
    );
    let utf16 =
        |text: &str| -> Vec<u8> { text.encode_utf16().flat_map(u16::to_le_bytes).collect() };
    for &(result, expected) in checks {
        assert!(!am.completed(applet));
        let request = am.read(am.object(applet, Request::new(104)));
        let size = (expected.len() as u64 + 1) * 2;
        assert_eq!(&request[..8], &size.to_le_bytes());
        assert_eq!(&request[8..8 + size as usize - 2], utf16(expected));
        let mut reply = vec![0; 0x7d8];
        reply[..4].copy_from_slice(&result.to_le_bytes());
        reply[4..8].copy_from_slice(&utf16("no"));
        let push = Request::new(103).input_object(am.storage(&reply));
        assert_eq!(am.call(applet, push).result, ResultCode::SUCCESS);
    }
    assert!(am.completed(applet));
    assert_eq!(
        am.call(applet, Request::new(104)).result,
        NO_DATA_IN_CHANNEL
    );
    let out = am.pop_out_data(applet);
    assert_eq!(&out[..4], &0u32.to_le_bytes(), "SwkbdResult::Ok");
    let units: Vec<u16> = out[4..]
        .chunks(2)
        .map(|unit| u16::from_le_bytes([unit[0], unit[1]]))
        .take_while(|&unit| unit != 0)
        .collect();
    String::from_utf16(&units).unwrap()
}

#[test]
fn a_frontend_keyboard_goes_through_the_titles_text_check() {
    assert_eq!(
        type_with_text_check(false, &[(1, "Lichi"), (0, "Lichii")]),
        "Lichii"
    );
}

#[test]
fn confirmed_text_exits_but_edited_text_is_checked_again() {
    assert_eq!(type_with_text_check(false, &[(2, "Lichi")]), "Lichi");
    assert_eq!(
        type_with_text_check(true, &[(2, "Lichi"), (0, "Lichii!")]),
        "Lichii!"
    );
}

#[test]
fn error_applet_completes_with_zeroed_out_data() {
    let (_context, client, _) = booted(SWITCH1_TITLE);
    let am = Domain::open(&client);
    let mut show_error = vec![0; 0x14];
    show_error[0x10..].copy_from_slice(&ResultCode::new(2, 1).0.to_le_bytes());
    let applet = am.run(library_applet::ERROR, 0, 0x20000, &[&show_error]);
    assert!(am.completed(applet));
    assert_eq!(am.pop_out_data(applet), vec![0; 0x1000]);
    assert_eq!(
        am.call(applet, Request::new(101)).result,
        NO_DATA_IN_CHANNEL
    );
}

/// Runs the controller applet for a Switch 1 11.0.0 title and returns its
/// `ControllerSupportResultInfo`.
fn configure_controllers(am: &Domain) -> Vec<u8> {
    let mut private = vec![0; 0x14];
    private[0..4].copy_from_slice(&0x14u32.to_le_bytes());
    private[4..8].copy_from_slice(&0x430u32.to_le_bytes());
    private[0xc..0x10].copy_from_slice(&3u32.to_le_bytes());
    let mut argument = vec![0; 0x430];
    argument[..2].copy_from_slice(&[1, 4]);
    let applet = am.run(library_applet::CONTROLLER, 0, 8, &[&private, &argument]);
    assert!(am.completed(applet));
    let out = am.pop_out_data(applet);
    assert_eq!(out.len(), 0xc);
    assert_eq!(
        &out[8..12],
        &0u32.to_le_bytes(),
        "ControllerSupportResult::Success"
    );
    out
}

#[test]
fn controller_applet_reports_the_npads_hid_connected() {
    let (context, client, _) = booted(SWITCH1_TITLE);
    crate::hid::register(&context).unwrap();
    let am = Domain::open(&client);
    let out = configure_controllers(&am);
    assert_eq!(
        out[0], 0,
        "no npad is connected before the title starts hid"
    );
    assert_eq!(&out[4..8], &0u32.to_le_bytes(), "player 1");

    const ARUID: u64 = 0x51;
    const FULL_KEY: u32 = 1;
    let hid = client.connect("hid").unwrap();
    for request in [
        Request::new(0).send_pid().arg(ARUID),
        Request::new(103).send_pid().arg(ARUID),
        Request::new(100).send_pid().arg(FULL_KEY).arg(ARUID),
    ] {
        assert_eq!(
            client.send(hid, &request).unwrap().result,
            ResultCode::SUCCESS
        );
    }
    let handheld = lichii_input::NpadState {
        connected: true,
        controller: lichii_common::settings::ControllerType::Handheld,
        ..context.npad.read(0)
    };
    context.npad.publish(0, &handheld);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while context.hid.get().unwrap().controller_setup().player_count == 0 {
        assert!(
            std::time::Instant::now() < deadline,
            "player 1 never connected"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let out = configure_controllers(&am);
    assert_eq!(out[0], 1, "player count");
    assert_eq!(
        &out[4..8],
        &0u32.to_le_bytes(),
        "a handheld the title does not accept is player 1"
    );
}

#[test]
fn unknown_applets_exit_without_out_data() {
    let (_context, client, _) = booted(SWITCH1_TITLE);
    let am = Domain::open(&client);
    let applet = am.run(0x0100_0013, 0, 0, &[]);
    assert!(am.completed(applet));
    let mut info = am.call(applet, Request::new(120));
    assert_eq!(
        info.pop::<[u32; 2]>(),
        [0x13, 0],
        "normalized to the low byte"
    );
    assert_eq!(
        am.call(applet, Request::new(101)).result,
        NO_DATA_IN_CHANNEL
    );
}

#[test]
fn push_in_data_rejects_objects_that_are_not_storage() {
    let (_context, client, _) = booted(SWITCH1_TITLE);
    let am = Domain::open(&client);
    let applet = am.object(
        am.creator,
        Request::new(0).arg(library_applet::ERROR).arg(0u32),
    );
    let storage = am.storage(&[0; 0x20]);
    let accessor = am.object(storage, Request::new(0));
    for object in [am.creator, accessor] {
        let push = Request::new(100).input_object(object);
        assert_eq!(am.call(applet, push).result, UNKNOWN);
    }
    assert_eq!(am.call(applet, Request::new(100)).result, UNKNOWN);
    assert!(!am.completed(applet));
}

#[test]
fn push_in_data_takes_a_storage_moved_as_a_handle_outside_domains() {
    let (_context, client, proxy) = booted(SWITCH1_TITLE);
    let creator = interface(&client, proxy, 11);
    let reply = client
        .send(
            creator,
            &Request::new(0).arg(library_applet::ERROR).arg(0u32),
        )
        .unwrap();
    let applet = reply.move_handles[0];
    let storage = client
        .send(creator, &Request::new(10).arg(0x20i64))
        .unwrap()
        .move_handles[0];
    let push = Request::new(100).move_handle(storage);
    assert_eq!(
        client.send(applet, &push).unwrap().result,
        ResultCode::SUCCESS
    );
    let push = Request::new(100).move_handle(creator);
    assert_eq!(client.send(applet, &push).unwrap().result, UNKNOWN);
}
