use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_common::ResultCode;
use lichii_common::settings::{ControllerType, Settings};
use lichii_input::{MotionSample, NpadButtons, NpadState, StickPosition, VibrationValue};
use lichii_kernel::ipc::Handle;
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Reply, Request};

use super::npad::{HANDHELD, Style};
use super::shared_memory::{
    self as layout, GcTriggerState, Lifo, LifoState, NpadEntry, NpadGenericState, PadLifo,
    SixAxisLifo, SixAxisSensorState,
};
use super::*;
use crate::testing::{SWITCH1_TITLE, context};

const ARUID: u64 = 0x51;
const STYLES: u32 = 0b10111;

/// A test title with a hid session and the HID shared memory it got from `IAppletResource`.
struct Hid {
    context: Arc<ServiceContext>,
    client: IpcClient,
    hid: Handle,
    memory: Arc<SharedMemory>,
}

impl Hid {
    fn new() -> Self {
        let (context, client) = context(SWITCH1_TITLE, Settings::default());
        register(&context).unwrap();
        let hid = client.connect("hid").expect("hid registered");
        let resource = ok(client.send(hid, &Request::new(0).send_pid().arg(ARUID))).move_handles[0];
        let handle = ok(client.send(resource, &Request::new(0))).copy_handles[0];
        let memory = match client.object(handle) {
            Some(KObject::SharedMemory(memory)) => memory,
            _ => panic!("GetSharedMemoryHandle did not return shared memory"),
        };
        assert_eq!(memory.size(), layout::SIZE);
        Self {
            context,
            client,
            hid,
            memory,
        }
    }

    fn send(&self, request: Request) -> Answer {
        self.request(self.hid, request)
    }

    fn request(&self, session: Handle, request: Request) -> Answer {
        let reply = self.client.send(session, &request).unwrap();
        Answer {
            result: reply.result,
            copies: reply.copy_handles.clone(),
            moves: reply.move_handles.clone(),
            data: reply.data().to_vec(),
        }
    }

    fn read<T: bytemuck::Pod>(&self, offset: u64) -> T {
        layout::get(&self.memory, offset)
    }

    /// The newest entry of a lifo: `(stored sampling number, state, valid count)`.
    fn newest<S: LifoState>(&self, lifo_offset: u64) -> (i64, S, i64) {
        let tail: i64 = self.read(lifo_offset + 0x10);
        let count: i64 = self.read(lifo_offset + 0x18);
        let entry = lifo_offset + 0x20 + tail as u64 * (8 + size_of::<S>() as u64);
        (self.read(entry), self.read(entry + 8), count)
    }

    fn publish(&self, player: usize, state: NpadState) {
        self.context.npad.publish(player, &state);
    }

    fn start_npads(&self, styles: u32) {
        self.send(Request::new(103).send_pid().arg(ARUID)).ok();
        self.send(Request::new(100).send_pid().arg(styles).arg(ARUID))
            .ok();
    }

    fn style(&self, index: usize) -> u32 {
        self.read(NpadEntry::new(index).at(NpadEntry::STYLE_SET))
    }

    fn event(&self, handle: Handle) -> Arc<lichii_kernel::Event> {
        match self.client.object(handle) {
            Some(KObject::ReadableEvent(event)) => event,
            _ => panic!("handle {handle:#x} is not a readable event"),
        }
    }
}

fn ok(reply: Result<Reply, ResultCode>) -> Reply {
    let reply = reply.unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    reply
}

/// A reply with its data kept for reads at fixed offsets.
#[derive(Debug)]
struct Answer {
    result: ResultCode,
    copies: Vec<Handle>,
    moves: Vec<Handle>,
    data: Vec<u8>,
}

impl Answer {
    fn ok(self) -> Self {
        assert_eq!(self.result, ResultCode::SUCCESS, "{self:?}");
        self
    }

    fn get<T: bytemuck::Pod>(&self, offset: usize) -> T {
        bytemuck::pod_read_unaligned(&self.data[offset..offset + size_of::<T>()])
    }
}

fn eventually(what: &str, check: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !check() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn lifo_offset<S: LifoState>(lifo: Lifo<S>) -> u64 {
    lifo.offset()
}

fn pro_controller() -> NpadState {
    NpadState {
        connected: true,
        controller: ControllerType::ProController,
        buttons: NpadButtons::A
            | NpadButtons::ZL
            | NpadButtons::LEFT_SL
            | NpadButtons::C
            | NpadButtons::HOME,
        left_stick: StickPosition { x: 100, y: -200 },
        right_stick: StickPosition {
            x: -0x7fff,
            y: 0x7fff,
        },
        ..NpadState::default()
    }
}

#[test]
fn shared_memory_starts_with_empty_lifos_and_no_controllers() {
    let hid = Hid::new();
    for lifo in [
        0x0, 0x400, 0x3400, 0x3800, 0x4c00, 0x4e00, 0x5000, 0x3_ba00, 0x3_dc00,
    ] {
        assert_eq!(hid.read::<i64>(lifo + 8), 17, "lifo at {lifo:#x}");
    }
    for index in 0..NPAD_COUNT {
        let entry = NpadEntry::new(index);
        assert_eq!(hid.style(index), 0);
        assert_eq!(hid.read::<u32>(entry.at(NpadEntry::FULL_KEY_COLOR)), 2);
        assert_eq!(hid.read::<u32>(entry.at(NpadEntry::JOY_COLOR)), 2);
        assert_eq!(
            hid.read::<i64>(lifo_offset(entry.pad(PadLifo::SystemExt)) + 8),
            17
        );
        assert_eq!(
            hid.read::<i64>(lifo_offset(entry.sixaxis(SixAxisLifo::Right)) + 8),
            17
        );
        assert_eq!(hid.read::<i64>(lifo_offset(entry.gc_trigger()) + 8), 17);
    }
    assert_eq!(hid.read::<[u32; 4]>(0x3_e200), [0, 1, 0, 1]);
    assert_eq!(
        lifo_offset(NpadEntry::new(1).pad(PadLifo::Handheld)),
        0xea00 + 0x28 + 0x350
    );
}

#[test]
fn npads_stay_disconnected_until_the_title_sets_its_styles() {
    let hid = Hid::new();
    hid.publish(0, pro_controller());
    hid.send(Request::new(103).send_pid().arg(ARUID)).ok();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(hid.style(0), 0);
    assert_eq!(
        hid.send(Request::new(101).send_pid().arg(ARUID))
            .ok()
            .get::<u32>(0),
        0
    );
    hid.send(Request::new(100).send_pid().arg(STYLES).arg(ARUID))
        .ok();
    assert_eq!(
        hid.send(Request::new(101).send_pid().arg(ARUID))
            .ok()
            .get::<u32>(0),
        STYLES
    );
    eventually("player 1 to connect", || {
        hid.style(0) == Style::FullKey.tag()
    });
}

#[test]
fn the_updater_samples_npad_hub_into_the_lifos() {
    let hid = Hid::new();
    hid.publish(0, pro_controller());
    let event = hid
        .send(Request::new(106).send_pid().arg(0u32).arg(ARUID).arg(0u64))
        .ok()
        .copies[0];
    let event = hid.event(event);
    assert!(event.is_signaled(), "acquiring signals the event");
    event.clear();
    hid.start_npads(STYLES);
    eventually("player 1 to connect", || {
        hid.style(0) == Style::FullKey.tag()
    });
    assert!(event.is_signaled(), "connecting signals the event");

    let entry = NpadEntry::new(0);
    assert_eq!(hid.read::<u32>(entry.at(NpadEntry::DEVICE_TYPE)), 1);
    assert_eq!(hid.read::<u32>(entry.at(NpadEntry::FULL_KEY_COLOR)), 0);
    assert_eq!(hid.read::<u8>(entry.at(NpadEntry::FOOTER_TYPE)), 12);
    assert_eq!(hid.read::<u8>(entry.at(NpadEntry::SIXAXIS_PROPERTIES)), 1);
    let full_key = lifo_offset(entry.pad(PadLifo::FullKey));
    eventually("a full key sample with input", || {
        hid.newest::<NpadGenericState>(full_key).1.buttons != 0
    });
    let (stored, state, count) = hid.newest::<NpadGenericState>(full_key);
    assert_eq!(state.buttons, (NpadButtons::A | NpadButtons::ZL).bits());
    assert_eq!(state.left_stick, [100, -200]);
    assert_eq!(state.right_stick, [-0x7fff, 0x7fff]);
    assert_eq!(state.attributes, 0b11);
    assert_eq!(stored, state.sampling_number << 1);
    assert!((1..=16).contains(&count));
    eventually("the lifo to fill", || {
        hid.newest::<NpadGenericState>(full_key).2 == 16
    });
    let later = hid.newest::<NpadGenericState>(full_key).1.sampling_number;
    assert!(later > state.sampling_number);

    let (_, ext, _) = hid.newest::<NpadGenericState>(lifo_offset(entry.pad(PadLifo::SystemExt)));
    assert_eq!(ext.buttons, state.buttons);
    let (_, handheld, _) =
        hid.newest::<NpadGenericState>(lifo_offset(entry.pad(PadLifo::Handheld)));
    assert_eq!(handheld.buttons, 0);
}

#[test]
fn handheld_moves_player_1_to_the_handheld_entry() {
    let hid = Hid::new();
    hid.start_npads(STYLES);
    hid.publish(0, pro_controller());
    eventually("player 1 to connect", || {
        hid.style(0) == Style::FullKey.tag()
    });
    hid.publish(
        0,
        NpadState {
            controller: ControllerType::Switch2Handheld,
            buttons: NpadButtons::B | NpadButtons::LEFT_SL,
            ..pro_controller()
        },
    );
    eventually("the handheld to connect", || {
        hid.style(HANDHELD) == Style::Handheld.tag() && hid.style(0) == 0
    });
    let entry = NpadEntry::new(HANDHELD);
    assert_eq!(hid.read::<u32>(entry.at(NpadEntry::DEVICE_TYPE)), 0b1100);
    let handheld = lifo_offset(entry.pad(PadLifo::Handheld));
    eventually("a handheld sample", || {
        hid.newest::<NpadGenericState>(handheld).1.buttons == NpadButtons::B.bits()
    });
    assert_eq!(hid.newest::<NpadGenericState>(handheld).1.attributes, 0x3f);
}

#[test]
fn unsupported_styles_fall_back_and_unsupported_ids_stay_off() {
    let hid = Hid::new();
    hid.publish(
        1,
        NpadState {
            controller: ControllerType::GameCube,
            left_trigger: 0x4000,
            ..pro_controller()
        },
    );
    hid.publish(2, pro_controller());
    let ids = hid.client.alloc(bytemuck::cast_slice(&[0u32, 1, 0x20]));
    hid.send(Request::new(102).send_pid().arg(ARUID).buffer_x(ids, 12))
        .ok();
    hid.start_npads(Style::FullKey.tag() | Style::JoyDual.tag());
    eventually("player 2 to connect", || {
        hid.style(1) == Style::FullKey.tag()
    });
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(hid.style(2), 0);

    let too_many = [0u32; 11];
    let ids = hid.client.alloc(bytemuck::cast_slice(&too_many));
    let reply = hid.send(Request::new(102).send_pid().arg(ARUID).buffer_x(ids, 44));
    assert_eq!(reply.result, INVALID_ARRAY_SIZE);
    let ids = hid.client.alloc(bytemuck::cast_slice(&[7u32, 0x30]));
    let reply = hid.send(Request::new(102).send_pid().arg(ARUID).buffer_x(ids, 8));
    assert_eq!(reply.result, INVALID_NPAD_ID);
}

#[test]
fn game_cube_triggers_and_buttons_are_remapped() {
    let hid = Hid::new();
    hid.publish(
        0,
        NpadState {
            controller: ControllerType::GameCube,
            buttons: NpadButtons::R | NpadButtons::ZL,
            left_trigger: 0x1234,
            right_trigger: 0x7fff,
            ..pro_controller()
        },
    );
    hid.start_npads(Style::Gc.tag());
    eventually("the GameCube controller", || {
        hid.style(0) == Style::Gc.tag()
    });
    let entry = NpadEntry::new(0);
    let full_key = lifo_offset(entry.pad(PadLifo::FullKey));
    eventually("a sample", || {
        hid.newest::<NpadGenericState>(full_key).1.buttons != 0
    });
    assert_eq!(
        hid.newest::<NpadGenericState>(full_key).1.buttons,
        (NpadButtons::ZR | NpadButtons::L).bits()
    );
    let (_, trigger, _) = hid.newest::<GcTriggerState>(lifo_offset(entry.gc_trigger()));
    assert_eq!((trigger.left, trigger.right), (0x1234, 0x7fff));
}

#[test]
fn disconnect_npad_keeps_the_npad_off_until_it_reconnects() {
    let hid = Hid::new();
    hid.publish(0, pro_controller());
    hid.start_npads(STYLES);
    eventually("player 1 to connect", || hid.style(0) != 0);
    hid.send(Request::new(107).send_pid().arg(0u32).arg(ARUID))
        .ok();
    assert_eq!(hid.style(0), 0);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(hid.style(0), 0);
    hid.publish(0, NpadState::default());
    std::thread::sleep(Duration::from_millis(20));
    hid.publish(0, pro_controller());
    eventually("player 1 to reconnect", || hid.style(0) != 0);
}

#[test]
fn six_axis_lifos_get_motion_once_the_sensor_starts() {
    let hid = Hid::new();
    hid.publish(
        0,
        NpadState {
            motion: Some(MotionSample {
                accel: [0.5, 0.25, -1.0],
                gyro: [0.0, 0.0, 1.0],
            }),
            ..pro_controller()
        },
    );
    hid.start_npads(STYLES);
    eventually("player 1 to connect", || hid.style(0) != 0);
    let full_key = lifo_offset(NpadEntry::new(0).sixaxis(SixAxisLifo::FullKey));
    eventually("a six-axis sample", || {
        hid.newest::<SixAxisSensorState>(full_key).2 > 0
    });
    let (_, idle, _) = hid.newest::<SixAxisSensorState>(full_key);
    assert_eq!((idle.accel, idle.attributes), ([0.0, 0.0, -1.0], 1));
    assert_eq!(idle.orientation[0], [1.0, 0.0, 0.0]);

    let handle = DeviceHandle {
        style_index: 3,
        npad_id: 0,
        device_index: 2,
        reserved: 0,
    };
    hid.send(Request::new(66).send_pid().arg(handle).arg(ARUID))
        .ok();
    eventually("motion in the lifo", || {
        hid.newest::<SixAxisSensorState>(full_key).1.gyro == [0.0, 0.0, 1.0]
    });
    let (stored, state, _) = hid.newest::<SixAxisSensorState>(full_key);
    assert_eq!(stored, state.sampling_number << 1);
    assert_eq!(state.accel, [0.5, 0.25, -1.0]);
    assert_eq!(state.delta_time, 5_000_000);
    assert!(state.rotation[2] > 0.0);
    let (_, unused, _) =
        hid.newest::<SixAxisSensorState>(lifo_offset(NpadEntry::new(0).sixaxis(SixAxisLifo::Left)));
    assert_eq!(unused.attributes, 0);

    assert_eq!(
        hid.send(Request::new(68).send_pid().arg(handle).arg(ARUID))
            .ok()
            .get::<u8>(0),
        1
    );
    hid.send(
        Request::new(70)
            .send_pid()
            .arg(handle)
            .arg([0.5f32, 0.75])
            .arg(ARUID),
    )
    .ok();
    let params = hid
        .send(Request::new(71).send_pid().arg(handle).arg(ARUID))
        .ok();
    assert_eq!(params.get::<[f32; 2]>(0), [0.5, 0.75]);
    let bad = DeviceHandle {
        npad_id: 9,
        ..handle
    };
    assert_eq!(
        hid.send(Request::new(66).send_pid().arg(bad).arg(ARUID))
            .result,
        INVALID_NPAD_ID
    );
}

#[test]
fn vibration_reaches_the_npad_hub() {
    let hid = Hid::new();
    let npad = &hid.context.npad;
    let handle = |style_index, npad_id, device_index| DeviceHandle {
        style_index,
        npad_id,
        device_index,
        reserved: 0,
    };
    let info = |h: DeviceHandle| hid.send(Request::new(200).arg(h));
    assert_eq!(info(handle(3, 0, 0)).ok().get::<[u32; 2]>(0), [1, 1]);
    assert_eq!(info(handle(7, 0, 1)).ok().get::<[u32; 2]>(0), [1, 2]);
    assert_eq!(info(handle(8, 0, 0)).ok().get::<[u32; 2]>(0), [2, 0]);
    assert_eq!(info(handle(0, 0, 0)).result, VIBRATION_INVALID_STYLE_INDEX);
    assert_eq!(info(handle(3, 9, 0)).result, VIBRATION_INVALID_NPAD_ID);
    assert_eq!(
        info(handle(3, 0, 3)).result,
        VIBRATION_DEVICE_INDEX_OUT_OF_RANGE
    );

    let value = VibrationValue {
        low_amplitude: 0.5,
        low_frequency: 160.0,
        high_amplitude: 0.25,
        high_frequency: 320.0,
    };
    let mut seen = [0u64; 8];
    hid.send(
        Request::new(201)
            .send_pid()
            .arg(handle(5, 1, 1))
            .arg(RawVibration::from(value))
            .arg(ARUID),
    )
    .ok();
    assert_eq!(npad.take_vibration(1, &mut seen[1]), Some(value));
    let actual = hid
        .send(Request::new(202).send_pid().arg(handle(5, 1, 1)).arg(ARUID))
        .ok();
    assert_eq!(actual.get::<[f32; 4]>(0), [0.5, 160.0, 0.25, 320.0]);

    let handles = hid
        .client
        .alloc(bytemuck::cast_slice(&[handle(4, 0x20, 0), handle(3, 2, 0)]));
    let strong = VibrationValue {
        low_amplitude: 1.0,
        ..value
    };
    let values = hid.client.alloc(bytemuck::cast_slice(&[
        RawVibration::from(strong),
        RawVibration::from(value),
    ]));
    hid.send(
        Request::new(206)
            .send_pid()
            .arg(ARUID)
            .buffer_x(handles, 8)
            .buffer_x(values, 32),
    )
    .ok();
    assert_eq!(npad.take_vibration(0, &mut seen[0]), Some(strong));
    assert_eq!(npad.take_vibration(2, &mut seen[2]), Some(value));
    let mismatch = hid.send(
        Request::new(206)
            .send_pid()
            .arg(ARUID)
            .buffer_x(handles, 8)
            .buffer_x(values, 16),
    );
    assert_eq!(mismatch.result, VIBRATION_ARRAY_SIZE_MISMATCH);

    hid.send(Request::new(204).arg(0u8)).ok();
    assert_eq!(hid.send(Request::new(205)).ok().get::<u8>(0), 0);
    hid.send(
        Request::new(201)
            .send_pid()
            .arg(handle(5, 1, 1))
            .arg(RawVibration::from(strong))
            .arg(ARUID),
    )
    .ok();
    assert_eq!(npad.take_vibration(1, &mut seen[1]), None);

    let list = hid.send(Request::new(203)).ok().moves[0];
    hid.request(list, Request::new(0).arg(handle(3, 0, 0))).ok();
    let reply = hid.request(list, Request::new(0).arg(handle(1, 0, 0)));
    assert_eq!(reply.result, VIBRATION_INVALID_STYLE_INDEX);
}

#[test]
fn npad_settings_round_trip() {
    let hid = Hid::new();
    assert_eq!(
        hid.send(Request::new(121).send_pid().arg(ARUID))
            .ok()
            .get::<u64>(0),
        0
    );
    hid.send(Request::new(120).send_pid().arg(ARUID).arg(1u64))
        .ok();
    assert_eq!(
        hid.send(Request::new(121).send_pid().arg(ARUID))
            .ok()
            .get::<u64>(0),
        1
    );
    assert_eq!(hid.read::<[u32; 4]>(0x3_e200), [0, 1, 1, 1]);
    hid.send(Request::new(128).send_pid().arg(ARUID).arg(2u64))
        .ok();
    assert_eq!(
        hid.send(Request::new(129).send_pid().arg(ARUID))
            .ok()
            .get::<u64>(0),
        2
    );
    assert_eq!(
        hid.send(Request::new(108).arg(2u32)).ok().get::<u64>(0),
        0b0111
    );
    hid.send(Request::new(123).send_pid().arg(1u32).arg(ARUID).arg(0i64))
        .ok();
    assert_eq!(
        hid.read::<u32>(NpadEntry::new(1).at(NpadEntry::ASSIGNMENT_MODE)),
        1
    );
    hid.send(Request::new(124).send_pid().arg(1u32).arg(ARUID))
        .ok();
    assert_eq!(
        hid.read::<u32>(NpadEntry::new(1).at(NpadEntry::ASSIGNMENT_MODE)),
        0
    );
    let reply = hid.send(
        Request::new(106)
            .send_pid()
            .arg(0x11u32)
            .arg(ARUID)
            .arg(0u64),
    );
    assert_eq!(reply.result, INVALID_NPAD_ID);
    assert_eq!(
        hid.send(Request::new(5555)).result,
        lichii_kernel::ipc::UNKNOWN_COMMAND_ID
    );
}

#[test]
fn activated_devices_advance_their_sampling_numbers() {
    let hid = Hid::new();
    hid.send(Request::new(11).send_pid().arg(ARUID)).ok();
    hid.send(Request::new(31).send_pid().arg(ARUID)).ok();
    eventually("touch and keyboard samples", || {
        hid.read::<i64>(0x400 + 0x18) > 2 && hid.read::<i64>(0x3800 + 0x18) > 2
    });
    assert_eq!(hid.read::<i64>(0x18), 0, "debug pad was never activated");
    assert!(
        hid.read::<i64>(0x4c00 + 0x18) > 0,
        "home button always samples"
    );
    assert_eq!(
        ResultCode::SUCCESS,
        hid.send(Request::new(1).send_pid().arg(ARUID)).result
    );
    eventually("debug pad samples", || hid.read::<i64>(0x18) > 0);
}
