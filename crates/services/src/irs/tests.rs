use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::ipc::UNKNOWN_COMMAND_ID;
use lichii_kernel::memory_objects::{SharedMemory, TransferMemory};
use lichii_kernel::object::KObject;
use lichii_kernel::testing::IpcClient;
use lichii_memory::MemoryPermission;

use super::{
    DEVICE_SIZE, INVALID_IR_CAMERA_HANDLE, INVALID_NPAD_ID, INVALID_PROCESSOR_STATE, register,
};
use crate::testing::{Reply, ReplyExt, Request, SWITCH1_TITLE, SWITCH2_TITLE, booted, context};

const ARUID: u64 = 0x51;

fn shared_memory(client: &IpcClient) -> Arc<SharedMemory> {
    let irs = client.connect("irs").unwrap();
    let reply = client
        .send(irs, &Request::new(304).arg(ARUID).send_pid())
        .unwrap()
        .ok();
    match client.object(reply.handle()) {
        Some(KObject::SharedMemory(memory)) => memory,
        _ => panic!("not shared memory"),
    }
}

fn device(memory: &SharedMemory, index: u64) -> (u32, u32, u64) {
    let mut raw = [0u8; 16];
    memory.read(index * DEVICE_SIZE, &mut raw);
    (
        u32::from_le_bytes(raw[0..4].try_into().unwrap()),
        u32::from_le_bytes(raw[4..8].try_into().unwrap()),
        u64::from_le_bytes(raw[8..16].try_into().unwrap()),
    )
}

/// Camera handle, aruid, then the config words.
fn run(client: &IpcClient, command: u32, camera: u32, config: &[u32]) -> Reply {
    let irs = client.connect("irs").unwrap();
    let mut request = Request::new(command).arg(camera);
    request = match command {
        310 | 312 | 317 => config
            .iter()
            .fold(request, |r, &word| r.arg(word))
            .arg(ARUID),
        _ => config
            .iter()
            .fold(request.arg(ARUID), |r, &word| r.arg(word)),
    };
    client.send(irs, &request.send_pid()).unwrap()
}

#[test]
fn shared_memory_holds_nine_unconnected_ready_cameras() {
    let client = booted(SWITCH1_TITLE, Settings::default(), register);
    let memory = shared_memory(&client);
    assert_eq!(memory.size(), 0x8000);
    for index in 0..9 {
        assert_eq!(device(&memory, index), (2, 7, 0), "camera {index}");
    }
    assert!(Arc::ptr_eq(&memory, &shared_memory(&client)));
}

#[test]
fn camera_handles_follow_the_reference_npad_indices() {
    let client = booted(SWITCH1_TITLE, Settings::default(), register);
    let irs = client.connect("irs").unwrap();
    for (npad_id, index) in [(0u32, 0u32), (7, 7), (0x20, 8), (0x10, 9)] {
        let mut reply = client
            .send(irs, &Request::new(311).arg(npad_id))
            .unwrap()
            .ok();
        assert_eq!(reply.pop::<u32>(), index, "npad {npad_id:#x}");
    }
    let reply = client.send(irs, &Request::new(311).arg(0x30u32)).unwrap();
    assert_eq!(reply.result, INVALID_NPAD_ID);
}

#[test]
fn activation_and_handle_checks_succeed_for_valid_cameras_only() {
    let client = booted(SWITCH1_TITLE, Settings::default(), register);
    let irs = client.connect("irs").unwrap();
    for command in [302, 303] {
        let request = Request::new(command).arg(ARUID).send_pid();
        client.send(irs, &request).unwrap().ok();
    }
    let request = Request::new(319).arg(1u32).arg(ARUID).send_pid();
    client.send(irs, &request).unwrap().ok();

    for command in [305, 313, 318] {
        run(&client, command, 8, &[]).ok();
        assert_eq!(
            run(&client, command, 9, &[]).result,
            INVALID_IR_CAMERA_HANDLE
        );
    }
    for command in [314, 315] {
        let irs_request = |camera: u32| {
            Request::new(command)
                .arg(camera)
                .arg(0x0003_0005u32)
                .arg(ARUID)
                .send_pid()
        };
        client.send(irs, &irs_request(0)).unwrap().ok();
        let reply = client.send(irs, &irs_request(0x0100)).unwrap();
        assert_eq!(
            reply.result, INVALID_IR_CAMERA_HANDLE,
            "npad style must be none"
        );
    }
}

#[test]
fn processors_report_a_stopped_unconnected_camera_in_their_mode() {
    let client = booted(SWITCH1_TITLE, Settings::default(), register);
    let memory = shared_memory(&client);
    memory.write(DEVICE_SIZE + 0x10, &[0xaa; 0x40]);
    for (command, camera, config, mode) in [
        (306u32, 1u32, vec![0u32; 8], 1u64),
        (307, 2, vec![0; 10], 2),
        (310, 3, vec![0; 2], 5),
        (312, 4, vec![0; 3], 4),
        (317, 8, vec![0; 2], 6),
    ] {
        run(&client, command, camera, &config).ok();
        assert_eq!(
            device(&memory, camera.into()),
            (2, 0, mode),
            "command {command}"
        );
    }
    let mut lifo = [0xffu8; 0x40];
    memory.read(DEVICE_SIZE + 0x10, &mut lifo);
    assert_eq!(lifo, [0; 0x40], "the moment processor clears its lifo");
    assert_eq!(
        device(&memory, 0),
        (2, 7, 0),
        "other cameras keep their state"
    );
    assert_eq!(
        run(&client, 306, 9, &[0; 8]).result,
        INVALID_IR_CAMERA_HANDLE
    );
}

#[test]
fn image_transfer_reads_the_titles_transfer_memory() {
    let (context, client) = context(SWITCH1_TITLE, Settings::default());
    register(&context).unwrap();
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    let image: Vec<u8> = (0..300u32).map(|i| i as u8).collect();
    process.memory().write_bytes(heap, &image).unwrap();
    let transfer =
        TransferMemory::new(process.clone(), heap, 0x1000, MemoryPermission::READ).unwrap();
    let transfer = process
        .handles()
        .write()
        .add(KObject::TransferMemory(Arc::new(transfer)))
        .unwrap();

    let irs = client.connect("irs").unwrap();
    let out = heap + 0x1000;
    let state = |camera: u32| {
        Request::new(309)
            .arg(camera)
            .arg(ARUID)
            .send_pid()
            .buffer_b(out, 0x200)
    };
    let reply = client.send(irs, &state(1)).unwrap();
    assert_eq!(reply.result, INVALID_PROCESSOR_STATE);

    let mut config = [0u32; 6];
    config[5] = 4;
    let run = config
        .iter()
        .fold(Request::new(308).arg(1u32).arg(ARUID), |r, &word| {
            r.arg(word)
        })
        .arg(0x1000u64)
        .send_pid()
        .copy_handle(transfer);
    client.send(irs, &run).unwrap().ok();
    assert_eq!(device(&shared_memory(&client), 1), (2, 0, 3));

    let reply = client.send(irs, &state(1)).unwrap().ok();
    assert_eq!(reply.data()[..16], [0; 16], "the camera never sampled");
    assert_eq!(client.read(out, 300), image, "a 20x15 image");
    assert_eq!(client.read(out + 300, 4), [0; 4]);

    let reply = client
        .send(irs, &Request::new(308).arg(2u32).send_pid())
        .unwrap();
    assert_ne!(reply.result, ResultCode::SUCCESS, "no transfer memory");
}

#[test]
fn irs_sys_is_auto_stubbed_only_for_switch_2_titles() {
    for (title, expected) in [
        (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
        (SWITCH2_TITLE, ResultCode::SUCCESS),
    ] {
        let client = booted(title, Settings::default(), register);
        let sys = client.connect("irs:sys").unwrap();
        let reply = client.send(sys, &Request::new(500).arg(ARUID)).unwrap();
        assert_eq!(reply.result, expected, "title {title:#x}");
    }
}
