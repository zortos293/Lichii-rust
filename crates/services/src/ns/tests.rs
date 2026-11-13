use lichii_common::settings::{Language, Settings};

use super::control::{APPLICATION_LANGUAGE_NOT_FOUND, RESULT_UNKNOWN};
use super::*;
use crate::testing::{self, Guest, PROGRAM_ID, ReplyExt, Request};

/// Supports American English (0), French (3) and German (4).
const SUPPORTED: u32 = 1 | 1 << 3 | 1 << 4;

fn guest(language: Language) -> Guest {
    let mut image = testing::image();
    image.nacp = Some(testing::nacp("Lichii Test", SUPPORTED, 0));
    let mut settings = Settings::default();
    settings.system.language = language;
    Guest::new(image, settings, |context| register(context).unwrap())
}

fn interface(guest: &Guest, getter: &str, command: u32) -> u32 {
    let service = guest.service(getter);
    guest.call(service, Request::new(command)).ok().interface()
}

#[test]
fn every_getter_hands_out_every_interface() {
    let guest = guest(Language::AmericanEnglish);
    for name in ["ns:am2", "ns:ec", "ns:rid", "ns:rt", "ns:web", "ns:ro"] {
        let getter = guest.service(name);
        for command in [
            7988, 7989, 7991, 7992, 7993, 7994, 7995, 7996, 7997, 7998, 7999,
        ] {
            guest.call(getter, Request::new(command)).ok().interface();
        }
    }
    guest.service("ns:dev");
    guest.service("pdm:ntfy");
}

#[test]
fn control_data_is_the_nacp() {
    let guest = guest(Language::AmericanEnglish);
    let control = interface(&guest, "ns:ro", 7989);
    let out = guest.heap;
    guest.write(out, &[0xaa; 0x4100]);
    let reply = guest
        .call(
            control,
            Request::new(0)
                .arg(1u8)
                .arg(PROGRAM_ID)
                .buffer_b(out, 0x4100),
        )
        .ok();
    assert_eq!(reply.get::<u32>(0), 0x4000);
    let data = guest.read(out, 0x4100);
    assert_eq!(&data[..11], b"Lichii Test");
    assert_eq!(data[0x302c..0x3030], SUPPORTED.to_le_bytes());
    assert!(data[0x4000..].iter().all(|&b| b == 0));

    let small = guest.call(
        control,
        Request::new(0)
            .arg(1u8)
            .arg(PROGRAM_ID)
            .buffer_b(out, 0x100),
    );
    assert_eq!(small.result, RESULT_UNKNOWN);

    guest.write(out, &[0xaa; 0x4000]);
    guest
        .call(
            control,
            Request::new(0)
                .arg(1u8)
                .arg(0x0100_0000_0009_0000u64)
                .buffer_b(out, 0x4000),
        )
        .ok();
    assert!(guest.read(out, 0x4000).iter().all(|&b| b == 0));
}

#[test]
fn control_data_with_flags() {
    let guest = guest(Language::AmericanEnglish);
    let control = interface(&guest, "ns:am2", 7989);
    let out = guest.heap;
    let packed = guest
        .call(
            control,
            Request::new(5)
                .arg(1u8)
                .arg(1u8)
                .arg(0u8)
                .arg(PROGRAM_ID)
                .buffer_b(out, 0x4000),
        )
        .ok();
    assert_eq!(packed.get::<u64>(0), 0x4000 << 32 | 1);
    assert_eq!(&guest.read(out, 6), b"Lichii");
    for command in [19, 23] {
        let flags = guest
            .call(
                control,
                Request::new(command)
                    .arg(1u8)
                    .arg(0u8)
                    .arg(0u8)
                    .arg(PROGRAM_ID)
                    .buffer_b(out, 0x4000),
            )
            .ok();
        assert_eq!(
            [
                flags.get::<u32>(0),
                flags.get::<u32>(4),
                flags.get::<u32>(8)
            ],
            [0x10001, 0, 0x4000]
        );
    }
}

#[test]
fn desired_language_follows_the_priority_list() {
    let guest = guest(Language::CanadianFrench);
    let control = interface(&guest, "ns:am2", 7989);
    let desired = |supported: u32| guest.call(control, Request::new(1).arg(supported));
    assert_eq!(desired(SUPPORTED).ok().get::<u8>(0), 3);
    assert_eq!(desired(1 << 4).ok().get::<u8>(0), 4);
    assert_eq!(desired(0).ok().get::<u8>(0), 9);
    assert_eq!(desired(1 << 20).result, APPLICATION_LANGUAGE_NOT_FOUND);

    let code = guest.call(control, Request::new(2).arg(9u8)).ok();
    assert_eq!(code.get::<u64>(0), u64::from_le_bytes(*b"fr-CA\0\0\0"));
    assert_eq!(
        guest.call(control, Request::new(2).arg(18u8)).result,
        APPLICATION_LANGUAGE_NOT_FOUND
    );
}

#[test]
fn list_titles_and_icons_into_transfer_memory() {
    let guest = guest(Language::German);
    let control = interface(&guest, "ns:am2", 7989);
    let (ids, memory) = (guest.heap, guest.heap + 0x1000);
    guest.write(
        ids,
        &[PROGRAM_ID.to_le_bytes(), 7u64.to_le_bytes()].concat(),
    );
    let handle = guest.transfer_memory(memory, 0x1000);

    let titles = guest
        .call(
            control,
            Request::new(13).copy_handle(handle).buffer_a(ids, 16),
        )
        .ok();
    assert!(guest.is_signaled(titles.handle()));
    let value = titles.interface();
    assert_eq!(guest.call(value, Request::new(0)).ok().get::<i64>(0), 0x600);
    let out = guest.heap + 0x3000;
    guest.call(value, Request::new(1).buffer_b(out, 4)).ok();
    assert_eq!(guest.read(out, 4), [0; 4]);
    let entries = guest.read(memory, 0x600);
    assert_eq!(&entries[..11], b"Lichii Test");
    assert_eq!(&entries[0x200..0x206], b"Lichii");
    assert!(entries[0x300..].iter().all(|&b| b == 0));

    let icons = guest
        .call(
            control,
            Request::new(10).copy_handle(handle).buffer_a(ids, 16),
        )
        .ok();
    let value = icons.interface();
    assert_eq!(guest.call(value, Request::new(0)).ok().get::<i64>(0), 24);
    assert_eq!(guest.read(memory, 8), 2u64.to_le_bytes());
}

#[test]
fn application_manager() {
    let guest = guest(Language::AmericanEnglish);
    let manager = interface(&guest, "ns:am2", 7996);
    let out = guest.heap;

    let records = guest
        .call(manager, Request::new(0).arg(0i32).buffer_b(out, 0x30))
        .ok();
    assert_eq!(records.get::<i32>(0), 1);
    let record = guest.read(out, 0x18);
    assert_eq!(record[..8], PROGRAM_ID.to_le_bytes());
    assert_eq!(record[8], 3);
    let past = guest
        .call(manager, Request::new(0).arg(1i32).buffer_b(out, 0x30))
        .ok();
    assert_eq!(past.get::<i32>(0), 0);

    let event = guest.call(manager, Request::new(2)).ok().handle();
    assert!(guest.is_signaled(event));
    for command in [44, 52, 505, 511, 4022] {
        guest.call(manager, Request::new(command)).ok().handle();
    }

    let total = guest.call(manager, Request::new(47).arg(4u8)).ok();
    assert_eq!(total.get::<i64>(0), 0x6_8000_0000);
    let free = guest.call(manager, Request::new(48).arg(4u8)).ok();
    assert_eq!(free.get::<i64>(0), 0x6_8000_0000 - 0x10_0000);
    let both = guest.call(manager, Request::new(71).arg(5u8)).ok();
    assert_eq!([both.get::<i64>(0), both.get::<i64>(8)], [4 << 30, 4 << 30]);

    let language = guest.call(manager, Request::new(55).arg(SUPPORTED)).ok();
    assert_eq!(language.get::<u8>(0), 0);
    let code = guest.call(manager, Request::new(59).arg(0u8)).ok();
    assert_eq!(code.get::<u64>(0), u64::from_le_bytes(*b"en-US\0\0\0"));
    let control = guest
        .call(
            manager,
            Request::new(400)
                .arg(0u8)
                .arg(PROGRAM_ID)
                .buffer_b(out, 0x4000),
        )
        .ok();
    assert_eq!(control.get::<u32>(0), 0x4000);

    assert_eq!(guest.call(manager, Request::new(512)).ok().get::<u8>(0), 0);
    let update = guest.call(manager, Request::new(906).arg(PROGRAM_ID)).ok();
    assert_eq!([update.get::<u8>(0), update.get::<u32>(4) as u8], [0, 0]);
    assert_eq!(guest.call(manager, Request::new(936)).ok().get::<u64>(0), 0);
    assert_eq!(
        guest.call(manager, Request::new(4023)).ok().get::<u64>(0),
        0
    );
    assert_eq!(guest.call(manager, Request::new(1300)).ok().get::<u8>(0), 1);
    assert_eq!(
        guest
            .call(manager, Request::new(2520).arg(1u64))
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        guest
            .call(manager, Request::new(2100).arg(PROGRAM_ID))
            .ok()
            .get::<u32>(0),
        0
    );
    assert_eq!(guest.call(manager, Request::new(40)).result, RESULT_UNKNOWN);

    let async_result = guest
        .call(manager, Request::new(4042).arg(0u64).arg(0u64))
        .ok();
    let (event, result) = (async_result.handle(), async_result.interface());
    guest.call(result, Request::new(1)).ok();
    assert!(guest.is_signaled(event));
}

#[test]
fn application_views_and_rights() {
    let guest = guest(Language::AmericanEnglish);
    let manager = interface(&guest, "ns:am2", 7996);
    let (ids, out) = (guest.heap, guest.heap + 0x1000);
    guest.write(
        ids,
        &[PROGRAM_ID.to_le_bytes(), 5u64.to_le_bytes()].concat(),
    );
    for (command, size) in [(1701, 0x50), (1706, 0x58)] {
        guest.write(out, &[0xaa; 0x100]);
        guest
            .call(
                manager,
                Request::new(command).buffer_a(ids, 16).buffer_b(out, 0x100),
            )
            .ok();
        let views = guest.read(out, 2 * size);
        assert_eq!(views[..8], PROGRAM_ID.to_le_bytes());
        assert_eq!(views[8..16], [0, 0, 7, 0, 0x17, 0x1f, 0x40, 0]);
        assert!(views[16..size].iter().all(|&b| b == 0));
        assert_eq!(views[size..size + 8], 5u64.to_le_bytes());
    }
    let promoted = guest
        .call(
            manager,
            Request::new(1704).buffer_a(ids, 16).buffer_b(out, 0x100),
        )
        .ok();
    assert_eq!(promoted.get::<u32>(0), 2);

    let uid = [0x11u8; 16];
    let rights = guest
        .call(
            manager,
            Request::new(2050)
                .arg(0u32)
                .arg(PROGRAM_ID)
                .arg(uid)
                .buffer_b(out, 0x20),
        )
        .ok();
    assert_eq!(rights.get::<u32>(0), 1);
    let entry = guest.read(out, 0x20);
    assert_eq!(entry[..8], PROGRAM_ID.to_le_bytes());
    assert_eq!(entry[8..24], uid);
}

#[test]
fn records_documents_rights_and_content() {
    let guest = guest(Language::AmericanEnglish);
    let record = interface(&guest, "ns:am2", 7991);
    assert_eq!(
        guest
            .call(record, Request::new(0).arg(PROGRAM_ID))
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        guest
            .call(record, Request::new(2).arg(0u32))
            .ok()
            .get::<u8>(0),
        0
    );
    let out = guest.heap;
    let list = guest
        .call(record, Request::new(3).arg(0i32).buffer_b(out, 0x18))
        .ok();
    assert_eq!(list.get::<i32>(0), 1);

    let document = interface(&guest, "ns:am2", 7999);
    let running = guest.call(document, Request::new(92).arg(0u64)).ok();
    assert_eq!(running.get::<u64>(0), PROGRAM_ID);

    let rights = interface(&guest, "ns:am2", 7988);
    assert_eq!(
        guest
            .call(rights, Request::new(13).arg(0u64))
            .ok()
            .get::<u32>(0),
        0
    );
    assert_eq!(guest.call(rights, Request::new(26)).ok().get::<u8>(0), 0);

    let content = interface(&guest, "ns:am2", 7998);
    let occupied = guest.call(content, Request::new(11).arg(PROGRAM_ID)).ok();
    assert_eq!(
        [
            occupied.get::<u64>(0),
            occupied.get::<u64>(8),
            occupied.get::<u64>(0x60)
        ],
        [5, 8 << 30, 5]
    );
    let total = guest.call(content, Request::new(47).arg(3u8)).ok();
    assert_eq!(total.get::<i64>(0), 0xa000_0000);
    let free = guest.call(content, Request::new(48).arg(1u8)).ok();
    assert_eq!(free.get::<i64>(0), 0xa000_0000 + 0x6_8000_0000 - 0x10_0000);
    let unknown = guest.call(content, Request::new(71).arg(0u8)).ok();
    assert_eq!([unknown.get::<u64>(0), unknown.get::<u64>(8)], [0, 0]);

    let download = interface(&guest, "ns:am2", 7997);
    guest.call(download, Request::new(707)).ok();
}

#[test]
fn system_update_and_vulnerability() {
    let guest = guest(Language::AmericanEnglish);
    let update = guest.service("ns:su");
    assert_eq!(guest.call(update, Request::new(0)).ok().get::<u8>(0), 0);
    guest.call(update, Request::new(1)).ok().interface();
    guest.call(update, Request::new(9)).ok().handle();

    let vm = guest.service("ns:vm");
    assert_eq!(guest.call(vm, Request::new(1200)).ok().get::<u8>(0), 0);
    let info = guest.call(vm, Request::new(3100)).ok();
    assert_eq!(info.get::<[u8; 0x20]>(0), [0; 0x20]);
    guest.call(vm, Request::new(3102)).ok();
}
