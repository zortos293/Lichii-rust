use lichii_common::ConsoleGeneration;
use lichii_common::settings::{Language, Region};

use super::{INVALID_LANGUAGE, KEY_CODE_MAP_SIZE, NULL_POINTER, key_code_map};
use crate::testing::{Guest, Request, settings};

fn guest(language: Language, region: Region) -> Guest {
    let mut settings = settings();
    settings.system.language = language;
    settings.system.region = region;
    Guest::new(ConsoleGeneration::Switch2, settings, super::register)
}

fn code(text: &str) -> u64 {
    let mut raw = [0; 8];
    raw[..text.len()].copy_from_slice(text.as_bytes());
    u64::from_le_bytes(raw)
}

fn name(text: &str) -> Vec<u8> {
    let mut raw = vec![0; 0x48];
    raw[..text.len()].copy_from_slice(text.as_bytes());
    raw
}

#[test]
fn language_and_region_follow_the_settings() {
    let guest = guest(Language::BritishEnglish, Region::Europe);
    let set = guest.connect("set");
    assert_eq!(
        guest.call(set, &Request::new(0)).ok().get::<u64>(0),
        code("en-GB")
    );
    assert_eq!(guest.call(set, &Request::new(4)).ok().get::<u32>(0), 2);
    assert_eq!(
        guest
            .call(set, &Request::new(2).arg(14u32))
            .ok()
            .get::<u64>(0),
        code("es-419")
    );
    assert_eq!(
        guest.call(set, &Request::new(2).arg(20u32)).result,
        INVALID_LANGUAGE.0
    );
    assert_eq!(guest.call(set, &Request::new(3)).ok().get::<i32>(0), 15);
    assert_eq!(guest.call(set, &Request::new(6)).ok().get::<i32>(0), 64);
    assert_eq!(guest.call(set, &Request::new(8)).ok().get::<u8>(0), 0);
    assert_eq!(guest.call(set, &Request::new(10)).ok().data, [0; 128]);
}

#[test]
fn korea_and_taiwan_share_a_region_code() {
    let guest = guest(Language::Korean, Region::Korea);
    let set = guest.connect("set");
    assert_eq!(guest.call(set, &Request::new(4)).ok().get::<u32>(0), 4);
}

#[test]
fn available_language_codes_fit_the_buffer() {
    let guest = guest(Language::AmericanEnglish, Region::Usa);
    let set = guest.connect("set");
    let buffer = guest.buffer(0x40 * 8);
    let reply = guest.call(set, &Request::new(1).b(buffer, 0x40 * 8)).ok();
    assert_eq!(reply.get::<i32>(0), 15);
    let reply = guest.call(set, &Request::new(5).b(buffer, 0x40 * 8)).ok();
    assert_eq!(reply.get::<i32>(0), 20);
    assert_eq!(guest.read_value::<u64>(buffer), code("ja"));
    assert_eq!(guest.read_value::<u64>(buffer + 19 * 8), code("th"));
    let small = guest.buffer(3 * 8);
    assert_eq!(
        guest
            .call(set, &Request::new(5).b(small, 3 * 8))
            .ok()
            .get::<i32>(0),
        3
    );
}

#[test]
fn key_code_maps_follow_the_language() {
    let guest = guest(Language::German, Region::Europe);
    let set = guest.connect("set");
    let buffer = guest.buffer(KEY_CODE_MAP_SIZE as u64);
    for command in [7, 9, 12] {
        guest
            .call(
                set,
                &Request::new(command)
                    .arg(0u32)
                    .b(buffer, KEY_CODE_MAP_SIZE as u64),
            )
            .ok();
        let map = guest.read(buffer, KEY_CODE_MAP_SIZE);
        let german = key_code_map(code("de"));
        assert_eq!(&map[..german.len()], german);
        assert!(map[german.len()..].iter().all(|&b| b == 0));
    }
    assert_eq!(guest.call(set, &Request::new(7)).result, NULL_POINTER.0);
    assert_eq!(key_code_map(code("ko")), super::key_code_map::KOREAN);
    assert_eq!(
        key_code_map(code("pl")),
        super::key_code_map::ENGLISH_US_INTERNATIONAL
    );
}

#[test]
fn system_settings_update_what_set_reads() {
    let guest = guest(Language::AmericanEnglish, Region::Usa);
    let set = guest.connect("set");
    let sys = guest.connect("set:sys");
    let nickname = guest.buffer(0x80);
    guest.call(set, &Request::new(11).b(nickname, 0x80)).ok();
    assert_eq!(&guest.read(nickname, 7), b"Lichii\0");

    guest.call(sys, &Request::new(0).arg(code("fr"))).ok();
    guest.call(sys, &Request::new(57).arg(0u32)).ok();
    guest.call(sys, &Request::new(48).arg(1u8)).ok();
    let name = guest.buffer_with(b"Switch\0");
    guest.call(sys, &Request::new(78).a(name, 7)).ok();
    assert_eq!(
        guest.call(set, &Request::new(0)).ok().get::<u64>(0),
        code("fr")
    );
    assert_eq!(guest.call(set, &Request::new(4)).ok().get::<u32>(0), 0);
    assert_eq!(guest.call(set, &Request::new(8)).ok().get::<u8>(0), 1);
    assert_eq!(guest.call(sys, &Request::new(47)).ok().get::<u8>(0), 1);
    guest.call(sys, &Request::new(77).b(nickname, 0x80)).ok();
    assert_eq!(&guest.read(nickname, 7), b"Switch\0");
}

#[test]
fn firmware_version_is_23_0_0() {
    let guest = guest(Language::AmericanEnglish, Region::Usa);
    let sys = guest.connect("set:sys");
    let buffer = guest.buffer(0x100);
    for command in [3, 4] {
        guest
            .call(sys, &Request::new(command).b(buffer, 0x100))
            .ok();
        let version = guest.read(buffer, 0x100);
        assert_eq!(version[..8], [23, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&version[0x08..0x0B], b"NX\0");
        assert_eq!(&version[0x68..0x6F], b"23.0.0\0");
        assert!(version[0x80..].starts_with(b"NintendoSDK Firmware for NX 23.0.0-4.0\0"));
    }
}

#[test]
fn settings_items_return_their_values() {
    let guest = guest(Language::AmericanEnglish, Region::Usa);
    let sys = guest.connect("set:sys");
    let category = guest.buffer_with(&name("hbloader"));
    let item = guest.buffer_with(&name("applet_heap_reservation_size"));
    let request = Request::new(37).x(category, 0x48).x(item, 0x48);
    assert_eq!(guest.call(sys, &request).ok().get::<u64>(0), 8);
    let out = guest.buffer(8);
    let request = Request::new(38).b(out, 8).x(category, 0x48).x(item, 0x48);
    assert_eq!(guest.call(sys, &request).ok().get::<u64>(0), 8);
    assert_eq!(guest.read_value::<u64>(out), 0x860_0000);

    let year = guest.buffer_with(&name("standard_user_clock_initial_year"));
    let time = guest.buffer_with(&name("time"));
    let request = Request::new(38).b(out, 8).x(time, 0x48).x(year, 0x48);
    assert_eq!(guest.call(sys, &request).ok().get::<u64>(0), 4);
    assert_eq!(guest.read_value::<i32>(out), 2023);

    let missing = guest.buffer_with(&name("missing"));
    let reply = guest.call(sys, &Request::new(37).x(time, 0x48).x(missing, 0x48));
    assert_eq!(reply.result, u32::MAX);
}

#[test]
fn system_values_have_sysmodule_defaults_and_round_trip() {
    let guest = guest(Language::Japanese, Region::Japan);
    let sys = guest.connect("set:sys");
    let get = |command| guest.call(sys, &Request::new(command)).ok();
    assert_eq!(get(29).get::<[u32; 6]>(0), [0x300, 2, 9, 0, 21, 0]);
    assert_eq!(get(35).get::<f32>(0), 1.0);
    let tv = get(39).get::<[u32; 8]>(0);
    assert_eq!(
        (tv[0], tv[2], f32::from_bits(tv[6]), f32::from_bits(tv[7])),
        (0xC, 4, 1.0, 0.5)
    );
    assert_eq!(&get(53).data[..4], b"UTC\0");
    assert_eq!(get(60).get::<u8>(0), 1);
    assert_eq!(get(62).get::<u8>(0), 0);
    assert_eq!(get(63).get::<u32>(0), 1);
    assert_eq!(get(71).get::<[u32; 3]>(0), [3, 3, 0]);
    assert_eq!(get(75).get::<u32>(0), 0x10001);
    assert_eq!(get(79).get::<u32>(0), 1);
    assert_eq!(&get(90).data, b"Lichii Author ID");
    assert_eq!(get(136).get::<u32>(0), 0);
    assert_eq!(&get(149).data[0x20..0x26], b"0.0.0\0");
    assert_eq!(
        get(174).get::<[u32; 5]>(0),
        [0xFF32_3232, 0xFF32_3232, !0, !0, 0xFF00_0000]
    );
    assert_eq!(get(183).get::<i32>(0), 1);
    assert_eq!(get(187).get::<u32>(0), 1);
    for command in [7, 25, 27, 45, 65, 95, 201] {
        assert_eq!(get(command).get::<u8>(0), 0, "command {command}");
    }
    for command in [69, 73, 88, 99] {
        assert_eq!(get(command).get::<u8>(0), 1, "command {command}");
    }
    for command in [17, 23, 120, 124, 126, 170, 185, 203, 315] {
        assert_eq!(get(command).get::<u32>(0), 0, "command {command}");
    }
    assert_eq!(get(13).data, [0; 0x10]);
    assert_eq!(get(15).data, [0; 0x20]);
    assert_eq!(get(58).data, [0; 0x20]);
    assert_eq!(get(106).get::<i64>(0), 0);
    assert_eq!(get(150).data, [0; 0x18]);
    assert_eq!(get(152).data, [0; 0x18]);
    assert_eq!(get(321).data, [0; 0x40]);

    let context: [u8; 0x20] = std::array::from_fn(|i| i as u8);
    for (set, get_command) in [(16, 15), (59, 58)] {
        guest.call(sys, &Request::new(set).arg(context)).ok();
        assert_eq!(get(get_command).get::<[u8; 0x20]>(0), context);
    }
    guest.call(sys, &Request::new(24).arg(1u32)).ok();
    assert_eq!(get(23).get::<u32>(0), 1);
    guest.call(sys, &Request::new(105).arg(-5i64)).ok();
    assert_eq!(get(106).get::<i64>(0), -5);
    guest.call(sys, &Request::new(44).arg(1u32).arg(2u32)).ok();
    assert_eq!(
        guest
            .call(sys, &Request::new(43).arg(1u32))
            .ok()
            .get::<u32>(0),
        2
    );
    assert_eq!(
        guest
            .call(sys, &Request::new(43).arg(0u32))
            .ok()
            .get::<u32>(0),
        0
    );
}

#[test]
fn serial_numbers_carry_the_region_and_a_check_digit() {
    let guest = guest(Language::AmericanEnglish, Region::Usa);
    let sys = guest.connect("set:sys");
    let serial = guest.call(sys, &Request::new(68)).ok().data;
    assert_eq!(&serial[..15], b"XAW10000000009\0");
    let lot = guest.call(sys, &Request::new(67)).ok().data;
    assert_eq!(&lot[..22], b"BHACHZZADA10000000009\0");
}

#[test]
fn eula_versions_round_trip() {
    let guest = guest(Language::AmericanEnglish, Region::Usa);
    let sys = guest.connect("set:sys");
    let entries: Vec<u8> = (0..0x60).map(|i| i as u8).collect();
    let input = guest.buffer_with(&entries);
    guest.call(sys, &Request::new(22).a(input, 0x60)).ok();
    let out = guest.buffer(0x30 * 4);
    assert_eq!(
        guest
            .call(sys, &Request::new(21).b(out, 0x30 * 4))
            .ok()
            .get::<i32>(0),
        2
    );
    assert_eq!(guest.read(out, 0x60), entries);

    let settings = guest.buffer_with(&[7; 0x18]);
    guest.call(sys, &Request::new(32).a(settings, 0x18)).ok();
    assert_eq!(
        guest
            .call(sys, &Request::new(31).b(out, 0x30 * 4))
            .ok()
            .get::<i32>(0),
        1
    );
    assert_eq!(guest.read(out, 0x18), [7; 0x18]);
}
