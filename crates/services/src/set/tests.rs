use std::path::PathBuf;

use lichii_common::settings::{Language, Region};

use super::{INVALID_LANGUAGE, KEY_CODE_MAP_SIZE, NULL_POINTER, key_code_map};
use lichii_common::ResultCode;
use lichii_kernel::testing::{IpcClient, Request};

use crate::testing::{ReplyExt, SWITCH2_TITLE, TempDir, booted, read_value};

/// The settings services on a NAND of their own, which lives as long as the returned directory.
fn boot(language: Language, region: Region) -> (IpcClient, TempDir) {
    let dir = TempDir::new("set");
    (boot_in(&dir, language, region), dir)
}

fn boot_in(dir: &TempDir, language: Language, region: Region) -> IpcClient {
    let mut settings = dir.settings();
    settings.system.language = language;
    settings.system.region = region;
    booted(SWITCH2_TITLE, settings, super::register)
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
    let (client, _dir) = boot(Language::BritishEnglish, Region::Europe);
    let set = client.connect("set").unwrap();
    assert_eq!(
        client
            .send(set, &Request::new(0))
            .unwrap()
            .ok()
            .get::<u64>(0),
        code("en-GB")
    );
    assert_eq!(
        client
            .send(set, &Request::new(4))
            .unwrap()
            .ok()
            .get::<u32>(0),
        2
    );
    assert_eq!(
        client
            .send(set, &Request::new(2).arg(14u32))
            .unwrap()
            .ok()
            .get::<u64>(0),
        code("es-419")
    );
    assert_eq!(
        client
            .send(set, &Request::new(2).arg(20u32))
            .unwrap()
            .result,
        INVALID_LANGUAGE
    );
    assert_eq!(
        client
            .send(set, &Request::new(3))
            .unwrap()
            .ok()
            .get::<i32>(0),
        15
    );
    assert_eq!(
        client
            .send(set, &Request::new(6))
            .unwrap()
            .ok()
            .get::<i32>(0),
        64
    );
    assert_eq!(
        client
            .send(set, &Request::new(8))
            .unwrap()
            .ok()
            .get::<u8>(0),
        0
    );
    assert_eq!(
        client.send(set, &Request::new(10)).unwrap().ok().data(),
        [0; 128]
    );
}

#[test]
fn korea_and_taiwan_share_a_region_code() {
    let (client, _dir) = boot(Language::Korean, Region::Korea);
    let set = client.connect("set").unwrap();
    assert_eq!(
        client
            .send(set, &Request::new(4))
            .unwrap()
            .ok()
            .get::<u32>(0),
        4
    );
}

#[test]
fn available_language_codes_fit_the_buffer() {
    let (client, _dir) = boot(Language::AmericanEnglish, Region::Usa);
    let set = client.connect("set").unwrap();
    let buffer = client.alloc(&vec![0; 0x40 * 8]);
    let reply = client
        .send(set, &Request::new(1).buffer_b(buffer, 0x40 * 8))
        .unwrap()
        .ok();
    assert_eq!(reply.get::<i32>(0), 15);
    let reply = client
        .send(set, &Request::new(5).buffer_b(buffer, 0x40 * 8))
        .unwrap()
        .ok();
    assert_eq!(reply.get::<i32>(0), 20);
    assert_eq!(read_value::<u64>(&client, buffer), code("ja"));
    assert_eq!(read_value::<u64>(&client, buffer + 19 * 8), code("th"));
    let small = client.alloc(&[0; 3 * 8]);
    assert_eq!(
        client
            .send(set, &Request::new(5).buffer_b(small, 3 * 8))
            .unwrap()
            .ok()
            .get::<i32>(0),
        3
    );
}

#[test]
fn key_code_maps_follow_the_language() {
    let (client, _dir) = boot(Language::German, Region::Europe);
    let set = client.connect("set").unwrap();
    let buffer = client.alloc(&vec![0; KEY_CODE_MAP_SIZE]);
    for command in [7, 9, 12] {
        client
            .send(
                set,
                &Request::new(command)
                    .arg(0u32)
                    .buffer_b(buffer, KEY_CODE_MAP_SIZE as u64),
            )
            .unwrap()
            .ok();
        let map = client.read(buffer, KEY_CODE_MAP_SIZE);
        let german = key_code_map(code("de"));
        assert_eq!(&map[..german.len()], german);
        assert!(map[german.len()..].iter().all(|&b| b == 0));
    }
    assert_eq!(
        client.send(set, &Request::new(7)).unwrap().result,
        NULL_POINTER
    );
    assert_eq!(key_code_map(code("ko")), super::key_code_map::KOREAN);
    assert_eq!(
        key_code_map(code("pl")),
        super::key_code_map::ENGLISH_US_INTERNATIONAL
    );
}

#[test]
fn system_settings_update_what_set_reads() {
    let (client, _dir) = boot(Language::AmericanEnglish, Region::Usa);
    let set = client.connect("set").unwrap();
    let sys = client.connect("set:sys").unwrap();
    let nickname = client.alloc(&[0; 0x80]);
    client
        .send(set, &Request::new(11).buffer_b(nickname, 0x80))
        .unwrap()
        .ok();
    assert_eq!(&client.read(nickname, 7), b"Lichii\0");

    client
        .send(sys, &Request::new(0).arg(code("fr")))
        .unwrap()
        .ok();
    client.send(sys, &Request::new(57).arg(0u32)).unwrap().ok();
    client.send(sys, &Request::new(48).arg(1u8)).unwrap().ok();
    let name = client.alloc(b"Switch\0");
    client
        .send(sys, &Request::new(78).buffer_a(name, 7))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(set, &Request::new(0))
            .unwrap()
            .ok()
            .get::<u64>(0),
        code("fr")
    );
    assert_eq!(
        client
            .send(set, &Request::new(4))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0
    );
    assert_eq!(
        client
            .send(set, &Request::new(8))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        client
            .send(sys, &Request::new(47))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    client
        .send(sys, &Request::new(77).buffer_b(nickname, 0x80))
        .unwrap()
        .ok();
    assert_eq!(&client.read(nickname, 7), b"Switch\0");
}

#[test]
fn firmware_version_is_23_0_0() {
    let (client, _dir) = boot(Language::AmericanEnglish, Region::Usa);
    let sys = client.connect("set:sys").unwrap();
    let buffer = client.alloc(&vec![0; 0x100]);
    for command in [3, 4] {
        client
            .send(sys, &Request::new(command).buffer_b(buffer, 0x100))
            .unwrap()
            .ok();
        let version = client.read(buffer, 0x100);
        assert_eq!(version[..8], [23, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&version[0x08..0x0B], b"NX\0");
        assert_eq!(&version[0x68..0x6F], b"23.0.0\0");
        assert!(version[0x80..].starts_with(b"NintendoSDK Firmware for NX 23.0.0-4.0\0"));
    }
}

#[test]
fn settings_items_return_their_values() {
    let (client, _dir) = boot(Language::AmericanEnglish, Region::Usa);
    let sys = client.connect("set:sys").unwrap();
    let category = client.alloc(&name("hbloader"));
    let item = client.alloc(&name("applet_heap_reservation_size"));
    let request = Request::new(37)
        .buffer_x(category, 0x48)
        .buffer_x(item, 0x48);
    assert_eq!(client.send(sys, &request).unwrap().ok().get::<u64>(0), 8);
    let out = client.alloc(&[0; 8]);
    let request = Request::new(38)
        .buffer_b(out, 8)
        .buffer_x(category, 0x48)
        .buffer_x(item, 0x48);
    assert_eq!(client.send(sys, &request).unwrap().ok().get::<u64>(0), 8);
    assert_eq!(read_value::<u64>(&client, out), 0x860_0000);

    let year = client.alloc(&name("standard_user_clock_initial_year"));
    let time = client.alloc(&name("time"));
    let request = Request::new(38)
        .buffer_b(out, 8)
        .buffer_x(time, 0x48)
        .buffer_x(year, 0x48);
    assert_eq!(client.send(sys, &request).unwrap().ok().get::<u64>(0), 4);
    assert_eq!(read_value::<i32>(&client, out), 2023);

    let missing = client.alloc(&name("missing"));
    let reply = client
        .send(
            sys,
            &Request::new(37)
                .buffer_x(time, 0x48)
                .buffer_x(missing, 0x48),
        )
        .unwrap();
    assert_eq!(reply.result, ResultCode(u32::MAX));
}

#[test]
fn system_values_have_sysmodule_defaults_and_round_trip() {
    let (client, _dir) = boot(Language::Japanese, Region::Japan);
    let sys = client.connect("set:sys").unwrap();
    let get = |command| client.send(sys, &Request::new(command)).unwrap().ok();
    assert_eq!(get(29).get::<[u32; 6]>(0), [0x300, 2, 9, 0, 21, 0]);
    assert_eq!(get(35).get::<f32>(0), 1.0);
    let tv = get(39).get::<[u32; 8]>(0);
    assert_eq!(
        (tv[0], tv[2], f32::from_bits(tv[6]), f32::from_bits(tv[7])),
        (0xC, 4, 1.0, 0.5)
    );
    assert_eq!(&get(53).data()[..4], b"UTC\0");
    assert_eq!(get(60).get::<u8>(0), 1);
    assert_eq!(get(62).get::<u8>(0), 0);
    assert_eq!(get(63).get::<u32>(0), 1);
    assert_eq!(get(71).get::<[u32; 3]>(0), [3, 3, 0]);
    assert_eq!(get(75).get::<u32>(0), 0x10001);
    assert_eq!(get(79).get::<u32>(0), 1);
    assert_eq!(&get(90).data(), b"Lichii Author ID");
    assert_eq!(get(136).get::<u32>(0), 0);
    assert_eq!(&get(149).data()[0x20..0x26], b"0.0.0\0");
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
    assert_eq!(get(13).data(), [0; 0x10]);
    assert_eq!(get(15).data(), [0; 0x20]);
    assert_eq!(get(58).data(), [0; 0x20]);
    assert_eq!(get(106).get::<i64>(0), 0);
    assert_eq!(get(150).data(), [0; 0x18]);
    assert_eq!(get(152).data(), [0; 0x18]);
    assert_eq!(get(321).data(), [0; 0x40]);

    let context: [u8; 0x20] = std::array::from_fn(|i| i as u8);
    for (set, get_command) in [(16, 15), (59, 58)] {
        client
            .send(sys, &Request::new(set).arg(context))
            .unwrap()
            .ok();
        assert_eq!(get(get_command).get::<[u8; 0x20]>(0), context);
    }
    client.send(sys, &Request::new(24).arg(1u32)).unwrap().ok();
    assert_eq!(get(23).get::<u32>(0), 1);
    client
        .send(sys, &Request::new(105).arg(-5i64))
        .unwrap()
        .ok();
    assert_eq!(get(106).get::<i64>(0), -5);
    client
        .send(sys, &Request::new(44).arg(1u32).arg(2u32))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(sys, &Request::new(43).arg(1u32))
            .unwrap()
            .ok()
            .get::<u32>(0),
        2
    );
    assert_eq!(
        client
            .send(sys, &Request::new(43).arg(0u32))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0
    );
}

#[test]
fn serial_numbers_carry_the_region_and_a_check_digit() {
    let (client, _dir) = boot(Language::AmericanEnglish, Region::Usa);
    let sys = client.connect("set:sys").unwrap();
    let serial = client
        .send(sys, &Request::new(68))
        .unwrap()
        .ok()
        .data()
        .to_vec();
    assert_eq!(&serial[..15], b"XAW10000000009\0");
    let lot = client
        .send(sys, &Request::new(67))
        .unwrap()
        .ok()
        .data()
        .to_vec();
    assert_eq!(&lot[..22], b"BHACHZZADA10000000009\0");
}

#[test]
fn eula_versions_round_trip() {
    let (client, _dir) = boot(Language::AmericanEnglish, Region::Usa);
    let sys = client.connect("set:sys").unwrap();
    let entries: Vec<u8> = (0..0x60).map(|i| i as u8).collect();
    let input = client.alloc(&entries);
    client
        .send(sys, &Request::new(22).buffer_a(input, 0x60))
        .unwrap()
        .ok();
    let out = client.alloc(&[0; 0x30 * 4]);
    assert_eq!(
        client
            .send(sys, &Request::new(21).buffer_b(out, 0x30 * 4))
            .unwrap()
            .ok()
            .get::<i32>(0),
        2
    );
    assert_eq!(client.read(out, 0x60), entries);

    let settings = client.alloc(&[7; 0x18]);
    client
        .send(sys, &Request::new(32).buffer_a(settings, 0x18))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(sys, &Request::new(31).buffer_b(out, 0x30 * 4))
            .unwrap()
            .ok()
            .get::<i32>(0),
        1
    );
    assert_eq!(client.read(out, 0x18), [7; 0x18]);
}

fn settings_file(dir: &TempDir, save: &str) -> PathBuf {
    dir.settings()
        .paths
        .nand_dir
        .unwrap()
        .join("system/save")
        .join(save)
        .join("settings.dat")
}

fn location(name: &str) -> [u8; 0x24] {
    let mut location = [0; 0x24];
    location[..name.len()].copy_from_slice(name.as_bytes());
    location
}

#[test]
fn system_values_survive_a_reboot() {
    let dir = TempDir::new("set");
    let launch: [u32; 8] = [0x10001, 0, 5, 6, 7, 8, 9, 10];
    {
        let client = boot_in(&dir, Language::AmericanEnglish, Region::Usa);
        let sys = client.connect("set:sys").unwrap();
        for request in [
            Request::new(8).arg(1u8),
            Request::new(14).arg([5u8; 16]),
            Request::new(24).arg(1u32),
            Request::new(44).arg(3u32).arg(2u32),
            Request::new(48).arg(1u8),
            Request::new(54).arg(location("Europe/Amsterdam")),
            Request::new(76).arg(launch),
            Request::new(105).arg(-7i64),
        ] {
            client.send(sys, &request).unwrap().ok();
        }
        let name = client.alloc(b"Switch\0");
        client
            .send(sys, &Request::new(78).buffer_a(name, 7))
            .unwrap()
            .ok();
        let eula = client.alloc(&[3; 0x30]);
        client
            .send(sys, &Request::new(22).buffer_a(eula, 0x30))
            .unwrap()
            .ok();
    }

    let system = std::fs::read(settings_file(&dir, "8000000000000050")).unwrap();
    assert_eq!(system.len(), 0x10 + 0x336a0);
    assert_eq!(&system[..12], b"yuzu_set\x04\0\0\0");
    assert_eq!(system[0x10..0x18], [0, 0, 0x14, 0, 7, 0, 0, 0]);
    assert_eq!(system[0x10 + 0x29470], 1);
    assert_eq!(system[0x10 + 0x2a950..][..7], *b"Switch\0");
    let private = std::fs::read(settings_file(&dir, "8000000000000052")).unwrap();
    assert_eq!(private.len(), 0x10 + 0xd8);
    assert_eq!(private[0x10 + 0x50..][..16], [5; 16]);

    let client = boot_in(&dir, Language::AmericanEnglish, Region::Usa);
    let sys = client.connect("set:sys").unwrap();
    let get = |command: u32| client.send(sys, &Request::new(command)).unwrap().ok();
    assert_eq!(get(7).get::<u8>(0), 1);
    assert_eq!(get(13).get::<[u8; 16]>(0), [5; 16]);
    assert_eq!(get(23).get::<u32>(0), 1);
    assert_eq!(get(47).get::<u8>(0), 1);
    assert_eq!(get(53).get::<[u8; 0x24]>(0), location("Europe/Amsterdam"));
    assert_eq!(get(75).get::<[u32; 8]>(0), launch);
    assert_eq!(get(106).get::<i64>(0), -7);
    let mode = client
        .send(sys, &Request::new(43).arg(3u32))
        .unwrap()
        .ok()
        .get::<u32>(0);
    assert_eq!(mode, 2);
    let out = client.alloc(&[0; 0x80]);
    client
        .send(sys, &Request::new(77).buffer_b(out, 0x80))
        .unwrap()
        .ok();
    assert_eq!(&client.read(out, 7), b"Switch\0");
    let count = client
        .send(sys, &Request::new(21).buffer_b(out, 0x60))
        .unwrap()
        .ok()
        .get::<i32>(0);
    assert_eq!((count, client.read(out, 0x30)), (1, vec![3; 0x30]));
    let set = client.connect("set").unwrap();
    assert_eq!(
        client
            .send(set, &Request::new(8))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
}

#[test]
fn settings_files_in_the_cpp_layout_are_read_and_kept() {
    let dir = TempDir::new("set");
    let mut system = b"yuzu_set\x04\0\0\0".to_vec();
    system.resize(0x10 + 0x336a0, 0);
    let mut put = |offset: usize, bytes: &[u8]| {
        system[0x10 + offset..][..bytes.len()].copy_from_slice(bytes);
    };
    put(0x99b0, &8u32.to_le_bytes());
    put(0x29478, &1u32.to_le_bytes());
    put(0x29a40, &0.25f32.to_le_bytes());
    put(0x2a140, &99i32.to_le_bytes());
    put(0x100, b"not served");
    let path = settings_file(&dir, "8000000000000050");
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &system).unwrap();
    let private = settings_file(&dir, "8000000000000052");
    std::fs::create_dir_all(private.parent().unwrap()).unwrap();
    std::fs::write(&private, [0xff; 0x40]).unwrap();

    let client = boot_in(&dir, Language::AmericanEnglish, Region::Usa);
    let sys = client.connect("set:sys").unwrap();
    let get = |command: u32| client.send(sys, &Request::new(command)).unwrap().ok();
    assert_eq!(get(136).get::<u32>(0), 8);
    assert_eq!(get(23).get::<u32>(0), 1);
    assert_eq!(get(35).get::<f32>(0), 0.25);
    assert_eq!(get(13).get::<[u8; 16]>(0), [0; 16]);
    let out = client.alloc(&[0; 0x30 * 40]);
    let count = client
        .send(sys, &Request::new(21).buffer_b(out, 0x30 * 40))
        .unwrap()
        .ok()
        .get::<i32>(0);
    assert_eq!(count, 32);

    client.send(sys, &Request::new(8).arg(1u8)).unwrap().ok();
    let saved = std::fs::read(&path).unwrap();
    assert_eq!(&saved[0x10 + 0x100..][..10], b"not served");
    assert_eq!(saved[0x10 + 0x29470], 1);
    assert_eq!(std::fs::read(&private).unwrap().len(), 0x10 + 0xd8);
}
