use lichii_common::settings::Settings;
use lichii_fs::vfs::MemFile;
use lichii_loader::program::AddOnContent;

use super::lr;
use super::*;
use crate::testing::{self, Guest, PROGRAM_ID, Request};

const ADD_ON: u64 = PROGRAM_ID + 0x1001;

fn guest() -> Guest {
    let mut image = testing::image();
    image.add_ons = vec![AddOnContent {
        title_id: ADD_ON,
        version: 0x10000,
        romfs: MemFile::from_bytes(vec![0u8; 16]),
    }];
    Guest::new(image, Settings::default(), |context| {
        register(context).unwrap()
    })
}

fn key(id: u64, version: u32, meta_type: u8) -> [u8; 16] {
    let mut key = [0u8; 16];
    key[..8].copy_from_slice(&id.to_le_bytes());
    key[8..12].copy_from_slice(&version.to_le_bytes());
    key[12] = meta_type;
    key
}

#[test]
fn content_meta_database_knows_the_title_and_its_add_ons() {
    let guest = guest();
    let ncm = guest.service("ncm");
    let db = guest.call(ncm, Request::new(5).arg(5u8)).ok().interface();
    let has = |key: [u8; 16]| guest.call(db, Request::new(8).arg(key)).ok().at::<u8>(0);
    assert_eq!(has(key(PROGRAM_ID, 0, 0x80)), 1);
    assert_eq!(has(key(ADD_ON, 0, 0x82)), 1);
    let other = key(0x0100_0000_0009_0000, 1, 0x80);
    assert_eq!(has(other), 0);
    guest.call(db, Request::new(0).arg(other)).ok();
    assert_eq!(has(other), 1);
    guest.call(db, Request::new(2).arg(other)).ok();
    assert_eq!(has(other), 0);
    guest.call(db, Request::new(15)).ok();

    let latest = guest.call(db, Request::new(6).arg(ADD_ON)).ok();
    assert_eq!(latest.at::<[u8; 16]>(0), key(ADD_ON, 0x10000, 0x82));
    let latest = guest.call(db, Request::new(6).arg(PROGRAM_ID)).ok();
    assert_eq!(latest.at::<[u8; 16]>(0), key(PROGRAM_ID, 0, 0x80));
    assert_eq!(
        guest.call(db, Request::new(6).arg(1u64)).result,
        CONTENT_META_NOT_FOUND
    );
}

#[test]
fn content_storage_has_no_placeholder_cache() {
    let guest = guest();
    let ncm = guest.service("ncm");
    let storage = guest.call(ncm, Request::new(4).arg(4u8)).ok().interface();
    let first = guest.call(storage, Request::new(0)).ok().at::<[u8; 16]>(0);
    let second = guest.call(storage, Request::new(0)).ok().at::<[u8; 16]>(0);
    assert_ne!(first, second);
    let create = guest.call(storage, Request::new(1).arg(first).arg(first).arg(16i64));
    assert_eq!(create.result, RESULT_UNKNOWN);
    guest.service("ncm:v");
}

fn path(guest: &Guest, address: u64) -> String {
    let bytes = guest.read(address, 0x300);
    let len = bytes.iter().position(|&b| b == 0).unwrap();
    String::from_utf8(bytes[..len].to_vec()).unwrap()
}

#[test]
fn location_resolver_resolves_and_redirects() {
    let guest = guest();
    let lr = guest.service("lr");
    let resolver = guest.call(lr, Request::new(0).arg(1u8)).ok().interface();
    let out = guest.heap;
    let resolve = |command: u32, id: u64| {
        guest.call(resolver, Request::new(command).arg(id).output(out, 0x300))
    };
    for (command, file) in [
        (0, "program"),
        (2, "control"),
        (3, "htmldocument"),
        (7, "legalinformation"),
        (16, "program"),
    ] {
        resolve(command, PROGRAM_ID).ok();
        assert_eq!(
            path(&guest, out),
            format!("@Host:/{PROGRAM_ID:016X}/{file}")
        );
    }
    resolve(4, ADD_ON).ok();
    assert_eq!(path(&guest, out), format!("@Host:/{ADD_ON:016X}/data"));
    assert_eq!(resolve(0, 7).result, lr::PROGRAM_NOT_FOUND);
    assert_eq!(resolve(4, PROGRAM_ID).result, lr::DATA_NOT_FOUND);

    let input = guest.heap + 0x1000;
    guest.write(input, b"@Sd:/moved\0");
    guest
        .call(resolver, Request::new(1).arg(7u64).input(input, 0x300))
        .ok();
    resolve(0, 7).ok();
    assert_eq!(path(&guest, out), "@Sd:/moved");
    guest.call(resolver, Request::new(12).arg(7u64)).ok();
    assert_eq!(resolve(0, 7).result, lr::PROGRAM_NOT_FOUND);
    guest
        .call(
            resolver,
            Request::new(5).arg(PROGRAM_ID).input(input, 0x300),
        )
        .ok();
    resolve(2, PROGRAM_ID).ok();
    assert_eq!(path(&guest, out), "@Sd:/moved");
    guest.call(resolver, Request::new(11)).ok();
    resolve(2, PROGRAM_ID).ok();
    assert_eq!(
        path(&guest, out),
        format!("@Host:/{PROGRAM_ID:016X}/control")
    );
    guest.call(lr, Request::new(2).arg(1u8)).ok();
}

#[test]
fn registered_and_add_on_resolvers() {
    let guest = guest();
    let lr = guest.service("lr");
    let registered = guest.call(lr, Request::new(1)).ok().interface();
    let (input, out) = (guest.heap, guest.heap + 0x1000);
    let resolve = |command| {
        guest.call(
            registered,
            Request::new(command).arg(PROGRAM_ID).output(out, 0x300),
        )
    };
    assert_eq!(resolve(0).result, lr::PROGRAM_NOT_FOUND);
    assert_eq!(resolve(4).result, lr::HTML_DOCUMENT_NOT_FOUND);
    guest.write(input, b"@Host:/registered\0");
    guest
        .call(
            registered,
            Request::new(1).arg(PROGRAM_ID).input(input, 0x300),
        )
        .ok();
    resolve(0).ok();
    assert_eq!(path(&guest, out), "@Host:/registered");
    guest.call(registered, Request::new(2).arg(PROGRAM_ID)).ok();
    assert_eq!(resolve(0).result, lr::PROGRAM_NOT_FOUND);

    let add_ons = guest.call(lr, Request::new(3)).ok().interface();
    guest
        .call(add_ons, Request::new(0).arg(ADD_ON).output(out, 0x300))
        .ok();
    assert_eq!(path(&guest, out), format!("@Host:/{ADD_ON:016X}/data"));
    let missing = guest.call(add_ons, Request::new(0).arg(PROGRAM_ID).output(out, 0x300));
    assert_eq!(missing.result, lr::ADD_ON_CONTENT_NOT_FOUND);
}
