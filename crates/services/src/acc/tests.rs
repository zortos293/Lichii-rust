use super::{
    APPLICATION_INFO_ALREADY_INITIALIZED, AVATAR, INVALID_ARRAY_LENGTH, INVALID_USER_ID, UNKNOWN,
    account_id, user_id,
};
use lichii_common::settings::Settings;
use lichii_kernel::testing::{IpcClient, Request};

use crate::testing::{ReplyExt, SWITCH2_TITLE, booted};

fn boot(username: &str) -> IpcClient {
    let mut settings = Settings::default();
    settings.system.username = username.into();
    booted(SWITCH2_TITLE, settings, super::register)
}

#[test]
fn user_ids_are_stable_per_name() {
    assert_eq!(user_id("lichii"), user_id("lichii"));
    assert_ne!(user_id("lichii"), user_id("player"));
    assert_ne!(user_id(""), [0; 16]);
}

#[test]
fn the_one_user_is_open_and_selected() {
    let client = boot("player");
    let acc = client.connect("acc:u0").unwrap();
    let uid = user_id("player");
    assert_eq!(
        client
            .send(acc, &Request::new(0))
            .unwrap()
            .ok()
            .get::<u32>(0),
        1
    );
    assert_eq!(
        client
            .send(acc, &Request::new(1).arg(uid))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        client
            .send(acc, &Request::new(1).arg([1u8; 16]))
            .unwrap()
            .ok()
            .get::<u8>(0),
        0
    );
    let list = client.alloc(&[0; 0x80]);
    for command in [2, 3, 141] {
        client
            .send(acc, &Request::new(command).buffer_b(list, 0x80))
            .unwrap()
            .ok();
        let users = client.read(list, 0x80);
        assert_eq!(
            (&users[..16], &users[16..]),
            (&uid[..], &[0; 0x70][..]),
            "command {command}"
        );
    }
    for command in [4, 51, 52] {
        assert_eq!(
            client
                .send(acc, &Request::new(command))
                .unwrap()
                .ok()
                .get::<[u8; 16]>(0),
            uid
        );
    }
    assert_eq!(
        client
            .send(acc, &Request::new(50))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        client
            .send(acc, &Request::new(150))
            .unwrap()
            .ok()
            .get::<u8>(0),
        0
    );
}

#[test]
fn profiles_carry_the_username_and_avatar() {
    let client = boot("player");
    let acc = client.connect("acc:u0").unwrap();
    let uid = user_id("player");
    let profile = client
        .send(acc, &Request::new(5).arg(uid))
        .unwrap()
        .ok()
        .interface();
    let user_data = client.alloc(&[0; 0x80]);
    let base = client
        .send(profile, &Request::new(0).buffer_b(user_data, 0x80))
        .unwrap()
        .ok();
    assert_eq!(base.data().len(), 0x38);
    assert_eq!(base.get::<[u8; 16]>(0), uid);
    assert_eq!(&base.data()[0x18..0x1F], b"player\0");
    assert_eq!(
        client.send(profile, &Request::new(1)).unwrap().ok().data(),
        base.data()
    );
    assert_eq!(
        client
            .send(profile, &Request::new(10))
            .unwrap()
            .ok()
            .get::<u32>(0),
        AVATAR.len() as u32
    );
    let image = client.alloc(&vec![0; 0x2000]);
    let reply = client
        .send(profile, &Request::new(11).buffer_b(image, 0x2000))
        .unwrap()
        .ok();
    assert_eq!(reply.get::<u32>(0), AVATAR.len() as u32);
    assert_eq!(client.read(image, AVATAR.len()), AVATAR);
    assert_eq!(&AVATAR[..2], [0xFF, 0xD8]);
    for command in [20, 21, 30] {
        client.send(profile, &Request::new(command)).unwrap().ok();
    }

    let stranger = client
        .send(acc, &Request::new(5).arg([7u8; 16]))
        .unwrap()
        .ok()
        .interface();
    assert_eq!(
        client.send(stranger, &Request::new(1)).unwrap().result,
        UNKNOWN
    );
}

#[test]
fn the_profile_editor_stores_profiles() {
    let client = boot("player");
    let su = client.connect("acc:su").unwrap();
    let uid = user_id("player");
    let editor = client
        .send(su, &Request::new(205).arg(uid))
        .unwrap()
        .ok()
        .interface();
    let mut base = [0u8; 0x38];
    base[..16].copy_from_slice(&uid);
    base[0x18..0x1D].copy_from_slice(b"Renan");
    let data = client.alloc(&[5; 0x80]);
    client
        .send(editor, &Request::new(100).arg(base).buffer_a(data, 0x80))
        .unwrap()
        .ok();
    let short = client.alloc(&[5; 0x10]);
    let reply = client
        .send(editor, &Request::new(100).arg(base).buffer_a(short, 0x10))
        .unwrap();
    assert_eq!(reply.result, INVALID_ARRAY_LENGTH);

    let acc = client.connect("acc:u0").unwrap();
    let profile = client
        .send(acc, &Request::new(5).arg(uid))
        .unwrap()
        .ok()
        .interface();
    let user_data = client.alloc(&[0; 0x80]);
    let reply = client
        .send(profile, &Request::new(0).buffer_b(user_data, 0x80))
        .unwrap()
        .ok();
    assert_eq!(reply.get::<[u8; 0x38]>(0), base);
    assert_eq!(client.read(user_data, 0x80), [5; 0x80]);
    let mut other = base;
    other[0x18..0x1D].copy_from_slice(b"Other");
    client
        .send(profile, &Request::new(100).arg(other).buffer_a(data, 0x80))
        .unwrap();
    let reply = client.send(profile, &Request::new(1)).unwrap().ok();
    assert_eq!(reply.get::<[u8; 0x38]>(0), base, "IProfile has no Store");
    assert_eq!(
        client
            .send(su, &Request::new(401))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0
    );
    let admin = client
        .send(su, &Request::new(250).arg(uid))
        .unwrap()
        .ok()
        .interface();
    assert_eq!(
        client
            .send(admin, &Request::new(250))
            .unwrap()
            .ok()
            .get::<u8>(0),
        0
    );
}

#[test]
fn application_info_initializes_once() {
    let client = boot("lichii");
    let acc = client.connect("acc:u0").unwrap();
    client
        .send(acc, &Request::new(100).send_pid())
        .unwrap()
        .ok();
    let reply = client.send(acc, &Request::new(140).send_pid()).unwrap();
    assert_eq!(reply.result, APPLICATION_INFO_ALREADY_INITIALIZED);
    client
        .send(acc, &Request::new(160).send_pid())
        .unwrap()
        .ok();
}

#[test]
fn the_account_manager_reports_an_unlinked_account() {
    let client = boot("lichii");
    let acc = client.connect("acc:u0").unwrap();
    let uid = user_id("lichii");
    let manager = client
        .send(acc, &Request::new(101))
        .unwrap()
        .ok()
        .interface();
    client.send(manager, &Request::new(0)).unwrap().ok();
    assert_eq!(
        client
            .send(manager, &Request::new(1))
            .unwrap()
            .ok()
            .get::<u64>(0),
        account_id(&uid)
    );
    let token = client.alloc(&[0xAA; 0x100]);
    let reply = client
        .send(manager, &Request::new(4).buffer_b(token, 0x100))
        .unwrap()
        .ok();
    assert_eq!(
        (reply.get::<u32>(0), client.read(token, 0x100)),
        (0x100, vec![0; 0x100])
    );
    assert_eq!(
        client
            .send(manager, &Request::new(3))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0
    );
    let cache = client.alloc(&[0xAA; 0x68]);
    let reply = client
        .send(manager, &Request::new(130).buffer_b(cache, 0x68))
        .unwrap()
        .ok();
    assert_eq!(
        (reply.get::<u64>(0), client.read(cache, 0x68)),
        (account_id(&uid), vec![0; 0x68])
    );

    let async_context = client
        .send(manager, &Request::new(2))
        .unwrap()
        .ok()
        .interface();
    assert_eq!(
        client
            .send(async_context, &Request::new(0))
            .unwrap()
            .ok()
            .copy_handles
            .len(),
        1
    );
    assert_eq!(
        client
            .send(async_context, &Request::new(2))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    client.send(async_context, &Request::new(3)).unwrap().ok();
    for command in [150, 170] {
        client
            .send(manager, &Request::new(command))
            .unwrap()
            .ok()
            .interface();
    }

    let list = client.alloc(&[0; 0x80]);
    client
        .send(acc, &Request::new(131).buffer_b(list, 0x80))
        .unwrap()
        .ok();
    assert_eq!(client.read(list, 16), [0; 16]);
    client.send(manager, &Request::new(160)).unwrap().ok();
    client
        .send(acc, &Request::new(131).buffer_b(list, 0x80))
        .unwrap()
        .ok();
    assert_eq!(client.read(list, 16), uid);
}

#[test]
fn system_services_get_their_own_manager() {
    let client = boot("lichii");
    let acc = client.connect("acc:u1").unwrap();
    let uid = user_id("lichii");
    let manager = client
        .send(acc, &Request::new(102).arg(uid))
        .unwrap()
        .ok()
        .interface();
    assert_eq!(
        client
            .send(manager, &Request::new(1))
            .unwrap()
            .ok()
            .get::<u64>(0),
        account_id(&uid)
    );
    let license = client.send(manager, &Request::new(143)).unwrap().ok();
    assert_eq!(license.data(), [0; 16]);
    let list = client.alloc(&[0; 0x80]);
    client
        .send(acc, &Request::new(140).buffer_b(list, 0x80))
        .unwrap()
        .ok();
    assert_eq!(client.read(list, 16), uid);
}

#[test]
fn thumbnails_need_a_user_and_the_full_size() {
    let client = boot("lichii");
    let acc = client.connect("acc:u0").unwrap();
    let uid = user_id("lichii");
    let thumbnail = client.alloc(&vec![0; 0x24000]);
    client
        .send(
            acc,
            &Request::new(110).arg(uid).buffer_a(thumbnail, 0x24000),
        )
        .unwrap()
        .ok();
    let reply = client
        .send(
            acc,
            &Request::new(110)
                .arg([0u8; 16])
                .buffer_a(thumbnail, 0x24000),
        )
        .unwrap();
    assert_eq!(reply.result, INVALID_USER_ID);
    let reply = client
        .send(acc, &Request::new(110).arg(uid).buffer_a(thumbnail, 0x100))
        .unwrap();
    assert_eq!(reply.result, INVALID_ARRAY_LENGTH);
}
