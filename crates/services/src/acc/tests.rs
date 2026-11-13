use lichii_common::ConsoleGeneration;

use super::{
    APPLICATION_INFO_ALREADY_INITIALIZED, AVATAR, INVALID_ARRAY_LENGTH, INVALID_USER_ID, UNKNOWN,
    account_id, user_id,
};
use crate::testing::{Guest, Request, settings};

fn guest(username: &str) -> Guest {
    let mut settings = settings();
    settings.system.username = username.into();
    Guest::new(ConsoleGeneration::Switch2, settings, super::register)
}

#[test]
fn user_ids_are_stable_per_name() {
    assert_eq!(user_id("lichii"), user_id("lichii"));
    assert_ne!(user_id("lichii"), user_id("player"));
    assert_ne!(user_id(""), [0; 16]);
}

#[test]
fn the_one_user_is_open_and_selected() {
    let guest = guest("player");
    let acc = guest.connect("acc:u0");
    let uid = user_id("player");
    assert_eq!(guest.call(acc, &Request::new(0)).ok().get::<u32>(0), 1);
    assert_eq!(
        guest.call(acc, &Request::new(1).arg(uid)).ok().get::<u8>(0),
        1
    );
    assert_eq!(
        guest
            .call(acc, &Request::new(1).arg([1u8; 16]))
            .ok()
            .get::<u8>(0),
        0
    );
    let list = guest.buffer(0x80);
    for command in [2, 3, 141] {
        guest.call(acc, &Request::new(command).b(list, 0x80)).ok();
        let users = guest.read(list, 0x80);
        assert_eq!(
            (&users[..16], &users[16..]),
            (&uid[..], &[0; 0x70][..]),
            "command {command}"
        );
    }
    for command in [4, 51, 52] {
        assert_eq!(
            guest
                .call(acc, &Request::new(command))
                .ok()
                .get::<[u8; 16]>(0),
            uid
        );
    }
    assert_eq!(guest.call(acc, &Request::new(50)).ok().get::<u8>(0), 1);
    assert_eq!(guest.call(acc, &Request::new(150)).ok().get::<u8>(0), 0);
}

#[test]
fn profiles_carry_the_username_and_avatar() {
    let guest = guest("player");
    let acc = guest.connect("acc:u0");
    let uid = user_id("player");
    let profile = guest.call(acc, &Request::new(5).arg(uid)).ok().interface();
    let user_data = guest.buffer(0x80);
    let base = guest
        .call(profile, &Request::new(0).b(user_data, 0x80))
        .ok();
    assert_eq!(base.data.len(), 0x38);
    assert_eq!(base.get::<[u8; 16]>(0), uid);
    assert_eq!(&base.data[0x18..0x1F], b"player\0");
    assert_eq!(guest.call(profile, &Request::new(1)).ok().data, base.data);
    assert_eq!(
        guest.call(profile, &Request::new(10)).ok().get::<u32>(0),
        AVATAR.len() as u32
    );
    let image = guest.buffer(0x2000);
    let reply = guest.call(profile, &Request::new(11).b(image, 0x2000)).ok();
    assert_eq!(reply.get::<u32>(0), AVATAR.len() as u32);
    assert_eq!(guest.read(image, AVATAR.len()), AVATAR);
    assert_eq!(&AVATAR[..2], [0xFF, 0xD8]);
    for command in [20, 21, 30] {
        guest.call(profile, &Request::new(command)).ok();
    }

    let stranger = guest
        .call(acc, &Request::new(5).arg([7u8; 16]))
        .ok()
        .interface();
    assert_eq!(guest.call(stranger, &Request::new(1)).result, UNKNOWN.0);
}

#[test]
fn the_profile_editor_stores_profiles() {
    let guest = guest("player");
    let su = guest.connect("acc:su");
    let uid = user_id("player");
    let editor = guest.call(su, &Request::new(205).arg(uid)).ok().interface();
    let mut base = [0u8; 0x38];
    base[..16].copy_from_slice(&uid);
    base[0x18..0x1D].copy_from_slice(b"Renan");
    let data = guest.buffer_with(&[5; 0x80]);
    guest
        .call(editor, &Request::new(100).arg(base).a(data, 0x80))
        .ok();
    let short = guest.buffer_with(&[5; 0x10]);
    let reply = guest.call(editor, &Request::new(100).arg(base).a(short, 0x10));
    assert_eq!(reply.result, INVALID_ARRAY_LENGTH.0);

    let acc = guest.connect("acc:u0");
    let profile = guest.call(acc, &Request::new(5).arg(uid)).ok().interface();
    let user_data = guest.buffer(0x80);
    let reply = guest
        .call(profile, &Request::new(0).b(user_data, 0x80))
        .ok();
    assert_eq!(reply.get::<[u8; 0x38]>(0), base);
    assert_eq!(guest.read(user_data, 0x80), [5; 0x80]);
    assert_ne!(guest.call(profile, &Request::new(100).arg(base)).result, 0);
    assert_eq!(guest.call(su, &Request::new(401)).ok().get::<u32>(0), 0);
    let admin = guest.call(su, &Request::new(250).arg(uid)).ok().interface();
    assert_eq!(guest.call(admin, &Request::new(250)).ok().get::<u8>(0), 0);
}

#[test]
fn application_info_initializes_once() {
    let guest = guest("lichii");
    let acc = guest.connect("acc:u0");
    guest.call(acc, &Request::new(100).pid()).ok();
    let reply = guest.call(acc, &Request::new(140).pid());
    assert_eq!(reply.result, APPLICATION_INFO_ALREADY_INITIALIZED.0);
    guest.call(acc, &Request::new(160).pid()).ok();
}

#[test]
fn the_account_manager_reports_an_unlinked_account() {
    let guest = guest("lichii");
    let acc = guest.connect("acc:u0");
    let uid = user_id("lichii");
    let manager = guest.call(acc, &Request::new(101)).ok().interface();
    guest.call(manager, &Request::new(0)).ok();
    assert_eq!(
        guest.call(manager, &Request::new(1)).ok().get::<u64>(0),
        account_id(&uid)
    );
    let token = guest.buffer_with(&[0xAA; 0x100]);
    let reply = guest.call(manager, &Request::new(4).b(token, 0x100)).ok();
    assert_eq!(
        (reply.get::<u32>(0), guest.read(token, 0x100)),
        (0x100, vec![0; 0x100])
    );
    assert_eq!(guest.call(manager, &Request::new(3)).ok().get::<u32>(0), 0);
    let cache = guest.buffer_with(&[0xAA; 0x68]);
    let reply = guest.call(manager, &Request::new(130).b(cache, 0x68)).ok();
    assert_eq!(
        (reply.get::<u64>(0), guest.read(cache, 0x68)),
        (account_id(&uid), vec![0; 0x68])
    );

    let async_context = guest.call(manager, &Request::new(2)).ok().interface();
    assert_eq!(
        guest
            .call(async_context, &Request::new(0))
            .ok()
            .copies
            .len(),
        1
    );
    assert_eq!(
        guest
            .call(async_context, &Request::new(2))
            .ok()
            .get::<u8>(0),
        1
    );
    guest.call(async_context, &Request::new(3)).ok();
    for command in [150, 170] {
        guest.call(manager, &Request::new(command)).ok().interface();
    }

    let list = guest.buffer(0x80);
    guest.call(acc, &Request::new(131).b(list, 0x80)).ok();
    assert_eq!(guest.read(list, 16), [0; 16]);
    guest.call(manager, &Request::new(160)).ok();
    guest.call(acc, &Request::new(131).b(list, 0x80)).ok();
    assert_eq!(guest.read(list, 16), uid);
}

#[test]
fn system_services_get_their_own_manager() {
    let guest = guest("lichii");
    let acc = guest.connect("acc:u1");
    let uid = user_id("lichii");
    let manager = guest
        .call(acc, &Request::new(102).arg(uid))
        .ok()
        .interface();
    assert_eq!(
        guest.call(manager, &Request::new(1)).ok().get::<u64>(0),
        account_id(&uid)
    );
    let license = guest.call(manager, &Request::new(143)).ok();
    assert_eq!(license.data, [0; 16]);
    let list = guest.buffer(0x80);
    guest.call(acc, &Request::new(140).b(list, 0x80)).ok();
    assert_eq!(guest.read(list, 16), uid);
}

#[test]
fn thumbnails_need_a_user_and_the_full_size() {
    let guest = guest("lichii");
    let acc = guest.connect("acc:u0");
    let uid = user_id("lichii");
    let thumbnail = guest.buffer(0x24000);
    guest
        .call(acc, &Request::new(110).arg(uid).a(thumbnail, 0x24000))
        .ok();
    let reply = guest.call(acc, &Request::new(110).arg([0u8; 16]).a(thumbnail, 0x24000));
    assert_eq!(reply.result, INVALID_USER_ID.0);
    let reply = guest.call(acc, &Request::new(110).arg(uid).a(thumbnail, 0x100));
    assert_eq!(reply.result, INVALID_ARRAY_LENGTH.0);
}
