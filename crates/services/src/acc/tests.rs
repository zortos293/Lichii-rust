use super::{
    APPLICATION_INFO_ALREADY_INITIALIZED, AVATAR, INVALID_ARRAY_LENGTH, INVALID_USER_ID, Profile,
    UNKNOWN, account_id, user_id,
};
use lichii_kernel::testing::{IpcClient, Request};

use crate::testing::{ReplyExt, SWITCH2_TITLE, TempDir, booted};

/// The account services on a NAND of their own, which lives as long as the returned directory.
fn boot(username: &str) -> (IpcClient, TempDir) {
    let dir = TempDir::new("acc");
    (boot_in(&dir, username), dir)
}

fn boot_in(dir: &TempDir, username: &str) -> IpcClient {
    let mut settings = dir.settings();
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
    let (client, _dir) = boot("player");
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
    let (client, _dir) = boot("player");
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
    let (client, _dir) = boot("player");
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
    let (client, _dir) = boot("lichii");
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
    let (client, _dir) = boot("lichii");
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
    let (client, _dir) = boot("lichii");
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
    let (client, _dir) = boot("lichii");
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

fn profiles_file(dir: &TempDir) -> std::path::PathBuf {
    dir.settings()
        .paths
        .nand_dir
        .unwrap()
        .join("system/save/8000000000000010/su/avators/profiles.dat")
}

/// A `UserRaw` of `profiles.dat`.
fn user_raw(uid: [u8; 16], nickname: &str, timestamp: u64, data: u8) -> Vec<u8> {
    let mut user = vec![0; 0xc8];
    user[..16].copy_from_slice(&uid);
    user[16..32].copy_from_slice(&uid);
    user[32..40].copy_from_slice(&timestamp.to_le_bytes());
    user[40..40 + nickname.len()].copy_from_slice(nickname.as_bytes());
    user[72..].fill(data);
    user
}

fn base(client: &IpcClient, uid: [u8; 16]) -> ([u8; 0x38], Vec<u8>) {
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
    (reply.get(0), client.read(user_data, 0x80))
}

#[test]
fn the_profile_survives_a_reboot() {
    let dir = TempDir::new("acc");
    let uid = user_id("player");
    let mut edited = [0u8; 0x38];
    edited[..16].copy_from_slice(&uid);
    edited[0x10..0x18].copy_from_slice(&99u64.to_le_bytes());
    edited[0x18..0x1d].copy_from_slice(b"Renan");
    {
        let client = boot_in(&dir, "player");
        let file = std::fs::read(profiles_file(&dir)).unwrap();
        assert_eq!(file.len(), 0x650);
        assert_eq!(file[0x10..0x10 + 0xc8], user_raw(uid, "player", 0, 0));
        assert!(file[0x10 + 0xc8..].iter().all(|&b| b == 0));

        let su = client.connect("acc:su").unwrap();
        let editor = client
            .send(su, &Request::new(205).arg(uid))
            .unwrap()
            .ok()
            .interface();
        let data = client.alloc(&[5; 0x80]);
        client
            .send(editor, &Request::new(100).arg(edited).buffer_a(data, 0x80))
            .unwrap()
            .ok();
    }
    let client = boot_in(&dir, "player");
    assert_eq!(base(&client, uid), (edited, vec![5; 0x80]));
    let file = std::fs::read(profiles_file(&dir)).unwrap();
    assert_eq!(file[0x10 + 16..0x10 + 32], uid);
}

#[test]
fn users_of_a_cpp_profiles_file_are_opened_by_nickname_and_kept() {
    let dir = TempDir::new("acc");
    let eden = [0x42; 16];
    let other = [0x24; 16];
    let mut file = vec![0; 0x10];
    file.extend(user_raw(other, "Other", 7, 1));
    file.extend(user_raw(eden, "Eden", 0x1234, 2));
    file.resize(0x650, 0);
    let path = profiles_file(&dir);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &file).unwrap();

    let client = boot_in(&dir, "Eden");
    let acc = client.connect("acc:u0").unwrap();
    let last = client.send(acc, &Request::new(4)).unwrap().ok();
    assert_eq!(last.get::<[u8; 16]>(0), eden);
    let (eden_base, user_data) = base(&client, eden);
    assert_eq!(&eden_base[0x18..0x1d], b"Eden\0");
    assert_eq!(eden_base[0x10..0x18], 0x1234u64.to_le_bytes());
    assert_eq!(user_data, vec![2; 0x80]);
    assert_eq!(std::fs::read(&path).unwrap(), file, "opening saves nothing");
    drop(client);

    let client = boot_in(&dir, "player");
    assert_eq!(base(&client, user_id("player")).0[0x18..0x1f], *b"player\0");
    let saved = std::fs::read(&path).unwrap();
    assert_eq!(saved[..0x10 + 2 * 0xc8], file[..0x10 + 2 * 0xc8]);
    assert_eq!(saved[0x10 + 2 * 0xc8..][..16], user_id("player"));
}

#[test]
fn an_exact_user_id_wins_over_a_nickname() {
    let dir = TempDir::new("acc");
    let mut file = vec![0; 0x10];
    file.extend(user_raw([0x42; 16], "player", 1, 1));
    file.extend(user_raw(user_id("player"), "Other", 2, 2));
    file.resize(0x650, 0);
    let path = profiles_file(&dir);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &file).unwrap();

    let client = boot_in(&dir, "player");
    let acc = client.connect("acc:u0").unwrap();
    let last = client.send(acc, &Request::new(4)).unwrap().ok();
    assert_eq!(last.get::<[u8; 16]>(0), user_id("player"));
}

#[test]
fn an_unreadable_profiles_file_is_never_written() {
    let dir = TempDir::new("acc");
    let path = profiles_file(&dir);
    std::fs::create_dir_all(&path).unwrap();
    let profile = Profile::open("player", &dir.settings().paths.nand_dir.unwrap());
    assert!(profile.slot.is_none());
    assert!(path.is_dir());
}
