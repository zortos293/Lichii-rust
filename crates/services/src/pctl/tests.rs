use super::{NO_CAPABILITY, NO_RESTRICTION_ENABLED};
use lichii_common::settings::Settings;
use lichii_kernel::testing::{IpcClient, Request};

use crate::testing::{ReplyExt, SWITCH1_TITLE, SWITCH2_TITLE, booted};

fn service(client: &IpcClient, name: &str) -> u32 {
    let factory = client.connect(name).unwrap();
    client
        .send(factory, &Request::new(0).arg(0u64).send_pid())
        .unwrap()
        .ok()
        .interface()
}

#[test]
fn applications_have_no_restrictions() {
    let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
    let pctl = service(&client, "pctl");
    for command in [
        1, 1001, 1002, 1003, 1004, 1005, 1013, 1017, 1018, 1064, 1451, 1452,
    ] {
        client.send(pctl, &Request::new(command)).unwrap().ok();
    }
    for command in [1006, 1010, 1031, 1062, 1403, 1453, 1455, 1458] {
        assert_eq!(
            client
                .send(pctl, &Request::new(command))
                .unwrap()
                .ok()
                .get::<u8>(0),
            0,
            "command {command}"
        );
    }
    assert_eq!(
        client
            .send(pctl, &Request::new(1065))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        client
            .send(pctl, &Request::new(1032))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0
    );
    assert_eq!(
        client
            .send(pctl, &Request::new(1206))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0
    );
    assert_eq!(
        client.send(pctl, &Request::new(1035)).unwrap().ok().data()[..3],
        [0, 0, 0]
    );
    assert_eq!(
        client
            .send(pctl, &Request::new(1039))
            .unwrap()
            .ok()
            .get::<i32>(0),
        4
    );
    assert_eq!(
        client
            .send(pctl, &Request::new(1454))
            .unwrap()
            .ok()
            .get::<i32>(0),
        i32::MAX
    );
    assert_eq!(
        client
            .send(pctl, &Request::new(1456))
            .unwrap()
            .ok()
            .data()
            .len(),
        0x34
    );
    let display = client.send(pctl, &Request::new(1459)).unwrap().ok();
    assert_eq!((display.data().len(), display.data()[0]), (0x18, 7));
    assert_eq!(
        client.send(pctl, &Request::new(1061)).unwrap().result,
        NO_RESTRICTION_ENABLED
    );
    for command in [1432, 1457, 1473] {
        assert_eq!(
            client
                .send(pctl, &Request::new(command))
                .unwrap()
                .ok()
                .copy_handles
                .len(),
            1
        );
    }
}

#[test]
fn play_timer_settings_round_trip() {
    let client = booted(SWITCH2_TITLE, Settings::default(), super::register);
    let pctl = service(&client, "pctl");
    let settings: [u32; 17] = std::array::from_fn(|i| i as u32 * 3);
    client
        .send(pctl, &Request::new(195101).arg(settings))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(pctl, &Request::new(145601))
            .unwrap()
            .ok()
            .get::<[u32; 17]>(0),
        settings
    );
}

#[test]
fn system_interfaces_lack_application_capabilities() {
    let client = booted(SWITCH1_TITLE, Settings::default(), super::register);
    let pctl = service(&client, "pctl:s");
    assert_eq!(
        client.send(pctl, &Request::new(1)).unwrap().result,
        NO_CAPABILITY
    );
    assert_eq!(
        client.send(pctl, &Request::new(1031)).unwrap().result,
        NO_CAPABILITY
    );
    assert_eq!(
        client.send(pctl, &Request::new(1062)).unwrap().result,
        NO_CAPABILITY
    );
}
