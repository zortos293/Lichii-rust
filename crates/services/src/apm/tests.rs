use lichii_common::settings::Settings;
use lichii_kernel::testing::{IpcClient, Request};

use crate::testing::{ReplyExt, SWITCH2_TITLE, booted};

fn boot(docked: bool) -> IpcClient {
    let mut settings = Settings::default();
    settings.system.docked = docked;
    booted(SWITCH2_TITLE, settings, super::register)
}

#[test]
fn performance_mode_follows_the_docked_setting() {
    let docked = boot(true);
    let apm = docked.connect("apm").unwrap();
    assert_eq!(
        docked
            .send(apm, &Request::new(1))
            .unwrap()
            .ok()
            .get::<i32>(0),
        1
    );
    assert_eq!(
        docked
            .send(apm, &Request::new(6))
            .unwrap()
            .ok()
            .get::<u8>(0),
        0
    );
    drop(docked);
    let handheld = boot(false);
    let apm = handheld.connect("apm:am").unwrap();
    assert_eq!(
        handheld
            .send(apm, &Request::new(1))
            .unwrap()
            .ok()
            .get::<i32>(0),
        0
    );
}

#[test]
fn sessions_set_and_get_configurations() {
    let client = boot(true);
    let apm = client.connect("apm").unwrap();
    let session = client.send(apm, &Request::new(0)).unwrap().ok().interface();
    assert_eq!(
        client
            .send(session, &Request::new(1).arg(0i32))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0x0002_0003
    );
    client
        .send(session, &Request::new(0).arg(0i32).arg(0x0001_0002u32))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(session, &Request::new(1).arg(0i32))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0x0001_0002
    );
    client
        .send(session, &Request::new(0).arg(0i32).arg(0x1234u32))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(session, &Request::new(1).arg(0i32))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0x0001_0002
    );
    let sys = client.connect("apm:sys").unwrap();
    assert_eq!(
        client
            .send(sys, &Request::new(7))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0x0002_0003
    );
    client.send(sys, &Request::new(6).arg(1u32)).unwrap().ok();
    assert_eq!(
        client
            .send(sys, &Request::new(7))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0x9222_0009
    );
    let event_session = client.send(sys, &Request::new(1)).unwrap().ok().interface();
    assert_eq!(
        client
            .send(event_session, &Request::new(1).arg(1i32))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0x9222_0009
    );
}
