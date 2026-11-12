use lichii_common::ConsoleGeneration;

use crate::testing::{Guest, Request, settings};

fn guest(docked: bool) -> Guest {
    let mut settings = settings();
    settings.system.docked = docked;
    Guest::new(ConsoleGeneration::Switch2, settings, super::register)
}

#[test]
fn performance_mode_follows_the_docked_setting() {
    let docked = guest(true);
    let apm = docked.connect("apm");
    assert_eq!(docked.call(apm, &Request::new(1)).ok().get::<i32>(0), 1);
    assert_eq!(docked.call(apm, &Request::new(6)).ok().get::<u8>(0), 0);
    drop(docked);
    let handheld = guest(false);
    let apm = handheld.connect("apm:am");
    assert_eq!(handheld.call(apm, &Request::new(1)).ok().get::<i32>(0), 0);
}

#[test]
fn sessions_set_and_get_configurations() {
    let guest = guest(true);
    let apm = guest.connect("apm");
    let session = guest.call(apm, &Request::new(0)).ok().interface();
    assert_eq!(guest.call(session, &Request::new(1).arg(0i32)).ok().get::<u32>(0), 0x0002_0003);
    guest.call(session, &Request::new(0).arg(0i32).arg(0x0001_0002u32)).ok();
    assert_eq!(guest.call(session, &Request::new(1).arg(0i32)).ok().get::<u32>(0), 0x0001_0002);
    guest.call(session, &Request::new(0).arg(0i32).arg(0x1234u32)).ok();
    assert_eq!(guest.call(session, &Request::new(1).arg(0i32)).ok().get::<u32>(0), 0x0001_0002);
    let sys = guest.connect("apm:sys");
    assert_eq!(guest.call(sys, &Request::new(7)).ok().get::<u32>(0), 0x0002_0003);
    guest.call(sys, &Request::new(6).arg(1u32)).ok();
    assert_eq!(guest.call(sys, &Request::new(7)).ok().get::<u32>(0), 0x9222_0009);
    let event_session = guest.call(sys, &Request::new(1)).ok().interface();
    assert_eq!(guest.call(event_session, &Request::new(1).arg(1i32)).ok().get::<u32>(0), 0x9222_0009);
}
