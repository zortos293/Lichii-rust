use lichii_common::ConsoleGeneration;

use super::{NO_CAPABILITY, NO_RESTRICTION_ENABLED};
use crate::testing::{Guest, Request, settings};

fn service(guest: &Guest, name: &str) -> u32 {
    let factory = guest.connect(name);
    guest.call(factory, &Request::new(0).arg(0u64).pid()).ok().interface()
}

#[test]
fn applications_have_no_restrictions() {
    let guest = Guest::new(ConsoleGeneration::Switch1, settings(), super::register);
    let pctl = service(&guest, "pctl");
    for command in [1, 1001, 1002, 1003, 1004, 1005, 1013, 1017, 1018, 1064, 1451, 1452] {
        guest.call(pctl, &Request::new(command)).ok();
    }
    for command in [1006, 1010, 1031, 1062, 1403, 1453, 1455, 1458] {
        assert_eq!(guest.call(pctl, &Request::new(command)).ok().get::<u8>(0), 0, "command {command}");
    }
    assert_eq!(guest.call(pctl, &Request::new(1065)).ok().get::<u8>(0), 1);
    assert_eq!(guest.call(pctl, &Request::new(1032)).ok().get::<u32>(0), 0);
    assert_eq!(guest.call(pctl, &Request::new(1206)).ok().get::<u32>(0), 0);
    assert_eq!(guest.call(pctl, &Request::new(1035)).ok().data[..3], [0, 0, 0]);
    assert_eq!(guest.call(pctl, &Request::new(1039)).ok().get::<i32>(0), 4);
    assert_eq!(guest.call(pctl, &Request::new(1454)).ok().get::<i32>(0), i32::MAX);
    assert_eq!(guest.call(pctl, &Request::new(1456)).ok().data.len(), 0x34);
    let display = guest.call(pctl, &Request::new(1459)).ok();
    assert_eq!((display.data.len(), display.data[0]), (0x18, 7));
    assert_eq!(guest.call(pctl, &Request::new(1061)).result, NO_RESTRICTION_ENABLED.0);
    for command in [1432, 1457, 1473] {
        assert_eq!(guest.call(pctl, &Request::new(command)).ok().copies.len(), 1);
    }
}

#[test]
fn play_timer_settings_round_trip() {
    let guest = Guest::new(ConsoleGeneration::Switch2, settings(), super::register);
    let pctl = service(&guest, "pctl");
    let settings: [u32; 17] = std::array::from_fn(|i| i as u32 * 3);
    guest.call(pctl, &Request::new(195101).arg(settings)).ok();
    assert_eq!(guest.call(pctl, &Request::new(145601)).ok().get::<[u32; 17]>(0), settings);
}

#[test]
fn system_interfaces_lack_application_capabilities() {
    let guest = Guest::new(ConsoleGeneration::Switch1, settings(), super::register);
    let pctl = service(&guest, "pctl:s");
    assert_eq!(guest.call(pctl, &Request::new(1)).result, NO_CAPABILITY.0);
    assert_eq!(guest.call(pctl, &Request::new(1031)).result, NO_CAPABILITY.0);
    assert_eq!(guest.call(pctl, &Request::new(1062)).result, NO_CAPABILITY.0);
}
