use std::time::{SystemTime, UNIX_EPOCH};

use lichii_common::ConsoleGeneration;

use super::zone::RULE_SIZE;
use super::{
    CLOCK_MISMATCH, CLOCK_SOURCE_ID, CalendarAdditionalInfo, CalendarTime, ClockSnapshot, Clocks,
    LOCAL_CONTEXT, NOT_IMPLEMENTED, PERMISSION_DENIED, STEADY_TIME_POINT, SteadyClockTimePoint,
    SystemClockContext, TIME_ZONE_NOT_FOUND, utc_location,
};
use crate::testing::{Guest, Request, settings};

fn guest() -> Guest {
    Guest::new(ConsoleGeneration::Switch2, settings(), super::register)
}

fn host_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

fn interface(guest: &Guest, service: u32, command: u32) -> u32 {
    guest.call(service, &Request::new(command)).ok().interface()
}

#[test]
fn system_clocks_follow_the_host_clock() {
    let guest = guest();
    let time = guest.connect("time:u");
    for command in [0, 1, 4] {
        let clock = interface(&guest, time, command);
        let now = guest.call(clock, &Request::new(0)).ok().get::<i64>(0);
        assert!((now - host_now()).abs() <= 2, "command {command}: {now}");
        let context = guest
            .call(clock, &Request::new(2))
            .ok()
            .get::<SystemClockContext>(0);
        assert_eq!(context.steady_time_point.source_id, CLOCK_SOURCE_ID);
        assert_eq!(
            guest.call(clock, &Request::new(1).arg(0i64)).result,
            PERMISSION_DENIED.0
        );
        assert_eq!(guest.call(clock, &Request::new(4)).ok().copies.len(), 1);
    }
    let ephemeral = interface(&guest, time, 5);
    assert_eq!(
        guest.call(ephemeral, &Request::new(0)).result,
        CLOCK_MISMATCH.0
    );
}

#[test]
fn writable_clocks_take_new_times() {
    let guest = guest();
    let time = guest.connect("time:a");
    let local = interface(&guest, time, 4);
    guest
        .call(local, &Request::new(1).arg(1_000_000_000i64))
        .ok();
    let now = guest.call(local, &Request::new(0)).ok().get::<i64>(0);
    assert!((now - 1_000_000_000).abs() <= 1);
    let user = interface(&guest, time, 0);
    assert_eq!(
        guest.call(user, &Request::new(1).arg(5i64)).result,
        NOT_IMPLEMENTED.0
    );

    guest.call(time, &Request::new(101).arg(0u8)).ok();
    assert_eq!(guest.call(time, &Request::new(100)).ok().get::<u8>(0), 0);
    let user_now = guest.call(user, &Request::new(0)).ok().get::<i64>(0);
    assert!(
        (user_now - host_now()).abs() <= 2,
        "the user clock took the network context"
    );
    guest
        .call(local, &Request::new(1).arg(1_000_000_000i64))
        .ok();
    let user_now = guest.call(user, &Request::new(0)).ok().get::<i64>(0);
    assert!(
        (user_now - 1_000_000_000).abs() <= 1,
        "without correction the user clock is local"
    );
    let updated = guest
        .call(time, &Request::new(201))
        .ok()
        .get::<SteadyClockTimePoint>(0);
    assert_eq!(updated.source_id, CLOCK_SOURCE_ID);

    let network = interface(&guest, time, 1);
    assert_eq!(
        guest.call(network, &Request::new(1).arg(0i64)).result,
        PERMISSION_DENIED.0
    );
}

#[test]
fn the_steady_clock_counts_seconds_since_the_epoch() {
    let guest = guest();
    let time = guest.connect("time:r");
    let steady = interface(&guest, time, 2);
    let point = guest
        .call(steady, &Request::new(0))
        .ok()
        .get::<SteadyClockTimePoint>(0);
    assert_eq!(point.source_id, CLOCK_SOURCE_ID);
    assert!((point.time_point - host_now()).abs() <= 2);
    guest
        .call(steady, &Request::new(3).arg(3_600_000_000_000i64))
        .ok();
    assert_eq!(
        guest.call(steady, &Request::new(2)).ok().get::<i64>(0),
        3_600_000_000_000
    );
    let later = guest
        .call(steady, &Request::new(0))
        .ok()
        .get::<SteadyClockTimePoint>(0);
    assert!((later.time_point - point.time_point - 3600).abs() <= 1);
    assert_eq!(
        guest.call(steady, &Request::new(100)).result,
        NOT_IMPLEMENTED.0
    );
    assert_eq!(guest.call(steady, &Request::new(101)).ok().get::<u8>(0), 0);
    assert_eq!(guest.call(steady, &Request::new(102)).ok().get::<u32>(0), 0);
    assert_eq!(guest.call(steady, &Request::new(200)).ok().get::<i64>(0), 0);
    guest.call(time, &Request::new(50).arg(0i64)).ok();

    let user_port = guest.connect("time:u");
    let read_only = interface(&guest, user_port, 2);
    assert_eq!(
        guest.call(read_only, &Request::new(3).arg(0i64)).result,
        PERMISSION_DENIED.0
    );
    assert_eq!(
        guest.call(user_port, &Request::new(50).arg(0i64)).result,
        PERMISSION_DENIED.0
    );
}

#[test]
fn static_service_values() {
    let guest = guest();
    let time = guest.connect("time:u");
    assert_eq!(guest.call(time, &Request::new(20)).ok().copies.len(), 1);
    assert!((guest.call(time, &Request::new(51)).ok().get::<i64>(0) - host_now()).abs() <= 2);
    assert_eq!(guest.call(time, &Request::new(100)).ok().get::<u8>(0), 1);
    assert_eq!(
        guest.call(time, &Request::new(101).arg(0u8)).result,
        PERMISSION_DENIED.0
    );
    assert_eq!(
        guest.call(time, &Request::new(102)).ok().get::<i32>(0),
        2023
    );
    assert_eq!(guest.call(time, &Request::new(200)).ok().get::<u8>(0), 1);
    assert_eq!(guest.call(time, &Request::new(201)).ok().data, [0; 0x18]);

    let clock = interface(&guest, time, 4);
    let context = guest
        .call(clock, &Request::new(2))
        .ok()
        .get::<SystemClockContext>(0);
    let base = guest
        .call(time, &Request::new(300).arg(context))
        .ok()
        .get::<i64>(0);
    assert!(
        (base - host_now()).abs() <= 2,
        "boot happened just now: {base}"
    );
    let foreign = SystemClockContext::default();
    assert_eq!(
        guest.call(time, &Request::new(300).arg(foreign)).result,
        CLOCK_MISMATCH.0
    );
}

#[test]
fn clock_snapshots_carry_calendar_times() {
    let guest = guest();
    let time = guest.connect("time:u");
    let size = std::mem::size_of::<ClockSnapshot>() as u64;
    assert_eq!(size, 0xD0);
    let a = guest.buffer(size);
    guest
        .call(time, &Request::new(400).arg(1u8).b(a, size))
        .ok();
    let snapshot: ClockSnapshot = guest.read_value(a);
    assert!((snapshot.user_time - host_now()).abs() <= 2);
    assert_eq!(snapshot.network_time, snapshot.user_time);
    assert_eq!(&snapshot.location_name, &utc_location());
    assert_eq!(
        (snapshot.is_automatic_correction_enabled, snapshot.kind),
        (1, 1)
    );
    assert_eq!(&snapshot.user_calendar_additional_info.name[..4], b"UTC\0");
    assert!(snapshot.user_calendar_time.year >= 2026);

    let mut user = snapshot.user_context;
    user.offset += 3600;
    let b = guest.buffer(size);
    let request = Request::new(401)
        .arg(0u8)
        .arg(user)
        .arg(SystemClockContext::default())
        .b(b, size);
    guest.call(time, &request).ok();
    let shifted: ClockSnapshot = guest.read_value(b);
    assert_eq!(shifted.network_time, 0);
    assert!((shifted.user_time - snapshot.user_time - 3600).abs() <= 1);

    let span = guest
        .call(time, &Request::new(501).x(a, size).x(b, size))
        .ok()
        .get::<i64>(0);
    assert!((0..=1_000_000_000).contains(&span));
    let difference = guest
        .call(time, &Request::new(500).x(a, size).x(b, size))
        .ok()
        .get::<i64>(0);
    assert_eq!(
        difference, 0,
        "correction is on and the network clock matches"
    );
    let mut manual = shifted;
    manual.is_automatic_correction_enabled = 0;
    let c = guest.buffer_with(bytemuck::bytes_of(&manual));
    let difference = guest
        .call(time, &Request::new(500).x(a, size).x(c, size))
        .ok()
        .get::<i64>(0);
    assert_eq!(difference, 3_600_000_000_000);
}

#[test]
fn the_time_zone_is_utc() {
    let guest = guest();
    let time = guest.connect("time:u");
    let zone = interface(&guest, time, 3);
    assert_eq!(
        guest.call(zone, &Request::new(0)).ok().get::<[u8; 0x24]>(0),
        utc_location()
    );
    assert_eq!(guest.call(zone, &Request::new(2)).ok().get::<u32>(0), 1);
    let names = guest.buffer(0x24 * 4);
    assert_eq!(
        guest
            .call(zone, &Request::new(3).arg(0u32).b(names, 0x24 * 4))
            .ok()
            .get::<u32>(0),
        1
    );
    assert_eq!(guest.read(names, 0x24), utc_location());
    assert_eq!(
        guest
            .call(zone, &Request::new(3).arg(1u32).b(names, 0x24 * 4))
            .ok()
            .get::<u32>(0),
        0
    );
    assert_eq!(guest.call(zone, &Request::new(5)).ok().data, [0; 0x10]);
    let reply = guest.call(zone, &Request::new(6)).ok();
    assert_eq!(
        (reply.data.len(), reply.get::<[u8; 0x24]>(0)),
        (0x40, utc_location())
    );
    assert_eq!(guest.call(zone, &Request::new(20)).ok().copies.len(), 1);
    assert_eq!(
        guest
            .call(zone, &Request::new(1).arg(utc_location()))
            .result,
        PERMISSION_DENIED.0
    );
    assert_eq!(guest.call(zone, &Request::new(8)).result, NOT_IMPLEMENTED.0);

    let rule = guest.buffer(RULE_SIZE as u64);
    guest
        .call(
            zone,
            &Request::new(4)
                .arg(utc_location())
                .b(rule, RULE_SIZE as u64),
        )
        .ok();
    let mut tokyo = [0u8; 0x24];
    tokyo[..10].copy_from_slice(b"Asia/Tokyo");
    let reply = guest.call(zone, &Request::new(4).arg(tokyo).b(rule, RULE_SIZE as u64));
    assert_eq!(reply.result, TIME_ZONE_NOT_FOUND.0);

    for request in [
        Request::new(100)
            .arg(1_700_000_000i64)
            .a(rule, RULE_SIZE as u64),
        Request::new(101).arg(1_700_000_000i64),
    ] {
        let reply = guest.call(zone, &request).ok();
        let calendar: CalendarTime = reply.get(0);
        let info: CalendarAdditionalInfo = reply.get(8);
        assert_eq!(
            (calendar.year, calendar.month, calendar.day, calendar.hour),
            (2023, 11, 14, 22)
        );
        assert_eq!(
            (info.day_of_week, info.day_of_year, &info.name[..4]),
            (2, 317, &b"UTC\0"[..])
        );
    }
    let calendar = CalendarTime {
        year: 2023,
        month: 11,
        day: 14,
        hour: 22,
        minute: 13,
        second: 20,
        pad: 0,
    };
    let times = guest.buffer(16);
    for request in [
        Request::new(201)
            .arg(calendar)
            .a(rule, RULE_SIZE as u64)
            .b(times, 16),
        Request::new(202).arg(calendar).b(times, 16),
    ] {
        assert_eq!(guest.call(zone, &request).ok().get::<u32>(0), 1);
        assert_eq!(guest.read_value::<i64>(times), 1_700_000_000);
    }
}

#[test]
fn administrators_set_the_location() {
    let guest = guest();
    let time = guest.connect("time:a");
    let zone = interface(&guest, time, 3);
    guest.call(zone, &Request::new(1).arg(utc_location())).ok();
    let reply = guest.call(zone, &Request::new(6)).ok();
    assert_eq!(
        reply.get::<SteadyClockTimePoint>(0x28).source_id,
        CLOCK_SOURCE_ID
    );
    let mut tokyo = [0u8; 0x24];
    tokyo[..10].copy_from_slice(b"Asia/Tokyo");
    assert_eq!(
        guest.call(zone, &Request::new(1).arg(tokyo)).result,
        TIME_ZONE_NOT_FOUND.0
    );
    assert_eq!(guest.call(zone, &Request::new(7)).result, NOT_IMPLEMENTED.0);
}

#[test]
fn the_shared_memory_holds_the_clock_contexts() {
    let guest = guest();
    let clocks = Clocks::new(&guest.context).unwrap();
    let read = |offset: u64, len: usize| {
        let mut data = vec![0; len];
        clocks.shared.read(offset, &mut data);
        data
    };
    let counter = u32::from_le_bytes(read(STEADY_TIME_POINT.counter, 4).try_into().unwrap());
    assert_eq!(counter, 1);
    let steady: SteadyClockTimePoint =
        bytemuck::pod_read_unaligned(&read(STEADY_TIME_POINT.values + 0x18, 0x18));
    assert_eq!(steady.source_id, CLOCK_SOURCE_ID);
    assert!((steady.time_point / 1_000_000_000 - host_now()).abs() <= 2);
    let local: SystemClockContext =
        bytemuck::pod_read_unaligned(&read(LOCAL_CONTEXT.values + 0x20, 0x20));
    assert_eq!(local, clocks.context(super::ClockKind::Local));
    assert_eq!(read(0xC8, 4), 1u32.to_le_bytes());
    assert_eq!(read(0xCC + 1, 1), [1]);

    let mut moved = local;
    moved.offset += 10;
    clocks.set_context(super::ClockKind::Local, moved);
    let local: SystemClockContext = bytemuck::pod_read_unaligned(&read(LOCAL_CONTEXT.values, 0x20));
    assert_eq!(local, moved);
    assert_eq!(read(LOCAL_CONTEXT.counter, 4), 2u32.to_le_bytes());
}
