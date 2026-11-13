use std::time::{SystemTime, UNIX_EPOCH};

use super::zone::RULE_SIZE;
use super::{
    CLOCK_MISMATCH, CLOCK_SOURCE_ID, CalendarAdditionalInfo, CalendarTime, ClockSnapshot,
    LOCAL_CONTEXT, NOT_IMPLEMENTED, OVERFLOW, PERMISSION_DENIED, STEADY_TIME_POINT,
    SteadyClockTimePoint, SystemClockContext, TIME_ZONE_NOT_FOUND,
};
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Request};

use crate::testing::{ReplyExt, SWITCH2_TITLE, TempDir, booted, read_value};

/// The time services with `time_zone` as the configured zone, over the synthesized archive.
fn boot_in(time_zone: &str) -> IpcClient {
    let dir = TempDir::new("time");
    let mut settings = dir.settings();
    settings.system.time_zone = time_zone.into();
    booted(SWITCH2_TITLE, settings, super::register)
}

fn boot() -> IpcClient {
    boot_in("UTC")
}

fn location(name: &str) -> [u8; 0x24] {
    let mut location = [0; 0x24];
    location[..name.len()].copy_from_slice(name.as_bytes());
    location
}

fn utc_location() -> [u8; 0x24] {
    location("UTC")
}

/// The zones the synthesized archive lists.
const SYNTHESIZED_ZONES: [&str; 4] = ["Etc/GMT", "Etc/UTC", "GMT", "UTC"];

fn host_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64
}

fn interface(client: &IpcClient, service: u32, command: u32) -> u32 {
    client
        .send(service, &Request::new(command))
        .unwrap()
        .ok()
        .interface()
}

#[test]
fn system_clocks_follow_the_host_clock() {
    let client = boot();
    let time = client.connect("time:u").unwrap();
    for command in [0, 1, 4] {
        let clock = interface(&client, time, command);
        let now = client
            .send(clock, &Request::new(0))
            .unwrap()
            .ok()
            .get::<i64>(0);
        assert!((now - host_now()).abs() <= 2, "command {command}: {now}");
        let context = client
            .send(clock, &Request::new(2))
            .unwrap()
            .ok()
            .get::<SystemClockContext>(0);
        assert_eq!(context.steady_time_point.source_id, CLOCK_SOURCE_ID);
        assert_eq!(
            client
                .send(clock, &Request::new(1).arg(0i64))
                .unwrap()
                .result,
            PERMISSION_DENIED
        );
        assert_eq!(
            client
                .send(clock, &Request::new(4))
                .unwrap()
                .ok()
                .copy_handles
                .len(),
            1
        );
    }
    let ephemeral = interface(&client, time, 5);
    assert_eq!(
        client.send(ephemeral, &Request::new(0)).unwrap().result,
        CLOCK_MISMATCH
    );
}

#[test]
fn reading_the_user_clock_leaves_its_event_alone() {
    let client = boot();
    let time = client.connect("time:u").unwrap();
    let clock = interface(&client, time, 0);
    let handle = client
        .send(clock, &Request::new(4))
        .unwrap()
        .ok()
        .copy_handles[0];
    let Some(KObject::ReadableEvent(event)) = client.object(handle) else {
        panic!("handle {handle:#x} is not a readable event");
    };
    event.clear();
    client.send(clock, &Request::new(0)).unwrap().ok();
    client.send(clock, &Request::new(2)).unwrap().ok();
    assert!(!event.is_signaled());
}

#[test]
fn writable_clocks_take_new_times() {
    let client = boot();
    let time = client.connect("time:a").unwrap();
    let local = interface(&client, time, 4);
    client
        .send(local, &Request::new(1).arg(1_000_000_000i64))
        .unwrap()
        .ok();
    let now = client
        .send(local, &Request::new(0))
        .unwrap()
        .ok()
        .get::<i64>(0);
    assert!((now - 1_000_000_000).abs() <= 1);
    let user = interface(&client, time, 0);
    assert_eq!(
        client
            .send(user, &Request::new(1).arg(5i64))
            .unwrap()
            .result,
        NOT_IMPLEMENTED
    );

    client.send(time, &Request::new(101).arg(0u8)).unwrap().ok();
    assert_eq!(
        client
            .send(time, &Request::new(100))
            .unwrap()
            .ok()
            .get::<u8>(0),
        0
    );
    let user_now = client
        .send(user, &Request::new(0))
        .unwrap()
        .ok()
        .get::<i64>(0);
    assert!(
        (user_now - host_now()).abs() <= 2,
        "the user clock took the network context"
    );
    client
        .send(local, &Request::new(1).arg(1_000_000_000i64))
        .unwrap()
        .ok();
    let user_now = client
        .send(user, &Request::new(0))
        .unwrap()
        .ok()
        .get::<i64>(0);
    assert!(
        (user_now - 1_000_000_000).abs() <= 1,
        "without correction the user clock is local"
    );
    let updated = client
        .send(time, &Request::new(201))
        .unwrap()
        .ok()
        .get::<SteadyClockTimePoint>(0);
    assert_eq!(updated.source_id, CLOCK_SOURCE_ID);

    let network = interface(&client, time, 1);
    assert_eq!(
        client
            .send(network, &Request::new(1).arg(0i64))
            .unwrap()
            .result,
        PERMISSION_DENIED
    );
}

#[test]
fn the_steady_clock_counts_seconds_since_the_epoch() {
    let client = boot();
    let time = client.connect("time:r").unwrap();
    let steady = interface(&client, time, 2);
    let point = client
        .send(steady, &Request::new(0))
        .unwrap()
        .ok()
        .get::<SteadyClockTimePoint>(0);
    assert_eq!(point.source_id, CLOCK_SOURCE_ID);
    assert!((point.time_point - host_now()).abs() <= 2);
    client
        .send(steady, &Request::new(3).arg(3_600_000_000_000i64))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(steady, &Request::new(2))
            .unwrap()
            .ok()
            .get::<i64>(0),
        3_600_000_000_000
    );
    let later = client
        .send(steady, &Request::new(0))
        .unwrap()
        .ok()
        .get::<SteadyClockTimePoint>(0);
    assert!((later.time_point - point.time_point - 3600).abs() <= 1);
    assert_eq!(
        client.send(steady, &Request::new(100)).unwrap().result,
        NOT_IMPLEMENTED
    );
    assert_eq!(
        client
            .send(steady, &Request::new(101))
            .unwrap()
            .ok()
            .get::<u8>(0),
        0
    );
    assert_eq!(
        client
            .send(steady, &Request::new(102))
            .unwrap()
            .ok()
            .get::<u32>(0),
        0
    );
    assert_eq!(
        client
            .send(steady, &Request::new(200))
            .unwrap()
            .ok()
            .get::<i64>(0),
        0
    );
    client.send(time, &Request::new(50).arg(0i64)).unwrap().ok();

    let user_port = client.connect("time:u").unwrap();
    let read_only = interface(&client, user_port, 2);
    assert_eq!(
        client
            .send(read_only, &Request::new(3).arg(0i64))
            .unwrap()
            .result,
        PERMISSION_DENIED
    );
    assert_eq!(
        client
            .send(user_port, &Request::new(50).arg(0i64))
            .unwrap()
            .result,
        PERMISSION_DENIED
    );
}

#[test]
fn static_service_values() {
    let client = boot();
    let time = client.connect("time:u").unwrap();
    assert_eq!(
        client
            .send(time, &Request::new(20))
            .unwrap()
            .ok()
            .copy_handles
            .len(),
        1
    );
    assert!(
        (client
            .send(time, &Request::new(51))
            .unwrap()
            .ok()
            .get::<i64>(0)
            - host_now())
        .abs()
            <= 2
    );
    assert_eq!(
        client
            .send(time, &Request::new(100))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        client
            .send(time, &Request::new(101).arg(0u8))
            .unwrap()
            .result,
        PERMISSION_DENIED
    );
    assert_eq!(
        client
            .send(time, &Request::new(102))
            .unwrap()
            .ok()
            .get::<i32>(0),
        2023
    );
    assert_eq!(
        client
            .send(time, &Request::new(200))
            .unwrap()
            .ok()
            .get::<u8>(0),
        1
    );
    assert_eq!(
        client.send(time, &Request::new(201)).unwrap().ok().data(),
        [0; 0x18]
    );

    let clock = interface(&client, time, 4);
    let context = client
        .send(clock, &Request::new(2))
        .unwrap()
        .ok()
        .get::<SystemClockContext>(0);
    let base = client
        .send(time, &Request::new(300).arg(context))
        .unwrap()
        .ok()
        .get::<i64>(0);
    assert!(
        (base - host_now()).abs() <= 2,
        "boot happened just now: {base}"
    );
    let foreign = SystemClockContext::default();
    assert_eq!(
        client
            .send(time, &Request::new(300).arg(foreign))
            .unwrap()
            .result,
        CLOCK_MISMATCH
    );
}

#[test]
fn clock_snapshots_carry_calendar_times() {
    let client = boot();
    let time = client.connect("time:u").unwrap();
    let size = std::mem::size_of::<ClockSnapshot>() as u64;
    assert_eq!(size, 0xD0);
    let a = client.alloc(&vec![0; size as usize]);
    client
        .send(time, &Request::new(400).arg(1u8).buffer_b(a, size))
        .unwrap()
        .ok();
    let snapshot: ClockSnapshot = read_value(&client, a);
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
    let b = client.alloc(&vec![0; size as usize]);
    let request = Request::new(401)
        .arg(0u8)
        .arg(user)
        .arg(SystemClockContext::default())
        .buffer_b(b, size);
    client.send(time, &request).unwrap().ok();
    let shifted: ClockSnapshot = read_value(&client, b);
    assert_eq!(shifted.network_time, 0);
    assert!((shifted.user_time - snapshot.user_time - 3600).abs() <= 1);

    let span = client
        .send(time, &Request::new(501).buffer_x(a, size).buffer_x(b, size))
        .unwrap()
        .ok()
        .get::<i64>(0);
    assert!((0..=1_000_000_000).contains(&span));
    let difference = client
        .send(time, &Request::new(500).buffer_x(a, size).buffer_x(b, size))
        .unwrap()
        .ok()
        .get::<i64>(0);
    assert_eq!(
        difference, 0,
        "correction is on and the network clock matches"
    );
    let mut manual = shifted;
    manual.is_automatic_correction_enabled = 0;
    let c = client.alloc(bytemuck::bytes_of(&manual));
    let difference = client
        .send(time, &Request::new(500).buffer_x(a, size).buffer_x(c, size))
        .unwrap()
        .ok()
        .get::<i64>(0);
    assert_eq!(difference, 3_600_000_000_000);
}

#[test]
fn the_time_zone_is_utc() {
    let client = boot();
    let time = client.connect("time:u").unwrap();
    let zone = interface(&client, time, 3);
    assert_eq!(
        client
            .send(zone, &Request::new(0))
            .unwrap()
            .ok()
            .get::<[u8; 0x24]>(0),
        utc_location()
    );
    assert_eq!(
        client
            .send(zone, &Request::new(2))
            .unwrap()
            .ok()
            .get::<u32>(0),
        4
    );
    let names = client.alloc(&[0; 0x24 * 4]);
    let list = |index: u32, size: u64| {
        let count = client
            .send(zone, &Request::new(3).arg(index).buffer_b(names, size))
            .unwrap()
            .ok()
            .get::<u32>(0) as usize;
        (0..count)
            .map(|i| client.read(names + 0x24 * i as u64, 0x24))
            .collect::<Vec<_>>()
    };
    let expected: Vec<Vec<u8>> = SYNTHESIZED_ZONES.map(|z| location(z).to_vec()).into();
    assert_eq!(list(0, 0x24 * 4), expected);
    assert_eq!(list(1, 0x24 * 2), expected[1..3]);
    assert_eq!(list(3, 0x24 * 4), expected[3..]);
    assert!(list(4, 0x24 * 4).is_empty());
    assert!(list(0, 0x23).is_empty());
    let mut version = [0u8; 0x10];
    version[..6].copy_from_slice(
        b"2025b
",
    );
    assert_eq!(
        client.send(zone, &Request::new(5)).unwrap().ok().data(),
        version
    );
    let reply = client.send(zone, &Request::new(6)).unwrap().ok();
    assert_eq!(
        (reply.data().len(), reply.get::<[u8; 0x24]>(0)),
        (0x40, utc_location())
    );
    assert_eq!(
        client
            .send(zone, &Request::new(20))
            .unwrap()
            .ok()
            .copy_handles
            .len(),
        1
    );
    assert_eq!(
        client
            .send(zone, &Request::new(1).arg(utc_location()))
            .unwrap()
            .result,
        PERMISSION_DENIED
    );
    assert_eq!(
        client.send(zone, &Request::new(8)).unwrap().result,
        NOT_IMPLEMENTED
    );

    let rule = client.alloc(&vec![0; RULE_SIZE]);
    client
        .send(
            zone,
            &Request::new(4)
                .arg(utc_location())
                .buffer_b(rule, RULE_SIZE as u64),
        )
        .unwrap()
        .ok();
    let mut tokyo = [0u8; 0x24];
    tokyo[..10].copy_from_slice(b"Asia/Tokyo");
    let reply = client
        .send(
            zone,
            &Request::new(4).arg(tokyo).buffer_b(rule, RULE_SIZE as u64),
        )
        .unwrap();
    assert_eq!(reply.result, TIME_ZONE_NOT_FOUND);

    for request in [
        Request::new(100)
            .arg(1_700_000_000i64)
            .buffer_a(rule, RULE_SIZE as u64),
        Request::new(101).arg(1_700_000_000i64),
    ] {
        let reply = client.send(zone, &request).unwrap().ok();
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
    for request in [
        Request::new(100)
            .arg(1_000_000_000_000i64)
            .buffer_a(rule, RULE_SIZE as u64),
        Request::new(101).arg(1_000_000_000_000i64),
    ] {
        assert_eq!(client.send(zone, &request).unwrap().result, OVERFLOW);
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
    let times = client.alloc(&[0; 16]);
    for request in [
        Request::new(201)
            .arg(calendar)
            .buffer_a(rule, RULE_SIZE as u64)
            .buffer_b(times, 16),
        Request::new(202).arg(calendar).buffer_b(times, 16),
    ] {
        assert_eq!(client.send(zone, &request).unwrap().ok().get::<u32>(0), 1);
        assert_eq!(read_value::<i64>(&client, times), 1_700_000_000);
    }
}

#[test]
fn administrators_set_the_location() {
    let client = boot();
    let time = client.connect("time:a").unwrap();
    let zone = interface(&client, time, 3);
    client
        .send(zone, &Request::new(1).arg(utc_location()))
        .unwrap()
        .ok();
    let reply = client.send(zone, &Request::new(6)).unwrap().ok();
    assert_eq!(
        reply.get::<SteadyClockTimePoint>(0x28).source_id,
        CLOCK_SOURCE_ID
    );
    client
        .send(zone, &Request::new(1).arg(location("GMT")))
        .unwrap()
        .ok();
    assert_eq!(
        client
            .send(zone, &Request::new(0))
            .unwrap()
            .ok()
            .get::<[u8; 0x24]>(0),
        location("GMT")
    );
    assert_eq!(
        client
            .send(zone, &Request::new(1).arg(location("Asia/Tokyo")))
            .unwrap()
            .result,
        TIME_ZONE_NOT_FOUND
    );
    assert_eq!(
        client.send(zone, &Request::new(7)).unwrap().result,
        NOT_IMPLEMENTED
    );
}

#[test]
fn the_shared_memory_holds_the_clock_contexts() {
    let client = boot();
    let time = client.connect("time:a").unwrap();
    let handle = client
        .send(time, &Request::new(20))
        .unwrap()
        .ok()
        .copy_handles[0];
    let Some(KObject::SharedMemory(shared)) = client.object(handle) else {
        panic!("handle {handle:#x} is not shared memory");
    };
    let read = |offset: u64, len: usize| {
        let mut data = vec![0; len];
        shared.read(offset, &mut data);
        data
    };
    let counter = u32::from_le_bytes(read(STEADY_TIME_POINT.counter, 4).try_into().unwrap());
    assert_eq!(counter, 1);
    let steady: SteadyClockTimePoint =
        bytemuck::pod_read_unaligned(&read(STEADY_TIME_POINT.values + 0x18, 0x18));
    assert_eq!(steady.source_id, CLOCK_SOURCE_ID);
    assert!((steady.time_point / 1_000_000_000 - host_now()).abs() <= 2);
    let clock = interface(&client, time, 4);
    let context = client
        .send(clock, &Request::new(2))
        .unwrap()
        .ok()
        .get::<SystemClockContext>(0);
    let local: SystemClockContext =
        bytemuck::pod_read_unaligned(&read(LOCAL_CONTEXT.values + 0x20, 0x20));
    assert_eq!(local, context);
    assert_eq!(read(0xC8, 4), 1u32.to_le_bytes());
    assert_eq!(read(0xCC + 1, 1), [1]);

    let mut moved = local;
    moved.offset += 10;
    client
        .send(clock, &Request::new(3).arg(moved))
        .unwrap()
        .ok();
    let local: SystemClockContext = bytemuck::pod_read_unaligned(&read(LOCAL_CONTEXT.values, 0x20));
    assert_eq!(local, moved);
    assert_eq!(read(LOCAL_CONTEXT.counter, 4), 2u32.to_le_bytes());
}

fn device_location(client: &IpcClient) -> [u8; 0x24] {
    let time = client.connect("time:u").unwrap();
    let zone = interface(client, time, 3);
    client
        .send(zone, &Request::new(0))
        .unwrap()
        .ok()
        .get::<[u8; 0x24]>(0)
}

#[test]
fn the_device_location_follows_the_settings() {
    for zone_name in ["Etc/GMT", "GMT", "Etc/UTC"] {
        let client = boot_in(zone_name);
        assert_eq!(device_location(&client), location(zone_name));
        let time = client.connect("time:u").unwrap();
        let snapshot = client.alloc(&[0; size_of::<ClockSnapshot>()]);
        client
            .send(
                time,
                &Request::new(400)
                    .arg(0u8)
                    .buffer_b(snapshot, size_of::<ClockSnapshot>() as u64),
            )
            .unwrap()
            .ok();
        let snapshot: ClockSnapshot = read_value(&client, snapshot);
        assert_eq!(snapshot.location_name, location(zone_name));
    }
}

#[test]
fn an_unknown_zone_falls_back_to_the_host_zone_then_utc() {
    let host = super::binary::host_zone().filter(|zone| SYNTHESIZED_ZONES.contains(&zone.as_str()));
    let expected = location(host.as_deref().unwrap_or("UTC"));
    for setting in ["Mars/Olympus", "auto", "Asia/Tokyo"] {
        assert_eq!(device_location(&boot_in(setting)), expected, "{setting}");
    }
}
