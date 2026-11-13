use super::{DateRule, OutOfRange, PosixZone, RULE_SIZE, Rule, Transition, utc_tzif};
use crate::time::CalendarTime;

/// A TZif file with the same data in its version 1 and version 2 blocks.
fn tzif(
    times: &[i64],
    indices: &[u8],
    types: &[(i32, bool, u8)],
    chars: &[u8],
    footer: &str,
) -> Vec<u8> {
    let mut file = Vec::new();
    for time_size in [4, 8] {
        file.extend_from_slice(b"TZif2");
        file.extend_from_slice(&[0; 15]);
        let counts = [0, 0, 0, times.len(), types.len(), chars.len()];
        for count in counts {
            file.extend_from_slice(&(count as u32).to_be_bytes());
        }
        for &time in times {
            match time_size {
                4 => file.extend_from_slice(&(time as i32).to_be_bytes()),
                _ => file.extend_from_slice(&time.to_be_bytes()),
            }
        }
        file.extend_from_slice(indices);
        for &(utoff, is_dst, abbr) in types {
            file.extend_from_slice(&utoff.to_be_bytes());
            file.extend_from_slice(&[u8::from(is_dst), abbr]);
        }
        file.extend_from_slice(chars);
    }
    file.extend_from_slice(format!("\n{footer}\n").as_bytes());
    file
}

fn calendar(year: i16, month: i8, day: i8, hour: i8, minute: i8, second: i8) -> CalendarTime {
    CalendarTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
        pad: 0,
    }
}

fn cet() -> Rule {
    Rule::parse_tzif(&tzif(
        &[],
        &[],
        &[(3600, false, 0)],
        b"CET\0",
        "CET-1CEST,M3.5.0,M10.5.0/3",
    ))
    .unwrap()
}

#[test]
fn utc_converts_both_ways() {
    let rule = Rule::parse_tzif(&utc_tzif()).unwrap();
    let (time, info) = rule.to_calendar(1_700_000_000).unwrap();
    assert_eq!(time, calendar(2023, 11, 14, 22, 13, 20));
    assert_eq!(
        (
            info.day_of_week,
            info.day_of_year,
            info.is_dst,
            info.ut_offset
        ),
        (2, 317, 0, 0)
    );
    assert_eq!(&info.name, b"UTC\0\0\0\0\0");
    let (time, info) = rule.to_calendar(-1).unwrap();
    assert_eq!(
        (time, info.day_of_week, info.day_of_year),
        (calendar(1969, 12, 31, 23, 59, 59), 3, 364)
    );
    assert_eq!(
        rule.to_posix(&calendar(2023, 11, 14, 22, 13, 20)),
        ([1_700_000_000, 0], 1)
    );
    assert_eq!(
        rule.to_posix(&calendar(2024, 2, 29, 0, 0, 0)),
        ([1_709_164_800, 0], 1)
    );
    assert_eq!(rule.to_posix(&calendar(2023, 2, 29, 0, 0, 0)).1, 0);
    assert_eq!(rule.to_posix(&calendar(2023, 1, 1, 24, 0, 0)).1, 0);
}

#[test]
fn footer_rules_switch_daylight_saving() {
    let rule = cet();
    let (time, info) = rule.to_calendar(1_719_835_200).unwrap();
    assert_eq!(time, calendar(2024, 7, 1, 14, 0, 0));
    assert_eq!(
        (
            info.day_of_week,
            info.day_of_year,
            info.is_dst,
            info.ut_offset
        ),
        (1, 182, 1, 7200)
    );
    assert_eq!(&info.name[..5], b"CEST\0");
    let (time, info) = rule.to_calendar(1_705_320_000).unwrap();
    assert_eq!(
        (time, info.is_dst, info.ut_offset),
        (calendar(2024, 1, 15, 13, 0, 0), 0, 3600)
    );
    assert_eq!(
        rule.to_calendar(1_711_846_799).unwrap().0,
        calendar(2024, 3, 31, 1, 59, 59)
    );
    assert_eq!(
        rule.to_calendar(1_711_846_800).unwrap().0,
        calendar(2024, 3, 31, 3, 0, 0)
    );
}

#[test]
fn gaps_have_no_time_and_overlaps_two() {
    let rule = cet();
    assert_eq!(rule.to_posix(&calendar(2024, 3, 31, 2, 30, 0)).1, 0);
    assert_eq!(
        rule.to_posix(&calendar(2024, 10, 27, 2, 30, 0)),
        ([1_729_989_000, 1_729_992_600], 2)
    );
    assert_eq!(
        rule.to_posix(&calendar(2024, 7, 1, 14, 0, 0)),
        ([1_719_835_200, 0], 1)
    );
}

#[test]
fn transitions_pick_their_local_type() {
    let types = [(-18000, false, 0), (-14400, true, 4)];
    let file = tzif(
        &[1_710_054_000, 1_730_613_600],
        &[1, 0],
        &types,
        b"EST\0EDT\0",
        "",
    );
    let rule = Rule::parse_tzif(&file).unwrap();
    assert_eq!(rule.to_calendar(1_710_053_999).unwrap().1.ut_offset, -18000);
    assert_eq!(rule.to_calendar(1_710_054_000).unwrap().1.ut_offset, -14400);
    assert_eq!(rule.to_calendar(1_730_613_600).unwrap().1.ut_offset, -18000);
    assert!(Rule::parse_tzif(&file[..file.len() - 1]).is_none());
}

#[test]
fn default_us_rules_apply_without_dates() {
    let rule =
        Rule::parse_tzif(&tzif(&[], &[], &[(-18000, false, 0)], b"EST\0", "EST5EDT")).unwrap();
    assert_eq!(rule.to_calendar(1_710_053_999).unwrap().1.ut_offset, -18000);
    assert_eq!(rule.to_calendar(1_710_054_000).unwrap().1.ut_offset, -14400);
    assert_eq!(rule.to_calendar(1_730_613_599).unwrap().1.ut_offset, -14400);
    assert_eq!(rule.to_calendar(1_730_613_600).unwrap().1.ut_offset, -18000);
}

#[test]
fn posix_strings_parse() {
    let zone = PosixZone::parse(b"<+0330>-3:30").unwrap();
    assert_eq!(
        (zone.std_abbr, zone.std_utoff, zone.dst),
        (&b"+0330"[..], 12600, None)
    );
    let zone = PosixZone::parse(b"NZST-12NZDT,M9.5.0,M4.1.0/3").unwrap();
    let (abbr, utoff, start, end) = zone.dst.unwrap();
    assert_eq!((abbr, utoff), (&b"NZDT"[..], 46800));
    assert_eq!(
        start,
        Transition {
            date: DateRule::MonthWeekDay(9, 5, 0),
            time: 7200
        }
    );
    assert_eq!(
        end,
        Transition {
            date: DateRule::MonthWeekDay(4, 1, 0),
            time: 10800
        }
    );
    let zone = PosixZone::parse(b"XST3XDT,J60/-1,300/26").unwrap();
    let (_, utoff, start, end) = zone.dst.unwrap();
    assert_eq!(
        (utoff, start.date, start.time, end.date, end.time),
        (
            -7200,
            DateRule::Julian(60),
            -3600,
            DateRule::DayOfYear(300),
            93600
        )
    );
    assert!(PosixZone::parse(b"UT0").is_none());
    assert!(PosixZone::parse(b"CET-1CEST,M13.5.0,M10.5.0").is_none());
}

#[test]
fn rule_days_follow_the_calendar() {
    assert_eq!(
        DateRule::Julian(60).days(2024),
        DateRule::DayOfYear(60).days(2024)
    );
    assert_eq!(
        DateRule::Julian(59).days(2024),
        DateRule::DayOfYear(58).days(2024)
    );
    assert_eq!(
        DateRule::MonthWeekDay(3, 5, 0).days(2024) * 86400,
        1_711_843_200
    );
}

#[test]
fn southern_hemisphere_rules_wrap_the_year() {
    let file = tzif(
        &[],
        &[],
        &[(43200, false, 0)],
        b"NZST\0",
        "NZST-12NZDT,M9.5.0,M4.1.0/3",
    );
    let rule = Rule::parse_tzif(&file).unwrap();
    assert_eq!(rule.to_calendar(1_705_320_000).unwrap().1.ut_offset, 46800);
    assert_eq!(rule.to_calendar(1_719_835_200).unwrap().1.ut_offset, 43200);
}

#[test]
fn rules_round_trip_through_the_title_buffer() {
    let rule = cet();
    let mut buffer = [0; RULE_SIZE];
    rule.write(&mut buffer);
    assert_eq!(Rule::read(&buffer).unwrap(), rule);
    assert_eq!(Rule::read(&[]).unwrap(), Rule::default());
    assert_eq!(
        Rule::read(&[]).unwrap().to_calendar(0).unwrap().1.name,
        [0; 8]
    );
    buffer[super::INDICES_OFFSET] = 9;
    assert_eq!(Rule::read(&buffer), Err(OutOfRange));
    buffer[0..2].copy_from_slice(&5000u16.to_le_bytes());
    assert_eq!(Rule::read(&buffer), Err(OutOfRange));
}

#[test]
fn malformed_files_are_rejected() {
    assert!(Rule::parse_tzif(b"TZif").is_none());
    assert!(Rule::parse_tzif(&tzif(&[], &[], &[], b"", "")).is_none());
    assert!(Rule::parse_tzif(&tzif(&[5], &[3], &[(0, false, 0)], b"UTC\0", "")).is_none());
    assert!(Rule::parse_tzif(&tzif(&[9, 5], &[0, 0], &[(0, false, 0)], b"UTC\0", "")).is_none());
    assert!(Rule::parse_tzif(&tzif(&[], &[], &[(0, false, 0)], b"UTC\0", "bad")).is_none());
    assert!(Rule::to_calendar(&cet(), i64::MAX).is_none());
}
