//! Time zone rules: TZif parsing, the rule buffer titles hold, and conversion between POSIX
//! time and calendar time, after the tz code Horizon embeds.

use super::{CalendarAdditionalInfo, CalendarTime};

/// Size of `nn::time::TimeZoneRule`, which titles treat as opaque.
pub(super) const RULE_SIZE: usize = 0x4000;
const MAX_TIMES: usize = 1000;
const MAX_TYPES: usize = 128;
const MAX_CHARS: usize = 256;
/// Daylight saving rules from a TZif footer become transitions up to this year.
const LAST_EXPANDED_YEAR: i64 = 2100;

const TIMES_OFFSET: usize = 0x10;
const INDICES_OFFSET: usize = TIMES_OFFSET + MAX_TIMES * 8;
const TYPES_OFFSET: usize = INDICES_OFFSET + MAX_TIMES;
const CHARS_OFFSET: usize = TYPES_OFFSET + MAX_TYPES * 6;
const _: () = assert!(CHARS_OFFSET + MAX_CHARS <= RULE_SIZE);

const SECONDS_PER_DAY: i64 = 86400;

/// A local time type: offset from UTC, daylight saving, abbreviation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct LocalType {
    utoff: i32,
    is_dst: bool,
    abbr: u8,
}

/// A rule's values do not fit, as `ResultTimeZoneOutOfRange` reports.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct OutOfRange;

/// Transitions between local time types. An empty rule is UTC with no abbreviation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Rule {
    times: Vec<i64>,
    indices: Vec<u8>,
    types: Vec<LocalType>,
    chars: Vec<u8>,
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * (month + if month > 2 { -3 } else { 9 }) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

/// The date of a day since 1970-01-01: year, month (1-12), day (1-31).
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1460 + day_of_era / 36524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let mp = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    (year_of_era + era * 400 + i64::from(month <= 2), month, day)
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 if is_leap(year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// The day a POSIX TZ rule names in `year`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DateRule {
    /// `Jn`: day 1-365, February 29 never counted.
    Julian(i64),
    /// `n`: zero-based day of the year, February 29 counted.
    DayOfYear(i64),
    /// `Mm.w.d`: weekday `d` (0 = Sunday) of week `w` (5 = last) of month `m`.
    MonthWeekDay(i64, i64, i64),
}

impl DateRule {
    fn days(self, year: i64) -> i64 {
        let jan1 = days_from_civil(year, 1, 1);
        match self {
            Self::Julian(n) => jan1 + n - 1 + i64::from(is_leap(year) && n >= 60),
            Self::DayOfYear(n) => jan1 + n,
            Self::MonthWeekDay(month, week, weekday) => {
                let first = days_from_civil(year, month, 1);
                let first_weekday = (first + 4).rem_euclid(7);
                let mut day = (weekday - first_weekday).rem_euclid(7) + 7 * (week - 1);
                while day >= days_in_month(year, month) {
                    day -= 7;
                }
                first + day
            }
        }
    }
}

/// A transition rule of a POSIX TZ string: the day and the local time of day, in seconds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Transition {
    date: DateRule,
    time: i64,
}

/// A POSIX TZ string, such as `CET-1CEST,M3.5.0,M10.5.0/3`.
#[derive(Debug, PartialEq, Eq)]
struct PosixZone<'a> {
    std_abbr: &'a [u8],
    std_utoff: i32,
    dst: Option<(&'a [u8], i32, Transition, Transition)>,
}

struct Cursor<'a> {
    text: &'a [u8],
}

impl<'a> Cursor<'a> {
    fn peek(&self) -> Option<u8> {
        self.text.first().copied()
    }

    fn eat(&mut self, byte: u8) -> bool {
        let found = self.peek() == Some(byte);
        if found {
            self.text = &self.text[1..];
        }
        found
    }

    fn number(&mut self) -> Option<i64> {
        let len = self.text.iter().take_while(|b| b.is_ascii_digit()).count();
        if len == 0 {
            return None;
        }
        let (digits, rest) = self.text.split_at(len);
        self.text = rest;
        digits.iter().try_fold(0i64, |n, d| {
            n.checked_mul(10)?.checked_add(i64::from(d - b'0'))
        })
    }

    fn name(&mut self) -> Option<&'a [u8]> {
        let (name, rest) = if self.eat(b'<') {
            let len = self.text.iter().position(|&b| b == b'>')?;
            (&self.text[..len], &self.text[len + 1..])
        } else {
            let len = self
                .text
                .iter()
                .take_while(|b| b.is_ascii_alphabetic())
                .count();
            (&self.text[..len], &self.text[len..])
        };
        self.text = rest;
        (name.len() >= 3).then_some(name)
    }

    /// `[+-]hh[:mm[:ss]]` in seconds.
    fn offset(&mut self) -> Option<i64> {
        let sign = if self.eat(b'-') {
            -1
        } else {
            self.eat(b'+');
            1
        };
        let mut seconds = self.number()? * 3600;
        for scale in [60, 1] {
            if !self.eat(b':') {
                break;
            }
            seconds += self.number()? * scale;
        }
        Some(sign * seconds)
    }

    fn transition(&mut self) -> Option<Transition> {
        let date = if self.eat(b'J') {
            DateRule::Julian(self.number().filter(|n| (1..=365).contains(n))?)
        } else if self.eat(b'M') {
            let month = self.number().filter(|n| (1..=12).contains(n))?;
            self.eat(b'.').then_some(())?;
            let week = self.number().filter(|n| (1..=5).contains(n))?;
            self.eat(b'.').then_some(())?;
            DateRule::MonthWeekDay(month, week, self.number().filter(|n| (0..=6).contains(n))?)
        } else {
            DateRule::DayOfYear(self.number().filter(|n| (0..=365).contains(n))?)
        };
        let time = if self.eat(b'/') {
            self.offset()?
        } else {
            2 * 3600
        };
        Some(Transition { date, time })
    }
}

impl<'a> PosixZone<'a> {
    fn parse(text: &'a [u8]) -> Option<Self> {
        let mut cursor = Cursor { text };
        let std_abbr = cursor.name()?;
        let std_utoff = i32::try_from(-cursor.offset()?).ok()?;
        if cursor.peek().is_none() {
            return Some(Self {
                std_abbr,
                std_utoff,
                dst: None,
            });
        }
        let dst_abbr = cursor.name()?;
        let dst_utoff = match cursor.peek() {
            Some(b',') | None => std_utoff + 3600,
            _ => i32::try_from(-cursor.offset()?).ok()?,
        };
        let (start, end) = if cursor.eat(b',') {
            let start = cursor.transition()?;
            cursor.eat(b',').then_some(())?;
            (start, cursor.transition()?)
        } else {
            let us = |month, week| Transition {
                date: DateRule::MonthWeekDay(month, week, 0),
                time: 2 * 3600,
            };
            (us(3, 2), us(11, 1))
        };
        cursor.peek().is_none().then_some(())?;
        Some(Self {
            std_abbr,
            std_utoff,
            dst: Some((dst_abbr, dst_utoff, start, end)),
        })
    }
}

fn be_u32(data: &[u8], offset: usize) -> Option<usize> {
    let bytes = data.get(offset..offset + 4)?;
    Some(u32::from_be_bytes(bytes.try_into().ok()?) as usize)
}

impl Rule {
    /// Parses a TZif file (versions 1 to 4). Footer rules of version 2+ files become
    /// transitions up to [`LAST_EXPANDED_YEAR`].
    pub(super) fn parse_tzif(data: &[u8]) -> Option<Self> {
        let header = |offset: usize| -> Option<[usize; 6]> {
            (data.get(offset..offset + 4)? == b"TZif").then_some(())?;
            let mut counts = [0; 6];
            for (i, count) in counts.iter_mut().enumerate() {
                *count = be_u32(data, offset + 20 + i * 4)?;
            }
            Some(counts)
        };
        let block_size = |[isut, isstd, leap, time, typ, chr]: [usize; 6], time_size: usize| {
            time * (time_size + 1) + typ * 6 + chr + leap * (time_size + 4) + isstd + isut
        };
        let mut counts = header(0)?;
        let mut offset = 44;
        let mut time_size = 4;
        let version = *data.get(4)?;
        if version >= b'2' {
            offset += block_size(counts, 4);
            counts = header(offset)?;
            offset += 44;
            time_size = 8;
        }
        let [_, _, _, time_count, type_count, char_count] = counts;
        if time_count > MAX_TIMES
            || !(1..=MAX_TYPES).contains(&type_count)
            || char_count > MAX_CHARS
        {
            return None;
        }
        let body = data.get(offset..offset + block_size(counts, time_size))?;
        let (times, rest) = body.split_at(time_count * time_size);
        let times: Vec<i64> = times
            .chunks_exact(time_size)
            .map(|t| match time_size {
                8 => i64::from_be_bytes(t.try_into().unwrap()),
                _ => i64::from(i32::from_be_bytes(t.try_into().unwrap())),
            })
            .collect();
        let (indices, rest) = rest.split_at(time_count);
        let (types, rest) = rest.split_at(type_count * 6);
        let types: Vec<LocalType> = types
            .as_chunks::<6>()
            .0
            .iter()
            .map(|t| LocalType {
                utoff: i32::from_be_bytes([t[0], t[1], t[2], t[3]]),
                is_dst: t[4] != 0,
                abbr: t[5],
            })
            .collect();
        let mut rule = Self {
            times,
            indices: indices.to_vec(),
            types,
            chars: rest[..char_count].to_vec(),
        };
        rule.validate().ok()?;
        if !rule.times.is_sorted_by(|a, b| a < b) {
            return None;
        }
        if version >= b'2' {
            let footer = &data[offset + body.len()..];
            let footer = footer.strip_prefix(b"\n")?;
            let end = footer.iter().position(|&b| b == b'\n')?;
            if end > 0 {
                rule.extend(&PosixZone::parse(&footer[..end])?)?;
            }
        }
        Some(rule)
    }

    /// The index of the local type, adding it when the rule has none like it.
    fn local_type_index(&mut self, utoff: i32, is_dst: bool, abbr: &[u8]) -> Option<u8> {
        let abbr_index = (0..self.chars.len()).find(|&i| {
            self.chars[i..].starts_with(abbr) && self.chars.get(i + abbr.len()) == Some(&0)
        });
        let abbr_index = match abbr_index {
            Some(index) => index,
            None => {
                if self.chars.len() + abbr.len() + 1 > MAX_CHARS {
                    return None;
                }
                self.chars.extend_from_slice(abbr);
                self.chars.push(0);
                self.chars.len() - abbr.len() - 1
            }
        };
        let local = LocalType {
            utoff,
            is_dst,
            abbr: u8::try_from(abbr_index).ok()?,
        };
        if let Some(index) = self.types.iter().position(|t| *t == local) {
            return u8::try_from(index).ok();
        }
        if self.types.len() == MAX_TYPES {
            return None;
        }
        self.types.push(local);
        u8::try_from(self.types.len() - 1).ok()
    }

    /// Applies a footer rule to the times after the last transition.
    fn extend(&mut self, zone: &PosixZone) -> Option<()> {
        let std = self.local_type_index(zone.std_utoff, false, zone.std_abbr)?;
        let Some((dst_abbr, dst_utoff, start, end)) = zone.dst else {
            return Some(());
        };
        let dst = self.local_type_index(dst_utoff, true, dst_abbr)?;
        let last = self.times.last().copied().unwrap_or(i64::MIN);
        let first_year = civil_from_days(last.max(0).div_euclid(SECONDS_PER_DAY)).0;
        for year in first_year..=LAST_EXPANDED_YEAR {
            let start = (
                start.date.days(year) * SECONDS_PER_DAY + start.time - i64::from(zone.std_utoff),
                dst,
            );
            let end = (
                end.date.days(year) * SECONDS_PER_DAY + end.time - i64::from(dst_utoff),
                std,
            );
            let mut pair = [start, end];
            pair.sort_unstable();
            for (time, index) in pair {
                if self.times.last().is_some_and(|&t| time <= t) {
                    continue;
                }
                if self.times.len() == MAX_TIMES {
                    return Some(());
                }
                self.times.push(time);
                self.indices.push(index);
            }
        }
        Some(())
    }

    /// The checks `ValidateRule` makes before a conversion.
    fn validate(&self) -> Result<(), OutOfRange> {
        if self.times.len() > MAX_TIMES
            || self.types.len() > MAX_TYPES
            || self.chars.len() > MAX_CHARS
        {
            return Err(OutOfRange);
        }
        if self
            .indices
            .iter()
            .any(|&i| usize::from(i) >= self.types.len())
        {
            return Err(OutOfRange);
        }
        if self
            .types
            .iter()
            .any(|t| usize::from(t.abbr) >= self.chars.len().max(1))
        {
            return Err(OutOfRange);
        }
        Ok(())
    }

    /// Writes the rule as a title stores it.
    pub(super) fn write(&self, out: &mut [u8; RULE_SIZE]) {
        out.fill(0);
        out[0..2].copy_from_slice(&(self.times.len() as u16).to_le_bytes());
        out[2..4].copy_from_slice(&(self.types.len() as u16).to_le_bytes());
        out[4..6].copy_from_slice(&(self.chars.len() as u16).to_le_bytes());
        for (i, time) in self.times.iter().enumerate() {
            out[TIMES_OFFSET + i * 8..][..8].copy_from_slice(&time.to_le_bytes());
        }
        out[INDICES_OFFSET..][..self.indices.len()].copy_from_slice(&self.indices);
        for (i, t) in self.types.iter().enumerate() {
            let entry = &mut out[TYPES_OFFSET + i * 6..][..6];
            entry[..4].copy_from_slice(&t.utoff.to_le_bytes());
            entry[4] = u8::from(t.is_dst);
            entry[5] = t.abbr;
        }
        out[CHARS_OFFSET..][..self.chars.len()].copy_from_slice(&self.chars);
    }

    /// Reads a rule a title passes back; a short buffer reads as zeros past its end.
    pub(super) fn read(data: &[u8]) -> Result<Self, OutOfRange> {
        let mut bytes = [0; RULE_SIZE];
        let len = data.len().min(RULE_SIZE);
        bytes[..len].copy_from_slice(&data[..len]);
        let count =
            |offset: usize| usize::from(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]));
        let (times, types, chars) = (count(0), count(2), count(4));
        if times > MAX_TIMES || types > MAX_TYPES || chars > MAX_CHARS {
            return Err(OutOfRange);
        }
        let rule = Self {
            times: bytes[TIMES_OFFSET..][..times * 8]
                .as_chunks::<8>()
                .0
                .iter()
                .map(|t| i64::from_le_bytes(*t))
                .collect(),
            indices: bytes[INDICES_OFFSET..][..times].to_vec(),
            types: bytes[TYPES_OFFSET..][..types * 6]
                .as_chunks::<6>()
                .0
                .iter()
                .map(|t| LocalType {
                    utoff: i32::from_le_bytes([t[0], t[1], t[2], t[3]]),
                    is_dst: t[4] != 0,
                    abbr: t[5],
                })
                .collect(),
            chars: bytes[CHARS_OFFSET..][..chars].to_vec(),
        };
        rule.validate()?;
        Ok(rule)
    }

    fn local_type(&self, time: i64) -> LocalType {
        let first_standard = || {
            self.types
                .iter()
                .find(|t| !t.is_dst)
                .or(self.types.first())
                .copied()
                .unwrap_or_default()
        };
        match self.times.partition_point(|&t| t <= time) {
            0 => first_standard(),
            index => self.types[usize::from(self.indices[index - 1])],
        }
    }

    /// `ToCalendarTime`; `None` when the year does not fit, as `ResultOverflow`.
    pub(super) fn to_calendar(&self, time: i64) -> Option<(CalendarTime, CalendarAdditionalInfo)> {
        let local = self.local_type(time);
        let local_time = time.checked_add(i64::from(local.utoff))?;
        let days = local_time.div_euclid(SECONDS_PER_DAY);
        let seconds = local_time.rem_euclid(SECONDS_PER_DAY);
        let (year, month, day) = civil_from_days(days);
        let calendar_year = i16::try_from(year).ok()?;
        let mut name = [0; 8];
        if let Some(abbr) = self.chars.get(usize::from(local.abbr)..) {
            let len = abbr
                .iter()
                .position(|&b| b == 0)
                .unwrap_or(abbr.len())
                .min(7);
            name[..len].copy_from_slice(&abbr[..len]);
        }
        Some((
            CalendarTime {
                year: calendar_year,
                month: month as i8,
                day: day as i8,
                hour: (seconds / 3600) as i8,
                minute: (seconds / 60 % 60) as i8,
                second: (seconds % 60) as i8,
                pad: 0,
            },
            CalendarAdditionalInfo {
                day_of_week: (days + 4).rem_euclid(7) as i32,
                day_of_year: (days - days_from_civil(year, 1, 1)) as i32,
                name,
                is_dst: i32::from(local.is_dst),
                ut_offset: local.utoff,
            },
        ))
    }

    /// `ToPosixTime`: the times, earliest first, that show `calendar` locally. None in a gap
    /// that skips it, two where an overlap repeats it.
    pub(super) fn to_posix(&self, calendar: &CalendarTime) -> ([i64; 2], usize) {
        let (year, month, day) = (
            i64::from(calendar.year),
            i64::from(calendar.month),
            i64::from(calendar.day),
        );
        let valid = (1..=12).contains(&month)
            && (1..=days_in_month(year, month)).contains(&day)
            && (0..24).contains(&calendar.hour)
            && (0..60).contains(&calendar.minute)
            && (0..60).contains(&calendar.second);
        let mut times = [0; 2];
        if !valid {
            return (times, 0);
        }
        let local = days_from_civil(year, month, day) * SECONDS_PER_DAY
            + i64::from(calendar.hour) * 3600
            + i64::from(calendar.minute) * 60
            + i64::from(calendar.second);
        let mut offsets: Vec<i32> = self.types.iter().map(|t| t.utoff).collect();
        if offsets.is_empty() {
            offsets.push(0);
        }
        offsets.sort_unstable();
        offsets.dedup();
        let mut count = 0;
        for utoff in offsets {
            let time = local - i64::from(utoff);
            if self.local_type(time).utoff == utoff && count < 2 {
                times[count] = time;
                count += 1;
            }
        }
        times[..count].sort_unstable();
        (times, count)
    }
}

const UTC_TZIF_SIZE: usize = 2 * (44 + 10) + 6;

/// A TZif file for UTC, the only location Lichii has a rule for.
pub(super) fn utc_tzif() -> [u8; UTC_TZIF_SIZE] {
    let mut file = [0; UTC_TZIF_SIZE];
    for header in [0, 44 + 10] {
        file[header..][..5].copy_from_slice(b"TZif2");
        file[header + 36..][..4].copy_from_slice(&1u32.to_be_bytes());
        file[header + 40..][..4].copy_from_slice(&4u32.to_be_bytes());
        file[header + 50..][..4].copy_from_slice(b"UTC\0");
    }
    file[UTC_TZIF_SIZE - 6..].copy_from_slice(b"\nUTC0\n");
    file
}

#[cfg(test)]
mod tests;
