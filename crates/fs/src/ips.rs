//! Binary patches for executables and RomFS files: IPS, IPS32 and IPSwitch text (`.pchtxt`).
//!
//! Executable patches address the NSO as a file with its 0x100-byte header in front of the
//! decompressed image, the layout mods are written against.

use std::collections::BTreeMap;

use crate::error::{FsResult, corrupt};

/// A build id from hex text such as an `.ips` file stem or `@nsobid-`, zero-padded on the
/// right the way mod folders abbreviate it. `None` when the text is longer than 64 digits or
/// not hex.
pub fn parse_build_id(hex: &str) -> Option<[u8; 32]> {
    let digits = hex.as_bytes();
    if digits.len() > 64 {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, &digit) in digits.iter().enumerate() {
        let nibble = char::from(digit).to_digit(16)? as u8;
        out[i / 2] |= if i.is_multiple_of(2) {
            nibble << 4
        } else {
            nibble
        };
    }
    Some(out)
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum IpsRecord {
    Copy { offset: u32, bytes: Vec<u8> },
    Fill { offset: u32, len: u16, byte: u8 },
}

/// An IPS (`PATCH`, 24-bit offsets) or IPS32 (`IPS32`, 32-bit offsets) patch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Ips {
    records: Vec<IpsRecord>,
}

impl Ips {
    /// Parses a whole IPS or IPS32 file. Fails on a missing magic, a record cut off by the end
    /// of the data, or a missing `EOF`/`EEOF` marker.
    pub fn parse(data: &[u8]) -> FsResult<Self> {
        let (offset_len, eof): (usize, &[u8]) = match data.get(..5) {
            Some(b"PATCH") => (3, b"EOF"),
            Some(b"IPS32") => (4, b"EEOF"),
            _ => return Err(corrupt("IPS", "no PATCH or IPS32 magic")),
        };
        let mut rest = &data[5..];
        let mut take = |len: usize| -> FsResult<&[u8]> {
            let (head, tail) = rest
                .split_at_checked(len)
                .ok_or_else(|| corrupt("IPS", "record past the end of the patch"))?;
            rest = tail;
            Ok(head)
        };
        let mut records = Vec::new();
        loop {
            let offset = take(offset_len)?;
            if offset == eof {
                return Ok(Self { records });
            }
            let offset = offset
                .iter()
                .fold(0u32, |acc, &byte| (acc << 8) | u32::from(byte));
            let size = u16::from_be_bytes(take(2)?.try_into().unwrap());
            records.push(if size == 0 {
                let len = u16::from_be_bytes(take(2)?.try_into().unwrap());
                IpsRecord::Fill {
                    offset,
                    len,
                    byte: take(1)?[0],
                }
            } else {
                IpsRecord::Copy {
                    offset,
                    bytes: take(usize::from(size))?.to_vec(),
                }
            });
        }
    }

    /// Writes every record, clipping writes at the end of `target`. Refuses the whole patch,
    /// leaving `target` untouched, when a record starts past the end.
    pub fn apply(&self, target: &mut [u8]) -> FsResult<()> {
        let start = |record: &IpsRecord| match *record {
            IpsRecord::Copy { offset, .. } | IpsRecord::Fill { offset, .. } => offset as usize,
        };
        if self.records.iter().any(|r| start(r) > target.len()) {
            return Err(corrupt("IPS", "record starts past the end of the target"));
        }
        for record in &self.records {
            let tail = &mut target[start(record)..];
            match record {
                IpsRecord::Copy { bytes, .. } => {
                    let len = bytes.len().min(tail.len());
                    tail[..len].copy_from_slice(&bytes[..len]);
                }
                IpsRecord::Fill { len, byte, .. } => {
                    let len = usize::from(*len).min(tail.len());
                    tail[..len].fill(*byte);
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct IpSwitchPatch {
    enabled: bool,
    records: BTreeMap<u32, Vec<u8>>,
}

/// An IPSwitch text patch: `@nsobid-<build id>`, then `@enabled` or `@disabled` blocks of
/// `<hex offset> <hex bytes>` or `<hex offset> "string"` records. `@flag offset_shift <n>`
/// moves later offsets and `@stop` ends parsing. Lines that are not records are ignored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IpSwitch {
    build_id: [u8; 32],
    patches: Vec<IpSwitchPatch>,
}

impl IpSwitch {
    /// Parses IPSwitch text. Never fails: unknown flags, lines before the first block and
    /// malformed records are skipped, as the reference parsers do.
    pub fn parse(text: &[u8]) -> Self {
        let mut out = Self {
            build_id: [0; 32],
            patches: Vec::new(),
        };
        let mut offset_shift = 0i64;
        let mut little_endian = true;
        for line in text.split(|&b| b == b'\n' || b == b'\r') {
            let line = strip_comment(line);
            if line.is_empty() {
                continue;
            }
            if line.len() > 2 && line[0] == b'@' {
                match line[1].to_ascii_lowercase() {
                    b'n' => {
                        let hex = String::from_utf8_lossy(line.get(8..).unwrap_or_default());
                        out.build_id = parse_build_id(&hex).unwrap_or_default();
                    }
                    b's' => break,
                    b'e' | b'd' => out.patches.push(IpSwitchPatch {
                        enabled: line[1].eq_ignore_ascii_case(&b'e'),
                        records: BTreeMap::new(),
                    }),
                    b'f' => {
                        if let Some(value) = line.strip_prefix(b"@flag offset_shift") {
                            offset_shift = parse_c_integer(value);
                        }
                    }
                    b'l' => little_endian = true,
                    b'b' => little_endian = false,
                    _ => tracing::warn!("IPSwitch: unknown flag {}", String::from_utf8_lossy(line)),
                }
                continue;
            }
            let Some(patch) = out.patches.last_mut() else {
                continue;
            };
            if !is_patch_record(line) {
                continue;
            }
            let offset = parse_hex_prefix(line).wrapping_add(offset_shift as u64) as u32;
            let bytes = if let Some(quote) = line.iter().position(|&b| b == b'"' || b == b'\'') {
                let start = quote + 1;
                let mut end = start;
                while end < line.len() && line[end] != line[quote] {
                    end += if line[end] == b'\\' { 2 } else { 1 };
                }
                if end > line.len() {
                    continue;
                }
                unescape(&line[start..end])
            } else {
                let value_start = line
                    .iter()
                    .rposition(|&b| matches!(b, b' ' | b'/' | b'\t'))
                    .map_or(0, |p| p + 1);
                let hex = &line[value_start..];
                let hex = &hex[if little_endian { hex.len() % 2 } else { 0 }..];
                hex.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|&[hi, lo]| (hex_digit(hi) << 4) | hex_digit(lo))
                    .collect()
            };
            patch.records.insert(offset, bytes);
        }
        out
    }

    /// The build id from `@nsobid-`, zero when the patch names none.
    pub fn build_id(&self) -> &[u8; 32] {
        &self.build_id
    }

    /// Writes the records of every enabled block, clipping at the end of `target` and
    /// skipping records that start past it.
    pub fn apply(&self, target: &mut [u8]) {
        for patch in self.patches.iter().filter(|p| p.enabled) {
            for (&offset, bytes) in &patch.records {
                let Some(tail) = target.get_mut(offset as usize..) else {
                    continue;
                };
                let len = bytes.len().min(tail.len());
                tail[..len].copy_from_slice(&bytes[..len]);
            }
        }
    }
}

fn is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c)
}

fn hex_digit(byte: u8) -> u8 {
    char::from(byte).to_digit(16).unwrap_or(0) as u8
}

/// Trims the line and cuts a `/` or `#` comment that is not inside quotes.
fn strip_comment(line: &[u8]) -> &[u8] {
    let start = line
        .iter()
        .position(|&b| !is_space(b))
        .unwrap_or(line.len());
    let mut quote = None;
    let mut end = start;
    let mut i = start;
    while i < line.len() {
        let byte = line[i];
        if is_space(byte) {
            i += 1;
            continue;
        }
        if quote.is_none() && (byte == b'/' || byte == b'#') {
            break;
        }
        if byte == b'"' || byte == b'\'' {
            match quote {
                None => quote = Some(byte),
                Some(open) if open == byte => quote = None,
                Some(_) => {}
            }
            i += 1;
        } else if byte == b'\\' && i + 1 < line.len() {
            i += 2;
        } else {
            i += 1;
        }
        end = i;
    }
    &line[start..end]
}

/// A record is a hex offset followed by hex digits or by a quoted string that ends the line.
fn is_patch_record(line: &[u8]) -> bool {
    let Some(offset_end) = line
        .iter()
        .position(|&b| matches!(b, b' ' | b'\t' | b'"' | b'\''))
    else {
        return false;
    };
    let offset = &line[..offset_end];
    let offset = offset
        .strip_prefix(b"0x")
        .or_else(|| offset.strip_prefix(b"0X"))
        .unwrap_or(offset);
    if offset.is_empty() || !offset.iter().all(u8::is_ascii_hexdigit) {
        return false;
    }
    let Some(value) = line[offset_end..]
        .iter()
        .position(|&b| b != b' ' && b != b'\t')
        .map(|p| &line[offset_end + p..])
    else {
        return false;
    };
    let quote = value[0];
    if quote == b'"' || quote == b'\'' {
        let mut i = 1;
        while i < value.len() {
            if value[i] == b'\\' {
                i += 2;
                continue;
            }
            if value[i] == quote {
                return i + 1 == value.len();
            }
            i += 1;
        }
        return false;
    }
    value
        .iter()
        .all(|&b| b.is_ascii_hexdigit() || b == b' ' || b == b'\t')
}

fn parse_hex_prefix(line: &[u8]) -> u64 {
    let digits = line
        .strip_prefix(b"0x")
        .or_else(|| line.strip_prefix(b"0X"))
        .unwrap_or(line);
    digits
        .iter()
        .map_while(|&b| char::from(b).to_digit(16))
        .fold(0u64, |acc, d| {
            acc.saturating_mul(16).saturating_add(u64::from(d))
        })
}

/// `strtoll` with base 0: optional sign, then `0x` hex, `0` octal or decimal.
fn parse_c_integer(text: &[u8]) -> i64 {
    let text = &text[text
        .iter()
        .position(|&b| !is_space(b))
        .unwrap_or(text.len())..];
    let (negative, text) = match text.first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let (radix, digits) = if let Some(hex) = text
        .strip_prefix(b"0x")
        .or_else(|| text.strip_prefix(b"0X"))
    {
        (16, hex)
    } else if text.len() > 1 && text[0] == b'0' {
        (8, &text[1..])
    } else {
        (10, text)
    };
    let value = digits
        .iter()
        .map_while(|&b| char::from(b).to_digit(radix))
        .fold(0i64, |acc, d| {
            acc.wrapping_mul(i64::from(radix))
                .wrapping_add(i64::from(d))
        });
    if negative {
        value.wrapping_neg()
    } else {
        value
    }
}

fn unescape(text: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i] == b'\\' && i + 1 < text.len() {
            out.push(match text[i + 1] {
                b'a' => 0x07,
                b'b' => 0x08,
                b'e' => 0x1b,
                b'f' => 0x0c,
                b'n' => b'\n',
                b'r' => b'\r',
                b't' => b'\t',
                b'v' => 0x0b,
                other => other,
            });
            i += 2;
        } else {
            out.push(text[i]);
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ips(records: &[(u32, &[u8])], rle: Option<(u32, u16, u8)>) -> Vec<u8> {
        let mut out = b"PATCH".to_vec();
        for &(offset, bytes) in records {
            out.extend_from_slice(&offset.to_be_bytes()[1..]);
            out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
            out.extend_from_slice(bytes);
        }
        if let Some((offset, len, byte)) = rle {
            out.extend_from_slice(&offset.to_be_bytes()[1..]);
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&len.to_be_bytes());
            out.push(byte);
        }
        out.extend_from_slice(b"EOF");
        out
    }

    #[test]
    fn build_ids_pad_and_reject() {
        let id = parse_build_id("ABcd01").unwrap();
        assert_eq!(id[..4], [0xab, 0xcd, 0x01, 0x00]);
        assert_eq!(parse_build_id("abc").unwrap()[..2], [0xab, 0xc0]);
        assert!(parse_build_id("xyz").is_none());
        assert!(parse_build_id(&"0".repeat(65)).is_none());
    }

    #[test]
    fn ips_copies_fills_and_clips() {
        let patch = Ips::parse(&ips(&[(1, b"ab"), (6, b"xyz")], Some((3, 2, 0xee)))).unwrap();
        let mut target = [0u8; 8];
        patch.apply(&mut target).unwrap();
        assert_eq!(target, [0, b'a', b'b', 0xee, 0xee, 0, b'x', b'y']);
    }

    #[test]
    fn ips32_uses_four_byte_offsets() {
        let mut data = b"IPS32".to_vec();
        data.extend_from_slice(&0x0001_0002u32.to_be_bytes());
        data.extend_from_slice(&1u16.to_be_bytes());
        data.push(0x5a);
        data.extend_from_slice(b"EEOF");
        let mut target = vec![0u8; 0x1_0004];
        Ips::parse(&data).unwrap().apply(&mut target).unwrap();
        assert_eq!(target[0x1_0002], 0x5a);
    }

    #[test]
    fn ips_refuses_bad_input_without_writing() {
        assert!(Ips::parse(b"NOPE!EOF").is_err());
        let mut no_eof = ips(&[(0, b"a")], None);
        no_eof.truncate(no_eof.len() - 3);
        assert!(Ips::parse(&no_eof).is_err());
        let mut short = ips(&[(0, b"abcd")], None);
        short.truncate(short.len() - 5);
        assert!(Ips::parse(&short).is_err());

        let patch = Ips::parse(&ips(&[(0, b"a"), (9, b"b")], None)).unwrap();
        let mut target = [0u8; 8];
        assert!(patch.apply(&mut target).is_err());
        assert_eq!(target, [0; 8]);
    }

    #[test]
    fn ipswitch_parses_blocks_flags_and_comments() {
        let text = b"@nsobid-0123456789ABCDEF\r\n\
            # header comment\n\
            junk before any block\n\
            @enabled\n\
            00000010 AABBCC // trailing comment\n\
            0x00000014 \"hi\\n/#\"\n\
            Credits: someone\n\
            @disabled\n\
            00000000 FFFF\n\
            @enabled\n\
            @flag offset_shift 0x100\n\
            00000004 01 02\n\
            00000008 ABC\n\
            @stop\n\
            00000000 EE\n";
        let patch = IpSwitch::parse(text);
        assert_eq!(
            patch.build_id()[..9],
            [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0]
        );
        let mut target = vec![0u8; 0x10a];
        patch.apply(&mut target);
        assert_eq!(target[..4], [0, 0, 0, 0]);
        assert_eq!(target[0x10..0x13], [0xaa, 0xbb, 0xcc]);
        assert_eq!(&target[0x14..0x19], b"hi\n/#");
        assert_eq!(target[0x104], 0x02);
        assert_eq!(target[0x105], 0);
        assert_eq!(target[0x108..0x10a], [0xbc, 0]);
    }

    #[test]
    fn ipswitch_clips_and_ignores_out_of_range() {
        let patch = IpSwitch::parse(b"@enabled\n2 AABBCC\n10 DD\n");
        let mut target = [0u8; 4];
        patch.apply(&mut target);
        assert_eq!(target, [0, 0, 0xaa, 0xbb]);
    }

    #[test]
    fn c_integers_follow_strtoll() {
        assert_eq!(parse_c_integer(b" 0x10"), 16);
        assert_eq!(parse_c_integer(b" -010"), -8);
        assert_eq!(parse_c_integer(b" 42abc"), 42);
        assert_eq!(parse_c_integer(b""), 0);
    }
}
