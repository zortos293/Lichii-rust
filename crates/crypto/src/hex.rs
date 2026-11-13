/// Parses exactly `N` bytes of hex, ignoring ASCII whitespace. `None` on bad digits or length.
pub fn parse_hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    let mut out = [0u8; N];
    let mut digits = text.bytes().filter(|b| !b.is_ascii_whitespace());
    for byte in &mut out {
        let high = hex_digit(digits.next()?)?;
        let low = hex_digit(digits.next()?)?;
        *byte = (high << 4) | low;
    }
    digits.next().is_none().then_some(out)
}

/// The first `N` bytes of `text`, which must start with at least `2 * N` hex digits;
/// whitespace is skipped and anything after those digits is ignored. Key files are read this
/// way, because some dumpers append bytes to 16-byte sources and other emulators accept them.
pub(crate) fn parse_hex_prefix<const N: usize>(text: &str) -> Option<[u8; N]> {
    let mut out = [0u8; N];
    let mut digits = text.bytes().filter(|b| !b.is_ascii_whitespace());
    for byte in &mut out {
        let high = hex_digit(digits.next()?)?;
        let low = hex_digit(digits.next()?)?;
        *byte = (high << 4) | low;
    }
    Some(out)
}

fn hex_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'a'..=b'f' => Some(c - b'a' + 10),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_exact_lengths() {
        assert_eq!(parse_hex::<2>("aB 0f"), Some([0xab, 0x0f]));
        assert_eq!(parse_hex::<2>("ab0"), None);
        assert_eq!(parse_hex::<2>("ab0f00"), None);
        assert_eq!(parse_hex::<1>("zz"), None);
    }

    #[test]
    fn prefixes_ignore_what_follows_the_key() {
        assert_eq!(parse_hex_prefix::<2>("aB 0f"), Some([0xab, 0x0f]));
        assert_eq!(parse_hex_prefix::<2>("ab0f00"), Some([0xab, 0x0f]));
        assert_eq!(parse_hex_prefix::<2>("ab0fzz"), Some([0xab, 0x0f]));
        assert_eq!(parse_hex_prefix::<2>("ab0"), None);
        assert_eq!(parse_hex_prefix::<2>("abzz"), None);
    }
}
