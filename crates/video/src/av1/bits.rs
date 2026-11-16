//! The AV1 bit reader (AV1 specification section 4.10): MSB-first fixed-width fields, `uvlc`,
//! `le`, `leb128`, `su`, `ns` and the sub-exponential codes of global motion parameters.

/// Why a header did not parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Av1Error {
    /// The data ended inside a header or an OBU.
    #[error("AV1 data ends early")]
    Truncated,
    /// A field holds a value the specification forbids, or one this reader does not follow.
    #[error("AV1 header field {0} is invalid or unsupported")]
    Invalid(&'static str),
    /// A frame came before any sequence header.
    #[error("AV1 frame without a sequence header")]
    NoSequenceHeader,
}

pub(crate) type Result<T> = std::result::Result<T, Av1Error>;

/// Reads bits MSB-first from a byte slice.
pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    /// Bits read so far.
    pub(crate) fn position(&self) -> usize {
        self.position
    }

    /// `f(n)`, `n` at most 32.
    pub(crate) fn f(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        let mut value = 0u64;
        for _ in 0..n {
            let byte = *self
                .data
                .get(self.position / 8)
                .ok_or(Av1Error::Truncated)?;
            let bit = (byte >> (7 - self.position % 8)) & 1;
            value = value << 1 | u64::from(bit);
            self.position += 1;
        }
        Ok(value as u32)
    }

    pub(crate) fn flag(&mut self) -> Result<bool> {
        Ok(self.f(1)? != 0)
    }

    /// `uvlc()`: leading zeros, a one, then as many value bits.
    pub(crate) fn uvlc(&mut self) -> Result<u32> {
        let mut leading = 0;
        while !self.flag()? {
            leading += 1;
            if leading >= 32 {
                return Ok(u32::MAX);
            }
        }
        let value = self.f(leading)?;
        Ok(value + ((1u64 << leading) - 1) as u32)
    }

    /// `su(n)`: an `n`-bit two's complement value.
    pub(crate) fn su(&mut self, n: u32) -> Result<i32> {
        let value = self.f(n)? as i64;
        let sign = 1i64 << (n - 1);
        Ok((if value & sign != 0 {
            value - 2 * sign
        } else {
            value
        }) as i32)
    }

    /// `ns(n)`: a value below `n`, in fewer bits for the small ones.
    pub(crate) fn ns(&mut self, n: u32) -> Result<u32> {
        if n <= 1 {
            return Ok(0);
        }
        let w = 32 - n.leading_zeros();
        let m = (1u32 << w) - n;
        let v = self.f(w - 1)?;
        if v < m {
            return Ok(v);
        }
        let extra = self.f(1)?;
        Ok((v << 1) - m + extra)
    }

    /// `le(n)`: `n` little-endian bytes, byte aligned.
    pub(crate) fn le(&mut self, n: u32) -> Result<u64> {
        let mut value = 0u64;
        for i in 0..n {
            value |= u64::from(self.f(8)?) << (8 * i);
        }
        Ok(value)
    }

    /// Skips to the next byte boundary.
    pub(crate) fn byte_alignment(&mut self) {
        self.position = self.position.next_multiple_of(8);
    }

    /// `decode_signed_subexp_with_ref(low, high, r)`.
    pub(crate) fn signed_subexp_with_ref(&mut self, low: i32, high: i32, r: i32) -> Result<i32> {
        let x = self.unsigned_subexp_with_ref(high - low, r - low)?;
        Ok(x + low)
    }

    fn unsigned_subexp_with_ref(&mut self, mx: i32, r: i32) -> Result<i32> {
        let v = self.subexp(mx)?;
        Ok(if (r << 1) <= mx {
            inverse_recenter(r, v)
        } else {
            mx - 1 - inverse_recenter(mx - 1 - r, v)
        })
    }

    fn subexp(&mut self, num_syms: i32) -> Result<i32> {
        let (mut i, mut mk, k) = (0, 0, 3);
        loop {
            let b2 = if i != 0 { k + i - 1 } else { k };
            let a = 1 << b2;
            if num_syms <= mk + 3 * a {
                return Ok(self.ns((num_syms - mk) as u32)? as i32 + mk);
            }
            if self.flag()? {
                i += 1;
                mk += a;
            } else {
                return Ok(self.f(b2 as u32)? as i32 + mk);
            }
        }
    }
}

fn inverse_recenter(r: i32, v: i32) -> i32 {
    if v > 2 * r {
        v
    } else if v & 1 != 0 {
        r - ((v + 1) >> 1)
    } else {
        r + (v >> 1)
    }
}

/// `leb128()` at the start of `data`: the value and the bytes it took.
pub(crate) fn leb128(data: &[u8]) -> Result<(u64, usize)> {
    let mut value = 0u64;
    for i in 0..8 {
        let byte = *data.get(i).ok_or(Av1Error::Truncated)?;
        value |= u64::from(byte & 0x7f) << (7 * i);
        if byte & 0x80 == 0 {
            return Ok((value, i + 1));
        }
    }
    Err(Av1Error::Invalid("leb128"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Writes bits MSB-first, for building test streams.
    #[derive(Default)]
    pub(crate) struct BitWriter {
        pub(crate) bytes: Vec<u8>,
        bits: usize,
    }

    impl BitWriter {
        pub(crate) fn f(&mut self, value: u64, n: u32) -> &mut Self {
            for i in (0..n).rev() {
                if self.bits.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let bit = (value >> i) & 1;
                *self.bytes.last_mut().unwrap() |= (bit as u8) << (7 - self.bits % 8);
                self.bits += 1;
            }
            self
        }

        pub(crate) fn flag(&mut self, value: bool) -> &mut Self {
            self.f(u64::from(value), 1)
        }

        pub(crate) fn su(&mut self, value: i32, n: u32) -> &mut Self {
            self.f((value as u64) & ((1 << n) - 1), n)
        }

        pub(crate) fn uvlc(&mut self, value: u32) -> &mut Self {
            let code = u64::from(value) + 1;
            let size = 64 - code.leading_zeros();
            self.f(0, size - 1).f(code, size)
        }

        pub(crate) fn align(&mut self) -> &mut Self {
            self.bits = self.bits.next_multiple_of(8);
            self
        }
    }

    pub(crate) fn leb128_bytes(mut value: u64) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            if value == 0 {
                out.push(byte);
                return out;
            }
            out.push(byte | 0x80);
        }
    }

    #[test]
    fn fields_read_back_as_written() {
        let mut w = BitWriter::default();
        w.f(5, 3)
            .flag(true)
            .su(-3, 7)
            .uvlc(0)
            .uvlc(5)
            .uvlc(1000)
            .f(0xabcd, 16);
        let mut r = BitReader::new(&w.bytes);
        assert_eq!(r.f(3), Ok(5));
        assert_eq!(r.flag(), Ok(true));
        assert_eq!(r.su(7), Ok(-3));
        assert_eq!(r.uvlc(), Ok(0));
        assert_eq!(r.uvlc(), Ok(5));
        assert_eq!(r.uvlc(), Ok(1000));
        assert_eq!(r.f(16), Ok(0xabcd));
        assert_eq!(r.f(16), Err(Av1Error::Truncated));
    }

    #[test]
    fn ns_reads_every_value_below_n() {
        for n in [1u32, 2, 3, 5, 7, 8, 13, 64] {
            for v in 0..n {
                let w_bits = 32 - n.leading_zeros();
                let m = (1u32 << w_bits) - n;
                let mut w = BitWriter::default();
                if n > 1 {
                    if v < m {
                        w.f(u64::from(v), w_bits - 1);
                    } else {
                        let coded = v + m;
                        w.f(u64::from(coded >> 1), w_bits - 1)
                            .f(u64::from(coded & 1), 1);
                    }
                }
                w.f(0, 8);
                assert_eq!(BitReader::new(&w.bytes).ns(n), Ok(v), "n {n} v {v}");
            }
        }
    }

    #[test]
    fn leb128_reads_one_to_eight_bytes() {
        for value in [0u64, 1, 127, 128, 300, 1 << 20, (1 << 56) - 1] {
            let bytes = leb128_bytes(value);
            assert_eq!(leb128(&bytes), Ok((value, bytes.len())));
        }
        assert_eq!(leb128(&[0x80; 9]), Err(Av1Error::Invalid("leb128")));
        assert_eq!(leb128(&[0x80]), Err(Av1Error::Truncated));
    }

    #[test]
    fn little_endian_bytes() {
        let mut r = BitReader::new(&[0x34, 0x12, 0xff]);
        assert_eq!(r.le(2), Ok(0x1234));
    }
}
