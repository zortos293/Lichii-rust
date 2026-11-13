//! Bit-level writers for the headers the builders compose: an MSB-first bit packer (H.264 RBSP
//! with Exp-Golomb codes, VP9 uncompressed header) and the VP8/VP9 boolean range encoder.

/// Packs bits MSB-first into a borrowed buffer, appending whole bytes as they fill.
pub(crate) struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    acc: u64,
    pending: u32,
}

impl<'a> BitWriter<'a> {
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        Self {
            out,
            acc: 0,
            pending: 0,
        }
    }

    /// Writes the low `count` (at most 56) bits of `value`.
    pub(crate) fn bits(&mut self, value: u64, count: u32) {
        debug_assert!(count <= 56);
        if count == 0 {
            return;
        }
        self.acc = self.acc << count | value & ((1 << count) - 1);
        self.pending += count;
        while self.pending >= 8 {
            self.pending -= 8;
            self.out.push((self.acc >> self.pending) as u8);
        }
        self.acc &= (1 << self.pending) - 1;
    }

    pub(crate) fn bit(&mut self, bit: bool) {
        self.bits(u64::from(bit), 1);
    }

    /// Unsigned Exp-Golomb code, ue(v).
    pub(crate) fn ue(&mut self, value: u32) {
        let code = u64::from(value) + 1;
        let size = 64 - code.leading_zeros();
        self.bits(0, size - 1);
        self.bits(code, size);
    }

    /// Signed Exp-Golomb code, se(v).
    pub(crate) fn se(&mut self, value: i32) {
        let value = i64::from(value);
        let code = if value > 0 { 2 * value - 1 } else { -2 * value };
        self.ue(code as u32);
    }

    /// VP9 su(n): `size` bits of magnitude, then the sign.
    pub(crate) fn signed(&mut self, value: i32, size: u32) {
        self.bits(u64::from(value.unsigned_abs()), size);
        self.bit(value < 0);
    }

    /// Pads the last byte with zero bits.
    pub(crate) fn flush(&mut self) {
        if self.pending != 0 {
            let pad = 8 - self.pending;
            self.bits(0, pad);
        }
    }

    /// H.264 rbsp_trailing_bits: a stop bit, then zero padding.
    pub(crate) fn rbsp_trailing_bits(&mut self) {
        self.bit(true);
        self.flush();
    }
}

/// The VP8/VP9 boolean encoder (libvpx `vpx_writer`), appending to a borrowed buffer.
pub(crate) struct RangeEncoder<'a> {
    out: &'a mut Vec<u8>,
    low: u32,
    range: u32,
    count: i32,
}

impl<'a> RangeEncoder<'a> {
    /// Starts a stream: writes the marker bit the decoder expects to read as 0.
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        let mut encoder = Self {
            out,
            low: 0,
            range: 0xFF,
            count: -24,
        };
        encoder.bit(false);
        encoder
    }

    /// Writes `bit` with probability `probability`/256 of it being 0.
    pub(crate) fn bool(&mut self, bit: bool, probability: u32) {
        let split = 1 + (((self.range - 1) * probability) >> 8);
        let mut range = split;
        if bit {
            self.low = self.low.wrapping_add(split);
            range = self.range - split;
        }
        let mut shift = range.leading_zeros() as i32 - 24;
        range <<= shift;
        self.count += shift;
        if self.count >= 0 {
            let offset = shift - self.count;
            if self.low.wrapping_shl((offset - 1) as u32) & 0x8000_0000 != 0 {
                self.propagate_carry();
            }
            self.out.push((self.low >> (24 - offset)) as u8);
            self.low = self.low.wrapping_shl(offset as u32) & 0xFF_FFFF;
            shift = self.count;
            self.count -= 8;
        }
        self.low = self.low.wrapping_shl(shift as u32);
        self.range = range;
    }

    fn propagate_carry(&mut self) {
        for byte in self.out.iter_mut().rev() {
            if *byte != 0xFF {
                *byte += 1;
                return;
            }
            *byte = 0;
        }
    }

    /// Writes a bit with even probability.
    pub(crate) fn bit(&mut self, bit: bool) {
        self.bool(bit, 128);
    }

    /// Writes the low `count` bits of `value` MSB-first, each with even probability.
    pub(crate) fn literal(&mut self, value: u32, count: u32) {
        for bit in (0..count).rev() {
            self.bit(value >> bit & 1 != 0);
        }
    }

    /// Flushes the stream.
    pub(crate) fn finish(mut self) {
        for _ in 0..32 {
            self.bit(false);
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn written(f: impl FnOnce(&mut BitWriter)) -> Vec<u8> {
        let mut out = Vec::new();
        let mut writer = BitWriter::new(&mut out);
        f(&mut writer);
        writer.flush();
        out
    }

    fn bit_string(bytes: &[u8], len: usize) -> String {
        bytes
            .iter()
            .flat_map(|b| {
                (0..8)
                    .rev()
                    .map(move |i| if b >> i & 1 != 0 { '1' } else { '0' })
            })
            .take(len)
            .collect()
    }

    #[test]
    fn bits_pack_msb_first_across_byte_boundaries() {
        let out = written(|w| {
            w.bits(0b101, 3);
            w.bits(0x1FF, 9);
            w.bits(0xFFFF_FFFF_FFFF, 4);
            w.bits(1, 24);
            w.bit(true);
        });
        assert_eq!(
            out,
            [0b1011_1111, 0b1111_1111, 0, 0, 0b0000_0001, 0b1000_0000]
        );
    }

    #[test]
    fn exp_golomb_codes() {
        let cases: &[(u32, &str)] = &[
            (0, "1"),
            (1, "010"),
            (2, "011"),
            (3, "00100"),
            (6, "00111"),
            (7, "0001000"),
            (254, "000000011111111"),
        ];
        for &(value, code) in cases {
            let out = written(|w| w.ue(value));
            assert_eq!(bit_string(&out, code.len()), code, "ue({value})");
        }
        let signed: &[(i32, &str)] = &[
            (0, "1"),
            (1, "010"),
            (-1, "011"),
            (2, "00100"),
            (-2, "00101"),
            (-3, "00111"),
        ];
        for &(value, code) in signed {
            let out = written(|w| w.se(value));
            assert_eq!(bit_string(&out, code.len()), code, "se({value})");
        }
        let out = written(|w| w.ue(u32::MAX));
        assert_eq!(
            bit_string(&out, 65),
            format!("{}1{}", "0".repeat(32), "0".repeat(32))
        );
    }

    #[test]
    fn rbsp_trailing_bits_and_signed_values() {
        assert_eq!(written(|w| w.rbsp_trailing_bits()), [0x80]);
        assert_eq!(
            written(|w| {
                w.bits(0x3F, 6);
                w.rbsp_trailing_bits()
            }),
            [0xFE]
        );
        assert_eq!(
            written(|w| {
                w.bits(0xFF, 8);
                w.rbsp_trailing_bits()
            }),
            [0xFF, 0x80]
        );
        assert_eq!(bit_string(&written(|w| w.signed(-5, 6)), 7), "0001011");
        assert_eq!(bit_string(&written(|w| w.signed(5, 4)), 5), "01010");
    }

    /// The RFC 6386 boolean decoder, for round trips.
    pub(crate) struct RangeDecoder<'a> {
        data: &'a [u8],
        pos: usize,
        value: u32,
        range: u32,
        bit_count: u32,
    }

    impl<'a> RangeDecoder<'a> {
        pub(crate) fn new(data: &'a [u8]) -> Self {
            let mut decoder = Self {
                data,
                pos: 0,
                value: 0,
                range: 255,
                bit_count: 0,
            };
            decoder.value = u32::from(decoder.next_byte()) << 8 | u32::from(decoder.next_byte());
            assert!(!decoder.bool(128), "marker bit");
            decoder
        }

        fn next_byte(&mut self) -> u8 {
            let byte = self.data.get(self.pos).copied().unwrap_or(0);
            self.pos += 1;
            byte
        }

        pub(crate) fn bool(&mut self, probability: u32) -> bool {
            let split = 1 + (((self.range - 1) * probability) >> 8);
            let big_split = split << 8;
            let bit = if self.value >= big_split {
                self.range -= split;
                self.value -= big_split;
                true
            } else {
                self.range = split;
                false
            };
            while self.range < 128 {
                self.value <<= 1;
                self.range <<= 1;
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.bit_count = 0;
                    self.value |= u32::from(self.next_byte());
                }
            }
            bit
        }

        pub(crate) fn literal(&mut self, count: u32) -> u32 {
            (0..count).fold(0, |v, _| v << 1 | u32::from(self.bool(128)))
        }
    }

    #[test]
    fn range_coder_round_trips_random_symbols() {
        let mut state = 0x1234_5678_u32;
        let mut symbols = Vec::new();
        for _ in 0..20_000 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let probability = 1 + state % 255;
            let bit = (state >> 9) % 256 >= probability;
            symbols.push((bit, probability));
        }
        let mut out = Vec::new();
        let mut encoder = RangeEncoder::new(&mut out);
        for &(bit, probability) in &symbols {
            encoder.bool(bit, probability);
        }
        encoder.literal(0x2A5, 10);
        encoder.finish();

        let mut decoder = RangeDecoder::new(&out);
        for (i, &(bit, probability)) in symbols.iter().enumerate() {
            assert_eq!(decoder.bool(probability), bit, "symbol {i}");
        }
        assert_eq!(decoder.literal(10), 0x2A5);
    }

    #[test]
    fn range_coder_carry_increments_through_ff_bytes() {
        let mut out = vec![0x12, 0xFF, 0xFF];
        let mut encoder = RangeEncoder::new(&mut out);
        encoder.propagate_carry();
        assert_eq!(out, [0x13, 0x00, 0x00]);
    }
}
