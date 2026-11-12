//! O(1) Maxwell opcode decoding.

use super::opcode::Opcode;

static TABLE: [Option<Opcode>; 1 << 16] = build_table();

/// Fills every 16-bit prefix with its opcode. Opcodes are written lowest priority first, so a
/// prefix ends up holding the first opcode of `maxwell.inc` that matches it.
const fn build_table() -> [Option<Opcode>; 1 << 16] {
    let mut table = [None; 1 << 16];
    let mut i = Opcode::ALL.len();
    while i > 0 {
        i -= 1;
        let opcode = Opcode::ALL[i];
        let pattern = opcode.pattern();
        let free = !pattern.mask;
        let mut subset = 0u16;
        loop {
            table[(pattern.value | subset) as usize] = Some(opcode);
            subset = subset.wrapping_sub(free) & free;
            if subset == 0 {
                break;
            }
        }
    }
    table
}

/// Decodes the opcode of a Maxwell instruction, or `None` when no opcode matches.
pub fn decode(insn: u64) -> Option<Opcode> {
    TABLE[(insn >> 48) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(insn: u64) -> Option<Opcode> {
        Opcode::ALL
            .iter()
            .copied()
            .find(|opcode| opcode.pattern().matches(insn))
    }

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn encode(opcode: Opcode, dont_care: u64) -> u64 {
        let pattern = opcode.pattern();
        let fixed = u64::from(pattern.mask) << 48;
        (u64::from(pattern.value) << 48) | (dont_care & !fixed)
    }

    #[test]
    fn table_matches_linear_first_match_scan() {
        for prefix in 0..=u16::MAX {
            let insn = u64::from(prefix) << 48;
            assert_eq!(decode(insn), linear(insn), "prefix {prefix:#06x}");
        }
    }

    #[test]
    fn low_bits_never_change_the_opcode() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        for _ in 0..4096 {
            let insn = rng.next();
            let prefix_only = insn & 0xffff_0000_0000_0000;
            assert_eq!(decode(insn), decode(prefix_only));
        }
    }

    #[test]
    fn every_opcode_decodes_to_itself_or_an_earlier_entry() {
        let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
        for &opcode in Opcode::ALL {
            for dont_care in [0, u64::MAX, rng.next(), rng.next(), rng.next()] {
                let insn = encode(opcode, dont_care);
                let decoded = decode(insn).expect("pattern words always decode");
                assert!(decoded <= opcode, "{opcode} decoded as later {decoded}");
                assert_eq!(Some(decoded), linear(insn));
            }
        }
    }

    #[test]
    fn hand_encoded_words() {
        let cases = [
            (0xe300_0000_0007_000f, Opcode::Exit),
            (0x50b0_0000_0007_0f00, Opcode::Nop),
            (0xe240_0fff_ff87_000f, Opcode::Bra),
            (0xe290_0000_0300_0000, Opcode::Ssy),
            (0xf0f8_0000_0000_000f, Opcode::Sync),
            (0xe2a0_0000_0400_0000, Opcode::Pbk),
            (0xe340_0000_0000_000f, Opcode::Brk),
            (0xe330_0000_0007_000f, Opcode::Kil),
            (0xe260_0000_0100_0040, Opcode::Cal),
            (0xe320_0000_0007_000f, Opcode::Ret),
            (0xe250_0000_0007_030f, Opcode::Brx),
            (0x0103_f800_0007_f000, Opcode::Mov32i),
            (0x5c98_0780_0007_0002, Opcode::MovReg),
            (0x4c98_0780_0207_0002, Opcode::MovCbuf),
            (0x3898_0780_0007_0002, Opcode::MovImm),
            (0x5c58_0000_0027_0100, Opcode::FaddReg),
            (0x0883_f800_0007_0100, Opcode::Fadd32i),
            (0x5980_0000_0017_0201, Opcode::FfmaReg),
            (0x5b6b_0380_0007_0207, Opcode::IsetpReg),
            (0x5b00_0000_0017_0201, Opcode::XmadReg),
            (0xef94_0000_0207_0203, Opcode::Ldc),
            (0x3848_0000_0027_0201, Opcode::ShlImm),
            (0x3820_0000_0037_0001, Opcode::ImnmxImm),
            (0xeff0_0000_0007_ff00, Opcode::Ast),
            (0xefd8_0000_0007_ff00, Opcode::Ald),
            (0xe003_ff87_cff7_ff03, Opcode::Ipa),
            (0xf0c8_0000_0217_0000, Opcode::S2r),
            (0xc038_0000_0007_0200, Opcode::Tex),
            (0xd822_0000_0007_0200, Opcode::Texs),
            (0x8000_0000_0007_0200, Opcode::Imad32i),
            (0x9000_0000_0007_0200, Opcode::Ld),
            (0xa000_0000_0007_0200, Opcode::St),
            (0xeed4_0000_0007_0200, Opcode::Ldg),
            (0xef10_0000_0007_0200, Opcode::Shfl),
            (0x5080_0000_0007_0200, Opcode::Mufu),
        ];
        for (insn, expected) in cases {
            assert_eq!(decode(insn), Some(expected), "{insn:#018x}");
            assert!(expected.pattern().matches(insn));
        }
    }

    #[test]
    fn unassigned_prefixes_do_not_decode() {
        assert_eq!(decode(0xffff_ffff_ffff_ffff), None);
        assert_eq!(decode(0x0000_0000_0000_0000), None);
        assert_eq!(decode(0xf000_0000_0000_0000), None);
    }
}
