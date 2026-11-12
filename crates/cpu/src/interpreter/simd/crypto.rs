//! Cryptographic extensions: AES, SHA-1, SHA-256, SHA-512, SHA-3, SM3 and SM4.

use super::{Exec, Undefined, bit, bits};
use crate::ThreadContext;

const fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut product = 0;
    while b != 0 {
        if b & 1 != 0 {
            product ^= a;
        }
        let carry = a & 0x80;
        a <<= 1;
        if carry != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    product
}

const fn mul_table(c: u8) -> [u8; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        table[i] = gf_mul(i as u8, c);
        i += 1;
    }
    table
}

const fn aes_sbox() -> [u8; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let x = i as u8;
        let mut inverse = 1;
        let mut bit = 8;
        while bit > 0 {
            bit -= 1;
            inverse = gf_mul(inverse, inverse);
            if (0xfe >> bit) & 1 != 0 {
                inverse = gf_mul(inverse, x);
            }
        }
        table[i] = inverse
            ^ inverse.rotate_left(1)
            ^ inverse.rotate_left(2)
            ^ inverse.rotate_left(3)
            ^ inverse.rotate_left(4)
            ^ 0x63;
        i += 1;
    }
    table
}

const fn invert_table(table: &[u8; 256]) -> [u8; 256] {
    let mut inverse = [0; 256];
    let mut i = 0;
    while i < 256 {
        inverse[table[i] as usize] = i as u8;
        i += 1;
    }
    inverse
}

const AES_SBOX: [u8; 256] = aes_sbox();
const AES_INV_SBOX: [u8; 256] = invert_table(&AES_SBOX);
const MUL1: [u8; 256] = mul_table(1);
const MUL2: [u8; 256] = mul_table(2);
const MUL3: [u8; 256] = mul_table(3);
const MUL9: [u8; 256] = mul_table(9);
const MUL11: [u8; 256] = mul_table(11);
const MUL13: [u8; 256] = mul_table(13);
const MUL14: [u8; 256] = mul_table(14);

const AES_SHIFT_ROWS: [usize; 16] = [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11];
const AES_INV_SHIFT_ROWS: [usize; 16] = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3];

const SM4_SBOX: [u8; 256] = [
    0xd6, 0x90, 0xe9, 0xfe, 0xcc, 0xe1, 0x3d, 0xb7, 0x16, 0xb6, 0x14, 0xc2, 0x28, 0xfb, 0x2c, 0x05,
    0x2b, 0x67, 0x9a, 0x76, 0x2a, 0xbe, 0x04, 0xc3, 0xaa, 0x44, 0x13, 0x26, 0x49, 0x86, 0x06, 0x99,
    0x9c, 0x42, 0x50, 0xf4, 0x91, 0xef, 0x98, 0x7a, 0x33, 0x54, 0x0b, 0x43, 0xed, 0xcf, 0xac, 0x62,
    0xe4, 0xb3, 0x1c, 0xa9, 0xc9, 0x08, 0xe8, 0x95, 0x80, 0xdf, 0x94, 0xfa, 0x75, 0x8f, 0x3f, 0xa6,
    0x47, 0x07, 0xa7, 0xfc, 0xf3, 0x73, 0x17, 0xba, 0x83, 0x59, 0x3c, 0x19, 0xe6, 0x85, 0x4f, 0xa8,
    0x68, 0x6b, 0x81, 0xb2, 0x71, 0x64, 0xda, 0x8b, 0xf8, 0xeb, 0x0f, 0x4b, 0x70, 0x56, 0x9d, 0x35,
    0x1e, 0x24, 0x0e, 0x5e, 0x63, 0x58, 0xd1, 0xa2, 0x25, 0x22, 0x7c, 0x3b, 0x01, 0x21, 0x78, 0x87,
    0xd4, 0x00, 0x46, 0x57, 0x9f, 0xd3, 0x27, 0x52, 0x4c, 0x36, 0x02, 0xe7, 0xa0, 0xc4, 0xc8, 0x9e,
    0xea, 0xbf, 0x8a, 0xd2, 0x40, 0xc7, 0x38, 0xb5, 0xa3, 0xf7, 0xf2, 0xce, 0xf9, 0x61, 0x15, 0xa1,
    0xe0, 0xae, 0x5d, 0xa4, 0x9b, 0x34, 0x1a, 0x55, 0xad, 0x93, 0x32, 0x30, 0xf5, 0x8c, 0xb1, 0xe3,
    0x1d, 0xf6, 0xe2, 0x2e, 0x82, 0x66, 0xca, 0x60, 0xc0, 0x29, 0x23, 0xab, 0x0d, 0x53, 0x4e, 0x6f,
    0xd5, 0xdb, 0x37, 0x45, 0xde, 0xfd, 0x8e, 0x2f, 0x03, 0xff, 0x6a, 0x72, 0x6d, 0x6c, 0x5b, 0x51,
    0x8d, 0x1b, 0xaf, 0x92, 0xbb, 0xdd, 0xbc, 0x7f, 0x11, 0xd9, 0x5c, 0x41, 0x1f, 0x10, 0x5a, 0xd8,
    0x0a, 0xc1, 0x31, 0x88, 0xa5, 0xcd, 0x7b, 0xbd, 0x2d, 0x74, 0xd0, 0x12, 0xb8, 0xe5, 0xb4, 0xb0,
    0x89, 0x69, 0x97, 0x4a, 0x0c, 0x96, 0x77, 0x7e, 0x65, 0xb9, 0xf1, 0x09, 0xc5, 0x6e, 0xc6, 0x84,
    0x18, 0xf0, 0x7d, 0xec, 0x3a, 0xdc, 0x4d, 0x20, 0x79, 0xee, 0x5f, 0x3e, 0xd7, 0xcb, 0x39, 0x48,
];

/// Executes inst if it is in a cryptographic encoding class.
///
/// Returns None when inst belongs to no cryptographic class, so the Advanced SIMD decoder can
/// take it, and an error for an unallocated encoding inside one.
pub(super) fn execute(ctx: &mut ThreadContext, inst: u32) -> Option<Exec> {
    let two_register = bits(inst, 21, 17) == 0b10100 && bits(inst, 11, 10) == 0b10;
    match inst >> 24 {
        0x4e if two_register => Some(aes(ctx, inst)),
        0x5e if two_register => Some(sha_two_register(ctx, inst)),
        0x5e if !bit(inst, 21) && !bit(inst, 15) && bits(inst, 11, 10) == 0 => {
            Some(sha_three_register(ctx, inst))
        }
        0xce => Some(sha3_sha512_sm3_sm4(ctx, inst)),
        _ => None,
    }
}

fn words(v: u128) -> [u32; 4] {
    [
        v as u32,
        (v >> 32) as u32,
        (v >> 64) as u32,
        (v >> 96) as u32,
    ]
}

fn from_words(w: [u32; 4]) -> u128 {
    w[0] as u128 | (w[1] as u128) << 32 | (w[2] as u128) << 64 | (w[3] as u128) << 96
}

fn halves(v: u128) -> [u64; 2] {
    [v as u64, (v >> 64) as u64]
}

fn from_halves(h: [u64; 2]) -> u128 {
    h[0] as u128 | (h[1] as u128) << 64
}

fn aes(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bits(inst, 23, 22) != 0 {
        return Err(Undefined);
    }
    let d = bits(inst, 4, 0) as usize;
    let n = bits(inst, 9, 5) as usize;
    ctx.v[d] = match bits(inst, 16, 12) {
        0b00100 => aes_round(ctx.v[d] ^ ctx.v[n], &AES_SHIFT_ROWS, &AES_SBOX),
        0b00101 => aes_round(ctx.v[d] ^ ctx.v[n], &AES_INV_SHIFT_ROWS, &AES_INV_SBOX),
        0b00110 => mix_columns(ctx.v[n], [&MUL2, &MUL3, &MUL1, &MUL1]),
        0b00111 => mix_columns(ctx.v[n], [&MUL14, &MUL11, &MUL13, &MUL9]),
        _ => return Err(Undefined),
    };
    Ok(())
}

fn aes_round(state: u128, shift_rows: &[usize; 16], sbox: &[u8; 256]) -> u128 {
    let bytes = state.to_le_bytes();
    u128::from_le_bytes(shift_rows.map(|i| sbox[bytes[i] as usize]))
}

fn mix_columns(state: u128, tables: [&[u8; 256]; 4]) -> u128 {
    let s = state.to_le_bytes();
    let mut out = [0u8; 16];
    for column in 0..4 {
        for row in 0..4 {
            let at = |k: usize| s[column * 4 + (row + k) % 4] as usize;
            out[column * 4 + row] =
                tables[0][at(0)] ^ tables[1][at(1)] ^ tables[2][at(2)] ^ tables[3][at(3)];
        }
    }
    u128::from_le_bytes(out)
}

fn sha_two_register(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bits(inst, 23, 22) != 0 {
        return Err(Undefined);
    }
    let d = bits(inst, 4, 0) as usize;
    let n = ctx.v[bits(inst, 9, 5) as usize];
    let x = ctx.v[d];
    ctx.v[d] = match bits(inst, 16, 12) {
        0b00000 => (n as u32).rotate_left(30) as u128,
        0b00001 => {
            let t = words(x ^ (n >> 32)).map(|w| w.rotate_left(1));
            from_words([t[0], t[1], t[2], t[3] ^ t[0].rotate_left(1)])
        }
        0b00010 => {
            let x = words(x);
            let t = [x[1], x[2], x[3], n as u32];
            from_words([0, 1, 2, 3].map(|i| x[i].wrapping_add(small_sigma(t[i], 7, 18, 3))))
        }
        _ => return Err(Undefined),
    };
    Ok(())
}

fn small_sigma(x: u32, rotate_a: u32, rotate_b: u32, shift: u32) -> u32 {
    x.rotate_right(rotate_a) ^ x.rotate_right(rotate_b) ^ (x >> shift)
}

fn sha_three_register(ctx: &mut ThreadContext, inst: u32) -> Exec {
    if bits(inst, 23, 22) != 0 {
        return Err(Undefined);
    }
    let d = bits(inst, 4, 0) as usize;
    let n = ctx.v[bits(inst, 9, 5) as usize];
    let m = ctx.v[bits(inst, 20, 16) as usize];
    let x = ctx.v[d];
    ctx.v[d] = match bits(inst, 14, 12) {
        0b000 => sha1_hash_update(x, n as u32, m, |x, y, z| ((y ^ z) & x) ^ z),
        0b001 => sha1_hash_update(x, n as u32, m, |x, y, z| x ^ y ^ z),
        0b010 => sha1_hash_update(x, n as u32, m, |x, y, z| (x & y) | ((x | y) & z)),
        0b011 => from_halves([halves(x)[1], halves(n)[0]]) ^ x ^ m,
        0b100 => from_words(sha256_hash(words(x), words(n), words(m), true)),
        0b101 => from_words(sha256_hash(words(n), words(x), words(m), false)),
        0b110 => sha256_schedule1(x, n, m),
        _ => return Err(Undefined),
    };
    Ok(())
}

fn sha1_hash_update(x: u128, y: u32, w: u128, f: fn(u32, u32, u32) -> u32) -> u128 {
    let (mut x, mut y) = (words(x), y);
    for w in words(w) {
        y = y
            .wrapping_add(x[0].rotate_left(5))
            .wrapping_add(f(x[1], x[2], x[3]))
            .wrapping_add(w);
        let high = x[3];
        x = [y, x[0], x[1].rotate_left(30), x[2]];
        y = high;
    }
    from_words(x)
}

fn sha256_hash(mut x: [u32; 4], mut y: [u32; 4], w: [u32; 4], part1: bool) -> [u32; 4] {
    for w in w {
        let choose = ((y[1] ^ y[2]) & y[0]) ^ y[2];
        let majority = (x[0] & x[1]) | ((x[0] | x[1]) & x[2]);
        let big_sigma1 = y[0].rotate_right(6) ^ y[0].rotate_right(11) ^ y[0].rotate_right(25);
        let big_sigma0 = x[0].rotate_right(2) ^ x[0].rotate_right(13) ^ x[0].rotate_right(22);
        let t = y[3]
            .wrapping_add(big_sigma1)
            .wrapping_add(choose)
            .wrapping_add(w);
        let new_x = t.wrapping_add(x[3]);
        let new_y = t.wrapping_add(big_sigma0).wrapping_add(majority);
        x = [new_y, x[0], x[1], x[2]];
        y = [new_x, y[0], y[1], y[2]];
    }
    if part1 { x } else { y }
}

fn sha256_schedule1(x: u128, y: u128, z: u128) -> u128 {
    let (x, y, z) = (words(x), words(y), words(z));
    let t = [y[1], y[2], y[3], z[0]];
    let mut r = [0u32; 4];
    for i in 0..4 {
        let source = if i < 2 { z[i + 2] } else { r[i - 2] };
        r[i] = small_sigma(source, 17, 19, 10)
            .wrapping_add(x[i])
            .wrapping_add(t[i]);
    }
    from_words(r)
}

fn sha3_sha512_sm3_sm4(ctx: &mut ThreadContext, inst: u32) -> Exec {
    let d = bits(inst, 4, 0) as usize;
    let n = ctx.v[bits(inst, 9, 5) as usize];
    let m = ctx.v[bits(inst, 20, 16) as usize];
    let a = ctx.v[bits(inst, 14, 10) as usize];
    let x = ctx.v[d];
    ctx.v[d] = match bits(inst, 23, 21) {
        0b000 if !bit(inst, 15) => n ^ m ^ a,
        0b001 if !bit(inst, 15) => n ^ (m & !a),
        0b010 if !bit(inst, 15) => {
            let sum = words(n)[3].rotate_left(12).wrapping_add(words(m)[3]);
            from_words([0, 0, 0, sum.wrapping_add(words(a)[3]).rotate_left(7)])
        }
        0b010 if bits(inst, 15, 14) == 0b10 => {
            sm3_tt(x, n, m, bits(inst, 13, 12) as usize, bits(inst, 11, 10))
        }
        0b011 if bit(inst, 15) && bits(inst, 13, 12) == 0 => match bits(inst, 15, 10) {
            0b100000 => sha512_hash(x, n, m, true),
            0b100001 => sha512_hash(x, n, m, false),
            0b100010 => {
                let sigma1 = |v: u64| v.rotate_right(19) ^ v.rotate_right(61) ^ (v >> 6);
                let (w, y, z) = (halves(x), halves(n), halves(m));
                from_halves([0, 1].map(|i| w[i].wrapping_add(z[i]).wrapping_add(sigma1(y[i]))))
            }
            0b100011 => n ^ from_halves(halves(m).map(|lane| lane.rotate_left(1))),
            0b110000 => sm3_partw1(x, n, m),
            0b110001 => sm3_partw2(x, n, m),
            0b110010 => sm4_rounds(words(m), words(n), true),
            _ => return Err(Undefined),
        },
        0b100 => from_halves(halves(n ^ m).map(|lane| lane.rotate_right(bits(inst, 15, 10)))),
        0b110 if bits(inst, 20, 12) == 0b0_0000_1000 => match bits(inst, 11, 10) {
            0b00 => {
                let sigma0 = |v: u64| v.rotate_right(1) ^ v.rotate_right(8) ^ (v >> 7);
                let (w, x) = (halves(x), halves(n));
                from_halves([
                    w[0].wrapping_add(sigma0(w[1])),
                    w[1].wrapping_add(sigma0(x[0])),
                ])
            }
            0b01 => sm4_rounds(words(n), words(x), false),
            _ => return Err(Undefined),
        },
        _ => return Err(Undefined),
    };
    Ok(())
}

fn sha512_hash(w: u128, x: u128, y: u128, part1: bool) -> u128 {
    let (w, x, y) = (halves(w), halves(x), halves(y));
    let sigma = |v: u64| {
        if part1 {
            v.rotate_right(14) ^ v.rotate_right(18) ^ v.rotate_right(41)
        } else {
            v.rotate_right(28) ^ v.rotate_right(34) ^ v.rotate_right(39)
        }
    };
    let (high, low) = if part1 {
        let high = (y[1] & x[0]) ^ (!y[1] & x[1]);
        let high = high.wrapping_add(sigma(y[1])).wrapping_add(w[1]);
        let tmp = high.wrapping_add(y[0]);
        let low = (tmp & y[1]) ^ (!tmp & x[0]);
        (high, low.wrapping_add(sigma(tmp)).wrapping_add(w[0]))
    } else {
        let high = (x[0] & y[1]) ^ (x[0] & y[0]) ^ (y[1] & y[0]);
        let high = high.wrapping_add(sigma(y[0])).wrapping_add(w[1]);
        let low = (high & y[0]) ^ (high & y[1]) ^ (y[1] & y[0]);
        (high, low.wrapping_add(sigma(high)).wrapping_add(w[0]))
    };
    from_halves([low, high])
}

fn sm3_tt(d: u128, n: u128, m: u128, index: usize, opcode: u32) -> u128 {
    let (d, n, m) = (words(d), words(n), words(m));
    let w = m[index];
    let (rotated, result) = match opcode {
        0b00 | 0b01 => {
            let f = if opcode == 0 {
                d[1] ^ d[3] ^ d[2]
            } else {
                (d[3] & d[1]) | (d[3] & d[2]) | (d[1] & d[2])
            };
            let ss2 = n[3] ^ d[3].rotate_left(12);
            let tt1 = f.wrapping_add(d[0]).wrapping_add(ss2).wrapping_add(w);
            (d[2].rotate_left(9), tt1)
        }
        _ => {
            let f = if opcode == 0b10 {
                d[1] ^ d[3] ^ d[2]
            } else {
                (d[3] & d[2]) | (!d[3] & d[1])
            };
            let tt2 = f.wrapping_add(d[0]).wrapping_add(n[3]).wrapping_add(w);
            (
                d[2].rotate_left(19),
                tt2 ^ tt2.rotate_left(9) ^ tt2.rotate_left(17),
            )
        }
    };
    from_words([d[1], rotated, d[3], result])
}

fn sm3_p1(x: u32) -> u32 {
    x ^ x.rotate_left(15) ^ x.rotate_left(23)
}

fn sm3_partw1(d: u128, n: u128, m: u128) -> u128 {
    let (d, n, m) = (words(d), words(n), words(m));
    let mut t = [0u32; 4];
    for i in 0..4 {
        let rotated = if i < 3 { m[i + 1] } else { t[0] };
        t[i] = sm3_p1(d[i] ^ n[i] ^ rotated.rotate_left(15));
    }
    from_words(t)
}

fn sm3_partw2(d: u128, n: u128, m: u128) -> u128 {
    let (d, n, m) = (words(d), words(n), words(m));
    let t = [0, 1, 2, 3].map(|i| n[i] ^ m[i].rotate_left(7));
    let mut r = [0, 1, 2, 3].map(|i| d[i] ^ t[i]);
    r[3] ^= sm3_p1(t[0].rotate_left(15));
    from_words(r)
}

fn sm4_rounds(keys: [u32; 4], mut state: [u32; 4], key_schedule: bool) -> u128 {
    for key in keys {
        let t = u32::from_le_bytes(
            (state[1] ^ state[2] ^ state[3] ^ key)
                .to_le_bytes()
                .map(|b| SM4_SBOX[b as usize]),
        );
        let linear = if key_schedule {
            t ^ t.rotate_left(13) ^ t.rotate_left(23)
        } else {
            t ^ t.rotate_left(2) ^ t.rotate_left(10) ^ t.rotate_left(18) ^ t.rotate_left(24)
        };
        state = [state[1], state[2], state[3], state[0] ^ linear];
    }
    from_words(state)
}

#[cfg(test)]
mod tests {
    use super::super::tests::Harness;
    use super::execute;
    use crate::ThreadContext;

    const ITERATIONS: usize = 20_000;

    const RD_RN: u32 = 0x3ff;
    const RM: u32 = 0x1f << 16;
    const SIZE: u32 = 3 << 22;

    fn fuzz(seed: u64, base: u32, random: u32) {
        Harness::new(seed).fuzz(base, random, ITERATIONS);
    }

    #[test]
    fn aes_differential() {
        fuzz(21, 0x4e28_0800, RD_RN | SIZE | (0x1f << 12));
    }

    #[test]
    fn sha_two_register_differential() {
        fuzz(22, 0x5e28_0800, RD_RN | SIZE | (0x1f << 12));
    }

    #[test]
    fn sha_three_register_differential() {
        fuzz(23, 0x5e00_0000, RD_RN | RM | SIZE | (7 << 12));
    }

    #[test]
    fn sha3_four_register_differential() {
        fuzz(25, 0xce00_0000, RD_RN | RM | (3 << 21) | (0x1f << 10));
    }

    #[test]
    fn sha3_xar_differential() {
        fuzz(26, 0xce80_0000, RD_RN | RM | (0x3f << 10));
    }

    #[test]
    fn sha512_and_sm3_three_register_differential() {
        fuzz(27, 0xce60_8000, RD_RN | RM | (0xf << 10) | (1 << 14));
    }

    #[test]
    fn sm3_tt_differential() {
        fuzz(28, 0xce40_8000, RD_RN | RM | (0xf << 10) | (3 << 14));
    }

    #[test]
    fn sha512_and_sm4_two_register_differential() {
        fuzz(29, 0xcec0_8000, RD_RN | (3 << 10) | (0xf << 12));
    }

    #[test]
    fn whole_cryptographic_group_differential() {
        fuzz(30, 0xce00_0000, 0x01ff_ffff);
    }

    fn run(ctx: &mut ThreadContext, inst: u32) {
        assert!(
            matches!(execute(ctx, inst), Some(Ok(()))),
            "{inst:#010x} did not execute"
        );
    }

    fn hex_bytes(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).expect("hex"))
            .collect()
    }

    fn from_hex(s: &str) -> u128 {
        u128::from_le_bytes(hex_bytes(s).try_into().expect("16 bytes"))
    }

    fn lanes(w: [u32; 4]) -> u128 {
        super::from_words(w)
    }

    fn pair(low: u64, high: u64) -> u128 {
        super::from_halves([low, high])
    }

    fn ctx() -> ThreadContext {
        ThreadContext::default()
    }

    fn encode(base: u32, d: u32, n: u32, m: u32) -> u32 {
        base | (m << 16) | (n << 5) | d
    }

    fn encode4(base: u32, d: u32, n: u32, m: u32, a: u32) -> u32 {
        encode(base, d, n, m) | (a << 10)
    }

    #[test]
    fn aes_matches_fips_197_round_one() {
        let mut c = ctx();
        c.v[0] = from_hex("3243f6a8885a308d313198a2e0370734");
        c.v[1] = from_hex("2b7e151628aed2a6abf7158809cf4f3c");
        run(&mut c, encode(0x4e28_4800, 0, 1, 0));
        assert_eq!(c.v[0], from_hex("d4bf5d30e0b452aeb84111f11e2798e5"));
        run(&mut c, 0x4e28_6800);
        assert_eq!(c.v[0], from_hex("046681e5e0cb199a48f8d37a2806264c"));
        run(&mut c, 0x4e28_7800);
        assert_eq!(c.v[0], from_hex("d4bf5d30e0b452aeb84111f11e2798e5"));
        c.v[1] = 0;
        run(&mut c, encode(0x4e28_5800, 0, 1, 0));
        assert_eq!(c.v[0], from_hex("193de3bea0f4e22b9ac68d2ae9f84808"));
    }

    #[test]
    fn aes_sbox_is_the_fips_table() {
        assert_eq!(super::AES_SBOX[0x00], 0x63);
        assert_eq!(super::AES_SBOX[0x53], 0xed);
        assert_eq!(super::AES_SBOX[0xff], 0x16);
        assert_eq!(super::AES_INV_SBOX[0x63], 0x00);
    }

    fn sha1_abc_digest() -> [u32; 5] {
        let mut block = [0u32; 16];
        block[0] = 0x6162_6380;
        block[15] = 24;
        let mut schedule = [0u128; 20];
        for g in 0..4 {
            schedule[g] = lanes([
                block[4 * g],
                block[4 * g + 1],
                block[4 * g + 2],
                block[4 * g + 3],
            ]);
        }
        let mut c = ctx();
        for g in 4..20 {
            c.v[8] = schedule[g - 4];
            c.v[9] = schedule[g - 3];
            c.v[10] = schedule[g - 2];
            c.v[11] = schedule[g - 1];
            run(&mut c, encode(0x5e00_3000, 8, 9, 10));
            run(&mut c, 0x5e28_1800 | (11 << 5) | 8);
            schedule[g] = c.v[8];
        }
        let init = [
            0x6745_2301,
            0xefcd_ab89,
            0x98ba_dcfe,
            0x1032_5476,
            0xc3d2_e1f0,
        ];
        let mut abcd = lanes([init[0], init[1], init[2], init[3]]);
        let mut e = init[4];
        for (g, w) in schedule.iter().enumerate() {
            let k = [0x5a82_7999u32, 0x6ed9_eba1, 0x8f1b_bcdc, 0xca62_c1d6][g / 5];
            let wk = super::words(*w).map(|w| w.wrapping_add(k));
            c.v[0] = abcd;
            c.v[1] = e as u128;
            c.v[2] = lanes(wk);
            run(&mut c, 0x5e28_0800 | 3);
            let op = [0x5e00_0000, 0x5e00_1000, 0x5e00_2000, 0x5e00_1000][g / 5];
            run(&mut c, encode(op, 0, 1, 2));
            abcd = c.v[0];
            e = c.v[3] as u32;
        }
        let a = super::words(abcd);
        [
            a[0].wrapping_add(init[0]),
            a[1].wrapping_add(init[1]),
            a[2].wrapping_add(init[2]),
            a[3].wrapping_add(init[3]),
            e.wrapping_add(init[4]),
        ]
    }

    #[test]
    fn sha1_of_abc() {
        assert_eq!(
            sha1_abc_digest(),
            [
                0xa999_3e36,
                0x4706_816a,
                0xba3e_2571,
                0x7850_c26c,
                0x9cd0_d89d
            ]
        );
    }

    #[test]
    fn sha256_of_abc() {
        let init = [
            0x6a09_e667u32,
            0xbb67_ae85,
            0x3c6e_f372,
            0xa54f_f53a,
            0x510e_527f,
            0x9b05_688c,
            0x1f83_d9ab,
            0x5be0_cd19,
        ];
        let mut block = [0u32; 16];
        block[0] = 0x6162_6380;
        block[15] = 24;
        let mut schedule = [0u128; 16];
        for g in 0..4 {
            schedule[g] = lanes([
                block[4 * g],
                block[4 * g + 1],
                block[4 * g + 2],
                block[4 * g + 3],
            ]);
        }
        let mut c = ctx();
        for g in 4..16 {
            c.v[8] = schedule[g - 4];
            c.v[9] = schedule[g - 3];
            run(&mut c, 0x5e28_2800 | (9 << 5) | 8);
            c.v[10] = schedule[g - 2];
            c.v[11] = schedule[g - 1];
            run(&mut c, encode(0x5e00_6000, 8, 10, 11));
            schedule[g] = c.v[8];
        }
        let mut abcd = lanes([init[0], init[1], init[2], init[3]]);
        let mut efgh = lanes([init[4], init[5], init[6], init[7]]);
        for g in 0..16 {
            let k = [0, 1, 2, 3].map(|i| SHA256_K[4 * g + i]);
            let wk = super::words(schedule[g]);
            c.v[0] = abcd;
            c.v[1] = efgh;
            c.v[2] = abcd;
            c.v[3] = lanes([0, 1, 2, 3].map(|i| wk[i].wrapping_add(k[i])));
            run(&mut c, encode(0x5e00_4000, 0, 1, 3));
            run(&mut c, encode(0x5e00_5000, 1, 2, 3));
            abcd = c.v[0];
            efgh = c.v[1];
        }
        let (a, e) = (super::words(abcd), super::words(efgh));
        let digest = [a[0], a[1], a[2], a[3], e[0], e[1], e[2], e[3]];
        let sum: Vec<u32> = (0..8).map(|i| digest[i].wrapping_add(init[i])).collect();
        assert_eq!(
            sum,
            [
                0xba78_16bf,
                0x8f01_cfea,
                0x4141_40de,
                0x5dae_2223,
                0xb003_61a3,
                0x9617_7a9c,
                0xb410_ff61,
                0xf200_15ad
            ]
        );
    }

    #[test]
    fn sha512_of_abc() {
        let init = [
            0x6a09_e667_f3bc_c908u64,
            0xbb67_ae85_84ca_a73b,
            0x3c6e_f372_fe94_f82b,
            0xa54f_f53a_5f1d_36f1,
            0x510e_527f_ade6_82d1,
            0x9b05_688c_2b3e_6c1f,
            0x1f83_d9ab_fb41_bd6b,
            0x5be0_cd19_137e_2179,
        ];
        let mut block = [0u64; 16];
        block[0] = 0x6162_6380_0000_0000;
        block[15] = 24;
        let mut pairs = [0u128; 40];
        for k in 0..8 {
            pairs[k] = pair(block[2 * k], block[2 * k + 1]);
        }
        let mut c = ctx();
        for k in 8..40 {
            c.v[8] = pairs[k - 8];
            c.v[9] = pairs[k - 7];
            run(&mut c, 0xcec0_8000 | (9 << 5) | 8);
            c.v[10] = pairs[k - 1];
            c.v[11] = pair((pairs[k - 4] >> 64) as u64, pairs[k - 3] as u64);
            run(&mut c, encode(0xce60_8800, 8, 10, 11));
            pairs[k] = c.v[8];
        }
        let w: Vec<u64> = pairs.iter().flat_map(|p| super::halves(*p)).collect();
        let (mut a, mut cd, mut e, mut g) = (
            pair(init[0], init[1]),
            pair(init[2], init[3]),
            pair(init[4], init[5]),
            pair(init[6], init[7]),
        );
        let lane_add = |x: u128, y: u128| {
            let (x, y) = (super::halves(x), super::halves(y));
            pair(x[0].wrapping_add(y[0]), x[1].wrapping_add(y[1]))
        };
        for t in (0..80).step_by(2) {
            let wk0 = w[t].wrapping_add(SHA512_K[t]);
            let wk1 = w[t + 1].wrapping_add(SHA512_K[t + 1]);
            c.v[0] = lane_add(g, pair(wk1, wk0));
            c.v[1] = pair((e >> 64) as u64, g as u64);
            c.v[2] = pair((cd >> 64) as u64, e as u64);
            run(&mut c, encode(0xce60_8000, 0, 1, 2));
            let new_e = lane_add(cd, c.v[0]);
            c.v[3] = cd;
            c.v[4] = a;
            run(&mut c, encode(0xce60_8400, 0, 3, 4));
            let new_a = c.v[0];
            g = e;
            e = new_e;
            cd = a;
            a = new_a;
        }
        let state = [a, cd, e, g].map(super::halves);
        let sum: Vec<u64> = (0..8)
            .map(|i| state[i / 2][i % 2].wrapping_add(init[i]))
            .collect();
        assert_eq!(
            sum,
            [
                0xddaf_35a1_9361_7aba,
                0xcc41_7349_ae20_4131,
                0x12e6_fa4e_89a9_7ea2,
                0x0a9e_eee6_4b55_d39a,
                0x2192_992a_274f_c1a8,
                0x36ba_3c23_a3fe_ebbd,
                0x454d_4423_643c_e80e,
                0x2a9a_c94f_a54c_a49f
            ]
        );
    }

    fn keccak_round_constant(round: usize) -> u64 {
        let lfsr = |t: usize| {
            let mut r = 1u16;
            for _ in 0..t % 255 {
                r <<= 1;
                if r & 0x100 != 0 {
                    r ^= 0x171;
                }
            }
            r & 1 == 1
        };
        (0..7)
            .filter(|&j| lfsr(j + 7 * round))
            .fold(0, |rc, j| rc | 1 << ((1 << j) - 1))
    }

    fn keccak_f(a: &mut [u64; 25]) {
        const ROTATIONS: [u32; 24] = [
            1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20,
            44,
        ];
        const LANES: [usize; 24] = [
            10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
        ];
        let mut c = ctx();
        for round in 0..24 {
            let mut parity = [0u64; 5];
            for x in 0..5 {
                c.v[1] = a[x] as u128;
                c.v[2] = a[x + 5] as u128;
                c.v[3] = a[x + 10] as u128;
                run(&mut c, encode4(0xce00_0000, 0, 1, 2, 3));
                c.v[1] = c.v[0];
                c.v[2] = a[x + 15] as u128;
                c.v[3] = a[x + 20] as u128;
                run(&mut c, encode4(0xce00_0000, 0, 1, 2, 3));
                parity[x] = c.v[0] as u64;
            }
            for x in 0..5 {
                c.v[1] = parity[(x + 4) % 5] as u128;
                c.v[2] = parity[(x + 1) % 5] as u128;
                run(&mut c, encode(0xce60_8c00, 0, 1, 2));
                let d = c.v[0] as u64;
                for y in 0..5 {
                    a[x + 5 * y] ^= d;
                }
            }
            let mut carried = a[1];
            for i in 0..24 {
                let lane = a[LANES[i]];
                c.v[1] = carried as u128;
                c.v[2] = 0;
                run(
                    &mut c,
                    encode(0xce80_0000 | (((64 - ROTATIONS[i]) % 64) << 10), 0, 1, 2),
                );
                a[LANES[i]] = c.v[0] as u64;
                carried = lane;
            }
            for y in 0..5 {
                let row = [0, 1, 2, 3, 4].map(|x| a[x + 5 * y]);
                for x in 0..5 {
                    c.v[1] = row[x] as u128;
                    c.v[2] = row[(x + 2) % 5] as u128;
                    c.v[3] = row[(x + 1) % 5] as u128;
                    run(&mut c, encode4(0xce20_0000, 0, 1, 2, 3));
                    a[x + 5 * y] = c.v[0] as u64;
                }
            }
            a[0] ^= keccak_round_constant(round);
        }
    }

    #[test]
    fn sha3_256_of_empty_message() {
        let mut state = [0u64; 25];
        state[0] = 0x06;
        state[16] = 0x8000_0000_0000_0000;
        keccak_f(&mut state);
        let digest: Vec<u8> = state[..4].iter().flat_map(|l| l.to_le_bytes()).collect();
        assert_eq!(
            digest,
            hex_bytes("a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a")
        );
    }

    #[test]
    fn keccak_f_of_zero_state() {
        let mut state = [0u64; 25];
        keccak_f(&mut state);
        assert_eq!(state[0], 0xf125_8f79_40e1_dde7);
        assert_eq!(state[1], 0x84d5_ccf9_33c0_478a);
    }

    fn be_words(bytes: &[u8]) -> [u32; 4] {
        [0, 1, 2, 3].map(|i| u32::from_be_bytes(bytes[4 * i..4 * i + 4].try_into().expect("word")))
    }

    #[test]
    fn sm4_encrypts_the_standard_example() {
        let key = be_words(&hex_bytes("0123456789abcdeffedcba9876543210"));
        let fk = [0xa3b1_bac6u32, 0x56aa_3350, 0x677d_9197, 0xb270_22dc];
        let mut c = ctx();
        let mut window = lanes([0, 1, 2, 3].map(|i| key[i] ^ fk[i]));
        let mut round_keys = [0u128; 8];
        for (g, round_key) in round_keys.iter_mut().enumerate() {
            let ck = [0, 1, 2, 3].map(|j| {
                u32::from_be_bytes([0, 1, 2, 3].map(|b| ((4 * (4 * g + j) + b) * 7) as u8))
            });
            c.v[1] = window;
            c.v[2] = lanes(ck);
            run(&mut c, encode(0xce60_c800, 0, 1, 2));
            window = c.v[0];
            *round_key = window;
        }
        assert_eq!(super::words(round_keys[0])[0], 0xf121_86f9);
        assert_eq!(super::words(round_keys[7])[3], 0x9124_a012);
        c.v[0] = lanes(key);
        for round_key in round_keys {
            c.v[1] = round_key;
            run(&mut c, 0xcec0_8400 | (1 << 5));
        }
        let x = super::words(c.v[0]);
        let ciphertext: Vec<u8> = [x[3], x[2], x[1], x[0]]
            .iter()
            .flat_map(|w| w.to_be_bytes())
            .collect();
        assert_eq!(ciphertext, hex_bytes("681edf34d206965e86b3e94f536e4246"));
    }

    #[test]
    fn sm3_of_abc() {
        let init = [
            0x7380_166fu32,
            0x4914_b2b9,
            0x1724_42d7,
            0xda8a_0600,
            0xa96f_30bc,
            0x1631_38aa,
            0xe38d_ee4d,
            0xb0fb_0e4e,
        ];
        let mut w = [0u32; 68];
        w[0] = 0x6162_6380;
        w[15] = 24;
        let window = |w: &[u32], at: usize| lanes([w[at], w[at + 1], w[at + 2], w[at + 3]]);
        let mut c = ctx();
        for j in (16..68).step_by(4) {
            c.v[0] = window(&w, j - 16);
            c.v[1] = window(&w, j - 9);
            c.v[2] = window(&w, j - 4);
            run(&mut c, encode(0xce60_c000, 0, 1, 2));
            c.v[1] = window(&w, j - 6);
            c.v[2] = window(&w, j - 13);
            run(&mut c, encode(0xce60_c400, 0, 1, 2));
            w[j..j + 4].copy_from_slice(&super::words(c.v[0]));
        }
        let w_prime: Vec<u32> = (0..64).map(|j| w[j] ^ w[j + 4]).collect();
        c.v[0] = lanes([init[3], init[2], init[1], init[0]]);
        c.v[1] = lanes([init[7], init[6], init[5], init[4]]);
        for j in 0..64usize {
            let t = if j < 16 { 0x79cc_4519u32 } else { 0x7a87_9d8a };
            c.v[3] = lanes([0, 0, 0, t.rotate_left(j as u32 % 32)]);
            run(&mut c, encode4(0xce40_0000, 2, 0, 1, 3));
            c.v[4] = window(&w_prime, j - j % 4);
            c.v[5] = window(&w, j - j % 4);
            let (tt1, tt2) = if j < 16 {
                (0, 2 << 10)
            } else {
                (1 << 10, 3 << 10)
            };
            let index = (j as u32 % 4) << 12;
            run(&mut c, encode(0xce40_8000 | index | tt1, 0, 2, 4));
            run(&mut c, encode(0xce40_8000 | index | tt2, 1, 2, 5));
        }
        let (abcd, efgh) = (super::words(c.v[0]), super::words(c.v[1]));
        let state = [
            abcd[3], abcd[2], abcd[1], abcd[0], efgh[3], efgh[2], efgh[1], efgh[0],
        ];
        let sum: Vec<u32> = (0..8).map(|i| state[i] ^ init[i]).collect();
        assert_eq!(
            sum,
            [
                0x66c7_f0f4,
                0x62ee_edd9,
                0xd1f2_d46b,
                0xdc10_e4e2,
                0x4167_c487,
                0x5cf2_f7a2,
                0x297d_a02b,
                0x8f4b_a8e0
            ]
        );
    }

    #[test]
    fn unallocated_encodings_are_undefined() {
        let mut c = ctx();
        for inst in [
            0x4e68_4800u32,
            0x4e28_0800,
            0x5e00_7000,
            0x5e40_0000,
            0x5e28_3800,
            0xce60_cc00,
            0xcec0_8800,
            0xcee0_0000,
            0xce60_a000,
            0xce60_9000,
        ] {
            assert!(
                matches!(execute(&mut c, inst), Some(Err(_))),
                "{inst:#010x} should be undefined"
            );
        }
        assert!(execute(&mut c, 0x5e08_0400).is_none());
        assert!(execute(&mut c, 0x4e20_0400).is_none());
        assert!(execute(&mut c, 0xcf00_0000).is_none());
    }

    const SHA256_K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    const SHA512_K: [u64; 80] = [
        0x428a2f98d728ae22,
        0x7137449123ef65cd,
        0xb5c0fbcfec4d3b2f,
        0xe9b5dba58189dbbc,
        0x3956c25bf348b538,
        0x59f111f1b605d019,
        0x923f82a4af194f9b,
        0xab1c5ed5da6d8118,
        0xd807aa98a3030242,
        0x12835b0145706fbe,
        0x243185be4ee4b28c,
        0x550c7dc3d5ffb4e2,
        0x72be5d74f27b896f,
        0x80deb1fe3b1696b1,
        0x9bdc06a725c71235,
        0xc19bf174cf692694,
        0xe49b69c19ef14ad2,
        0xefbe4786384f25e3,
        0x0fc19dc68b8cd5b5,
        0x240ca1cc77ac9c65,
        0x2de92c6f592b0275,
        0x4a7484aa6ea6e483,
        0x5cb0a9dcbd41fbd4,
        0x76f988da831153b5,
        0x983e5152ee66dfab,
        0xa831c66d2db43210,
        0xb00327c898fb213f,
        0xbf597fc7beef0ee4,
        0xc6e00bf33da88fc2,
        0xd5a79147930aa725,
        0x06ca6351e003826f,
        0x142929670a0e6e70,
        0x27b70a8546d22ffc,
        0x2e1b21385c26c926,
        0x4d2c6dfc5ac42aed,
        0x53380d139d95b3df,
        0x650a73548baf63de,
        0x766a0abb3c77b2a8,
        0x81c2c92e47edaee6,
        0x92722c851482353b,
        0xa2bfe8a14cf10364,
        0xa81a664bbc423001,
        0xc24b8b70d0f89791,
        0xc76c51a30654be30,
        0xd192e819d6ef5218,
        0xd69906245565a910,
        0xf40e35855771202a,
        0x106aa07032bbd1b8,
        0x19a4c116b8d2d0c8,
        0x1e376c085141ab53,
        0x2748774cdf8eeb99,
        0x34b0bcb5e19b48a8,
        0x391c0cb3c5c95a63,
        0x4ed8aa4ae3418acb,
        0x5b9cca4f7763e373,
        0x682e6ff3d6b2b8a3,
        0x748f82ee5defb2fc,
        0x78a5636f43172f60,
        0x84c87814a1f0ab72,
        0x8cc702081a6439ec,
        0x90befffa23631e28,
        0xa4506cebde82bde9,
        0xbef9a3f7b2c67915,
        0xc67178f2e372532b,
        0xca273eceea26619c,
        0xd186b8c721c0c207,
        0xeada7dd6cde0eb1e,
        0xf57d4f7fee6ed178,
        0x06f067aa72176fba,
        0x0a637dc5a2c898a6,
        0x113f9804bef90dae,
        0x1b710b35131c471b,
        0x28db77f523047d84,
        0x32caab7b40c72493,
        0x3c9ebe0a15c9bebc,
        0x431d67c49c100d4c,
        0x4cc5d4becb3e42b6,
        0x597f299cfc657e2a,
        0x5fcb6fab3ad6faec,
        0x6c44198c4a475817,
    ];
}
