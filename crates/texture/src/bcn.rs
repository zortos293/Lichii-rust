//! BCn block decoders (BC1-BC5, BC6H, BC7) and fast BC1/BC3 encoders.
//!
//! Block functions take one block and return its 16 texels row-major. Image functions take a
//! tightly packed, already deswizzled block array and write tightly packed texels, cropping the
//! blocks that overhang the right and bottom edges.
//!
//! BC1-BC3 endpoints expand and interpolate with exact rounding of the D3D float definitions;
//! BC4/BC5, BC6H and BC7 follow the D3D integer definitions. Results match the `bcdec` decoder
//! bit for bit, which the differential tests check.

mod encode;

pub use encode::{encode_bc1, encode_bc1_block, encode_bc3, encode_bc3_block};

use crate::block::image_decoder;

const fn div_round(n: u32, d: u32) -> u32 {
    (2 * n + d) / (2 * d)
}

/// The four BC1 palette entries for endpoints `c0` and `c1`; `four_color` selects the
/// interpolated fourth entry instead of transparent black.
pub(crate) fn bc1_palette(c0: u16, c1: u16, four_color: bool) -> [[u8; 4]; 4] {
    let (c0, c1) = (u32::from(c0), u32::from(c1));
    let (r0, g0, b0) = (c0 >> 11, (c0 >> 5) & 63, c0 & 31);
    let (r1, g1, b1) = (c1 >> 11, (c1 >> 5) & 63, c1 & 31);
    let rgb = |r: u32, g: u32, b: u32| [r as u8, g as u8, b as u8, 255];
    let mut palette = [
        rgb(
            div_round(r0 * 255, 31),
            div_round(g0 * 255, 63),
            div_round(b0 * 255, 31),
        ),
        rgb(
            div_round(r1 * 255, 31),
            div_round(g1 * 255, 63),
            div_round(b1 * 255, 31),
        ),
        [0; 4],
        [0; 4],
    ];
    if four_color {
        palette[2] = rgb(
            div_round((2 * r0 + r1) * 255, 93),
            div_round((2 * g0 + g1) * 255, 189),
            div_round((2 * b0 + b1) * 255, 93),
        );
        palette[3] = rgb(
            div_round((r0 + 2 * r1) * 255, 93),
            div_round((g0 + 2 * g1) * 255, 189),
            div_round((b0 + 2 * b1) * 255, 93),
        );
    } else {
        palette[2] = rgb(
            div_round((r0 + r1) * 255, 62),
            div_round((g0 + g1) * 255, 126),
            div_round((b0 + b1) * 255, 62),
        );
    }
    palette
}

fn color_block(block: &[u8; 8], always_four_color: bool) -> [[u8; 4]; 16] {
    let c0 = u16::from_le_bytes([block[0], block[1]]);
    let c1 = u16::from_le_bytes([block[2], block[3]]);
    let palette = bc1_palette(c0, c1, always_four_color || c0 > c1);
    let indices = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    std::array::from_fn(|i| palette[(indices >> (2 * i)) as usize & 3])
}

/// The eight BC3 alpha palette entries for endpoints `a0` and `a1`.
pub(crate) fn bc3_alpha_palette(a0: u8, a1: u8) -> [u8; 8] {
    let (a0, a1) = (u32::from(a0), u32::from(a1));
    let mut palette = [a0 as u8, a1 as u8, 0, 0, 0, 0, 0, 255];
    if a0 > a1 {
        for k in 1..7 {
            palette[k + 1] = (((7 - k as u32) * a0 + k as u32 * a1 + 1) / 7) as u8;
        }
    } else {
        for k in 1..5 {
            palette[k + 1] = (((5 - k as u32) * a0 + k as u32 * a1 + 1) / 5) as u8;
        }
    }
    palette
}

fn split_halves(block: &[u8; 16]) -> (&[u8; 8], &[u8; 8]) {
    let (first, second) = block.split_at(8);
    (first.try_into().unwrap(), second.try_into().unwrap())
}

/// Decodes a BC1 block to RGBA8. In three-colour mode index 3 is transparent black.
pub fn decode_bc1_block(block: &[u8; 8]) -> [[u8; 4]; 16] {
    color_block(block, false)
}

/// Decodes a BC2 block (explicit 4-bit alpha) to RGBA8.
pub fn decode_bc2_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let (alpha, color) = split_halves(block);
    let mut texels = color_block(color, true);
    let alpha = u64::from_le_bytes(*alpha);
    for (i, texel) in texels.iter_mut().enumerate() {
        texel[3] = ((alpha >> (4 * i)) & 15) as u8 * 17;
    }
    texels
}

/// Decodes a BC3 block (interpolated alpha) to RGBA8.
pub fn decode_bc3_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let (alpha, color) = split_halves(block);
    let mut texels = color_block(color, true);
    let palette = bc3_alpha_palette(alpha[0], alpha[1]);
    let indices = u64::from_le_bytes(*alpha) >> 16;
    for (i, texel) in texels.iter_mut().enumerate() {
        texel[3] = palette[(indices >> (3 * i)) as usize & 7];
    }
    texels
}

fn bc4_values<const SIGNED: bool>(block: &[u8; 8]) -> [i32; 16] {
    const WEIGHTS6: [i32; 6] = [9363, 18724, 28086, 37450, 46812, 56173];
    const WEIGHTS4: [i32; 4] = [13107, 26215, 39321, 52429];
    let (a0, a1, lo, hi) = if SIGNED {
        let a0 = i32::from(block[0] as i8).max(-127);
        let a1 = i32::from(block[1] as i8).max(-127);
        (a0, a1, -127, 127)
    } else {
        (i32::from(block[0]), i32::from(block[1]), 0, 255)
    };
    let mut palette = [a0, a1, 0, 0, 0, 0, lo, hi];
    if a0 > a1 {
        for k in 1..7 {
            palette[k + 1] = (WEIGHTS6[6 - k] * a0 + WEIGHTS6[k - 1] * a1 + 32768) >> 16;
        }
    } else {
        for k in 1..5 {
            palette[k + 1] = (WEIGHTS4[4 - k] * a0 + WEIGHTS4[k - 1] * a1 + 32768) >> 16;
        }
    }
    let indices = u64::from_le_bytes(*block) >> 16;
    std::array::from_fn(|i| palette[(indices >> (3 * i)) as usize & 7])
}

/// Decodes a BC4 unsigned block to R8.
pub fn decode_bc4_unorm_block(block: &[u8; 8]) -> [u8; 16] {
    bc4_values::<false>(block).map(|v| v as u8)
}

/// Decodes a BC4 signed block to R8 snorm; the endpoint -128 reads as -127.
pub fn decode_bc4_snorm_block(block: &[u8; 8]) -> [i8; 16] {
    bc4_values::<true>(block).map(|v| v as i8)
}

/// Decodes a BC5 unsigned block to RG8.
pub fn decode_bc5_unorm_block(block: &[u8; 16]) -> [[u8; 2]; 16] {
    let (r, g) = split_halves(block);
    let (r, g) = (decode_bc4_unorm_block(r), decode_bc4_unorm_block(g));
    std::array::from_fn(|i| [r[i], g[i]])
}

/// Decodes a BC5 signed block to RG8 snorm.
pub fn decode_bc5_snorm_block(block: &[u8; 16]) -> [[i8; 2]; 16] {
    let (r, g) = split_halves(block);
    let (r, g) = (decode_bc4_snorm_block(r), decode_bc4_snorm_block(g));
    std::array::from_fn(|i| [r[i], g[i]])
}

const fn expand_partitions2(compact: [u16; 64]) -> [u32; 64] {
    let mut out = [0; 64];
    let mut i = 0;
    while i < 64 {
        let mut t = 0;
        while t < 16 {
            out[i] |= ((compact[i] as u32 >> t) & 1) << (2 * t);
            t += 1;
        }
        i += 1;
    }
    out
}

const fn anchor_masks(second: [u8; 64], third: Option<[u8; 64]>) -> [u16; 64] {
    let mut out = [0; 64];
    let mut i = 0;
    while i < 64 {
        out[i] = 1 | 1 << second[i];
        if let Some(third) = third {
            out[i] |= 1 << third[i];
        }
        i += 1;
    }
    out
}

/// Two-subset partitions, two bits per texel (texel `t` at bit `2 * t`). BC6H uses the first 32.
const PARTITIONS2: [u32; 64] = expand_partitions2([
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80, //
    0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000, //
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce, //
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c, //
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a, //
    0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660, //
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c, //
    0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22, //
]);

/// Three-subset partitions, two bits per texel.
const PARTITIONS3: [u32; 64] = [
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
    0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
    0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
    0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
    0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
    0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
    0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
    0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
];

/// Texels that store one index bit fewer (each subset's anchor), for two-subset partitions.
const ANCHORS2: [u16; 64] = anchor_masks(
    [
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, //
        15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, //
        15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, //
        6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15, //
    ],
    None,
);

/// Anchor texels for three-subset partitions.
const ANCHORS3: [u16; 64] = anchor_masks(
    [
        3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, //
        3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, //
        8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, //
        3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3, //
    ],
    Some([
        15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, //
        15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, //
        15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, //
        15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8, //
    ]),
);

const WEIGHTS2: [u32; 4] = [0, 21, 43, 64];
const WEIGHTS3: [u32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS4: [u32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

fn weights(index_bits: u32) -> &'static [u32] {
    match index_bits {
        2 => &WEIGHTS2,
        3 => &WEIGHTS3,
        _ => &WEIGHTS4,
    }
}

struct Bits(u128);

impl Bits {
    fn take(&mut self, count: u32) -> u32 {
        let value = self.0 as u32 & ((1 << count) - 1);
        self.0 >>= count;
        value
    }

    /// Reads one index per texel, one bit fewer for the texels in `anchors`. Every index set
    /// fits in 64 bits, so the reads shift a u64.
    fn take_indices(&mut self, bits: u32, anchors: u16) -> [u8; 16] {
        let mut word = self.0 as u64;
        let mut used = 0;
        let indices = std::array::from_fn(|t| {
            let count = bits - u32::from((anchors >> t) & 1);
            let index = word as u8 & ((1 << count) - 1);
            word >>= count;
            used += count;
            index
        });
        self.0 >>= used;
        indices
    }
}

const HALF_ONE: u16 = 0x3c00;

/// Field order: red, green, blue endpoints w, x, y, z (index `channel * 4 + endpoint`), then the
/// partition. Each entry is (field, first bit, bit count), read in stream order after the mode.
#[rustfmt::skip]
const BC6H_FIELDS: [&[(u8, u8, u8)]; 14] = [
    &[
        (6, 4, 1), (10, 4, 1), (11, 4, 1), (0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 5),
        (7, 4, 1), (6, 0, 4), (5, 0, 5), (11, 0, 1), (7, 0, 4), (9, 0, 5), (11, 1, 1),
        (10, 0, 4), (2, 0, 5), (11, 2, 1), (3, 0, 5), (11, 3, 1), (12, 0, 5),
    ],
    &[
        (6, 5, 1), (7, 4, 1), (7, 5, 1), (0, 0, 7), (11, 0, 1), (11, 1, 1), (10, 4, 1),
        (4, 0, 7), (10, 5, 1), (11, 2, 1), (6, 4, 1), (8, 0, 7), (11, 3, 1), (11, 5, 1),
        (11, 4, 1), (1, 0, 6), (6, 0, 4), (5, 0, 6), (7, 0, 4), (9, 0, 6), (10, 0, 4),
        (2, 0, 6), (3, 0, 6), (12, 0, 5),
    ],
    &[
        (0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 5), (0, 10, 1), (6, 0, 4), (5, 0, 4),
        (4, 10, 1), (11, 0, 1), (7, 0, 4), (9, 0, 4), (8, 10, 1), (11, 1, 1), (10, 0, 4),
        (2, 0, 5), (11, 2, 1), (3, 0, 5), (11, 3, 1), (12, 0, 5),
    ],
    &[
        (0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 4), (0, 10, 1), (7, 4, 1), (6, 0, 4),
        (5, 0, 5), (4, 10, 1), (7, 0, 4), (9, 0, 4), (8, 10, 1), (11, 1, 1), (10, 0, 4),
        (2, 0, 4), (11, 0, 1), (11, 2, 1), (3, 0, 4), (6, 4, 1), (11, 3, 1), (12, 0, 5),
    ],
    &[
        (0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 4), (0, 10, 1), (10, 4, 1), (6, 0, 4),
        (5, 0, 4), (4, 10, 1), (11, 0, 1), (7, 0, 4), (9, 0, 5), (8, 10, 1), (10, 0, 4),
        (2, 0, 4), (11, 1, 1), (11, 2, 1), (3, 0, 4), (11, 4, 1), (11, 3, 1), (12, 0, 5),
    ],
    &[
        (0, 0, 9), (10, 4, 1), (4, 0, 9), (6, 4, 1), (8, 0, 9), (11, 4, 1), (1, 0, 5),
        (7, 4, 1), (6, 0, 4), (5, 0, 5), (11, 0, 1), (7, 0, 4), (9, 0, 5), (11, 1, 1),
        (10, 0, 4), (2, 0, 5), (11, 2, 1), (3, 0, 5), (11, 3, 1), (12, 0, 5),
    ],
    &[
        (0, 0, 8), (7, 4, 1), (10, 4, 1), (4, 0, 8), (11, 2, 1), (6, 4, 1), (8, 0, 8),
        (11, 3, 1), (11, 4, 1), (1, 0, 6), (6, 0, 4), (5, 0, 5), (11, 0, 1), (7, 0, 4),
        (9, 0, 5), (11, 1, 1), (10, 0, 4), (2, 0, 6), (3, 0, 6), (12, 0, 5),
    ],
    &[
        (0, 0, 8), (11, 0, 1), (10, 4, 1), (4, 0, 8), (6, 5, 1), (6, 4, 1), (8, 0, 8),
        (7, 5, 1), (11, 4, 1), (1, 0, 5), (7, 4, 1), (6, 0, 4), (5, 0, 6), (7, 0, 4),
        (9, 0, 5), (11, 1, 1), (10, 0, 4), (2, 0, 5), (11, 2, 1), (3, 0, 5), (11, 3, 1),
        (12, 0, 5),
    ],
    &[
        (0, 0, 8), (11, 1, 1), (10, 4, 1), (4, 0, 8), (10, 5, 1), (6, 4, 1), (8, 0, 8),
        (11, 5, 1), (11, 4, 1), (1, 0, 5), (7, 4, 1), (6, 0, 4), (5, 0, 5), (11, 0, 1),
        (7, 0, 4), (9, 0, 6), (10, 0, 4), (2, 0, 5), (11, 2, 1), (3, 0, 5), (11, 3, 1),
        (12, 0, 5),
    ],
    &[
        (0, 0, 6), (7, 4, 1), (11, 0, 1), (11, 1, 1), (10, 4, 1), (4, 0, 6), (6, 5, 1),
        (10, 5, 1), (11, 2, 1), (6, 4, 1), (8, 0, 6), (7, 5, 1), (11, 3, 1), (11, 5, 1),
        (11, 4, 1), (1, 0, 6), (6, 0, 4), (5, 0, 6), (7, 0, 4), (9, 0, 6), (10, 0, 4),
        (2, 0, 6), (3, 0, 6), (12, 0, 5),
    ],
    &[(0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 10), (5, 0, 10), (9, 0, 10)],
    &[
        (0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 9), (0, 10, 1), (5, 0, 9), (4, 10, 1),
        (9, 0, 9), (8, 10, 1),
    ],
    &[
        (0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 8), (0, 11, 1), (0, 10, 1), (5, 0, 8),
        (4, 11, 1), (4, 10, 1), (9, 0, 8), (8, 11, 1), (8, 10, 1),
    ],
    &[
        (0, 0, 10), (4, 0, 10), (8, 0, 10), (1, 0, 4), (0, 15, 1), (0, 14, 1), (0, 13, 1),
        (0, 12, 1), (0, 11, 1), (0, 10, 1), (5, 0, 4), (4, 15, 1), (4, 14, 1), (4, 13, 1),
        (4, 12, 1), (4, 11, 1), (4, 10, 1), (9, 0, 4), (8, 15, 1), (8, 14, 1), (8, 13, 1),
        (8, 12, 1), (8, 11, 1), (8, 10, 1),
    ],
];

const BC6H_ENDPOINT_BITS: [u32; 14] = [10, 7, 11, 11, 11, 9, 8, 8, 8, 6, 10, 11, 12, 16];

const BC6H_DELTA_BITS: [[u32; 3]; 14] = [
    [5, 5, 5],
    [6, 6, 6],
    [5, 4, 4],
    [4, 5, 4],
    [4, 4, 5],
    [5, 5, 5],
    [6, 5, 5],
    [5, 6, 5],
    [5, 5, 6],
    [6, 6, 6],
    [10, 10, 10],
    [9, 9, 9],
    [8, 8, 8],
    [4, 4, 4],
];

const BC6H_RESERVED: u8 = u8::MAX;

/// Mode index for the low five block bits; modes 0 and 1 use only two of them.
const BC6H_MODES: [u8; 32] = {
    let five_bit: [u8; 12] = [
        0b00010, 0b00110, 0b01010, 0b01110, 0b10010, 0b10110, 0b11010, 0b11110, 0b00011, 0b00111,
        0b01011, 0b01111,
    ];
    let mut modes = [BC6H_RESERVED; 32];
    let mut v = 0;
    while v < 32 {
        if v & 3 < 2 {
            modes[v] = (v & 3) as u8;
        }
        v += 1;
    }
    let mut m = 0;
    while m < 12 {
        modes[five_bit[m] as usize] = m as u8 + 2;
        m += 1;
    }
    modes
};

fn sign_extend(value: i32, bits: u32) -> i32 {
    (value << (32 - bits)) >> (32 - bits)
}

fn bc6h_unquantize<const SIGNED: bool>(value: i32, bits: u32) -> i32 {
    if !SIGNED {
        return if bits >= 15 {
            value
        } else if value == 0 {
            0
        } else if value == (1 << bits) - 1 {
            0xffff
        } else {
            ((value << 16) + 0x8000) >> bits
        };
    }
    if bits >= 16 {
        return value;
    }
    let magnitude = value.abs();
    let unquantized = if magnitude == 0 {
        0
    } else if magnitude >= (1 << (bits - 1)) - 1 {
        0x7fff
    } else {
        ((magnitude << 15) + 0x4000) >> (bits - 1)
    };
    if value < 0 { -unquantized } else { unquantized }
}

fn bc6h_finish<const SIGNED: bool>(value: i32) -> u16 {
    if !SIGNED {
        return ((value * 31) >> 6) as u16;
    }
    let magnitude = ((value.abs() * 31) >> 5) as u16;
    if value < 0 && magnitude != 0 {
        0x8000 | magnitude
    } else {
        magnitude
    }
}

fn decode_bc6h<const SIGNED: bool>(block: &[u8; 16]) -> [[u16; 4]; 16] {
    let mode = BC6H_MODES[block[0] as usize & 31];
    if mode == BC6H_RESERVED {
        return [[0, 0, 0, HALF_ONE]; 16];
    }
    let mode = mode as usize;
    let raw = u128::from_le_bytes(*block);
    let mut position = if mode < 2 { 2 } else { 5 };
    let mut fields = [0i32; 13];
    for &(field, shift, count) in BC6H_FIELDS[mode] {
        let value = (raw >> position) as u32 & ((1 << count) - 1);
        fields[usize::from(field)] |= (value << shift) as i32;
        position += u32::from(count);
    }
    let mut bits = Bits(raw >> position);
    let two_subsets = mode < 10;
    let endpoint_count = if two_subsets { 4 } else { 2 };
    let base_bits = BC6H_ENDPOINT_BITS[mode];
    let transformed = mode != 9 && mode != 10;
    for (channel, endpoints) in fields.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let endpoints = &mut endpoints[..endpoint_count];
        if SIGNED {
            endpoints[0] = sign_extend(endpoints[0], base_bits);
        }
        if transformed || SIGNED {
            for e in &mut endpoints[1..] {
                *e = sign_extend(*e, BC6H_DELTA_BITS[mode][channel]);
            }
        }
        if transformed {
            let base = endpoints[0];
            for e in &mut endpoints[1..] {
                let sum = (*e + base) & ((1 << base_bits) - 1);
                *e = if SIGNED {
                    sign_extend(sum, base_bits)
                } else {
                    sum
                };
            }
        }
        for e in endpoints {
            *e = bc6h_unquantize::<SIGNED>(*e, base_bits);
        }
    }
    let (partition, anchors, index_bits) = if two_subsets {
        let p = fields[12] as usize;
        (PARTITIONS2[p], ANCHORS2[p], 3)
    } else {
        (0, 1, 4)
    };
    let weights = weights(index_bits);
    let indices = bits.take_indices(index_bits, anchors);
    std::array::from_fn(|t| {
        let weight = weights[usize::from(indices[t])] as i32;
        let subset = ((partition >> (2 * t)) & 1) as usize;
        let channel = |c: usize| {
            let e0 = fields[c * 4 + subset * 2];
            let e1 = fields[c * 4 + subset * 2 + 1];
            bc6h_finish::<SIGNED>((e0 * (64 - weight) + e1 * weight + 32) >> 6)
        };
        [channel(0), channel(1), channel(2), HALF_ONE]
    })
}

/// Decodes a BC6H unsigned-float block to RGBA16F half-float bits with alpha 1.0. Reserved modes
/// decode to black.
pub fn decode_bc6h_ufloat_block(block: &[u8; 16]) -> [[u16; 4]; 16] {
    decode_bc6h::<false>(block)
}

/// Decodes a BC6H signed-float block to RGBA16F half-float bits with alpha 1.0. Reserved modes
/// decode to black.
pub fn decode_bc6h_sfloat_block(block: &[u8; 16]) -> [[u16; 4]; 16] {
    decode_bc6h::<true>(block)
}

#[derive(Clone, Copy)]
enum PBits {
    None,
    PerSubset,
    PerEndpoint,
}

#[derive(Clone, Copy)]
struct Bc7Mode {
    subsets: usize,
    partition_bits: u32,
    rotation_bits: u32,
    selector_bits: u32,
    color_bits: u32,
    alpha_bits: u32,
    pbits: PBits,
    index_bits: u32,
    index2_bits: u32,
}

#[rustfmt::skip]
const BC7_MODES: [Bc7Mode; 8] = {
    #[allow(clippy::too_many_arguments)]
    const fn mode(
        subsets: usize, partition_bits: u32, rotation_bits: u32, selector_bits: u32,
        color_bits: u32, alpha_bits: u32, pbits: PBits, index_bits: u32, index2_bits: u32,
    ) -> Bc7Mode {
        Bc7Mode {
            subsets, partition_bits, rotation_bits, selector_bits, color_bits, alpha_bits, pbits,
            index_bits, index2_bits,
        }
    }
    [
        mode(3, 4, 0, 0, 4, 0, PBits::PerEndpoint, 3, 0),
        mode(2, 6, 0, 0, 6, 0, PBits::PerSubset, 3, 0),
        mode(3, 6, 0, 0, 5, 0, PBits::None, 2, 0),
        mode(2, 6, 0, 0, 7, 0, PBits::PerEndpoint, 2, 0),
        mode(1, 0, 2, 1, 5, 6, PBits::None, 2, 3),
        mode(1, 0, 2, 0, 7, 8, PBits::None, 2, 2),
        mode(1, 0, 0, 0, 7, 7, PBits::PerEndpoint, 4, 0),
        mode(2, 6, 0, 0, 5, 5, PBits::PerEndpoint, 2, 0),
    ]
};

fn expand_bits(value: u32, bits: u32) -> u32 {
    let value = value << (8 - bits);
    value | value >> bits
}

/// Decodes a BC7 block to RGBA8. Blocks without a valid mode bit decode to transparent black.
pub fn decode_bc7_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mode_index = block[0].trailing_zeros();
    if mode_index >= 8 {
        return [[0; 4]; 16];
    }
    let mode = BC7_MODES[mode_index as usize];
    let mut bits = Bits(u128::from_le_bytes(*block) >> (mode_index + 1));
    let partition = bits.take(mode.partition_bits) as usize;
    let rotation = bits.take(mode.rotation_bits) as usize;
    let selector = bits.take(mode.selector_bits);

    let endpoint_count = mode.subsets * 2;
    let mut endpoints = [[0u32; 4]; 6];
    for channel in 0..3 {
        for endpoint in &mut endpoints[..endpoint_count] {
            endpoint[channel] = bits.take(mode.color_bits);
        }
    }
    for endpoint in &mut endpoints[..endpoint_count] {
        endpoint[3] = bits.take(mode.alpha_bits);
    }
    let pbit_count = match mode.pbits {
        PBits::None => 0,
        PBits::PerSubset => {
            for pair in endpoints[..endpoint_count].as_chunks_mut::<2>().0 {
                let p = bits.take(1);
                for endpoint in pair {
                    *endpoint = endpoint.map(|v| v << 1 | p);
                }
            }
            1
        }
        PBits::PerEndpoint => {
            for endpoint in &mut endpoints[..endpoint_count] {
                let p = bits.take(1);
                *endpoint = endpoint.map(|v| v << 1 | p);
            }
            1
        }
    };
    for endpoint in &mut endpoints[..endpoint_count] {
        for channel in &mut endpoint[..3] {
            *channel = expand_bits(*channel, mode.color_bits + pbit_count);
        }
        endpoint[3] = if mode.alpha_bits == 0 {
            255
        } else {
            expand_bits(endpoint[3], mode.alpha_bits + pbit_count)
        };
    }

    let (partitions, anchors) = match mode.subsets {
        1 => (0, 1),
        2 => (PARTITIONS2[partition], ANCHORS2[partition]),
        _ => (PARTITIONS3[partition], ANCHORS3[partition]),
    };
    let indices = bits.take_indices(mode.index_bits, anchors);
    let indices2 = if mode.index2_bits == 0 {
        [0; 16]
    } else {
        bits.take_indices(mode.index2_bits, 1)
    };
    let weights1 = weights(mode.index_bits);
    let weights2 = weights(mode.index2_bits);

    std::array::from_fn(|t| {
        let subset = ((partitions >> (2 * t)) & 3) as usize;
        let (e0, e1) = (endpoints[subset * 2], endpoints[subset * 2 + 1]);
        let w1 = weights1[usize::from(indices[t])];
        let (color_weight, alpha_weight) = match (mode.index2_bits, selector) {
            (0, _) => (w1, w1),
            (_, 0) => (w1, weights2[usize::from(indices2[t])]),
            _ => (weights2[usize::from(indices2[t])], w1),
        };
        let interpolate = |c: usize, w: u32| ((e0[c] * (64 - w) + e1[c] * w + 32) >> 6) as u8;
        let mut texel = [
            interpolate(0, color_weight),
            interpolate(1, color_weight),
            interpolate(2, color_weight),
            interpolate(3, alpha_weight),
        ];
        if rotation != 0 {
            texel.swap(3, rotation - 1);
        }
        texel
    })
}

fn half4_bytes(texel: [u16; 4]) -> [u8; 8] {
    let [[a, b], [c, d], [e, f], [g, h]] = texel.map(u16::to_le_bytes);
    [a, b, c, d, e, f, g, h]
}

image_decoder!(
    /// Decodes a BC1 image to RGBA8.
    decode_bc1,
    decode_bc1_block
);
image_decoder!(
    /// Decodes a BC2 image to RGBA8.
    decode_bc2,
    decode_bc2_block
);
image_decoder!(
    /// Decodes a BC3 image to RGBA8.
    decode_bc3,
    decode_bc3_block
);
image_decoder!(
    /// Decodes a BC4 unsigned image to R8.
    decode_bc4_unorm,
    |b: &[u8; 8]| decode_bc4_unorm_block(b).map(|v| [v])
);
image_decoder!(
    /// Decodes a BC4 signed image to R8 snorm.
    decode_bc4_snorm,
    |b: &[u8; 8]| decode_bc4_snorm_block(b).map(|v| [v as u8])
);
image_decoder!(
    /// Decodes a BC5 unsigned image to RG8.
    decode_bc5_unorm,
    decode_bc5_unorm_block
);
image_decoder!(
    /// Decodes a BC5 signed image to RG8 snorm.
    decode_bc5_snorm,
    |b: &[u8; 16]| decode_bc5_snorm_block(b).map(|t| t.map(|v| v as u8))
);
image_decoder!(
    /// Decodes a BC6H unsigned-float image to RGBA16F (little-endian half floats).
    decode_bc6h_ufloat,
    |b: &[u8; 16]| decode_bc6h_ufloat_block(b).map(half4_bytes)
);
image_decoder!(
    /// Decodes a BC6H signed-float image to RGBA16F (little-endian half floats).
    decode_bc6h_sfloat,
    |b: &[u8; 16]| decode_bc6h_sfloat_block(b).map(half4_bytes)
);
image_decoder!(
    /// Decodes a BC7 image to RGBA8.
    decode_bc7,
    decode_bc7_block
);
