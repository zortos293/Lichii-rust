//! ETC2 and EAC block decoders. Desktop GPUs lack these formats, so the renderer decodes them on
//! the CPU.
//!
//! Block functions take one block and return its 16 texels row-major. Image functions take a
//! tightly packed, already deswizzled block array and write tightly packed texels, cropping the
//! blocks that overhang the right and bottom edges. 11-bit EAC values widen to 16 bits by bit
//! replication (negative values replicate their magnitude), as Mesa does.

use crate::block::image_decoder;

const ETC_MODIFIERS: [[i32; 2]; 8] = [
    [2, 8],
    [5, 17],
    [9, 29],
    [13, 42],
    [18, 60],
    [24, 80],
    [33, 106],
    [47, 183],
];

const ETC_DISTANCES: [i32; 8] = [3, 6, 11, 16, 23, 32, 41, 64];

const EAC_MODIFIERS: [[i32; 8]; 16] = [
    [-3, -6, -9, -15, 2, 5, 8, 14],
    [-3, -7, -10, -13, 2, 6, 9, 12],
    [-2, -5, -8, -13, 1, 4, 7, 12],
    [-2, -4, -6, -13, 1, 3, 5, 12],
    [-3, -6, -8, -12, 2, 5, 7, 11],
    [-3, -7, -9, -11, 2, 6, 8, 10],
    [-4, -7, -8, -11, 3, 6, 7, 10],
    [-3, -5, -8, -11, 2, 4, 7, 10],
    [-2, -6, -8, -10, 1, 5, 7, 9],
    [-2, -5, -8, -10, 1, 4, 7, 9],
    [-2, -4, -8, -10, 1, 3, 7, 9],
    [-2, -5, -7, -10, 1, 4, 6, 9],
    [-3, -4, -7, -10, 2, 3, 6, 9],
    [-1, -2, -3, -10, 0, 1, 2, 9],
    [-4, -6, -8, -9, 3, 5, 7, 8],
    [-3, -5, -7, -9, 2, 4, 6, 8],
];

const TRANSPARENT: [u8; 4] = [0; 4];

fn field(bits: u64, shift: u32, width: u32) -> i32 {
    ((bits >> shift) & ((1 << width) - 1)) as i32
}

fn offset(color: [i32; 3], delta: i32) -> [u8; 4] {
    let c = |v: i32| (v + delta).clamp(0, 255) as u8;
    [c(color[0]), c(color[1]), c(color[2]), 255]
}

/// The 2-bit palette index of the texel at row-major position `t`; ETC stores indices
/// column-major, most significant bits in the upper half.
fn etc_index(bits: u64, t: usize) -> usize {
    let k = (t % 4) * 4 + t / 4;
    ((((bits >> (k + 16)) & 1) << 1) | ((bits >> k) & 1)) as usize
}

fn individual_or_differential(bits: u64, colors: [[i32; 3]; 2], opaque: bool) -> [[u8; 4]; 16] {
    let flip = bits & (1 << 32) != 0;
    let palettes = [field(bits, 37, 3), field(bits, 34, 3)].map(|table| {
        let [small, large] = ETC_MODIFIERS[table as usize];
        [small, large, -small, -large]
    });
    let palettes: [[[u8; 4]; 4]; 2] = std::array::from_fn(|s| {
        std::array::from_fn(|i| match (opaque, i) {
            (false, 0) => offset(colors[s], 0),
            (false, 2) => TRANSPARENT,
            _ => offset(colors[s], palettes[s][i]),
        })
    });
    std::array::from_fn(|t| {
        let (x, y) = (t % 4, t / 4);
        let subblock = usize::from(if flip { y >= 2 } else { x >= 2 });
        palettes[subblock][etc_index(bits, t)]
    })
}

fn with_palette(bits: u64, palette: [[u8; 4]; 4], opaque: bool) -> [[u8; 4]; 16] {
    std::array::from_fn(|t| match etc_index(bits, t) {
        2 if !opaque => TRANSPARENT,
        i => palette[i],
    })
}

fn expand4(v: i32) -> i32 {
    v << 4 | v
}

fn t_mode(bits: u64, opaque: bool) -> [[u8; 4]; 16] {
    let c0 = [
        field(bits, 59, 2) << 2 | field(bits, 56, 2),
        field(bits, 52, 4),
        field(bits, 48, 4),
    ]
    .map(expand4);
    let c1 = [field(bits, 44, 4), field(bits, 40, 4), field(bits, 36, 4)].map(expand4);
    let d = ETC_DISTANCES[(field(bits, 34, 2) << 1 | field(bits, 32, 1)) as usize];
    let palette = [offset(c0, 0), offset(c1, d), offset(c1, 0), offset(c1, -d)];
    with_palette(bits, palette, opaque)
}

fn h_mode(bits: u64, opaque: bool) -> [[u8; 4]; 16] {
    let c0 = [
        field(bits, 59, 4),
        field(bits, 56, 3) << 1 | field(bits, 52, 1),
        field(bits, 51, 1) << 3 | field(bits, 47, 3),
    ]
    .map(expand4);
    let c1 = [field(bits, 43, 4), field(bits, 39, 4), field(bits, 35, 4)].map(expand4);
    let packed = |c: [i32; 3]| c[0] << 16 | c[1] << 8 | c[2];
    let distance =
        field(bits, 34, 1) << 2 | field(bits, 32, 1) << 1 | i32::from(packed(c0) >= packed(c1));
    let d = ETC_DISTANCES[distance as usize];
    let palette = [offset(c0, d), offset(c0, -d), offset(c1, d), offset(c1, -d)];
    with_palette(bits, palette, opaque)
}

fn planar_mode(bits: u64) -> [[u8; 4]; 16] {
    let e6 = |v: i32| v << 2 | v >> 4;
    let e7 = |v: i32| v << 1 | v >> 6;
    let origin = [
        e6(field(bits, 57, 6)),
        e7(field(bits, 56, 1) << 6 | field(bits, 49, 6)),
        e6(field(bits, 48, 1) << 5 | field(bits, 43, 2) << 3 | field(bits, 39, 3)),
    ];
    let horizontal = [
        e6(field(bits, 34, 5) << 1 | field(bits, 32, 1)),
        e7(field(bits, 25, 7)),
        e6(field(bits, 19, 6)),
    ];
    let vertical = [
        e6(field(bits, 13, 6)),
        e7(field(bits, 6, 7)),
        e6(field(bits, 0, 6)),
    ];
    std::array::from_fn(|t| {
        let (x, y) = ((t % 4) as i32, (t / 4) as i32);
        let c = |i: usize| {
            let o = origin[i];
            ((x * (horizontal[i] - o) + y * (vertical[i] - o) + 4 * o + 2) >> 2).clamp(0, 255) as u8
        };
        [c(0), c(1), c(2), 255]
    })
}

fn decode_etc(block: &[u8; 8], punch_through: bool) -> [[u8; 4]; 16] {
    let bits = u64::from_be_bytes(*block);
    let differential = bits & (1 << 33) != 0;
    if !punch_through && !differential {
        let c = |shift| expand4(field(bits, shift, 4));
        return individual_or_differential(
            bits,
            [[c(60), c(52), c(44)], [c(56), c(48), c(40)]],
            true,
        );
    }
    let opaque = !punch_through || differential;
    let base = [field(bits, 59, 5), field(bits, 51, 5), field(bits, 43, 5)];
    let delta =
        [field(bits, 56, 3), field(bits, 48, 3), field(bits, 40, 3)].map(|d| (d << 29) >> 29);
    let second: [i32; 3] = std::array::from_fn(|i| base[i] + delta[i]);
    let out_of_range = |v: i32| !(0..32).contains(&v);
    if out_of_range(second[0]) {
        t_mode(bits, opaque)
    } else if out_of_range(second[1]) {
        h_mode(bits, opaque)
    } else if out_of_range(second[2]) {
        planar_mode(bits)
    } else {
        let e5 = |c: [i32; 3]| c.map(|v| v << 3 | v >> 2);
        individual_or_differential(bits, [e5(base), e5(second)], opaque)
    }
}

/// Decodes an ETC2 RGB block (ETC1 modes plus T, H and planar) to RGBA8 with alpha 255.
pub fn decode_etc2_rgb_block(block: &[u8; 8]) -> [[u8; 4]; 16] {
    decode_etc(block, false)
}

/// Decodes an ETC2 RGB punch-through-alpha block to RGBA8; transparent texels are (0, 0, 0, 0).
pub fn decode_etc2_rgb_pta_block(block: &[u8; 8]) -> [[u8; 4]; 16] {
    decode_etc(block, true)
}

struct EacBlock {
    base: i32,
    multiplier: i32,
    modifiers: &'static [i32; 8],
    bits: u64,
}

impl EacBlock {
    fn new(block: &[u8; 8]) -> Self {
        let bits = u64::from_be_bytes(*block);
        Self {
            base: i32::from(block[0]),
            multiplier: field(bits, 52, 4),
            modifiers: &EAC_MODIFIERS[field(bits, 48, 4) as usize],
            bits,
        }
    }

    fn eleven_bit_scale(&self) -> i32 {
        if self.multiplier == 0 {
            1
        } else {
            self.multiplier * 8
        }
    }

    /// Maps every texel's 3-bit index (stored column-major, first texel in the top bits) through
    /// `palette`.
    fn texels<T: Copy>(&self, palette: [T; 8]) -> [T; 16] {
        std::array::from_fn(|t| {
            let k = (t % 4) * 4 + t / 4;
            palette[field(self.bits, (15 - k as u32) * 3, 3) as usize]
        })
    }
}

fn eac_alpha(block: &[u8; 8]) -> [u8; 16] {
    let eac = EacBlock::new(block);
    eac.texels(
        eac.modifiers
            .map(|m| (eac.base + m * eac.multiplier).clamp(0, 255) as u8),
    )
}

/// Decodes an ETC2 RGBA block (EAC alpha, then ETC2 RGB) to RGBA8.
pub fn decode_etc2_rgba_block(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let (alpha, color) = block.split_at(8);
    let mut texels = decode_etc(color.try_into().unwrap(), false);
    for (texel, a) in texels.iter_mut().zip(eac_alpha(alpha.try_into().unwrap())) {
        texel[3] = a;
    }
    texels
}

/// Decodes an EAC R11 unsigned block to R16 unorm.
pub fn decode_eac_r11_unorm_block(block: &[u8; 8]) -> [u16; 16] {
    let eac = EacBlock::new(block);
    let scale = eac.eleven_bit_scale();
    eac.texels(eac.modifiers.map(|m| {
        let v = (eac.base * 8 + 4 + m * scale).clamp(0, 2047) as u16;
        v << 5 | v >> 6
    }))
}

/// Decodes an EAC R11 signed block to R16 snorm; the base -128 reads as -127.
pub fn decode_eac_r11_snorm_block(block: &[u8; 8]) -> [i16; 16] {
    let eac = EacBlock::new(block);
    let base = i32::from(block[0] as i8).max(-127);
    let scale = eac.eleven_bit_scale();
    eac.texels(eac.modifiers.map(|m| {
        let v = (base * 8 + m * scale).clamp(-1023, 1023);
        let magnitude = (v.abs() << 5 | v.abs() >> 5) as i16;
        if v < 0 { -magnitude } else { magnitude }
    }))
}

fn split_halves(block: &[u8; 16]) -> (&[u8; 8], &[u8; 8]) {
    let (first, second) = block.split_at(8);
    (first.try_into().unwrap(), second.try_into().unwrap())
}

/// Decodes an EAC RG11 unsigned block to RG16 unorm.
pub fn decode_eac_rg11_unorm_block(block: &[u8; 16]) -> [[u16; 2]; 16] {
    let (r, g) = split_halves(block);
    let (r, g) = (decode_eac_r11_unorm_block(r), decode_eac_r11_unorm_block(g));
    std::array::from_fn(|t| [r[t], g[t]])
}

/// Decodes an EAC RG11 signed block to RG16 snorm.
pub fn decode_eac_rg11_snorm_block(block: &[u8; 16]) -> [[i16; 2]; 16] {
    let (r, g) = split_halves(block);
    let (r, g) = (decode_eac_r11_snorm_block(r), decode_eac_r11_snorm_block(g));
    std::array::from_fn(|t| [r[t], g[t]])
}

image_decoder!(
    /// Decodes an ETC2 RGB image to RGBA8.
    decode_etc2_rgb,
    decode_etc2_rgb_block
);
image_decoder!(
    /// Decodes an ETC2 RGB punch-through-alpha image to RGBA8.
    decode_etc2_rgb_pta,
    decode_etc2_rgb_pta_block
);
image_decoder!(
    /// Decodes an ETC2 RGBA image to RGBA8.
    decode_etc2_rgba,
    decode_etc2_rgba_block
);
image_decoder!(
    /// Decodes an EAC R11 unsigned image to R16 unorm (little-endian).
    decode_eac_r11_unorm,
    |b: &[u8; 8]| decode_eac_r11_unorm_block(b).map(u16::to_le_bytes)
);
image_decoder!(
    /// Decodes an EAC R11 signed image to R16 snorm (little-endian).
    decode_eac_r11_snorm,
    |b: &[u8; 8]| decode_eac_r11_snorm_block(b).map(i16::to_le_bytes)
);
image_decoder!(
    /// Decodes an EAC RG11 unsigned image to RG16 unorm (little-endian).
    decode_eac_rg11_unorm,
    |b: &[u8; 16]| decode_eac_rg11_unorm_block(b).map(|[r, g]| {
        let ([r0, r1], [g0, g1]) = (r.to_le_bytes(), g.to_le_bytes());
        [r0, r1, g0, g1]
    })
);
image_decoder!(
    /// Decodes an EAC RG11 signed image to RG16 snorm (little-endian).
    decode_eac_rg11_snorm,
    |b: &[u8; 16]| decode_eac_rg11_snorm_block(b).map(|[r, g]| {
        let ([r0, r1], [g0, g1]) = (r.to_le_bytes(), g.to_le_bytes());
        [r0, r1, g0, g1]
    })
);
