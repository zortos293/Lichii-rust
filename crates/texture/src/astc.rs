//! CPU decoder for 2D ASTC blocks, following the Khronos ASTC specification (the
//! `KHR_texture_compression_astc_ldr`/`_hdr` decode, as in the Khronos Data Format Specification).
//!
//! Three decode profiles exist:
//! - LDR linear: 8-bit output using the `decode_unorm8` rules (top 8 bits of the 16-bit
//!   interpolation result).
//! - LDR sRGB: 8-bit sRGB-encoded output; endpoints expand as `(e << 8) | 0x80`.
//! - HDR: half-float output; LDR endpoints become UNORM16 values and HDR endpoints are
//!   logarithmic values converted to FP16.
//!
//! Illegal encodings decode to the error color: opaque magenta in the LDR profiles and NaN
//! (`0xFFFF`) in the HDR profile. A partition with HDR endpoints decodes as magenta in the LDR
//! profiles, while the block's other partitions decode normally.

/// Bytes in one ASTC block, for every footprint.
pub const BLOCK_BYTES: usize = 16;

const MAX_TEXELS: usize = 144;
const MAX_WEIGHTS: usize = 64;

/// Decodes one block in the LDR profile into `block_w * block_h` RGBA8 texels, row-major.
///
/// With `srgb` set the texels are sRGB-encoded bytes (the caller samples them through an sRGB
/// format); otherwise they follow `decode_unorm8`.
///
/// # Panics
///
/// Panics if the footprint is not within 4x4..=12x12 or `out` holds fewer than
/// `block_w * block_h` texels.
pub fn decode_block_rgba8(
    block: &[u8; 16],
    block_w: u32,
    block_h: u32,
    srgb: bool,
    out: &mut [[u8; 4]],
) {
    if srgb {
        decode_block::<Srgb>(block, block_w, block_h, out);
    } else {
        decode_block::<Ldr>(block, block_w, block_h, out);
    }
}

/// Decodes one block in the HDR profile into `block_w * block_h` RGBA half-float texels
/// (IEEE 754 binary16 bit patterns), row-major.
///
/// # Panics
///
/// Panics if the footprint is not within 4x4..=12x12 or `out` holds fewer than
/// `block_w * block_h` texels.
pub fn decode_block_rgba16f(block: &[u8; 16], block_w: u32, block_h: u32, out: &mut [[u16; 4]]) {
    decode_block::<Hdr>(block, block_w, block_h, out);
}

/// Decodes a linear (deswizzled) image of 2D ASTC blocks into tightly packed RGBA8.
///
/// `data` holds `depth` slices, each `ceil(width / block_w) * ceil(height / block_h)` blocks in
/// row-major order. `out` receives `width * height * depth * 4` bytes; texels of edge blocks
/// that fall outside the image are dropped. `srgb` selects the sRGB decode as in
/// [`decode_block_rgba8`].
///
/// # Panics
///
/// Panics if the footprint is not within 4x4..=12x12, or `data` or `out` is too short.
#[allow(clippy::too_many_arguments)]
pub fn decode_rgba8(
    data: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    block_w: u32,
    block_h: u32,
    srgb: bool,
    out: &mut [u8],
) {
    let texels = image_texels(width, height, depth);
    let out: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut out[..texels * 4]);
    if srgb {
        decode_image::<Srgb>(data, width, height, depth, block_w, block_h, out);
    } else {
        decode_image::<Ldr>(data, width, height, depth, block_w, block_h, out);
    }
}

/// Decodes a linear (deswizzled) image of 2D ASTC blocks into tightly packed RGBA half floats
/// (binary16 bit patterns) in the HDR profile.
///
/// The layout of `data` matches [`decode_rgba8`]; `out` receives `width * height * depth * 4`
/// values.
///
/// # Panics
///
/// Panics if the footprint is not within 4x4..=12x12, or `data` or `out` is too short.
pub fn decode_rgba16f(
    data: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    block_w: u32,
    block_h: u32,
    out: &mut [u16],
) {
    let texels = image_texels(width, height, depth);
    let out: &mut [[u16; 4]] = bytemuck::cast_slice_mut(&mut out[..texels * 4]);
    decode_image::<Hdr>(data, width, height, depth, block_w, block_h, out);
}

fn image_texels(width: u32, height: u32, depth: u32) -> usize {
    width as usize * height as usize * depth as usize
}

fn check_footprint(block_w: u32, block_h: u32) {
    assert!(
        (4..=12).contains(&block_w) && (4..=12).contains(&block_h),
        "unsupported ASTC footprint {block_w}x{block_h}"
    );
}

fn decode_image<P: Profile>(
    data: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    block_w: u32,
    block_h: u32,
    out: &mut [P::Texel],
) {
    check_footprint(block_w, block_h);
    let (width, height, depth) = (width as usize, height as usize, depth as usize);
    let (bw, bh) = (block_w as usize, block_h as usize);
    let blocks_x = width.div_ceil(bw);
    let blocks_y = height.div_ceil(bh);
    let block_count = blocks_x * blocks_y * depth;
    assert!(
        data.len() >= block_count * BLOCK_BYTES,
        "ASTC data holds {} bytes, {width}x{height}x{depth} at {bw}x{bh} needs {}",
        data.len(),
        block_count * BLOCK_BYTES
    );
    if width * height == 0 {
        return;
    }

    let mut texels = [P::Texel::default(); MAX_TEXELS];
    let (blocks, _) = data.as_chunks::<BLOCK_BYTES>();
    let slices = out
        .chunks_exact_mut(width * height)
        .zip(blocks.chunks_exact(blocks_x * blocks_y));
    for (slice, slice_blocks) in slices {
        for (by, row_blocks) in slice_blocks.chunks_exact(blocks_x).enumerate() {
            let rows = bh.min(height - by * bh);
            for (bx, block) in row_blocks.iter().enumerate() {
                decode_block::<P>(block, block_w, block_h, &mut texels);
                let x0 = bx * bw;
                let cols = bw.min(width - x0);
                for (row, src) in texels.chunks_exact(bw).take(rows).enumerate() {
                    let dst = (by * bh + row) * width + x0;
                    slice[dst..dst + cols].copy_from_slice(&src[..cols]);
                }
            }
        }
    }
}

/// How a decode profile expands endpoints and converts interpolated values.
trait Profile {
    type Texel: Copy + Default;
    const ERROR: Self::Texel;

    fn void_extent(color: [u16; 4], hdr: bool) -> Self::Texel;
    fn expand(endpoints: Endpoints) -> Expanded;
    fn texel(color: [i32; 4], lns: [bool; 4]) -> Self::Texel;
}

struct Ldr;
struct Srgb;
struct Hdr;

/// Endpoints as the color endpoint mode produced them: 8-bit LDR values, or 16-bit values in
/// the lanes flagged HDR.
#[derive(Clone, Copy)]
struct Endpoints {
    e0: [i32; 4],
    e1: [i32; 4],
    rgb_hdr: bool,
    alpha_hdr: bool,
    default_alpha: bool,
}

/// Endpoints expanded to 16 bits, with the lanes holding logarithmic HDR values.
#[derive(Clone, Copy, Default)]
struct Expanded {
    e0: [i32; 4],
    e1: [i32; 4],
    lns: [bool; 4],
}

const MAGENTA: [i32; 4] = [0xFF, 0x00, 0xFF, 0xFF];

fn ldr_endpoints(ep: Endpoints, expand: impl Fn(i32) -> i32) -> Expanded {
    let (e0, e1) = if ep.rgb_hdr || ep.alpha_hdr {
        (MAGENTA, MAGENTA)
    } else {
        (ep.e0, ep.e1)
    };
    Expanded {
        e0: e0.map(&expand),
        e1: e1.map(&expand),
        lns: [false; 4],
    }
}

fn top_byte(color: [i32; 4]) -> [u8; 4] {
    color.map(|c| (c >> 8) as u8)
}

impl Profile for Ldr {
    type Texel = [u8; 4];
    const ERROR: [u8; 4] = [0xFF, 0x00, 0xFF, 0xFF];

    fn void_extent(color: [u16; 4], hdr: bool) -> [u8; 4] {
        if hdr {
            Self::ERROR
        } else {
            color.map(|c| (c >> 8) as u8)
        }
    }

    fn expand(endpoints: Endpoints) -> Expanded {
        ldr_endpoints(endpoints, |e| e * 257)
    }

    fn texel(color: [i32; 4], _lns: [bool; 4]) -> [u8; 4] {
        top_byte(color)
    }
}

impl Profile for Srgb {
    type Texel = [u8; 4];
    const ERROR: [u8; 4] = Ldr::ERROR;

    fn void_extent(color: [u16; 4], hdr: bool) -> [u8; 4] {
        Ldr::void_extent(color, hdr)
    }

    fn expand(endpoints: Endpoints) -> Expanded {
        ldr_endpoints(endpoints, |e| (e << 8) | 0x80)
    }

    fn texel(color: [i32; 4], _lns: [bool; 4]) -> [u8; 4] {
        top_byte(color)
    }
}

impl Profile for Hdr {
    type Texel = [u16; 4];
    const ERROR: [u16; 4] = [0xFFFF; 4];

    fn void_extent(color: [u16; 4], hdr: bool) -> [u16; 4] {
        if hdr {
            color.map(quiet_nan)
        } else {
            color.map(|c| unorm16_to_f16(c as i32))
        }
    }

    fn expand(mut ep: Endpoints) -> Expanded {
        if ep.default_alpha {
            ep.e0[3] = 0x7800;
            ep.e1[3] = 0x7800;
            ep.alpha_hdr = true;
        }
        let lns = [ep.rgb_hdr, ep.rgb_hdr, ep.rgb_hdr, ep.alpha_hdr];
        let scale = |e: [i32; 4]| std::array::from_fn(|i| if lns[i] { e[i] } else { e[i] * 257 });
        Expanded {
            e0: scale(ep.e0),
            e1: scale(ep.e1),
            lns,
        }
    }

    fn texel(color: [i32; 4], lns: [bool; 4]) -> [u16; 4] {
        std::array::from_fn(|i| {
            if lns[i] {
                lns_to_f16(color[i])
            } else {
                unorm16_to_f16(color[i])
            }
        })
    }
}

/// Quiets a signaling NaN, as a pass through FP32 does; the reference decoder does this to HDR
/// void-extent colors.
fn quiet_nan(half: u16) -> u16 {
    if half & 0x7C00 == 0x7C00 && half & 0x03FF != 0 {
        half | 0x0200
    } else {
        half
    }
}

/// Converts `value / 65536` to FP16 with truncation, except that 0xFFFF becomes 1.0.
fn unorm16_to_f16(value: i32) -> u16 {
    let value = value as u32;
    let normal = ((value as f32).to_bits() >> 13).wrapping_sub(128 << 10);
    match value {
        0xFFFF => 0x3C00,
        0..4 => (value << 8) as u16,
        _ => normal as u16,
    }
}

fn lns_to_f16(value: i32) -> u16 {
    let mc = value & 0x7FF;
    let ec = value >> 11;
    let mt = if mc < 512 {
        mc * 3
    } else if mc < 1536 {
        mc * 4 - 512
    } else {
        mc * 5 - 2048
    };
    ((ec << 10) | (mt >> 3)).min(0x7BFF) as u16
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Ise {
    Bits,
    Trits,
    Quints,
}

/// Integer sequence encodings by quantization level as `(bits, kind)`, for the ranges 2, 3, 4,
/// 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192 and 256.
const ISE_LEVELS: [(u32, Ise); 21] = [
    (1, Ise::Bits),
    (0, Ise::Trits),
    (2, Ise::Bits),
    (0, Ise::Quints),
    (1, Ise::Trits),
    (3, Ise::Bits),
    (1, Ise::Quints),
    (2, Ise::Trits),
    (4, Ise::Bits),
    (2, Ise::Quints),
    (3, Ise::Trits),
    (5, Ise::Bits),
    (3, Ise::Quints),
    (4, Ise::Trits),
    (6, Ise::Bits),
    (4, Ise::Quints),
    (5, Ise::Trits),
    (7, Ise::Bits),
    (5, Ise::Quints),
    (6, Ise::Trits),
    (8, Ise::Bits),
];

const QUANT_6: usize = 4;

const fn ise_range(level: usize) -> u32 {
    match ISE_LEVELS[level] {
        (n, Ise::Bits) => 1 << n,
        (n, Ise::Trits) => 3 << n,
        (n, Ise::Quints) => 5 << n,
    }
}

const fn ise_bit_count(count: u32, level: usize) -> u32 {
    let (bits, kind) = ISE_LEVELS[level];
    bits * count
        + match kind {
            Ise::Bits => 0,
            Ise::Trits => (8 * count).div_ceil(5),
            Ise::Quints => (7 * count).div_ceil(3),
        }
}

const fn bit(value: u32, n: u32) -> u32 {
    (value >> n) & 1
}

const fn bits(value: u32, lo: u32, count: u32) -> u32 {
    (value >> lo) & ((1 << count) - 1)
}

/// Trit digits (least significant first) of each 8-bit packed trit block.
static TRITS: [[u8; 5]; 256] = {
    let mut table = [[0u8; 5]; 256];
    let mut t = 0;
    while t < 256 {
        let (c, t4, t3) = if bits(t, 2, 3) == 0b111 {
            ((bits(t, 5, 3) << 2) | bits(t, 0, 2), 2, 2)
        } else if bits(t, 5, 2) == 0b11 {
            (bits(t, 0, 5), 2, bit(t, 7))
        } else {
            (bits(t, 0, 5), bit(t, 7), bits(t, 5, 2))
        };
        let (t2, t1, t0) = if bits(c, 0, 2) == 0b11 {
            (
                2,
                bit(c, 4),
                (bit(c, 3) << 1) | (bit(c, 2) & !bit(c, 3) & 1),
            )
        } else if bits(c, 2, 2) == 0b11 {
            (2, 2, bits(c, 0, 2))
        } else {
            (
                bit(c, 4),
                bits(c, 2, 2),
                (bit(c, 1) << 1) | (bit(c, 0) & !bit(c, 1) & 1),
            )
        };
        table[t as usize] = [t0 as u8, t1 as u8, t2 as u8, t3 as u8, t4 as u8];
        t += 1;
    }
    table
};

/// Quint digits (least significant first) of each 7-bit packed quint block.
static QUINTS: [[u8; 3]; 128] = {
    let mut table = [[0u8; 3]; 128];
    let mut q = 0;
    while q < 128 {
        let (q2, q1, q0) = if bits(q, 1, 2) == 0b11 && bits(q, 5, 2) == 0 {
            let not0 = !bit(q, 0) & 1;
            (
                (bit(q, 0) << 2) | ((bit(q, 4) & not0) << 1) | (bit(q, 3) & not0),
                4,
                4,
            )
        } else {
            let (q2, c) = if bits(q, 1, 2) == 0b11 {
                (
                    4,
                    (bits(q, 3, 2) << 3) | ((!bits(q, 5, 2) & 0b11) << 1) | bit(q, 0),
                )
            } else {
                (bits(q, 5, 2), bits(q, 0, 5))
            };
            if bits(c, 0, 3) == 0b101 {
                (q2, 4, bits(c, 3, 2))
            } else {
                (q2, bits(c, 3, 2), bits(c, 0, 3))
            }
        };
        table[q as usize] = [q0 as u8, q1 as u8, q2 as u8];
        q += 1;
    }
    table
};

/// Color endpoint unquantization to 0..=255, indexed by level and encoded value. Color values
/// never use the levels below range 6.
static COLOR_UNQUANT: [[u8; 256]; 21] = {
    let mut table = [[0u8; 256]; 21];
    let mut level = QUANT_6;
    while level < 21 {
        let (n, kind) = ISE_LEVELS[level];
        let mut v = 0;
        while v < ise_range(level) {
            table[level][v as usize] = if let Ise::Bits = kind {
                replicate(v, n, 8) as u8
            } else {
                let m = v & ((1 << n) - 1);
                let (b, c, d, e, f) = (bit(m, 1), bit(m, 2), bit(m, 3), bit(m, 4), bit(m, 5));
                let (big_b, big_c) = match (kind, n) {
                    (Ise::Trits, 1) => (0, 204),
                    (Ise::Quints, 1) => (0, 113),
                    (Ise::Trits, 2) => ((b << 8) | (b << 4) | (b << 2) | (b << 1), 93),
                    (Ise::Quints, 2) => ((b << 8) | (b << 3) | (b << 2), 54),
                    (Ise::Trits, 3) => {
                        ((c << 8) | (b << 7) | (c << 3) | (b << 2) | (c << 1) | b, 44)
                    }
                    (Ise::Quints, 3) => ((c << 8) | (b << 7) | (c << 2) | (b << 1) | c, 26),
                    (Ise::Trits, 4) => {
                        ((d << 8) | (c << 7) | (b << 6) | (d << 2) | (c << 1) | b, 22)
                    }
                    (Ise::Quints, 4) => ((d << 8) | (c << 7) | (b << 6) | (d << 1) | c, 13),
                    (Ise::Trits, 5) => {
                        ((e << 8) | (d << 7) | (c << 6) | (b << 5) | (e << 1) | d, 11)
                    }
                    (Ise::Quints, 5) => ((e << 8) | (d << 7) | (c << 6) | (b << 5) | e, 6),
                    _ => ((f << 8) | (e << 7) | (d << 6) | (c << 5) | (b << 4) | f, 5),
                };
                let a = if bit(m, 0) == 1 { 0x1FF } else { 0 };
                let t = ((v >> n) * big_c + big_b) ^ a;
                ((a & 0x80) | (t >> 2)) as u8
            };
            v += 1;
        }
        level += 1;
    }
    table
};

/// Weight unquantization to 0..=64, indexed by level (ranges 2 through 32) and encoded value.
static WEIGHT_UNQUANT: [[u8; 32]; 12] = {
    let mut table = [[0u8; 32]; 12];
    let mut level = 0;
    while level < 12 {
        let (n, kind) = ISE_LEVELS[level];
        let mut v = 0;
        while v < ise_range(level) {
            let m = v & ((1 << n) - 1);
            let d = v >> n;
            let (b, c) = (bit(m, 1), bit(m, 2));
            let unquant = match (kind, n) {
                (Ise::Bits, _) => replicate(v, n, 6),
                (Ise::Trits, 0) => [0, 32, 63][d as usize],
                (Ise::Quints, 0) => [0, 16, 32, 47, 63][d as usize],
                _ => {
                    let (big_b, big_c) = match (kind, n) {
                        (Ise::Trits, 1) => (0, 50),
                        (Ise::Quints, 1) => (0, 28),
                        (Ise::Trits, 2) => ((b << 6) | (b << 2) | b, 23),
                        (Ise::Quints, 2) => ((b << 6) | (b << 1), 13),
                        _ => ((c << 6) | (b << 5) | (c << 1) | b, 11),
                    };
                    let a = if bit(m, 0) == 1 { 0x7F } else { 0 };
                    let t = (d * big_c + big_b) ^ a;
                    (a & 0x20) | (t >> 2)
                }
            };
            table[level][v as usize] = (if unquant > 32 { unquant + 1 } else { unquant }) as u8;
            v += 1;
        }
        level += 1;
    }
    table
};

const fn replicate(value: u32, from: u32, to: u32) -> u32 {
    let mut out = 0;
    let mut filled = 0;
    while filled < to {
        out = (out << from) | value;
        filled += from;
    }
    out >> (filled - to)
}

/// Reads `count` bits (at most 8) starting at bit `pos`; bits past the block read as zero.
fn read(block: u128, pos: u32, count: u32) -> u32 {
    if pos >= 128 {
        return 0;
    }
    ((block >> pos) as u32) & ((1 << count) - 1)
}

/// Decodes `out.len()` integers of a bounded integer sequence starting at bit `pos`. Bits past
/// the sequence read as zero, which is how a final partial trit or quint block decodes.
fn decode_ise(block: u128, pos: u32, level: usize, out: &mut [u8]) {
    let (n, kind) = ISE_LEVELS[level];
    let len = ise_bit_count(out.len() as u32, level);
    let mut src = block.checked_shr(pos).unwrap_or(0) & (u128::MAX >> (128 - len));
    let mask = (1u32 << n) - 1;
    match kind {
        Ise::Bits => {
            for v in out {
                *v = (src as u32 & mask) as u8;
                src >>= n;
            }
        }
        Ise::Trits => {
            for group in out.chunks_mut(5) {
                let bits = src as u64;
                let field = |shift: u32, width: u32| (bits >> shift) as u32 & ((1 << width) - 1);
                let packed = field(n, 2)
                    | field(2 * n + 2, 2) << 2
                    | field(3 * n + 4, 1) << 4
                    | field(4 * n + 5, 2) << 5
                    | field(5 * n + 7, 1) << 7;
                let low = [0, n + 2, 2 * n + 4, 3 * n + 5, 4 * n + 7];
                for ((v, &t), shift) in group.iter_mut().zip(&TRITS[packed as usize]).zip(low) {
                    *v = (t << n) | field(shift, n) as u8;
                }
                src >>= 5 * n + 8;
            }
        }
        Ise::Quints => {
            for group in out.chunks_mut(3) {
                let bits = src as u64;
                let field = |shift: u32, width: u32| (bits >> shift) as u32 & ((1 << width) - 1);
                let packed = field(n, 3) | field(2 * n + 3, 2) << 3 | field(3 * n + 5, 2) << 5;
                let low = [0, n + 3, 2 * n + 5];
                for ((v, &q), shift) in group.iter_mut().zip(&QUINTS[packed as usize]).zip(low) {
                    *v = (q << n) | field(shift, n) as u8;
                }
                src >>= 3 * n + 7;
            }
        }
    }
}

/// Weight grid layout of a block mode.
struct BlockMode {
    grid_w: usize,
    grid_h: usize,
    dual_plane: bool,
    level: usize,
    weight_bits: u32,
}

fn decode_block_mode(mode: u32) -> Option<BlockMode> {
    let a = bits(mode, 5, 2) as usize;
    let mut dual_plane = bit(mode, 10) == 1;
    let mut high_precision = bit(mode, 9) == 1;
    let low = bits(mode, 0, 2);
    let range = if low != 0 { low } else { bits(mode, 2, 2) };
    if range == 0 {
        return None;
    }
    let (grid_w, grid_h) = if low != 0 {
        let b = bits(mode, 7, 2) as usize;
        match bits(mode, 2, 2) {
            0 => (b + 4, a + 2),
            1 => (b + 8, a + 2),
            2 => (a + 2, b + 8),
            _ if bit(mode, 8) == 1 => ((b & 1) + 2, a + 2),
            _ => (a + 2, (b & 1) + 6),
        }
    } else {
        match bits(mode, 7, 2) {
            0 => (12, a + 2),
            1 => (a + 2, 12),
            2 => {
                dual_plane = false;
                high_precision = false;
                (a + 6, bits(mode, 9, 2) as usize + 6)
            }
            _ => match a {
                0 => (6, 10),
                1 => (10, 6),
                _ => return None,
            },
        }
    };
    let level = ((range << 1) | bit(mode, 4)) as usize - 2 + if high_precision { 6 } else { 0 };
    let count = grid_w * grid_h * (1 + dual_plane as usize);
    let weight_bits = ise_bit_count(count as u32, level);
    (count <= MAX_WEIGHTS && (24..=96).contains(&weight_bits)).then_some(BlockMode {
        grid_w,
        grid_h,
        dual_plane,
        level,
        weight_bits,
    })
}

fn decode_block<P: Profile>(block: &[u8; 16], block_w: u32, block_h: u32, out: &mut [P::Texel]) {
    check_footprint(block_w, block_h);
    let (bw, bh) = (block_w as usize, block_h as usize);
    let out = &mut out[..bw * bh];
    if decode_texels::<P>(block, bw, bh, out).is_none() {
        out.fill(P::ERROR);
    }
}

/// Decodes a block into `out`, or returns `None` for an illegal encoding.
fn decode_texels<P: Profile>(
    block: &[u8; 16],
    bw: usize,
    bh: usize,
    out: &mut [P::Texel],
) -> Option<()> {
    let raw = u128::from_le_bytes(*block);
    let mode = read(raw, 0, 11);

    if mode & 0x1FF == 0x1FC {
        if read(raw, 10, 2) != 0b11 {
            return None;
        }
        let extent = |pos: u32| (raw >> pos) as u32 & 0x1FFF;
        let (s0, s1, t0, t1) = (extent(12), extent(25), extent(38), extent(51));
        let all_ones = s0 & s1 & t0 & t1 == 0x1FFF;
        if !all_ones && (s0 >= s1 || t0 >= t1) {
            return None;
        }
        let color = std::array::from_fn(|i| (raw >> (64 + 16 * i)) as u16);
        out.fill(P::void_extent(color, bit(mode, 9) == 1));
        return Some(());
    }

    let bm = decode_block_mode(mode)?;
    if bm.grid_w > bw || bm.grid_h > bh {
        return None;
    }
    let partitions = read(raw, 11, 2) as usize + 1;
    if bm.dual_plane && partitions == 4 {
        return None;
    }

    let mut below_weights = 128 - bm.weight_bits;
    let mut cems = [0u32; 4];
    let config_start = if partitions == 1 {
        cems[0] = read(raw, 13, 4);
        17
    } else {
        let mut encoded = read(raw, 23, 6);
        if encoded & 3 == 0 {
            cems = [encoded >> 2; 4];
        } else {
            let high_bits = 3 * partitions as u32 - 4;
            below_weights -= high_bits;
            encoded |= read(raw, below_weights, high_bits) << 6;
            let base_class = (encoded & 3) - 1;
            for (i, cem) in (0u32..).zip(cems.iter_mut().take(partitions)) {
                let class = base_class + bit(encoded, 2 + i);
                *cem = (class << 2) | bits(encoded, 2 + partitions as u32 + 2 * i, 2);
            }
        }
        29
    };

    let value_count: u32 = cems[..partitions]
        .iter()
        .map(|cem| ((cem >> 2) + 1) * 2)
        .sum();
    if value_count > 18 {
        return None;
    }
    let plane2_component = if bm.dual_plane {
        below_weights -= 2;
        read(raw, below_weights, 2) as usize
    } else {
        4
    };
    let color_bits = below_weights.saturating_sub(config_start);
    let color_level = (QUANT_6..21)
        .rev()
        .find(|&level| ise_bit_count(value_count, level) <= color_bits)?;

    let mut values = [0u8; 18];
    let values = &mut values[..value_count as usize];
    decode_ise(raw, config_start, color_level, values);
    let unquant = &COLOR_UNQUANT[color_level];
    for v in values.iter_mut() {
        *v = unquant[*v as usize];
    }

    let mut endpoints = [Expanded::default(); 4];
    let mut rest = &values[..];
    for (ep, &cem) in endpoints.iter_mut().zip(&cems[..partitions]) {
        let (used, tail) = rest.split_at(((cem >> 2) as usize + 1) * 2);
        *ep = P::expand(unpack_endpoints(cem, used));
        rest = tail;
    }

    let weight_count = bm.grid_w * bm.grid_h * (1 + bm.dual_plane as usize);
    let mut weights = [0u8; MAX_WEIGHTS];
    decode_ise(
        raw.reverse_bits(),
        0,
        bm.level,
        &mut weights[..weight_count],
    );
    let unquant = &WEIGHT_UNQUANT[bm.level];
    for w in &mut weights[..weight_count] {
        *w = unquant[*w as usize];
    }

    let mut plane0 = [0u8; MAX_TEXELS];
    let mut plane1 = [0u8; MAX_TEXELS];
    infill(&weights, bm.dual_plane, 0, &bm, bw, bh, &mut plane0);
    if bm.dual_plane {
        infill(&weights, true, 1, &bm, bw, bh, &mut plane1);
    }

    let mut partition_of = [0u8; MAX_TEXELS];
    if partitions > 1 {
        let selector = PartitionSelector::new(read(raw, 13, 10), partitions, bw * bh);
        for (y, row) in partition_of.chunks_exact_mut(bw).take(bh).enumerate() {
            for (x, partition) in row.iter_mut().enumerate() {
                *partition = selector.select(x, y) as u8;
            }
        }
    }

    let deltas = endpoints.map(|ep| std::array::from_fn::<i32, 4, _>(|c| ep.e1[c] - ep.e0[c]));
    let lerp = |partition: u8, w: [i32; 4]| {
        let partition = partition as usize & 3;
        let (base, delta) = (&endpoints[partition].e0, &deltas[partition]);
        let color = std::array::from_fn(|c| base[c] + ((delta[c] * w[c] + 32) >> 6));
        P::texel(color, endpoints[partition].lns)
    };
    let texels = out.iter_mut().zip(&partition_of);
    if bm.dual_plane {
        for ((texel, &partition), (&w0, &w1)) in texels.zip(plane0.iter().zip(&plane1)) {
            let mut w = [i32::from(w0); 4];
            w[plane2_component & 3] = i32::from(w1);
            *texel = lerp(partition, w);
        }
    } else {
        for ((texel, &partition), &w) in texels.zip(&plane0) {
            *texel = lerp(partition, [i32::from(w); 4]);
        }
    }
    Some(())
}

/// Bilinearly infills one plane of the weight grid to the block's texels.
fn infill(
    weights: &[u8; MAX_WEIGHTS],
    dual_plane: bool,
    plane: usize,
    bm: &BlockMode,
    bw: usize,
    bh: usize,
    out: &mut [u8; MAX_TEXELS],
) {
    let (gw, gh) = (bm.grid_w, bm.grid_h);
    let mut grid = [0u8; MAX_WEIGHTS + 16];
    let plane_weights = weights.iter().skip(plane).step_by(1 + dual_plane as usize);
    for (dst, src) in grid.iter_mut().zip(plane_weights).take(gw * gh) {
        *dst = *src;
    }
    if gw == bw && gh == bh {
        out[..bw * bh].copy_from_slice(&grid[..bw * bh]);
        return;
    }

    let ds = (1024 + bw / 2) / (bw - 1);
    let dt = (1024 + bh / 2) / (bh - 1);
    let mut columns = [(0usize, 0u32); 12];
    for (s, column) in columns.iter_mut().enumerate().take(bw) {
        let gs = (ds * s * (gw - 1) + 32) >> 6;
        *column = (gs >> 4, (gs & 0xF) as u32);
    }
    for (t, row) in out.chunks_exact_mut(bw).take(bh).enumerate() {
        let gt = (dt * t * (gh - 1) + 32) >> 6;
        let (jt, ft) = (gt >> 4, (gt & 0xF) as u32);
        for (texel, &(js, fs)) in row.iter_mut().zip(&columns) {
            let v0 = js + jt * gw;
            let w11 = (fs * ft + 8) >> 4;
            let w10 = ft - w11;
            let w01 = fs - w11;
            let w00 = 16 + w11 - fs - ft;
            let sum = grid[v0] as u32 * w00
                + grid[v0 + 1] as u32 * w01
                + grid[v0 + gw] as u32 * w10
                + grid[v0 + gw + 1] as u32 * w11;
            *texel = ((sum + 8) >> 4) as u8;
        }
    }
}

/// The per-block state of the partition hash, evaluated per texel.
struct PartitionSelector {
    seeds: [u32; 8],
    offsets: [u32; 4],
    partitions: usize,
    shift: u32,
}

impl PartitionSelector {
    fn new(index: u32, partitions: usize, texels: usize) -> Self {
        let seed = index + (partitions as u32 - 1) * 1024;
        let rnum = hash52(seed);
        let (sh1, sh2) = if seed & 1 == 1 {
            (
                if seed & 2 != 0 { 4 } else { 5 },
                if partitions == 3 { 6 } else { 5 },
            )
        } else {
            (
                if partitions == 3 { 6 } else { 5 },
                if seed & 2 != 0 { 4 } else { 5 },
            )
        };
        let seeds = std::array::from_fn(|i| {
            let s = (rnum >> (4 * i)) & 0xF;
            (s * s) >> if i % 2 == 0 { sh1 } else { sh2 }
        });
        Self {
            seeds,
            offsets: [rnum >> 14, rnum >> 10, rnum >> 6, rnum >> 2],
            partitions,
            shift: u32::from(texels < 31),
        }
    }

    fn select(&self, x: usize, y: usize) -> usize {
        let (x, y) = ((x as u32) << self.shift, (y as u32) << self.shift);
        let s = &self.seeds;
        let o = &self.offsets;
        let a = (s[0] * x + s[1] * y + o[0]) & 0x3F;
        let b = (s[2] * x + s[3] * y + o[1]) & 0x3F;
        let c = if self.partitions >= 3 {
            (s[4] * x + s[5] * y + o[2]) & 0x3F
        } else {
            0
        };
        let d = if self.partitions >= 4 {
            (s[6] * x + s[7] * y + o[3]) & 0x3F
        } else {
            0
        };
        if a >= b && a >= c && a >= d {
            0
        } else if b >= c && b >= d {
            1
        } else if c >= d {
            2
        } else {
            3
        }
    }
}

fn hash52(mut p: u32) -> u32 {
    p ^= p >> 15;
    p = p.wrapping_sub(p << 17);
    p = p.wrapping_add(p << 7);
    p = p.wrapping_add(p << 4);
    p ^= p >> 5;
    p = p.wrapping_add(p << 16);
    p ^= p >> 7;
    p ^= p >> 3;
    p ^= p << 6;
    p ^= p >> 17;
    p
}

/// Returns `(signed delta, base)` from the raw `(a, b)` pair of a base+offset mode.
fn bit_transfer_signed(a: i32, b: i32) -> (i32, i32) {
    let b = (b >> 1) | (a & 0x80);
    let a = (a >> 1) & 0x3F;
    (if a & 0x20 != 0 { a - 0x40 } else { a }, b)
}

fn blue_contract(c: [i32; 4]) -> [i32; 4] {
    [(c[0] + c[2]) >> 1, (c[1] + c[2]) >> 1, c[2], c[3]]
}

fn ldr(e0: [i32; 4], e1: [i32; 4]) -> Endpoints {
    Endpoints {
        e0,
        e1,
        rgb_hdr: false,
        alpha_hdr: false,
        default_alpha: false,
    }
}

fn hdr(e0: [i32; 4], e1: [i32; 4], alpha_hdr: bool, default_alpha: bool) -> Endpoints {
    Endpoints {
        e0,
        e1,
        rgb_hdr: true,
        alpha_hdr,
        default_alpha,
    }
}

/// Decodes the unquantized endpoint values of one partition for color endpoint mode `cem`.
fn unpack_endpoints(cem: u32, values: &[u8]) -> Endpoints {
    let v: [i32; 8] = std::array::from_fn(|i| values.get(i).map_or(0, |&x| i32::from(x)));
    match cem {
        0 => ldr([v[0], v[0], v[0], 0xFF], [v[1], v[1], v[1], 0xFF]),
        1 => {
            let l0 = (v[0] >> 2) | (v[1] & 0xC0);
            let l1 = (l0 + (v[1] & 0x3F)).min(0xFF);
            ldr([l0, l0, l0, 0xFF], [l1, l1, l1, 0xFF])
        }
        2 => {
            let (y0, y1) = if v[1] >= v[0] {
                (v[0] << 4, v[1] << 4)
            } else {
                ((v[1] << 4) + 8, (v[0] << 4) - 8)
            };
            hdr(
                [y0 << 4, y0 << 4, y0 << 4, 0],
                [y1 << 4, y1 << 4, y1 << 4, 0],
                false,
                true,
            )
        }
        3 => {
            let (y0, d) = if v[0] & 0x80 != 0 {
                (
                    ((v[1] & 0xE0) << 4) | ((v[0] & 0x7F) << 2),
                    (v[1] & 0x1F) << 2,
                )
            } else {
                (
                    ((v[1] & 0xF0) << 4) | ((v[0] & 0x7F) << 1),
                    (v[1] & 0xF) << 1,
                )
            };
            let y1 = (y0 + d).min(0xFFF);
            hdr(
                [y0 << 4, y0 << 4, y0 << 4, 0],
                [y1 << 4, y1 << 4, y1 << 4, 0],
                false,
                true,
            )
        }
        4 => ldr([v[0], v[0], v[0], v[2]], [v[1], v[1], v[1], v[3]]),
        5 => {
            let (l1, l0) = bit_transfer_signed(v[1], v[0]);
            let (a1, a0) = bit_transfer_signed(v[3], v[2]);
            let l1 = (l0 + l1).clamp(0, 0xFF);
            let a1 = (a0 + a1).clamp(0, 0xFF);
            ldr([l0, l0, l0, a0], [l1, l1, l1, a1])
        }
        6 => {
            let s = v[3];
            ldr(
                [(v[0] * s) >> 8, (v[1] * s) >> 8, (v[2] * s) >> 8, 0xFF],
                [v[0], v[1], v[2], 0xFF],
            )
        }
        7 => {
            let (e0, e1) = hdr_rgbo([v[0], v[1], v[2], v[3]]);
            hdr(e0, e1, false, true)
        }
        8 | 12 => {
            let alpha = if cem == 8 { [0xFF, 0xFF] } else { [v[6], v[7]] };
            let e0 = [v[0], v[2], v[4], alpha[0]];
            let e1 = [v[1], v[3], v[5], alpha[1]];
            if v[1] + v[3] + v[5] >= v[0] + v[2] + v[4] {
                ldr(e0, e1)
            } else {
                ldr(blue_contract(e1), blue_contract(e0))
            }
        }
        9 | 13 => {
            let pairs: [(i32, i32); 4] =
                std::array::from_fn(|i| bit_transfer_signed(v[2 * i + 1], v[2 * i]));
            let mut e0 = pairs.map(|(_, base)| base);
            let mut e1 = pairs.map(|(delta, base)| base + delta);
            if cem == 9 {
                e0[3] = 0xFF;
                e1[3] = 0xFF;
            }
            let delta_sum = pairs[0].0 + pairs[1].0 + pairs[2].0;
            let (e0, e1) = if delta_sum >= 0 {
                (e0, e1)
            } else {
                (blue_contract(e1), blue_contract(e0))
            };
            ldr(e0.map(|c| c.clamp(0, 0xFF)), e1.map(|c| c.clamp(0, 0xFF)))
        }
        10 => {
            let s = v[3];
            ldr(
                [(v[0] * s) >> 8, (v[1] * s) >> 8, (v[2] * s) >> 8, v[4]],
                [v[0], v[1], v[2], v[5]],
            )
        }
        11 => {
            let (e0, e1) = hdr_rgb(&v);
            hdr(e0, e1, false, true)
        }
        14 => {
            let (mut e0, mut e1) = hdr_rgb(&v);
            e0[3] = v[6];
            e1[3] = v[7];
            hdr(e0, e1, false, false)
        }
        _ => {
            let (mut e0, mut e1) = hdr_rgb(&v);
            (e0[3], e1[3]) = hdr_alpha(v[6], v[7]);
            hdr(e0, e1, true, false)
        }
    }
}

fn hdr_rgbo(v: [i32; 4]) -> ([i32; 4], [i32; 4]) {
    let modeval = ((v[0] & 0xC0) >> 6) | (((v[1] & 0x80) >> 7) << 2) | (((v[2] & 0x80) >> 7) << 3);
    let (major, mode) = if modeval & 0xC != 0xC {
        (modeval >> 2, modeval & 3)
    } else if modeval != 0xF {
        (modeval & 3, 4)
    } else {
        (0, 5)
    };
    let mut red = v[0] & 0x3F;
    let mut green = v[1] & 0x1F;
    let mut blue = v[2] & 0x1F;
    let mut scale = v[3] & 0x1F;
    let b = |value: i32, n: u32| (value >> n) & 1;
    let (bit0, bit1, bit2, bit3) = (b(v[1], 6), b(v[1], 5), b(v[2], 6), b(v[2], 5));
    let (bit4, bit5, bit6) = (b(v[3], 7), b(v[3], 6), b(v[3], 5));
    let one_hot = 1 << mode;
    let on = |mask: i32| one_hot & mask != 0;
    if on(0x30) {
        green |= bit0 << 6;
        blue |= bit2 << 6;
    }
    if on(0x3A) {
        green |= bit1 << 5;
        blue |= bit3 << 5;
    }
    if on(0x3D) {
        scale |= bit6 << 5;
    }
    if on(0x2D) {
        scale |= bit5 << 6;
    }
    if on(0x04) {
        scale |= bit4 << 7;
        red |= bit3 << 6;
    }
    if on(0x3B) {
        red |= bit4 << 6;
    }
    if on(0x10) {
        red |= bit5 << 7;
    }
    if on(0x0F) {
        red |= bit2 << 7;
    }
    if on(0x05) {
        red |= (bit1 << 8) | (bit0 << 9);
    }
    if on(0x0A) {
        red |= bit0 << 8;
    }
    if on(0x02) {
        red |= (bit6 << 9) | (bit5 << 10);
    }
    if on(0x01) {
        red |= bit3 << 10;
    }
    let shift = [1, 1, 2, 3, 4, 5][mode as usize];
    red <<= shift;
    green <<= shift;
    blue <<= shift;
    scale <<= shift;
    if mode != 5 {
        green = red - green;
        blue = red - blue;
    }
    match major {
        1 => std::mem::swap(&mut red, &mut green),
        2 => std::mem::swap(&mut red, &mut blue),
        _ => {}
    }
    let e1 = [red, green, blue].map(|c| c.max(0) << 4);
    let e0 = [red, green, blue].map(|c| (c - scale).max(0) << 4);
    ([e0[0], e0[1], e0[2], 0], [e1[0], e1[1], e1[2], 0])
}

fn hdr_rgb(v: &[i32; 8]) -> ([i32; 4], [i32; 4]) {
    let modeval = ((v[1] & 0x80) >> 7) | (((v[2] & 0x80) >> 7) << 1) | (((v[3] & 0x80) >> 7) << 2);
    let major = ((v[4] & 0x80) >> 7) | (((v[5] & 0x80) >> 7) << 1);
    if major == 3 {
        return (
            [v[0] << 8, v[2] << 8, (v[4] & 0x7F) << 9, 0],
            [v[1] << 8, v[3] << 8, (v[5] & 0x7F) << 9, 0],
        );
    }
    let mut a = v[0] | ((v[1] & 0x40) << 2);
    let mut b0 = v[2] & 0x3F;
    let mut b1 = v[3] & 0x3F;
    let mut c = v[1] & 0x3F;
    let mut d0 = v[4] & 0x7F;
    let mut d1 = v[5] & 0x7F;
    let dbits = [7, 6, 7, 6, 5, 6, 5, 6][modeval as usize];
    let b = |value: i32, n: u32| (value >> n) & 1;
    let (bit0, bit1, bit2, bit3) = (b(v[2], 6), b(v[3], 6), b(v[4], 6), b(v[5], 6));
    let (bit4, bit5) = (b(v[4], 5), b(v[5], 5));
    let one_hot = 1 << modeval;
    let on = |mask: i32| one_hot & mask != 0;
    if on(0xA4) {
        a |= bit0 << 9;
    }
    if on(0x8) {
        a |= bit2 << 9;
    }
    if on(0x50) {
        a |= (bit4 << 9) | (bit5 << 10);
    }
    if on(0xA0) {
        a |= bit1 << 10;
    }
    if on(0xC0) {
        a |= bit2 << 11;
    }
    if on(0x4) {
        c |= bit1 << 6;
    }
    if on(0xE8) {
        c |= bit3 << 6;
    }
    if on(0x20) {
        c |= bit2 << 7;
    }
    if on(0x5B) {
        b0 |= bit0 << 6;
        b1 |= bit1 << 6;
    }
    if on(0x12) {
        b0 |= bit2 << 7;
        b1 |= bit3 << 7;
    }
    if on(0xAF) {
        d0 |= bit4 << 5;
        d1 |= bit5 << 5;
    }
    if on(0x5) {
        d0 |= bit2 << 6;
        d1 |= bit3 << 6;
    }
    let sign_shift = 32 - dbits;
    d0 = (d0 << sign_shift) >> sign_shift;
    d1 = (d1 << sign_shift) >> sign_shift;
    let shift = (modeval >> 1) ^ 3;
    let [a, b0, b1, c, d0, d1] = [a, b0, b1, c, d0, d1].map(|x| x << shift);
    let mut e1 = [a, a - b0, a - b1];
    let mut e0 = [a - c, a - b0 - c - d0, a - b1 - c - d1];
    match major {
        1 => {
            e0.swap(0, 1);
            e1.swap(0, 1);
        }
        2 => {
            e0.swap(0, 2);
            e1.swap(0, 2);
        }
        _ => {}
    }
    let e0 = e0.map(|x| x.clamp(0, 0xFFF) << 4);
    let e1 = e1.map(|x| x.clamp(0, 0xFFF) << 4);
    ([e0[0], e0[1], e0[2], 0], [e1[0], e1[1], e1[2], 0])
}

fn hdr_alpha(v6: i32, v7: i32) -> (i32, i32) {
    let modeval = ((v6 >> 7) & 1) | ((v7 >> 6) & 2);
    let (mut v6, mut v7) = (v6 & 0x7F, v7 & 0x7F);
    let (a0, a1) = if modeval == 3 {
        (v6 << 5, v7 << 5)
    } else {
        v6 |= (v7 << (modeval + 1)) & 0x780;
        v7 &= 0x3F >> modeval;
        v7 ^= 32 >> modeval;
        v7 -= 32 >> modeval;
        v6 <<= 4 - modeval;
        v7 <<= 4 - modeval;
        (v6, (v6 + v7).clamp(0, 0xFFF))
    };
    (a0 << 4, a1 << 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unquant_tables_span_their_ranges() {
        for (level, table) in COLOR_UNQUANT.iter().enumerate().skip(QUANT_6) {
            let used = &table[..ise_range(level) as usize];
            assert_eq!(used.iter().min(), Some(&0), "color level {level}");
            assert_eq!(used.iter().max(), Some(&255), "color level {level}");
        }
        for (level, table) in WEIGHT_UNQUANT.iter().enumerate() {
            let used = &table[..ise_range(level) as usize];
            assert_eq!(used.iter().min(), Some(&0), "weight level {level}");
            assert_eq!(used.iter().max(), Some(&64), "weight level {level}");
        }
    }

    #[test]
    fn trit_and_quint_tables_cover_every_digit_combination() {
        let mut seen = [false; 243];
        for t in TRITS {
            assert!(t.iter().all(|&d| d < 3));
            seen[t.iter().rev().fold(0, |acc, &d| acc * 3 + d as usize)] = true;
        }
        assert!(seen.iter().all(|&s| s));
        let mut seen = [false; 125];
        for q in QUINTS {
            assert!(q.iter().all(|&d| d < 5));
            seen[q.iter().rev().fold(0, |acc, &d| acc * 5 + d as usize)] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn unorm16_conversion_truncates_every_value() {
        for value in 0..0xFFFF {
            let x = f64::from(value) / 65536.0;
            let expected = if value < 4 {
                (x / 2f64.powi(-24)) as u16
            } else {
                let exponent = x.log2().floor() as i32;
                let mantissa = ((x / 2f64.powi(exponent) - 1.0) * 1024.0) as u16;
                (((exponent + 15) as u16) << 10) | mantissa
            };
            assert_eq!(unorm16_to_f16(value), expected, "value {value:#06x}");
        }
        assert_eq!(unorm16_to_f16(0xFFFF), 0x3C00);
    }

    #[test]
    fn half_float_conversions() {
        assert_eq!(unorm16_to_f16(0xFFFF), 0x3C00);
        assert_eq!(unorm16_to_f16(0x8000), 0x3800);
        assert_eq!(unorm16_to_f16(1), 0x0100);
        assert_eq!(unorm16_to_f16(0), 0);
        assert_eq!(lns_to_f16(0x7800), 0x3C00);
        assert_eq!(lns_to_f16(0xFFFF), 0x7BFF);
    }
}
