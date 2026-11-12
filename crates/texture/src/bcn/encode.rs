//! Fast BC1 and BC3 encoders for RGBA8 images, used to recompress decoded ASTC so it takes less
//! VRAM. Colour endpoints come from the principal axis of the block's colours, refined once by
//! least squares; uniform blocks use exact single-colour tables. Quality is close to stb_dxt's
//! normal mode, and speed comes first.

use std::sync::LazyLock;

use super::{bc1_palette, bc3_alpha_palette};
use crate::block::{BLOCK_DIM, Extent, ImageSizeError, check_len};

/// Below this alpha a BC1 texel is encoded as transparent.
const ALPHA_THRESHOLD: u8 = 128;

type Color = [f32; 3];

/// A palette value no texel can be nearest to, standing in for the transparent fourth entry.
const UNREACHABLE: i32 = 1 << 12;

/// For each 8-bit value, the endpoint pair (5 or 6 bits) whose two-thirds point decodes closest.
struct SingleColorTable([[u8; 2]; 256]);

impl SingleColorTable {
    fn new(bits: u32) -> Self {
        let max = (1 << bits) - 1;
        let expand = |q: u32| (2 * q * 255 + max) / (2 * max);
        let third = |hi: u32, lo: u32| (2 * (2 * hi + lo) * 255 + 3 * max) / (6 * max);
        let mut table = [[0; 2]; 256];
        for (value, entry) in table.iter_mut().enumerate() {
            let mut best = (u32::MAX, u32::MAX);
            for hi in 0..=max {
                for lo in 0..=max {
                    let error = third(hi, lo).abs_diff(value as u32);
                    let spread = expand(hi).abs_diff(expand(lo));
                    if (error, spread) < best {
                        best = (error, spread);
                        *entry = [hi as u8, lo as u8];
                    }
                }
            }
        }
        Self(table)
    }
}

static SINGLE5: LazyLock<SingleColorTable> = LazyLock::new(|| SingleColorTable::new(5));
static SINGLE6: LazyLock<SingleColorTable> = LazyLock::new(|| SingleColorTable::new(6));

fn quantize565(c: Color) -> u16 {
    let q = |v: f32, max: f32| (v.clamp(0.0, 255.0) * (max / 255.0) + 0.5) as u16;
    q(c[0], 31.0) << 11 | q(c[1], 63.0) << 5 | q(c[2], 31.0)
}

struct Fit {
    c0: u16,
    c1: u16,
    indices: u32,
    error: u32,
}

/// Orders the endpoints for the mode and picks each texel's nearest palette entry. Texels in
/// `transparent` take index 3, which only the three-colour mode decodes as transparent.
fn fit_indices(texels: &[[u8; 4]; 16], transparent: u16, c0: u16, c1: u16) -> Fit {
    let three_color = transparent != 0;
    let (c0, c1) = if three_color == (c0 > c1) {
        (c1, c0)
    } else {
        (c0, c1)
    };
    let four_color = c0 > c1;
    let mut palette = bc1_palette(c0, c1, four_color).map(|e| [e[0], e[1], e[2]].map(i32::from));
    if !four_color {
        palette[3] = [UNREACHABLE; 3];
    }
    let mut indices = 0;
    let mut error = 0;
    for (t, texel) in texels.iter().enumerate() {
        if transparent & (1 << t) != 0 {
            indices |= 3 << (2 * t);
            continue;
        }
        let [r, g, b] = [texel[0], texel[1], texel[2]].map(i32::from);
        let distance = |e: &[i32; 3]| (r - e[0]).pow(2) + (g - e[1]).pow(2) + (b - e[2]).pow(2);
        let (mut best, mut best_index) = (distance(&palette[0]), 0);
        for (i, entry) in palette.iter().enumerate().skip(1) {
            let d = distance(entry);
            if d < best {
                (best, best_index) = (d, i as u32);
            }
        }
        error += best as u32;
        indices |= best_index << (2 * t);
    }
    Fit {
        c0,
        c1,
        indices,
        error,
    }
}

/// Least-squares endpoints for the texels' current palette indices, or `None` when every texel
/// uses one blend.
fn refine(texels: &[[u8; 4]; 16], transparent: u16, fit: &Fit) -> Option<(Color, Color)> {
    let four_color = fit.c0 > fit.c1;
    let (mut aa, mut bb, mut ab) = (0.0f32, 0.0f32, 0.0f32);
    let (mut ax, mut bx) = ([0.0f32; 3], [0.0f32; 3]);
    for (t, texel) in texels.iter().enumerate() {
        if transparent & (1 << t) != 0 {
            continue;
        }
        let a = match ((fit.indices >> (2 * t)) & 3, four_color) {
            (0, _) => 1.0,
            (1, _) => 0.0,
            (2, true) => 2.0 / 3.0,
            (3, true) => 1.0 / 3.0,
            _ => 0.5,
        };
        let b = 1.0 - a;
        aa += a * a;
        bb += b * b;
        ab += a * b;
        for c in 0..3 {
            ax[c] += a * f32::from(texel[c]);
            bx[c] += b * f32::from(texel[c]);
        }
    }
    let det = aa * bb - ab * ab;
    if det.abs() < 1e-3 {
        return None;
    }
    let e0 = std::array::from_fn(|c| (ax[c] * bb - bx[c] * ab) / det);
    let e1 = std::array::from_fn(|c| (bx[c] * aa - ax[c] * ab) / det);
    Some((e0, e1))
}

fn principal_axis(points: &[Color], mean: Color, fallback: Color) -> Color {
    let mut cov = [0.0f32; 6];
    for p in points {
        let d = [p[0] - mean[0], p[1] - mean[1], p[2] - mean[2]];
        cov[0] += d[0] * d[0];
        cov[1] += d[0] * d[1];
        cov[2] += d[0] * d[2];
        cov[3] += d[1] * d[1];
        cov[4] += d[1] * d[2];
        cov[5] += d[2] * d[2];
    }
    let mut axis = fallback;
    for _ in 0..4 {
        let next = [
            cov[0] * axis[0] + cov[1] * axis[1] + cov[2] * axis[2],
            cov[1] * axis[0] + cov[3] * axis[1] + cov[4] * axis[2],
            cov[2] * axis[0] + cov[4] * axis[1] + cov[5] * axis[2],
        ];
        let scale = next.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        if scale < 1e-6 {
            return fallback;
        }
        axis = next.map(|v| v / scale);
    }
    axis
}

fn encode_color(texels: &[[u8; 4]; 16], transparent: u16) -> Fit {
    if transparent == u16::MAX {
        return Fit {
            c0: 0,
            c1: 0,
            indices: u32::MAX,
            error: 0,
        };
    }
    let mut points = [[0.0f32; 3]; 16];
    let mut count = 0;
    let (mut lo, mut hi) = ([255u8; 3], [0u8; 3]);
    for (t, texel) in texels.iter().enumerate() {
        if transparent & (1 << t) != 0 {
            continue;
        }
        for c in 0..3 {
            lo[c] = lo[c].min(texel[c]);
            hi[c] = hi[c].max(texel[c]);
        }
        points[count] = [texel[0], texel[1], texel[2]].map(f32::from);
        count += 1;
    }
    let points = &points[..count];

    if lo == hi {
        if transparent != 0 {
            let c = quantize565(lo.map(f32::from));
            return fit_indices(texels, transparent, c, c);
        }
        let [r, g, b] = lo.map(usize::from);
        let ([r0, r1], [g0, g1], [b0, b1]) = (SINGLE5.0[r], SINGLE6.0[g], SINGLE5.0[b]);
        let pack = |r: u8, g: u8, b: u8| u16::from(r) << 11 | u16::from(g) << 5 | u16::from(b);
        return fit_indices(texels, transparent, pack(r0, g0, b0), pack(r1, g1, b1));
    }

    let inv = 1.0 / count as f32;
    let mean: Color = std::array::from_fn(|c| points.iter().map(|p| p[c]).sum::<f32>() * inv);
    let axis = principal_axis(
        points,
        mean,
        std::array::from_fn(|c| f32::from(hi[c] - lo[c])),
    );
    let project = |p: &Color| p[0] * axis[0] + p[1] * axis[1] + p[2] * axis[2];
    let (mut min_p, mut max_p) = (points[0], points[0]);
    let (mut min_d, mut max_d) = (project(&points[0]), project(&points[0]));
    for p in &points[1..] {
        let d = project(p);
        if d < min_d {
            (min_d, min_p) = (d, *p);
        }
        if d > max_d {
            (max_d, max_p) = (d, *p);
        }
    }

    let fit = fit_indices(texels, transparent, quantize565(max_p), quantize565(min_p));
    let Some((e0, e1)) = refine(texels, transparent, &fit) else {
        return fit;
    };
    let refined = fit_indices(texels, transparent, quantize565(e0), quantize565(e1));
    if refined.error < fit.error {
        refined
    } else {
        fit
    }
}

fn color_bytes(fit: &Fit) -> [u8; 8] {
    let mut out = [0; 8];
    out[..2].copy_from_slice(&fit.c0.to_le_bytes());
    out[2..4].copy_from_slice(&fit.c1.to_le_bytes());
    out[4..].copy_from_slice(&fit.indices.to_le_bytes());
    out
}

/// Encodes 16 RGBA8 texels (row-major) as a BC1 block. Texels with alpha below 128 become
/// transparent through the three-colour mode; the others ignore alpha.
pub fn encode_bc1_block(texels: &[[u8; 4]; 16]) -> [u8; 8] {
    let transparent = texels.iter().enumerate().fold(0u16, |mask, (t, texel)| {
        mask | u16::from(texel[3] < ALPHA_THRESHOLD) << t
    });
    color_bytes(&encode_color(texels, transparent))
}

fn encode_alpha(texels: &[[u8; 4]; 16]) -> [u8; 8] {
    let (lo, hi) = texels
        .iter()
        .fold((255u8, 0u8), |(lo, hi), t| (lo.min(t[3]), hi.max(t[3])));
    let mut out = [hi, lo, 0, 0, 0, 0, 0, 0];
    if lo == hi {
        return out;
    }
    let palette = bc3_alpha_palette(hi, lo);
    let mut indices = 0u64;
    for (t, texel) in texels.iter().enumerate() {
        let (mut best, mut best_index) = (u8::MAX, 0);
        for (i, &entry) in palette.iter().enumerate() {
            let d = entry.abs_diff(texel[3]);
            if d < best {
                (best, best_index) = (d, i as u64);
            }
        }
        indices |= best_index << (3 * t);
    }
    out[2..].copy_from_slice(&indices.to_le_bytes()[..6]);
    out
}

/// Encodes 16 RGBA8 texels (row-major) as a BC3 block: interpolated alpha and a four-colour block.
pub fn encode_bc3_block(texels: &[[u8; 4]; 16]) -> [u8; 16] {
    let mut out = [0; 16];
    out[..8].copy_from_slice(&encode_alpha(texels));
    out[8..].copy_from_slice(&color_bytes(&encode_color(texels, 0)));
    out
}

fn encode_image<const B: usize>(
    rgba: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    out: &mut [u8],
    encode: impl Fn(&[[u8; 4]; 16]) -> [u8; B],
) -> Result<(), ImageSizeError> {
    let extent = Extent::new(width, height, depth);
    let in_len = extent.texel_bytes(4);
    let out_len = extent.block_bytes(B);
    check_len(rgba.len(), in_len, out.len(), out_len)?;
    if in_len == 0 {
        return Ok(());
    }
    let (w, h) = (extent.width, extent.height);
    let (texels, _) = rgba[..in_len].as_chunks::<4>();
    let (blocks, _) = out[..out_len].as_chunks_mut::<B>();
    for (plane, plane_blocks) in texels
        .chunks_exact(w * h)
        .zip(blocks.chunks_exact_mut(extent.blocks_x * extent.blocks_y))
    {
        for (by, row_blocks) in plane_blocks.chunks_exact_mut(extent.blocks_x).enumerate() {
            let rows: [usize; BLOCK_DIM] =
                std::array::from_fn(|r| (by * BLOCK_DIM + r).min(h - 1) * w);
            for (bx, block) in row_blocks.iter_mut().enumerate() {
                let cols: [usize; BLOCK_DIM] =
                    std::array::from_fn(|c| (bx * BLOCK_DIM + c).min(w - 1));
                let tile =
                    std::array::from_fn(|t| plane[rows[t / BLOCK_DIM] + cols[t % BLOCK_DIM]]);
                *block = encode(&tile);
            }
        }
    }
    Ok(())
}

/// Encodes a tightly packed RGBA8 image as BC1 into `ceil(width / 4) * ceil(height / 4) * depth`
/// blocks. Edge blocks repeat the last row and column so the padding cannot widen the endpoints.
pub fn encode_bc1(
    rgba: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    out: &mut [u8],
) -> Result<(), ImageSizeError> {
    encode_image(rgba, width, height, depth, out, encode_bc1_block)
}

/// Encodes a tightly packed RGBA8 image as BC3, like [`encode_bc1`].
pub fn encode_bc3(
    rgba: &[u8],
    width: u32,
    height: u32,
    depth: u32,
    out: &mut [u8],
) -> Result<(), ImageSizeError> {
    encode_image(rgba, width, height, depth, out, encode_bc3_block)
}
