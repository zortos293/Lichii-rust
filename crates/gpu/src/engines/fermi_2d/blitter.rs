//! The software blit the 2D engine falls back to when the renderer does not accelerate one:
//! read the source rectangle, scale and convert it, write the destination rectangle back.
//!
//! Only the bytes a rectangle covers are read and written: a block-linear surface contributes
//! the rows of blocks the rectangle touches, a pitch-linear one its rows. Scratch buffers live
//! in the blitter, so a blit allocates nothing once they have grown.

use lichii_common::GpuVAddr;
use lichii_texture::block_linear::{
    GOB_HEIGHT, GobLayout, Region, Tiling, swizzle_region, unswizzle_region,
};
use lichii_texture::guest::RenderTargetFormat;

use super::converter::Layout;
use crate::backend::CopySurface;
use crate::engines::EngineContext;
use crate::gmmu::{Gmmu, RangeTarget};

/// Largest source or destination rectangle (and block-linear band) a blit handles, in bytes.
const MAX_BLIT_BYTES: u64 = 256 << 20;
/// Block-linear surfaces larger than this in either direction are refused.
const MAX_DIMENSION: u32 = 1 << 16;

/// A rectangle in surface pixels; the end is exclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    fn size(&self) -> Option<(u32, u32)> {
        let width = self.x1.checked_sub(self.x0).filter(|w| *w > 0)?;
        let height = self.y1.checked_sub(self.y0).filter(|h| *h > 0)?;
        Some((width as u32, height as u32))
    }
}

/// Where the destination pixels sample the source: 32.32 fixed-point start and step per axis, in
/// source pixels relative to the surface origin.
#[derive(Clone, Copy, Debug)]
pub(super) struct Sampling {
    pub x0: i64,
    pub y0: i64,
    pub du_dx: i64,
    pub dv_dy: i64,
}

/// One blit: a source rectangle scaled into a destination rectangle.
pub(super) struct Job {
    pub src: CopySurface,
    pub dst: CopySurface,
    pub src_format: RenderTargetFormat,
    pub dst_format: RenderTargetFormat,
    pub src_rect: Rect,
    pub dst_rect: Rect,
    pub sampling: Sampling,
    pub linear_filter: bool,
}

/// Where a rectangle of a surface lies in guest memory.
enum Placement {
    Pitch {
        first_row: GpuVAddr,
        pitch: u64,
        rows: u32,
        row_bytes: usize,
    },
    Tiled {
        address: GpuVAddr,
        band: Tiling,
        region: Region,
    },
}

impl Placement {
    fn new(surface: &CopySurface, rect: &Rect, gob_layout: GobLayout) -> Option<Self> {
        let (width, height) = rect.size()?;
        let x = u32::try_from(rect.x0).ok()?;
        let y = u32::try_from(rect.y0).ok()?;
        let bytes_per_pixel = surface.bytes_per_pixel;
        let Some((_, block_height, block_depth)) = surface.block else {
            return Some(Self::Pitch {
                first_row: surface.address.wrapping_add(
                    u64::from(y) * u64::from(surface.pitch)
                        + u64::from(x) * u64::from(bytes_per_pixel),
                ),
                pitch: u64::from(surface.pitch),
                rows: height,
                row_bytes: width as usize * bytes_per_pixel as usize,
            });
        };
        let depth = surface.depth.max(1);
        if u64::from(x) + u64::from(width) > u64::from(surface.width)
            || u64::from(y) + u64::from(height) > u64::from(surface.height)
            || surface.width > MAX_DIMENSION
            || surface.height > MAX_DIMENSION
            || depth > MAX_DIMENSION
            || block_height > 5
            || block_depth > 5
        {
            return None;
        }
        let tiling = Tiling::new(
            surface.width,
            surface.height,
            depth,
            bytes_per_pixel,
            block_height,
            block_depth.min(depth.next_power_of_two().trailing_zeros()),
            gob_layout,
        );
        let region = Region {
            x,
            y,
            z: surface.layer.min(depth - 1),
            width,
            height,
            depth: 1,
        };
        let (offset, band, region) = if depth == 1 {
            let rows = GOB_HEIGHT << block_height;
            let first = y / rows * rows;
            let last = (y + height).div_ceil(rows) * rows;
            let band = Tiling {
                height: last - first,
                ..tiling
            };
            let region = Region {
                y: y - first,
                z: 0,
                ..region
            };
            (tiling.offset(0, first, 0) as u64, band, region)
        } else {
            (0, tiling, region)
        };
        (band.size_bytes() as u64 <= MAX_BLIT_BYTES).then_some(Self::Tiled {
            address: surface.address.wrapping_add(offset),
            band,
            region,
        })
    }

    /// The guest bytes the rectangle touches, as an address and a length.
    fn span(&self) -> (GpuVAddr, u64) {
        match self {
            Self::Pitch {
                first_row,
                pitch,
                rows,
                row_bytes,
            } => (
                *first_row,
                (u64::from(*rows) - 1) * pitch + *row_bytes as u64,
            ),
            Self::Tiled { address, band, .. } => (*address, band.size_bytes() as u64),
        }
    }

    /// Reads the rectangle into `linear` (tightly packed rows).
    fn read(&self, memory: &Gmmu, tiled: &mut Vec<u8>, linear: &mut [u8]) {
        match self {
            Self::Pitch {
                first_row,
                pitch,
                row_bytes,
                ..
            } => {
                for (i, row) in linear.chunks_exact_mut(*row_bytes).enumerate() {
                    if let Err(fault) =
                        memory.read_block(first_row.wrapping_add(i as u64 * pitch), row)
                    {
                        tracing::warn!(?fault, "2D blit reads unmapped GPU memory");
                    }
                }
            }
            Self::Tiled {
                address,
                band,
                region,
            } => {
                tiled.resize(band.size_bytes(), 0);
                if let Err(fault) = memory.read_block(*address, tiled) {
                    tracing::warn!(?fault, "2D blit reads unmapped GPU memory");
                }
                unswizzle_region(band, *region, tiled, linear, None);
            }
        }
    }

    /// Writes `linear` (tightly packed rows) into the rectangle; the rest of a block-linear
    /// band keeps its contents.
    fn write(&self, memory: &Gmmu, tiled: &mut Vec<u8>, linear: &[u8]) {
        match self {
            Self::Pitch {
                first_row,
                pitch,
                row_bytes,
                ..
            } => {
                for (i, row) in linear.chunks_exact(*row_bytes).enumerate() {
                    if let Err(fault) =
                        memory.write_block(first_row.wrapping_add(i as u64 * pitch), row)
                    {
                        tracing::warn!(?fault, "2D blit writes unmapped GPU memory");
                    }
                }
            }
            Self::Tiled {
                address,
                band,
                region,
            } => {
                tiled.resize(band.size_bytes(), 0);
                if let Err(fault) = memory.read_block(*address, tiled) {
                    tracing::warn!(?fault, "2D blit reads unmapped GPU memory");
                }
                swizzle_region(band, *region, linear, tiled, None);
                if let Err(fault) = memory.write_block(*address, tiled) {
                    tracing::warn!(?fault, "2D blit writes unmapped GPU memory");
                }
            }
        }
    }
}

/// Bytes a scratch buffer keeps between blits: a 1080p RGBA8 frame's worth. Converting blits
/// decode two source rows and one destination row at a time, so no f32 frame is ever held.
const KEEP_BYTES: usize = 16 << 20;

/// The software blitter and its scratch buffers.
#[derive(Default)]
pub(super) struct SoftwareBlitter {
    src: Vec<u8>,
    dst: Vec<u8>,
    tiled: Vec<u8>,
    /// Decoded source rows and the source row each holds.
    ir_rows: [Vec<f32>; 2],
    ir_row_ids: [usize; 2],
    /// One destination row before it is encoded.
    ir_out: Vec<f32>,
}

impl SoftwareBlitter {
    /// Performs `job` in software and reports the written range to the renderer. Source
    /// positions before the surface origin or past its end sample the edge texel, as the
    /// hardware's clamped sampling does. Blits it cannot represent (empty or flipped rectangles,
    /// rectangles outside a block-linear surface, oversized rectangles) are dropped with a
    /// warning.
    pub fn blit(&mut self, ctx: &mut EngineContext<'_>, job: &Job) {
        let (Some(src_layout), Some(dst_layout)) =
            (Layout::of(job.src_format), Layout::of(job.dst_format))
        else {
            return;
        };
        let unit = 1i64 << 32;
        let whole = |start: i64| start & (unit - 1) == 0;
        let sampling = &job.sampling;
        let filter = job.linear_filter
            && !(sampling.du_dx == unit
                && sampling.dv_dy == unit
                && whole(sampling.x0)
                && whole(sampling.y0));
        let neighbour = i32::from(filter);
        let clamp = |end: i32, extent: u32| match extent {
            0 => end,
            _ => end
                .saturating_add(neighbour)
                .min(extent.min(i32::MAX as u32) as i32),
        };
        let src_rect = Rect {
            x0: job.src_rect.x0.max(0),
            y0: job.src_rect.y0.max(0),
            x1: clamp(job.src_rect.x1, job.src.width),
            y1: clamp(job.src_rect.y1, job.src.height),
        };
        let (Some((src_w, src_h)), Some((dst_w, dst_h))) = (src_rect.size(), job.dst_rect.size())
        else {
            tracing::debug!(?job.src_rect, ?job.dst_rect, "2D blit with an empty or flipped rectangle");
            return;
        };
        let src_bytes = u64::from(src_w) * u64::from(src_h) * src_layout.bytes as u64;
        let dst_bytes = u64::from(dst_w) * u64::from(dst_h) * dst_layout.bytes as u64;
        if src_bytes > MAX_BLIT_BYTES || dst_bytes > MAX_BLIT_BYTES {
            tracing::warn!(src_bytes, dst_bytes, "2D blit too large");
            return;
        }
        let src = CopySurface {
            bytes_per_pixel: src_layout.bytes as u32,
            ..job.src
        };
        let dst = CopySurface {
            bytes_per_pixel: dst_layout.bytes as u32,
            ..job.dst
        };
        let (Some(src_place), Some(dst_place)) = (
            Placement::new(&src, &src_rect, ctx.gob_layout),
            Placement::new(&dst, &job.dst_rect, ctx.gob_layout),
        ) else {
            tracing::warn!(?src_rect, ?job.dst_rect, "2D blit rectangle outside its surface");
            return;
        };

        for place in [&src_place, &dst_place] {
            let (address, size) = place.span();
            ctx.memory.for_each_range(address, size, |range| {
                if let RangeTarget::Device(daddr) = range.target {
                    ctx.rasterizer.flush_region(daddr, range.size);
                }
            });
        }

        self.src.resize(src_bytes as usize, 0);
        self.dst.resize(dst_bytes as usize, 0);
        src_place.read(ctx.memory, &mut self.tiled, &mut self.src);

        let origin = |rect_start: i32| i64::from(rect_start).wrapping_shl(32);
        let sampling = Sampling {
            x0: job.sampling.x0.wrapping_sub(origin(src_rect.x0)),
            y0: job.sampling.y0.wrapping_sub(origin(src_rect.y0)),
            ..job.sampling
        };
        let same_format = job.src_format == job.dst_format;
        let one_to_one = !filter
            && (src_w, src_h) == (dst_w, dst_h)
            && sampling.du_dx == unit
            && sampling.dv_dy == unit;
        if same_format && one_to_one {
            std::mem::swap(&mut self.src, &mut self.dst);
        } else if same_format && !filter {
            nearest(
                &self.src,
                &mut self.dst,
                dst_layout.bytes,
                (src_w, src_h),
                (dst_w, dst_h),
                &sampling,
            );
        } else {
            self.convert(
                (src_layout, dst_layout),
                (src_w, src_h),
                (dst_w, dst_h),
                &sampling,
                filter,
            );
        }

        dst_place.write(ctx.memory, &mut self.tiled, &self.dst);
        let (address, size) = dst_place.span();
        ctx.rasterizer
            .inline_written(ctx.channel, ctx.memory, address, size);
        self.release_oversized();
    }

    /// Converts `self.src` into `self.dst` through f32 texels, one destination row at a time:
    /// each row decodes the (at most two) source rows it samples, reusing the rows the previous
    /// destination row decoded.
    fn convert(
        &mut self,
        (src_layout, dst_layout): (Layout, Layout),
        from: (u32, u32),
        to: (u32, u32),
        sampling: &Sampling,
        filter: bool,
    ) {
        let src_pitch = from.0 as usize * src_layout.bytes;
        let dst_pitch = to.0 as usize * dst_layout.bytes;
        for row in &mut self.ir_rows {
            row.resize(from.0 as usize * 4, 0.0);
        }
        self.ir_row_ids = [usize::MAX; 2];
        self.ir_out.resize(to.0 as usize * 4, 0.0);
        for (y, dst_row) in self.dst.chunks_exact_mut(dst_pitch).enumerate() {
            let (y0, y1, wy) = bilinear_axis(sampling.y0, sampling.dv_dy, y, from.1);
            let pair = [y0, y1];
            let wanted = if filter && y1 != y0 {
                &pair[..]
            } else {
                &pair[..1]
            };
            if self.ir_row_ids[1] == y0 && self.ir_row_ids[0] != y0 {
                self.ir_rows.swap(0, 1);
                self.ir_row_ids.swap(0, 1);
            }
            for (slot, &source) in wanted.iter().enumerate() {
                if self.ir_row_ids[slot] != source {
                    let bytes = &self.src[source * src_pitch..(source + 1) * src_pitch];
                    src_layout.decode(bytes, &mut self.ir_rows[slot]);
                    self.ir_row_ids[slot] = source;
                }
            }
            let top = &self.ir_rows[0];
            let bottom = &self.ir_rows[wanted.len() - 1];
            if filter {
                bilinear_row(top, bottom, wy, &mut self.ir_out, from.0, sampling);
            } else {
                nearest_row(top, &mut self.ir_out, from.0, sampling);
            }
            dst_layout.encode(&self.ir_out, dst_row);
        }
    }

    /// Frees scratch buffers past [`KEEP_BYTES`]: full-frame blits reuse theirs, a rare
    /// oversized blit does not pin its memory for the engine's lifetime.
    fn release_oversized(&mut self) {
        fn release<T>(buffer: &mut Vec<T>) {
            if buffer.capacity() * size_of::<T>() > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
        release(&mut self.src);
        release(&mut self.dst);
        release(&mut self.tiled);
        for row in &mut self.ir_rows {
            release(row);
        }
        release(&mut self.ir_out);
    }
}

/// The source position of destination index `index` along one axis, in 32.32 source texels
/// relative to the start of the read rectangle.
fn position(start: i64, step: i64, index: u32) -> i64 {
    start.wrapping_add(step.wrapping_mul(i64::from(index)))
}

/// Nearest-neighbor scale of `texel`-element texels (bytes of a pixel, or four channels): each
/// destination pixel takes the texel its 32.32 source position falls in, clamped to the source.
fn nearest<T: Copy>(
    src: &[T],
    dst: &mut [T],
    texel: usize,
    from: (u32, u32),
    to: (u32, u32),
    sampling: &Sampling,
) {
    let index = |start: i64, step: i64, i: usize, extent: u32| {
        (position(start, step, i as u32) >> 32).clamp(0, i64::from(extent) - 1) as usize
    };
    for (y, row) in dst.chunks_exact_mut(to.0 as usize * texel).enumerate() {
        let src_row = index(sampling.y0, sampling.dv_dy, y, from.1) * from.0 as usize;
        for (x, out) in row.chunks_exact_mut(texel).enumerate() {
            let at = (src_row + index(sampling.x0, sampling.du_dx, x, from.0)) * texel;
            out.copy_from_slice(&src[at..at + texel]);
        }
    }
}

/// The two source texels around destination index `i`'s 32.32 position along one axis, clamped
/// to the source, and the weight of the second.
fn bilinear_axis(start: i64, step: i64, i: usize, extent: u32) -> (usize, usize, f32) {
    let position = position(start, step, i as u32);
    let last = i64::from(extent) - 1;
    let first = (position >> 32).clamp(0, last);
    let weight = if position < 0 || position >> 32 > last {
        0.0
    } else {
        (position & 0xffff_ffff) as f32 / 4_294_967_296.0
    };
    (first as usize, (first + 1).min(last) as usize, weight)
}

/// One destination row of a bilinear scale of four-channel texels: each pixel blends the four
/// texels around its 32.32 source position from the source rows `top` and `bottom`, weighted by
/// `wy`, with the neighbours clamped to the row.
fn bilinear_row(
    top: &[f32],
    bottom: &[f32],
    wy: f32,
    out: &mut [f32],
    width: u32,
    sampling: &Sampling,
) {
    for (x, out) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (x0, x1, wx) = bilinear_axis(sampling.x0, sampling.du_dx, x, width);
        let (p00, p10) = (&top[x0 * 4..x0 * 4 + 4], &top[x1 * 4..x1 * 4 + 4]);
        let (p01, p11) = (&bottom[x0 * 4..x0 * 4 + 4], &bottom[x1 * 4..x1 * 4 + 4]);
        for c in 0..4 {
            let upper = p00[c] + (p10[c] - p00[c]) * wx;
            let lower = p01[c] + (p11[c] - p01[c]) * wx;
            out[c] = upper + (lower - upper) * wy;
        }
    }
}

/// One destination row of a nearest-neighbor scale of four-channel texels from source row `row`.
fn nearest_row(row: &[f32], out: &mut [f32], width: u32, sampling: &Sampling) {
    for (x, out) in out.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let (at, _, _) = bilinear_axis(sampling.x0, sampling.du_dx, x, width);
        out.copy_from_slice(&row[at * 4..at * 4 + 4]);
    }
}

/// Bilinear scale of whole four-channel frames, row by row.
#[cfg(test)]
fn bilinear(src: &[f32], dst: &mut [f32], from: (u32, u32), to: (u32, u32), sampling: &Sampling) {
    let row = |y: usize| &src[y * from.0 as usize * 4..(y + 1) * from.0 as usize * 4];
    for (y, out) in dst.chunks_exact_mut(to.0 as usize * 4).enumerate() {
        let (y0, y1, wy) = bilinear_axis(sampling.y0, sampling.dv_dy, y, from.1);
        bilinear_row(row(y0), row(y1), wy, out, from.0, sampling);
    }
}

#[cfg(test)]
mod tests {
    use lichii_texture::guest::RenderTargetFormat as F;

    use super::*;
    use crate::engines::testing::Harness;

    const UNIT: i64 = 1 << 32;

    fn sampling(x0: i64, y0: i64, du_dx: i64, dv_dy: i64) -> Sampling {
        Sampling {
            x0,
            y0,
            du_dx,
            dv_dy,
        }
    }

    #[test]
    fn nearest_picks_every_other_texel_when_halving() {
        let src: Vec<u8> = (0..16).collect();
        let mut dst = [0u8; 4];
        nearest(
            &src,
            &mut dst,
            1,
            (4, 4),
            (2, 2),
            &sampling(0, 0, 2 * UNIT, 2 * UNIT),
        );
        assert_eq!(dst, [0, 2, 8, 10]);
    }

    #[test]
    fn nearest_repeats_texels_when_doubling() {
        let src = [1u8, 2, 3, 4];
        let mut dst = [0u8; 16];
        nearest(
            &src,
            &mut dst,
            1,
            (2, 2),
            (4, 4),
            &sampling(0, 0, UNIT / 2, UNIT / 2),
        );
        assert_eq!(dst, [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
    }

    #[test]
    fn nearest_follows_a_fractional_step_instead_of_the_rectangle_ratio() {
        let src = [0u8, 1, 2, 3];
        let mut dst = [0u8; 3];
        nearest(
            &src,
            &mut dst,
            1,
            (4, 1),
            (3, 1),
            &sampling(0, 0, UNIT * 3 / 2, UNIT),
        );
        assert_eq!(dst, [0, 1, 3]);
    }

    #[test]
    fn nearest_honours_the_fractional_start() {
        let src = [0u8, 1, 2, 3];
        let mut dst = [0u8; 2];
        nearest(
            &src,
            &mut dst,
            1,
            (4, 1),
            (2, 1),
            &sampling(UNIT / 2, 0, UNIT, UNIT),
        );
        assert_eq!(dst, [0, 1]);
        nearest(
            &src,
            &mut dst,
            1,
            (4, 1),
            (2, 1),
            &sampling(UNIT * 3 / 4, 0, UNIT / 2, UNIT),
        );
        assert_eq!(dst, [0, 1]);
    }

    #[test]
    fn bilinear_interpolates_along_the_programmed_step() {
        let src = [0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut dst = [0.0f32; 12];
        bilinear(
            &src,
            &mut dst,
            (2, 1),
            (3, 1),
            &sampling(0, 0, UNIT / 2, UNIT),
        );
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[4], 0.5);
        assert_eq!(dst[8], 1.0);
    }

    #[test]
    fn bilinear_clamps_neighbours_to_the_source() {
        let src = [0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut dst = [0.0f32; 8];
        bilinear(
            &src,
            &mut dst,
            (2, 1),
            (2, 1),
            &sampling(UNIT * 3 / 4, 0, UNIT / 2, UNIT),
        );
        assert_eq!(dst[0], 0.75);
        assert_eq!(dst[4], 1.0);
    }

    #[test]
    fn large_blits_keep_their_scratch_buffers() {
        let mut h = Harness::new();
        let (width, height) = (1024, 640);
        let surface = CopySurface {
            address: 0,
            width,
            height,
            depth: 1,
            layer: 0,
            pitch: width * 4,
            block: None,
            bytes_per_pixel: 4,
        };
        let rect = Rect {
            x0: 0,
            y0: 0,
            x1: width as i32,
            y1: height as i32,
        };
        let job = Job {
            src: surface,
            dst: surface,
            src_format: F::A8B8G8R8Unorm,
            dst_format: F::A8B8G8R8Unorm,
            src_rect: rect,
            dst_rect: rect,
            sampling: sampling(0, 0, UNIT, UNIT),
            linear_filter: false,
        };
        let bytes = (width * height * 4) as usize;
        let mut blitter = SoftwareBlitter::default();
        blitter.blit(&mut h.context(), &job);
        assert!(blitter.src.capacity() >= bytes);
        assert!(blitter.dst.capacity() >= bytes);
        assert_eq!(h.rasterizer.inline_writes.len(), 1);
    }

    #[test]
    fn bilinear_one_to_one_copies_keep_32_bit_integers_exact() {
        let mut h = Harness::new();
        let surface = CopySurface {
            address: 0,
            width: 8,
            height: 1,
            depth: 1,
            layer: 0,
            pitch: 32,
            block: None,
            bytes_per_pixel: 4,
        };
        let words: Vec<u32> = (0..8).map(|i| 0x0100_0001 + i).collect();
        h.memory
            .write_block(0, bytemuck::cast_slice(&words))
            .unwrap();
        let rect = |x0, x1| Rect {
            x0,
            y0: 0,
            x1,
            y1: 1,
        };
        let job = Job {
            src: surface,
            dst: CopySurface {
                address: 0x1000,
                ..surface
            },
            src_format: F::R32Uint,
            dst_format: F::R32Uint,
            src_rect: rect(0, 4),
            dst_rect: rect(0, 4),
            sampling: sampling(0, 0, UNIT, UNIT),
            linear_filter: true,
        };
        SoftwareBlitter::default().blit(&mut h.context(), &job);
        let mut out = [0u32; 4];
        h.memory
            .read_block(0x1000, bytemuck::cast_slice_mut(&mut out))
            .unwrap();
        assert_eq!(out, words[..4]);
    }

    #[test]
    fn bilinear_copies_with_a_fractional_start_at_the_edge_blend() {
        let mut h = Harness::new();
        let surface = CopySurface {
            address: 0,
            width: 8,
            height: 1,
            depth: 1,
            layer: 0,
            pitch: 32,
            block: None,
            bytes_per_pixel: 4,
        };
        let texels: Vec<f32> = (0..8).map(|i| i as f32).collect();
        h.memory
            .write_block(0, bytemuck::cast_slice(&texels))
            .unwrap();
        let rect = |x0, x1| Rect {
            x0,
            y0: 0,
            x1,
            y1: 1,
        };
        let job = Job {
            src: surface,
            dst: CopySurface {
                address: 0x1000,
                ..surface
            },
            src_format: F::R32Float,
            dst_format: F::R32Float,
            src_rect: rect(4, 8),
            dst_rect: rect(0, 4),
            sampling: sampling(4 * UNIT + UNIT / 2, 0, UNIT, UNIT),
            linear_filter: true,
        };
        SoftwareBlitter::default().blit(&mut h.context(), &job);
        let mut out = [0f32; 4];
        h.memory
            .read_block(0x1000, bytemuck::cast_slice_mut(&mut out))
            .unwrap();
        assert_eq!(out[..3], [4.5, 5.5, 6.5]);
    }

    #[test]
    fn scratch_keeps_a_1080p_frame() {
        let bytes = 1920 * 1080 * 4;
        let mut blitter = SoftwareBlitter::default();
        blitter.src.reserve_exact(bytes);
        blitter.release_oversized();
        assert!(blitter.src.capacity() >= bytes);
    }

    #[test]
    fn converting_blits_hold_rows_not_frames() {
        let mut h = Harness::new();
        let (width, height) = (512, 256);
        let surface = |address| CopySurface {
            address,
            width,
            height,
            depth: 1,
            layer: 0,
            pitch: width * 4,
            block: None,
            bytes_per_pixel: 4,
        };
        let rect = Rect {
            x0: 0,
            y0: 0,
            x1: width as i32,
            y1: height as i32,
        };
        let pixels: Vec<u32> = (0..width * height)
            .map(|i| i.wrapping_mul(0x0101_0103))
            .collect();
        h.memory
            .write_block(0, bytemuck::cast_slice(&pixels))
            .unwrap();
        let job = Job {
            src: surface(0),
            dst: surface(0x10_0000),
            src_format: F::A8B8G8R8Unorm,
            dst_format: F::A8R8G8B8Unorm,
            src_rect: rect,
            dst_rect: rect,
            sampling: sampling(0, 0, UNIT, UNIT),
            linear_filter: false,
        };
        let mut blitter = SoftwareBlitter::default();
        blitter.blit(&mut h.context(), &job);
        let mut out = vec![0u32; pixels.len()];
        h.memory
            .read_block(0x10_0000, bytemuck::cast_slice_mut(&mut out))
            .unwrap();
        let swapped = |p: u32| (p & 0xff00_ff00) | (p & 0xff) << 16 | (p >> 16) & 0xff;
        assert!(out.iter().zip(&pixels).all(|(&o, &p)| o == swapped(p)));
        let row_floats = width as usize * 4;
        assert!(
            blitter
                .ir_rows
                .iter()
                .all(|row| row.capacity() < 2 * row_floats)
        );
        assert!(blitter.ir_out.capacity() < 2 * row_floats);
        assert_eq!(
            blitter.ir_row_ids[1],
            usize::MAX,
            "a nearest conversion decodes each source row once"
        );
    }

    #[test]
    fn oversized_scratch_is_released_after_the_blit() {
        let mut blitter = SoftwareBlitter::default();
        blitter.ir_out.reserve(KEEP_BYTES / 4 + 1);
        blitter.src.reserve(1024);
        blitter.release_oversized();
        assert_eq!(blitter.ir_out.capacity(), 0);
        assert!(blitter.src.capacity() >= 1024);
    }
}
