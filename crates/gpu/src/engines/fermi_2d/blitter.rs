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
/// Scratch buffers that grew past this many bytes are released after the blit.
const KEEP_CAPACITY: usize = 2 << 20;

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

/// One blit: a source rectangle scaled into a destination rectangle.
pub(super) struct Job {
    pub src: CopySurface,
    pub dst: CopySurface,
    pub src_format: RenderTargetFormat,
    pub dst_format: RenderTargetFormat,
    pub src_rect: Rect,
    pub dst_rect: Rect,
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

/// The software blitter and its scratch buffers.
#[derive(Default)]
pub(super) struct SoftwareBlitter {
    src: Vec<u8>,
    dst: Vec<u8>,
    tiled: Vec<u8>,
    ir_src: Vec<f32>,
    ir_dst: Vec<f32>,
}

impl SoftwareBlitter {
    /// Performs `job` in software and reports the written range to the renderer. Blits it
    /// cannot represent (empty or flipped rectangles, rectangles outside a block-linear
    /// surface, oversized rectangles) are dropped with a warning.
    pub fn blit(&mut self, ctx: &mut EngineContext<'_>, job: &Job) {
        let (Some(src_layout), Some(dst_layout)) =
            (Layout::of(job.src_format), Layout::of(job.dst_format))
        else {
            return;
        };
        let (Some((src_w, src_h)), Some((dst_w, dst_h))) =
            (job.src_rect.size(), job.dst_rect.size())
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
            Placement::new(&src, &job.src_rect, ctx.gob_layout),
            Placement::new(&dst, &job.dst_rect, ctx.gob_layout),
        ) else {
            tracing::warn!(?job.src_rect, ?job.dst_rect, "2D blit rectangle outside its surface");
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

        let same_format = job.src_format == job.dst_format;
        if same_format && (src_w, src_h) == (dst_w, dst_h) {
            std::mem::swap(&mut self.src, &mut self.dst);
        } else if same_format && !job.linear_filter {
            nearest(
                &self.src,
                &mut self.dst,
                dst_layout.bytes,
                (src_w, src_h),
                (dst_w, dst_h),
            );
        } else {
            self.ir_src.resize(src_w as usize * src_h as usize * 4, 0.0);
            self.ir_dst.resize(dst_w as usize * dst_h as usize * 4, 0.0);
            src_layout.decode(&self.src, &mut self.ir_src);
            if job.linear_filter {
                bilinear(
                    &self.ir_src,
                    &mut self.ir_dst,
                    (src_w, src_h),
                    (dst_w, dst_h),
                );
            } else {
                nearest(
                    &self.ir_src,
                    &mut self.ir_dst,
                    4,
                    (src_w, src_h),
                    (dst_w, dst_h),
                );
            }
            dst_layout.encode(&self.ir_dst, &mut self.dst);
        }

        dst_place.write(ctx.memory, &mut self.tiled, &self.dst);
        let (address, size) = dst_place.span();
        ctx.rasterizer
            .inline_written(ctx.channel, ctx.memory, address, size);
        self.release_large_buffers();
    }

    fn release_large_buffers(&mut self) {
        for buffer in [&mut self.src, &mut self.dst, &mut self.tiled] {
            if buffer.capacity() > KEEP_CAPACITY {
                *buffer = Vec::new();
            }
        }
        for buffer in [&mut self.ir_src, &mut self.ir_dst] {
            if buffer.capacity() * 4 > KEEP_CAPACITY {
                *buffer = Vec::new();
            }
        }
    }
}

/// Source texel for destination index `index` of a nearest-neighbor scale from `from` to `to`
/// texels: the step is the 32.32 ratio, rounded, and positions truncate.
fn nearest_index(index: u32, step: u64, from: u32) -> usize {
    ((u64::from(index) * step) >> 32).min(u64::from(from) - 1) as usize
}

/// Nearest-neighbor scale of `texel`-element texels (bytes of a pixel, or four channels).
fn nearest<T: Copy>(src: &[T], dst: &mut [T], texel: usize, from: (u32, u32), to: (u32, u32)) {
    let step = |from: u32, to: u32| ((u64::from(from) << 32) + u64::from(to) / 2) / u64::from(to);
    let (step_x, step_y) = (step(from.0, to.0), step(from.1, to.1));
    for (y, row) in dst.chunks_exact_mut(to.0 as usize * texel).enumerate() {
        let src_row = nearest_index(y as u32, step_y, from.1) * from.0 as usize;
        for (x, out) in row.chunks_exact_mut(texel).enumerate() {
            let at = (src_row + nearest_index(x as u32, step_x, from.0)) * texel;
            out.copy_from_slice(&src[at..at + texel]);
        }
    }
}

/// Bilinear scale of four-channel texels with the corner texels of both images aligned.
fn bilinear(src: &[f32], dst: &mut [f32], from: (u32, u32), to: (u32, u32)) {
    let scale = |from: u32, to: u32| {
        if to > 1 {
            (from - 1) as f32 / (to - 1) as f32
        } else {
            0.0
        }
    };
    let (scale_x, scale_y) = (scale(from.0, to.0), scale(from.1, to.1));
    let texel = |x: usize, y: usize| -> &[f32] {
        let at = (y * from.0 as usize + x) * 4;
        &src[at..at + 4]
    };
    for (y, row) in dst.chunks_exact_mut(to.0 as usize * 4).enumerate() {
        let fy = y as f32 * scale_y;
        let (y0, wy) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(from.1 as usize - 1);
        for (x, out) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let fx = x as f32 * scale_x;
            let (x0, wx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(from.0 as usize - 1);
            let (p00, p10, p01, p11) = (texel(x0, y0), texel(x1, y0), texel(x0, y1), texel(x1, y1));
            for c in 0..4 {
                let top = p00[c] + (p10[c] - p00[c]) * wx;
                let bottom = p01[c] + (p11[c] - p01[c]) * wx;
                out[c] = top + (bottom - top) * wy;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_picks_every_other_texel_when_halving() {
        let src: Vec<u8> = (0..16).collect();
        let mut dst = [0u8; 4];
        nearest(&src, &mut dst, 1, (4, 4), (2, 2));
        assert_eq!(dst, [0, 2, 8, 10]);
    }

    #[test]
    fn nearest_repeats_texels_when_doubling() {
        let src = [1u8, 2, 3, 4];
        let mut dst = [0u8; 16];
        nearest(&src, &mut dst, 1, (2, 2), (4, 4));
        assert_eq!(dst, [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
    }

    #[test]
    fn bilinear_interpolates_between_aligned_corners() {
        let src = [0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let mut dst = [0.0f32; 12];
        bilinear(&src, &mut dst, (2, 1), (3, 1));
        assert_eq!(dst[0], 0.0);
        assert_eq!(dst[4], 0.5);
        assert_eq!(dst[8], 1.0);
    }
}
