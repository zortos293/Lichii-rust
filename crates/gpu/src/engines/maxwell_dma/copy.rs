//! Copy paths of the copy engine: buffer copies, pitch and block-linear surface copies and the
//! component remap. Each is offered to the renderer first; the software path moves data through
//! the GMMU in chunks of at most [`CHUNK`] bytes held in the engine's scratch buffers.

use lichii_common::{DAddr, GpuVAddr};
use lichii_texture::block_linear::{
    GOB_HEIGHT, GOB_SIZE, GobLayout, Region, Tiling, gob_offset, swizzle_region, unswizzle_region,
};

use super::regs::DmaParameters;
use super::remap::Remap;
use crate::backend::{CopySurface, ImageCopy, Rasterizer};
use crate::engines::EngineContext;

pub(super) const CHUNK: usize = 1 << 20;
const MAX_BAND_BYTES: usize = 256 << 20;
const MAX_ROW_BYTES: u32 = 1 << 24;
const MAX_EXTENT: u32 = 1 << 20;
const MAX_BLOCK_HEIGHT_LOG2: u32 = 7;
const MAX_BLOCK_DEPTH_LOG2: u32 = 5;
const GOB_BYTES: u64 = GOB_SIZE as u64;
const PTE_KIND_PITCH: u8 = 0x00;
const PTE_KIND_PITCH_NO_SWIZZLE: u8 = 0xfd;

/// Buffers the software paths reuse from copy to copy.
#[derive(Default)]
pub(super) struct Scratch {
    pub(super) packed: Vec<u8>,
    pub(super) converted: Vec<u8>,
    pub(super) tiled: Vec<u8>,
}

/// How one side of a copy lies in memory.
#[derive(Clone, Copy)]
pub(super) enum Layout {
    /// Rows this many bytes apart (negative walks backwards).
    Pitch(i32),
    Block(DmaParameters),
}

/// One side of a copy as the registers describe it.
#[derive(Clone, Copy)]
pub(super) struct Operand {
    pub address: GpuVAddr,
    pub layout: Layout,
}

#[derive(Clone, Copy)]
enum Shape {
    Pitch(i64),
    Block {
        tiling: Tiling,
        x: u32,
        y: u32,
        z: u32,
    },
}

/// A copy side resolved to elements: bytes without a remap, the remap's elements with one.
#[derive(Clone, Copy)]
struct Side {
    address: GpuVAddr,
    bytes_per_element: u32,
    shape: Shape,
}

/// The part of a tiled surface a group of rows touches.
struct Band {
    base: u64,
    tiling: Tiling,
    region: Region,
}

fn for_each_device_range(
    ctx: &mut EngineContext<'_>,
    address: GpuVAddr,
    size: u64,
    mut f: impl FnMut(&mut dyn Rasterizer, DAddr, u64),
) {
    let rasterizer = &mut *ctx.rasterizer;
    ctx.memory.for_each_range(address, size, |range| {
        if let crate::gmmu::RangeTarget::Device(daddr) = range.target {
            f(rasterizer, daddr, range.size);
        }
    });
}

fn flush(ctx: &mut EngineContext<'_>, address: GpuVAddr, size: u64) {
    for_each_device_range(ctx, address, size, |r, daddr, len| {
        r.flush_region(daddr, len)
    });
}

fn invalidate(ctx: &mut EngineContext<'_>, address: GpuVAddr, size: u64) {
    for_each_device_range(ctx, address, size, |r, daddr, len| {
        r.invalidate_region(daddr, len);
    });
}

fn read_memory(ctx: &EngineContext<'_>, address: GpuVAddr, out: &mut [u8]) {
    if let Err(fault) = ctx.memory.read_block(address, out) {
        tracing::warn!(?fault, "copy engine read from unmapped GPU memory");
    }
}

fn write_memory(ctx: &EngineContext<'_>, address: GpuVAddr, data: &[u8]) {
    if let Err(fault) = ctx.memory.write_block(address, data) {
        tracing::warn!(?fault, "copy engine write to unmapped GPU memory");
    }
}

fn pitch_span(address: GpuVAddr, pitch: i64, rows: u32, row_bytes: usize) -> (GpuVAddr, u64) {
    let last = address.wrapping_add_signed((i64::from(rows) - 1) * pitch);
    (
        address.min(last),
        pitch.unsigned_abs() * (u64::from(rows) - 1) + row_bytes as u64,
    )
}

impl Side {
    fn new(
        operand: Operand,
        bytes_per_element: u32,
        gob_layout: GobLayout,
        width: u32,
    ) -> Option<Self> {
        let shape = match operand.layout {
            Layout::Pitch(pitch) => Shape::Pitch(pitch.into()),
            Layout::Block(params) => {
                let block = params.block_size;
                let depth = params.depth.max(1);
                let (x, y) = (params.origin.x(), params.origin.y());
                let sane = block.width() == 0
                    && block.height() <= MAX_BLOCK_HEIGHT_LOG2
                    && block.depth() <= MAX_BLOCK_DEPTH_LOG2
                    && params.height <= MAX_EXTENT
                    && depth <= MAX_EXTENT
                    && params
                        .width
                        .checked_mul(bytes_per_element)
                        .is_some_and(|row| row <= MAX_ROW_BYTES);
                if !sane
                    || u64::from(x) + u64::from(width) > u64::from(params.width)
                    || y >= params.height
                {
                    tracing::warn!(?params, width, "copy engine block-linear surface rejected");
                    return None;
                }
                Shape::Block {
                    tiling: Tiling::new(
                        params.width,
                        params.height,
                        depth,
                        bytes_per_element,
                        block.height(),
                        block.depth(),
                        gob_layout,
                    ),
                    x,
                    y,
                    z: params.layer.min(depth - 1),
                }
            }
        };
        Some(Self {
            address: operand.address,
            bytes_per_element,
            shape,
        })
    }

    fn rows_available(&self) -> u32 {
        match self.shape {
            Shape::Pitch(_) => u32::MAX,
            Shape::Block { tiling, y, .. } => tiling.height - y,
        }
    }

    fn advance(&mut self, rows: u32) {
        match &mut self.shape {
            Shape::Pitch(pitch) => {
                self.address = self.address.wrapping_add_signed(i64::from(rows) * *pitch);
            }
            Shape::Block { y, .. } => *y += rows,
        }
    }

    fn band(tiling: Tiling, x: u32, y: u32, z: u32, width: u32, rows: u32) -> Option<Band> {
        let rows_per_block = GOB_HEIGHT << tiling.block_height_log2;
        let first = y / rows_per_block * rows_per_block;
        let last = (y + rows).div_ceil(rows_per_block) * rows_per_block;
        let slab = 1 << tiling.block_depth_log2;
        let first_z = z / slab * slab;
        let band = Tiling {
            height: last - first,
            depth: slab,
            ..tiling
        };
        if band.size_bytes() > MAX_BAND_BYTES {
            tracing::warn!(?tiling, "copy engine block-linear band too large");
            return None;
        }
        Some(Band {
            base: tiling.offset(0, first, first_z) as u64,
            tiling: band,
            region: Region {
                x,
                y: y - first,
                z: z - first_z,
                width,
                height: rows,
                depth: 1,
            },
        })
    }

    fn read(
        &self,
        ctx: &mut EngineContext<'_>,
        tiled: &mut Vec<u8>,
        out: &mut [u8],
        rows: u32,
        width: u32,
    ) -> bool {
        let row_bytes = width as usize * self.bytes_per_element as usize;
        match self.shape {
            Shape::Pitch(pitch) => {
                let (start, span) = pitch_span(self.address, pitch, rows, row_bytes);
                flush(ctx, start, span);
                if pitch == row_bytes as i64 {
                    read_memory(ctx, self.address, &mut out[..row_bytes * rows as usize]);
                    return true;
                }
                for (row, line) in out
                    .chunks_exact_mut(row_bytes)
                    .take(rows as usize)
                    .enumerate()
                {
                    let address = self.address.wrapping_add_signed(row as i64 * pitch);
                    read_memory(ctx, address, line);
                }
            }
            Shape::Block { tiling, x, y, z } => {
                let Some(band) = Self::band(tiling, x, y, z, width, rows) else {
                    return false;
                };
                tiled.resize(band.tiling.size_bytes(), 0);
                let target = self.address + band.base;
                flush(ctx, target, tiled.len() as u64);
                read_memory(ctx, target, tiled);
                unswizzle_region(&band.tiling, band.region, tiled, out, None);
            }
        }
        true
    }

    fn write(
        &self,
        ctx: &mut EngineContext<'_>,
        tiled: &mut Vec<u8>,
        data: &[u8],
        rows: u32,
        width: u32,
    ) {
        let row_bytes = width as usize * self.bytes_per_element as usize;
        match self.shape {
            Shape::Pitch(pitch) => {
                if pitch == row_bytes as i64 {
                    write_memory(ctx, self.address, &data[..row_bytes * rows as usize]);
                } else {
                    for (row, line) in data.chunks_exact(row_bytes).take(rows as usize).enumerate()
                    {
                        let address = self.address.wrapping_add_signed(row as i64 * pitch);
                        write_memory(ctx, address, line);
                    }
                }
                let (start, span) = pitch_span(self.address, pitch, rows, row_bytes);
                invalidate(ctx, start, span);
            }
            Shape::Block { tiling, x, y, z } => {
                let Some(band) = Self::band(tiling, x, y, z, width, rows) else {
                    return;
                };
                tiled.resize(band.tiling.size_bytes(), 0);
                let target = self.address + band.base;
                flush(ctx, target, tiled.len() as u64);
                read_memory(ctx, target, tiled);
                swizzle_region(&band.tiling, band.region, data, tiled, None);
                write_memory(ctx, target, tiled);
                invalidate(ctx, target, tiled.len() as u64);
            }
        }
    }
}

/// Whether a GPU page kind stores bytes in plain pitch order. Unmapped memory counts as pitch.
fn is_pitch_kind(kind: Option<u8>) -> bool {
    matches!(
        kind,
        None | Some(PTE_KIND_PITCH | PTE_KIND_PITCH_NO_SWIZZLE)
    )
}

/// Address inside a GOB-organised region of the byte that sits at linear offset `address`.
fn gob_address(address: u64, layout: GobLayout) -> u64 {
    let x = (address & 63) as u32;
    let y = ((address >> 6) & 7) as u32;
    (address & !(GOB_BYTES - 1)) | u64::from(gob_offset(x, y, layout))
}

/// Reads `out.len()` bytes at `address`. With `gob` the bytes come in GOB raster order out of
/// block-linear memory.
fn read_linear(
    ctx: &mut EngineContext<'_>,
    tiled: &mut Vec<u8>,
    address: GpuVAddr,
    gob: bool,
    out: &mut [u8],
) {
    let len = out.len() as u64;
    if !gob {
        flush(ctx, address, len);
        read_memory(ctx, address, out);
        return;
    }
    let start = address & !(GOB_BYTES - 1);
    let span = (address + len).next_multiple_of(GOB_BYTES) - start;
    tiled.resize(span as usize, 0);
    flush(ctx, start, span);
    read_memory(ctx, start, tiled);
    let mut done = 0;
    while done < out.len() {
        let at = address + done as u64;
        let run = ((16 - (at & 15)) as usize).min(out.len() - done);
        let from = (gob_address(at, ctx.gob_layout) - start) as usize;
        out[done..done + run].copy_from_slice(&tiled[from..from + run]);
        done += run;
    }
}

fn write_linear(
    ctx: &mut EngineContext<'_>,
    tiled: &mut Vec<u8>,
    address: GpuVAddr,
    gob: bool,
    data: &[u8],
) {
    let len = data.len() as u64;
    if !gob {
        write_memory(ctx, address, data);
        invalidate(ctx, address, len);
        return;
    }
    let start = address & !(GOB_BYTES - 1);
    let span = (address + len).next_multiple_of(GOB_BYTES) - start;
    tiled.resize(span as usize, 0);
    if span != len {
        flush(ctx, start, span);
        read_memory(ctx, start, tiled);
    }
    let mut done = 0;
    while done < data.len() {
        let at = address + done as u64;
        let run = ((16 - (at & 15)) as usize).min(data.len() - done);
        let to = (gob_address(at, ctx.gob_layout) - start) as usize;
        tiled[to..to + run].copy_from_slice(&data[done..done + run]);
        done += run;
    }
    write_memory(ctx, start, tiled);
    invalidate(ctx, start, span);
}

/// Copies `elements` elements between two flat ranges, through the remap when there is one.
/// `src_gob` and `dst_gob` mark ranges in block-linear memory that a flat copy addresses in GOB
/// raster order.
pub(super) fn copy_linear(
    ctx: &mut EngineContext<'_>,
    scratch: &mut Scratch,
    src: GpuVAddr,
    dst: GpuVAddr,
    elements: u64,
    remap: Option<Remap>,
) {
    let src_gob = !is_pitch_kind(ctx.memory.kind(src));
    let dst_gob = !is_pitch_kind(ctx.memory.kind(dst));
    copy_elements(
        ctx,
        scratch,
        (src, src_gob),
        (dst, dst_gob),
        elements,
        remap,
    );
}

fn copy_elements(
    ctx: &mut EngineContext<'_>,
    scratch: &mut Scratch,
    (src, src_gob): (GpuVAddr, bool),
    (dst, dst_gob): (GpuVAddr, bool),
    elements: u64,
    remap: Option<Remap>,
) {
    let Scratch {
        packed,
        converted,
        tiled,
    } = scratch;
    let (src_bytes, dst_bytes) = remap.map_or((1, 1), |r| {
        (u64::from(r.src_bytes()), u64::from(r.dst_bytes()))
    });
    if remap.is_none() && !src_gob && !dst_gob {
        let size = elements;
        if ctx
            .rasterizer
            .accelerate_buffer_copy(ctx.channel, ctx.memory, src, dst, size)
        {
            return;
        }
    }
    if let Some(value) = remap.and_then(|remap| remap.fill_value())
        && !dst_gob
        && ctx.rasterizer.accelerate_buffer_clear(
            ctx.channel,
            ctx.memory,
            dst,
            elements * dst_bytes,
            value,
        )
    {
        return;
    }
    let step = (CHUNK as u64 / src_bytes.max(dst_bytes)).max(1);
    let chunks = elements.div_ceil(step);
    let backwards = remap.is_none() && dst > src && dst - src < elements;
    for index in 0..chunks {
        let chunk = if backwards { chunks - 1 - index } else { index };
        let first = chunk * step;
        let count = step.min(elements - first);
        let (from, to) = (
            src.wrapping_add(first * src_bytes),
            dst.wrapping_add(first * dst_bytes),
        );
        packed.resize((count * src_bytes) as usize, 0);
        if remap.is_none_or(|r| r.reads_source()) {
            read_linear(ctx, tiled, from, src_gob, packed);
        }
        let data: &[u8] = match remap {
            None => packed,
            Some(remap) => {
                converted.resize((count * dst_bytes) as usize, 0);
                if remap.keeps_destination() {
                    read_linear(ctx, tiled, to, dst_gob, converted);
                }
                remap.apply(packed, converted, count as usize);
                converted
            }
        };
        write_linear(ctx, tiled, to, dst_gob, data);
    }
}

fn copy_surface(
    operand: Operand,
    shift: u32,
    bytes_per_pixel: u32,
    lines: u32,
    line_length: u32,
) -> (CopySurface, (u32, u32)) {
    match operand.layout {
        Layout::Pitch(pitch) => (
            CopySurface {
                address: operand.address,
                width: line_length >> shift,
                height: lines,
                depth: 1,
                layer: 0,
                pitch: pitch.unsigned_abs(),
                block: None,
                bytes_per_pixel,
            },
            (0, 0),
        ),
        Layout::Block(params) => (
            CopySurface {
                address: operand.address,
                width: params.width >> shift,
                height: params.height,
                depth: params.depth.max(1),
                layer: params.layer,
                pitch: 0,
                block: Some((
                    params.block_size.width(),
                    params.block_size.height(),
                    params.block_size.depth(),
                )),
                bytes_per_pixel,
            },
            (params.origin.x() >> shift, params.origin.y()),
        ),
    }
}

fn image_copy(
    src: Operand,
    dst: Operand,
    remap: Option<Remap>,
    line_length: u32,
    lines: u32,
) -> ImageCopy {
    let (shift, bytes_per_pixel) = match remap {
        Some(remap) => (0, remap.dst_bytes()),
        None => {
            let mut shift = 4
                .min(line_length.trailing_zeros())
                .min((src.address as u32).trailing_zeros())
                .min((dst.address as u32).trailing_zeros());
            for operand in [src, dst] {
                shift = shift.min(match operand.layout {
                    Layout::Pitch(pitch) => (pitch as u32).trailing_zeros(),
                    Layout::Block(params) => params
                        .width
                        .trailing_zeros()
                        .min(params.origin.x().trailing_zeros()),
                });
            }
            (shift, 1 << shift)
        }
    };
    let (src_surface, src_origin) = copy_surface(src, shift, bytes_per_pixel, lines, line_length);
    let (dst_surface, dst_origin) = copy_surface(dst, shift, bytes_per_pixel, lines, line_length);
    ImageCopy {
        src: src_surface,
        dst: dst_surface,
        src_origin,
        dst_origin,
        width: line_length >> shift,
        height: lines,
    }
}

/// A multi-line copy of `lines` lines of `line_length` elements between two surfaces.
pub(super) fn copy_lines(
    ctx: &mut EngineContext<'_>,
    scratch: &mut Scratch,
    src: Operand,
    dst: Operand,
    remap: Option<Remap>,
    line_length: u32,
    lines: u32,
) {
    if line_length == 0 || lines == 0 {
        return;
    }
    if let (Layout::Pitch(src_pitch), Layout::Pitch(dst_pitch)) = (src.layout, dst.layout)
        && remap.is_none()
    {
        copy_pitch_lines(
            ctx,
            scratch,
            src,
            dst,
            (src_pitch, dst_pitch),
            line_length,
            lines,
        );
        return;
    }
    let negative_pitch = [src, dst]
        .iter()
        .any(|operand| matches!(operand.layout, Layout::Pitch(pitch) if pitch < 0));
    if !negative_pitch && remap.is_none_or(|r| r.is_identity()) {
        let copy = image_copy(src, dst, remap, line_length, lines);
        if ctx
            .rasterizer
            .accelerate_image_copy(ctx.channel, ctx.memory, &copy)
        {
            return;
        }
    }
    let (src_bytes, dst_bytes) = remap.map_or((1, 1), |r| (r.src_bytes(), r.dst_bytes()));
    let reads_source = remap.is_none_or(|r| r.reads_source());
    let row_bytes = line_length as usize * src_bytes.max(dst_bytes) as usize;
    if let Some(remap) = remap
        && row_bytes > CHUNK
        && let Layout::Pitch(dst_pitch) = dst.layout
        && (!reads_source || matches!(src.layout, Layout::Pitch(_)))
    {
        let src_pitch = match src.layout {
            Layout::Pitch(pitch) => i64::from(pitch),
            Layout::Block(_) => 0,
        };
        for line in 0..i64::from(lines) {
            let from = src.address.wrapping_add_signed(line * src_pitch);
            let to = dst.address.wrapping_add_signed(line * i64::from(dst_pitch));
            copy_elements(
                ctx,
                scratch,
                (from, false),
                (to, false),
                u64::from(line_length),
                Some(remap),
            );
        }
        return;
    }
    let mut src = if reads_source {
        let Some(side) = Side::new(src, src_bytes, ctx.gob_layout, line_length) else {
            return;
        };
        Some(side)
    } else {
        None
    };
    let Some(mut dst) = Side::new(dst, dst_bytes, ctx.gob_layout, line_length) else {
        return;
    };
    let mut left = lines
        .min(src.as_ref().map_or(u32::MAX, Side::rows_available))
        .min(dst.rows_available());
    let group = (CHUNK / row_bytes).max(1) as u32;
    let Scratch {
        packed,
        converted,
        tiled,
    } = scratch;
    while left > 0 {
        let rows = left.min(group);
        packed.resize(rows as usize * line_length as usize * src_bytes as usize, 0);
        if let Some(src) = &src
            && !src.read(ctx, tiled, packed, rows, line_length)
        {
            return;
        }
        let data: &[u8] = match remap {
            None => packed,
            Some(remap) => {
                converted.resize(rows as usize * line_length as usize * dst_bytes as usize, 0);
                if remap.keeps_destination() && !dst.read(ctx, tiled, converted, rows, line_length)
                {
                    return;
                }
                remap.apply(packed, converted, rows as usize * line_length as usize);
                converted
            }
        };
        dst.write(ctx, tiled, data, rows, line_length);
        if let Some(src) = &mut src {
            src.advance(rows);
        }
        dst.advance(rows);
        left -= rows;
    }
}

fn copy_pitch_lines(
    ctx: &mut EngineContext<'_>,
    scratch: &mut Scratch,
    src: Operand,
    dst: Operand,
    (src_pitch, dst_pitch): (i32, i32),
    line_length: u32,
    lines: u32,
) {
    let length = u64::from(line_length);
    if lines == 1
        || (i64::from(src_pitch) == length as i64 && i64::from(dst_pitch) == length as i64)
    {
        let size = length * u64::from(lines);
        copy_elements(
            ctx,
            scratch,
            (src.address, false),
            (dst.address, false),
            size,
            None,
        );
        return;
    }
    for line in 0..i64::from(lines) {
        let from = src.address.wrapping_add_signed(line * i64::from(src_pitch));
        let to = dst.address.wrapping_add_signed(line * i64::from(dst_pitch));
        copy_elements(ctx, scratch, (from, false), (to, false), length, None);
    }
}
