//! The VIC engine: composes up to eight video slots into one output surface. A slot's picture
//! comes from the NVDEC channel that decoded into the slot's surface ([`DecodedFrames`]), or from
//! the NV12 surface in guest memory when no decoded picture is kept for it.
//!
//! Each output row is built over row buffers that stay in cache: the source row is gathered
//! (scaled nearest-neighbour from the source to the destination rectangle, chroma upsampled),
//! converted through the slot's colour matrix in the hardware's 10-bit fixed point, and stored
//! into the output plane in its final layout, tiled for block-linear output. The finished surface
//! is written to guest memory once and reported to the process's write tracking, so GPU caches
//! holding it reload it as they do after a CPU write.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_common::{PAGE_MASK, PAGE_SIZE};
use lichii_memory::{AddressSpace, DeviceAddressSpace};
use lichii_texture::block_linear::{
    GOB_SIZE, GOB_WIDTH, GobLayout, Region, Tiling, gob_offset, swizzle_region, unswizzle_region,
};
use lichii_video::{Picture, PixelFormat, Plane};

use super::frames::DecodedFrames;

const REGISTER_COUNT: usize = 0x446;
pub(super) const EXECUTE: u32 = 0xc0;
/// `surfaces[slot][surface][plane]`: the luma, chroma U and chroma V offsets of the eight surfaces
/// (current, previous, next, ...) of each slot.
const SURFACES: usize = 0x100;
const SURFACE_WORDS_PER_SLOT: usize = 8 * 3;
const CONFIG_STRUCT_OFFSET: usize = 0x1c2;
const OUTPUT_LUMA_OFFSET: usize = 0x1c8;
const OUTPUT_CHROMA_OFFSET: usize = 0x1c9;
const SLOT_COUNT: usize = 8;

const FORMAT_A8B8G8R8: u64 = 31;
const FORMAT_A8R8G8B8: u64 = 32;
const FORMAT_X8B8G8R8: u64 = 35;
const FORMAT_X8R8G8B8: u64 = 36;
const FORMAT_Y8_U8V8_N420: u64 = 67;
const FORMAT_Y8_V8U8_N420: u64 = 68;

const BLOCK_KIND_PITCH: u64 = 0;
const BLOCK_KIND_GENERIC_16BX2: u64 = 1;

const FRAME_FORMAT_TOP_FIELD: u64 = 3;
const FRAME_FORMAT_BOTTOM_FIELD: u64 = 4;

const ALPHA_FILL_OPAQUE: u64 = 0;

/// The configuration a VIC job reads from guest memory (the hardware's `ConfigStruct`).
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ConfigStruct([u64; 0x610 / 8]);

impl ConfigStruct {
    fn u64_at(&self, offset: usize) -> u64 {
        self.0[offset / 8]
    }

    fn u32_at(&self, offset: usize) -> u64 {
        (self.0[offset / 8] >> (offset % 8 * 8)) & 0xffff_ffff
    }
}

fn bits(value: u64, low: u32, len: u32) -> u64 {
    (value >> low) & ((1 << len) - 1)
}

fn signed_bits(value: u64, low: u32, len: u32) -> i32 {
    (((value << (64 - low - len)) as i64) >> (64 - len)) as i32
}

/// A rectangle with exclusive right and bottom edges.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    left: u32,
    top: u32,
    right: u32,
    bottom: u32,
}

impl Rect {
    fn sized(width: u32, height: u32) -> Self {
        Self {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        }
    }

    /// From the hardware's inclusive edges.
    fn inclusive(left: u64, top: u64, right: u64, bottom: u64) -> Self {
        Self {
            left: left as u32,
            top: top as u32,
            right: right as u32 + 1,
            bottom: bottom as u32 + 1,
        }
    }

    fn width(self) -> u32 {
        self.right.saturating_sub(self.left)
    }

    fn height(self) -> u32 {
        self.bottom.saturating_sub(self.top)
    }

    fn is_empty(self) -> bool {
        self.width() == 0 || self.height() == 0
    }

    fn intersect(self, other: Self) -> Self {
        Self {
            left: self.left.max(other.left),
            top: self.top.max(other.top),
            right: self.right.min(other.right),
            bottom: self.bottom.min(other.bottom),
        }
    }
}

/// The layout of a slot's input surface or of the output surface.
#[derive(Clone, Copy, Debug)]
struct SurfaceConfig {
    format: u64,
    /// Block height in GOBs (log2) of a block-linear surface; `None` when pitch-linear.
    block_height: Option<u32>,
    width: u32,
    height: u32,
    /// Luma plane size in pixels; the width is the plane's pitch.
    luma_width: u32,
    luma_height: u32,
    /// Chroma plane size in chroma samples.
    chroma_width: u32,
    chroma_height: u32,
}

impl SurfaceConfig {
    /// `None` for block kinds the VIC cannot read or write.
    fn parse(config: &ConfigStruct, offset: usize) -> Option<Self> {
        let word = config.u32_at(offset);
        let block_height = match bits(word, 11, 4) {
            BLOCK_KIND_PITCH => None,
            BLOCK_KIND_GENERIC_16BX2 => Some(bits(word, 15, 4) as u32),
            _ => return None,
        };
        let size = |at: usize| {
            let word = config.u32_at(offset + at);
            (bits(word, 0, 14) as u32 + 1, bits(word, 14, 14) as u32 + 1)
        };
        let (width, height) = size(4);
        let (luma_width, luma_height) = size(8);
        let (chroma_width, chroma_height) = size(12);
        Some(Self {
            format: bits(word, 0, 7),
            block_height,
            width,
            height,
            luma_width,
            luma_height,
            chroma_width,
            chroma_height,
        })
    }
}

/// The pixel formats the VIC writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum OutputFormat {
    /// `A8R8G8B8`: bytes B, G, R, A.
    Bgra,
    /// `A8B8G8R8`: bytes R, G, B, A.
    Rgba,
    /// `Y8__V8U8_N420`: a luma plane and an interleaved U, V plane.
    Nv12,
}

struct OutputConfig {
    format: OutputFormat,
    surface: SurfaceConfig,
    target: Rect,
    /// Background in output byte order.
    background: [u8; 4],
    opaque: bool,
}

impl OutputConfig {
    fn parse(config: &ConfigStruct) -> Result<Self, u64> {
        let surface = SurfaceConfig::parse(config, 0x20).ok_or(u64::MAX)?;
        let format = match surface.format {
            FORMAT_A8R8G8B8 | FORMAT_X8R8G8B8 => OutputFormat::Bgra,
            FORMAT_A8B8G8R8 | FORMAT_X8B8G8R8 => OutputFormat::Rgba,
            FORMAT_Y8_V8U8_N420 => OutputFormat::Nv12,
            other => return Err(other),
        };
        let word = config.u64_at(0x10);
        let opaque = bits(word, 0, 3) == ALPHA_FILL_OPAQUE;
        let component = |low| (bits(word, low, 10) >> 2) as u8;
        let alpha = if opaque { 0xff } else { component(6) };
        let (r, g, b) = (component(16), component(26), component(36));
        let background = match format {
            OutputFormat::Bgra => [b, g, r, alpha],
            OutputFormat::Rgba | OutputFormat::Nv12 => [r, g, b, alpha],
        };
        let horizontal = config.u32_at(0x18);
        let vertical = config.u32_at(0x1c);
        Ok(Self {
            format,
            surface,
            target: Rect::inclusive(
                bits(horizontal, 0, 14),
                bits(vertical, 0, 14),
                bits(horizontal, 16, 14),
                bits(vertical, 16, 14),
            ),
            background,
            opaque,
        })
    }
}

/// YUV to RGB in the VIC's fixed point: 10-bit samples, a 3x4 matrix of signed 20-bit
/// coefficients whose products are shifted right by `shift`, then by 8 after the offset column,
/// and the result clamped to the slot's soft clamp range.
#[derive(Clone, Copy, Debug)]
struct Conversion {
    /// The matrix rows, sample coefficients times 4 to take 8-bit samples.
    rows: [[i32; 4]; 3],
    shift: u32,
    low: i32,
    high: i32,
    matrix: bool,
    alpha: u8,
}

/// Pixels converted together, so each step is a loop over fixed-size arrays the compiler turns into
/// vector instructions.
const BLOCK: usize = 16;

impl Conversion {
    /// Converts BLOCK samples to pixels, R, G and B at bits shifts[0], shifts[1] and shifts[2],
    /// alpha in the top byte. Every step works on 32-bit lanes, so no byte shuffles are needed.
    #[inline(always)]
    fn block(
        &self,
        y: &[u8; BLOCK],
        u: &[u8; BLOCK],
        v: &[u8; BLOCK],
        out: &mut [u32; BLOCK],
        shifts: [u32; 3],
    ) {
        let (shift, low, high) = (self.shift, self.low, self.high);
        let (y, u, v) = (y.map(i32::from), u.map(i32::from), v.map(i32::from));
        let mut pixels = [u32::from(self.alpha) << 24; BLOCK];
        for channel in 0..3 {
            let [cy, cu, cv, offset] = self.rows[channel];
            let position = shifts[channel];
            for i in 0..BLOCK {
                let sum = (y[i] * cy + u[i] * cu + v[i] * cv) >> shift;
                pixels[i] |= ((((sum + offset) >> 8).max(low).min(high) >> 2) as u32) << position;
            }
        }
        *out = pixels.map(u32::to_le);
    }
}

struct SlotConfig {
    /// The field this slot reads, 0 for the top field and 1 for the bottom one.
    field: Option<usize>,
    /// In pixels of the slot surface (or of its field).
    source: Rect,
    dest: Rect,
    conversion: Conversion,
    surface: Option<SurfaceConfig>,
}

impl SlotConfig {
    /// `None` for a disabled slot.
    fn parse(config: &ConfigStruct, slot: usize, opaque: bool) -> Option<Self> {
        let base = 0x90 + slot * 0xb0;
        let word = |at: usize| config.u64_at(base + at);
        let flags = word(0);
        if bits(flags, 0, 1) == 0 {
            return None;
        }
        let field = match bits(flags, 16, 4) {
            FRAME_FORMAT_TOP_FIELD => Some(0),
            FRAME_FORMAT_BOTTOM_FIELD => Some(1),
            _ => None,
        };
        let clamp = word(0x10);
        let low = bits(clamp, 0, 10) as i32;
        let high = bits(clamp, 10, 10) as i32;
        let alpha = if opaque {
            0xff
        } else {
            (bits(clamp, 32, 10) >> 2) as u8
        };
        let fixed = |value: u64| value >> 16;
        let (horizontal, vertical) = (word(0x20), word(0x28));
        let dest = word(0x30);
        let matrix = [word(0x60), word(0x68), word(0x70), word(0x78)];
        let coefficient = |row: u32, column: usize| signed_bits(matrix[column], row * 20, 20);
        let row = |row: u32| {
            std::array::from_fn(|column| coefficient(row, column) * if column < 3 { 4 } else { 1 })
        };
        Some(Self {
            field,
            source: Rect::inclusive(
                fixed(bits(horizontal, 0, 30)),
                fixed(bits(vertical, 0, 30)),
                fixed(bits(horizontal, 32, 30)),
                fixed(bits(vertical, 32, 30)),
            ),
            dest: Rect::inclusive(
                bits(dest, 0, 14),
                bits(dest, 32, 14),
                bits(dest, 16, 14),
                bits(dest, 48, 14),
            ),
            conversion: Conversion {
                rows: [row(0), row(1), row(2)],
                shift: bits(matrix[0], 60, 4) as u32,
                low,
                high: high.min(1023),
                matrix: bits(matrix[1], 63, 1) != 0,
                alpha,
            },
            surface: SurfaceConfig::parse(config, base + 0x40),
        })
    }
}

/// Plane `y`'s row, the last row for any `y` past it.
fn plane_row<'a>(plane: &Plane<'a>, y: usize) -> &'a [u8] {
    let rows = plane.data.len() / plane.stride.max(1);
    let y = y.min(rows.saturating_sub(1));
    plane
        .data
        .get(y * plane.stride..(y + 1) * plane.stride)
        .unwrap_or_default()
}

#[derive(Clone, Copy)]
enum Chroma<'a> {
    /// U and V interleaved (V first when `swapped`).
    Interleaved {
        plane: Plane<'a>,
        swapped: bool,
    },
    Planar(Plane<'a>, Plane<'a>),
}

/// A slot's input picture: 4:2:0 8-bit planes.
#[derive(Clone, Copy)]
struct Source<'a> {
    width: u32,
    height: u32,
    luma: Plane<'a>,
    chroma: Chroma<'a>,
    field: Option<usize>,
}

impl<'a> Source<'a> {
    fn from_picture(picture: Picture<'a>) -> Option<Self> {
        let [luma, first, second] = picture.planes;
        let chroma = match picture.format {
            PixelFormat::Yuv420 => Chroma::Planar(first, second),
            PixelFormat::Nv12 => Chroma::Interleaved {
                plane: first,
                swapped: false,
            },
            PixelFormat::Yuv420P10 => return None,
        };
        Some(Self {
            width: picture.width,
            height: picture.height,
            luma,
            chroma,
            field: None,
        })
    }

    /// Lines the slot reads: every other line of a field.
    fn lines(&self) -> u32 {
        match self.field {
            Some(_) => self.height / 2,
            None => self.height,
        }
    }

    fn luma_line(&self, line: usize) -> &'a [u8] {
        let row = match self.field {
            Some(parity) => line * 2 + parity,
            None => line,
        };
        plane_row(&self.luma, row)
    }

    fn chroma_row(&self, line: usize) -> usize {
        match self.field {
            Some(parity) => line / 2 * 2 + parity,
            None => line / 2,
        }
    }
}

/// Per-row buffers, sized to the widest row seen.
#[derive(Default)]
struct Rows {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    x_map: Vec<u32>,
    pixels: Vec<u32>,
    bytes: Vec<u8>,
}

impl Rows {
    fn reserve(&mut self, width: usize) {
        if self.y.len() < width {
            self.y.resize(width, 0);
            self.u.resize(width, 0);
            self.v.resize(width, 0);
            self.x_map.resize(width, 0);
            self.pixels.resize(width, 0);
            self.bytes.resize(width + 2, 0);
        }
    }

    /// Fills `y`, `u` and `v` with the `width` samples of `line`: from column `start` on, or
    /// at the columns of `x_map` when `start` is `None`.
    #[inline(always)]
    fn gather(&mut self, source: &Source, line: usize, start: Option<usize>, width: usize) {
        let luma = source.luma_line(line);
        let chroma_row = source.chroma_row(line);
        let (y, u, v) = (
            &mut self.y[..width],
            &mut self.u[..width],
            &mut self.v[..width],
        );
        let Some(start) = start else {
            let map = &self.x_map[..width];
            let sample = |row: &[u8], at: usize| row.get(at).copied().unwrap_or(0);
            for (out, &x) in y.iter_mut().zip(map) {
                *out = sample(luma, x as usize);
            }
            match source.chroma {
                Chroma::Interleaved { plane, swapped } => {
                    let row = plane_row(&plane, chroma_row);
                    let (u, v) = if swapped { (v, u) } else { (u, v) };
                    for ((u, v), &x) in u.iter_mut().zip(v.iter_mut()).zip(map) {
                        let at = x as usize & !1;
                        *u = sample(row, at);
                        *v = sample(row, at + 1);
                    }
                }
                Chroma::Planar(first, second) => {
                    let (first, second) = (
                        plane_row(&first, chroma_row),
                        plane_row(&second, chroma_row),
                    );
                    for ((u, v), &x) in u.iter_mut().zip(v.iter_mut()).zip(map) {
                        *u = sample(first, x as usize / 2);
                        *v = sample(second, x as usize / 2);
                    }
                }
            }
            return;
        };
        if let Some(row) = luma.get(start..start + width) {
            y.copy_from_slice(row);
        }
        let odd = start % 2;
        match source.chroma {
            Chroma::Interleaved { plane, swapped } => {
                let row = plane_row(&plane, chroma_row);
                let row = row.get(start - odd..).unwrap_or_default();
                let (u, v) = if swapped { (v, u) } else { (u, v) };
                if odd == 1
                    && let [first_u, first_v, ..] = row
                {
                    u[0] = *first_u;
                    v[0] = *first_v;
                }
                let pairs = row.get(odd * 2..).unwrap_or_default().as_chunks::<2>().0;
                upsample_interleaved(&mut u[odd..], &mut v[odd..], pairs);
            }
            Chroma::Planar(first, second) => {
                for (out, plane) in [(u, first), (v, second)] {
                    let row = plane_row(&plane, chroma_row);
                    let row = row.get(start / 2..).unwrap_or_default();
                    if odd == 1
                        && let Some(&sample) = row.first()
                    {
                        out[0] = sample;
                    }
                    upsample(&mut out[odd..], &row[odd.min(row.len())..]);
                }
            }
        }
    }
}

/// Doubles each chroma sample into two pixels.
#[inline(always)]
fn upsample(out: &mut [u8], chroma: &[u8]) {
    let (pairs, tail) = out.as_chunks_mut::<2>();
    for (pair, &sample) in pairs.iter_mut().zip(chroma) {
        *pair = [sample; 2];
    }
    if let (Some(last), Some(&sample)) = (tail.first_mut(), chroma.get(pairs.len())) {
        *last = sample;
    }
}

/// Splits interleaved U, V samples into two rows, each sample doubled into two pixels.
#[inline(always)]
fn upsample_interleaved(u: &mut [u8], v: &mut [u8], chroma: &[[u8; 2]]) {
    let (u_pairs, u_tail) = u.as_chunks_mut::<2>();
    let (v_pairs, v_tail) = v.as_chunks_mut::<2>();
    for ((u, v), &[cu, cv]) in u_pairs.iter_mut().zip(v_pairs.iter_mut()).zip(chroma) {
        *u = [cu; 2];
        *v = [cv; 2];
    }
    if let (Some(u), Some(v), Some(&[cu, cv])) = (
        u_tail.first_mut(),
        v_tail.first_mut(),
        chroma.get(u_pairs.len()),
    ) {
        *u = cu;
        *v = cv;
    }
}

/// Converts a row of samples to 32-bit pixels: R, G, B, A in memory, or B, G, R, A with BGRA.
#[inline(always)]
fn convert_row<const BGRA: bool>(
    conversion: &Conversion,
    y: &[u8],
    u: &[u8],
    v: &[u8],
    out: &mut [u32],
) {
    let shifts = if BGRA { [16, 8, 0] } else { [0, 8, 16] };
    if !conversion.matrix {
        let alpha = u32::from(conversion.alpha) << 24;
        for (pixel, ((&y, &u), &v)) in out.iter_mut().zip(y.iter().zip(u).zip(v)) {
            let [y, u, v] = [y, u, v].map(u32::from);
            *pixel = (alpha | y << shifts[0] | u << shifts[1] | v << shifts[2]).to_le();
        }
        return;
    }
    let len = out.len().min(y.len()).min(u.len()).min(v.len());
    let (y, u, v, out) = (&y[..len], &u[..len], &v[..len], &mut out[..len]);
    let (y_blocks, y_tail) = y.as_chunks::<BLOCK>();
    let (u_blocks, u_tail) = u.as_chunks::<BLOCK>();
    let (v_blocks, v_tail) = v.as_chunks::<BLOCK>();
    let (out_blocks, out_tail) = out.as_chunks_mut::<BLOCK>();
    let samples = y_blocks.iter().zip(u_blocks).zip(v_blocks);
    for (out, ((y, u), v)) in out_blocks.iter_mut().zip(samples) {
        conversion.block(y, u, v, out, shifts);
    }
    let mut padded = [[0u8; BLOCK]; 3];
    for (padded, tail) in padded.iter_mut().zip([y_tail, u_tail, v_tail]) {
        padded[..tail.len()].copy_from_slice(tail);
    }
    let mut pixels = [0u32; BLOCK];
    conversion.block(&padded[0], &padded[1], &padded[2], &mut pixels, shifts);
    out_tail.copy_from_slice(&pixels[..out_tail.len()]);
}

/// How a plane lies in memory.
#[derive(Clone, Copy, Debug, Default)]
struct PlaneLayout {
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    pitch: usize,
    block_height: Option<u32>,
}

impl PlaneLayout {
    fn linear_size(&self) -> usize {
        self.pitch * self.height as usize
    }

    fn tiling(&self, block_height: u32) -> Tiling {
        Tiling::new(
            self.width,
            self.height,
            1,
            self.bytes_per_pixel,
            block_height,
            0,
            GobLayout::Tegra,
        )
    }
}

/// Reports a write to device memory to the process's write tracking, page by page.
fn report_writes(memory: &DeviceAddressSpace, process: &AddressSpace, address: u64, len: u64) {
    let mut page = address & !PAGE_MASK;
    while page < address + len {
        if let Some(pointer) = memory.translate(page) {
            process.report_write(pointer);
        }
        page += PAGE_SIZE;
    }
}

/// Tiles whole 64-byte GOB rows of row y, from byte x on when x starts a GOB, and returns how
/// many it wrote. A GOB row is four 16-byte units at fixed offsets inside the GOB, and GOBs along
/// the row are a fixed stride apart, so this costs a few copies per GOB.
#[inline(always)]
fn write_gob_rows(tiling: &Tiling, tiled: &mut [u8], x: u32, y: u32, bytes: &[u8]) -> usize {
    if !x.is_multiple_of(GOB_WIDTH) {
        return 0;
    }
    let first = gob_offset(0, y, tiling.layout) as usize;
    let units: [usize; 4] =
        std::array::from_fn(|unit| gob_offset(unit as u32 * 16, y, tiling.layout) as usize - first);
    let stride = tiling.offset(GOB_WIDTH, y, 0) - tiling.offset(0, y, 0);
    let mut base = tiling.offset(x, y, 0) - first;
    let gobs = bytes.as_chunks::<64>().0;
    for row in gobs {
        let Some(gob) = tiled.get_mut(base..base + GOB_SIZE as usize) else {
            return 0;
        };
        for (unit, &at) in units.iter().enumerate() {
            gob[at + first..at + first + 16].copy_from_slice(&row[unit * 16..unit * 16 + 16]);
        }
        base += stride;
    }
    gobs.len()
}

/// One plane of the output surface, kept in its final layout (pitch-linear or tiled) as it is
/// composed, so that it goes to guest memory in one write.
#[derive(Default)]
struct OutputPlane {
    layout: PlaneLayout,
    tiling: Option<Tiling>,
    data: Vec<u8>,
}

impl OutputPlane {
    /// Sizes the plane for layout and fills every pixel with pattern.
    fn prepare<const N: usize>(&mut self, layout: PlaneLayout, pattern: [u8; N]) {
        self.layout = layout;
        self.tiling = layout
            .block_height
            .map(|block_height| layout.tiling(block_height));
        let size = self
            .tiling
            .map_or(layout.linear_size(), |tiling| tiling.size_bytes());
        self.data.resize(size, 0);
        self.data.as_chunks_mut::<N>().0.fill(pattern);
    }

    /// Writes the pixels in bytes to row y from pixel x on, dropping what lies past the plane.
    #[inline(always)]
    fn write_row(&mut self, x: u32, y: u32, bytes: &[u8]) {
        let layout = self.layout;
        if x >= layout.width || y >= layout.height {
            return;
        }
        let width = (bytes.len() as u32 / layout.bytes_per_pixel).min(layout.width - x);
        let bytes = &bytes[..(width * layout.bytes_per_pixel) as usize];
        match &self.tiling {
            None => {
                let at = y as usize * layout.pitch + (x * layout.bytes_per_pixel) as usize;
                self.data[at..at + bytes.len()].copy_from_slice(bytes);
            }
            Some(tiling) => {
                let gobs =
                    write_gob_rows(tiling, &mut self.data, x * layout.bytes_per_pixel, y, bytes);
                let done = gobs as u32 * GOB_WIDTH / layout.bytes_per_pixel;
                if done < width {
                    let region = Region {
                        x: x + done,
                        y,
                        z: 0,
                        width: width - done,
                        height: 1,
                        depth: 1,
                    };
                    let rest = &bytes[(done * layout.bytes_per_pixel) as usize..];
                    swizzle_region(tiling, region, rest, &mut self.data, None);
                }
            }
        }
    }

    fn store(&self, memory: &DeviceAddressSpace, process: Option<&AddressSpace>, address: u64) {
        if memory.write_bytes(address, &self.data).is_err() {
            tracing::debug!(
                address,
                len = self.data.len(),
                "vic output surface is not mapped"
            );
        }
        if let Some(process) = process {
            report_writes(memory, process, address, self.data.len() as u64);
        }
    }
}

/// Reads a plane from guest memory into `linear` (rows `layout.pitch` apart), untiling it for
/// block-linear layouts.
fn load_plane(
    memory: &DeviceAddressSpace,
    address: u64,
    layout: PlaneLayout,
    linear: &mut Vec<u8>,
    tiled: &mut Vec<u8>,
) -> bool {
    linear.resize(layout.linear_size(), 0);
    match layout.block_height {
        None => memory.read_bytes(address, linear).is_ok(),
        Some(block_height) => {
            let tiling = layout.tiling(block_height);
            tiled.resize(tiling.size_bytes(), 0);
            if memory.read_bytes(address, tiled).is_err() {
                return false;
            }
            unswizzle_region(
                &tiling,
                Region::whole(&tiling),
                tiled,
                linear,
                Some(layout.pitch),
            );
            true
        }
    }
}

/// An NV12 surface read from guest memory, for slots no decoded picture is kept for.
#[derive(Default)]
struct GuestSurface {
    luma: Vec<u8>,
    chroma: Vec<u8>,
    tiled: Vec<u8>,
}

impl GuestSurface {
    fn read(
        &mut self,
        memory: &DeviceAddressSpace,
        surface: &SurfaceConfig,
        luma: u64,
        chroma: u64,
    ) -> Option<Source<'_>> {
        let swapped = match surface.format {
            FORMAT_Y8_V8U8_N420 => false,
            FORMAT_Y8_U8V8_N420 => true,
            _ => return None,
        };
        let luma_layout = PlaneLayout {
            width: surface.luma_width,
            height: surface.luma_height,
            bytes_per_pixel: 1,
            pitch: surface.luma_width as usize,
            block_height: surface.block_height,
        };
        let chroma_layout = PlaneLayout {
            width: surface.chroma_width,
            height: surface.chroma_height,
            bytes_per_pixel: 2,
            pitch: surface.chroma_width as usize * 2,
            block_height: surface.block_height,
        };
        if !load_plane(memory, luma, luma_layout, &mut self.luma, &mut self.tiled)
            || !load_plane(
                memory,
                chroma,
                chroma_layout,
                &mut self.chroma,
                &mut self.tiled,
            )
        {
            return None;
        }
        Some(Source {
            width: surface
                .width
                .min(surface.luma_width)
                .min(surface.chroma_width * 2),
            height: surface
                .height
                .min(surface.luma_height)
                .min(surface.chroma_height * 2),
            luma: Plane {
                data: &self.luma,
                stride: luma_layout.pitch,
            },
            chroma: Chroma::Interleaved {
                plane: Plane {
                    data: &self.chroma,
                    stride: chroma_layout.pitch,
                },
                swapped,
            },
            field: None,
        })
    }
}

/// The output surface as it is composed.
struct Staging {
    format: OutputFormat,
    /// The part of the surface slots are composed into.
    visible: Rect,
    luma: OutputPlane,
    chroma: OutputPlane,
}

impl Default for Staging {
    fn default() -> Self {
        Self {
            format: OutputFormat::Rgba,
            visible: Rect::sized(0, 0),
            luma: OutputPlane::default(),
            chroma: OutputPlane::default(),
        }
    }
}

impl Staging {
    /// Sizes the planes for output and fills them with its background colour.
    fn prepare(&mut self, output: &OutputConfig) {
        let surface = &output.surface;
        self.format = output.format;
        self.visible = Rect::sized(
            surface.width.min(surface.luma_width),
            surface.height.min(surface.luma_height),
        );
        let layout = |width: u32, height: u32, bytes_per_pixel: u32| PlaneLayout {
            width,
            height,
            bytes_per_pixel,
            pitch: (width as usize * bytes_per_pixel as usize).next_multiple_of(16),
            block_height: surface.block_height,
        };
        let [r, g, b, _] = output.background;
        match output.format {
            OutputFormat::Bgra | OutputFormat::Rgba => {
                let luma = layout(surface.luma_width, surface.luma_height, 4);
                self.luma.prepare(luma, output.background);
            }
            OutputFormat::Nv12 => {
                self.luma
                    .prepare(layout(surface.luma_width, surface.luma_height, 1), [r]);
                let chroma = layout(surface.chroma_width, surface.chroma_height, 2);
                self.chroma.prepare(chroma, [g, b]);
            }
        }
    }

    /// Composes one slot. Runs a copy compiled for AVX2 where the CPU has it: the baseline x86-64
    /// target has no 32-bit vector multiply, which makes the conversion six times slower.
    fn compose(&mut self, slot: &SlotConfig, source: &Source, target: Rect, rows: &mut Rows) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU supports AVX2.
            return unsafe { self.compose_avx2(slot, source, target, rows) };
        }
        self.compose_rows(slot, source, target, rows);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    fn compose_avx2(&mut self, slot: &SlotConfig, source: &Source, target: Rect, rows: &mut Rows) {
        self.compose_rows(slot, source, target, rows);
    }

    #[inline(always)]
    fn compose_rows(&mut self, slot: &SlotConfig, source: &Source, target: Rect, rows: &mut Rows) {
        let src = slot
            .source
            .intersect(Rect::sized(source.width, source.lines()));
        let full = slot.dest;
        let dest = full.intersect(target).intersect(self.visible);
        if src.is_empty() || dest.is_empty() || full.is_empty() {
            return;
        }
        let step_x = (u64::from(src.width()) << 16) / u64::from(full.width());
        let step_y = (u64::from(src.height()) << 16) / u64::from(full.height());
        let map = |at: u32, origin: u32, step: u64, from: u32, end: u32| {
            (from + ((u64::from(at - origin) * step) >> 16) as u32).min(end - 1)
        };
        let width = dest.width() as usize;
        rows.reserve(width);
        let start = (step_x == 1 << 16)
            .then(|| map(dest.left, full.left, step_x, src.left, src.right) as usize);
        if start.is_none() {
            for (i, out) in rows.x_map[..width].iter_mut().enumerate() {
                *out = map(dest.left + i as u32, full.left, step_x, src.left, src.right);
            }
        }
        let conversion = &slot.conversion;
        for y in dest.top..dest.bottom {
            let line = map(y, full.top, step_y, src.top, src.bottom) as usize;
            rows.gather(source, line, start, width);
            let (ys, us, vs) = (&rows.y[..width], &rows.u[..width], &rows.v[..width]);
            let pixels = &mut rows.pixels[..width];
            match self.format {
                OutputFormat::Bgra => convert_row::<true>(conversion, ys, us, vs, pixels),
                OutputFormat::Rgba | OutputFormat::Nv12 => {
                    convert_row::<false>(conversion, ys, us, vs, pixels);
                }
            }
            if self.format != OutputFormat::Nv12 {
                self.luma
                    .write_row(dest.left, y, bytemuck::cast_slice(pixels));
                continue;
            }
            let bytes = &mut rows.bytes;
            for (out, &pixel) in bytes.iter_mut().zip(&*pixels) {
                *out = u32::from_le(pixel) as u8;
            }
            self.luma.write_row(dest.left, y, &bytes[..width]);
            if y % 2 == 1 {
                continue;
            }
            let first = dest.left.next_multiple_of(2);
            let skip = (first - dest.left) as usize;
            let pairs = bytes.as_chunks_mut::<2>().0;
            let mut count = 0;
            for (pair, &pixel) in pairs.iter_mut().zip(pixels.iter().skip(skip).step_by(2)) {
                let [_, u, v, _] = u32::from_le(pixel).to_le_bytes();
                *pair = [u, v];
                count += 1;
            }
            self.chroma.write_row(first / 2, y / 2, &bytes[..count * 2]);
        }
    }

    fn store(
        &self,
        memory: &DeviceAddressSpace,
        process: Option<&AddressSpace>,
        luma: u64,
        chroma: u64,
    ) {
        self.luma.store(memory, process, luma);
        if self.format == OutputFormat::Nv12 {
            self.chroma.store(memory, process, chroma);
        }
    }
}

pub(super) struct Vic {
    regs: Box<[u32; REGISTER_COUNT]>,
    memory: Arc<DeviceAddressSpace>,
    process: Option<Arc<AddressSpace>>,
    frames: Arc<DecodedFrames>,
    config: Box<ConfigStruct>,
    staging: Staging,
    rows: Rows,
    guest: GuestSurface,
    warned: bool,
    /// Whether jobs compose their output; without, they write nothing.
    compose: bool,
}

impl Vic {
    pub(super) fn new(
        memory: Arc<DeviceAddressSpace>,
        process: Option<Arc<AddressSpace>>,
        frames: Arc<DecodedFrames>,
        compose: bool,
    ) -> Self {
        Self {
            regs: Box::new([0; REGISTER_COUNT]),
            memory,
            process,
            frames,
            config: Box::new(Zeroable::zeroed()),
            staging: Staging::default(),
            rows: Rows::default(),
            guest: GuestSurface::default(),
            warned: false,
            compose,
        }
    }

    /// An engine method write: VIC's registers are 32 bits wide.
    pub(super) fn write(&mut self, method: u32, argument: u32) {
        if let Some(reg) = self.regs.get_mut(method as usize) {
            *reg = argument;
        }
        if method == EXECUTE {
            self.execute();
        }
    }

    fn address(&self, register: usize) -> u64 {
        u64::from(self.regs[register]) << 8
    }

    fn execute(&mut self) {
        if !self.compose {
            return;
        }
        let config = self.address(CONFIG_STRUCT_OFFSET);
        if self
            .memory
            .read_bytes(config, bytemuck::bytes_of_mut(&mut *self.config))
            .is_err()
        {
            tracing::warn!(config, "vic config struct is not mapped");
            return;
        }
        let output = match OutputConfig::parse(&self.config) {
            Ok(output) => output,
            Err(format) => {
                if !std::mem::replace(&mut self.warned, true) {
                    tracing::warn!(format, "unsupported vic output surface");
                }
                return;
            }
        };
        self.staging.prepare(&output);
        for index in 0..SLOT_COUNT {
            let Some(slot) = SlotConfig::parse(&self.config, index, output.opaque) else {
                continue;
            };
            let surface = SURFACES + index * SURFACE_WORDS_PER_SLOT;
            let (luma, chroma) = (self.address(surface), self.address(surface + 1));
            let lent = self.frames.lend(luma);
            let source = match lent.as_deref().and_then(lichii_video::Frame::picture) {
                Some(picture) => Source::from_picture(picture),
                None => slot
                    .surface
                    .and_then(|surface| self.guest.read(&self.memory, &surface, luma, chroma)),
            };
            if let Some(mut source) = source {
                source.field = slot.field;
                self.staging
                    .compose(&slot, &source, output.target, &mut self.rows);
            }
            if let Some(lent) = lent {
                self.frames.recycle(lent);
            }
        }
        let (luma, chroma) = (
            self.address(OUTPUT_LUMA_OFFSET),
            self.address(OUTPUT_CHROMA_OFFSET),
        );
        self.staging
            .store(&self.memory, self.process.as_deref(), luma, chroma);
    }
}
