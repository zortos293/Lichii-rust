//! Tegra block-linear tiling.
//!
//! A GOB ("group of bytes") is 64 bytes wide and 8 rows tall. GOBs stack into blocks of
//! `1 << block_height_log2` GOBs vertically and `1 << block_depth_log2` GOBs in depth, and blocks
//! fill a surface row by row. Inside a GOB, byte `x` of row `y` lies at
//! `pdep(x, 0b1_0010_1111) | pdep(y, 0b0_1101_0000)`, so every 16-byte unit stays contiguous.
//! Switch 2 additionally swaps address bits 5 and 6 within each GOB ([`GobLayout::Switch2`]).

use crate::PixelFormat;

/// Width of a GOB in bytes.
pub const GOB_WIDTH: u32 = 64;
/// Height of a GOB in rows of blocks.
pub const GOB_HEIGHT: u32 = 8;
/// Bytes in one GOB.
pub const GOB_SIZE: u32 = GOB_WIDTH * GOB_HEIGHT;
const GOB_SIZE_LOG2: u32 = 9;

/// Byte order inside a GOB.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum GobLayout {
    /// Tegra X1 (Switch 1) and the Switch 2 in its compatibility layout.
    #[default]
    Tegra,
    /// Switch 2 native layout: address bits 5 and 6 inside a GOB are swapped.
    Switch2,
}

impl GobLayout {
    const fn remap(self, offset: u32) -> u32 {
        match self {
            Self::Tegra => offset,
            Self::Switch2 => {
                let differ = ((offset >> 5) ^ (offset >> 6)) & 1;
                offset ^ (differ << 5) ^ (differ << 6)
            }
        }
    }

    /// Offset of each 16-byte unit inside a GOB, by row (`y & 7`) and unit (`x / 16 & 3`).
    const fn unit_offsets(self) -> [[u32; 4]; 8] {
        let mut table = [[0; 4]; 8];
        let mut row = 0;
        while row < 8 {
            let mut unit = 0;
            while unit < 4 {
                let x = unit as u32 * 16;
                let y = row as u32;
                let base =
                    ((x & 0x20) << 3) | ((x & 0x10) << 1) | ((y & 0x6) << 5) | ((y & 1) << 4);
                table[row][unit] = self.remap(base);
                unit += 1;
            }
            row += 1;
        }
        table
    }
}

const UNITS_TEGRA: [[u32; 4]; 8] = GobLayout::Tegra.unit_offsets();
const UNITS_SWITCH2: [[u32; 4]; 8] = GobLayout::Switch2.unit_offsets();

/// Byte offset of byte `x` in row `y` inside one GOB.
pub const fn gob_offset(x: u32, y: u32, layout: GobLayout) -> u32 {
    let base =
        ((x & 0x20) << 3) | ((x & 0x10) << 1) | (x & 0xf) | ((y & 0x6) << 5) | ((y & 1) << 4);
    layout.remap(base)
}

/// One tiled image level (all its depth slices), measured in format blocks: texels for
/// uncompressed formats, compression blocks otherwise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tiling {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub bytes_per_block: u32,
    pub block_height_log2: u32,
    pub block_depth_log2: u32,
    /// GOBs per row of blocks; at least `ceil(width * bytes_per_block / 64)`.
    pub row_gobs: u32,
    pub layout: GobLayout,
}

impl Tiling {
    /// A level whose rows are exactly as many GOBs wide as its texels need.
    pub const fn new(
        width: u32,
        height: u32,
        depth: u32,
        bytes_per_block: u32,
        block_height_log2: u32,
        block_depth_log2: u32,
        layout: GobLayout,
    ) -> Self {
        Self {
            width,
            height,
            depth,
            bytes_per_block,
            block_height_log2,
            block_depth_log2,
            row_gobs: (width * bytes_per_block).div_ceil(GOB_WIDTH),
            layout,
        }
    }

    const fn x_shift(&self) -> u32 {
        GOB_SIZE_LOG2 + self.block_height_log2 + self.block_depth_log2
    }

    const fn block_size(&self) -> usize {
        (self.row_gobs as usize) << self.x_shift()
    }

    const fn slice_size(&self) -> usize {
        let block_rows = self.height.div_ceil(GOB_HEIGHT << self.block_height_log2);
        block_rows as usize * self.block_size()
    }

    /// Bytes the tiled level covers, including the padding up to whole blocks.
    pub const fn size_bytes(&self) -> usize {
        let slabs = self.depth.div_ceil(1 << self.block_depth_log2) as usize;
        slabs * self.slice_size()
    }

    /// Bytes of the same level stored linearly with tightly packed rows.
    pub const fn linear_size_bytes(&self) -> usize {
        self.width as usize
            * self.bytes_per_block as usize
            * self.height as usize
            * self.depth as usize
    }

    const fn slice_offset(&self, z: u32) -> usize {
        let mask = (1 << self.block_depth_log2) - 1;
        (z >> self.block_depth_log2) as usize * self.slice_size()
            + (((z & mask) as usize) << (GOB_SIZE_LOG2 + self.block_height_log2))
    }

    const fn row_offset(&self, y: u32) -> usize {
        let gob_y = y / GOB_HEIGHT;
        let mask = (1 << self.block_height_log2) - 1;
        (gob_y >> self.block_height_log2) as usize * self.block_size()
            + (((gob_y & mask) as usize) << GOB_SIZE_LOG2)
    }

    /// Tiled offset of byte `x` (in bytes from the row start) of row `y` in slice `z`.
    pub const fn offset(&self, x: u32, y: u32, z: u32) -> usize {
        self.slice_offset(z)
            + self.row_offset(y)
            + (((x / GOB_WIDTH) as usize) << self.x_shift())
            + gob_offset(x, y, self.layout) as usize
    }

    fn units(&self) -> &'static [[u32; 4]; 8] {
        match self.layout {
            GobLayout::Tegra => &UNITS_TEGRA,
            GobLayout::Switch2 => &UNITS_SWITCH2,
        }
    }
}

/// A box inside a level, in blocks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Region {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl Region {
    /// The whole level, every slice included.
    pub const fn whole(tiling: &Tiling) -> Self {
        Self {
            x: 0,
            y: 0,
            z: 0,
            width: tiling.width,
            height: tiling.height,
            depth: tiling.depth,
        }
    }
}

macro_rules! for_each_run {
    ($tiling:expr, $region:expr, $row_pitch:expr, $slice_pitch:expr, |$tiled_at:ident, $linear_at:ident, $len:ident| $body:block) => {{
        let tiling: &Tiling = $tiling;
        let region: Region = $region;
        let units = tiling.units();
        let x_shift = tiling.x_shift();
        let bpb = tiling.bytes_per_block;
        let x_begin = region.x * bpb;
        let x_end = (region.x + region.width) * bpb;
        for dz in 0..region.depth {
            let slice = tiling.slice_offset(region.z + dz);
            let linear_slice = dz as usize * $slice_pitch;
            for dy in 0..region.height {
                let y = region.y + dy;
                let row = slice + tiling.row_offset(y);
                let row_units = &units[(y & 7) as usize];
                let linear_row = linear_slice + dy as usize * $row_pitch;
                let mut x = x_begin;
                while x < x_end {
                    let unit_start = row
                        + (((x >> 6) as usize) << x_shift)
                        + row_units[((x >> 4) & 3) as usize] as usize;
                    let $linear_at = linear_row + (x - x_begin) as usize;
                    if x & 15 == 0 && x + 16 <= x_end {
                        let $len: usize = 16;
                        let $tiled_at = unit_start;
                        $body
                        x += 16;
                    } else {
                        let $len = ((16 - (x & 15)).min(x_end - x)) as usize;
                        let $tiled_at = unit_start + (x & 15) as usize;
                        $body
                        x += $len as u32;
                    }
                }
            }
        }
    }};
}

fn linear_pitches(tiling: &Tiling, region: &Region, row_pitch: Option<usize>) -> (usize, usize) {
    let row = row_pitch.unwrap_or(region.width as usize * tiling.bytes_per_block as usize);
    (row, row * region.height as usize)
}

/// Copies `region` of a tiled level into `linear`, whose rows are `row_pitch` bytes apart
/// (tightly packed when `None`) and whose slices are `row_pitch * region.height` apart.
pub fn unswizzle_region(
    tiling: &Tiling,
    region: Region,
    tiled: &[u8],
    linear: &mut [u8],
    row_pitch: Option<usize>,
) {
    let (row_pitch, slice_pitch) = linear_pitches(tiling, &region, row_pitch);
    for_each_run!(tiling, region, row_pitch, slice_pitch, |t, l, len| {
        linear[l..l + len].copy_from_slice(&tiled[t..t + len]);
    });
}

/// Copies a linear image into `region` of a tiled level; the inverse of [`unswizzle_region`].
pub fn swizzle_region(
    tiling: &Tiling,
    region: Region,
    linear: &[u8],
    tiled: &mut [u8],
    row_pitch: Option<usize>,
) {
    let (row_pitch, slice_pitch) = linear_pitches(tiling, &region, row_pitch);
    for_each_run!(tiling, region, row_pitch, slice_pitch, |t, l, len| {
        tiled[t..t + len].copy_from_slice(&linear[l..l + len]);
    });
}

/// Untiles a whole level into tightly packed rows and slices.
pub fn unswizzle(tiling: &Tiling, tiled: &[u8], linear: &mut [u8]) {
    unswizzle_region(tiling, Region::whole(tiling), tiled, linear, None);
}

/// Tiles a tightly packed linear level.
pub fn swizzle(tiling: &Tiling, linear: &[u8], tiled: &mut [u8]) {
    swizzle_region(tiling, Region::whole(tiling), linear, tiled, None);
}

/// Where a block-linear surface's levels and layers lie in guest memory, following the
/// hardware's rules: each level shrinks its block height and depth to fit, and layers are
/// aligned to whole blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SurfaceLayout {
    pub format: PixelFormat,
    /// Level 0 size in texels.
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub levels: u32,
    pub layers: u32,
    /// Level 0 block height and depth in GOBs, log2, as the guest programs them.
    pub block_height_log2: u32,
    pub block_depth_log2: u32,
    /// Row alignment in GOBs, log2 (the texture header's tile width spacing).
    pub tile_width_spacing: u32,
    pub layout: GobLayout,
}

/// Reduces a block dimension (log2, in GOBs) while the level fits in half of it.
const fn shrink_block(mut block_log2: u32, extent: u32, gob_extent: u32) -> u32 {
    while block_log2 > 0 && extent <= (gob_extent << (block_log2 - 1)) {
        block_log2 -= 1;
    }
    block_log2
}

impl SurfaceLayout {
    const fn level_extent(&self, level: u32) -> (u32, u32, u32) {
        let info = self.format.info();
        let w = if self.width >> level == 0 {
            1
        } else {
            self.width >> level
        };
        let h = if self.height >> level == 0 {
            1
        } else {
            self.height >> level
        };
        let d = if self.depth >> level == 0 {
            1
        } else {
            self.depth >> level
        };
        (
            w.div_ceil(info.block_width),
            h.div_ceil(info.block_height),
            d,
        )
    }

    const fn single_level(&self) -> bool {
        self.levels == 1
    }

    /// The tiling of one level, ready for [`unswizzle`] and [`swizzle`].
    pub const fn level_tiling(&self, level: u32) -> Tiling {
        let (width, height, depth) = self.level_extent(level);
        let bytes_per_block = self.format.bytes_per_block();
        let block_height_log2 = shrink_block(self.block_height_log2, height, GOB_HEIGHT);
        let block_depth_log2 = if level == 0 && self.single_level() {
            self.block_depth_log2
        } else {
            shrink_block(self.block_depth_log2, depth, 1)
        };
        let mut tiling = Tiling::new(
            width,
            height,
            depth,
            bytes_per_block,
            block_height_log2,
            block_depth_log2,
            self.layout,
        );
        if !self.is_small(width, height, depth, self.block_height_log2) {
            let align = 1 << self.tile_width_spacing;
            tiling.row_gobs = tiling.row_gobs.div_ceil(align) * align;
        }
        tiling
    }

    const fn is_small(&self, width: u32, height: u32, depth: u32, block_height_log2: u32) -> bool {
        let gob_texels = GOB_WIDTH / self.format.bytes_per_block();
        width <= gob_texels << self.tile_width_spacing
            || height <= GOB_HEIGHT << block_height_log2
            || depth < 1 << self.block_depth_log2
    }

    /// Guest bytes of one level of one layer.
    pub const fn level_size(&self, level: u32) -> u64 {
        let (width, height, depth) = self.level_extent(level);
        let (shift_h, shift_d) = if level == 0 && self.single_level() {
            (self.block_height_log2, self.block_depth_log2)
        } else {
            (
                shrink_block(self.block_height_log2, height, GOB_HEIGHT),
                shrink_block(self.block_depth_log2, depth, 1),
            )
        };
        let mut gobs_x = (width * self.format.bytes_per_block()).div_ceil(GOB_WIDTH);
        if !self.is_small(width, height, depth, self.block_height_log2) {
            let align = 1 << self.tile_width_spacing;
            gobs_x = gobs_x.div_ceil(align) * align;
        }
        let gobs_y = height.div_ceil(GOB_HEIGHT);
        let tiles_y = gobs_y.div_ceil(1 << shift_h);
        let tiles_z = depth.div_ceil(1 << shift_d);
        (gobs_x as u64 * tiles_y as u64 * tiles_z as u64) << (GOB_SIZE_LOG2 + shift_h + shift_d)
    }

    /// Offset of `level` from the start of its layer.
    pub const fn level_offset(&self, level: u32) -> u64 {
        let mut offset = 0;
        let mut l = 0;
        while l < level {
            offset += self.level_size(l);
            l += 1;
        }
        offset
    }

    /// Distance between consecutive array layers: the whole mip chain, aligned to a block.
    pub const fn layer_stride(&self) -> u64 {
        let chain = self.level_offset(self.levels);
        if self.tile_width_spacing > 0 {
            let align = 1u64
                << (GOB_SIZE_LOG2
                    + self.tile_width_spacing
                    + self.block_height_log2
                    + self.block_depth_log2);
            return chain.div_ceil(align) * align;
        }
        let block_rows = self.height.div_ceil(self.format.block_height());
        let block_height = shrink_block(self.block_height_log2, block_rows, GOB_HEIGHT);
        let block_depth = shrink_block(self.block_depth_log2, self.depth, 1);
        let align = 1u64 << (GOB_SIZE_LOG2 + block_height + block_depth);
        chain.div_ceil(align) * align
    }

    /// Guest bytes the whole surface covers.
    pub const fn size_bytes(&self) -> u64 {
        if self.layers > 1 {
            self.layer_stride() * self.layers as u64
        } else {
            self.level_offset(self.levels)
        }
    }

    /// Offset of one subresource from the surface's base address.
    pub const fn subresource_offset(&self, level: u32, layer: u32) -> u64 {
        layer as u64 * self.layer_stride() + self.level_offset(level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pdep(mut value: u32, mut mask: u32) -> u32 {
        let mut out = 0;
        while mask != 0 {
            let bit = mask.isolate_lowest_one();
            if value & 1 != 0 {
                out |= bit;
            }
            value >>= 1;
            mask &= mask - 1;
        }
        out
    }

    fn reference_offset(t: &Tiling, x: u32, y: u32, z: u32) -> usize {
        let block_size = (t.row_gobs as usize) << (9 + t.block_height_log2 + t.block_depth_log2);
        let block_rows = t.height.div_ceil(8 << t.block_height_log2) as usize;
        let gob_y = y / 8;
        let in_gob = pdep(x % 64, 0b1_0010_1111) | pdep(y % 8, 0b0_1101_0000);
        let in_gob = match t.layout {
            GobLayout::Tegra => in_gob,
            GobLayout::Switch2 => {
                let b5 = (in_gob >> 5) & 1;
                let b6 = (in_gob >> 6) & 1;
                (in_gob & !0x60) | (b5 << 6) | (b6 << 5)
            }
        };
        (z >> t.block_depth_log2) as usize * block_rows * block_size
            + (gob_y >> t.block_height_log2) as usize * block_size
            + (x / 64) as usize * block_size / t.row_gobs as usize
            + (((z & ((1 << t.block_depth_log2) - 1)) as usize) << (9 + t.block_height_log2))
            + (((gob_y & ((1 << t.block_height_log2) - 1)) as usize) << 9)
            + in_gob as usize
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn below(&mut self, n: u32) -> u32 {
            self.next() % n
        }
    }

    fn random_tiling(rng: &mut Rng) -> Tiling {
        let bpb = [1, 2, 3, 4, 6, 8, 12, 16][rng.below(8) as usize];
        let layout = if rng.below(2) == 0 {
            GobLayout::Tegra
        } else {
            GobLayout::Switch2
        };
        let mut t = Tiling::new(
            1 + rng.below(150),
            1 + rng.below(90),
            1 + rng.below(5),
            bpb,
            rng.below(5),
            rng.below(3),
            layout,
        );
        t.row_gobs += rng.below(3);
        t
    }

    #[test]
    fn offsets_match_the_bit_formula() {
        let mut rng = Rng(0x1234_5678_9abc_def0);
        for _ in 0..200 {
            let t = random_tiling(&mut rng);
            for _ in 0..200 {
                let x = rng.below(t.width * t.bytes_per_block);
                let y = rng.below(t.height);
                let z = rng.below(t.depth);
                assert_eq!(
                    t.offset(x, y, z),
                    reference_offset(&t, x, y, z),
                    "{t:?} {x} {y} {z}"
                );
                assert!(t.offset(x, y, z) < t.size_bytes());
            }
        }
    }

    #[test]
    fn unswizzle_matches_the_reference_and_round_trips() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for _ in 0..300 {
            let t = random_tiling(&mut rng);
            let tiled: Vec<u8> = (0..t.size_bytes()).map(|_| rng.next() as u8).collect();
            let mut linear = vec![0; t.linear_size_bytes()];
            unswizzle(&t, &tiled, &mut linear);
            let row = (t.width * t.bytes_per_block) as usize;
            for z in 0..t.depth {
                for y in 0..t.height {
                    for x in 0..t.width * t.bytes_per_block {
                        let l = (z * t.height + y) as usize * row + x as usize;
                        assert_eq!(linear[l], tiled[reference_offset(&t, x, y, z)], "{t:?}");
                    }
                }
            }
            let mut back = vec![0; t.size_bytes()];
            swizzle(&t, &linear, &mut back);
            let mut again = vec![0; linear.len()];
            unswizzle(&t, &back, &mut again);
            assert_eq!(again, linear);
        }
    }

    #[test]
    fn regions_copy_only_their_box() {
        let mut rng = Rng(0x0bad_cafe_1234_5678);
        for _ in 0..300 {
            let t = random_tiling(&mut rng);
            let x = rng.below(t.width);
            let y = rng.below(t.height);
            let z = rng.below(t.depth);
            let region = Region {
                x,
                y,
                z,
                width: 1 + rng.below(t.width - x),
                height: 1 + rng.below(t.height - y),
                depth: 1 + rng.below(t.depth - z),
            };
            let pitch = (region.width * t.bytes_per_block + rng.below(8)) as usize;
            let linear: Vec<u8> = (0..pitch * (region.height * region.depth) as usize)
                .map(|_| rng.next() as u8)
                .collect();
            let mut tiled = vec![0u8; t.size_bytes()];
            swizzle_region(&t, region, &linear, &mut tiled, Some(pitch));
            let mut written = 0;
            for dz in 0..region.depth {
                for dy in 0..region.height {
                    for dx in 0..region.width * t.bytes_per_block {
                        let src = (dz * region.height + dy) as usize * pitch + dx as usize;
                        let dst = reference_offset(
                            &t,
                            region.x * t.bytes_per_block + dx,
                            region.y + dy,
                            region.z + dz,
                        );
                        assert_eq!(tiled[dst], linear[src]);
                        written += 1;
                    }
                }
            }
            let nonzero_outside = tiled.iter().filter(|&&b| b != 0).count();
            assert!(nonzero_outside <= written);
            let mut out = vec![0u8; linear.len()];
            unswizzle_region(&t, region, &tiled, &mut out, Some(pitch));
            for dz in 0..region.depth {
                for dy in 0..region.height {
                    let start = (dz * region.height + dy) as usize * pitch;
                    let len = (region.width * t.bytes_per_block) as usize;
                    assert_eq!(out[start..start + len], linear[start..start + len]);
                }
            }
        }
    }

    #[test]
    fn switch2_swaps_bits_5_and_6() {
        for y in 0..8 {
            for x in 0..64 {
                let tegra = gob_offset(x, y, GobLayout::Tegra);
                let switch2 = gob_offset(x, y, GobLayout::Switch2);
                let swapped = (tegra & !0x60) | ((tegra >> 5) & 1) << 6 | ((tegra >> 6) & 1) << 5;
                assert_eq!(switch2, swapped);
            }
        }
        assert_eq!(gob_offset(16, 0, GobLayout::Tegra), 0x20);
        assert_eq!(gob_offset(16, 0, GobLayout::Switch2), 0x40);
        assert_eq!(gob_offset(0, 2, GobLayout::Switch2), 0x20);
    }

    fn layout(
        format: PixelFormat,
        size: (u32, u32, u32),
        levels: u32,
        layers: u32,
        bh: u32,
    ) -> SurfaceLayout {
        SurfaceLayout {
            format,
            width: size.0,
            height: size.1,
            depth: size.2,
            levels,
            layers,
            block_height_log2: bh,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            layout: GobLayout::Tegra,
        }
    }

    #[test]
    fn mip_chain_sizes_follow_the_block_shrinking_rule() {
        let l = layout(PixelFormat::A8B8G8R8Unorm, (256, 256, 1), 9, 1, 4);
        let sizes: Vec<u64> = (0..9).map(|level| l.level_size(level)).collect();
        // 256x256 RGBA8 is 16 GOBs wide and 32 tall: 2 blocks of 16 GOBs = 256 KiB. Level 1
        // (128x128, 16 rows of GOBs) keeps block height 16, level 2 shrinks it to 8, level 3
        // to 4: every level fills its blocks exactly until rows drop below one GOB.
        assert_eq!(sizes[0], 256 * 256 * 4);
        assert_eq!(sizes[1], 128 * 128 * 4);
        assert_eq!(sizes[2], 64 * 64 * 4);
        assert_eq!(sizes[3], 32 * 32 * 4);
        assert_eq!(sizes[4], 512 * 2);
        assert_eq!(&sizes[5..], &[512, 512, 512, 512]);
        assert_eq!(l.level_offset(2), sizes[0] + sizes[1]);
        assert_eq!(
            l.layer_stride(),
            sizes.iter().sum::<u64>().div_ceil(8192) * 8192
        );
        assert_eq!(l.level_tiling(2).block_height_log2, 3);
        assert_eq!(l.level_tiling(4).block_height_log2, 1);
        assert_eq!(l.level_tiling(5).block_height_log2, 0);
    }

    #[test]
    fn single_level_surfaces_keep_the_programmed_block_height() {
        let l = layout(PixelFormat::A8B8G8R8Unorm, (16, 16, 1), 1, 1, 4);
        assert_eq!(l.level_size(0), 16 * 512);
        assert_eq!(l.level_tiling(0).block_height_log2, 1);
    }

    #[test]
    fn compressed_levels_count_blocks() {
        let l = layout(PixelFormat::Bc1RgbaUnorm, (1280, 720, 1), 1, 1, 4);
        let t = l.level_tiling(0);
        assert_eq!((t.width, t.height, t.bytes_per_block), (320, 180, 8));
        assert_eq!(t.row_gobs, 40);
        assert_eq!(l.level_size(0), 40 * 2 * 16 * 512);
        let a = layout(PixelFormat::Astc8x8Srgb, (1920, 1080, 1), 1, 6, 3);
        assert_eq!(a.level_tiling(0).width, 240);
        assert_eq!(a.size_bytes(), a.layer_stride() * 6);
        assert_eq!(a.subresource_offset(0, 2), a.layer_stride() * 2);
    }

    #[test]
    fn compressed_layers_align_to_block_rows() {
        let l = layout(PixelFormat::Bc1RgbaUnorm, (64, 64, 1), 7, 2, 4);
        assert_eq!(l.level_offset(7), 2048 + 6 * 512);
        assert_eq!(l.layer_stride(), 5120);
        assert_eq!(l.subresource_offset(0, 1), 5120);
    }

    #[test]
    fn tile_width_spacing_aligns_rows_and_layers() {
        let mut l = layout(PixelFormat::R8Unorm, (300, 300, 1), 1, 2, 2);
        let plain = l.level_tiling(0).row_gobs;
        l.tile_width_spacing = 2;
        assert_eq!(plain, 5);
        assert_eq!(l.level_tiling(0).row_gobs, 8);
        assert_eq!(l.layer_stride() % (512 << 4), 0);
    }
}
