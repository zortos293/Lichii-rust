use lichii_common::{DAddr, GpuVAddr};
use lichii_texture::block_linear::{
    GobLayout, Region, Tiling, gob_offset, swizzle, unswizzle_region,
};

use super::regs::RemapConstSwizzle as Swizzle;
use super::*;
use crate::backend::{
    ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GraphicsContext,
    ImageCopy, IndirectDraw, Rasterizer,
};
use crate::engines::testing::Harness;
use crate::gmmu::Gmmu;

#[derive(Default)]
struct Double {
    accept_buffer: bool,
    accept_image: bool,
    accept_clear: bool,
    buffer_copies: Vec<(GpuVAddr, GpuVAddr, u64)>,
    buffer_clears: Vec<(GpuVAddr, u64, u32)>,
    image_copies: Vec<ImageCopy>,
    queries: Vec<Query>,
    flushed: Vec<(DAddr, u64)>,
    invalidated: Vec<(DAddr, u64)>,
}

impl Rasterizer for Double {
    fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}
    fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}
    fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}
    fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}
    fn accelerate_buffer_copy(
        &mut self,
        _channel: ChannelId,
        _memory: &Gmmu,
        src: GpuVAddr,
        dst: GpuVAddr,
        size: u64,
    ) -> bool {
        self.buffer_copies.push((src, dst, size));
        self.accept_buffer
    }
    fn accelerate_buffer_clear(
        &mut self,
        _channel: ChannelId,
        _memory: &Gmmu,
        dst: GpuVAddr,
        size: u64,
        value: u32,
    ) -> bool {
        self.buffer_clears.push((dst, size, value));
        self.accept_clear
    }
    fn accelerate_image_copy(
        &mut self,
        _channel: ChannelId,
        _memory: &Gmmu,
        copy: &ImageCopy,
    ) -> bool {
        self.image_copies.push(*copy);
        self.accept_image
    }
    fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, query: &Query) {
        self.queries.push(*query);
    }
    fn signal_fence(&mut self, _action: FenceAction) {}
    fn release_fences(&mut self, _wait: bool) {}
    fn invalidate_region(&mut self, addr: DAddr, size: u64) {
        self.invalidated.push((addr, size));
    }
    fn flush_region(&mut self, addr: DAddr, size: u64) {
        self.flushed.push((addr, size));
    }
    fn wait_for_idle(&mut self) {}
    fn present(&mut self, _framebuffer: &FramebufferConfig) {}
}

const PITCH: u32 = 1;
const BLOCK: u32 = 0;

struct Rig {
    h: Harness,
    double: Double,
    engine: MaxwellDma,
}

struct Surface {
    block_height: u32,
    block_depth: u32,
    width: u32,
    height: u32,
    depth: u32,
    layer: u32,
    origin: (u32, u32),
}

impl Surface {
    fn new(width: u32, height: u32, block_height: u32) -> Self {
        Self {
            block_height,
            block_depth: 0,
            width,
            height,
            depth: 1,
            layer: 0,
            origin: (0, 0),
        }
    }

    fn at(mut self, x: u32, y: u32) -> Self {
        self.origin = (x, y);
        self
    }

    fn tiling(&self, layout: GobLayout) -> Tiling {
        Tiling::new(
            self.width,
            self.height,
            self.depth,
            1,
            self.block_height,
            self.block_depth,
            layout,
        )
    }
}

fn launch_word(src_layout: u32, dst_layout: u32, multi_line: bool, remap: bool) -> u32 {
    2 | src_layout << 7 | dst_layout << 8 | u32::from(multi_line) << 9 | u32::from(remap) << 10
}

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u32).wrapping_mul(31).wrapping_add(seed) as u8 ^ (i >> 8) as u8)
        .collect()
}

impl Rig {
    fn new(layout: GobLayout) -> Self {
        let mut h = Harness::new();
        h.gob_layout = layout;
        Self {
            h,
            double: Double::default(),
            engine: MaxwellDma::new(),
        }
    }

    fn call(&mut self, method: u32, value: u32) {
        let mut ctx = EngineContext {
            channel: ChannelId(0),
            memory: &self.h.memory,
            rasterizer: &mut self.double,
            syncpoints: &self.h.syncpoints,
            gob_layout: self.h.gob_layout,
            qmd_scheduler: &mut self.h.qmd_scheduler,
            written_pages: &self.h.written_pages,
        };
        self.engine.call_method(&mut ctx, method, value);
    }

    fn addresses(&mut self, src: u64, dst: u64) {
        self.call(reg_index!(Regs, offset_in.upper), (src >> 32) as u32);
        self.call(reg_index!(Regs, offset_in.lower), src as u32);
        self.call(reg_index!(Regs, offset_out.upper), (dst >> 32) as u32);
        self.call(reg_index!(Regs, offset_out.lower), dst as u32);
    }

    fn lines(&mut self, length: u32, count: u32) {
        self.call(reg_index!(Regs, line_length_in), length);
        self.call(reg_index!(Regs, line_count), count);
    }

    fn pitches(&mut self, pitch_in: u32, pitch_out: u32) {
        self.call(reg_index!(Regs, pitch_in), pitch_in);
        self.call(reg_index!(Regs, pitch_out), pitch_out);
    }

    fn surface(&mut self, destination: bool, s: &Surface) {
        let base = if destination {
            reg_index!(Regs, dst_params)
        } else {
            reg_index!(Regs, src_params)
        };
        self.call(base, s.block_height << 4 | s.block_depth << 8);
        self.call(base + 1, s.width);
        self.call(base + 2, s.height);
        self.call(base + 3, s.depth);
        self.call(base + 4, s.layer);
        self.call(base + 5, s.origin.0 | s.origin.1 << 16);
    }

    fn remap(&mut self, consts: [u32; 2], dst: [Swizzle; 4], size: u32, src: u32, count: u32) {
        let base = reg_index!(Regs, remap_const);
        self.call(base, consts[0]);
        self.call(base + 1, consts[1]);
        let swizzles = dst[0].0 | dst[1].0 << 4 | dst[2].0 << 8 | dst[3].0 << 12;
        self.call(
            base + 2,
            swizzles | (size - 1) << 16 | (src - 1) << 20 | (count - 1) << 24,
        );
    }

    fn launch(&mut self, word: u32) {
        self.call(reg_index!(Regs, launch_dma), word);
    }

    fn write(&self, address: GpuVAddr, data: &[u8]) {
        self.h.memory.write_block(address, data).unwrap();
    }

    fn read(&self, address: GpuVAddr, len: usize) -> Vec<u8> {
        let mut out = vec![0; len];
        self.h.memory.read_block(address, &mut out).unwrap();
        out
    }

    fn daddr(&self, address: GpuVAddr) -> DAddr {
        self.h.memory.translate(address).unwrap()
    }
}

const LAYOUTS: [GobLayout; 2] = [GobLayout::Tegra, GobLayout::Switch2];

#[test]
fn linear_copy_runs_in_software_when_the_renderer_refuses() {
    let mut rig = Rig::new(GobLayout::Tegra);
    let data = pattern(100, 1);
    rig.write(0x1000, &data);
    rig.addresses(0x1000, 0x2000);
    rig.lines(100, 1);
    rig.launch(launch_word(PITCH, PITCH, false, false));
    assert_eq!(rig.double.buffer_copies, [(0x1000, 0x2000, 100)]);
    assert_eq!(rig.read(0x2000, 100), data);
    assert_eq!(rig.double.flushed, [(rig.daddr(0x1000), 100)]);
    assert_eq!(rig.double.invalidated, [(rig.daddr(0x2000), 100)]);
}

#[test]
fn accelerated_linear_copy_leaves_memory_alone() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.double.accept_buffer = true;
    rig.write(0x1000, &pattern(64, 2));
    rig.addresses(0x1000, 0x2000);
    rig.lines(64, 1);
    rig.launch(launch_word(PITCH, PITCH, false, false));
    assert_eq!(rig.double.buffer_copies, [(0x1000, 0x2000, 64)]);
    assert_eq!(rig.read(0x2000, 64), vec![0; 64]);
    assert!(rig.double.flushed.is_empty() && rig.double.invalidated.is_empty());
}

#[test]
fn overlapping_linear_copy_moves_like_memmove() {
    let mut rig = Rig::new(GobLayout::Tegra);
    let len = 0x18_0000;
    let data = pattern(len, 3);
    rig.write(0x1000, &data);
    rig.addresses(0x1000, 0x1100);
    rig.lines(len as u32, 1);
    rig.launch(launch_word(PITCH, PITCH, false, false));
    assert_eq!(rig.read(0x1100, len), data);
}

#[test]
fn multi_line_pitch_copy_honours_both_pitches() {
    let mut rig = Rig::new(GobLayout::Tegra);
    let rows: Vec<Vec<u8>> = (0..4).map(|row| pattern(10, row)).collect();
    for (row, data) in rows.iter().enumerate() {
        rig.write(0x1000 + row as u64 * 16, data);
    }
    rig.write(0x2000, &[0xee; 0x80]);
    rig.addresses(0x1000, 0x2000);
    rig.pitches(16, 32);
    rig.lines(10, 4);
    rig.launch(launch_word(PITCH, PITCH, true, false));
    for (row, data) in rows.iter().enumerate() {
        let at = 0x2000 + row as u64 * 32;
        assert_eq!(&rig.read(at, 10), data);
        assert_eq!(rig.read(at + 10, 22), vec![0xee; 22]);
    }
    assert_eq!(rig.double.buffer_copies.len(), 4);
}

#[test]
fn tightly_packed_pitch_copy_is_one_buffer_copy() {
    let mut rig = Rig::new(GobLayout::Tegra);
    let data = pattern(96, 4);
    rig.write(0x1000, &data);
    rig.addresses(0x1000, 0x3000);
    rig.pitches(24, 24);
    rig.lines(24, 4);
    rig.launch(launch_word(PITCH, PITCH, true, false));
    assert_eq!(rig.double.buffer_copies, [(0x1000, 0x3000, 96)]);
    assert_eq!(rig.read(0x3000, 96), data);
}

fn fill_pitch(rig: &Rig, address: GpuVAddr, pitch: u64, rows: &[Vec<u8>]) {
    for (row, data) in rows.iter().enumerate() {
        rig.write(address + row as u64 * pitch, data);
    }
}

#[test]
fn pitch_to_block_linear_and_back_round_trips() {
    for layout in LAYOUTS {
        let mut rig = Rig::new(layout);
        let rows: Vec<Vec<u8>> = (0..20).map(|row| pattern(100, row)).collect();
        fill_pitch(&rig, 0x1000, 128, &rows);
        let surface = Surface::new(100, 20, 1);
        rig.addresses(0x1000, 0x1_0000);
        rig.pitches(128, 0);
        rig.lines(100, 20);
        rig.surface(true, &surface);
        rig.launch(launch_word(PITCH, BLOCK, true, false));
        let tiling = surface.tiling(layout);
        let linear: Vec<u8> = rows.concat();
        let mut expected = vec![0; tiling.size_bytes()];
        swizzle(&tiling, &linear, &mut expected);
        assert_eq!(rig.read(0x1_0000, expected.len()), expected, "{layout:?}");
        assert!(!rig.double.image_copies.is_empty());

        rig.addresses(0x1_0000, 0x4000);
        rig.pitches(0, 112);
        rig.surface(false, &surface);
        rig.launch(launch_word(BLOCK, PITCH, true, false));
        for (row, data) in rows.iter().enumerate() {
            assert_eq!(
                &rig.read(0x4000 + row as u64 * 112, 100),
                data,
                "{layout:?}"
            );
        }
    }
}

#[test]
fn pitch_to_block_linear_writes_only_the_rectangle_at_its_origin() {
    for layout in LAYOUTS {
        let mut rig = Rig::new(layout);
        let rows: Vec<Vec<u8>> = (0..4).map(|row| pattern(16, 0x40 + row)).collect();
        fill_pitch(&rig, 0x1000, 16, &rows);
        let surface = Surface::new(64, 16, 1).at(20, 5);
        rig.addresses(0x1000, 0x2_0000);
        rig.pitches(16, 0);
        rig.lines(16, 4);
        rig.surface(true, &surface);
        rig.launch(launch_word(PITCH, BLOCK, true, false));
        let tiling = surface.tiling(layout);
        let tiled = rig.read(0x2_0000, tiling.size_bytes());
        for y in 0..16u32 {
            for x in 0..64u32 {
                let inside = (20..36).contains(&x) && (5..9).contains(&y);
                let expected = if inside {
                    rows[(y - 5) as usize][(x - 20) as usize]
                } else {
                    0
                };
                assert_eq!(
                    tiled[tiling.offset(x, y, 0)],
                    expected,
                    "{layout:?} ({x},{y})"
                );
            }
        }
    }
}

#[test]
fn block_linear_to_block_linear_copies_a_sub_rectangle() {
    for layout in LAYOUTS {
        let mut rig = Rig::new(layout);
        let src_surface = Surface::new(128, 40, 0).at(10, 3);
        let dst_surface = Surface::new(96, 64, 2).at(30, 17);
        let src_tiling = src_surface.tiling(layout);
        let source = pattern(128 * 40, 9);
        let mut tiled = vec![0; src_tiling.size_bytes()];
        swizzle(&src_tiling, &source, &mut tiled);
        rig.write(0x2_0000, &tiled);
        rig.addresses(0x2_0000, 0x4_0000);
        rig.lines(50, 20);
        rig.surface(false, &src_surface);
        rig.surface(true, &dst_surface);
        rig.launch(launch_word(BLOCK, BLOCK, true, false));
        let dst_tiling = dst_surface.tiling(layout);
        let written = rig.read(0x4_0000, dst_tiling.size_bytes());
        let mut copied = vec![0; 50 * 20];
        let region = Region {
            x: 30,
            y: 17,
            z: 0,
            width: 50,
            height: 20,
            depth: 1,
        };
        unswizzle_region(&dst_tiling, region, &written, &mut copied, None);
        for row in 0..20usize {
            let from = (3 + row) * 128 + 10;
            assert_eq!(
                copied[row * 50..row * 50 + 50],
                source[from..from + 50],
                "{layout:?}"
            );
        }
        let outside = written[dst_tiling.offset(29, 17, 0)];
        assert_eq!(outside, 0);
    }
}

#[test]
fn image_copy_is_offered_in_pixels_of_the_widest_alignment() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.double.accept_image = true;
    rig.addresses(0x1000, 0x2_0000);
    rig.pitches(256, 0);
    rig.lines(256, 4);
    rig.surface(true, &Surface::new(256, 16, 1).at(32, 2));
    rig.launch(launch_word(PITCH, BLOCK, true, false));
    let copy = rig.double.image_copies[0];
    assert_eq!((copy.width, copy.height), (16, 4));
    assert_eq!(copy.src.bytes_per_pixel, 16);
    assert_eq!(
        (copy.src.address, copy.src.pitch, copy.src.block),
        (0x1000, 256, None)
    );
    assert_eq!(copy.dst.width, 16);
    assert_eq!(copy.dst.block, Some((0, 1, 0)));
    assert_eq!(copy.dst_origin, (2, 2));
    assert_eq!(rig.read(0x2_0000, 0x1000), vec![0; 0x1000]);
}

#[test]
fn constant_fill_writes_the_component_size() {
    for (size, expected) in [(4, vec![0x44, 0x33, 0x22, 0x11]), (2, vec![0x44, 0x33])] {
        let mut rig = Rig::new(GobLayout::Tegra);
        rig.write(0x2000, &[0xee; 64]);
        rig.addresses(0, 0x2000);
        rig.lines(8, 1);
        rig.remap(
            [0x1122_3344, 0],
            [
                Swizzle::CONST_A,
                Swizzle::NO_WRITE,
                Swizzle::NO_WRITE,
                Swizzle::NO_WRITE,
            ],
            size,
            1,
            1,
        );
        rig.launch(launch_word(PITCH, PITCH, false, true));
        let written = size as usize * 8;
        assert_eq!(rig.read(0x2000, written), expected.repeat(8));
        assert_eq!(rig.read(0x2000 + written as u64, 4), [0xee; 4]);
        assert!(rig.double.buffer_copies.is_empty());
        assert_eq!(
            rig.double.invalidated,
            [(rig.daddr(0x2000), written as u64)]
        );
        let offered: &[_] = if size == 4 {
            &[(0x2000, 32, 0x1122_3344)]
        } else {
            &[]
        };
        assert_eq!(rig.double.buffer_clears, offered);
    }
}

#[test]
fn an_accepted_buffer_clear_skips_the_software_fill() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.double.accept_clear = true;
    rig.write(0x2000, &[0xee; 64]);
    rig.addresses(0, 0x2000);
    rig.lines(4, 1);
    rig.remap(
        [0, 0xdead_beef],
        [
            Swizzle::CONST_B,
            Swizzle::CONST_B,
            Swizzle::NO_WRITE,
            Swizzle::NO_WRITE,
        ],
        4,
        1,
        2,
    );
    rig.launch(launch_word(PITCH, PITCH, false, true));
    assert_eq!(rig.double.buffer_clears, [(0x2000, 32, 0xdead_beef)]);
    assert_eq!(rig.read(0x2000, 64), [0xee; 64]);
}

#[test]
fn oversized_remapped_rows_move_through_bounded_scratch() {
    let mut rig = Rig::new(GobLayout::Tegra);
    let length = 0x4_0001_usize;
    let (src_pitch, dst_pitch) = (length * 4, length * 2);
    let src = pattern(src_pitch * 2, 7);
    rig.write(0, &src);
    rig.addresses(0, 0x28_0000);
    rig.pitches(src_pitch as u32, dst_pitch as u32);
    rig.lines(length as u32, 2);
    rig.remap(
        [0, 0],
        [
            Swizzle::SRC_W,
            Swizzle::SRC_X,
            Swizzle::NO_WRITE,
            Swizzle::NO_WRITE,
        ],
        1,
        4,
        2,
    );
    rig.launch(launch_word(PITCH, PITCH, true, true));
    let expected: Vec<u8> = (0..2)
        .flat_map(|row| {
            let row = &src[row * src_pitch..][..src_pitch];
            row.as_chunks::<4>().0.iter().flat_map(|e| [e[3], e[0]])
        })
        .collect();
    assert_eq!(rig.read(0x28_0000, expected.len()), expected);
    assert!(rig.engine.scratch.packed.capacity() <= copy::CHUNK);
    assert!(rig.engine.scratch.converted.capacity() <= copy::CHUNK);
}

#[test]
fn constant_fill_ignores_the_unused_source_surface() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.write(0x2000, &[0xee; 256]);
    rig.addresses(0, 0x2000);
    rig.pitches(0, 32);
    rig.lines(8, 4);
    rig.remap(
        [0x1122_3344, 0],
        [
            Swizzle::CONST_A,
            Swizzle::NO_WRITE,
            Swizzle::NO_WRITE,
            Swizzle::NO_WRITE,
        ],
        4,
        1,
        1,
    );
    rig.surface(false, &Surface::new(0, 0, 1));
    rig.launch(launch_word(BLOCK, PITCH, true, true));
    for row in 0..4 {
        assert_eq!(
            rig.read(0x2000 + row * 32, 32),
            [0x44, 0x33, 0x22, 0x11].repeat(8)
        );
    }
    assert_eq!(rig.read(0x2000 + 4 * 32, 4), [0xee; 4]);
}

#[test]
fn remap_reorders_components_and_fills_constants_across_lines() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.write(0x1000, &[1, 2, 3, 4, 5, 6, 7, 8]);
    rig.write(0x1010, &[9, 10, 11, 12, 13, 14, 15, 16]);
    rig.addresses(0x1000, 0x2000);
    rig.pitches(16, 8);
    rig.lines(2, 2);
    rig.remap(
        [0xaa, 0],
        [
            Swizzle::SRC_Z,
            Swizzle::SRC_Y,
            Swizzle::SRC_X,
            Swizzle::CONST_A,
        ],
        1,
        4,
        4,
    );
    rig.launch(launch_word(PITCH, PITCH, true, true));
    assert_eq!(
        rig.read(0x2000, 16),
        [
            3, 2, 1, 0xaa, 7, 6, 5, 0xaa, 11, 10, 9, 0xaa, 15, 14, 13, 0xaa
        ]
    );
    assert!(rig.double.image_copies.is_empty());
}

#[test]
fn remap_leaves_unwritten_components_and_changes_element_width() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.write(0x1000, &[0x11, 0x22, 0x33, 0x44]);
    rig.write(0x2000, &[0x77; 16]);
    rig.addresses(0x1000, 0x2000);
    rig.pitches(4, 6);
    rig.lines(2, 1);
    rig.remap(
        [0, 0x5a],
        [
            Swizzle::SRC_Y,
            Swizzle::NO_WRITE,
            Swizzle::CONST_B,
            Swizzle::SRC_X,
        ],
        1,
        2,
        3,
    );
    rig.launch(launch_word(PITCH, PITCH, true, true));
    assert_eq!(
        rig.read(0x2000, 8),
        [0x22, 0x77, 0x5a, 0x44, 0x77, 0x5a, 0x77, 0x77]
    );
}

#[test]
fn identity_remap_is_offered_as_an_image_copy_of_wide_pixels() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.double.accept_image = true;
    rig.addresses(0x1000, 0x2_0000);
    rig.pitches(64, 0);
    rig.lines(16, 8);
    rig.surface(true, &Surface::new(16, 8, 0));
    rig.remap(
        [0, 0],
        [
            Swizzle::SRC_X,
            Swizzle::SRC_Y,
            Swizzle::SRC_Z,
            Swizzle::SRC_W,
        ],
        1,
        4,
        4,
    );
    rig.launch(launch_word(PITCH, BLOCK, true, true));
    let copy = rig.double.image_copies[0];
    assert_eq!((copy.width, copy.src.bytes_per_pixel), (16, 4));
}

#[test]
fn flat_copy_addresses_block_linear_kinds_in_gob_raster_order() {
    for layout in LAYOUTS {
        let mut rig = Rig::new(layout);
        rig.h.memory.set_kind(0x1_0000, 0x1000, 0xfe).unwrap();
        let raster = pattern(1024, 5);
        let mut tiled = vec![0; 1024];
        for (at, byte) in raster.iter().enumerate() {
            let gob = at & !511;
            let x = (at & 63) as u32;
            let y = ((at >> 6) & 7) as u32;
            tiled[gob + gob_offset(x, y, layout) as usize] = *byte;
        }
        rig.write(0x1_0000, &tiled);
        rig.addresses(0x1_0000, 0x2000);
        rig.lines(1024, 1);
        rig.launch(launch_word(PITCH, PITCH, false, false));
        assert_eq!(rig.read(0x2000, 1024), raster, "{layout:?} untile");

        rig.addresses(0x2000, 0x1_1000);
        rig.h.memory.set_kind(0x1_1000, 0x1000, 0xfe).unwrap();
        rig.h.memory.set_kind(0x1_0000, 0x1000, 0).unwrap();
        rig.lines(1024, 1);
        rig.launch(launch_word(PITCH, PITCH, false, false));
        assert_eq!(rig.read(0x1_1000, 1024), tiled, "{layout:?} tile");
    }
}

#[test]
fn semaphore_release_follows_the_copy() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.write(0x1000, &[1, 2, 3, 4]);
    rig.addresses(0x1000, 0x2000);
    rig.lines(4, 1);
    rig.call(reg_index!(Regs, semaphore.address.upper), 0);
    rig.call(reg_index!(Regs, semaphore.address.lower), 0x3000);
    rig.call(reg_index!(Regs, semaphore.payload), 0xdead_beef);
    rig.launch(launch_word(PITCH, PITCH, false, false) | 1 << 3);
    rig.launch(launch_word(PITCH, PITCH, false, false) | 2 << 3);
    assert_eq!(rig.read(0x2000, 4), [1, 2, 3, 4]);
    let release = |size| Query {
        address: 0x3000,
        kind: QueryKind::Payload,
        payload: 0xdead_beef,
        size,
        after_work: true,
    };
    assert_eq!(
        rig.double.queries,
        [release(ReleaseSize::Four), release(ReleaseSize::Sixteen)]
    );
}

#[test]
fn launch_without_transfer_only_releases_the_semaphore() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.write(0x1000, &[9; 4]);
    rig.addresses(0x1000, 0x2000);
    rig.lines(4, 1);
    rig.launch(1 << 3);
    assert_eq!(rig.read(0x2000, 4), [0; 4]);
    assert_eq!(rig.double.queries.len(), 1);
    rig.launch(0);
    assert_eq!(rig.double.queries.len(), 1);
}

#[test]
fn malformed_surfaces_are_ignored() {
    let mut rig = Rig::new(GobLayout::Tegra);
    rig.addresses(0x1000, 0x2_0000);
    rig.pitches(64, 0);
    rig.lines(64, 4);
    for surface in [
        Surface::new(u32::MAX, 16, 1),
        Surface::new(32, 16, 1),
        Surface::new(64, 0, 1),
        Surface::new(64, 16, 15),
    ] {
        rig.surface(true, &surface);
        rig.launch(launch_word(PITCH, BLOCK, true, false));
    }
    rig.surface(true, &Surface::new(64, 16, 1).at(0, 16));
    rig.launch(launch_word(PITCH, BLOCK, true, false));
    assert_eq!(rig.read(0x2_0000, 0x1000), vec![0; 0x1000]);
    rig.call(NUM_REGS_PAST_END, 1);
}

const NUM_REGS_PAST_END: u32 = regs::NUM_REGS as u32 + 4;
