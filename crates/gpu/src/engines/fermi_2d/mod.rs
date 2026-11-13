//! The 2D engine (Fermi 902D).
//!
//! The only method that does work is the high word of `pixels_from_memory.src_y0`: it blits the
//! source surface into the destination surface. The engine builds the backend's [`Blit`] from
//! the registers and offers it to the renderer; when the renderer declines it runs the blit in
//! software (`blitter`) through the GMMU.
//!
//! Supported as in the reference: `SRC_COPY` blits without clipping, through the render-target
//! formats in `converter`. Other operations, clipping, the pixels-from-CPU path and solid
//! primitives are not executed; the software path ignores the operation, as the reference does.

mod blitter;
mod converter;
pub mod regs;

use lichii_texture::PixelFormat;
use lichii_texture::guest::RenderTargetFormat;

use self::blitter::{Job, Rect, Sampling, SoftwareBlitter};
use super::EngineContext;
use crate::backend::{Blit, CopySurface};
use crate::reg_index;
pub use regs::Regs;
use regs::{Filter, MemoryLayout, Operation, Origin, Surface};

/// The 2D engine's registers and the software blitter's scratch buffers.
pub struct Fermi2d {
    pub regs: Box<Regs>,
    blitter: SoftwareBlitter,
}

impl Default for Fermi2d {
    fn default() -> Self {
        Self::new()
    }
}

impl Fermi2d {
    pub fn new() -> Self {
        let mut regs: Box<Regs> = bytemuck::zeroed_box();
        regs.src.depth = 1;
        regs.dst.depth = 1;
        Self {
            regs,
            blitter: SoftwareBlitter::default(),
        }
    }

    /// Stores `argument` in register `method`; the high word of `src_y0` runs the blit.
    pub fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        let Some(word) =
            bytemuck::cast_slice_mut::<Regs, u32>(std::slice::from_mut(&mut *self.regs))
                .get_mut(method as usize)
        else {
            return;
        };
        *word = argument;
        if method == reg_index!(Regs, pixels_from_memory.src_y0) + 1 {
            self.blit(ctx);
        }
    }

    fn blit(&mut self, ctx: &mut EngineContext<'_>) {
        let regs = &*self.regs;
        let args = &regs.pixels_from_memory;
        if regs.operation != Operation::SRC_COPY {
            tracing::debug!(operation = ?regs.operation, "2D blit operation is not a plain copy");
        }
        let (Some((src_format, src_pixel)), Some((dst_format, dst_pixel))) = (
            decode_format(regs.src.format),
            decode_format(regs.dst.format),
        ) else {
            tracing::warn!(
                src = regs.src.format,
                dst = regs.dst.format,
                "2D blit with an undecodable surface format"
            );
            return;
        };
        let mut src = copy_surface(&regs.src, src_pixel.bytes_per_block());
        let dst = copy_surface(&regs.dst, dst_pixel.bytes_per_block());

        let (mut src_x, mut src_y) = (args.src_x0, args.src_y0);
        if args.sample_mode.origin() == Origin::CORNER {
            src_x = src_x.wrapping_sub(args.du_dx >> 1);
            src_y = src_y.wrapping_sub(args.dv_dy >> 1);
        }
        let end = |start: i64, step: i64, count: i32| {
            (start.wrapping_add(step.wrapping_mul(i64::from(count))) >> 32) as i32
        };
        let mut src_rect = Rect {
            x0: (src_x >> 32) as i32,
            y0: (src_y >> 32) as i32,
            x1: end(src_x, args.du_dx, args.dst_width),
            y1: end(src_y, args.dv_dy, args.dst_height),
        };
        let dst_rect = Rect {
            x0: args.dst_x0,
            y0: args.dst_y0,
            x1: args.dst_x0.wrapping_add(args.dst_width),
            y1: args.dst_y0.wrapping_add(args.dst_height),
        };

        let bytes_per_pixel = src.bytes_per_pixel;
        if src.block.is_none()
            && src.width as i32 == src_rect.x1
            && src_rect.x1 > (src.pitch / bytes_per_pixel) as i32
            && src_rect.x0 > 0
        {
            let x0 = src_rect.x0;
            src.address = src
                .address
                .wrapping_add(u64::from(x0 as u32) * u64::from(bytes_per_pixel));
            src.width -= x0 as u32;
            src_rect.x1 -= x0;
            src_rect.x0 = 0;
            src_x = src_x.wrapping_sub(i64::from(x0) << 32);
        }

        let linear_filter = args.sample_mode.filter() == Filter::BILINEAR;
        let blit = Blit {
            src,
            dst,
            src_format: src_pixel,
            dst_format: dst_pixel,
            src_x0: src_x,
            src_y0: src_y,
            du_dx: args.du_dx,
            dv_dy: args.dv_dy,
            dst_x0: dst_rect.x0,
            dst_y0: dst_rect.y0,
            dst_width: args.dst_width,
            dst_height: args.dst_height,
            linear_filter,
        };
        if ctx
            .rasterizer
            .accelerate_blit(ctx.channel, ctx.memory, &blit)
        {
            return;
        }
        self.blitter.blit(
            ctx,
            &Job {
                src,
                dst,
                src_format,
                dst_format,
                src_rect,
                dst_rect,
                sampling: Sampling {
                    x0: src_x,
                    y0: src_y,
                    du_dx: args.du_dx,
                    dv_dy: args.dv_dy,
                },
                linear_filter,
            },
        );
    }
}

fn decode_format(raw: u32) -> Option<(RenderTargetFormat, PixelFormat)> {
    let format = RenderTargetFormat::try_from(raw).ok()?;
    Some((format, PixelFormat::from_render_target(format)?))
}

fn copy_surface(surface: &Surface, bytes_per_pixel: u32) -> CopySurface {
    CopySurface {
        address: u64::from(surface.addr_upper) << 32 | u64::from(surface.addr_lower),
        width: surface.width,
        height: surface.height,
        depth: surface.depth.max(1),
        layer: surface.layer,
        pitch: surface.pitch,
        block: (surface.linear == MemoryLayout::BLOCK_LINEAR).then(|| {
            (
                surface.block_width(),
                surface.block_height(),
                surface.block_depth(),
            )
        }),
        bytes_per_pixel,
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::{DAddr, GpuVAddr};
    use lichii_texture::block_linear::{Tiling, swizzle, unswizzle};
    use lichii_texture::guest::RenderTargetFormat as F;

    use super::*;
    use crate::backend::{
        ChannelId, ComputeContext, DrawParams, FenceAction, FramebufferConfig, GraphicsContext,
        IndirectDraw, Query, Rasterizer,
    };
    use crate::engines::testing::Harness;
    use crate::gmmu::Gmmu;

    #[derive(Default)]
    struct Accelerator {
        accept: bool,
        blits: Vec<Blit>,
        flushes: Vec<(DAddr, u64)>,
        inline_writes: Vec<(GpuVAddr, u64)>,
    }

    impl Rasterizer for Accelerator {
        fn draw(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &DrawParams<'_>) {}
        fn draw_indirect(&mut self, _ctx: &mut GraphicsContext<'_>, _draw: &IndirectDraw) {}
        fn clear(&mut self, _ctx: &mut GraphicsContext<'_>, _layer_count: u32) {}
        fn dispatch_compute(&mut self, _ctx: &ComputeContext<'_>) {}
        fn accelerate_blit(&mut self, _channel: ChannelId, _memory: &Gmmu, blit: &Blit) -> bool {
            self.blits.push(*blit);
            self.accept
        }
        fn inline_written(&mut self, _c: ChannelId, _m: &Gmmu, address: GpuVAddr, size: u64) {
            self.inline_writes.push((address, size));
        }
        fn query(&mut self, _channel: ChannelId, _memory: &Gmmu, _query: &Query) {}
        fn signal_fence(&mut self, _action: FenceAction) {}
        fn release_fences(&mut self, _wait: bool) {}
        fn invalidate_region(&mut self, _addr: DAddr, _size: u64) {}
        fn flush_region(&mut self, addr: DAddr, size: u64) {
            self.flushes.push((addr, size));
        }
        fn wait_for_idle(&mut self) {}
        fn present(&mut self, _framebuffer: &FramebufferConfig) {}
    }

    #[derive(Clone, Copy)]
    struct Surf {
        format: F,
        address: u64,
        width: u32,
        height: u32,
        /// Pitch in bytes, or the log2 block height in GOBs for a block-linear surface.
        layout: Layout,
    }

    #[derive(Clone, Copy)]
    enum Layout {
        Pitch(u32),
        Block(u32),
    }

    struct Setup {
        src: Surf,
        dst: Surf,
        src_x0: i64,
        src_y0: i64,
        du_dx: i64,
        dv_dy: i64,
        dst_rect: (i32, i32, i32, i32),
        bilinear: bool,
        corner: bool,
    }

    impl Setup {
        fn copy(src: Surf, dst: Surf, size: (u32, u32)) -> Self {
            Self {
                src,
                dst,
                src_x0: 0,
                src_y0: 0,
                du_dx: 1 << 32,
                dv_dy: 1 << 32,
                dst_rect: (0, 0, size.0 as i32, size.1 as i32),
                bilinear: false,
                corner: false,
            }
        }
    }

    fn write(engine: &mut Fermi2d, ctx: &mut EngineContext<'_>, method: u32, value: u32) {
        engine.call_method(ctx, method, value);
    }

    fn write64(engine: &mut Fermi2d, ctx: &mut EngineContext<'_>, method: u32, value: i64) {
        write(engine, ctx, method, value as u32);
        write(engine, ctx, method + 1, (value >> 32) as u32);
    }

    fn set_surface(engine: &mut Fermi2d, ctx: &mut EngineContext<'_>, base: u32, s: &Surf) {
        let (linear, pitch, bits) = match s.layout {
            Layout::Pitch(pitch) => (1, pitch, 0),
            Layout::Block(height) => (0, 0, height << 4),
        };
        write(engine, ctx, base, s.format as u32);
        write(engine, ctx, base + 1, linear);
        write(engine, ctx, base + 2, bits);
        write(engine, ctx, base + 3, 1);
        write(engine, ctx, base + 4, 0);
        write(engine, ctx, base + 5, pitch);
        write(engine, ctx, base + 6, s.width);
        write(engine, ctx, base + 7, s.height);
        write(engine, ctx, base + 8, (s.address >> 32) as u32);
        write(engine, ctx, base + 9, s.address as u32);
    }

    fn run(engine: &mut Fermi2d, ctx: &mut EngineContext<'_>, setup: &Setup) {
        set_surface(engine, ctx, reg_index!(Regs, dst), &setup.dst);
        set_surface(engine, ctx, reg_index!(Regs, src), &setup.src);
        let args = reg_index!(Regs, pixels_from_memory);
        let mode = u32::from(setup.corner) | u32::from(setup.bilinear) << 4;
        write(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.sample_mode),
            mode,
        );
        let (x, y, w, h) = setup.dst_rect;
        write(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.dst_x0),
            x as u32,
        );
        write(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.dst_y0),
            y as u32,
        );
        write(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.dst_width),
            w as u32,
        );
        write(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.dst_height),
            h as u32,
        );
        write64(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.du_dx),
            setup.du_dx,
        );
        write64(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.dv_dy),
            setup.dv_dy,
        );
        write64(
            engine,
            ctx,
            reg_index!(Regs, pixels_from_memory.src_x0),
            setup.src_x0,
        );
        let src_y0 = reg_index!(Regs, pixels_from_memory.src_y0);
        assert_eq!(src_y0, args + 22);
        write(engine, ctx, src_y0, setup.src_y0 as u32);
        write(engine, ctx, src_y0 + 1, (setup.src_y0 >> 32) as u32);
    }

    fn put(h: &Harness, address: u64, bytes: &[u8]) {
        h.memory.write_block(address, bytes).unwrap();
    }

    fn get(h: &Harness, address: u64, len: usize) -> Vec<u8> {
        let mut out = vec![0; len];
        h.memory.read_block(address, &mut out).unwrap();
        out
    }

    fn rgba8(size: (u32, u32), pixel: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
        (0..size.1)
            .flat_map(|y| (0..size.0).map(move |x| (x, y)))
            .flat_map(|(x, y)| pixel(x, y))
            .collect()
    }

    fn pitch_surface(format: F, address: u64, size: (u32, u32), pitch: u32) -> Surf {
        Surf {
            format,
            address,
            width: size.0,
            height: size.1,
            layout: Layout::Pitch(pitch),
        }
    }

    fn block_surface(format: F, address: u64, size: (u32, u32), block_height: u32) -> Surf {
        Surf {
            format,
            address,
            width: size.0,
            height: size.1,
            layout: Layout::Block(block_height),
        }
    }

    #[test]
    fn method_numbers_match_the_hardware() {
        assert_eq!(reg_index!(Regs, pixels_from_memory.src_y0) + 1, 0x237);
        assert_eq!(reg_index!(Regs, pixels_from_memory.dst_x0), 0x22c);
    }

    #[test]
    fn same_format_pitch_copy_keeps_the_line_gaps() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let pixels = rgba8((8, 4), |x, y| [x as u8, y as u8, 0x80, 0xff]);
        for y in 0..4 {
            put(&h, 0x1000 + y * 64, &pixels[y as usize * 32..][..32]);
        }
        put(&h, 0x2000, &[0xee; 256]);
        let setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (8, 4), 64),
            pitch_surface(F::A8B8G8R8Unorm, 0x2000, (8, 4), 64),
            (8, 4),
        );
        run(&mut engine, &mut h.context(), &setup);
        for y in 0..4 {
            assert_eq!(
                get(&h, 0x2000 + y * 64, 32),
                &pixels[y as usize * 32..][..32]
            );
            assert_eq!(get(&h, 0x2000 + y * 64 + 32, 32), [0xee; 32]);
        }
        assert_eq!(h.rasterizer.inline_writes, vec![(0x2000, 3 * 64 + 32)]);
    }

    #[test]
    fn the_blit_runs_on_the_high_word_of_src_y0_only() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(&h, 0x1000, &[7; 16]);
        let setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (4, 1), 16),
            pitch_surface(F::A8B8G8R8Unorm, 0x2000, (4, 1), 16),
            (4, 1),
        );
        let mut ctx = h.context();
        run(&mut engine, &mut ctx, &setup);
        write(
            &mut engine,
            &mut ctx,
            reg_index!(Regs, pixels_from_memory.src_y0),
            0,
        );
        assert_eq!(h.rasterizer.inline_writes.len(), 1);
    }

    #[test]
    fn nearest_downscale_by_two_takes_every_other_pixel() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let pixels = rgba8((8, 8), |x, y| [x as u8, y as u8, 0, 0xff]);
        put(&h, 0x1000, &pixels);
        let mut setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (8, 8), 32),
            pitch_surface(F::A8B8G8R8Unorm, 0x2000, (4, 4), 16),
            (4, 4),
        );
        setup.du_dx = 2 << 32;
        setup.dv_dy = 2 << 32;
        run(&mut engine, &mut h.context(), &setup);
        let expected = rgba8((4, 4), |x, y| [x as u8 * 2, y as u8 * 2, 0, 0xff]);
        assert_eq!(get(&h, 0x2000, 64), expected);
    }

    #[test]
    fn source_offset_and_destination_offset_select_the_rectangles() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let pixels = rgba8((8, 8), |x, y| [x as u8, y as u8, 0, 0xff]);
        put(&h, 0x1000, &pixels);
        put(&h, 0x2000, &[0; 8 * 8 * 4]);
        let mut setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (8, 8), 32),
            pitch_surface(F::A8B8G8R8Unorm, 0x2000, (8, 8), 32),
            (2, 2),
        );
        setup.src_x0 = 3 << 32;
        setup.src_y0 = 5 << 32;
        setup.dst_rect = (1, 2, 2, 2);
        run(&mut engine, &mut h.context(), &setup);
        let out = get(&h, 0x2000, 256);
        for (x, y, want) in [
            (1, 2, [3, 5]),
            (2, 2, [4, 5]),
            (1, 3, [3, 6]),
            (2, 3, [4, 6]),
        ] {
            let at = (y * 8 + x) * 4;
            assert_eq!(&out[at..at + 2], &want, "({x}, {y})");
        }
        assert_eq!(out.iter().filter(|b| **b != 0).count(), 4 * 3);
    }

    #[test]
    fn block_linear_destination_is_swizzled_and_neighbours_survive() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let tiling = Tiling::new(32, 32, 1, 4, 1, 0, h.gob_layout);
        put(&h, 0x4000, &vec![0xee; tiling.size_bytes()]);
        let pixels = rgba8((8, 8), |x, y| [x as u8 + 1, y as u8 + 1, 7, 0xff]);
        for y in 0..8 {
            put(&h, 0x1000 + y * 32, &pixels[y as usize * 32..][..32]);
        }
        let mut setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (8, 8), 32),
            block_surface(F::A8B8G8R8Unorm, 0x4000, (32, 32), 1),
            (8, 8),
        );
        setup.dst_rect = (5, 11, 8, 8);
        run(&mut engine, &mut h.context(), &setup);

        let tiled = get(&h, 0x4000, tiling.size_bytes());
        let mut linear = vec![0u8; tiling.linear_size_bytes()];
        unswizzle(&tiling, &tiled, &mut linear);
        for y in 0..32usize {
            for x in 0..32usize {
                let at = (y * 32 + x) * 4;
                let inside = (5..13).contains(&x) && (11..19).contains(&y);
                let want: &[u8] = if inside {
                    &pixels[((y - 11) * 8 + x - 5) * 4..][..4]
                } else {
                    &[0xee; 4]
                };
                assert_eq!(&linear[at..at + 4], want, "({x}, {y})");
            }
        }
        assert_eq!(h.rasterizer.inline_writes.len(), 1);
    }

    #[test]
    fn block_linear_source_is_unswizzled() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let tiling = Tiling::new(16, 16, 1, 4, 2, 0, h.gob_layout);
        let pixels = rgba8((16, 16), |x, y| [x as u8, y as u8, 3, 9]);
        let mut tiled = vec![0u8; tiling.size_bytes()];
        swizzle(&tiling, &pixels, &mut tiled);
        put(&h, 0x8000, &tiled);
        let mut setup = Setup::copy(
            block_surface(F::A8B8G8R8Unorm, 0x8000, (16, 16), 2),
            pitch_surface(F::A8B8G8R8Unorm, 0x2000, (4, 4), 16),
            (4, 4),
        );
        setup.src_x0 = 6 << 32;
        setup.src_y0 = 9 << 32;
        run(&mut engine, &mut h.context(), &setup);
        let expected = rgba8((4, 4), |x, y| [x as u8 + 6, y as u8 + 9, 3, 9]);
        assert_eq!(get(&h, 0x2000, 64), expected);
    }

    #[test]
    fn rgba_to_bgra_conversion_swaps_red_and_blue() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(
            &h,
            0x1000,
            &[0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80],
        );
        let setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (2, 1), 8),
            pitch_surface(F::A8R8G8B8Unorm, 0x2000, (2, 1), 8),
            (2, 1),
        );
        run(&mut engine, &mut h.context(), &setup);
        assert_eq!(
            get(&h, 0x2000, 8),
            [0x30, 0x20, 0x10, 0x40, 0x70, 0x60, 0x50, 0x80]
        );
    }

    #[test]
    fn half_float_converts_to_rgb10a2_and_r32_float() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let halves: Vec<u8> = [0x3c00u16, 0x3800, 0x0000, 0x3c00]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        put(&h, 0x1000, &halves);
        let setup = Setup::copy(
            pitch_surface(F::R16G16B16A16Float, 0x1000, (1, 1), 8),
            pitch_surface(F::A2B10G10R10Unorm, 0x2000, (1, 1), 4),
            (1, 1),
        );
        run(&mut engine, &mut h.context(), &setup);
        let word = u32::from_le_bytes(get(&h, 0x2000, 4).try_into().unwrap());
        assert_eq!(word, 0xc000_0000 | 0x200 << 10 | 0x3ff);

        put(&h, 0x3000, &0.25f32.to_le_bytes());
        let setup = Setup::copy(
            pitch_surface(F::R32Float, 0x3000, (1, 1), 4),
            pitch_surface(F::R8Unorm, 0x3100, (1, 1), 1),
            (1, 1),
        );
        run(&mut engine, &mut h.context(), &setup);
        assert_eq!(get(&h, 0x3100, 1), [64]);
    }

    #[test]
    fn bilinear_upscale_blends_neighbours() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(&h, 0x1000, &[0, 255]);
        let mut setup = Setup::copy(
            pitch_surface(F::R8Unorm, 0x1000, (2, 1), 2),
            pitch_surface(F::R8Unorm, 0x2000, (4, 1), 4),
            (4, 1),
        );
        setup.du_dx = 1 << 31;
        setup.bilinear = true;
        run(&mut engine, &mut h.context(), &setup);
        assert_eq!(get(&h, 0x2000, 4), [0, 128, 255, 255]);
    }

    #[test]
    fn an_accelerating_renderer_suppresses_the_software_path() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(&h, 0x1000, &[9; 64]);
        put(&h, 0x2000, &[1; 64]);
        let mut setup = Setup::copy(
            block_surface(F::A8B8G8R8Unorm, 0x1000, (4, 4), 0),
            pitch_surface(F::A2R10G10B10Unorm, 0x2000, (4, 4), 16),
            (4, 4),
        );
        setup.src_x0 = (1 << 32) | 0x8000_0000;
        setup.du_dx = 3 << 32;
        setup.bilinear = true;
        setup.dst_rect = (2, 3, 4, 4);
        let mut accelerator = Accelerator {
            accept: true,
            ..Default::default()
        };
        let mut ctx = EngineContext {
            channel: ChannelId(3),
            memory: &h.memory,
            rasterizer: &mut accelerator,
            syncpoints: &h.syncpoints,
            gob_layout: h.gob_layout,
            qmd_scheduler: &mut h.qmd_scheduler,
        };
        run(&mut engine, &mut ctx, &setup);
        assert_eq!(get(&h, 0x2000, 64), [1; 64]);
        assert!(accelerator.inline_writes.is_empty());
        assert!(accelerator.flushes.is_empty());
        let [blit] = accelerator.blits[..] else {
            panic!("expected one blit");
        };
        assert_eq!(blit.src.address, 0x1000);
        assert_eq!(blit.src.block, Some((0, 0, 0)));
        assert_eq!(blit.dst.block, None);
        assert_eq!(blit.dst.pitch, 16);
        assert_eq!(blit.src_format, PixelFormat::A8B8G8R8Unorm);
        assert_eq!(blit.dst_format, PixelFormat::A2R10G10B10Unorm);
        assert_eq!(blit.src_x0, (1 << 32) | 0x8000_0000);
        assert_eq!(blit.du_dx, 3 << 32);
        assert_eq!(blit.dv_dy, 1 << 32);
        assert_eq!((blit.dst_x0, blit.dst_y0), (2, 3));
        assert_eq!((blit.dst_width, blit.dst_height), (4, 4));
        assert!(blit.linear_filter);
    }

    #[test]
    fn a_declining_renderer_gets_the_blit_then_the_software_path_runs() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(&h, 0x1000, &[9; 16]);
        let setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (4, 1), 16),
            pitch_surface(F::A8B8G8R8Unorm, 0x2000, (4, 1), 16),
            (4, 1),
        );
        let mut accelerator = Accelerator::default();
        let mut ctx = EngineContext {
            channel: ChannelId(0),
            memory: &h.memory,
            rasterizer: &mut accelerator,
            syncpoints: &h.syncpoints,
            gob_layout: h.gob_layout,
            qmd_scheduler: &mut h.qmd_scheduler,
        };
        run(&mut engine, &mut ctx, &setup);
        assert_eq!(accelerator.blits.len(), 1);
        assert_eq!(accelerator.inline_writes, vec![(0x2000, 16)]);
        assert_eq!(accelerator.flushes.len(), 2);
        assert_eq!(get(&h, 0x2000, 16), [9; 16]);
    }

    #[test]
    fn corner_origin_moves_the_source_back_by_half_the_step() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let mut setup = Setup::copy(
            block_surface(F::R8Unorm, 0, (8, 8), 0),
            pitch_surface(F::R8Unorm, 0x2000, (2, 2), 2),
            (2, 2),
        );
        setup.src_x0 = 4 << 32;
        setup.du_dx = 2 << 32;
        setup.corner = true;
        let mut accelerator = Accelerator {
            accept: true,
            ..Default::default()
        };
        let mut ctx = EngineContext {
            channel: ChannelId(0),
            memory: &h.memory,
            rasterizer: &mut accelerator,
            syncpoints: &h.syncpoints,
            gob_layout: h.gob_layout,
            qmd_scheduler: &mut h.qmd_scheduler,
        };
        run(&mut engine, &mut ctx, &setup);
        assert_eq!(accelerator.blits[0].src_x0, 3 << 32);
    }

    #[test]
    fn corner_origin_keeps_the_fractional_half_step() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        let mut setup = Setup::copy(
            block_surface(F::R8Unorm, 0, (8, 8), 0),
            pitch_surface(F::R8Unorm, 0x2000, (2, 2), 2),
            (2, 2),
        );
        setup.src_x0 = 4 << 32;
        setup.src_y0 = 4 << 32;
        setup.du_dx = 1 << 32;
        setup.dv_dy = 3 << 31;
        setup.corner = true;
        let mut accelerator = Accelerator {
            accept: true,
            ..Default::default()
        };
        let mut ctx = EngineContext {
            channel: ChannelId(0),
            memory: &h.memory,
            rasterizer: &mut accelerator,
            syncpoints: &h.syncpoints,
            gob_layout: h.gob_layout,
            qmd_scheduler: &mut h.qmd_scheduler,
        };
        run(&mut engine, &mut ctx, &setup);
        assert_eq!(accelerator.blits[0].src_x0, (3 << 32) | 0x8000_0000);
        assert_eq!(accelerator.blits[0].src_y0, (3 << 32) | 0x4000_0000);
    }

    #[test]
    fn nearest_sampling_follows_the_programmed_fractional_step() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(&h, 0x1000, &[10, 11, 12, 13]);
        let mut setup = Setup::copy(
            pitch_surface(F::R8Unorm, 0x1000, (4, 1), 4),
            pitch_surface(F::R8Unorm, 0x2000, (3, 1), 3),
            (3, 1),
        );
        setup.du_dx = 3 << 31;
        run(&mut engine, &mut h.context(), &setup);
        assert_eq!(get(&h, 0x2000, 3), [10, 11, 13]);
    }

    #[test]
    fn bilinear_sampling_uses_the_programmed_start_and_step() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(&h, 0x1000, &[0, 100, 200, 255]);
        let mut setup = Setup::copy(
            pitch_surface(F::R8Unorm, 0x1000, (4, 1), 4),
            pitch_surface(F::R8Unorm, 0x2000, (3, 1), 3),
            (3, 1),
        );
        setup.src_x0 = 1 << 31;
        setup.du_dx = 3 << 31;
        setup.bilinear = true;
        run(&mut engine, &mut h.context(), &setup);
        assert_eq!(get(&h, 0x2000, 3), [50, 200, 255]);
    }

    #[test]
    fn undecodable_formats_and_empty_rectangles_do_nothing() {
        let mut h = Harness::new();
        let mut engine = Fermi2d::new();
        put(&h, 0x1000, &[9; 16]);
        let mut setup = Setup::copy(
            pitch_surface(F::A8B8G8R8Unorm, 0x1000, (4, 1), 16),
            pitch_surface(F::A8B8G8R8Unorm, 0x2000, (4, 1), 16),
            (4, 1),
        );
        setup.dst_rect = (0, 0, 0, 1);
        run(&mut engine, &mut h.context(), &setup);
        setup.dst_rect = (0, 0, 4, 1);
        setup.dst.format = F::None;
        run(&mut engine, &mut h.context(), &setup);
        assert_eq!(get(&h, 0x2000, 16), [0; 16]);
        assert!(h.rasterizer.inline_writes.is_empty());
    }

    #[test]
    fn layouts_match_the_pixel_format_sizes() {
        for &format in F::ALL {
            let (Some(layout), Some(pixel)) = (
                converter::Layout::of(format),
                PixelFormat::from_render_target(format),
            ) else {
                continue;
            };
            assert_eq!(layout.bytes as u32, pixel.bytes_per_block(), "{format:?}");
        }
    }
}
