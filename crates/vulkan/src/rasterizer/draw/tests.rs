//! Draws of hand-assembled Maxwell programs on synthetic 3D engine state.

use lichii_common::settings::{Anisotropy, GraphicsSettings, ResolutionScale};
use lichii_gpu::backend::{
    ChannelId, ComputeContext, ConstBufferBinding, DrawParams, DrawTexture, GraphicsConstBuffers,
    GraphicsContext, IndirectDraw, QmdVersion, Rasterizer,
};
use lichii_gpu::engines::maxwell_3d::regs::{
    ClearSurface, DepthMode, MAX_CONST_BUFFERS, MAX_SHADER_STAGE, PipelineBits, PrimitiveTopology,
    RenderTargetConfigBits, RtControl, ScissorTestBits, ScissorTestBits1, ShaderType,
    SurfaceClipBits, SurfaceClipBits1, TileMode, VertexAttribute, VertexStreamBits,
};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};
use lichii_gpu::gmmu::Gmmu;
use lichii_shader::DescriptorKind;
use lichii_shader::maxwell::opcode::Opcode;
use lichii_texture::block_linear::{GobLayout, Tiling, swizzle};

use super::super::tests::{DADDR, Fixture, GPU_VA, fixture, gmmu, guest_linear};
use super::{StageSources, Warning};
use crate::pipeline::environment::TexturePool;

const SIZE: u32 = 16;
const PROGRAMS: u64 = 0x10_0000;
const FRAGMENT: u32 = 0x200;
const VERTICES: u64 = 0x20_0000;
const VERTEX_CBUF: u64 = 0x30_0000;
const FRAGMENT_CBUF: u64 = 0x30_1000;
const RZ: u64 = 0xff;

/// `op` with `fields`, guarded by PT.
fn sass(op: Opcode, fields: u64) -> u64 {
    let pattern = op.pattern();
    (u64::from(pattern.value) << 48) | ((fields | 7 << 16) & !(u64::from(pattern.mask) << 48))
}

fn exit() -> u64 {
    sass(Opcode::Exit, 0xf)
}

/// MOV `dest`, c[0][`offset`].
fn mov_cbuf(dest: u64, offset: u64) -> u64 {
    sass(Opcode::MovCbuf, (0xf << 39) | (offset / 4) << 20 | dest)
}

/// A program at a 32-byte aligned address: its 0x50-byte header, then `code` with a scheduling
/// word in the first slot of every 32 bytes.
fn program(header: [u32; 20], code: &[u64]) -> Vec<u8> {
    let mut bytes: Vec<u8> = bytemuck::cast_slice(&header).to_vec();
    for insn in code {
        if bytes.len().is_multiple_of(32) {
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    bytes
}

fn header(sph_type: u32, shader_type: ShaderType) -> [u32; 20] {
    let mut header = [0; 20];
    header[0] = sph_type | 3 << 5 | shader_type.0 << 10;
    header
}

/// Position XY from generic attribute 0 (which the header's input map enables), ZW from
/// c[0][0..8].
fn vertex_program() -> Vec<u8> {
    let mut header = header(1, ShaderType::VERTEX_B);
    header[6] = 0b11;
    program(
        header,
        &[
            sass(
                Opcode::Ald,
                (1 << 47) | (RZ << 39) | (0x80 << 20) | (RZ << 8),
            ),
            mov_cbuf(2, 0),
            mov_cbuf(3, 4),
            sass(
                Opcode::Ast,
                (3 << 47) | (RZ << 39) | (0x70 << 20) | (RZ << 8),
            ),
            exit(),
        ],
    )
}

/// Writes render target 0 from c[0][0..16], with the first two words in `order`.
fn fragment_program(order: [u64; 2]) -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    program(
        header,
        &[
            mov_cbuf(0, order[0]),
            mov_cbuf(1, order[1]),
            mov_cbuf(2, 8),
            mov_cbuf(3, 12),
            exit(),
        ],
    )
}

fn write(f: &Fixture, offset: u64, bytes: &[u8]) {
    f.memory.write_bytes(DADDR + offset, bytes).unwrap();
}

/// A 16x16 A8B8G8R8 target at `GPU_VA`, the two programs at `PROGRAMS` and one triangle
/// covering the target in `VERTICES`.
fn draw_regs() -> Box<Regs> {
    let mut regs = Regs::with_defaults();
    let rt = &mut regs.rt[0];
    rt.address_high = (GPU_VA >> 32) as u32;
    rt.address_low = GPU_VA as u32;
    rt.width = SIZE;
    rt.height = SIZE;
    rt.format = 0xd5;
    rt.tile_mode = TileMode(1 << 4);
    rt.bits = RenderTargetConfigBits(1);
    regs.rt_control = RtControl(1);
    regs.surface_clip.bits = SurfaceClipBits(SIZE << 16);
    regs.surface_clip.bits1 = SurfaceClipBits1(SIZE << 16);
    let programs = GPU_VA + PROGRAMS;
    regs.program_region.address_high = (programs >> 32) as u32;
    regs.program_region.address_low = programs as u32;
    regs.pipelines[1].bits = PipelineBits(1 | ShaderType::VERTEX_B.0 << 4);
    regs.pipelines[5].bits = PipelineBits(1 | ShaderType::PIXEL.0 << 4);
    regs.pipelines[5].offset = FRAGMENT;
    regs.vertex_attrib_format[0] = VertexAttribute(0x4 << 21 | 0x7 << 27);
    let vertices = GPU_VA + VERTICES;
    regs.vertex_streams[0].bits = VertexStreamBits(8 | 1 << 12);
    regs.vertex_streams[0].address_high = (vertices >> 32) as u32;
    regs.vertex_streams[0].address_low = vertices as u32;
    let limit = vertices + 3 * 8 - 1;
    regs.vertex_stream_limits[0].address_high = (limit >> 32) as u32;
    regs.vertex_stream_limits[0].address_low = limit as u32;
    let half = SIZE as f32 / 2.0;
    let viewport = &mut regs.viewport_transform[0];
    viewport.scale_x = half;
    viewport.translate_x = half;
    viewport.scale_y = half;
    viewport.translate_y = half;
    viewport.scale_z = 1.0;
    regs.viewport_scale_offset_enabled = 1;
    regs.depth_mode = DepthMode::ZERO_TO_ONE;
    regs
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Memory for `draw_regs`: the vertices, both constant buffers and the vertex program.
fn draw_fixture() -> Option<(Fixture, Gmmu)> {
    let mut f = fixture(GobLayout::Tegra)?;
    f.rasterizer.draw.pipelines.set_async_compilation(false);
    let gmmu = gmmu(&f);
    write(
        &f,
        VERTICES,
        &f32_bytes(&[-1.0, -1.0, 3.0, -1.0, -1.0, 3.0]),
    );
    write(&f, VERTEX_CBUF, &f32_bytes(&[0.0, 1.0]));
    write(&f, FRAGMENT_CBUF, &f32_bytes(&[1.0, 0.5, 0.0, 1.0]));
    write(&f, PROGRAMS, &vertex_program());
    Some((f, gmmu))
}

fn const_buffers() -> GraphicsConstBuffers {
    let mut const_buffers = [[ConstBufferBinding::default(); MAX_CONST_BUFFERS]; MAX_SHADER_STAGE];
    const_buffers[0][0] = ConstBufferBinding {
        address: GPU_VA + VERTEX_CBUF,
        size: 0x100,
        enabled: true,
    };
    const_buffers[4][0] = ConstBufferBinding {
        address: GPU_VA + FRAGMENT_CBUF,
        size: 0x200,
        enabled: true,
    };
    const_buffers
}

fn draw(f: &mut Fixture, gmmu: &Gmmu, regs: &Regs) {
    let mut dirty = DirtyState::new();
    let const_buffers = const_buffers();
    let root_tables = [[0; 64]; 8];
    let mut ctx = GraphicsContext {
        channel: ChannelId(0),
        memory: gmmu,
        regs,
        dirty: &mut dirty,
        const_buffers: &const_buffers,
        root_tables: &root_tables,
    };
    f.rasterizer.draw(
        &mut ctx,
        &DrawParams {
            topology: PrimitiveTopology::TRIANGLES,
            indexed: false,
            first: 0,
            count: 3,
            base_vertex: 0,
            base_instance: 0,
            instance_count: 1,
            inline_indices: None,
        },
    );
}

/// The render target's texels after writing it back to guest memory.
fn target(f: &mut Fixture) -> Vec<[u8; 4]> {
    let tiling = Tiling::new(SIZE, SIZE, 1, 4, 1, 0, GobLayout::Tegra);
    f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
    guest_linear(f, DADDR, &tiling).as_chunks::<4>().0.to_vec()
}

#[test]
fn draws_hand_assembled_programs_and_reads_the_target_back() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &fragment_program([0, 4]),
    );
    let regs = draw_regs();
    draw(&mut f, &gmmu, &regs);
    let texels = target(&mut f);
    assert!(
        texels.iter().all(|&t| t == [255, 128, 0, 255]),
        "every texel takes the fragment program's colour: {:?}",
        &texels[..4]
    );

    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!((stats.translated, stats.failed), (2, 0));
    let [Some(vertex), _, _, _, Some(fragment)] = &f.rasterizer.draw.stages else {
        panic!("vertex and fragment modules bound");
    };
    let binding = |m: &lichii_shader::ShaderModule| {
        let b = &m.resources.sets[0].bindings[0];
        (b.binding, b.kind)
    };
    assert_eq!(binding(vertex), (0, DescriptorKind::UniformBuffer));
    assert_eq!(
        binding(fragment),
        (1, DescriptorKind::UniformBuffer),
        "the fragment stage numbers its bindings after the vertex stage's"
    );
    f.assert_no_validation_errors();
}

#[test]
fn cached_programs_translate_again_only_after_a_guest_write() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let fragment = PROGRAMS + u64::from(FRAGMENT);
    write(&f, fragment, &fragment_program([0, 4]));
    let regs = draw_regs();
    draw(&mut f, &gmmu, &regs);
    draw(&mut f, &gmmu, &regs);
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!(
        (stats.translated, stats.hits),
        (2, 2),
        "the second draw finds both modules"
    );

    let swapped = fragment_program([4, 0]);
    write(&f, fragment, &swapped);
    f.rasterizer
        .invalidate_region(DADDR + fragment, swapped.len() as u64);
    draw(&mut f, &gmmu, &regs);
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!(
        (stats.translated, stats.hits),
        (3, 3),
        "only the rewritten fragment program translates again"
    );
    assert!(
        target(&mut f).iter().all(|&t| t == [128, 255, 0, 255]),
        "the new program's colour"
    );
    f.assert_no_validation_errors();
}

#[test]
fn draws_of_untranslatable_programs_are_skipped_without_retrying() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let unsupported = sass(Opcode::Ast, 2 << 8);
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &program(header, &[unsupported, exit()]),
    );
    let regs = draw_regs();
    for _ in 0..3 {
        draw(&mut f, &gmmu, &regs);
    }
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!(
        (stats.translated, stats.failed),
        (2, 1),
        "the failure is cached: one warning, no retranslation"
    );
    assert!(
        target(&mut f).iter().all(|&t| t == [0; 4]),
        "nothing was drawn"
    );
    f.assert_no_validation_errors();
}

const TIC: u64 = 0x40_0000;
const TSC: u64 = 0x40_1000;
const TEXTURE: u64 = 0x50_0000;
const OTHER_TEXTURE: u64 = 0x58_0000;

/// A block-linear 2D A8B8G8R8 texture header (block height 2 GOBs) with identity swizzle.
fn tic(gpu_va: u64, width: u32, height: u32) -> [u32; 8] {
    [
        0x08 | 2 << 7 | 2 << 10 | 2 << 13 | 2 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28,
        gpu_va as u32,
        (gpu_va >> 32) as u32 | 3 << 21,
        1 << 3,
        (width - 1) | 1 << 23,
        (height - 1) | 1 << 31,
        0,
        0,
    ]
}

/// MOV32I `dest`, `value`.
fn mov_u32(dest: u64, value: u32) -> u64 {
    sass(
        Opcode::Mov32i,
        (u64::from(value) << 20) | (0xf << 12) | dest,
    )
}

fn mov_f32(dest: u64, value: f32) -> u64 {
    mov_u32(dest, value.to_bits())
}

/// Samples the texture whose handle is c[0][0x40] at (0.5, 0.5) into render target 0.
fn sampling_program() -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let tex = sass(
        Opcode::Tex,
        (0x10 << 36) | (7 << 51) | (0b1111 << 31) | (2 << 28) | (2 << 8),
    );
    program(header, &[mov_f32(2, 0.5), mov_f32(3, 0.5), tex, exit()])
}

/// `draw_regs` with texture headers at `TIC` (two entries), samplers at `TSC` and bound
/// textures in constant buffer 0.
fn sampling_regs() -> Box<Regs> {
    let mut regs = draw_regs();
    let (tic, tsc) = (GPU_VA + TIC, GPU_VA + TSC);
    regs.tex_header.address_high = (tic >> 32) as u32;
    regs.tex_header.address_low = tic as u32;
    regs.tex_header.limit = 1;
    regs.tex_sampler.address_high = (tsc >> 32) as u32;
    regs.tex_sampler.address_low = tsc as u32;
    regs.tex_sampler.limit = 0;
    regs.bindless_texture_const_buffer_slot = 0;
    regs
}

/// Fills the 16x16 texture at `offset` with `texel` and points texture header `index` at it.
fn uniform_texture(f: &Fixture, index: u64, offset: u64, texel: [u8; 4]) {
    let tiling = Tiling::new(SIZE, SIZE, 1, 4, 1, 0, GobLayout::Tegra);
    write(f, offset, &texel.repeat(tiling.size_bytes() / 4));
    let header = tic(GPU_VA + offset, SIZE, SIZE);
    write(f, TIC + index * 32, bytemuck::cast_slice(&header));
}

#[test]
fn guest_barriers_record_between_draws() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &fragment_program([0, 4]),
    );
    let regs = draw_regs();
    draw(&mut f, &gmmu, &regs);
    f.rasterizer.barrier();
    draw(&mut f, &gmmu, &regs);
    f.rasterizer.barrier();
    assert!(target(&mut f).iter().all(|&t| t == [255, 128, 0, 255]));
    f.assert_no_validation_errors();
}

#[test]
fn indirect_draws_rebind_buffers_their_commands_merged() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &fragment_program([0, 4]),
    );
    let commands = VERTICES + (64 << 10) - 8;
    write(&f, commands, bytemuck::cast_slice(&[3u32, 1, 0, 0]));
    let regs = draw_regs();
    let merged = f.rasterizer.buffer_cache.stats().merged_buffers;
    let mut dirty = DirtyState::new();
    let const_buffers = const_buffers();
    let root_tables = [[0; 64]; 8];
    let mut ctx = GraphicsContext {
        channel: ChannelId(0),
        memory: &gmmu,
        regs: &regs,
        dirty: &mut dirty,
        const_buffers: &const_buffers,
        root_tables: &root_tables,
    };
    f.rasterizer.draw_indirect(
        &mut ctx,
        &IndirectDraw {
            topology: PrimitiveTopology::TRIANGLES,
            indexed: false,
            buffer: GPU_VA + commands,
            stride: 16,
            max_draws: 1,
            count_buffer: None,
        },
    );
    assert_eq!(
        f.rasterizer.buffer_cache.stats().merged_buffers,
        merged + 1,
        "the command buffer merged with the vertex buffer's"
    );
    let r = &mut f.rasterizer;
    let current = r
        .buffer_cache
        .acquire(&mut r.scheduler, &mut r.staging, DADDR + VERTICES, 24)
        .unwrap();
    assert_eq!(
        r.draw.vertex_buffers[0], current.buffer,
        "the vertex binding points at the merged buffer"
    );
    assert!(target(&mut f).iter().all(|&t| t == [255, 128, 0, 255]));
    f.assert_no_validation_errors();
}

#[test]
fn changed_bindless_handles_translate_again() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(&f, PROGRAMS + u64::from(FRAGMENT), &sampling_program());
    uniform_texture(&f, 0, TEXTURE, [10, 200, 30, 255]);
    uniform_texture(&f, 1, OTHER_TEXTURE, [50, 60, 70, 255]);
    let regs = sampling_regs();
    draw(&mut f, &gmmu, &regs);
    assert!(
        target(&mut f).iter().all(|&t| t == [10, 200, 30, 255]),
        "the draw samples texture 0"
    );
    draw(&mut f, &gmmu, &regs);
    assert_eq!(f.rasterizer.draw.shaders.stats().translated, 2);

    write(&f, FRAGMENT_CBUF + 0x40, &1u32.to_le_bytes());
    f.rasterizer
        .invalidate_region(DADDR + FRAGMENT_CBUF + 0x40, 4);
    draw(&mut f, &gmmu, &regs);
    assert_eq!(
        f.rasterizer.draw.shaders.stats().translated,
        3,
        "the handle the translator read changed"
    );
    assert!(target(&mut f).iter().all(|&t| t == [50, 60, 70, 255]));
    f.assert_no_validation_errors();
}

#[test]
fn cached_samplers_follow_the_anisotropy_setting() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let mut tsc = [0u32; 8];
    tsc[1] = 2 | 2 << 4;
    write(&f, TSC, bytemuck::cast_slice(&tsc));
    let cbufs = const_buffers();
    let src = StageSources {
        gmmu: &gmmu,
        cbufs: &cbufs[4],
        pool: TexturePool {
            address: GPU_VA + TIC,
            limit: 0,
            via_header_index: false,
            switch2: false,
        },
        samplers: GPU_VA + TSC,
        sampler_limit: 0,
    };
    let r = &mut f.rasterizer;
    r.draw.set_anisotropy(Anisotropy::X1);
    let x1 = r.sampler(&src, 0).unwrap();
    r.draw.set_anisotropy(Anisotropy::X16);
    let x16 = r.sampler(&src, 0).unwrap();
    assert_ne!(x1, x16, "a new setting creates a new sampler");
    r.draw.set_anisotropy(Anisotropy::X1);
    assert_eq!(r.sampler(&src, 0).unwrap(), x1);
    assert_eq!(r.draw.samplers.len(), 2);
    f.assert_no_validation_errors();
}

#[test]
fn writes_to_any_page_of_a_scattered_program_translate_it_again() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    const SECOND_PAGE: u64 = PROGRAMS + 0x1000;
    const ELSEWHERE: u64 = 0x60_0000;
    gmmu.map(GPU_VA + SECOND_PAGE, DADDR + ELSEWHERE, 0x1000, 0, false)
        .unwrap();
    let start = SECOND_PAGE - 0x60;
    let fragment = |order: [u64; 2]| {
        let mut header = header(2, ShaderType::PIXEL);
        header[18] = 0xf;
        program(
            header,
            &[
                mov_cbuf(2, 8),
                mov_cbuf(3, 12),
                mov_cbuf(0, order[0]),
                mov_cbuf(1, order[1]),
                exit(),
            ],
        )
    };
    gmmu.write_block(GPU_VA + start, &fragment([0, 4])).unwrap();
    let mut regs = draw_regs();
    regs.pipelines[5].offset = (start - PROGRAMS) as u32;
    draw(&mut f, &gmmu, &regs);
    assert!(target(&mut f).iter().all(|&t| t == [255, 128, 0, 255]));

    let swapped = fragment([4, 0]);
    let second_page = &swapped[0x60..];
    write(&f, ELSEWHERE, second_page);
    f.rasterizer
        .invalidate_region(DADDR + ELSEWHERE, second_page.len() as u64);
    draw(&mut f, &gmmu, &regs);
    assert_eq!(
        f.rasterizer.draw.shaders.stats().translated,
        3,
        "the write to the second page's device memory reached the program"
    );
    assert!(target(&mut f).iter().all(|&t| t == [128, 255, 0, 255]));
    f.assert_no_validation_errors();
}

#[test]
fn compute_programs_key_on_their_local_memory_size() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    const COMPUTE: u64 = 0x10_8000;
    let mut code = 0u64.to_le_bytes().to_vec();
    code.extend_from_slice(&exit().to_le_bytes());
    write(&f, COMPUTE, &code);
    let const_buffers = [ConstBufferBinding::default(); 8];
    let qmd = [0u32; 64];
    let dispatch = |f: &mut Fixture, local_memory_size: u32| {
        f.rasterizer.dispatch_compute(&ComputeContext {
            channel: ChannelId(0),
            memory: &gmmu,
            program: GPU_VA + COMPUTE,
            grid: [1; 3],
            block: [1; 3],
            shared_memory_size: 0,
            local_memory_size,
            const_buffers: &const_buffers,
            texture_headers: 0,
            texture_header_limit: 0,
            samplers: 0,
            sampler_limit: 0,
            linked_samplers: false,
            qmd_version: QmdVersion::V2,
            qmd: &qmd,
        });
    };
    dispatch(&mut f, 0);
    dispatch(&mut f, 0);
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!((stats.translated, stats.failed, stats.hits), (1, 0, 1));
    dispatch(&mut f, 64);
    assert_eq!(
        f.rasterizer.draw.shaders.stats().translated,
        2,
        "a different local memory size translates its own module"
    );
    f.rasterizer.wait_for_idle();
    f.assert_no_validation_errors();
}

/// texel (x, y) of the 8x8 draw-texture source: every texel distinct.
fn source_texel(x: u32, y: u32) -> [u8; 4] {
    [(x * 16) as u8, (y * 16) as u8, (x + y * 8) as u8, 255]
}

fn draw_texture(f: &mut Fixture, gmmu: &Gmmu, regs: &Regs, draw: DrawTexture) {
    let mut dirty = DirtyState::new();
    let const_buffers = const_buffers();
    let root_tables = [[0; 64]; 8];
    let mut ctx = GraphicsContext {
        channel: ChannelId(0),
        memory: gmmu,
        regs,
        dirty: &mut dirty,
        const_buffers: &const_buffers,
        root_tables: &root_tables,
    };
    f.rasterizer.draw_texture(&mut ctx, &draw);
}

#[test]
fn draw_texture_maps_sub_rects_with_the_sampler_filter() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let tiling = Tiling::new(8, 8, 1, 4, 1, 0, GobLayout::Tegra);
    let linear: Vec<u8> = (0..64).flat_map(|i| source_texel(i % 8, i / 8)).collect();
    let mut tiled = vec![0; tiling.size_bytes()];
    swizzle(&tiling, &linear, &mut tiled);
    write(&f, TEXTURE, &tiled);
    write(&f, TIC, bytemuck::cast_slice(&tic(GPU_VA + TEXTURE, 8, 8)));
    let mut linear_tsc = [0u32; 8];
    linear_tsc[1] = 2 | 2 << 4;
    write(&f, TSC + 32, bytemuck::cast_slice(&linear_tsc));
    let mut regs = sampling_regs();
    regs.tex_sampler.limit = 1;
    let rect = |dst: [f32; 4], src: [f32; 4], sampler: u32| DrawTexture {
        dst_x0: dst[0],
        dst_y0: dst[1],
        dst_x1: dst[2],
        dst_y1: dst[3],
        src_x0: src[0],
        src_y0: src[1],
        src_x1: src[2],
        src_y1: src[3],
        texture: 0,
        sampler,
    };
    // A: texels [2, 6) x [1, 5) at 1:1 into a destination given right to left.
    draw_texture(
        &mut f,
        &gmmu,
        &regs,
        rect([10.0, 2.0, 6.0, 6.0], [2.0, 1.0, 6.0, 5.0], 0),
    );
    // B: the same texels with the source given bottom to top, which flips them.
    draw_texture(
        &mut f,
        &gmmu,
        &regs,
        rect([10.0, 10.0, 14.0, 14.0], [2.0, 5.0, 6.0, 1.0], 0),
    );
    // C: texels [0, 4) x [0, 4) magnified twice, nearest.
    draw_texture(
        &mut f,
        &gmmu,
        &regs,
        rect([0.0, 8.0, 8.0, 16.0], [0.0, 0.0, 4.0, 4.0], 0),
    );
    // D: texels [0, 2) of row 0 stretched over four pixels, linear.
    draw_texture(
        &mut f,
        &gmmu,
        &regs,
        rect([12.0, 0.0, 16.0, 1.0], [0.0, 0.0, 2.0, 1.0], 1),
    );

    let texels = target(&mut f);
    for (i, &texel) in texels.iter().enumerate() {
        let (x, y) = (i as u32 % SIZE, i as u32 / SIZE);
        let want = if (6..10).contains(&x) && (2..6).contains(&y) {
            source_texel(x - 4, y - 1)
        } else if (10..14).contains(&x) && (10..14).contains(&y) {
            source_texel(x - 8, 14 - y)
        } else if x < 8 && y >= 8 {
            source_texel(x / 2, (y - 8) / 2)
        } else if (12..16).contains(&x) && y == 0 {
            if matches!(x, 13 | 14) {
                let red = if x == 13 { 4 } else { 12 };
                assert!(
                    texel[0].abs_diff(red) <= 1 && texel[3] == 255,
                    "linear texel ({x}, {y}): {texel:?}"
                );
            }
            continue;
        } else {
            [0; 4]
        };
        assert_eq!(texel, want, "texel ({x}, {y})");
    }
    f.assert_no_validation_errors();
}

#[test]
fn draw_texture_skips_integer_sources() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let mut header = tic(GPU_VA + TEXTURE, 8, 8);
    header[0] = (header[0] & !(0xfff << 7)) | 4 << 7 | 4 << 10 | 4 << 13 | 4 << 16;
    write(&f, TIC, bytemuck::cast_slice(&header));
    draw_texture(&mut f, &gmmu, &sampling_regs(), full_rect(8.0));
    assert_ne!(
        f.rasterizer.draw.warned & Warning::DrawTextureSource as u32,
        0
    );
    assert!(
        target(&mut f).iter().all(|&t| t == [0; 4]),
        "nothing was drawn"
    );
    f.assert_no_validation_errors();
}

#[test]
fn draw_texture_reads_layer_zero_of_array_sources() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let tiling = Tiling::new(8, 8, 1, 4, 1, 0, GobLayout::Tegra);
    let linear: Vec<u8> = (0..64).flat_map(|i| source_texel(i % 8, i / 8)).collect();
    let mut tiled = vec![0; tiling.size_bytes()];
    swizzle(&tiling, &linear, &mut tiled);
    write(&f, TEXTURE, &tiled);
    let mut header = tic(GPU_VA + TEXTURE, 8, 8);
    header[4] = (header[4] & !(0xf << 23)) | 5 << 23;
    header[5] |= 1 << 16;
    write(&f, TIC, bytemuck::cast_slice(&header));
    draw_texture(&mut f, &gmmu, &sampling_regs(), full_rect(8.0));
    for (i, &texel) in target(&mut f).iter().enumerate() {
        let (x, y) = (i as u32 % SIZE, i as u32 / SIZE);
        let want = if x < 8 && y < 8 {
            source_texel(x, y)
        } else {
            [0; 4]
        };
        assert_eq!(texel, want, "texel ({x}, {y})");
    }
    f.assert_no_validation_errors();
}

fn full_rect(size: f32) -> DrawTexture {
    DrawTexture {
        dst_x0: 0.0,
        dst_y0: 0.0,
        dst_x1: size,
        dst_y1: size,
        src_x0: 0.0,
        src_y0: 0.0,
        src_x1: size,
        src_y1: size,
        texture: 0,
        sampler: 0,
    }
}

/// The guest size of the render target at `TEXTURE` the rescaling tests render at 2x: tall
/// enough to rescale.
const SOURCE: (u32, u32) = (64, 320);
const GREEN: [u8; 4] = [0, 255, 0, 255];
const BLUE: [u8; 4] = [0, 0, 255, 255];
/// Where the rescaling tests place the fragment programs that read the source.
const READER: u32 = 0x400;

/// `draw_fixture` at 2x resolution scale.
fn rescaled_fixture() -> Option<(Fixture, Gmmu)> {
    let (mut f, gmmu) = draw_fixture()?;
    f.rasterizer.apply_settings(&GraphicsSettings {
        resolution_scale: ResolutionScale::X2,
        async_shader_compilation: false,
        ..GraphicsSettings::default()
    });
    Some((f, gmmu))
}

/// `draw_regs` rendering into the `width`x`height` A8B8G8R8 target at `offset`.
fn target_regs(offset: u64, (width, height): (u32, u32)) -> Box<Regs> {
    let mut regs = sampling_regs();
    let rt = &mut regs.rt[0];
    rt.address_high = ((GPU_VA + offset) >> 32) as u32;
    rt.address_low = (GPU_VA + offset) as u32;
    rt.width = width;
    rt.height = height;
    regs.surface_clip.bits = SurfaceClipBits(width << 16);
    regs.surface_clip.bits1 = SurfaceClipBits1(height << 16);
    let viewport = &mut regs.viewport_transform[0];
    viewport.scale_x = width as f32 / 2.0;
    viewport.translate_x = width as f32 / 2.0;
    viewport.scale_y = height as f32 / 2.0;
    viewport.translate_y = height as f32 / 2.0;
    regs
}

/// Renders `color` into guest texels [x0, x1) x [y0, y1) of the source target, under a
/// scissor.
fn paint(f: &mut Fixture, gmmu: &Gmmu, [x0, y0, x1, y1]: [u32; 4], color: [u8; 4]) {
    let mut regs = target_regs(TEXTURE, SOURCE);
    let scissor = &mut regs.scissor_test[0];
    scissor.enable = 1;
    scissor.bits = ScissorTestBits(x0 | x1 << 16);
    scissor.bits1 = ScissorTestBits1(y0 | y1 << 16);
    write(f, PROGRAMS + u64::from(FRAGMENT), &fragment_program([0, 4]));
    write(
        f,
        FRAGMENT_CBUF,
        &f32_bytes(&color.map(|c| f32::from(c) / 255.0)),
    );
    f.rasterizer.invalidate_region(DADDR + FRAGMENT_CBUF, 16);
    draw(f, gmmu, &regs);
}

/// The source target painted green with its top-left quarter, [0, 32) x [0, 160), blue, bound
/// as texture header 0.
fn paint_source(f: &mut Fixture, gmmu: &Gmmu) {
    paint(f, gmmu, [0, 0, SOURCE.0, SOURCE.1], GREEN);
    paint(f, gmmu, [0, 0, 32, 160], BLUE);
    let header = tic(GPU_VA + TEXTURE, SOURCE.0, SOURCE.1);
    write(f, TIC, bytemuck::cast_slice(&header));
}

/// Draws the 16x16 target with `code` as the fragment program and the texture handle
/// `handle` in c[0][0x40].
fn read_with(f: &mut Fixture, gmmu: &Gmmu, regs: &mut Regs, code: &[u8], handle: u32) {
    let at = PROGRAMS + u64::from(READER);
    write(f, at, code);
    f.rasterizer
        .invalidate_region(DADDR + at, code.len() as u64);
    write(f, FRAGMENT_CBUF + 0x40, &handle.to_le_bytes());
    f.rasterizer
        .invalidate_region(DADDR + FRAGMENT_CBUF + 0x40, 4);
    regs.pipelines[5].offset = READER;
    regs.tex_sampler.limit = 1;
    draw(f, gmmu, regs);
}

/// TLDS.LZ of the 2D texture whose handle is c[0][`handle`] at integer texel (R`x`, R`y`),
/// storing the components `swizzle` selects as F32 from R`dest` (and R`dest_b`).
fn tlds(handle: u64, (x, y): (u64, u64), dest: u64, dest_b: u64, swizzle: u64) -> u64 {
    sass(
        Opcode::Tlds,
        (1 << 59)
            | (2 << 53)
            | (swizzle << 50)
            | (handle / 4) << 36
            | dest_b << 28
            | y << 20
            | x << 8
            | dest,
    )
}

/// Fetches texel (`x`, `y`) of the texture c[0][0x40] into render target 0.
fn fetch_program(x: u32, y: u32) -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let fetch = tlds(0x40, (4, 5), 0, 2, 4);
    program(header, &[mov_u32(4, x), mov_u32(5, y), fetch, exit()])
}

/// Samples the texture whose handle is c[0][0x40] at (`u`, `v`) into render target 0, with an
/// implicit level of detail.
fn filter_program(u: f32, v: f32) -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let tex = sass(
        Opcode::Tex,
        (0x10 << 36) | (7 << 51) | (0b1111 << 31) | (2 << 28) | (2 << 8),
    );
    program(header, &[mov_f32(2, u), mov_f32(3, v), tex, exit()])
}

/// Writes size component `component` (0 width, 1 height) of the texture c[0][0x40], as TXQ
/// returns it, to render target 0.
fn query_program(component: u32) -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let txq = sass(
        Opcode::Txq,
        (0x10 << 36) | (1 << (31 + component)) | (1 << 22) | (RZ << 8),
    );
    program(header, &[txq, exit()])
}

#[test]
fn texture_scaling_bits_number_descriptors_across_stages() {
    let mut scaling = super::ScalingBits::<4>::default();
    scaling.set(1);
    scaling.next_stage(3);
    scaling.set(0);
    scaling.next_stage(1);
    scaling.next_stage(30);
    scaling.set(0);
    scaling.set(2);
    scaling.set(200);
    assert_eq!(
        scaling.words,
        [1 << 1 | 1 << 3, 1 << (34 - 32) | 1 << (36 - 32), 0, 0],
        "bit 1 + 3 * 0, then 3 + 0, then 34 + 0 and 34 + 2; past 128 bits nothing"
    );
}

#[test]
fn draws_fetch_texels_of_rescaled_targets_at_guest_coordinates() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_source(&mut f, &gmmu);
    for (x, y, want) in [
        (0, 0, BLUE),
        (31, 159, BLUE),
        (32, 159, GREEN),
        (31, 160, GREEN),
        (63, 319, GREEN),
    ] {
        read_with(&mut f, &gmmu, &mut sampling_regs(), &fetch_program(x, y), 0);
        assert!(
            target(&mut f).iter().all(|&t| t == want),
            "texel ({x}, {y}) is {want:?}: {:?}",
            target(&mut f)[0]
        );
    }
    let source = f.rasterizer.draw.textures[0];
    let scale = |f: &Fixture, id| f.rasterizer.texture_cache.state(id).map(|s| s.scale);
    assert!(
        scale(&f, source) == Some(ResolutionScale::X2),
        "the source renders at 2x"
    );
    let target_id = f.rasterizer.draw.colors[0].expect("a bound target").id;
    assert!(
        scale(&f, target_id) == Some(ResolutionScale::X1),
        "the 16x16 target is too small to rescale, so scales differ"
    );
    assert_eq!(f.rasterizer.draw.texture_scaling.words, [1, 0, 0, 0]);
    f.assert_no_validation_errors();
}

#[test]
fn draws_query_rescaled_textures_at_their_guest_size() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_source(&mut f, &gmmu);
    let mut regs = sampling_regs();
    regs.rt[0].format = 0xe4;
    for (component, want) in [(0, SOURCE.0), (1, SOURCE.1)] {
        read_with(&mut f, &gmmu, &mut regs, &query_program(component), 0);
        let texels = target(&mut f);
        assert!(
            texels.iter().all(|&t| u32::from_le_bytes(t) == want),
            "size component {component} is {want}: {}",
            u32::from_le_bytes(texels[0])
        );
    }
    f.assert_no_validation_errors();
}

#[test]
fn draws_sample_images_rescaled_at_an_earlier_scale_at_the_new_scale() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_source(&mut f, &gmmu);
    f.rasterizer.apply_settings(&GraphicsSettings {
        resolution_scale: ResolutionScale::X3,
        async_shader_compilation: false,
        ..GraphicsSettings::default()
    });
    let mut regs = sampling_regs();
    regs.rt[0].format = 0xe4;
    for (component, want) in [(0, SOURCE.0), (1, SOURCE.1)] {
        read_with(&mut f, &gmmu, &mut regs, &query_program(component), 0);
        assert!(
            target(&mut f)
                .iter()
                .all(|&t| u32::from_le_bytes(t) == want),
            "size component {component} is {want} after the scale change"
        );
    }
    f.assert_no_validation_errors();
}

#[test]
fn draws_filter_rescaled_textures_at_guest_coordinates() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_source(&mut f, &gmmu);
    let mut linear_tsc = [0u32; 8];
    linear_tsc[1] = 2 | 2 << 4;
    write(&f, TSC + 32, bytemuck::cast_slice(&linear_tsc));
    let half = [0, 128, 128, 255];
    let (w, h) = (SOURCE.0 as f32, SOURCE.1 as f32);
    for (u, v, want) in [
        (31.5 / w, 0.5 / h, BLUE),
        (32.5 / w, 0.5 / h, GREEN),
        (32.0 / w, 0.5 / h, half),
        (0.5 / w, 160.0 / h, half),
        (0.5 / w, 160.5 / h, GREEN),
    ] {
        read_with(
            &mut f,
            &gmmu,
            &mut sampling_regs(),
            &filter_program(u, v),
            1 << 20,
        );
        let texels = target(&mut f);
        assert!(
            texels
                .iter()
                .all(|&t| (0..4).all(|c| t[c].abs_diff(want[c]) <= 1)),
            "({u}, {v}) filters to {want:?}: {:?}",
            texels[0]
        );
    }
    f.assert_no_validation_errors();
}

#[test]
fn draws_set_the_scaling_bit_of_each_rescaled_descriptor() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_source(&mut f, &gmmu);
    uniform_texture(&f, 1, OTHER_TEXTURE, [10, 200, 30, 255]);
    write(&f, FRAGMENT_CBUF + 0x44, &0u32.to_le_bytes());
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let code = program(
        header,
        &[
            mov_u32(4, 3),
            mov_u32(5, 3),
            mov_u32(6, 32),
            mov_u32(7, 159),
            tlds(0x40, (4, 5), 0, RZ, 4),
            tlds(0x44, (6, 7), 2, RZ, 7),
            exit(),
        ],
    );
    read_with(&mut f, &gmmu, &mut sampling_regs(), &code, 1);
    assert_eq!(
        f.rasterizer.draw.texture_scaling.words,
        [0b10, 0, 0, 0],
        "c[0][0x40] is the unscaled texture 1, c[0][0x44] the 2x source"
    );
    assert!(
        target(&mut f)
            .iter()
            .all(|&t| t == [10, 200, GREEN[2], GREEN[3]]),
        "red and green of the unscaled texture, blue and alpha of source texel (32, 159): {:?}",
        target(&mut f)[0]
    );
    f.assert_no_validation_errors();
}

/// Renders source texel (x, y) of [0, 4) x [0, 4) as `source_texel` into the green source.
fn paint_texels(f: &mut Fixture, gmmu: &Gmmu) {
    paint(f, gmmu, [0, 0, SOURCE.0, SOURCE.1], GREEN);
    for y in 0..4 {
        for x in 0..4 {
            paint(f, gmmu, [x, y, x + 1, y + 1], source_texel(x, y));
        }
    }
    let header = tic(GPU_VA + TEXTURE, SOURCE.0, SOURCE.1);
    write(f, TIC, bytemuck::cast_slice(&header));
}

/// The painted source at guest texel (x, y).
fn painted(x: u32, y: u32) -> [u8; 4] {
    if x < 4 && y < 4 {
        source_texel(x, y)
    } else {
        GREEN
    }
}

/// The `width`x`height` target at `offset` after writing it back to guest memory.
fn read_target(f: &mut Fixture, offset: u64, (width, height): (u32, u32)) -> Vec<[u8; 4]> {
    let tiling = Tiling::new(width, height, 1, 4, 1, 0, GobLayout::Tegra);
    f.rasterizer
        .flush_region(DADDR + offset, tiling.size_bytes() as u64);
    guest_linear(f, DADDR + offset, &tiling)
        .as_chunks::<4>()
        .0
        .to_vec()
}

#[test]
fn draw_texture_reads_rescaled_sources_exactly() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_texels(&mut f, &gmmu);
    let rect = |dst: [f32; 4], src: [f32; 4]| DrawTexture {
        dst_x0: dst[0],
        dst_y0: dst[1],
        dst_x1: dst[2],
        dst_y1: dst[3],
        src_x0: src[0],
        src_y0: src[1],
        src_x1: src[2],
        src_y1: src[3],
        texture: 0,
        sampler: 0,
    };
    let draws = [
        rect([0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0]),
        rect([8.0, 8.0, 16.0, 16.0], [0.0, 0.0, 4.0, 4.0]),
        rect([0.0, 8.0, 4.0, 12.0], [2.0, 2.0, 6.0, 6.0]),
    ];
    let want = |x: u32, y: u32| {
        if x < 4 && y < 4 {
            painted(x, y)
        } else if x >= 8 && (8..16).contains(&y) && x < 16 {
            painted((x - 8) / 2, (y - 8) / 2)
        } else if x < 4 && (8..12).contains(&y) {
            painted(x + 2, y - 6)
        } else {
            [0; 4]
        }
    };
    // Into the 16x16 target, too small to rescale, then into a 64x320 one rendered at 2x.
    let targets = [(0, (SIZE, SIZE)), (OTHER_TEXTURE, SOURCE)];
    for (offset, size) in targets {
        let regs = target_regs(offset, size);
        for draw in draws {
            draw_texture(&mut f, &gmmu, &regs, draw);
        }
        let texels = read_target(&mut f, offset, size);
        for (i, &texel) in texels.iter().enumerate() {
            let (x, y) = (i as u32 % size.0, i as u32 / size.0);
            assert_eq!(texel, want(x, y), "texel ({x}, {y}) of the {size:?} target");
        }
    }
    let scale = |f: &Fixture, id| f.rasterizer.texture_cache.state(id).map(|s| s.scale);
    let target_id = f.rasterizer.draw.colors[0].expect("a bound target").id;
    assert!(scale(&f, target_id) == Some(ResolutionScale::X2));
    f.assert_no_validation_errors();
}

const PAGE: u64 = 0x1000;
const SSBO: u64 = 0x60_0000;

/// Whether the written-pages bitmap may hold a GPU write in guest `[offset, +size)`.
fn written(f: &Fixture, offset: u64, size: u64) -> bool {
    f.rasterizer
        .handles
        .written_pages
        .may_be_written(DADDR + offset, size)
}

/// Exactly the `pages` pages from `offset` are marked, and not the page on either side.
fn assert_marked(f: &Fixture, offset: u64, pages: u64, what: &str) {
    for page in 0..pages {
        assert!(
            written(f, offset + page * PAGE, 1),
            "{what}: page {page} is marked"
        );
    }
    assert!(!written(f, offset - PAGE, PAGE), "{what}: the page before");
    assert!(
        !written(f, offset + pages * PAGE, PAGE),
        "{what}: the page after"
    );
}

/// Stores 0xdeadbeef through the storage buffer c[0][0x110] (address) and c[0][0x118] (size)
/// describe, and writes render target 0 from the zeroed registers.
fn storing_program() -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let stg = sass(Opcode::Stg, (1 << 45) | (4 << 48) | (4 << 8) | 6);
    program(
        header,
        &[
            mov_cbuf(4, 0x110),
            mov_cbuf(5, 0x114),
            mov_u32(6, 0xdead_beef),
            stg,
            exit(),
        ],
    )
}

#[test]
fn draws_mark_the_pages_of_their_writable_storage_buffers() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let buffer = SSBO + PAGE;
    let mut binding = (GPU_VA + buffer).to_le_bytes().to_vec();
    binding.extend_from_slice(&0x100u32.to_le_bytes());
    write(&f, FRAGMENT_CBUF + 0x110, &binding);
    write(&f, PROGRAMS + u64::from(FRAGMENT), &storing_program());
    assert!(!written(&f, SSBO, 3 * PAGE));
    draw(&mut f, &gmmu, &draw_regs());
    assert_marked(&f, buffer, 1, "the storage buffer");
    assert!(written(&f, buffer, 0x100));
    f.rasterizer.flush_region(DADDR + buffer, 4);
    let mut word = [0; 4];
    f.memory.read_bytes(DADDR + buffer, &mut word).unwrap();
    assert_eq!(u32::from_le_bytes(word), 0xdead_beef, "the program stored");
    f.assert_no_validation_errors();
}

#[test]
fn draws_clears_and_texture_draws_mark_their_targets() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let source_pages = 20;
    assert!(!written(&f, 0, 0x60_0000));

    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &fragment_program([0, 4]),
    );
    draw(&mut f, &gmmu, &draw_regs());
    assert!(written(&f, 0, 1));
    assert!(!written(&f, PAGE, PAGE), "the 16x16 target takes 1 KiB");

    let mut regs = target_regs(TEXTURE, SOURCE);
    regs.clear_surface = ClearSurface(0b1111 << 2);
    regs.clear_color = [0.0, 1.0, 0.0, 1.0];
    let mut dirty = DirtyState::new();
    let const_buffers = const_buffers();
    let root_tables = [[0; 64]; 8];
    let mut ctx = GraphicsContext {
        channel: ChannelId(0),
        memory: &gmmu,
        regs: &regs,
        dirty: &mut dirty,
        const_buffers: &const_buffers,
        root_tables: &root_tables,
    };
    f.rasterizer.clear(&mut ctx, 1);
    assert_marked(&f, TEXTURE, source_pages, "the cleared 64x320 target");

    let header = tic(GPU_VA + TEXTURE, SOURCE.0, SOURCE.1);
    write(&f, TIC, bytemuck::cast_slice(&header));
    assert!(!written(&f, OTHER_TEXTURE, source_pages * PAGE));
    let rect = DrawTexture {
        dst_x0: 0.0,
        dst_y0: 0.0,
        dst_x1: 4.0,
        dst_y1: 4.0,
        src_x0: 0.0,
        src_y0: 0.0,
        src_x1: 4.0,
        src_y1: 4.0,
        texture: 0,
        sampler: 0,
    };
    draw_texture(&mut f, &gmmu, &target_regs(OTHER_TEXTURE, SOURCE), rect);
    assert_marked(&f, OTHER_TEXTURE, source_pages, "the texture draw's target");
    f.assert_no_validation_errors();
}

const TEXEL_BUFFER: u64 = 0x68_0000;
const STORAGE_IMAGE: u64 = 0x6c_0000;

/// A one-dimensional A8B8G8R8 buffer texture header of `texels` texels at `gpu_va`.
fn buffer_tic(gpu_va: u64, texels: u32) -> [u32; 8] {
    let last = texels - 1;
    [
        tic(gpu_va, 1, 1)[0],
        gpu_va as u32,
        (gpu_va >> 32) as u32,
        last >> 16,
        (last & 0xffff) | 6 << 23,
        0,
        0,
        0,
    ]
}

/// A fragment program running `code` that writes render target 0 from R0..R3.
fn pixel_program(code: &[u64]) -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let mut code = code.to_vec();
    code.push(exit());
    program(header, &code)
}

/// SUST.D.32 (bound, typed R32) of R6 to the storage image or image buffer whose handle is
/// c[0][0x40], at the coordinates from R4 (and R5): `dim` 1 is a buffer, 3 a 2D image.
fn sust_r32(dim: u64) -> u64 {
    sass(
        Opcode::Sust,
        (1 << 52) | (1 << 51) | (0x10 << 36) | (dim << 33) | (4 << 20) | (4 << 8) | 6,
    )
}

/// Texel `i` of the texel buffer: every texel distinct.
fn buffer_texel(i: u32) -> [u8; 4] {
    [i as u8, (i * 3) as u8, 255 - i as u8, 255]
}

#[test]
fn draws_fetch_uniform_texel_buffers() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let texels: Vec<u8> = (0..64).flat_map(buffer_texel).collect();
    write(&f, TEXEL_BUFFER, &texels);
    let header = buffer_tic(GPU_VA + TEXEL_BUFFER, 64);
    write(&f, TIC, bytemuck::cast_slice(&header));
    let tld = sass(
        Opcode::Tld,
        (0x10 << 36) | (0b1111 << 31) | (RZ << 20) | (4 << 8),
    );
    for i in [5, 63] {
        let code = pixel_program(&[mov_u32(4, i), tld]);
        read_with(&mut f, &gmmu, &mut sampling_regs(), &code, 0);
        assert!(
            target(&mut f).iter().all(|&t| t == buffer_texel(i)),
            "texel {i}: {:?}",
            target(&mut f)[0]
        );
    }
    f.assert_no_validation_errors();
}

#[test]
fn draws_store_texel_buffers_for_flushes() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let header = buffer_tic(GPU_VA + TEXEL_BUFFER, 1024);
    write(&f, TIC, bytemuck::cast_slice(&header));
    assert!(!written(&f, TEXEL_BUFFER, PAGE));
    let code = pixel_program(&[mov_u32(4, 7), mov_u32(6, 0xdead_beef), sust_r32(1)]);
    read_with(&mut f, &gmmu, &mut sampling_regs(), &code, 0);
    assert_marked(&f, TEXEL_BUFFER, 1, "the image buffer");
    f.rasterizer.flush_region(DADDR + TEXEL_BUFFER, 4096);
    let mut words = [0u8; 32];
    f.memory
        .read_bytes(DADDR + TEXEL_BUFFER, &mut words)
        .unwrap();
    let words: Vec<u32> = words
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect();
    assert_eq!(words, [0, 0, 0, 0, 0, 0, 0, 0xdead_beef]);
    f.assert_no_validation_errors();
}

#[test]
fn draws_store_storage_images_for_sampling_and_flushes() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let header = tic(GPU_VA + STORAGE_IMAGE, SIZE, SIZE);
    write(&f, TIC, bytemuck::cast_slice(&header));
    assert!(!written(&f, STORAGE_IMAGE, PAGE));
    let code = pixel_program(&[
        mov_u32(4, 3),
        mov_u32(5, 5),
        mov_u32(6, 0xff40_ff00),
        sust_r32(3),
    ]);
    read_with(&mut f, &gmmu, &mut sampling_regs(), &code, 0);
    assert_marked(&f, STORAGE_IMAGE, 1, "the storage image");
    let stored = [0x00, 0xff, 0x40, 0xff];

    read_with(&mut f, &gmmu, &mut sampling_regs(), &fetch_program(3, 5), 0);
    assert!(
        target(&mut f).iter().all(|&t| t == stored),
        "the sampled texel: {:?}",
        target(&mut f)[0]
    );
    read_with(&mut f, &gmmu, &mut sampling_regs(), &fetch_program(4, 5), 0);
    assert!(target(&mut f).iter().all(|&t| t == [0; 4]), "other texels");

    let tiling = Tiling::new(SIZE, SIZE, 1, 4, 1, 0, GobLayout::Tegra);
    let image = read_target(&mut f, STORAGE_IMAGE, (SIZE, SIZE));
    assert_eq!(tiling.size_bytes(), image.len() * 4);
    for (i, &texel) in image.iter().enumerate() {
        let want = if i == 5 * SIZE as usize + 3 {
            stored
        } else {
            [0; 4]
        };
        assert_eq!(texel, want, "flushed texel {i}");
    }
    f.assert_no_validation_errors();
}

#[test]
fn storage_writes_to_rescaled_images_scale_them_down() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_source(&mut f, &gmmu);
    let source = |f: &Fixture| f.rasterizer.draw.textures.first().copied();
    read_with(&mut f, &gmmu, &mut sampling_regs(), &fetch_program(0, 0), 0);
    let id = source(&f).expect("the source was sampled");
    let scale = |f: &Fixture| f.rasterizer.texture_cache.state(id).map(|s| s.scale);
    assert!(scale(&f) == Some(ResolutionScale::X2));
    let code = pixel_program(&[
        mov_u32(4, 40),
        mov_u32(5, 200),
        mov_u32(6, 0xff40_ff00),
        sust_r32(3),
    ]);
    read_with(&mut f, &gmmu, &mut sampling_regs(), &code, 0);
    assert!(
        scale(&f) == Some(ResolutionScale::X1),
        "scaled down for the store"
    );
    assert_eq!(f.rasterizer.draw.image_scaling.words, [0, 0]);
    for (x, y, want) in [
        (40, 200, [0x00, 0xff, 0x40, 0xff]),
        (31, 159, BLUE),
        (32, 159, GREEN),
    ] {
        read_with(&mut f, &gmmu, &mut sampling_regs(), &fetch_program(x, y), 0);
        assert!(
            target(&mut f).iter().all(|&t| t == want),
            "texel ({x}, {y}) is {want:?}: {:?}",
            target(&mut f)[0]
        );
    }
    f.assert_no_validation_errors();
}

#[test]
fn draws_load_rescaled_storage_images_at_guest_coordinates() {
    let Some((mut f, gmmu)) = rescaled_fixture() else {
        return;
    };
    paint_source(&mut f, &gmmu);
    let suld = sass(
        Opcode::Suld,
        (1 << 52) | (1 << 51) | (0x10 << 36) | (3 << 33) | (4 << 20) | (4 << 8),
    );
    let mut regs = sampling_regs();
    regs.rt[0].format = 0xe4;
    for (x, y, want) in [(31, 159, BLUE), (32, 159, GREEN), (31, 160, GREEN)] {
        let code = pixel_program(&[mov_u32(4, x), mov_u32(5, y), suld]);
        read_with(&mut f, &gmmu, &mut regs, &code, 0);
        let texels = target(&mut f);
        assert!(
            texels.iter().all(|&t| t == want),
            "texel ({x}, {y}) is {want:?}: {:?}",
            texels[0]
        );
    }
    let (id, written) = f.rasterizer.draw.storage_images[0];
    assert!(!written);
    let scale = f.rasterizer.texture_cache.state(id).map(|s| s.scale);
    assert!(
        scale == Some(ResolutionScale::X2),
        "read-only images keep their scale"
    );
    assert_eq!(f.rasterizer.draw.image_scaling.words, [1, 0]);
    f.assert_no_validation_errors();
}
