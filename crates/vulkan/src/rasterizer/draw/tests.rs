//! Draws of hand-assembled Maxwell programs on synthetic 3D engine state.

use lichii_common::settings::{Anisotropy, GraphicsSettings, ResolutionScale};
use lichii_gpu::backend::{
    ChannelId, ComputeContext, ConstBufferBinding, DrawParams, DrawTexture, GraphicsConstBuffers,
    GraphicsContext, IndirectDraw, QmdVersion, Rasterizer,
};
use lichii_gpu::engines::maxwell_3d::regs::{
    ClearSurface, DepthMode, IndexFormat, MAX_CONST_BUFFERS, MAX_SHADER_STAGE, MESH_CONTROL,
    MESH_INIT_SHADER, MESH_SHADER_A, MESH_SHADER_B, PipelineBits, PrimitiveTopology,
    RenderTargetConfigBits, RtControl, ScissorTestBits, ScissorTestBits1, ShaderType,
    SurfaceClipBits, SurfaceClipBits1, TileMode, VertexAttribute, VertexStreamBits,
};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};
use lichii_gpu::gmmu::Gmmu;
use lichii_shader::ampere::{
    Decoded, Form, GRAPHICS_HEADER_SIZE, Instruction, Opcode as AmpereOpcode, decode,
};
use lichii_shader::maxwell::opcode::Opcode;
use lichii_shader::{DescriptorKind, ShaderStage};
use lichii_texture::block_linear::{GobLayout, Tiling, swizzle};

use super::super::tests::{DADDR, Fixture, GPU_VA, fixture, gmmu, guest_linear};
use super::{StageSources, Warning};
use crate::pipeline::environment::TexturePool;
use crate::pipeline::shader_cache::ShaderCache;
use crate::rasterizer::VulkanRasterizer;

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
            indirect_grid: None,
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

const INDICES: u64 = 0x70_0000;
const COMMANDS: u64 = 0x71_0000;
const COUNT: u64 = 0x72_0000;
const ORANGE: [u8; 4] = [255, 128, 0, 255];

/// `draw_regs` over the XY `vertices`, with the u16 `indices` bound when set.
fn assembly_regs(
    f: &Fixture,
    vertices: &[f32],
    indices: Option<&[u16]>,
    restart: bool,
) -> Box<Regs> {
    let mut regs = draw_regs();
    write(f, VERTICES, &f32_bytes(vertices));
    let limit = GPU_VA + VERTICES + 4 * vertices.len() as u64 - 1;
    regs.vertex_stream_limits[0].address_high = (limit >> 32) as u32;
    regs.vertex_stream_limits[0].address_low = limit as u32;
    if let Some(indices) = indices {
        write(f, INDICES, bytemuck::cast_slice(indices));
        let start = GPU_VA + INDICES;
        let limit = start + 2 * indices.len() as u64 - 1;
        let buffer = &mut regs.index_buffer;
        buffer.start_addr_high = (start >> 32) as u32;
        buffer.start_addr_low = start as u32;
        buffer.limit_addr_high = (limit >> 32) as u32;
        buffer.limit_addr_low = limit as u32;
        buffer.format = IndexFormat::UNSIGNED_SHORT;
    }
    regs.primitive_restart.enabled = u32::from(restart);
    regs
}

/// The target after `issue` draws over `assembly_regs` on a fresh fixture, with `memory`
/// written first, as rows of `#` for drawn and `.` for untouched texels; `None` without a
/// Vulkan device.
fn render(
    vertices: &[f32],
    indices: Option<&[u16]>,
    restart: bool,
    memory: &[(u64, &[u32])],
    issue: impl FnOnce(&mut VulkanRasterizer, &mut GraphicsContext<'_>),
) -> Option<Vec<String>> {
    let (mut f, gmmu) = draw_fixture()?;
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &fragment_program([0, 4]),
    );
    for &(offset, words) in memory {
        write(&f, offset, bytemuck::cast_slice(words));
    }
    let regs = assembly_regs(&f, vertices, indices, restart);
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
    issue(&mut f.rasterizer, &mut ctx);
    let texels = target(&mut f);
    f.assert_no_validation_errors();
    let picture = texels
        .chunks(SIZE as usize)
        .map(|row| {
            row.iter()
                .map(|&t| match t {
                    ORANGE => '#',
                    [0, 0, 0, 0] => '.',
                    _ => '?',
                })
                .collect()
        })
        .collect();
    Some(picture)
}

fn direct(
    topology: PrimitiveTopology,
    indexed: bool,
    count: u32,
) -> impl FnOnce(&mut VulkanRasterizer, &mut GraphicsContext<'_>) {
    move |r, ctx| {
        r.draw(
            ctx,
            &DrawParams {
                topology,
                indexed,
                first: 0,
                count,
                base_vertex: 0,
                base_instance: 0,
                instance_count: 1,
                inline_indices: None,
            },
        )
    }
}

fn indirect(
    topology: PrimitiveTopology,
    indexed: bool,
    stride: u32,
    max_draws: u32,
    count_buffer: Option<u64>,
) -> impl FnOnce(&mut VulkanRasterizer, &mut GraphicsContext<'_>) {
    move |r, ctx| {
        r.draw_indirect(
            ctx,
            &IndirectDraw {
                topology,
                indexed,
                buffer: GPU_VA + COMMANDS,
                stride,
                max_draws,
                count_buffer: count_buffer.map(|offset| GPU_VA + offset),
            },
        )
    }
}

fn colored(picture: &[String]) -> usize {
    picture.iter().map(|row| row.matches('#').count()).sum()
}

/// Quad A covers the left half of the target, quad B a quarter of the right half.
const TWO_QUADS: [f32; 16] = [
    -1.0, -1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 1.0, //
    0.0, -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0,
];

#[test]
fn indirect_quads_draw_the_pixels_of_direct_quads() {
    let Some(expected) = render(
        &TWO_QUADS,
        None,
        false,
        &[],
        direct(PrimitiveTopology::QUADS, false, 8),
    ) else {
        return;
    };
    assert_eq!(colored(&expected), 128 + 64, "both quads drew");

    let commands = [4, 1, 0, 0, 4, 1, 4, 0];
    let arrays = render(
        &TWO_QUADS,
        None,
        false,
        &[(COMMANDS, &commands)],
        indirect(PrimitiveTopology::QUADS, false, 16, 2, None),
    )
    .unwrap();
    assert_eq!(arrays, expected, "one quad per non-indexed command");

    let indices = [7, 0, 1, 2, 3, 2, 3, 4, 5];
    let commands = [4, 1, 1, 0, 0, 0, 4, 1, 5, 2, 0, 0];
    let elements = render(
        &TWO_QUADS,
        Some(&indices),
        false,
        &[(COMMANDS, &commands)],
        indirect(PrimitiveTopology::QUADS, true, 24, 2, None),
    )
    .unwrap();
    assert_eq!(
        elements, expected,
        "indexed commands with their first index and vertex offset"
    );
}

#[test]
fn indirect_quad_strips_restart_like_direct_ones() {
    let vertices = [
        -1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0, //
        0.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0,
    ];
    let indices = [0, 1, 2, 3, 0xffff, 4, 5, 6, 7];
    let Some(expected) = render(
        &vertices,
        Some(&indices),
        true,
        &[],
        direct(PrimitiveTopology::QUAD_STRIP, true, 9),
    ) else {
        return;
    };
    assert_eq!(colored(&expected), 128 + 64, "both strips drew");
    let drawn = render(
        &vertices,
        Some(&indices),
        true,
        &[(COMMANDS, &[9, 1, 0, 0, 0])],
        indirect(PrimitiveTopology::QUAD_STRIP, true, 20, 1, None),
    )
    .unwrap();
    assert_eq!(drawn, expected);
}

#[test]
fn indirect_line_loops_close() {
    let (near, far) = (2.5 / 8.0 - 1.0, 13.5 / 8.0 - 1.0);
    let square = [near, near, far, near, far, far, near, far];
    let left_edge = |picture: &[String]| picture[8].as_bytes()[2] == b'#';
    let Some(expected) = render(
        &square,
        None,
        false,
        &[],
        direct(PrimitiveTopology::LINE_LOOP, false, 4),
    ) else {
        return;
    };
    assert!(left_edge(&expected), "the closing line drew");
    assert_eq!(expected[8].as_bytes()[8], b'.', "lines only");
    let drawn = render(
        &square,
        None,
        false,
        &[(COMMANDS, &[4, 1, 0, 0])],
        indirect(PrimitiveTopology::LINE_LOOP, false, 16, 1, None),
    )
    .unwrap();
    assert_eq!(drawn, expected);

    let mut vertices = square.to_vec();
    vertices.extend([-0.25, -0.25, 0.25, -0.25, 0.0, 0.25]);
    let indices = [0, 1, 2, 3, 0xffff, 4, 5, 6];
    let expected = render(
        &vertices,
        Some(&indices),
        true,
        &[],
        direct(PrimitiveTopology::LINE_LOOP, true, 8),
    )
    .unwrap();
    let drawn = render(
        &vertices,
        Some(&indices),
        true,
        &[(COMMANDS, &[8, 1, 0, 0, 0])],
        indirect(PrimitiveTopology::LINE_LOOP, true, 20, 1, None),
    )
    .unwrap();
    assert!(left_edge(&drawn), "the first loop closed");
    assert_eq!(drawn, expected, "each run between restart indices closes");
}

#[test]
fn indirect_quads_draw_only_the_commands_the_count_buffer_names() {
    let Some(expected) = render(
        &TWO_QUADS,
        None,
        false,
        &[],
        direct(PrimitiveTopology::QUADS, false, 4),
    ) else {
        return;
    };
    assert_eq!(colored(&expected), 128, "quad A alone");
    let commands = [4, 1, 0, 0, 4, 1, 4, 0];
    let drawn = render(
        &TWO_QUADS,
        None,
        false,
        &[(COMMANDS, &commands), (COUNT, &[1])],
        indirect(PrimitiveTopology::QUADS, false, 16, 2, Some(COUNT)),
    )
    .unwrap();
    assert_eq!(drawn, expected);
}

#[test]
fn indirect_assembly_carries_runs_across_workgroup_chunks() {
    let mut indices = vec![0, 1, 2, 3];
    indices.resize(62, 0xffff);
    indices.extend([4, 5, 6, 7]);
    let Some(expected) = render(
        &TWO_QUADS,
        Some(&indices),
        true,
        &[],
        direct(PrimitiveTopology::QUADS, true, indices.len() as u32),
    ) else {
        return;
    };
    assert_eq!(colored(&expected), 128 + 64, "both quads drew");
    let count = indices.len() as u32;
    let drawn = render(
        &TWO_QUADS,
        Some(&indices),
        true,
        &[(COMMANDS, &[count, 1, 0, 0, 0])],
        indirect(PrimitiveTopology::QUADS, true, 20, 1, None),
    )
    .unwrap();
    assert_eq!(
        drawn, expected,
        "quad B spans indices 62..66, across the first 64-index chunk"
    );
}

const COUNTER: u64 = 0x74_0000;
const GRID: u64 = 0x75_0000;
const COUNTER_CBUF: u64 = 0x76_0000;
const GRID_CBUF: u64 = 0x76_1000;
const COUNTING: u64 = 0x10_8000;
const GRID_WRITER: u64 = 0x10_9000;

/// Compute `code` with a scheduling word in the first slot of every 32 bytes.
fn compute_code(code: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::new();
    for insn in code {
        if bytes.len().is_multiple_of(32) {
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    bytes
}

/// Every workgroup of one thread adds 1 to the word at c[0][0x110] and its CTAID.Y to the next.
fn counting_program() -> Vec<u8> {
    compute_code(&[
        mov_cbuf(4, 0x110),
        mov_cbuf(5, 0x114),
        mov_u32(6, 1),
        sass(Opcode::S2r, (38 << 20) | 7),
        sass(Opcode::Red, (1 << 48) | (4 << 8) | 6),
        sass(Opcode::Red, (1 << 48) | (4 << 28) | (4 << 8) | 7),
        exit(),
    ])
}

/// Stores `words` at c[0][0x110].
fn grid_writer_program(words: [u32; 2]) -> Vec<u8> {
    let stg = |data: u64, offset: u64| {
        sass(
            Opcode::Stg,
            (1 << 45) | (4 << 48) | (offset << 20) | (4 << 8) | data,
        )
    };
    compute_code(&[
        mov_cbuf(4, 0x110),
        mov_cbuf(5, 0x114),
        mov_u32(6, words[0]),
        mov_u32(7, words[1]),
        stg(6, 0),
        stg(7, 4),
        exit(),
    ])
}

/// Points the storage buffer at c[0][0x110] of the constant buffer at `cbuf` to `buffer`.
fn storage_binding(f: &Fixture, cbuf: u64, buffer: u64) {
    let mut binding = (GPU_VA + buffer).to_le_bytes().to_vec();
    binding.extend_from_slice(&0x100u32.to_le_bytes());
    write(f, cbuf + 0x110, &binding);
}

fn launch(
    f: &mut Fixture,
    gmmu: &Gmmu,
    program: u64,
    cbuf: u64,
    grid: [u32; 3],
    indirect_grid: Option<u64>,
) {
    let mut const_buffers = [ConstBufferBinding::default(); 8];
    const_buffers[0] = ConstBufferBinding {
        address: GPU_VA + cbuf,
        size: 0x200,
        enabled: true,
    };
    let qmd = [0u32; 64];
    f.rasterizer.dispatch_compute(&ComputeContext {
        channel: ChannelId(0),
        memory: gmmu,
        program: GPU_VA + program,
        grid,
        block: [1; 3],
        shared_memory_size: 0,
        local_memory_size: 0,
        const_buffers: &const_buffers,
        texture_headers: 0,
        texture_header_limit: 0,
        samplers: 0,
        sampler_limit: 0,
        linked_samplers: false,
        qmd_version: QmdVersion::V2,
        qmd: &qmd,
        indirect_grid: indirect_grid.map(|offset| GPU_VA + offset),
    });
}

/// The workgroups that ran and the sum of their CTAID.Y.
fn counted(f: &mut Fixture) -> [u32; 2] {
    f.rasterizer.flush_region(DADDR + COUNTER, 8);
    let mut words = [0u32; 2];
    f.memory
        .read_bytes(DADDR + COUNTER, bytemuck::cast_slice_mut(&mut words))
        .unwrap();
    words
}

#[test]
fn indirect_dispatches_unpack_the_descriptor_grid() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(&f, COUNTING, &counting_program());
    storage_binding(&f, COUNTER_CBUF, COUNTER);
    write(&f, COUNTER, &[0; 8]);
    let packed = [3 | 1 << 31, 4 | 5 << 16, 7];
    let grid = GRID + 0x24;
    write(&f, grid, bytemuck::cast_slice(&packed));
    launch(&mut f, &gmmu, COUNTING, COUNTER_CBUF, [1; 3], Some(grid));
    assert_eq!(
        counted(&mut f),
        [3 * 4 * 5, 3 * 5 * (1 + 2 + 3)],
        "width 3 (bit 31 is another field), height 4 and depth 5"
    );
    assert_eq!(f.rasterizer.draw.shaders.stats().failed, 0);
    f.assert_no_validation_errors();
}

#[test]
fn indirect_dispatches_read_the_grid_earlier_gpu_work_wrote() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(&f, COUNTING, &counting_program());
    write(&f, GRID_WRITER, &grid_writer_program([2, 3 | 4 << 16]));
    storage_binding(&f, COUNTER_CBUF, COUNTER);
    storage_binding(&f, GRID_CBUF, GRID);
    write(&f, COUNTER, &[0; 8]);
    write(&f, GRID, bytemuck::cast_slice(&[1u32, 1 | 1 << 16]));
    launch(&mut f, &gmmu, GRID_WRITER, GRID_CBUF, [1; 3], None);
    launch(&mut f, &gmmu, COUNTING, COUNTER_CBUF, [1; 3], Some(GRID));
    assert_eq!(
        counted(&mut f),
        [2 * 3 * 4, 2 * 4 * (1 + 2)],
        "the 2x3x4 grid the first dispatch stored, not the guest's 1x1x1"
    );
    assert_eq!(f.rasterizer.draw.shaders.stats().failed, 0);
    f.assert_no_validation_errors();
}

/// A handle table in the storage buffer c[0][0x110] describes, at a pointer no storage
/// alignment divides.
const TABLE: u64 = SSBO + 0x24;

/// Samples at (0.5, 0.5), into render target 0, the texture whose handle is word 4 of the
/// table the storage buffer c[0][0x110] points to.
fn table_sampling_program() -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let ldg = sass(
        Opcode::Ldg,
        (4 << 48) | (1 << 45) | (0x10 << 20) | (4 << 8) | 6,
    );
    let tex = sass(
        Opcode::TexB,
        (7 << 51) | (0b1111 << 31) | (2 << 28) | (6 << 20) | (2 << 8),
    );
    program(
        header,
        &[
            mov_cbuf(4, 0x110),
            mov_cbuf(5, 0x114),
            ldg,
            mov_f32(2, 0.5),
            mov_f32(3, 0.5),
            tex,
            exit(),
        ],
    )
}

/// Points the storage buffer c[0][0x110] at `TABLE` and stores `handle` as its word 4.
fn handle_table(f: &mut Fixture, handle: u32) {
    let mut binding = (GPU_VA + TABLE).to_le_bytes().to_vec();
    binding.extend_from_slice(&0x100u32.to_le_bytes());
    write(f, FRAGMENT_CBUF + 0x110, &binding);
    write(f, TABLE + 0x10, &handle.to_le_bytes());
    f.rasterizer.invalidate_region(DADDR + TABLE + 0x10, 4);
}

#[test]
fn draws_bind_the_textures_handle_programs_read_from_storage_tables() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &table_sampling_program(),
    );
    uniform_texture(&f, 0, TEXTURE, [10, 200, 30, 255]);
    uniform_texture(&f, 1, OTHER_TEXTURE, [50, 60, 70, 255]);
    let regs = sampling_regs();
    handle_table(&mut f, 1);
    draw(&mut f, &gmmu, &regs);
    assert!(
        target(&mut f).iter().all(|&t| t == [50, 60, 70, 255]),
        "the draw samples the texture the table names"
    );
    let translated = f.rasterizer.draw.shaders.stats().translated;

    handle_table(&mut f, 0);
    draw(&mut f, &gmmu, &regs);
    assert!(
        target(&mut f).iter().all(|&t| t == [10, 200, 30, 255]),
        "the next draw evaluates the changed table"
    );
    assert_eq!(
        f.rasterizer.draw.shaders.stats().translated,
        translated,
        "a handle program needs no new translation"
    );
    assert_eq!(f.rasterizer.draw.shaders.stats().failed, 0);
    f.assert_no_validation_errors();
}

#[test]
fn handle_programs_naming_an_image_of_another_kind_bind_the_null_texture() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &table_sampling_program(),
    );
    uniform_texture(&f, 0, TEXTURE, [10, 200, 30, 255]);
    let mut integer = tic(GPU_VA + TEXTURE, SIZE, SIZE);
    integer[0] =
        0x08 | 4 << 7 | 4 << 10 | 4 << 13 | 4 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28;
    write(&f, TIC + 32, bytemuck::cast_slice(&integer));
    let regs = sampling_regs();
    handle_table(&mut f, 1);
    draw(&mut f, &gmmu, &regs);
    assert!(
        target(&mut f).iter().all(|&t| t == [0; 4]),
        "an integer image under a float descriptor reads the null texture"
    );

    let mut array = tic(GPU_VA + TEXTURE, SIZE, SIZE);
    array[4] |= 5 << 23;
    write(&f, TIC + 32, bytemuck::cast_slice(&array));
    f.rasterizer.invalidate_region(DADDR + TIC + 32, 32);
    draw(&mut f, &gmmu, &regs);
    assert!(
        target(&mut f).iter().all(|&t| t == [0; 4]),
        "a 2D array image under a 2D descriptor reads the null texture"
    );

    handle_table(&mut f, 0);
    draw(&mut f, &gmmu, &regs);
    assert!(target(&mut f).iter().all(|&t| t == [10, 200, 30, 255]));
    f.assert_no_validation_errors();
}

/// Stores `handle` as word 4 of the table the storage buffer c[0][0x110] points to.
fn table_writing_program(handle: u32) -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let stg = sass(
        Opcode::Stg,
        (4 << 48) | (1 << 45) | (0x10 << 20) | (4 << 8) | 6,
    );
    program(
        header,
        &[
            mov_cbuf(4, 0x110),
            mov_cbuf(5, 0x114),
            mov_u32(6, handle),
            stg,
            exit(),
        ],
    )
}

#[test]
fn handle_programs_read_the_table_words_earlier_gpu_work_wrote() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &table_writing_program(1),
    );
    uniform_texture(&f, 0, TEXTURE, [10, 200, 30, 255]);
    uniform_texture(&f, 1, OTHER_TEXTURE, [50, 60, 70, 255]);
    let mut regs = sampling_regs();
    handle_table(&mut f, 0);
    draw(&mut f, &gmmu, &regs);
    let downloaded = f.rasterizer.buffer_cache.stats().downloaded_bytes;

    write(&f, PROGRAMS + u64::from(READER), &table_sampling_program());
    regs.pipelines[5].offset = READER;
    draw(&mut f, &gmmu, &regs);
    assert!(
        target(&mut f).iter().all(|&t| t == [50, 60, 70, 255]),
        "the draw samples the texture the first draw stored, not the guest's"
    );
    assert_eq!(
        f.rasterizer.buffer_cache.stats().downloaded_bytes,
        downloaded + 4,
        "only the handle word downloads"
    );
    let mut word = [0; 4];
    f.memory
        .read_bytes(DADDR + TABLE + 0x10, &mut word)
        .unwrap();
    assert_eq!(
        u32::from_le_bytes(word),
        1,
        "guest memory holds the stored handle"
    );

    draw(&mut f, &gmmu, &regs);
    assert_eq!(
        f.rasterizer.buffer_cache.stats().downloaded_bytes,
        downloaded + 4,
        "a downloaded word is not downloaded again"
    );
    assert_eq!(f.rasterizer.draw.shaders.stats().failed, 0);
    f.assert_no_validation_errors();
}

#[test]
fn handle_programs_skip_typeless_image_loads_of_integer_images() {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    let suld = sass(Opcode::Suld, (0xf << 20) | (6 << 39) | (3 << 33) | (4 << 8));
    let ldg = sass(
        Opcode::Ldg,
        (4 << 48) | (1 << 45) | (0x10 << 20) | (4 << 8) | 6,
    );
    let code = pixel_program(&[
        mov_cbuf(4, 0x110),
        mov_cbuf(5, 0x114),
        ldg,
        mov_u32(4, 3),
        mov_u32(5, 5),
        suld,
    ]);
    write(&f, PROGRAMS + u64::from(FRAGMENT), &code);
    uniform_texture(&f, 0, TEXTURE, [10, 200, 30, 255]);
    let mut integer = tic(GPU_VA + OTHER_TEXTURE, SIZE, SIZE);
    integer[0] =
        0x08 | 4 << 7 | 4 << 10 | 4 << 13 | 4 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28;
    write(&f, TIC + 32, bytemuck::cast_slice(&integer));
    let regs = sampling_regs();
    handle_table(&mut f, 0);
    draw(&mut f, &gmmu, &regs);
    assert!(
        target(&mut f).iter().all(|&t| t == [10, 200, 30, 255]),
        "the typeless load reads the float image the table names"
    );
    assert_eq!(
        f.rasterizer.draw.warned & Warning::NullStorageImage as u32,
        0
    );

    handle_table(&mut f, 1);
    draw(&mut f, &gmmu, &regs);
    assert_ne!(
        f.rasterizer.draw.warned & Warning::NullStorageImage as u32,
        0,
        "an integer image under the float declaration skips the draw"
    );
    assert_eq!(f.rasterizer.draw.shaders.stats().failed, 0);
    f.assert_no_validation_errors();
}

/// Per-invocation table indices: the storage buffer c[0][0x120] describes, one word per column.
const INDICES_TABLE: u64 = SSBO + 0x400;

/// Samples at (0.5, 0.5), into render target 0, the texture whose handle is entry `i` of the
/// 8-byte entry table the storage buffer c[0][0x110] points to, where `i` is the word of the
/// storage buffer c[0][0x120] the pixel's column selects: the table index varies per invocation.
fn indexed_table_sampling_program() -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    let tex = sass(
        Opcode::TexB,
        (7 << 51) | (0b1111 << 31) | (2 << 28) | (6 << 20) | (2 << 8),
    );
    let shl = |shift: u64| sass(Opcode::ShlImm, (1 << 39) | (shift << 20) | (1 << 8) | 1);
    let add_low = |r: u64| sass(Opcode::IaddReg, (1 << 47) | (1 << 20) | (r << 8) | r);
    let add_high = |r: u64| sass(Opcode::IaddReg, (1 << 43) | (RZ << 20) | (r << 8) | r);
    let ldg = |dest: u64, r: u64| sass(Opcode::Ldg, (4 << 48) | (1 << 45) | (r << 8) | dest);
    program(
        header,
        &[
            sass(Opcode::Ipa, (28 << 30) | (RZ << 8) | 1),
            sass(
                Opcode::F2iReg,
                (1 << 39) | (1 << 20) | (2 << 10) | (2 << 8) | 1,
            ),
            shl(2),
            mov_cbuf(8, 0x120),
            mov_cbuf(9, 0x124),
            add_low(8),
            add_high(9),
            ldg(1, 8),
            shl(3),
            mov_cbuf(4, 0x110),
            mov_cbuf(5, 0x114),
            add_low(4),
            add_high(5),
            ldg(6, 4),
            mov_f32(2, 0.5),
            mov_f32(3, 0.5),
            tex,
            exit(),
        ],
    )
}

#[test]
fn per_invocation_table_handles_bind_a_texture_per_table_entry() {
    draw_per_invocation_table_handles(true);
}

#[test]
fn table_handles_sample_with_explicit_gradients_without_quad_divergent_implicit_lod() {
    draw_per_invocation_table_handles(false);
}

/// Whether `spirv` takes a screen-space derivative (OpDPdx).
fn takes_derivatives(spirv: &[u32]) -> bool {
    let mut words = &spirv[5..];
    while let [first, ..] = words {
        if first & 0xffff == 207 {
            return true;
        }
        words = &words[((first >> 16) as usize).max(1)..];
    }
    false
}

/// Draws a fragment program that samples a texture handle table at a per-column index;
/// `quad_divergent_implicit_lod: false` translates as for a device without that feature.
fn draw_per_invocation_table_handles(quad_divergent_implicit_lod: bool) {
    let Some((mut f, gmmu)) = draw_fixture() else {
        return;
    };
    if !quad_divergent_implicit_lod {
        let mut profile = f.rasterizer.draw.shaders.profile().clone();
        profile.support_quad_divergent_implicit_lod = false;
        let host = f.rasterizer.device.shader_host.clone();
        f.rasterizer.draw.shaders = ShaderCache::new(profile, host);
    }
    if !f
        .rasterizer
        .device
        .features
        .sampled_image_array_non_uniform_indexing
    {
        eprintln!("skipping: no non-uniform sampled image indexing");
        return;
    }
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &indexed_table_sampling_program(),
    );
    let (even, odd) = ([10, 200, 30, 255], [50, 60, 70, 255]);
    uniform_texture(&f, 0, TEXTURE, even);
    uniform_texture(&f, 1, OTHER_TEXTURE, odd);
    for (cbuf, buffer) in [(0x110, TABLE), (0x120, INDICES_TABLE)] {
        let mut binding = (GPU_VA + buffer).to_le_bytes().to_vec();
        binding.extend_from_slice(&0x200u32.to_le_bytes());
        write(&f, FRAGMENT_CBUF + cbuf, &binding);
    }
    let entries: Vec<u32> = (0..64).flat_map(|i| [i & 1, 0]).collect();
    write(&f, TABLE, bytemuck::cast_slice(&entries));
    let indices: Vec<u32> = (0..SIZE).map(|x| x & 1).collect();
    write(&f, INDICES_TABLE, bytemuck::cast_slice(&indices));
    draw(&mut f, &gmmu, &sampling_regs());
    for (i, &texel) in target(&mut f).iter().enumerate() {
        let x = i as u32 % SIZE;
        let want = if x.is_multiple_of(2) { even } else { odd };
        assert_eq!(texel, want, "column {x} samples the entry its index names");
    }
    assert_eq!(f.rasterizer.draw.shaders.stats().failed, 0);
    let fragment = f.rasterizer.draw.stages[4]
        .as_ref()
        .expect("a fragment module");
    assert_eq!(
        takes_derivatives(&fragment.spirv),
        !f.rasterizer
            .draw
            .shaders
            .profile()
            .support_quad_divergent_implicit_lod,
        "explicit gradients only without quadDivergentImplicitLod"
    );
    f.assert_no_validation_errors();
}

fn cache_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("lichii-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// `draw_fixture` with the disk cache in `dir` and the fragment program `fragment`.
fn disk_fixture(dir: &std::path::Path, fragment: [u64; 2]) -> Option<(Fixture, Gmmu)> {
    let (mut f, gmmu) = draw_fixture()?;
    write(
        &f,
        PROGRAMS + u64::from(FRAGMENT),
        &fragment_program(fragment),
    );
    f.rasterizer
        .open_disk_cache(dir, &GraphicsSettings::default())
        .unwrap();
    Some((f, gmmu))
}

#[test]
fn the_disk_cache_serves_programs_and_pipelines_of_an_earlier_run() {
    let dir = cache_dir("disk-shaders");
    let Some((mut f, gmmu)) = disk_fixture(&dir, [0, 4]) else {
        return;
    };
    draw(&mut f, &gmmu, &draw_regs());
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!((stats.translated, stats.disk_hits), (2, 0));
    f.assert_no_validation_errors();
    drop(f);
    let file = std::fs::read(dir.join("shaders.bin")).unwrap();
    assert!(
        file.len() > lichii_shader::cache::HEADER_LEN,
        "both programs were written"
    );

    let (mut f, gmmu) = disk_fixture(&dir, [0, 4]).unwrap();
    f.rasterizer.draw.pipelines.wait_for_builds();
    assert_eq!(
        f.rasterizer.draw.pipelines.stats().built,
        1,
        "the pipeline is built from the stored modules before any draw"
    );
    draw(&mut f, &gmmu, &draw_regs());
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!((stats.translated, stats.disk_hits), (0, 2));
    assert_eq!(f.rasterizer.draw.pipelines.stats().built, 1);
    assert!(target(&mut f).iter().all(|&t| t == [255, 128, 0, 255]));
    f.assert_no_validation_errors();
    drop(f);

    let (mut f, gmmu) = disk_fixture(&dir, [4, 0]).unwrap();
    draw(&mut f, &gmmu, &draw_regs());
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!(
        (stats.translated, stats.disk_hits),
        (1, 1),
        "a changed fragment program translates again; the vertex program is reused"
    );
    assert!(target(&mut f).iter().all(|&t| t == [128, 255, 0, 255]));
    f.assert_no_validation_errors();
    drop(f);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn stale_or_damaged_shader_cache_files_are_replaced_or_cut() {
    let dir = cache_dir("stale-shaders");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("shaders.bin");
    std::fs::write(&path, b"not a shader cache").unwrap();
    let Some((mut f, gmmu)) = disk_fixture(&dir, [0, 4]) else {
        return;
    };
    assert_eq!(
        std::fs::read(&path).unwrap(),
        lichii_shader::cache::header(),
        "a stale file starts over"
    );
    draw(&mut f, &gmmu, &draw_regs());
    drop(f);
    let good = std::fs::read(&path).unwrap();
    let mut damaged = good.clone();
    damaged.extend_from_slice(&[0xff; 7]);
    std::fs::write(&path, &damaged).unwrap();

    let (mut f, gmmu) = disk_fixture(&dir, [0, 4]).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        good,
        "the damaged tail is cut"
    );
    draw(&mut f, &gmmu, &draw_regs());
    assert_eq!(f.rasterizer.draw.shaders.stats().disk_hits, 2);
    f.assert_no_validation_errors();
    drop(f);
    let _ = std::fs::remove_dir_all(&dir);
}

/// An Ampere `op` in `form` with the other bits from `fields`, guarded by PT.
fn ampere(op: AmpereOpcode, form: Form, fields: u128) -> u128 {
    let opcode = (0..4096u128)
        .find(|&bits| decode(Instruction(bits)) == Some(Decoded { opcode: op, form }))
        .expect("an encoding of the opcode");
    (fields & !0xffff) | 7 << 12 | opcode
}

/// MOV R`dest`, `value`.
fn ampere_mov(dest: u128, value: u32) -> u128 {
    ampere(
        AmpereOpcode::Mov,
        Form::ImmReg,
        dest << 16 | u128::from(value) << 32 | 0xf << 72,
    )
}

fn ampere_exit() -> u128 {
    ampere(AmpereOpcode::Exit, Form::Fixed, 7 << 87)
}

/// Stage buffer regions an ISBEWR writes.
#[derive(Clone, Copy)]
enum Isbe {
    Attribute,
    MapU32,
    MapU8,
}

/// ISBEWR R`value` to byte `address` of `region`.
fn isbewr(value: u128, address: u32, region: Isbe) -> u128 {
    let modifiers = match region {
        Isbe::Attribute => 3 << 76 | 1 << 75,
        Isbe::MapU32 => 1 << 75,
        Isbe::MapU8 => 0,
    };
    ampere(
        AmpereOpcode::Isbewr,
        Form::Fixed,
        RZ_A | value << 32 | u128::from(address) << 40 | modifiers | 1 << 79,
    )
}

const RZ_A: u128 = 0xff << 24;

/// An NVN2 graphics program: its header padded to 0x80 bytes, then `code`.
fn ampere_program(header: [u32; 20], code: &[u128]) -> Vec<u8> {
    let mut bytes: Vec<u8> = bytemuck::cast_slice(&header).to_vec();
    bytes.resize(GRAPHICS_HEADER_SIZE as usize, 0);
    for insn in code {
        bytes.extend_from_slice(&insn.to_le_bytes());
    }
    bytes
}

/// One triangle covering the target: three vertices with only a position (lines 0..4 of the
/// attribute region), one primitive in the map.
fn mesh_program() -> Vec<u8> {
    let mut header = header(1, ShaderType::VERTEX_B);
    header[13] |= 0xf0 << 8;
    let positions = [
        [-1.0f32, -1.0, 0.0, 1.0],
        [3.0, -1.0, 0.0, 1.0],
        [-1.0, 3.0, 0.0, 1.0],
    ];
    let mut code = Vec::new();
    for (vertex, position) in positions.iter().enumerate() {
        for (line, component) in position.iter().enumerate() {
            code.push(ampere_mov(0, component.to_bits()));
            let word = line as u32 * 32 + vertex as u32;
            code.push(isbewr(0, word * 4, Isbe::Attribute));
        }
        code.push(ampere_mov(1, vertex as u32));
        code.push(isbewr(1, 4 + vertex as u32, Isbe::MapU8));
    }
    code.push(ampere_mov(2, 1));
    code.push(isbewr(2, 3, Isbe::MapU32));
    code.push(ampere_exit());
    ampere_program(header, &code)
}

/// Launches one mesh workgroup: the count at byte 4 of its output region.
fn task_program() -> Vec<u8> {
    ampere_program(
        header(1, ShaderType::VERTEX_B),
        &[
            ampere_mov(0, 1),
            isbewr(0, 4, Isbe::Attribute),
            ampere_exit(),
        ],
    )
}

/// Writes render target 0 as (1, 0.5, 0, 1).
fn ampere_fragment_program() -> Vec<u8> {
    let mut header = header(2, ShaderType::PIXEL);
    header[18] = 0xf;
    ampere_program(
        header,
        &[
            ampere_mov(0, 1.0f32.to_bits()),
            ampere_mov(1, 0.5f32.to_bits()),
            ampere_mov(2, 0),
            ampere_mov(3, 1.0f32.to_bits()),
            ampere_exit(),
        ],
    )
}

const MESH: u64 = PROGRAMS;
const TASK: u64 = PROGRAMS + 0x1000;
const MESH_FRAGMENT: u64 = PROGRAMS + 0x2000;

fn bind_program(regs: &mut Regs, slot: usize, ty: ShaderType, offset: u64) {
    let address = GPU_VA + offset;
    let pipeline = &mut regs.pipelines[slot];
    pipeline.bits = PipelineBits(1 | ty.0 << 4);
    pipeline.reserved[0] = (address >> 32) as u32;
    pipeline.reserved[1] = address as u32;
}

/// `draw_regs`' target and viewport with mesh control on: one triangle of at most three
/// vertices, one thread per workgroup, and the mesh program in `mesh_slot`.
fn mesh_regs(mesh_slot: usize) -> Box<Regs> {
    let mut regs = draw_regs();
    regs.pipelines[1].bits = PipelineBits(0);
    regs.vertex_attrib_format[0] = VertexAttribute(0);
    bind_program(&mut regs, mesh_slot, ShaderType::VERTEX_B, MESH);
    bind_program(&mut regs, 5, ShaderType::PIXEL, MESH_FRAGMENT);
    let words = regs.words_mut();
    words[MESH_CONTROL] = 1;
    words[MESH_SHADER_A] = 4 | 3 << 4 | 1 << 13;
    words[MESH_SHADER_B] = 1 << 20;
    words[MESH_INIT_SHADER] = 1 | 1 << 12;
    regs
}

/// A Switch 2 fixture with the mesh, task and fragment programs, or `None` (skipping) without
/// a device or without mesh shaders.
fn mesh_fixture() -> Option<(Fixture, Gmmu)> {
    let mut f = fixture(GobLayout::Switch2)?;
    if !f.rasterizer.device.features.mesh_shader {
        eprintln!("skipping: no VK_EXT_mesh_shader");
        return None;
    }
    f.rasterizer.draw.pipelines.set_async_compilation(false);
    let gmmu = gmmu(&f);
    write(&f, MESH, &mesh_program());
    write(&f, TASK, &task_program());
    write(&f, MESH_FRAGMENT, &ampere_fragment_program());
    Some((f, gmmu))
}

fn mesh_draw(f: &mut Fixture, gmmu: &Gmmu, regs: &Regs, groups: u32) {
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
            count: groups,
            base_vertex: 0,
            base_instance: 0,
            instance_count: 1,
            inline_indices: None,
        },
    );
}

fn switch2_target(f: &mut Fixture) -> Vec<[u8; 4]> {
    let tiling = Tiling::new(SIZE, SIZE, 1, 4, 1, 0, GobLayout::Switch2);
    f.rasterizer.flush_region(DADDR, tiling.size_bytes() as u64);
    guest_linear(f, DADDR, &tiling).as_chunks::<4>().0.to_vec()
}

/// The stages `mesh_draw` bound in host slots 0, 2 and 4.
fn mesh_stages(f: &Fixture) -> [Option<ShaderStage>; 3] {
    let stage = |i: usize| f.rasterizer.draw.stages[i].as_ref().map(|m| m.stage);
    [stage(0), stage(2), stage(4)]
}

const TASK_STAGES: [Option<ShaderStage>; 3] = [
    Some(ShaderStage::Task),
    Some(ShaderStage::Mesh),
    Some(ShaderStage::Fragment),
];

#[test]
fn mesh_draws_build_task_and_mesh_pipelines() {
    let Some((mut f, gmmu)) = mesh_fixture() else {
        return;
    };
    mesh_draw(&mut f, &gmmu, &mesh_regs(1), 1);
    assert_eq!(
        mesh_stages(&f),
        [Some(ShaderStage::Mesh), None, Some(ShaderStage::Fragment)],
        "without a task program the VertexB slot holds the mesh program"
    );
    let mut regs = mesh_regs(3);
    bind_program(&mut regs, 1, ShaderType::VERTEX_B, TASK);
    mesh_draw(&mut f, &gmmu, &regs, 1);
    assert_eq!(
        mesh_stages(&f),
        TASK_STAGES,
        "with one the tessellation slot holds the mesh program"
    );
    let pipelines = f.rasterizer.draw.pipelines.stats();
    assert_eq!((pipelines.built, pipelines.failed), (2, 0));
    assert_eq!(f.rasterizer.draw.shaders.stats().failed, 0);
    f.assert_no_validation_errors();
}

#[test]
fn mesh_draws_rasterize_the_primitives_their_mesh_program_writes() {
    let Some((mut f, gmmu)) = mesh_fixture() else {
        return;
    };
    let regs = mesh_regs(1);
    mesh_draw(&mut f, &gmmu, &regs, 0);
    assert!(
        switch2_target(&mut f).iter().all(|&t| t == [0; 4]),
        "no workgroups draw nothing"
    );
    mesh_draw(&mut f, &gmmu, &regs, 1);
    let texels = switch2_target(&mut f);
    assert!(
        texels.iter().all(|&t| t == [255, 128, 0, 255]),
        "the triangle covers the target: {:?}",
        &texels[..4]
    );
    f.assert_no_validation_errors();
}

#[test]
fn task_programs_launch_the_mesh_workgroups_they_count() {
    let Some((mut f, gmmu)) = mesh_fixture() else {
        return;
    };
    let mut regs = mesh_regs(3);
    bind_program(&mut regs, 1, ShaderType::VERTEX_B, TASK);
    mesh_draw(&mut f, &gmmu, &regs, 1);
    let texels = switch2_target(&mut f);
    assert!(
        texels.iter().all(|&t| t == [255, 128, 0, 255]),
        "the task's mesh workgroup covers the target: {:?}",
        &texels[..4]
    );
    assert_eq!(mesh_stages(&f), TASK_STAGES);
    f.assert_no_validation_errors();
}

const GEOMETRY: u64 = PROGRAMS + 0x3000;
const PRIMITIVE_FRAGMENT: u64 = PROGRAMS + 0x4000;

#[test]
fn per_primitive_fragment_inputs_skip_the_draw() {
    let Some((mut f, gmmu)) = mesh_fixture() else {
        return;
    };
    let mut geometry = header(1, ShaderType::GEOMETRY);
    geometry[13] |= 1 << 16;
    write(&f, GEOMETRY, &ampere_program(geometry, &[]));
    let mut fragment = header(2, ShaderType::PIXEL);
    fragment[6] = 0b10;
    fragment[18] = 0xf;
    let ipa = ampere(
        AmpereOpcode::Ipa,
        Form::Fixed,
        0xff << 32 | 0x20 << 64 | 7 << 81,
    );
    let code = [
        ipa,
        ampere_mov(1, 0),
        ampere_mov(2, 0),
        ampere_mov(3, 0),
        ampere_exit(),
    ];
    write(&f, PRIMITIVE_FRAGMENT, &ampere_program(fragment, &code));
    let mut regs = mesh_regs(1);
    bind_program(&mut regs, 4, ShaderType::GEOMETRY, GEOMETRY);
    bind_program(&mut regs, 5, ShaderType::PIXEL, PRIMITIVE_FRAGMENT);
    mesh_draw(&mut f, &gmmu, &regs, 1);
    let stats = f.rasterizer.draw.shaders.stats();
    assert_eq!(
        (stats.translated, stats.failed),
        (2, 1),
        "the fragment program reads generic 0, which the geometry header makes per-primitive"
    );
    assert_eq!(f.rasterizer.draw.pipelines.stats().built, 0);
    f.assert_no_validation_errors();
}

#[test]
fn indirect_mesh_draws_are_skipped() {
    let Some((mut f, gmmu)) = mesh_fixture() else {
        return;
    };
    let regs = mesh_regs(1);
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
            buffer: GPU_VA + VERTICES,
            stride: 16,
            max_draws: 1,
            count_buffer: None,
        },
    );
    assert_ne!(f.rasterizer.draw.warned & Warning::IndirectMesh as u32, 0);
    assert_eq!(f.rasterizer.draw.shaders.stats().translated, 0);
    f.assert_no_validation_errors();
}
