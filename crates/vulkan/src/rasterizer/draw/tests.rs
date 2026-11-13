//! Draws of hand-assembled Maxwell programs on synthetic 3D engine state.

use lichii_gpu::backend::{ChannelId, Rasterizer};
use lichii_gpu::backend::{ConstBufferBinding, DrawParams, GraphicsContext};
use lichii_gpu::engines::maxwell_3d::regs::{
    DepthMode, MAX_CONST_BUFFERS, MAX_SHADER_STAGE, PipelineBits, PrimitiveTopology,
    RenderTargetConfigBits, RtControl, ShaderType, SurfaceClipBits, SurfaceClipBits1, TileMode,
    VertexAttribute, VertexStreamBits,
};
use lichii_gpu::engines::maxwell_3d::{DirtyState, Regs};
use lichii_gpu::gmmu::Gmmu;
use lichii_shader::DescriptorKind;
use lichii_shader::maxwell::opcode::Opcode;
use lichii_texture::block_linear::{GobLayout, Tiling};

use super::super::tests::{DADDR, Fixture, GPU_VA, fixture, gmmu, guest_linear};

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

fn draw(f: &mut Fixture, gmmu: &Gmmu, regs: &Regs) {
    let mut dirty = DirtyState::new();
    let mut const_buffers = [[ConstBufferBinding::default(); MAX_CONST_BUFFERS]; MAX_SHADER_STAGE];
    const_buffers[0][0] = ConstBufferBinding {
        address: GPU_VA + VERTEX_CBUF,
        size: 0x100,
        enabled: true,
    };
    const_buffers[4][0] = ConstBufferBinding {
        address: GPU_VA + FRAGMENT_CBUF,
        size: 0x100,
        enabled: true,
    };
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
