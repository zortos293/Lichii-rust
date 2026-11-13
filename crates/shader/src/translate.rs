//! The translation driver: a guest program becomes optimized IR plus the [`ShaderInfo`] the
//! backend and the pipeline need, following the C++ `TranslateProgram` pass order.

use lichii_common::settings::ResolutionScale;

use std::hash::{Hash, Hasher};

use crate::ampere::{AmpereFlow, GRAPHICS_HEADER_SIZE};
use crate::flow::Cfg;
use crate::flow::structure::{StructureOptions, structure};
use crate::hash::Recorder;
use crate::info::{Interpolation, StorageBufferDescriptor, StorageWriteExtent};
use crate::ir::{Attribute, Opcode, Program, SyntaxNode};
use crate::maxwell::flow::MaxwellFlow;
use crate::maxwell::header::{OutputTopology, PixelImap, ProgramHeader, SystemValuesC};
use crate::maxwell::passthrough::{lower_geometry_passthrough, output_topology_vertices};
use crate::opt;
use crate::profile::HostTranslateInfo;
use crate::{Profile, ShaderArch, ShaderEnvironment, ShaderInfo, ShaderStage, TranslateError};

/// Stable hash of the translator's sources (`build.rs`): modules translated by another build
/// never match, with no version to bump by hand.
pub const TRANSLATOR_ID: u64 = parse_u64(env!("LICHII_SHADER_TRANSLATOR_ID"));

const fn parse_u64(digits: &str) -> u64 {
    let bytes = digits.as_bytes();
    let mut value = 0u64;
    let mut i = 0;
    while i < bytes.len() {
        value = value * 10 + (bytes[i] - b'0') as u64;
        i += 1;
    }
    value
}

/// Bytes between a graphics program's program header and its first instruction.
pub(crate) fn header_size(env: &dyn ShaderEnvironment) -> u64 {
    match env.arch() {
        ShaderArch::Maxwell => 0x50,
        ShaderArch::Ampere => GRAPHICS_HEADER_SIZE,
    }
}

/// A translated program: optimized SSA IR and what the backend needs to emit it.
#[derive(Clone, Debug)]
pub struct TranslatedProgram {
    /// Stable hash of everything translation read from the environment, with the options.
    pub hash: u64,
    pub ir: Program,
    pub info: ShaderInfo,
    pub stage: ShaderStage,
    pub workgroup_size: [u32; 3],
    pub output_topology: OutputTopology,
    pub output_vertices: u32,
    pub invocations: u32,
    pub local_memory_size: u32,
    pub shared_memory_size: u32,
    pub is_geometry_passthrough: bool,
}

/// Host and run-time inputs that change how a program translates.
#[derive(Clone, Copy, Debug)]
pub struct TranslateOptions<'a> {
    pub profile: &'a Profile,
    pub host: &'a HostTranslateInfo,
    pub resolution_scale: ResolutionScale,
}

/// Translates the program `env` describes.
pub fn translate(
    env: &dyn ShaderEnvironment,
    options: TranslateOptions<'_>,
) -> Result<TranslatedProgram, TranslateError> {
    translate_into(env, options, Program::new())
}

/// Translates a VertexA and VertexB pair into one VertexB program in which A runs first and
/// falls through into B. B's descriptors and storage buffers join A's, which keep their indices.
pub fn translate_dual_vertex(
    env_a: &dyn ShaderEnvironment,
    env_b: &dyn ShaderEnvironment,
    options: TranslateOptions<'_>,
) -> Result<TranslatedProgram, TranslateError> {
    let TranslatedProgram {
        hash: a_hash,
        mut ir,
        mut info,
        local_memory_size,
        ..
    } = translate(env_a, options)?;
    opt::vertex_a_transform(&mut ir);
    let a_syntax = std::mem::take(&mut ir.syntax_list);
    let a_blocks = std::mem::take(&mut ir.block_order);
    let a_post_order = std::mem::take(&mut ir.post_order);
    let mut b = translate_into(env_b, options, ir)?;
    opt::vertex_b_transform(&mut b.ir);
    for (a, b) in info.loads.0.iter_mut().zip(b.info.loads.0) {
        *a |= b;
    }
    for (a, b) in info.stores.0.iter_mut().zip(b.info.stores.0) {
        *a |= b;
    }
    let storage_bindings = opt::join_storage_info(
        &mut info,
        &b.info,
        opt::storage_buffer_limit(options.profile),
    )?;
    let descriptors = opt::join_texture_info(&mut info, &b.info, &storage_bindings);
    opt::remap_texture_descriptors(&mut b.ir, &descriptors);
    opt::remap_storage_bindings(&mut b.ir, &storage_bindings);

    let ir = &mut b.ir;
    let mut syntax: Vec<SyntaxNode> = a_syntax
        .into_iter()
        .filter(|n| *n != SyntaxNode::Return)
        .collect();
    syntax.append(&mut ir.syntax_list);
    ir.syntax_list = syntax;
    let mut blocks = a_blocks;
    blocks.append(&mut ir.block_order);
    ir.block_order = blocks;
    ir.post_order.extend(a_post_order);
    opt::dead_code_elimination(ir);
    if cfg!(debug_assertions) {
        opt::verify(ir).map_err(TranslateError::InvalidIr)?;
    }
    opt::collect_shader_info(env_b, ir, &mut info);
    b.info = info;
    b.local_memory_size = b.local_memory_size.max(local_memory_size);
    let mut hasher = crate::hash::StableHasher::new();
    (a_hash, b.hash).hash(&mut hasher);
    b.hash = hasher.finish();
    Ok(b)
}

/// Translates into `ir`, whose syntax list and block lists must be empty; instructions and
/// blocks already in its arena stay untouched.
fn translate_into(
    env: &dyn ShaderEnvironment,
    options: TranslateOptions<'_>,
    ir: Program,
) -> Result<TranslatedProgram, TranslateError> {
    let recorder = Recorder::new(env);
    let mut program = translate_recorded(&recorder, options, ir)?;
    let mut hasher = recorder.hasher();
    TRANSLATOR_ID.hash(&mut hasher);
    format!(
        "{:?}{:?}{:?}",
        options.profile, options.host, options.resolution_scale
    )
    .hash(&mut hasher);
    program.hash = hasher.finish();
    Ok(program)
}

fn translate_recorded(
    env: &dyn ShaderEnvironment,
    options: TranslateOptions<'_>,
    mut ir: Program,
) -> Result<TranslatedProgram, TranslateError> {
    let host = options.host.clone().with_descriptor_limits();
    let stage = env.stage();
    build(env, &host, &mut ir)?;
    let mut program = TranslatedProgram {
        hash: 0,
        ir,
        info: ShaderInfo::default(),
        stage,
        workgroup_size: [1; 3],
        output_topology: OutputTopology::PointList,
        output_vertices: 0,
        invocations: 0,
        local_memory_size: env.local_memory_size(),
        shared_memory_size: 0,
        is_geometry_passthrough: false,
    };
    match stage {
        ShaderStage::TessellationControl => {
            program.invocations = header(env)?.threads_per_input_primitive();
        }
        ShaderStage::Geometry => {
            let sph = header(env)?;
            program.output_topology =
                sph.output_topology().ok_or(TranslateError::InvalidHeader {
                    address: env.start_address() - header_size(env),
                })?;
            program.output_vertices = sph.max_output_vertices();
            program.invocations = sph.threads_per_input_primitive();
            program.is_geometry_passthrough = sph.geometry_passthrough();
            if program.is_geometry_passthrough {
                let mask = env.gp_passthrough_mask();
                for i in 0..mask.len() * 32 {
                    let passthrough = (mask[i / 32] >> (i % 32)) & 1 == 0;
                    program
                        .info
                        .passthrough
                        .set(Attribute(i as u16), passthrough);
                }
                if !host.support_geometry_shader_passthrough {
                    let copied = Attribute::POSITION_X.0
                        ..Attribute::GENERIC_0_X.0 + Attribute::NUM_GENERICS * 4;
                    for i in (0..mask.len() as u16 * 32).filter(|i| !copied.contains(i)) {
                        program.info.passthrough.set(Attribute(i), false);
                    }
                    program.output_vertices = output_topology_vertices(program.output_topology);
                    lower_geometry_passthrough(
                        &mut program.ir,
                        &program.info,
                        program.output_vertices,
                        &host,
                    );
                }
            }
        }
        ShaderStage::Compute => {
            program.workgroup_size = env.workgroup_size();
            program.shared_memory_size = env.shared_memory_size();
        }
        ShaderStage::Task | ShaderStage::Mesh => {
            let state = env
                .mesh_state()
                .ok_or(TranslateError::MissingMeshState { stage })?;
            let init = state.init_shader;
            let lines = ((init >> 12) & 0x1ff).max((init >> 22) & 0x1ff);
            let mesh = &mut program.info.mesh;
            mesh.payload_words = lines.max(1) * 32;
            mesh.topology = state.shader_a & 0xf;
            mesh.max_vertices = (state.shader_a >> 4) & 0x1ff;
            mesh.max_primitives = (state.shader_a >> 13) & 0x3ff;
            let threads = if stage == ShaderStage::Task {
                init & 0x3f
            } else {
                (state.shader_b >> 20) & 0x3f
            };
            program.workgroup_size = [threads.max(1), 1, 1];
            if stage == ShaderStage::Mesh {
                if !matches!(mesh.topology, 0 | 1 | 4) {
                    return Err(TranslateError::UnsupportedMesh("output topology"));
                }
                mesh.vertex_lines = mesh_lines(&header(env)?)?;
                mesh.primitive_lines = mesh_lines(&state.primitive_header)?;
            }
        }
        _ => {}
    }
    remove_unreachable_blocks(&mut program.ir);
    optimize(
        env,
        options.profile,
        &host,
        options.resolution_scale,
        &mut program,
    )?;
    Ok(program)
}

/// The attributes a mesh program's stage buffer lines hold, in line order: the system values
/// from PrimitiveId on, then the generic components, as `header`'s output map lists them.
fn mesh_lines(header: &ProgramHeader) -> Result<Vec<u32>, TranslateError> {
    let vtg = header.vtg();
    if vtg.omap_system_c().intersects(SystemValuesC::all()) {
        return Err(TranslateError::UnsupportedMesh(
            "outputs past the generic attributes",
        ));
    }
    let system = vtg.omap_system_b().bits();
    let system_lines = (0..8)
        .filter(|bit| system >> bit & 1 != 0)
        .map(|bit| u32::from(Attribute::PRIMITIVE_ID.0) + bit);
    let generic_lines = (0..Attribute::NUM_GENERICS as usize).flat_map(|generic| {
        let mask = vtg.output_generic(generic);
        (0..4u16)
            .filter(move |&element| mask[element as usize])
            .map(move |element| u32::from(Attribute::generic(generic as u16, element).0))
    });
    Ok(system_lines.chain(generic_lines).collect())
}

fn header(env: &dyn ShaderEnvironment) -> Result<ProgramHeader, TranslateError> {
    let start = env.start_address();
    if start < header_size(env) {
        return Err(TranslateError::InvalidHeader { address: start });
    }
    Ok(ProgramHeader::read(env, start - header_size(env)))
}

/// Decodes, structures and translates the program into `ir`, then orders its blocks.
fn build(
    env: &dyn ShaderEnvironment,
    host: &HostTranslateInfo,
    ir: &mut Program,
) -> Result<(), TranslateError> {
    let stage = env.stage();
    let arch = env.arch();
    let base = env.start_address();
    let cfg = match arch {
        ShaderArch::Maxwell => {
            Cfg::build(&MaxwellFlow { base }, env, 0, stage == ShaderStage::VertexA)?
        }
        ShaderArch::Ampere => Cfg::build(&AmpereFlow { base }, env, 0, false)?,
    };
    let options = StructureOptions {
        converge_exits: matches!(stage, ShaderStage::Task | ShaderStage::Mesh),
        reorder_demotes: host.needs_demote_reorder,
    };
    structure(
        ir,
        &cfg,
        options,
        &mut |prog, block, begin, end| match arch {
            ShaderArch::Maxwell => {
                crate::maxwell::translate::translate_block(prog, block, env, begin, end)
            }
            ShaderArch::Ampere => {
                crate::ampere::translate::translate_block(prog, block, env, begin, end)
            }
        },
    )?;
    opt::compute_post_order(ir);
    Ok(())
}

/// Drops blocks the entry cannot reach (code after an unconditional call that never returns).
fn remove_unreachable_blocks(ir: &mut Program) {
    if ir.block_order.len() == ir.post_order.len() {
        return;
    }
    let entry = ir.block_order.first().copied();
    let mut reachable = vec![false; ir.num_blocks()];
    for &block in &ir.post_order {
        reachable[block.0 as usize] = true;
    }
    ir.block_order
        .retain(|&b| Some(b) == entry || reachable[b.0 as usize]);
}

fn optimize(
    env: &dyn ShaderEnvironment,
    profile: &Profile,
    host: &HostTranslateInfo,
    scale: ResolutionScale,
    program: &mut TranslatedProgram,
) -> Result<(), TranslateError> {
    let TranslatedProgram {
        ir, info, stage, ..
    } = program;
    let stage = *stage;
    if !host.support_float64 {
        opt::lower_fp64_to_fp32(ir);
    }
    if !host.support_float16 {
        opt::lower_fp16_to_fp32(ir);
    }
    if !host.support_int64 {
        opt::lower_int64_to_int32(ir).map_err(TranslateError::Lowering)?;
    }
    if !host.support_conditional_barrier {
        opt::conditional_barrier(ir, info);
    }
    opt::ssa_rewrite(ir);
    opt::constant_propagation(env, ir);
    opt::position(env, ir, info);
    opt::blit_constants(env, ir, info);
    opt::global_memory_to_storage_buffer(env, ir, info, profile)?;
    opt::geometry_compaction(ir, stage, program.invocations, host);
    opt::texture_pass(env, ir, info, host)?;
    if scale != ResolutionScale::X1 {
        opt::rescaling(ir, stage, scale);
    }
    opt::dead_code_elimination(ir);
    opt::identity_removal(ir);
    tighten_output_vertices(program);
    let TranslatedProgram { ir, info, .. } = program;
    if cfg!(debug_assertions) {
        opt::verify(ir).map_err(TranslateError::InvalidIr)?;
    }
    opt::collect_shader_info(env, ir, info);
    if stage == ShaderStage::Mesh {
        for &line in info
            .mesh
            .vertex_lines
            .iter()
            .chain(&info.mesh.primitive_lines)
        {
            info.stores.set(Attribute(line as u16), true);
        }
    }
    opt::layer(ir, stage, info, host);
    opt::vendor_workaround(ir);
    collect_interpolation_info(env, info)?;
    add_nvn_storage_buffers(info, profile)?;
    opt::limit_sampled_texture_arrays(ir, info, host)?;
    Ok(())
}

/// Lowers a geometry program's declared vertex count to the number of vertices it emits when
/// it emits a fixed number (no loops).
fn tighten_output_vertices(program: &mut TranslatedProgram) {
    if program.stage != ShaderStage::Geometry || program.is_geometry_passthrough {
        return;
    }
    let ir = &program.ir;
    if ir
        .syntax_list
        .iter()
        .any(|n| matches!(n, SyntaxNode::Loop { .. }))
    {
        return;
    }
    let emits = ir
        .block_order
        .iter()
        .flat_map(|&b| ir.insts_of(b))
        .filter(|&id| ir.op(id) == Opcode::EmitVertex)
        .count() as u32;
    if emits != 0 {
        program.output_vertices = program.output_vertices.min(emits);
    }
}

/// Reads each fragment input's interpolation mode from the program header.
fn collect_interpolation_info(
    env: &dyn ShaderEnvironment,
    info: &mut ShaderInfo,
) -> Result<(), TranslateError> {
    if env.stage() != ShaderStage::Fragment {
        return Ok(());
    }
    let sph = header(env)?;
    for (index, slot) in info.interpolation.iter_mut().enumerate() {
        let mut imap = None;
        for value in sph.ps().generic_input_map(index) {
            if value == PixelImap::Unused {
                continue;
            }
            if imap.is_some_and(|m| m != value) {
                return Err(TranslateError::Lowering(format!(
                    "per-component interpolation of generic {index}"
                )));
            }
            imap = Some(value);
        }
        *slot = match imap {
            None => continue,
            Some(PixelImap::Unused | PixelImap::Perspective) => Interpolation::Smooth,
            Some(PixelImap::Constant) => Interpolation::Flat,
            Some(PixelImap::ScreenLinear) => Interpolation::NoPerspective,
        };
    }
    Ok(())
}

/// Binds the NVN driver's per-stage storage buffer slots (constant buffer 0) that global memory
/// accesses left in place may reach.
fn add_nvn_storage_buffers(info: &mut ShaderInfo, profile: &Profile) -> Result<(), TranslateError> {
    if !info.uses_global_memory {
        return Ok(());
    }
    const DESCRIPTOR_SIZE: u32 = 0x10;
    let limit = opt::storage_buffer_limit(profile);
    let stores = info.stores_global_memory;
    for index in 0..16u32 {
        if info.nvn_buffer_used & (1 << index) == 0 {
            continue;
        }
        let offset = info.nvn_buffer_base + index * DESCRIPTOR_SIZE;
        let descs = &mut info.storage_buffers_descriptors;
        if let Some(desc) = descs
            .iter_mut()
            .find(|d| d.cbuf_index == 0 && d.cbuf_offset == offset)
        {
            desc.is_written |= stores;
            if stores {
                desc.write_extent = StorageWriteExtent::default();
            }
            continue;
        }
        if descs.len() >= limit {
            return Err(TranslateError::TooManyStorageBuffers {
                count: descs.len() + 1,
                limit,
            });
        }
        descs.push(StorageBufferDescriptor {
            cbuf_index: 0,
            cbuf_offset: offset,
            count: 1,
            is_written: stores,
            indirect_offset: StorageBufferDescriptor::DIRECT,
            write_extent: StorageWriteExtent::default(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lichii_common::GpuVAddr;

    use crate::maxwell::opcode::Opcode as Sass;
    use crate::maxwell::translate::tests::{TestEnv, encode};

    const PT: u64 = 7 << 16;

    fn options<'a>(profile: &'a Profile, host: &'a HostTranslateInfo) -> TranslateOptions<'a> {
        TranslateOptions {
            profile,
            host,
            resolution_scale: ResolutionScale::X1,
        }
    }

    fn run(env: &TestEnv) -> Result<TranslatedProgram, TranslateError> {
        let host = HostTranslateInfo {
            support_float64: true,
            support_float16: true,
            support_int64: true,
            support_conditional_barrier: true,
            ..HostTranslateInfo::default()
        };
        translate(env, options(&Profile::default(), &host))
    }

    fn shared_store(offset: u64, value: u64) -> [u64; 2] {
        [
            encode(Sass::Mov32i, PT | (value << 20) | (0xf << 12) | 1),
            encode(Sass::Sts, PT | (4 << 48) | (offset << 20) | (0xff << 8) | 1),
        ]
    }

    #[test]
    fn dual_vertex_programs_merge_with_a_first() {
        let program = |stage, offset, value| {
            let [mov, sts] = shared_store(offset, value);
            let mut env = TestEnv::new(&[mov, sts, encode(Sass::Exit, PT | 0xf)]);
            env.stage = stage;
            env
        };
        let a = program(ShaderStage::VertexA, 0x10, 5);
        let b = program(ShaderStage::VertexB, 0x20, 7);
        let host = HostTranslateInfo::default();
        let merged =
            translate_dual_vertex(&a, &b, options(&Profile::default(), &host)).expect("merge");
        let dump = merged.ir.dump();
        let first = dump.find("WriteSharedU32 #0x10, #0x5").expect(&dump);
        let second = dump.find("WriteSharedU32 #0x20, #0x7").expect(&dump);
        assert!(first < second, "{dump}");
        assert_eq!(dump.matches("Prologue").count(), 1, "{dump}");
        assert_eq!(dump.matches("Epilogue").count(), 1, "{dump}");
        assert_eq!(dump.matches("return").count(), 1, "{dump}");
        assert_eq!(merged.stage, ShaderStage::VertexB);
        opt::verify(&merged.ir).expect("valid IR");
    }

    #[test]
    fn hash_follows_code_and_options_but_not_the_address() {
        let host = HostTranslateInfo::default();
        let profile = Profile::default();
        let hash = |code: &[u64], base: GpuVAddr, profile: &Profile| {
            let mut env = TestEnv::new(code);
            env.base = base;
            translate(&env, options(profile, &host))
                .expect("translate")
                .hash
        };
        let [mov, sts] = shared_store(0x10, 5);
        let exit = encode(Sass::Exit, PT | 0xf);
        let a = hash(&[mov, sts, exit], 0x1000, &profile);
        assert_ne!(a, 0);
        assert_eq!(a, hash(&[mov, sts, exit], 0x9000, &profile));
        let [mov7, sts7] = shared_store(0x10, 7);
        assert_ne!(a, hash(&[mov7, sts7, exit], 0x1000, &profile));
        let other = Profile {
            support_int64: !profile.support_int64,
            ..profile.clone()
        };
        assert_ne!(a, hash(&[mov, sts, exit], 0x1000, &other));
    }

    #[test]
    fn compute_program_reaches_ssa_with_constants_folded() {
        let env = TestEnv::new(&[
            encode(Sass::Mov32i, PT | (5 << 20) | (0xf << 12) | 1),
            encode(Sass::Sts, PT | (4 << 48) | (0x10 << 20) | (0xff << 8) | 1),
            encode(Sass::Exit, PT | 0xf),
        ]);
        let program = run(&env).expect("translate");
        let dump = program.ir.dump();
        assert!(dump.contains("WriteSharedU32 #0x10, #0x5"), "{dump}");
        assert!(!dump.contains("Register"), "{dump}");
        assert_eq!(program.workgroup_size, [1, 1, 1]);
        opt::verify(&program.ir).expect("valid IR");
    }

    /// Random instruction streams through the whole pipeline: every program either translates
    /// to valid IR or fails with an error, and none panics. The host lacks fp16, fp64 and int64
    /// so the lowering passes run too.
    #[test]
    fn random_programs_translate_or_fail_cleanly() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let host = HostTranslateInfo::default();
        let profile = Profile::default();
        let mut translated = 0;
        for _ in 0..3000 {
            let len = 1 + (next() % 12) as usize;
            let mut code: Vec<u64> = (0..len)
                .map(|_| {
                    let op = Sass::ALL[(next() % Sass::ALL.len() as u64) as usize];
                    let word = encode(op, next());
                    if next() % 4 == 0 {
                        word
                    } else {
                        (word & !(0xf << 16)) | PT
                    }
                })
                .collect();
            code.push(encode(Sass::Exit, PT | 0xf));
            let env = TestEnv::new(&code);
            match translate(&env, options(&profile, &host)) {
                Ok(program) => {
                    opt::verify(&program.ir).unwrap_or_else(|e| panic!("{code:x?}: {e}"));
                    translated += 1;
                }
                Err(TranslateError::InvalidIr(e)) => panic!("{code:x?}: {e}"),
                Err(_) => {}
            }
        }
        assert!(translated > 50, "only {translated} programs translated");
    }
}
