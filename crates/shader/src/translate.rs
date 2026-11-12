//! The translation driver: a guest program becomes optimized IR plus the [`ShaderInfo`] the
//! backend and the pipeline need, following the C++ `TranslateProgram` pass order.

use lichii_common::settings::ResolutionScale;

use crate::flow::Cfg;
use crate::flow::structure::{StructureOptions, structure};
use crate::info::{Interpolation, StorageBufferDescriptor, StorageWriteExtent};
use crate::ir::{Attribute, Opcode, Program, SyntaxNode};
use crate::maxwell::flow::MaxwellFlow;
use crate::maxwell::header::{OutputTopology, PixelImap, ProgramHeader};
use crate::maxwell::translate::translate_block;
use crate::opt;
use crate::profile::HostTranslateInfo;
use crate::{Profile, ShaderArch, ShaderEnvironment, ShaderInfo, ShaderStage, TranslateError};

/// Size of the Maxwell program header in front of a graphics program's first instruction.
const HEADER_SIZE: u64 = 0x50;

/// A translated program: optimized SSA IR and what the backend needs to emit it.
#[derive(Clone, Debug)]
pub struct TranslatedProgram {
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
    let host = options.host.clone().with_descriptor_limits();
    let stage = env.stage();
    let mut ir = Program::new();
    build(env, &host, &mut ir)?;
    let mut program = TranslatedProgram {
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
                    address: env.start_address() - HEADER_SIZE,
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
            }
        }
        ShaderStage::Compute => {
            program.workgroup_size = env.workgroup_size();
            program.shared_memory_size = env.shared_memory_size();
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

fn header(env: &dyn ShaderEnvironment) -> Result<ProgramHeader, TranslateError> {
    let start = env.start_address();
    if start < HEADER_SIZE {
        return Err(TranslateError::InvalidHeader { address: start });
    }
    Ok(ProgramHeader::read(env, start - HEADER_SIZE))
}

/// Decodes, structures and translates the program into `ir`, then orders its blocks.
fn build(
    env: &dyn ShaderEnvironment,
    host: &HostTranslateInfo,
    ir: &mut Program,
) -> Result<(), TranslateError> {
    let stage = env.stage();
    let cfg = match env.arch() {
        ShaderArch::Maxwell => {
            let isa = MaxwellFlow {
                base: env.start_address(),
            };
            Cfg::build(&isa, env, 0, stage == ShaderStage::VertexA)?
        }
        arch => return Err(TranslateError::UnsupportedProgram { arch, stage }),
    };
    let options = StructureOptions {
        converge_exits: matches!(stage, ShaderStage::Task | ShaderStage::Mesh),
        reorder_demotes: host.needs_demote_reorder,
    };
    structure(ir, &cfg, options, &mut |prog, block, begin, end| {
        translate_block(prog, block, env, begin, end)
    })?;
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
        opt::verify(ir).map_err(TranslateError::Lowering)?;
    }
    opt::collect_shader_info(env, ir, info);
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
}
