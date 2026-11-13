//! The emission driver, ported from the C++ `emit_spirv.cpp`: the entry point function with
//! structured control flow, execution modes, capabilities and phi patching.

use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::spirv::{
    Capability, ExecutionMode, ExecutionModel, FunctionControl, LoopControl, Op, SelectionControl,
    StorageClass, Word,
};
use smallvec::SmallVec;

use super::context::{DeferredPhi, EmitContext};
use super::emit::emit_inst;
use super::{EmitError, EmitResult};
use crate::backend::Bindings;
use crate::ir::{Attribute, InstId, SyntaxNode, Type};
use crate::maxwell::header::OutputTopology;
use crate::runtime::{InputTopology, TessPrimitive, TessSpacing};
use crate::translate::TranslatedProgram;
use crate::{Profile, RuntimeInfo, ShaderStage};

/// Iterations a guest loop may run before the loop safety check breaks out of it.
const LOOP_SAFETY_ITERATIONS: u32 = 0x2000;

pub(super) fn emit(
    profile: &Profile,
    runtime: &RuntimeInfo,
    program: &TranslatedProgram,
    bindings: &mut Bindings,
) -> EmitResult<Vec<u32>> {
    if matches!(
        program.stage,
        ShaderStage::VertexA | ShaderStage::Task | ShaderStage::Mesh
    ) {
        return Err(EmitError::UnsupportedStage(program.stage));
    }
    let mut ctx = EmitContext::new(profile, runtime, program, bindings)?;
    let main = define_main(&mut ctx)?;
    ctx.define_deferred_functions()?;
    define_entry_point(&mut ctx, main);
    if profile.support_float_controls {
        ctx.add_extension("SPV_KHR_float_controls");
        setup_denorm_control(&mut ctx, main);
        setup_signed_nan_capabilities(&mut ctx, main);
    }
    setup_capabilities(&mut ctx);
    if !runtime.xfb_varyings.is_empty() {
        ctx.add_capability(Capability::TransformFeedback);
        ctx.b.execution_mode(main, ExecutionMode::Xfb, []);
    }
    patch_phis(&mut ctx)?;
    Ok(ctx.b.module().assemble())
}

fn define_main(ctx: &mut EmitContext<'_>) -> EmitResult<Word> {
    let void_function = ctx.b.type_function(ctx.void_id, []);
    let main = ctx
        .b
        .begin_function(ctx.void_id, None, FunctionControl::NONE, void_function)?;
    ctx.main_function = ctx.b.selected_function().unwrap_or_default();
    ctx.assign_labels();
    traverse(ctx)?;
    ctx.b.end_function()?;
    Ok(main)
}

fn traverse(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let program = ctx.program;
    let mut in_block = false;
    for node in &program.syntax_list {
        match *node {
            SyntaxNode::Block(block) => {
                let label = ctx.label(block);
                if in_block {
                    ctx.b.branch(label)?;
                }
                ctx.b.begin_block(Some(label))?;
                for inst in program.insts_of(block) {
                    emit_inst(ctx, inst)?;
                }
                in_block = true;
                continue;
            }
            SyntaxNode::If { cond, body, merge } => {
                let cond = ctx.def(cond)?;
                ctx.b
                    .selection_merge(ctx.label(merge), SelectionControl::NONE)?;
                ctx.b
                    .branch_conditional(cond, ctx.label(body), ctx.label(merge), [])?;
            }
            SyntaxNode::Loop {
                body,
                continue_block,
                merge,
            } => {
                ctx.b.loop_merge(
                    ctx.label(merge),
                    ctx.label(continue_block),
                    LoopControl::NONE,
                    [],
                )?;
                ctx.b.branch(ctx.label(body))?;
            }
            SyntaxNode::Break { cond, merge, skip } => {
                let cond = ctx.def(cond)?;
                ctx.b
                    .branch_conditional(cond, ctx.label(merge), ctx.label(skip), [])?;
            }
            SyntaxNode::EndIf { merge } => {
                if in_block {
                    ctx.b.branch(ctx.label(merge))?;
                }
            }
            SyntaxNode::Repeat {
                cond,
                loop_header,
                merge,
            } => {
                let cond = loop_safety_condition(ctx, cond)?;
                ctx.b
                    .branch_conditional(cond, ctx.label(loop_header), ctx.label(merge), [])?;
            }
            SyntaxNode::Return => ctx.b.ret()?,
            SyntaxNode::Unreachable => ctx.b.unreachable()?,
        }
        in_block = false;
    }
    Ok(())
}

/// ANDs a loop's continue condition with a per-loop iteration budget, so a guest loop that
/// never exits cannot hang the GPU.
fn loop_safety_condition(ctx: &mut EmitContext<'_>, cond: crate::ir::Value) -> EmitResult<Word> {
    let cond = ctx.def(cond)?;
    let u32_t = ctx.u32[1];
    let pointer_type = ctx.type_pointer(StorageClass::Private, u32_t);
    let initial = ctx.const_u32(LOOP_SAFETY_ITERATIONS);
    let counter = ctx.global_variable(pointer_type, StorageClass::Private, Some(initial));
    ctx.add_interface(counter, StorageClass::Private);
    let one = ctx.const_u32(1);
    let old = ctx.b.load(u32_t, None, counter, None, [])?;
    let new = ctx.b.i_sub(u32_t, None, old, one)?;
    ctx.b.store(counter, new, None, [])?;
    let safe = ctx
        .b
        .s_greater_than_equal(ctx.u1, None, new, ctx.u32_zero_value)?;
    Ok(ctx.b.logical_and(ctx.u1, None, cond, safe)?)
}

/// Emits a phi with placeholder values; [`patch_phis`] fills them in once every block has
/// been emitted, since a loop phi names values defined after it.
pub(super) fn emit_phi(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let program = ctx.program;
    let ir_inst = program.inst(inst);
    let ty = ctx.type_id(Type::from_bits_truncate(ir_inst.flags))?;
    let block = ctx
        .b
        .selected_block()
        .ok_or_else(|| EmitError::InvalidIr("phi outside a block".into()))?;
    let index = ctx.b.module_ref().functions[ctx.main_function].blocks[block]
        .instructions
        .len();
    let mut operands = SmallVec::<[(Word, Word); 4]>::new();
    for i in 0..ir_inst.num_args() {
        operands.push((0, ctx.label(ir_inst.phi_block(i))));
    }
    let id = ctx.b.phi(ty, None, operands)?;
    ctx.phis.push(DeferredPhi { block, index, inst });
    Ok(id)
}

fn patch_phis(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    let phis = std::mem::take(&mut ctx.phis);
    let mut values = SmallVec::<[Word; 4]>::new();
    for DeferredPhi { block, index, inst } in phis {
        values.clear();
        for i in 0..ctx.program.inst(inst).num_args() {
            let value = ctx.arg(inst, i);
            values.push(ctx.def(value)?);
        }
        let main = ctx.main_function;
        let spv_inst = &mut ctx.b.module_mut().functions[main].blocks[block].instructions[index];
        if spv_inst.class.opcode != Op::Phi {
            return Err(EmitError::InvalidIr("phi moved during emission".into()));
        }
        for (i, &value) in values.iter().enumerate() {
            spv_inst.operands[i * 2] = Operand::IdRef(value);
        }
    }
    Ok(())
}

fn define_entry_point(ctx: &mut EmitContext<'_>, main: Word) {
    let program = ctx.translated;
    let runtime = ctx.runtime;
    let profile = ctx.profile;
    let model = match program.stage {
        ShaderStage::Compute => {
            ctx.b
                .execution_mode(main, ExecutionMode::LocalSize, program.workgroup_size);
            ExecutionModel::GLCompute
        }
        ShaderStage::TessellationControl => {
            ctx.add_capability(Capability::Tessellation);
            ctx.b
                .execution_mode(main, ExecutionMode::OutputVertices, [program.invocations]);
            ExecutionModel::TessellationControl
        }
        ShaderStage::TessellationEval => {
            ctx.add_capability(Capability::Tessellation);
            let primitive = match runtime.tess_primitive {
                TessPrimitive::Isolines => ExecutionMode::Isolines,
                TessPrimitive::Triangles => ExecutionMode::Triangles,
                TessPrimitive::Quads => ExecutionMode::Quads,
            };
            let spacing = match runtime.tess_spacing {
                TessSpacing::Equal => ExecutionMode::SpacingEqual,
                TessSpacing::FractionalOdd => ExecutionMode::SpacingFractionalOdd,
                TessSpacing::FractionalEven => ExecutionMode::SpacingFractionalEven,
            };
            let order = if runtime.tess_clockwise {
                ExecutionMode::VertexOrderCw
            } else {
                ExecutionMode::VertexOrderCcw
            };
            for mode in [primitive, spacing, order] {
                ctx.b.execution_mode(main, mode, []);
            }
            ExecutionModel::TessellationEvaluation
        }
        ShaderStage::Geometry => {
            ctx.add_capability(Capability::Geometry);
            if profile.support_geometry_streams {
                ctx.add_capability(Capability::GeometryStreams);
            }
            let input = match runtime.input_topology {
                InputTopology::Points => ExecutionMode::InputPoints,
                InputTopology::Lines => ExecutionMode::InputLines,
                InputTopology::LinesAdjacency => ExecutionMode::InputLinesAdjacency,
                InputTopology::Triangles => ExecutionMode::Triangles,
                InputTopology::TrianglesAdjacency => ExecutionMode::InputTrianglesAdjacency,
            };
            let output = match program.output_topology {
                OutputTopology::PointList => ExecutionMode::OutputPoints,
                OutputTopology::LineStrip => ExecutionMode::OutputLineStrip,
                OutputTopology::TriangleStrip => ExecutionMode::OutputTriangleStrip,
            };
            ctx.b.execution_mode(main, input, []);
            ctx.b.execution_mode(main, output, []);
            if program.info.stores.get(Attribute::POINT_SIZE) {
                ctx.add_capability(Capability::GeometryPointSize);
            }
            ctx.b.execution_mode(
                main,
                ExecutionMode::OutputVertices,
                [program.output_vertices],
            );
            ctx.b
                .execution_mode(main, ExecutionMode::Invocations, [program.invocations]);
            if program.is_geometry_passthrough {
                if profile.support_geometry_shader_passthrough {
                    ctx.add_extension("SPV_NV_geometry_shader_passthrough");
                    ctx.add_capability(Capability::GeometryShaderPassthroughNV);
                } else {
                    tracing::warn!("geometry shader passthrough used with no host support");
                }
            }
            ExecutionModel::Geometry
        }
        ShaderStage::Fragment => {
            let origin = if profile.lower_left_origin_mode {
                ExecutionMode::OriginLowerLeft
            } else {
                ExecutionMode::OriginUpperLeft
            };
            ctx.b.execution_mode(main, origin, []);
            if program.info.stores_frag_depth {
                ctx.b
                    .execution_mode(main, ExecutionMode::DepthReplacing, []);
            }
            if runtime.force_early_z {
                ctx.b
                    .execution_mode(main, ExecutionMode::EarlyFragmentTests, []);
            }
            if profile.support_shader_quad_control && program.info.uses_quad_shuffles {
                ctx.add_extension("SPV_KHR_quad_control");
                ctx.add_capability(Capability::QuadControlKHR);
                ctx.b
                    .execution_mode(main, ExecutionMode::RequireFullQuadsKHR, []);
            }
            ExecutionModel::Fragment
        }
        ShaderStage::VertexA | ShaderStage::VertexB | ShaderStage::Task | ShaderStage::Mesh => {
            ExecutionModel::Vertex
        }
    };
    let interfaces = std::mem::take(&mut ctx.interfaces);
    ctx.b.entry_point(model, main, "main", &interfaces);
}

fn setup_denorm_control(ctx: &mut EmitContext<'_>, main: Word) {
    let info = ctx.info;
    let profile = ctx.profile;
    if info.uses_fp32_denorms_flush && info.uses_fp32_denorms_preserve {
        tracing::debug!("fp32 denorm flush and preserve in the same program");
    } else if info.uses_fp32_denorms_flush {
        if profile.support_fp32_denorm_flush && !profile.has_broken_fp32_denorm_flush {
            ctx.add_capability(Capability::DenormFlushToZero);
            ctx.b
                .execution_mode(main, ExecutionMode::DenormFlushToZero, [32]);
        }
    } else if info.uses_fp32_denorms_preserve {
        if profile.support_fp32_denorm_preserve {
            ctx.add_capability(Capability::DenormPreserve);
            ctx.b
                .execution_mode(main, ExecutionMode::DenormPreserve, [32]);
        } else {
            tracing::debug!("fp32 denorm preserve used without host support");
        }
    }
    if !profile.support_separate_denorm_behavior || profile.has_broken_fp16_float_controls {
        return;
    }
    if info.uses_fp16_denorms_flush && info.uses_fp16_denorms_preserve {
        tracing::debug!("fp16 denorm flush and preserve in the same program");
    } else if info.uses_fp16_denorms_flush {
        if profile.support_fp16_denorm_flush {
            ctx.add_capability(Capability::DenormFlushToZero);
            ctx.b
                .execution_mode(main, ExecutionMode::DenormFlushToZero, [16]);
        }
    } else if info.uses_fp16_denorms_preserve {
        if profile.support_fp16_denorm_preserve {
            ctx.add_capability(Capability::DenormPreserve);
            ctx.b
                .execution_mode(main, ExecutionMode::DenormPreserve, [16]);
        } else {
            tracing::debug!("fp16 denorm preserve used without host support");
        }
    }
}

fn setup_signed_nan_capabilities(ctx: &mut EmitContext<'_>, main: Word) {
    let info = ctx.info;
    let profile = ctx.profile;
    if profile.has_broken_fp16_float_controls && info.uses_fp16 {
        return;
    }
    for (used, supported, width) in [
        (
            info.uses_fp16,
            profile.support_fp16_signed_zero_nan_preserve,
            16,
        ),
        (true, profile.support_fp32_signed_zero_nan_preserve, 32),
        (
            info.uses_fp64,
            profile.support_fp64_signed_zero_nan_preserve,
            64,
        ),
    ] {
        if used && supported {
            ctx.add_capability(Capability::SignedZeroInfNanPreserve);
            ctx.b
                .execution_mode(main, ExecutionMode::SignedZeroInfNanPreserve, [width]);
        }
    }
}

fn setup_capabilities(ctx: &mut EmitContext<'_>) {
    let info = ctx.info;
    let profile = ctx.profile;
    if info.uses_sparse_residency {
        ctx.add_capability(Capability::SparseResidency);
    }
    if info.uses_demote_to_helper_invocation && profile.support_demote_to_helper_invocation {
        if profile.supported_spirv < 0x0001_0600 {
            ctx.add_extension("SPV_EXT_demote_to_helper_invocation");
        }
        ctx.add_capability(Capability::DemoteToHelperInvocation);
    }
    if info.stores.get(Attribute::VIEWPORT_INDEX) && profile.support_multi_viewport {
        ctx.add_capability(Capability::MultiViewport);
    }
    if info.stores.get(Attribute::VIEWPORT_MASK) && profile.support_viewport_mask {
        ctx.add_extension("SPV_NV_viewport_array2");
        ctx.add_capability(Capability::ShaderViewportMaskNV);
    }
    if (info.stores.get(Attribute::LAYER) || info.stores.get(Attribute::VIEWPORT_INDEX))
        && profile.support_viewport_index_layer_non_geometry
        && ctx.stage != ShaderStage::Geometry
    {
        ctx.add_extension("SPV_EXT_shader_viewport_index_layer");
        ctx.add_capability(Capability::ShaderViewportIndexLayerEXT);
    }
    if (info.uses_subgroup_vote
        || info.uses_subgroup_invocation_id
        || info.uses_subgroup_shuffles
        || info.uses_subgroup_mask)
        && profile.support_vote
        && ctx.supports_subgroups()
    {
        ctx.add_capability(Capability::GroupNonUniformBallot);
        ctx.add_capability(Capability::GroupNonUniformShuffle);
        if info.uses_subgroup_shuffles && profile.support_shuffle_relative {
            ctx.add_capability(Capability::GroupNonUniformShuffleRelative);
        }
        if !profile.warp_size_potentially_larger_than_guest {
            ctx.add_capability(Capability::GroupNonUniformVote);
        }
    }
    if info.uses_quad_shuffles {
        if profile.support_quad_shuffles {
            ctx.add_capability(Capability::GroupNonUniformQuad);
        }
        ctx.add_capability(Capability::GroupNonUniformShuffle);
    }
    if info.uses_int64_bit_atomics
        && (profile.support_int64_atomics || profile.support_shared_int64_atomics)
    {
        ctx.add_capability(Capability::Int64Atomics);
    }
    if info.uses_typeless_image_reads && profile.support_typeless_image_loads {
        ctx.add_capability(Capability::StorageImageReadWithoutFormat);
    }
    if info.uses_typeless_image_writes {
        ctx.add_capability(Capability::StorageImageWriteWithoutFormat);
    }
    if info.uses_sample_id {
        ctx.add_capability(Capability::SampleRateShading);
    }
    if info.uses_derivatives {
        ctx.add_capability(Capability::DerivativeControl);
    }
    ctx.add_capability(Capability::ImageGatherExtended);
    ctx.add_capability(Capability::ImageQuery);
    ctx.add_capability(Capability::SampledBuffer);
    if !ctx.non_uniform_ids.is_empty() {
        if profile.supported_spirv < 0x0001_0500 {
            ctx.add_extension("SPV_EXT_descriptor_indexing");
        }
        ctx.add_capability(Capability::ShaderNonUniform);
        if ctx.uses_nonuniform_sampled_image {
            ctx.add_capability(Capability::SampledImageArrayNonUniformIndexing);
        }
        if ctx.uses_nonuniform_storage_image {
            ctx.add_capability(Capability::StorageImageArrayNonUniformIndexing);
        }
        if ctx.uses_nonuniform_uniform_texel_buffer {
            ctx.add_capability(Capability::UniformTexelBufferArrayNonUniformIndexing);
        }
        if ctx.uses_nonuniform_storage_texel_buffer {
            ctx.add_capability(Capability::StorageTexelBufferArrayNonUniformIndexing);
        }
    }
}
