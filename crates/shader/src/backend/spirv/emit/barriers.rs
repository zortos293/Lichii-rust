//! Barriers, from the C++ `emit_spirv_barriers.cpp`.

use rspirv::spirv::{MemorySemantics, Scope};

use crate::ShaderStage;
use crate::backend::spirv::EmitResult;
use crate::backend::spirv::context::EmitContext;
use crate::ir::InstId;

fn memory_barrier(ctx: &mut EmitContext<'_>, scope: Scope) -> EmitResult<()> {
    let semantics = MemorySemantics::ACQUIRE_RELEASE
        | MemorySemantics::UNIFORM_MEMORY
        | MemorySemantics::WORKGROUP_MEMORY
        | MemorySemantics::ATOMIC_COUNTER_MEMORY
        | MemorySemantics::IMAGE_MEMORY;
    let scope = ctx.const_u32(scope as u32);
    let semantics = ctx.const_u32(semantics.bits());
    Ok(ctx.b.memory_barrier(scope, semantics)?)
}

/// Vulkan's GLSL450 memory model allows workgroup execution scope in compute, task, mesh and
/// tessellation control programs, and workgroup memory scope in all of those but tessellation
/// control. A tessellation control barrier synchronises execution like GLSL's `barrier()`;
/// elsewhere `BAR.SYNC` synchronises nothing and is dropped.
pub(super) fn barrier(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    let workgroup = ctx.const_u32(Scope::Workgroup as u32);
    match ctx.stage {
        ShaderStage::Compute | ShaderStage::Task | ShaderStage::Mesh => {
            let mut semantics =
                MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY;
            if ctx.requires_coherent_global_memory {
                semantics |= MemorySemantics::UNIFORM_MEMORY;
            }
            if ctx.requires_coherent_image_memory {
                semantics |= MemorySemantics::IMAGE_MEMORY;
            }
            let semantics = ctx.const_u32(semantics.bits());
            Ok(ctx.b.control_barrier(workgroup, workgroup, semantics)?)
        }
        ShaderStage::TessellationControl => {
            let invocation = ctx.const_u32(Scope::Invocation as u32);
            let none = ctx.const_u32(0);
            Ok(ctx.b.control_barrier(workgroup, invocation, none)?)
        }
        _ => Ok(()),
    }
}

/// Widened to device scope outside compute, task and mesh programs, which have no workgroup
/// memory scope.
pub(super) fn workgroup_memory_barrier(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    let scope = if matches!(
        ctx.stage,
        ShaderStage::Compute | ShaderStage::Task | ShaderStage::Mesh
    ) {
        Scope::Workgroup
    } else {
        Scope::Device
    };
    memory_barrier(ctx, scope)
}

pub(super) fn device_memory_barrier(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    memory_barrier(ctx, Scope::Device)
}
