//! Barriers, from the C++ `emit_spirv_barriers.cpp`.

use rspirv::spirv::{MemorySemantics, Scope};

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

pub(super) fn barrier(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    let mut semantics = MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY;
    if ctx.requires_coherent_global_memory {
        semantics |= MemorySemantics::UNIFORM_MEMORY;
    }
    if ctx.requires_coherent_image_memory {
        semantics |= MemorySemantics::IMAGE_MEMORY;
    }
    let workgroup = ctx.const_u32(Scope::Workgroup as u32);
    let semantics = ctx.const_u32(semantics.bits());
    Ok(ctx.b.control_barrier(workgroup, workgroup, semantics)?)
}

pub(super) fn workgroup_memory_barrier(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    memory_barrier(ctx, Scope::Workgroup)
}

pub(super) fn device_memory_barrier(ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    memory_barrier(ctx, Scope::Device)
}
