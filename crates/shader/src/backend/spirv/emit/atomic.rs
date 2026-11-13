//! Shared, storage and global memory atomics, from the C++ `emit_spirv_atomic.cpp`.
//!
//! 32-bit integer atomics map to SPIR-V atomics. Increment, decrement and the float atomics
//! call the compare-exchange loops the context defines. 64-bit storage atomics need
//! `Int64Atomics`; without it, and for the 32x2 forms, they fall back to a plain
//! load-modify-store, as the C++ does.

use rspirv::spirv::{GlslStd450Op, Scope, Word};

use super::memory::{storage_base, storage_pointer};
use crate::backend::spirv::context::{EmitContext, GlobalAtomic, StorageElement};
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::InstId;

#[derive(Clone, Copy)]
enum AtomicOp {
    IAdd,
    SMin,
    UMin,
    SMax,
    UMax,
    And,
    Or,
    Xor,
    Exchange,
}

impl AtomicOp {
    fn atomic(
        self,
        ctx: &mut EmitContext<'_>,
        ty: Word,
        pointer: Word,
        value: Word,
    ) -> EmitResult<Word> {
        let scope = ctx.const_u32(Scope::Device as u32);
        let semantics = ctx.u32_zero_value;
        let b = &mut ctx.b;
        Ok(match self {
            AtomicOp::IAdd => b.atomic_i_add(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::SMin => b.atomic_s_min(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::UMin => b.atomic_u_min(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::SMax => b.atomic_s_max(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::UMax => b.atomic_u_max(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::And => b.atomic_and(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::Or => b.atomic_or(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::Xor => b.atomic_xor(ty, None, pointer, scope, semantics, value)?,
            AtomicOp::Exchange => b.atomic_exchange(ty, None, pointer, scope, semantics, value)?,
        })
    }

    /// The operation applied non-atomically, for hosts without 64-bit atomics.
    fn plain(
        self,
        ctx: &mut EmitContext<'_>,
        ty: Word,
        value: Word,
        old: Word,
    ) -> EmitResult<Word> {
        let glsl = |op| (op, [value, old]);
        let (op, operands) = match self {
            AtomicOp::IAdd => return Ok(ctx.b.i_add(ty, None, value, old)?),
            AtomicOp::And => return Ok(ctx.b.bitwise_and(ty, None, value, old)?),
            AtomicOp::Or => return Ok(ctx.b.bitwise_or(ty, None, value, old)?),
            AtomicOp::Xor => return Ok(ctx.b.bitwise_xor(ty, None, value, old)?),
            AtomicOp::Exchange => return Ok(value),
            AtomicOp::SMin => glsl(GlslStd450Op::SMin),
            AtomicOp::UMin => glsl(GlslStd450Op::UMin),
            AtomicOp::SMax => glsl(GlslStd450Op::SMax),
            AtomicOp::UMax => glsl(GlslStd450Op::UMax),
        };
        ctx.glsl_inst(ty, op, &operands)
    }
}

/// Pointer to the shared memory word at byte `offset`, plus `index_offset` words.
fn shared_pointer(ctx: &mut EmitContext<'_>, offset: Word, index_offset: u32) -> EmitResult<Word> {
    let index = shared_word_index(ctx, offset)?;
    let u32_t = ctx.u32[1];
    let index = if index_offset == 0 {
        index
    } else {
        let index_offset = ctx.const_u32(index_offset);
        ctx.b.i_add(u32_t, None, index, index_offset)?
    };
    let (pointer_type, array) = (ctx.shared_u32, ctx.shared_memory_u32);
    Ok(if ctx.uses_explicit_workgroup_layout {
        let zero = ctx.u32_zero_value;
        ctx.b
            .access_chain(pointer_type, None, array, [zero, index])?
    } else {
        ctx.b.access_chain(pointer_type, None, array, [index])?
    })
}

fn shared_word_index(ctx: &mut EmitContext<'_>, offset: Word) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    let two = ctx.const_u32(2);
    Ok(ctx.b.shift_right_arithmetic(u32_t, None, offset, two)?)
}

fn shared_atomic(ctx: &mut EmitContext<'_>, inst: InstId, op: AtomicOp) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let pointer = shared_pointer(ctx, offset, 0)?;
    let u32_t = ctx.u32[1];
    op.atomic(ctx, u32_t, pointer, value)
}

fn shared_cas(ctx: &mut EmitContext<'_>, inst: InstId, func: Word) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let index = shared_word_index(ctx, offset)?;
    let u32_t = ctx.u32[1];
    Ok(ctx
        .b
        .function_call(u32_t, None, defined(func)?, [index, value])?)
}

/// Swaps the two shared memory words at the offset non-atomically, returning the old pair.
fn shared_exchange_words(ctx: &mut EmitContext<'_>, offset: Word, value: Word) -> EmitResult<Word> {
    let u32_t = ctx.u32[1];
    let u32x2 = ctx.u32[2];
    let low = shared_pointer(ctx, offset, 0)?;
    let high = shared_pointer(ctx, offset, 1)?;
    let old_low = ctx.b.load(u32_t, None, low, None, [])?;
    let old_high = ctx.b.load(u32_t, None, high, None, [])?;
    let new_low = ctx.b.composite_extract(u32_t, None, value, [0])?;
    let new_high = ctx.b.composite_extract(u32_t, None, value, [1])?;
    ctx.b.store(low, new_low, None, [])?;
    ctx.b.store(high, new_high, None, [])?;
    Ok(ctx
        .b
        .composite_construct(u32x2, None, [old_low, old_high])?)
}

/// A helper function the context defines when the program's info asks for it.
fn defined(func: Word) -> EmitResult<Word> {
    if func == 0 {
        return Err(EmitError::InvalidIr(
            "atomic whose helper the shader info did not request".into(),
        ));
    }
    Ok(func)
}

fn storage_atomic32(ctx: &mut EmitContext<'_>, inst: InstId, op: AtomicOp) -> EmitResult<Word> {
    let value = ctx.def_arg(inst, 2)?;
    let pointer = storage_pointer(ctx, inst, StorageElement::U32, 4, 0)?;
    let u32_t = ctx.u32[1];
    op.atomic(ctx, u32_t, pointer, value)
}

fn storage_atomic64(ctx: &mut EmitContext<'_>, inst: InstId, op: AtomicOp) -> EmitResult<Word> {
    let u64_t = ctx.u64;
    if !ctx.profile.support_descriptor_aliasing {
        tracing::warn!("descriptor aliasing not supported, 64-bit storage atomic ignored");
        return Ok(ctx.const_null(u64_t));
    }
    let value = ctx.def_arg(inst, 2)?;
    if ctx.profile.support_int64_atomics {
        let pointer = storage_pointer(ctx, inst, StorageElement::U64, 8, 0)?;
        return op.atomic(ctx, u64_t, pointer, value);
    }
    tracing::warn!("int64 atomics not supported, falling back to non-atomic");
    let u32x2 = ctx.u32[2];
    let pointer = storage_pointer(ctx, inst, StorageElement::U32x2, 8, 0)?;
    let words = ctx.b.load(u32x2, None, pointer, None, [])?;
    let old = ctx.b.bitcast(u64_t, None, words)?;
    let new = op.plain(ctx, u64_t, value, old)?;
    let new_words = ctx.b.bitcast(u32x2, None, new)?;
    ctx.b.store(pointer, new_words, None, [])?;
    Ok(old)
}

fn storage_atomic32x2(ctx: &mut EmitContext<'_>, inst: InstId, op: AtomicOp) -> EmitResult<Word> {
    let u32x2 = ctx.u32[2];
    if !ctx.profile.support_descriptor_aliasing {
        tracing::warn!("descriptor aliasing not supported, 64-bit storage atomic ignored");
        return Ok(ctx.const_null(u32x2));
    }
    tracing::warn!("int64 atomics not supported, falling back to non-atomic");
    let value = ctx.def_arg(inst, 2)?;
    let pointer = storage_pointer(ctx, inst, StorageElement::U32x2, 8, 0)?;
    let old = ctx.b.load(u32x2, None, pointer, None, [])?;
    let new = op.plain(ctx, u32x2, value, old)?;
    ctx.b.store(pointer, new, None, [])?;
    Ok(old)
}

/// Calls a storage compare-exchange loop on the word the instruction addresses, returning the
/// old value as `ty`.
fn storage_cas(ctx: &mut EmitContext<'_>, inst: InstId, func: Word, ty: Word) -> EmitResult<Word> {
    let func = defined(func)?;
    let value = ctx.def_arg(inst, 2)?;
    let (ssbo, index) = storage_base(ctx, inst, StorageElement::U32, 4)?;
    Ok(ctx.b.function_call(ty, None, func, [index, value, ssbo])?)
}

fn storage_cas_f16x2(ctx: &mut EmitContext<'_>, inst: InstId, func: Word) -> EmitResult<Word> {
    let (f16x2, u32_t) = (ctx.f16[2], ctx.u32[1]);
    let old = storage_cas(ctx, inst, func, f16x2)?;
    Ok(ctx.b.bitcast(u32_t, None, old)?)
}

fn storage_cas_f32x2(ctx: &mut EmitContext<'_>, inst: InstId, func: Word) -> EmitResult<Word> {
    let (f32x2, u32_t) = (ctx.f32[2], ctx.u32[1]);
    let old = storage_cas(ctx, inst, func, f32x2)?;
    ctx.glsl_inst(u32_t, GlslStd450Op::PackHalf2x16, &[old])
}

/// Checks that untracked global atomics can be routed, as the C++ `GlobalAtomic` does.
fn global_routable(ctx: &EmitContext<'_>) -> EmitResult<()> {
    if !ctx.profile.support_int64 {
        return Err(EmitError::Unsupported(
            "global atomic addressing requires 64-bit integers",
        ));
    }
    Ok(())
}

fn global_atomic(ctx: &mut EmitContext<'_>, inst: InstId, kind: GlobalAtomic) -> EmitResult<Word> {
    global_routable(ctx)?;
    let func = ctx
        .global_atomic_function(kind)
        .ok_or(EmitError::Unsupported(
            "global atomic with no routable storage buffer",
        ))?;
    let ty = if kind == GlobalAtomic::AddF32 {
        ctx.f32[1]
    } else {
        ctx.u32[1]
    };
    let address = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    Ok(ctx.b.function_call(ty, None, func, [address, value])?)
}

const UNTRACKED_WIDE_GLOBAL_ATOMIC: EmitError = EmitError::Unsupported(
    "64-bit, 32x2 and packed float global atomics the storage buffer pass left",
);

pub(super) fn shared_atomic_i_add32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::IAdd)
}

pub(super) fn shared_atomic_s_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::SMin)
}

pub(super) fn shared_atomic_u_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::UMin)
}

pub(super) fn shared_atomic_s_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::SMax)
}

pub(super) fn shared_atomic_u_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::UMax)
}

pub(super) fn shared_atomic_inc32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.increment_cas_shared;
    shared_cas(ctx, inst, func)
}

pub(super) fn shared_atomic_dec32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let func = ctx.decrement_cas_shared;
    shared_cas(ctx, inst, func)
}

pub(super) fn shared_atomic_and32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::And)
}

pub(super) fn shared_atomic_or32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::Or)
}

pub(super) fn shared_atomic_xor32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::Xor)
}

pub(super) fn shared_atomic_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    shared_atomic(ctx, inst, AtomicOp::Exchange)
}

pub(super) fn shared_atomic_exchange64(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    let u64_t = ctx.u64;
    if ctx.profile.support_shared_int64_atomics
        && ctx.uses_explicit_workgroup_layout
        && ctx.shared_memory_u64 != 0
    {
        let u32_t = ctx.u32[1];
        let three = ctx.const_u32(3);
        let index = ctx.b.shift_right_arithmetic(u32_t, None, offset, three)?;
        let zero = ctx.u32_zero_value;
        let (pointer_type, array) = (ctx.shared_u64, ctx.shared_memory_u64);
        let pointer = ctx
            .b
            .access_chain(pointer_type, None, array, [zero, index])?;
        return AtomicOp::Exchange.atomic(ctx, u64_t, pointer, value);
    }
    tracing::warn!("shared int64 atomics not supported, falling back to non-atomic");
    let u32x2 = ctx.u32[2];
    let words = ctx.b.bitcast(u32x2, None, value)?;
    let old = shared_exchange_words(ctx, offset, words)?;
    Ok(ctx.b.bitcast(u64_t, None, old)?)
}

pub(super) fn shared_atomic_exchange32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    tracing::warn!("shared int64 atomics not supported, falling back to non-atomic");
    let offset = ctx.def_arg(inst, 0)?;
    let value = ctx.def_arg(inst, 1)?;
    shared_exchange_words(ctx, offset, value)
}

pub(super) fn shared_atomic_compare_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let offset = ctx.def_arg(inst, 0)?;
    let comparator = ctx.def_arg(inst, 1)?;
    let value = ctx.def_arg(inst, 2)?;
    let pointer = shared_pointer(ctx, offset, 0)?;
    let u32_t = ctx.u32[1];
    let scope = ctx.const_u32(Scope::Device as u32);
    let semantics = ctx.u32_zero_value;
    Ok(ctx.b.atomic_compare_exchange(
        u32_t, None, pointer, scope, semantics, semantics, value, comparator,
    )?)
}

pub(super) fn global_atomic_i_add32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::IAdd)
}

pub(super) fn global_atomic_s_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::SMin)
}

pub(super) fn global_atomic_u_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::UMin)
}

pub(super) fn global_atomic_s_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::SMax)
}

pub(super) fn global_atomic_u_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::UMax)
}

pub(super) fn global_atomic_inc32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::Inc)
}

pub(super) fn global_atomic_dec32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::Dec)
}

pub(super) fn global_atomic_and32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::And)
}

pub(super) fn global_atomic_or32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::Or)
}

pub(super) fn global_atomic_xor32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::Xor)
}

pub(super) fn global_atomic_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::Exchange)
}

pub(super) fn global_atomic_compare_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    global_routable(ctx)?;
    let func = ctx
        .global_compare_exchange_function()
        .ok_or(EmitError::Unsupported(
            "global atomic compare-exchange with no routable storage buffer",
        ))?;
    let address = ctx.def_arg(inst, 0)?;
    let comparator = ctx.def_arg(inst, 1)?;
    let value = ctx.def_arg(inst, 2)?;
    let u32_t = ctx.u32[1];
    Ok(ctx
        .b
        .function_call(u32_t, None, func, [address, comparator, value])?)
}

pub(super) fn global_atomic_i_add64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_s_min64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_u_min64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_s_max64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_u_max64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_and64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_or64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_xor64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_exchange64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_i_add32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_s_min32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_u_min32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_s_max32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_u_max32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_and32x2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_or32x2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_xor32x2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_exchange32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_add_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    global_atomic(ctx, inst, GlobalAtomic::AddF32)
}

pub(super) fn global_atomic_add_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_add_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_min_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_min_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_max_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn global_atomic_max_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(UNTRACKED_WIDE_GLOBAL_ATOMIC)
}

pub(super) fn storage_atomic_i_add32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::IAdd)
}

pub(super) fn storage_atomic_s_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::SMin)
}

pub(super) fn storage_atomic_u_min32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::UMin)
}

pub(super) fn storage_atomic_s_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::SMax)
}

pub(super) fn storage_atomic_u_max32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::UMax)
}

pub(super) fn storage_atomic_inc32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (func, ty) = (ctx.increment_cas_ssbo, ctx.u32[1]);
    storage_cas(ctx, inst, func, ty)
}

pub(super) fn storage_atomic_dec32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (func, ty) = (ctx.decrement_cas_ssbo, ctx.u32[1]);
    storage_cas(ctx, inst, func, ty)
}

pub(super) fn storage_atomic_and32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::And)
}

pub(super) fn storage_atomic_or32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::Or)
}

pub(super) fn storage_atomic_xor32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::Xor)
}

pub(super) fn storage_atomic_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic32(ctx, inst, AtomicOp::Exchange)
}

pub(super) fn storage_atomic_compare_exchange32(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let comparator = ctx.def_arg(inst, 2)?;
    let value = ctx.def_arg(inst, 3)?;
    let pointer = storage_pointer(ctx, inst, StorageElement::U32, 4, 0)?;
    let u32_t = ctx.u32[1];
    let scope = ctx.const_u32(Scope::Device as u32);
    let semantics = ctx.u32_zero_value;
    Ok(ctx.b.atomic_compare_exchange(
        u32_t, None, pointer, scope, semantics, semantics, value, comparator,
    )?)
}

pub(super) fn storage_atomic_i_add64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::IAdd)
}

pub(super) fn storage_atomic_s_min64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::SMin)
}

pub(super) fn storage_atomic_u_min64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::UMin)
}

pub(super) fn storage_atomic_s_max64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::SMax)
}

pub(super) fn storage_atomic_u_max64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::UMax)
}

pub(super) fn storage_atomic_and64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::And)
}

pub(super) fn storage_atomic_or64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::Or)
}

pub(super) fn storage_atomic_xor64(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::Xor)
}

pub(super) fn storage_atomic_exchange64(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic64(ctx, inst, AtomicOp::Exchange)
}

pub(super) fn storage_atomic_i_add32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::IAdd)
}

pub(super) fn storage_atomic_s_min32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::SMin)
}

pub(super) fn storage_atomic_u_min32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::UMin)
}

pub(super) fn storage_atomic_s_max32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::SMax)
}

pub(super) fn storage_atomic_u_max32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::UMax)
}

pub(super) fn storage_atomic_and32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::And)
}

pub(super) fn storage_atomic_or32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::Or)
}

pub(super) fn storage_atomic_xor32x2(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::Xor)
}

pub(super) fn storage_atomic_exchange32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    storage_atomic32x2(ctx, inst, AtomicOp::Exchange)
}

pub(super) fn storage_atomic_add_f32(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<Word> {
    let (func, ty) = (ctx.f32_add_cas, ctx.f32[1]);
    storage_cas(ctx, inst, func, ty)
}

pub(super) fn storage_atomic_add_f16x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let func = ctx.f16x2_add_cas;
    storage_cas_f16x2(ctx, inst, func)
}

pub(super) fn storage_atomic_add_f32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let func = ctx.f32x2_add_cas;
    storage_cas_f32x2(ctx, inst, func)
}

pub(super) fn storage_atomic_min_f16x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let func = ctx.f16x2_min_cas;
    storage_cas_f16x2(ctx, inst, func)
}

pub(super) fn storage_atomic_min_f32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let func = ctx.f32x2_min_cas;
    storage_cas_f32x2(ctx, inst, func)
}

pub(super) fn storage_atomic_max_f16x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let func = ctx.f16x2_max_cas;
    storage_cas_f16x2(ctx, inst, func)
}

pub(super) fn storage_atomic_max_f32x2(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
) -> EmitResult<Word> {
    let func = ctx.f32x2_max_cas;
    storage_cas_f32x2(ctx, inst, func)
}
