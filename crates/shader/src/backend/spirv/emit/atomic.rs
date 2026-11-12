//! Shared, storage and global memory atomics, from the C++ `emit_spirv_atomic.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn shared_atomic_i_add32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicIAdd32))
}

pub(super) fn shared_atomic_s_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicSMin32))
}

pub(super) fn shared_atomic_u_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicUMin32))
}

pub(super) fn shared_atomic_s_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicSMax32))
}

pub(super) fn shared_atomic_u_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicUMax32))
}

pub(super) fn shared_atomic_inc32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicInc32))
}

pub(super) fn shared_atomic_dec32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicDec32))
}

pub(super) fn shared_atomic_and32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicAnd32))
}

pub(super) fn shared_atomic_or32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicOr32))
}

pub(super) fn shared_atomic_xor32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicXor32))
}

pub(super) fn shared_atomic_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicExchange32))
}

pub(super) fn shared_atomic_exchange64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicExchange64))
}

pub(super) fn shared_atomic_exchange32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::SharedAtomicExchange32x2))
}

pub(super) fn global_atomic_i_add32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicIAdd32))
}

pub(super) fn global_atomic_s_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicSMin32))
}

pub(super) fn global_atomic_u_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicUMin32))
}

pub(super) fn global_atomic_s_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicSMax32))
}

pub(super) fn global_atomic_u_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicUMax32))
}

pub(super) fn global_atomic_inc32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicInc32))
}

pub(super) fn global_atomic_dec32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicDec32))
}

pub(super) fn global_atomic_and32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicAnd32))
}

pub(super) fn global_atomic_or32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicOr32))
}

pub(super) fn global_atomic_xor32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicXor32))
}

pub(super) fn global_atomic_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicExchange32))
}

pub(super) fn global_atomic_compare_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::GlobalAtomicCompareExchange32,
    ))
}

pub(super) fn global_atomic_i_add64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicIAdd64))
}

pub(super) fn global_atomic_s_min64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicSMin64))
}

pub(super) fn global_atomic_u_min64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicUMin64))
}

pub(super) fn global_atomic_s_max64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicSMax64))
}

pub(super) fn global_atomic_u_max64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicUMax64))
}

pub(super) fn global_atomic_and64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicAnd64))
}

pub(super) fn global_atomic_or64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicOr64))
}

pub(super) fn global_atomic_xor64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicXor64))
}

pub(super) fn global_atomic_exchange64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicExchange64))
}

pub(super) fn global_atomic_i_add32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicIAdd32x2))
}

pub(super) fn global_atomic_s_min32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicSMin32x2))
}

pub(super) fn global_atomic_u_min32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicUMin32x2))
}

pub(super) fn global_atomic_s_max32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicSMax32x2))
}

pub(super) fn global_atomic_u_max32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicUMax32x2))
}

pub(super) fn global_atomic_and32x2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicAnd32x2))
}

pub(super) fn global_atomic_or32x2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicOr32x2))
}

pub(super) fn global_atomic_xor32x2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicXor32x2))
}

pub(super) fn global_atomic_exchange32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicExchange32x2))
}

pub(super) fn global_atomic_add_f32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicAddF32))
}

pub(super) fn global_atomic_add_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicAddF16x2))
}

pub(super) fn global_atomic_add_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicAddF32x2))
}

pub(super) fn global_atomic_min_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicMinF16x2))
}

pub(super) fn global_atomic_min_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicMinF32x2))
}

pub(super) fn global_atomic_max_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicMaxF16x2))
}

pub(super) fn global_atomic_max_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::GlobalAtomicMaxF32x2))
}

pub(super) fn storage_atomic_i_add32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicIAdd32))
}

pub(super) fn storage_atomic_s_min32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicSMin32))
}

pub(super) fn storage_atomic_u_min32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicUMin32))
}

pub(super) fn storage_atomic_s_max32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicSMax32))
}

pub(super) fn storage_atomic_u_max32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicUMax32))
}

pub(super) fn storage_atomic_inc32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicInc32))
}

pub(super) fn storage_atomic_dec32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicDec32))
}

pub(super) fn storage_atomic_and32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicAnd32))
}

pub(super) fn storage_atomic_or32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicOr32))
}

pub(super) fn storage_atomic_xor32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicXor32))
}

pub(super) fn storage_atomic_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicExchange32))
}

pub(super) fn storage_atomic_compare_exchange32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(
        Opcode::StorageAtomicCompareExchange32,
    ))
}

pub(super) fn storage_atomic_i_add64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicIAdd64))
}

pub(super) fn storage_atomic_s_min64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicSMin64))
}

pub(super) fn storage_atomic_u_min64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicUMin64))
}

pub(super) fn storage_atomic_s_max64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicSMax64))
}

pub(super) fn storage_atomic_u_max64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicUMax64))
}

pub(super) fn storage_atomic_and64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicAnd64))
}

pub(super) fn storage_atomic_or64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicOr64))
}

pub(super) fn storage_atomic_xor64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicXor64))
}

pub(super) fn storage_atomic_exchange64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicExchange64))
}

pub(super) fn storage_atomic_i_add32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicIAdd32x2))
}

pub(super) fn storage_atomic_s_min32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicSMin32x2))
}

pub(super) fn storage_atomic_u_min32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicUMin32x2))
}

pub(super) fn storage_atomic_s_max32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicSMax32x2))
}

pub(super) fn storage_atomic_u_max32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicUMax32x2))
}

pub(super) fn storage_atomic_and32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicAnd32x2))
}

pub(super) fn storage_atomic_or32x2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicOr32x2))
}

pub(super) fn storage_atomic_xor32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicXor32x2))
}

pub(super) fn storage_atomic_exchange32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicExchange32x2))
}

pub(super) fn storage_atomic_add_f32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicAddF32))
}

pub(super) fn storage_atomic_add_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicAddF16x2))
}

pub(super) fn storage_atomic_add_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicAddF32x2))
}

pub(super) fn storage_atomic_min_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicMinF16x2))
}

pub(super) fn storage_atomic_min_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicMinF32x2))
}

pub(super) fn storage_atomic_max_f16x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicMaxF16x2))
}

pub(super) fn storage_atomic_max_f32x2(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::StorageAtomicMaxF32x2))
}
