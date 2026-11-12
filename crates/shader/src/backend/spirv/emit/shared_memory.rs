//! Shared memory loads and stores, from the C++ `emit_spirv_shared_memory.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn load_shared_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadSharedU8))
}

pub(super) fn load_shared_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadSharedS8))
}

pub(super) fn load_shared_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadSharedU16))
}

pub(super) fn load_shared_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadSharedS16))
}

pub(super) fn load_shared_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadSharedU32))
}

pub(super) fn load_shared_u64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadSharedU64))
}

pub(super) fn load_shared_u128(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadSharedU128))
}

pub(super) fn write_shared_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteSharedU8))
}

pub(super) fn write_shared_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteSharedU16))
}

pub(super) fn write_shared_u32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteSharedU32))
}

pub(super) fn write_shared_u64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteSharedU64))
}

pub(super) fn write_shared_u128(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteSharedU128))
}
