//! Global and storage buffer loads and stores, from the C++ `emit_spirv_memory.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn load_global_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadGlobalU8))
}

pub(super) fn load_global_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadGlobalS8))
}

pub(super) fn load_global_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadGlobalU16))
}

pub(super) fn load_global_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadGlobalS16))
}

pub(super) fn load_global32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadGlobal32))
}

pub(super) fn load_global64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadGlobal64))
}

pub(super) fn load_global128(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadGlobal128))
}

pub(super) fn write_global_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteGlobalU8))
}

pub(super) fn write_global_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteGlobalS8))
}

pub(super) fn write_global_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteGlobalU16))
}

pub(super) fn write_global_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteGlobalS16))
}

pub(super) fn write_global32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteGlobal32))
}

pub(super) fn write_global64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteGlobal64))
}

pub(super) fn write_global128(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteGlobal128))
}

pub(super) fn load_storage_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadStorageU8))
}

pub(super) fn load_storage_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadStorageS8))
}

pub(super) fn load_storage_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadStorageU16))
}

pub(super) fn load_storage_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadStorageS16))
}

pub(super) fn load_storage32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadStorage32))
}

pub(super) fn load_storage64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadStorage64))
}

pub(super) fn load_storage128(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::LoadStorage128))
}

pub(super) fn write_storage_u8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteStorageU8))
}

pub(super) fn write_storage_s8(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteStorageS8))
}

pub(super) fn write_storage_u16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteStorageU16))
}

pub(super) fn write_storage_s16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteStorageS16))
}

pub(super) fn write_storage32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteStorage32))
}

pub(super) fn write_storage64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteStorage64))
}

pub(super) fn write_storage128(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<()> {
    Err(EmitError::Unimplemented(Opcode::WriteStorage128))
}
