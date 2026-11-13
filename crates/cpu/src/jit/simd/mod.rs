//! The JIT side of the SIMD/FP data-processing group.
//!
//! Instructions [`lower`] recognises become [`SimdOp`]s that the backend compiles inline. Every
//! other one becomes `Op::CallSimd`, which runs the interpreter's `simd::execute` on the live
//! context through [`call`].

mod ir;
mod translate;
mod x64;

pub(super) use self::ir::SimdOp;
pub(super) use self::translate::lower;
pub(super) use self::x64::State;
use super::JitState;
use crate::interpreter::simd::execute;
use crate::{CpuFault, ThreadContext};

/// Runs `Op::CallSimd` on the live context. Returns true, with the fault recorded in `state`,
/// when the instruction is undefined.
pub(super) extern "sysv64" fn call(
    ctx: &mut ThreadContext,
    word: u32,
    state: &mut JitState,
    pc: u64,
) -> bool {
    if let Some(Ok(())) = execute(ctx, word) {
        return false;
    }
    state.fault = Some(CpuFault::Undefined {
        pc,
        instruction: word,
    });
    true
}
