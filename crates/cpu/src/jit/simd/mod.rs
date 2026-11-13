//! The JIT side of the SIMD/FP data-processing group.

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
