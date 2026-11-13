//! The JIT side of the SIMD/FP data-processing group.
//!
//! Instructions [`lower`] recognises become [`SimdOp`]s that the backend compiles inline. Every
//! other one becomes `Op::CallSimd`, which runs the interpreter's `simd::execute` on the live
//! context through [`call`].

mod ir;
mod translate;
mod vector;
mod x64;

pub(super) use self::ir::SimdOp;
pub(super) use self::translate::lower;
pub(super) use self::vector::{load_store_multiple, load_store_single};
pub(super) use self::x64::State;
use super::JitState;
use crate::interpreter::simd::execute;
use crate::{CpuFault, ThreadContext};

/// MXCSR with every exception masked, round to nearest, and neither flush-to-zero nor
/// denormals-are-zero: the host mode the native FP ops assume.
const MXCSR: u32 = 0x1f80;
const MXCSR_FLAGS: u32 = 0x3f;

pub(super) fn mxcsr() -> u32 {
    let mut csr = 0u32;
    // SAFETY: STMXCSR only stores this thread's SSE control register to a local.
    unsafe {
        std::arch::asm!("stmxcsr dword ptr [{}]", in(reg) &mut csr, options(nostack, preserves_flags))
    };
    csr
}

pub(super) fn set_mxcsr(csr: u32) {
    // SAFETY: LDMXCSR changes only this thread's SSE mode; callers pass a value STMXCSR produced
    // or `MXCSR`, so no reserved bit is set.
    unsafe {
        std::arch::asm!("ldmxcsr dword ptr [{}]", in(reg) &csr, options(nostack, preserves_flags, readonly))
    };
}

/// Holds the host in the mode the native FP ops assume and restores the caller's MXCSR on drop.
pub(super) struct HostFp(u32);

impl HostFp {
    pub(super) fn enter() -> Self {
        let saved = mxcsr();
        if saved & !MXCSR_FLAGS != MXCSR {
            set_mxcsr(MXCSR);
        }
        HostFp(saved)
    }
}

impl Drop for HostFp {
    fn drop(&mut self) {
        set_mxcsr(self.0);
    }
}

/// [`call`] for `Op::CallSimd`, counted in the core's profile.
pub(super) extern "sysv64" fn call_profiled(
    ctx: &mut ThreadContext,
    word: u32,
    state: &mut JitState,
    pc: u64,
) -> bool {
    if let Some(p) = state.profile.as_deref_mut() {
        p.simd_call(word);
    }
    call(ctx, word, state, pc)
}

/// [`call`] for the slow path of a native op, counted in the core's profile.
pub(super) extern "sysv64" fn slow_profiled(
    ctx: &mut ThreadContext,
    word: u32,
    state: &mut JitState,
    pc: u64,
) -> bool {
    if let Some(p) = state.profile.as_deref_mut() {
        p.simd_slow(word, ctx.fpcr & x64::HOST_FPCR as u32 != 0);
    }
    call(ctx, word, state, pc)
}

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
