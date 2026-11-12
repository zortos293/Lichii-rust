//! SIMD and floating-point data processing (the A64 "Data Processing -- Scalar Floating-Point and
//! Advanced SIMD" encoding group). Loads and stores of SIMD registers belong to the load/store
//! decoder, not here.
//!
//! All floating point goes through [`softfloat`], which follows the Arm ARM pseudocode bit for
//! bit, so results and FPSR flags match hardware under every FPCR setting.

mod advsimd;
mod fp_vector;
mod indexed;
mod int;
mod lanes;
mod movement;
mod scalar_fp;
mod shift;
pub(crate) mod softfloat;
#[cfg(test)]
mod tests;

use crate::{CpuFault, HaltReason, ThreadContext};

use softfloat::{Fmt, FpEnv};

/// An unallocated encoding inside the group.
struct Undefined;

type Exec = Result<(), Undefined>;

/// Executes `instruction` if it is in the SIMD/FP data-processing group.
///
/// Returns `None` when the instruction belongs to another group, `Some(Ok(()))` when it
/// executed (the caller advances `pc`), and `Some(Err(_))` for an undefined encoding.
pub(super) fn execute(ctx: &mut ThreadContext, instruction: u32) -> Option<Result<(), HaltReason>> {
    if bits(instruction, 27, 25) != 0b111 {
        return None;
    }
    let result = match bits(instruction, 31, 28) {
        op0 if op0 & 0b0101 == 0b0001 => scalar_fp::execute(ctx, instruction),
        op0 if op0 & 0b1001 == 0 => advsimd::vector(ctx, instruction),
        op0 if op0 & 0b1101 == 0b0101 => advsimd::scalar(ctx, instruction),
        _ => Err(Undefined),
    };
    Some(result.map_err(|Undefined| {
        HaltReason::Fault(CpuFault::Undefined {
            pc: ctx.pc,
            instruction,
        })
    }))
}

fn bits(instruction: u32, hi: u32, lo: u32) -> u32 {
    (instruction >> lo) & ((1 << (hi - lo + 1)) - 1)
}

fn bit(instruction: u32, n: u32) -> bool {
    (instruction >> n) & 1 != 0
}

/// Calls `$f::<N>(args)` with the width an FP `type`/`ftype` field selects.
macro_rules! with_fp_width {
    ($ftype:expr, $f:ident($($arg:expr),* $(,)?)) => {
        match $ftype {
            0 => $f::<32>($($arg),*),
            1 => $f::<64>($($arg),*),
            3 => $f::<16>($($arg),*),
            _ => Err(Undefined),
        }
    };
}
use with_fp_width;

fn read_x(ctx: &ThreadContext, n: u32) -> u64 {
    if n == 31 { 0 } else { ctx.x[n as usize] }
}

fn write_x(ctx: &mut ThreadContext, d: u32, value: u64) {
    if d != 31 {
        ctx.x[d as usize] = value;
    }
}

fn read_scalar<const N: u32>(ctx: &ThreadContext, n: u32) -> u64 {
    ctx.v[n as usize] as u64 & Fmt::<N>::MASK
}

/// Writes a scalar result, clearing the rest of the vector register.
fn write_scalar(ctx: &mut ThreadContext, d: u32, value: u64) {
    ctx.v[d as usize] = value as u128;
}

fn set_nzcv(ctx: &mut ThreadContext, nzcv: u32) {
    ctx.pstate = (ctx.pstate & 0x0fff_ffff) | (nzcv << 28);
}

fn commit_flags(ctx: &mut ThreadContext, env: &FpEnv) {
    ctx.fpsr |= env.flags;
}

/// `ConditionHolds` against PSTATE.NZCV.
fn condition_holds(pstate: u32, cond: u32) -> bool {
    let n = bit(pstate, 31);
    let z = bit(pstate, 30);
    let c = bit(pstate, 29);
    let v = bit(pstate, 28);
    let result = match cond >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if cond & 1 == 1 && cond != 0b1111 {
        !result
    } else {
        result
    }
}
