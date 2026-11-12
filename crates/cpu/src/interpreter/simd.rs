//! SIMD and floating-point data processing (the A64 "Data Processing -- Scalar Floating-Point and
//! Advanced SIMD" encoding group). Loads and stores of SIMD registers belong to the load/store
//! decoder, not here.

use crate::{HaltReason, ThreadContext};

/// Executes \`instruction\` if it is in the SIMD/FP data-processing group.
///
/// Returns \`None\` when the instruction belongs to another group, \`Some(Ok(()))\` when it
/// executed (the caller advances \`pc\`), and \`Some(Err(_))\` for an undefined encoding.
pub(super) fn execute(
    _ctx: &mut ThreadContext,
    _instruction: u32,
) -> Option<Result<(), HaltReason>> {
    None
}
