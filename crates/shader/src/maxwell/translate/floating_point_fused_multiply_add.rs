//! Stubs until ported from the C++ floating_point_fused_multiply_add.cpp.

use super::{TResult, Translator};

pub(super) fn ffma_cr(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn ffma_rc(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn ffma_reg(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn ffma_imm(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}

pub(super) fn ffma32i(t: &mut Translator<'_>) -> TResult {
    Err(t.unsupported())
}
