//! Attributes, interpolation, internal stage buffers, geometry output, special registers and barriers.

use super::{TResult, Translator};
use crate::ampere::opcode::Opcode;

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    let _ = op;
    Err(t.unsupported())
}
