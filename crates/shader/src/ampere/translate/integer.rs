//! Integer, logic, move and predicate operations, on both datapaths.

use super::{TResult, Translator};
use crate::ampere::opcode::Opcode;

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    let _ = op;
    Err(t.unsupported())
}
