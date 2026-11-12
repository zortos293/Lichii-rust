//! Rewrites IR patterns host drivers miscompile, ported from `vendor_workaround_pass.cpp`.

use crate::ShaderStage;
use crate::ir::{Attribute, Emitter, InstId, Opcode, Program, Value};

/// Matches `(BitFieldUExtract(a, 0, 16) [* b]) << 16 + BitFieldUExtract(a, 16, 16)`.
fn is_byte_swap_add(prog: &Program, add: InstId) -> bool {
    let inst_of = |value: Value| prog.resolve(value).inst();
    let (Some(lhs_shl), Some(rhs_bfe)) = (inst_of(prog.arg(add, 0)), inst_of(prog.arg(add, 1)))
    else {
        return false;
    };
    if prog.op(lhs_shl) != Opcode::ShiftLeftLogical32 || prog.arg(lhs_shl, 1) != Value::U32(16) {
        return false;
    }
    if prog.op(rhs_bfe) != Opcode::BitFieldUExtract
        || prog.arg(rhs_bfe, 1) != Value::U32(16)
        || prog.arg(rhs_bfe, 2) != Value::U32(16)
    {
        return false;
    }
    let Some(lhs_mul) = inst_of(prog.arg(lhs_shl, 0)) else {
        return false;
    };
    let lhs_bfe = match prog.op(lhs_mul) {
        Opcode::BitFieldUExtract => Some(lhs_mul),
        Opcode::IMul32 => inst_of(prog.arg(lhs_mul, 0)),
        _ => None,
    };
    lhs_bfe.is_some_and(|bfe| {
        prog.op(bfe) == Opcode::BitFieldUExtract
            && prog.arg(bfe, 1) == Value::U32(0)
            && prog.arg(bfe, 2) == Value::U32(16)
    })
}

/// Turns the half-word swap add Super Mario RPG uses into a bitwise or, which NVIDIA drivers
/// compile correctly. The halves never overlap, so the result is the same.
pub fn vendor_workaround(prog: &mut Program) {
    let adds: Vec<InstId> = prog
        .post_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .filter(|&id| prog.op(id) == Opcode::IAdd32 && is_byte_swap_add(prog, id))
        .collect();
    for add in adds {
        let (lhs, rhs) = (prog.arg(add, 0), prog.arg(add, 1));
        let mut ir = Emitter::before(prog, add);
        let or = ir.bitwise_or(lhs, rhs);
        prog.replace_uses_with(add, or);
    }
}

/// Whether a fragment program shuffles the bits of its X or Y fragment position, which some
/// drivers miscompile.
pub fn has_broken_pattern(prog: &Program, stage: ShaderStage) -> bool {
    if stage != ShaderStage::Fragment {
        return false;
    }
    let inst_of = |value: Value| prog.resolve(value).inst();
    prog.post_order
        .iter()
        .flat_map(|&block| prog.insts_of(block))
        .any(|id| {
            if !matches!(
                prog.op(id),
                Opcode::ShuffleIndex
                    | Opcode::ShuffleUp
                    | Opcode::ShuffleDown
                    | Opcode::ShuffleButterfly
            ) {
                return false;
            }
            let Some(bitcast) = inst_of(prog.arg(id, 0)) else {
                return false;
            };
            if prog.op(bitcast) != Opcode::BitCastU32F32 {
                return false;
            }
            let Some(attribute) = inst_of(prog.arg(bitcast, 0)) else {
                return false;
            };
            prog.op(attribute) == Opcode::GetAttribute
                && matches!(
                    prog.arg(attribute, 0),
                    Value::Attribute(Attribute::POSITION_X | Attribute::POSITION_Y)
                )
        })
}
