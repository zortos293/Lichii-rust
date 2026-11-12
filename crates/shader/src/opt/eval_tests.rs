//! A small interpreter for straight-line integer IR, so lowering tests check results numerically.

use crate::ir::{Opcode, Program, Value};

fn low_bits(count: u32) -> u32 {
    if count >= 32 {
        u32::MAX
    } else {
        (1 << count) - 1
    }
}

/// Value of `value` as raw bits, with `U32x2` pairs packed low word first.
pub fn eval(prog: &Program, value: Value) -> u64 {
    use Opcode::*;
    let Value::Inst(id) = prog.resolve(value) else {
        return prog.resolve(value).imm_u64().expect("numeric immediate");
    };
    let a = |i: usize| eval(prog, prog.arg(id, i));
    let w = |i: usize| a(i) as u32;
    let word = match prog.op(id) {
        CompositeConstructU32x2 => return u64::from(w(0)) | (u64::from(w(1)) << 32),
        CompositeExtractU32x2 => (a(0) >> (32 * w(1))) as u32,
        BitFieldUExtract => w(0).checked_shr(w(1)).unwrap_or(0) & low_bits(w(2)),
        BitFieldInsert => {
            let mask = low_bits(w(3)).checked_shl(w(2)).unwrap_or(0);
            (w(0) & !mask) | (w(1).checked_shl(w(2)).unwrap_or(0) & mask)
        }
        ShiftLeftLogical32 => w(0).checked_shl(w(1)).unwrap_or(0),
        ShiftRightLogical32 => w(0).checked_shr(w(1)).unwrap_or(0),
        ShiftRightArithmetic32 => (w(0) as i32 >> w(1).min(31)) as u32,
        BitwiseAnd32 => w(0) & w(1),
        BitwiseOr32 => w(0) | w(1),
        BitwiseXor32 => w(0) ^ w(1),
        BitwiseNot32 => !w(0),
        IAdd32 => w(0).wrapping_add(w(1)),
        ISub32 => w(0).wrapping_sub(w(1)),
        IEqual => u32::from(w(0) == w(1)),
        INotEqual => u32::from(w(0) != w(1)),
        UGreaterThan => u32::from(w(0) > w(1)),
        SGreaterThanEqual => u32::from(w(0) as i32 >= w(1) as i32),
        SelectU32 => {
            if a(0) != 0 {
                w(1)
            } else {
                w(2)
            }
        }
        GetCarryFromOp => {
            let sum = prog
                .resolve(prog.arg(id, 0))
                .inst()
                .expect("carry of an instruction");
            assert_eq!(prog.op(sum), IAdd32);
            let operand = |i: usize| eval(prog, prog.arg(sum, i)) & 0xffff_ffff;
            ((operand(0) + operand(1)) >> 32) as u32
        }
        BitCastF32U32 | BitCastU32F32 => w(0),
        op => panic!("eval does not model {}", op.name()),
    };
    u64::from(word)
}
