//! Half precision arithmetic rewritten as single precision, for hosts without `shaderFloat16`.

use crate::ir::{Opcode, Program, Type};

/// Replaces every F16 opcode with its F32 counterpart and retypes F16 phis to match. Half
/// float conversions become identities.
pub fn lower_fp16_to_fp32(prog: &mut Program) {
    for i in 0..prog.block_order.len() {
        let mut cur = prog.block(prog.block_order[i]).first();
        while let Some(id) = cur {
            cur = prog.inst(id).next();
            let op = prog.op(id);
            if op == Opcode::Phi {
                let ty = Type::from_bits_truncate(prog.inst(id).flags);
                prog.set_flags(id, retype(ty).bits());
            } else if let Some(lowered) = replace(op) {
                prog.replace_opcode(id, lowered);
            }
        }
    }
}

fn retype(ty: Type) -> Type {
    match ty {
        t if t == Type::F16 => Type::F32,
        t if t == Type::F16X2 => Type::F32X2,
        t if t == Type::F16X3 => Type::F32X3,
        t if t == Type::F16X4 => Type::F32X4,
        t => t,
    }
}

fn replace(op: Opcode) -> Option<Opcode> {
    use Opcode::*;
    Some(match op {
        FPAbs16 => FPAbs32,
        FPAdd16 => FPAdd32,
        FPCeil16 => FPCeil32,
        FPFloor16 => FPFloor32,
        FPFma16 => FPFma32,
        FPMul16 => FPMul32,
        FPNeg16 => FPNeg32,
        FPRoundEven16 => FPRoundEven32,
        FPSaturate16 => FPSaturate32,
        FPClamp16 => FPClamp32,
        FPTrunc16 => FPTrunc32,
        SelectF16 => SelectF32,
        CompositeConstructF16x2 => CompositeConstructF32x2,
        CompositeConstructF16x3 => CompositeConstructF32x3,
        CompositeConstructF16x4 => CompositeConstructF32x4,
        CompositeExtractF16x2 => CompositeExtractF32x2,
        CompositeExtractF16x3 => CompositeExtractF32x3,
        CompositeExtractF16x4 => CompositeExtractF32x4,
        CompositeInsertF16x2 => CompositeInsertF32x2,
        CompositeInsertF16x3 => CompositeInsertF32x3,
        CompositeInsertF16x4 => CompositeInsertF32x4,
        FPOrdEqual16 => FPOrdEqual32,
        FPUnordEqual16 => FPUnordEqual32,
        FPOrdNotEqual16 => FPOrdNotEqual32,
        FPUnordNotEqual16 => FPUnordNotEqual32,
        FPOrdLessThan16 => FPOrdLessThan32,
        FPUnordLessThan16 => FPUnordLessThan32,
        FPOrdGreaterThan16 => FPOrdGreaterThan32,
        FPUnordGreaterThan16 => FPUnordGreaterThan32,
        FPOrdLessThanEqual16 => FPOrdLessThanEqual32,
        FPUnordLessThanEqual16 => FPUnordLessThanEqual32,
        FPOrdGreaterThanEqual16 => FPOrdGreaterThanEqual32,
        FPUnordGreaterThanEqual16 => FPUnordGreaterThanEqual32,
        FPIsNan16 => FPIsNan32,
        ConvertS16F16 => ConvertS16F32,
        ConvertS32F16 => ConvertS32F32,
        ConvertS64F16 => ConvertS64F32,
        ConvertU16F16 => ConvertU16F32,
        ConvertU32F16 => ConvertU32F32,
        ConvertU64F16 => ConvertU64F32,
        PackFloat2x16 => PackHalf2x16,
        UnpackFloat2x16 => UnpackHalf2x16,
        ConvertF32F16 | ConvertF16F32 => Identity,
        ConvertF16S8 => ConvertF32S8,
        ConvertF16S16 => ConvertF32S16,
        ConvertF16S32 => ConvertF32S32,
        ConvertF16S64 => ConvertF32S64,
        ConvertF16U8 => ConvertF32U8,
        ConvertF16U16 => ConvertF32U16,
        ConvertF16U32 => ConvertF32U32,
        ConvertF16U64 => ConvertF32U64,
        GlobalAtomicAddF16x2 => GlobalAtomicAddF32x2,
        StorageAtomicAddF16x2 => StorageAtomicAddF32x2,
        GlobalAtomicMinF16x2 => GlobalAtomicMinF32x2,
        StorageAtomicMinF16x2 => StorageAtomicMinF32x2,
        GlobalAtomicMaxF16x2 => GlobalAtomicMaxF32x2,
        StorageAtomicMaxF16x2 => StorageAtomicMaxF32x2,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Attribute, BlockId, Value};

    fn program() -> (Program, BlockId) {
        let mut prog = Program::new();
        let block = prog.new_block();
        prog.block_order = vec![block];
        prog.post_order = vec![block];
        (prog, block)
    }

    #[test]
    fn half_ops_become_single_ops_and_phis_follow() {
        let (mut prog, b) = program();
        let x = prog.append(
            b,
            Opcode::GetAttribute,
            &[Value::Attribute(Attribute::generic(0, 0)), Value::U32(0)],
            0,
        );
        let phi = prog.append(b, Opcode::Phi, &[], Type::F16X2.bits());
        let half = prog.append(b, Opcode::ConvertF16F32, &[Value::Inst(x)], 0);
        let sum = prog.append(
            b,
            Opcode::FPAdd16,
            &[Value::Inst(half), Value::Inst(half)],
            0,
        );
        let unrelated = prog.append(b, Opcode::FPAdd32, &[Value::Inst(x), Value::Inst(x)], 0);
        lower_fp16_to_fp32(&mut prog);
        assert_eq!(prog.type_of(Value::Inst(phi)), Type::F32X2);
        assert_eq!(prog.op(half), Opcode::Identity);
        assert_eq!(prog.type_of(Value::Inst(half)), Type::F32);
        assert_eq!(prog.op(sum), Opcode::FPAdd32);
        assert_eq!(prog.op(unrelated), Opcode::FPAdd32);
        assert_eq!(prog.op(x), Opcode::GetAttribute);
    }
}
