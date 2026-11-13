//! TENSOR_MUL_ADD: the software matrix multiply-accumulate lowered native programs emit.

use super::{TResult, Translator};
use crate::ir::{Opcode, Reg, Value};

pub(super) fn tensor_mul_add(t: &mut Translator<'_>) -> TResult {
    let i = t.insn;
    if !i.bit(48) {
        return Err(t.unsupported());
    }
    let short_depth = i.bit(49);
    let ar = i.src_a_reg();
    let br = i.src_b_reg();
    let cr = Reg(i.bits(39, 8) as u16);
    let has_c = !cr.is_zero();
    let zero = Value::U32(0);

    let a0 = t.x(ar);
    let a1 = t.x(ar.offset(1));
    let a2 = if short_depth { zero } else { t.x(ar.offset(2)) };
    let a3 = if short_depth { zero } else { t.x(ar.offset(3)) };
    let b0 = t.x(br);
    let b1 = if short_depth { zero } else { t.x(br.offset(1)) };
    let c0 = if has_c { t.x(cr) } else { zero };
    let c1 = if has_c { t.x(cr.offset(1)) } else { zero };

    let a = t.ir.composite_construct(&[a0, a1, a2, a3]);
    let b = t.ir.composite_construct(&[b0, b1]);
    let c = t.ir.composite_construct(&[c0, c1]);
    let depth = if short_depth { 8 } else { 16 };
    let result = t.ir.inst_flags(Opcode::TensorMulAdd, depth, &[a, b, c]);
    let dest = i.dest_reg();
    let d0 = t.ir.composite_extract(result, 0);
    t.set_x(dest, d0);
    let d1 = t.ir.composite_extract(result, 1);
    t.set_x(dest.offset(1), d1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::floating_point_add::tests::{check, rejects};
    use super::super::tests::encode;
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn tensor_mul_add_full_depth() {
        check(
            encode(
                Opcode::TensorMulAdd,
                2 | 4 << 8 | 8 << 20 | 10 << 39 | 1 << 48,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = GetRegister R6",
                "%3 = GetRegister R7",
                "%4 = GetRegister R8",
                "%5 = GetRegister R9",
                "%6 = GetRegister R10",
                "%7 = GetRegister R11",
                "%8 = CompositeConstructU32x4 %0, %1, %2, %3",
                "%9 = CompositeConstructU32x2 %4, %5",
                "%10 = CompositeConstructU32x2 %6, %7",
                "%11 = TensorMulAdd %8, %9, %10 (flags 0x10)",
                "%12 = CompositeExtractU32x2 %11, #0x0",
                "SetRegister R2, %12",
                "%13 = CompositeExtractU32x2 %11, #0x1",
                "SetRegister R3, %13",
            ],
        );
    }

    #[test]
    fn tensor_mul_add_short_depth_without_c() {
        check(
            encode(
                Opcode::TensorMulAdd,
                2 | 4 << 8 | 8 << 20 | 0xff << 39 | 1 << 48 | 1 << 49,
            ),
            &[
                "%0 = GetRegister R4",
                "%1 = GetRegister R5",
                "%2 = GetRegister R8",
                "%3 = CompositeConstructU32x4 %0, %1, #0x0, #0x0",
                "%4 = CompositeConstructU32x2 %2, #0x0",
                "%5 = CompositeConstructU32x2 #0x0, #0x0",
                "%6 = TensorMulAdd %3, %4, %5 (flags 0x8)",
                "%7 = CompositeExtractU32x2 %6, #0x0",
                "SetRegister R2, %7",
                "%8 = CompositeExtractU32x2 %6, #0x1",
                "SetRegister R3, %8",
            ],
        );
    }

    #[test]
    fn tensor_mul_add_rejects_other_encodings() {
        rejects(encode(Opcode::TensorMulAdd, 2 | 4 << 8 | 8 << 20));
    }
}
