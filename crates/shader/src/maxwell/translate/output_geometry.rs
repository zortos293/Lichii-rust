//! OUT: geometry shader vertex emission and primitive restart.

use super::{TResult, Translator};
use crate::ir::Value;

fn out(t: &mut Translator<'_>, stream: Value) -> TResult {
    let stream = t.ir.bitwise_and(stream, Value::U32(0b11));
    if t.insn.bit(39) {
        t.ir.emit_vertex(stream);
    }
    if t.insn.bit(40) {
        t.ir.end_primitive(stream);
    }
    t.set_x(t.insn.dest_reg(), Value::U32(0));
    Ok(())
}

pub(super) fn out_cbuf(t: &mut Translator<'_>) -> TResult {
    let stream = t.get_cbuf()?;
    out(t, stream)
}

pub(super) fn out_reg(t: &mut Translator<'_>) -> TResult {
    let stream = t.get_reg20();
    out(t, stream)
}

pub(super) fn out_imm(t: &mut Translator<'_>) -> TResult {
    let stream = t.get_imm20();
    out(t, stream)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn out_reg_emit_and_cut() {
        let insn = encode(
            Opcode::OutReg,
            (1 << 40) | (1 << 39) | (4 << 20) | (2 << 8) | 1,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = BitwiseAnd32 %0, #0x3\n\
             \x20 EmitVertex %1\n\
             \x20 EndPrimitive %1\n\
             \x20 SetRegister R1, #0x0\n"
        );
    }

    #[test]
    fn out_imm_emit() {
        let insn = encode(Opcode::OutImm, (1 << 39) | (1 << 20) | 3);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n  %0 = BitwiseAnd32 #0x1, #0x3\n  EmitVertex %0\n  SetRegister R3, #0x0\n"
        );
    }

    #[test]
    fn out_cbuf_cut() {
        let insn = encode(Opcode::OutCbuf, (1 << 40) | (3 << 34) | (2 << 20) | 0xff);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n  %0 = GetCbufU32 #0x3, #0x8\n  %1 = BitwiseAnd32 %0, #0x3\n  EndPrimitive %1\n"
        );
    }
}
