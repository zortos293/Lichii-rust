//! BRX and JMX. Control flow analysis resolves the targets; translation only validates the
//! encoding.

use super::{TResult, Translator};

fn check(t: &Translator<'_>) -> TResult {
    if t.insn.bit(5) || t.insn.bit(6) {
        return Err(t.unsupported());
    }
    Ok(())
}

pub(super) fn brx(t: &mut Translator<'_>) -> TResult {
    check(t)
}

pub(super) fn jmx(t: &mut Translator<'_>) -> TResult {
    check(t)
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn indirect_branches_emit_nothing() {
        assert_eq!(translate(&[encode(Opcode::Brx, 0xf)]).unwrap(), "{0}:\n");
        assert_eq!(translate(&[encode(Opcode::Jmx, 0xf)]).unwrap(), "{0}:\n");
    }

    #[test]
    fn cbuf_and_lmt_are_unsupported() {
        for op in [Opcode::Brx, Opcode::Jmx] {
            assert!(translate(&[encode(op, 1 << 5)]).is_err());
            assert!(translate(&[encode(op, 1 << 6)]).is_err());
        }
    }
}
