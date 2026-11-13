//! VOTE: warp-wide predicate votes and ballots.

use super::{TResult, Translator};
use crate::ir::{Opcode, Pred};

pub(super) fn vote(t: &mut Translator<'_>) -> TResult {
    let op = match t.insn.bits(48, 2) {
        0 => Opcode::VoteAll,
        1 => Opcode::VoteAny,
        2 => Opcode::VoteEqual,
        _ => return Err(t.unsupported()),
    };
    let pred =
        t.ir.get_pred(Pred::p(t.insn.bits(39, 3) as u8), t.insn.bit(42));
    let result = t.ir.inst(op, &[pred]);
    t.ir.set_pred(Pred::p(t.insn.bits(45, 3) as u8), result);
    let ballot = t.ir.inst(Opcode::SubgroupBallot, &[pred]);
    t.set_x(t.insn.dest_reg(), ballot);
    Ok(())
}

/// Stubbed like the C++ frontend: it emits nothing.
pub(super) fn vote_vtg(_: &mut Translator<'_>) -> TResult {
    crate::fallback::note("VOTE", "vtg_stubbed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::{encode, translate};
    use crate::maxwell::opcode::Opcode;

    #[test]
    fn vote_any_negated_pred() {
        let insn = encode(
            Opcode::Vote,
            (1 << 48) | (2 << 45) | (1 << 42) | (3 << 39) | 4,
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetPred P3\n\
             \x20 %1 = LogicalNot %0\n\
             \x20 %2 = VoteAny %1\n\
             \x20 SetPred P2, %2\n\
             \x20 %3 = SubgroupBallot %1\n\
             \x20 SetRegister R4, %3\n"
        );
    }

    #[test]
    fn vote_all_pt_into_rz() {
        let insn = encode(Opcode::Vote, (7 << 45) | (7 << 39) | 0xff);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n  %0 = VoteAll #true\n  %1 = SubgroupBallot #true\n"
        );
    }

    #[test]
    fn vote_reserved_op_is_unsupported() {
        assert!(translate(&[encode(Opcode::Vote, 3 << 48)]).is_err());
    }

    #[test]
    fn vote_vtg_emits_nothing() {
        assert_eq!(translate(&[encode(Opcode::VoteVtg, 0)]).unwrap(), "{0}:\n");
    }
}
