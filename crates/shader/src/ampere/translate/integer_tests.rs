//! Tests for the integer translators: IR dumps of hand-encoded instructions, and numeric checks
//! that run the translated IR through a small warp interpreter against a reference model.

use rustc_hash::FxHashMap;

use super::super::tests::{TestEnv, encode, translate};
use super::super::translate_block;
use crate::ampere::decode::Form;
use crate::ampere::instruction::Instruction;
use crate::ampere::opcode::Opcode;
use crate::ir::{InstId, Opcode as Ir, Pred, Program, Reg, Value};
use crate::{ShaderStage, TranslateError};

pub(crate) const RZ: u128 = 255;
pub(crate) const URZ: u128 = 63;
pub(crate) const PT: u128 = 7;

pub(crate) fn dst(reg: u128) -> u128 {
    reg << 16
}

pub(crate) fn a(reg: u128) -> u128 {
    reg << 24
}

/// The operand at bit 32: a register, uniform register or immediate.
pub(crate) fn b(value: u128) -> u128 {
    value << 32
}

/// The register at bit 64.
pub(crate) fn c(reg: u128) -> u128 {
    reg << 64
}

pub(crate) fn bit(n: u32) -> u128 {
    1 << n
}

/// The predicate field at `offset` naming predicate `index`.
pub(crate) fn pred(offset: u32, index: u128) -> u128 {
    index << offset
}

/// Predicate outputs at 81 and 84 discarded.
const NO_OUTPUTS: u128 = (PT << 81) | (PT << 84);

/// `!PT` in the predicate source at 87 (negated by 90) and at 77 (negated by 80).
const NO_CARRY_INS: u128 = (0xf << 87) | (0xf << 77);

/// Lanes in a warp.
pub(crate) const WARP: usize = 32;

/// Register and predicate state of one lane.
#[derive(Clone, Debug, Default)]
pub(crate) struct State {
    pub regs: FxHashMap<Reg, u32>,
    pub preds: FxHashMap<Pred, bool>,
}

impl State {
    pub(crate) fn with_regs(regs: &[(Reg, u32)]) -> State {
        State {
            regs: regs.iter().copied().collect(),
            preds: FxHashMap::default(),
        }
    }

    pub(crate) fn reg(&self, reg: Reg) -> u32 {
        self.regs.get(&reg).copied().unwrap_or(0)
    }

    pub(crate) fn pred(&self, pred: Pred) -> bool {
        pred.is_true() || self.preds.get(&pred).copied().unwrap_or(false)
    }
}

fn low_bits(count: u32) -> u32 {
    if count >= 32 {
        u32::MAX
    } else {
        (1 << count) - 1
    }
}

type Lanes = [u64; WARP];

fn operand(prog: &Program, values: &FxHashMap<InstId, Lanes>, value: Value, lane: usize) -> u64 {
    match prog.resolve(value) {
        Value::Inst(id) => values[&id][lane],
        imm => imm.imm_u64().expect("numeric immediate"),
    }
}

/// The lane a shuffle reads and whether it is in bounds, as the SPIR-V backend lowers
/// ShuffleIndex and Maxwell's SHFL defines the relative modes.
fn shuffle_source(op: Ir, lane: usize, index: u32, clamp: u32, segment_mask: u32) -> (usize, bool) {
    let lane = lane as u32;
    let min_lane = lane & segment_mask;
    let max_lane = min_lane | (clamp & !segment_mask);
    let index = index & 31;
    let (source, in_bounds) = match op {
        Ir::ShuffleIndex => {
            let source = (index & !segment_mask) | min_lane;
            (source, source <= max_lane)
        }
        Ir::ShuffleUp => {
            let source = lane.wrapping_sub(index);
            (source, source as i32 >= min_lane as i32)
        }
        Ir::ShuffleDown => {
            let source = lane + index;
            (source, source <= max_lane)
        }
        _ => {
            let source = lane ^ index;
            (source, source <= max_lane)
        }
    };
    if in_bounds {
        (source as usize, true)
    } else {
        (lane as usize, false)
    }
}

/// Translates `code` as one block of a `stage` program and runs it in lockstep on the lanes
/// set in `active`, each with its own state, with `cbuf` as constant buffer 0.
pub(crate) fn run_warp(
    stage: ShaderStage,
    code: &[Instruction],
    cbuf: &[u32],
    lanes: &mut [State],
    active: u32,
) {
    use Ir::*;
    let mut env = TestEnv::new(code);
    env.stage = stage;
    env.cbuf = cbuf.to_vec();
    let mut prog = Program::new();
    let block = prog.new_block();
    translate_block(&mut prog, block, &env, 0, env.end()).expect("translates");
    let mut values: FxHashMap<InstId, Lanes> = FxHashMap::default();
    let mut in_bounds: FxHashMap<InstId, [bool; WARP]> = FxHashMap::default();
    let is_active = |lane: usize| active >> lane & 1 != 0;
    for id in prog.insts_of(block) {
        let op = prog.op(id);
        let v = |i: usize, lane: usize| operand(&prog, &values, prog.arg(id, i), lane);
        let ballot = |i: usize| -> u32 {
            (0..WARP)
                .filter(|&lane| is_active(lane) && v(i, lane) != 0)
                .fold(0, |mask, lane| mask | 1 << lane)
        };
        let mut result = [0u64; WARP];
        match op {
            SetRegister | SetPred => {
                for (lane, state) in lanes.iter_mut().enumerate().filter(|&(l, _)| is_active(l)) {
                    match prog.arg(id, 0) {
                        Value::Reg(reg) => {
                            state.regs.insert(reg, v(1, lane) as u32);
                        }
                        Value::Pred(pred) => {
                            state.preds.insert(pred, v(1, lane) != 0);
                        }
                        other => panic!("cannot write {other:?}"),
                    }
                }
                continue;
            }
            SubgroupBallot => result = [u64::from(ballot(0)); WARP],
            VoteAll | VoteAny | VoteEqual => {
                let set = ballot(0);
                let vote = match op {
                    VoteAll => set == active,
                    VoteAny => set != 0,
                    _ => set == active || set == 0,
                };
                result = [u64::from(vote); WARP];
            }
            ShuffleIndex | ShuffleUp | ShuffleDown | ShuffleButterfly => {
                let mut bounds = [false; WARP];
                for lane in (0..WARP).filter(|&l| is_active(l)) {
                    let (source, inside) = shuffle_source(
                        op,
                        lane,
                        v(1, lane) as u32,
                        v(2, lane) as u32,
                        v(3, lane) as u32,
                    );
                    result[lane] = v(0, source);
                    bounds[lane] = inside;
                }
                in_bounds.insert(id, bounds);
            }
            QuadBroadcast | QuadSwap => {
                for lane in (0..WARP).filter(|&l| is_active(l)) {
                    let k = v(1, lane) as usize;
                    let source = if op == QuadBroadcast {
                        (lane & !3) | k
                    } else {
                        lane ^ (k + 1)
                    };
                    result[lane] = v(0, source);
                }
            }
            GetInBoundsFromOp => {
                let Value::Inst(shuffle) = prog.resolve(prog.arg(id, 0)) else {
                    panic!("in-bounds flag of an instruction")
                };
                let bounds = in_bounds[&shuffle];
                for lane in 0..WARP {
                    result[lane] = u64::from(bounds[lane]);
                }
            }
            _ => {
                for (lane, state) in lanes.iter().enumerate().filter(|&(l, _)| is_active(l)) {
                    let carry = |i: usize| {
                        let Value::Inst(sum) = prog.resolve(prog.arg(id, i)) else {
                            panic!("carry of an instruction")
                        };
                        assert_eq!(prog.op(sum), IAdd32);
                        let term = |j: usize| {
                            operand(&prog, &values, prog.arg(sum, j), lane) & 0xffff_ffff
                        };
                        (term(0) + term(1)) >> 32
                    };
                    result[lane] = scalar(&prog, id, |i| v(i, lane), carry, state, cbuf, lane);
                }
            }
        }
        values.insert(id, result);
    }
}

/// One lane's result of a non-warp instruction.
fn scalar(
    prog: &Program,
    id: InstId,
    v: impl Fn(usize) -> u64,
    carry: impl Fn(usize) -> u64,
    state: &State,
    cbuf: &[u32],
    lane: usize,
) -> u64 {
    use Ir::*;
    let w = |i: usize| v(i) as u32;
    let s = |i: usize| v(i) as u32 as i32;
    match prog.op(id) {
        GetRegister => {
            let Value::Reg(reg) = prog.arg(id, 0) else {
                panic!("register operand")
            };
            u64::from(state.reg(reg))
        }
        GetPred => {
            let Value::Pred(pred) = prog.arg(id, 0) else {
                panic!("predicate operand")
            };
            u64::from(state.pred(pred))
        }
        LaneId => lane as u64,
        GetCbufU32 => u64::from(cbuf[(w(1) / 4) as usize]),
        GetCbufU32x2 => {
            let index = (w(1) / 4) as usize;
            u64::from(cbuf[index]) | (u64::from(cbuf[index + 1]) << 32)
        }
        IAdd32 => u64::from(w(0).wrapping_add(w(1))),
        GetCarryFromOp => carry(0),
        ISub32 => u64::from(w(0).wrapping_sub(w(1))),
        IMul32 => u64::from(w(0).wrapping_mul(w(1))),
        INeg32 => u64::from(w(0).wrapping_neg()),
        IAbs32 => u64::from(s(0).wrapping_abs() as u32),
        IAdd64 => v(0).wrapping_add(v(1)),
        ISub64 => v(0).wrapping_sub(v(1)),
        ShiftLeftLogical32 => u64::from(w(0).checked_shl(w(1)).unwrap_or(0)),
        ShiftRightLogical32 => u64::from(w(0).checked_shr(w(1)).unwrap_or(0)),
        ShiftRightArithmetic32 => u64::from((s(0) >> w(1).min(31)) as u32),
        ShiftLeftLogical64 => v(0).checked_shl(w(1)).unwrap_or(0),
        ShiftRightLogical64 => v(0).checked_shr(w(1)).unwrap_or(0),
        ShiftRightArithmetic64 => ((v(0) as i64) >> w(1).min(63)) as u64,
        BitwiseAnd32 => u64::from(w(0) & w(1)),
        BitwiseOr32 => u64::from(w(0) | w(1)),
        BitwiseXor32 => u64::from(w(0) ^ w(1)),
        BitwiseNot32 => u64::from(!w(0)),
        BitFieldUExtract => u64::from(w(0).checked_shr(w(1)).unwrap_or(0) & low_bits(w(2))),
        BitFieldSExtract => {
            let count = w(2);
            if count == 0 {
                0
            } else {
                let field = w(0).checked_shr(w(1)).unwrap_or(0) << (32 - count);
                u64::from(((field as i32) >> (32 - count)) as u32)
            }
        }
        BitFieldInsert => {
            let mask = low_bits(w(3)).checked_shl(w(2)).unwrap_or(0);
            u64::from((w(0) & !mask) | (w(1).checked_shl(w(2)).unwrap_or(0) & mask))
        }
        BitReverse32 => u64::from(w(0).reverse_bits()),
        BitCount32 => u64::from(w(0).count_ones()),
        FindUMsb32 => u64::from(31u32.wrapping_sub(w(0).leading_zeros())),
        FindSMsb32 => {
            let x = if s(0) < 0 { !w(0) } else { w(0) };
            u64::from(31u32.wrapping_sub(x.leading_zeros()))
        }
        SMin32 => u64::from(s(0).min(s(1)) as u32),
        SMax32 => u64::from(s(0).max(s(1)) as u32),
        UMin32 => u64::from(w(0).min(w(1))),
        UMax32 => u64::from(w(0).max(w(1))),
        IEqual => u64::from(w(0) == w(1)),
        INotEqual => u64::from(w(0) != w(1)),
        SLessThan => u64::from(s(0) < s(1)),
        ULessThan => u64::from(w(0) < w(1)),
        SLessThanEqual => u64::from(s(0) <= s(1)),
        ULessThanEqual => u64::from(w(0) <= w(1)),
        SGreaterThan => u64::from(s(0) > s(1)),
        UGreaterThan => u64::from(w(0) > w(1)),
        SGreaterThanEqual => u64::from(s(0) >= s(1)),
        UGreaterThanEqual => u64::from(w(0) >= w(1)),
        LogicalOr => u64::from(v(0) != 0 || v(1) != 0),
        LogicalAnd => u64::from(v(0) != 0 && v(1) != 0),
        LogicalXor => u64::from((v(0) != 0) != (v(1) != 0)),
        LogicalNot => u64::from(v(0) == 0),
        SelectU1 | SelectU32 | SelectU64 => {
            if v(0) != 0 {
                v(1)
            } else {
                v(2)
            }
        }
        CompositeConstructU32x2 => u64::from(w(0)) | (u64::from(w(1)) << 32),
        CompositeExtractU32x2 => (v(0) >> (32 * w(1))) & 0xffff_ffff,
        PackUint2x32 | UnpackUint2x32 => v(0),
        ConvertU32U64 => v(0) & 0xffff_ffff,
        op => panic!("the interpreter does not model {}", op.name()),
    }
}

/// Runs `code` on one lane.
fn run(code: &[Instruction], cbuf: &[u32], state: &mut State) {
    let mut lanes = vec![State::default(); WARP];
    lanes[0] = std::mem::take(state);
    run_warp(ShaderStage::Compute, code, cbuf, &mut lanes, 1);
    *state = std::mem::take(&mut lanes[0]);
}

/// Runs one instruction on registers `regs` and returns the final state.
fn exec(inst: Instruction, regs: &[(Reg, u32)]) -> State {
    let mut state = State::with_regs(regs);
    run(&[inst], &[], &mut state);
    state
}

/// A deterministic stream of test values that favours the edges of the 32-bit range.
pub(crate) struct Values(u64);

impl Values {
    pub(crate) fn new() -> Values {
        Values(0x2545_f491_4f6c_dd1d)
    }

    pub(crate) fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        match self.0 % 8 {
            0 => [0, 1, u32::MAX, 0x8000_0000, 0x7fff_ffff, 0xffff][(self.0 >> 8) as usize % 6],
            _ => (self.0 >> 16) as u32,
        }
    }
}

pub(crate) fn r(index: u8) -> Reg {
    Reg::r(index)
}

pub(crate) fn p(index: u8) -> Pred {
    Pred::p(index)
}

pub(crate) fn is_unsupported(result: Result<String, TranslateError>) -> bool {
    matches!(result, Err(TranslateError::Unsupported { .. }))
}

// IADD3

#[test]
fn iadd3_adds_three_registers() {
    let inst = encode(
        Opcode::Iadd3,
        Form::RegReg,
        dst(0) | a(1) | b(2) | c(3) | NO_OUTPUTS | NO_CARRY_INS,
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetRegister R3
  %3 = IAdd32 %0, %1
  %4 = IAdd32 %3, %2
  SetRegister R0, %4
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn iadd3_carry_out_and_negation() {
    let inst = encode(
        Opcode::Iadd3,
        Form::RegReg,
        dst(4) | a(0) | b(2) | c(RZ) | bit(63) | pred(81, 0) | pred(84, PT) | NO_CARRY_INS,
    );
    let expected = "\
{0}:
  %0 = GetRegister R0
  %1 = GetRegister R2
  %2 = BitwiseNot32 %1
  %3 = IAdd32 %0, %2
  %4 = GetCarryFromOp %3
  %5 = IAdd32 %3, #0x1
  %6 = GetCarryFromOp %5
  SetRegister R4, %5
  %7 = LogicalOr %4, %6
  SetPred P0, %7
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

/// NAK's IADD3 fold: a negated source adds `!x + 1`, and the carry outputs report a sum of at
/// least 2^32 and 2^33.
#[test]
fn iadd3_matches_the_reference_sum_and_carries() {
    let mut values = Values::new();
    for negations in 0..8u32 {
        for _ in 0..64 {
            let srcs = [values.next(), values.next(), values.next()];
            let fields = dst(10)
                | a(1)
                | b(2)
                | c(3)
                | (u128::from(negations & 1) << 72)
                | (u128::from(negations >> 1 & 1) << 63)
                | (u128::from(negations >> 2 & 1) << 75)
                | pred(81, 0)
                | pred(84, 1)
                | NO_CARRY_INS;
            let inst = encode(Opcode::Iadd3, Form::RegReg, fields);
            let state = exec(inst, &[(r(1), srcs[0]), (r(2), srcs[1]), (r(3), srcs[2])]);
            let sum: u64 = (0..3)
                .map(|i| {
                    if negations >> i & 1 != 0 {
                        u64::from(!srcs[i]) + 1
                    } else {
                        u64::from(srcs[i])
                    }
                })
                .sum();
            assert_eq!(state.reg(r(10)), sum as u32, "{srcs:x?} {negations}");
            assert_eq!(state.pred(p(0)), sum >= 1 << 32, "{srcs:x?} {negations}");
            assert_eq!(state.pred(p(1)), sum >= 2 << 32, "{srcs:x?} {negations}");
        }
    }
}

#[test]
fn iadd3_x_adds_both_carries_and_inverts() {
    let mut values = Values::new();
    for case in 0..256u32 {
        let srcs = [values.next(), values.next(), values.next()];
        let (carry0, carry1) = (case & 1 != 0, case & 2 != 0);
        let negations = (case >> 2) & 7;
        let fields = dst(10)
            | a(1)
            | b(2)
            | c(3)
            | bit(74)
            | (u128::from(negations & 1) << 72)
            | (u128::from(negations >> 1 & 1) << 63)
            | (u128::from(negations >> 2 & 1) << 75)
            | pred(87, 2)
            | pred(77, 3)
            | pred(81, 0)
            | pred(84, 1);
        let inst = encode(Opcode::Iadd3, Form::RegReg, fields);
        let mut state = State::with_regs(&[(r(1), srcs[0]), (r(2), srcs[1]), (r(3), srcs[2])]);
        state.preds.insert(p(2), carry0);
        state.preds.insert(p(3), carry1);
        run(&[inst], &[], &mut state);
        let sum: u64 = (0..3)
            .map(|i| {
                u64::from(if negations >> i & 1 != 0 {
                    !srcs[i]
                } else {
                    srcs[i]
                })
            })
            .sum::<u64>()
            + u64::from(carry0)
            + u64::from(carry1);
        assert_eq!(state.reg(r(10)), sum as u32, "{srcs:x?} {case}");
        assert_eq!(state.pred(p(0)), sum >= 1 << 32, "{srcs:x?} {case}");
        assert_eq!(state.pred(p(1)), sum >= 2 << 32, "{srcs:x?} {case}");
    }
}

/// The 64-bit add and subtract the compiler emits: IADD3 with a carry out, then IADD3.X.
#[test]
fn iadd3_pairs_add_and_subtract_64_bit_values() {
    let mut values = Values::new();
    for subtract in [false, true] {
        let low = encode(
            Opcode::Iadd3,
            Form::RegReg,
            dst(4)
                | a(0)
                | b(2)
                | c(RZ)
                | (u128::from(subtract) << 63)
                | pred(81, 0)
                | pred(84, PT)
                | NO_CARRY_INS,
        );
        let high = encode(
            Opcode::Iadd3,
            Form::RegReg,
            dst(5)
                | a(1)
                | b(3)
                | c(RZ)
                | bit(74)
                | (u128::from(subtract) << 63)
                | pred(87, 0)
                | (0xf << 77)
                | NO_OUTPUTS,
        );
        for _ in 0..256 {
            let x = u64::from(values.next()) | (u64::from(values.next()) << 32);
            let y = u64::from(values.next()) | (u64::from(values.next()) << 32);
            let mut state = State::with_regs(&[
                (r(0), x as u32),
                (r(1), (x >> 32) as u32),
                (r(2), y as u32),
                (r(3), (y >> 32) as u32),
            ]);
            run(&[low, high], &[], &mut state);
            let result = u64::from(state.reg(r(4))) | (u64::from(state.reg(r(5))) << 32);
            let expected = if subtract {
                x.wrapping_sub(y)
            } else {
                x.wrapping_add(y)
            };
            assert_eq!(result, expected, "{x:x} {y:x} {subtract}");
        }
    }
}

#[test]
fn iadd3_immediate_and_constant_buffer_forms() {
    let mut values = Values::new();
    for _ in 0..32 {
        let (x, y, k) = (values.next(), values.next(), values.next());
        let inst = encode(
            Opcode::Iadd3,
            Form::ImmReg,
            dst(0) | a(1) | b(u128::from(k)) | c(2) | NO_OUTPUTS | NO_CARRY_INS,
        );
        let state = exec(inst, &[(r(1), x), (r(2), y)]);
        assert_eq!(state.reg(r(0)), x.wrapping_add(k).wrapping_add(y));
        let inst = encode(
            Opcode::Iadd3,
            Form::RegCbuf,
            dst(0) | a(1) | c(2) | (8 << 38) | bit(63) | NO_OUTPUTS | NO_CARRY_INS,
        );
        let mut state = State::with_regs(&[(r(1), x), (r(2), y)]);
        run(&[inst], &[0, 0, k], &mut state);
        assert_eq!(state.reg(r(0)), x.wrapping_add(y).wrapping_sub(k));
    }
}

#[test]
fn uiadd3_writes_uniform_registers_and_predicates() {
    let inst = encode(
        Opcode::Uiadd3,
        Form::RegReg,
        dst(4) | a(6) | b(8) | c(URZ) | bit(91) | pred(81, 1) | pred(84, PT) | NO_CARRY_INS,
    );
    let expected = "\
{0}:
  %0 = GetRegister UR6
  %1 = GetRegister UR8
  %2 = IAdd32 %0, %1
  %3 = GetCarryFromOp %2
  SetRegister UR4, %2
  SetPred UP1, %3
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn iadd3_x_reads_uniform_carry_predicates() {
    let inst = encode(
        Opcode::Uiadd3,
        Form::RegReg,
        dst(5) | a(7) | b(9) | c(URZ) | bit(91) | bit(74) | pred(87, 1) | (0xf << 77) | NO_OUTPUTS,
    );
    let expected = "\
{0}:
  %0 = GetRegister UR7
  %1 = GetRegister UR9
  %2 = IAdd32 %0, %1
  %3 = GetPred UP1
  %4 = SelectU32 %3, #0x1, #0x0
  %5 = IAdd32 %2, %4
  SetRegister UR5, %5
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

// IMAD

#[test]
fn imad_multiplies_and_adds() {
    let inst = encode(
        Opcode::Imad,
        Form::RegReg,
        dst(0) | a(1) | b(2) | c(3) | NO_OUTPUTS,
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetRegister R3
  %3 = IMul32 %0, %1
  %4 = IAdd32 %3, %2
  SetRegister R0, %4
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn imad_mov_copies_the_addend() {
    let inst = encode(
        Opcode::Imad,
        Form::RegReg,
        dst(0) | a(RZ) | b(RZ) | c(5) | NO_OUTPUTS,
    );
    let expected = "\
{0}:
  %0 = GetRegister R5
  SetRegister R0, %0
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn imad_mov_of_an_immediate() {
    let inst = encode(
        Opcode::Imad,
        Form::RegImm,
        dst(0) | a(RZ) | b(0x3f80_0000) | c(RZ) | NO_OUTPUTS,
    );
    let expected = "\
{0}:
  SetRegister R0, #0x3f800000
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn imad_shl_by_a_power_of_two_is_a_shift() {
    let inst = encode(
        Opcode::Imad,
        Form::ImmReg,
        dst(0) | a(1) | b(16) | c(RZ) | NO_OUTPUTS,
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = ShiftLeftLogical32 %0, #0x4
  SetRegister R0, %1
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn uimad_uses_uniform_registers() {
    let inst = encode(
        Opcode::Uimad,
        Form::RegReg,
        dst(4) | a(5) | b(6) | c(7) | bit(91) | NO_OUTPUTS,
    );
    let expected = "\
{0}:
  %0 = GetRegister UR5
  %1 = GetRegister UR6
  %2 = GetRegister UR7
  %3 = IMul32 %0, %1
  %4 = IAdd32 %3, %2
  SetRegister UR4, %4
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn imad_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..128u32 {
        let (x, y, z) = (values.next(), values.next(), values.next());
        let (negate_b, negate_c, extended, carry) =
            (case & 1 != 0, case & 2 != 0, case & 4 != 0, case & 8 != 0);
        let fields = dst(0)
            | a(1)
            | b(2)
            | c(3)
            | (u128::from(negate_b) << 63)
            | (u128::from(negate_c) << 75)
            | (u128::from(extended) << 74)
            | pred(87, 1)
            | NO_OUTPUTS;
        let inst = encode(Opcode::Imad, Form::RegReg, fields);
        let mut state = State::with_regs(&[(r(1), x), (r(2), y), (r(3), z)]);
        state.preds.insert(p(1), carry);
        run(&[inst], &[], &mut state);
        let y = if negate_b { y.wrapping_neg() } else { y };
        let z = match (negate_c, extended) {
            (false, _) => z,
            (true, false) => z.wrapping_neg(),
            (true, true) => !z,
        };
        let expected = x
            .wrapping_mul(y)
            .wrapping_add(z)
            .wrapping_add(u32::from(extended && carry));
        assert_eq!(state.reg(r(0)), expected, "{case}");
    }
}

// IMAD.WIDE and IMAD.HI

#[test]
fn imad_wide_and_hi_match_the_reference() {
    let mut values = Values::new();
    for case in 0..512u32 {
        let (signed, immediate, wide) = (case & 1 != 0, case & 2 != 0, case & 4 != 0);
        let (x, y) = (values.next(), values.next());
        let addend = u64::from(values.next()) | (u64::from(values.next()) << 32);
        let (op, out) = if wide {
            (Opcode::ImadWide, 4)
        } else {
            (Opcode::ImadHi, 6)
        };
        let (form, multiplier) = if immediate {
            (Form::ImmReg, u128::from(y))
        } else {
            (Form::RegReg, 1)
        };
        let fields = dst(out)
            | a(0)
            | b(multiplier)
            | c(2)
            | (u128::from(signed) << 73)
            | pred(81, 0)
            | pred(84, PT)
            | NO_CARRY_INS;
        let inst = encode(op, form, fields);
        let state = exec(
            inst,
            &[
                (r(0), x),
                (r(1), y),
                (r(2), addend as u32),
                (r(3), (addend >> 32) as u32),
            ],
        );
        let product = if signed {
            (i64::from(x as i32) * i64::from(y as i32)) as u64
        } else {
            u64::from(x) * u64::from(y)
        };
        let sum = u128::from(product) + u128::from(addend);
        let context = format!("{x:x} {y:x} {addend:x} {case}");
        if wide {
            assert_eq!(state.reg(r(4)), sum as u32, "{context}");
            assert_eq!(state.reg(r(5)), (sum >> 32) as u32, "{context}");
        } else {
            assert_eq!(state.reg(r(6)), (sum >> 32) as u32, "{context}");
        }
        assert_eq!(state.pred(p(0)), sum >> 64 != 0, "{context}");
    }
}

#[test]
fn imad_wide_adds_a_constant_buffer_pair() {
    let inst = encode(
        Opcode::ImadWide,
        Form::RegCbuf,
        dst(4) | a(0) | c(1) | (8 << 38) | NO_OUTPUTS | NO_CARRY_INS,
    );
    let mut state = State::with_regs(&[(r(0), 0xdead_beef), (r(1), 0x1234_5678)]);
    run(&[inst], &[0, 0, 0xffff_ffff, 7], &mut state);
    let expected = 0xdead_beefu64 * 0x1234_5678 + 0x7_ffff_ffff;
    assert_eq!(state.reg(r(4)), expected as u32);
    assert_eq!(state.reg(r(5)), (expected >> 32) as u32);
}

#[test]
fn imad_wide_by_a_power_of_two_folds_the_long_multiplication() {
    let inst = encode(
        Opcode::ImadWide,
        Form::ImmReg,
        dst(2) | a(4) | b(4) | c(6) | NO_OUTPUTS | NO_CARRY_INS,
    );
    let expected = "\
{0}:
  %0 = GetRegister R4
  %1 = GetRegister R6
  %2 = GetRegister R7
  %3 = ShiftLeftLogical32 %0, #0x2
  %4 = BitFieldUExtract %0, #0x0, #0x10
  %5 = ShiftRightLogical32 %0, #0x10
  %6 = ShiftLeftLogical32 %4, #0x2
  %7 = ShiftLeftLogical32 %5, #0x2
  %8 = ShiftRightLogical32 %6, #0x10
  %9 = BitFieldUExtract %7, #0x0, #0x10
  %10 = IAdd32 %8, %9
  %11 = ShiftRightLogical32 %7, #0x10
  %12 = ShiftRightLogical32 %10, #0x10
  %13 = IAdd32 %11, %12
  %14 = IAdd32 %3, %1
  %15 = GetCarryFromOp %14
  %16 = IAdd32 %13, %2
  %17 = SelectU32 %15, #0x1, #0x0
  %18 = IAdd32 %16, %17
  SetRegister R2, %14
  SetRegister R3, %18
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn uimad_wide_writes_a_uniform_pair() {
    let inst = encode(
        Opcode::UimadWide,
        Form::RegReg,
        dst(4) | a(6) | b(7) | c(URZ) | bit(91) | NO_OUTPUTS | NO_CARRY_INS,
    );
    let expected = "\
{0}:
  %0 = GetRegister UR6
  %1 = GetRegister UR7
  %2 = IMul32 %0, %1
  %3 = BitFieldUExtract %0, #0x0, #0x10
  %4 = ShiftRightLogical32 %0, #0x10
  %5 = BitFieldUExtract %1, #0x0, #0x10
  %6 = ShiftRightLogical32 %1, #0x10
  %7 = IMul32 %3, %5
  %8 = IMul32 %3, %6
  %9 = IMul32 %4, %5
  %10 = IMul32 %4, %6
  %11 = ShiftRightLogical32 %7, #0x10
  %12 = BitFieldUExtract %8, #0x0, #0x10
  %13 = IAdd32 %11, %12
  %14 = BitFieldUExtract %9, #0x0, #0x10
  %15 = IAdd32 %13, %14
  %16 = ShiftRightLogical32 %8, #0x10
  %17 = IAdd32 %10, %16
  %18 = ShiftRightLogical32 %9, #0x10
  %19 = IAdd32 %17, %18
  %20 = ShiftRightLogical32 %15, #0x10
  %21 = IAdd32 %19, %20
  SetRegister UR4, %2
  SetRegister UR5, %21
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn imad_wide_rejects_an_odd_destination() {
    let inst = encode(
        Opcode::ImadWide,
        Form::RegReg,
        dst(3) | a(0) | b(1) | c(RZ) | NO_OUTPUTS,
    );
    assert!(is_unsupported(translate(&[inst])));
}

// IDP

#[test]
fn idp_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..64u32 {
        let (a_signed, b_signed, two) = (case & 1 != 0, case & 2 != 0, case & 4 != 0);
        let (x, y, z) = (values.next(), values.next(), values.next());
        let fields = dst(0)
            | a(1)
            | b(2)
            | c(3)
            | (u128::from(a_signed) << 73)
            | (u128::from(b_signed) << 74)
            | (u128::from(two) << 76);
        let inst = encode(Opcode::Idp, Form::RegReg, fields);
        let state = exec(inst, &[(r(1), x), (r(2), y), (r(3), z)]);
        let part = |v: u32, i: u32, bits: u32, signed: bool| -> i64 {
            let field = (v >> (i * bits)) & ((1u64 << bits) - 1) as u32;
            if signed && field >> (bits - 1) != 0 {
                i64::from(field) - (1i64 << bits)
            } else {
                i64::from(field)
            }
        };
        let (parts, a_bits) = if two { (2, 16) } else { (4, 8) };
        let dot: i64 = (0..parts)
            .map(|i| part(x, i, a_bits, a_signed) * part(y, i, 8, b_signed))
            .sum();
        assert_eq!(
            state.reg(r(0)),
            (dot as u32).wrapping_add(z),
            "{x:x} {y:x} {case}"
        );
    }
}

#[test]
fn idp_4a_dump() {
    let inst = encode(
        Opcode::Idp,
        Form::RegReg,
        dst(0) | a(1) | b(2) | c(RZ) | bit(73) | bit(74),
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = BitFieldSExtract %0, #0x0, #0x8
  %3 = BitFieldSExtract %1, #0x0, #0x8
  %4 = IMul32 %2, %3
  %5 = BitFieldSExtract %0, #0x8, #0x8
  %6 = BitFieldSExtract %1, #0x8, #0x8
  %7 = IMul32 %5, %6
  %8 = IAdd32 %4, %7
  %9 = BitFieldSExtract %0, #0x10, #0x8
  %10 = BitFieldSExtract %1, #0x10, #0x8
  %11 = IMul32 %9, %10
  %12 = IAdd32 %8, %11
  %13 = BitFieldSExtract %0, #0x18, #0x8
  %14 = BitFieldSExtract %1, #0x18, #0x8
  %15 = IMul32 %13, %14
  %16 = IAdd32 %12, %15
  SetRegister R0, %16
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

// IABS, IMNMX

#[test]
fn iabs_of_a_register() {
    let inst = encode(Opcode::Iabs, Form::RegReg, dst(0) | b(1));
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = IAbs32 %0
  SetRegister R0, %1
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn imnmx_selects_by_predicate() {
    let inst = encode(
        Opcode::Imnmx,
        Form::RegReg,
        dst(0) | a(1) | b(2) | bit(73) | pred(87, 3),
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetPred P3
  %3 = SMin32 %0, %1
  %4 = SMax32 %0, %1
  %5 = SelectU32 %2, %3, %4
  SetRegister R0, %5
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn imnmx_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..64u32 {
        let (signed, min) = (case & 1 != 0, case & 2 != 0);
        let (x, y) = (values.next(), values.next());
        let fields = dst(0)
            | a(1)
            | b(2)
            | (u128::from(signed) << 73)
            | pred(87, PT)
            | (u128::from(!min) << 90);
        let state = exec(
            encode(Opcode::Imnmx, Form::RegReg, fields),
            &[(r(1), x), (r(2), y)],
        );
        let expected = match (signed, min) {
            (true, true) => (x as i32).min(y as i32) as u32,
            (true, false) => (x as i32).max(y as i32) as u32,
            (false, true) => x.min(y),
            (false, false) => x.max(y),
        };
        assert_eq!(state.reg(r(0)), expected, "{x:x} {y:x} {case}");
    }
}

// ISETP

#[test]
fn isetp_compares_into_a_predicate() {
    let inst = encode(
        Opcode::Isetp,
        Form::RegReg,
        a(1) | b(2) | (6 << 76) | pred(81, 0) | pred(84, PT) | pred(87, PT),
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = UGreaterThanEqual %0, %1
  SetPred P0, %2
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn uisetp_compares_uniform_registers() {
    let inst = encode(
        Opcode::Uisetp,
        Form::ImmReg,
        a(4) | b(16)
            | bit(73)
            | (1 << 76)
            | (1 << 74)
            | pred(81, 2)
            | pred(84, 3)
            | pred(87, 1)
            | bit(91),
    );
    let expected = "\
{0}:
  %0 = GetRegister UR4
  %1 = SLessThan %0, #0x10
  %2 = GetPred UP1
  %3 = LogicalOr %1, %2
  SetPred UP2, %3
  %4 = LogicalNot %1
  %5 = LogicalOr %4, %2
  SetPred UP3, %5
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

/// ISETP then ISETP.EX compare 64-bit values: the high words decide unless they are equal.
#[test]
fn isetp_ex_compares_64_bit_values() {
    let mut values = Values::new();
    for op in 0..8u128 {
        for signed in [false, true] {
            let low = encode(
                Opcode::Isetp,
                Form::RegReg,
                a(0) | b(2) | (op << 76) | pred(81, 0) | pred(84, PT) | pred(87, PT),
            );
            let high = encode(
                Opcode::Isetp,
                Form::RegReg,
                a(1) | b(3)
                    | bit(72)
                    | (u128::from(signed) << 73)
                    | (op << 76)
                    | pred(68, 0)
                    | pred(81, 1)
                    | pred(84, 2)
                    | pred(87, PT),
            );
            for i in 0..128 {
                let x = u64::from(values.next()) | (u64::from(values.next()) << 32);
                let y = match i % 4 {
                    0 => x,
                    1 => (x & 0xffff_ffff_0000_0000) | u64::from(values.next()),
                    _ => u64::from(values.next()) | (u64::from(values.next()) << 32),
                };
                let mut state = State::with_regs(&[
                    (r(0), x as u32),
                    (r(1), (x >> 32) as u32),
                    (r(2), y as u32),
                    (r(3), (y >> 32) as u32),
                ]);
                run(&[low, high], &[], &mut state);
                let ordering = if signed {
                    (x as i64).cmp(&(y as i64))
                } else {
                    x.cmp(&y)
                };
                let expected = match op {
                    0 => false,
                    1 => ordering.is_lt(),
                    2 => ordering.is_eq(),
                    3 => ordering.is_le(),
                    4 => ordering.is_gt(),
                    5 => ordering.is_ne(),
                    6 => ordering.is_ge(),
                    _ => true,
                };
                let context = format!("{x:x} {y:x} op {op} signed {signed}");
                assert_eq!(state.pred(p(1)), expected, "{context}");
                assert_eq!(state.pred(p(2)), !expected, "{context}");
            }
        }
    }
}

#[test]
fn isetp_combines_with_the_accumulator() {
    let mut values = Values::new();
    for case in 0..96u32 {
        let bop = u128::from(case % 3);
        let accumulator = case & 4 != 0;
        let (x, y) = (values.next(), values.next());
        let inst = encode(
            Opcode::Isetp,
            Form::RegReg,
            a(1) | b(2) | (bop << 74) | (1 << 76) | pred(81, 0) | pred(84, 1) | pred(87, 5),
        );
        let mut state = State::with_regs(&[(r(1), x), (r(2), y)]);
        state.preds.insert(p(5), accumulator);
        run(&[inst], &[], &mut state);
        let less = x < y;
        let apply = |v: bool| match bop {
            0 => v && accumulator,
            1 => v || accumulator,
            _ => v != accumulator,
        };
        assert_eq!(state.pred(p(0)), apply(less), "{case}");
        assert_eq!(state.pred(p(1)), apply(!less), "{case}");
    }
}

// LOP3, PLOP3

fn lut_reference(lut: u8, x: u32, y: u32, z: u32) -> u32 {
    (0..32).fold(0, |acc, i| {
        let index = ((x >> i) & 1) << 2 | ((y >> i) & 1) << 1 | ((z >> i) & 1);
        acc | ((u32::from(lut) >> index) & 1) << i
    })
}

#[test]
fn lop3_matches_every_truth_table() {
    let mut values = Values::new();
    for lut in 0..=255u8 {
        let inst = encode(
            Opcode::Lop3,
            Form::RegReg,
            dst(0) | a(1) | b(2) | c(3) | (u128::from(lut) << 72) | pred(81, 0) | (0xf << 87),
        );
        for _ in 0..4 {
            let (x, y, z) = (values.next(), values.next(), values.next());
            let state = exec(inst, &[(r(1), x), (r(2), y), (r(3), z)]);
            let expected = lut_reference(lut, x, y, z);
            assert_eq!(state.reg(r(0)), expected, "{lut:#x} {x:x} {y:x} {z:x}");
            assert_eq!(state.pred(p(0)), expected != 0, "{lut:#x}");
        }
    }
}

#[test]
fn lop3_with_a_predicate_output() {
    let inst = encode(
        Opcode::Lop3,
        Form::RegReg,
        dst(0) | a(1) | b(2) | c(3) | (0x96 << 72) | pred(81, 1) | (0xf << 87),
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetRegister R3
  %3 = BitwiseXor32 %1, %2
  %4 = BitwiseXor32 %0, %3
  SetRegister R0, %4
  %5 = INotEqual %4, #0x0
  SetPred P1, %5
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn ulop3_with_an_immediate() {
    let inst = encode(
        Opcode::Ulop3,
        Form::ImmReg,
        dst(4) | a(5) | b(0xff) | c(URZ) | (0xc0 << 72) | pred(81, PT) | (0xf << 87) | bit(91),
    );
    let expected = "\
{0}:
  %0 = GetRegister UR5
  %1 = BitwiseAnd32 %0, #0xff
  SetRegister UR4, %1
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn plop3_matches_every_truth_table() {
    let mut values = Values::new();
    for lut in 0..=255u32 {
        let second = values.next() as u8;
        let negations = u128::from(values.next() & 7);
        let first_lut = (u128::from(lut) & 7) << 64 | (u128::from(lut) >> 3) << 72;
        let inst = encode(
            Opcode::Plop3,
            Form::Fixed,
            first_lut
                | (u128::from(second) << 16)
                | pred(87, 0)
                | ((negations & 1) << 90)
                | pred(77, 1)
                | ((negations >> 1 & 1) << 80)
                | pred(68, 2)
                | ((negations >> 2) << 71)
                | pred(81, 3)
                | pred(84, 4),
        );
        for inputs in 0..8u32 {
            let mut state = State::default();
            for i in 0..3u8 {
                state.preds.insert(p(i), inputs >> i & 1 != 0);
            }
            run(&[inst], &[], &mut state);
            let index = |n: u32| -> u32 {
                let value = |i: u32| ((inputs >> i) & 1) ^ ((n >> i) & 1);
                value(0) << 2 | value(1) << 1 | value(2)
            };
            let n = negations as u32;
            assert_eq!(
                state.pred(p(3)),
                (lut >> index(n)) & 1 != 0,
                "{lut:#x} {inputs}"
            );
            assert_eq!(
                state.pred(p(4)),
                (u32::from(second) >> index(n)) & 1 != 0,
                "{second:#x} {inputs}"
            );
        }
    }
}

#[test]
fn plop3_reads_a_uniform_third_source() {
    let inst = encode(
        Opcode::Plop3,
        Form::Fixed,
        (0x80 >> 3 << 72)
            | pred(87, 0)
            | pred(77, 1)
            | pred(68, 2)
            | bit(67)
            | pred(81, 3)
            | pred(84, PT),
    );
    let expected = "\
{0}:
  %0 = GetPred P0
  %1 = GetPred P1
  %2 = GetPred UP2
  %3 = LogicalAnd %1, %2
  %4 = LogicalAnd %0, %3
  SetPred P3, %4
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn uplop3_combines_uniform_predicates() {
    let inst = encode(
        Opcode::Uplop3,
        Form::Fixed,
        (0xfe >> 3 << 72)
            | (0xfe & 7) << 64
            | pred(87, 0)
            | pred(77, 1)
            | pred(68, PT)
            | (1 << 71)
            | pred(81, 2)
            | pred(84, PT),
    );
    let expected = "\
{0}:
  %0 = GetPred UP0
  %1 = GetPred UP1
  %2 = LogicalOr %1, #false
  %3 = LogicalOr %0, %2
  SetPred UP2, %3
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

// SHF

#[test]
fn shf_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..2048u32 {
        let data_type = case & 3;
        let (right, high, wrap, immediate) =
            (case & 4 != 0, case & 8 != 0, case & 16 != 0, case & 32 != 0);
        let (low, high_word) = (values.next(), values.next());
        let shift = match values.next() % 4 {
            0 => values.next() % 80,
            1 => values.next(),
            _ => values.next() % 34,
        };
        let (form, shift_operand) = if immediate {
            (Form::ImmReg, u128::from(shift))
        } else {
            (Form::RegReg, 2)
        };
        let fields = dst(0)
            | a(1)
            | b(shift_operand)
            | c(3)
            | (u128::from(data_type) << 73)
            | (u128::from(wrap) << 75)
            | (u128::from(right) << 76)
            | (u128::from(high) << 80);
        let state = exec(
            encode(Opcode::Shf, form, fields),
            &[(r(1), low), (r(2), shift), (r(3), high_word)],
        );
        let bits = if data_type < 2 { 64 } else { 32 };
        let amount = if wrap {
            shift & (bits - 1)
        } else {
            shift.min(bits)
        };
        let x = u64::from(low) | (u64::from(high_word) << 32);
        let shifted = match (right, data_type % 2 == 0) {
            (false, _) => x.checked_shl(amount).unwrap_or(0),
            (true, true) => (x as i64).checked_shr(amount).unwrap_or((x as i64) >> 63) as u64,
            (true, false) => x.checked_shr(amount).unwrap_or(0),
        };
        let expected = if high {
            (shifted >> 32) as u32
        } else {
            shifted as u32
        };
        assert_eq!(
            state.reg(r(0)),
            expected,
            "{low:x} {high_word:x} shift {shift} case {case}"
        );
    }
}

#[test]
fn shf_right_of_the_high_word() {
    let inst = encode(
        Opcode::Shf,
        Form::ImmReg,
        dst(0) | a(RZ) | b(4) | c(1) | (3 << 73) | bit(76) | bit(80),
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = CompositeConstructU32x2 #0x0, %0
  %2 = PackUint2x32 %1
  %3 = ShiftRightLogical64 %2, #0x4
  %4 = UnpackUint2x32 %3
  %5 = CompositeExtractU32x2 %4, #0x1
  SetRegister R0, %5
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn ushf_left_of_uniform_registers() {
    let inst = encode(
        Opcode::Ushf,
        Form::RegReg,
        dst(4) | a(5) | b(6) | c(URZ) | (3 << 73) | bit(75) | bit(91),
    );
    let expected = "\
{0}:
  %0 = GetRegister UR5
  %1 = GetRegister UR6
  %2 = CompositeConstructU32x2 %0, #0x0
  %3 = PackUint2x32 %2
  %4 = BitwiseAnd32 %1, #0x1f
  %5 = ShiftLeftLogical64 %3, %4
  %6 = UnpackUint2x32 %5
  %7 = CompositeExtractU32x2 %6, #0x0
  SetRegister UR4, %7
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

// PRMT

fn prmt_reference(x: u32, selector: u32, z: u32) -> u32 {
    (0..4).fold(0, |acc, byte| {
        let nibble = (selector >> (byte * 4)) & 0xf;
        let source = if nibble & 4 == 0 { x } else { z };
        let value = (source >> ((nibble & 3) * 8)) & 0xff;
        let value = match nibble & 8 {
            0 => value,
            _ if value & 0x80 != 0 => 0xff,
            _ => 0,
        };
        acc | value << (byte * 8)
    })
}

#[test]
fn prmt_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..256 {
        let (x, selector, z) = (values.next(), values.next(), values.next());
        let (form, operand) = if case % 2 == 0 {
            (Form::ImmReg, u128::from(selector))
        } else {
            (Form::RegReg, 2)
        };
        let inst = encode(Opcode::Prmt, form, dst(0) | a(1) | b(operand) | c(3));
        let state = exec(inst, &[(r(1), x), (r(2), selector), (r(3), z)]);
        assert_eq!(
            state.reg(r(0)),
            prmt_reference(x, selector, z),
            "{x:x} {selector:x} {z:x}"
        );
    }
}

#[test]
fn prmt_with_an_immediate_selector() {
    let inst = encode(Opcode::Prmt, Form::ImmReg, dst(0) | a(1) | b(0x5410) | c(2));
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = BitFieldUExtract %0, #0x0, #0x8
  %3 = BitFieldInsert #0x0, %2, #0x0, #0x8
  %4 = BitFieldUExtract %0, #0x8, #0x8
  %5 = BitFieldInsert %3, %4, #0x8, #0x8
  %6 = BitFieldUExtract %1, #0x0, #0x8
  %7 = BitFieldInsert %5, %6, #0x10, #0x8
  %8 = BitFieldUExtract %1, #0x8, #0x8
  %9 = BitFieldInsert %7, %8, #0x18, #0x8
  SetRegister R0, %9
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

/// PRMT modes from the PTX `prmt` tables: each entry lists the source bytes of d.b3..d.b0 for
/// the selector's low two bits.
fn prmt_mode_reference(mode: u128, x: u32, selector: u32, z: u32) -> u32 {
    const TABLE: [[[u32; 4]; 4]; 6] = [
        [[3, 2, 1, 0], [4, 3, 2, 1], [5, 4, 3, 2], [6, 5, 4, 3]],
        [[5, 6, 7, 0], [6, 7, 0, 1], [7, 0, 1, 2], [0, 1, 2, 3]],
        [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]],
        [[3, 2, 1, 0], [3, 2, 1, 1], [3, 2, 2, 2], [3, 3, 3, 3]],
        [[0, 0, 0, 0], [1, 1, 1, 0], [2, 2, 1, 0], [3, 2, 1, 0]],
        [[1, 0, 1, 0], [3, 2, 3, 2], [1, 0, 1, 0], [3, 2, 3, 2]],
    ];
    let bytes = u64::from(z) << 32 | u64::from(x);
    let row = TABLE[mode as usize - 1][(selector & 3) as usize];
    (0..4).fold(0, |acc, byte| {
        let index = row[3 - byte];
        acc | (((bytes >> (index * 8)) & 0xff) as u32) << (byte * 8)
    })
}

#[test]
fn prmt_modes_match_the_reference() {
    let mut values = Values::new();
    for mode in 1..=6u128 {
        for case in 0..32 {
            let (x, selector, z) = (values.next(), values.next(), values.next());
            let (form, operand) = if case % 2 == 0 {
                (Form::ImmReg, u128::from(selector))
            } else {
                (Form::RegReg, 2)
            };
            let inst = encode(
                Opcode::Prmt,
                form,
                dst(0) | a(1) | b(operand) | c(3) | (mode << 72),
            );
            let state = exec(inst, &[(r(1), x), (r(2), selector), (r(3), z)]);
            assert_eq!(
                state.reg(r(0)),
                prmt_mode_reference(mode, x, selector, z),
                "mode {mode} {x:x} {selector:x} {z:x}"
            );
        }
    }
}

#[test]
fn prmt_modes_compile_to_valid_spirv() {
    let code: Vec<Instruction> = (1..=6u128)
        .map(|mode| {
            encode(
                Opcode::Prmt,
                Form::RegReg,
                dst(mode - 1) | a(6) | b(7) | c(8) | (mode << 72),
            )
        })
        .collect();
    compile_fragment_and_validate("ampere-prmt-modes", &code);
}

#[test]
fn prmt_reserved_mode_is_unsupported() {
    let inst = encode(
        Opcode::Prmt,
        Form::ImmReg,
        dst(0) | a(1) | b(0) | c(2) | (7 << 72),
    );
    assert!(is_unsupported(translate(&[inst])));
}

// POPC, FLO, BREV

#[test]
fn popc_flo_and_brev_match_the_reference() {
    let mut values = Values::new();
    for case in 0..512u32 {
        let x = values.next();
        let not = case & 1 != 0;
        let source = if not { !x } else { x };
        let state = exec(
            encode(
                Opcode::Popc,
                Form::RegReg,
                dst(0) | b(1) | (u128::from(not) << 63),
            ),
            &[(r(1), x)],
        );
        assert_eq!(state.reg(r(0)), source.count_ones());
        let (signed, shift_amount) = (case & 2 != 0, case & 4 != 0);
        let fields = dst(0)
            | b(1)
            | (u128::from(not) << 63)
            | (u128::from(signed) << 73)
            | (u128::from(shift_amount) << 74)
            | pred(81, PT);
        let state = exec(encode(Opcode::Flo, Form::RegReg, fields), &[(r(1), x)]);
        let bits = if signed && (source as i32) < 0 {
            !source
        } else {
            source
        };
        let expected = match (bits.leading_zeros(), shift_amount) {
            (32, _) => u32::MAX,
            (zeros, true) => zeros,
            (zeros, false) => 31 - zeros,
        };
        assert_eq!(state.reg(r(0)), expected, "{x:x} {case}");
        let state = exec(
            encode(Opcode::Brev, Form::RegReg, dst(0) | b(1)),
            &[(r(1), x)],
        );
        assert_eq!(state.reg(r(0)), x.reverse_bits());
    }
}

#[test]
fn uflo_shift_amount() {
    let inst = encode(
        Opcode::Uflo,
        Form::RegReg,
        dst(4) | b(5) | bit(74) | pred(81, PT) | bit(91),
    );
    let expected = "\
{0}:
  %0 = GetRegister UR5
  %1 = FindUMsb32 %0
  %2 = IEqual %1, #0xffffffff
  %3 = BitwiseXor32 %1, #0x1f
  %4 = SelectU32 %2, %1, %3
  SetRegister UR4, %4
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn upopc_and_ubrev_read_uniform_registers() {
    let popc = encode(Opcode::Upopc, Form::RegReg, dst(4) | b(5) | bit(91));
    let brev = encode(Opcode::Ubrev, Form::RegReg, dst(6) | b(7) | bit(91));
    let expected = "\
{0}:
  %0 = GetRegister UR5
  %1 = BitCount32 %0
  SetRegister UR4, %1
  %2 = GetRegister UR7
  %3 = BitReverse32 %2
  SetRegister UR6, %3
";
    assert_eq!(translate(&[popc, brev]).unwrap(), expected);
}

// LEA

/// LEA with a carry out, then LEA.HI.X, the compiler's scaled 64-bit address.
#[test]
fn lea_pairs_scale_and_add_64_bit_values() {
    let mut values = Values::new();
    for case in 0..512u32 {
        let shift = case % 32;
        let sign_extend = case & 32 != 0;
        let low = encode(
            Opcode::Lea,
            Form::RegReg,
            dst(4) | a(0) | b(2) | c(RZ) | (u128::from(shift) << 75) | pred(81, 0),
        );
        let high = encode(
            Opcode::Lea,
            Form::RegReg,
            dst(5)
                | a(0)
                | b(3)
                | c(if sign_extend { RZ } else { 1 })
                | bit(74)
                | (u128::from(sign_extend) << 73)
                | (u128::from(shift) << 75)
                | bit(80)
                | pred(81, PT)
                | pred(87, 0),
        );
        let (x_low, x_high) = (values.next(), values.next());
        let y = u64::from(values.next()) | (u64::from(values.next()) << 32);
        let mut state = State::with_regs(&[
            (r(0), x_low),
            (r(1), x_high),
            (r(2), y as u32),
            (r(3), (y >> 32) as u32),
        ]);
        run(&[low, high], &[], &mut state);
        let x = if sign_extend {
            i64::from(x_low as i32) as u64
        } else {
            u64::from(x_low) | (u64::from(x_high) << 32)
        };
        let expected = (x << shift).wrapping_add(y);
        let result = u64::from(state.reg(r(4))) | (u64::from(state.reg(r(5))) << 32);
        assert_eq!(result, expected, "{x:x} {y:x} {case}");
    }
}

#[test]
fn lea_negations_match_the_reference() {
    let mut values = Values::new();
    for case in 0..128u32 {
        let (negate_shifted, negate_b) = (case & 1 != 0, case & 2 != 0);
        let shift = (case >> 2) % 32;
        let (x, y) = (values.next(), values.next());
        let fields = dst(0)
            | a(1)
            | b(2)
            | c(RZ)
            | (u128::from(negate_shifted) << 72)
            | (u128::from(negate_b) << 63)
            | (u128::from(shift) << 75)
            | pred(81, 0);
        let state = exec(
            encode(Opcode::Lea, Form::RegReg, fields),
            &[(r(1), x), (r(2), y)],
        );
        let terms = [(x << shift, negate_shifted), (y, negate_b)];
        let sum: u64 = terms
            .iter()
            .map(|&(v, negated)| {
                if negated {
                    u64::from(!v) + 1
                } else {
                    u64::from(v)
                }
            })
            .sum();
        assert_eq!(state.reg(r(0)), sum as u32, "{x:x} {y:x} {case}");
        assert_eq!(state.pred(p(0)), sum >> 32 != 0, "{x:x} {y:x} {case}");
    }
}

#[test]
fn lea_shifts_and_adds() {
    let inst = encode(
        Opcode::Lea,
        Form::RegReg,
        dst(0) | a(1) | b(2) | c(RZ) | (3 << 75) | pred(81, PT),
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = ShiftLeftLogical32 %0, #0x3
  %3 = IAdd32 %2, %1
  SetRegister R0, %3
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

/// Without .HI only the low word of `a << s` is added, which the sign of `a` never reaches.
#[test]
fn lea_sx32_low_form_is_the_plain_lea() {
    let fields = dst(0) | a(1) | b(2) | c(RZ) | (3 << 75) | pred(81, PT);
    let plain = translate(&[encode(Opcode::Lea, Form::RegReg, fields)]).unwrap();
    let sign_extended = translate(&[encode(Opcode::Lea, Form::RegReg, fields | bit(73))]).unwrap();
    assert_eq!(sign_extended, plain);
}

#[test]
fn ulea_hi_of_uniform_registers() {
    let inst = encode(
        Opcode::Ulea,
        Form::RegReg,
        dst(4) | a(5) | b(6) | c(7) | (2 << 75) | bit(80) | pred(81, PT) | bit(91),
    );
    let expected = "\
{0}:
  %0 = GetRegister UR5
  %1 = GetRegister UR6
  %2 = GetRegister UR7
  %3 = ShiftRightLogical32 %0, #0x1e
  %4 = ShiftLeftLogical32 %2, #0x2
  %5 = BitwiseOr32 %3, %4
  %6 = IAdd32 %5, %1
  SetRegister UR4, %6
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

// SEL, SGXT, BMSK

#[test]
fn sel_picks_by_predicate() {
    let inst = encode(
        Opcode::Sel,
        Form::RegReg,
        dst(0) | a(1) | b(2) | pred(87, 2) | bit(90),
    );
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetPred P2
  %3 = LogicalNot %2
  %4 = SelectU32 %3, %0, %1
  SetRegister R0, %4
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn usel_picks_by_uniform_predicate() {
    let inst = encode(
        Opcode::Usel,
        Form::ImmReg,
        dst(4) | a(5) | b(9) | pred(87, 1) | bit(91),
    );
    let expected = "\
{0}:
  %0 = GetRegister UR5
  %1 = GetPred UP1
  %2 = SelectU32 %1, %0, #0x9
  SetRegister UR4, %2
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn sgxt_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..256u32 {
        let signed = case & 1 != 0;
        let (x, bits) = (values.next(), values.next() % 40);
        let fields = dst(0) | a(1) | b(2) | (u128::from(signed) << 73);
        let state = exec(
            encode(Opcode::Sgxt, Form::RegReg, fields),
            &[(r(1), x), (r(2), bits)],
        );
        let expected = match bits {
            0 => 0,
            32.. => x,
            n if signed => (((x << (32 - n)) as i32) >> (32 - n)) as u32,
            n => (x << (32 - n)) >> (32 - n),
        };
        assert_eq!(state.reg(r(0)), expected, "{x:x} {bits} {signed}");
    }
}

#[test]
fn bmsk_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..512u32 {
        let (wrap, immediate) = (case & 1 != 0, case & 2 != 0);
        let (start, width) = (values.next() % 40, values.next() % 40);
        let (form, width_operand) = if immediate {
            (Form::ImmReg, u128::from(width))
        } else {
            (Form::RegReg, 2)
        };
        let fields = dst(0) | a(1) | b(width_operand) | (u128::from(wrap) << 75);
        let state = exec(
            encode(Opcode::Bmsk, form, fields),
            &[(r(1), start), (r(2), width)],
        );
        let (s, w) = if wrap {
            (start & 31, width & 31)
        } else {
            (start.min(32), width.min(32))
        };
        let expected = (((1u64 << w) - 1) << s) as u32;
        assert_eq!(state.reg(r(0)), expected, "{start} {width} {case}");
    }
}

// MOV, UMOV

#[test]
fn mov_from_every_operand_kind() {
    let quad = 0xf << 72;
    let code = [
        encode(Opcode::Mov, Form::RegReg, dst(0) | b(1) | quad),
        encode(Opcode::Mov, Form::ImmReg, dst(2) | b(0x1234) | quad),
        encode(
            Opcode::Mov,
            Form::CbufReg,
            dst(3) | (1 << 54) | (0x10 << 38) | quad,
        ),
        encode(Opcode::Mov, Form::UregReg, dst(4) | b(9) | quad | bit(91)),
    ];
    let expected = "\
{0}:
  %0 = GetRegister R1
  SetRegister R0, %0
  SetRegister R2, #0x1234
  %1 = GetCbufU32 #0x1, #0x10
  SetRegister R3, %1
  %2 = GetRegister UR9
  SetRegister R4, %2
";
    assert_eq!(translate(&code).unwrap(), expected);
}

#[test]
fn mov_to_part_of_a_quad_is_unsupported() {
    let inst = encode(Opcode::Mov, Form::RegReg, dst(0) | b(1) | (0x3 << 72));
    assert!(is_unsupported(translate(&[inst])));
}

#[test]
fn umov_writes_a_uniform_register() {
    let code = [
        encode(Opcode::Umov, Form::ImmReg, dst(5) | b(0x20) | bit(91)),
        encode(Opcode::Umov, Form::UregReg, dst(6) | b(5) | bit(91)),
    ];
    let expected = "\
{0}:
  SetRegister UR5, #0x20
  %0 = GetRegister UR5
  SetRegister UR6, %0
";
    assert_eq!(translate(&code).unwrap(), expected);
}

// P2R, UP2UR, R2P, R2UR

#[test]
fn p2r_and_r2p_match_the_reference() {
    let mut values = Values::new();
    for case in 0..256u32 {
        let byte = case % 4;
        let immediate = case & 4 != 0;
        let (merge, mask, source) = (values.next(), values.next() & 0x7f, values.next());
        let preds = values.next() & 0x7f;
        let (form, mask_operand) = if immediate {
            (Form::ImmReg, u128::from(mask))
        } else {
            (Form::RegReg, 2)
        };
        let p2r = encode(
            Opcode::P2r,
            form,
            dst(0) | a(1) | b(mask_operand) | (u128::from(byte) << 76),
        );
        let r2p = encode(
            Opcode::R2p,
            form,
            a(3) | b(mask_operand) | (u128::from(byte) << 76),
        );
        let mut state = State::with_regs(&[(r(1), merge), (r(2), mask), (r(3), source)]);
        for i in 0..7u8 {
            state.preds.insert(p(i), preds >> i & 1 != 0);
        }
        run(&[p2r, r2p], &[], &mut state);
        let shift = byte * 8;
        let expected = (merge & !(mask << shift)) | ((preds & mask) << shift);
        assert_eq!(state.reg(r(0)), expected, "{case}");
        for i in 0..7u8 {
            let expected = if mask >> i & 1 != 0 {
                source >> (shift + u32::from(i)) & 1 != 0
            } else {
                preds >> i & 1 != 0
            };
            assert_eq!(state.pred(p(i)), expected, "P{i} {case}");
        }
    }
}

#[test]
fn up2ur_packs_uniform_predicates() {
    let inst = encode(
        Opcode::Up2ur,
        Form::ImmReg,
        dst(4) | a(URZ) | b(0x5) | bit(91),
    );
    let expected = "\
{0}:
  %0 = GetPred UP0
  %1 = SelectU32 %0, #0x1, #0x0
  %2 = GetPred UP2
  %3 = SelectU32 %2, #0x4, #0x0
  %4 = BitwiseOr32 %1, %3
  SetRegister UR4, %4
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn r2ur_broadcasts_the_first_active_lane() {
    let inst = encode(Opcode::R2ur, Form::Fixed, dst(4) | a(1) | pred(81, PT));
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = SubgroupBallot #true
  %2 = INeg32 %1
  %3 = BitwiseAnd32 %1, %2
  %4 = FindUMsb32 %3
  %5 = ShuffleIndex %0, %4, #0x1f, #0x0
  SetRegister UR4, %5
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

// End to end

/// The opcodes of the instructions in the SPIR-V module `code`.
pub(crate) fn spirv_opcodes(code: &[u32]) -> Vec<u16> {
    let mut opcodes = Vec::new();
    let mut i = 5;
    while i < code.len() {
        opcodes.push(code[i] as u16);
        i += ((code[i] >> 16) as usize).max(1);
    }
    opcodes
}

/// Compiles `code` and an EXIT as a fragment program writing R0..R11 to three render
/// targets, checks the module with spirv-val, and returns its opcodes.
pub(crate) fn compile_fragment_and_validate(name: &str, code: &[Instruction]) -> Vec<u16> {
    let mut code = code.to_vec();
    code.push(encode(Opcode::Exit, Form::Fixed, pred(87, PT)));
    let mut env = TestEnv::new(&code);
    env.stage = ShaderStage::Fragment;
    env.header[18] = 0xfff;
    env.cbuf = (0..64).map(|i| i * 0x0101_0101).collect();
    let profile = crate::Profile::default();
    let options = crate::translate::TranslateOptions {
        profile: &profile,
        host: &crate::profile::HostTranslateInfo::default(),
        resolution_scale: lichii_common::settings::ResolutionScale::X1,
    };
    let module = crate::module::compile(
        &env,
        options,
        &crate::RuntimeInfo::default(),
        &mut crate::backend::Bindings::default(),
    )
    .expect("compiles");
    crate::backend::spirv::tests::validate(name, &profile, &module.spirv);
    spirv_opcodes(&module.spirv)
}

#[test]
fn integer_family_compiles_to_valid_spirv() {
    let code = [
        encode(
            Opcode::Mov,
            Form::CbufReg,
            dst(0) | (0x10 << 38) | (0xf << 72),
        ),
        encode(
            Opcode::Iadd3,
            Form::ImmReg,
            dst(1) | a(0) | b(0x1234) | c(RZ) | NO_OUTPUTS | NO_CARRY_INS,
        ),
        encode(
            Opcode::Imad,
            Form::RegReg,
            dst(2) | a(0) | b(1) | c(0) | NO_OUTPUTS,
        ),
        encode(
            Opcode::Lop3,
            Form::RegReg,
            dst(3) | a(1) | b(2) | c(0) | (0x96 << 72) | pred(81, PT) | (0xf << 87),
        ),
        encode(
            Opcode::Shf,
            Form::ImmReg,
            dst(4) | a(0) | b(5) | c(1) | (3 << 73) | bit(80),
        ),
        encode(Opcode::Prmt, Form::ImmReg, dst(5) | a(0) | b(0x5410) | c(1)),
        encode(
            Opcode::Isetp,
            Form::RegReg,
            a(2) | b(3) | (1 << 76) | pred(81, 0) | pred(84, PT) | pred(87, PT),
        ),
        encode(
            Opcode::Sel,
            Form::RegReg,
            dst(6) | a(4) | b(5) | pred(87, 0),
        ),
        encode(
            Opcode::Lea,
            Form::RegReg,
            dst(7) | a(0) | b(6) | c(RZ) | (3 << 75) | pred(81, PT),
        ),
        encode(
            Opcode::ImadWide,
            Form::RegReg,
            dst(8) | a(0) | b(1) | c(2) | NO_OUTPUTS | NO_CARRY_INS,
        ),
        encode(Opcode::Popc, Form::RegReg, dst(10) | b(7)),
        encode(Opcode::Flo, Form::RegReg, dst(11) | b(8) | pred(81, PT)),
        encode(Opcode::Umov, Form::ImmReg, dst(4) | b(0x20) | bit(91)),
        encode(
            Opcode::Uiadd3,
            Form::RegReg,
            dst(5) | a(4) | b(4) | c(URZ) | bit(91) | NO_OUTPUTS | NO_CARRY_INS,
        ),
        encode(
            Opcode::Iadd3,
            Form::UregReg,
            dst(9) | a(9) | b(5) | c(RZ) | NO_OUTPUTS | NO_CARRY_INS,
        ),
    ];
    let opcodes = compile_fragment_and_validate("ampere-integer", &code);
    let (op_i_mul, op_bit_count) = (132, 205);
    assert!(opcodes.contains(&op_i_mul) && opcodes.contains(&op_bit_count));
}
