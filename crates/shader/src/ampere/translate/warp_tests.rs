//! Tests for the warp translators: IR dumps of hand-encoded instructions, and numeric checks
//! that run the translated IR on a whole warp against PTX's definitions.

use super::super::integer::tests::{
    PT, RZ, State, Values, WARP, a, b, bit, c, dst, is_unsupported, p, pred, r, run_warp,
};
use super::super::tests::{encode, translate, translate_stage};
use crate::ShaderStage;
use crate::ampere::decode::Form;
use crate::ampere::opcode::Opcode;
use crate::ir::Reg;

/// A warp whose lane `i` holds `values[i]` in register `reg`.
fn warp_with(reg: Reg, values: &[u32]) -> Vec<State> {
    values
        .iter()
        .map(|&value| State::with_regs(&[(reg, value)]))
        .collect()
}

fn active_lanes(mask: u32) -> impl Iterator<Item = usize> {
    (0..WARP).filter(move |&lane| mask >> lane & 1 != 0)
}

/// Random active masks, never empty, including the full warp.
fn active_mask(values: &mut Values, case: u32) -> u32 {
    match case % 4 {
        0 => u32::MAX,
        _ => values.next() | 1 << (case % 32),
    }
}

// VOTE, VOTEU

#[test]
fn vote_any_with_a_ballot() {
    let inst = encode(
        Opcode::Vote,
        Form::Fixed,
        dst(0) | (1 << 72) | pred(81, 1) | pred(87, 2) | bit(90),
    );
    let expected = "\
{0}:
  %0 = GetPred P2
  %1 = LogicalNot %0
  %2 = VoteAny %1
  SetPred P1, %2
  %3 = SubgroupBallot %1
  SetRegister R0, %3
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn voteu_writes_a_uniform_register_and_predicate() {
    let inst = encode(
        Opcode::Voteu,
        Form::Fixed,
        dst(4) | pred(81, 1) | pred(87, 0),
    );
    let expected = "\
{0}:
  %0 = GetPred P0
  %1 = VoteAll %0
  SetPred UP1, %1
  %2 = SubgroupBallot %0
  SetRegister UR4, %2
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

#[test]
fn vote_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..96u32 {
        let (mode, negated) = (case % 3, case & 8 != 0);
        let active = active_mask(&mut values, case);
        let set = match case % 5 {
            0 => u32::MAX,
            1 => 0,
            _ => values.next(),
        };
        let inst = encode(
            Opcode::Vote,
            Form::Fixed,
            dst(0)
                | (u128::from(mode) << 72)
                | pred(81, 1)
                | pred(87, 2)
                | (u128::from(negated) << 90),
        );
        let mut lanes: Vec<State> = (0..WARP)
            .map(|lane| {
                let mut state = State::default();
                state.preds.insert(p(2), set >> lane & 1 != 0);
                state
            })
            .collect();
        run_warp(ShaderStage::Compute, &[inst], &[], &mut lanes, active);
        let ballot = if negated { !set } else { set } & active;
        let vote = match mode {
            0 => ballot == active,
            1 => ballot != 0,
            _ => ballot == active || ballot == 0,
        };
        for lane in active_lanes(active) {
            assert_eq!(lanes[lane].reg(r(0)), ballot, "{case}");
            assert_eq!(lanes[lane].pred(p(1)), vote, "{case}");
        }
    }
}

#[test]
fn vote_reserved_mode_is_unsupported() {
    let inst = encode(Opcode::Vote, Form::Fixed, dst(0) | (3 << 72) | pred(87, 0));
    assert!(is_unsupported(translate(&[inst])));
}

// SHFL

/// PTX's shfl.sync: the source lane, or this lane when out of bounds, and the bounds flag.
fn shfl_reference(mode: u32, lane: u32, b: u32, c: u32) -> (u32, bool) {
    let (b, clamp, segment_mask) = (b & 31, c & 31, (c >> 8) & 31);
    let max_lane = (lane & segment_mask) | (clamp & !segment_mask);
    let min_lane = lane & segment_mask;
    let (source, in_bounds) = match mode {
        0 => {
            let source = min_lane | (b & !segment_mask);
            (source, source <= max_lane)
        }
        1 => {
            let source = lane.wrapping_sub(b);
            (source, source as i32 >= max_lane as i32)
        }
        2 => (lane + b, lane + b <= max_lane),
        _ => (lane ^ b, lane ^ b <= max_lane),
    };
    if in_bounds {
        (source, true)
    } else {
        (lane, false)
    }
}

/// Every mode and operand layout, at the widths CUDA's __shfl_*_sync encode, and in fragment
/// programs the quad shuffles that become quad operations.
#[test]
fn shfl_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..512u32 {
        let mode = case % 4;
        let width = 1 << (1 + (case / 4) % 5);
        let segment_mask = !(width - 1) & 31;
        let clamp = if mode == 1 { 0 } else { width - 1 };
        let c_value = (segment_mask << 8) | clamp;
        let b_value = values.next() % 40;
        let (lane_immediate, mask_immediate) = (case & 32 != 0, case & 64 != 0);
        let form = match (lane_immediate, mask_immediate) {
            (false, false) => Form::RegReg,
            (false, true) => Form::RegImm,
            (true, false) => Form::ImmReg,
            (true, true) => Form::ImmImm,
        };
        let lane_field = if lane_immediate {
            u128::from(b_value & 31) << 53
        } else {
            b(2)
        };
        let mask_field = if mask_immediate {
            u128::from(c_value) << 40
        } else {
            c(3)
        };
        let inst = encode(
            Opcode::Shfl,
            form,
            dst(4) | a(1) | lane_field | mask_field | (u128::from(mode) << 58) | pred(81, 0),
        );
        let stage = if case & 128 != 0 {
            ShaderStage::Fragment
        } else {
            ShaderStage::Compute
        };
        let data: Vec<u32> = (0..WARP).map(|_| values.next()).collect();
        let mut lanes: Vec<State> = data
            .iter()
            .map(|&value| State::with_regs(&[(r(1), value), (r(2), b_value), (r(3), c_value)]))
            .collect();
        run_warp(stage, &[inst], &[], &mut lanes, u32::MAX);
        for (lane, state) in lanes.iter().enumerate() {
            let (source, in_bounds) = shfl_reference(mode, lane as u32, b_value, c_value);
            let context = format!("case {case} lane {lane} b {b_value} c {c_value:#x}");
            assert_eq!(state.reg(r(4)), data[source as usize], "{context}");
            assert_eq!(state.pred(p(0)), in_bounds, "{context}");
        }
    }
}

#[test]
fn fragment_quad_shuffles_become_quad_operations() {
    let quad = (28 << 8 | 3) << 40;
    let code = [
        encode(
            Opcode::Shfl,
            Form::ImmImm,
            dst(4) | a(1) | (2 << 53) | quad | pred(81, PT),
        ),
        encode(
            Opcode::Shfl,
            Form::ImmImm,
            dst(5) | a(1) | (1 << 53) | quad | (3 << 58) | pred(81, PT),
        ),
    ];
    let expected = "\
{0}:
  %0 = GetRegister R1
  %1 = QuadBroadcast %0, #0x2
  SetRegister R4, %1
  %2 = GetRegister R1
  %3 = QuadSwap %2, #0x0
  SetRegister R5, %3
";
    assert_eq!(
        translate_stage(ShaderStage::Fragment, &code).unwrap(),
        expected
    );
}

#[test]
fn shfl_down_with_registers() {
    let inst = encode(
        Opcode::Shfl,
        Form::RegReg,
        dst(4) | a(1) | b(2) | c(3) | (2 << 58) | pred(81, 0),
    );
    let expected = "\
{0}:
  %0 = GetRegister R2
  %1 = GetRegister R3
  %2 = GetRegister R1
  %3 = BitFieldUExtract %1, #0x0, #0x5
  %4 = BitFieldUExtract %1, #0x8, #0x5
  %5 = ShuffleDown %2, %0, %3, %4
  %6 = GetInBoundsFromOp %5
  SetPred P0, %6
  SetRegister R4, %5
";
    assert_eq!(translate(&[inst]).unwrap(), expected);
}

// MATCH

#[test]
fn match_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..128u32 {
        let (any, wide) = (case & 1 != 0, case & 2 != 0);
        let active = active_mask(&mut values, case);
        let palette = [values.next(), values.next(), values.next()];
        let distinct = 1 + (case / 4) as usize % 3;
        let words: Vec<(u32, u32)> = (0..WARP)
            .map(|_| {
                let pick = |values: &mut Values| palette[values.next() as usize % distinct];
                (pick(&mut values), pick(&mut values))
            })
            .collect();
        let inst = encode(
            Opcode::Match,
            Form::Fixed,
            dst(0)
                | a(2)
                | (u128::from(wide) << 73)
                | (u128::from(any) << 79)
                | pred(81, if any { PT } else { 1 }),
        );
        let mut lanes: Vec<State> = words
            .iter()
            .map(|&(low, high)| State::with_regs(&[(r(2), low), (r(3), high)]))
            .collect();
        run_warp(ShaderStage::Compute, &[inst], &[], &mut lanes, active);
        let key = |lane: usize| {
            if wide {
                words[lane]
            } else {
                (words[lane].0, 0)
            }
        };
        let first = active.trailing_zeros() as usize;
        let all_same = active_lanes(active).all(|lane| key(lane) == key(first));
        for lane in active_lanes(active) {
            let expected = if any {
                active_lanes(active)
                    .filter(|&other| key(other) == key(lane))
                    .fold(0, |mask, other| mask | 1 << other)
            } else if all_same {
                active
            } else {
                0
            };
            assert_eq!(lanes[lane].reg(r(0)), expected, "case {case} lane {lane}");
            if !any {
                assert_eq!(lanes[lane].pred(p(1)), all_same, "case {case}");
            }
        }
    }
}

#[test]
fn match_of_an_odd_pair_is_unsupported() {
    let inst = encode(
        Opcode::Match,
        Form::Fixed,
        dst(0) | a(3) | bit(73) | bit(79),
    );
    assert!(is_unsupported(translate(&[inst])));
}

// REDUX

#[test]
fn redux_matches_the_reference() {
    let mut values = Values::new();
    for case in 0..256u32 {
        let op = case % 6;
        let signed = op >= 4 && case & 8 != 0;
        let active = active_mask(&mut values, case);
        let data: Vec<u32> = (0..WARP).map(|_| values.next()).collect();
        let inst = encode(
            Opcode::Redux,
            Form::Fixed,
            dst(4) | a(1) | (u128::from(signed) << 73) | (u128::from(op) << 78),
        );
        let mut lanes = warp_with(r(1), &data);
        run_warp(ShaderStage::Compute, &[inst], &[], &mut lanes, active);
        let terms = active_lanes(active).map(|lane| data[lane]);
        let expected = match (op, signed) {
            (0, _) => terms.fold(u32::MAX, |x, y| x & y),
            (1, _) => terms.fold(0, |x, y| x | y),
            (2, _) => terms.fold(0, |x, y| x ^ y),
            (3, _) => terms.fold(0, u32::wrapping_add),
            (4, false) => terms.min().unwrap(),
            (5, false) => terms.max().unwrap(),
            (4, true) => terms.map(|x| x as i32).min().unwrap() as u32,
            _ => terms.map(|x| x as i32).max().unwrap() as u32,
        };
        for lane in active_lanes(active) {
            assert_eq!(
                lanes[lane].reg(Reg::ur(4)),
                expected,
                "case {case} lane {lane}"
            );
        }
    }
}

#[test]
fn redux_reserved_operations_are_unsupported() {
    for fields in [6 << 78, 7 << 78, bit(73)] {
        let inst = encode(Opcode::Redux, Form::Fixed, dst(4) | a(1) | fields);
        assert!(is_unsupported(translate(&[inst])), "{fields:#x}");
    }
}

#[test]
fn r2ur_matches_the_first_active_lane() {
    let mut values = Values::new();
    for case in 0..32u32 {
        let active = active_mask(&mut values, case);
        let data: Vec<u32> = (0..WARP).map(|_| values.next()).collect();
        let inst = encode(Opcode::R2ur, Form::Fixed, dst(4) | a(1) | pred(81, PT));
        let mut lanes = warp_with(r(1), &data);
        run_warp(ShaderStage::Compute, &[inst], &[], &mut lanes, active);
        let first = active.trailing_zeros() as usize;
        for lane in active_lanes(active) {
            assert_eq!(lanes[lane].reg(Reg::ur(4)), data[first], "case {case}");
        }
    }
}

#[test]
fn warp_family_compiles_to_valid_spirv() {
    let code = [
        encode(
            Opcode::Vote,
            Form::Fixed,
            dst(0) | (1 << 72) | pred(81, 1) | pred(87, PT),
        ),
        encode(
            Opcode::Shfl,
            Form::ImmImm,
            dst(1) | a(0) | (1 << 53) | (0x1f << 40) | (3 << 58) | pred(81, 2),
        ),
        encode(
            Opcode::Shfl,
            Form::ImmImm,
            dst(2) | a(1) | (1 << 53) | ((28 << 8 | 3) << 40) | (3 << 58) | pred(81, PT),
        ),
        encode(
            Opcode::Match,
            Form::Fixed,
            dst(3) | a(2) | bit(79) | pred(81, PT),
        ),
        encode(
            Opcode::Match,
            Form::Fixed,
            dst(4) | a(2) | bit(73) | pred(81, 3),
        ),
        encode(Opcode::Redux, Form::Fixed, dst(5) | a(3) | (3 << 78)),
        encode(
            Opcode::Redux,
            Form::Fixed,
            dst(6) | a(4) | bit(73) | (5 << 78),
        ),
        encode(Opcode::R2ur, Form::Fixed, dst(7) | a(1) | pred(81, PT)),
        encode(
            Opcode::Voteu,
            Form::Fixed,
            dst(8) | pred(81, 2) | pred(87, 1),
        ),
        encode(
            Opcode::Iadd3,
            Form::UregReg,
            dst(0) | a(0) | b(5) | c(RZ) | (PT << 81) | (PT << 84) | (0xf << 87) | (0xf << 77),
        ),
        encode(
            Opcode::Iadd3,
            Form::UregReg,
            dst(1) | a(1) | b(6) | c(RZ) | (PT << 81) | (PT << 84) | (0xf << 87) | (0xf << 77),
        ),
        encode(
            Opcode::Iadd3,
            Form::UregReg,
            dst(2) | a(2) | b(7) | c(RZ) | (PT << 81) | (PT << 84) | (0xf << 87) | (0xf << 77),
        ),
        encode(
            Opcode::Iadd3,
            Form::UregReg,
            dst(3) | a(3) | b(8) | c(RZ) | (PT << 81) | (PT << 84) | (0xf << 87) | (0xf << 77),
        ),
    ];
    let opcodes = super::super::integer::tests::compile_fragment_and_validate("ampere-warp", &code);
    let (op_ballot, op_quad_swap) = (339, 366);
    assert!(opcodes.contains(&op_ballot) && opcodes.contains(&op_quad_swap));
}
