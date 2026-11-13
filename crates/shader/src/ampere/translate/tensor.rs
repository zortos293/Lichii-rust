//! Tensor-core instructions: HMMA.1688 and HMMA.16816 with half accumulators, which become the
//! IR's software matrix multiply-add, and MOVM.16.MT88, an 8x8 transpose across the warp.

use super::{TResult, Translator};
use crate::ampere::opcode::Opcode;
use crate::ir::{Opcode as Op, Reg, Value};

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    match op {
        Opcode::Hmma => hmma(t),
        Opcode::Movm => movm(t),
        _ => Err(t.unsupported()),
    }
}

fn aligned(reg: Reg, count: u16) -> bool {
    reg.is_zero() || reg.0.is_multiple_of(count)
}

/// `HMMA.16816.F16 Rd, Ra, Rb, Rc` (.1688 when bit 75 is clear): D = A * B + C on 16x8 tiles with
/// K = 16 (or 8), every operand packed halves spread over the warp. A takes four registers
/// (two for K = 8), B two (one), C and D two; Rc = RZ adds zero.
fn hmma(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let long = insn.bit(75);
    let (dest, a, b, c) = (insn.dst(), insn.src0(), insn.src1(), insn.reg_at(64));
    let (a_count, b_count) = if long { (4, 2) } else { (2, 1) };
    if insn.bits(40, 24) != 0
        || insn.bits(72, 24) & !(1 << 3) != 0
        || insn.bits(96, 9) != 0
        || !aligned(dest, 2)
        || !aligned(a, a_count)
        || !aligned(b, b_count)
        || !aligned(c, 2)
    {
        return Err(t.unsupported());
    }
    let words = |t: &mut Translator<'_>, reg: Reg, count: u16, total: u16| {
        let values: Vec<Value> = (0..total)
            .map(|i| {
                if i < count && !reg.is_zero() {
                    t.x(reg.offset(i))
                } else {
                    Value::U32(0)
                }
            })
            .collect();
        t.ir.composite_construct(&values)
    };
    let a = words(t, a, a_count, 4);
    let b = words(t, b, b_count, 2);
    let c = words(t, c, 2, 2);
    let depth = if long { 16 } else { 8 };
    let result = t.ir.inst_flags(Op::TensorMulAdd, depth, &[a, b, c]);
    for word in 0..2 {
        let value = t.ir.composite_extract(result, word);
        if !dest.is_zero() {
            t.set_x(dest.offset(word as u16), value);
        }
    }
    Ok(())
}

/// `MOVM.16.MT88 Rd, Ra`: transposes the 8x8 matrix of halves the warp holds one row pair per
/// lane (lane t: row t/4, columns 2(t%4) and 2(t%4)+1). Lane t's result halves are column t/4 at
/// rows 2(t%4) and 2(t%4)+1: half (t/4)%2 of lanes 8(t%4)+t/8 and 8(t%4)+t/8+4.
fn movm(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if insn.bits(32, 64) != 0 {
        return Err(t.unsupported());
    }
    let source = t.x(insn.src0());
    let lane = t.ir.lane_id();
    let quad = t.ir.bitwise_and(lane, Value::U32(3));
    let row_lane = t.ir.shift_left_logical(quad, Value::U32(3));
    let column = t.ir.shift_right_logical(lane, Value::U32(3));
    let low_lane = t.ir.iadd(row_lane, column);
    let high_lane = t.ir.iadd(low_lane, Value::U32(4));
    let pair = t.ir.shift_right_logical(lane, Value::U32(2));
    let pair = t.ir.bitwise_and(pair, Value::U32(1));
    let shift = t.ir.shift_left_logical(pair, Value::U32(4));
    let half = |t: &mut Translator<'_>, from: Value| {
        let word = t.ir.inst(
            Op::ShuffleIndex,
            &[source, from, Value::U32(31), Value::U32(0)],
        );
        t.ir.shift_right_logical(word, shift)
    };
    let low = half(t, low_lane);
    let low = t.ir.bitwise_and(low, Value::U32(0xffff));
    let high = half(t, high_lane);
    let high = t.ir.shift_left_logical(high, Value::U32(16));
    let value = t.ir.bitwise_or(low, high);
    let dest = insn.dst();
    if !dest.is_zero() {
        t.set_x(dest, value);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::ResolutionScale;
    use rustc_hash::FxHashMap;

    use super::super::tests::{TestEnv, encode, translate};
    use super::super::translate_block;
    use crate::ShaderStage;
    use crate::ampere::decode::Form;
    use crate::ampere::instruction::Instruction;
    use crate::ampere::opcode::Opcode;
    use crate::backend::Bindings;
    use crate::backend::spirv::tests::validate;
    use crate::ir::{InstId, Opcode as Op, Program, Value};
    use crate::profile::HostTranslateInfo;
    use crate::translate::TranslateOptions;
    use crate::{Profile, RuntimeInfo};

    fn f(bit: u32, value: u128) -> u128 {
        value << bit
    }

    /// Runs `code` on a 32-lane warp whose registers start as `regs(lane, reg)` over the bytes
    /// of `shared`, and returns each lane's register writes.
    fn run_warp(
        code: &[Instruction],
        regs: impl Fn(u32, u16) -> u32,
        shared: &[u8],
    ) -> Vec<FxHashMap<u16, u32>> {
        let mut env = TestEnv::new(code);
        env.stage = ShaderStage::Compute;
        let mut prog = Program::new();
        let block = prog.new_block();
        translate_block(&mut prog, block, &env, 0, env.end()).unwrap();
        let mut values = vec![FxHashMap::<InstId, u32>::default(); 32];
        let mut writes = vec![FxHashMap::default(); 32];
        let word = |at: u32, bytes: usize| {
            (0..bytes).fold(0, |v, i| v | u32::from(shared[at as usize + i]) << (8 * i))
        };
        for id in prog.insts_of(block).collect::<Vec<_>>() {
            let lane_values: Vec<Option<u32>> = (0..32u32)
                .map(|lane| {
                    let v = |i: usize| match prog.arg(id, i) {
                        Value::U32(c) => c,
                        Value::Inst(arg) => values[lane as usize][&arg],
                        other => panic!("{other:?}"),
                    };
                    Some(match prog.op(id) {
                        Op::GetRegister => match prog.arg(id, 0) {
                            Value::Reg(reg) if reg.is_zero() => 0,
                            Value::Reg(reg) => regs(lane, reg.0),
                            other => panic!("{other:?}"),
                        },
                        Op::LaneId => lane,
                        Op::BitwiseAnd32 => v(0) & v(1),
                        Op::BitwiseOr32 => v(0) | v(1),
                        Op::ShiftLeftLogical32 => v(0) << v(1),
                        Op::ShiftRightLogical32 => v(0) >> v(1),
                        Op::IAdd32 => v(0).wrapping_add(v(1)),
                        Op::ShuffleIndex => match prog.arg(id, 0) {
                            Value::Inst(arg) => values[(v(1) & 31) as usize][&arg],
                            Value::U32(c) => c,
                            other => panic!("{other:?}"),
                        },
                        Op::LoadSharedU16 => word(v(0), 2),
                        Op::LoadSharedU32 => word(v(0), 4),
                        Op::SetRegister => {
                            let Value::Reg(reg) = prog.arg(id, 0) else {
                                panic!()
                            };
                            writes[lane as usize].insert(reg.0, v(1));
                            return None;
                        }
                        op => panic!("{op:?} in a warp test"),
                    })
                })
                .collect();
            for (lane, value) in lane_values.into_iter().enumerate() {
                if let Some(value) = value {
                    values[lane].insert(id, value);
                }
            }
        }
        writes
    }

    /// Element (row, column) of 8x8 matrix `m`.
    fn element(m: u32, row: u32, column: u32) -> u32 {
        m << 12 | row << 4 | column
    }

    fn pack(low: u32, high: u32) -> u32 {
        low | high << 16
    }

    #[test]
    fn ldsm_gives_each_lane_its_share_of_every_matrix() {
        const ROW_BYTES: u32 = 32;
        let row_address = |m: u32, row: u32| 0x100 + m * 0x200 + row * ROW_BYTES;
        let mut shared = vec![0u8; 0x1000];
        for m in 0..4 {
            for row in 0..8 {
                for column in 0..8 {
                    let at = (row_address(m, row) + column * 2) as usize;
                    shared[at..at + 2]
                        .copy_from_slice(&(element(m, row, column) as u16).to_le_bytes());
                }
            }
        }
        let regs = |lane: u32, reg: u16| {
            assert_eq!(reg, 2);
            row_address(lane / 8, lane % 8) - 0x10
        };
        for transpose in [false, true] {
            let insn = encode(
                Opcode::Ldsm,
                Form::UniformAddress,
                f(16, 4) | f(24, 2) | f(40, 0x10) | f(72, 2) | f(78, u128::from(transpose)),
            );
            let writes = run_warp(&[insn], regs, &shared);
            for (lane, writes) in (0u32..).zip(&writes) {
                for m in 0..4 {
                    let (q, g) = (lane % 4, lane / 4);
                    let expected = if transpose {
                        pack(element(m, 2 * q, g), element(m, 2 * q + 1, g))
                    } else {
                        pack(element(m, g, 2 * q), element(m, g, 2 * q + 1))
                    };
                    assert_eq!(writes[&(4 + m as u16)], expected, "lane {lane} matrix {m}");
                }
            }
        }
    }

    #[test]
    fn movm_transposes_the_warp_matrix() {
        let regs = |lane: u32, reg: u16| {
            assert_eq!(reg, 2);
            let (q, g) = (lane % 4, lane / 4);
            pack(element(0, g, 2 * q), element(0, g, 2 * q + 1))
        };
        let insn = encode(Opcode::Movm, Form::Fixed, f(16, 4) | f(24, 2));
        let writes = run_warp(&[insn], regs, &[]);
        for (lane, writes) in (0u32..).zip(&writes) {
            let (q, g) = (lane % 4, lane / 4);
            let expected = pack(element(0, 2 * q, g), element(0, 2 * q + 1, g));
            assert_eq!(writes[&4], expected, "lane {lane}");
        }
    }

    #[test]
    fn hmma_becomes_a_software_multiply_add() {
        let long = encode(
            Opcode::Hmma,
            Form::Fixed,
            f(16, 8) | f(24, 4) | f(32, 2) | f(64, 6) | f(75, 1),
        );
        assert_eq!(
            translate(&[long]).unwrap(),
            "{0}:
  %0 = GetRegister R4
  %1 = GetRegister R5
  %2 = GetRegister R6
  %3 = GetRegister R7
  %4 = CompositeConstructU32x4 %0, %1, %2, %3
  %5 = GetRegister R2
  %6 = GetRegister R3
  %7 = CompositeConstructU32x2 %5, %6
  %8 = GetRegister R6
  %9 = GetRegister R7
  %10 = CompositeConstructU32x2 %8, %9
  %11 = TensorMulAdd %4, %7, %10 (flags 0x10)
  %12 = CompositeExtractU32x2 %11, #0x0
  SetRegister R8, %12
  %13 = CompositeExtractU32x2 %11, #0x1
  SetRegister R9, %13
"
        );
        let short = encode(
            Opcode::Hmma,
            Form::Fixed,
            f(16, 8) | f(24, 4) | f(32, 3) | f(64, 0xff),
        );
        let ir = translate(&[short]).unwrap();
        assert!(
            ir.contains("CompositeConstructU32x4 %0, %1, #0x0, #0x0"),
            "{ir}"
        );
        assert!(ir.contains("CompositeConstructU32x2 %3, #0x0"), "{ir}");
        assert!(ir.contains("CompositeConstructU32x2 #0x0, #0x0"), "{ir}");
        assert!(ir.contains("TensorMulAdd %2, %4, %5 (flags 0x8)"), "{ir}");
    }

    /// A warp loads its matrices with LDSM, transposes one with MOVM, multiplies them with HMMA
    /// and stores the product: the whole program compiles to valid SPIR-V.
    #[test]
    fn tensor_program_compiles() {
        let code = [
            encode(
                Opcode::Ldsm,
                Form::UniformAddress,
                f(16, 4) | f(24, 2) | f(72, 2),
            ),
            encode(Opcode::Movm, Form::Fixed, f(16, 8) | f(24, 4)),
            encode(Opcode::Movm, Form::Fixed, f(16, 9) | f(24, 5)),
            encode(
                Opcode::Hmma,
                Form::Fixed,
                f(16, 12) | f(24, 4) | f(32, 8) | f(64, 0xff) | f(75, 1),
            ),
            encode(
                Opcode::Sts,
                Form::UniformAddress,
                f(24, 2) | f(32, 12) | f(73, 5),
            ),
            encode(Opcode::Exit, Form::Fixed, 7 << 87),
        ];
        let mut env = TestEnv::new(&code);
        env.shared_memory = 0x1000;
        env.workgroup = [32, 1, 1];
        let profile = Profile::default();
        let host = HostTranslateInfo::default();
        let options = TranslateOptions {
            profile: &profile,
            host: &host,
            resolution_scale: ResolutionScale::X1,
        };
        let program = crate::translate::translate(&env, options).unwrap();
        let ir = program.ir.dump();
        assert!(ir.contains("TensorMulAdd"), "{ir}");
        assert!(ir.contains("LoadSharedU32"), "{ir}");
        let module = crate::module::compile(
            &env,
            options,
            &RuntimeInfo::default(),
            &mut Bindings::default(),
        )
        .unwrap();
        validate("ampere_tensor", &profile, &module.spirv);
    }

    #[test]
    fn unsupported_matrix_forms_are_rejected() {
        let hmma = |fields| {
            encode(
                Opcode::Hmma,
                Form::Fixed,
                f(16, 8) | f(24, 4) | f(32, 2) | fields,
            )
        };
        for insn in [
            hmma(f(76, 1)),
            hmma(f(40, 1)),
            encode(
                Opcode::Hmma,
                Form::Fixed,
                f(16, 8) | f(24, 6) | f(32, 2) | f(75, 1),
            ),
            hmma(f(64, 3)),
            hmma(f(96, 1)),
            encode(Opcode::Hmma, Form::Fixed, f(16, 9)),
            encode(Opcode::Movm, Form::Fixed, f(32, 1)),
            encode(Opcode::Movm, Form::Fixed, f(72, 1)),
        ] {
            assert!(translate(&[insn]).is_err(), "{:#034x}", insn.0);
        }
    }
}
