//! Emission of the integer, floating-point, conversion and bit-cast families: every opcode
//! in one program per host profile, validated with `spirv-val`, plus the profile-dependent
//! paths (int64, float16, int8/int16, driver workarounds, float controls).

use rspirv::dr;
use rspirv::spirv::{Capability, Decoration, ExecutionMode, GlslStd450Op, Op};

use super::{Synthetic, count_ops, emit};
use crate::backend::Bindings;
use crate::backend::spirv::{EmitError, emit_spirv};
use crate::ir::{BlockId, FmzMode, FpControl, Opcode, SyntaxNode, Type, Value};
use crate::maxwell::translate::tests::TestEnv;
use crate::opt;
use crate::translate::TranslatedProgram;
use crate::{Profile, RuntimeInfo, ShaderInfo, ShaderStage};

use Opcode::*;

const ARITHMETIC: [Opcode; 198] = [
    IAdd32,
    IAdd64,
    ISub32,
    ISub64,
    IMul32,
    SDiv32,
    UDiv32,
    INeg32,
    INeg64,
    IAbs32,
    IAbs64,
    ShiftLeftLogical32,
    ShiftLeftLogical64,
    ShiftRightLogical32,
    ShiftRightLogical64,
    ShiftRightArithmetic32,
    ShiftRightArithmetic64,
    BitwiseAnd32,
    BitwiseOr32,
    BitwiseXor32,
    BitFieldInsert,
    BitFieldSExtract,
    BitFieldUExtract,
    BitReverse32,
    BitCount32,
    BitwiseNot32,
    FindSMsb32,
    FindUMsb32,
    SMin32,
    UMin32,
    SMax32,
    UMax32,
    SClamp32,
    UClamp32,
    SLessThan,
    ULessThan,
    IEqual,
    SLessThanEqual,
    ULessThanEqual,
    SGreaterThan,
    UGreaterThan,
    INotEqual,
    SGreaterThanEqual,
    UGreaterThanEqual,
    FPAbs16,
    FPAbs32,
    FPAbs64,
    FPAdd16,
    FPAdd32,
    FPAdd64,
    FPFma16,
    FPFma32,
    FPFma64,
    FPMax32,
    FPMax64,
    FPMin32,
    FPMin64,
    FPMul16,
    FPMul32,
    FPMul64,
    FPNeg16,
    FPNeg32,
    FPNeg64,
    FPRecip32,
    FPRecip64,
    FPRecipSqrt32,
    FPRecipSqrt64,
    FPSqrt,
    FPSin,
    FPExp2,
    FPTanh,
    FPCos,
    FPLog2,
    FPSaturate16,
    FPSaturate32,
    FPSaturate64,
    FPClamp16,
    FPClamp32,
    FPClamp64,
    FPRoundEven16,
    FPRoundEven32,
    FPRoundEven64,
    FPFloor16,
    FPFloor32,
    FPFloor64,
    FPCeil16,
    FPCeil32,
    FPCeil64,
    FPTrunc16,
    FPTrunc32,
    FPTrunc64,
    FPOrdEqual16,
    FPOrdEqual32,
    FPOrdEqual64,
    FPUnordEqual16,
    FPUnordEqual32,
    FPUnordEqual64,
    FPOrdNotEqual16,
    FPOrdNotEqual32,
    FPOrdNotEqual64,
    FPUnordNotEqual16,
    FPUnordNotEqual32,
    FPUnordNotEqual64,
    FPOrdLessThan16,
    FPOrdLessThan32,
    FPOrdLessThan64,
    FPUnordLessThan16,
    FPUnordLessThan32,
    FPUnordLessThan64,
    FPOrdGreaterThan16,
    FPOrdGreaterThan32,
    FPOrdGreaterThan64,
    FPUnordGreaterThan16,
    FPUnordGreaterThan32,
    FPUnordGreaterThan64,
    FPOrdLessThanEqual16,
    FPOrdLessThanEqual32,
    FPOrdLessThanEqual64,
    FPUnordLessThanEqual16,
    FPUnordLessThanEqual32,
    FPUnordLessThanEqual64,
    FPOrdGreaterThanEqual16,
    FPOrdGreaterThanEqual32,
    FPOrdGreaterThanEqual64,
    FPUnordGreaterThanEqual16,
    FPUnordGreaterThanEqual32,
    FPUnordGreaterThanEqual64,
    FPIsNan16,
    FPIsNan32,
    FPIsNan64,
    ConvertS16F16,
    ConvertS16F32,
    ConvertS16F64,
    ConvertS32F16,
    ConvertS32F32,
    ConvertS32F64,
    ConvertS64F16,
    ConvertS64F32,
    ConvertS64F64,
    ConvertU16F16,
    ConvertU16F32,
    ConvertU16F64,
    ConvertU32F16,
    ConvertU32F32,
    ConvertU32F64,
    ConvertU64F16,
    ConvertU64F32,
    ConvertU64F64,
    ConvertU64U32,
    ConvertU32U64,
    ConvertF16F32,
    ConvertF32F16,
    ConvertF32F64,
    ConvertF64F32,
    ConvertF16S8,
    ConvertF16S16,
    ConvertF16S32,
    ConvertF16S64,
    ConvertF16U8,
    ConvertF16U16,
    ConvertF16U32,
    ConvertF16U64,
    ConvertF32S8,
    ConvertF32S16,
    ConvertF32S32,
    ConvertF32S64,
    ConvertF32U8,
    ConvertF32U16,
    ConvertF32U32,
    ConvertF32U64,
    ConvertF64S8,
    ConvertF64S16,
    ConvertF64S32,
    ConvertF64S64,
    ConvertF64U8,
    ConvertF64U16,
    ConvertF64U32,
    ConvertF64U64,
    ConvertU16U32,
    ConvertU32U16,
    ConvertU8U32,
    ConvertU32U8,
    ConvertS32S8,
    ConvertS32S16,
    BitCastU16F16,
    BitCastU32F32,
    BitCastU64F64,
    BitCastF16U16,
    BitCastF32U32,
    BitCastF64U64,
    PackUint2x32,
    UnpackUint2x32,
    PackFloat2x16,
    UnpackFloat2x16,
    PackHalf2x16,
    UnpackHalf2x16,
    PackDouble2x32,
    UnpackDouble2x32,
];

const NO_CONTRACTION: FpControl = FpControl {
    no_contraction: true,
    rounding: crate::ir::FpRounding::DontCare,
    fmz_mode: FmzMode::DontCare,
};

/// A host with every optional integer and float width.
fn native() -> Profile {
    Profile {
        support_int8: true,
        support_int16: true,
        support_int64: true,
        support_float16: true,
        support_float64: true,
        ..Profile::default()
    }
}

/// An operand of type `ty`: an immediate where the IR has one, else a conversion or a
/// composite of immediates. U64 values come from PackUint2x32, as in translated programs, since
/// int64 lowering has no U32x2 form of a U64 immediate.
fn operand(s: &mut Synthetic, b: BlockId, ty: Type) -> Value {
    let half = |s: &mut Synthetic| s.inst(b, ConvertF16F32, &[Value::F32(0.75)]);
    match ty {
        t if t == Type::U1 => Value::U1(true),
        t if t == Type::U32 => Value::U32(0x8000_1234),
        t if t == Type::U64 => {
            let pair = operand(s, b, Type::U32X2);
            s.inst(b, PackUint2x32, &[pair])
        }
        t if t == Type::F32 => Value::F32(1.5),
        t if t == Type::F64 => Value::F64(-2.5),
        t if t == Type::F16 => half(s),
        t if t == Type::U16 => s.inst(b, ConvertU16U32, &[Value::U32(0xfffe)]),
        t if t == Type::U8 => s.inst(b, ConvertU8U32, &[Value::U32(0x80)]),
        t if t == Type::U32X2 => {
            s.inst(b, CompositeConstructU32x2, &[Value::U32(1), Value::U32(2)])
        }
        t if t == Type::F32X2 => s.inst(
            b,
            CompositeConstructF32x2,
            &[Value::F32(1.0), Value::F32(-1.0)],
        ),
        t if t == Type::F16X2 => {
            let (x, y) = (half(s), half(s));
            s.inst(b, CompositeConstructF16x2, &[x, y])
        }
        t => panic!("no test operand of type {t}"),
    }
}

fn is_contractible(op: Opcode) -> bool {
    matches!(
        op,
        FPAdd16 | FPAdd32 | FPAdd64 | FPMul16 | FPMul32 | FPMul64 | FPFma16 | FPFma32 | FPFma64
    )
}

/// One compute program running `ops` on immediates. Contractible float operations are marked
/// no-contraction, and the flag producers carry every pseudo-instruction they can define.
fn program(ops: &[Opcode]) -> TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    for &op in ops {
        let args: Vec<Value> = op
            .arg_types()
            .iter()
            .map(|&ty| operand(&mut s, b, ty))
            .collect();
        let flags = if is_contractible(op) {
            NO_CONTRACTION.to_flags()
        } else {
            0
        };
        let value = s.inst_flags(b, op, &args, flags);
        if op == IAdd32 {
            s.inst(b, GetCarryFromOp, &[value]);
            s.inst(b, GetOverflowFromOp, &[value]);
        }
        if matches!(
            op,
            IAdd32
                | BitwiseAnd32
                | BitwiseOr32
                | BitwiseXor32
                | BitFieldSExtract
                | BitFieldUExtract
                | SClamp32
                | UClamp32
        ) {
            s.inst(b, GetZeroFromOp, &[value]);
            s.inst(b, GetSignFromOp, &[value]);
        }
    }
    s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        ShaderInfo::default(),
    )
}

/// Runs the lowering passes a host without `profile`'s widths needs, then collects the shader
/// info the way translation does.
fn lower_for(profile: &Profile, mut program: TranslatedProgram) -> TranslatedProgram {
    if !profile.support_float16 {
        opt::lower_fp16_to_fp32(&mut program.ir);
    }
    if !profile.support_int64 {
        opt::lower_int64_to_int32(&mut program.ir).expect("int64 lowering");
    }
    opt::collect_shader_info(&TestEnv::new(&[]), &program.ir, &mut program.info);
    program
}

fn emit_ops(name: &str, profile: &Profile, ops: &[Opcode]) -> dr::Module {
    let program = lower_for(profile, program(ops));
    emit(name, profile, &RuntimeInfo::default(), &program)
}

fn has_capability(module: &dr::Module, capability: Capability) -> bool {
    module
        .capabilities
        .iter()
        .any(|i| i.operands.first() == Some(&dr::Operand::Capability(capability)))
}

fn count_glsl(module: &dr::Module, op: GlslStd450Op) -> usize {
    module
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter(|i| {
            i.class.opcode == Op::ExtInst
                && i.operands.get(1) == Some(&dr::Operand::LiteralExtInstInteger(op as u32))
        })
        .count()
}

fn count_decorations(module: &dr::Module, decoration: Decoration) -> usize {
    module
        .annotations
        .iter()
        .filter(|a| a.operands.get(1) == Some(&dr::Operand::Decoration(decoration)))
        .count()
}

fn has_execution_mode(module: &dr::Module, mode: ExecutionMode, width: u32) -> bool {
    module.execution_modes.iter().any(|i| {
        i.operands.get(1) == Some(&dr::Operand::ExecutionMode(mode))
            && i.operands.get(2) == Some(&dr::Operand::LiteralBit32(width))
    })
}

#[test]
fn every_arithmetic_opcode_on_a_host_with_every_width() {
    let module = emit_ops("arith_native", &native(), &ARITHMETIC);
    for capability in [
        Capability::Int8,
        Capability::Int16,
        Capability::Int64,
        Capability::Float16,
        Capability::Float64,
    ] {
        assert!(has_capability(&module, capability), "{capability:?}");
    }
    assert_eq!(count_ops(&module, Op::IAddCarry), 1);
    assert_eq!(
        count_decorations(&module, Decoration::NoContraction),
        ARITHMETIC.iter().filter(|&&op| is_contractible(op)).count()
    );
}

#[test]
fn eight_and_sixteen_bit_integers_live_in_u32_without_host_support() {
    let profile = Profile {
        support_int8: false,
        support_int16: false,
        ..native()
    };
    let module = emit_ops("arith_no_int8_int16", &profile, &ARITHMETIC);
    assert!(!has_capability(&module, Capability::Int8));
    assert!(!has_capability(&module, Capability::Int16));
    assert!(count_ops(&module, Op::BitFieldSExtract) > 1);
}

#[test]
fn sixteen_bit_moves_the_shader_info_misses_stay_in_u32() {
    let module = emit_ops(
        "arith_untracked_int16",
        &native(),
        &[
            ConvertU16U32,
            ConvertU32U16,
            ConvertS32S16,
            ConvertU8U32,
            ConvertU32U8,
        ],
    );
    assert!(!has_capability(&module, Capability::Int16));
    assert_eq!(count_ops(&module, Op::UConvert), 0);
}

#[test]
fn every_arithmetic_opcode_after_int64_lowering() {
    let profile = Profile {
        support_int64: false,
        ..native()
    };
    let module = emit_ops("arith_no_int64", &profile, &ARITHMETIC);
    assert!(!has_capability(&module, Capability::Int64));
    assert!(
        count_ops(&module, Op::IAddCarry) > 1,
        "64-bit adds carry in 32 bits"
    );
}

/// The 16-bit bit casts keep their F16 operand through fp16 lowering, which leaves them alone.
const FP16_LOWERING_MISSES: [Opcode; 2] = [BitCastU16F16, BitCastF16U16];

#[test]
fn every_arithmetic_opcode_after_float16_lowering() {
    let profile = Profile {
        support_float16: false,
        ..native()
    };
    let ops: Vec<Opcode> = ARITHMETIC
        .into_iter()
        .filter(|op| !FP16_LOWERING_MISSES.contains(op))
        .collect();
    let module = emit_ops("arith_no_fp16", &profile, &ops);
    assert!(!has_capability(&module, Capability::Float16));
}

#[test]
fn every_arithmetic_opcode_on_a_minimal_host() {
    let ops: Vec<Opcode> = ARITHMETIC
        .into_iter()
        .filter(|op| !FP16_LOWERING_MISSES.contains(op))
        .collect();
    let profile = Profile {
        support_float64: true,
        ..Profile::default()
    };
    let module = emit_ops("arith_minimal", &profile, &ops);
    for capability in [
        Capability::Int8,
        Capability::Int16,
        Capability::Int64,
        Capability::Float16,
    ] {
        assert!(!has_capability(&module, capability), "{capability:?}");
    }
}

#[test]
fn every_arithmetic_opcode_with_driver_workarounds() {
    let profile = Profile {
        has_broken_signed_operations: true,
        has_broken_spirv_clamp: true,
        ignore_nan_fp_comparisons: true,
        ..native()
    };
    let module = emit_ops("arith_workarounds", &profile, &ARITHMETIC);
    assert_eq!(count_glsl(&module, GlslStd450Op::SClamp), 0);
    assert_eq!(count_glsl(&module, GlslStd450Op::UClamp), 0);
    let baseline = emit_ops("arith_no_workarounds", &native(), &ARITHMETIC);
    let nan_checked_compares = 3 * 6;
    assert_eq!(
        count_ops(&module, Op::IsNan) - count_ops(&baseline, Op::IsNan),
        2 * nan_checked_compares,
        "two IsNan per unordered compare (but not-equal) and ordered not-equal"
    );
    let signed_only = Profile {
        has_broken_signed_operations: true,
        ..native()
    };
    let module = emit_ops("arith_broken_signed", &signed_only, &[SClamp32, SMin32]);
    assert_eq!(count_glsl(&module, GlslStd450Op::SClamp), 1);
}

#[test]
fn float_min_max_and_clamp_are_nan_defined() {
    for broken_clamp in [false, true] {
        let profile = Profile {
            has_broken_spirv_clamp: broken_clamp,
            ..native()
        };
        let module = emit_ops(
            "arith_nan_min_max",
            &profile,
            &[FPMin32, FPMax32, FPMin64, FPMax64, FPSaturate32, FPClamp32],
        );
        for op in [
            GlslStd450Op::FMin,
            GlslStd450Op::FMax,
            GlslStd450Op::FClamp,
            GlslStd450Op::NClamp,
        ] {
            assert_eq!(count_glsl(&module, op), 0, "{op:?}");
        }
        assert_eq!(count_glsl(&module, GlslStd450Op::NMin), 4);
        assert_eq!(count_glsl(&module, GlslStd450Op::NMax), 4);
    }
}

fn denorm_program(op: Opcode, fmz_mode: FmzMode) -> TranslatedProgram {
    let mut s = Synthetic::new();
    let b = s.block();
    let ty = op.arg_types()[0];
    let (x, y) = (operand(&mut s, b, ty), operand(&mut s, b, ty));
    let control = FpControl {
        fmz_mode,
        ..FpControl::default()
    };
    s.inst_flags(b, op, &[x, y], control.to_flags());
    let mut program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        ShaderInfo::default(),
    );
    opt::collect_shader_info(&TestEnv::new(&[]), &program.ir, &mut program.info);
    program
}

#[test]
fn denormal_modes_become_float_controls_execution_modes() {
    let profile = Profile {
        supported_spirv: 0x0001_0400,
        support_float_controls: true,
        support_separate_denorm_behavior: true,
        support_fp16_denorm_flush: true,
        support_fp32_denorm_flush: true,
        support_fp16_denorm_preserve: true,
        support_fp32_denorm_preserve: true,
        ..native()
    };
    let runtime = RuntimeInfo::default();
    for (op, width) in [(FPAdd32, 32), (FPMul16, 16)] {
        let flush = emit(
            "arith_denorm_flush",
            &profile,
            &runtime,
            &denorm_program(op, FmzMode::Ftz),
        );
        assert!(has_execution_mode(
            &flush,
            ExecutionMode::DenormFlushToZero,
            width
        ));
        assert!(!has_execution_mode(
            &flush,
            ExecutionMode::DenormPreserve,
            width
        ));
        let preserve = emit(
            "arith_denorm_preserve",
            &profile,
            &runtime,
            &denorm_program(op, FmzMode::None),
        );
        assert!(has_execution_mode(
            &preserve,
            ExecutionMode::DenormPreserve,
            width
        ));
        assert!(!has_execution_mode(
            &preserve,
            ExecutionMode::DenormFlushToZero,
            width
        ));
    }
    let without = emit(
        "arith_denorm_no_float_controls",
        &native(),
        &runtime,
        &denorm_program(FPAdd32, FmzMode::Ftz),
    );
    assert!(without.execution_modes.iter().all(|i| {
        i.operands.get(1)
            != Some(&dr::Operand::ExecutionMode(
                ExecutionMode::DenormFlushToZero,
            ))
    }));
}

#[test]
fn a_wide_operation_the_shader_info_missed_is_an_error() {
    let mut s = Synthetic::new();
    let b = s.block();
    s.inst(b, IAbs64, &[Value::U64(1)]);
    let program = s.finish(
        vec![SyntaxNode::Block(b), SyntaxNode::Return],
        ShaderStage::Compute,
        ShaderInfo::default(),
    );
    let error = emit_spirv(
        &native(),
        &RuntimeInfo::default(),
        &program,
        &mut Bindings::default(),
    )
    .expect_err("u64 is not declared");
    assert!(matches!(error, EmitError::InvalidIr(_)), "{error}");
}
