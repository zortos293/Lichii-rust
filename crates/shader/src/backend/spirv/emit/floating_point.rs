//! Floating-point arithmetic and comparisons, from the C++ `emit_spirv_floating_point.cpp`.

use rspirv::spirv::Word;

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};

pub(super) fn fp_abs16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPAbs16))
}

pub(super) fn fp_abs32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPAbs32))
}

pub(super) fn fp_abs64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPAbs64))
}

pub(super) fn fp_add16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPAdd16))
}

pub(super) fn fp_add32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPAdd32))
}

pub(super) fn fp_add64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPAdd64))
}

pub(super) fn fp_fma16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPFma16))
}

pub(super) fn fp_fma32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPFma32))
}

pub(super) fn fp_fma64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPFma64))
}

pub(super) fn fp_max32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPMax32))
}

pub(super) fn fp_max64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPMax64))
}

pub(super) fn fp_min32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPMin32))
}

pub(super) fn fp_min64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPMin64))
}

pub(super) fn fp_mul16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPMul16))
}

pub(super) fn fp_mul32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPMul32))
}

pub(super) fn fp_mul64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPMul64))
}

pub(super) fn fp_neg16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPNeg16))
}

pub(super) fn fp_neg32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPNeg32))
}

pub(super) fn fp_neg64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPNeg64))
}

pub(super) fn fp_recip32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPRecip32))
}

pub(super) fn fp_recip64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPRecip64))
}

pub(super) fn fp_recip_sqrt32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPRecipSqrt32))
}

pub(super) fn fp_recip_sqrt64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPRecipSqrt64))
}

pub(super) fn fp_sqrt(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPSqrt))
}

pub(super) fn fp_sin(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPSin))
}

pub(super) fn fp_exp2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPExp2))
}

pub(super) fn fp_tanh(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPTanh))
}

pub(super) fn fp_cos(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPCos))
}

pub(super) fn fp_log2(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPLog2))
}

pub(super) fn fp_saturate16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPSaturate16))
}

pub(super) fn fp_saturate32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPSaturate32))
}

pub(super) fn fp_saturate64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPSaturate64))
}

pub(super) fn fp_clamp16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPClamp16))
}

pub(super) fn fp_clamp32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPClamp32))
}

pub(super) fn fp_clamp64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPClamp64))
}

pub(super) fn fp_round_even16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPRoundEven16))
}

pub(super) fn fp_round_even32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPRoundEven32))
}

pub(super) fn fp_round_even64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPRoundEven64))
}

pub(super) fn fp_floor16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPFloor16))
}

pub(super) fn fp_floor32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPFloor32))
}

pub(super) fn fp_floor64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPFloor64))
}

pub(super) fn fp_ceil16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPCeil16))
}

pub(super) fn fp_ceil32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPCeil32))
}

pub(super) fn fp_ceil64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPCeil64))
}

pub(super) fn fp_trunc16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPTrunc16))
}

pub(super) fn fp_trunc32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPTrunc32))
}

pub(super) fn fp_trunc64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPTrunc64))
}

pub(super) fn fp_ord_equal16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdEqual16))
}

pub(super) fn fp_ord_equal32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdEqual32))
}

pub(super) fn fp_ord_equal64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdEqual64))
}

pub(super) fn fp_unord_equal16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordEqual16))
}

pub(super) fn fp_unord_equal32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordEqual32))
}

pub(super) fn fp_unord_equal64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordEqual64))
}

pub(super) fn fp_ord_not_equal16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdNotEqual16))
}

pub(super) fn fp_ord_not_equal32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdNotEqual32))
}

pub(super) fn fp_ord_not_equal64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdNotEqual64))
}

pub(super) fn fp_unord_not_equal16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordNotEqual16))
}

pub(super) fn fp_unord_not_equal32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordNotEqual32))
}

pub(super) fn fp_unord_not_equal64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordNotEqual64))
}

pub(super) fn fp_ord_less_than16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdLessThan16))
}

pub(super) fn fp_ord_less_than32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdLessThan32))
}

pub(super) fn fp_ord_less_than64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdLessThan64))
}

pub(super) fn fp_unord_less_than16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordLessThan16))
}

pub(super) fn fp_unord_less_than32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordLessThan32))
}

pub(super) fn fp_unord_less_than64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordLessThan64))
}

pub(super) fn fp_ord_greater_than16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdGreaterThan16))
}

pub(super) fn fp_ord_greater_than32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdGreaterThan32))
}

pub(super) fn fp_ord_greater_than64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdGreaterThan64))
}

pub(super) fn fp_unord_greater_than16(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordGreaterThan16))
}

pub(super) fn fp_unord_greater_than32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordGreaterThan32))
}

pub(super) fn fp_unord_greater_than64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordGreaterThan64))
}

pub(super) fn fp_ord_less_than_equal16(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdLessThanEqual16))
}

pub(super) fn fp_ord_less_than_equal32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdLessThanEqual32))
}

pub(super) fn fp_ord_less_than_equal64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdLessThanEqual64))
}

pub(super) fn fp_unord_less_than_equal16(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordLessThanEqual16))
}

pub(super) fn fp_unord_less_than_equal32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordLessThanEqual32))
}

pub(super) fn fp_unord_less_than_equal64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordLessThanEqual64))
}

pub(super) fn fp_ord_greater_than_equal16(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdGreaterThanEqual16))
}

pub(super) fn fp_ord_greater_than_equal32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdGreaterThanEqual32))
}

pub(super) fn fp_ord_greater_than_equal64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPOrdGreaterThanEqual64))
}

pub(super) fn fp_unord_greater_than_equal16(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordGreaterThanEqual16))
}

pub(super) fn fp_unord_greater_than_equal32(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordGreaterThanEqual32))
}

pub(super) fn fp_unord_greater_than_equal64(
    _ctx: &mut EmitContext<'_>,
    _inst: InstId,
) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPUnordGreaterThanEqual64))
}

pub(super) fn fp_is_nan16(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPIsNan16))
}

pub(super) fn fp_is_nan32(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPIsNan32))
}

pub(super) fn fp_is_nan64(_ctx: &mut EmitContext<'_>, _inst: InstId) -> EmitResult<Word> {
    Err(EmitError::Unimplemented(Opcode::FPIsNan64))
}
