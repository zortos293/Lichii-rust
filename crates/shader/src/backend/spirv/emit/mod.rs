//! Opcode dispatch. One exhaustive match sends every IR opcode to the function of its
//! emission family, one file per family, so a family is filled in by editing only its file.
//!
//! The family convention: `pub(super) fn <opcode_in_snake_case>(ctx: &mut EmitContext<'_>,
//! inst: InstId) -> EmitResult<Word>` for an opcode with a result, `-> EmitResult<()>` for a
//! void one. The function reads its arguments with `ctx.arg`, `ctx.def_arg` and
//! `ctx.imm_u32`, its flags with `ctx.flags`, emits through `ctx.b` and returns the result id,
//! which the dispatcher records as the definition of `inst`. Pseudo-instructions attached to
//! it (`ctx.program.pseudo_op`) are defined by the family with `ctx.set_def`.

pub(super) mod atomic;
pub(super) mod barriers;
pub(super) mod bitwise_conversion;
pub(super) mod composite;
pub(super) mod context_get_set;
pub(super) mod control_flow;
pub(super) mod convert;
pub(super) mod floating_point;
pub(super) mod image;
pub(super) mod image_atomic;
pub(super) mod integer;
pub(super) mod logical;
pub(super) mod memory;
pub(super) mod mesh;
pub(super) mod select;
pub(super) mod shared_memory;
pub(super) mod special;
pub(super) mod tensor;
pub(super) mod undefined;
pub(super) mod warp;

use super::context::EmitContext;
use super::{EmitError, EmitResult, driver};
use crate::ir::{InstId, Opcode, Type};

/// Emits one instruction of the block being built.
pub(super) fn emit_inst(ctx: &mut EmitContext<'_>, inst: InstId) -> EmitResult<()> {
    let op = ctx.program.op(inst);
    if ctx.shared_memory_u32 == 0 && is_shared_memory_access(op) {
        crate::fallback::note("shared_memory", "no_shared_memory_declared");
        if op.ty() != Type::VOID {
            let ty = ctx.declared_type(op.ty())?;
            let zero = ctx.const_null(ty);
            ctx.set_def(inst, zero);
        }
        return Ok(());
    }
    let id = match op {
        Opcode::Phi => driver::emit_phi(ctx, inst)?,
        Opcode::Identity | Opcode::ConditionRef => ctx.def_arg(inst, 0)?,
        Opcode::Void | Opcode::Reference => return Ok(()),
        Opcode::PhiMove => {
            return Err(EmitError::InvalidIr(
                "PhiMove survived SSA construction".into(),
            ));
        }
        Opcode::GetZeroFromOp
        | Opcode::GetSignFromOp
        | Opcode::GetCarryFromOp
        | Opcode::GetOverflowFromOp
        | Opcode::GetSparseFromOp
        | Opcode::GetInBoundsFromOp => {
            if ctx.is_defined(inst) {
                return Ok(());
            }
            return Err(EmitError::InvalidIr(format!(
                "{op:?} whose producer did not define it"
            )));
        }
        Opcode::IsbeRead => mesh::isbe_read(ctx, inst)?,
        Opcode::IsbeWriteAttribute => return mesh::isbe_write_attribute(ctx, inst),
        Opcode::IsbeWriteMapU8 => return mesh::isbe_write_map_u8(ctx, inst),
        Opcode::IsbeWriteMapU32 => return mesh::isbe_write_map_u32(ctx, inst),
        Opcode::TensorMulAdd => tensor::tensor_mul_add(ctx, inst, false)?,
        Opcode::TensorMulAddF32 => tensor::tensor_mul_add(ctx, inst, true)?,
        Opcode::Prologue => return special::prologue(ctx, inst),
        Opcode::Epilogue => return special::epilogue(ctx, inst),
        Opcode::Join => return control_flow::join(ctx, inst),
        Opcode::DemoteToHelperInvocation => {
            return control_flow::demote_to_helper_invocation(ctx, inst);
        }
        Opcode::EmitVertex => return special::emit_vertex(ctx, inst),
        Opcode::EndPrimitive => return special::end_primitive(ctx, inst),
        Opcode::Barrier => return barriers::barrier(ctx, inst),
        Opcode::WorkgroupMemoryBarrier => return barriers::workgroup_memory_barrier(ctx, inst),
        Opcode::DeviceMemoryBarrier => return barriers::device_memory_barrier(ctx, inst),
        Opcode::GetRegister => context_get_set::get_register(ctx, inst)?,
        Opcode::SetRegister => return context_get_set::set_register(ctx, inst),
        Opcode::GetPred => context_get_set::get_pred(ctx, inst)?,
        Opcode::SetPred => return context_get_set::set_pred(ctx, inst),
        Opcode::GetGotoVariable => context_get_set::get_goto_variable(ctx, inst)?,
        Opcode::SetGotoVariable => return context_get_set::set_goto_variable(ctx, inst),
        Opcode::GetIndirectBranchVariable => {
            context_get_set::get_indirect_branch_variable(ctx, inst)?
        }
        Opcode::SetIndirectBranchVariable => {
            return context_get_set::set_indirect_branch_variable(ctx, inst);
        }
        Opcode::GetCbufU8 => context_get_set::get_cbuf_u8(ctx, inst)?,
        Opcode::GetCbufS8 => context_get_set::get_cbuf_s8(ctx, inst)?,
        Opcode::GetCbufU16 => context_get_set::get_cbuf_u16(ctx, inst)?,
        Opcode::GetCbufS16 => context_get_set::get_cbuf_s16(ctx, inst)?,
        Opcode::GetCbufU32 => context_get_set::get_cbuf_u32(ctx, inst)?,
        Opcode::GetCbufF32 => context_get_set::get_cbuf_f32(ctx, inst)?,
        Opcode::GetCbufU32x2 => context_get_set::get_cbuf_u32x2(ctx, inst)?,
        Opcode::GetAttribute => context_get_set::get_attribute(ctx, inst)?,
        Opcode::GetAttributeU32 => context_get_set::get_attribute_u32(ctx, inst)?,
        Opcode::SetAttribute => return context_get_set::set_attribute(ctx, inst),
        Opcode::GetAttributeIndexed => context_get_set::get_attribute_indexed(ctx, inst)?,
        Opcode::InterpolateAtOffset => context_get_set::interpolate_at_offset(ctx, inst)?,
        Opcode::SetAttributeIndexed => return context_get_set::set_attribute_indexed(ctx, inst),
        Opcode::GetPatch => context_get_set::get_patch(ctx, inst)?,
        Opcode::SetPatch => return context_get_set::set_patch(ctx, inst),
        Opcode::SetFragColor => return context_get_set::set_frag_color(ctx, inst),
        Opcode::SetSampleMask => return context_get_set::set_sample_mask(ctx, inst),
        Opcode::SetFragDepth => return context_get_set::set_frag_depth(ctx, inst),
        Opcode::GetZFlag => context_get_set::get_z_flag(ctx, inst)?,
        Opcode::GetSFlag => context_get_set::get_s_flag(ctx, inst)?,
        Opcode::GetCFlag => context_get_set::get_c_flag(ctx, inst)?,
        Opcode::GetOFlag => context_get_set::get_o_flag(ctx, inst)?,
        Opcode::SetZFlag => return context_get_set::set_z_flag(ctx, inst),
        Opcode::SetSFlag => return context_get_set::set_s_flag(ctx, inst),
        Opcode::SetCFlag => return context_get_set::set_c_flag(ctx, inst),
        Opcode::SetOFlag => return context_get_set::set_o_flag(ctx, inst),
        Opcode::WorkgroupId => context_get_set::workgroup_id(ctx, inst)?,
        Opcode::LocalInvocationId => context_get_set::local_invocation_id(ctx, inst)?,
        Opcode::InvocationId => context_get_set::invocation_id(ctx, inst)?,
        Opcode::InvocationInfo => context_get_set::invocation_info(ctx, inst)?,
        Opcode::SampleId => context_get_set::sample_id(ctx, inst)?,
        Opcode::IsHelperInvocation => context_get_set::is_helper_invocation(ctx, inst)?,
        Opcode::WScaleFactorXY => context_get_set::w_scale_factor_xy(ctx, inst)?,
        Opcode::WScaleFactorZ => context_get_set::w_scale_factor_z(ctx, inst)?,
        Opcode::YDirection => context_get_set::y_direction(ctx, inst)?,
        Opcode::ResolutionDownFactor => context_get_set::resolution_down_factor(ctx, inst)?,
        Opcode::RenderArea => context_get_set::render_area(ctx, inst)?,
        Opcode::UndefU1 => undefined::undef_u1(ctx, inst)?,
        Opcode::UndefU8 => undefined::undef_u8(ctx, inst)?,
        Opcode::UndefU16 => undefined::undef_u16(ctx, inst)?,
        Opcode::UndefU32 => undefined::undef_u32(ctx, inst)?,
        Opcode::UndefU64 => undefined::undef_u64(ctx, inst)?,
        Opcode::LoadGlobalU8 => memory::load_global_u8(ctx, inst)?,
        Opcode::LoadGlobalS8 => memory::load_global_s8(ctx, inst)?,
        Opcode::LoadGlobalU16 => memory::load_global_u16(ctx, inst)?,
        Opcode::LoadGlobalS16 => memory::load_global_s16(ctx, inst)?,
        Opcode::LoadGlobal32 => memory::load_global32(ctx, inst)?,
        Opcode::LoadGlobal64 => memory::load_global64(ctx, inst)?,
        Opcode::LoadGlobal128 => memory::load_global128(ctx, inst)?,
        Opcode::WriteGlobalU8 => return memory::write_global_u8(ctx, inst),
        Opcode::WriteGlobalS8 => return memory::write_global_s8(ctx, inst),
        Opcode::WriteGlobalU16 => return memory::write_global_u16(ctx, inst),
        Opcode::WriteGlobalS16 => return memory::write_global_s16(ctx, inst),
        Opcode::WriteGlobal32 => return memory::write_global32(ctx, inst),
        Opcode::WriteGlobal64 => return memory::write_global64(ctx, inst),
        Opcode::WriteGlobal128 => return memory::write_global128(ctx, inst),
        Opcode::LoadStorageU8 => memory::load_storage_u8(ctx, inst)?,
        Opcode::LoadStorageS8 => memory::load_storage_s8(ctx, inst)?,
        Opcode::LoadStorageU16 => memory::load_storage_u16(ctx, inst)?,
        Opcode::LoadStorageS16 => memory::load_storage_s16(ctx, inst)?,
        Opcode::LoadStorage32 => memory::load_storage32(ctx, inst)?,
        Opcode::LoadStorage64 => memory::load_storage64(ctx, inst)?,
        Opcode::LoadStorage128 => memory::load_storage128(ctx, inst)?,
        Opcode::WriteStorageU8 => return memory::write_storage_u8(ctx, inst),
        Opcode::WriteStorageS8 => return memory::write_storage_s8(ctx, inst),
        Opcode::WriteStorageU16 => return memory::write_storage_u16(ctx, inst),
        Opcode::WriteStorageS16 => return memory::write_storage_s16(ctx, inst),
        Opcode::WriteStorage32 => return memory::write_storage32(ctx, inst),
        Opcode::WriteStorage64 => return memory::write_storage64(ctx, inst),
        Opcode::WriteStorage128 => return memory::write_storage128(ctx, inst),
        Opcode::LoadLocal => context_get_set::load_local(ctx, inst)?,
        Opcode::WriteLocal => return context_get_set::write_local(ctx, inst),
        Opcode::LoadSharedU8 => shared_memory::load_shared_u8(ctx, inst)?,
        Opcode::LoadSharedS8 => shared_memory::load_shared_s8(ctx, inst)?,
        Opcode::LoadSharedU16 => shared_memory::load_shared_u16(ctx, inst)?,
        Opcode::LoadSharedS16 => shared_memory::load_shared_s16(ctx, inst)?,
        Opcode::LoadSharedU32 => shared_memory::load_shared_u32(ctx, inst)?,
        Opcode::LoadSharedU64 => shared_memory::load_shared_u64(ctx, inst)?,
        Opcode::LoadSharedU128 => shared_memory::load_shared_u128(ctx, inst)?,
        Opcode::WriteSharedU8 => return shared_memory::write_shared_u8(ctx, inst),
        Opcode::WriteSharedU16 => return shared_memory::write_shared_u16(ctx, inst),
        Opcode::WriteSharedU32 => return shared_memory::write_shared_u32(ctx, inst),
        Opcode::WriteSharedU64 => return shared_memory::write_shared_u64(ctx, inst),
        Opcode::WriteSharedU128 => return shared_memory::write_shared_u128(ctx, inst),
        Opcode::CompositeConstructU32x2 => composite::composite_construct_u32x2(ctx, inst)?,
        Opcode::CompositeConstructU32x3 => composite::composite_construct_u32x3(ctx, inst)?,
        Opcode::CompositeConstructU32x4 => composite::composite_construct_u32x4(ctx, inst)?,
        Opcode::CompositeExtractU32x2 => composite::composite_extract_u32x2(ctx, inst)?,
        Opcode::CompositeExtractU32x3 => composite::composite_extract_u32x3(ctx, inst)?,
        Opcode::CompositeExtractU32x4 => composite::composite_extract_u32x4(ctx, inst)?,
        Opcode::CompositeInsertU32x2 => composite::composite_insert_u32x2(ctx, inst)?,
        Opcode::CompositeInsertU32x3 => composite::composite_insert_u32x3(ctx, inst)?,
        Opcode::CompositeInsertU32x4 => composite::composite_insert_u32x4(ctx, inst)?,
        Opcode::CompositeConstructF16x2 => composite::composite_construct_f16x2(ctx, inst)?,
        Opcode::CompositeConstructF16x3 => composite::composite_construct_f16x3(ctx, inst)?,
        Opcode::CompositeConstructF16x4 => composite::composite_construct_f16x4(ctx, inst)?,
        Opcode::CompositeExtractF16x2 => composite::composite_extract_f16x2(ctx, inst)?,
        Opcode::CompositeExtractF16x3 => composite::composite_extract_f16x3(ctx, inst)?,
        Opcode::CompositeExtractF16x4 => composite::composite_extract_f16x4(ctx, inst)?,
        Opcode::CompositeInsertF16x2 => composite::composite_insert_f16x2(ctx, inst)?,
        Opcode::CompositeInsertF16x3 => composite::composite_insert_f16x3(ctx, inst)?,
        Opcode::CompositeInsertF16x4 => composite::composite_insert_f16x4(ctx, inst)?,
        Opcode::CompositeConstructF32x2 => composite::composite_construct_f32x2(ctx, inst)?,
        Opcode::CompositeConstructF32x3 => composite::composite_construct_f32x3(ctx, inst)?,
        Opcode::CompositeConstructF32x4 => composite::composite_construct_f32x4(ctx, inst)?,
        Opcode::CompositeExtractF32x2 => composite::composite_extract_f32x2(ctx, inst)?,
        Opcode::CompositeExtractF32x3 => composite::composite_extract_f32x3(ctx, inst)?,
        Opcode::CompositeExtractF32x4 => composite::composite_extract_f32x4(ctx, inst)?,
        Opcode::CompositeInsertF32x2 => composite::composite_insert_f32x2(ctx, inst)?,
        Opcode::CompositeInsertF32x3 => composite::composite_insert_f32x3(ctx, inst)?,
        Opcode::CompositeInsertF32x4 => composite::composite_insert_f32x4(ctx, inst)?,
        Opcode::CompositeConstructF64x2 => composite::composite_construct_f64x2(ctx, inst)?,
        Opcode::CompositeConstructF64x3 => composite::composite_construct_f64x3(ctx, inst)?,
        Opcode::CompositeConstructF64x4 => composite::composite_construct_f64x4(ctx, inst)?,
        Opcode::CompositeExtractF64x2 => composite::composite_extract_f64x2(ctx, inst)?,
        Opcode::CompositeExtractF64x3 => composite::composite_extract_f64x3(ctx, inst)?,
        Opcode::CompositeExtractF64x4 => composite::composite_extract_f64x4(ctx, inst)?,
        Opcode::CompositeInsertF64x2 => composite::composite_insert_f64x2(ctx, inst)?,
        Opcode::CompositeInsertF64x3 => composite::composite_insert_f64x3(ctx, inst)?,
        Opcode::CompositeInsertF64x4 => composite::composite_insert_f64x4(ctx, inst)?,
        Opcode::SelectU1 => select::select_u1(ctx, inst)?,
        Opcode::SelectU8 => select::select_u8(ctx, inst)?,
        Opcode::SelectU16 => select::select_u16(ctx, inst)?,
        Opcode::SelectU32 => select::select_u32(ctx, inst)?,
        Opcode::SelectU64 => select::select_u64(ctx, inst)?,
        Opcode::SelectF16 => select::select_f16(ctx, inst)?,
        Opcode::SelectF32 => select::select_f32(ctx, inst)?,
        Opcode::SelectF64 => select::select_f64(ctx, inst)?,
        Opcode::BitCastU16F16 => bitwise_conversion::bit_cast_u16_f16(ctx, inst)?,
        Opcode::BitCastU32F32 => bitwise_conversion::bit_cast_u32_f32(ctx, inst)?,
        Opcode::BitCastU64F64 => bitwise_conversion::bit_cast_u64_f64(ctx, inst)?,
        Opcode::BitCastF16U16 => bitwise_conversion::bit_cast_f16_u16(ctx, inst)?,
        Opcode::BitCastF32U32 => bitwise_conversion::bit_cast_f32_u32(ctx, inst)?,
        Opcode::BitCastF64U64 => bitwise_conversion::bit_cast_f64_u64(ctx, inst)?,
        Opcode::PackUint2x32 => bitwise_conversion::pack_uint2x32(ctx, inst)?,
        Opcode::UnpackUint2x32 => bitwise_conversion::unpack_uint2x32(ctx, inst)?,
        Opcode::PackFloat2x16 => bitwise_conversion::pack_float2x16(ctx, inst)?,
        Opcode::UnpackFloat2x16 => bitwise_conversion::unpack_float2x16(ctx, inst)?,
        Opcode::PackHalf2x16 => bitwise_conversion::pack_half2x16(ctx, inst)?,
        Opcode::UnpackHalf2x16 => bitwise_conversion::unpack_half2x16(ctx, inst)?,
        Opcode::PackDouble2x32 => bitwise_conversion::pack_double2x32(ctx, inst)?,
        Opcode::UnpackDouble2x32 => bitwise_conversion::unpack_double2x32(ctx, inst)?,
        Opcode::FPAbs16 => floating_point::fp_abs16(ctx, inst)?,
        Opcode::FPAbs32 => floating_point::fp_abs32(ctx, inst)?,
        Opcode::FPAbs64 => floating_point::fp_abs64(ctx, inst)?,
        Opcode::FPAdd16 => floating_point::fp_add16(ctx, inst)?,
        Opcode::FPAdd32 => floating_point::fp_add32(ctx, inst)?,
        Opcode::FPAdd64 => floating_point::fp_add64(ctx, inst)?,
        Opcode::FPFma16 => floating_point::fp_fma16(ctx, inst)?,
        Opcode::FPFma32 => floating_point::fp_fma32(ctx, inst)?,
        Opcode::FPFma64 => floating_point::fp_fma64(ctx, inst)?,
        Opcode::FPMax32 => floating_point::fp_max32(ctx, inst)?,
        Opcode::FPMax64 => floating_point::fp_max64(ctx, inst)?,
        Opcode::FPMin32 => floating_point::fp_min32(ctx, inst)?,
        Opcode::FPMin64 => floating_point::fp_min64(ctx, inst)?,
        Opcode::FPMul16 => floating_point::fp_mul16(ctx, inst)?,
        Opcode::FPMul32 => floating_point::fp_mul32(ctx, inst)?,
        Opcode::FPMul64 => floating_point::fp_mul64(ctx, inst)?,
        Opcode::FPNeg16 => floating_point::fp_neg16(ctx, inst)?,
        Opcode::FPNeg32 => floating_point::fp_neg32(ctx, inst)?,
        Opcode::FPNeg64 => floating_point::fp_neg64(ctx, inst)?,
        Opcode::FPRecip32 => floating_point::fp_recip32(ctx, inst)?,
        Opcode::FPRecip64 => floating_point::fp_recip64(ctx, inst)?,
        Opcode::FPRecipSqrt32 => floating_point::fp_recip_sqrt32(ctx, inst)?,
        Opcode::FPRecipSqrt64 => floating_point::fp_recip_sqrt64(ctx, inst)?,
        Opcode::FPSqrt => floating_point::fp_sqrt(ctx, inst)?,
        Opcode::FPSin => floating_point::fp_sin(ctx, inst)?,
        Opcode::FPExp2 => floating_point::fp_exp2(ctx, inst)?,
        Opcode::FPTanh => floating_point::fp_tanh(ctx, inst)?,
        Opcode::FPCos => floating_point::fp_cos(ctx, inst)?,
        Opcode::FPLog2 => floating_point::fp_log2(ctx, inst)?,
        Opcode::FPSaturate16 => floating_point::fp_saturate16(ctx, inst)?,
        Opcode::FPSaturate32 => floating_point::fp_saturate32(ctx, inst)?,
        Opcode::FPSaturate64 => floating_point::fp_saturate64(ctx, inst)?,
        Opcode::FPClamp16 => floating_point::fp_clamp16(ctx, inst)?,
        Opcode::FPClamp32 => floating_point::fp_clamp32(ctx, inst)?,
        Opcode::FPClamp64 => floating_point::fp_clamp64(ctx, inst)?,
        Opcode::FPRoundEven16 => floating_point::fp_round_even16(ctx, inst)?,
        Opcode::FPRoundEven32 => floating_point::fp_round_even32(ctx, inst)?,
        Opcode::FPRoundEven64 => floating_point::fp_round_even64(ctx, inst)?,
        Opcode::FPFloor16 => floating_point::fp_floor16(ctx, inst)?,
        Opcode::FPFloor32 => floating_point::fp_floor32(ctx, inst)?,
        Opcode::FPFloor64 => floating_point::fp_floor64(ctx, inst)?,
        Opcode::FPCeil16 => floating_point::fp_ceil16(ctx, inst)?,
        Opcode::FPCeil32 => floating_point::fp_ceil32(ctx, inst)?,
        Opcode::FPCeil64 => floating_point::fp_ceil64(ctx, inst)?,
        Opcode::FPTrunc16 => floating_point::fp_trunc16(ctx, inst)?,
        Opcode::FPTrunc32 => floating_point::fp_trunc32(ctx, inst)?,
        Opcode::FPTrunc64 => floating_point::fp_trunc64(ctx, inst)?,
        Opcode::FPOrdEqual16 => floating_point::fp_ord_equal16(ctx, inst)?,
        Opcode::FPOrdEqual32 => floating_point::fp_ord_equal32(ctx, inst)?,
        Opcode::FPOrdEqual64 => floating_point::fp_ord_equal64(ctx, inst)?,
        Opcode::FPUnordEqual16 => floating_point::fp_unord_equal16(ctx, inst)?,
        Opcode::FPUnordEqual32 => floating_point::fp_unord_equal32(ctx, inst)?,
        Opcode::FPUnordEqual64 => floating_point::fp_unord_equal64(ctx, inst)?,
        Opcode::FPOrdNotEqual16 => floating_point::fp_ord_not_equal16(ctx, inst)?,
        Opcode::FPOrdNotEqual32 => floating_point::fp_ord_not_equal32(ctx, inst)?,
        Opcode::FPOrdNotEqual64 => floating_point::fp_ord_not_equal64(ctx, inst)?,
        Opcode::FPUnordNotEqual16 => floating_point::fp_unord_not_equal16(ctx, inst)?,
        Opcode::FPUnordNotEqual32 => floating_point::fp_unord_not_equal32(ctx, inst)?,
        Opcode::FPUnordNotEqual64 => floating_point::fp_unord_not_equal64(ctx, inst)?,
        Opcode::FPOrdLessThan16 => floating_point::fp_ord_less_than16(ctx, inst)?,
        Opcode::FPOrdLessThan32 => floating_point::fp_ord_less_than32(ctx, inst)?,
        Opcode::FPOrdLessThan64 => floating_point::fp_ord_less_than64(ctx, inst)?,
        Opcode::FPUnordLessThan16 => floating_point::fp_unord_less_than16(ctx, inst)?,
        Opcode::FPUnordLessThan32 => floating_point::fp_unord_less_than32(ctx, inst)?,
        Opcode::FPUnordLessThan64 => floating_point::fp_unord_less_than64(ctx, inst)?,
        Opcode::FPOrdGreaterThan16 => floating_point::fp_ord_greater_than16(ctx, inst)?,
        Opcode::FPOrdGreaterThan32 => floating_point::fp_ord_greater_than32(ctx, inst)?,
        Opcode::FPOrdGreaterThan64 => floating_point::fp_ord_greater_than64(ctx, inst)?,
        Opcode::FPUnordGreaterThan16 => floating_point::fp_unord_greater_than16(ctx, inst)?,
        Opcode::FPUnordGreaterThan32 => floating_point::fp_unord_greater_than32(ctx, inst)?,
        Opcode::FPUnordGreaterThan64 => floating_point::fp_unord_greater_than64(ctx, inst)?,
        Opcode::FPOrdLessThanEqual16 => floating_point::fp_ord_less_than_equal16(ctx, inst)?,
        Opcode::FPOrdLessThanEqual32 => floating_point::fp_ord_less_than_equal32(ctx, inst)?,
        Opcode::FPOrdLessThanEqual64 => floating_point::fp_ord_less_than_equal64(ctx, inst)?,
        Opcode::FPUnordLessThanEqual16 => floating_point::fp_unord_less_than_equal16(ctx, inst)?,
        Opcode::FPUnordLessThanEqual32 => floating_point::fp_unord_less_than_equal32(ctx, inst)?,
        Opcode::FPUnordLessThanEqual64 => floating_point::fp_unord_less_than_equal64(ctx, inst)?,
        Opcode::FPOrdGreaterThanEqual16 => floating_point::fp_ord_greater_than_equal16(ctx, inst)?,
        Opcode::FPOrdGreaterThanEqual32 => floating_point::fp_ord_greater_than_equal32(ctx, inst)?,
        Opcode::FPOrdGreaterThanEqual64 => floating_point::fp_ord_greater_than_equal64(ctx, inst)?,
        Opcode::FPUnordGreaterThanEqual16 => {
            floating_point::fp_unord_greater_than_equal16(ctx, inst)?
        }
        Opcode::FPUnordGreaterThanEqual32 => {
            floating_point::fp_unord_greater_than_equal32(ctx, inst)?
        }
        Opcode::FPUnordGreaterThanEqual64 => {
            floating_point::fp_unord_greater_than_equal64(ctx, inst)?
        }
        Opcode::FPIsNan16 => floating_point::fp_is_nan16(ctx, inst)?,
        Opcode::FPIsNan32 => floating_point::fp_is_nan32(ctx, inst)?,
        Opcode::FPIsNan64 => floating_point::fp_is_nan64(ctx, inst)?,
        Opcode::IAdd32 => integer::i_add32(ctx, inst)?,
        Opcode::IAdd64 => integer::i_add64(ctx, inst)?,
        Opcode::ISub32 => integer::i_sub32(ctx, inst)?,
        Opcode::ISub64 => integer::i_sub64(ctx, inst)?,
        Opcode::IMul32 => integer::i_mul32(ctx, inst)?,
        Opcode::SDiv32 => integer::s_div32(ctx, inst)?,
        Opcode::UDiv32 => integer::u_div32(ctx, inst)?,
        Opcode::INeg32 => integer::i_neg32(ctx, inst)?,
        Opcode::INeg64 => integer::i_neg64(ctx, inst)?,
        Opcode::IAbs32 => integer::i_abs32(ctx, inst)?,
        Opcode::IAbs64 => integer::i_abs64(ctx, inst)?,
        Opcode::ShiftLeftLogical32 => integer::shift_left_logical32(ctx, inst)?,
        Opcode::ShiftLeftLogical64 => integer::shift_left_logical64(ctx, inst)?,
        Opcode::ShiftRightLogical32 => integer::shift_right_logical32(ctx, inst)?,
        Opcode::ShiftRightLogical64 => integer::shift_right_logical64(ctx, inst)?,
        Opcode::ShiftRightArithmetic32 => integer::shift_right_arithmetic32(ctx, inst)?,
        Opcode::ShiftRightArithmetic64 => integer::shift_right_arithmetic64(ctx, inst)?,
        Opcode::BitwiseAnd32 => integer::bitwise_and32(ctx, inst)?,
        Opcode::BitwiseOr32 => integer::bitwise_or32(ctx, inst)?,
        Opcode::BitwiseXor32 => integer::bitwise_xor32(ctx, inst)?,
        Opcode::BitFieldInsert => integer::bit_field_insert(ctx, inst)?,
        Opcode::BitFieldSExtract => integer::bit_field_s_extract(ctx, inst)?,
        Opcode::BitFieldUExtract => integer::bit_field_u_extract(ctx, inst)?,
        Opcode::BitReverse32 => integer::bit_reverse32(ctx, inst)?,
        Opcode::BitCount32 => integer::bit_count32(ctx, inst)?,
        Opcode::BitwiseNot32 => integer::bitwise_not32(ctx, inst)?,
        Opcode::FindSMsb32 => integer::find_s_msb32(ctx, inst)?,
        Opcode::FindUMsb32 => integer::find_u_msb32(ctx, inst)?,
        Opcode::SMin32 => integer::s_min32(ctx, inst)?,
        Opcode::UMin32 => integer::u_min32(ctx, inst)?,
        Opcode::SMax32 => integer::s_max32(ctx, inst)?,
        Opcode::UMax32 => integer::u_max32(ctx, inst)?,
        Opcode::SClamp32 => integer::s_clamp32(ctx, inst)?,
        Opcode::UClamp32 => integer::u_clamp32(ctx, inst)?,
        Opcode::SLessThan => integer::s_less_than(ctx, inst)?,
        Opcode::ULessThan => integer::u_less_than(ctx, inst)?,
        Opcode::IEqual => integer::i_equal(ctx, inst)?,
        Opcode::SLessThanEqual => integer::s_less_than_equal(ctx, inst)?,
        Opcode::ULessThanEqual => integer::u_less_than_equal(ctx, inst)?,
        Opcode::SGreaterThan => integer::s_greater_than(ctx, inst)?,
        Opcode::UGreaterThan => integer::u_greater_than(ctx, inst)?,
        Opcode::INotEqual => integer::i_not_equal(ctx, inst)?,
        Opcode::SGreaterThanEqual => integer::s_greater_than_equal(ctx, inst)?,
        Opcode::UGreaterThanEqual => integer::u_greater_than_equal(ctx, inst)?,
        Opcode::SharedAtomicIAdd32 => atomic::shared_atomic_i_add32(ctx, inst)?,
        Opcode::SharedAtomicSMin32 => atomic::shared_atomic_s_min32(ctx, inst)?,
        Opcode::SharedAtomicUMin32 => atomic::shared_atomic_u_min32(ctx, inst)?,
        Opcode::SharedAtomicSMax32 => atomic::shared_atomic_s_max32(ctx, inst)?,
        Opcode::SharedAtomicUMax32 => atomic::shared_atomic_u_max32(ctx, inst)?,
        Opcode::SharedAtomicInc32 => atomic::shared_atomic_inc32(ctx, inst)?,
        Opcode::SharedAtomicDec32 => atomic::shared_atomic_dec32(ctx, inst)?,
        Opcode::SharedAtomicAnd32 => atomic::shared_atomic_and32(ctx, inst)?,
        Opcode::SharedAtomicOr32 => atomic::shared_atomic_or32(ctx, inst)?,
        Opcode::SharedAtomicXor32 => atomic::shared_atomic_xor32(ctx, inst)?,
        Opcode::SharedAtomicExchange32 => atomic::shared_atomic_exchange32(ctx, inst)?,
        Opcode::SharedAtomicExchange64 => atomic::shared_atomic_exchange64(ctx, inst)?,
        Opcode::SharedAtomicExchange32x2 => atomic::shared_atomic_exchange32x2(ctx, inst)?,
        Opcode::SharedAtomicCompareExchange32 => {
            atomic::shared_atomic_compare_exchange32(ctx, inst)?
        }
        Opcode::SharedAtomicCompareExchange64 => {
            atomic::shared_atomic_compare_exchange64(ctx, inst)?
        }
        Opcode::GlobalAtomicIAdd32 => atomic::global_atomic_i_add32(ctx, inst)?,
        Opcode::GlobalAtomicSMin32 => atomic::global_atomic_s_min32(ctx, inst)?,
        Opcode::GlobalAtomicUMin32 => atomic::global_atomic_u_min32(ctx, inst)?,
        Opcode::GlobalAtomicSMax32 => atomic::global_atomic_s_max32(ctx, inst)?,
        Opcode::GlobalAtomicUMax32 => atomic::global_atomic_u_max32(ctx, inst)?,
        Opcode::GlobalAtomicInc32 => atomic::global_atomic_inc32(ctx, inst)?,
        Opcode::GlobalAtomicDec32 => atomic::global_atomic_dec32(ctx, inst)?,
        Opcode::GlobalAtomicAnd32 => atomic::global_atomic_and32(ctx, inst)?,
        Opcode::GlobalAtomicOr32 => atomic::global_atomic_or32(ctx, inst)?,
        Opcode::GlobalAtomicXor32 => atomic::global_atomic_xor32(ctx, inst)?,
        Opcode::GlobalAtomicExchange32 => atomic::global_atomic_exchange32(ctx, inst)?,
        Opcode::GlobalAtomicCompareExchange32 => {
            atomic::global_atomic_compare_exchange32(ctx, inst)?
        }
        Opcode::GlobalAtomicIAdd64 => atomic::global_atomic_i_add64(ctx, inst)?,
        Opcode::GlobalAtomicSMin64 => atomic::global_atomic_s_min64(ctx, inst)?,
        Opcode::GlobalAtomicUMin64 => atomic::global_atomic_u_min64(ctx, inst)?,
        Opcode::GlobalAtomicSMax64 => atomic::global_atomic_s_max64(ctx, inst)?,
        Opcode::GlobalAtomicUMax64 => atomic::global_atomic_u_max64(ctx, inst)?,
        Opcode::GlobalAtomicAnd64 => atomic::global_atomic_and64(ctx, inst)?,
        Opcode::GlobalAtomicOr64 => atomic::global_atomic_or64(ctx, inst)?,
        Opcode::GlobalAtomicXor64 => atomic::global_atomic_xor64(ctx, inst)?,
        Opcode::GlobalAtomicExchange64 => atomic::global_atomic_exchange64(ctx, inst)?,
        Opcode::GlobalAtomicCompareExchange64 => {
            atomic::global_atomic_compare_exchange64(ctx, inst)?
        }
        Opcode::GlobalAtomicIAdd32x2 => atomic::global_atomic_i_add32x2(ctx, inst)?,
        Opcode::GlobalAtomicSMin32x2 => atomic::global_atomic_s_min32x2(ctx, inst)?,
        Opcode::GlobalAtomicUMin32x2 => atomic::global_atomic_u_min32x2(ctx, inst)?,
        Opcode::GlobalAtomicSMax32x2 => atomic::global_atomic_s_max32x2(ctx, inst)?,
        Opcode::GlobalAtomicUMax32x2 => atomic::global_atomic_u_max32x2(ctx, inst)?,
        Opcode::GlobalAtomicAnd32x2 => atomic::global_atomic_and32x2(ctx, inst)?,
        Opcode::GlobalAtomicOr32x2 => atomic::global_atomic_or32x2(ctx, inst)?,
        Opcode::GlobalAtomicXor32x2 => atomic::global_atomic_xor32x2(ctx, inst)?,
        Opcode::GlobalAtomicExchange32x2 => atomic::global_atomic_exchange32x2(ctx, inst)?,
        Opcode::GlobalAtomicAddF32 => atomic::global_atomic_add_f32(ctx, inst)?,
        Opcode::GlobalAtomicAddF16x2 => atomic::global_atomic_add_f16x2(ctx, inst)?,
        Opcode::GlobalAtomicAddF32x2 => atomic::global_atomic_add_f32x2(ctx, inst)?,
        Opcode::GlobalAtomicMinF16x2 => atomic::global_atomic_min_f16x2(ctx, inst)?,
        Opcode::GlobalAtomicMinF32x2 => atomic::global_atomic_min_f32x2(ctx, inst)?,
        Opcode::GlobalAtomicMaxF16x2 => atomic::global_atomic_max_f16x2(ctx, inst)?,
        Opcode::GlobalAtomicMaxF32x2 => atomic::global_atomic_max_f32x2(ctx, inst)?,
        Opcode::StorageAtomicIAdd32 => atomic::storage_atomic_i_add32(ctx, inst)?,
        Opcode::StorageAtomicSMin32 => atomic::storage_atomic_s_min32(ctx, inst)?,
        Opcode::StorageAtomicUMin32 => atomic::storage_atomic_u_min32(ctx, inst)?,
        Opcode::StorageAtomicSMax32 => atomic::storage_atomic_s_max32(ctx, inst)?,
        Opcode::StorageAtomicUMax32 => atomic::storage_atomic_u_max32(ctx, inst)?,
        Opcode::StorageAtomicInc32 => atomic::storage_atomic_inc32(ctx, inst)?,
        Opcode::StorageAtomicDec32 => atomic::storage_atomic_dec32(ctx, inst)?,
        Opcode::StorageAtomicAnd32 => atomic::storage_atomic_and32(ctx, inst)?,
        Opcode::StorageAtomicOr32 => atomic::storage_atomic_or32(ctx, inst)?,
        Opcode::StorageAtomicXor32 => atomic::storage_atomic_xor32(ctx, inst)?,
        Opcode::StorageAtomicExchange32 => atomic::storage_atomic_exchange32(ctx, inst)?,
        Opcode::StorageAtomicCompareExchange32 => {
            atomic::storage_atomic_compare_exchange32(ctx, inst)?
        }
        Opcode::StorageAtomicIAdd64 => atomic::storage_atomic_i_add64(ctx, inst)?,
        Opcode::StorageAtomicSMin64 => atomic::storage_atomic_s_min64(ctx, inst)?,
        Opcode::StorageAtomicUMin64 => atomic::storage_atomic_u_min64(ctx, inst)?,
        Opcode::StorageAtomicSMax64 => atomic::storage_atomic_s_max64(ctx, inst)?,
        Opcode::StorageAtomicUMax64 => atomic::storage_atomic_u_max64(ctx, inst)?,
        Opcode::StorageAtomicAnd64 => atomic::storage_atomic_and64(ctx, inst)?,
        Opcode::StorageAtomicOr64 => atomic::storage_atomic_or64(ctx, inst)?,
        Opcode::StorageAtomicXor64 => atomic::storage_atomic_xor64(ctx, inst)?,
        Opcode::StorageAtomicExchange64 => atomic::storage_atomic_exchange64(ctx, inst)?,
        Opcode::StorageAtomicCompareExchange64 => {
            atomic::storage_atomic_compare_exchange64(ctx, inst)?
        }
        Opcode::StorageAtomicIAdd32x2 => atomic::storage_atomic_i_add32x2(ctx, inst)?,
        Opcode::StorageAtomicSMin32x2 => atomic::storage_atomic_s_min32x2(ctx, inst)?,
        Opcode::StorageAtomicUMin32x2 => atomic::storage_atomic_u_min32x2(ctx, inst)?,
        Opcode::StorageAtomicSMax32x2 => atomic::storage_atomic_s_max32x2(ctx, inst)?,
        Opcode::StorageAtomicUMax32x2 => atomic::storage_atomic_u_max32x2(ctx, inst)?,
        Opcode::StorageAtomicAnd32x2 => atomic::storage_atomic_and32x2(ctx, inst)?,
        Opcode::StorageAtomicOr32x2 => atomic::storage_atomic_or32x2(ctx, inst)?,
        Opcode::StorageAtomicXor32x2 => atomic::storage_atomic_xor32x2(ctx, inst)?,
        Opcode::StorageAtomicExchange32x2 => atomic::storage_atomic_exchange32x2(ctx, inst)?,
        Opcode::StorageAtomicAddF32 => atomic::storage_atomic_add_f32(ctx, inst)?,
        Opcode::StorageAtomicAddF16x2 => atomic::storage_atomic_add_f16x2(ctx, inst)?,
        Opcode::StorageAtomicAddF32x2 => atomic::storage_atomic_add_f32x2(ctx, inst)?,
        Opcode::StorageAtomicMinF16x2 => atomic::storage_atomic_min_f16x2(ctx, inst)?,
        Opcode::StorageAtomicMinF32x2 => atomic::storage_atomic_min_f32x2(ctx, inst)?,
        Opcode::StorageAtomicMaxF16x2 => atomic::storage_atomic_max_f16x2(ctx, inst)?,
        Opcode::StorageAtomicMaxF32x2 => atomic::storage_atomic_max_f32x2(ctx, inst)?,
        Opcode::LogicalOr => logical::logical_or(ctx, inst)?,
        Opcode::LogicalAnd => logical::logical_and(ctx, inst)?,
        Opcode::LogicalXor => logical::logical_xor(ctx, inst)?,
        Opcode::LogicalNot => logical::logical_not(ctx, inst)?,
        Opcode::ConvertS16F16 => convert::convert_s16_f16(ctx, inst)?,
        Opcode::ConvertS16F32 => convert::convert_s16_f32(ctx, inst)?,
        Opcode::ConvertS16F64 => convert::convert_s16_f64(ctx, inst)?,
        Opcode::ConvertS32F16 => convert::convert_s32_f16(ctx, inst)?,
        Opcode::ConvertS32F32 => convert::convert_s32_f32(ctx, inst)?,
        Opcode::ConvertS32F64 => convert::convert_s32_f64(ctx, inst)?,
        Opcode::ConvertS64F16 => convert::convert_s64_f16(ctx, inst)?,
        Opcode::ConvertS64F32 => convert::convert_s64_f32(ctx, inst)?,
        Opcode::ConvertS64F64 => convert::convert_s64_f64(ctx, inst)?,
        Opcode::ConvertU16F16 => convert::convert_u16_f16(ctx, inst)?,
        Opcode::ConvertU16F32 => convert::convert_u16_f32(ctx, inst)?,
        Opcode::ConvertU16F64 => convert::convert_u16_f64(ctx, inst)?,
        Opcode::ConvertU32F16 => convert::convert_u32_f16(ctx, inst)?,
        Opcode::ConvertU32F32 => convert::convert_u32_f32(ctx, inst)?,
        Opcode::ConvertU32F64 => convert::convert_u32_f64(ctx, inst)?,
        Opcode::ConvertU64F16 => convert::convert_u64_f16(ctx, inst)?,
        Opcode::ConvertU64F32 => convert::convert_u64_f32(ctx, inst)?,
        Opcode::ConvertU64F64 => convert::convert_u64_f64(ctx, inst)?,
        Opcode::ConvertU64U32 => convert::convert_u64_u32(ctx, inst)?,
        Opcode::ConvertU32U64 => convert::convert_u32_u64(ctx, inst)?,
        Opcode::ConvertF16F32 => convert::convert_f16_f32(ctx, inst)?,
        Opcode::ConvertF32F16 => convert::convert_f32_f16(ctx, inst)?,
        Opcode::ConvertF32F64 => convert::convert_f32_f64(ctx, inst)?,
        Opcode::ConvertF64F32 => convert::convert_f64_f32(ctx, inst)?,
        Opcode::ConvertF16S8 => convert::convert_f16_s8(ctx, inst)?,
        Opcode::ConvertF16S16 => convert::convert_f16_s16(ctx, inst)?,
        Opcode::ConvertF16S32 => convert::convert_f16_s32(ctx, inst)?,
        Opcode::ConvertF16S64 => convert::convert_f16_s64(ctx, inst)?,
        Opcode::ConvertF16U8 => convert::convert_f16_u8(ctx, inst)?,
        Opcode::ConvertF16U16 => convert::convert_f16_u16(ctx, inst)?,
        Opcode::ConvertF16U32 => convert::convert_f16_u32(ctx, inst)?,
        Opcode::ConvertF16U64 => convert::convert_f16_u64(ctx, inst)?,
        Opcode::ConvertF32S8 => convert::convert_f32_s8(ctx, inst)?,
        Opcode::ConvertF32S16 => convert::convert_f32_s16(ctx, inst)?,
        Opcode::ConvertF32S32 => convert::convert_f32_s32(ctx, inst)?,
        Opcode::ConvertF32S64 => convert::convert_f32_s64(ctx, inst)?,
        Opcode::ConvertF32U8 => convert::convert_f32_u8(ctx, inst)?,
        Opcode::ConvertF32U16 => convert::convert_f32_u16(ctx, inst)?,
        Opcode::ConvertF32U32 => convert::convert_f32_u32(ctx, inst)?,
        Opcode::ConvertF32U64 => convert::convert_f32_u64(ctx, inst)?,
        Opcode::ConvertF64S8 => convert::convert_f64_s8(ctx, inst)?,
        Opcode::ConvertF64S16 => convert::convert_f64_s16(ctx, inst)?,
        Opcode::ConvertF64S32 => convert::convert_f64_s32(ctx, inst)?,
        Opcode::ConvertF64S64 => convert::convert_f64_s64(ctx, inst)?,
        Opcode::ConvertF64U8 => convert::convert_f64_u8(ctx, inst)?,
        Opcode::ConvertF64U16 => convert::convert_f64_u16(ctx, inst)?,
        Opcode::ConvertF64U32 => convert::convert_f64_u32(ctx, inst)?,
        Opcode::ConvertF64U64 => convert::convert_f64_u64(ctx, inst)?,
        Opcode::ConvertU16U32 => convert::convert_u16_u32(ctx, inst)?,
        Opcode::ConvertU32U16 => convert::convert_u32_u16(ctx, inst)?,
        Opcode::ConvertU8U32 => convert::convert_u8_u32(ctx, inst)?,
        Opcode::ConvertU32U8 => convert::convert_u32_u8(ctx, inst)?,
        Opcode::ConvertS32S8 => convert::convert_s32_s8(ctx, inst)?,
        Opcode::ConvertS32S16 => convert::convert_s32_s16(ctx, inst)?,
        Opcode::BindlessImageSampleImplicitLod => {
            image::bindless_image_sample_implicit_lod(ctx, inst)?
        }
        Opcode::BindlessImageSampleExplicitLod => {
            image::bindless_image_sample_explicit_lod(ctx, inst)?
        }
        Opcode::BindlessImageSampleDrefImplicitLod => {
            image::bindless_image_sample_dref_implicit_lod(ctx, inst)?
        }
        Opcode::BindlessImageSampleDrefExplicitLod => {
            image::bindless_image_sample_dref_explicit_lod(ctx, inst)?
        }
        Opcode::BindlessImageGather => image::bindless_image_gather(ctx, inst)?,
        Opcode::BindlessImageGatherDref => image::bindless_image_gather_dref(ctx, inst)?,
        Opcode::BindlessImageFetch => image::bindless_image_fetch(ctx, inst)?,
        Opcode::BindlessImageQueryDimensions => image::bindless_image_query_dimensions(ctx, inst)?,
        Opcode::BindlessImageQueryLod => image::bindless_image_query_lod(ctx, inst)?,
        Opcode::BindlessImageQuerySamples | Opcode::BoundImageQuerySamples => {
            return Err(image::unlowered(ctx, inst));
        }
        Opcode::BindlessImageGradient => image::bindless_image_gradient(ctx, inst)?,
        Opcode::BindlessImageRead => image::bindless_image_read(ctx, inst)?,
        Opcode::BindlessImageWrite => return image::bindless_image_write(ctx, inst),
        Opcode::BoundImageSampleImplicitLod => image::bound_image_sample_implicit_lod(ctx, inst)?,
        Opcode::BoundImageSampleExplicitLod => image::bound_image_sample_explicit_lod(ctx, inst)?,
        Opcode::BoundImageSampleDrefImplicitLod => {
            image::bound_image_sample_dref_implicit_lod(ctx, inst)?
        }
        Opcode::BoundImageSampleDrefExplicitLod => {
            image::bound_image_sample_dref_explicit_lod(ctx, inst)?
        }
        Opcode::BoundImageGather => image::bound_image_gather(ctx, inst)?,
        Opcode::BoundImageGatherDref => image::bound_image_gather_dref(ctx, inst)?,
        Opcode::BoundImageFetch => image::bound_image_fetch(ctx, inst)?,
        Opcode::BoundImageQueryDimensions => image::bound_image_query_dimensions(ctx, inst)?,
        Opcode::BoundImageQueryLod => image::bound_image_query_lod(ctx, inst)?,
        Opcode::BoundImageGradient => image::bound_image_gradient(ctx, inst)?,
        Opcode::BoundImageRead => image::bound_image_read(ctx, inst)?,
        Opcode::BoundImageWrite => return image::bound_image_write(ctx, inst),
        Opcode::ImageSampleImplicitLod => image::image_sample_implicit_lod(ctx, inst)?,
        Opcode::ImageSampleExplicitLod => image::image_sample_explicit_lod(ctx, inst)?,
        Opcode::ImageSampleDrefImplicitLod => image::image_sample_dref_implicit_lod(ctx, inst)?,
        Opcode::ImageSampleDrefExplicitLod => image::image_sample_dref_explicit_lod(ctx, inst)?,
        Opcode::ImageGather => image::image_gather(ctx, inst)?,
        Opcode::ImageGatherDref => image::image_gather_dref(ctx, inst)?,
        Opcode::ImageFetch => image::image_fetch(ctx, inst)?,
        Opcode::ImageQueryDimensions => image::image_query_dimensions(ctx, inst)?,
        Opcode::ImageQueryLod => image::image_query_lod(ctx, inst)?,
        Opcode::ImageQuerySamples => image::image_query_samples(ctx, inst)?,
        Opcode::ImageGradient => image::image_gradient(ctx, inst)?,
        Opcode::ImageRead => image::image_read(ctx, inst)?,
        Opcode::ImageWrite => return image::image_write(ctx, inst),
        Opcode::IsTextureScaled => image::is_texture_scaled(ctx, inst)?,
        Opcode::IsImageScaled => image::is_image_scaled(ctx, inst)?,
        Opcode::BindlessImageAtomicIAdd32 => {
            image_atomic::bindless_image_atomic_i_add32(ctx, inst)?
        }
        Opcode::BindlessImageAtomicSMin32 => {
            image_atomic::bindless_image_atomic_s_min32(ctx, inst)?
        }
        Opcode::BindlessImageAtomicUMin32 => {
            image_atomic::bindless_image_atomic_u_min32(ctx, inst)?
        }
        Opcode::BindlessImageAtomicSMax32 => {
            image_atomic::bindless_image_atomic_s_max32(ctx, inst)?
        }
        Opcode::BindlessImageAtomicUMax32 => {
            image_atomic::bindless_image_atomic_u_max32(ctx, inst)?
        }
        Opcode::BindlessImageAtomicInc32 => image_atomic::bindless_image_atomic_inc32(ctx, inst)?,
        Opcode::BindlessImageAtomicDec32 => image_atomic::bindless_image_atomic_dec32(ctx, inst)?,
        Opcode::BindlessImageAtomicAnd32 => image_atomic::bindless_image_atomic_and32(ctx, inst)?,
        Opcode::BindlessImageAtomicOr32 => image_atomic::bindless_image_atomic_or32(ctx, inst)?,
        Opcode::BindlessImageAtomicXor32 => image_atomic::bindless_image_atomic_xor32(ctx, inst)?,
        Opcode::BindlessImageAtomicExchange32 => {
            image_atomic::bindless_image_atomic_exchange32(ctx, inst)?
        }
        Opcode::BoundImageAtomicIAdd32 => image_atomic::bound_image_atomic_i_add32(ctx, inst)?,
        Opcode::BoundImageAtomicSMin32 => image_atomic::bound_image_atomic_s_min32(ctx, inst)?,
        Opcode::BoundImageAtomicUMin32 => image_atomic::bound_image_atomic_u_min32(ctx, inst)?,
        Opcode::BoundImageAtomicSMax32 => image_atomic::bound_image_atomic_s_max32(ctx, inst)?,
        Opcode::BoundImageAtomicUMax32 => image_atomic::bound_image_atomic_u_max32(ctx, inst)?,
        Opcode::BoundImageAtomicInc32 => image_atomic::bound_image_atomic_inc32(ctx, inst)?,
        Opcode::BoundImageAtomicDec32 => image_atomic::bound_image_atomic_dec32(ctx, inst)?,
        Opcode::BoundImageAtomicAnd32 => image_atomic::bound_image_atomic_and32(ctx, inst)?,
        Opcode::BoundImageAtomicOr32 => image_atomic::bound_image_atomic_or32(ctx, inst)?,
        Opcode::BoundImageAtomicXor32 => image_atomic::bound_image_atomic_xor32(ctx, inst)?,
        Opcode::BoundImageAtomicExchange32 => {
            image_atomic::bound_image_atomic_exchange32(ctx, inst)?
        }
        Opcode::ImageAtomicIAdd32 => image_atomic::image_atomic_i_add32(ctx, inst)?,
        Opcode::ImageAtomicSMin32 => image_atomic::image_atomic_s_min32(ctx, inst)?,
        Opcode::ImageAtomicUMin32 => image_atomic::image_atomic_u_min32(ctx, inst)?,
        Opcode::ImageAtomicSMax32 => image_atomic::image_atomic_s_max32(ctx, inst)?,
        Opcode::ImageAtomicUMax32 => image_atomic::image_atomic_u_max32(ctx, inst)?,
        Opcode::ImageAtomicInc32 => image_atomic::image_atomic_inc32(ctx, inst)?,
        Opcode::ImageAtomicDec32 => image_atomic::image_atomic_dec32(ctx, inst)?,
        Opcode::ImageAtomicAnd32 => image_atomic::image_atomic_and32(ctx, inst)?,
        Opcode::ImageAtomicOr32 => image_atomic::image_atomic_or32(ctx, inst)?,
        Opcode::ImageAtomicXor32 => image_atomic::image_atomic_xor32(ctx, inst)?,
        Opcode::ImageAtomicExchange32 => image_atomic::image_atomic_exchange32(ctx, inst)?,
        Opcode::LaneId => warp::lane_id(ctx, inst)?,
        Opcode::VoteAll => warp::vote_all(ctx, inst)?,
        Opcode::VoteAny => warp::vote_any(ctx, inst)?,
        Opcode::VoteEqual => warp::vote_equal(ctx, inst)?,
        Opcode::SubgroupBallot => warp::subgroup_ballot(ctx, inst)?,
        Opcode::SubgroupEqMask => warp::subgroup_eq_mask(ctx, inst)?,
        Opcode::SubgroupLtMask => warp::subgroup_lt_mask(ctx, inst)?,
        Opcode::SubgroupLeMask => warp::subgroup_le_mask(ctx, inst)?,
        Opcode::SubgroupGtMask => warp::subgroup_gt_mask(ctx, inst)?,
        Opcode::SubgroupGeMask => warp::subgroup_ge_mask(ctx, inst)?,
        Opcode::ShuffleIndex => warp::shuffle_index(ctx, inst)?,
        Opcode::ShuffleUp => warp::shuffle_up(ctx, inst)?,
        Opcode::ShuffleDown => warp::shuffle_down(ctx, inst)?,
        Opcode::ShuffleButterfly => warp::shuffle_butterfly(ctx, inst)?,
        Opcode::QuadBroadcast => warp::quad_broadcast(ctx, inst)?,
        Opcode::QuadSwap => warp::quad_swap(ctx, inst)?,
        Opcode::FSwizzleAdd => warp::f_swizzle_add(ctx, inst)?,
        Opcode::DPdxFine => warp::d_pdx_fine(ctx, inst)?,
        Opcode::DPdyFine => warp::d_pdy_fine(ctx, inst)?,
        Opcode::DPdxCoarse => warp::d_pdx_coarse(ctx, inst)?,
        Opcode::DPdyCoarse => warp::d_pdy_coarse(ctx, inst)?,
    };
    ctx.set_def(inst, id);
    Ok(())
}

/// Shared memory exists only in compute programs that declare a size; elsewhere a stray
/// access reads zero and its stores are dropped.
fn is_shared_memory_access(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        LoadSharedU8
            | LoadSharedS8
            | LoadSharedU16
            | LoadSharedS16
            | LoadSharedU32
            | LoadSharedU64
            | LoadSharedU128
            | WriteSharedU8
            | WriteSharedU16
            | WriteSharedU32
            | WriteSharedU64
            | WriteSharedU128
            | SharedAtomicIAdd32
            | SharedAtomicSMin32
            | SharedAtomicUMin32
            | SharedAtomicSMax32
            | SharedAtomicUMax32
            | SharedAtomicInc32
            | SharedAtomicDec32
            | SharedAtomicAnd32
            | SharedAtomicOr32
            | SharedAtomicXor32
            | SharedAtomicExchange32
            | SharedAtomicExchange64
            | SharedAtomicExchange32x2
            | SharedAtomicCompareExchange32
            | SharedAtomicCompareExchange64
    )
}
