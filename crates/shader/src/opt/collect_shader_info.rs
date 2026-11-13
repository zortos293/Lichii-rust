//! Records the features, types and resources a program uses, ported from
//! `collect_shader_info_pass.cpp`.

use crate::info::{ConstantBufferDescriptor, ShaderInfo, VaryingState};
use crate::ir::{
    Attribute, FmzMode, FpControl, ImageFormat, InstId, Opcode, Patch, Program, TextureInstInfo,
    TextureType, Type, Value,
};
use crate::maxwell::ProgramHeader;
use crate::maxwell::header::{PixelImap, SystemValuesB, SystemValuesC};
use crate::{ShaderArch, ShaderEnvironment, ShaderStage, ampere};

const NVN_DESCRIPTOR_SIZE: u32 = 0x10;
const NVN_DESCRIPTORS: u32 = 16;
const UNBOUNDED_CBUF_SIZE: u32 = 0x1_0000;

/// Fills `info` with what `prog` uses: capabilities, constant buffers, attributes read and
/// written (refined by the program header when the program indexes attributes) and the NVN
/// storage buffer slots it may address.
pub fn collect_shader_info(env: &dyn ShaderEnvironment, prog: &Program, info: &mut ShaderInfo) {
    let stage = env.stage();
    info.nvn_buffer_base = match stage {
        ShaderStage::VertexA | ShaderStage::VertexB | ShaderStage::Task => 0x110,
        ShaderStage::TessellationControl => 0x210,
        ShaderStage::TessellationEval | ShaderStage::Mesh | ShaderStage::Compute => 0x310,
        ShaderStage::Geometry => 0x410,
        ShaderStage::Fragment => 0x510,
    };
    if matches!(stage, ShaderStage::Task | ShaderStage::Mesh) {
        info.uses_workgroup_id = true;
        info.uses_local_invocation_id = true;
    }
    for &block in &prog.post_order {
        for id in prog.insts_of(block) {
            visit_usages(info, prog, id);
            visit_fp_modifiers(info, prog, id);
            if is_get_cbuf(prog.op(id)) {
                check_cbuf_nvn(info, prog, id);
            }
        }
    }
    gather_info_from_header(env, info);
}

fn is_get_cbuf(op: Opcode) -> bool {
    use Opcode::*;
    matches!(
        op,
        GetCbufU8 | GetCbufS8 | GetCbufU16 | GetCbufS16 | GetCbufU32 | GetCbufF32 | GetCbufU32x2
    )
}

fn add_constant_buffer_descriptor(info: &mut ShaderInfo, index: u32) {
    if index as usize >= ShaderInfo::MAX_CBUFS || info.constant_buffer_mask & (1 << index) != 0 {
        return;
    }
    info.constant_buffer_mask |= 1 << index;
    let cbufs = &mut info.constant_buffer_descriptors;
    let at = cbufs.partition_point(|d| d.index < index);
    cbufs.insert(at, ConstantBufferDescriptor { index, count: 1 });
}

fn add_register_indexed_ldc(info: &mut ShaderInfo) {
    info.uses_cbuf_indirect = true;
    for index in 0..ShaderInfo::MAX_INDIRECT_CBUFS {
        add_constant_buffer_descriptor(info, index as u32);
        info.constant_buffer_used_sizes[index] = UNBOUNDED_CBUF_SIZE;
    }
}

/// Bytes a constant buffer read loads, recording its type in `used_types`.
fn cbuf_element_size(used_types: &mut Type, op: Opcode) -> u32 {
    let (ty, size) = match op {
        Opcode::GetCbufU8 | Opcode::GetCbufS8 => (Type::U8, 1),
        Opcode::GetCbufU16 | Opcode::GetCbufS16 => (Type::U16, 2),
        Opcode::GetCbufU32 => (Type::U32, 4),
        Opcode::GetCbufF32 => (Type::F32, 4),
        Opcode::GetCbufU32x2 => (Type::U32X2, 8),
        op => unreachable!("{op:?} is not a constant buffer read"),
    };
    *used_types |= ty;
    size
}

fn visit_cbuf_read(info: &mut ShaderInfo, prog: &Program, id: InstId) {
    let op = prog.op(id);
    let Some(index) = prog.arg(id, 0).imm_u32() else {
        add_register_indexed_ldc(info);
        cbuf_element_size(&mut info.used_indirect_cbuf_types, op);
        return;
    };
    add_constant_buffer_descriptor(info, index);
    let element_size = cbuf_element_size(&mut info.used_constant_buffer_types, op);
    let Some(size) = info.constant_buffer_used_sizes.get_mut(index as usize) else {
        return;
    };
    *size = match prog.arg(id, 1).imm_u32() {
        Some(offset) => (*size)
            .max(offset.saturating_add(element_size))
            .min(UNBOUNDED_CBUF_SIZE)
            .next_multiple_of(16),
        None => UNBOUNDED_CBUF_SIZE,
    };
}

fn check_cbuf_nvn(info: &mut ShaderInfo, prog: &Program, id: InstId) {
    let Some(index) = prog.arg(id, 0).imm_u32() else {
        info.nvn_buffer_used = u16::MAX;
        return;
    };
    if index != 0 {
        return;
    }
    let Some(offset) = prog.arg(id, 1).imm_u32() else {
        info.nvn_buffer_used = u16::MAX;
        return;
    };
    let base = info.nvn_buffer_base;
    if (base..base + NVN_DESCRIPTOR_SIZE * NVN_DESCRIPTORS).contains(&offset) {
        info.nvn_buffer_used |= 1 << ((offset - base) / NVN_DESCRIPTOR_SIZE);
    }
}

fn mark_patch(info: &mut ShaderInfo, patch: Patch) {
    if let Some(used) = info.uses_patches.get_mut(patch.generic_index() as usize) {
        *used = true;
    }
}

fn set_patch(info: &mut ShaderInfo, patch: Patch) {
    match patch {
        p if p.is_generic() => mark_patch(info, p),
        Patch::LOD_LEFT | Patch::LOD_TOP | Patch::LOD_RIGHT | Patch::LOD_BOTTOM => {
            info.stores_tess_level_outer = true;
        }
        _ => info.stores_tess_level_inner = true,
    }
}

fn is_1d(ty: TextureType) -> bool {
    matches!(ty, TextureType::Color1D | TextureType::ColorArray1D)
}

fn visit_usages(info: &mut ShaderInfo, prog: &Program, id: InstId) {
    use Opcode::*;
    let inst = prog.inst(id);
    let op = inst.op;
    let texture = TextureInstInfo::from_flags(inst.flags);
    let sparse = || prog.pseudo_op(id, GetSparseFromOp).is_some();

    match op {
        ImageRead | ImageWrite => {
            info.requires_coherent_image_memory |=
                inst.flags & TextureInstInfo::IMAGE_COHERENT != 0;
        }
        LoadGlobalU8 | LoadGlobalS8 | LoadGlobalU16 | LoadGlobalS16 | LoadGlobal32
        | LoadGlobal64 | LoadGlobal128 | LoadStorageU8 | LoadStorageS8 | LoadStorageU16
        | LoadStorageS16 | LoadStorage32 | LoadStorage64 | LoadStorage128 => {
            info.requires_coherent_global_memory |= inst.flags != 0;
        }
        _ => {}
    }

    match op {
        CompositeConstructF16x2
        | CompositeConstructF16x3
        | CompositeConstructF16x4
        | CompositeExtractF16x2
        | CompositeExtractF16x3
        | CompositeExtractF16x4
        | CompositeInsertF16x2
        | CompositeInsertF16x3
        | CompositeInsertF16x4
        | SelectF16
        | BitCastU16F16
        | BitCastF16U16
        | PackFloat2x16
        | UnpackFloat2x16
        | ConvertS16F16
        | ConvertS32F16
        | ConvertS64F16
        | ConvertU16F16
        | ConvertU32F16
        | ConvertU64F16
        | ConvertF16S8
        | ConvertF16S16
        | ConvertF16S32
        | ConvertF16S64
        | ConvertF16U8
        | ConvertF16U16
        | ConvertF16U32
        | ConvertF16U64
        | ConvertF16F32
        | ConvertF32F16
        | FPAbs16
        | FPAdd16
        | FPCeil16
        | FPFloor16
        | FPFma16
        | FPMul16
        | FPNeg16
        | FPRoundEven16
        | FPSaturate16
        | FPClamp16
        | FPTrunc16
        | FPOrdEqual16
        | FPUnordEqual16
        | FPOrdNotEqual16
        | FPUnordNotEqual16
        | FPOrdLessThan16
        | FPUnordLessThan16
        | FPOrdGreaterThan16
        | FPUnordGreaterThan16
        | FPOrdLessThanEqual16
        | FPUnordLessThanEqual16
        | FPOrdGreaterThanEqual16
        | FPUnordGreaterThanEqual16
        | FPIsNan16
        | GlobalAtomicAddF16x2
        | GlobalAtomicMinF16x2
        | GlobalAtomicMaxF16x2
        | StorageAtomicAddF16x2
        | StorageAtomicMinF16x2
        | StorageAtomicMaxF16x2 => {
            info.uses_fp16 = true;
        }
        CompositeConstructF64x2
        | CompositeConstructF64x3
        | CompositeConstructF64x4
        | CompositeExtractF64x2
        | CompositeExtractF64x3
        | CompositeExtractF64x4
        | CompositeInsertF64x2
        | CompositeInsertF64x3
        | CompositeInsertF64x4
        | SelectF64
        | BitCastU64F64
        | BitCastF64U64
        | PackDouble2x32
        | UnpackDouble2x32
        | FPAbs64
        | FPAdd64
        | FPCeil64
        | FPFloor64
        | FPFma64
        | FPMax64
        | FPMin64
        | FPMul64
        | FPNeg64
        | FPRecip64
        | FPRecipSqrt64
        | FPRoundEven64
        | FPSaturate64
        | FPClamp64
        | FPTrunc64
        | FPOrdEqual64
        | FPUnordEqual64
        | FPOrdNotEqual64
        | FPUnordNotEqual64
        | FPOrdLessThan64
        | FPUnordLessThan64
        | FPOrdGreaterThan64
        | FPUnordGreaterThan64
        | FPOrdLessThanEqual64
        | FPUnordLessThanEqual64
        | FPOrdGreaterThanEqual64
        | FPUnordGreaterThanEqual64
        | FPIsNan64
        | ConvertS16F64
        | ConvertS32F64
        | ConvertS64F64
        | ConvertU16F64
        | ConvertU32F64
        | ConvertU64F64
        | ConvertF32F64
        | ConvertF64F32
        | ConvertF64S8
        | ConvertF64S16
        | ConvertF64S32
        | ConvertF64S64
        | ConvertF64U8
        | ConvertF64U16
        | ConvertF64U32
        | ConvertF64U64 => {
            info.uses_fp64 = true;
        }
        _ => {}
    }

    if matches!(
        op,
        GetCbufU8
            | GetCbufS8
            | UndefU8
            | LoadGlobalU8
            | LoadGlobalS8
            | WriteGlobalU8
            | WriteGlobalS8
            | LoadStorageU8
            | LoadStorageS8
            | WriteStorageU8
            | WriteStorageS8
            | LoadSharedU8
            | LoadSharedS8
            | WriteSharedU8
            | SelectU8
            | ConvertF16S8
            | ConvertF16U8
            | ConvertF32S8
            | ConvertF32U8
            | ConvertF64S8
            | ConvertF64U8
    ) {
        info.uses_int8 = true;
    }

    if matches!(
        op,
        GetCbufU16
            | GetCbufS16
            | UndefU16
            | LoadGlobalU16
            | LoadGlobalS16
            | WriteGlobalU16
            | WriteGlobalS16
            | LoadStorageU16
            | LoadStorageS16
            | WriteStorageU16
            | WriteStorageS16
            | LoadSharedU16
            | LoadSharedS16
            | WriteSharedU16
            | SelectU16
            | BitCastU16F16
            | BitCastF16U16
            | ConvertS16F16
            | ConvertS16F32
            | ConvertS16F64
            | ConvertU16F16
            | ConvertU16F32
            | ConvertU16F64
            | ConvertF16S16
            | ConvertF16U16
            | ConvertF32S16
            | ConvertF32U16
            | ConvertF64S16
            | ConvertF64U16
    ) {
        info.uses_int16 = true;
    }

    if matches!(
        op,
        UndefU64
            | LoadGlobalU8
            | LoadGlobalS8
            | LoadGlobalU16
            | LoadGlobalS16
            | LoadGlobal32
            | LoadGlobal64
            | LoadGlobal128
            | WriteGlobalU8
            | WriteGlobalS8
            | WriteGlobalU16
            | WriteGlobalS16
            | WriteGlobal32
            | WriteGlobal64
            | WriteGlobal128
            | SelectU64
            | BitCastU64F64
            | BitCastF64U64
            | PackUint2x32
            | UnpackUint2x32
            | IAdd64
            | ISub64
            | INeg64
            | ShiftLeftLogical64
            | ShiftRightLogical64
            | ShiftRightArithmetic64
            | ConvertS64F16
            | ConvertS64F32
            | ConvertS64F64
            | ConvertF16S64
            | ConvertF32S64
            | ConvertF64S64
            | IAbs64
            | ConvertU64F16
            | ConvertU64F32
            | ConvertU64F64
            | ConvertU64U32
            | ConvertU32U64
            | ConvertF16U64
            | ConvertF32U64
            | ConvertF64U64
            | SharedAtomicExchange64
            | SharedAtomicCompareExchange64
            | GlobalAtomicIAdd64
            | GlobalAtomicSMin64
            | GlobalAtomicUMin64
            | GlobalAtomicSMax64
            | GlobalAtomicUMax64
            | GlobalAtomicAnd64
            | GlobalAtomicOr64
            | GlobalAtomicXor64
            | GlobalAtomicExchange64
            | GlobalAtomicCompareExchange64
            | StorageAtomicIAdd64
            | StorageAtomicSMin64
            | StorageAtomicUMin64
            | StorageAtomicSMax64
            | StorageAtomicUMax64
            | StorageAtomicAnd64
            | StorageAtomicOr64
            | StorageAtomicXor64
            | StorageAtomicExchange64
            | StorageAtomicCompareExchange64
    ) {
        info.uses_int64 = true;
    }

    let global_store = matches!(
        op,
        WriteGlobalU8
            | WriteGlobalS8
            | WriteGlobalU16
            | WriteGlobalS16
            | WriteGlobal32
            | WriteGlobal64
            | WriteGlobal128
            | GlobalAtomicIAdd32
            | GlobalAtomicSMin32
            | GlobalAtomicUMin32
            | GlobalAtomicSMax32
            | GlobalAtomicUMax32
            | GlobalAtomicInc32
            | GlobalAtomicDec32
            | GlobalAtomicAnd32
            | GlobalAtomicOr32
            | GlobalAtomicXor32
            | GlobalAtomicExchange32
            | GlobalAtomicCompareExchange32
            | GlobalAtomicIAdd64
            | GlobalAtomicSMin64
            | GlobalAtomicUMin64
            | GlobalAtomicSMax64
            | GlobalAtomicUMax64
            | GlobalAtomicAnd64
            | GlobalAtomicOr64
            | GlobalAtomicXor64
            | GlobalAtomicExchange64
            | GlobalAtomicCompareExchange64
            | GlobalAtomicIAdd32x2
            | GlobalAtomicSMin32x2
            | GlobalAtomicUMin32x2
            | GlobalAtomicSMax32x2
            | GlobalAtomicUMax32x2
            | GlobalAtomicAnd32x2
            | GlobalAtomicOr32x2
            | GlobalAtomicXor32x2
            | GlobalAtomicExchange32x2
            | GlobalAtomicAddF32
            | GlobalAtomicAddF16x2
            | GlobalAtomicAddF32x2
            | GlobalAtomicMinF16x2
            | GlobalAtomicMinF32x2
            | GlobalAtomicMaxF16x2
            | GlobalAtomicMaxF32x2
    );
    let global_load = matches!(
        op,
        LoadGlobalU8
            | LoadGlobalS8
            | LoadGlobalU16
            | LoadGlobalS16
            | LoadGlobal32
            | LoadGlobal64
            | LoadGlobal128
    );
    if global_store || global_load {
        info.stores_global_memory |= global_store;
        info.uses_int64 = true;
        info.uses_global_memory = true;
        info.used_constant_buffer_types |= Type::U32 | Type::U32X2;
        info.used_storage_buffer_types |= Type::U32 | Type::U32X2 | Type::U32X4;
    }
    match op {
        LoadGlobalU8 | LoadGlobalS8 | WriteGlobalU8 | WriteGlobalS8 => {
            info.used_storage_buffer_types |= Type::U8;
        }
        LoadGlobalU16 | LoadGlobalS16 | WriteGlobalU16 | WriteGlobalS16 => {
            info.used_storage_buffer_types |= Type::U16;
        }
        _ => {}
    }
    if matches!(op, LoadLocal | WriteLocal) {
        info.uses_local_memory = true;
    }

    match op {
        DemoteToHelperInvocation => info.uses_demote_to_helper_invocation = true,
        GetAttribute | GetAttributeU32 | InterpolateAtOffset => {
            if let Value::Attribute(attribute) = inst.arg(0) {
                info.loads.set(attribute, true);
            }
        }
        SetAttribute => {
            if let Value::Attribute(attribute) = inst.arg(0) {
                info.stores.set(attribute, true);
            }
        }
        GetPatch => {
            if let Value::Patch(patch) = inst.arg(0)
                && patch.is_generic()
            {
                mark_patch(info, patch);
            }
        }
        SetPatch => {
            if let Value::Patch(patch) = inst.arg(0) {
                set_patch(info, patch);
            }
        }
        GetAttributeIndexed => info.loads_indexed_attributes = true,
        SetAttributeIndexed => info.stores_indexed_attributes = true,
        SetFragColor => {
            if let (Some(rt), Some(component)) = (inst.arg(0).imm_u32(), inst.arg(1).imm_u32())
                && (rt as usize) < info.stores_frag_color.len()
            {
                info.stores_frag_color[rt as usize] = true;
                info.frag_color_components[rt as usize] |= 1 << (component & 3);
            }
        }
        SetSampleMask => info.stores_sample_mask = true,
        SetFragDepth => info.stores_frag_depth = true,
        WorkgroupId => info.uses_workgroup_id = true,
        LocalInvocationId => info.uses_local_invocation_id = true,
        InvocationId => info.uses_invocation_id = true,
        InvocationInfo => info.uses_invocation_info = true,
        SampleId => info.uses_sample_id = true,
        IsHelperInvocation => info.uses_is_helper_invocation = true,
        ResolutionDownFactor | IsTextureScaled | IsImageScaled => {
            info.uses_rescaling_uniform = true;
        }
        TensorMulAdd | TensorMulAddF32 => {
            info.uses_subgroup_shuffles = true;
            info.uses_fp32_denorms_preserve = true;
            info.uses_subgroup_invocation_id = true;
        }
        LaneId => info.uses_subgroup_invocation_id = true,
        ShuffleIndex | ShuffleUp | ShuffleDown | ShuffleButterfly => {
            info.uses_subgroup_shuffles = true;
        }
        QuadBroadcast | QuadSwap => info.uses_quad_shuffles = true,
        GetCbufU8 | GetCbufS8 | GetCbufU16 | GetCbufS16 | GetCbufU32 | GetCbufF32
        | GetCbufU32x2 => visit_cbuf_read(info, prog, id),
        BindlessImageSampleImplicitLod
        | BindlessImageSampleExplicitLod
        | BindlessImageSampleDrefImplicitLod
        | BindlessImageSampleDrefExplicitLod
        | BindlessImageGather
        | BindlessImageGatherDref
        | BindlessImageFetch
        | BindlessImageQueryDimensions
        | BindlessImageQueryLod
        | BindlessImageQuerySamples
        | BindlessImageGradient
        | BoundImageSampleImplicitLod
        | BoundImageSampleExplicitLod
        | BoundImageSampleDrefImplicitLod
        | BoundImageSampleDrefExplicitLod
        | BoundImageGather
        | BoundImageGatherDref
        | BoundImageFetch
        | BoundImageQueryDimensions
        | BoundImageQueryLod
        | BoundImageQuerySamples
        | BoundImageGradient
        | ImageGather
        | ImageGatherDref
        | ImageFetch
        | ImageQueryDimensions
        | ImageQuerySamples
        | ImageGradient => {
            info.uses_sampled_1d |= is_1d(texture.ty);
            info.uses_sparse_residency |= sparse();
        }
        ImageSampleImplicitLod
        | ImageSampleExplicitLod
        | ImageSampleDrefImplicitLod
        | ImageSampleDrefExplicitLod
        | ImageQueryLod => {
            info.uses_sampled_1d |= is_1d(texture.ty);
            info.uses_shadow_lod |= texture.is_depth;
            info.uses_sparse_residency |= sparse();
        }
        ImageRead => {
            info.uses_typeless_image_reads |= texture.image_format == ImageFormat::Typeless;
            info.uses_image_1d |= is_1d(texture.ty);
            info.uses_sparse_residency |= sparse();
        }
        ImageWrite => {
            info.uses_typeless_image_writes |= texture.image_format == ImageFormat::Typeless;
            info.uses_image_buffers |= texture.ty == TextureType::Buffer;
            info.uses_image_1d |= is_1d(texture.ty);
        }
        SubgroupEqMask | SubgroupLtMask | SubgroupLeMask | SubgroupGtMask | SubgroupGeMask => {
            info.uses_subgroup_mask = true;
        }
        VoteAll | VoteAny | VoteEqual | SubgroupBallot => info.uses_subgroup_vote = true,
        FSwizzleAdd => info.uses_fswzadd = true,
        DPdxFine | DPdyFine | DPdxCoarse | DPdyCoarse => info.uses_derivatives = true,
        LoadStorageU8 | LoadStorageS8 | WriteStorageU8 | WriteStorageS8 => {
            info.used_storage_buffer_types |= Type::U8;
        }
        LoadStorageU16 | LoadStorageS16 | WriteStorageU16 | WriteStorageS16 => {
            info.used_storage_buffer_types |= Type::U16;
        }
        LoadStorage32
        | WriteStorage32
        | StorageAtomicIAdd32
        | StorageAtomicUMin32
        | StorageAtomicUMax32
        | StorageAtomicAnd32
        | StorageAtomicOr32
        | StorageAtomicXor32
        | StorageAtomicExchange32
        | StorageAtomicCompareExchange32 => info.used_storage_buffer_types |= Type::U32,
        LoadStorage64
        | WriteStorage64
        | StorageAtomicIAdd32x2
        | StorageAtomicSMin32x2
        | StorageAtomicUMin32x2
        | StorageAtomicSMax32x2
        | StorageAtomicUMax32x2
        | StorageAtomicAnd32x2
        | StorageAtomicOr32x2
        | StorageAtomicXor32x2
        | StorageAtomicExchange32x2 => info.used_storage_buffer_types |= Type::U32X2,
        LoadStorage128 | WriteStorage128 => info.used_storage_buffer_types |= Type::U32X4,
        SharedAtomicSMin32 => info.uses_atomic_s32_min = true,
        SharedAtomicSMax32 => info.uses_atomic_s32_max = true,
        SharedAtomicInc32 => info.uses_shared_increment = true,
        SharedAtomicDec32 => info.uses_shared_decrement = true,
        SharedAtomicExchange64 | SharedAtomicCompareExchange64 => {
            info.uses_int64_bit_atomics = true;
        }
        GlobalAtomicInc32 | StorageAtomicInc32 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_global_increment = true;
        }
        GlobalAtomicDec32 | StorageAtomicDec32 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_global_decrement = true;
        }
        GlobalAtomicAddF32 | StorageAtomicAddF32 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_f32_add = true;
        }
        GlobalAtomicAddF16x2 | StorageAtomicAddF16x2 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_f16x2_add = true;
        }
        GlobalAtomicAddF32x2 | StorageAtomicAddF32x2 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_f32x2_add = true;
        }
        GlobalAtomicMinF16x2 | StorageAtomicMinF16x2 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_f16x2_min = true;
        }
        GlobalAtomicMinF32x2 | StorageAtomicMinF32x2 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_f32x2_min = true;
        }
        GlobalAtomicMaxF16x2 | StorageAtomicMaxF16x2 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_f16x2_max = true;
        }
        GlobalAtomicMaxF32x2 | StorageAtomicMaxF32x2 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_f32x2_max = true;
        }
        StorageAtomicSMin32 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_s32_min = true;
        }
        StorageAtomicSMax32 => {
            info.used_storage_buffer_types |= Type::U32;
            info.uses_atomic_s32_max = true;
        }
        GlobalAtomicIAdd64
        | GlobalAtomicSMin64
        | GlobalAtomicUMin64
        | GlobalAtomicSMax64
        | GlobalAtomicUMax64
        | GlobalAtomicAnd64
        | GlobalAtomicOr64
        | GlobalAtomicXor64
        | GlobalAtomicExchange64
        | GlobalAtomicCompareExchange64
        | StorageAtomicIAdd64
        | StorageAtomicSMin64
        | StorageAtomicUMin64
        | StorageAtomicSMax64
        | StorageAtomicUMax64
        | StorageAtomicAnd64
        | StorageAtomicOr64
        | StorageAtomicXor64
        | StorageAtomicExchange64
        | StorageAtomicCompareExchange64 => {
            info.used_storage_buffer_types |= Type::U64 | Type::U32X2;
            info.uses_int64_bit_atomics = true;
        }
        BindlessImageAtomicIAdd32
        | BindlessImageAtomicSMin32
        | BindlessImageAtomicUMin32
        | BindlessImageAtomicSMax32
        | BindlessImageAtomicUMax32
        | BindlessImageAtomicInc32
        | BindlessImageAtomicDec32
        | BindlessImageAtomicAnd32
        | BindlessImageAtomicOr32
        | BindlessImageAtomicXor32
        | BindlessImageAtomicExchange32
        | BoundImageAtomicIAdd32
        | BoundImageAtomicSMin32
        | BoundImageAtomicUMin32
        | BoundImageAtomicSMax32
        | BoundImageAtomicUMax32
        | BoundImageAtomicInc32
        | BoundImageAtomicDec32
        | BoundImageAtomicAnd32
        | BoundImageAtomicOr32
        | BoundImageAtomicXor32
        | BoundImageAtomicExchange32
        | ImageAtomicIAdd32
        | ImageAtomicSMin32
        | ImageAtomicUMin32
        | ImageAtomicSMax32
        | ImageAtomicUMax32
        | ImageAtomicInc32
        | ImageAtomicDec32
        | ImageAtomicAnd32
        | ImageAtomicOr32
        | ImageAtomicXor32
        | ImageAtomicExchange32 => {
            info.uses_atomic_image_u32 = true;
            info.uses_image_1d |= is_1d(texture.ty);
        }
        _ => {}
    }
}

fn visit_fp_modifiers(info: &mut ShaderInfo, prog: &Program, id: InstId) {
    use Opcode::*;
    let inst = prog.inst(id);
    let (flush, preserve) = match inst.op {
        FPAdd16 | FPFma16 | FPMul16 | FPRoundEven16 | FPFloor16 | FPCeil16 | FPTrunc16 => (
            &mut info.uses_fp16_denorms_flush,
            &mut info.uses_fp16_denorms_preserve,
        ),
        FPAdd32
        | FPFma32
        | FPMul32
        | FPMin32
        | FPMax32
        | FPRoundEven32
        | FPFloor32
        | FPCeil32
        | FPTrunc32
        | FPOrdEqual32
        | FPUnordEqual32
        | FPOrdNotEqual32
        | FPUnordNotEqual32
        | FPOrdLessThan32
        | FPUnordLessThan32
        | FPOrdGreaterThan32
        | FPUnordGreaterThan32
        | FPOrdLessThanEqual32
        | FPUnordLessThanEqual32
        | FPOrdGreaterThanEqual32
        | FPUnordGreaterThanEqual32
        | ConvertF16F32
        | ConvertF64F32 => (
            &mut info.uses_fp32_denorms_flush,
            &mut info.uses_fp32_denorms_preserve,
        ),
        _ => return,
    };
    match FpControl::from_flags(inst.flags).fmz_mode {
        FmzMode::DontCare => {}
        FmzMode::Ftz | FmzMode::Fmz => *flush = true,
        FmzMode::None => *preserve = true,
    }
}

fn program_header(env: &dyn ShaderEnvironment) -> Option<ProgramHeader> {
    match env.arch() {
        ShaderArch::Maxwell => {
            let address = env.start_address().checked_sub(ProgramHeader::SIZE)?;
            Some(ProgramHeader::read(env, address))
        }
        ShaderArch::Ampere => Some(ProgramHeader {
            words: ampere::read_sph(env),
        }),
    }
}

/// Copies one direction of a VTG header's attribute map into `state`, returning the clip
/// distance mask.
fn apply_vtg_map(
    state: &mut VaryingState,
    generic: impl Fn(usize) -> [bool; 4],
    b: SystemValuesB,
    c: SystemValuesC,
) -> u8 {
    for index in 0..Attribute::NUM_GENERICS {
        for (component, used) in generic(index as usize).into_iter().enumerate() {
            state.set(Attribute::generic(index, component as u16), used);
        }
    }
    let clip = c.clip_distances();
    for index in 0..8 {
        state.set(
            Attribute::CLIP_DISTANCE_0.offset(index),
            (clip >> index) & 1 != 0,
        );
    }
    let system = [
        (
            Attribute::PRIMITIVE_ID,
            b.contains(SystemValuesB::PRIMITIVE_ARRAY_ID),
        ),
        (Attribute::LAYER, b.contains(SystemValuesB::RT_ARRAY_INDEX)),
        (
            Attribute::VIEWPORT_INDEX,
            b.contains(SystemValuesB::VIEWPORT_INDEX),
        ),
        (Attribute::POINT_SIZE, b.contains(SystemValuesB::POINT_SIZE)),
        (Attribute::POSITION_X, b.contains(SystemValuesB::POSITION_X)),
        (Attribute::POSITION_Y, b.contains(SystemValuesB::POSITION_Y)),
        (Attribute::POSITION_Z, b.contains(SystemValuesB::POSITION_Z)),
        (Attribute::POSITION_W, b.contains(SystemValuesB::POSITION_W)),
        (
            Attribute::POINT_SPRITE_S,
            c.contains(SystemValuesC::POINT_SPRITE_S),
        ),
        (
            Attribute::POINT_SPRITE_T,
            c.contains(SystemValuesC::POINT_SPRITE_T),
        ),
        (
            Attribute::FOG_COORDINATE,
            c.contains(SystemValuesC::FOG_COORDINATE),
        ),
        (
            Attribute::TESS_EVAL_POINT_U,
            c.contains(SystemValuesC::TESS_EVAL_POINT_U),
        ),
        (
            Attribute::TESS_EVAL_POINT_V,
            c.contains(SystemValuesC::TESS_EVAL_POINT_V),
        ),
        (
            Attribute::INSTANCE_ID,
            c.contains(SystemValuesC::INSTANCE_ID),
        ),
        (Attribute::VERTEX_ID, c.contains(SystemValuesC::VERTEX_ID)),
    ];
    for (attribute, used) in system {
        state.set(attribute, used);
    }
    clip
}

/// Indexed attribute accesses can touch anything, so the header's maps say what they use.
fn gather_info_from_header(env: &dyn ShaderEnvironment, info: &mut ShaderInfo) {
    let stage = env.stage();
    if stage == ShaderStage::Compute
        || !(info.loads_indexed_attributes || info.stores_indexed_attributes)
    {
        return;
    }
    let Some(header) = program_header(env) else {
        return;
    };
    if stage == ShaderStage::Fragment {
        if !info.loads_indexed_attributes {
            return;
        }
        let ps = header.ps();
        for index in 0..Attribute::NUM_GENERICS {
            let map = ps.generic_input_map(index as usize);
            for (component, imap) in map.into_iter().enumerate() {
                info.loads.set(
                    Attribute::generic(index, component as u16),
                    imap != PixelImap::Unused,
                );
            }
        }
        return;
    }
    let vtg = header.vtg();
    if info.loads_indexed_attributes {
        apply_vtg_map(
            &mut info.loads,
            |index| vtg.input_generic(index),
            vtg.imap_system_b(),
            vtg.imap_system_c(),
        );
    }
    if info.stores_indexed_attributes {
        let clip = apply_vtg_map(
            &mut info.stores,
            |index| vtg.output_generic(index),
            vtg.omap_system_b(),
            vtg.omap_system_c(),
        );
        if clip != 0 {
            info.used_clip_distances = u8::BITS - clip.leading_zeros();
        }
    }
}
