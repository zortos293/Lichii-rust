//! TENSOR_MUL_ADD, from the software path of the C++ `emit_spirv_tensor.cpp`: HMMA.16816.F16
//! (depth 16) and HMMA.1688.F16 (depth 8) of a 32-lane warp, each lane holding its share of A
//! (four words), B and C (two words each) as packed halves. The lanes trade operands with
//! subgroup shuffles and accumulate in f32, rounding to half after every K=8 group as the
//! hardware does. TENSOR_MUL_ADD_F32 is the .F32 form: C and D are four f32 words each, and
//! nothing rounds to half.
//!
//! With [`Profile::support_cooperative_hmma`] a compute program's HMMA.16816 instead runs on a
//! 16x8x16 cooperative matrix, which needs the Vulkan memory model for the whole module.

use rspirv::binary::Assemble;
use rspirv::dr::{self, Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, GlslStd450Op, LoopControl, MemoryModel, Scope, StorageClass, Word,
};

use crate::backend::spirv::context::EmitContext;
use crate::backend::spirv::{EmitError, EmitResult};
use crate::ir::{InstId, Opcode};
use crate::translate::TranslatedProgram;
use crate::{Profile, ShaderStage};

const WARP: u32 = 32;
const MATRIX_A: u32 = 0;
const MATRIX_B: u32 = 1;
const MATRIX_ACCUMULATOR: u32 = 2;

/// Words of one lane in the buffer of [`cooperative_hmma_test`]: A in 0..4, B in 4..6, C in
/// 6..10 (6..8 with f16 accumulators) and D, which the test writes, in 10..14 (10..12).
pub const COOPERATIVE_HMMA_TEST_LANE_WORDS: u32 = 16;

/// Whether `program` runs its HMMA.16816 on cooperative matrices: a compute program whose
/// workgroup rows fill whole warps, since the pipeline runs with full subgroups, and that needs
/// no coherent memory, which the Vulkan memory model cannot express.
pub(crate) fn uses_cooperative_hmma(profile: &Profile, program: &TranslatedProgram) -> bool {
    if !(profile.support_cooperative_hmma || profile.support_cooperative_hmma_f32)
        || program.stage != ShaderStage::Compute
        || program.workgroup_size[0] == 0
        || !program.workgroup_size[0].is_multiple_of(WARP)
        || program.info.requires_coherent_global_memory
        || program.info.requires_coherent_image_memory
    {
        return false;
    }
    let listed = |name: &str| {
        std::env::var(name).ok().map(|list| {
            list.split(',')
                .any(|h| u64::from_str_radix(h.trim(), 16).ok() == Some(program.hash))
        })
    };
    if listed("LIRU14_COOP_ONLY") == Some(false) || listed("LIRU14_COOP_SKIP") == Some(true) {
        return false;
    }
    let ir = &program.ir;
    ir.post_order
        .iter()
        .flat_map(|&block| ir.insts_of(block))
        .any(|id| cooperative(profile, ir.op(id), ir.inst(id).flags))
}

fn cooperative(profile: &Profile, op: Opcode, depth: u32) -> bool {
    depth == 16
        && match op {
            Opcode::TensorMulAdd => profile.support_cooperative_hmma,
            Opcode::TensorMulAddF32 => profile.support_cooperative_hmma_f32,
            _ => false,
        }
}

/// The types of one cooperative HMMA.16816: A is 16x16, B and the accumulator 16x8.
struct CooperativeTypes {
    u32: Word,
    f16: Word,
    f16x2: Word,
    f32: Word,
    a: Word,
    b: Word,
    c: Word,
    result: Word,
    f32_accumulators: bool,
}

/// Defines the matrix types; `constant` returns a u32 constant.
fn cooperative_types(
    b: &mut Builder,
    u32_t: Word,
    f32_t: Word,
    f32_accumulators: bool,
    mut constant: impl FnMut(&mut Builder, u32) -> Word,
) -> CooperativeTypes {
    let f16 = b.type_float(16, None);
    let f16x2 = b.type_vector(f16, 2);
    let scope = constant(b, Scope::Subgroup as u32);
    let rows = constant(b, 16);
    let mut matrix = |b: &mut Builder, component, columns, usage| {
        let columns = constant(b, columns);
        let usage = constant(b, usage);
        b.type_cooperative_matrix_khr(component, scope, rows, columns, usage)
    };
    let a = matrix(b, f16, 16, MATRIX_A);
    let bm = matrix(b, f16, 8, MATRIX_B);
    let accumulator = if f32_accumulators { f32_t } else { f16 };
    let c = matrix(b, accumulator, 8, MATRIX_ACCUMULATOR);
    let result = b.type_vector(u32_t, if f32_accumulators { 4 } else { 2 });
    CooperativeTypes {
        u32: u32_t,
        f16,
        f16x2,
        f32: f32_t,
        a,
        b: bm,
        c,
        result,
        f32_accumulators,
    }
}

/// Inserts the two halves of word `word` of `vector` as elements `2 * slot` and `2 * slot + 1`.
fn insert_halves(
    b: &mut Builder,
    t: &CooperativeTypes,
    matrix_type: Word,
    mut matrix: Word,
    vector: Word,
    word: u32,
    slot: u32,
) -> Result<Word, dr::Error> {
    let packed = b.composite_extract(t.u32, None, vector, [word])?;
    let halves = b.bitcast(t.f16x2, None, packed)?;
    for half in 0..2 {
        let element = b.composite_extract(t.f16, None, halves, [half])?;
        matrix = b.composite_insert(matrix_type, None, element, matrix, [slot * 2 + half])?;
    }
    Ok(matrix)
}

/// HMMA.16816 on cooperative matrices. Each lane already holds its share of the matrices, so
/// nothing moves between lanes: the lane's words become the matrices' local elements. Vulkan
/// leaves their order to the driver; NVIDIA's is the guest's except that A's middle words swap,
/// which [`cooperative_hmma_test`] checks on the device.
fn cooperative_mul_add(
    b: &mut Builder,
    t: &CooperativeTypes,
    a: Word,
    bm: Word,
    c: Word,
) -> Result<Word, dr::Error> {
    let mut matrix_a = b.undef(t.a, None);
    for (slot, word) in [0, 2, 1, 3].into_iter().enumerate() {
        matrix_a = insert_halves(b, t, t.a, matrix_a, a, word, slot as u32)?;
    }
    let mut matrix_b = b.undef(t.b, None);
    for slot in 0..2 {
        matrix_b = insert_halves(b, t, t.b, matrix_b, bm, slot, slot)?;
    }
    let mut matrix_c = b.undef(t.c, None);
    if t.f32_accumulators {
        for element in 0..4 {
            let word = b.composite_extract(t.u32, None, c, [element])?;
            let value = b.bitcast(t.f32, None, word)?;
            matrix_c = b.composite_insert(t.c, None, value, matrix_c, [element])?;
        }
    } else {
        for slot in 0..2 {
            matrix_c = insert_halves(b, t, t.c, matrix_c, c, slot, slot)?;
        }
    }
    let d = b.cooperative_matrix_mul_add_khr(t.c, None, matrix_a, matrix_b, matrix_c, None)?;
    let mut words = [0; 4];
    if t.f32_accumulators {
        for (element, word) in words.iter_mut().enumerate() {
            let value = b.composite_extract(t.f32, None, d, [element as u32])?;
            *word = b.bitcast(t.u32, None, value)?;
        }
        return b.composite_construct(t.result, None, words);
    }
    for (index, word) in words.iter_mut().take(2).enumerate() {
        let index = index as u32;
        let low = b.composite_extract(t.f16, None, d, [index * 2])?;
        let high = b.composite_extract(t.f16, None, d, [index * 2 + 1])?;
        let halves = b.composite_construct(t.f16x2, None, [low, high])?;
        *word = b.bitcast(t.u32, None, halves)?;
    }
    b.composite_construct(t.result, None, words[..2].iter().copied())
}

/// A compute shader of one 32-lane workgroup for the renderer's self-test: each lane reads its
/// guest registers for HMMA.16816 from word `lane * COOPERATIVE_HMMA_TEST_LANE_WORDS` of the
/// storage buffer at set 0, binding 0, multiplies on cooperative matrices exactly as translated
/// programs do, and writes D after them. The pipeline needs full subgroups.
pub fn cooperative_hmma_test(spirv_version: u32, f32_accumulators: bool) -> Vec<u32> {
    let mut b = Builder::new();
    b.set_version(
        ((spirv_version >> 16) & 0xff) as u8,
        ((spirv_version >> 8) & 0xff) as u8,
    );
    for capability in [
        Capability::Shader,
        Capability::GroupNonUniform,
        Capability::CooperativeMatrixKHR,
        Capability::Float16,
        Capability::VulkanMemoryModel,
        Capability::VulkanMemoryModelDeviceScope,
    ] {
        b.capability(capability);
    }
    b.extension("SPV_KHR_cooperative_matrix");
    if spirv_version < 0x0001_0500 {
        b.extension("SPV_KHR_vulkan_memory_model");
    }
    b.memory_model(AddressingModel::Logical, MemoryModel::Vulkan);
    let void = b.type_void();
    let u32_t = b.type_int(32, 0);
    let f32_t = b.type_float(32, None);
    let u32x2 = b.type_vector(u32_t, 2);
    let u32x4 = b.type_vector(u32_t, 4);
    let t = cooperative_types(&mut b, u32_t, f32_t, f32_accumulators, |b, value| {
        b.constant_bit32(u32_t, value)
    });
    let words = b.type_runtime_array(u32_t);
    b.decorate(words, Decoration::ArrayStride, [Operand::LiteralBit32(4)]);
    let block = b.type_struct([words]);
    b.decorate(block, Decoration::Block, []);
    b.member_decorate(block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let block_pointer = b.type_pointer(None, StorageClass::StorageBuffer, block);
    let buffer = b.variable(block_pointer, None, StorageClass::StorageBuffer, None);
    b.decorate(
        buffer,
        Decoration::DescriptorSet,
        [Operand::LiteralBit32(0)],
    );
    b.decorate(buffer, Decoration::Binding, [Operand::LiteralBit32(0)]);
    let input_u32 = b.type_pointer(None, StorageClass::Input, u32_t);
    let lane_variable = b.variable(input_u32, None, StorageClass::Input, None);
    b.decorate(
        lane_variable,
        Decoration::BuiltIn,
        [Operand::BuiltIn(BuiltIn::SubgroupLocalInvocationId)],
    );
    let word_pointer = b.type_pointer(None, StorageClass::StorageBuffer, u32_t);
    let zero = b.constant_bit32(u32_t, 0);
    let lane_words = b.constant_bit32(u32_t, COOPERATIVE_HMMA_TEST_LANE_WORDS);
    let indices: Vec<Word> = (0..COOPERATIVE_HMMA_TEST_LANE_WORDS)
        .map(|index| b.constant_bit32(u32_t, index))
        .collect();
    let function_type = b.type_function(void, []);
    let main = b
        .begin_function(void, None, FunctionControl::NONE, function_type)
        .expect("no function is open");
    let mut body = || -> Result<(), dr::Error> {
        b.begin_block(None)?;
        let lane = b.load(u32_t, None, lane_variable, None, [])?;
        let base = b.i_mul(u32_t, None, lane, lane_words)?;
        let word_at = |b: &mut Builder, index: u32| -> Result<Word, dr::Error> {
            let offset = b.i_add(u32_t, None, base, indices[index as usize])?;
            b.access_chain(word_pointer, None, buffer, [zero, offset])
        };
        let load = |b: &mut Builder, first: u32, count: u32| -> Result<Word, dr::Error> {
            let mut values = Vec::new();
            for index in first..first + count {
                let pointer = word_at(b, index)?;
                values.push(b.load(u32_t, None, pointer, None, [])?);
            }
            b.composite_construct(if count == 4 { u32x4 } else { u32x2 }, None, values)
        };
        let accumulator_words = if f32_accumulators { 4 } else { 2 };
        let a = load(&mut b, 0, 4)?;
        let bm = load(&mut b, 4, 2)?;
        let c = load(&mut b, 6, accumulator_words)?;
        let d = cooperative_mul_add(&mut b, &t, a, bm, c)?;
        for word in 0..accumulator_words {
            let value = b.composite_extract(u32_t, None, d, [word])?;
            let pointer = word_at(&mut b, 10 + word)?;
            b.store(pointer, value, None, [])?;
        }
        b.ret()?;
        b.end_function()
    };
    body().expect("the test shader is well formed");
    let mut interfaces = vec![lane_variable];
    if spirv_version >= 0x0001_0400 {
        interfaces.push(buffer);
    }
    b.entry_point(ExecutionModel::GLCompute, main, "main", interfaces);
    b.execution_mode(main, ExecutionMode::LocalSize, [WARP, 1, 1]);
    b.module().assemble()
}

impl EmitContext<'_> {
    /// Declares what cooperative matrices need, for a module [`uses_cooperative_hmma`] picked.
    pub(crate) fn enable_cooperative_matrices(&mut self) {
        self.add_capability(Capability::CooperativeMatrixKHR);
        self.add_capability(Capability::Float16);
        self.add_capability(Capability::VulkanMemoryModel);
        self.add_capability(Capability::VulkanMemoryModelDeviceScope);
        self.add_extension("SPV_KHR_cooperative_matrix");
        if self.profile.supported_spirv < 0x0001_0500 {
            self.add_extension("SPV_KHR_vulkan_memory_model");
        }
    }
}

fn cooperative_tensor_mul_add(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    f32_accumulators: bool,
) -> EmitResult<Word> {
    let (u32_t, f32_t) = (ctx.u32[1], ctx.f32[1]);
    let constants = [
        Scope::Subgroup as u32,
        16,
        8,
        MATRIX_A,
        MATRIX_B,
        MATRIX_ACCUMULATOR,
    ]
    .map(|value| (value, ctx.const_u32(value)));
    let constant = |_: &mut Builder, value: u32| {
        constants
            .iter()
            .find(|&&(known, _)| known == value)
            .map_or(0, |&(_, id)| id)
    };
    let t = cooperative_types(&mut ctx.b, u32_t, f32_t, f32_accumulators, constant);
    let a = ctx.def_arg(inst, 0)?;
    let b = ctx.def_arg(inst, 1)?;
    let c = ctx.def_arg(inst, 2)?;
    Ok(cooperative_mul_add(&mut ctx.b, &t, a, b, c)?)
}

/// The function index of a multiply depth: 0 for K=8, 1 for K=16; 2 and 3 with F32
/// accumulators.
fn variant(depth: u32, f32_accumulators: bool) -> EmitResult<usize> {
    let depth = match depth {
        8 => 0,
        16 => 1,
        _ => {
            return Err(EmitError::Unsupported(
                "matrix multiply depth other than 8 or 16",
            ));
        }
    };
    Ok(depth + 2 * usize::from(f32_accumulators))
}

pub(super) fn tensor_mul_add(
    ctx: &mut EmitContext<'_>,
    inst: InstId,
    f32_accumulators: bool,
) -> EmitResult<Word> {
    if ctx.vulkan_memory_model
        && std::env::var_os("LIRU14_COOP_SOFT").is_none()
        && cooperative(ctx.profile, ctx.program.op(inst), ctx.flags(inst))
    {
        return cooperative_tensor_mul_add(ctx, inst, f32_accumulators);
    }
    let variant = variant(ctx.flags(inst), f32_accumulators)?;
    if !ctx.supports_subgroups() || ctx.profile.has_broken_spirv_subgroup_shuffle {
        return Err(EmitError::Unsupported(
            "software HMMA without subgroup shuffles",
        ));
    }
    ctx.add_capability(Capability::GroupNonUniformShuffle);
    ctx.subgroup_invocation_id();
    if ctx.tensor_functions[variant] == 0 {
        ctx.tensor_functions[variant] = ctx.b.id();
    }
    let function = ctx.tensor_functions[variant];
    let a = ctx.def_arg(inst, 0)?;
    let b = ctx.def_arg(inst, 1)?;
    let c = ctx.def_arg(inst, 2)?;
    let result_type = ctx.u32[if f32_accumulators { 4 } else { 2 }];
    Ok(ctx
        .b
        .function_call(result_type, None, function, [a, b, c])?)
}

/// Emits the bodies of the multiply-add functions [`tensor_mul_add`] called.
pub(crate) fn define_functions(ctx: &mut EmitContext<'_>) -> EmitResult<()> {
    for variant in 0..4 {
        let function = ctx.tensor_functions[variant];
        if function != 0 {
            define_function(ctx, function, variant % 2 == 1, variant >= 2)?;
        }
    }
    Ok(())
}

fn imm_and(ctx: &mut EmitContext<'_>, value: Word, mask: u32) -> EmitResult<Word> {
    let mask = ctx.const_u32(mask);
    Ok(ctx.b.bitwise_and(ctx.u32[1], None, value, mask)?)
}

fn add(ctx: &mut EmitContext<'_>, a: Word, b: Word) -> EmitResult<Word> {
    Ok(ctx.b.i_add(ctx.u32[1], None, a, b)?)
}

fn add_imm(ctx: &mut EmitContext<'_>, a: Word, b: u32) -> EmitResult<Word> {
    let b = ctx.const_u32(b);
    add(ctx, a, b)
}

fn unpack(ctx: &mut EmitContext<'_>, packed: Word) -> EmitResult<Word> {
    ctx.glsl_inst(ctx.f32[2], GlslStd450Op::UnpackHalf2x16, &[packed])
}

fn at_least(ctx: &mut EmitContext<'_>, value: Word, bound: u32) -> EmitResult<Word> {
    let bound = ctx.const_u32(bound);
    Ok(ctx.b.u_greater_than_equal(ctx.u1, None, value, bound)?)
}

/// Rounds an f32 to the nearest half, ties to even, keeping it in f32. A round trip through
/// PackHalf2x16 would be the same, but drivers may eliminate it. Normal significands round
/// explicitly; half subnormals lie on the exactly representable 2^-24 grid.
fn round_half(ctx: &mut EmitContext<'_>, value: Word) -> EmitResult<Word> {
    let (u32_t, f32_t) = (ctx.u32[1], ctx.f32[1]);
    let bits = ctx.b.bitcast(u32_t, None, value)?;
    let magnitude = imm_and(ctx, bits, 0x7fff_ffff)?;
    let thirteen = ctx.const_u32(13);
    let shifted = ctx
        .b
        .shift_right_logical(u32_t, None, magnitude, thirteen)?;
    let tie = imm_and(ctx, shifted, 1)?;
    let biased = add_imm(ctx, magnitude, 0xfff)?;
    let biased = add(ctx, biased, tie)?;
    let rounded = imm_and(ctx, biased, !0x1fff)?;
    let overflow = at_least(ctx, rounded, 0x4780_0000)?;
    let infinity = ctx.const_u32(0x7f80_0000);
    let normal = ctx.b.select(u32_t, None, overflow, infinity, rounded)?;
    let magnitude_f = ctx.b.bitcast(f32_t, None, magnitude)?;
    let to_grid = ctx.const_f32(16_777_216.0);
    let scaled = ctx.b.f_mul(f32_t, None, magnitude_f, to_grid)?;
    let scaled = ctx.glsl_inst(f32_t, GlslStd450Op::RoundEven, &[scaled])?;
    let from_grid = ctx.const_f32(1.0 / 16_777_216.0);
    let subnormal = ctx.b.f_mul(f32_t, None, scaled, from_grid)?;
    let subnormal = ctx.b.bitcast(u32_t, None, subnormal)?;
    let is_normal = at_least(ctx, magnitude, 0x3880_0000)?;
    let finite = ctx.b.select(u32_t, None, is_normal, normal, subnormal)?;
    let is_special = at_least(ctx, magnitude, 0x7f80_0000)?;
    let result = ctx.b.select(u32_t, None, is_special, magnitude, finite)?;
    let sign = imm_and(ctx, bits, 0x8000_0000)?;
    let result = ctx.b.bitwise_or(u32_t, None, result, sign)?;
    Ok(ctx.b.bitcast(f32_t, None, result)?)
}

/// Which K=8 group's operand words a group reads: the low ones, or the high ones when the
/// condition holds.
#[derive(Clone, Copy)]
enum Half {
    Low,
    Select(Word),
}

impl Half {
    fn pick(self, ctx: &mut EmitContext<'_>, low: Word, high: Word) -> EmitResult<Word> {
        Ok(match self {
            Half::Low => low,
            Half::Select(upper) => ctx.b.select(ctx.u32[1], None, upper, high, low)?,
        })
    }
}

struct Lanes {
    scope: Word,
    a_base: Word,
    b_base: Word,
}

/// One K=8 group, unrolled so its shuffles overlap: latency-bound small dispatches need that,
/// while unrolling all of K=16 raises register pressure and slows large dispatches.
fn emit_group(
    ctx: &mut EmitContext<'_>,
    lanes: &Lanes,
    inputs: &[Word; 6],
    accumulators: &[Word; 4],
    half: Half,
    round: bool,
) -> EmitResult<()> {
    let (u32_t, f32_t) = (ctx.u32[1], ctx.f32[1]);
    for component in 0..4 {
        let b = half.pick(ctx, inputs[4], inputs[5])?;
        let mut bs = [0; 2];
        for (col, slot) in bs.iter_mut().enumerate() {
            let lane = add_imm(ctx, lanes.b_base, component + col as u32 * 4)?;
            let shuffled = ctx
                .b
                .group_non_uniform_shuffle(u32_t, None, lanes.scope, b, lane)?;
            *slot = unpack(ctx, shuffled)?;
        }
        for row in 0..2 {
            let a = half.pick(ctx, inputs[row], inputs[row + 2])?;
            let lane = add_imm(ctx, lanes.a_base, component)?;
            let shuffled = ctx
                .b
                .group_non_uniform_shuffle(u32_t, None, lanes.scope, a, lane)?;
            let a = unpack(ctx, shuffled)?;
            for (col, &b) in bs.iter().enumerate() {
                let accumulator = accumulators[row * 2 + col];
                let mut value = ctx.b.load(f32_t, None, accumulator, None, [])?;
                for element in 0..2 {
                    let a = ctx.b.composite_extract(f32_t, None, a, [element])?;
                    let b = ctx.b.composite_extract(f32_t, None, b, [element])?;
                    value = ctx.glsl_inst(f32_t, GlslStd450Op::Fma, &[a, b, value])?;
                    ctx.decorate(value, Decoration::NoContraction, &[]);
                }
                ctx.b.store(accumulator, value, None, [])?;
            }
        }
    }
    if !round {
        return Ok(());
    }
    for &accumulator in accumulators {
        let value = ctx.b.load(f32_t, None, accumulator, None, [])?;
        let value = round_half(ctx, value)?;
        ctx.b.store(accumulator, value, None, [])?;
    }
    Ok(())
}

fn define_function(
    ctx: &mut EmitContext<'_>,
    function: Word,
    full_depth: bool,
    f32_accumulators: bool,
) -> EmitResult<()> {
    let (u1, u32_t, u32x2, u32x4) = (ctx.u1, ctx.u32[1], ctx.u32[2], ctx.u32[4]);
    let (f32_t, f32x2) = (ctx.f32[1], ctx.f32[2]);
    let accumulator_type = if f32_accumulators { u32x4 } else { u32x2 };
    let function_type = ctx
        .b
        .type_function(accumulator_type, [u32x4, u32x2, accumulator_type]);
    ctx.b.begin_function(
        accumulator_type,
        Some(function),
        FunctionControl::DONT_INLINE,
        function_type,
    )?;
    let matrix_a = ctx.b.function_parameter(u32x4)?;
    let matrix_b = ctx.b.function_parameter(u32x2)?;
    let matrix_c = ctx.b.function_parameter(accumulator_type)?;
    ctx.b.begin_block(None)?;
    let function_u32 = ctx.type_pointer(StorageClass::Function, u32_t);
    let function_f32 = ctx.type_pointer(StorageClass::Function, f32_t);
    let counter = ctx
        .b
        .variable(function_u32, None, StorageClass::Function, None);
    let accumulators: [Word; 4] = std::array::from_fn(|_| {
        ctx.b
            .variable(function_f32, None, StorageClass::Function, None)
    });
    let mut inputs = [0; 6];
    for (i, input) in inputs[..4].iter_mut().enumerate() {
        *input = ctx.b.composite_extract(u32_t, None, matrix_a, [i as u32])?;
    }
    for i in 0..2 {
        inputs[4 + i] = ctx.b.composite_extract(u32_t, None, matrix_b, [i as u32])?;
    }
    for (i, &accumulator) in accumulators.iter().enumerate() {
        let (row, col) = (i as u32 / 2, i as u32 % 2);
        let value = if f32_accumulators {
            let word = ctx.b.composite_extract(u32_t, None, matrix_c, [i as u32])?;
            ctx.b.bitcast(f32_t, None, word)?
        } else {
            let word = ctx.b.composite_extract(u32_t, None, matrix_c, [row])?;
            let pair = unpack(ctx, word)?;
            ctx.b.composite_extract(f32_t, None, pair, [col])?
        };
        ctx.b.store(accumulator, value, None, [])?;
    }
    ctx.b.store(counter, ctx.u32_zero_value, None, [])?;
    let lane = ctx
        .b
        .load(u32_t, None, ctx.subgroup_local_invocation_id, None, [])?;
    let warp_base = imm_and(ctx, lane, !31)?;
    let row_base = imm_and(ctx, lane, 28)?;
    let quad = imm_and(ctx, lane, 3)?;
    let three = ctx.const_u32(3);
    let col_base = ctx.b.shift_left_logical(u32_t, None, quad, three)?;
    let lanes = Lanes {
        scope: ctx.const_u32(Scope::Subgroup as u32),
        a_base: add(ctx, warp_base, row_base)?,
        b_base: add(ctx, warp_base, col_base)?,
    };
    if full_depth {
        let header = ctx.b.id();
        let body = ctx.b.id();
        let next = ctx.b.id();
        let end = ctx.b.id();
        ctx.b.branch(header)?;
        ctx.b.begin_block(Some(header))?;
        let group = ctx.b.load(u32_t, None, counter, None, [])?;
        let two = ctx.const_u32(2);
        let more = ctx.b.u_less_than(u1, None, group, two)?;
        ctx.b.loop_merge(end, next, LoopControl::DONT_UNROLL, [])?;
        ctx.b.branch_conditional(more, body, end, [])?;
        ctx.b.begin_block(Some(body))?;
        let one = ctx.const_u32(1);
        let upper = ctx.b.i_equal(u1, None, group, one)?;
        emit_group(
            ctx,
            &lanes,
            &inputs,
            &accumulators,
            Half::Select(upper),
            !f32_accumulators,
        )?;
        ctx.b.branch(next)?;
        ctx.b.begin_block(Some(next))?;
        let incremented = add(ctx, group, one)?;
        ctx.b.store(counter, incremented, None, [])?;
        ctx.b.branch(header)?;
        ctx.b.begin_block(Some(end))?;
    } else {
        emit_group(
            ctx,
            &lanes,
            &inputs,
            &accumulators,
            Half::Low,
            !f32_accumulators,
        )?;
    }
    if f32_accumulators {
        let mut words = [0; 4];
        for (word, &accumulator) in words.iter_mut().zip(&accumulators) {
            let value = ctx.b.load(f32_t, None, accumulator, None, [])?;
            *word = ctx.b.bitcast(u32_t, None, value)?;
        }
        let result = ctx.b.composite_construct(u32x4, None, words)?;
        ctx.b.ret_value(result)?;
        ctx.b.end_function()?;
        return Ok(());
    }
    let mut packed = [0; 2];
    for (row, slot) in packed.iter_mut().enumerate() {
        let low = ctx.b.load(f32_t, None, accumulators[row * 2], None, [])?;
        let high = ctx
            .b
            .load(f32_t, None, accumulators[row * 2 + 1], None, [])?;
        let pair = ctx.b.composite_construct(f32x2, None, [low, high])?;
        *slot = ctx.glsl_inst(u32_t, GlslStd450Op::PackHalf2x16, &[pair])?;
    }
    let result = ctx.b.composite_construct(u32x2, None, packed)?;
    ctx.b.ret_value(result)?;
    ctx.b.end_function()?;
    Ok(())
}
