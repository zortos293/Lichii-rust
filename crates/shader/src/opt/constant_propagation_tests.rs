use lichii_common::GpuVAddr;

use super::*;
use crate::info::TexturePixelFormat;
use crate::ir::{BlockId, Pred, Reg};
use crate::opt::{dead_code_elimination, identity_removal, verify};
use crate::{ShaderArch, ShaderStage};

#[derive(Default)]
struct Env {
    replacement: Option<ReplaceConstant>,
    proprietary: bool,
}

impl ShaderEnvironment for Env {
    fn arch(&self) -> ShaderArch {
        ShaderArch::Maxwell
    }
    fn stage(&self) -> ShaderStage {
        ShaderStage::Fragment
    }
    fn start_address(&self) -> GpuVAddr {
        0
    }
    fn read_u64(&self, _: GpuVAddr) -> u64 {
        0
    }
    fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
        index << 16 | offset
    }
    fn shared_memory_size(&self) -> u32 {
        0
    }
    fn workgroup_size(&self) -> [u32; 3] {
        [1; 3]
    }
    fn texture_type(&self, _: u32) -> TextureType {
        TextureType::Color2D
    }
    fn texture_pixel_format(&self, _: u32) -> TexturePixelFormat {
        TexturePixelFormat::A8b8g8r8Unorm
    }
    fn is_texture_pixel_format_integer(&self, _: u32) -> bool {
        false
    }
    fn viewport_transform_state(&self) -> u32 {
        1
    }
    fn texture_bound_buffer(&self) -> u32 {
        2
    }
    fn local_memory_size(&self) -> u32 {
        0
    }
    fn has_hle_macro_state(&self) -> bool {
        self.replacement.is_some()
    }
    fn replace_const_buffer(&self, _: u32, offset: u32) -> Option<ReplaceConstant> {
        self.replacement.filter(|_| offset == 0x10)
    }
    fn is_proprietary_driver(&self) -> bool {
        self.proprietary
    }
}

struct Test {
    prog: Program,
    block: BlockId,
}

impl Test {
    fn new() -> Test {
        let mut prog = Program::new();
        let block = prog.new_block();
        prog.block_order = vec![block];
        prog.post_order = vec![block];
        Test { prog, block }
    }

    fn flags(&mut self, op: Opcode, flags: u32, args: &[Value]) -> Value {
        Value::Inst(self.prog.append(self.block, op, args, flags))
    }

    fn op(&mut self, op: Opcode, args: &[Value]) -> Value {
        self.flags(op, 0, args)
    }

    fn reg(&mut self, n: u8) -> Value {
        self.op(Opcode::GetRegister, &[Value::Reg(Reg::r(n))])
    }

    fn pred(&mut self, n: u8) -> Value {
        self.op(Opcode::GetPred, &[Value::Pred(Pred::p(n))])
    }

    fn attr(&mut self, index: u16) -> Value {
        self.op(
            Opcode::GetAttribute,
            &[
                Value::Attribute(Attribute::generic(index, 0)),
                Value::U32(0),
            ],
        )
    }

    fn keep(&mut self, v: Value) {
        self.op(Opcode::Reference, &[v]);
    }

    fn propagate(&mut self, env: &Env) {
        constant_propagation(env, &mut self.prog);
        assert_eq!(verify(&self.prog), Ok(()));
    }

    fn folded(&mut self, v: Value) -> Value {
        self.prog.resolve(v)
    }

    fn optimized_dump(mut self, env: &Env) -> String {
        self.propagate(env);
        identity_removal(&mut self.prog);
        dead_code_elimination(&mut self.prog);
        assert_eq!(verify(&self.prog), Ok(()));
        self.prog.dump()
    }
}

#[test]
fn integer_immediates_fold() {
    use Opcode::*;
    let mut t = Test::new();
    let cases = [
        (t.op(IAdd32, &[Value::U32(2), Value::U32(3)]), Value::U32(5)),
        (
            t.op(ISub32, &[Value::U32(2), Value::U32(3)]),
            Value::U32(u32::MAX),
        ),
        (
            t.op(IMul32, &[Value::U32(6), Value::U32(7)]),
            Value::U32(42),
        ),
        (
            t.op(IAdd64, &[Value::U64(1 << 40), Value::U64(2)]),
            Value::U64((1 << 40) + 2),
        ),
        (
            t.op(SLessThan, &[Value::U32(u32::MAX), Value::U32(1)]),
            Value::U1(true),
        ),
        (
            t.op(ULessThan, &[Value::U32(u32::MAX), Value::U32(1)]),
            Value::U1(false),
        ),
        (
            t.op(IEqual, &[Value::U32(4), Value::U32(4)]),
            Value::U1(true),
        ),
        (
            t.op(BitwiseXor32, &[Value::U32(0xf0), Value::U32(0xff)]),
            Value::U32(0x0f),
        ),
        (
            t.op(
                ShiftRightArithmetic32,
                &[Value::U32(0x8000_0000), Value::U32(4)],
            ),
            Value::U32(0xf800_0000),
        ),
        (
            t.op(
                BitFieldUExtract,
                &[Value::U32(0xabcd_1234), Value::U32(8), Value::U32(8)],
            ),
            Value::U32(0x12),
        ),
        (
            t.op(
                BitFieldUExtract,
                &[Value::U32(0xabcd_1234), Value::U32(0), Value::U32(32)],
            ),
            Value::U32(0xabcd_1234),
        ),
        (
            t.op(
                BitFieldSExtract,
                &[Value::U32(0xf000), Value::U32(12), Value::U32(4)],
            ),
            Value::U32(u32::MAX),
        ),
        (
            t.op(
                BitFieldInsert,
                &[
                    Value::U32(0),
                    Value::U32(0x1ff),
                    Value::U32(4),
                    Value::U32(8),
                ],
            ),
            Value::U32(0xff0),
        ),
    ];
    t.propagate(&Env::default());
    for (value, expected) in cases {
        assert_eq!(t.folded(value), expected);
    }
}

#[test]
fn undefined_or_flag_read_results_stay() {
    use Opcode::*;
    let mut t = Test::new();
    let wide = t.op(
        BitFieldUExtract,
        &[Value::U32(1), Value::U32(30), Value::U32(8)],
    );
    let shift = t.op(ShiftRightArithmetic32, &[Value::U32(1), Value::U32(32)]);
    let flagged = t.op(IAdd32, &[Value::U32(1), Value::U32(2)]);
    t.op(GetCarryFromOp, &[flagged]);
    t.propagate(&Env::default());
    for v in [wide, shift, flagged] {
        assert_eq!(t.folded(v), v);
    }
}

#[test]
fn additions_merge_immediates_and_drop_zero() {
    use Opcode::*;
    let mut t = Test::new();
    let x = t.reg(1);
    let inner = t.op(IAdd32, &[x, Value::U32(5)]);
    let merged = t.op(IAdd32, &[Value::U32(3), inner]);
    let swapped = t.op(IAdd32, &[Value::U32(7), x]);
    let zero = t.op(IAdd32, &[x, Value::U32(0)]);
    t.propagate(&Env::default());
    let id = |v: Value| v.inst().unwrap();
    assert_eq!(t.prog.inst(id(merged)).args(), &[x, Value::U32(8)]);
    assert_eq!(t.prog.inst(id(swapped)).args(), &[x, Value::U32(7)]);
    assert_eq!(t.folded(zero), x);
}

#[test]
fn xmad_multiply_becomes_one_multiply() {
    use Opcode::*;
    let mut t = Test::new();
    let (a, b) = (t.reg(1), t.reg(2));
    let a_high = t.op(BitFieldUExtract, &[a, Value::U32(16), Value::U32(16)]);
    let high = t.op(IMul32, &[a_high, b]);
    let high = t.op(ShiftLeftLogical32, &[high, Value::U32(16)]);
    let a_low = t.op(BitFieldUExtract, &[a, Value::U32(0), Value::U32(16)]);
    let low = t.op(IMul32, &[a_low, b]);
    let sum = t.op(IAdd32, &[high, low]);
    t.keep(sum);
    assert_eq!(
        t.optimized_dump(&Env::default()),
        "{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = IMul32 %0, %1
  Reference %2
"
    );
}

#[test]
fn xmad_multiply_add_becomes_multiply_and_add() {
    use Opcode::*;
    let mut t = Test::new();
    let (a, b, c) = (t.reg(1), t.reg(2), t.reg(3));
    let bfe = |t: &mut Test, v: Value, offset: u32| {
        t.op(BitFieldUExtract, &[v, Value::U32(offset), Value::U32(16)])
    };
    let v6 = bfe(&mut t, b, 0);
    let v7 = bfe(&mut t, a, 16);
    let v8 = t.op(IMul32, &[v6, v7]);
    let v10 = bfe(&mut t, a, 0);
    let v11 = t.op(BitFieldInsert, &[v8, v10, Value::U32(16), Value::U32(16)]);
    let v15 = bfe(&mut t, b, 0);
    let v16 = bfe(&mut t, a, 0);
    let v17 = t.op(IMul32, &[v15, v16]);
    let v18 = t.op(IAdd32, &[v17, c]);
    let v22 = bfe(&mut t, b, 16);
    let v23 = bfe(&mut t, v11, 16);
    let v24 = t.op(IMul32, &[v22, v23]);
    let v25 = t.op(ShiftLeftLogical32, &[v24, Value::U32(16)]);
    let v26 = t.op(ShiftLeftLogical32, &[v11, Value::U32(16)]);
    let v27 = t.op(IAdd32, &[v26, v18]);
    let result = t.op(IAdd32, &[v25, v27]);
    t.keep(result);
    assert_eq!(
        t.optimized_dump(&Env::default()),
        "{0}:
  %0 = GetRegister R1
  %1 = GetRegister R2
  %2 = GetRegister R3
  %3 = IMul32 %0, %1
  %4 = IAdd32 %3, %2
  Reference %4
"
    );
}

#[test]
fn products_by_zero_and_one_and_recombined_halves_fold() {
    use Opcode::*;
    let mut t = Test::new();
    let x = t.reg(1);
    let by_one = t.op(IMul32, &[x, Value::U32(1)]);
    let by_zero = t.op(IMul32, &[Value::U32(0), x]);
    let high = t.op(BitFieldUExtract, &[x, Value::U32(16), Value::U32(16)]);
    let high = t.op(ShiftLeftLogical32, &[high, Value::U32(16)]);
    let low = t.op(BitFieldUExtract, &[x, Value::U32(0), Value::U32(16)]);
    let halves = t.op(IAdd32, &[low, high]);
    t.propagate(&Env::default());
    assert_eq!(t.folded(by_one), x);
    assert_eq!(t.folded(by_zero), Value::U32(0));
    assert_eq!(t.folded(halves), x);
}

#[test]
fn subtraction_cancels_the_same_cbuf_word() {
    use Opcode::*;
    let mut t = Test::new();
    let x = t.reg(1);
    let c1 = t.op(GetCbufU32, &[Value::U32(0), Value::U32(8)]);
    let c2 = t.op(GetCbufU32, &[Value::U32(0), Value::U32(8)]);
    let other = t.op(GetCbufU32, &[Value::U32(0), Value::U32(12)]);
    let zero = t.op(ISub32, &[c1, c2]);
    let add = t.op(IAdd32, &[x, c1]);
    let back = t.op(ISub32, &[add, c2]);
    let negated = t.op(ISub32, &[c2, add]);
    let different = t.op(ISub32, &[add, other]);
    t.propagate(&Env::default());
    assert_eq!(t.folded(zero), Value::U32(0));
    assert_eq!(t.folded(back), x);
    assert_eq!(t.folded(negated), negated);
    assert_eq!(t.folded(different), different);
}

#[test]
fn logic_selects_and_context_reads_fold() {
    use Opcode::*;
    let mut t = Test::new();
    let p = t.pred(0);
    let (x, y) = (t.reg(1), t.reg(2));
    let cases = [
        (t.op(LogicalAnd, &[p, Value::U1(true)]), p),
        (t.op(LogicalAnd, &[Value::U1(false), p]), Value::U1(false)),
        (t.op(LogicalOr, &[p, Value::U1(true)]), Value::U1(true)),
        (t.op(LogicalOr, &[Value::U1(false), p]), p),
        (t.op(LogicalNot, &[Value::U1(false)]), Value::U1(true)),
        (t.op(SelectU32, &[Value::U1(true), x, y]), x),
        (t.op(SelectU32, &[Value::U1(false), x, y]), y),
        (t.op(GetPred, &[Value::Pred(Pred::PT)]), Value::U1(true)),
        (t.op(GetRegister, &[Value::Reg(Reg::RZ)]), Value::U32(0)),
        (t.op(GetRegister, &[Value::Reg(Reg::URZ)]), Value::U32(0)),
    ];
    let not = t.op(LogicalNot, &[p]);
    let not_not = t.op(LogicalNot, &[not]);
    t.propagate(&Env::default());
    for (value, expected) in cases {
        assert_eq!(t.folded(value), expected);
    }
    assert_eq!(t.folded(not_not), p);
    assert_eq!(t.folded(not), not);
}

#[test]
fn bitcasts_fold_round_trips_and_type_their_source() {
    use Opcode::*;
    let mut t = Test::new();
    let x = t.reg(1);
    let float = t.op(BitCastF32U32, &[x]);
    let round_trip = t.op(BitCastU32F32, &[float]);
    let one = t.op(BitCastF32U32, &[Value::U32(0x3f80_0000)]);
    let cbuf = t.op(GetCbufU32, &[Value::U32(1), Value::U32(4)]);
    let cbuf_float = t.op(BitCastF32U32, &[cbuf]);
    t.keep(cbuf_float);
    let instance = t.op(
        GetAttribute,
        &[Value::Attribute(Attribute::INSTANCE_ID), Value::U32(0)],
    );
    let instance_bits = t.op(BitCastU32F32, &[instance]);
    t.keep(instance_bits);
    let generic = t.attr(0);
    let generic_bits = t.op(BitCastU32F32, &[generic]);
    t.keep(generic_bits);
    t.propagate(&Env::default());
    assert_eq!(t.folded(round_trip), x);
    assert_eq!(t.folded(one), Value::F32(1.0));
    assert_eq!(t.folded(generic_bits), generic_bits);
    let typed = |t: &mut Test, v: Value| {
        let id = t.folded(v).inst().unwrap();
        (t.prog.op(id), t.prog.inst(id).args().to_vec())
    };
    assert_eq!(
        typed(&mut t, cbuf_float),
        (GetCbufF32, vec![Value::U32(1), Value::U32(4)])
    );
    assert_eq!(
        typed(&mut t, instance_bits),
        (
            GetAttributeU32,
            vec![Value::Attribute(Attribute::INSTANCE_ID), Value::U32(0)]
        )
    );
}

#[test]
fn extracts_see_through_inserts_at_other_indices() {
    use Opcode::*;
    let mut t = Test::new();
    let [x, y, z, w, q] = [1, 2, 3, 4, 5].map(|n| t.reg(n));
    let v = t.op(CompositeConstructU32x3, &[x, y, z]);
    let v = t.op(CompositeInsertU32x3, &[v, w, Value::U32(1)]);
    let v = t.op(CompositeInsertU32x3, &[v, q, Value::U32(2)]);
    let extracts = [0, 1, 2].map(|i| t.op(CompositeExtractU32x3, &[v, Value::U32(i)]));
    let again = t.op(CompositeExtractU32x3, &[v, Value::U32(0)]);
    let undef = t.op(UndefU32, &[]);
    let pair = t.op(CompositeConstructU32x2, &[undef, undef]);
    let opaque = t.op(LoadGlobal64, &[Value::U64(0)]);
    let unknown = t.op(CompositeExtractU32x2, &[opaque, Value::U32(0)]);
    let from_pair = t.op(CompositeExtractU32x2, &[pair, Value::U32(1)]);
    t.propagate(&Env::default());
    assert_eq!(extracts.map(|e| t.folded(e)), [x, w, q]);
    assert_eq!(t.folded(again), x);
    assert_eq!(t.folded(unknown), unknown);
    assert_eq!(t.folded(from_pair), undef);
}

#[test]
fn float_immediates_neutral_adds_and_interpolation_fold() {
    use Opcode::*;
    let mut t = Test::new();
    let (a, w) = (t.attr(0), t.attr(1));
    let sum = t.op(FPAdd32, &[Value::F32(1.5), Value::F32(2.0)]);
    let product = t.op(FPMul32, &[Value::F32(1.5), Value::F32(2.0)]);
    let negated = t.op(FPNeg32, &[Value::F32(1.0)]);
    let plus_zero = t.op(FPAdd32, &[a, Value::F32(0.0)]);
    let minus_zero = t.op(FPAdd32, &[Value::F32(-0.0), a]);
    let scaled = t.op(FPMul32, &[a, w]);
    let recip = t.op(FPRecip32, &[w]);
    let interpolated = t.op(FPMul32, &[scaled, recip]);
    let exact = FpControl {
        no_contraction: true,
        ..FpControl::default()
    };
    let kept = t.flags(FPMul32, exact.to_flags(), &[scaled, recip]);
    let y = t.op(YDirection, &[]);
    let dy = t.op(DPdyFine, &[a]);
    let corrected = t.op(FPMul32, &[y, dy]);
    t.propagate(&Env::default());
    assert_eq!(t.folded(sum), Value::F32(3.5));
    assert_eq!(t.folded(product), Value::F32(3.0));
    assert_eq!(t.folded(negated), Value::F32(-1.0));
    assert_eq!(t.folded(plus_zero), a);
    assert_eq!(t.folded(minus_zero), a);
    assert_eq!(t.folded(interpolated), a);
    assert_eq!(t.folded(kept), kept);
    assert_eq!(t.folded(corrected), dy);
}

fn swizzle_derivative(shuffle: Opcode, swizzle: u32) -> String {
    use Opcode::*;
    let mut t = Test::new();
    let x = t.attr(0);
    let bits = t.op(BitCastU32F32, &[x]);
    let shuffled = match shuffle {
        QuadSwap => t.op(QuadSwap, &[bits, Value::U32(swizzle >> 7)]),
        _ => t.op(
            ShuffleButterfly,
            &[bits, Value::U32(1), Value::U32(3), Value::U32(28)],
        ),
    };
    let float = t.op(BitCastF32U32, &[shuffled]);
    let sum = t.op(FSwizzleAdd, &[float, x, Value::U32(swizzle)]);
    t.keep(sum);
    t.optimized_dump(&Env::default())
}

#[test]
fn quad_swizzle_adds_become_fine_derivatives() {
    assert_eq!(
        swizzle_derivative(Opcode::ShuffleButterfly, 0x99),
        "{0}:
  %0 = GetAttribute Generic0X, #0x0
  %1 = DPdxFine %0
  Reference %1
"
    );
    assert_eq!(
        swizzle_derivative(Opcode::QuadSwap, 0xa5),
        "{0}:
  %0 = GetAttribute Generic0X, #0x0
  %1 = DPdyFine %0
  Reference %1
"
    );
    assert!(swizzle_derivative(Opcode::ShuffleButterfly, 0xa5).contains("FSwizzleAdd"));
}

#[test]
fn cbuf_reads_become_draw_parameters_and_driver_constants() {
    use Opcode::*;
    let mut t = Test::new();
    let base_vertex = t.op(GetCbufU32, &[Value::U32(0), Value::U32(0x10)]);
    let driver = t.op(GetCbufU32, &[Value::U32(1), Value::U32(0x20)]);
    let driver_float = t.op(GetCbufF32, &[Value::U32(1), Value::U32(0x24)]);
    let user = t.op(GetCbufU32, &[Value::U32(2), Value::U32(0x20)]);
    t.propagate(&Env {
        replacement: Some(ReplaceConstant::BaseVertex),
        proprietary: true,
    });
    let id = t.folded(base_vertex).inst().unwrap();
    assert_eq!(t.prog.op(id), GetAttributeU32);
    assert_eq!(t.prog.arg(id, 0), Value::Attribute(Attribute::BASE_VERTEX));
    assert_eq!(t.folded(driver), Value::U32(0x1_0020));
    assert_eq!(t.folded(driver_float), Value::F32(f32::from_bits(0x1_0024)));
    assert_eq!(t.folded(user), user);
}

#[test]
fn cbuf_reads_stay_without_hle_state_or_proprietary_driver() {
    let mut t = Test::new();
    let read = t.op(Opcode::GetCbufU32, &[Value::U32(1), Value::U32(0x10)]);
    t.propagate(&Env::default());
    assert_eq!(t.folded(read), read);
}

#[test]
fn mask_normalisation_drops_the_max() {
    use Opcode::*;
    let mut prog = Program::new();
    let blocks: Vec<BlockId> = (0..3).map(|_| prog.new_block()).collect();
    prog.block_order = blocks.clone();
    prog.post_order = blocks.iter().rev().copied().collect();
    let mut t = Test {
        prog,
        block: blocks[0],
    };
    let p = t.pred(0);
    let sel = t.op(SelectU32, &[p, Value::U32(u32::MAX), Value::U32(0)]);
    let neg = t.op(INeg32, &[sel]);
    let abs = t.op(SMax32, &[sel, neg]);
    let one_or_zero = t.op(SelectU32, &[p, Value::U32(1), Value::U32(0)]);
    let left = t.op(INeg32, &[one_or_zero]);
    t.block = blocks[1];
    let right = t.op(IAdd32, &[one_or_zero, Value::U32(u32::MAX)]);
    t.block = blocks[2];
    let phi = t.prog.append(t.block, Phi, &[], Type::U32.bits());
    t.prog.add_phi_operand(phi, blocks[0], left);
    t.prog.add_phi_operand(phi, blocks[1], right);
    let phi = Value::Inst(phi);
    let neg_phi = t.op(INeg32, &[phi]);
    let normalised = t.op(SMax32, &[neg_phi, phi]);
    let x = t.reg(1);
    let neg_x = t.op(INeg32, &[x]);
    let unknown = t.op(SMax32, &[x, neg_x]);
    t.propagate(&Env::default());
    let id = t.folded(abs).inst().unwrap();
    assert_eq!(
        (t.prog.op(id), t.prog.inst(id).args()),
        (SelectU32, &[p, Value::U32(1), Value::U32(0)][..])
    );
    let id = t.folded(normalised).inst().unwrap();
    assert_eq!(
        (t.prog.op(id), t.prog.inst(id).args()),
        (BitwiseAnd32, &[phi, Value::U32(1)][..])
    );
    assert_eq!(t.folded(unknown), unknown);
}

/// One coordinate of an NVN implicit-LOD 3D sample: quad swizzles that derive the coordinate,
/// its x derivative and its y derivative.
fn swizzled_coordinate(t: &mut Test, x: Value, dx: Value, dy: Value) -> Value {
    use Opcode::*;
    let broadcast = |t: &mut Test, v: Value| {
        let bits = t.op(BitCastU32F32, &[v]);
        let lane = t.op(QuadBroadcast, &[bits, Value::U32(0)]);
        t.op(BitCastF32U32, &[lane])
    };
    let sx = broadcast(t, dx);
    let bits = t.op(BitCastU32F32, &[dy]);
    let lane = t.op(
        ShuffleIndex,
        &[bits, Value::U32(3), Value::U32(3), Value::U32(0)],
    );
    let sy = t.op(BitCastF32U32, &[lane]);
    let inner = t.op(FSwizzleAdd, &[sx, sy, Value::U32(0xcc)]);
    let center = broadcast(t, x);
    t.op(FSwizzleAdd, &[inner, center, Value::U32(0x0f)])
}

fn sample_3d(ndv_is_active: bool) -> (Test, Value, [Value; 9]) {
    let mut t = Test::new();
    let inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| t.attr(i));
    let coords = [0, 1, 2]
        .map(|i| swizzled_coordinate(&mut t, inputs[i * 3], inputs[i * 3 + 1], inputs[i * 3 + 2]));
    let coords = t.op(Opcode::CompositeConstructF32x3, &coords);
    let info = TextureInstInfo {
        ty: TextureType::Color3D,
        ndv_is_active,
        ..TextureInstInfo::default()
    };
    let sample = t.flags(
        Opcode::BoundImageSampleImplicitLod,
        info.to_flags(),
        &[Value::U32(0x14), coords, Value::Void, Value::Void],
    );
    t.keep(sample);
    t.propagate(&Env::default());
    (t, sample, inputs)
}

#[test]
fn swizzled_3d_sample_becomes_a_gradient_sample() {
    let (t, sample, v) = sample_3d(true);
    let id = t.prog.resolve(sample).inst().unwrap();
    assert_eq!(t.prog.op(id), Opcode::BoundImageGradient);
    assert_eq!(
        TextureInstInfo::from_flags(t.prog.inst(id).flags).num_derivatives,
        3
    );
    let composite = |index: usize| {
        let c = t.prog.arg(id, index).inst().unwrap();
        (t.prog.op(c), t.prog.inst(c).args().to_vec())
    };
    assert_eq!(t.prog.arg(id, 0), Value::U32(0x14));
    assert_eq!(
        composite(1),
        (Opcode::CompositeConstructF32x3, vec![v[0], v[3], v[6]])
    );
    assert_eq!(
        composite(2),
        (
            Opcode::CompositeConstructF32x4,
            vec![v[1], v[2], v[4], v[5]]
        )
    );
    assert_eq!(
        composite(3),
        (Opcode::CompositeConstructF32x2, vec![v[7], v[8]])
    );
    assert_eq!(t.prog.arg(id, 4), Value::Void);
}

#[test]
fn sample_3d_without_ndv_stays() {
    let (t, sample, _) = sample_3d(false);
    assert_eq!(t.prog.resolve(sample), sample);
}

#[test]
fn unrelated_ir_is_left_alone() {
    use Opcode::*;
    let build = || {
        let mut t = Test::new();
        let (x, y) = (t.reg(1), t.reg(2));
        let sum = t.op(IAdd32, &[x, y]);
        let a = t.attr(0);
        let product = t.op(FPMul32, &[a, a]);
        let (p, q) = (t.pred(0), t.pred(1));
        let both = t.op(LogicalAnd, &[p, q]);
        let pick = t.op(SelectU32, &[both, sum, x]);
        let cbuf = t.op(GetCbufU32, &[Value::U32(3), sum]);
        for v in [pick, product, cbuf] {
            t.keep(v);
        }
        t
    };
    let before = build().prog.dump();
    let mut t = build();
    t.propagate(&Env::default());
    assert_eq!(t.prog.dump(), before);
}
