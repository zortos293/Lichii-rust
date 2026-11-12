//! Builder that appends IR instructions to a block, choosing typed opcodes from operand types.
//!
//! Area-specific helpers (memory, images, warp, conversions) live in sibling `impl Emitter`
//! blocks so translators of different instruction families can extend it independently.

use super::condition::{Condition, FlowTest};
use super::modifiers::FpControl;
use super::opcode::Opcode;
use super::program::{BlockId, InstId, Program, Value};
use super::types::Type;
use super::value::{Attribute, Patch, Pred, Reg};

pub struct Emitter<'a> {
    pub prog: &'a mut Program,
    pub block: BlockId,
    /// When set, instructions are inserted before this one instead of appended.
    pub insert_before: Option<InstId>,
}

macro_rules! by_type {
    ($self:ident, $v:expr, { $($ty:ident => $op:ident),+ $(,)? }) => {{
        match $self.ty($v) {
            $(t if t == Type::$ty => Opcode::$op,)+
            t => panic!("{} has no variant for type {}", stringify!($($op)|+), t),
        }
    }};
}

impl<'a> Emitter<'a> {
    pub fn new(prog: &'a mut Program, block: BlockId) -> Emitter<'a> {
        Emitter {
            prog,
            block,
            insert_before: None,
        }
    }

    pub fn before(prog: &'a mut Program, inst: InstId) -> Emitter<'a> {
        let block = prog.inst(inst).block();
        Emitter {
            prog,
            block,
            insert_before: Some(inst),
        }
    }

    pub fn ty(&self, v: Value) -> Type {
        self.prog.type_of(v)
    }

    pub fn inst_flags(&mut self, op: Opcode, flags: u32, args: &[Value]) -> Value {
        let id = match self.insert_before {
            Some(before) => self.prog.insert_before(before, op, args, flags),
            None => self.prog.append(self.block, op, args, flags),
        };
        Value::Inst(id)
    }

    pub fn inst(&mut self, op: Opcode, args: &[Value]) -> Value {
        self.inst_flags(op, 0, args)
    }

    fn fp(&mut self, op: Opcode, control: FpControl, args: &[Value]) -> Value {
        self.inst_flags(op, control.to_flags(), args)
    }

    pub fn imm1(&self, v: bool) -> Value {
        Value::U1(v)
    }

    pub fn imm32(&self, v: u32) -> Value {
        Value::U32(v)
    }

    pub fn imm_s32(&self, v: i32) -> Value {
        Value::U32(v as u32)
    }

    pub fn imm_f32(&self, v: f32) -> Value {
        Value::F32(v)
    }

    pub fn imm64(&self, v: u64) -> Value {
        Value::U64(v)
    }

    pub fn imm_f64(&self, v: f64) -> Value {
        Value::F64(v)
    }

    pub fn condition_ref(&mut self, v: Value) -> Value {
        self.inst(Opcode::ConditionRef, &[v])
    }

    pub fn reference(&mut self, v: Value) {
        self.inst(Opcode::Reference, &[v]);
    }

    pub fn prologue(&mut self) {
        self.inst(Opcode::Prologue, &[]);
    }

    pub fn epilogue(&mut self) {
        self.inst(Opcode::Epilogue, &[]);
    }

    pub fn demote_to_helper_invocation(&mut self) {
        self.inst(Opcode::DemoteToHelperInvocation, &[]);
    }

    pub fn emit_vertex(&mut self, stream: Value) {
        self.inst(Opcode::EmitVertex, &[stream]);
    }

    pub fn end_primitive(&mut self, stream: Value) {
        self.inst(Opcode::EndPrimitive, &[stream]);
    }

    pub fn barrier(&mut self) {
        self.inst(Opcode::Barrier, &[]);
    }

    pub fn workgroup_memory_barrier(&mut self) {
        self.inst(Opcode::WorkgroupMemoryBarrier, &[]);
    }

    pub fn device_memory_barrier(&mut self) {
        self.inst(Opcode::DeviceMemoryBarrier, &[]);
    }

    /// Reads a register; the zero registers read as 0.
    pub fn get_reg(&mut self, reg: Reg) -> Value {
        if reg.is_zero() {
            return Value::U32(0);
        }
        self.inst(Opcode::GetRegister, &[Value::Reg(reg)])
    }

    /// Writes a register; writes to the zero registers are dropped.
    pub fn set_reg(&mut self, reg: Reg, value: Value) {
        if reg.is_zero() {
            return;
        }
        self.inst(Opcode::SetRegister, &[Value::Reg(reg), value]);
    }

    pub fn get_pred(&mut self, pred: Pred, negated: bool) -> Value {
        if pred.is_true() {
            return Value::U1(!negated);
        }
        let v = self.inst(Opcode::GetPred, &[Value::Pred(pred)]);
        if negated { self.logical_not(v) } else { v }
    }

    pub fn set_pred(&mut self, pred: Pred, value: Value) {
        if pred.is_true() {
            return;
        }
        self.inst(Opcode::SetPred, &[Value::Pred(pred), value]);
    }

    pub fn get_goto_variable(&mut self, id: u32) -> Value {
        self.inst(Opcode::GetGotoVariable, &[Value::U32(id)])
    }

    pub fn set_goto_variable(&mut self, id: u32, value: Value) {
        self.inst(Opcode::SetGotoVariable, &[Value::U32(id), value]);
    }

    pub fn get_indirect_branch_variable(&mut self) -> Value {
        self.inst(Opcode::GetIndirectBranchVariable, &[])
    }

    pub fn set_indirect_branch_variable(&mut self, value: Value) {
        self.inst(Opcode::SetIndirectBranchVariable, &[value]);
    }

    pub fn get_cbuf(&mut self, binding: Value, byte_offset: Value) -> Value {
        self.inst(Opcode::GetCbufU32, &[binding, byte_offset])
    }

    /// Constant buffer read of 8, 16, 32 or 64 bits; 64 bits returns a U32x2.
    pub fn get_cbuf_sized(
        &mut self,
        binding: Value,
        byte_offset: Value,
        bits: u32,
        signed: bool,
    ) -> Value {
        let op = match (bits, signed) {
            (8, false) => Opcode::GetCbufU8,
            (8, true) => Opcode::GetCbufS8,
            (16, false) => Opcode::GetCbufU16,
            (16, true) => Opcode::GetCbufS16,
            (32, _) => Opcode::GetCbufU32,
            (64, _) => Opcode::GetCbufU32x2,
            _ => panic!("invalid constant buffer read size {bits}"),
        };
        self.inst(op, &[binding, byte_offset])
    }

    pub fn get_float_cbuf(&mut self, binding: Value, byte_offset: Value) -> Value {
        self.inst(Opcode::GetCbufF32, &[binding, byte_offset])
    }

    pub fn get_z_flag(&mut self) -> Value {
        self.inst(Opcode::GetZFlag, &[])
    }

    pub fn get_s_flag(&mut self) -> Value {
        self.inst(Opcode::GetSFlag, &[])
    }

    pub fn get_c_flag(&mut self) -> Value {
        self.inst(Opcode::GetCFlag, &[])
    }

    pub fn get_o_flag(&mut self) -> Value {
        self.inst(Opcode::GetOFlag, &[])
    }

    pub fn set_z_flag(&mut self, v: Value) {
        self.inst(Opcode::SetZFlag, &[v]);
    }

    pub fn set_s_flag(&mut self, v: Value) {
        self.inst(Opcode::SetSFlag, &[v]);
    }

    pub fn set_c_flag(&mut self, v: Value) {
        self.inst(Opcode::SetCFlag, &[v]);
    }

    pub fn set_o_flag(&mut self, v: Value) {
        self.inst(Opcode::SetOFlag, &[v]);
    }

    /// Evaluates a Maxwell condition-code test. The CSM tests are not emulated and read false.
    pub fn flow_test(&mut self, test: FlowTest) -> Value {
        use FlowTest::*;
        match test {
            F | CsmTa | CsmTr | CsmMx | FcsmTa | FcsmTr | FcsmMx => Value::U1(false),
            T => Value::U1(true),
            LT => {
                let (s, z, o) = (self.get_s_flag(), self.get_z_flag(), self.get_o_flag());
                let nz = self.logical_not(z);
                let a = self.logical_and(s, nz);
                self.logical_xor(a, o)
            }
            EQ => {
                let (s, z) = (self.get_s_flag(), self.get_z_flag());
                let ns = self.logical_not(s);
                self.logical_and(ns, z)
            }
            LE => {
                let (s, z, o) = (self.get_s_flag(), self.get_z_flag(), self.get_o_flag());
                let zo = self.logical_or(z, o);
                self.logical_xor(s, zo)
            }
            GT => {
                let (s, z, o) = (self.get_s_flag(), self.get_z_flag(), self.get_o_flag());
                let ns = self.logical_not(s);
                let a = self.logical_xor(ns, o);
                let nz = self.logical_not(z);
                self.logical_and(a, nz)
            }
            NE => {
                let z = self.get_z_flag();
                self.logical_not(z)
            }
            GE => {
                let (s, o) = (self.get_s_flag(), self.get_o_flag());
                let x = self.logical_xor(s, o);
                self.logical_not(x)
            }
            NUM => {
                let (s, z) = (self.get_s_flag(), self.get_z_flag());
                let (ns, nz) = (self.logical_not(s), self.logical_not(z));
                self.logical_or(ns, nz)
            }
            NaN => {
                let (s, z) = (self.get_s_flag(), self.get_z_flag());
                self.logical_and(s, z)
            }
            LTU => {
                let (s, o) = (self.get_s_flag(), self.get_o_flag());
                self.logical_xor(s, o)
            }
            EQU => self.get_z_flag(),
            LEU => {
                let (s, o, z) = (self.get_s_flag(), self.get_o_flag(), self.get_z_flag());
                let x = self.logical_xor(s, o);
                self.logical_or(x, z)
            }
            GTU => {
                let (s, z, o) = (self.get_s_flag(), self.get_z_flag(), self.get_o_flag());
                let ns = self.logical_not(s);
                let zo = self.logical_or(z, o);
                self.logical_xor(ns, zo)
            }
            NEU => {
                let (s, z) = (self.get_s_flag(), self.get_z_flag());
                let nz = self.logical_not(z);
                self.logical_or(s, nz)
            }
            GEU => {
                let (s, z, o) = (self.get_s_flag(), self.get_z_flag(), self.get_o_flag());
                let ns = self.logical_not(s);
                let a = self.logical_or(ns, z);
                self.logical_xor(a, o)
            }
            OFF => {
                let o = self.get_o_flag();
                self.logical_not(o)
            }
            LO => {
                let c = self.get_c_flag();
                self.logical_not(c)
            }
            SFF => {
                let s = self.get_s_flag();
                self.logical_not(s)
            }
            LS => {
                let (z, c) = (self.get_z_flag(), self.get_c_flag());
                let nc = self.logical_not(c);
                self.logical_or(z, nc)
            }
            HI => {
                let (c, z) = (self.get_c_flag(), self.get_z_flag());
                let nz = self.logical_not(z);
                self.logical_and(c, nz)
            }
            SFT => self.get_s_flag(),
            HS => self.get_c_flag(),
            OFT => self.get_o_flag(),
            RLE => {
                let (s, z) = (self.get_s_flag(), self.get_z_flag());
                self.logical_or(s, z)
            }
            RGT => {
                let (s, z) = (self.get_s_flag(), self.get_z_flag());
                let (ns, nz) = (self.logical_not(s), self.logical_not(z));
                self.logical_and(ns, nz)
            }
        }
    }

    /// Evaluates a block condition: predicate, optional second predicate, and flow test.
    pub fn condition(&mut self, cond: Condition) -> Value {
        let mut v = self.get_pred(cond.pred, cond.negated);
        if let Some((pred, negated)) = cond.extra {
            let extra = self.get_pred(pred, negated);
            v = self.logical_and(v, extra);
        }
        if cond.flow_test != FlowTest::T {
            let test = self.flow_test(cond.flow_test);
            v = self.logical_and(v, test);
        }
        v
    }

    pub fn get_attribute(&mut self, attribute: Attribute, vertex: Value) -> Value {
        self.inst(Opcode::GetAttribute, &[Value::Attribute(attribute), vertex])
    }

    pub fn get_attribute_u32(&mut self, attribute: Attribute, vertex: Value) -> Value {
        self.inst(
            Opcode::GetAttributeU32,
            &[Value::Attribute(attribute), vertex],
        )
    }

    pub fn set_attribute(&mut self, attribute: Attribute, value: Value, vertex: Value) {
        self.inst(
            Opcode::SetAttribute,
            &[Value::Attribute(attribute), value, vertex],
        );
    }

    pub fn get_attribute_indexed(&mut self, phys_address: Value, vertex: Value) -> Value {
        self.inst(Opcode::GetAttributeIndexed, &[phys_address, vertex])
    }

    pub fn set_attribute_indexed(&mut self, phys_address: Value, value: Value, vertex: Value) {
        self.inst(Opcode::SetAttributeIndexed, &[phys_address, value, vertex]);
    }

    pub fn get_patch(&mut self, patch: Patch) -> Value {
        self.inst(Opcode::GetPatch, &[Value::Patch(patch)])
    }

    pub fn set_patch(&mut self, patch: Patch, value: Value) {
        self.inst(Opcode::SetPatch, &[Value::Patch(patch), value]);
    }

    pub fn set_frag_color(&mut self, index: u32, component: u32, value: Value) {
        self.inst(
            Opcode::SetFragColor,
            &[Value::U32(index), Value::U32(component), value],
        );
    }

    pub fn set_sample_mask(&mut self, value: Value) {
        self.inst(Opcode::SetSampleMask, &[value]);
    }

    pub fn set_frag_depth(&mut self, value: Value) {
        self.inst(Opcode::SetFragDepth, &[value]);
    }

    pub fn get_zero_from_op(&mut self, op: Value) -> Value {
        self.inst(Opcode::GetZeroFromOp, &[op])
    }

    pub fn get_sign_from_op(&mut self, op: Value) -> Value {
        self.inst(Opcode::GetSignFromOp, &[op])
    }

    pub fn get_carry_from_op(&mut self, op: Value) -> Value {
        self.inst(Opcode::GetCarryFromOp, &[op])
    }

    pub fn get_overflow_from_op(&mut self, op: Value) -> Value {
        self.inst(Opcode::GetOverflowFromOp, &[op])
    }

    pub fn get_sparse_from_op(&mut self, op: Value) -> Value {
        self.inst(Opcode::GetSparseFromOp, &[op])
    }

    pub fn get_in_bounds_from_op(&mut self, op: Value) -> Value {
        self.inst(Opcode::GetInBoundsFromOp, &[op])
    }

    /// Builds a vector from 2, 3 or 4 elements of the same scalar type.
    pub fn composite_construct(&mut self, elements: &[Value]) -> Value {
        let ty = self.ty(elements[0]);
        let op = match (ty, elements.len()) {
            (t, 2) if t == Type::U32 => Opcode::CompositeConstructU32x2,
            (t, 3) if t == Type::U32 => Opcode::CompositeConstructU32x3,
            (t, 4) if t == Type::U32 => Opcode::CompositeConstructU32x4,
            (t, 2) if t == Type::F16 => Opcode::CompositeConstructF16x2,
            (t, 3) if t == Type::F16 => Opcode::CompositeConstructF16x3,
            (t, 4) if t == Type::F16 => Opcode::CompositeConstructF16x4,
            (t, 2) if t == Type::F32 => Opcode::CompositeConstructF32x2,
            (t, 3) if t == Type::F32 => Opcode::CompositeConstructF32x3,
            (t, 4) if t == Type::F32 => Opcode::CompositeConstructF32x4,
            (t, 2) if t == Type::F64 => Opcode::CompositeConstructF64x2,
            (t, 3) if t == Type::F64 => Opcode::CompositeConstructF64x3,
            (t, 4) if t == Type::F64 => Opcode::CompositeConstructF64x4,
            (t, n) => panic!("cannot construct a {n}-element vector of {t}"),
        };
        self.inst(op, elements)
    }

    pub fn composite_extract(&mut self, vector: Value, element: u32) -> Value {
        let op = by_type!(self, vector, {
            U32X2 => CompositeExtractU32x2, U32X3 => CompositeExtractU32x3, U32X4 => CompositeExtractU32x4,
            F16X2 => CompositeExtractF16x2, F16X3 => CompositeExtractF16x3, F16X4 => CompositeExtractF16x4,
            F32X2 => CompositeExtractF32x2, F32X3 => CompositeExtractF32x3, F32X4 => CompositeExtractF32x4,
            F64X2 => CompositeExtractF64x2, F64X3 => CompositeExtractF64x3, F64X4 => CompositeExtractF64x4,
        });
        self.inst(op, &[vector, Value::U32(element)])
    }

    pub fn composite_insert(&mut self, vector: Value, object: Value, element: u32) -> Value {
        let op = by_type!(self, vector, {
            U32X2 => CompositeInsertU32x2, U32X3 => CompositeInsertU32x3, U32X4 => CompositeInsertU32x4,
            F16X2 => CompositeInsertF16x2, F16X3 => CompositeInsertF16x3, F16X4 => CompositeInsertF16x4,
            F32X2 => CompositeInsertF32x2, F32X3 => CompositeInsertF32x3, F32X4 => CompositeInsertF32x4,
            F64X2 => CompositeInsertF64x2, F64X3 => CompositeInsertF64x3, F64X4 => CompositeInsertF64x4,
        });
        self.inst(op, &[vector, object, Value::U32(element)])
    }

    pub fn select(&mut self, cond: Value, t: Value, f: Value) -> Value {
        let op = by_type!(self, t, {
            U1 => SelectU1, U8 => SelectU8, U16 => SelectU16, U32 => SelectU32, U64 => SelectU64,
            F16 => SelectF16, F32 => SelectF32, F64 => SelectF64,
        });
        self.inst(op, &[cond, t, f])
    }

    pub fn bitcast_f32(&mut self, v: Value) -> Value {
        if let Value::U32(bits) = v {
            return Value::F32(f32::from_bits(bits));
        }
        self.inst(Opcode::BitCastF32U32, &[v])
    }

    pub fn bitcast_u32(&mut self, v: Value) -> Value {
        if let Value::F32(f) = v {
            return Value::U32(f.to_bits());
        }
        self.inst(Opcode::BitCastU32F32, &[v])
    }

    pub fn bitcast_f64(&mut self, v: Value) -> Value {
        self.inst(Opcode::BitCastF64U64, &[v])
    }

    pub fn bitcast_u64(&mut self, v: Value) -> Value {
        self.inst(Opcode::BitCastU64F64, &[v])
    }

    pub fn pack_uint_2x32(&mut self, v: Value) -> Value {
        self.inst(Opcode::PackUint2x32, &[v])
    }

    pub fn unpack_uint_2x32(&mut self, v: Value) -> Value {
        self.inst(Opcode::UnpackUint2x32, &[v])
    }

    pub fn pack_half_2x16(&mut self, v: Value) -> Value {
        self.inst(Opcode::PackHalf2x16, &[v])
    }

    pub fn unpack_half_2x16(&mut self, v: Value) -> Value {
        self.inst(Opcode::UnpackHalf2x16, &[v])
    }

    pub fn pack_float_2x16(&mut self, v: Value) -> Value {
        self.inst(Opcode::PackFloat2x16, &[v])
    }

    pub fn unpack_float_2x16(&mut self, v: Value) -> Value {
        self.inst(Opcode::UnpackFloat2x16, &[v])
    }

    pub fn pack_double_2x32(&mut self, v: Value) -> Value {
        self.inst(Opcode::PackDouble2x32, &[v])
    }

    pub fn unpack_double_2x32(&mut self, v: Value) -> Value {
        self.inst(Opcode::UnpackDouble2x32, &[v])
    }

    pub fn fp_add(&mut self, a: Value, b: Value, control: FpControl) -> Value {
        let op = by_type!(self, a, { F16 => FPAdd16, F32 => FPAdd32, F64 => FPAdd64 });
        self.fp(op, control, &[a, b])
    }

    pub fn fp_mul(&mut self, a: Value, b: Value, control: FpControl) -> Value {
        let op = by_type!(self, a, { F16 => FPMul16, F32 => FPMul32, F64 => FPMul64 });
        self.fp(op, control, &[a, b])
    }

    pub fn fp_fma(&mut self, a: Value, b: Value, c: Value, control: FpControl) -> Value {
        let op = by_type!(self, a, { F16 => FPFma16, F32 => FPFma32, F64 => FPFma64 });
        self.fp(op, control, &[a, b, c])
    }

    pub fn fp_abs(&mut self, v: Value) -> Value {
        let op = by_type!(self, v, { F16 => FPAbs16, F32 => FPAbs32, F64 => FPAbs64 });
        self.inst(op, &[v])
    }

    pub fn fp_neg(&mut self, v: Value) -> Value {
        let op = by_type!(self, v, { F16 => FPNeg16, F32 => FPNeg32, F64 => FPNeg64 });
        self.inst(op, &[v])
    }

    pub fn fp_abs_neg(&mut self, v: Value, abs: bool, neg: bool) -> Value {
        let v = if abs { self.fp_abs(v) } else { v };
        if neg { self.fp_neg(v) } else { v }
    }

    pub fn fp_saturate(&mut self, v: Value) -> Value {
        let op =
            by_type!(self, v, { F16 => FPSaturate16, F32 => FPSaturate32, F64 => FPSaturate64 });
        self.inst(op, &[v])
    }

    pub fn fp_clamp(&mut self, v: Value, min: Value, max: Value) -> Value {
        let op = by_type!(self, v, { F16 => FPClamp16, F32 => FPClamp32, F64 => FPClamp64 });
        self.inst(op, &[v, min, max])
    }

    pub fn fp_unary32(&mut self, op: Opcode, v: Value) -> Value {
        self.inst(op, &[v])
    }

    pub fn fp_recip(&mut self, v: Value) -> Value {
        let op = by_type!(self, v, { F32 => FPRecip32, F64 => FPRecip64 });
        self.inst(op, &[v])
    }

    pub fn fp_recip_sqrt(&mut self, v: Value) -> Value {
        let op = by_type!(self, v, { F32 => FPRecipSqrt32, F64 => FPRecipSqrt64 });
        self.inst(op, &[v])
    }

    pub fn fp_round_even(&mut self, v: Value, control: FpControl) -> Value {
        let op =
            by_type!(self, v, { F16 => FPRoundEven16, F32 => FPRoundEven32, F64 => FPRoundEven64 });
        self.fp(op, control, &[v])
    }

    pub fn fp_floor(&mut self, v: Value, control: FpControl) -> Value {
        let op = by_type!(self, v, { F16 => FPFloor16, F32 => FPFloor32, F64 => FPFloor64 });
        self.fp(op, control, &[v])
    }

    pub fn fp_ceil(&mut self, v: Value, control: FpControl) -> Value {
        let op = by_type!(self, v, { F16 => FPCeil16, F32 => FPCeil32, F64 => FPCeil64 });
        self.fp(op, control, &[v])
    }

    pub fn fp_trunc(&mut self, v: Value, control: FpControl) -> Value {
        let op = by_type!(self, v, { F16 => FPTrunc16, F32 => FPTrunc32, F64 => FPTrunc64 });
        self.fp(op, control, &[v])
    }

    pub fn fp_max(&mut self, a: Value, b: Value, control: FpControl) -> Value {
        let op = by_type!(self, a, { F32 => FPMax32, F64 => FPMax64 });
        self.fp(op, control, &[a, b])
    }

    pub fn fp_min(&mut self, a: Value, b: Value, control: FpControl) -> Value {
        let op = by_type!(self, a, { F32 => FPMin32, F64 => FPMin64 });
        self.fp(op, control, &[a, b])
    }

    pub fn fp_is_nan(&mut self, v: Value) -> Value {
        let op = by_type!(self, v, { F16 => FPIsNan16, F32 => FPIsNan32, F64 => FPIsNan64 });
        self.inst(op, &[v])
    }

    pub fn fp_ordered(&mut self, a: Value, b: Value) -> Value {
        let (na, nb) = (self.fp_is_nan(a), self.fp_is_nan(b));
        let (oa, ob) = (self.logical_not(na), self.logical_not(nb));
        self.logical_and(oa, ob)
    }

    pub fn fp_unordered(&mut self, a: Value, b: Value) -> Value {
        let (na, nb) = (self.fp_is_nan(a), self.fp_is_nan(b));
        self.logical_or(na, nb)
    }

    /// Floating-point comparison; `ordered` selects the ordered or unordered form.
    pub fn fp_compare(
        &mut self,
        cmp: FpCompare,
        a: Value,
        b: Value,
        ordered: bool,
        control: FpControl,
    ) -> Value {
        use FpCompare::*;
        let size = match self.ty(a) {
            t if t == Type::F16 => 0,
            t if t == Type::F32 => 1,
            t if t == Type::F64 => 2,
            t => panic!("fp_compare on {t}"),
        };
        const TABLE: [[[Opcode; 3]; 2]; 6] = [
            [
                [
                    Opcode::FPUnordEqual16,
                    Opcode::FPUnordEqual32,
                    Opcode::FPUnordEqual64,
                ],
                [
                    Opcode::FPOrdEqual16,
                    Opcode::FPOrdEqual32,
                    Opcode::FPOrdEqual64,
                ],
            ],
            [
                [
                    Opcode::FPUnordNotEqual16,
                    Opcode::FPUnordNotEqual32,
                    Opcode::FPUnordNotEqual64,
                ],
                [
                    Opcode::FPOrdNotEqual16,
                    Opcode::FPOrdNotEqual32,
                    Opcode::FPOrdNotEqual64,
                ],
            ],
            [
                [
                    Opcode::FPUnordLessThan16,
                    Opcode::FPUnordLessThan32,
                    Opcode::FPUnordLessThan64,
                ],
                [
                    Opcode::FPOrdLessThan16,
                    Opcode::FPOrdLessThan32,
                    Opcode::FPOrdLessThan64,
                ],
            ],
            [
                [
                    Opcode::FPUnordGreaterThan16,
                    Opcode::FPUnordGreaterThan32,
                    Opcode::FPUnordGreaterThan64,
                ],
                [
                    Opcode::FPOrdGreaterThan16,
                    Opcode::FPOrdGreaterThan32,
                    Opcode::FPOrdGreaterThan64,
                ],
            ],
            [
                [
                    Opcode::FPUnordLessThanEqual16,
                    Opcode::FPUnordLessThanEqual32,
                    Opcode::FPUnordLessThanEqual64,
                ],
                [
                    Opcode::FPOrdLessThanEqual16,
                    Opcode::FPOrdLessThanEqual32,
                    Opcode::FPOrdLessThanEqual64,
                ],
            ],
            [
                [
                    Opcode::FPUnordGreaterThanEqual16,
                    Opcode::FPUnordGreaterThanEqual32,
                    Opcode::FPUnordGreaterThanEqual64,
                ],
                [
                    Opcode::FPOrdGreaterThanEqual16,
                    Opcode::FPOrdGreaterThanEqual32,
                    Opcode::FPOrdGreaterThanEqual64,
                ],
            ],
        ];
        let row = match cmp {
            Equal => 0,
            NotEqual => 1,
            LessThan => 2,
            GreaterThan => 3,
            LessThanEqual => 4,
            GreaterThanEqual => 5,
        };
        self.fp(TABLE[row][ordered as usize][size], control, &[a, b])
    }

    pub fn iadd(&mut self, a: Value, b: Value) -> Value {
        let op = by_type!(self, a, { U32 => IAdd32, U64 => IAdd64 });
        self.inst(op, &[a, b])
    }

    pub fn isub(&mut self, a: Value, b: Value) -> Value {
        let op = by_type!(self, a, { U32 => ISub32, U64 => ISub64 });
        self.inst(op, &[a, b])
    }

    pub fn imul(&mut self, a: Value, b: Value) -> Value {
        self.inst(Opcode::IMul32, &[a, b])
    }

    pub fn ineg(&mut self, v: Value) -> Value {
        let op = by_type!(self, v, { U32 => INeg32, U64 => INeg64 });
        self.inst(op, &[v])
    }

    pub fn iabs(&mut self, v: Value) -> Value {
        let op = by_type!(self, v, { U32 => IAbs32, U64 => IAbs64 });
        self.inst(op, &[v])
    }

    pub fn shift_left_logical(&mut self, base: Value, shift: Value) -> Value {
        let op = by_type!(self, base, { U32 => ShiftLeftLogical32, U64 => ShiftLeftLogical64 });
        self.inst(op, &[base, shift])
    }

    pub fn shift_right_logical(&mut self, base: Value, shift: Value) -> Value {
        let op = by_type!(self, base, { U32 => ShiftRightLogical32, U64 => ShiftRightLogical64 });
        self.inst(op, &[base, shift])
    }

    pub fn shift_right_arithmetic(&mut self, base: Value, shift: Value) -> Value {
        let op =
            by_type!(self, base, { U32 => ShiftRightArithmetic32, U64 => ShiftRightArithmetic64 });
        self.inst(op, &[base, shift])
    }

    pub fn bitwise_and(&mut self, a: Value, b: Value) -> Value {
        let op = by_type!(self, a, { U32 => BitwiseAnd32 });
        self.inst(op, &[a, b])
    }

    pub fn bitwise_or(&mut self, a: Value, b: Value) -> Value {
        let op = by_type!(self, a, { U32 => BitwiseOr32 });
        self.inst(op, &[a, b])
    }

    pub fn bitwise_xor(&mut self, a: Value, b: Value) -> Value {
        let op = by_type!(self, a, { U32 => BitwiseXor32 });
        self.inst(op, &[a, b])
    }

    pub fn bitwise_not(&mut self, v: Value) -> Value {
        self.inst(Opcode::BitwiseNot32, &[v])
    }

    pub fn bit_field_insert(
        &mut self,
        base: Value,
        insert: Value,
        offset: Value,
        count: Value,
    ) -> Value {
        self.inst(Opcode::BitFieldInsert, &[base, insert, offset, count])
    }

    pub fn bit_field_extract(
        &mut self,
        base: Value,
        offset: Value,
        count: Value,
        signed: bool,
    ) -> Value {
        let op = if signed {
            Opcode::BitFieldSExtract
        } else {
            Opcode::BitFieldUExtract
        };
        self.inst(op, &[base, offset, count])
    }

    pub fn bit_reverse(&mut self, v: Value) -> Value {
        self.inst(Opcode::BitReverse32, &[v])
    }

    pub fn bit_count(&mut self, v: Value) -> Value {
        self.inst(Opcode::BitCount32, &[v])
    }

    pub fn find_s_msb(&mut self, v: Value) -> Value {
        self.inst(Opcode::FindSMsb32, &[v])
    }

    pub fn find_u_msb(&mut self, v: Value) -> Value {
        self.inst(Opcode::FindUMsb32, &[v])
    }

    pub fn imin(&mut self, a: Value, b: Value, signed: bool) -> Value {
        self.inst(
            if signed {
                Opcode::SMin32
            } else {
                Opcode::UMin32
            },
            &[a, b],
        )
    }

    pub fn imax(&mut self, a: Value, b: Value, signed: bool) -> Value {
        self.inst(
            if signed {
                Opcode::SMax32
            } else {
                Opcode::UMax32
            },
            &[a, b],
        )
    }

    pub fn iclamp(&mut self, v: Value, min: Value, max: Value, signed: bool) -> Value {
        self.inst(
            if signed {
                Opcode::SClamp32
            } else {
                Opcode::UClamp32
            },
            &[v, min, max],
        )
    }

    pub fn icompare(&mut self, cmp: ICompare, a: Value, b: Value) -> Value {
        use ICompare::*;
        let op = match cmp {
            Equal => Opcode::IEqual,
            NotEqual => Opcode::INotEqual,
            SLessThan => Opcode::SLessThan,
            ULessThan => Opcode::ULessThan,
            SLessThanEqual => Opcode::SLessThanEqual,
            ULessThanEqual => Opcode::ULessThanEqual,
            SGreaterThan => Opcode::SGreaterThan,
            UGreaterThan => Opcode::UGreaterThan,
            SGreaterThanEqual => Opcode::SGreaterThanEqual,
            UGreaterThanEqual => Opcode::UGreaterThanEqual,
        };
        self.inst(op, &[a, b])
    }

    pub fn logical_or(&mut self, a: Value, b: Value) -> Value {
        self.inst(Opcode::LogicalOr, &[a, b])
    }

    pub fn logical_and(&mut self, a: Value, b: Value) -> Value {
        self.inst(Opcode::LogicalAnd, &[a, b])
    }

    pub fn logical_xor(&mut self, a: Value, b: Value) -> Value {
        self.inst(Opcode::LogicalXor, &[a, b])
    }

    pub fn logical_not(&mut self, v: Value) -> Value {
        self.inst(Opcode::LogicalNot, &[v])
    }

    /// Float to integer of `bits` (16, 32 or 64) from an F16, F32 or F64 value.
    pub fn convert_f_to_i(&mut self, bits: u32, signed: bool, v: Value) -> Value {
        let src = self.ty(v);
        let op = match (signed, bits, src) {
            (true, 16, t) if t == Type::F16 => Opcode::ConvertS16F16,
            (true, 16, t) if t == Type::F32 => Opcode::ConvertS16F32,
            (true, 16, t) if t == Type::F64 => Opcode::ConvertS16F64,
            (true, 32, t) if t == Type::F16 => Opcode::ConvertS32F16,
            (true, 32, t) if t == Type::F32 => Opcode::ConvertS32F32,
            (true, 32, t) if t == Type::F64 => Opcode::ConvertS32F64,
            (true, 64, t) if t == Type::F16 => Opcode::ConvertS64F16,
            (true, 64, t) if t == Type::F32 => Opcode::ConvertS64F32,
            (true, 64, t) if t == Type::F64 => Opcode::ConvertS64F64,
            (false, 16, t) if t == Type::F16 => Opcode::ConvertU16F16,
            (false, 16, t) if t == Type::F32 => Opcode::ConvertU16F32,
            (false, 16, t) if t == Type::F64 => Opcode::ConvertU16F64,
            (false, 32, t) if t == Type::F16 => Opcode::ConvertU32F16,
            (false, 32, t) if t == Type::F32 => Opcode::ConvertU32F32,
            (false, 32, t) if t == Type::F64 => Opcode::ConvertU32F64,
            (false, 64, t) if t == Type::F16 => Opcode::ConvertU64F16,
            (false, 64, t) if t == Type::F32 => Opcode::ConvertU64F32,
            (false, 64, t) if t == Type::F64 => Opcode::ConvertU64F64,
            (s, b, t) => panic!(
                "no float to {}{b} conversion from {t}",
                if s { "s" } else { "u" }
            ),
        };
        self.inst(op, &[v])
    }

    /// Integer of `src_bits` (8, 16, 32, 64) to a float of `dst_bits` (16, 32, 64).
    pub fn convert_i_to_f(
        &mut self,
        dst_bits: u32,
        src_bits: u32,
        signed: bool,
        v: Value,
        control: FpControl,
    ) -> Value {
        use Opcode::*;
        const SIGNED: [[Opcode; 4]; 3] = [
            [ConvertF16S8, ConvertF16S16, ConvertF16S32, ConvertF16S64],
            [ConvertF32S8, ConvertF32S16, ConvertF32S32, ConvertF32S64],
            [ConvertF64S8, ConvertF64S16, ConvertF64S32, ConvertF64S64],
        ];
        const UNSIGNED: [[Opcode; 4]; 3] = [
            [ConvertF16U8, ConvertF16U16, ConvertF16U32, ConvertF16U64],
            [ConvertF32U8, ConvertF32U16, ConvertF32U32, ConvertF32U64],
            [ConvertF64U8, ConvertF64U16, ConvertF64U32, ConvertF64U64],
        ];
        let d = match dst_bits {
            16 => 0,
            32 => 1,
            64 => 2,
            _ => panic!("invalid float size {dst_bits}"),
        };
        let s = match src_bits {
            8 => 0,
            16 => 1,
            32 => 2,
            64 => 3,
            _ => panic!("invalid integer size {src_bits}"),
        };
        let op = if signed { SIGNED[d][s] } else { UNSIGNED[d][s] };
        self.fp(op, control, &[v])
    }

    pub fn uconvert(&mut self, bits: u32, v: Value) -> Value {
        let src = self.ty(v);
        let op = match (bits, src) {
            (8, t) if t == Type::U8 => return v,
            (8, t) if t == Type::U32 => Opcode::ConvertU8U32,
            (16, t) if t == Type::U16 => return v,
            (16, t) if t == Type::U32 => Opcode::ConvertU16U32,
            (32, t) if t == Type::U8 => Opcode::ConvertU32U8,
            (32, t) if t == Type::U16 => Opcode::ConvertU32U16,
            (32, t) if t == Type::U32 => return v,
            (32, t) if t == Type::U64 => Opcode::ConvertU32U64,
            (64, t) if t == Type::U32 => Opcode::ConvertU64U32,
            (64, t) if t == Type::U64 => return v,
            (b, t) => panic!("no u{b} conversion from {t}"),
        };
        self.inst(op, &[v])
    }

    pub fn fp_convert(&mut self, bits: u32, v: Value, control: FpControl) -> Value {
        let src = self.ty(v);
        let op = match (bits, src) {
            (16, t) if t == Type::F16 => return v,
            (16, t) if t == Type::F32 => Opcode::ConvertF16F32,
            (32, t) if t == Type::F16 => Opcode::ConvertF32F16,
            (32, t) if t == Type::F32 => return v,
            (32, t) if t == Type::F64 => Opcode::ConvertF32F64,
            (64, t) if t == Type::F32 => Opcode::ConvertF64F32,
            (64, t) if t == Type::F64 => return v,
            (b, t) => panic!("no f{b} conversion from {t}"),
        };
        self.fp(op, control, &[v])
    }

    pub fn workgroup_id(&mut self) -> Value {
        self.inst(Opcode::WorkgroupId, &[])
    }

    pub fn local_invocation_id(&mut self) -> Value {
        self.inst(Opcode::LocalInvocationId, &[])
    }

    pub fn lane_id(&mut self) -> Value {
        self.inst(Opcode::LaneId, &[])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpCompare {
    Equal,
    NotEqual,
    LessThan,
    GreaterThan,
    LessThanEqual,
    GreaterThanEqual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ICompare {
    Equal,
    NotEqual,
    SLessThan,
    ULessThan,
    SLessThanEqual,
    ULessThanEqual,
    SGreaterThan,
    UGreaterThan,
    SGreaterThanEqual,
    UGreaterThanEqual,
}
