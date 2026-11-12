//! Maxwell instruction translators: one function per opcode, grouped like the C++ frontend.

mod atomic_operations_global_memory;
mod atomic_operations_shared_memory;
mod attribute_memory_to_physical;
mod barrier_operations;
mod bitfield_extract;
mod bitfield_insert;
mod branch_indirect;
mod common;
mod condition_code_set;
mod double_add;
mod double_compare_and_set;
mod double_fused_multiply_add;
mod double_min_max;
mod double_multiply;
mod double_set_predicate;
mod exit_program;
mod find_leading_one;
mod floating_point_add;
mod floating_point_compare;
mod floating_point_compare_and_set;
mod floating_point_conversion_floating_point;
mod floating_point_conversion_integer;
mod floating_point_fused_multiply_add;
mod floating_point_min_max;
mod floating_point_multi_function;
mod floating_point_multiply;
mod floating_point_range_reduction;
mod floating_point_set_predicate;
mod floating_point_swizzled_add;
mod half_floating_point_add;
mod half_floating_point_fused_multiply_add;
mod half_floating_point_multiply;
mod half_floating_point_set;
mod half_floating_point_set_predicate;
mod integer_add;
mod integer_add_three_input;
mod integer_byte_permute;
mod integer_compare;
mod integer_compare_and_set;
mod integer_floating_point_conversion;
mod integer_funnel_shift;
mod integer_minimum_maximum;
mod integer_popcount;
mod integer_scaled_add;
mod integer_set_predicate;
mod integer_shift_left;
mod integer_shift_right;
mod integer_short_multiply_add;
mod integer_to_integer_conversion;
mod internal_stage_buffer_entry_read;
mod load_constant;
mod load_effective_address;
mod load_store_attribute;
mod load_store_local_shared;
mod load_store_memory;
mod logic_operation;
mod logic_operation_three_input;
mod move_predicate_to_register;
mod move_register;
mod move_register_to_predicate;
mod move_special_register;
mod output_geometry;
mod pixel_load;
mod predicate_set_predicate;
mod predicate_set_register;
mod select_source_with_predicate;
mod surface_atomic_operations;
mod surface_load_store;
mod tensor_mul_add;
#[cfg(test)]
mod tests;
mod texture_fetch;
mod texture_fetch_swizzled;
mod texture_gather;
mod texture_gather_swizzled;
mod texture_gradient;
mod texture_load;
mod texture_load_swizzled;
mod texture_mipmap_level;
mod texture_query;
mod video_minimum_maximum;
mod video_multiply_add;
mod video_set_predicate;
mod vote;
mod warp_shuffle;

pub use common::*;

use lichii_common::GpuVAddr;

use super::instruction::Instruction;
use super::opcode::Opcode;
use crate::ShaderEnvironment;
use crate::TranslateError;
use crate::ir::{BlockId, Emitter, Program, Reg, Value};

pub type TResult = Result<(), TranslateError>;

/// State shared by every translator while one instruction is translated.
pub struct Translator<'a> {
    pub ir: Emitter<'a>,
    pub env: &'a dyn ShaderEnvironment,
    /// Address of the program's first instruction.
    pub base: GpuVAddr,
    /// Offset of the current instruction from `base`.
    pub pc: u32,
    pub insn: Instruction,
}

impl<'a> Translator<'a> {
    pub fn new(
        prog: &'a mut Program,
        block: BlockId,
        env: &'a dyn ShaderEnvironment,
        base: GpuVAddr,
    ) -> Translator<'a> {
        Translator {
            ir: Emitter::new(prog, block),
            env,
            base,
            pc: 0,
            insn: Instruction(0),
        }
    }

    /// The error for an encoding or feature this translator does not handle.
    pub fn unsupported(&self) -> TranslateError {
        TranslateError::Unsupported {
            address: self.base + u64::from(self.pc),
            opcode: u128::from(self.insn.0),
        }
    }

    pub fn x(&mut self, reg: Reg) -> Value {
        self.ir.get_reg(reg)
    }

    pub fn set_x(&mut self, reg: Reg, value: Value) {
        self.ir.set_reg(reg, value);
    }

    /// A 64-bit value from an aligned register pair.
    pub fn l(&mut self, reg: Reg) -> Result<Value, TranslateError> {
        if !reg.is_zero() && !reg.0.is_multiple_of(2) {
            return Err(self.unsupported());
        }
        let (lo, hi) = (self.x(reg), self.x(reg.offset(1)));
        let v = self.ir.composite_construct(&[lo, hi]);
        Ok(self.ir.pack_uint_2x32(v))
    }

    pub fn set_l(&mut self, reg: Reg, value: Value) -> TResult {
        if !reg.is_zero() && !reg.0.is_multiple_of(2) {
            return Err(self.unsupported());
        }
        let v = self.ir.unpack_uint_2x32(value);
        for i in 0..2 {
            let e = self.ir.composite_extract(v, i);
            self.set_x(reg.offset(i as u16), e);
        }
        Ok(())
    }

    pub fn f(&mut self, reg: Reg) -> Value {
        let v = self.x(reg);
        self.ir.bitcast_f32(v)
    }

    pub fn set_f(&mut self, reg: Reg, value: Value) {
        let v = self.ir.bitcast_u32(value);
        self.set_x(reg, v);
    }

    pub fn d(&mut self, reg: Reg) -> Result<Value, TranslateError> {
        if !reg.is_zero() && !reg.0.is_multiple_of(2) {
            return Err(self.unsupported());
        }
        let (lo, hi) = (self.x(reg), self.x(reg.offset(1)));
        let v = self.ir.composite_construct(&[lo, hi]);
        Ok(self.ir.pack_double_2x32(v))
    }

    pub fn set_d(&mut self, reg: Reg, value: Value) -> TResult {
        if !reg.is_zero() && !reg.0.is_multiple_of(2) {
            return Err(self.unsupported());
        }
        let v = self.ir.unpack_double_2x32(value);
        for i in 0..2 {
            let e = self.ir.composite_extract(v, i);
            self.set_x(reg.offset(i as u16), e);
        }
        Ok(())
    }

    fn reg_at(&self, bit: u32) -> Reg {
        Reg(self.insn.bits(bit, 8) as u16)
    }

    pub fn get_reg8(&mut self) -> Value {
        self.x(self.reg_at(8))
    }

    pub fn get_reg20(&mut self) -> Value {
        self.x(self.reg_at(20))
    }

    pub fn get_reg39(&mut self) -> Value {
        self.x(self.reg_at(39))
    }

    pub fn get_float_reg8(&mut self) -> Value {
        self.f(self.reg_at(8))
    }

    pub fn get_float_reg20(&mut self) -> Value {
        self.f(self.reg_at(20))
    }

    pub fn get_float_reg39(&mut self) -> Value {
        self.f(self.reg_at(39))
    }

    pub fn get_double_reg20(&mut self) -> Result<Value, TranslateError> {
        self.d(self.reg_at(20))
    }

    pub fn get_double_reg39(&mut self) -> Result<Value, TranslateError> {
        self.d(self.reg_at(39))
    }

    /// Binding and byte offset of the instruction's `c[binding][offset]` operand.
    pub fn cbuf_addr(&self) -> Result<(u32, u32), TranslateError> {
        let binding = self.insn.bits(34, 5) as u32;
        if binding >= 18 {
            return Err(self.unsupported());
        }
        Ok((binding, self.insn.bits(20, 14) as u32 * 4))
    }

    pub fn get_cbuf(&mut self) -> Result<Value, TranslateError> {
        let (binding, offset) = self.cbuf_addr()?;
        Ok(self.ir.get_cbuf(Value::U32(binding), Value::U32(offset)))
    }

    pub fn get_float_cbuf(&mut self) -> Result<Value, TranslateError> {
        let (binding, offset) = self.cbuf_addr()?;
        Ok(self
            .ir
            .get_float_cbuf(Value::U32(binding), Value::U32(offset)))
    }

    pub fn get_double_cbuf(&mut self) -> Result<Value, TranslateError> {
        let (binding, offset) = self.cbuf_addr()?;
        let unaligned = self.insn.bit(20);
        let addr = if unaligned {
            offset | 4
        } else {
            (offset & !7) | 4
        };
        let high = self.ir.get_cbuf(Value::U32(binding), Value::U32(addr));
        let low = if unaligned {
            Value::U32(0)
        } else {
            self.ir.get_cbuf(Value::U32(binding), Value::U32(offset))
        };
        let v = self.ir.composite_construct(&[low, high]);
        Ok(self.ir.pack_double_2x32(v))
    }

    pub fn get_packed_cbuf(&mut self) -> Result<Value, TranslateError> {
        if self.insn.bit(20) {
            return Err(self.unsupported());
        }
        let (binding, offset) = self.cbuf_addr()?;
        let low = self.ir.get_cbuf(Value::U32(binding), Value::U32(offset));
        let high = self
            .ir
            .get_cbuf(Value::U32(binding), Value::U32(offset + 4));
        let v = self.ir.composite_construct(&[low, high]);
        Ok(self.ir.pack_uint_2x32(v))
    }

    /// 19-bit immediate with its sign at bit 56, sign extended to 32 bits.
    pub fn imm20(&self) -> u32 {
        let value = self.insn.bits(20, 19) as u32;
        if self.insn.bit(56) {
            value.wrapping_sub(1 << 19)
        } else {
            value
        }
    }

    pub fn get_imm20(&self) -> Value {
        Value::U32(self.imm20())
    }

    pub fn get_float_imm20(&self) -> Value {
        let sign = if self.insn.bit(56) { 1 << 31 } else { 0 };
        Value::F32(f32::from_bits(
            ((self.insn.bits(20, 19) as u32) << 12) | sign,
        ))
    }

    pub fn get_double_imm20(&self) -> Value {
        let sign = if self.insn.bit(56) { 1 << 63 } else { 0 };
        Value::F64(f64::from_bits((self.insn.bits(20, 19) << 44) | sign))
    }

    pub fn get_packed_imm20(&self) -> Value {
        Value::U64((i64::from(self.imm20() as i32) << 32) as u64)
    }

    pub fn get_imm32(&self) -> Value {
        Value::U32(self.insn.bits(20, 32) as u32)
    }

    pub fn get_float_imm32(&self) -> Value {
        Value::F32(f32::from_bits(self.insn.bits(20, 32) as u32))
    }

    pub fn reset_zero(&mut self) {
        self.ir.set_z_flag(Value::U1(false));
    }

    pub fn reset_s_flag(&mut self) {
        self.ir.set_s_flag(Value::U1(false));
    }

    pub fn reset_c_flag(&mut self) {
        self.ir.set_c_flag(Value::U1(false));
    }

    pub fn reset_o_flag(&mut self) {
        self.ir.set_o_flag(Value::U1(false));
    }
}

/// Translates the instructions in `[begin, end)` (program offsets) into `block`.
pub fn translate_block(
    prog: &mut Program,
    block: BlockId,
    env: &dyn ShaderEnvironment,
    begin: u32,
    end: u32,
) -> TResult {
    let base = env.start_address();
    let mut t = Translator::new(prog, block, env, base);
    let mut pc = begin;
    while pc < end {
        if super::instruction::is_scheduling_word(base, pc) {
            pc += 8;
            continue;
        }
        t.pc = pc;
        t.insn = Instruction(env.read_u64(base + u64::from(pc)));
        translate_instruction(&mut t)?;
        pc += 8;
    }
    Ok(())
}

/// Translates `t.insn`.
pub fn translate_instruction(t: &mut Translator<'_>) -> TResult {
    let Some(opcode) = t.insn.opcode() else {
        return Err(TranslateError::Invalid {
            address: t.base + u64::from(t.pc),
            raw: u128::from(t.insn.0),
        });
    };
    dispatch(t, opcode)
}

fn dispatch(t: &mut Translator<'_>, opcode: Opcode) -> TResult {
    match opcode {
        Opcode::TensorMulAdd => tensor_mul_add::tensor_mul_add(t),
        Opcode::Cctlt => Err(t.unsupported()),
        Opcode::Al2p => attribute_memory_to_physical::al2p(t),
        Opcode::Ald => load_store_attribute::ald(t),
        Opcode::Ast => load_store_attribute::ast(t),
        Opcode::B2r => Err(t.unsupported()),
        Opcode::Bar => barrier_operations::bar(t),
        Opcode::BfeCbuf => bitfield_extract::bfe_cbuf(t),
        Opcode::BfeReg => bitfield_extract::bfe_reg(t),
        Opcode::BfiCr => bitfield_insert::bfi_cr(t),
        Opcode::BfiRc => bitfield_insert::bfi_rc(t),
        Opcode::BfiReg => bitfield_insert::bfi_reg(t),
        Opcode::Cs2r => Err(t.unsupported()),
        Opcode::Cset => condition_code_set::cset(t),
        Opcode::Csetp => condition_code_set::csetp(t),
        Opcode::DaddCbuf => double_add::dadd_cbuf(t),
        Opcode::DaddReg => double_add::dadd_reg(t),
        Opcode::Depbar => barrier_operations::depbar(t),
        Opcode::DmnmxCbuf => double_min_max::dmnmx_cbuf(t),
        Opcode::DmnmxReg => double_min_max::dmnmx_reg(t),
        Opcode::DmulCbuf => double_multiply::dmul_cbuf(t),
        Opcode::DmulReg => double_multiply::dmul_reg(t),
        Opcode::F2fCbuf => floating_point_conversion_floating_point::f2f_cbuf(t),
        Opcode::F2fReg => floating_point_conversion_floating_point::f2f_reg(t),
        Opcode::F2iCbuf => floating_point_conversion_integer::f2i_cbuf(t),
        Opcode::F2iReg => floating_point_conversion_integer::f2i_reg(t),
        Opcode::FaddCbuf => floating_point_add::fadd_cbuf(t),
        Opcode::FaddReg => floating_point_add::fadd_reg(t),
        Opcode::FchkCbuf => Err(t.unsupported()),
        Opcode::FchkReg => Err(t.unsupported()),
        Opcode::FloCbuf => find_leading_one::flo_cbuf(t),
        Opcode::FloReg => find_leading_one::flo_reg(t),
        Opcode::FmnmxCbuf => floating_point_min_max::fmnmx_cbuf(t),
        Opcode::FmnmxReg => floating_point_min_max::fmnmx_reg(t),
        Opcode::FmulCbuf => floating_point_multiply::fmul_cbuf(t),
        Opcode::FmulReg => floating_point_multiply::fmul_reg(t),
        Opcode::Fswzadd => floating_point_swizzled_add::fswzadd(t),
        Opcode::Hadd2Reg => half_floating_point_add::hadd2_reg(t),
        Opcode::Hfma2Reg => half_floating_point_fused_multiply_add::hfma2_reg(t),
        Opcode::Hmul2Reg => half_floating_point_multiply::hmul2_reg(t),
        Opcode::Hset2Reg => half_floating_point_set::hset2_reg(t),
        Opcode::Hsetp2Reg => half_floating_point_set_predicate::hsetp2_reg(t),
        Opcode::I2fCbuf => integer_floating_point_conversion::i2f_cbuf(t),
        Opcode::I2fReg => integer_floating_point_conversion::i2f_reg(t),
        Opcode::I2iCbuf => integer_to_integer_conversion::i2i_cbuf(t),
        Opcode::I2iReg => integer_to_integer_conversion::i2i_reg(t),
        Opcode::IaddCbuf => integer_add::iadd_cbuf(t),
        Opcode::IaddReg => integer_add::iadd_reg(t),
        Opcode::IdpImm => Err(t.unsupported()),
        Opcode::IdpReg => Err(t.unsupported()),
        Opcode::ImnmxCbuf => integer_minimum_maximum::imnmx_cbuf(t),
        Opcode::ImnmxReg => integer_minimum_maximum::imnmx_reg(t),
        Opcode::ImulCbuf => Err(t.unsupported()),
        Opcode::ImulReg => Err(t.unsupported()),
        Opcode::Isberd => internal_stage_buffer_entry_read::isberd(t),
        Opcode::IsbeNx2 => internal_stage_buffer_entry_read::isbe_nx2(t),
        Opcode::IscaddCbuf => integer_scaled_add::iscadd_cbuf(t),
        Opcode::IscaddReg => integer_scaled_add::iscadd_reg(t),
        Opcode::Ldc => load_constant::ldc(t),
        Opcode::Ldg => load_store_memory::ldg(t),
        Opcode::Ldl => load_store_local_shared::ldl(t),
        Opcode::Lds => load_store_local_shared::lds(t),
        Opcode::LeaHiReg => load_effective_address::lea_hi_reg(t),
        Opcode::LeaLoReg => load_effective_address::lea_lo_reg(t),
        Opcode::Lepc => Err(t.unsupported()),
        Opcode::Lop3Reg => logic_operation_three_input::lop3_reg(t),
        Opcode::LopCbuf => logic_operation::lop_cbuf(t),
        Opcode::LopReg => logic_operation::lop_reg(t),
        Opcode::Membar => barrier_operations::membar(t),
        Opcode::MovCbuf => move_register::mov_cbuf(t),
        Opcode::MovReg => move_register::mov_reg(t),
        Opcode::Mufu => floating_point_multi_function::mufu(t),
        Opcode::Nop => Err(t.unsupported()),
        Opcode::OutCbuf => output_geometry::out_cbuf(t),
        Opcode::OutReg => output_geometry::out_reg(t),
        Opcode::P2rCbuf => move_predicate_to_register::p2r_cbuf(t),
        Opcode::P2rImm => move_predicate_to_register::p2r_imm(t),
        Opcode::P2rReg => move_predicate_to_register::p2r_reg(t),
        Opcode::Pixld => pixel_load::pixld(t),
        Opcode::PopcCbuf => integer_popcount::popc_cbuf(t),
        Opcode::PopcReg => integer_popcount::popc_reg(t),
        Opcode::Pset => predicate_set_register::pset(t),
        Opcode::Psetp => predicate_set_predicate::psetp(t),
        Opcode::R2b => Err(t.unsupported()),
        Opcode::R2pCbuf => move_register_to_predicate::r2p_cbuf(t),
        Opcode::R2pReg => move_register_to_predicate::r2p_reg(t),
        Opcode::Red => atomic_operations_global_memory::red(t),
        Opcode::RroCbuf => floating_point_range_reduction::rro_cbuf(t),
        Opcode::RroReg => floating_point_range_reduction::rro_reg(t),
        Opcode::S2r => move_special_register::s2r(t),
        Opcode::SelCbuf => select_source_with_predicate::sel_cbuf(t),
        Opcode::SelReg => select_source_with_predicate::sel_reg(t),
        Opcode::Shfl => warp_shuffle::shfl(t),
        Opcode::ShfLReg => integer_funnel_shift::shf_lreg(t),
        Opcode::ShfRReg => integer_funnel_shift::shf_rreg(t),
        Opcode::ShlCbuf => integer_shift_left::shl_cbuf(t),
        Opcode::ShlReg => integer_shift_left::shl_reg(t),
        Opcode::ShrCbuf => integer_shift_right::shr_cbuf(t),
        Opcode::ShrReg => integer_shift_right::shr_reg(t),
        Opcode::Stg => load_store_memory::stg(t),
        Opcode::Stl => load_store_local_shared::stl(t),
        Opcode::Stp => Err(t.unsupported()),
        Opcode::Sts => load_store_local_shared::sts(t),
        Opcode::Sync => Err(t.unsupported()),
        Opcode::Tmml => texture_mipmap_level::tmml(t),
        Opcode::TmmlB => texture_mipmap_level::tmml_b(t),
        Opcode::Txa => Err(t.unsupported()),
        Opcode::Txq => texture_query::txq(t),
        Opcode::TxqB => texture_query::txq_b(t),
        Opcode::Vote => vote::vote(t),
        Opcode::VoteVtg => vote::vote_vtg(t),
        Opcode::Vsetp => video_set_predicate::vsetp(t),
        Opcode::AtomCas => atomic_operations_global_memory::atom_cas(t),
        Opcode::BfeImm => bitfield_extract::bfe_imm(t),
        Opcode::BfiImm => bitfield_insert::bfi_imm(t),
        Opcode::Bpt => Err(t.unsupported()),
        Opcode::Bra => Err(t.unsupported()),
        Opcode::Brk => Err(t.unsupported()),
        Opcode::Brx => branch_indirect::brx(t),
        Opcode::Cal => Err(t.unsupported()),
        Opcode::Cont => Err(t.unsupported()),
        Opcode::DaddImm => double_add::dadd_imm(t),
        Opcode::DfmaCr => double_fused_multiply_add::dfma_cr(t),
        Opcode::DfmaRc => double_fused_multiply_add::dfma_rc(t),
        Opcode::DfmaReg => double_fused_multiply_add::dfma_reg(t),
        Opcode::DmnmxImm => double_min_max::dmnmx_imm(t),
        Opcode::DmulImm => double_multiply::dmul_imm(t),
        Opcode::DsetpCbuf => double_set_predicate::dsetp_cbuf(t),
        Opcode::DsetpReg => double_set_predicate::dsetp_reg(t),
        Opcode::Exit => exit_program::exit(t),
        Opcode::F2fImm => floating_point_conversion_floating_point::f2f_imm(t),
        Opcode::F2iImm => floating_point_conversion_integer::f2i_imm(t),
        Opcode::FaddImm => floating_point_add::fadd_imm(t),
        Opcode::FchkImm => Err(t.unsupported()),
        Opcode::FcmpCr => floating_point_compare::fcmp_cr(t),
        Opcode::FcmpRc => floating_point_compare::fcmp_rc(t),
        Opcode::FcmpReg => floating_point_compare::fcmp_reg(t),
        Opcode::FloImm => find_leading_one::flo_imm(t),
        Opcode::FmnmxImm => floating_point_min_max::fmnmx_imm(t),
        Opcode::FmulImm => floating_point_multiply::fmul_imm(t),
        Opcode::FsetpCbuf => floating_point_set_predicate::fsetp_cbuf(t),
        Opcode::FsetpReg => floating_point_set_predicate::fsetp_reg(t),
        Opcode::Getcrsptr => Err(t.unsupported()),
        Opcode::Getlmembase => Err(t.unsupported()),
        Opcode::I2fImm => integer_floating_point_conversion::i2f_imm(t),
        Opcode::I2iImm => integer_to_integer_conversion::i2i_imm(t),
        Opcode::Iadd3Cbuf => integer_add_three_input::iadd3_cbuf(t),
        Opcode::Iadd3Reg => integer_add_three_input::iadd3_reg(t),
        Opcode::IaddImm => integer_add::iadd_imm(t),
        Opcode::IcmpCr => integer_compare::icmp_cr(t),
        Opcode::IcmpRc => integer_compare::icmp_rc(t),
        Opcode::IcmpReg => integer_compare::icmp_reg(t),
        Opcode::Ide => Err(t.unsupported()),
        Opcode::ImnmxImm => integer_minimum_maximum::imnmx_imm(t),
        Opcode::ImulImm => Err(t.unsupported()),
        Opcode::IscaddImm => integer_scaled_add::iscadd_imm(t),
        Opcode::IsetpCbuf => integer_set_predicate::isetp_cbuf(t),
        Opcode::IsetpReg => integer_set_predicate::isetp_reg(t),
        Opcode::IsetCbuf => integer_compare_and_set::iset_cbuf(t),
        Opcode::IsetReg => integer_compare_and_set::iset_reg(t),
        Opcode::Jcal => Err(t.unsupported()),
        Opcode::Jmp => Err(t.unsupported()),
        Opcode::Jmx => branch_indirect::jmx(t),
        Opcode::Kil => Err(t.unsupported()),
        Opcode::LeaLoCbuf => load_effective_address::lea_lo_cbuf(t),
        Opcode::LeaLoImm => load_effective_address::lea_lo_imm(t),
        Opcode::Longjmp => Err(t.unsupported()),
        Opcode::LopImm => logic_operation::lop_imm(t),
        Opcode::Mov32i => move_register::mov32i(t),
        Opcode::MovImm => move_register::mov_imm(t),
        Opcode::OutImm => output_geometry::out_imm(t),
        Opcode::Pbk => Err(t.unsupported()),
        Opcode::Pcnt => Err(t.unsupported()),
        Opcode::Pexit => Err(t.unsupported()),
        Opcode::Plongjmp => Err(t.unsupported()),
        Opcode::PopcImm => integer_popcount::popc_imm(t),
        Opcode::Pret => Err(t.unsupported()),
        Opcode::PrmtCr => integer_byte_permute::prmt_cr(t),
        Opcode::PrmtRc => integer_byte_permute::prmt_rc(t),
        Opcode::PrmtReg => integer_byte_permute::prmt_reg(t),
        Opcode::R2pImm => move_register_to_predicate::r2p_imm(t),
        Opcode::Ram => Err(t.unsupported()),
        Opcode::Ret => Err(t.unsupported()),
        Opcode::RroImm => floating_point_range_reduction::rro_imm(t),
        Opcode::Rtt => Err(t.unsupported()),
        Opcode::Sam => Err(t.unsupported()),
        Opcode::SelImm => select_source_with_predicate::sel_imm(t),
        Opcode::Setcrsptr => Err(t.unsupported()),
        Opcode::Setlmembase => Err(t.unsupported()),
        Opcode::ShfLImm => integer_funnel_shift::shf_limm(t),
        Opcode::ShfRImm => integer_funnel_shift::shf_rimm(t),
        Opcode::ShlImm => integer_shift_left::shl_imm(t),
        Opcode::ShrImm => integer_shift_right::shr_imm(t),
        Opcode::Ssy => Err(t.unsupported()),
        Opcode::Cctl => Err(t.unsupported()),
        Opcode::Cctll => Err(t.unsupported()),
        Opcode::DfmaImm => double_fused_multiply_add::dfma_imm(t),
        Opcode::DsetpImm => double_set_predicate::dsetp_imm(t),
        Opcode::FcmpImm => floating_point_compare::fcmp_imm(t),
        Opcode::FsetpImm => floating_point_set_predicate::fsetp_imm(t),
        Opcode::Iadd3Imm => integer_add_three_input::iadd3_imm(t),
        Opcode::IcmpImm => integer_compare::icmp_imm(t),
        Opcode::IsetpImm => integer_set_predicate::isetp_imm(t),
        Opcode::IsetImm => integer_compare_and_set::iset_imm(t),
        Opcode::PrmtImm => integer_byte_permute::prmt_imm(t),
        Opcode::Suld => surface_load_store::suld(t),
        Opcode::Sured => surface_atomic_operations::sured(t),
        Opcode::Sust => surface_load_store::sust(t),
        Opcode::TexB => texture_fetch::tex_b(t),
        Opcode::Tld4B => texture_gather::tld4_b(t),
        Opcode::Txd => texture_gradient::txd(t),
        Opcode::TxdB => texture_gradient::txd_b(t),
        Opcode::XmadReg => integer_short_multiply_add::xmad_reg(t),
        Opcode::DsetCbuf => double_compare_and_set::dset_cbuf(t),
        Opcode::DsetReg => double_compare_and_set::dset_reg(t),
        Opcode::FfmaCr => floating_point_fused_multiply_add::ffma_cr(t),
        Opcode::FfmaRc => floating_point_fused_multiply_add::ffma_rc(t),
        Opcode::FfmaReg => floating_point_fused_multiply_add::ffma_reg(t),
        Opcode::ImadspCr => Err(t.unsupported()),
        Opcode::ImadspRc => Err(t.unsupported()),
        Opcode::ImadspReg => Err(t.unsupported()),
        Opcode::ImadCr => Err(t.unsupported()),
        Opcode::ImadRc => Err(t.unsupported()),
        Opcode::ImadReg => Err(t.unsupported()),
        Opcode::Suatom => surface_atomic_operations::suatom(t),
        Opcode::SuatomCas => Err(t.unsupported()),
        Opcode::Tld4s => texture_gather_swizzled::tld4s(t),
        Opcode::Vabsdiff4 => Err(t.unsupported()),
        Opcode::XmadImm => integer_short_multiply_add::xmad_imm(t),
        Opcode::XmadRc => integer_short_multiply_add::xmad_rc(t),
        Opcode::Atom => atomic_operations_global_memory::atom(t),
        Opcode::Atoms => atomic_operations_shared_memory::atoms(t),
        Opcode::AtomsCas => Err(t.unsupported()),
        Opcode::DsetImm => double_compare_and_set::dset_imm(t),
        Opcode::FfmaImm => floating_point_fused_multiply_add::ffma_imm(t),
        Opcode::Fmul32i => floating_point_multiply::fmul32i(t),
        Opcode::FsetCbuf => floating_point_compare_and_set::fset_cbuf(t),
        Opcode::FsetReg => floating_point_compare_and_set::fset_reg(t),
        Opcode::Hadd2Cbuf => half_floating_point_add::hadd2_cbuf(t),
        Opcode::Hadd2Imm => half_floating_point_add::hadd2_imm(t),
        Opcode::Hmul2Cbuf => half_floating_point_multiply::hmul2_cbuf(t),
        Opcode::Hmul2Imm => half_floating_point_multiply::hmul2_imm(t),
        Opcode::Hset2Cbuf => half_floating_point_set::hset2_cbuf(t),
        Opcode::Hset2Imm => half_floating_point_set::hset2_imm(t),
        Opcode::Hsetp2Cbuf => half_floating_point_set_predicate::hsetp2_cbuf(t),
        Opcode::Hsetp2Imm => half_floating_point_set_predicate::hsetp2_imm(t),
        Opcode::ImadspImm => Err(t.unsupported()),
        Opcode::ImadImm => Err(t.unsupported()),
        Opcode::Imul32i => Err(t.unsupported()),
        Opcode::Ipa => load_store_attribute::ipa(t),
        Opcode::Tld => texture_load::tld(t),
        Opcode::TldB => texture_load::tld_b(t),
        Opcode::Vabsdiff => Err(t.unsupported()),
        Opcode::Vmad => video_multiply_add::vmad(t),
        Opcode::Vshl => Err(t.unsupported()),
        Opcode::Vshr => Err(t.unsupported()),
        Opcode::FsetImm => floating_point_compare_and_set::fset_imm(t),
        Opcode::Hadd2_32i => half_floating_point_add::hadd2_32i(t),
        Opcode::Hfma2_32i => half_floating_point_fused_multiply_add::hfma2_32i(t),
        Opcode::Hmul2_32i => half_floating_point_multiply::hmul2_32i(t),
        Opcode::Iadd32i => integer_add::iadd32i(t),
        Opcode::Lop3Cbuf => logic_operation_three_input::lop3_cbuf(t),
        Opcode::Vmnmx => video_minimum_maximum::vmnmx(t),
        Opcode::Vset => Err(t.unsupported()),
        Opcode::XmadCr => integer_short_multiply_add::xmad_cr(t),
        Opcode::Fadd32i => floating_point_add::fadd32i(t),
        Opcode::Ffma32i => floating_point_fused_multiply_add::ffma32i(t),
        Opcode::Hfma2Cr => half_floating_point_fused_multiply_add::hfma2_cr(t),
        Opcode::Hfma2Imm => half_floating_point_fused_multiply_add::hfma2_imm(t),
        Opcode::Hfma2Rc => half_floating_point_fused_multiply_add::hfma2_rc(t),
        Opcode::Imad32i => Err(t.unsupported()),
        Opcode::Iscadd32i => integer_scaled_add::iscadd32i(t),
        Opcode::LeaHiCbuf => load_effective_address::lea_hi_cbuf(t),
        Opcode::Lop32i => logic_operation::lop32i(t),
        Opcode::Lop3Imm => logic_operation_three_input::lop3_imm(t),
        Opcode::Texs => texture_fetch_swizzled::texs(t),
        Opcode::Tld4 => texture_gather::tld4(t),
        Opcode::Tlds => texture_load_swizzled::tlds(t),
        Opcode::Vadd => Err(t.unsupported()),
        Opcode::Tex => texture_fetch::tex(t),
        Opcode::Ld => Err(t.unsupported()),
        Opcode::St => Err(t.unsupported()),
    }
}
