//! Register and operand access shared by the Ampere translators.

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ampere::decode::Form;
use crate::ampere::instruction::CbufBank;
use crate::ir::{Pred, Reg, Value};

impl Translator<'_> {
    /// The error for an encoding or feature this translator does not handle.
    pub fn unsupported(&self) -> TranslateError {
        TranslateError::Unsupported {
            address: self.base + u64::from(self.pc),
            opcode: self.insn.0,
        }
    }

    pub fn x(&mut self, reg: Reg) -> Value {
        self.ir.get_reg(reg)
    }

    pub fn set_x(&mut self, reg: Reg, value: Value) {
        self.ir.set_reg(reg, value);
    }

    pub fn f(&mut self, reg: Reg) -> Value {
        let v = self.x(reg);
        self.ir.bitcast_f32(v)
    }

    pub fn set_f(&mut self, reg: Reg, value: Value) {
        let v = self.ir.bitcast_u32(value);
        self.set_x(reg, v);
    }

    fn check_pair(&self, reg: Reg) -> TResult {
        if reg.is_zero() || reg.0.is_multiple_of(2) {
            Ok(())
        } else {
            Err(self.unsupported())
        }
    }

    /// A 64-bit integer from an aligned register pair.
    pub fn l(&mut self, reg: Reg) -> Result<Value, TranslateError> {
        self.check_pair(reg)?;
        let (lo, hi) = (self.x(reg), self.x(reg.offset(1)));
        let v = self.ir.composite_construct(&[lo, hi]);
        Ok(self.ir.pack_uint_2x32(v))
    }

    pub fn set_l(&mut self, reg: Reg, value: Value) -> TResult {
        self.check_pair(reg)?;
        let v = self.ir.unpack_uint_2x32(value);
        for i in 0..2 {
            let e = self.ir.composite_extract(v, i);
            self.set_x(reg.offset(i as u16), e);
        }
        Ok(())
    }

    /// A double from an aligned register pair.
    pub fn d(&mut self, reg: Reg) -> Result<Value, TranslateError> {
        self.check_pair(reg)?;
        let (lo, hi) = (self.x(reg), self.x(reg.offset(1)));
        let v = self.ir.composite_construct(&[lo, hi]);
        Ok(self.ir.pack_double_2x32(v))
    }

    pub fn set_d(&mut self, reg: Reg, value: Value) -> TResult {
        self.check_pair(reg)?;
        let v = self.ir.unpack_double_2x32(value);
        for i in 0..2 {
            let e = self.ir.composite_extract(v, i);
            self.set_x(reg.offset(i as u16), e);
        }
        Ok(())
    }

    /// Reads a predicate source with its negation; `PT` and `UPT` read as true.
    pub fn pred(&mut self, (pred, negated): (Pred, bool)) -> Value {
        self.ir.get_pred(pred, negated)
    }

    /// The constant buffer word the instruction's constant-buffer operand names. A bindless
    /// buffer (`cx[UR]`) is not supported.
    pub fn cbuf_u32(&mut self) -> Result<Value, TranslateError> {
        let operand = self.insn.cbuf();
        let CbufBank::Index(index) = operand.bank else {
            return Err(self.unsupported());
        };
        Ok(self
            .ir
            .get_cbuf(Value::U32(index), Value::U32(operand.offset)))
    }

    /// The first source, a register at 24..32 (a uniform register on the uniform datapath).
    pub fn src_a(&mut self) -> Value {
        let reg = self.reg_a();
        self.x(reg)
    }

    /// The register of the first source.
    pub fn reg_a(&self) -> Reg {
        if self.is_uniform_datapath() {
            self.insn.usrc0()
        } else {
            self.insn.src0()
        }
    }

    /// The destination register at 16..24 (a uniform register on the uniform datapath).
    pub fn reg_d(&self) -> Reg {
        if self.is_uniform_datapath() {
            self.insn.udst()
        } else {
            self.insn.dst()
        }
    }

    fn is_uniform_datapath(&self) -> bool {
        crate::ampere::decode::decode(self.insn).is_some_and(|d| d.opcode.is_uniform())
    }

    /// The second ALU source as a 32-bit integer, wherever the form puts it.
    pub fn src_b(&mut self) -> Result<Value, TranslateError> {
        let uniform = self.is_uniform_datapath();
        Ok(match self.form {
            Form::RegReg if uniform => {
                let reg = self.insn.usrc1();
                self.x(reg)
            }
            Form::RegReg | Form::Fixed => {
                let reg = self.insn.src1();
                self.x(reg)
            }
            Form::RegImm | Form::RegCbuf | Form::RegUreg => {
                let reg = if uniform {
                    self.insn.usrc2()
                } else {
                    self.insn.src2()
                };
                self.x(reg)
            }
            Form::ImmReg => Value::U32(self.insn.imm32()),
            Form::CbufReg => self.cbuf_u32()?,
            Form::UregReg => {
                let reg = self.insn.usrc1();
                self.x(reg)
            }
            _ => return Err(self.unsupported()),
        })
    }

    /// The third ALU source as a 32-bit integer, wherever the form puts it.
    pub fn src_c(&mut self) -> Result<Value, TranslateError> {
        let uniform = self.is_uniform_datapath();
        Ok(match self.form {
            Form::RegReg | Form::ImmReg | Form::CbufReg | Form::UregReg | Form::Fixed => {
                let reg = if uniform {
                    self.insn.usrc2()
                } else {
                    self.insn.src2()
                };
                self.x(reg)
            }
            Form::RegImm => Value::U32(self.insn.imm32()),
            Form::RegCbuf => self.cbuf_u32()?,
            Form::RegUreg => {
                let reg = self.insn.usrc1();
                self.x(reg)
            }
            _ => return Err(self.unsupported()),
        })
    }
}
