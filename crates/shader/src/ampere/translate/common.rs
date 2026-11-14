//! Register and operand access shared by the Ampere translators.

use super::{TResult, Translator};
use crate::ShaderInfo;
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
        self.note_cbuf_bank("ALU", Value::U32(index));
        Ok(self
            .ir
            .get_cbuf(Value::U32(index), Value::U32(operand.offset)))
    }

    /// Counts a constant buffer read from a bank the host never binds, which reads zero.
    pub fn note_cbuf_bank(&self, opcode: &'static str, bank: Value) {
        if let Some(bank) = bank.imm_u32()
            && bank as usize >= ShaderInfo::MAX_CBUFS
        {
            crate::fallback::note_value(opcode, "bank_past_limit", bank);
        }
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

/// Where an ALU source operand lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    Reg(Reg),
    Imm(u32),
    /// The instruction's constant-buffer operand.
    Cbuf,
}

/// An ALU source with the float modifiers and the packed-half swizzle its slot encodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AluSrc {
    pub operand: Operand,
    pub abs: bool,
    pub neg: bool,
    /// The 2-bit swizzle of a packed-half or F16 source (zero for immediates).
    pub swizzle: u64,
}

impl Translator<'_> {
    /// The first ALU source: absolute value at bit 73, negation at 72, swizzle at 74..76.
    pub fn alu_a(&self) -> AluSrc {
        AluSrc {
            operand: Operand::Reg(self.reg_a()),
            abs: self.insn.bit(73),
            neg: self.insn.bit(72),
            swizzle: self.insn.bits(74, 2),
        }
    }

    /// The second ALU source with its modifiers, wherever the form puts it. `half` selects the
    /// packed-half layout, which moves the modifiers of a register at 64..72 to bits 83 and 84
    /// and gives it a swizzle at 81..83.
    pub fn alu_b(&self, half: bool) -> Result<AluSrc, TranslateError> {
        Ok(match self.form {
            Form::RegReg | Form::Fixed => {
                let reg = if self.is_uniform_datapath() {
                    self.insn.usrc1()
                } else {
                    self.insn.src1()
                };
                self.wide_src(Operand::Reg(reg))
            }
            Form::RegImm | Form::RegCbuf | Form::RegUreg => self.high_src(half),
            Form::ImmReg => self.wide_src(Operand::Imm(self.insn.imm32())),
            Form::CbufReg => self.wide_src(Operand::Cbuf),
            Form::UregReg => self.wide_src(Operand::Reg(self.insn.usrc1())),
            _ => return Err(self.unsupported()),
        })
    }

    /// The third ALU source with its modifiers; `half` as for [`Translator::alu_b`].
    pub fn alu_c(&self, half: bool) -> Result<AluSrc, TranslateError> {
        Ok(match self.form {
            Form::RegReg | Form::ImmReg | Form::CbufReg | Form::UregReg | Form::Fixed => {
                self.high_src(half)
            }
            Form::RegImm => self.wide_src(Operand::Imm(self.insn.imm32())),
            Form::RegCbuf => self.wide_src(Operand::Cbuf),
            Form::RegUreg => self.wide_src(Operand::Reg(self.insn.usrc1())),
            _ => return Err(self.unsupported()),
        })
    }

    /// An operand at bit 32: modifiers at 62 and 63 and swizzle at 60..62, none for immediates.
    fn wide_src(&self, operand: Operand) -> AluSrc {
        if let Operand::Imm(_) = operand {
            return AluSrc {
                operand,
                abs: false,
                neg: false,
                swizzle: 0,
            };
        }
        AluSrc {
            operand,
            abs: self.insn.bit(62),
            neg: self.insn.bit(63),
            swizzle: self.insn.bits(60, 2),
        }
    }

    /// The register at 64..72.
    fn high_src(&self, half: bool) -> AluSrc {
        let reg = if self.is_uniform_datapath() {
            self.insn.usrc2()
        } else {
            self.insn.src2()
        };
        let (abs, neg, swizzle) = if half {
            (self.insn.bit(83), self.insn.bit(84), self.insn.bits(81, 2))
        } else {
            (self.insn.bit(74), self.insn.bit(75), 0)
        };
        AluSrc {
            operand: Operand::Reg(reg),
            abs,
            neg,
            swizzle,
        }
    }

    /// The bound buffer and byte offset of the constant-buffer operand.
    fn cbuf_slot(&self) -> Result<(u32, u32), TranslateError> {
        let operand = self.insn.cbuf();
        let CbufBank::Index(index) = operand.bank else {
            return Err(self.unsupported());
        };
        Ok((index, operand.offset))
    }

    /// The two words of a 64-bit constant-buffer operand, as a vector.
    fn cbuf_pair(&mut self) -> Result<Value, TranslateError> {
        let (index, offset) = self.cbuf_slot()?;
        self.note_cbuf_bank("DALU", Value::U32(index));
        let lo = self.ir.get_cbuf(Value::U32(index), Value::U32(offset));
        let hi = self
            .ir
            .get_cbuf(Value::U32(index), Value::U32(offset.wrapping_add(4)));
        Ok(self.ir.composite_construct(&[lo, hi]))
    }

    /// The 32 bits of a source, without its modifiers.
    pub fn read_u32(&mut self, src: AluSrc) -> Result<Value, TranslateError> {
        Ok(match src.operand {
            Operand::Reg(reg) => self.x(reg),
            Operand::Imm(imm) => Value::U32(imm),
            Operand::Cbuf => self.cbuf_u32()?,
        })
    }

    /// A source as an F32 with its absolute value and negation applied.
    pub fn read_f32(&mut self, src: AluSrc) -> Result<Value, TranslateError> {
        let value = match src.operand {
            Operand::Reg(reg) => self.f(reg),
            Operand::Imm(imm) => Value::F32(f32::from_bits(imm)),
            Operand::Cbuf => {
                let (index, offset) = self.cbuf_slot()?;
                self.note_cbuf_bank("FALU", Value::U32(index));
                self.ir
                    .get_float_cbuf(Value::U32(index), Value::U32(offset))
            }
        };
        Ok(self.ir.fp_abs_neg(value, src.abs, src.neg))
    }

    /// A source as an F64 with its modifiers: an aligned register pair, two constant-buffer
    /// words, or an immediate holding the high word.
    pub fn read_f64(&mut self, src: AluSrc) -> Result<Value, TranslateError> {
        let value = match src.operand {
            Operand::Reg(reg) => self.d(reg)?,
            Operand::Imm(imm) => Value::F64(f64::from_bits(u64::from(imm) << 32)),
            Operand::Cbuf => {
                let pair = self.cbuf_pair()?;
                self.ir.pack_double_2x32(pair)
            }
        };
        Ok(self.ir.fp_abs_neg(value, src.abs, src.neg))
    }

    /// A source as a U64: an aligned register pair or two constant-buffer words. Immediates have
    /// no 64-bit form.
    pub fn read_u64(&mut self, src: AluSrc) -> Result<Value, TranslateError> {
        match src.operand {
            Operand::Reg(reg) => self.l(reg),
            Operand::Imm(_) => Err(self.unsupported()),
            Operand::Cbuf => {
                let pair = self.cbuf_pair()?;
                Ok(self.ir.pack_uint_2x32(pair))
            }
        }
    }
}
