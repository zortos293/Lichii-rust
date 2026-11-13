//! Executes decoded instructions against a [`ThreadContext`].

use std::sync::atomic::{
    AtomicI8, AtomicI16, AtomicI32, AtomicI64, AtomicU8, AtomicU16, AtomicU32, AtomicU64, Ordering,
    fence,
};

use lichii_common::VAddr;

use super::simd;
use crate::decoder::*;
use crate::{CpuConfig, CpuFault, HaltReason, ThreadContext};

/// What the run loop does after an instruction.
pub(super) enum Flow {
    Next,
    Jump(u64),
    /// Drop translated code in `[addr, addr + size)`, then continue at the next instruction.
    Invalidate(VAddr, u64),
    InvalidateAll,
}

pub(super) type Step = Result<Flow, HaltReason>;

pub(super) struct Exec<'a> {
    pub ctx: &'a mut ThreadContext,
    pub cfg: &'a CpuConfig,
    /// Address of the instruction being executed.
    pub pc: u64,
}

const NZCV_SHIFT: u32 = 28;

#[inline]
const fn width_mask(sf: bool) -> u64 {
    if sf { u64::MAX } else { u32::MAX as u64 }
}

#[inline]
fn add_with_carry(sf: bool, a: u64, b: u64, carry: bool) -> (u64, u32) {
    if sf {
        let (r1, c1) = a.overflowing_add(b);
        let (r, c2) = r1.overflowing_add(carry as u64);
        let v = ((a ^ r) & (b ^ r)) >> 63;
        (r, nzcv(r >> 63 != 0, r == 0, c1 | c2, v != 0))
    } else {
        let (a, b) = (a as u32, b as u32);
        let (r1, c1) = a.overflowing_add(b);
        let (r, c2) = r1.overflowing_add(carry as u32);
        let v = ((a ^ r) & (b ^ r)) >> 31;
        (r as u64, nzcv(r >> 31 != 0, r == 0, c1 | c2, v != 0))
    }
}

#[inline]
const fn nzcv(n: bool, z: bool, c: bool, v: bool) -> u32 {
    (n as u32) << 3 | (z as u32) << 2 | (c as u32) << 1 | v as u32
}

#[inline]
fn logic_flags(sf: bool, r: u64) -> u32 {
    let n = if sf { r >> 63 } else { r >> 31 & 1 };
    nzcv(n != 0, r == 0, false, false)
}

/// `ConditionHolds` for a 4-bit condition and the NZCV nibble.
#[inline]
pub(crate) const fn condition_holds(cond: u8, flags: u32) -> bool {
    let (n, z, c, v) = (
        flags & 8 != 0,
        flags & 4 != 0,
        flags & 2 != 0,
        flags & 1 != 0,
    );
    let result = match cond >> 1 {
        0 => z,
        1 => c,
        2 => n,
        3 => v,
        4 => c && !z,
        5 => n == v,
        6 => n == v && !z,
        _ => true,
    };
    if cond & 1 != 0 && cond != 15 {
        !result
    } else {
        result
    }
}

#[inline]
fn shift(sf: bool, value: u64, ty: ShiftType, amount: u8) -> u64 {
    let amount = amount as u32;
    if sf {
        match ty {
            ShiftType::Lsl => value << amount,
            ShiftType::Lsr => value >> amount,
            ShiftType::Asr => ((value as i64) >> amount) as u64,
            ShiftType::Ror => value.rotate_right(amount),
        }
    } else {
        let value = value as u32;
        (match ty {
            ShiftType::Lsl => value << amount,
            ShiftType::Lsr => value >> amount,
            ShiftType::Asr => ((value as i32) >> amount) as u32,
            ShiftType::Ror => value.rotate_right(amount),
        }) as u64
    }
}

#[inline]
fn extend(value: u64, ext: Extend, amount: u8) -> u64 {
    let extended = match ext {
        Extend::Uxtb => value as u8 as u64,
        Extend::Uxth => value as u16 as u64,
        Extend::Uxtw => value as u32 as u64,
        Extend::Uxtx | Extend::Sxtx => value,
        Extend::Sxtb => value as i8 as u64,
        Extend::Sxth => value as i16 as u64,
        Extend::Sxtw => value as i32 as u64,
    };
    extended << amount
}

#[inline]
fn sign_extend(value: u64, size: u8) -> u64 {
    let shift = 64 - (8u32 << size);
    (((value << shift) as i64) >> shift) as u64
}

#[inline]
fn load_extend(value: u64, size: u8, ext: LoadExt) -> u64 {
    match ext {
        LoadExt::Zero => value,
        LoadExt::Sign32 => sign_extend(value, size) as u32 as u64,
        LoadExt::Sign64 => sign_extend(value, size),
    }
}

#[inline]
fn lane(v: u128, size: u8, index: usize) -> u64 {
    let bits = 8u32 << size;
    ((v >> (index as u32 * bits)) as u64) & width_bits(bits)
}

#[inline]
fn set_lane(v: &mut u128, size: u8, index: usize, value: u64) {
    let bits = 8u32 << size;
    let shift = index as u32 * bits;
    let mask = (width_bits(bits) as u128) << shift;
    *v = (*v & !mask) | (((value & width_bits(bits)) as u128) << shift);
}

#[inline]
const fn width_bits(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1 << bits) - 1
    }
}

const fn crc_table(poly: u32) -> [u32; 256] {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut bit = 0;
        while bit < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ poly
            } else {
                crc >> 1
            };
            bit += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

static CRC32: [u32; 256] = crc_table(0xedb8_8320);
static CRC32C: [u32; 256] = crc_table(0x82f6_3b78);

impl Exec<'_> {
    #[inline(always)]
    fn x(&self, r: Reg) -> u64 {
        match r.0 {
            31 => 0,
            32 => self.ctx.sp,
            n => self.ctx.x[n as usize],
        }
    }

    #[inline(always)]
    fn set(&mut self, r: Reg, value: u64) {
        match r.0 {
            31 => {}
            32 => self.ctx.sp = value,
            n => self.ctx.x[n as usize] = value,
        }
    }

    #[inline(always)]
    fn set_sf(&mut self, sf: bool, r: Reg, value: u64) {
        self.set(r, value & width_mask(sf));
    }

    #[inline(always)]
    fn flags(&self) -> u32 {
        self.ctx.pstate >> NZCV_SHIFT
    }

    #[inline(always)]
    fn set_flags(&mut self, nzcv: u32) {
        self.ctx.pstate = (self.ctx.pstate & !(0xf << NZCV_SHIFT)) | (nzcv << NZCV_SHIFT);
    }

    fn fault(&self, fault: CpuFault) -> HaltReason {
        HaltReason::Fault(fault)
    }

    fn data_fault(&self, addr: VAddr, write: bool) -> HaltReason {
        self.fault(CpuFault::Data {
            pc: self.pc,
            addr,
            write,
        })
    }

    fn read(&self, addr: VAddr, size: u8) -> Result<u64, HaltReason> {
        let mem = &*self.cfg.memory;
        match size {
            0 => mem.read::<u8>(addr).map(u64::from),
            1 => mem.read::<u16>(addr).map(u64::from),
            2 => mem.read::<u32>(addr).map(u64::from),
            _ => mem.read::<u64>(addr),
        }
        .map_err(|f| self.data_fault(f.addr, false))
    }

    fn write(&self, addr: VAddr, size: u8, value: u64) -> Result<(), HaltReason> {
        let mem = &*self.cfg.memory;
        match size {
            0 => mem.write(addr, value as u8),
            1 => mem.write(addr, value as u16),
            2 => mem.write(addr, value as u32),
            _ => mem.write(addr, value),
        }
        .map_err(|f| self.data_fault(f.addr, true))
    }

    fn read_vec(&self, addr: VAddr, size: u8) -> Result<u128, HaltReason> {
        if size < 4 {
            return self.read(addr, size).map(u128::from);
        }
        self.cfg
            .memory
            .read::<u128>(addr)
            .map_err(|f| self.data_fault(f.addr, false))
    }

    fn write_vec(&self, addr: VAddr, size: u8, value: u128) -> Result<(), HaltReason> {
        if size < 4 {
            return self.write(addr, size, value as u64);
        }
        self.cfg
            .memory
            .write(addr, value)
            .map_err(|f| self.data_fault(f.addr, true))
    }

    /// The access address and, for pre/post-indexed forms, the new base.
    #[inline]
    fn address(&self, rn: Reg, addr: Addr) -> (u64, Option<u64>) {
        let base = self.x(rn);
        match addr {
            Addr::Offset(o) => (base.wrapping_add(o as i64 as u64), None),
            Addr::Pre(o) => {
                let a = base.wrapping_add(o as i64 as u64);
                (a, Some(a))
            }
            Addr::Post(o) => (base, Some(base.wrapping_add(o as i64 as u64))),
            Addr::Reg {
                rm,
                extend: ext,
                shift,
            } => (base.wrapping_add(extend(self.x(rm), ext, shift)), None),
        }
    }

    #[inline]
    fn writeback(&mut self, rn: Reg, new_base: Option<u64>) {
        if let Some(b) = new_base {
            self.set(rn, b);
        }
    }

    /// Host pointer for an atomic access of `bytes` bytes, which must be naturally aligned.
    fn atomic_ptr(&self, addr: VAddr, bytes: u64) -> Result<*mut u8, HaltReason> {
        if addr & (bytes - 1) != 0 {
            return Err(self.fault(CpuFault::Alignment { pc: self.pc, addr }));
        }
        let (ptr, _) = self
            .cfg
            .memory
            .translate(addr)
            .ok_or_else(|| self.data_fault(addr, true))?;
        Ok(ptr)
    }

    fn check_exclusive_alignment(&self, addr: VAddr, bytes: u64) -> Result<(), HaltReason> {
        if addr & (bytes - 1) != 0 {
            return Err(self.fault(CpuFault::Alignment { pc: self.pc, addr }));
        }
        Ok(())
    }

    fn post_index(&mut self, rn: Reg, post: PostIndex, base: u64) {
        match post {
            PostIndex::None => {}
            PostIndex::Imm(n) => self.set(rn, base.wrapping_add(n as u64)),
            PostIndex::Reg(rm) => self.set(rn, base.wrapping_add(self.x(rm))),
        }
    }

    #[inline(always)]
    pub(super) fn execute(&mut self, inst: &Inst) -> Step {
        let pc = self.pc;
        match *inst {
            Inst::Undefined { word } => {
                return Err(self.fault(CpuFault::Undefined {
                    pc,
                    instruction: word,
                }));
            }
            Inst::SimdFp { word } => match simd::execute(self.ctx, word) {
                Some(Ok(())) => {}
                Some(Err(_)) | None => {
                    return Err(self.fault(CpuFault::Undefined {
                        pc,
                        instruction: word,
                    }));
                }
            },
            Inst::Nop | Inst::Hint { .. } => {}

            Inst::Adr { rd, offset } => self.set(rd, pc.wrapping_add(offset as i64 as u64)),
            Inst::Adrp { rd, offset } => self.set(rd, (pc & !0xfff).wrapping_add(offset as u64)),
            Inst::AddSubImm {
                sf,
                sub,
                setflags,
                rd,
                rn,
                imm,
            } => self.add_sub(sf, sub, setflags, rd, self.x(rn), imm as u64),
            Inst::LogicalImm {
                sf,
                op,
                rd,
                rn,
                imm,
            } => self.logical(sf, op, rd, self.x(rn), imm),
            Inst::MovWide {
                sf,
                op,
                rd,
                shift,
                imm,
            } => {
                let imm = (imm as u64) << shift;
                let value = match op {
                    MovOp::N => !imm,
                    MovOp::Z => imm,
                    MovOp::K => (self.x(rd) & !(0xffff << shift)) | imm,
                };
                self.set_sf(sf, rd, value);
            }
            Inst::Bitfield {
                sf,
                op,
                rd,
                rn,
                immr,
                imms,
            } => {
                let value = self.bitfield(sf, op, self.x(rd), self.x(rn), immr, imms);
                self.set_sf(sf, rd, value);
            }
            Inst::Extr {
                sf,
                rd,
                rn,
                rm,
                lsb,
            } => {
                let (n, m) = (self.x(rn), self.x(rm));
                let lsb = lsb as u32;
                let value = if sf {
                    if lsb == 0 {
                        m
                    } else {
                        (m >> lsb) | (n << (64 - lsb))
                    }
                } else {
                    ((n << 32 | (m & 0xffff_ffff)) >> lsb) & 0xffff_ffff
                };
                self.set(rd, value);
            }

            Inst::AddSubShift {
                sf,
                sub,
                setflags,
                rd,
                rn,
                rm,
                shift: ty,
                amount,
            } => {
                let b = shift(sf, self.x(rm), ty, amount);
                self.add_sub(sf, sub, setflags, rd, self.x(rn), b);
            }
            Inst::AddSubExt {
                sf,
                sub,
                setflags,
                rd,
                rn,
                rm,
                extend: ext,
                amount,
            } => {
                let b = extend(self.x(rm), ext, amount);
                self.add_sub(sf, sub, setflags, rd, self.x(rn), b);
            }
            Inst::AddSubCarry {
                sf,
                sub,
                setflags,
                rd,
                rn,
                rm,
            } => {
                let b = if sub { !self.x(rm) } else { self.x(rm) };
                let carry = self.flags() & 2 != 0;
                let (r, flags) = add_with_carry(sf, self.x(rn), b, carry);
                if setflags {
                    self.set_flags(flags);
                }
                self.set_sf(sf, rd, r);
            }
            Inst::LogicalShift {
                sf,
                op,
                invert,
                rd,
                rn,
                rm,
                shift: ty,
                amount,
            } => {
                let b = shift(sf, self.x(rm), ty, amount);
                self.logical(sf, op, rd, self.x(rn), if invert { !b } else { b });
            }
            Inst::CondCompare {
                sf,
                sub,
                rn,
                rhs,
                imm,
                nzcv,
                cond,
            } => {
                let flags = if condition_holds(cond, self.flags()) {
                    let b = if imm { rhs as u64 } else { self.x(Reg(rhs)) };
                    add_with_carry(sf, self.x(rn), if sub { !b } else { b }, sub).1
                } else {
                    nzcv as u32
                };
                self.set_flags(flags);
            }
            Inst::CondSelect {
                sf,
                op,
                rd,
                rn,
                rm,
                cond,
            } => {
                let value = if condition_holds(cond, self.flags()) {
                    self.x(rn)
                } else {
                    let m = self.x(rm);
                    match op {
                        CselOp::Csel => m,
                        CselOp::Csinc => m.wrapping_add(1),
                        CselOp::Csinv => !m,
                        CselOp::Csneg => m.wrapping_neg(),
                    }
                };
                self.set_sf(sf, rd, value);
            }
            Inst::Dp1 { sf, op, rd, rn } => {
                let n = self.x(rn);
                let value = if sf {
                    match op {
                        Dp1Op::Rbit => n.reverse_bits(),
                        Dp1Op::Rev16 => {
                            ((n & 0x00ff_00ff_00ff_00ff) << 8) | ((n >> 8) & 0x00ff_00ff_00ff_00ff)
                        }
                        Dp1Op::Rev32 => n.swap_bytes().rotate_left(32),
                        Dp1Op::Rev64 => n.swap_bytes(),
                        Dp1Op::Clz => n.leading_zeros() as u64,
                        Dp1Op::Cls => {
                            (((n ^ (n >> 1)) & (u64::MAX >> 1)).leading_zeros() - 1) as u64
                        }
                    }
                } else {
                    let n = n as u32;
                    (match op {
                        Dp1Op::Rbit => n.reverse_bits(),
                        Dp1Op::Rev16 => ((n & 0x00ff_00ff) << 8) | ((n >> 8) & 0x00ff_00ff),
                        Dp1Op::Rev32 | Dp1Op::Rev64 => n.swap_bytes(),
                        Dp1Op::Clz => n.leading_zeros(),
                        Dp1Op::Cls => ((n ^ (n >> 1)) & (u32::MAX >> 1)).leading_zeros() - 1,
                    }) as u64
                };
                self.set(rd, value);
            }
            Inst::Dp2 { sf, op, rd, rn, rm } => {
                let (n, m) = (self.x(rn), self.x(rm));
                let value = if sf {
                    match op {
                        Dp2Op::Udiv => n.checked_div(m).unwrap_or(0),
                        Dp2Op::Sdiv => {
                            if m == 0 {
                                0
                            } else {
                                (n as i64).wrapping_div(m as i64) as u64
                            }
                        }
                        Dp2Op::Lslv => n << (m & 63),
                        Dp2Op::Lsrv => n >> (m & 63),
                        Dp2Op::Asrv => ((n as i64) >> (m & 63)) as u64,
                        Dp2Op::Rorv => n.rotate_right((m & 63) as u32),
                    }
                } else {
                    let (n, m) = (n as u32, m as u32);
                    (match op {
                        Dp2Op::Udiv => n.checked_div(m).unwrap_or(0),
                        Dp2Op::Sdiv => {
                            if m == 0 {
                                0
                            } else {
                                (n as i32).wrapping_div(m as i32) as u32
                            }
                        }
                        Dp2Op::Lslv => n << (m & 31),
                        Dp2Op::Lsrv => n >> (m & 31),
                        Dp2Op::Asrv => ((n as i32) >> (m & 31)) as u32,
                        Dp2Op::Rorv => n.rotate_right(m & 31),
                    }) as u64
                };
                self.set(rd, value);
            }
            Inst::Dp3 {
                sf,
                op,
                rd,
                rn,
                rm,
                ra,
            } => {
                let (n, m, a) = (self.x(rn), self.x(rm), self.x(ra));
                let value = match op {
                    Dp3Op::Madd => a.wrapping_add(n.wrapping_mul(m)),
                    Dp3Op::Msub => a.wrapping_sub(n.wrapping_mul(m)),
                    Dp3Op::Smaddl => {
                        a.wrapping_add((n as i32 as i64).wrapping_mul(m as i32 as i64) as u64)
                    }
                    Dp3Op::Smsubl => {
                        a.wrapping_sub((n as i32 as i64).wrapping_mul(m as i32 as i64) as u64)
                    }
                    Dp3Op::Umaddl => a.wrapping_add((n & 0xffff_ffff) * (m & 0xffff_ffff)),
                    Dp3Op::Umsubl => a.wrapping_sub((n & 0xffff_ffff) * (m & 0xffff_ffff)),
                    Dp3Op::Smulh => ((n as i64 as i128 * m as i64 as i128) >> 64) as u64,
                    Dp3Op::Umulh => ((n as u128 * m as u128) >> 64) as u64,
                };
                self.set_sf(sf, rd, value);
            }
            Inst::Crc32 {
                castagnoli,
                size,
                rd,
                rn,
                rm,
            } => {
                let table = if castagnoli { &CRC32C } else { &CRC32 };
                let mut crc = self.x(rn) as u32;
                let data = self.x(rm);
                for i in 0..1u32 << size {
                    let byte = (data >> (i * 8)) as u8;
                    crc = table[((crc ^ byte as u32) & 0xff) as usize] ^ (crc >> 8);
                }
                self.set(rd, crc as u64);
            }
            Inst::Pacga { rd, rn, rm } => {
                self.set(rd, (self.x(rn) ^ self.x(rm)) & 0xffff_ffff_0000_0000);
            }
            Inst::Flag { op } => {
                let f = self.flags();
                let (z, c, v) = (f & 4 != 0, f & 2 != 0, f & 1 != 0);
                let flags = match op {
                    FlagOp::Cfinv => f ^ 2,
                    FlagOp::Xaflag => nzcv(!c && !z, z && c, c || z, !c && z),
                    FlagOp::Axflag => nzcv(false, z || v, c && !v, false),
                };
                self.set_flags(flags);
            }
            Inst::Rmif { rn, shift, mask } => {
                let value = self.x(rn).rotate_right(shift as u32) as u32;
                let mask = mask as u32;
                self.set_flags((self.flags() & !mask) | (value & mask));
            }
            Inst::Setf { rn, sixteen } => {
                let n = self.x(rn) as u32;
                let top = if sixteen { 15 } else { 7 };
                let neg = n >> top & 1 != 0;
                let zero = n & ((2u32 << top) - 1) == 0;
                let overflow = (n >> (top + 1) & 1 != 0) != neg;
                self.set_flags(nzcv(neg, zero, self.flags() & 2 != 0, overflow));
            }

            Inst::B { offset } => return Ok(Flow::Jump(pc.wrapping_add(offset as i64 as u64))),
            Inst::Bl { offset } => {
                self.set(Reg::LR, pc + 4);
                return Ok(Flow::Jump(pc.wrapping_add(offset as i64 as u64)));
            }
            Inst::BCond { cond, offset } => {
                if condition_holds(cond, self.flags()) {
                    return Ok(Flow::Jump(pc.wrapping_add(offset as i64 as u64)));
                }
                return Ok(Flow::Jump(pc + 4));
            }
            Inst::Cbz {
                sf,
                nonzero,
                rt,
                offset,
            } => {
                let zero = self.x(rt) & width_mask(sf) == 0;
                if zero != nonzero {
                    return Ok(Flow::Jump(pc.wrapping_add(offset as i64 as u64)));
                }
                return Ok(Flow::Jump(pc + 4));
            }
            Inst::Tbz {
                nonzero,
                rt,
                bit,
                offset,
            } => {
                if (self.x(rt) >> bit & 1 != 0) == nonzero {
                    return Ok(Flow::Jump(pc.wrapping_add(offset as i64 as u64)));
                }
                return Ok(Flow::Jump(pc + 4));
            }
            Inst::Br { rn } | Inst::Ret { rn } => return Ok(Flow::Jump(self.x(rn))),
            Inst::Blr { rn } => {
                let target = self.x(rn);
                self.set(Reg::LR, pc + 4);
                return Ok(Flow::Jump(target));
            }
            Inst::Svc { imm } => return Err(HaltReason::SupervisorCall(imm as u32)),
            Inst::Brk { imm } => return Err(HaltReason::Breakpoint(imm)),

            Inst::Barrier {
                op: BarrierOp::Dmb | BarrierOp::Dsb,
            } => fence(Ordering::SeqCst),
            Inst::Barrier { .. } => {}
            Inst::Clrex => self.cfg.monitor.clear(self.cfg.core_index),
            Inst::Mrs { rt, reg } => {
                let value = match reg {
                    SysReg::Nzcv => (self.ctx.pstate & (0xf << NZCV_SHIFT)) as u64,
                    SysReg::Fpcr => self.ctx.fpcr as u64,
                    SysReg::Fpsr => self.ctx.fpsr as u64,
                    SysReg::TpidrEl0 => self.ctx.tpidr_el0,
                    SysReg::TpidrroEl0 => self.ctx.tpidrro_el0,
                    SysReg::CntfrqEl0 => self.cfg.clock.frequency(),
                    SysReg::CntpctEl0 | SysReg::CntvctEl0 => self.cfg.clock.ticks(),
                    SysReg::CtrEl0 => self.cfg.ctr_el0 as u64,
                    SysReg::DczidEl0 => self.cfg.dczid_el0 as u64,
                };
                self.set(rt, value);
            }
            Inst::Msr { rt, reg } => {
                let value = self.x(rt);
                match reg {
                    SysReg::Nzcv => self.set_flags((value >> NZCV_SHIFT) as u32 & 0xf),
                    SysReg::Fpcr => self.ctx.fpcr = value as u32,
                    SysReg::Fpsr => self.ctx.fpsr = value as u32,
                    SysReg::TpidrEl0 => self.ctx.tpidr_el0 = value,
                    _ => unreachable!("the decoder rejects writes to read-only registers"),
                }
            }
            Inst::DcZva { rt } => {
                let size = 4u64 << (self.cfg.dczid_el0 & 0xf);
                let addr = self.x(rt) & !(size - 1);
                self.cfg
                    .memory
                    .fill(addr, size, 0)
                    .map_err(|f| self.data_fault(f.addr, true))?;
            }
            Inst::IcIvau { rt } => {
                let line = 4u64 << (self.cfg.ctr_el0 & 0xf);
                return Ok(Flow::Invalidate(self.x(rt) & !(line - 1), line));
            }
            Inst::IcIallu => return Ok(Flow::InvalidateAll),

            Inst::LoadLiteral {
                size,
                ext,
                rt,
                offset,
            } => {
                let value = self.read(pc.wrapping_add(offset as i64 as u64), size)?;
                self.set(rt, load_extend(value, size, ext));
            }
            Inst::LoadLiteralVec { size, vt, offset } => {
                self.ctx.v[vt as usize] =
                    self.read_vec(pc.wrapping_add(offset as i64 as u64), size)?;
            }
            Inst::Load {
                size,
                ext,
                acquire,
                rt,
                rn,
                addr,
            } => {
                let (address, wb) = self.address(rn, addr);
                let value = self.read(address, size)?;
                if acquire {
                    fence(Ordering::Acquire);
                }
                self.writeback(rn, wb);
                self.set(rt, load_extend(value, size, ext));
            }
            Inst::Store {
                size,
                release,
                rt,
                rn,
                addr,
            } => {
                let (address, wb) = self.address(rn, addr);
                if release {
                    fence(Ordering::Release);
                }
                self.write(address, size, self.x(rt))?;
                self.writeback(rn, wb);
            }
            Inst::LoadVec { size, vt, rn, addr } => {
                let (address, wb) = self.address(rn, addr);
                self.ctx.v[vt as usize] = self.read_vec(address, size)?;
                self.writeback(rn, wb);
            }
            Inst::StoreVec { size, vt, rn, addr } => {
                let (address, wb) = self.address(rn, addr);
                self.write_vec(address, size, self.ctx.v[vt as usize])?;
                self.writeback(rn, wb);
            }
            Inst::LoadPair {
                size,
                ext,
                rt,
                rt2,
                rn,
                addr,
            } => {
                let (address, wb) = self.address(rn, addr);
                let first = self.read(address, size)?;
                let second = self.read(address.wrapping_add(1 << size), size)?;
                self.writeback(rn, wb);
                self.set(rt, load_extend(first, size, ext));
                self.set(rt2, load_extend(second, size, ext));
            }
            Inst::StorePair {
                size,
                rt,
                rt2,
                rn,
                addr,
            } => {
                let (address, wb) = self.address(rn, addr);
                self.write(address, size, self.x(rt))?;
                self.write(address.wrapping_add(1 << size), size, self.x(rt2))?;
                self.writeback(rn, wb);
            }
            Inst::LoadPairVec {
                size,
                vt,
                vt2,
                rn,
                addr,
            } => {
                let (address, wb) = self.address(rn, addr);
                let first = self.read_vec(address, size)?;
                let second = self.read_vec(address.wrapping_add(1 << size), size)?;
                self.writeback(rn, wb);
                self.ctx.v[vt as usize] = first;
                self.ctx.v[vt2 as usize] = second;
            }
            Inst::StorePairVec {
                size,
                vt,
                vt2,
                rn,
                addr,
            } => {
                let (address, wb) = self.address(rn, addr);
                self.write_vec(address, size, self.ctx.v[vt as usize])?;
                self.write_vec(
                    address.wrapping_add(1 << size),
                    size,
                    self.ctx.v[vt2 as usize],
                )?;
                self.writeback(rn, wb);
            }
            Inst::StoreTag {
                zero_granules,
                rn,
                addr,
            } => {
                let (address, wb) = self.address(rn, addr);
                if zero_granules != 0 {
                    self.cfg
                        .memory
                        .fill(address, 16 * zero_granules as u64, 0)
                        .map_err(|f| self.data_fault(f.addr, true))?;
                }
                self.writeback(rn, wb);
            }

            Inst::LoadExclusive {
                size,
                pair,
                acquire,
                rt,
                rt2,
                rn,
            } => {
                let address = self.x(rn);
                let bytes = (1u64 << size) << pair as u32;
                self.check_exclusive_alignment(address, bytes)?;
                let value = self
                    .cfg
                    .monitor
                    .read(self.cfg.core_index, &self.cfg.memory, address, bytes as u8)
                    .map_err(|f| self.data_fault(f.addr, false))?;
                if acquire {
                    fence(Ordering::Acquire);
                }
                if pair {
                    let bits = 8 << size;
                    self.set(rt, value as u64 & width_bits(bits));
                    self.set(rt2, (value >> bits) as u64 & width_bits(bits));
                } else {
                    self.set(rt, value as u64);
                }
            }
            Inst::StoreExclusive {
                size,
                pair,
                release,
                rs,
                rt,
                rt2,
                rn,
            } => {
                let address = self.x(rn);
                let bytes = (1u64 << size) << pair as u32;
                self.check_exclusive_alignment(address, bytes)?;
                let bits = 8 << size;
                let value = if pair {
                    (self.x(rt2) as u128 & width_bits(bits) as u128) << bits
                        | (self.x(rt) & width_bits(bits)) as u128
                } else {
                    (self.x(rt) & width_bits(bits)) as u128
                };
                if release {
                    fence(Ordering::Release);
                }
                let stored = self
                    .cfg
                    .monitor
                    .write(
                        self.cfg.core_index,
                        &self.cfg.memory,
                        address,
                        bytes as u8,
                        value,
                    )
                    .map_err(|f| self.data_fault(f.addr, true))?;
                self.set(rs, !stored as u64);
            }
            Inst::AtomicRmw {
                op,
                size,
                rs,
                rt,
                rn,
                ..
            } => {
                let ptr = self.atomic_ptr(self.x(rn), 1 << size)?;
                let operand = self.x(rs);
                // SAFETY: `atomic_ptr` checked that the access is naturally aligned and inside one
                // mapped guest page, which outlives this call.
                let old = unsafe { atomic_rmw(ptr, size, op, operand) };
                self.set(rt, old);
            }
            Inst::Cas {
                size, rs, rt, rn, ..
            } => {
                let ptr = self.atomic_ptr(self.x(rn), 1 << size)?;
                let (expected, new) = (self.x(rs), self.x(rt));
                // SAFETY: as for `AtomicRmw`.
                let old = unsafe { atomic_cas(ptr, size, expected, new) };
                self.set(rs, old);
            }
            Inst::Casp {
                size, rs, rt, rn, ..
            } => {
                let ptr = self.atomic_ptr(self.x(rn), 2 << size)?;
                let pair = |r: u8| (self.x(Reg(r)), self.x(Reg(r + 1)));
                let ((e0, e1), (n0, n1)) = (pair(rs), pair(rt));
                let (o0, o1) = if size == 2 {
                    let expected = (e1 << 32) | (e0 & 0xffff_ffff);
                    let new = (n1 << 32) | (n0 & 0xffff_ffff);
                    // SAFETY: as for `AtomicRmw`.
                    let old = unsafe { atomic_cas(ptr, 3, expected, new) };
                    (old & 0xffff_ffff, old >> 32)
                } else {
                    let expected = (e1 as u128) << 64 | e0 as u128;
                    let new = (n1 as u128) << 64 | n0 as u128;
                    // SAFETY: `atomic_ptr` checked 16-byte alignment, so the pair lies in one page.
                    let old = unsafe { cas128(ptr, expected, new) };
                    (old as u64, (old >> 64) as u64)
                };
                self.set(Reg(rs), o0);
                self.set(Reg(rs + 1), o1);
            }

            Inst::VecMulti {
                load,
                q,
                size,
                rpt,
                selem,
                vt,
                rn,
                post,
            } => {
                let base = self.x(rn);
                let elements = (if q { 16 } else { 8 }) >> size;
                let mut regs = [0u128; 4];
                let mut address = base;
                for r in 0..rpt as usize {
                    for e in 0..elements {
                        for s in 0..selem as usize {
                            let reg = r + s;
                            if load {
                                let value = self.read(address, size)?;
                                set_lane(&mut regs[reg], size, e, value);
                            } else {
                                let v = self.ctx.v[(vt as usize + reg) % 32];
                                self.write(address, size, lane(v, size, e))?;
                            }
                            address = address.wrapping_add(1 << size);
                        }
                    }
                }
                if load {
                    for (reg, value) in regs.iter().enumerate().take((rpt * selem) as usize) {
                        self.ctx.v[(vt as usize + reg) % 32] = *value;
                    }
                }
                self.post_index(rn, post, base);
            }
            Inst::VecSingle {
                load,
                size,
                selem,
                index,
                vt,
                rn,
                post,
            } => {
                let base = self.x(rn);
                let mut address = base;
                for s in 0..selem as usize {
                    let reg = (vt as usize + s) % 32;
                    if load {
                        let value = self.read(address, size)?;
                        set_lane(&mut self.ctx.v[reg], size, index as usize, value);
                    } else {
                        self.write(address, size, lane(self.ctx.v[reg], size, index as usize))?;
                    }
                    address = address.wrapping_add(1 << size);
                }
                self.post_index(rn, post, base);
            }
            Inst::VecReplicate {
                q,
                size,
                selem,
                vt,
                rn,
                post,
            } => {
                let base = self.x(rn);
                let mut address = base;
                let lanes = (if q { 16 } else { 8 }) >> size;
                for s in 0..selem as usize {
                    let value = self.read(address, size)?;
                    let mut v = 0u128;
                    for e in 0..lanes {
                        set_lane(&mut v, size, e, value);
                    }
                    self.ctx.v[(vt as usize + s) % 32] = v;
                    address = address.wrapping_add(1 << size);
                }
                self.post_index(rn, post, base);
            }
        }
        Ok(Flow::Next)
    }

    #[inline]
    fn add_sub(&mut self, sf: bool, sub: bool, setflags: bool, rd: Reg, a: u64, b: u64) {
        let (r, flags) = add_with_carry(sf, a, if sub { !b } else { b }, sub);
        if setflags {
            self.set_flags(flags);
        }
        self.set_sf(sf, rd, r);
    }

    #[inline]
    fn logical(&mut self, sf: bool, op: LogicOp, rd: Reg, a: u64, b: u64) {
        let r = match op {
            LogicOp::And | LogicOp::Ands => a & b,
            LogicOp::Orr => a | b,
            LogicOp::Eor => a ^ b,
        } & width_mask(sf);
        if op == LogicOp::Ands {
            self.set_flags(logic_flags(sf, r));
        }
        self.set(rd, r);
    }

    #[inline]
    fn bitfield(&self, sf: bool, op: BitfieldOp, dst: u64, src: u64, immr: u8, imms: u8) -> u64 {
        let Some((wmask, tmask)) = decode_bit_masks(sf, imms as u32, immr as u32, false) else {
            unreachable!("the decoder only emits valid bitfield immediates");
        };
        let rotated = if sf {
            src.rotate_right(immr as u32)
        } else {
            (src as u32).rotate_right(immr as u32) as u64
        };
        match op {
            BitfieldOp::Sbfm => {
                let top = if src >> imms & 1 != 0 { u64::MAX } else { 0 };
                (top & !tmask) | (rotated & wmask & tmask)
            }
            BitfieldOp::Bfm => {
                let bottom = (dst & !wmask) | (rotated & wmask);
                (dst & !tmask) | (bottom & tmask)
            }
            BitfieldOp::Ubfm => rotated & wmask & tmask,
        }
    }
}

macro_rules! rmw {
    ($unsigned:ty, $signed:ty, $u:ty, $i:ty, $ptr:expr, $op:expr, $operand:expr) => {{
        // SAFETY: guaranteed by `atomic_rmw`'s caller.
        let (u, s) = unsafe {
            (
                <$unsigned>::from_ptr($ptr.cast()),
                <$signed>::from_ptr($ptr.cast()),
            )
        };
        let order = Ordering::SeqCst;
        let v = $operand;
        (match $op {
            AtomicOp::Add => u.fetch_add(v as $u, order),
            AtomicOp::Clr => u.fetch_and(!(v as $u), order),
            AtomicOp::Eor => u.fetch_xor(v as $u, order),
            AtomicOp::Set => u.fetch_or(v as $u, order),
            AtomicOp::Smax => s.fetch_max(v as $i, order) as $u,
            AtomicOp::Smin => s.fetch_min(v as $i, order) as $u,
            AtomicOp::Umax => u.fetch_max(v as $u, order),
            AtomicOp::Umin => u.fetch_min(v as $u, order),
            AtomicOp::Swp => u.swap(v as $u, order),
        }) as u64
    }};
}

/// # Safety
/// `ptr` must be valid for `1 << size` bytes and aligned to them.
pub(crate) unsafe fn atomic_rmw(ptr: *mut u8, size: u8, op: AtomicOp, operand: u64) -> u64 {
    match size {
        0 => rmw!(AtomicU8, AtomicI8, u8, i8, ptr, op, operand),
        1 => rmw!(AtomicU16, AtomicI16, u16, i16, ptr, op, operand),
        2 => rmw!(AtomicU32, AtomicI32, u32, i32, ptr, op, operand),
        _ => rmw!(AtomicU64, AtomicI64, u64, i64, ptr, op, operand),
    }
}

/// Returns the value memory held before the compare-and-swap.
///
/// # Safety
/// `ptr` must be valid for `1 << size` bytes and aligned to them.
pub(crate) unsafe fn atomic_cas(ptr: *mut u8, size: u8, expected: u64, new: u64) -> u64 {
    macro_rules! cas {
        ($atomic:ty) => {{
            // SAFETY: guaranteed by the caller.
            let a = unsafe { <$atomic>::from_ptr(ptr.cast()) };
            let r = a.compare_exchange(expected as _, new as _, Ordering::SeqCst, Ordering::SeqCst);
            r.unwrap_or_else(|v| v) as u64
        }};
    }
    match size {
        0 => cas!(AtomicU8),
        1 => cas!(AtomicU16),
        2 => cas!(AtomicU32),
        _ => cas!(AtomicU64),
    }
}

/// # Safety
/// `ptr` must be valid for 16 bytes and 16-byte aligned.
unsafe fn cas128(ptr: *mut u8, expected: u128, new: u128) -> u128 {
    // SAFETY: guaranteed by the caller.
    let a = unsafe { portable_atomic::AtomicU128::from_ptr(ptr.cast()) };
    a.compare_exchange(expected, new, Ordering::SeqCst, Ordering::SeqCst)
        .unwrap_or_else(|v| v)
}
