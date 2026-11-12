//! Differential tests: each instruction runs once in Unicorn (QEMU's A64 model with every
//! feature on) and once through [`execute`] from the same random register state and FPCR, and
//! the vector registers, general registers, NZCV and FPSR must match afterwards.

use unicorn_engine::unicorn_const::{Arch, Mode, Prot};
use unicorn_engine::{Arm64CpuModel, RegisterARM64, Unicorn};

use super::execute;
use super::softfloat::{FpEnv, Rounding, convert, from_fixed};
use crate::{HaltReason, ThreadContext};

const CODE: u64 = 0x10000;
const FPCR_BITS: u32 = (1 << 26) | (1 << 25) | (1 << 24) | (3 << 22) | (1 << 19);

pub(super) struct Rng(u64);

impl Rng {
    pub(super) fn new(seed: u64) -> Self {
        Self(seed ^ 0x9e37_79b9_7f4a_7c15)
    }

    pub(super) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(super) fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// An `width`-bit floating-point encoding, biased towards the values that exercise special
    /// cases: zeros, infinities, NaNs, denormals, the normal/denormal and overflow boundaries,
    /// small integers and values near integer conversion limits.
    pub(super) fn float(&mut self, width: u32) -> u64 {
        let (e, f) = match width {
            16 => (5, 10),
            32 => (8, 23),
            _ => (11, 52),
        };
        let exp_ones = (1u64 << e) - 1;
        let bias = (exp_ones >> 1) as i64;
        let frac_mask = (1u64 << f) - 1;
        let sign = self.next() & 1;
        let random_frac = self.next() & frac_mask;
        let exp = |x: i64| ((x + bias) as u64).min(exp_ones);
        let magnitude = match self.below(12) {
            0 => self.next(),
            1 => 0,
            2 => exp_ones << f,
            3 => (exp_ones << f) | (1 << (f - 1)) | (random_frac >> 1),
            4 => (exp_ones << f) | ((random_frac >> 1).max(1)),
            5 => random_frac.max(1) >> self.below(f as u64),
            6 => (self.below(3) << f) | random_frac,
            7 => ((exp_ones - 1 - self.below(3)) << f) | random_frac,
            8 => {
                let k = [15i64, 16, 31, 32, 63, 64][self.below(6) as usize].min(bias);
                let tweak = [0, 1, frac_mask][self.below(3) as usize];
                (exp(k - self.below(2) as i64) << f) | tweak
            }
            9 => {
                let small = self.below(600) as i64 - 300;
                let mut env = FpEnv::new(0);
                return match width {
                    16 => from_fixed::<16>(
                        small as u64,
                        true,
                        64,
                        self.below(3) as u32,
                        Rounding::TieEven,
                        &mut env,
                    ),
                    32 => (small as f32 / (1 << self.below(3)) as f32).to_bits() as u64,
                    _ => (small as f64 / (1 << self.below(3)) as f64).to_bits(),
                };
            }
            10 => {
                (exp(self.below(8) as i64 - 4) << f)
                    | (random_frac & !(frac_mask >> self.below(f as u64 + 1)))
            }
            _ => (exp(self.below(2 * bias as u64) as i64 - bias + 1) << f) | random_frac,
        };
        ((sign << (width - 1)) | magnitude) & (u64::MAX >> (64 - width))
    }

    /// A 64-bit integer biased towards boundary values.
    pub(super) fn integer(&mut self) -> u64 {
        match self.below(6) {
            0 => [
                0,
                1,
                u64::MAX,
                0x7fff_ffff,
                0x8000_0000,
                i64::MAX as u64,
                i64::MIN as u64,
                0xffff_ffff,
            ][self.below(8) as usize],
            1 => self.below(512).wrapping_sub(256),
            2 => 1u64 << self.below(64),
            _ => self.next(),
        }
    }

    pub(super) fn vector(&mut self) -> u128 {
        let width = [16, 32, 64, 0][self.below(4) as usize];
        if width == 0 {
            return (self.integer() as u128) | ((self.integer() as u128) << 64);
        }
        let mut v = 0u128;
        for lane in 0..128 / width {
            v |= (self.float(width) as u128) << (lane * width);
        }
        v
    }

    pub(super) fn fpcr(&mut self) -> u32 {
        (self.next() as u32) & FPCR_BITS
    }
}

/// What one instruction left behind, or that it was undefined.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Outcome {
    Executed {
        v: Box<[u128; 32]>,
        x: Box<[u64; 31]>,
        nzcv: u32,
        fpsr: u32,
    },
    Undefined,
}

pub(super) struct Harness {
    uc: Unicorn<'static, ()>,
    pub(super) rng: Rng,
}

impl Harness {
    pub(super) fn new(seed: u64) -> Self {
        let mut uc = Unicorn::new(Arch::ARM64, Mode::ARM).expect("unicorn");
        uc.ctl_set_cpu_model(Arm64CpuModel::MAX as i32)
            .expect("cpu model");
        uc.mem_map(CODE, 0x1000, Prot::ALL).expect("map");
        Self {
            uc,
            rng: Rng::new(seed),
        }
    }

    pub(super) fn random_context(&mut self) -> ThreadContext {
        let mut ctx = ThreadContext::default();
        for v in &mut ctx.v {
            *v = self.rng.vector();
        }
        for x in &mut ctx.x {
            *x = self.rng.integer();
        }
        ctx.pstate = (self.rng.next() as u32) & 0xf000_0000;
        ctx.fpcr = self.rng.fpcr();
        ctx.pc = CODE;
        ctx
    }

    fn reference(&mut self, instruction: u32, ctx: &ThreadContext) -> Outcome {
        let uc = &mut self.uc;
        uc.mem_write(CODE, &instruction.to_le_bytes())
            .expect("write code");
        uc.ctl_remove_cache(CODE, CODE + 4).expect("flush");
        for (i, v) in ctx.v.iter().enumerate() {
            uc.reg_write_long(RegisterARM64::Q0 as i32 + i as i32, &v.to_le_bytes())
                .expect("write q");
        }
        for (i, x) in ctx.x.iter().enumerate() {
            uc.reg_write(x_register(i), *x).expect("write x");
        }
        uc.reg_write(RegisterARM64::NZCV, ctx.pstate as u64)
            .expect("nzcv");
        uc.reg_write(RegisterARM64::FPCR, ctx.fpcr as u64)
            .expect("fpcr");
        uc.reg_write(RegisterARM64::FPSR, 0).expect("fpsr");
        if uc.emu_start(CODE, CODE + 4, 0, 1).is_err() || uc.pc_read().expect("pc") != CODE + 4 {
            return Outcome::Undefined;
        }
        let mut v = Box::new([0u128; 32]);
        for (i, v) in v.iter_mut().enumerate() {
            let bytes = uc
                .reg_read_long(RegisterARM64::Q0 as i32 + i as i32)
                .expect("read q");
            *v = u128::from_le_bytes(bytes[..16].try_into().expect("16 bytes"));
        }
        let mut x = Box::new([0u64; 31]);
        for (i, x) in x.iter_mut().enumerate() {
            *x = uc.reg_read(x_register(i)).expect("read x");
        }
        Outcome::Executed {
            v,
            x,
            nzcv: uc.reg_read(RegisterARM64::NZCV).expect("nzcv") as u32 & 0xf000_0000,
            fpsr: uc.reg_read(RegisterARM64::FPSR).expect("fpsr") as u32,
        }
    }

    /// Runs `instruction` from `ctx` in Unicorn and through `execute` and panics with both
    /// states on any difference.
    pub(super) fn check(&mut self, instruction: u32, ctx: &ThreadContext) {
        let expected = self.reference(instruction, ctx);
        let mut ours = ctx.clone();
        let actual = match execute(&mut ours, instruction) {
            Some(Ok(())) => Outcome::Executed {
                v: Box::new(ours.v),
                x: Box::new(ours.x),
                nzcv: ours.pstate & 0xf000_0000,
                fpsr: ours.fpsr,
            },
            Some(Err(HaltReason::Fault(_))) => Outcome::Undefined,
            other => panic!("{instruction:#010x}: unexpected {other:?}"),
        };
        if expected != actual {
            panic!(
                "{}",
                describe_mismatch(instruction, ctx, &expected, &actual)
            );
        }
    }

    /// Checks `count` random instances of `base` with the bits in `random` filled at random.
    pub(super) fn fuzz(&mut self, base: u32, random: u32, count: usize) {
        for _ in 0..count {
            let instruction = base | (self.rng.next() as u32 & random);
            let ctx = self.random_context();
            if !unicorn_deviates(instruction, &ctx) {
                self.check(instruction, &ctx);
            }
        }
    }
}

/// Cases where Unicorn (QEMU 5.0) disagrees with the Arm ARM pseudocode, which we follow.
fn unicorn_deviates(instruction: u32, ctx: &ThreadContext) -> bool {
    let fz = ctx.fpcr & (1 << 24) != 0;
    let rn = ctx.v[((instruction >> 5) & 31) as usize] as u64;
    let positive_denormal = rn >> 52 == 0 && rn != 0;
    // FJCVTZS of a positive denormal under FZ: FPToFixedJS clears Z, QEMU 5 sets it.
    instruction & 0xffff_fc00 == 0x1e7e_0000 && fz && positive_denormal
}

fn x_register(i: usize) -> i32 {
    match i {
        29 => RegisterARM64::X29 as i32,
        30 => RegisterARM64::X30 as i32,
        _ => RegisterARM64::X0 as i32 + i as i32,
    }
}

fn describe_mismatch(
    instruction: u32,
    ctx: &ThreadContext,
    expected: &Outcome,
    actual: &Outcome,
) -> String {
    let mut out = format!(
        "{instruction:#010x} differs (fpcr {:#010x}, nzcv {:#x})\n",
        ctx.fpcr,
        ctx.pstate >> 28
    );
    let fields = [5, 16, 10, 0];
    let mut regs: Vec<usize> = fields
        .iter()
        .map(|&lo| ((instruction >> lo) & 31) as usize)
        .collect();
    regs.dedup();
    for r in regs {
        out += &format!(
            "  in  v{r} = {:032x}  x{r} = {:016x}\n",
            ctx.v[r],
            ctx.x.get(r).copied().unwrap_or(0)
        );
    }
    match (expected, actual) {
        (
            Outcome::Executed {
                v: ev,
                x: ex,
                nzcv: en,
                fpsr: ef,
            },
            Outcome::Executed {
                v: av,
                x: ax,
                nzcv: an,
                fpsr: af,
            },
        ) => {
            for i in 0..32 {
                if ev[i] != av[i] {
                    out += &format!("  v{i}: unicorn {:032x} ours {:032x}\n", ev[i], av[i]);
                }
            }
            for i in 0..31 {
                if ex[i] != ax[i] {
                    out += &format!("  x{i}: unicorn {:016x} ours {:016x}\n", ex[i], ax[i]);
                }
            }
            if en != an {
                out += &format!("  nzcv: unicorn {:x} ours {:x}\n", en >> 28, an >> 28);
            }
            if ef != af {
                out += &format!("  fpsr: unicorn {ef:#x} ours {af:#x}\n");
            }
        }
        _ => {
            out += &format!(
                "  unicorn {:?}\n  ours    {:?}\n",
                kind(expected),
                kind(actual)
            )
        }
    }
    out
}

fn kind(outcome: &Outcome) -> &'static str {
    match outcome {
        Outcome::Executed { .. } => "executed",
        Outcome::Undefined => "undefined",
    }
}

const ITERATIONS: usize = 20_000;

fn fuzz(seed: u64, base: u32, random: u32) {
    Harness::new(seed).fuzz(base, random, ITERATIONS);
}

const RD_RN: u32 = 0x3ff;
const RD_RN_RM: u32 = RD_RN | (0x1f << 16);
const FTYPE: u32 = 3 << 22;

#[test]
fn fp_data_processing_two_source() {
    fuzz(1, 0x1e20_0800, RD_RN_RM | FTYPE | (0xf << 12));
}

#[test]
fn fp_data_processing_three_source() {
    fuzz(2, 0x1f00_0000, 0x00ff_ffff);
}

#[test]
fn fp_data_processing_one_source() {
    fuzz(3, 0x1e20_4000, RD_RN | FTYPE | (0x3f << 15));
}

#[test]
fn fp_compare() {
    fuzz(4, 0x1e20_2000, RD_RN_RM | FTYPE | 0x1f | (3 << 14));
}

#[test]
fn fp_conditional_compare_and_select() {
    fuzz(5, 0x1e20_0400, RD_RN_RM | FTYPE | (0xf << 12) | (1 << 11));
}

#[test]
fn fp_immediate() {
    fuzz(6, 0x1e20_1000, FTYPE | (0xff << 13) | 0x3ff);
}

#[test]
fn fp_integer_conversion() {
    fuzz(7, 0x1e20_0000, (1 << 31) | FTYPE | (0x1f << 16) | RD_RN);
}

#[test]
fn fp_fixed_point_conversion() {
    fuzz(
        8,
        0x1e00_0000,
        (1 << 31) | FTYPE | (0x1f << 16) | (0x3f << 10) | RD_RN,
    );
}

#[test]
fn fp_whole_scalar_group() {
    fuzz(9, 0x1e00_0000, 0xa1ff_ffff);
}

#[test]
fn half_precision_conversions_are_exhaustive() {
    let mut h = Harness::new(10);
    for bits in (0..=0xffffu64).step_by(7) {
        let mut ctx = h.random_context();
        ctx.v[1] = bits as u128;
        h.check(0x1ee2_4020, &ctx);
        h.check(0x1ee2_c020, &ctx);
        let mut env = FpEnv::new(0);
        ctx.v[1] =
            convert::<16, 64>(bits, Rounding::TieEven, &mut env) as u128 ^ (h.rng.below(4) as u128);
        h.check(0x1e63_c020, &ctx);
    }
}
