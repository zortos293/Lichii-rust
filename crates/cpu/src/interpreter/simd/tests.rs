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
        if qemu_aborts(instruction) {
            let mut ours = ctx.clone();
            assert!(
                matches!(execute(&mut ours, instruction), Some(Err(_))),
                "{instruction:#010x} is unallocated"
            );
            return;
        }
        let mut expected = self.reference(instruction, ctx);
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
        if qemu_computes_hidden_lanes(instruction)
            && let (Outcome::Executed { fpsr: theirs, .. }, Outcome::Executed { fpsr: ours, .. }) =
                (&mut expected, &actual)
        {
            *theirs = *ours;
        }
        if expected != actual {
            panic!(
                "{}",
                describe_mismatch(instruction, ctx, &expected, &actual)
            );
        }
    }

    /// Checks `count` random instances of `base` with the bits in `random` filled at random.
    pub(super) fn fuzz(&mut self, base: u32, random: u32, count: usize) {
        self.fuzz_where(base, random, count, |_| true);
    }

    /// Like [`Harness::fuzz`], skipping the instructions `keep` rejects.
    pub(super) fn fuzz_where(
        &mut self,
        base: u32,
        random: u32,
        count: usize,
        keep: impl Fn(u32) -> bool,
    ) {
        let mut checked = 0;
        while checked < count {
            let instruction = base | (self.rng.next() as u32 & random);
            if !keep(instruction) {
                continue;
            }
            let ctx = self.random_context();
            if !unicorn_deviates(instruction, &ctx) {
                self.check(instruction, &ctx);
            }
            checked += 1;
        }
    }
}

/// Unallocated encodings on which QEMU 5.0 hits `g_assert_not_reached` and aborts the process
/// instead of raising an undefined-instruction exception.
fn qemu_aborts(instruction: u32) -> bool {
    let fp16_three_same =
        instruction & 0x9f60_c400 == 0x0e40_0400 || instruction & 0xdf60_c400 == 0x5e40_0400;
    let key = (
        (instruction >> 29) & 1,
        (instruction >> 23) & 1,
        (instruction >> 11) & 7,
    );
    let three_same_unallocated = matches!(
        key,
        (0, 0, 0b101) | (0, 1, 0b011 | 0b100 | 0b101) | (1, _, 0b001) | (1, 1, 0b011 | 0b111)
    );
    let fp16_two_misc =
        instruction & 0x9f7e_0c00 == 0x0e78_0800 || instruction & 0xdf7e_0c00 == 0x5e78_0800;
    let key = (
        (instruction >> 29) & 1,
        (instruction >> 23) & 1,
        (instruction >> 12) & 0x1f,
    );
    let two_misc_allocated = if instruction & (1 << 28) == 0 {
        matches!(
            key,
            (_, 0, 0b11000..=0b11101)
                | (0, 1, 0b01100..=0b01111 | 0b11000..=0b11011 | 0b11101)
                | (1, 1, 0b01100 | 0b01101 | 0b01111 | 0b11001..=0b11011 | 0b11101 | 0b11111)
        )
    } else {
        matches!(
            key,
            (_, 0, 0b11010..=0b11101)
                | (
                    0,
                    1,
                    0b01100..=0b01110 | 0b11010 | 0b11011 | 0b11101 | 0b11111
                )
                | (1, 1, 0b01100 | 0b01101 | 0b11010 | 0b11011 | 0b11101)
        )
    };
    (fp16_three_same && three_same_unallocated) || (fp16_two_misc && !two_misc_allocated)
}

/// FCMLA (by element, half precision) with Q = 0: QEMU 5 also computes the upper four lanes and
/// raises their FPSR flags before discarding them, so only the registers are comparable.
fn qemu_computes_hidden_lanes(instruction: u32) -> bool {
    instruction & 0xffc0_9400 == 0x2f40_1000
}

/// Cases where Unicorn (QEMU 5.0) disagrees with the Arm ARM pseudocode, which we follow.
fn unicorn_deviates(instruction: u32, ctx: &ThreadContext) -> bool {
    let fz = ctx.fpcr & (1 << 24) != 0;
    let rn = ctx.v[((instruction >> 5) & 31) as usize] as u64;
    let positive_denormal = rn >> 52 == 0 && rn != 0;
    // FJCVTZS of a positive denormal under FZ: FPToFixedJS clears Z, QEMU 5 sets it.
    let fjcvtzs = instruction & 0xffff_fc00 == 0x1e7e_0000 && fz && positive_denormal;
    // Modified immediate with op = 1, cmode = 1111, o2 = 1 is unallocated; QEMU 5 runs it as
    // FMOV (vector, immediate) double precision.
    let fmov_o2 = instruction & 0xbff8_fc00 == 0x2f00_fc00;
    // Scalar FCVTZS (fixed point) to a half-precision register: QEMU 5 sign-extends the 16-bit
    // result into 32 bits instead of clearing the rest of the register.
    let fcvtzs_h = instruction & 0xff70_fc00 == 0x5f10_fc00;
    // The half-precision FP reductions (U = 0) need size<0> = 0; QEMU 5 ignores it.
    let half_reduction_sz = (instruction & 0x9f7e_0c00 == 0x0e70_0800
        || instruction & 0xdf7e_0c00 == 0x5e70_0800)
        && matches!((instruction >> 12) & 0x1f, 0b01100 | 0b01101 | 0b01111);
    fjcvtzs || fmov_o2 || fcvtzs_h || half_reduction_sz
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

const Q: u32 = 1 << 30;
const U: u32 = 1 << 29;
const SIZE: u32 = 3 << 22;
const RM: u32 = 0x1f << 16;

fn fuzz_where(seed: u64, base: u32, random: u32, keep: impl Fn(u32) -> bool) {
    Harness::new(seed).fuzz_where(base, random, ITERATIONS, keep);
}

fn field(instruction: u32, hi: u32, lo: u32) -> u32 {
    (instruction >> lo) & ((1 << (hi - lo + 1)) - 1)
}

fn integer_three_same(i: u32) -> bool {
    field(i, 15, 11) < 0b11000
}

fn integer_two_misc(i: u32) -> bool {
    let opcode = field(i, 16, 12);
    opcode <= 0b01011 || (0b10010..=0b10100).contains(&opcode)
}

fn integer_indexed(i: u32) -> bool {
    let u = i & U != 0;
    matches!(
        (field(i, 15, 12), u),
        (0b0010 | 0b0110 | 0b1010 | 0b1110, _)
            | (0b0011 | 0b0111 | 0b1011 | 0b1100 | 0b1101 | 0b1000, false)
            | (0b0000 | 0b0100 | 0b1101 | 0b1111, true)
    )
}

#[test]
fn advsimd_three_same_integer() {
    fuzz_where(
        20,
        0x0e20_0400,
        Q | U | SIZE | RM | (0x1f << 11) | RD_RN,
        integer_three_same,
    );
    fuzz_where(
        21,
        0x5e20_0400,
        U | SIZE | RM | (0x1f << 11) | RD_RN,
        integer_three_same,
    );
}

#[test]
fn advsimd_three_different() {
    fuzz(22, 0x0e20_0000, Q | U | SIZE | RM | (0xf << 12) | RD_RN);
    fuzz(23, 0x5e20_0000, U | SIZE | RM | (0xf << 12) | RD_RN);
}

#[test]
fn advsimd_three_same_extra_integer() {
    let keep = |i: u32| field(i, 14, 11) < 0b1000;
    fuzz_where(
        24,
        0x0e00_8400,
        Q | U | SIZE | RM | (0xf << 11) | RD_RN,
        keep,
    );
    fuzz_where(25, 0x5e00_8400, U | SIZE | RM | (0xf << 11) | RD_RN, keep);
}

#[test]
fn advsimd_two_register_misc_integer() {
    fuzz_where(
        26,
        0x0e20_0800,
        Q | U | SIZE | (0x1f << 12) | RD_RN,
        integer_two_misc,
    );
    fuzz_where(
        27,
        0x5e20_0800,
        U | SIZE | (0x1f << 12) | RD_RN,
        integer_two_misc,
    );
}

#[test]
fn advsimd_across_lanes_and_pairwise_integer() {
    let keep = |i: u32| !matches!(field(i, 16, 12), 0b01100 | 0b01101 | 0b01111);
    fuzz_where(28, 0x0e30_0800, Q | U | SIZE | (0x1f << 12) | RD_RN, keep);
    fuzz_where(29, 0x5e30_0800, U | SIZE | (0x1f << 12) | RD_RN, keep);
}

#[test]
fn advsimd_shift_by_immediate() {
    fuzz(30, 0x0f00_0400, Q | U | (0x7f << 16) | (0x1f << 11) | RD_RN);
    fuzz(31, 0x5f00_0400, U | (0x7f << 16) | (0x1f << 11) | RD_RN);
}

#[test]
fn advsimd_modified_immediate() {
    fuzz(32, 0x0f00_0400, Q | U | (7 << 16) | (0x1f << 11) | RD_RN);
}

#[test]
fn advsimd_copy() {
    fuzz(33, 0x0e00_0400, Q | U | RM | (0xf << 11) | RD_RN);
    fuzz(34, 0x5e00_0400, U | RM | (0xf << 11) | RD_RN);
}

#[test]
fn advsimd_permute_extract_table() {
    fuzz(35, 0x0e00_0800, Q | SIZE | RM | (7 << 12) | RD_RN);
    fuzz(36, 0x2e00_0000, Q | SIZE | RM | (0xf << 11) | RD_RN);
    fuzz(37, 0x0e00_0000, Q | SIZE | RM | (7 << 12) | RD_RN);
}

#[test]
fn advsimd_indexed_integer() {
    let random = U | SIZE | (3 << 20) | (0xf << 16) | (0xf << 12) | (1 << 11) | RD_RN;
    fuzz_where(38, 0x0f00_0000, Q | random, integer_indexed);
    fuzz_where(39, 0x5f00_0000, random, integer_indexed);
}

#[test]
fn switch2_sdk_integer_encodings_execute() {
    let mut h = Harness::new(40);
    for instruction in [0x4ea0_5c42, 0x0f41_3044] {
        for _ in 0..200 {
            let ctx = h.random_context();
            h.check(instruction, &ctx);
            let mut ours = ctx.clone();
            assert_eq!(
                execute(&mut ours, instruction),
                Some(Ok(())),
                "{instruction:#010x}"
            );
        }
    }
}

#[test]
fn advsimd_three_same_fp() {
    let random = U | SIZE | RM | (7 << 11) | RD_RN;
    fuzz(41, 0x0e20_c400, Q | random);
    fuzz(42, 0x5e20_c400, random);
}

#[test]
fn advsimd_three_same_fp16() {
    let random = U | (1 << 23) | RM | (7 << 11) | RD_RN;
    fuzz(43, 0x0e40_0400, Q | random);
    fuzz(44, 0x5e40_0400, random);
}

#[test]
fn advsimd_two_register_misc_fp() {
    let fp = |i: u32| !integer_two_misc(i);
    fuzz_where(45, 0x0e20_0800, Q | U | SIZE | (0x1f << 12) | RD_RN, fp);
    fuzz_where(46, 0x5e20_0800, U | SIZE | (0x1f << 12) | RD_RN, fp);
    fuzz(47, 0x0e78_0800, Q | U | (1 << 23) | (0x1f << 12) | RD_RN);
    fuzz(48, 0x5e78_0800, U | (1 << 23) | (0x1f << 12) | RD_RN);
}

#[test]
fn advsimd_across_lanes_and_pairwise_fp() {
    let fp = |i: u32| matches!(field(i, 16, 12), 0b01100 | 0b01101 | 0b01111);
    fuzz_where(49, 0x0e30_0800, Q | U | SIZE | (0x1f << 12) | RD_RN, fp);
    fuzz_where(50, 0x5e30_0800, U | SIZE | (0x1f << 12) | RD_RN, fp);
}

#[test]
fn advsimd_indexed_fp() {
    let random = U | SIZE | (3 << 20) | (0xf << 16) | (0xf << 12) | (1 << 11) | RD_RN;
    let fp = |i: u32| !integer_indexed(i);
    fuzz_where(51, 0x0f00_0000, Q | random, fp);
    fuzz_where(52, 0x5f00_0000, random, fp);
}

#[test]
fn advsimd_complex() {
    fuzz(53, 0x2e00_c400, Q | SIZE | RM | (7 << 11) | RD_RN);
}

#[test]
fn switch2_sdk_fp16_by_element_executes() {
    let mut h = Harness::new(54);
    for _ in 0..200 {
        let ctx = h.random_context();
        h.check(0x5f11_9f9c, &ctx);
        let mut ours = ctx.clone();
        assert_eq!(execute(&mut ours, 0x5f11_9f9c), Some(Ok(())));
    }
}
