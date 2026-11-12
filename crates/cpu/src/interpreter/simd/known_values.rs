//! Known-value tests: Arm ARM behaviour stated directly, including the cases where the Unicorn
//! comparison is skipped because QEMU 5.0 deviates from the architecture.

use super::execute;
use super::lanes::QC;
use super::softfloat::{DZC, IDC, IOC, IXC, OFC, UFC};
use crate::ThreadContext;

const FZ: u32 = 1 << 24;
const DN: u32 = 1 << 25;
const FZ16: u32 = 1 << 19;
const RMODE_ZERO: u32 = 3 << 22;

fn run(instruction: u32, fpcr: u32, setup: impl FnOnce(&mut ThreadContext)) -> ThreadContext {
    let mut ctx = ThreadContext {
        fpcr,
        ..ThreadContext::default()
    };
    setup(&mut ctx);
    assert_eq!(
        execute(&mut ctx, instruction),
        Some(Ok(())),
        "{instruction:#010x}"
    );
    ctx
}

fn undefined(instruction: u32) -> bool {
    let mut ctx = ThreadContext::default();
    matches!(execute(&mut ctx, instruction), Some(Err(_)))
}

fn s(v: f32) -> u128 {
    v.to_bits() as u128
}

const FADD_S0_S1_S2: u32 = 0x1e22_2820;
const FMADD_S0_S1_S2_S3: u32 = 0x1f02_0c20;
const FMSUB_S0_S1_S2_S3: u32 = 0x1f02_8c20;
const FNMUL_S0_S1_S2: u32 = 0x1e22_8820;

#[test]
fn instructions_outside_the_group_are_not_claimed() {
    let mut ctx = ThreadContext::default();
    assert_eq!(execute(&mut ctx, 0x8b02_0020), None);
    assert_eq!(execute(&mut ctx, 0xd503_201f), None);
}

#[test]
fn nan_selection_and_signs() {
    let snan = 0x7f80_0001u128;
    let qnan = 0x7fc0_0002u128;
    let ctx = run(FADD_S0_S1_S2, 0, |c| (c.v[1], c.v[2]) = (qnan, snan));
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (0x7fc0_0001, IOC),
        "a signalling NaN wins over a quiet one"
    );

    let ctx = run(FMSUB_S0_S1_S2_S3, 0, |c| {
        (c.v[1], c.v[2], c.v[3]) = (qnan, s(1.0), s(1.0))
    });
    assert_eq!(
        ctx.v[0], 0xffc0_0002,
        "FMSUB negates op1 before NaN propagation"
    );

    let ctx = run(FNMUL_S0_S1_S2, DN, |c| (c.v[1], c.v[2]) = (snan, s(1.0)));
    assert_eq!(ctx.v[0], 0xffc0_0000, "FNMUL negates even the default NaN");

    let ctx = run(FMADD_S0_S1_S2_S3, 0, |c| {
        (c.v[1], c.v[2], c.v[3]) = (s(0.0), s(f32::INFINITY), qnan)
    });
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (0x7fc0_0000, IOC),
        "QNaN addend with 0 x inf is the default NaN"
    );
}

#[test]
fn flush_to_zero() {
    let tiny = 1u128;
    let ctx = run(FADD_S0_S1_S2, FZ, |c| (c.v[1], c.v[2]) = (tiny, s(1.0)));
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (s(1.0), IDC),
        "denormal inputs flush and raise IDC"
    );

    let ctx = run(0x1e22_0820, FZ, |c| (c.v[1], c.v[2]) = (s(1e-30), s(1e-10)));
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (0, UFC),
        "flushed outputs raise UFC but not IXC"
    );

    let ctx = run(0x1ee2_2820, FZ16, |c| (c.v[1], c.v[2]) = (1, 0x3c00));
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (0x3c00, 0),
        "FZ16 flushes half inputs without IDC"
    );

    let ctx = run(0x1e22_0820, 0, |c| (c.v[1], c.v[2]) = (s(1e-30), s(1e-10)));
    assert_eq!(ctx.fpsr, UFC | IXC, "tininess is detected before rounding");
}

#[test]
fn division_and_overflow_flags() {
    let ctx = run(0x1e22_1820, 0, |c| (c.v[1], c.v[2]) = (s(1.0), s(0.0)));
    assert_eq!((ctx.v[0], ctx.fpsr), (s(f32::INFINITY), DZC));
    let ctx = run(0x1e22_0820, RMODE_ZERO, |c| {
        (c.v[1], c.v[2]) = (s(f32::MAX), s(2.0))
    });
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (s(f32::MAX), OFC | IXC),
        "round to zero overflows to the max normal"
    );
}

#[test]
fn conversion_rounding_modes() {
    let convert =
        |instruction, value: f32| run(instruction, 0, |c| c.v[1] = s(value)).x[0] as u32 as i32;
    for (value, zero, nearest, away) in [
        (2.5, 2, 2, 3),
        (-2.5, -2, -2, -3),
        (3.5, 3, 4, 4),
        (-0.5, 0, 0, -1),
    ] {
        assert_eq!(convert(0x1e38_0020, value), zero, "FCVTZS {value}");
        assert_eq!(convert(0x1e20_0020, value), nearest, "FCVTNS {value}");
        assert_eq!(convert(0x1e24_0020, value), away, "FCVTAS {value}");
    }
    let ctx = run(0x1e38_0020, 0, |c| c.v[1] = s(f32::NAN));
    assert_eq!(
        (ctx.x[0], ctx.fpsr),
        (0, IOC),
        "NaN converts to zero with IOC"
    );
    let ctx = run(0x1e38_0020, 0, |c| c.v[1] = s(3e9));
    assert_eq!(
        (ctx.x[0], ctx.fpsr),
        (0x7fff_ffff, IOC),
        "out-of-range conversions saturate"
    );
}

#[test]
fn reciprocal_estimates() {
    assert_eq!(
        run(0x5ea1_d820, 0, |c| c.v[1] = s(1.0)).v[0],
        0x3f7f_8000,
        "FRECPE 1.0"
    );
    assert_eq!(
        run(0x5ea1_d820, 0, |c| c.v[1] = s(2.0)).v[0],
        0x3eff_8000,
        "FRECPE 2.0"
    );
    assert_eq!(
        run(0x7ea1_d820, 0, |c| c.v[1] = s(1.0)).v[0],
        0x3f7f_8000,
        "FRSQRTE 1.0"
    );
    assert_eq!(
        run(0x7ea1_d820, 0, |c| c.v[1] = s(4.0)).v[0],
        0x3eff_8000,
        "FRSQRTE 4.0"
    );
}

#[test]
fn saturation_sets_qc() {
    let ctx = run(0x4e22_0c20, 0, |c| (c.v[1], c.v[2]) = (0x7f, 0x01));
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (0x7f, QC),
        "SQADD .16B saturates at 127"
    );
    let ctx = run(0x4e22_0c20, 0, |c| {
        (c.v[1], c.v[2]) = (1, 1);
        c.fpsr = QC;
    });
    assert_eq!(ctx.fpsr, QC, "QC is cumulative");
}

#[test]
fn architecture_where_qemu_deviates() {
    let ctx = run(0x1e7e_0020, FZ, |c| c.v[1] = 1);
    assert_eq!(
        (ctx.x[0], ctx.pstate >> 28, ctx.fpsr),
        (0, 0, IDC),
        "FJCVTZS of a flushed positive denormal clears Z"
    );

    assert!(
        undefined(0x6f00_fc00),
        "modified immediate op=1 cmode=1111 o2=1 is unallocated"
    );
    assert!(
        undefined(0x4e70_f820),
        "half-precision FMAXV needs size<0> = 0"
    );
    assert!(
        undefined(0x5e70_f820),
        "half-precision FMAXP needs size<0> = 0"
    );

    let ctx = run(0x5f14_fc20, 0, |c| c.v[1] = 0xbc00);
    assert_eq!(
        ctx.v[0], 0xf000,
        "scalar FCVTZS to H clears the rest of the register"
    );

    let ctx = run(0x2f42_1020, 0, |c| {
        c.v[1] = 0x3555_3555_3555_3555_0000_0000_0000_0000
    });
    assert_eq!(
        (ctx.v[0], ctx.fpsr),
        (0, 0),
        "FCMLA .4H by element computes only the low four lanes"
    );
}
