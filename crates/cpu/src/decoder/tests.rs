use super::*;

const fn r(n: u8) -> Reg {
    Reg(n)
}

#[test]
fn decodes_reference_encodings() {
    use Inst::*;
    let cases: &[(u32, Inst)] = &[
        (
            0xb0000023,
            Adrp {
                rd: r(3),
                offset: 0x5000,
            },
        ),
        (
            0x10ffffc1,
            Adr {
                rd: r(1),
                offset: -8,
            },
        ),
        (
            0x91448c5f,
            AddSubImm {
                sf: true,
                sub: false,
                setflags: false,
                rd: Reg::SP,
                rn: r(2),
                imm: 0x123000,
            },
        ),
        (
            0x71000483,
            AddSubImm {
                sf: false,
                sub: true,
                setflags: true,
                rd: r(3),
                rn: r(4),
                imm: 1,
            },
        ),
        (
            0x92089c20,
            LogicalImm {
                sf: true,
                op: LogicOp::And,
                rd: r(0),
                rn: r(1),
                imm: 0xff00_ff00_ff00_ff00,
            },
        ),
        (
            0x320003e5,
            LogicalImm {
                sf: false,
                op: LogicOp::Orr,
                rd: r(5),
                rn: Reg::ZR,
                imm: 1,
            },
        ),
        (
            0xf2f7dde9,
            MovWide {
                sf: true,
                op: MovOp::K,
                rd: r(9),
                shift: 48,
                imm: 0xbeef,
            },
        ),
        (
            0x12800001,
            MovWide {
                sf: false,
                op: MovOp::N,
                rd: r(1),
                shift: 0,
                imm: 0,
            },
        ),
        (
            0x93431c41,
            Bitfield {
                sf: true,
                op: BitfieldOp::Sbfm,
                rd: r(1),
                rn: r(2),
                immr: 3,
                imms: 7,
            },
        ),
        (
            0x53042c83,
            Bitfield {
                sf: false,
                op: BitfieldOp::Ubfm,
                rd: r(3),
                rn: r(4),
                immr: 4,
                imms: 11,
            },
        ),
        (
            0x93c34441,
            Extr {
                sf: true,
                rd: r(1),
                rn: r(2),
                rm: r(3),
                lsb: 17,
            },
        ),
        (
            0xab831441,
            AddSubShift {
                sf: true,
                sub: false,
                setflags: true,
                rd: r(1),
                rn: r(2),
                rm: r(3),
                shift: ShiftType::Asr,
                amount: 5,
            },
        ),
        (
            0xcb23cbe1,
            AddSubExt {
                sf: true,
                sub: true,
                setflags: false,
                rd: r(1),
                rn: Reg::SP,
                rm: r(3),
                extend: Extend::Sxtw,
                amount: 2,
            },
        ),
        (
            0x7a030041,
            AddSubCarry {
                sf: false,
                sub: true,
                setflags: true,
                rd: r(1),
                rn: r(2),
                rm: r(3),
            },
        ),
        (
            0xeae6fca4,
            LogicalShift {
                sf: true,
                op: LogicOp::Ands,
                invert: true,
                rd: r(4),
                rn: r(5),
                rm: r(6),
                shift: ShiftType::Ror,
                amount: 63,
            },
        ),
        (
            0xfa431825,
            CondCompare {
                sf: true,
                sub: true,
                rn: r(1),
                rhs: 3,
                imm: true,
                nzcv: 5,
                cond: 1,
            },
        ),
        (
            0x5a83b441,
            CondSelect {
                sf: false,
                op: CselOp::Csneg,
                rd: r(1),
                rn: r(2),
                rm: r(3),
                cond: 11,
            },
        ),
        (
            0x5ac00841,
            Dp1 {
                sf: false,
                op: Dp1Op::Rev32,
                rd: r(1),
                rn: r(2),
            },
        ),
        (
            0xdac00c41,
            Dp1 {
                sf: true,
                op: Dp1Op::Rev64,
                rd: r(1),
                rn: r(2),
            },
        ),
        (
            0xdac01483,
            Dp1 {
                sf: true,
                op: Dp1Op::Cls,
                rd: r(3),
                rn: r(4),
            },
        ),
        (
            0x9ac30c41,
            Dp2 {
                sf: true,
                op: Dp2Op::Sdiv,
                rd: r(1),
                rn: r(2),
                rm: r(3),
            },
        ),
        (
            0x9ac35c41,
            Crc32 {
                castagnoli: true,
                size: 3,
                rd: r(1),
                rn: r(2),
                rm: r(3),
            },
        ),
        (
            0x9bc37c41,
            Dp3 {
                sf: true,
                op: Dp3Op::Umulh,
                rd: r(1),
                rn: r(2),
                rm: r(3),
                ra: Reg::ZR,
            },
        ),
        (
            0x9b239041,
            Dp3 {
                sf: true,
                op: Dp3Op::Smsubl,
                rd: r(1),
                rn: r(2),
                rm: r(3),
                ra: r(4),
            },
        ),
        (0x17ffffff, B { offset: -4 }),
        (0x94000040, Bl { offset: 256 }),
        (
            0x54ffff8a,
            BCond {
                cond: 10,
                offset: -16,
            },
        ),
        (
            0x35000043,
            Cbz {
                sf: false,
                nonzero: true,
                rt: r(3),
                offset: 8,
            },
        ),
        (
            0xb60fff05,
            Tbz {
                nonzero: false,
                rt: r(5),
                bit: 33,
                offset: -32,
            },
        ),
        (0xd63f0100, Blr { rn: r(8) }),
        (0xd65f03c0, Ret { rn: Reg::LR }),
        (0xd65f0fff, Ret { rn: Reg::LR }),
        (0xd71f0822, Br { rn: r(1) }),
        (0xd63f087f, Blr { rn: r(3) }),
        (0xd4000441, Svc { imm: 0x22 }),
        (0xd42000e0, Brk { imm: 7 }),
        (0xd503203f, Hint { op: HintOp::Yield }),
        (0xd503237f, Nop),
        (0xd5033bbf, Barrier { op: BarrierOp::Dmb }),
        (0xd5033fdf, Barrier { op: BarrierOp::Isb }),
        (0xd5033f5f, Clrex),
        (
            0xd53bd061,
            Mrs {
                rt: r(1),
                reg: SysReg::TpidrroEl0,
            },
        ),
        (
            0xd51b4402,
            Msr {
                rt: r(2),
                reg: SysReg::Fpcr,
            },
        ),
        (
            0xd53be023,
            Mrs {
                rt: r(3),
                reg: SysReg::CntpctEl0,
            },
        ),
        (0xd50b7424, DcZva { rt: r(4) }),
        (0xd50b7525, IcIvau { rt: r(5) }),
        (0xd50b7e26, Nop),
        (0xd500401f, Flag { op: FlagOp::Cfinv }),
        (
            0xba01842a,
            Rmif {
                rn: r(1),
                shift: 3,
                mask: 10,
            },
        ),
        (
            0x3a00486d,
            Setf {
                rn: r(3),
                sixteen: true,
            },
        ),
        (0xdac10041, Nop),
        (0xdac143e3, Nop),
        (
            0x9adf3041,
            Pacga {
                rd: r(1),
                rn: r(2),
                rm: Reg::SP,
            },
        ),
        (0xdac133e4, Nop),
        (
            0xf9400441,
            Load {
                size: 3,
                ext: LoadExt::Zero,
                acquire: false,
                rt: r(1),
                rn: r(2),
                addr: Addr::Offset(8),
            },
        ),
        (
            0x38dfffe1,
            Load {
                size: 0,
                ext: LoadExt::Sign32,
                acquire: false,
                rt: r(1),
                rn: Reg::SP,
                addr: Addr::Pre(-1),
            },
        ),
        (
            0xb8804441,
            Load {
                size: 2,
                ext: LoadExt::Sign64,
                acquire: false,
                rt: r(1),
                rn: r(2),
                addr: Addr::Post(4),
            },
        ),
        (
            0x78237841,
            Store {
                size: 1,
                release: false,
                rt: r(1),
                rn: r(2),
                addr: Addr::Reg {
                    rm: r(3),
                    extend: Extend::Uxtx,
                    shift: 1,
                },
            },
        ),
        (
            0x3ce3d841,
            LoadVec {
                size: 4,
                vt: 1,
                rn: r(2),
                addr: Addr::Reg {
                    rm: r(3),
                    extend: Extend::Sxtw,
                    shift: 4,
                },
            },
        ),
        (
            0x3d3ffc62,
            StoreVec {
                size: 0,
                vt: 2,
                rn: r(3),
                addr: Addr::Offset(4095),
            },
        ),
        (
            0xf85f8841,
            Load {
                size: 3,
                ext: LoadExt::Zero,
                acquire: false,
                rt: r(1),
                rn: r(2),
                addr: Addr::Offset(-8),
            },
        ),
        (0xf9800020, Nop),
        (
            0xfc5fd0c5,
            LoadVec {
                size: 3,
                vt: 5,
                rn: r(6),
                addr: Addr::Offset(-3),
            },
        ),
        (
            0x18ffffe1,
            LoadLiteral {
                size: 2,
                ext: LoadExt::Zero,
                rt: r(1),
                offset: -4,
            },
        ),
        (
            0x9c000042,
            LoadLiteralVec {
                size: 4,
                vt: 2,
                offset: 8,
            },
        ),
        (
            0x98000083,
            LoadLiteral {
                size: 2,
                ext: LoadExt::Sign64,
                rt: r(3),
                offset: 16,
            },
        ),
        (
            0xa9ff0be1,
            LoadPair {
                size: 3,
                ext: LoadExt::Zero,
                rt: r(1),
                rt2: r(2),
                rn: Reg::SP,
                addr: Addr::Pre(-16),
            },
        ),
        (
            0x28810861,
            StorePair {
                size: 2,
                rt: r(1),
                rt2: r(2),
                rn: r(3),
                addr: Addr::Post(8),
            },
        ),
        (
            0x697f0861,
            LoadPair {
                size: 2,
                ext: LoadExt::Sign64,
                rt: r(1),
                rt2: r(2),
                rn: r(3),
                addr: Addr::Offset(-8),
            },
        ),
        (
            0xac410861,
            LoadPairVec {
                size: 4,
                vt: 1,
                vt2: 2,
                rn: r(3),
                addr: Addr::Offset(32),
            },
        ),
        (
            0x69010861,
            StorePair {
                size: 3,
                rt: r(1),
                rt2: r(2),
                rn: r(3),
                addr: Addr::Offset(32),
            },
        ),
        (
            0x885f7c41,
            LoadExclusive {
                size: 2,
                pair: false,
                acquire: false,
                rt: r(1),
                rt2: Reg::ZR,
                rn: r(2),
            },
        ),
        (
            0xc803fca4,
            StoreExclusive {
                size: 3,
                pair: false,
                release: true,
                rs: r(3),
                rt: r(4),
                rt2: Reg::ZR,
                rn: r(5),
            },
        ),
        (
            0xc87f8861,
            LoadExclusive {
                size: 3,
                pair: true,
                acquire: true,
                rt: r(1),
                rt2: r(2),
                rn: r(3),
            },
        ),
        (
            0x88210c82,
            StoreExclusive {
                size: 2,
                pair: true,
                release: false,
                rs: r(1),
                rt: r(2),
                rt2: r(3),
                rn: r(4),
            },
        ),
        (
            0xc8dfffe1,
            Load {
                size: 3,
                ext: LoadExt::Zero,
                acquire: true,
                rt: r(1),
                rn: Reg::SP,
                addr: Addr::Offset(0),
            },
        ),
        (
            0x089ffc41,
            Store {
                size: 0,
                release: true,
                rt: r(1),
                rn: r(2),
                addr: Addr::Offset(0),
            },
        ),
        (
            0xc8e1fc62,
            Cas {
                size: 3,
                acquire: true,
                release: true,
                rs: r(1),
                rt: r(2),
                rn: r(3),
            },
        ),
        (
            0x08627cc4,
            Casp {
                size: 2,
                acquire: true,
                release: false,
                rs: 2,
                rt: 4,
                rn: r(6),
            },
        ),
        (
            0xb8e10062,
            AtomicRmw {
                op: AtomicOp::Add,
                size: 2,
                acquire: true,
                release: true,
                rs: r(1),
                rt: r(2),
                rn: r(3),
            },
        ),
        (
            0xf821705f,
            AtomicRmw {
                op: AtomicOp::Umin,
                size: 3,
                acquire: false,
                release: false,
                rs: r(1),
                rt: Reg::ZR,
                rn: r(2),
            },
        ),
        (
            0x38218062,
            AtomicRmw {
                op: AtomicOp::Swp,
                size: 0,
                acquire: false,
                release: false,
                rs: r(1),
                rt: r(2),
                rn: r(3),
            },
        ),
        (
            0x78bfc041,
            Load {
                size: 1,
                ext: LoadExt::Zero,
                acquire: true,
                rt: r(1),
                rn: r(2),
                addr: Addr::Offset(0),
            },
        ),
        (
            0x999fb041,
            Load {
                size: 2,
                ext: LoadExt::Sign64,
                acquire: true,
                rt: r(1),
                rn: r(2),
                addr: Addr::Offset(-5),
            },
        ),
        (
            0x99003041,
            Store {
                size: 2,
                release: true,
                rt: r(1),
                rn: r(2),
                addr: Addr::Offset(3),
            },
        ),
        (
            0xf8600441,
            Load {
                size: 3,
                ext: LoadExt::Zero,
                acquire: false,
                rt: r(1),
                rn: r(2),
                addr: Addr::Offset(-4096),
            },
        ),
        (
            0xf8a01c41,
            Load {
                size: 3,
                ext: LoadExt::Zero,
                acquire: false,
                rt: r(1),
                rn: r(2),
                addr: Addr::Pre(8),
            },
        ),
        (
            0x9adf13e1,
            AddSubImm {
                sf: true,
                sub: false,
                setflags: false,
                rd: r(1),
                rn: Reg::SP,
                imm: 0,
            },
        ),
        (
            0x9ac31441,
            LogicalImm {
                sf: true,
                op: LogicOp::Orr,
                rd: r(1),
                rn: r(3),
                imm: 1,
            },
        ),
        (
            0x91830be1,
            AddSubImm {
                sf: true,
                sub: false,
                setflags: false,
                rd: r(1),
                rn: Reg::SP,
                imm: 48,
            },
        ),
        (
            0xd9201841,
            StoreTag {
                zero_granules: 0,
                rn: r(2),
                addr: Addr::Offset(16),
            },
        ),
        (
            0xd97fe441,
            StoreTag {
                zero_granules: 1,
                rn: r(2),
                addr: Addr::Post(-32),
            },
        ),
        (
            0xd9a02c41,
            StoreTag {
                zero_granules: 0,
                rn: r(2),
                addr: Addr::Pre(32),
            },
        ),
        (0xd9600041, Nop),
        (
            0x4cdfa061,
            VecMulti {
                load: true,
                q: true,
                size: 0,
                rpt: 2,
                selem: 1,
                vt: 1,
                rn: r(3),
                post: PostIndex::Imm(32),
            },
        ),
        (
            0x4c8608a1,
            VecMulti {
                load: false,
                q: true,
                size: 2,
                rpt: 1,
                selem: 4,
                vt: 1,
                rn: r(5),
                post: PostIndex::Reg(r(6)),
            },
        ),
        (
            0x4d408420,
            VecSingle {
                load: true,
                size: 3,
                selem: 1,
                index: 1,
                vt: 0,
                rn: r(1),
                post: PostIndex::None,
            },
        ),
        (
            0x4d9f6841,
            VecSingle {
                load: false,
                size: 1,
                selem: 3,
                index: 5,
                vt: 1,
                rn: r(2),
                post: PostIndex::Imm(6),
            },
        ),
        (
            0x4d60c461,
            VecReplicate {
                q: true,
                size: 1,
                selem: 2,
                vt: 1,
                rn: r(3),
                post: PostIndex::None,
            },
        ),
        (
            0x0c407c20,
            VecMulti {
                load: true,
                q: false,
                size: 3,
                rpt: 1,
                selem: 1,
                vt: 0,
                rn: r(1),
                post: PostIndex::None,
            },
        ),
        (0x1e222820, SimdFp { word: 0x1e222820 }),
        (0x00000000, Undefined { word: 0 }),
    ];
    for &(word, expected) in cases {
        assert_eq!(decode(word), expected, "{word:#010x}");
    }
}

#[test]
fn rejects_reserved_fields() {
    for word in [
        0x12400000u32,        // logical immediate, 32-bit with N set
        0x32800000,           // move wide with opc 01
        0x52c00000,           // MOVZ 32-bit with hw = 2
        0x8b000000 | 3 << 22, // add (shifted register) with shift 11
        0xd65e0bff,           // RETAA with a non-31 op2
        0x08607cc5,           // CASP with an odd Rt
        0xd51bd060,           // MSR TPIDRRO_EL0, read-only at EL0
    ] {
        assert_eq!(decode(word), Inst::Undefined { word }, "{word:#010x}");
    }
}

#[test]
fn bit_masks_match_reference_values() {
    assert_eq!(decode_bit_masks(true, 0, 0, true), Some((1, 1)));
    assert_eq!(
        decode_bit_masks(false, 0b111100, 0, true),
        Some((0x5555_5555_5555_5555, 0x5555_5555_5555_5555))
    );
    assert_eq!(decode_bit_masks(true, 0b111111, 0, true), None);
    let (wmask, tmask) = decode_bit_masks(true, 7, 3, false).unwrap();
    assert_eq!(wmask, 0xe000_0000_0000_001f);
    assert_eq!(tmask, 0x1f);
}
