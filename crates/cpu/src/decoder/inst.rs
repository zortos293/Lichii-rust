//! The decoded form of one A64 instruction.
//!
//! Decoding resolves everything that only depends on the instruction word: register 31 is
//! already `SP` or `ZR`, immediates are extended and scaled, aliases that share semantics are
//! folded (PAC and MTE forms become their plain equivalents), and unallocated encodings become
//! [`Inst::Undefined`]. Both the interpreter and the JIT consume this form, so neither needs to
//! look at instruction bits again.

/// A general-purpose register operand: `X0`..`X30`, [`Reg::ZR`] or [`Reg::SP`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Reg(pub u8);

impl Reg {
    /// Reads as zero, writes are discarded.
    pub const ZR: Self = Self(31);
    pub const SP: Self = Self(32);
    pub const LR: Self = Self(30);

    /// Register field value 31 means `ZR`.
    pub const fn zr(field: u32) -> Self {
        Self(field as u8)
    }

    /// Register field value 31 means `SP`.
    pub const fn sp(field: u32) -> Self {
        if field == 31 {
            Self::SP
        } else {
            Self(field as u8)
        }
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftType {
    Lsl,
    Lsr,
    Asr,
    Ror,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extend {
    Uxtb,
    Uxth,
    Uxtw,
    Uxtx,
    Sxtb,
    Sxth,
    Sxtw,
    Sxtx,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogicOp {
    And,
    Orr,
    Eor,
    /// `AND` that sets N and Z and clears C and V.
    Ands,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MovOp {
    /// `MOVN`: the inverted immediate.
    N,
    /// `MOVZ`: the immediate.
    Z,
    /// `MOVK`: the immediate replaces one halfword of the destination.
    K,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BitfieldOp {
    Sbfm,
    Bfm,
    Ubfm,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CselOp {
    Csel,
    Csinc,
    Csinv,
    Csneg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp1Op {
    Rbit,
    /// Reverse the bytes of each halfword.
    Rev16,
    /// Reverse the bytes of each word (`REV Wd` when 32-bit).
    Rev32,
    /// Reverse all eight bytes; 64-bit only.
    Rev64,
    Clz,
    Cls,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp2Op {
    Udiv,
    Sdiv,
    Lslv,
    Lsrv,
    Asrv,
    Rorv,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dp3Op {
    Madd,
    Msub,
    Smaddl,
    Smsubl,
    Umaddl,
    Umsubl,
    Smulh,
    Umulh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HintOp {
    Nop,
    Yield,
    Wfe,
    Wfi,
    Sev,
    Sevl,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarrierOp {
    Dsb,
    Dmb,
    Isb,
    Sb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlagOp {
    Cfinv,
    Xaflag,
    Axflag,
}

/// System registers reachable from EL0 on Horizon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SysReg {
    Nzcv,
    Fpcr,
    Fpsr,
    TpidrEl0,
    TpidrroEl0,
    CntfrqEl0,
    CntpctEl0,
    CntvctEl0,
    CtrEl0,
    DczidEl0,
}

/// How a single-register or pair access forms its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Addr {
    /// `[Xn, #imm]`.
    Offset(i32),
    /// `[Xn, #imm]!`.
    Pre(i32),
    /// `[Xn], #imm`.
    Post(i32),
    /// `[Xn, Rm{, extend #shift}]`.
    Reg { rm: Reg, extend: Extend, shift: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadExt {
    Zero,
    /// Sign-extend to 32 bits, then zero the upper half.
    Sign32,
    Sign64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Clr,
    Eor,
    Set,
    Smax,
    Smin,
    Umax,
    Umin,
    Swp,
}

/// Post-index update of an Advanced SIMD structure access.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PostIndex {
    None,
    /// Add the number of bytes transferred.
    Imm(u8),
    Reg(Reg),
}

/// One decoded A64 instruction. `sf` selects 64-bit (true) or 32-bit operation; `size` fields
/// are log2 of the access size in bytes. Branch and literal offsets are relative to the
/// instruction's own address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Inst {
    /// Unallocated or unsupported encoding.
    Undefined {
        word: u32,
    },
    /// SIMD and floating-point data processing, executed by `interpreter::simd`.
    SimdFp {
        word: u32,
    },
    Nop,

    Adr {
        rd: Reg,
        offset: i32,
    },
    Adrp {
        rd: Reg,
        offset: i64,
    },
    AddSubImm {
        sf: bool,
        sub: bool,
        setflags: bool,
        rd: Reg,
        rn: Reg,
        imm: u32,
    },
    LogicalImm {
        sf: bool,
        op: LogicOp,
        rd: Reg,
        rn: Reg,
        imm: u64,
    },
    MovWide {
        sf: bool,
        op: MovOp,
        rd: Reg,
        shift: u8,
        imm: u16,
    },
    Bitfield {
        sf: bool,
        op: BitfieldOp,
        rd: Reg,
        rn: Reg,
        immr: u8,
        imms: u8,
    },
    Extr {
        sf: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        lsb: u8,
    },

    AddSubShift {
        sf: bool,
        sub: bool,
        setflags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: ShiftType,
        amount: u8,
    },
    AddSubExt {
        sf: bool,
        sub: bool,
        setflags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        extend: Extend,
        amount: u8,
    },
    AddSubCarry {
        sf: bool,
        sub: bool,
        setflags: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    LogicalShift {
        sf: bool,
        op: LogicOp,
        invert: bool,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        shift: ShiftType,
        amount: u8,
    },
    /// `CCMP`/`CCMN`. `rhs` is a register index or, when `imm`, a 5-bit immediate.
    CondCompare {
        sf: bool,
        sub: bool,
        rn: Reg,
        rhs: u8,
        imm: bool,
        nzcv: u8,
        cond: u8,
    },
    CondSelect {
        sf: bool,
        op: CselOp,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        cond: u8,
    },
    Dp1 {
        sf: bool,
        op: Dp1Op,
        rd: Reg,
        rn: Reg,
    },
    Dp2 {
        sf: bool,
        op: Dp2Op,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    Dp3 {
        sf: bool,
        op: Dp3Op,
        rd: Reg,
        rn: Reg,
        rm: Reg,
        ra: Reg,
    },
    Crc32 {
        castagnoli: bool,
        size: u8,
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    /// `PACGA` with pointer authentication as the identity: `(Xn ^ Xm) & 0xffff_ffff_0000_0000`.
    Pacga {
        rd: Reg,
        rn: Reg,
        rm: Reg,
    },
    Flag {
        op: FlagOp,
    },
    Rmif {
        rn: Reg,
        shift: u8,
        mask: u8,
    },
    /// `SETF8` (`sixteen == false`) or `SETF16`.
    Setf {
        rn: Reg,
        sixteen: bool,
    },

    B {
        offset: i32,
    },
    Bl {
        offset: i32,
    },
    BCond {
        cond: u8,
        offset: i32,
    },
    Cbz {
        sf: bool,
        nonzero: bool,
        rt: Reg,
        offset: i32,
    },
    Tbz {
        nonzero: bool,
        rt: Reg,
        bit: u8,
        offset: i32,
    },
    Br {
        rn: Reg,
    },
    Blr {
        rn: Reg,
    },
    Ret {
        rn: Reg,
    },
    Svc {
        imm: u16,
    },
    Brk {
        imm: u16,
    },

    Hint {
        op: HintOp,
    },
    Barrier {
        op: BarrierOp,
    },
    Clrex,
    Mrs {
        rt: Reg,
        reg: SysReg,
    },
    Msr {
        rt: Reg,
        reg: SysReg,
    },
    DcZva {
        rt: Reg,
    },
    IcIvau {
        rt: Reg,
    },
    IcIallu,

    LoadLiteral {
        size: u8,
        ext: LoadExt,
        rt: Reg,
        offset: i32,
    },
    LoadLiteralVec {
        size: u8,
        vt: u8,
        offset: i32,
    },
    Load {
        size: u8,
        ext: LoadExt,
        acquire: bool,
        rt: Reg,
        rn: Reg,
        addr: Addr,
    },
    Store {
        size: u8,
        release: bool,
        rt: Reg,
        rn: Reg,
        addr: Addr,
    },
    LoadVec {
        size: u8,
        vt: u8,
        rn: Reg,
        addr: Addr,
    },
    StoreVec {
        size: u8,
        vt: u8,
        rn: Reg,
        addr: Addr,
    },
    /// `addr` is never [`Addr::Reg`].
    LoadPair {
        size: u8,
        ext: LoadExt,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
        addr: Addr,
    },
    StorePair {
        size: u8,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
        addr: Addr,
    },
    LoadPairVec {
        size: u8,
        vt: u8,
        vt2: u8,
        rn: Reg,
        addr: Addr,
    },
    StorePairVec {
        size: u8,
        vt: u8,
        vt2: u8,
        rn: Reg,
        addr: Addr,
    },
    /// MTE tag store (`STG`, `STZG`, `ST2G`, `STZ2G`) with tags ignored: zeroes
    /// `zero_granules` 16-byte granules at the address (none for the non-zeroing forms) and does
    /// the writeback.
    StoreTag {
        zero_granules: u8,
        rn: Reg,
        addr: Addr,
    },

    /// `LDXR`/`LDAXR`/`LDXP`/`LDAXP`; `size` is per register.
    LoadExclusive {
        size: u8,
        pair: bool,
        acquire: bool,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
    },
    /// `STXR`/`STLXR`/`STXP`/`STLXP`; writes 0 to `rs` on success.
    StoreExclusive {
        size: u8,
        pair: bool,
        release: bool,
        rs: Reg,
        rt: Reg,
        rt2: Reg,
        rn: Reg,
    },
    /// LSE `LD<op>`/`ST<op>`/`SWP`: `rt` receives the old value, `rs` is the operand.
    AtomicRmw {
        op: AtomicOp,
        size: u8,
        acquire: bool,
        release: bool,
        rs: Reg,
        rt: Reg,
        rn: Reg,
    },
    Cas {
        size: u8,
        acquire: bool,
        release: bool,
        rs: Reg,
        rt: Reg,
        rn: Reg,
    },
    /// `rs`/`rt` name the even register of each pair; `size` is per register (2 or 3).
    Casp {
        size: u8,
        acquire: bool,
        release: bool,
        rs: u8,
        rt: u8,
        rn: Reg,
    },

    /// `LD1`..`LD4`/`ST1`..`ST4` (multiple structures): `rpt` register groups of `selem`
    /// interleaved registers, elements of `size`, 8 or 16 bytes per register (`q`).
    VecMulti {
        load: bool,
        q: bool,
        size: u8,
        rpt: u8,
        selem: u8,
        vt: u8,
        rn: Reg,
        post: PostIndex,
    },
    /// `LD1`..`LD4`/`ST1`..`ST4` (single structure) at lane `index`.
    VecSingle {
        load: bool,
        size: u8,
        selem: u8,
        index: u8,
        vt: u8,
        rn: Reg,
        post: PostIndex,
    },
    /// `LD1R`..`LD4R`.
    VecReplicate {
        q: bool,
        size: u8,
        selem: u8,
        vt: u8,
        rn: Reg,
        post: PostIndex,
    },
}

impl Inst {
    /// Whether execution cannot fall through to the next instruction of the same block: control
    /// flow, exceptions, undefined encodings, and operations that invalidate translated code.
    pub const fn ends_block(&self) -> bool {
        matches!(
            self,
            Self::Undefined { .. }
                | Self::B { .. }
                | Self::Bl { .. }
                | Self::BCond { .. }
                | Self::Cbz { .. }
                | Self::Tbz { .. }
                | Self::Br { .. }
                | Self::Blr { .. }
                | Self::Ret { .. }
                | Self::Svc { .. }
                | Self::Brk { .. }
                | Self::IcIvau { .. }
                | Self::IcIallu
                | Self::Barrier { op: BarrierOp::Isb }
        )
    }
}

const _: () = assert!(std::mem::size_of::<Inst>() <= 16);
