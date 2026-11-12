//! The A64 encoding table and its handlers.

use std::sync::OnceLock;

use super::inst::*;
use super::{decode_bit_masks, ones_u32};

struct Entry {
    mask: u32,
    value: u32,
    decode: fn(u32) -> Option<Inst>,
}

const fn pattern(bits: &str) -> (u32, u32) {
    let bytes = bits.as_bytes();
    assert!(bytes.len() == 32, "patterns are 32 characters");
    let mut mask = 0;
    let mut value = 0;
    let mut i = 0;
    while i < 32 {
        let bit = 1u32 << (31 - i);
        match bytes[i] {
            b'0' => mask |= bit,
            b'1' => {
                mask |= bit;
                value |= bit;
            }
            _ => {}
        }
        i += 1;
    }
    (mask, value)
}

macro_rules! table {
    ($($bits:literal => $handler:ident,)*) => {
        const ENTRIES: &[Entry] = &[$({
            let (mask, value) = pattern($bits);
            Entry { mask, value, decode: $handler }
        }),*];
    };
}

table! {
    "xxx10000xxxxxxxxxxxxxxxxxxxxxxxx" => pc_rel,
    "xxx100010xxxxxxxxxxxxxxxxxxxxxxx" => add_sub_imm,
    "1x01000110xxxxxxxxxxxxxxxxxxxxxx" => add_sub_tag,
    "xxx100100xxxxxxxxxxxxxxxxxxxxxxx" => logical_imm,
    "xxx100101xxxxxxxxxxxxxxxxxxxxxxx" => move_wide,
    "xxx100110xxxxxxxxxxxxxxxxxxxxxxx" => bitfield,
    "xxx100111xxxxxxxxxxxxxxxxxxxxxxx" => extract,

    "x00101xxxxxxxxxxxxxxxxxxxxxxxxxx" => branch_imm,
    "x011010xxxxxxxxxxxxxxxxxxxxxxxxx" => compare_branch,
    "x011011xxxxxxxxxxxxxxxxxxxxxxxxx" => test_branch,
    "01010100xxxxxxxxxxxxxxxxxxxxxxxx" => cond_branch,
    "11010100xxxxxxxxxxxxxxxxxxxxxxxx" => exception,
    "11010101000000110010xxxxxxx11111" => hint,
    "11010101000000110011xxxxxxx11111" => barrier,
    "1101010100000xxx0100xxxxxxx11111" => pstate,
    "1101010100001xxxxxxxxxxxxxxxxxxx" => sys,
    "1101010100x1xxxxxxxxxxxxxxxxxxxx" => sys_reg,
    "1101011xxxxxxxxxxxxxxxxxxxxxxxxx" => branch_reg,

    "xx001000xxxxxxxxxxxxxxxxxxxxxxxx" => exclusive,
    "xx011001xx0xxxxxxxxx00xxxxxxxxxx" => rcpc_unscaled,
    "11011001xx1xxxxxxxxxxxxxxxxxxxxx" => tag_store,
    "xx011x00xxxxxxxxxxxxxxxxxxxxxxxx" => load_literal,
    "xx101x0xxxxxxxxxxxxxxxxxxxxxxxxx" => pair,
    "xx111x01xxxxxxxxxxxxxxxxxxxxxxxx" => single_unsigned,
    "xx111x00xx0xxxxxxxxxxxxxxxxxxxxx" => single_imm9,
    "xx111x00xx1xxxxxxxxx10xxxxxxxxxx" => single_reg,
    "xx111000xx1xxxxxxxxx00xxxxxxxxxx" => atomic,
    "11111000xx1xxxxxxxxxx1xxxxxxxxxx" => ldra,
    "0x0011000x000000xxxxxxxxxxxxxxxx" => vec_multi,
    "0x0011001x0xxxxxxxxxxxxxxxxxxxxx" => vec_multi,
    "0x0011010xx00000xxxxxxxxxxxxxxxx" => vec_single,
    "0x0011011xxxxxxxxxxxxxxxxxxxxxxx" => vec_single,

    "x0x11010110xxxxxxxxxxxxxxxxxxxxx" => dp2,
    "x1x11010110xxxxxxxxxxxxxxxxxxxxx" => dp1,
    "xxx01010xxxxxxxxxxxxxxxxxxxxxxxx" => logical_shift,
    "xxx01011xx0xxxxxxxxxxxxxxxxxxxxx" => add_sub_shift,
    "xxx01011xx1xxxxxxxxxxxxxxxxxxxxx" => add_sub_ext,
    "xxx11010000xxxxx000000xxxxxxxxxx" => add_sub_carry,
    "10111010000xxxxxx00001xxxxx0xxxx" => rmif,
    "00111010000000000x0010xxxxx01101" => setf,
    "xx111010010xxxxxxxxxx0xxxxx0xxxx" => cond_compare,
    "xx011010100xxxxxxxxxxxxxxxxxxxxx" => cond_select,
    "xxx11011xxxxxxxxxxxxxxxxxxxxxxxx" => dp3,
}

/// The candidates for one value of [`bucket`], most specific first.
type Buckets = Box<[Box<[u8]>]>;

/// Buckets on bits 10..14 and 22..30, which separate most encoding classes.
const fn bucket(word: u32) -> usize {
    (((word >> 10) & 0x00f) | ((word >> 18) & 0xff0)) as usize
}

fn buckets() -> &'static Buckets {
    static BUCKETS: OnceLock<Buckets> = OnceLock::new();
    BUCKETS.get_or_init(|| {
        let mut order: Vec<u8> = (0..ENTRIES.len() as u8).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(ENTRIES[i as usize].mask.count_ones()));
        (0..0x1000)
            .map(|b| {
                order
                    .iter()
                    .copied()
                    .filter(|&i| {
                        let e = &ENTRIES[i as usize];
                        b & bucket(e.mask) == bucket(e.value)
                    })
                    .collect()
            })
            .collect()
    })
}

pub(super) fn lookup(word: u32) -> Option<Inst> {
    buckets()[bucket(word)].iter().find_map(|&i| {
        let e = &ENTRIES[i as usize];
        if word & e.mask == e.value {
            (e.decode)(word)
        } else {
            None
        }
    })
}

#[inline]
const fn field(w: u32, lo: u32, len: u32) -> u32 {
    (w >> lo) & ones_u32(len)
}

#[inline]
const fn bit(w: u32, n: u32) -> bool {
    (w >> n) & 1 != 0
}

#[inline]
const fn sext(value: u32, bits: u32) -> i32 {
    ((value << (32 - bits)) as i32) >> (32 - bits)
}

const fn rd(w: u32) -> u32 {
    field(w, 0, 5)
}

const fn rn(w: u32) -> u32 {
    field(w, 5, 5)
}

const fn rm(w: u32) -> u32 {
    field(w, 16, 5)
}

const fn ra(w: u32) -> u32 {
    field(w, 10, 5)
}

const fn shift_type(v: u32) -> ShiftType {
    match v & 3 {
        0 => ShiftType::Lsl,
        1 => ShiftType::Lsr,
        2 => ShiftType::Asr,
        _ => ShiftType::Ror,
    }
}

const fn extend(option: u32) -> Extend {
    match option & 7 {
        0 => Extend::Uxtb,
        1 => Extend::Uxth,
        2 => Extend::Uxtw,
        3 => Extend::Uxtx,
        4 => Extend::Sxtb,
        5 => Extend::Sxth,
        6 => Extend::Sxtw,
        _ => Extend::Sxtx,
    }
}

const fn logic_op(opc: u32) -> LogicOp {
    match opc & 3 {
        0 => LogicOp::And,
        1 => LogicOp::Orr,
        2 => LogicOp::Eor,
        _ => LogicOp::Ands,
    }
}

fn pc_rel(w: u32) -> Option<Inst> {
    let imm = sext((field(w, 5, 19) << 2) | field(w, 29, 2), 21);
    let rd = Reg::zr(rd(w));
    Some(if bit(w, 31) {
        Inst::Adrp {
            rd,
            offset: (imm as i64) << 12,
        }
    } else {
        Inst::Adr { rd, offset: imm }
    })
}

fn add_sub_imm(w: u32) -> Option<Inst> {
    let setflags = bit(w, 29);
    Some(Inst::AddSubImm {
        sf: bit(w, 31),
        sub: bit(w, 30),
        setflags,
        rd: if setflags {
            Reg::zr(rd(w))
        } else {
            Reg::sp(rd(w))
        },
        rn: Reg::sp(rn(w)),
        imm: field(w, 10, 12) << if bit(w, 22) { 12 } else { 0 },
    })
}

fn add_sub_tag(w: u32) -> Option<Inst> {
    Some(Inst::AddSubImm {
        sf: true,
        sub: bit(w, 30),
        setflags: false,
        rd: Reg::sp(rd(w)),
        rn: Reg::sp(rn(w)),
        imm: field(w, 16, 6) << 4,
    })
}

fn logical_imm(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    let n = bit(w, 22);
    if !sf && n {
        return None;
    }
    let (mask, _) = decode_bit_masks(n, field(w, 10, 6), field(w, 16, 6), true)?;
    let op = logic_op(field(w, 29, 2));
    Some(Inst::LogicalImm {
        sf,
        op,
        rd: if op == LogicOp::Ands {
            Reg::zr(rd(w))
        } else {
            Reg::sp(rd(w))
        },
        rn: Reg::zr(rn(w)),
        imm: if sf { mask } else { mask & 0xffff_ffff },
    })
}

fn move_wide(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    let hw = field(w, 21, 2);
    if !sf && hw >= 2 {
        return None;
    }
    let op = match field(w, 29, 2) {
        0 => MovOp::N,
        2 => MovOp::Z,
        3 => MovOp::K,
        _ => return None,
    };
    Some(Inst::MovWide {
        sf,
        op,
        rd: Reg::zr(rd(w)),
        shift: (hw * 16) as u8,
        imm: field(w, 5, 16) as u16,
    })
}

fn bitfield(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    let immr = field(w, 16, 6);
    let imms = field(w, 10, 6);
    if bit(w, 22) != sf || (!sf && (immr | imms) & 0x20 != 0) {
        return None;
    }
    let op = match field(w, 29, 2) {
        0 => BitfieldOp::Sbfm,
        1 => BitfieldOp::Bfm,
        2 => BitfieldOp::Ubfm,
        _ => return None,
    };
    Some(Inst::Bitfield {
        sf,
        op,
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(rn(w)),
        immr: immr as u8,
        imms: imms as u8,
    })
}

fn extract(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    let lsb = field(w, 10, 6);
    if field(w, 29, 2) != 0 || bit(w, 22) != sf || bit(w, 21) || (!sf && lsb >= 32) {
        return None;
    }
    Some(Inst::Extr {
        sf,
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(rn(w)),
        rm: Reg::zr(rm(w)),
        lsb: lsb as u8,
    })
}

fn branch_imm(w: u32) -> Option<Inst> {
    let offset = sext(field(w, 0, 26), 26) << 2;
    Some(if bit(w, 31) {
        Inst::Bl { offset }
    } else {
        Inst::B { offset }
    })
}

fn compare_branch(w: u32) -> Option<Inst> {
    Some(Inst::Cbz {
        sf: bit(w, 31),
        nonzero: bit(w, 24),
        rt: Reg::zr(rd(w)),
        offset: sext(field(w, 5, 19), 19) << 2,
    })
}

fn test_branch(w: u32) -> Option<Inst> {
    Some(Inst::Tbz {
        nonzero: bit(w, 24),
        rt: Reg::zr(rd(w)),
        bit: ((field(w, 31, 1) << 5) | field(w, 19, 5)) as u8,
        offset: sext(field(w, 5, 14), 14) << 2,
    })
}

fn cond_branch(w: u32) -> Option<Inst> {
    Some(Inst::BCond {
        cond: field(w, 0, 4) as u8,
        offset: sext(field(w, 5, 19), 19) << 2,
    })
}

fn exception(w: u32) -> Option<Inst> {
    let imm = field(w, 5, 16) as u16;
    match (field(w, 21, 3), field(w, 2, 3), field(w, 0, 2)) {
        (0, 0, 1) => Some(Inst::Svc { imm }),
        (1, 0, 0) => Some(Inst::Brk { imm }),
        _ => None,
    }
}

fn hint(w: u32) -> Option<Inst> {
    let op = match field(w, 5, 7) {
        1 => HintOp::Yield,
        2 => HintOp::Wfe,
        3 => HintOp::Wfi,
        4 => HintOp::Sev,
        5 => HintOp::Sevl,
        _ => return Some(Inst::Nop),
    };
    Some(Inst::Hint { op })
}

fn barrier(w: u32) -> Option<Inst> {
    let op = match field(w, 5, 3) {
        1 | 4 => BarrierOp::Dsb,
        2 => return Some(Inst::Clrex),
        5 => BarrierOp::Dmb,
        6 => BarrierOp::Isb,
        7 => BarrierOp::Sb,
        _ => return None,
    };
    Some(Inst::Barrier { op })
}

fn pstate(w: u32) -> Option<Inst> {
    match (field(w, 16, 3), field(w, 8, 4), field(w, 5, 3)) {
        (0, 0, 0) => Some(Inst::Flag { op: FlagOp::Cfinv }),
        (0, 0, 1) => Some(Inst::Flag { op: FlagOp::Xaflag }),
        (0, 0, 2) => Some(Inst::Flag { op: FlagOp::Axflag }),
        (3, _, 1 | 2 | 4) => Some(Inst::Nop),
        _ => None,
    }
}

fn sys(w: u32) -> Option<Inst> {
    let rt = Reg::zr(rd(w));
    match (
        field(w, 16, 3),
        field(w, 12, 4),
        field(w, 8, 4),
        field(w, 5, 3),
    ) {
        (0, 7, 1 | 5, 0) => Some(Inst::IcIallu),
        (3, 7, 5, 1) => Some(Inst::IcIvau { rt }),
        (3, 7, 4, 1 | 4) => Some(Inst::DcZva { rt }),
        (3, 7, 4, 3) | (3, 7, 10..=14, 1 | 3 | 5) | (0, 7, 6, 1 | 2) | (0, 7, 10 | 14, 2) => {
            Some(Inst::Nop)
        }
        _ => None,
    }
}

fn sys_reg(w: u32) -> Option<Inst> {
    if !bit(w, 19) {
        return None;
    }
    let reg = match (
        field(w, 16, 3),
        field(w, 12, 4),
        field(w, 8, 4),
        field(w, 5, 3),
    ) {
        (3, 4, 2, 0) => SysReg::Nzcv,
        (3, 4, 4, 0) => SysReg::Fpcr,
        (3, 4, 4, 1) => SysReg::Fpsr,
        (3, 13, 0, 2) => SysReg::TpidrEl0,
        (3, 13, 0, 3) => SysReg::TpidrroEl0,
        (3, 14, 0, 0) => SysReg::CntfrqEl0,
        (3, 14, 0, 1) => SysReg::CntpctEl0,
        (3, 14, 0, 2) => SysReg::CntvctEl0,
        (3, 0, 0, 1) => SysReg::CtrEl0,
        (3, 0, 0, 7) => SysReg::DczidEl0,
        _ => return None,
    };
    let rt = Reg::zr(rd(w));
    if bit(w, 21) {
        return Some(Inst::Mrs { rt, reg });
    }
    match reg {
        SysReg::Nzcv | SysReg::Fpcr | SysReg::Fpsr | SysReg::TpidrEl0 => {
            Some(Inst::Msr { rt, reg })
        }
        _ => None,
    }
}

fn branch_reg(w: u32) -> Option<Inst> {
    if field(w, 16, 5) != 31 {
        return None;
    }
    let opc = field(w, 21, 4);
    let op3 = field(w, 10, 6);
    let op4 = field(w, 0, 5);
    let rn = Reg::zr(rn(w));
    match (opc, op3, op4) {
        (0, 0, 0) | (0, 2 | 3, 31) | (8, 2 | 3, _) => Some(Inst::Br { rn }),
        (1, 0, 0) | (1, 2 | 3, 31) | (9, 2 | 3, _) => Some(Inst::Blr { rn }),
        (2, 0, 0) => Some(Inst::Ret { rn }),
        (2, 2 | 3, 31) if rn.0 == 31 => Some(Inst::Ret { rn: Reg::LR }),
        _ => None,
    }
}

fn exclusive(w: u32) -> Option<Inst> {
    let size = field(w, 30, 2) as u8;
    let (o2, load, o1, o0) = (bit(w, 23), bit(w, 22), bit(w, 21), bit(w, 15));
    let rs = Reg::zr(rm(w));
    let rt = Reg::zr(rd(w));
    let rn = Reg::sp(rn(w));
    match (o2, o1) {
        (false, false) if load => Some(Inst::LoadExclusive {
            size,
            pair: false,
            acquire: o0,
            rt,
            rt2: Reg::ZR,
            rn,
        }),
        (false, false) => Some(Inst::StoreExclusive {
            size,
            pair: false,
            release: o0,
            rs,
            rt,
            rt2: Reg::ZR,
            rn,
        }),
        (false, true) if size >= 2 => {
            let size = size & 1 | 2;
            let rt2 = Reg::zr(ra(w));
            Some(if load {
                Inst::LoadExclusive {
                    size,
                    pair: true,
                    acquire: o0,
                    rt,
                    rt2,
                    rn,
                }
            } else {
                Inst::StoreExclusive {
                    size,
                    pair: true,
                    release: o0,
                    rs,
                    rt,
                    rt2,
                    rn,
                }
            })
        }
        (false, true) => {
            let (s, t) = (rm(w), rd(w));
            if ra(w) != 31 || s & 1 != 0 || t & 1 != 0 {
                return None;
            }
            Some(Inst::Casp {
                size: size & 1 | 2,
                acquire: load,
                release: o0,
                rs: s as u8,
                rt: t as u8,
                rn,
            })
        }
        (true, false) if load => Some(Inst::Load {
            size,
            ext: LoadExt::Zero,
            acquire: true,
            rt,
            rn,
            addr: Addr::Offset(0),
        }),
        (true, false) => Some(Inst::Store {
            size,
            release: true,
            rt,
            rn,
            addr: Addr::Offset(0),
        }),
        (true, true) => {
            if ra(w) != 31 {
                return None;
            }
            Some(Inst::Cas {
                size,
                acquire: load,
                release: o0,
                rs,
                rt,
                rn,
            })
        }
    }
}

fn rcpc_unscaled(w: u32) -> Option<Inst> {
    let size = field(w, 30, 2) as u8;
    let rt = Reg::zr(rd(w));
    let rn = Reg::sp(rn(w));
    let addr = Addr::Offset(sext(field(w, 12, 9), 9));
    let ext = match field(w, 22, 2) {
        0 => {
            return Some(Inst::Store {
                size,
                release: true,
                rt,
                rn,
                addr,
            });
        }
        1 => LoadExt::Zero,
        2 if size < 3 => LoadExt::Sign64,
        3 if size < 2 => LoadExt::Sign32,
        _ => return None,
    };
    Some(Inst::Load {
        size,
        ext,
        acquire: true,
        rt,
        rn,
        addr,
    })
}

fn tag_store(w: u32) -> Option<Inst> {
    let opc = field(w, 22, 2);
    let offset = sext(field(w, 12, 9), 9) << 4;
    let addr = match field(w, 10, 2) {
        0 if opc == 1 => return Some(Inst::Nop),
        0 => return None,
        1 => Addr::Post(offset),
        2 => Addr::Offset(offset),
        _ => Addr::Pre(offset),
    };
    let zero_granules = match opc {
        1 => 1,
        3 => 2,
        _ => 0,
    };
    Some(Inst::StoreTag {
        zero_granules,
        rn: Reg::sp(rn(w)),
        addr,
    })
}

fn load_literal(w: u32) -> Option<Inst> {
    let offset = sext(field(w, 5, 19), 19) << 2;
    let rt = rd(w);
    let opc = field(w, 30, 2);
    if bit(w, 26) {
        if opc == 3 {
            return None;
        }
        return Some(Inst::LoadLiteralVec {
            size: opc as u8 + 2,
            vt: rt as u8,
            offset,
        });
    }
    let (size, ext) = match opc {
        0 => (2, LoadExt::Zero),
        1 => (3, LoadExt::Zero),
        2 => (2, LoadExt::Sign64),
        _ => return Some(Inst::Nop),
    };
    Some(Inst::LoadLiteral {
        size,
        ext,
        rt: Reg::zr(rt),
        offset,
    })
}

fn pair(w: u32) -> Option<Inst> {
    let opc = field(w, 30, 2);
    let mode = field(w, 23, 2);
    let load = bit(w, 22);
    let imm7 = sext(field(w, 15, 7), 7);
    let rn = Reg::sp(rn(w));
    let (size, ext, scale) = if bit(w, 26) {
        if opc == 3 {
            return None;
        }
        (opc as u8 + 2, LoadExt::Zero, opc + 2)
    } else {
        match (opc, load) {
            (0, _) => (2, LoadExt::Zero, 2),
            (1, true) if mode != 0 => (2, LoadExt::Sign64, 2),
            (1, false) if mode != 0 => (3, LoadExt::Zero, 4),
            (2, _) => (3, LoadExt::Zero, 3),
            _ => return None,
        }
    };
    let offset = imm7 << scale;
    let addr = match mode {
        1 => Addr::Post(offset),
        3 => Addr::Pre(offset),
        _ => Addr::Offset(offset),
    };
    let (t, t2) = (rd(w), ra(w));
    Some(match (bit(w, 26), load) {
        (true, true) => Inst::LoadPairVec {
            size,
            vt: t as u8,
            vt2: t2 as u8,
            rn,
            addr,
        },
        (true, false) => Inst::StorePairVec {
            size,
            vt: t as u8,
            vt2: t2 as u8,
            rn,
            addr,
        },
        (false, true) => Inst::LoadPair {
            size,
            ext,
            rt: Reg::zr(t),
            rt2: Reg::zr(t2),
            rn,
            addr,
        },
        (false, false) => Inst::StorePair {
            size,
            rt: Reg::zr(t),
            rt2: Reg::zr(t2),
            rn,
            addr,
        },
    })
}

/// What a single-register load/store addressing form permits.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Form {
    /// Unsigned offset, register offset, unscaled: `PRFM` allowed.
    Plain,
    /// Pre- and post-index.
    Writeback,
    /// `LDTR`/`STTR`: general registers only.
    Unprivileged,
}

/// The load/store register class shared by every addressing form. `addr` receives the access
/// size in log2 bytes.
fn single(w: u32, form: Form, addr: impl Fn(u8) -> Addr) -> Option<Inst> {
    let size = field(w, 30, 2) as u8;
    let opc = field(w, 22, 2);
    let rn = Reg::sp(rn(w));
    let t = rd(w);
    if bit(w, 26) {
        if form == Form::Unprivileged {
            return None;
        }
        let (size, load) = match opc {
            0 | 1 => (size, opc == 1),
            _ if size == 0 => (4, opc == 3),
            _ => return None,
        };
        let addr = addr(size);
        return Some(if load {
            Inst::LoadVec {
                size,
                vt: t as u8,
                rn,
                addr,
            }
        } else {
            Inst::StoreVec {
                size,
                vt: t as u8,
                rn,
                addr,
            }
        });
    }
    let rt = Reg::zr(t);
    let ext = match opc {
        0 => {
            return Some(Inst::Store {
                size,
                release: false,
                rt,
                rn,
                addr: addr(size),
            });
        }
        1 => LoadExt::Zero,
        2 if size == 3 => return (form == Form::Plain).then_some(Inst::Nop),
        2 => LoadExt::Sign64,
        _ if size < 2 => LoadExt::Sign32,
        _ => return None,
    };
    Some(Inst::Load {
        size,
        ext,
        acquire: false,
        rt,
        rn,
        addr: addr(size),
    })
}

fn single_unsigned(w: u32) -> Option<Inst> {
    let imm12 = field(w, 10, 12) as i32;
    single(w, Form::Plain, |size| Addr::Offset(imm12 << size))
}

fn single_imm9(w: u32) -> Option<Inst> {
    let imm9 = sext(field(w, 12, 9), 9);
    let (form, addr) = match field(w, 10, 2) {
        0 => (Form::Plain, Addr::Offset(imm9)),
        1 => (Form::Writeback, Addr::Post(imm9)),
        2 => (Form::Unprivileged, Addr::Offset(imm9)),
        _ => (Form::Writeback, Addr::Pre(imm9)),
    };
    single(w, form, |_| addr)
}

fn single_reg(w: u32) -> Option<Inst> {
    let option = field(w, 13, 3);
    if option & 2 == 0 {
        return None;
    }
    let scaled = bit(w, 12);
    let rm = Reg::zr(rm(w));
    single(w, Form::Plain, |size| Addr::Reg {
        rm,
        extend: extend(option),
        shift: if scaled { size } else { 0 },
    })
}

fn atomic(w: u32) -> Option<Inst> {
    let size = field(w, 30, 2) as u8;
    let (acquire, release) = (bit(w, 23), bit(w, 22));
    let rs = Reg::zr(rm(w));
    let rt = Reg::zr(rd(w));
    let rn = Reg::sp(rn(w));
    let op = match (bit(w, 15), field(w, 12, 3)) {
        (false, 0) => AtomicOp::Add,
        (false, 1) => AtomicOp::Clr,
        (false, 2) => AtomicOp::Eor,
        (false, 3) => AtomicOp::Set,
        (false, 4) => AtomicOp::Smax,
        (false, 5) => AtomicOp::Smin,
        (false, 6) => AtomicOp::Umax,
        (false, 7) => AtomicOp::Umin,
        (true, 0) => AtomicOp::Swp,
        (true, 4) if acquire && !release && rs.0 == 31 => {
            return Some(Inst::Load {
                size,
                ext: LoadExt::Zero,
                acquire: true,
                rt,
                rn,
                addr: Addr::Offset(0),
            });
        }
        _ => return None,
    };
    Some(Inst::AtomicRmw {
        op,
        size,
        acquire,
        release,
        rs,
        rt,
        rn,
    })
}

fn ldra(w: u32) -> Option<Inst> {
    let offset = sext((field(w, 22, 1) << 9) | field(w, 12, 9), 10) << 3;
    Some(Inst::Load {
        size: 3,
        ext: LoadExt::Zero,
        acquire: false,
        rt: Reg::zr(rd(w)),
        rn: Reg::sp(rn(w)),
        addr: if bit(w, 11) {
            Addr::Pre(offset)
        } else {
            Addr::Offset(offset)
        },
    })
}

fn post_index(w: u32, bytes: u32) -> PostIndex {
    match (bit(w, 23), rm(w)) {
        (false, _) => PostIndex::None,
        (true, 31) => PostIndex::Imm(bytes as u8),
        (true, m) => PostIndex::Reg(Reg::zr(m)),
    }
}

fn vec_multi(w: u32) -> Option<Inst> {
    let q = bit(w, 30);
    let size = field(w, 10, 2) as u8;
    let (rpt, selem) = match field(w, 12, 4) {
        0 => (1, 4),
        2 => (4, 1),
        4 => (1, 3),
        6 => (3, 1),
        7 => (1, 1),
        8 => (1, 2),
        10 => (2, 1),
        _ => return None,
    };
    if size == 3 && !q && selem != 1 {
        return None;
    }
    Some(Inst::VecMulti {
        load: bit(w, 22),
        q,
        size,
        rpt,
        selem,
        vt: rd(w) as u8,
        rn: Reg::sp(rn(w)),
        post: post_index(w, (rpt * selem) as u32 * if q { 16 } else { 8 }),
    })
}

fn vec_single(w: u32) -> Option<Inst> {
    let q = field(w, 30, 1);
    let load = bit(w, 22);
    let opcode = field(w, 13, 3);
    let s = field(w, 12, 1);
    let size = field(w, 10, 2);
    let selem = (((opcode & 1) << 1) | field(w, 21, 1)) + 1;
    let vt = rd(w) as u8;
    let rn = Reg::sp(rn(w));
    let (esize, index) = match opcode >> 1 {
        0 => (0, (q << 3) | (s << 2) | size),
        1 if size & 1 == 0 => (1, (q << 2) | (s << 1) | (size >> 1)),
        2 if size == 0 => (2, (q << 1) | s),
        2 if size == 1 && s == 0 => (3, q),
        3 if load && s == 0 => {
            return Some(Inst::VecReplicate {
                q: q != 0,
                size: size as u8,
                selem: selem as u8,
                vt,
                rn,
                post: post_index(w, selem << size),
            });
        }
        _ => return None,
    };
    Some(Inst::VecSingle {
        load,
        size: esize,
        selem: selem as u8,
        index: index as u8,
        vt,
        rn,
        post: post_index(w, selem << esize),
    })
}

fn dp2(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    if bit(w, 29) {
        return None;
    }
    let (rd, rn) = (Reg::zr(rd(w)), Reg::zr(rn(w)));
    let rm_field = rm(w);
    let rm = Reg::zr(rm_field);
    let op = match field(w, 10, 6) {
        2 => Dp2Op::Udiv,
        3 => Dp2Op::Sdiv,
        8 => Dp2Op::Lslv,
        9 => Dp2Op::Lsrv,
        10 => Dp2Op::Asrv,
        11 => Dp2Op::Rorv,
        opcode @ 16..=23 => {
            let size = (opcode & 3) as u8;
            if (size == 3) != sf {
                return None;
            }
            return Some(Inst::Crc32 {
                castagnoli: opcode & 4 != 0,
                size,
                rd,
                rn,
                rm,
            });
        }
        4 if sf => {
            return Some(Inst::AddSubImm {
                sf: true,
                sub: false,
                setflags: false,
                rd: Reg::sp(rd.0 as u32),
                rn: Reg::sp(rn.0 as u32),
                imm: 0,
            });
        }
        5 if sf => {
            return Some(Inst::LogicalImm {
                sf: true,
                op: LogicOp::Orr,
                rd,
                rn: rm,
                imm: 1,
            });
        }
        12 if sf => {
            return Some(Inst::Pacga {
                rd,
                rn,
                rm: Reg::sp(rm_field),
            });
        }
        _ => return None,
    };
    Some(Inst::Dp2 { sf, op, rd, rn, rm })
}

fn dp1(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    if bit(w, 29) {
        return None;
    }
    let opcode = field(w, 10, 6);
    let n = rn(w);
    let op = match (field(w, 16, 5), opcode) {
        (0, 0) => Dp1Op::Rbit,
        (0, 1) => Dp1Op::Rev16,
        (0, 2) => Dp1Op::Rev32,
        (0, 3) if sf => Dp1Op::Rev64,
        (0, 4) => Dp1Op::Clz,
        (0, 5) => Dp1Op::Cls,
        (1, 0..=7) if sf => return Some(Inst::Nop),
        (1, 8..=17) if sf && n == 31 => return Some(Inst::Nop),
        _ => return None,
    };
    Some(Inst::Dp1 {
        sf,
        op,
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(n),
    })
}

fn logical_shift(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    let amount = field(w, 10, 6);
    if !sf && amount >= 32 {
        return None;
    }
    Some(Inst::LogicalShift {
        sf,
        op: logic_op(field(w, 29, 2)),
        invert: bit(w, 21),
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(rn(w)),
        rm: Reg::zr(rm(w)),
        shift: shift_type(field(w, 22, 2)),
        amount: amount as u8,
    })
}

fn add_sub_shift(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    let amount = field(w, 10, 6);
    let shift = field(w, 22, 2);
    if shift == 3 || (!sf && amount >= 32) {
        return None;
    }
    Some(Inst::AddSubShift {
        sf,
        sub: bit(w, 30),
        setflags: bit(w, 29),
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(rn(w)),
        rm: Reg::zr(rm(w)),
        shift: shift_type(shift),
        amount: amount as u8,
    })
}

fn add_sub_ext(w: u32) -> Option<Inst> {
    let amount = field(w, 10, 3);
    if field(w, 22, 2) != 0 || amount > 4 {
        return None;
    }
    let setflags = bit(w, 29);
    Some(Inst::AddSubExt {
        sf: bit(w, 31),
        sub: bit(w, 30),
        setflags,
        rd: if setflags {
            Reg::zr(rd(w))
        } else {
            Reg::sp(rd(w))
        },
        rn: Reg::sp(rn(w)),
        rm: Reg::zr(rm(w)),
        extend: extend(field(w, 13, 3)),
        amount: amount as u8,
    })
}

fn add_sub_carry(w: u32) -> Option<Inst> {
    Some(Inst::AddSubCarry {
        sf: bit(w, 31),
        sub: bit(w, 30),
        setflags: bit(w, 29),
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(rn(w)),
        rm: Reg::zr(rm(w)),
    })
}

fn rmif(w: u32) -> Option<Inst> {
    Some(Inst::Rmif {
        rn: Reg::zr(rn(w)),
        shift: field(w, 15, 6) as u8,
        mask: field(w, 0, 4) as u8,
    })
}

fn setf(w: u32) -> Option<Inst> {
    Some(Inst::Setf {
        rn: Reg::zr(rn(w)),
        sixteen: bit(w, 14),
    })
}

fn cond_compare(w: u32) -> Option<Inst> {
    Some(Inst::CondCompare {
        sf: bit(w, 31),
        sub: bit(w, 30),
        rn: Reg::zr(rn(w)),
        rhs: rm(w) as u8,
        imm: bit(w, 11),
        nzcv: field(w, 0, 4) as u8,
        cond: field(w, 12, 4) as u8,
    })
}

fn cond_select(w: u32) -> Option<Inst> {
    if bit(w, 29) || bit(w, 11) {
        return None;
    }
    let op = match (bit(w, 30), bit(w, 10)) {
        (false, false) => CselOp::Csel,
        (false, true) => CselOp::Csinc,
        (true, false) => CselOp::Csinv,
        (true, true) => CselOp::Csneg,
    };
    Some(Inst::CondSelect {
        sf: bit(w, 31),
        op,
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(rn(w)),
        rm: Reg::zr(rm(w)),
        cond: field(w, 12, 4) as u8,
    })
}

fn dp3(w: u32) -> Option<Inst> {
    let sf = bit(w, 31);
    if field(w, 29, 2) != 0 {
        return None;
    }
    let op = match (field(w, 21, 3), bit(w, 15)) {
        (0, false) => Dp3Op::Madd,
        (0, true) => Dp3Op::Msub,
        (1, false) if sf => Dp3Op::Smaddl,
        (1, true) if sf => Dp3Op::Smsubl,
        (2, false) if sf => Dp3Op::Smulh,
        (5, false) if sf => Dp3Op::Umaddl,
        (5, true) if sf => Dp3Op::Umsubl,
        (6, false) if sf => Dp3Op::Umulh,
        _ => return None,
    };
    Some(Inst::Dp3 {
        sf,
        op,
        rd: Reg::zr(rd(w)),
        rn: Reg::zr(rn(w)),
        rm: Reg::zr(rm(w)),
        ra: Reg::zr(ra(w)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linear(word: u32) -> Option<Inst> {
        let mut order: Vec<&Entry> = ENTRIES.iter().collect();
        order.sort_by_key(|e| std::cmp::Reverse(e.mask.count_ones()));
        order
            .into_iter()
            .filter(|e| word & e.mask == e.value)
            .find_map(|e| (e.decode)(word))
    }

    #[test]
    fn buckets_agree_with_a_linear_scan() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..200_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let word = state as u32;
            assert_eq!(lookup(word), linear(word), "{word:#010x}");
        }
    }
}
