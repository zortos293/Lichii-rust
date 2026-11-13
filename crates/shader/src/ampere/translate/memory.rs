//! Constant, global, local and shared memory, atomics and memory barriers.
//!
//! Global accesses become the IR's global memory operations on 64-bit addresses, which the
//! storage buffer pass rebases on the buffers whose pointers live in constant buffers. Generic
//! accesses (LD, ST, ATOM) are treated as global: NVN programs use them on global pointers.
//! Cache control and dependency barriers (CCTL, DEPBAR, LDGDEPBAR, ERRBAR) have no effect,
//! because every translated access, LDGSTS included, completes in program order. Bindless
//! constant buffers are unsupported.

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ampere::decode::Form;
use crate::ampere::opcode::Opcode;
use crate::ir::{FmzMode, FpControl, FpRounding, ICompare, Opcode as Op, Pred, Reg, Value};

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    use Opcode::*;
    match op {
        Ldc => ldc(t),
        Uldc => uldc(t),
        Ldcg => ldcg(t),
        Ld | Ldg => ldg(t),
        St | Stg => stg(t),
        Ldl => ldl(t),
        Stl => stl(t),
        Lds => lds(t),
        Sts => sts(t),
        Ldgsts => ldgsts(t),
        Ldsm => ldsm(t),
        Atom | Atomg => atomg(t),
        AtomCas | AtomgCas => atomg_cas(t),
        Red => red(t),
        Atoms => atoms(t),
        AtomsCas => atoms_cas(t),
        AtomsPopcInc => atoms_popc_inc(t),
        Membar => membar(t),
        Cctl | Errbar | Depbar | Ldgdepbar => Ok(()),
        _ => Err(t.unsupported()),
    }
}

/// Access size and signedness, bits 73..76.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MemType {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
    B128,
}

impl MemType {
    fn decode(t: &Translator<'_>) -> Result<MemType, TranslateError> {
        Ok(match t.insn.bits(73, 3) {
            0 => MemType::U8,
            1 => MemType::S8,
            2 => MemType::U16,
            3 => MemType::S16,
            4 => MemType::B32,
            5 => MemType::B64,
            6 => MemType::B128,
            _ => return Err(t.unsupported()),
        })
    }

    /// A constant buffer access, which has no 128-bit size.
    fn decode_cbuf(t: &Translator<'_>) -> Result<MemType, TranslateError> {
        match MemType::decode(t)? {
            MemType::B128 => Err(t.unsupported()),
            ty => Ok(ty),
        }
    }

    const fn words(self) -> u16 {
        match self {
            MemType::B64 => 2,
            MemType::B128 => 4,
            _ => 1,
        }
    }

    const fn bits(self) -> u32 {
        match self {
            MemType::U8 | MemType::S8 => 8,
            MemType::U16 | MemType::S16 => 16,
            MemType::B32 => 32,
            MemType::B64 => 64,
            MemType::B128 => 128,
        }
    }

    const fn signed(self) -> bool {
        matches!(self, MemType::S8 | MemType::S16)
    }
}

fn check_aligned(t: &Translator<'_>, reg: Reg, count: u16) -> TResult {
    if reg.is_zero() || reg.0.is_multiple_of(count) {
        Ok(())
    } else {
        Err(t.unsupported())
    }
}

/// The `count` (1, 2 or 4) registers from `reg`: one word, or a vector.
fn read_words(t: &mut Translator<'_>, reg: Reg, count: u16) -> Result<Value, TranslateError> {
    check_aligned(t, reg, count)?;
    if count == 1 {
        return Ok(t.x(reg));
    }
    let mut words = [Value::U32(0); 4];
    for (i, word) in words.iter_mut().take(usize::from(count)).enumerate() {
        *word = t.x(reg.offset(i as u16));
    }
    Ok(t.ir.composite_construct(&words[..usize::from(count)]))
}

/// Writes a loaded word or vector to the `count` registers from `reg`, zeros where `enable`
/// is false.
fn write_words(t: &mut Translator<'_>, reg: Reg, value: Value, count: u16, enable: Value) {
    for i in 0..count {
        let word = if count == 1 {
            value
        } else {
            t.ir.composite_extract(value, u32::from(i))
        };
        let word = zero_unless(t, enable, word);
        t.set_x(reg.offset(i), word);
    }
}

fn zero_unless(t: &mut Translator<'_>, enable: Value, word: Value) -> Value {
    if enable == Value::U1(true) {
        word
    } else {
        t.ir.select(enable, word, Value::U32(0))
    }
}

/// `reg` plus `imm`, folded when either is zero.
fn add_u32(t: &mut Translator<'_>, reg: Reg, imm: u32) -> Value {
    if reg.is_zero() {
        return Value::U32(imm);
    }
    let base = t.x(reg);
    if imm == 0 {
        base
    } else {
        t.ir.iadd(base, Value::U32(imm))
    }
}

/// The registers a global address sums: `reg` and `ureg`, each a 64-bit pair when wide.
struct GlobalAddress {
    reg: Reg,
    wide: bool,
    ureg: Reg,
    uwide: bool,
}

/// The address registers of a global access. Fixed forms take `Ra` alone, 64-bit with bit 72.
/// Uniform-address forms in UGPR mode (bit 91) add the uniform register at `ureg_at`, 64-bit
/// with bit 72, and read the width of `Ra` from `wide_bit`; with bit 76 the uniform register
/// is a memory descriptor (`desc[URb][Ra.64]`) and takes no part in the address. Without UGPR
/// mode bit 72 is the width of `Ra`, and only URZ is accepted.
fn global_operands(
    t: &Translator<'_>,
    ureg_at: u32,
    wide_bit: u32,
) -> Result<GlobalAddress, TranslateError> {
    let insn = t.insn;
    let reg = insn.src0();
    let ureg = insn.ureg_at(ureg_at);
    if t.form == Form::Fixed || !insn.bit(91) {
        if t.form != Form::Fixed && !ureg.is_zero() {
            return Err(t.unsupported());
        }
        return Ok(GlobalAddress {
            reg,
            wide: insn.bit(72),
            ureg: Reg::URZ,
            uwide: false,
        });
    }
    Ok(GlobalAddress {
        reg,
        wide: insn.bit(wide_bit),
        ureg: if insn.bit(76) { Reg::URZ } else { ureg },
        uwide: insn.bit(72),
    })
}

fn address_term(
    t: &mut Translator<'_>,
    reg: Reg,
    wide: bool,
) -> Result<Option<Value>, TranslateError> {
    if reg.is_zero() {
        return Ok(None);
    }
    Ok(Some(if wide {
        t.l(reg)?
    } else {
        let low = t.x(reg);
        t.ir.uconvert(64, low)
    }))
}

/// The 64-bit address `a.reg + a.ureg + offset`. A lone pair plus an immediate keeps the
/// `IAdd64(PackUint2x32(..), imm)` shape the storage buffer pass tracks.
fn global_address(
    t: &mut Translator<'_>,
    a: GlobalAddress,
    offset: i64,
) -> Result<Value, TranslateError> {
    let reg = address_term(t, a.reg, a.wide)?;
    let ureg = address_term(t, a.ureg, a.uwide)?;
    let base = match (reg, ureg) {
        (Some(reg), Some(ureg)) => Some(t.ir.iadd(reg, ureg)),
        (reg, ureg) => reg.or(ureg),
    };
    Ok(match base {
        None => Value::U64(offset as u64),
        Some(base) if offset == 0 => base,
        Some(base) => t.ir.iadd(base, Value::U64(offset as u64)),
    })
}

/// The uniform register at `at` that a uniform-address form adds in UGPR mode (bit 91).
fn address_ureg(t: &Translator<'_>, at: u32) -> Reg {
    if t.form == Form::UniformAddress && t.insn.bit(91) {
        t.insn.ureg_at(at)
    } else {
        Reg::URZ
    }
}

/// A 32-bit shared or local byte address: `Ra` (scaled by the stride at 78..80 when `scaled`)
/// plus `ureg` plus the signed 24-bit offset at 40..64.
fn byte_address(t: &mut Translator<'_>, ureg: Reg, scaled: bool) -> Value {
    let insn = t.insn;
    let reg = insn.src0();
    let offset = insn.sbits(40, 24) as u32;
    let mut address = None;
    if !reg.is_zero() {
        let base = t.x(reg);
        let stride = if scaled { insn.bits(78, 2) as u32 } else { 0 };
        address = Some(if stride == 0 {
            base
        } else {
            t.ir.shift_left_logical(base, Value::U32(stride + 1))
        });
    }
    if !ureg.is_zero() {
        let uniform = t.x(ureg);
        address = Some(match address {
            Some(base) => t.ir.iadd(base, uniform),
            None => uniform,
        });
    }
    match address {
        None => Value::U32(offset),
        Some(base) if offset == 0 => base,
        Some(base) => t.ir.iadd(base, Value::U32(offset)),
    }
}

/// `LDC Rd, c[bank][Ra + imm]`; `.IS` (mode 2) takes the buffer from `Ra[31:16]` plus
/// `bank` and the offset from `Ra[15:0]` plus `imm`. Modes 1 (`.IL`) and 3 (`.ISL`) are
/// unsupported: neither NAK nor the C++ defines them.
fn ldc(t: &mut Translator<'_>) -> TResult {
    if t.form != Form::Bound {
        return Err(t.unsupported());
    }
    let ty = MemType::decode_cbuf(t)?;
    let dest = t.insn.dst();
    check_aligned(t, dest, ty.words())?;
    let bank = t.insn.bits(54, 5) as u32;
    let imm = t.insn.bits(38, 16) as u32;
    let reg = t.insn.src0();
    let (index, offset) = match t.insn.bits(78, 2) {
        0 => (Value::U32(bank), add_u32(t, reg, imm)),
        2 => {
            let base = t.x(reg);
            let high =
                t.ir.bit_field_extract(base, Value::U32(16), Value::U32(16), false);
            let low =
                t.ir.bit_field_extract(base, Value::U32(0), Value::U32(16), false);
            (
                t.ir.iadd(high, Value::U32(bank)),
                t.ir.iadd(low, Value::U32(imm)),
            )
        }
        _ => return Err(t.unsupported()),
    };
    let value = t.ir.get_cbuf_sized(index, offset, ty.bits(), ty.signed());
    write_words(t, dest, value, ty.words(), Value::U1(true));
    Ok(())
}

/// `ULDC URd, c[bank][imm]`, or `c[bank][URa + imm]` in the indexed form. A bindless buffer
/// (bit 91 of the direct form, `cx[URa]`) is unsupported.
fn uldc(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode_cbuf(t)?;
    let dest = t.insn.udst();
    check_aligned(t, dest, ty.words())?;
    let bank = t.insn.bits(54, 5) as u32;
    let imm = t.insn.bits(38, 16) as u32;
    let offset = match t.form {
        Form::Bound if t.insn.bit(91) => return Err(t.unsupported()),
        Form::Bound => Value::U32(imm),
        _ => add_u32(t, t.insn.usrc0(), imm),
    };
    let value =
        t.ir.get_cbuf_sized(Value::U32(bank), offset, ty.bits(), ty.signed());
    write_words(t, dest, value, ty.words(), Value::U1(true));
    Ok(())
}

fn load_global_op(ty: MemType) -> Op {
    match ty {
        MemType::U8 => Op::LoadGlobalU8,
        MemType::S8 => Op::LoadGlobalS8,
        MemType::U16 => Op::LoadGlobalU16,
        MemType::S16 => Op::LoadGlobalS16,
        MemType::B32 => Op::LoadGlobal32,
        MemType::B64 => Op::LoadGlobal64,
        MemType::B128 => Op::LoadGlobal128,
    }
}

/// `ULDC URd, [URa.64 + imm32], UPp` (LDCG): a uniform global load through the constant
/// cache; a false predicate loads zero.
fn ldcg(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode_cbuf(t)?;
    let dest = t.insn.udst();
    check_aligned(t, dest, ty.words())?;
    let operands = GlobalAddress {
        reg: Reg::RZ,
        wide: false,
        ureg: t.insn.usrc0(),
        uwide: true,
    };
    let address = global_address(t, operands, t.insn.sbits(38, 32))?;
    let value = t.ir.inst(load_global_op(ty), &[address]);
    let enable = t.pred(t.insn.upred_src(87, 90));
    write_words(t, dest, value, ty.words(), enable);
    Ok(())
}

/// `LDG Rd, [Ra + URb + imm], Pp`: the load predicate (64..67, encoded as 7 - P, negated by
/// bit 67) loads zero when false.
fn ldg(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode(t)?;
    let dest = t.insn.dst();
    check_aligned(t, dest, ty.words())?;
    let operands = global_operands(t, 32, 90)?;
    let address = global_address(t, operands, t.insn.sbits(40, 24))?;
    let value = t.ir.inst(load_global_op(ty), &[address]);
    let pred = Pred::p(7 - t.insn.bits(64, 3) as u8);
    let enable = t.pred((pred, t.insn.bit(67)));
    write_words(t, dest, value, ty.words(), enable);
    Ok(())
}

/// `STG [Ra + URb + imm], Rb`.
fn stg(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode(t)?;
    let op = match ty {
        MemType::U8 => Op::WriteGlobalU8,
        MemType::S8 => Op::WriteGlobalS8,
        MemType::U16 => Op::WriteGlobalU16,
        MemType::S16 => Op::WriteGlobalS16,
        MemType::B32 => Op::WriteGlobal32,
        MemType::B64 => Op::WriteGlobal64,
        MemType::B128 => Op::WriteGlobal128,
    };
    let data = t.insn.src1();
    check_aligned(t, data, ty.words())?;
    let operands = global_operands(t, 64, 90)?;
    let address = global_address(t, operands, t.insn.sbits(40, 24))?;
    let value = read_words(t, data, ty.words())?;
    t.ir.inst(op, &[address, value]);
    Ok(())
}

/// Bit position of the byte (`mask` 24) or half word (`mask` 16) at `offset` in its word.
fn sub_word_bit(t: &mut Translator<'_>, offset: Value, mask: u32) -> Value {
    let bits = t.ir.shift_left_logical(offset, Value::U32(3));
    t.ir.bitwise_and(bits, Value::U32(mask))
}

/// The local memory word index of a byte offset.
fn local_word(t: &mut Translator<'_>, offset: Value) -> Value {
    match offset {
        Value::U32(bytes) => Value::U32(bytes / 4),
        _ => t.ir.shift_right_arithmetic(offset, Value::U32(2)),
    }
}

/// Loads a local memory word; reads past the end return zero.
fn load_local(t: &mut Translator<'_>, word: Value, offset: Value, size: u32) -> Value {
    let in_bounds = t.ir.icompare(ICompare::ULessThan, offset, Value::U32(size));
    let value = t.ir.inst(Op::LoadLocal, &[word]);
    t.ir.select(in_bounds, value, Value::U32(0))
}

/// `LDL Rd, [Ra + URb + imm]`.
fn ldl(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode(t)?;
    let dest = t.insn.dst();
    let count = ty.words();
    check_aligned(t, dest, count)?;
    let ureg = address_ureg(t, 32);
    let offset = byte_address(t, ureg, false);
    let word = local_word(t, offset);
    let size = t.env.local_memory_size();
    let value = load_local(t, word, offset, size);
    if ty.bits() < 32 {
        let bit = sub_word_bit(t, offset, if ty.bits() == 8 { 24 } else { 16 });
        let field =
            t.ir.bit_field_extract(value, bit, Value::U32(ty.bits()), ty.signed());
        t.set_x(dest, field);
        return Ok(());
    }
    t.set_x(dest, value);
    for i in 1..count {
        let sub_word = t.ir.iadd(word, Value::U32(u32::from(i)));
        let sub_offset = t.ir.iadd(offset, Value::U32(u32::from(i) * 4));
        let value = load_local(t, sub_word, sub_offset, size);
        t.set_x(dest.offset(i), value);
    }
    Ok(())
}

/// `STL [Ra + URb + imm], Rb`; stores past the end of local memory are dropped.
fn stl(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode(t)?;
    let data = t.insn.src1();
    let count = ty.words();
    check_aligned(t, data, count)?;
    let ureg = address_ureg(t, 64);
    let offset = byte_address(t, ureg, false);
    if let Value::U32(bytes) = offset
        && bytes >= t.env.local_memory_size()
    {
        return Ok(());
    }
    let word = local_word(t, offset);
    let src = t.x(data);
    if ty.bits() < 32 {
        let bit = sub_word_bit(t, offset, if ty.bits() == 8 { 24 } else { 16 });
        let old = t.ir.inst(Op::LoadLocal, &[word]);
        let value = t.ir.bit_field_insert(old, src, bit, Value::U32(ty.bits()));
        t.ir.inst(Op::WriteLocal, &[word, value]);
        return Ok(());
    }
    t.ir.inst(Op::WriteLocal, &[word, src]);
    for i in 1..count {
        let sub_word = t.ir.iadd(word, Value::U32(u32::from(i)));
        let value = t.x(data.offset(i));
        t.ir.inst(Op::WriteLocal, &[sub_word, value]);
    }
    Ok(())
}

/// `LDS Rd, [Ra.Xn + URb + imm]`.
fn lds(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode(t)?;
    let op = match ty {
        MemType::U8 => Op::LoadSharedU8,
        MemType::S8 => Op::LoadSharedS8,
        MemType::U16 => Op::LoadSharedU16,
        MemType::S16 => Op::LoadSharedS16,
        MemType::B32 => Op::LoadSharedU32,
        MemType::B64 => Op::LoadSharedU64,
        MemType::B128 => Op::LoadSharedU128,
    };
    let dest = t.insn.dst();
    check_aligned(t, dest, ty.words())?;
    let ureg = address_ureg(t, 32);
    let address = byte_address(t, ureg, true);
    let value = t.ir.inst(op, &[address]);
    write_words(t, dest, value, ty.words(), Value::U1(true));
    Ok(())
}

/// `LDSM.16.M88.{1,2,4} Rd, [Ra + URb + imm]` (ldmatrix): 8x8 matrices of 16-bit values
/// from shared memory, as many as bits 72..74 give. Lanes 8m..8m+7 hold the row addresses of
/// matrix m, and lane t receives the 32 bits at row t/4, column 2(t%4) of each matrix in Rd+m.
/// `.MT88` (78) transposes: lane t receives column t/4 of rows 2(t%4) and 2(t%4)+1.
fn ldsm(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let uniform = insn.bit(91);
    if insn.bits(72, 2) == 3
        || insn.bits(64, 32) & !0x0800_4300 != 0
        || (!uniform && insn.bits(32, 8) != 0)
    {
        return Err(t.unsupported());
    }
    let count = 1u16 << insn.bits(72, 2);
    let dest = insn.dst();
    check_aligned(t, dest, count)?;
    let ureg = address_ureg(t, 32);
    let base = byte_address(t, ureg, false);
    let lane = t.ir.lane_id();
    let quad = t.ir.bitwise_and(lane, Value::U32(3));
    let group = t.ir.shift_right_logical(lane, Value::U32(2));
    let transpose = insn.bit(78);
    let (first_row, column) = if transpose {
        (
            t.ir.shift_left_logical(quad, Value::U32(1)),
            t.ir.shift_left_logical(group, Value::U32(1)),
        )
    } else {
        (group, t.ir.shift_left_logical(quad, Value::U32(2)))
    };
    let load = |t: &mut Translator<'_>, row: u32, op: Op| {
        let source = t.ir.iadd(first_row, Value::U32(row));
        let address = t.ir.inst(
            Op::ShuffleIndex,
            &[base, source, Value::U32(31), Value::U32(0)],
        );
        let address = t.ir.iadd(address, column);
        t.ir.inst(op, &[address])
    };
    for matrix in 0..count {
        let row = u32::from(matrix) * 8;
        let value = if transpose {
            let low = load(t, row, Op::LoadSharedU16);
            let high = load(t, row + 1, Op::LoadSharedU16);
            let high = t.ir.shift_left_logical(high, Value::U32(16));
            t.ir.bitwise_or(low, high)
        } else {
            load(t, row, Op::LoadSharedU32)
        };
        if !dest.is_zero() {
            t.set_x(dest.offset(matrix), value);
        }
    }
    Ok(())
}

fn write_shared_op(ty: MemType) -> Op {
    match ty {
        MemType::U8 | MemType::S8 => Op::WriteSharedU8,
        MemType::U16 | MemType::S16 => Op::WriteSharedU16,
        MemType::B32 => Op::WriteSharedU32,
        MemType::B64 => Op::WriteSharedU64,
        MemType::B128 => Op::WriteSharedU128,
    }
}

/// `STS [Ra.Xn + URb + imm], Rb`.
fn sts(t: &mut Translator<'_>) -> TResult {
    let ty = MemType::decode(t)?;
    let data = t.insn.src1();
    check_aligned(t, data, ty.words())?;
    let ureg = address_ureg(t, 64);
    let address = byte_address(t, ureg, true);
    let value = read_words(t, data, ty.words())?;
    t.ir.inst(write_shared_op(ty), &[address, value]);
    Ok(())
}

/// `LDGSTS [Rs + imm20], [Ra.64 + URb + imm12], Pp`: an asynchronous copy from global to
/// shared memory, done here as a load and a store; a false predicate stores zeros. The two
/// layouts the C++ accepts: UGPR mode with `URb` at 64..70 and the predicate negated by bit
/// 90, and a memory descriptor at 64..70 with the predicate negated by bit 70.
fn ldgsts(t: &mut Translator<'_>) -> TResult {
    const UNIFORM_ADDRESS: u32 = 0x0810_1140;
    const DESCRIPTOR: u32 = 0x0c10_1000;
    let insn = t.insn;
    let mode = insn.bits(64, 32) as u32 & !0xe00;
    let (ureg, not_bit) = if mode & !0x0780_003f == UNIFORM_ADDRESS {
        (insn.ureg_at(64), 90)
    } else if mode & !0x0380_007f == DESCRIPTOR {
        (Reg::URZ, 70)
    } else {
        return Err(t.unsupported());
    };
    let (load, words) = match insn.bits(73, 3) {
        4 => (Op::LoadGlobal32, 1),
        5 => (Op::LoadGlobal64, 2),
        6 => (Op::LoadGlobal128, 4),
        _ => return Err(t.unsupported()),
    };
    let operands = GlobalAddress {
        reg: insn.src0(),
        wide: true,
        ureg,
        uwide: true,
    };
    let address = global_address(t, operands, insn.sbits(32, 12))?;
    let loaded = t.ir.inst(load, &[address]);
    let enable = t.pred(insn.pred_src(87, not_bit));
    let value = if words == 1 {
        zero_unless(t, enable, loaded)
    } else if enable == Value::U1(true) {
        loaded
    } else {
        let mut elements = [Value::U32(0); 4];
        for (i, element) in elements.iter_mut().take(words).enumerate() {
            let word = t.ir.composite_extract(loaded, i as u32);
            *element = zero_unless(t, enable, word);
        }
        t.ir.composite_construct(&elements[..words])
    };
    let shared = add_u32(t, insn.dst(), insn.sbits(44, 20) as u32);
    let store = match words {
        1 => MemType::B32,
        2 => MemType::B64,
        _ => MemType::B128,
    };
    t.ir.inst(write_shared_op(store), &[shared, value]);
    Ok(())
}

/// Atomic operation, bits 87..91 (87..90 for RED).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomOp {
    Add,
    Min,
    Max,
    Inc,
    Dec,
    And,
    Or,
    Xor,
    Exch,
}

impl AtomOp {
    fn decode(t: &Translator<'_>, width: u32) -> Result<AtomOp, TranslateError> {
        Ok(match t.insn.bits(87, width) {
            0 => AtomOp::Add,
            1 => AtomOp::Min,
            2 => AtomOp::Max,
            3 => AtomOp::Inc,
            4 => AtomOp::Dec,
            5 => AtomOp::And,
            6 => AtomOp::Or,
            7 => AtomOp::Xor,
            8 => AtomOp::Exch,
            _ => return Err(t.unsupported()),
        })
    }
}

/// Atomic operand type, bits 73..76.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomType {
    U32,
    S32,
    U64,
    F32,
    F16x2,
    S64,
    F64,
}

impl AtomType {
    fn decode(t: &Translator<'_>) -> Result<AtomType, TranslateError> {
        Ok(match t.insn.bits(73, 3) {
            0 => AtomType::U32,
            1 => AtomType::S32,
            2 => AtomType::U64,
            3 => AtomType::F32,
            4 => AtomType::F16x2,
            5 => AtomType::S64,
            6 => AtomType::F64,
            _ => return Err(t.unsupported()),
        })
    }
}

/// Applies `op` to the global word at `address` with `Rdata` and writes the old value to
/// `dest` (nothing for RZ).
fn global_atomic(
    t: &mut Translator<'_>,
    dest: Reg,
    data: Reg,
    address: Value,
    ty: AtomType,
    op: AtomOp,
) -> TResult {
    use AtomOp::*;
    const F32_CONTROL: FpControl = FpControl {
        no_contraction: false,
        rounding: FpRounding::Rn,
        fmz_mode: FmzMode::Ftz,
    };
    const F16_CONTROL: FpControl = FpControl {
        no_contraction: false,
        rounding: FpRounding::Rn,
        fmz_mode: FmzMode::DontCare,
    };
    let (signed, wide) = match ty {
        AtomType::U32 => (false, false),
        AtomType::S32 => (true, false),
        AtomType::U64 => (false, true),
        AtomType::S64 => (true, true),
        AtomType::F32 if op == Add => {
            let value = t.f(data);
            let old = t.ir.inst_flags(
                Op::GlobalAtomicAddF32,
                F32_CONTROL.to_flags(),
                &[address, value],
            );
            t.set_f(dest, old);
            return Ok(());
        }
        AtomType::F16x2 if matches!(op, Add | Min | Max) => {
            let opcode = match op {
                Add => Op::GlobalAtomicAddF16x2,
                Min => Op::GlobalAtomicMinF16x2,
                _ => Op::GlobalAtomicMaxF16x2,
            };
            let packed = t.x(data);
            let value = t.ir.unpack_float_2x16(packed);
            let old =
                t.ir.inst_flags(opcode, F16_CONTROL.to_flags(), &[address, value]);
            t.set_x(dest, old);
            return Ok(());
        }
        _ => return Err(t.unsupported()),
    };
    let opcode = match (op, signed, wide) {
        (Add, _, false) => Op::GlobalAtomicIAdd32,
        (Add, _, true) => Op::GlobalAtomicIAdd64,
        (Min, false, false) => Op::GlobalAtomicUMin32,
        (Min, true, false) => Op::GlobalAtomicSMin32,
        (Min, false, true) => Op::GlobalAtomicUMin64,
        (Min, true, true) => Op::GlobalAtomicSMin64,
        (Max, false, false) => Op::GlobalAtomicUMax32,
        (Max, true, false) => Op::GlobalAtomicSMax32,
        (Max, false, true) => Op::GlobalAtomicUMax64,
        (Max, true, true) => Op::GlobalAtomicSMax64,
        (Inc, _, false) => Op::GlobalAtomicInc32,
        (Dec, _, false) => Op::GlobalAtomicDec32,
        (And, _, false) => Op::GlobalAtomicAnd32,
        (And, _, true) => Op::GlobalAtomicAnd64,
        (Or, _, false) => Op::GlobalAtomicOr32,
        (Or, _, true) => Op::GlobalAtomicOr64,
        (Xor, _, false) => Op::GlobalAtomicXor32,
        (Xor, _, true) => Op::GlobalAtomicXor64,
        (Exch, _, false) => Op::GlobalAtomicExchange32,
        (Exch, _, true) => Op::GlobalAtomicExchange64,
        (Inc | Dec, _, true) => return Err(t.unsupported()),
    };
    if wide {
        let value = t.l(data)?;
        let old = t.ir.inst(opcode, &[address, value]);
        if !dest.is_zero() {
            t.set_l(dest, old)?;
        }
    } else {
        let value = t.x(data);
        let old = t.ir.inst(opcode, &[address, value]);
        t.set_x(dest, old);
    }
    Ok(())
}

/// `ATOMG Rd, [Ra + URb + imm], Rb` (and the generic ATOM).
fn atomg(t: &mut Translator<'_>) -> TResult {
    let ty = AtomType::decode(t)?;
    let op = AtomOp::decode(t, 4)?;
    let operands = global_operands(t, 64, 70)?;
    let address = global_address(t, operands, t.insn.sbits(40, 24))?;
    global_atomic(t, t.insn.dst(), t.insn.src1(), address, ty, op)
}

/// `RED [Ra + URb + imm], Rb`: an atomic without a result.
fn red(t: &mut Translator<'_>) -> TResult {
    let ty = AtomType::decode(t)?;
    let op = AtomOp::decode(t, 3)?;
    let operands = global_operands(t, 64, 90)?;
    let address = global_address(t, operands, t.insn.sbits(40, 24))?;
    global_atomic(t, Reg::RZ, t.insn.src1(), address, ty, op)
}

/// `ATOMG.CAS Rd, [Ra + imm], Rb, Rc`: stores `Rc` where the word equals `Rb`; `Rd`
/// receives the old word. 64-bit words take the compare and the data as register pairs, the
/// compare in `Rb` (32..40) and the data in `Rc` (64..72), as Mesa NAK's SM70 encoder places
/// `OpAtom`'s `cmpr` and `data` for `AtomOp::CmpExch`.
fn atomg_cas(t: &mut Translator<'_>) -> TResult {
    let wide = match AtomType::decode(t)? {
        AtomType::U32 | AtomType::S32 => false,
        AtomType::U64 | AtomType::S64 => true,
        _ => return Err(t.unsupported()),
    };
    let operands = global_operands(t, 64, 70)?;
    let address = global_address(t, operands, t.insn.sbits(40, 24))?;
    if wide {
        let compare = t.l(t.insn.src1())?;
        let value = t.l(t.insn.src2())?;
        let old = t.ir.inst(
            Op::GlobalAtomicCompareExchange64,
            &[address, compare, value],
        );
        return t.set_l(t.insn.dst(), old);
    }
    let compare = t.x(t.insn.src1());
    let value = t.x(t.insn.src2());
    let old = t.ir.inst(
        Op::GlobalAtomicCompareExchange32,
        &[address, compare, value],
    );
    t.set_x(t.insn.dst(), old);
    Ok(())
}

/// `ATOMS Rd, [Ra.Xn + URb + imm], Rb`: integer atomics; 64-bit words only exchange.
fn atoms(t: &mut Translator<'_>) -> TResult {
    use AtomOp::*;
    let ty = AtomType::decode(t)?;
    let op = AtomOp::decode(t, 4)?;
    let signed = ty == AtomType::S32;
    let wide = matches!(ty, AtomType::U64 | AtomType::S64);
    let opcode = match (ty, op) {
        (AtomType::U64 | AtomType::S64, Exch) => Op::SharedAtomicExchange64,
        (AtomType::U32 | AtomType::S32, _) => match op {
            Add => Op::SharedAtomicIAdd32,
            Min if signed => Op::SharedAtomicSMin32,
            Min => Op::SharedAtomicUMin32,
            Max if signed => Op::SharedAtomicSMax32,
            Max => Op::SharedAtomicUMax32,
            Inc => Op::SharedAtomicInc32,
            Dec => Op::SharedAtomicDec32,
            And => Op::SharedAtomicAnd32,
            Or => Op::SharedAtomicOr32,
            Xor => Op::SharedAtomicXor32,
            Exch => Op::SharedAtomicExchange32,
        },
        _ => return Err(t.unsupported()),
    };
    let ureg = address_ureg(t, 64);
    let address = byte_address(t, ureg, true);
    let (dest, data) = (t.insn.dst(), t.insn.src1());
    if wide {
        let value = t.l(data)?;
        let old = t.ir.inst(opcode, &[address, value]);
        return t.set_l(dest, old);
    }
    let value = t.x(data);
    let old = t.ir.inst(opcode, &[address, value]);
    t.set_x(dest, old);
    Ok(())
}

/// `ATOMS.CAS Rd, [Ra.Xn + imm], Rb, Rc`. 64-bit words take the compare and the data as
/// register pairs, the compare in `Rb` (32..40) and the data in `Rc` (64..72), as Mesa
/// NAK's SM70 encoder places `OpAtom`'s `cmpr` and `data` for a shared `AtomOp::CmpExch`.
fn atoms_cas(t: &mut Translator<'_>) -> TResult {
    let wide = match AtomType::decode(t)? {
        AtomType::U32 | AtomType::S32 => false,
        AtomType::U64 | AtomType::S64 => true,
        _ => return Err(t.unsupported()),
    };
    let address = byte_address(t, Reg::URZ, true);
    if wide {
        let compare = t.l(t.insn.src1())?;
        let value = t.l(t.insn.src2())?;
        let old = t.ir.inst(
            Op::SharedAtomicCompareExchange64,
            &[address, compare, value],
        );
        return t.set_l(t.insn.dst(), old);
    }
    let compare = t.x(t.insn.src1());
    let value = t.x(t.insn.src2());
    let old = t.ir.inst(
        Op::SharedAtomicCompareExchange32,
        &[address, compare, value],
    );
    t.set_x(t.insn.dst(), old);
    Ok(())
}

/// `ATOMS.POPC.INC RZ, [Ra.Xn + URb + imm]`: adds the number of active lanes, done as one
/// increment per lane. A returned value is unsupported.
fn atoms_popc_inc(t: &mut Translator<'_>) -> TResult {
    if !t.insn.dst().is_zero() {
        return Err(t.unsupported());
    }
    let ureg = if t.insn.bit(91) {
        t.insn.ureg_at(64)
    } else {
        Reg::URZ
    };
    let address = byte_address(t, ureg, true);
    t.ir.inst(Op::SharedAtomicIAdd32, &[address, Value::U32(1)]);
    Ok(())
}

/// `MEMBAR`: the CTA and SM scopes (76..79) order workgroup memory, wider scopes all memory.
fn membar(t: &mut Translator<'_>) -> TResult {
    if t.insn.bits(76, 3) <= 1 {
        t.ir.workgroup_memory_barrier();
    } else {
        t.ir.device_memory_barrier();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use lichii_common::settings::ResolutionScale;

    use super::super::tests::{TestEnv, encode, translate};
    use super::super::translate_block;
    use crate::ampere::decode::Form;
    use crate::ampere::instruction::Instruction;
    use crate::ampere::opcode::Opcode;
    use crate::backend::Bindings;
    use crate::ir::Program;
    use crate::profile::HostTranslateInfo;
    use crate::translate::TranslateOptions;
    use crate::{Profile, RuntimeInfo};

    fn f(offset: u32, value: u128) -> u128 {
        value << offset
    }

    fn with_local_memory(code: &[Instruction], size: u32) -> String {
        let mut env = TestEnv::new(code);
        env.local_memory = size;
        let mut prog = Program::new();
        let block = prog.new_block();
        translate_block(&mut prog, block, &env, 0, env.end()).unwrap();
        prog.dump()
    }

    #[test]
    fn ldc_b64_register_offset() {
        let insn = encode(
            Opcode::Ldc,
            Form::Bound,
            f(16, 2) | f(24, 4) | f(38, 0x10) | f(54, 3) | f(73, 5),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = IAdd32 %0, #0x10\n\
             \x20 %2 = GetCbufU32x2 #0x3, %1\n\
             \x20 %3 = CompositeExtractU32x2 %2, #0x0\n\
             \x20 SetRegister R2, %3\n\
             \x20 %4 = CompositeExtractU32x2 %2, #0x1\n\
             \x20 SetRegister R3, %4\n"
        );
    }

    #[test]
    fn ldc_s8_indexed_segmented() {
        let insn = encode(
            Opcode::Ldc,
            Form::Bound,
            f(16, 1) | f(24, 5) | f(38, 8) | f(54, 1) | f(73, 1) | f(78, 2),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R5\n\
             \x20 %1 = BitFieldUExtract %0, #0x10, #0x10\n\
             \x20 %2 = BitFieldUExtract %0, #0x0, #0x10\n\
             \x20 %3 = IAdd32 %1, #0x1\n\
             \x20 %4 = IAdd32 %2, #0x8\n\
             \x20 %5 = GetCbufS8 %3, %4\n\
             \x20 SetRegister R1, %5\n"
        );
    }

    #[test]
    fn ldc_rz_offset_is_unsigned() {
        let insn = encode(
            Opcode::Ldc,
            Form::Bound,
            f(16, 1) | f(24, 255) | f(38, 0x8010) | f(73, 4),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetCbufU32 #0x0, #0x8010\n\
             \x20 SetRegister R1, %0\n"
        );
    }

    #[test]
    fn uldc_b64_direct() {
        let insn = encode(
            Opcode::Uldc,
            Form::Bound,
            f(16, 4) | f(38, 0x118) | f(73, 5),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetCbufU32x2 #0x0, #0x118\n\
             \x20 %1 = CompositeExtractU32x2 %0, #0x0\n\
             \x20 SetRegister UR4, %1\n\
             \x20 %2 = CompositeExtractU32x2 %0, #0x1\n\
             \x20 SetRegister UR5, %2\n"
        );
    }

    #[test]
    fn uldc_s16_uniform_indexed() {
        let insn = encode(
            Opcode::Uldc,
            Form::UregIndexed,
            f(16, 5) | f(24, 6) | f(38, 0x20) | f(54, 2) | f(73, 3) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister UR6\n\
             \x20 %1 = IAdd32 %0, #0x20\n\
             \x20 %2 = GetCbufS16 #0x2, %1\n\
             \x20 SetRegister UR5, %2\n"
        );
    }

    #[test]
    fn ldcg_uniform_pointer_with_predicate() {
        let insn = encode(
            Opcode::Ldcg,
            Form::Fixed,
            f(16, 4) | f(24, 6) | f(38, 0x40) | f(73, 4) | f(87, 1) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister UR6\n\
             \x20 %1 = GetRegister UR7\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x40u64\n\
             \x20 %5 = LoadGlobal32 %4\n\
             \x20 %6 = GetPred UP1\n\
             \x20 %7 = SelectU32 %6, %5, #0x0\n\
             \x20 SetRegister UR4, %7\n"
        );
    }

    #[test]
    fn ldg_b64_register_pair() {
        let insn = encode(
            Opcode::Ldg,
            Form::Fixed,
            f(16, 4) | f(24, 2) | f(40, 8) | f(72, 1) | f(73, 5),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x8u64\n\
             \x20 %5 = LoadGlobal64 %4\n\
             \x20 %6 = CompositeExtractU32x2 %5, #0x0\n\
             \x20 SetRegister R4, %6\n\
             \x20 %7 = CompositeExtractU32x2 %5, #0x1\n\
             \x20 SetRegister R5, %7\n"
        );
    }

    #[test]
    fn ldg_u8_register_plus_uniform_pair_with_negated_load_predicate() {
        let insn = encode(
            Opcode::Ldg,
            Form::UniformAddress,
            f(16, 1)
                | f(24, 2)
                | f(32, 4)
                | f(40, 0xfffff0)
                | f(64, 6)
                | f(67, 1)
                | f(72, 1)
                | f(90, 1)
                | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister UR4\n\
             \x20 %5 = GetRegister UR5\n\
             \x20 %6 = CompositeConstructU32x2 %4, %5\n\
             \x20 %7 = PackUint2x32 %6\n\
             \x20 %8 = IAdd64 %3, %7\n\
             \x20 %9 = IAdd64 %8, #0xfffffffffffffff0u64\n\
             \x20 %10 = LoadGlobalU8 %9\n\
             \x20 %11 = GetPred P1\n\
             \x20 %12 = LogicalNot %11\n\
             \x20 %13 = SelectU32 %12, %10, #0x0\n\
             \x20 SetRegister R1, %13\n"
        );
    }

    #[test]
    fn narrow_global_accesses_compile_to_valid_spirv() {
        let load = encode(
            Opcode::Ldg,
            Form::UniformAddress,
            f(16, 1)
                | f(24, 2)
                | f(32, 4)
                | f(40, 0xfffff0)
                | f(64, 6)
                | f(67, 1)
                | f(72, 1)
                | f(90, 1)
                | f(91, 1),
        );
        let store = encode(
            Opcode::Stg,
            Form::UniformAddress,
            f(24, 255) | f(32, 1) | f(64, 4) | f(72, 1) | f(73, 3) | f(91, 1),
        );
        let env = super::super::integer::tests::fragment_env(&[load, store]);
        let words = crate::Profile {
            support_int64: true,
            ..crate::Profile::default()
        };
        let views = crate::Profile {
            support_int8: true,
            support_int16: true,
            support_storage_buffer_8bit: true,
            support_storage_buffer_16bit: true,
            ..words.clone()
        };
        let host = crate::profile::HostTranslateInfo {
            support_int64: true,
            ..crate::profile::HostTranslateInfo::default()
        };
        for (name, profile) in [
            ("ampere-narrow-global-words", &words),
            ("ampere-narrow-global-views", &views),
        ] {
            let options = crate::translate::TranslateOptions {
                profile,
                host: &host,
                resolution_scale: ResolutionScale::X1,
            };
            let module = crate::module::compile(
                &env,
                options,
                &crate::RuntimeInfo::default(),
                &mut crate::backend::Bindings::default(),
            )
            .expect("compiles");
            crate::backend::spirv::tests::validate(name, profile, &module.spirv);
        }
    }

    #[test]
    fn ldg_memory_descriptor_is_not_an_address() {
        let insn = encode(
            Opcode::Ldg,
            Form::UniformAddress,
            f(16, 1) | f(24, 2) | f(32, 4) | f(72, 1) | f(73, 4) | f(76, 1) | f(90, 1) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = LoadGlobal32 %3\n\
             \x20 SetRegister R1, %4\n"
        );
    }

    #[test]
    fn ldg_without_ugpr_mode_reads_bit_72_as_register_width() {
        let insn = encode(
            Opcode::Ldg,
            Form::UniformAddress,
            f(16, 1) | f(24, 2) | f(32, 63) | f(72, 1) | f(73, 4),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = LoadGlobal32 %3\n\
             \x20 SetRegister R1, %4\n"
        );
    }

    #[test]
    fn ldg_b128_uniform_pair_only() {
        let insn = encode(
            Opcode::Ldg,
            Form::UniformAddress,
            f(16, 8) | f(24, 255) | f(32, 4) | f(40, 0x20) | f(72, 1) | f(73, 6) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister UR4\n\
             \x20 %1 = GetRegister UR5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x20u64\n\
             \x20 %5 = LoadGlobal128 %4\n\
             \x20 %6 = CompositeExtractU32x4 %5, #0x0\n\
             \x20 SetRegister R8, %6\n\
             \x20 %7 = CompositeExtractU32x4 %5, #0x1\n\
             \x20 SetRegister R9, %7\n\
             \x20 %8 = CompositeExtractU32x4 %5, #0x2\n\
             \x20 SetRegister R10, %8\n\
             \x20 %9 = CompositeExtractU32x4 %5, #0x3\n\
             \x20 SetRegister R11, %9\n"
        );
    }

    #[test]
    fn ld_generic_is_global() {
        let insn = encode(
            Opcode::Ld,
            Form::UniformAddress,
            f(16, 1) | f(24, 2) | f(32, 63) | f(72, 1) | f(73, 4) | f(90, 1) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = LoadGlobal32 %3\n\
             \x20 SetRegister R1, %4\n"
        );
    }

    #[test]
    fn stg_b128_register_pair() {
        let insn = encode(
            Opcode::Stg,
            Form::Fixed,
            f(24, 2) | f(32, 8) | f(40, 0x10) | f(72, 1) | f(73, 6),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x10u64\n\
             \x20 %5 = GetRegister R8\n\
             \x20 %6 = GetRegister R9\n\
             \x20 %7 = GetRegister R10\n\
             \x20 %8 = GetRegister R11\n\
             \x20 %9 = CompositeConstructU32x4 %5, %6, %7, %8\n\
             \x20 WriteGlobal128 %4, %9\n"
        );
    }

    #[test]
    fn stg_s16_uniform_pair() {
        let insn = encode(
            Opcode::Stg,
            Form::UniformAddress,
            f(24, 255) | f(32, 3) | f(64, 4) | f(72, 1) | f(73, 3) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister UR4\n\
             \x20 %1 = GetRegister UR5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister R3\n\
             \x20 WriteGlobalS16 %3, %4\n"
        );
    }

    #[test]
    fn st_generic_is_global() {
        let insn = encode(
            Opcode::St,
            Form::UniformAddress,
            f(24, 2) | f(32, 3) | f(64, 63) | f(72, 1) | f(73, 4) | f(90, 1) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister R3\n\
             \x20 WriteGlobal32 %3, %4\n"
        );
    }

    #[test]
    fn lds_u16_scaled_register_plus_uniform() {
        let insn = encode(
            Opcode::Lds,
            Form::UniformAddress,
            f(16, 1) | f(24, 3) | f(32, 5) | f(40, 4) | f(73, 2) | f(78, 1) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R3\n\
             \x20 %1 = ShiftLeftLogical32 %0, #0x2\n\
             \x20 %2 = GetRegister UR5\n\
             \x20 %3 = IAdd32 %1, %2\n\
             \x20 %4 = IAdd32 %3, #0x4\n\
             \x20 %5 = LoadSharedU16 %4\n\
             \x20 SetRegister R1, %5\n"
        );
    }

    #[test]
    fn lds_b128_immediate() {
        let insn = encode(
            Opcode::Lds,
            Form::UniformAddress,
            f(16, 4) | f(24, 255) | f(40, 0x20) | f(73, 6),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = LoadSharedU128 #0x20\n\
             \x20 %1 = CompositeExtractU32x4 %0, #0x0\n\
             \x20 SetRegister R4, %1\n\
             \x20 %2 = CompositeExtractU32x4 %0, #0x1\n\
             \x20 SetRegister R5, %2\n\
             \x20 %3 = CompositeExtractU32x4 %0, #0x2\n\
             \x20 SetRegister R6, %3\n\
             \x20 %4 = CompositeExtractU32x4 %0, #0x3\n\
             \x20 SetRegister R7, %4\n"
        );
    }

    #[test]
    fn sts_b64_uniform_base() {
        let insn = encode(
            Opcode::Sts,
            Form::UniformAddress,
            f(24, 255) | f(32, 2) | f(40, 0x130) | f(64, 9) | f(73, 5) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister UR9\n\
             \x20 %1 = IAdd32 %0, #0x130\n\
             \x20 %2 = GetRegister R2\n\
             \x20 %3 = GetRegister R3\n\
             \x20 %4 = CompositeConstructU32x2 %2, %3\n\
             \x20 WriteSharedU64 %1, %4\n"
        );
    }

    #[test]
    fn sts_u8_negative_offset() {
        let insn = encode(
            Opcode::Sts,
            Form::Fixed,
            f(24, 6) | f(32, 7) | f(40, 0xfffffc),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R6\n\
             \x20 %1 = IAdd32 %0, #0xfffffffc\n\
             \x20 %2 = GetRegister R7\n\
             \x20 WriteSharedU8 %1, %2\n"
        );
    }

    #[test]
    fn ldgsts_b128_uniform_address_form() {
        let insn = encode(
            Opcode::Ldgsts,
            Form::Fixed,
            f(16, 5) | f(24, 2) | f(32, 0x10) | f(44, 0x40) | f(64, 0x0810_1d40 | 0x3f | (7 << 23)),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x10u64\n\
             \x20 %5 = LoadGlobal128 %4\n\
             \x20 %6 = GetRegister R5\n\
             \x20 %7 = IAdd32 %6, #0x40\n\
             \x20 WriteSharedU128 %7, %5\n"
        );
    }

    #[test]
    fn ldgsts_b32_descriptor_form_zero_fills_when_disabled() {
        let insn = encode(
            Opcode::Ldgsts,
            Form::Fixed,
            f(16, 5) | f(24, 2) | f(64, 0x0c10_1800 | 4 | (2 << 23) | 0x40),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = LoadGlobal32 %3\n\
             \x20 %5 = GetPred P2\n\
             \x20 %6 = LogicalNot %5\n\
             \x20 %7 = SelectU32 %6, %4, #0x0\n\
             \x20 %8 = GetRegister R5\n\
             \x20 WriteSharedU32 %8, %7\n"
        );
    }

    #[test]
    fn atomg_add_through_uniform_pointer() {
        let insn = encode(
            Opcode::Atomg,
            Form::UniformAddress,
            f(16, 1) | f(24, 255) | f(32, 3) | f(40, 8) | f(64, 0x081e_e140 | 4),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister UR4\n\
             \x20 %1 = GetRegister UR5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x8u64\n\
             \x20 %5 = GetRegister R3\n\
             \x20 %6 = GlobalAtomicIAdd32 %4, %5\n\
             \x20 SetRegister R1, %6\n"
        );
    }

    #[test]
    fn atomg_exch_u64() {
        let insn = encode(
            Opcode::Atomg,
            Form::Fixed,
            f(16, 2) | f(24, 4) | f(32, 6) | f(72, 1) | f(73, 2) | f(87, 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister R6\n\
             \x20 %5 = GetRegister R7\n\
             \x20 %6 = CompositeConstructU32x2 %4, %5\n\
             \x20 %7 = PackUint2x32 %6\n\
             \x20 %8 = GlobalAtomicExchange64 %3, %7\n\
             \x20 %9 = UnpackUint2x32 %8\n\
             \x20 %10 = CompositeExtractU32x2 %9, #0x0\n\
             \x20 SetRegister R2, %10\n\
             \x20 %11 = CompositeExtractU32x2 %9, #0x1\n\
             \x20 SetRegister R3, %11\n"
        );
    }

    #[test]
    fn atomg_add_f32() {
        let insn = encode(
            Opcode::Atomg,
            Form::Fixed,
            f(16, 1) | f(24, 4) | f(32, 6) | f(72, 1) | f(73, 3),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister R6\n\
             \x20 %5 = BitCastF32U32 %4\n\
             \x20 %6 = GlobalAtomicAddF32 %3, %5 (flags 0x12)\n\
             \x20 %7 = BitCastU32F32 %6\n\
             \x20 SetRegister R1, %7\n"
        );
    }

    #[test]
    fn atomg_min_f16x2() {
        let insn = encode(
            Opcode::Atomg,
            Form::Fixed,
            f(16, 1) | f(24, 4) | f(32, 6) | f(72, 1) | f(73, 4) | f(87, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister R6\n\
             \x20 %5 = UnpackFloat2x16 %4\n\
             \x20 %6 = GlobalAtomicMinF16x2 %3, %5 (flags 0x2)\n\
             \x20 SetRegister R1, %6\n"
        );
    }

    #[test]
    fn atomg_inc_u32() {
        let insn = encode(
            Opcode::Atomg,
            Form::Fixed,
            f(16, 1) | f(24, 4) | f(32, 6) | f(72, 1) | f(87, 3),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister R6\n\
             \x20 %5 = GlobalAtomicInc32 %3, %4\n\
             \x20 SetRegister R1, %5\n"
        );
    }

    #[test]
    fn atomg_cas_u32() {
        let insn = encode(
            Opcode::AtomgCas,
            Form::Fixed,
            f(16, 1) | f(24, 2) | f(32, 4) | f(40, 4) | f(64, 5) | f(72, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0x4u64\n\
             \x20 %5 = GetRegister R4\n\
             \x20 %6 = GetRegister R5\n\
             \x20 %7 = GlobalAtomicCompareExchange32 %4, %5, %6\n\
             \x20 SetRegister R1, %7\n"
        );
    }

    #[test]
    fn atomg_cas_u64_takes_the_compare_and_data_pairs() {
        let insn = encode(
            Opcode::AtomgCas,
            Form::Fixed,
            f(16, 2) | f(24, 4) | f(32, 6) | f(64, 8) | f(72, 1) | f(73, 2),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = GetRegister R6\n\
             \x20 %5 = GetRegister R7\n\
             \x20 %6 = CompositeConstructU32x2 %4, %5\n\
             \x20 %7 = PackUint2x32 %6\n\
             \x20 %8 = GetRegister R8\n\
             \x20 %9 = GetRegister R9\n\
             \x20 %10 = CompositeConstructU32x2 %8, %9\n\
             \x20 %11 = PackUint2x32 %10\n\
             \x20 %12 = GlobalAtomicCompareExchange64 %3, %7, %11\n\
             \x20 %13 = UnpackUint2x32 %12\n\
             \x20 %14 = CompositeExtractU32x2 %13, #0x0\n\
             \x20 SetRegister R2, %14\n\
             \x20 %15 = CompositeExtractU32x2 %13, #0x1\n\
             \x20 SetRegister R3, %15\n"
        );
    }

    #[test]
    fn atom_cas_s64_is_the_same_compare_exchange() {
        let insn = encode(
            Opcode::AtomCas,
            Form::Fixed,
            f(16, 2) | f(24, 4) | f(32, 6) | f(40, 0x10) | f(64, 8) | f(72, 1) | f(73, 5),
        );
        let dump = translate(&[insn]).unwrap();
        assert!(dump.contains("  %4 = IAdd64 %3, #0x10u64\n"), "{dump}");
        assert!(
            dump.contains("  %13 = GlobalAtomicCompareExchange64 %4, %8, %12\n"),
            "{dump}"
        );
    }

    #[test]
    fn red_or_s32() {
        let insn = encode(
            Opcode::Red,
            Form::UniformAddress,
            f(24, 2)
                | f(32, 6)
                | f(40, 0xfffff8)
                | f(64, 63)
                | f(72, 1)
                | f(73, 1)
                | f(87, 6)
                | f(90, 1)
                | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = IAdd64 %3, #0xfffffffffffffff8u64\n\
             \x20 %5 = GetRegister R6\n\
             \x20 %6 = GlobalAtomicOr32 %4, %5\n"
        );
    }

    #[test]
    fn atoms_max_s32_scaled() {
        let insn = encode(
            Opcode::Atoms,
            Form::Fixed,
            f(16, 1) | f(24, 2) | f(32, 3) | f(40, 4) | f(73, 1) | f(78, 1) | f(87, 2),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = ShiftLeftLogical32 %0, #0x2\n\
             \x20 %2 = IAdd32 %1, #0x4\n\
             \x20 %3 = GetRegister R3\n\
             \x20 %4 = SharedAtomicSMax32 %2, %3\n\
             \x20 SetRegister R1, %4\n"
        );
    }

    #[test]
    fn atoms_exch_u64() {
        let insn = encode(
            Opcode::Atoms,
            Form::Fixed,
            f(16, 2) | f(24, 255) | f(32, 4) | f(40, 8) | f(73, 2) | f(87, 8),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R4\n\
             \x20 %1 = GetRegister R5\n\
             \x20 %2 = CompositeConstructU32x2 %0, %1\n\
             \x20 %3 = PackUint2x32 %2\n\
             \x20 %4 = SharedAtomicExchange64 #0x8, %3\n\
             \x20 %5 = UnpackUint2x32 %4\n\
             \x20 %6 = CompositeExtractU32x2 %5, #0x0\n\
             \x20 SetRegister R2, %6\n\
             \x20 %7 = CompositeExtractU32x2 %5, #0x1\n\
             \x20 SetRegister R3, %7\n"
        );
    }

    #[test]
    fn atoms_exch_s64() {
        let insn = encode(
            Opcode::Atoms,
            Form::Fixed,
            f(16, 2) | f(24, 255) | f(32, 4) | f(40, 8) | f(73, 5) | f(87, 8),
        );
        let dump = translate(&[insn]).unwrap();
        assert!(
            dump.contains("  %4 = SharedAtomicExchange64 #0x8, %3\n"),
            "{dump}"
        );
    }

    /// Mesa NAK's SM70 encoder asserts "64-bit Shared atomics only support CmpExch or Exch",
    /// so ATOMS has no 64-bit arithmetic form to translate.
    #[test]
    fn atoms_arithmetic_64_is_rejected() {
        for ty in [2, 5] {
            for op in 0..8 {
                let insn = encode(
                    Opcode::Atoms,
                    Form::Fixed,
                    f(16, 2) | f(24, 255) | f(32, 4) | f(73, ty) | f(87, op),
                );
                assert!(translate(&[insn]).is_err(), "type {ty} op {op}");
            }
        }
    }

    #[test]
    fn atoms_inc_uniform_base() {
        let insn = encode(
            Opcode::Atoms,
            Form::UniformAddress,
            f(16, 1) | f(24, 2) | f(32, 3) | f(64, 7) | f(87, 3) | f(91, 1),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister UR7\n\
             \x20 %2 = IAdd32 %0, %1\n\
             \x20 %3 = GetRegister R3\n\
             \x20 %4 = SharedAtomicInc32 %2, %3\n\
             \x20 SetRegister R1, %4\n"
        );
    }

    #[test]
    fn atoms_cas_u32() {
        let insn = encode(
            Opcode::AtomsCas,
            Form::Fixed,
            f(16, 1) | f(24, 2) | f(32, 3) | f(64, 4),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = GetRegister R3\n\
             \x20 %2 = GetRegister R4\n\
             \x20 %3 = SharedAtomicCompareExchange32 %0, %1, %2\n\
             \x20 SetRegister R1, %3\n"
        );
    }

    #[test]
    fn atoms_cas_u64_takes_the_compare_and_data_pairs() {
        let insn = encode(
            Opcode::AtomsCas,
            Form::Fixed,
            f(16, 2) | f(24, 6) | f(32, 4) | f(64, 8) | f(73, 2) | f(78, 2),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R6\n\
             \x20 %1 = ShiftLeftLogical32 %0, #0x3\n\
             \x20 %2 = GetRegister R4\n\
             \x20 %3 = GetRegister R5\n\
             \x20 %4 = CompositeConstructU32x2 %2, %3\n\
             \x20 %5 = PackUint2x32 %4\n\
             \x20 %6 = GetRegister R8\n\
             \x20 %7 = GetRegister R9\n\
             \x20 %8 = CompositeConstructU32x2 %6, %7\n\
             \x20 %9 = PackUint2x32 %8\n\
             \x20 %10 = SharedAtomicCompareExchange64 %1, %5, %9\n\
             \x20 %11 = UnpackUint2x32 %10\n\
             \x20 %12 = CompositeExtractU32x2 %11, #0x0\n\
             \x20 SetRegister R2, %12\n\
             \x20 %13 = CompositeExtractU32x2 %11, #0x1\n\
             \x20 SetRegister R3, %13\n"
        );
    }

    #[test]
    fn atoms_popc_inc_increments_per_lane() {
        let insn = encode(
            Opcode::AtomsPopcInc,
            Form::Fixed,
            f(16, 255) | f(24, 2) | f(40, 8) | f(64, 0x0d00_003f) | f(78, 2),
        );
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = ShiftLeftLogical32 %0, #0x3\n\
             \x20 %2 = IAdd32 %1, #0x8\n\
             \x20 %3 = SharedAtomicIAdd32 %2, #0x1\n"
        );
    }

    #[test]
    fn membar_cta_orders_workgroup_memory() {
        let insn = encode(Opcode::Membar, Form::Fixed, 0);
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 WorkgroupMemoryBarrier\n"
        );
    }

    #[test]
    fn membar_gpu_orders_all_memory() {
        let insn = encode(Opcode::Membar, Form::Fixed, f(76, 2));
        assert_eq!(
            translate(&[insn]).unwrap(),
            "{0}:\n\
             \x20 DeviceMemoryBarrier\n"
        );
    }

    #[test]
    fn ldl_s8_register_offset() {
        let insn = encode(
            Opcode::Ldl,
            Form::UniformAddress,
            f(16, 1) | f(24, 3) | f(40, 2) | f(73, 1),
        );
        assert_eq!(
            with_local_memory(&[insn], 0x40),
            "{0}:\n\
             \x20 %0 = GetRegister R3\n\
             \x20 %1 = IAdd32 %0, #0x2\n\
             \x20 %2 = ShiftRightArithmetic32 %1, #0x2\n\
             \x20 %3 = ULessThan %1, #0x40\n\
             \x20 %4 = LoadLocal %2\n\
             \x20 %5 = SelectU32 %3, %4, #0x0\n\
             \x20 %6 = ShiftLeftLogical32 %1, #0x3\n\
             \x20 %7 = BitwiseAnd32 %6, #0x18\n\
             \x20 %8 = BitFieldSExtract %5, %7, #0x8\n\
             \x20 SetRegister R1, %8\n"
        );
    }

    #[test]
    fn ldl_b64_uniform_base() {
        let insn = encode(
            Opcode::Ldl,
            Form::UniformAddress,
            f(16, 4) | f(24, 255) | f(32, 5) | f(40, 8) | f(73, 5) | f(91, 1),
        );
        assert_eq!(
            with_local_memory(&[insn], 0x40),
            "{0}:\n\
             \x20 %0 = GetRegister UR5\n\
             \x20 %1 = IAdd32 %0, #0x8\n\
             \x20 %2 = ShiftRightArithmetic32 %1, #0x2\n\
             \x20 %3 = ULessThan %1, #0x40\n\
             \x20 %4 = LoadLocal %2\n\
             \x20 %5 = SelectU32 %3, %4, #0x0\n\
             \x20 SetRegister R4, %5\n\
             \x20 %6 = IAdd32 %2, #0x1\n\
             \x20 %7 = IAdd32 %1, #0x4\n\
             \x20 %8 = ULessThan %7, #0x40\n\
             \x20 %9 = LoadLocal %6\n\
             \x20 %10 = SelectU32 %8, %9, #0x0\n\
             \x20 SetRegister R5, %10\n"
        );
    }

    #[test]
    fn stl_s16_immediate_offset() {
        let insn = encode(
            Opcode::Stl,
            Form::Fixed,
            f(24, 255) | f(32, 2) | f(40, 6) | f(73, 3),
        );
        assert_eq!(
            with_local_memory(&[insn], 0x40),
            "{0}:\n\
             \x20 %0 = GetRegister R2\n\
             \x20 %1 = ShiftLeftLogical32 #0x6, #0x3\n\
             \x20 %2 = BitwiseAnd32 %1, #0x10\n\
             \x20 %3 = LoadLocal #0x1\n\
             \x20 %4 = BitFieldInsert %3, %0, %2, #0x10\n\
             \x20 WriteLocal #0x1, %4\n"
        );
    }

    #[test]
    fn stl_b128_register_plus_uniform() {
        let insn = encode(
            Opcode::Stl,
            Form::UniformAddress,
            f(24, 5) | f(32, 8) | f(64, 1) | f(73, 6) | f(91, 1),
        );
        assert_eq!(
            with_local_memory(&[insn], 0x40),
            "{0}:\n\
             \x20 %0 = GetRegister R5\n\
             \x20 %1 = GetRegister UR1\n\
             \x20 %2 = IAdd32 %0, %1\n\
             \x20 %3 = ShiftRightArithmetic32 %2, #0x2\n\
             \x20 %4 = GetRegister R8\n\
             \x20 WriteLocal %3, %4\n\
             \x20 %5 = IAdd32 %3, #0x1\n\
             \x20 %6 = GetRegister R9\n\
             \x20 WriteLocal %5, %6\n\
             \x20 %7 = IAdd32 %3, #0x2\n\
             \x20 %8 = GetRegister R10\n\
             \x20 WriteLocal %7, %8\n\
             \x20 %9 = IAdd32 %3, #0x3\n\
             \x20 %10 = GetRegister R11\n\
             \x20 WriteLocal %9, %10\n"
        );
    }

    #[test]
    fn stl_past_the_end_of_local_memory_is_dropped() {
        let insn = encode(
            Opcode::Stl,
            Form::Fixed,
            f(24, 255) | f(32, 2) | f(40, 0x40) | f(73, 4),
        );
        assert_eq!(with_local_memory(&[insn], 0x40), "{0}:\n");
    }

    #[test]
    fn cache_control_and_dependency_barriers_translate_to_nothing() {
        for (op, fields) in [
            (Opcode::Cctl, f(24, 2) | f(87, 4)),
            (Opcode::Depbar, f(32, 0x8000)),
            (Opcode::Errbar, 0),
            (Opcode::Ldgdepbar, 0),
        ] {
            assert_eq!(
                translate(&[encode(op, Form::Fixed, fields)]).unwrap(),
                "{0}:\n"
            );
        }
    }

    #[test]
    fn unsupported_encodings_report_errors() {
        use Form::*;
        use Opcode::*;
        let rejected = [
            encode(Ldc, Bound, f(73, 4) | f(78, 1)),
            encode(Ldc, Bound, f(73, 4) | f(78, 3)),
            encode(Ldc, Bound, f(73, 6)),
            encode(Ldc, Bound, f(16, 3) | f(73, 5)),
            encode(Ldc, UniformBindless, f(73, 4)),
            encode(Uldc, Bound, f(73, 4) | f(91, 1)),
            encode(Ldcg, Fixed, f(73, 6)),
            encode(Ldg, UniformAddress, f(24, 2) | f(32, 4) | f(73, 4)),
            encode(Ldg, Fixed, f(73, 7)),
            encode(Ldg, Fixed, f(16, 2) | f(73, 6)),
            encode(Ldg, Fixed, f(24, 3) | f(72, 1) | f(73, 4)),
            encode(Stg, Fixed, f(32, 5) | f(73, 5)),
            encode(Lds, UniformAddress, f(16, 1) | f(73, 5)),
            encode(Stl, Fixed, f(32, 2) | f(73, 6)),
            encode(Ldgsts, Fixed, 0),
            encode(Ldgsts, Fixed, f(64, 0x0810_1140 | 0xe00)),
            encode(Atomg, Fixed, f(73, 6)),
            encode(Atomg, Fixed, f(73, 2) | f(87, 3)),
            encode(Atomg, Fixed, f(87, 9)),
            encode(Atomg, Fixed, f(73, 3) | f(87, 1)),
            encode(Atomg, Fixed, f(73, 2) | f(87, 4)),
            encode(Atomg, Fixed, f(73, 5) | f(87, 3)),
            encode(Atomg, Fixed, f(73, 5) | f(87, 4)),
            encode(AtomgCas, Fixed, f(73, 3)),
            encode(AtomgCas, Fixed, f(73, 6)),
            encode(AtomgCas, Fixed, f(32, 5) | f(73, 2)),
            encode(AtomgCas, Fixed, f(64, 5) | f(73, 2)),
            encode(Atoms, Fixed, f(73, 2)),
            encode(Atoms, Fixed, f(73, 2) | f(87, 3)),
            encode(Atoms, Fixed, f(73, 2) | f(87, 4)),
            encode(Atoms, Fixed, f(73, 5) | f(87, 3)),
            encode(Atoms, Fixed, f(73, 5) | f(87, 4)),
            encode(Atoms, Fixed, f(73, 3)),
            encode(Atoms, Fixed, f(73, 3) | f(87, 8)),
            encode(Atoms, Fixed, f(73, 6)),
            encode(Atoms, Fixed, f(32, 5) | f(73, 2)),
            encode(AtomsCas, Fixed, f(73, 3)),
            encode(AtomsCas, Fixed, f(73, 6)),
            encode(AtomsCas, Fixed, f(32, 5) | f(73, 2)),
            encode(AtomsCas, Fixed, f(64, 5) | f(73, 5)),
            encode(AtomsPopcInc, Fixed, f(16, 1)),
            encode(Ldsm, UniformAddress, f(72, 3)),
            encode(Ldsm, UniformAddress, f(74, 1)),
            encode(Ldsm, UniformAddress, f(32, 1)),
            encode(Ldsm, UniformAddress, f(16, 1) | f(72, 1)),
        ];
        for insn in rejected {
            assert!(translate(&[insn]).is_err(), "{:#034x}", insn.0);
        }
    }

    /// Translates a compute program through every pass, checks that its global accesses
    /// became accesses to the storage buffer whose pointer is at c[0][0x210], and validates
    /// the compiled module.
    fn assert_storage_buffer_form(name: &str, code: &[Instruction]) {
        let env = TestEnv::new(code);
        let profile = Profile::default();
        let host = HostTranslateInfo {
            support_int64: true,
            ..HostTranslateInfo::default()
        };
        let options = TranslateOptions {
            profile: &profile,
            host: &host,
            resolution_scale: ResolutionScale::X1,
        };
        let program = crate::translate::translate(&env, options).unwrap();
        let buffers: Vec<_> = program
            .info
            .storage_buffers_descriptors
            .iter()
            .map(|d| (d.cbuf_index, d.cbuf_offset, d.is_written))
            .collect();
        assert_eq!(buffers, [(0, 0x210, true)]);
        let dump = program.ir.dump();
        assert!(!dump.contains("Global"), "{dump}");
        assert!(dump.contains("LoadStorage32"), "{dump}");
        assert!(dump.contains("WriteStorage32"), "{dump}");
        let mut bindings = Bindings::default();
        let module =
            crate::module::compile(&env, options, &RuntimeInfo::default(), &mut bindings).unwrap();
        crate::backend::spirv::tests::validate(name, &profile, &module.spirv);
    }

    /// The LDC that loads the storage buffer pointer at c[0][0x210] into R2:R3.
    fn ldc_storage_pointer() -> Instruction {
        encode(
            Opcode::Ldc,
            Form::Bound,
            f(16, 2) | f(24, 255) | f(38, 0x210) | f(73, 5),
        )
    }

    /// Compiles a compute program with 0x100 bytes of shared memory whose global accesses
    /// all go through the storage buffer at c[0][0x210], validates the module and returns its
    /// atomic compare-exchange and exchange counts.
    fn compile_atomics(
        name: &str,
        code: &[Instruction],
        profile: &Profile,
        host: &HostTranslateInfo,
    ) -> [usize; 2] {
        let mut env = TestEnv::new(code);
        env.shared_memory = 0x100;
        let options = TranslateOptions {
            profile,
            host,
            resolution_scale: ResolutionScale::X1,
        };
        let program = crate::translate::translate(&env, options).unwrap();
        let buffers: Vec<_> = program
            .info
            .storage_buffers_descriptors
            .iter()
            .map(|d| (d.cbuf_index, d.cbuf_offset, d.is_written))
            .collect();
        assert_eq!(buffers, [(0, 0x210, true)], "{name}");
        let dump = program.ir.dump();
        assert!(!dump.contains("Global"), "{name}: {dump}");
        let mut bindings = Bindings::default();
        let module =
            crate::module::compile(&env, options, &RuntimeInfo::default(), &mut bindings).unwrap();
        crate::backend::spirv::tests::validate(name, profile, &module.spirv);
        let module = rspirv::dr::load_words(&module.spirv).unwrap();
        [
            rspirv::spirv::Op::AtomicCompareExchange,
            rspirv::spirv::Op::AtomicExchange,
        ]
        .map(|op| {
            module
                .functions
                .iter()
                .flat_map(|f| &f.blocks)
                .flat_map(|b| &b.instructions)
                .filter(|i| i.class.opcode == op)
                .count()
        })
    }

    /// ATOMG.CAS.64 on the storage buffer, then ATOMS.CAS.64 of the old value into shared
    /// memory.
    fn global_and_shared_cas64() -> [Instruction; 4] {
        [
            ldc_storage_pointer(),
            encode(
                Opcode::AtomgCas,
                Form::Fixed,
                f(16, 8) | f(24, 2) | f(32, 4) | f(40, 0x10) | f(64, 6) | f(72, 1) | f(73, 2),
            ),
            encode(
                Opcode::AtomsCas,
                Form::Fixed,
                f(16, 10) | f(24, 255) | f(32, 4) | f(40, 8) | f(64, 8) | f(73, 2),
            ),
            encode(Opcode::Exit, Form::Fixed, f(87, 7)),
        ]
    }

    #[test]
    fn atomg_cas_64_and_atoms_cas_64_compile_natively_and_as_fallbacks() {
        let code = global_and_shared_cas64();
        let int64 = HostTranslateInfo {
            support_int64: true,
            ..HostTranslateInfo::default()
        };
        let native = Profile {
            supported_spirv: 0x0001_0400,
            support_int64: true,
            support_int64_atomics: true,
            support_shared_int64_atomics: true,
            support_explicit_workgroup_layout: true,
            ..Profile::default()
        };
        assert_eq!(
            compile_atomics("ampere-wide-cas-native", &code, &native, &int64),
            [2, 0]
        );
        let plain64 = Profile {
            support_int64: true,
            ..Profile::default()
        };
        assert_eq!(
            compile_atomics("ampere-wide-cas-plain", &code, &plain64, &int64),
            [0, 0]
        );
        assert_eq!(
            compile_atomics(
                "ampere-wide-cas-no-int64",
                &code,
                &Profile::default(),
                &HostTranslateInfo::default(),
            ),
            [0, 0]
        );
    }

    #[test]
    fn atomg_exch_64_alone_declares_the_u64_storage_view() {
        let code = [
            ldc_storage_pointer(),
            encode(
                Opcode::Atomg,
                Form::Fixed,
                f(16, 8) | f(24, 2) | f(32, 4) | f(40, 0x10) | f(72, 1) | f(73, 2) | f(87, 8),
            ),
            encode(Opcode::Exit, Form::Fixed, f(87, 7)),
        ];
        let profile = Profile {
            support_int64: true,
            support_int64_atomics: true,
            ..Profile::default()
        };
        let host = HostTranslateInfo {
            support_int64: true,
            ..HostTranslateInfo::default()
        };
        assert_eq!(
            compile_atomics("ampere-atomg-exch-64", &code, &profile, &host),
            [0, 1]
        );
    }

    #[test]
    fn ldc_pointer_with_ldg_and_stg_becomes_a_storage_buffer() {
        assert_storage_buffer_form(
            "ampere-ldc-pointer",
            &[
                encode(
                    Opcode::Ldc,
                    Form::Bound,
                    f(16, 2) | f(24, 255) | f(38, 0x210) | f(73, 5),
                ),
                encode(
                    Opcode::Ldg,
                    Form::Fixed,
                    f(24, 2) | f(40, 0x10) | f(72, 1) | f(73, 4),
                ),
                encode(
                    Opcode::Stg,
                    Form::Fixed,
                    f(24, 2) | f(40, 0x20) | f(72, 1) | f(73, 4),
                ),
                encode(Opcode::Exit, Form::Fixed, f(87, 7)),
            ],
        );
    }

    #[test]
    fn uldc_pointer_with_uniform_address_ldg_and_stg_becomes_a_storage_buffer() {
        let ugpr = f(24, 255) | f(72, 1) | f(73, 4) | f(91, 1);
        assert_storage_buffer_form(
            "ampere-uldc-pointer",
            &[
                encode(
                    Opcode::Uldc,
                    Form::Bound,
                    f(16, 4) | f(38, 0x210) | f(73, 5),
                ),
                encode(
                    Opcode::Ldg,
                    Form::UniformAddress,
                    ugpr | f(32, 4) | f(40, 0x10),
                ),
                encode(
                    Opcode::Stg,
                    Form::UniformAddress,
                    ugpr | f(64, 4) | f(40, 0x20),
                ),
                encode(Opcode::Exit, Form::Fixed, f(87, 7)),
            ],
        );
    }
}
