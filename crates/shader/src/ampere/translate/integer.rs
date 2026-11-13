//! Integer, logic, move and predicate operations, on both datapaths.
//!
//! Each uniform-datapath twin shares its operation's translator: the operand helpers read
//! uniform registers and [`pred_field`] names uniform predicates when the operation runs on the
//! uniform datapath.
//!
//! Source modifiers follow the operand's slot, not its role. The operand at bit 32 (a register,
//! constant or uniform register) has its absolute value at bit 62 and its negation at bit 63,
//! and an immediate there has neither; the register at bit 64 is negated by bit 75. Which
//! source sits in which slot depends on the form.
//!
//! Carries use the IR's carry pseudo-operation on each 32-bit addition, as the Maxwell IADD and
//! IADD3 translators do; Ampere keeps them in predicates instead of a condition code.

use super::{TResult, Translator};
use crate::TranslateError;
use crate::ampere::decode::{Form, decode};
use crate::ampere::instruction::CbufBank;
use crate::ampere::opcode::Opcode;
use crate::ir::{ICompare, Pred, Reg, Value};
use crate::maxwell::translate::logic_operation_three_input::{LUT, Step};
use crate::maxwell::translate::{BooleanOp, CompareOp, integer_compare, predicate_combine};

pub(super) fn translate(t: &mut Translator<'_>, op: Opcode) -> TResult {
    use Opcode::*;
    match op {
        Iadd3 | Uiadd3 => iadd3(t),
        Imad | Uimad => imad(t),
        ImadWide | UimadWide => imad_wide(t, true),
        ImadHi => imad_wide(t, false),
        Idp => idp(t),
        Iabs => iabs(t),
        Imnmx => imnmx(t),
        Isetp | Uisetp => isetp(t),
        Lop3 | Ulop3 => lop3(t),
        Shf | Ushf => shf(t),
        Prmt | Uprmt => prmt(t),
        Popc | Upopc => popc(t),
        Flo | Uflo => flo(t),
        Brev | Ubrev => brev(t),
        Lea | Ulea => lea(t),
        Sel | Usel => sel(t),
        Sgxt | Usgxt => sgxt(t),
        Bmsk | Ubmsk => bmsk(t),
        Mov | Umov => mov(t),
        P2r | Up2ur => p2r(t),
        R2p => r2p(t),
        Plop3 | Uplop3 => plop3(t),
        R2ur => r2ur(t),
        _ => Err(t.unsupported()),
    }
}

fn uniform(t: &Translator<'_>) -> bool {
    decode(t.insn).is_some_and(|d| d.opcode.is_uniform())
}

/// The predicate field at `offset`: a uniform predicate on the uniform datapath.
fn pred_field(t: &Translator<'_>, offset: u32) -> Pred {
    if uniform(t) {
        t.insn.upred_at(offset)
    } else {
        t.insn.pred_at(offset)
    }
}

/// Predicate `index` (0 to 6) of the datapath's predicate file.
fn pred_index(t: &Translator<'_>, index: u8) -> Pred {
    if uniform(t) {
        Pred::up(index)
    } else {
        Pred::p(index)
    }
}

fn read_pred(t: &mut Translator<'_>, offset: u32, not_bit: u32) -> Value {
    let pred = pred_field(t, offset);
    let negated = t.insn.bit(not_bit);
    t.pred((pred, negated))
}

fn write_pred(t: &mut Translator<'_>, offset: u32, value: Value) {
    let pred = pred_field(t, offset);
    t.ir.set_pred(pred, value);
}

fn writes_pred(t: &Translator<'_>, offset: u32) -> bool {
    t.insn.bits(offset, 3) != 7
}

/// Whether the predicate source at `offset` is `!PT`, the constant false.
fn is_false_pred(t: &Translator<'_>, offset: u32, not_bit: u32) -> bool {
    t.insn.bits(offset, 3) == 7 && t.insn.bit(not_bit)
}

fn slot32_is_imm(t: &Translator<'_>) -> bool {
    matches!(t.form, Form::ImmReg | Form::RegImm)
}

/// Whether the third source is the operand at bit 32 and the second the register at 64.
fn c_in_slot32(t: &Translator<'_>) -> bool {
    matches!(t.form, Form::RegImm | Form::RegCbuf | Form::RegUreg)
}

fn slot32_abs(t: &Translator<'_>) -> bool {
    !slot32_is_imm(t) && t.insn.bit(62)
}

fn slot32_neg(t: &Translator<'_>) -> bool {
    !slot32_is_imm(t) && t.insn.bit(63)
}

fn slot32_has_modifier(t: &Translator<'_>) -> bool {
    slot32_abs(t) || slot32_neg(t)
}

fn neg_b(t: &Translator<'_>) -> bool {
    if c_in_slot32(t) {
        t.insn.bit(75)
    } else {
        slot32_neg(t)
    }
}

fn neg_c(t: &Translator<'_>) -> bool {
    if c_in_slot32(t) {
        slot32_neg(t)
    } else {
        t.insn.bit(75)
    }
}

/// The register of the third source when it is the register at bit 64.
fn reg_c(t: &Translator<'_>) -> Option<Reg> {
    if c_in_slot32(t) {
        None
    } else if uniform(t) {
        Some(t.insn.usrc2())
    } else {
        Some(t.insn.src2())
    }
}

fn is_pair(reg: Reg) -> bool {
    reg.is_zero() || reg.0.is_multiple_of(2)
}

fn add(t: &mut Translator<'_>, a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::U32(0), v) | (v, Value::U32(0)) => v,
        _ => t.ir.iadd(a, b),
    }
}

fn sub(t: &mut Translator<'_>, a: Value, b: Value) -> Value {
    match b {
        Value::U32(0) => a,
        _ => t.ir.isub(a, b),
    }
}

fn negate(t: &mut Translator<'_>, v: Value) -> Value {
    match v {
        Value::U32(k) => Value::U32(k.wrapping_neg()),
        v => t.ir.ineg(v),
    }
}

fn invert(t: &mut Translator<'_>, v: Value) -> Value {
    match v {
        Value::U32(k) => Value::U32(!k),
        v => t.ir.bitwise_not(v),
    }
}

fn and(t: &mut Translator<'_>, a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::U32(0), _) | (_, Value::U32(0)) => Value::U32(0),
        (Value::U32(u32::MAX), v) | (v, Value::U32(u32::MAX)) => v,
        (Value::U32(x), Value::U32(y)) => Value::U32(x & y),
        _ => t.ir.bitwise_and(a, b),
    }
}

fn or(t: &mut Translator<'_>, a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::U32(0), v) | (v, Value::U32(0)) => v,
        (Value::U32(x), Value::U32(y)) => Value::U32(x | y),
        _ => t.ir.bitwise_or(a, b),
    }
}

fn shift_left(t: &mut Translator<'_>, v: Value, shift: u32) -> Value {
    match v {
        _ if shift == 0 => v,
        Value::U32(k) => Value::U32(k.checked_shl(shift).unwrap_or(0)),
        v => t.ir.shift_left_logical(v, Value::U32(shift)),
    }
}

fn shift_right(t: &mut Translator<'_>, v: Value, shift: u32) -> Value {
    match v {
        _ if shift == 0 => v,
        Value::U32(k) => Value::U32(k.checked_shr(shift).unwrap_or(0)),
        v => t.ir.shift_right_logical(v, Value::U32(shift)),
    }
}

/// `count` bits (1 to 32) of `v` from bit `offset`, with `offset + count` at most 32.
fn extract(t: &mut Translator<'_>, v: Value, offset: u32, count: u32, signed: bool) -> Value {
    match v {
        Value::U32(k) => {
            let field = (k >> offset) << (32 - count);
            Value::U32(if signed {
                ((field as i32) >> (32 - count)) as u32
            } else {
                field >> (32 - count)
            })
        }
        v => {
            t.ir.bit_field_extract(v, Value::U32(offset), Value::U32(count), signed)
        }
    }
}

fn select(t: &mut Translator<'_>, cond: Value, a: Value, b: Value) -> Value {
    match cond {
        Value::U1(true) => a,
        Value::U1(false) => b,
        cond => t.ir.select(cond, a, b),
    }
}

/// A carry predicate as the integer 0 or 1.
fn carry_value(t: &mut Translator<'_>, carry: Value) -> Value {
    match carry {
        Value::U1(c) => Value::U32(c as u32),
        c => t.ir.select(c, Value::U32(1), Value::U32(0)),
    }
}

/// The low 32 bits of `a * b`.
fn multiply(t: &mut Translator<'_>, a: Value, b: Value) -> Value {
    match (a, b) {
        (Value::U32(x), Value::U32(y)) => Value::U32(x.wrapping_mul(y)),
        (Value::U32(0), _) | (_, Value::U32(0)) => Value::U32(0),
        (Value::U32(1), v) | (v, Value::U32(1)) => v,
        (v, Value::U32(m)) | (Value::U32(m), v) if m.is_power_of_two() => {
            shift_left(t, v, m.trailing_zeros())
        }
        _ => t.ir.imul(a, b),
    }
}

/// The high word of the 64-bit product `a * b`, by base-2^16 long multiplication: every
/// partial sum fits 32 bits. The signed product subtracts `b` when `a` is negative and `a`
/// when `b` is.
fn multiply_high(t: &mut Translator<'_>, a: Value, b: Value, signed: bool) -> Value {
    let (a_low, a_high) = (extract(t, a, 0, 16, false), shift_right(t, a, 16));
    let (b_low, b_high) = (extract(t, b, 0, 16, false), shift_right(t, b, 16));
    let low_low = multiply(t, a_low, b_low);
    let low_high = multiply(t, a_low, b_high);
    let high_low = multiply(t, a_high, b_low);
    let high_high = multiply(t, a_high, b_high);
    let mut middle = shift_right(t, low_low, 16);
    for partial in [low_high, high_low] {
        let low = extract(t, partial, 0, 16, false);
        middle = add(t, middle, low);
    }
    let mut high = high_high;
    for partial in [low_high, high_low, middle] {
        let carried = shift_right(t, partial, 16);
        high = add(t, high, carried);
    }
    if signed {
        for (x, y) in [(a, b), (b, a)] {
            let correction = match x {
                Value::U32(k) if (k as i32) < 0 => y,
                Value::U32(_) => Value::U32(0),
                x => {
                    let negative = t.ir.icompare(ICompare::SLessThan, x, Value::U32(0));
                    t.ir.select(negative, y, Value::U32(0))
                }
            };
            high = sub(t, high, correction);
        }
    }
    high
}

/// A running 32-bit sum. A tracked sum keeps the carry out of each addition for the
/// carry-out predicates; its count of carries is that of the hardware's adder.
struct Sum {
    value: Value,
    carries: [Value; 6],
    count: usize,
    track: bool,
    ones: u32,
}

impl Sum {
    fn new(track: bool) -> Sum {
        Sum {
            value: Value::U32(0),
            carries: [Value::Void; 6],
            count: 0,
            track,
            ones: 0,
        }
    }

    fn add(&mut self, t: &mut Translator<'_>, term: Value) {
        if matches!(term, Value::U32(0)) {
            return;
        }
        if matches!(self.value, Value::U32(0)) {
            self.value = term;
            return;
        }
        self.value = t.ir.iadd(self.value, term);
        if self.track && self.count < self.carries.len() {
            self.carries[self.count] = t.ir.get_carry_from_op(self.value);
            self.count += 1;
        }
    }

    /// Adds a source that may be negated: the two's complement normally, the one's complement
    /// in an extended (.X) addition, whose carry-in completes it. A tracked two's complement
    /// adds `!v` and leaves the 1 for [`Sum::add_ones`], so it carries as the hardware does.
    fn add_source(&mut self, t: &mut Translator<'_>, v: Value, negated: bool, extended: bool) {
        let term = match (negated, extended) {
            (false, _) => v,
            (true, true) => invert(t, v),
            (true, false) if self.track => {
                self.ones += 1;
                invert(t, v)
            }
            (true, false) => negate(t, v),
        };
        self.add(t, term);
    }

    fn add_ones(&mut self, t: &mut Translator<'_>) {
        let ones = std::mem::take(&mut self.ones);
        self.add(t, Value::U32(ones));
    }

    fn add_carry(&mut self, t: &mut Translator<'_>, carry: Value) {
        let term = carry_value(t, carry);
        self.add(t, term);
    }

    /// Whether the additions carried at least once.
    fn carried(&self, t: &mut Translator<'_>) -> Value {
        match self.carries[..self.count] {
            [] => Value::U1(false),
            [carry] => carry,
            [first, second] => t.ir.logical_or(first, second),
            _ => {
                let total = self.carry_count(t);
                t.ir.icompare(ICompare::NotEqual, total, Value::U32(0))
            }
        }
    }

    /// Whether the additions carried at least twice.
    fn carried_twice(&self, t: &mut Translator<'_>) -> Value {
        match self.carries[..self.count] {
            [] | [_] => Value::U1(false),
            [first, second] => t.ir.logical_and(first, second),
            _ => {
                let total = self.carry_count(t);
                t.ir.icompare(ICompare::UGreaterThan, total, Value::U32(1))
            }
        }
    }

    fn carry_count(&self, t: &mut Translator<'_>) -> Value {
        let mut total = Value::U32(0);
        for &carry in &self.carries[..self.count] {
            let term = carry_value(t, carry);
            total = add(t, total, term);
        }
        total
    }
}

/// IADD3: `a + b + c`, each optionally negated (72, and the slot bits), with carry-out
/// predicates at 81 (one carry or more) and 84 (two). IADD3.X (74) adds the carry-in predicates
/// at 87 and 77 and negates by one's complement.
fn iadd3(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let extended = insn.bit(74);
    if insn.bit(73)
        || insn.bit(76)
        || slot32_abs(t)
        || (!extended && !(is_false_pred(t, 87, 90) && is_false_pred(t, 77, 80)))
    {
        return Err(t.unsupported());
    }
    let negated = [insn.bit(72), neg_b(t), neg_c(t)];
    let srcs = [t.src_a(), t.src_b()?, t.src_c()?];
    let mut sum = Sum::new(writes_pred(t, 81) || writes_pred(t, 84));
    for (src, negated) in srcs.into_iter().zip(negated) {
        sum.add_source(t, src, negated, extended);
    }
    sum.add_ones(t);
    if extended {
        for (offset, not_bit) in [(87, 90), (77, 80)] {
            let carry = read_pred(t, offset, not_bit);
            sum.add_carry(t, carry);
        }
    }
    let dst = t.reg_d();
    t.set_x(dst, sum.value);
    if writes_pred(t, 81) {
        let once = sum.carried(t);
        write_pred(t, 81, once);
    }
    if writes_pred(t, 84) {
        let twice = sum.carried_twice(t);
        write_pred(t, 84, twice);
    }
    Ok(())
}

/// IMAD: the low word of `a * b + c`; IMAD.MOV, IMAD.SHL and IMAD.IADD are this with a zero,
/// power-of-two or unit factor. IMAD.X (74) adds the carry predicate at 87 and inverts a
/// negated `c` instead of negating it.
fn imad(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let extended = insn.bit(74);
    if insn.bit(72) || slot32_abs(t) || writes_pred(t, 81) {
        return Err(t.unsupported());
    }
    let (negate_b, negate_c) = (neg_b(t), neg_c(t));
    let (a, b, c) = (t.src_a(), t.src_b()?, t.src_c()?);
    let b = if negate_b { negate(t, b) } else { b };
    let c = match (negate_c, extended) {
        (false, _) => c,
        (true, false) => negate(t, c),
        (true, true) => invert(t, c),
    };
    let product = multiply(t, a, b);
    let mut result = add(t, product, c);
    if extended {
        let carry = read_pred(t, 87, 90);
        let carry = carry_value(t, carry);
        result = add(t, result, carry);
    }
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// The third source as a 64-bit value in two words: an aligned register pair or two constant
/// buffer words.
fn src_c_pair(t: &mut Translator<'_>) -> Result<(Value, Value), TranslateError> {
    let reg = match t.form {
        Form::RegCbuf => {
            let operand = t.insn.cbuf();
            let CbufBank::Index(index) = operand.bank else {
                return Err(t.unsupported());
            };
            let pair =
                t.ir.get_cbuf_sized(Value::U32(index), Value::U32(operand.offset), 64, false);
            let low = t.ir.composite_extract(pair, 0);
            let high = t.ir.composite_extract(pair, 1);
            return Ok((low, high));
        }
        Form::RegUreg => t.insn.usrc1(),
        _ => reg_c(t).ok_or_else(|| t.unsupported())?,
    };
    if !is_pair(reg) {
        return Err(t.unsupported());
    }
    Ok((t.x(reg), t.x(reg.offset(1))))
}

/// IMAD.WIDE (`wide`) writes the 64-bit `a * b + c` to a register pair, IMAD.HI its high word;
/// `c` is 64 bits, the product signed when bit 73 is set. The carry out of the 64-bit addition
/// goes to the predicate at 81.
fn imad_wide(t: &mut Translator<'_>, wide: bool) -> TResult {
    let insn = t.insn;
    let dst = t.reg_d();
    if insn.bit(72) || insn.bit(74) || insn.bit(75) || slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    if wide && !is_pair(dst) {
        return Err(t.unsupported());
    }
    let signed = insn.bit(73);
    let (a, b) = (t.src_a(), t.src_b()?);
    let (c_low, c_high) = src_c_pair(t)?;
    let low = if wide || !matches!(c_low, Value::U32(0)) {
        multiply(t, a, b)
    } else {
        Value::U32(0)
    };
    let high = multiply_high(t, a, b, signed);
    let mut sum_low = Sum::new(true);
    sum_low.add(t, low);
    sum_low.add(t, c_low);
    let carry = sum_low.carried(t);
    let mut sum_high = Sum::new(writes_pred(t, 81));
    sum_high.add(t, high);
    sum_high.add(t, c_high);
    sum_high.add_carry(t, carry);
    let carry_out = sum_high.carried(t);
    if wide {
        t.set_x(dst, sum_low.value);
        t.set_x(dst.offset(1), sum_high.value);
    } else {
        t.set_x(dst, sum_high.value);
    }
    write_pred(t, 81, carry_out);
    Ok(())
}

/// IDP.4A: the dot product of the bytes of `a` and `b` plus `c`. IDP.2A (76) takes the two
/// halves of `a` and the low two bytes of `b`. Bits 73 and 74 make `a`'s and `b`'s parts signed.
fn idp(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if insn.bit(72) || insn.bit(75) || slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let (a_signed, b_signed) = (insn.bit(73), insn.bit(74));
    let (a, b, c) = (t.src_a(), t.src_b()?, t.src_c()?);
    let (parts, a_bits) = if insn.bit(76) { (2, 16) } else { (4, 8) };
    let mut result = Value::U32(0);
    for i in 0..parts {
        let x = extract(t, a, i * a_bits, a_bits, a_signed);
        let y = extract(t, b, i * 8, 8, b_signed);
        let product = multiply(t, x, y);
        result = add(t, result, product);
    }
    let result = add(t, result, c);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

fn iabs(t: &mut Translator<'_>) -> TResult {
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let src = t.src_b()?;
    let result = t.ir.iabs(src);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// IMNMX: the minimum when the predicate at 87 holds, else the maximum; signed with bit 73.
fn imnmx(t: &mut Translator<'_>) -> TResult {
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let signed = t.insn.bit(73);
    let (a, b) = (t.src_a(), t.src_b()?);
    let min = read_pred(t, 87, 90);
    let result = match min {
        Value::U1(true) => t.ir.imin(a, b, signed),
        Value::U1(false) => t.ir.imax(a, b, signed),
        min => {
            let lower = t.ir.imin(a, b, signed);
            let upper = t.ir.imax(a, b, signed);
            t.ir.select(min, lower, upper)
        }
    };
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// ISETP: compares `a` with `b` (76..79, signed with 73) and combines the result and its
/// negation with the predicate at 87 (74..76) into the predicates at 81 and 84. ISETP.EX (72)
/// continues a wider compare: when the words are equal the low words' result, the predicate at
/// 68, decides.
fn isetp(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let Some(bop) = BooleanOp::from_bits(insn.bits(74, 2)) else {
        return Err(t.unsupported());
    };
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let signed = insn.bit(73);
    let op = CompareOp::from_bits(insn.bits(76, 3));
    let (a, b) = (t.src_a(), t.src_b()?);
    let mut comparison = integer_compare(&mut t.ir, a, b, op, signed);
    if insn.bit(72) && !matches!(op, CompareOp::False | CompareOp::True) {
        let low = read_pred(t, 68, 71);
        let equal = t.ir.icompare(ICompare::Equal, a, b);
        comparison = t.ir.select(equal, low, comparison);
    }
    let accumulator = read_pred(t, 87, 90);
    if writes_pred(t, 81) {
        let result = combine(t, comparison, accumulator, bop);
        write_pred(t, 81, result);
    }
    if writes_pred(t, 84) {
        let inverse = t.ir.logical_not(comparison);
        let result = combine(t, inverse, accumulator, bop);
        write_pred(t, 84, result);
    }
    Ok(())
}

/// Combines a predicate with the accumulator, dropping an accumulator that leaves it unchanged.
fn combine(t: &mut Translator<'_>, value: Value, accumulator: Value, bop: BooleanOp) -> Value {
    match (bop, accumulator) {
        (BooleanOp::And, Value::U1(true)) | (BooleanOp::Or | BooleanOp::Xor, Value::U1(false)) => {
            value
        }
        _ => predicate_combine(&mut t.ir, value, accumulator, bop),
    }
}

/// Evaluates the three-input truth table `lut` on `srcs`, bitwise on registers or logically on
/// predicates, with the Maxwell LOP3 translator's expression for each table.
fn apply_lut(t: &mut Translator<'_>, srcs: [Value; 3], lut: u8, logical: bool) -> Value {
    let mut stack = [Value::Void; 8];
    let mut depth = 0;
    for &step in LUT[lut as usize] {
        let value = match step {
            Step::A => srcs[0],
            Step::B => srcs[1],
            Step::C => srcs[2],
            Step::Zero if logical => Value::U1(false),
            Step::Zero => Value::U32(0),
            Step::Ones if logical => Value::U1(true),
            Step::Ones => Value::U32(u32::MAX),
            Step::Not => {
                depth -= 1;
                if logical {
                    t.ir.logical_not(stack[depth])
                } else {
                    t.ir.bitwise_not(stack[depth])
                }
            }
            Step::And | Step::Or | Step::Xor => {
                depth -= 2;
                let (lhs, rhs) = (stack[depth], stack[depth + 1]);
                match (step, logical) {
                    (Step::And, true) => t.ir.logical_and(lhs, rhs),
                    (Step::Or, true) => t.ir.logical_or(lhs, rhs),
                    (_, true) => t.ir.logical_xor(lhs, rhs),
                    (Step::And, false) => t.ir.bitwise_and(lhs, rhs),
                    (Step::Or, false) => t.ir.bitwise_or(lhs, rhs),
                    (_, false) => t.ir.bitwise_xor(lhs, rhs),
                }
            }
        };
        stack[depth] = value;
        depth += 1;
    }
    stack[0]
}

/// LOP3.LUT: the truth table at 72..80 applied to `a`, `b` and `c`. The predicate at 81, when
/// written, is whether the result is non-zero (the form with a `!PT` input and no .PAND).
fn lop3(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let writes = writes_pred(t, 81);
    if slot32_has_modifier(t) || (writes && (insn.bit(80) || !is_false_pred(t, 87, 90))) {
        return Err(t.unsupported());
    }
    let srcs = [t.src_a(), t.src_b()?, t.src_c()?];
    let result = apply_lut(t, srcs, insn.bits(72, 8) as u8, false);
    let dst = t.reg_d();
    t.set_x(dst, result);
    if writes {
        let non_zero = t.ir.icompare(ICompare::NotEqual, result, Value::U32(0));
        write_pred(t, 81, non_zero);
    }
    Ok(())
}

/// PLOP3.LUT: two truth tables over the predicates at 87, 77 and 68 (a uniform predicate with
/// bit 67), one at 64..67 and 72..77 into the predicate at 81, one at 16..24 into 84.
fn plop3(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let third = if !uniform(t) && insn.bit(67) {
        insn.upred_at(68)
    } else {
        pred_field(t, 68)
    };
    let srcs = [
        read_pred(t, 87, 90),
        read_pred(t, 77, 80),
        t.pred((third, insn.bit(71))),
    ];
    let first_lut = insn.bits(64, 3) | (insn.bits(72, 5) << 3);
    for (offset, lut) in [(81, first_lut), (84, insn.bits(16, 8))] {
        if writes_pred(t, offset) {
            let result = apply_lut(t, srcs, lut as u8, true);
            write_pred(t, offset, result);
        }
    }
    Ok(())
}

/// SHF: shifts the 64-bit `c:a` by `b` and keeps the low word, or the high word with bit 80.
/// Bits 73..75 pick the type (S64, U64, S32, U32): its width bounds the shift, clamped or, with
/// bit 75, wrapped, and a signed right shift (76) is arithmetic.
fn shf(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if insn.bit(72) || slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let data_type = insn.bits(73, 2);
    let (wide, signed) = (data_type < 2, data_type.is_multiple_of(2));
    let (wrap, right, high) = (insn.bit(75), insn.bit(76), insn.bit(80));
    let (low, shift, high_word) = (t.src_a(), t.src_b()?, t.src_c()?);
    let pair = t.ir.composite_construct(&[low, high_word]);
    let value = t.ir.pack_uint_2x32(pair);
    let (width_mask, limit) = if wide { (63, 63) } else { (31, 32) };
    let amount = match (shift, wrap) {
        (Value::U32(k), true) => Value::U32(k & width_mask),
        (Value::U32(k), false) => Value::U32(k.min(limit)),
        (shift, true) => t.ir.bitwise_and(shift, Value::U32(width_mask)),
        (shift, false) => t.ir.imin(shift, Value::U32(limit), false),
    };
    let arithmetic = right && signed;
    let mut shifted = match (right, arithmetic) {
        (false, _) => t.ir.shift_left_logical(value, amount),
        (true, true) => t.ir.shift_right_arithmetic(value, amount),
        (true, false) => t.ir.shift_right_logical(value, amount),
    };
    if wide && !wrap && !arithmetic {
        shifted = match shift {
            Value::U32(k) if k < 64 => shifted,
            Value::U32(_) => Value::U64(0),
            shift => {
                let in_range = t.ir.icompare(ICompare::ULessThan, shift, Value::U32(64));
                t.ir.select(in_range, shifted, Value::U64(0))
            }
        };
    }
    let words = t.ir.unpack_uint_2x32(shifted);
    let result = t.ir.composite_extract(words, high as u32);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// Index-mode selectors of the PRMT modes 1..=6 (F4E, B4E, RC8, ECL, ECR, RC16; NAK's
/// `PrmtMode`), one per value of the selector's low two bits, from the PTX `prmt` tables.
const PRMT_MODE_SELECTORS: [[u32; 4]; 6] = [
    [0x3210, 0x4321, 0x5432, 0x6543],
    [0x5670, 0x6701, 0x7012, 0x0123],
    [0x0000, 0x1111, 0x2222, 0x3333],
    [0x3210, 0x3211, 0x3222, 0x3333],
    [0x0000, 0x1110, 0x2210, 0x3210],
    [0x1010, 0x3232, 0x1010, 0x3232],
];

/// PRMT: in the default index mode each nibble of the selector `b` picks a byte of `c:a`, its
/// top bit replicating the byte's sign. The other modes (72..75) pick a fixed byte pattern by
/// the selector's low two bits, done as the equivalent index-mode selector.
fn prmt(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let mode = insn.bits(72, 4) as usize;
    if mode > PRMT_MODE_SELECTORS.len() || slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let (a, selector, c) = (t.src_a(), t.src_b()?, t.src_c()?);
    let selector = match (mode, selector) {
        (0, selector) => selector,
        (mode, Value::U32(selector)) => {
            Value::U32(PRMT_MODE_SELECTORS[mode - 1][(selector & 3) as usize])
        }
        (mode, selector) => {
            let [s0, s1, s2, s3] = PRMT_MODE_SELECTORS[mode - 1].map(Value::U32);
            let low = t.ir.bitwise_and(selector, Value::U32(1));
            let odd = t.ir.icompare(ICompare::NotEqual, low, Value::U32(0));
            let high = t.ir.bitwise_and(selector, Value::U32(2));
            let upper = t.ir.icompare(ICompare::NotEqual, high, Value::U32(0));
            let lower_pair = t.ir.select(odd, s1, s0);
            let upper_pair = t.ir.select(odd, s3, s2);
            t.ir.select(upper, upper_pair, lower_pair)
        }
    };
    let mut result = Value::U32(0);
    for byte in 0..4u32 {
        let picked = match selector {
            Value::U32(selector) => {
                let nibble = (selector >> (byte * 4)) & 0xf;
                let source = if nibble & 4 == 0 { a } else { c };
                let offset = (nibble & 3) * 8;
                if nibble & 8 == 0 {
                    extract(t, source, offset, 8, false)
                } else {
                    extract(t, source, offset + 7, 1, true)
                }
            }
            selector => {
                let nibble =
                    t.ir.bit_field_extract(selector, Value::U32(byte * 4), Value::U32(4), false);
                let index = t.ir.bitwise_and(nibble, Value::U32(7));
                let from_a = t.ir.icompare(ICompare::ULessThan, index, Value::U32(4));
                let source = t.ir.select(from_a, a, c);
                let byte_index = t.ir.bitwise_and(index, Value::U32(3));
                let offset = t.ir.shift_left_logical(byte_index, Value::U32(3));
                let value = t.ir.bit_field_extract(source, offset, Value::U32(8), false);
                let sign_offset = t.ir.iadd(offset, Value::U32(7));
                let sign =
                    t.ir.bit_field_extract(source, sign_offset, Value::U32(1), true);
                let replicate_bit = t.ir.bitwise_and(nibble, Value::U32(8));
                let replicate =
                    t.ir.icompare(ICompare::NotEqual, replicate_bit, Value::U32(0));
                t.ir.select(replicate, sign, value)
            }
        };
        result = match (result, picked) {
            (Value::U32(r), Value::U32(p)) => Value::U32(r | ((p & 0xff) << (byte * 8))),
            _ => {
                t.ir.bit_field_insert(result, picked, Value::U32(byte * 8), Value::U32(8))
            }
        };
    }
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// POPC: the set bits of `b`, inverted first with bit 63.
fn popc(t: &mut Translator<'_>) -> TResult {
    if slot32_abs(t) {
        return Err(t.unsupported());
    }
    let src = t.src_b()?;
    let src = if slot32_neg(t) { invert(t, src) } else { src };
    let result = t.ir.bit_count(src);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// FLO: the index of the highest set bit of `b` (inverted first with bit 63; for a signed
/// value, 73, the highest bit unlike the sign), or its distance from bit 31 with .SH (74); -1
/// when there is none.
fn flo(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    if slot32_abs(t) || writes_pred(t, 81) {
        return Err(t.unsupported());
    }
    let src = t.src_b()?;
    let src = if slot32_neg(t) { invert(t, src) } else { src };
    let mut result = if insn.bit(73) {
        t.ir.find_s_msb(src)
    } else {
        t.ir.find_u_msb(src)
    };
    if insn.bit(74) {
        let not_found = t.ir.icompare(ICompare::Equal, result, Value::U32(u32::MAX));
        let flipped = t.ir.bitwise_xor(result, Value::U32(31));
        result = t.ir.select(not_found, result, flipped);
    }
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

fn brev(t: &mut Translator<'_>) -> TResult {
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let src = t.src_b()?;
    let result = t.ir.bit_reverse(src);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// LEA: `(a << s) + b` with `s` at 75..80, or with .HI (80) the high word of `c:a << s`, where
/// .SX32 (73) makes `c` the sign of `a`. Bit 72 negates the shifted value; LEA.X (74) adds the
/// carry predicate at 87 and inverts instead. The carry out goes to the predicate at 81.
fn lea(t: &mut Translator<'_>) -> TResult {
    let insn = t.insn;
    let (extended, high, sign_extend) = (insn.bit(74), insn.bit(80), insn.bit(73));
    let shift = insn.bits(75, 5) as u32;
    let sign_extends_zero_c = high && reg_c(t).is_some_and(Reg::is_zero);
    if slot32_abs(t) || (c_in_slot32(t) && slot32_neg(t)) || (sign_extend && !sign_extends_zero_c) {
        return Err(t.unsupported());
    }
    let negate_b = !c_in_slot32(t) && slot32_neg(t);
    let (a, b) = (t.src_a(), t.src_b()?);
    let shifted = if high {
        let a_high = if sign_extend {
            t.ir.shift_right_arithmetic(a, Value::U32(31))
        } else {
            t.src_c()?
        };
        if shift == 0 {
            a_high
        } else {
            let low_part = shift_right(t, a, 32 - shift);
            let high_part = shift_left(t, a_high, shift);
            or(t, low_part, high_part)
        }
    } else {
        shift_left(t, a, shift)
    };
    let mut sum = Sum::new(writes_pred(t, 81));
    sum.add_source(t, shifted, insn.bit(72), extended);
    sum.add_source(t, b, negate_b, extended);
    sum.add_ones(t);
    if extended {
        let carry = read_pred(t, 87, 90);
        sum.add_carry(t, carry);
    }
    let carry_out = sum.carried(t);
    let dst = t.reg_d();
    t.set_x(dst, sum.value);
    write_pred(t, 81, carry_out);
    Ok(())
}

/// SEL: `a` when the predicate at 87 holds, else `b`.
fn sel(t: &mut Translator<'_>) -> TResult {
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let (a, b) = (t.src_a(), t.src_b()?);
    let cond = read_pred(t, 87, 90);
    let result = select(t, cond, a, b);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// SGXT: the low `b` bits of `a`, sign-extended with bit 73; at least 32 bits keeps `a`. The
/// wrapping form (75) is not supported.
fn sgxt(t: &mut Translator<'_>) -> TResult {
    if t.insn.bit(75) || slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let signed = t.insn.bit(73);
    let (a, bits) = (t.src_a(), t.src_b()?);
    let count = match bits {
        Value::U32(k) => Value::U32(k.min(32)),
        bits => t.ir.imin(bits, Value::U32(32), false),
    };
    let result = t.ir.bit_field_extract(a, Value::U32(0), count, signed);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// BMSK: a mask of `b` ones starting at bit `a`, both clamped to 32 or, with bit 75, wrapped
/// to 0..31. It is built in 64 bits so a full-width mask and a start of 32 stay defined.
fn bmsk(t: &mut Translator<'_>) -> TResult {
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let wrap = t.insn.bit(75);
    let (start, width) = (t.src_a(), t.src_b()?);
    let [start, width] = [start, width].map(|v| match (v, wrap) {
        (Value::U32(k), true) => Value::U32(k & 31),
        (Value::U32(k), false) => Value::U32(k.min(32)),
        (v, true) => t.ir.bitwise_and(v, Value::U32(31)),
        (v, false) => t.ir.imin(v, Value::U32(32), false),
    });
    let result = match (start, width) {
        (Value::U32(s), Value::U32(w)) => Value::U32((((1u64 << w) - 1) << s) as u32),
        _ => {
            let one = Value::U64(1);
            let bit = t.ir.shift_left_logical(one, width);
            let ones = t.ir.isub(bit, one);
            let mask = t.ir.shift_left_logical(ones, start);
            t.ir.uconvert(32, mask)
        }
    };
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// MOV and UMOV: the second source into the destination. A MOV restricted to some lanes of a
/// quad (72..76 other than all four) is not supported.
fn mov(t: &mut Translator<'_>) -> TResult {
    if (!uniform(t) && t.insn.bits(72, 4) != 0xf) || slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let src = t.src_b()?;
    let dst = t.reg_d();
    t.set_x(dst, src);
    Ok(())
}

/// P2R and UP2UR: the predicates selected by the mask `b` packed into the byte of `a` that
/// bits 76..78 pick, keeping `a`'s other bits.
fn p2r(t: &mut Translator<'_>) -> TResult {
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let shift = t.insn.bits(76, 2) as u32 * 8;
    let (merge, mask) = (t.src_a(), t.src_b()?);
    let selected = match mask {
        Value::U32(m) if m > 0x7f => return Err(t.unsupported()),
        Value::U32(m) => m,
        _ => 0x7f,
    };
    let mut packed = Value::U32(0);
    for index in 0..7u8 {
        if selected & (1 << index) == 0 {
            continue;
        }
        let pred = pred_index(t, index);
        let set = t.pred((pred, false));
        let bit = select(t, set, Value::U32(1 << index), Value::U32(0));
        packed = or(t, packed, bit);
    }
    let (field, packed) = match mask {
        Value::U32(m) => (Value::U32(m << shift), packed),
        mask => {
            let mask = and(t, mask, Value::U32(0x7f));
            let field = shift_left(t, mask, shift);
            (field, and(t, packed, mask))
        }
    };
    let keep = invert(t, field);
    let kept = and(t, merge, keep);
    let placed = shift_left(t, packed, shift);
    let result = or(t, kept, placed);
    let dst = t.reg_d();
    t.set_x(dst, result);
    Ok(())
}

/// R2P: each predicate selected by the mask `b` takes its bit of the byte of `a` that bits
/// 76..78 pick.
fn r2p(t: &mut Translator<'_>) -> TResult {
    if slot32_has_modifier(t) {
        return Err(t.unsupported());
    }
    let shift = t.insn.bits(76, 2) as u32 * 8;
    let (src, mask) = (t.src_a(), t.src_b()?);
    for index in 0..7u8 {
        if let Value::U32(m) = mask
            && m & (1 << index) == 0
        {
            continue;
        }
        let bit = and(t, src, Value::U32(1 << (shift + u32::from(index))));
        let value = t.ir.icompare(ICompare::NotEqual, bit, Value::U32(0));
        let pred = Pred::p(index);
        let value = match mask {
            Value::U32(_) => value,
            mask => {
                let enable_bit = t.ir.bitwise_and(mask, Value::U32(1 << index));
                let enabled = t.ir.icompare(ICompare::NotEqual, enable_bit, Value::U32(0));
                let old = t.ir.get_pred(pred, false);
                t.ir.select(enabled, value, old)
            }
        };
        t.ir.set_pred(pred, value);
    }
    Ok(())
}

/// R2UR: the register of the lowest active lane into a uniform register.
fn r2ur(t: &mut Translator<'_>) -> TResult {
    let src = t.src_a();
    let value = super::warp::broadcast_first_active(t, src);
    let dst = t.insn.udst();
    t.set_x(dst, value);
    Ok(())
}

#[cfg(test)]
#[path = "integer_tests.rs"]
pub(crate) mod tests;
