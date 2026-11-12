//! Helpers shared by several instruction families.

use crate::ir::{Emitter, FpCompare, FpControl, ICompare, Value};

/// Integer comparison selector (bits 49..52 on most compare instructions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompareOp {
    False,
    LessThan,
    Equal,
    LessThanEqual,
    GreaterThan,
    NotEqual,
    GreaterThanEqual,
    True,
}

impl CompareOp {
    pub const fn from_bits(bits: u64) -> CompareOp {
        match bits & 7 {
            0 => CompareOp::False,
            1 => CompareOp::LessThan,
            2 => CompareOp::Equal,
            3 => CompareOp::LessThanEqual,
            4 => CompareOp::GreaterThan,
            5 => CompareOp::NotEqual,
            6 => CompareOp::GreaterThanEqual,
            _ => CompareOp::True,
        }
    }
}

/// Predicate combiner (AND, OR, XOR; 3 is reserved).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    And,
    Or,
    Xor,
}

impl BooleanOp {
    pub const fn from_bits(bits: u64) -> Option<BooleanOp> {
        match bits & 3 {
            0 => Some(BooleanOp::And),
            1 => Some(BooleanOp::Or),
            2 => Some(BooleanOp::Xor),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredicateOp {
    False,
    True,
    Zero,
    NonZero,
}

impl PredicateOp {
    pub const fn from_bits(bits: u64) -> PredicateOp {
        match bits & 3 {
            0 => PredicateOp::False,
            1 => PredicateOp::True,
            2 => PredicateOp::Zero,
            _ => PredicateOp::NonZero,
        }
    }
}

/// Floating-point comparison selector (4 bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FpCompareOp {
    F,
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
    Num,
    Nan,
    Ltu,
    Equ,
    Leu,
    Gtu,
    Neu,
    Geu,
    T,
}

impl FpCompareOp {
    pub const fn from_bits(bits: u64) -> FpCompareOp {
        use FpCompareOp::*;
        [
            F, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu, T,
        ][(bits & 15) as usize]
    }

    pub const fn is_ordered(self) -> bool {
        !matches!(
            self,
            FpCompareOp::Ltu
                | FpCompareOp::Equ
                | FpCompareOp::Leu
                | FpCompareOp::Gtu
                | FpCompareOp::Neu
                | FpCompareOp::Geu
        )
    }
}

pub fn integer_compare(
    ir: &mut Emitter<'_>,
    a: Value,
    b: Value,
    op: CompareOp,
    signed: bool,
) -> Value {
    let cmp = match op {
        CompareOp::False => return Value::U1(false),
        CompareOp::True => return Value::U1(true),
        CompareOp::LessThan if signed => ICompare::SLessThan,
        CompareOp::LessThan => ICompare::ULessThan,
        CompareOp::Equal => ICompare::Equal,
        CompareOp::LessThanEqual if signed => ICompare::SLessThanEqual,
        CompareOp::LessThanEqual => ICompare::ULessThanEqual,
        CompareOp::GreaterThan if signed => ICompare::SGreaterThan,
        CompareOp::GreaterThan => ICompare::UGreaterThan,
        CompareOp::NotEqual => ICompare::NotEqual,
        CompareOp::GreaterThanEqual if signed => ICompare::SGreaterThanEqual,
        CompareOp::GreaterThanEqual => ICompare::UGreaterThanEqual,
    };
    ir.icompare(cmp, a, b)
}

/// Comparison that continues a wider compare through the carry and zero flags (.X).
pub fn extended_integer_compare(
    ir: &mut Emitter<'_>,
    a: Value,
    b: Value,
    op: CompareOp,
    signed: bool,
) -> Value {
    let zero = Value::U32(0);
    let c = ir.get_c_flag();
    let carry = ir.select(c, Value::U32(1), zero);
    let z_flag = ir.get_z_flag();
    let not_b = ir.bitwise_not(b);
    let sum = ir.iadd(a, not_b);
    let intermediate = ir.iadd(sum, carry);
    let flip = if signed {
        Value::U1(false)
    } else {
        let a_neg = ir.icompare(ICompare::SLessThan, a, zero);
        let b_neg = ir.icompare(ICompare::SLessThan, b, zero);
        ir.logical_xor(a_neg, b_neg)
    };
    let flipped = |ir: &mut Emitter<'_>, if_flip: ICompare, otherwise: ICompare| {
        let t = ir.icompare(if_flip, intermediate, zero);
        let f = ir.icompare(otherwise, intermediate, zero);
        ir.select(flip, t, f)
    };
    let eq_and = |ir: &mut Emitter<'_>, z: Value| {
        let eq = ir.icompare(ICompare::Equal, intermediate, zero);
        ir.logical_and(eq, z)
    };
    match op {
        CompareOp::False => Value::U1(false),
        CompareOp::True => Value::U1(true),
        CompareOp::LessThan => flipped(ir, ICompare::SGreaterThanEqual, ICompare::SLessThan),
        CompareOp::Equal => eq_and(ir, z_flag),
        CompareOp::LessThanEqual => {
            let base = flipped(ir, ICompare::SGreaterThanEqual, ICompare::SLessThan);
            let eq = eq_and(ir, z_flag);
            ir.logical_or(base, eq)
        }
        CompareOp::GreaterThan => {
            let base = flipped(ir, ICompare::SLessThanEqual, ICompare::SGreaterThan);
            let not_z = ir.logical_not(z_flag);
            let eq = eq_and(ir, not_z);
            ir.logical_or(base, eq)
        }
        CompareOp::NotEqual => {
            let ne = ir.icompare(ICompare::NotEqual, intermediate, zero);
            let not_z = ir.logical_not(z_flag);
            let eq = eq_and(ir, not_z);
            ir.logical_or(ne, eq)
        }
        CompareOp::GreaterThanEqual => {
            let base = flipped(ir, ICompare::SLessThan, ICompare::SGreaterThanEqual);
            let eq = eq_and(ir, z_flag);
            ir.logical_or(base, eq)
        }
    }
}

pub fn predicate_combine(ir: &mut Emitter<'_>, a: Value, b: Value, op: BooleanOp) -> Value {
    match op {
        BooleanOp::And => ir.logical_and(a, b),
        BooleanOp::Or => ir.logical_or(a, b),
        BooleanOp::Xor => ir.logical_xor(a, b),
    }
}

pub fn predicate_operation(ir: &mut Emitter<'_>, result: Value, op: PredicateOp) -> Value {
    match op {
        PredicateOp::False => Value::U1(false),
        PredicateOp::True => Value::U1(true),
        PredicateOp::Zero => ir.icompare(ICompare::Equal, result, Value::U32(0)),
        PredicateOp::NonZero => ir.icompare(ICompare::NotEqual, result, Value::U32(0)),
    }
}

pub fn floating_point_compare(
    ir: &mut Emitter<'_>,
    a: Value,
    b: Value,
    op: FpCompareOp,
    control: FpControl,
) -> Value {
    use FpCompareOp::*;
    let ordered = op.is_ordered();
    let cmp = match op {
        F => return Value::U1(false),
        T => return Value::U1(true),
        Num => return ir.fp_ordered(a, b),
        Nan => return ir.fp_unordered(a, b),
        Lt | Ltu => FpCompare::LessThan,
        Eq | Equ => FpCompare::Equal,
        Le | Leu => FpCompare::LessThanEqual,
        Gt | Gtu => FpCompare::GreaterThan,
        Ne | Neu => FpCompare::NotEqual,
        Ge | Geu => FpCompare::GreaterThanEqual,
    };
    ir.fp_compare(cmp, a, b, ordered, control)
}
