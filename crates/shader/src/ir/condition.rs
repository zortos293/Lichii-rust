//! Branch conditions: a predicate and a Maxwell condition-code flow test.

use std::fmt;

use super::value::Pred;

/// Maxwell condition-code test. Ampere programs only use `T`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum FlowTest {
    F,
    LT,
    EQ,
    LE,
    GT,
    NE,
    GE,
    NUM,
    NaN,
    LTU,
    EQU,
    LEU,
    GTU,
    NEU,
    GEU,
    T,
    OFF,
    LO,
    SFF,
    LS,
    HI,
    SFT,
    HS,
    OFT,
    CsmTa,
    CsmTr,
    CsmMx,
    FcsmTa,
    FcsmTr,
    FcsmMx,
    RLE,
    RGT,
}

impl FlowTest {
    const ALL: [FlowTest; 32] = [
        FlowTest::F,
        FlowTest::LT,
        FlowTest::EQ,
        FlowTest::LE,
        FlowTest::GT,
        FlowTest::NE,
        FlowTest::GE,
        FlowTest::NUM,
        FlowTest::NaN,
        FlowTest::LTU,
        FlowTest::EQU,
        FlowTest::LEU,
        FlowTest::GTU,
        FlowTest::NEU,
        FlowTest::GEU,
        FlowTest::T,
        FlowTest::OFF,
        FlowTest::LO,
        FlowTest::SFF,
        FlowTest::LS,
        FlowTest::HI,
        FlowTest::SFT,
        FlowTest::HS,
        FlowTest::OFT,
        FlowTest::CsmTa,
        FlowTest::CsmTr,
        FlowTest::CsmMx,
        FlowTest::FcsmTa,
        FlowTest::FcsmTr,
        FlowTest::FcsmMx,
        FlowTest::RLE,
        FlowTest::RGT,
    ];

    /// Decodes the 5-bit flow test field of a Maxwell branch.
    pub const fn from_bits(bits: u32) -> FlowTest {
        Self::ALL[(bits & 31) as usize]
    }
}

/// Condition under which a block's terminator is taken.
///
/// The predicate (possibly negated) is ANDed with the flow test, and with `extra` when an
/// Ampere branch names a second predicate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Condition {
    pub flow_test: FlowTest,
    pub pred: Pred,
    pub negated: bool,
    pub extra: Option<(Pred, bool)>,
}

impl Condition {
    pub const TRUE: Condition = Condition {
        flow_test: FlowTest::T,
        pred: Pred::PT,
        negated: false,
        extra: None,
    };
    pub const FALSE: Condition = Condition {
        flow_test: FlowTest::T,
        pred: Pred::PT,
        negated: true,
        extra: None,
    };

    pub const fn pred(pred: Pred, negated: bool) -> Condition {
        Condition {
            flow_test: FlowTest::T,
            pred,
            negated,
            extra: None,
        }
    }

    pub const fn with_flow_test(flow_test: FlowTest, pred: Pred, negated: bool) -> Condition {
        Condition {
            flow_test,
            pred,
            negated,
            extra: None,
        }
    }

    pub fn is_true(&self) -> bool {
        self.flow_test == FlowTest::T
            && self.pred.is_true()
            && !self.negated
            && self.extra.is_none_or(|(p, n)| p.is_true() && !n)
    }

    pub fn is_false(&self) -> bool {
        self.flow_test == FlowTest::F
            || (self.pred.is_true() && self.negated)
            || self.extra.is_some_and(|(p, n)| p.is_true() && n)
    }
}

impl fmt::Display for Condition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let neg = if self.negated { "!" } else { "" };
        write!(f, "{neg}{}", self.pred)?;
        if let Some((p, n)) = self.extra {
            write!(f, "&{}{p}", if n { "!" } else { "" })?;
        }
        if self.flow_test != FlowTest::T {
            write!(f, ".{:?}", self.flow_test)?;
        }
        Ok(())
    }
}
