//! Handle programs: texture handles a shader reads from storage buffers, as small stack programs
//! over constant buffer and storage buffer words that the renderer evaluates for each draw.
//!
//! The texture pass builds one for a bindless handle no constant buffer word gives (a handle
//! table in a storage buffer, NVN2's usual layout). A descriptor names its program by
//! [`crate::info::TextureDescriptor::handle_program`]; [`HandleProgram::evaluate`] computes the
//! raw handle the descriptor would otherwise read from its constant buffer word.

/// One operation of a [`HandleProgram`], a stack machine over 32-bit words. Binary operations
/// pop the right operand first; comparisons and logical operations push 1 or 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HandleOp {
    /// Pushes the value.
    Immediate(u32),
    /// Pushes the word of constant buffer `index` at byte `offset`.
    Cbuf {
        index: u32,
        offset: u32,
    },
    /// Pops a byte offset and pushes the word of constant buffer `index` there.
    CbufDynamic {
        index: u32,
    },
    /// Pops a byte offset and pushes the word of storage buffer `binding` there, relative to
    /// the address the shader binds the buffer at (its pointer aligned down to
    /// [`crate::Profile::min_ssbo_alignment`]).
    Storage {
        binding: u32,
    },
    Add,
    Sub,
    Mul,
    And,
    Or,
    Xor,
    /// Shifts by 32 or more give 0.
    Shl,
    /// Shifts by 32 or more give 0.
    Shr,
    /// Shifts by 32 or more give the sign.
    Sar,
    Neg,
    Not,
    UMin,
    UMax,
    SMin,
    SMax,
    SLt,
    SLe,
    SGt,
    SGe,
    ULt,
    ULe,
    UGt,
    UGe,
    Eq,
    Ne,
    LAnd,
    LOr,
    LXor,
    LNot,
    /// Pops the false value, the true value and the condition.
    Select,
    /// Keeps the top value in the slot, leaving it on the stack (a shared subexpression).
    Tee(u8),
    /// Pushes the value kept in the slot.
    Load(u8),
    /// Pushes the element of the descriptor array being computed: the program computes one
    /// handle per element of a table the shader indexes.
    Index,
}

impl HandleOp {
    /// Values popped and pushed.
    const fn arity(self) -> (usize, usize) {
        match self {
            HandleOp::Immediate(_)
            | HandleOp::Cbuf { .. }
            | HandleOp::Load(_)
            | HandleOp::Index => (0, 1),
            HandleOp::Tee(_) => (1, 1),
            HandleOp::CbufDynamic { .. }
            | HandleOp::Storage { .. }
            | HandleOp::Neg
            | HandleOp::Not
            | HandleOp::LNot => (1, 1),
            HandleOp::Select => (3, 1),
            _ => (2, 1),
        }
    }
}

/// A program computing one texture handle; see the [module docs](self).
#[derive(Clone, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "Vec<HandleOp>")]
pub struct HandleProgram(Vec<HandleOp>);

impl TryFrom<Vec<HandleOp>> for HandleProgram {
    type Error = &'static str;

    fn try_from(ops: Vec<HandleOp>) -> Result<HandleProgram, &'static str> {
        HandleProgram::new(ops).ok_or("invalid handle program")
    }
}

impl HandleProgram {
    /// Values the evaluation stack holds at most.
    pub const MAX_STACK: usize = 32;

    /// A program of `ops`, or `None` when it would pop an empty stack, load a slot nothing kept,
    /// need more than [`Self::MAX_STACK`] values, or end with other than one value.
    pub fn new(ops: Vec<HandleOp>) -> Option<HandleProgram> {
        let mut depth = 0usize;
        let mut kept = [false; 256];
        for &op in &ops {
            let (pop, push) = op.arity();
            depth = depth.checked_sub(pop)? + push;
            if depth > Self::MAX_STACK {
                return None;
            }
            match op {
                HandleOp::Tee(slot) => kept[usize::from(slot)] = true,
                HandleOp::Load(slot) if !kept[usize::from(slot)] => return None,
                _ => {}
            }
        }
        (depth == 1).then_some(HandleProgram(ops))
    }

    pub fn ops(&self) -> &[HandleOp] {
        &self.0
    }

    /// The program reading storage buffer `bindings[b]` wherever it read binding `b`; a
    /// binding past the end keeps its value.
    pub(crate) fn with_storage_bindings(&self, bindings: &[u32]) -> HandleProgram {
        let remap = |op| match op {
            HandleOp::Storage { binding } => HandleOp::Storage {
                binding: bindings.get(binding as usize).copied().unwrap_or(binding),
            },
            op => op,
        };
        HandleProgram(self.0.iter().copied().map(remap).collect())
    }

    /// Computes the handle of descriptor array `element` (0 for a single descriptor).
    /// `read_cbuf(index, offset)` and `read_storage(binding, offset)` return the words the
    /// program reads, 0 for an unbound buffer. Allocates nothing.
    pub fn evaluate(
        &self,
        mut read_cbuf: impl FnMut(u32, u32) -> u32,
        mut read_storage: impl FnMut(u32, u32) -> u32,
        element: u32,
    ) -> u32 {
        let mut stack = [0u32; Self::MAX_STACK];
        let mut slots = [0u32; 256];
        let mut size = 0;
        for &op in &self.0 {
            let (pop, _) = op.arity();
            size -= pop;
            let [a, b, c] = [0, 1, 2].map(|i| stack.get(size + i).copied().unwrap_or(0));
            let (sa, sb) = (a as i32, b as i32);
            let value = match op {
                HandleOp::Immediate(value) => value,
                HandleOp::Cbuf { index, offset } => read_cbuf(index, offset),
                HandleOp::CbufDynamic { index } => read_cbuf(index, a),
                HandleOp::Storage { binding } => read_storage(binding, a),
                HandleOp::Add => a.wrapping_add(b),
                HandleOp::Sub => a.wrapping_sub(b),
                HandleOp::Mul => a.wrapping_mul(b),
                HandleOp::And => a & b,
                HandleOp::Or => a | b,
                HandleOp::Xor => a ^ b,
                HandleOp::Shl => a.checked_shl(b).unwrap_or(0),
                HandleOp::Shr => a.checked_shr(b).unwrap_or(0),
                HandleOp::Sar => (sa >> b.min(31)) as u32,
                HandleOp::Neg => a.wrapping_neg(),
                HandleOp::Not => !a,
                HandleOp::UMin => a.min(b),
                HandleOp::UMax => a.max(b),
                HandleOp::SMin => sa.min(sb) as u32,
                HandleOp::SMax => sa.max(sb) as u32,
                HandleOp::SLt => u32::from(sa < sb),
                HandleOp::SLe => u32::from(sa <= sb),
                HandleOp::SGt => u32::from(sa > sb),
                HandleOp::SGe => u32::from(sa >= sb),
                HandleOp::ULt => u32::from(a < b),
                HandleOp::ULe => u32::from(a <= b),
                HandleOp::UGt => u32::from(a > b),
                HandleOp::UGe => u32::from(a >= b),
                HandleOp::Eq => u32::from(a == b),
                HandleOp::Ne => u32::from(a != b),
                HandleOp::LAnd => u32::from(a != 0 && b != 0),
                HandleOp::LOr => u32::from(a != 0 || b != 0),
                HandleOp::LXor => u32::from((a != 0) != (b != 0)),
                HandleOp::LNot => u32::from(a == 0),
                HandleOp::Select => {
                    if a != 0 {
                        b
                    } else {
                        c
                    }
                }
                HandleOp::Tee(slot) => {
                    slots[usize::from(slot)] = a;
                    a
                }
                HandleOp::Load(slot) => slots[usize::from(slot)],
                HandleOp::Index => element,
            };
            stack[size] = value;
            size += 1;
        }
        stack[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(ops: Vec<HandleOp>) -> u32 {
        let program = HandleProgram::new(ops).expect("well-formed");
        program.evaluate(
            |index, offset| index << 16 | offset,
            |binding, offset| 0x8000_0000 | binding << 16 | offset,
            7,
        )
    }

    #[test]
    fn reads_and_arithmetic() {
        use HandleOp::*;
        assert_eq!(
            eval(vec![Cbuf {
                index: 1,
                offset: 0x30
            }]),
            0x1_0030
        );
        assert_eq!(
            eval(vec![Immediate(0x10), CbufDynamic { index: 2 }]),
            0x2_0010
        );
        assert_eq!(
            eval(vec![Index, Immediate(3), Shl, Storage { binding: 4 }]),
            0x8004_0038
        );
        assert_eq!(eval(vec![Immediate(1), Immediate(40), Shl]), 0);
        assert_eq!(
            eval(vec![Immediate(0x8000_0000), Immediate(40), Sar]),
            u32::MAX
        );
        assert_eq!(eval(vec![Immediate(u32::MAX), Immediate(1), SLt]), 1);
        assert_eq!(eval(vec![Immediate(u32::MAX), Immediate(1), ULt]), 0);
        assert_eq!(eval(vec![Immediate(10), Immediate(3), Sub]), 7);
        assert_eq!(
            eval(vec![Immediate(0), Immediate(5), Immediate(9), Select]),
            9
        );
        assert_eq!(eval(vec![Immediate(6), Tee(0), Load(0), Mul]), 36);
    }

    #[test]
    fn malformed_programs_are_rejected() {
        use HandleOp::*;
        assert_eq!(HandleProgram::new(vec![Add]), None);
        assert_eq!(HandleProgram::new(vec![]), None);
        assert_eq!(HandleProgram::new(vec![Immediate(1), Immediate(2)]), None);
        assert_eq!(HandleProgram::new(vec![Load(3)]), None);
        let deep = vec![Immediate(1); HandleProgram::MAX_STACK + 1];
        assert_eq!(HandleProgram::new(deep), None);
    }
}
