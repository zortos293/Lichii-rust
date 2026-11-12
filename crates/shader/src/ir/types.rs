//! IR value types.

use std::fmt;

bitflags::bitflags! {
    /// Type of an IR value. Opcode signatures use single flags; `OPAQUE` matches anything.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
    pub struct Type: u32 {
        const OPAQUE = 1 << 0;
        const REG = 1 << 1;
        const PRED = 1 << 2;
        const ATTRIBUTE = 1 << 3;
        const PATCH = 1 << 4;
        const U1 = 1 << 5;
        const U8 = 1 << 6;
        const U16 = 1 << 7;
        const U32 = 1 << 8;
        const U64 = 1 << 9;
        const F16 = 1 << 10;
        const F32 = 1 << 11;
        const F64 = 1 << 12;
        const U32X2 = 1 << 13;
        const U32X3 = 1 << 14;
        const U32X4 = 1 << 15;
        const F16X2 = 1 << 16;
        const F16X3 = 1 << 17;
        const F16X4 = 1 << 18;
        const F32X2 = 1 << 19;
        const F32X3 = 1 << 20;
        const F32X4 = 1 << 21;
        const F64X2 = 1 << 22;
        const F64X3 = 1 << 23;
        const F64X4 = 1 << 24;
    }
}

impl Type {
    pub const VOID: Type = Type::empty();

    /// True when a value of type `self` may be passed where `expected` is required.
    pub fn compatible(self, expected: Type) -> bool {
        self == expected || self == Type::OPAQUE || expected == Type::OPAQUE
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return write!(f, "Void");
        }
        let mut first = true;
        for (name, _) in self.iter_names() {
            if !first {
                write!(f, "|")?;
            }
            first = false;
            write!(f, "{name}")?;
        }
        Ok(())
    }
}
