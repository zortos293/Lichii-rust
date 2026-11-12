//! Register field helpers shared by every engine: open enums (named values of a register field
//! that may also hold values the hardware documentation does not name) and bitfield words.

/// A value decoded from a register field `bits` wide.
pub trait Field: Copy {
    fn from_field(raw: u32, bits: u32) -> Self;
}

impl Field for u32 {
    #[inline]
    fn from_field(raw: u32, _bits: u32) -> Self {
        raw
    }
}

impl Field for bool {
    #[inline]
    fn from_field(raw: u32, _bits: u32) -> Self {
        raw != 0
    }
}

impl Field for i32 {
    #[inline]
    fn from_field(raw: u32, bits: u32) -> Self {
        let shift = 32 - bits;
        ((raw << shift) as i32) >> shift
    }
}

/// Bits `[lo, lo + len)` of `raw`.
#[inline]
pub const fn extract(raw: u32, lo: u32, len: u32) -> u32 {
    let shifted = raw >> lo;
    if len >= 32 {
        shifted
    } else {
        shifted & ((1 << len) - 1)
    }
}

/// `raw` with bits `[lo, lo + len)` replaced by the low bits of `value`.
#[inline]
pub const fn insert(raw: u32, lo: u32, len: u32, value: u32) -> u32 {
    let mask = if len >= 32 {
        u32::MAX
    } else {
        ((1 << len) - 1) << lo
    };
    (raw & !mask) | ((value << lo) & mask)
}

/// A 32-bit register value with named constants. Unknown values stay representable, so decoding
/// a guest register never fails.
macro_rules! open_enum {
    ($(#[$meta:meta])* pub struct $name:ident { $($variant:ident = $value:expr,)* }) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name(pub u32);

        impl $name {
            $(pub const $variant: Self = Self($value);)*

            /// The constant's name for known values.
            pub fn name(self) -> Option<&'static str> {
                $(if self.0 == $value {
                    return Some(stringify!($variant));
                })*
                None
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self.name() {
                    Some(name) => write!(f, "{}::{}", stringify!($name), name),
                    None => write!(f, "{}({:#x})", stringify!($name), self.0),
                }
            }
        }

        impl $crate::bits::Field for $name {
            #[inline]
            fn from_field(raw: u32, _bits: u32) -> Self {
                Self(raw)
            }
        }
    };
}

/// A 32-bit register made of bitfields, with a getter per field.
macro_rules! bitfield {
    ($(#[$meta:meta])* pub struct $name:ident { $($field:ident: $ty:ident @ $lo:literal, $len:literal;)* }) => {
        $(#[$meta])*
        #[repr(transparent)]
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Default, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $name(pub u32);

        impl $name {
            $(
                #[inline]
                pub fn $field(self) -> $ty {
                    <$ty as $crate::bits::Field>::from_field(
                        $crate::bits::extract(self.0, $lo, $len),
                        $len,
                    )
                }
            )*

            /// This value with bits `[lo, lo + len)` set to `value`.
            #[inline]
            pub const fn with_bits(self, lo: u32, len: u32, value: u32) -> Self {
                Self($crate::bits::insert(self.0, lo, len, value))
            }
        }

        impl core::fmt::Debug for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.debug_struct(stringify!($name))
                    $(.field(stringify!($field), &self.$field()))*
                    .finish()
            }
        }

        impl $crate::bits::Field for $name {
            #[inline]
            fn from_field(raw: u32, _bits: u32) -> Self {
                Self(raw)
            }
        }
    };
}

pub(crate) use {bitfield, open_enum};

#[cfg(test)]
mod tests {
    open_enum! {
        pub struct Mode {
            A = 1,
            B = 2,
        }
    }

    bitfield! {
        pub struct Word {
            low: u32 @ 0, 4;
            flag: bool @ 4, 1;
            mode: Mode @ 8, 2;
            signed: i32 @ 16, 8;
            top: u32 @ 0, 32;
        }
    }

    #[test]
    fn decodes_fields() {
        let w = Word(0x00fe_0219);
        assert_eq!(w.low(), 9);
        assert!(w.flag());
        assert_eq!(w.mode(), Mode::B);
        assert_eq!(w.signed(), -2);
        assert_eq!(w.top(), 0x00fe_0219);
        assert_eq!(w.with_bits(0, 4, 3).low(), 3);
        assert_eq!(format!("{:?}", Mode(7)), "Mode(0x7)");
        assert_eq!(format!("{:?}", Mode::A), "Mode::A");
    }
}
