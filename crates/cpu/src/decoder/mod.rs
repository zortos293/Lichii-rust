//! Table-driven A64 decoder.
//!
//! [`decode`] turns one instruction word into an [`Inst`]. Patterns live in `table.rs` as
//! 32-character bit strings in the style of the ARM ARM encoding tables; the most specific pattern
//! wins, and a pattern's handler may reject reserved field values so the next candidate is tried.

mod inst;
mod table;
#[cfg(test)]
mod tests;

pub use inst::*;

/// Decodes one A64 instruction word. Never fails: unallocated encodings decode to
/// [`Inst::Undefined`] and the whole SIMD/FP data-processing group to [`Inst::SimdFp`].
pub fn decode(word: u32) -> Inst {
    if word & 0x0e00_0000 == 0x0e00_0000 {
        return Inst::SimdFp { word };
    }
    table::lookup(word).unwrap_or(Inst::Undefined { word })
}

/// `DecodeBitMasks` from the ARM ARM: the `(wmask, tmask)` pair for logical immediates
/// (`immediate`) and bitfield moves. Masks cover 64 bits; 32-bit users truncate them.
pub fn decode_bit_masks(n: bool, imms: u32, immr: u32, immediate: bool) -> Option<(u64, u64)> {
    let combined = ((n as u32) << 6) | (!imms & 0x3f);
    if combined <= 1 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    let levels = (1u32 << len) - 1;
    if immediate && imms & levels == levels {
        return None;
    }
    let s = imms & levels;
    let r = immr & levels;
    let d = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let welem = ones(s + 1);
    let telem = ones(d + 1);
    let rotated = if r == 0 {
        welem
    } else {
        ((welem >> r) | (welem << (esize - r))) & ones(esize)
    };
    Some((replicate(rotated, esize), replicate(telem, esize)))
}

pub(crate) const fn ones_u32(count: u32) -> u32 {
    if count >= 32 {
        u32::MAX
    } else {
        (1u32 << count) - 1
    }
}

const fn ones(count: u32) -> u64 {
    if count >= 64 {
        u64::MAX
    } else {
        (1u64 << count) - 1
    }
}

const fn replicate(mut value: u64, mut esize: u32) -> u64 {
    while esize < 64 {
        value |= value << esize;
        esize *= 2;
    }
    value
}
