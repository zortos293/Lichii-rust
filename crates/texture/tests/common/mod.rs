#![allow(dead_code)]

/// splitmix64, the same generator the ETC fixture generator uses.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// 16 bytes from two draws, little-endian.
    pub fn block16(&mut self) -> [u8; 16] {
        let (a, b) = (self.next_u64(), self.next_u64());
        let mut out = [0; 16];
        out[..8].copy_from_slice(&a.to_le_bytes());
        out[8..].copy_from_slice(&b.to_le_bytes());
        out
    }

    pub fn block8(&mut self) -> [u8; 8] {
        self.next_u64().to_le_bytes()
    }
}

/// texture2ddecoder's packed texel (BGRA in a little-endian u32) as RGBA.
pub fn unpack_bgra(texel: u32) -> [u8; 4] {
    let [b, g, r, a] = texel.to_le_bytes();
    [r, g, b, a]
}
