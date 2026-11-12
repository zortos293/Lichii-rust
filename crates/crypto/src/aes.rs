use aes::Aes128;
use aes::cipher::{Array, BlockCipherDecrypt, BlockCipherEncrypt, KeyInit};

use crate::Key128;

/// Size of one AES block in bytes.
pub const AES_BLOCK_SIZE: usize = 16;

const BATCH_BLOCKS: usize = 8;

type Block = Array<u8, aes::cipher::consts::U16>;

fn cipher(key: &Key128) -> Aes128 {
    Aes128::new(&Array::from(*key))
}

fn blocks_mut(data: &mut [u8]) -> &mut [Block] {
    let (blocks, rest) = Block::slice_as_chunks_mut(data);
    assert!(rest.is_empty(), "AES data must be a multiple of 16 bytes");
    blocks
}

/// AES-128 in ECB mode, used for key derivation and NCA key areas.
#[derive(Clone)]
pub struct AesEcb {
    cipher: Aes128,
}

impl AesEcb {
    /// Expands `key` once; the cipher can then be reused for any number of blocks.
    pub fn new(key: &Key128) -> Self {
        Self {
            cipher: cipher(key),
        }
    }

    /// Encrypts `data` in place. Panics unless its length is a multiple of 16.
    pub fn encrypt(&self, data: &mut [u8]) {
        self.cipher.encrypt_blocks(blocks_mut(data));
    }

    /// Decrypts `data` in place. Panics unless its length is a multiple of 16.
    pub fn decrypt(&self, data: &mut [u8]) {
        self.cipher.decrypt_blocks(blocks_mut(data));
    }

    /// Decrypts one 16-byte key (or any single block) and returns it.
    pub fn decrypt_key(&self, key: &Key128) -> Key128 {
        let mut out = *key;
        self.decrypt(&mut out);
        out
    }
}

/// AES-128 in CTR mode with random access: the counter for byte `offset` is `iv + offset / 16`
/// as a 128-bit big-endian integer, which is how NCA sections and BKTR buckets lay out their
/// counters.
#[derive(Clone)]
pub struct AesCtr {
    cipher: Aes128,
}

impl AesCtr {
    /// Expands `key` once; [`AesCtr::apply`] then needs no per-call setup.
    pub fn new(key: &Key128) -> Self {
        Self {
            cipher: cipher(key),
        }
    }

    /// XORs the keystream for `[offset, offset + data.len())` into `data`. Encryption and
    /// decryption are the same operation.
    pub fn apply(&self, iv: &[u8; 16], offset: u64, data: &mut [u8]) {
        let base = u128::from_be_bytes(*iv);
        let mut counter = base.wrapping_add(u128::from(offset / AES_BLOCK_SIZE as u64));
        let mut skip = (offset % AES_BLOCK_SIZE as u64) as usize;
        let mut keystream = [Block::default(); BATCH_BLOCKS];
        let mut data = data;
        while !data.is_empty() {
            let needed = (skip + data.len())
                .div_ceil(AES_BLOCK_SIZE)
                .min(BATCH_BLOCKS);
            for block in &mut keystream[..needed] {
                *block = Array::from(counter.to_be_bytes());
                counter = counter.wrapping_add(1);
            }
            self.cipher.encrypt_blocks(&mut keystream[..needed]);
            let stream = Block::slice_as_flattened(&keystream[..needed]);
            let take = (stream.len() - skip).min(data.len());
            let (head, tail) = data.split_at_mut(take);
            for (byte, key) in head.iter_mut().zip(&stream[skip..]) {
                *byte ^= key;
            }
            data = tail;
            skip = 0;
        }
    }
}

/// How an XTS sector number becomes the 16-byte tweak block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XtsTweak {
    /// IEEE P1619: the sector number as a little-endian 128-bit integer.
    Standard,
    /// Nintendo (NCA headers, NAND partitions): the sector number as a big-endian 128-bit integer.
    Nintendo,
}

impl XtsTweak {
    fn block(self, sector: u64) -> [u8; 16] {
        match self {
            Self::Standard => u128::from(sector).to_le_bytes(),
            Self::Nintendo => u128::from(sector).to_be_bytes(),
        }
    }
}

/// AES-128-XTS over whole sectors. The 32-byte key is the data key followed by the tweak key.
#[derive(Clone)]
pub struct AesXts {
    data: Aes128,
    tweak: Aes128,
    kind: XtsTweak,
}

impl AesXts {
    /// `key` is the 16-byte data key followed by the 16-byte tweak key.
    pub fn new(key: &[u8; 32], kind: XtsTweak) -> Self {
        let (data, tweak) = key.split_at(16);
        Self {
            data: Aes128::new(&Array::try_from(data).unwrap()),
            tweak: Aes128::new(&Array::try_from(tweak).unwrap()),
            kind,
        }
    }

    /// Decrypts whole sectors in place, the first one numbered `first_sector`. Panics unless
    /// `data` is a multiple of `sector_size` and `sector_size` a multiple of 16.
    pub fn decrypt(&self, data: &mut [u8], sector_size: usize, first_sector: u64) {
        self.run(data, sector_size, first_sector, |cipher, blocks| {
            cipher.decrypt_blocks(blocks)
        });
    }

    /// Encrypts whole sectors in place; see [`AesXts::decrypt`].
    pub fn encrypt(&self, data: &mut [u8], sector_size: usize, first_sector: u64) {
        self.run(data, sector_size, first_sector, |cipher, blocks| {
            cipher.encrypt_blocks(blocks)
        });
    }

    fn run(
        &self,
        data: &mut [u8],
        sector_size: usize,
        first_sector: u64,
        transform: impl Fn(&Aes128, &mut [Block]),
    ) {
        assert!(
            sector_size > 0
                && sector_size.is_multiple_of(AES_BLOCK_SIZE)
                && data.len().is_multiple_of(sector_size),
            "XTS data must be whole sectors of a multiple of 16 bytes"
        );
        for (index, sector) in data.chunks_exact_mut(sector_size).enumerate() {
            let mut tweak_block = Array::from(self.kind.block(first_sector + index as u64));
            self.tweak.encrypt_block(&mut tweak_block);
            let mut tweak = u128::from_le_bytes(tweak_block.0);
            for chunk in blocks_mut(sector).chunks_mut(BATCH_BLOCKS) {
                let mut tweaks = [0u128; BATCH_BLOCKS];
                for (block, slot) in chunk.iter_mut().zip(&mut tweaks) {
                    *slot = tweak;
                    xor_block(block, tweak);
                    tweak = gf_double(tweak);
                }
                transform(&self.data, chunk);
                for (block, slot) in chunk.iter_mut().zip(&tweaks) {
                    xor_block(block, *slot);
                }
            }
        }
    }
}

fn xor_block(block: &mut Block, tweak: u128) {
    let value = u128::from_le_bytes(block.0) ^ tweak;
    block.0 = value.to_le_bytes();
}

fn gf_double(value: u128) -> u128 {
    let carry = (value >> 127) as u8;
    (value << 1) ^ u128::from(carry * 0x87)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hex::parse_hex;

    fn hex<const N: usize>(s: &str) -> [u8; N] {
        parse_hex(s).unwrap()
    }

    fn hex_vec(s: &str) -> Vec<u8> {
        let s: String = s.split_whitespace().collect();
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    const SP800_38A_KEY: &str = "2b7e151628aed2a6abf7158809cf4f3c";
    const SP800_38A_PLAIN: &str =
        "6bc1bee22e409f96e93d7e117393172a ae2d8a571e03ac9c9eb76fac45af8e51
        30c81c46a35ce411e5fbc1191a0a52ef f69f2445df4f9b17ad2b417be66c3710";

    #[test]
    fn fips197_single_block() {
        let ecb = AesEcb::new(&hex("000102030405060708090a0b0c0d0e0f"));
        let mut block: [u8; 16] = hex("00112233445566778899aabbccddeeff");
        ecb.encrypt(&mut block);
        assert_eq!(block, hex::<16>("69c4e0d86a7b0430d8cdb78070b4c55a"));
        ecb.decrypt(&mut block);
        assert_eq!(block, hex::<16>("00112233445566778899aabbccddeeff"));
    }

    #[test]
    fn sp800_38a_ecb() {
        let ecb = AesEcb::new(&hex(SP800_38A_KEY));
        let mut data = hex_vec(SP800_38A_PLAIN);
        ecb.encrypt(&mut data);
        assert_eq!(
            data,
            hex_vec(
                "3ad77bb40d7a3660a89ecaf32466ef97 f5d3d58503b9699de785895a96fdbaaf
                 43b1cd7f598ece23881b00e3ed030688 7b0c785e27e8ad3f8223207104725dd4"
            )
        );
        ecb.decrypt(&mut data);
        assert_eq!(data, hex_vec(SP800_38A_PLAIN));
    }

    #[test]
    fn sp800_38a_ctr_with_random_access() {
        let ctr = AesCtr::new(&hex(SP800_38A_KEY));
        let iv = hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff");
        let expected = hex_vec(
            "874d6191b620e3261bef6864990db6ce 9806f66b7970fdff8617187bb9fffdff
             5ae4df3edbd5d35e5b4f09020db03eab 1e031dda2fbe03d1792170a0f3009cee",
        );
        let mut data = hex_vec(SP800_38A_PLAIN);
        ctr.apply(&iv, 0, &mut data);
        assert_eq!(data, expected);

        for (start, end) in [(0, 1), (5, 37), (16, 64), (17, 18), (31, 64)] {
            let mut slice = hex_vec(SP800_38A_PLAIN)[start..end].to_vec();
            ctr.apply(&iv, start as u64, &mut slice);
            assert_eq!(slice, expected[start..end], "range {start}..{end}");
        }
    }

    #[test]
    fn ctr_long_buffer_matches_blockwise() {
        let ctr = AesCtr::new(&[7; 16]);
        let iv = [0xff; 16];
        let mut long = vec![0u8; 16 * 21 + 5];
        ctr.apply(&iv, 3, &mut long);
        let mut piecewise = vec![0u8; long.len()];
        for (index, byte) in piecewise.iter_mut().enumerate() {
            ctr.apply(&iv, 3 + index as u64, std::slice::from_mut(byte));
        }
        assert_eq!(long, piecewise);
    }

    #[test]
    fn ieee1619_xts_vectors() {
        let cases = [
            (
                "00000000000000000000000000000000 00000000000000000000000000000000",
                0u64,
                "0000000000000000000000000000000000000000000000000000000000000000",
                "917cf69ebd68b2ec9b9fe9a3eadda692cd43d2f59598ed858c02c2652fbf922e",
            ),
            (
                "11111111111111111111111111111111 22222222222222222222222222222222",
                0x3333333333,
                "4444444444444444444444444444444444444444444444444444444444444444",
                "c454185e6a16936e39334038acef838bfb186fff7480adc4289382ecd6d394f0",
            ),
            (
                "fffefdfcfbfaf9f8f7f6f5f4f3f2f1f0 22222222222222222222222222222222",
                0x3333333333,
                "4444444444444444444444444444444444444444444444444444444444444444",
                "af85336b597afc1a900b2eb21ec949d292df4c047e0b21532186a5971a227a89",
            ),
        ];
        for (key, sector, plain, cipher_text) in cases {
            let key: [u8; 32] = hex_vec(key).try_into().unwrap();
            let xts = AesXts::new(&key, XtsTweak::Standard);
            let mut data = hex_vec(plain);
            xts.encrypt(&mut data, 32, sector);
            assert_eq!(data, hex_vec(cipher_text));
            xts.decrypt(&mut data, 32, sector);
            assert_eq!(data, hex_vec(plain));
        }
    }

    #[test]
    fn ieee1619_xts_full_sector() {
        let key: [u8; 32] =
            hex_vec("27182818284590452353602874713526 31415926535897932384626433832795")
                .try_into()
                .unwrap();
        let plain: Vec<u8> = (0..512).map(|i| i as u8).collect();
        let xts = AesXts::new(&key, XtsTweak::Standard);
        let mut data = plain.clone();
        xts.encrypt(&mut data, 512, 0);
        assert_eq!(
            data[..32],
            hex_vec("27a7479befa1d476489f308cd4cfa6e2a96e4bbe3208ff25287dd3819616e89c")
        );
        assert_eq!(
            data[480..],
            hex_vec("eb4a427d1923ce3ff262735779a418f20a282df920147beabe421ee5319d0568")
        );
        xts.decrypt(&mut data, 512, 0);
        assert_eq!(data, plain);
    }

    #[test]
    fn nintendo_tweak_is_big_endian_sector() {
        let key: [u8; 32] = std::array::from_fn(|i| i as u8);
        let plain: Vec<u8> = (2..4u64)
            .flat_map(|sector| (0..0x200u64).map(move |i| ((i * 7 + sector) % 256) as u8))
            .collect();
        let xts = AesXts::new(&key, XtsTweak::Nintendo);
        let mut data = plain.clone();
        xts.encrypt(&mut data, 0x200, 2);
        assert_eq!(
            data[..32],
            hex_vec("31b7248b5fdcd019d9adab668bacde57fcbc31f725d7d6ccbb1e3c890628b2ff")
        );
        assert_eq!(
            data[0x200..0x220],
            hex_vec("1ade20ed9e354a9481456d1cf6828637f5d220cfdcdd765f02f5491733904a57")
        );
        assert_eq!(data[0x3f0..], hex_vec("aff22779945258d8145a8d7c512459c9"));
        xts.decrypt(&mut data[0x200..], 0x200, 3);
        xts.decrypt(&mut data[..0x200], 0x200, 2);
        assert_eq!(data, plain);
    }
}
