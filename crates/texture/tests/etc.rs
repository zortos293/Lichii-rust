//! ETC2/EAC decoder tests: bit-exact against reference hashes from etcdec (ETC2, unsigned EAC)
//! and Mesa (signed EAC) in data/etc_reference.bin, which data/gen_etc_reference.c regenerates,
//! plus a live differential run against texture2ddecoder's ETC2 decoders.

mod common;

use common::{Rng, unpack_bgra};
use lichii_texture::etc::*;

const REFERENCE: &[u8] = include_bytes!("data/etc_reference.bin");
const REFERENCE_BLOCKS: usize = 2048;

fn fnv1a(bytes: &[u8]) -> u32 {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    });
    hash as u32
}

fn check_reference(format: usize, decode: impl Fn(&[u8; 16]) -> Vec<u8>) {
    let mut rng = Rng::new(0x4554_4332_0000_0000 + format as u64);
    let hashes = &REFERENCE[format * REFERENCE_BLOCKS * 4..][..REFERENCE_BLOCKS * 4];
    for (n, expected) in hashes.as_chunks::<4>().0.iter().enumerate() {
        let block = rng.block16();
        let texels = decode(&block);
        let expected = u32::from_le_bytes(*expected);
        assert_eq!(
            fnv1a(&texels),
            expected,
            "format {format} block {n} {block:02x?} decoded {texels:02x?}"
        );
    }
}

fn first8(block: &[u8; 16]) -> &[u8; 8] {
    block[..8].try_into().unwrap()
}

#[test]
fn etc2_matches_etcdec() {
    check_reference(0, |b| {
        decode_etc2_rgb_block(first8(b)).as_flattened().to_vec()
    });
    check_reference(1, |b| {
        decode_etc2_rgb_pta_block(first8(b)).as_flattened().to_vec()
    });
    check_reference(2, |b| decode_etc2_rgba_block(b).as_flattened().to_vec());
}

#[test]
fn eac_matches_etcdec_and_mesa() {
    check_reference(3, |b| {
        decode_eac_r11_unorm_block(first8(b))
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    });
    check_reference(4, |b| {
        decode_eac_r11_snorm_block(first8(b))
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    });
    check_reference(5, |b| {
        decode_eac_rg11_unorm_block(b)
            .as_flattened()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    });
    check_reference(6, |b| {
        decode_eac_rg11_snorm_block(b)
            .as_flattened()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    });
}

#[test]
fn eac_signed_base_minus_128_reads_as_minus_127() {
    let block = [0x80, 0x00, 0, 0, 0, 0, 0, 0];
    let alias = [0x81, 0x00, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        decode_eac_r11_snorm_block(&block),
        decode_eac_r11_snorm_block(&alias)
    );
    assert!(
        decode_eac_r11_snorm_block(&block)
            .iter()
            .all(|&v| v == -(1019 << 5 | 1019 >> 5))
    );
}

/// texture2ddecoder keeps the colour of punch-through transparent texels, where the format
/// defines them as (0, 0, 0, 0); everything else must match exactly.
#[test]
fn etc2_matches_texture2ddecoder() {
    let mut rng = Rng::new(6);
    for _ in 0..100_000 {
        let block = rng.block16();
        let mut packed = [0u32; 16];

        texture2ddecoder::decode_etc2_rgb_block(first8(&block), &mut packed);
        assert_eq!(
            decode_etc2_rgb_block(first8(&block)),
            packed.map(unpack_bgra),
            "{block:02x?}"
        );

        texture2ddecoder::decode_etc2_rgba8_block(&block, &mut packed);
        assert_eq!(
            decode_etc2_rgba_block(&block),
            packed.map(unpack_bgra),
            "{block:02x?}"
        );

        texture2ddecoder::decode_etc2_rgba1_block(first8(&block), &mut packed);
        let expected = packed.map(|p| match unpack_bgra(p) {
            [_, _, _, 0] => [0; 4],
            texel => texel,
        });
        assert_eq!(
            decode_etc2_rgb_pta_block(first8(&block)),
            expected,
            "{block:02x?}"
        );
    }
}

#[test]
fn images_crop_partial_blocks() {
    let mut rng = Rng::new(7);
    let (width, height, depth) = (7u32, 5u32, 2u32);
    let (w, h, d) = (7usize, 5usize, 2usize);
    let blocks: Vec<[u8; 16]> = (0..2 * 2 * d).map(|_| rng.block16()).collect();
    let bytes: Vec<u8> = blocks.iter().flatten().copied().collect();
    let halves: Vec<u8> = blocks.iter().flat_map(|b| b[..8].to_vec()).collect();

    let mut rgba = vec![0; w * h * d * 4];
    decode_etc2_rgb(&halves, width, height, depth, &mut rgba).unwrap();
    let mut rg = vec![0; w * h * d * 4];
    decode_eac_rg11_snorm(&bytes, width, height, depth, &mut rg).unwrap();
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let block = &blocks[(z * 2 + y / 4) * 2 + x / 4];
                let t = (y % 4) * 4 + x % 4;
                let offset = ((z * h + y) * w + x) * 4;
                assert_eq!(
                    rgba[offset..offset + 4],
                    decode_etc2_rgb_block(first8(block))[t]
                );
                let [r, g] = decode_eac_rg11_snorm_block(block)[t];
                let expected = [r.to_le_bytes(), g.to_le_bytes()].concat();
                assert_eq!(rg[offset..offset + 4], expected[..]);
            }
        }
    }
}
