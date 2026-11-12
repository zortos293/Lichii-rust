//! Differential tests of the BCn decoders against bcdec_rs (bit-exact) and, for BC7, the
//! independent texture2ddecoder, over seeded random blocks with every mode forced in turn.
//! texture2ddecoder's BC6H is not used: its unquantization mishandles negative values and the
//! unsigned maximum, and it only returns 8-bit tonemapped texels.

mod common;

use common::{Rng, unpack_bgra};
use lichii_texture::ImageSizeError;
use lichii_texture::bcn::*;

const BLOCKS: usize = 20_000;

fn bcdec_rgba(block: &[u8], f: fn(&[u8], &mut [u8], usize)) -> [[u8; 4]; 16] {
    let mut out = [[0u8; 4]; 16];
    f(block, out.as_flattened_mut(), 16);
    out
}

#[test]
fn bc1_bc2_bc3_match_bcdec() {
    let mut rng = Rng::new(1);
    for n in 0..BLOCKS {
        let mut block = rng.block16();
        if n % 4 == 0 {
            block.copy_within(10..12, 8);
            block.copy_within(2..4, 0);
        }
        let b8: &[u8; 8] = block[..8].try_into().unwrap();
        assert_eq!(
            decode_bc1_block(b8),
            bcdec_rgba(b8, bcdec_rs::bc1),
            "bc1 {block:02x?}"
        );
        assert_eq!(
            decode_bc2_block(&block),
            bcdec_rgba(&block, bcdec_rs::bc2),
            "bc2 {block:02x?}"
        );
        assert_eq!(
            decode_bc3_block(&block),
            bcdec_rgba(&block, bcdec_rs::bc3),
            "bc3 {block:02x?}"
        );
    }
}

#[test]
fn bc4_bc5_match_bcdec() {
    let mut rng = Rng::new(2);
    for n in 0..BLOCKS {
        let mut block = rng.block16();
        if n % 8 == 0 {
            block[0] = 0x80;
            block[9] = 0x80;
        }
        let b8: &[u8; 8] = block[..8].try_into().unwrap();
        for signed in [false, true] {
            let mut r = [0u8; 16];
            bcdec_rs::bc4(b8, &mut r, 4, signed);
            let mut rg = [[0u8; 2]; 16];
            bcdec_rs::bc5(&block, rg.as_flattened_mut(), 8, signed);
            if signed {
                assert_eq!(
                    decode_bc4_snorm_block(b8).map(|v| v as u8),
                    r,
                    "bc4s {block:02x?}"
                );
                let ours = decode_bc5_snorm_block(&block).map(|t| t.map(|v| v as u8));
                assert_eq!(ours, rg, "bc5s {block:02x?}");
            } else {
                assert_eq!(decode_bc4_unorm_block(b8), r, "bc4u {block:02x?}");
                assert_eq!(decode_bc5_unorm_block(&block), rg, "bc5u {block:02x?}");
            }
        }
    }
}

const BC6H_MODE_BITS: [u8; 18] = [
    0b00, 0b01, 0b00010, 0b00110, 0b01010, 0b01110, 0b10010, 0b10110, 0b11010, 0b11110, 0b00011,
    0b00111, 0b01011, 0b01111, 0b10011, 0b10111, 0b11011, 0b11111,
];

#[test]
fn bc6h_matches_bcdec() {
    let mut rng = Rng::new(3);
    for n in 0..BLOCKS * 2 {
        let mut block = rng.block16();
        let mode = BC6H_MODE_BITS[n % BC6H_MODE_BITS.len()];
        let mask = if mode < 2 { 0b11 } else { 0b11111 };
        block[0] = (block[0] & !mask) | mode;
        for signed in [false, true] {
            let ours = if signed {
                decode_bc6h_sfloat_block(&block)
            } else {
                decode_bc6h_ufloat_block(&block)
            };
            let mut reference = [[0u16; 3]; 16];
            bcdec_rs::bc6h_half(&block, reference.as_flattened_mut(), 12, signed);
            for (t, (o, r)) in ours.iter().zip(&reference).enumerate() {
                assert_eq!(o[..3], r[..], "bc6h signed={signed} texel {t} {block:02x?}");
                assert_eq!(o[3], 0x3c00);
            }
        }
    }
}

#[test]
fn bc7_matches_bcdec_and_texture2ddecoder() {
    let mut rng = Rng::new(4);
    for n in 0..BLOCKS * 2 {
        let mut block = rng.block16();
        let mode = n % 9;
        if mode == 8 {
            block[0] = 0;
        } else {
            block[0] = (block[0] & !((2u8 << mode).wrapping_sub(1))) | (1 << mode);
        }
        let ours = decode_bc7_block(&block);
        assert_eq!(
            ours,
            bcdec_rgba(&block, bcdec_rs::bc7),
            "bc7 mode {mode} {block:02x?}"
        );
        if mode == 8 {
            continue;
        }
        let mut packed = [0u32; 16];
        texture2ddecoder::decode_bc7_block(&block, &mut packed);
        assert_eq!(
            ours,
            packed.map(unpack_bgra),
            "bc7 t2d mode {mode} {block:02x?}"
        );
    }
}

/// Lays out per-block results the way the image decoders should.
fn expected_image<const B: usize, const T: usize>(
    blocks: &[u8],
    width: usize,
    height: usize,
    depth: usize,
    decode: impl Fn(&[u8; B]) -> [[u8; T]; 16],
) -> Vec<u8> {
    let (bx, by) = (width.div_ceil(4), height.div_ceil(4));
    let mut out = vec![0; width * height * depth * T];
    for z in 0..depth {
        for y in 0..height {
            for x in 0..width {
                let index = (z * by + y / 4) * bx + x / 4;
                let block: &[u8; B] = blocks[index * B..][..B].try_into().unwrap();
                let texel = decode(block)[(y % 4) * 4 + x % 4];
                out[((z * height + y) * width + x) * T..][..T].copy_from_slice(&texel);
            }
        }
    }
    out
}

#[test]
fn images_crop_partial_blocks() {
    let mut rng = Rng::new(5);
    for (width, height, depth) in [(13, 7, 2), (4, 4, 1), (1, 1, 1), (5, 9, 3), (16, 3, 1)] {
        let blocks_len =
            (width as usize).div_ceil(4) * (height as usize).div_ceil(4) * depth as usize;
        let data: Vec<u8> = (0..blocks_len).flat_map(|_| rng.block16()).collect();
        let (w, h, d) = (width as usize, height as usize, depth as usize);
        let texels = w * h * d;
        let half = &data[..blocks_len * 8];

        let mut out = vec![0; texels * 4];
        decode_bc1(half, width, height, depth, &mut out).unwrap();
        assert_eq!(out, expected_image(half, w, h, d, decode_bc1_block));
        decode_bc7(&data, width, height, depth, &mut out).unwrap();
        assert_eq!(out, expected_image(&data, w, h, d, decode_bc7_block));
        decode_bc3(&data, width, height, depth, &mut out).unwrap();
        assert_eq!(out, expected_image(&data, w, h, d, decode_bc3_block));

        let mut out = vec![0; texels];
        decode_bc4_snorm(half, width, height, depth, &mut out).unwrap();
        let bc4 = |b: &[u8; 8]| decode_bc4_snorm_block(b).map(|v| [v as u8]);
        assert_eq!(out, expected_image(half, w, h, d, bc4));

        let mut out = vec![0; texels * 2];
        decode_bc5_unorm(&data, width, height, depth, &mut out).unwrap();
        assert_eq!(out, expected_image(&data, w, h, d, decode_bc5_unorm_block));

        let mut out = vec![0; texels * 8];
        decode_bc6h_sfloat(&data, width, height, depth, &mut out).unwrap();
        let bc6h = |b: &[u8; 16]| {
            decode_bc6h_sfloat_block(b).map(|t| {
                let [[a, b], [c, d], [e, f], [g, h]] = t.map(u16::to_le_bytes);
                [a, b, c, d, e, f, g, h]
            })
        };
        assert_eq!(out, expected_image(&data, w, h, d, bc6h));
    }
}

#[test]
fn images_reject_short_buffers() {
    let blocks = [0u8; 4 * 16];
    let mut out = [0u8; 8 * 8 * 4];
    assert_eq!(
        decode_bc7(&blocks[..63], 8, 8, 1, &mut out),
        Err(ImageSizeError::Input {
            required: 64,
            actual: 63
        })
    );
    assert_eq!(
        decode_bc7(&blocks, 8, 8, 1, &mut out[..255]),
        Err(ImageSizeError::Output {
            required: 256,
            actual: 255
        })
    );
    assert_eq!(decode_bc7(&[], 0, 0, 1, &mut []), Ok(()));
}
