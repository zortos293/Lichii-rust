//! BC1/BC3 encoder tests: round-trip PSNR on synthetic images, exact single colours, punch-through
//! alpha, stability of re-encoding, and edge blocks.

mod common;

use common::Rng;
use lichii_texture::bcn::*;

const SIZE: usize = 64;

fn image(mut f: impl FnMut(usize, usize) -> [u8; 4]) -> Vec<u8> {
    (0..SIZE * SIZE)
        .flat_map(|i| f(i % SIZE, i / SIZE))
        .collect()
}

fn gradient() -> Vec<u8> {
    image(|x, y| {
        [
            (x * 4) as u8,
            (y * 4) as u8,
            ((x + y) * 2) as u8,
            (255 - x * 4) as u8,
        ]
    })
}

fn noise() -> Vec<u8> {
    let mut rng = Rng::new(10);
    image(|_, _| rng.next_u64().to_le_bytes()[..4].try_into().unwrap())
}

fn edges() -> Vec<u8> {
    image(|x, y| {
        if (x / 3 + y / 5) % 2 == 0 || x == y {
            [230, 40, 20, 255]
        } else {
            [10, 90, 200, 0]
        }
    })
}

fn smooth_noisy() -> Vec<u8> {
    let mut rng = Rng::new(11);
    image(|x, y| {
        let n = (rng.next_u64() % 16) as f32 - 8.0;
        let (fx, fy) = (x as f32 / SIZE as f32, y as f32 / SIZE as f32);
        let c = |v: f32| (v + n).clamp(0.0, 255.0) as u8;
        [
            c(128.0 + 100.0 * (fx * 9.0).sin()),
            c(128.0 + 90.0 * (fy * 7.0).cos()),
            c(60.0 + 150.0 * fx * fy),
            c(255.0 * fx),
        ]
    })
}

fn psnr(a: &[u8], b: &[u8], channels: std::ops::Range<usize>) -> f64 {
    let (mut sum, mut count) = (0.0, 0.0);
    for (pa, pb) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0) {
        for c in channels.clone() {
            sum += (f64::from(pa[c]) - f64::from(pb[c])).powi(2);
            count += 1.0;
        }
    }
    if sum == 0.0 {
        return f64::INFINITY;
    }
    10.0 * (255.0f64.powi(2) * count / sum).log10()
}

fn opaque(mut rgba: Vec<u8>) -> Vec<u8> {
    for texel in rgba.as_chunks_mut::<4>().0 {
        texel[3] = 255;
    }
    rgba
}

fn bc1_round_trip(rgba: &[u8]) -> Vec<u8> {
    let s = SIZE as u32;
    let mut blocks = vec![0; SIZE * SIZE / 2];
    encode_bc1(rgba, s, s, 1, &mut blocks).unwrap();
    let mut out = vec![0; rgba.len()];
    decode_bc1(&blocks, s, s, 1, &mut out).unwrap();
    out
}

fn bc3_round_trip(rgba: &[u8]) -> Vec<u8> {
    let s = SIZE as u32;
    let mut blocks = vec![0; SIZE * SIZE];
    encode_bc3(rgba, s, s, 1, &mut blocks).unwrap();
    let mut out = vec![0; rgba.len()];
    decode_bc3(&blocks, s, s, 1, &mut out).unwrap();
    out
}

/// Thresholds sit just under what the encoder reaches: 38.6, 13.5, 41.1 and 33.6 dB for colour.
/// On the same images stb_dxt's normal mode, the C++ emulator's encoder, scores 38.5, 13.5, 41.1
/// and 33.6 dB, and the alpha results are identical.
#[test]
fn round_trip_quality() {
    let cases: [(&str, Vec<u8>, f64, f64); 4] = [
        ("gradient", gradient(), 38.3, 53.5),
        ("noise", noise(), 13.3, 29.0),
        ("edges", edges(), 40.8, f64::INFINITY),
        ("smooth_noisy", smooth_noisy(), 33.3, 49.0),
    ];
    for (name, rgba, color_db, alpha_db) in cases {
        let rgb = opaque(rgba.clone());
        let bc1 = psnr(&rgb, &bc1_round_trip(&rgb), 0..3);
        let decoded = bc3_round_trip(&rgba);
        let bc3 = psnr(&rgba, &decoded, 0..3);
        let alpha = psnr(&rgba, &decoded, 3..4);
        assert!(bc1 >= color_db, "{name}: BC1 {bc1:.2} dB");
        assert!(bc3 >= color_db, "{name}: BC3 colour {bc3:.2} dB");
        assert!(alpha >= alpha_db, "{name}: BC3 alpha {alpha:.2} dB");
    }
}

/// A second encode of decoded output stays within a fraction of a dB of the first generation.
#[test]
fn re_encoding_decoded_output_is_stable() {
    for rgba in [gradient(), noise(), edges(), smooth_noisy()] {
        let rgb = opaque(rgba.clone());
        for (source, round_trip) in [
            (rgba, bc3_round_trip as fn(&[u8]) -> Vec<u8>),
            (rgb, bc1_round_trip),
        ] {
            let once = round_trip(&source);
            let twice = round_trip(&once);
            assert!(psnr(&once, &twice, 0..4) >= 50.0);
            assert!(psnr(&source, &twice, 0..4) >= psnr(&source, &once, 0..4) - 0.25);
        }
    }
}

#[test]
fn single_colours_decode_within_one_step() {
    for v in 0..=255u8 {
        for colour in [
            [v, 0, 0, 255],
            [0, v, 0, 255],
            [0, 0, v, 255],
            [v, v, v, 255],
        ] {
            let decoded = decode_bc1_block(&encode_bc1_block(&[colour; 16]));
            for texel in decoded {
                for c in 0..3 {
                    assert!(texel[c].abs_diff(colour[c]) <= 1, "{colour:?} -> {texel:?}");
                }
            }
        }
        let decoded = decode_bc3_block(&encode_bc3_block(&[[v, v, v, v]; 16]));
        assert!(decoded.iter().all(|t| t[3] == v));
    }
}

#[test]
fn bc1_punch_through_alpha() {
    let mut rng = Rng::new(12);
    for _ in 0..2000 {
        let texels: [[u8; 4]; 16] = std::array::from_fn(|_| {
            let [r, g, b, a, ..] = rng.next_u64().to_le_bytes();
            [r, g, b, a]
        });
        let decoded = decode_bc1_block(&encode_bc1_block(&texels));
        for (texel, out) in texels.iter().zip(decoded) {
            if texel[3] < 128 {
                assert_eq!(out, [0; 4]);
            } else {
                assert_eq!(out[3], 255);
            }
        }
    }
    let clear = decode_bc1_block(&encode_bc1_block(&[[200, 100, 50, 0]; 16]));
    assert_eq!(clear, [[0; 4]; 16]);
}

#[test]
fn edge_blocks_repeat_the_last_row_and_column() {
    let rgba: Vec<u8> = (0..5 * 3)
        .flat_map(|i| [i as u8 * 16, 255 - i as u8 * 16, 77, 255])
        .collect();
    let mut blocks = [0u8; 2 * 16];
    encode_bc3(&rgba, 5, 3, 1, &mut blocks).unwrap();
    let edge: [[u8; 4]; 16] = std::array::from_fn(|t| {
        let (x, y) = (4, (t / 4).min(2));
        rgba[(y * 5 + x) * 4..][..4].try_into().unwrap()
    });
    assert_eq!(blocks[16..], encode_bc3_block(&edge));
}
