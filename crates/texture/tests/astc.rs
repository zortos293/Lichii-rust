//! Bit-exact comparison of the ASTC decoder against ARM's astcenc reference decoder.
//!
//! `tests/data/astc` holds blocks astcenc encoded from synthetic images at every 2D footprint,
//! and `manifest.txt` the FNV-1a 64 hash of astcenc's decode of each file in the LDR (`-dl`),
//! sRGB (`-ds`) and HDR (`-dH`) profiles. Rows named `fuzz-WxH` hash the blocks
//! [`fuzz_blocks`] produces, decoded as a single row of blocks; it mirrors `generate.py`.

use std::path::PathBuf;

use lichii_texture::astc::{
    decode_block_rgba8, decode_block_rgba16f, decode_rgba8, decode_rgba16f,
};

const RANDOM_BLOCKS: usize = 2048;
const STRUCTURED_BLOCKS: usize = 2048;
const MAGENTA: [u8; 4] = [0xFF, 0x00, 0xFF, 0xFF];
const HDR_ERROR: [u16; 4] = [0xFFFF; 4];

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/astc")
}

fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

struct Image {
    block_w: u32,
    block_h: u32,
    width: u32,
    height: u32,
    blocks: Vec<u8>,
}

fn parse_astc(file: &[u8]) -> Image {
    assert_eq!(file[..4], 0x5CA1_AB13u32.to_le_bytes());
    let dim = |at: usize| u32::from_le_bytes([file[at], file[at + 1], file[at + 2], 0]);
    Image {
        block_w: u32::from(file[4]),
        block_h: u32::from(file[5]),
        width: dim(7),
        height: dim(10),
        blocks: file[16..].to_vec(),
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Random blocks, a quarter forced to void-extent headers and a quarter to one partition; then
/// single-partition blocks with a 4x4 grid of 2-bit weights, random 8-bit endpoint values and
/// every color endpoint mode; then two copies of every block of `fixtures` with one or two bits
/// flipped.
fn fuzz_blocks(block_w: u32, block_h: u32, fixtures: &[Vec<u8>]) -> Vec<u8> {
    let mut state = 0xA57C_0000 | u64::from(block_w) << 8 | u64::from(block_h);
    let mut out = Vec::with_capacity(RANDOM_BLOCKS * 16);
    for i in 0..RANDOM_BLOCKS {
        let lo = splitmix64(&mut state);
        let hi = splitmix64(&mut state);
        let mut v = u128::from(lo) | u128::from(hi) << 64;
        match i % 4 {
            1 => {
                v = (v & !0xDFF) | 0xDFC;
                if i % 8 == 1 {
                    v |= ((1u128 << 52) - 1) << 12;
                }
            }
            2 => v &= !(3 << 11),
            _ => {}
        }
        out.extend_from_slice(&v.to_le_bytes());
    }
    for _ in 0..STRUCTURED_BLOCKS {
        let lo = splitmix64(&mut state);
        let hi = splitmix64(&mut state);
        let v =
            (u128::from(lo) | u128::from(hi) << 64) & !0x1FFFF | 0x042 | u128::from(lo & 0xF) << 13;
        out.extend_from_slice(&v.to_le_bytes());
    }
    for fixture in fixtures {
        for block in fixture[16..].as_chunks::<16>().0 {
            let v = u128::from_le_bytes(*block);
            for _ in 0..2 {
                let r = splitmix64(&mut state);
                let mut m = v ^ 1 << (r & 127);
                if (r >> 7) & 1 == 1 {
                    m ^= 1 << ((r >> 8) & 127);
                }
                out.extend_from_slice(&m.to_le_bytes());
            }
        }
    }
    out
}

/// The fixtures of one footprint, sorted by file name.
fn footprint_fixtures(block_w: u32, block_h: u32) -> Vec<Vec<u8>> {
    let suffix = format!("-{block_w}x{block_h}.astc");
    let mut names: Vec<String> = std::fs::read_dir(data_dir())
        .expect("fixture directory")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .into_string()
                .expect("UTF-8 name")
        })
        .filter(|name| name.ends_with(&suffix) && !name[..name.len() - suffix.len()].contains('x'))
        .collect();
    names.sort();
    names
        .iter()
        .map(|name| std::fs::read(data_dir().join(name)).expect("fixture"))
        .collect()
}

fn decode_hashes(image: &Image) -> [u64; 3] {
    let Image {
        block_w,
        block_h,
        width,
        height,
        ref blocks,
    } = *image;
    let texels = (width * height) as usize;
    let mut rgba8 = vec![0u8; texels * 4];
    decode_rgba8(
        blocks, width, height, 1, block_w, block_h, false, &mut rgba8,
    );
    let ldr = fnv1a(rgba8.iter().copied());
    decode_rgba8(blocks, width, height, 1, block_w, block_h, true, &mut rgba8);
    let srgb = fnv1a(rgba8.iter().copied());
    let mut rgba16 = vec![0u16; texels * 4];
    decode_rgba16f(blocks, width, height, 1, block_w, block_h, &mut rgba16);
    let hdr = fnv1a(rgba16.iter().flat_map(|v| v.to_le_bytes()));
    [ldr, srgb, hdr]
}

#[test]
fn matches_astcenc_bit_exactly() {
    let manifest = std::fs::read_to_string(data_dir().join("manifest.txt")).expect("manifest");
    let mut mismatches = Vec::new();
    let mut rows = 0;
    for line in manifest.lines().filter(|l| !l.trim().is_empty()) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let expected: Vec<u64> = fields[1..]
            .iter()
            .map(|h| u64::from_str_radix(h, 16).expect("hash"))
            .collect();
        let image = match fields[0].strip_prefix("fuzz-") {
            Some(footprint) => {
                let (w, h) = footprint.split_once('x').expect("footprint");
                let (block_w, block_h) = (w.parse().expect("width"), h.parse().expect("height"));
                let blocks = fuzz_blocks(block_w, block_h, &footprint_fixtures(block_w, block_h));
                Image {
                    block_w,
                    block_h,
                    width: block_w * (blocks.len() / 16) as u32,
                    height: block_h,
                    blocks,
                }
            }
            None => parse_astc(&std::fs::read(data_dir().join(fields[0])).expect("fixture")),
        };
        let actual = decode_hashes(&image);
        for (profile, (a, e)) in ["ldr", "srgb", "hdr"]
            .iter()
            .zip(actual.iter().zip(&expected))
        {
            if a != e {
                mismatches.push(format!("{} {profile}: {a:016x} != {e:016x}", fields[0]));
            }
        }
        rows += 1;
    }
    assert_eq!(rows, 14 * 9, "manifest rows");
    assert!(
        mismatches.is_empty(),
        "decodes differ from astcenc:\n{}",
        mismatches.join("\n")
    );
}

fn void_extent(hdr: bool, color: [u16; 4]) -> [u8; 16] {
    let mut v: u128 = 0x1FC | 0xC00 | ((1u128 << 52) - 1) << 12;
    if hdr {
        v |= 0x200;
    }
    for (i, c) in color.iter().enumerate() {
        v |= u128::from(*c) << (64 + 16 * i);
    }
    v.to_le_bytes()
}

#[test]
fn void_extent_blocks() {
    let mut ldr = [[0u8; 4]; 36];
    let mut hdr = [[0u16; 4]; 36];

    let block = void_extent(false, [0x1234, 0xFFFF, 0x0000, 0x80FF]);
    decode_block_rgba8(&block, 6, 6, false, &mut ldr);
    assert!(ldr.iter().all(|&t| t == [0x12, 0xFF, 0x00, 0x80]));
    decode_block_rgba8(&block, 6, 6, true, &mut ldr);
    assert!(ldr.iter().all(|&t| t == [0x12, 0xFF, 0x00, 0x80]));
    decode_block_rgba16f(&block, 6, 6, &mut hdr);
    assert!(hdr.iter().all(|&t| t == [0x2C8D, 0x3C00, 0x0000, 0x3807]));

    let block = void_extent(true, [0x3C00, 0x4900, 0x0001, 0xBC00]);
    decode_block_rgba16f(&block, 6, 6, &mut hdr);
    assert!(hdr.iter().all(|&t| t == [0x3C00, 0x4900, 0x0001, 0xBC00]));
    decode_block_rgba8(&block, 6, 6, false, &mut ldr);
    assert!(
        ldr.iter().all(|&t| t == MAGENTA),
        "HDR void extent in the LDR profile"
    );

    let mut bad_extent = u128::from_le_bytes(void_extent(false, [0; 4]));
    bad_extent &= !(0x1FFFu128 << 12);
    decode_block_rgba8(&bad_extent.to_le_bytes(), 6, 6, false, &mut ldr);
    assert!(
        ldr.iter().all(|&t| t == MAGENTA),
        "low S coordinate above high S"
    );
}

#[test]
fn illegal_blocks_decode_to_error_color() {
    let reserved_mode = [0u8; 16];
    let mut dual_plane_four_partitions = [0u8; 16];
    dual_plane_four_partitions[0] = 0x42;
    dual_plane_four_partitions[1] = 0x1C;
    for block in [reserved_mode, dual_plane_four_partitions] {
        for &(w, h) in &[(4, 4), (8, 8), (12, 12)] {
            let mut ldr = [[0u8; 4]; 144];
            let mut hdr = [[0u16; 4]; 144];
            decode_block_rgba8(&block, w, h, false, &mut ldr);
            decode_block_rgba16f(&block, w, h, &mut hdr);
            let n = (w * h) as usize;
            assert!(ldr[..n].iter().all(|&t| t == MAGENTA));
            assert!(hdr[..n].iter().all(|&t| t == HDR_ERROR));
        }
    }

    let weight_grid_larger_than_block = 0x0042u16 | 0b11 << 7;
    let mut ldr = [[0u8; 4]; 16];
    decode_block_rgba8(
        &u128::from(weight_grid_larger_than_block).to_le_bytes(),
        4,
        4,
        false,
        &mut ldr,
    );
    assert!(ldr.iter().all(|&t| t == MAGENTA));
}

#[test]
fn image_decode_crops_edges_and_stacks_slices() {
    let blocks = fuzz_blocks(6, 5, &[]);
    let (width, height, depth) = (17u32, 9u32, 3u32);
    let (bx, by) = (3usize, 2usize);
    let mut image = vec![0u8; (width * height * depth * 4) as usize];
    decode_rgba8(&blocks, width, height, depth, 6, 5, false, &mut image);
    let mut texels = [[0u8; 4]; 30];
    for z in 0..depth as usize {
        for y in 0..height as usize {
            for x in 0..width as usize {
                let index = (z * by + y / 5) * bx + x / 6;
                let block: &[u8; 16] = blocks[index * 16..][..16].try_into().unwrap();
                decode_block_rgba8(block, 6, 5, false, &mut texels);
                let at = ((z * height as usize + y) * width as usize + x) * 4;
                assert_eq!(
                    image[at..at + 4],
                    texels[(y % 5) * 6 + x % 6],
                    "texel {x},{y},{z}"
                );
            }
        }
    }
}
