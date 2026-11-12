//! NSO executables: three segments, each optionally LZ4- or ZBIC-compressed and hashed.

use lichii_fs::{File, FileExt};

use crate::error::{LoaderError, LoaderResult};
use crate::program::Module;
use crate::zstd;

const HEADER_SIZE: usize = 0x100;
const MAX_SEGMENT_SIZE: u32 = 1 << 30;
const FLAG_ZBIC: u32 = 1 << 7;

/// Whether to check each decompressed segment against the SHA-256 in the header (when the
/// header's check flag for that segment is set).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HashCheck {
    Skip,
    Verify,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SegmentHeader {
    file_offset: u32,
    memory_offset: u32,
    size: u32,
}

/// Reads `file` as an NSO and lays it out as a [`Module`] named `name`.
pub fn load_nso(file: &dyn File, name: &str, hashes: HashCheck) -> LoaderResult<Module> {
    let header: [u8; HEADER_SIZE] = file
        .read_array(0)
        .map_err(|_| LoaderError::Corrupt("NSO", "shorter than its header"))?;
    if &header[..4] != b"NSO0" {
        return Err(LoaderError::BadMagic("NSO"));
    }
    let word = |offset: usize| u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap());
    let flags = word(0x0c);
    let segments: [SegmentHeader; 3] = std::array::from_fn(|i| SegmentHeader {
        file_offset: word(0x10 + i * 0x10),
        memory_offset: word(0x14 + i * 0x10),
        size: word(0x18 + i * 0x10),
    });
    let bss_size = u64::from(word(0x3c));
    let build_id: [u8; 32] = header[0x40..0x60].try_into().unwrap();
    let compressed_sizes: [u32; 3] = std::array::from_fn(|i| word(0x60 + i * 4));

    let mut end = 0u64;
    for segment in &segments {
        if segment.size > MAX_SEGMENT_SIZE || segment.memory_offset % 0x1000 != 0 {
            return Err(LoaderError::Corrupt(
                "NSO",
                "segment size or alignment out of range",
            ));
        }
        let segment_end = u64::from(segment.memory_offset) + u64::from(segment.size);
        if segment_end < end && segment.size != 0 {
            return Err(LoaderError::Corrupt(
                "NSO",
                "segments overlap or are out of order",
            ));
        }
        end = end.max(segment_end);
    }

    let mut image = vec![0u8; end as usize];
    let mut compressed = Vec::new();
    for (index, segment) in segments.iter().enumerate() {
        let dst = &mut image[segment.memory_offset as usize..][..segment.size as usize];
        let is_compressed = flags & (1 << index) != 0;
        let stored = if is_compressed {
            compressed_sizes[index]
        } else {
            segment.size
        };
        if stored > MAX_SEGMENT_SIZE {
            return Err(LoaderError::Corrupt(
                "NSO",
                "compressed segment size out of range",
            ));
        }
        let decompress_error = |reason: String| LoaderError::Decompress {
            module: name.to_owned(),
            segment: index,
            reason,
        };
        if !is_compressed {
            file.read_exact_at(u64::from(segment.file_offset), dst)?;
        } else {
            compressed.resize(stored as usize, 0);
            file.read_exact_at(u64::from(segment.file_offset), &mut compressed)?;
            let written = if flags & FLAG_ZBIC != 0 {
                zstd::decompress(zstd::Variant::Zbic, &compressed, dst)
                    .map_err(|e| decompress_error(e.to_string()))?
            } else {
                lz4_flex::block::decompress_into(&compressed, dst)
                    .map_err(|e| decompress_error(e.to_string()))?
            };
            if written != dst.len() {
                return Err(decompress_error(format!(
                    "produced {written:#x} of {:#x} bytes",
                    dst.len()
                )));
            }
        }
        let check = flags & (1 << (3 + index)) != 0;
        if hashes == HashCheck::Verify && check {
            let expected = &header[0xa0 + index * 0x20..0xc0 + index * 0x20];
            if lichii_crypto::sha256(dst) != expected {
                return Err(LoaderError::HashMismatch {
                    module: name.to_owned(),
                    segment: index,
                });
            }
        }
    }

    let layout = segments.map(|s| (u64::from(s.memory_offset), u64::from(s.size)));
    Ok(Module::from_layout(
        name.to_owned(),
        image,
        layout,
        bss_size,
        build_id,
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::program::Permission;
    use lichii_fs::MemFile;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Compression {
        None,
        Lz4,
    }

    /// Segment contents for a synthetic module: a branch at 0, the MOD0 offset at 4 and a MOD0
    /// header in .text, then rodata and data patterns.
    pub fn segments(seed: u8) -> [Vec<u8>; 3] {
        let mut text = vec![0u8; 0x1800];
        text[..4].copy_from_slice(&0x1400_0002u32.to_le_bytes());
        text[4..8].copy_from_slice(&0x10u32.to_le_bytes());
        text[0x10..0x14].copy_from_slice(b"MOD0");
        for (i, word) in [0x3000i32, 0x4000 - 0x10, 0x5000 - 0x10, 0, 0, 0x4800 - 0x10]
            .iter()
            .enumerate()
        {
            text[0x14 + i * 4..0x18 + i * 4].copy_from_slice(&word.to_le_bytes());
        }
        for (i, byte) in text[0x100..].iter_mut().enumerate() {
            *byte = (i as u8).wrapping_mul(seed) ^ (i >> 8) as u8;
        }
        let rodata = b"rodata string table "
            .iter()
            .copied()
            .cycle()
            .take(0x900)
            .collect();
        let data = (0..0x200u32)
            .flat_map(|i| (i * u32::from(seed)).to_le_bytes())
            .collect();
        [text, rodata, data]
    }

    pub fn build_nso(
        segs: &[Vec<u8>; 3],
        compression: Compression,
        bss: u32,
        hashed: bool,
    ) -> Vec<u8> {
        let mut header = vec![0u8; HEADER_SIZE];
        header[..4].copy_from_slice(b"NSO0");
        let stored: Vec<Vec<u8>> = segs
            .iter()
            .map(|s| match compression {
                Compression::None => s.clone(),
                Compression::Lz4 => lz4_flex::block::compress(s),
            })
            .collect();
        let mut flags: u32 = if compression == Compression::Lz4 {
            0b111
        } else {
            0
        };
        if hashed {
            flags |= 0b111 << 3;
        }
        header[0x0c..0x10].copy_from_slice(&flags.to_le_bytes());
        let mut file_offset = HEADER_SIZE as u32;
        let mut memory_offset = 0u32;
        for (i, (segment, stored)) in segs.iter().zip(&stored).enumerate() {
            let at = 0x10 + i * 0x10;
            header[at..at + 4].copy_from_slice(&file_offset.to_le_bytes());
            header[at + 4..at + 8].copy_from_slice(&memory_offset.to_le_bytes());
            header[at + 8..at + 12].copy_from_slice(&(segment.len() as u32).to_le_bytes());
            header[0x60 + i * 4..0x64 + i * 4]
                .copy_from_slice(&(stored.len() as u32).to_le_bytes());
            header[0xa0 + i * 0x20..0xc0 + i * 0x20]
                .copy_from_slice(&lichii_crypto::sha256(segment));
            file_offset += stored.len() as u32;
            memory_offset += (segment.len() as u32).next_multiple_of(0x1000);
        }
        header[0x3c..0x40].copy_from_slice(&bss.to_le_bytes());
        header[0x40..0x44].copy_from_slice(b"BILD");
        header.extend(stored.concat());
        header
    }

    #[test]
    fn loads_uncompressed_and_lz4_modules() {
        let segs = segments(3);
        for compression in [Compression::None, Compression::Lz4] {
            let nso = build_nso(&segs, compression, 0x2345, true);
            let module =
                load_nso(MemFile::from_bytes(nso).as_ref(), "main", HashCheck::Verify).unwrap();
            assert_eq!(&module.image[..0x1800], &segs[0][..]);
            assert_eq!(&module.image[0x2000..0x2900], &segs[1][..]);
            assert_eq!(&module.image[0x3000..0x3800], &segs[2][..]);
            assert_eq!(module.image.len(), 0x4000);
            assert_eq!(module.size, 0x6000);
            assert_eq!(module.segments[0].size, 0x2000);
            assert_eq!(module.segments[0].permission, Permission::ReadExecute);
            assert_eq!(
                (module.segments[1].offset, module.segments[1].size),
                (0x2000, 0x1000)
            );
            assert_eq!(
                (module.segments[2].offset, module.segments[2].size),
                (0x3000, 0x3000)
            );
            assert_eq!(&module.build_id[..4], b"BILD");
            let mod0 = module.mod0.unwrap();
            assert_eq!(
                (mod0.offset, mod0.dynamic, mod0.bss_start, mod0.bss_end),
                (0x10, 0x3010, 0x4000, 0x5000)
            );
            assert_eq!(mod0.module_object, 0x4800);
        }
    }

    #[test]
    fn loads_switch2_zbic_segments() {
        let frame: &[u8] = include_bytes!("../tests/data/zstd/arm_l19_cs.zbic");
        let mut plain = vec![0u8; 1 << 20];
        let len = zstd::decompress(zstd::Variant::Zbic, frame, &mut plain).unwrap();
        plain.truncate(len);
        let rodata = vec![0x11u8; 0x10];
        let data = vec![0x22u8; 0x20];

        let mut nso = vec![0u8; HEADER_SIZE];
        nso[..4].copy_from_slice(b"NSO0");
        nso[0x0c..0x10].copy_from_slice(&(1u32 | FLAG_ZBIC | (1 << 3)).to_le_bytes());
        let ro_offset = (plain.len() as u32).next_multiple_of(0x1000);
        let data_offset = ro_offset + 0x1000;
        let stored: [(&[u8], u32, u32); 3] = [
            (frame, 0, plain.len() as u32),
            (&rodata, ro_offset, rodata.len() as u32),
            (&data, data_offset, data.len() as u32),
        ];
        let mut file_offset = HEADER_SIZE as u32;
        for (i, (bytes, memory_offset, size)) in stored.iter().enumerate() {
            let at = 0x10 + i * 0x10;
            nso[at..at + 4].copy_from_slice(&file_offset.to_le_bytes());
            nso[at + 4..at + 8].copy_from_slice(&memory_offset.to_le_bytes());
            nso[at + 8..at + 12].copy_from_slice(&size.to_le_bytes());
            nso[0x60 + i * 4..0x64 + i * 4].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
            file_offset += bytes.len() as u32;
        }
        nso[0xa0..0xc0].copy_from_slice(&lichii_crypto::sha256(&plain));
        for (bytes, _, _) in &stored {
            nso.extend_from_slice(bytes);
        }

        let module = load_nso(
            MemFile::from_bytes(nso.clone()).as_ref(),
            "main",
            HashCheck::Verify,
        )
        .unwrap();
        assert_eq!(&module.image[..plain.len()], &plain[..]);
        assert_eq!(&module.image[ro_offset as usize..][..0x10], &rodata[..]);
        assert_eq!(&module.image[data_offset as usize..][..0x20], &data[..]);

        nso[0x0c..0x10].copy_from_slice(&1u32.to_le_bytes());
        assert!(matches!(
            load_nso(MemFile::from_bytes(nso).as_ref(), "main", HashCheck::Skip),
            Err(LoaderError::Decompress { segment: 0, .. })
        ));
    }

    #[test]
    fn hash_mismatches_and_bad_input_are_errors() {
        let segs = segments(5);
        let mut nso = build_nso(&segs, Compression::None, 0, true);
        let last = nso.len() - 1;
        nso[last] ^= 1;
        let file = MemFile::from_bytes(nso);
        assert!(matches!(
            load_nso(file.as_ref(), "sdk", HashCheck::Verify),
            Err(LoaderError::HashMismatch { segment: 2, .. })
        ));
        assert!(load_nso(file.as_ref(), "sdk", HashCheck::Skip).is_ok());

        let mut lz4 = build_nso(&segs, Compression::Lz4, 0, false);
        lz4[0x60..0x64].copy_from_slice(&8u32.to_le_bytes());
        assert!(matches!(
            load_nso(MemFile::from_bytes(lz4).as_ref(), "main", HashCheck::Skip),
            Err(LoaderError::Decompress { segment: 0, .. })
        ));
        let mut huge = build_nso(&segs, Compression::None, 0, false);
        huge[0x18..0x1c].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(load_nso(MemFile::from_bytes(huge).as_ref(), "main", HashCheck::Skip).is_err());
        assert!(matches!(
            load_nso(
                MemFile::from_bytes(vec![0u8; 0x100]).as_ref(),
                "main",
                HashCheck::Skip
            ),
            Err(LoaderError::BadMagic("NSO"))
        ));
        assert!(
            load_nso(
                MemFile::from_bytes(b"NSO0".to_vec()).as_ref(),
                "main",
                HashCheck::Skip
            )
            .is_err()
        );
    }
}
