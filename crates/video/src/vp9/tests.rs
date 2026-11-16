use super::*;

/// The frames of an IVF file: a 32-byte file header, then a 12-byte header before each frame.
fn ivf_frames(bytes: &[u8]) -> Vec<&[u8]> {
    let mut frames = Vec::new();
    let mut at = 32;
    while at + 12 <= bytes.len() {
        let size = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        frames.push(&bytes[at + 12..at + 12 + size]);
        at += 12 + size;
    }
    frames
}

/// Six lossless 64x48 frames from libvpx (see `tests/data/generate.py`).
fn quadrants() -> Vec<&'static [u8]> {
    ivf_frames(include_bytes!("../../tests/data/quadrants.vp9.ivf"))
}

#[test]
fn su_puts_the_sign_after_the_magnitude() {
    // 5 then sign 1, 5 then sign 0, in 4-bit magnitudes: 0101 1 0101 0.
    let data = [0b0101_1010, 0b1000_0000];
    let mut r = BitReader::new(&data);
    assert_eq!(r.su(4).unwrap(), -5);
    assert_eq!(r.su(4).unwrap(), 5);
}

#[test]
fn a_superframe_index_splits_the_packet() {
    // Two frames of 3 and 2 bytes, sizes on one byte each: marker 0b110_00_001.
    let marker = 0xc1;
    let data = [1, 2, 3, 4, 5, marker, 3, 2, marker];
    assert_eq!(superframe(&data), vec![(0, 3), (3, 2)]);
}

#[test]
fn a_packet_without_an_index_is_one_frame() {
    assert_eq!(superframe(&[0x82, 0x49, 0x83]), vec![(0, 3)]);
    assert!(superframe(&[]).is_empty());
    // A marker byte at the end without a matching first index byte is frame data.
    assert_eq!(superframe(&[1, 2, 0xc1]), vec![(0, 3)]);
}

#[test]
fn the_key_frame_header_reads_as_libvpx_wrote_it() {
    let frames = quadrants();
    let mut stream = Vp9Stream::default();
    let (offset, size) = superframe(frames[0])[0];
    let data = &frames[0][offset..offset + size];
    let h = stream.frame(data).unwrap();
    assert!(h.key_frame && h.show_frame && !h.show_existing_frame);
    assert_eq!((h.profile, h.width, h.height), (0, 64, 48));
    assert_eq!(h.color.bit_depth, 8);
    assert!(h.color.subsampling_x && h.color.subsampling_y);
    assert_eq!(h.refresh_frame_flags, 0xff);
    // Lossless: quantizer index and deltas zero.
    assert_eq!(h.base_q_idx, 0);
    assert_eq!(
        (h.delta_q_y_dc, h.delta_q_uv_dc, h.delta_q_uv_ac),
        (0, 0, 0)
    );
    // One 64-wide superblock column allows one tile column.
    assert_eq!(h.tile_cols_log2, 0);
    assert!(h.uncompressed_header_size > 0 && h.compressed_header_size > 0);
    assert!((h.uncompressed_header_size + h.compressed_header_size) as usize <= data.len());
    assert!(!h.use_prev_frame_mvs);
    for slot in 0..NUM_REF_FRAMES {
        assert_eq!(stream.ref_size(slot), Some((64, 48)));
    }
}

#[test]
fn every_frame_of_the_stream_parses_and_six_are_shown() {
    let mut stream = Vp9Stream::default();
    let mut shown = 0;
    for packet in quadrants() {
        for (offset, size) in superframe(packet) {
            let h = stream.frame(&packet[offset..offset + size]).unwrap();
            assert_eq!((h.width, h.height), (64, 48));
            if h.show_existing_frame {
                assert!(
                    stream
                        .ref_size(usize::from(h.frame_to_show_map_idx))
                        .is_some()
                );
            } else if !h.frame_is_intra() {
                for index in h.ref_frame_idx {
                    assert!(stream.ref_size(usize::from(index)).is_some());
                }
            }
            if h.show_frame {
                shown += 1;
            }
        }
    }
    assert_eq!(shown, 6);
}

#[test]
fn an_inter_frame_needs_a_key_frame_first() {
    let frames = quadrants();
    let (offset, size) = *superframe(frames[1]).last().unwrap();
    let mut stream = Vp9Stream::default();
    assert_eq!(
        stream.frame(&frames[1][offset..offset + size]),
        Err(Vp9Error::NoKeyFrame)
    );
}

#[test]
fn a_bad_frame_marker_is_refused() {
    let mut stream = Vp9Stream::default();
    assert_eq!(
        stream.frame(&[0x00, 0x00]),
        Err(Vp9Error::Invalid("frame_marker"))
    );
    assert_eq!(stream.frame(&[0x80]), Err(Vp9Error::Truncated));
}
