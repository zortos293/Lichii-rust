use super::bits::tests::{BitWriter, leb128_bytes};
use super::frame::FilmGrain;
use super::sequence::tests::movie_sequence;
use super::*;

fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = vec![obu_type << 3 | 0x2];
    out.extend(leb128_bytes(payload.len() as u64));
    out.extend_from_slice(payload);
    out
}

/// A shown key frame of the movie sequence: two tile columns, loop filter deltas, CDEF,
/// restoration and film grain, then a tile group of two tiles (10 and 7 bytes).
fn key_frame() -> Vec<u8> {
    let mut w = BitWriter::default();
    w.flag(false).f(0, 2).flag(true); // not existing, KEY, shown
    w.flag(false).flag(false); // disable_cdf_update, allow_screen_content_tools
    w.flag(false).f(0, 7); // no size override, order hint 0
    w.flag(false); // render size same
    w.flag(false); // disable_frame_end_update_cdf
    w.flag(true).flag(true).flag(false).flag(false); // uniform, cols log2 1, rows log2 0
    w.f(0, 1).f(3, 2); // context_update_tile_id, tile_size_bytes_minus_1
    w.f(100, 8).flag(false).flag(false).flag(false).flag(false); // q 100, no deltas, no qm
    w.flag(false); // no segmentation
    w.flag(false); // no delta q
    w.f(10, 6).f(12, 6).f(5, 6).f(6, 6).f(2, 3); // levels, sharpness
    w.flag(true).flag(true); // delta enabled and updated
    w.flag(false).flag(true).su(3, 7); // LAST delta 3
    for _ in 0..6 {
        w.flag(false);
    }
    w.flag(false).flag(false);
    w.f(2, 2).f(1, 2); // cdef damping 5, two strengths
    w.f(5, 4)
        .f(3, 2)
        .f(2, 4)
        .f(1, 2)
        .f(9, 4)
        .f(0, 2)
        .f(4, 4)
        .f(2, 2);
    w.f(2, 2).f(0, 2).f(1, 2); // lr: Wiener, none, switchable
    w.f(1, 1).f(1, 1).f(1, 1); // unit shift 2, uv shift 1
    w.flag(true); // tx mode select
    w.flag(false); // reduced_tx_set
    w.flag(true).f(0x1234, 16); // apply grain, seed
    w.f(2, 4).f(16, 8).f(64, 8).f(128, 8).f(32, 8);
    w.flag(false); // chroma_scaling_from_luma
    w.f(1, 4).f(64, 8).f(20, 8).f(0, 4); // cb 1 point, cr none
    w.f(3, 2).f(1, 2); // grain scaling 11, lag 1
    for c in [120, 130, 140, 150, 1, 2, 3, 4, 5] {
        w.f(c, 8);
    }
    w.f(1, 2)
        .f(0, 2)
        .f(128, 8)
        .f(192, 8)
        .f(256, 9)
        .flag(true)
        .flag(false);
    w.align();
    let mut payload = w.bytes.clone();
    payload.push(0x00); // tile_start_and_end_present_flag 0, aligned
    payload.extend_from_slice(&[9, 0, 0, 0]); // first tile: 10 bytes
    payload.extend(std::iter::repeat_n(0xaa, 10));
    payload.extend(std::iter::repeat_n(0xbb, 7));
    payload
}

/// An inter frame on the key frame: one tile, LAST translated by global motion, film grain
/// loaded from slot 0.
fn inter_frame() -> Vec<u8> {
    let mut w = BitWriter::default();
    w.flag(false).f(1, 2).flag(true).flag(false); // INTER, shown, not error resilient
    w.flag(false).flag(false).flag(false).f(1, 7); // cdf, sct, override, order hint 1
    w.f(0, 3).f(0x01, 8); // primary_ref_frame 0, refresh slot 0
    w.flag(false); // frame_refs_short_signaling
    for i in 0..7 {
        w.f(i, 3);
    }
    w.flag(false); // render size
    w.flag(true); // allow_high_precision_mv
    w.flag(false).f(1, 2); // interpolation filter 1
    w.flag(true).flag(true); // motion mode switchable, ref frame mvs
    w.flag(true); // disable_frame_end_update_cdf
    w.flag(true).flag(false).flag(false); // one tile
    w.f(50, 8).flag(false).flag(false).flag(false).flag(false);
    w.flag(false).flag(false); // segmentation, delta q
    w.f(0, 6).f(0, 6).f(0, 3).flag(true).flag(false); // filter off, deltas kept
    w.f(0, 2).f(0, 2).f(0, 4).f(0, 2).f(0, 4).f(0, 2); // cdef
    w.f(0, 2).f(0, 2).f(0, 2); // no restoration
    w.flag(false); // tx mode largest
    w.flag(true); // reference_select
    w.flag(false).flag(true); // warped motion, reduced_tx_set
    w.flag(true).flag(false).flag(true); // LAST: global, translation
    w.flag(false).f(0, 3); // param 0: subexp 0 -> 0
    w.flag(false).f(2, 3); // param 1: subexp 2 -> +1
    for _ in 0..6 {
        w.flag(false);
    }
    w.flag(true).f(7, 16).flag(false).f(0, 3); // grain from slot 0, seed 7
    w.align();
    let mut payload = w.bytes.clone();
    payload.extend(std::iter::repeat_n(0xcc, 5));
    payload
}

fn temporal_unit(frames: &[(u8, Vec<u8>)]) -> Vec<u8> {
    let mut data = obu(obu_type::TEMPORAL_DELIMITER, &[]);
    data.extend(obu(obu_type::SEQUENCE_HEADER, &movie_sequence()));
    for (kind, payload) in frames {
        data.extend(obu(*kind, payload));
    }
    data
}

#[test]
fn obus_split_with_sizes_and_extensions() {
    let mut data = obu(obu_type::TEMPORAL_DELIMITER, &[]);
    data.extend([obu_type::PADDING << 3 | 0x6, 0b0100_1000, 2, 0xee, 0xee]);
    data.extend([obu_type::METADATA << 3, 1, 2, 3]);
    let obus = obus(&data).unwrap();
    assert_eq!(obus.len(), 3);
    assert_eq!(obus[0].obu_type, obu_type::TEMPORAL_DELIMITER);
    assert_eq!(obus[0].start, obus[0].end);
    assert_eq!(
        obus[1].layer,
        Some(Layer {
            temporal_id: 2,
            spatial_id: 1
        })
    );
    assert_eq!(&data[obus[1].start..obus[1].end], &[0xee, 0xee]);
    assert_eq!(&data[obus[2].start..obus[2].end], &[1, 2, 3]);
    assert_eq!(
        super::obus(&[0x80]),
        Err(Av1Error::Invalid("obu_forbidden_bit"))
    );
    assert_eq!(super::obus(&[0x12, 5, 0]), Err(Av1Error::Truncated));
}

#[test]
fn a_key_frame_reads_its_header_and_tiles() {
    let data = temporal_unit(&[(obu_type::FRAME, key_frame())]);
    let mut stream = Av1Stream::default();
    let mut seen = Vec::new();
    stream
        .temporal_unit(&data, |frame, seq, refs| {
            assert!(refs.slots.iter().all(|s| !s.valid));
            assert_eq!(seq.max_frame_width_minus_1, 1919);
            seen.push(frame.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(stream.sequence_generation, 1);
    let [frame] = seen.as_slice() else {
        panic!("one frame expected, got {}", seen.len());
    };
    let h = &frame.header;
    assert_eq!(h.frame_type, FrameType::Key);
    assert!(h.show_frame && h.error_resilient_mode && h.force_integer_mv);
    assert_eq!(h.refresh_frame_flags, 0xff);
    assert_eq!(
        (h.frame_width, h.frame_height, h.upscaled_width),
        (1920, 1080, 1920)
    );
    assert_eq!((h.mi_cols, h.mi_rows), (480, 270));
    assert_eq!(h.primary_ref_frame, frame::PRIMARY_REF_NONE);
    let t = &h.tile_info;
    assert!(t.uniform_tile_spacing);
    assert_eq!((t.tile_cols, t.tile_rows, t.tile_cols_log2), (2, 1, 1));
    assert_eq!(t.mi_col_starts, [0, 240, 480]);
    assert_eq!(t.mi_row_starts, [0, 270]);
    assert_eq!(t.width_in_sbs_minus_1, [14, 14]);
    assert_eq!(t.height_in_sbs_minus_1, [16]);
    assert_eq!(t.tile_size_bytes_minus_1, 3);
    assert_eq!(h.quantization.base_q_idx, 100);
    assert_eq!(h.loop_filter.level, [10, 12, 5, 6]);
    assert_eq!(h.loop_filter.sharpness, 2);
    assert_eq!(h.loop_filter.ref_deltas, [1, 3, 0, 0, -1, 0, -1, -1]);
    assert_eq!(h.cdef.damping_minus_3, 2);
    assert_eq!(h.cdef.bits, 1);
    assert_eq!(h.cdef.y_pri_strength[..2], [5, 9]);
    assert_eq!(h.cdef.y_sec_strength[..2], [3, 0]);
    assert_eq!(h.cdef.uv_sec_strength[..2], [1, 2]);
    assert_eq!(h.loop_restoration.lr_type, [2, 0, 1]);
    assert_eq!(
        (
            h.loop_restoration.lr_unit_shift,
            h.loop_restoration.lr_uv_shift
        ),
        (2, 1)
    );
    assert_eq!(h.tx_mode, 2);
    let g = &h.film_grain;
    assert!(g.apply_grain && g.update_grain && g.overlap_flag && !g.clip_to_restricted_range);
    assert_eq!(g.grain_seed, 0x1234);
    assert_eq!(g.num_y_points, 2);
    assert_eq!(g.point_y_value[..2], [16, 128]);
    assert_eq!(g.point_cb_scaling[0], 20);
    assert_eq!(g.ar_coeffs_y_plus_128[..4], [120, 130, 140, 150]);
    assert_eq!(g.ar_coeffs_cb_plus_128[..5], [1, 2, 3, 4, 5]);
    assert_eq!((g.cb_mult, g.cb_luma_mult, g.cb_offset), (128, 192, 256));
    assert_eq!(frame.tiles.len(), 2);
    let tile = |i: usize| &data[frame.tiles[i].offset..][..frame.tiles[i].size];
    assert_eq!(tile(0), [0xaa; 10]);
    assert_eq!(tile(1), [0xbb; 7]);
    assert!(
        stream
            .refs
            .slots
            .iter()
            .all(|s| s.valid && s.frame_type == FrameType::Key)
    );
}

#[test]
fn an_inter_frame_inherits_from_its_primary_reference() {
    let mut stream = Av1Stream::default();
    let key = temporal_unit(&[(obu_type::FRAME, key_frame())]);
    stream.temporal_unit(&key, |_, _, _| Ok(())).unwrap();
    let mut data = obu(obu_type::TEMPORAL_DELIMITER, &[]);
    data.extend(obu(obu_type::FRAME, &inter_frame()));
    let mut seen = Vec::new();
    stream
        .temporal_unit(&data, |frame, _, _| {
            seen.push(frame.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(stream.sequence_generation, 1);
    let h = &seen[0].header;
    assert_eq!(h.frame_type, FrameType::Inter);
    assert_eq!(h.order_hint, 1);
    assert_eq!(h.primary_ref_frame, 0);
    assert_eq!(h.ref_frame_idx, [0, 1, 2, 3, 4, 5, 6]);
    assert!(h.allow_high_precision_mv && h.use_ref_frame_mvs && h.disable_frame_end_update_cdf);
    assert_eq!(h.interpolation_filter, 1);
    assert_eq!(h.order_hints, [0; 8]);
    assert_eq!(h.ref_frame_sign_bias, 0);
    assert_eq!(h.loop_filter.ref_deltas, [1, 3, 0, 0, -1, 0, -1, -1]);
    assert!(h.reference_select && !h.skip_mode_present);
    assert_eq!(h.tx_mode, 1);
    assert_eq!(h.global_motion.gm_type[1], 1);
    assert_eq!(h.global_motion.gm_params[1], [0, 8192, 65536, 0, 0, 65536]);
    assert_eq!(h.global_motion.gm_type[2..], [0; 6]);
    let key_grain = stream.refs.slots[1].film_grain;
    assert_eq!(
        h.film_grain,
        FilmGrain {
            grain_seed: 7,
            update_grain: false,
            film_grain_params_ref_idx: 0,
            ..key_grain
        }
    );
    assert_eq!(seen[0].tiles.len(), 1);
    assert_eq!(&data[seen[0].tiles[0].offset..][..5], [0xcc; 5]);
    assert_eq!(stream.refs.slots[0].order_hint, 1);
    assert_eq!(stream.refs.slots[0].frame_type, FrameType::Inter);
    assert_eq!(stream.refs.slots[1].frame_type, FrameType::Key);
}

#[test]
fn showing_an_existing_key_frame_refreshes_every_slot() {
    let mut stream = Av1Stream::default();
    let key = temporal_unit(&[(obu_type::FRAME, key_frame())]);
    stream.temporal_unit(&key, |_, _, _| Ok(())).unwrap();
    let mut inter = obu(obu_type::TEMPORAL_DELIMITER, &[]);
    inter.extend(obu(obu_type::FRAME, &inter_frame()));
    stream.temporal_unit(&inter, |_, _, _| Ok(())).unwrap();
    let mut w = BitWriter::default();
    w.flag(true).f(1, 3).align(); // show the key frame in slot 1
    let mut data = obu(obu_type::TEMPORAL_DELIMITER, &[]);
    data.extend(obu(obu_type::FRAME_HEADER, &w.bytes));
    let mut seen = Vec::new();
    stream
        .temporal_unit(&data, |frame, _, _| {
            seen.push(frame.clone());
            Ok(())
        })
        .unwrap();
    let h = &seen[0].header;
    assert!(h.show_existing_frame && h.show_frame && seen[0].tiles.is_empty());
    assert_eq!(h.frame_to_show_map_idx, 1);
    assert_eq!(h.frame_type, FrameType::Key);
    assert_eq!(h.refresh_frame_flags, 0xff);
    assert_eq!(h.film_grain.grain_seed, 0x1234);
    assert!(
        stream
            .refs
            .slots
            .iter()
            .all(|s| s.frame_type == FrameType::Key)
    );
}

#[test]
fn a_frame_before_any_sequence_header_is_refused() {
    let mut stream = Av1Stream::default();
    let data = obu(obu_type::FRAME, &key_frame());
    assert_eq!(
        stream.temporal_unit(&data, |_, _, _| Ok(())),
        Err(Av1Error::NoSequenceHeader)
    );
}

#[test]
fn split_frame_header_and_tile_group_obus_make_one_frame() {
    let key = key_frame();
    let header_len = {
        let seq = SequenceHeader::parse(&movie_sequence()).unwrap();
        FrameHeader::parse(&seq, &References::default(), Layer::default(), &key)
            .unwrap()
            .1
    };
    let data = temporal_unit(&[
        (obu_type::FRAME_HEADER, key[..header_len].to_vec()),
        (obu_type::REDUNDANT_FRAME_HEADER, key[..header_len].to_vec()),
        (obu_type::TILE_GROUP, key[header_len..].to_vec()),
    ]);
    let mut stream = Av1Stream::default();
    let mut tiles = Vec::new();
    stream
        .temporal_unit(&data, |frame, _, _| {
            tiles = frame.tiles.clone();
            Ok(())
        })
        .unwrap();
    assert_eq!(tiles.len(), 2);
    assert_eq!(&data[tiles[0].offset..][..tiles[0].size], [0xaa; 10]);
    assert_eq!(&data[tiles[1].offset..][..tiles[1].size], [0xbb; 7]);
}
