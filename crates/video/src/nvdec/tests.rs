//! Differential tests: the expected headers were produced by the C++ reference builders (Lichii
//! `nx2`, src/video_core/host1x/codecs) fed the same synthetic registers and guest memory, built
//! with the fixes this port makes: VP9 `comp_inter` default probabilities, signed `delta_q`,
//! range coder carries, loop filter and segmentation resets on intra frames, frame size from
//! references, VP9 frame contexts reset and left unsaved by error-resilient frames, and H.264
//! 4:4:4 PPS scaling lists and wrapped scaling list deltas.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use bytemuck::Zeroable;

use super::picture::{
    H264DecoderContext, Vp8PictureInfo, Vp9EntropyProbs, Vp9FrameDimensions, Vp9NvdecEntropyProbs,
    Vp9PictureInfo, vp9_flags,
};
use super::registers::index;
use super::test_memory::{TestMemory, payload};
use super::vp9_tables::DEFAULT_PROBS;
use super::*;

const PICTURE_INFO: u64 = 0x10_0000;
const BITSTREAM: u64 = 0x20_0000;
const PROBS: u64 = 0x30_0000;

struct Step {
    regs: NvdecRegisters,
    memory: TestMemory,
    payload: Vec<u8>,
}

struct Scenario {
    codec: VideoCodec,
    steps: Vec<Step>,
}

fn luma_surface(id: u64) -> u64 {
    0x1000 + id * 0x40
}

fn chroma_surface(id: u64) -> u64 {
    0x3000 + id * 0x40
}

fn step(codec: VideoCodec, surfaces: [u64; 4], picture: &[u8], payload: Vec<u8>) -> Step {
    let mut regs = NvdecRegisters::default();
    regs.write(index::SET_CODEC_ID, codec.raw());
    regs.write(index::PICTURE_INFO_OFFSET, PICTURE_INFO >> 8);
    regs.write(index::FRAME_BITSTREAM_OFFSET, BITSTREAM >> 8);
    regs.write(index::VP9_PROB_TAB_BUFFER_OFFSET, PROBS >> 8);
    for i in 0..registers::SURFACE_COUNT {
        let id = surfaces.get(i).copied().unwrap_or(i as u64 + 16);
        regs.write(index::SURFACE_LUMA_OFFSETS + i, luma_surface(id));
        regs.write(index::SURFACE_CHROMA_OFFSETS + i, chroma_surface(id));
    }
    let mut memory = TestMemory::default();
    memory.put(PICTURE_INFO, picture);
    memory.put(BITSTREAM, &payload);
    Step {
        regs,
        memory,
        payload,
    }
}

fn h264_flags(fields: &[(u32, u64)]) -> u64 {
    fields
        .iter()
        .fold(0, |flags, &(shift, value)| flags | value << shift)
}

fn h264_step(context: &H264DecoderContext, seed: u8) -> Step {
    let mut slices = payload(context.stream_len as usize, seed);
    slices[..5].copy_from_slice(&[0, 0, 0, 1, 0x65]);
    step(
        VideoCodec::H264,
        [0, 1, 2, 3],
        bytemuck::bytes_of(context),
        slices,
    )
}

fn h264_progressive_flat_scaling() -> Scenario {
    let mut context = H264DecoderContext::zeroed();
    context.stream_len = 100;
    let params = &mut context.parameter_set;
    params.log2_max_pic_order_cnt_lsb_minus4 = 2;
    params.frame_mbs_only_flag = 1;
    params.pic_width_in_mbs = 80;
    params.frame_height_in_mbs = 45;
    params.entropy_coding_mode_flag = 1;
    params.num_refidx_l0_default_active = 2;
    params.deblocking_filter_control_present_flag = 1;
    params.luma_frame_offset = 0x4;
    params.chroma_frame_offset = 0x10;
    context.weight_scale_4x4 = [16; 0x60];
    context.weight_scale_8x8 = [16; 0x80];
    let flags = |frame_number| {
        h264_flags(&[
            (1, 1),
            (8, 4),
            (12, 1),
            (14, 0),
            (16, (-3i64 as u64) & 0x3F),
            (22, 2),
            (27, 2),
            (34, 5),
            (46, frame_number),
        ])
    };
    let steps = [5, 6, 0]
        .into_iter()
        .enumerate()
        .map(|(i, frame_number)| {
            context.parameter_set.set_flags(flags(frame_number));
            h264_step(&context, i as u8)
        })
        .collect();
    Scenario {
        codec: VideoCodec::H264,
        steps,
    }
}

fn h264_mbaff_scaling_weighted() -> Scenario {
    let mut context = H264DecoderContext::zeroed();
    context.stream_len = 90;
    context.lossless_flags = 0b10;
    let params = &mut context.parameter_set;
    params.delta_pic_order_always_zero_flag = 1;
    params.pic_width_in_mbs = 120;
    params.frame_height_in_mbs = 68;
    params.pic_order_present_flag = 1;
    params.num_refidx_l0_default_active = 5;
    params.num_refidx_l1_default_active = 2;
    params.redundant_pic_cnt_present_flag = 1;
    params.transform_8x8_mode_flag = 1;
    params.luma_top_offset = 0x1;
    params.luma_bot_offset = 0x21;
    params.chroma_top_offset = 0x41;
    params.chroma_bot_offset = 0x51;
    for (i, value) in context.weight_scale_4x4.iter_mut().enumerate() {
        *value = 6 + (i * 7 % 40) as u8;
    }
    for (i, value) in context.weight_scale_8x8.iter_mut().enumerate() {
        *value = 4 + (i * 13 % 60) as u8;
    }
    let flags = |frame_number| {
        h264_flags(&[
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (8, 0),
            (12, 3),
            (14, 1),
            (16, (-20i64 as u64) & 0x3F),
            (22, (-12i64 as u64) & 0x1F),
            (27, (-5i64 as u64) & 0x1F),
            (32, 2),
            (34, 16),
            (46, frame_number),
        ])
    };
    let steps = [1, 2]
        .into_iter()
        .enumerate()
        .map(|(i, frame_number)| {
            context.parameter_set.set_flags(flags(frame_number));
            h264_step(&context, 40 + i as u8)
        })
        .collect();
    Scenario {
        codec: VideoCodec::H264,
        steps,
    }
}

fn h264_field_poc_type_2() -> Scenario {
    let mut context = H264DecoderContext::zeroed();
    context.stream_len = 64;
    let params = &mut context.parameter_set;
    params.pic_width_in_mbs = 40;
    params.frame_height_in_mbs = 30;
    params.num_refidx_l0_default_active = 1;
    params.num_refidx_l1_default_active = 1;
    params.luma_bot_offset = 0x8;
    params.chroma_bot_offset = 0x9;
    for (i, value) in context.weight_scale_4x4.iter_mut().enumerate() {
        *value = 255 - (i * 11 % 200) as u8;
    }
    params.set_flags(h264_flags(&[
        (1, 1),
        (5, 1),
        (6, 1),
        (8, 12),
        (12, 1),
        (14, 2),
        (16, 25),
        (22, 10),
        (27, (-12i64 as u64) & 0x1F),
        (32, 1),
        (34, 7),
        (46, 9),
    ]));
    Scenario {
        codec: VideoCodec::H264,
        steps: vec![h264_step(&context, 99)],
    }
}

fn vp8_step(
    key_frame: bool,
    version: u8,
    size: (u16, u16),
    first_part_size: u32,
    seed: u8,
) -> Step {
    let mut info = Vp8PictureInfo::zeroed();
    info.key_frame = u8::from(key_frame);
    info.version = version;
    info.frame_width = size.0;
    info.frame_height = size.1;
    info.first_part_size = first_part_size;
    info.vld_buffer_size = 48 + u32::from(seed % 16);
    step(
        VideoCodec::Vp8,
        [4, 5, 6, 7],
        bytemuck::bytes_of(&info),
        payload(info.vld_buffer_size as usize, seed),
    )
}

fn vp8_key_and_inter_frames() -> Scenario {
    Scenario {
        codec: VideoCodec::Vp8,
        steps: vec![
            vp8_step(true, 0, (1280, 720), 0x2345, 1),
            vp8_step(false, 3, (1280, 720), 0x7_FFFF, 2),
            vp8_step(true, 9, (4000, 0x4321), 0x8_1234, 3),
        ],
    }
}

fn nvdec_probs(
    probs: &Vp9EntropyProbs,
    seg_tree: [u8; 7],
    seg_pred: [u8; 3],
) -> Vp9NvdecEntropyProbs {
    let mut nvdec = Vp9NvdecEntropyProbs::zeroed();
    nvdec.inter_mode_prob = probs.inter_mode_prob;
    nvdec.intra_inter_prob = probs.intra_inter_prob;
    nvdec.tx_8x8_prob = probs.tx_8x8_prob;
    nvdec.tx_16x16_prob = probs.tx_16x16_prob;
    nvdec.tx_32x32_prob = probs.tx_32x32_prob;
    nvdec.partition_prob = probs.partition_prob;
    nvdec.switchable_interp_prob = probs.switchable_interp_prob;
    nvdec.comp_inter_prob = probs.comp_inter_prob;
    nvdec.skip_probs = probs.skip_probs;
    nvdec.joints = probs.joints;
    nvdec.sign = probs.sign;
    nvdec.class_0 = probs.class_0;
    nvdec.fr = probs.fr;
    nvdec.class_0_hp = probs.class_0_hp;
    nvdec.high_precision = probs.high_precision;
    nvdec.classes = probs.classes;
    nvdec.class_0_fr = probs.class_0_fr;
    nvdec.pred_bits = probs.prob_bits;
    nvdec.single_ref_prob = probs.single_ref_prob;
    nvdec.comp_ref_prob = probs.comp_ref_prob;
    for (i, row) in probs.y_mode_prob.chunks(9).enumerate() {
        nvdec.y_mode_prob_e0e7[i].copy_from_slice(&row[..8]);
        nvdec.y_mode_prob_e8[i] = row[8];
    }
    for (dst, src) in nvdec
        .coef_probs
        .chunks_mut(4)
        .zip(probs.coef_probs.chunks(3))
    {
        dst[..3].copy_from_slice(src);
    }
    nvdec.mb_segment_tree_probs = seg_tree;
    nvdec.segment_pred_probs = seg_pred;
    nvdec
}

/// Default probabilities with the given number of pseudo-random bytes replaced by 1..=255.
fn perturbed_probs(seed: u32, count: usize) -> Vp9EntropyProbs {
    let mut probs = DEFAULT_PROBS;
    let bytes = bytemuck::bytes_of_mut(&mut probs);
    let mut state = seed.wrapping_mul(0x9E37_79B9) | 1;
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let position = state as usize % bytes.len();
        bytes[position] = 1 + (state >> 16) as u8 % 255;
    }
    probs
}

struct Vp9Frame {
    flags: u32,
    size: (i16, i16),
    ref_sizes: [(i16, i16); 3],
    surfaces: [u64; 4],
    probs: Vp9EntropyProbs,
    edit: fn(&mut Vp9PictureInfo),
}

fn dimensions((width, height): (i16, i16)) -> Vp9FrameDimensions {
    Vp9FrameDimensions {
        width,
        height,
        luma_pitch: width,
        chroma_pitch: width / 2,
    }
}

fn vp9_scenario(frames: &[Vp9Frame]) -> Scenario {
    let steps = frames
        .iter()
        .enumerate()
        .map(|(i, frame)| {
            let mut picture = Vp9PictureInfo::zeroed();
            picture.bitstream_size = 32 + i as u32 * 3;
            picture.vp9_flags = frame.flags;
            picture.current_frame_size = dimensions(frame.size);
            picture.last_frame_size = dimensions(frame.ref_sizes[0]);
            picture.golden_frame_size = dimensions(frame.ref_sizes[1]);
            picture.alt_frame_size = dimensions(frame.ref_sizes[2]);
            (frame.edit)(&mut picture);
            let mut step = step(
                VideoCodec::Vp9,
                frame.surfaces,
                bytemuck::bytes_of(&picture),
                payload(picture.bitstream_size as usize, 0x50 + i as u8),
            );
            let probs = nvdec_probs(
                &frame.probs,
                [10, 255, 30, 200, 255, 1, 128],
                [255, 77, 254],
            );
            step.memory.put_pod(PROBS, &probs);
            step
        })
        .collect();
    Scenario {
        codec: VideoCodec::Vp9,
        steps,
    }
}

const HD: (i16, i16) = (1920, 1080);

fn vp9_lf_defaults(picture: &mut Vp9PictureInfo) {
    picture.loop_filter.mode_ref_delta_enabled = 1;
    picture.loop_filter.ref_deltas = [1, 0, -1, -1];
}

fn vp9_segmentation(picture: &mut Vp9PictureInfo) {
    let segmentation = &mut picture.segmentation;
    segmentation.enabled = 1;
    segmentation.update_map = 1;
    segmentation.temporal_update = 1;
    segmentation.feature_enabled = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [0; 4],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ];
    segmentation.feature_data = [
        [-20, 0, 0, 0],
        [0, -9, 0, 0],
        [0, 0, 2, 0],
        [0; 4],
        [255, 63, 3, 0],
        [0; 4],
        [7, 0, 1, 0],
        [0, 33, 0, 0],
    ];
}

fn vp9_inter_frames_hidden_frame_segmentation() -> Scenario {
    use vp9_flags::*;
    vp9_scenario(&[
        Vp9Frame {
            flags: IS_KEY_FRAME,
            size: HD,
            ref_sizes: [HD; 3],
            surfaces: [0, 0, 0, 0],
            probs: perturbed_probs(1, 40),
            edit: |p| {
                vp9_lf_defaults(p);
                p.tx_mode = 4;
                p.first_level = 10;
                p.sharpness_level = 2;
                p.base_q_index = 60;
                p.log2_tile_cols = 1;
                p.log2_tile_rows = 2;
            },
        },
        Vp9Frame {
            flags: LAST_FRAME_IS_KEY_FRAME | LAST_SHOW_FRAME,
            size: HD,
            ref_sizes: [HD; 3],
            surfaces: [0, 0, 0, 1],
            probs: perturbed_probs(2, 60),
            edit: |p| {
                vp9_lf_defaults(p);
                p.tx_mode = 3;
                p.interp_filter = 4;
                p.allow_high_precision_mv = 1;
                p.base_q_index = 80;
                p.log2_tile_cols = 2;
            },
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME,
            size: HD,
            ref_sizes: [HD; 3],
            surfaces: [1, 0, 0, 2],
            probs: perturbed_probs(3, 25),
            edit: |p| {
                vp9_lf_defaults(p);
                p.tx_mode = 2;
                p.interp_filter = 1;
                p.base_q_index = 90;
            },
        },
        Vp9Frame {
            flags: 0,
            size: HD,
            ref_sizes: [HD; 3],
            surfaces: [1, 2, 0, 3],
            probs: perturbed_probs(4, 80),
            edit: |p| {
                p.loop_filter.mode_ref_delta_enabled = 1;
                p.loop_filter.ref_deltas = [2, 0, -2, -1];
                p.loop_filter.mode_deltas = [1, -1];
                p.tx_mode = 4;
                p.interp_filter = 4;
                p.reference_mode = 2;
                p.ref_frame_sign_bias = [0, 0, 1, 1];
                p.base_q_index = 100;
                p.y_dc_delta_q = -3i8 as u8;
                p.uv_dc_delta_q = 2;
                p.uv_ac_delta_q = -1i8 as u8;
                vp9_segmentation(p);
            },
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME,
            size: HD,
            ref_sizes: [HD; 3],
            surfaces: [3, 2, 1, 4],
            probs: perturbed_probs(5, 30),
            edit: |p| {
                p.loop_filter.mode_ref_delta_enabled = 1;
                p.loop_filter.ref_deltas = [2, 0, -2, -1];
                p.loop_filter.mode_deltas = [1, -1];
                p.tx_mode = 1;
                p.reference_mode = 1;
                p.ref_frame_sign_bias = [0, 0, 1, 0];
                p.allow_high_precision_mv = 1;
                p.base_q_index = 70;
                vp9_segmentation(p);
            },
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME,
            size: HD,
            ref_sizes: [HD; 3],
            surfaces: [4, 1, 2, 5],
            probs: perturbed_probs(6, 50),
            edit: |p| {
                vp9_lf_defaults(p);
                p.tx_mode = 4;
                p.interp_filter = 2;
                p.base_q_index = 50;
            },
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME,
            size: HD,
            ref_sizes: [HD; 3],
            surfaces: [5, 1, 2, 6],
            probs: perturbed_probs(7, 10),
            edit: vp9_lf_defaults,
        },
    ])
}

const SD: (i16, i16) = (640, 360);
const SMALL: (i16, i16) = (320, 180);
const WIDE: (i16, i16) = (1280, 720);

fn vp9_resolution_changes_intra_only_error_resilient() -> Scenario {
    use vp9_flags::*;
    vp9_scenario(&[
        Vp9Frame {
            flags: IS_KEY_FRAME,
            size: SD,
            ref_sizes: [SD; 3],
            surfaces: [0, 0, 0, 0],
            probs: perturbed_probs(11, 20),
            edit: |p| p.lossless = 1,
        },
        Vp9Frame {
            flags: LAST_FRAME_IS_KEY_FRAME | LAST_SHOW_FRAME | ERROR_RESILIENT_MODE,
            size: SD,
            ref_sizes: [SD; 3],
            surfaces: [0, 0, 0, 1],
            probs: perturbed_probs(12, 20),
            edit: |p| {
                vp9_lf_defaults(p);
                p.base_q_index = 40;
            },
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME | FRAME_SIZE_CHANGED,
            size: SMALL,
            ref_sizes: [SD; 3],
            surfaces: [1, 0, 0, 2],
            probs: perturbed_probs(13, 20),
            edit: |p| p.base_q_index = 41,
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME,
            size: SMALL,
            ref_sizes: [SD, SMALL, SD],
            surfaces: [1, 2, 0, 3],
            probs: perturbed_probs(14, 20),
            edit: |p| p.base_q_index = 42,
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME | INTRA_ONLY,
            size: SMALL,
            ref_sizes: [SMALL; 3],
            surfaces: [3, 2, 0, 4],
            probs: perturbed_probs(15, 20),
            edit: |p| {
                vp9_lf_defaults(p);
                p.loop_filter.mode_deltas = [0, 2];
                p.base_q_index = 43;
            },
        },
        Vp9Frame {
            flags: 0,
            size: SMALL,
            ref_sizes: [SMALL; 3],
            surfaces: [3, 4, 0, 5],
            probs: perturbed_probs(16, 20),
            edit: |p| p.base_q_index = 44,
        },
        Vp9Frame {
            flags: IS_KEY_FRAME | LAST_SHOW_FRAME,
            size: WIDE,
            ref_sizes: [SMALL; 3],
            surfaces: [0, 0, 0, 0],
            probs: perturbed_probs(17, 20),
            edit: |p| {
                p.base_q_index = 45;
                p.log2_tile_cols = 2;
            },
        },
        Vp9Frame {
            flags: LAST_FRAME_IS_KEY_FRAME | LAST_SHOW_FRAME,
            size: WIDE,
            ref_sizes: [WIDE; 3],
            surfaces: [0, 0, 0, 1],
            probs: perturbed_probs(18, 20),
            edit: |p| p.base_q_index = 46,
        },
    ])
}

fn vp9_compound_reference_after_key_frame() -> Scenario {
    use vp9_flags::*;
    vp9_scenario(&[
        Vp9Frame {
            flags: IS_KEY_FRAME,
            size: WIDE,
            ref_sizes: [WIDE; 3],
            surfaces: [0, 0, 0, 0],
            probs: DEFAULT_PROBS,
            edit: |p| p.tx_mode = 4,
        },
        Vp9Frame {
            flags: LAST_FRAME_IS_KEY_FRAME | LAST_SHOW_FRAME,
            size: WIDE,
            ref_sizes: [WIDE; 3],
            surfaces: [0, 0, 0, 1],
            probs: perturbed_probs(21, 600),
            edit: |p| {
                p.tx_mode = 4;
                p.interp_filter = 4;
                p.reference_mode = 2;
                p.ref_frame_sign_bias = [0, 0, 1, 1];
                p.allow_high_precision_mv = 1;
                p.base_q_index = 120;
            },
        },
        Vp9Frame {
            flags: LAST_SHOW_FRAME,
            size: WIDE,
            ref_sizes: [WIDE; 3],
            surfaces: [1, 0, 0, 2],
            probs: perturbed_probs(22, 600),
            edit: |p| {
                p.tx_mode = 4;
                p.reference_mode = 2;
                p.ref_frame_sign_bias = [0, 0, 1, 1];
            },
        },
    ])
}

fn scenarios() -> Vec<(&'static str, Scenario)> {
    vec![
        (
            "h264_progressive_flat_scaling",
            h264_progressive_flat_scaling(),
        ),
        ("h264_mbaff_scaling_weighted", h264_mbaff_scaling_weighted()),
        ("h264_field_poc_type_2", h264_field_poc_type_2()),
        ("vp8_key_and_inter_frames", vp8_key_and_inter_frames()),
        (
            "vp9_inter_frames_hidden_frame_segmentation",
            vp9_inter_frames_hidden_frame_segmentation(),
        ),
        (
            "vp9_resolution_changes_intra_only_error_resilient",
            vp9_resolution_changes_intra_only_error_resilient(),
        ),
        (
            "vp9_compound_reference_after_key_frame",
            vp9_compound_reference_after_key_frame(),
        ),
    ]
}

struct Expected {
    header: &'static str,
    hidden: bool,
    offsets: SurfaceOffsets,
    dimensions: Option<(u32, u32)>,
}

const fn progressive(luma: u64, chroma: u64) -> SurfaceOffsets {
    SurfaceOffsets::Progressive { luma, chroma }
}

const fn interlaced(offsets: [u64; 4]) -> SurfaceOffsets {
    SurfaceOffsets::Interlaced {
        luma_top: offsets[0],
        luma_bottom: offsets[1],
        chroma_top: offsets[2],
        chroma_bottom: offsets[3],
    }
}

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn assert_matches_reference(scenario: Scenario, expected: &[Expected]) {
    assert_eq!(scenario.steps.len(), expected.len());
    let mut builder = FrameBuilder::new(scenario.codec);
    assert_eq!(builder.codec(), scenario.codec);
    for (i, (step, expected)) in scenario.steps.iter().zip(expected).enumerate() {
        let source = match scenario.codec {
            VideoCodec::Vp9 => i.saturating_sub(1),
            _ => i,
        };
        let mut data = from_hex(expected.header);
        data.extend_from_slice(&scenario.steps[source].payload);
        let frame = builder.build(&step.regs, &step.memory);
        assert_eq!(frame.data, data, "frame {i}");
        assert_eq!(frame.hidden, expected.hidden, "frame {i}");
        assert_eq!(frame.offsets, expected.offsets, "frame {i}");
        let dimensions = frame.dimensions.map(|d| (d.width, d.height));
        assert_eq!(dimensions, expected.dimensions, "frame {i}");
    }
}

#[test]
fn h264_progressive_frames_with_flat_scaling_lists_match_reference() {
    assert_matches_reference(
        h264_progressive_flat_scaling(),
        H264_PROGRESSIVE_FLAT_SCALING,
    );
}

#[test]
fn h264_mbaff_frames_with_scaling_lists_and_weighted_prediction_match_reference() {
    assert_matches_reference(h264_mbaff_scaling_weighted(), H264_MBAFF_SCALING_WEIGHTED);
}

#[test]
fn h264_field_picture_with_poc_type_2_matches_reference() {
    assert_matches_reference(h264_field_poc_type_2(), H264_FIELD_POC_TYPE_2);
}

#[test]
fn vp8_key_and_inter_frames_match_reference() {
    assert_matches_reference(vp8_key_and_inter_frames(), VP8_KEY_AND_INTER_FRAMES);
}

#[test]
fn vp9_inter_frames_with_refresh_flags_hidden_frame_and_segmentation_match_reference() {
    assert_matches_reference(
        vp9_inter_frames_hidden_frame_segmentation(),
        VP9_INTER_FRAMES_HIDDEN_FRAME_SEGMENTATION,
    );
}

#[test]
fn vp9_resolution_changes_intra_only_and_error_resilient_frames_match_reference() {
    assert_matches_reference(
        vp9_resolution_changes_intra_only_error_resilient(),
        VP9_RESOLUTION_CHANGES_INTRA_ONLY_ERROR_RESILIENT,
    );
}

#[test]
fn vp9_compound_reference_after_key_frame_matches_reference() {
    assert_matches_reference(
        vp9_compound_reference_after_key_frame(),
        VP9_COMPOUND_REFERENCE_AFTER_KEY_FRAME,
    );
}

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: every call forwards to the system allocator with the caller's arguments unchanged.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn steady_state_builds_do_not_allocate() {
    for (name, scenario) in scenarios() {
        let mut builder = FrameBuilder::new(scenario.codec);
        for step in &scenario.steps {
            builder.build(&step.regs, &step.memory);
        }
        for (i, step) in scenario.steps.iter().enumerate() {
            let before = ALLOCATIONS.with(Cell::get);
            builder.build(&step.regs, &step.memory);
            assert_eq!(ALLOCATIONS.with(Cell::get), before, "{name} frame {i}");
        }
    }
}

const H264_PROGRESSIVE_FLAT_SCALING: &[Expected] = &[
    Expected {
        header: "0000016764001fac2d9405005b9000000168eb83c91843fffc21fffe10ffff087fff843fffc21fffc9",
        hidden: false,
        offsets: progressive(0x154400, 0x355000),
        dimensions: Some((1280, 720)),
    },
    Expected {
        header: "",
        hidden: false,
        offsets: progressive(0x154400, 0x355000),
        dimensions: Some((1280, 720)),
    },
    Expected {
        header: "0000016764001fac2d9405005b9000000168eb83c91843fffc21fffe10ffff087fff843fffc21fffc9",
        hidden: false,
        offsets: progressive(0x154400, 0x355000),
        dimensions: Some((1280, 720)),
    },
];

const H264_MBAFF_SCALING_WEIGHTED: &[Expected] = &[
    Expected {
        header: concat!(
            "0000016764001f91d5e701e0113200000168d99e0530cbe51c0a832098158e04e0a8138e04c0ac1c0271d078",
            "38138641302b1c0a81382a0218260560e0138e82c1c09c3209813010c1502705438158260c8271d0e0e04e0e",
            "0158261c09c150271c0ac1306415010e30702a0c82b04c381382a04e381302b0c82a1d28e0541904c0ac7027",
            "05409c70260560e0138e890d03e0420e81f8681d83e07605807e0e81f83a02f83e0760f81d83e0420e81f83a",
            "07e0e81f8681d83e0760f81d83e0761a0740fc1d03f0740fc0b00ec1d83e0760f8340fc1d03f0740841f03b0",
            "761a0740fc0b00ec35030017c1f0210740fc340ec1f03b02c03f0740fc1d0681d83e0760ec1f0210740fc1d0",
            "3f0740fc340f81d83e0760f81d83b0d03a07e0e81f83a07e0580760f81d83e0761a07e0e81f83a0420f81d83",
            "e02f83a07e0580761a0170",
        ),
        hidden: false,
        offsets: interlaced([0x180100, 0x182100, 0x384100, 0x385100]),
        dimensions: Some((1920, 1088)),
    },
    Expected {
        header: "",
        hidden: false,
        offsets: interlaced([0x180100, 0x182100, 0x384100, 0x385100]),
        dimensions: Some((1920, 1088)),
    },
];

const H264_FIELD_POC_TYPE_2: &[Expected] = &[Expected {
    header: concat!(
        "0000016764001fac1aca0503c900000168ca4419428184c2e04302c810808417021810c0861702101080b204",
        "30bc04705c059816408402c8043810c0860430b810808405902185e02f82e043019204201641701660430218",
        "5c08404202c810c2f01dc1702180c901640420b810c059810c2e04202101640861780e30b810c0b204202105",
        "c08602cc0861702100b201920430bc05985c08605902101082e04302180b30b80b202100c902185c33",
    ),
    hidden: false,
    offsets: interlaced([0x15c000, 0x15c800, 0x35c000, 0x35c900]),
    dimensions: Some((640, 480)),
}];

const VP8_KEY_AND_INTER_FRAMES: &[Expected] = &[
    Expected {
        header: "b068049d012a0005d002",
        hidden: false,
        offsets: progressive(0x11c000, 0x31c000),
        dimensions: Some((1280, 720)),
    },
    Expected {
        header: "f7ffff",
        hidden: false,
        offsets: progressive(0x11c000, 0x31c000),
        dimensions: Some((1280, 720)),
    },
    Expected {
        header: "9246029d012aa00f2103",
        hidden: false,
        offsets: progressive(0x11c000, 0x31c000),
        dimensions: Some((4000, 17185)),
    },
];

const VP9_INTER_FRAMES_HIDDEN_FRAME_SEGMENTATION: &[Expected] = &[
    Expected {
        header: concat!(
            "824983420077f0437614a3c0b004e077bfda20e2bf4273926cc6f748a161d66fd560f813db4f236cf34aed6e",
            "06133d92cdf6f03f2cd4adb7d85c1bfe7f5c4f22f9f20d9d3bf79bff2091adcf75f5c8211bfff7b2616affa6",
            "2ebb9e2c00",
        ),
        hidden: false,
        offsets: progressive(0x100000, 0x300000),
        dimensions: Some((1920, 1080)),
    },
    Expected {
        header: concat!(
            "824983420077f0437614a3c0b004e077bfda20e2bf4273926cc6f748a161d66fd560f813db4f236cf34aed6e",
            "06133d92cdf6f03f2cd4adb7d85c1bfe7f5c4f22f9f20d9d3bf79bff2091adcf75f5c8211bfff7b2616affa6",
            "2ebb9e2c00",
        ),
        hidden: false,
        offsets: progressive(0x104000, 0x304000),
        dimensions: Some((1920, 1080)),
    },
    Expected {
        header: concat!(
            "86004092f001280600b56db6e0d5d5466b8db507c057c817fd5dcfd9df6569e7f8ddb3ada375de761e9cc71e",
            "5bffb50445ee60922edf07a738c15df89e7f2ca8bfb9397f75bbfdef1ebf3fc7ee67ec5e796e7c4ffc773c34",
            "a22a1f80bda7354d9551e77ff5b5ebde325ff9bf55e47c761efb3fbda588bcd53b6a863d05efa1c3a9dfd37f",
            "e369f8764ace1c4db4ae9f83a3e1b78813bff8fb75ee5f1e5f938ecfafcb4f0386b257f3db23cbbda8d4333c",
            "e85b6ff92ac11adecda7fa12db7800",
        ),
        hidden: false,
        offsets: progressive(0x108000, 0x308000),
        dimensions: Some((1920, 1080)),
    },
    Expected {
        header: concat!(
            "840040490e8025a0000a40568eeb5f8def101638313956d7191334b0af51e6ce18375af0bcfbac346c36aa85",
            "ef7964c4a8b82000",
        ),
        hidden: true,
        offsets: progressive(0x10c000, 0x30c000),
        dimensions: Some((1920, 1080)),
    },
    Expected {
        header: concat!(
            "860040d67001c22150506c93c91f0a47b9080e02a6ffa8a424c307fefef08394c2400ef07efb6c2f204bcd7a",
            "8d8fa3beac6f2650b365c20fb97d782a9dd8da0f1eba6de775dd6ebb8ee3ed9c34d67f7b8ad2c8f7b5373ff3",
            "6a51bff0ad512d9b6e8de3ada78ae7a0e5a9e65ffc5df7a7dc3cbefff9929f3b3c747cf0fdf51ffc197fc397",
            "dd576b7bbb6e3c879e2f3f21bf2bafeedf91ffed7be774f47efdbfbea7edb603c75db4a07da50ee1bc7f8783",
            "00feb92d404511e8564d56f4fff2637c91406b76e0772d8c6fb606f7b7dba651abb27890d9b25b8a6aeae498",
            "72d0f7eeca986b21a2b0a0494bb7c0cf2db6dfc3dbbf1ac5f3e727d9586e256af560fa0f3d477fc7c3fbeabe",
            "1df408fe1f4ea287700000",
        ),
        hidden: false,
        offsets: progressive(0x110000, 0x310000),
        dimensions: Some((1920, 1080)),
    },
    Expected {
        header: concat!(
            "860040d28c0048c38523dc840701537fc001bc3c3bbb6d7b77854eafe7e766d14aad68aa575c754e8c648f30",
            "31c86cab6f2bbfdb1d89f9fc3fe49d39e1c62dab3ce01b52d96026cfda97e78c4fafadc5f8eff4f3b87989ba",
            "0aae33f95c7e0a4a4c9f39ffe781e269cf75f8e78d52ad5235bf9255f6f6efd9f8f73c7bac3351bf5000",
        ),
        hidden: false,
        offsets: progressive(0x114000, 0x314000),
        dimensions: Some((1920, 1080)),
    },
    Expected {
        header: concat!(
            "8600410a2c007048340401900013007efe5647aa4e1d485bdf64721f7b0fd967358f13d76e81aefc37c7bbaf",
            "d58ce4b78f9b9e37bfb57e90dd09561bdf8f554fe1f1e5d96d1fdb5efb0aaa88e54721f56aee307c7d0f7d74",
            "4e77b6b6badb87576f5a9f60f9dfb3baf6e63b3f5cfc66fe355dbffce766fc567e0a29f0d85fb9dcbf57b22f",
            "21e0325f2f2ec2cee8ff8605d6d6034dcb9f390339fea73f142f72890b7f3e6a018800",
        ),
        hidden: false,
        offsets: progressive(0x118000, 0x318000),
        dimensions: Some((1920, 1080)),
    },
];

const VP9_RESOLUTION_CHANGES_INTRA_ONLY_ERROR_RESILIENT: &[Expected] = &[
    Expected {
        header: "824983420027f0167600000000068078ff8d1db2665d33a258bf4000",
        hidden: false,
        offsets: progressive(0x100000, 0x300000),
        dimensions: Some((640, 360)),
    },
    Expected {
        header: "824983420027f0167600000000068078ff8d1db2665d33a258bf4000",
        hidden: false,
        offsets: progressive(0x104000, 0x304000),
        dimensions: Some((640, 360)),
    },
    Expected {
        header: "8701024800045000009019ef5587465e6fcf47cbca76e50074c10000",
        hidden: false,
        offsets: progressive(0x108000, 0x308000),
        dimensions: Some((640, 360)),
    },
    Expected {
        header: "86008090009f8059830002900008001d3ee018b8e0ed9be7dbd57b6f366000",
        hidden: false,
        offsets: progressive(0x10c000, 0x30c000),
        dimensions: Some((320, 180)),
    },
    Expected {
        header: "860040910600054000171d3d783a746e1f7076cdf2eaae95ea7fd95b2d31448000",
        hidden: false,
        offsets: progressive(0x110000, 0x310000),
        dimensions: Some((320, 180)),
    },
    Expected {
        header: "84893068404027e0166d0060842b00006011b2b5cc5e68219d2f2f5000",
        hidden: true,
        offsets: progressive(0x114000, 0x314000),
        dimensions: Some((320, 180)),
    },
    Expected {
        header: "860000920c000b00002c166e48c1ac539f664753896f49662acd779d770d0c00",
        hidden: false,
        offsets: progressive(0x100000, 0x300000),
        dimensions: Some((320, 180)),
    },
    Expected {
        header: "82498342004ff02cf60005a18001c010f9ebe1a60000",
        hidden: false,
        offsets: progressive(0x104000, 0x304000),
        dimensions: Some((1280, 720)),
    },
];

const VP9_COMPOUND_REFERENCE_AFTER_KEY_FRAME: &[Expected] = &[
    Expected {
        header: "82498342004ff02cf6000000000180700000",
        hidden: false,
        offsets: progressive(0x100000, 0x300000),
        dimensions: Some((1280, 720)),
    },
    Expected {
        header: "82498342004ff02cf6000000000180700000",
        hidden: false,
        offsets: progressive(0x104000, 0x304000),
        dimensions: Some((1280, 720)),
    },
    Expected {
        header: concat!(
            "860040d6f00078000eac7ffd6f7a5ff53f7f9f7d4fdf2bff27ef27e7a5ebe17fe63ff7dfddf6075f27fe23fc",
            "3ff734fa7f83de3bbfafffd0f76fff5def94f3f1fbf77ac6fff86a9eb91fbcc7db9af67f60fbef7d8ffe0bae",
            "9fbe4ffaf5d7d74ffd5ffd9f5cefbff7ffcdef94ffd36feb3fd2f7daf9eabef21efeef3d1f07ffff7dfbfef0",
            "df7b3e32bdfd7ed5f727efdbffc4f9179cc7dfc7ff09feee9f4bc8bfd5fbc9ffecf8f0bffebefbec7f9ffdaf",
            "5f2f8f1fd723ef39bf71f77df775ffe2fc3f759ff81f4fef2bfff5fbc0ffcbe7ed7df65c6df9e0bbfebf7dcf",
            "91ffccf7afffdbfff9fdf75fec7dd9edf0befb4ffe9ffc7fb93f72bfe5fde7f3fe9fe46bd573771ff3efb1ff",
            "defb91ffeff3a9e70dbfe9fb98fbdb7e3ffbfe3abff93dba6fb80fbc17b679efb3b7efbdff92efa4f74bef87",
            "ef8aeadfb88fba1ffdbfff6fce93be33ee7fbd7f79efbcef1f47ec2ffe679c87fc47d5f9d57fb9a7bfd3aff3",
            "faffefbe4fb3dfec4cb7ffb1fd15ba47cf5df33cc27cc9ffc32d4ee7a7314a79c4b3f6f3dd72541ced9f3a8f",
            "fb430f27fef4e7ccf948fccab09ffa2e151e7fcfcf4be6784f0fcff6f95078c1e72df25b99cffce5f2ffcdbf",
            "a6b99ffdd7fea3b775e7776de064f5dc036c6f6bbbfc4a7d354836d11fb4d26f16bcd59ff39db897da7febaa",
            "b728d675c48a3f6f6e2bfee2e7f0f3d176dfbb547ceb5ffbdf96b7f6cfabfe1fcf39b63179bfd3bcf45db93e",
            "da9ade07e7f9f3a773b4fbc7cf5d56437bede9190f97bfeb2e7d167e67fc03cc57cb2bce2be5b98730a92ffa",
            "a7984f312e76b679b7f33cee2db8f7ef7e487265b7effe67f84c18fd8e0fd8529ff999399f147fa82aa6b0d2",
            "9e588d03078ff42ff93e7d3b6c573b13fa779a476b8bf48f2d6ff506def7250bcfc3f25bff159bf6e2f2eef3",
            "e7f3d170b1f9f69fc878bbe56bceeefc3cfe4ff2dfc9fe37f1be6be6e73445b5667cf3fe55cf63cfe1d5f63b",
            "f71779f3ffed9fe2b6ebbc25bf54a4d84cd4a3b3dcff3ffd65fd1dfdef5306799f9ff3fad58e8bce8bfbdcdf",
            "bfdcbe71bf3c3e1fab3abffef7e40d584f94d7963ff595e0cf21dbc3f3dfbf8df34479567fe7365adbf2799b",
            "7fe97da7fba39fe1e553e3c72fdcd293fae93ad79c933b43ff63ffabfb163537fb67fccedf03cf18ff96f42e",
            "757dbe73cb9765af2c4b3ddac1934bcff1b567ff6fea3cdddcfedfef4ffd16db4ff96f2eeb682b1d29b3979b",
            "ebcdefdddff84936e7f597f33fd419ec4ff446dcb79e317d7fee8e766f9e133da1681b2d7fa2dbd0a47bc2be",
            "f3506db13e07e73d94631ce4fe8befc7902e95e76387e3a400",
        ),
        hidden: false,
        offsets: progressive(0x108000, 0x308000),
        dimensions: Some((1280, 720)),
    },
];
