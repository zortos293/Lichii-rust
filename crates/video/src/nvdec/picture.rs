//! The picture parameter structures a guest's NVDEC driver writes to guest memory, laid out
//! byte-for-byte as the engine reads them.

use bytemuck::{Pod, Zeroable};

fn bits(value: u64, shift: u32, width: u32) -> u64 {
    value >> shift & ((1 << width) - 1)
}

fn signed_bits(value: u64, shift: u32, width: u32) -> i64 {
    ((value << (64 - shift - width)) as i64) >> (64 - width)
}

/// H.264 sequence and picture parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct H264ParameterSet {
    pub log2_max_pic_order_cnt_lsb_minus4: i32,
    pub delta_pic_order_always_zero_flag: i32,
    pub frame_mbs_only_flag: i32,
    pub pic_width_in_mbs: u32,
    pub frame_height_in_mbs: u32,
    /// `tile_format` (bits 0..2), `gob_height` (2..5), reserved surface format bits.
    pub surface_format: u32,
    pub entropy_coding_mode_flag: u32,
    pub pic_order_present_flag: i32,
    pub num_refidx_l0_default_active: i32,
    pub num_refidx_l1_default_active: i32,
    pub deblocking_filter_control_present_flag: i32,
    pub redundant_pic_cnt_present_flag: i32,
    pub transform_8x8_mode_flag: u32,
    pub pitch_luma: u32,
    pub pitch_chroma: u32,
    /// Plane offsets in units of 256 bytes, relative to the surface's register offset.
    pub luma_top_offset: u32,
    pub luma_bot_offset: u32,
    pub luma_frame_offset: u32,
    pub chroma_top_offset: u32,
    pub chroma_bot_offset: u32,
    pub chroma_frame_offset: u32,
    pub hist_buffer_size: u32,
    /// A 64-bit little-endian bit field split in two words, since the engine only aligns it to 4.
    pub flags: [u32; 2],
}

impl H264ParameterSet {
    /// The 64-bit flags word.
    pub fn flags(&self) -> u64 {
        u64::from(self.flags[0]) | u64::from(self.flags[1]) << 32
    }

    /// Sets the 64-bit flags word.
    pub fn set_flags(&mut self, flags: u64) {
        self.flags = [flags as u32, (flags >> 32) as u32];
    }

    pub fn mbaff_frame(&self) -> bool {
        bits(self.flags(), 0, 1) != 0
    }

    pub fn direct_8x8_inference(&self) -> bool {
        bits(self.flags(), 1, 1) != 0
    }

    pub fn weighted_pred(&self) -> bool {
        bits(self.flags(), 2, 1) != 0
    }

    pub fn constrained_intra_pred(&self) -> bool {
        bits(self.flags(), 3, 1) != 0
    }

    pub fn ref_pic(&self) -> bool {
        bits(self.flags(), 4, 1) != 0
    }

    pub fn field_pic(&self) -> bool {
        bits(self.flags(), 5, 1) != 0
    }

    pub fn bottom_field(&self) -> bool {
        bits(self.flags(), 6, 1) != 0
    }

    pub fn second_field(&self) -> bool {
        bits(self.flags(), 7, 1) != 0
    }

    pub fn log2_max_frame_num_minus4(&self) -> u32 {
        bits(self.flags(), 8, 4) as u32
    }

    pub fn chroma_format_idc(&self) -> u32 {
        bits(self.flags(), 12, 2) as u32
    }

    pub fn pic_order_cnt_type(&self) -> u32 {
        bits(self.flags(), 14, 2) as u32
    }

    pub fn pic_init_qp_minus26(&self) -> i32 {
        signed_bits(self.flags(), 16, 6) as i32
    }

    pub fn chroma_qp_index_offset(&self) -> i32 {
        signed_bits(self.flags(), 22, 5) as i32
    }

    pub fn second_chroma_qp_index_offset(&self) -> i32 {
        signed_bits(self.flags(), 27, 5) as i32
    }

    pub fn weighted_bipred_idc(&self) -> u32 {
        bits(self.flags(), 32, 2) as u32
    }

    /// Index of the surface registers the current picture decodes into.
    pub fn curr_pic_idx(&self) -> usize {
        bits(self.flags(), 34, 7) as usize
    }

    pub fn curr_col_idx(&self) -> u32 {
        bits(self.flags(), 41, 5) as u32
    }

    pub fn frame_number(&self) -> u32 {
        bits(self.flags(), 46, 16) as u32
    }

    pub fn frame_surfaces(&self) -> bool {
        bits(self.flags(), 62, 1) != 0
    }

    pub fn output_memory_layout(&self) -> bool {
        bits(self.flags(), 63, 1) != 0
    }
}

/// One H.264 decoded picture buffer entry.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct H264DpbEntry {
    /// `index` (bits 0..7), `col_idx` (7..12), `state` (12..14), `is_long_term` (14),
    /// `non_existing` (15), `is_field` (16), top/bottom field marking (17..21, 21..25),
    /// `output_memory_layout` (25).
    pub flags: u32,
    pub field_order_cnt: [u32; 2],
    pub frame_idx: u32,
}

/// NVDEC display (post-processing) parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct DisplayParam {
    pub output_flags: u32,
    pub output_top: [i32; 2],
    pub output_bottom: [i32; 2],
    pub histogram_start: u32,
    pub histogram_end: u32,
}

/// The H.264 picture info structure at `PICTURE_INFO_OFFSET`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct H264DecoderContext {
    pub reserved0: [u32; 13],
    pub eos: [u8; 16],
    pub explicit_eos_present_flag: u8,
    pub hint_dump_en: u8,
    pub reserved1: [u8; 2],
    /// Bytes of slice data at `FRAME_BITSTREAM_OFFSET`, Annex B start codes included.
    pub stream_len: u32,
    pub slice_count: u32,
    pub mbhist_buffer_size: u32,
    pub gptimer_timeout_value: u32,
    pub parameter_set: H264ParameterSet,
    pub curr_field_order_cnt: [i32; 2],
    pub dpb: [H264DpbEntry; 16],
    /// Six 4x4 scaling lists in raster order.
    pub weight_scale_4x4: [u8; 0x60],
    /// Two 8x8 scaling lists in raster order.
    pub weight_scale_8x8: [u8; 0x80],
    pub num_inter_view_refs: [u8; 2],
    pub reserved2: [u8; 14],
    pub inter_view_refidx: [[i8; 16]; 2],
    /// `lossless_ipred8x8_filter_enable` (bit 0), `qpprime_y_zero_transform_bypass_flag` (1).
    pub lossless_flags: u32,
    pub display_param: DisplayParam,
    pub reserved4: [u32; 3],
}

impl Default for H264DecoderContext {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl H264DecoderContext {
    pub fn qpprime_y_zero_transform_bypass_flag(&self) -> bool {
        self.lossless_flags >> 1 & 1 != 0
    }
}

/// Frame size and plane pitches of a VP9 frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Vp9FrameDimensions {
    pub width: i16,
    pub height: i16,
    pub luma_pitch: i16,
    pub chroma_pitch: i16,
}

/// VP9 segmentation parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Vp9Segmentation {
    pub enabled: u8,
    pub update_map: u8,
    pub temporal_update: u8,
    pub abs_delta: u8,
    /// Per segment: quantizer, loop filter, reference frame, skip.
    pub feature_enabled: [[u8; 4]; 8],
    pub feature_data: [[i16; 4]; 8],
}

/// VP9 loop filter deltas.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Vp9LoopFilter {
    pub mode_ref_delta_enabled: u8,
    pub ref_deltas: [i8; 4],
    pub mode_deltas: [i8; 2],
}

/// Bits of `Vp9PictureInfo::vp9_flags`.
pub mod vp9_flags {
    pub const IS_KEY_FRAME: u32 = 1 << 0;
    pub const LAST_FRAME_IS_KEY_FRAME: u32 = 1 << 1;
    pub const FRAME_SIZE_CHANGED: u32 = 1 << 2;
    pub const ERROR_RESILIENT_MODE: u32 = 1 << 3;
    pub const LAST_SHOW_FRAME: u32 = 1 << 4;
    pub const INTRA_ONLY: u32 = 1 << 5;
}

/// The VP9 picture info structure at `PICTURE_INFO_OFFSET`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Vp9PictureInfo {
    pub reserved0: [u32; 12],
    pub bitstream_size: u32,
    pub reserved1: [u32; 5],
    pub last_frame_size: Vp9FrameDimensions,
    pub golden_frame_size: Vp9FrameDimensions,
    pub alt_frame_size: Vp9FrameDimensions,
    pub current_frame_size: Vp9FrameDimensions,
    /// See [`vp9_flags`].
    pub vp9_flags: u32,
    pub ref_frame_sign_bias: [i8; 4],
    pub first_level: u8,
    pub sharpness_level: u8,
    pub base_q_index: u8,
    pub y_dc_delta_q: u8,
    pub uv_ac_delta_q: u8,
    pub uv_dc_delta_q: u8,
    pub lossless: u8,
    pub tx_mode: u8,
    pub allow_high_precision_mv: u8,
    pub interp_filter: u8,
    pub reference_mode: u8,
    pub reserved2: [u8; 3],
    pub log2_tile_cols: u8,
    pub log2_tile_rows: u8,
    pub segmentation: Vp9Segmentation,
    pub loop_filter: Vp9LoopFilter,
    pub reserved3: [u8; 21],
}

impl Default for Vp9PictureInfo {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// VP9 probabilities in bitstream order, as the compressed header codes them.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Vp9EntropyProbs {
    pub y_mode_prob: [u8; 36],
    /// 16 contexts of 3 probabilities, each padded to 4.
    pub partition_prob: [u8; 64],
    /// 4 transform sizes x 2 planes x 2 references x 6 bands x 6 contexts x 3 probabilities.
    pub coef_probs: [u8; 1728],
    pub switchable_interp_prob: [u8; 8],
    /// 7 contexts of 3 probabilities, each padded to 4.
    pub inter_mode_prob: [u8; 28],
    pub intra_inter_prob: [u8; 4],
    pub comp_inter_prob: [u8; 5],
    pub single_ref_prob: [u8; 10],
    pub comp_ref_prob: [u8; 5],
    pub tx_32x32_prob: [u8; 6],
    pub tx_16x16_prob: [u8; 4],
    pub tx_8x8_prob: [u8; 2],
    pub skip_probs: [u8; 3],
    pub joints: [u8; 3],
    pub sign: [u8; 2],
    pub classes: [u8; 20],
    pub class_0: [u8; 2],
    pub prob_bits: [u8; 20],
    pub class_0_fr: [u8; 12],
    pub fr: [u8; 6],
    pub class_0_hp: [u8; 2],
    pub high_precision: [u8; 2],
}

impl Default for Vp9EntropyProbs {
    fn default() -> Self {
        Self::zeroed()
    }
}

/// VP9 probabilities as the guest's driver lays them out at `VP9_PROB_TAB_BUFFER_OFFSET`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Vp9NvdecEntropyProbs {
    pub kf_bmode_prob: [u8; 800],
    pub kf_bmode_prob_b: [u8; 100],
    pub ref_pred_probs: [u8; 3],
    pub mb_segment_tree_probs: [u8; 7],
    pub segment_pred_probs: [u8; 3],
    pub ref_scores: [u8; 4],
    pub prob_comppred: [u8; 2],
    pub reserved0: [u8; 9],
    pub kf_uv_mode_prob: [u8; 80],
    pub kf_uv_mode_prob_b: [u8; 10],
    pub reserved1: [u8; 6],
    pub inter_mode_prob: [u8; 28],
    pub intra_inter_prob: [u8; 4],
    pub reserved2: [u8; 80],
    pub tx_8x8_prob: [u8; 2],
    pub tx_16x16_prob: [u8; 4],
    pub tx_32x32_prob: [u8; 6],
    pub y_mode_prob_e8: [u8; 4],
    pub y_mode_prob_e0e7: [[u8; 8]; 4],
    pub reserved3: [u8; 64],
    pub partition_prob: [u8; 64],
    pub reserved4: [u8; 10],
    pub switchable_interp_prob: [u8; 8],
    pub comp_inter_prob: [u8; 5],
    pub skip_probs: [u8; 3],
    pub reserved5: [u8; 1],
    pub joints: [u8; 3],
    pub sign: [u8; 2],
    pub class_0: [u8; 2],
    pub fr: [u8; 6],
    pub class_0_hp: [u8; 2],
    pub high_precision: [u8; 2],
    pub classes: [u8; 20],
    pub class_0_fr: [u8; 12],
    pub pred_bits: [u8; 20],
    pub single_ref_prob: [u8; 10],
    pub comp_ref_prob: [u8; 5],
    pub reserved6: [u8; 17],
    /// Coefficient probabilities padded to 4 per context; the 4th is unused.
    pub coef_probs: [u8; 2304],
}

impl Default for Vp9NvdecEntropyProbs {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Vp9NvdecEntropyProbs {
    /// Repacks the driver's layout into bitstream order.
    pub fn to_bitstream_order(&self) -> Vp9EntropyProbs {
        let mut probs = Vp9EntropyProbs {
            inter_mode_prob: self.inter_mode_prob,
            intra_inter_prob: self.intra_inter_prob,
            tx_8x8_prob: self.tx_8x8_prob,
            tx_16x16_prob: self.tx_16x16_prob,
            tx_32x32_prob: self.tx_32x32_prob,
            partition_prob: self.partition_prob,
            switchable_interp_prob: self.switchable_interp_prob,
            comp_inter_prob: self.comp_inter_prob,
            skip_probs: self.skip_probs,
            joints: self.joints,
            sign: self.sign,
            class_0: self.class_0,
            fr: self.fr,
            class_0_hp: self.class_0_hp,
            high_precision: self.high_precision,
            classes: self.classes,
            class_0_fr: self.class_0_fr,
            prob_bits: self.pred_bits,
            single_ref_prob: self.single_ref_prob,
            comp_ref_prob: self.comp_ref_prob,
            ..Vp9EntropyProbs::zeroed()
        };
        for (i, row) in probs
            .y_mode_prob
            .as_chunks_mut::<9>()
            .0
            .iter_mut()
            .enumerate()
        {
            row[..8].copy_from_slice(&self.y_mode_prob_e0e7[i]);
            row[8] = self.y_mode_prob_e8[i];
        }
        for (dst, src) in probs
            .coef_probs
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
            .zip(self.coef_probs.as_chunks::<4>().0)
        {
            dst.copy_from_slice(&src[..3]);
        }
        probs
    }
}

/// The VP8 picture info structure at `PICTURE_INFO_OFFSET`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct Vp8PictureInfo {
    pub reserved0: [u32; 14],
    pub frame_width: u16,
    pub frame_height: u16,
    pub key_frame: u8,
    pub version: u8,
    /// `tile_format` (bits 0..2), `gob_height` (2..5), reserved surface format bits.
    pub surface_format: u8,
    pub error_conceal_on: u8,
    /// Size of the first partition (frame header and macroblock headers).
    pub first_part_size: u32,
    pub hist_buffer_size: u32,
    /// Bytes of frame data at `FRAME_BITSTREAM_OFFSET`.
    pub vld_buffer_size: u32,
    pub frame_stride: [u32; 2],
    pub luma_top_offset: u32,
    pub luma_bot_offset: u32,
    pub luma_frame_offset: u32,
    pub chroma_top_offset: u32,
    pub chroma_bot_offset: u32,
    pub chroma_frame_offset: u32,
    pub display_param: [u8; 0x1C],
    pub current_output_memory_layout: i8,
    /// Golden, alt-ref, last.
    pub output_memory_layout: [i8; 3],
    pub segmentation_feature_data_update: u8,
    pub reserved1: [u8; 3],
    pub result_value: u32,
    pub partition_offset: [u32; 8],
    pub reserved2: [u32; 3],
}

impl Default for Vp8PictureInfo {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    #[test]
    fn layouts_match_the_engine() {
        assert_eq!(size_of::<H264ParameterSet>(), 0x60);
        assert_eq!(offset_of!(H264ParameterSet, luma_top_offset), 0x3C);
        assert_eq!(offset_of!(H264ParameterSet, hist_buffer_size), 0x54);
        assert_eq!(offset_of!(H264ParameterSet, flags), 0x58);
        assert_eq!(size_of::<H264DpbEntry>(), 0x10);
        assert_eq!(size_of::<DisplayParam>(), 0x1C);
        assert_eq!(size_of::<H264DecoderContext>(), 0x2FC);
        assert_eq!(offset_of!(H264DecoderContext, stream_len), 0x48);
        assert_eq!(offset_of!(H264DecoderContext, parameter_set), 0x58);
        assert_eq!(offset_of!(H264DecoderContext, dpb), 0xC0);
        assert_eq!(offset_of!(H264DecoderContext, weight_scale_4x4), 0x1C0);
        assert_eq!(offset_of!(H264DecoderContext, weight_scale_8x8), 0x220);
        assert_eq!(offset_of!(H264DecoderContext, lossless_flags), 0x2D0);
        assert_eq!(offset_of!(H264DecoderContext, display_param), 0x2D4);

        assert_eq!(size_of::<Vp9FrameDimensions>(), 0x8);
        assert_eq!(size_of::<Vp9Segmentation>(), 0x64);
        assert_eq!(size_of::<Vp9LoopFilter>(), 0x7);
        assert_eq!(size_of::<Vp9EntropyProbs>(), 0x7B4);
        assert_eq!(offset_of!(Vp9EntropyProbs, partition_prob), 0x24);
        assert_eq!(offset_of!(Vp9EntropyProbs, switchable_interp_prob), 0x724);
        assert_eq!(offset_of!(Vp9EntropyProbs, sign), 0x772);
        assert_eq!(offset_of!(Vp9EntropyProbs, class_0_fr), 0x79E);
        assert_eq!(offset_of!(Vp9EntropyProbs, high_precision), 0x7B2);
        assert_eq!(size_of::<Vp9PictureInfo>(), 0x100);
        assert_eq!(offset_of!(Vp9PictureInfo, bitstream_size), 0x30);
        assert_eq!(offset_of!(Vp9PictureInfo, last_frame_size), 0x48);
        assert_eq!(offset_of!(Vp9PictureInfo, vp9_flags), 0x68);
        assert_eq!(offset_of!(Vp9PictureInfo, first_level), 0x70);
        assert_eq!(offset_of!(Vp9PictureInfo, log2_tile_cols), 0x7E);
        assert_eq!(offset_of!(Vp9PictureInfo, segmentation), 0x80);
        assert_eq!(offset_of!(Vp9PictureInfo, loop_filter), 0xE4);
        assert_eq!(size_of::<Vp9NvdecEntropyProbs>(), 0xEA0);
        assert_eq!(
            offset_of!(Vp9NvdecEntropyProbs, mb_segment_tree_probs),
            0x387
        );
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, segment_pred_probs), 0x38E);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, inter_mode_prob), 0x400);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, tx_8x8_prob), 0x470);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, y_mode_prob_e0e7), 0x480);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, partition_prob), 0x4E0);
        assert_eq!(
            offset_of!(Vp9NvdecEntropyProbs, switchable_interp_prob),
            0x52A
        );
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, joints), 0x53B);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, class_0), 0x540);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, class_0_fr), 0x560);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, single_ref_prob), 0x580);
        assert_eq!(offset_of!(Vp9NvdecEntropyProbs, coef_probs), 0x5A0);

        assert_eq!(size_of::<Vp8PictureInfo>(), 0xC0);
        assert_eq!(offset_of!(Vp8PictureInfo, frame_width), 0x38);
        assert_eq!(offset_of!(Vp8PictureInfo, first_part_size), 0x40);
        assert_eq!(offset_of!(Vp8PictureInfo, vld_buffer_size), 0x48);
        assert_eq!(offset_of!(Vp8PictureInfo, display_param), 0x6C);
        assert_eq!(offset_of!(Vp8PictureInfo, result_value), 0x90);
        assert_eq!(offset_of!(Vp8PictureInfo, partition_offset), 0x94);
    }

    #[test]
    fn h264_flag_fields_sign_extend() {
        let mut params = H264ParameterSet::default();
        let flags = 0b1010_0101
            | 0x9 << 8
            | 0x3 << 12
            | 0x2 << 14
            | (0x3D_u64) << 16
            | (0x1E_u64) << 22
            | (0x05_u64) << 27
            | 0x2 << 32
            | 0x45 << 34
            | 0x11 << 41
            | 0xBEEF << 46
            | 1 << 63;
        params.set_flags(flags);
        assert_eq!(params.flags(), flags);
        assert!(params.mbaff_frame());
        assert!(!params.direct_8x8_inference());
        assert!(params.weighted_pred());
        assert!(params.field_pic());
        assert!(params.second_field());
        assert_eq!(params.log2_max_frame_num_minus4(), 9);
        assert_eq!(params.chroma_format_idc(), 3);
        assert_eq!(params.pic_order_cnt_type(), 2);
        assert_eq!(params.pic_init_qp_minus26(), -3);
        assert_eq!(params.chroma_qp_index_offset(), -2);
        assert_eq!(params.second_chroma_qp_index_offset(), 5);
        assert_eq!(params.weighted_bipred_idc(), 2);
        assert_eq!(params.curr_pic_idx(), 0x45);
        assert_eq!(params.curr_col_idx(), 0x11);
        assert_eq!(params.frame_number(), 0xBEEF);
        assert!(!params.frame_surfaces());
        assert!(params.output_memory_layout());
    }

    #[test]
    fn nvdec_entropy_repacks_y_mode_and_drops_the_fourth_coef_probability() {
        let mut nvdec = Vp9NvdecEntropyProbs::zeroed();
        for (i, row) in nvdec.y_mode_prob_e0e7.iter_mut().enumerate() {
            for (j, p) in row.iter_mut().enumerate() {
                *p = (i * 10 + j) as u8;
            }
            nvdec.y_mode_prob_e8[i] = 200 + i as u8;
        }
        for (i, p) in nvdec.coef_probs.iter_mut().enumerate() {
            *p = if i % 4 == 3 { 0xEE } else { (i % 200) as u8 };
        }
        nvdec.pred_bits[19] = 77;
        let probs = nvdec.to_bitstream_order();
        assert_eq!(&probs.y_mode_prob[..9], &[0, 1, 2, 3, 4, 5, 6, 7, 200]);
        assert_eq!(
            &probs.y_mode_prob[27..],
            &[30, 31, 32, 33, 34, 35, 36, 37, 203]
        );
        assert!(!probs.coef_probs.contains(&0xEE));
        assert_eq!(&probs.coef_probs[..6], &[0, 1, 2, 4, 5, 6]);
        assert_eq!(probs.coef_probs[1727], (2302 % 200) as u8);
        assert_eq!(probs.prob_bits[19], 77);
    }
}
