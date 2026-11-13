use bytemuck::Zeroable;

use super::bits::{BitWriter, RangeEncoder};
use super::picture::{
    Vp9EntropyProbs, Vp9FrameDimensions, Vp9NvdecEntropyProbs, Vp9PictureInfo, Vp9Segmentation,
    vp9_flags,
};
use super::vp9_tables::DEFAULT_PROBS;
use super::{ComposedFrame, FrameDimensions, NvdecMemory, NvdecRegisters, SurfaceOffsets};

const CURRENT_SURFACE: usize = 3;
const FRAME_SYNC_CODE: u64 = 0x49_8342;
const DIFF_UPDATE_PROBABILITY: u32 = 252;
const SWITCHABLE_FILTER: u8 = 4;
const TX_MODE_SELECT: u8 = 4;
const DEFAULT_REF_DELTAS: [i8; 4] = [1, 0, -1, -1];

/// One frame's parameters, converted from the guest's picture info and probability tables.
#[derive(Clone, Copy, Zeroable)]
struct FrameInfo {
    surface_offsets: [u64; 4],
    ref_frame_sign_bias: [i8; 4],
    base_q_index: u8,
    y_dc_delta_q: i8,
    uv_dc_delta_q: i8,
    uv_ac_delta_q: i8,
    tx_mode: u8,
    interp_filter: u8,
    reference_mode: u8,
    log2_tile_cols: u8,
    log2_tile_rows: u8,
    ref_deltas: [i8; 4],
    mode_deltas: [i8; 2],
    frame_size: Vp9FrameDimensions,
    ref_frame_sizes: [Vp9FrameDimensions; 3],
    first_level: u8,
    sharpness_level: u8,
    is_key_frame: bool,
    intra_only: bool,
    error_resilient_mode: bool,
    last_frame_shown: bool,
    show_frame: bool,
    lossless: bool,
    allow_high_precision_mv: bool,
    mode_ref_delta_enabled: bool,
    segmentation: Vp9Segmentation,
    seg_tree_probs: [u8; 7],
    seg_pred_probs: [u8; 3],
    entropy: Vp9EntropyProbs,
}

impl FrameInfo {
    fn new(picture: &Vp9PictureInfo, probs: &Vp9NvdecEntropyProbs, regs: &NvdecRegisters) -> Self {
        let flag = |bit| picture.vp9_flags & bit != 0;
        Self {
            surface_offsets: core::array::from_fn(|i| regs.surface_luma_address(i)),
            ref_frame_sign_bias: picture.ref_frame_sign_bias,
            base_q_index: picture.base_q_index,
            y_dc_delta_q: picture.y_dc_delta_q as i8,
            uv_dc_delta_q: picture.uv_dc_delta_q as i8,
            uv_ac_delta_q: picture.uv_ac_delta_q as i8,
            tx_mode: picture.tx_mode,
            interp_filter: picture.interp_filter,
            reference_mode: picture.reference_mode,
            log2_tile_cols: picture.log2_tile_cols,
            log2_tile_rows: picture.log2_tile_rows,
            ref_deltas: picture.loop_filter.ref_deltas,
            mode_deltas: picture.loop_filter.mode_deltas,
            frame_size: picture.current_frame_size,
            ref_frame_sizes: [
                picture.last_frame_size,
                picture.golden_frame_size,
                picture.alt_frame_size,
            ],
            first_level: picture.first_level,
            sharpness_level: picture.sharpness_level,
            is_key_frame: flag(vp9_flags::IS_KEY_FRAME),
            intra_only: flag(vp9_flags::INTRA_ONLY),
            error_resilient_mode: flag(vp9_flags::ERROR_RESILIENT_MODE),
            last_frame_shown: flag(vp9_flags::LAST_SHOW_FRAME),
            show_frame: true,
            lossless: picture.lossless != 0,
            allow_high_precision_mv: picture.allow_high_precision_mv != 0,
            mode_ref_delta_enabled: picture.loop_filter.mode_ref_delta_enabled != 0,
            segmentation: picture.segmentation,
            seg_tree_probs: probs.mb_segment_tree_probs,
            seg_pred_probs: probs.segment_pred_probs,
            entropy: probs.to_bitstream_order(),
        }
    }
}

/// Composes VP9 frames: an uncompressed header and a compressed (probability update) header
/// rebuilt from the guest's parameters, followed by the guest's frame data.
///
/// The guest's driver only reveals that a frame was hidden (`show_frame == 0`) and which
/// reference slots it refreshes with the frame after it, so every frame is held back by one
/// submission: `build` composes the previous submission and buffers the current one. The very
/// first submission is composed straight away and then again on the second call.
pub struct Vp9Builder {
    picture: Vp9PictureInfo,
    nvdec_probs: Vp9NvdecEntropyProbs,
    current: FrameInfo,
    next: FrameInfo,
    stream: Vec<u8>,
    next_stream: Vec<u8>,
    compressed: Vec<u8>,
    frame: Vec<u8>,
    frame_ctxs: [Vp9EntropyProbs; 4],
    loop_filter_ref_deltas: [i8; 4],
    loop_filter_mode_deltas: [i8; 2],
    last_segmentation: Vp9Segmentation,
    swap_ref_indices: bool,
}

impl Default for Vp9Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Vp9Builder {
    /// A builder for a new stream, with no frame buffered.
    pub fn new() -> Self {
        Self {
            picture: Vp9PictureInfo::zeroed(),
            nvdec_probs: Vp9NvdecEntropyProbs::zeroed(),
            current: FrameInfo::zeroed(),
            next: FrameInfo::zeroed(),
            stream: Vec::new(),
            next_stream: Vec::new(),
            compressed: Vec::new(),
            frame: Vec::new(),
            frame_ctxs: [Vp9EntropyProbs::zeroed(); 4],
            loop_filter_ref_deltas: [0; 4],
            loop_filter_mode_deltas: [0; 2],
            last_segmentation: Vp9Segmentation::zeroed(),
            swap_ref_indices: false,
        }
    }

    /// The picture info of the last submission (not necessarily the frame last composed).
    pub fn picture_info(&self) -> &Vp9PictureInfo {
        &self.picture
    }

    /// Buffers the submission the registers describe and composes the one before it.
    ///
    /// The returned offsets are the current registers' output surface, as the guest expects a
    /// picture there once this submission completes.
    pub fn build(
        &mut self,
        regs: &NvdecRegisters,
        memory: &(impl NvdecMemory + ?Sized),
    ) -> ComposedFrame<'_> {
        memory.read(
            regs.picture_info_address(),
            bytemuck::bytes_of_mut(&mut self.picture),
        );
        memory.read(
            regs.vp9_prob_tab_buffer_address(),
            bytemuck::bytes_of_mut(&mut self.nvdec_probs),
        );
        let submitted = FrameInfo::new(&self.picture, &self.nvdec_probs, regs);
        self.stream.resize(self.picture.bitstream_size as usize, 0);
        memory.read(regs.frame_bitstream_address(), &mut self.stream);

        if self.next_stream.is_empty() {
            self.current = submitted;
            self.next = submitted;
            self.next_stream.clear();
            self.next_stream.extend_from_slice(&self.stream);
        } else {
            self.current = self.next;
            self.current.show_frame = submitted.last_frame_shown;
            self.next = submitted;
            core::mem::swap(&mut self.stream, &mut self.next_stream);
        }

        let mut frame = core::mem::take(&mut self.frame);
        frame.clear();
        let mut header = BitWriter::new(&mut frame);
        let ctx = self.write_uncompressed_header(&mut header);

        let mut compressed = core::mem::take(&mut self.compressed);
        compressed.clear();
        write_compressed_header(&self.current, &self.frame_ctxs[ctx], &mut compressed);
        self.frame_ctxs[ctx] = self.current.entropy;

        header.bits(compressed.len() as u64, 16);
        header.flush();
        frame.extend_from_slice(&compressed);
        frame.extend_from_slice(&self.stream);
        self.compressed = compressed;
        self.frame = frame;

        ComposedFrame {
            data: &self.frame,
            dimensions: FrameDimensions::new(
                i64::from(self.current.frame_size.width),
                i64::from(self.current.frame_size.height),
            ),
            hidden: !self.current.show_frame,
            offsets: SurfaceOffsets::Progressive {
                luma: regs.surface_luma_address(CURRENT_SURFACE),
                chroma: regs.surface_chroma_address(CURRENT_SURFACE),
            },
        }
    }

    /// Writes the uncompressed header up to (excluding) the compressed header size and returns
    /// the frame context the compressed header codes against.
    fn write_uncompressed_header(&mut self, w: &mut BitWriter) -> usize {
        let info = &mut self.current;
        w.bits(2, 2);
        w.bits(0, 2);
        w.bit(false);
        w.bit(!info.is_key_frame);
        w.bit(info.show_frame);
        w.bit(info.error_resilient_mode);

        if info.is_key_frame {
            w.bits(FRAME_SYNC_CODE, 24);
            w.bits(0, 3);
            w.bits(0, 1);
            write_frame_size(w, info.frame_size);
            w.bit(false);
            self.swap_ref_indices = false;
            self.frame_ctxs = [DEFAULT_PROBS; 4];
            info.intra_only = true;
        } else {
            if info.show_frame {
                info.intra_only = false;
            } else {
                w.bit(info.intra_only);
            }
            if !info.error_resilient_mode {
                w.bits(0, 2);
            }
            let curr = info.surface_offsets;
            let next = self.next.surface_offsets;
            let needs_ref_swap = curr[1] != curr[2] && (next[1] == curr[2] || next[2] == curr[1]);
            self.swap_ref_indices ^= needs_ref_swap;
            let mut refresh = (0..3)
                .filter(|&i| curr[3] == next[i])
                .fold(0u64, |flags, i| flags | 1 << i);
            if self.swap_ref_indices {
                refresh = refresh & 1 | (refresh >> 2 & 1) << 1 | (refresh >> 1 & 1) << 2;
            }
            if info.intra_only {
                w.bits(FRAME_SYNC_CODE, 24);
                w.bits(refresh, 8);
                write_frame_size(w, info.frame_size);
                w.bit(false);
            } else {
                let ref_frame_index = if needs_ref_swap ^ self.swap_ref_indices {
                    [0, 2, 1]
                } else {
                    [0, 1, 2]
                };
                w.bits(refresh, 8);
                for (slot, sign_bias) in ref_frame_index.iter().zip(&info.ref_frame_sign_bias[1..])
                {
                    w.bits(*slot, 3);
                    w.bits(*sign_bias as u64, 1);
                }
                let size = info.frame_size;
                match info
                    .ref_frame_sizes
                    .iter()
                    .position(|r| r.width == size.width && r.height == size.height)
                {
                    Some(found) => (0..=found).for_each(|i| w.bit(i == found)),
                    None => {
                        w.bits(0, 3);
                        write_frame_size(w, info.frame_size);
                    }
                }
                w.bit(false);
                w.bit(info.allow_high_precision_mv);
                w.bit(info.interp_filter == SWITCHABLE_FILTER);
                if info.interp_filter != SWITCHABLE_FILTER {
                    w.bits(u64::from(info.interp_filter), 2);
                }
            }
        }

        if info.is_key_frame || info.intra_only || info.error_resilient_mode {
            self.loop_filter_ref_deltas = DEFAULT_REF_DELTAS;
            self.loop_filter_mode_deltas = [0; 2];
            self.last_segmentation = Vp9Segmentation::zeroed();
        }

        if !info.error_resilient_mode {
            w.bit(true);
            w.bit(true);
        }
        let ctx = usize::from(!info.show_frame);
        w.bits(ctx as u64, 2);

        w.bits(u64::from(info.first_level), 6);
        w.bits(u64::from(info.sharpness_level), 3);
        w.bit(info.mode_ref_delta_enabled);
        if info.mode_ref_delta_enabled {
            let update = info.ref_deltas != self.loop_filter_ref_deltas
                || info.mode_deltas != self.loop_filter_mode_deltas;
            w.bit(update);
            if update {
                for (new, old) in info.ref_deltas.iter().zip(&self.loop_filter_ref_deltas) {
                    w.bit(new != old);
                    if new != old {
                        w.signed(i32::from(*new), 6);
                    }
                }
                for (new, old) in info.mode_deltas.iter().zip(&self.loop_filter_mode_deltas) {
                    w.bit(new != old);
                    if new != old {
                        w.signed(i32::from(*new), 6);
                    }
                }
                self.loop_filter_ref_deltas = info.ref_deltas;
                self.loop_filter_mode_deltas = info.mode_deltas;
            }
        }

        w.bits(u64::from(info.base_q_index), 8);
        for delta in [info.y_dc_delta_q, info.uv_dc_delta_q, info.uv_ac_delta_q] {
            w.bit(delta != 0);
            if delta != 0 {
                w.signed(i32::from(delta), 4);
            }
        }

        let segmentation = &info.segmentation;
        w.bit(segmentation.enabled != 0);
        if segmentation.enabled != 0 {
            w.bit(segmentation.update_map != 0);
            if segmentation.update_map != 0 {
                let write_prob = |w: &mut BitWriter, prob: u8| {
                    w.bit(prob != 255);
                    if prob != 255 {
                        w.bits(u64::from(prob), 8);
                    }
                };
                for &prob in &info.seg_tree_probs {
                    write_prob(w, prob);
                }
                w.bit(segmentation.temporal_update != 0);
                if segmentation.temporal_update != 0 {
                    for &prob in &info.seg_pred_probs {
                        write_prob(w, prob);
                    }
                }
            }
            let update_data = *segmentation != self.last_segmentation;
            w.bit(update_data);
            if update_data {
                self.last_segmentation = *segmentation;
                w.bit(segmentation.abs_delta != 0);
                for (enabled, data) in segmentation
                    .feature_enabled
                    .iter()
                    .zip(&segmentation.feature_data)
                {
                    for (feature, bits) in [8, 6, 2, 0].into_iter().enumerate() {
                        w.bit(enabled[feature] != 0);
                        if enabled[feature] == 0 {
                            continue;
                        }
                        match feature {
                            0 | 1 => w.signed(i32::from(data[feature]), bits),
                            _ => w.bits(data[feature] as u64, bits),
                        }
                    }
                }
            }
        }

        let width = i32::from(info.frame_size.width);
        let sb64_cols = (width + 63) / 64;
        let min_log2 = (0..).find(|&log2| 64 << log2 >= sb64_cols).unwrap_or(0);
        let max_log2 = (1..).find(|&log2| sb64_cols >> log2 < 4).unwrap_or(1) - 1;
        let log2_tile_cols = i32::from(info.log2_tile_cols);
        let increments = (log2_tile_cols - min_log2).max(0) as u32;
        let ones = (1u64 << increments) - 1;
        if log2_tile_cols < max_log2 {
            w.bits(ones << 1, increments + 1);
        } else {
            w.bits(ones, increments);
        }
        w.bit(info.log2_tile_rows != 0);
        if info.log2_tile_rows != 0 {
            w.bit(info.log2_tile_rows > 1);
        }
        ctx
    }
}

fn write_frame_size(w: &mut BitWriter, size: Vp9FrameDimensions) {
    w.bits((i32::from(size.width) - 1) as u64, 16);
    w.bits((i32::from(size.height) - 1) as u64, 16);
}

fn write_compressed_header(info: &FrameInfo, old: &Vp9EntropyProbs, out: &mut Vec<u8>) {
    let new = &info.entropy;
    let mut e = RangeEncoder::new(out);
    if !info.lossless {
        if info.tx_mode >= 3 {
            e.literal(3, 2);
            e.bit(info.tx_mode == TX_MODE_SELECT);
        } else {
            e.literal(u32::from(info.tx_mode), 2);
        }
    }
    if info.tx_mode == TX_MODE_SELECT {
        update_probs(&mut e, &new.tx_8x8_prob, &old.tx_8x8_prob);
        update_probs(&mut e, &new.tx_16x16_prob, &old.tx_16x16_prob);
        update_probs(&mut e, &new.tx_32x32_prob, &old.tx_32x32_prob);
    }

    const BLOCK: usize = 2 * 2 * 6 * 6 * 3;
    for (tx_size, (new_block, old_block)) in new
        .coef_probs
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .zip(old.coef_probs.as_chunks::<BLOCK>().0)
        .enumerate()
    {
        let update = new_block != old_block;
        e.bit(update);
        if update {
            for (context, (new_ctx, old_ctx)) in new_block
                .as_chunks::<3>()
                .0
                .iter()
                .zip(old_block.as_chunks::<3>().0)
                .enumerate()
            {
                let band = context / 6 % 6;
                if band != 0 || context % 6 < 3 {
                    update_probs(&mut e, new_ctx, old_ctx);
                }
            }
        }
        if tx_size == usize::from(info.tx_mode) {
            break;
        }
    }
    update_probs(&mut e, &new.skip_probs, &old.skip_probs);

    if !info.intra_only {
        update_probs_aligned4(&mut e, &new.inter_mode_prob, &old.inter_mode_prob);
        if info.interp_filter == SWITCHABLE_FILTER {
            update_probs(
                &mut e,
                &new.switchable_interp_prob,
                &old.switchable_interp_prob,
            );
        }
        update_probs(&mut e, &new.intra_inter_prob, &old.intra_inter_prob);

        let sign_bias = info.ref_frame_sign_bias.map(|bias| bias & 1);
        if sign_bias[1] != sign_bias[2] || sign_bias[1] != sign_bias[3] {
            e.bit(info.reference_mode >= 1);
            if info.reference_mode >= 1 {
                e.bit(info.reference_mode == 2);
            }
        }
        if info.reference_mode == 2 {
            update_probs(&mut e, &new.comp_inter_prob, &old.comp_inter_prob);
        }
        if info.reference_mode != 1 {
            update_probs(&mut e, &new.single_ref_prob, &old.single_ref_prob);
        }
        if info.reference_mode != 0 {
            update_probs(&mut e, &new.comp_ref_prob, &old.comp_ref_prob);
        }
        update_probs(&mut e, &new.y_mode_prob, &old.y_mode_prob);
        update_probs_aligned4(&mut e, &new.partition_prob, &old.partition_prob);

        update_mv_probs(&mut e, &new.joints, &old.joints);
        for i in 0..2 {
            let ten = i * 10..i * 10 + 10;
            update_mv_probs(&mut e, &new.sign[i..=i], &old.sign[i..=i]);
            update_mv_probs(&mut e, &new.classes[ten.clone()], &old.classes[ten.clone()]);
            update_mv_probs(&mut e, &new.class_0[i..=i], &old.class_0[i..=i]);
            update_mv_probs(&mut e, &new.prob_bits[ten.clone()], &old.prob_bits[ten]);
        }
        for i in 0..2 {
            let six = i * 6..i * 6 + 6;
            let three = i * 3..i * 3 + 3;
            update_mv_probs(&mut e, &new.class_0_fr[six.clone()], &old.class_0_fr[six]);
            update_mv_probs(&mut e, &new.fr[three.clone()], &old.fr[three]);
        }
        if info.allow_high_precision_mv {
            for i in 0..2 {
                update_mv_probs(&mut e, &new.class_0_hp[i..=i], &old.class_0_hp[i..=i]);
                update_mv_probs(
                    &mut e,
                    &new.high_precision[i..=i],
                    &old.high_precision[i..=i],
                );
            }
        }
    }
    e.finish();
}

fn update_probs(e: &mut RangeEncoder, new: &[u8], old: &[u8]) {
    for (&new, &old) in new.iter().zip(old) {
        e.bool(new != old, DIFF_UPDATE_PROBABILITY);
        if new != old {
            encode_term_subexp(e, remap_probability(new, old));
        }
    }
}

fn update_probs_aligned4(e: &mut RangeEncoder, new: &[u8], old: &[u8]) {
    for (new, old) in new.as_chunks::<4>().0.iter().zip(old.as_chunks::<4>().0) {
        update_probs(e, &new[..3], &old[..3]);
    }
}

fn update_mv_probs(e: &mut RangeEncoder, new: &[u8], old: &[u8]) {
    for (&new, &old) in new.iter().zip(old) {
        e.bool(new != old, DIFF_UPDATE_PROBABILITY);
        if new != old {
            e.literal(u32::from(new >> 1), 7);
        }
    }
}

fn recenter_nonneg(new: i32, old: i32) -> i32 {
    if new > old * 2 {
        new
    } else if new >= old {
        (new - old) * 2
    } else {
        (old - new) * 2 - 1
    }
}

/// The inverse of the VP9 `inv_remap_prob` (spec 6.3.5).
fn remap_probability(new: u8, old: u8) -> u32 {
    let new = i32::from(new) - 1;
    let old = i32::from(old) - 1;
    let recentered = if old * 2 <= 0xFF {
        recenter_nonneg(new, old)
    } else {
        recenter_nonneg(0xFF - 1 - new, 0xFF - 1 - old)
    };
    let i = u32::from((recentered - 1).max(0) as u8);
    if (i + 7) % 13 == 0 {
        (i + 7) / 13 - 1
    } else {
        i + 20 - (i + 7) / 13
    }
}

/// The inverse of the VP9 `decode_term_subexp` (spec 6.3.4).
fn encode_term_subexp(e: &mut RangeEncoder, value: u32) {
    for (limit, base, bits) in [(16, 0, 4), (32, 16, 4), (64, 32, 5)] {
        e.bit(value >= limit);
        if value < limit {
            e.literal(value - base, bits);
            return;
        }
    }
    let value = value - 64;
    const SPLIT: u32 = 65;
    if value < SPLIT {
        e.literal(value, 7);
    } else {
        let delta = value - SPLIT;
        e.literal(delta / 2 + SPLIT, 7);
        e.literal(delta & 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvdec::bits::tests::RangeDecoder;

    const INV_MAP_TABLE: [u8; 255] = [
        7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254,
        1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27,
        28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52,
        53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77,
        78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101,
        102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120,
        121, 122, 123, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140,
        141, 142, 143, 144, 145, 146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159,
        160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179,
        180, 181, 182, 183, 184, 185, 186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198,
        199, 200, 201, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 216, 217, 218,
        219, 220, 221, 222, 223, 224, 225, 226, 227, 229, 230, 231, 232, 233, 234, 235, 236, 237,
        238, 239, 240, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 253,
    ];

    fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
        if v > 2 * m {
            v
        } else if v & 1 != 0 {
            m - ((v + 1) >> 1)
        } else {
            m + (v >> 1)
        }
    }

    fn decode_probability_delta(d: &mut RangeDecoder, old: u8) -> u8 {
        let index = if !d.bool(128) {
            d.literal(4)
        } else if !d.bool(128) {
            d.literal(4) + 16
        } else if !d.bool(128) {
            d.literal(5) + 32
        } else {
            let mut index = d.literal(7);
            if index >= 65 {
                index = (index << 1) - 65 + u32::from(d.bool(128));
            }
            index + 64
        };
        let v = i32::from(INV_MAP_TABLE[index as usize]);
        let p = i32::from(old);
        let new = if p <= 128 {
            1 + inv_recenter_nonneg(v, p - 1)
        } else {
            255 - inv_recenter_nonneg(v, 255 - p)
        };
        new as u8
    }

    #[test]
    fn probability_deltas_decode_to_the_new_probability_for_every_pair() {
        for old in 1..=255u8 {
            let mut out = Vec::new();
            let mut encoder = RangeEncoder::new(&mut out);
            for new in 1..=255u8 {
                update_probs(&mut encoder, &[new], &[old]);
            }
            encoder.finish();
            let mut decoder = RangeDecoder::new(&out);
            for new in 1..=255u8 {
                let updated = decoder.bool(DIFF_UPDATE_PROBABILITY);
                assert_eq!(updated, new != old, "old {old} new {new}");
                if updated {
                    assert_eq!(
                        decode_probability_delta(&mut decoder, old),
                        new,
                        "old {old}"
                    );
                }
            }
        }
    }

    #[test]
    fn mv_probability_updates_keep_the_top_seven_bits() {
        let mut out = Vec::new();
        let mut encoder = RangeEncoder::new(&mut out);
        update_mv_probs(&mut encoder, &[200, 9, 128], &[200, 10, 1]);
        encoder.finish();
        let mut decoder = RangeDecoder::new(&out);
        assert!(!decoder.bool(DIFF_UPDATE_PROBABILITY));
        assert!(decoder.bool(DIFF_UPDATE_PROBABILITY));
        assert_eq!(decoder.literal(7) << 1 | 1, 9);
        assert!(decoder.bool(DIFF_UPDATE_PROBABILITY));
        assert_eq!(decoder.literal(7) << 1 | 1, 129);
    }
}
