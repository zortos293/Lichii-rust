use super::bits::BitWriter;
use super::picture::H264DecoderContext;
use super::{ComposedFrame, FrameDimensions, NvdecMemory, NvdecRegisters, SurfaceOffsets};

const ZIGZAG_4X4: [u8; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

const ZIGZAG_8X8: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Composes H.264 Annex B access units: the guest's slice data, preceded by an SPS and PPS built
/// from the picture parameters on the first frame and on every frame numbered 0.
pub struct H264Builder {
    context: H264DecoderContext,
    header: Vec<u8>,
    frame: Vec<u8>,
    is_first_frame: bool,
}

impl Default for H264Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl H264Builder {
    /// A builder for a new stream: the first frame carries an SPS and PPS.
    pub fn new() -> Self {
        Self {
            context: H264DecoderContext::default(),
            header: Vec::new(),
            frame: Vec::new(),
            is_first_frame: true,
        }
    }

    /// The picture parameters of the last built frame.
    pub fn context(&self) -> &H264DecoderContext {
        &self.context
    }

    /// Composes the access unit the registers describe.
    pub fn build(
        &mut self,
        regs: &NvdecRegisters,
        memory: &(impl NvdecMemory + ?Sized),
    ) -> ComposedFrame<'_> {
        memory.read(
            regs.picture_info_address(),
            bytemuck::bytes_of_mut(&mut self.context),
        );
        let params = &self.context.parameter_set;

        self.header.clear();
        if self.is_first_frame || params.frame_number() == 0 {
            self.is_first_frame = false;
            write_parameter_sets(&self.context, &mut self.header);
        }

        let header_len = self.header.len();
        let stream_len = self.context.stream_len as usize;
        self.frame.resize(header_len + stream_len, 0);
        self.frame[..header_len].copy_from_slice(&self.header);
        memory.read(
            regs.frame_bitstream_address(),
            &mut self.frame[header_len..],
        );

        let surface = params.curr_pic_idx();
        let luma = regs.surface_luma_address(surface);
        let chroma = regs.surface_chroma_address(surface);
        let interlaced = params.luma_top_offset != 0 || params.luma_bot_offset != 0;
        let offsets = if interlaced {
            SurfaceOffsets::Interlaced {
                luma_top: luma + (u64::from(params.luma_top_offset) << 8),
                luma_bottom: luma + (u64::from(params.luma_bot_offset) << 8),
                chroma_top: chroma + (u64::from(params.chroma_top_offset) << 8),
                chroma_bottom: chroma + (u64::from(params.chroma_bot_offset) << 8),
            }
        } else {
            SurfaceOffsets::Progressive {
                luma: luma + (u64::from(params.luma_frame_offset) << 8),
                chroma: chroma + (u64::from(params.chroma_frame_offset) << 8),
            }
        };

        ComposedFrame {
            data: &self.frame,
            dimensions: FrameDimensions::new(
                i64::from(params.pic_width_in_mbs) * 16,
                i64::from(params.frame_height_in_mbs) * 16,
            ),
            hidden: false,
            offsets,
        }
    }
}

fn write_parameter_sets(context: &H264DecoderContext, out: &mut Vec<u8>) {
    let params = &context.parameter_set;
    let mut w = BitWriter::new(out);

    w.bits(1, 24);
    w.bits(0, 1);
    w.bits(3, 2);
    w.bits(7, 5);
    w.bits(100, 8);
    w.bits(0, 8);
    w.bits(31, 8);
    w.ue(0);
    let chroma_format_idc = params.chroma_format_idc();
    w.ue(chroma_format_idc);
    if chroma_format_idc == 3 {
        w.bit(false);
    }
    w.ue(0);
    w.ue(0);
    w.bit(context.qpprime_y_zero_transform_bypass_flag());
    w.bit(false);
    w.ue(params.log2_max_frame_num_minus4());
    let pic_order_cnt_type = params.pic_order_cnt_type();
    w.ue(pic_order_cnt_type);
    if pic_order_cnt_type == 0 {
        w.ue(params.log2_max_pic_order_cnt_lsb_minus4 as u32);
    } else if pic_order_cnt_type == 1 {
        w.bit(params.delta_pic_order_always_zero_flag != 0);
        w.se(0);
        w.se(0);
        w.ue(0);
    }
    let frame_mbs_only = params.frame_mbs_only_flag != 0;
    let pic_height_in_map_units = params.frame_height_in_mbs / if frame_mbs_only { 1 } else { 2 };
    let max_num_ref_frames = (params
        .num_refidx_l0_default_active
        .max(params.num_refidx_l1_default_active)
        + 1)
    .max(4);
    w.ue(max_num_ref_frames as u32);
    w.bit(false);
    w.ue(params.pic_width_in_mbs.wrapping_sub(1));
    w.ue(pic_height_in_map_units.wrapping_sub(1));
    w.bit(frame_mbs_only);
    if !frame_mbs_only {
        w.bit(params.mbaff_frame());
    }
    w.bit(params.direct_8x8_inference());
    w.bit(false);
    w.bit(false);
    w.rbsp_trailing_bits();

    w.bits(1, 24);
    w.bits(0, 1);
    w.bits(3, 2);
    w.bits(8, 5);
    w.ue(0);
    w.ue(0);
    w.bit(params.entropy_coding_mode_flag != 0);
    w.bit(params.pic_order_present_flag != 0);
    w.ue(0);
    w.ue(params.num_refidx_l0_default_active as u32);
    w.ue(params.num_refidx_l1_default_active as u32);
    w.bit(params.weighted_pred());
    w.bits(u64::from(params.weighted_bipred_idc()), 2);
    w.se(params.pic_init_qp_minus26());
    w.se(0);
    w.se(params.chroma_qp_index_offset());
    w.bit(params.deblocking_filter_control_present_flag != 0);
    w.bit(params.constrained_intra_pred());
    w.bit(params.redundant_pic_cnt_present_flag != 0);
    w.bit(params.transform_8x8_mode_flag != 0);
    w.bit(true);
    for list in context.weight_scale_4x4.as_chunks::<16>().0 {
        w.bit(true);
        write_scaling_list(&mut w, list, &ZIGZAG_4X4);
    }
    if params.transform_8x8_mode_flag != 0 {
        for list in context.weight_scale_8x8.as_chunks::<64>().0 {
            w.bit(true);
            write_scaling_list(&mut w, list, &ZIGZAG_8X8);
        }
        if chroma_format_idc == 3 {
            for _ in 0..4 {
                w.bit(false);
            }
        }
    }
    w.se(params.second_chroma_qp_index_offset());
    w.rbsp_trailing_bits();
}

fn write_scaling_list(w: &mut BitWriter, list: &[u8], scan: &[u8]) {
    let mut last = 8;
    for &position in scan {
        let value = i32::from(list[usize::from(position)]);
        w.se(i32::from((value - last) as i8));
        last = value;
    }
}
