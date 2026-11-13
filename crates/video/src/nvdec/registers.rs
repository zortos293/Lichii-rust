//! The NVDEC register file: 0x178 64-bit method registers, written one method at a time by the
//! host1x channel. Buffer offsets are stored as device addresses shifted right by 8.

/// Codecs NVDEC can be configured for. Only the ones the builders support are representable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    H264,
    Vp8,
    Vp9,
}

impl VideoCodec {
    /// Decodes the codec id written to `SET_CODEC_ID`; `None` for codecs without a builder
    /// (none, H.265, AV1, or garbage).
    pub fn from_raw(raw: u64) -> Option<Self> {
        match raw {
            0x3 => Some(Self::H264),
            0x5 => Some(Self::Vp8),
            0x9 => Some(Self::Vp9),
            _ => None,
        }
    }

    /// The raw codec id as NVDEC encodes it.
    pub fn raw(self) -> u64 {
        match self {
            Self::H264 => 0x3,
            Self::Vp8 => 0x5,
            Self::Vp9 => 0x9,
        }
    }
}

/// Register indices (method numbers) of the fields the builders and the channel use.
pub mod index {
    pub const SET_CODEC_ID: usize = 0x80;
    pub const EXECUTE: usize = 0xC0;
    pub const CONTROL_PARAMS: usize = 0x100;
    pub const PICTURE_INFO_OFFSET: usize = 0x101;
    pub const FRAME_BITSTREAM_OFFSET: usize = 0x102;
    pub const FRAME_NUMBER: usize = 0x103;
    pub const H264_SLICE_DATA_OFFSETS: usize = 0x104;
    pub const H264_MV_DUMP_OFFSET: usize = 0x105;
    pub const FRAME_STATS_OFFSET: usize = 0x109;
    pub const H264_LAST_SURFACE_LUMA_OFFSET: usize = 0x10A;
    pub const H264_LAST_SURFACE_CHROMA_OFFSET: usize = 0x10B;
    pub const SURFACE_LUMA_OFFSETS: usize = 0x10C;
    pub const SURFACE_CHROMA_OFFSETS: usize = 0x11D;
    pub const PIC_SCRATCH_BUF_OFFSET: usize = 0x12E;
    pub const EXTERNAL_MVBUFFER_OFFSET: usize = 0x12F;
    pub const H264_MBHIST_BUFFER_OFFSET: usize = 0x140;
    pub const VP8_PROB_DATA_OFFSET: usize = 0x150;
    pub const VP8_HEADER_PARTITION_BUF_OFFSET: usize = 0x151;
    pub const VP9_PROB_TAB_BUFFER_OFFSET: usize = 0x170;
    pub const VP9_CTX_COUNTER_BUFFER_OFFSET: usize = 0x171;
    pub const VP9_SEGMENT_READ_BUFFER_OFFSET: usize = 0x172;
    pub const VP9_SEGMENT_WRITE_BUFFER_OFFSET: usize = 0x173;
    pub const VP9_TILE_SIZE_BUFFER_OFFSET: usize = 0x174;
    pub const VP9_COL_MVWRITE_BUFFER_OFFSET: usize = 0x175;
    pub const VP9_COL_MVREAD_BUFFER_OFFSET: usize = 0x176;
    pub const VP9_FILTER_BUFFER_OFFSET: usize = 0x177;
}

/// Number of surface offset registers per plane.
pub const SURFACE_COUNT: usize = 17;

/// The `CONTROL_PARAMS` register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControlParams(pub u64);

impl ControlParams {
    /// The low 3 bits of the codec id. VP9 (9) and AV1 (0xA) don't fit, so the decoder is chosen
    /// from `SET_CODEC_ID` instead.
    pub fn codec(self) -> u64 {
        self.0 & 0x7
    }

    pub fn gp_timer_on(self) -> bool {
        self.0 >> 4 & 1 != 0
    }

    pub fn mb_timer_on(self) -> bool {
        self.0 >> 13 & 1 != 0
    }

    pub fn intra_frame_pslc(self) -> bool {
        self.0 >> 14 & 1 != 0
    }

    pub fn all_intra_frame(self) -> bool {
        self.0 >> 17 & 1 != 0
    }
}

/// The NVDEC register file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NvdecRegisters {
    regs: [u64; Self::COUNT],
}

impl Default for NvdecRegisters {
    fn default() -> Self {
        Self {
            regs: [0; Self::COUNT],
        }
    }
}

impl NvdecRegisters {
    /// Number of registers.
    pub const COUNT: usize = 0x178;

    /// Writes `value` to register `method`. Writes past the register file are dropped and
    /// reported as `false`.
    pub fn write(&mut self, method: usize, value: u64) -> bool {
        match self.regs.get_mut(method) {
            Some(reg) => {
                *reg = value;
                true
            }
            None => false,
        }
    }

    /// The raw value of register `method`.
    pub fn read(&self, method: usize) -> u64 {
        self.regs[method]
    }

    fn address(&self, method: usize) -> u64 {
        self.regs[method] << 8
    }

    /// The codec id written to `SET_CODEC_ID`.
    pub fn codec_id(&self) -> Option<VideoCodec> {
        VideoCodec::from_raw(self.regs[index::SET_CODEC_ID])
    }

    pub fn execute(&self) -> u64 {
        self.regs[index::EXECUTE]
    }

    pub fn control_params(&self) -> ControlParams {
        ControlParams(self.regs[index::CONTROL_PARAMS])
    }

    /// Device address of the codec's picture info structure.
    pub fn picture_info_address(&self) -> u64 {
        self.address(index::PICTURE_INFO_OFFSET)
    }

    /// Device address of the compressed frame data.
    pub fn frame_bitstream_address(&self) -> u64 {
        self.address(index::FRAME_BITSTREAM_OFFSET)
    }

    pub fn frame_number(&self) -> u64 {
        self.regs[index::FRAME_NUMBER]
    }

    pub fn h264_slice_data_address(&self) -> u64 {
        self.address(index::H264_SLICE_DATA_OFFSETS)
    }

    pub fn h264_mv_dump_address(&self) -> u64 {
        self.address(index::H264_MV_DUMP_OFFSET)
    }

    pub fn frame_stats_address(&self) -> u64 {
        self.address(index::FRAME_STATS_OFFSET)
    }

    pub fn h264_last_surface_luma_address(&self) -> u64 {
        self.address(index::H264_LAST_SURFACE_LUMA_OFFSET)
    }

    pub fn h264_last_surface_chroma_address(&self) -> u64 {
        self.address(index::H264_LAST_SURFACE_CHROMA_OFFSET)
    }

    /// Device address of the luma plane of surface `surface`; 0 past [`SURFACE_COUNT`], as the
    /// index comes from guest picture parameters.
    pub fn surface_luma_address(&self, surface: usize) -> u64 {
        self.surface_address(index::SURFACE_LUMA_OFFSETS, surface)
    }

    /// Device address of the chroma plane of surface `surface`; 0 past [`SURFACE_COUNT`].
    pub fn surface_chroma_address(&self, surface: usize) -> u64 {
        self.surface_address(index::SURFACE_CHROMA_OFFSETS, surface)
    }

    fn surface_address(&self, first: usize, surface: usize) -> u64 {
        if surface < SURFACE_COUNT {
            self.address(first + surface)
        } else {
            0
        }
    }

    pub fn pic_scratch_buf_address(&self) -> u64 {
        self.address(index::PIC_SCRATCH_BUF_OFFSET)
    }

    pub fn external_mvbuffer_address(&self) -> u64 {
        self.address(index::EXTERNAL_MVBUFFER_OFFSET)
    }

    pub fn h264_mbhist_buffer_address(&self) -> u64 {
        self.address(index::H264_MBHIST_BUFFER_OFFSET)
    }

    pub fn vp8_prob_data_address(&self) -> u64 {
        self.address(index::VP8_PROB_DATA_OFFSET)
    }

    pub fn vp8_header_partition_buf_address(&self) -> u64 {
        self.address(index::VP8_HEADER_PARTITION_BUF_OFFSET)
    }

    /// Device address of the VP9 probability tables (`Vp9NvdecEntropyProbs`).
    pub fn vp9_prob_tab_buffer_address(&self) -> u64 {
        self.address(index::VP9_PROB_TAB_BUFFER_OFFSET)
    }

    pub fn vp9_ctx_counter_buffer_address(&self) -> u64 {
        self.address(index::VP9_CTX_COUNTER_BUFFER_OFFSET)
    }

    pub fn vp9_segment_read_buffer_address(&self) -> u64 {
        self.address(index::VP9_SEGMENT_READ_BUFFER_OFFSET)
    }

    pub fn vp9_segment_write_buffer_address(&self) -> u64 {
        self.address(index::VP9_SEGMENT_WRITE_BUFFER_OFFSET)
    }

    pub fn vp9_tile_size_buffer_address(&self) -> u64 {
        self.address(index::VP9_TILE_SIZE_BUFFER_OFFSET)
    }

    pub fn vp9_col_mvwrite_buffer_address(&self) -> u64 {
        self.address(index::VP9_COL_MVWRITE_BUFFER_OFFSET)
    }

    pub fn vp9_col_mvread_buffer_address(&self) -> u64 {
        self.address(index::VP9_COL_MVREAD_BUFFER_OFFSET)
    }

    pub fn vp9_filter_buffer_address(&self) -> u64 {
        self.address(index::VP9_FILTER_BUFFER_OFFSET)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_ids_round_trip_and_unsupported_ids_are_rejected() {
        for codec in [VideoCodec::H264, VideoCodec::Vp8, VideoCodec::Vp9] {
            assert_eq!(VideoCodec::from_raw(codec.raw()), Some(codec));
        }
        for raw in [0x0, 0x7, 0xA, 0x1234] {
            assert_eq!(VideoCodec::from_raw(raw), None);
        }
    }

    #[test]
    fn offsets_are_shifted_device_addresses() {
        let mut regs = NvdecRegisters::default();
        assert!(regs.write(index::SET_CODEC_ID, 0x9));
        assert!(regs.write(index::PICTURE_INFO_OFFSET, 0x12_3456));
        assert!(regs.write(index::SURFACE_LUMA_OFFSETS + 16, 0x77));
        assert!(regs.write(index::SURFACE_CHROMA_OFFSETS + 3, 0x88));
        assert!(regs.write(index::VP9_FILTER_BUFFER_OFFSET, 1));
        assert!(!regs.write(NvdecRegisters::COUNT, 1));
        assert_eq!(regs.codec_id(), Some(VideoCodec::Vp9));
        assert_eq!(regs.picture_info_address(), 0x1234_5600);
        assert_eq!(regs.surface_luma_address(16), 0x7700);
        assert_eq!(regs.surface_chroma_address(3), 0x8800);
        assert_eq!(regs.surface_luma_address(SURFACE_COUNT), 0);
        assert_eq!(regs.vp9_filter_buffer_address(), 0x100);
    }

    #[test]
    fn control_params_bits() {
        let params = ControlParams(0x3 | 1 << 4 | 1 << 13 | 1 << 14 | 1 << 17);
        assert_eq!(params.codec(), 3);
        assert!(params.gp_timer_on());
        assert!(params.mb_timer_on());
        assert!(params.intra_frame_pslc());
        assert!(params.all_intra_frame());
        assert!(!ControlParams(0).all_intra_frame());
    }
}
