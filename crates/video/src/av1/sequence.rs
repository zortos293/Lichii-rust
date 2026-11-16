//! `sequence_header_obu()` (AV1 specification section 5.5).

use super::bits::{Av1Error, BitReader, Result};

/// `SELECT_SCREEN_CONTENT_TOOLS` and `SELECT_INTEGER_MV`: the frame header chooses.
pub const SELECT: u8 = 2;

/// Timing information of a sequence (`timing_info()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TimingInfo {
    pub num_units_in_display_tick: u32,
    pub time_scale: u32,
    pub equal_picture_interval: bool,
    pub num_ticks_per_picture_minus_1: u32,
}

/// The decoder model of a sequence (`decoder_model_info()`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecoderModelInfo {
    pub buffer_delay_length_minus_1: u8,
    pub num_units_in_decoding_tick: u32,
    pub buffer_removal_time_length_minus_1: u8,
    pub frame_presentation_time_length_minus_1: u8,
}

/// One operating point of a sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OperatingPoint {
    pub idc: u16,
    pub seq_level_idx: u8,
    pub seq_tier: u8,
    pub decoder_model_present: bool,
}

/// `color_config()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorConfig {
    pub bit_depth: u8,
    pub mono_chrome: bool,
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
    pub chroma_sample_position: u8,
    pub separate_uv_delta_q: bool,
}

impl ColorConfig {
    pub fn num_planes(&self) -> usize {
        if self.mono_chrome { 1 } else { 3 }
    }
}

/// A parsed sequence header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SequenceHeader {
    pub seq_profile: u8,
    pub still_picture: bool,
    pub reduced_still_picture_header: bool,
    pub timing_info: Option<TimingInfo>,
    pub decoder_model_info: Option<DecoderModelInfo>,
    pub initial_display_delay_present: bool,
    pub operating_points: Vec<OperatingPoint>,
    pub frame_width_bits_minus_1: u8,
    pub frame_height_bits_minus_1: u8,
    pub max_frame_width_minus_1: u16,
    pub max_frame_height_minus_1: u16,
    pub frame_id_numbers_present: bool,
    pub delta_frame_id_length_minus_2: u8,
    pub additional_frame_id_length_minus_1: u8,
    pub use_128x128_superblock: bool,
    pub enable_filter_intra: bool,
    pub enable_intra_edge_filter: bool,
    pub enable_interintra_compound: bool,
    pub enable_masked_compound: bool,
    pub enable_warped_motion: bool,
    pub enable_dual_filter: bool,
    pub enable_order_hint: bool,
    pub enable_jnt_comp: bool,
    pub enable_ref_frame_mvs: bool,
    /// 0, 1 or [`SELECT`].
    pub seq_force_screen_content_tools: u8,
    /// 0, 1 or [`SELECT`].
    pub seq_force_integer_mv: u8,
    /// `OrderHintBits`, 0 without order hints.
    pub order_hint_bits: u8,
    pub enable_superres: bool,
    pub enable_cdef: bool,
    pub enable_restoration: bool,
    pub color: ColorConfig,
    pub film_grain_params_present: bool,
}

impl SequenceHeader {
    /// Parses the payload of a sequence header OBU.
    pub fn parse(data: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(data);
        let mut s = SequenceHeader {
            seq_profile: r.f(3)? as u8,
            still_picture: r.flag()?,
            reduced_still_picture_header: r.flag()?,
            ..Default::default()
        };
        if s.seq_profile > 2 {
            return Err(Av1Error::Invalid("seq_profile"));
        }
        if s.reduced_still_picture_header {
            s.operating_points.push(OperatingPoint {
                seq_level_idx: r.f(5)? as u8,
                ..Default::default()
            });
        } else {
            if r.flag()? {
                let mut timing = TimingInfo {
                    num_units_in_display_tick: r.f(32)?,
                    time_scale: r.f(32)?,
                    equal_picture_interval: r.flag()?,
                    num_ticks_per_picture_minus_1: 0,
                };
                if timing.equal_picture_interval {
                    timing.num_ticks_per_picture_minus_1 = r.uvlc()?;
                }
                s.timing_info = Some(timing);
                if r.flag()? {
                    s.decoder_model_info = Some(DecoderModelInfo {
                        buffer_delay_length_minus_1: r.f(5)? as u8,
                        num_units_in_decoding_tick: r.f(32)?,
                        buffer_removal_time_length_minus_1: r.f(5)? as u8,
                        frame_presentation_time_length_minus_1: r.f(5)? as u8,
                    });
                }
            }
            s.initial_display_delay_present = r.flag()?;
            let count = r.f(5)? + 1;
            for _ in 0..count {
                let mut point = OperatingPoint {
                    idc: r.f(12)? as u16,
                    seq_level_idx: r.f(5)? as u8,
                    ..Default::default()
                };
                if point.seq_level_idx > 7 {
                    point.seq_tier = r.f(1)? as u8;
                }
                if let Some(model) = s.decoder_model_info {
                    point.decoder_model_present = r.flag()?;
                    if point.decoder_model_present {
                        let n = u32::from(model.buffer_delay_length_minus_1) + 1;
                        r.f(n)?;
                        r.f(n)?;
                        r.f(1)?;
                    }
                }
                if s.initial_display_delay_present && r.flag()? {
                    r.f(4)?;
                }
                s.operating_points.push(point);
            }
        }
        s.frame_width_bits_minus_1 = r.f(4)? as u8;
        s.frame_height_bits_minus_1 = r.f(4)? as u8;
        s.max_frame_width_minus_1 = r.f(u32::from(s.frame_width_bits_minus_1) + 1)? as u16;
        s.max_frame_height_minus_1 = r.f(u32::from(s.frame_height_bits_minus_1) + 1)? as u16;
        if !s.reduced_still_picture_header {
            s.frame_id_numbers_present = r.flag()?;
        }
        if s.frame_id_numbers_present {
            s.delta_frame_id_length_minus_2 = r.f(4)? as u8;
            s.additional_frame_id_length_minus_1 = r.f(3)? as u8;
        }
        s.use_128x128_superblock = r.flag()?;
        s.enable_filter_intra = r.flag()?;
        s.enable_intra_edge_filter = r.flag()?;
        s.seq_force_screen_content_tools = SELECT;
        s.seq_force_integer_mv = SELECT;
        if !s.reduced_still_picture_header {
            s.enable_interintra_compound = r.flag()?;
            s.enable_masked_compound = r.flag()?;
            s.enable_warped_motion = r.flag()?;
            s.enable_dual_filter = r.flag()?;
            s.enable_order_hint = r.flag()?;
            if s.enable_order_hint {
                s.enable_jnt_comp = r.flag()?;
                s.enable_ref_frame_mvs = r.flag()?;
            }
            if !r.flag()? {
                s.seq_force_screen_content_tools = r.f(1)? as u8;
            }
            if s.seq_force_screen_content_tools > 0 {
                if !r.flag()? {
                    s.seq_force_integer_mv = r.f(1)? as u8;
                }
            } else {
                s.seq_force_integer_mv = SELECT;
            }
            if s.enable_order_hint {
                s.order_hint_bits = r.f(3)? as u8 + 1;
            }
        }
        s.enable_superres = r.flag()?;
        s.enable_cdef = r.flag()?;
        s.enable_restoration = r.flag()?;
        s.color = color_config(&mut r, s.seq_profile)?;
        s.film_grain_params_present = r.flag()?;
        Ok(s)
    }

    /// `idLen` of frame ids.
    pub fn frame_id_length(&self) -> u32 {
        u32::from(self.additional_frame_id_length_minus_1)
            + u32::from(self.delta_frame_id_length_minus_2)
            + 3
    }
}

fn color_config(r: &mut BitReader<'_>, seq_profile: u8) -> Result<ColorConfig> {
    let high_bitdepth = r.flag()?;
    let mut c = ColorConfig {
        bit_depth: if high_bitdepth { 10 } else { 8 },
        color_primaries: 2,
        transfer_characteristics: 2,
        matrix_coefficients: 2,
        ..Default::default()
    };
    if seq_profile == 2 && high_bitdepth && r.flag()? {
        c.bit_depth = 12;
    }
    if seq_profile != 1 {
        c.mono_chrome = r.flag()?;
    }
    if r.flag()? {
        c.color_primaries = r.f(8)? as u8;
        c.transfer_characteristics = r.f(8)? as u8;
        c.matrix_coefficients = r.f(8)? as u8;
    }
    if c.mono_chrome {
        c.color_range = r.flag()?;
        c.subsampling_x = true;
        c.subsampling_y = true;
        return Ok(c);
    }
    if c.color_primaries == 1 && c.transfer_characteristics == 13 && c.matrix_coefficients == 0 {
        c.color_range = true;
    } else {
        c.color_range = r.flag()?;
        match seq_profile {
            0 => (c.subsampling_x, c.subsampling_y) = (true, true),
            1 => {}
            _ if c.bit_depth == 12 => {
                c.subsampling_x = r.flag()?;
                if c.subsampling_x {
                    c.subsampling_y = r.flag()?;
                }
            }
            _ => c.subsampling_x = true,
        }
        if c.subsampling_x && c.subsampling_y {
            c.chroma_sample_position = r.f(2)? as u8;
        }
    }
    c.separate_uv_delta_q = r.flag()?;
    Ok(c)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::av1::bits::tests::BitWriter;

    /// A 1920x1080 profile 0 8-bit 4:2:0 sequence with order hints of 7 bits, CDEF, restoration
    /// and film grain: what a movie encoder writes.
    pub(crate) fn movie_sequence() -> Vec<u8> {
        let mut w = BitWriter::default();
        w.f(0, 3).flag(false).flag(false); // profile 0, not still, full header
        w.flag(true).f(1001, 32).f(60000, 32).flag(true).uvlc(0); // timing, no decoder model
        w.flag(false);
        w.flag(false); // initial_display_delay_present
        w.f(0, 5).f(0, 12).f(8, 5).f(0, 1); // one operating point, level 4.0 (8), tier 0
        w.f(10, 4).f(10, 4).f(1919, 11).f(1079, 11); // 11-bit sizes
        w.flag(false); // frame ids
        w.flag(false).flag(true).flag(true); // 64x64, filter intra, intra edge
        w.flag(true).flag(true).flag(true).flag(true).flag(true); // ii, masked, warp, dual, oh
        w.flag(true).flag(true); // jnt, ref mvs
        w.flag(true); // seq_choose_screen_content_tools -> SELECT
        w.flag(true); // seq_choose_integer_mv -> SELECT
        w.f(6, 3); // order_hint_bits_minus_1
        w.flag(false).flag(true).flag(true); // superres, cdef, restoration
        w.flag(false).flag(false).flag(true).f(1, 8).f(1, 8).f(1, 8); // 8-bit, colour BT.709
        w.flag(false).f(0, 2).flag(false); // limited range, csp, separate uv q
        w.flag(true); // film grain
        w.flag(true).align(); // trailing bit
        w.bytes.clone()
    }

    #[test]
    fn a_movie_sequence_header_reads_back() {
        let s = SequenceHeader::parse(&movie_sequence()).unwrap();
        assert_eq!(s.seq_profile, 0);
        assert_eq!(
            s.timing_info,
            Some(TimingInfo {
                num_units_in_display_tick: 1001,
                time_scale: 60000,
                equal_picture_interval: true,
                num_ticks_per_picture_minus_1: 0,
            })
        );
        assert_eq!(s.operating_points.len(), 1);
        assert_eq!(s.operating_points[0].seq_level_idx, 8);
        assert_eq!(
            (s.max_frame_width_minus_1, s.max_frame_height_minus_1),
            (1919, 1079)
        );
        assert!(!s.use_128x128_superblock && s.enable_order_hint && s.enable_ref_frame_mvs);
        assert_eq!(s.seq_force_screen_content_tools, SELECT);
        assert_eq!(s.seq_force_integer_mv, SELECT);
        assert_eq!(s.order_hint_bits, 7);
        assert!(s.enable_cdef && s.enable_restoration && !s.enable_superres);
        assert_eq!(s.color.bit_depth, 8);
        assert_eq!(s.color.matrix_coefficients, 1);
        assert!(s.color.subsampling_x && s.color.subsampling_y && !s.color.color_range);
        assert!(s.film_grain_params_present);
    }

    #[test]
    fn a_reduced_still_picture_header_takes_defaults() {
        let mut w = BitWriter::default();
        w.f(0, 3).flag(true).flag(true).f(5, 5); // still, reduced, level 5
        w.f(7, 4).f(7, 4).f(255, 8).f(127, 8); // 256x128
        w.flag(true).flag(false).flag(false); // 128x128 sb
        w.flag(false).flag(false).flag(false); // superres, cdef, restoration
        w.flag(false).flag(true).flag(false); // 8-bit, mono, no colour description
        w.flag(true); // full range
        w.flag(false); // film grain
        let s = SequenceHeader::parse(&w.bytes).unwrap();
        assert!(s.reduced_still_picture_header && s.use_128x128_superblock);
        assert_eq!(s.operating_points[0].seq_level_idx, 5);
        assert_eq!(s.order_hint_bits, 0);
        assert_eq!(s.seq_force_integer_mv, SELECT);
        assert!(s.color.mono_chrome && s.color.color_range);
        assert_eq!(s.color.num_planes(), 1);
    }

    #[test]
    fn twelve_bit_444_and_decoder_model() {
        let mut w = BitWriter::default();
        w.f(2, 3).flag(false).flag(false);
        w.flag(true).f(1, 32).f(30, 32).flag(false); // timing without equal interval
        w.flag(true).f(4, 5).f(90000, 32).f(9, 5).f(4, 5); // decoder model
        w.flag(true); // initial display delay present
        w.f(1, 5); // two operating points
        w.f(0x101, 12)
            .f(9, 5)
            .f(1, 1)
            .flag(true)
            .f(3, 5)
            .f(4, 5)
            .f(1, 1)
            .flag(true)
            .f(9, 4);
        w.f(0x103, 12).f(4, 5).flag(false).flag(false);
        w.f(11, 4).f(11, 4).f(4095, 12).f(2159, 12);
        w.flag(true).f(13, 4).f(6, 3); // frame ids
        w.flag(false).flag(false).flag(false);
        w.flag(false)
            .flag(false)
            .flag(false)
            .flag(false)
            .flag(false); // no order hint
        w.flag(false).f(1, 1); // screen content forced on
        w.flag(false).f(0, 1); // integer mv forced off
        w.flag(true).flag(false).flag(false);
        w.flag(true).flag(true).flag(false).flag(false); // 12-bit, not mono, no description
        w.flag(true).flag(false).flag(true); // full range, subsampling_x 0, sep uv q
        w.flag(false);
        let s = SequenceHeader::parse(&w.bytes).unwrap();
        assert_eq!(s.operating_points.len(), 2);
        assert_eq!(s.operating_points[0].seq_tier, 1);
        assert!(s.operating_points[0].decoder_model_present);
        assert_eq!(s.operating_points[1].idc, 0x103);
        assert_eq!(s.frame_id_length(), 6 + 13 + 3);
        assert_eq!(
            (s.seq_force_screen_content_tools, s.seq_force_integer_mv),
            (1, 0)
        );
        assert_eq!(s.color.bit_depth, 12);
        assert!(!s.color.subsampling_x && !s.color.subsampling_y && s.color.separate_uv_delta_q);
        assert!(s.enable_superres);
    }
}
