//! `uncompressed_header()` (AV1 specification section 5.9), the reference frame state it reads
//! and updates (section 7.20), and `tile_group_obu()` (section 5.11.1) up to the tile data.

use super::bits::{Av1Error, BitReader, Result};
use super::sequence::{SELECT, SequenceHeader};

pub const NUM_REF_FRAMES: usize = 8;
pub const REFS_PER_FRAME: usize = 7;
pub const PRIMARY_REF_NONE: u8 = 7;
pub const MAX_SEGMENTS: usize = 8;
pub const SEG_LVL_MAX: usize = 8;
pub const MAX_TILE_COLS: usize = 64;
pub const MAX_TILE_ROWS: usize = 64;
const WARPEDMODEL_PREC_BITS: u32 = 16;
const SEGMENTATION_FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 6, 6, 6, 3, 0, 0];
const SEGMENTATION_FEATURE_SIGNED: [bool; SEG_LVL_MAX] =
    [true, true, true, true, true, false, false, false];
const SEGMENTATION_FEATURE_MAX: [i32; SEG_LVL_MAX] = [255, 63, 63, 63, 63, 7, 0, 0];
/// Loop filter deltas after `setup_past_independence()`.
const DEFAULT_REF_DELTAS: [i8; NUM_REF_FRAMES] = [1, 0, 0, 0, -1, 0, -1, -1];

/// `frame_type`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameType {
    #[default]
    Key = 0,
    Inter = 1,
    IntraOnly = 2,
    Switch = 3,
}

impl FrameType {
    fn from_bits(bits: u32) -> Self {
        match bits {
            0 => Self::Key,
            1 => Self::Inter,
            2 => Self::IntraOnly,
            _ => Self::Switch,
        }
    }
}

/// Global motion: `GmType` and `gm_params` of each reference name (index 0 unused).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlobalMotion {
    pub gm_type: [u8; NUM_REF_FRAMES],
    pub gm_params: [[i32; 6]; NUM_REF_FRAMES],
}

impl Default for GlobalMotion {
    fn default() -> Self {
        let identity = [
            0,
            0,
            1 << WARPEDMODEL_PREC_BITS,
            0,
            0,
            1 << WARPEDMODEL_PREC_BITS,
        ];
        Self {
            gm_type: [0; NUM_REF_FRAMES],
            gm_params: [identity; NUM_REF_FRAMES],
        }
    }
}

/// `film_grain_params()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FilmGrain {
    pub apply_grain: bool,
    pub grain_seed: u16,
    pub update_grain: bool,
    pub film_grain_params_ref_idx: u8,
    pub num_y_points: u8,
    pub point_y_value: [u8; 14],
    pub point_y_scaling: [u8; 14],
    pub chroma_scaling_from_luma: bool,
    pub num_cb_points: u8,
    pub point_cb_value: [u8; 10],
    pub point_cb_scaling: [u8; 10],
    pub num_cr_points: u8,
    pub point_cr_value: [u8; 10],
    pub point_cr_scaling: [u8; 10],
    pub grain_scaling_minus_8: u8,
    pub ar_coeff_lag: u8,
    pub ar_coeffs_y_plus_128: [u8; 24],
    pub ar_coeffs_cb_plus_128: [u8; 25],
    pub ar_coeffs_cr_plus_128: [u8; 25],
    pub ar_coeff_shift_minus_6: u8,
    pub grain_scale_shift: u8,
    pub cb_mult: u8,
    pub cb_luma_mult: u8,
    pub cb_offset: u16,
    pub cr_mult: u8,
    pub cr_luma_mult: u8,
    pub cr_offset: u16,
    pub overlap_flag: bool,
    pub clip_to_restricted_range: bool,
}

/// `tile_info()`, with the start of every tile column and row in mode info units.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TileInfo {
    pub uniform_tile_spacing: bool,
    pub tile_cols: u32,
    pub tile_rows: u32,
    pub tile_cols_log2: u32,
    pub tile_rows_log2: u32,
    pub context_update_tile_id: u32,
    /// `TileSizeBytes - 1`.
    pub tile_size_bytes_minus_1: u8,
    /// `MiColStarts`/`MiRowStarts`, one past the last tile ending at `MiCols`/`MiRows`.
    pub mi_col_starts: Vec<u16>,
    pub mi_row_starts: Vec<u16>,
    pub width_in_sbs_minus_1: Vec<u16>,
    pub height_in_sbs_minus_1: Vec<u16>,
}

/// `quantization_params()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Quantization {
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_u_dc: i8,
    pub delta_q_u_ac: i8,
    pub delta_q_v_dc: i8,
    pub delta_q_v_ac: i8,
    pub diff_uv_delta: bool,
    pub using_qmatrix: bool,
    pub qm_y: u8,
    pub qm_u: u8,
    pub qm_v: u8,
}

/// `segmentation_params()`, with the features in force (loaded from the primary reference when
/// not updated).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

/// `loop_filter_params()`, with the deltas in force.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopFilter {
    pub level: [u8; 4],
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    pub ref_deltas: [i8; NUM_REF_FRAMES],
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilter {
    fn default() -> Self {
        Self {
            level: [0; 4],
            sharpness: 0,
            delta_enabled: true,
            delta_update: false,
            ref_deltas: DEFAULT_REF_DELTAS,
            mode_deltas: [0; 2],
        }
    }
}

/// `cdef_params()`; secondary strengths as coded (3 means 4).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cdef {
    pub damping_minus_3: u8,
    pub bits: u8,
    pub y_pri_strength: [u8; 8],
    pub y_sec_strength: [u8; 8],
    pub uv_pri_strength: [u8; 8],
    pub uv_sec_strength: [u8; 8],
}

/// `lr_params()`: the coded restoration types and unit shifts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoopRestoration {
    /// `lr_type` of each plane as coded (0 none, 1 switchable, 2 Wiener, 3 self-guided).
    pub lr_type: [u8; 3],
    /// `lr_unit_shift` after its increments: units of `64 << lr_unit_shift` luma samples.
    pub lr_unit_shift: u8,
    pub lr_uv_shift: u8,
}

impl LoopRestoration {
    pub fn uses_lr(&self) -> bool {
        self.lr_type.iter().any(|&t| t != 0)
    }
}

/// A parsed frame header.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameHeader {
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub frame_type: FrameType,
    pub show_frame: bool,
    pub showable_frame: bool,
    pub error_resilient_mode: bool,
    pub disable_cdf_update: bool,
    pub allow_screen_content_tools: bool,
    pub force_integer_mv: bool,
    pub current_frame_id: u32,
    pub frame_size_override_flag: bool,
    pub order_hint: u32,
    pub primary_ref_frame: u8,
    pub buffer_removal_time_present: bool,
    pub refresh_frame_flags: u8,
    pub frame_width: u32,
    pub frame_height: u32,
    pub upscaled_width: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub render_and_frame_size_different: bool,
    pub use_superres: bool,
    pub coded_denom: u8,
    pub allow_intrabc: bool,
    pub frame_refs_short_signaling: bool,
    /// `ref_frame_idx`: the reference slot each of LAST..ALTREF reads.
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    pub allow_high_precision_mv: bool,
    pub is_filter_switchable: bool,
    pub interpolation_filter: u8,
    pub is_motion_mode_switchable: bool,
    pub use_ref_frame_mvs: bool,
    /// `OrderHints[refFrame]` (index 0 is the intra frame, 0).
    pub order_hints: [u32; NUM_REF_FRAMES],
    /// `RefFrameSignBias`, a bit per reference name.
    pub ref_frame_sign_bias: u8,
    pub disable_frame_end_update_cdf: bool,
    pub tile_info: TileInfo,
    pub quantization: Quantization,
    pub segmentation: Segmentation,
    pub delta_q_present: bool,
    pub delta_q_res: u8,
    pub delta_lf_present: bool,
    pub delta_lf_res: u8,
    pub delta_lf_multi: bool,
    pub coded_lossless: bool,
    pub all_lossless: bool,
    pub loop_filter: LoopFilter,
    pub cdef: Cdef,
    pub loop_restoration: LoopRestoration,
    /// `TxMode`: 0 ONLY_4X4, 1 TX_MODE_LARGEST, 2 TX_MODE_SELECT.
    pub tx_mode: u8,
    pub reference_select: bool,
    pub skip_mode_present: bool,
    /// `SkipModeFrame`, reference names.
    pub skip_mode_frame: [u8; 2],
    pub allow_warped_motion: bool,
    pub reduced_tx_set: bool,
    pub global_motion: GlobalMotion,
    pub film_grain: FilmGrain,
    pub mi_cols: u32,
    pub mi_rows: u32,
}

impl FrameHeader {
    pub fn frame_is_intra(&self) -> bool {
        matches!(self.frame_type, FrameType::Key | FrameType::IntraOnly)
    }
}

/// What a reference slot holds for later frames (section 7.20).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RefFrame {
    pub valid: bool,
    pub frame_id: u32,
    pub upscaled_width: u32,
    pub frame_width: u32,
    pub frame_height: u32,
    pub render_width: u32,
    pub render_height: u32,
    pub mi_cols: u32,
    pub mi_rows: u32,
    pub frame_type: FrameType,
    pub order_hint: u32,
    pub saved_order_hints: [u32; NUM_REF_FRAMES],
    pub ref_frame_sign_bias: u8,
    pub global_motion: GlobalMotion,
    pub loop_filter_ref_deltas: [i8; NUM_REF_FRAMES],
    pub loop_filter_mode_deltas: [i8; 2],
    pub feature_enabled: [[bool; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
    pub film_grain: FilmGrain,
    pub disable_frame_end_update_cdf: bool,
    pub segmentation_enabled: bool,
    /// The frame header as decoded, for `show_existing_frame` of a key frame.
    pub showable: bool,
}

/// The reference slots of a stream.
#[derive(Clone, Debug, Default)]
pub struct References {
    pub slots: [RefFrame; NUM_REF_FRAMES],
}

impl References {
    /// The reference update process: every slot `refresh_frame_flags` names takes this frame.
    pub fn update(&mut self, h: &FrameHeader) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if h.refresh_frame_flags & (1 << i) == 0 {
                continue;
            }
            *slot = RefFrame {
                valid: true,
                frame_id: h.current_frame_id,
                upscaled_width: h.upscaled_width,
                frame_width: h.frame_width,
                frame_height: h.frame_height,
                render_width: h.render_width,
                render_height: h.render_height,
                mi_cols: h.mi_cols,
                mi_rows: h.mi_rows,
                frame_type: h.frame_type,
                order_hint: h.order_hint,
                saved_order_hints: h.order_hints,
                ref_frame_sign_bias: h.ref_frame_sign_bias,
                global_motion: h.global_motion,
                loop_filter_ref_deltas: h.loop_filter.ref_deltas,
                loop_filter_mode_deltas: h.loop_filter.mode_deltas,
                feature_enabled: h.segmentation.feature_enabled,
                feature_data: h.segmentation.feature_data,
                film_grain: h.film_grain,
                disable_frame_end_update_cdf: h.disable_frame_end_update_cdf,
                segmentation_enabled: h.segmentation.enabled,
                showable: h.showable_frame,
            };
        }
    }
}

/// `temporal_id` and `spatial_id` of the OBU the header came in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Layer {
    pub temporal_id: u8,
    pub spatial_id: u8,
}

struct Parser<'s, 'r, 'd> {
    seq: &'s SequenceHeader,
    refs: &'r References,
    r: BitReader<'d>,
    h: FrameHeader,
}

impl FrameHeader {
    /// Parses `uncompressed_header()` from the start of `data`. Returns the header and the bytes
    /// it took (rounded up to a byte, as `frame_header_obu()` ends with `byte_alignment()`).
    pub fn parse(
        seq: &SequenceHeader,
        refs: &References,
        layer: Layer,
        data: &[u8],
    ) -> Result<(Self, usize)> {
        let mut p = Parser {
            seq,
            refs,
            r: BitReader::new(data),
            h: FrameHeader::default(),
        };
        p.uncompressed_header(layer)?;
        let mut r = p.r;
        r.byte_alignment();
        Ok((p.h, r.position() / 8))
    }
}

impl Parser<'_, '_, '_> {
    fn relative_dist(&self, a: u32, b: u32) -> i32 {
        if !self.seq.enable_order_hint {
            return 0;
        }
        let bits = u32::from(self.seq.order_hint_bits);
        let diff = a as i32 - b as i32;
        let m = 1i32 << (bits - 1);
        (diff & (m - 1)) - (diff & m)
    }

    fn uncompressed_header(&mut self, layer: Layer) -> Result<()> {
        let seq = self.seq;
        let all_frames = 0xffu8;
        let id_len = seq.frame_id_length();
        let equal_interval = seq.timing_info.is_some_and(|t| t.equal_picture_interval);
        if seq.reduced_still_picture_header {
            self.h.frame_type = FrameType::Key;
            self.h.show_frame = true;
        } else {
            self.h.show_existing_frame = self.r.flag()?;
            if self.h.show_existing_frame {
                self.h.frame_to_show_map_idx = self.r.f(3)? as u8;
                if let Some(model) = seq.decoder_model_info
                    && !equal_interval
                {
                    self.r
                        .f(u32::from(model.frame_presentation_time_length_minus_1) + 1)?;
                }
                if seq.frame_id_numbers_present {
                    self.r.f(id_len)?;
                }
                let shown = &self.refs.slots[usize::from(self.h.frame_to_show_map_idx)];
                if !shown.valid {
                    return Err(Av1Error::Invalid("frame_to_show_map_idx"));
                }
                self.h.frame_type = shown.frame_type;
                if self.h.frame_type == FrameType::Key {
                    self.h.refresh_frame_flags = all_frames;
                }
                if seq.film_grain_params_present {
                    self.h.film_grain = shown.film_grain;
                }
                self.load_shown_frame(shown);
                return Ok(());
            }
            self.h.frame_type = FrameType::from_bits(self.r.f(2)?);
            self.h.show_frame = self.r.flag()?;
            if self.h.show_frame
                && let Some(model) = seq.decoder_model_info
                && !equal_interval
            {
                self.r
                    .f(u32::from(model.frame_presentation_time_length_minus_1) + 1)?;
            }
            self.h.showable_frame = if self.h.show_frame {
                self.h.frame_type != FrameType::Key
            } else {
                self.r.flag()?
            };
            self.h.error_resilient_mode = self.h.frame_type == FrameType::Switch
                || (self.h.frame_type == FrameType::Key && self.h.show_frame)
                || self.r.flag()?;
        }
        let key_shown = self.h.frame_type == FrameType::Key && self.h.show_frame;
        let mut ref_order_hint: [u32; NUM_REF_FRAMES] =
            std::array::from_fn(|i| self.refs.slots[i].order_hint);
        if key_shown {
            ref_order_hint = [0; NUM_REF_FRAMES];
        }
        self.h.disable_cdf_update = self.r.flag()?;
        self.h.allow_screen_content_tools = if seq.seq_force_screen_content_tools == SELECT {
            self.r.flag()?
        } else {
            seq.seq_force_screen_content_tools != 0
        };
        if self.h.allow_screen_content_tools {
            self.h.force_integer_mv = if seq.seq_force_integer_mv == SELECT {
                self.r.flag()?
            } else {
                seq.seq_force_integer_mv != 0
            };
        }
        if self.h.frame_is_intra() {
            self.h.force_integer_mv = true;
        }
        if seq.frame_id_numbers_present {
            self.h.current_frame_id = self.r.f(id_len)?;
        }
        self.h.frame_size_override_flag = if self.h.frame_type == FrameType::Switch {
            true
        } else if seq.reduced_still_picture_header {
            false
        } else {
            self.r.flag()?
        };
        self.h.order_hint = self.r.f(u32::from(seq.order_hint_bits))?;
        self.h.primary_ref_frame = if self.h.frame_is_intra() || self.h.error_resilient_mode {
            PRIMARY_REF_NONE
        } else {
            self.r.f(3)? as u8
        };
        if let Some(model) = seq.decoder_model_info {
            self.h.buffer_removal_time_present = self.r.flag()?;
            if self.h.buffer_removal_time_present {
                for point in &seq.operating_points {
                    if !point.decoder_model_present {
                        continue;
                    }
                    let idc = u32::from(point.idc);
                    let in_temporal = (idc >> layer.temporal_id) & 1 != 0;
                    let in_spatial = (idc >> (u32::from(layer.spatial_id) + 8)) & 1 != 0;
                    if idc == 0 || (in_temporal && in_spatial) {
                        self.r
                            .f(u32::from(model.buffer_removal_time_length_minus_1) + 1)?;
                    }
                }
            }
        }
        self.h.refresh_frame_flags = if self.h.frame_type == FrameType::Switch || key_shown {
            all_frames
        } else {
            self.r.f(8)? as u8
        };
        if (!self.h.frame_is_intra() || self.h.refresh_frame_flags != all_frames)
            && self.h.error_resilient_mode
            && seq.enable_order_hint
        {
            for hint in &mut ref_order_hint {
                *hint = self.r.f(u32::from(seq.order_hint_bits))?;
            }
        }
        if self.h.frame_is_intra() {
            self.frame_size()?;
            self.render_size()?;
            if self.h.allow_screen_content_tools && self.h.upscaled_width == self.h.frame_width {
                self.h.allow_intrabc = self.r.flag()?;
            }
        } else {
            if seq.enable_order_hint {
                self.h.frame_refs_short_signaling = self.r.flag()?;
                if self.h.frame_refs_short_signaling {
                    let last = self.r.f(3)? as u8;
                    let gold = self.r.f(3)? as u8;
                    self.set_frame_refs(last, gold, &ref_order_hint)?;
                }
            }
            for i in 0..REFS_PER_FRAME {
                if !self.h.frame_refs_short_signaling {
                    self.h.ref_frame_idx[i] = self.r.f(3)? as u8;
                }
                if seq.frame_id_numbers_present {
                    self.r.f(u32::from(seq.delta_frame_id_length_minus_2) + 2)?;
                }
            }
            if self.h.frame_size_override_flag && !self.h.error_resilient_mode {
                self.frame_size_with_refs()?;
            } else {
                self.frame_size()?;
                self.render_size()?;
            }
            self.h.allow_high_precision_mv = !self.h.force_integer_mv && self.r.flag()?;
            self.h.is_filter_switchable = self.r.flag()?;
            self.h.interpolation_filter = if self.h.is_filter_switchable {
                4
            } else {
                self.r.f(2)? as u8
            };
            self.h.is_motion_mode_switchable = self.r.flag()?;
            self.h.use_ref_frame_mvs =
                !self.h.error_resilient_mode && seq.enable_ref_frame_mvs && self.r.flag()?;
            for i in 0..REFS_PER_FRAME {
                let hint = ref_order_hint[usize::from(self.h.ref_frame_idx[i])];
                self.h.order_hints[i + 1] = hint;
                if seq.enable_order_hint && self.relative_dist(hint, self.h.order_hint) > 0 {
                    self.h.ref_frame_sign_bias |= 1 << (i + 1);
                }
            }
        }
        self.h.disable_frame_end_update_cdf =
            seq.reduced_still_picture_header || self.h.disable_cdf_update || self.r.flag()?;
        let primary = (self.h.primary_ref_frame != PRIMARY_REF_NONE).then(|| {
            &self.refs.slots
                [usize::from(self.h.ref_frame_idx[usize::from(self.h.primary_ref_frame)])]
        });
        let prev_gm = primary.map_or_else(GlobalMotion::default, |p| p.global_motion);
        if let Some(p) = primary {
            self.h.loop_filter.ref_deltas = p.loop_filter_ref_deltas;
            self.h.loop_filter.mode_deltas = p.loop_filter_mode_deltas;
            self.h.segmentation.feature_enabled = p.feature_enabled;
            self.h.segmentation.feature_data = p.feature_data;
        }
        self.tile_info()?;
        self.quantization_params()?;
        self.segmentation_params()?;
        self.delta_params()?;
        self.lossless();
        self.loop_filter_params()?;
        self.cdef_params()?;
        self.lr_params()?;
        self.h.tx_mode = if self.h.coded_lossless {
            0
        } else if self.r.flag()? {
            2
        } else {
            1
        };
        self.h.reference_select = !self.h.frame_is_intra() && self.r.flag()?;
        self.skip_mode_params(&ref_order_hint)?;
        self.h.allow_warped_motion = !self.h.frame_is_intra()
            && !self.h.error_resilient_mode
            && seq.enable_warped_motion
            && self.r.flag()?;
        self.h.reduced_tx_set = self.r.flag()?;
        self.global_motion_params(&prev_gm)?;
        self.film_grain_params()?;
        Ok(())
    }

    /// The frame loading process for `show_existing_frame` (section 7.21).
    fn load_shown_frame(&mut self, shown: &RefFrame) {
        let h = &mut self.h;
        h.current_frame_id = shown.frame_id;
        h.upscaled_width = shown.upscaled_width;
        h.frame_width = shown.frame_width;
        h.frame_height = shown.frame_height;
        h.render_width = shown.render_width;
        h.render_height = shown.render_height;
        h.mi_cols = shown.mi_cols;
        h.mi_rows = shown.mi_rows;
        h.order_hint = shown.order_hint;
        h.order_hints = shown.saved_order_hints;
        h.ref_frame_sign_bias = shown.ref_frame_sign_bias;
        h.global_motion = shown.global_motion;
        h.loop_filter.ref_deltas = shown.loop_filter_ref_deltas;
        h.loop_filter.mode_deltas = shown.loop_filter_mode_deltas;
        h.segmentation.feature_enabled = shown.feature_enabled;
        h.segmentation.feature_data = shown.feature_data;
        h.disable_frame_end_update_cdf = shown.disable_frame_end_update_cdf;
        h.segmentation.enabled = shown.segmentation_enabled;
        h.show_frame = true;
        h.showable_frame = false;
    }

    fn set_frame_refs(&mut self, last: u8, gold: u8, ref_order_hint: &[u32; 8]) -> Result<()> {
        const LAST: usize = 0;
        const GOLDEN: usize = 3;
        let mut idx = [-1i32; REFS_PER_FRAME];
        idx[LAST] = i32::from(last);
        idx[GOLDEN] = i32::from(gold);
        let mut used = [false; NUM_REF_FRAMES];
        used[usize::from(last)] = true;
        used[usize::from(gold)] = true;
        let bits = u32::from(self.seq.order_hint_bits);
        let cur = 1i32 << (bits - 1);
        let shifted: [i32; NUM_REF_FRAMES] =
            std::array::from_fn(|i| cur + self.relative_dist(ref_order_hint[i], self.h.order_hint));
        let last_hint = shifted[usize::from(last)];
        let gold_hint = shifted[usize::from(gold)];
        if last_hint >= cur || gold_hint >= cur {
            return Err(Av1Error::Invalid("frame_refs_short_signaling"));
        }
        let find_latest_backward = |used: &[bool; 8]| {
            let mut found = -1i32;
            let mut hint_found = 0;
            for i in 0..NUM_REF_FRAMES {
                let hint = shifted[i];
                if !used[i] && hint >= cur && (found < 0 || hint >= hint_found) {
                    found = i as i32;
                    hint_found = hint;
                }
            }
            found
        };
        let find_earliest_backward = |used: &[bool; 8]| {
            let mut found = -1i32;
            let mut hint_found = 0;
            for i in 0..NUM_REF_FRAMES {
                let hint = shifted[i];
                if !used[i] && hint >= cur && (found < 0 || hint < hint_found) {
                    found = i as i32;
                    hint_found = hint;
                }
            }
            found
        };
        let find_latest_forward = |used: &[bool; 8]| {
            let mut found = -1i32;
            let mut hint_found = 0;
            for i in 0..NUM_REF_FRAMES {
                let hint = shifted[i];
                if !used[i] && hint < cur && (found < 0 || hint >= hint_found) {
                    found = i as i32;
                    hint_found = hint;
                }
            }
            found
        };
        // ALTREF_FRAME - LAST_FRAME = 6, BWDREF 4, ALTREF2 5.
        for (slot, pick) in [(6usize, 0u8), (4, 1), (5, 1)] {
            let r = if pick == 0 {
                find_latest_backward(&used)
            } else {
                find_earliest_backward(&used)
            };
            if r >= 0 {
                idx[slot] = r;
                used[r as usize] = true;
            }
        }
        // Ref_Frame_List: LAST2, LAST3, BWDREF, ALTREF2, ALTREF.
        for slot in [1usize, 2, 4, 5, 6] {
            if idx[slot] < 0 {
                let r = find_latest_forward(&used);
                if r >= 0 {
                    idx[slot] = r;
                    used[r as usize] = true;
                }
            }
        }
        let mut earliest = -1i32;
        let mut earliest_hint = 0;
        for (i, &hint) in shifted.iter().enumerate() {
            if earliest < 0 || hint < earliest_hint {
                earliest = i as i32;
                earliest_hint = hint;
            }
        }
        for (slot, value) in idx.iter_mut().enumerate() {
            if *value < 0 {
                *value = earliest;
            }
            self.h.ref_frame_idx[slot] = *value as u8;
        }
        Ok(())
    }

    fn superres_params(&mut self) -> Result<()> {
        self.h.use_superres = self.seq.enable_superres && self.r.flag()?;
        let denom = if self.h.use_superres {
            self.h.coded_denom = self.r.f(3)? as u8;
            u32::from(self.h.coded_denom) + 9
        } else {
            8
        };
        self.h.upscaled_width = self.h.frame_width;
        self.h.frame_width = (self.h.upscaled_width * 8 + denom / 2) / denom;
        Ok(())
    }

    fn compute_image_size(&mut self) {
        self.h.mi_cols = 2 * self.h.frame_width.div_ceil(8);
        self.h.mi_rows = 2 * self.h.frame_height.div_ceil(8);
    }

    fn frame_size(&mut self) -> Result<()> {
        if self.h.frame_size_override_flag {
            self.h.frame_width = self.r.f(u32::from(self.seq.frame_width_bits_minus_1) + 1)? + 1;
            self.h.frame_height = self
                .r
                .f(u32::from(self.seq.frame_height_bits_minus_1) + 1)?
                + 1;
        } else {
            self.h.frame_width = u32::from(self.seq.max_frame_width_minus_1) + 1;
            self.h.frame_height = u32::from(self.seq.max_frame_height_minus_1) + 1;
        }
        self.superres_params()?;
        self.compute_image_size();
        Ok(())
    }

    fn render_size(&mut self) -> Result<()> {
        self.h.render_and_frame_size_different = self.r.flag()?;
        if self.h.render_and_frame_size_different {
            self.h.render_width = self.r.f(16)? + 1;
            self.h.render_height = self.r.f(16)? + 1;
        } else {
            self.h.render_width = self.h.upscaled_width;
            self.h.render_height = self.h.frame_height;
        }
        Ok(())
    }

    fn frame_size_with_refs(&mut self) -> Result<()> {
        for i in 0..REFS_PER_FRAME {
            if self.r.flag()? {
                let slot = &self.refs.slots[usize::from(self.h.ref_frame_idx[i])];
                self.h.upscaled_width = slot.upscaled_width;
                self.h.frame_width = self.h.upscaled_width;
                self.h.frame_height = slot.frame_height;
                self.h.render_width = slot.render_width;
                self.h.render_height = slot.render_height;
                self.superres_params()?;
                self.compute_image_size();
                return Ok(());
            }
        }
        self.frame_size()?;
        self.render_size()
    }

    fn tile_info(&mut self) -> Result<()> {
        let sb128 = self.seq.use_128x128_superblock;
        let (mi_cols, mi_rows) = (self.h.mi_cols, self.h.mi_rows);
        let sb_cols = if sb128 {
            mi_cols.div_ceil(32)
        } else {
            mi_cols.div_ceil(16)
        };
        let sb_rows = if sb128 {
            mi_rows.div_ceil(32)
        } else {
            mi_rows.div_ceil(16)
        };
        let sb_shift = if sb128 { 5 } else { 4 };
        let sb_size = sb_shift + 2;
        let max_tile_width_sb = 4096 >> sb_size;
        let mut max_tile_area_sb = (4096 * 2304) >> (2 * sb_size);
        let min_log2_tile_cols = tile_log2(max_tile_width_sb, sb_cols);
        let max_log2_tile_cols = tile_log2(1, sb_cols.min(MAX_TILE_COLS as u32));
        let max_log2_tile_rows = tile_log2(1, sb_rows.min(MAX_TILE_ROWS as u32));
        let min_log2_tiles = min_log2_tile_cols.max(tile_log2(max_tile_area_sb, sb_rows * sb_cols));
        let t = &mut self.h.tile_info;
        t.uniform_tile_spacing = self.r.flag()?;
        t.mi_col_starts.clear();
        t.mi_row_starts.clear();
        t.width_in_sbs_minus_1.clear();
        t.height_in_sbs_minus_1.clear();
        if t.uniform_tile_spacing {
            t.tile_cols_log2 = min_log2_tile_cols;
            while t.tile_cols_log2 < max_log2_tile_cols && self.r.flag()? {
                t.tile_cols_log2 += 1;
            }
            let tile_width_sb = (sb_cols + (1 << t.tile_cols_log2) - 1) >> t.tile_cols_log2;
            let mut start = 0;
            while start < sb_cols {
                t.mi_col_starts.push((start << sb_shift) as u16);
                t.width_in_sbs_minus_1
                    .push((tile_width_sb.min(sb_cols - start) - 1) as u16);
                start += tile_width_sb;
            }
            let min_log2_tile_rows = min_log2_tiles.saturating_sub(t.tile_cols_log2);
            t.tile_rows_log2 = min_log2_tile_rows;
            while t.tile_rows_log2 < max_log2_tile_rows && self.r.flag()? {
                t.tile_rows_log2 += 1;
            }
            let tile_height_sb = (sb_rows + (1 << t.tile_rows_log2) - 1) >> t.tile_rows_log2;
            let mut start = 0;
            while start < sb_rows {
                t.mi_row_starts.push((start << sb_shift) as u16);
                t.height_in_sbs_minus_1
                    .push((tile_height_sb.min(sb_rows - start) - 1) as u16);
                start += tile_height_sb;
            }
        } else {
            let mut widest = 0;
            let mut start = 0;
            while start < sb_cols {
                if t.mi_col_starts.len() >= MAX_TILE_COLS {
                    return Err(Av1Error::Invalid("tile columns"));
                }
                t.mi_col_starts.push((start << sb_shift) as u16);
                let max_width = (sb_cols - start).min(max_tile_width_sb);
                let size = self.r.ns(max_width)? + 1;
                t.width_in_sbs_minus_1.push((size - 1) as u16);
                widest = widest.max(size);
                start += size;
            }
            t.tile_cols_log2 = tile_log2(1, t.mi_col_starts.len() as u32);
            if min_log2_tiles > 0 {
                max_tile_area_sb = (sb_rows * sb_cols) >> (min_log2_tiles + 1);
            } else {
                max_tile_area_sb = sb_rows * sb_cols;
            }
            let max_tile_height_sb = (max_tile_area_sb / widest.max(1)).max(1);
            let mut start = 0;
            while start < sb_rows {
                if t.mi_row_starts.len() >= MAX_TILE_ROWS {
                    return Err(Av1Error::Invalid("tile rows"));
                }
                t.mi_row_starts.push((start << sb_shift) as u16);
                let max_height = (sb_rows - start).min(max_tile_height_sb);
                let size = self.r.ns(max_height)? + 1;
                t.height_in_sbs_minus_1.push((size - 1) as u16);
                start += size;
            }
            t.tile_rows_log2 = tile_log2(1, t.mi_row_starts.len() as u32);
        }
        t.tile_cols = t.mi_col_starts.len() as u32;
        t.tile_rows = t.mi_row_starts.len() as u32;
        t.mi_col_starts.push(mi_cols as u16);
        t.mi_row_starts.push(mi_rows as u16);
        if t.tile_cols_log2 > 0 || t.tile_rows_log2 > 0 {
            t.context_update_tile_id = self.r.f(t.tile_rows_log2 + t.tile_cols_log2)?;
            t.tile_size_bytes_minus_1 = self.r.f(2)? as u8;
        } else {
            t.context_update_tile_id = 0;
            t.tile_size_bytes_minus_1 = 0;
        }
        Ok(())
    }

    fn delta_q(&mut self) -> Result<i8> {
        Ok(if self.r.flag()? {
            self.r.su(7)? as i8
        } else {
            0
        })
    }

    fn quantization_params(&mut self) -> Result<()> {
        let planes = self.seq.color.num_planes();
        let separate = self.seq.color.separate_uv_delta_q;
        let mut q = Quantization {
            base_q_idx: self.r.f(8)? as u8,
            delta_q_y_dc: self.delta_q()?,
            ..Default::default()
        };
        if planes > 1 {
            q.diff_uv_delta = separate && self.r.flag()?;
            q.delta_q_u_dc = self.delta_q()?;
            q.delta_q_u_ac = self.delta_q()?;
            if q.diff_uv_delta {
                q.delta_q_v_dc = self.delta_q()?;
                q.delta_q_v_ac = self.delta_q()?;
            } else {
                q.delta_q_v_dc = q.delta_q_u_dc;
                q.delta_q_v_ac = q.delta_q_u_ac;
            }
        }
        q.using_qmatrix = self.r.flag()?;
        if q.using_qmatrix {
            q.qm_y = self.r.f(4)? as u8;
            q.qm_u = self.r.f(4)? as u8;
            q.qm_v = if separate { self.r.f(4)? as u8 } else { q.qm_u };
        }
        self.h.quantization = q;
        Ok(())
    }

    fn segmentation_params(&mut self) -> Result<()> {
        let s = &mut self.h.segmentation;
        s.enabled = self.r.flag()?;
        if !s.enabled {
            s.feature_enabled = [[false; SEG_LVL_MAX]; MAX_SEGMENTS];
            s.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
            return Ok(());
        }
        if self.h.primary_ref_frame == PRIMARY_REF_NONE {
            s.update_map = true;
            s.temporal_update = false;
            s.update_data = true;
        } else {
            s.update_map = self.r.flag()?;
            s.temporal_update = s.update_map && self.r.flag()?;
            s.update_data = self.r.flag()?;
        }
        if s.update_data {
            for i in 0..MAX_SEGMENTS {
                for j in 0..SEG_LVL_MAX {
                    let enabled = self.r.flag()?;
                    let mut value = 0;
                    if enabled {
                        let bits = SEGMENTATION_FEATURE_BITS[j];
                        let limit = SEGMENTATION_FEATURE_MAX[j];
                        value = if SEGMENTATION_FEATURE_SIGNED[j] {
                            self.r.su(1 + bits)?.clamp(-limit, limit)
                        } else {
                            (self.r.f(bits)? as i32).clamp(0, limit)
                        };
                    }
                    s.feature_enabled[i][j] = enabled;
                    s.feature_data[i][j] = value as i16;
                }
            }
        }
        Ok(())
    }

    fn delta_params(&mut self) -> Result<()> {
        if self.h.quantization.base_q_idx > 0 {
            self.h.delta_q_present = self.r.flag()?;
        }
        if self.h.delta_q_present {
            self.h.delta_q_res = self.r.f(2)? as u8;
            if !self.h.allow_intrabc {
                self.h.delta_lf_present = self.r.flag()?;
            }
            if self.h.delta_lf_present {
                self.h.delta_lf_res = self.r.f(2)? as u8;
                self.h.delta_lf_multi = self.r.flag()?;
            }
        }
        Ok(())
    }

    fn lossless(&mut self) {
        let q = &self.h.quantization;
        let s = &self.h.segmentation;
        let zero_deltas = q.delta_q_y_dc == 0
            && q.delta_q_u_ac == 0
            && q.delta_q_u_dc == 0
            && q.delta_q_v_ac == 0
            && q.delta_q_v_dc == 0;
        self.h.coded_lossless = (0..MAX_SEGMENTS).all(|segment| {
            let mut qindex = i32::from(q.base_q_idx);
            if s.enabled && s.feature_enabled[segment][0] {
                qindex = (qindex + i32::from(s.feature_data[segment][0])).clamp(0, 255);
            }
            qindex == 0 && zero_deltas
        });
        self.h.all_lossless = self.h.coded_lossless && self.h.frame_width == self.h.upscaled_width;
    }

    fn loop_filter_params(&mut self) -> Result<()> {
        let planes = self.seq.color.num_planes();
        let lf = &mut self.h.loop_filter;
        if self.h.coded_lossless || self.h.allow_intrabc {
            lf.level = [0; 4];
            lf.ref_deltas = DEFAULT_REF_DELTAS;
            lf.mode_deltas = [0; 2];
            return Ok(());
        }
        if self.h.primary_ref_frame == PRIMARY_REF_NONE {
            lf.ref_deltas = DEFAULT_REF_DELTAS;
            lf.mode_deltas = [0; 2];
        }
        lf.level[0] = self.r.f(6)? as u8;
        lf.level[1] = self.r.f(6)? as u8;
        if planes > 1 && (lf.level[0] != 0 || lf.level[1] != 0) {
            lf.level[2] = self.r.f(6)? as u8;
            lf.level[3] = self.r.f(6)? as u8;
        }
        lf.sharpness = self.r.f(3)? as u8;
        lf.delta_enabled = self.r.flag()?;
        lf.delta_update = false;
        if lf.delta_enabled {
            lf.delta_update = self.r.flag()?;
            if lf.delta_update {
                for delta in lf.ref_deltas.iter_mut().chain(lf.mode_deltas.iter_mut()) {
                    if self.r.flag()? {
                        *delta = self.r.su(7)? as i8;
                    }
                }
            }
        }
        Ok(())
    }

    fn cdef_params(&mut self) -> Result<()> {
        let planes = self.seq.color.num_planes();
        let c = &mut self.h.cdef;
        *c = Cdef::default();
        if self.h.coded_lossless || self.h.allow_intrabc || !self.seq.enable_cdef {
            return Ok(());
        }
        c.damping_minus_3 = self.r.f(2)? as u8;
        c.bits = self.r.f(2)? as u8;
        for i in 0..(1usize << c.bits) {
            c.y_pri_strength[i] = self.r.f(4)? as u8;
            c.y_sec_strength[i] = self.r.f(2)? as u8;
            if planes > 1 {
                c.uv_pri_strength[i] = self.r.f(4)? as u8;
                c.uv_sec_strength[i] = self.r.f(2)? as u8;
            }
        }
        Ok(())
    }

    fn lr_params(&mut self) -> Result<()> {
        let planes = self.seq.color.num_planes();
        let lr = &mut self.h.loop_restoration;
        *lr = LoopRestoration::default();
        if self.h.all_lossless || self.h.allow_intrabc || !self.seq.enable_restoration {
            return Ok(());
        }
        let mut uses_chroma = false;
        for (i, lr_type) in lr.lr_type.iter_mut().enumerate().take(planes) {
            *lr_type = self.r.f(2)? as u8;
            if i > 0 && *lr_type != 0 {
                uses_chroma = true;
            }
        }
        if lr.uses_lr() {
            if self.seq.use_128x128_superblock {
                lr.lr_unit_shift = self.r.f(1)? as u8 + 1;
            } else {
                lr.lr_unit_shift = self.r.f(1)? as u8;
                if lr.lr_unit_shift != 0 {
                    lr.lr_unit_shift += self.r.f(1)? as u8;
                }
            }
            let c = &self.seq.color;
            if c.subsampling_x && c.subsampling_y && uses_chroma {
                lr.lr_uv_shift = self.r.f(1)? as u8;
            }
        }
        Ok(())
    }

    fn skip_mode_params(&mut self, ref_order_hint: &[u32; 8]) -> Result<()> {
        let allowed =
            if self.h.frame_is_intra() || !self.h.reference_select || !self.seq.enable_order_hint {
                None
            } else {
                let hint_of = |i: usize| ref_order_hint[usize::from(self.h.ref_frame_idx[i])];
                let cur = self.h.order_hint;
                let (mut forward, mut forward_hint) = (None, 0);
                let (mut backward, mut backward_hint) = (None, 0);
                for i in 0..REFS_PER_FRAME {
                    let hint = hint_of(i);
                    let dist = self.relative_dist(hint, cur);
                    if dist < 0 {
                        if forward.is_none() || self.relative_dist(hint, forward_hint) > 0 {
                            forward = Some(i);
                            forward_hint = hint;
                        }
                    } else if dist > 0
                        && (backward.is_none() || self.relative_dist(hint, backward_hint) < 0)
                    {
                        backward = Some(i);
                        backward_hint = hint;
                    }
                }
                match (forward, backward) {
                    (None, _) => None,
                    (Some(f), Some(b)) => Some((f.min(b), f.max(b))),
                    (Some(f), None) => {
                        let (mut second, mut second_hint) = (None, 0);
                        for i in 0..REFS_PER_FRAME {
                            let hint = hint_of(i);
                            if self.relative_dist(hint, forward_hint) < 0
                                && (second.is_none() || self.relative_dist(hint, second_hint) > 0)
                            {
                                second = Some(i);
                                second_hint = hint;
                            }
                        }
                        second.map(|s| (f.min(s), f.max(s)))
                    }
                }
            };
        if let Some((a, b)) = allowed {
            self.h.skip_mode_frame = [a as u8 + 1, b as u8 + 1];
            self.h.skip_mode_present = self.r.flag()?;
        }
        Ok(())
    }

    fn global_motion_params(&mut self, prev: &GlobalMotion) -> Result<()> {
        self.h.global_motion = GlobalMotion::default();
        if self.h.frame_is_intra() {
            return Ok(());
        }
        for reference in 1..NUM_REF_FRAMES {
            let kind = if self.r.flag()? {
                if self.r.flag()? {
                    2
                } else if self.r.flag()? {
                    1
                } else {
                    3
                }
            } else {
                0
            };
            self.h.global_motion.gm_type[reference] = kind;
            if kind >= 2 {
                self.global_param(kind, reference, 2, prev)?;
                self.global_param(kind, reference, 3, prev)?;
                if kind == 3 {
                    self.global_param(kind, reference, 4, prev)?;
                    self.global_param(kind, reference, 5, prev)?;
                } else {
                    let params = &mut self.h.global_motion.gm_params[reference];
                    params[4] = -params[3];
                    params[5] = params[2];
                }
            }
            if kind >= 1 {
                self.global_param(kind, reference, 0, prev)?;
                self.global_param(kind, reference, 1, prev)?;
            }
        }
        Ok(())
    }

    fn global_param(
        &mut self,
        kind: u8,
        reference: usize,
        idx: usize,
        prev: &GlobalMotion,
    ) -> Result<()> {
        let mut abs_bits = 12;
        let mut prec_bits = 15;
        if idx < 2 {
            if kind == 1 {
                let hp = u32::from(!self.h.allow_high_precision_mv);
                abs_bits = 9 - hp;
                prec_bits = 3 - hp;
            } else {
                abs_bits = 12;
                prec_bits = 6;
            }
        }
        let prec_diff = WARPEDMODEL_PREC_BITS - prec_bits;
        let (round, sub) = if idx % 3 == 2 {
            (1 << WARPEDMODEL_PREC_BITS, 1 << prec_bits)
        } else {
            (0, 0)
        };
        let mx = 1i32 << abs_bits;
        let r = (prev.gm_params[reference][idx] >> prec_diff) - sub;
        let value = self.r.signed_subexp_with_ref(-mx, mx + 1, r)?;
        self.h.global_motion.gm_params[reference][idx] = (value << prec_diff) + round;
        Ok(())
    }

    fn film_grain_params(&mut self) -> Result<()> {
        let seq = self.seq;
        if !seq.film_grain_params_present || (!self.h.show_frame && !self.h.showable_frame) {
            self.h.film_grain = FilmGrain::default();
            return Ok(());
        }
        let mut g = FilmGrain {
            apply_grain: self.r.flag()?,
            ..Default::default()
        };
        if !g.apply_grain {
            self.h.film_grain = FilmGrain::default();
            return Ok(());
        }
        g.grain_seed = self.r.f(16)? as u16;
        g.update_grain = self.h.frame_type != FrameType::Inter || self.r.flag()?;
        if !g.update_grain {
            g.film_grain_params_ref_idx = self.r.f(3)? as u8;
            let loaded = &self.refs.slots[usize::from(g.film_grain_params_ref_idx)];
            if !loaded.valid {
                return Err(Av1Error::Invalid("film_grain_params_ref_idx"));
            }
            let seed = g.grain_seed;
            let index = g.film_grain_params_ref_idx;
            g = loaded.film_grain;
            g.grain_seed = seed;
            g.film_grain_params_ref_idx = index;
            g.update_grain = false;
            self.h.film_grain = g;
            return Ok(());
        }
        let c = &seq.color;
        g.num_y_points = self.r.f(4)? as u8;
        if g.num_y_points > 14 {
            return Err(Av1Error::Invalid("num_y_points"));
        }
        for i in 0..usize::from(g.num_y_points) {
            g.point_y_value[i] = self.r.f(8)? as u8;
            g.point_y_scaling[i] = self.r.f(8)? as u8;
        }
        g.chroma_scaling_from_luma = !c.mono_chrome && self.r.flag()?;
        if !(c.mono_chrome
            || g.chroma_scaling_from_luma
            || (c.subsampling_x && c.subsampling_y && g.num_y_points == 0))
        {
            g.num_cb_points = self.r.f(4)? as u8;
            if g.num_cb_points > 10 {
                return Err(Av1Error::Invalid("num_cb_points"));
            }
            for i in 0..usize::from(g.num_cb_points) {
                g.point_cb_value[i] = self.r.f(8)? as u8;
                g.point_cb_scaling[i] = self.r.f(8)? as u8;
            }
            g.num_cr_points = self.r.f(4)? as u8;
            if g.num_cr_points > 10 {
                return Err(Av1Error::Invalid("num_cr_points"));
            }
            for i in 0..usize::from(g.num_cr_points) {
                g.point_cr_value[i] = self.r.f(8)? as u8;
                g.point_cr_scaling[i] = self.r.f(8)? as u8;
            }
        }
        g.grain_scaling_minus_8 = self.r.f(2)? as u8;
        g.ar_coeff_lag = self.r.f(2)? as u8;
        let lag = usize::from(g.ar_coeff_lag);
        let num_pos_luma = 2 * lag * (lag + 1);
        let num_pos_chroma = if g.num_y_points > 0 {
            for coeff in &mut g.ar_coeffs_y_plus_128[..num_pos_luma] {
                *coeff = self.r.f(8)? as u8;
            }
            num_pos_luma + 1
        } else {
            num_pos_luma
        };
        if g.chroma_scaling_from_luma || g.num_cb_points > 0 {
            for coeff in &mut g.ar_coeffs_cb_plus_128[..num_pos_chroma] {
                *coeff = self.r.f(8)? as u8;
            }
        }
        if g.chroma_scaling_from_luma || g.num_cr_points > 0 {
            for coeff in &mut g.ar_coeffs_cr_plus_128[..num_pos_chroma] {
                *coeff = self.r.f(8)? as u8;
            }
        }
        g.ar_coeff_shift_minus_6 = self.r.f(2)? as u8;
        g.grain_scale_shift = self.r.f(2)? as u8;
        if g.num_cb_points > 0 {
            g.cb_mult = self.r.f(8)? as u8;
            g.cb_luma_mult = self.r.f(8)? as u8;
            g.cb_offset = self.r.f(9)? as u16;
        }
        if g.num_cr_points > 0 {
            g.cr_mult = self.r.f(8)? as u8;
            g.cr_luma_mult = self.r.f(8)? as u8;
            g.cr_offset = self.r.f(9)? as u16;
        }
        g.overlap_flag = self.r.flag()?;
        g.clip_to_restricted_range = self.r.flag()?;
        self.h.film_grain = g;
        Ok(())
    }
}

fn tile_log2(blk_size: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk_size << k) < target {
        k += 1;
    }
    k
}

/// One tile of a tile group: where its data starts in the OBU payload and how many bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub offset: usize,
    pub size: usize,
}

/// `tile_group_obu()` up to the tile data: the tiles of `payload` (a tile group OBU's payload,
/// or what follows the frame header in a frame OBU). Returns the tiles and whether the group
/// ends the frame.
pub fn tile_group(info: &TileInfo, payload: &[u8], tiles: &mut Vec<Tile>) -> Result<bool> {
    let num_tiles = info.tile_cols * info.tile_rows;
    let mut r = BitReader::new(payload);
    let present = num_tiles > 1 && r.flag()?;
    let (start, end) = if present {
        let bits = info.tile_cols_log2 + info.tile_rows_log2;
        (r.f(bits)?, r.f(bits)?)
    } else {
        (0, num_tiles.saturating_sub(1))
    };
    if end < start || end >= num_tiles {
        return Err(Av1Error::Invalid("tg_end"));
    }
    r.byte_alignment();
    let mut at = r.position() / 8;
    let size_bytes = usize::from(info.tile_size_bytes_minus_1) + 1;
    for tile in start..=end {
        let size = if tile == end {
            payload.len().checked_sub(at).ok_or(Av1Error::Truncated)?
        } else {
            let mut r = BitReader::new(payload.get(at..).ok_or(Av1Error::Truncated)?);
            let size = r.le(size_bytes as u32)? as usize + 1;
            at += size_bytes;
            size
        };
        if at + size > payload.len() {
            return Err(Av1Error::Truncated);
        }
        tiles.push(Tile { offset: at, size });
        at += size;
    }
    Ok(end == num_tiles - 1)
}
