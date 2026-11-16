//! VP9 headers for decoders that take them parsed (Vulkan Video): superframe splitting (VP9
//! bitstream specification Annex B) and the uncompressed header of each frame (section 6.2),
//! with the state it carries from frame to frame: reference frame sizes, loop filter deltas,
//! segmentation parameters and colour configuration. The compressed header and the tiles are
//! left to the decoder; only their offsets are found.

#[cfg(test)]
mod tests;

/// Why a VP9 frame did not parse.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Vp9Error {
    /// The data ended inside the uncompressed header.
    #[error("VP9 data ends early")]
    Truncated,
    /// A field holds a value the specification forbids.
    #[error("VP9 header field {0} is invalid")]
    Invalid(&'static str),
    /// An inter frame came before any key frame.
    #[error("VP9 inter frame without a key frame")]
    NoKeyFrame,
}

type Result<T> = std::result::Result<T, Vp9Error>;

/// Reference frame slots.
pub const NUM_REF_FRAMES: usize = 8;
/// References an inter frame names (LAST, GOLDEN, ALTREF).
pub const REFS_PER_FRAME: usize = 3;
/// Segments of a segmentation map.
pub const MAX_SEGMENTS: usize = 8;
/// Segment features (quantizer, loop filter, reference frame, skip).
pub const SEG_LVL_MAX: usize = 4;

/// `color_space` of the colour configuration (section 7.2.2).
pub const CS_BT_601: u8 = 1;
pub const CS_BT_709: u8 = 2;
pub const CS_BT_2020: u8 = 5;
pub const CS_RGB: u8 = 7;

/// Reads bits MSB-first (section 4.9 `f(n)`) and VP9's sign-after-magnitude `su(n)`.
struct BitReader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }

    fn f(&mut self, n: u32) -> Result<u32> {
        let mut value = 0u32;
        for _ in 0..n {
            let byte = *self
                .data
                .get(self.position / 8)
                .ok_or(Vp9Error::Truncated)?;
            value = value << 1 | u32::from(byte >> (7 - self.position % 8) & 1);
            self.position += 1;
        }
        Ok(value)
    }

    fn flag(&mut self) -> Result<bool> {
        Ok(self.f(1)? != 0)
    }

    /// `su(n)`: `n` magnitude bits, then a sign bit.
    fn su(&mut self, n: u32) -> Result<i32> {
        let value = self.f(n)? as i32;
        Ok(if self.flag()? { -value } else { value })
    }

    fn bytes_read(&self) -> usize {
        self.position.div_ceil(8)
    }
}

/// The frames of one VP9 packet: the frames a superframe index at its end lists (Annex B), or
/// the whole packet. Each range is `(offset, size)` within `data`.
pub fn superframe(data: &[u8]) -> Vec<(usize, usize)> {
    let whole = vec![(0, data.len())];
    let Some(&marker) = data.last() else {
        return Vec::new();
    };
    if marker & 0xe0 != 0xc0 {
        return whole;
    }
    let frames = usize::from(marker & 7) + 1;
    let size_bytes = usize::from(marker >> 3 & 3) + 1;
    let index_size = 2 + size_bytes * frames;
    if data.len() < index_size || data[data.len() - index_size] != marker {
        return whole;
    }
    let index = &data[data.len() - index_size + 1..data.len() - 1];
    let mut ranges = Vec::with_capacity(frames);
    let mut offset = 0;
    for entry in index.chunks_exact(size_bytes) {
        let size = entry
            .iter()
            .rev()
            .fold(0usize, |size, &byte| size << 8 | usize::from(byte));
        if size == 0 || offset + size > data.len() - index_size {
            return whole;
        }
        ranges.push((offset, size));
        offset += size;
    }
    ranges
}

/// The colour configuration (section 6.2.2).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ColorConfig {
    pub bit_depth: u8,
    pub color_space: u8,
    pub color_range: bool,
    pub subsampling_x: bool,
    pub subsampling_y: bool,
}

/// Loop filter parameters (section 6.2.8), the deltas as they stand after this frame's updates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopFilter {
    pub level: u8,
    pub sharpness: u8,
    pub delta_enabled: bool,
    pub delta_update: bool,
    /// Bit `i` set when this frame updated `ref_deltas[i]`.
    pub update_ref_delta: u8,
    pub ref_deltas: [i8; 4],
    /// Bit `i` set when this frame updated `mode_deltas[i]`.
    pub update_mode_delta: u8,
    pub mode_deltas: [i8; 2],
}

impl Default for LoopFilter {
    fn default() -> Self {
        Self {
            level: 0,
            sharpness: 0,
            delta_enabled: false,
            delta_update: false,
            update_ref_delta: 0,
            ref_deltas: [1, 0, -1, -1],
            update_mode_delta: 0,
            mode_deltas: [0, 0],
        }
    }
}

/// Segmentation parameters (section 6.2.11), the features as they stand after this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segmentation {
    pub enabled: bool,
    pub update_map: bool,
    pub temporal_update: bool,
    pub update_data: bool,
    pub abs_or_delta_update: bool,
    pub tree_probs: [u8; 7],
    pub pred_probs: [u8; 3],
    /// Bit `j` of entry `i` set when segment `i` has feature `j`.
    pub feature_enabled: [u8; MAX_SEGMENTS],
    pub feature_data: [[i16; SEG_LVL_MAX]; MAX_SEGMENTS],
}

impl Default for Segmentation {
    fn default() -> Self {
        Self {
            enabled: false,
            update_map: false,
            temporal_update: false,
            update_data: false,
            abs_or_delta_update: false,
            tree_probs: [255; 7],
            pred_probs: [255; 3],
            feature_enabled: [0; MAX_SEGMENTS],
            feature_data: [[0; SEG_LVL_MAX]; MAX_SEGMENTS],
        }
    }
}

/// The interpolation filter of an inter frame, named as Vulkan's `StdVideoVP9InterpolationFilter`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InterpolationFilter {
    #[default]
    EightTap,
    EightTapSmooth,
    EightTapSharp,
    Bilinear,
    Switchable,
}

/// One frame's uncompressed header and where the rest of the frame starts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameHeader {
    pub profile: u8,
    pub show_existing_frame: bool,
    pub frame_to_show_map_idx: u8,
    pub key_frame: bool,
    pub show_frame: bool,
    pub error_resilient_mode: bool,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub color: ColorConfig,
    pub width: u32,
    pub height: u32,
    pub refresh_frame_flags: u8,
    pub ref_frame_idx: [u8; REFS_PER_FRAME],
    /// Bit `1 + i` set when reference `i` (LAST, GOLDEN, ALTREF) has a sign bias.
    pub ref_frame_sign_bias_mask: u8,
    pub allow_high_precision_mv: bool,
    pub interpolation_filter: InterpolationFilter,
    pub refresh_frame_context: bool,
    pub frame_parallel_decoding_mode: bool,
    /// The frame context the header names, as read (the decoder applies the resets).
    pub frame_context_idx: u8,
    pub loop_filter: LoopFilter,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_uv_dc: i8,
    pub delta_q_uv_ac: i8,
    pub segmentation: Segmentation,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    /// Bytes of the uncompressed header, where the compressed header starts.
    pub uncompressed_header_size: u32,
    /// Bytes of the compressed header; the tiles follow it.
    pub compressed_header_size: u32,
    /// The previous frame's motion vectors predict this frame's (libvpx's `use_prev_frame_mvs`).
    pub use_prev_frame_mvs: bool,
}

impl FrameHeader {
    pub fn frame_is_intra(&self) -> bool {
        self.key_frame || self.intra_only
    }
}

/// The state VP9 headers carry between frames.
#[derive(Clone, Debug, Default)]
pub struct Vp9Stream {
    color: Option<ColorConfig>,
    ref_sizes: [Option<(u32, u32)>; NUM_REF_FRAMES],
    loop_filter: LoopFilter,
    segmentation: Segmentation,
    last_size: Option<(u32, u32)>,
    last_show_frame: bool,
    last_intra_only: bool,
}

impl Vp9Stream {
    /// Forgets every frame, as at the start of a stream.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// The size of reference slot `index`, `None` before a frame filled it.
    pub fn ref_size(&self, index: usize) -> Option<(u32, u32)> {
        self.ref_sizes[index]
    }

    /// Reads the uncompressed header of one frame (one superframe entry) and updates the
    /// stream state as decoding it will.
    pub fn frame(&mut self, data: &[u8]) -> Result<FrameHeader> {
        let mut r = BitReader::new(data);
        let mut h = FrameHeader::default();
        if r.f(2)? != 2 {
            return Err(Vp9Error::Invalid("frame_marker"));
        }
        let low = r.f(1)?;
        let high = r.f(1)?;
        h.profile = (high << 1 | low) as u8;
        if h.profile == 3 && r.flag()? {
            return Err(Vp9Error::Invalid("reserved_zero"));
        }
        h.show_existing_frame = r.flag()?;
        if h.show_existing_frame {
            h.frame_to_show_map_idx = r.f(3)? as u8;
            h.show_frame = true;
            let (width, height) =
                self.ref_sizes[usize::from(h.frame_to_show_map_idx)].ok_or(Vp9Error::NoKeyFrame)?;
            h.width = width;
            h.height = height;
            h.color = self.color.unwrap_or_default();
            self.last_show_frame = true;
            return Ok(h);
        }
        h.key_frame = !r.flag()?;
        h.show_frame = r.flag()?;
        h.error_resilient_mode = r.flag()?;
        if h.key_frame {
            sync_code(&mut r)?;
            h.color = color_config(&mut r, h.profile)?;
            self.color = Some(h.color);
            frame_size(&mut r, &mut h)?;
            render_size(&mut r)?;
            h.refresh_frame_flags = 0xff;
        } else {
            h.intra_only = if h.show_frame { false } else { r.flag()? };
            h.reset_frame_context = if h.error_resilient_mode {
                0
            } else {
                r.f(2)? as u8
            };
            if h.intra_only {
                sync_code(&mut r)?;
                h.color = if h.profile > 0 {
                    color_config(&mut r, h.profile)?
                } else {
                    ColorConfig {
                        bit_depth: 8,
                        color_space: CS_BT_601,
                        color_range: false,
                        subsampling_x: true,
                        subsampling_y: true,
                    }
                };
                self.color = Some(h.color);
                h.refresh_frame_flags = r.f(8)? as u8;
                frame_size(&mut r, &mut h)?;
                render_size(&mut r)?;
            } else {
                h.color = self.color.ok_or(Vp9Error::NoKeyFrame)?;
                h.refresh_frame_flags = r.f(8)? as u8;
                for i in 0..REFS_PER_FRAME {
                    h.ref_frame_idx[i] = r.f(3)? as u8;
                    if r.flag()? {
                        h.ref_frame_sign_bias_mask |= 1 << (1 + i);
                    }
                }
                self.frame_size_with_refs(&mut r, &mut h)?;
                h.allow_high_precision_mv = r.flag()?;
                h.interpolation_filter = if r.flag()? {
                    InterpolationFilter::Switchable
                } else {
                    [
                        InterpolationFilter::EightTapSmooth,
                        InterpolationFilter::EightTap,
                        InterpolationFilter::EightTapSharp,
                        InterpolationFilter::Bilinear,
                    ][r.f(2)? as usize]
                };
            }
        }
        if h.error_resilient_mode {
            h.refresh_frame_context = false;
            h.frame_parallel_decoding_mode = true;
        } else {
            h.refresh_frame_context = r.flag()?;
            h.frame_parallel_decoding_mode = r.flag()?;
        }
        h.frame_context_idx = r.f(2)? as u8;
        if h.frame_is_intra() || h.error_resilient_mode {
            self.loop_filter = LoopFilter::default();
            let segmentation = &mut self.segmentation;
            segmentation.feature_enabled = [0; MAX_SEGMENTS];
            segmentation.feature_data = [[0; SEG_LVL_MAX]; MAX_SEGMENTS];
            segmentation.abs_or_delta_update = false;
        }
        self.loop_filter_params(&mut r)?;
        h.loop_filter = self.loop_filter;
        h.base_q_idx = r.f(8)? as u8;
        h.delta_q_y_dc = delta_q(&mut r)?;
        h.delta_q_uv_dc = delta_q(&mut r)?;
        h.delta_q_uv_ac = delta_q(&mut r)?;
        self.segmentation_params(&mut r)?;
        h.segmentation = self.segmentation;
        tile_info(&mut r, &mut h)?;
        h.compressed_header_size = r.f(16)?;
        if h.compressed_header_size == 0 {
            return Err(Vp9Error::Invalid("header_size_in_bytes"));
        }
        h.uncompressed_header_size = r.bytes_read() as u32;
        if (h.uncompressed_header_size + h.compressed_header_size) as usize > data.len() {
            return Err(Vp9Error::Truncated);
        }

        h.use_prev_frame_mvs = !h.error_resilient_mode
            && self.last_size == Some((h.width, h.height))
            && self.last_show_frame
            && !self.last_intra_only;
        for (i, size) in self.ref_sizes.iter_mut().enumerate() {
            if h.refresh_frame_flags & (1 << i) != 0 {
                *size = Some((h.width, h.height));
            }
        }
        self.last_size = Some((h.width, h.height));
        self.last_show_frame = h.show_frame;
        self.last_intra_only = h.intra_only;
        Ok(h)
    }

    fn frame_size_with_refs(&self, r: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<()> {
        for i in 0..REFS_PER_FRAME {
            if r.flag()? {
                let (width, height) =
                    self.ref_sizes[usize::from(h.ref_frame_idx[i])].ok_or(Vp9Error::NoKeyFrame)?;
                h.width = width;
                h.height = height;
                return render_size(r);
            }
        }
        frame_size(r, h)?;
        render_size(r)
    }

    fn loop_filter_params(&mut self, r: &mut BitReader<'_>) -> Result<()> {
        let lf = &mut self.loop_filter;
        lf.level = r.f(6)? as u8;
        lf.sharpness = r.f(3)? as u8;
        lf.delta_enabled = r.flag()?;
        lf.delta_update = false;
        lf.update_ref_delta = 0;
        lf.update_mode_delta = 0;
        if lf.delta_enabled {
            lf.delta_update = r.flag()?;
            if lf.delta_update {
                for i in 0..4 {
                    if r.flag()? {
                        lf.update_ref_delta |= 1 << i;
                        lf.ref_deltas[i] = r.su(6)? as i8;
                    }
                }
                for i in 0..2 {
                    if r.flag()? {
                        lf.update_mode_delta |= 1 << i;
                        lf.mode_deltas[i] = r.su(6)? as i8;
                    }
                }
            }
        }
        Ok(())
    }

    fn segmentation_params(&mut self, r: &mut BitReader<'_>) -> Result<()> {
        const FEATURE_BITS: [u32; SEG_LVL_MAX] = [8, 6, 2, 0];
        const FEATURE_SIGNED: [bool; SEG_LVL_MAX] = [true, true, false, false];
        let s = &mut self.segmentation;
        s.enabled = r.flag()?;
        s.update_map = false;
        s.temporal_update = false;
        s.update_data = false;
        if !s.enabled {
            return Ok(());
        }
        s.update_map = r.flag()?;
        if s.update_map {
            for prob in &mut s.tree_probs {
                *prob = read_prob(r)?;
            }
            s.temporal_update = r.flag()?;
            for prob in &mut s.pred_probs {
                *prob = if s.temporal_update {
                    read_prob(r)?
                } else {
                    255
                };
            }
        }
        s.update_data = r.flag()?;
        if s.update_data {
            s.abs_or_delta_update = r.flag()?;
            for segment in 0..MAX_SEGMENTS {
                s.feature_enabled[segment] = 0;
                for feature in 0..SEG_LVL_MAX {
                    let mut value = 0i16;
                    if r.flag()? {
                        s.feature_enabled[segment] |= 1 << feature;
                        value = r.f(FEATURE_BITS[feature])? as i16;
                        if FEATURE_SIGNED[feature] && r.flag()? {
                            value = -value;
                        }
                    }
                    s.feature_data[segment][feature] = value;
                }
            }
        }
        Ok(())
    }
}

fn sync_code(r: &mut BitReader<'_>) -> Result<()> {
    if r.f(24)? != 0x49_83_42 {
        return Err(Vp9Error::Invalid("frame_sync_code"));
    }
    Ok(())
}

fn color_config(r: &mut BitReader<'_>, profile: u8) -> Result<ColorConfig> {
    let bit_depth = if profile >= 2 {
        if r.flag()? { 12 } else { 10 }
    } else {
        8
    };
    let color_space = r.f(3)? as u8;
    let mut color = ColorConfig {
        bit_depth,
        color_space,
        color_range: true,
        subsampling_x: profile == 0 || profile == 2,
        subsampling_y: profile == 0 || profile == 2,
    };
    if color_space != CS_RGB {
        color.color_range = r.flag()?;
        if profile == 1 || profile == 3 {
            color.subsampling_x = r.flag()?;
            color.subsampling_y = r.flag()?;
            if r.flag()? {
                return Err(Vp9Error::Invalid("reserved_zero"));
            }
        } else {
            color.subsampling_x = true;
            color.subsampling_y = true;
        }
    } else if profile == 1 || profile == 3 {
        color.subsampling_x = false;
        color.subsampling_y = false;
        if r.flag()? {
            return Err(Vp9Error::Invalid("reserved_zero"));
        }
    }
    Ok(color)
}

fn frame_size(r: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<()> {
    h.width = r.f(16)? + 1;
    h.height = r.f(16)? + 1;
    Ok(())
}

fn render_size(r: &mut BitReader<'_>) -> Result<()> {
    if r.flag()? {
        r.f(32)?;
    }
    Ok(())
}

fn delta_q(r: &mut BitReader<'_>) -> Result<i8> {
    Ok(if r.flag()? { r.su(4)? as i8 } else { 0 })
}

fn read_prob(r: &mut BitReader<'_>) -> Result<u8> {
    Ok(if r.flag()? { r.f(8)? as u8 } else { 255 })
}

/// `tile_info()` (section 6.2.13): column counts bounded by the frame width in superblocks.
fn tile_info(r: &mut BitReader<'_>, h: &mut FrameHeader) -> Result<()> {
    let mi_cols = h.width.div_ceil(8);
    let sb64_cols = mi_cols.div_ceil(8);
    let mut min_log2 = 0;
    while (64 << min_log2) < sb64_cols {
        min_log2 += 1;
    }
    let mut max_log2 = 1;
    while (sb64_cols >> max_log2) >= 4 {
        max_log2 += 1;
    }
    max_log2 -= 1;
    let mut cols = min_log2;
    while cols < max_log2 && r.flag()? {
        cols += 1;
    }
    h.tile_cols_log2 = cols as u8;
    let mut rows = r.f(1)?;
    if rows != 0 {
        rows += r.f(1)?;
    }
    h.tile_rows_log2 = rows as u8;
    Ok(())
}
