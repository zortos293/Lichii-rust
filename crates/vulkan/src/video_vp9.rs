//! VP9 decoding on the host GPU's video queue (`VK_KHR_video_decode_vp9`).
//!
//! `lichii_video::vp9` splits superframes and reads each frame's uncompressed header; this fills
//! Vulkan's VP9 std structures and decodes the frame through [`DecodeCore`]. VP9 has no session
//! parameters, and the compressed header is left to the video engine.

use std::collections::VecDeque;
use std::sync::Arc;

use ash::vk;
use lichii_video::vp9::{
    self, FrameHeader, InterpolationFilter, NUM_REF_FRAMES, REFS_PER_FRAME, Vp9Stream,
};
use lichii_video::{Codec, Color, GpuStream, Matrix, OwnedPicture, VideoError};

use crate::video::VideoDevice;
use crate::video_decode::{CodecProfile, DecodeCore, MAX_SLOTS, Slot};
use crate::vp9_std::{
    self, StdVideoDecodeVP9PictureInfo, StdVideoVP9ColorConfig, StdVideoVP9LoopFilter,
    StdVideoVP9Segmentation, VideoDecodeVp9PictureInfo,
};

/// The Vulkan side of one VP9 stream.
struct Gpu {
    core: DecodeCore,
    /// The DPB slot each reference slot's picture is in, and its size.
    slot_of_ref: [Option<(u32, vk::Extent2D)>; NUM_REF_FRAMES],
}

/// One VP9 stream decoding on the video queue.
pub(crate) struct Vp9Decoder {
    parser: Vp9Stream,
    gpu: Gpu,
    ready: VecDeque<OwnedPicture>,
    spare: Vec<Vec<u8>>,
}

impl Vp9Decoder {
    pub(crate) fn new(device: Arc<VideoDevice>) -> Result<Self, VideoError> {
        let vp9 = device
            .video
            .vp9
            .clone()
            .ok_or(VideoError::NoDecoder(Codec::Vp9))?;
        let core = DecodeCore::new(
            device,
            CodecProfile::Vp9(Box::new(vp9_std::VideoDecodeVp9ProfileInfo::profile_0())),
            &vp9.caps,
            vp9_std::STD_EXTENSION_NAME,
            REFS_PER_FRAME as u32,
        )?;
        Ok(Self {
            parser: Vp9Stream::default(),
            gpu: Gpu {
                core,
                slot_of_ref: [None; NUM_REF_FRAMES],
            },
            ready: VecDeque::new(),
            spare: Vec::new(),
        })
    }
}

impl GpuStream for Vp9Decoder {
    fn send(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        for (offset, size) in vp9::superframe(data) {
            let frame = &data[offset..offset + size];
            let header = match self.parser.frame(frame) {
                Ok(header) => header,
                Err(error) => {
                    tracing::debug!("VP9 frame at pts {pts} does not parse: {error}");
                    return Err(VideoError::Gpu(0));
                }
            };
            self.gpu
                .frame(&header, frame, pts, &mut self.ready, &mut self.spare)?;
        }
        Ok(())
    }

    fn end_of_stream(&mut self) {}

    fn receive(&mut self, out: &mut OwnedPicture) -> Result<bool, VideoError> {
        let Some(picture) = self.ready.pop_front() else {
            return Ok(false);
        };
        let old = std::mem::replace(out, picture);
        if !old.data.is_empty() {
            self.spare.push(old.data);
        }
        Ok(true)
    }

    fn flush(&mut self) {
        while let Some(picture) = self.ready.pop_front() {
            self.spare.push(picture.data);
        }
        self.parser.reset();
        self.gpu.slot_of_ref = [None; NUM_REF_FRAMES];
    }
}

/// The colour description of pictures with colour configuration `c`.
fn color(c: &vp9::ColorConfig) -> Color {
    Color {
        matrix: match c.color_space {
            vp9::CS_BT_601 | 3 => Some(Matrix::Bt601),
            vp9::CS_BT_709 => Some(Matrix::Bt709),
            vp9::CS_BT_2020 => Some(Matrix::Bt2020),
            _ => None,
        },
        full_range: c.color_range,
    }
}

/// A frame header as Vulkan's VP9 std structures; the caller links the pointers.
struct StdPicture {
    info: StdVideoDecodeVP9PictureInfo,
    color: StdVideoVP9ColorConfig,
    loop_filter: StdVideoVP9LoopFilter,
    segmentation: StdVideoVP9Segmentation,
}

impl StdPicture {
    fn new(h: &FrameHeader) -> Self {
        let bit = |set: bool, n: u32| u32::from(set) << n;
        let lf = &h.loop_filter;
        let s = &h.segmentation;
        let info = StdVideoDecodeVP9PictureInfo {
            flags: bit(h.error_resilient_mode, 0)
                | bit(h.intra_only, 1)
                | bit(h.allow_high_precision_mv, 2)
                | bit(h.refresh_frame_context, 3)
                | bit(h.frame_parallel_decoding_mode, 4)
                | bit(s.enabled, 5)
                | bit(h.show_frame, 6)
                | bit(h.use_prev_frame_mvs, 7),
            profile: i32::from(h.profile),
            frame_type: i32::from(!h.key_frame),
            frame_context_idx: h.frame_context_idx,
            reset_frame_context: h.reset_frame_context,
            refresh_frame_flags: h.refresh_frame_flags,
            ref_frame_sign_bias_mask: h.ref_frame_sign_bias_mask,
            interpolation_filter: match h.interpolation_filter {
                InterpolationFilter::EightTap => 0,
                InterpolationFilter::EightTapSmooth => 1,
                InterpolationFilter::EightTapSharp => 2,
                InterpolationFilter::Bilinear => 3,
                InterpolationFilter::Switchable => 4,
            },
            base_q_idx: h.base_q_idx,
            delta_q_y_dc: h.delta_q_y_dc,
            delta_q_uv_dc: h.delta_q_uv_dc,
            delta_q_uv_ac: h.delta_q_uv_ac,
            tile_cols_log2: h.tile_cols_log2,
            tile_rows_log2: h.tile_rows_log2,
            reserved1: [0; 3],
            pColorConfig: std::ptr::null(),
            pLoopFilter: std::ptr::null(),
            pSegmentation: std::ptr::null(),
        };
        let c = &h.color;
        Self {
            info,
            color: StdVideoVP9ColorConfig {
                flags: bit(c.color_range, 0),
                BitDepth: c.bit_depth,
                subsampling_x: u8::from(c.subsampling_x),
                subsampling_y: u8::from(c.subsampling_y),
                reserved1: 0,
                color_space: i32::from(c.color_space),
            },
            loop_filter: StdVideoVP9LoopFilter {
                flags: bit(lf.delta_enabled, 0) | bit(lf.delta_update, 1),
                loop_filter_level: lf.level,
                loop_filter_sharpness: lf.sharpness,
                update_ref_delta: lf.update_ref_delta,
                loop_filter_ref_deltas: lf.ref_deltas,
                update_mode_delta: lf.update_mode_delta,
                loop_filter_mode_deltas: lf.mode_deltas,
            },
            segmentation: StdVideoVP9Segmentation {
                flags: bit(s.update_map, 0)
                    | bit(s.temporal_update, 1)
                    | bit(s.update_data, 2)
                    | bit(s.abs_or_delta_update, 3),
                segmentation_tree_probs: s.tree_probs,
                segmentation_pred_prob: s.pred_probs,
                FeatureEnabled: s.feature_enabled,
                FeatureData: s.feature_data,
            },
        }
    }
}

impl Gpu {
    /// Decodes (or shows) one frame and queues its picture when it is shown.
    fn frame(
        &mut self,
        h: &FrameHeader,
        data: &[u8],
        pts: i64,
        ready: &mut VecDeque<OwnedPicture>,
        spare: &mut Vec<Vec<u8>>,
    ) -> Result<(), VideoError> {
        if h.show_existing_frame {
            if let Some((slot, extent)) = self.slot_of_ref[usize::from(h.frame_to_show_map_idx)] {
                let picture =
                    self.core
                        .copy_out(slot, extent, false, color(&h.color), pts, spare)?;
                ready.push_back(picture);
            }
            return Ok(());
        }
        let c = &h.color;
        if h.profile != 0 || c.bit_depth != 8 || !c.subsampling_x || !c.subsampling_y {
            return Err(VideoError::Gpu(0));
        }
        if self.core.ensure_session(h.width, h.height)? {
            self.slot_of_ref = [None; NUM_REF_FRAMES];
        }
        let mut name_slots = [-1i32; REFS_PER_FRAME];
        let mut refs: Vec<Slot> = Vec::new();
        if !h.frame_is_intra() {
            for (name, &index) in name_slots.iter_mut().zip(&h.ref_frame_idx) {
                let Some((slot, extent)) = self.slot_of_ref[usize::from(index)] else {
                    // A reference this stream never decoded (a session made again mid-stream):
                    // wait for the next key frame.
                    return Ok(());
                };
                *name = slot as i32;
                if !refs.iter().any(|r| r.index == slot) {
                    refs.push(Slot {
                        index: slot,
                        extent,
                        info: std::ptr::null(),
                    });
                }
            }
        }
        let used: Vec<u32> = self.slot_of_ref.iter().flatten().map(|r| r.0).collect();
        let setup = (0..MAX_SLOTS)
            .find(|s| !used.contains(s))
            .expect("nine slots hold eight references and the new picture");
        let extent = vk::Extent2D {
            width: h.width,
            height: h.height,
        };

        self.core.staging.clear();
        let start = self.core.stage(data);
        let std = StdPicture::new(h);
        let mut info = std.info;
        info.pColorConfig = &std.color;
        info.pLoopFilter = &std.loop_filter;
        info.pSegmentation = &std.segmentation;
        let mut vp9_info = VideoDecodeVp9PictureInfo::new(&info);
        vp9_info.reference_name_slot_indices = name_slots;
        vp9_info.uncompressed_header_offset = start;
        vp9_info.compressed_header_offset = start + h.uncompressed_header_size;
        vp9_info.tiles_offset = start + h.uncompressed_header_size + h.compressed_header_size;
        let setup_slot = Slot {
            index: setup,
            extent,
            info: std::ptr::null(),
        };
        // SAFETY: a VP9 picture info for this VP9 core, alive until the call returns; VP9 has no
        // DPB slot infos.
        unsafe {
            self.core.decode(
                (&vp9_info as *const VideoDecodeVp9PictureInfo).cast(),
                &setup_slot,
                &refs,
                extent,
            )?;
        }
        let picture = self
            .core
            .copy_out(setup, extent, true, color(c), pts, spare)?;
        if h.show_frame {
            ready.push_back(picture);
        } else {
            spare.push(picture.data);
        }
        for (i, slot) in self.slot_of_ref.iter_mut().enumerate() {
            if h.refresh_frame_flags & (1 << i) != 0 {
                *slot = Some((setup, extent));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::{Instance, SurfaceSupport};
    use lichii_common::settings::GraphicsSettings;

    /// The frames of an IVF file: a header of the size its bytes 6-7 give, then a 12-byte header
    /// before each frame.
    fn ivf_frames(bytes: &[u8]) -> Vec<&[u8]> {
        let mut at = usize::from(u16::from_le_bytes([bytes[6], bytes[7]]));
        let mut frames = Vec::new();
        while at + 12 <= bytes.len() {
            let size = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            frames.push(&bytes[at + 12..at + 12 + size]);
            at += 12 + size;
        }
        frames
    }

    /// (Y, U, V) of quadrant `index` of frame `frame` (see `lichii-video`'s
    /// `tests/data/generate.py`).
    fn quadrant(frame: usize, index: usize) -> [u8; 3] {
        [
            (40 + 30 * frame + 10 * index) as u8,
            (90 + 20 * index) as u8,
            (200 - 15 * frame - 20 * index) as u8,
        ]
    }

    #[test]
    fn a_lossless_stream_decodes_to_its_exact_colours() {
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let device = match crate::device::Device::new(instance, &GraphicsSettings::default(), None)
        {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let fits = device
            .video
            .as_ref()
            .and_then(|v| v.video.vp9.as_ref())
            .is_some_and(|vp9| vp9.caps.min_extent.width <= 64 && vp9.caps.min_extent.height <= 48);
        if !fits {
            eprintln!("skipping: no VP9 decode queue for 64x48 pictures");
            return;
        }
        let video = device.video.clone().expect("checked above");
        let mut decoder = Vp9Decoder::new(video).unwrap();
        let stream = include_bytes!("../../video/tests/data/quadrants.vp9.ivf");
        let mut pictures = Vec::new();
        for (pts, packet) in ivf_frames(stream).into_iter().enumerate() {
            decoder.send(packet, pts as i64).unwrap();
            let mut picture = OwnedPicture::default();
            while decoder.receive(&mut picture).unwrap() {
                pictures.push(std::mem::take(&mut picture));
            }
        }
        assert_eq!(pictures.len(), 6);
        for (frame, picture) in pictures.iter().enumerate() {
            assert_eq!((picture.width, picture.height), (64, 48));
            let stride = picture.stride();
            let chroma = stride * 48;
            for index in 0..4 {
                let (x, y) = (8 + 32 * (index % 2), 6 + 24 * (index / 2));
                let luma = picture.data[y * stride + x];
                let uv = chroma + (y / 2) * stride + (x / 2) * 2;
                let got = [luma, picture.data[uv], picture.data[uv + 1]];
                assert_eq!(
                    got,
                    quadrant(frame, index),
                    "frame {frame} quadrant {index}"
                );
            }
        }
    }
}
