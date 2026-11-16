//! `VK_KHR_video_decode_vp9` and the `vulkan_video_codec_vp9std` structures, which ash 0.38
//! (Vulkan 1.3.281) predates. Layouts follow `vulkan_core.h` and
//! `vulkan_video_codec_vp9std{,_decode}.h`; C bit-fields are `u32` words with the fields from
//! bit 0 up in declaration order.

#![allow(non_snake_case)]

use std::ffi::{CStr, c_void};

use ash::vk;

pub(crate) const EXTENSION_NAME: &CStr = c"VK_KHR_video_decode_vp9";
pub(crate) const STD_EXTENSION_NAME: &CStr = c"VK_STD_vulkan_video_codec_vp9_decode";

pub(crate) const DECODE_VP9: vk::VideoCodecOperationFlagsKHR =
    vk::VideoCodecOperationFlagsKHR::from_raw(0x0000_0008);

const PHYSICAL_DEVICE_FEATURES: vk::StructureType = vk::StructureType::from_raw(1_000_514_000);
const CAPABILITIES: vk::StructureType = vk::StructureType::from_raw(1_000_514_001);
const PICTURE_INFO: vk::StructureType = vk::StructureType::from_raw(1_000_514_002);
const PROFILE_INFO: vk::StructureType = vk::StructureType::from_raw(1_000_514_003);

/// `STD_VIDEO_VP9_PROFILE_0`: 8-bit 4:2:0.
pub(crate) const PROFILE_0: i32 = 0;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PhysicalDeviceVideoDecodeVp9Features {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    pub video_decode_vp9: vk::Bool32,
}

impl Default for PhysicalDeviceVideoDecodeVp9Features {
    fn default() -> Self {
        Self {
            s_type: PHYSICAL_DEVICE_FEATURES,
            p_next: std::ptr::null_mut(),
            video_decode_vp9: vk::FALSE,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VideoDecodeVp9ProfileInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub std_profile: i32,
}

impl VideoDecodeVp9ProfileInfo {
    pub(crate) fn profile_0() -> Self {
        Self {
            s_type: PROFILE_INFO,
            p_next: std::ptr::null(),
            std_profile: PROFILE_0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VideoDecodeVp9Capabilities {
    pub s_type: vk::StructureType,
    pub p_next: *mut c_void,
    /// `StdVideoVP9Level`: 0 is level 1.0, then 1.1, 2.0, 2.1 and so on.
    pub max_level: i32,
}

impl Default for VideoDecodeVp9Capabilities {
    fn default() -> Self {
        Self {
            s_type: CAPABILITIES,
            p_next: std::ptr::null_mut(),
            max_level: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct VideoDecodeVp9PictureInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub p_std_picture_info: *const StdVideoDecodeVP9PictureInfo,
    pub reference_name_slot_indices: [i32; 3],
    pub uncompressed_header_offset: u32,
    pub compressed_header_offset: u32,
    pub tiles_offset: u32,
}

impl VideoDecodeVp9PictureInfo {
    pub(crate) fn new(std: &StdVideoDecodeVP9PictureInfo) -> Self {
        Self {
            s_type: PICTURE_INFO,
            p_next: std::ptr::null(),
            p_std_picture_info: std,
            reference_name_slot_indices: [-1; 3],
            uncompressed_header_offset: 0,
            compressed_header_offset: 0,
            tiles_offset: 0,
        }
    }
}

/// `StdVideoVP9ColorConfig`; `flags` bit 0 is `color_range`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StdVideoVP9ColorConfig {
    pub flags: u32,
    pub BitDepth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub reserved1: u8,
    pub color_space: i32,
}

/// `StdVideoVP9LoopFilter`; `flags` bit 0 is `loop_filter_delta_enabled`, bit 1
/// `loop_filter_delta_update`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StdVideoVP9LoopFilter {
    pub flags: u32,
    pub loop_filter_level: u8,
    pub loop_filter_sharpness: u8,
    pub update_ref_delta: u8,
    pub loop_filter_ref_deltas: [i8; 4],
    pub update_mode_delta: u8,
    pub loop_filter_mode_deltas: [i8; 2],
}

/// `StdVideoVP9Segmentation`; `flags` bits 0 to 3 are `segmentation_update_map`,
/// `segmentation_temporal_update`, `segmentation_update_data` and
/// `segmentation_abs_or_delta_update`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct StdVideoVP9Segmentation {
    pub flags: u32,
    pub segmentation_tree_probs: [u8; 7],
    pub segmentation_pred_prob: [u8; 3],
    pub FeatureEnabled: [u8; 8],
    pub FeatureData: [[i16; 4]; 8],
}

/// `StdVideoDecodeVP9PictureInfo`; `flags` bits 0 to 7 are `error_resilient_mode`,
/// `intra_only`, `allow_high_precision_mv`, `refresh_frame_context`,
/// `frame_parallel_decoding_mode`, `segmentation_enabled`, `show_frame` and `UsePrevFrameMvs`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct StdVideoDecodeVP9PictureInfo {
    pub flags: u32,
    pub profile: i32,
    pub frame_type: i32,
    pub frame_context_idx: u8,
    pub reset_frame_context: u8,
    pub refresh_frame_flags: u8,
    pub ref_frame_sign_bias_mask: u8,
    pub interpolation_filter: i32,
    pub base_q_idx: u8,
    pub delta_q_y_dc: i8,
    pub delta_q_uv_dc: i8,
    pub delta_q_uv_ac: i8,
    pub tile_cols_log2: u8,
    pub tile_rows_log2: u8,
    pub reserved1: [u16; 3],
    pub pColorConfig: *const StdVideoVP9ColorConfig,
    pub pLoopFilter: *const StdVideoVP9LoopFilter,
    pub pSegmentation: *const StdVideoVP9Segmentation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// Offsets and sizes as a C compiler lays out the Khronos headers on 64-bit targets.
    #[test]
    fn layouts_match_the_c_headers() {
        assert_eq!(size_of::<StdVideoVP9ColorConfig>(), 12);
        assert_eq!(offset_of!(StdVideoVP9ColorConfig, color_space), 8);
        assert_eq!(size_of::<StdVideoVP9LoopFilter>(), 16);
        assert_eq!(offset_of!(StdVideoVP9LoopFilter, loop_filter_ref_deltas), 7);
        assert_eq!(offset_of!(StdVideoVP9LoopFilter, update_mode_delta), 11);
        assert_eq!(
            offset_of!(StdVideoVP9LoopFilter, loop_filter_mode_deltas),
            12
        );
        assert_eq!(
            offset_of!(StdVideoVP9Segmentation, segmentation_pred_prob),
            11
        );
        assert_eq!(offset_of!(StdVideoVP9Segmentation, FeatureEnabled), 14);
        assert_eq!(offset_of!(StdVideoVP9Segmentation, FeatureData), 22);
        assert_eq!(size_of::<StdVideoVP9Segmentation>(), 88);
        assert_eq!(
            offset_of!(StdVideoDecodeVP9PictureInfo, interpolation_filter),
            16
        );
        assert_eq!(offset_of!(StdVideoDecodeVP9PictureInfo, base_q_idx), 20);
        assert_eq!(offset_of!(StdVideoDecodeVP9PictureInfo, tile_rows_log2), 25);
        assert_eq!(offset_of!(StdVideoDecodeVP9PictureInfo, pColorConfig), 32);
        assert_eq!(size_of::<StdVideoDecodeVP9PictureInfo>(), 56);
        assert_eq!(
            offset_of!(VideoDecodeVp9PictureInfo, reference_name_slot_indices),
            24
        );
        assert_eq!(offset_of!(VideoDecodeVp9PictureInfo, tiles_offset), 44);
        assert_eq!(size_of::<VideoDecodeVp9PictureInfo>(), 48);
    }
}
