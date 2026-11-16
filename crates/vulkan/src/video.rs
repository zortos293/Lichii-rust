//! The host GPU's video decode queue (`VK_KHR_video_decode_queue`) and what it decodes.
//!
//! Switch 2 movies (`nx2:vdecsvc`) are AV1, VP9 or H.264. A host GPU with a video decode queue
//! decodes them on its fixed-function video engine, so the CPU does no decoding work and the
//! pictures can stay in GPU memory. This module finds that queue and reads the AV1 decode
//! capabilities; decoding itself comes on top of it.

use std::ffi::CStr;

use ash::khr;
use ash::vk;
use parking_lot::Mutex;

/// Device extensions AV1 decoding needs, all enabled together.
pub(crate) const AV1_EXTENSIONS: [&CStr; 3] = [
    khr::video_queue::NAME,
    khr::video_decode_queue::NAME,
    khr::video_decode_av1::NAME,
];

/// The video decode queue and the AV1 decoding it offers.
pub struct VideoQueue {
    /// The queue family the video queue belongs to.
    pub family: u32,
    queue: Mutex<vk::Queue>,
    /// What the queue decodes in AV1 Main, 4:2:0, 8 bits.
    pub av1: Av1Decode,
}

/// AV1 Main 4:2:0 8-bit decoding on the video queue (`vkGetPhysicalDeviceVideoCapabilitiesKHR`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1Decode {
    /// The highest AV1 level, as `StdVideoAV1Level` (0 is level 2.0, each step a minor level).
    pub max_level: u32,
    /// The smallest coded picture size, in pixels.
    pub min_extent: vk::Extent2D,
    /// The largest coded picture size, in pixels.
    pub max_extent: vk::Extent2D,
    /// Granularity of picture sizes and offsets in decode output and DPB images, in pixels.
    pub picture_granularity: vk::Extent2D,
    /// The most DPB slots (reference pictures plus the one being decoded) a session may have.
    pub max_dpb_slots: u32,
    /// The most reference pictures one decode may read.
    pub max_active_references: u32,
    /// Alignment of a picture's offset in the bitstream buffer, in bytes.
    pub bitstream_offset_alignment: u64,
    /// Alignment of a picture's size in the bitstream buffer, in bytes.
    pub bitstream_size_alignment: u64,
    /// Decode output and the reference picture are the same image (one image per picture).
    pub dpb_and_output_coincide: bool,
    /// Decode output and the reference picture are separate images.
    pub dpb_and_output_distinct: bool,
    /// Formats decode output images may have.
    pub output_formats: Vec<vk::Format>,
    /// Formats DPB (reference picture) images may have.
    pub dpb_formats: Vec<vk::Format>,
    /// The queue applies AV1 film grain itself.
    pub film_grain: bool,
}

impl Av1Decode {
    /// The highest level as "major.minor".
    pub fn level_name(&self) -> String {
        format!("{}.{}", 2 + self.max_level / 4, self.max_level % 4)
    }
}

/// The first queue family other than `universal` that decodes AV1, from each family's flags and
/// video codec operations.
pub(crate) fn decode_family(
    families: &[(vk::QueueFlags, vk::VideoCodecOperationFlagsKHR)],
    universal: u32,
) -> Option<u32> {
    families
        .iter()
        .enumerate()
        .find(|&(index, (flags, operations))| {
            index as u32 != universal
                && flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR)
                && operations.contains(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
        })
        .map(|(index, _)| index as u32)
}

/// Each queue family's flags and video codec operations. Call only when the device reports
/// `VK_KHR_video_queue`.
pub(crate) fn queue_families(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Vec<(vk::QueueFlags, vk::VideoCodecOperationFlagsKHR)> {
    // SAFETY: `physical` came from `instance`, which reports the video queue extension for it.
    unsafe {
        let count = instance.get_physical_device_queue_family_properties2_len(physical);
        let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
        let mut families: Vec<vk::QueueFamilyProperties2<'_>> = video
            .iter_mut()
            .map(|v| vk::QueueFamilyProperties2::default().push_next(v))
            .collect();
        instance.get_physical_device_queue_family_properties2(physical, &mut families);
        let flags: Vec<vk::QueueFlags> = families
            .iter()
            .map(|f| f.queue_family_properties.queue_flags)
            .collect();
        drop(families);
        flags
            .into_iter()
            .zip(video.iter().map(|v| v.video_codec_operations))
            .collect()
    }
}

/// The AV1 decode capabilities of `physical`, or `None` when it cannot decode AV1 Main 4:2:0
/// 8-bit.
pub(crate) fn av1_capabilities(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Option<Av1Decode> {
    let video = khr::video_queue::Instance::new(entry, instance);
    let fp = video.fp();
    let query = |film_grain: bool| {
        let mut av1_profile = vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
            .film_grain_support(film_grain);
        let profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut av1_profile);
        let mut av1 = vk::VideoDecodeAV1CapabilitiesKHR::default();
        let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
        let mut caps = vk::VideoCapabilitiesKHR::default()
            .push_next(&mut decode)
            .push_next(&mut av1);
        // SAFETY: valid profile and output chain for a device that reports AV1 decoding.
        let result = unsafe {
            (fp.get_physical_device_video_capabilities_khr)(physical, &profile, &mut caps)
        };
        if result != vk::Result::SUCCESS {
            return None;
        }
        let formats = |usage: vk::ImageUsageFlags| {
            let profiles = [profile];
            let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
            let info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
                .image_usage(usage)
                .push_next(&mut list);
            let mut count = 0;
            // SAFETY: the two-call idiom with a valid format info; the output holds `count`.
            unsafe {
                let first = (fp.get_physical_device_video_format_properties_khr)(
                    physical,
                    &info,
                    &mut count,
                    std::ptr::null_mut(),
                );
                if first != vk::Result::SUCCESS || count == 0 {
                    return Vec::new();
                }
                let mut out = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
                let second = (fp.get_physical_device_video_format_properties_khr)(
                    physical,
                    &info,
                    &mut count,
                    out.as_mut_ptr(),
                );
                if second != vk::Result::SUCCESS && second != vk::Result::INCOMPLETE {
                    return Vec::new();
                }
                out.truncate(count as usize);
                out.iter().map(|f| f.format).collect()
            }
        };
        let (min_extent, max_extent, picture_granularity) = (
            caps.min_coded_extent,
            caps.max_coded_extent,
            caps.picture_access_granularity,
        );
        let (max_dpb_slots, max_active_references) =
            (caps.max_dpb_slots, caps.max_active_reference_pictures);
        let (bitstream_offset_alignment, bitstream_size_alignment) = (
            caps.min_bitstream_buffer_offset_alignment,
            caps.min_bitstream_buffer_size_alignment,
        );
        let flags = decode.flags;
        Some(Av1Decode {
            max_level: av1.max_level,
            min_extent,
            max_extent,
            picture_granularity,
            max_dpb_slots,
            max_active_references,
            bitstream_offset_alignment,
            bitstream_size_alignment,
            dpb_and_output_coincide: flags
                .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE),
            dpb_and_output_distinct: flags
                .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT),
            output_formats: formats(vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR),
            dpb_formats: formats(vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR),
            film_grain,
        })
    };
    query(true).or_else(|| query(false))
}

impl VideoQueue {
    pub(crate) fn new(family: u32, queue: vk::Queue, av1: Av1Decode) -> Self {
        Self {
            family,
            queue: Mutex::new(queue),
            av1,
        }
    }

    /// The queue, locked for a submission: video queue submissions are externally synchronised.
    pub fn queue(&self) -> parking_lot::MutexGuard<'_, vk::Queue> {
        self.queue.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decode_family_is_the_first_that_decodes_av1() {
        let none = vk::VideoCodecOperationFlagsKHR::empty();
        let families = [
            (vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, none),
            (
                vk::QueueFlags::VIDEO_DECODE_KHR,
                vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            ),
            (
                vk::QueueFlags::VIDEO_ENCODE_KHR,
                vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
            ),
            (
                vk::QueueFlags::VIDEO_DECODE_KHR | vk::QueueFlags::TRANSFER,
                vk::VideoCodecOperationFlagsKHR::DECODE_H264
                    | vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
            ),
        ];
        assert_eq!(decode_family(&families, 0), Some(3));
        assert_eq!(decode_family(&families[..3], 0), None);
    }

    #[test]
    fn the_universal_family_is_never_the_decode_family() {
        let both =
            vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE | vk::QueueFlags::VIDEO_DECODE_KHR;
        let av1 = vk::VideoCodecOperationFlagsKHR::DECODE_AV1;
        let families = [(both, av1), (vk::QueueFlags::VIDEO_DECODE_KHR, av1)];
        assert_eq!(decode_family(&families, 0), Some(1));
        assert_eq!(decode_family(&families[..1], 0), None);
    }

    #[test]
    fn a_device_decode_queue_reports_usable_av1_capabilities() {
        use crate::device::Device;
        use crate::instance::{Instance, SurfaceSupport};
        use lichii_common::settings::GraphicsSettings;
        use std::sync::Arc;
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let device = match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let Some(video) = &device.video else {
            eprintln!("skipping: no AV1 decode queue");
            return;
        };
        let av1 = &video.av1;
        assert_ne!(video.family, device.queue_family());
        assert!(
            av1.max_extent.width >= av1.min_extent.width.max(1),
            "{av1:?}"
        );
        assert!(
            av1.max_extent.height >= av1.min_extent.height.max(1),
            "{av1:?}"
        );
        assert!(
            av1.max_dpb_slots > 0 && av1.max_active_references > 0,
            "{av1:?}"
        );
        assert!(
            av1.dpb_and_output_coincide || av1.dpb_and_output_distinct,
            "{av1:?}"
        );
        assert!(
            !av1.output_formats.is_empty() && !av1.dpb_formats.is_empty(),
            "{av1:?}"
        );
        drop(video.queue());
    }

    #[test]
    fn levels_read_as_major_dot_minor() {
        let mut av1 = Av1Decode {
            max_level: 0,
            min_extent: vk::Extent2D::default(),
            max_extent: vk::Extent2D::default(),
            picture_granularity: vk::Extent2D::default(),
            max_dpb_slots: 0,
            max_active_references: 0,
            bitstream_offset_alignment: 1,
            bitstream_size_alignment: 1,
            dpb_and_output_coincide: false,
            dpb_and_output_distinct: false,
            output_formats: Vec::new(),
            dpb_formats: Vec::new(),
            film_grain: false,
        };
        assert_eq!(av1.level_name(), "2.0");
        av1.max_level = 13;
        assert_eq!(av1.level_name(), "5.1");
    }
}
