//! AV1 decoding on the host GPU's video queue (`VK_KHR_video_decode_av1`).
//!
//! `lichii_video::av1` reads the headers; this records one decode per frame on the video queue,
//! into a reference picture (DPB) array with a layer per slot and an output array beside it, then
//! copies each shown picture to CPU memory as NV12 on the universal queue. The output array is
//! shared by both queue families, so no ownership transfers are needed. Each picture is decoded and
//! copied before the next one starts: a movie frame costs the CPU only the header reading and the
//! copy out.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::{Arc, Weak};

use ash::vk::native as sv;
use ash::{khr, vk};
use lichii_video::av1::frame::{FrameHeader, FrameType, NUM_REF_FRAMES, REFS_PER_FRAME, RefFrame};
use lichii_video::av1::{Av1Stream, ParsedFrame, SequenceHeader};
use lichii_video::{Codec, Color, GpuStream, GpuVideo, Matrix, OwnedPicture, VideoError};

use crate::device::Device;

const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
/// DPB slots used at most: the eight reference slots and the picture being decoded.
const MAX_SLOTS: u32 = 9;
const STD_AV1_DECODE: &std::ffi::CStr = c"VK_STD_vulkan_video_codec_av1_decode";

/// The host GPU's AV1 decoder, as `lichii_video` sees it.
struct VulkanVideo {
    device: Weak<Device>,
}

impl GpuVideo for VulkanVideo {
    fn open(&self, codec: Codec) -> Option<Box<dyn GpuStream>> {
        if codec != Codec::Av1 {
            return None;
        }
        let device = self.device.upgrade()?;
        match Av1Decoder::new(device) {
            Ok(decoder) => Some(Box::new(decoder)),
            Err(error) => {
                tracing::warn!("Vulkan video: opening an AV1 decoder failed: {error:?}");
                None
            }
        }
    }
}

/// Makes `device`'s video queue decode AV1 for `lichii_video` decoders opened from now on.
pub(crate) fn register(device: &Arc<Device>) {
    if device.video.is_some() {
        lichii_video::set_gpu_video(Some(Arc::new(VulkanVideo {
            device: Arc::downgrade(device),
        })));
    }
}

/// Declares an AV1 Main 4:2:0 8-bit decode profile `$profile` (with its AV1 part `$av1`).
macro_rules! av1_profile {
    ($av1:ident, $profile:ident, $grain:expr) => {
        let mut $av1 = vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(sv::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
            .film_grain_support($grain);
        let $profile = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_AV1)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut $av1);
    };
}

/// An all-zero std flags structure: plain bitfields, every bit cleared.
fn zeroed<T: Copy>() -> T {
    // SAFETY: only used for the `StdVideo*Flags` bitfield structs, for which zero is valid.
    unsafe { std::mem::zeroed() }
}

fn vk_error(result: vk::Result) -> VideoError {
    VideoError::Gpu(result.as_raw())
}

fn check(result: vk::Result) -> Result<(), VideoError> {
    match result {
        vk::Result::SUCCESS => Ok(()),
        other => Err(vk_error(other)),
    }
}

/// An image with its memory and a view of all its layers.
struct Image {
    raw: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// A persistently mapped host-visible buffer.
struct HostBuffer {
    raw: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    ptr: *mut u8,
}

struct Session {
    raw: vk::VideoSessionKHR,
    memory: Vec<vk::DeviceMemory>,
    extent: vk::Extent2D,
    dpb: Image,
    out: Image,
    /// The session still needs its reset before the first decode.
    fresh: bool,
    /// The DPB array left `UNDEFINED`.
    dpb_ready: bool,
    out_layout: [vk::ImageLayout; MAX_SLOTS as usize],
}

/// The Vulkan side of one AV1 stream.
struct Gpu {
    device: Arc<Device>,
    video_queue: khr::video_queue::Device,
    video_decode: khr::video_decode_queue::Device,
    family: u32,
    film_grain: bool,
    max_references: u32,
    granularity: vk::Extent2D,
    max_extent: vk::Extent2D,
    offset_alignment: u64,
    size_alignment: u64,
    std_version: u32,
    video_pool: vk::CommandPool,
    video_cmd: vk::CommandBuffer,
    video_fence: vk::Fence,
    main_pool: vk::CommandPool,
    main_cmd: vk::CommandBuffer,
    main_fence: vk::Fence,
    decoded: vk::Semaphore,
    session: Option<Session>,
    /// Session parameters and the sequence header they hold.
    params: Option<(vk::VideoSessionParametersKHR, SequenceHeader)>,
    bitstream: Option<HostBuffer>,
    readback: Option<HostBuffer>,
    /// The DPB slot each reference slot's picture is in.
    slot_of_ref: [Option<u32>; NUM_REF_FRAMES],
    staging: Vec<u8>,
}

// SAFETY: the Vulkan objects and mapped pointers are owned by this stream alone and only used
// through `&mut self`, from the thread that decodes it.
unsafe impl Send for Gpu {}

/// One AV1 stream decoding on the video queue.
pub(crate) struct Av1Decoder {
    parser: Av1Stream,
    gpu: Gpu,
    ready: VecDeque<OwnedPicture>,
    spare: Vec<Vec<u8>>,
}

impl Av1Decoder {
    fn new(device: Arc<Device>) -> Result<Self, VideoError> {
        Ok(Self {
            parser: Av1Stream::default(),
            gpu: Gpu::new(device)?,
            ready: VecDeque::new(),
            spare: Vec::new(),
        })
    }
}

impl GpuStream for Av1Decoder {
    fn send(&mut self, data: &[u8], pts: i64) -> Result<(), VideoError> {
        let mut failed = None;
        let Self {
            parser,
            gpu,
            ready,
            spare,
        } = self;
        let parsed = parser.temporal_unit(data, |frame, seq, refs| {
            if failed.is_none()
                && let Err(error) = gpu.frame(frame, seq, refs, data, pts, ready, spare)
            {
                failed = Some(error);
            }
            Ok(())
        });
        if let Err(error) = parsed {
            tracing::debug!("AV1 temporal unit at pts {pts} does not parse: {error}");
            return Err(VideoError::Gpu(0));
        }
        failed.map_or(Ok(()), Err)
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

fn memory_type(
    device: &Device,
    bits: u32,
    need: vk::MemoryPropertyFlags,
    prefer: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let props = &device.memory_properties;
    let types = &props.memory_types[..props.memory_type_count as usize];
    let fits = |i: usize, flags: vk::MemoryPropertyFlags| {
        bits & (1 << i) != 0 && types[i].property_flags.contains(flags)
    };
    (0..types.len())
        .find(|&i| fits(i, need | prefer))
        .or_else(|| (0..types.len()).find(|&i| fits(i, need)))
        .map(|i| i as u32)
}

fn allocate(
    device: &Device,
    requirements: vk::MemoryRequirements,
    need: vk::MemoryPropertyFlags,
    prefer: vk::MemoryPropertyFlags,
) -> Result<vk::DeviceMemory, VideoError> {
    let index = memory_type(device, requirements.memory_type_bits, need, prefer).ok_or(
        VideoError::Gpu(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw()),
    )?;
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(index);
    // SAFETY: a valid allocation on a live device.
    unsafe { device.raw.allocate_memory(&info, None) }.map_err(vk_error)
}

impl Gpu {
    fn new(device: Arc<Device>) -> Result<Self, VideoError> {
        let video = device
            .video
            .as_ref()
            .ok_or(VideoError::NoDecoder(Codec::Av1))?;
        let raw = &device.raw;
        let video_queue = khr::video_queue::Device::new(&device.instance.raw, raw);
        let video_decode = khr::video_decode_queue::Device::new(&device.instance.raw, raw);
        let av1 = &video.av1;
        let pool = |family: u32| {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            // SAFETY: valid create info on a live device.
            let pool = unsafe { raw.create_command_pool(&info, None) }.map_err(vk_error)?;
            let alloc = vk::CommandBufferAllocateInfo::default()
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            // SAFETY: the pool was just created on this device.
            let cmd = unsafe { raw.allocate_command_buffers(&alloc) }.map_err(vk_error)?[0];
            Ok::<_, VideoError>((pool, cmd))
        };
        let (video_pool, video_cmd) = pool(video.family)?;
        let (main_pool, main_cmd) = pool(device.queue_family())?;
        // SAFETY: plain object creation on a live device.
        let (video_fence, main_fence, decoded) = unsafe {
            (
                raw.create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(vk_error)?,
                raw.create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(vk_error)?,
                raw.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                    .map_err(vk_error)?,
            )
        };
        Ok(Self {
            video_queue,
            video_decode,
            family: video.family,
            film_grain: av1.film_grain,
            max_references: av1.max_active_references.min(REFS_PER_FRAME as u32),
            granularity: av1.picture_granularity,
            max_extent: av1.max_extent,
            offset_alignment: av1.bitstream_offset_alignment.max(1),
            size_alignment: av1.bitstream_size_alignment.max(1),
            std_version: av1.std_header_version,
            video_pool,
            video_cmd,
            video_fence,
            main_pool,
            main_cmd,
            main_fence,
            decoded,
            session: None,
            params: None,
            bitstream: None,
            readback: None,
            slot_of_ref: [None; NUM_REF_FRAMES],
            staging: Vec::new(),
            device,
        })
    }

    fn image(
        &self,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        families: &[u32],
    ) -> Result<Image, VideoError> {
        let raw = &self.device.raw;
        av1_profile!(av1, profile, self.film_grain);
        let profiles = [profile];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let info = vk::ImageCreateInfo::default()
            .push_next(&mut list)
            .image_type(vk::ImageType::TYPE_2D)
            .format(NV12)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(MAX_SLOTS)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(if families.len() > 1 {
                vk::SharingMode::CONCURRENT
            } else {
                vk::SharingMode::EXCLUSIVE
            })
            .queue_family_indices(families)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid create info with the decode profile; the memory is bound once below.
        unsafe {
            let image = raw.create_image(&info, None).map_err(vk_error)?;
            let requirements = raw.get_image_memory_requirements(image);
            let memory = allocate(
                &self.device,
                requirements,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::empty(),
            )?;
            raw.bind_image_memory(image, memory, 0).map_err(vk_error)?;
            let view_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D_ARRAY)
                .format(NV12)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: MAX_SLOTS,
                });
            let view = raw.create_image_view(&view_info, None).map_err(vk_error)?;
            Ok(Image {
                raw: image,
                memory,
                view,
            })
        }
    }

    fn host_buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        video: bool,
    ) -> Result<HostBuffer, VideoError> {
        let raw = &self.device.raw;
        av1_profile!(av1, profile, self.film_grain);
        let profiles = [profile];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let mut info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        if video {
            info = info.push_next(&mut list);
        }
        // SAFETY: valid create info; the memory is bound once and mapped whole.
        unsafe {
            let buffer = raw.create_buffer(&info, None).map_err(vk_error)?;
            let requirements = raw.get_buffer_memory_requirements(buffer);
            let memory = allocate(
                &self.device,
                requirements,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                if video {
                    vk::MemoryPropertyFlags::empty()
                } else {
                    vk::MemoryPropertyFlags::HOST_CACHED
                },
            )?;
            raw.bind_buffer_memory(buffer, memory, 0)
                .map_err(vk_error)?;
            let ptr = raw
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
                .map_err(vk_error)?
                .cast::<u8>();
            Ok(HostBuffer {
                raw: buffer,
                memory,
                size,
                ptr,
            })
        }
    }

    fn free_buffer(&self, buffer: HostBuffer) {
        // SAFETY: no submitted work uses the buffer: every submission is waited for.
        unsafe {
            self.device.raw.unmap_memory(buffer.memory);
            self.device.raw.destroy_buffer(buffer.raw, None);
            self.device.raw.free_memory(buffer.memory, None);
        }
    }

    fn free_image(&self, image: Image) {
        // SAFETY: no submitted work uses the image: every submission is waited for.
        unsafe {
            self.device.raw.destroy_image_view(image.view, None);
            self.device.raw.destroy_image(image.raw, None);
            self.device.raw.free_memory(image.memory, None);
        }
    }

    fn free_session(&mut self) {
        if let Some((params, _)) = self.params.take() {
            // SAFETY: no submitted work uses the parameters.
            unsafe {
                (self.video_queue.fp().destroy_video_session_parameters_khr)(
                    self.device.raw.handle(),
                    params,
                    std::ptr::null(),
                )
            };
        }
        if let Some(session) = self.session.take() {
            // SAFETY: no submitted work uses the session.
            unsafe {
                (self.video_queue.fp().destroy_video_session_khr)(
                    self.device.raw.handle(),
                    session.raw,
                    std::ptr::null(),
                );
                for memory in &session.memory {
                    self.device.raw.free_memory(*memory, None);
                }
            }
            self.free_image(session.dpb);
            self.free_image(session.out);
        }
    }

    /// A session for pictures up to `extent` (rounded to the access granularity).
    fn ensure_session(&mut self, seq: &SequenceHeader) -> Result<(), VideoError> {
        if seq.color.bit_depth != 8
            || seq.color.mono_chrome
            || !seq.color.subsampling_x
            || !seq.color.subsampling_y
        {
            return Err(VideoError::Gpu(0));
        }
        let align = |v: u32, g: u32| v.next_multiple_of(g.max(1));
        let extent = vk::Extent2D {
            width: align(
                u32::from(seq.max_frame_width_minus_1) + 1,
                self.granularity.width,
            ),
            height: align(
                u32::from(seq.max_frame_height_minus_1) + 1,
                self.granularity.height,
            ),
        };
        if extent.width > self.max_extent.width || extent.height > self.max_extent.height {
            return Err(VideoError::Gpu(0));
        }
        if let Some(session) = &self.session
            && session.extent.width >= extent.width
            && session.extent.height >= extent.height
        {
            return Ok(());
        }
        self.free_session();
        let raw = &self.device.raw;
        let fp = self.video_queue.fp();
        av1_profile!(av1, profile, self.film_grain);
        let std = vk::ExtensionProperties::default()
            .extension_name(STD_AV1_DECODE)
            .map_err(|_| VideoError::Gpu(0))?
            .spec_version(self.std_version);
        let info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(self.family)
            .video_profile(&profile)
            .picture_format(NV12)
            .max_coded_extent(extent)
            .reference_picture_format(NV12)
            .max_dpb_slots(MAX_SLOTS)
            .max_active_reference_pictures(self.max_references)
            .std_header_version(&std);
        let mut session = vk::VideoSessionKHR::null();
        // SAFETY: valid create info for a profile the queue reported.
        check(unsafe {
            (fp.create_video_session_khr)(raw.handle(), &info, std::ptr::null(), &mut session)
        })?;
        let mut count = 0;
        // SAFETY: the two-call idiom on the session just created.
        unsafe {
            check((fp.get_video_session_memory_requirements_khr)(
                raw.handle(),
                session,
                &mut count,
                std::ptr::null_mut(),
            ))?;
        }
        let mut requirements =
            vec![vk::VideoSessionMemoryRequirementsKHR::default(); count as usize];
        // SAFETY: `requirements` holds `count` entries.
        unsafe {
            check((fp.get_video_session_memory_requirements_khr)(
                raw.handle(),
                session,
                &mut count,
                requirements.as_mut_ptr(),
            ))?;
        }
        let mut memory = Vec::new();
        let mut binds = Vec::new();
        for r in &requirements {
            let m = allocate(
                &self.device,
                r.memory_requirements,
                vk::MemoryPropertyFlags::empty(),
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )?;
            memory.push(m);
            binds.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(r.memory_bind_index)
                    .memory(m)
                    .memory_offset(0)
                    .memory_size(r.memory_requirements.size),
            );
        }
        // SAFETY: each bind names a requirement of this session and memory of its size.
        check(unsafe {
            (fp.bind_video_session_memory_khr)(
                raw.handle(),
                session,
                binds.len() as u32,
                binds.as_ptr(),
            )
        })?;
        let dpb = self.image(
            extent,
            vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR,
            &[self.family],
        )?;
        let out = self.image(
            extent,
            vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR | vk::ImageUsageFlags::TRANSFER_SRC,
            &[self.family, self.device.queue_family()],
        )?;
        self.session = Some(Session {
            raw: session,
            memory,
            extent,
            dpb,
            out,
            fresh: true,
            dpb_ready: false,
            out_layout: [vk::ImageLayout::UNDEFINED; MAX_SLOTS as usize],
        });
        self.slot_of_ref = [None; NUM_REF_FRAMES];
        Ok(())
    }

    /// Session parameters holding `seq`, made again when the sequence header changes.
    fn ensure_params(&mut self, seq: &SequenceHeader) -> Result<(), VideoError> {
        if self.params.as_ref().is_some_and(|(_, held)| held == seq) {
            return Ok(());
        }
        let session = self.session.as_ref().expect("session made first").raw;
        let fp = self.video_queue.fp();
        if let Some((old, _)) = self.params.take() {
            // SAFETY: no submitted work uses the parameters.
            unsafe {
                (fp.destroy_video_session_parameters_khr)(
                    self.device.raw.handle(),
                    old,
                    std::ptr::null(),
                )
            };
        }
        let c = &seq.color;
        let mut color_flags = zeroed::<sv::StdVideoAV1ColorConfigFlags>();
        color_flags.set_mono_chrome(u32::from(c.mono_chrome));
        color_flags.set_color_range(u32::from(c.color_range));
        color_flags.set_separate_uv_delta_q(u32::from(c.separate_uv_delta_q));
        let color = sv::StdVideoAV1ColorConfig {
            flags: color_flags,
            BitDepth: c.bit_depth,
            subsampling_x: u8::from(c.subsampling_x),
            subsampling_y: u8::from(c.subsampling_y),
            reserved1: 0,
            color_primaries: u32::from(c.color_primaries),
            transfer_characteristics: u32::from(c.transfer_characteristics),
            matrix_coefficients: u32::from(c.matrix_coefficients),
            chroma_sample_position: u32::from(c.chroma_sample_position),
        };
        let t = seq.timing_info.unwrap_or_default();
        let mut timing_flags = zeroed::<sv::StdVideoAV1TimingInfoFlags>();
        timing_flags.set_equal_picture_interval(u32::from(t.equal_picture_interval));
        let timing = sv::StdVideoAV1TimingInfo {
            flags: timing_flags,
            num_units_in_display_tick: t.num_units_in_display_tick,
            time_scale: t.time_scale,
            num_ticks_per_picture_minus_1: t.num_ticks_per_picture_minus_1,
        };
        let mut f = zeroed::<sv::StdVideoAV1SequenceHeaderFlags>();
        f.set_still_picture(u32::from(seq.still_picture));
        f.set_reduced_still_picture_header(u32::from(seq.reduced_still_picture_header));
        f.set_use_128x128_superblock(u32::from(seq.use_128x128_superblock));
        f.set_enable_filter_intra(u32::from(seq.enable_filter_intra));
        f.set_enable_intra_edge_filter(u32::from(seq.enable_intra_edge_filter));
        f.set_enable_interintra_compound(u32::from(seq.enable_interintra_compound));
        f.set_enable_masked_compound(u32::from(seq.enable_masked_compound));
        f.set_enable_warped_motion(u32::from(seq.enable_warped_motion));
        f.set_enable_dual_filter(u32::from(seq.enable_dual_filter));
        f.set_enable_order_hint(u32::from(seq.enable_order_hint));
        f.set_enable_jnt_comp(u32::from(seq.enable_jnt_comp));
        f.set_enable_ref_frame_mvs(u32::from(seq.enable_ref_frame_mvs));
        f.set_frame_id_numbers_present_flag(u32::from(seq.frame_id_numbers_present));
        f.set_enable_superres(u32::from(seq.enable_superres));
        f.set_enable_cdef(u32::from(seq.enable_cdef));
        f.set_enable_restoration(u32::from(seq.enable_restoration));
        f.set_film_grain_params_present(u32::from(seq.film_grain_params_present));
        f.set_timing_info_present_flag(u32::from(seq.timing_info.is_some()));
        f.set_initial_display_delay_present_flag(u32::from(seq.initial_display_delay_present));
        let header = sv::StdVideoAV1SequenceHeader {
            flags: f,
            seq_profile: u32::from(seq.seq_profile),
            frame_width_bits_minus_1: seq.frame_width_bits_minus_1,
            frame_height_bits_minus_1: seq.frame_height_bits_minus_1,
            max_frame_width_minus_1: seq.max_frame_width_minus_1,
            max_frame_height_minus_1: seq.max_frame_height_minus_1,
            delta_frame_id_length_minus_2: seq.delta_frame_id_length_minus_2,
            additional_frame_id_length_minus_1: seq.additional_frame_id_length_minus_1,
            order_hint_bits_minus_1: seq.order_hint_bits.saturating_sub(1),
            seq_force_integer_mv: seq.seq_force_integer_mv,
            seq_force_screen_content_tools: seq.seq_force_screen_content_tools,
            reserved1: [0; 5],
            pColorConfig: &color,
            pTimingInfo: &timing,
        };
        let mut av1 = vk::VideoDecodeAV1SessionParametersCreateInfoKHR::default()
            .std_sequence_header(&header);
        let info = vk::VideoSessionParametersCreateInfoKHR::default()
            .video_session(session)
            .push_next(&mut av1);
        let mut params = vk::VideoSessionParametersKHR::null();
        // SAFETY: valid parameters for the live session; the std structs outlive the call.
        check(unsafe {
            (fp.create_video_session_parameters_khr)(
                self.device.raw.handle(),
                &info,
                std::ptr::null(),
                &mut params,
            )
        })?;
        self.params = Some((params, seq.clone()));
        Ok(())
    }

    /// Decodes (or shows) one frame and queues its picture when it is shown.
    #[allow(clippy::too_many_arguments)]
    fn frame(
        &mut self,
        frame: &ParsedFrame,
        seq: &SequenceHeader,
        refs: &lichii_video::av1::References,
        data: &[u8],
        pts: i64,
        ready: &mut VecDeque<OwnedPicture>,
        spare: &mut Vec<Vec<u8>>,
    ) -> Result<(), VideoError> {
        let h = &frame.header;
        if h.show_existing_frame {
            let index = usize::from(h.frame_to_show_map_idx);
            let Some(slot) = self.slot_of_ref[index] else {
                return Ok(());
            };
            let shown = &refs.slots[index];
            let extent = vk::Extent2D {
                width: shown.upscaled_width,
                height: shown.frame_height,
            };
            let picture = self.copy_out(slot, extent, None, seq, pts, spare)?;
            ready.push_back(picture);
            if h.frame_type == FrameType::Key {
                self.slot_of_ref = [Some(slot); NUM_REF_FRAMES];
            }
            return Ok(());
        }
        self.ensure_session(seq)?;
        self.ensure_params(seq)?;
        let used: Vec<u32> = self.slot_of_ref.iter().flatten().copied().collect();
        let setup = (0..MAX_SLOTS)
            .find(|s| !used.contains(s))
            .expect("nine slots hold eight references and the new picture");
        let extent = vk::Extent2D {
            width: h.upscaled_width,
            height: h.frame_height,
        };
        self.decode(frame, refs, data, setup, extent)?;
        if h.show_frame {
            let picture = self.copy_out(setup, extent, Some(()), seq, pts, spare)?;
            ready.push_back(picture);
        } else {
            self.copy_out(setup, extent, Some(()), seq, pts, spare)
                .map(|p| spare.push(p.data))?;
        }
        for (i, slot) in self.slot_of_ref.iter_mut().enumerate() {
            if h.refresh_frame_flags & (1 << i) != 0 {
                *slot = Some(setup);
            }
        }
        Ok(())
    }

    fn ensure_bitstream(&mut self, size: u64) -> Result<(), VideoError> {
        if self.bitstream.as_ref().is_some_and(|b| b.size >= size) {
            return Ok(());
        }
        if let Some(old) = self.bitstream.take() {
            self.free_buffer(old);
        }
        let size = size.next_power_of_two().max(1 << 20);
        self.bitstream =
            Some(self.host_buffer(size, vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR, true)?);
        Ok(())
    }

    fn decode(
        &mut self,
        frame: &ParsedFrame,
        refs: &lichii_video::av1::References,
        data: &[u8],
        setup: u32,
        extent: vk::Extent2D,
    ) -> Result<(), VideoError> {
        let h = &frame.header;
        let mut offsets = Vec::with_capacity(frame.tiles.len());
        let mut sizes = Vec::with_capacity(frame.tiles.len());
        self.staging.clear();
        for tile in &frame.tiles {
            let at = self
                .staging
                .len()
                .next_multiple_of(self.offset_alignment as usize);
            self.staging.resize(at, 0);
            offsets.push(at as u32);
            sizes.push(tile.size as u32);
            self.staging
                .extend_from_slice(&data[tile.offset..tile.offset + tile.size]);
        }
        let range = (self.staging.len() as u64).next_multiple_of(self.size_alignment);
        self.ensure_bitstream(range)?;
        let bitstream = self.bitstream.as_ref().expect("just made");
        // SAFETY: the buffer is mapped, holds at least `range` bytes and no work reads it now.
        unsafe {
            std::ptr::copy_nonoverlapping(self.staging.as_ptr(), bitstream.ptr, self.staging.len());
            std::ptr::write_bytes(
                bitstream.ptr.add(self.staging.len()),
                0,
                (range as usize) - self.staging.len(),
            );
        }

        let std = StdPicture::new(h, self.film_grain);
        let mut info = std.info;
        let tile = sv::StdVideoAV1TileInfo {
            pMiColStarts: std.mi_col.as_ptr(),
            pMiRowStarts: std.mi_row.as_ptr(),
            pWidthInSbsMinus1: std.width_sbs.as_ptr(),
            pHeightInSbsMinus1: std.height_sbs.as_ptr(),
            ..std.tile
        };
        info.pTileInfo = &tile;
        info.pQuantization = &std.quantization;
        info.pSegmentation = &std.segmentation;
        info.pLoopFilter = &std.loop_filter;
        info.pCDEF = &std.cdef;
        info.pLoopRestoration = &std.restoration;
        info.pGlobalMotion = &std.global_motion;
        info.pFilmGrain = if std.apply_grain {
            &std.film_grain
        } else {
            std::ptr::null()
        };

        let session = self.session.as_ref().expect("session made first");
        let resource = |slot: u32, extent: vk::Extent2D, view: vk::ImageView| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D::default())
                .coded_extent(extent)
                .base_array_layer(slot)
                .image_view_binding(view)
        };
        let mut name_slots = [-1i32; REFS_PER_FRAME];
        let mut ref_slots: Vec<u32> = Vec::new();
        let mut ref_sources: Vec<usize> = Vec::new();
        if !h.frame_is_intra() {
            for (i, name) in name_slots.iter_mut().enumerate() {
                let index = usize::from(h.ref_frame_idx[i]);
                if let Some(slot) = self.slot_of_ref[index] {
                    *name = slot as i32;
                    if !ref_slots.contains(&slot) {
                        ref_slots.push(slot);
                        ref_sources.push(index);
                    }
                }
            }
        }
        let ref_std: Vec<sv::StdVideoDecodeAV1ReferenceInfo> = ref_sources
            .iter()
            .map(|&index| reference_info(&refs.slots[index]))
            .collect();
        let setup_std = current_reference_info(h);
        let mut ref_dpb: Vec<vk::VideoDecodeAV1DpbSlotInfoKHR> = ref_std
            .iter()
            .map(|r| vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(r))
            .collect();
        let ref_resources: Vec<vk::VideoPictureResourceInfoKHR> = ref_sources
            .iter()
            .zip(&ref_slots)
            .map(|(&index, &slot)| {
                let r = &refs.slots[index];
                resource(
                    slot,
                    vk::Extent2D {
                        width: r.upscaled_width,
                        height: r.frame_height,
                    },
                    session.dpb.view,
                )
            })
            .collect();
        let mut slot_infos: Vec<vk::VideoReferenceSlotInfoKHR> = Vec::new();
        for i in 0..ref_slots.len() {
            let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(ref_slots[i] as i32)
                .picture_resource(&ref_resources[i]);
            slot.p_next =
                (&mut ref_dpb[i] as *mut vk::VideoDecodeAV1DpbSlotInfoKHR).cast::<c_void>();
            slot_infos.push(slot);
        }
        let setup_resource = resource(setup, extent, session.dpb.view);
        let mut setup_dpb =
            vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(&setup_std);
        let mut setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup as i32)
            .picture_resource(&setup_resource);
        setup_slot.p_next =
            (&mut setup_dpb as *mut vk::VideoDecodeAV1DpbSlotInfoKHR).cast::<c_void>();
        let mut begin_slots = slot_infos.clone();
        let mut pending = setup_slot;
        pending.slot_index = -1;
        begin_slots.push(pending);

        let mut av1_info = vk::VideoDecodeAV1PictureInfoKHR::default()
            .std_picture_info(&info)
            .reference_name_slot_indices(name_slots)
            .frame_header_offset(0)
            .tile_offsets(&offsets)
            .tile_sizes(&sizes);
        let out_resource = resource(setup, extent, session.out.view);
        let decode = vk::VideoDecodeInfoKHR::default()
            .push_next(&mut av1_info)
            .src_buffer(bitstream.raw)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(out_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&slot_infos);
        let params = self.params.as_ref().expect("parameters made first").0;
        let begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(session.raw)
            .video_session_parameters(params)
            .reference_slots(&begin_slots);

        let raw = &self.device.raw;
        let cmd = self.video_cmd;
        let layers = |base: u32, count: u32| vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: base,
            layer_count: count,
        };
        let decode_access =
            vk::AccessFlags2::VIDEO_DECODE_READ_KHR | vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR;
        let barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .src_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(decode_access)
                .old_layout(if session.dpb_ready {
                    vk::ImageLayout::VIDEO_DECODE_DPB_KHR
                } else {
                    vk::ImageLayout::UNDEFINED
                })
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DPB_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(session.dpb.raw)
                .subresource_range(layers(0, MAX_SLOTS)),
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                .src_access_mask(vk::AccessFlags2::NONE)
                .dst_stage_mask(vk::PipelineStageFlags2::VIDEO_DECODE_KHR)
                .dst_access_mask(vk::AccessFlags2::VIDEO_DECODE_WRITE_KHR)
                .old_layout(session.out_layout[setup as usize])
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DST_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(session.out.raw)
                .subresource_range(layers(setup, 1)),
        ];
        let fresh = session.fresh;
        let vq = self.video_queue.fp();
        let vd = self.video_decode.fp();
        // SAFETY: `cmd` belongs to the video queue's pool and is not pending (every submission
        // is waited for); every struct the commands point at lives until the call returns.
        unsafe {
            raw.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_error)?;
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&barriers),
            );
            (vq.cmd_begin_video_coding_khr)(cmd, &begin);
            if fresh {
                let control = vk::VideoCodingControlInfoKHR::default()
                    .flags(vk::VideoCodingControlFlagsKHR::RESET);
                (vq.cmd_control_video_coding_khr)(cmd, &control);
            }
            (vd.cmd_decode_video_khr)(cmd, &decode);
            (vq.cmd_end_video_coding_khr)(cmd, &vk::VideoEndCodingInfoKHR::default());
            raw.end_command_buffer(cmd).map_err(vk_error)?;
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let signals = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.decoded)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let submit = vk::SubmitInfo2::default()
                .command_buffer_infos(&cmds)
                .signal_semaphore_infos(&signals);
            let video = self.device.video.as_ref().expect("checked at open");
            let queue = video.queue();
            raw.queue_submit2(*queue, &[submit], self.video_fence)
                .map_err(vk_error)?;
        }
        let session = self.session.as_mut().expect("session made first");
        session.fresh = false;
        session.dpb_ready = true;
        session.out_layout[setup as usize] = vk::ImageLayout::VIDEO_DECODE_DST_KHR;
        Ok(())
    }

    /// Copies output layer `slot` to CPU memory on the universal queue. With `after_decode`, the
    /// copy waits for the decode just submitted.
    fn copy_out(
        &mut self,
        slot: u32,
        extent: vk::Extent2D,
        after_decode: Option<()>,
        seq: &SequenceHeader,
        pts: i64,
        spare: &mut Vec<Vec<u8>>,
    ) -> Result<OwnedPicture, VideoError> {
        let size = OwnedPicture::size(extent.width, extent.height) as u64;
        if self.readback.as_ref().is_none_or(|b| b.size < size) {
            if let Some(old) = self.readback.take() {
                self.free_buffer(old);
            }
            self.readback =
                Some(self.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST, false)?);
        }
        let Some(session) = self.session.as_ref() else {
            return Err(VideoError::Gpu(0));
        };
        let readback = self.readback.as_ref().expect("just made");
        let raw = &self.device.raw;
        let cmd = self.main_cmd;
        let stride = extent.width.next_multiple_of(2);
        let layer = |aspect| vk::ImageSubresourceLayers {
            aspect_mask: aspect,
            mip_level: 0,
            base_array_layer: slot,
            layer_count: 1,
        };
        let regions = [
            vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: stride,
                buffer_image_height: extent.height,
                image_subresource: layer(vk::ImageAspectFlags::PLANE_0),
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
            },
            vk::BufferImageCopy {
                buffer_offset: u64::from(stride) * u64::from(extent.height),
                buffer_row_length: stride / 2,
                buffer_image_height: extent.height.div_ceil(2),
                image_subresource: layer(vk::ImageAspectFlags::PLANE_1),
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D {
                    width: extent.width.div_ceil(2),
                    height: extent.height.div_ceil(2),
                    depth: 1,
                },
            },
        ];
        let to_transfer = [vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COPY)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
            .old_layout(session.out_layout[slot as usize])
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(session.out.raw)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: slot,
                layer_count: 1,
            })];
        let to_host = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COPY)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::HOST)
            .dst_access_mask(vk::AccessFlags2::HOST_READ)];
        // SAFETY: `cmd` belongs to the universal queue's pool and is not pending; the image layer
        // was written by a decode this submission waits for (or an earlier, completed one), and
        // the buffer holds the whole NV12 picture.
        unsafe {
            raw.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vk_error)?;
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&to_transfer),
            );
            raw.cmd_copy_image_to_buffer(
                cmd,
                session.out.raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback.raw,
                &regions,
            );
            raw.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().memory_barriers(&to_host),
            );
            raw.end_command_buffer(cmd).map_err(vk_error)?;
            let cmds = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
            let waits = [vk::SemaphoreSubmitInfo::default()
                .semaphore(self.decoded)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
            let mut submit = vk::SubmitInfo2::default().command_buffer_infos(&cmds);
            if after_decode.is_some() {
                submit = submit.wait_semaphore_infos(&waits);
            }
            self.device
                .submit(&[submit], self.main_fence)
                .map_err(|_| VideoError::Gpu(vk::Result::ERROR_DEVICE_LOST.as_raw()))?;
            let mut fences = vec![self.main_fence];
            if after_decode.is_some() {
                fences.push(self.video_fence);
            }
            raw.wait_for_fences(&fences, true, u64::MAX)
                .map_err(vk_error)?;
            raw.reset_fences(&fences).map_err(vk_error)?;
        }
        if let Some(session) = self.session.as_mut() {
            session.out_layout[slot as usize] = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        }
        let mut data = spare.pop().unwrap_or_default();
        data.resize(size as usize, 0);
        // SAFETY: the copy above completed and the buffer is coherent host memory of `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(readback.ptr, data.as_mut_ptr(), size as usize) };
        let c = &seq.color;
        Ok(OwnedPicture {
            width: extent.width,
            height: extent.height,
            pts,
            data,
            color: Color {
                matrix: match c.matrix_coefficients {
                    1 => Some(Matrix::Bt709),
                    5 | 6 => Some(Matrix::Bt601),
                    9 | 10 => Some(Matrix::Bt2020),
                    _ => None,
                },
                full_range: c.color_range,
            },
        })
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.free_session();
        if let Some(buffer) = self.bitstream.take() {
            self.free_buffer(buffer);
        }
        if let Some(buffer) = self.readback.take() {
            self.free_buffer(buffer);
        }
        // SAFETY: nothing is pending: every submission was waited for.
        unsafe {
            let raw = &self.device.raw;
            raw.destroy_fence(self.video_fence, None);
            raw.destroy_fence(self.main_fence, None);
            raw.destroy_semaphore(self.decoded, None);
            raw.destroy_command_pool(self.video_pool, None);
            raw.destroy_command_pool(self.main_pool, None);
        }
    }
}

fn reference_info(slot: &RefFrame) -> sv::StdVideoDecodeAV1ReferenceInfo {
    let mut flags = zeroed::<sv::StdVideoDecodeAV1ReferenceInfoFlags>();
    flags.set_disable_frame_end_update_cdf(u32::from(slot.disable_frame_end_update_cdf));
    flags.set_segmentation_enabled(u32::from(slot.segmentation_enabled));
    sv::StdVideoDecodeAV1ReferenceInfo {
        flags,
        frame_type: slot.frame_type as u8,
        RefFrameSignBias: slot.ref_frame_sign_bias,
        OrderHint: slot.order_hint as u8,
        SavedOrderHints: slot.saved_order_hints.map(|h| h as u8),
    }
}

fn current_reference_info(h: &FrameHeader) -> sv::StdVideoDecodeAV1ReferenceInfo {
    let mut flags = zeroed::<sv::StdVideoDecodeAV1ReferenceInfoFlags>();
    flags.set_disable_frame_end_update_cdf(u32::from(h.disable_frame_end_update_cdf));
    flags.set_segmentation_enabled(u32::from(h.segmentation.enabled));
    sv::StdVideoDecodeAV1ReferenceInfo {
        flags,
        frame_type: h.frame_type as u8,
        RefFrameSignBias: h.ref_frame_sign_bias,
        OrderHint: h.order_hint as u8,
        SavedOrderHints: h.order_hints.map(|h| h as u8),
    }
}

/// A frame header as Vulkan's std structures; the caller links the pointers.
struct StdPicture {
    info: sv::StdVideoDecodeAV1PictureInfo,
    tile: sv::StdVideoAV1TileInfo,
    mi_col: Vec<u16>,
    mi_row: Vec<u16>,
    width_sbs: Vec<u16>,
    height_sbs: Vec<u16>,
    quantization: sv::StdVideoAV1Quantization,
    segmentation: sv::StdVideoAV1Segmentation,
    loop_filter: sv::StdVideoAV1LoopFilter,
    cdef: sv::StdVideoAV1CDEF,
    restoration: sv::StdVideoAV1LoopRestoration,
    global_motion: sv::StdVideoAV1GlobalMotion,
    film_grain: sv::StdVideoAV1FilmGrain,
    apply_grain: bool,
}

impl StdPicture {
    fn new(h: &FrameHeader, grain_on_gpu: bool) -> Self {
        let b = |v: bool| u32::from(v);
        let apply_grain = grain_on_gpu && h.film_grain.apply_grain;
        let mut f = zeroed::<sv::StdVideoDecodeAV1PictureInfoFlags>();
        f.set_error_resilient_mode(b(h.error_resilient_mode));
        f.set_disable_cdf_update(b(h.disable_cdf_update));
        f.set_use_superres(b(h.use_superres));
        f.set_render_and_frame_size_different(b(h.render_and_frame_size_different));
        f.set_allow_screen_content_tools(b(h.allow_screen_content_tools));
        f.set_is_filter_switchable(b(h.is_filter_switchable));
        f.set_force_integer_mv(b(h.force_integer_mv));
        f.set_frame_size_override_flag(b(h.frame_size_override_flag));
        f.set_buffer_removal_time_present_flag(b(h.buffer_removal_time_present));
        f.set_allow_intrabc(b(h.allow_intrabc));
        f.set_frame_refs_short_signaling(b(h.frame_refs_short_signaling));
        f.set_allow_high_precision_mv(b(h.allow_high_precision_mv));
        f.set_is_motion_mode_switchable(b(h.is_motion_mode_switchable));
        f.set_use_ref_frame_mvs(b(h.use_ref_frame_mvs));
        f.set_disable_frame_end_update_cdf(b(h.disable_frame_end_update_cdf));
        f.set_allow_warped_motion(b(h.allow_warped_motion));
        f.set_reduced_tx_set(b(h.reduced_tx_set));
        f.set_reference_select(b(h.reference_select));
        f.set_skip_mode_present(b(h.skip_mode_present));
        f.set_delta_q_present(b(h.delta_q_present));
        f.set_delta_lf_present(b(h.delta_lf_present));
        f.set_delta_lf_multi(b(h.delta_lf_multi));
        f.set_segmentation_enabled(b(h.segmentation.enabled));
        f.set_segmentation_update_map(b(h.segmentation.update_map));
        f.set_segmentation_temporal_update(b(h.segmentation.temporal_update));
        f.set_segmentation_update_data(b(h.segmentation.update_data));
        f.set_UsesLr(b(h.loop_restoration.uses_lr()));
        f.set_usesChromaLr(b(h.loop_restoration.lr_type[1..].iter().any(|&t| t != 0)));
        f.set_apply_grain(b(apply_grain));
        let info = sv::StdVideoDecodeAV1PictureInfo {
            flags: f,
            frame_type: h.frame_type as u32,
            current_frame_id: h.current_frame_id,
            OrderHint: h.order_hint as u8,
            primary_ref_frame: h.primary_ref_frame,
            refresh_frame_flags: h.refresh_frame_flags,
            reserved1: 0,
            interpolation_filter: u32::from(h.interpolation_filter),
            TxMode: u32::from(h.tx_mode),
            delta_q_res: h.delta_q_res,
            delta_lf_res: h.delta_lf_res,
            SkipModeFrame: h.skip_mode_frame,
            coded_denom: h.coded_denom,
            reserved2: [0; 3],
            OrderHints: h.order_hints.map(|o| o as u8),
            expectedFrameId: [0; 8],
            pTileInfo: std::ptr::null(),
            pQuantization: std::ptr::null(),
            pSegmentation: std::ptr::null(),
            pLoopFilter: std::ptr::null(),
            pCDEF: std::ptr::null(),
            pLoopRestoration: std::ptr::null(),
            pGlobalMotion: std::ptr::null(),
            pFilmGrain: std::ptr::null(),
        };
        let t = &h.tile_info;
        let mut tile_flags = zeroed::<sv::StdVideoAV1TileInfoFlags>();
        tile_flags.set_uniform_tile_spacing_flag(b(t.uniform_tile_spacing));
        let tile = sv::StdVideoAV1TileInfo {
            flags: tile_flags,
            TileCols: t.tile_cols as u8,
            TileRows: t.tile_rows as u8,
            context_update_tile_id: t.context_update_tile_id as u16,
            tile_size_bytes_minus_1: t.tile_size_bytes_minus_1,
            reserved1: [0; 7],
            pMiColStarts: std::ptr::null(),
            pMiRowStarts: std::ptr::null(),
            pWidthInSbsMinus1: std::ptr::null(),
            pHeightInSbsMinus1: std::ptr::null(),
        };
        let q = &h.quantization;
        let mut q_flags = zeroed::<sv::StdVideoAV1QuantizationFlags>();
        q_flags.set_using_qmatrix(b(q.using_qmatrix));
        q_flags.set_diff_uv_delta(b(q.diff_uv_delta));
        let quantization = sv::StdVideoAV1Quantization {
            flags: q_flags,
            base_q_idx: q.base_q_idx,
            DeltaQYDc: q.delta_q_y_dc,
            DeltaQUDc: q.delta_q_u_dc,
            DeltaQUAc: q.delta_q_u_ac,
            DeltaQVDc: q.delta_q_v_dc,
            DeltaQVAc: q.delta_q_v_ac,
            qm_y: q.qm_y,
            qm_u: q.qm_u,
            qm_v: q.qm_v,
        };
        let s = &h.segmentation;
        let segmentation = sv::StdVideoAV1Segmentation {
            FeatureEnabled: s.feature_enabled.map(|features| {
                features
                    .iter()
                    .enumerate()
                    .fold(0u8, |mask, (j, &on)| mask | (u8::from(on) << j))
            }),
            FeatureData: s.feature_data,
        };
        let lf = &h.loop_filter;
        let mut lf_flags = zeroed::<sv::StdVideoAV1LoopFilterFlags>();
        lf_flags.set_loop_filter_delta_enabled(b(lf.delta_enabled));
        lf_flags.set_loop_filter_delta_update(b(lf.delta_update));
        let loop_filter = sv::StdVideoAV1LoopFilter {
            flags: lf_flags,
            loop_filter_level: lf.level,
            loop_filter_sharpness: lf.sharpness,
            update_ref_delta: 0,
            loop_filter_ref_deltas: lf.ref_deltas,
            update_mode_delta: 0,
            loop_filter_mode_deltas: lf.mode_deltas,
        };
        let c = &h.cdef;
        let cdef = sv::StdVideoAV1CDEF {
            cdef_damping_minus_3: c.damping_minus_3,
            cdef_bits: c.bits,
            cdef_y_pri_strength: c.y_pri_strength,
            cdef_y_sec_strength: c.y_sec_strength,
            cdef_uv_pri_strength: c.uv_pri_strength,
            cdef_uv_sec_strength: c.uv_sec_strength,
        };
        let lr = &h.loop_restoration;
        // Coded lr_type to StdVideoAV1FrameRestorationType: none, switchable, Wiener, self-guided.
        let remap = [
            sv::StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_NONE,
            sv::StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SWITCHABLE,
            sv::StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_WIENER,
            sv::StdVideoAV1FrameRestorationType_STD_VIDEO_AV1_FRAME_RESTORATION_TYPE_SGRPROJ,
        ];
        let luma_size = 1 + u16::from(lr.lr_unit_shift);
        let chroma_size = luma_size - u16::from(lr.lr_uv_shift);
        let restoration = sv::StdVideoAV1LoopRestoration {
            FrameRestorationType: lr.lr_type.map(|t| remap[usize::from(t & 3)]),
            LoopRestorationSize: [luma_size, chroma_size, chroma_size],
        };
        let global_motion = sv::StdVideoAV1GlobalMotion {
            GmType: h.global_motion.gm_type,
            gm_params: h.global_motion.gm_params,
        };
        let g = &h.film_grain;
        let mut g_flags = zeroed::<sv::StdVideoAV1FilmGrainFlags>();
        g_flags.set_chroma_scaling_from_luma(b(g.chroma_scaling_from_luma));
        g_flags.set_overlap_flag(b(g.overlap_flag));
        g_flags.set_clip_to_restricted_range(b(g.clip_to_restricted_range));
        g_flags.set_update_grain(b(g.update_grain));
        let film_grain = sv::StdVideoAV1FilmGrain {
            flags: g_flags,
            grain_scaling_minus_8: g.grain_scaling_minus_8,
            ar_coeff_lag: g.ar_coeff_lag,
            ar_coeff_shift_minus_6: g.ar_coeff_shift_minus_6,
            grain_scale_shift: g.grain_scale_shift,
            grain_seed: g.grain_seed,
            film_grain_params_ref_idx: g.film_grain_params_ref_idx,
            num_y_points: g.num_y_points,
            point_y_value: g.point_y_value,
            point_y_scaling: g.point_y_scaling,
            num_cb_points: g.num_cb_points,
            point_cb_value: g.point_cb_value,
            point_cb_scaling: g.point_cb_scaling,
            num_cr_points: g.num_cr_points,
            point_cr_value: g.point_cr_value,
            point_cr_scaling: g.point_cr_scaling,
            ar_coeffs_y_plus_128: g.ar_coeffs_y_plus_128.map(|v| v as i8),
            ar_coeffs_cb_plus_128: g.ar_coeffs_cb_plus_128.map(|v| v as i8),
            ar_coeffs_cr_plus_128: g.ar_coeffs_cr_plus_128.map(|v| v as i8),
            cb_mult: g.cb_mult,
            cb_luma_mult: g.cb_luma_mult,
            cb_offset: g.cb_offset,
            cr_mult: g.cr_mult,
            cr_luma_mult: g.cr_luma_mult,
            cr_offset: g.cr_offset,
        };
        Self {
            info,
            tile,
            mi_col: t.mi_col_starts.clone(),
            mi_row: t.mi_row_starts.clone(),
            width_sbs: t.width_in_sbs_minus_1.clone(),
            height_sbs: t.height_in_sbs_minus_1.clone(),
            quantization,
            segmentation,
            loop_filter,
            cdef,
            restoration,
            global_motion,
            film_grain,
            apply_grain,
        }
    }
}
