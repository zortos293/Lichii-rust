//! The codec-neutral side of decoding on the host GPU's video queue.
//!
//! A [`DecodeCore`] owns one stream's video session, its reference picture (DPB) array with a
//! layer per slot, an output array beside it, the bitstream buffer and the readback buffer.
//! Codecs (`video_av1`, `video_vp9`) read their headers, fill the codec's std structures and call
//! [`DecodeCore::decode`] for each picture, then [`DecodeCore::copy_out`] for each shown one,
//! which copies it to CPU memory as NV12 on the universal queue. The output array is shared by
//! both queue families, so no ownership transfers are needed. Each picture is decoded and copied
//! before the next one starts.

use std::ffi::{CStr, c_void};
use std::sync::{Arc, Weak};

use ash::{khr, vk};
use lichii_video::{Codec, Color, GpuStream, GpuVideo, OwnedPicture, VideoError};

use crate::video::{DecodeCaps, VideoDevice};
use crate::vp9_std;

pub(crate) const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
/// DPB slots used at most: eight reference slots and the picture being decoded.
pub(crate) const MAX_SLOTS: u32 = 9;

/// The host GPU's video decoders, as `lichii_video` sees them.
struct VulkanVideo {
    device: Weak<VideoDevice>,
}

impl GpuVideo for VulkanVideo {
    fn open(&self, codec: Codec) -> Option<Box<dyn GpuStream>> {
        let device = self.device.upgrade()?;
        let video = &device.video;
        let opened: Result<Box<dyn GpuStream>, VideoError> = match codec {
            Codec::Av1 if video.av1.is_some() => {
                crate::video_av1::Av1Decoder::new(device).map(|d| Box::new(d) as _)
            }
            Codec::Vp9 if video.vp9.is_some() => {
                crate::video_vp9::Vp9Decoder::new(device).map(|d| Box::new(d) as _)
            }
            _ => return None,
        };
        match opened {
            Ok(decoder) => Some(decoder),
            Err(error) => {
                tracing::warn!("Vulkan video: opening a {codec:?} decoder failed: {error:?}");
                None
            }
        }
    }
}

/// Makes `device`'s video queue decode for `lichii_video` decoders opened from now on.
pub(crate) fn register(device: &crate::device::Device) {
    if let Some(video) = &device.video {
        lichii_video::set_gpu_video(Some(Arc::new(VulkanVideo {
            device: Arc::downgrade(video),
        })));
    }
}

/// An all-zero std flags structure: plain bitfields, every bit cleared.
pub(crate) fn zeroed<T: Copy>() -> T {
    // SAFETY: only used for the `StdVideo*Flags` bitfield structs, for which zero is valid.
    unsafe { std::mem::zeroed() }
}

pub(crate) fn vk_error(result: vk::Result) -> VideoError {
    VideoError::Gpu(result.as_raw())
}

pub(crate) fn check(result: vk::Result) -> Result<(), VideoError> {
    match result {
        vk::Result::SUCCESS => Ok(()),
        other => Err(vk_error(other)),
    }
}

/// The codec part of a decode profile, boxed so the profile chain can point at it.
pub(crate) enum CodecProfile {
    Av1(Box<vk::VideoDecodeAV1ProfileInfoKHR<'static>>),
    Vp9(Box<vp9_std::VideoDecodeVp9ProfileInfo>),
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

/// A DPB slot a decode reads or writes: its layer, picture size and the codec's DPB slot info
/// (null for codecs without one).
pub(crate) struct Slot {
    pub index: u32,
    pub extent: vk::Extent2D,
    pub info: *const c_void,
}

/// The Vulkan side of one stream.
pub(crate) struct DecodeCore {
    device: Arc<VideoDevice>,
    video_queue: khr::video_queue::Device,
    video_decode: khr::video_decode_queue::Device,
    family: u32,
    codec: CodecProfile,
    operation: vk::VideoCodecOperationFlagsKHR,
    std_name: &'static CStr,
    std_version: u32,
    pub(crate) max_references: u32,
    granularity: vk::Extent2D,
    max_extent: vk::Extent2D,
    pub(crate) offset_alignment: u64,
    size_alignment: u64,
    video_pool: vk::CommandPool,
    video_cmd: vk::CommandBuffer,
    video_fence: vk::Fence,
    main_pool: vk::CommandPool,
    main_cmd: vk::CommandBuffer,
    main_fence: vk::Fence,
    decoded: vk::Semaphore,
    session: Option<Session>,
    params: vk::VideoSessionParametersKHR,
    bitstream: Option<HostBuffer>,
    readback: Option<HostBuffer>,
    /// The bytes of the next decode, laid out as its bitstream buffer.
    pub(crate) staging: Vec<u8>,
}

// SAFETY: the Vulkan objects and mapped pointers are owned by this stream alone and only used
// through `&mut self`, from the thread that decodes it.
unsafe impl Send for DecodeCore {}

fn memory_type(
    device: &VideoDevice,
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
    device: &VideoDevice,
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

impl DecodeCore {
    /// The core of a stream of `codec`, whose decode capabilities are `caps` and whose std
    /// header is `std_name`; at most `max_references` references a picture.
    pub(crate) fn new(
        device: Arc<VideoDevice>,
        codec: CodecProfile,
        caps: &DecodeCaps,
        std_name: &'static CStr,
        max_references: u32,
    ) -> Result<Self, VideoError> {
        let operation = match codec {
            CodecProfile::Av1(_) => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
            CodecProfile::Vp9(_) => vp9_std::DECODE_VP9,
        };
        let video = &device.video;
        let raw = &device.raw;
        let video_queue = khr::video_queue::Device::new(&device.instance.raw, raw);
        let video_decode = khr::video_decode_queue::Device::new(&device.instance.raw, raw);
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
        let (main_pool, main_cmd) = pool(device.transfer_family)?;
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
            codec,
            operation,
            std_name,
            std_version: caps.std_header_version,
            max_references: caps.max_active_references.min(max_references),
            granularity: caps.picture_granularity,
            max_extent: caps.max_extent,
            offset_alignment: caps.bitstream_offset_alignment.max(1),
            size_alignment: caps.bitstream_size_alignment.max(1),
            video_pool,
            video_cmd,
            video_fence,
            main_pool,
            main_cmd,
            main_fence,
            decoded,
            session: None,
            params: vk::VideoSessionParametersKHR::null(),
            bitstream: None,
            readback: None,
            staging: Vec::new(),
            device,
        })
    }

    pub(crate) fn device(&self) -> &VideoDevice {
        &self.device
    }

    pub(crate) fn video_queue(&self) -> &khr::video_queue::Device {
        &self.video_queue
    }

    /// The decode profile, its codec part chained.
    fn profile(&self) -> vk::VideoProfileInfoKHR<'_> {
        let codec: *const c_void = match &self.codec {
            CodecProfile::Av1(av1) => (&**av1 as *const vk::VideoDecodeAV1ProfileInfoKHR).cast(),
            CodecProfile::Vp9(vp9) => (&**vp9 as *const vp9_std::VideoDecodeVp9ProfileInfo).cast(),
        };
        crate::video::profile_420_8bit(self.operation, codec)
    }

    fn image(
        &self,
        extent: vk::Extent2D,
        usage: vk::ImageUsageFlags,
        families: &[u32],
    ) -> Result<Image, VideoError> {
        let raw = &self.device.raw;
        let profiles = [self.profile()];
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
        let profiles = [self.profile()];
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

    /// Replaces the session parameters (destroying the old ones).
    pub(crate) fn set_params(&mut self, params: vk::VideoSessionParametersKHR) {
        if self.params != vk::VideoSessionParametersKHR::null() {
            // SAFETY: no submitted work uses the parameters.
            unsafe {
                (self.video_queue.fp().destroy_video_session_parameters_khr)(
                    self.device.raw.handle(),
                    self.params,
                    std::ptr::null(),
                )
            };
        }
        self.params = params;
    }

    fn free_session(&mut self) {
        self.set_params(vk::VideoSessionParametersKHR::null());
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

    /// The live session (made by [`Self::ensure_session`]).
    pub(crate) fn session(&self) -> vk::VideoSessionKHR {
        self.session.as_ref().expect("session made first").raw
    }

    /// A session for pictures up to `width` x `height` (rounded to the access granularity).
    /// Returns whether it is new: its DPB holds no pictures and it has no parameters.
    pub(crate) fn ensure_session(&mut self, width: u32, height: u32) -> Result<bool, VideoError> {
        let align = |v: u32, g: u32| v.next_multiple_of(g.max(1));
        let extent = vk::Extent2D {
            width: align(width, self.granularity.width),
            height: align(height, self.granularity.height),
        };
        if extent.width > self.max_extent.width || extent.height > self.max_extent.height {
            return Err(VideoError::Gpu(0));
        }
        if let Some(session) = &self.session
            && session.extent.width >= extent.width
            && session.extent.height >= extent.height
        {
            return Ok(false);
        }
        self.free_session();
        let raw = &self.device.raw;
        let fp = self.video_queue.fp();
        let profile = self.profile();
        let std = vk::ExtensionProperties::default()
            .extension_name(self.std_name)
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
            &[self.family, self.device.transfer_family],
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
        Ok(true)
    }

    /// Appends `bytes` to the staging bytes at the bitstream offset alignment and returns the
    /// offset they start at.
    pub(crate) fn stage(&mut self, bytes: &[u8]) -> u32 {
        let at = self
            .staging
            .len()
            .next_multiple_of(self.offset_alignment as usize);
        self.staging.resize(at, 0);
        self.staging.extend_from_slice(bytes);
        at as u32
    }

    /// Copies the staging bytes into the bitstream buffer and returns the range to decode from.
    fn upload(&mut self) -> Result<u64, VideoError> {
        let range = (self.staging.len() as u64).next_multiple_of(self.size_alignment);
        if self.bitstream.as_ref().is_none_or(|b| b.size < range) {
            if let Some(old) = self.bitstream.take() {
                self.free_buffer(old);
            }
            let size = range.next_power_of_two().max(1 << 20);
            self.bitstream =
                Some(self.host_buffer(size, vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR, true)?);
        }
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
        Ok(range)
    }

    /// Decodes the staged bytes into DPB slot `setup` (and the output layer of the same index)
    /// at `extent`, reading `refs`. `codec_info` is the codec's picture info, chained to the
    /// decode info.
    ///
    /// # Safety
    ///
    /// `codec_info` and every slot's `info` are valid for the codec of this core and live until
    /// the call returns.
    pub(crate) unsafe fn decode(
        &mut self,
        codec_info: *const c_void,
        setup: &Slot,
        refs: &[Slot],
        extent: vk::Extent2D,
    ) -> Result<(), VideoError> {
        let range = self.upload()?;
        let bitstream = self.bitstream.as_ref().expect("uploaded");
        let session = self.session.as_ref().expect("session made first");
        let resource = |slot: u32, extent: vk::Extent2D, view: vk::ImageView| {
            vk::VideoPictureResourceInfoKHR::default()
                .coded_offset(vk::Offset2D::default())
                .coded_extent(extent)
                .base_array_layer(slot)
                .image_view_binding(view)
        };
        let ref_resources: Vec<vk::VideoPictureResourceInfoKHR> = refs
            .iter()
            .map(|r| resource(r.index, r.extent, session.dpb.view))
            .collect();
        let slot_infos: Vec<vk::VideoReferenceSlotInfoKHR> = refs
            .iter()
            .zip(&ref_resources)
            .map(|(r, res)| {
                let mut slot = vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(r.index as i32)
                    .picture_resource(res);
                slot.p_next = r.info;
                slot
            })
            .collect();
        let setup_resource = resource(setup.index, extent, session.dpb.view);
        let mut setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(setup.index as i32)
            .picture_resource(&setup_resource);
        setup_slot.p_next = setup.info;
        let mut begin_slots = slot_infos.clone();
        let mut pending = setup_slot;
        pending.slot_index = -1;
        begin_slots.push(pending);

        let out_resource = resource(setup.index, extent, session.out.view);
        let mut decode = vk::VideoDecodeInfoKHR::default()
            .src_buffer(bitstream.raw)
            .src_buffer_offset(0)
            .src_buffer_range(range)
            .dst_picture_resource(out_resource)
            .setup_reference_slot(&setup_slot)
            .reference_slots(&slot_infos);
        decode.p_next = codec_info;
        let begin = vk::VideoBeginCodingInfoKHR::default()
            .video_session(session.raw)
            .video_session_parameters(self.params)
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
                .old_layout(session.out_layout[setup.index as usize])
                .new_layout(vk::ImageLayout::VIDEO_DECODE_DST_KHR)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(session.out.raw)
                .subresource_range(layers(setup.index, 1)),
        ];
        let fresh = session.fresh;
        let vq = self.video_queue.fp();
        let vd = self.video_decode.fp();
        // SAFETY: `cmd` belongs to the video queue's pool and is not pending (every submission
        // is waited for); every struct the commands point at lives until the call returns, the
        // codec's as the caller vouches.
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
            let queue = self.device.video.queue();
            raw.queue_submit2(*queue, &[submit], self.video_fence)
                .map_err(vk_error)?;
        }
        let session = self.session.as_mut().expect("session made first");
        session.fresh = false;
        session.dpb_ready = true;
        session.out_layout[setup.index as usize] = vk::ImageLayout::VIDEO_DECODE_DST_KHR;
        Ok(())
    }

    /// Copies output layer `slot` to CPU memory on the universal queue. With `after_decode`, the
    /// copy waits for the decode just submitted.
    pub(crate) fn copy_out(
        &mut self,
        slot: u32,
        extent: vk::Extent2D,
        after_decode: bool,
        color: Color,
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
            if after_decode {
                submit = submit.wait_semaphore_infos(&waits);
            }
            self.device
                .submit(&[submit], self.main_fence)
                .map_err(vk_error)?;
            let mut fences = vec![self.main_fence];
            if after_decode {
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
        Ok(OwnedPicture {
            width: extent.width,
            height: extent.height,
            pts,
            data,
            color,
        })
    }
}

impl Drop for DecodeCore {
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
