//! AV1 decoding on the host GPU's video queue (`VK_KHR_video_decode_av1`).
//!
//! `lichii_video::av1` reads the headers; this fills Vulkan's AV1 std structures for each frame
//! and decodes it through [`DecodeCore`], which copies each shown picture to CPU memory.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::sync::Arc;

use ash::vk;
use ash::vk::native as sv;
use lichii_video::av1::frame::{FrameHeader, FrameType, NUM_REF_FRAMES, REFS_PER_FRAME, RefFrame};
use lichii_video::av1::{Av1Stream, ParsedFrame, SequenceHeader};
use lichii_video::{Codec, Color, GpuStream, Matrix, OwnedPicture, VideoError};

use crate::video::VideoDevice;
use crate::video_decode::{CodecProfile, DecodeCore, MAX_SLOTS, Slot, check, zeroed};

const STD_AV1_DECODE: &std::ffi::CStr = c"VK_STD_vulkan_video_codec_av1_decode";

/// The Vulkan side of one AV1 stream.
struct Gpu {
    core: DecodeCore,
    film_grain: bool,
    /// The sequence header the session parameters hold.
    params_for: Option<SequenceHeader>,
    /// The DPB slot each reference slot's picture is in.
    slot_of_ref: [Option<u32>; NUM_REF_FRAMES],
}

/// One AV1 stream decoding on the video queue.
pub(crate) struct Av1Decoder {
    parser: Av1Stream,
    gpu: Gpu,
    ready: VecDeque<OwnedPicture>,
    spare: Vec<Vec<u8>>,
}

impl Av1Decoder {
    pub(crate) fn new(device: Arc<VideoDevice>) -> Result<Self, VideoError> {
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

/// The colour description of pictures of `seq`.
fn color(seq: &SequenceHeader) -> Color {
    let c = &seq.color;
    Color {
        matrix: match c.matrix_coefficients {
            1 => Some(Matrix::Bt709),
            5 | 6 => Some(Matrix::Bt601),
            9 | 10 => Some(Matrix::Bt2020),
            _ => None,
        },
        full_range: c.color_range,
    }
}

impl Gpu {
    fn new(device: Arc<VideoDevice>) -> Result<Self, VideoError> {
        let av1 = device
            .video
            .av1
            .clone()
            .ok_or(VideoError::NoDecoder(Codec::Av1))?;
        let profile = vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(sv::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
            .film_grain_support(av1.film_grain);
        let core = DecodeCore::new(
            device,
            CodecProfile::Av1(Box::new(profile)),
            &av1.caps,
            STD_AV1_DECODE,
            REFS_PER_FRAME as u32,
        )?;
        Ok(Self {
            core,
            film_grain: av1.film_grain,
            params_for: None,
            slot_of_ref: [None; NUM_REF_FRAMES],
        })
    }

    /// A session for the largest pictures of `seq`.
    fn ensure_session(&mut self, seq: &SequenceHeader) -> Result<(), VideoError> {
        if seq.color.bit_depth != 8
            || seq.color.mono_chrome
            || !seq.color.subsampling_x
            || !seq.color.subsampling_y
        {
            return Err(VideoError::Gpu(0));
        }
        let made = self.core.ensure_session(
            u32::from(seq.max_frame_width_minus_1) + 1,
            u32::from(seq.max_frame_height_minus_1) + 1,
        )?;
        if made {
            self.params_for = None;
            self.slot_of_ref = [None; NUM_REF_FRAMES];
        }
        Ok(())
    }

    /// Session parameters holding `seq`, made again when the sequence header changes.
    fn ensure_params(&mut self, seq: &SequenceHeader) -> Result<(), VideoError> {
        if self.params_for.as_ref() == Some(seq) {
            return Ok(());
        }
        let session = self.core.session();
        let fp = self.core.video_queue().fp();
        let device = self.core.device();
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
                device.raw.handle(),
                &info,
                std::ptr::null(),
                &mut params,
            )
        })?;
        self.core.set_params(params);
        self.params_for = Some(seq.clone());
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
            let picture = self
                .core
                .copy_out(slot, extent, false, color(seq), pts, spare)?;
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
        let picture = self
            .core
            .copy_out(setup, extent, true, color(seq), pts, spare)?;
        if h.show_frame {
            ready.push_back(picture);
        } else {
            spare.push(picture.data);
        }
        for (i, slot) in self.slot_of_ref.iter_mut().enumerate() {
            if h.refresh_frame_flags & (1 << i) != 0 {
                *slot = Some(setup);
            }
        }
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
        self.core.staging.clear();
        let mut offsets = Vec::with_capacity(frame.tiles.len());
        let mut sizes = Vec::with_capacity(frame.tiles.len());
        for tile in &frame.tiles {
            offsets.push(self.core.stage(&data[tile.offset..tile.offset + tile.size]));
            sizes.push(tile.size as u32);
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
        let ref_dpb: Vec<vk::VideoDecodeAV1DpbSlotInfoKHR> = ref_std
            .iter()
            .map(|r| vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(r))
            .collect();
        let ref_list: Vec<Slot> = ref_sources
            .iter()
            .zip(&ref_slots)
            .zip(&ref_dpb)
            .map(|((&index, &slot), dpb)| {
                let r = &refs.slots[index];
                Slot {
                    index: slot,
                    extent: vk::Extent2D {
                        width: r.upscaled_width,
                        height: r.frame_height,
                    },
                    info: (dpb as *const vk::VideoDecodeAV1DpbSlotInfoKHR).cast::<c_void>(),
                }
            })
            .collect();
        let setup_std = current_reference_info(h);
        let setup_dpb = vk::VideoDecodeAV1DpbSlotInfoKHR::default().std_reference_info(&setup_std);
        let setup_slot = Slot {
            index: setup,
            extent,
            info: (&setup_dpb as *const vk::VideoDecodeAV1DpbSlotInfoKHR).cast(),
        };
        let av1_info = vk::VideoDecodeAV1PictureInfoKHR::default()
            .std_picture_info(&info)
            .reference_name_slot_indices(name_slots)
            .frame_header_offset(0)
            .tile_offsets(&offsets)
            .tile_sizes(&sizes);
        // SAFETY: AV1 picture and DPB slot infos for this AV1 core, alive until the call returns.
        unsafe {
            self.core.decode(
                (&av1_info as *const vk::VideoDecodeAV1PictureInfoKHR).cast(),
                &setup_slot,
                &ref_list,
                extent,
            )
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
