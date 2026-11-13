//! `/dev/nvhost-ctrl-gpu`: what the GPU is (characteristics, TPC masks, zcull, ZBC tables) and a
//! few GPU-wide controls.
//!
//! Switch 1 titles see the GM20B (Maxwell). Switch 2 titles see chip 0x17E with the Ampere classes,
//! the 0xD0-byte characteristics block, interface versions 0x40, TSG and subcontext flags and 64
//! subcontexts per TSG (NX2_NOTES rows 24 and 45); ioctls nobody identified yet succeed with
//! zeroes for them.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use lichii_kernel::Event;
use parking_lot::Mutex;

use super::ioctl::{Args, Variant, read_struct, write_struct};
use super::{NvResult, Nvdrv};

/// Size of the Switch 1 characteristics block.
const SWITCH1_BLOCK: u64 = 0xA0;
/// Size of the Switch 2 characteristics block.
const SWITCH2_BLOCK: u64 = 0xD0;
/// Offset of the Switch 2 block's subcontexts-per-TSG word.
const MAX_SUBCONTEXTS_OFFSET: usize = 0xB4;
/// Ampere VEIDs per TSG.
const MAX_SUBCONTEXTS: u32 = 64;
/// Where the Switch 2 block keeps nine s16 interface versions.
const INTERFACE_VERSIONS: std::ops::RangeInclusive<usize> = 0x70..=0x80;
const INTERFACE_VERSION: i16 = 0x40;

const FLAG_SUPPORT_TSG: u64 = 1 << 8;
const FLAG_SUPPORT_TSG_SUBCONTEXTS: u64 = 1 << 22;
const FLAG_SUPPORT_SYNCPOINT_ADDRESS: u64 = 1 << 26;
const FLAG_SUPPORT_MAPPING_MODIFY: u64 = 1 << 34;
const FLAG_SUPPORT_REMAP: u64 = 1 << 35;
/// 2D, 3D, compute, inline-to-memory and ZBC engines.
const FLAG_ENGINES: u64 = 0x1F << 40;

/// `nvgpu_gpu_characteristics` up to the end of the Switch 1 layout.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct GpuCharacteristics {
    arch: u32,
    implementation: u32,
    rev: u32,
    num_gpc: u32,
    l2_cache_size: u64,
    on_board_video_memory_size: u64,
    num_tpc_per_gpc: u32,
    bus_type: u32,
    big_page_size: u32,
    compression_page_size: u32,
    pde_coverage_bit_count: u32,
    available_big_page_sizes: u32,
    gpc_mask: u32,
    sm_arch_sm_version: u32,
    sm_arch_spa_version: u32,
    sm_arch_warp_count: u32,
    gpu_va_bit_count: u32,
    reserved: u32,
    flags: u64,
    twod_class: u32,
    threed_class: u32,
    compute_class: u32,
    gpfifo_class: u32,
    inline_to_memory_class: u32,
    dma_copy_class: u32,
    max_fbps_count: u32,
    fbp_en_mask: u32,
    max_ltc_per_fbp: u32,
    max_lts_per_ltc: u32,
    max_tex_per_tpc: u32,
    max_gpc_count: u32,
    rop_l2_en_mask_0: u32,
    rop_l2_en_mask_1: u32,
    chipname: u64,
    gr_compbit_store_base_hw: u64,
}

const _: () = assert!(size_of::<GpuCharacteristics>() == SWITCH1_BLOCK as usize);

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Characteristics {
    buf_size: u64,
    buf_addr: u64,
    gc: GpuCharacteristics,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct TpcMasks {
    mask_buffer_size: u32,
    _pad0: u32,
    mask_buffer_address: u64,
    tpc_mask: u32,
    _pad1: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ZcullInfo {
    width_align_pixels: u32,
    height_align_pixels: u32,
    pixel_squares_by_aliquots: u32,
    aliquot_total: u32,
    region_byte_multiplier: u32,
    region_header_size: u32,
    subregion_header_size: u32,
    subregion_width_align_pixels: u32,
    subregion_height_align_pixels: u32,
    subregion_count: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ZbcSetTable {
    color_ds: [u32; 4],
    color_l2: [u32; 4],
    depth: u32,
    format: u32,
    kind: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct ZbcQueryTable {
    color_ds: [u32; 4],
    color_l2: [u32; 4],
    depth: u32,
    ref_count: u32,
    format: u32,
    kind: u32,
    index_size: u32,
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct EngineInfoItem {
    engine_id: u32,
    engine_instance: u32,
    runlist_id: i32,
    reserved: u32,
}

/// The graphics engine (runlist 0) and an async copy engine (runlist 1).
const ENGINES: [EngineInfoItem; 2] = [
    EngineInfoItem {
        engine_id: 0,
        engine_instance: 0,
        runlist_id: 0,
        reserved: 0,
    },
    EngineInfoItem {
        engine_id: 2,
        engine_instance: 0,
        runlist_id: 1,
        reserved: 0,
    },
];

const ZBC_COLOR: u32 = 1;
const ZBC_DEPTH: u32 = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
struct ZbcColor {
    color_ds: [u32; 4],
    color_l2: [u32; 4],
    format: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ZbcDepth {
    depth: u32,
    format: u32,
}

#[derive(Default)]
struct Zbc {
    colors: Vec<(ZbcColor, u32)>,
    depths: Vec<(ZbcDepth, u32)>,
}

pub(crate) struct CtrlGpu {
    nvdrv: Arc<Nvdrv>,
    error_notifier: Arc<Event>,
    unknown: Arc<Event>,
    zbc: Mutex<Zbc>,
}

impl CtrlGpu {
    pub fn new(nvdrv: Arc<Nvdrv>) -> Self {
        Self {
            nvdrv,
            error_notifier: Arc::new(Event::new()),
            unknown: Arc::new(Event::new()),
            zbc: Mutex::default(),
        }
    }

    pub fn ioctl(&self, args: &mut Args) -> Option<NvResult> {
        let nx2 = self.nvdrv.nx2;
        Some(match (args.variant, args.number()) {
            (Variant::One, (b'G', 0x1)) => args.fixed(|size: &mut u32| {
                *size = 1;
                NvResult::Success
            }),
            (Variant::One, (b'G', 0x2)) => args.fixed(Self::zcull_info),
            (Variant::One, (b'G', 0x3)) => args.fixed(|p| self.zbc_set(p)),
            (Variant::One, (b'G', 0x4)) => args.fixed(|p| self.zbc_query(p)),
            (Variant::One, (b'G', 0x5)) => args.fixed(|p| self.characteristics(p)),
            (Variant::Three, (b'G', 0x5)) => self.characteristics_inline(args),
            (Variant::One, (b'G', 0x6)) => args.fixed(Self::tpc_masks),
            (Variant::Three, (b'G', 0x6)) => {
                let result = args.fixed(Self::tpc_masks);
                let masks: TpcMasks = read_struct(args.output);
                write_struct(args.inline_output, &masks.tpc_mask);
                result
            }
            (Variant::One, (b'G', 0x7)) => NvResult::Success,
            (Variant::Three, (b'G', 0x13)) => Self::vsms_mapping(args),
            (Variant::One, (b'G', 0x14)) => args.fixed(|mask: &mut [u32; 2]| {
                *mask = [0x07, 0x01];
                NvResult::Success
            }),
            (Variant::One, (b'G', 0x15)) => args.fixed(|load: &mut u32| {
                *load = 100;
                NvResult::Success
            }),
            (Variant::One, (b'G', 0x1c)) => {
                let now = self.nvdrv.gpu.handles().timestamp.now();
                args.fixed(|time: &mut [u64; 2]| {
                    *time = [now, 0];
                    NvResult::Success
                })
            }
            (Variant::Three, (b'G', 0x1f)) if nx2 => Self::engine_info(args),
            _ if nx2 => {
                tracing::warn!(
                    ioctl = format_args!("{:#010x}", args.ioctl.0),
                    variant = ?args.variant,
                    "(stubbed) Switch 2 nvhost-ctrl-gpu ioctl answered with zeroes"
                );
                NvResult::Success
            }
            _ => return None,
        })
    }

    fn zcull_info(info: &mut ZcullInfo) -> NvResult {
        *info = ZcullInfo {
            width_align_pixels: 0x20,
            height_align_pixels: 0x20,
            pixel_squares_by_aliquots: 0x400,
            aliquot_total: 0x800,
            region_byte_multiplier: 0x20,
            region_header_size: 0x20,
            subregion_header_size: 0xc0,
            subregion_width_align_pixels: 0x20,
            subregion_height_align_pixels: 0x40,
            subregion_count: 0x10,
        };
        NvResult::Success
    }

    fn tpc_masks(params: &mut TpcMasks) -> NvResult {
        if params.mask_buffer_size != 0 {
            params.tpc_mask = 3;
        }
        NvResult::Success
    }

    /// The characteristics block and the size reported for it.
    fn gpu_characteristics(&self) -> (GpuCharacteristics, u64) {
        let mut gc = GpuCharacteristics {
            arch: 0x120,
            implementation: 0xb,
            rev: 0xa1,
            num_gpc: 0x1,
            l2_cache_size: 0x40000,
            on_board_video_memory_size: 0,
            num_tpc_per_gpc: 0x2,
            bus_type: 0x20,
            big_page_size: 0x20000,
            compression_page_size: 0x20000,
            pde_coverage_bit_count: 0x1B,
            available_big_page_sizes: 0x30000,
            gpc_mask: 0x1,
            sm_arch_sm_version: 0x503,
            sm_arch_spa_version: 0x503,
            sm_arch_warp_count: 0x80,
            gpu_va_bit_count: 0x28,
            reserved: 0,
            flags: 0x55,
            twod_class: 0x902D,
            threed_class: 0xB197,
            compute_class: 0xB1C0,
            gpfifo_class: 0xB06F,
            inline_to_memory_class: 0xA140,
            dma_copy_class: 0xB0B5,
            max_fbps_count: 0x1,
            fbp_en_mask: 0,
            max_ltc_per_fbp: 0x2,
            max_lts_per_ltc: 0x1,
            max_tex_per_tpc: 0,
            max_gpc_count: 0x1,
            rop_l2_en_mask_0: 0x21D70,
            rop_l2_en_mask_1: 0,
            chipname: u64::from_le_bytes(*b"gm20b\0\0\0"),
            gr_compbit_store_base_hw: 0,
        };
        if !self.nvdrv.nx2 {
            return (gc, SWITCH1_BLOCK);
        }
        gc.arch = 0x170;
        gc.implementation = 0xE;
        gc.num_tpc_per_gpc = 0x6;
        gc.sm_arch_sm_version = 0x808;
        gc.sm_arch_spa_version = 0x808;
        gc.threed_class = 0xC797;
        gc.compute_class = 0xC7C0;
        gc.gpfifo_class = 0xC56F;
        gc.dma_copy_class = 0xC7B5;
        gc.chipname = u64::from_le_bytes(*b"ga10f\0\0\0");
        gc.flags |= FLAG_SUPPORT_TSG
            | FLAG_SUPPORT_TSG_SUBCONTEXTS
            | FLAG_SUPPORT_SYNCPOINT_ADDRESS
            | FLAG_SUPPORT_MAPPING_MODIFY
            | FLAG_SUPPORT_REMAP
            | FLAG_ENGINES;
        let bytes = bytemuck::bytes_of_mut(&mut gc);
        for offset in INTERFACE_VERSIONS.step_by(2) {
            bytes[offset..offset + 2].copy_from_slice(&INTERFACE_VERSION.to_le_bytes());
        }
        (gc, SWITCH2_BLOCK)
    }

    fn characteristics(&self, params: &mut Characteristics) -> NvResult {
        let (gc, size) = self.gpu_characteristics();
        params.gc = gc;
        params.buf_size = size.min(SWITCH1_BLOCK);
        params.buf_addr = 0xdead_beef;
        NvResult::Success
    }

    /// Ioctl3 form: the block goes to the inline output, the Switch 2 block with its tail.
    fn characteristics_inline(&self, args: &mut Args) -> NvResult {
        let (gc, size) = self.gpu_characteristics();
        let result = args.fixed(|params: &mut Characteristics| {
            params.gc = gc;
            params.buf_size = size;
            params.buf_addr = 0xdead_beef;
            NvResult::Success
        });
        if args.inline_output.len() >= size_of::<GpuCharacteristics>() {
            write_struct(args.inline_output, &gc);
        }
        if self.nvdrv.nx2 {
            write_struct(
                args.inline_output
                    .get_mut(MAX_SUBCONTEXTS_OFFSET..)
                    .unwrap_or_default(),
                &MAX_SUBCONTEXTS,
            );
        }
        result
    }

    /// VSMS_MAPPING: a {gpc, tpc} byte pair per virtual SM; one GPC, two SMs per TPC.
    fn vsms_mapping(args: &mut Args) -> NvResult {
        if args.ioctl.0 != 0xc008_4713
            || args.input.len() != 8
            || args.output.len() != 8
            || args.inline_output.len() < 2
        {
            return NvResult::BadParameter;
        }
        args.echo();
        for (index, pair) in args
            .inline_output
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .enumerate()
        {
            pair[0] = 0;
            pair[1] = (index / 2) as u8;
        }
        NvResult::Success
    }

    /// GET_ENGINE_INFO as the Switch 2 NvRm asks for it: once for the size, once with an inline
    /// buffer of items.
    fn engine_info(args: &mut Args) -> NvResult {
        let needed = size_of_val(&ENGINES) as u32;
        let buf_size: u32 = read_struct(args.input);
        if buf_size >= needed && args.inline_output.len() >= needed as usize {
            args.inline_output[..needed as usize].copy_from_slice(bytemuck::bytes_of(&ENGINES));
        }
        args.echo();
        write_struct(args.output, &needed);
        NvResult::Success
    }

    fn zbc_set(&self, params: &mut ZbcSetTable) -> NvResult {
        let mut zbc = self.zbc.lock();
        match params.kind {
            ZBC_COLOR => {
                let color = ZbcColor {
                    color_ds: params.color_ds,
                    color_l2: params.color_l2,
                    format: params.format,
                };
                match zbc.colors.iter_mut().find(|(entry, _)| *entry == color) {
                    Some((_, references)) => *references += 1,
                    None => zbc.colors.push((color, 1)),
                }
            }
            ZBC_DEPTH => {
                let depth = ZbcDepth {
                    depth: params.depth,
                    format: params.format,
                };
                match zbc.depths.iter_mut().find(|(entry, _)| *entry == depth) {
                    Some((_, references)) => *references += 1,
                    None => zbc.depths.push((depth, 1)),
                }
            }
            0 => {}
            kind => {
                tracing::error!(kind, "ZBC set with an invalid type");
                return NvResult::BadParameter;
            }
        }
        NvResult::Success
    }

    fn zbc_query(&self, params: &mut ZbcQueryTable) -> NvResult {
        let zbc = self.zbc.lock();
        let index = params.index_size as usize;
        match params.kind {
            ZBC_COLOR => {
                let Some((color, references)) = zbc.colors.get(index) else {
                    return NvResult::BadParameter;
                };
                params.color_ds = color.color_ds;
                params.color_l2 = color.color_l2;
                params.depth = 0;
                params.ref_count = *references;
                params.format = color.format;
                params.index_size = zbc.colors.len() as u32;
            }
            ZBC_DEPTH => {
                let Some((depth, references)) = zbc.depths.get(index) else {
                    return NvResult::BadParameter;
                };
                params.color_ds = [0; 4];
                params.color_l2 = [0; 4];
                params.depth = depth.depth;
                params.ref_count = *references;
                params.format = depth.format;
                params.index_size = zbc.depths.len() as u32;
            }
            0 => {}
            _ => return NvResult::BadParameter,
        }
        NvResult::Success
    }

    pub fn query_event(&self, id: u32) -> Option<Arc<Event>> {
        match id {
            1 => Some(self.error_notifier.clone()),
            2 => Some(self.unknown.clone()),
            _ => None,
        }
    }
}
