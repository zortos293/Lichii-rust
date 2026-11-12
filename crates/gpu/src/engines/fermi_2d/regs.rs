//! 2D engine (902D) registers.
//!
//! Translated from the reference's register struct; every field offset there was checked against
//! its method number, and the offsets are asserted below.

use crate::bits::{bitfield, open_enum};

pub const NUM_REGS: usize = 0x258;

open_enum! {
    pub struct Origin {
        CENTER = 0x0,
        CORNER = 0x1,
    }
}

open_enum! {
    pub struct Filter {
        POINT = 0x0,
        BILINEAR = 0x1,
    }
}

open_enum! {
    pub struct Operation {
        SRC_COPY_AND = 0x0,
        ROP_AND = 0x1,
        BLEND = 0x2,
        SRC_COPY = 0x3,
        ROP = 0x4,
        SRC_COPY_PREMULT = 0x5,
        BLEND_PREMULT = 0x6,
        SRC_COPY_SRGB_TO_LINEAR = 0x7,
    }
}

open_enum! {
    pub struct MemoryLayout {
        BLOCK_LINEAR = 0x0,
        PITCH = 0x1,
    }
}

open_enum! {
    pub struct CpuIndexWrap {
        WRAP = 0x0,
        NO_WRAP = 0x1,
    }
}

open_enum! {
    pub struct SectorPromotion {
        NO_PROMOTION = 0x0,
        PROMOTE_TO2_V = 0x1,
        PROMOTE_TO2_H = 0x2,
        PROMOTE_TO4 = 0x3,
    }
}

open_enum! {
    pub struct NumTpcs {
        ALL = 0x0,
        ONE = 0x1,
    }
}

open_enum! {
    pub struct RenderEnableMode {
        FALSE = 0x0,
        TRUE = 0x1,
        CONDITIONAL = 0x2,
        RENDER_IF_EQUAL = 0x3,
        RENDER_IF_NOT_EQUAL = 0x4,
    }
}

open_enum! {
    pub struct ColorKeyFormat {
        A16R56G6B5 = 0x0,
        A1R5G55B5 = 0x1,
        A8R8G8B8 = 0x2,
        A2R10G10B10 = 0x3,
        Y8 = 0x4,
        Y16 = 0x5,
        Y32 = 0x6,
    }
}

open_enum! {
    pub struct PatternSelect {
        MONO_CHROME8X8 = 0x0,
        MONO_CHROME64X1 = 0x1,
        MONO_CHROME1X64 = 0x2,
        COLOR = 0x3,
    }
}

open_enum! {
    pub struct NotifyType {
        WRITE_ONLY = 0x0,
        WRITE_THEN_AWAKEN = 0x1,
    }
}

open_enum! {
    pub struct MonochromePatternColorFormat {
        A8X8R8G6B5 = 0x0,
        A1R5G5B5 = 0x1,
        A8R8G8B8 = 0x2,
        A8Y8 = 0x3,
        A8X8Y16 = 0x4,
        Y32 = 0x5,
    }
}

open_enum! {
    pub struct MonochromePatternFormat {
        CGA6_M1 = 0x0,
        LE_M1 = 0x1,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Surface {
    pub format: u32,
    pub linear: MemoryLayout,
    pub bits: SurfaceBits,
    pub depth: u32,
    pub layer: u32,
    pub pitch: u32,
    pub width: u32,
    pub height: u32,
    pub addr_upper: u32,
    pub addr_lower: u32,
}

impl Surface {
    #[inline]
    pub fn block_width(&self) -> u32 {
        self.bits.block_width()
    }

    #[inline]
    pub fn block_height(&self) -> u32 {
        self.bits.block_height()
    }

    #[inline]
    pub fn block_depth(&self) -> u32 {
        self.bits.block_depth()
    }
}

const _: () = assert!(core::mem::size_of::<Surface>() == 0x28);

bitfield! {
    pub struct SurfaceBits {
        block_width: u32 @ 0, 4;
        block_height: u32 @ 4, 4;
        block_depth: u32 @ 8, 4;
    }
}

bitfield! {
    pub struct Beta4 {
        b: u32 @ 0, 8;
        g: u32 @ 8, 8;
        r: u32 @ 16, 8;
        a: u32 @ 24, 8;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

const _: () = assert!(core::mem::size_of::<Point>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Regs {
    pub object: u32,
    _pad0: [u32; 0x3f],
    pub no_operation: u32,
    pub notify: NotifyType,
    _pad1: [u32; 0x2],
    pub wait_for_idle: u32,
    _pad2: [u32; 0xb],
    pub pm_trigger: u32,
    _pad3: [u32; 0xf],
    pub context_dma_notify: u32,
    pub dst_context_dma: u32,
    pub src_context_dma: u32,
    pub semaphore_context_dma: u32,
    _pad4: [u32; 0x1c],
    pub dst: Surface,
    pub pixels_from_cpu_index_wrap: CpuIndexWrap,
    pub kind2d_check_enable: u32,
    pub src: Surface,
    pub pixels_from_memory_sector_promotion: SectorPromotion,
    _pad5: [u32; 0x1],
    pub num_tpcs: NumTpcs,
    pub render_enable_addr_upper: u32,
    pub render_enable_addr_lower: u32,
    pub render_enable_mode: RenderEnableMode,
    _pad6: [u32; 0x4],
    pub clip_x0: u32,
    pub clip_y0: u32,
    pub clip_width: u32,
    pub clip_height: u32,
    pub clip_enable: ClipEnable,
    pub color_key_format: ColorKeyFormatReg,
    pub color_key: u32,
    pub color_key_enable: ColorKeyEnable,
    pub rop: Rop,
    pub beta1: u32,
    pub beta4: Beta4,
    pub operation: Operation,
    pub pattern_offset: RegsPatternOffset,
    pub pattern_select: PatternSelectReg,
    _pad7: [u32; 0xc],
    pub monochrome_pattern: RegsMonochromePattern,
    pub color_pattern: RegsColorPattern,
    _pad8: [u32; 0x10],
    pub render_solid: RegsRenderSolid,
    pub pixels_from_cpu: RegsPixelsFromCpu,
    _pad9: [u32; 0x3],
    pub big_endian_control: u32,
    _pad10: [u32; 0x3],
    pub pixels_from_memory: RegsPixelsFromMemory,
    _pad11: [u32; 0x20],
}

bitfield! {
    pub struct ClipEnable {
        value: u32 @ 0, 1;
    }
}

bitfield! {
    pub struct ColorKeyFormatReg {
        value: ColorKeyFormat @ 0, 3;
    }
}

bitfield! {
    pub struct ColorKeyEnable {
        value: u32 @ 0, 1;
    }
}

bitfield! {
    pub struct Rop {
        value: u32 @ 0, 8;
    }
}

bitfield! {
    pub struct RegsPatternOffset {
        x: u32 @ 0, 6;
        y: u32 @ 8, 6;
    }
}

bitfield! {
    pub struct PatternSelectReg {
        value: PatternSelect @ 0, 2;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsMonochromePattern {
    pub color_format: RegsMonochromePatternColorFormat,
    pub format: RegsMonochromePatternFormat,
    pub color0: u32,
    pub color1: u32,
    pub pattern0: u32,
    pub pattern1: u32,
}

const _: () = assert!(core::mem::size_of::<RegsMonochromePattern>() == 0x18);

bitfield! {
    pub struct RegsMonochromePatternColorFormat {
        value: MonochromePatternColorFormat @ 0, 3;
    }
}

bitfield! {
    pub struct RegsMonochromePatternFormat {
        value: MonochromePatternFormat @ 0, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsColorPattern {
    pub x8_r8_g8_b8: [u32; 0x40],
    pub r5_g6_b5: [u32; 0x20],
    pub x1_r5_g5_b5: [u32; 0x20],
    pub y8: [u32; 16],
}

const _: () = assert!(core::mem::size_of::<RegsColorPattern>() == 0x240);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsRenderSolid {
    pub prim_mode: u32,
    pub prim_color_format: u32,
    pub prim_color: u32,
    pub line_tie_break_bits: u32,
    _pad0: [u32; 0x14],
    pub prim_point_xy: u32,
    _pad1: [u32; 0x7],
    pub prim_point: [Point; 0x40],
}

const _: () = assert!(core::mem::size_of::<RegsRenderSolid>() == 0x280);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsPixelsFromCpu {
    pub data_type: u32,
    pub color_format: u32,
    pub index_format: u32,
    pub mono_format: u32,
    pub wrap: u32,
    pub color0: u32,
    pub color1: u32,
    pub mono_opacity: u32,
    _pad0: [u32; 0x6],
    pub src_width: u32,
    pub src_height: u32,
    pub dx_du_frac: u32,
    pub dx_du_int: u32,
    pub dx_dv_frac: u32,
    pub dy_dv_int: u32,
    pub dst_x0_frac: u32,
    pub dst_x0_int: u32,
    pub dst_y0_frac: u32,
    pub dst_y0_int: u32,
    pub data: u32,
}

const _: () = assert!(core::mem::size_of::<RegsPixelsFromCpu>() == 0x64);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsPixelsFromMemory {
    pub block_shape: RegsPixelsFromMemoryBlockShape,
    pub corral_size: RegsPixelsFromMemoryCorralSize,
    pub safe_overlap: RegsPixelsFromMemorySafeOverlap,
    pub sample_mode: RegsPixelsFromMemorySampleMode,
    _pad0: [u32; 0x8],
    pub dst_x0: i32,
    pub dst_y0: i32,
    pub dst_width: i32,
    pub dst_height: i32,
    pub du_dx: i64,
    pub dv_dy: i64,
    pub src_x0: i64,
    pub src_y0: i64,
}

const _: () = assert!(core::mem::size_of::<RegsPixelsFromMemory>() == 0x60);

bitfield! {
    pub struct RegsPixelsFromMemoryBlockShape {
        value: u32 @ 0, 3;
    }
}

bitfield! {
    pub struct RegsPixelsFromMemoryCorralSize {
        value: u32 @ 0, 5;
    }
}

bitfield! {
    pub struct RegsPixelsFromMemorySafeOverlap {
        value: u32 @ 0, 1;
    }
}

bitfield! {
    pub struct RegsPixelsFromMemorySampleMode {
        origin: Origin @ 0, 1;
        filter: Filter @ 4, 1;
    }
}

const _: () = {
    assert!(core::mem::offset_of!(Regs, object) == 0x0000);
    assert!(core::mem::offset_of!(Regs, no_operation) == 0x0100);
    assert!(core::mem::offset_of!(Regs, notify) == 0x0104);
    assert!(core::mem::offset_of!(Regs, wait_for_idle) == 0x0110);
    assert!(core::mem::offset_of!(Regs, pm_trigger) == 0x0140);
    assert!(core::mem::offset_of!(Regs, context_dma_notify) == 0x0180);
    assert!(core::mem::offset_of!(Regs, dst_context_dma) == 0x0184);
    assert!(core::mem::offset_of!(Regs, src_context_dma) == 0x0188);
    assert!(core::mem::offset_of!(Regs, semaphore_context_dma) == 0x018c);
    assert!(core::mem::offset_of!(Regs, dst) == 0x0200);
    assert!(core::mem::offset_of!(Regs, pixels_from_cpu_index_wrap) == 0x0228);
    assert!(core::mem::offset_of!(Regs, kind2d_check_enable) == 0x022c);
    assert!(core::mem::offset_of!(Regs, src) == 0x0230);
    assert!(core::mem::offset_of!(Regs, pixels_from_memory_sector_promotion) == 0x0258);
    assert!(core::mem::offset_of!(Regs, num_tpcs) == 0x0260);
    assert!(core::mem::offset_of!(Regs, render_enable_addr_upper) == 0x0264);
    assert!(core::mem::offset_of!(Regs, render_enable_addr_lower) == 0x0268);
    assert!(core::mem::offset_of!(Regs, render_enable_mode) == 0x026c);
    assert!(core::mem::offset_of!(Regs, clip_x0) == 0x0280);
    assert!(core::mem::offset_of!(Regs, clip_y0) == 0x0284);
    assert!(core::mem::offset_of!(Regs, clip_width) == 0x0288);
    assert!(core::mem::offset_of!(Regs, clip_height) == 0x028c);
    assert!(core::mem::offset_of!(Regs, clip_enable) == 0x0290);
    assert!(core::mem::offset_of!(Regs, color_key_format) == 0x0294);
    assert!(core::mem::offset_of!(Regs, color_key) == 0x0298);
    assert!(core::mem::offset_of!(Regs, color_key_enable) == 0x029c);
    assert!(core::mem::offset_of!(Regs, rop) == 0x02a0);
    assert!(core::mem::offset_of!(Regs, beta1) == 0x02a4);
    assert!(core::mem::offset_of!(Regs, beta4) == 0x02a8);
    assert!(core::mem::offset_of!(Regs, operation) == 0x02ac);
    assert!(core::mem::offset_of!(Regs, pattern_offset) == 0x02b0);
    assert!(core::mem::offset_of!(Regs, pattern_select) == 0x02b4);
    assert!(core::mem::offset_of!(Regs, monochrome_pattern) == 0x02e8);
    assert!(core::mem::offset_of!(Regs, color_pattern) == 0x0300);
    assert!(core::mem::offset_of!(Regs, render_solid) == 0x0580);
    assert!(core::mem::offset_of!(Regs, pixels_from_cpu) == 0x0800);
    assert!(core::mem::offset_of!(Regs, big_endian_control) == 0x0870);
    assert!(core::mem::offset_of!(Regs, pixels_from_memory) == 0x0880);
    assert!(core::mem::size_of::<Regs>() == NUM_REGS * 4);
};
