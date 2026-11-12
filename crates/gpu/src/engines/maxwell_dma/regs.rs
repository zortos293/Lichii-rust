//! Copy engine (B0B5, C7B5) registers.
//!
//! Translated from the reference's register struct; every field offset there was checked against
//! its method number, and the offsets are asserted below.

use crate::bits::{bitfield, open_enum};

pub const NUM_REGS: usize = 0x800;

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
    pub struct LaunchDmaDataTransferType {
        NONE = 0x0,
        PIPELINED = 0x1,
        NON_PIPELINED = 0x2,
    }
}

open_enum! {
    pub struct LaunchDmaSemaphoreType {
        NONE = 0x0,
        RELEASE_ONE_WORD_SEMAPHORE = 0x1,
        RELEASE_FOUR_WORD_SEMAPHORE = 0x2,
    }
}

open_enum! {
    pub struct LaunchDmaInterruptType {
        NONE = 0x0,
        BLOCKING = 0x1,
        NON_BLOCKING = 0x2,
    }
}

open_enum! {
    pub struct LaunchDmaMemoryLayout {
        BLOCKLINEAR = 0x0,
        PITCH = 0x1,
    }
}

open_enum! {
    pub struct LaunchDmaType {
        VIRTUAL = 0x0,
        PHYSICAL = 0x1,
    }
}

open_enum! {
    pub struct LaunchDmaSemaphoreReduction {
        IMIN = 0x0,
        IMAX = 0x1,
        IXOR = 0x2,
        IAND = 0x3,
        IOR = 0x4,
        IADD = 0x5,
        INC = 0x6,
        DEC = 0x7,
        FADD = 0xa,
    }
}

open_enum! {
    pub struct LaunchDmaSemaphoreReductionSign {
        SIGNED = 0x0,
        UNSIGNED = 0x1,
    }
}

open_enum! {
    pub struct LaunchDmaBypassL2 {
        USE_PTE_SETTING = 0x0,
        FORCE_VOLATILE = 0x1,
    }
}

open_enum! {
    pub struct RemapConstSwizzle {
        SRC_X = 0x0,
        SRC_Y = 0x1,
        SRC_Z = 0x2,
        SRC_W = 0x3,
        CONST_A = 0x4,
        CONST_B = 0x5,
        NO_WRITE = 0x6,
    }
}

bitfield! {
    pub struct DmaOrigin {
        x: u32 @ 0, 16;
        y: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct DmaBlockSize {
        width: u32 @ 0, 4;
        height: u32 @ 4, 4;
        depth: u32 @ 8, 4;
        gob_height: u32 @ 12, 4;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DmaParameters {
    pub block_size: DmaBlockSize,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub layer: u32,
    pub origin: DmaOrigin,
}

const _: () = assert!(core::mem::size_of::<DmaParameters>() == 0x18);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedGpuvAddr {
    pub upper: u32,
    pub lower: u32,
}

impl PackedGpuvAddr {
    /// The GPU virtual address the two words hold.
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.upper) << 32) | u64::from(self.lower)
    }
}

const _: () = assert!(core::mem::size_of::<PackedGpuvAddr>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Semaphore {
    pub address: PackedGpuvAddr,
    pub payload: u32,
}

const _: () = assert!(core::mem::size_of::<Semaphore>() == 0xc);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RenderEnable {
    pub address: PackedGpuvAddr,
    pub mode: RenderEnableModeReg,
}

const _: () = assert!(core::mem::size_of::<RenderEnable>() == 0xc);

bitfield! {
    pub struct RenderEnableModeReg {
        value: RenderEnableMode @ 0, 3;
    }
}

bitfield! {
    pub struct LaunchDma {
        data_transfer_type: LaunchDmaDataTransferType @ 0, 2;
        flush_enable: u32 @ 2, 1;
        semaphore_type: LaunchDmaSemaphoreType @ 3, 2;
        interrupt_type: LaunchDmaInterruptType @ 5, 2;
        src_memory_layout: LaunchDmaMemoryLayout @ 7, 1;
        dst_memory_layout: LaunchDmaMemoryLayout @ 8, 1;
        multi_line_enable: u32 @ 9, 1;
        remap_enable: u32 @ 10, 1;
        rmwdisable: u32 @ 11, 1;
        src_type: LaunchDmaType @ 12, 1;
        dst_type: LaunchDmaType @ 13, 1;
        semaphore_reduction: LaunchDmaSemaphoreReduction @ 14, 4;
        semaphore_reduction_sign: LaunchDmaSemaphoreReductionSign @ 18, 1;
        reduction_enable: u32 @ 19, 1;
        bypass_l2: LaunchDmaBypassL2 @ 20, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RemapConst {
    pub remap_consta_value: u32,
    pub remap_constb_value: u32,
    pub bits: RemapConstBits,
}

impl RemapConst {
    #[inline]
    pub fn dst_components_raw(&self) -> u32 {
        self.bits.dst_components_raw()
    }

    #[inline]
    pub fn dst_x(&self) -> RemapConstSwizzle {
        self.bits.dst_x()
    }

    #[inline]
    pub fn dst_y(&self) -> RemapConstSwizzle {
        self.bits.dst_y()
    }

    #[inline]
    pub fn dst_z(&self) -> RemapConstSwizzle {
        self.bits.dst_z()
    }

    #[inline]
    pub fn dst_w(&self) -> RemapConstSwizzle {
        self.bits.dst_w()
    }

    #[inline]
    pub fn component_size_minus_one(&self) -> u32 {
        self.bits.component_size_minus_one()
    }

    #[inline]
    pub fn num_src_components_minus_one(&self) -> u32 {
        self.bits.num_src_components_minus_one()
    }

    #[inline]
    pub fn num_dst_components_minus_one(&self) -> u32 {
        self.bits.num_dst_components_minus_one()
    }
}

const _: () = assert!(core::mem::size_of::<RemapConst>() == 0xc);

bitfield! {
    pub struct RemapConstBits {
        dst_components_raw: u32 @ 0, 12;
        dst_x: RemapConstSwizzle @ 0, 3;
        dst_y: RemapConstSwizzle @ 4, 3;
        dst_z: RemapConstSwizzle @ 8, 3;
        dst_w: RemapConstSwizzle @ 12, 3;
        component_size_minus_one: u32 @ 16, 2;
        num_src_components_minus_one: u32 @ 20, 2;
        num_dst_components_minus_one: u32 @ 24, 2;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Regs {
    _pad0: [u32; 0x40],
    pub nop: u32,
    _pad1: [u32; 0xf],
    pub pm_trigger: u32,
    _pad2: [u32; 0x3f],
    pub semaphore: Semaphore,
    _pad3: [u32; 0x2],
    pub render_enable: RenderEnable,
    pub src_phys_mode: u32,
    pub dst_phys_mode: u32,
    _pad4: [u32; 0x26],
    pub launch_dma: LaunchDma,
    _pad5: [u32; 0x3f],
    pub offset_in: PackedGpuvAddr,
    pub offset_out: PackedGpuvAddr,
    pub pitch_in: i32,
    pub pitch_out: i32,
    pub line_length_in: u32,
    pub line_count: u32,
    _pad6: [u32; 0xb8],
    pub remap_const: RemapConst,
    pub dst_params: DmaParameters,
    _pad7: [u32; 0x1],
    pub src_params: DmaParameters,
    _pad8: [u32; 0x275],
    pub pm_trigger_end: u32,
    _pad9: [u32; 0x3ba],
}

const _: () = {
    assert!(core::mem::offset_of!(Regs, nop) == 0x0100);
    assert!(core::mem::offset_of!(Regs, pm_trigger) == 0x0140);
    assert!(core::mem::offset_of!(Regs, semaphore) == 0x0240);
    assert!(core::mem::offset_of!(Regs, render_enable) == 0x0254);
    assert!(core::mem::offset_of!(Regs, src_phys_mode) == 0x0260);
    assert!(core::mem::offset_of!(Regs, dst_phys_mode) == 0x0264);
    assert!(core::mem::offset_of!(Regs, launch_dma) == 0x0300);
    assert!(core::mem::offset_of!(Regs, offset_in) == 0x0400);
    assert!(core::mem::offset_of!(Regs, offset_out) == 0x0408);
    assert!(core::mem::offset_of!(Regs, pitch_in) == 0x0410);
    assert!(core::mem::offset_of!(Regs, pitch_out) == 0x0414);
    assert!(core::mem::offset_of!(Regs, line_length_in) == 0x0418);
    assert!(core::mem::offset_of!(Regs, line_count) == 0x041c);
    assert!(core::mem::offset_of!(Regs, remap_const) == 0x0700);
    assert!(core::mem::offset_of!(Regs, dst_params) == 0x070c);
    assert!(core::mem::offset_of!(Regs, src_params) == 0x0728);
    assert!(core::mem::offset_of!(Regs, pm_trigger_end) == 0x1114);
    assert!(core::mem::size_of::<Regs>() == NUM_REGS * 4);
};
