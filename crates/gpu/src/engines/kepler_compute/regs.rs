//! Kepler/Ampere compute engine (B1C0, C7C0) registers.
//!
//! Translated from the reference's register struct; every field offset there was checked against
//! its method number, and the offsets are asserted below. The C7C0-only registers (PCAS, inline
//! QMD, report semaphore) take their offsets from NVIDIA's `clc7c0.h`.

use crate::bits::bitfield;
use crate::engines::upload::UploadRegisters;

pub const NUM_CONST_BUFFERS: usize = 0x8;
pub const NUM_REGS: usize = 0xcf8;
pub const NUM_LAUNCH_PARAMETERS: usize = 0x40;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Regs {
    _pad0: [u32; 0x60],
    pub upload: UploadRegisters,
    pub exec_upload: RegsExecUpload,
    pub data_upload: u32,
    _pad1a: [u32; 0x39],
    /// SET_QMD_VIRTUALIZATION_CONTROL (C7C0).
    pub qmd_virtualization_control: RegsQmdVirtualizationControl,
    _pad1b: [u32; 0x5],
    /// LAUNCH_DESC_LOC; SEND_PCAS_A (the QMD address shifted right by 8) on C7C0.
    pub launch_desc_loc: RegsLaunchDescLoc,
    _pad2: [u32; 0x1],
    /// LAUNCH; SEND_SIGNALING_PCAS_B on C7C0.
    pub launch: u32,
    /// SEND_SIGNALING_PCAS2_B (C7C0).
    pub send_signaling_pcas2_b: u32,
    _pad3a: [u32; 0x15],
    /// SET_INLINE_QMD_ADDRESS_A/B (C7C0).
    pub inline_qmd_address: RegsInlineQmdAddress,
    /// LOAD_INLINE_QMD_DATA (C7C0): the descriptor itself; its last word loads it.
    pub inline_qmd: [u32; 0x40],
    _pad3b: [u32; 0x44f],
    pub tsc: RegsTsc,
    _pad4: [u32; 0x3],
    pub tic: RegsTic,
    _pad5: [u32; 0x22],
    pub code_loc: RegsCodeLoc,
    _pad6a: [u32; 0x13c],
    /// SET_REPORT_SEMAPHORE_A-D (C7C0).
    pub report_semaphore: RegsReportSemaphore,
    _pad6b: [u32; 0x2be],
    pub tex_cb_index: u32,
    _pad7: [u32; 0x375],
}

bitfield! {
    pub struct RegsExecUpload {
        linear: u32 @ 0, 1;
    }
}

bitfield! {
    /// `send_pcas` schedules an inline QMD once loaded, whatever its REQUIRE_SCHEDULING_PCAS says.
    pub struct RegsQmdVirtualizationControl {
        send_pcas: bool @ 24, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsLaunchDescLoc {
    pub address: u32,
}

impl RegsLaunchDescLoc {
    /// GPU address of the launch descriptor (the register holds it shifted right by 8).
    #[inline]
    pub fn address(&self) -> u64 {
        u64::from(self.address) << 8
    }
}

const _: () = assert!(core::mem::size_of::<RegsLaunchDescLoc>() == 0x4);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsTsc {
    pub address_high: u32,
    pub address_low: u32,
    pub limit: u32,
}

impl RegsTsc {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<RegsTsc>() == 0xc);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsTic {
    pub address_high: u32,
    pub address_low: u32,
    pub limit: u32,
}

impl RegsTic {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<RegsTic>() == 0xc);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsCodeLoc {
    pub address_high: u32,
    pub address_low: u32,
}

impl RegsCodeLoc {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high) << 32) | u64::from(self.address_low)
    }
}

const _: () = assert!(core::mem::size_of::<RegsCodeLoc>() == 0x8);

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsInlineQmdAddress {
    pub shifted8_upper: u32,
    pub shifted8_lower: u32,
}

impl RegsInlineQmdAddress {
    /// GPU address the inline descriptor is written to.
    #[inline]
    pub fn address(&self) -> u64 {
        ((u64::from(self.shifted8_upper) << 32) | u64::from(self.shifted8_lower)) << 8
    }
}

bitfield! {
    /// SET_REPORT_SEMAPHORE_D: written last, it performs the semaphore operation.
    pub struct RegsReportSemaphoreControl {
        operation: u32 @ 0, 2;
        reduction_enable: bool @ 3, 1;
        reduction_op: u32 @ 9, 3;
        reduction_signed: bool @ 17, 1;
        one_word: bool @ 28, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsReportSemaphore {
    pub address_high: u32,
    pub address_low: u32,
    pub payload: u32,
    pub control: RegsReportSemaphoreControl,
}

impl RegsReportSemaphore {
    #[inline]
    pub fn address(&self) -> u64 {
        (u64::from(self.address_high & 0x1ff_ffff) << 32) | u64::from(self.address_low)
    }
}

const _: () = {
    assert!(core::mem::offset_of!(Regs, upload) == 0x0180);
    assert!(core::mem::offset_of!(Regs, exec_upload) == 0x01b0);
    assert!(core::mem::offset_of!(Regs, data_upload) == 0x01b4);
    assert!(core::mem::offset_of!(Regs, qmd_virtualization_control) == 0x029c);
    assert!(core::mem::offset_of!(Regs, launch_desc_loc) == 0x02b4);
    assert!(core::mem::offset_of!(Regs, launch) == 0x02bc);
    assert!(core::mem::offset_of!(Regs, send_signaling_pcas2_b) == 0x02c0);
    assert!(core::mem::offset_of!(Regs, inline_qmd_address) == 0x0318);
    assert!(core::mem::offset_of!(Regs, inline_qmd) == 0x0320);
    assert!(core::mem::offset_of!(Regs, tsc) == 0x155c);
    assert!(core::mem::offset_of!(Regs, tic) == 0x1574);
    assert!(core::mem::offset_of!(Regs, code_loc) == 0x1608);
    assert!(core::mem::offset_of!(Regs, report_semaphore) == 0x1b00);
    assert!(core::mem::offset_of!(Regs, tex_cb_index) == 0x2608);
    assert!(core::mem::size_of::<Regs>() == NUM_REGS * 4);
};
