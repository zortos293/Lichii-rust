//! Kepler/Ampere compute engine (B1C0, C7C0) registers.
//!
//! Translated from the reference's register struct; every field offset there was checked against
//! its method number, and the offsets are asserted below.

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
    _pad1: [u32; 0x3f],
    pub launch_desc_loc: RegsLaunchDescLoc,
    _pad2: [u32; 0x1],
    pub launch: u32,
    _pad3: [u32; 0x4a7],
    pub tsc: RegsTsc,
    _pad4: [u32; 0x3],
    pub tic: RegsTic,
    _pad5: [u32; 0x22],
    pub code_loc: RegsCodeLoc,
    _pad6: [u32; 0x3fe],
    pub tex_cb_index: u32,
    _pad7: [u32; 0x375],
}

bitfield! {
    pub struct RegsExecUpload {
        linear: u32 @ 0, 1;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RegsLaunchDescLoc {
    pub address: u32,
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

const _: () = {
    assert!(core::mem::offset_of!(Regs, upload) == 0x0180);
    assert!(core::mem::offset_of!(Regs, exec_upload) == 0x01b0);
    assert!(core::mem::offset_of!(Regs, data_upload) == 0x01b4);
    assert!(core::mem::offset_of!(Regs, launch_desc_loc) == 0x02b4);
    assert!(core::mem::offset_of!(Regs, launch) == 0x02bc);
    assert!(core::mem::offset_of!(Regs, tsc) == 0x155c);
    assert!(core::mem::offset_of!(Regs, tic) == 0x1574);
    assert!(core::mem::offset_of!(Regs, code_loc) == 0x1608);
    assert!(core::mem::offset_of!(Regs, tex_cb_index) == 0x2608);
    assert!(core::mem::size_of::<Regs>() == NUM_REGS * 4);
};
