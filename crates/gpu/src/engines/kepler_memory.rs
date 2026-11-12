//! The inline-to-memory engine (KeplerMemory, A140). NVN uses it on subchannel 2 without binding
//! a class, so channels bind it there by default (NX2_NOTES row 49).

use super::EngineContext;
use super::upload::{Upload, UploadRegisters};
use crate::bits::bitfield;
use crate::reg_index;

pub mod regs {
    use super::{UploadRegisters, bitfield};

    pub const NUM_REGS: usize = 0x7f;

    #[repr(C)]
    #[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct Regs {
        _pad0: [u32; 0x60],
        pub upload: UploadRegisters,
        pub exec: RegsExec,
        pub data: u32,
        _pad1: [u32; 0x11],
    }

    bitfield! {
        pub struct RegsExec {
            linear: u32 @ 0, 1;
        }
    }

    const _: () = {
        assert!(core::mem::offset_of!(Regs, upload) == 0x0180);
        assert!(core::mem::offset_of!(Regs, exec) == 0x01b0);
        assert!(core::mem::offset_of!(Regs, data) == 0x01b4);
        assert!(core::mem::size_of::<Regs>() == NUM_REGS * 4);
    };
}

pub use regs::Regs;

pub struct KeplerMemory {
    pub regs: Box<Regs>,
    upload: Upload,
}

impl Default for KeplerMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl KeplerMemory {
    pub fn new() -> Self {
        Self {
            regs: bytemuck::zeroed_box(),
            upload: Upload::default(),
        }
    }

    pub fn call_method(&mut self, ctx: &mut EngineContext<'_>, method: u32, argument: u32) {
        let Some(word) =
            bytemuck::cast_slice_mut::<Regs, u32>(std::slice::from_mut(&mut *self.regs))
                .get_mut(method as usize)
        else {
            return;
        };
        *word = argument;
        if method == reg_index!(Regs, exec) {
            self.upload
                .exec(&self.regs.upload, self.regs.exec.linear() != 0);
        } else if method == reg_index!(Regs, data) {
            self.upload.data(ctx, argument);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::testing::Harness;

    fn write(engine: &mut KeplerMemory, h: &mut Harness, method: u32, value: u32) {
        let mut ctx = h.context();
        engine.call_method(&mut ctx, method, value);
    }

    fn setup(engine: &mut KeplerMemory, h: &mut Harness, address: u64, line: u32, lines: u32) {
        let upload = reg_index!(Regs, upload);
        write(engine, h, upload, line);
        write(engine, h, upload + 1, lines);
        write(engine, h, upload + 2, (address >> 32) as u32);
        write(engine, h, upload + 3, address as u32);
    }

    #[test]
    fn linear_upload_writes_lines_at_their_pitch() {
        let mut h = Harness::new();
        let mut engine = KeplerMemory::new();
        setup(&mut engine, &mut h, 0x10_0000, 6, 2);
        write(
            &mut engine,
            &mut h,
            reg_index!(Regs, upload.dest.pitch),
            0x100,
        );
        write(&mut engine, &mut h, reg_index!(Regs, exec), 1);
        for word in [0x0403_0201u32, 0x0807_0605, 0x0c0b_0a09] {
            write(&mut engine, &mut h, reg_index!(Regs, data), word);
        }
        let mut first = [0u8; 8];
        h.memory.read_block(0x10_0000, &mut first).unwrap();
        assert_eq!(first, [1, 2, 3, 4, 5, 6, 0, 0]);
        let mut second = [0u8; 6];
        h.memory.read_block(0x10_0100, &mut second).unwrap();
        assert_eq!(second, [7, 8, 9, 10, 11, 12]);
        assert_eq!(
            h.rasterizer.inline_writes,
            vec![(0x10_0000, 6), (0x10_0100, 6)]
        );
    }

    #[test]
    fn block_linear_upload_swizzles_into_place() {
        let mut h = Harness::new();
        let mut engine = KeplerMemory::new();
        setup(&mut engine, &mut h, 0x20_0000, 64, 2);
        let dest = reg_index!(Regs, upload.dest);
        write(&mut engine, &mut h, dest + 3, 0);
        write(&mut engine, &mut h, dest + 4, 64);
        write(&mut engine, &mut h, dest + 5, 8);
        write(&mut engine, &mut h, dest + 6, 1);
        write(&mut engine, &mut h, dest + 9, 1);
        write(&mut engine, &mut h, reg_index!(Regs, exec), 0);
        for i in 0..32u32 {
            write(&mut engine, &mut h, reg_index!(Regs, data), i);
        }
        let mut gob = [0u8; 512];
        h.memory.read_block(0x20_0000, &mut gob).unwrap();
        let row1 = lichii_texture::block_linear::gob_offset(0, 1, Default::default()) as usize;
        assert_eq!(&gob[row1..row1 + 4], &0u32.to_le_bytes());
        assert_eq!(&gob[row1 + 4..row1 + 8], &1u32.to_le_bytes());
        let row2 = lichii_texture::block_linear::gob_offset(0, 2, Default::default()) as usize;
        assert_eq!(&gob[row2..row2 + 4], &16u32.to_le_bytes());
    }

    #[test]
    fn upload_past_the_surface_is_dropped() {
        let mut h = Harness::new();
        let mut engine = KeplerMemory::new();
        setup(&mut engine, &mut h, 0x20_0000, 4, 1);
        let dest = reg_index!(Regs, upload.dest);
        write(&mut engine, &mut h, dest + 4, 64);
        write(&mut engine, &mut h, dest + 5, 8);
        write(&mut engine, &mut h, dest + 9, u32::MAX);
        write(&mut engine, &mut h, reg_index!(Regs, exec), 0);
        write(&mut engine, &mut h, reg_index!(Regs, data), 0xdead_beef);
        assert!(h.rasterizer.inline_writes.is_empty());
    }

    #[test]
    fn long_linear_upload_streams_in_chunks() {
        let mut h = Harness::new();
        let mut engine = KeplerMemory::new();
        let words = 0x1800u32;
        setup(&mut engine, &mut h, 0x30_0000, words * 4, 1);
        write(&mut engine, &mut h, reg_index!(Regs, exec), 1);
        for i in 0..words {
            write(&mut engine, &mut h, reg_index!(Regs, data), i);
        }
        let mut out = vec![0u32; words as usize];
        h.memory
            .read_block(0x30_0000, bytemuck::cast_slice_mut(&mut out))
            .unwrap();
        assert!(out.iter().enumerate().all(|(i, &w)| w == i as u32));
        let chunks: Vec<_> = (0..6).map(|i| (0x30_0000 + i * 0x1000, 0x1000)).collect();
        assert_eq!(h.rasterizer.inline_writes, chunks);
    }
}
