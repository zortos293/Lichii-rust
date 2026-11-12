//! Where an NVN2 program starts relative to its header.
//!
//! An NVN2 graphics program is bound by a full GPU address: the pipeline register's two
//! reserved words hold bits 32..40 (low byte of the first) and 0..32 (the second) of a
//! 128-byte header. Its first 0x50 bytes are a Maxwell-layout shader program header (SPH); the
//! first instruction follows the whole header, at `header + 0x80`. Compute programs have no
//! header: the first instruction is at the code base plus the launch descriptor's program
//! start.

use lichii_common::GpuVAddr;

use crate::ShaderEnvironment;

/// Bytes between an NVN2 graphics program's header and its first instruction.
pub const GRAPHICS_HEADER_SIZE: u64 = 0x80;
/// Bytes of the Maxwell-layout program header at the start of the NVN2 header.
pub const SPH_SIZE: u64 = 0x50;

/// Address of a graphics program's header from the two reserved words of its pipeline
/// register.
pub const fn graphics_header_address(reserved0: u32, reserved1: u32) -> GpuVAddr {
    (((reserved0 & 0xff) as u64) << 32) | reserved1 as u64
}

/// Address of the first instruction of the graphics program whose header is at `header`.
pub const fn graphics_start_address(header: GpuVAddr) -> GpuVAddr {
    header + GRAPHICS_HEADER_SIZE
}

/// Reads the shader program header of the graphics program `env` translates, as the 20
/// little-endian words the Maxwell header layout describes.
pub fn read_sph(env: &dyn ShaderEnvironment) -> [u32; 20] {
    let header = env.start_address() - GRAPHICS_HEADER_SIZE;
    std::array::from_fn(|word| {
        let value = env.read_u64(header + word as u64 / 2 * 8);
        (value >> (word % 2 * 32)) as u32
    })
}
