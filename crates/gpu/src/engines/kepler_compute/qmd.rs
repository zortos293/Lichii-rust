//! Compute launch descriptors (QMD): the 256-byte structure LAUNCH points at.
//!
//! Maxwell/Pascal compute (B1C0) uses the version 1.x layout ([`QmdV2`]); Ampere native compute
//! (C7C0, NVN2) uses version 3.0 ([`QmdV3`]). Both are plain words in GPU memory, read as a
//! whole and decoded through the typed fields below. Word offsets are asserted against the
//! reference (`kepler_compute.h` LaunchParams and `native_qmd.h`).

use crate::backend::ConstBufferBinding;
use crate::bits::bitfield;
use crate::host::Reduction;
use lichii_common::GpuVAddr;

/// Words in a launch descriptor.
pub const QMD_WORDS: usize = 64;

/// The version byte (word 18, bits 7:0) of a version 3.0 descriptor.
pub const V3_VERSION: u32 = 0x30;

/// Why a descriptor cannot be launched.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QmdError {
    /// Word 18 does not carry the version 3.0 tag.
    #[error("QMD version byte {0:#x} is not 3.0")]
    UnsupportedVersion(u32),
    /// The descriptor targets a hardware queue the emulation does not drive.
    #[error("QMD launches through a queue")]
    Queue,
    /// A constant buffer address has bits above the GPU address width.
    #[error("QMD constant buffer {0} address exceeds the GPU address width")]
    ConstBufferAddress(usize),
}

/// A launch decoded from either descriptor version.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// GPU address of the program's first instruction.
    pub program: GpuVAddr,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_memory_size: u32,
    pub const_buffers: [ConstBufferBinding; 8],
    /// Samplers are indexed with the texture header index.
    pub linked_samplers: bool,
    /// Per-thread local memory in bytes (positive plus call-return-stack allocation).
    pub local_memory_size: u32,
}

bitfield! {
    pub struct V2Flags {
        linked_samplers: bool @ 30, 1;
    }
}

bitfield! {
    pub struct V2GridX {
        x: u32 @ 0, 31;
    }
}

bitfield! {
    pub struct V2Dim16 {
        low: u32 @ 0, 16;
        high: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct V2Shared {
        size: u32 @ 0, 18;
    }
}

bitfield! {
    pub struct V2BlockX {
        x: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct V2ConstBufferMask {
        enable_mask: u32 @ 0, 8;
    }
}

bitfield! {
    pub struct V2ConstBufferBits {
        address_high: u32 @ 0, 8;
        size: u32 @ 15, 17;
    }
}

/// A constant buffer entry of a version 1.x descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct V2ConstBuffer {
    pub address_low: u32,
    pub bits: V2ConstBufferBits,
}

/// A Maxwell/Pascal launch descriptor (version 1.x).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QmdV2 {
    _pad0: [u32; 8],
    /// Offset of the program from the code location register.
    pub program_start: u32,
    _pad1: [u32; 2],
    pub flags: V2Flags,
    pub grid_x: V2GridX,
    /// Grid height in bits 15:0, depth in bits 31:16.
    pub grid_yz: V2Dim16,
    _pad2: [u32; 3],
    pub shared: V2Shared,
    pub block_x: V2BlockX,
    /// Block height in bits 15:0, depth in bits 31:16.
    pub block_yz: V2Dim16,
    pub const_buffer_mask: V2ConstBufferMask,
    _pad3: [u32; 8],
    pub const_buffers: [V2ConstBuffer; 8],
    pub local_pos_alloc: u32,
    pub local_neg_alloc: u32,
    pub local_crs_alloc: u32,
    _pad4: [u32; 16],
}

const _: () = {
    assert!(core::mem::size_of::<QmdV2>() == QMD_WORDS * 4);
    assert!(core::mem::offset_of!(QmdV2, program_start) == 0x8 * 4);
    assert!(core::mem::offset_of!(QmdV2, flags) == 0xb * 4);
    assert!(core::mem::offset_of!(QmdV2, grid_x) == 0xc * 4);
    assert!(core::mem::offset_of!(QmdV2, shared) == 0x11 * 4);
    assert!(core::mem::offset_of!(QmdV2, block_x) == 0x12 * 4);
    assert!(core::mem::offset_of!(QmdV2, const_buffer_mask) == 0x14 * 4);
    assert!(core::mem::offset_of!(QmdV2, const_buffers) == 0x1d * 4);
};

impl QmdV2 {
    /// The launch, with the program at `code_loc` (the compute engine's code location register)
    /// plus the descriptor's offset.
    pub fn launch(&self, code_loc: GpuVAddr) -> Launch {
        let mask = self.const_buffer_mask.enable_mask();
        let mut const_buffers = [ConstBufferBinding::default(); 8];
        for (slot, (binding, entry)) in const_buffers
            .iter_mut()
            .zip(&self.const_buffers)
            .enumerate()
        {
            *binding = ConstBufferBinding {
                address: (u64::from(entry.bits.address_high()) << 32)
                    | u64::from(entry.address_low),
                size: entry.bits.size(),
                enabled: mask & (1 << slot) != 0,
            };
        }
        Launch {
            program: code_loc.wrapping_add(u64::from(self.program_start)),
            grid: [self.grid_x.x(), self.grid_yz.low(), self.grid_yz.high()],
            block: [self.block_x.x(), self.block_yz.low(), self.block_yz.high()],
            shared_memory_size: self.shared.size(),
            const_buffers,
            linked_samplers: self.flags.linked_samplers(),
            local_memory_size: (self.local_pos_alloc & 0xf_ffff)
                + (self.local_crs_alloc & 0xf_ffff),
        }
    }
}

bitfield! {
    /// Word 4: the hardware queue bit, whether an inline load waits for a scheduling PCAS, and
    /// the dependence counter a scheduled QMD waits on.
    pub struct V3Control {
        queue: bool @ 8, 1;
        require_scheduling_pcas: bool @ 12, 1;
        dependence_counter: u32 @ 14, 16;
    }
}

bitfield! {
    /// Word 16: what the descriptor does to the QMD at `dependent_qmd_pointer` once it ran.
    pub struct V3Dependent {
        enable: bool @ 0, 1;
        action: u32 @ 1, 3;
        second_enable: bool @ 5, 1;
    }
}

bitfield! {
    pub struct V3Dim16 {
        value: u32 @ 0, 16;
    }
}

bitfield! {
    pub struct V3Shared {
        size: u32 @ 0, 18;
    }
}

bitfield! {
    pub struct V3VersionBlockX {
        version: u32 @ 0, 8;
        block_x: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct V3BlockYz {
        y: u32 @ 0, 16;
        z: u32 @ 16, 16;
    }
}

bitfield! {
    pub struct V3ConstBufferMask {
        enable_mask: u32 @ 0, 8;
    }
}

bitfield! {
    pub struct V3ConstBufferBits {
        address_high: u32 @ 0, 8;
        address_overflow: u32 @ 8, 9;
        size_shifted4: u32 @ 19, 13;
    }
}

bitfield! {
    pub struct V3ProgramHigh {
        address_high: u32 @ 0, 17;
    }
}

bitfield! {
    /// Release flags: 7:0 address high, 19 max release, 22:20 reduction operation, 23 enable,
    /// 25:24 reduction format, 26 reduction enable, 29 64-bit payload, 31:30 structure size.
    pub struct V3ReleaseFlags {
        address_high: u32 @ 0, 8;
        max: bool @ 19, 1;
        reduction_op: u32 @ 20, 3;
        enable: bool @ 23, 1;
        reduction_format: u32 @ 24, 2;
        reduction_enable: bool @ 26, 1;
        payload_64: bool @ 29, 1;
        structure_size: u32 @ 30, 2;
    }
}

/// A constant buffer entry of a version 3.0 descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct V3ConstBuffer {
    pub address_low: u32,
    pub bits: V3ConstBufferBits,
}

/// A semaphore release slot of a version 3.0 descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct V3Release {
    pub address_low: u32,
    pub flags: V3ReleaseFlags,
    pub payload: u32,
    pub payload_high: u32,
}

/// What a release writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReleaseKind {
    /// The payload, as one word.
    Word,
    /// The payload, then a 64-bit GPU timestamp (a four-word structure).
    WordWithTimestamp,
    /// The payload combined with the word at the address.
    Reduction { op: Reduction, signed: bool },
    /// The larger of the payload and the word at the address.
    Max,
}

/// An enabled, supported release slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    pub address: GpuVAddr,
    pub payload: u32,
    pub kind: ReleaseKind,
}

impl V3Release {
    /// Whether the slot asks for a release.
    pub fn enabled(&self) -> bool {
        self.flags.enable()
    }

    /// The release, or `None` for a format the reference does not decode (64-bit payloads, two
    /// word structures, reductions that are not one word or not 32-bit).
    pub fn decode(&self) -> Option<Release> {
        let flags = self.flags;
        let size = flags.structure_size();
        let kind = if flags.payload_64() {
            return None;
        } else if flags.reduction_enable() {
            if size != 1 || flags.reduction_format() >= 2 {
                return None;
            }
            ReleaseKind::Reduction {
                op: semaphore_reduction(flags.reduction_op()),
                signed: flags.reduction_format() == 1,
            }
        } else if size == 0 {
            ReleaseKind::WordWithTimestamp
        } else if size == 1 && flags.max() {
            ReleaseKind::Max
        } else if size == 1 {
            ReleaseKind::Word
        } else {
            return None;
        };
        Some(Release {
            address: (u64::from(flags.address_high()) << 32) | u64::from(self.address_low),
            payload: self.payload,
            kind,
        })
    }
}

/// The reduction a 3-bit semaphore reduction field selects, in the order QMD releases and the
/// compute report semaphore share (ADD, MIN, MAX, INC, DEC, AND, OR, XOR).
pub fn semaphore_reduction(op: u32) -> Reduction {
    match op & 7 {
        0 => Reduction::Add,
        1 => Reduction::Min,
        2 => Reduction::Max,
        3 => Reduction::Inc,
        4 => Reduction::Dec,
        5 => Reduction::And,
        6 => Reduction::Or,
        _ => Reduction::Xor,
    }
}

/// An Ampere launch descriptor (version 3.0).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QmdV3 {
    _pad0: [u32; 4],
    pub control: V3Control,
    _pad1: [u32; 7],
    pub grid_width: u32,
    pub grid_height: V3Dim16,
    pub grid_depth: V3Dim16,
    /// The dependent QMD's address shifted right by 8.
    pub dependent_qmd_pointer: u32,
    pub dependent: V3Dependent,
    pub shared: V3Shared,
    pub version_block_x: V3VersionBlockX,
    pub block_yz: V3BlockYz,
    pub const_buffer_mask: V3ConstBufferMask,
    _pad3: [u32; 2],
    pub local_pos_alloc: u32,
    /// Release slots 0 and 1.
    pub releases: [V3Release; 2],
    pub const_buffers: [V3ConstBuffer; 8],
    pub program_low: u32,
    pub program_high: V3ProgramHigh,
    pub local_crs_alloc: u32,
    _pad4: u32,
    /// Release slot 2, patched by NVN2 for queue-to-queue syncs.
    pub release2: V3Release,
    _pad5: [u32; 8],
}

const _: () = {
    assert!(core::mem::size_of::<QmdV3>() == QMD_WORDS * 4);
    assert!(core::mem::offset_of!(QmdV3, control) == 4 * 4);
    assert!(core::mem::offset_of!(QmdV3, grid_width) == 12 * 4);
    assert!(core::mem::offset_of!(QmdV3, dependent_qmd_pointer) == 15 * 4);
    assert!(core::mem::offset_of!(QmdV3, dependent) == 16 * 4);
    assert!(core::mem::offset_of!(QmdV3, shared) == 17 * 4);
    assert!(core::mem::offset_of!(QmdV3, version_block_x) == 18 * 4);
    assert!(core::mem::offset_of!(QmdV3, const_buffer_mask) == 20 * 4);
    assert!(core::mem::offset_of!(QmdV3, local_pos_alloc) == 23 * 4);
    assert!(core::mem::offset_of!(QmdV3, releases) == 24 * 4);
    assert!(core::mem::offset_of!(QmdV3, const_buffers) == 32 * 4);
    assert!(core::mem::offset_of!(QmdV3, program_low) == 48 * 4);
    assert!(core::mem::offset_of!(QmdV3, local_crs_alloc) == 50 * 4);
    assert!(core::mem::offset_of!(QmdV3, release2) == 52 * 4);
};

impl QmdV3 {
    /// Whether the descriptor can launch at all (version 3.0, not queue-launched). Nothing of
    /// it, releases included, is acted on otherwise.
    pub fn check(&self) -> Result<(), QmdError> {
        match self.version_block_x.version() {
            V3_VERSION => {}
            version => return Err(QmdError::UnsupportedVersion(version)),
        }
        if self.control.queue() {
            return Err(QmdError::Queue);
        }
        Ok(())
    }

    /// The dispatch the descriptor describes.
    pub fn launch(&self) -> Result<Launch, QmdError> {
        let mask = self.const_buffer_mask.enable_mask();
        let mut const_buffers = [ConstBufferBinding::default(); 8];
        for (slot, (binding, entry)) in const_buffers
            .iter_mut()
            .zip(&self.const_buffers)
            .enumerate()
        {
            if entry.bits.address_overflow() != 0 {
                return Err(QmdError::ConstBufferAddress(slot));
            }
            *binding = ConstBufferBinding {
                address: (u64::from(entry.bits.address_high()) << 32)
                    | u64::from(entry.address_low),
                size: entry.bits.size_shifted4() << 4,
                enabled: mask & (1 << slot) != 0,
            };
        }
        Ok(Launch {
            program: (u64::from(self.program_high.address_high()) << 32)
                | u64::from(self.program_low),
            grid: [
                self.grid_width,
                self.grid_height.value(),
                self.grid_depth.value(),
            ],
            block: [
                self.version_block_x.block_x(),
                self.block_yz.y(),
                self.block_yz.z(),
            ],
            shared_memory_size: self.shared.size(),
            const_buffers,
            linked_samplers: false,
            local_memory_size: (self.local_pos_alloc & 0xff_ffff)
                + (self.local_crs_alloc & 0xff_ffff),
        })
    }

    /// The three release slots, in the order the reference performs them (words 24, 28, 52).
    pub fn release_slots(&self) -> [&V3Release; 3] {
        [&self.releases[0], &self.releases[1], &self.release2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_decodes_address_and_kind() {
        let words = [0x2000u32, (1 << 23) | (1 << 30) | 0x1f, 9, 0];
        let release = bytemuck::cast_ref::<[u32; 4], V3Release>(&words);
        assert!(release.enabled());
        assert_eq!(
            release.decode(),
            Some(Release {
                address: 0x1f_0000_2000,
                payload: 9,
                kind: ReleaseKind::Word,
            })
        );
        let reduction = [0u32, (1 << 23) | (1 << 26) | (1 << 30) | (3 << 20), 1, 0];
        let release = bytemuck::cast_ref::<[u32; 4], V3Release>(&reduction);
        assert_eq!(
            release.decode().map(|r| r.kind),
            Some(ReleaseKind::Reduction {
                op: Reduction::Inc,
                signed: false
            })
        );
    }

    #[test]
    fn local_memory_adds_the_positive_and_call_stack_allocations() {
        let mut words = [0u32; QMD_WORDS];
        words[0x2d] = 0xfff0_0100;
        words[0x2f] = 0x40;
        let v2 = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV2>(&words);
        assert_eq!(v2.launch(0).local_memory_size, 0x140);

        let mut words = [0u32; QMD_WORDS];
        words[23] = 0xff10_0000;
        words[50] = 0x20;
        let v3 = bytemuck::cast_ref::<[u32; QMD_WORDS], QmdV3>(&words);
        assert_eq!(v3.launch().unwrap().local_memory_size, 0x10_0020);
    }
}
