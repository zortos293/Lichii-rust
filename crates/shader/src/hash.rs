//! Stable 64-bit hashing for module cache keys: the same inputs give the same hash in every
//! run and build, unlike `std`'s randomly seeded hashers.

use std::cell::Cell;
use std::hash::{Hash, Hasher};

use lichii_common::GpuVAddr;

use crate::info::{HandleSource, ReplaceConstant, TexturePixelFormat};
use crate::ir::TextureType;
use crate::{MeshState, ShaderArch, ShaderEnvironment, ShaderStage};

/// FNV-1a over the written bytes with a final avalanche step.
#[derive(Clone, Copy)]
pub(crate) struct StableHasher(u64);

impl StableHasher {
    pub(crate) const fn new() -> StableHasher {
        StableHasher(0xcbf2_9ce4_8422_2325)
    }
}

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn finish(&self) -> u64 {
        let mut x = self.0;
        x ^= x >> 33;
        x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
        x ^= x >> 33;
        x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        x ^= x >> 33;
        x.max(1)
    }
}

/// Forwards every query to the environment and hashes the questions and answers, so the hash
/// covers exactly the guest state translation read. Program memory is hashed by content, not
/// address, so the same code at another address hashes the same.
pub(crate) struct Recorder<'a> {
    env: &'a dyn ShaderEnvironment,
    hasher: Cell<StableHasher>,
}

impl<'a> Recorder<'a> {
    pub(crate) fn new(env: &'a dyn ShaderEnvironment) -> Recorder<'a> {
        Recorder {
            env,
            hasher: Cell::new(StableHasher::new()),
        }
    }

    pub(crate) fn hasher(&self) -> StableHasher {
        self.hasher.get()
    }

    fn record<T: Hash>(&self, query: u8, value: T) -> T {
        let mut hasher = self.hasher.get();
        query.hash(&mut hasher);
        value.hash(&mut hasher);
        self.hasher.set(hasher);
        value
    }
}

impl ShaderEnvironment for Recorder<'_> {
    fn arch(&self) -> ShaderArch {
        self.record(0, self.env.arch())
    }
    fn stage(&self) -> ShaderStage {
        self.record(1, self.env.stage())
    }
    fn start_address(&self) -> GpuVAddr {
        let address = self.env.start_address();
        self.record(2, address % 32);
        address
    }
    fn read_u64(&self, address: GpuVAddr) -> u64 {
        self.record(3, self.env.read_u64(address))
    }
    fn read_cbuf(&self, index: u32, offset: u32) -> u32 {
        self.record(4, (index, offset));
        self.record(4, self.env.read_cbuf(index, offset))
    }
    fn shared_memory_size(&self) -> u32 {
        self.record(5, self.env.shared_memory_size())
    }
    fn workgroup_size(&self) -> [u32; 3] {
        self.record(6, self.env.workgroup_size())
    }
    fn texture_type(&self, raw_handle: u32) -> TextureType {
        self.record(7, raw_handle);
        self.record(7, self.env.texture_type(raw_handle))
    }
    fn texture_pixel_format(&self, raw_handle: u32) -> TexturePixelFormat {
        self.record(8, raw_handle);
        self.record(8, self.env.texture_pixel_format(raw_handle))
    }
    fn is_texture_pixel_format_integer(&self, raw_handle: u32) -> bool {
        self.record(9, raw_handle);
        self.record(9, self.env.is_texture_pixel_format_integer(raw_handle))
    }
    fn viewport_transform_state(&self) -> u32 {
        self.record(10, self.env.viewport_transform_state())
    }
    fn texture_bound_buffer(&self) -> u32 {
        self.record(11, self.env.texture_bound_buffer())
    }
    fn local_memory_size(&self) -> u32 {
        self.record(12, self.env.local_memory_size())
    }
    fn has_hle_macro_state(&self) -> bool {
        self.record(13, self.env.has_hle_macro_state())
    }
    fn is_cbuf_storage_buffer_pointer(&self, index: u32, offset: u32) -> bool {
        self.record(14, (index, offset));
        self.record(14, self.env.is_cbuf_storage_buffer_pointer(index, offset))
    }
    fn is_native_program(&self) -> bool {
        self.record(15, self.env.is_native_program())
    }
    fn is_cbuf_indirect_storage_buffer_pointer(
        &self,
        index: u32,
        offset: u32,
        indirect: u32,
    ) -> bool {
        self.record(16, (index, offset, indirect));
        let answer = self
            .env
            .is_cbuf_indirect_storage_buffer_pointer(index, offset, indirect);
        self.record(16, answer)
    }
    fn gp_passthrough_mask(&self) -> [u32; 8] {
        self.record(17, self.env.gp_passthrough_mask())
    }
    fn replace_const_buffer(&self, bank: u32, offset: u32) -> Option<ReplaceConstant> {
        self.record(18, (bank, offset));
        self.record(18, self.env.replace_const_buffer(bank, offset))
    }
    fn is_proprietary_driver(&self) -> bool {
        self.record(19, self.env.is_proprietary_driver())
    }
    fn mesh_state(&self) -> Option<MeshState> {
        self.record(20, self.env.mesh_state())
    }
    fn is_texture_multisample(&self, raw_handle: u32) -> bool {
        self.record(21, raw_handle);
        self.record(21, self.env.is_texture_multisample(raw_handle))
    }
    fn texture_type_of(&self, source: HandleSource) -> TextureType {
        self.record(22, source);
        self.record(22, self.env.texture_type_of(source))
    }
    fn texture_pixel_format_of(&self, source: HandleSource) -> TexturePixelFormat {
        self.record(23, source);
        self.record(23, self.env.texture_pixel_format_of(source))
    }
    fn is_texture_pixel_format_integer_of(&self, source: HandleSource) -> bool {
        self.record(24, source);
        self.record(24, self.env.is_texture_pixel_format_integer_of(source))
    }
    fn is_texture_multisample_of(&self, source: HandleSource) -> bool {
        self.record(25, source);
        self.record(25, self.env.is_texture_multisample_of(source))
    }
}
