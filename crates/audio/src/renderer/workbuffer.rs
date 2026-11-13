//! Work buffer layout.
//!
//! The game sizes the renderer's work buffer with `GetWorkBufferSize`, and the size counts the
//! reference implementation's in-memory structures. The renderer keeps its state in owned Rust
//! types, so those sizes are recorded here as constants. They are the reference's x86-64 layout
//! (GCC/Clang), measured with `sizeof`.

/// Byte sizes of the reference's structures.
pub mod host_size {
    pub const VOICE_INFO: u64 = 0x230;
    pub const VOICE_STATE: u64 = 0x128;
    pub const VOICE_CHANNEL_RESOURCE: u64 = 0xC8;
    pub const MIX_INFO: u64 = 0x940;
    pub const MEMORY_POOL_INFO: u64 = 0x20;
    pub const EFFECT_INFO: u64 = 0x638;
    pub const EFFECT_RESULT_STATE: u64 = 0x80;
    pub const SINK_INFO: u64 = 0x160;
    pub const SPLITTER_INFO: u64 = 0x20;
    pub const SPLITTER_DESTINATION_DATA: u64 = 0x108;
    pub const UPSAMPLER_INFO: u64 = 0x268;
    pub const UPSAMPLER_MANAGER: u64 = 0x50;
    pub const POINTER: u64 = 8;

    /// Command sizes, used for the variadic command buffer size.
    pub mod command {
        pub const ADPCM_DATA_SOURCE_V1: u64 = 0x120;
        pub const PCM_INT16_DATA_SOURCE_V1: u64 = 0x118;
        pub const PCM_FLOAT_DATA_SOURCE_V1: u64 = 0x118;
        pub const ADPCM_DATA_SOURCE_V2: u64 = 0x128;
        pub const PCM_INT16_DATA_SOURCE_V2: u64 = 0x118;
        pub const PCM_FLOAT_DATA_SOURCE_V2: u64 = 0x118;
        pub const BIQUAD_FILTER: u64 = 0x50;
        pub const VOLUME_RAMP: u64 = 0x28;
        pub const MIX_RAMP_GROUPED: u64 = 0x148;
        pub const DEPOP_PREPARE: u64 = 0x60;
        pub const DEPOP_FOR_MIX_BUFFERS: u64 = 0x30;
        pub const MIX: u64 = 0x28;
        pub const VOLUME: u64 = 0x28;
        pub const MIX_RAMP: u64 = 0x30;
        pub const DEVICE_SINK: u64 = 0x140;
        pub const CIRCULAR_BUFFER_SINK: u64 = 0x38;
        pub const PERFORMANCE: u64 = 0x40;
    }
}

/// Carves a buffer into aligned pieces, tracking offsets only.
#[derive(Clone, Copy, Debug)]
pub struct WorkbufferAllocator {
    size: u64,
    offset: u64,
}

impl WorkbufferAllocator {
    pub fn new(size: u64) -> Self {
        Self { size, offset: 0 }
    }

    /// Offset of `bytes` bytes aligned to `alignment`. An empty request or one that does not fit
    /// returns `None`, the reference's empty span.
    pub fn allocate(&mut self, bytes: u64, alignment: u64) -> Option<u64> {
        if bytes == 0 {
            return None;
        }
        let start = self.offset.checked_next_multiple_of(alignment)?;
        let end = start.checked_add(bytes)?;
        if end > self.size {
            tracing::error!(
                size = self.size,
                offset = self.offset,
                bytes,
                alignment,
                "audio renderer work buffer too small"
            );
            return None;
        }
        self.offset = end;
        Some(start)
    }

    pub fn align(&mut self, alignment: u64) {
        self.offset = self.offset.next_multiple_of(alignment);
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn remaining(&self) -> u64 {
        self.size.saturating_sub(self.offset)
    }
}
