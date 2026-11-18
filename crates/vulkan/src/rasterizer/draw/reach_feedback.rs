use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use ash::vk;

use crate::buffer_cache::BufferBinding;
use crate::device::Device;
use crate::error::Result;
use crate::memory::Buffer;

/// Bytes of each host-visible buffer the feedback words of many modules share.
const CHUNK_BYTES: u64 = 64 << 10;

/// Where one module's reach feedback words live: one u32 per storage buffer slot, in binding
/// order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FeedbackWords {
    chunk: u32,
    offset: u64,
    count: u32,
}

/// The reach feedback words of every module that reports its storage reach, for the session.
/// The GPU raises them with atomic maxima and never lowers them; the host reads them at bind time
/// without waiting, so a bind sees what the work completed so far reported.
pub(super) struct ReachFeedback {
    device: Arc<Device>,
    chunks: Vec<Buffer>,
    used: u64,
    modules: HashMap<u64, FeedbackWords>,
}

impl ReachFeedback {
    pub(super) fn new(device: &Arc<Device>) -> Self {
        Self {
            device: device.clone(),
            chunks: Vec::new(),
            used: CHUNK_BYTES,
            modules: HashMap::new(),
        }
    }

    /// The words of the module with cache key `hash` and `count` storage buffer slots, zeroed
    /// the first time the module binds.
    pub(super) fn words(&mut self, hash: u64, count: u32) -> Result<FeedbackWords> {
        if let Some(&words) = self.modules.get(&hash) {
            return Ok(words);
        }
        let limits = &self.device.properties.limits;
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let count = count.max(1);
        let bytes = (u64::from(count) * 4).next_multiple_of(align);
        let mut offset = self.used.next_multiple_of(align);
        if offset + bytes > CHUNK_BYTES {
            let mut chunk = Buffer::new(
                &self.device,
                CHUNK_BYTES.max(bytes),
                vk::BufferUsageFlags::STORAGE_BUFFER,
                gpu_allocator::MemoryLocation::GpuToCpu,
                "reach feedback",
            )?;
            chunk
                .mapped_mut()
                .expect("reach feedback memory is host-visible")
                .fill(0);
            self.chunks.push(chunk);
            offset = 0;
        }
        let words = FeedbackWords {
            chunk: (self.chunks.len() - 1) as u32,
            offset,
            count,
        };
        self.used = offset + bytes;
        self.modules.insert(hash, words);
        Ok(words)
    }

    /// The binding of `words` for the module's [`lichii_shader::DescriptorKind::ReachFeedback`]
    /// descriptor.
    pub(super) fn binding(&self, words: FeedbackWords) -> BufferBinding {
        BufferBinding {
            buffer: self.chunks[words.chunk as usize].raw(),
            offset: words.offset,
            size: u64::from(words.count) * 4,
        }
    }

    /// The furthest byte from its binding's start that storage buffer slot `slot` of the module
    /// was seen reaching past the end of its binding; 0 before any such access completed.
    pub(super) fn observed(&self, words: FeedbackWords, slot: u32) -> u64 {
        if slot >= words.count {
            return 0;
        }
        let bytes = self.chunks[words.chunk as usize]
            .mapped()
            .expect("reach feedback memory is host-visible");
        let at = (words.offset + u64::from(slot) * 4) as usize;
        let word = bytes[at..at + 4].as_ptr().cast::<AtomicU32>();
        // SAFETY: `word` is 4-byte aligned (an aligned mapping, offsets in whole words) and lies
        // inside the live mapping; the GPU writes it only with atomics.
        u64::from(unsafe { &*word }.load(Ordering::Relaxed))
    }
}
