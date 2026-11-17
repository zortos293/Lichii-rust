//! The buffer ranges the dispatches since the last compute barrier accessed and wrote. A dispatch
//! that accesses none of their writes and writes none of their ranges cannot race them, so it
//! records beside them and one barrier after the batch orders them all before later commands.

use ash::vk;
use lichii_shader::info::{ShaderInfo, StorageBufferDescriptor};

/// Dispatches one compute barrier covers at most, which bounds the overlap test.
const MAX_DISPATCHES: u32 = 16;

#[derive(Default)]
pub(super) struct ComputeBatch {
    accessed: Vec<vk::DescriptorBufferInfo>,
    written: Vec<vk::DescriptorBufferInfo>,
    dispatches: u32,
}

impl ComputeBatch {
    /// Whether a dispatch binding `accessed`, of which it writes `written`, may record beside
    /// the batch without a barrier.
    pub(super) fn admits(
        &self,
        accessed: &[vk::DescriptorBufferInfo],
        written: &[vk::DescriptorBufferInfo],
    ) -> bool {
        self.dispatches < MAX_DISPATCHES
            && !accessed.iter().any(|a| overlaps_any(&self.written, a))
            && !written.iter().any(|w| overlaps_any(&self.accessed, w))
    }

    /// Adds a dispatch binding `accessed`, of which it writes `written`.
    pub(super) fn add(
        &mut self,
        accessed: &[vk::DescriptorBufferInfo],
        written: &[vk::DescriptorBufferInfo],
    ) {
        self.accessed.extend_from_slice(accessed);
        self.written.extend_from_slice(written);
        self.dispatches += 1;
    }

    /// Forgets the batch once a barrier ordered it.
    pub(super) fn clear(&mut self) {
        self.accessed.clear();
        self.written.clear();
        self.dispatches = 0;
    }
}

/// Whether every buffer access of a dispatch running `info` stays inside the ranges it binds, so
/// the batch can tell what it touches: no Switch 2 native pointer (`switch2`), whose size the
/// renderer guesses; no storage pointer read from a table; no global access routed among the
/// bound buffers at run time; and no texture handle read from storage.
pub(super) fn bounded_by_bindings(info: &ShaderInfo, switch2: bool) -> bool {
    let descriptors = &info.storage_buffers_descriptors;
    !info.uses_global_memory
        && info.handle_programs.is_empty()
        && (!switch2 || descriptors.is_empty())
        && descriptors
            .iter()
            .all(|d| d.indirect_offset == StorageBufferDescriptor::DIRECT)
}

fn overlaps_any(ranges: &[vk::DescriptorBufferInfo], r: &vk::DescriptorBufferInfo) -> bool {
    ranges.iter().any(|o| {
        o.buffer == r.buffer
            && o.offset < r.offset.saturating_add(r.range)
            && r.offset < o.offset.saturating_add(o.range)
    })
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;

    fn range(buffer: u64, offset: u64, range: u64) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: vk::Buffer::from_raw(buffer),
            offset,
            range,
        }
    }

    #[test]
    fn dispatches_race_only_on_overlapping_ranges_one_of_them_writes() {
        let mut batch = ComputeBatch::default();
        let (a, b) = (range(1, 0, 0x100), range(1, 0x100, 0x100));
        batch.add(&[a], &[a]);
        assert!(batch.admits(&[b], &[b]), "disjoint ranges of one buffer");
        assert!(batch.admits(&[range(2, 0, 0x100)], &[]), "another buffer");
        assert!(
            !batch.admits(&[range(1, 0xff, 2)], &[]),
            "a read of the write"
        );
        assert!(!batch.admits(&[a], &[a]), "a second write of it");

        batch.clear();
        batch.add(&[a], &[]);
        assert!(batch.admits(&[a], &[]), "two reads");
        assert!(!batch.admits(&[a], &[a]), "a write of what the batch read");
    }

    #[test]
    fn only_dispatches_bounded_by_their_bindings_batch() {
        let direct = StorageBufferDescriptor {
            cbuf_index: 0,
            cbuf_offset: 0x110,
            count: 1,
            is_written: true,
            indirect_offset: StorageBufferDescriptor::DIRECT,
            write_extent: Default::default(),
        };
        let info = ShaderInfo {
            storage_buffers_descriptors: vec![direct],
            ..ShaderInfo::default()
        };
        assert!(bounded_by_bindings(&info, false), "a sized guest pointer");
        assert!(
            bounded_by_bindings(&ShaderInfo::default(), true),
            "a Switch 2 program without storage buffers"
        );
        assert!(
            !bounded_by_bindings(&info, true),
            "a Switch 2 native pointer"
        );
        let table = ShaderInfo {
            storage_buffers_descriptors: vec![StorageBufferDescriptor {
                indirect_offset: 8,
                ..direct
            }],
            ..ShaderInfo::default()
        };
        assert!(
            !bounded_by_bindings(&table, false),
            "a pointer from a table"
        );
        let routed = ShaderInfo {
            uses_global_memory: true,
            ..info.clone()
        };
        assert!(
            !bounded_by_bindings(&routed, false),
            "global accesses routed at run time"
        );
    }

    #[test]
    fn a_batch_closes_after_its_last_dispatch() {
        let mut batch = ComputeBatch::default();
        for i in 0..MAX_DISPATCHES {
            let r = range(1, u64::from(i) * 0x100, 0x100);
            assert!(batch.admits(&[r], &[r]));
            batch.add(&[r], &[r]);
        }
        assert!(!batch.admits(&[range(2, 0, 4)], &[]));
        batch.clear();
        assert!(batch.admits(&[range(1, 0, 4)], &[]));
    }
}
