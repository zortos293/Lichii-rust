//! Performance metrics: per-frame timing entries the DSP fills in, kept in a ring of history
//! frames the game reads back through `RequestUpdate`.
//!
//! The buffer holds the frame being rendered, then `perf_frames` history frames. Each frame is a
//! header, one entry per voice, effect, sink and mix, then up to [`MAX_DETAIL_ENTRIES`] details.

use bytemuck::{Pod, Zeroable};

use super::behavior::BehaviorInfo;
use super::make_magic;
use super::parameter::AudioRendererParameterInternal;

/// Detail entries per frame.
pub const MAX_DETAIL_ENTRIES: u32 = 100;
const PERF_MAGIC: u32 = make_magic(b'P', b'E', b'R', b'F');

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PerformanceInParameter {
    pub target_node_id: i32,
    pub _unk_04: [u8; 0xC],
}
const _: () = assert!(size_of::<PerformanceInParameter>() == 0x10);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PerformanceOutStatus {
    pub history_size: i32,
    pub _unk_04: [u8; 0xC],
}
const _: () = assert!(size_of::<PerformanceOutStatus>() == 0x10);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PerformanceEntryType {
    Invalid = 0,
    Voice = 1,
    SubMix = 2,
    FinalMix = 3,
    Sink = 4,
}

/// System detail types (version 2 only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PerformanceSysDetailType {
    PcmInt16 = 15,
    PcmFloat = 16,
    Adpcm = 17,
    LightLimiter = 37,
}

/// Where the DSP writes one entry's times, as byte offsets into [`PerformanceManager::buffer`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PerformanceEntryAddresses {
    /// The frame header counter (entries or details) this entry belongs to.
    pub header_entry_count_offset: usize,
    pub entry_start_time_offset: usize,
    pub entry_processed_time_offset: usize,
}

/// Sizes of the two frame layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Layout {
    header: usize,
    entry: usize,
    detail: usize,
}

const LAYOUT_V1: Layout = Layout {
    header: 0x18,
    entry: 0x10,
    detail: 0x10,
};
const LAYOUT_V2: Layout = Layout {
    header: 0x30,
    entry: 0x18,
    detail: 0x18,
};

mod header {
    pub const MAGIC: usize = 0x00;
    pub const ENTRY_COUNT: usize = 0x04;
    pub const DETAIL_COUNT: usize = 0x08;
    pub const NEXT_OFFSET: usize = 0x0C;
    pub const TOTAL_PROCESSING_TIME: usize = 0x10;
    pub const V1_FRAME_INDEX: usize = 0x14;
    pub const V2_VOICES_DROPPED: usize = 0x14;
    pub const V2_START_TIME: usize = 0x18;
    pub const V2_FRAME_INDEX: usize = 0x20;
    pub const V2_RENDER_TIME_EXCEEDED: usize = 0x24;
}

mod entry {
    pub const NODE_ID: usize = 0x00;
    pub const START_TIME: usize = 0x04;
    pub const PROCESSED_TIME: usize = 0x08;
    pub const ENTRY_TYPE: usize = 0x0C;
    pub const DETAIL_TYPE: usize = 0x0C;
    pub const DETAIL_ENTRY_TYPE: usize = 0x0D;
    pub const DETAIL_UNK_10: usize = 0x10;
}

fn layout_for(format: u32) -> Layout {
    if format == 2 { LAYOUT_V2 } else { LAYOUT_V1 }
}

fn read_u32(buffer: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(buffer[offset..offset + 4].try_into().unwrap())
}

fn write_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Bytes of one performance frame.
pub fn frame_size(behavior: &BehaviorInfo, params: &AudioRendererParameterInternal) -> u64 {
    let entry_count = params.voices as u64
        + params.effects as u64
        + params.sub_mixes as u64
        + params.sinks as u64
        + 1;
    let layout = layout_for(behavior.performance_metrics_format());
    layout.header as u64
        + MAX_DETAIL_ENTRIES as u64 * layout.detail as u64
        + entry_count * layout.entry as u64
}

/// Performance frames of one renderer.
#[derive(Clone, Debug)]
pub struct PerformanceManager {
    version: u32,
    layout: Layout,
    buffer: Vec<u8>,
    entries_per_frame: u32,
    frame_size: usize,
    max_frames: u32,
    entry_count: u32,
    detail_count: u32,
    output_frame_index: u32,
    last_output_frame_index: u32,
    history_frame_index: u32,
    target_node_id: u32,
}

impl PerformanceManager {
    /// Buffer for `params.perf_frames` history frames, sized as the reference sizes it.
    pub fn new(behavior: &BehaviorInfo, params: &AudioRendererParameterInternal) -> Self {
        let version = behavior.performance_metrics_format();
        let frame_size = frame_size(behavior, params) as usize;
        let buffer_size = frame_size * (params.perf_frames as usize + 1) + 0xC;
        let frame_count = (buffer_size / frame_size) as u32;
        Self {
            version,
            layout: layout_for(version),
            buffer: vec![0; buffer_size],
            entries_per_frame: params.voices + params.effects + params.sinks + params.sub_mixes + 1,
            frame_size,
            max_frames: frame_count - 1,
            entry_count: 0,
            detail_count: 0,
            output_frame_index: 0,
            last_output_frame_index: 0,
            history_frame_index: 0,
            target_node_id: 0,
        }
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    /// The frame the DSP writes into, followed by the history ring.
    pub fn buffer(&self) -> &[u8] {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        &mut self.buffer
    }

    pub fn is_detail_target(&self, node_id: u32) -> bool {
        self.target_node_id == node_id
    }

    pub fn set_detail_target(&mut self, node_id: u32) {
        self.target_node_id = node_id;
    }

    fn details_offset(&self) -> usize {
        self.layout.header + self.entries_per_frame as usize * self.layout.entry
    }

    /// Claims the next entry of the current frame.
    pub fn next_entry(
        &mut self,
        entry_type: PerformanceEntryType,
        node_id: i32,
    ) -> Option<PerformanceEntryAddresses> {
        if self.entry_count >= self.entries_per_frame {
            return None;
        }
        let offset = self.layout.header + self.entry_count as usize * self.layout.entry;
        self.entry_count += 1;
        let slot = &mut self.buffer[offset..offset + self.layout.entry];
        slot.fill(0);
        write_u32(slot, entry::NODE_ID, node_id as u32);
        slot[entry::ENTRY_TYPE] = entry_type as u8;
        Some(PerformanceEntryAddresses {
            header_entry_count_offset: header::ENTRY_COUNT,
            entry_start_time_offset: offset + entry::START_TIME,
            entry_processed_time_offset: offset + entry::PROCESSED_TIME,
        })
    }

    fn claim_detail(&mut self, node_id: i32) -> Option<(usize, PerformanceEntryAddresses)> {
        if self.detail_count >= MAX_DETAIL_ENTRIES {
            return None;
        }
        let offset = self.details_offset() + self.detail_count as usize * self.layout.detail;
        self.detail_count += 1;
        let slot = &mut self.buffer[offset..offset + self.layout.detail];
        slot.fill(0);
        write_u32(slot, entry::NODE_ID, node_id as u32);
        Some((
            offset,
            PerformanceEntryAddresses {
                header_entry_count_offset: header::DETAIL_COUNT,
                entry_start_time_offset: offset + entry::START_TIME,
                entry_processed_time_offset: offset + entry::PROCESSED_TIME,
            },
        ))
    }

    /// Claims the next detail of the current frame. `detail_type` is the command kind.
    pub fn next_detail(
        &mut self,
        detail_type: u8,
        entry_type: PerformanceEntryType,
        node_id: i32,
    ) -> Option<PerformanceEntryAddresses> {
        let (offset, addresses) = self.claim_detail(node_id)?;
        self.buffer[offset + entry::DETAIL_TYPE] = detail_type;
        self.buffer[offset + entry::DETAIL_ENTRY_TYPE] = entry_type as u8;
        Some(addresses)
    }

    /// Claims a system detail (version 2 only). Also returns the offset of the detail's extra
    /// word, which the DSP fills in.
    pub fn next_sys_detail(
        &mut self,
        detail_type: PerformanceSysDetailType,
        node_id: i32,
    ) -> Option<(PerformanceEntryAddresses, usize)> {
        if self.version != 2 {
            return None;
        }
        let (offset, addresses) = self.claim_detail(node_id)?;
        self.buffer[offset + entry::DETAIL_TYPE] = detail_type as u8;
        Some((addresses, offset + entry::DETAIL_UNK_10))
    }

    /// Moves the current frame into the history ring and starts a new one.
    pub fn tap_frame(&mut self, dsp_behind: bool, voices_dropped: u32, rendering_start_tick: u64) {
        if self.max_frames > 0 {
            let frame_size = self.frame_size;
            let history = frame_size + self.output_frame_index as usize * frame_size;
            self.buffer.copy_within(0..frame_size, history);
            let frame = &mut self.buffer[history..history + frame_size];
            if self.version == 2 {
                frame[header::V2_RENDER_TIME_EXCEEDED] = dsp_behind as u8;
                write_u32(frame, header::V2_VOICES_DROPPED, voices_dropped);
                frame[header::V2_START_TIME..header::V2_START_TIME + 8]
                    .copy_from_slice(&rendering_start_tick.to_le_bytes());
                write_u32(frame, header::V2_FRAME_INDEX, self.history_frame_index);
            } else {
                write_u32(frame, header::V1_FRAME_INDEX, self.history_frame_index);
            }
            self.history_frame_index = self.history_frame_index.wrapping_add(1);
            self.output_frame_index = (self.output_frame_index + 1) % self.max_frames;
        }
        self.entry_count = 0;
        self.detail_count = 0;
        write_u32(&mut self.buffer, header::ENTRY_COUNT, 0);
        write_u32(&mut self.buffer, header::DETAIL_COUNT, 0);
    }

    /// Writes the history frames the game has not read yet into `out`, dropping entries that
    /// never ran. Returns the bytes written.
    pub fn copy_histories(&mut self, out: &mut [u8]) -> u32 {
        if out.is_empty() || self.last_output_frame_index == self.output_frame_index {
            return 0;
        }
        let layout = self.layout;
        let mut written = 0usize;
        let mut wrote_frame = false;
        while self.last_output_frame_index != self.output_frame_index {
            let frame_start = self.frame_size * (1 + self.last_output_frame_index as usize);
            let frame = &self.buffer[frame_start..frame_start + self.frame_size];
            let entry_count = read_u32(frame, header::ENTRY_COUNT);
            let detail_count = read_u32(frame, header::DETAIL_COUNT);
            let remaining = out.len() - written;
            let needed = entry_count as u64 * layout.entry as u64
                + detail_count as u64 * layout.detail as u64
                + 2 * layout.header as u64;
            if (remaining as u64) < needed {
                break;
            }

            let out_frame = &mut out[written..];
            let mut out_offset = layout.header;
            let mut out_entries = 0u32;
            let mut total_processing_time = 0u32;
            let entries = frame[layout.header..]
                .chunks_exact(layout.entry)
                .take(entry_count.min(self.entries_per_frame) as usize);
            for item in entries {
                let processed = read_u32(item, entry::PROCESSED_TIME);
                if processed > 0 || read_u32(item, entry::START_TIME) > 0 {
                    out_frame[out_offset..out_offset + layout.entry].copy_from_slice(item);
                    out_offset += layout.entry;
                    out_entries += 1;
                    total_processing_time = total_processing_time.wrapping_add(processed);
                }
            }
            let mut out_details = 0u32;
            let details = frame[self.details_offset()..]
                .chunks_exact(layout.detail)
                .take(detail_count.min(MAX_DETAIL_ENTRIES) as usize);
            for item in details {
                if read_u32(item, entry::PROCESSED_TIME) > 0
                    || read_u32(item, entry::START_TIME) > 0
                {
                    out_frame[out_offset..out_offset + layout.detail].copy_from_slice(item);
                    out_offset += layout.detail;
                    out_details += 1;
                }
            }

            write_u32(out_frame, header::MAGIC, PERF_MAGIC);
            write_u32(out_frame, header::ENTRY_COUNT, out_entries);
            write_u32(out_frame, header::DETAIL_COUNT, out_details);
            write_u32(out_frame, header::NEXT_OFFSET, out_offset as u32);
            write_u32(
                out_frame,
                header::TOTAL_PROCESSING_TIME,
                total_processing_time,
            );
            if self.version == 2 {
                out_frame[header::V2_VOICES_DROPPED..header::V2_RENDER_TIME_EXCEEDED + 1]
                    .copy_from_slice(
                        &frame[header::V2_VOICES_DROPPED..header::V2_RENDER_TIME_EXCEEDED + 1],
                    );
            } else {
                out_frame[header::V1_FRAME_INDEX..header::V1_FRAME_INDEX + 4]
                    .copy_from_slice(&frame[header::V1_FRAME_INDEX..header::V1_FRAME_INDEX + 4]);
            }

            written += out_offset;
            wrote_frame = true;
            self.last_output_frame_index = (self.last_output_frame_index + 1) % self.max_frames;
        }
        if wrote_frame && out.len() - written > layout.header {
            out[written..written + layout.header].fill(0);
        }
        written as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::revision::encode_revision;

    fn params(perf_frames: u32) -> AudioRendererParameterInternal {
        AudioRendererParameterInternal {
            voices: 2,
            effects: 1,
            sinks: 1,
            sub_mixes: 1,
            perf_frames,
            ..Default::default()
        }
    }

    #[test]
    fn frame_sizes_per_format() {
        let p = params(1);
        assert_eq!(
            frame_size(&BehaviorInfo::new(encode_revision(4, false)), &p),
            0x18 + 1600 + 6 * 0x10
        );
        assert_eq!(
            frame_size(&BehaviorInfo::new(encode_revision(5, false)), &p),
            0x30 + 2400 + 6 * 0x18
        );
    }

    #[test]
    fn history_keeps_entries_that_ran() {
        let behavior = BehaviorInfo::new(encode_revision(15, false));
        let mut perf = PerformanceManager::new(&behavior, &params(2));
        let voice = perf
            .next_entry(PerformanceEntryType::Voice, 0x1001)
            .unwrap();
        let skipped = perf.next_entry(PerformanceEntryType::Sink, 0x4001).unwrap();
        let detail = perf
            .next_detail(5, PerformanceEntryType::Voice, 0x1001)
            .unwrap();
        assert_eq!(skipped.entry_start_time_offset, 0x30 + 0x18 + 4);
        let buffer = perf.buffer_mut();
        write_u32(buffer, voice.entry_start_time_offset, 10);
        write_u32(buffer, voice.entry_processed_time_offset, 25);
        write_u32(buffer, detail.entry_processed_time_offset, 7);
        write_u32(buffer, header::ENTRY_COUNT, 2);
        write_u32(buffer, header::DETAIL_COUNT, 1);
        perf.tap_frame(true, 3, 0x1234);

        let mut out = vec![0xAAu8; 0x200];
        let size = perf.copy_histories(&mut out) as usize;
        assert_eq!(size, 0x30 + 0x18 + 0x18);
        assert_eq!(read_u32(&out, header::MAGIC), PERF_MAGIC);
        assert_eq!(read_u32(&out, header::ENTRY_COUNT), 1);
        assert_eq!(read_u32(&out, header::DETAIL_COUNT), 1);
        assert_eq!(read_u32(&out, header::NEXT_OFFSET), size as u32);
        assert_eq!(read_u32(&out, header::TOTAL_PROCESSING_TIME), 25);
        assert_eq!(read_u32(&out, header::V2_VOICES_DROPPED), 3);
        assert_eq!(out[header::V2_RENDER_TIME_EXCEEDED], 1);
        assert_eq!(read_u32(&out, 0x30), 0x1001);
        assert_eq!(read_u32(&out, 0x30 + 0x18 + 8), 7);
        assert!(out[size..size + 0x30].iter().all(|&b| b == 0));
        assert_eq!(perf.copy_histories(&mut out), 0);
    }
}
