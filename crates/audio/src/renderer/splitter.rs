//! Splitters: fan a voice or mix out to several destination mixes, each with its own volumes
//! and (from revision 12) biquad filters.
//!
//! Destinations form singly linked chains, one per splitter, rebuilt from the ids each update
//! sends. Links are destination indices.

use bytemuck::{Pod, Zeroable};

use super::behavior::BehaviorInfo;
use super::parameter::AudioRendererParameterInternal;
use super::revision::SupportTag;
use super::voice::{BiquadFilterParameter, BiquadFilterParameter2};
use super::workbuffer::host_size;
use super::{MAX_BIQUAD_FILTERS, MAX_MIX_BUFFERS, UNUSED_MIX_ID, make_magic};
use lichii_common::align_up;

pub const SPLITTER_HEADER_MAGIC: u32 = make_magic(b'S', b'N', b'D', b'H');
pub const SPLITTER_INFO_MAGIC: u32 = make_magic(b'S', b'N', b'D', b'I');
pub const SPLITTER_DATA_MAGIC: u32 = make_magic(b'S', b'N', b'D', b'D');

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SplitterInParameterHeader {
    pub magic: u32,
    pub info_count: i32,
    pub destination_count: i32,
    pub _unk_0c: [u8; 0x14],
}
const _: () = assert!(size_of::<SplitterInParameterHeader>() == 0x20);

/// A splitter entry; `destination_count` destination ids (u32) follow it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SplitterInfoInParameter {
    pub magic: u32,
    pub id: i32,
    pub sample_rate: u32,
    pub destination_count: u32,
}
const _: () = assert!(size_of::<SplitterInfoInParameter>() == 0x10);

/// Destination before revision 12.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SplitterDestinationInParameter {
    pub magic: u32,
    pub id: i32,
    pub mix_volumes: [f32; MAX_MIX_BUFFERS],
    pub mix_id: u32,
    pub in_use: u8,
    pub reset_prev_volume: u8,
    pub _pad: [u8; 2],
}
const _: () = assert!(size_of::<SplitterDestinationInParameter>() == 0x70);

/// Destination from revision 12: Q2.14 biquads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SplitterDestinationInParameterV2a {
    pub magic: u32,
    pub id: i32,
    pub mix_volumes: [f32; MAX_MIX_BUFFERS],
    pub mix_id: u32,
    pub biquad_filters: [BiquadFilterParameter; MAX_BIQUAD_FILTERS],
    pub in_use: u8,
    pub reset_prev_volume: u8,
    pub _reserved: [u8; 10],
}
const _: () = assert!(size_of::<SplitterDestinationInParameterV2a>() == 0x90);

/// Destination from revision 15: float biquads.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SplitterDestinationInParameterV2b {
    pub magic: u32,
    pub id: i32,
    pub mix_volumes: [f32; MAX_MIX_BUFFERS],
    pub mix_id: u32,
    pub biquad_filters: [BiquadFilterParameter2; MAX_BIQUAD_FILTERS],
    pub in_use: u8,
    pub reset_prev_volume: u8,
    pub _reserved: [u8; 10],
}
const _: () = assert!(size_of::<SplitterDestinationInParameterV2b>() == 0xA8);

/// One splitter destination.
#[derive(Clone, Debug)]
pub struct SplitterDestinationData {
    id: i32,
    destination_id: i32,
    mix_volumes: [f32; MAX_MIX_BUFFERS],
    prev_mix_volumes: [f32; MAX_MIX_BUFFERS],
    biquad_filters: [BiquadFilterParameter2; MAX_BIQUAD_FILTERS],
    next: Option<u32>,
    in_use: bool,
    need_update: bool,
}

impl SplitterDestinationData {
    fn new(id: i32) -> Self {
        Self {
            id,
            destination_id: UNUSED_MIX_ID,
            mix_volumes: [0.0; MAX_MIX_BUFFERS],
            prev_mix_volumes: [0.0; MAX_MIX_BUFFERS],
            biquad_filters: Default::default(),
            next: None,
            in_use: false,
            need_update: false,
        }
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn is_configured(&self) -> bool {
        self.in_use && self.destination_id != UNUSED_MIX_ID
    }

    pub fn mix_id(&self) -> i32 {
        self.destination_id
    }

    pub fn mix_volumes(&self) -> &[f32; MAX_MIX_BUFFERS] {
        &self.mix_volumes
    }

    pub fn prev_mix_volumes(&self) -> &[f32; MAX_MIX_BUFFERS] {
        &self.prev_mix_volumes
    }

    pub fn biquad_filters(&self) -> &[BiquadFilterParameter2; MAX_BIQUAD_FILTERS] {
        &self.biquad_filters
    }

    pub fn next(&self) -> Option<u32> {
        self.next
    }

    pub fn clear_mix_volume(&mut self) {
        self.mix_volumes = [0.0; MAX_MIX_BUFFERS];
        self.prev_mix_volumes = [0.0; MAX_MIX_BUFFERS];
    }

    pub fn mark_as_need_to_update_internal_state(&mut self) {
        self.need_update = true;
    }

    /// Carries the volumes over to the previous volumes once a ramp has been rendered.
    pub fn update_internal_state(&mut self) {
        if self.in_use && self.need_update {
            self.prev_mix_volumes = self.mix_volumes;
        }
        self.need_update = false;
    }

    fn update(&mut self, id: i32, mix_volumes: &[f32; MAX_MIX_BUFFERS], mix_id: u32, in_use: bool) {
        if id != self.id {
            return;
        }
        self.destination_id = mix_id as i32;
        self.mix_volumes = *mix_volumes;
        if !self.in_use && in_use {
            self.prev_mix_volumes = self.mix_volumes;
            self.need_update = false;
        }
        self.in_use = in_use;
    }
}

/// One splitter.
#[derive(Clone, Debug)]
pub struct SplitterInfo {
    id: i32,
    sample_rate: u32,
    destination_count: u32,
    has_new_connection: bool,
    destinations: Option<u32>,
}

impl SplitterInfo {
    fn new(id: i32) -> Self {
        Self {
            id,
            sample_rate: 0,
            destination_count: 0,
            has_new_connection: true,
            destinations: None,
        }
    }

    pub fn id(&self) -> i32 {
        self.id
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn destination_count(&self) -> u32 {
        self.destination_count
    }

    pub fn has_new_connection(&self) -> bool {
        self.has_new_connection
    }

    pub fn set_new_connection_flag(&mut self) {
        self.has_new_connection = true;
    }

    pub fn clear_new_connection_flag(&mut self) {
        self.has_new_connection = false;
    }

    /// First destination of the chain.
    pub fn head(&self) -> Option<u32> {
        self.destinations
    }
}

/// All splitters and destinations of a renderer.
#[derive(Clone, Debug, Default)]
pub struct SplitterContext {
    infos: Vec<SplitterInfo>,
    destinations: Vec<SplitterDestinationData>,
    bug_fixed: bool,
    biquad_param_supported: bool,
    float_coeff_supported: bool,
}

fn read<T: Pod>(input: &[u8], offset: usize) -> Option<T> {
    input
        .get(offset..offset.checked_add(size_of::<T>())?)
        .map(bytemuck::pod_read_unaligned)
}

impl SplitterContext {
    pub fn new(behavior: &BehaviorInfo, params: &AudioRendererParameterInternal) -> Self {
        if !behavior.supports(SupportTag::Splitter)
            || params.splitter_infos == 0
            || params.splitter_destinations <= 0
        {
            return Self::default();
        }
        Self {
            infos: (0..params.splitter_infos as i32)
                .map(SplitterInfo::new)
                .collect(),
            destinations: (0..params.splitter_destinations)
                .map(SplitterDestinationData::new)
                .collect(),
            bug_fixed: behavior.supports(SupportTag::SplitterBugFix),
            biquad_param_supported: behavior.supports(SupportTag::SplitterBiquadFilterParameter),
            float_coeff_supported: behavior.supports(SupportTag::SplitterDestinationV2b),
        }
    }

    /// Work buffer bytes the reference reserves for splitters.
    pub fn work_buffer_size(
        behavior: &BehaviorInfo,
        params: &AudioRendererParameterInternal,
    ) -> u64 {
        if !behavior.supports(SupportTag::Splitter) {
            return 0;
        }
        let destinations = params.splitter_destinations as i64 as u64;
        let mut size = destinations.wrapping_mul(host_size::SPLITTER_DESTINATION_DATA)
            + params.splitter_infos as u64 * host_size::SPLITTER_INFO;
        if behavior.supports(SupportTag::SplitterBugFix) {
            size += align_up(destinations.wrapping_mul(4), 0x10);
        }
        size
    }

    pub fn info_count(&self) -> u32 {
        self.infos.len() as u32
    }

    pub fn data_count(&self) -> u32 {
        self.destinations.len() as u32
    }

    pub fn info(&self, id: i32) -> Option<&SplitterInfo> {
        usize::try_from(id).ok().and_then(|id| self.infos.get(id))
    }

    pub fn infos(&self) -> &[SplitterInfo] {
        &self.infos
    }

    pub fn data(&self, index: u32) -> Option<&SplitterDestinationData> {
        self.destinations.get(index as usize)
    }

    pub fn data_mut(&mut self, index: u32) -> Option<&mut SplitterDestinationData> {
        self.destinations.get_mut(index as usize)
    }

    /// Index of the `n`th destination of splitter `splitter_id`.
    pub fn destination_index(&self, splitter_id: i32, n: u32) -> Option<u32> {
        let mut current = self.info(splitter_id)?.destinations;
        for _ in 0..n {
            current = self.destinations.get(current? as usize)?.next;
        }
        current
    }

    /// The `n`th destination of splitter `splitter_id`.
    pub fn destination_data(&self, splitter_id: i32, n: u32) -> Option<&SplitterDestinationData> {
        self.data(self.destination_index(splitter_id, n)?)
    }

    pub fn using_splitter(&self) -> bool {
        !self.infos.is_empty() && !self.destinations.is_empty()
    }

    /// Marks every splitter as newly connected.
    pub fn clear_all_new_connection_flag(&mut self) {
        for info in &mut self.infos {
            info.set_new_connection_flag();
        }
    }

    /// Runs [`SplitterDestinationData::update_internal_state`] along every chain.
    pub fn update_internal_state(&mut self) {
        let limit = self.destinations.len();
        for info in &self.infos {
            let mut current = info.destinations;
            for _ in 0..limit {
                let Some(destination) = current.and_then(|i| self.destinations.get_mut(i as usize))
                else {
                    break;
                };
                destination.update_internal_state();
                current = destination.next;
            }
        }
    }

    fn dest_count_per_info_for_compat(&self) -> u32 {
        if self.infos.is_empty() {
            return 0;
        }
        self.destinations.len() as u32 / self.infos.len() as u32
    }

    /// Parses the splitter section at the start of `input`. Fails on a bad header magic or on
    /// ids or reads outside the section.
    pub fn update(&mut self, input: &[u8]) -> bool {
        if self.destinations.is_empty() || self.infos.is_empty() {
            return true;
        }
        let Some(header) = read::<SplitterInParameterHeader>(input, 0) else {
            return false;
        };
        if header.magic != SPLITTER_HEADER_MAGIC {
            return false;
        }
        for info in &mut self.infos {
            info.clear_new_connection_flag();
        }
        let offset = size_of::<SplitterInParameterHeader>();
        let Some(offset) = self.update_infos(input, offset, header.info_count as u32) else {
            return false;
        };
        self.update_destinations(input, offset, header.destination_count as u32)
            .is_some()
    }

    fn update_infos(&mut self, input: &[u8], mut offset: usize, count: u32) -> Option<usize> {
        for _ in 0..count {
            let header = read::<SplitterInfoInParameter>(input, offset)?;
            if header.magic != SPLITTER_INFO_MAGIC {
                continue;
            }
            let Some(index) = usize::try_from(header.id)
                .ok()
                .filter(|&id| id < self.infos.len())
            else {
                break;
            };
            self.recompose_destination(index, &header, input, offset)?;
            let info = &mut self.infos[index];
            info.sample_rate = header.sample_rate;
            info.has_new_connection = true;
            offset += size_of::<SplitterInfoInParameter>()
                + 3 * 4
                + header.destination_count as usize * 4;
        }
        Some(offset)
    }

    fn recompose_destination(
        &mut self,
        index: usize,
        header: &SplitterInfoInParameter,
        input: &[u8],
        offset: usize,
    ) -> Option<()> {
        let limit = self.destinations.len();
        let mut current = self.infos[index].destinations;
        for _ in 0..limit {
            let Some(destination) = current.and_then(|i| self.destinations.get_mut(i as usize))
            else {
                break;
            };
            current = destination.next.take();
        }
        self.infos[index].destinations = None;

        let mut dest_count = header.destination_count;
        if !self.bug_fixed {
            dest_count = dest_count.min(self.dest_count_per_info_for_compat());
        }
        if dest_count == 0 {
            return Some(());
        }

        let ids_offset = offset + size_of::<SplitterInfoInParameter>();
        let id_at = |i: u32| -> Option<u32> {
            let id = read::<u32>(input, ids_offset + i as usize * 4)?;
            ((id as usize) < limit).then_some(id)
        };
        let head = id_at(0)?;
        let mut previous = head;
        for i in 1..dest_count {
            let next = id_at(i)?;
            self.destinations[previous as usize].next = Some(next);
            previous = next;
        }
        let info = &mut self.infos[index];
        info.destinations = Some(head);
        info.destination_count = dest_count;
        Some(())
    }

    fn update_destinations(
        &mut self,
        input: &[u8],
        mut offset: usize,
        count: u32,
    ) -> Option<usize> {
        for _ in 0..count {
            if !self.biquad_param_supported {
                let data = read::<SplitterDestinationInParameter>(input, offset)?;
                if data.magic != SPLITTER_DATA_MAGIC || !self.is_destination_id(data.id) {
                    continue;
                }
                self.destinations[data.id as usize].update(
                    data.id,
                    &data.mix_volumes,
                    data.mix_id,
                    data.in_use != 0,
                );
                offset += size_of::<SplitterDestinationInParameter>();
            } else if !self.float_coeff_supported {
                let data = read::<SplitterDestinationInParameterV2a>(input, offset)?;
                if data.magic != SPLITTER_DATA_MAGIC || !self.is_destination_id(data.id) {
                    continue;
                }
                let destination = &mut self.destinations[data.id as usize];
                destination.update(data.id, &data.mix_volumes, data.mix_id, data.in_use != 0);
                const SCALE: f32 = 1.0 / (1 << 14) as f32;
                for (out, legacy) in destination
                    .biquad_filters
                    .iter_mut()
                    .zip(&data.biquad_filters)
                {
                    out.enabled = legacy.enabled;
                    out.numerator = legacy.b.map(|b| b as f32 * SCALE);
                    out.denominator = legacy.a.map(|a| a as f32 * SCALE);
                }
                offset += size_of::<SplitterDestinationInParameterV2a>();
            } else {
                let data = read::<SplitterDestinationInParameterV2b>(input, offset)?;
                if data.magic != SPLITTER_DATA_MAGIC || !self.is_destination_id(data.id) {
                    continue;
                }
                let destination = &mut self.destinations[data.id as usize];
                destination.update(data.id, &data.mix_volumes, data.mix_id, data.in_use != 0);
                destination.biquad_filters = data.biquad_filters;
                offset += size_of::<SplitterDestinationInParameterV2b>();
            }
        }
        Some(offset)
    }

    fn is_destination_id(&self, id: i32) -> bool {
        usize::try_from(id).is_ok_and(|id| id < self.destinations.len())
    }
}
