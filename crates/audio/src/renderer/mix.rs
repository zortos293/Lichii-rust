//! Mixes: the buffers voices and effects render into, their routing and render order.

use bytemuck::{Pod, Zeroable};

use super::behavior::BehaviorInfo;
use super::effect::EffectContext;
use super::nodes::{EdgeMatrix, NodeStates};
use super::revision::SupportTag;
use super::splitter::SplitterContext;
use super::{
    FINAL_MIX_ID, INVALID_DISTANCE_FROM_FINAL_MIX, MAX_MIX_BUFFERS, UNUSED_MIX_ID,
    UNUSED_SPLITTER_ID,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MixInParameter {
    pub volume: f32,
    pub sample_rate: u32,
    pub buffer_count: u32,
    pub in_use: u8,
    pub is_dirty: u8,
    pub _pad: [u8; 2],
    pub mix_id: i32,
    pub effect_count: u32,
    pub node_id: i32,
    pub _unk_01c: [u8; 8],
    pub mix_volumes: [[f32; MAX_MIX_BUFFERS]; MAX_MIX_BUFFERS],
    pub dest_mix_id: i32,
    pub dest_splitter_id: i32,
    pub _unk_92c: [u8; 4],
}
const _: () = assert!(size_of::<MixInParameter>() == 0x930);

/// Header of a dirty-only mix section (revision 7+).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MixInDirtyParameter {
    pub magic: u32,
    pub count: i32,
    pub _unk_08: [u8; 0x18],
}
const _: () = assert!(size_of::<MixInDirtyParameter>() == 0x20);

/// One mix. Mix 0 is the final mix.
#[derive(Clone, Debug)]
pub struct MixInfo {
    pub volume: f32,
    pub sample_rate: u32,
    pub buffer_count: i16,
    pub in_use: bool,
    pub mix_id: i32,
    pub node_id: i32,
    pub buffer_offset: i16,
    pub distance_from_final_mix: i32,
    pub dst_mix_id: i32,
    pub mix_volumes: [[f32; MAX_MIX_BUFFERS]; MAX_MIX_BUFFERS],
    pub dst_splitter_id: i32,
    pub long_size_pre_delay_supported: bool,
}

impl MixInfo {
    fn new(behavior: &BehaviorInfo) -> Self {
        Self {
            volume: 0.0,
            sample_rate: 0,
            buffer_count: 0,
            in_use: false,
            mix_id: UNUSED_MIX_ID,
            node_id: 0,
            buffer_offset: 0,
            distance_from_final_mix: INVALID_DISTANCE_FROM_FINAL_MIX,
            dst_mix_id: UNUSED_MIX_ID,
            mix_volumes: [[0.0; MAX_MIX_BUFFERS]; MAX_MIX_BUFFERS],
            dst_splitter_id: UNUSED_SPLITTER_ID,
            long_size_pre_delay_supported: behavior.supports(SupportTag::LongSizePreDelay),
        }
    }

    pub fn cleanup(&mut self) {
        self.mix_id = UNUSED_MIX_ID;
        self.dst_mix_id = UNUSED_MIX_ID;
        self.dst_splitter_id = UNUSED_SPLITTER_ID;
    }

    pub fn has_any_connection(&self) -> bool {
        self.dst_mix_id != UNUSED_MIX_ID || self.dst_splitter_id != UNUSED_SPLITTER_ID
    }

    fn update_connection(
        &mut self,
        edges: &mut EdgeMatrix,
        params: &MixInParameter,
        splitters: &SplitterContext,
    ) -> bool {
        let has_new_connection = self.dst_splitter_id != UNUSED_SPLITTER_ID
            && splitters
                .info(self.dst_splitter_id)
                .is_some_and(|splitter| splitter.has_new_connection());
        if self.dst_mix_id == params.dest_mix_id
            && self.dst_splitter_id == params.dest_splitter_id
            && !has_new_connection
        {
            return false;
        }

        let id = self.mix_id as u32;
        edges.remove_edges(id);
        if params.dest_mix_id == UNUSED_MIX_ID {
            if params.dest_splitter_id != UNUSED_SPLITTER_ID {
                let destinations = splitters
                    .info(params.dest_splitter_id)
                    .map_or(0, |splitter| splitter.destination_count());
                for i in 0..destinations {
                    if let Some(destination) =
                        splitters.destination_data(params.dest_splitter_id, i)
                    {
                        let destination_id = destination.mix_id();
                        if destination_id != UNUSED_MIX_ID {
                            edges.connect(id, destination_id as u32);
                        }
                    }
                }
            }
        } else {
            edges.connect(id, params.dest_mix_id as u32);
        }
        self.dst_mix_id = params.dest_mix_id;
        self.dst_splitter_id = params.dest_splitter_id;
        true
    }
}

/// All mixes, their render order and effect order.
#[derive(Clone, Debug)]
pub struct MixContext {
    infos: Vec<MixInfo>,
    sorted: Vec<u32>,
    effect_order: Vec<i32>,
    effect_count: usize,
    node_states: NodeStates,
    edge_matrix: EdgeMatrix,
}

impl MixContext {
    /// `count` mixes (sub mixes plus the final mix), each able to chain `effect_count` effects.
    pub fn new(count: u32, effect_count: u32, behavior: &BehaviorInfo) -> Self {
        let graph_count = if behavior.supports(SupportTag::Splitter) {
            count
        } else {
            0
        };
        Self {
            infos: vec![MixInfo::new(behavior); count as usize],
            sorted: (0..count).collect(),
            effect_order: vec![-1; count as usize * effect_count as usize],
            effect_count: effect_count as usize,
            node_states: NodeStates::new(graph_count),
            edge_matrix: EdgeMatrix::new(graph_count),
        }
    }

    pub fn count(&self) -> u32 {
        self.infos.len() as u32
    }

    pub fn info(&self, index: usize) -> &MixInfo {
        &self.infos[index]
    }

    pub fn infos(&self) -> &[MixInfo] {
        &self.infos
    }

    pub fn final_mix(&self) -> &MixInfo {
        &self.infos[0]
    }

    /// Mix indices in render order: farthest from the final mix first.
    pub fn sorted(&self) -> &[u32] {
        &self.sorted
    }

    /// Effect indices of mix `index` by process order; -1 marks an empty slot.
    pub fn effect_order(&self, index: usize) -> &[i32] {
        &self.effect_order[index * self.effect_count..(index + 1) * self.effect_count]
    }

    pub fn edge_matrix(&self) -> &EdgeMatrix {
        &self.edge_matrix
    }

    pub(crate) fn clear_effect_order(&mut self, index: usize) {
        let count = self.effect_count;
        self.effect_order[index * count..(index + 1) * count].fill(-1);
    }

    pub(crate) fn info_mut(&mut self, index: usize) -> &mut MixInfo {
        &mut self.infos[index]
    }

    /// Applies one mix's input. Returns whether the graph needs sorting again.
    pub(crate) fn update(
        &mut self,
        index: usize,
        params: &MixInParameter,
        effects: &EffectContext,
        splitters: &SplitterContext,
        behavior: &BehaviorInfo,
    ) -> bool {
        let info = &mut self.infos[index];
        info.volume = params.volume;
        info.sample_rate = params.sample_rate;
        info.buffer_count = params.buffer_count as i16;
        info.in_use = params.in_use != 0;
        info.mix_id = params.mix_id;
        info.node_id = params.node_id;
        info.mix_volumes = params.mix_volumes;

        let sort_required = if behavior.supports(SupportTag::Splitter) {
            info.update_connection(&mut self.edge_matrix, params, splitters)
        } else {
            info.dst_splitter_id = UNUSED_SPLITTER_ID;
            if info.dst_mix_id != params.dest_mix_id {
                info.dst_mix_id = params.dest_mix_id;
                true
            } else {
                false
            }
        };

        let mix_id = info.mix_id;
        self.clear_effect_order(index);
        let order = &mut self.effect_order[index * self.effect_count..][..self.effect_count];
        for (effect_index, effect) in effects.infos().iter().enumerate() {
            if effect.mix_id != mix_id {
                continue;
            }
            let Some(slot) = usize::try_from(effect.process_order)
                .ok()
                .and_then(|process_order| order.get_mut(process_order))
            else {
                break;
            };
            *slot = effect_index as i32;
        }
        sort_required
    }

    fn update_distances_from_final_mix(&mut self) {
        let count = self.infos.len() as i32;
        for info in &mut self.infos {
            info.distance_from_final_mix = INVALID_DISTANCE_FROM_FINAL_MIX;
        }
        for i in 0..self.infos.len() {
            self.sorted[i] = i as u32;
            if !self.infos[i].in_use {
                continue;
            }
            let mut mix_id = self.infos[i].mix_id;
            let mut distance = FINAL_MIX_ID;
            while distance < count {
                if mix_id == FINAL_MIX_ID {
                    break;
                }
                let Some(next) = usize::try_from(mix_id)
                    .ok()
                    .filter(|_| mix_id != UNUSED_MIX_ID)
                    .and_then(|id| self.infos.get(id))
                else {
                    distance = INVALID_DISTANCE_FROM_FINAL_MIX;
                    break;
                };
                if next.distance_from_final_mix != INVALID_DISTANCE_FROM_FINAL_MIX {
                    distance = next.distance_from_final_mix + 1;
                    break;
                }
                distance += 1;
                mix_id = next.dst_mix_id;
            }
            if distance >= count {
                distance = INVALID_DISTANCE_FROM_FINAL_MIX;
            }
            self.infos[i].distance_from_final_mix = distance;
        }
    }

    /// Orders mixes by distance from the final mix, farthest first, ties by index.
    pub fn sort_info(&mut self) {
        self.update_distances_from_final_mix();
        let infos = &self.infos;
        self.sorted.sort_unstable_by(|&a, &b| {
            infos[b as usize]
                .distance_from_final_mix
                .cmp(&infos[a as usize].distance_from_final_mix)
                .then(a.cmp(&b))
        });
        self.calc_mix_buffer_offset();
    }

    /// Topological order through splitter routes. Fails when the graph has a cycle.
    pub fn tsort_info(&mut self, splitters: &SplitterContext) -> bool {
        if !splitters.using_splitter() {
            self.calc_mix_buffer_offset();
            return true;
        }
        if !self.node_states.tsort(&self.edge_matrix) {
            return false;
        }
        let count = self.sorted.len();
        for (slot, id) in self
            .sorted
            .iter_mut()
            .zip(self.node_states.sorted_results())
            .take(count)
        {
            *slot = id;
        }
        self.calc_mix_buffer_offset();
        true
    }

    /// Hands out mix buffers in render order.
    pub fn calc_mix_buffer_offset(&mut self) {
        let mut offset: i16 = 0;
        for &index in &self.sorted {
            let Some(info) = self.infos.get_mut(index as usize) else {
                continue;
            };
            if info.in_use {
                info.buffer_offset = offset;
                offset = offset.wrapping_add(info.buffer_count);
            }
        }
    }
}
