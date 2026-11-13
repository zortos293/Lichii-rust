//! Pieces the golden replays share: a fixture reader and the renderer state digest that
//! golden/golden.cpp computes as DigestState.

use lichii_audio::renderer::RendererSystem;
use lichii_audio::renderer::memory::AddressInfo;
use lichii_audio::renderer::revision::SupportTag;
use lichii_audio::renderer::sink::SinkKind;

pub struct Reader<'a> {
    pub data: &'a [u8],
    pub offset: usize,
}

impl<'a> Reader<'a> {
    pub fn bytes(&mut self, len: usize) -> &'a [u8] {
        let out = &self.data[self.offset..self.offset + len];
        self.offset += len;
        out
    }

    pub fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
}

#[derive(Default)]
pub struct Digest(pub Vec<u32>);

impl Digest {
    pub fn u(&mut self, value: impl Into<u32>) {
        self.0.push(value.into());
    }

    pub fn s(&mut self, value: impl Into<i32>) {
        self.0.push(value.into() as u32);
    }

    pub fn f(&mut self, value: f32) {
        self.0.push(value.to_bits());
    }

    pub fn q(&mut self, value: u64) {
        self.0.push(value as u32);
        self.0.push((value >> 32) as u32);
    }

    pub fn address(&mut self, address: &AddressInfo) {
        self.q(address.cpu_address());
        self.q(address.size());
        self.u(address.pool().unwrap_or(u32::MAX));
        self.q(address.force_mapped_dsp_address());
    }
}

/// The reference ORs the pitch-and-SRC-skipped request into bit 0 instead of bit 1, so the DSP
/// never sees it. This port keeps it in bit 1; fold it back to compare against the reference.
pub fn reference_voice_flags(flags: u16) -> u32 {
    u32::from((flags & 1) | ((flags >> 1) & 1))
}

/// Mirrors \`DigestState\` in \`golden/golden.cpp\`.
pub fn digest(system: &RendererSystem) -> Vec<u32> {
    let mut d = Digest::default();
    for pool in &system.memory_pools {
        d.q(pool.cpu_address());
        d.q(pool.dsp_address());
        d.q(pool.size());
        d.u(pool.location() as u32);
        d.u(pool.is_used());
    }

    let voices = &system.voices;
    for resource in voices.channel_resources() {
        d.u(resource.in_use);
        resource.mix_volumes.iter().for_each(|&v| d.f(v));
        resource.prev_mix_volumes.iter().for_each(|&v| d.f(v));
    }
    for v in voices.infos() {
        d.u(v.in_use);
        d.u(v.is_new);
        d.u(v.was_playing);
        d.u(v.sample_format as u32);
        d.u(v.sample_rate);
        d.s(v.channel_count);
        d.u(v.id);
        d.u(v.node_id);
        d.u(v.mix_id);
        d.u(v.current_play_state as u32);
        d.u(v.last_play_state as u32);
        d.s(v.priority);
        d.s(v.sort_order);
        d.f(v.pitch);
        d.f(v.volume);
        d.f(v.prev_volume);
        for b in &v.biquads {
            d.u(b.enabled);
            b.b.iter().for_each(|&x| d.s(x));
            b.a.iter().for_each(|&x| d.s(x));
        }
        for b in &v.biquads_float {
            d.u(b.enabled);
            b.numerator.iter().for_each(|&x| d.f(x));
            b.denominator.iter().for_each(|&x| d.f(x));
        }
        d.u(v.use_float_biquads);
        d.u(v.wave_buffer_count);
        d.u(v.wave_buffer_index);
        d.u(reference_voice_flags(v.flags));
        d.address(&v.data_address);
        for w in &v.wave_buffers {
            d.address(&w.buffer_address);
            d.address(&w.context_address);
            d.u(w.start_offset);
            d.u(w.end_offset);
            d.u(w.looping);
            d.u(w.stream_ended);
            d.u(w.sent_to_dsp);
            d.u(w.loop_start_offset);
            d.u(w.loop_end_offset);
            d.s(w.loop_count);
        }
        v.channel_resource_ids.iter().for_each(|&id| d.u(id));
        d.s(v.splitter_id);
        d.u(v.src_quality as u32);
        d.u(v.voice_dropped);
        d.u(v.data_unmapped);
        d.u(v.buffer_unmapped);
    }
    for s in voices.states() {
        d.q(s.played_sample_count);
        d.u(s.offset);
        d.u(s.wave_buffer_index);
        s.wave_buffer_valid.iter().for_each(|&valid| d.u(valid));
        d.u(s.wave_buffers_consumed);
    }

    for e in system.effects.infos() {
        d.u(e.effect_type as u32);
        d.u(e.enabled);
        d.u(e.buffer_unmapped);
        d.u(e.usage_state as u32);
        d.s(e.mix_id);
        d.s(e.process_order);
        d.address(&e.workbuffers[0]);
        d.address(&e.workbuffers[1]);
        d.q(e.send_buffer_info);
        d.q(e.send_buffer);
        d.q(e.return_buffer_info);
        d.q(e.return_buffer);
        for word in e.parameter[..0xA0].as_chunks::<4>().0 {
            d.u(u32::from_le_bytes(*word));
        }
    }

    let mixes = &system.mixes;
    for &index in mixes.sorted() {
        d.s(mixes.info(index as usize).mix_id);
    }
    for (i, m) in mixes.infos().iter().enumerate() {
        d.u(m.in_use);
        d.s(m.mix_id);
        d.s(m.node_id);
        d.f(m.volume);
        d.u(m.sample_rate);
        d.s(m.buffer_count);
        d.s(m.buffer_offset);
        d.s(m.distance_from_final_mix);
        d.s(m.dst_mix_id);
        d.s(m.dst_splitter_id);
        d.f(m.mix_volumes[0][0]);
        d.f(m.mix_volumes[1][1]);
        d.f(m.mix_volumes[5][5]);
        mixes.effect_order(i).iter().for_each(|&order| d.s(order));
    }

    if system.behavior.supports(SupportTag::Splitter) {
        let count = mixes.count();
        for i in 0..count {
            for j in 0..count {
                d.u(mixes.edge_matrix().connected(i, j));
            }
        }
        let splitters = &system.splitters;
        for info in splitters.infos() {
            d.s(info.id());
            d.u(info.sample_rate());
            d.u(info.destination_count());
            d.u(info.has_new_connection());
            d.u(info.head().unwrap_or(u32::MAX));
        }
        for i in 0..splitters.data_count() {
            let dest = splitters.data(i).unwrap();
            d.s(dest.id());
            d.s(dest.mix_id());
            d.u(dest.is_configured());
            dest.mix_volumes()[..2].iter().for_each(|&v| d.f(v));
            dest.prev_mix_volumes()[..2].iter().for_each(|&v| d.f(v));
            let biquad = &dest.biquad_filters()[0];
            d.u(biquad.enabled);
            d.f(biquad.numerator[0]);
            d.f(biquad.denominator[1]);
            d.u(dest.next().unwrap_or(u32::MAX));
        }
    }

    for s in system.sinks.infos() {
        d.u(s.sink_type() as u32);
        d.u(s.in_use);
        d.u(s.buffer_unmapped);
        d.u(s.node_id);
        match &s.kind {
            SinkKind::Invalid => {}
            SinkKind::Device { state, .. } => state.downmix_coeff.iter().for_each(|&c| d.s(c)),
            SinkKind::CircularBuffer { state, .. } => {
                d.u(state.last_pos2);
                d.s(state.current_pos);
                d.u(state.last_pos);
                d.address(&state.address_info);
            }
        }
    }
    d.0
}

pub fn first_difference_words(actual: &[u32], expected: &[u32]) -> Option<usize> {
    actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .or((actual.len() != expected.len()).then(|| actual.len().min(expected.len())))
}
