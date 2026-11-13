//! Replays \`RequestUpdate\` fixtures produced by the C++ reference.
//!
//! Each fixture in \`golden/\` holds the renderer's open parameters and a sequence of synthetic
//! updates with the input, the reference's output, its performance buffer and its result code.
//! \`golden/golden.cpp\` builds the inputs with the reference's own structs and records what the
//! reference returns, so these tests pin the input ABI, the output bytes and, through a digest of
//! every context, the state each update leaves behind.

use lichii_audio::renderer::RendererSystem;
use lichii_audio::renderer::memory::AddressInfo;
use lichii_audio::renderer::parameter::{AudioRendererParameterInternal, work_buffer_size};
use lichii_audio::renderer::revision::SupportTag;
use lichii_audio::renderer::sink::SinkKind;

struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, len: usize) -> &'a [u8] {
        let out = &self.data[self.offset..self.offset + len];
        self.offset += len;
        out
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.bytes(4).try_into().unwrap())
    }
}

#[derive(Default)]
struct Digest(Vec<u32>);

impl Digest {
    fn u(&mut self, value: impl Into<u32>) {
        self.0.push(value.into());
    }

    fn s(&mut self, value: impl Into<i32>) {
        self.0.push(value.into() as u32);
    }

    fn f(&mut self, value: f32) {
        self.0.push(value.to_bits());
    }

    fn q(&mut self, value: u64) {
        self.0.push(value as u32);
        self.0.push((value >> 32) as u32);
    }

    fn address(&mut self, address: &AddressInfo) {
        self.q(address.cpu_address());
        self.q(address.size());
        self.u(address.pool().unwrap_or(u32::MAX));
        self.q(address.force_mapped_dsp_address());
    }
}

/// The reference ORs the pitch-and-SRC-skipped request into bit 0 instead of bit 1, so the DSP
/// never sees it. This port keeps it in bit 1; fold it back to compare against the reference.
fn reference_voice_flags(flags: u16) -> u32 {
    u32::from((flags & 1) | ((flags >> 1) & 1))
}

/// Mirrors \`DigestState\` in \`golden/golden.cpp\`.
fn digest(system: &RendererSystem) -> Vec<u32> {
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

fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
    actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .or((actual.len() != expected.len()).then(|| actual.len().min(expected.len())))
}

fn first_difference_words(actual: &[u32], expected: &[u32]) -> Option<usize> {
    actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a != b)
        .or((actual.len() != expected.len()).then(|| actual.len().min(expected.len())))
}

fn replay(name: &str, fixture: &[u8]) {
    let mut reader = Reader {
        data: fixture,
        offset: 0,
    };
    assert_eq!(reader.bytes(4), b"LRUG");
    assert_eq!(reader.u32(), 2);
    let params: AudioRendererParameterInternal =
        bytemuck::pod_read_unaligned(reader.bytes(size_of::<AudioRendererParameterInternal>()));
    let active = reader.u32() != 0;
    let steps = reader.u32();

    let mut system = RendererSystem::new(&params, work_buffer_size(&params))
        .unwrap_or_else(|code| panic!("{name}: open failed with {code:?}"));
    if active {
        system.start();
    }
    for step in 0..steps {
        let result = reader.u32();
        let input_len = reader.u32() as usize;
        let output_len = reader.u32() as usize;
        let perf_len = reader.u32() as usize;
        let input = reader.bytes(input_len);
        let expected_output = reader.bytes(output_len);
        let expected_perf = reader.bytes(perf_len);

        let mut output = vec![0xCD; output_len];
        let mut perf = vec![0; perf_len];
        let actual = match system.update(input, &mut perf, &mut output) {
            Ok(()) => 0,
            Err(code) => code.0,
        };
        assert_eq!(actual, result, "{name} step {step}: result");
        if let Some(at) = first_difference(&output, expected_output) {
            let end = (at + 16).min(output_len);
            panic!(
                "{name} step {step}: output differs at {at:#x}\n  actual   {:02x?}\n  expected {:02x?}",
                &output[at..end],
                &expected_output[at..end]
            );
        }
        if let Some(at) = first_difference(&perf, expected_perf) {
            panic!("{name} step {step}: performance buffer differs at {at:#x}");
        }
        let digest_len = reader.u32() as usize;
        let expected_digest: Vec<u32> = reader
            .bytes(digest_len * 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        let actual_digest = digest(&system);
        if let Some(at) = first_difference_words(&actual_digest, &expected_digest) {
            let end = (at + 8).min(actual_digest.len()).min(expected_digest.len());
            panic!(
                "{name} step {step}: state digest differs at word {at}\n  actual   {:08x?}\n  expected {:08x?}",
                &actual_digest[at..end],
                &expected_digest[at..end]
            );
        }
    }
    assert_eq!(reader.offset, fixture.len(), "{name}: trailing bytes");
}

macro_rules! golden {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                replay(
                    stringify!($name),
                    include_bytes!(concat!("golden/", stringify!($name), ".bin")),
                );
            }
        )*
    };
}

golden!(
    rev1,
    rev5_perf,
    rev9_effects_v2,
    rev13_splitter_v2a,
    rev15_current,
    rev15_opened_rev_tagged_rev_lower,
    rev15_native,
    rev16_native,
    fault_behavior_size,
    fault_voices_size,
    fault_revision_mismatch,
    fault_crossfade,
);
