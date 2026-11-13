//! Replays \`RequestUpdate\` fixtures produced by the C++ reference.
//!
//! Each fixture in \`golden/\` holds the renderer's open parameters and a sequence of synthetic
//! updates with the input, the reference's output, its performance buffer and its result code.
//! \`golden/golden.cpp\` builds the inputs with the reference's own structs and records what the
//! reference returns, so these tests pin the input ABI, the output bytes and, through a digest of
//! every context, the state each update leaves behind.

mod support;

use lichii_audio::renderer::RendererSystem;
use lichii_audio::renderer::parameter::{AudioRendererParameterInternal, work_buffer_size};
use support::{Reader, digest, first_difference_words};

fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
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
