//! `csrng`: random bytes for `nn::crypto` and `nn::os`. The bytes come from a generator seeded with
//! `system.rng_seed`, or the host clock when it is unset, so runs with a seed repeat.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use lichii_common::{HleResult, ResultCode};
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use crate::{Service, ServiceContext};

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let seed = context.settings.system.rng_seed.map_or_else(
        || {
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |time| time.as_nanos() as u64)
        },
        u64::from,
    );
    let state = Arc::new(Mutex::new(seed));
    context.register("csrng", "services", move |_| Csrng {
        state: state.clone(),
    })
}

struct Csrng {
    /// SplitMix64 state, shared by every session.
    state: Arc<Mutex<u64>>,
}

impl Service for Csrng {
    fn name(&self) -> &str {
        "csrng"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        if ctx.command() != 0 {
            return None;
        }
        let size = ctx.writable_buffer(0).map_or(0, |buffer| buffer.size) as usize;
        let mut bytes = vec![0; size];
        let mut state = self.state.lock();
        for chunk in bytes.chunks_mut(8) {
            *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = *state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            let z = z ^ (z >> 31);
            chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
        }
        drop(state);
        Some(
            ctx.write_buffer(0, &bytes)
                .err()
                .unwrap_or(ResultCode::SUCCESS),
        )
    }
}

#[cfg(test)]
mod tests {
    use lichii_common::ResultCode;
    use lichii_common::settings::Settings;

    use crate::testing::{Request, SWITCH1_TITLE, booted};

    #[test]
    fn random_bytes_fill_the_buffer_and_repeat_with_a_seed() {
        let mut settings = Settings::default();
        settings.system.rng_seed = Some(7);
        let draw = |settings: &Settings| {
            let client = booted(SWITCH1_TITLE, settings.clone(), super::register);
            let csrng = client.connect("csrng").unwrap();
            let buffer = client.alloc(&[0; 0x20]);
            let reply = client
                .send(csrng, &Request::new(0).buffer_b(buffer, 0x13))
                .unwrap();
            assert_eq!(reply.result, ResultCode::SUCCESS);
            let first = client.read(buffer, 0x20);
            client
                .send(csrng, &Request::new(0).buffer_b(buffer, 0x13))
                .unwrap();
            (first, client.read(buffer, 0x20))
        };
        let (first, second) = draw(&settings);
        assert!(first[..0x13].iter().any(|&b| b != 0));
        assert_eq!(&first[0x13..], &[0; 0xd], "nothing past the buffer");
        assert_ne!(first, second, "each call draws new bytes");
        assert_eq!(draw(&settings), (first, second), "a seed repeats the bytes");
    }
}
