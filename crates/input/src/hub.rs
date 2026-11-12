//! The hand-off between the frontend and the hid service: one seqlock per player, so the input
//! thread publishes and hid samples without locks, allocation or blocking each other.

use std::sync::atomic::{AtomicU64, Ordering, fence};

use lichii_common::settings::{ControllerType, MAX_PLAYERS};

use crate::npad::{MotionSample, NpadButtons, NpadState, StickPosition, VibrationValue};

const STATE_WORDS: usize = 8;
const VIBRATION_WORDS: usize = 2;

/// A single-writer, many-reader cell of `N` words. Readers retry while a write is in flight;
/// writes never wait.
struct SeqCell<const N: usize> {
    seq: AtomicU64,
    words: [AtomicU64; N],
}

impl<const N: usize> SeqCell<N> {
    fn new() -> Self {
        Self {
            seq: AtomicU64::new(0),
            words: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    fn write(&self, words: [u64; N]) {
        let seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(seq.wrapping_add(1), Ordering::Relaxed);
        fence(Ordering::Release);
        for (slot, word) in self.words.iter().zip(words) {
            slot.store(word, Ordering::Relaxed);
        }
        self.seq.store(seq.wrapping_add(2), Ordering::Release);
    }

    /// The words and the sequence number they were written under (0 = never written).
    fn read(&self) -> ([u64; N], u64) {
        loop {
            let before = self.seq.load(Ordering::Acquire);
            if before & 1 == 1 {
                std::hint::spin_loop();
                continue;
            }
            let words = std::array::from_fn(|i| self.words[i].load(Ordering::Relaxed));
            fence(Ordering::Acquire);
            if self.seq.load(Ordering::Relaxed) == before {
                return (words, before);
            }
        }
    }
}

/// Npad state for players 1-8 plus the rumble the guest asks for. Share it as `Arc<NpadHub>`.
///
/// The frontend's input thread is the only writer of states (`publish`); the hid service is the
/// only writer of vibration (`request_vibration`). Handheld mode is player 1's slot with a
/// handheld `controller`; hid maps its handheld npad id there.
pub struct NpadHub {
    states: [SeqCell<STATE_WORDS>; MAX_PLAYERS],
    vibration: [SeqCell<VIBRATION_WORDS>; MAX_PLAYERS],
}

impl Default for NpadHub {
    fn default() -> Self {
        Self {
            states: std::array::from_fn(|_| SeqCell::new()),
            vibration: std::array::from_fn(|_| SeqCell::new()),
        }
    }
}

impl NpadHub {
    pub fn publish(&self, player: usize, state: &NpadState) {
        self.states[player].write(pack_state(state));
    }

    /// The latest state of `player` (0-based); disconnected until the frontend publishes one.
    pub fn read(&self, player: usize) -> NpadState {
        unpack_state(self.states[player].read().0)
    }

    pub fn request_vibration(&self, player: usize, value: VibrationValue) {
        let f = |v: f32| u64::from(v.to_bits());
        self.vibration[player].write([
            f(value.low_amplitude) | f(value.low_frequency) << 32,
            f(value.high_amplitude) | f(value.high_frequency) << 32,
        ]);
    }

    /// The newest vibration request for `player` if it changed since `*seen`, which it updates.
    pub fn take_vibration(&self, player: usize, seen: &mut u64) -> Option<VibrationValue> {
        let (words, seq) = self.vibration[player].read();
        if seq == *seen {
            return None;
        }
        *seen = seq;
        let lo = |w: u64| f32::from_bits(w as u32);
        let hi = |w: u64| f32::from_bits((w >> 32) as u32);
        Some(VibrationValue {
            low_amplitude: lo(words[0]),
            low_frequency: hi(words[0]),
            high_amplitude: lo(words[1]),
            high_frequency: hi(words[1]),
        })
    }
}

fn pair(a: u32, b: u32) -> u64 {
    u64::from(a) | u64::from(b) << 32
}

fn split(word: u64) -> (u32, u32) {
    (word as u32, (word >> 32) as u32)
}

fn pack_state(s: &NpadState) -> [u64; STATE_WORDS] {
    let motion = s.motion.unwrap_or_default();
    let controller = ControllerType::ALL
        .iter()
        .position(|&c| c == s.controller)
        .unwrap_or(0) as u64;
    let flags = controller | u64::from(s.connected) << 8 | u64::from(s.motion.is_some()) << 9;
    let b = f32::to_bits;
    [
        s.buttons.bits(),
        pair(s.left_stick.x as u32, s.left_stick.y as u32),
        pair(s.right_stick.x as u32, s.right_stick.y as u32),
        pair(s.left_trigger, s.right_trigger),
        pair(b(motion.accel[0]), b(motion.accel[1])),
        pair(b(motion.accel[2]), b(motion.gyro[0])),
        pair(b(motion.gyro[1]), b(motion.gyro[2])),
        flags,
    ]
}

fn unpack_state(w: [u64; STATE_WORDS]) -> NpadState {
    let stick = |word| {
        let (x, y) = split(word);
        StickPosition {
            x: x as i32,
            y: y as i32,
        }
    };
    let f = f32::from_bits;
    let (ax, ay) = split(w[4]);
    let (az, gx) = split(w[5]);
    let (gy, gz) = split(w[6]);
    let (left_trigger, right_trigger) = split(w[3]);
    let flags = w[7];
    NpadState {
        connected: flags & 1 << 8 != 0,
        controller: ControllerType::ALL[(flags & 0xff) as usize % ControllerType::ALL.len()],
        buttons: NpadButtons::from_bits_retain(w[0]),
        left_stick: stick(w[1]),
        right_stick: stick(w[2]),
        left_trigger,
        right_trigger,
        motion: (flags & 1 << 9 != 0).then(|| MotionSample {
            accel: [f(ax), f(ay), f(az)],
            gyro: [f(gx), f(gy), f(gz)],
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn sample() -> NpadState {
        NpadState {
            connected: true,
            controller: ControllerType::Switch2RightJoycon,
            buttons: NpadButtons::A | NpadButtons::C | NpadButtons::STICK_R_DOWN,
            left_stick: StickPosition { x: -0x7fff, y: 12 },
            right_stick: StickPosition { x: 300, y: -0x4000 },
            left_trigger: 0x7fff,
            right_trigger: 7,
            motion: Some(MotionSample {
                accel: [0.0, -1.0, 0.25],
                gyro: [1.5, -0.125, 3.0],
            }),
        }
    }

    #[test]
    fn states_round_trip() {
        let hub = NpadHub::default();
        assert_eq!(hub.read(3), NpadState::default());
        hub.publish(3, &sample());
        assert_eq!(hub.read(3), sample());
        let no_motion = NpadState {
            motion: None,
            ..sample()
        };
        hub.publish(3, &no_motion);
        assert_eq!(hub.read(3), no_motion);
        assert_eq!(hub.read(2), NpadState::default());
    }

    #[test]
    fn every_controller_type_round_trips() {
        let hub = NpadHub::default();
        for controller in ControllerType::ALL {
            let state = NpadState {
                controller,
                ..sample()
            };
            hub.publish(0, &state);
            assert_eq!(hub.read(0).controller, controller);
        }
    }

    #[test]
    fn vibration_is_taken_once_per_request() {
        let hub = NpadHub::default();
        let mut seen = 0;
        assert_eq!(hub.take_vibration(1, &mut seen), None);
        let value = VibrationValue {
            low_amplitude: 0.5,
            low_frequency: 160.0,
            high_amplitude: 0.25,
            high_frequency: 320.0,
        };
        hub.request_vibration(1, value);
        assert_eq!(hub.take_vibration(1, &mut seen), Some(value));
        assert_eq!(hub.take_vibration(1, &mut seen), None);
        hub.request_vibration(1, value);
        assert_eq!(hub.take_vibration(1, &mut seen), Some(value));
    }

    #[test]
    fn readers_never_see_torn_states() {
        let hub = Arc::new(NpadHub::default());
        let writer = {
            let hub = hub.clone();
            std::thread::spawn(move || {
                for i in 0..20_000i32 {
                    let state = NpadState {
                        connected: true,
                        left_stick: StickPosition { x: i, y: -i },
                        right_stick: StickPosition { x: i, y: i },
                        left_trigger: i as u32,
                        ..NpadState::default()
                    };
                    hub.publish(0, &state);
                }
            })
        };
        for _ in 0..20_000 {
            let s = hub.read(0);
            assert_eq!(s.left_stick.x, -s.left_stick.y);
            assert_eq!(s.left_stick.x, s.right_stick.x);
            assert_eq!(s.right_stick.x, s.right_stick.y);
            assert_eq!(s.left_trigger, s.left_stick.x as u32);
        }
        writer.join().unwrap();
    }
}
