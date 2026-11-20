//! Mario Kart World's adaptive frame mode. The title runs one 1/60 s game step per frame (mode 1)
//! or one 1/30 s step per frame (mode 2, full speed at half the frame rate), as the word its
//! patched mode load reads says. Frames that keep arriving late at 60 FPS move it to 30 FPS; it
//! tries 60 FPS again after a back-off that doubles while 60 FPS keeps failing at once.

use std::time::{Duration, Instant};

use lichii_loader::{FRAME_MODE_SIXTY, FRAME_MODE_THIRTY};

/// A present further than this after the previous one is late (a 60 FPS frame takes 16.7 ms).
const LATE: Duration = Duration::from_millis(19);
/// Consecutive late presents that give up 60 FPS.
const LATE_PRESENTS: u32 = 3;
const MIN_BACKOFF: Duration = Duration::from_secs(2);
const MAX_BACKOFF: Duration = Duration::from_secs(16);
/// A 60 FPS try that fails within this long of its start doubles the back-off.
const QUICK_FAILURE: Duration = Duration::from_secs(5);

pub(super) struct FrameMode {
    mode: u32,
    last_present: Option<Instant>,
    late: u32,
    /// When the current mode began.
    since: Option<Instant>,
    backoff: Duration,
    /// When 30 FPS tries 60 FPS again.
    retry: Instant,
}

impl FrameMode {
    pub(super) fn new(now: Instant) -> Self {
        Self {
            mode: FRAME_MODE_SIXTY,
            last_present: None,
            late: 0,
            since: None,
            backoff: MIN_BACKOFF,
            retry: now,
        }
    }

    /// Notes a present at `now` and returns the mode to write to the title's mode word when it
    /// changed.
    pub(super) fn present(&mut self, now: Instant) -> Option<u32> {
        let late = self
            .last_present
            .replace(now)
            .is_some_and(|at| now.saturating_duration_since(at) > LATE);
        let next = if self.mode == FRAME_MODE_SIXTY {
            self.late = if late { self.late + 1 } else { 0 };
            if self.late < LATE_PRESENTS {
                return None;
            }
            let quick = self
                .since
                .is_some_and(|since| now.saturating_duration_since(since) < QUICK_FAILURE);
            self.backoff = if quick {
                (self.backoff * 2).clamp(MIN_BACKOFF, MAX_BACKOFF)
            } else {
                MIN_BACKOFF
            };
            self.retry = now + self.backoff;
            self.late = 0;
            FRAME_MODE_THIRTY
        } else if now >= self.retry {
            FRAME_MODE_SIXTY
        } else {
            return None;
        };
        self.mode = next;
        self.since = Some(now);
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FAST: Duration = Duration::from_millis(16);
    const SLOW: Duration = Duration::from_millis(40);

    /// Presents `count` frames `gap` apart from `*now`.
    fn run(mode: &mut FrameMode, now: &mut Instant, gap: Duration, count: u32) -> Vec<u32> {
        (0..count)
            .filter_map(|_| {
                *now += gap;
                mode.present(*now)
            })
            .collect()
    }

    #[test]
    fn three_late_presents_in_a_row_give_up_60_fps() {
        let mut now = Instant::now();
        let mut mode = FrameMode::new(now);
        assert_eq!(run(&mut mode, &mut now, SLOW, 3), Vec::<u32>::new());
        assert_eq!(run(&mut mode, &mut now, SLOW, 1), [FRAME_MODE_THIRTY]);
    }

    #[test]
    fn an_on_time_present_resets_the_count() {
        let mut now = Instant::now();
        let mut mode = FrameMode::new(now);
        for _ in 0..5 {
            assert_eq!(run(&mut mode, &mut now, SLOW, 2), Vec::<u32>::new());
            assert!(run(&mut mode, &mut now, FAST, 1).is_empty());
        }
    }

    #[test]
    fn sixty_fps_is_tried_again_after_the_back_off() {
        let mut now = Instant::now();
        let mut mode = FrameMode::new(now);
        assert_eq!(run(&mut mode, &mut now, SLOW, 4), [FRAME_MODE_THIRTY]);
        assert_eq!(run(&mut mode, &mut now, SLOW, 45), Vec::<u32>::new());
        assert_eq!(run(&mut mode, &mut now, SLOW, 5), [FRAME_MODE_SIXTY]);
    }

    #[test]
    fn quick_failures_double_the_back_off_up_to_16_seconds() {
        let mut now = Instant::now();
        let mut mode = FrameMode::new(now);
        let mut backoffs = Vec::new();
        for _ in 0..6 {
            run(&mut mode, &mut now, SLOW, 5);
            backoffs.push(mode.backoff);
            now = mode.retry;
            assert_eq!(mode.present(now), Some(FRAME_MODE_SIXTY));
        }
        let secs: Vec<u64> = backoffs.iter().map(Duration::as_secs).collect();
        assert_eq!(secs, [2, 4, 8, 16, 16, 16]);
    }

    #[test]
    fn a_lasting_60_fps_resets_the_back_off() {
        let mut now = Instant::now();
        let mut mode = FrameMode::new(now);
        run(&mut mode, &mut now, SLOW, 5);
        run(&mut mode, &mut now, SLOW, 5);
        now = mode.retry;
        assert_eq!(mode.present(now), Some(FRAME_MODE_SIXTY));
        run(&mut mode, &mut now, FAST, 400);
        assert_eq!(run(&mut mode, &mut now, SLOW, 3), [FRAME_MODE_THIRTY]);
        assert_eq!(mode.backoff, MIN_BACKOFF);
    }
}
