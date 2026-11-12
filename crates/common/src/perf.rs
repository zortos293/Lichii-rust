//! Performance numbers for the frame view overlay. The renderer records presents and VRAM
//! use, the core records guest frames, and the frontend reads snapshots.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::{Duration, Instant};

const HISTORY: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug)]
pub struct FrameSample {
    pub presented_at: Instant,
    /// Time since the previous present.
    pub frame_time: Duration,
    /// Host GPU time of the frame's command buffers, when timestamps are available.
    pub gpu_time: Option<Duration>,
}

#[derive(Clone, Debug, Default)]
pub struct PerfSnapshot {
    /// Host presents in the last second.
    pub fps: f32,
    pub frame_time_ms: f32,
    /// Present-to-present times in milliseconds, oldest first, covering the requested window.
    pub frame_times_ms: Vec<f32>,
    /// Guest frames in the last second.
    pub game_fps: f32,
    /// Times between guest frames in milliseconds, oldest first, covering the requested window.
    pub game_frame_times_ms: Vec<f32>,
    pub gpu_time_ms: Option<f32>,
    /// Guest frames per second relative to the title's target rate (1.0 = full speed).
    pub emulation_speed: f32,
    pub vram_used: u64,
    pub vram_budget: u64,
    pub shaders_building: u32,
    pub render_width: u32,
    pub render_height: u32,
}

#[derive(Default)]
struct State {
    presents: VecDeque<FrameSample>,
    guest_frames: VecDeque<Instant>,
    guest_target_hz: f32,
    vram_used: u64,
    vram_budget: u64,
    shaders_building: u32,
    render_size: (u32, u32),
}

/// Shared, thread-safe performance counters. Writers call it at most a few times per frame.
#[derive(Default)]
pub struct PerfStats {
    state: Mutex<State>,
}

impl PerfStats {
    pub fn record_present(&self, sample: FrameSample) {
        let mut state = self.state.lock().unwrap();
        state.presents.push_back(sample);
        while state
            .presents
            .front()
            .is_some_and(|s| sample.presented_at.duration_since(s.presented_at) > HISTORY)
        {
            state.presents.pop_front();
        }
    }

    /// A guest frame finished (the title queued a buffer); `target_hz` is its intended rate.
    pub fn record_guest_frame(&self, at: Instant, target_hz: f32) {
        let mut state = self.state.lock().unwrap();
        state.guest_frames.push_back(at);
        state.guest_target_hz = target_hz;
        while state
            .guest_frames
            .front()
            .is_some_and(|t| at.duration_since(*t) > HISTORY)
        {
            state.guest_frames.pop_front();
        }
    }

    pub fn set_vram(&self, used: u64, budget: u64) {
        let mut state = self.state.lock().unwrap();
        state.vram_used = used;
        state.vram_budget = budget;
    }

    pub fn set_shaders_building(&self, count: u32) {
        self.state.lock().unwrap().shaders_building = count;
    }

    pub fn set_render_size(&self, width: u32, height: u32) {
        self.state.lock().unwrap().render_size = (width, height);
    }

    /// Numbers over the last `window` (e.g. the frame view's graph length).
    pub fn snapshot(&self, now: Instant, window: Duration) -> PerfSnapshot {
        let state = self.state.lock().unwrap();
        let recent: Vec<&FrameSample> = state
            .presents
            .iter()
            .filter(|s| now.duration_since(s.presented_at) <= window)
            .collect();
        let last_second = recent
            .iter()
            .filter(|s| now.duration_since(s.presented_at) <= Duration::from_secs(1))
            .count();
        let guest_last_second = state
            .guest_frames
            .iter()
            .filter(|t| now.duration_since(**t) <= Duration::from_secs(1))
            .count();
        let guest_recent: Vec<Instant> = state
            .guest_frames
            .iter()
            .copied()
            .filter(|t| now.duration_since(*t) <= window)
            .collect();
        let emulation_speed = if state.guest_target_hz > 0.0 {
            guest_last_second as f32 / state.guest_target_hz
        } else {
            0.0
        };
        PerfSnapshot {
            fps: last_second as f32,
            frame_time_ms: recent
                .last()
                .map_or(0.0, |s| s.frame_time.as_secs_f32() * 1000.0),
            frame_times_ms: recent
                .iter()
                .map(|s| s.frame_time.as_secs_f32() * 1000.0)
                .collect(),
            game_fps: guest_last_second as f32,
            game_frame_times_ms: guest_recent
                .windows(2)
                .map(|w| (w[1] - w[0]).as_secs_f32() * 1000.0)
                .collect(),
            gpu_time_ms: recent
                .last()
                .and_then(|s| s.gpu_time)
                .map(|t| t.as_secs_f32() * 1000.0),
            emulation_speed,
            vram_used: state.vram_used,
            vram_budget: state.vram_budget,
            shaders_building: state.shaders_building,
            render_width: state.render_size.0,
            render_height: state.render_size.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_frames_in_the_last_second() {
        let stats = PerfStats::default();
        let start = Instant::now();
        for i in 0..120u64 {
            let at = start + Duration::from_millis(i * 1000 / 60);
            let frame_time = Duration::from_micros(16_667);
            stats.record_present(FrameSample {
                presented_at: at,
                frame_time,
                gpu_time: None,
            });
            stats.record_guest_frame(at, 60.0);
        }
        let now = start + Duration::from_millis(119 * 1000 / 60);
        let snap = stats.snapshot(now, Duration::from_secs(5));
        assert!((59.0..=61.0).contains(&snap.fps), "{}", snap.fps);
        assert!(
            (0.98..=1.02).contains(&snap.emulation_speed),
            "{}",
            snap.emulation_speed
        );
        assert_eq!(snap.frame_times_ms.len(), 120);
        assert!((59.0..=61.0).contains(&snap.game_fps), "{}", snap.game_fps);
        assert_eq!(snap.game_frame_times_ms.len(), 119);
        assert!((snap.game_frame_times_ms[0] - 16.67).abs() < 0.1);
    }
}
