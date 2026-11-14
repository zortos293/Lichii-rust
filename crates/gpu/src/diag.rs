//! Diagnostic draw and query counters for the MTF character investigation (not for merging).

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[derive(Clone, Copy)]
        pub enum K { $($name),* }
        const NAMES: &[&str] = &[$(stringify!($name)),*];
    };
}

counters!(
    eng_draw,
    eng_draw_inline,
    eng_indirect,
    eng_texture,
    eng_clear,
    eng_skip_draw,
    eng_skip_indirect,
    eng_skip_texture,
    eng_skip_clear,
    re_false,
    re_true,
    re_cond_true,
    re_cond_false,
    re_other,
    override_never,
    override_always,
    q_payload,
    q_samples,
    q_prims,
    q_xfb,
    q_counter,
    q_samples_short,
    clear_samples,
    clear_other,
    kc_launch,
    vk_dispatch,
    vk_draw,
    vk_draw_indexed,
    vk_inline,
    vk_indirect,
    vk_indirect_indexed,
    vk_indirect_count,
    vk_mesh,
    vk_mesh_indirect,
    vk_skip_stages,
    vk_skip_targets,
    vk_skip_pipeline,
    vk_skip_resources,
    vk_skip_indirect,
    vk_skip_zero_index,
    vk_skip_churn,
    vk_recorded,
    vk_depth_only,
    vk_color,
);

static COUNTS: [AtomicU64; NAMES.len()] = [const { AtomicU64::new(0) }; NAMES.len()];

pub fn add(k: K) {
    COUNTS[k as usize].fetch_add(1, Relaxed);
}

#[derive(Default)]
struct Sources {
    depth_only: HashMap<u64, u64>,
    color: HashSet<u64>,
    window_start: Option<Instant>,
    presents: u64,
}

static SOURCES: Mutex<Option<Sources>> = Mutex::new(None);

/// Records the vertex sources (stream addresses, or the mesh program address) of a draw.
pub fn sources(depth_only: bool, keys: &[u64]) {
    let mut guard = SOURCES.lock().unwrap();
    let s = guard.get_or_insert_with(Sources::default);
    for &k in keys {
        if depth_only {
            *s.depth_only.entry(k).or_default() += 1;
        } else {
            s.color.insert(k);
        }
    }
}

/// Called once per present: logs and resets every 5 seconds.
pub fn present() {
    let mut guard = SOURCES.lock().unwrap();
    let s = guard.get_or_insert_with(Sources::default);
    s.presents += 1;
    let start = *s.window_start.get_or_insert_with(Instant::now);
    if start.elapsed().as_secs() < 5 {
        return;
    }
    let counts: Vec<String> = NAMES
        .iter()
        .zip(&COUNTS)
        .filter_map(|(name, c)| {
            let v = c.swap(0, Relaxed);
            (v != 0).then(|| format!("{name}={v}"))
        })
        .collect();
    let shared = s.depth_only.keys().filter(|k| s.color.contains(k)).count();
    let mut shadow_only: Vec<(u64, u64)> = s
        .depth_only
        .iter()
        .filter(|(k, _)| !s.color.contains(k))
        .map(|(&k, &n)| (k, n))
        .collect();
    shadow_only.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let top: Vec<String> = shadow_only
        .iter()
        .take(12)
        .map(|(k, n)| format!("{k:#x}x{n}"))
        .collect();
    tracing::info!(
        target: "diag_draws",
        "window presents={} {} | sources depth_only={} color={} shared={} depth_only_not_in_color={} top=[{}]",
        s.presents,
        counts.join(" "),
        s.depth_only.len(),
        s.color.len(),
        shared,
        shadow_only.len(),
        top.join(" ")
    );
    *s = Sources {
        window_start: Some(Instant::now()),
        ..Sources::default()
    };
}
