use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

fn tsc() -> u64 {
    // SAFETY: probe only.
    unsafe {
        std::arch::x86_64::_mm_lfence();
        let t = std::arch::x86_64::_rdtsc();
        std::arch::x86_64::_mm_lfence();
        t
    }
}

#[test]
fn probe_host_clock() {
    let source = std::fs::read_to_string("/sys/devices/system/clocksource/clocksource0/current_clocksource").unwrap_or_default();
    let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let flags: Vec<&str> = cpuinfo.lines().find(|l| l.starts_with("flags")).unwrap_or("").split_whitespace().filter(|f| f.contains("tsc") || *f == "hypervisor").collect();
    let model = cpuinfo.lines().find(|l| l.starts_with("model name")).unwrap_or("").to_string();
    let last = Arc::new(AtomicU64::new(0));
    let stop = Arc::new(AtomicBool::new(false));
    let worst = Arc::new(AtomicU64::new(0));
    let backs = Arc::new(AtomicU64::new(0));
    let threads: Vec<_> = (0..4).map(|_| {
        let (last, stop, worst, backs) = (last.clone(), stop.clone(), worst.clone(), backs.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let seen = last.load(Ordering::SeqCst);
                let now = tsc();
                if now < seen {
                    backs.fetch_add(1, Ordering::Relaxed);
                    worst.fetch_max(seen - now, Ordering::Relaxed);
                } else {
                    last.fetch_max(now, Ordering::SeqCst);
                }
            }
        })
    }).collect();
    let mono_backs = Arc::new(AtomicU64::new(0));
    let mono_last = Arc::new(AtomicU64::new(0));
    let start = std::time::Instant::now();
    let mono: Vec<_> = (0..2).map(|_| {
        let (last, stop, backs) = (mono_last.clone(), stop.clone(), mono_backs.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let seen = last.load(Ordering::SeqCst);
                let now = start.elapsed().as_nanos() as u64;
                if now < seen { backs.fetch_add(1, Ordering::Relaxed); } else { last.fetch_max(now, Ordering::SeqCst); }
            }
        })
    }).collect();
    std::thread::sleep(std::time::Duration::from_secs(3));
    stop.store(true, Ordering::Relaxed);
    for t in threads.into_iter().chain(mono) { t.join().unwrap(); }
    panic!("PROBE source={} flags={:?} model={} cpus={} tsc_backs={} tsc_worst_cycles={} instant_backs={}", source.trim(), flags, model, std::thread::available_parallelism().unwrap(), backs.load(Ordering::Relaxed), worst.load(Ordering::Relaxed), mono_backs.load(Ordering::Relaxed));
}
