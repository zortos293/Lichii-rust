use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use lichii_common::ConsoleGeneration;
use parking_lot::Mutex;

use super::{Script, TestKernel, word};
use crate::process::Process;
use crate::util::Rng;

pub(super) fn heap_base(test: &TestKernel) -> u64 {
    let params = test.params();
    let layout = Process::layout(
        &params,
        crate::memory::Region::new(super::CODE_BASE, 0x100_0000),
    );
    crate::memory::Regions::compute(layout, &mut Rng::new(0))
        .unwrap()
        .heap
        .start
}

/// Main thread script: creates `count` threads running `entry` with arg = index, priority and core
/// from `spec`, starts them and waits for each to exit. The caller appends the rest (and the
/// exit).
pub(super) fn spawn_and_join(
    heap: u64,
    count: u64,
    entry: u64,
    spec: impl Fn(u64) -> (u32, u32) + Send + Sync + 'static,
) -> Script {
    let spec = Arc::new(spec);
    let handles = heap + 0x1000;
    Script::new()
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x08, move |ctx, _| {
            let i = ctx.x[19];
            let (priority, core) = spec(i);
            ctx.x[1] = entry;
            ctx.x[2] = i;
            ctx.x[3] = heap + 0x10_0000 + 0x1_0000 * i;
            ctx.x[4] = priority as u64;
            ctx.x[5] = core as u64;
        })
        .then(move |ctx, env| {
            assert_eq!(ctx.x[0], 0, "CreateThread failed");
            env.memory
                .write(handles + 4 * ctx.x[19], ctx.x[1] as u32)
                .unwrap();
        })
        .svc(0x09, move |ctx, env| {
            ctx.x[0] = env.memory.read::<u32>(handles + 4 * ctx.x[19]).unwrap() as u64;
        })
        .then(|ctx, _| {
            assert_eq!(ctx.x[0], 0, "StartThread failed");
            ctx.x[19] += 1;
        })
        .jump_if(1, move |ctx, _| ctx.x[19] < count)
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x18, move |ctx, _| {
            ctx.x[1] = handles + 4 * ctx.x[19];
            ctx.x[2] = 1;
            ctx.x[3] = u64::MAX;
        })
        .then(|ctx, _| {
            assert_eq!(ctx.x[0], 0, "WaitSynchronization on a thread failed");
            ctx.x[19] += 1;
        })
        .jump_if(7, move |ctx, _| ctx.x[19] < count)
}

#[test]
fn switch2_runs_six_cores_natively() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let record = seen.clone();
    let worker = test.program(
        Script::new()
            .then(|ctx, _| ctx.x[20] = ctx.x[0])
            .svc(0x10, |_, _| {})
            .then(move |ctx, env| record.lock().push((ctx.x[20], ctx.x[0], env.core as u64)))
            .exit(),
    );
    let main = test.program(spawn_and_join(heap, 6, worker, |i| (40, i as u32)).exit());
    test.start(test.params(), main);
    test.run_until_exit(Duration::from_secs(10));
    let mut seen = seen.lock().clone();
    seen.sort();
    assert_eq!(seen, (0..6).map(|i| (i, i, i)).collect::<Vec<_>>());
}

#[test]
fn switch1_rejects_cores_four_and_five() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let result = Arc::new(AtomicU64::new(0));
    let r = result.clone();
    let main = test.program(
        Script::new()
            .svc(0x08, |ctx, _| {
                ctx.x[1] = super::CODE_BASE;
                ctx.x[4] = 44;
                ctx.x[5] = 4;
            })
            .then(move |ctx, _| r.store(ctx.x[0], Ordering::SeqCst))
            .exit(),
    );
    test.start(test.params(), main);
    test.run_until_exit(Duration::from_secs(10));
    assert_eq!(
        result.load(Ordering::SeqCst),
        crate::result::INVALID_CORE_ID.0 as u64
    );
}

#[test]
fn higher_priority_thread_preempts_a_spinner() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let flag = heap + 0x800;
    let spinner = test.program(
        Script::new()
            .spin_until(move |_, env| word(env, flag).load(Ordering::SeqCst) == 1)
            .exit(),
    );
    let setter = test.program(
        Script::new()
            .then(move |_, env| word(env, flag).store(1, Ordering::SeqCst))
            .exit(),
    );
    let main = test.program(spawn_and_join_entries(
        heap,
        vec![spinner, setter],
        vec![(50, 1), (30, 1)],
    ));
    test.start(test.params(), main);
    test.run_until_exit(Duration::from_secs(10));
}

/// Like [`spawn_and_join`] with a different entry per thread; waits a little between starts so
/// the first thread is running when the second becomes ready.
fn spawn_and_join_entries(heap: u64, entries: Vec<u64>, spec: Vec<(u32, u32)>) -> Script {
    let count = entries.len() as u64;
    let handles = heap + 0x1000;
    let entries = Arc::new(entries);
    let spec = Arc::new(spec);
    Script::new()
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x08, move |ctx, _| {
            let i = ctx.x[19] as usize;
            ctx.x[1] = entries[i];
            ctx.x[2] = i as u64;
            ctx.x[3] = heap + 0x10_0000 + 0x1_0000 * i as u64;
            ctx.x[4] = spec[i].0 as u64;
            ctx.x[5] = spec[i].1 as u64;
        })
        .then(move |ctx, env| {
            assert_eq!(ctx.x[0], 0);
            env.memory
                .write(handles + 4 * ctx.x[19], ctx.x[1] as u32)
                .unwrap();
        })
        .svc(0x09, move |ctx, env| {
            ctx.x[0] = env.memory.read::<u32>(handles + 4 * ctx.x[19]).unwrap() as u64;
        })
        .svc(0x0b, |ctx, _| ctx.x[0] = 2_000_000)
        .then(|ctx, _| ctx.x[19] += 1)
        .jump_if(1, move |ctx, _| ctx.x[19] < count)
        .then(|ctx, _| ctx.x[19] = 0)
        .svc(0x18, move |ctx, _| {
            ctx.x[1] = handles + 4 * ctx.x[19];
            ctx.x[2] = 1;
            ctx.x[3] = u64::MAX;
        })
        .then(|ctx, _| ctx.x[19] += 1)
        .jump_if(8, move |ctx, _| ctx.x[19] < count)
        .exit()
}

#[test]
fn preemption_priority_threads_rotate() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let counters = heap + 0x900;
    let worker = test.program(
        Script::new()
            .then(|ctx, _| ctx.x[20] = ctx.x[0])
            .spin_until(move |ctx, env| {
                let me = ctx.x[20];
                word(env, counters + 4 * me).fetch_add(1, Ordering::SeqCst);
                word(env, counters + 4 * (1 - me)).load(Ordering::SeqCst) > 0
            })
            .exit(),
    );
    let main = test.program(spawn_and_join(heap, 2, worker, |_| (59, 1)).exit());
    test.start(test.params(), main);
    test.run_until_exit(Duration::from_secs(10));
}

#[test]
fn sleep_and_yield_return() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let elapsed = Arc::new(AtomicU64::new(0));
    let e = elapsed.clone();
    let start = Arc::new(Mutex::new(None));
    let s1 = start.clone();
    let main = test.program(
        Script::new()
            .svc(0x0b, |ctx, _| ctx.x[0] = 0)
            .svc(0x0b, |ctx, _| ctx.x[0] = u64::MAX)
            .svc(0x0b, move |ctx, _| {
                *s1.lock() = Some(std::time::Instant::now());
                ctx.x[0] = 5_000_000;
            })
            .then(move |_, _| {
                let t = start.lock().expect("start time");
                e.store(t.elapsed().as_micros() as u64, Ordering::SeqCst);
            })
            .exit(),
    );
    test.start(test.params(), main);
    test.run_until_exit(Duration::from_secs(10));
    assert!(elapsed.load(Ordering::SeqCst) >= 5_000);
}

#[test]
fn shutdown_stops_a_spinning_core() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let main = test.program(Script::new().step(|_, _| super::Action::Stay));
    test.start(test.params(), main);
    test.kernel.run().unwrap();
    std::thread::sleep(Duration::from_millis(50));
    let (tx, rx) = crossbeam_channel::bounded(1);
    let kernel = test.kernel.clone();
    std::thread::spawn(move || {
        kernel.shutdown();
        let _ = tx.send(());
    });
    rx.recv_timeout(Duration::from_secs(5))
        .expect("shutdown hung on a spinning core");
}
