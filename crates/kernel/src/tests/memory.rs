use std::sync::Arc;
use std::time::Duration;

use lichii_common::ConsoleGeneration;
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use super::scheduling::heap_base;
use super::{Script, TestKernel};
use crate::memory::{MemoryInfo, MemoryState, attribute};
use crate::result;

fn info(env: &super::Env, addr: u64) -> MemoryInfo {
    let mut bytes = [0u8; 0x28];
    env.memory.read_bytes(addr, &mut bytes).unwrap();
    bytemuck::pod_read_unaligned(&bytes)
}

#[test]
fn heap_alias_and_query_memory() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let heap = heap_base(&test);
    let out = heap + 0x5000;
    let log = Arc::new(Mutex::new(Vec::<(String, u64)>::new()));
    let l = |name: &'static str| {
        let log = log.clone();
        move |ctx: &mut lichii_cpu::ThreadContext, _: &super::Env| {
            log.lock().push((name.into(), ctx.x[0]))
        }
    };
    let (l2, l3) = (log.clone(), log.clone());
    let main = test.program(
        Script::new()
            .svc(0x01, |ctx, _| ctx.x[1] = 4 << 20)
            .then(move |ctx, _| l2.lock().push(("heap".into(), ctx.x[1])))
            .svc(0x06, move |ctx, _| {
                ctx.x[0] = out;
                ctx.x[2] = heap + 0x30_0000;
            })
            .then(move |_, env| {
                let i = info(env, out);
                l3.lock()
                    .push(("query".into(), (i.base_address << 8) | i.state as u64));
                l3.lock().push(("query size".into(), i.size));
            })
            .svc(0x04, move |ctx, _| {
                ctx.x[0] = 0;
                ctx.x[1] = heap;
                ctx.x[2] = 0x1000;
            })
            .then(l("map memory outside stack"))
            .svc(0x02, move |ctx, _| {
                ctx.x[0] = heap;
                ctx.x[1] = 0x1000;
                ctx.x[2] = 1;
            })
            .then(l("protect read-only"))
            .svc(0x03, move |ctx, _| {
                ctx.x[0] = heap;
                ctx.x[1] = 0x1000;
                ctx.x[2] = attribute::UNCACHED as u64;
                ctx.x[3] = attribute::UNCACHED as u64;
            })
            .then(l("uncached"))
            .svc(0x80, move |ctx, _| {
                ctx.x[0] = heap + 0x1000;
                ctx.x[1] = 0x1000;
                ctx.x[2] = attribute::GPU_ACCESSIBLE as u64;
                ctx.x[3] = attribute::GPU_ACCESSIBLE as u64;
            })
            .then(l("gpu accessible"))
            .svc(0x80, move |ctx, _| {
                ctx.x[0] = heap;
                ctx.x[1] = 0x1000;
                ctx.x[2] = attribute::UNCACHED as u64;
                ctx.x[3] = 0;
            })
            .then(l("gpu attr rejects other bits"))
            .svc(0x01, |ctx, _| ctx.x[1] = 2 << 20)
            .then(l("shrink heap"))
            .exit(),
    );
    let process = test.start(test.params(), main);
    test.run_until_exit(Duration::from_secs(10));
    let log = log.lock().clone();
    let get = |name: &str| {
        log.iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| *v)
            .unwrap()
    };
    assert_eq!(get("heap"), heap);
    assert_eq!(get("query"), (heap << 8) | 5);
    assert_eq!(get("query size"), 4 << 20);
    assert_eq!(
        get("map memory outside stack"),
        result::INVALID_MEMORY_REGION.0 as u64
    );
    assert_eq!(get("protect read-only"), 0);
    assert_eq!(get("uncached"), 0);
    assert_eq!(get("gpu accessible"), 0);
    assert_eq!(
        get("gpu attr rejects other bits"),
        result::INVALID_COMBINATION.0 as u64
    );
    assert_eq!(get("shrink heap"), 0);
    let _ = process;
}

#[test]
fn map_memory_and_physical_memory() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let process = test
        .kernel
        .create_process(
            test.params(),
            crate::memory::Region::new(super::CODE_BASE, 0x100_0000),
        )
        .unwrap();
    let table = process.page_table();
    let regions = *table.regions();
    let heap = table.set_heap_size(2 << 20).unwrap();
    process.memory().write(heap, 0x55u32).unwrap();

    let stack = regions.stack.start;
    table.map_memory(stack, heap, 0x2000).unwrap();
    assert_eq!(process.memory().read::<u32>(stack).unwrap(), 0x55);
    assert_eq!(table.query(heap).permission, 0);
    assert_eq!(table.query(heap).attribute, attribute::LOCKED);
    assert_eq!(table.query(stack).state, MemoryState::STACK.svc_value());
    assert_eq!(table.set_heap_size(0), Err(result::INVALID_CURRENT_MEMORY));
    table.unmap_memory(stack, heap, 0x2000).unwrap();
    assert_eq!(table.query(heap).permission, 3);

    let alias = regions.alias.start + regions.alias.size() / 2;
    let free_before = test
        .kernel
        .memory_manager()
        .free_size(crate::Pool::Application);
    table.map_physical_memory(alias + 0x1000, 0x1000).unwrap();
    table.map_physical_memory(alias, 0x4000).unwrap();
    assert_eq!(table.query(alias).size, 0x4000);
    assert_eq!(
        free_before
            - test
                .kernel
                .memory_manager()
                .free_size(crate::Pool::Application),
        0x4000
    );
    assert_eq!(process.used_memory(), (2 << 20) + 0x4000 + 0x100_0000);
    table.unmap_physical_memory(alias, 0x4000).unwrap();
    assert_eq!(
        test.kernel
            .memory_manager()
            .free_size(crate::Pool::Application),
        free_before
    );
    assert_eq!(
        table.map_physical_memory(heap, 0x1000),
        Err(result::INVALID_MEMORY_REGION)
    );
}

#[test]
fn transfer_memory_locks_until_closed() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let process = test
        .kernel
        .create_process(
            test.params(),
            crate::memory::Region::new(super::CODE_BASE, 0x100_0000),
        )
        .unwrap();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    process.memory().write(heap + 8, 9u64).unwrap();
    let transfer = crate::memory_objects::TransferMemory::new(
        process.clone(),
        heap,
        0x2000,
        MemoryPermission::NONE,
    )
    .unwrap();
    assert_eq!(
        process.page_table().query(heap).attribute,
        attribute::LOCKED
    );
    let mut value = [0u8; 8];
    transfer.read(8, &mut value).unwrap();
    assert_eq!(u64::from_le_bytes(value), 9);
    assert_eq!(
        crate::memory_objects::TransferMemory::new(
            process.clone(),
            heap,
            0x1000,
            MemoryPermission::NONE
        )
        .err(),
        Some(result::INVALID_CURRENT_MEMORY)
    );
    drop(transfer);
    assert_eq!(process.page_table().query(heap).attribute, 0);
    assert_eq!(process.page_table().query(heap).permission, 3);
}

#[test]
fn switch2_get_info_ids() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let values = Arc::new(Mutex::new(Vec::new()));
    let mut script = Script::new();
    for id in [0u64, 2, 3, 6, 21, 28, 29, 30, 31, 32, 33] {
        let v = values.clone();
        script = script
            .svc(0x29, move |ctx, _| {
                ctx.x[1] = id;
                ctx.x[2] = crate::handle_table::CURRENT_PROCESS as u64;
                ctx.x[3] = 0;
            })
            .then(move |ctx, _| v.lock().push((id, ctx.x[0], ctx.x[1])));
    }
    let main = test.program(script.exit());
    let mut params = test.params();
    params.capabilities = vec![0b0111 | (59 << 4) | (28 << 10) | (5 << 24)];
    let process = test.start(params, main);
    let regions = *process.page_table().regions();
    test.run_until_exit(Duration::from_secs(10));
    let values = values.lock().clone();
    let get = |id: u64| {
        let (_, result, value) = values.iter().find(|(i, _, _)| *i == id).copied().unwrap();
        assert_eq!(result, 0, "GetInfo {id} failed");
        value
    };
    assert_eq!(get(0), 0x3f);
    assert_eq!(get(2), regions.alias.start);
    assert_eq!(get(3), regions.alias.size());
    assert_eq!(get(29), regions.alias.start + regions.alias.size() / 2);
    assert_eq!(get(30), regions.alias.size() / 2);
    assert_eq!(get(31), 0);
    assert_eq!(get(32), 0);
    assert_eq!(get(33), 0x1000);
    assert_eq!(get(28), 0);
    assert!(get(6) > 0);
}
