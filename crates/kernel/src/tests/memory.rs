use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use lichii_common::ConsoleGeneration;
use lichii_loader::npdm::{AddressSpaceType, PoolPartition};
use lichii_loader::program::{Module, Permission, Segment};
use lichii_memory::MemoryPermission;
use parking_lot::Mutex;

use super::scheduling::heap_base;
use super::{Script, TestKernel};
use crate::memory::{MemoryInfo, MemoryState, Pool, attribute};
use crate::memory_objects::SharedMemory;
use crate::object::KObject;
use crate::resource_limit::LimitableResource;
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
    let process = test.start(test.image(), main);
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
    let process = test.kernel.create_process(&test.image()).unwrap();
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
    let process = test.kernel.create_process(&test.image()).unwrap();
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
    let mut image = test.image();
    image.kernel_capabilities = vec![0b0111 | (59 << 4) | (28 << 10) | (5 << 24)];
    let process = test.start(image, main);
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

#[test]
fn closed_events_and_transfer_memories_return_their_quota() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let main = test.program(
        Script::new()
            .svc(0x45, |_, _| {})
            .then(|ctx, _| {
                assert_eq!(ctx.x[0], 0);
                ctx.x[19] = ctx.x[1];
                ctx.x[20] = ctx.x[2];
            })
            .svc(0x16, |ctx, _| ctx.x[0] = ctx.x[19])
            .svc(0x16, |ctx, _| ctx.x[0] = ctx.x[20])
            .svc(0x15, move |ctx, _| {
                ctx.x[1] = heap;
                ctx.x[2] = 0x1000;
                ctx.x[3] = 0;
            })
            .then(|ctx, _| {
                assert_eq!(ctx.x[0], 0);
                ctx.x[19] = ctx.x[1];
            })
            .svc(0x16, |ctx, _| ctx.x[0] = ctx.x[19])
            .exit(),
    );
    let process = test.start(test.image(), main);
    let limit = process.resource_limit().clone();
    test.run_until_exit(Duration::from_secs(10));
    for which in [
        LimitableResource::Events,
        LimitableResource::TransferMemories,
    ] {
        assert_eq!(limit.current(which), 0, "{which:?}");
        assert_eq!(limit.peak(which), 1, "{which:?}");
    }
}

#[test]
fn shared_memory_outlives_its_handle_while_mapped() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let heap = heap_base(&test);
    let shared = Arc::new(
        SharedMemory::new(
            test.kernel.memory_manager().clone(),
            None,
            Pool::Application,
            0x1000,
            MemoryPermission::READ_WRITE,
            MemoryPermission::READ_WRITE,
        )
        .unwrap(),
    );
    let weak = Arc::downgrade(&shared);
    let alive = weak.clone();
    let handle = Arc::new(AtomicU64::new(0));
    let (h, h2) = (handle.clone(), handle.clone());
    let addr = heap + 0x100_0000;
    let main = test.program(
        Script::new()
            .svc(0x13, move |ctx, _| {
                ctx.x[0] = h.load(Ordering::SeqCst);
                ctx.x[1] = addr;
                ctx.x[2] = 0x1000;
                ctx.x[3] = 3;
            })
            .then(|ctx, _| assert_eq!(ctx.x[0], 0, "MapSharedMemory failed"))
            .svc(0x16, move |ctx, _| ctx.x[0] = h2.load(Ordering::SeqCst))
            .then(|ctx, _| assert_eq!(ctx.x[0], 0, "CloseHandle failed"))
            .then(move |_, env| {
                assert!(alive.upgrade().is_some(), "freed while mapped");
                env.memory.write(addr, 0xabu32).unwrap();
                assert_eq!(env.memory.read::<u32>(addr).unwrap(), 0xab);
            })
            .exit(),
    );
    let process = test.start(test.image(), main);
    let raw = process
        .handles()
        .write()
        .add(KObject::SharedMemory(shared))
        .unwrap();
    handle.store(raw as u64, Ordering::SeqCst);
    test.run_until_exit(Duration::from_secs(10));
    assert!(
        weak.upgrade().is_none(),
        "kept alive after the process exited"
    );
}

fn module(name: &str, offset: u64, fill: u8, pages: [u64; 4]) -> Module {
    let [text, ro, data, bss] = pages.map(|p| p * 0x1000);
    Module {
        name: name.into(),
        offset,
        image: vec![fill; (text + ro + data) as usize],
        size: text + ro + data + bss,
        segments: [
            Segment {
                offset: 0,
                size: text,
                permission: Permission::ReadExecute,
            },
            Segment {
                offset: text,
                size: ro,
                permission: Permission::Read,
            },
            Segment {
                offset: text + ro,
                size: data + bss,
                permission: Permission::ReadWrite,
            },
        ],
        build_id: [0; 32],
        mod0: None,
    }
}

#[test]
fn program_image_modules_are_mapped_with_their_segment_permissions() {
    let test = TestKernel::new(ConsoleGeneration::Switch2);
    let mut image = test.image();
    image.modules = vec![
        module("rtld", 0, 0x11, [1, 1, 1, 1]),
        module("main", 0x4000, 0x22, [2, 1, 1, 2]),
    ];
    image.code_size = 0xa000;
    image.entry_offset = 0;
    image.kernel_capabilities = vec![0b0111 | (59 << 4) | (28 << 10) | (5 << 24)];
    let process = test.kernel.create_process(&image).unwrap();
    let base = process.code_region().start;
    assert_eq!(base, super::CODE_BASE);
    assert_eq!(process.code_region().size(), 0xa000);
    assert_eq!(process.entry(), base);
    assert_eq!(process.capabilities().priority_mask, ((1 << 32) - 1) << 28);
    let table = process.page_table();
    let expect = [
        (0x0000, MemoryState::CODE, 5, 0x11),
        (0x1000, MemoryState::CODE, 1, 0x11),
        (0x2000, MemoryState::CODE_DATA, 3, 0x11),
        (0x3000, MemoryState::CODE_DATA, 3, 0),
        (0x4000, MemoryState::CODE, 5, 0x22),
        (0x6000, MemoryState::CODE, 1, 0x22),
        (0x7000, MemoryState::CODE_DATA, 3, 0x22),
        (0x8000, MemoryState::CODE_DATA, 3, 0),
        (0x9fff, MemoryState::CODE_DATA, 3, 0),
    ];
    for (offset, state, permission, byte) in expect {
        let info = table.query(base + offset);
        assert_eq!(info.state, state.svc_value(), "state at {offset:#x}");
        assert_eq!(info.permission, permission, "permission at {offset:#x}");
        assert_eq!(
            process.memory().read::<u8>(base + offset).unwrap(),
            byte,
            "byte at {offset:#x}"
        );
    }
}

#[test]
fn program_image_address_space_and_validation() {
    let test = TestKernel::new(ConsoleGeneration::Switch1);
    let mut image = test.image();
    image.address_space = AddressSpaceType::Is32BitNoMap;
    image.is_64bit = false;
    let process = test.kernel.create_process(&image).unwrap();
    assert_eq!(process.code_region().start, 0x20_0000);
    assert_eq!(process.params().address_space_bits, 32);
    assert!(process.params().without_alias);
    assert_eq!(process.page_table().regions().alias.size(), 0);

    let mut image = test.image();
    image.pool_partition = PoolPartition::Unknown(7);
    assert_eq!(
        test.kernel.create_process(&image).err(),
        Some(result::INVALID_ENUM_VALUE)
    );
    let mut image = test.image();
    image.entry_offset = image.code_size;
    assert_eq!(
        test.kernel.create_process(&image).err(),
        Some(result::INVALID_ARGUMENT)
    );
    let mut image = test.image();
    image.modules[0].size = super::CODE_SIZE + 0x1000;
    assert_eq!(
        test.kernel.create_process(&image).err(),
        Some(result::INVALID_MEMORY_REGION)
    );
}
