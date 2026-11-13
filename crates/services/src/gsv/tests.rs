use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::Event;
use lichii_kernel::ipc::Handle;
use lichii_kernel::memory::Pool;
use lichii_kernel::memory_objects::TransferMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Request};
use lichii_memory::MemoryPermission;

use crate::ServiceContext;
use crate::testing::{SWITCH2_TITLE, context};

const CURRENT_PROCESS: Handle = 0xffff_8001;

fn gsv() -> (Arc<ServiceContext>, IpcClient, Handle) {
    let (context, client) = context(SWITCH2_TITLE, Settings::default());
    crate::am::register(&context).unwrap();
    let service = client.connect("appletOE").unwrap();
    let open = Request::new(0)
        .send_pid()
        .arg(0u64)
        .copy_handle(CURRENT_PROCESS);
    let proxy = client.send(service, &open).unwrap().move_handles[0];
    let self_controller = client.send(proxy, &Request::new(1)).unwrap().move_handles[0];
    let reply = client.send(self_controller, &Request::new(2000)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    (context, client, reply.copy_handles[0])
}

fn session(client: &IpcClient, gsv: Handle, command: u32) -> Handle {
    let reply = client.send(gsv, &Request::tipc(command)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS, "gsv {command}");
    reply.copy_handles[0]
}

fn event(client: &IpcClient, handle: Handle) -> Arc<Event> {
    match client.object(handle) {
        Some(KObject::ReadableEvent(event) | KObject::WritableEvent(event)) => event,
        _ => panic!("handle {handle:#x} is not an event"),
    }
}

#[test]
fn router_grants_features_and_hands_out_sessions() {
    let (_context, client, gsv) = gsv();
    let mut reply = client
        .send(gsv, &Request::tipc(115).arg(0x1234_5678_9abcu64))
        .unwrap();
    assert_eq!(reply.pop::<u64>(), 0x1234_5678_9abc);
    assert_eq!(
        client.send(gsv, &Request::tipc(11)).unwrap().result,
        ResultCode::SUCCESS
    );
    assert_eq!(
        client.send(gsv, &Request::tipc(11)).unwrap().result,
        super::UNKNOWN
    );

    let time = session(&client, gsv, 1121);
    let mut reply = client
        .send(time, &Request::tipc(0).arg(0xfeed_u64))
        .unwrap();
    assert_eq!(reply.pop::<u64>(), 0xfeed);
    assert_eq!(
        client.send(time, &Request::tipc(11)).unwrap().result,
        ResultCode::new(11, 1)
    );
    for command in [602, 902, 906, 1002, 1202] {
        session(&client, gsv, command);
    }
    let mut reply = client.send(gsv, &Request::tipc(122)).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    assert_eq!(reply.pop::<u32>(), 0);
}

#[test]
fn syncpt_slots_signal_when_the_syncpoint_is_reached() {
    let (context, client, gsv) = gsv();
    let syncpt = session(&client, gsv, 202);
    let guest = context.gpu.host1x().syncpoints.guest();
    let reply = client.send(syncpt, &Request::tipc(22).arg(3u32)).unwrap();
    let slot = event(&client, reply.copy_handles[0]);
    slot.signal();

    let request = Request::tipc(201).arg(3u32).arg(!7u32).arg(2u64).arg(0u32);
    assert_eq!(
        client.send(syncpt, &request).unwrap().result,
        ResultCode::SUCCESS
    );
    assert!(!slot.is_signaled(), "a new request starts unsignalled");
    guest.increment(7).unwrap();
    assert!(!slot.is_signaled());
    guest.increment(7).unwrap();
    assert!(slot.is_signaled());

    let mut reply = client.send(syncpt, &Request::tipc(103).arg(!7u32)).unwrap();
    assert_eq!(reply.pop::<u64>(), 2);

    let request = Request::tipc(201).arg(3u32).arg(!7u32).arg(9u64).arg(0u32);
    client.send(syncpt, &request).unwrap();
    assert!(!slot.is_signaled());
    client.send(syncpt, &Request::tipc(202).arg(3u32)).unwrap();
    assert!(slot.is_signaled(), "a cancel signals the slot");
    slot.clear();
    for _ in 0..7 {
        guest.increment(7).unwrap();
    }
    assert!(!slot.is_signaled(), "a cancelled request never signals");
}

struct SyncptPage {
    client: IpcClient,
    gsv: Handle,
    rw: u64,
    ro: u64,
    request: Arc<Event>,
    notify: Arc<Event>,
}

impl SyncptPage {
    fn arm(&self, id: u64) {
        let address = self.rw + 0x100 + id / 64 * 8;
        let memory = self.client.process().memory();
        let bits: u64 = memory.read(address).unwrap();
        memory.write(address, bits ^ (1 << (id % 64))).unwrap();
        self.request.signal();
    }

    fn acknowledged(&self, id: u64) -> bool {
        let memory = self.client.process().memory();
        let armed: u64 = memory.read(self.rw + 0x100 + id / 64 * 8).unwrap();
        let acked: u64 = memory.read(self.ro + 0x1100 + id / 64 * 8).unwrap();
        (armed ^ acked) & (1 << (id % 64)) == 0
    }

    fn value(&self, id: u64) -> u64 {
        self.client
            .process()
            .memory()
            .read(self.ro + 0x40 + id * 8)
            .unwrap()
    }
}

fn syncpt_page() -> (Arc<ServiceContext>, SyncptPage) {
    let (context, client, gsv) = gsv();
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    let (rw, ro) = (heap, heap + 0x1000);
    let transfer = |address, size, permission| {
        let memory = TransferMemory::new(process.clone(), address, size, permission).unwrap();
        process
            .handles()
            .write()
            .add(KObject::TransferMemory(Arc::new(memory)))
            .unwrap()
    };
    let rw_handle = transfer(rw, 0x1000, MemoryPermission::READ_WRITE);
    let ro_handle = transfer(ro, 0x2000, MemoryPermission::READ);
    let attach = Request::tipc(204)
        .copy_handle(rw_handle)
        .copy_handle(ro_handle)
        .arg(1u32)
        .arg(96u32);
    let reply = client.send(gsv, &attach).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let request = event(&client, reply.copy_handles[0]);
    let notify = event(&client, reply.copy_handles[1]);
    let page = SyncptPage {
        client,
        gsv,
        rw,
        ro,
        request,
        notify,
    };
    (context, page)
}

#[test]
fn syncpt_page_header_describes_the_layout() {
    let (_context, page) = syncpt_page();
    let header: Vec<u32> = page
        .client
        .read(page.ro, 32)
        .chunks(4)
        .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    assert_eq!(header, [1, 0, 96, 0x40, 0, 0x1040, 0x100, 0x1100]);
}

#[test]
fn syncpt_page_answers_armed_waits_on_the_next_increment() {
    let (context, page) = syncpt_page();
    let guest = context.gpu.host1x().syncpoints.guest();
    page.arm(70);
    assert!(!page.acknowledged(70), "nothing changed yet");
    assert!(!page.notify.is_signaled());
    guest.increment(70).unwrap();
    assert!(page.acknowledged(70));
    assert!(page.notify.is_signaled());
    assert_eq!(page.value(70), 1);

    page.notify.clear();
    guest.increment(5).unwrap();
    page.arm(5);
    assert!(
        page.acknowledged(5),
        "a stale page value is answered at once"
    );
    assert!(page.notify.is_signaled());
    assert_eq!(page.value(5), 1);

    page.arm(100);
    assert!(
        page.acknowledged(100),
        "ids past the page's count are answered"
    );
}

#[test]
fn syncpt_slots_mirror_the_value_into_the_page_before_signalling() {
    let (context, page) = syncpt_page();
    let guest = context.gpu.host1x().syncpoints.guest();
    let syncpt = session(&page.client, page.gsv, 202);
    let reply = page
        .client
        .send(syncpt, &Request::tipc(22).arg(0u32))
        .unwrap();
    let slot = event(&page.client, reply.copy_handles[0]);
    let request = Request::tipc(201).arg(0u32).arg(!9u32).arg(2u64).arg(0u32);
    page.client.send(syncpt, &request).unwrap();
    guest.increment(9).unwrap();
    assert_eq!(
        page.value(9),
        0,
        "the page is not written on every increment"
    );
    guest.increment(9).unwrap();
    assert!(slot.is_signaled());
    assert_eq!(page.value(9), 2);
}

#[test]
fn a_slot_signal_does_not_swallow_a_page_wait_on_the_same_syncpoint() {
    let (context, page) = syncpt_page();
    let guest = context.gpu.host1x().syncpoints.guest();
    let syncpt = session(&page.client, page.gsv, 202);
    let request = Request::tipc(201).arg(0u32).arg(!9u32).arg(1u64).arg(0u32);
    page.client.send(syncpt, &request).unwrap();
    page.arm(9);
    guest.increment(9).unwrap();
    assert_eq!(page.value(9), 1);
    assert!(page.acknowledged(9));
    assert!(page.notify.is_signaled());
}

#[test]
fn slots_past_the_page_count_leave_the_page_alone() {
    let (context, page) = syncpt_page();
    let guest = context.gpu.host1x().syncpoints.guest();
    let syncpt = session(&page.client, page.gsv, 202);
    let request = Request::tipc(201)
        .arg(0u32)
        .arg(!600u32)
        .arg(1u64)
        .arg(0u32);
    page.client.send(syncpt, &request).unwrap();
    guest.increment(600).unwrap();
    assert_eq!(page.value(600), 0);
}

#[test]
fn a_wait_armed_before_a_slot_mirrors_is_answered_when_the_request_comes() {
    let (context, page) = syncpt_page();
    let guest = context.gpu.host1x().syncpoints.guest();
    let syncpt = session(&page.client, page.gsv, 202);
    let request = Request::tipc(201).arg(0u32).arg(!9u32).arg(1u64).arg(0u32);
    page.client.send(syncpt, &request).unwrap();
    let address = page.rw + 0x100;
    let memory = page.client.process().memory();
    let bits: u64 = memory.read(address).unwrap();
    memory.write(address, bits ^ (1 << 9)).unwrap();
    guest.increment(9).unwrap();
    assert_eq!(page.value(9), 1, "the slot mirrored the value");
    page.request.signal();
    assert!(page.acknowledged(9));
    assert!(page.notify.is_signaled());
}

fn words(request: Request, values: &[u32]) -> Request {
    values.iter().fold(request, |request, &w| request.arg(w))
}

fn split(value: u64) -> [u32; 2] {
    [value as u32, (value >> 32) as u32]
}

#[test]
fn gpu_session_slots_devices_and_partitions() {
    let (_context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let mut reply = client.send(gpu, &Request::tipc(11)).unwrap();
    assert_eq!(reply.pop::<[u32; 2]>(), [0, 31]);
    let reply = client.send(gpu, &Request::tipc(12).arg(37u32)).unwrap();
    let slot = event(&client, reply.copy_handles[0]);
    client.send(gpu, &Request::tipc(21).arg(5u32)).unwrap();
    assert!(slot.is_signaled(), "ids wrap at 32");

    let sub = session(&client, gpu, 22);
    let reply = client.send(sub, &Request::tipc(12).arg(5u32)).unwrap();
    assert!(Arc::ptr_eq(&slot, &event(&client, reply.copy_handles[0])));

    let mut first = client
        .send(gpu, &Request::tipc(41).arg(0u64).arg(0u64))
        .unwrap();
    let mut second = client
        .send(gpu, &Request::tipc(41).arg(0u64).arg(0u64))
        .unwrap();
    assert_eq!((first.pop::<u32>(), second.pop::<u32>()), (1, 2));
    assert_eq!(
        client
            .send(gpu, &Request::tipc(42).arg(1u32))
            .unwrap()
            .result,
        ResultCode::SUCCESS
    );
    assert_eq!(
        client
            .send(gpu, &Request::tipc(42).arg(1u32))
            .unwrap()
            .result,
        lichii_kernel::result::NOT_FOUND
    );

    let mut reply = client.send(gpu, &Request::tipc(201)).unwrap();
    assert_eq!(
        reply.pop::<[u64; 2]>(),
        [0x800_0000, 0xc0_0000_0000 - 0x800_0000]
    );
    let mut reply = client.send(gpu, &Request::tipc(202)).unwrap();
    assert_eq!(reply.pop::<[u64; 2]>(), [u64::MAX, 0]);
    let mut reply = client.send(gpu, &Request::tipc(203)).unwrap();
    assert_eq!(reply.pop::<[u64; 3]>()[0], 1);
}

#[test]
fn gpu_memory_is_mapped_for_every_address_space() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    process
        .memory()
        .write(heap + 0x1004, 0xdead_beefu32)
        .unwrap();
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);

    let last = heap + 0x10_0000 - 1;
    let create = words(
        Request::tipc(101),
        &[split(heap), split(last), [0, 0]].concat(),
    );
    let mut reply = client.send(gpu, &create).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let handle = reply.pop::<u32>();
    assert_eq!(
        gmmu.read::<u32>(heap + 0x1004).unwrap(),
        0xdead_beef,
        "GPU VA == CPU VA"
    );

    let va = 0x2_0000_0000u64;
    let mut map = vec![7];
    map.extend(split(va));
    map.extend(split(va + 0xfff));
    map.push(handle);
    map.extend(split(0x1000));
    map.resize(15, 0);
    map.push(9);
    client.send(gpu, &words(Request::tipc(211), &map)).unwrap();
    context.gpu.synchronize();
    assert_eq!(gmmu.read::<u32>(va + 4).unwrap(), 0xdead_beef);
    let reply = client.send(gpu, &Request::tipc(12).arg(9u32)).unwrap();
    assert!(event(&client, reply.copy_handles[0]).is_signaled());

    let mut unmap = vec![0];
    unmap.extend(split(va));
    unmap.extend(split(va + 0xfff));
    unmap.extend([0, 10]);
    client
        .send(gpu, &words(Request::tipc(213), &unmap))
        .unwrap();
    context.gpu.synchronize();
    assert!(gmmu.translate(va).is_none());
    let reply = client.send(gpu, &Request::tipc(12).arg(10u32)).unwrap();
    assert!(event(&client, reply.copy_handles[0]).is_signaled());

    let region = 0x3_0000_0000u64;
    let mut sparse = vec![0];
    sparse.extend(split(region));
    sparse.extend(split(region + 0x10_0000 - 1));
    sparse.extend([0x200, 0, 0, 0, 11]);
    client
        .send(gpu, &words(Request::tipc(214), &sparse))
        .unwrap();
    context.gpu.synchronize();
    assert_eq!(
        gmmu.read::<u32>(region + 4).unwrap(),
        0,
        "sparse reads zero"
    );

    let mut request = Vec::new();
    request.extend((region + 0x1_0000).to_le_bytes());
    request.extend(0x1000u64.to_le_bytes());
    request.extend([0, 0, 0, 0xdb, 0, 0, 0, 0]);
    request.extend(handle.to_le_bytes());
    request.extend(0u32.to_le_bytes());
    request.extend(0x1000u64.to_le_bytes());
    let input = client.alloc(&request);
    let kinds = client.alloc(&[0]);
    let mut map = vec![0; 6];
    map.push(12);
    let mut reply = client
        .send(
            gpu,
            &words(Request::tipc(216), &map)
                .buffer_a(input, request.len() as u64)
                .buffer_b(kinds, 1),
        )
        .unwrap();
    assert_eq!(reply.pop::<u64>(), 1);
    context.gpu.synchronize();
    assert_eq!(client.read(kinds, 1), [0xdb]);
    assert_eq!(gmmu.read::<u32>(region + 0x1_0004).unwrap(), 0xdead_beef);
    assert_eq!(gmmu.kind(region + 0x1_0000), Some(0xdb));
    context.gpu.synchronize();
    let reply = client.send(gpu, &Request::tipc(12).arg(12u32)).unwrap();
    assert!(event(&client, reply.copy_handles[0]).is_signaled());

    let destroy = Request::tipc(102).arg(handle).arg(13u32);
    client.send(gpu, &destroy).unwrap();
    context.gpu.synchronize();
    assert!(gmmu.translate(heap).is_none());
    assert_eq!(
        gmmu.read::<u32>(region + 0x1_0004).unwrap(),
        0,
        "the tile reads zero again"
    );
}

#[test]
fn gpu_only_regions_keep_what_the_gpu_writes() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);
    let region = 0x81_0000_0000u64;
    let mut create = vec![0];
    create.extend(split(region));
    create.extend(split(region + 0x20_0000 - 1));
    create.extend([0, 0, 0, 0, 3]);
    client
        .send(gpu, &words(Request::tipc(214), &create))
        .unwrap();
    context.gpu.synchronize();
    gmmu.write(region + 0x1_0000, 0x1234_5678u32).unwrap();
    assert_eq!(gmmu.read::<u32>(region + 0x1_0000).unwrap(), 0x1234_5678);

    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    process.memory().write(heap, 0xdead_beefu32).unwrap();
    let memory = words(
        Request::tipc(101),
        &[split(heap), split(heap + 0xffff), [0, 0]].concat(),
    );
    let handle = client.send(gpu, &memory).unwrap().pop::<u32>();
    let tile = |handle: u32| {
        let mut request = Vec::new();
        request.extend((region + 0x1_0000).to_le_bytes());
        request.extend(0x1000u64.to_le_bytes());
        request.extend([0; 8]);
        request.extend(handle.to_le_bytes());
        request.extend(0u32.to_le_bytes());
        request.extend(0u64.to_le_bytes());
        let input = client.alloc(&request);
        let mut payload = vec![0; 6];
        payload.push(4);
        client
            .send(
                gpu,
                &words(Request::tipc(216), &payload).buffer_a(input, 40),
            )
            .unwrap();
        context.gpu.synchronize();
    };
    tile(handle);
    assert_eq!(gmmu.read::<u32>(region + 0x1_0000).unwrap(), 0xdead_beef);
    tile(0);
    assert_eq!(
        gmmu.read::<u32>(region + 0x1_0000).unwrap(),
        0x1234_5678,
        "an unmapped tile shows the region's own memory again"
    );

    let mut destroy = vec![0];
    destroy.extend(split(region));
    destroy.extend(split(region + 0x20_0000 - 1));
    destroy.extend([0, 5]);
    client
        .send(gpu, &words(Request::tipc(215), &destroy))
        .unwrap();
    context.gpu.synchronize();
    assert!(gmmu.translate(region).is_none());
}

#[test]
fn gpu_only_regions_take_system_memory_only_for_the_blocks_written() {
    use super::reserve::READY_BLOCKS;
    use lichii_gpu::lazy::LAZY_BLOCK_SIZE;
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);
    let process = client.process().clone();
    let manager = process.page_table().memory_manager().clone();
    let total = process.total_memory();
    let system = manager.free_size(Pool::SystemNonSecure);
    let ready = READY_BLOCKS as u64;
    let settles_at = |blocks: u64, why: &str| {
        let want = system - blocks * LAZY_BLOCK_SIZE;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while manager.free_size(Pool::SystemNonSecure) != want {
            assert!(std::time::Instant::now() < deadline, "{why}");
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    };

    let region = 0x81_0000_0000u64;
    let size = 4u64 << 30;
    assert!(
        size > system,
        "the region is larger than the whole carve-out"
    );
    client
        .send(
            gpu,
            &range_words(214, region, region + size - 1, &[0, 0, 0, 0, 3]),
        )
        .unwrap();
    context.gpu.synchronize();
    settles_at(ready, "the region takes nothing beyond the ready blocks");
    assert_eq!(gmmu.read::<u32>(region + size - 4).unwrap(), 0);

    gmmu.write(region + LAZY_BLOCK_SIZE + 8, 0x1234_5678u32)
        .unwrap();
    assert_eq!(
        gmmu.read::<u32>(region + LAZY_BLOCK_SIZE + 8).unwrap(),
        0x1234_5678
    );
    settles_at(
        ready + 1,
        "the written block took a ready block, which is replaced",
    );
    assert_eq!(
        process.total_memory(),
        total,
        "the title's memory total stays whole"
    );

    client
        .send(gpu, &range_words(215, region, region + size - 1, &[0, 5]))
        .unwrap();
    context.gpu.synchronize();
    assert!(gmmu.translate(region).is_none());
    settles_at(ready, "the destroyed region's block goes back to the pool");
}

fn sparse_request(va: u64, size: u64, handle: u32, offset: u64, kind: u8) -> Vec<u8> {
    let mut request = Vec::new();
    request.extend(va.to_le_bytes());
    request.extend(size.to_le_bytes());
    request.extend([0, 0, 0, kind, 0, 0, 0, 0]);
    request.extend(handle.to_le_bytes());
    request.extend(0u32.to_le_bytes());
    request.extend(offset.to_le_bytes());
    request
}

fn range_words(command: u32, first: u64, last: u64, rest: &[u32]) -> Request {
    let mut values = vec![0];
    values.extend(split(first));
    values.extend(split(last));
    values.extend(rest);
    words(Request::tipc(command), &values)
}

#[test]
fn memory_requests_apply_in_the_order_they_came() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    process.memory().write(heap, 0x1111_1111u32).unwrap();
    process
        .memory()
        .write(heap + 0x1000, 0x2222_2222u32)
        .unwrap();
    let create = words(
        Request::tipc(101),
        &[split(heap), split(heap + 0xffff), [0, 0]].concat(),
    );
    let handle = client.send(gpu, &create).unwrap().pop::<u32>();

    let va = 0x2_0000_0000u64;
    let map = |offset: u64| {
        let mut values = vec![7];
        values.extend(split(va));
        values.extend(split(va + 0xfff));
        values.push(handle);
        values.extend(split(offset));
        values.resize(15, 0);
        values.push(1);
        client
            .send(gpu, &words(Request::tipc(211), &values))
            .unwrap();
    };
    map(0x1000);
    client
        .send(gpu, &range_words(213, va, va + 0xfff, &[0, 2]))
        .unwrap();
    map(0);
    context.gpu.synchronize();
    assert_eq!(
        gmmu.read::<u32>(va).unwrap(),
        0x1111_1111,
        "an unmap does not undo the map that came after it"
    );

    let region = 0x3_0000_0000u64;
    let last = region + 0x10_0000 - 1;
    let sparse = range_words(214, region, last, &[0x200, 0, 0, 0, 3]);
    client.send(gpu, &sparse).unwrap();
    client
        .send(gpu, &range_words(215, region, last, &[0, 4]))
        .unwrap();
    client.send(gpu, &sparse).unwrap();
    context.gpu.synchronize();
    assert_eq!(
        gmmu.read::<u32>(region).unwrap(),
        0,
        "a region made again after its destroy stays"
    );
}

#[test]
fn a_failed_sparse_map_leaves_the_kind_alone() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);
    let region = 0x3_0000_0000u64;
    client
        .send(
            gpu,
            &range_words(214, region, region + 0x10_0000 - 1, &[0x200, 0, 0, 0, 3]),
        )
        .unwrap();
    context.gpu.synchronize();
    let before = gmmu.kind(region + 0x1_0000);
    let request = sparse_request(region + 0x1_0000, 0x1000, 99, 0, 0xdb);
    let input = client.alloc(&request);
    let mut payload = vec![0; 6];
    payload.push(4);
    client
        .send(
            gpu,
            &words(Request::tipc(216), &payload).buffer_a(input, 40),
        )
        .unwrap();
    context.gpu.synchronize();
    assert_eq!(gmmu.kind(region + 0x1_0000), before);
}

#[test]
fn gpu_only_regions_that_cannot_be_mapped_are_not_backed() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);
    let unaligned = 0x81_0000_0800u64;
    client
        .send(
            gpu,
            &range_words(214, unaligned, unaligned + 0x20_0000 - 1, &[0, 0, 0, 0, 3]),
        )
        .unwrap();
    assert!(gmmu.read::<u32>(unaligned).is_err());

    let outside = lichii_gpu::gmmu::GPU_VA_SIZE;
    client
        .send(
            gpu,
            &range_words(214, outside, outside + 0x1_0000 - 1, &[0, 0, 0, 0, 3]),
        )
        .unwrap();
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    let create = words(
        Request::tipc(101),
        &[split(heap), split(heap + 0xffff), [0, 0]].concat(),
    );
    let handle = client.send(gpu, &create).unwrap().pop::<u32>();
    assert_eq!(handle, 1, "the failed regions took no memory");
    context.gpu.synchronize();
    assert!(gmmu.read::<u32>(unaligned).is_err());
}

#[test]
fn an_unmap_waits_for_the_gpu_work_queued_before_it() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    process.memory().write(heap, 0x1111_1111u32).unwrap();
    let create = words(
        Request::tipc(101),
        &[split(heap), split(heap + 0xffff), [0, 0]].concat(),
    );
    let handle = client.send(gpu, &create).unwrap().pop::<u32>();

    let (release, queued_work) = std::sync::mpsc::channel::<()>();
    let (seen, reads) = std::sync::mpsc::channel();
    let reader = gmmu.clone();
    context.gpu.run_in_order(move |_| {
        queued_work.recv().unwrap();
        seen.send(reader.read::<u32>(heap).ok()).unwrap();
    });
    let destroy = Request::tipc(102).arg(handle).arg(6u32);
    client.send(gpu, &destroy).unwrap();
    let reply = client.send(gpu, &Request::tipc(12).arg(6u32)).unwrap();
    let slot = event(&client, reply.copy_handles[0]);
    assert!(!slot.is_signaled());
    assert_eq!(gmmu.read::<u32>(heap).unwrap(), 0x1111_1111);

    release.send(()).unwrap();
    context.gpu.synchronize();
    assert_eq!(
        reads.recv().unwrap(),
        Some(0x1111_1111),
        "work queued before the destroy still reads the memory"
    );
    assert!(gmmu.read::<u32>(heap).is_err());
    assert!(slot.is_signaled());
}

#[test]
fn memory_made_after_an_unmap_stays_mapped() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let gmmu = context
        .gpu
        .create_address_space(lichii_gpu::gmmu::BigPageSize::Size64K);
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    process.memory().write(heap, 0x1111_1111u32).unwrap();
    let create = words(
        Request::tipc(101),
        &[split(heap), split(heap + 0xffff), [0, 0]].concat(),
    );
    client.send(gpu, &create).unwrap();

    let (release, queued_work) = std::sync::mpsc::channel::<()>();
    context
        .gpu
        .run_in_order(move |_| queued_work.recv().unwrap());
    client
        .send(gpu, &range_words(213, heap, heap + 0xffff, &[0, 2]))
        .unwrap();
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        release.send(()).unwrap();
    });
    let reply = client.send(gpu, &create).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    releaser.join().unwrap();
    context.gpu.synchronize();
    assert_eq!(
        gmmu.read::<u32>(heap).unwrap(),
        0x1111_1111,
        "the unmap sent before the memory was made does not remove it"
    );
}

const COLOR_PREFERENCE: [u32; 16] = [
    0,
    2,
    (2160 - 1) << 16 | (3840 - 1),
    0,
    7,
    240,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
];

#[test]
fn native_image_layouts_follow_the_preference() {
    use super::native::ImageLayout;
    let layout = ImageLayout::requirements(&COLOR_PREFERENCE, 1);
    assert_eq!((layout.width, layout.height), (3840, 2160));
    assert_eq!((layout.pitch, layout.block_rows_log2), (0x3c00, 7));
    assert_eq!((layout.size, layout.alignment), (0x1fe_0000, 0x1000));
    assert_eq!(ImageLayout::of(&COLOR_PREFERENCE).alignment, 0x2000);

    let mut fp16 = COLOR_PREFERENCE;
    fp16[6] = 5;
    fp16[5] = 480;
    let layout = ImageLayout::of(&fp16);
    assert_eq!((layout.pitch, layout.size), (0x7800, 0x3fc_0000));

    fp16[5] = 240;
    fp16[4] = 3;
    let layout = ImageLayout::of(&fp16);
    assert_eq!(
        (layout.pitch, layout.block_rows_log2),
        (0x7800, 7),
        "a pitch too small for 8-byte pixels falls back"
    );

    let mut tall = COLOR_PREFERENCE;
    tall[4] = 8;
    let layout = ImageLayout::requirements(&tall, 1);
    assert_eq!((layout.size, layout.alignment), (0x21c_0000, 0x1000));

    let mut hd = COLOR_PREFERENCE;
    hd[2] = (1080 - 1) << 16 | (1920 - 1);
    hd[5] = 120;
    assert_eq!(ImageLayout::requirements(&hd, 1).size, 0x87_0000);
}

fn hd_preference() -> Vec<u32> {
    let mut preference = COLOR_PREFERENCE;
    preference[2] = (1080 - 1) << 16 | (1920 - 1);
    preference[5] = 120;
    preference.to_vec()
}

#[test]
fn native_memory_and_images_are_registered() {
    let (_context, client, gsv) = gsv();
    let mut reply = client
        .send(gsv, &words(Request::tipc(111), &[1, 0x87_0000, 0]))
        .unwrap();
    let memory = reply.pop::<u32>();
    assert_eq!(memory, 1);
    let bind = words(
        Request::tipc(114),
        &[
            [memory].as_slice(),
            &split(0x10_0000_0000),
            &split(0x87_0000),
            &[0; 4],
        ]
        .concat(),
    );
    assert_eq!(client.send(gsv, &bind).unwrap().result, ResultCode::SUCCESS);

    let mut request = hd_preference();
    request.extend([2, memory]);
    let mut reply = client
        .send(gsv, &words(Request::tipc(429), &request))
        .unwrap();
    let image = reply.pop::<u32>();
    let mut reply = client
        .send(gsv, &words(Request::tipc(429), &request))
        .unwrap();
    assert_eq!(
        reply.pop::<u32>(),
        image + 2,
        "a pool keeps a handle per image"
    );

    let mut sized = hd_preference();
    sized.push(2);
    let mut reply = client
        .send(gsv, &words(Request::tipc(435), &sized))
        .unwrap();
    assert_eq!(reply.pop::<u64>(), 2 * 0x87_0000);
    let single = [hd_preference(), vec![1]].concat();
    let mut reply = client
        .send(gsv, &words(Request::tipc(1510), &single))
        .unwrap();
    assert_eq!(reply.pop::<[u64; 2]>(), [0x87_0000, 0x1000]);

    let mut reply = client
        .send(gsv, &Request::tipc(426).arg(image + 1))
        .unwrap();
    let [gpu_address, offset] = reply.pop::<[u64; 2]>();
    assert_eq!(offset, 0x87_0000, "image 1 of the pool");
    assert_eq!(gpu_address, 0x10_0000_0000 + 0x87_0000);
    let descriptor = reply.pop::<[u32; 12]>();
    assert_eq!(descriptor[0], 1);
    assert_eq!(descriptor[2], (1080 - 1) << 16 | (1920 - 1));
    assert_eq!(descriptor[5], 120);
    assert_eq!(descriptor[11], 0x87_0000);

    client.send(gsv, &Request::tipc(430).arg(image)).unwrap();
    let mut reply = client.send(gsv, &Request::tipc(428).arg(image)).unwrap();
    reply.pop::<[u64; 2]>();
    assert_eq!(reply.pop::<[u32; 12]>()[2], 0, "a destroyed image is gone");
}

fn present(image: u32, timing: u32, fence: (u32, u32)) -> Request {
    let mut values = vec![0u32; 23];
    values[3] = image;
    values[10] = timing;
    values[20] = !fence.0;
    values[22] = fence.1;
    words(Request::tipc(229), &values)
}

#[test]
fn vi2_presents_native_images_and_counts_guest_frames() {
    let (context, client, gsv) = gsv();
    let gpu = session(&client, gsv, 497);
    let vi2 = session(&client, gsv, 305);
    let mut reply = client.send(vi2, &Request::tipc(5)).unwrap();
    assert_eq!(reply.pop::<u32>(), 7);
    let mut reply = client
        .send(
            vi2,
            &Request::tipc(106).arg(u64::from_le_bytes(*b"default\0")),
        )
        .unwrap();
    assert_eq!(reply.pop::<u64>(), 4, "docked: up to 1440p");

    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(16 << 20).unwrap();
    let create = words(
        Request::tipc(101),
        &[split(heap), split(heap + 0x87_0000 - 1), [0, 0]].concat(),
    );
    let handle = client.send(gpu, &create).unwrap().pop::<u32>();
    let memory = client
        .send(gsv, &words(Request::tipc(111), &[1, 0x87_0000, 0]))
        .unwrap()
        .pop::<u32>();
    let bind = words(
        Request::tipc(423),
        &[
            [memory].as_slice(),
            &split(heap),
            &split(0x87_0000),
            &[handle],
            &split(0),
        ]
        .concat(),
    );
    client.send(gsv, &bind).unwrap();
    let mut request = hd_preference();
    request.extend([1, memory]);
    let image = client
        .send(gsv, &words(Request::tipc(429), &request))
        .unwrap()
        .pop::<u32>();

    let syncpoint = 20;
    let started = std::time::Instant::now();
    for frame in 1..=3u32 {
        context
            .gpu
            .host1x()
            .syncpoints
            .guest()
            .increment(syncpoint)
            .unwrap();
        let mut reply = client
            .send(vi2, &present(image, 1 << 16 | 60, (syncpoint, frame)))
            .unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u64>(), 0x7fff);
    }
    assert!(
        started.elapsed() >= std::time::Duration::from_millis(30),
        "presents are paced to 60 Hz"
    );
    let now = std::time::Instant::now();
    let perf = context
        .perf
        .snapshot(now, std::time::Duration::from_secs(1));
    assert_eq!(
        perf.game_frame_times_ms.len(),
        2,
        "three guest frames recorded"
    );
}

#[test]
fn vi2_native_events_fire_with_vsync_until_rebound() {
    let (_context, client, gsv) = gsv();
    let vi2 = session(&client, gsv, 305);
    let reply = client.send(vi2, &Request::tipc(31).arg(0u32)).unwrap();
    let vsync = event(&client, reply.copy_handles[0]);
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(vsync.is_signaled(), "an acquired slot fires every vblank");

    let reply = client.send(vi2, &Request::tipc(31).arg(1u32)).unwrap();
    let change = event(&client, reply.copy_handles[0]);
    let name = u64::from_le_bytes(*b"default\0");
    client
        .send(vi2, &Request::tipc(263).arg(name).arg(1u32))
        .unwrap();
    assert!(change.is_signaled(), "a change slot fires once when bound");
    change.clear();
    std::thread::sleep(std::time::Duration::from_millis(50));
    assert!(!change.is_signaled(), "and not with vsync");
    assert_eq!(
        client
            .send(vi2, &Request::tipc(31).arg(8u32))
            .unwrap()
            .result,
        super::UNKNOWN
    );
}

#[test]
fn vi2_shows_the_drawn_area_of_a_native_image() {
    use lichii_gpu::backend::CropRect;
    use lichii_texture::PixelFormat;
    use lichii_texture::block_linear::GobLayout;
    let (context, client) = context(SWITCH2_TITLE, Settings::default());
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(32 << 20).unwrap();
    let memory = context.gpu.nx2_memory().unwrap();
    let (handle, _) = memory
        .create_gpu_accessible_memory(process.memory(), heap, 0x200_0000)
        .unwrap();
    let daddr = memory.memory_info(handle).unwrap().daddr;
    let (objects, image) = super::native::NativeObjects::with_image(heap, None, COLOR_PREFERENCE);
    let vi2 = super::vi2::Vi2::new(&context, Arc::new(parking_lot::Mutex::new(objects)));

    let shown = vi2
        .framebuffer(image, (1440 - 1) << 16 | (2560 - 1))
        .unwrap();
    assert_eq!(
        shown.address, daddr,
        "found through the display's address space"
    );
    assert_eq!((shown.width, shown.height), (3840, 2160));
    assert_eq!(shown.format, PixelFormat::A8B8G8R8Unorm);
    assert_eq!(
        shown.block_height_log2,
        Some(4),
        "128-row blocks are 16 GOBs"
    );
    assert_eq!(shown.gob_layout, GobLayout::Switch2);
    assert_eq!(
        shown.crop,
        Some(CropRect {
            left: 0,
            top: 0,
            right: 2560,
            bottom: 1440
        })
    );
    assert_eq!(
        vi2.framebuffer(image, 0).unwrap().crop.unwrap().right,
        3840,
        "without a drawn area the whole image shows"
    );

    let mut handheld = Settings::default();
    handheld.system.docked = false;
    let (context, client) = crate::testing::context(SWITCH2_TITLE, handheld);
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(32 << 20).unwrap();
    let (objects, image) = super::native::NativeObjects::with_image(heap, None, COLOR_PREFERENCE);
    let vi2 = super::vi2::Vi2::new(&context, Arc::new(parking_lot::Mutex::new(objects)));
    assert!(
        vi2.framebuffer(image, 0).is_none(),
        "memory no GPU address space reaches is not shown"
    );
    let memory = context.gpu.nx2_memory().unwrap();
    memory
        .create_gpu_accessible_memory(process.memory(), heap, 0x200_0000)
        .unwrap();
    let crop = vi2.framebuffer(image, 0).unwrap().crop.unwrap();
    assert_eq!(
        (crop.right, crop.bottom),
        (1920, 1080),
        "handheld shows the panel's corner of a docked image"
    );
}

#[test]
fn vi2_shows_the_gpu_memory_a_native_image_was_bound_to() {
    let (context, client) = context(SWITCH2_TITLE, Settings::default());
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(64 << 20).unwrap();
    let memory = context.gpu.nx2_memory().unwrap();
    let (parent, _) = memory
        .create_gpu_accessible_memory(process.memory(), heap, 0x200_0000)
        .unwrap();
    let (other, _) = memory
        .create_gpu_accessible_memory(process.memory(), heap + 0x200_0000, 0x200_0000)
        .unwrap();
    memory
        .normal_map_4k(heap + 0x1000, heap + 0x1000 + 0x1fe_0000 - 1, other, 0)
        .unwrap();
    let parent_daddr = memory.memory_info(parent).unwrap().daddr;
    let (objects, image) = super::native::NativeObjects::with_image(
        heap + 0x1000,
        Some((parent, 0x1000)),
        COLOR_PREFERENCE,
    );
    let vi2 = super::vi2::Vi2::new(&context, Arc::new(parking_lot::Mutex::new(objects)));
    assert_eq!(
        vi2.framebuffer(image, 0).unwrap().address,
        parent_daddr + 0x1000,
        "the 423 binding wins over whatever is mapped at the address now"
    );
}

#[test]
fn gsv_hands_out_initialized_nvsrv_sessions() {
    let (context, client, gsv) = gsv();
    assert_eq!(
        client.send(gsv, &Request::tipc(1402)).unwrap().result,
        super::UNKNOWN,
        "no nvsrv before nvdrv is registered"
    );
    crate::nvdrv::register(&context).unwrap();
    let nvsrv = session(&client, gsv, 1402);
    let mut name = b"/dev/nvhost-tsg-gpu".to_vec();
    name.push(0);
    let buffer = client.alloc(&name);
    let request = Request::new(0).buffer_a(buffer, name.len() as u64);
    let mut reply = client.send(nvsrv, &request).unwrap();
    assert_eq!(reply.result, ResultCode::SUCCESS);
    let (fd, error) = (reply.pop::<i32>(), reply.pop::<u32>());
    assert!(fd > 0, "opened without Initialize");
    assert_eq!(error, 0);
}

#[test]
fn video_decoder_sessions_hand_out_decoder_services() {
    let (context, client, gsv) = gsv();
    let decoder = session(&client, gsv, 902);
    assert_eq!(
        client.send(decoder, &Request::tipc(1)).unwrap().result,
        super::UNKNOWN,
        "no decoder service before nvdrv is registered"
    );
    crate::nvdrv::register(&context).unwrap();
    for command in [902, 905, 906, 1002] {
        let decoder = session(&client, gsv, command);
        let service = session(&client, decoder, 1);
        let ready = session(&client, service, 3);
        assert!(event(&client, ready).is_signaled(), "gsv {command}");
        let reply = client.send(service, &Request::tipc(200)).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
    }
}

struct Vdec {
    context: Arc<ServiceContext>,
    client: IpcClient,
    service: Handle,
    frame: u64,
}

impl Vdec {
    /// A decoder service through gsv 902, initialized (200), configured (101) for a 1280x720
    /// VP9 movie and started (6), as Mario Tennis Fever opens its movie.
    fn open() -> Self {
        let (context, client, gsv) = gsv();
        crate::nvdrv::register(&context).unwrap();
        let decoder = session(&client, gsv, 902);
        let service = session(&client, decoder, 1);
        let frame = client.alloc(&[0x82; 64]);
        let vdec = Self {
            context,
            client,
            service,
            frame,
        };
        let initialize = Request::tipc(200).arg(0x5a_0000u64).arg([1u32, 0, 0]);
        assert_eq!(vdec.send(&initialize).result, ResultCode::SUCCESS);
        let codec = vdec.client.alloc(b"video/vp9\0");
        let configure = Request::tipc(101)
            .buffer_a(codec, 10)
            .arg([1280u32, 720, 60, 0]);
        assert_eq!(vdec.send(&configure).result, ResultCode::SUCCESS);
        assert_eq!(vdec.send(&Request::tipc(6)).result, ResultCode::SUCCESS);
        vdec
    }

    fn send(&self, request: &Request) -> lichii_kernel::testing::Reply {
        self.client.send(self.service, request).unwrap()
    }

    /// 311 or 314 for `image`: the NativeSyncHandle's syncpoint and value.
    fn hand_out(&self, command: u32, image: u32) -> (u32, u32) {
        let mut request = Request::tipc(command).arg(image);
        if command == 314 {
            request = request.arg(0u32).arg([!40u32, 0]).arg(7u64);
        }
        let mut reply = self.send(&request);
        assert_eq!(reply.result, ResultCode::SUCCESS, "{command} {image}");
        let (counter, mode, value) = (reply.pop::<u32>(), reply.pop::<u32>(), reply.pop::<u64>());
        assert_eq!(mode, 0);
        (!counter, value as u32)
    }

    fn submit(&self, pts: u64, end: bool) -> lichii_kernel::testing::Reply {
        let size = if end { 0 } else { 64 };
        self.send(
            &Request::tipc(109)
                .buffer_a(self.frame, size)
                .arg(pts)
                .arg(u8::from(end)),
        )
    }

    fn wait(&self, syncpoint: u32, value: u32) {
        let guest = self.context.gpu.host1x().syncpoints.guest();
        let reached = guest.wait(syncpoint, value, Some(Duration::from_secs(5)));
        assert!(
            reached.is_ok(),
            "syncpoint {syncpoint} never reached {value}"
        );
    }

    /// 213: the image, width, pts in ns and end of stream of the next filled image.
    fn output(&self) -> Option<(u32, u32, u64, bool)> {
        let mut reply = self.send(&Request::tipc(213));
        if reply.result != ResultCode::SUCCESS {
            assert_eq!(reply.result, ResultCode::new(111, 3604));
            return None;
        }
        let words = reply.pop::<[u32; 6]>();
        let (pts, end) = (reply.pop::<u64>(), reply.pop::<u32>());
        assert_eq!(reply.data().len(), 64);
        Some((words[0], words[1], pts, end != 0))
    }

    /// Closes the session and waits until its syncpoint is free again.
    fn close(self, syncpoint: u32) {
        self.client.process().handles().write().remove(self.service);
        let nvdrv = self.context.nvdrv.get().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let id = nvdrv.allocate_syncpoint();
            nvdrv.retire_syncpoint(id);
            if id == syncpoint {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "syncpoint {syncpoint} was not freed"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

#[test]
fn video_decoder_plays_a_movie_at_its_pts_pace_and_ends_it() {
    let vdec = Vdec::open();
    let fences: Vec<_> = (1..=3).map(|image| vdec.hand_out(311, image)).collect();
    let syncpoint = fences[0].0;
    let base = vdec
        .context
        .gpu
        .host1x()
        .syncpoints
        .guest()
        .value(syncpoint)
        .unwrap();
    assert_eq!(
        fences,
        [
            (syncpoint, base + 1),
            (syncpoint, base + 2),
            (syncpoint, base + 3)
        ]
    );
    let ready = event(&vdec.client, vdec.send(&Request::tipc(3)).copy_handles[0]);
    assert!(ready.is_signaled());
    ready.clear();
    std::thread::sleep(Duration::from_millis(20));
    assert!(ready.is_signaled(), "signalled again while there is room");

    let start = Instant::now();
    for (index, pts) in [0, 20_000, 40_000].into_iter().enumerate() {
        let mut reply = vdec.submit(pts, false);
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u64>(), index as u64 + 1);
    }
    assert_eq!(vdec.submit(40_000, true).result, ResultCode::SUCCESS);
    vdec.wait(syncpoint, base + 3);
    assert!(
        start.elapsed() >= Duration::from_millis(40),
        "the last frame came {:?} after the first",
        start.elapsed()
    );
    for (image, pts) in [(1, 0), (2, 20_000_000), (3, 40_000_000)] {
        assert_eq!(vdec.output(), Some((image, 1280, pts, false)));
    }
    assert_eq!(vdec.output(), None);

    assert_eq!(
        vdec.hand_out(314, 1),
        (syncpoint, base + 4),
        "a released image waits for the next frame"
    );
    vdec.wait(syncpoint, base + 4);
    assert_eq!(
        vdec.output(),
        Some((1, 1280, 40_000_000, true)),
        "end of stream completes as an empty frame"
    );
    assert_eq!(vdec.output(), None);
    ready.clear();
    std::thread::sleep(Duration::from_millis(20));
    assert!(
        !ready.is_signaled(),
        "no input is wanted after the end of stream"
    );

    let (_, value) = vdec.hand_out(314, 2);
    let guest = vdec.context.gpu.host1x().syncpoints.clone();
    vdec.close(syncpoint);
    assert!(
        guest.guest().is_reached(syncpoint, value).unwrap(),
        "closing completes the fences handed out"
    );
}

#[test]
fn video_decoder_counts_unregistered_images_off_in_turn() {
    let vdec = Vdec::open();
    let fences: Vec<_> = (1..=3).map(|image| vdec.hand_out(311, image)).collect();
    let (syncpoint, last) = fences[2];
    assert_eq!(
        vdec.send(&Request::tipc(212).arg(2u32)).result,
        ResultCode::SUCCESS
    );
    for pts in [0, 1_000] {
        assert_eq!(vdec.submit(pts, false).result, ResultCode::SUCCESS);
    }
    vdec.wait(syncpoint, last);
    assert_eq!(vdec.output(), Some((1, 1280, 0, false)));
    assert_eq!(
        vdec.output(),
        Some((3, 1280, 1_000_000, false)),
        "the unregistered image took no frame"
    );
    assert_eq!(vdec.output(), None);
    vdec.close(syncpoint);
}

#[test]
fn video_decoder_stop_completes_handed_fences_and_full_queues_are_busy() {
    let vdec = Vdec::open();
    for pts in 0..24 {
        assert_eq!(vdec.submit(pts * 16_667, false).result, ResultCode::SUCCESS);
    }
    assert_eq!(
        vdec.submit(24 * 16_667, false).result,
        ResultCode::new(111, 3603),
        "busy while no image takes a frame"
    );
    assert_eq!(
        vdec.submit(25 * 16_667, true).result,
        ResultCode::SUCCESS,
        "end of stream is never refused"
    );

    let fences: Vec<_> = (1..=3).map(|image| vdec.hand_out(311, image)).collect();
    let (syncpoint, last) = fences[2];
    assert_eq!(vdec.send(&Request::tipc(207)).result, ResultCode::SUCCESS);
    let guest = vdec.context.gpu.host1x().syncpoints.guest();
    assert!(guest.is_reached(syncpoint, last).unwrap());
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(
        guest.value(syncpoint).unwrap(),
        last,
        "stopped images take no more frames"
    );
    vdec.close(syncpoint);
}

fn decoder_preference(code: u32, width: u32, height: u32) -> Vec<u32> {
    let mut preference = vec![0; 16];
    preference[1] = code;
    preference[2] = (height - 1) << 16 | (width - 1);
    preference
}

#[test]
fn decoder_image_pools_are_sized_and_described_like_the_codec_expects() {
    let (_context, client, gsv) = gsv();
    let nv12 = decoder_preference(1, 640, 360);
    let mut reply = client
        .send(
            gsv,
            &words(Request::tipc(991), &[nv12.clone(), vec![2]].concat()),
        )
        .unwrap();
    assert_eq!(
        reply.pop::<[u64; 2]>(),
        [2 * 0x6_4000, 0x2000],
        "640x360 NV12 is 0x64000 bytes"
    );

    let mut reply = client
        .send(
            gsv,
            &words(Request::tipc(911), &[nv12, vec![3, 7]].concat()),
        )
        .unwrap();
    let pool = reply.pop::<u32>();
    let mut reply = client.send(gsv, &Request::tipc(961).arg(pool + 1)).unwrap();
    let descriptor = reply.pop::<[u32; 32]>();
    assert_eq!(
        [
            descriptor[0],
            descriptor[2],
            descriptor[4],
            descriptor[6],
            descriptor[8]
        ],
        [0xa, 0x6_4000, 640, 0x6_4000 + 640 * 384, 640]
    );

    let mut huge = decoder_preference(1, 640, 360);
    huge.extend([u32::MAX, 7]);
    let reply = client.send(gsv, &words(Request::tipc(911), &huge)).unwrap();
    assert_ne!(reply.result, ResultCode::SUCCESS, "pools are bounded");

    let bgra = decoder_preference(2, 100, 50);
    let mut reply = client
        .send(
            gsv,
            &words(Request::tipc(911), &[bgra, vec![1, 7]].concat()),
        )
        .unwrap();
    let image = reply.pop::<u32>();
    assert_eq!(image, pool + 3);
    let mut reply = client.send(gsv, &Request::tipc(961).arg(image)).unwrap();
    let descriptor = reply.pop::<[u32; 32]>();
    assert_eq!((descriptor[0], descriptor[4], descriptor[6]), (0x9, 448, 0));

    client.send(gsv, &Request::tipc(912).arg(pool + 1)).unwrap();
    let mut reply = client.send(gsv, &Request::tipc(961).arg(pool + 1)).unwrap();
    assert_eq!(
        reply.pop::<[u32; 32]>(),
        [0; 32],
        "a destroyed image is gone"
    );
    let mut reply = client.send(gsv, &Request::tipc(961).arg(pool + 2)).unwrap();
    assert_eq!(
        reply.pop::<[u32; 32]>()[2],
        2 * 0x6_4000,
        "its siblings keep their place in the pool"
    );

    let mut reply = client
        .send(
            gsv,
            &words(Request::tipc(429), &[hd_preference(), vec![1, 1]].concat()),
        )
        .unwrap();
    let native = reply.pop::<u32>();
    let mut reply = client.send(gsv, &Request::tipc(961).arg(native)).unwrap();
    assert_eq!(
        reply.pop::<[u32; 32]>()[0],
        0,
        "only decoder images have a CPU descriptor"
    );
}
