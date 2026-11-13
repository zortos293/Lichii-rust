use std::sync::Arc;

use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::Event;
use lichii_kernel::ipc::Handle;
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
