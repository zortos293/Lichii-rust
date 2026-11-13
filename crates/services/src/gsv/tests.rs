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
