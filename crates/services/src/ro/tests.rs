use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_cpu::HaltReason;
use lichii_crypto::sha256;
use lichii_kernel::ipc::Handle;
use lichii_kernel::memory::{MemoryState, attribute};
use lichii_kernel::testing::JitCore;

use super::*;
use crate::testing::{self, Guest, Request};
use lichii_kernel::testing::program_image;

const CURRENT_PROCESS: Handle = 0xffff_8001;
const BRK: u32 = 0xd420_0000;
const NRR: u64 = 0;
const NRO_A: u64 = 0x1000;
const BSS_A: u64 = 0x4000;
const NRO_B: u64 = 0x10000;

/// A three-page NRO whose code at 0x100 returns `value` in w0.
fn nro(value: u32, module: u8, bss_size: u32) -> Vec<u8> {
    let mut image = vec![0u8; 0x3000];
    for (offset, word) in [
        (0x10, u32::from_le_bytes(*b"NRO0")),
        (0x18, 0x3000),
        (0x24, 0x1000),
        (0x28, 0x1000),
        (0x2c, 0x1000),
        (0x30, 0x2000),
        (0x34, 0x1000),
        (0x38, bss_size),
        (0x100, 0x5280_0000 | value << 5),
        (0x104, BRK),
    ] {
        image[offset..offset + 4].copy_from_slice(&u32::to_le_bytes(word));
    }
    image[0x40] = module;
    image
}

/// A one-page NRR listing `hashes`, with the hash list fields where `fields` says.
fn nrr(fields: usize, hashes: &[Sha256Hash]) -> Vec<u8> {
    let mut image = vec![0u8; 0x1000];
    let offset = 0x500;
    image[fields..fields + 4].copy_from_slice(&(offset as u32).to_le_bytes());
    image[fields + 4..fields + 8].copy_from_slice(&(hashes.len() as u32).to_le_bytes());
    image[offset..offset + hashes.len() * 32].copy_from_slice(hashes.as_flattened());
    image
}

struct Client {
    guest: Guest,
    heap: VAddr,
}

impl Client {
    fn new(program_id: u64) -> Self {
        Self::with(program_id, |c| register(c).unwrap())
    }

    /// A client whose ro sessions place every module at the same free page, so a module loads
    /// where the last one was unloaded.
    fn placing_at_one_address(program_id: u64) -> Self {
        Self::with(program_id, |c| {
            let area = c.process.page_table().regions().alias_code;
            let page = (area.size() - 0x100_0000) / PAGE_SIZE;
            register_with(c, Box::new(move || page)).unwrap()
        })
    }

    fn with(program_id: u64, register: impl FnOnce(&Arc<ServiceContext>)) -> Self {
        let guest = Guest::new(program_image(program_id), Settings::default(), register);
        let heap = guest.process.page_table().set_heap_size(2 << 20).unwrap();
        Self { guest, heap }
    }

    fn call(&self, session: Handle, command: u32, args: &[u64]) -> (ResultCode, u64) {
        let mut request = Request::new(command).send_pid().arg(0u64);
        for &arg in args {
            request = request.arg(arg);
        }
        if command == 4 {
            request = request.copy_handle(CURRENT_PROCESS);
        }
        let mut reply = self.guest.call(session, request);
        let out = if reply.result == ResultCode::SUCCESS && command == 0 {
            reply.pop()
        } else {
            0
        };
        (reply.result, out)
    }

    /// A session to `name` with the client's process registered.
    fn session(&self, name: &str) -> Handle {
        let session = self.guest.service(name);
        assert_eq!(self.call(session, 4, &[]).0, ResultCode::SUCCESS);
        session
    }

    fn register_nrr(&self, session: Handle) -> ResultCode {
        self.call(session, 2, &[self.heap + NRR, 0x1000]).0
    }

    fn load(&self, session: Handle, nro: u64, bss: u64, bss_size: u64) -> (ResultCode, u64) {
        self.call(
            session,
            0,
            &[self.heap + nro, 0x3000, self.heap + bss, bss_size],
        )
    }
}

#[test]
fn nros_load_run_unload_and_reload_at_the_same_address() {
    let client = Client::placing_at_one_address(testing::SWITCH1_TITLE);
    let (heap, table) = (client.heap, client.guest.process.page_table());
    let (a, b) = (nro(1, 0xa, 0x1000), nro(2, 0xb, 0));
    client
        .guest
        .write(heap + NRR, &nrr(0x340, &[sha256(&a), sha256(&b)]));
    client.guest.write(heap + NRO_A, &a);
    client.guest.write(heap + NRO_B, &b);
    let mut cpu = JitCore::new(client.guest.kernel(), &client.guest.process);

    let session = client.session("ldr:ro");
    assert_eq!(client.register_nrr(session), ResultCode::SUCCESS);
    let (result, base) = client.load(session, NRO_A, BSS_A, 0x1000);
    assert_eq!(result, ResultCode::SUCCESS);
    let perms = [0, 0x1000, 0x2000, 0x3000].map(|offset| table.query(base + offset).permission);
    assert_eq!(perms, [5, 1, 3, 3]);
    assert_eq!(
        table.query(base + 0x2000).state,
        MemoryState::ALIAS_CODE_DATA.svc_value()
    );
    assert_eq!(table.query(heap + NRO_A).attribute, attribute::LOCKED);
    assert_eq!(cpu.run(base + 0x100), (HaltReason::Breakpoint(0), 1));

    assert_eq!(client.call(session, 1, &[base]).0, ResultCode::SUCCESS);
    assert_eq!(table.query(base).state, MemoryState::FREE.svc_value());
    let heap_info = table.query(heap + NRO_A);
    assert_eq!((heap_info.permission, heap_info.attribute), (3, 0));
    let (result, reloaded) = client.load(session, NRO_B, 0, 0);
    assert_eq!(result, ResultCode::SUCCESS);
    assert_eq!(
        reloaded, base,
        "the generator picks the page the last module left"
    );
    assert_eq!(cpu.run(base + 0x100), (HaltReason::Breakpoint(0), 2));
}

#[test]
fn modules_must_be_listed_in_a_registered_nrr() {
    let client = Client::new(testing::SWITCH1_TITLE);
    let heap = client.heap;
    let a = nro(1, 0xa, 0);
    client.guest.write(heap + NRO_A, &a);
    client.guest.write(heap + NRR, &nrr(0x340, &[sha256(&a)]));

    let session = client.guest.service("ro:1");
    assert_eq!(client.register_nrr(session), INVALID_PROCESS);
    assert_eq!(client.call(session, 4, &[]).0, ResultCode::SUCCESS);
    assert_eq!(client.call(session, 4, &[]).0, INVALID_SESSION);
    assert_eq!(client.load(session, NRO_A, 0, 0).0, NOT_AUTHORIZED);
    assert_eq!(
        client
            .guest
            .process
            .page_table()
            .query(heap + NRO_A)
            .permission,
        3
    );

    assert_eq!(
        client.call(session, 10, &[heap + NRR + 1, 0x1000]).0,
        INVALID_ADDRESS
    );
    assert_eq!(client.call(session, 10, &[heap + NRR, 0]).0, INVALID_SIZE);
    assert_eq!(
        client.call(session, 10, &[heap + NRR, 0x1000]).0,
        ResultCode::SUCCESS
    );
    let (result, base) = client.load(session, NRO_A, 0, 0);
    assert_eq!(result, ResultCode::SUCCESS);
    assert_eq!(client.load(session, NRO_B, 0, 0).0, INVALID_NRO);
    assert_eq!(client.call(session, 1, &[base + 0x1000]).0, NOT_LOADED);
    assert_eq!(client.call(session, 3, &[heap + 0x2000]).0, NOT_REGISTERED);
    assert_eq!(
        client.call(session, 3, &[heap + NRR]).0,
        ResultCode::SUCCESS
    );
    assert_eq!(client.call(session, 1, &[base]).0, ResultCode::SUCCESS);
    assert_eq!(client.load(session, NRO_A, 0, 0).0, NOT_AUTHORIZED);
}

#[test]
fn a_module_loads_once_and_the_bss_size_must_match() {
    let client = Client::new(testing::SWITCH1_TITLE);
    let heap = client.heap;
    let (a, copy) = (nro(1, 0xa, 0), nro(2, 0xa, 0));
    client.guest.write(heap + NRO_A, &a);
    client.guest.write(heap + NRO_B, &copy);
    client
        .guest
        .write(heap + NRR, &nrr(0x340, &[sha256(&a), sha256(&copy)]));
    let session = client.session("ldr:ro");
    assert_eq!(client.register_nrr(session), ResultCode::SUCCESS);
    assert_eq!(client.load(session, NRO_A, BSS_A, 0x1000).0, INVALID_NRO);
    assert_eq!(client.load(session, NRO_A, 0, 0).0, ResultCode::SUCCESS);
    assert_eq!(client.load(session, NRO_B, 0, 0).0, ALREADY_LOADED);
}

#[test]
fn nrr_hash_lists_stay_inside_the_nrr() {
    let client = Client::new(testing::SWITCH1_TITLE);
    let session = client.session("ldr:ro");
    let mut image = nrr(0x340, &[]);
    image[0x344..0x348].copy_from_slice(&0x100u32.to_le_bytes());
    client.guest.write(client.heap + NRR, &image);
    assert_eq!(client.register_nrr(session), INVALID_NRR);
}

#[test]
fn switch2_nrrs_keep_their_hash_list_fields_after_the_larger_keys() {
    let client = Client::new(testing::SWITCH2_TITLE);
    let heap = client.heap;
    let a = nro(1, 0xa, 0);
    client.guest.write(heap + NRO_A, &a);
    client.guest.write(heap + NRR, &nrr(0x4c0, &[sha256(&a)]));
    let session = client.session("ldr:ro");
    assert_eq!(client.register_nrr(session), ResultCode::SUCCESS);
    assert_eq!(client.load(session, NRO_A, 0, 0).0, ResultCode::SUCCESS);
}
