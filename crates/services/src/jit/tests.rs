use std::sync::Arc;

use lichii_common::settings::Settings;
use lichii_common::{ResultCode, VAddr};
use lichii_kernel::handle_table::CURRENT_PROCESS;
use lichii_kernel::ipc::{Handle, UNKNOWN_COMMAND_ID};
use lichii_kernel::memory::{MemoryState, Region, Regions};
use lichii_kernel::memory_objects::CodeMemory;
use lichii_kernel::object::KObject;
use lichii_kernel::result;
use lichii_kernel::testing::IpcClient;
use lichii_memory::MemoryPermission;

use super::{UNKNOWN, register, skip_reserved};
use crate::testing::{ReplyExt, Request, SWITCH1_TITLE, SWITCH2_TITLE, context};

const SIZE: u64 = 0x2000;

/// A client in `title` with jit:u and two code memory objects over its heap.
fn setup(title: u64) -> (IpcClient, Handle, Handle) {
    let (context, client) = context(title, Settings::default());
    register(&context).unwrap();
    let process = client.process().clone();
    let heap = process.page_table().set_heap_size(2 << 20).unwrap();
    let code = |address: VAddr| {
        let memory = CodeMemory::new(process.clone(), address, SIZE).unwrap();
        process
            .handles()
            .write()
            .add(KObject::CodeMemory(Arc::new(memory)))
            .unwrap()
    };
    let (rx, ro) = (code(heap), code(heap + SIZE));
    (client, rx, ro)
}

fn create(client: &IpcClient, rx: Handle, ro: Handle, rx_size: u64) -> crate::testing::Reply {
    let jit = client.connect("jit:u").unwrap();
    let request = Request::new(0)
        .arg(rx_size)
        .arg(SIZE)
        .copy_handle(CURRENT_PROCESS)
        .copy_handle(rx)
        .copy_handle(ro);
    client.send(jit, &request).unwrap()
}

#[test]
fn jit_environment_maps_both_code_ranges_in_the_alias_code_region() {
    let (client, rx, ro) = setup(SWITCH1_TITLE);
    let environment = create(&client, rx, ro, SIZE).ok().interface();
    let reply = client.send(environment, &Request::new(1001)).unwrap().ok();
    let (rx_address, ro_address) = (reply.get::<u64>(0), reply.get::<u64>(8));
    assert_ne!(rx_address, ro_address);

    let table = client.process().page_table();
    for (address, permission) in [
        (rx_address, MemoryPermission::READ_EXECUTE),
        (ro_address, MemoryPermission::READ),
    ] {
        assert!(table.can_contain_alias_code(address, SIZE), "{address:#x}");
        let info = table.query(address);
        assert_eq!(info.state, MemoryState::CODE_OUT.0 & 0xff);
        assert_eq!(info.permission, u32::from(permission.0));
    }
}

#[test]
fn code_ranges_skip_the_heap_and_alias_regions() {
    let alias_code = Region::new(0x800_0000, 0x10_0000_0000);
    let regions = Regions {
        address_space_bits: 39,
        code: alias_code,
        alias_code,
        alias: Region::new(0x1_0000_0000, 0x1_0000_0000),
        heap: Region::new(0x2_0000_0000, 0x1_0000_0000),
        stack: Region::new(0x3_0000_0000, 0x1000_0000),
        kernel_map: Region::new(0x3_1000_0000, 0x1000_0000),
        alias_extra_size: 0,
    };
    assert_eq!(skip_reserved(&regions, 0x900_0000, SIZE), 0x900_0000);
    assert_eq!(
        skip_reserved(&regions, 0x1_0000_0000 - 0x1000, SIZE),
        0x3_0000_0000,
        "crossing into the alias region, then the adjacent heap"
    );
    assert_eq!(skip_reserved(&regions, 0x2_8000_0000, SIZE), 0x3_0000_0000);
    assert_eq!(skip_reserved(&regions, 0x3_0000_0000, SIZE), 0x3_0000_0000);
}

#[test]
fn create_needs_all_three_handles_and_the_code_memory_size() {
    let (client, rx, ro) = setup(SWITCH1_TITLE);
    let jit = client.connect("jit:u").unwrap();
    let reply = client
        .send(
            jit,
            &Request::new(0)
                .arg(SIZE)
                .arg(SIZE)
                .copy_handle(CURRENT_PROCESS),
        )
        .unwrap();
    assert_eq!(reply.result, UNKNOWN);
    assert_eq!(create(&client, rx, ro, 0x1000).result, result::INVALID_SIZE);
    create(&client, rx, ro, SIZE).ok();
}

#[test]
fn plugin_commands_are_auto_stubbed_only_for_switch_2_titles() {
    for (title, expected) in [
        (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
        (SWITCH2_TITLE, ResultCode::SUCCESS),
    ] {
        let (client, rx, ro) = setup(title);
        let environment = create(&client, rx, ro, SIZE).ok().interface();
        let reply = client.send(environment, &Request::new(1000)).unwrap();
        assert_eq!(reply.result, expected, "title {title:#x}");
    }
}
