//! Synthetic `mii:u` and `mii:e` tests. The byte vectors come from the reference's own mii types
//! (`types/*.cpp`), compiled with a fixed create id.

use bytemuck::{Zeroable, bytes_of, pod_read_unaligned};
use lichii_common::ResultCode;
use lichii_common::settings::Settings;
use lichii_kernel::ipc::{Handle, UNKNOWN_COMMAND_ID};
use lichii_kernel::testing::IpcClient;

use super::types::{ALL, CharInfo, CoreData, GENDER_ALL, Rng, StoreData, Ver3StoreData};
use super::{
    INVALID_ARGUMENT, INVALID_ARGUMENT_SIZE, INVALID_CHAR_INFO, NOT_FOUND, NOT_UPDATED,
    PERMISSION_DENIED, TEST_MODE_ONLY,
};
use crate::testing::{ReplyExt, Request, SWITCH1_TITLE, SWITCH2_TITLE, booted};

/// The create id the reference vectors were made with.
const REFERENCE_ID: [u8; 16] = [
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

/// The reference's `CharInfo` of each built-in Mii under [`REFERENCE_ID`].
const DEFAULT_CHAR_INFO: [[u8; 0x58]; 6] = [
    [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x6e, 0x00, 0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x40, 0x40, 0x00, 0x00,
        0x00, 0x04, 0x00, 0x00, 0x44, 0x08, 0x00, 0x02, 0x08, 0x04, 0x03, 0x04, 0x02, 0x0c, 0x06,
        0x08, 0x04, 0x03, 0x06, 0x02, 0x0a, 0x01, 0x04, 0x09, 0x17, 0x13, 0x04, 0x03, 0x0d, 0x08,
        0x00, 0x00, 0x04, 0x0a, 0x00, 0x08, 0x04, 0x0a, 0x00, 0x04, 0x02, 0x14, 0x00,
    ],
    [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x6e, 0x00, 0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x05, 0x00, 0x40, 0x40, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x37, 0x06, 0x00, 0x02, 0x0c, 0x04, 0x03, 0x04, 0x02, 0x0c, 0x06,
        0x06, 0x04, 0x03, 0x06, 0x02, 0x0a, 0x01, 0x04, 0x09, 0x17, 0x13, 0x04, 0x03, 0x0d, 0x08,
        0x00, 0x00, 0x04, 0x0a, 0x00, 0x08, 0x04, 0x0a, 0x00, 0x04, 0x02, 0x14, 0x00,
    ],
    [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x6e, 0x00, 0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x40, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x21, 0x01, 0x00, 0x02, 0x08, 0x04, 0x03, 0x04, 0x02, 0x0c, 0x06,
        0x01, 0x04, 0x03, 0x06, 0x02, 0x0a, 0x01, 0x04, 0x09, 0x17, 0x13, 0x04, 0x03, 0x0d, 0x08,
        0x00, 0x00, 0x04, 0x0a, 0x00, 0x08, 0x04, 0x0a, 0x00, 0x04, 0x02, 0x14, 0x00,
    ],
    [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x6e, 0x00, 0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x01, 0x40, 0x40, 0x00, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x18, 0x08, 0x00, 0x04, 0x08, 0x04, 0x03, 0x03, 0x02, 0x0c, 0x00,
        0x08, 0x04, 0x03, 0x06, 0x02, 0x0a, 0x01, 0x04, 0x09, 0x17, 0x13, 0x04, 0x03, 0x0d, 0x08,
        0x00, 0x00, 0x04, 0x0a, 0x00, 0x08, 0x04, 0x0a, 0x00, 0x04, 0x02, 0x14, 0x00,
    ],
    [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x6e, 0x00, 0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x01, 0x40, 0x40, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0e, 0x07, 0x00, 0x04, 0x0d, 0x04, 0x03, 0x03, 0x02, 0x0c, 0x00,
        0x07, 0x04, 0x03, 0x06, 0x02, 0x0a, 0x01, 0x04, 0x09, 0x17, 0x13, 0x04, 0x03, 0x0d, 0x08,
        0x00, 0x00, 0x04, 0x0a, 0x00, 0x08, 0x04, 0x0a, 0x00, 0x04, 0x02, 0x14, 0x00,
    ],
    [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e,
        0x1f, 0x6e, 0x00, 0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x07, 0x01, 0x40, 0x40, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x0c, 0x01, 0x00, 0x04, 0x08, 0x04, 0x03, 0x03, 0x02, 0x0c, 0x00,
        0x01, 0x04, 0x03, 0x06, 0x02, 0x0a, 0x01, 0x04, 0x09, 0x17, 0x13, 0x04, 0x03, 0x0d, 0x08,
        0x00, 0x00, 0x04, 0x0a, 0x00, 0x08, 0x04, 0x0a, 0x00, 0x04, 0x02, 0x14, 0x00,
    ],
];

/// The reference's `StoreData` of each built-in Mii under [`REFERENCE_ID`], checksums included.
const DEFAULT_STORE_DATA: [[u8; 0x44]; 6] = [
    [
        0x44, 0x40, 0x40, 0x08, 0x08, 0x08, 0x13, 0x08, 0x08, 0x02, 0x17, 0x8c, 0x06, 0x01, 0x69,
        0x6d, 0x8a, 0x6a, 0x82, 0x14, 0x00, 0x04, 0x04, 0x20, 0x64, 0x72, 0x44, 0x44, 0x6e, 0x00,
        0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f, 0x2f, 0x17, 0xc8, 0xd6,
    ],
    [
        0x37, 0x40, 0x40, 0x06, 0x0c, 0x06, 0x13, 0x08, 0x08, 0x02, 0x17, 0x8c, 0x06, 0x01, 0x69,
        0x6d, 0x8a, 0x6a, 0x82, 0x14, 0x00, 0x05, 0x00, 0x20, 0x64, 0x72, 0x44, 0x44, 0x6e, 0x00,
        0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f, 0xe3, 0x09, 0xc8, 0xd6,
    ],
    [
        0x21, 0x40, 0x40, 0x01, 0x08, 0x01, 0x13, 0x08, 0x08, 0x02, 0x17, 0x8c, 0x06, 0x01, 0x69,
        0x6d, 0x8a, 0x6a, 0x82, 0x14, 0x00, 0x00, 0x01, 0x20, 0x64, 0x72, 0x44, 0x44, 0x6e, 0x00,
        0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f, 0x3f, 0xcc, 0xc8, 0xd6,
    ],
    [
        0x18, 0x40, 0x40, 0x08, 0x88, 0x08, 0x13, 0x08, 0x08, 0x04, 0x17, 0x8c, 0x00, 0x01, 0x69,
        0x6d, 0x6a, 0x6a, 0x82, 0x14, 0x00, 0x02, 0x02, 0x20, 0x64, 0x72, 0x44, 0x44, 0x6e, 0x00,
        0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f, 0xdc, 0xf6, 0xc8, 0xd6,
    ],
    [
        0x0e, 0x40, 0x40, 0x07, 0x8d, 0x07, 0x13, 0x08, 0x08, 0x04, 0x17, 0x8c, 0x00, 0x01, 0x69,
        0x6d, 0x6a, 0x6a, 0x82, 0x14, 0x00, 0x06, 0x00, 0x20, 0x64, 0x72, 0x44, 0x44, 0x6e, 0x00,
        0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f, 0x1d, 0x35, 0xc8, 0xd6,
    ],
    [
        0x0c, 0x40, 0x40, 0x01, 0x88, 0x01, 0x13, 0x08, 0x08, 0x04, 0x17, 0x8c, 0x00, 0x01, 0x69,
        0x6d, 0x6a, 0x6a, 0x82, 0x14, 0x00, 0x07, 0x00, 0x20, 0x64, 0x72, 0x44, 0x44, 0x6e, 0x00,
        0x6f, 0x00, 0x20, 0x00, 0x6e, 0x00, 0x61, 0x00, 0x6d, 0x00, 0x65, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b,
        0x1c, 0x1d, 0x1e, 0x1f, 0x87, 0x2d, 0xc8, 0xd6,
    ],
];

/// A synthetic Ver3 Mii that sets every value the conversion reads.
const VER3: [u8; 0x60] = [
    0x03, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x1c, 0x56, 0x00, 0x33, 0x00, 0x6d, 0x00,
    0x69, 0x00, 0x69, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x5a, 0x1e,
    0x6a, 0x42, 0x4d, 0x0e, 0xa8, 0x4a, 0x26, 0x1f, 0xb4, 0x57, 0x69, 0x1d, 0xd0, 0x22, 0xde, 0xae,
    0x90, 0x00, 0xbb, 0x35, 0x58, 0x9b, 0xef, 0x75, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// The reference's `CharInfo` of [`VER3`] under [`REFERENCE_ID`].
const VER3_CHAR_INFO: [u8; 0x58] = [
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x46, 0x17, 0x98, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
    0x56, 0x00, 0x33, 0x00, 0x6d, 0x00, 0x69, 0x00, 0x69, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x5a, 0x1e, 0x00, 0x00, 0x05, 0x03, 0x02,
    0x04, 0x4d, 0x06, 0x01, 0x28, 0x0a, 0x05, 0x02, 0x06, 0x09, 0x0f, 0x14, 0x05, 0x07, 0x05, 0x09,
    0x0b, 0x0e, 0x10, 0x06, 0x11, 0x1e, 0x16, 0x07, 0x05, 0x10, 0x07, 0x03, 0x04, 0x06, 0x0d, 0x08,
    0x12, 0x06, 0x13, 0x01, 0x07, 0x0f, 0x1d, 0x00,
];

/// The reference's `CoreData` of [`VER3_CHAR_INFO`].
const VER3_CORE_DATA: [u8; 0x30] = [
    0x4d, 0xda, 0x9e, 0x06, 0x8a, 0x05, 0x16, 0x07, 0x12, 0x28, 0x5e, 0xcf, 0x94, 0x70, 0xb1, 0xb0,
    0xcd, 0x53, 0xaf, 0x1d, 0x08, 0x57, 0x23, 0x94, 0x97, 0xbb, 0x76, 0x76, 0x56, 0x00, 0x33, 0x00,
    0x6d, 0x00, 0x69, 0x00, 0x69, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

const DATABASE: u32 = 1;
const DEFAULT: u32 = 2;
const CHAR_INFO: usize = 0x58;
const STORE_DATA: usize = 0x44;

fn seeded(title: u64, seed: u32) -> IpcClient {
    let mut settings = Settings::default();
    settings.system.rng_seed = Some(seed);
    booted(title, settings, super::register)
}

/// A new `IDatabaseService` from `service`.
fn database(client: &IpcClient, service: &str) -> Handle {
    let session = client.connect(service).unwrap();
    client
        .send(session, &Request::new(0))
        .unwrap()
        .ok()
        .interface()
}

fn call(client: &IpcClient, session: Handle, request: Request) -> (ResultCode, Vec<u8>) {
    let reply = client.send(session, &request).unwrap();
    (reply.result, reply.data().to_vec())
}

fn ok(client: &IpcClient, session: Handle, request: Request) -> Vec<u8> {
    let (result, data) = call(client, session, request);
    assert_eq!(result, ResultCode::SUCCESS);
    data
}

fn word<T: bytemuck::Pod>(data: &[u8]) -> T {
    pod_read_unaligned(&data[..size_of::<T>()])
}

fn count(client: &IpcClient, session: Handle, source_flag: u32) -> u32 {
    word(&ok(client, session, Request::new(2).arg(source_flag)))
}

/// `Get` (3), `Get1` (4), `Get2` (8) or `Get3` (9) into a buffer of `capacity` elements.
fn get(
    client: &IpcClient,
    session: Handle,
    command: u32,
    source_flag: u32,
    element: usize,
    capacity: usize,
) -> (ResultCode, u32, Vec<u8>) {
    let buffer = client.alloc(&vec![0xff; element * capacity]);
    let request = Request::new(command)
        .arg(source_flag)
        .buffer_b(buffer, (element * capacity) as u64);
    let (result, data) = call(client, session, request);
    let count = if result == ResultCode::SUCCESS {
        word(&data)
    } else {
        0
    };
    (result, count, client.read(buffer, element * capacity))
}

fn is_version_4(create_id: &[u8]) -> bool {
    create_id[6] & 0xf0 == 0x40 && create_id[8] & 0xc0 == 0x80
}

#[test]
fn built_in_miis_match_the_reference_bytes() {
    for (index, (char_info, store_data)) in DEFAULT_CHAR_INFO
        .iter()
        .zip(&DEFAULT_STORE_DATA)
        .enumerate()
    {
        let store = StoreData::default_mii(index, REFERENCE_ID);
        assert_eq!(bytes_of(&store), store_data, "store data {index}");
        assert!(store.is_valid());
        assert_eq!(
            bytes_of(&CharInfo::from_store_data(&store)),
            char_info,
            "char info {index}"
        );
    }
}

#[test]
fn ver3_conversion_matches_the_reference_bytes() {
    let ver3 = Ver3StoreData(VER3);
    assert!(ver3.is_valid());
    let info = CharInfo::from_store_data(&ver3.to_store_data(REFERENCE_ID));
    assert_eq!(bytes_of(&info), &VER3_CHAR_INFO);
    assert_eq!(bytes_of(&CoreData::from_char_info(&info)), &VER3_CORE_DATA);

    let mut bad = VER3;
    bad[0] = 2;
    assert!(!Ver3StoreData(bad).is_valid(), "version 2");
    let mut nameless = VER3;
    nameless[0x1a] = 0;
    assert!(!Ver3StoreData(nameless).is_valid(), "empty name");
}

#[test]
fn char_info_and_store_data_round_trip() {
    let mut rng = Rng(5);
    let mut miis: Vec<StoreData> = (0..6)
        .map(|i| StoreData::default_mii(i, rng.create_id()))
        .collect();
    miis.extend((0..64).map(|_| StoreData::random(&mut rng, ALL, GENDER_ALL, ALL)));
    miis.push(Ver3StoreData(VER3).to_store_data(rng.create_id()));
    for store in miis {
        let info = CharInfo::from_store_data(&store);
        assert!(info.is_valid());
        let core = CoreData::from_char_info(&info);
        assert_eq!(core, store.core);
        let again = StoreData::new(core, info.create_id);
        assert_eq!(again, store);
        assert!(again.matches(&store));
        assert!(info.matches(&CharInfo::from_store_data(&again)));
    }
}

#[test]
fn random_miis_are_valid_for_every_age_gender_and_race() {
    let mut rng = Rng(1);
    for age in 0..=ALL {
        for gender in 0..=GENDER_ALL {
            for race in 0..=ALL {
                for _ in 0..32 {
                    let store = StoreData::random(&mut rng, age, gender, race);
                    assert!(store.is_valid(), "age {age} gender {gender} race {race}");
                    assert!(is_version_4(&store.create_id));
                    assert!(!store.is_special());
                    assert!(CharInfo::from_store_data(&store).is_valid());
                }
            }
        }
    }
}

#[test]
fn both_services_hand_out_working_database_sessions() {
    let client = seeded(SWITCH1_TITLE, 1);
    for service in ["mii:u", "mii:e"] {
        let session = database(&client, service);
        assert_eq!(count(&client, session, DEFAULT), 6, "{service}");
        assert_eq!(count(&client, session, DATABASE), 0, "{service}");
        assert_eq!(count(&client, session, DATABASE | DEFAULT), 6, "{service}");
        assert_eq!(ok(&client, session, Request::new(1))[0], 0, "not full");
    }
}

#[test]
fn build_default_zero_is_the_reference_char_info() {
    let client = seeded(SWITCH1_TITLE, 2);
    let session = database(&client, "mii:u");
    for (index, expected) in DEFAULT_CHAR_INFO.iter().enumerate() {
        let info = ok(&client, session, Request::new(7).arg(index as i32));
        assert_eq!(info.len(), CHAR_INFO);
        assert_eq!(info[16..], expected[16..], "built-in Mii {index}");
        assert!(is_version_4(&info[..16]));
    }
    for index in [6, -1] {
        let (result, _) = call(&client, session, Request::new(7).arg(index));
        assert_eq!(result, INVALID_ARGUMENT, "index {index}");
    }
}

#[test]
fn get_returns_the_built_in_miis_after_the_database() {
    let client = seeded(SWITCH1_TITLE, 3);
    let session = database(&client, "mii:u");

    let (result, n, out) = get(&client, session, 3, DATABASE | DEFAULT, CHAR_INFO + 4, 6);
    assert_eq!((result, n), (ResultCode::SUCCESS, 6));
    for (element, expected) in out.chunks(CHAR_INFO + 4).zip(&DEFAULT_CHAR_INFO) {
        assert_eq!(element[16..CHAR_INFO], expected[16..]);
        assert_eq!(word::<u32>(&element[CHAR_INFO..]), 1, "Source::Default");
    }

    let (result, n, out) = get(&client, session, 4, DEFAULT, CHAR_INFO, 6);
    assert_eq!((result, n), (ResultCode::SUCCESS, 6));
    assert_eq!(
        out[CHAR_INFO + 16..2 * CHAR_INFO],
        DEFAULT_CHAR_INFO[1][16..]
    );

    let (result, n, out) = get(&client, session, 8, DEFAULT, STORE_DATA + 4, 6);
    assert_eq!((result, n), (ResultCode::SUCCESS, 6));
    for (element, expected) in out.chunks(STORE_DATA + 4).zip(&DEFAULT_STORE_DATA) {
        assert_eq!(element[..0x30], expected[..0x30]);
        assert_eq!(word::<u32>(&element[STORE_DATA..]), 1, "Source::Default");
    }

    let (result, n, out) = get(&client, session, 9, DEFAULT, STORE_DATA, 6);
    assert_eq!((result, n), (ResultCode::SUCCESS, 6));
    let store: StoreData = pod_read_unaligned(&out[..STORE_DATA]);
    assert!(store.is_valid());

    let (result, n, _) = get(&client, session, 3, DATABASE, CHAR_INFO + 4, 0);
    assert_eq!((result, n), (ResultCode::SUCCESS, 0), "empty database");
    let (result, ..) = get(&client, session, 4, DEFAULT, CHAR_INFO, 5);
    assert_eq!(result, INVALID_ARGUMENT_SIZE);
}

#[test]
fn build_random_repeats_with_a_seed_and_checks_its_arguments() {
    let build = |seed| {
        let client = seeded(SWITCH2_TITLE, seed);
        let session = database(&client, "mii:u");
        let request = Request::new(6).arg(3u32).arg(2u32).arg(3u32);
        (
            ok(&client, session, request.clone()),
            ok(&client, session, request),
        )
    };
    let (first, second) = build(9);
    assert_eq!(
        build(9),
        (first.clone(), second.clone()),
        "same seed, same Miis"
    );
    assert_ne!(first, second);
    let info: CharInfo = pod_read_unaligned(&first);
    assert!(info.is_valid());

    let client = seeded(SWITCH1_TITLE, 9);
    let session = database(&client, "mii:u");
    for (age, gender, race) in [(0u32, 0u32, 0u32), (2, 1, 2)] {
        let info: CharInfo = pod_read_unaligned(&ok(
            &client,
            session,
            Request::new(6).arg(age).arg(gender).arg(race),
        ));
        assert!(info.is_valid());
        assert_eq!(u32::from(info.get(super::Field::Gender)), gender);
    }
    for (age, gender, race) in [(4u32, 0u32, 0u32), (0, 3, 0), (0, 0, 4)] {
        let (result, _) = call(
            &client,
            session,
            Request::new(6).arg(age).arg(gender).arg(race),
        );
        assert_eq!(result, INVALID_ARGUMENT, "{age} {gender} {race}");
    }
}

#[test]
fn only_mii_e_edits_the_database() {
    let client = seeded(SWITCH1_TITLE, 4);
    let user = database(&client, "mii:u");
    let system = database(&client, "mii:e");
    let (_, _, out) = get(&client, system, 9, DEFAULT, STORE_DATA, 6);
    let store: StoreData = pod_read_unaligned(&out[..STORE_DATA]);
    let id = store.create_id;
    let info = CharInfo::from_store_data(&store);

    let add = Request::new(13).arg(store);
    assert_eq!(call(&client, user, add.clone()).0, PERMISSION_DENIED);
    assert_eq!(
        call(&client, user, Request::new(14).arg(id)).0,
        PERMISSION_DENIED
    );
    assert_eq!(
        call(&client, user, Request::new(12).arg(id).arg(0i32)).0,
        PERMISSION_DENIED
    );
    assert_eq!(
        call(&client, user, Request::new(10).arg(store).arg(DATABASE)).0,
        PERMISSION_DENIED
    );
    assert_eq!(call(&client, user, Request::new(20)).0, PERMISSION_DENIED);
    assert_eq!(ok(&client, system, Request::new(20))[0], 0, "not broken");

    let is_updated = |session| ok(&client, session, Request::new(0).arg(DATABASE))[0];
    assert_eq!(is_updated(user), 1, "opening a session counts as an update");
    assert_eq!(is_updated(user), 0, "seen");
    assert_eq!(ok(&client, user, Request::new(0).arg(DEFAULT))[0], 0);
    ok(&client, system, add);
    assert_eq!(count(&client, user, DATABASE), 1);
    assert_eq!(is_updated(user), 1, "the other session changed it");
    assert_eq!(is_updated(user), 0, "seen");
    assert_eq!(is_updated(system), 0, "its own change");

    let (result, n, out) = get(&client, user, 3, DATABASE | DEFAULT, CHAR_INFO + 4, 7);
    assert_eq!((result, n), (ResultCode::SUCCESS, 7));
    assert_eq!(out[..CHAR_INFO], *bytes_of(&info));
    assert_eq!(word::<u32>(&out[CHAR_INFO..]), 0, "Source::Database");

    assert_eq!(
        word::<i32>(&ok(&client, user, Request::new(11).arg(id).arg(0u8))),
        0
    );
    assert_eq!(
        word::<i32>(&ok(&client, user, Request::new(21).arg(info))),
        0
    );
    let update = Request::new(5).arg(info).arg(DATABASE);
    assert_eq!(call(&client, user, update).0, NOT_UPDATED);
    assert_eq!(
        call(&client, user, Request::new(5).arg(info).arg(DEFAULT)).0,
        NOT_FOUND
    );
    let update = Request::new(10).arg(store).arg(DATABASE);
    assert_eq!(call(&client, system, update).0, NOT_UPDATED);

    ok(&client, system, Request::new(14).arg(id));
    assert_eq!(count(&client, user, DATABASE), 0);
    assert_eq!(
        word::<i32>(&ok(&client, user, Request::new(11).arg(id).arg(0u8))),
        -1
    );
    assert_eq!(call(&client, user, Request::new(21).arg(info)).0, NOT_FOUND);
    assert_eq!(call(&client, system, Request::new(14).arg(id)).0, NOT_FOUND);
}

#[test]
fn update_latest_returns_the_newer_database_entry() {
    let client = seeded(SWITCH1_TITLE, 6);
    let system = database(&client, "mii:e");
    let (_, _, out) = get(&client, system, 9, DEFAULT, STORE_DATA, 6);
    let old: StoreData = pod_read_unaligned(&out[..STORE_DATA]);
    let newer: StoreData = pod_read_unaligned(&out[STORE_DATA..2 * STORE_DATA]);
    let newer = StoreData::new(newer.core, old.create_id);
    ok(&client, system, Request::new(13).arg(newer));

    let latest = ok(&client, system, Request::new(10).arg(old).arg(DATABASE));
    assert_eq!(latest, bytes_of(&newer));
    let info = CharInfo::from_store_data(&old);
    let latest = ok(&client, system, Request::new(5).arg(info).arg(DATABASE));
    assert_eq!(latest, bytes_of(&CharInfo::from_store_data(&newer)));
}

#[test]
fn move_reorders_the_database() {
    let client = seeded(SWITCH1_TITLE, 7);
    let system = database(&client, "mii:e");
    let (_, _, out) = get(&client, system, 9, DEFAULT, STORE_DATA, 6);
    let ids: Vec<[u8; 16]> = out
        .chunks(STORE_DATA)
        .take(3)
        .map(|raw| {
            let store: StoreData = pod_read_unaligned(raw);
            ok(&client, system, Request::new(13).arg(store));
            store.create_id
        })
        .collect();
    let index = |id: [u8; 16]| word::<i32>(&ok(&client, system, Request::new(11).arg(id).arg(0u8)));

    ok(&client, system, Request::new(12).arg(ids[0]).arg(2i32));
    assert_eq!([index(ids[0]), index(ids[1]), index(ids[2])], [2, 0, 1]);
    assert_eq!(
        call(&client, system, Request::new(12).arg(ids[0]).arg(2i32)).0,
        NOT_UPDATED
    );
    assert_eq!(
        call(&client, system, Request::new(12).arg(ids[0]).arg(3i32)).0,
        INVALID_ARGUMENT
    );
    assert_eq!(
        call(&client, system, Request::new(12).arg([1u8; 16]).arg(0i32)).0,
        NOT_FOUND
    );
}

#[test]
fn append_adds_a_char_info_under_a_new_create_id() {
    let client = seeded(SWITCH1_TITLE, 8);
    let user = database(&client, "mii:u");
    let info: CharInfo = pod_read_unaligned(&ok(&client, user, Request::new(7).arg(1i32)));
    ok(&client, user, Request::new(26).arg(info));
    assert_eq!(count(&client, user, DATABASE), 1);
    let (_, _, out) = get(&client, user, 4, DATABASE, CHAR_INFO, 1);
    let stored: CharInfo = pod_read_unaligned(&out);
    assert_ne!(stored.create_id, info.create_id);
    assert_eq!(bytes_of(&stored)[16..], bytes_of(&info)[16..]);
    assert_eq!(
        call(&client, user, Request::new(26).arg(CharInfo::zeroed())).0,
        NOT_FOUND
    );
}

#[test]
fn conversions_match_the_reference_bytes() {
    let client = seeded(SWITCH1_TITLE, 10);
    let user = database(&client, "mii:u");
    let info = ok(&client, user, Request::new(23).arg(VER3));
    assert_eq!(info[16..], VER3_CHAR_INFO[16..]);
    assert!(is_version_4(&info[..16]));
    let core = ok(&client, user, Request::new(25).arg(VER3_CHAR_INFO));
    assert_eq!(core, VER3_CORE_DATA);
    let info = ok(&client, user, Request::new(24).arg(VER3_CORE_DATA));
    assert_eq!(info[16..], VER3_CHAR_INFO[16..]);

    assert_eq!(
        call(&client, user, Request::new(23).arg([0u8; 0x60])).0,
        INVALID_CHAR_INFO
    );
    assert_eq!(
        call(&client, user, Request::new(24).arg([0u8; 0x30])).0,
        INVALID_CHAR_INFO
    );
    assert_eq!(
        call(&client, user, Request::new(25).arg([0u8; 0x58])).0,
        INVALID_CHAR_INFO
    );
    assert_eq!(
        call(&client, user, Request::new(21).arg([0u8; 0x58])).0,
        INVALID_CHAR_INFO
    );
}

#[test]
fn interface_version_one_checks_update_latest_input() {
    let client = seeded(SWITCH1_TITLE, 11);
    let user = database(&client, "mii:u");
    let invalid = Request::new(5).arg([0u8; 0x58]).arg(DATABASE);
    assert_eq!(call(&client, user, invalid.clone()).0, NOT_FOUND);
    ok(&client, user, Request::new(22).arg(1u32));
    assert_eq!(call(&client, user, invalid).0, INVALID_CHAR_INFO);
}

#[test]
fn test_mode_commands_are_refused() {
    let client = seeded(SWITCH1_TITLE, 12);
    let system = database(&client, "mii:e");
    for command in [15, 16, 17] {
        assert_eq!(
            call(&client, system, Request::new(command)).0,
            TEST_MODE_ONLY,
            "{command}"
        );
    }
}

#[test]
fn unhandled_commands_are_stubbed_only_for_switch2_titles() {
    for (title, expected) in [
        (SWITCH1_TITLE, UNKNOWN_COMMAND_ID),
        (SWITCH2_TITLE, ResultCode::SUCCESS),
    ] {
        let client = seeded(title, 13);
        let session = database(&client, "mii:u");
        assert_eq!(
            call(&client, session, Request::new(18)).0,
            expected,
            "Import"
        );
        let service = client.connect("mii:e").unwrap();
        assert_eq!(call(&client, service, Request::new(1)).0, expected);
    }
}
