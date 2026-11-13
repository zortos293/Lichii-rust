//! fsp-srv driven through the kernel's IPC path with the shared host-side client: real CMIF
//! requests, handles, guest buffers and the client's pid.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};
use lichii_common::settings::Settings;
use lichii_common::{ResultCode, VAddr};
use lichii_fs::romfs::{build_romfs, open_romfs};
use lichii_fs::{DirectoryExt, FileExt, FileRef, MemDir, MemFile};
use lichii_kernel::Kernel;
use lichii_kernel::ipc::{Handle, UNKNOWN_COMMAND_ID};
use lichii_kernel::testing::{self, IpcClient, Reply, Request};
use lichii_loader::program::{AddOnContent, ProgramImage};

use super::result::*;
use super::save::{RawSaveDataAttribute, SaveDataInfo};
use crate::ServiceContext;
use crate::testing::{SWITCH1_TITLE, SWITCH2_TITLE};

const SCRATCH_SIZE: usize = 2 << 20;

/// A guest buffer: address and size.
type Buffer = (VAddr, u64);

trait Buffers {
    fn x(self, buffer: Buffer) -> Self;
    fn a(self, buffer: Buffer) -> Self;
    fn b(self, buffer: Buffer) -> Self;
}

impl Buffers for Request {
    fn x(self, (address, size): Buffer) -> Self {
        self.buffer_x(address, size)
    }

    fn a(self, (address, size): Buffer) -> Self {
        self.buffer_a(address, size)
    }

    fn b(self, (address, size): Buffer) -> Self {
        self.buffer_b(address, size)
    }
}

trait Checked {
    fn ok(self) -> Self;
    fn get<T: Pod>(&self, offset: usize) -> T;
    fn handle(&self) -> Handle;
}

impl Checked for Reply {
    #[track_caller]
    fn ok(self) -> Self {
        assert_eq!(self.result, ResultCode::SUCCESS);
        self
    }

    #[track_caller]
    fn get<T: Pod>(&self, offset: usize) -> T {
        bytemuck::pod_read_unaligned(&self.data()[offset..offset + size_of::<T>()])
    }

    #[track_caller]
    fn handle(&self) -> Handle {
        assert_eq!(self.result, ResultCode::SUCCESS);
        self.move_handles[0]
    }
}

static NEXT_ROOT: AtomicU32 = AtomicU32::new(0);

/// A title's process with the fs services registered, an IPC client in it, 2 MiB of guest
/// scratch memory, and host NAND and SD card directories under `root`.
struct Guest {
    kernel: Kernel,
    client: IpcClient,
    _context: Arc<ServiceContext>,
    scratch: VAddr,
    root: PathBuf,
}

impl Guest {
    fn new(program_id: u64, configure: impl FnOnce(&mut ProgramImage)) -> Self {
        let root = std::env::temp_dir().join(format!(
            "lichii-fsp-{}-{}",
            std::process::id(),
            NEXT_ROOT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&root);
        let mut settings = Settings::default();
        settings.paths.nand_dir = Some(root.join("nand"));
        settings.paths.sdmc_dir = Some(root.join("sdmc"));
        settings.paths.keys_dir = Some(root.join("keys"));

        let mut image = testing::program_image(program_id);
        configure(&mut image);
        let (context, client) = crate::testing::context_for(image, settings);
        let kernel = context.kernel.clone();
        super::register(&context).unwrap();
        let scratch = client.alloc(&vec![0; SCRATCH_SIZE]);
        Self {
            kernel,
            client,
            _context: context,
            scratch,
            root,
        }
    }

    #[track_caller]
    fn call(&self, session: Handle, request: Request) -> Reply {
        self.client
            .send(session, &request)
            .expect("SendSyncRequest")
    }

    #[track_caller]
    fn service(&self, name: &str) -> Handle {
        self.client.connect(name).expect("service registered")
    }

    /// Closes a session the way `svcCloseHandle` does.
    #[track_caller]
    fn close(&self, handle: Handle) {
        let session = self.client.process().handles().write().remove(handle);
        assert!(session.is_some(), "handle {handle:#x} is not open");
    }

    /// `bytes` written to scratch memory at `offset`.
    fn put(&self, offset: u64, bytes: &[u8]) -> Buffer {
        let address = self.scratch + offset;
        self.client
            .process()
            .memory()
            .write_bytes(address, bytes)
            .unwrap();
        (address, bytes.len() as u64)
    }

    /// A NUL-terminated path in path slot `slot`, sized like `nn::fssrv::sf::Path`.
    fn path(&self, slot: u64, path: &str) -> Buffer {
        let mut raw = vec![0u8; 0x301];
        raw[..path.len()].copy_from_slice(path.as_bytes());
        self.put(0x1000 + slot * 0x400, &raw)
    }

    /// `size` bytes of `fill` at `offset` into scratch memory.
    fn buffer(&self, offset: u64, size: u64, fill: u8) -> Buffer {
        let address = self.scratch + offset;
        self.client
            .process()
            .memory()
            .fill(address, size, fill)
            .unwrap();
        (address, size)
    }

    fn read(&self, (address, size): Buffer) -> Vec<u8> {
        self.client.read(address, size as usize)
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        self.kernel.shutdown();
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

const AOC: u64 = SWITCH2_TITLE + 0x1001;
const USER: [u64; 2] = [0x8877_6655_4433_2211, 0x0102_0304_0506_0708];
const USER_DIR: &str = "01020304050607088877665544332211";

fn romfs() -> FileRef {
    let root = MemDir::new().with_bytes("top.bin", vec![7; 0x40]).with_dir(
        "dir",
        MemDir::new()
            .with_bytes("a.txt", b"hello romfs".to_vec())
            .with_dir("sub", MemDir::new()),
    );
    build_romfs(&root).unwrap()
}

fn aoc_romfs() -> FileRef {
    build_romfs(&MemDir::new().with_bytes("dlc.txt", b"extra".to_vec())).unwrap()
}

fn switch2_title() -> Guest {
    Guest::new(SWITCH2_TITLE, |image| {
        image.romfs = Some(romfs());
        image.add_ons.push(AddOnContent {
            title_id: AOC,
            version: 0,
            romfs: aoc_romfs(),
        });
    })
}

/// Reads a whole storage into guest memory at heap + 0x100000 and opens it as a RomFS.
fn read_storage(guest: &Guest, storage: Handle) -> FileRef {
    let size: u64 = guest.call(storage, Request::new(4)).ok().get(0);
    let buffer = guest.buffer(0x10_0000, size, 0);
    guest
        .call(
            storage,
            Request::new(0).arg(0i64).arg(size as i64).b(buffer),
        )
        .ok();
    MemFile::from_bytes(guest.read(buffer))
}

fn attribute(program_id: u64, save_type: u8, system_save_data_id: u64) -> RawSaveDataAttribute {
    RawSaveDataAttribute {
        program_id,
        user_id: USER,
        system_save_data_id,
        save_type,
        ..RawSaveDataAttribute::zeroed()
    }
}

#[test]
fn set_current_process_accepts_only_the_application() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    guest.call(fsp, Request::new(1).send_pid().arg(0u64)).ok();
    guest.call(fsp, Request::new(210).send_pid().arg(0u64)).ok();
    let other = guest.call(fsp, Request::new(1).arg(0x9999u64));
    assert_eq!(other.result, TARGET_NOT_FOUND);
}

#[test]
fn data_storage_of_the_current_process() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    let storage = guest.call(fsp, Request::new(200)).handle();
    let size: u64 = guest.call(storage, Request::new(4)).ok().get(0);
    assert_eq!(size, romfs().size());
    let bytes = read_storage(&guest, storage).read_all().unwrap();
    assert_eq!(bytes, romfs().read_all().unwrap());

    let buffer = guest.buffer(0x2000, 0x10, 0);
    let short = Request::new(0).arg(4i64).arg(0x100i64).b(buffer);
    guest.call(storage, short).ok();
    assert_eq!(guest.read(buffer), romfs().read_vec(4, 0x10).unwrap());
    let negative = Request::new(0).arg(0i64).arg(-1i64).b(buffer);
    assert_eq!(guest.call(storage, negative).result, INVALID_SIZE);
    let negative = Request::new(0).arg(-1i64).arg(1i64).b(buffer);
    assert_eq!(guest.call(storage, negative).result, INVALID_OFFSET);
    let write = Request::new(1).arg(0i64).arg(4i64).a(buffer);
    assert_eq!(
        guest.call(storage, write).result,
        UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE
    );
    assert_eq!(
        guest.call(storage, Request::new(3).arg(0i64)).result,
        UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE
    );
    guest.call(storage, Request::new(2)).ok();
    let query = guest
        .call(storage, Request::new(5).arg(3u32).arg(0i64).arg(0i64))
        .ok();
    assert_eq!(query.data(), [0; 0x40]);
    guest
        .call(storage, Request::new(5).arg(2u32).arg(0i64).arg(0i64))
        .ok();
    let unsupported = guest.call(storage, Request::new(5).arg(9u32).arg(0i64).arg(0i64));
    assert_eq!(unsupported.result, UNSUPPORTED_OPERATE_RANGE_FOR_STORAGE);

    let by_id = guest
        .call(fsp, Request::new(201).arg(SWITCH2_TITLE))
        .handle();
    assert_eq!(read_storage(&guest, by_id).size(), romfs().size());
    let aoc = guest.call(fsp, Request::new(201).arg(AOC)).handle();
    assert_eq!(read_storage(&guest, aoc).size(), aoc_romfs().size());
    let missing = guest.call(fsp, Request::new(201).arg(SWITCH1_TITLE));
    assert_eq!(missing.result, TARGET_NOT_FOUND);

    let index0 = guest.call(fsp, Request::new(205).arg(0u8)).handle();
    assert_eq!(read_storage(&guest, index0).size(), romfs().size());
    assert_eq!(guest.call(fsp, Request::new(205).arg(1u8)).result, UNKNOWN);
    assert_eq!(
        guest.call(fsp, Request::new(203).arg(0u8).arg(0u64)).result,
        TARGET_NOT_FOUND
    );
    let patch = Request::new(7).arg(4u8).arg(SWITCH2_TITLE);
    assert_eq!(guest.call(fsp, patch).result, TARGET_NOT_FOUND);
}

#[test]
fn a_title_without_romfs_has_no_data_storage() {
    let guest = Guest::new(SWITCH2_TITLE, |_| {});
    let fsp = guest.service("fsp-srv");
    assert_eq!(guest.call(fsp, Request::new(200)).result, UNKNOWN);
    assert_eq!(guest.call(fsp, Request::new(2)).result, UNKNOWN);
}

#[test]
fn data_storage_by_data_id_serves_add_ons_and_system_archives() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    let open = |data_id: u64| guest.call(fsp, Request::new(202).arg(4u8).arg(data_id));

    let aoc = open_romfs(&read_storage(&guest, open(AOC).handle())).unwrap();
    let dlc = aoc.open_file_path("dlc.txt").unwrap().unwrap();
    assert_eq!(dlc.read_all().unwrap(), b"extra");

    let version = open_romfs(&read_storage(&guest, open(0x0100_0000_0000_0809).handle())).unwrap();
    let file = version
        .open_file("file")
        .unwrap()
        .unwrap()
        .read_all()
        .unwrap();
    assert_eq!(file[..3], [23, 0, 0]);
    let zones = open_romfs(&read_storage(&guest, open(0x0100_0000_0000_080E).handle())).unwrap();
    assert!(zones.open_file_path("zoneinfo/UTC").unwrap().is_some());

    let nx2 = open_romfs(&read_storage(&guest, open(0x0400_0000_0000_0849).handle())).unwrap();
    assert!(nx2.entries().unwrap().is_empty());
    assert_eq!(open(0x0100_0000_0000_0800).result, UNKNOWN);
    assert_eq!(open(0x0100_0000_0000_0811).result, UNKNOWN);
}

#[test]
fn romfs_file_system_is_read_only() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    let fs = guest.call(fsp, Request::new(2)).handle();
    let entry_type = |path: &str| guest.call(fs, Request::new(7).x(guest.path(0, path)));
    assert_eq!(entry_type("/").ok().get::<u32>(0), 0);
    assert_eq!(entry_type("/dir").ok().get::<u32>(0), 0);
    assert_eq!(entry_type("/dir/a.txt").ok().get::<u32>(0), 1);
    assert_eq!(entry_type("\\dir\\a.txt").ok().get::<u32>(0), 1);
    assert_eq!(entry_type("/missing").result, PATH_NOT_FOUND);
    assert_eq!(entry_type("/top.bin/x").result, PATH_NOT_FOUND);

    let open = |mode: u32| guest.call(fs, Request::new(8).x(guest.path(0, "/dir/a.txt")).arg(mode));
    assert_eq!(open(0).result, INVALID_OPEN_MODE);
    assert_eq!(open(8 | 1).result, INVALID_OPEN_MODE);
    let file = open(1).handle();
    let buffer = guest.buffer(0x2000, 0x40, 0xff);
    let read = guest
        .call(
            file,
            Request::new(0).arg(0u32).arg(6i64).arg(0x40i64).b(buffer),
        )
        .ok();
    assert_eq!(read.get::<i64>(0), 5);
    assert_eq!(&guest.read(buffer)[..5], b"romfs");
    let write = Request::new(1).arg(0u32).arg(0i64).arg(1i64).a(buffer);
    assert_eq!(guest.call(file, write).result, WRITE_NOT_PERMITTED);

    let create = Request::new(0).x(guest.path(0, "/new")).arg(0i32).arg(0i64);
    assert_eq!(
        guest.call(fs, create).result,
        UNSUPPORTED_WRITE_FOR_READ_ONLY_FILE_SYSTEM
    );
    let dir = guest
        .call(fs, Request::new(9).x(guest.path(0, "/")).arg(3u32))
        .handle();
    assert_eq!(guest.call(dir, Request::new(1)).ok().get::<i64>(0), 2);
    let entries = guest.buffer(0x4000, 0x310 * 4, 0);
    let read = guest.call(dir, Request::new(0).b(entries)).ok();
    assert_eq!(read.get::<i64>(0), 2);
    let raw = guest.read(entries);
    assert_eq!(&raw[..4], b"dir\0");
    assert_eq!(raw[0x304], 0);
    assert_eq!(&raw[0x310..0x318], b"top.bin\0");
    assert_eq!(raw[0x310 + 0x304], 1);
    assert_eq!(
        i64::from_le_bytes(raw[0x310 + 0x308..0x620].try_into().unwrap()),
        0x40
    );

    let total = guest.call(fs, Request::new(12).x(guest.path(0, "/"))).ok();
    assert_eq!(total.get::<i64>(0), 0x6_8000_0000);
    let attribute = guest.call(fs, Request::new(16)).ok();
    assert_eq!(attribute.data().len(), 0xc0);
    assert_eq!(attribute.data()[..13], [1; 13]);
    assert_eq!(attribute.get::<i32>(0x28), 0x80);
    assert_eq!(attribute.get::<i32>(0x30), 0x300);
}

#[test]
fn sd_card_file_system() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    let fs = guest.call(fsp, Request::new(18)).handle();
    let sd = guest.root.join("sdmc");
    let call1 =
        |command: u32, path: &str| guest.call(fs, Request::new(command).x(guest.path(0, path)));
    let call2 = |command: u32, from: &str, to: &str| {
        guest.call(
            fs,
            Request::new(command)
                .x(guest.path(0, from))
                .x(guest.path(1, to)),
        )
    };
    let create = |path: &str, size: i64| {
        guest.call(
            fs,
            Request::new(0).x(guest.path(0, path)).arg(0i32).arg(size),
        )
    };
    let open =
        |path: &str, mode: u32| guest.call(fs, Request::new(8).x(guest.path(0, path)).arg(mode));

    call1(2, "/a/b/c").ok();
    assert!(sd.join("a/b/c").is_dir());
    assert_eq!(call1(2, "/a/b").result, PATH_ALREADY_EXISTS);
    create("/a/f.bin", 4).ok();
    assert_eq!(std::fs::metadata(sd.join("a/f.bin")).unwrap().len(), 4);
    assert_eq!(create("/a/f.bin", 4).result, PATH_ALREADY_EXISTS);
    assert_eq!(create("/a/b", 0).result, PATH_ALREADY_EXISTS);
    assert_eq!(create("/zz/f", 0).result, PATH_NOT_FOUND);
    assert_eq!(create("/a/neg", -1).result, OUT_OF_RANGE);
    assert_eq!(call1(7, "/../x").result, INVALID_PATH);

    let data = guest.put(0x2000, b"abcd");
    let file = open("/a/f.bin", 3).handle();
    let write = |file: Handle, option: u32, offset: i64, size: i64| {
        guest.call(
            file,
            Request::new(1).arg(option).arg(offset).arg(size).a(data),
        )
    };
    write(file, 0, 0, 4).ok();
    assert_eq!(std::fs::read(sd.join("a/f.bin")).unwrap(), b"abcd");
    assert_eq!(
        write(file, 0, 2, 4).result,
        FILE_EXTENSION_WITHOUT_OPEN_MODE_ALLOW_APPEND
    );
    assert_eq!(write(file, 0, -1, 1).result, OUT_OF_RANGE);
    let out = guest.buffer(0x3000, 0x10, 0);
    let read = |file: Handle, offset: i64, size: i64| {
        guest.call(file, Request::new(0).arg(0u32).arg(offset).arg(size).b(out))
    };
    assert_eq!(read(file, 1, 0x10).ok().get::<i64>(0), 3);
    assert_eq!(&guest.read(out)[..3], b"bcd");
    assert_eq!(read(file, 5, 1).result, OUT_OF_RANGE);
    assert_eq!(read(file, 0, 0).ok().get::<i64>(0), 0);
    guest.call(file, Request::new(3).arg(8i64)).ok();
    assert_eq!(guest.call(file, Request::new(4)).ok().get::<i64>(0), 8);
    assert_eq!(
        guest.call(file, Request::new(3).arg(-1i64)).result,
        OUT_OF_RANGE
    );
    guest.call(file, Request::new(2)).ok();
    let query = guest
        .call(file, Request::new(5).arg(3u32).arg(0i64).arg(8i64))
        .ok();
    assert_eq!(query.data(), [0; 0x40]);
    let invalid = guest.call(file, Request::new(5).arg(0u32).arg(0i64).arg(8i64));
    assert_eq!(invalid.result, UNSUPPORTED_OPERATE_RANGE_FOR_FILE);

    let read_only = open("/a/f.bin", 1).handle();
    assert_eq!(write(read_only, 0, 0, 1).result, WRITE_NOT_PERMITTED);
    assert_eq!(
        guest.call(read_only, Request::new(3).arg(1i64)).result,
        WRITE_NOT_PERMITTED
    );
    let append = open("/a/f.bin", 2 | 4).handle();
    assert_eq!(read(append, 0, 1).result, READ_NOT_PERMITTED);
    write(append, 1, 8, 3).ok();
    assert_eq!(
        std::fs::read(sd.join("a/f.bin")).unwrap(),
        b"abcd\0\0\0\0abc"
    );
    assert_eq!(open("/a/b", 1).result, PATH_NOT_FOUND);
    assert_eq!(open("/a/none", 1).result, PATH_NOT_FOUND);
    guest.call(fs, Request::new(10)).ok();

    let big: Vec<u8> = (0..0x15_0000u32).map(|i| (i * 7 + i / 251) as u8).collect();
    std::fs::write(sd.join("big.bin"), &big).unwrap();
    let big_file = open("/big.bin", 1).handle();
    let big_out = guest.buffer(0x2_0000, big.len() as u64, 0);
    let read = guest.call(
        big_file,
        Request::new(0)
            .arg(0u32)
            .arg(0i64)
            .arg(big.len() as i64)
            .b(big_out),
    );
    assert_eq!(read.ok().get::<i64>(0), big.len() as i64);
    assert_eq!(guest.read(big_out), big);
    guest.close(big_file);
    std::fs::remove_file(sd.join("big.bin")).unwrap();

    assert_eq!(call1(7, "/a").ok().get::<u32>(0), 0);
    assert_eq!(call1(7, "/a/f.bin").ok().get::<u32>(0), 1);
    std::fs::write(sd.join("a/.yuzu_save_size"), [0; 16]).unwrap();
    let dir = guest
        .call(fs, Request::new(9).x(guest.path(0, "/a")).arg(3u32))
        .handle();
    assert_eq!(guest.call(dir, Request::new(1)).ok().get::<i64>(0), 2);
    let one = guest.buffer(0x4000, 0x310, 0);
    let first = guest.call(dir, Request::new(0).b(one)).ok();
    assert_eq!(first.get::<i64>(0), 1);
    assert_eq!(&guest.read(one)[..2], b"b\0");
    guest.call(dir, Request::new(0).b(one)).ok();
    let raw = guest.read(one);
    assert_eq!(&raw[..6], b"f.bin\0");
    assert_eq!(raw[0x304], 1);
    assert_eq!(i64::from_le_bytes(raw[0x308..].try_into().unwrap()), 11);
    assert_eq!(
        guest.call(dir, Request::new(0).b(one)).ok().get::<i64>(0),
        0
    );
    assert_eq!(guest.call(dir, Request::new(1)).ok().get::<i64>(0), 0);
    let files = guest
        .call(
            fs,
            Request::new(9).x(guest.path(0, "/a")).arg(2u32 | 1 << 31),
        )
        .handle();
    assert_eq!(guest.call(files, Request::new(1)).ok().get::<i64>(0), 1);
    let bad_mode = Request::new(9).x(guest.path(0, "/a")).arg(0u32);
    assert_eq!(guest.call(fs, bad_mode).result, INVALID_OPEN_MODE);
    std::fs::remove_file(sd.join("a/.yuzu_save_size")).unwrap();

    let stamp = call1(14, "/a/f.bin").ok();
    assert!(stamp.get::<u64>(16) > 0);
    assert_eq!(stamp.get::<u64>(24), 0);
    assert_eq!(call1(14, "/a/none").result, PATH_NOT_FOUND);

    call2(5, "/a/f.bin", "/a/g.bin").ok();
    assert!(sd.join("a/g.bin").is_file() && !sd.join("a/f.bin").exists());
    call2(5, "/a/g.bin", "/a/b/h.bin").ok();
    assert_eq!(
        std::fs::read(sd.join("a/b/h.bin")).unwrap(),
        b"abcd\0\0\0\0abc"
    );
    assert!(!sd.join("a/g.bin").exists());
    create("/a/x", 0).ok();
    assert_eq!(call2(5, "/a/x", "/a/b/h.bin").result, PATH_ALREADY_EXISTS);
    assert_eq!(call2(5, "/a/none", "/a/y").result, PATH_NOT_FOUND);
    assert_eq!(call2(5, "/a/b", "/a/y").result, PATH_NOT_FOUND);
    for handle in [file, read_only, append] {
        guest.close(handle);
    }
    call2(6, "/a/b", "/a/d").ok();
    assert!(sd.join("a/d/c").is_dir());
    call2(6, "/a/d", "/e").ok();
    assert_eq!(std::fs::read(sd.join("e/h.bin")).unwrap().len(), 11);
    assert!(sd.join("e/c").is_dir() && !sd.join("a/d").exists());
    assert_eq!(call2(6, "/a/x", "/f").result, PATH_NOT_FOUND);

    assert_eq!(call1(3, "/e").result, DIRECTORY_NOT_EMPTY);
    call1(3, "/e/c").ok();
    assert_eq!(call1(3, "/e/c").result, PATH_NOT_FOUND);
    call1(4, "/e").ok();
    assert!(!sd.join("e").exists());
    assert_eq!(call1(4, "/e").result, PATH_NOT_FOUND);
    assert_eq!(call1(1, "/a").result, PATH_NOT_FOUND);
    call1(1, "/a/x").ok();
    assert_eq!(call1(1, "/a/x").result, PATH_NOT_FOUND);
    call1(1, "").ok();
    call1(2, "/a/k/l").ok();
    create("/a/k/m", 3).ok();
    call1(13, "/a").ok();
    assert!(sd.join("a").is_dir());
    assert_eq!(std::fs::read_dir(sd.join("a")).unwrap().count(), 0);
    assert_eq!(call1(13, "/none").result, PATH_NOT_FOUND);

    create("/used", 0x1000).ok();
    let total: i64 = call1(12, "/").ok().get(0);
    let free: i64 = call1(11, "/").ok().get(0);
    assert_eq!(total, 4 << 30);
    assert_eq!(free, total - 0x1000);

    let query = |id: i32, path: &str| {
        guest.call(
            fs,
            Request::new(15)
                .b(guest.buffer(0x5000, 1, 0xff))
                .a(guest.buffer(0x5100, 0, 0))
                .x(guest.path(0, path))
                .arg(id),
        )
    };
    query(0, "/used").ok();
    assert_eq!(query(0, "/none").result, PATH_NOT_FOUND);
    query(1, "/used").ok();
    query(2, "/").ok();
    assert_eq!(guest.read((guest.scratch + 0x5000, 1)), [0]);
    assert_eq!(query(3, "/used").result, NOT_IMPLEMENTED);
}

#[test]
fn renames_keep_open_files_and_refuse_descendants() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    let fs = guest.call(fsp, Request::new(18)).handle();
    let sd = guest.root.join("sdmc");
    let path1 =
        |command: u32, path: &str| guest.call(fs, Request::new(command).x(guest.path(0, path)));
    let path2 = |command: u32, from: &str, to: &str| {
        guest.call(
            fs,
            Request::new(command)
                .x(guest.path(0, from))
                .x(guest.path(1, to)),
        )
    };
    path1(2, "/a").ok();
    path1(2, "/b").ok();
    guest
        .call(
            fs,
            Request::new(0)
                .x(guest.path(0, "/a/save.bin"))
                .arg(0i32)
                .arg(4i64),
        )
        .ok();
    let file = guest
        .call(
            fs,
            Request::new(8).x(guest.path(0, "/a/save.bin")).arg(3u32),
        )
        .handle();
    path2(5, "/a/save.bin", "/b/save.bin").ok();
    let data = guest.put(0x2000, b"wxyz");
    guest
        .call(file, Request::new(1).arg(0u32).arg(0i64).arg(4i64).a(data))
        .ok();
    assert_eq!(std::fs::read(sd.join("b/save.bin")).unwrap(), b"wxyz");
    assert!(!sd.join("a/save.bin").exists());

    assert_eq!(path2(6, "/a", "/a/inner").result, DIRECTORY_NOT_RENAMABLE);
    assert!(!sd.join("a/inner").exists());
    path2(6, "/a", "/b/moved").ok();
    assert!(sd.join("b/moved").is_dir() && !sd.join("a").exists());
}

#[test]
fn save_data_file_systems() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    let nand = guest.root.join("nand");
    let system = attribute(0, 0, 0x8000_0000_0000_0010);
    let create_system = Request::new(23).arg(system).arg([0u8; 0x40]);
    guest.call(fsp, create_system).ok();
    let system_dir = nand.join(format!("system/save/8000000000000010/{USER_DIR}"));
    assert!(system_dir.is_dir());
    let open_system =
        |attr: RawSaveDataAttribute| guest.call(fsp, Request::new(52).arg(0u8).arg(attr));
    let system_fs = open_system(system).handle();
    let create = Request::new(0)
        .x(guest.path(0, "/config"))
        .arg(0i32)
        .arg(8i64);
    guest.call(system_fs, create).ok();
    assert!(system_dir.join("config").is_file());
    let size: i64 = guest
        .call(system_fs, Request::new(12).x(guest.path(0, "/")))
        .ok()
        .get(0);
    assert_eq!(size, 0xA000_0000);
    let stamp = guest
        .call(system_fs, Request::new(14).x(guest.path(0, "/config")))
        .ok();
    assert!(stamp.get::<u64>(16) > 0);
    assert_eq!(open_system(attribute(0, 0, 0)).result, INVALID_ARGUMENT);
    assert_eq!(open_system(attribute(0, 0, 0x99)).result, TARGET_NOT_FOUND);

    let account = attribute(0, 1, 0);
    let creation = [0u8; 0x40];
    guest
        .call(
            fsp,
            Request::new(22).arg(account).arg(creation).arg([0u8; 0x10]),
        )
        .ok();
    let account_dir = nand.join(format!(
        "user/save/0000000000000000/{USER_DIR}/{SWITCH2_TITLE:016X}"
    ));
    assert!(account_dir.is_dir());
    let open = |command: u32, space: u8, attr: RawSaveDataAttribute| {
        guest.call(fsp, Request::new(command).arg(space).arg(attr))
    };
    let account_fs = open(51, 1, account).handle();
    let free: i64 = guest
        .call(account_fs, Request::new(11).x(guest.path(0, "/")))
        .ok()
        .get(0);
    assert_eq!(free, 0x6_8000_0000 - 0x10_0000);
    open(53, 1, account).handle();
    let other = attribute(SWITCH2_TITLE + 0x2000, 1, 0);
    assert_eq!(open(51, 1, other).result, TARGET_NOT_FOUND);

    let temporary = attribute(SWITCH2_TITLE, 4, 0);
    let temporary_fs = open(51, 3, temporary).handle();
    let create = Request::new(0)
        .x(guest.path(0, "/scratch"))
        .arg(0i32)
        .arg(1i64);
    guest.call(temporary_fs, create).ok();
    let cache = attribute(SWITCH2_TITLE, 5, 0);
    open(51, 1, cache).handle();
    assert!(
        nand.join(format!("user/save/cache/{SWITCH2_TITLE:016X}"))
            .is_dir()
    );
    guest.call(fsp, Request::new(1003)).ok();
    let late_cache = attribute(SWITCH2_TITLE + 0x3000, 5, 0);
    assert_eq!(open(51, 1, late_cache).result, TARGET_NOT_FOUND);

    let mut info2 = [0u8; 0xa0];
    info2[8..0x48].copy_from_slice(bytemuck::bytes_of(&attribute(0, 0, 0x20)));
    info2[0x6c] = 0;
    let info2: [u64; 20] = bytemuck::cast(info2);
    guest.call(fsp, Request::new(37).arg(info2)).ok();
    assert!(
        nand.join(format!("system/save/0000000000000020/{USER_DIR}"))
            .is_dir()
    );

    let read_info = |reader: Handle| -> Vec<SaveDataInfo> {
        let buffer = guest.buffer(0x8000, 0x60 * 8, 0);
        let count: u64 = guest.call(reader, Request::new(0).b(buffer)).ok().get(0);
        let raw = guest.read(buffer);
        raw[..count as usize * 0x60]
            .chunks(0x60)
            .map(bytemuck::pod_read_unaligned)
            .collect()
    };
    let user = read_info(guest.call(fsp, Request::new(61).arg(1u8)).handle());
    let titles: Vec<_> = user.iter().map(|i| (i.save_type, i.title_id)).collect();
    assert_eq!(titles, [(1, SWITCH2_TITLE)]);
    assert_eq!(user[0].user_id, bytemuck::cast::<[u64; 2], [u8; 16]>(USER));
    assert_eq!(user[0].space, 1);
    let systems = read_info(guest.call(fsp, Request::new(61).arg(0u8)).handle());
    let mut ids: Vec<_> = systems.iter().map(|i| (i.save_type, i.save_id)).collect();
    ids.sort();
    assert_eq!(ids, [(0, 0x20), (0, 0x8000_0000_0000_0010)]);
    assert_eq!(systems.iter().map(|i| i.save_image_size).max(), Some(8));
    let temporary = read_info(guest.call(fsp, Request::new(62)).handle());
    assert_eq!(temporary.len(), 1);
    assert_eq!(
        (temporary[0].save_type, temporary[0].title_id),
        (4, SWITCH2_TITLE)
    );
    let all = read_info(guest.call(fsp, Request::new(60)).handle());
    assert_eq!(all.len(), 3);
    let reader = guest.call(fsp, Request::new(61).arg(1u8)).handle();
    let empty = guest.buffer(0x8000, 0, 0);
    assert_eq!(
        guest
            .call(reader, Request::new(0).b(empty))
            .ok()
            .get::<u64>(0),
        0
    );

    let extra = guest.buffer(0x9000, 0x200, 0xff);
    guest.call(fsp, Request::new(58).arg(1u64).b(extra)).ok();
    assert_eq!(guest.read(extra), [0; 0x200]);
    let extra = guest.buffer(0x9000, 0x200, 0xff);
    guest
        .call(fsp, Request::new(57).arg(1u8).arg(1u64).b(extra))
        .ok();
    assert_eq!(guest.read(extra), [0; 0x200]);
    let extra = guest.buffer(0x9000, 0x200, 0xff);
    guest
        .call(fsp, Request::new(69).arg(1u8).arg(account).b(extra))
        .ok();
    assert_eq!(guest.read(extra), [0; 0x200]);
    let extra = guest.buffer(0x9000, 0x200, 0xff);
    let mask = guest.buffer(0xa000, 0x200, 0xff);
    guest
        .call(fsp, Request::new(71).arg(1u8).arg(account).a(mask).b(extra))
        .ok();
    assert_eq!(guest.read(extra), [0; 0x200]);
    guest
        .call(fsp, Request::new(59).arg(1u8).arg(1u64).a(extra))
        .ok();
    guest
        .call(fsp, Request::new(70).arg(1u8).arg(account).a(extra).a(mask))
        .ok();
    let find = Request::new(67).arg(1u8).b(extra);
    assert_eq!(guest.call(fsp, find).result, TARGET_NOT_FOUND);
    guest
        .call(fsp, Request::new(32).arg(1u8).arg(1u64).arg(0i64).arg(0i64))
        .ok();
    let cache_size = guest.call(fsp, Request::new(34).arg(0i32)).ok();
    assert_eq!(cache_size.get::<[i64; 2]>(0), [0, 0]);
    let prohibiter = guest.call(fsp, Request::new(83).arg(1u64)).handle();
    guest.call(prohibiter, Request::new(0)).ok();
    let commit = guest.call(fsp, Request::new(1200)).handle();
    guest.call(commit, Request::new(1)).ok();
    guest.call(commit, Request::new(2)).ok();
}

#[test]
fn proxy_settings_and_stubs() {
    let guest = switch2_title();
    let fsp = guest.service("fsp-srv");
    assert_eq!(guest.call(fsp, Request::new(27)).ok().get::<u8>(0), 1);
    assert_eq!(guest.call(fsp, Request::new(631)).ok().get::<u8>(0), 1);
    assert_eq!(guest.call(fsp, Request::new(1005)).ok().get::<u32>(0), 0);
    guest.call(fsp, Request::new(1004).arg(2u32)).ok();
    assert_eq!(guest.call(fsp, Request::new(1005)).ok().get::<u32>(0), 2);
    let log = guest.put(0x2000, b"FS_ACCESS: { start: 1 }\0junk");
    guest.call(fsp, Request::new(1006).a(log)).ok();
    let index = guest.call(fsp, Request::new(1011)).ok();
    assert_eq!(index.get::<[u32; 2]>(0), [2, 0]);
    guest.call(fsp, Request::new(1016)).ok();
    guest.call(fsp, Request::new(606)).ok();
    for name in ["fsp-ldr", "fsp-pr"] {
        let session = guest.service(name);
        guest.call(session, Request::new(0)).ok();
    }

    let switch1 = Guest::new(SWITCH1_TITLE, |_| {});
    let fsp = switch1.service("fsp-srv");
    assert_eq!(
        switch1.call(fsp, Request::new(606)).result,
        UNKNOWN_COMMAND_ID
    );
    for name in ["fsp-ldr", "fsp-pr"] {
        let session = switch1.service(name);
        assert_eq!(
            switch1.call(session, Request::new(0)).result,
            UNKNOWN_COMMAND_ID
        );
    }
}

#[test]
fn switch1_storage_reads_are_paced_like_a_game_card() {
    let guest = Guest::new(SWITCH1_TITLE, |image| {
        image.romfs = Some(MemFile::from_bytes(vec![0x5a; 0x2_0000]));
    });
    let fsp = guest.service("fsp-srv");
    let storage = guest.call(fsp, Request::new(200)).handle();
    let buffer = guest.buffer(0x2_0000, 0x1_0000, 0);
    let read = || {
        guest
            .call(
                storage,
                Request::new(0).arg(0i64).arg(0x1_0000i64).b(buffer),
            )
            .ok()
    };
    let start = std::time::Instant::now();
    read();
    read();
    read();
    assert!(start.elapsed() >= std::time::Duration::from_micros(3200));
    assert_eq!(guest.read(buffer), vec![0x5a; 0x1_0000]);
}
