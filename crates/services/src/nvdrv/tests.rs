use std::sync::Arc;
use std::time::Duration;

use lichii_common::settings::Settings;
use lichii_common::{PAGE_SIZE, ResultCode, VAddr, align_up};
use lichii_gpu::pushbuffer::{CommandHeader, GpEntry, SubmissionMode};
use lichii_kernel::ipc::Handle;
use lichii_kernel::object::KObject;
use lichii_kernel::testing::{IpcClient, Request};

use super::*;
use crate::testing::{SWITCH1_TITLE, SWITCH2_TITLE, context};

const CURRENT_PROCESS: Handle = 0xffff_8001;
const SYNCPOINT_A: u32 = 0x1c;

/// An `_IOWR` ioctl number.
const fn iowr(group: u8, cmd: u8, size: usize) -> u32 {
    0xC000_0000 | (size as u32) << 16 | (group as u32) << 8 | cmd as u32
}

/// Little-endian argument bytes.
#[derive(Default)]
struct Bytes(Vec<u8>);

impl Bytes {
    fn u32(mut self, value: u32) -> Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    fn u64(mut self, value: u64) -> Self {
        self.0.extend_from_slice(&value.to_le_bytes());
        self
    }

    fn zeroes(mut self, count: usize) -> Self {
        self.0.resize(self.0.len() + count, 0);
        self
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

struct Reply {
    result: u32,
    output: Vec<u8>,
    inline_output: Vec<u8>,
}

struct Nv {
    nvdrv: Arc<Nvdrv>,
    context: Arc<ServiceContext>,
    client: IpcClient,
    session: Handle,
}

impl Nv {
    fn new(program_id: u64, service: &str) -> Self {
        let (context, client) = context(program_id, Settings::default());
        let nvdrv = register(&context).unwrap();
        let session = if service == "nx2:nvsrv" {
            let thread = context.service_thread("nvdrv").unwrap();
            let handler = nvdrv.nvsrv_session(&context, context.process.clone());
            client.connect_handler(&thread, handler)
        } else {
            client.connect(service).expect("registered")
        };
        let nv = Self {
            nvdrv,
            context,
            client,
            session,
        };
        if service != "nx2:nvsrv" {
            nv.initialize(nv.session);
        }
        nv
    }

    fn initialize(&self, session: Handle) {
        let request = Request::new(3)
            .arg(0x30_0000u32)
            .copy_handle(CURRENT_PROCESS);
        let mut reply = self.client.send(session, &request).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        assert_eq!(reply.pop::<u32>(), 0);
    }

    fn open_on(&self, session: Handle, path: &str) -> (i32, u32) {
        let mut name = path.as_bytes().to_vec();
        name.push(0);
        let buffer = self.client.alloc(&name);
        let request = Request::new(0).buffer_a(buffer, name.len() as u64);
        let mut reply = self.client.send(session, &request).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        (reply.pop::<i32>(), reply.pop::<u32>())
    }

    fn open(&self, path: &str) -> i32 {
        let (fd, result) = self.open_on(self.session, path);
        assert_eq!(result, 0, "open {path}");
        assert!(fd > 0);
        fd
    }

    fn call(&self, request: Request, outputs: &[(VAddr, usize)]) -> Reply {
        self.call_on(self.session, request, outputs)
    }

    fn call_on(&self, session: Handle, request: Request, outputs: &[(VAddr, usize)]) -> Reply {
        let mut reply = self.client.send(session, &request).unwrap();
        assert_eq!(reply.result, ResultCode::SUCCESS);
        let result = reply.pop::<u32>();
        let mut read = outputs.iter().map(|&(at, len)| self.client.read(at, len));
        Reply {
            result,
            output: read.next().unwrap_or_default(),
            inline_output: read.next().unwrap_or_default(),
        }
    }

    fn ioctl(&self, fd: i32, ioctl: u32, input: &[u8], out_len: usize) -> Reply {
        self.ioctl_on(self.session, fd, ioctl, input, out_len)
    }

    fn ioctl_on(
        &self,
        session: Handle,
        fd: i32,
        ioctl: u32,
        input: &[u8],
        out_len: usize,
    ) -> Reply {
        let input_at = self.client.alloc(input);
        let output_at = self.client.alloc(&vec![0xAA; out_len]);
        let request = Request::new(1)
            .arg(fd)
            .arg(ioctl)
            .buffer_a(input_at, input.len() as u64)
            .buffer_b(output_at, out_len as u64);
        self.call_on(session, request, &[(output_at, out_len)])
    }

    fn ioctl2(&self, fd: i32, ioctl: u32, input: &[u8], inline: &[u8], out_len: usize) -> Reply {
        let input_at = self.client.alloc(input);
        let inline_at = self.client.alloc(inline);
        let output_at = self.client.alloc(&vec![0xAA; out_len]);
        let request = Request::new(11)
            .arg(fd)
            .arg(ioctl)
            .buffer_a(input_at, input.len() as u64)
            .buffer_a(inline_at, inline.len() as u64)
            .buffer_b(output_at, out_len as u64);
        self.call(request, &[(output_at, out_len)])
    }

    fn ioctl3(
        &self,
        fd: i32,
        ioctl: u32,
        input: &[u8],
        out_len: usize,
        inline_len: usize,
    ) -> Reply {
        let input_at = self.client.alloc(input);
        let output_at = self.client.alloc(&vec![0xAA; out_len]);
        let inline_at = self.client.alloc(&vec![0xAA; inline_len]);
        let request = Request::new(12)
            .arg(fd)
            .arg(ioctl)
            .buffer_a(input_at, input.len() as u64)
            .buffer_b(output_at, out_len as u64)
            .buffer_b(inline_at, inline_len as u64);
        self.call(request, &[(output_at, out_len), (inline_at, inline_len)])
    }

    fn close(&self, fd: i32) -> u32 {
        self.call(Request::new(2).arg(fd), &[]).result
    }

    /// Page-aligned guest memory holding `data`.
    fn guest_pages(&self, data: &[u8]) -> VAddr {
        let size = align_up(data.len() as u64, PAGE_SIZE);
        let raw = self.client.alloc(&vec![0; (size + PAGE_SIZE) as usize]);
        let address = align_up(raw, PAGE_SIZE);
        self.client
            .process()
            .memory()
            .write_bytes(address, data)
            .unwrap();
        address
    }

    /// An allocated nvmap handle over `data` in guest memory.
    fn nvmap_buffer(&self, nvmap: i32, data: &[u8], align: u32) -> (u32, VAddr) {
        let address = self.guest_pages(data);
        let size = data.len() as u32;
        let reply = self.ioctl(
            nvmap,
            iowr(1, 0x1, 8),
            &Bytes::default().u32(size).u32(0).0,
            8,
        );
        assert_eq!(reply.result, 0);
        let handle = u32_at(&reply.output, 4);
        let alloc = Bytes::default()
            .u32(handle)
            .u32(0)
            .u32(0)
            .u32(align)
            .zeroes(8)
            .u64(address);
        assert_eq!(self.ioctl(nvmap, iowr(1, 0x4, 32), &alloc.0, 32).result, 0);
        (handle, address)
    }

    /// An initialized address space with default (Switch 1) ranges.
    fn address_space(&self) -> i32 {
        let as_fd = self.open("/dev/nvhost-as-gpu");
        let init = Bytes::default().u32(1).u32(0).u32(0x10000).zeroes(28);
        assert_eq!(
            self.ioctl(as_fd, iowr(b'A', 0x9, 40), &init.0, 40).result,
            0
        );
        as_fd
    }

    /// Maps nvmap `handle` anywhere; returns the GPU address.
    fn map(&self, as_fd: i32, handle: u32) -> u64 {
        let map = Bytes::default()
            .u32(0)
            .u32(0)
            .u32(handle)
            .u32(0)
            .u64(0)
            .u64(0)
            .u64(0);
        let reply = self.ioctl(as_fd, iowr(b'A', 0x6, 40), &map.0, 40);
        assert_eq!(reply.result, 0);
        u64_at(&reply.output, 32)
    }

    /// A channel bound to `as_fd` with a GPFIFO; returns its descriptor and syncpoint.
    fn channel(&self, as_fd: i32) -> (i32, u32) {
        let gpu = self.open("/dev/nvhost-gpu");
        let bind = self.ioctl(as_fd, iowr(b'A', 0x1, 4), &gpu.to_le_bytes(), 4);
        assert_eq!(bind.result, 0);
        let alloc = self.ioctl(
            gpu,
            iowr(b'H', 0x1a, 32),
            &Bytes::default().u32(0x400).zeroes(28).0,
            32,
        );
        assert_eq!(alloc.result, 0);
        (gpu, u32_at(&alloc.output, 12))
    }

    fn submit(&self, gpu: i32, entries: &[GpEntry], flags: u32, fence: (u32, u32)) -> Reply {
        let mut inline = Bytes::default();
        for entry in entries {
            inline = inline.u64(entry.0);
        }
        let params = Bytes::default()
            .u64(0)
            .u32(entries.len() as u32)
            .u32(flags)
            .u32(fence.0)
            .u32(fence.1);
        self.ioctl2(gpu, iowr(b'H', 0x1b, 24), &params.0, &inline.0, 24)
    }

    /// MapBufferEx; returns the result and the GPU address.
    fn map_ex(
        &self,
        as_fd: i32,
        flags: u32,
        handle: u32,
        buffer_offset: i64,
        mapping_size: u64,
        offset: u64,
    ) -> (u32, u64) {
        let map = Bytes::default()
            .u32(flags)
            .u32(0)
            .u32(handle)
            .u32(0)
            .u64(buffer_offset as u64)
            .u64(mapping_size)
            .u64(offset);
        let reply = self.ioctl(as_fd, iowr(b'A', 0x6, 40), &map.0, 40);
        (reply.result, u64_at(&reply.output, 32))
    }

    /// AllocSpace of `pages` pages of `page_size`; returns the GPU address.
    fn alloc_space(&self, as_fd: i32, pages: u32, page_size: u32, flags: u32) -> u64 {
        let input = Bytes::default()
            .u32(pages)
            .u32(page_size)
            .u32(flags)
            .u32(0)
            .u64(0);
        let reply = self.ioctl(as_fd, iowr(b'A', 0x2, 24), &input.0, 24);
        assert_eq!(reply.result, 0);
        u64_at(&reply.output, 16)
    }

    fn free_space(&self, as_fd: i32, offset: u64, pages: u32, page_size: u32) -> u32 {
        let input = Bytes::default().u64(offset).u32(pages).u32(page_size);
        self.ioctl(as_fd, iowr(b'A', 0x3, 16), &input.0, 16).result
    }

    /// Remaps one 64 KiB-page range of a sparse allocation.
    fn remap(&self, as_fd: i32, handle: u32, handle_offset: u32, va: u64, pages: u32) -> u32 {
        let input = Bytes::default()
            .u32(0)
            .u32(handle)
            .u32(handle_offset)
            .u32((va >> 16) as u32)
            .u32(pages);
        self.ioctl(as_fd, iowr(b'A', 0x14, 20), &input.0, 20).result
    }

    fn syncpoints(&self) -> &lichii_gpu::host1x::Syncpoints {
        &self.context.gpu.host1x().syncpoints
    }

    fn event_signaled(&self, handle: Handle) -> bool {
        match self.client.object(handle) {
            Some(KObject::ReadableEvent(event)) => event.is_signaled(),
            _ => panic!("handle {handle:#x} is not a readable event"),
        }
    }
}

#[test]
fn open_ioctl_close() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv:a");
    let fd = nv.open("/dev/nvhost-ctrl-gpu");
    let reply = nv.ioctl(fd, iowr(b'G', 0x1, 4), &[0; 4], 4);
    assert_eq!((reply.result, u32_at(&reply.output, 0)), (0, 1));
    assert_eq!(nv.close(fd), 0);
    assert_eq!(
        nv.ioctl(fd, iowr(b'G', 0x1, 4), &[0; 4], 4).result,
        NvResult::NotImplemented as u32
    );
    assert_eq!(nv.close(fd), NvResult::NotImplemented as u32);
    assert_eq!(nv.close(-1), NvResult::InvalidState as u32);
    assert_eq!(
        nv.open_on(nv.session, "/dev/nope"),
        (-1, NvResult::FileOperationFailed as u32)
    );
    assert_eq!(
        nv.open_on(nv.session, "/dev/nvhost-prof-gpu").1,
        NvResult::NotSupported as u32
    );
    assert_eq!(nv.open("/dev/nvmap"), fd, "descriptors are reused");

    let fresh = nv.client.connect("nvdrv").unwrap();
    assert_eq!(
        nv.open_on(fresh, "/dev/nvmap"),
        (0, NvResult::NotInitialized as u32)
    );
    nv.initialize(fresh);
    assert_eq!(nv.open_on(fresh, "/dev/nvmap"), (1, 0));
}

#[test]
fn nvsrv_needs_no_initialize_and_serves_switch2_nodes() {
    let nv = Nv::new(SWITCH2_TITLE, "nx2:nvsrv");
    let tsg = nv.open("/dev/nvhost-tsg-gpu");
    let input = Bytes::default().u32(7).u32(9).0;
    let reply = nv.ioctl(tsg, iowr(b'T', 0x3, 8), &input, 12);
    assert_eq!(reply.result, 0);
    assert_eq!(reply.output, [7, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0]);
    assert!(nv.open("/dev/nvhost-prof-gpu") > 0);
}

#[test]
fn nvmap_handles_pin_guest_memory_into_device_space() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let data: Vec<u8> = (0..0x3000u32).map(|i| i as u8).collect();
    let device = &nv.context.gpu.host1x().memory;
    let allocated = nv.context.gpu.host1x().allocator.allocated_bytes();
    let (handle, _) = nv.nvmap_buffer(nvmap, &data, 0x1000);

    let param = |kind| {
        let reply = nv.ioctl(
            nvmap,
            iowr(1, 0x9, 12),
            &Bytes::default().u32(handle).u32(kind).u32(0).0,
            12,
        );
        assert_eq!(reply.result, 0);
        u32_at(&reply.output, 8)
    };
    assert_eq!(
        (param(1), param(2), param(4)),
        (0x3000, 0x1000, 0x4000_0000)
    );
    let reply = nv.ioctl(
        nvmap,
        iowr(1, 0xe, 8),
        &Bytes::default().u32(0).u32(handle).0,
        8,
    );
    assert_eq!((reply.result, u32_at(&reply.output, 0)), (0, handle));
    let reply = nv.ioctl(
        nvmap,
        iowr(1, 0x3, 8),
        &Bytes::default().u32(handle).u32(0).0,
        8,
    );
    assert_eq!((reply.result, u32_at(&reply.output, 4)), (0, handle));

    assert_eq!(
        nv.nvdrv.nvmap.device_address(handle),
        0,
        "pinned on first use"
    );
    let as_fd = nv.address_space();
    let gpu_va = nv.map(as_fd, handle);
    let daddr = nv.nvdrv.nvmap.device_address(handle);
    assert_ne!(daddr, 0);
    let mut seen = vec![0; data.len()];
    device.read_bytes(daddr, &mut seen).unwrap();
    assert_eq!(seen, data);
    assert!(
        gpu_va < 1 << 34,
        "4 KiB-aligned handles map with small pages"
    );

    let unmap = nv.ioctl(as_fd, iowr(b'A', 0x5, 8), &gpu_va.to_le_bytes(), 8);
    assert_eq!(unmap.result, 0);
    let free = |expect_address| {
        let reply = nv.ioctl(
            nvmap,
            iowr(1, 0x5, 24),
            &Bytes::default().u32(handle).zeroes(20).0,
            24,
        );
        assert_eq!(reply.result, 0);
        assert_eq!(u64_at(&reply.output, 8) != 0, expect_address);
    };
    free(true);
    assert_ne!(
        nv.nvdrv.nvmap.device_address(handle),
        0,
        "one reference left"
    );
    free(true);
    assert_eq!(nv.nvdrv.nvmap.device_address(handle), 0);
    assert_eq!(
        nv.context.gpu.host1x().allocator.allocated_bytes(),
        allocated
    );
    let reply = nv.ioctl(
        nvmap,
        iowr(1, 0xe, 8),
        &Bytes::default().u32(0).u32(handle).0,
        8,
    );
    assert_eq!(reply.result, NvResult::AccessDenied as u32);
}

#[test]
fn a_closing_session_drops_every_nvmap_reference_it_holds() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let (shared, _) = nv.nvmap_buffer(nvmap, &[1; 0x1000], 0x1000);

    let other = nv.client.connect("nvdrv").unwrap();
    nv.initialize(other);
    let (other_nvmap, _) = nv.open_on(other, "/dev/nvmap");
    let create = nv.ioctl_on(
        other,
        other_nvmap,
        iowr(1, 0x1, 8),
        &Bytes::default().u32(0x2000).u32(0).0,
        8,
    );
    assert_eq!(create.result, 0);
    let unallocated = u32_at(&create.output, 4);
    for _ in 0..2 {
        let from_id = nv.ioctl_on(
            other,
            other_nvmap,
            iowr(1, 0x3, 8),
            &Bytes::default().u32(shared).u32(0).0,
            8,
        );
        assert_eq!(from_id.result, 0);
    }
    assert_eq!(nv.nvdrv.nvmap.size(unallocated), Some(0x2000));

    let session = nv.client.process().handles().write().remove(other);
    drop(session);
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while nv.nvdrv.nvmap.size(unallocated).is_some() {
        assert!(
            std::time::Instant::now() < deadline,
            "the session never released its handles"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        nv.nvdrv.nvmap.size(shared),
        Some(0x1000),
        "this session's reference stays"
    );

    let free = nv.ioctl(
        nvmap,
        iowr(1, 0x5, 24),
        &Bytes::default().u32(shared).zeroes(20).0,
        24,
    );
    assert_eq!(free.result, 0);
    assert_eq!(
        nv.nvdrv.nvmap.size(shared),
        None,
        "the last reference is gone"
    );
}

#[test]
fn a_session_frees_only_the_nvmap_references_it_holds() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let (handle, _) = nv.nvmap_buffer(nvmap, &[1; 0x1000], 0x1000);
    let other = nv.client.connect("nvdrv").unwrap();
    nv.initialize(other);
    let (other_nvmap, _) = nv.open_on(other, "/dev/nvmap");
    let from_id = nv.ioctl_on(
        other,
        other_nvmap,
        iowr(1, 0x3, 8),
        &Bytes::default().u32(handle).u32(0).0,
        8,
    );
    assert_eq!(from_id.result, 0);

    for _ in 0..2 {
        let free = nv.ioctl_on(
            other,
            other_nvmap,
            iowr(1, 0x5, 24),
            &Bytes::default().u32(handle).zeroes(20).0,
            24,
        );
        assert_eq!(free.result, 0);
    }
    assert_eq!(
        nv.nvdrv.nvmap.size(handle),
        Some(0x1000),
        "the creator's reference stays"
    );
    assert_eq!(nv.map_ex(nv.address_space(), 0, handle, 0, 0, 0).0, 0);
}

#[test]
fn ctrl_reads_increments_and_waits_on_syncpoints() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let ctrl = nv.open("/dev/nvhost-ctrl");
    assert_eq!(
        nv.ioctl(ctrl, iowr(0, 0x15, 4), &5u32.to_le_bytes(), 4)
            .result,
        0
    );
    let read = nv.ioctl(ctrl, iowr(0, 0x14, 8), &Bytes::default().u32(5).u32(0).0, 8);
    assert_eq!((read.result, u32_at(&read.output, 4)), (0, 1));
    let max = nv.ioctl(ctrl, iowr(0, 0x1a, 8), &Bytes::default().u32(5).u32(0).0, 8);
    assert_eq!(u32_at(&max.output, 4), 1);
    assert_eq!(nv.syncpoints().max(5), Ok(1));

    let wait = |threshold, timeout: i32| {
        let input = Bytes::default()
            .u32(5)
            .u32(threshold)
            .u32(timeout as u32)
            .u32(0)
            .0;
        let reply = nv.ioctl(ctrl, iowr(0, 0x19, 16), &input, 16);
        (reply.result, u32_at(&reply.output, 12))
    };
    assert_eq!(wait(1, 0), (0, 1));
    assert_eq!(wait(2, 5), (NvResult::Timeout as u32, 1));
    let read = nv.ioctl(
        ctrl,
        iowr(0, 0x14, 8),
        &Bytes::default().u32(0x400).u32(0).0,
        8,
    );
    assert_eq!(read.result, NvResult::BadParameter as u32);
    assert_eq!(
        nv.ioctl(ctrl, iowr(0, 0x1b, 387), &[0; 387], 387).result,
        NvResult::ConfigVarNotFound as u32
    );
}

#[test]
fn gpfifo_fences_follow_host1x_and_ctrl_events_signal() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let as_fd = nv.address_space();
    let (gpu, syncpoint) = nv.channel(as_fd);
    assert!(nv.nvdrv.syncpoints.is_allocated(syncpoint));
    let ctrl = nv.open("/dev/nvhost-ctrl");

    nv.context.gpu.pause(true);
    let reply = nv.submit(gpu, &[], 1 << 1, (0, 0));
    assert_eq!(reply.result, 0);
    assert_eq!(u32_at(&reply.output, 16), syncpoint);
    let fence = u32_at(&reply.output, 20);
    assert_eq!(fence, 2);
    assert_eq!(nv.syncpoints().max(syncpoint), Ok(fence));
    assert_eq!(u32_at(&reply.output, 12), 0, "flags are cleared");

    let wait = Bytes::default().u32(syncpoint).u32(fence).u32(100).u32(0).0;
    let reply = nv.ioctl(ctrl, iowr(0, 0x1d, 16), &wait, 16);
    assert_eq!(reply.result, NvResult::Timeout as u32);
    let event_id = u32_at(&reply.output, 12);
    assert_eq!(event_id >> 16 & 0xfff, syncpoint);
    let mut query = nv
        .client
        .send(nv.session, &Request::new(4).arg(ctrl).arg(event_id))
        .unwrap();
    assert_eq!(query.pop::<u32>(), 0);
    let event = query.copy_handles[0];
    assert!(!nv.event_signaled(event));

    nv.context.gpu.pause(false);
    nv.context.gpu.synchronize();
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    while !nv.event_signaled(event) {
        assert!(std::time::Instant::now() < deadline, "event never signaled");
        std::thread::sleep(Duration::from_millis(1));
    }
    let reply = nv.ioctl(
        ctrl,
        iowr(0, 0x1e, 16),
        &Bytes::default().u32(syncpoint).u32(fence).u32(100).u32(0).0,
        16,
    );
    assert_eq!((reply.result, u32_at(&reply.output, 12)), (0, fence));
    let clear = nv.ioctl(
        ctrl,
        iowr(0, 0x1c, 4),
        &(event_id & 0xffff).to_le_bytes(),
        4,
    );
    assert_eq!(clear.result, 0);
    assert!(!nv.event_signaled(event));
}

#[test]
fn a_closed_channel_retires_the_increments_of_its_dropped_work() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let as_fd = nv.address_space();
    let (gpu, syncpoint) = nv.channel(as_fd);
    let guest = nv.syncpoints().guest();

    nv.syncpoints().increase_max(26, 1).unwrap();
    let reply = nv.submit(gpu, &[], 1 << 0 | 1 << 1, (26, 1));
    assert_eq!(reply.result, 0);
    let fence = u32_at(&reply.output, 20);
    assert_eq!(fence, 2);
    nv.context.gpu.synchronize();
    assert_eq!(
        guest.value(syncpoint),
        Ok(0),
        "suspended on vblank syncpoint 26"
    );

    assert_eq!(nv.close(gpu), 0);
    nv.context.gpu.synchronize();
    assert_eq!(guest.value(syncpoint), Ok(fence));
    assert_eq!(nv.syncpoints().host().value(syncpoint), Ok(fence));
    assert!(!nv.nvdrv.syncpoints.is_allocated(syncpoint));

    let (reused, id) = nv.channel(as_fd);
    assert_eq!(id, syncpoint);
    let reply = nv.submit(reused, &[], 1 << 1, (0, 0));
    assert_eq!(reply.result, 0);
    let fence = u32_at(&reply.output, 20);
    assert_eq!(fence, 4);
    assert_eq!(
        guest.wait(syncpoint, fence, Some(Duration::from_secs(5))),
        Ok(fence)
    );
}

#[test]
fn gpfifo_runs_a_pushbuffer_mapped_through_nvmap_and_the_gmmu() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let as_fd = nv.address_space();
    let (gpu, syncpoint) = nv.channel(as_fd);

    let header = CommandHeader::new(SubmissionMode::Increasing, 0, SYNCPOINT_A, 2);
    let words = [header.0, 0, syncpoint << 8 | 1];
    let (handle, _) = nv.nvmap_buffer(nvmap, bytemuck::cast_slice(&words), 0x1000);
    let gpu_va = nv.map(as_fd, handle);

    let flags = 1 << 1 | 1 << 8;
    let reply = nv.submit(
        gpu,
        &[GpEntry::new(gpu_va, words.len() as u32)],
        flags,
        (0, 1),
    );
    assert_eq!(reply.result, 0);
    let fence = u32_at(&reply.output, 20);
    assert_eq!(fence, 3, "one increment in the pushbuffer, two after it");
    assert_eq!(nv.syncpoints().max(syncpoint), Ok(fence));
    let value = nv
        .syncpoints()
        .guest()
        .wait(syncpoint, fence, Some(Duration::from_secs(5)));
    assert_eq!(value, Ok(3));
    nv.context.gpu.synchronize();
    assert_eq!(nv.syncpoints().guest().value(syncpoint), Ok(3));

    let fence_wait = nv.submit(gpu, &[], 1 << 0 | 1 << 8, (syncpoint, 3));
    assert_eq!(fence_wait.result, NvResult::BadParameter as u32);
}

#[test]
fn mappings_stay_inside_their_nvmap_handle() {
    const BAD: u32 = NvResult::BadValue as u32;
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let as_fd = nv.address_space();
    let (handle, _) = nv.nvmap_buffer(nvmap, &[7; 0x3000], 0x1000);
    let pins = || nv.nvdrv.nvmap.pins(handle);

    assert_eq!(nv.map_ex(as_fd, 0, handle, 0x2000, 0x2000, 0).0, BAD);
    assert_eq!(nv.map_ex(as_fd, 0, handle, -0x1000, 0x1000, 0).0, BAD);
    assert_eq!(nv.map_ex(as_fd, 0, handle, i64::MAX, u64::MAX, 0).0, BAD);
    assert_eq!(nv.map_ex(as_fd, 0, handle, 0x3000, 0, 0).0, BAD);
    assert_eq!(pins(), 0, "rejected before pinning");
    let (result, gpu_va) = nv.map_ex(as_fd, 0, handle, 0, 0, 0);
    assert_eq!((result, pins()), (0, 1));

    let remap = 1 << 8;
    assert_eq!(
        nv.map_ex(as_fd, remap, handle, 0x2000, 0x2000, gpu_va).0,
        BAD
    );
    assert_eq!(nv.map_ex(as_fd, remap, handle, -1, 0x1000, gpu_va).0, BAD);
    assert_eq!(nv.map_ex(as_fd, remap, handle, 0x1000, 0x2000, gpu_va).0, 0);
    assert_eq!(pins(), 2, "the remapped range holds its own pin");

    let (big, _) = nv.nvmap_buffer(nvmap, &[9; 0x10000], 0x10000);
    let sparse = nv.alloc_space(as_fd, 4, 0x10000, 1 << 1);
    assert_eq!(nv.remap(as_fd, big, 1, sparse, 1), BAD);
    assert_eq!(nv.remap(as_fd, handle, 0, sparse, 1), BAD);
    assert_eq!(nv.remap(as_fd, big, u32::MAX, sparse, 1), BAD);
    assert_eq!(nv.nvdrv.nvmap.pins(big), 0);
    assert_eq!(nv.remap(as_fd, big, 0, sparse, 1), 0);
    assert_eq!(nv.nvdrv.nvmap.pins(big), 1);
}

#[test]
fn every_gmmu_range_holding_a_pin_unpins_once() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let as_fd = nv.address_space();
    let (a, _) = nv.nvmap_buffer(nvmap, &[1; 0x3000], 0x1000);
    let (b, _) = nv.nvmap_buffer(nvmap, &[2; 0x1000], 0x1000);
    let pins = |handle| nv.nvdrv.nvmap.pins(handle);

    let space = nv.alloc_space(as_fd, 16, 0x1000, 0);
    assert_eq!(nv.map_ex(as_fd, 1, a, 0, 0, space).0, 0);
    assert_eq!(pins(a), 1);
    assert_eq!(nv.map_ex(as_fd, 1, b, 0, 0, space + 0x1000).0, 0);
    assert_eq!((pins(a), pins(b)), (2, 1), "a keeps both ends");
    assert_eq!(nv.map_ex(as_fd, 1 << 8, a, 0x1000, 0x1000, space).0, 0);
    assert_eq!((pins(a), pins(b)), (3, 0), "the remap replaced b");
    let unmap = nv.ioctl(
        as_fd,
        iowr(b'A', 0x5, 8),
        &(space + 0x1000).to_le_bytes(),
        8,
    );
    assert_eq!(unmap.result, 0);
    assert_eq!(pins(a), 2);
    assert_eq!(nv.free_space(as_fd, space, 16, 0x1000), 0);
    assert_eq!((pins(a), pins(b)), (0, 0));

    let (big, _) = nv.nvmap_buffer(nvmap, &[3; 0x20000], 0x10000);
    let sparse = nv.alloc_space(as_fd, 4, 0x10000, 1 << 1);
    assert_eq!(nv.remap(as_fd, big, 0, sparse, 2), 0);
    assert_eq!(nv.remap(as_fd, big, 0, sparse, 2), 0);
    assert_eq!(pins(big), 1, "a remap over a remap replaces its pin");
    assert_eq!(nv.remap(as_fd, big, 1, sparse + 0x20000, 1), 0);
    assert_eq!(nv.remap(as_fd, 0, 0, sparse + 0x10000, 1), 0);
    assert_eq!(pins(big), 2, "back to sparse in the middle keeps both ends");
    assert_eq!(nv.free_space(as_fd, sparse, 4, 0x10000), 0);
    assert_eq!(pins(big), 0);

    nv.map(as_fd, a);
    nv.map(as_fd, big);
    nv.alloc_space(as_fd, 2, 0x10000, 1 << 1);
    let sparse = nv.alloc_space(as_fd, 2, 0x10000, 1 << 1);
    assert_eq!(nv.remap(as_fd, big, 0, sparse, 2), 0);
    assert_eq!((pins(a), pins(big)), (1, 2));
    assert_eq!(nv.close(as_fd), 0);
    assert_eq!(
        (pins(a), pins(big)),
        (0, 0),
        "dropping the address space unpins"
    );
}

#[test]
fn big_pages_reach_the_end_of_an_address_space_with_a_split() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let as_fd = nv.open("/dev/nvhost-as-gpu");
    let (start, split, end) = (0x400_0000u64, 0x1_0000_0000u64, 0x10_0000_0000u64);
    let init = Bytes::default()
        .u32(1)
        .u32(0)
        .u32(0x10000)
        .u32(0)
        .u64(start)
        .u64(end)
        .u64(split);
    assert_eq!(nv.ioctl(as_fd, iowr(b'A', 0x9, 40), &init.0, 40).result, 0);
    let regions = nv.ioctl(as_fd, iowr(b'A', 0x8, 64), &[0; 64], 64);
    assert_eq!(regions.result, 0);
    assert_eq!(u64_at(&regions.output, 16), start);
    assert_eq!(u64_at(&regions.output, 32), (split - start) >> 12);
    assert_eq!(u64_at(&regions.output, 40), split);
    assert_eq!(u64_at(&regions.output, 56), (end - split) >> 16);
    let pages = ((end - split) >> 16) as u32;
    assert_eq!(nv.alloc_space(as_fd, pages, 0x10000, 0), split);
}

#[test]
fn ctrl_gpu_characteristics_switch1() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let fd = nv.open("/dev/nvhost-ctrl-gpu");
    let input = Bytes::default().u64(0xA0).u64(1).zeroes(0xA0).0;
    let reply = nv.ioctl(fd, iowr(b'G', 0x5, 0xB0), &input, 0xB0);
    assert_eq!(reply.result, 0);
    let gc = &reply.output[16..];
    assert_eq!(u64_at(&reply.output, 0), 0xA0);
    assert_eq!((u32_at(gc, 0), u32_at(gc, 4)), (0x120, 0xB));
    assert_eq!(u32_at(gc, 0x5C), 0xB197);
    assert_eq!(u64_at(gc, 0x50), 0x55);
    assert_eq!(&gc[0x90..0x95], b"gm20b");

    let reply = nv.ioctl3(fd, iowr(b'G', 0x5, 0xB0), &input, 0xB0, 0xA0);
    assert_eq!(&reply.inline_output[..], gc);

    let masks = nv.ioctl3(
        fd,
        iowr(b'G', 0x6, 24),
        &Bytes::default().u32(4).zeroes(20).0,
        24,
        4,
    );
    assert_eq!(
        (u32_at(&masks.output, 16), u32_at(&masks.inline_output, 0)),
        (3, 3)
    );
    let slots = nv.ioctl(fd, iowr(b'G', 0x14, 8), &[0; 8], 8);
    assert_eq!((u32_at(&slots.output, 0), u32_at(&slots.output, 4)), (7, 1));
    assert_eq!(nv.ioctl(fd, iowr(b'G', 0x7, 8), &[0; 8], 8).result, 0);
    assert_eq!(
        nv.ioctl3(fd, iowr(b'G', 0x40, 8), &[0; 8], 8, 8).result,
        NvResult::NotImplemented as u32
    );

    let zbc = Bytes::default()
        .u32(1)
        .u32(2)
        .u32(3)
        .u32(4)
        .zeroes(16)
        .u32(0)
        .u32(0x28)
        .u32(1)
        .0;
    assert_eq!(nv.ioctl(fd, iowr(b'G', 0x3, 44), &zbc, 44).result, 0);
    let query = Bytes::default().zeroes(36).u32(0).u32(0).u32(1).u32(0).0;
    let reply = nv.ioctl(fd, iowr(b'G', 0x4, 52), &query, 52);
    assert_eq!(reply.result, 0);
    assert_eq!(
        (
            u32_at(&reply.output, 0),
            u32_at(&reply.output, 40),
            u32_at(&reply.output, 48)
        ),
        (1, 0x28, 1)
    );
}

#[test]
fn ctrl_gpu_characteristics_switch2() {
    let nv = Nv::new(SWITCH2_TITLE, "nx2:nvsrv");
    let fd = nv.open("/dev/nvhost-ctrl-gpu");
    let input = Bytes::default().u64(0xD0).u64(1).zeroes(0xA0).0;
    let reply = nv.ioctl3(fd, iowr(b'G', 0x5, 0xB0), &input, 0xB0, 0xD0);
    assert_eq!(reply.result, 0);
    assert_eq!(u64_at(&reply.output, 0), 0xD0);
    let gc = &reply.inline_output;
    assert_eq!((u32_at(gc, 0), u32_at(gc, 4)), (0x170, 0xE), "chip 0x17E");
    assert_eq!(
        (u32_at(gc, 0x5C), u32_at(gc, 0x60), u32_at(gc, 0x64)),
        (0xC797, 0xC7C0, 0xC56F)
    );
    let flags = u64_at(gc, 0x50);
    for bit in [8, 22, 26, 34, 35, 40, 44] {
        assert_ne!(flags & 1 << bit, 0, "flag bit {bit}");
    }
    for offset in (0x70..=0x80).step_by(2) {
        assert_eq!(u16::from_le_bytes([gc[offset], gc[offset + 1]]), 0x40);
    }
    assert_eq!(u32_at(gc, 0xB4), 64, "subcontexts per TSG");
    assert_eq!(&gc[0x90..0x95], b"ga10f");

    let unknown = nv.ioctl3(fd, iowr(b'G', 0x40, 8), &[1; 8], 8, 8);
    assert_eq!(
        (unknown.result, unknown.output, unknown.inline_output),
        (0, vec![0; 8], vec![0; 8])
    );
    let size = nv.ioctl3(fd, iowr(b'G', 0x1f, 16), &[0; 16], 16, 0);
    assert_eq!(u32_at(&size.output, 0), 32);
    let info = nv.ioctl3(
        fd,
        iowr(b'G', 0x1f, 16),
        &Bytes::default().u32(32).zeroes(12).0,
        16,
        32,
    );
    assert_eq!(
        (
            u32_at(&info.inline_output, 16),
            u32_at(&info.inline_output, 24)
        ),
        (2, 1)
    );
}

#[test]
fn switch2_address_space_regions_and_syncpoint_shim() {
    let nv = Nv::new(SWITCH2_TITLE, "nx2:nvsrv");
    let as_fd = nv.open("/dev/nvhost-as-gpu");
    let init = Bytes::default()
        .u32(5)
        .u32(0)
        .u32(0)
        .u32(0)
        .u64(0x800_0000)
        .u64(0xC0_0000_0000)
        .u64(0)
        .0;
    assert_eq!(nv.ioctl(as_fd, iowr(b'A', 0x9, 40), &init, 40).result, 0);
    let regions = nv.ioctl3(as_fd, iowr(b'A', 0x8, 64), &[0; 16], 64, 48);
    assert_eq!(regions.result, 0);
    let r = &regions.inline_output;
    assert_eq!((u64_at(r, 0), u32_at(r, 8)), (0x800_0000, 0x1000));
    assert_eq!(u64_at(r, 16), (0xC0_0000_0000 - 0x800_0000) >> 12);
    assert_eq!((u64_at(r, 24), u32_at(r, 32)), (0xC0_0000_0000, 0x20000));
    assert_eq!(&regions.output[16..], &r[..]);

    let shim = nv.ioctl(as_fd, iowr(b'A', 0xb, 16), &[0; 16], 16);
    assert_eq!(shim.result, 0);
    let address = u64_at(&shim.output, 0);
    assert_eq!(address, 0xC0_0000_0000 + (4 << 30) - 0x1000);
    assert_eq!(
        (u32_at(&shim.output, 8), u32_at(&shim.output, 12)),
        (4, 192)
    );
    let again = nv.ioctl(as_fd, iowr(b'A', 0xb, 16), &[0; 16], 16);
    assert_eq!(u64_at(&again.output, 0), address);
}

#[test]
fn switch2_channels_echo_unknown_ioctls_and_map_ampere_classes() {
    let nv = Nv::new(SWITCH2_TITLE, "nx2:nvsrv");
    let gpu = nv.open("/dev/nvhost-gpu");
    let echo = nv.ioctl(gpu, iowr(b'H', 0x1f, 8), &[5; 8], 8);
    assert_eq!((echo.result, echo.output), (0, vec![5; 8]));
    let obj = |class: u32| {
        let reply = nv.ioctl(
            gpu,
            iowr(b'H', 0x9, 16),
            &Bytes::default().u32(class).zeroes(12).0,
            16,
        );
        (reply.result, u32_at(&reply.output, 0))
    };
    assert_eq!(obj(0xC797), (0, 0xB197));
    assert_eq!(obj(0), (0, 0xB1C0));
    assert_eq!(obj(0xB197), (NvResult::AlreadyAllocated as u32, 0xB197));
    let mut query = nv
        .client
        .send(nv.session, &Request::new(4).arg(gpu).arg(7u32))
        .unwrap();
    assert_eq!(query.pop::<u32>(), 0);
}

#[test]
fn host1x_engine_channels_increment_syncpoints_as_their_command_buffers_run() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let nvdec = nv.open("/dev/nvhost-nvdec");
    let reply = nv.ioctl(nvdec, iowr(0, 0x2, 8), &[0; 8], 8);
    let syncpoint = u32_at(&reply.output, 4);
    assert!(nv.nvdrv.syncpoints.is_allocated(syncpoint));
    let commands = Bytes::default()
        .zeroes(8)
        .u32((4 << 28) | syncpoint)
        .u32((1 << 28) | (0x10 << 16) | 2)
        .u32(0xc0)
        .u32(0)
        .u32((4 << 28) | (1 << 8) | syncpoint)
        .0;
    let (buffer, _) = nv.nvmap_buffer(nvmap, &commands, 0x1000);
    let submit = Bytes::default()
        .u32(1)
        .u32(0)
        .u32(1)
        .u32(1)
        .u32(buffer)
        .u32(8)
        .u32(5)
        .u32(syncpoint)
        .u32(2)
        .zeroes(12)
        .u32(0)
        .0;
    let reply = nv.ioctl(nvdec, iowr(0, 0x1, 16), &submit, submit.len());
    assert_eq!(reply.result, 0);
    assert_eq!(u32_at(&reply.output, 48), 2);
    nv.syncpoints()
        .host()
        .wait(syncpoint, 2, Some(std::time::Duration::from_secs(5)))
        .expect("the channel ran both increments");
    assert_eq!(nv.syncpoints().guest().value(syncpoint), Ok(2));
    assert_eq!(nv.close(nvdec), 0);
    assert!(!nv.nvdrv.syncpoints.is_allocated(syncpoint));
}

#[test]
fn closing_a_host1x_channel_retires_increments_it_never_ran() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let vic = nv.open("/dev/nvhost-vic");
    let reply = nv.ioctl(vic, iowr(0, 0x2, 8), &[0; 8], 8);
    let syncpoint = u32_at(&reply.output, 4);
    let never = nv.nvdrv.syncpoints.allocate();
    let commands = Bytes::default()
        .u32((4 << 28) | syncpoint)
        .u32(1 << 6)
        .u32((4 << 28) | (0x4e << 16) | 1)
        .u32((4 << 28) | (0x50 << 16) | never)
        .u32((4 << 28) | syncpoint)
        .0;
    let (buffer, _) = nv.nvmap_buffer(nvmap, &commands, 0x1000);
    let submit = Bytes::default()
        .u32(1)
        .u32(0)
        .u32(1)
        .u32(1)
        .u32(buffer)
        .u32(0)
        .u32(5)
        .u32(syncpoint)
        .u32(2)
        .zeroes(12)
        .u32(0)
        .0;
    assert_eq!(
        nv.ioctl(vic, iowr(0, 0x1, 16), &submit, submit.len())
            .result,
        0
    );
    nv.syncpoints()
        .host()
        .wait(syncpoint, 1, Some(std::time::Duration::from_secs(5)))
        .expect("the first increment ran");
    assert_eq!(nv.close(vic), 0);
    assert!(!nv.nvdrv.syncpoints.is_allocated(syncpoint));
    assert_eq!(nv.syncpoints().guest().value(syncpoint), Ok(2));
    assert_eq!(nv.syncpoints().host().value(syncpoint), Ok(2));
}

#[test]
fn flips_present_pinned_nvmap_buffers() {
    use lichii_gpu::backend::FramebufferConfig;
    use lichii_texture::PixelFormat;
    use lichii_texture::block_linear::GobLayout;

    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let (handle, _) = nv.nvmap_buffer(nvmap, &[0; 0x4000], 0x1000);
    let framebuffer = FramebufferConfig {
        address: 0,
        width: 64,
        height: 64,
        stride: 256,
        format: PixelFormat::ALL[0],
        block_height_log2: None,
        gob_layout: GobLayout::Tegra,
        flip_vertical: false,
        crop: None,
    };
    let unused = NvFence { id: -1, value: 0 };
    assert_eq!(
        nv.nvdrv.flip(handle, 0, framebuffer, &[unused], || {}),
        NvResult::BadValue
    );
    let as_fd = nv.address_space();
    nv.map(as_fd, handle);
    assert_eq!(
        nv.nvdrv.flip(handle, 0, framebuffer, &[unused], || {}),
        NvResult::Success
    );
    assert!(nv.open("/dev/nvdisp_disp0") > 0);
}

#[test]
fn a_flip_waits_for_its_acquire_fence_on_the_syncpoint_not_in_the_caller() {
    use lichii_gpu::backend::FramebufferConfig;
    use lichii_texture::PixelFormat;
    use lichii_texture::block_linear::GobLayout;

    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let (handle, _) = nv.nvmap_buffer(nvmap, &[0; 0x4000], 0x1000);
    let as_fd = nv.address_space();
    nv.map(as_fd, handle);
    let framebuffer = FramebufferConfig {
        address: 0,
        width: 64,
        height: 64,
        stride: 256,
        format: PixelFormat::ALL[0],
        block_height_log2: None,
        gob_layout: GobLayout::Tegra,
        flip_vertical: false,
        crop: None,
    };
    let guest = nv.syncpoints().guest();
    let syncpoint = 30;
    let fence = NvFence {
        id: syncpoint as i32,
        value: guest.value(syncpoint).unwrap() + 1,
    };
    let presented = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let seen = presented.clone();
    let started = std::time::Instant::now();
    assert_eq!(
        nv.nvdrv.flip(handle, 0, framebuffer, &[fence], move || {
            seen.store(true, std::sync::atomic::Ordering::SeqCst)
        }),
        NvResult::Success
    );
    assert!(
        started.elapsed() < std::time::Duration::from_millis(100),
        "the flip returned after {:?}",
        started.elapsed()
    );
    assert_eq!(
        guest.next_action_threshold(syncpoint),
        Ok(Some(fence.value)),
        "the present waits on the syncpoint"
    );
    assert!(
        !presented.load(std::sync::atomic::Ordering::SeqCst),
        "the buffer stays with the display until the present is queued"
    );
    guest.increment(syncpoint).unwrap();
    assert_eq!(
        guest.next_action_threshold(syncpoint),
        Ok(None),
        "reaching the fence presents the frame"
    );
    assert!(presented.load(std::sync::atomic::Ordering::SeqCst));
}

#[test]
fn buffers_vi_holds_survive_the_guest_freeing_them() {
    let nv = Nv::new(SWITCH1_TITLE, "nvdrv");
    let nvmap = nv.open("/dev/nvmap");
    let (handle, _) = nv.nvmap_buffer(nvmap, &[0; 0x4000], 0x1000);
    assert!(nv.nvdrv.pin_buffer(handle));
    let daddr = nv.nvdrv.nvmap.device_address(handle);
    assert_ne!(daddr, 0);
    let free = Bytes::default().u32(handle).zeroes(20);
    assert_eq!(nv.ioctl(nvmap, iowr(1, 0x5, 24), &free.0, 24).result, 0);
    assert_eq!(
        nv.nvdrv.nvmap.device_address(handle),
        daddr,
        "vi's reference keeps the mapping"
    );
    nv.nvdrv.unpin_buffer(handle);
    assert_eq!(
        nv.nvdrv.nvmap.size(handle),
        None,
        "the last reference removes it"
    );
    assert!(!nv.nvdrv.pin_buffer(handle));
}
