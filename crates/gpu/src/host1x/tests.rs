//! NVDEC and VIC on synthetic data: VIC conversion and output layouts against known pixels, and
//! H.264 decoded through NVDEC and composed by VIC from synthetic command buffers.
//!
//! Tests that decode or compose need FFmpeg's shared libraries; they pass without them unless
//! LICHII_REQUIRE_FFMPEG is set, as CI does.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use lichii_memory::{AddressSpace, MemoryPermission};
use lichii_video::nvdec::VideoCodec;
use lichii_video::nvdec::picture::H264DecoderContext;
use lichii_video::nvdec::registers::index;

use super::channel::tests::{host1x, immediate, incrementing, set_class, wait_for};
use super::channel::{THI_INCR_SYNCPT, THI_METHOD_0};
use super::nvdec::Nvdec;
use super::vic::Vic;
use super::{EngineChannel, Host1x, VideoEngine};

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

// SAFETY: every call forwards to the system allocator with the caller's arguments unchanged.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated by `System` with `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|count| count.set(count.get() + 1));
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn ffmpeg() -> bool {
    if lichii_video::available() {
        return true;
    }
    assert!(
        std::env::var_os("LICHII_REQUIRE_FFMPEG").is_none(),
        "LICHII_REQUIRE_FFMPEG is set but FFmpeg did not load"
    );
    eprintln!("FFmpeg is not installed; skipping");
    false
}

const MEMORY_SIZE: u64 = 1 << 26;
const CONFIG: u64 = 0x10_0000;
const PICTURE_INFO: u64 = 0x20_0000;
const BITSTREAM: u64 = 0x30_0000;
const SOURCE_LUMA: u64 = 0x40_0000;
const SOURCE_CHROMA: u64 = 0x60_0000;
const OUTPUT_LUMA: u64 = 0x80_0000;
const OUTPUT_CHROMA: u64 = 0xc0_0000;

/// Luma plane of NVDEC output surface `index`; the chroma plane follows 1.5 MiB later.
fn surface_luma(index: usize) -> u64 {
    0x100_0000 + index as u64 * 0x20_0000
}

fn surface_chroma(index: usize) -> u64 {
    surface_luma(index) + 0x18_0000
}

/// A Host1x whose whole device space maps DRAM one to one.
fn mapped_host1x() -> Host1x {
    let host1x = host1x();
    host1x.memory.map(0, 0, MEMORY_SIZE);
    host1x
}

fn read(host1x: &Host1x, address: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0; len];
    host1x.memory.read_bytes(address, &mut bytes).unwrap();
    bytes
}

/// VIC register indices (byte offsets in the reference's `VicRegisters` divided by 4).
mod vic_reg {
    pub const EXECUTE: u32 = 0x300 / 4;
    pub const SURFACES: u32 = 0x400 / 4;
    pub const CONFIG_STRUCT_OFFSET: u32 = 0x708 / 4;
    pub const OUTPUT_LUMA_OFFSET: u32 = 0x720 / 4;
    pub const OUTPUT_CHROMA_OFFSET: u32 = 0x724 / 4;
}

const A8B8G8R8: u64 = 31;
const A8R8G8B8: u64 = 32;
const Y8_V8U8_N420: u64 = 68;

/// Surface layout fields shared by the output and slot surface configs.
#[derive(Clone, Copy)]
struct Surface {
    format: u64,
    block_height: Option<u64>,
    width: u64,
    height: u64,
}

/// BT.601 limited range in the VIC's fixed point: 10-bit samples, coefficients in 1/256.
const BT601: [[i32; 4]; 3] = [
    [298, 0, 409, -298 * 64 - 409 * 512],
    [298, -100, -208, -298 * 64 + 308 * 512],
    [298, 516, 0, -298 * 64 - 516 * 512],
];

#[derive(Clone, Copy)]
struct Slot {
    surface: Surface,
    /// Inclusive pixel edges: left, top, right, bottom.
    source: [u64; 4],
    dest: [u64; 4],
    matrix: Option<[[i32; 4]; 3]>,
}

impl Slot {
    fn whole(surface: Surface, matrix: Option<[[i32; 4]; 3]>) -> Self {
        let edges = [0, 0, surface.width - 1, surface.height - 1];
        Self {
            surface,
            source: edges,
            dest: edges,
            matrix,
        }
    }
}

/// A VIC `ConfigStruct`, field by field from the reference's vic.h.
struct VicConfig([u64; 0x610 / 8]);

impl VicConfig {
    fn new(output: Surface) -> Self {
        let mut config = Self([0; 0x610 / 8]);
        config.surface(0x20, output);
        config.target([0, 0, output.width - 1, output.height - 1]);
        config
    }

    fn set(&mut self, offset: usize, low: u32, len: u32, value: u64) {
        let low = low + (offset % 8 * 8) as u32;
        let mask = if len == 64 { u64::MAX } else { (1 << len) - 1 };
        let word = &mut self.0[offset / 8];
        *word = (*word & !(mask << low)) | ((value & mask) << low);
    }

    fn surface(&mut self, offset: usize, surface: Surface) {
        self.set(offset, 0, 7, surface.format);
        self.set(offset, 11, 4, u64::from(surface.block_height.is_some()));
        self.set(offset, 15, 4, surface.block_height.unwrap_or(0));
        let pitch = surface.width;
        let chroma_height = surface.height.div_ceil(2);
        for (at, (w, h)) in [
            (4, (surface.width, surface.height)),
            (8, (pitch, surface.height)),
            (12, (pitch / 2, chroma_height)),
        ] {
            self.set(offset + at, 0, 14, w - 1);
            self.set(offset + at, 14, 14, h - 1);
        }
    }

    fn target(&mut self, [left, top, right, bottom]: [u64; 4]) {
        self.set(0x18, 0, 14, left);
        self.set(0x18, 16, 14, right);
        self.set(0x1c, 0, 14, top);
        self.set(0x1c, 16, 14, bottom);
    }

    /// Background colour in 10-bit components; alpha fill mode 1 (background).
    fn background(&mut self, [r, g, b, a]: [u64; 4]) {
        self.set(0x10, 0, 3, 1);
        self.set(0x10, 6, 10, a);
        self.set(0x10, 16, 10, r);
        self.set(0x10, 26, 10, g);
        self.set(0x10, 36, 10, b);
    }

    fn slot(&mut self, index: usize, slot: Slot) {
        let base = 0x90 + index * 0xb0;
        self.set(base, 0, 1, 1);
        self.set(base + 0x10, 0, 10, 0);
        self.set(base + 0x10, 10, 10, 1023);
        self.set(base + 0x10, 32, 10, 1023);
        let [left, top, right, bottom] = slot.source;
        self.set(base + 0x20, 0, 30, left << 16);
        self.set(base + 0x20, 32, 30, right << 16);
        self.set(base + 0x28, 0, 30, top << 16);
        self.set(base + 0x28, 32, 30, bottom << 16);
        let [left, top, right, bottom] = slot.dest;
        self.set(base + 0x30, 0, 14, left);
        self.set(base + 0x30, 16, 14, right);
        self.set(base + 0x30, 32, 14, top);
        self.set(base + 0x30, 48, 14, bottom);
        self.surface(base + 0x40, slot.surface);
        if let Some(matrix) = slot.matrix {
            for column in 0..4 {
                let word = base + 0x60 + column * 8;
                for (row, coefficients) in matrix.iter().enumerate() {
                    self.set(word, row as u32 * 20, 20, coefficients[column] as u64);
                }
            }
            self.set(base + 0x68, 63, 1, 1);
        }
    }

    fn bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

/// The engine method writes of one VIC job composing `slots` (slot index, luma, chroma).
fn vic_methods(slots: &[(usize, u64, u64)]) -> Vec<(u32, u32)> {
    let mut methods = Vec::new();
    for &(slot, luma, chroma) in slots {
        let surface = vic_reg::SURFACES + slot as u32 * 24;
        methods.push((surface, (luma >> 8) as u32));
        methods.push((surface + 1, (chroma >> 8) as u32));
    }
    methods.extend([
        (vic_reg::CONFIG_STRUCT_OFFSET, (CONFIG >> 8) as u32),
        (vic_reg::OUTPUT_LUMA_OFFSET, (OUTPUT_LUMA >> 8) as u32),
        (vic_reg::OUTPUT_CHROMA_OFFSET, (OUTPUT_CHROMA >> 8) as u32),
        (vic_reg::EXECUTE, 0),
    ]);
    methods
}

/// A command buffer that writes `methods` through the THI block of `engine` and then increments
/// `syncpoint`.
fn command_buffer(engine: VideoEngine, methods: &[(u32, u32)], syncpoint: u32) -> Vec<u32> {
    let class = match engine {
        VideoEngine::NvDec => 0xf0,
        VideoEngine::Vic => 0x5d,
        VideoEngine::NvJpg => 0xc0,
    };
    let mut words = vec![set_class(class, 0, 0)];
    for &(method, value) in methods {
        words.extend([incrementing(THI_METHOD_0, 2), method, value]);
    }
    words.push(immediate(THI_INCR_SYNCPT, syncpoint));
    words
}

/// The VIC's integer YUV to RGB, written out from the hardware description.
fn vic_rgb(matrix: &[[i32; 4]; 3], [y, u, v]: [u8; 3]) -> [u8; 3] {
    let samples = [i32::from(y) * 4, i32::from(u) * 4, i32::from(v) * 4];
    matrix.map(|row| {
        let sum: i32 = (0..3).map(|i| samples[i] * row[i]).sum();
        (((sum + row[3]) >> 8).clamp(0, 1023) / 4) as u8
    })
}

/// A synthetic NV12 picture: four flat quadrants.
struct Nv12 {
    width: usize,
    height: usize,
    luma: Vec<u8>,
    chroma: Vec<u8>,
}

const QUADRANTS: [[u8; 3]; 4] = [
    [81, 90, 240],
    [145, 54, 34],
    [41, 240, 110],
    [235, 128, 128],
];

impl Nv12 {
    fn quadrants(width: usize, height: usize) -> Self {
        let quadrant = |x: usize, y: usize| {
            QUADRANTS[usize::from(y >= height / 2) * 2 + usize::from(x >= width / 2)]
        };
        let luma = (0..height)
            .flat_map(|y| (0..width).map(move |x| quadrant(x, y)[0]))
            .collect();
        let chroma = (0..height / 2)
            .flat_map(|y| (0..width).map(move |x| quadrant(x & !1, y * 2)[1 + x % 2]))
            .collect();
        Self {
            width,
            height,
            luma,
            chroma,
        }
    }

    fn yuv(&self, x: usize, y: usize) -> [u8; 3] {
        let c = (y / 2) * self.width + (x & !1);
        [
            self.luma[y * self.width + x],
            self.chroma[c],
            self.chroma[c + 1],
        ]
    }

    fn surface(&self) -> Surface {
        Surface {
            format: Y8_V8U8_N420,
            block_height: None,
            width: self.width as u64,
            height: self.height as u64,
        }
    }

    fn store_pitch(&self, host1x: &Host1x) {
        host1x.memory.write_bytes(SOURCE_LUMA, &self.luma).unwrap();
        host1x
            .memory
            .write_bytes(SOURCE_CHROMA, &self.chroma)
            .unwrap();
    }
}

fn run_vic(host1x: &Host1x, config: &VicConfig, slots: &[(usize, u64, u64)]) {
    host1x.memory.write_bytes(CONFIG, config.bytes()).unwrap();
    let mut vic = Vic::new(host1x.memory.clone(), None, host1x.decoded_frames.clone());
    for (method, value) in vic_methods(slots) {
        vic.write(method, value);
    }
}

fn rgba_output(width: u64, height: u64, format: u64, block_height: Option<u64>) -> Surface {
    Surface {
        format,
        block_height,
        width,
        height,
    }
}

#[test]
fn vic_converts_nv12_with_the_slot_matrix_into_pitch_rgb() {
    if !ffmpeg() {
        return;
    }
    let host1x = mapped_host1x();
    let picture = Nv12::quadrants(32, 16);
    picture.store_pitch(&host1x);
    for (format, bgra) in [(A8B8G8R8, false), (A8R8G8B8, true)] {
        let mut config = VicConfig::new(rgba_output(32, 16, format, None));
        config.slot(0, Slot::whole(picture.surface(), Some(BT601)));
        run_vic(&host1x, &config, &[(0, SOURCE_LUMA, SOURCE_CHROMA)]);
        let out = read(&host1x, OUTPUT_LUMA, 32 * 16 * 4);
        for y in 0..16 {
            for x in 0..32 {
                let yuv = picture.yuv(x, y);
                let [r, g, b] = vic_rgb(&BT601, yuv);
                let expected = if bgra {
                    [b, g, r, 0xff]
                } else {
                    [r, g, b, 0xff]
                };
                let at = (y * 32 + x) * 4;
                assert_eq!(out[at..at + 4], expected, "format {format} pixel {x},{y}");
                let [yf, uf, vf] = yuv.map(f32::from);
                let float = [
                    1.164 * (yf - 16.0) + 1.596 * (vf - 128.0),
                    1.164 * (yf - 16.0) - 0.391 * (uf - 128.0) - 0.813 * (vf - 128.0),
                    1.164 * (yf - 16.0) + 2.018 * (uf - 128.0),
                ];
                for (channel, value) in [r, g, b].into_iter().zip(float) {
                    assert!(
                        (f32::from(channel) - value.clamp(0.0, 255.0)).abs() <= 2.0,
                        "pixel {x},{y}: {channel} vs {value}"
                    );
                }
            }
        }
    }
}

/// Byte offset of byte `x` of row `y` in a Tegra block-linear surface `row_bytes` wide, with
/// blocks `1 << block_height` GOBs tall: GOBs are 64 bytes by 8 rows, 16-byte units inside.
fn block_linear_offset(x: usize, y: usize, row_bytes: usize, block_height: u32) -> usize {
    let gob_rows = 8 << block_height;
    let block_size = 512 << block_height;
    let blocks_per_row = row_bytes.div_ceil(64);
    let block = (y / gob_rows) * blocks_per_row + x / 64;
    let gob = (y % gob_rows) / 8;
    let in_gob =
        ((x % 64) / 32) * 256 + ((y % 8) / 2) * 64 + ((x % 32) / 16) * 32 + (y % 2) * 16 + x % 16;
    block * block_size + gob * 512 + in_gob
}

#[test]
fn vic_block_linear_output_matches_pitch_output() {
    if !ffmpeg() {
        return;
    }
    let host1x = mapped_host1x();
    let picture = Nv12::quadrants(48, 40);
    picture.store_pitch(&host1x);
    let mut config = VicConfig::new(rgba_output(48, 40, A8B8G8R8, None));
    config.slot(0, Slot::whole(picture.surface(), Some(BT601)));
    run_vic(&host1x, &config, &[(0, SOURCE_LUMA, SOURCE_CHROMA)]);
    let pitch = read(&host1x, OUTPUT_LUMA, 48 * 40 * 4);
    for block_height in [0, 1, 4] {
        host1x.memory.fill(OUTPUT_LUMA, 0x10_0000, 0).unwrap();
        let mut config = VicConfig::new(rgba_output(48, 40, A8B8G8R8, Some(block_height)));
        config.slot(0, Slot::whole(picture.surface(), Some(BT601)));
        run_vic(&host1x, &config, &[(0, SOURCE_LUMA, SOURCE_CHROMA)]);
        let tiled = read(&host1x, OUTPUT_LUMA, 0x10_0000);
        for y in 0..40 {
            for x in 0..48 * 4 {
                let at = block_linear_offset(x, y, 48 * 4, block_height as u32);
                assert_eq!(
                    tiled[at],
                    pitch[y * 48 * 4 + x],
                    "block height {block_height} byte {x},{y}"
                );
            }
        }
    }
}

#[test]
fn vic_keeps_yuv_for_nv12_output_without_a_matrix() {
    if !ffmpeg() {
        return;
    }
    let host1x = mapped_host1x();
    let picture = Nv12::quadrants(32, 16);
    picture.store_pitch(&host1x);
    let mut config = VicConfig::new(Surface {
        format: Y8_V8U8_N420,
        ..picture.surface()
    });
    config.slot(0, Slot::whole(picture.surface(), None));
    run_vic(&host1x, &config, &[(0, SOURCE_LUMA, SOURCE_CHROMA)]);
    assert_eq!(read(&host1x, OUTPUT_LUMA, 32 * 16), picture.luma);
    assert_eq!(read(&host1x, OUTPUT_CHROMA, 32 * 8), picture.chroma);

    let mut config = VicConfig::new(Surface {
        format: Y8_V8U8_N420,
        block_height: Some(1),
        ..picture.surface()
    });
    config.slot(0, Slot::whole(picture.surface(), None));
    run_vic(&host1x, &config, &[(0, SOURCE_LUMA, SOURCE_CHROMA)]);
    let (luma, chroma) = (
        read(&host1x, OUTPUT_LUMA, 0x1000),
        read(&host1x, OUTPUT_CHROMA, 0x1000),
    );
    for y in 0..16 {
        for x in 0..32 {
            let at = block_linear_offset(x, y, 32, 1);
            assert_eq!(luma[at], picture.luma[y * 32 + x], "luma {x},{y}");
            if y < 8 {
                assert_eq!(chroma[at], picture.chroma[y * 32 + x], "chroma {x},{y}");
            }
        }
    }
}

#[test]
fn vic_scales_into_the_destination_and_clips_to_the_target() {
    if !ffmpeg() {
        return;
    }
    let host1x = mapped_host1x();
    let picture = Nv12::quadrants(16, 8);
    picture.store_pitch(&host1x);
    let mut config = VicConfig::new(rgba_output(40, 24, A8B8G8R8, None));
    config.background([400, 200, 100, 0]);
    config.target([0, 0, 27, 13]);
    config.slot(
        0,
        Slot {
            dest: [4, 2, 35, 17],
            ..Slot::whole(picture.surface(), Some(BT601))
        },
    );
    run_vic(&host1x, &config, &[(0, SOURCE_LUMA, SOURCE_CHROMA)]);
    let out = read(&host1x, OUTPUT_LUMA, 40 * 24 * 4);
    for y in 0..24 {
        for x in 0..40 {
            let inside = (4..28).contains(&x) && (2..14).contains(&y);
            let expected = if inside {
                let [r, g, b] = vic_rgb(&BT601, picture.yuv((x - 4) / 2, (y - 2) / 2));
                [r, g, b, 0xff]
            } else {
                [100, 50, 25, 0]
            };
            let at = (y * 40 + x) * 4;
            assert_eq!(out[at..at + 4], expected, "pixel {x},{y}");
        }
    }
}

#[test]
fn vic_reads_block_linear_guest_surfaces() {
    if !ffmpeg() {
        return;
    }
    let host1x = mapped_host1x();
    let picture = Nv12::quadrants(64, 32);
    let block_height = 1;
    let mut luma = vec![0; 0x10_0000];
    let mut chroma = vec![0; 0x10_0000];
    for y in 0..32 {
        for x in 0..64 {
            luma[block_linear_offset(x, y, 64, block_height)] = picture.luma[y * 64 + x];
            if y < 16 {
                chroma[block_linear_offset(x, y, 64, block_height)] = picture.chroma[y * 64 + x];
            }
        }
    }
    host1x.memory.write_bytes(SOURCE_LUMA, &luma).unwrap();
    host1x.memory.write_bytes(SOURCE_CHROMA, &chroma).unwrap();
    let mut config = VicConfig::new(rgba_output(64, 32, A8B8G8R8, None));
    let surface = Surface {
        block_height: Some(u64::from(block_height)),
        ..picture.surface()
    };
    config.slot(0, Slot::whole(surface, Some(BT601)));
    run_vic(&host1x, &config, &[(0, SOURCE_LUMA, SOURCE_CHROMA)]);
    let out = read(&host1x, OUTPUT_LUMA, 64 * 32 * 4);
    for y in 0..32 {
        for x in 0..64 {
            let [r, g, b] = vic_rgb(&BT601, picture.yuv(x, y));
            let at = (y * 64 + x) * 4;
            assert_eq!(out[at..at + 4], [r, g, b, 0xff], "pixel {x},{y}");
        }
    }
}

#[test]
fn vic_output_is_reported_to_write_tracking() {
    if !ffmpeg() {
        return;
    }
    let host1x = mapped_host1x();
    let process = Arc::new(AddressSpace::new(host1x.memory.physical().clone(), 39));
    process.map(0x1000_0000, 0, MEMORY_SIZE, MemoryPermission::READ_WRITE);
    host1x.memory.attach_process(process.clone());
    let picture = Nv12::quadrants(32, 16);
    picture.store_pitch(&host1x);
    let mut config = VicConfig::new(rgba_output(32, 64, A8B8G8R8, None));
    config.slot(0, Slot::whole(picture.surface(), Some(BT601)));
    host1x.memory.write_bytes(CONFIG, config.bytes()).unwrap();
    let cached = host1x.memory.cache_pages(OUTPUT_LUMA - 0x1000, 0x4000);
    let mut vic = Vic::new(
        host1x.memory.clone(),
        Some(process),
        host1x.decoded_frames.clone(),
    );
    for (method, value) in vic_methods(&[(0, SOURCE_LUMA, SOURCE_CHROMA)]) {
        vic.write(method, value);
    }
    let mut written = Vec::new();
    host1x
        .memory
        .take_cpu_writes(|daddr, size| written.push((daddr, size)));
    assert_eq!(written, [(OUTPUT_LUMA, 0x2000)]);
    drop(cached);
}

/// Just enough H.264 parsing to describe a stream to NVDEC the way a guest's decoder library does.
mod h264 {
    use lichii_video::nvdec::picture::H264DecoderContext;

    pub struct Reader {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl Reader {
        /// Over a NAL unit's payload, emulation prevention bytes removed.
        pub fn new(nal: &[u8]) -> Self {
            let mut bytes = Vec::with_capacity(nal.len());
            let mut zeros = 0;
            for &byte in nal {
                if zeros >= 2 && byte == 3 {
                    zeros = 0;
                    continue;
                }
                zeros = if byte == 0 { zeros + 1 } else { 0 };
                bytes.push(byte);
            }
            Self { bytes, bit: 0 }
        }

        pub fn u(&mut self, bits: u32) -> u32 {
            let mut value = 0;
            for _ in 0..bits {
                let byte = self.bytes.get(self.bit / 8).copied().unwrap_or(0);
                value = value << 1 | u32::from(byte >> (7 - self.bit % 8) & 1);
                self.bit += 1;
            }
            value
        }

        pub fn flag(&mut self) -> bool {
            self.u(1) != 0
        }

        pub fn ue(&mut self) -> u32 {
            let mut zeros = 0;
            while self.u(1) == 0 {
                zeros += 1;
            }
            (1 << zeros) - 1 + self.u(zeros)
        }

        pub fn se(&mut self) -> i32 {
            let value = self.ue() as i32;
            if value & 1 == 1 {
                (value + 1) / 2
            } else {
                -value / 2
            }
        }

        pub fn more_data(&self) -> bool {
            let last = self.bytes.iter().rposition(|&b| b != 0).unwrap_or(0);
            let stop = last * 8 + 7 - self.bytes[last].trailing_zeros() as usize;
            self.bit < stop
        }
    }

    /// The SPS and PPS fields NVDEC's picture parameters carry.
    #[derive(Default, Clone, Copy)]
    pub struct Params {
        pub log2_max_frame_num_minus4: u32,
        pub poc_type: u32,
        pub log2_max_poc_lsb_minus4: u32,
        pub width_mbs: u32,
        pub height_mbs: u32,
        pub frame_mbs_only: bool,
        pub direct_8x8: bool,
        pub entropy: bool,
        pub bottom_field_poc: bool,
        pub ref_l0: u32,
        pub ref_l1: u32,
        pub weighted_pred: bool,
        pub weighted_bipred: u32,
        pub init_qp_minus26: i32,
        pub chroma_qp_offset: i32,
        pub deblocking_control: bool,
        pub constrained_intra: bool,
        pub redundant_pic_cnt: bool,
        pub transform_8x8: bool,
        pub second_chroma_qp_offset: i32,
    }

    impl Params {
        pub fn sps(&mut self, nal: &[u8]) {
            let mut r = Reader::new(nal);
            let profile = r.u(8);
            r.u(16);
            r.ue();
            if matches!(profile, 100 | 110 | 122 | 244) {
                assert_eq!(r.ue(), 1, "4:2:0 only");
                r.ue();
                r.ue();
                r.flag();
                assert!(!r.flag(), "no SPS scaling lists");
            }
            self.log2_max_frame_num_minus4 = r.ue();
            self.poc_type = r.ue();
            assert_eq!(self.poc_type, 0, "pic_order_cnt_type 0 only");
            self.log2_max_poc_lsb_minus4 = r.ue();
            r.ue();
            r.flag();
            self.width_mbs = r.ue() + 1;
            let map_units = r.ue() + 1;
            self.frame_mbs_only = r.flag();
            assert!(self.frame_mbs_only, "progressive only");
            self.height_mbs = map_units;
            self.direct_8x8 = r.flag();
        }

        pub fn pps(&mut self, nal: &[u8]) {
            let mut r = Reader::new(nal);
            r.ue();
            r.ue();
            self.entropy = r.flag();
            self.bottom_field_poc = r.flag();
            assert_eq!(r.ue(), 0, "one slice group");
            self.ref_l0 = r.ue();
            self.ref_l1 = r.ue();
            self.weighted_pred = r.flag();
            self.weighted_bipred = r.u(2);
            self.init_qp_minus26 = r.se();
            r.se();
            self.chroma_qp_offset = r.se();
            self.deblocking_control = r.flag();
            self.constrained_intra = r.flag();
            self.redundant_pic_cnt = r.flag();
            self.second_chroma_qp_offset = self.chroma_qp_offset;
            if r.more_data() {
                self.transform_8x8 = r.flag();
                assert!(!r.flag(), "no PPS scaling lists");
                self.second_chroma_qp_offset = r.se();
            }
        }

        /// The pic_order_cnt_lsb of a slice NAL unit, header byte included.
        pub fn poc_lsb(&self, nal: &[u8]) -> u32 {
            let mut r = Reader::new(&nal[1..]);
            r.ue();
            r.ue();
            r.ue();
            r.u(self.log2_max_frame_num_minus4 + 4);
            if nal[0] & 0x1f == 5 {
                r.ue();
            }
            r.u(self.log2_max_poc_lsb_minus4 + 4)
        }

        /// NVDEC's picture parameters for a frame decoding into surface `surface`.
        pub fn picture_info(
            &self,
            stream_len: usize,
            surface: usize,
            frame_number: u32,
        ) -> H264DecoderContext {
            let mut context = H264DecoderContext {
                stream_len: stream_len as u32,
                slice_count: 1,
                weight_scale_4x4: [16; 0x60],
                weight_scale_8x8: [16; 0x80],
                ..Default::default()
            };
            let p = &mut context.parameter_set;
            p.log2_max_pic_order_cnt_lsb_minus4 = self.log2_max_poc_lsb_minus4 as i32;
            p.frame_mbs_only_flag = i32::from(self.frame_mbs_only);
            p.pic_width_in_mbs = self.width_mbs;
            p.frame_height_in_mbs = self.height_mbs;
            p.entropy_coding_mode_flag = u32::from(self.entropy);
            p.pic_order_present_flag = i32::from(self.bottom_field_poc);
            p.num_refidx_l0_default_active = self.ref_l0 as i32;
            p.num_refidx_l1_default_active = self.ref_l1 as i32;
            p.deblocking_filter_control_present_flag = i32::from(self.deblocking_control);
            p.redundant_pic_cnt_present_flag = i32::from(self.redundant_pic_cnt);
            p.transform_8x8_mode_flag = u32::from(self.transform_8x8);
            let signed = |value: i32, bits: u32| (value as u64) & ((1 << bits) - 1);
            p.set_flags(
                u64::from(self.direct_8x8) << 1
                    | u64::from(self.weighted_pred) << 2
                    | u64::from(self.constrained_intra) << 3
                    | u64::from(self.log2_max_frame_num_minus4) << 8
                    | 1 << 12
                    | u64::from(self.poc_type) << 14
                    | signed(self.init_qp_minus26, 6) << 16
                    | signed(self.chroma_qp_offset, 5) << 22
                    | signed(self.second_chroma_qp_offset, 5) << 27
                    | u64::from(self.weighted_bipred) << 32
                    | (surface as u64) << 34
                    | u64::from(frame_number) << 46,
            );
            context
        }
    }

    /// One access unit: its slice NAL units with start codes, and its display position.
    pub struct Frame {
        pub slices: Vec<u8>,
        pub poc: u32,
    }

    /// Splits an Annex B stream into NAL units, start codes removed.
    fn nal_units(bytes: &[u8]) -> Vec<&[u8]> {
        let starts: Vec<usize> = (0..bytes.len().saturating_sub(3))
            .filter(|&i| bytes[i..i + 3] == [0, 0, 1])
            .map(|i| i + 3)
            .collect();
        starts
            .iter()
            .zip(
                starts
                    .iter()
                    .skip(1)
                    .map(|&next| next - 3)
                    .chain([bytes.len()]),
            )
            .map(|(&start, end)| {
                let mut nal = &bytes[start..end];
                while let [rest @ .., 0] = nal {
                    nal = rest;
                }
                nal
            })
            .collect()
    }

    /// The stream's parameters and its frames in decode order.
    pub fn parse(bytes: &[u8]) -> (Params, Vec<Frame>) {
        let mut params = Params::default();
        let mut frames: Vec<Frame> = Vec::new();
        for nal in nal_units(bytes) {
            match nal[0] & 0x1f {
                7 => params.sps(&nal[1..]),
                8 => params.pps(&nal[1..]),
                9 => frames.push(Frame {
                    slices: Vec::new(),
                    poc: 0,
                }),
                kind @ (1 | 5) => {
                    let frame = frames.last_mut().expect("access unit delimiters");
                    if frame.slices.is_empty() {
                        frame.poc = params.poc_lsb(nal) | u32::from(kind == 5) << 31;
                    }
                    frame.slices.extend([0, 0, 1]);
                    frame.slices.extend(nal);
                }
                _ => {}
            }
        }
        frames.retain(|frame| !frame.slices.is_empty());
        (params, frames)
    }
}

/// The display index of each frame of a stream whose frames follow one IDR.
fn display_order(frames: &[h264::Frame]) -> Vec<usize> {
    let pocs: Vec<u32> = frames.iter().map(|f| f.poc & 0x7fff_ffff).collect();
    pocs.iter()
        .map(|poc| pocs.iter().filter(|&&other| other < *poc).count())
        .collect()
}

/// The engine method writes of one NVDEC job decoding into `surface`.
fn nvdec_methods(surface: usize) -> Vec<(u32, u32)> {
    let index = |i: usize| i as u32;
    vec![
        (index(index::SET_CODEC_ID), VideoCodec::H264.raw() as u32),
        (
            index(index::PICTURE_INFO_OFFSET),
            (PICTURE_INFO >> 8) as u32,
        ),
        (
            index(index::FRAME_BITSTREAM_OFFSET),
            (BITSTREAM >> 8) as u32,
        ),
        (
            index(index::SURFACE_LUMA_OFFSETS + surface),
            (surface_luma(surface) >> 8) as u32,
        ),
        (
            index(index::SURFACE_CHROMA_OFFSETS + surface),
            (surface_chroma(surface) >> 8) as u32,
        ),
        (index(index::EXECUTE), 0),
    ]
}

fn store_frame(
    host1x: &Host1x,
    params: &h264::Params,
    frame: &h264::Frame,
    surface: usize,
    number: u32,
) {
    let info: H264DecoderContext = params.picture_info(frame.slices.len(), surface, number);
    host1x
        .memory
        .write_bytes(PICTURE_INFO, bytemuck::bytes_of(&info))
        .unwrap();
    host1x.memory.write_bytes(BITSTREAM, &frame.slices).unwrap();
}

/// Y, U and V of quadrant `index` of frame `frame` of the streams in crates/video/tests/data.
fn stream_quadrant(frame: usize, index: usize) -> [u8; 3] {
    [
        (40 + 30 * frame + 10 * index) as u8,
        (90 + 20 * index) as u8,
        (200 - 15 * frame - 20 * index) as u8,
    ]
}

const QUADRANTS_H264: &[u8] = include_bytes!("../../../video/tests/data/quadrants.h264");

/// Decodes the B-frame stream through an NVDEC channel and composes it through a VIC channel, as a
/// guest player does: a picture is shown, in display order, once one more frame than it has been
/// decoded (one frame of reordering), and the stream's end is flushed by the next stream's IDR
/// frame. Each surface must hold the picture coded for it, whichever order the decoder returned
/// the pictures in.
#[test]
fn h264_decodes_through_nvdec_and_composes_through_vic_by_surface() {
    if !ffmpeg() {
        return;
    }
    let (params, frames) = h264::parse(QUADRANTS_H264);
    assert_eq!(frames.len(), 6);
    let display = display_order(&frames);
    assert_ne!(display, [0, 1, 2, 3, 4, 5], "the stream reorders B-frames");
    let host1x = mapped_host1x();
    let nvdec = EngineChannel::open(VideoEngine::NvDec, &host1x, None);
    let vic = EngineChannel::open(VideoEngine::Vic, &host1x, None);
    let (nvdec_done, vic_done) = (20, 21);
    let (width, height) = (64u64, 48u64);
    let mut config = VicConfig::new(rgba_output(width, height, A8B8G8R8, None));
    let surface = Surface {
        format: Y8_V8U8_N420,
        block_height: Some(1),
        width,
        height,
    };
    config.slot(0, Slot::whole(surface, None));
    host1x.memory.write_bytes(CONFIG, config.bytes()).unwrap();
    let (mut decodes, mut composes) = (0, 0);
    let mut decode = |frame: &h264::Frame, surface: usize| {
        store_frame(&host1x, &params, frame, surface, decodes + 1);
        decodes += 1;
        nvdec.submit(command_buffer(
            VideoEngine::NvDec,
            &nvdec_methods(surface),
            nvdec_done,
        ));
        wait_for(&host1x.syncpoints, nvdec_done, decodes);
    };
    let mut show = |decoded: usize| {
        let methods = vic_methods(&[(0, surface_luma(decoded), surface_chroma(decoded))]);
        composes += 1;
        vic.submit(command_buffer(VideoEngine::Vic, &methods, vic_done));
        wait_for(&host1x.syncpoints, vic_done, composes);
        let out = read(&host1x, OUTPUT_LUMA, (width * height * 4) as usize);
        for (index, (x, y)) in [(16, 12), (48, 12), (16, 36), (48, 36)]
            .into_iter()
            .enumerate()
        {
            let at = (y * width as usize + x) * 4;
            let expected = stream_quadrant(display[decoded], index);
            for (channel, (&got, want)) in out[at..at + 3].iter().zip(expected).enumerate() {
                assert!(
                    got.abs_diff(want) <= 3,
                    "decoded frame {decoded} (display {}) quadrant {index} channel {channel}: {got} vs {want}",
                    display[decoded]
                );
            }
        }
    };
    let mut waiting: Vec<usize> = Vec::new();
    let next_shown = |waiting: &mut Vec<usize>| {
        let at = (0..waiting.len())
            .min_by_key(|&i| display[waiting[i]])
            .unwrap();
        waiting.remove(at)
    };
    for (decoded, frame) in frames.iter().enumerate() {
        decode(frame, decoded);
        waiting.push(decoded);
        if waiting.len() > 1 {
            show(next_shown(&mut waiting));
        }
    }
    decode(&frames[0], frames.len());
    while !waiting.is_empty() {
        show(next_shown(&mut waiting));
    }
}

#[test]
fn steady_state_nvdec_and_vic_jobs_do_not_allocate() {
    if !ffmpeg() {
        return;
    }
    let (params, frames) = h264::parse(QUADRANTS_H264);
    let host1x = mapped_host1x();
    let mut nvdec = Nvdec::new(host1x.memory.clone(), host1x.decoded_frames.clone());
    let mut vic = Vic::new(host1x.memory.clone(), None, host1x.decoded_frames.clone());
    let mut config = VicConfig::new(rgba_output(64, 48, A8B8G8R8, Some(1)));
    config.slot(
        0,
        Slot::whole(
            Surface {
                format: Y8_V8U8_N420,
                block_height: Some(1),
                width: 64,
                height: 48,
            },
            Some(BT601),
        ),
    );
    host1x.memory.write_bytes(CONFIG, config.bytes()).unwrap();
    for pass in 0..3 {
        for (decoded, frame) in frames.iter().enumerate() {
            store_frame(&host1x, &params, frame, decoded, decoded as u32 + 1);
            let decode = nvdec_methods(decoded);
            let compose = vic_methods(&[(0, surface_luma(decoded), surface_chroma(decoded))]);
            let before = ALLOCATIONS.with(Cell::get);
            for &(method, value) in &decode {
                nvdec.write(method, value);
            }
            for &(method, value) in &compose {
                vic.write(method, value);
            }
            if pass == 2 {
                assert_eq!(ALLOCATIONS.with(Cell::get), before, "frame {decoded}");
            }
        }
    }
    assert!(host1x.decoded_frames.len() > 0);
}

/// Decodes a 1280x720 H.264 clip the ffmpeg CLI encodes from its test pattern, through NVDEC and
/// VIC (BT.601 to A8B8G8R8, block-linear), and prints the time per frame. Run in release:
/// `cargo test --release -p lichii-gpu nvdec_vic_720p -- --ignored --nocapture`.
#[test]
#[ignore]
fn nvdec_vic_720p_frame_time() {
    if !ffmpeg() {
        return;
    }
    let path =
        std::env::temp_dir().join(format!("lichii-nvdec-vic-720p-{}.h264", std::process::id()));
    let encoded = std::process::Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30",
        ])
        .args([
            "-frames:v",
            "240",
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
        ])
        .args([
            "-bf",
            "2",
            "-g",
            "60",
            "-b:v",
            "4M",
            "-bsf:v",
            "h264_metadata=aud=insert",
            "-f",
            "h264",
        ])
        .arg(&path)
        .status();
    if !encoded.is_ok_and(|status| status.success()) {
        eprintln!("the ffmpeg CLI with libx264 is not installed; skipping");
        return;
    }
    let bytes = std::fs::read(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let (params, frames) = h264::parse(&bytes);
    let host1x = mapped_host1x();
    let mut nvdec = Nvdec::new(host1x.memory.clone(), host1x.decoded_frames.clone());
    let mut vic = Vic::new(host1x.memory.clone(), None, host1x.decoded_frames.clone());
    let mut config = VicConfig::new(rgba_output(1280, 720, A8B8G8R8, Some(4)));
    config.slot(
        0,
        Slot::whole(
            Surface {
                format: Y8_V8U8_N420,
                block_height: Some(4),
                width: 1280,
                height: 720,
            },
            Some(BT601),
        ),
    );
    host1x.memory.write_bytes(CONFIG, config.bytes()).unwrap();
    let (mut decode, mut compose) = (Duration::ZERO, Duration::ZERO);
    let warmup = 30;
    for (decoded, frame) in frames.iter().enumerate() {
        let surface = decoded % 16;
        store_frame(&host1x, &params, frame, surface, decoded as u32 + 1);
        let started = Instant::now();
        for (method, value) in nvdec_methods(surface) {
            nvdec.write(method, value);
        }
        let decoded_at = Instant::now();
        for (method, value) in vic_methods(&[(0, surface_luma(surface), surface_chroma(surface))]) {
            vic.write(method, value);
        }
        if decoded >= warmup {
            decode += decoded_at - started;
            compose += decoded_at.elapsed();
        }
    }
    let counted = (frames.len() - warmup) as f64;
    let ms = |total: Duration| total.as_secs_f64() * 1000.0 / counted;
    println!(
        "1280x720 H.264, {} frames: NVDEC {:.3} ms + VIC {:.3} ms = {:.3} ms per frame",
        frames.len() - warmup,
        ms(decode),
        ms(compose),
        ms(decode + compose)
    );
}
