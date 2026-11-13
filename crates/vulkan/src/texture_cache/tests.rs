use std::time::Instant;

use lichii_common::settings::GraphicsSettings;
use lichii_gpu::gmmu::BigPageSize;
use lichii_memory::PhysicalMemory;
use lichii_texture::block_linear::{SurfaceLayout, Tiling};
use lichii_texture::tic::TicEntry;
use lichii_texture::{Compression, astc, bcn, etc};

use super::*;
use crate::{Instance, SurfaceSupport};

const DRAM: u64 = 64 << 20;
const DADDR: DAddr = 0x1_0000_0000;
const GPU_VA: GpuVAddr = 0x40_0000_0000 >> 4;

struct Fixture {
    device: Arc<Device>,
    memory: Arc<DeviceAddressSpace>,
    gmmu: Gmmu,
    scheduler: Scheduler,
    staging: StagingRing,
    cache: TextureCache,
}

fn fixture(gob: GobLayout, recompression: AstcRecompression) -> Option<Fixture> {
    let instance = match Instance::new(true, SurfaceSupport::None) {
        Ok(instance) => Arc::new(instance),
        Err(error) => {
            eprintln!("skipping: {error}");
            return None;
        }
    };
    let device = match Device::new(instance, &GraphicsSettings::default(), None) {
        Ok(device) => device,
        Err(error) => {
            eprintln!("skipping: {error}");
            return None;
        }
    };
    let physical = Arc::new(PhysicalMemory::new(DRAM).unwrap());
    physical.commit(0, DRAM).unwrap();
    let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
    memory.map(DADDR, 0, DRAM);
    let gmmu = Gmmu::new(memory.clone(), BigPageSize::Size64K);
    gmmu.map(GPU_VA, DADDR, DRAM, 0, false).unwrap();
    Some(Fixture {
        scheduler: Scheduler::new(&device).unwrap(),
        staging: StagingRing::new(&device, 32 << 20).unwrap(),
        cache: TextureCache::new(
            &device,
            Arc::new(GpuWrittenPages::new(36)),
            Arc::new(lichii_gpu::lazy::LazyDeviceRanges::new(memory.clone())),
            gob,
            recompression,
        ),
        device,
        memory,
        gmmu,
    })
}

impl Fixture {
    fn tx(&mut self) -> (Transfer<'_>, &mut TextureCache, &Gmmu) {
        (
            Transfer {
                scheduler: &mut self.scheduler,
                staging: &mut self.staging,
                memory: &self.memory,
            },
            &mut self.cache,
            &self.gmmu,
        )
    }

    fn texture(&mut self, tic: &TicEntry) -> (ImageId, ViewInfo) {
        let desc = tic.decode().unwrap();
        let (mut tx, cache, gmmu) = self.tx();
        cache
            .texture(&mut tx, gmmu, &desc)
            .unwrap()
            .expect("a cached texture")
    }

    /// Every level of every layer of a colour image as tightly packed host texels, layer-major.
    fn read_image(&mut self, id: ImageId) -> Vec<Vec<u8>> {
        let image = self.cache.get(id).unwrap();
        let (format, host) = (image.info.format, image.host);
        let desc = *image.image.desc();
        let mut subresources = Vec::new();
        let mut offset = 0;
        for layer in 0..desc.array_layers {
            for level in 0..desc.mip_levels {
                let extent = image.host_extent(level);
                let size = convert::host_layout(
                    format,
                    &host,
                    (extent.width, extent.height, extent.depth),
                )
                .main;
                subresources.push((layer, level, extent, offset, size));
                offset = (offset + size).next_multiple_of(16);
            }
        }
        let buffer = Buffer::new(
            &self.device,
            offset as u64,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "readback",
        )
        .unwrap();
        let cmd = self.scheduler.record().unwrap();
        let raw = self
            .cache
            .prepare(cmd, id, ImageAccess::TRANSFER_SRC)
            .expect("a live image");
        let regions: Vec<_> = subresources
            .iter()
            .map(|&(layer, level, extent, offset, _)| vk::BufferImageCopy {
                buffer_offset: offset as u64,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: level,
                    base_array_layer: layer,
                    layer_count: 1,
                },
                image_extent: extent,
                ..Default::default()
            })
            .collect();
        // SAFETY: the image is in TRANSFER_SRC layout and the buffer holds every region.
        unsafe {
            self.device.raw.cmd_copy_image_to_buffer(
                cmd,
                raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.raw(),
                &regions,
            )
        };
        transfer_to_all_barrier(&self.device, cmd);
        self.scheduler.finish().unwrap();
        let data = buffer.mapped().unwrap();
        subresources
            .iter()
            .map(|&(.., offset, size)| data[offset..offset + size].to_vec())
            .collect()
    }

    /// Device memory of the live images.
    fn image_bytes(&self) -> u64 {
        self.cache
            .slots
            .iter()
            .filter_map(|s| s.image.as_ref())
            .map(|i| i.image.memory_size())
            .sum()
    }

    /// Ends a frame once the GPU finished everything, the collector seeing the live images as
    /// the device's whole memory use under `budget`.
    fn tick(&mut self, budget: u64) {
        self.scheduler.finish().unwrap();
        let usage = MemoryUsage {
            used: self.image_bytes(),
            budget,
            ..MemoryUsage::default()
        };
        let (mut tx, cache, _) = self.tx();
        cache.tick_frame(&mut tx, || usage).unwrap();
    }

    fn assert_no_validation_errors(&self) {
        assert_eq!(
            self.device.instance().validation_errors(),
            0,
            "validation reported errors"
        );
    }
}

fn pattern(len: usize, seed: u32) -> Vec<u8> {
    (0..len as u32)
        .map(|i| (i.wrapping_add(seed).wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect()
}

/// A block-linear 2D (array) texture header with identity swizzle.
#[allow(clippy::too_many_arguments)]
fn tic(
    format: u32,
    gpu_va: GpuVAddr,
    width: u32,
    height: u32,
    layers: u32,
    levels: u32,
    block_height_log2: u32,
) -> TicEntry {
    let texture_type = if layers > 1 { 5 } else { 1 };
    TicEntry([
        format | 2 << 7 | 2 << 10 | 2 << 13 | 2 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28,
        gpu_va as u32,
        (gpu_va >> 32) as u32 | 3 << 21,
        block_height_log2 << 3 | (levels - 1) << 28,
        (width - 1) | texture_type << 23,
        (height - 1) | (layers - 1) << 16,
        0,
        (levels - 1) << 4,
    ])
}

#[test]
fn switch2_two_dimensional_headers_ignore_their_base_layer() {
    let mut header = tic(0x08, GPU_VA, 64, 32, 1, 1, 2);
    header.0[2] |= 4 << 16;
    let desc = header.decode().unwrap();
    assert_eq!(desc.base_layer, 32);
    let switch2 = ImageInfo::from_texture(&desc, GobLayout::Switch2).unwrap();
    assert_eq!(switch2.layers, 1);
    assert_eq!(ViewInfo::from_texture(&desc, &switch2).base_layer, 0);
    let tegra = ImageInfo::from_texture(&desc, GobLayout::Tegra).unwrap();
    assert_eq!(tegra.layers, 33, "Tegra headers keep selecting a layer");
    assert_eq!(ViewInfo::from_texture(&desc, &tegra).base_layer, 32);
}

/// Writes `linear` level data into guest memory as the block-linear surface stores it.
fn write_surface(memory: &DeviceAddressSpace, surface: &SurfaceLayout, data: &[Vec<u8>]) {
    let mut i = 0;
    for layer in 0..surface.layers {
        for level in 0..surface.levels {
            let tiling = surface.level_tiling(level);
            let mut tiled = vec![0; tiling.size_bytes()];
            swizzle(&tiling, &data[i], &mut tiled);
            memory
                .write_bytes(DADDR + surface.subresource_offset(level, layer), &tiled)
                .unwrap();
            i += 1;
        }
    }
}

fn level_data(surface: &SurfaceLayout, seed: u32) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    for layer in 0..surface.layers {
        for level in 0..surface.levels {
            let len = surface.level_tiling(level).linear_size_bytes();
            out.push(pattern(len, seed + layer * 97 + level * 13));
        }
    }
    out
}

fn surface(
    format: PixelFormat,
    w: u32,
    h: u32,
    levels: u32,
    layers: u32,
    bh: u32,
    gob: GobLayout,
) -> SurfaceLayout {
    SurfaceLayout {
        format,
        width: w,
        height: h,
        depth: 1,
        levels,
        layers,
        block_height_log2: bh,
        block_depth_log2: 0,
        tile_width_spacing: 0,
        layout: gob,
    }
}

#[test]
fn uploads_block_linear_mips_and_layers_in_both_gob_layouts() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let s = surface(PixelFormat::A8B8G8R8Unorm, 70, 40, 4, 3, 2, gob);
        let data = level_data(&s, 7);
        write_surface(&f.memory, &s, &data);
        let (id, view) = f.texture(&tic(0x08, GPU_VA, 70, 40, 3, 4, 2));
        assert_eq!((view.levels, view.layers), (4, 3));
        assert_eq!(view.view_type, vk::ImageViewType::TYPE_2D_ARRAY);
        let read = f.read_image(id);
        assert_eq!(read.len(), 12);
        for (i, (got, want)) in read.iter().zip(&data).enumerate() {
            assert!(
                got == want,
                "{gob:?}: layer {} level {} differs",
                i / 4,
                i % 4
            );
        }
        assert!(f.cache.view(id, &view).unwrap().is_some());
        let again = f.texture(&tic(0x08, GPU_VA, 70, 40, 3, 4, 2));
        assert_eq!(again.0, id, "a second lookup reuses the image");
        f.assert_no_validation_errors();
    }
}

#[test]
fn decodes_compressed_formats_the_host_cannot_sample() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    for (raw, format) in [
        (0x06, PixelFormat::Etc2RgbUnorm),
        (0x44, PixelFormat::Astc8x8Unorm),
    ] {
        let host = host_format(&f.device, format, AstcRecompression::Uncompressed);
        if host.conversion != Conversion::Decode {
            eprintln!("skipping {format:?}: the device samples it natively");
            continue;
        }
        let (w, h) = (40, 24);
        let s = surface(format, w, h, 2, 1, 1, GobLayout::Tegra);
        let data = level_data(&s, raw);
        write_surface(&f.memory, &s, &data);
        let (id, _) = f.texture(&tic(raw, GPU_VA, w, h, 1, 2, 1));
        let read = f.read_image(id);
        for (level, (got, blocks)) in read.iter().zip(&data).enumerate() {
            let (lw, lh) = ((w >> level).max(1), (h >> level).max(1));
            let mut want = vec![0; (lw * lh * 4) as usize];
            if format.is_astc() {
                astc::decode_rgba8(blocks, lw, lh, 1, 8, 8, false, &mut want);
            } else {
                etc::decode_etc2_rgb(blocks, lw, lh, 1, &mut want).unwrap();
            }
            assert!(*got == want, "{format:?} level {level} decodes differently");
        }
        f.cache.unmap(&mut f.scheduler, DADDR, 1 << 20);
        assert!(f.cache.is_empty());
    }
    f.assert_no_validation_errors();
}

#[test]
fn recompresses_astc_to_bc1_and_bc3() {
    for (recompression, bc) in [
        (AstcRecompression::Bc1, Compression::Bc1),
        (AstcRecompression::Bc3, Compression::Bc3),
    ] {
        let Some(mut f) = fixture(GobLayout::Switch2, recompression) else {
            return;
        };
        let host = host_format(&f.device, PixelFormat::Astc4x4Unorm, recompression);
        if host.conversion != Conversion::DecodeToBc(bc) {
            eprintln!("skipping: ASTC is not recompressed on this device ({host:?})");
            return;
        }
        let (w, h) = (36, 20);
        let s = surface(PixelFormat::Astc4x4Unorm, w, h, 1, 1, 0, GobLayout::Switch2);
        let data = level_data(&s, 3);
        write_surface(&f.memory, &s, &data);
        let (id, _) = f.texture(&tic(0x40, GPU_VA, w, h, 1, 1, 0));
        let read = f.read_image(id);
        let mut rgba = vec![0; (w * h * 4) as usize];
        astc::decode_rgba8(&data[0], w, h, 1, 4, 4, false, &mut rgba);
        let block_bytes = if bc == Compression::Bc1 { 8 } else { 16 };
        let mut want = vec![0; (w.div_ceil(4) * h.div_ceil(4)) as usize * block_bytes];
        if bc == Compression::Bc1 {
            bcn::encode_bc1(&rgba, w, h, 1, &mut want).unwrap();
        } else {
            bcn::encode_bc3(&rgba, w, h, 1, &mut want).unwrap();
        }
        assert!(read[0] == want, "{bc:?} blocks differ");
        f.assert_no_validation_errors();
    }
}

#[test]
fn converts_formats_only_a_fallback_host_format_holds() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let expand4 = |v: u32| (v * 17) as u8;
    let mut tested = 0;
    for (raw, format) in [
        (0x1e, PixelFormat::G4R4Unorm),
        (0x12, PixelFormat::A4B4G4R4Unorm),
        (0x15, PixelFormat::B5G6R5Unorm),
    ] {
        let host = host_format(&f.device, format, AstcRecompression::Uncompressed);
        if host.conversion != Conversion::Convert {
            eprintln!("{format:?} needs no conversion here ({:?})", host.format);
            continue;
        }
        tested += 1;
        let (w, h) = (24, 16);
        let s = surface(format, w, h, 1, 1, 1, GobLayout::Tegra);
        let data = level_data(&s, raw);
        write_surface(&f.memory, &s, &data);
        let (id, _) = f.texture(&tic(raw, GPU_VA, w, h, 1, 1, 1));
        let read = f.read_image(id);
        let want: Vec<u8> = match format {
            PixelFormat::G4R4Unorm => data[0]
                .iter()
                .flat_map(|&v| [expand4(u32::from(v) & 15), expand4(u32::from(v) >> 4)])
                .collect(),
            PixelFormat::A4B4G4R4Unorm => data[0]
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&t| {
                    let v = u32::from(u16::from_le_bytes(t));
                    [0, 4, 8, 12].map(|s| expand4((v >> s) & 15))
                })
                .collect(),
            _ => data[0]
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&t| {
                    let v = u32::from(u16::from_le_bytes(t));
                    let e = |x: u32, max: u32| ((x * 255 + max / 2) / max) as u8;
                    [e(v & 31, 31), e((v >> 5) & 63, 63), e(v >> 11, 31), 255]
                })
                .collect(),
        };
        assert!(read[0] == want, "{format:?} converts differently");
        f.cache.unmap(&mut f.scheduler, DADDR, 1 << 20);
    }
    if tested == 0 {
        eprintln!("skipping: every candidate format is native on this device");
    }
    f.assert_no_validation_errors();
}

#[test]
fn cpu_writes_re_upload_on_next_use() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let s = surface(
        PixelFormat::A8B8G8R8Unorm,
        64,
        32,
        1,
        1,
        2,
        GobLayout::Tegra,
    );
    let first = level_data(&s, 1);
    write_surface(&f.memory, &s, &first);
    let header = tic(0x08, GPU_VA, 64, 32, 1, 1, 2);
    let (id, _) = f.texture(&header);
    assert!(f.read_image(id)[0] == first[0]);

    let second = level_data(&s, 2);
    write_surface(&f.memory, &s, &second);
    let (same, _) = f.texture(&header);
    assert_eq!(same, id);
    assert!(
        f.read_image(id)[0] == first[0],
        "without an invalidation the cached copy stays"
    );

    {
        let (mut tx, cache, _) = f.tx();
        cache.invalidate(&mut tx, DADDR + 100, 4).unwrap();
    }
    assert!(f.cache.state(id).unwrap().cpu_dirty);
    let (same, _) = f.texture(&header);
    assert_eq!(same, id);
    assert!(f.read_image(id)[0] == second[0], "the image re-uploaded");
    assert!(!f.cache.state(id).unwrap().cpu_dirty);

    {
        let (mut tx, cache, _) = f.tx();
        cache.invalidate(&mut tx, DADDR + (1 << 20), 4).unwrap();
    }
    assert!(
        !f.cache.state(id).unwrap().cpu_dirty,
        "a write elsewhere leaves it"
    );
    f.assert_no_validation_errors();
}

fn rt_info(width: u32, height: u32, bh: u32, gob: GobLayout) -> ImageInfo {
    let s = surface(PixelFormat::A8B8G8R8Unorm, width, height, 1, 1, bh, gob);
    ImageInfo {
        format: PixelFormat::A8B8G8R8Unorm,
        kind: ImageKind::D2,
        layout: GuestLayout::BlockLinear {
            block_height_log2: bh,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            gob,
        },
        width,
        height,
        depth: 1,
        levels: 1,
        layers: 1,
        samples: 1,
        layer_stride: s.layer_stride(),
        rescaleable: height > 288,
        downscaleable: height > 512,
    }
}

fn clear_rt(f: &mut Fixture, info: &ImageInfo, color: [f32; 4], rect: Rect) -> Found {
    clear_rt_at(f, GPU_VA, info, color, rect)
}

fn clear_rt_at(
    f: &mut Fixture,
    gpu_va: GpuVAddr,
    info: &ImageInfo,
    color: [f32; 4],
    rect: Rect,
) -> Found {
    let (mut tx, cache, gmmu) = f.tx();
    let target = cache
        .render_target(&mut tx, gmmu, gpu_va, info, rect == Rect::ALL)
        .unwrap()
        .unwrap();
    cache
        .clear(
            &mut tx,
            target,
            ClearValue::Color {
                value: vk::ClearColorValue { float32: color },
                mask: vk::ColorComponentFlags::RGBA,
            },
            rect,
            0,
            1,
        )
        .unwrap();
    target
}

#[test]
fn cpu_writes_into_gpu_modified_images_keep_the_other_gpu_texels() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(64, 64, 3, GobLayout::Tegra);
    let target = clear_rt(&mut f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    f.memory.write_bytes(DADDR + 8, &[9; 4]).unwrap();
    {
        let (mut tx, cache, _) = f.tx();
        cache.invalidate(&mut tx, DADDR + 8, 4).unwrap();
    }
    let state = f.cache.state(target.id).unwrap();
    assert!(state.cpu_dirty && !state.gpu_modified);
    let mut guest = vec![0; info.size_bytes() as usize];
    f.memory.read_bytes(DADDR, &mut guest).unwrap();
    for (i, texel) in guest.as_chunks::<4>().0.iter().enumerate() {
        let expected = if i == 2 { [9; 4] } else { [255, 0, 0, 255] };
        assert_eq!(*texel, expected, "guest texel at byte {}", i * 4);
    }
    let (id, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 1, 3));
    assert_eq!(id, target.id);
    let texels = f.read_image(id);
    assert_eq!(
        texels[0]
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|t| **t == [9; 4])
            .count(),
        1
    );
    f.assert_no_validation_errors();
}

#[test]
fn render_targets_sampled_later_reuse_the_gpu_written_image() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(64, 64, 3, GobLayout::Tegra);
    let target = clear_rt(&mut f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let (id, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 1, 3));
    assert_eq!(id, target.id);
    assert!(
        f.read_image(id)[0]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [255, 0, 0, 255]),
        "sampled from the cleared image, not re-uploaded from guest memory"
    );
    let r32f = TicEntry({
        let mut w = tic(0x0f, GPU_VA, 64, 64, 1, 1, 3).0;
        w[0] = 0x0f | 7 << 7 | 7 << 10 | 7 << 13 | 7 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28;
        w
    });
    let (same, view) = f.texture(&r32f);
    assert_eq!(same, target.id, "a same-size format views the same image");
    assert!(f.cache.view(same, &view).unwrap().is_some());
    f.assert_no_validation_errors();
}

#[test]
fn incompatible_overlaps_stay_live_and_take_newer_texels_through_guest_memory() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(64, 64, 3, GobLayout::Tegra);
    let target = clear_rt(&mut f, &info, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    assert_eq!(info.size_bytes(), 16 << 10);
    let (id, _) = f.texture(&tic(0x08, GPU_VA, 128, 32, 1, 1, 2));
    assert_ne!(id, target.id);
    assert!(
        f.cache.state(target.id).is_some(),
        "the render target stays live"
    );
    assert_eq!(f.cache.len(), 2);
    let mut guest = vec![0; 16 << 10];
    f.memory.read_bytes(DADDR, &mut guest).unwrap();
    assert!(
        guest
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [0, 255, 0, 255]),
        "flushed first"
    );
    assert!(
        f.read_image(id)[0]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [0, 255, 0, 255])
    );
    f.assert_no_validation_errors();
}

const GREEN: [u8; 4] = [0, 255, 0, 255];
const RED: [u8; 4] = [255, 0, 0, 255];

fn guest_texels(f: &Fixture, offset: u64, len: usize) -> Vec<u8> {
    let mut guest = vec![0; len];
    f.memory.read_bytes(DADDR + offset, &mut guest).unwrap();
    guest
}

fn count_texels(texels: &[u8], expected: [u8; 4]) -> usize {
    texels
        .as_chunks::<4>()
        .0
        .iter()
        .filter(|t| **t == expected)
        .count()
}

#[test]
fn targets_cleared_whole_over_overlaps_skip_the_guest_round_trip() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    assert_eq!(small.size_bytes(), 4 << 10);
    let before = f.cache.traffic();
    let a = clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let b = clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let again = clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let b_again = clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    assert_eq!((again.id, b_again.id), (a.id, b.id), "both stay live");
    assert_eq!(
        f.cache.traffic(),
        before,
        "no write-back or upload between whole clears"
    );
    let (mut tx, cache, _) = f.tx();
    cache.flush(&mut tx, DADDR, 16 << 10).unwrap();
    assert!(
        all_texels(&guest_texels(&f, 0, 4 << 10), RED),
        "the newer target's bytes"
    );
    assert!(all_texels(&guest_texels(&f, 4 << 10, 12 << 10), GREEN));
    f.assert_no_validation_errors();
}

#[test]
fn write_backs_leave_the_bytes_of_newer_overlaps_alone() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let (mut tx, cache, _) = f.tx();
    cache.flush(&mut tx, DADDR + (8 << 10), 4 << 10).unwrap();
    assert_eq!(
        count_texels(&guest_texels(&f, 0, 4 << 10), GREEN),
        0,
        "the older target skips the newer one's bytes"
    );
    assert!(all_texels(&guest_texels(&f, 4 << 10, 12 << 10), GREEN));
    let (mut tx, cache, _) = f.tx();
    cache.flush(&mut tx, DADDR, 4 << 10).unwrap();
    assert!(all_texels(&guest_texels(&f, 0, 4 << 10), RED));
    assert!(all_texels(&guest_texels(&f, 4 << 10, 12 << 10), GREEN));
    f.assert_no_validation_errors();
}

#[test]
fn write_backs_leave_the_bytes_of_removed_newer_overlaps_alone() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let b = clear_rt_at(
        &mut f,
        GPU_VA + (8 << 10),
        &big,
        [1.0, 0.0, 0.0, 1.0],
        Rect::ALL,
    );
    let (mut tx, cache, _) = f.tx();
    cache.flush(&mut tx, DADDR + (16 << 10), 8 << 10).unwrap();
    assert!(all_texels(&guest_texels(&f, 8 << 10, 16 << 10), RED));
    let chain = ImageInfo { levels: 2, ..big };
    let (mut tx, cache, gmmu) = f.tx();
    let found = cache
        .find(&mut tx, gmmu, GPU_VA + (8 << 10), &chain)
        .unwrap()
        .unwrap();
    assert!(
        f.cache.state(b.id).is_none(),
        "the written-back target is a level of the new chain"
    );
    assert!(all_texels(&guest_texels(&f, 0, 8 << 10), GREEN));
    assert!(
        all_texels(&guest_texels(&f, 8 << 10, 16 << 10), RED),
        "the older target's write-back skips the removed one's bytes"
    );
    assert!(all_texels(&f.read_image(found.id)[0], RED));
    f.assert_no_validation_errors();
}

#[test]
fn stale_overlaps_are_not_joined_over_newer_whole_clears() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let chain = ImageInfo { levels: 2, ..small };
    let (mut tx, cache, gmmu) = f.tx();
    let found = cache.find(&mut tx, gmmu, GPU_VA, &chain).unwrap().unwrap();
    assert!(
        all_texels(&f.read_image(found.id)[0], GREEN),
        "the newer whole clear, not the stale small target"
    );
    let (mut tx, cache, _) = f.tx();
    cache.flush(&mut tx, DADDR, 16 << 10).unwrap();
    assert!(all_texels(&guest_texels(&f, 0, 16 << 10), GREEN));
    f.assert_no_validation_errors();
}

#[test]
fn stale_overlaps_take_newer_texels_before_they_are_drawn_into() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    let a = clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let b = clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let (mut tx, cache, gmmu) = f.tx();
    let found = cache
        .render_target(&mut tx, gmmu, GPU_VA, &big, false)
        .unwrap()
        .unwrap();
    assert_eq!(found.id, a.id);
    let read = f.read_image(a.id);
    assert_eq!(
        count_texels(&read[0], RED),
        32 * 32,
        "the small target's texels"
    );
    assert_eq!(count_texels(&read[0], GREEN), 64 * 64 - 32 * 32);
    assert!(all_texels(&f.read_image(b.id)[0], RED));
    f.assert_no_validation_errors();
}

#[test]
fn mip_chains_take_in_a_render_target_level_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let s = surface(
        PixelFormat::A8B8G8R8Unorm,
        64,
        64,
        3,
        1,
        3,
        GobLayout::Tegra,
    );
    let data = level_data(&s, 5);
    write_surface(&f.memory, &s, &data);
    let info = rt_info(64, 64, 3, GobLayout::Tegra);
    let target = clear_rt(&mut f, &info, [0.0, 0.0, 1.0, 1.0], Rect::ALL);
    let (id, view) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 3, 3));
    assert_ne!(id, target.id);
    assert_eq!(view.levels, 3);
    let read = f.read_image(id);
    assert!(
        read[0]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [0, 0, 255, 255]),
        "level 0 copied on the GPU"
    );
    assert!(
        read[1] == data[1] && read[2] == data[2],
        "other levels from guest memory"
    );
    assert!(f.cache.state(id).unwrap().gpu_modified);
    let mut guest = vec![0; 4];
    f.memory.read_bytes(DADDR, &mut guest).unwrap();
    assert_ne!(guest, [0, 0, 255, 255], "guest memory was not written");
    f.assert_no_validation_errors();
}

#[test]
fn clears_flush_as_swizzled_texels_and_scaled_targets_downscale() {
    for scale in [
        ResolutionScale::X1,
        ResolutionScale::X2,
        ResolutionScale::X1_2,
    ] {
        let gob = GobLayout::Switch2;
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        f.cache.set_resolution_scale(scale);
        let (w, h) = (64, 640);
        let info = rt_info(w, h, 4, gob);
        clear_rt(&mut f, &info, [1.0, 0.5, 0.25, 0.0], Rect::ALL);
        let inner = Rect {
            x: 8,
            y: 16,
            width: 16,
            height: 32,
        };
        let target = clear_rt(&mut f, &info, [0.0, 0.0, 0.0, 1.0], inner);
        let state = f.cache.state(target.id).unwrap();
        assert_eq!(state.scale, scale);
        assert_eq!(
            (state.extent.width, state.extent.height),
            (scale.scale(w), scale.scale(h))
        );
        let (mut tx, cache, _) = f.tx();
        cache.flush(&mut tx, DADDR, info.size_bytes()).unwrap();
        assert!(!f.cache.state(target.id).unwrap().gpu_modified);
        let tiling = Tiling::new(w, h, 1, 4, 4, 0, gob);
        let mut tiled = vec![0; tiling.size_bytes()];
        f.memory.read_bytes(DADDR, &mut tiled).unwrap();
        let mut linear = vec![0; tiling.linear_size_bytes()];
        unswizzle(&tiling, &tiled, &mut linear);
        let near_edge = |x: u32, y: u32| {
            let close = |v: u32, edges: [u32; 2]| edges.iter().any(|&e| v.abs_diff(e) <= 1);
            scale < ResolutionScale::X1 && (close(x, [8, 24]) || close(y, [16, 48]))
        };
        for (i, texel) in linear.as_chunks::<4>().0.iter().enumerate() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            if near_edge(x, y) {
                continue;
            }
            let inside = (8..24).contains(&x) && (16..48).contains(&y);
            let want: [u8; 4] = if inside {
                [0, 0, 0, 255]
            } else {
                [255, 128, 64, 0]
            };
            assert_eq!(*texel, want, "{scale:?}: texel ({x}, {y})");
        }
        f.assert_no_validation_errors();
    }
}

#[test]
fn depth_targets_clear_and_flush() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    for (format, check) in [
        (PixelFormat::D32Float, 0usize),
        (PixelFormat::S8UintD24Unorm, 1),
    ] {
        let mut info = rt_info(32, 32, 2, GobLayout::Tegra);
        info.format = format;
        let (mut tx, cache, gmmu) = f.tx();
        let target = cache
            .render_target(&mut tx, gmmu, GPU_VA, &info, true)
            .unwrap()
            .unwrap();
        cache
            .clear(
                &mut tx,
                target,
                ClearValue::DepthStencil {
                    depth: Some(0.5),
                    stencil: Some(0x5a),
                    stencil_mask: 0xff,
                },
                Rect::ALL,
                0,
                1,
            )
            .unwrap();
        cache.flush(&mut tx, DADDR, info.size_bytes()).unwrap();
        let mut guest = vec![0; 32 * 32 * 4];
        f.memory.read_bytes(DADDR, &mut guest).unwrap();
        for &texel in guest.as_chunks::<4>().0 {
            let v = u32::from_le_bytes(texel);
            if check == 0 {
                assert_eq!(f32::from_bits(v), 0.5);
            } else {
                assert_eq!(v & 0xff, 0x5a, "stencil in the low byte");
                assert!((v >> 8).abs_diff(0x80_0000) <= 1, "depth {:#x}", v >> 8);
            }
        }
        f.cache.unmap(&mut f.scheduler, DADDR, 1 << 20);
    }
    f.assert_no_validation_errors();
}

#[test]
fn guest_descriptors_describe_render_targets_depth_and_framebuffers() {
    use lichii_gpu::engines::maxwell_3d::regs::{
        RenderTargetConfig, TileMode, Zeta, ZetaSize, ZetaSizeBits,
    };
    let mut rt: RenderTargetConfig = bytemuck::Zeroable::zeroed();
    rt.address_low = 0x10_0000;
    rt.width = 1280;
    rt.height = 720;
    rt.format = 0xd5;
    rt.tile_mode = TileMode(4 << 4);
    rt.bits = lichii_gpu::engines::maxwell_3d::regs::RenderTargetConfigBits(3);
    rt.array_pitch = 0x10_0000 >> 2;
    let info = ImageInfo::from_render_target(&rt, 2, GobLayout::Switch2).unwrap();
    assert_eq!(info.format, PixelFormat::A8B8G8R8Unorm);
    assert_eq!(
        (info.width, info.height, info.layers, info.samples),
        (2560, 1440, 3, 4)
    );
    assert_eq!(info.layer_stride, 0x10_0000);
    assert!(info.rescaleable && info.downscaleable);
    rt.height = 240;
    let small = ImageInfo::from_render_target(&rt, 0, GobLayout::Tegra).unwrap();
    assert!(!small.rescaleable, "240 rows stay native");
    rt.format = 0;
    assert!(ImageInfo::from_render_target(&rt, 0, GobLayout::Tegra).is_none());

    let mut zeta: Zeta = bytemuck::Zeroable::zeroed();
    zeta.address_low = 0x20_0000;
    zeta.format = 0x14;
    zeta.tile_mode = TileMode(3 << 4);
    let size = ZetaSize {
        width: 640,
        height: 360,
        bits: ZetaSizeBits(1 | 1 << 16),
    };
    let depth = ImageInfo::from_zeta(&zeta, &size, 0, GobLayout::Tegra).unwrap();
    assert_eq!(depth.format, PixelFormat::S8UintD24Unorm);
    assert_eq!((depth.width, depth.height, depth.layers), (640, 360, 1));
    assert!(depth.rescaleable && !depth.downscaleable);

    let fb = FramebufferConfig {
        address: DADDR,
        width: 1280,
        height: 720,
        stride: 1280 * 4,
        format: PixelFormat::A8B8G8R8Unorm,
        block_height_log2: Some(4),
        flip_vertical: false,
        crop: None,
        gob_layout: GobLayout::Switch2,
    };
    let info = ImageInfo::from_framebuffer(&fb);
    assert_eq!((info.width, info.height), (1280, 720));
    assert_eq!(
        info.size_bytes(),
        Tiling::new(1280, 720, 1, 4, 4, 0, GobLayout::Switch2).size_bytes() as u64
    );
    let pitch = ImageInfo::from_framebuffer(&FramebufferConfig {
        block_height_log2: None,
        stride: 1280 * 4 + 256,
        gob_layout: GobLayout::Tegra,
        ..fb
    });
    assert_eq!(
        pitch.layout,
        GuestLayout::Pitch {
            pitch: 1280 * 4 + 256
        }
    );
    assert_eq!(pitch.size_bytes(), (1280 * 4 + 256) * 720);
}

#[test]
fn measures_upload_and_lookup_cost() {
    let Some(mut f) = fixture(GobLayout::Switch2, AstcRecompression::Uncompressed) else {
        return;
    };
    let s = surface(
        PixelFormat::A8B8G8R8Unorm,
        1024,
        1024,
        1,
        1,
        4,
        GobLayout::Switch2,
    );
    let data = level_data(&s, 9);
    write_surface(&f.memory, &s, &data);
    let header = tic(0x08, GPU_VA, 1024, 1024, 1, 1, 4);
    let (id, _) = f.texture(&header);
    f.scheduler.finish().unwrap();
    let rounds = 20;
    let start = Instant::now();
    for _ in 0..rounds {
        {
            let (mut tx, cache, _) = f.tx();
            cache.invalidate(&mut tx, DADDR, 4).unwrap();
        }
        f.texture(&header);
    }
    f.scheduler.finish().unwrap();
    let upload = start.elapsed();
    let mb = (rounds * 4) as f64;
    let desc = header.decode().unwrap();
    let lookups = 100_000;
    let start = Instant::now();
    for _ in 0..lookups {
        let (mut tx, cache, gmmu) = f.tx();
        let found = cache.texture(&mut tx, gmmu, &desc).unwrap().unwrap();
        assert_eq!(found.0, id);
    }
    let lookup = start.elapsed();
    let info = ImageInfo::from_texture(&desc, GobLayout::Switch2).unwrap();
    let start = Instant::now();
    for _ in 0..lookups {
        let (mut tx, cache, gmmu) = f.tx();
        let found = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
        assert_eq!(found.id, id);
    }
    let find = start.elapsed();
    eprintln!(
        "upload 1024x1024 RGBA8 (Switch 2 GOBs): {:.0} MB/s; cached texture lookup from a \
         descriptor: {:.0} ns; cached find: {:.0} ns",
        mb / upload.as_secs_f64(),
        lookup.as_nanos() as f64 / f64::from(lookups),
        find.as_nanos() as f64 / f64::from(lookups)
    );
    assert!(f.read_image(id)[0] == data[0]);
    f.assert_no_validation_errors();
}

const HUGE_BUDGET: u64 = 1 << 40;

#[test]
fn collector_thresholds_follow_the_budget_and_vram_setting() {
    const GIB: u64 = 1 << 30;
    let t = |budget, vram| Thresholds::new(budget, vram);
    let want = |expected, critical, retain| Thresholds {
        expected,
        critical,
        retain,
    };
    assert_eq!(
        t(8 * GIB, VramUsage::Conservative),
        want(4 * GIB, 5 * GIB, 7 * GIB)
    );
    assert_eq!(
        t(8 * GIB, VramUsage::Aggressive),
        want(6 * GIB, 7 * GIB, 7 * GIB)
    );
    assert_eq!(
        t(12 * GIB, VramUsage::Conservative),
        want(13 * GIB / 2, 15 * GIB / 2, 11 * GIB),
        "large budgets collect from 5/8 of them, not from a fixed ~7 GB cap"
    );
    assert_eq!(
        t(4 << 20, VramUsage::Conservative),
        want(3 << 20, 7 << 19, 7 << 19),
        "small budgets keep an eighth free"
    );
}

#[test]
fn collector_evicts_least_recently_used_images_first() {
    let Some(mut f) = fixture(GobLayout::Switch2, AstcRecompression::Uncompressed) else {
        return;
    };
    let tics: Vec<_> = (0..12)
        .map(|i| tic(0x08, GPU_VA + i * 0x1_0000, 64, 64, 1, 1, 2))
        .collect();
    let ids: Vec<_> = tics.iter().map(|t| f.texture(t).0).collect();
    let size = f.cache.get(ids[0]).unwrap().image.memory_size();
    assert!(
        ids.iter()
            .all(|&id| f.cache.get(id).unwrap().image.memory_size() == size)
    );
    for _ in 0..OLD_AGE {
        for t in &tics[6..] {
            f.texture(t);
        }
        f.tick(HUGE_BUDGET);
    }
    assert_eq!(f.cache.collector_stats(), CollectorStats::default());

    for t in &tics[6..] {
        f.texture(t);
    }
    // 12 images under a 14-image budget: past the expected line (10.5 images), under the
    // critical one (12.25), so only images unused for OLD_AGE frames go until use drops below
    // the expected line.
    f.tick(14 * size);
    assert_eq!(
        f.cache.collector_stats(),
        CollectorStats {
            evicted: 2,
            evicted_bytes: 2 * size,
            flushed: 0,
        }
    );
    let live =
        |f: &Fixture| -> Vec<bool> { ids.iter().map(|&id| f.cache.get(id).is_some()).collect() };
    let mut want = vec![true; 12];
    want[..2].fill(false);
    assert_eq!(live(&f), want, "the two least recently used images went");

    for t in &tics[6..] {
        f.texture(t);
    }
    // Far past the budget: aggressive, every image unused for RECENT_AGE frames goes, and none
    // used this frame.
    f.tick(size);
    assert_eq!(f.cache.collector_stats().evicted, 6);
    assert_eq!(f.cache.collector_stats().evicted_bytes, 6 * size);
    want[..6].fill(false);
    assert_eq!(live(&f), want, "images bound this frame stay");
    f.assert_no_validation_errors();
}

#[test]
fn collector_flushes_gpu_modified_images_before_evicting_them() {
    let gob = GobLayout::Switch2;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(64, 64, 2, gob);
    let tiling = Tiling::new(64, 64, 1, 4, 2, 0, gob);
    let other_va = GPU_VA + 0x10_0000;
    let evicted = clear_rt_at(&mut f, GPU_VA, &info, [0.0, 1.0, 0.0, 1.0], Rect::ALL).id;
    let kept = clear_rt_at(&mut f, other_va, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL).id;
    let guest = |f: &Fixture, daddr: DAddr| {
        let mut tiled = vec![0; tiling.size_bytes()];
        f.memory.read_bytes(daddr, &mut tiled).unwrap();
        let mut linear = vec![0; tiling.linear_size_bytes()];
        unswizzle(&tiling, &tiled, &mut linear);
        linear
    };
    for _ in 0..RECENT_AGE {
        f.tick(HUGE_BUDGET);
    }
    f.tick(1);
    assert_eq!(f.cache.pending_downloads.len(), 2, "both readbacks queued");
    assert_eq!(f.cache.collector_stats(), CollectorStats::default());
    assert!(f.cache.state(evicted).is_some() && f.cache.state(kept).is_some());
    assert!(
        guest(&f, DADDR).iter().all(|&b| b == 0),
        "queued, not yet written back"
    );

    let (again, _) = f.texture(&tic(0x08, other_va, 64, 64, 1, 1, 2));
    assert_eq!(again, kept, "used again while its readback runs");
    f.tick(1);
    let size = info.size_bytes();
    assert_eq!(
        f.cache.collector_stats(),
        CollectorStats {
            evicted: 1,
            evicted_bytes: f.cache.stats.evicted_bytes,
            flushed: 1,
        }
    );
    assert!(f.cache.collector_stats().evicted_bytes >= size);
    assert!(f.cache.state(evicted).is_none());
    assert!(
        f.cache.state(kept).is_some_and(|s| s.gpu_modified),
        "an image used again stays, still GPU-modified"
    );
    assert!(
        guest(&f, DADDR)
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [0, 255, 0, 255]),
        "the evicted image reached guest memory first"
    );
    assert!(guest(&f, DADDR + 0x10_0000).iter().all(|&b| b == 0));

    let (reloaded, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 1, 2));
    assert_ne!(reloaded, evicted);
    assert!(
        f.read_image(reloaded)[0]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [0, 255, 0, 255]),
        "re-uploaded from guest memory"
    );
    f.assert_no_validation_errors();
}

#[test]
fn collector_work_per_frame_is_bounded() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    for i in 0..50 {
        f.texture(&tic(0x08, GPU_VA + i * 0x1000, 16, 16, 1, 1, 0));
    }
    let info = rt_info(512, 512, 4, gob);
    assert_eq!(info.size_bytes(), 1 << 20);
    for i in 0..24 {
        clear_rt_at(
            &mut f,
            GPU_VA + (i + 1) * 0x10_0000,
            &info,
            [0.0, 0.0, 1.0, 1.0],
            Rect::ALL,
        );
    }
    for _ in 0..RECENT_AGE {
        f.tick(HUGE_BUDGET);
    }
    let stats = |f: &Fixture| {
        let s = f.cache.collector_stats();
        (s.evicted, s.flushed, f.cache.pending_downloads.len())
    };
    f.tick(1);
    assert_eq!(
        stats(&f),
        (AGGRESSIVE_EVICTIONS as u64, 0, 0),
        "at most AGGRESSIVE_EVICTIONS images a frame"
    );
    f.tick(1);
    let per_frame = (MAX_FLUSH_BYTES >> 20) as usize;
    assert_eq!(
        stats(&f),
        (50, 0, per_frame),
        "the last 10 clean images go, and MAX_FLUSH_BYTES of readbacks start"
    );
    f.tick(1);
    assert_eq!(
        stats(&f),
        (50 + per_frame as u64, per_frame as u64, 24 - per_frame)
    );
    f.tick(1);
    assert_eq!(stats(&f), (74, 24, 0));
    assert_eq!(f.cache.len(), 0);
    f.assert_no_validation_errors();
}

fn pitch_target(pitch: u32, height: u32) -> ImageInfo {
    use lichii_gpu::engines::maxwell_3d::regs::{RenderTargetConfig, TileMode};
    let mut rt: RenderTargetConfig = bytemuck::Zeroable::zeroed();
    rt.address_low = 0x10_0000;
    rt.width = pitch;
    rt.height = height;
    rt.format = 0xd5;
    rt.tile_mode = TileMode(1 << 12);
    ImageInfo::from_render_target(&rt, 0, GobLayout::Tegra).unwrap()
}

/// A pitch-linear 2D A8B8G8R8 texture header with identity swizzle.
fn pitch_tic(gpu_va: GpuVAddr, width: u32, height: u32, pitch: u32) -> TicEntry {
    let mut w = tic(0x08, gpu_va, width, height, 1, 1, 0).0;
    w[2] = (gpu_va >> 32) as u32 | 2 << 21;
    w[3] = pitch >> 5;
    TicEntry(w)
}

fn all_texels(texels: &[u8], expected: [u8; 4]) -> bool {
    texels.as_chunks::<4>().0.iter().all(|t| *t == expected)
}

#[test]
fn pitch_overlaps_map_shared_texels_into_both_images() {
    let wide = pitch_target(256, 8);
    let narrow = ImageInfo { width: 60, ..wide };
    assert_eq!(
        pitch_overlap(&wide, DADDR, &narrow, DADDR),
        Some(((0, 0), (0, 0), (60, 8)))
    );
    let below = ImageInfo { width: 16, ..wide };
    assert_eq!(
        pitch_overlap(&wide, DADDR, &below, DADDR + 2 * 256 + 8 * 4),
        Some(((8, 2), (0, 0), (16, 6)))
    );
    assert_eq!(
        pitch_overlap(&below, DADDR + 2 * 256 + 8 * 4, &wide, DADDR),
        Some(((0, 0), (8, 2), (16, 6)))
    );
    assert_eq!(
        pitch_overlap(&wide, DADDR, &below, DADDR + 60 * 4),
        None,
        "rows of the later image would wrap past the pitch"
    );
    assert_eq!(pitch_overlap(&wide, DADDR, &narrow, DADDR + 2), None);
    let other_pitch = pitch_target(512, 8);
    assert_eq!(pitch_overlap(&wide, DADDR, &other_pitch, DADDR), None);
}

#[test]
fn narrower_pitch_textures_alias_their_render_target_without_a_flush() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = pitch_target(256, 8);
    assert_eq!(info.width, 64);
    let target = clear_rt(&mut f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let header = pitch_tic(GPU_VA, 60, 8, 256);
    let (texture, _) = f.texture(&header);
    assert_ne!(texture, target.id);
    assert!(
        f.cache.state(target.id).is_some(),
        "the render target stays"
    );
    assert_eq!(
        f.cache.traffic().downloads,
        0,
        "no write-back to guest memory"
    );
    assert!(all_texels(&f.read_image(texture)[0], [255, 0, 0, 255]));

    let again = clear_rt(&mut f, &info, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    assert_eq!(again.id, target.id);
    assert_eq!(f.texture(&header).0, texture);
    assert!(
        all_texels(&f.read_image(texture)[0], [0, 255, 0, 255]),
        "the texture takes the render target's newer texels"
    );
    assert_eq!(f.cache.traffic().downloads, 0);

    clear_rt(&mut f, &info, [0.0, 0.0, 1.0, 1.0], Rect::ALL);
    f.memory.write_bytes(DADDR + 62 * 4, &[7; 4]).unwrap();
    {
        let (mut tx, cache, _) = f.tx();
        cache.invalidate(&mut tx, DADDR + 62 * 4, 4).unwrap();
    }
    let (refreshed, _) = f.texture(&header);
    assert_eq!(refreshed, texture);
    assert!(
        all_texels(&f.read_image(texture)[0], [0, 0, 255, 255]),
        "a CPU write next to the texture keeps the render target's texels in it"
    );
    let mut guest = vec![0; info.size_bytes() as usize];
    f.memory.read_bytes(DADDR, &mut guest).unwrap();
    for (i, texel) in guest.as_chunks::<4>().0.iter().enumerate() {
        let expected = if i == 62 { [7; 4] } else { [0, 0, 255, 255] };
        assert_eq!(*texel, expected, "guest texel {i}");
    }
    f.assert_no_validation_errors();
}

#[test]
fn flushing_aliases_writes_the_newest_texels_back() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let wide = pitch_target(256, 8);
    let narrow = ImageInfo { width: 60, ..wide };
    let flushed = |f: &mut Fixture| {
        {
            let (mut tx, cache, _) = f.tx();
            cache.flush(&mut tx, DADDR, wide.size_bytes()).unwrap();
        }
        let mut guest = vec![0; wide.size_bytes() as usize];
        f.memory.read_bytes(DADDR, &mut guest).unwrap();
        guest
    };

    let wide_id = clear_rt(&mut f, &wide, [1.0, 0.0, 0.0, 1.0], Rect::ALL).id;
    let narrow_id = clear_rt(&mut f, &narrow, [0.0, 1.0, 0.0, 1.0], Rect::ALL).id;
    assert_ne!(wide_id, narrow_id);
    for (i, texel) in flushed(&mut f).as_chunks::<4>().0.iter().enumerate() {
        let expected = if i % 64 < 60 {
            [0, 255, 0, 255]
        } else {
            [255, 0, 0, 255]
        };
        assert_eq!(
            *texel, expected,
            "guest texel {i} with the narrow target newer"
        );
    }

    clear_rt(&mut f, &narrow, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    clear_rt(&mut f, &wide, [0.0, 0.0, 1.0, 1.0], Rect::ALL);
    for (i, texel) in flushed(&mut f).as_chunks::<4>().0.iter().enumerate() {
        assert_eq!(
            *texel,
            [0, 0, 255, 255],
            "guest texel {i} with the wide target newer"
        );
    }
    assert!(!f.cache.state(narrow_id).unwrap().gpu_modified);
    assert!(
        all_texels(&f.read_image(narrow_id)[0], [0, 0, 255, 255]),
        "the flush hands the newer texels to the narrow target"
    );
    f.assert_no_validation_errors();
}

#[test]
fn removed_aliases_hand_their_texels_to_the_other_image() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = pitch_target(256, 8);
    let target = clear_rt(&mut f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let header = pitch_tic(GPU_VA, 60, 8, 256);
    let (texture, _) = f.texture(&header);
    clear_rt(&mut f, &info, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    {
        let (tx, cache, _) = f.tx();
        cache.remove(tx.scheduler, target.id);
    }
    assert!(
        all_texels(&f.read_image(texture)[0], [0, 255, 0, 255]),
        "the texture holds the dropped render target's last texels"
    );
    assert!(f.cache.state(texture).unwrap().gpu_modified);
    f.assert_no_validation_errors();
}

fn texel_rows(texels: &[u8], width: usize) -> Vec<&[[u8; 4]]> {
    texels.as_chunks::<4>().0.chunks(width).collect()
}

#[test]
fn rejected_overlaps_keep_the_lookups_alias_rectangles() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let small = ImageInfo {
        width: 16,
        ..pitch_target(256, 8)
    };
    let left = clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL).id;
    let right = clear_rt_at(&mut f, GPU_VA + 32, &small, [0.0, 0.0, 1.0, 1.0], Rect::ALL).id;
    assert_ne!(left, right);
    let strip = ImageInfo {
        width: 60,
        height: 2,
        ..small
    };
    clear_rt_at(
        &mut f,
        GPU_VA + 256,
        &strip,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    );
    assert!(
        f.cache.state(right).is_some(),
        "the overlap that wraps the pitch stays live beside the strip"
    );
    let (texture, _) = f.texture(&pitch_tic(GPU_VA, 16, 8, 256));
    assert_eq!(texture, left);
    for (y, row) in texel_rows(&f.read_image(texture)[0], 16).iter().enumerate() {
        for (x, texel) in row.iter().enumerate() {
            let expected = match (x, y) {
                (_, 1..=2) => [0, 255, 0, 255],
                (8.., _) => [0, 0, 255, 255],
                _ => [255, 0, 0, 255],
            };
            assert_eq!(*texel, expected, "texel ({x}, {y})");
        }
    }
    f.assert_no_validation_errors();
}

#[test]
fn partial_flushes_keep_every_aliased_rectangle() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let wide = pitch_target(256, 8);
    let strip = ImageInfo {
        width: 60,
        height: 2,
        ..wide
    };
    let target = clear_rt(&mut f, &wide, [1.0, 0.0, 0.0, 1.0], Rect::ALL).id;
    clear_rt(&mut f, &strip, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    clear_rt_at(
        &mut f,
        GPU_VA + 4 * 256,
        &strip,
        [0.0, 0.0, 1.0, 1.0],
        Rect::ALL,
    );
    {
        let (mut tx, cache, _) = f.tx();
        cache
            .flush(&mut tx, DADDR + 4 * 256, strip.size_bytes())
            .unwrap();
    }
    let expected = |x: usize, y: usize| match (x, y) {
        (0..60, 0..=1) => [0, 255, 0, 255],
        (0..60, 4..=5) => [0, 0, 255, 255],
        _ => [255, 0, 0, 255],
    };
    let (texture, _) = f.texture(&pitch_tic(GPU_VA, 64, 8, 256));
    assert_eq!(texture, target);
    for (y, row) in texel_rows(&f.read_image(texture)[0], 64).iter().enumerate() {
        for (x, texel) in row.iter().enumerate() {
            assert_eq!(*texel, expected(x, y), "texel ({x}, {y})");
        }
    }
    let mut guest = vec![0; strip.size_bytes() as usize];
    f.memory.read_bytes(DADDR + 4 * 256, &mut guest).unwrap();
    for (y, row) in texel_rows(&guest, 64).iter().enumerate() {
        for (x, texel) in row.iter().enumerate() {
            assert_eq!(*texel, expected(x, y + 4), "guest texel ({x}, {})", y + 4);
        }
    }
    f.assert_no_validation_errors();
}

#[test]
fn presented_targets_take_their_aliases_texels() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let wide = pitch_target(256, 8);
    let narrow = ImageInfo { width: 60, ..wide };
    let target = clear_rt(&mut f, &wide, [1.0, 0.0, 0.0, 1.0], Rect::ALL).id;
    clear_rt(&mut f, &narrow, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let fb = FramebufferConfig {
        address: DADDR,
        width: 64,
        height: 8,
        stride: 256,
        format: PixelFormat::A8B8G8R8Unorm,
        block_height_log2: None,
        flip_vertical: false,
        crop: None,
        gob_layout: GobLayout::Tegra,
    };
    let presented = f.cache.presentable(&mut f.scheduler, &fb).unwrap();
    assert_eq!(presented, Some(target));
    for (y, row) in texel_rows(&f.read_image(target)[0], 64).iter().enumerate() {
        for (x, texel) in row.iter().enumerate() {
            let expected = if x < 60 {
                [0, 255, 0, 255]
            } else {
                [255, 0, 0, 255]
            };
            assert_eq!(*texel, expected, "texel ({x}, {y})");
        }
    }
    f.assert_no_validation_errors();
}

#[test]
fn handed_off_aliases_keep_newer_writes_in_other_aliases() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let wide = pitch_target(256, 8);
    let strip = ImageInfo {
        width: 60,
        height: 2,
        ..wide
    };
    clear_rt(&mut f, &wide, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let top = clear_rt(&mut f, &strip, [0.0, 1.0, 0.0, 1.0], Rect::ALL).id;
    let bottom = clear_rt_at(
        &mut f,
        GPU_VA + 4 * 256,
        &strip,
        [0.0, 0.0, 1.0, 1.0],
        Rect::ALL,
    )
    .id;
    {
        let (tx, cache, _) = f.tx();
        cache.remove(tx.scheduler, bottom);
    }
    let found = {
        let (mut tx, cache, gmmu) = f.tx();
        cache
            .render_target(&mut tx, gmmu, GPU_VA, &strip, false)
            .unwrap()
            .unwrap()
    };
    assert_eq!(found.id, top);
    assert!(
        all_texels(&f.read_image(top)[0], [0, 255, 0, 255]),
        "the top strip keeps its own newer texels"
    );
    f.assert_no_validation_errors();
}

#[test]
fn depth_and_stencil_views_read_green_as_red() {
    use SwizzleSource::{A, B, G, OneFloat, R, Zero};
    let shadow = [G, Zero, Zero, OneFloat];
    for aspect in [
        vk::ImageAspectFlags::DEPTH,
        vk::ImageAspectFlags::STENCIL,
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
    ] {
        assert_eq!(sampled_swizzle(shadow, aspect), [R, Zero, Zero, OneFloat]);
        assert_eq!(sampled_swizzle([R, G, B, A], aspect), [R, R, B, A]);
    }
    assert_eq!(sampled_swizzle(shadow, vk::ImageAspectFlags::COLOR), shadow);
}
