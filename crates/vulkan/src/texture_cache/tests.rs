use std::time::Instant;

use lichii_common::settings::GraphicsSettings;
use lichii_gpu::backend::{CopySurface, ImageCopy};
use lichii_gpu::gmmu::BigPageSize;
use lichii_memory::PhysicalMemory;
use lichii_texture::block_linear::{SurfaceLayout, Tiling};
use lichii_texture::tic::{TextureType, TicEntry};
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
    buffers: BufferCache,
}

fn fixture(gob: GobLayout, recompression: AstcRecompression) -> Option<Fixture> {
    fixture_with_dram(gob, recompression, DRAM)
}

fn fixture_with_dram(
    gob: GobLayout,
    recompression: AstcRecompression,
    dram: u64,
) -> Option<Fixture> {
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
    let physical = Arc::new(PhysicalMemory::new(dram).unwrap());
    physical.commit(0, dram).unwrap();
    let memory = Arc::new(DeviceAddressSpace::new(physical, 36));
    memory.map(DADDR, 0, dram);
    let gmmu = Gmmu::new(memory.clone(), BigPageSize::Size64K);
    gmmu.map(GPU_VA, DADDR, dram, 0, false).unwrap();
    let written = Arc::new(GpuWrittenPages::new(36));
    let lazy = Arc::new(lichii_gpu::lazy::LazyDeviceRanges::new(memory.clone()));
    Some(Fixture {
        scheduler: Scheduler::new(&device).unwrap(),
        staging: StagingRing::new(&device, 32 << 20).unwrap(),
        buffers: BufferCache::new(&device, memory.clone(), written.clone(), lazy.clone()).unwrap(),
        cache: TextureCache::new(&device, written, lazy, gob, recompression),
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
                buffers: &mut self.buffers,
            },
            &mut self.cache,
            &self.gmmu,
        )
    }

    fn texture(&mut self, tic: &TicEntry) -> (ImageId, ViewInfo) {
        let desc = tic.decode().unwrap();
        let (mut tx, cache, gmmu) = self.tx();
        cache
            .texture(&mut tx, gmmu, &desc, Usage::Sample)
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
            .prepare(&mut self.scheduler, id, ImageAccess::TRANSFER_SRC)
            .unwrap()
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

    /// One aspect of level 0, layer 0 as tightly packed texels of `bytes` bytes each.
    fn read_plane(&mut self, id: ImageId, aspect: vk::ImageAspectFlags, bytes: u32) -> Vec<u8> {
        let (w, h, _) = self.cache.get(id).unwrap().info.level_extent(0);
        let size = u64::from(w * h * bytes).next_multiple_of(4);
        let buffer = Buffer::new(
            &self.device,
            size,
            vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuToCpu,
            "plane readback",
        )
        .unwrap();
        let cmd = self.scheduler.record().unwrap();
        let raw = self
            .cache
            .prepare(&mut self.scheduler, id, ImageAccess::TRANSFER_SRC)
            .unwrap()
            .expect("a live image");
        let region = vk::BufferImageCopy {
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_extent: vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            },
            ..Default::default()
        };
        // SAFETY: the image is in TRANSFER_SRC layout and the buffer holds the region.
        unsafe {
            self.device.raw.cmd_copy_image_to_buffer(
                cmd,
                raw,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.raw(),
                &[region],
            )
        };
        transfer_to_all_barrier(&self.device, cmd);
        self.scheduler.finish().unwrap();
        buffer.mapped().unwrap()[..(w * h * bytes) as usize].to_vec()
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

#[test]
fn switch2_array_headers_ignore_their_base_layer() {
    let mut header = tic(0x08, GPU_VA, 64, 32, 4, 1, 2);
    header.0[2] |= 4 << 16;
    let desc = header.decode().unwrap();
    assert_eq!(
        (desc.texture_type, desc.base_layer),
        (TextureType::D2Array, 32)
    );
    let switch2 = ImageInfo::from_texture(&desc, GobLayout::Switch2).unwrap();
    assert_eq!(switch2.layers, 4);
    let view = ViewInfo::from_texture(&desc, &switch2);
    assert_eq!((view.base_layer, view.layers), (0, 4));
    let tegra = ImageInfo::from_texture(&desc, GobLayout::Tegra).unwrap();
    assert_eq!(
        tegra.layers, 36,
        "Tegra array headers keep their base layer"
    );
    let view = ViewInfo::from_texture(&desc, &tegra);
    assert_eq!((view.base_layer, view.layers), (32, 4));
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
fn recompresses_astc_to_bc1_and_bc3_on_the_cpu() {
    for (recompression, bc) in [
        (AstcRecompression::Bc1, Compression::Bc1),
        (AstcRecompression::Bc3, Compression::Bc3),
    ] {
        let Some(mut f) = fixture(GobLayout::Switch2, recompression) else {
            return;
        };
        f.cache.set_astc_decode(AstcDecode::Cpu);
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
fn incompatible_overlaps_stay_live_and_new_images_take_newer_texels_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(64, 64, 3, GobLayout::Tegra);
    let target = clear_rt(&mut f, &info, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    assert_eq!(info.size_bytes(), 16 << 10);
    let traffic = f.cache.traffic();
    let (id, _) = f.texture(&tic(0x08, GPU_VA, 128, 32, 1, 1, 2));
    let after = f.cache.traffic();
    assert_ne!(id, target.id);
    assert!(
        f.cache.state(target.id).is_some(),
        "the render target stays live"
    );
    assert_eq!(f.cache.len(), 2);
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
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
const BLUE: [u8; 4] = [0, 0, 255, 255];

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
fn targets_drawn_into_keep_their_own_texels_over_newer_overlaps() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    let a = clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let b = clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let before = f.cache.traffic();
    let (mut tx, cache, gmmu) = f.tx();
    let found = cache
        .render_target(&mut tx, gmmu, GPU_VA, &big, false)
        .unwrap()
        .unwrap();
    assert_eq!(found.id, a.id);
    assert_eq!(f.cache.traffic(), before, "no sync, upload or write-back");
    assert!(
        all_texels(&f.read_image(a.id)[0], GREEN),
        "the target's own texels"
    );
    assert!(all_texels(&f.read_image(b.id)[0], RED));
    let (id, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 1, 3));
    assert_eq!(id, a.id);
    assert_eq!(
        f.cache.traffic().gpu_syncs,
        before.gpu_syncs + 1,
        "a bind with no draw leaves the newer overlap to the next read"
    );
    assert_eq!(count_texels(&f.read_image(a.id)[0], RED), 32 * 32);
    f.assert_no_validation_errors();
}

#[test]
fn targets_drawn_into_take_newer_overlaps_on_their_texel_grid() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let one = rt_info(32, 32, 2, GobLayout::Tegra);
    let two = ImageInfo { layers: 2, ..one };
    let a = clear_rt(&mut f, &one, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let b = clear_rt(&mut f, &two, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    assert_ne!(a.id, b.id);
    let (mut tx, cache, gmmu) = f.tx();
    let found = cache
        .render_target(&mut tx, gmmu, GPU_VA, &one, false)
        .unwrap()
        .unwrap();
    let read = f.read_image(found.id);
    assert!(
        all_texels(&read[found.layer as usize], RED),
        "the newer layered target's texels"
    );
    f.assert_no_validation_errors();
}

#[test]
fn depth_arrays_cleared_a_layer_at_a_time_join_into_one_image() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let format = PixelFormat::X8D24Unorm;
    let array = ImageInfo {
        format,
        layers: 4,
        layer_stride: surface(format, 64, 64, 1, 4, 2, GobLayout::Tegra).layer_stride(),
        ..rt_info(64, 64, 2, GobLayout::Tegra)
    };
    let depths = [0.25, 0.5, 0.75, 1.0];
    let mut last = None;
    for (layer, depth) in (0u32..).zip(depths) {
        let (mut tx, cache, gmmu) = f.tx();
        let target = cache
            .render_target(
                &mut tx,
                gmmu,
                GPU_VA,
                &array.with_layers(layer + 1),
                layer == 0,
            )
            .unwrap()
            .unwrap();
        cache
            .clear(
                &mut tx,
                target,
                ClearValue::DepthStencil {
                    depth: Some(depth),
                    stencil: None,
                    stencil_mask: 0xff,
                },
                Rect::ALL,
                layer,
                1,
            )
            .unwrap();
        last = Some(target.id);
    }
    assert_eq!(f.cache.len(), 1, "each larger array joins the smaller one");
    let traffic = f.cache.traffic();
    for layers in 1..=4 {
        let (mut tx, cache, gmmu) = f.tx();
        let found = cache
            .render_target(&mut tx, gmmu, GPU_VA, &array.with_layers(layers), false)
            .unwrap()
            .unwrap();
        assert_eq!((Some(found.id), found.layer), (last, 0));
    }
    assert_eq!(f.cache.traffic(), traffic, "binds sync nothing");
    let (mut tx, cache, _) = f.tx();
    cache.flush(&mut tx, DADDR, array.size_bytes()).unwrap();
    let stride = array.layer_stride as usize;
    let guest = guest_texels(&f, 0, stride * 4);
    for (layer, depth) in depths.into_iter().enumerate() {
        let expected = (f64::from(depth) * 16_777_215.0).round() as u32;
        for &texel in guest[layer * stride..][..64 * 64 * 4].as_chunks::<4>().0 {
            let d = u32::from_le_bytes(texel) & 0xff_ffff;
            assert!(d.abs_diff(expected) <= 1, "layer {layer} depth {d:#x}");
        }
    }
    f.assert_no_validation_errors();
}

#[test]
fn targets_drawn_into_take_newer_mip_chains_over_them() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    let b = clear_rt(&mut f, &small, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let (chain, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 3, 3));
    assert_ne!(chain, b.id);
    let level = clear_rt(
        &mut f,
        &rt_info(64, 64, 3, GobLayout::Tegra),
        [1.0, 0.0, 0.0, 1.0],
        Rect::ALL,
    );
    assert_eq!(level.id, chain, "the chain's level 0 is drawn into");
    let (mut tx, cache, gmmu) = f.tx();
    let found = cache
        .render_target(&mut tx, gmmu, GPU_VA, &small, false)
        .unwrap()
        .unwrap();
    assert_eq!(found.id, b.id);
    assert!(
        all_texels(&f.read_image(b.id)[0], RED),
        "the chain's newer texels"
    );
    f.assert_no_validation_errors();
}

#[test]
fn new_targets_take_newer_overlaps_before_they_are_drawn_into() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    let a = clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let corner = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    };
    let b = clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], corner);
    assert_ne!(a.id, b.id);
    let read = f.read_image(b.id);
    assert_eq!(count_texels(&read[0], RED), 8 * 8);
    assert_eq!(
        count_texels(&read[0], GREEN),
        32 * 32 - 8 * 8,
        "the older target's texels"
    );
    f.assert_no_validation_errors();
}

#[test]
fn sampled_images_still_take_newer_overlaps() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let big = rt_info(64, 64, 3, GobLayout::Tegra);
    let small = rt_info(32, 32, 2, GobLayout::Tegra);
    let a = clear_rt(&mut f, &big, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let b = clear_rt(&mut f, &small, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let syncs = f.cache.traffic().gpu_syncs;
    let (id, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 1, 3));
    assert_eq!(id, a.id);
    assert_eq!(f.cache.traffic().gpu_syncs, syncs + 1);
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
fn clears_of_a_whole_level_take_no_overlap_bytes_inside_it() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let (chain, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 2, 3));
    let level_zero = rt_info(64, 64, 3, GobLayout::Tegra);
    let overlap = rt_info(128, 32, 2, GobLayout::Tegra);
    assert_eq!(overlap.size_bytes(), level_zero.size_bytes());
    clear_rt(&mut f, &overlap, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let syncs = f.cache.traffic().gpu_syncs;
    let level = clear_rt(&mut f, &level_zero, [0.0, 0.0, 1.0, 1.0], Rect::ALL);
    assert_eq!((level.id, level.level), (chain, 0));
    assert_eq!(
        f.cache.traffic().gpu_syncs,
        syncs,
        "the overlap's bytes all lie in the cleared level"
    );
    assert!(all_texels(&f.read_image(chain)[0], BLUE));
    f.assert_no_validation_errors();
}

#[test]
fn clears_of_a_whole_level_still_take_overlap_bytes_past_it() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let (chain, _) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 2, 3));
    clear_rt(
        &mut f,
        &rt_info(128, 64, 3, GobLayout::Tegra),
        [1.0, 0.0, 0.0, 1.0],
        Rect::ALL,
    );
    let syncs = f.cache.traffic().gpu_syncs;
    let level = clear_rt(
        &mut f,
        &rt_info(64, 64, 3, GobLayout::Tegra),
        [0.0, 0.0, 1.0, 1.0],
        Rect::ALL,
    );
    assert_eq!((level.id, level.level), (chain, 0));
    assert_eq!(f.cache.traffic().gpu_syncs, syncs + 1);
    let read = f.read_image(chain);
    assert!(all_texels(&read[0], BLUE));
    assert!(
        all_texels(&read[1], RED),
        "level 1 takes the overlap's bytes"
    );
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
        clear_rt(&mut f, &info, [1.0, 128.0 / 255.0, 0.25, 0.0], Rect::ALL);
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
        let found = cache
            .texture(&mut tx, gmmu, &desc, Usage::Sample)
            .unwrap()
            .unwrap();
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

#[test]
fn images_over_the_texel_limit_are_sparse_and_bind_what_the_gpu_writes() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = ImageInfo {
        layers: 65,
        ..rt_info(2048, 2048, 4, GobLayout::Tegra)
    };
    assert!(info.texels() > MAX_TEXELS);
    let layer = 2048 * 2048 * 4;
    let (mut tx, cache, gmmu) = f.tx();
    if !cache.can_be_sparse(&info) {
        eprintln!("skipping: no sparse residency");
        return;
    }
    let found = cache
        .find(&mut tx, gmmu, GPU_VA, &info)
        .unwrap()
        .expect("huge images are cached, not refused");
    let memory = |cache: &TextureCache| cache.get(found.id).unwrap().image.memory_size();
    assert!(cache.get(found.id).unwrap().image.is_sparse());
    assert_eq!(
        memory(cache),
        0,
        "nothing is uploaded or bound for sampling"
    );
    let desc = tic(0x08, GPU_VA, 2048, 2048, 65, 1, 4).decode().unwrap();
    assert_eq!(
        cache
            .texture(&mut tx, gmmu, &desc, Usage::Sample)
            .unwrap()
            .map(|(id, _)| id),
        Some(found.id),
        "storage writes find the image to bind"
    );
    assert!(
        cache.unwritten(found.id),
        "an image nothing wrote samples as no image"
    );
    cache.bind_written(&mut tx, found.id, 0, 3..4).unwrap();
    assert_eq!(memory(cache), layer, "the written layer");
    assert!(!cache.unwritten(found.id));
    cache.bind_written(&mut tx, found.id, 0, 3..5).unwrap();
    assert_eq!(memory(cache), 2 * layer, "only the layer not bound yet");
    assert!(!cache.get(found.id).unwrap().downloadable());
    let (mut tx, cache, _) = f.tx();
    cache
        .clear(
            &mut tx,
            Found {
                id: found.id,
                level: 0,
                layer: 0,
            },
            ClearValue::Color {
                value: vk::ClearColorValue {
                    float32: [0.0, 1.0, 0.0, 1.0],
                },
                mask: vk::ColorComponentFlags::RGBA,
            },
            Rect {
                x: 0,
                y: 0,
                width: 16,
                height: 16,
            },
            4,
            1,
        )
        .unwrap();
    cache.bind_written(&mut tx, found.id, 0, 0..6).unwrap();
    assert_eq!(
        cache.get(found.id).unwrap().image.memory_size(),
        6 * layer,
        "the layers around the bound ones"
    );
    let buffer = Buffer::new(
        &f.device,
        32 * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "readback",
    )
    .unwrap();
    let cmd = f.scheduler.record().unwrap();
    let raw = f
        .cache
        .prepare(&mut f.scheduler, found.id, ImageAccess::TRANSFER_SRC)
        .unwrap()
        .unwrap();
    let region = vk::BufferImageCopy {
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 4,
            layer_count: 1,
        },
        image_extent: vk::Extent3D {
            width: 32,
            height: 1,
            depth: 1,
        },
        ..Default::default()
    };
    // SAFETY: the image is a transfer source and the buffer holds the row.
    unsafe {
        f.device.raw.cmd_copy_image_to_buffer(
            cmd,
            raw,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.raw(),
            &[region],
        )
    };
    f.scheduler.finish().unwrap();
    let row = buffer.mapped().unwrap();
    assert!(all_texels(&row[..16 * 4], GREEN), "the cleared texels");
    assert!(
        row[16 * 4..].iter().all(|&b| b == 0),
        "the rest of the bound layer reads zero"
    );
    f.assert_no_validation_errors();
}

#[test]
fn sparse_images_take_nothing_from_overlaps_through_guest_memory() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = ImageInfo {
        layers: 65,
        ..rt_info(2048, 2048, 4, GobLayout::Tegra)
    };
    let (mut tx, cache, gmmu) = f.tx();
    if !cache.can_be_sparse(&info) {
        eprintln!("skipping: no sparse residency");
        return;
    }
    let sparse = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
    let small = rt_info(64, 64, 4, GobLayout::Tegra);
    for frame in 0..3 {
        let color = [frame as f32 / 2.0, 0.0, 1.0, 1.0];
        clear_rt_at(&mut f, GPU_VA + 0x10000, &small, color, Rect::ALL);
        let (mut tx, cache, gmmu) = f.tx();
        let again = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
        assert_eq!(again.id, sparse.id);
    }
    assert_eq!(
        f.cache.traffic().downloads,
        0,
        "render targets inside a sparse image are not written back to refresh it"
    );
    f.assert_no_validation_errors();
}

#[test]
fn sparse_images_bind_the_layers_they_join() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let single = rt_info(2048, 2048, 4, GobLayout::Tegra);
    let info = ImageInfo {
        layers: 65,
        ..single
    };
    if !f.cache.can_be_sparse(&info) {
        eprintln!("skipping: no sparse residency");
        return;
    }
    let target = clear_rt(&mut f, &single, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let (mut tx, cache, gmmu) = f.tx();
    let array = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
    assert_ne!(array.id, target.id);
    cache.joins.push((target.id, 0, 0));
    cache.join(&mut tx, array.id).unwrap();
    assert!(
        cache.get(target.id).is_none(),
        "the target joined the array"
    );
    assert_eq!(
        cache.get(array.id).unwrap().image.memory_size(),
        2048 * 2048 * 4,
        "the joined layer has memory"
    );
    assert!(all_texels(
        &f.read_image(array.id)[0][..64],
        [0, 255, 0, 255]
    ));
    f.assert_no_validation_errors();
}

#[test]
fn sparse_images_take_the_layers_of_a_join_their_cap_admits() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let source = ImageInfo {
        layers: 4,
        ..rt_info(1024, 1024, 4, GobLayout::Tegra)
    };
    let info = ImageInfo {
        layers: 300,
        ..source
    };
    let layer = 1024 * 1024 * 4;
    if !f.cache.can_be_sparse(&info) {
        eprintln!("skipping: no sparse residency");
        return;
    }
    f.cache.sparse_cap = 2 * layer;
    let (mut tx, cache, gmmu) = f.tx();
    let target = cache
        .render_target(&mut tx, gmmu, GPU_VA, &source, true)
        .unwrap()
        .unwrap();
    cache
        .clear(
            &mut tx,
            target,
            ClearValue::Color {
                value: vk::ClearColorValue {
                    float32: [0.0, 1.0, 0.0, 1.0],
                },
                mask: vk::ColorComponentFlags::RGBA,
            },
            Rect::ALL,
            0,
            4,
        )
        .unwrap();
    let array = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
    assert_ne!(array.id, target.id);
    cache.joins.push((target.id, 0, 0));
    cache.join(&mut tx, array.id).unwrap();
    assert_eq!(
        cache.get(array.id).unwrap().image.memory_size(),
        2 * layer,
        "the cap admits two layers"
    );
    assert!(
        cache.get(target.id).is_some(),
        "the source keeps the layers past the cap"
    );
    let again = cache.find(&mut tx, gmmu, GPU_VA, &source).unwrap().unwrap();
    assert_eq!(again.id, target.id);
    assert!(
        all_texels(&f.read_image(target.id)[3][..64], [0, 255, 0, 255]),
        "a layer past the cap survives the source's next lookup"
    );
    assert_eq!(f.cache.traffic().uploads, 0);
    let buffer = Buffer::new(
        &f.device,
        16 * 4,
        vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuToCpu,
        "readback",
    )
    .unwrap();
    let cmd = f.scheduler.record().unwrap();
    let raw = f
        .cache
        .prepare(&mut f.scheduler, array.id, ImageAccess::TRANSFER_SRC)
        .unwrap()
        .unwrap();
    let region = vk::BufferImageCopy {
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 1,
            layer_count: 1,
        },
        image_extent: vk::Extent3D {
            width: 16,
            height: 1,
            depth: 1,
        },
        ..Default::default()
    };
    // SAFETY: the image is a transfer source and the buffer holds the row.
    unsafe {
        f.device.raw.cmd_copy_image_to_buffer(
            cmd,
            raw,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer.raw(),
            &[region],
        )
    };
    f.scheduler.finish().unwrap();
    assert!(
        all_texels(&buffer.mapped().unwrap()[..16 * 4], [0, 255, 0, 255]),
        "a bound layer holds the join's texels"
    );
    f.assert_no_validation_errors();
}

fn blit_side(gpu_va: GpuVAddr, info: ImageInfo, rect: [i32; 4]) -> BlitSide {
    BlitSide { gpu_va, info, rect }
}

#[test]
fn blits_scale_between_render_targets_on_the_gpu() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let src = rt_info(64, 64, 3, gob);
    let dst = rt_info(32, 32, 2, gob);
    let dst_va = GPU_VA + (64 << 10);
    clear_rt(&mut f, &src, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let quarter = Rect {
        x: 0,
        y: 0,
        width: 32,
        height: 32,
    };
    clear_rt(&mut f, &src, [1.0, 0.0, 0.0, 1.0], quarter);
    let before = f.cache.traffic();
    let (mut tx, cache, gmmu) = f.tx();
    let blitted = cache
        .blit(
            &mut tx,
            gmmu,
            &blit_side(GPU_VA, src, [0, 0, 64, 64]),
            &blit_side(dst_va, dst, [0, 0, 32, 32]),
            vk::Filter::NEAREST,
        )
        .unwrap();
    assert!(blitted);
    assert_eq!(
        f.cache.traffic(),
        before,
        "no upload or write-back: the source stays on the GPU and the whole target is replaced"
    );
    let (mut tx, cache, _) = f.tx();
    cache
        .flush(&mut tx, DADDR + (64 << 10), dst.size_bytes())
        .unwrap();
    let tiling = Tiling::new(32, 32, 1, 4, 2, 0, gob);
    let mut tiled = vec![0; tiling.size_bytes()];
    f.memory.read_bytes(DADDR + (64 << 10), &mut tiled).unwrap();
    let mut linear = vec![0; tiling.linear_size_bytes()];
    unswizzle(&tiling, &tiled, &mut linear);
    for (i, texel) in linear.as_chunks::<4>().0.iter().enumerate() {
        let (x, y) = (i % 32, i / 32);
        let want = if x < 16 && y < 16 { RED } else { GREEN };
        assert_eq!(*texel, want, "texel ({x}, {y})");
    }
    f.assert_no_validation_errors();
}

#[test]
fn linear_blits_average_the_source_texels_on_the_gpu() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let src = rt_info(64, 64, 3, gob);
    let dst = rt_info(32, 32, 2, gob);
    let dst_va = GPU_VA + (64 << 10);
    clear_rt(&mut f, &src, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let column = Rect {
        x: 0,
        y: 0,
        width: 1,
        height: 64,
    };
    clear_rt(&mut f, &src, [1.0, 0.0, 0.0, 1.0], column);
    let before = f.cache.traffic();
    let (mut tx, cache, gmmu) = f.tx();
    let blitted = cache
        .blit(
            &mut tx,
            gmmu,
            &blit_side(GPU_VA, src, [0, 0, 64, 64]),
            &blit_side(dst_va, dst, [0, 0, 32, 32]),
            vk::Filter::LINEAR,
        )
        .unwrap();
    assert!(blitted);
    assert_eq!(f.cache.traffic(), before, "the downscale stays on the GPU");
    let (mut tx, cache, _) = f.tx();
    cache
        .flush(&mut tx, DADDR + (64 << 10), dst.size_bytes())
        .unwrap();
    let tiling = Tiling::new(32, 32, 1, 4, 2, 0, gob);
    let mut tiled = vec![0; tiling.size_bytes()];
    f.memory.read_bytes(DADDR + (64 << 10), &mut tiled).unwrap();
    let mut linear = vec![0; tiling.linear_size_bytes()];
    unswizzle(&tiling, &tiled, &mut linear);
    for (i, texel) in linear.as_chunks::<4>().0.iter().enumerate() {
        let (x, y) = (i % 32, i / 32);
        let want = if x == 0 { [128, 128, 0, 255] } else { GREEN };
        assert!(
            texel
                .iter()
                .zip(want)
                .all(|(&got, want)| got.abs_diff(want) <= 1),
            "texel ({x}, {y}) is {texel:?}, not {want:?}"
        );
    }
    f.assert_no_validation_errors();
}

#[test]
fn linear_blits_of_integer_texels_go_to_software() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let integer = |width, height, bh| ImageInfo {
        format: PixelFormat::A8B8G8R8Uint,
        ..rt_info(width, height, bh, gob)
    };
    let (src, dst) = (integer(64, 64, 3), integer(32, 32, 2));
    clear_rt(&mut f, &src, [0.0; 4], Rect::ALL);
    let (mut tx, cache, gmmu) = f.tx();
    let side = blit_side(GPU_VA, src, [0, 0, 64, 64]);
    let target = blit_side(GPU_VA + (64 << 10), dst, [0, 0, 32, 32]);
    assert!(
        !cache
            .blit(&mut tx, gmmu, &side, &target, vk::Filter::LINEAR)
            .unwrap(),
        "integer texels have no linear filter"
    );
    assert!(
        cache
            .blit(&mut tx, gmmu, &side, &target, vk::Filter::NEAREST)
            .unwrap(),
        "but blit nearest"
    );
    f.assert_no_validation_errors();
}

#[test]
fn blits_the_host_cannot_do_as_they_are_go_to_software() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let src = rt_info(64, 64, 3, gob);
    clear_rt(&mut f, &src, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let integer = ImageInfo {
        format: PixelFormat::A8B8G8R8Uint,
        ..rt_info(32, 32, 2, gob)
    };
    let (mut tx, cache, gmmu) = f.tx();
    let side = blit_side(GPU_VA, src, [0, 0, 64, 64]);
    assert!(
        !cache
            .blit(
                &mut tx,
                gmmu,
                &side,
                &blit_side(GPU_VA + (64 << 10), integer, [0, 0, 32, 32]),
                vk::Filter::NEAREST,
            )
            .unwrap(),
        "a float format into an integer one"
    );
    assert!(
        !cache
            .blit(
                &mut tx,
                gmmu,
                &side,
                &blit_side(GPU_VA, src, [0, 0, 32, 32]),
                vk::Filter::NEAREST,
            )
            .unwrap(),
        "one image on both sides"
    );
    assert!(
        !cache
            .blit(
                &mut tx,
                gmmu,
                &side,
                &blit_side(GPU_VA + (128 << 10), src, [0, 0, 65, 64]),
                vk::Filter::NEAREST,
            )
            .unwrap(),
        "a rectangle leaving its image"
    );
    f.assert_no_validation_errors();
}

#[test]
fn blits_from_guest_data_stay_in_software() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let src = rt_info(64, 64, 3, gob);
    let dst = rt_info(32, 32, 2, gob);
    let (mut tx, cache, gmmu) = f.tx();
    cache.find(&mut tx, gmmu, GPU_VA, &src).unwrap().unwrap();
    let before = f.cache.traffic();
    let (mut tx, cache, gmmu) = f.tx();
    let blitted = cache
        .blit(
            &mut tx,
            gmmu,
            &blit_side(GPU_VA, src, [0, 0, 64, 64]),
            &blit_side(GPU_VA + (64 << 10), dst, [0, 0, 32, 32]),
            vk::Filter::NEAREST,
        )
        .unwrap();
    assert!(!blitted, "an uploaded source no GPU work wrote");
    assert_eq!(f.cache.traffic(), before, "no upload for a software blit");
    f.assert_no_validation_errors();
}

/// Texel `(2x + 1, 2y + 1)` of a 64-texel-wide RGBA8 level for each texel of the 32x32 level a
/// nearest 2:1 `vkCmdBlitImage` makes of it.
fn nearest_halving(level: &[u8]) -> Vec<u8> {
    (0..32 * 32)
        .flat_map(|i| {
            let (x, y) = (i % 32, i / 32);
            let at = ((2 * y + 1) * 64 + 2 * x + 1) * 4;
            level[at..at + 4].to_vec()
        })
        .collect()
}

#[test]
fn guest_data_blits_into_a_cached_image_on_the_gpu() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let s = surface(PixelFormat::A8B8G8R8Unorm, 64, 64, 1, 1, 3, gob);
    let data = level_data(&s, 7);
    write_surface(&f.memory, &s, &data);
    let src = rt_info(64, 64, 3, gob);
    let dst = rt_info(32, 32, 2, gob);
    let dst_va = GPU_VA + (64 << 10);
    let target = clear_rt_at(&mut f, dst_va, &dst, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let (mut tx, cache, gmmu) = f.tx();
    let blitted = cache
        .blit(
            &mut tx,
            gmmu,
            &blit_side(GPU_VA, src, [0, 0, 64, 64]),
            &blit_side(dst_va, dst, [0, 0, 32, 32]),
            vk::Filter::NEAREST,
        )
        .unwrap();
    assert!(
        blitted,
        "the GPU reads the target, so the source uploads once"
    );
    assert_eq!(f.read_image(target.id)[0], nearest_halving(&data[0]));
    f.assert_no_validation_errors();
}

#[test]
fn mip_chain_levels_blit_into_each_other_on_the_gpu() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let s = surface(PixelFormat::A8B8G8R8Unorm, 64, 64, 2, 1, 3, gob);
    let data = level_data(&s, 9);
    write_surface(&f.memory, &s, &data);
    let (chain, view) = f.texture(&tic(0x08, GPU_VA, 64, 64, 1, 2, 3));
    assert_eq!(view.levels, 2);
    let level_va = GPU_VA + s.level_offset(1);
    let (mut tx, cache, gmmu) = f.tx();
    let blitted = cache
        .blit(
            &mut tx,
            gmmu,
            &blit_side(GPU_VA, rt_info(64, 64, 3, gob), [0, 0, 64, 64]),
            &blit_side(level_va, rt_info(32, 32, 2, gob), [0, 0, 32, 32]),
            vk::Filter::NEAREST,
        )
        .unwrap();
    assert!(blitted, "level 0 into level 1 of one image");
    let read = f.read_image(chain);
    assert_eq!(read[0], data[0]);
    assert_eq!(read[1], nearest_halving(&data[0]));
    f.assert_no_validation_errors();
}

#[test]
fn nearest_blits_gather_the_texels_the_software_blit_samples_on_the_gpu() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = |format: PixelFormat, width: u32, height: u32| {
        let s = surface(format, width, height, 1, 1, 2, gob);
        ImageInfo {
            format,
            layer_stride: s.layer_stride(),
            ..rt_info(width, height, 2, gob)
        }
    };
    const UNIT: i64 = 1 << 32;
    let quarter = NearestSampling {
        x0: 0,
        y0: -0x0144_0000,
        du: 4 * UNIT,
        dv: 0x4_0288_0000,
        width: 64,
        height: 64,
    };
    let narrower = NearestSampling {
        width: 50,
        ..quarter
    };
    let clamped = NearestSampling {
        x0: -3 * UNIT / 2,
        y0: UNIT / 4,
        du: 3 * UNIT,
        dv: UNIT / 4,
        width: 64,
        height: 64,
    };
    for (format, sampling) in [
        (PixelFormat::A8B8G8R8Unorm, quarter),
        (PixelFormat::R32Float, narrower),
        (PixelFormat::R16G16B16A16Float, clamped),
        (PixelFormat::R32G32B32A32Float, quarter),
    ] {
        let (sw, sh, dw, dh) = (64u32, 64u32, 16u32, 16u32);
        let src = info(format, sw, sh);
        let dst = info(format, 24, 20);
        let dst_va = GPU_VA + (128 << 10);
        let bytes = format.bytes_per_block() as usize;
        let s = surface(format, sw, sh, 1, 1, 2, gob);
        let data = level_data(&s, format as u32);
        write_surface(&f.memory, &s, &data);
        let (mut tx, cache, gmmu) = f.tx();
        let source = cache
            .render_target(&mut tx, gmmu, GPU_VA, &src, false)
            .unwrap()
            .unwrap();
        cache.mark_modified(source.id);
        let rect = [3, 2, 3 + dw as i32, 2 + dh as i32];
        let before = f.cache.traffic();
        let (mut tx, cache, gmmu) = f.tx();
        let blitted = cache
            .gather(
                &mut tx,
                gmmu,
                GPU_VA,
                &src,
                &blit_side(dst_va, dst, rect),
                &sampling,
            )
            .unwrap();
        assert!(blitted, "{format:?}");
        assert_eq!(
            f.cache.traffic().downloads,
            before.downloads,
            "{format:?}: no write-back"
        );
        let (mut tx, cache, gmmu) = f.tx();
        let target = cache.find(&mut tx, gmmu, dst_va, &dst).unwrap().unwrap();
        let texels = f.read_image(target.id).remove(0);
        let index = |start: i64, step: i64, i: usize, count: u32, extent: u32| {
            let first = (start >> 32).max(0);
            let end = ((start + step * i64::from(count)) >> 32).min(i64::from(extent));
            ((start + step * i as i64) >> 32).clamp(first, end - 1) as usize
        };
        for y in 0..dh as usize {
            let row = index(sampling.y0, sampling.dv, y, dh, sampling.height);
            for x in 0..dw as usize {
                let column = index(sampling.x0, sampling.du, x, dw, sampling.width);
                let from = (row * sw as usize + column) * bytes;
                let to = ((y + 2) * 24 + x + 3) * bytes;
                assert_eq!(
                    texels[to..to + bytes],
                    data[0][from..from + bytes],
                    "{format:?}: texel ({x}, {y}) should be source ({column}, {row})"
                );
            }
        }
        f.cache.unmap(&mut f.scheduler, DADDR, 1 << 20);
    }
    f.assert_no_validation_errors();
}

#[test]
fn nearest_blits_the_gather_cannot_do_go_to_software() {
    let gob = GobLayout::Tegra;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let src = rt_info(64, 64, 2, gob);
    clear_rt(&mut f, &src, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let sampling = NearestSampling {
        x0: 0,
        y0: 0,
        du: 4 << 32,
        dv: 4 << 32,
        width: 64,
        height: 64,
    };
    let integer = ImageInfo {
        format: PixelFormat::A8B8G8R8Uint,
        ..rt_info(16, 16, 2, gob)
    };
    let reversed = NearestSampling {
        x0: 63 << 32,
        du: -(4 << 32),
        ..sampling
    };
    let wrapping = NearestSampling {
        du: 1 << 62,
        ..sampling
    };
    let small = rt_info(16, 16, 2, gob);
    let (mut tx, cache, gmmu) = f.tx();
    for (dst, rect, sampling, why) in [
        (integer, [0, 0, 16, 16], sampling, "another format"),
        (src, [0, 0, 16, 16], sampling, "one image on both sides"),
        (
            small,
            [0, 0, 17, 16],
            sampling,
            "a rectangle leaving its image",
        ),
        (small, [0, 0, 16, 16], reversed, "a reversed step"),
        (small, [0, 0, 16, 16], wrapping, "a step whose end wraps"),
    ] {
        let va = if why == "one image on both sides" {
            GPU_VA
        } else {
            GPU_VA + (64 << 10)
        };
        assert!(
            !cache
                .gather(
                    &mut tx,
                    gmmu,
                    GPU_VA,
                    &src,
                    &blit_side(va, dst, rect),
                    &sampling
                )
                .unwrap(),
            "{why}"
        );
    }
    f.assert_no_validation_errors();
}

#[test]
fn narrower_block_linear_textures_alias_their_render_target_without_a_flush() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(32, 24, 1, GobLayout::Tegra);
    let target = clear_rt(&mut f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let header = tic(0x08, GPU_VA, 24, 24, 1, 1, 1);
    let (texture, _) = f.texture(&header);
    assert_ne!(texture, target.id);
    assert!(
        f.cache.state(target.id).is_some(),
        "the render target stays"
    );
    assert!(all_texels(&f.read_image(texture)[0], [255, 0, 0, 255]));
    clear_rt(&mut f, &info, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    assert_eq!(f.texture(&header).0, texture);
    assert!(
        all_texels(&f.read_image(texture)[0], [0, 255, 0, 255]),
        "the texture takes the render target's newer texels"
    );
    assert_eq!(
        f.cache.traffic().downloads,
        0,
        "no write-back to guest memory"
    );
    f.assert_no_validation_errors();
}

#[test]
fn overlaps_of_another_layout_drawn_into_in_turn_keep_their_own_texels() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let wide = rt_info(256, 64, 2, GobLayout::Tegra);
    let narrow = rt_info(64, 64, 2, GobLayout::Tegra);
    let corner = Rect {
        x: 0,
        y: 0,
        width: 8,
        height: 8,
    };
    let first = clear_rt(&mut f, &wide, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let second = clear_rt(&mut f, &narrow, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    assert_ne!(first.id, second.id);
    let before = f.cache.traffic();
    for _ in 0..3 {
        assert_eq!(
            clear_rt(&mut f, &wide, [1.0, 0.0, 0.0, 1.0], corner).id,
            first.id
        );
        assert_eq!(
            clear_rt(&mut f, &narrow, [0.0, 1.0, 0.0, 1.0], corner).id,
            second.id
        );
    }
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, before.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(
        after.uploads, before.uploads,
        "no re-upload from guest memory"
    );
    assert_eq!(after.gpu_syncs, before.gpu_syncs, "no switch synced");
    assert!(all_texels(&f.read_image(first.id)[0], RED));
    assert!(all_texels(&f.read_image(second.id)[0], GREEN));
    f.assert_no_validation_errors();
}

#[test]
fn gpu_overlap_syncs_move_texels_through_the_guest_layout() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let wide = rt_info(256, 64, 2, GobLayout::Tegra);
    let narrow = rt_info(64, 64, 2, GobLayout::Tegra);
    let pattern: Vec<u8> = (0..wide.size_bytes())
        .map(|i| (i * 7 + i / 251) as u8)
        .collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let wide_rect = Rect {
        x: 4,
        y: 3,
        width: 20,
        height: 9,
    };
    let target = clear_rt(&mut f, &wide, [0.0, 0.0, 1.0, 1.0], wide_rect);
    let mut expected = f.read_image(target.id).remove(0);
    clear_rt(
        &mut f,
        &narrow,
        [0.0, 1.0, 0.0, 1.0],
        Rect {
            x: 8,
            y: 40,
            width: 16,
            height: 8,
        },
    );
    for y in 8..16 {
        for x in 72..88 {
            let at = (y * 256 + x) * 4;
            expected[at..at + 4].copy_from_slice(&[0, 255, 0, 255]);
        }
    }
    let syncs = f.cache.traffic().gpu_syncs;
    let downloads = f.cache.traffic().downloads;
    let (mut tx, cache, gmmu) = f.tx();
    let synced = cache.find(&mut tx, gmmu, GPU_VA, &wide).unwrap().unwrap();
    assert_eq!(synced.id, target.id);
    let texels = f.read_image(target.id).remove(0);
    assert!(f.cache.traffic().gpu_syncs > syncs);
    assert_eq!(f.cache.traffic().downloads, downloads);
    let wrong: Vec<usize> = (0..texels.len() / 4)
        .filter(|i| texels[i * 4..i * 4 + 4] != expected[i * 4..i * 4 + 4])
        .take(8)
        .collect();
    assert!(wrong.is_empty(), "texels differ at {wrong:?}");
    f.assert_no_validation_errors();
}

#[test]
fn gpu_overlap_syncs_skip_overlaps_of_block_padding_alone() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let padded = rt_info(16, 16, 4, GobLayout::Tegra);
    let small = rt_info(16, 8, 0, GobLayout::Tegra);
    assert!(padded.size_bytes() >= 5 << 10);
    let target = clear_rt(&mut f, &padded, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    let expected = f.read_image(target.id).remove(0);
    clear_rt_at(
        &mut f,
        GPU_VA + (4 << 10),
        &small,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    );
    let syncs = f.cache.traffic().gpu_syncs;
    let (mut tx, cache, gmmu) = f.tx();
    let synced = cache.find(&mut tx, gmmu, GPU_VA, &padded).unwrap().unwrap();
    assert_eq!(synced.id, target.id);
    assert!(f.cache.traffic().gpu_syncs > syncs);
    assert_eq!(f.read_image(target.id).remove(0), expected);
    f.assert_no_validation_errors();
}

#[test]
fn gpu_overlap_syncs_give_aliased_bytes_to_the_last_source_page() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(64, 64, 0, GobLayout::Tegra);
    assert_eq!(info.size_bytes(), 16 << 10);
    let alias = GPU_VA + DRAM;
    f.gmmu.map(alias, DADDR + 0x1000, 0x1000, 0, false).unwrap();
    f.gmmu
        .map(alias + 0x1000, DADDR + 0x1000, 0x3000, 0, false)
        .unwrap();
    let (red, green, blue) = (
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    );
    let target = clear_rt(&mut f, &info, red, Rect::ALL);
    clear_rt_at(&mut f, alias, &info, green, Rect::ALL);
    clear_rt_at(
        &mut f,
        alias,
        &info,
        blue,
        Rect {
            x: 0,
            y: 16,
            width: 64,
            height: 16,
        },
    );
    let syncs = f.cache.traffic().gpu_syncs;
    let (mut tx, cache, gmmu) = f.tx();
    let synced = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
    assert_eq!(synced.id, target.id);
    assert!(f.cache.traffic().gpu_syncs > syncs);
    let texels = f.read_image(target.id).remove(0);
    let wrong: Vec<(usize, [u8; 4])> = (0..64 * 64)
        .filter_map(|i| {
            let expected = match i / 64 {
                0..16 => [255, 0, 0, 255],
                16..32 => [0, 0, 255, 255],
                _ => [0, 255, 0, 255],
            };
            let got: [u8; 4] = texels[i * 4..i * 4 + 4].try_into().unwrap();
            (got != expected).then_some((i, got))
        })
        .take(8)
        .collect();
    assert!(wrong.is_empty(), "texels differ: {wrong:?}");
    f.assert_no_validation_errors();
}

#[test]
fn aliased_sources_whose_rows_end_inside_a_word_write_back_through_guest_memory() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(64, 64, 0, GobLayout::Tegra);
    let mut bytes = rt_info(125, 64, 0, GobLayout::Tegra);
    bytes.format = PixelFormat::R8Unorm;
    assert_eq!(bytes.size_bytes(), 8 << 10);
    let alias = GPU_VA + DRAM;
    f.gmmu.map(alias, DADDR + 0x1000, 0x1000, 0, false).unwrap();
    f.gmmu
        .map(alias + 0x1000, DADDR + 0x1000, 0x1000, 0, false)
        .unwrap();
    let target = clear_rt(&mut f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    clear_rt_at(&mut f, alias, &bytes, [1.0, 0.0, 0.0, 0.0], Rect::ALL);
    let before = f.cache.traffic();
    let (mut tx, cache, gmmu) = f.tx();
    let synced = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
    assert_eq!(synced.id, target.id);
    let after = f.cache.traffic();
    assert_eq!(after.gpu_syncs, before.gpu_syncs, "no GPU sync");
    assert!(after.downloads > before.downloads, "written back instead");
    f.assert_no_validation_errors();
}

#[test]
fn gpu_overlap_syncs_keep_texels_whose_later_alias_is_padding() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let info = rt_info(16, 16, 4, GobLayout::Tegra);
    assert_eq!(info.size_bytes(), 8 << 10);
    let alias = GPU_VA + DRAM;
    f.gmmu.map(alias, DADDR, 0x1000, 0, false).unwrap();
    f.gmmu.map(alias + 0x1000, DADDR, 0x1000, 0, false).unwrap();
    let target = clear_rt(&mut f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    clear_rt_at(&mut f, alias, &info, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
    let syncs = f.cache.traffic().gpu_syncs;
    let (mut tx, cache, gmmu) = f.tx();
    let synced = cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().unwrap();
    assert_eq!(synced.id, target.id);
    assert!(f.cache.traffic().gpu_syncs > syncs);
    let texels = f.read_image(target.id).remove(0);
    assert!(
        texels.chunks(4).all(|t| t == [0, 255, 0, 255]),
        "the source's texels reach the target: {:?}",
        &texels[..16]
    );
    f.assert_no_validation_errors();
}

#[test]
fn depth_stencil_overlaps_sync_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let clear = |f: &mut Fixture, info: &ImageInfo, depth: f32, stencil: u8, rect: Rect| {
        let (mut tx, cache, gmmu) = f.tx();
        let target = cache
            .render_target(&mut tx, gmmu, GPU_VA, info, rect == Rect::ALL)
            .unwrap()
            .unwrap();
        cache
            .clear(
                &mut tx,
                target,
                ClearValue::DepthStencil {
                    depth: Some(depth),
                    stencil: Some(stencil),
                    stencil_mask: 0xff,
                },
                rect,
                0,
                1,
            )
            .unwrap();
        target
    };
    for format in [PixelFormat::S8UintD24Unorm, PixelFormat::D24UnormS8Uint] {
        let mut wide = rt_info(256, 64, 2, GobLayout::Tegra);
        let mut narrow = rt_info(64, 64, 2, GobLayout::Tegra);
        wide.format = format;
        narrow.format = format;
        let target = clear(&mut f, &wide, 0.5, 0x5a, Rect::ALL);
        let float_depth = match f.cache.get(target.id).unwrap().host.format {
            vk::Format::D24_UNORM_S8_UINT => false,
            vk::Format::D32_SFLOAT_S8_UINT => true,
            host => panic!("{format:?} held as {host:?}"),
        };
        clear(
            &mut f,
            &narrow,
            0.25,
            0x33,
            Rect {
                x: 8,
                y: 40,
                width: 16,
                height: 8,
            },
        );
        let syncs = f.cache.traffic().gpu_syncs;
        let downloads = f.cache.traffic().downloads;
        let (mut tx, cache, gmmu) = f.tx();
        let synced = cache.find(&mut tx, gmmu, GPU_VA, &wide).unwrap().unwrap();
        assert_eq!(synced.id, target.id);
        assert!(f.cache.traffic().gpu_syncs > syncs, "{format:?}");
        assert_eq!(f.cache.traffic().downloads, downloads, "{format:?}");
        let depth = f.read_plane(target.id, vk::ImageAspectFlags::DEPTH, 4);
        let stencil = f.read_plane(target.id, vk::ImageAspectFlags::STENCIL, 1);
        for (i, word) in depth.as_chunks::<4>().0.iter().enumerate() {
            let (x, y) = (i % 256, i / 256);
            let inside = (72..88).contains(&x) && (8..16).contains(&y);
            let (want_depth, want_stencil) = if inside {
                (0x40_0000, 0x33)
            } else {
                (0x80_0000, 0x5a)
            };
            let word = u32::from_le_bytes(*word);
            let d = if float_depth {
                (f64::from(f32::from_bits(word)) * 16_777_215.0).round() as u32
            } else {
                word & 0xff_ffff
            };
            assert!(
                d.abs_diff(want_depth) <= 1,
                "{format:?}: depth {d:#x} at ({x}, {y})"
            );
            assert_eq!(
                stencil[i], want_stencil,
                "{format:?}: stencil at ({x}, {y})"
            );
        }
        f.cache.unmap(&mut f.scheduler, DADDR, 1 << 20);
    }
    f.assert_no_validation_errors();
}

#[test]
fn x8d24_overlaps_sync_on_the_gpu_converting_as_the_cpu_does() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let (w, h) = (256, 64);
    let mut depth = rt_info(w, h, 2, GobLayout::Tegra);
    depth.format = PixelFormat::X8D24Unorm;
    let colour = rt_info(64, 64, 2, GobLayout::Tegra);
    let inside = |k: usize| (8..24).contains(&(k % 64)) && (40..48).contains(&(k / 64));
    let depth_tiling = depth.surface().level_tiling(0);
    let colour_tiling = colour.surface().level_tiling(0);
    let mut tiled = vec![0; depth_tiling.size_bytes()];
    let numbered: Vec<u8> = (1..=64 * 64u32).flat_map(u32::to_le_bytes).collect();
    swizzle(
        &colour_tiling,
        &numbered,
        &mut tiled[..colour_tiling.size_bytes()],
    );
    let mut numbered_in_depth = vec![0; depth_tiling.linear_size_bytes()];
    unswizzle(&depth_tiling, &tiled, &mut numbered_in_depth);
    let colour_texel: Vec<Option<usize>> = numbered_in_depth
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&n| (u32::from_le_bytes(n) as usize).checked_sub(1))
        .collect();
    let d24 = |i: u32| match i % 4 {
        0 => i / 4,
        1 => 0xff_ffff - i / 4,
        _ => i.wrapping_mul(0x9e37_79b9) >> 8,
    };
    let words: Vec<u32> = (0..w * h)
        .map(|i| d24(i) | (i.wrapping_mul(37) & 0xff) << 24)
        .collect();
    let float_host = HostFormat {
        format: vk::Format::D32_SFLOAT,
        conversion: Conversion::Convert,
        swizzle: vk::ComponentMapping::default(),
        attachable: true,
        storage: false,
    };
    let native = host_format(&f.device, depth.format, AstcRecompression::Uncompressed);
    for host in [float_host, native] {
        let float = match host.format {
            vk::Format::D32_SFLOAT => true,
            vk::Format::X8_D24_UNORM_PACK32 => false,
            other => {
                eprintln!("skipping X8D24 held as {other:?}");
                continue;
            }
        };
        f.cache.host_formats[depth.format as usize] = Some(host);
        let to_host = |word: u32| {
            let d = word & 0xff_ffff;
            if float {
                ((f64::from(d) / 16_777_215.0) as f32).to_bits()
            } else {
                d
            }
        };
        let host_word = |word: [u8; 4]| {
            let word = u32::from_le_bytes(word);
            if float { word } else { word & 0xff_ffff }
        };
        let to_guest = |host: u32| {
            if float {
                (f64::from(f32::from_bits(host)).clamp(0.0, 1.0) * 16_777_215.0).round() as u32
            } else {
                host & 0xff_ffff
            }
        };
        let linear: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let mut guest = vec![0; depth_tiling.size_bytes()];
        swizzle(&depth_tiling, &linear, &mut guest);
        f.memory.write_bytes(DADDR, &guest).unwrap();
        let (mut tx, cache, gmmu) = f.tx();
        let target = cache
            .render_target(&mut tx, gmmu, GPU_VA, &depth, false)
            .unwrap()
            .unwrap();
        assert_eq!(f.cache.get(target.id).unwrap().host.format, host.format);
        let uploaded = f.read_plane(target.id, vk::ImageAspectFlags::DEPTH, 4);
        for (i, &texel) in uploaded.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(host_word(texel), to_host(words[i]), "uploaded texel {i}");
        }
        let blue = 0x4433_2211;
        clear_rt(
            &mut f,
            &colour,
            [0x11u8, 0x22, 0x33, 0x44].map(|c| f32::from(c) / 255.0),
            Rect {
                x: 8,
                y: 40,
                width: 16,
                height: 8,
            },
        );
        let before = f.cache.traffic();
        let (mut tx, cache, gmmu) = f.tx();
        let synced = cache.find(&mut tx, gmmu, GPU_VA, &depth).unwrap().unwrap();
        assert_eq!(synced.id, target.id);
        assert!(f.cache.traffic().gpu_syncs > before.gpu_syncs, "{host:?}");
        assert_eq!(f.cache.traffic().downloads, before.downloads, "{host:?}");
        let synced = f.read_plane(target.id, vk::ImageAspectFlags::DEPTH, 4);
        for (i, &texel) in synced.as_chunks::<4>().0.iter().enumerate() {
            let from = match colour_texel[i] {
                Some(k) if inside(k) => blue,
                _ => words[i],
            };
            assert_eq!(host_word(texel), to_host(from), "{host:?}: texel {i}");
        }

        let (mut tx, cache, gmmu) = f.tx();
        let target = cache
            .render_target(&mut tx, gmmu, GPU_VA, &depth, false)
            .unwrap()
            .unwrap();
        cache
            .clear(
                &mut tx,
                target,
                ClearValue::DepthStencil {
                    depth: Some(0.3),
                    stencil: None,
                    stencil_mask: 0,
                },
                Rect {
                    x: 0,
                    y: 0,
                    width: 32,
                    height: 4,
                },
                0,
                1,
            )
            .unwrap();
        let newest: Vec<u32> = f
            .read_plane(target.id, vk::ImageAspectFlags::DEPTH, 4)
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&t| host_word(t))
            .collect();
        let before = f.cache.traffic();
        let (mut tx, cache, gmmu) = f.tx();
        let taken = cache.find(&mut tx, gmmu, GPU_VA, &colour).unwrap().unwrap();
        assert!(f.cache.traffic().gpu_syncs > before.gpu_syncs, "{host:?}");
        assert_eq!(f.cache.traffic().downloads, before.downloads, "{host:?}");
        let texels = f.read_image(taken.id).remove(0);
        let mut expected = vec![0; 64 * 64];
        for (i, k) in colour_texel.iter().enumerate() {
            if let Some(k) = *k {
                expected[k] = to_guest(newest[i]);
            }
        }
        for (k, &texel) in texels.as_chunks::<4>().0.iter().enumerate() {
            assert_eq!(
                u32::from_le_bytes(texel),
                expected[k],
                "{host:?}: guest word of colour texel {k}"
            );
        }
        f.cache.unmap(&mut f.scheduler, DADDR, 1 << 20);
    }
    f.assert_no_validation_errors();
}

#[test]
fn block_linear_aliases_need_the_same_depth_stencil_encoding() {
    let info = |format, width| {
        let mut info = rt_info(width, 64, 2, GobLayout::Tegra);
        info.format = format;
        info
    };
    let d24s8 = info(PixelFormat::D24UnormS8Uint, 256);
    let s8d24 = info(PixelFormat::S8UintD24Unorm, 252);
    assert!(block_linear_overlap(&d24s8, DADDR, &s8d24, DADDR).is_none());
    let narrower = info(PixelFormat::D24UnormS8Uint, 252);
    assert_eq!(
        block_linear_overlap(&d24s8, DADDR, &narrower, DADDR),
        Some(((0, 0), (0, 0), (252, 64)))
    );
}

fn copy_engine_setup(f: &mut Fixture) -> (ImageCopy, u64) {
    let info = rt_info(64, 8, 0, GobLayout::Tegra);
    clear_rt(f, &info, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
    clear_rt(
        f,
        &info,
        [0.0, 1.0, 0.0, 1.0],
        Rect {
            x: 10,
            y: 3,
            width: 4,
            height: 2,
        },
    );
    let copy = ImageCopy {
        src: copy_surface(GPU_VA, 0, Some((0, 0, 0))),
        dst: copy_surface(GPU_VA + (1 << 20), 128, None),
        src_origin: (8, 2),
        dst_origin: (0, 0),
        width: 16,
        height: 4,
    };
    (copy, 3 * 128 + 16 * 4)
}

fn copy_surface(address: GpuVAddr, pitch: u32, block: Option<(u32, u32, u32)>) -> CopySurface {
    CopySurface {
        address,
        width: 64,
        height: 8,
        depth: 1,
        layer: 0,
        pitch,
        block,
        bytes_per_pixel: 4,
    }
}

fn copied_texel(col: usize, row: usize) -> [u8; 4] {
    if (2..6).contains(&col) && (1..3).contains(&row) {
        [0, 255, 0, 255]
    } else {
        [255, 0, 0, 255]
    }
}

#[test]
fn copy_engine_reads_of_gpu_modified_images_stay_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let (copy, span) = copy_engine_setup(&mut f);
    let downloads = f.cache.traffic().downloads;
    let (mut tx, cache, gmmu) = f.tx();
    assert!(cache.copy_to_buffer(&mut tx, gmmu, &copy).unwrap());
    f.scheduler.finish().unwrap();
    assert_eq!(
        f.cache.traffic().downloads,
        downloads,
        "no write-back to guest memory"
    );
    assert!(
        guest_texels(&f, 1 << 20, span as usize)
            .iter()
            .all(|&b| b == 0),
        "guest memory waits for a read"
    );
    f.buffers
        .flush(&mut f.scheduler, DADDR + (1 << 20), span)
        .unwrap();
    let guest = guest_texels(&f, 1 << 20, span as usize);
    for row in 0..4 {
        for col in 0..16 {
            let at = row * 128 + col * 4;
            assert_eq!(
                guest[at..at + 4],
                copied_texel(col, row),
                "texel ({col}, {row})"
            );
        }
    }
    let fresh = ImageCopy {
        src: copy_surface(GPU_VA + (2 << 20), 0, Some((0, 0, 0))),
        ..copy
    };
    let (mut tx, cache, gmmu) = f.tx();
    assert!(!cache.copy_to_buffer(&mut tx, gmmu, &fresh).unwrap());
    f.assert_no_validation_errors();
}

#[test]
fn textures_over_copy_engine_destinations_sample_the_copied_texels() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let (copy, _) = copy_engine_setup(&mut f);
    let (mut tx, cache, gmmu) = f.tx();
    assert!(cache.copy_to_buffer(&mut tx, gmmu, &copy).unwrap());
    let (id, _) = f.texture(&pitch_tic(GPU_VA + (1 << 20), 16, 4, 128));
    let texels = f.read_image(id).remove(0);
    for row in 0..4 {
        for col in 0..16 {
            let at = (row * 16 + col) * 4;
            assert_eq!(
                texels[at..at + 4],
                copied_texel(col, row),
                "texel ({col}, {row})"
            );
        }
    }
    f.assert_no_validation_errors();
}

#[test]
fn copy_engine_reads_addressed_with_another_block_height_copy_in_software() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let (copy, _) = copy_engine_setup(&mut f);
    let taller = ImageCopy {
        src: copy_surface(GPU_VA, 0, Some((0, 1, 0))),
        ..copy
    };
    let (mut tx, cache, gmmu) = f.tx();
    assert!(!cache.copy_to_buffer(&mut tx, gmmu, &taller).unwrap());
}

#[test]
fn full_clears_over_copy_engine_destinations_stay_newer_than_the_copy() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let (copy, _) = copy_engine_setup(&mut f);
    let (mut tx, cache, gmmu) = f.tx();
    assert!(cache.copy_to_buffer(&mut tx, gmmu, &copy).unwrap());
    let info = pitch_target(128, 4);
    clear_rt_at(
        &mut f,
        GPU_VA + (1 << 20),
        &info,
        [0.0, 0.0, 1.0, 1.0],
        Rect::ALL,
    );
    let (mut tx, cache, _) = f.tx();
    cache.flush(&mut tx, DADDR + (1 << 20), 512).unwrap();
    f.buffers
        .flush(&mut f.scheduler, DADDR + (1 << 20), 512)
        .unwrap();
    f.scheduler.finish().unwrap();
    assert!(all_texels(
        &guest_texels(&f, 1 << 20, 512),
        [0, 0, 255, 255]
    ));
    f.assert_no_validation_errors();
}

#[test]
fn cpu_writes_upload_only_the_layers_they_touch() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let header = tic(0x08, GPU_VA, 64, 64, 4, 1, 3);
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    let pattern: Vec<u8> = (0..info.size_bytes())
        .map(|i| (i * 13 + i / 97) as u8)
        .collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    let before = f.read_image(texture);
    let uploaded = f.cache.traffic();
    let layer = info.layer_stride;
    f.memory
        .write_bytes(DADDR + 2 * layer + 256, &[0xab; 64])
        .unwrap();
    {
        let (mut tx, cache, _) = f.tx();
        cache
            .invalidate(&mut tx, DADDR + 2 * layer + 256, 64)
            .unwrap();
    }
    assert_eq!(f.texture(&header).0, texture);
    let traffic = f.cache.traffic();
    assert_eq!(traffic.uploads, uploaded.uploads + 1);
    assert_eq!(
        traffic.upload_bytes - uploaded.upload_bytes,
        64 * 64 * 4,
        "only layer 2 uploads again"
    );
    let after = f.read_image(texture);
    for l in [0, 1, 3] {
        assert_eq!(after[l], before[l], "layer {l} keeps its texels");
    }
    assert_ne!(after[2], before[2], "layer 2 takes the CPU's bytes");
    let changed = after[2]
        .iter()
        .zip(&before[2])
        .filter(|(a, b)| a != b)
        .count();
    assert!(changed > 0 && changed <= 64, "{changed} bytes changed");
    f.assert_no_validation_errors();
}

#[test]
fn overlaps_written_back_upload_only_the_layers_they_cover() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let header = tic(0x08, GPU_VA, 64, 64, 4, 1, 3);
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    let pattern: Vec<u8> = (0..info.size_bytes())
        .map(|i| (i * 13 + i / 97) as u8)
        .collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    let before = f.read_image(texture);
    let bytes = ImageInfo {
        format: PixelFormat::R8Unorm,
        layout: GuestLayout::Pitch { pitch: 128 },
        ..rt_info(126, 32, 2, GobLayout::Tegra)
    };
    assert_eq!(bytes.size_bytes(), 4096);
    let layer_va = GPU_VA + 2 * info.layer_stride;
    let target = clear_rt_at(&mut f, layer_va, &bytes, [1.0, 0.0, 0.0, 0.0], Rect::ALL);
    assert_ne!(target.id, texture);
    let uploaded = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let traffic = f.cache.traffic();
    assert_eq!(
        traffic.upload_bytes - uploaded.upload_bytes,
        64 * 64 * 4,
        "only layer 2 uploads again"
    );
    let after = f.read_image(texture);
    for l in [0, 1, 3] {
        assert_eq!(after[l], before[l], "layer {l} keeps its texels");
    }
    let tiling = info.surface().level_tiling(0);
    let layer = (2 * info.layer_stride) as usize;
    let mut tiled = pattern[layer..layer + tiling.size_bytes()].to_vec();
    for y in 0..32 {
        for x in 0..126 {
            tiled[y * 128 + x] = 255;
        }
    }
    let mut want = vec![0; 64 * 64 * 4];
    lichii_texture::block_linear::unswizzle(&tiling, &tiled, &mut want);
    assert_eq!(after[2], want, "layer 2 takes the render target's bytes");
    f.assert_no_validation_errors();
}

#[test]
fn cpu_writes_between_layers_upload_nothing() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let header = tic(0x08, GPU_VA, 64, 64, 4, 3, 3);
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    let used = info.subresource_offset(2, 0) + info.level_size(2);
    assert!(
        used < info.layer_stride,
        "{used} of {} bytes",
        info.layer_stride
    );
    let (texture, _) = f.texture(&header);
    let uploaded = f.cache.traffic();
    {
        let (mut tx, cache, _) = f.tx();
        cache.invalidate(&mut tx, DADDR + used, 64).unwrap();
    }
    assert_eq!(f.texture(&header).0, texture);
    assert_eq!(f.cache.traffic().upload_bytes, uploaded.upload_bytes);
    f.scheduler.finish().unwrap();
    f.assert_no_validation_errors();
}

#[test]
fn narrow_formats_sync_into_the_layers_they_share_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let header = tic(0x08, GPU_VA, 64, 64, 4, 1, 3);
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    let pattern: Vec<u8> = (0..info.size_bytes())
        .map(|i| (i * 13 + i / 97) as u8)
        .collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    let before = f.read_image(texture);
    let mut halves = rt_info(64, 32, 2, GobLayout::Tegra);
    halves.format = PixelFormat::R16Unorm;
    let mut bytes = rt_info(128, 32, 2, GobLayout::Tegra);
    bytes.format = PixelFormat::R8Unorm;
    let layer = |l: u64| GPU_VA + l * info.layer_stride;
    clear_rt_at(&mut f, layer(1), &halves, [1.0, 0.0, 0.0, 0.0], Rect::ALL);
    clear_rt_at(&mut f, layer(2), &bytes, [1.0, 0.0, 0.0, 0.0], Rect::ALL);
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(
        after.uploads, traffic.uploads,
        "no re-upload from guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    let texels = f.read_image(texture);
    for l in [0, 3] {
        assert_eq!(texels[l], before[l], "layer {l} keeps its texels");
    }
    let tiling = info.surface().level_tiling(0);
    for (l, rt, width) in [(1, &halves, 128), (2, &bytes, 128)] {
        let start = (l as u64 * info.layer_stride) as usize;
        let mut tiled = pattern[start..start + tiling.size_bytes()].to_vec();
        let rt = rt.surface().level_tiling(0);
        for y in 0..32 {
            for x in 0..width {
                tiled[rt.offset(x, y, 0)] = 255;
            }
        }
        let mut want = vec![0; 64 * 64 * 4];
        lichii_texture::block_linear::unswizzle(&tiling, &tiled, &mut want);
        assert_eq!(texels[l], want, "layer {l} takes the render target's bytes");
    }
    f.assert_no_validation_errors();
}

#[test]
fn rows_ending_inside_a_word_sync_on_the_gpu_and_keep_the_bytes_past_them() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let header = tic(0x08, GPU_VA, 64, 64, 4, 1, 3);
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    let pattern: Vec<u8> = (0..info.size_bytes())
        .map(|i| (i * 13 + i / 97) as u8)
        .collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    let before = f.read_image(texture);
    let mut halves = rt_info(63, 32, 2, GobLayout::Tegra);
    halves.format = PixelFormat::R16Unorm;
    let mut bytes = rt_info(125, 32, 2, GobLayout::Tegra);
    bytes.format = PixelFormat::R8Unorm;
    let layer = |l: u64| GPU_VA + l * info.layer_stride;
    clear_rt_at(&mut f, layer(1), &halves, [1.0, 0.0, 0.0, 0.0], Rect::ALL);
    clear_rt_at(&mut f, layer(2), &bytes, [1.0, 0.0, 0.0, 0.0], Rect::ALL);
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(
        after.uploads, traffic.uploads,
        "no re-upload from guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    let texels = f.read_image(texture);
    for l in [0, 3] {
        assert_eq!(texels[l], before[l], "layer {l} keeps its texels");
    }
    let tiling = info.surface().level_tiling(0);
    for (l, rt, width) in [(1, &halves, 126), (2, &bytes, 125)] {
        let start = (l as u64 * info.layer_stride) as usize;
        let mut tiled = pattern[start..start + tiling.size_bytes()].to_vec();
        let rt = rt.surface().level_tiling(0);
        for y in 0..32 {
            for x in 0..width {
                tiled[rt.offset(x, y, 0)] = 255;
            }
        }
        let mut want = vec![0; 64 * 64 * 4];
        lichii_texture::block_linear::unswizzle(&tiling, &tiled, &mut want);
        assert_eq!(texels[l], want, "layer {l} takes the render target's bytes");
    }
    f.assert_no_validation_errors();
}

#[test]
fn images_split_across_device_ranges_sync_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let split = GPU_VA + (1 << 30);
    f.gmmu
        .map(split, DADDR + 0x10_0000, 0x1_0000, 0, false)
        .unwrap();
    f.gmmu
        .map(split + 0x1_0000, DADDR + 0x30_0000, 0x1_0000, 0, false)
        .unwrap();
    let header = tic(0x08, split, 128, 128, 2, 1, 4);
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    assert_eq!(info.layer_stride, 0x1_0000);
    let pattern: Vec<u8> = (0..0x1_0000).map(|i: u32| (i * 7 + i / 89) as u8).collect();
    f.memory.write_bytes(DADDR + 0x10_0000, &pattern).unwrap();
    f.memory.write_bytes(DADDR + 0x30_0000, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    assert_eq!(f.cache.get(texture).unwrap().segments.len(), 2);
    let before = f.read_image(texture);
    let target = rt_info(128, 128, 4, GobLayout::Tegra);
    clear_rt_at(
        &mut f,
        GPU_VA + 0x30_0000,
        &target,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    );
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    let texels = f.read_image(texture);
    assert_eq!(texels[0], before[0], "the other range keeps its texels");
    assert!(
        texels[1]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [0, 255, 0, 255]),
        "the shared range takes the render target's texels"
    );
    f.assert_no_validation_errors();
}

#[test]
fn images_the_cpu_wrote_upload_those_writes_and_sync_overlaps_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let header = tic(0x08, GPU_VA, 128, 128, 2, 1, 4);
    let pattern: Vec<u8> = (0..0x2_0000).map(|i: u32| (i * 7 + i / 89) as u8).collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    f.memory.write_bytes(DADDR, &[0x5a; 0x1_0000]).unwrap();
    f.cache.cpu_written(DADDR, 0x1_0000);
    let target = rt_info(128, 128, 3, GobLayout::Tegra);
    clear_rt_at(
        &mut f,
        GPU_VA + 0x1_0000,
        &target,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    );
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    let texels = f.read_image(texture);
    assert!(all_texels(&texels[0], [0x5a; 4]), "the CPU's bytes");
    assert!(all_texels(&texels[1], GREEN), "the render target's texels");
    f.assert_no_validation_errors();
}

#[test]
fn overlaps_the_cpu_wrote_sync_on_the_gpu_with_those_writes() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let header = tic(0x08, GPU_VA, 128, 128, 2, 1, 4);
    let (texture, _) = f.texture(&header);
    let target = rt_info(128, 128, 3, GobLayout::Tegra);
    let rt = clear_rt_at(
        &mut f,
        GPU_VA + 0x1_0000,
        &target,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    );
    f.memory.write_bytes(DADDR + 0x1_0008, &[9; 4]).unwrap();
    {
        let (mut tx, cache, _) = f.tx();
        cache.invalidate(&mut tx, DADDR + 0x1_0008, 4).unwrap();
    }
    assert!(f.cache.state(rt.id).unwrap().cpu_dirty);
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    let layer = &f.read_image(texture)[1];
    assert_eq!(count_texels(layer, [9; 4]), 1, "the CPU's bytes");
    assert_eq!(
        count_texels(layer, GREEN),
        layer.len() / 4 - 1,
        "the render target's texels"
    );
    f.assert_no_validation_errors();
}

#[test]
fn single_slice_volumes_of_a_deeper_block_sync_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let mut header = tic(0x08, GPU_VA, 32, 32, 1, 1, 0);
    header.0[3] |= 4 << 6;
    header.0[4] = 31 | 2 << 23;
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    assert_eq!(info.kind, ImageKind::D3);
    let pattern: Vec<u8> = (0..info.surface().level_tiling(0).size_bytes())
        .map(|i| (i * 11 + i / 61) as u8)
        .collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    let target = rt_info(32, 8, 0, GobLayout::Tegra);
    clear_rt_at(&mut f, GPU_VA, &target, [0.0, 0.0, 1.0, 1.0], Rect::ALL);
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    let tiling = info.surface().level_tiling(0);
    assert_eq!(tiling.block_depth_log2, 4);
    let mut tiled = pattern[..tiling.size_bytes()].to_vec();
    let rt = target.surface().level_tiling(0);
    for y in 0..8 {
        for x in 0..32 * 4 {
            tiled[rt.offset(x, y, 0)] = [0, 0, 255, 255][x as usize % 4];
        }
    }
    let mut want = vec![0; 32 * 32 * 4];
    lichii_texture::block_linear::unswizzle(&tiling, &tiled, &mut want);
    assert_eq!(f.read_image(texture)[0], want);
    f.assert_no_validation_errors();
}

#[test]
fn volumes_deeper_than_a_slice_sync_every_slice_on_the_gpu() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let mut header = tic(0x08, GPU_VA, 32, 32, 1, 1, 0);
    header.0[3] |= 4 << 6;
    header.0[4] = 31 | 2 << 23;
    header.0[5] = 31 | 19 << 16;
    let info = ImageInfo::from_texture(&header.decode().unwrap(), GobLayout::Tegra).unwrap();
    assert_eq!((info.kind, info.depth), (ImageKind::D3, 20));
    let tiling = info.surface().level_tiling(0);
    assert_eq!(tiling.block_depth_log2, 4);
    let pattern: Vec<u8> = (0..tiling.size_bytes())
        .map(|i| (i * 11 + i / 61) as u8)
        .collect();
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    let slab = tiling.size_bytes() / 2;
    let target = rt_info(32, 8, 0, GobLayout::Tegra);
    clear_rt_at(
        &mut f,
        GPU_VA + slab as u64,
        &target,
        [0.0, 0.0, 1.0, 1.0],
        Rect::ALL,
    );
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    let mut tiled = pattern.clone();
    let rt = target.surface().level_tiling(0);
    for y in 0..8 {
        for x in 0..32 * 4 {
            tiled[slab + rt.offset(x, y, 0)] = [0, 0, 255, 255][x as usize % 4];
        }
    }
    let mut want = vec![0; 32 * 32 * 20 * 4];
    lichii_texture::block_linear::unswizzle(&tiling, &tiled, &mut want);
    assert_eq!(f.read_image(texture)[0], want);
    f.assert_no_validation_errors();
}

#[test]
fn subresources_split_across_device_ranges_sync_every_part() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    for va in [GPU_VA + (1 << 30), GPU_VA + (2 << 30)] {
        f.gmmu
            .map(va, DADDR + 0x10_0000, 0x1_0000, 0, false)
            .unwrap();
        f.gmmu
            .map(va + 0x1_0000, DADDR + 0x30_0000, 0x1_0000, 0, false)
            .unwrap();
    }
    let header = tic(0x08, GPU_VA + (1 << 30), 128, 256, 1, 1, 4);
    let pattern: Vec<u8> = (0..0x1_0000).map(|i: u32| (i * 7 + i / 89) as u8).collect();
    f.memory.write_bytes(DADDR + 0x10_0000, &pattern).unwrap();
    f.memory.write_bytes(DADDR + 0x30_0000, &pattern).unwrap();
    let (texture, _) = f.texture(&header);
    assert_eq!(f.cache.get(texture).unwrap().segments.len(), 2);
    let target = rt_info(256, 128, 4, GobLayout::Tegra);
    let found = clear_rt_at(
        &mut f,
        GPU_VA + (2 << 30),
        &target,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    );
    assert_ne!(found.id, texture);
    let traffic = f.cache.traffic();
    assert_eq!(f.texture(&header).0, texture);
    let after = f.cache.traffic();
    assert_eq!(
        after.downloads, traffic.downloads,
        "no write-back to guest memory"
    );
    assert_eq!(after.gpu_syncs, traffic.gpu_syncs + 1);
    assert!(
        f.read_image(texture)[0]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|t| *t == [0, 255, 0, 255]),
        "both device ranges take the render target's texels"
    );
    f.assert_no_validation_errors();
}

#[test]
fn cached_misses_end_when_the_gmmu_maps_their_address() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let unmapped = GPU_VA + 2 * DRAM;
    let info = rt_info(64, 64, 4, GobLayout::Tegra);
    for _ in 0..2 {
        let (mut tx, cache, gmmu) = f.tx();
        assert!(
            cache
                .find(&mut tx, gmmu, unmapped, &info)
                .unwrap()
                .is_none()
        );
    }
    let miss = f.cache.lookup.iter().find(|(k, _)| k.gpu_va == unmapped);
    assert!(
        miss.is_some_and(|(_, e)| e.found.is_none()),
        "the miss is cached"
    );
    f.gmmu.map(unmapped, DADDR, 1 << 20, 0, false).unwrap();
    let (mut tx, cache, gmmu) = f.tx();
    assert!(
        cache
            .find(&mut tx, gmmu, unmapped, &info)
            .unwrap()
            .is_some(),
        "a GMMU map ends the cached miss"
    );
    f.assert_no_validation_errors();
}

#[test]
fn cached_misses_end_when_an_image_is_created() {
    let Some(mut f) = fixture(GobLayout::Tegra, AstcRecompression::Uncompressed) else {
        return;
    };
    let unmapped = GPU_VA + 2 * DRAM;
    let info = rt_info(64, 64, 4, GobLayout::Tegra);
    let miss = |cache: &TextureCache| {
        cache
            .lookup
            .iter()
            .find(|(k, _)| k.gpu_va == unmapped)
            .map(|(_, e)| (e.found.is_none(), e.creations == cache.creations))
    };
    let (mut tx, cache, gmmu) = f.tx();
    assert!(
        cache
            .find(&mut tx, gmmu, unmapped, &info)
            .unwrap()
            .is_none()
    );
    assert_eq!(miss(cache), Some((true, true)));
    assert!(cache.find(&mut tx, gmmu, GPU_VA, &info).unwrap().is_some());
    assert_eq!(
        miss(cache),
        Some((true, false)),
        "a created image ends the cached miss"
    );
    assert!(
        cache
            .find(&mut tx, gmmu, unmapped, &info)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        miss(cache),
        Some((true, true)),
        "the next lookup caches the miss again"
    );
    f.assert_no_validation_errors();
}

/// Colour targets bound one layer at a time, at their layer's address, which a later 2D-array
/// or cube texture over the same memory samples as its layers.
#[test]
fn layer_targets_rendered_one_at_a_time_form_the_array_and_cube_textures_over_them() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for (layers, texture_type, size, bh) in [(5, 5, 128, 4), (6, 3, 64, 3), (2, 5, 256, 4)] {
            let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                return;
            };
            let mut header = tic(0x08, GPU_VA, size, size, layers, 1, bh);
            header.0[4] = (header.0[4] & !(0xf << 23)) | texture_type << 23;
            let desc = header.decode().unwrap();
            let array = ImageInfo::from_texture(&desc, gob).unwrap();
            let info = rt_info(size, size, bh, gob);
            for layer in 0..layers {
                let value = (layer + 1) as f32 / 8.0;
                clear_rt_at(
                    &mut f,
                    GPU_VA + u64::from(layer) * array.layer_stride,
                    &info,
                    [value, 0.0, 1.0 - value, 1.0],
                    Rect::ALL,
                );
            }
            let (id, view) = f.texture(&header);
            let texels = f.read_image(id);
            for layer in 0..layers {
                let value = ((layer + 1) as f32 / 8.0 * 255.0).round() as u8;
                let got = &texels[(view.base_layer + layer) as usize];
                assert!(
                    got.as_chunks::<4>()
                        .0
                        .iter()
                        .all(|&t| unorm8_close(t, [value, 0, 255 - value, 255])),
                    "{gob:?} {size}x{size}x{layers} type {texture_type}: layer {layer} holds {:?}",
                    &got[..4]
                );
            }
            f.assert_no_validation_errors();
        }
    }
}

fn unorm8_close(a: [u8; 4], b: [u8; 4]) -> bool {
    a.iter().zip(b).all(|(&x, y)| x.abs_diff(y) <= 1)
}

/// The same as above for the formats MKW's light probes use, whose targets read back as one
/// uniform value per layer, distinct between layers and never zero.
#[test]
fn layer_targets_of_float_and_snorm_formats_form_the_textures_over_them() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for (format, code, kind) in [
            (PixelFormat::R16G16B16A16Float, 0x03, 7),
            (PixelFormat::B10G11R11Float, 0x21, 7),
            (PixelFormat::R8Snorm, 0x1d, 1),
        ] {
            for (layers, texture_type, size, bh) in [(5, 5, 128, 4), (6, 3, 64, 3)] {
                let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                    return;
                };
                let mut header = tic(code, GPU_VA, size, size, layers, 1, bh);
                header.0[0] = (header.0[0] & !(0xfff << 7))
                    | kind << 7
                    | kind << 10
                    | kind << 13
                    | kind << 16;
                header.0[4] = (header.0[4] & !(0xf << 23)) | texture_type << 23;
                let desc = header.decode().unwrap();
                let array = ImageInfo::from_texture(&desc, gob).unwrap();
                assert_eq!(array.format, format);
                let s = surface(format, size, size, 1, 1, bh, gob);
                let info = ImageInfo {
                    format,
                    layers: 1,
                    layer_stride: s.layer_stride(),
                    ..rt_info(size, size, bh, gob)
                };
                for layer in 0..layers {
                    let value = (layer + 1) as f32 / 8.0;
                    clear_rt_at(
                        &mut f,
                        GPU_VA + u64::from(layer) * array.layer_stride,
                        &info,
                        [value, 0.25, 1.0 - value, 1.0],
                        Rect::ALL,
                    );
                }
                let (id, view) = f.texture(&header);
                let texels = f.read_image(id);
                let texel = format.bytes_per_block() as usize;
                let mut seen = Vec::new();
                for layer in 0..layers {
                    let got = &texels[(view.base_layer + layer) as usize];
                    let first = &got[..texel];
                    assert!(
                        first.iter().any(|&b| b != 0)
                            && got.chunks(texel).all(|t| t == first)
                            && !seen.contains(&first.to_vec()),
                        "{gob:?} {format:?} {size}x{size}x{layers} type {texture_type}: layer {layer} holds {first:?}"
                    );
                    seen.push(first.to_vec());
                }
                f.assert_no_validation_errors();
            }
        }
    }
}

/// Every level of every face of a mipmapped cube rendered as its own target, at the address
/// and block height the level has in the cube, then sampled as the cube.
#[test]
fn level_and_face_targets_form_the_mipmapped_cube_over_them() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for (format, code, kind) in [
            (PixelFormat::A8B8G8R8Unorm, 0x08, 2),
            (PixelFormat::B10G11R11Float, 0x21, 7),
            (PixelFormat::R16G16B16A16Float, 0x03, 7),
        ] {
            let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                return;
            };
            let (size, levels, bh) = (64, 7, 3);
            let mut header = tic(code, GPU_VA, size, size, 1, levels, bh);
            header.0[0] =
                (header.0[0] & !(0xfff << 7)) | kind << 7 | kind << 10 | kind << 13 | kind << 16;
            header.0[4] = (header.0[4] & !(0xf << 23)) | 3 << 23;
            let desc = header.decode().unwrap();
            let cube = ImageInfo::from_texture(&desc, gob).unwrap();
            assert_eq!((cube.layers, cube.levels), (6, levels));
            let layout = cube.surface();
            for layer in 0..6 {
                for level in 0..levels {
                    let tiling = layout.level_tiling(level);
                    let (w, h) = ((size >> level).max(1), (size >> level).max(1));
                    let s = surface(format, w, h, 1, 1, tiling.block_height_log2, gob);
                    let info = ImageInfo {
                        format,
                        width: w,
                        height: h,
                        layer_stride: s.layer_stride(),
                        ..rt_info(w, h, tiling.block_height_log2, gob)
                    };
                    let value = (layer * levels + level + 1) as f32 / 64.0;
                    clear_rt_at(
                        &mut f,
                        GPU_VA + cube.subresource_offset(level, layer),
                        &info,
                        [value, 0.25, 1.0 - value, 1.0],
                        Rect::ALL,
                    );
                }
            }
            let (id, view) = f.texture(&header);
            assert_eq!(view.base_layer, 0);
            let texels = f.read_image(id);
            let texel = format.bytes_per_block() as usize;
            let mut seen = Vec::new();
            for layer in 0..6 {
                for level in 0..levels {
                    let got = &texels[(layer * levels + level) as usize];
                    let first = &got[..texel];
                    assert!(
                        first.iter().any(|&b| b != 0)
                            && got.chunks(texel).all(|t| t == first)
                            && !seen.contains(&first.to_vec()),
                        "{gob:?} {format:?}: face {layer} level {level} holds {first:?}"
                    );
                    seen.push(first.to_vec());
                }
            }
            f.assert_no_validation_errors();
        }
    }
}
/// The same with the cube sampled before its faces and levels are rendered.
#[test]
fn level_and_face_targets_render_into_the_mipmapped_cube_sampled_before_them() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for (format, code, kind) in [
            (PixelFormat::A8B8G8R8Unorm, 0x08, 2),
            (PixelFormat::B10G11R11Float, 0x21, 7),
            (PixelFormat::R16G16B16A16Float, 0x03, 7),
        ] {
            let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                return;
            };
            let (size, levels, bh) = (64, 7, 3);
            let mut header = tic(code, GPU_VA, size, size, 1, levels, bh);
            header.0[0] =
                (header.0[0] & !(0xfff << 7)) | kind << 7 | kind << 10 | kind << 13 | kind << 16;
            header.0[4] = (header.0[4] & !(0xf << 23)) | 3 << 23;
            let desc = header.decode().unwrap();
            let cube = ImageInfo::from_texture(&desc, gob).unwrap();
            assert_eq!((cube.layers, cube.levels), (6, levels));
            let layout = cube.surface();
            let (first, _) = f.texture(&header);
            for layer in 0..6 {
                for level in 0..levels {
                    let tiling = layout.level_tiling(level);
                    let (w, h) = ((size >> level).max(1), (size >> level).max(1));
                    let s = surface(format, w, h, 1, 1, tiling.block_height_log2, gob);
                    let info = ImageInfo {
                        format,
                        width: w,
                        height: h,
                        layer_stride: s.layer_stride(),
                        ..rt_info(w, h, tiling.block_height_log2, gob)
                    };
                    let value = (layer * levels + level + 1) as f32 / 64.0;
                    clear_rt_at(
                        &mut f,
                        GPU_VA + cube.subresource_offset(level, layer),
                        &info,
                        [value, 0.25, 1.0 - value, 1.0],
                        Rect::ALL,
                    );
                }
            }
            let (id, view) = f.texture(&header);
            assert_eq!(id, first, "the faces render into the cube");
            assert_eq!(view.base_layer, 0);
            let texels = f.read_image(id);
            let texel = format.bytes_per_block() as usize;
            let mut seen = Vec::new();
            for layer in 0..6 {
                for level in 0..levels {
                    let got = &texels[(layer * levels + level) as usize];
                    let first = &got[..texel];
                    assert!(
                        first.iter().any(|&b| b != 0)
                            && got.chunks(texel).all(|t| t == first)
                            && !seen.contains(&first.to_vec()),
                        "{gob:?} {format:?}: face {layer} level {level} holds {first:?}"
                    );
                    seen.push(first.to_vec());
                }
            }
            f.assert_no_validation_errors();
        }
    }
}

/// A single-layer block-linear colour target as a draw's register state gives it.
fn mkw_target(format: u32, w: u32, h: u32, bh: u32, array_pitch: u32, gob: GobLayout) -> ImageInfo {
    use lichii_gpu::engines::maxwell_3d::regs::{RenderTargetConfig, TileMode};
    let mut rt: RenderTargetConfig = bytemuck::Zeroable::zeroed();
    rt.address_low = 0x10_0000;
    rt.width = w;
    rt.height = h;
    rt.format = format;
    rt.tile_mode = TileMode(bh << 4);
    rt.bits.0 = 1;
    rt.array_pitch = array_pitch;
    ImageInfo::from_render_target(&rt, 0, gob).unwrap()
}

/// Mario Kart World's light-probe cube array: two B10G11R11 128x128 cubes of 4 levels, 12
/// layers 0x16000 apart. Each draw renders one level of one cube with its six faces bound as
/// C0..C5, every face a single-layer target at its layer and level offset; the forward
/// shaders then sample the whole cube array.
fn mkw_cube_array(gob: GobLayout) -> (TicEntry, ImageInfo) {
    let mut header = tic(0x21, GPU_VA, 128, 128, 2, 4, 4);
    header.0[0] = (header.0[0] & !(0xfff << 7)) | 7 << 7 | 7 << 10 | 7 << 13 | 7 << 16;
    header.0[4] = (header.0[4] & !(0xf << 23)) | 8 << 23;
    let desc = header.decode().unwrap();
    assert_eq!(desc.texture_type, TextureType::CubeArray);
    let array = ImageInfo::from_texture(&desc, gob).unwrap();
    assert_eq!((array.layers, array.levels), (12, 4));
    assert_eq!(array.layer_stride, 0x16000, "{gob:?}");
    assert_eq!(
        [1, 2, 3].map(|l| array.subresource_offset(l, 0)),
        [0x10000, 0x14000, 0x15000],
        "{gob:?}"
    );
    assert_eq!(array.subresource_offset(0, 6), 0x84000, "{gob:?}");
    (header, array)
}

/// Renders every level of both cubes as six-target draws, resolving all six faces before
/// clearing any as a draw binds them. Returns, per subresource, the image, level and layer
/// each face resolved to.
fn render_mkw_cube_faces(
    f: &mut Fixture,
    array: &ImageInfo,
    target_bh: impl Fn(u32) -> u32,
    green: f32,
    gob: GobLayout,
) -> Vec<Found> {
    let layout = array.surface();
    let mut founds = Vec::new();
    for cube in 0..2 {
        for level in 0..4 {
            let size = 128 >> level;
            let bh = target_bh(layout.level_tiling(level).block_height_log2);
            let info = mkw_target(0xe0, size, size, bh, 0x5800, gob);
            let (mut tx, cache, gmmu) = f.tx();
            let faces: Vec<Found> = (0..6)
                .map(|face| {
                    let address = GPU_VA + array.subresource_offset(level, cube * 6 + face);
                    cache
                        .render_target(&mut tx, gmmu, address, &info, false)
                        .unwrap()
                        .unwrap()
                })
                .collect();
            for (face, &target) in faces.iter().enumerate() {
                assert!(
                    cache.state(target.id).is_some(),
                    "{gob:?} cube {cube} level {level}: C{face}'s image died resolving the later faces"
                );
                let value = ((cube * 6 + face as u32) * 4 + level + 1) as f32 / 64.0;
                cache
                    .clear(
                        &mut tx,
                        target,
                        ClearValue::Color {
                            value: vk::ClearColorValue {
                                float32: [value, green, 1.0 - value, 1.0],
                            },
                            mask: vk::ColorComponentFlags::RGBA,
                        },
                        Rect::ALL,
                        0,
                        1,
                    )
                    .unwrap();
            }
            founds.extend(faces);
        }
    }
    founds
}

/// Whether every subresource of the cube array reads back uniform, nonzero and distinct
/// from the others and from `before`; the failures as text.
fn check_mkw_cube(texels: &[Vec<u8>], before: Option<&[Vec<u8>]>) -> Vec<String> {
    let mut seen = Vec::new();
    let mut failures = Vec::new();
    for layer in 0..12 {
        for level in 0..4 {
            let i = (layer * 4 + level) as usize;
            let got = &texels[i];
            let first = &got[..4];
            if first.iter().all(|&b| b == 0)
                || !got.chunks(4).all(|t| t == first)
                || seen.contains(&first.to_vec())
                || before.is_some_and(|b| &b[i][..4] == first)
            {
                failures.push(format!("layer {layer} level {level} holds {first:?}"));
            }
            seen.push(first.to_vec());
        }
    }
    failures
}

#[test]
fn mkw_cube_array_faces_rendered_before_the_array_join_it_and_then_render_into_it() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let (header, array) = mkw_cube_array(gob);
        let faces = render_mkw_cube_faces(&mut f, &array, |bh| bh, 0.25, gob);
        assert!(faces.iter().all(|t| (t.level, t.layer) == (0, 0)));
        let (id, view) = f.texture(&header);
        assert_eq!(
            (view.view_type, view.base_layer, view.layers, view.levels),
            (vk::ImageViewType::CUBE_ARRAY, 0, 12, 4)
        );
        let live: Vec<_> = faces
            .iter()
            .filter(|t| t.id != id && f.cache.state(t.id).is_some())
            .collect();
        assert!(
            live.is_empty(),
            "{gob:?}: face targets left beside the array: {live:?}"
        );
        let first = f.read_image(id);
        let failures = check_mkw_cube(&first, None);
        assert!(failures.is_empty(), "{gob:?} joined: {failures:#?}");
        let again = render_mkw_cube_faces(&mut f, &array, |bh| bh, 0.5, gob);
        for (i, t) in again.iter().enumerate() {
            let (cube, level, face) = (i as u32 / 24, i as u32 / 6 % 4, i as u32 % 6);
            assert_eq!(
                (t.id, t.level, t.layer),
                (id, level, cube * 6 + face),
                "{gob:?} cube {cube} level {level} face {face}"
            );
        }
        assert_eq!(f.texture(&header).0, id);
        let failures = check_mkw_cube(&f.read_image(id), Some(&first));
        assert!(failures.is_empty(), "{gob:?} second frame: {failures:#?}");
        f.assert_no_validation_errors();
    }
}

#[test]
fn mkw_cube_array_faces_render_into_the_array_sampled_before_them() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let (header, array) = mkw_cube_array(gob);
        let (id, _) = f.texture(&header);
        let faces = render_mkw_cube_faces(&mut f, &array, |bh| bh, 0.25, gob);
        for (i, t) in faces.iter().enumerate() {
            let (cube, level, face) = (i as u32 / 24, i as u32 / 6 % 4, i as u32 % 6);
            assert_eq!(
                (t.id, t.level, t.layer),
                (id, level, cube * 6 + face),
                "{gob:?} cube {cube} level {level} face {face}"
            );
        }
        assert_eq!(f.texture(&header).0, id);
        let failures = check_mkw_cube(&f.read_image(id), None);
        assert!(failures.is_empty(), "{gob:?}: {failures:#?}");
        f.assert_no_validation_errors();
    }
}

/// The same with every level's targets keeping the base level's block height, as a target
/// set up from the cube's own header would. The C++ texture cache does not compare block
/// sizes when it matches a target to a level.
#[test]
fn mkw_cube_array_level_targets_at_the_base_block_height_render_into_the_array() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let (header, array) = mkw_cube_array(gob);
        let (id, _) = f.texture(&header);
        let faces = render_mkw_cube_faces(&mut f, &array, |_| 4, 0.25, gob);
        let mut failures = Vec::new();
        for (i, t) in faces.iter().enumerate() {
            let (cube, level, face) = (i as u32 / 24, i as u32 / 6 % 4, i as u32 % 6);
            if (t.id, t.level, t.layer) != (id, level, cube * 6 + face) {
                failures.push(format!("cube {cube} level {level} face {face} -> {t:?}"));
            }
        }
        assert_eq!(f.texture(&header).0, id);
        failures.extend(check_mkw_cube(&f.read_image(id), None));
        assert!(failures.is_empty(), "{gob:?} (array {id:?}): {failures:#?}");
        f.assert_no_validation_errors();
    }
}

/// Mario Kart World's 832x466x2 A8B8G8R8 array: each layer is rendered as its own target,
/// layer 1 at the padded layer stride 0x1a0000, and the pair sampled as a 2D array.
#[test]
fn mkw_npot_array_layers_rendered_as_targets_form_and_render_into_the_array() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for sample_first in [false, true] {
            let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                return;
            };
            let header = tic(0x08, GPU_VA, 832, 466, 2, 1, 4);
            let desc = header.decode().unwrap();
            let array = ImageInfo::from_texture(&desc, gob).unwrap();
            assert_eq!((array.layers, array.layer_stride), (2, 0x1a0000), "{gob:?}");
            let info = mkw_target(0xd5, 832, 466, 4, 0x1a0000 / 4, gob);
            let early = sample_first.then(|| f.texture(&header).0);
            let render = |f: &mut Fixture, blue: f32| -> Vec<Found> {
                (0..2u32)
                    .map(|layer| {
                        clear_rt_at(
                            f,
                            GPU_VA + u64::from(layer) * 0x1a0000,
                            &info,
                            [(layer + 1) as f32 / 4.0, 0.0, blue, 1.0],
                            Rect::ALL,
                        )
                    })
                    .collect()
            };
            let founds = render(&mut f, 0.25);
            let (id, view) = f.texture(&header);
            let case = format!("{gob:?} sample_first {sample_first}");
            assert_eq!(early.unwrap_or(id), id, "{case}");
            assert_eq!((view.base_layer, view.layers), (0, 2), "{case}");
            if sample_first {
                assert_eq!(
                    founds
                        .iter()
                        .map(|t| (t.id, t.level, t.layer))
                        .collect::<Vec<_>>(),
                    [(id, 0, 0), (id, 0, 1)],
                    "{case}"
                );
            } else {
                let live: Vec<_> = founds
                    .iter()
                    .filter(|t| t.id != id && f.cache.state(t.id).is_some())
                    .collect();
                assert!(
                    live.is_empty(),
                    "{case}: targets left beside the array: {live:?}"
                );
            }
            let check = |texels: &[Vec<u8>], blue: u8| {
                for (layer, got) in texels.iter().enumerate() {
                    let red = ((layer + 1) as f32 / 4.0 * 255.0).round() as u8;
                    assert!(
                        got.as_chunks::<4>()
                            .0
                            .iter()
                            .all(|&t| unorm8_close(t, [red, 0, blue, 255])),
                        "{case}: layer {layer} holds {:?}",
                        &got[..4]
                    );
                }
            };
            check(&f.read_image(id), 64);
            let again = render(&mut f, 0.75);
            assert_eq!(
                again
                    .iter()
                    .map(|t| (t.id, t.level, t.layer))
                    .collect::<Vec<_>>(),
                [(id, 0, 0), (id, 0, 1)],
                "{case} second frame"
            );
            assert_eq!(f.texture(&header).0, id, "{case}");
            check(&f.read_image(id), 191);
            f.assert_no_validation_errors();
        }
    }
}

const CHUNK: u64 = 1 << 20;

/// Maps `size` bytes at `gpu_va` to the device pages at `daddr` in 1 MiB chunks, chunk `i`
/// to device chunk `place(i, chunks)`: device backing scattered as Mario Kart World's 16 MiB
/// chunks are. Returns the device offset of each chunk.
fn map_scattered(
    f: &Fixture,
    gpu_va: GpuVAddr,
    daddr: DAddr,
    size: u64,
    place: fn(u64, u64) -> u64,
) -> Vec<u64> {
    let chunks = size.div_ceil(CHUNK);
    (0..chunks)
        .map(|i| {
            let at = place(i, chunks) * CHUNK;
            f.gmmu
                .map(gpu_va + i * CHUNK, daddr + at, CHUNK, 0, false)
                .unwrap();
            at
        })
        .collect()
}

/// `bytes` written through the mapping `from` and read back through the mapping `to`, both
/// chunk maps from [`map_scattered`]; device bytes `from` does not reach read `fill`.
fn remap(bytes: &[u8], from: &[u64], to: &[u64], fill: u8) -> Vec<u8> {
    let chunk = CHUNK as usize;
    let mut device = vec![fill; (from.iter().chain(to).max().unwrap() + CHUNK) as usize];
    for (i, &at) in from.iter().enumerate() {
        let src = &bytes[(i * chunk).min(bytes.len())..((i + 1) * chunk).min(bytes.len())];
        device[at as usize..at as usize + src.len()].copy_from_slice(src);
    }
    to.iter()
        .flat_map(|&at| device[at as usize..at as usize + chunk].to_vec())
        .collect()
}

/// The guest bytes a single-level render target holding `linear` stores.
fn target_bytes(info: &ImageInfo, linear: &[u8]) -> Vec<u8> {
    let tiling = info.surface().level_tiling(0);
    let mut tiled = vec![0; info.size_bytes() as usize];
    swizzle(&tiling, linear, &mut tiled[..tiling.size_bytes()]);
    tiled
}

/// Every subresource of `info` read from guest bytes `bytes`, layer-major.
fn subresources_of(info: &ImageInfo, bytes: &[u8]) -> Vec<Vec<u8>> {
    let surface = info.surface();
    let mut out = Vec::new();
    for layer in 0..info.layers {
        for level in 0..info.levels {
            let tiling = surface.level_tiling(level);
            let at = info.subresource_offset(level, layer) as usize;
            let mut linear = vec![0; tiling.linear_size_bytes()];
            unswizzle(&tiling, &bytes[at..at + tiling.size_bytes()], &mut linear);
            out.push(linear);
        }
    }
    out
}

/// Mario Kart World's street light grid, scaled down 4x: a single-sample A8B8G8R8 Snorm
/// 6144x6144 target written through one GPU mapping, and a 3072x3072x2 array of 4 levels in
/// the same format sampled through another mapping of the same device pages, scattered in
/// chunks. The array reads the target's bytes in its own layout.
#[test]
fn arrays_sampled_through_another_mapping_of_a_target_read_its_bytes_in_their_layout() {
    for scale in [ResolutionScale::X1, ResolutionScale::X2] {
        read_target_through_another_mapping(768, 64 << 20, scale);
    }
}

/// The same at Mario Kart World's own size.
#[test]
fn mkw_light_grid_sampled_through_another_mapping_reads_its_target_bytes() {
    read_target_through_another_mapping(3072, 192 << 20, ResolutionScale::X1);
}

fn read_target_through_another_mapping(size: u32, dram: u64, scale: ResolutionScale) {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for array_first in [true, false] {
            let Some(mut f) = fixture_with_dram(gob, AstcRecompression::Uncompressed, dram) else {
                return;
            };
            f.cache.set_resolution_scale(scale);
            let (writer, reader) = (GPU_VA + (1 << 30), GPU_VA + (2 << 30));
            let target = mkw_target(0xd7, 2 * size, 2 * size, 4, 0, gob);
            assert_eq!(target.format, PixelFormat::A8B8G8R8Snorm);
            let mut header = tic(0x08, reader, size, size, 2, 4, 4);
            header.0[0] = (header.0[0] & !(0xfff << 7)) | 1 << 7 | 1 << 10 | 1 << 13 | 1 << 16;
            let array = ImageInfo::from_texture(&header.decode().unwrap(), gob).unwrap();
            assert!(array.size_bytes() < target.size_bytes());
            let from = map_scattered(
                &f,
                writer,
                DADDR + (16 << 20),
                target.size_bytes(),
                |i, n| n - 1 - i,
            );
            let to = map_scattered(&f, reader, DADDR + (16 << 20), array.size_bytes(), |i, _| i);
            let case = format!("{gob:?} {scale:?} array_first {array_first}");
            let early = array_first.then(|| f.texture(&header).0);
            let quarter = |f: &mut Fixture, color: [f32; 4]| {
                clear_rt_at(f, writer, &target, [0.5, -0.5, 0.25, 1.0], Rect::ALL);
                clear_rt_at(
                    f,
                    writer,
                    &target,
                    color,
                    Rect {
                        x: 0,
                        y: 0,
                        width: size,
                        height: size,
                    },
                )
            };
            for color in [[-1.0, 1.0, 0.0, 0.5], [0.75, 0.0, -0.75, 1.0]] {
                let rt = quarter(&mut f, color);
                let host = f.read_image(rt.id).swap_remove(0);
                let (inside, outside) = (&host[..4], &host[host.len() - 4..]);
                let written: Vec<u8> = (0..4 * size * size)
                    .flat_map(|i| {
                        let (x, y) = (i % (2 * size), i / (2 * size));
                        if x < size && y < size {
                            inside
                        } else {
                            outside
                        }
                        .to_vec()
                    })
                    .collect();
                let expected = subresources_of(
                    &array,
                    &remap(&target_bytes(&target, &written), &from, &to, 0),
                );
                let (id, _) = f.texture(&header);
                assert_eq!(early.unwrap_or(id), id, "{case}");
                let got = f.read_image(id);
                assert_eq!(got.len(), expected.len(), "{case}");
                for (i, (got, expected)) in got.iter().zip(&expected).enumerate() {
                    let (layer, level) = (i as u32 / 4, i as u32 % 4);
                    let wrong = got
                        .chunks(4)
                        .zip(expected.chunks(4))
                        .filter(|(g, e)| g != e)
                        .count();
                    assert_eq!(
                        wrong,
                        0,
                        "{case} {color:?}: layer {layer} level {level}: {wrong} of {} texels differ, first {:?} vs {:?}",
                        got.len() / 4,
                        &got[..4],
                        &expected[..4]
                    );
                }
            }
            {
                let (mut tx, cache, _) = f.tx();
                cache
                    .flush(&mut tx, DADDR + (16 << 20), target.size_bytes())
                    .unwrap();
            }
            f.assert_no_validation_errors();
        }
    }
}

/// Mario Kart World's colour-grading LUTs: a 32x32x32 RGBA16F volume whose blocks are one GOB
/// tall and 16 deep, rendered four draws of eight targets at a time, each target a single slice
/// at the volume's base plus the slice's offset inside the 16-slice block.
fn mkw_lut() -> TicEntry {
    let mut header = tic(0x03, GPU_VA, 32, 32, 1, 1, 0);
    header.0[0] = (header.0[0] & !(0xfff << 7)) | 7 << 7 | 7 << 10 | 7 << 13 | 7 << 16;
    header.0[3] |= 4 << 6;
    header.0[4] = 31 | 2 << 23;
    header.0[5] = 31 | 31 << 16;
    header
}

fn mkw_lut_slice(volume: bool, gob: GobLayout) -> ImageInfo {
    use lichii_gpu::engines::maxwell_3d::regs::{RenderTargetConfig, TileMode};
    let mut rt: RenderTargetConfig = bytemuck::Zeroable::zeroed();
    rt.address_low = 0x10_0000;
    rt.width = 32;
    rt.height = 32;
    rt.format = 0xca;
    rt.tile_mode = TileMode(4 << 8 | u32::from(volume) << 16);
    rt.bits.0 = 1;
    ImageInfo::from_render_target(&rt, 0, gob).unwrap()
}

/// Renders every slice of the LUT as MKW does, eight targets resolved per draw before any is
/// cleared, slice `z` to `[value(z), 0.5, 1 - value(z), 1]`.
fn render_mkw_lut(
    f: &mut Fixture,
    volume: bool,
    gob: GobLayout,
    value: fn(u32) -> f32,
) -> Vec<Found> {
    let info = mkw_lut_slice(volume, gob);
    let past_lut = GPU_VA + 0x4_0000;
    f.gmmu
        .map(past_lut, DADDR + (DRAM >> 1), 0x1_0000, 0, false)
        .unwrap();
    let mut founds = Vec::new();
    for draw in 0..4 {
        let (mut tx, cache, gmmu) = f.tx();
        let targets: Vec<Found> = (0..8)
            .map(|i| {
                let z = draw * 8 + i;
                let address = GPU_VA + u64::from(z / 16) * 0x2_0000 + u64::from(z % 16) * 0x200;
                cache
                    .render_target(&mut tx, gmmu, address, &info, false)
                    .unwrap()
                    .unwrap()
            })
            .collect();
        for (i, &target) in targets.iter().enumerate() {
            let z = draw * 8 + i as u32;
            assert!(cache.state(target.id).is_some(), "slice {z}'s image died");
            cache
                .clear(
                    &mut tx,
                    target,
                    ClearValue::Color {
                        value: vk::ClearColorValue {
                            float32: [value(z), 0.5, 1.0 - value(z), 1.0],
                        },
                        mask: vk::ColorComponentFlags::RGBA,
                    },
                    Rect::ALL,
                    0,
                    1,
                )
                .unwrap();
        }
        founds.extend(targets);
    }
    founds
}

/// The slices of the LUT that do not read back uniform at the value slice `z` was rendered.
fn wrong_lut_slices(f: &mut Fixture, id: ImageId, value: fn(u32) -> f32) -> Vec<String> {
    let texels = f.read_image(id).swap_remove(0);
    let slice = 32 * 32 * 8;
    assert_eq!(texels.len(), 32 * slice);
    let half = |v: f32| {
        let bits = v.to_bits();
        let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let half = ((bits >> 16) & 0x8000) | ((exponent as u32) << 10) | ((bits >> 13) & 0x3ff);
        (half as u16).to_le_bytes()
    };
    (0..32u32)
        .filter_map(|z| {
            let want: Vec<u8> = [value(z), 0.5, 1.0 - value(z), 1.0]
                .iter()
                .flat_map(|&v| half(v))
                .collect();
            let got = &texels[z as usize * slice..(z as usize + 1) * slice];
            let wrong = got.chunks(8).filter(|t| *t != want.as_slice()).count();
            (wrong != 0).then(|| {
                format!(
                    "slice {z}: {wrong} of 1024 texels wrong, first {:?}",
                    &got[..8]
                )
            })
        })
        .collect()
}

#[test]
fn slice_targets_rendered_before_a_volume_form_it() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for volume in [false, true] {
            let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                return;
            };
            let header = mkw_lut();
            let info = ImageInfo::from_texture(&header.decode().unwrap(), gob).unwrap();
            assert_eq!((info.kind, info.depth, info.levels), (ImageKind::D3, 32, 1));
            assert_eq!(info.surface().level_tiling(0).block_depth_log2, 4);
            let value = |z: u32| (z + 1) as f32 / 64.0;
            render_mkw_lut(&mut f, volume, gob, value);
            let (id, _) = f.texture(&header);
            let wrong = wrong_lut_slices(&mut f, id, value);
            assert!(wrong.is_empty(), "{gob:?} volume {volume}: {wrong:#?}");
            assert!(
                f.cache.state(id).unwrap().gpu_modified,
                "{gob:?} volume {volume}"
            );
            let again = render_mkw_lut(&mut f, volume, gob, |z| (z + 33) as f32 / 128.0);
            for (z, t) in again.iter().enumerate() {
                assert_eq!(
                    (t.id, t.level, t.layer),
                    (id, 0, z as u32),
                    "{gob:?} volume {volume}"
                );
            }
            assert_eq!(f.texture(&header).0, id);
            let wrong = wrong_lut_slices(&mut f, id, |z| (z + 33) as f32 / 128.0);
            assert!(
                wrong.is_empty(),
                "{gob:?} volume {volume} second frame: {wrong:#?}"
            );
            f.assert_no_validation_errors();
        }
    }
}

#[test]
fn slice_targets_render_into_the_volume_sampled_before_them() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for volume in [false, true] {
            let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                return;
            };
            let header = mkw_lut();
            let (id, _) = f.texture(&header);
            let value = |z: u32| (z + 1) as f32 / 64.0;
            let founds = render_mkw_lut(&mut f, volume, gob, value);
            for (z, t) in founds.iter().enumerate() {
                assert_eq!(
                    (t.id, t.level, t.layer),
                    (id, 0, z as u32),
                    "{gob:?} volume {volume}"
                );
            }
            assert!(f.cache.state(id).unwrap().gpu_modified);
            assert_eq!(f.texture(&header).0, id);
            let wrong = wrong_lut_slices(&mut f, id, value);
            assert!(wrong.is_empty(), "{gob:?} volume {volume}: {wrong:#?}");
            f.assert_no_validation_errors();
        }
    }
}

#[test]
fn blits_into_a_volume_slice_address_stay_off_the_volume() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let (volume, _) = f.texture(&mkw_lut());
        let slice = mkw_lut_slice(false, gob);
        let slice_va = GPU_VA + 0x200;
        let (mut tx, cache, gmmu) = f.tx();
        let found = cache
            .render_target(&mut tx, gmmu, slice_va, &slice, false)
            .unwrap()
            .unwrap();
        assert_eq!((found.id, found.layer), (volume, 1));
        let src_va = GPU_VA + (1 << 20);
        clear_rt_at(&mut f, src_va, &slice, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
        let (mut tx, cache, gmmu) = f.tx();
        cache
            .blit(
                &mut tx,
                gmmu,
                &blit_side(src_va, slice, [0, 0, 32, 32]),
                &blit_side(slice_va, slice, [0, 0, 32, 32]),
                vk::Filter::NEAREST,
            )
            .unwrap();
        f.assert_no_validation_errors();
    }
}

#[test]
fn slice_targets_join_with_their_aliases_newer_texels() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let slice = mkw_lut_slice(false, gob);
        let narrow = ImageInfo { width: 31, ..slice };
        clear_rt_at(&mut f, GPU_VA, &slice, [1.0, 0.0, 0.0, 1.0], Rect::ALL);
        clear_rt_at(&mut f, GPU_VA, &narrow, [0.0, 1.0, 0.0, 1.0], Rect::ALL);
        let (volume, _) = f.texture(&mkw_lut());
        let texels = f.read_image(volume).remove(0);
        let green = [0, 0, 0, 60, 0, 0, 0, 60];
        let red = [0, 60, 0, 0, 0, 0, 0, 60];
        let wrong = (0..32 * 32)
            .filter(|&i| {
                let want = if i % 32 < 31 { green } else { red };
                texels[i * 8..i * 8 + 8] != want
            })
            .count();
        assert_eq!(wrong, 0, "{gob:?}: slice 0 texels");
        f.assert_no_validation_errors();
    }
}

/// The TIC of a `w`x`h` array of `layers` layers and `levels` levels in `raw` format with
/// `float` components (BC6H) or sRGB (ASTC).
fn array_tic(
    raw: u32,
    srgb: bool,
    float: bool,
    gpu_va: GpuVAddr,
    size: (u32, u32, u32, u32),
) -> TicEntry {
    let (w, h, layers, levels) = size;
    let mut header = tic(raw, gpu_va, w, h, layers, levels, 3);
    if float {
        header.0[0] =
            raw | 7 << 7 | 7 << 10 | 7 << 13 | 7 << 16 | 2 << 19 | 3 << 22 | 4 << 25 | 5 << 28;
    }
    if srgb {
        header.0[4] |= 1 << 22;
    }
    header
}

#[test]
fn sampled_arrays_too_large_for_the_host_keep_their_smaller_levels_over_sparse_pages() {
    let gob = GobLayout::Switch2;
    for (raw, srgb, float, format, trim) in [
        (0x44, true, false, PixelFormat::Astc8x8Srgb, 2),
        (0x11, false, true, PixelFormat::Bc6hUfloat, 1),
    ] {
        let size = (512, 512, 4, 6);
        let (w, h, layers, levels) = size;
        let s = surface(format, w, h, levels, layers, 3, gob);
        let data = level_data(&s, 5);
        let Some(mut reference) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        write_surface(&reference.memory, &s, &data);
        let (full, _) = reference.texture(&array_tic(raw, srgb, float, GPU_VA, size));
        let want = reference.read_image(full);
        drop(reference);
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        write_surface(&f.memory, &s, &data);
        f.cache.trim_above = 1 << 20;
        f.cache.trim_to = 512 << 10;
        let va = GPU_VA + DRAM;
        let page = 0x1000;
        let bytes = s.size_bytes().next_multiple_of(page);
        f.gmmu.map_sparse(va, bytes, false).unwrap();
        let map_layer = |f: &mut Fixture, layer: u32, from: u32| {
            let lo = s.subresource_offset(from, layer) / page * page;
            let hi = (s.subresource_offset(levels - 1, layer) + s.level_size(levels - 1))
                .next_multiple_of(page);
            f.gmmu.map(va + lo, DADDR + lo, hi - lo, 0, false).unwrap();
        };
        for layer in [0, 1, 3] {
            map_layer(&mut f, layer, trim);
        }
        let header = array_tic(raw, srgb, float, va, size);
        let (id, view) = f.texture(&header);
        let extent = f.cache.state(id).unwrap().extent;
        let kept = levels - trim;
        for (base, want_base, want_levels) in [(0, 0, kept), (trim + 1, 1, kept - 1)] {
            let view = ViewInfo {
                base_level: base,
                levels: levels - base,
                ..view
            };
            assert!(f.cache.view(id, &view).unwrap().is_some(), "{format:?}");
            let key = f.cache.get(id).unwrap().views.last().unwrap().0;
            assert_eq!(
                (key.base_level, key.levels),
                (want_base, want_levels),
                "{format:?}: guest levels {base}.. view host levels from {want_base}"
            );
        }
        assert_eq!(
            extent.width,
            w >> trim,
            "{format:?}: keeps levels from {trim}"
        );
        assert_eq!(extent.height, h >> trim);
        let check = |f: &mut Fixture, unmapped: &[u32]| {
            let got = f.read_image(id);
            assert_eq!(got.len(), (layers * kept) as usize, "{format:?}");
            for layer in 0..layers {
                for level in trim..levels {
                    let texels = &got[(layer * kept + level - trim) as usize];
                    let reference = &want[(layer * levels + level) as usize];
                    if unmapped.contains(&layer) {
                        assert!(
                            texels.iter().all(|&b| b == 0),
                            "{format:?}: unmapped layer {layer} level {level} reads zero"
                        );
                    } else {
                        assert!(
                            texels == reference,
                            "{format:?}: layer {layer} level {level} holds guest level {level}"
                        );
                    }
                }
            }
        };
        check(&mut f, &[2]);
        map_layer(&mut f, 2, trim);
        let (again, _) = f.texture(&header);
        assert_eq!(again, id, "{format:?}: newly mapped pages join the image");
        check(&mut f, &[]);
        f.assert_no_validation_errors();
    }
}

#[test]
fn sampled_arrays_whose_leading_pages_are_sparse_read_zero_there() {
    let gob = GobLayout::Switch2;
    for (raw, srgb, float, format) in [
        (0x44, true, false, PixelFormat::Astc8x8Srgb),
        (0x11, false, true, PixelFormat::Bc6hUfloat),
    ] {
        let size = (256, 256, 2, 3);
        let (w, h, layers, levels) = size;
        let s = surface(format, w, h, levels, layers, 3, gob);
        let data = level_data(&s, 9);
        let Some(mut reference) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        write_surface(&reference.memory, &s, &data);
        let (full, _) = reference.texture(&array_tic(raw, srgb, float, GPU_VA, size));
        let want = reference.read_image(full);
        drop(reference);
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        write_surface(&f.memory, &s, &data);
        let va = GPU_VA + DRAM;
        let page = 0x1000;
        let bytes = s.size_bytes().next_multiple_of(page);
        let hole = s.subresource_offset(0, 1) / page * page;
        assert!(
            hole >= s.level_size(0),
            "{format:?}: layer 0 level 0 is unmapped"
        );
        f.gmmu.map_sparse(va, bytes, false).unwrap();
        f.gmmu
            .map(va + hole, DADDR + hole, bytes - hole, 0, false)
            .unwrap();
        let (id, _) = f.texture(&array_tic(raw, srgb, float, va, size));
        let got = f.read_image(id);
        assert!(
            got[0].iter().all(|&b| b == 0),
            "{format:?}: the unmapped level reads zero"
        );
        for level in 0..levels {
            let i = (levels + level) as usize;
            assert!(
                got[i] == want[i],
                "{format:?}: mapped layer 1 level {level}"
            );
        }
        f.assert_no_validation_errors();
    }
}

#[test]
fn a_remap_dirties_only_the_bytes_whose_mapping_changed() {
    let seg = |offset, daddr, size| Segment {
        offset,
        daddr,
        size,
    };
    let old = [
        seg(0, 0x1000, 100),
        seg(200, 0x5000, 100),
        seg(400, 0x9000, 50),
    ];
    let new = [
        seg(0, 0x1000, 150),
        seg(200, 0x7000, 100),
        seg(400, 0x9000, 50),
    ];
    let mut changed = Vec::new();
    remapped_bytes(&old, &new, |lo, hi| changed.push((lo, hi)));
    assert_eq!(changed, [(100, 150), (200, 300)]);
    changed.clear();
    remapped_bytes(&new, &old[..1], |lo, hi| changed.push((lo, hi)));
    assert_eq!(changed, [(100, 150), (200, 300), (400, 450)]);
}

#[test]
fn an_allocation_out_of_memory_evicts_clean_images_unused_this_frame() {
    let gob = GobLayout::Switch2;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let tics: Vec<_> = (0..6)
        .map(|i| tic(0x08, GPU_VA + i * 0x1_0000, 64, 64, 1, 1, 2))
        .collect();
    let ids: Vec<_> = tics.iter().map(|t| f.texture(t).0).collect();
    let info = rt_info(64, 64, 2, gob);
    let modified = clear_rt_at(
        &mut f,
        GPU_VA + 0x10_0000,
        &info,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    )
    .id;
    f.tick(HUGE_BUDGET);
    for t in &tics[4..] {
        f.texture(t);
    }
    let size = f.cache.get(ids[0]).unwrap().image.memory_size();
    let freed = {
        let (mut tx, cache, _) = f.tx();
        cache.evict_unused(&mut tx, u64::MAX).unwrap()
    };
    assert_eq!(freed, 4 * size);
    let live: Vec<bool> = ids.iter().map(|&id| f.cache.get(id).is_some()).collect();
    assert_eq!(
        live,
        [false, false, false, false, true, true],
        "images used this frame stay"
    );
    assert!(f.cache.get(modified).is_some(), "GPU-modified images stay");
    {
        let (mut tx, cache, _) = f.tx();
        assert_eq!(cache.evict_unused(&mut tx, u64::MAX).unwrap(), 0);
    }
    f.assert_no_validation_errors();
}

/// A xorshift generator for synthetic ASTC blocks.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn block(&mut self) -> u128 {
        u128::from(self.next()) | u128::from(self.next()) << 64
    }
}

/// The kinds of block synthetic ASTC data cycles through.
#[derive(Clone, Copy, Debug)]
enum BlockKind {
    /// A random block the CPU decodes without the error colour, with this many partitions and
    /// one or two weight planes.
    Encoded { partitions: u128, dual: bool },
    /// An LDR void-extent block of a random colour.
    VoidExtent,
    /// [`ASTC_TRANSPARENT_BLACK`].
    TransparentBlack,
    /// An HDR void-extent block, the error colour in the LDR profiles.
    HdrVoidExtent,
    /// Random bits, mostly illegal encodings.
    Raw,
}

const BLOCK_KINDS: [BlockKind; 11] = [
    BlockKind::Encoded {
        partitions: 1,
        dual: false,
    },
    BlockKind::Encoded {
        partitions: 1,
        dual: true,
    },
    BlockKind::Encoded {
        partitions: 2,
        dual: false,
    },
    BlockKind::Encoded {
        partitions: 3,
        dual: false,
    },
    BlockKind::Encoded {
        partitions: 4,
        dual: false,
    },
    BlockKind::Encoded {
        partitions: 2,
        dual: true,
    },
    BlockKind::Encoded {
        partitions: 3,
        dual: true,
    },
    BlockKind::VoidExtent,
    BlockKind::TransparentBlack,
    BlockKind::HdrVoidExtent,
    BlockKind::Raw,
];

const ASTC_ERROR: [u8; 4] = [0xFF, 0x00, 0xFF, 0xFF];
const LDR_VOID_EXTENT: u128 = 0xFFFF_FFFF_FFFF_FDFC;

fn synthetic_block(rng: &mut Rng, kind: BlockKind, bw: u32, bh: u32) -> [u8; 16] {
    let block = match kind {
        BlockKind::Encoded { partitions, dual } => (0..1_000_000)
            .map(|_| {
                let mut block = rng.block() & !(0x7 << 10);
                block |= (partitions - 1) << 11 | u128::from(dual) << 10;
                block
            })
            .find(|&block| {
                let mode = block as u32 & 0x7FF;
                let ignores_dual = mode & 3 == 0 && (mode >> 7) & 3 == 2;
                let mut texels = [[0; 4]; 144];
                astc::decode_block_rgba8(&block.to_le_bytes(), bw, bh, false, &mut texels);
                mode & 0x1FF != 0x1FC
                    && !(dual && ignores_dual)
                    && texels[..(bw * bh) as usize]
                        .iter()
                        .any(|&t| t != ASTC_ERROR)
            })
            .expect("a legal block"),
        BlockKind::VoidExtent => LDR_VOID_EXTENT | u128::from(rng.next()) << 64,
        BlockKind::TransparentBlack => u128::from_le_bytes(ASTC_TRANSPARENT_BLACK),
        BlockKind::HdrVoidExtent => LDR_VOID_EXTENT | 1 << 9 | u128::from(rng.next()) << 64,
        BlockKind::Raw => rng.block(),
    };
    block.to_le_bytes()
}

/// Linear ASTC blocks for every subresource of `surface`, cycling through [`BLOCK_KINDS`].
fn astc_level_data(surface: &SurfaceLayout, seed: u64, kinds: &[BlockKind]) -> Vec<Vec<u8>> {
    let (bw, bh) = (surface.format.block_width(), surface.format.block_height());
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut out = Vec::new();
    for layer in 0..surface.layers {
        for level in 0..surface.levels {
            let blocks = surface.level_tiling(level).linear_size_bytes() / 16;
            let data = (0..blocks)
                .flat_map(|i| {
                    let kind = kinds[(i + (layer * 5 + level) as usize) % kinds.len()];
                    synthetic_block(&mut rng, kind, bw, bh)
                })
                .collect();
            out.push(data);
        }
    }
    out
}

fn decode_astc_level(format: PixelFormat, blocks: &[u8], extent: (u32, u32)) -> Vec<u8> {
    let (w, h) = extent;
    let mut out = vec![0; (w * h * 4) as usize];
    astc::decode_rgba8(
        blocks,
        w,
        h,
        1,
        format.block_width(),
        format.block_height(),
        format.is_srgb(),
        &mut out,
    );
    out
}

/// The first texel of a `width`-wide RGBA8 image where `got` and `want` differ, for messages.
fn first_difference(got: &[u8], want: &[u8], width: u32) -> Option<(u32, u32, [u8; 4], [u8; 4])> {
    let (got, _) = got.as_chunks::<4>();
    let (want, _) = want.as_chunks::<4>();
    let i = got.iter().zip(want).position(|(a, b)| a != b)?;
    Some((i as u32 % width, i as u32 / width, got[i], want[i]))
}

const ASTC_FOOTPRINTS: [(u32, PixelFormat, PixelFormat); 5] = [
    (0x40, PixelFormat::Astc4x4Unorm, PixelFormat::Astc4x4Srgb),
    (0x42, PixelFormat::Astc6x6Unorm, PixelFormat::Astc6x6Srgb),
    (0x44, PixelFormat::Astc8x8Unorm, PixelFormat::Astc8x8Srgb),
    (
        0x45,
        PixelFormat::Astc10x10Unorm,
        PixelFormat::Astc10x10Srgb,
    ),
    (
        0x46,
        PixelFormat::Astc12x12Unorm,
        PixelFormat::Astc12x12Srgb,
    ),
];

#[test]
fn gpu_astc_decode_matches_the_cpu_decoder() {
    let gob = GobLayout::Switch2;
    let (w, h, layers, levels) = (100, 76, 2, 3);
    for scratch in [super::astc::SCRATCH_BYTES, 8 << 10] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        f.cache.set_astc_decode(AstcDecode::Gpu);
        f.cache.astc_scratch = scratch;
        for (raw, unorm, srgb_format) in ASTC_FOOTPRINTS {
            for (srgb, format) in [(false, unorm), (true, srgb_format)] {
                let host = host_format(&f.device, format, AstcRecompression::Uncompressed);
                if host.conversion != Conversion::Decode {
                    eprintln!("skipping {format:?}: the device samples it natively");
                    continue;
                }
                let s = surface(format, w, h, levels, layers, 3, gob);
                let data = astc_level_data(&s, u64::from(raw) * 2 + u64::from(srgb), &BLOCK_KINDS);
                write_surface(&f.memory, &s, &data);
                let (id, _) =
                    f.texture(&array_tic(raw, srgb, false, GPU_VA, (w, h, layers, levels)));
                let read = f.read_image(id);
                assert_eq!(read.len(), (layers * levels) as usize);
                for (i, (got, blocks)) in read.iter().zip(&data).enumerate() {
                    let (layer, level) = (i as u32 / levels, i as u32 % levels);
                    let extent = ((w >> level).max(1), (h >> level).max(1));
                    let want = decode_astc_level(format, blocks, extent);
                    assert!(
                        got == &want,
                        "{format:?} layer {layer} level {level} (scratch {scratch}): first \
                         difference (x, y, gpu, cpu) {:?}",
                        first_difference(got, &want, extent.0)
                    );
                }
                f.cache.unmap(&mut f.scheduler, DADDR, s.size_bytes());
                assert!(f.cache.is_empty());
            }
        }
        f.assert_no_validation_errors();
    }
}

#[test]
fn gpu_astc_decode_fills_trimmed_arrays_from_their_kept_levels() {
    let gob = GobLayout::Switch2;
    let format = PixelFormat::Astc8x8Srgb;
    let size = (512, 512, 4, 6);
    let (w, h, layers, levels) = size;
    let trim = 2;
    for recompression in [AstcRecompression::Uncompressed, AstcRecompression::Bc3] {
        let Some(mut f) = fixture(gob, recompression) else {
            return;
        };
        let rgba8 = host_format(&f.device, format, AstcRecompression::Uncompressed);
        if rgba8.conversion != Conversion::Decode {
            eprintln!("skipping: the device samples {format:?} natively");
            return;
        }
        f.cache.set_astc_decode(AstcDecode::Gpu);
        f.cache.trim_above = 1 << 20;
        f.cache.trim_to = 512 << 10;
        let s = surface(format, w, h, levels, layers, 3, gob);
        let data = astc_level_data(&s, 11, &BLOCK_KINDS);
        write_surface(&f.memory, &s, &data);
        let va = GPU_VA + DRAM;
        let page = 0x1000;
        f.gmmu
            .map_sparse(va, s.size_bytes().next_multiple_of(page), false)
            .unwrap();
        for layer in [0, 1, 3] {
            let lo = s.subresource_offset(trim, layer) / page * page;
            let hi = (s.subresource_offset(levels - 1, layer) + s.level_size(levels - 1))
                .next_multiple_of(page);
            f.gmmu.map(va + lo, DADDR + lo, hi - lo, 0, false).unwrap();
        }
        let (id, _) = f.texture(&array_tic(0x44, true, false, va, size));
        let image = f.cache.get(id).unwrap();
        assert_eq!(image.trim, trim, "{recompression:?}");
        assert_eq!(
            (image.host.format, image.host.conversion),
            (rgba8.format, Conversion::Decode),
            "huge arrays stay RGBA8 with {recompression:?}"
        );
        let kept = levels - trim;
        let got = f.read_image(id);
        assert_eq!(got.len(), (layers * kept) as usize);
        for layer in 0..layers {
            for level in trim..levels {
                let texels = &got[(layer * kept + level - trim) as usize];
                if layer == 2 {
                    assert!(
                        texels.iter().all(|&b| b == 0),
                        "unmapped layer {layer} level {level} decodes transparent black"
                    );
                    continue;
                }
                let extent = ((w >> level).max(1), (h >> level).max(1));
                let want =
                    decode_astc_level(format, &data[(layer * levels + level) as usize], extent);
                assert!(
                    texels == &want,
                    "{recompression:?}: layer {layer} level {level} holds guest level {level}: \
                     first difference {:?}",
                    first_difference(texels, &want, extent.0)
                );
            }
        }
        f.assert_no_validation_errors();
    }
}

/// Root mean square error over every channel of two RGBA8 images.
fn rmse(a: &[u8], b: &[u8]) -> f64 {
    let sum: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
        .sum();
    (sum / a.len() as f64).sqrt()
}

fn decode_bc(bc: Compression, blocks: &[u8], extent: (u32, u32)) -> Vec<u8> {
    let (w, h) = extent;
    let mut out = vec![0; (w * h * 4) as usize];
    if bc == Compression::Bc1 {
        bcn::decode_bc1(blocks, w, h, 1, &mut out).unwrap();
    } else {
        bcn::decode_bc3(blocks, w, h, 1, &mut out).unwrap();
    }
    out
}

/// The GPU encoder's error, measured by decoding its blocks back and comparing with the CPU's
/// ASTC decode, stays within 10% (plus one level) of the CPU encoder's error on every
/// subresource, and within the 565 rounding (4 levels, alpha exact in BC3) on constant blocks.
#[test]
fn gpu_bc_recompression_stays_close_to_the_cpu_decode() {
    let gob = GobLayout::Switch2;
    let (w, h, layers, levels) = (70, 45, 2, 3);
    for (recompression, bc) in [
        (AstcRecompression::Bc1, Compression::Bc1),
        (AstcRecompression::Bc3, Compression::Bc3),
    ] {
        for scratch in [super::astc::SCRATCH_BYTES, 4 << 10] {
            let Some(mut f) = fixture(gob, recompression) else {
                return;
            };
            f.cache.set_astc_decode(AstcDecode::Gpu);
            f.cache.astc_scratch = scratch;
            let constant: &[BlockKind] = &[BlockKind::VoidExtent, BlockKind::TransparentBlack];
            for (raw, format, kinds) in [
                (0x40, PixelFormat::Astc4x4Unorm, constant),
                (0x42, PixelFormat::Astc6x6Unorm, &BLOCK_KINDS[..]),
                (0x46, PixelFormat::Astc12x12Unorm, &BLOCK_KINDS[..]),
            ] {
                let host = host_format(&f.device, format, recompression);
                if host.conversion != Conversion::DecodeToBc(bc) {
                    eprintln!("skipping: ASTC is not recompressed on this device ({host:?})");
                    return;
                }
                let s = surface(format, w, h, levels, layers, 3, gob);
                let mut data = astc_level_data(&s, u64::from(raw), kinds);
                if bc == Compression::Bc1 && kinds.len() == 2 {
                    for block in data
                        .iter_mut()
                        .flat_map(|level| level.as_chunks_mut::<16>().0)
                    {
                        block[14..].copy_from_slice(&[0xFF, 0xFF]);
                    }
                }
                write_surface(&f.memory, &s, &data);
                let (id, _) = f.texture(&array_tic(
                    raw,
                    false,
                    false,
                    GPU_VA,
                    (w, h, layers, levels),
                ));
                let read = f.read_image(id);
                assert_eq!(read.len(), (layers * levels) as usize);
                for (i, (got, blocks)) in read.iter().zip(&data).enumerate() {
                    let (layer, level) = (i as u32 / levels, i as u32 % levels);
                    let extent = ((w >> level).max(1), (h >> level).max(1));
                    let want = decode_astc_level(format, blocks, extent);
                    let gpu = decode_bc(bc, got, extent);
                    let at =
                        format!("{bc:?} {format:?} layer {layer} level {level} scratch {scratch}");
                    if kinds.len() == 2 {
                        for (g, c) in gpu.as_chunks::<4>().0.iter().zip(want.as_chunks::<4>().0) {
                            let rgb = (0..3).map(|i| g[i].abs_diff(c[i])).max().unwrap();
                            let alpha_exact = bc == Compression::Bc1 || g[3] == c[3];
                            assert!(rgb <= 4 && alpha_exact, "{at}: {g:?} for {c:?}");
                        }
                        continue;
                    }
                    let mut cpu = vec![0; got.len()];
                    if bc == Compression::Bc1 {
                        bcn::encode_bc1(&want, extent.0, extent.1, 1, &mut cpu).unwrap();
                    } else {
                        bcn::encode_bc3(&want, extent.0, extent.1, 1, &mut cpu).unwrap();
                    }
                    let (gpu_error, cpu_error) =
                        (rmse(&gpu, &want), rmse(&decode_bc(bc, &cpu, extent), &want));
                    assert!(
                        gpu_error <= cpu_error * 1.1 + 1.0,
                        "{at}: RMSE {gpu_error:.2} against the CPU encoder's {cpu_error:.2}"
                    );
                }
                f.cache.unmap(&mut f.scheduler, DADDR, s.size_bytes());
                assert!(f.cache.is_empty());
            }
            f.assert_no_validation_errors();
        }
    }
}

#[test]
fn storage_lookups_keep_every_level_of_arrays_that_samples_trim() {
    let gob = GobLayout::Switch2;
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    f.cache.trim_above = 1 << 20;
    f.cache.trim_to = 512 << 10;
    let header = tic(0x08, GPU_VA, 512, 512, 4, 4, 3);
    let (sampled, _) = f.texture(&header);
    assert_eq!(f.cache.state(sampled).unwrap().extent.width, 128);
    let desc = header.decode().unwrap();
    let (stored, _) = {
        let (mut tx, cache, gmmu) = f.tx();
        cache
            .texture(&mut tx, gmmu, &desc, Usage::Storage)
            .unwrap()
            .unwrap()
    };
    assert_ne!(
        stored, sampled,
        "a storage image never stands in for a trimmed one"
    );
    assert_eq!(f.cache.state(stored).unwrap().extent.width, 512);
    assert_eq!(f.cache.get(stored).unwrap().image.desc().mip_levels, 4);
    f.assert_no_validation_errors();
}

#[test]
fn a_layer_lookup_leaves_a_remapped_array_whole() {
    let gob = GobLayout::Switch2;
    let s = surface(PixelFormat::A8B8G8R8Unorm, 64, 64, 1, 2, 2, gob);
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    write_surface(&f.memory, &s, &level_data(&s, 3));
    let va = GPU_VA + DRAM;
    let bytes = s.size_bytes().next_multiple_of(0x1000);
    let stride = s.subresource_offset(0, 1);
    f.gmmu.map_sparse(va, bytes, false).unwrap();
    f.gmmu.map(va, DADDR, stride, 0, false).unwrap();
    let array = tic(0x08, va, 64, 64, 2, 1, 2);
    let (id, _) = f.texture(&array);
    f.gmmu.map(va, DADDR, bytes, 0, false).unwrap();
    assert_eq!(f.texture(&array).0, id, "the backed array keeps its image");
    let want = f.read_image(id);
    assert!(want[1].iter().any(|&b| b != 0));
    f.texture(&tic(0x08, va, 64, 64, 1, 1, 2));
    assert_eq!(f.texture(&array).0, id);
    assert!(f.read_image(id) == want, "layer 1 keeps its texels");
    f.assert_no_validation_errors();
}

#[test]
fn a_remapped_array_takes_newer_gpu_writes_over_its_new_pages() {
    let gob = GobLayout::Switch2;
    let s = surface(PixelFormat::A8B8G8R8Unorm, 64, 64, 1, 2, 2, gob);
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    let va = GPU_VA + DRAM;
    let bytes = s.size_bytes().next_multiple_of(0x1000);
    let stride = s.subresource_offset(0, 1);
    f.gmmu.map_sparse(va, bytes, false).unwrap();
    f.gmmu.map(va, DADDR, stride, 0, false).unwrap();
    let array = tic(0x08, va, 64, 64, 2, 1, 2);
    let (id, _) = f.texture(&array);
    let info = rt_info(64, 64, 2, gob);
    assert_eq!(info.size_bytes(), bytes - stride);
    clear_rt_at(
        &mut f,
        GPU_VA + 0x10_0000,
        &info,
        [0.0, 1.0, 0.0, 1.0],
        Rect::ALL,
    );
    f.gmmu
        .map(va + stride, DADDR + 0x10_0000, bytes - stride, 0, false)
        .unwrap();
    assert_eq!(f.texture(&array).0, id);
    let got = f.read_image(id);
    assert!(
        got[1].chunks(4).all(|t| t == [0, 255, 0, 255]),
        "layer 1 holds the render target's clear"
    );
    f.assert_no_validation_errors();
}

#[test]
fn segments_alias_when_any_two_share_device_memory() {
    let s = |daddr, size| Segment {
        offset: 0,
        daddr,
        size,
    };
    assert!(!self_aliased(&[
        s(0x2000, 0x1000),
        s(0x1000, 0x1000),
        s(0x4000, 0x10)
    ]));
    assert!(self_aliased(&[
        s(0x1000, 0x3000),
        s(0x8000, 0x1000),
        s(0x3fff, 1)
    ]));
    assert!(self_aliased(&[s(0x1000, 0x1000), s(0x1000, 0x1000)]));
}

#[test]
fn sampled_mip_chains_that_run_out_of_memory_keep_the_levels_that_fit() {
    let gob = GobLayout::Tegra;
    let header = tic(0x08, GPU_VA, 256, 256, 1, 9, 2);
    let info = ImageInfo::from_texture(&header.decode().unwrap(), gob).unwrap();
    let pattern: Vec<u8> = (0..info.size_bytes())
        .map(|i| (i * 7 + i / 251) as u8)
        .collect();
    let Some(mut reference) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    reference.memory.write_bytes(DADDR, &pattern).unwrap();
    let (full, _) = reference.texture(&header);
    let want = reference.read_image(full);
    let full_size = reference.cache.get(full).unwrap().image.memory_size();
    drop(reference);
    let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
        return;
    };
    f.memory.write_bytes(DADDR, &pattern).unwrap();
    f.cache.trim_to = 128 << 10;
    f.cache.allocation_cap = full_size - 1;
    let (id, _) = f.texture(&header);
    let image = f.cache.get(id).unwrap();
    assert_eq!(image.trim, 1, "keeps the levels from 1 on");
    assert!(image.image.memory_size() < full_size);
    assert_eq!(f.read_image(id), want[1..]);
    f.assert_no_validation_errors();
}

#[test]
fn padded_level_targets_match_only_depth_free_levels_below_the_base() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let mut chain = ImageInfo {
            levels: 2,
            ..rt_info(16, 16, 0, gob)
        };
        chain.layer_stride = chain.surface().layer_stride();
        let level_1 = chain.surface().level_offset(1);
        let padded = rt_info(16, 8, 0, gob);
        assert_eq!(chain.find_subresource(&padded, level_1), None, "{gob:?}");
        assert_eq!(
            chain.find_padded_level(&padded, level_1),
            Some((1, 0)),
            "{gob:?}"
        );
        assert_eq!(
            rt_info(8, 8, 0, gob).find_padded_level(&padded, 0),
            None,
            "{gob:?}: a wider target over a base level keeps its own texels"
        );
        let GuestLayout::BlockLinear {
            block_height_log2,
            tile_width_spacing,
            ..
        } = padded.layout
        else {
            unreachable!()
        };
        let deep = ImageInfo {
            layout: GuestLayout::BlockLinear {
                block_height_log2,
                block_depth_log2: 1,
                tile_width_spacing,
                gob,
            },
            ..padded
        };
        assert_eq!(
            chain.find_padded_level(&deep, level_1),
            None,
            "{gob:?}: the target's block depth moves its GOBs"
        );
    }
}

/// Mario Kart World's light-probe cube arrays at their full 7 levels: levels 4 to 6 (8x8,
/// 4x4, 2x2) are rendered as targets one GOB wide (16x8, 16x4, 16x2 at block height 1),
/// which cover exactly the GOB each of those levels is stored in.
#[test]
fn mkw_cube_array_levels_narrower_than_a_gob_render_into_the_array() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        for sample_first in [true, false] {
            let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
                return;
            };
            let levels = 7;
            let mut header = tic(0x21, GPU_VA, 128, 128, 2, levels, 4);
            header.0[0] = (header.0[0] & !(0xfff << 7)) | 7 << 7 | 7 << 10 | 7 << 13 | 7 << 16;
            header.0[4] = (header.0[4] & !(0xf << 23)) | 8 << 23;
            let array = ImageInfo::from_texture(&header.decode().unwrap(), gob).unwrap();
            assert_eq!((array.layers, array.levels), (12, levels));
            let pitch = (array.layer_stride / 4) as u32;
            let target = |level: u32| {
                let size = 128 >> level;
                match level {
                    0..4 => mkw_target(
                        0xe0,
                        size,
                        size,
                        array.surface().level_tiling(level).block_height_log2,
                        pitch,
                        gob,
                    ),
                    _ => mkw_target(0xe0, 16, size, 0, pitch, gob),
                }
            };
            let early = sample_first.then(|| f.texture(&header).0);
            let mut founds = Vec::new();
            for level in 0..levels {
                let info = target(level);
                for layer in 0..12 {
                    let value = (layer * levels + level + 1) as f32 / 128.0;
                    let found = clear_rt_at(
                        &mut f,
                        GPU_VA + array.subresource_offset(level, layer),
                        &info,
                        [value, 0.25, 1.0 - value, 1.0],
                        Rect::ALL,
                    );
                    founds.push((level, layer, found));
                }
            }
            let (id, _) = f.texture(&header);
            let case = format!("{gob:?} sample_first {sample_first}");
            assert_eq!(early.unwrap_or(id), id, "{case}");
            let strays: Vec<_> = founds
                .iter()
                .filter(|(_, _, t)| t.id != id && f.cache.state(t.id).is_some())
                .collect();
            assert!(
                strays.is_empty(),
                "{case}: targets left beside the array: {strays:?}"
            );
            let texels = f.read_image(id);
            let mut seen = Vec::new();
            for layer in 0..12 {
                for level in 0..levels {
                    let got = &texels[(layer * levels + level) as usize];
                    let first = &got[..4];
                    assert!(
                        first.iter().any(|&b| b != 0)
                            && got.chunks(4).all(|t| t == first)
                            && !seen.contains(&first.to_vec()),
                        "{case}: layer {layer} level {level} holds {first:?}"
                    );
                    seen.push(first.to_vec());
                }
            }
            for level in 4..levels {
                let t = clear_rt_at(
                    &mut f,
                    GPU_VA + array.subresource_offset(level, 7),
                    &target(level),
                    [0.5, 0.5, 0.5, 1.0],
                    Rect::ALL,
                );
                assert_eq!(
                    (t.id, t.level, t.layer),
                    (id, level, 7),
                    "{case} second frame"
                );
            }
            f.assert_no_validation_errors();
        }
    }
}

#[test]
fn cube_levels_take_newer_overlaps_in_one_gpu_sync() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let levels = 8;
        let mut header = tic(0x08, GPU_VA, 128, 128, 1, levels, 4);
        header.0[4] = (header.0[4] & !(0xf << 23)) | 8 << 23;
        let cube = ImageInfo::from_texture(&header.decode().unwrap(), gob).unwrap();
        assert_eq!((cube.layers, cube.levels), (6, levels));
        let own =
            |level: u32, layer: u32| [(layer * levels + level + 1) as u8 * 4, 0x40, 0x80, 0xff];
        for layer in 0..6 {
            for level in 0..levels {
                let bytes: Vec<u8> = own(level, layer)
                    .into_iter()
                    .cycle()
                    .take(cube.level_size(level) as usize)
                    .collect();
                f.memory
                    .write_bytes(DADDR + cube.subresource_offset(level, layer), &bytes)
                    .unwrap();
            }
        }
        let (id, _) = f.texture(&header);
        let mut expected: Vec<[u8; 4]> = (0..6)
            .flat_map(|layer| (0..levels).map(move |level| own(level, layer)))
            .collect();
        let mut paint = |f: &mut Fixture, overlaps: &[(u32, u32, u32, [u8; 4])]| {
            for &(first, last, layer, color) in overlaps {
                let start = cube.subresource_offset(first, layer);
                let end = cube.subresource_offset(last, layer) + cube.level_size(last);
                let target = clear_rt_at(
                    f,
                    GPU_VA + start,
                    &rt_info(16, ((end - start) / 64) as u32, 0, gob),
                    color.map(|c| f32::from(c) / 255.0),
                    Rect::ALL,
                );
                assert_ne!(target.id, id, "{gob:?}: an incompatible overlap");
                for level in first..=last {
                    expected[(layer * levels + level) as usize] = color;
                }
            }
            let syncs = f.cache.traffic().gpu_syncs;
            let (again, _) = f.texture(&header);
            assert_eq!(again, id, "{gob:?}");
            assert_eq!(f.cache.traffic().gpu_syncs, syncs + 1, "{gob:?}");
        };
        paint(
            &mut f,
            &[
                (0, 0, 0, [255, 0, 0, 255]),
                (1, 1, 2, [0, 255, 0, 255]),
                (4, 7, 3, [0, 0, 255, 255]),
                (0, 0, 5, [255, 255, 0, 255]),
            ],
        );
        paint(
            &mut f,
            &[(2, 3, 1, [0, 255, 255, 255]), (5, 6, 4, [255, 0, 255, 255])],
        );
        let texels = f.read_image(id);
        for layer in 0..6 {
            for level in 0..levels {
                let i = (layer * levels + level) as usize;
                assert!(
                    texels[i].chunks(4).all(|t| t == expected[i]),
                    "{gob:?}: layer {layer} level {level} holds {:?}, not {:?}",
                    &texels[i][..4],
                    expected[i]
                );
            }
        }
        f.assert_no_validation_errors();
    }
}

/// Donkey Kong Bananza's B10G11R11 128x128 cube: 6 faces 0x16000 apart, 8 levels at block
/// height 16. Each level of each face is rendered as its own single-level target, the mip tail
/// levels 4 to 7 as targets one GOB wide (16x8, 16x4, 16x2 and 16x1 at block height 1).
fn dkb_cube(gob: GobLayout) -> (TicEntry, ImageInfo) {
    let mut header = tic(0x21, GPU_VA, 128, 128, 1, 8, 4);
    header.0[0] = (header.0[0] & !(0xfff << 7)) | 7 << 7 | 7 << 10 | 7 << 13 | 7 << 16;
    header.0[4] = (header.0[4] & !(0xf << 23)) | 3 << 23;
    let cube = ImageInfo::from_texture(&header.decode().unwrap(), gob).unwrap();
    assert_eq!((cube.layers, cube.levels), (6, 8));
    assert_eq!(cube.layer_stride, 0x16000, "{gob:?}");
    assert_eq!(cube.subresource_offset(6, 4), 0x6d800, "{gob:?}");
    (header, cube)
}

fn dkb_target(cube: &ImageInfo, level: u32, gob: GobLayout) -> ImageInfo {
    let size = 128 >> level;
    let pitch = (cube.layer_stride / 4) as u32;
    match level {
        0..4 => {
            let bh = cube.surface().level_tiling(level).block_height_log2;
            mkw_target(0xe0, size, size, bh, pitch, gob)
        }
        _ => mkw_target(0xe0, 16, size, 0, pitch, gob),
    }
}

/// A 2D header over one tail level of one face, the shape its target has.
fn dkb_tail_tic(cube: &ImageInfo, level: u32, layer: u32) -> TicEntry {
    let address = GPU_VA + cube.subresource_offset(level, layer);
    let mut header = tic(0x21, address, 16, 128 >> level, 1, 1, 0);
    header.0[0] = (header.0[0] & !(0xfff << 7)) | 7 << 7 | 7 << 10 | 7 << 13 | 7 << 16;
    header
}

fn dkb_color(layer: u32, level: u32, frame: u32) -> [f32; 4] {
    [
        (layer + 1) as f32 / 8.0,
        (level + 1) as f32 / 16.0,
        (frame + 1) as f32 / 16.0,
        1.0,
    ]
}

/// The B10G11R11 texel of a colour whose channels the format holds exactly.
fn b10g11r11(color: [f32; 4]) -> [u8; 4] {
    let small = |v: f32, mantissa: u32| {
        let bits = v.to_bits();
        let exponent = ((bits >> 23) & 0xff) + 15 - 127;
        exponent << mantissa | (bits & 0x7f_ffff) >> (23 - mantissa)
    };
    (small(color[0], 6) | small(color[1], 6) << 11 | small(color[2], 5) << 22).to_le_bytes()
}

/// A draw over all of a target: rendered into without a discard, then cleared whole.
fn draw_rt_at(f: &mut Fixture, gpu_va: GpuVAddr, info: &ImageInfo, color: [f32; 4]) -> Found {
    let (mut tx, cache, gmmu) = f.tx();
    let target = cache
        .render_target(&mut tx, gmmu, gpu_va, info, false)
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
            Rect::ALL,
            0,
            1,
        )
        .unwrap();
    target
}

/// The cube's tail levels were sampled as 2D textures before they were first drawn, so older
/// single-level images lie over them. Drawing the tail targets resolves into the cube and
/// retires those images, instead of drawing into them and syncing them with the cube both ways
/// every frame; the tail addresses still sample as 2D textures afterwards.
#[test]
fn cube_tail_targets_over_older_single_level_images_draw_into_the_cube() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let (header, cube) = dkb_cube(gob);
        let (id, _) = f.texture(&header);
        for level in 4..8 {
            for layer in 0..6 {
                let (tail, _) = f.texture(&dkb_tail_tic(&cube, level, layer));
                assert_ne!(tail, id, "{gob:?}");
            }
        }
        let frames = 4;
        let mut syncs = Vec::new();
        let mut strays = Vec::new();
        for frame in 0..frames {
            let before = f.cache.traffic().gpu_syncs;
            for level in 0..8 {
                for layer in 0..6 {
                    let found = draw_rt_at(
                        &mut f,
                        GPU_VA + cube.subresource_offset(level, layer),
                        &dkb_target(&cube, level, gob),
                        dkb_color(layer, level, frame),
                    );
                    if (found.id, found.level, found.layer) != (id, level, layer) {
                        strays.push((frame, level, layer));
                    }
                }
            }
            assert_eq!(f.texture(&header).0, id, "{gob:?}");
            syncs.push(f.cache.traffic().gpu_syncs - before);
            f.tick(1 << 40);
        }
        assert_eq!(syncs, [0; 4], "{gob:?}: GPU syncs per frame");
        assert!(
            strays.is_empty(),
            "{gob:?}: targets beside the cube {strays:?}"
        );
        let last = frames - 1;
        let texels = f.read_image(id);
        for layer in 0..6 {
            for level in 0..8 {
                let want = b10g11r11(dkb_color(layer, level, last));
                let got = &texels[(layer * 8 + level) as usize];
                assert!(
                    got.chunks(4).all(|t| t == want),
                    "{gob:?}: face {layer} level {level} holds {:?}, not {want:?}",
                    &got[..4]
                );
            }
        }
        for level in 4..8 {
            let (tail, _) = f.texture(&dkb_tail_tic(&cube, level, 4));
            assert_ne!(tail, id, "{gob:?}");
            let texels = &f.read_image(tail)[0];
            let want = b10g11r11(dkb_color(4, level, last));
            let side = (128 >> level) as usize;
            for (y, row) in texels.chunks(16 * 4).take(side).enumerate() {
                assert!(
                    row[..side * 4].chunks(4).all(|t| t == want),
                    "{gob:?}: level {level} sampled as 2D holds {:?} in row {y}, not {want:?}",
                    &row[..4]
                );
            }
        }
        assert_eq!(f.texture(&header).0, id, "{gob:?}");
        f.assert_no_validation_errors();
    }
}

/// The same with the older images drawn by copies, which never resolve into a padded level,
/// and the tail targets looked up again after the lookups were forgotten: the cube took the
/// copies' texels when it was drawn next, so retiring them loses none.
#[test]
fn cube_tail_targets_retire_drawn_single_level_images_the_cube_took_texels_from() {
    for gob in [GobLayout::Tegra, GobLayout::Switch2] {
        let Some(mut f) = fixture(gob, AstcRecompression::Uncompressed) else {
            return;
        };
        let (header, cube) = dkb_cube(gob);
        let (id, _) = f.texture(&header);
        for level in 0..8 {
            for layer in 0..6 {
                draw_rt_at(
                    &mut f,
                    GPU_VA + cube.subresource_offset(level, layer),
                    &dkb_target(&cube, level, gob),
                    dkb_color(layer, level, 0),
                );
            }
        }
        let mut copied = Vec::new();
        for level in 4..8 {
            let address = GPU_VA + cube.subresource_offset(level, 3);
            let desc = dkb_tail_tic(&cube, level, 3).decode().unwrap();
            let info = ImageInfo::from_texture(&desc, gob).unwrap();
            let (mut tx, cache, gmmu) = f.tx();
            let tail = cache
                .prepare_target(&mut tx, gmmu, address, &info, true, Usage::Transfer)
                .unwrap()
                .unwrap();
            assert_ne!(tail.id, id, "{gob:?}");
            cache
                .clear(
                    &mut tx,
                    tail,
                    ClearValue::Color {
                        value: vk::ClearColorValue {
                            float32: dkb_color(3, level, 1),
                        },
                        mask: vk::ColorComponentFlags::RGBA,
                    },
                    Rect::ALL,
                    0,
                    1,
                )
                .unwrap();
            copied.push(tail.id);
        }
        draw_rt_at(
            &mut f,
            GPU_VA,
            &dkb_target(&cube, 0, gob),
            dkb_color(0, 0, 0),
        );
        f.cache.invalidate_lookups();
        let syncs = f.cache.traffic().gpu_syncs;
        for level in 4..8 {
            let (mut tx, cache, gmmu) = f.tx();
            let found = cache
                .render_target(
                    &mut tx,
                    gmmu,
                    GPU_VA + cube.subresource_offset(level, 3),
                    &dkb_target(&cube, level, gob),
                    false,
                )
                .unwrap()
                .unwrap();
            assert_eq!(
                (found.id, found.level, found.layer),
                (id, level, 3),
                "{gob:?}"
            );
        }
        assert_eq!(f.cache.traffic().gpu_syncs, syncs, "{gob:?}");
        assert!(
            copied.iter().all(|&t| f.cache.state(t).is_none()),
            "{gob:?}"
        );
        let texels = f.read_image(id);
        for layer in 0..6 {
            for level in 0..8 {
                let frame = u32::from(layer == 3 && level >= 4);
                let want = b10g11r11(dkb_color(layer, level, frame));
                let got = &texels[(layer * 8 + level) as usize];
                assert!(
                    got.chunks(4).all(|t| t == want),
                    "{gob:?}: face {layer} level {level} holds {:?}, not {want:?}",
                    &got[..4]
                );
            }
        }
        f.assert_no_validation_errors();
    }
}
