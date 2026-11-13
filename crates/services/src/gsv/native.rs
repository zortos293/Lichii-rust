//! Native memory and images (gsv 111-114, 423-435, 911-991, 1211, 1291 and 1510): the graphics
//! server's registry of the memory and images the Switch 2 SDKs make, which vi2 presents from.

use std::collections::BTreeMap;

use lichii_common::{ResultCode, VAddr};
use lichii_gpu::nx2_memory::GpuMemoryHandle;
use lichii_kernel::ipc::HleRequestContext;
use lichii_memory::AddressSpace;
use lichii_texture::PixelFormat;
use lichii_texture::block_linear::{GobLayout, SurfaceLayout, Tiling, swizzle};
use parking_lot::Mutex;

use super::{UNKNOWN, Words};

/// An InterModuleUserImagePreference, 16 words.
pub(super) type Preference = [u32; 16];

/// Preference word 6: the pixel format.
pub(super) const FORMAT_RGB10A2: u32 = 4;
pub(super) const FORMAT_RGBA16_FLOAT: u32 = 5;

/// Where a native image's texels lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ImageLayout {
    pub width: u32,
    pub height: u32,
    /// Bytes per row of GOBs, a multiple of 64.
    pub pitch: u64,
    /// log2 of the rows in one block.
    pub block_rows_log2: u32,
    pub size: u64,
    pub alignment: u64,
}

impl ImageLayout {
    /// The block-linear layout a preference describes: (width - 1) | (height - 1) << 16 at word 2,
    /// log2 block rows at word 4, the pitch in GOBs at word 5, the format at word 6. A block
    /// height or pitch that cannot hold the image falls back to 128-row blocks and the tightest
    /// pitch.
    pub fn of(preference: &Preference) -> Self {
        let width = (preference[2] & 0xffff) + 1;
        let height = (preference[2] >> 16) + 1;
        let row = u64::from(width) * bytes_per_pixel(preference[6]);
        let pitch_gobs = u64::from(preference[5]);
        let (pitch, block_rows_log2) =
            if preference[4] <= 8 && (row.div_ceil(64)..=4096).contains(&pitch_gobs) {
                (pitch_gobs * 64, preference[4])
            } else {
                (row.next_multiple_of(64), 7)
            };
        let block_rows = 1u64 << block_rows_log2;
        Self {
            width,
            height,
            pitch,
            block_rows_log2,
            size: pitch * u64::from(height).next_multiple_of(block_rows),
            alignment: (64 * block_rows).max(0x1000),
        }
    }

    /// 1510's layout: a single RGBA8 colour image whose pitch the preference gives exactly only
    /// needs page alignment.
    pub fn requirements(preference: &Preference, count: u32) -> Self {
        let mut layout = Self::of(preference);
        let plain_color = count == 1
            && preference[0] == 0
            && preference[1] == 2
            && preference[3] == 0
            && (3..=8).contains(&preference[4])
            && preference[6] == 1
            && preference[7] == 1
            && layout.pitch == u64::from(preference[5]) * 64
            && preference[8..].iter().all(|&word| word == 0);
        if plain_color {
            layout.alignment = 0x1000;
        }
        layout
    }
}

/// Alignment of a decoder image pool (gsv 991): 128 rows of 64 bytes.
const DECODER_ALIGNMENT: u64 = 64 * 128;
/// Most images a decoder pool (gsv 911) may hold. Movie players use a few frames; each image
/// gets its own handle, so the count is bounded.
const MAX_DECODER_IMAGES: u32 = 64;

/// Pixel layout of a video decoder output image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DecoderFormat {
    /// Luma rows, then interleaved CbCr rows from `chroma_offset`.
    Nv12 {
        chroma_offset: u64,
    },
    Bgra,
    Rgba,
}

/// Layout of a video decoder output image in an `nn::codec` image pool (gsv 911, 961, 991):
/// pitch-linear rows padded to 128-row blocks. NV12 keeps its chroma plane after the luma plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecoderLayout {
    pub width: u32,
    pub height: u32,
    pub format: DecoderFormat,
    pub pitch: u64,
    pub size: u64,
}

impl DecoderLayout {
    pub fn of(preference: &Preference) -> Self {
        let width = (preference[2] & 0xffff) + 1;
        let height = (preference[2] >> 16) + 1;
        let nv12 = matches!(preference[1], 1 | 4);
        let pitch = (u64::from(width) * if nv12 { 1 } else { 4 }).next_multiple_of(64);
        let luma = pitch * u64::from(height).next_multiple_of(128);
        let (format, chroma) = match preference[1] {
            1 | 4 => (
                DecoderFormat::Nv12 {
                    chroma_offset: luma,
                },
                pitch * u64::from(height.div_ceil(2)).next_multiple_of(128),
            ),
            2 | 5 => (DecoderFormat::Bgra, 0),
            _ => (DecoderFormat::Rgba, 0),
        };
        Self {
            width,
            height,
            format,
            pitch,
            size: luma + chroma,
        }
    }

    /// The layout an RGBA or BGRA image takes once the title reads it on the GPU (426/428): RGBA
    /// texels in Switch 2 block-linear with 16-GOB blocks, as the reference writes them, shrunk to
    /// fit a short image the way the hardware and the texture cache read it.
    pub fn tiling(&self) -> Tiling {
        SurfaceLayout {
            format: PixelFormat::A8B8G8R8Unorm,
            width: (self.pitch / 4) as u32,
            height: self.height,
            depth: 1,
            levels: 1,
            layers: 1,
            block_height_log2: 4,
            block_depth_log2: 0,
            tile_width_spacing: 0,
            layout: GobLayout::Switch2,
        }
        .level_tiling(0)
    }

    /// The picture kind gsv 961 reports.
    pub fn kind(&self) -> u32 {
        match self.format {
            DecoderFormat::Nv12 { .. } => 0xa,
            DecoderFormat::Bgra => 0x9,
            DecoderFormat::Rgba => 0x8,
        }
    }
}

/// Where a decoder output image lies in the title's memory.
#[derive(Clone, Copy, Debug)]
pub(super) struct DecoderImage {
    pub address: VAddr,
    pub layout: DecoderLayout,
    /// Frames go in [`DecoderLayout::tiling`]: the title reads this RGBA or BGRA image on the GPU.
    pub block_linear: bool,
}

/// Rewrites a decoder image holding a pitch-linear RGBA or BGRA frame in [`DecoderLayout::tiling`],
/// for a frame written before the title's first 426/428 for the image.
pub(super) fn relayout_decoder_image(
    memory: &AddressSpace,
    address: VAddr,
    layout: &DecoderLayout,
) {
    if matches!(layout.format, DecoderFormat::Nv12 { .. }) {
        return;
    }
    let tiling = layout.tiling();
    let mut linear = vec![0u8; tiling.linear_size_bytes()];
    if memory.read_bytes(address, &mut linear).is_err() {
        return;
    }
    if layout.format == DecoderFormat::Bgra {
        for texel in linear.as_chunks_mut::<4>().0 {
            texel.swap(0, 2);
        }
    }
    let mut tiled = vec![0u8; tiling.size_bytes()];
    swizzle(&tiling, &linear, &mut tiled);
    if memory.write_bytes(address, &tiled).is_err() {
        tracing::warn!("nx2:gsv: cannot relayout the decoder image at {address:#x}");
    }
}

pub(super) fn bytes_per_pixel(format: u32) -> u64 {
    if format == FORMAT_RGBA16_FLOAT { 8 } else { 4 }
}

#[derive(Clone, Copy, Debug, Default)]
struct Memory {
    size: u64,
    /// The CPU (== GPU) address 114 or 423 bound it to.
    address: VAddr,
    /// 423: the GPU memory it is part of, and where in it.
    parent: Option<(GpuMemoryHandle, u64)>,
}

#[derive(Clone, Copy, Debug)]
struct Image {
    memory: u32,
    preference: Preference,
    size: u64,
    /// Images in the pool: handle + i is image i.
    count: u32,
    /// Where image 0 starts in the memory.
    offset: u64,
    /// A video decoder output image pool (gsv 911), laid out by [`DecoderLayout`].
    decoder: bool,
    /// A decoder image the title read on the GPU (426/428).
    gpu_read: bool,
    /// A decoder image holding a frame written pitch-linear, before its first 426/428.
    pitch_frame: bool,
}

/// A native image vi2 can show.
#[derive(Clone, Copy, Debug)]
pub(super) struct Presentable {
    pub layout: ImageLayout,
    pub format: u32,
    pub address: VAddr,
    pub parent: Option<(GpuMemoryHandle, u64)>,
}

/// The native memory and images of one graphics system session, shared with its vi2 sessions.
#[derive(Default)]
pub(super) struct NativeObjects {
    memories: BTreeMap<u32, Memory>,
    images: BTreeMap<u32, Image>,
    last_memory: u32,
    last_image: u32,
}

impl NativeObjects {
    /// The pool image `handle` names, and its offset in the pool's memory.
    fn image(&self, handle: u32) -> Option<(&Image, u64)> {
        let (&first, image) = self.images.range(..=handle).next_back()?;
        let index = handle - first;
        (index < image.count).then(|| (image, image.offset + u64::from(index) * image.size))
    }

    /// The decoder output image `handle` names, when its memory is bound.
    pub fn decoder_image(&self, handle: u32) -> Option<DecoderImage> {
        let (image, offset) = self.image(handle)?;
        let memory = self.memories.get(&image.memory)?;
        let layout = DecoderLayout::of(&image.preference);
        (image.decoder && memory.address != 0).then(|| DecoderImage {
            address: memory.address + offset,
            layout,
            block_linear: image.gpu_read && !matches!(layout.format, DecoderFormat::Nv12 { .. }),
        })
    }

    /// The decoder wrote a pitch-linear frame into decoder image `handle`. True when a 426/428
    /// came in meanwhile, so the frame needs [`relayout_decoder_image`] now.
    pub fn wrote_pitch_frame(&mut self, handle: u32) -> bool {
        let Some(image) = self.images.get_mut(&handle) else {
            return false;
        };
        image.pitch_frame = !image.gpu_read;
        image.gpu_read
    }

    pub fn presentable(&self, handle: u32) -> Option<Presentable> {
        let (image, offset) = self.image(handle)?;
        let memory = self.memories.get(&image.memory)?;
        (memory.address != 0).then(|| Presentable {
            layout: ImageLayout::of(&image.preference),
            format: image.preference[6],
            address: memory.address + offset,
            parent: memory.parent.map(|(handle, at)| (handle, at + offset)),
        })
    }
}

#[cfg(test)]
impl NativeObjects {
    /// Objects holding one image of `preference` in memory bound at `address`.
    pub fn with_image(
        address: VAddr,
        parent: Option<(GpuMemoryHandle, u64)>,
        preference: Preference,
    ) -> (Self, u32) {
        let mut objects = Self::default();
        objects.memories.insert(
            1,
            Memory {
                size: ImageLayout::of(&preference).size,
                address,
                parent,
            },
        );
        objects.images.insert(
            1,
            Image {
                memory: 1,
                preference,
                size: ImageLayout::of(&preference).size,
                count: 1,
                offset: 0,
                decoder: false,
                gpu_read: false,
                pitch_frame: false,
            },
        );
        (objects, 1)
    }
}

fn preference(words: &Words) -> Preference {
    std::array::from_fn(|i| words.u32(2 + i))
}

fn push_u64s(ctx: &mut HleRequestContext, values: &[u64]) {
    for &value in values {
        ctx.push(value);
    }
}

/// Serves the native object commands of the graphics system session.
pub(super) fn handle(objects: &Mutex<NativeObjects>, ctx: &mut HleRequestContext) -> ResultCode {
    let command = ctx.command();
    let words = Words::command(ctx);
    match command {
        111 => {
            let size = words.u64(3);
            let mut objects = objects.lock();
            objects.last_memory += 1;
            let id = objects.last_memory;
            objects.memories.insert(
                id,
                Memory {
                    size,
                    ..Memory::default()
                },
            );
            tracing::info!("nx2:gsv 111: native memory {id}, {size:#x} bytes");
            ctx.push(id);
        }
        112 => {
            objects.lock().memories.remove(&words.u32(2));
        }
        114 | 423 => {
            let (id, address, size) = (words.u32(2), words.u64(3), words.u64(5));
            let parent = (command == 423)
                .then(|| GpuMemoryHandle::new(words.u32(7)).map(|handle| (handle, words.u64(8))))
                .flatten();
            let mut objects = objects.lock();
            let memory = objects.memories.entry(id).or_default();
            memory.address = address;
            memory.parent = parent;
            if memory.size == 0 {
                memory.size = size;
            }
            tracing::info!("nx2:gsv {command}: native memory {id} at {address:#x}+{size:#x}");
        }
        429 | 911 | 1211 => {
            let preference = preference(&words);
            let count = words.u32(18).max(1);
            let memory = words.u32(19);
            let decoder = command == 911;
            if decoder && count > MAX_DECODER_IMAGES {
                tracing::warn!(
                    "nx2:gsv 911: {count} decoder images, more than {MAX_DECODER_IMAGES}"
                );
                return UNKNOWN;
            }
            let size = if decoder {
                DecoderLayout::of(&preference).size
            } else {
                ImageLayout::of(&preference).size
            };
            let mut objects = objects.lock();
            let id = objects.last_image + 1;
            objects.last_image += count;
            let image = Image {
                memory,
                preference,
                size,
                count,
                offset: 0,
                decoder,
                gpu_read: false,
                pitch_frame: false,
            };
            if decoder {
                for i in 0..count {
                    let offset = u64::from(i) * size;
                    let one = Image {
                        count: 1,
                        offset,
                        ..image
                    };
                    objects.images.insert(id + i, one);
                }
            } else {
                objects.images.insert(id, image);
            }
            tracing::info!(
                "nx2:gsv {command}: native image {id} x{count} in memory {memory}, {size:#x} bytes, \
                 format {}",
                preference[6]
            );
            ctx.push(id);
        }
        430 | 912 => {
            objects.lock().images.remove(&words.u32(2));
        }
        961 => {
            let handle = words.u32(2);
            let mut descriptor = [0u32; 32];
            match objects.lock().image(handle) {
                Some((image, offset)) if image.decoder => {
                    let layout = DecoderLayout::of(&image.preference);
                    descriptor[0] = layout.kind();
                    descriptor[2] = offset as u32;
                    descriptor[4] = layout.pitch as u32;
                    if let DecoderFormat::Nv12 { chroma_offset } = layout.format {
                        descriptor[6] = (offset + chroma_offset) as u32;
                        descriptor[8] = layout.pitch as u32;
                    }
                }
                _ => tracing::warn!("nx2:gsv 961: no decoder image {handle}"),
            }
            for word in descriptor {
                ctx.push(word);
            }
        }
        991 => {
            let layout = DecoderLayout::of(&preference(&words));
            let size = layout.size * u64::from(words.u32(18).max(1));
            push_u64s(ctx, &[size, DECODER_ALIGNMENT]);
        }
        426 | 428 => {
            let handle = words.u32(2);
            let (preference, offset, address) = {
                let mut objects = objects.lock();
                let found = objects.image(handle).map(|(image, offset)| {
                    let address = objects.memories.get(&image.memory).map_or(0, |m| m.address);
                    (image.preference, offset, address, image.decoder)
                });
                match found {
                    Some((preference, offset, address, decoder)) => {
                        if decoder && let Some(image) = objects.images.get_mut(&handle) {
                            image.gpu_read = true;
                            if std::mem::take(&mut image.pitch_frame) && address != 0 {
                                let layout = DecoderLayout::of(&preference);
                                relayout_decoder_image(ctx.memory(), address + offset, &layout);
                            }
                        }
                        (preference, offset, address)
                    }
                    None => {
                        tracing::warn!("nx2:gsv {command}: no native image {handle}");
                        (Preference::default(), 0, 0)
                    }
                }
            };
            let layout = ImageLayout::of(&preference);
            let gpu_address = if command == 426 { address + offset } else { 0 };
            push_u64s(ctx, &[gpu_address, offset]);
            let mut descriptor = [0u32; 33];
            descriptor[0] = 1;
            descriptor[1] = preference[1];
            descriptor[2] = preference[2];
            descriptor[4] = layout.block_rows_log2;
            descriptor[5] = (layout.pitch / 64) as u32;
            descriptor[8] = 1;
            descriptor[9] = 1;
            descriptor[11] = layout.size as u32;
            for word in descriptor {
                ctx.push(word);
            }
        }
        435 => {
            let size = ImageLayout::of(&preference(&words)).size * u64::from(words.u32(18).max(1));
            ctx.push(size);
        }
        1291 => {
            let layout = ImageLayout::of(&preference(&words));
            let size = layout.size * u64::from(words.u32(18).max(1));
            push_u64s(ctx, &[size, layout.alignment]);
        }
        _ => {
            let layout = ImageLayout::requirements(&preference(&words), words.u32(18));
            push_u64s(ctx, &[layout.size, layout.alignment]);
        }
    }
    ResultCode::SUCCESS
}
