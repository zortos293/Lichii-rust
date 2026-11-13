//! Native memory and images (gsv 111-114, 423-435, 911-991, 1211, 1291 and 1510): the graphics
//! server's registry of the memory and images the Switch 2 SDKs make, which vi2 presents from.

use std::collections::BTreeMap;

use lichii_common::{ResultCode, VAddr};
use lichii_gpu::nx2_memory::GpuMemoryHandle;
use lichii_kernel::ipc::HleRequestContext;
use parking_lot::Mutex;

use super::Words;

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

/// Layout of a video decoder output image in an `nn::codec` image pool (gsv 911, 961, 991):
/// pitch-linear rows padded to 128-row blocks. NV12 keeps its chroma plane after the luma plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct DecoderLayout {
    /// The picture kind gsv 961 reports: 0xa NV12, 0x9 BGRA8, 0x8 RGBA8.
    pub kind: u32,
    pub pitch: u64,
    pub chroma_offset: Option<u64>,
    pub size: u64,
}

impl DecoderLayout {
    pub fn of(preference: &Preference) -> Self {
        let width = u64::from(preference[2] & 0xffff) + 1;
        let height = u64::from(preference[2] >> 16) + 1;
        let (kind, nv12) = match preference[1] {
            1 | 4 => (0xa, true),
            2 | 5 => (0x9, false),
            _ => (0x8, false),
        };
        let pitch = (width * if nv12 { 1 } else { 4 }).next_multiple_of(64);
        let luma = pitch * height.next_multiple_of(128);
        let chroma = if nv12 {
            pitch * height.div_ceil(2).next_multiple_of(128)
        } else {
            0
        };
        Self {
            kind,
            pitch,
            chroma_offset: nv12.then_some(luma),
            size: luma + chroma,
        }
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
                    descriptor[0] = layout.kind;
                    descriptor[2] = offset as u32;
                    descriptor[4] = layout.pitch as u32;
                    if let Some(chroma) = layout.chroma_offset {
                        descriptor[6] = (offset + chroma) as u32;
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
                let objects = objects.lock();
                match objects.image(handle) {
                    Some((image, offset)) => {
                        let address = objects.memories.get(&image.memory).map_or(0, |m| m.address);
                        (image.preference, offset, address)
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
