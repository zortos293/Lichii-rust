//! Native memory and images (gsv 111-114, 423-435, 1211, 1291 and 1510): the graphics server's
//! registry of the memory and images the Switch 2 SDKs make, which vi2 presents from.

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
        (index < image.count).then(|| (image, u64::from(index) * image.size))
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
        429 | 1211 => {
            let preference = preference(&words);
            let count = words.u32(18).max(1);
            let memory = words.u32(19);
            let mut objects = objects.lock();
            let id = objects.last_image + 1;
            objects.last_image += count;
            let size = ImageLayout::of(&preference).size;
            objects.images.insert(
                id,
                Image {
                    memory,
                    preference,
                    size,
                    count,
                },
            );
            tracing::info!(
                "nx2:gsv {command}: native image {id} x{count} in memory {memory}, {size:#x} bytes, \
                 format {}",
                preference[6]
            );
            ctx.push(id);
        }
        430 => {
            objects.lock().images.remove(&words.u32(2));
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
