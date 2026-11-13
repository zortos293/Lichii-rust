//! `pl:u`, the shared font service. Switch 1 titles map one 17 MiB shared memory holding every
//! font (each behind an 8-byte header); Switch 2 titles ask for each font with 2010
//! `GetBuiltInFontData` and read plain copies from the shared memory 2000 returns, or have them
//! copied into their own transfer memory with 2100 (NX2_NOTES row 36).
//!
//! Fonts load on first use. Without fonts every font reads as empty, but the shared memory still
//! exists, so titles that only map it keep going.

mod fonts;

use std::sync::{Arc, OnceLock};

use lichii_common::{HleResult, PAGE_SIZE, ResultCode, align_up};
use lichii_kernel::Pool;
use lichii_kernel::ipc::HleRequestContext;
use lichii_kernel::memory_objects::{SharedMemory, TransferMemory};
use lichii_kernel::object::KObject;
use lichii_memory::MemoryPermission;

use crate::{Service, ServiceContext};
use fonts::FONT_COUNT;

/// Size of the Switch 1 font shared memory; titles map exactly this much.
const SHARED_FONT_MEMORY_SIZE: u64 = 0x110_0000;
/// `GetSharedFontInOrderOfPriority` returns at most this many fonts.
const MAX_PRIORITY_FONTS: usize = 6;
/// `LoadState::Loaded`.
const LOADED: u32 = 1;

pub(crate) fn register(context: &Arc<ServiceContext>) -> HleResult<()> {
    let fonts = Arc::new(SharedFonts {
        context: context.clone(),
        loaded: OnceLock::new(),
        builtin: OnceLock::new(),
    });
    context.register("pl:u", "pl", move |_| PlU {
        fonts: fonts.clone(),
    })
}

/// Where a font's data starts in a shared memory, and its size.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct FontRegion {
    offset: u32,
    size: u32,
}

struct LoadedFonts {
    memory: Arc<SharedMemory>,
    /// Each font type's data in `memory`, after its header; empty when the font is missing.
    regions: [FontRegion; FONT_COUNT],
    /// Where the Switch 2 built-in font memory (and the 2100 transfer memory) has each font.
    builtin: [FontRegion; FONT_COUNT],
    builtin_size: u32,
}

impl LoadedFonts {
    fn load(context: &ServiceContext) -> HleResult<Self> {
        let memory = new_shared_memory(context, SHARED_FONT_MEMORY_SIZE)?;
        let (data, searched) = fonts::find(&context.settings);
        let mut regions = [FontRegion::default(); FONT_COUNT];
        let mut offset = 0u64;
        for (region, font) in regions.iter_mut().zip(&data) {
            let Some(font) = font else { continue };
            let end = offset + 8 + align_up(font.len() as u64, 4);
            if end > SHARED_FONT_MEMORY_SIZE {
                tracing::error!("shared fonts exceed {SHARED_FONT_MEMORY_SIZE:#x} bytes");
                break;
            }
            memory.write(offset, &fonts::shared_memory_header(font.len() as u32));
            memory.write(offset + 8, font);
            *region = FontRegion {
                offset: offset as u32 + 8,
                size: font.len() as u32,
            };
            offset = end;
        }
        if regions.iter().all(|region| region.size == 0) {
            tracing::warn!(
                ?searched,
                "no shared fonts found: pl:u serves empty font memory"
            );
        }
        let mut builtin = [FontRegion::default(); FONT_COUNT];
        let mut builtin_offset = 0u64;
        for (slot, region) in builtin.iter_mut().zip(&regions) {
            *slot = FontRegion {
                offset: builtin_offset as u32,
                size: region.size,
            };
            builtin_offset = align_up(builtin_offset + u64::from(region.size), PAGE_SIZE);
        }
        Ok(Self {
            memory,
            regions,
            builtin,
            builtin_size: builtin_offset.max(PAGE_SIZE) as u32,
        })
    }

    fn region(&self, font_type: u32) -> FontRegion {
        self.regions
            .get(font_type as usize)
            .copied()
            .unwrap_or_default()
    }

    /// The font of `font_type` in the built-in layout, if it exists.
    fn builtin_region(&self, font_type: u32) -> Option<FontRegion> {
        self.builtin
            .get(font_type as usize)
            .copied()
            .filter(|region| region.size != 0)
    }

    /// Copies every font in the built-in layout through `write(offset, data)`.
    fn copy_builtin(&self, mut write: impl FnMut(u64, &[u8]) -> HleResult<()>) -> HleResult<()> {
        let mut chunk = [0u8; 0x4000];
        for (target, source) in self.builtin.iter().zip(&self.regions) {
            let mut done = 0u64;
            while done < u64::from(source.size) {
                let len = (u64::from(source.size) - done).min(chunk.len() as u64) as usize;
                self.memory
                    .read(u64::from(source.offset) + done, &mut chunk[..len]);
                write(u64::from(target.offset) + done, &chunk[..len])?;
                done += len as u64;
            }
        }
        Ok(())
    }
}

/// Font state every `pl:u` session shares.
struct SharedFonts {
    context: Arc<ServiceContext>,
    loaded: OnceLock<HleResult<LoadedFonts>>,
    builtin: OnceLock<HleResult<Arc<SharedMemory>>>,
}

impl SharedFonts {
    fn get(&self) -> HleResult<&LoadedFonts> {
        self.loaded
            .get_or_init(|| LoadedFonts::load(&self.context))
            .as_ref()
            .map_err(|code| *code)
    }

    /// The Switch 2 built-in font shared memory, made on first use.
    fn builtin_memory(&self) -> HleResult<(&LoadedFonts, Arc<SharedMemory>)> {
        let fonts = self.get()?;
        let memory = self.builtin.get_or_init(|| {
            let memory = new_shared_memory(&self.context, u64::from(fonts.builtin_size))?;
            fonts.copy_builtin(|offset, data| {
                memory.write(offset, data);
                Ok(())
            })?;
            Ok(memory)
        });
        Ok((fonts, memory.clone()?))
    }
}

fn new_shared_memory(context: &ServiceContext, size: u64) -> HleResult<Arc<SharedMemory>> {
    let memory = SharedMemory::new(
        context.kernel.memory_manager().clone(),
        None,
        Pool::System,
        size,
        MemoryPermission::NONE,
        MemoryPermission::READ,
    )?;
    let zero = [0u8; 0x1000];
    for offset in (0..memory.size()).step_by(zero.len()) {
        memory.write(offset, &zero);
    }
    Ok(Arc::new(memory))
}

/// The reply of 2010 and of `INx2FontService` 0: whether the font exists, a flags word, and
/// where it is.
fn push_font_data(ctx: &mut HleRequestContext, region: Option<FontRegion>) {
    let region = region.unwrap_or_default();
    ctx.push(u32::from(region.size != 0));
    ctx.push(0u32);
    ctx.push(region.offset);
    ctx.push(region.size);
}

struct PlU {
    fonts: Arc<SharedFonts>,
}

impl PlU {
    /// 5/6 `GetSharedFontInOrderOfPriority(u64 language) -> (bool loaded, u32 count)` with the
    /// font types, offsets and sizes in three output buffers.
    fn shared_font_in_order_of_priority(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let fonts = self.fonts.get()?;
        let capacity = (0..3)
            .map(|i| ctx.writable_buffer(i).map_or(0, |b| b.size as usize / 4))
            .min()
            .unwrap_or(0)
            .min(MAX_PRIORITY_FONTS);
        let mut types = [0u32; MAX_PRIORITY_FONTS];
        let mut offsets = [0u32; MAX_PRIORITY_FONTS];
        let mut sizes = [0u32; MAX_PRIORITY_FONTS];
        let mut count = 0;
        for (font_type, region) in fonts.regions.iter().enumerate() {
            if count == capacity {
                break;
            }
            if region.size == 0 {
                continue;
            }
            types[count] = font_type as u32;
            offsets[count] = region.offset;
            sizes[count] = region.size;
            count += 1;
        }
        for (i, values) in [&types, &offsets, &sizes].into_iter().enumerate() {
            ctx.write_buffer(i, bytemuck::cast_slice(&values[..count]))?;
        }
        ctx.push(1u8);
        ctx.push(count as u32);
        Ok(())
    }

    /// 2100 `RequestLoadNxFont(u32 flags, u32 size, transfer memory)`: copies the fonts into the
    /// title's transfer memory in the built-in layout and returns the interface that locates them.
    fn request_load_nx_font(&self, ctx: &mut HleRequestContext) -> HleResult<()> {
        let flags: u32 = ctx.pop();
        let size: u32 = ctx.pop();
        let transfer = ctx.copy_handle::<TransferMemory>(0)?;
        let fonts = self.fonts.get()?;
        if u64::from(size) > transfer.size() || size < fonts.builtin_size {
            tracing::error!(
                size,
                transfer = transfer.size(),
                fonts = fonts.builtin_size,
                "font transfer memory too small"
            );
            return Err(lichii_kernel::result::INVALID_SIZE);
        }
        tracing::info!(flags, size, "loading Switch 2 fonts into transfer memory");
        fonts.copy_builtin(|offset, data| transfer.write(offset, data))?;
        ctx.push_interface(self.fonts.context.handler(Nx2FontService {
            regions: fonts.builtin,
        }));
        Ok(())
    }
}

impl Service for PlU {
    fn name(&self) -> &str {
        "pl:u"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        let result = match ctx.command() {
            0 => Ok(()),
            1 => {
                ctx.push(LOADED);
                Ok(())
            }
            2 => {
                let font_type: u32 = ctx.pop();
                self.fonts
                    .get()
                    .map(|fonts| ctx.push(fonts.region(font_type).size))
            }
            3 => {
                let font_type: u32 = ctx.pop();
                self.fonts
                    .get()
                    .map(|fonts| ctx.push(fonts.region(font_type).offset))
            }
            4 => self.fonts.get().map(|fonts| {
                ctx.push_copy_object(KObject::SharedMemory(fonts.memory.clone()));
            }),
            5 | 6 => self.shared_font_in_order_of_priority(ctx),
            2000 => self.fonts.builtin_memory().map(|(fonts, memory)| {
                ctx.push(fonts.builtin_size);
                ctx.push_copy_object(KObject::SharedMemory(memory));
            }),
            2010 => {
                let args: u32 = ctx.pop();
                self.fonts
                    .get()
                    .map(|fonts| push_font_data(ctx, fonts.builtin_region(args & 0xff)))
            }
            2100 => self.request_load_nx_font(ctx),
            _ => return None,
        };
        Some(result.err().unwrap_or(ResultCode::SUCCESS))
    }
}

/// The interface 2100 returns: command 0 locates a font in the transfer memory. The rest of it
/// is unknown, so it is an `nx2:` interface and its other commands auto-stub.
struct Nx2FontService {
    regions: [FontRegion; FONT_COUNT],
}

impl Service for Nx2FontService {
    fn name(&self) -> &str {
        "nx2:pl:font"
    }

    fn handle(&self, ctx: &mut HleRequestContext) -> Option<ResultCode> {
        match ctx.command() {
            0 => {
                let font_type = ctx.pop::<u32>() & 0xff;
                let region = self
                    .regions
                    .get(font_type as usize)
                    .copied()
                    .filter(|region| region.size != 0);
                push_font_data(ctx, region);
                Some(ResultCode::SUCCESS)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests;
