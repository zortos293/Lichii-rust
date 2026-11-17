//! Constant buffer uploads and binds, and the NVN2 root tables.
//!
//! Uploads write guest memory at once (the address is the buffer's GPU address plus its running
//! offset) and collect the changed range; the renderer hears of it once per run of uploads, before
//! the next method that is not a constant buffer write (as the reference's FlushCaching does).
//! Binds update the per-stage table the renderer borrows at each draw.

use super::regs::{BindGroupRawConfig, ConstantBuffer, MAX_CONST_BUFFERS};
use crate::backend::{ConstBufferBinding, GraphicsConstBuffers, RootTables};
use crate::engines::EngineContext;
use lichii_common::GpuVAddr;

/// Constant buffer bytes written since the renderer last heard of them: one range, grown while
/// uploads touch it.
#[derive(Default)]
pub(super) struct Written(Option<(GpuVAddr, GpuVAddr)>);

impl Written {
    #[inline]
    pub fn is_pending(&self) -> bool {
        self.0.is_some()
    }

    /// Tells the renderer about the collected range.
    pub fn flush(&mut self, ctx: &mut EngineContext<'_>) {
        if let Some((start, end)) = self.0.take() {
            ctx.rasterizer
                .inline_written(ctx.channel, ctx.memory, start, end - start);
        }
    }

    /// Writes `bytes` at `start` and adds them to the range. A range they don't touch is announced
    /// before the write: announcing lets the renderer write GPU-modified images back over memory
    /// outside the announced bytes.
    fn write(&mut self, ctx: &mut EngineContext<'_>, start: GpuVAddr, bytes: &[u8]) {
        let end = start + bytes.len() as u64;
        if self.0.is_some_and(|(from, to)| start > to || end < from) {
            self.flush(ctx);
        }
        if let Err(fault) = ctx.memory.write_block(start, bytes) {
            tracing::warn!(?fault, "constant buffer upload to unmapped GPU memory");
        }
        self.0 = Some(
            self.0
                .map_or((start, end), |(from, to)| (from.min(start), to.max(end))),
        );
    }
}

/// Writes `words` at the bound buffer's offset and advances the offset by 4 per word. Words past
/// the buffer's size are dropped, but still advance the offset, as on hardware.
pub(super) fn upload(
    cb: &mut ConstantBuffer,
    written_ranges: &mut Written,
    ctx: &mut EngineContext<'_>,
    words: &[u32],
) {
    let offset = cb.offset;
    let room = cb.size.saturating_sub(offset) as usize / 4;
    let written = &words[..words.len().min(room)];
    if !written.is_empty() {
        let address = cb.address() + u64::from(offset);
        written_ranges.write(ctx, address, bytemuck::cast_slice(written));
    }
    if written.len() != words.len() {
        tracing::debug!(
            offset,
            size = cb.size,
            words = words.len(),
            "constant buffer words past the end dropped"
        );
    }
    cb.offset = offset.wrapping_add(words.len() as u32 * 4);
}

/// Binds the buffer the selector registers describe to `stage`'s slot from `config`. Returns
/// whether the binding changed.
pub(super) fn bind(
    bindings: &mut GraphicsConstBuffers,
    selector: &ConstantBuffer,
    stage: usize,
    config: BindGroupRawConfig,
) -> bool {
    let slot = config.shader_slot() as usize;
    if slot >= MAX_CONST_BUFFERS {
        return false;
    }
    let binding = ConstBufferBinding {
        address: selector.address(),
        size: selector.size,
        enabled: config.valid() != 0,
    };
    let entry = &mut bindings[stage][slot];
    let changed = *entry != binding;
    *entry = binding;
    changed
}

/// NVN2 root tables: ROOT_TABLE_SELECT picks a table and a byte offset, each ROOT_TABLE_LOAD
/// word lands at that offset and advances it by 4. Shaders read the tables as constant banks
/// 24-31.
pub(super) struct RootTableState {
    pub tables: RootTables,
    selected: usize,
    offset: u32,
}

impl RootTableState {
    pub fn new() -> Self {
        Self {
            tables: [[0; 64]; 8],
            selected: 0,
            offset: 0,
        }
    }

    pub fn select(&mut self, value: u32) {
        self.selected = (value & 7) as usize;
        self.offset = (value >> 8) & 0xff;
    }

    /// Stores `words` from the current offset on. Unaligned offsets and offsets past the table
    /// store nothing. Returns whether a stored word changed.
    pub fn load(&mut self, words: &[u32]) -> bool {
        let mut changed = false;
        for &word in words {
            if self.offset.is_multiple_of(4)
                && let Some(slot) = self.tables[self.selected].get_mut(self.offset as usize / 4)
            {
                changed |= *slot != word;
                *slot = word;
            }
            self.offset = self.offset.wrapping_add(4);
        }
        changed
    }
}
