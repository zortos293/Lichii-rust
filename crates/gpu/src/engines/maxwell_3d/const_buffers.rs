//! Constant buffer uploads and binds, and the NVN2 root tables.
//!
//! Uploads write guest memory at once (the address is the buffer's GPU address plus its running
//! offset) and tell the renderer which range changed; binds update the per-stage table the
//! renderer borrows at each draw.

use super::regs::{BindGroupRawConfig, ConstantBuffer, MAX_CONST_BUFFERS};
use crate::backend::{ConstBufferBinding, GraphicsConstBuffers, RootTables};
use crate::engines::EngineContext;

/// Writes `words` at the bound buffer's offset and advances the offset by 4 per word. Words past
/// the buffer's size are dropped, but still advance the offset, as on hardware.
pub(super) fn upload(cb: &mut ConstantBuffer, ctx: &mut EngineContext<'_>, words: &[u32]) {
    let offset = cb.offset;
    let room = cb.size.saturating_sub(offset) as usize / 4;
    let written = &words[..words.len().min(room)];
    if !written.is_empty() {
        let address = cb.address() + u64::from(offset);
        let bytes: &[u8] = bytemuck::cast_slice(written);
        if let Err(fault) = ctx.memory.write_block(address, bytes) {
            tracing::warn!(?fault, "constant buffer upload to unmapped GPU memory");
        }
        crate::diag_inv::set(crate::diag_inv::CB);
        ctx.rasterizer
            .inline_written(ctx.channel, ctx.memory, address, bytes.len() as u64);
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
