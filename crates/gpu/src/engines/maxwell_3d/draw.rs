//! Draw assembly: begin/end pairs, instance loops and inline indices turned into
//! [`DrawParams`].
//!
//! NVN draws `n` instances as one begin/end pair with instance id FIRST followed by `n - 1` pairs
//! with SUBSEQUENT, writing only draw methods in between. The engine keeps the last direct draw
//! pending and lets each following instance of the same draw join it, so a loop reaches the
//! renderer as one instanced draw. The reference draws the first instance on its own and then all
//! of them again (DrawManager::DrawDeferred); here every instance is drawn once.

use super::regs::{
    DrawBegin, DrawInstanceId, PrimitiveTopology, PrimitiveTopologyControl,
    PrimitiveTopologyOverride, Regs,
};
use crate::backend::DrawParams;
use crate::reg_index;

pub(super) const DRAW_END: u32 = reg_index!(Regs, draw.end);
pub(super) const DRAW_BEGIN: u32 = reg_index!(Regs, draw.begin);
pub(super) const VERTEX_BUFFER_FIRST: u32 = reg_index!(Regs, vertex_buffer.first);
pub(super) const VERTEX_BUFFER_COUNT: u32 = reg_index!(Regs, vertex_buffer.count);
pub(super) const INDEX_BUFFER_FIRST: u32 = reg_index!(Regs, index_buffer.first);
pub(super) const INDEX_BUFFER_COUNT: u32 = reg_index!(Regs, index_buffer.count);
pub(super) const INDEX_BUFFER32_FIRST: u32 = reg_index!(Regs, index_buffer32_first);
pub(super) const INDEX_BUFFER16_FIRST: u32 = reg_index!(Regs, index_buffer16_first);
pub(super) const INDEX_BUFFER8_FIRST: u32 = reg_index!(Regs, index_buffer8_first);
pub(super) const INDEX_BUFFER32_SUBSEQUENT: u32 = reg_index!(Regs, index_buffer32_subsequent);
pub(super) const INDEX_BUFFER16_SUBSEQUENT: u32 = reg_index!(Regs, index_buffer16_subsequent);
pub(super) const INDEX_BUFFER8_SUBSEQUENT: u32 = reg_index!(Regs, index_buffer8_subsequent);
pub(super) const DRAW_INLINE_INDEX: u32 = reg_index!(Regs, draw_inline_index);
pub(super) const INLINE_INDEX_2X16_COUNT: u32 = reg_index!(Regs, inline_index_2x16.bits);
pub(super) const INLINE_INDEX_2X16: u32 = reg_index!(Regs, inline_index_2x16.bits1);
pub(super) const INLINE_INDEX_4X8_COUNT: u32 = reg_index!(Regs, inline_index_4x8.bits);
pub(super) const INLINE_INDEX_4X8: u32 = reg_index!(Regs, inline_index_4x8.bits1);
pub(super) const VERTEX_ARRAY_INSTANCE_FIRST: u32 = reg_index!(Regs, vertex_array_instance_first);
pub(super) const VERTEX_ARRAY_INSTANCE_SUBSEQUENT: u32 =
    reg_index!(Regs, vertex_array_instance_subsequent);

/// Methods an instance loop writes between its draws; any other write that changes state ends
/// the pending draw.
pub(super) fn is_loop_method(method: u32) -> bool {
    matches!(
        method,
        DRAW_END
            | DRAW_BEGIN
            | VERTEX_BUFFER_FIRST
            | VERTEX_BUFFER_COUNT
            | INDEX_BUFFER_FIRST
            | INDEX_BUFFER_COUNT
            | INDEX_BUFFER32_FIRST
            | INDEX_BUFFER16_FIRST
            | INDEX_BUFFER8_FIRST
            | INDEX_BUFFER32_SUBSEQUENT
            | INDEX_BUFFER16_SUBSEQUENT
            | INDEX_BUFFER8_SUBSEQUENT
            | DRAW_INLINE_INDEX
            | INLINE_INDEX_2X16_COUNT
            | INLINE_INDEX_2X16
            | INLINE_INDEX_4X8_COUNT
            | INLINE_INDEX_4X8
            | VERTEX_ARRAY_INSTANCE_FIRST
            | VERTEX_ARRAY_INSTANCE_SUBSEQUENT
    )
}

/// The topology a draw uses: the one its begin method names, unless the separate topology
/// state overrides it.
pub(super) fn topology(regs: &Regs, begin: PrimitiveTopology) -> PrimitiveTopology {
    if regs.primitive_topology_control != PrimitiveTopologyControl::USE_SEPARATE_STATE {
        return begin;
    }
    match regs.topology_override {
        PrimitiveTopologyOverride::NONE => begin,
        PrimitiveTopologyOverride::POINTS => PrimitiveTopology::POINTS,
        PrimitiveTopologyOverride::LINES => PrimitiveTopology::LINES,
        PrimitiveTopologyOverride::LINE_STRIP => PrimitiveTopology::LINE_STRIP,
        other => PrimitiveTopology(other.0),
    }
}

/// Inline indices the engine allocates room for up front, so that uploads do not allocate on the
/// method path. A larger draw grows the storage once and keeps it.
pub(super) const INLINE_INDEX_CAPACITY: usize = 1 << 18;

/// Draw state between begin and end, and the draw waiting for more instances.
pub(super) struct DrawState {
    /// Topology of the last draw.begin.
    pub topology: PrimitiveTopology,
    /// Instance the next draw draws, relative to the global base instance.
    pub instance: u32,
    /// index_buffer.count was written since the last draw.end.
    pub indexed: bool,
    /// Indices of the inline-index methods since the last draw.end, widened to 32 bits. Starts
    /// with [`INLINE_INDEX_CAPACITY`] and never shrinks.
    pub inline_indices: Vec<u32>,
    /// The last direct draw, held until a draw that does not continue it or a state change.
    pub pending: Option<DrawParams<'static>>,
}

impl DrawState {
    pub fn new() -> Self {
        Self {
            topology: PrimitiveTopology::POINTS,
            instance: 0,
            indexed: false,
            inline_indices: Vec::with_capacity(INLINE_INDEX_CAPACITY),
            pending: None,
        }
    }

    pub fn begin(&mut self, begin: DrawBegin) {
        match begin.instance_id() {
            DrawInstanceId::FIRST => self.instance = 0,
            DrawInstanceId::SUBSEQUENT => self.instance = self.instance.wrapping_add(1),
            _ => {}
        }
        self.topology = begin.topology();
    }

    /// Instance id handling of the small index-buffer and vertex-array draw methods: FIRST
    /// variants start at instance 0, SUBSEQUENT ones draw the next instance.
    pub fn next_instance(&mut self, subsequent: bool) -> u32 {
        self.instance = if subsequent {
            self.instance.wrapping_add(1)
        } else {
            0
        };
        self.instance
    }

    /// Appends the indices `words` carry through inline-index method `method`.
    pub fn push_inline(&mut self, method: u32, words: &[u32]) {
        let indices = &mut self.inline_indices;
        match method {
            INLINE_INDEX_2X16 => {
                indices.extend(words.iter().flat_map(|&w| [w & 0xffff, w >> 16]));
            }
            INLINE_INDEX_4X8 => {
                indices.extend(
                    words
                        .iter()
                        .flat_map(|&w| [w & 0xff, (w >> 8) & 0xff, (w >> 16) & 0xff, w >> 24]),
                );
            }
            _ => indices.extend_from_slice(words),
        }
    }

    /// Makes `draw` (one instance) pending, or adds it to the pending draw when it is that
    /// draw's next instance. Returns the draw it replaced, which must be drawn before it.
    pub fn queue(&mut self, draw: DrawParams<'static>) -> Option<DrawParams<'static>> {
        if let Some(pending) = &mut self.pending
            && continues(pending, &draw)
        {
            pending.instance_count += 1;
            return None;
        }
        self.pending.replace(draw)
    }
}

fn continues(pending: &DrawParams<'_>, next: &DrawParams<'_>) -> bool {
    next.base_instance == pending.base_instance.wrapping_add(pending.instance_count)
        && *pending
            == DrawParams {
                base_instance: pending.base_instance,
                instance_count: pending.instance_count,
                ..*next
            }
}
