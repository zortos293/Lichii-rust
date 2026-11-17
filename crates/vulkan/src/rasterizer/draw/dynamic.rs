//! Skips dynamic state commands whose arguments the open rendering already set.

use ash::vk;

/// The dynamic state commands a draw may record, one shadow slot each.
#[derive(Clone, Copy)]
pub(super) enum Slot {
    LineWidth,
    DepthBias,
    BlendConstants,
    DepthBounds,
    StencilCompareMask,
    StencilWriteMask,
    StencilReference,
    Viewport,
    Scissor,
    CullMode,
    FrontFace,
    PrimitiveTopology,
    DepthTestEnable,
    DepthWriteEnable,
    DepthCompareOp,
    DepthBoundsTestEnable,
    StencilTestEnable,
    StencilOp,
    RasterizerDiscardEnable,
    DepthBiasEnable,
    PrimitiveRestartEnable,
    VertexBuffers,
    VertexInput,
    ColorBlendEnable,
    ColorBlendEquation,
    ColorWriteMask,
    DepthClampEnable,
    LogicOpEnable,
    AlphaToCoverageEnable,
    AlphaToOneEnable,
    PolygonMode,
    DepthClipNegativeOneToOne,
    ProvokingVertexMode,
}

const SLOTS: usize = Slot::ProvokingVertexMode as usize + 1;

/// The arguments of the dynamic state commands recorded in the open rendering, so a draw
/// records only the commands whose arguments changed. Command buffer state outlives a
/// rendering, but other work may bind pipelines between renderings, so a new rendering starts
/// empty; so does a switch between mesh and vertex pipelines, which declare different dynamic
/// states.
pub(super) struct DynamicShadow {
    slots: [Vec<u32>; SLOTS],
    scratch: Vec<u32>,
    mesh: bool,
}

impl Default for DynamicShadow {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| Vec::new()),
            scratch: Vec::new(),
            mesh: false,
        }
    }
}

impl DynamicShadow {
    /// Forgets every slot when the draw began a rendering or switches between mesh and vertex
    /// pipelines.
    pub(super) fn begin(&mut self, began: bool, mesh: bool) {
        if began || mesh != self.mesh {
            self.slots.iter_mut().for_each(Vec::clear);
            self.mesh = mesh;
        }
    }

    /// Runs `record` unless `slot` already holds `args`.
    pub(super) fn set(&mut self, slot: Slot, args: impl Words, record: impl FnOnce()) {
        self.scratch.clear();
        args.words(&mut self.scratch);
        let held = &mut self.slots[slot as usize];
        if !held.is_empty() && *held == self.scratch {
            return;
        }
        held.clear();
        held.extend_from_slice(&self.scratch);
        record();
    }
}

/// Command arguments as the words a shadow slot compares.
pub(super) trait Words {
    fn words(&self, out: &mut Vec<u32>);
}

impl Words for u32 {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(*self);
    }
}

impl Words for u64 {
    fn words(&self, out: &mut Vec<u32>) {
        out.extend([*self as u32, (*self >> 32) as u32]);
    }
}

impl Words for f32 {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(self.to_bits());
    }
}

impl Words for bool {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(u32::from(*self));
    }
}

macro_rules! raw_words {
    ($($ty:ty),*) => {$(
        impl Words for $ty {
            fn words(&self, out: &mut Vec<u32>) {
                out.push(self.as_raw() as u32);
            }
        }
    )*};
}

raw_words!(
    vk::PrimitiveTopology,
    vk::FrontFace,
    vk::CompareOp,
    vk::StencilOp,
    vk::PolygonMode,
    vk::ProvokingVertexModeEXT,
    vk::DepthBiasRepresentationEXT,
    vk::BlendFactor,
    vk::BlendOp,
    vk::VertexInputRate,
    vk::Format,
    vk::CullModeFlags,
    vk::StencilFaceFlags,
    vk::ColorComponentFlags
);

impl Words for vk::Buffer {
    fn words(&self, out: &mut Vec<u32>) {
        ash::vk::Handle::as_raw(*self).words(out);
    }
}

impl Words for vk::Viewport {
    fn words(&self, out: &mut Vec<u32>) {
        (
            self.x,
            self.y,
            self.width,
            self.height,
            self.min_depth,
            self.max_depth,
        )
            .words(out);
    }
}

impl Words for vk::Rect2D {
    fn words(&self, out: &mut Vec<u32>) {
        let (o, e) = (self.offset, self.extent);
        (o.x as u32, o.y as u32, e.width, e.height).words(out);
    }
}

impl Words for vk::ColorBlendEquationEXT {
    fn words(&self, out: &mut Vec<u32>) {
        (
            self.src_color_blend_factor,
            self.dst_color_blend_factor,
            self.color_blend_op,
            self.src_alpha_blend_factor,
            self.dst_alpha_blend_factor,
            self.alpha_blend_op,
        )
            .words(out);
    }
}

impl Words for vk::VertexInputBindingDescription2EXT<'_> {
    fn words(&self, out: &mut Vec<u32>) {
        (self.binding, self.stride, self.input_rate, self.divisor).words(out);
    }
}

impl Words for vk::VertexInputAttributeDescription2EXT<'_> {
    fn words(&self, out: &mut Vec<u32>) {
        (self.location, self.binding, self.format, self.offset).words(out);
    }
}

impl<T: Words> Words for &[T] {
    fn words(&self, out: &mut Vec<u32>) {
        out.push(self.len() as u32);
        self.iter().for_each(|t| t.words(out));
    }
}

impl<T: Words, const N: usize> Words for [T; N] {
    fn words(&self, out: &mut Vec<u32>) {
        self.iter().for_each(|t| t.words(out));
    }
}

macro_rules! tuple_words {
    ($($name:ident),+) => {
        impl<$($name: Words),+> Words for ($($name,)+) {
            #[allow(non_snake_case)]
            fn words(&self, out: &mut Vec<u32>) {
                let ($($name,)+) = self;
                $($name.words(out);)+
            }
        }
    };
}

tuple_words!(A, B);
tuple_words!(A, B, C);
tuple_words!(A, B, C, D);
tuple_words!(A, B, C, D, E);
tuple_words!(A, B, C, D, E, F);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_record_only_changed_arguments_until_a_rendering_begins() {
        let mut shadow = DynamicShadow::default();
        let mut recorded = 0;
        let mut set = |shadow: &mut DynamicShadow, slot, width: f32| {
            shadow.set(slot, width, || recorded += 1);
        };
        set(&mut shadow, Slot::LineWidth, 1.0);
        set(&mut shadow, Slot::LineWidth, 1.0);
        set(&mut shadow, Slot::DepthBounds, 1.0);
        set(&mut shadow, Slot::LineWidth, 2.0);
        shadow.begin(false, false);
        set(&mut shadow, Slot::LineWidth, 2.0);
        shadow.begin(true, false);
        set(&mut shadow, Slot::LineWidth, 2.0);
        shadow.begin(false, true);
        set(&mut shadow, Slot::LineWidth, 2.0);
        drop(set);
        assert_eq!(recorded, 5);
    }
}
