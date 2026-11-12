//! The copy engine's component remap: each destination component takes a source component, a
//! constant or nothing.

use super::regs::{Regs, RemapConstSwizzle};

/// A decoded component remap. Elements are `num_src` components of `component_size` bytes on the
/// source side and `num_dst` components on the destination side.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Remap {
    constants: [u32; 2],
    dst: [RemapConstSwizzle; 4],
    component_size: usize,
    num_src: usize,
    num_dst: usize,
}

impl Remap {
    /// The remap LAUNCH_DMA selects, `None` when remapping is off.
    pub fn decode(regs: &Regs) -> Option<Self> {
        if regs.launch_dma.remap_enable() == 0 {
            return None;
        }
        let remap = &regs.remap_const;
        Some(Self {
            constants: [remap.remap_consta_value, remap.remap_constb_value],
            dst: [remap.dst_x(), remap.dst_y(), remap.dst_z(), remap.dst_w()],
            component_size: remap.component_size_minus_one() as usize + 1,
            num_src: remap.num_src_components_minus_one() as usize + 1,
            num_dst: remap.num_dst_components_minus_one() as usize + 1,
        })
    }

    /// Bytes of one source element.
    pub fn src_bytes(&self) -> u32 {
        (self.num_src * self.component_size) as u32
    }

    /// Bytes of one destination element.
    pub fn dst_bytes(&self) -> u32 {
        (self.num_dst * self.component_size) as u32
    }

    fn active(&self) -> &[RemapConstSwizzle] {
        &self.dst[..self.num_dst]
    }

    /// Whether any destination component comes from the source.
    pub fn reads_source(&self) -> bool {
        self.active()
            .iter()
            .any(|swizzle| swizzle.0 <= RemapConstSwizzle::SRC_W.0)
    }

    /// Whether some destination component keeps the memory's previous value.
    pub fn keeps_destination(&self) -> bool {
        self.active()
            .iter()
            .any(|swizzle| swizzle.0 >= RemapConstSwizzle::NO_WRITE.0)
    }

    /// Whether the remap moves bytes unchanged, so it is a plain copy of wider elements.
    pub fn is_identity(&self) -> bool {
        self.num_src == self.num_dst
            && self
                .active()
                .iter()
                .enumerate()
                .all(|(component, swizzle)| swizzle.0 == component as u32)
    }

    /// Builds `elements` destination elements from the same number of source elements. Components
    /// the remap does not write keep their bytes in `dst`.
    pub fn apply(&self, src: &[u8], dst: &mut [u8], elements: usize) {
        let size = self.component_size;
        let src_bytes = self.src_bytes() as usize;
        let dst_bytes = self.dst_bytes() as usize;
        for (source, target) in src
            .chunks_exact(src_bytes)
            .zip(dst.chunks_exact_mut(dst_bytes))
            .take(elements)
        {
            for (swizzle, out) in self.active().iter().zip(target.chunks_exact_mut(size)) {
                match swizzle.0 {
                    component @ 0..=3 => match source.chunks_exact(size).nth(component as usize) {
                        Some(value) if (component as usize) < self.num_src => {
                            out.copy_from_slice(value);
                        }
                        _ => out.fill(0),
                    },
                    constant @ 4..=5 => {
                        out.copy_from_slice(
                            &self.constants[constant as usize - 4].to_le_bytes()[..size],
                        );
                    }
                    _ => {}
                }
            }
        }
    }
}
