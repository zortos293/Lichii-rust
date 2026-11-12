//! The translated-module contract the renderer consumes: SPIR-V, a stable cache key, and the
//! descriptor layout the pipeline must bind.

use crate::{ShaderInfo, ShaderStage};

/// A translated program, ready to become a Vulkan shader module.
#[derive(Clone, Debug)]
pub struct ShaderModule {
    pub spirv: Vec<u32>,
    /// Cache key of the module: equal for the same guest code, the same GPU state read while
    /// translating, the same runtime info and the same host profile, across runs and builds of
    /// one translator version. Never 0.
    pub hash: u64,
    pub stage: ShaderStage,
    /// Descriptor sets and push constants the module declares.
    pub resources: ResourceLayout,
    /// Where each binding's data comes from (the descriptor lists that `DescriptorBinding::descriptor`
    /// indexes) and what else the module uses.
    pub info: ShaderInfo,
}

/// The descriptor sets and push-constant range a module declares.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResourceLayout {
    pub sets: Vec<DescriptorSetLayout>,
    /// Bytes of push constants, starting at offset 0; 0 when the module has none. Rescaling
    /// (fragment and compute) and render-area (vertex stages) data share offset 0.
    pub push_constant_size: u32,
}

/// One descriptor set: its index and its bindings in binding order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DescriptorSetLayout {
    pub set: u32,
    pub bindings: Vec<DescriptorBinding>,
}

/// One binding of a descriptor set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DescriptorBinding {
    pub binding: u32,
    pub kind: DescriptorKind,
    /// Descriptor count (array size).
    pub count: u32,
    /// Index into the [`ShaderInfo`] list for `kind`: `constant_buffer_descriptors`,
    /// `storage_buffers_descriptors`, `texture_buffer_descriptors`, `image_buffer_descriptors`,
    /// `texture_descriptors` or `image_descriptors`.
    pub descriptor: u32,
}

/// What a binding holds, as a Vulkan descriptor type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescriptorKind {
    /// A guest constant buffer (`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER`).
    UniformBuffer,
    /// A guest global memory region (`VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`). The renderer binds
    /// it from the guest address rounded down to `Profile::min_ssbo_alignment`; the shader adds
    /// the misalignment itself, reading the same guest pointer from its constant buffer.
    StorageBuffer,
    /// `VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER`.
    UniformTexelBuffer,
    /// `VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER`.
    StorageTexelBuffer,
    /// `VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER`.
    CombinedImageSampler,
    /// `VK_DESCRIPTOR_TYPE_STORAGE_IMAGE`.
    StorageImage,
}

/// Bytes of the rescaling push constants: four words of texture scaling bits at offset 0, two
/// words of image scaling bits at offset 16, the resolution down factor (f32) at offset 24.
pub const RESCALING_PUSH_CONSTANT_SIZE: u32 = 32;
/// Bytes of the render-area push constant (a vec4).
pub const RENDER_AREA_PUSH_CONSTANT_SIZE: u32 = 16;

impl ResourceLayout {
    /// The layout of a module translated with `info`, in descriptor set 0 from `first_binding`
    /// (stages of one pipeline share set 0, each starting after the previous stage's bindings).
    /// Binding order is uniform buffers, storage buffers, texel buffers, image texel buffers,
    /// textures, images; a buffer descriptor of count n takes n binding numbers.
    pub fn for_module(info: &ShaderInfo, first_binding: u32) -> ResourceLayout {
        let mut binding = first_binding;
        let mut bindings = Vec::new();
        let mut push = |kind, count: u32, descriptor: usize, step: u32| {
            bindings.push(DescriptorBinding {
                binding,
                kind,
                count,
                descriptor: descriptor as u32,
            });
            binding += step;
        };
        for (i, d) in info.constant_buffer_descriptors.iter().enumerate() {
            push(DescriptorKind::UniformBuffer, d.count, i, d.count);
        }
        for (i, d) in info.storage_buffers_descriptors.iter().enumerate() {
            push(DescriptorKind::StorageBuffer, d.count, i, d.count);
        }
        for (i, d) in info.texture_buffer_descriptors.iter().enumerate() {
            push(DescriptorKind::UniformTexelBuffer, d.count, i, 1);
        }
        for (i, d) in info.image_buffer_descriptors.iter().enumerate() {
            push(DescriptorKind::StorageTexelBuffer, d.count, i, 1);
        }
        for (i, d) in info.texture_descriptors.iter().enumerate() {
            push(DescriptorKind::CombinedImageSampler, d.count, i, 1);
        }
        for (i, d) in info.image_descriptors.iter().enumerate() {
            push(DescriptorKind::StorageImage, d.count, i, 1);
        }
        let push_constant_size = if info.uses_rescaling_uniform {
            RESCALING_PUSH_CONSTANT_SIZE
        } else if info.uses_render_area {
            RENDER_AREA_PUSH_CONSTANT_SIZE
        } else {
            0
        };
        let sets = if bindings.is_empty() {
            Vec::new()
        } else {
            vec![DescriptorSetLayout { set: 0, bindings }]
        };
        ResourceLayout {
            sets,
            push_constant_size,
        }
    }

    /// The binding number after this layout's last binding, where the next stage starts.
    pub fn next_binding(&self, first_binding: u32) -> u32 {
        self.sets
            .iter()
            .flat_map(|s| s.bindings.last())
            .map(|b| {
                let step = match b.kind {
                    DescriptorKind::UniformBuffer | DescriptorKind::StorageBuffer => b.count,
                    _ => 1,
                };
                b.binding + step
            })
            .max()
            .unwrap_or(first_binding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::info::{ConstantBufferDescriptor, StorageBufferDescriptor, TextureDescriptor};

    #[test]
    fn bindings_follow_descriptor_order_and_counts() {
        let info = ShaderInfo {
            constant_buffer_descriptors: vec![
                ConstantBufferDescriptor { index: 0, count: 1 },
                ConstantBufferDescriptor { index: 3, count: 2 },
            ],
            storage_buffers_descriptors: vec![StorageBufferDescriptor {
                cbuf_index: 0,
                cbuf_offset: 0x110,
                count: 1,
                is_written: true,
                indirect_offset: StorageBufferDescriptor::DIRECT,
                write_extent: Default::default(),
            }],
            texture_descriptors: vec![TextureDescriptor {
                ty: crate::ir::TextureType::Color2D,
                is_depth: false,
                is_multisample: false,
                is_integer: false,
                handle: Default::default(),
                count: 4,
                size_shift: 3,
                is_descriptor_array: true,
                handle_is_pointer_word: false,
            }],
            uses_rescaling_uniform: true,
            ..ShaderInfo::default()
        };
        let layout = ResourceLayout::for_module(&info, 5);
        let kinds: Vec<(u32, DescriptorKind, u32, u32)> = layout.sets[0]
            .bindings
            .iter()
            .map(|b| (b.binding, b.kind, b.count, b.descriptor))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (5, DescriptorKind::UniformBuffer, 1, 0),
                (6, DescriptorKind::UniformBuffer, 2, 1),
                (8, DescriptorKind::StorageBuffer, 1, 0),
                (9, DescriptorKind::CombinedImageSampler, 4, 0),
            ]
        );
        assert_eq!(layout.push_constant_size, RESCALING_PUSH_CONSTANT_SIZE);
        assert_eq!(layout.next_binding(5), 10);
        assert_eq!(
            ResourceLayout::for_module(&ShaderInfo::default(), 3).next_binding(3),
            3
        );
    }
}
