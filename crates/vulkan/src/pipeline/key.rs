//! Cache keys: the shader stages, attachments and fixed state of a graphics pipeline, and the
//! shader and workgroup of a compute pipeline.

use std::hash::{Hash, Hasher};
use std::mem::{offset_of, size_of};

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::{FixedPipelineState, GraphicsShaders};

/// Render target formats and sample count a graphics pipeline renders into with dynamic
/// rendering, as the texture cache resolved them for the draw.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct AttachmentState {
    pub(crate) colors: [i32; 8],
    pub(crate) depth_stencil: i32,
    pub(crate) samples: u32,
}

impl AttachmentState {
    /// `colors` holds up to eight formats, `vk::Format::UNDEFINED` for unbound targets.
    pub fn new(
        colors: &[vk::Format],
        depth_stencil: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Self {
        assert!(colors.len() <= 8, "at most eight colour attachments");
        let mut raw = [vk::Format::UNDEFINED.as_raw(); 8];
        for (slot, format) in raw.iter_mut().zip(colors) {
            *slot = format.as_raw();
        }
        Self {
            colors: raw,
            depth_stencil: depth_stencil.as_raw(),
            samples: samples.as_raw(),
        }
    }

    /// Colour formats up to the last bound one.
    pub fn color_formats(&self) -> impl Iterator<Item = vk::Format> + '_ {
        let count = self
            .colors
            .iter()
            .rposition(|&f| f != vk::Format::UNDEFINED.as_raw())
            .map_or(0, |last| last + 1);
        self.colors[..count]
            .iter()
            .map(|&f| vk::Format::from_raw(f))
    }

    pub fn depth_stencil_format(&self) -> vk::Format {
        vk::Format::from_raw(self.depth_stencil)
    }

    pub fn samples(&self) -> vk::SampleCountFlags {
        vk::SampleCountFlags::from_raw(self.samples.max(1))
    }
}

/// Everything a graphics pipeline is built from besides the SPIR-V itself.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GraphicsPipelineKey {
    /// [`ShaderProgram::hash`](super::ShaderProgram::hash) per [`GRAPHICS_STAGES`](super::GRAPHICS_STAGES) entry, 0 when absent.
    pub(crate) shaders: [u64; 5],
    pub(crate) attachments: AttachmentState,
    pub(crate) fixed: FixedPipelineState,
}

impl GraphicsPipelineKey {
    pub fn new(
        shaders: &GraphicsShaders,
        attachments: AttachmentState,
        fixed: &FixedPipelineState,
    ) -> Self {
        Self {
            shaders: shaders
                .each_ref()
                .map(|program| program.as_ref().map_or(0, |p| p.hash)),
            attachments,
            fixed: *fixed,
        }
    }

    pub fn shader_hashes(&self) -> &[u64; 5] {
        &self.shaders
    }

    pub fn attachments(&self) -> &AttachmentState {
        &self.attachments
    }

    pub fn fixed(&self) -> &FixedPipelineState {
        &self.fixed
    }

    /// The identifying bytes: stages, attachments and the fixed state's used prefix.
    pub(crate) fn as_bytes(&self) -> &[u8] {
        &bytemuck::bytes_of(self)[..offset_of!(Self, fixed) + self.fixed.len()]
    }

    /// The inverse of [`Self::as_bytes`]; `None` when `bytes` cannot have come from it.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() > size_of::<Self>() || bytes.is_empty() {
            return None;
        }
        let mut key = Self::zeroed();
        bytemuck::bytes_of_mut(&mut key)[..bytes.len()].copy_from_slice(bytes);
        (key.as_bytes().len() == bytes.len()).then_some(key)
    }
}

impl PartialEq for GraphicsPipelineKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for GraphicsPipelineKey {}

impl Hash for GraphicsPipelineKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write(self.as_bytes());
    }
}

/// A compute pipeline: its program and the launch shape it was translated for.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Pod, Zeroable)]
pub struct ComputePipelineKey {
    pub shader: u64,
    pub workgroup: [u32; 3],
    /// Bytes of shared memory per workgroup.
    pub shared_memory: u32,
}
