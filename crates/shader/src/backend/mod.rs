//! Backends that turn a translated program into host shader code.

pub mod spirv;

/// Next free descriptor binding of each kind, carried from one stage to the next so the stages
/// of a pipeline share one descriptor set without overlapping. With
/// [`crate::Profile::unified_descriptor_binding`] every kind takes its binding from `unified`.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct Bindings {
    pub unified: u32,
    pub uniform_buffer: u32,
    pub storage_buffer: u32,
    pub texture: u32,
    pub image: u32,
    /// Index of the next texture in the rescaling push constant bit set.
    pub texture_scaling_index: u32,
    /// Index of the next image in the rescaling push constant bit set.
    pub image_scaling_index: u32,
}
