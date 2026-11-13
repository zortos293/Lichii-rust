//! Pipeline state that changes how a program translates. Part of the pipeline cache key.

use crate::info::VaryingState;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AttributeType {
    #[default]
    Float,
    SignedInt,
    UnsignedInt,
    SignedScaled,
    UnsignedScaled,
    Disabled,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum InputTopology {
    #[default]
    Points,
    Lines,
    LinesAdjacency,
    Triangles,
    TrianglesAdjacency,
}

impl InputTopology {
    pub const fn vertices(self) -> u32 {
        match self {
            InputTopology::Points => 1,
            InputTopology::Lines => 2,
            InputTopology::LinesAdjacency => 4,
            InputTopology::Triangles => 3,
            InputTopology::TrianglesAdjacency => 6,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CompareFunction {
    Never,
    Less,
    Equal,
    LessThanEqual,
    Greater,
    NotEqual,
    GreaterThanEqual,
    #[default]
    Always,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TessPrimitive {
    Isolines,
    #[default]
    Triangles,
    Quads,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum TessSpacing {
    #[default]
    Equal,
    FractionalOdd,
    FractionalEven,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct TransformFeedbackVarying {
    pub buffer: u32,
    pub stream: u32,
    pub stride: u32,
    pub offset: u32,
    pub components: u32,
}

/// State of the pipeline a program is translated for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RuntimeInfo {
    /// Vertex attribute formats, for vertex programs.
    pub generic_input_types: [AttributeType; 32],
    /// Outputs written by the previous stage, so unwritten inputs read defaults.
    pub previous_stage_stores: VaryingState,
    pub previous_stage_legacy_stores_mapping: Vec<(crate::ir::Attribute, crate::ir::Attribute)>,
    pub per_primitive_generics: u32,
    pub noperspective_outputs: u32,
    /// Remap depth from [-1, 1] to [0, 1] in the last pre-rasterisation stage.
    pub convert_depth_mode: bool,
    pub force_early_z: bool,
    pub tess_primitive: TessPrimitive,
    pub tess_spacing: TessSpacing,
    pub tess_clockwise: bool,
    pub input_topology: InputTopology,
    pub fixed_state_point_size: Option<f32>,
    pub alpha_test_func: Option<CompareFunction>,
    pub alpha_test_reference: f32,
    pub y_negate: bool,
    pub xfb_varyings: Vec<TransformFeedbackVarying>,
    pub color_output_types: [AttributeType; 8],
    pub dual_source_blend: bool,
    /// Task and mesh programs on a host without `VK_EXT_mesh_shader` (the C++
    /// `mesh_as_compute`): emit them as compute shaders that write storage buffers, for
    /// [`crate::backend::spirv::emit_mesh_emulation_vertex`] to draw. `None` emits native
    /// `SPV_EXT_mesh_shader` stages.
    pub mesh_emulation: Option<MeshEmulation>,
}

/// Storage buffers (descriptor set 0) of the mesh emulation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MeshEmulation {
    /// The buffer the mesh program writes one block per workgroup to, laid out as
    /// [`crate::backend::spirv::mesh_emulation_block_words`] describes.
    pub output_binding: u32,
    /// The buffers between a task program and its mesh program, `None` without a task program.
    pub task: Option<MeshEmulationTask>,
}

/// Mesh emulation with a task program. The task's compute shader writes its payload (mesh
/// workgroup count at word 1) to the payload buffer, `payload_stride` words per task workgroup.
/// Each mesh workgroup finds its task by a binary search of the offsets buffer (first mesh
/// workgroup of each task, from [`crate::backend::spirv::emit_mesh_emulation_distribution`])
/// and copies that task's payload; the arguments buffer holds the draw's output base and task
/// count (`MESH_EMULATION_ARGS_*`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MeshEmulationTask {
    pub payload_binding: u32,
    pub offsets_binding: u32,
    pub args_binding: u32,
    /// Payload buffer words per task workgroup.
    pub payload_stride: u32,
}
