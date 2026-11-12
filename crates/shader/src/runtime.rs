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
}
